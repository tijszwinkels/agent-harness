from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from agent_harness.models import (
    LiveStateRequest,
    CreateRunRequest,
    CreateSessionRequest,
    Event,
    Message,
    RUN_TERMINAL_STATUSES,
    Run,
    RunStatus,
    Session,
    StopReason,
    Usage,
    merge_observed_session,
    observed_status,
    utc_now,
)
from agent_harness.live_state import (
    LiveClaim,
    LiveStateRegistry,
    ensure_accepts_live_state,
    status_for_claim,
)


class SessionNotFoundError(KeyError):
    pass


class RunNotFoundError(KeyError):
    pass


class MaterializationDeferred(SessionNotFoundError):
    """Phase 3: raised by ``DurableEventBus.publish`` when an event was
    successfully inserted (event row + bus history + subscribers
    notified) but its side-effect materialization needs the referenced
    session to exist first — typically a rollout-derived message event
    that arrived before ``POST /v1/sessions``.

    Carries ``event`` so the caller can buffer the published event
    (with its assigned sequence) for later replay via
    ``repository.materialize_event(event, store_event=False)``. Without
    the published event in hand, the caller would only see the
    original pre-publish event with no sequence and have no clean way
    to re-materialize against the existing event row.

    Inherits from ``SessionNotFoundError`` so call-sites that already
    catch the parent type keep working unchanged.
    """

    def __init__(self, event: Event, session_id: str) -> None:
        super().__init__(session_id)
        self.event = event
        self.session_id = session_id


class InMemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        # Live claims from producers of external sessions (in memory only).
        self.live_state = LiveStateRegistry()
        # Injectable for tests that drive time with a fake clock.
        self.clock = utc_now
        self._sessions: dict[str, Session] = {}
        self._runs: dict[str, Run] = {}
        self._messages: dict[str, list[Message]] = {}

    def create_session(self, request: CreateSessionRequest) -> Session:
        session = Session(
            backend=request.backend,
            model=request.model,
            effort=request.effort,
            project=request.project,
            title=request.title,
            bypass_permissions=request.bypass_permissions,
        )
        with self._lock:
            self._sessions[session.id] = session
            self._messages[session.id] = []
        return session.model_copy(deep=True)

    def create_forked_session(self, parent: Session, *, title: str | None) -> Session:
        child = Session.forked_child(parent, title=title)
        with self._lock:
            self._sessions[child.id] = child
            self._messages[child.id] = []
        return child.model_copy(deep=True)

    def list_sessions(self) -> list[Session]:
        with self._lock:
            return [session.model_copy(deep=True) for session in self._sessions.values()]

    def find_codex_harness_sessions(self, resume_id: str) -> list[Session]:
        """Find durable rollout owners, including idle and archived sessions."""
        with self._lock:
            return [
                session.model_copy(deep=True)
                for session in self._sessions.values()
                if session.backend == "codex"
                and session.origin == "harness"
                and session.codex_resume_id == resume_id
            ]

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session.model_copy(deep=True)

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def archive_session(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            archived = session.model_copy(update={"status": "archived", "updated_at": utc_now()})
            self._sessions[session_id] = archived
            return archived.model_copy(deep=True)

    def patch_session(self, session_id: str, fields: dict[str, object]) -> Session:
        # Apply the given user-mutable fields to the session, bumping
        # ``updated_at``. Unknown or empty payloads return the current
        # session unchanged (no-op). Caller (api.py) is responsible for
        # whitelisting fields against ``PatchSessionRequest``.
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            if not fields:
                return session.model_copy(deep=True)
            updated = session.model_copy(update={**fields, "updated_at": utc_now()})
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    def create_run(self, session_id: str, request: CreateRunRequest) -> Run:
        # Runs are born ``queued`` and stay that way until the orchestrator
        # actually spawns the subprocess (via ``start_run``). This lets the
        # RunManager serialize concurrent ``POST /runs`` calls on the same
        # session without lying about lifecycle state in the repo.
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            input_message = Message.user(request.message)
            run = Run(
                session_id=session.id,
                status="queued",
                started_at=None,
                input_message_id=input_message.id,
                origin="harness",
            )
            updated_session = session.model_copy(
                update={
                    "status": "running",
                    "updated_at": utc_now(),
                    "stats": session.stats.model_copy(update={"messages": session.stats.messages + 1}),
                }
            )
            self._sessions[session_id] = updated_session
            self._runs[run.id] = run
            self._messages.setdefault(session_id, []).append(input_message)
            return run.model_copy(deep=True)

    def start_run(self, session_id: str, run_id: str) -> Run:
        # Flip a queued run to ``running``, stamping ``started_at``. Idempotent:
        # calling on an already-running run just refreshes the timestamp.
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                raise RunNotFoundError(run_id)

            started = run.model_copy(update={"status": "running", "started_at": utc_now()})
            self._runs[run_id] = started
            return started.model_copy(deep=True)

    def drop_queued_runs(self, session_id: str) -> list[Run]:
        # Mark every still-queued run for this session as ``interrupted`` and
        # return the resulting Run records. Used by ``interrupt_run`` flow so
        # callers can observe (and surface to clients) which queued follow-ups
        # were cancelled as a side-effect of cancelling the active run.
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            dropped: list[Run] = []
            now = utc_now()
            for run_id, run in list(self._runs.items()):
                if run.session_id != session_id or run.status != "queued":
                    continue
                interrupted = run.model_copy(
                    update={
                        "status": "interrupted",
                        "completed_at": now,
                        "stop_reason": "interrupted",
                    }
                )
                self._runs[run_id] = interrupted
                dropped.append(interrupted.model_copy(deep=True))
            return dropped

    def has_active_run(self, session_id: str) -> bool:
        """True while the session has a queued or running run."""
        with self._lock:
            return self._has_active_run_locked(session_id)

    def observed_status_for(self, session_id: str, status: str) -> str:
        """The status an observation of ``session_id`` would be stored with."""
        with self._lock:
            return observed_status(
                status, self._sessions.get(session_id), has_active_run=self._status_owned_locked(session_id)
            )

    def status_owned(self, session_id: str) -> bool:
        """True while something more authoritative than transcript
        observation owns the status: a queued/running harness run, or an
        unexpired live ``busy`` claim."""
        with self._lock:
            return self._status_owned_locked(session_id)

    def observation_superseded(self, session_id: str, offset: int) -> bool:
        """A transcript line older than the session's last settled idle."""
        with self._lock:
            return self.live_state.superseded(session_id, offset)

    def note_live_state_owner(self, session_id: str, source: str) -> None:
        with self._lock:
            self.live_state.mark_owner(session_id, source)

    def _status_owned_locked(self, session_id: str) -> bool:
        return self._has_active_run_locked(session_id) or self.live_state.busy(session_id, self.clock())

    def apply_live_state(
        self,
        session_id: str,
        request: LiveStateRequest,
        *,
        settled_offset: int | None = None,
    ) -> tuple[Session, LiveClaim | None]:
        """Record a live claim and apply it to the session's status.

        Returns the (possibly updated) session and the accepted claim, or
        ``None`` for a stale/duplicate update. Raises ``SessionNotFoundError``
        or ``LiveStateRejected``. Claim and status change happen under one
        lock, so observations materialized concurrently see both.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            ensure_accepts_live_state(
                session, source=request.source, owner=self.live_state.owner(session_id)
            )
            now = self.clock()
            claim = self.live_state.offer(session_id, request, now, settled_offset=settled_offset)
            if claim is None:
                return session.model_copy(deep=True), None
            status = status_for_claim(session, claim, has_active_run=self._has_active_run_locked(session_id))
            if status != session.status:
                session = session.model_copy(update={"status": status, "updated_at": now})
                self.upsert_session(session)
            return session.model_copy(deep=True), claim

    def _has_active_run_locked(self, session_id: str) -> bool:
        return any(
            run.session_id == session_id and run.status in ("queued", "running")
            for run in self._runs.values()
        )

    def list_runs(self, session_id: str) -> list[Run]:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            return [run.model_copy(deep=True) for run in self._runs.values() if run.session_id == session_id]

    def get_run(self, session_id: str, run_id: str) -> Run:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            run = self._runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise RunNotFoundError(run_id)
        return run.model_copy(deep=True)

    def interrupt_run(self, session_id: str, run_id: str) -> Run:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                raise RunNotFoundError(run_id)

            # First-terminal-wins: a late DELETE on an already-completed
            # run must not rewrite its outcome (see investigation
            # 2026-05-15 where an 11-min-late interrupt corrupted the
            # historical record).
            if run.status in RUN_TERMINAL_STATUSES:
                return run.model_copy(deep=True)

            interrupted = run.model_copy(
                update={
                    "status": "interrupted",
                    "completed_at": utc_now(),
                    "stop_reason": "interrupted",
                }
            )
            self._runs[run_id] = interrupted
            return interrupted.model_copy(deep=True)

    def finish_run(
        self,
        session_id: str,
        run_id: str,
        *,
        status: RunStatus,
        stop_reason: StopReason | None = None,
    ) -> Run:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                raise RunNotFoundError(run_id)

            finished = run.model_copy(
                update={
                    "status": status,
                    "completed_at": utc_now(),
                    "stop_reason": stop_reason,
                }
            )
            self._runs[run_id] = finished
            # Only flip the session to idle if no other run for this
            # session is still queued or running. A successor may have
            # been promoted from the per-session FIFO queue before this
            # finish_run was scheduled — if so, leaving the session at
            # idle would contradict the repo's actual state.
            other_active = any(
                r.session_id == session_id
                and r.id != run_id
                and r.status in ("queued", "running")
                for r in self._runs.values()
            )
            if not other_active:
                # A still-valid live busy claim (lower priority than the run)
                # takes over again; otherwise the session is idle.
                after = "running" if self.live_state.busy(session_id, self.clock()) else "idle"
                self._sessions[session_id] = session.model_copy(
                    update={"status": after, "updated_at": utc_now()}
                )
            return finished.model_copy(deep=True)

    def list_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            return [message.model_copy(deep=True) for message in self._messages.get(session_id, [])]

    def materialize_event(self, event: Event, *, store_event: bool = True) -> None:
        del store_event
        if event.event == "session.updated":
            session_data = event.data.get("session")
            if isinstance(session_data, dict):
                incoming = Session.model_validate(session_data)
                # Read, merge and write under one lock so a concurrent
                # create_run/finish_run cannot interleave with the
                # active-run check.
                with self._lock:
                    existing = self._sessions.get(incoming.id)
                    active = self._status_owned_locked(incoming.id)
                # Preservation rules live in ``merge_observed_session``
                # so the in-memory and SQLite paths can't drift: an
                # observation refreshes the conversation's shape
                # (backend / model / project / status) and nothing
                # else. It returns None for the origin-downgrade case
                # (external observation of a harness-owned session),
                # which once caused the bridge to adopt a channel away
                # from its live session.
                    merged = merge_observed_session(incoming, existing, has_active_run=active)
                    if merged is not None:
                        self.upsert_session(merged)
            return

        if event.event in {"run.started", "run.completed", "run.failed", "run.interrupted"}:
            self._materialize_run_lifecycle_event(event)
            return

        if event.event == "message":
            message_data = event.data.get("message")
            if event.session_id and isinstance(message_data, dict):
                self.add_message(event.session_id, Message.model_validate(message_data))
            return

        if event.event == "run.usage":
            self._materialize_run_usage_event(event)
            return

    def _materialize_run_usage_event(self, event: Event) -> None:
        # Phase 3: apply per-turn ``Usage`` from the rollout to the
        # named ``Run`` (additive — usage sums across turns) and roll
        # the aggregate up into ``Session.stats.tokens``. Codex
        # ``token_count`` events additionally carry ``context_window``
        # so ``Session.stats.context_window`` updates in the same pass
        # (option (b) of the Phase 3 spec's open question).
        from agent_harness.usage import add_usage

        if event.session_id is None or event.run_id is None:
            return
        usage_data = event.data.get("usage")
        if not isinstance(usage_data, Mapping):
            return
        delta = Usage.model_validate(usage_data)
        context_window = _context_window_from(event.data)
        context_used = _context_used_from(event.data)
        with self._lock:
            # Check session presence BEFORE mutating the run. In SQLite
            # the outer ``materialize_event`` is wrapped in a single
            # transaction so a late raise rolls back; in-memory state
            # is not transactional and a mid-method raise would leave
            # ``Run.usage`` already incremented — a buffered re-flush
            # would then double-apply the delta.
            session = self._sessions.get(event.session_id)
            if session is None:
                raise SessionNotFoundError(event.session_id)
            run = self._runs.get(event.run_id)
            if run is None or run.session_id != event.session_id:
                return
            self._runs[event.run_id] = run.model_copy(
                update={"usage": add_usage(run.usage, delta)}
            )
            tokens = dict(session.stats.tokens or {})
            for key in ("input", "output", "cache_read", "cache_creation"):
                tokens[key] = int(tokens.get(key, 0)) + getattr(delta, key)
            new_cost = session.stats.cost_usd + delta.cost_usd
            stats_update: dict[str, object] = {
                "tokens": tokens,
                "cost_usd": new_cost,
            }
            if context_window is not None:
                stats_update["context_window"] = context_window
            if context_used is not None:
                # SNAPSHOT semantics: overwrite, never sum. A None
                # value means "no fresh observation" (omitted from
                # event); leave the prior snapshot untouched.
                stats_update["context_used"] = context_used
            self._sessions[event.session_id] = session.model_copy(
                update={
                    "stats": session.stats.model_copy(update=stats_update),
                    "updated_at": utc_now(),
                }
            )

    def _materialize_run_lifecycle_event(self, event: Event) -> None:
        if event.session_id is None or event.run_id is None:
            return
        with self._lock:
            run = self._runs.get(event.run_id)
            if run is None or run.session_id != event.session_id:
                return

            # First-terminal-wins (see ``interrupt_run``): once a run is
            # in a terminal status, a later lifecycle event from a
            # different source (e.g. API-published ``run.interrupted``
            # for an already-completed run) must not overwrite it.
            if run.status in RUN_TERMINAL_STATUSES:
                return

            status_by_event: dict[str, RunStatus] = {
                "run.started": "running",
                "run.completed": "completed",
                "run.failed": "failed",
                "run.interrupted": "interrupted",
            }
            update: dict[str, object] = {"status": status_by_event[event.event]}
            if event.event == "run.started" and run.started_at is None:
                update["started_at"] = event.created_at
            if event.event in {"run.completed", "run.failed", "run.interrupted"}:
                update["completed_at"] = event.created_at
            if event.event == "run.interrupted":
                update["stop_reason"] = "interrupted"

            self._runs[event.run_id] = run.model_copy(update=update)

    def upsert_session(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._messages.setdefault(session.id, [])

    def add_message(self, session_id: str, message: Message) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            messages = self._messages.setdefault(session_id, [])
            if any(_messages_equivalent(existing, message) for existing in messages):
                return
            messages.append(message)
            session = self._sessions[session_id]
            self._sessions[session_id] = session.model_copy(
                update={
                    "updated_at": utc_now(),
                    "stats": session.stats.model_copy(update={"messages": session.stats.messages + 1}),
                }
            )


def _messages_equivalent(left: Message, right: Message) -> bool:
    return _message_key(left) == _message_key(right)


def _message_key(message: Message) -> tuple[str, str]:
    return (message.role, "".join(block.model_dump_json() for block in message.blocks))


def _context_window_from(data: Mapping[str, object] | object) -> int | None:
    if not isinstance(data, Mapping):
        return None
    raw = data.get("context_window")
    if raw is None:
        return None
    try:
        parsed = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _context_used_from(data: Mapping[str, object] | object) -> int | None:
    # Parsed snapshot of currently-loaded context for
    # ``Session.stats.context_used``. Returns ``None`` when the
    # ``run.usage`` event omitted the field (preserve prior snapshot)
    # or when the value can't be coerced to a non-negative integer.
    # Unlike ``context_window`` (ge=1), a zero is rejected upstream by
    # the parsers (zero is indistinguishable from "no observation
    # yet"), but the materializer accepts ge=0 to match the model's
    # constraint.
    if not isinstance(data, Mapping):
        return None
    raw = data.get("context_used")
    if raw is None:
        return None
    try:
        parsed = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
