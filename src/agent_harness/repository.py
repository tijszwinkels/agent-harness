from __future__ import annotations

from datetime import datetime, timedelta
from threading import RLock

from agent_harness.models import (
    CreateRunRequest,
    CreateSessionRequest,
    Event,
    Message,
    RUN_TERMINAL_STATUSES,
    Run,
    RunStatus,
    Session,
    StopReason,
    utc_now,
)

# Session statuses that count as "terminal" for the codex reconcile pre-check.
# Session.status is one of {idle, running, waiting_for_input, archived}; only
# archived is a true terminal — the rest are transient states that the harness
# can resume from. Spec wording "completed/error/interrupted" describes Run
# lifecycle, not Session; the Session-level analogue is archived.
_RECONCILE_TERMINAL_SESSION_STATUSES = frozenset({"archived"})


class SessionNotFoundError(KeyError):
    pass


class RunNotFoundError(KeyError):
    pass


class InMemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, Session] = {}
        self._runs: dict[str, Run] = {}
        self._messages: dict[str, list[Message]] = {}

    def create_session(self, request: CreateSessionRequest) -> Session:
        session = Session(
            backend=request.backend,
            model=request.model,
            project=request.project,
            title=request.title,
            bypass_permissions=request.bypass_permissions,
        )
        with self._lock:
            self._sessions[session.id] = session
            self._messages[session.id] = []
        return session.model_copy(deep=True)

    def list_sessions(self) -> list[Session]:
        with self._lock:
            return [session.model_copy(deep=True) for session in self._sessions.values()]

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
                self._sessions[session_id] = session.model_copy(
                    update={"status": "idle", "updated_at": utc_now()}
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
                with self._lock:
                    existing = self._sessions.get(incoming.id)
                # The external transcript observer always emits payloads
                # with origin="external"; if a harness-spawned record
                # already exists under the canonical ses_<hex> id, skip
                # the upsert so we don't downgrade its origin /
                # bypass_permissions / etc. See test_storage.py for the
                # rationale (a downgrade causes the bridge to adopt the
                # channel away from the live session on next MM post).
                if existing is None or existing.origin == "external":
                    self.upsert_session(incoming)
            return

        if event.event in {"run.started", "run.completed", "run.failed", "run.interrupted"}:
            self._materialize_run_lifecycle_event(event)
            return

        if event.event == "message":
            message_data = event.data.get("message")
            if event.session_id and isinstance(message_data, dict):
                self.add_message(event.session_id, Message.model_validate(message_data))

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

    def find_recent_codex_harness_session(
        self,
        *,
        cwd: str,
        earliest_event_at: datetime,
        window_seconds: float = 30.0,
    ) -> Session | None:
        # Return the non-terminal harness codex session in ``cwd`` whose
        # ``created_at`` is CLOSEST in time to ``earliest_event_at``, within
        # ``window_seconds``. Sessions already bound to a different rollout
        # (``codex_internal_id`` set) are excluded so a second rollout in
        # the same cwd can't steal the first rollout's session.
        #
        # The "closest" tiebreaker matters when two harness codex sessions
        # are spawned back-to-back in the same cwd: picking "most recently
        # created" would mis-bind the first rollout to the second session,
        # because the second session's ``created_at`` is the larger value.
        window = timedelta(seconds=window_seconds)
        best: Session | None = None
        best_delta: timedelta | None = None
        with self._lock:
            for session in self._sessions.values():
                if session.backend != "codex":
                    continue
                if session.origin != "harness":
                    continue
                if session.status in _RECONCILE_TERMINAL_SESSION_STATUSES:
                    continue
                if session.codex_internal_id is not None:
                    continue
                if session.project.path != cwd:
                    continue
                delta = abs(session.created_at - earliest_event_at)
                if delta > window:
                    continue
                if best_delta is None or delta < best_delta:
                    best = session
                    best_delta = delta
        return best.model_copy(deep=True) if best is not None else None

    def find_session_by_codex_internal_id(self, internal_id: str) -> Session | None:
        # Used by the observer on restart to re-derive the path→harness
        # binding without rerunning the time-window match (which may now
        # fail if the harness session has been archived since the bind).
        with self._lock:
            for session in self._sessions.values():
                if session.codex_internal_id == internal_id:
                    return session.model_copy(deep=True)
        return None

    def bind_codex_rollout(self, session_id: str, internal_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            updated = session.model_copy(
                update={"codex_internal_id": internal_id, "updated_at": utc_now()}
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

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
