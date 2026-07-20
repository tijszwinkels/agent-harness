from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import RLock
from typing import TypeVar

from pydantic import BaseModel

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
    Usage,
    utc_now,
)
from agent_harness.repository import (
    RunNotFoundError,
    SessionNotFoundError,
    _context_used_from,
    _context_window_from,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
TModel = TypeVar("TModel", bound=BaseModel)
RUN_LIFECYCLE_EVENTS = {"run.started", "run.completed", "run.failed", "run.interrupted"}
RUN_TERMINAL_EVENTS = RUN_LIFECYCLE_EVENTS - {"run.started"}

_SCHEMA = """
create table if not exists schema_migrations (
    version integer primary key,
    applied_at text not null
);

create table if not exists sessions (
    id text primary key,
    payload text not null,
    created_at text not null,
    updated_at text not null
);

create table if not exists runs (
    id text primary key,
    session_id text not null,
    payload text not null,
    started_at text,
    completed_at text,
    foreign key (session_id) references sessions(id)
);

create table if not exists messages (
    row_id integer primary key autoincrement,
    message_id text not null,
    session_id text not null,
    run_id text,
    payload text not null,
    timestamp text not null,
    foreign key (session_id) references sessions(id)
);

create table if not exists events (
    row_id integer primary key autoincrement,
    sequence integer,
    event text not null,
    session_id text,
    run_id text,
    payload text not null,
    created_at text not null
);

create table if not exists observer_offsets (
    path text primary key,
    next_offset integer not null,
    updated_at text not null
);

create index if not exists idx_runs_session_id on runs(session_id);
create index if not exists idx_messages_message_id on messages(message_id);
create index if not exists idx_messages_session_id on messages(session_id);
create index if not exists idx_messages_run_id on messages(run_id);
create index if not exists idx_events_sequence on events(sequence);
create index if not exists idx_events_session_id on events(session_id);
create index if not exists idx_events_run_id on events(run_id);
"""


class SQLiteRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = RLock()
        self._message_keys: dict[str, set[tuple[str, str]]] = {}
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("pragma foreign_keys = on")
        self._initialize_schema()
        self._reconcile_dangling_state()

    def _reconcile_dangling_state(self) -> None:
        # Run/session liveness lives in the in-memory ``RunManager``; on a
        # harness restart that state is gone but the SQLite rows persist.
        # Without this sweep, runs left at ``running`` / ``queued`` and
        # harness-owned sessions stuck at ``running`` would lie about their
        # state forever (the orchestrator never re-attaches to a dead
        # subprocess). Flip them to terminal/idle so a fresh process can
        # resume cleanly.
        #
        # External sessions (``origin == "external"``) are excluded: their
        # lifecycle is owned by the transcript observer + the actual CLI
        # process (claude-code / codex), not by the in-memory RunManager.
        # A live external session legitimately stays ``running`` across
        # harness restarts; idling it here would lie in the other direction.
        with self._lock, self._connection:
            run_rows = self._connection.execute(
                """
                select id, session_id, payload
                from runs
                where json_extract(payload, '$.status') in ('queued', 'running')
                """,
            ).fetchall()
            now = utc_now()
            failed_runs = 0
            for row in run_rows:
                try:
                    run = _model_from_row(row, "payload", Run)
                except Exception:
                    logger.exception(
                        "Skipping run during startup reconcile: id=%s",
                        row["id"],
                    )
                    continue
                reconciled = run.model_copy(
                    update={"status": "failed", "completed_at": now},
                )
                self._upsert_run(reconciled)
                failed_runs += 1

            session_rows = self._connection.execute(
                """
                select id, payload
                from sessions
                where json_extract(payload, '$.status') = 'running'
                  and json_extract(payload, '$.origin') = 'harness'
                """,
            ).fetchall()
            idle_sessions = 0
            for row in session_rows:
                try:
                    session = _model_from_row(row, "payload", Session)
                except Exception:
                    logger.exception(
                        "Skipping session during startup reconcile: id=%s",
                        row["id"],
                    )
                    continue
                self._upsert_session(
                    session.model_copy(
                        update={"status": "idle", "updated_at": now},
                    )
                )
                idle_sessions += 1

        if failed_runs or idle_sessions:
            logger.info(
                "Startup reconcile: marked %d run(s) failed, %d session(s) idle",
                failed_runs,
                idle_sessions,
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def get_observer_offsets(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "select path, next_offset from observer_offsets"
            ).fetchall()
        return {row["path"]: int(row["next_offset"]) for row in rows}

    def set_observer_offset(self, path: str, next_offset: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                insert into observer_offsets(path, next_offset, updated_at)
                values (?, ?, ?)
                on conflict(path) do update set
                    next_offset = excluded.next_offset,
                    updated_at = excluded.updated_at
                """,
                (path, int(next_offset), utc_now().isoformat()),
            )

    def create_session(self, request: CreateSessionRequest) -> Session:
        session = Session(
            backend=request.backend,
            model=request.model,
            project=request.project,
            title=request.title,
            bypass_permissions=request.bypass_permissions,
        )
        with self._lock, self._connection:
            self._upsert_session(session)
        return session.model_copy(deep=True)

    def create_forked_session(self, parent: Session, *, title: str | None) -> Session:
        child = Session.forked_child(parent, title=title)
        with self._lock, self._connection:
            self._upsert_session(child)
        return child.model_copy(deep=True)

    def list_sessions(self) -> list[Session]:
        with self._lock:
            rows = self._connection.execute("select payload from sessions order by rowid").fetchall()
        return [_model_from_row(row, "payload", Session) for row in rows]

    def get_session(self, session_id: str) -> Session:
        session = self._find_session(session_id)
        if session is None:
            logger.warning("SQLite session lookup failed: %s", session_id)
            raise SessionNotFoundError(session_id)
        return session

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return self._find_session_locked(session_id) is not None

    def archive_session(self, session_id: str) -> Session:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite session archive failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)

            archived = session.model_copy(update={"status": "archived", "updated_at": utc_now()})
            self._upsert_session(archived)
        return archived.model_copy(deep=True)

    def patch_session(self, session_id: str, fields: dict[str, object]) -> Session:
        # Mirror of InMemoryRepository.patch_session. Whitelisting is the
        # caller's responsibility (api.py uses PatchSessionRequest).
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite session patch failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            if not fields:
                return session.model_copy(deep=True)
            updated = session.model_copy(update={**fields, "updated_at": utc_now()})
            self._upsert_session(updated)
        return updated.model_copy(deep=True)

    def create_run(self, session_id: str, request: CreateRunRequest) -> Run:
        # Runs are born ``queued`` and stay that way until the orchestrator
        # spawns the subprocess (see ``start_run``). Mirrors InMemoryRepository.
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite run create failed because session was not found: %s", session_id)
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
            self._upsert_session(updated_session)
            self._upsert_run(run)
            self._insert_message(session_id, input_message)
        return run.model_copy(deep=True)

    def start_run(self, session_id: str, run_id: str) -> Run:
        with self._lock, self._connection:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite run start failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            row = self._connection.execute(
                "select payload from runs where id = ? and session_id = ?",
                (run_id, session_id),
            ).fetchone()
            if row is None:
                logger.warning("SQLite run start failed: session=%s run=%s", session_id, run_id)
                raise RunNotFoundError(run_id)

            run = _model_from_row(row, "payload", Run)
            started = run.model_copy(update={"status": "running", "started_at": utc_now()})
            self._upsert_run(started)
        return started.model_copy(deep=True)

    def drop_queued_runs(self, session_id: str) -> list[Run]:
        with self._lock, self._connection:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite drop_queued_runs failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            rows = self._connection.execute(
                "select payload from runs where session_id = ?",
                (session_id,),
            ).fetchall()
            dropped: list[Run] = []
            now = utc_now()
            for row in rows:
                run = _model_from_row(row, "payload", Run)
                if run.status != "queued":
                    continue
                interrupted = run.model_copy(
                    update={
                        "status": "interrupted",
                        "completed_at": now,
                        "stop_reason": "interrupted",
                    }
                )
                self._upsert_run(interrupted)
                dropped.append(interrupted.model_copy(deep=True))
        return dropped

    def list_runs(self, session_id: str) -> list[Run]:
        with self._lock:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite run list failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            rows = self._connection.execute(
                "select payload from runs where session_id = ? order by rowid",
                (session_id,),
            ).fetchall()
        return [_model_from_row(row, "payload", Run) for row in rows]

    def get_run(self, session_id: str, run_id: str) -> Run:
        with self._lock:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite run lookup failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            row = self._connection.execute(
                "select payload from runs where id = ? and session_id = ?",
                (run_id, session_id),
            ).fetchone()
        if row is None:
            logger.warning("SQLite run lookup failed: session=%s run=%s", session_id, run_id)
            raise RunNotFoundError(run_id)
        return _model_from_row(row, "payload", Run)

    def interrupt_run(self, session_id: str, run_id: str) -> Run:
        with self._lock, self._connection:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite run interrupt failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            row = self._connection.execute(
                "select payload from runs where id = ? and session_id = ?",
                (run_id, session_id),
            ).fetchone()
            if row is None:
                logger.warning("SQLite run interrupt failed: session=%s run=%s", session_id, run_id)
                raise RunNotFoundError(run_id)

            run = _model_from_row(row, "payload", Run)
            # First-terminal-wins: see ``InMemoryRepository.interrupt_run``.
            if run.status in RUN_TERMINAL_STATUSES:
                return run.model_copy(deep=True)

            interrupted = run.model_copy(
                update={
                    "status": "interrupted",
                    "completed_at": utc_now(),
                    "stop_reason": "interrupted",
                }
            )
            self._upsert_run(interrupted)
        return interrupted.model_copy(deep=True)

    def finish_run(
        self,
        session_id: str,
        run_id: str,
        *,
        status: RunStatus,
        stop_reason: StopReason | None = None,
    ) -> Run:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite run finish failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            row = self._connection.execute(
                "select payload from runs where id = ? and session_id = ?",
                (run_id, session_id),
            ).fetchone()
            if row is None:
                logger.warning("SQLite run finish failed: session=%s run=%s", session_id, run_id)
                raise RunNotFoundError(run_id)

            run = _model_from_row(row, "payload", Run)
            finished = run.model_copy(
                update={
                    "status": status,
                    "completed_at": utc_now(),
                    "stop_reason": stop_reason,
                }
            )
            self._upsert_run(finished)
            # Only flip the session to idle if no other run for this
            # session is still queued or running. See the matching
            # InMemoryRepository.finish_run comment — a successor may
            # already be running by the time this completion is
            # materialized.
            other_active = self._connection.execute(
                """
                select 1 from runs
                where session_id = ?
                  and id != ?
                  and json_extract(payload, '$.status') in ('queued', 'running')
                limit 1
                """,
                (session_id, run_id),
            ).fetchone()
            if other_active is None:
                self._upsert_session(
                    session.model_copy(update={"status": "idle", "updated_at": utc_now()})
                )
        return finished.model_copy(deep=True)

    def list_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            if self._find_session_locked(session_id) is None:
                logger.warning("SQLite message list failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)
            rows = self._connection.execute(
                "select payload from messages where session_id = ? order by row_id",
                (session_id,),
            ).fetchall()
        return [_model_from_row(row, "payload", Message) for row in rows]

    def list_events(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        after: int = 0,
    ) -> list[Event]:
        query = "select payload from events where coalesce(sequence, 0) > ?"
        parameters: list[object] = [after]
        if session_id is not None:
            query += " and session_id = ?"
            parameters.append(session_id)
        if run_id is not None:
            query += " and run_id = ?"
            parameters.append(run_id)
        query += " order by sequence"

        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [_model_from_row(row, "payload", Event) for row in rows]

    def max_sequence(self, *, session_id: str | None = None) -> int:
        query = "select max(sequence) from events"
        parameters: list[object] = []
        if session_id is not None:
            query += " where session_id = ?"
            parameters.append(session_id)

        with self._lock:
            row = self._connection.execute(query, parameters).fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def append_event(self, event: Event) -> Event:
        # Phase 3: ``append_event`` is a pure event-row insert. All
        # side-effect materialization (run lifecycle, run.usage,
        # session.updated, message) is orchestrated by
        # ``DurableEventBus.publish``, which calls
        # ``materialize_event(store_event=False)`` after ``append_event``
        # under the same lock. Single materialization point per event;
        # no double-application even if a caller invokes both paths
        # (Falcon's PR #11 bug becomes structurally impossible).
        with self._lock, self._connection:
            published = event.with_sequence(self._next_event_sequence_locked())
            self._insert_event(published)
        return published.model_copy(deep=True)

    def materialize_event(self, event: Event, *, store_event: bool = True) -> None:
        with self._lock, self._connection:
            if store_event:
                self._insert_event(event)

            if event.event == "session.updated":
                session_data = event.data.get("session")
                if isinstance(session_data, dict):
                    incoming = Session.model_validate(session_data)
                    existing = self._find_session_locked(incoming.id)
                    # The external transcript observer (and the
                    # freshness tick) emit ``session.updated`` payloads
                    # built from a freshly-constructed ``Session``
                    # whose ``stats`` field defaults to a zero-valued
                    # ``SessionStats()``. If a harness-spawned record
                    # already exists under this canonical id we skip
                    # the upsert entirely (origin-downgrade guard —
                    # Phase 1 incident; a downgrade caused the bridge
                    # to adopt the channel away from the live session
                    # on next MM post).
                    #
                    # For absent or external-origin records we DO
                    # upsert, but we must preserve any accumulated
                    # stats on the existing row. Without this, every
                    # assistant / turn_context line in a multi-turn
                    # external-resume session would wipe prior turns'
                    # aggregated token counts (``run.usage`` is
                    # additive into Session.stats — see
                    # ``_materialize_run_usage_event``). PR #11
                    # originally added this guard; the Phase 3
                    # cherry-pick of the parsers dropped it.
                    if existing is None:
                        self._upsert_session(incoming)
                    elif existing.origin == "external":
                        self._upsert_session(
                            incoming.model_copy(update={"stats": existing.stats})
                        )
                    elif existing.origin == "harness" and incoming.origin == "harness":
                        # Harness→harness update: observer emits this
                        # to set Session.codex_resume_id after a codex
                        # rollout binds to a harness session
                        # (specs/2026-05-21-codex-resume.md). Same
                        # stats-preservation guard as the external
                        # branch — incoming has a zero-valued stats
                        # block and we mustn't wipe the accumulated
                        # tokens / context_window / context_used.
                        # The Phase 1 origin-downgrade guard
                        # (external→harness rejected) stays intact:
                        # only matching-origin incoming fires here.
                        #
                        # Halcyon's NEEDS-FIX on PR #18: also
                        # preserve existing.codex_resume_id when the
                        # incoming payload's field is None — a
                        # ``_maybe_publish_status_flip`` event whose
                        # payload was built BEFORE the resume-id
                        # event landed would otherwise clobber the
                        # just-written field. Same shape as the
                        # stats preservation.
                        updates: dict[str, object] = {"stats": existing.stats}
                        if (
                            incoming.codex_resume_id is None
                            and existing.codex_resume_id is not None
                        ):
                            updates["codex_resume_id"] = existing.codex_resume_id
                        self._upsert_session(
                            incoming.model_copy(update=updates)
                        )
                return

            if event.event in RUN_LIFECYCLE_EVENTS:
                self._materialize_run_lifecycle_event(event)
                return

            if event.event == "message":
                self._materialize_message_event(event)
                return

            if event.event == "run.usage":
                self._materialize_run_usage_event(event)

    def _materialize_message_event(self, event: Event, *, dedupe_by_content: bool = True) -> None:
        message_data = event.data.get("message")
        if not event.session_id or not isinstance(message_data, dict):
            return

        session = self._find_session_locked(event.session_id)
        if session is None:
            logger.warning(
                "SQLite event materialization skipped message because session was not found: %s",
                event.session_id,
            )
            raise SessionNotFoundError(event.session_id)

        message = Message.model_validate(message_data)
        inserted = (
            self._insert_message_if_new(event.session_id, message)
            if dedupe_by_content
            else self._insert_message_without_content_dedupe(event.session_id, message)
        )
        if inserted:
            self._upsert_session(
                session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "stats": session.stats.model_copy(update={"messages": session.stats.messages + 1}),
                    }
                )
            )

    def _insert_message_without_content_dedupe(self, session_id: str, message: Message) -> bool:
        row = self._connection.execute(
            "select 1 from messages where message_id = ?",
            (message.id,),
        ).fetchone()
        if row is not None:
            return False
        self._insert_message(session_id, message)
        return True

    def upsert_session(self, session: Session) -> None:
        with self._lock, self._connection:
            self._upsert_session(session)

    def add_message(self, session_id: str, message: Message) -> None:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite message add failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)

            if self._insert_message_if_new(session_id, message):
                self._upsert_session(
                    session.model_copy(
                        update={
                            "updated_at": utc_now(),
                            "stats": session.stats.model_copy(update={"messages": session.stats.messages + 1}),
                        }
                    )
                )

    def _initialize_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "insert or replace into schema_migrations(version, applied_at) values (?, ?)",
                (SCHEMA_VERSION, utc_now().isoformat()),
            )

    def _find_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._find_session_locked(session_id)

    def _find_session_locked(self, session_id: str) -> Session | None:
        row = self._connection.execute("select payload from sessions where id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return _model_from_row(row, "payload", Session)

    def _upsert_session(self, session: Session) -> None:
        self._connection.execute(
            """
            insert into sessions(id, payload, created_at, updated_at)
            values (?, ?, ?, ?)
            on conflict(id) do update set
                payload = excluded.payload,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.model_dump_json(),
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )

    def _upsert_run(self, run: Run) -> None:
        self._connection.execute(
            """
            insert into runs(id, session_id, payload, started_at, completed_at)
            values (?, ?, ?, ?, ?)
            on conflict(id) do update set
                session_id = excluded.session_id,
                payload = excluded.payload,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at
            """,
            (
                run.id,
                run.session_id,
                run.model_dump_json(),
                run.started_at.isoformat() if run.started_at else None,
                run.completed_at.isoformat() if run.completed_at else None,
            ),
        )

    def _insert_message(self, session_id: str, message: Message) -> None:
        self._connection.execute(
            """
            insert into messages(message_id, session_id, run_id, payload, timestamp)
            values (?, ?, ?, ?, ?)
            """,
            (
                message.id,
                session_id,
                message.run_id,
                message.model_dump_json(),
                message.timestamp.isoformat(),
            ),
        )
        self._message_keys.setdefault(session_id, set()).add(_message_key(message))

    def _insert_message_if_new(self, session_id: str, message: Message) -> bool:
        keys = self._message_keys.get(session_id)
        if keys is None:
            keys = {
                _message_key(_model_from_row(row, "payload", Message))
                for row in self._connection.execute(
                    "select payload from messages where session_id = ?",
                    (session_id,),
                ).fetchall()
            }
            self._message_keys[session_id] = keys

        if _message_key(message) in keys:
            return False
        self._insert_message(session_id, message)
        return True

    def _insert_event(self, event: Event) -> None:
        stored = event
        if stored.sequence is None:
            stored = event.with_sequence(self._next_event_sequence_locked())
        self._connection.execute(
            """
            insert into events(sequence, event, session_id, run_id, payload, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                stored.sequence,
                stored.event,
                stored.session_id,
                stored.run_id,
                stored.model_dump_json(),
                stored.created_at.isoformat(),
            ),
        )

    def _next_event_sequence_locked(self) -> int:
        row = self._connection.execute("select coalesce(max(sequence), 0) + 1 from events").fetchone()
        if row is None:
            logger.error("SQLite event sequence lookup returned no row")
            raise RuntimeError("Failed to allocate event sequence")
        return int(row[0])

    def _materialize_run_usage_event(self, event: Event) -> None:
        # Phase 3: additive per-turn usage from the rollout. Applies
        # the ``Usage`` to the named run AND rolls the same delta into
        # ``Session.stats.tokens``. Codex ``token_count`` events
        # additionally carry ``context_window`` → updates
        # ``Session.stats.context_window`` in the same pass.
        from collections.abc import Mapping as _Mapping
        from agent_harness.usage import add_usage

        if event.session_id is None or event.run_id is None:
            return
        usage_data = event.data.get("usage")
        if not isinstance(usage_data, _Mapping):
            return
        delta = Usage.model_validate(usage_data)
        context_window = _context_window_from(event.data)
        context_used = _context_used_from(event.data)

        row = self._connection.execute(
            "select payload from runs where id = ? and session_id = ?",
            (event.run_id, event.session_id),
        ).fetchone()
        if row is None:
            logger.warning(
                "SQLite run.usage materialization skipped missing run: session=%s run=%s",
                event.session_id,
                event.run_id,
            )
            return
        run = _model_from_row(row, "payload", Run)
        self._upsert_run(run.model_copy(update={"usage": add_usage(run.usage, delta)}))

        session = self._find_session_locked(event.session_id)
        if session is None:
            raise SessionNotFoundError(event.session_id)
        tokens = dict(session.stats.tokens or {})
        for key in ("input", "output", "cache_read", "cache_creation"):
            tokens[key] = int(tokens.get(key, 0)) + getattr(delta, key)
        stats_update: dict[str, object] = {
            "tokens": tokens,
            "cost_usd": session.stats.cost_usd + delta.cost_usd,
        }
        if context_window is not None:
            stats_update["context_window"] = context_window
        if context_used is not None:
            # SNAPSHOT semantics: overwrite, never sum. A None value
            # means "no fresh observation" (event omitted the field);
            # leave the prior snapshot in place.
            stats_update["context_used"] = context_used
        self._upsert_session(
            session.model_copy(
                update={
                    "stats": session.stats.model_copy(update=stats_update),
                    "updated_at": utc_now(),
                }
            )
        )

    def _materialize_run_lifecycle_event(self, event: Event) -> None:
        if event.session_id is None or event.run_id is None:
            logger.warning(
                "SQLite run lifecycle materialization skipped event without ids: event=%s sequence=%s",
                event.event,
                event.sequence,
            )
            return

        row = self._connection.execute(
            "select payload from runs where id = ? and session_id = ?",
            (event.run_id, event.session_id),
        ).fetchone()
        if row is None:
            logger.warning(
                "SQLite run lifecycle materialization skipped missing run: event=%s session=%s run=%s",
                event.event,
                event.session_id,
                event.run_id,
            )
            return

        run = _model_from_row(row, "payload", Run)
        # First-terminal-wins (see ``interrupt_run``): once a run is in
        # a terminal status, a later lifecycle event from a different
        # source must not overwrite its outcome.
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
        if event.event in RUN_TERMINAL_EVENTS:
            update["completed_at"] = event.created_at
        if event.event == "run.interrupted":
            update["stop_reason"] = "interrupted"

        self._upsert_run(run.model_copy(update=update))


def open_sqlite_repository(database_path: str | Path) -> SQLiteRepository:
    try:
        connection = sqlite3.connect(database_path, check_same_thread=False)
    except sqlite3.Error:
        logger.exception("Failed to open SQLite repository: %s", database_path)
        raise
    return SQLiteRepository(connection)


def _model_from_row(row: sqlite3.Row, column: str, model_type: type[TModel]) -> TModel:
    payload = row[column]
    try:
        return model_type.model_validate_json(payload)
    except ValueError:
        logger.exception("Failed to load %s payload from SQLite", model_type.__name__)
        raise


def _messages_equivalent(left: Message, right: Message) -> bool:
    return _message_key(left) == _message_key(right)


def _message_key(message: Message) -> tuple[str, str]:
    return (message.role, "".join(block.model_dump_json() for block in message.blocks))
