from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import RLock
from typing import TypeVar

from pydantic import BaseModel

from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Message, Run, Session, utc_now
from agent_harness.repository import RunNotFoundError, SessionNotFoundError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TModel = TypeVar("TModel", bound=BaseModel)

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
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("pragma foreign_keys = on")
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_session(self, request: CreateSessionRequest) -> Session:
        session = Session(
            backend=request.backend,
            model=request.model,
            project=request.project,
            title=request.title,
        )
        with self._lock, self._connection:
            self._upsert_session(session)
        return session.model_copy(deep=True)

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

    def archive_session(self, session_id: str) -> Session:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite session archive failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)

            archived = session.model_copy(update={"status": "archived", "updated_at": utc_now()})
            self._upsert_session(archived)
        return archived.model_copy(deep=True)

    def create_run(self, session_id: str, request: CreateRunRequest) -> Run:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite run create failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)

            input_message = Message.user(request.message)
            run = Run(
                session_id=session.id,
                status="running",
                started_at=utc_now(),
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
            interrupted = run.model_copy(
                update={
                    "status": "interrupted",
                    "completed_at": utc_now(),
                    "stop_reason": "interrupted",
                }
            )
            self._upsert_run(interrupted)
        return interrupted.model_copy(deep=True)

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
        query += " order by row_id"

        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [_model_from_row(row, "payload", Event) for row in rows]

    def materialize_event(self, event: Event) -> None:
        with self._lock, self._connection:
            self._insert_event(event)

            if event.event == "session.updated":
                session_data = event.data.get("session")
                if isinstance(session_data, dict):
                    self._upsert_session(Session.model_validate(session_data))
                return

            if event.event == "message":
                message_data = event.data.get("message")
                if event.session_id and isinstance(message_data, dict):
                    if self._find_session_locked(event.session_id) is None:
                        logger.warning(
                            "SQLite event materialization skipped message because session was not found: %s",
                            event.session_id,
                        )
                        raise SessionNotFoundError(event.session_id)
                    self._insert_message(event.session_id, Message.model_validate(message_data))

    def upsert_session(self, session: Session) -> None:
        with self._lock, self._connection:
            self._upsert_session(session)

    def add_message(self, session_id: str, message: Message) -> None:
        with self._lock, self._connection:
            session = self._find_session_locked(session_id)
            if session is None:
                logger.warning("SQLite message add failed because session was not found: %s", session_id)
                raise SessionNotFoundError(session_id)

            self._insert_message(session_id, message)
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
                "insert or ignore into schema_migrations(version, applied_at) values (?, ?)",
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

    def _insert_event(self, event: Event) -> None:
        self._connection.execute(
            """
            insert into events(sequence, event, session_id, run_id, payload, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                event.sequence,
                event.event,
                event.session_id,
                event.run_id,
                event.model_dump_json(),
                event.created_at.isoformat(),
            ),
        )


def open_sqlite_repository(database_path: str | Path) -> SQLiteRepository:
    try:
        connection = sqlite3.connect(database_path)
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
