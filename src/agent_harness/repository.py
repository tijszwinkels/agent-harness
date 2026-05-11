from __future__ import annotations

from threading import RLock

from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Message, Run, Session, utc_now


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

    def create_run(self, session_id: str, request: CreateRunRequest) -> Run:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
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
            self._sessions[session_id] = updated_session
            self._runs[run.id] = run
            self._messages.setdefault(session_id, []).append(input_message)
            return run.model_copy(deep=True)

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

            interrupted = run.model_copy(
                update={
                    "status": "interrupted",
                    "completed_at": utc_now(),
                    "stop_reason": "interrupted",
                }
            )
            self._runs[run_id] = interrupted
            return interrupted.model_copy(deep=True)

    def list_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            return [message.model_copy(deep=True) for message in self._messages.get(session_id, [])]

    def materialize_event(self, event: Event) -> None:
        if event.event == "session.updated":
            session_data = event.data.get("session")
            if isinstance(session_data, dict):
                self.upsert_session(Session.model_validate(session_data))
            return

        if event.event == "message":
            message_data = event.data.get("message")
            if event.session_id and isinstance(message_data, dict):
                self.add_message(event.session_id, Message.model_validate(message_data))

    def upsert_session(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._messages.setdefault(session.id, [])

    def add_message(self, session_id: str, message: Message) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            self._messages.setdefault(session_id, []).append(message)
            session = self._sessions[session_id]
            self._sessions[session_id] = session.model_copy(
                update={
                    "updated_at": utc_now(),
                    "stats": session.stats.model_copy(update={"messages": session.stats.messages + 1}),
                }
            )
