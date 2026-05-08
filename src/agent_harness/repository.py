from __future__ import annotations

from threading import RLock
from uuid import UUID

from agent_harness.models import CreateSessionRequest, Session, utc_now


class SessionNotFoundError(KeyError):
    pass


class InMemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[UUID, Session] = {}

    def create_session(self, request: CreateSessionRequest) -> Session:
        session = Session(
            backend_id=request.backend_id,
            title=request.title,
            metadata=request.metadata,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session.model_copy(deep=True)

    def list_sessions(self) -> list[Session]:
        with self._lock:
            return [session.model_copy(deep=True) for session in self._sessions.values()]

    def get_session(self, session_id: UUID) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(str(session_id))
        return session.model_copy(deep=True)

    def archive_session(self, session_id: UUID) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(str(session_id))

            archived = session.model_copy(update={"status": "archived", "updated_at": utc_now()})
            self._sessions[session_id] = archived
            return archived.model_copy(deep=True)
