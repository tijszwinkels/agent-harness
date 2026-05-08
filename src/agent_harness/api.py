from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from agent_harness.backends import BackendRegistry, default_backend_registry
from agent_harness.events import InMemoryEventBus
from agent_harness.models import (
    BackendListResponse,
    CreateSessionRequest,
    Event,
    SessionListResponse,
    SessionResponse,
)
from agent_harness.repository import InMemoryRepository, SessionNotFoundError

logger = logging.getLogger(__name__)


def create_app(
    *,
    repository: InMemoryRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
    backend_registry: BackendRegistry | None = None,
) -> FastAPI:
    repo = repository or InMemoryRepository()
    events = event_bus or InMemoryEventBus()
    backends = backend_registry or default_backend_registry()

    app = FastAPI(title="Agent Harness", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/backends")
    async def list_backends() -> BackendListResponse:
        return BackendListResponse(backends=backends.list())

    @app.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(request: CreateSessionRequest) -> SessionResponse:
        if not backends.has(request.backend_id):
            logger.warning("Rejecting session create for unknown backend: %s", request.backend_id)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown backend")

        session = repo.create_session(request)
        await events.publish(
            Event(
                type="session.created",
                session_id=session.id,
                data={"session": jsonable_encoder(session)},
            )
        )
        return SessionResponse(session=session)

    @app.get("/v1/sessions")
    async def list_sessions() -> SessionListResponse:
        return SessionListResponse(sessions=repo.list_sessions())

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: UUID) -> SessionResponse:
        try:
            return SessionResponse(session=repo.get_session(session_id))
        except SessionNotFoundError as exc:
            logger.warning("Session lookup failed: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    @app.post("/v1/sessions/{session_id}/archive")
    async def archive_session(session_id: UUID) -> SessionResponse:
        try:
            session = repo.archive_session(session_id)
        except SessionNotFoundError as exc:
            logger.warning("Session archive failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        await events.publish(
            Event(
                type="session.archived",
                session_id=session.id,
                data={"session": jsonable_encoder(session)},
            )
        )
        return SessionResponse(session=session)

    @app.get("/v1/events")
    async def stream_events(after: int = Query(default=0, ge=0)) -> StreamingResponse:
        return StreamingResponse(_sse_stream(events, after=after), media_type="text/event-stream")

    return app


async def _sse_stream(event_bus: InMemoryEventBus, *, after: int = 0) -> AsyncIterator[str]:
    async for event in event_bus.subscribe(after=after):
        yield _format_sse(event)


def _format_sse(event: Event) -> str:
    payload = json.dumps(jsonable_encoder(event), separators=(",", ":"))
    return f"id: {event.seq}\nevent: {event.type}\ndata: {payload}\n\n"
