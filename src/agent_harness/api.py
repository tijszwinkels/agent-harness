from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from agent_harness.backends import BackendRegistry, default_backend_registry
from agent_harness.events import InMemoryEventBus
from agent_harness.models import CreateSessionRequest, Event
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

    app = FastAPI(title="agent-harness", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/backends")
    async def list_backends() -> dict[str, object]:
        return {"data": backends.list()}

    @app.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(request: CreateSessionRequest) -> object:
        if not backends.has(request.backend):
            logger.warning("Rejecting session create for unknown backend: %s", request.backend)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown backend")

        session = repo.create_session(request)
        await events.publish(
            Event(
                event="session.updated",
                session_id=session.id,
                data={"session": jsonable_encoder(session)},
            )
        )
        return session

    @app.get("/v1/sessions")
    async def list_sessions() -> dict[str, object]:
        return {"data": repo.list_sessions()}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> object:
        try:
            return repo.get_session(session_id)
        except SessionNotFoundError as exc:
            logger.warning("Session lookup failed: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    @app.delete("/v1/sessions/{session_id}")
    async def archive_session(session_id: str) -> object:
        try:
            session = repo.archive_session(session_id)
        except SessionNotFoundError as exc:
            logger.warning("Session archive failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        await events.publish(
            Event(
                event="session.updated",
                session_id=session.id,
                data={"fields_changed": ["status", "updated_at"]},
            )
        )
        return session

    @app.get("/v1/events")
    async def stream_events(
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = 0 if from_ == "beginning" else after
        return StreamingResponse(_sse_stream(events, after=replay_after), media_type="text/event-stream")

    return app


async def _sse_stream(event_bus: InMemoryEventBus, *, after: int = 0) -> AsyncIterator[str]:
    async for event in event_bus.subscribe(after=after):
        yield _format_sse(event)


def _format_sse(event: Event) -> str:
    payload = json.dumps(jsonable_encoder(event), separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {payload}\n\n"
