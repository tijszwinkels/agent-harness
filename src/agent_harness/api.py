from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from agent_harness.backends import BackendRegistry, default_backend_registry
from agent_harness.events import InMemoryEventBus
from agent_harness.models import CreateRunRequest, CreateRunResponse, CreateSessionRequest, Event
from agent_harness.repository import InMemoryRepository, RunNotFoundError, SessionNotFoundError

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

    @app.get("/v1/sessions/{session_id}/messages")
    async def list_messages(session_id: str) -> dict[str, object]:
        try:
            return {"data": repo.list_messages(session_id)}
        except SessionNotFoundError as exc:
            logger.warning("Message list failed because session was not found: %s", session_id)
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

    @app.post("/v1/sessions/{session_id}/runs", status_code=status.HTTP_202_ACCEPTED)
    async def create_run(session_id: str, request: CreateRunRequest) -> CreateRunResponse:
        try:
            run = repo.create_run(session_id, request)
        except SessionNotFoundError as exc:
            logger.warning("Run create failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        await events.publish(Event(event="run.started", session_id=session_id, run_id=run.id, data={}))
        return CreateRunResponse(session_id=session_id, run_id=run.id)

    @app.get("/v1/sessions/{session_id}/runs")
    async def list_runs(session_id: str) -> dict[str, object]:
        try:
            return {"data": repo.list_runs(session_id)}
        except SessionNotFoundError as exc:
            logger.warning("Run list failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    @app.get("/v1/sessions/{session_id}/runs/{run_id}")
    async def get_run(session_id: str, run_id: str) -> object:
        try:
            return repo.get_run(session_id, run_id)
        except SessionNotFoundError as exc:
            logger.warning("Run lookup failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        except RunNotFoundError as exc:
            logger.warning("Run lookup failed: session=%s run=%s", session_id, run_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from exc

    @app.delete("/v1/sessions/{session_id}/runs/{run_id}")
    async def interrupt_run(session_id: str, run_id: str) -> object:
        try:
            run = repo.interrupt_run(session_id, run_id)
        except SessionNotFoundError as exc:
            logger.warning("Run interrupt failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        except RunNotFoundError as exc:
            logger.warning("Run interrupt failed: session=%s run=%s", session_id, run_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from exc

        await events.publish(Event(event="run.interrupted", session_id=session_id, run_id=run.id, data={}))
        return run

    @app.get("/v1/events")
    async def stream_events(
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = 0 if from_ == "beginning" else after
        return StreamingResponse(_sse_stream(events, after=replay_after), media_type="text/event-stream")

    @app.get("/v1/sessions/{session_id}/events")
    async def stream_session_events(
        session_id: str,
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = 0 if from_ == "beginning" else after
        return StreamingResponse(
            _sse_stream(events, after=replay_after, session_id=session_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/sessions/{session_id}/runs/{run_id}/events")
    async def stream_run_events(
        session_id: str,
        run_id: str,
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = 0 if from_ == "beginning" else after
        return StreamingResponse(
            _sse_stream(events, after=replay_after, session_id=session_id, run_id=run_id),
            media_type="text/event-stream",
        )

    return app


async def _sse_stream(
    event_bus: InMemoryEventBus,
    *,
    after: int = 0,
    session_id: str | None = None,
    run_id: str | None = None,
) -> AsyncIterator[str]:
    async for event in event_bus.subscribe(after=after, session_id=session_id, run_id=run_id):
        yield _format_sse(event)


def _format_sse(event: Event) -> str:
    payload = json.dumps(jsonable_encoder(event), separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {payload}\n\n"


app = create_app()
