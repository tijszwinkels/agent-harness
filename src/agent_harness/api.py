from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from agent_harness.backends import BackendRegistry, default_backend_registry
from agent_harness.events import InMemoryEventBus
from agent_harness.models import CreateRunRequest, CreateRunResponse, CreateSessionRequest, Event, StopReason
from agent_harness.observer import ExternalTranscriptObserver, TranscriptWatchService
from agent_harness.orchestrator import (
    BackendCommandBuilder,
    CommandBuildError,
    RunManager,
    RunProcessResult,
    default_command_builders,
    validate_session_resume_target,
)
from agent_harness.repository import InMemoryRepository, RunNotFoundError, SessionNotFoundError
from agent_harness.settings import ObserverSettings

logger = logging.getLogger(__name__)

OBSERVER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class WatchService(Protocol):
    async def watch_forever(self, *, stop_event: object | None = None) -> None:
        pass


WatchServiceFactory = Callable[..., WatchService]
TaskFactory = Callable[[Awaitable[None]], asyncio.Task[None]]


def create_app(
    *,
    repository: InMemoryRepository | None = None,
    event_bus: InMemoryEventBus | None = None,
    backend_registry: BackendRegistry | None = None,
    observer_settings: ObserverSettings | None = None,
    watch_service_factory: WatchServiceFactory | None = None,
    task_factory: TaskFactory | None = None,
    run_manager: RunManager | None = None,
    command_builders: Mapping[str, BackendCommandBuilder] | None = None,
) -> FastAPI:
    repo = repository or InMemoryRepository()
    events = event_bus or InMemoryEventBus()
    backends = backend_registry or default_backend_registry()
    settings = observer_settings or ObserverSettings()
    builders = command_builders or default_command_builders()

    app = FastAPI(
        title="agent-harness",
        version="0.1.0",
        lifespan=_lifespan(
            repository=repo,
            event_bus=events,
            observer_settings=settings,
            watch_service_factory=watch_service_factory,
            task_factory=task_factory,
        ),
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/health")
    async def versioned_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/backends")
    async def list_backends() -> dict[str, object]:
        return {"data": backends.list()}

    @app.get("/v1/backends/{backend_name}/models")
    async def list_backend_models(backend_name: str) -> dict[str, object]:
        try:
            return {"data": backends.models(backend_name)}
        except KeyError as exc:
            logger.warning("Model list failed for unknown backend: %s", backend_name)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backend not found") from exc

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
        preflight_session = None
        if run_manager is not None:
            try:
                preflight_session = repo.get_session(session_id)
                builders[preflight_session.backend]
                validate_session_resume_target(preflight_session)
            except SessionNotFoundError as exc:
                logger.warning("Run create failed because session was not found: %s", session_id)
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
            except KeyError as exc:
                logger.warning("No command builder configured for backend: %s", preflight_session.backend)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Backend cannot launch runs") from exc
            except CommandBuildError as exc:
                logger.warning("Run command build failed before run create: session=%s error=%s", session_id, exc)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        try:
            run = repo.create_run(session_id, request)
        except SessionNotFoundError as exc:
            logger.warning("Run create failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        if run_manager is not None:
            try:
                session = preflight_session or repo.get_session(session_id)
                message = _input_message_for_run(repo.list_messages(session_id), run.input_message_id)
                builder = builders[session.backend]
                # repo.create_run already inserted ``run``; if it's the only
                # row, this is the session's first run and the builder may
                # need a creation flag instead of a resume flag.
                is_first_run = len(repo.list_runs(session_id)) <= 1
                run_manager.start(
                    session=session,
                    run=run,
                    command=builder.build(
                        session=session,
                        run=run,
                        message=message,
                        is_first_run=is_first_run,
                    ),
                )
                _schedule_run_result_materialization(run_manager, repo, session_id=session_id, run_id=run.id)
            except KeyError as exc:
                logger.warning("No command builder configured for backend: %s", session.backend)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Backend cannot launch runs") from exc
            except CommandBuildError as exc:
                logger.warning("Run command build failed: session=%s run=%s error=%s", session_id, run.id, exc)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except ValueError as exc:
                logger.warning("Run command build failed: session=%s run=%s error=%s", session_id, run.id, exc)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        else:
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


def _lifespan(
    *,
    repository: InMemoryRepository,
    event_bus: InMemoryEventBus,
    observer_settings: ObserverSettings,
    watch_service_factory: WatchServiceFactory | None,
    task_factory: TaskFactory | None,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        observer_task: asyncio.Task[None] | None = None
        stop_event: asyncio.Event | None = None

        if observer_settings.enabled:
            try:
                observer_settings.validate()
                observer = ExternalTranscriptObserver(event_bus, repository=repository)
                service = _create_watch_service(
                    observer_settings.roots,
                    observer,
                    watch_service_factory=watch_service_factory,
                )
                stop_event = asyncio.Event()
                create_task = task_factory or asyncio.create_task
                observer_task = create_task(service.watch_forever(stop_event=stop_event))
                observer_task.add_done_callback(_log_observer_task_result)
                app.state.transcript_watch_service = service
                app.state.transcript_watch_task = observer_task

                await asyncio.sleep(0)
                if observer_task.done():
                    observer_task.result()
            except Exception:
                logger.exception("Failed to start transcript observer service")
                if observer_task is not None and not observer_task.done():
                    observer_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await observer_task
                raise

        try:
            yield
        finally:
            if observer_task is not None:
                await _stop_observer_task(observer_task, stop_event)

    return lifespan


def _create_watch_service(
    roots: tuple[Path, ...],
    observer: ExternalTranscriptObserver,
    *,
    watch_service_factory: WatchServiceFactory | None,
) -> WatchService:
    if watch_service_factory is not None:
        return watch_service_factory(roots=roots, observer=observer)
    return TranscriptWatchService(roots=roots, observer=observer)


def _input_message_for_run(messages, input_message_id: str | None):
    for message in messages:
        if message.id == input_message_id:
            return message
    raise ValueError("Run input message was not found")


def _schedule_run_result_materialization(
    run_manager: object,
    repository: object,
    *,
    session_id: str,
    run_id: str,
) -> None:
    wait = getattr(run_manager, "wait", None)
    finish_run = getattr(repository, "finish_run", None)
    if not callable(wait) or not callable(finish_run):
        return

    asyncio.create_task(
        _materialize_run_result(
            wait,
            finish_run,
            session_id=session_id,
            run_id=run_id,
        )
    )


async def _materialize_run_result(
    wait,
    finish_run,
    *,
    session_id: str,
    run_id: str,
) -> None:
    try:
        result: RunProcessResult = await wait(run_id)
        stop_reason: StopReason | None = "interrupted" if result.status == "interrupted" else None
        if result.status == "completed":
            stop_reason = "end_turn"
        finish_run(session_id, run_id, status=result.status, stop_reason=stop_reason)
    except Exception:
        logger.exception("Failed to materialize run result: session=%s run=%s", session_id, run_id)


async def _stop_observer_task(
    observer_task: asyncio.Task[None],
    stop_event: asyncio.Event | None,
) -> None:
    observer_task.remove_done_callback(_log_observer_task_result)
    if stop_event is not None:
        stop_event.set()
    try:
        await asyncio.wait_for(observer_task, timeout=OBSERVER_SHUTDOWN_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error(
            "Transcript observer service did not stop within %.1f seconds; cancelling task",
            OBSERVER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        observer_task.cancel()
        with suppress(asyncio.CancelledError):
            await observer_task
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Transcript observer service stopped with an error during shutdown")
        raise


def _log_observer_task_result(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        logger.debug("Transcript observer service task was cancelled")
        return
    try:
        task.result()
    except Exception as exc:
        logger.error(
            "Transcript observer service task failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.warning("Transcript observer service task exited")


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
