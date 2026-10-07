from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from agent_harness.backends import BackendRegistry, default_backend_registry
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import (
    LiveStateRequest,
    LiveStateResponse,
    CreateRunRequest,
    CreateRunResponse,
    CreateSessionRequest,
    Event,
    ForkSessionRequest,
    ForkSessionResponse,
    InterruptRunResponse,
    PatchSessionRequest,
    StopReason,
)
from agent_harness.observer import ExternalTranscriptObserver, TranscriptWatchService
from agent_harness.orchestrator import (
    BackendCommandBuilder,
    CommandBuildError,
    RunManager,
    RunProcessResult,
    claude_conversation_exists,
    default_command_builders,
    session_supports_fork,
    validate_fork_source,
    validate_session_resume_target,
)
from agent_harness.live_state import LiveStateRejected, scan_transcript_owner, transcript_size
from agent_harness.repository import InMemoryRepository, RunNotFoundError, SessionNotFoundError
from agent_harness.settings import ObserverSettings

logger = logging.getLogger(__name__)

OBSERVER_SHUTDOWN_TIMEOUT_SECONDS = 5.0

# Idle interval after which `_sse_stream` injects an SSE comment (`:ka\n\n`)
# so the wire never goes silent for too long. Without keepalives, a client
# with a stale "future" cursor sees no events at all after a harness restart
# (the in-memory bus filters them out by sequence), and httpx waits forever.
# A 15s cadence pairs with a 45s client read timeout to recover from such
# silent-stuck streams within ~one minute.
SSE_KEEPALIVE_SECONDS = 15.0


class WatchService(Protocol):
    async def watch_forever(self, *, stop_event: object | None = None) -> None:
        pass


WatchServiceFactory = Callable[..., WatchService]
TaskFactory = Callable[[Awaitable[None]], asyncio.Task[None]]


def create_app(
    *,
    repository: InMemoryRepository | None = None,
    event_bus: InMemoryEventBus | DurableEventBus | None = None,
    backend_registry: BackendRegistry | None = None,
    observer_settings: ObserverSettings | None = None,
    watch_service_factory: WatchServiceFactory | None = None,
    task_factory: TaskFactory | None = None,
    run_manager: RunManager | None = None,
    command_builders: Mapping[str, BackendCommandBuilder] | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    repo = repository or InMemoryRepository()
    events = event_bus or _event_bus_for_repository(repo)
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
            run_manager=run_manager,
        ),
    )

    if cors_origins:
        # Opt-in cross-origin browser access (e.g. a dataverse page embedding
        # a chat widget). Off by default: no flag, no CORS headers. Pure ASGI
        # middleware, so headers land on the SSE StreamingResponse too. No
        # allow_credentials: browser clients send no cookies or auth headers.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
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

    @app.put("/v1/sessions/{session_id}/live-state")
    async def put_live_state(session_id: str, request: LiveStateRequest) -> LiveStateResponse:
        """Live busy/idle claim for an external pi session from the process
        that drives it (see ``agent_harness.live_state``). Metadata only."""
        apply_live_state = getattr(repo, "apply_live_state", None)
        if not callable(apply_live_state):
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="live state unsupported")
        try:
            current_session = repo.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        before = current_session.status
        transcript = current_session.pi_transcript_path

        # Ownership: learn the session's owner from its own transcript if the
        # observer has not seen the owner entry in this process (e.g. after a
        # harness restart). Bounded and rate-limited; reads only the path the
        # harness recorded for this session.
        if repo.live_state.should_scan_owner(session_id, repo.clock()):
            owner = await asyncio.to_thread(scan_transcript_owner, transcript)
            if owner is not None:
                repo.note_live_state_owner(session_id, owner)

        # Settled boundary: everything already in the transcript belongs to
        # the turn that just ended (see agent_harness.live_state).
        settled_offset = None
        if request.state == "idle":
            settled_offset = await asyncio.to_thread(transcript_size, transcript)

        try:
            session, claim = apply_live_state(session_id, request, settled_offset=settled_offset)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        except LiveStateRejected as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        if claim is not None and session.status != before:
            # Same shape as observer status flips, so SSE consumers update.
            await events.publish(
                Event(
                    event="session.updated",
                    session_id=session.id,
                    data={"session": session.model_dump(mode="json"), "live_state": claim.state},
                )
            )
        current = repo.live_state.get(session_id)
        busy = current is not None and current.busy_at(repo.clock())
        return LiveStateResponse(
            accepted=claim is not None,
            status=session.status,
            busy=busy,
            expires_at=current.expires_at if busy else None,
        )

    @app.patch("/v1/sessions/{session_id}")
    async def patch_session(session_id: str, request: PatchSessionRequest) -> object:
        # ``exclude_unset`` so callers can patch a single field without
        # having to round-trip every other value — and so an absent field
        # stays at its current repo value. Null effort clears the override;
        # title retains its existing non-null patch contract.
        fields = request.model_dump(exclude_unset=True)
        if any(value is None and name != "effort" for name, value in fields.items()):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="patch field cannot be null; omit the field to leave it unchanged",
            )
        # A client-supplied title is explicit: it stops tracking the
        # backend's native conversation name (see ``Session.title_source``).
        repo_fields = {**fields, "title_source": None} if "title" in fields else fields
        try:
            session = repo.patch_session(session_id, repo_fields)
        except SessionNotFoundError as exc:
            logger.warning("Session patch failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        if fields:
            await events.publish(
                Event(
                    event="session.updated",
                    session_id=session.id,
                    data={"fields_changed": sorted([*fields.keys(), "updated_at"])},
                )
            )
        return session

    async def _create_and_launch_run(
        session_id: str, request: CreateRunRequest
    ) -> CreateRunResponse:
        """Shared run-launch body for ``POST .../runs`` and the fork route.

        Preflights the session, inserts the run, builds the backend command,
        and submits it to the RunManager (or the no-manager test path).
        Raises ``HTTPException`` on the documented failure statuses. The fork
        route relies on the command builder emitting the fork argv for a
        first run whose session carries ``forked_from`` — no special-casing
        needed here.
        """
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

        run_status = run.status
        if run_manager is not None:
            try:
                session = preflight_session or repo.get_session(session_id)
                message = _input_message_for_run(repo.list_messages(session_id), run.input_message_id)
                builder = builders[session.backend]
                # repo.create_run already inserted ``run``; if it's the only
                # row, this is the session's first run and the builder may
                # need a creation flag instead of a resume flag.
                is_first_run = len(repo.list_runs(session_id)) <= 1
                if session.backend == "claude-code":
                    # Prefer the on-disk signal: resume only once claude has
                    # actually created the conversation. A create that failed
                    # before writing the transcript (e.g. arg-parse error on a
                    # ``-``-prefixed prompt) must be retried as a create, not
                    # locked into --resume against a session that never existed.
                    is_first_run = not claude_conversation_exists(session)
                command = builder.build(
                    session=session,
                    run=run,
                    message=message,
                    is_first_run=is_first_run,
                )
            except KeyError as exc:
                logger.warning("No command builder configured for backend: %s", session.backend)
                # The run is in the repo but cannot be launched — leaving it
                # ``queued`` forever would deadlock the session. Flip it to
                # ``failed`` so the user-visible state matches what happened.
                repo.finish_run(session_id, run.id, status="failed")
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Backend cannot launch runs") from exc
            except CommandBuildError as exc:
                logger.warning("Run command build failed: session=%s run=%s error=%s", session_id, run.id, exc)
                repo.finish_run(session_id, run.id, status="failed")
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except ValueError as exc:
                logger.warning("Run command build failed: session=%s run=%s error=%s", session_id, run.id, exc)
                repo.finish_run(session_id, run.id, status="failed")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            # ``on_start`` is invoked by the RunManager at the exact moment
            # the subprocess actually spawns — that may be right now (no
            # active run for this session) or arbitrarily later (after the
            # current run for this session finishes draining the queue).
            # Coupling repo.start_run + materialization scheduling to that
            # moment keeps the repo's lifecycle aligned with the orchestrator.
            #
            # The two operations are isolated: if ``start_run`` raises
            # (e.g. the run was already finalized via a concurrent path),
            # the materializer must still be scheduled — otherwise the run
            # would stay stuck at ``queued`` in the repo forever even after
            # the subprocess completes.
            def on_start(_session_id: str = session_id, _run_id: str = run.id) -> None:
                try:
                    repo.start_run(_session_id, _run_id)
                except Exception:
                    logger.exception(
                        "on_start: repo.start_run raised — proceeding with materializer scheduling: session=%s run=%s",
                        _session_id, _run_id,
                    )
                _schedule_run_result_materialization(
                    run_manager, repo, session_id=_session_id, run_id=_run_id,
                )

            result = run_manager.submit(
                session=session,
                run=run,
                command=command,
                on_start=on_start,
            )
            if not result.accepted:
                logger.warning(
                    "Run rejected: session=%s run=%s reason=%s", session_id, run.id, result.reason,
                )
                repo.finish_run(session_id, run.id, status="failed")
                if result.reason == "queue_full":
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Per-session run queue is full; retry after the active run completes.",
                    )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Run rejected: {result.reason}",
                )
            run_status = result.status or run.status
        else:
            await events.publish(Event(event="run.started", session_id=session_id, run_id=run.id, data={}))
            # The no-run_manager path is used by tests that just want repo
            # bookkeeping without a real subprocess; mark the run "running"
            # to mirror pre-queue semantics.
            repo.start_run(session_id, run.id)
            run_status = "running"
        return CreateRunResponse(session_id=session_id, run_id=run.id, status=run_status)

    @app.post("/v1/sessions/{session_id}/runs", status_code=status.HTTP_202_ACCEPTED)
    async def create_run(session_id: str, request: CreateRunRequest) -> CreateRunResponse:
        return await _create_and_launch_run(session_id, request)

    @app.post("/v1/sessions/{session_id}/forks", status_code=status.HTTP_201_CREATED)
    async def fork_session(
        session_id: str, request: ForkSessionRequest
    ) -> ForkSessionResponse:
        # A fork is a new harness-owned child session whose first run resumes
        # the PARENT's whole conversation but writes to the child's own
        # transcript, leaving the parent untouched (claude --fork-session /
        # pi --fork; verified against the real CLIs). The child then diverges.
        try:
            parent = repo.get_session(session_id)
        except SessionNotFoundError as exc:
            logger.warning("Fork failed because parent session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

        # Backend/id must be forkable (codex and non-UUID ids are rejected).
        try:
            validate_fork_source(parent)
        except CommandBuildError as exc:
            logger.warning("Fork rejected for session=%s: %s", session_id, exc)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        # Forking a mid-write transcript would give an ill-defined fork point:
        # reject while any run for the parent is still queued or running.
        if any(run.status in ("queued", "running") for run in repo.list_runs(session_id)):
            logger.info("Fork rejected: parent %s has an in-progress run", session_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot fork a session with an in-progress run",
            )

        child = repo.create_forked_session(parent, title=request.title)
        await events.publish(
            Event(
                event="session.updated",
                session_id=child.id,
                data={"session": jsonable_encoder(child)},
            )
        )

        run = None
        if request.message is not None:
            # Reuse the run-launch path; the builder emits the fork argv
            # because the child is a first-run session carrying forked_from.
            launched = await _create_and_launch_run(
                child.id, CreateRunRequest(message=request.message)
            )
            run = repo.get_run(child.id, launched.run_id)

        return ForkSessionResponse(session=child, run=run)

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
    async def interrupt_run(session_id: str, run_id: str) -> InterruptRunResponse:
        # Confirm the target exists up-front so we never empty the queue for
        # a 404 request.
        try:
            repo.get_run(session_id, run_id)
        except SessionNotFoundError as exc:
            logger.warning("Run interrupt failed because session was not found: %s", session_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        except RunNotFoundError as exc:
            logger.warning("Run interrupt failed: session=%s run=%s", session_id, run_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from exc

        # Drop the orchestrator queue first so the in-flight run's
        # ``_run_and_forget`` finally-block doesn't immediately spawn a
        # follow-up that's about to be cancelled in the repo. ``run_manager``
        # is the source of truth for "what's queued to spawn"; the repo only
        # records the resulting status flips.
        if run_manager is not None:
            run_manager.drop_queued(session_id)
        dropped_in_repo = repo.drop_queued_runs(session_id)

        # Terminate the subprocess if the target is currently running. Safe
        # no-op when the target was queued (already handled by drop above).
        if run_manager is not None:
            await run_manager.interrupt(session_id, run_id)

        # Mark the target as interrupted. Idempotent against drop_queued_runs
        # for queued targets — the second flip is a no-op state-wise.
        try:
            target = repo.interrupt_run(session_id, run_id)
        except RunNotFoundError as exc:  # pragma: no cover - shouldn't happen
            logger.warning("Run vanished between preflight and interrupt: %s", run_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from exc

        other_dropped = [r for r in dropped_in_repo if r.id != run_id]
        await events.publish(
            Event(
                event="run.interrupted",
                session_id=session_id,
                run_id=run_id,
                data={"dropped_queued_run_ids": [r.id for r in other_dropped]},
            )
        )
        return InterruptRunResponse(run=target, dropped_queued=other_dropped)

    @app.get("/v1/events/max-sequence")
    async def events_max_sequence(
        session_id: str | None = Query(default=None),
    ) -> dict[str, int]:
        # Cheap synchronous probe for clients that need to know the current
        # max event sequence without opening an SSE stream (e.g. mm-bridge
        # uses it after reconnect to distinguish "I'm caught up" from "the
        # harness restarted"). Returns 0 on an empty bus or unknown session.
        return {"sequence": await events.max_sequence(session_id=session_id)}

    @app.get("/v1/events")
    async def stream_events(
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = await _replay_after(events, after=after, from_=from_)
        return StreamingResponse(_sse_stream(events, after=replay_after), media_type="text/event-stream")

    @app.get("/v1/sessions/{session_id}/events")
    async def stream_session_events(
        session_id: str,
        after: int = Query(default=0, ge=0),
        from_: str = Query(default="now", alias="from"),
    ) -> StreamingResponse:
        replay_after = await _replay_after(events, after=after, from_=from_, session_id=session_id)
        return StreamingResponse(
            _sse_stream(events, after=replay_after, session_id=session_id),
            media_type="text/event-stream",
        )

    return app


def _lifespan(
    *,
    repository: InMemoryRepository,
    event_bus: InMemoryEventBus | DurableEventBus,
    observer_settings: ObserverSettings,
    watch_service_factory: WatchServiceFactory | None,
    task_factory: TaskFactory | None,
    run_manager: RunManager | None,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        observer_task: asyncio.Task[None] | None = None
        stop_event: asyncio.Event | None = None

        if observer_settings.enabled:
            try:
                observer_settings.validate()
                observer = ExternalTranscriptObserver(
                    event_bus,
                    repository=repository,
                    codex_name_index=observer_settings.codex_name_index_path(),
                )
                # Hand the live observer to the RunManager so
                # RunProcess can call ``bind_rollout`` (claude pre-bind)
                # and ``expect_codex_rollout`` (codex pre-bind). Both
                # paths are unconditional for harness-origin sessions
                # — pre-bind is always on as of Phase 2.
                if run_manager is not None:
                    setter = getattr(run_manager, "set_observer", None)
                    if callable(setter):
                        setter(observer)
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


def _event_bus_for_repository(repository: InMemoryRepository) -> InMemoryEventBus | DurableEventBus:
    append_event = getattr(repository, "append_event", None)
    list_events = getattr(repository, "list_events", None)
    if callable(append_event) and callable(list_events):
        return DurableEventBus(repository)
    return InMemoryEventBus()


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
    event_bus: InMemoryEventBus | DurableEventBus,
    *,
    after: int = 0,
    session_id: str | None = None,
    keepalive_seconds: float = SSE_KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    # ``subscribe`` yields ``None`` after each silent ``keepalive_seconds``
    # window; surface those as SSE comment frames so clients never see the
    # wire go quiet for longer than the interval. Real events round-trip
    # unchanged.
    async for event in event_bus.subscribe(
        after=after,
        session_id=session_id,
        keepalive_seconds=keepalive_seconds,
    ):
        if event is None:
            yield ": ka\n\n"
            continue
        yield _format_sse(event)


async def _replay_after(
    event_bus: InMemoryEventBus | DurableEventBus,
    *,
    after: int,
    from_: str,
    session_id: str | None = None,
) -> int:
    if after > 0:
        return after
    if from_ == "beginning":
        return 0
    return await event_bus.max_sequence(session_id=session_id)


def _format_sse(event: Event) -> str:
    payload = json.dumps(jsonable_encoder(event), separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {payload}\n\n"


app = create_app()
