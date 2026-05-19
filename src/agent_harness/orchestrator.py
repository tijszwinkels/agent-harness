from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pathlib import Path

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock, utc_now
from agent_harness.rollout_discovery import RolloutDiscovery

logger = logging.getLogger(__name__)

END_TURN_GRACE_SECONDS = 20.0
END_TURN_HARD_KILL_AFTER_SECONDS = 40.0
IDLE_TIMEOUT_SECONDS = 30 * 60
IDLE_HARD_KILL_GRACE_SECONDS = 30.0
IDLE_CHECK_INTERVAL_SECONDS = 60.0

_END_TURN_EVENT = "__end_turn__"
# Phase 2: ``message.delta`` is no longer emitted from the supervisor
# (the observer is the sole writer for message-shaped events). We keep
# ``message`` and ``tool_use`` for forward-compatibility (Phase 3 will
# rewire the watchdog to subscribe to observer events directly) and add
# ``process.stderr`` so long stderr-active codex runs don't trip the
# 30-min idle watchdog spuriously while the orchestrator-side activity
# stream is otherwise sparse.
_ACTIVITY_EVENTS = frozenset({
    "message",
    "tool_use",
    "process.stderr",
})


class CommandBuildError(ValueError):
    """Raised when a session cannot be mapped to a runnable backend command."""


@dataclass(frozen=True)
class ProcessCommand:
    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("ProcessCommand requires at least one argv item")


class BackendCommandBuilder(Protocol):
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        """Build a non-interactive CLI command for a harness-owned run.

        ``is_first_run`` is ``True`` when no prior run exists for this
        session. Builders use it to pick between session-creation and
        session-resume flags (e.g. ``claude --session-id`` vs
        ``claude --resume``).
        """


class CodexCommandBuilder:
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        del run, is_first_run
        text = _message_text(message)
        bypass: tuple[str, ...] = (
            ("--dangerously-bypass-approvals-and-sandbox",)
            if session.bypass_permissions
            else ()
        )
        if session.origin == "external":
            return ProcessCommand(
                argv=(
                    "codex",
                    "exec",
                    "resume",
                    "--json",
                    "--model",
                    session.model,
                    *bypass,
                    _external_resume_id(session, prefix="codex_"),
                    text,
                ),
                cwd=session.project.path,
            )

        return ProcessCommand(
            argv=("codex", "exec", "--json", "--model", session.model, *bypass, text),
            cwd=session.project.path,
        )


def _harness_session_id_as_uuid(session_id: str) -> str:
    # Accepts any of the session-id shapes the harness or observer can
    # produce for a claude session and returns the canonical
    # 8-4-4-4-12 UUID string (claude --session-id / --resume validates UUID
    # format).
    #
    #   * ``ses_<32hex>``               — canonical harness + external form
    #   * ``claude_<uuid-with-dashes>`` — legacy external-observer form,
    #                                     kept for records persisted before
    #                                     the canonicalization.
    #   * bare 32-hex or dashed UUID    — best-effort fallback.
    if session_id.startswith("ses_"):
        body = session_id.removeprefix("ses_")
    elif session_id.startswith("claude_"):
        body = session_id.removeprefix("claude_")
    else:
        body = session_id
    hex_part = body.replace("-", "").lower()
    if len(hex_part) != 32 or not all(c in "0123456789abcdef" for c in hex_part):
        raise CommandBuildError(
            f"Cannot derive claude session UUID from harness session id {session_id!r}",
        )
    return f"{hex_part[0:8]}-{hex_part[8:12]}-{hex_part[12:16]}-{hex_part[16:20]}-{hex_part[20:32]}"


class ClaudeCodeCommandBuilder:
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        del run
        text = _message_text(message)
        argv = (
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            session.model,
        )
        if session.bypass_permissions:
            argv = (*argv, "--dangerously-skip-permissions")
        claude_uuid = _harness_session_id_as_uuid(session.id)
        if session.origin == "external":
            # External claude sessions are already running under this UUID
            # (claude wrote the .jsonl with it). We can only --resume — we
            # don't own session creation.
            argv = (*argv, "--resume", claude_uuid)
        else:
            # Harness-origin: pin a deterministic claude session UUID so
            # subsequent runs can --resume and retain conversation context.
            # claude --session-id creates the session on first use and
            # errors on duplicate, so we switch to --resume for follow-up.
            flag = "--session-id" if is_first_run else "--resume"
            argv = (*argv, flag, claude_uuid)

        return ProcessCommand(
            argv=(*argv, text),
            cwd=session.project.path,
        )


def default_command_builders() -> dict[str, BackendCommandBuilder]:
    return {
        "claude-code": ClaudeCodeCommandBuilder(),
        "codex": CodexCommandBuilder(),
    }


def _detect_end_turn_in_line(line: str, *, backend: str) -> bool:
    """Detect the end-of-turn signal in a single stdout line.

    Phase 2 collapses the per-backend stdout parsers into this single
    boolean helper — the only thing the supervisor still cares about
    in stdout is whether the watchdog's post-end_turn cleanup should
    fire. Message data flows through the observer.

    Recognized shapes:
    - claude-code: ``{"type":"result","stop_reason":"end_turn"}``
    - codex: ``{"type":"turn.completed"}``
    """
    if not line:
        return False
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(record, Mapping):
        return False
    if backend == "claude-code":
        return (
            record.get("type") == "result"
            and record.get("stop_reason") == "end_turn"
        )
    if backend == "codex":
        return record.get("type") == "turn.completed"
    return False


def validate_session_resume_target(session: Session) -> None:
    if session.origin != "external":
        return
    if session.backend == "codex":
        _external_resume_id(session, prefix="codex_")
        return
    if session.backend == "claude-code":
        # Validates session.id resolves to a claude UUID under any of the
        # accepted shapes (ses_<hex>, legacy claude_<uuid>, raw UUID).
        _harness_session_id_as_uuid(session.id)
        return
    raise CommandBuildError(f"Cannot resume external session for backend {session.backend}")


class AsyncLineReader(Protocol):
    async def readline(self) -> bytes:
        pass


class ManagedProcess(Protocol):
    stdout: AsyncLineReader | None
    stderr: AsyncLineReader | None
    returncode: int | None
    pid: int

    async def wait(self) -> int:
        pass

    def terminate(self) -> None:
        pass


class ProcessFactory(Protocol):
    async def __call__(self, command: ProcessCommand) -> ManagedProcess:
        pass


class AsyncioProcessFactory:
    async def __call__(self, command: ProcessCommand) -> ManagedProcess:
        env = None if not command.env else os.environ | dict(command.env)
        return await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=command.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )


@dataclass(frozen=True)
class RunProcessResult:
    run_id: str
    status: RunStatus
    returncode: int | None = None
    error: str | None = None


class RunProcess:
    def __init__(
        self,
        *,
        session: Session,
        run: Run,
        command: ProcessCommand,
        event_bus: InMemoryEventBus,
        process_factory: ProcessFactory | None = None,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        observer: Any | None = None,
        rollout_discovery: Any | None = None,
    ) -> None:
        self.session = session
        self.run_record = run
        self.command = command
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._process: ManagedProcess | None = None
        self._interrupted = False
        self._clock = clock
        self._sleep = sleep
        self._end_turn_event = asyncio.Event()
        self._last_activity_at = self._clock()
        self._last_activity_event: str | None = None
        self._watchdog_termination_in_progress = False
        self._watchdog_termination_task: asyncio.Task[None] | None = None
        # Phase 2: rollout pre-binding is unconditional for harness
        # sessions. claude uses the deterministic ``discover_claude``
        # path; codex uses the expectation registry (no fd probe).
        # ``observer`` and ``rollout_discovery`` may be ``None`` for
        # tests that don't exercise the pre-bind path; production wires
        # both via api.py's lifespan + cli.py.
        self._observer = observer
        self._rollout_discovery = rollout_discovery
        # Set to the bound rollout path once claude pre-bind succeeds,
        # so ``run()``'s finally block can unbind on terminal-state
        # cleanup. Codex uses the expectation registry instead; no
        # bound path to clean up there (the registry self-purges via
        # TTL + consume-on-match).
        self._bound_rollout_path: Path | None = None

    async def run(self) -> RunProcessResult:
        await self._publish("run.started", {})

        # Phase 2: codex pre-bind registers an expectation BEFORE the
        # spawn so a fast watchfiles fire on codex's first byte can
        # still resolve correctly. Claude pre-bind runs post-spawn
        # (deterministic; no race against the file existing).
        self._pre_register_codex_expectation_if_codex()

        try:
            self._process = await self._process_factory(self.command)
        except Exception as exc:
            logger.exception("Failed to start run process: session=%s run=%s", self.session.id, self.run_record.id)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))

        # Claude pre-bind (deterministic, post-spawn).
        self._pre_bind_claude_if_claude()

        stream_tasks = self._stream_tasks(self._process)
        wait_task = asyncio.create_task(self._process.wait())
        watchdog_tasks = [
            asyncio.create_task(self._watch_end_turn_cleanup(wait_task)),
            asyncio.create_task(self._watch_idle_timeout(wait_task)),
        ]
        try:
            returncode = await wait_task
            await self._finish_streams(stream_tasks)
        except Exception as exc:
            logger.exception("Run process failed while active: session=%s run=%s", self.session.id, self.run_record.id)
            await self._cancel_streams(stream_tasks)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))
        finally:
            await self._finish_watchdogs(watchdog_tasks)
            # Drop the rollout binding now that the subprocess is done —
            # keeps observer._path_to_session bounded across the harness
            # lifetime (one entry per *active* harness run, not per all
            # runs ever).
            self._unbind_rollout_if_bound()

        if self._interrupted:
            await self._publish("run.interrupted", {"returncode": returncode})
            return RunProcessResult(run_id=self.run_record.id, status="interrupted", returncode=returncode)

        if returncode == 0:
            await self._publish("run.completed", {"returncode": returncode})
            return RunProcessResult(run_id=self.run_record.id, status="completed", returncode=returncode)

        logger.warning(
            "Run process exited with non-zero status: session=%s run=%s returncode=%s",
            self.session.id,
            self.run_record.id,
            returncode,
        )
        await self._publish("run.failed", {"returncode": returncode})
        return RunProcessResult(run_id=self.run_record.id, status="failed", returncode=returncode)

    async def interrupt(self) -> bool:
        if self.run_record.origin != "harness":
            return False
        if self._process is None:
            return False

        self._interrupted = True
        self._process.terminate()
        return True

    def _pre_bind_enabled(self) -> bool:
        if self.session.origin != "harness":
            return False
        if self._observer is None:
            return False
        return True

    def _pre_register_codex_expectation_if_codex(self) -> None:
        """Phase 2 codex hand-off. Replaces Phase 1's psutil-based fd
        probe with a one-line expectation registration the observer
        will match content-side via ``session_meta`` peek. No spawn-
        time blocking, no race with watchfiles."""
        if self.session.backend != "codex":
            return
        if not self._pre_bind_enabled():
            return
        cwd_str = self.command.cwd
        if cwd_str is None:
            logger.warning(
                "Skipping codex expectation registration: command has no cwd; session=%s",
                self.session.id,
            )
            return
        try:
            self._observer.expect_codex_rollout(
                cwd=Path(cwd_str), session_id=self.session.id
            )
        except Exception:
            logger.exception(
                "Codex expectation registration failed: session=%s",
                self.session.id,
            )
            return
        logger.debug(
            "Codex rollout expectation registered: session=%s cwd=%s",
            self.session.id,
            cwd_str,
        )

    def _pre_bind_claude_if_claude(self) -> None:
        if self.session.backend != "claude-code":
            return
        if not self._pre_bind_enabled():
            return
        if self._rollout_discovery is None:
            return
        cwd_str = self.command.cwd
        if cwd_str is None:
            logger.warning(
                "Skipping claude rollout pre-bind: command has no cwd; session=%s",
                self.session.id,
            )
            return
        # Claude writes the rollout under the dashed-UUID stem it was
        # invoked with (orchestrator builds ``--session-id <uuid>``).
        # Bind under that exact filename — using the harness's
        # ``ses_<hex>`` form here would silently miss the file the
        # observer actually sees on disk.
        try:
            claude_uuid = _harness_session_id_as_uuid(self.session.id)
        except CommandBuildError as exc:
            logger.warning(
                "Skipping claude rollout pre-bind: cannot derive UUID; session=%s error=%s",
                self.session.id,
                exc,
            )
            return
        try:
            path = self._rollout_discovery.discover_claude(
                session_id=claude_uuid, cwd=Path(cwd_str)
            )
            self._observer.bind_rollout(path, self.session.id)
        except Exception:
            logger.exception(
                "Claude rollout pre-bind raised: session=%s",
                self.session.id,
            )
            return
        self._bound_rollout_path = path
        logger.debug(
            "Claude rollout pre-bound: session=%s path=%s",
            self.session.id,
            path,
        )

    def _unbind_rollout_if_bound(self) -> None:
        path = self._bound_rollout_path
        if path is None or self._observer is None:
            return
        try:
            self._observer.unbind_rollout(path)
        except Exception:
            logger.exception(
                "Failed to unbind rollout: session=%s path=%s",
                self.session.id,
                path,
            )
        finally:
            self._bound_rollout_path = None

    def _stream_tasks(self, process: ManagedProcess) -> list[asyncio.Task[None]]:
        tasks: list[asyncio.Task[None]] = []
        if process.stdout is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stdout", process.stdout)))
        if process.stderr is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stderr", process.stderr)))
        return tasks

    async def _stream_lines(self, stream_name: Literal["stdout", "stderr"], stream: AsyncLineReader) -> None:
        # Phase 2: stdout is end-turn-only; stderr forwards as
        # ``process.stderr`` events. The observer is the sole writer
        # for ``message`` / ``message.delta``; nothing on this path
        # synthesizes messages.
        detect_end_turn = (
            stream_name == "stdout" and self.session.origin == "harness"
        )
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                continue
            if stream_name == "stderr":
                await self._publish("process.stderr", {"text": text})
                continue
            if not detect_end_turn:
                continue
            try:
                if _detect_end_turn_in_line(text, backend=self.session.backend):
                    self._end_turn_event.set()
            except Exception:
                logger.exception(
                    "End-turn detector raised: session=%s run=%s backend=%s",
                    self.session.id,
                    self.run_record.id,
                    self.session.backend,
                )

    async def _finish_streams(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.exception(
                    "Run stream reader failed: session=%s run=%s",
                    self.session.id,
                    self.run_record.id,
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def _cancel_streams(self, tasks: list[asyncio.Task[None]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _finish_watchdogs(self, tasks: list[asyncio.Task[None]]) -> None:
        if self._watchdog_termination_in_progress:
            terminator = self._watchdog_termination_task
            to_cancel = [task for task in tasks if task is not terminator]
            await self._cancel_watchdogs(to_cancel)
            if terminator is not None:
                await asyncio.gather(terminator, return_exceptions=True)
            return
        await self._cancel_watchdogs(tasks)

    async def _cancel_watchdogs(self, tasks: list[asyncio.Task[None]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch_end_turn_cleanup(self, wait_task: asyncio.Task[int]) -> None:
        await self._end_turn_event.wait()
        if self._process_exited(wait_task):
            return

        await self._sleep(END_TURN_GRACE_SECONDS)
        if self._process_exited(wait_task):
            return

        self._watchdog_termination_in_progress = True
        self._watchdog_termination_task = asyncio.current_task()
        self._interrupted = True
        if not self._signal_process_group(signal.SIGTERM):
            return

        hard_kill = False
        remaining = max(0.0, END_TURN_HARD_KILL_AFTER_SECONDS - END_TURN_GRACE_SECONDS)
        if remaining:
            await self._wait_for_process_or_sleep(wait_task, remaining)

        if not self._process_exited(wait_task):
            hard_kill = self._signal_process_group(signal.SIGKILL)
        if hard_kill:
            await wait_task

        await self._publish(
            "run.terminated_after_end_turn",
            {
                "grace_seconds": int(END_TURN_GRACE_SECONDS),
                "hard_kill": hard_kill,
                "returncode": self._returncode(wait_task),
                "reason": "subprocess_did_not_exit_after_end_turn",
            },
        )

    async def _watch_idle_timeout(self, wait_task: asyncio.Task[int]) -> None:
        while True:
            await self._sleep(IDLE_CHECK_INTERVAL_SECONDS)
            if self._process_exited(wait_task):
                return

            idle_seconds = (self._clock() - self._last_activity_at).total_seconds()
            if idle_seconds <= IDLE_TIMEOUT_SECONDS:
                continue

            self._watchdog_termination_in_progress = True
            self._watchdog_termination_task = asyncio.current_task()
            self._interrupted = True
            if not self._signal_process_group(signal.SIGTERM):
                return

            hard_kill = False
            await self._wait_for_process_or_sleep(wait_task, IDLE_HARD_KILL_GRACE_SECONDS)
            if not self._process_exited(wait_task):
                hard_kill = self._signal_process_group(signal.SIGKILL)
                if hard_kill:
                    await wait_task

            await self._publish(
                "run.timed_out_idle",
                {
                    "idle_seconds": int(IDLE_TIMEOUT_SECONDS),
                    "last_activity_event": self._last_activity_event,
                    "last_activity_at": _format_timestamp(self._last_activity_at),
                    "hard_kill": hard_kill,
                    "reason": "no_activity_within_threshold",
                },
            )
            return

    async def _wait_for_process_or_sleep(self, wait_task: asyncio.Task[int], seconds: float) -> None:
        if wait_task.done():
            return
        sleep_task = asyncio.create_task(self._sleep(seconds))
        done, pending = await asyncio.wait({wait_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
        del done
        if sleep_task in pending:
            sleep_task.cancel()
            await asyncio.gather(sleep_task, return_exceptions=True)

    def _process_exited(self, wait_task: asyncio.Task[int]) -> bool:
        process = self._process
        return wait_task.done() or process is None or process.returncode is not None

    def _returncode(self, wait_task: asyncio.Task[int]) -> int | None:
        if wait_task.done() and not wait_task.cancelled():
            try:
                return wait_task.result()
            except Exception:
                logger.exception(
                    "Run process wait task failed while reading returncode: session=%s run=%s",
                    self.session.id,
                    self.run_record.id,
                )
        if self._process is not None:
            return self._process.returncode
        return None

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        process = self._process
        if process is None or process.returncode is not None:
            return False
        try:
            # POSIX-only by design: production harness runs on Linux, and the
            # subprocess is launched with start_new_session=True to isolate a
            # process group for watchdog cleanup.
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            logger.info(
                "Run process group no longer exists before watchdog signal: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False
        except Exception:
            logger.exception(
                "Failed to signal run process group: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False

    async def _publish(self, event: str, data: dict[str, object]) -> Event:
        if event in _ACTIVITY_EVENTS:
            self._last_activity_at = self._clock()
            self._last_activity_event = event
        return await self._event_bus.publish(
            Event(event=event, session_id=self.session.id, run_id=self.run_record.id, data=data)
        )


RUN_QUEUE_MAX_PER_SESSION = 16


@dataclass(frozen=True)
class SubmitResult:
    # ``status`` is "running" or "queued" on accept, None on reject. Reject
    # currently only happens when the per-session queue cap is exceeded —
    # ``reason`` carries the machine-readable code (``"queue_full"``) so
    # callers can map it to a transport-specific failure (api.py → 429).
    accepted: bool
    status: RunStatus | None
    reason: str | None = None


@dataclass(frozen=True)
class _QueuedRun:
    session: Session
    run: Run
    command: ProcessCommand
    on_start: Callable[[], None] | None


class RunManager:
    def __init__(
        self,
        *,
        event_bus: InMemoryEventBus,
        process_factory: ProcessFactory | None = None,
        queue_max_per_session: int = RUN_QUEUE_MAX_PER_SESSION,
        observer: Any | None = None,
        rollout_discovery: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._queue_max_per_session = queue_max_per_session
        self._active: dict[str, RunProcess] = {}
        self._tasks: dict[str, asyncio.Task[RunProcessResult]] = {}
        # Tracks which run_id currently owns the subprocess for each session
        # (at most one). New submits for an already-owned session land on the
        # per-session FIFO in ``_queues`` instead of spawning concurrently —
        # which would race on the shared claude-code rollout JSONL.
        self._active_run_by_session: dict[str, str] = {}
        self._queues: dict[str, deque[_QueuedRun]] = {}
        # Optional wiring for the rollout pre-binding path. The production
        # cli wires only ``rollout_discovery`` here — the observer is
        # constructed later inside FastAPI's lifespan and late-bound via
        # ``set_observer`` (see api.py). Tests usually pass both directly.
        # Pre-bind is unconditional in Phase 2 (no env-var gate); the
        # actual work skips itself when ``observer`` is unset.
        self._observer = observer
        self._rollout_discovery = rollout_discovery

    def set_observer(self, observer: Any) -> None:
        """Late binder used by api.py's lifespan: observer is constructed
        when the FastAPI app starts up, after RunManager already exists.
        Idempotent. Called once at most in normal flow."""
        self._observer = observer

    def submit(
        self,
        *,
        session: Session,
        run: Run,
        command: ProcessCommand,
        on_start: Callable[[], None] | None = None,
    ) -> SubmitResult:
        # Decide whether to spawn the run immediately or queue it. Must be
        # called from the event loop thread (it may schedule asyncio tasks);
        # FastAPI's single-loop model satisfies this.
        if session.id not in self._active_run_by_session:
            self._start_now(session, run, command, on_start)
            return SubmitResult(accepted=True, status="running")

        queue = self._queues.setdefault(session.id, deque())
        if len(queue) >= self._queue_max_per_session:
            return SubmitResult(accepted=False, status=None, reason="queue_full")
        queue.append(_QueuedRun(session=session, run=run, command=command, on_start=on_start))
        return SubmitResult(accepted=True, status="queued")

    def start(self, *, session: Session, run: Run, command: ProcessCommand) -> RunProcess:
        # Back-compat shim for callers and tests that pre-date ``submit``. It
        # raises on rejection — pre-queue callers had no concept of "queue full".
        result = self.submit(session=session, run=run, command=command)
        if not result.accepted:
            raise RuntimeError(f"RunManager.start cannot accept run: {result.reason}")
        # When the submit landed on the queue rather than spawning, there is
        # no RunProcess instance yet. Existing call sites only consult the
        # return value in tests; surface a clear error rather than a None.
        run_process = self._active.get(run.id)
        if run_process is None:
            raise RuntimeError(
                "RunManager.start returned a queued submit; callers needing the "
                "RunProcess object must use submit() and handle status='queued'.",
            )
        return run_process

    def drop_queued(self, session_id: str) -> list[str]:
        # Pop every queued entry for ``session_id`` and return their run ids.
        # The caller is responsible for reflecting the drop in the repository
        # (we deliberately don't reach into the repo from here). Idempotent:
        # returns [] when there's nothing queued.
        queue = self._queues.pop(session_id, None)
        if not queue:
            return []
        return [entry.run.id for entry in queue]

    async def interrupt(self, session_id: str, run_id: str) -> bool:
        run_process = self._active.get(run_id)
        if run_process is None or run_process.session.id != session_id:
            return False
        return await run_process.interrupt()

    async def wait(self, run_id: str) -> RunProcessResult:
        task = self._tasks[run_id]
        return await task

    def _start_now(
        self,
        session: Session,
        run: Run,
        command: ProcessCommand,
        on_start: Callable[[], None] | None,
    ) -> RunProcess:
        # ``on_start`` runs before the subprocess is launched so the caller
        # (typically api.py) can flip the run's repo status to "running" and
        # schedule materialization in lock-step with the actual spawn. We log
        # and swallow callback failures rather than abort the spawn — losing
        # a status update is preferable to leaving the user's prompt dropped.
        if on_start is not None:
            try:
                on_start()
            except Exception:
                logger.exception(
                    "RunManager on_start callback raised: session=%s run=%s",
                    session.id,
                    run.id,
                )

        run_process = RunProcess(
            session=session,
            run=run,
            command=command,
            event_bus=self._event_bus,
            process_factory=self._process_factory,
            observer=self._observer,
            rollout_discovery=self._rollout_discovery,
        )
        self._active[run.id] = run_process
        self._active_run_by_session[session.id] = run.id
        task = asyncio.create_task(self._run_and_forget(run_process))
        self._tasks[run.id] = task
        return run_process

    async def _run_and_forget(self, run_process: RunProcess) -> RunProcessResult:
        try:
            return await run_process.run()
        finally:
            run_id = run_process.run_record.id
            session_id = run_process.session.id
            self._active.pop(run_id, None)
            if self._active_run_by_session.get(session_id) == run_id:
                self._active_run_by_session.pop(session_id, None)
            self._spawn_next_queued(session_id)

    def _spawn_next_queued(self, session_id: str) -> None:
        queue = self._queues.get(session_id)
        if not queue:
            return
        nxt = queue.popleft()
        if not queue:
            self._queues.pop(session_id, None)
        self._start_now(nxt.session, nxt.run, nxt.command, nxt.on_start)


def _message_text(message: Message) -> str:
    text = "\n".join(block.text for block in message.blocks if isinstance(block, TextBlock))
    if not text:
        raise ValueError("Message must include at least one text block for CLI launch")
    return text


def _external_resume_id(session: Session, *, prefix: str) -> str:
    if session.id.startswith(prefix):
        resume_id = session.id.removeprefix(prefix)
        if resume_id:
            return resume_id

    backend = "claude" if prefix == "claude_" else prefix.rstrip("_")
    raise CommandBuildError(f"Cannot resume external {backend} session from id {session.id}")


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
