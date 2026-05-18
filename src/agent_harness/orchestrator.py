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
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock, utc_now
from agent_harness.usage import parse_claude_usage, parse_codex_token_count, parse_codex_usage

logger = logging.getLogger(__name__)

END_TURN_GRACE_SECONDS = 20.0
END_TURN_HARD_KILL_AFTER_SECONDS = 40.0
IDLE_TIMEOUT_SECONDS = 30 * 60
IDLE_HARD_KILL_GRACE_SECONDS = 30.0
IDLE_CHECK_INTERVAL_SECONDS = 60.0

_END_TURN_EVENT = "__end_turn__"
_ACTIVITY_EVENTS = frozenset({
    "message",
    "message.delta",
    "run.usage",
    # Reserved for future stdout parsers; activity tracking already handles it.
    "tool_use",
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


StdoutParser = Callable[[str], list[tuple[str, dict[str, Any]]]]


def parse_codex_stream_line(line: str) -> list[tuple[str, dict[str, Any]]]:
    if not line:
        return []
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(record, Mapping):
        return []
    if record.get("type") == "turn.completed":
        events: list[tuple[str, dict[str, Any]]] = []
        usage = parse_codex_usage(record.get("usage"))
        if usage is not None:
            events.append(
                (
                    "run.usage",
                    {
                        "usage": usage.model_dump(mode="json"),
                        "source_type": "turn.completed",
                    },
                )
            )
        events.append((_END_TURN_EVENT, {}))
        return events
    payload = record.get("payload")
    if record.get("type") == "event_msg" and isinstance(payload, Mapping) and payload.get("type") == "token_count":
        usage, context_window = parse_codex_token_count(payload)
        data: dict[str, Any] = {}
        if usage is not None:
            data["usage"] = usage.model_dump(mode="json")
            data["source_type"] = "token_count"
        if context_window is not None:
            data["context_window"] = context_window
        return [("run.usage", data)] if data else []
    if record.get("type") != "item.completed":
        return []
    item = record.get("item")
    if not isinstance(item, Mapping) or item.get("type") != "agent_message":
        return []
    text = item.get("text")
    if not isinstance(text, str) or not text:
        return []

    message = Message(role="assistant", blocks=[TextBlock(text=text)], model=None)
    return [
        (
            "message",
            {
                "message": message.model_dump(mode="json"),
                "source_type": "agent_message",
            },
        )
    ]


def parse_claude_stream_line(line: str) -> list[tuple[str, dict[str, Any]]]:
    if not line:
        return []
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(record, Mapping):
        return []
    if record.get("type") == "result" and record.get("stop_reason") == "end_turn":
        events: list[tuple[str, dict[str, Any]]] = []
        usage = parse_claude_usage(record.get("usage"), cost_usd=record.get("total_cost_usd"))
        if usage is not None:
            events.append(("run.usage", {"usage": usage.model_dump(mode="json")}))
        events.append((_END_TURN_EVENT, {}))
        return events
    return []


def default_stdout_parsers() -> dict[str, StdoutParser]:
    # Claude messages still come exclusively from ExternalTranscriptObserver,
    # but codex exec --json cannot be pinned to the harness session id. Its
    # assistant messages must be synthesized from stdout under the harness run.
    return {"codex": parse_codex_stream_line, "claude-code": parse_claude_stream_line}


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
        stdout_parsers: Mapping[str, StdoutParser] | None = None,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        codex_sessions_root: Path | None = None,
    ) -> None:
        self.session = session
        self.run_record = run
        self.command = command
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._stdout_parsers: Mapping[str, StdoutParser] = (
            stdout_parsers if stdout_parsers is not None else default_stdout_parsers()
        )
        self._process: ManagedProcess | None = None
        self._interrupted = False
        self._clock = clock
        self._sleep = sleep
        self._end_turn_event = asyncio.Event()
        self._last_activity_at = self._clock()
        self._last_activity_event: str | None = None
        self._watchdog_termination_in_progress = False
        self._watchdog_termination_task: asyncio.Task[None] | None = None
        self._codex_thread_id: str | None = None
        self._codex_token_count_usage_seen = False
        self._codex_sessions_root = codex_sessions_root

    async def run(self) -> RunProcessResult:
        await self._publish("run.started", {})

        try:
            self._process = await self._process_factory(self.command)
        except Exception as exc:
            logger.exception("Failed to start run process: session=%s run=%s", self.session.id, self.run_record.id)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))

        stream_tasks = self._stream_tasks(self._process)
        wait_task = asyncio.create_task(self._process.wait())
        watchdog_tasks = [
            asyncio.create_task(self._watch_end_turn_cleanup(wait_task)),
            asyncio.create_task(self._watch_idle_timeout(wait_task)),
        ]
        try:
            returncode = await wait_task
            await self._finish_streams(stream_tasks)
            await self._publish_codex_context_window()
        except Exception as exc:
            logger.exception("Run process failed while active: session=%s run=%s", self.session.id, self.run_record.id)
            await self._cancel_streams(stream_tasks)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))
        finally:
            await self._finish_watchdogs(watchdog_tasks)

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

    def _stream_tasks(self, process: ManagedProcess) -> list[asyncio.Task[None]]:
        tasks: list[asyncio.Task[None]] = []
        if process.stdout is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stdout", process.stdout)))
        if process.stderr is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stderr", process.stderr)))
        return tasks

    async def _stream_lines(self, stream_name: Literal["stdout", "stderr"], stream: AsyncLineReader) -> None:
        parser: StdoutParser | None = None
        if stream_name == "stdout" and self.session.origin == "harness":
            parser = self._stdout_parsers.get(self.session.backend)
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                continue
            self._capture_codex_thread_id(stream_name, text)
            await self._publish("message.delta", {"stream": stream_name, "text": text})
            if parser is None:
                continue
            try:
                events = parser(text)
            except Exception:
                logger.exception(
                    "Stdout parser raised: session=%s run=%s backend=%s",
                    self.session.id,
                    self.run_record.id,
                    self.session.backend,
                )
                continue
            for event_name, data in events:
                if event_name == _END_TURN_EVENT:
                    self._end_turn_event.set()
                    continue
                if self._should_skip_parsed_event(event_name, data):
                    continue
                await self._publish(event_name, data)

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

    def _capture_codex_thread_id(self, stream_name: str, text: str) -> None:
        if stream_name != "stdout" or self.session.backend != "codex" or self.session.origin != "harness":
            return
        if self._codex_thread_id is not None:
            return
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(record, Mapping) or record.get("type") != "thread.started":
            return
        thread_id = record.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            self._codex_thread_id = thread_id

    def _should_skip_parsed_event(self, event_name: str, data: dict[str, Any]) -> bool:
        if event_name != "run.usage" or self.session.backend != "codex":
            return False
        if "usage" not in data:
            return False
        source_type = data.get("source_type")
        if source_type == "token_count":
            self._codex_token_count_usage_seen = True
            return False
        return source_type == "turn.completed" and self._codex_token_count_usage_seen

    async def _publish_codex_context_window(self) -> None:
        if self.session.backend != "codex" or self.session.origin != "harness":
            return
        thread_id = self._codex_thread_id
        if thread_id is None:
            return
        context_window = await asyncio.to_thread(
            _context_window_from_codex_rollout,
            thread_id,
            self._codex_sessions_root,
        )
        if context_window is None:
            return
        await self._publish("run.usage", {"context_window": context_window})


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
        stdout_parsers: Mapping[str, StdoutParser] | None = None,
        queue_max_per_session: int = RUN_QUEUE_MAX_PER_SESSION,
    ) -> None:
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._stdout_parsers: Mapping[str, StdoutParser] = (
            stdout_parsers if stdout_parsers is not None else default_stdout_parsers()
        )
        self._queue_max_per_session = queue_max_per_session
        self._active: dict[str, RunProcess] = {}
        self._tasks: dict[str, asyncio.Task[RunProcessResult]] = {}
        # Tracks which run_id currently owns the subprocess for each session
        # (at most one). New submits for an already-owned session land on the
        # per-session FIFO in ``_queues`` instead of spawning concurrently —
        # which would race on the shared claude-code rollout JSONL.
        self._active_run_by_session: dict[str, str] = {}
        self._queues: dict[str, deque[_QueuedRun]] = {}

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
            stdout_parsers=self._stdout_parsers,
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


def _context_window_from_codex_rollout(thread_id: str, sessions_root: Path | None = None) -> int | None:
    root = sessions_root or Path.home() / ".codex" / "sessions"
    try:
        candidates = sorted(
            root.rglob(f"rollout-*-{thread_id}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        logger.exception("Failed to search Codex rollout root for thread context window: root=%s", root)
        return None

    if not candidates:
        logger.debug("No Codex rollout found for thread id: %s", thread_id)
        return None

    path = candidates[0]
    try:
        with path.open("r", encoding="utf-8") as transcript:
            for line in transcript:
                context_window = _context_window_from_codex_rollout_line(line)
                if context_window is not None:
                    return context_window
    except OSError:
        logger.exception("Failed to read Codex rollout for context window: path=%s", path)
    return None


def _context_window_from_codex_rollout_line(line: str) -> int | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, Mapping) or record.get("type") != "event_msg":
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    context_window = payload.get("model_context_window")
    if isinstance(context_window, int) and context_window >= 1:
        return context_window
    if payload.get("type") != "token_count":
        return None
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return None
    context_window = info.get("model_context_window")
    return context_window if isinstance(context_window, int) and context_window >= 1 else None


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
