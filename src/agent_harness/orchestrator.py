from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock
from agent_harness.observer import blocks_from_claude_message

logger = logging.getLogger(__name__)


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


def parse_claude_stream_line(line: str) -> list[tuple[str, dict[str, Any]]]:
    # Bridge needs the final `{"type":"assistant", "message":{...}}` record. Other shapes
    # (system/init, stream_event chunks from --include-partial-messages, result, hooks, etc.)
    # are ignored — the final assistant record already carries the full content.
    if not line:
        return []
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(record, dict):
        return []
    record_type = record.get("type")
    if record_type not in ("assistant", "user"):
        return []
    msg = record.get("message")
    if not isinstance(msg, Mapping):
        return []
    role = msg.get("role")
    if role not in ("assistant", "user"):
        return []
    blocks = blocks_from_claude_message(msg)
    if not blocks:
        return []
    message = Message(role=role, blocks=blocks, model=msg.get("model"))
    return [
        (
            "message",
            {
                "message": message.model_dump(mode="json"),
                "source_type": record_type,
            },
        )
    ]


def default_stdout_parsers() -> dict[str, StdoutParser]:
    return {
        "claude-code": parse_claude_stream_line,
    }


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

    async def run(self) -> RunProcessResult:
        await self._publish("run.started", {})

        try:
            self._process = await self._process_factory(self.command)
        except Exception as exc:
            logger.exception("Failed to start run process: session=%s run=%s", self.session.id, self.run_record.id)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))

        stream_tasks = self._stream_tasks(self._process)
        try:
            returncode = await self._process.wait()
            await self._finish_streams(stream_tasks)
        except Exception as exc:
            logger.exception("Run process failed while active: session=%s run=%s", self.session.id, self.run_record.id)
            await self._cancel_streams(stream_tasks)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))

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
        if stream_name == "stdout":
            parser = self._stdout_parsers.get(self.session.backend)
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                continue
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

    async def _publish(self, event: str, data: dict[str, object]) -> Event:
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
