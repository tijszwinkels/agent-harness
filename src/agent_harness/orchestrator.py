from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock

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


def default_stdout_parsers() -> dict[str, StdoutParser]:
    # No structured stdout parsers: ``message`` events are published
    # exclusively by ``ExternalTranscriptObserver`` from the backend's rollout
    # JSONL. RunProcess still emits ``message.delta`` raw stdout lines and the
    # ``run.*`` lifecycle events, which is enough for typing indicators and
    # log streaming.
    return {}


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


class RunManager:
    def __init__(
        self,
        *,
        event_bus: InMemoryEventBus,
        process_factory: ProcessFactory | None = None,
        stdout_parsers: Mapping[str, StdoutParser] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._stdout_parsers: Mapping[str, StdoutParser] = (
            stdout_parsers if stdout_parsers is not None else default_stdout_parsers()
        )
        self._active: dict[str, RunProcess] = {}
        self._tasks: dict[str, asyncio.Task[RunProcessResult]] = {}

    def start(self, *, session: Session, run: Run, command: ProcessCommand) -> RunProcess:
        run_process = RunProcess(
            session=session,
            run=run,
            command=command,
            event_bus=self._event_bus,
            process_factory=self._process_factory,
            stdout_parsers=self._stdout_parsers,
        )
        self._active[run.id] = run_process
        task = asyncio.create_task(self._run_and_forget(run_process))
        self._tasks[run.id] = task
        return run_process

    async def interrupt(self, session_id: str, run_id: str) -> bool:
        run_process = self._active.get(run_id)
        if run_process is None or run_process.session.id != session_id:
            return False
        return await run_process.interrupt()

    async def wait(self, run_id: str) -> RunProcessResult:
        task = self._tasks[run_id]
        return await task

    async def _run_and_forget(self, run_process: RunProcess) -> RunProcessResult:
        try:
            return await run_process.run()
        finally:
            self._active.pop(run_process.run_record.id, None)


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
