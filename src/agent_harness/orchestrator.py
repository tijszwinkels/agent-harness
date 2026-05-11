from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessCommand:
    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("ProcessCommand requires at least one argv item")


class BackendCommandBuilder(Protocol):
    def build(self, *, session: Session, run: Run, message: Message) -> ProcessCommand:
        """Build a non-interactive CLI command for a harness-owned run."""


class CodexCommandBuilder:
    def build(self, *, session: Session, run: Run, message: Message) -> ProcessCommand:
        del run
        return ProcessCommand(
            argv=("codex", "exec", "--json", "--model", session.model, _message_text(message)),
            cwd=session.project.path,
        )


class ClaudeCodeCommandBuilder:
    def build(self, *, session: Session, run: Run, message: Message) -> ProcessCommand:
        del run
        return ProcessCommand(
            argv=(
                "claude",
                "--print",
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--model",
                session.model,
                _message_text(message),
            ),
            cwd=session.project.path,
        )


def default_command_builders() -> dict[str, BackendCommandBuilder]:
    return {
        "claude-code": ClaudeCodeCommandBuilder(),
        "codex": CodexCommandBuilder(),
    }


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
    ) -> None:
        self.session = session
        self.run_record = run
        self.command = command
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
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
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                await self._publish("message.delta", {"stream": stream_name, "text": text})

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
    ) -> None:
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._active: dict[str, RunProcess] = {}
        self._tasks: dict[str, asyncio.Task[RunProcessResult]] = {}

    def start(self, *, session: Session, run: Run, command: ProcessCommand) -> RunProcess:
        run_process = RunProcess(
            session=session,
            run=run,
            command=command,
            event_bus=self._event_bus,
            process_factory=self._process_factory,
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
