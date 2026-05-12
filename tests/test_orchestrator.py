import asyncio
import json

import pytest

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Message, Project, Run, Session
from agent_harness.orchestrator import (
    ClaudeCodeCommandBuilder,
    CommandBuildError,
    CodexCommandBuilder,
    ProcessCommand,
    RunManager,
    RunProcess,
    parse_claude_stream_line,
)


class FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = asyncio.Queue()
        for line in lines:
            self._lines.put_nowait(line)
        self._lines.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._lines.get()


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: list[bytes] | None = None,
        stderr: list[bytes] | None = None,
        returncode: int = 0,
    ) -> None:
        self.stdout = FakeStream(stdout or [])
        self.stderr = FakeStream(stderr or [])
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._done = asyncio.Event()
        self.terminated = False

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode if self.returncode is not None else self._final_returncode

    def finish(self) -> None:
        self.returncode = self._final_returncode
        self._done.set()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._done.set()


class FakeFactory:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.commands: list[ProcessCommand] = []

    async def __call__(self, command: ProcessCommand) -> FakeProcess:
        self.commands.append(command)
        return self.process


def make_session(backend: str = "codex") -> Session:
    return Session(
        backend=backend,
        model="gpt-5.4",
        project=Project(path="/workspace/project", name="project"),
    )


def make_run(session: Session, *, origin: str = "harness") -> Run:
    return Run(
        session_id=session.id,
        status="running",
        started_at=session.created_at,
        input_message_id="msg_input",
        origin=origin,
    )


def test_codex_command_builder_uses_exec_json_mode_and_project_cwd() -> None:
    session = make_session("codex")
    run = make_run(session)
    message = Message.user("implement it")

    command = CodexCommandBuilder().build(session=session, run=run, message=message)

    assert command.argv == ("codex", "exec", "--json", "--model", "gpt-5.4", "implement it")
    assert command.cwd == "/workspace/project"
    assert command.env == {}


def test_codex_command_builder_resumes_external_codex_session() -> None:
    session = make_session("codex").model_copy(
        update={"id": "codex_019e07c3-4682-7ff1-99e8-948e64bb70c4", "origin": "external"}
    )
    run = make_run(session)
    message = Message.user("append this")

    command = CodexCommandBuilder().build(session=session, run=run, message=message)

    assert command.argv == (
        "codex",
        "exec",
        "resume",
        "--json",
        "--model",
        "gpt-5.4",
        "019e07c3-4682-7ff1-99e8-948e64bb70c4",
        "append this",
    )
    assert command.cwd == "/workspace/project"


def test_claude_code_command_builder_uses_headless_stream_json_mode() -> None:
    session = make_session("claude-code")
    run = make_run(session)
    message = Message.user("review it")

    command = ClaudeCodeCommandBuilder().build(session=session, run=run, message=message)

    assert command.argv == (
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        "gpt-5.4",
        "review it",
    )
    assert command.cwd == "/workspace/project"


def test_claude_code_command_builder_resumes_external_claude_session() -> None:
    session = make_session("claude-code").model_copy(
        update={"id": "claude_2a9857de-2f9d-4190-aa76-e433619602fb", "origin": "external"}
    )
    run = make_run(session)
    message = Message.user("continue this")

    command = ClaudeCodeCommandBuilder().build(session=session, run=run, message=message)

    assert command.argv == (
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        "gpt-5.4",
        "--resume",
        "2a9857de-2f9d-4190-aa76-e433619602fb",
        "continue this",
    )
    assert command.cwd == "/workspace/project"


def test_external_resume_requires_expected_session_id_prefix() -> None:
    session = make_session("codex").model_copy(update={"id": "external_without_backend_prefix", "origin": "external"})
    run = make_run(session)

    with pytest.raises(CommandBuildError, match="Cannot resume external codex session"):
        CodexCommandBuilder().build(session=session, run=run, message=Message.user("hello"))


@pytest.mark.asyncio
async def test_run_process_publishes_stdout_and_stderr_deltas_then_completion() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    run = make_run(session)
    process = FakeProcess(stdout=[b"hello\n"], stderr=[b"warn\n"], returncode=0)
    factory = FakeFactory(process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "hello"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
        ).run()
    )
    await asyncio.sleep(0)
    process.finish()
    result = await asyncio.wait_for(task, timeout=1)

    events = await bus.replay(session_id=session.id, run_id=run.id)
    assert result.status == "completed"
    assert [event.event for event in events] == [
        "run.started",
        "message.delta",
        "message.delta",
        "run.completed",
    ]
    assert events[1].data == {"stream": "stdout", "text": "hello"}
    assert events[2].data == {"stream": "stderr", "text": "warn"}
    assert events[3].data == {"returncode": 0}


@pytest.mark.asyncio
async def test_run_process_publishes_failed_for_nonzero_exit() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    run = make_run(session)
    process = FakeProcess(stderr=[b"boom\n"], returncode=2)
    factory = FakeFactory(process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "hello")),
            event_bus=bus,
            process_factory=factory,
        ).run()
    )
    await asyncio.sleep(0)
    process.finish()
    result = await asyncio.wait_for(task, timeout=1)

    events = await bus.replay(session_id=session.id, run_id=run.id)
    assert result.status == "failed"
    assert [event.event for event in events] == ["run.started", "message.delta", "run.failed"]
    assert events[-1].data == {"returncode": 2}


@pytest.mark.asyncio
async def test_run_manager_interrupts_owned_processes_only() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    owned_run = make_run(session)
    process = FakeProcess(stdout=[b"working\n"])
    manager = RunManager(event_bus=bus, process_factory=FakeFactory(process))

    manager.start(
        session=session,
        run=owned_run,
        command=ProcessCommand(argv=("codex", "exec", "hello")),
    )
    await asyncio.sleep(0)

    assert await manager.interrupt(session.id, owned_run.id) is True

    result = await asyncio.wait_for(manager.wait(owned_run.id), timeout=1)
    events = await bus.replay(session_id=session.id, run_id=owned_run.id)

    assert process.terminated is True
    assert result.status == "interrupted"
    assert events[-1].event == "run.interrupted"


def test_parse_claude_stream_line_extracts_assistant_message() -> None:
    record = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "pong"}],
        },
    }

    events = parse_claude_stream_line(json.dumps(record))

    assert len(events) == 1
    name, data = events[0]
    assert name == "message"
    assert data["source_type"] == "assistant"
    assert data["message"]["role"] == "assistant"
    assert data["message"]["model"] == "claude-sonnet-4-6"
    assert data["message"]["blocks"] == [{"type": "text", "text": "pong"}]


def test_parse_claude_stream_line_extracts_tool_use_blocks() -> None:
    record = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "looking…"},
                {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
            ],
        },
    }

    events = parse_claude_stream_line(json.dumps(record))

    assert len(events) == 1
    data = events[0][1]
    blocks = data["message"]["blocks"]
    assert blocks[0] == {"type": "text", "text": "looking…"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "Bash"
    assert blocks[1]["input"] == {"command": "ls"}
    assert blocks[1]["id"] == "tu_1"


def test_parse_claude_stream_line_ignores_system_and_stream_event_records() -> None:
    for record in (
        {"type": "system", "subtype": "init"},
        {"type": "stream_event", "event": {"type": "content_block_delta"}},
        {"type": "result", "subtype": "success", "result": "pong"},
        {"type": "rate_limit_event"},
    ):
        assert parse_claude_stream_line(json.dumps(record)) == []


def test_parse_claude_stream_line_ignores_non_json_and_empty_messages() -> None:
    assert parse_claude_stream_line("not json") == []
    assert parse_claude_stream_line("") == []
    assert parse_claude_stream_line(json.dumps({"type": "assistant"})) == []
    assert parse_claude_stream_line(json.dumps({"type": "assistant", "message": {}})) == []


@pytest.mark.asyncio
async def test_run_process_emits_structured_message_for_claude_assistant_stdout() -> None:
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    assistant_line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "pong"}],
            },
        }
    ).encode()
    process = FakeProcess(stdout=[assistant_line + b"\n"], returncode=0)
    factory = FakeFactory(process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print")),
            event_bus=bus,
            process_factory=factory,
        ).run()
    )
    await asyncio.sleep(0)
    process.finish()
    await asyncio.wait_for(task, timeout=1)

    events = await bus.replay(session_id=session.id, run_id=run.id)
    event_names = [event.event for event in events]
    assert "message" in event_names
    message_event = next(event for event in events if event.event == "message")
    assert message_event.data["message"]["role"] == "assistant"
    assert message_event.data["message"]["blocks"] == [{"type": "text", "text": "pong"}]


@pytest.mark.asyncio
async def test_run_manager_refuses_to_interrupt_active_external_runs() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    external_run = make_run(session, origin="external")
    process = FakeProcess()
    manager = RunManager(event_bus=bus, process_factory=FakeFactory(process))

    manager.start(
        session=session,
        run=external_run,
        command=ProcessCommand(argv=("codex", "exec", "hello")),
    )
    await asyncio.sleep(0)

    assert await manager.interrupt(session.id, external_run.id) is False
    assert process.terminated is False

    process.finish()
    await asyncio.wait_for(manager.wait(external_run.id), timeout=1)
