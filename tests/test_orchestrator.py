import asyncio
import json
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import agent_harness.orchestrator as orchestrator
from agent_harness.events import InMemoryEventBus
from agent_harness.models import Message, Project, Run, Session
from agent_harness.orchestrator import (
    ClaudeCodeCommandBuilder,
    CommandBuildError,
    CodexCommandBuilder,
    ProcessCommand,
    RunManager,
    RunProcess,
    SubmitResult,
)


class FakeStream:
    def __init__(self, lines: list[bytes], *, close: bool = True) -> None:
        self._lines = asyncio.Queue()
        for line in lines:
            self._lines.put_nowait(line)
        if close:
            self.close()

    async def readline(self) -> bytes:
        return await self._lines.get()

    def push(self, line: bytes) -> None:
        self._lines.put_nowait(line)

    def close(self) -> None:
        self._lines.put_nowait(b"")


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: list[bytes] | None = None,
        stderr: list[bytes] | None = None,
        returncode: int = 0,
        pid: int = 12345,
        close_stdout: bool = True,
        close_stderr: bool = True,
        exit_on_sigterm: bool = False,
    ) -> None:
        self.stdout = FakeStream(stdout or [], close=close_stdout)
        self.stderr = FakeStream(stderr or [], close=close_stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._done = asyncio.Event()
        self.terminated = False
        self.pid = pid
        self.group_signals: list[signal.Signals] = []
        self.exit_on_sigterm = exit_on_sigterm

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

    def signal_group(self, sig: signal.Signals) -> None:
        self.group_signals.append(sig)
        if sig == signal.SIGTERM and self.exit_on_sigterm:
            self.returncode = -signal.SIGTERM
            self._done.set()
        elif sig == signal.SIGKILL:
            self.returncode = -signal.SIGKILL
            self._done.set()


class FakeFactory:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.commands: list[ProcessCommand] = []

    async def __call__(self, command: ProcessCommand) -> FakeProcess:
        self.commands.append(command)
        return self.process


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 5, 17, 12, 0, tzinfo=UTC)
        self._sleepers: list[tuple[datetime, asyncio.Future[None]]] = []

    def __call__(self) -> datetime:
        return self._now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self._now + timedelta(seconds=delay), future))
        await future

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        ready = [item for item in self._sleepers if item[0] <= self._now]
        self._sleepers = [item for item in self._sleepers if item[0] > self._now]
        for _, future in ready:
            if not future.done():
                future.set_result(None)


def install_fake_process_group(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
    *,
    pgid: int = 67890,
) -> list[tuple[int, signal.Signals]]:
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(orchestrator.os, "getpgid", lambda pid: pgid)

    def fake_killpg(observed_pgid: int, sig: signal.Signals) -> None:
        calls.append((observed_pgid, sig))
        process.signal_group(sig)

    monkeypatch.setattr(orchestrator.os, "killpg", fake_killpg)
    return calls


async def flush_asyncio() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


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
    # Pin the session id to a known hex so the derived UUID is predictable.
    session = make_session("claude-code").model_copy(
        update={"id": "ses_3eb0e45b9d724deabdc3b472e0c4c2fc"}
    )
    run = make_run(session)
    message = Message.user("review it")

    command = ClaudeCodeCommandBuilder().build(
        session=session, run=run, message=message, is_first_run=True,
    )

    assert command.argv == (
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        "gpt-5.4",
        "--session-id",
        "3eb0e45b-9d72-4dea-bdc3-b472e0c4c2fc",
        "review it",
    )
    assert command.cwd == "/workspace/project"


def test_claude_code_command_builder_resumes_harness_session_on_subsequent_runs() -> None:
    session = make_session("claude-code").model_copy(
        update={"id": "ses_3eb0e45b9d724deabdc3b472e0c4c2fc"}
    )
    run = make_run(session)
    message = Message.user("follow-up question")

    command = ClaudeCodeCommandBuilder().build(
        session=session, run=run, message=message, is_first_run=False,
    )

    assert "--session-id" not in command.argv
    assert "--resume" in command.argv
    resume_idx = command.argv.index("--resume")
    assert command.argv[resume_idx + 1] == "3eb0e45b-9d72-4dea-bdc3-b472e0c4c2fc"


def test_claude_code_command_builder_rejects_non_hex_harness_session_id() -> None:
    session = make_session("claude-code").model_copy(update={"id": "ses_not-uuid-shaped"})
    run = make_run(session)

    with pytest.raises(CommandBuildError, match="Cannot derive claude session UUID"):
        ClaudeCodeCommandBuilder().build(
            session=session, run=run, message=Message.user("hi"), is_first_run=True,
        )


def test_claude_code_command_builder_resumes_external_claude_session() -> None:
    # External claude sessions now share the canonical ses_<32hex> form with
    # harness-origin sessions — the external observer's
    # external_session_id_from_claude_path emits this format. The builder
    # derives the claude UUID via _harness_session_id_as_uuid.
    session = make_session("claude-code").model_copy(
        update={"id": "ses_2a9857de2f9d4190aa76e433619602fb", "origin": "external"}
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


def test_claude_code_command_builder_resumes_legacy_claude_prefixed_external_session() -> None:
    # Backward-compat: external sessions persisted under the legacy
    # claude_<uuid-with-dashes> id (before the canonicalization) must still
    # resume cleanly. _harness_session_id_as_uuid accepts both forms.
    session = make_session("claude-code").model_copy(
        update={"id": "claude_2a9857de-2f9d-4190-aa76-e433619602fb", "origin": "external"}
    )
    run = make_run(session)

    command = ClaudeCodeCommandBuilder().build(
        session=session, run=run, message=Message.user("legacy resume"),
    )
    assert "--resume" in command.argv
    resume_idx = command.argv.index("--resume")
    assert command.argv[resume_idx + 1] == "2a9857de-2f9d-4190-aa76-e433619602fb"


def test_codex_command_builder_appends_dangerously_bypass_when_bypass_permissions_set() -> None:
    session = make_session("codex").model_copy(update={"bypass_permissions": True})
    run = make_run(session)
    message = Message.user("yolo run")

    command = CodexCommandBuilder().build(session=session, run=run, message=message)

    assert command.argv == (
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-5.4",
        "--dangerously-bypass-approvals-and-sandbox",
        "yolo run",
    )


def test_codex_command_builder_appends_dangerously_bypass_on_external_resume_when_bypass_permissions_set() -> None:
    session = make_session("codex").model_copy(
        update={
            "id": "codex_019e07c3-4682-7ff1-99e8-948e64bb70c4",
            "origin": "external",
            "bypass_permissions": True,
        }
    )
    run = make_run(session)

    command = CodexCommandBuilder().build(session=session, run=run, message=Message.user("yolo resume"))

    assert "--dangerously-bypass-approvals-and-sandbox" in command.argv
    # Must appear before the resume id + prompt positional args.
    flag_idx = command.argv.index("--dangerously-bypass-approvals-and-sandbox")
    assert command.argv[-2] == "019e07c3-4682-7ff1-99e8-948e64bb70c4"
    assert command.argv[-1] == "yolo resume"
    assert flag_idx < len(command.argv) - 2


def test_claude_code_command_builder_appends_dangerously_skip_when_flag_set() -> None:
    session = make_session("claude-code").model_copy(
        update={
            "id": "ses_3eb0e45b9d724deabdc3b472e0c4c2fc",
            "bypass_permissions": True,
        }
    )
    run = make_run(session)

    command = ClaudeCodeCommandBuilder().build(
        session=session, run=run, message=Message.user("yolo claude"), is_first_run=True,
    )

    assert command.argv == (
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        "gpt-5.4",
        "--dangerously-skip-permissions",
        "--session-id",
        "3eb0e45b-9d72-4dea-bdc3-b472e0c4c2fc",
        "yolo claude",
    )


def test_claude_code_command_builder_appends_dangerously_skip_on_external_resume() -> None:
    session = make_session("claude-code").model_copy(
        update={
            "id": "ses_2a9857de2f9d4190aa76e433619602fb",
            "origin": "external",
            "bypass_permissions": True,
        }
    )
    run = make_run(session)

    command = ClaudeCodeCommandBuilder().build(session=session, run=run, message=Message.user("yolo resume"))

    assert "--dangerously-skip-permissions" in command.argv
    skip_idx = command.argv.index("--dangerously-skip-permissions")
    resume_idx = command.argv.index("--resume")
    assert skip_idx < resume_idx


def test_external_resume_requires_expected_session_id_prefix() -> None:
    session = make_session("codex").model_copy(update={"id": "external_without_backend_prefix", "origin": "external"})
    run = make_run(session)

    with pytest.raises(CommandBuildError, match="Cannot resume external codex session"):
        CodexCommandBuilder().build(session=session, run=run, message=Message.user("hello"))


@pytest.mark.asyncio
async def test_run_process_publishes_process_stderr_and_completion() -> None:
    """Phase 2: stdout non-JSON lines do not travel the event bus
    (no message.delta). Stderr lines emit ``process.stderr`` with a
    ``{text: ...}`` payload."""
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

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert result.status == "completed"
    assert [event.event for event in events] == [
        "run.started",
        "process.stderr",
        "run.completed",
    ]
    assert events[1].data == {"text": "warn"}
    assert events[2].data == {"returncode": 0}


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

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert result.status == "failed"
    assert [event.event for event in events] == ["run.started", "process.stderr", "run.failed"]
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
    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == owned_run.id]

    assert process.terminated is True
    assert result.status == "interrupted"
    assert events[-1].event == "run.interrupted"


@pytest.mark.asyncio
async def test_external_codex_turn_completed_does_not_arm_end_turn_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External codex sessions (origin=external) do NOT run through the
    end-turn detector — Phase 2 keeps the harness-origin gate, so a
    ``turn.completed`` in an externally-managed codex stdout doesn't
    arm the supervisor's cleanup watchdog."""
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    monkeypatch.setattr(orchestrator, "END_TURN_HARD_KILL_AFTER_SECONDS", 40.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("codex").model_copy(
        update={"origin": "external", "id": "codex_019e07c3-4682-7ff1-99e8-948e64bb70c4"}
    )
    run = make_run(session, origin="external")
    clock = FakeClock()
    process = FakeProcess(stdout=[b'{"type":"turn.completed"}\n'])
    install_fake_process_group(monkeypatch, process)
    run_process = RunProcess(
        session=session,
        run=run,
        command=ProcessCommand(argv=("codex", "exec", "resume", "--json", "external", "hello")),
        event_bus=bus,
        process_factory=FakeFactory(process),
        clock=clock,
        sleep=clock.sleep,
    )

    task = asyncio.create_task(run_process.run())
    await flush_asyncio()

    assert run_process._end_turn_event.is_set() is False

    clock.advance(60)
    await flush_asyncio()
    clock.advance(60)
    await flush_asyncio()

    assert process.group_signals == []
    assert run_process._end_turn_event.is_set() is False

    process.finish()
    result = await asyncio.wait_for(task, timeout=1)

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert result.status == "completed"
    # No message.delta from stdout, no process.stderr (we didn't write
    # to stderr), just lifecycle events.
    assert [event.event for event in events] == ["run.started", "run.completed"]


@pytest.mark.asyncio
async def test_run_process_kills_after_end_turn_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    monkeypatch.setattr(orchestrator, "END_TURN_HARD_KILL_AFTER_SECONDS", 40.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    clock = FakeClock()
    end_turn_line = json.dumps(
        {"type": "result", "subtype": "success", "stop_reason": "end_turn"}
    ).encode()
    process = FakeProcess(stdout=[end_turn_line + b"\n"])
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()

    clock.advance(20)
    await flush_asyncio()
    assert process.group_signals == [signal.SIGTERM]

    clock.advance(20)
    await flush_asyncio()
    result = await asyncio.wait_for(task, timeout=1)
    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    watchdog_event = next(e for e in events if e.event == "run.terminated_after_end_turn")

    assert result.status == "interrupted"
    assert process.group_signals == [signal.SIGTERM, signal.SIGKILL]
    assert watchdog_event.data == {
        "grace_seconds": 20,
        "hard_kill": True,
        "returncode": -signal.SIGKILL,
        "reason": "subprocess_did_not_exit_after_end_turn",
    }


@pytest.mark.asyncio
async def test_run_process_end_turn_does_not_kill_when_subprocess_exits_naturally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    clock = FakeClock()
    end_turn_line = json.dumps(
        {"type": "result", "subtype": "success", "stop_reason": "end_turn"}
    ).encode()
    process = FakeProcess(stdout=[end_turn_line + b"\n"], returncode=0)
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()
    clock.advance(5)
    process.finish()
    result = await asyncio.wait_for(task, timeout=1)

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert result.status == "completed"
    assert process.group_signals == []
    assert "run.terminated_after_end_turn" not in [event.event for event in events]


@pytest.mark.asyncio
async def test_run_process_kill_only_on_end_turn_not_tool_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    clock = FakeClock()
    tool_use_line = json.dumps(
        {"type": "result", "subtype": "success", "stop_reason": "tool_use"}
    ).encode()
    process = FakeProcess(stdout=[tool_use_line + b"\n"])
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()
    clock.advance(60)
    await flush_asyncio()

    assert process.group_signals == []
    process.finish()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_run_process_idle_timeout_fires_after_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "IDLE_TIMEOUT_SECONDS", 30 * 60, raising=False)
    monkeypatch.setattr(orchestrator, "IDLE_CHECK_INTERVAL_SECONDS", 60.0, raising=False)
    monkeypatch.setattr(orchestrator, "IDLE_HARD_KILL_GRACE_SECONDS", 30.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session()
    run = make_run(session)
    clock = FakeClock()
    process = FakeProcess()
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "--json", "hello")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()

    clock.advance(31 * 60)
    await flush_asyncio()
    assert process.group_signals == [signal.SIGTERM]

    clock.advance(30)
    await flush_asyncio()
    result = await asyncio.wait_for(task, timeout=1)
    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    timeout_event = next(e for e in events if e.event == "run.timed_out_idle")

    assert result.status == "interrupted"
    assert process.group_signals == [signal.SIGTERM, signal.SIGKILL]
    assert timeout_event.data == {
        "idle_seconds": 1800,
        "last_activity_event": None,
        "last_activity_at": "2026-05-17T12:00:00Z",
        "hard_kill": True,
        "reason": "no_activity_within_threshold",
    }


@pytest.mark.asyncio
async def test_run_process_idle_timeout_resets_on_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 2: stdout non-JSON lines are silent on the bus (no
    message.delta) so they can't tick the idle watchdog anymore. The
    activity stream the supervisor still controls is stderr —
    ``process.stderr`` is in ``_ACTIVITY_EVENTS``. Long-running codex
    runs typically emit periodic stderr; the watchdog resets on each."""
    monkeypatch.setattr(orchestrator, "IDLE_TIMEOUT_SECONDS", 30 * 60, raising=False)
    monkeypatch.setattr(orchestrator, "IDLE_CHECK_INTERVAL_SECONDS", 60.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session()
    run = make_run(session)
    clock = FakeClock()
    process = FakeProcess(stdout=[], stderr=[], close_stdout=False, close_stderr=False)
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "--json", "hello")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()

    clock.advance(25 * 60)
    process.stderr.push(b"still working\n")
    await flush_asyncio()
    clock.advance(25 * 60)
    process.stderr.push(b"still working again\n")
    await flush_asyncio()

    assert process.group_signals == []
    process.finish()
    process.stdout.close()
    process.stderr.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_run_process_uses_process_group_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    clock = FakeClock()
    end_turn_line = json.dumps(
        {"type": "result", "subtype": "success", "stop_reason": "end_turn"}
    ).encode()
    process = FakeProcess(stdout=[end_turn_line + b"\n"], pid=24680, exit_on_sigterm=True)
    calls = install_fake_process_group(monkeypatch, process, pgid=13579)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()
    clock.advance(20)
    await asyncio.wait_for(task, timeout=1)

    assert calls == [(13579, signal.SIGTERM)]


@pytest.mark.asyncio
async def test_codex_end_turn_stop_signal_is_turn_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "END_TURN_GRACE_SECONDS", 20.0, raising=False)
    bus = InMemoryEventBus()
    session = make_session("codex")
    run = make_run(session)
    clock = FakeClock()
    process = FakeProcess(stdout=[b'{"type":"turn.completed"}\n'], exit_on_sigterm=True)
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "--json", "hello")),
            event_bus=bus,
            process_factory=FakeFactory(process),
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    await flush_asyncio()
    clock.advance(20)
    result = await asyncio.wait_for(task, timeout=1)

    assert result.status == "interrupted"
    assert process.group_signals == [signal.SIGTERM]


class FakeFactoryQueue:
    # Like FakeFactory but hands out a different FakeProcess on each call —
    # required for tests that exercise multiple sequential subprocess spawns
    # (queued runs draining into running ones).
    def __init__(self, processes: list[FakeProcess]) -> None:
        self.processes = list(processes)
        self.commands: list[ProcessCommand] = []
        self.spawned: list[FakeProcess] = []

    async def __call__(self, command: ProcessCommand) -> FakeProcess:
        self.commands.append(command)
        if not self.processes:
            raise AssertionError("FakeFactoryQueue exhausted — too many spawns")
        process = self.processes.pop(0)
        self.spawned.append(process)
        return process


def _make_queued_run(session: Session, *, run_id: str) -> Run:
    return Run(
        id=run_id,
        session_id=session.id,
        status="queued",
        started_at=None,
        input_message_id="msg_input",
        origin="harness",
    )


@pytest.mark.asyncio
async def test_run_manager_serializes_rapid_back_to_back_submits_on_same_session() -> None:
    # The core bug: two POST /runs in quick succession on one session must
    # not spawn concurrent subprocesses (they would race on the shared
    # claude rollout JSONL). First submit spawns, second submit queues.
    bus = InMemoryEventBus()
    session = make_session()
    run_a = _make_queued_run(session, run_id="run_a")
    run_b = _make_queued_run(session, run_id="run_b")
    proc_a = FakeProcess(returncode=0)
    proc_b = FakeProcess(returncode=0)
    factory = FakeFactoryQueue([proc_a, proc_b])
    manager = RunManager(event_bus=bus, process_factory=factory)

    started_callbacks: list[str] = []

    def on_start_a() -> None:
        started_callbacks.append("a")

    def on_start_b() -> None:
        started_callbacks.append("b")

    result_a = manager.submit(
        session=session, run=run_a,
        command=ProcessCommand(argv=("claude", "--print", "a")),
        on_start=on_start_a,
    )
    result_b = manager.submit(
        session=session, run=run_b,
        command=ProcessCommand(argv=("claude", "--print", "b")),
        on_start=on_start_b,
    )

    assert result_a == SubmitResult(accepted=True, status="running")
    assert result_b == SubmitResult(accepted=True, status="queued")
    # Only the first run's on_start fires synchronously. The second's is
    # held until the queue drains, which proves no concurrent spawn yet.
    assert started_callbacks == ["a"]
    await asyncio.sleep(0)
    assert len(factory.spawned) == 1

    # Drain run A; run B should now spawn and fire its on_start.
    proc_a.finish()
    await asyncio.wait_for(manager.wait("run_a"), timeout=1)
    # Yield to let _run_and_forget's finally schedule the next spawn.
    await asyncio.sleep(0)
    assert started_callbacks == ["a", "b"]
    assert len(factory.spawned) == 2

    proc_b.finish()
    await asyncio.wait_for(manager.wait("run_b"), timeout=1)


@pytest.mark.asyncio
async def test_run_manager_rejects_submit_when_queue_cap_reached() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    active_run = _make_queued_run(session, run_id="run_active")
    proc = FakeProcess(returncode=0)
    factory = FakeFactoryQueue([proc] + [FakeProcess() for _ in range(3)])
    # Cap at 3 to keep the test small but exercise the same code path.
    manager = RunManager(event_bus=bus, process_factory=factory, queue_max_per_session=3)

    assert manager.submit(
        session=session, run=active_run,
        command=ProcessCommand(argv=("claude", "--print", "active")),
    ).status == "running"

    # Three queued submits should all be accepted; the fourth must reject.
    for i in range(3):
        run = _make_queued_run(session, run_id=f"run_q{i}")
        assert manager.submit(
            session=session, run=run,
            command=ProcessCommand(argv=("claude", "--print", f"q{i}")),
        ).status == "queued"

    overflow_run = _make_queued_run(session, run_id="run_overflow")
    result = manager.submit(
        session=session, run=overflow_run,
        command=ProcessCommand(argv=("claude", "--print", "overflow")),
    )
    assert result == SubmitResult(accepted=False, status=None, reason="queue_full")

    # Cleanup: finish the active run and drain the queue so the test doesn't
    # leave dangling tasks. drop_queued() pops everything before the active
    # run's finally would re-spawn it.
    assert manager.drop_queued(session.id) == ["run_q0", "run_q1", "run_q2"]
    proc.finish()
    await asyncio.wait_for(manager.wait("run_active"), timeout=1)


@pytest.mark.asyncio
async def test_run_manager_queue_is_per_session_not_global() -> None:
    bus = InMemoryEventBus()
    session_a = make_session().model_copy(update={"id": "ses_aaa"})
    session_b = make_session().model_copy(update={"id": "ses_bbb"})
    run_a = _make_queued_run(session_a, run_id="run_a")
    run_b = _make_queued_run(session_b, run_id="run_b")
    proc_a = FakeProcess(returncode=0)
    proc_b = FakeProcess(returncode=0)
    factory = FakeFactoryQueue([proc_a, proc_b])
    manager = RunManager(event_bus=bus, process_factory=factory)

    assert manager.submit(
        session=session_a, run=run_a,
        command=ProcessCommand(argv=("claude", "a")),
    ).status == "running"
    # Different session — should spawn concurrently, NOT queue behind run_a.
    assert manager.submit(
        session=session_b, run=run_b,
        command=ProcessCommand(argv=("claude", "b")),
    ).status == "running"

    await asyncio.sleep(0)
    assert len(factory.spawned) == 2

    proc_a.finish()
    proc_b.finish()
    await asyncio.wait_for(manager.wait("run_a"), timeout=1)
    await asyncio.wait_for(manager.wait("run_b"), timeout=1)


@pytest.mark.asyncio
async def test_drop_queued_returns_popped_run_ids_and_prevents_spawn() -> None:
    bus = InMemoryEventBus()
    session = make_session()
    active = _make_queued_run(session, run_id="run_active")
    queued = _make_queued_run(session, run_id="run_queued")
    active_proc = FakeProcess(returncode=0)
    # If drop_queued fails, we'd attempt to spawn the queued run too — the
    # factory has no second process, so an assertion would fire.
    factory = FakeFactoryQueue([active_proc])
    manager = RunManager(event_bus=bus, process_factory=factory)

    manager.submit(
        session=session, run=active,
        command=ProcessCommand(argv=("claude", "active")),
    )
    manager.submit(
        session=session, run=queued,
        command=ProcessCommand(argv=("claude", "queued")),
    )

    popped = manager.drop_queued(session.id)
    assert popped == ["run_queued"]

    # Idempotency check.
    assert manager.drop_queued(session.id) == []

    active_proc.finish()
    await asyncio.wait_for(manager.wait("run_active"), timeout=1)
    # And the queued run must NOT have been spawned by the finally block.
    assert len(factory.spawned) == 1


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


# --- Phase 1: rollout pre-binding (flag-gated) ---------------------------


class _RecordingObserver:
    """Minimal stand-in for ``ExternalTranscriptObserver``. The
    orchestrator's pre-bind path calls ``bind_rollout`` at spawn and
    ``unbind_rollout`` once the run reaches a terminal state."""

    def __init__(self) -> None:
        self.bindings: list[tuple[object, str]] = []
        self.unbindings: list[object] = []

    def bind_rollout(self, path, session_id: str) -> None:
        self.bindings.append((path, session_id))

    def unbind_rollout(self, path) -> None:
        self.unbindings.append(path)


class _RecordingDiscovery:
    def __init__(
        self,
        *,
        claude_path=None,
        codex_path=None,
        codex_error: Exception | None = None,
    ) -> None:
        from pathlib import Path

        self._claude = claude_path or Path("/tmp/fake-claude-rollout.jsonl")
        self._codex = codex_path or Path("/tmp/fake-codex-rollout.jsonl")
        self._codex_error = codex_error
        self.claude_calls: list[dict] = []
        self.codex_calls: list[dict] = []

    def discover_claude(self, *, session_id: str, cwd):
        self.claude_calls.append({"session_id": session_id, "cwd": cwd})
        return self._claude

    async def discover_codex(self, *, pid: int, cwd, timeout_seconds: float = 5.0, poll_interval: float = 0.05):
        self.codex_calls.append({"pid": pid, "cwd": cwd})
        if self._codex_error is not None:
            raise self._codex_error
        return self._codex


@pytest.mark.asyncio
async def test_run_process_pre_binds_claude_rollout() -> None:
    """Phase 2: claude pre-bind is unconditional for harness sessions
    (no env-var gate). The session_id passed to ``discover_claude`` is
    the 8-4-4-4-12 dashed UUID derived from ``session.id`` (matching
    what claude writes on disk), NOT the harness's ``ses_<hex>`` form."""
    from pathlib import Path

    from agent_harness.orchestrator import _harness_session_id_as_uuid

    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    process = FakeProcess(returncode=0)
    factory = FakeFactory(process)
    observer = _RecordingObserver()
    claude_path = Path("/tmp/claude-pre-bind.jsonl")
    discovery = _RecordingDiscovery(claude_path=claude_path)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
            observer=observer,
            rollout_discovery=discovery,
        ).run()
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    process.finish()
    await asyncio.wait_for(task, timeout=1)

    expected_uuid = _harness_session_id_as_uuid(session.id)
    assert discovery.claude_calls == [
        {"session_id": expected_uuid, "cwd": Path("/workspace/project")}
    ]
    assert observer.bindings == [(claude_path, session.id)]
    # On terminal-state cleanup, claude's bound path is unbound — keeps
    # observer._path_to_session bounded across the harness lifetime.
    assert observer.unbindings == [claude_path]


# --- Phase 2: supervisor stops emitting message.delta; new process.stderr -----


class _ExpectationRecordingObserver:
    """Phase 2 stand-in for ``ExternalTranscriptObserver``. Replaces the
    Phase 1 ``bind_rollout`` + ``unbind_rollout`` orchestrator surface
    with ``expect_codex_rollout`` for the new spawn-side contract."""

    def __init__(self) -> None:
        self.expectations: list[tuple[object, str]] = []
        # Phase 1 plumbing — claude pre-bind is unchanged in Phase 2.
        self.bindings: list[tuple[object, str]] = []
        self.unbindings: list[object] = []

    def expect_codex_rollout(self, *, cwd, session_id: str) -> None:
        self.expectations.append((cwd, session_id))

    def bind_rollout(self, path, session_id: str) -> None:
        self.bindings.append((path, session_id))

    def unbind_rollout(self, path) -> None:
        self.unbindings.append(path)


@pytest.mark.asyncio
async def test_run_process_emits_no_message_delta_for_stdout() -> None:
    """After Phase 2, the supervisor's stdout pump stops emitting
    ``message.delta`` entirely. The observer is the sole writer for
    ``message``-shaped events. Stdout lines that aren't end-turn signals
    are diagnostic-only and don't travel the event bus."""
    bus = InMemoryEventBus()
    session = make_session()  # codex
    run = make_run(session)
    process = FakeProcess(stdout=[b"some noise\n", b"more noise\n"], returncode=0)
    factory = FakeFactory(process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "hi"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
        ).run()
    )
    await asyncio.sleep(0)
    process.finish()
    await asyncio.wait_for(task, timeout=1)

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert all(e.event != "message.delta" for e in events), [e.event for e in events]


@pytest.mark.asyncio
async def test_run_process_emits_process_stderr_for_stderr_lines() -> None:
    """Stderr lines are forwarded as ``process.stderr`` events with a
    ``{text: <line>}`` payload — diagnostic firehose, not parsed."""
    bus = InMemoryEventBus()
    session = make_session()
    run = make_run(session)
    process = FakeProcess(stderr=[b"warning A\n", b"warning B\n"], returncode=0)
    factory = FakeFactory(process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "hi"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
        ).run()
    )
    await asyncio.sleep(0)
    process.finish()
    await asyncio.wait_for(task, timeout=1)

    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    stderr_events = [e for e in events if e.event == "process.stderr"]
    assert [e.data["text"] for e in stderr_events] == ["warning A", "warning B"]
    # message.delta MUST NOT be emitted from stderr either.
    assert all(e.event != "message.delta" for e in events)


@pytest.mark.asyncio
async def test_run_process_still_detects_end_turn_for_watchdog_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-turn signal that drives the post-end_turn cleanup
    watchdog must still fire when codex emits ``turn.completed`` on
    stdout. (Phase 3 will rewire this to come from the observer; for
    now the stdout signal is what we have.)"""
    import json as _json

    from agent_harness.orchestrator import END_TURN_GRACE_SECONDS, END_TURN_HARD_KILL_AFTER_SECONDS

    bus = InMemoryEventBus()
    session = make_session("codex")
    run = make_run(session)
    process = FakeProcess(
        stdout=[(_json.dumps({"type": "turn.completed"}) + "\n").encode()],
        returncode=0,
        exit_on_sigterm=True,
    )
    factory = FakeFactory(process)
    clock = FakeClock()
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("codex", "exec", "hi"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    # Let the stdout line be ingested.
    for _ in range(5):
        await asyncio.sleep(0)
    # Past the post-end_turn grace window — SIGTERM should fire.
    clock.advance(END_TURN_GRACE_SECONDS + 1)
    await asyncio.wait_for(task, timeout=2)

    assert signal.SIGTERM in process.group_signals


@pytest.mark.asyncio
async def test_run_process_still_detects_end_turn_for_watchdog_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same watchdog signal for claude (``result`` + stop_reason=end_turn)."""
    import json as _json

    from agent_harness.orchestrator import END_TURN_GRACE_SECONDS

    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    process = FakeProcess(
        stdout=[
            (_json.dumps({"type": "result", "stop_reason": "end_turn"}) + "\n").encode()
        ],
        returncode=0,
        exit_on_sigterm=True,
    )
    factory = FakeFactory(process)
    clock = FakeClock()
    install_fake_process_group(monkeypatch, process)

    task = asyncio.create_task(
        RunProcess(
            session=session,
            run=run,
            command=ProcessCommand(argv=("claude", "--print"), cwd="/workspace/project"),
            event_bus=bus,
            process_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        ).run()
    )
    for _ in range(5):
        await asyncio.sleep(0)
    clock.advance(END_TURN_GRACE_SECONDS + 1)
    await asyncio.wait_for(task, timeout=2)

    assert signal.SIGTERM in process.group_signals


@pytest.mark.asyncio
async def test_pre_bind_codex_registers_expectation() -> None:
    """In Phase 2, the codex pre-bind step is a one-line synchronous
    ``observer.expect_codex_rollout(cwd, session_id)`` call. No psutil
    fd probe, no background task. Registration happens before (or
    around) spawn — the observer matches incoming rollouts by content."""
    bus = InMemoryEventBus()
    session = make_session("codex")
    run = make_run(session)
    process = FakeProcess(returncode=0)
    factory = FakeFactory(process)
    observer = _ExpectationRecordingObserver()

    rp = RunProcess(
        session=session,
        run=run,
        command=ProcessCommand(argv=("codex", "exec", "hi"), cwd="/workspace/project"),
        event_bus=bus,
        process_factory=factory,
        observer=observer,
    )
    task = asyncio.create_task(rp.run())
    process.finish()
    await asyncio.wait_for(task, timeout=1)

    assert observer.expectations == [
        (Path("/workspace/project"), session.id)
    ]
    # No bind/unbind for codex anymore — the expectation registry
    # replaces the bind_rollout codepath for codex spawns.
    assert observer.bindings == []
    assert observer.unbindings == []


@pytest.mark.asyncio
async def test_stream_lines_updates_last_activity_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2 dropped ``message.delta`` from ``_ACTIVITY_EVENTS``;
    observer-emitted ``message`` events don't go through
    ``RunProcess._publish`` so they can't tick the idle watchdog.
    Claude harness runs talk via the rollout (rarely via stderr) and
    would trip the 30-min SIGTERM without a stdout heartbeat.

    Fix: each non-empty stdout line ticks ``_last_activity_at``
    directly (no publish — diagnostic-only signal). Verify the
    timestamp advances across stdout lines, and that no event is
    emitted as a side effect.
    """
    bus = InMemoryEventBus()
    session = make_session("claude-code")
    run = make_run(session)
    process = FakeProcess(
        stdout=[b"alpha\n", b"beta\n", b"gamma\n"],
        returncode=0,
    )
    factory = FakeFactory(process)
    clock = FakeClock()

    rp = RunProcess(
        session=session,
        run=run,
        command=ProcessCommand(argv=("claude", "--print"), cwd="/workspace/project"),
        event_bus=bus,
        process_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    initial = rp._last_activity_at

    # Advance the clock BEFORE the stream pump runs so the heartbeat
    # tick (which reads ``clock()``) lands on a different value.
    clock.advance(5)
    task = asyncio.create_task(rp.run())
    for _ in range(8):
        await asyncio.sleep(0)

    # Activity timestamp advanced past the initial value, and the
    # last-activity event marker reflects the stdout heartbeat path.
    assert rp._last_activity_at > initial
    assert rp._last_activity_event == "stdout"

    process.finish()
    await asyncio.wait_for(task, timeout=1)

    # No ``message.delta`` events fired as a side effect of the
    # stdout heartbeat — observer is still the sole message writer.
    events = [e for e in await bus.replay(session_id=session.id) if e.run_id == run.id]
    assert all(e.event != "message.delta" for e in events), [e.event for e in events]
