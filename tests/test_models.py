import pytest
from pydantic import ValidationError

from agent_harness.models import (
    BackendCapabilities,
    CreateSessionRequest,
    Event,
    Message,
    Project,
    Run,
    Session,
    TextBlock,
    ToolUseBlock,
)


def test_session_defaults_and_message_blocks_match_contract() -> None:
    session = Session(
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/tmp/proj", name="proj"),
        title="Scaffold core",
    )
    message = Message(
        role="assistant",
        blocks=[ToolUseBlock(name="pytest", input={"args": ["-q"]})],
        model=session.model,
    )

    assert session.id.startswith("ses_")
    assert session.status == "idle"
    assert session.origin == "harness"
    assert session.backend == "codex"
    assert Message.user("hello").blocks == [TextBlock(text="hello")]
    assert message.blocks[0].type == "tool_use"


def test_models_reject_unknown_fields_and_invalid_backend_names() -> None:
    with pytest.raises(ValidationError):
        CreateSessionRequest(backend="unknown", model="x", project={"path": "/tmp", "name": "tmp"})

    with pytest.raises(ValidationError):
        Message(role="user", blocks=[{"type": "text", "text": "hello"}], unexpected=True)


def test_event_requires_positive_sequence_when_supplied() -> None:
    with pytest.raises(ValidationError):
        Event(sequence=0, event="session.updated", data={"id": "abc"})


def test_backend_capabilities_capture_v1_shape() -> None:
    capabilities = BackendCapabilities(
        fork=True,
        subagents=False,
        permission_detection=True,
        interactive_pty=True,
        stream_json=True,
        structured_output=True,
        session_id_choice=True,
        max_budget=True,
        mcp=True,
        sandbox=None,
        tools="granular",
    )

    assert capabilities.tools == "granular"
    assert capabilities.interrupt_external_runs is False


def test_run_model_defaults_to_queued_lifecycle() -> None:
    session = Session(backend="claude-code", model="opus", project=Project(path="/tmp/proj", name="proj"))
    run = Run(session_id=session.id, input_message_id="msg_input")

    assert run.id.startswith("run_")
    assert run.status == "queued"
    assert run.origin == "harness"
    assert run.session_id == session.id
