from uuid import UUID

import pytest
from pydantic import ValidationError

from agent_harness.models import (
    BackendCapabilities,
    CreateSessionRequest,
    Event,
    Message,
    Run,
    Session,
    TextBlock,
    ToolCallBlock,
)


def test_session_defaults_and_message_blocks_are_normalized() -> None:
    session = Session(
        backend_id="codex",
        title="Scaffold core",
        messages=[
            Message.user("hello"),
            Message(role="assistant", blocks=[ToolCallBlock(name="pytest", input={"args": ["-q"]})]),
        ],
    )

    assert isinstance(session.id, UUID)
    assert session.status == "active"
    assert session.backend_id == "codex"
    assert session.messages[0].blocks == [TextBlock(text="hello")]
    assert session.messages[1].blocks[0].type == "tool_call"


def test_models_reject_unknown_fields_and_invalid_backend_ids() -> None:
    with pytest.raises(ValidationError):
        CreateSessionRequest(backend_id="unknown")

    with pytest.raises(ValidationError):
        Message(role="user", blocks=[{"type": "text", "text": "hello"}], unexpected=True)


def test_event_requires_positive_sequence_when_supplied() -> None:
    with pytest.raises(ValidationError):
        Event(seq=0, type="session.created", data={"id": "abc"})


def test_backend_capabilities_capture_v1_shape() -> None:
    capabilities = BackendCapabilities(
        launch_modes=["orchestrated", "observed"],
        supports_sse=True,
        supports_transcript_observation=True,
        supports_working_directory=True,
        supports_resume=True,
    )

    assert capabilities.supports_sse is True
    assert capabilities.launch_modes == ["orchestrated", "observed"]


def test_run_model_defaults_to_queued_lifecycle() -> None:
    session = Session(backend_id="claude-code")
    run = Run(session_id=session.id, backend_id=session.backend_id, command=["claude", "--print"])

    assert run.status == "queued"
    assert run.session_id == session.id
    assert run.command == ["claude", "--print"]
