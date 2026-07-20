import pytest
from pydantic import ValidationError

from agent_harness.models import (
    BackendCapabilities,
    CreateRunRequest,
    CreateSessionRequest,
    Event,
    ForkSessionRequest,
    ForkSessionResponse,
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


def test_session_bypass_permissions_defaults_false_and_round_trips() -> None:
    session = Session(
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/tmp/proj", name="proj"),
    )
    assert session.bypass_permissions is False

    yolo = session.model_copy(update={"bypass_permissions": True})
    assert yolo.bypass_permissions is True


def test_create_session_request_accepts_bypass_permissions() -> None:
    request = CreateSessionRequest(
        backend="claude-code",
        model="claude-opus-4-7",
        project=Project(path="/tmp/proj", name="proj"),
        bypass_permissions=True,
    )
    assert request.bypass_permissions is True


def test_models_accept_pi_backend() -> None:
    """pi is a first-class backend alongside claude-code and codex:
    Session and CreateSessionRequest accept ``backend="pi"`` once the
    BackendName literal includes it (no other model edits needed)."""
    session = Session(
        backend="pi",
        model="gpt-5.4",
        project=Project(path="/tmp/proj", name="proj"),
    )
    assert session.backend == "pi"

    request = CreateSessionRequest(
        backend="pi",
        model="gpt-5.4",
        project=Project(path="/tmp/proj", name="proj"),
    )
    assert request.backend == "pi"


def test_session_and_request_model_is_optional() -> None:
    # A session may be created without a model — the backend CLI falls back to
    # its own configured default (pi callers omit it). ``None`` is allowed;
    # an empty string is still rejected (min_length=1 when a value is given).
    request = CreateSessionRequest(
        backend="pi", project=Project(path="/tmp/proj", name="proj"),
    )
    assert request.model is None

    session = Session(backend="pi", project=Project(path="/tmp/proj", name="proj"))
    assert session.model is None

    with pytest.raises(ValidationError):
        CreateSessionRequest(
            backend="pi", model="", project=Project(path="/tmp/proj", name="proj"),
        )


def test_session_carries_forked_from() -> None:
    # forked_from records the parent session a fork descends from; defaults None.
    session = Session(backend="claude-code", project=Project(path="/tmp/p", name="p"))
    assert session.forked_from is None

    forked = session.model_copy(update={"forked_from": "ses_parent"})
    assert forked.forked_from == "ses_parent"


def test_fork_session_request_message_and_title_optional() -> None:
    empty = ForkSessionRequest()
    assert empty.message is None
    assert empty.title is None

    full = ForkSessionRequest(message="hi", title="Thread reply")
    assert full.message == "hi"
    assert full.title == "Thread reply"

    # empty strings rejected (min_length=1) — same guard as PatchSessionRequest.
    with pytest.raises(ValidationError):
        ForkSessionRequest(message="")
    with pytest.raises(ValidationError):
        ForkSessionRequest(title="")


def test_fork_session_response_carries_session_and_optional_run() -> None:
    session = Session(backend="claude-code", project=Project(path="/tmp/p", name="p"))
    resp = ForkSessionResponse(session=session)
    assert resp.run is None

    run = Run(session_id=session.id, status="running")
    with_run = ForkSessionResponse(session=session, run=run)
    assert with_run.run is not None
    assert with_run.run.id == run.id


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


def test_create_run_request_requires_message_text() -> None:
    with pytest.raises(ValidationError):
        CreateRunRequest(message="")


def test_session_carries_codex_resume_id_default_none() -> None:
    """Session.codex_resume_id defaults to None and round-trips. Set
    by the observer once a codex rollout's UUID is extractable from
    its filename; used by CodexCommandBuilder to pick exec resume
    over fresh exec. Spec: specs/2026-05-21-codex-resume.md"""
    session = Session(
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/tmp/proj", name="proj"),
    )
    assert session.codex_resume_id is None

    resumed = session.model_copy(
        update={"codex_resume_id": "019e0500-0000-0000-0000-000000000000"}
    )
    assert resumed.codex_resume_id == "019e0500-0000-0000-0000-000000000000"
    # Round-trip through JSON to confirm the field is part of the wire shape.
    payload = resumed.model_dump(mode="json")
    assert payload["codex_resume_id"] == "019e0500-0000-0000-0000-000000000000"
    rehydrated = Session.model_validate(payload)
    assert rehydrated.codex_resume_id == resumed.codex_resume_id
