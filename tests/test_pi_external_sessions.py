"""External (terminal-launched) pi session discovery + headless resume.

pi splits the facts the harness needs to synthesize a session across
record types — ``session`` carries the cwd, ``model_change`` carries
provider + model id — so unlike claude/codex no single record can build
a ``session.updated``. These tests lock in the accumulating discovery
path and the resume guard that keeps ``POST /v1/runs`` from silently
starting a *new* pi conversation.

See specs/2026-09-14-external-pi-sessions.md.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from agent_harness.api import create_app
from agent_harness.events import InMemoryEventBus
from agent_harness.models import Message, Project, Run, Session
from agent_harness.observer import (
    ExternalTranscriptObserver,
    parse_transcript_record,
    pi_transcript_header_uuid,
    pi_transcript_path,
    transcript_identity_from_path,
)
from agent_harness import orchestrator
from agent_harness.orchestrator import (
    CommandBuildError,
    PiCommandBuilder,
    SubmitResult,
    pi_resume_target,
    validate_session_resume_target,
)
from agent_harness.pi_discovery import (
    PiSessionFacts,
    PiTranscriptRegistry,
    pi_facts_from_record,
    read_pi_head_facts,
)
from agent_harness.repository import InMemoryRepository

UUID = "e5a93149-9a70-4aef-a189-2681b4e08525"
SESSION_ID = "ses_e5a931499a704aefa1892681b4e08525"
CWD = "/home/me/project"
TS = "2026-09-14T11-25-51-562Z"

SESSION_RECORD = {"type": "session", "version": 3, "id": UUID, "cwd": CWD}
MODEL_CHANGE = {"type": "model_change", "provider": "ollama", "modelId": "glm-5.2:cloud"}
USER_RECORD = {
    "type": "message",
    "message": {"role": "user", "content": [{"type": "text", "text": "early question"}]},
}
ASSISTANT_RECORD = {
    "type": "message",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "an answer"}],
        "provider": "ollama",
        "model": "glm-5.2:cloud",
        "stopReason": "stop",
    },
}


def _identity(path: Path):
    return transcript_identity_from_path(path)


def _transcript(home: Path, *, cwd: str = CWD, uuid: str = UUID, ts: str = TS) -> Path:
    path = pi_transcript_path(cwd, ts, uuid, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Fact extraction                                                              #
# --------------------------------------------------------------------------- #


def test_facts_from_session_record_carry_cwd_only() -> None:
    assert pi_facts_from_record(SESSION_RECORD) == PiSessionFacts(cwd=CWD)


def test_facts_from_model_change_carry_provider_and_model() -> None:
    assert pi_facts_from_record(MODEL_CHANGE) == PiSessionFacts(
        provider="ollama", model="glm-5.2:cloud"
    )


def test_facts_from_assistant_message_carry_provider_and_model() -> None:
    """pi stamps provider+model on every assistant record — the fallback
    when the transcript's ``model_change`` was written before the
    observer's offset (verified against pi v0.84.2 output 2026-09-14)."""
    assert pi_facts_from_record(ASSISTANT_RECORD) == PiSessionFacts(
        provider="ollama", model="glm-5.2:cloud"
    )


@pytest.mark.parametrize(
    "record",
    [
        {"type": "session", "cwd": ""},
        {"type": "session", "cwd": 17},
        {"type": "session"},
        {"type": "model_change", "provider": None, "modelId": []},
        {"type": "message", "message": "not-a-mapping"},
        {"type": "message"},
        {},
    ],
)
def test_facts_from_malformed_records_are_empty(record) -> None:
    assert pi_facts_from_record(record) == PiSessionFacts()


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #


def test_registry_announces_as_soon_as_cwd_is_known_even_without_model() -> None:
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, SESSION_RECORD)

    announced = registry.take_announcement(path)
    assert announced == PiSessionFacts(cwd=CWD)


def test_registry_withholds_announcement_until_cwd_is_known() -> None:
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, MODEL_CHANGE)

    assert registry.take_announcement(path) is None


def test_registry_re_announces_when_the_model_arrives() -> None:
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, SESSION_RECORD)
    assert registry.take_announcement(path) == PiSessionFacts(cwd=CWD)

    registry.observe(path, MODEL_CHANGE)
    assert registry.take_announcement(path) == PiSessionFacts(
        cwd=CWD, provider="ollama", model="glm-5.2:cloud"
    )


def test_registry_does_not_re_announce_unchanged_facts() -> None:
    """Duplicate ingestion (watchfiles double-fire, observer restart
    re-tail) must not produce a second session.updated."""
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, SESSION_RECORD)
    registry.observe(path, MODEL_CHANGE)
    assert registry.take_announcement(path) is not None

    registry.observe(path, SESSION_RECORD)
    registry.observe(path, MODEL_CHANGE)
    registry.observe(path, USER_RECORD)
    assert registry.take_announcement(path) is None


def test_registry_never_regresses_a_known_model_to_none() -> None:
    """Facts are sticky: a later record that carries no model must not
    blank the one we already learned (the repository's external-origin
    upsert overwrites, so a None would wipe it)."""
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, SESSION_RECORD)
    registry.observe(path, MODEL_CHANGE)
    registry.observe(path, USER_RECORD)

    assert registry.facts(path) == PiSessionFacts(
        cwd=CWD, provider="ollama", model="glm-5.2:cloud"
    )


def test_registry_tracks_a_model_switch_mid_session() -> None:
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")

    registry.observe(path, SESSION_RECORD)
    registry.observe(path, MODEL_CHANGE)
    registry.take_announcement(path)
    registry.observe(
        path, {"type": "model_change", "provider": "anthropic", "modelId": "claude-opus-5"}
    )

    assert registry.take_announcement(path) == PiSessionFacts(
        cwd=CWD, provider="anthropic", model="claude-opus-5"
    )


def test_registry_is_bounded_and_evicts_least_recently_used() -> None:
    registry = PiTranscriptRegistry(max_entries=2)
    a, b, c = Path("/t/a.jsonl"), Path("/t/b.jsonl"), Path("/t/c.jsonl")

    registry.observe(a, SESSION_RECORD)
    registry.observe(b, SESSION_RECORD)
    registry.observe(a, MODEL_CHANGE)  # touches ``a``, making ``b`` the LRU
    registry.observe(c, SESSION_RECORD)

    assert len(registry) == 2
    assert registry.facts(b) == PiSessionFacts()
    assert registry.facts(a).model == "glm-5.2:cloud"
    assert registry.facts(c).cwd == CWD


def test_registry_forget_drops_an_entry() -> None:
    registry = PiTranscriptRegistry()
    path = Path("/t/a.jsonl")
    registry.observe(path, SESSION_RECORD)

    registry.forget(path)

    assert registry.facts(path) == PiSessionFacts()


# --------------------------------------------------------------------------- #
# Head peek (restart / pre-existing transcripts)                               #
# --------------------------------------------------------------------------- #


def test_read_pi_head_facts_recovers_cwd_provider_and_model(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, USER_RECORD])

    assert read_pi_head_facts(transcript) == PiSessionFacts(
        cwd=CWD, provider="ollama", model="glm-5.2:cloud"
    )


def test_read_pi_head_facts_is_bounded_by_max_lines(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [USER_RECORD] * 10 + [SESSION_RECORD])

    assert read_pi_head_facts(transcript, max_lines=3) == PiSessionFacts()


def test_read_pi_head_facts_tolerates_missing_and_malformed_files(tmp_path) -> None:
    assert read_pi_head_facts(tmp_path / "nope.jsonl") == PiSessionFacts()

    broken = _transcript(tmp_path)
    broken.write_text("{not json\n[]\n" + json.dumps(SESSION_RECORD) + "\n", encoding="utf-8")
    assert read_pi_head_facts(broken) == PiSessionFacts(cwd=CWD)


# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #


def test_parser_synthesizes_external_pi_session_from_the_session_record(tmp_path) -> None:
    identity = _identity(_transcript(tmp_path))
    registry = PiTranscriptRegistry()

    events = parse_transcript_record(
        SESSION_RECORD, identity=identity, offset=0, pi_registry=registry
    )

    assert [e.event for e in events] == ["session.updated"]
    session = events[0].data["session"]
    assert session["origin"] == "external"
    assert session["backend"] == "pi"
    assert session["project"]["path"] == CWD
    # Model stays unknown until ``model_change`` — the session is usable
    # before pi discloses it (``--model`` is simply omitted on resume).
    assert session["model"] is None
    # The transcript we are reading is the resume key — recorded up front,
    # because it cannot be reconstructed from the session id later.
    assert session["project"]["path"] == CWD
    assert session["pi_transcript_path"] == str(identity.path)


def test_parser_fills_model_and_provider_on_the_model_change_record(tmp_path) -> None:
    identity = _identity(_transcript(tmp_path))
    registry = PiTranscriptRegistry()

    parse_transcript_record(SESSION_RECORD, identity=identity, offset=0, pi_registry=registry)
    events = parse_transcript_record(
        MODEL_CHANGE, identity=identity, offset=1, pi_registry=registry
    )

    session = events[0].data["session"]
    # Provider-qualified: this string is the WHOLE model representation,
    # and the harness never emits --provider alongside it.
    assert session["model"] == "ollama/glm-5.2:cloud"


def test_parser_emits_session_then_message_for_an_early_user_turn(tmp_path) -> None:
    """The user turn that arrives BEFORE ``model_change`` must still be
    emitted — it is the message the operator actually typed."""
    identity = _identity(_transcript(tmp_path))
    registry = PiTranscriptRegistry()

    parse_transcript_record(SESSION_RECORD, identity=identity, offset=0, pi_registry=registry)
    events = parse_transcript_record(
        USER_RECORD, identity=identity, offset=1, pi_registry=registry
    )

    assert [e.event for e in events] == ["message"]
    assert events[0].data["message"]["blocks"][0]["text"] == "early question"


def test_parser_without_a_registry_keeps_the_pre_existing_shape(tmp_path) -> None:
    """Back-compat: callers that don't opt in see exactly what they saw
    before — metadata records stay silent."""
    identity = _identity(_transcript(tmp_path))

    assert parse_transcript_record(SESSION_RECORD, identity=identity, offset=0) == []
    assert parse_transcript_record(MODEL_CHANGE, identity=identity, offset=1) == []


def test_parser_suppresses_synthesis_for_a_rebound_identity(tmp_path) -> None:
    identity = _identity(_transcript(tmp_path))
    rebound = type(identity)(
        backend=identity.backend,
        path=identity.path,
        session_id=identity.session_id,
        is_rebound=True,
    )
    registry = PiTranscriptRegistry()

    assert (
        parse_transcript_record(
            SESSION_RECORD, identity=rebound, offset=0, pi_registry=registry
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Observer integration                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_observer_discovers_an_external_pi_session_and_all_its_messages(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, USER_RECORD, MODEL_CHANGE, ASSISTANT_RECORD])
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    published = await observer.tail_file(transcript)

    # One announcement, not one per record: the head peek on first sight of
    # the transcript already yields cwd + provider + model together.
    assert [e.event for e in published] == ["session.updated", "message", "message"]
    session = repository.get_session(SESSION_ID)
    assert session.origin == "external"
    assert session.backend == "pi"
    assert session.project.path == CWD
    assert session.model == "ollama/glm-5.2:cloud"
    # The user turn predates ``model_change`` — it must not be lost.
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_observer_announces_a_live_pi_session_before_its_model_is_known(tmp_path) -> None:
    """The realistic split: pi creates the transcript, the operator's first
    turn lands, and only then does ``model_change`` appear. Both the
    session and that first turn must surface, and the model must be filled
    in afterwards rather than stranding the session at ``None``."""
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, USER_RECORD])
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    first = await observer.tail_file(transcript)

    assert [e.event for e in first] == ["session.updated", "message"]
    assert repository.get_session(SESSION_ID).model is None
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["user"]

    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(MODEL_CHANGE) + "\n")
        fh.write(json.dumps(ASSISTANT_RECORD) + "\n")
    second = await observer.tail_file(transcript)

    assert [e.event for e in second] == ["session.updated", "message"]
    session = repository.get_session(SESSION_ID)
    assert session.model == "ollama/glm-5.2:cloud"
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_observer_leaves_a_harness_owned_pi_session_alone(tmp_path) -> None:
    """Existing owned-pi behavior is unchanged: no origin downgrade, and
    no synthetic session.updated on the bus for subscribers to misread."""
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, ASSISTANT_RECORD])
    repository = InMemoryRepository()
    repository.upsert_session(
        Session(
            id=SESSION_ID,
            backend="pi",
            model="claude-opus-5",
            project=Project(path=CWD, name="project"),
            origin="harness",
            status="running",
        )
    )
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    published = await observer.tail_file(transcript)

    assert [e.event for e in published if e.event == "session.updated"] == []
    session = repository.get_session(SESSION_ID)
    assert session.origin == "harness"
    assert session.model == "claude-opus-5"
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["assistant"]


@pytest.mark.asyncio
async def test_observer_backfills_pi_facts_from_the_file_head_after_a_restart(tmp_path) -> None:
    """Offsets are persisted but the in-memory registry is not: a restart
    resumes mid-file, past the ``session`` record. The observer recovers
    the cwd by peeking the head rather than dropping the turn."""
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE])
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    # Simulate the pre-restart process having consumed the head.
    observer._state.set_next_offset(transcript, transcript.stat().st_size)

    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ASSISTANT_RECORD) + "\n")
    published = await observer.tail_file(transcript)

    assert "session.updated" in [e.event for e in published]
    session = repository.get_session(SESSION_ID)
    assert session.project.path == CWD
    assert session.model == "ollama/glm-5.2:cloud"
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["assistant"]


@pytest.mark.asyncio
async def test_observer_skips_pi_messages_when_the_transcript_never_states_a_cwd(tmp_path) -> None:
    """A transcript with no usable ``session`` record can't be located on
    disk for resume, so synthesizing it would produce a dead session.
    Skip (never buffer — the operator has many interactive pi sessions)."""
    transcript = _transcript(tmp_path)
    _write(transcript, [{"type": "session", "id": UUID}, ASSISTANT_RECORD])
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    published = await observer.tail_file(transcript)

    assert [e.event for e in published] == []
    assert observer._pending_materialization == {}


@pytest.mark.asyncio
async def test_observer_does_not_duplicate_session_updated_on_a_re_tail(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, ASSISTANT_RECORD])
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    await observer.tail_file(transcript)
    observer._state.set_next_offset(transcript, 0)
    republished = await observer.tail_file(transcript)

    assert [e.event for e in republished if e.event == "session.updated"] == []


@pytest.mark.asyncio
async def test_observer_tolerates_malformed_pi_records_without_dropping_the_session(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    transcript.write_text(
        "\n".join(
            [
                "{ this is not json",
                "[]",
                json.dumps(SESSION_RECORD),
                json.dumps({"type": "message", "message": {"role": "assistant"}}),
                json.dumps(ASSISTANT_RECORD),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    await observer.tail_file(transcript)

    assert repository.get_session(SESSION_ID).project.path == CWD
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["assistant"]


# --------------------------------------------------------------------------- #
# Resume: locating the conversation, and refusing to fake one                  #
# --------------------------------------------------------------------------- #


def _external_pi_session(
    cwd: str = CWD, transcript_path: str | Path | None = None
) -> Session:
    return Session(
        id=SESSION_ID,
        backend="pi",
        # Provider-qualified — the whole model representation. pi infers the
        # provider from the prefix only when ``--provider`` is absent, so the
        # harness never emits that flag and this string cannot go stale
        # against a second, separately-stored provider.
        model="ollama/glm-5.2:cloud",
        project=Project(path=cwd, name=Path(cwd).name),
        origin="external",
        pi_transcript_path=str(transcript_path) if transcript_path else None,
    )


def _seed_transcript(tmp_path: Path, *, uuid: str = UUID, ts: str = TS) -> Path:
    """A transcript whose header identifies ``uuid`` — what pi wrote."""
    transcript = _transcript(tmp_path, uuid=uuid, ts=ts)
    _write(transcript, [{"type": "session", "version": 3, "id": uuid, "cwd": CWD}])
    return transcript


def test_transcript_header_uuid_reads_the_session_record(tmp_path) -> None:
    assert pi_transcript_header_uuid(_seed_transcript(tmp_path)) == UUID


@pytest.mark.parametrize(
    "body",
    [
        "",
        "\n",
        "{ not json\n",
        '{"type":"message","message":{"role":"user"}}\n',
        '{"type":"session","version":3}\n',
        "[]\n",
    ],
    ids=["empty", "blank", "malformed", "not-a-header", "header-without-id", "not-a-mapping"],
)
def test_transcript_header_uuid_is_none_for_anything_else(tmp_path, body) -> None:
    transcript = _transcript(tmp_path)
    transcript.write_text(body, encoding="utf-8")

    assert pi_transcript_header_uuid(transcript) is None


def test_transcript_header_uuid_is_none_for_a_missing_file(tmp_path) -> None:
    assert pi_transcript_header_uuid(tmp_path / "gone.jsonl") is None


def test_resume_target_is_the_observed_transcript(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)

    assert pi_resume_target(_external_pi_session(transcript_path=transcript)) == transcript


def test_resume_target_is_none_when_the_observer_never_recorded_a_path() -> None:
    assert pi_resume_target(_external_pi_session()) is None


def test_resume_target_is_none_when_the_transcript_is_gone(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)
    session = _external_pi_session(transcript_path=transcript)
    transcript.unlink()

    assert pi_resume_target(session) is None


def test_resume_target_rejects_a_transcript_for_a_different_conversation(tmp_path) -> None:
    """The whole reason the header is read rather than the filename: a file
    can be moved, truncated or replaced under the same name, and pi will
    create a fresh conversation at whatever path it is handed."""
    other = "11111111-2222-3333-4444-555555555555"
    transcript = _transcript(tmp_path)
    _write(transcript, [{"type": "session", "version": 3, "id": other, "cwd": CWD}])

    assert pi_resume_target(_external_pi_session(transcript_path=transcript)) is None


def test_resume_target_is_none_for_a_non_uuid_session_id(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)
    session = _external_pi_session(transcript_path=transcript).model_copy(
        update={"id": "ses_not-uuid-shaped"}
    )

    assert pi_resume_target(session) is None


def test_validate_resume_target_accepts_external_pi_with_its_transcript(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)

    validate_session_resume_target(_external_pi_session(transcript_path=transcript))


def test_validate_resume_target_rejects_external_pi_without_a_transcript(tmp_path) -> None:
    """Without a verified transcript pi would write a brand-new conversation
    at whatever it is pointed at and answer with no memory of the thread
    (rc=0, verified pi v0.84.2). A truthful 409 beats that."""
    with pytest.raises(CommandBuildError) as excinfo:
        validate_session_resume_target(_external_pi_session())

    detail = str(excinfo.value)
    assert SESSION_ID in detail
    assert "never observed" in detail
    assert "new one" in detail


def test_validate_resume_target_ignores_harness_pi_sessions() -> None:
    session = _external_pi_session().model_copy(update={"origin": "harness"})

    validate_session_resume_target(session)


def test_pi_command_builder_resumes_an_external_session_by_observed_path(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)
    session = _external_pi_session(transcript_path=transcript)

    command = PiCommandBuilder().build(
        session=session, run=Run(session_id=session.id), message=Message.user("continue please")
    )

    assert command.argv == (
        "pi",
        "-p",
        "--model",
        "ollama/glm-5.2:cloud",
        "--session",
        str(transcript),
        "continue please",
    )
    # Never --provider: pi ignores a `provider/` prefix when --provider is
    # given, so pinning both lets a stale provider outrank the model.
    assert "--provider" not in command.argv
    assert command.cwd == CWD


def test_pi_command_builder_still_uses_session_id_for_harness_sessions() -> None:
    """Harness-origin sessions own their id, so create-if-missing is exactly
    what the first run wants. Unchanged by external-pi support."""
    session = _external_pi_session().model_copy(update={"origin": "harness"})

    command = PiCommandBuilder().build(
        session=session, run=Run(session_id=session.id), message=Message.user("go")
    )

    assert command.argv[-3:] == ("--session-id", UUID, "go")


# --------------------------------------------------------------------------- #
# POST /v1/runs — the end-to-end acceptance for headless continuation          #
# --------------------------------------------------------------------------- #


class _RecordingRunManager:
    """RunManager double that accepts every submission and keeps the
    command, so a test can assert on the argv pi would actually be run
    with. Same contract as test_api's ``_AcceptingRunManager``."""

    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, *, session, run, command, on_start=None):
        self.submitted.append(command)
        if on_start is not None:
            on_start()
        return SubmitResult(accepted=True, status="running")


def _app_with_external_pi(
    repo: InMemoryRepository, manager, transcript: Path | None = None
) -> TestClient:
    repo.upsert_session(_external_pi_session(transcript_path=transcript))
    return TestClient(create_app(repository=repo, run_manager=manager))


def test_run_on_an_external_pi_session_resumes_the_same_conversation(tmp_path) -> None:
    transcript = _seed_transcript(tmp_path)
    repo = InMemoryRepository()
    manager = _RecordingRunManager()
    client = _app_with_external_pi(repo, manager, transcript)

    response = client.post(f"/v1/sessions/{SESSION_ID}/runs", json={"message": "carry on"})

    assert response.status_code == 202
    argv = manager.submitted[0].argv
    # Same conversation: the exact file the observer read, not a path
    # rebuilt from the session id.
    assert argv[argv.index("--session") + 1] == str(transcript)
    assert "--session-id" not in argv
    # Headless — never a TUI, which would fork the operator's transcript.
    assert "-p" in argv
    assert manager.submitted[0].cwd == CWD


def test_run_on_an_external_pi_session_409s_when_the_transcript_is_gone(tmp_path) -> None:
    """The whole point of the guard: pi creates a conversation at any path
    it is handed, so a vanished transcript must be refused rather than
    answered from a blank thread."""
    transcript = _seed_transcript(tmp_path)
    repo = InMemoryRepository()
    manager = _RecordingRunManager()
    client = _app_with_external_pi(repo, manager, transcript)
    transcript.unlink()

    response = client.post(f"/v1/sessions/{SESSION_ID}/runs", json={"message": "carry on"})

    assert response.status_code == 409
    assert "silently start a new one" in response.json()["detail"]
    assert manager.submitted == []
    # No half-created run or orphaned message left behind.
    assert repo.list_runs(SESSION_ID) == []
    assert repo.list_messages(SESSION_ID) == []


def test_run_on_an_external_pi_session_409s_when_the_transcript_is_a_stranger(
    tmp_path,
) -> None:
    """A file at the recorded path whose header names a different session —
    moved, replaced or reused. Filename-shaped checks would accept it."""
    transcript = _seed_transcript(tmp_path)
    repo = InMemoryRepository()
    manager = _RecordingRunManager()
    client = _app_with_external_pi(repo, manager, transcript)
    _write(
        transcript,
        [{"type": "session", "version": 3, "id": "11111111-2222-3333-4444-555555555555"}],
    )

    response = client.post(f"/v1/sessions/{SESSION_ID}/runs", json={"message": "carry on"})

    assert response.status_code == 409
    assert manager.submitted == []


# --------------------------------------------------------------------------- #
# Metadata refresh — regressions found in independent review (2026-09-14)      #
#                                                                             #
# Adapted from the reviewer's reproducers. An observation is a snapshot of    #
# the CONVERSATION; it must not rewrite the session's identity or the user's  #
# settings. All three failed before ``merge_observed_session`` and the        #
# persisted-facts-before-head-peek hydration order.                           #
# --------------------------------------------------------------------------- #


REVIEW_UUID = "08d12f77-3ff1-4167-8ba3-f51d3f084763"
REVIEW_SESSION_ID = "ses_08d12f773ff141678ba3f51d3f084763"


def _review_transcript(tmp_path: Path) -> tuple[Path, InMemoryRepository, ExternalTranscriptObserver]:
    cwd = tmp_path / "project"
    cwd.mkdir()
    transcript = pi_transcript_path(
        str(cwd), "2020-01-01T00-00-00-000Z", REVIEW_UUID, home=tmp_path
    )
    transcript.parent.mkdir(parents=True)
    _write(
        transcript,
        [
            {
                "type": "session",
                "id": REVIEW_UUID,
                "version": 3,
                "timestamp": "2020-01-01T00:00:00Z",
                "cwd": str(cwd),
            },
            {"type": "model_change", "provider": "provider-a", "modelId": "model-a"},
        ],
    )
    repository = InMemoryRepository()
    return transcript, repository, ExternalTranscriptObserver(
        InMemoryEventBus(), repository=repository
    )


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@pytest.mark.asyncio
async def test_session_keeps_the_conversations_original_creation_time(tmp_path) -> None:
    """A conversation started in 2020 and discovered today is not new. A
    `now` timestamp would sort it to the top of `.sessions` and misreport
    its age."""
    transcript, repository, observer = _review_transcript(tmp_path)

    await observer.tail_file(transcript)

    assert repository.get_session(REVIEW_SESSION_ID).created_at == datetime(
        2020, 1, 1, tzinfo=UTC
    )


@pytest.mark.asyncio
async def test_model_survives_restart_after_a_late_model_change(tmp_path) -> None:
    """The head peek only ever sees the model the conversation OPENED with.
    After a restart, re-deriving facts from the head would drag a session
    that has since switched models back to its first one — so the persisted
    session is consulted first, and the head only for sessions the harness
    has never heard of.
    """
    transcript, repository, observer = _review_transcript(tmp_path)
    for i in range(70):
        _append(transcript, {"type": "message", "message": {"role": "user", "content": f"q{i}"}})
    _append(transcript, {"type": "model_change", "provider": "provider-b", "modelId": "model-b"})
    await observer.tail_file(transcript)
    assert repository.get_session(REVIEW_SESSION_ID).model == "provider-b/model-b"

    restarted = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    restarted._state.set_next_offset(transcript, transcript.stat().st_size)
    _append(
        transcript,
        {"type": "message", "message": {"role": "user", "content": "after restart"}},
    )
    await restarted.tail_file(transcript)

    assert repository.get_session(REVIEW_SESSION_ID).model == "provider-b/model-b"


@pytest.mark.asyncio
async def test_metadata_refresh_keeps_user_owned_session_settings(tmp_path) -> None:
    transcript, repository, observer = _review_transcript(tmp_path)
    await observer.tail_file(transcript)
    initial = repository.get_session(REVIEW_SESSION_ID)
    repository.upsert_session(
        initial.model_copy(
            update={"title": "User selected title", "bypass_permissions": True, "effort": "high"}
        )
    )

    _append(transcript, {"type": "model_change", "provider": "provider-b", "modelId": "model-b"})
    await observer.tail_file(transcript)

    refreshed = repository.get_session(REVIEW_SESSION_ID)
    assert refreshed.title == "User selected title"
    assert refreshed.bypass_permissions is True
    assert refreshed.effort == "high"
    assert refreshed.created_at == initial.created_at
    # The conversation's own shape still refreshes.
    assert refreshed.model == "provider-b/model-b"


def test_qualified_model_never_strips_a_repeated_provider_segment() -> None:
    """pi model ids carry their own namespaces — openrouter's catalogue is
    full of ids like ``openrouter/free``. Treating the repeated segment as
    an already-applied prefix would name a model that doesn't exist. pi
    splits only on the FIRST slash, so the doubled form is correct.
    """
    facts = PiSessionFacts(provider="openrouter", model="openrouter/free")

    assert facts.qualified_model == "openrouter/openrouter/free"


@pytest.mark.asyncio
async def test_resume_target_is_absolute_even_for_a_relative_observation(
    tmp_path, monkeypatch
) -> None:
    """The run is launched with ``cwd`` set to the PROJECT directory, so a
    relative transcript path would resolve against the project and address
    a different file — which pi would then create as a new conversation.
    """
    transcript, repository, observer = _review_transcript(tmp_path)
    monkeypatch.chdir(tmp_path)

    await observer.tail_file(transcript.relative_to(tmp_path))

    target = pi_resume_target(repository.get_session(REVIEW_SESSION_ID))
    assert target is not None
    assert target.is_absolute()
    assert target == transcript.resolve()


@pytest.mark.asyncio
async def test_a_partial_model_change_cannot_double_qualify_the_model(tmp_path) -> None:
    """A ``model_change`` carrying a provider but no modelId must not
    re-qualify the already-qualified model stored on the session."""
    transcript, repository, observer = _review_transcript(tmp_path)
    await observer.tail_file(transcript)
    assert repository.get_session(REVIEW_SESSION_ID).model == "provider-a/model-a"

    restarted = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    restarted._state.set_next_offset(transcript, transcript.stat().st_size)
    _append(transcript, {"type": "model_change", "provider": "provider-a"})
    _append(transcript, {"type": "message", "message": {"role": "user", "content": "hi"}})
    await restarted.tail_file(transcript)

    assert repository.get_session(REVIEW_SESSION_ID).model == "provider-a/model-a"


@pytest.mark.asyncio
async def test_the_builder_refuses_rather_than_creating_when_the_source_vanishes(
    tmp_path,
) -> None:
    """Preflight and build happen at different moments. A transcript can
    disappear in between, and a caller could skip the preflight entirely —
    so the builder re-checks instead of falling back to ``--session-id``,
    which would start the very conversation the design exists to prevent.
    """
    transcript, repository, observer = _review_transcript(tmp_path)
    await observer.tail_file(transcript)
    session = repository.get_session(REVIEW_SESSION_ID)
    validate_session_resume_target(session)  # passes: the file is still there

    transcript.unlink()

    with pytest.raises(CommandBuildError):
        PiCommandBuilder().build(
            session=session,
            run=Run(session_id=REVIEW_SESSION_ID),
            message=Message.user("continue"),
        )


def test_the_builder_resolves_the_transcript_exactly_once(tmp_path, monkeypatch) -> None:
    """Resolving twice — once to test, once to use — leaves a window where
    the file reappears in between and the argv is built from the stale
    first answer. That produced a literal ``--session None``, which pi
    would have created as a file named ``None``.
    """
    transcript = _seed_transcript(tmp_path)
    session = _external_pi_session(transcript_path=transcript)
    calls: list[int] = []
    real = orchestrator.pi_resume_target

    def counting(target_session):
        calls.append(1)
        return real(target_session)

    monkeypatch.setattr(orchestrator, "pi_resume_target", counting)
    command = PiCommandBuilder().build(
        session=session, run=Run(session_id=session.id), message=Message.user("go")
    )

    assert calls == [1]
    assert "None" not in command.argv
    assert command.argv[command.argv.index("--session") + 1] == str(transcript)
