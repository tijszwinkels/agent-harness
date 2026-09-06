import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import Project, Session
from agent_harness.observer import (
    ExternalTranscriptObserver,
    codex_transcript_path,
    parse_transcript_record,
    transcript_identity_from_path,
)
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository


ROLLOUT_UUID = "019e0100-0000-0000-0000-000000000000"
CREATED = datetime(2026, 9, 6, 13, 12, 18, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def repository(request, tmp_path):
    repo = (
        InMemoryRepository()
        if request.param == "memory"
        else open_sqlite_repository(tmp_path / "harness.db")
    )
    yield repo
    if request.param == "sqlite":
        repo.close()


@pytest.fixture
def session(repository):
    session = Session(
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/repo", name="repo"),
        codex_resume_id=ROLLOUT_UUID,
    )
    repository.upsert_session(session)
    return session


@pytest.fixture
def rollout(tmp_path):
    path = codex_transcript_path(
        year=2026, month=9, day=6, timestamp="2026-09-06T13-12-18",
        rollout_uuid=ROLLOUT_UUID, home=tmp_path,
    )
    path.parent.mkdir(parents=True)
    records = [
        {"type": "session_meta", "timestamp": CREATED.isoformat(),
         "payload": {"id": ROLLOUT_UUID, "cwd": "/repo", "model": "gpt-5.4"}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Delivered answer"}],
        }},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


@pytest.mark.parametrize("scenario", ["expired", "restart", "unbound", "competing", "archived"])
async def test_persisted_uuid_routes_messages_to_harness_session(
    repository, session, rollout, scenario,
):
    now = CREATED
    bus = (
        InMemoryEventBus()
        if isinstance(repository, InMemoryRepository)
        else DurableEventBus(repository)
    )
    observer = ExternalTranscriptObserver(bus, repository=repository, clock=lambda: now)
    if scenario in {"expired", "unbound"}:
        observer.expect_codex_rollout(cwd="/repo", session_id=session.id)
        if scenario == "unbound":
            observer.bind_rollout(rollout, session.id)
            observer.unbind_session(session.id)
        now += timedelta(minutes=31)
    elif scenario == "competing":
        other = session.model_copy(update={"id": "ses_other", "codex_resume_id": None})
        repository.upsert_session(other)
        observer.expect_codex_rollout(cwd="/repo", session_id=other.id)
    elif scenario == "archived":
        repository.archive_session(session.id)

    published = await observer.tail_file(rollout)

    messages = repository.list_messages(session.id)
    assert [message.blocks[0].text for message in messages] == ["Delivered answer"]
    assert all(event.session_id == session.id for event in published)
    assert not repository.has_session(f"codex_{ROLLOUT_UUID}")
    if scenario == "competing":
        assert repository.get_session(other.id).codex_resume_id is None
        assert repository.list_messages(other.id) == []
    elif scenario == "archived":
        assert repository.get_session(session.id).status == "archived"
    replay = await bus.replay(session_id=session.id)
    assert len([event for event in replay if event.event == "message"]) == 1
    assert await observer.tail_file(rollout) == []


async def test_binding_survives_database_reopen(tmp_path, rollout):
    database = tmp_path / "harness.db"
    repository = open_sqlite_repository(database)
    session = Session(
        backend="codex", model="gpt-5.4", project=Project(path="/repo", name="repo"),
    )
    repository.upsert_session(session)
    observer = ExternalTranscriptObserver(
        DurableEventBus(repository), repository=repository, clock=lambda: CREATED,
    )
    observer.expect_codex_rollout(cwd="/repo", session_id=session.id)
    await observer.tail_file(rollout)
    assert repository.get_session(session.id).codex_resume_id == ROLLOUT_UUID
    repository.close()

    with rollout.open("a") as transcript:
        transcript.write(json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "After restart"}],
        }}) + "\n")
    reopened = open_sqlite_repository(database)
    try:
        observer = ExternalTranscriptObserver(DurableEventBus(reopened), repository=reopened)
        await observer.tail_file(rollout)
        assert [m.blocks[0].text for m in reopened.list_messages(session.id)] == [
            "Delivered answer", "After restart",
        ]
        assert not reopened.has_session(f"codex_{ROLLOUT_UUID}")
    finally:
        reopened.close()


async def test_persisted_uuid_overrides_cached_external_identity(repository, rollout):
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    assert observer._resolve_identity(rollout).is_rebound is False
    session = Session(
        backend="codex", model="gpt-5.4", project=Project(path="/repo", name="repo"),
        codex_resume_id=ROLLOUT_UUID,
    )
    repository.upsert_session(session)
    published = await observer.tail_file(rollout)
    assert published and all(event.session_id == session.id for event in published)
    assert not repository.has_session(f"codex_{ROLLOUT_UUID}")


async def test_existing_external_duplicate_does_not_receive_new_messages(
    repository, session, rollout,
):
    duplicate = session.model_copy(update={"id": f"codex_{ROLLOUT_UUID}", "origin": "external"})
    repository.upsert_session(duplicate)
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    await observer.tail_file(rollout)
    assert len(repository.list_messages(session.id)) == 1
    assert repository.list_messages(duplicate.id) == []


async def test_ambiguous_harness_ownership_defers_with_warning(
    repository, session, rollout, caplog,
):
    repository.upsert_session(session.model_copy(update={"id": "ses_duplicate"}))
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    with caplog.at_level(logging.WARNING):
        assert await observer.tail_file(rollout) == []
    assert "Ambiguous Codex rollout ownership" in caplog.text
    assert session.id in caplog.text and "ses_duplicate" in caplog.text
    assert not repository.has_session(f"codex_{ROLLOUT_UUID}")


async def test_ownership_lookup_error_preserves_bytes_for_retry(
    repository, session, rollout, monkeypatch, caplog,
):
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    lookup = repository.find_codex_harness_sessions

    def unavailable(resume_id):
        raise RuntimeError("repository temporarily unavailable")

    monkeypatch.setattr(repository, "find_codex_harness_sessions", unavailable)
    with caplog.at_level(logging.ERROR):
        assert await observer.tail_file(rollout) == []
    assert "Failed to resolve Codex rollout ownership" in caplog.text
    assert "repository temporarily unavailable" in caplog.text
    assert not repository.has_session(f"codex_{ROLLOUT_UUID}")

    monkeypatch.setattr(repository, "find_codex_harness_sessions", lookup)
    await observer.tail_file(rollout)
    assert [m.blocks[0].text for m in repository.list_messages(session.id)] == ["Delivered answer"]


@pytest.mark.parametrize("backend,origin", [("codex", "external"), ("claude-code", "harness")])
async def test_non_harness_codex_owner_does_not_rebind(repository, rollout, backend, origin):
    repository.upsert_session(Session(
        id="ses_unrelated", backend=backend, origin=origin, model="test-model",
        project=Project(path="/repo", name="repo"), codex_resume_id=ROLLOUT_UUID,
    ))
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    published = await observer.tail_file(rollout)
    assert published and all(event.session_id == f"codex_{ROLLOUT_UUID}" for event in published)


@pytest.mark.parametrize("payload_type", ["item_completed", "token_usage_record", "world_state"])
def test_known_codex_metadata_does_not_warn(rollout, payload_type, caplog):
    with caplog.at_level(logging.WARNING):
        events = parse_transcript_record(
            {"type": "event_msg", "payload": {"type": payload_type}},
            identity=transcript_identity_from_path(rollout),
        )
    assert events == []
    assert not caplog.records
