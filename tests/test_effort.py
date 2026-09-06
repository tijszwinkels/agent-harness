import json

import pytest
from fastapi.testclient import TestClient

from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import Event, Message, Project, Run, Session
from agent_harness.observer import ExternalTranscriptObserver
from agent_harness.orchestrator import default_command_builders
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository


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


@pytest.mark.parametrize("backend", ["claude-code", "codex", "pi"])
def test_patch_effort_can_restore_cli_default(repository, backend):
    bus = InMemoryEventBus()
    client = TestClient(create_app(repository=repository, event_bus=bus))
    response = client.post("/v1/sessions", json={
        "backend": backend,
        "project": {"path": "/repo", "name": "repo"},
        "title": "Keep this conversation",
        "effort": "low",
    })
    assert response.status_code == 201
    session_id = response.json()["id"]
    url = f"/v1/sessions/{session_id}"

    assert client.patch(url, json={"effort": "high"}).status_code == 200
    assert client.patch(url, json={"title": "Renamed"}).json()["effort"] == "high"
    response = client.patch(url, json={"effort": None})
    assert response.status_code == 200, response.text
    assert response.json()["effort"] is None
    assert response.json()["title"] == "Renamed"
    assert client.get(url).json()["effort"] is None

    session = repository.get_session(session_id)
    if backend == "codex":
        session = session.model_copy(update={"codex_resume_id": session_id[4:]})
    command = default_command_builders()[backend].build(
        session=session, run=Run(session_id=session_id),
        message=Message.user("Continue"), is_first_run=False,
    )
    assert not any(arg in ("--effort", "--thinking", "-c") for arg in command.argv)
    assert session.id == session_id


@pytest.mark.parametrize("origin", ["harness", "external"])
@pytest.mark.parametrize("effort, stale_effort", [("high", None), ("high", "low"), (None, "high")])
def test_session_events_preserve_current_effort(repository, origin, effort, stale_effort):
    session = Session(
        backend="claude-code", origin=origin,
        project=Project(path="/repo", name="repo"), effort=effort,
    )
    repository.upsert_session(session)
    stale = session.model_copy(update={"effort": stale_effort, "status": "running"})
    repository.materialize_event(Event(
        event="session.updated", session_id=session.id,
        data={"session": stale.model_dump(mode="json")},
    ))
    assert repository.get_session(session.id).effort == effort
    assert repository.get_session(session.id).status == "running"


async def test_observer_publishes_patched_effort(repository, tmp_path):
    bus = (
        InMemoryEventBus() if isinstance(repository, InMemoryRepository)
        else DurableEventBus(repository)
    )
    path = tmp_path / ".claude/projects/-repo/123e4567-e89b-12d3-a456-426614174000.jsonl"
    observer = ExternalTranscriptObserver(event_bus=bus, repository=repository)
    line = json.dumps({
        "type": "assistant", "cwd": "/repo",
        "message": {"id": "msg_effort", "model": "claude-sonnet-4-6", "content": []},
    })
    events = await observer.publish_line(path, line, offset=0)
    session_id = next(event.session_id for event in events if event.event == "session.updated")
    repository.patch_session(session_id, {"effort": "high"})

    events = await observer.publish_line(path, line, offset=1)
    updated = next(event for event in events if event.event == "session.updated")
    assert updated.data["session"]["effort"] == "high"
    assert repository.get_session(session_id).effort == "high"
