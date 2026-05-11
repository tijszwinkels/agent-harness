import pytest
from fastapi.testclient import TestClient

from agent_harness.api import _materialize_run_result, create_app
from agent_harness.models import CreateRunRequest, CreateSessionRequest, Project, Session
from agent_harness.orchestrator import RunProcessResult
from agent_harness.repository import InMemoryRepository
from agent_harness.settings import ObserverSettings


def test_health_and_backend_listing() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/health").json() == {"status": "ok"}

    response = client.get("/v1/backends")
    assert response.status_code == 200
    assert {item["name"] for item in response.json()["data"]} == {"claude-code", "codex"}

    assert client.get("/v1/backends/codex/models").json() == {"data": []}
    assert client.get("/v1/backends/claude-code/models").json() == {"data": []}
    assert client.get("/v1/backends/unknown/models").status_code == 404


def test_session_create_list_get_archive_flow() -> None:
    client = TestClient(create_app())
    payload = {
        "backend": "codex",
        "model": "gpt-5.4",
        "project": {"path": "/tmp/proj", "name": "proj"},
        "title": "Implement scaffold",
    }

    create_response = client.post("/v1/sessions", json=payload)
    assert create_response.status_code == 201
    session = create_response.json()
    assert session["backend"] == "codex"
    assert session["status"] == "idle"
    assert session["origin"] == "harness"

    list_response = client.get("/v1/sessions")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()["data"]] == [session["id"]]

    get_response = client.get(f"/v1/sessions/{session['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == session["id"]

    archive_response = client.delete(f"/v1/sessions/{session['id']}")
    assert archive_response.status_code == 200
    assert archive_response.json()["status"] == "archived"


def test_session_create_rejects_unknown_backend() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/sessions",
        json={"backend": "unknown", "model": "x", "project": {"path": "/tmp", "name": "tmp"}},
    )

    assert response.status_code == 422


def test_run_create_list_get_interrupt_flow() -> None:
    client = TestClient(create_app())
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()

    create_response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})
    assert create_response.status_code == 202
    created = create_response.json()
    assert created["session_id"] == session["id"]
    assert created["run_id"].startswith("run_")

    list_response = client.get(f"/v1/sessions/{session['id']}/runs")
    assert list_response.status_code == 200
    run = list_response.json()["data"][0]
    assert run["id"] == created["run_id"]
    assert run["status"] == "running"
    assert run["origin"] == "harness"

    get_response = client.get(f"/v1/sessions/{session['id']}/runs/{created['run_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == created["run_id"]

    interrupt_response = client.delete(f"/v1/sessions/{session['id']}/runs/{created['run_id']}")
    assert interrupt_response.status_code == 200
    assert interrupt_response.json()["status"] == "interrupted"
    assert interrupt_response.json()["stop_reason"] == "interrupted"


def test_run_create_starts_run_manager_when_configured() -> None:
    class FakeRunManager:
        def __init__(self) -> None:
            self.started = []

        def start(self, *, session, run, command):
            self.started.append((session, run, command))

    class FakeBuilder:
        def build(self, *, session, run, message):
            return ("fake", session.id, run.id, message.id)

    manager = FakeRunManager()
    client = TestClient(
        create_app(
            run_manager=manager,
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()

    response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})

    assert response.status_code == 202
    assert len(manager.started) == 1
    started_session, started_run, command = manager.started[0]
    assert started_session.id == session["id"]
    assert started_run.id == response.json()["run_id"]
    assert command[0] == "fake"


def test_run_create_returns_409_when_external_session_cannot_be_resumed() -> None:
    class FakeRunManager:
        def start(self, *, session, run, command):  # pragma: no cover - should not be reached
            raise AssertionError("run manager should not start when command build fails")

    repo = InMemoryRepository()
    repo.upsert_session(
        Session(
            id="external_without_backend_prefix",
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/tmp/proj", name="proj"),
            origin="external",
        )
    )
    client = TestClient(create_app(repository=repo, run_manager=FakeRunManager()))

    response = client.post("/v1/sessions/external_without_backend_prefix/runs", json={"message": "hello"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Cannot resume external codex session from id external_without_backend_prefix"
    assert repo.list_runs("external_without_backend_prefix") == []
    assert repo.list_messages("external_without_backend_prefix") == []


@pytest.mark.asyncio
async def test_run_result_materialization_updates_repository_status() -> None:
    repo = InMemoryRepository()
    session = repo.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4-mini",
            project=Project(path="/tmp/proj", name="proj"),
        )
    )
    run = repo.create_run(session.id, CreateRunRequest(message="hello"))

    async def wait(run_id):
        return RunProcessResult(run_id=run_id, status="completed", returncode=0)

    await _materialize_run_result(wait, repo.finish_run, session_id=session.id, run_id=run.id)

    stored = repo.get_run(session.id, run.id)
    assert stored.status == "completed"
    assert stored.stop_reason == "end_turn"
    assert stored.completed_at is not None
    assert repo.get_session(session.id).status == "idle"


def test_session_messages_endpoint_returns_materialized_messages() -> None:
    client = TestClient(create_app())
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()
    client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})

    response = client.get(f"/v1/sessions/{session['id']}/messages")

    assert response.status_code == 200
    messages = response.json()["data"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["blocks"][0] == {"type": "text", "text": "hello"}


def test_missing_session_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/sessions/ses_missing")

    assert response.status_code == 404


def test_observer_service_starts_and_stops_with_lifespan(tmp_path) -> None:
    root = tmp_path / "transcripts"
    root.mkdir()
    service = FakeWatchService()
    created_tasks = []

    def watch_service_factory(**kwargs):
        service.roots = kwargs["roots"]
        return service

    def task_factory(coro):
        import asyncio

        task = asyncio.create_task(coro)
        created_tasks.append(task)
        return task

    app = create_app(
        observer_settings=ObserverSettings.from_roots([root]),
        watch_service_factory=watch_service_factory,
        task_factory=task_factory,
    )

    with TestClient(app):
        assert service.started
        assert service.stop_event is not None
        assert created_tasks and not created_tasks[0].done()

    assert service.stopped
    assert created_tasks[0].done()
    assert service.roots == (root,)


def test_observer_startup_configuration_errors_are_logged_and_raised(tmp_path, caplog) -> None:
    missing = tmp_path / "missing"
    app = create_app(observer_settings=ObserverSettings.from_roots([missing]))

    with caplog.at_level("ERROR"):
        try:
            with TestClient(app):
                pass
        except Exception as exc:
            raised = exc
        else:
            raised = None

    assert raised is not None
    assert "Failed to start transcript observer service" in caplog.text
    assert "Observer root does not exist" in str(raised)


class FakeWatchService:
    def __init__(self) -> None:
        self.roots = ()
        self.started = False
        self.stopped = False
        self.stop_event = None

    async def watch_forever(self, *, stop_event=None) -> None:
        import asyncio

        self.started = True
        self.stop_event = stop_event
        while stop_event is not None and not stop_event.is_set():
            await asyncio.sleep(0)
        self.stopped = True
