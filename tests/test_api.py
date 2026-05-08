from fastapi.testclient import TestClient

from agent_harness.api import create_app


def test_health_and_backend_listing() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}

    response = client.get("/v1/backends")
    assert response.status_code == 200
    assert {item["name"] for item in response.json()["data"]} == {"claude-code", "codex"}


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


def test_missing_session_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/sessions/ses_missing")

    assert response.status_code == 404
