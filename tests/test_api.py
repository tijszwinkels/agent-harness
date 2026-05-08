from fastapi.testclient import TestClient

from agent_harness.api import create_app


def test_health_and_backend_listing() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}

    response = client.get("/v1/backends")
    assert response.status_code == 200
    assert {item["id"] for item in response.json()["backends"]} == {"claude-code", "codex"}


def test_session_create_list_get_archive_flow() -> None:
    client = TestClient(create_app())

    create_response = client.post(
        "/v1/sessions",
        json={"backend_id": "codex", "title": "Implement scaffold", "metadata": {"branch": "scaffold-core"}},
    )
    assert create_response.status_code == 201
    session = create_response.json()["session"]
    assert session["backend_id"] == "codex"
    assert session["status"] == "active"

    list_response = client.get("/v1/sessions")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()["sessions"]] == [session["id"]]

    get_response = client.get(f"/v1/sessions/{session['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["session"]["id"] == session["id"]

    archive_response = client.post(f"/v1/sessions/{session['id']}/archive")
    assert archive_response.status_code == 200
    assert archive_response.json()["session"]["status"] == "archived"


def test_session_create_rejects_unknown_backend() -> None:
    client = TestClient(create_app())

    response = client.post("/v1/sessions", json={"backend_id": "unknown"})

    assert response.status_code == 422


def test_missing_session_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/sessions/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
