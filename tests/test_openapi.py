from pathlib import Path

import yaml


def _openapi_spec() -> dict:
    return yaml.safe_load(Path("specs/openapi.yaml").read_text())


def test_openapi_documents_current_run_contracts() -> None:
    spec = _openapi_spec()
    schemas = spec["components"]["schemas"]

    create_session = schemas["CreateSessionRequest"]
    assert create_session["properties"]["bypass_permissions"] == {
        "type": "boolean",
        "default": False,
    }

    create_run = schemas["CreateRunResponse"]
    assert "status" in create_run["required"]
    assert create_run["properties"]["status"]["enum"] == [
        "queued",
        "running",
        "completed",
        "failed",
        "interrupted",
    ]

    interrupt_response = spec["paths"]["/v1/sessions/{id}/runs/{run_id}"]["delete"]["responses"]["200"]
    assert interrupt_response["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/InterruptRunResponse",
    }
    assert schemas["InterruptRunResponse"]["required"] == ["run", "dropped_queued"]

    create_run_responses = spec["paths"]["/v1/sessions/{id}/runs"]["post"]["responses"]
    assert create_run_responses["429"] == {"$ref": "#/components/responses/Error"}


def test_openapi_documents_patch_session() -> None:
    spec = _openapi_spec()

    patch = spec["paths"]["/v1/sessions/{id}"]["patch"]
    assert patch["operationId"] == "patchSession"
    assert patch["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PatchSessionRequest",
    }
    responses = patch["responses"]
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Session",
    }
    for code in ("404", "422"):
        assert responses[code] == {"$ref": "#/components/responses/Error"}

    patch_request = spec["components"]["schemas"]["PatchSessionRequest"]
    # title is optional (omit to leave unchanged) but must be a non-empty
    # string when supplied.
    assert "required" not in patch_request or patch_request["required"] == []
    assert patch_request["properties"]["title"] == {
        "type": "string",
        "minLength": 1,
    }


def test_openapi_documents_session_codex_internal_id() -> None:
    spec = _openapi_spec()
    session = spec["components"]["schemas"]["Session"]
    # Nullable string, optional (only set after the observer binds a codex
    # rollout to the harness session via the reconcile pre-check).
    assert session["properties"]["codex_internal_id"] == {
        "type": ["string", "null"],
        "minLength": 1,
    }
    assert "codex_internal_id" not in session.get("required", [])
