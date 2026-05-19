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


def test_openapi_documents_phase3_event_types() -> None:
    spec = _openapi_spec()
    event_enum = spec["components"]["schemas"]["Event"]["properties"]["event"]["enum"]
    # Phase 3 adds ``run.usage`` (observer-materialized from rollouts).
    assert "run.usage" in event_enum
    # Phase 2 added ``process.stderr`` (supervisor forwards stderr lines).
    assert "process.stderr" in event_enum


def test_openapi_documents_session_stats_context_window() -> None:
    spec = _openapi_spec()
    session_stats = spec["components"]["schemas"]["SessionStats"]
    cw = session_stats["properties"]["context_window"]
    # Nullable integer, minimum 1, NOT required (codex-only field).
    assert cw["type"] == ["integer", "null"]
    assert cw["minimum"] == 1
    assert "context_window" not in session_stats.get("required", [])
