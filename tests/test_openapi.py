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
