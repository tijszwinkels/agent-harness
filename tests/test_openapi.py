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


def test_openapi_documents_phase4_event_types() -> None:
    """Phase 4 introduces ``run.end_turn`` (observer-emitted; drives
    the watchdog's post-end_turn cleanup grace). The pre-existing
    watchdog terminal events (``run.terminated_after_end_turn``,
    ``run.timed_out_idle``) and ``run.warning`` were emitted but
    never enumerated; locked in here for SSE consumer drift-guard."""
    spec = _openapi_spec()
    event_enum = spec["components"]["schemas"]["Event"]["properties"]["event"]["enum"]
    assert "run.end_turn" in event_enum
    assert "run.terminated_after_end_turn" in event_enum
    assert "run.timed_out_idle" in event_enum
    assert "run.warning" in event_enum


def test_openapi_session_schema_does_not_document_codex_internal_id() -> None:
    """Drift guard: ``Session.codex_internal_id`` only ever lived on
    the abandoned PR #12 branch (never merged to main). Phase 4
    explicitly retires the field at the spec level; verify the
    OpenAPI schema doesn't list it (would be a stray reintroduction)."""
    spec = _openapi_spec()
    session_props = spec["components"]["schemas"]["Session"]["properties"]
    assert "codex_internal_id" not in session_props, (
        f"Session.codex_internal_id should be absent from OpenAPI; "
        f"properties={list(session_props)}"
    )


def test_openapi_documents_session_stats_context_window() -> None:
    spec = _openapi_spec()
    session_stats = spec["components"]["schemas"]["SessionStats"]
    cw = session_stats["properties"]["context_window"]
    # Nullable integer, minimum 1, NOT required (codex-only field).
    assert cw["type"] == ["integer", "null"]
    assert cw["minimum"] == 1
    assert "context_window" not in session_stats.get("required", [])


def test_openapi_documents_session_stats_context_used() -> None:
    """Drift guard: ``Session.stats.context_used`` is the per-session
    SNAPSHOT of currently-loaded context (overwrite-not-sum), distinct
    from cumulative ``stats.tokens``. Optional (None until first
    observation) and non-negative (a zero value is filtered upstream).
    Spec: specs/2026-05-19-context-used.md"""
    spec = _openapi_spec()
    session_stats = spec["components"]["schemas"]["SessionStats"]
    cu = session_stats["properties"]["context_used"]
    assert cu["type"] == ["integer", "null"]
    assert cu["minimum"] == 0
    assert "context_used" not in session_stats.get("required", [])


def test_openapi_documents_fork_route_and_forked_from() -> None:
    """Drift guard: the fork route contract mm-bridge depends on —
    POST /v1/sessions/{id}/forks with 201/404/409, ForkSessionRequest /
    ForkSessionResponse, and Session.forked_from lineage.
    Spec: specs/2026-07-20-session-fork-route.md"""
    spec = _openapi_spec()
    schemas = spec["components"]["schemas"]

    fork = spec["paths"]["/v1/sessions/{id}/forks"]["post"]
    assert set(fork["responses"]) >= {"201", "404", "409"}
    req = fork["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert req.endswith("ForkSessionRequest")
    ok = fork["responses"]["201"]["content"]["application/json"]["schema"]["$ref"]
    assert ok.endswith("ForkSessionResponse")

    assert schemas["ForkSessionResponse"]["required"] == ["session"]
    assert set(schemas["ForkSessionRequest"]["properties"]) == {"message", "title"}

    forked_from = schemas["Session"]["properties"]["forked_from"]
    assert forked_from["type"] == ["string", "null"]
    assert "forked_from" not in schemas["Session"]["required"]


def test_openapi_documents_optional_session_model() -> None:
    """Drift guard: ``model`` is optional on both Session and
    CreateSessionRequest (null ⇒ use the backend CLI's own default; pi
    callers omit it). Spec: specs/2026-07-20-session-fork-route.md"""
    spec = _openapi_spec()
    session = spec["components"]["schemas"]["Session"]
    assert session["properties"]["model"]["type"] == ["string", "null"]
    assert "model" not in session["required"]

    create = spec["components"]["schemas"]["CreateSessionRequest"]
    assert create["properties"]["model"]["type"] == ["string", "null"]
    assert "model" not in create["required"]


def test_openapi_documents_session_codex_resume_id() -> None:
    """Drift guard: ``Session.codex_resume_id`` is the codex rollout
    UUID used by CodexCommandBuilder to pick ``codex exec resume``
    over a fresh ``codex exec``. Nullable string, optional (None on
    non-codex sessions and on codex sessions whose first run hasn't
    completed binding). Spec: specs/2026-05-21-codex-resume.md"""
    spec = _openapi_spec()
    session = spec["components"]["schemas"]["Session"]
    cri = session["properties"]["codex_resume_id"]
    assert cri["type"] == ["string", "null"]
    assert "codex_resume_id" not in session.get("required", [])
