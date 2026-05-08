# Agent Harness

Clean-slate Python scaffold for an HTTP/SSE service and SDK primitives for CLI coding agents.

## What is included

- Modern `pyproject.toml` packaging with `uv` support.
- Pydantic v2 models for sessions, runs, normalized message blocks, backends, events, and common API responses.
- In-memory session repository.
- In-memory event bus with monotonic sequence numbers, replay, and live subscriptions.
- Backend registry stubs for `claude-code` and `codex`.
- FastAPI app factory with health, backend listing, session create/list/get/archive, and SSE event streaming endpoints.

## Development

```bash
uv run pytest
```

The service entry point is `agent_harness.create_app`.
