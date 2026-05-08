# Agent Harness

Clean-slate Python scaffold for an HTTP/SSE service and SDK primitives for CLI coding agents.

## What is included

- Modern `pyproject.toml` packaging with `uv` support.
- Pydantic v2 models for sessions, runs, normalized message blocks, backends, events, and common API responses.
- In-memory session repository.
- In-memory event bus with monotonic sequence numbers, replay, and live subscriptions.
- Backend registry stubs for `claude-code` and `codex`.
- FastAPI app factory with health, backend listing, session create/list/get, DELETE-based session archival, and SSE event streaming endpoints.

## Development

```bash
uv run pytest
```

Run the local service:

```bash
uv run agent-harness serve --host 127.0.0.1 --port 8000
```

The ASGI app import string is `agent_harness.api:app`.
