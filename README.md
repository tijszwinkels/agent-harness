# Agent Harness

HTTP/SSE service and SDK primitives for CLI coding agents.

## What is included

- Modern `pyproject.toml` packaging with `uv` support.
- Pydantic v2 models for sessions, runs, normalized message blocks, backends, events, and common API responses.
- In-memory session repository and optional SQLite persistence.
- In-memory event bus with monotonic sequence numbers, replay, and live subscriptions.
- Backend registry stubs for `claude-code`, `codex`, and `pi`.
- FastAPI app factory with health, backend listing, session and run endpoints, DELETE-based interruption/archive behavior, and SSE event streaming endpoints.
- External Claude Code, Codex, and pi transcript observation with `watchfiles`.

## Development

```bash
uv run pytest
```

Run the local service:

```bash
uv run agent-harness serve \
  --host 127.0.0.1 \
  --port 8876 \
  --database .agent-harness.db
```

The ASGI app import string is `agent_harness.api:app`.

Health checks can use either `/health` or `/v1/health`. Backend model discovery
is exposed at `/v1/backends/{name}/models`; it currently returns an empty list
when the harness does not have an authoritative model catalog, and callers may
still pass backend-supported model names directly when creating sessions.

Add `--execute-runs` when the service should launch real backend CLI
processes for `POST /v1/sessions/{id}/runs`. Without that flag, run creation
records the input message and publishes API events, but does not invoke Codex,
Claude Code, or pi.

By default, the server observes the known Claude Code, Codex, and pi transcript roots under `HOME` when those directories exist. Use `--no-observer` to disable that automatic observation. pi rollouts (`~/.pi/agent/sessions`) are parsed into `message`, `run.usage`, and `run.end_turn` events like the other backends; see `specs/2026-07-06-pi-transcript-observer.md`.
Transcript records are normalized into the existing message block models where possible; see `docs/transcript-normalization.md` for coverage and limits.

Appending to an observed external session uses the same run endpoint. When
`--execute-runs` is enabled:

- `codex_<uuid>` sessions launch
  `codex exec resume --json --model <model> <uuid> <message>`.
- `claude_<uuid>` sessions launch
  `claude --print --output-format stream-json --include-partial-messages --model <model> --resume <uuid> <message>`.

Harness-created sessions still launch a fresh non-interactive backend process in
v1. True resume for those sessions requires capturing and storing the backend's
own session id from process output.

For a fake transcript demo, run with an explicit observed root:

```bash
mkdir -p /tmp/agent-harness-observer-demo
uv run agent-harness serve \
  --host 127.0.0.1 \
  --port 8876 \
  --database .agent-harness.db \
  --observe-root /tmp/agent-harness-observer-demo
```

In another shell, write a safe fake Codex transcript into that watched root:

```bash
python examples/python/observer_external.py \
  --base-url http://127.0.0.1:8876 \
  --root /tmp/agent-harness-observer-demo
```

See `examples/` for curl and Python examples.
