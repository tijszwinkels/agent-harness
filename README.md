# Agent Harness

**Chat-ops for coding agents — the service layer.**

Agent Harness is an HTTP/SSE service (plus the Python primitives behind it)
that puts a single, uniform API in front of CLI coding agents — Claude Code,
Codex, and pi. You create a *session*, start a *run*, stream *events*, and read
back *normalized messages* — the same way regardless of which agent CLI is
underneath.

It is the backend half of a two-part chat-ops stack. The frontend,
[mm-bridge](https://github.com/tijszwinkels/mm-bridge), turns Mattermost
messages into harness runs and streams the replies back into chat. Agent
Harness is the engine that actually drives the agents; mm-bridge is one client
of its API, and you can write others.

## What it does

- **One API for many agent CLIs.** Backends for `claude-code`, `codex`, and
  `pi`, all exposed through the same `/v1` session/run/event endpoints.
- **Hybrid orchestrator *and* observer.** Runs the harness launches itself are
  fully managed (subprocess lifecycle, queueing, interrupt, live output). Agent
  sessions started *outside* the harness are discovered by watching each
  backend's transcript files and surfaced through the *same* API and event
  schema. `Session.origin` / `Run.origin` distinguish `harness` from
  `external`.
- **Live event streaming.** A server-sent-events stream with monotonic
  sequence numbers, replay from any offset, and live subscription — so a client
  can reconnect and catch up without missing or double-counting events.
- **Normalized messages.** Per-backend transcript records are parsed into one
  shared message-block model (text, thinking, tool use, usage); see
  [`docs/transcript-normalization.md`](docs/transcript-normalization.md) for
  coverage and limits.
- **Durable or in-memory.** Runs against an in-memory repository by default, or
  a SQLite database (`--database`) for state that survives restarts.

## Architecture

```
                        ┌──────────────────────────────┐
   chat / other client  │        Agent Harness         │
   ───────────────────► │        FastAPI  /v1          │
     (e.g. mm-bridge)    │  sessions · runs · events    │
                        │        SSE  /v1/events        │
                        └───────┬───────────────┬───────┘
                                │               │
                    orchestrator│               │observer
                 (harness-owned │               │ (external sessions,
                    subprocess  │               │  watchfiles on
                    lifecycle)  │               │  transcript roots)
                                ▼               ▼
                       claude-code · codex · pi CLIs
                                │               │
                                └──── event bus ┘
                             (durable / in-memory,
                          monotonic seq, replay + live)
```

Everything a run produces — whether the harness launched it or merely observed
it — lands on one event bus and is materialized into the same repository, so
`GET /v1/sessions/{id}/messages` and the SSE stream look identical for both
origins.

Source modules: `api.py` (routes), `orchestrator.py` (harness-owned runs),
`observer.py` + `rollout_discovery.py` (external-session watching), `events.py`
(bus, sequencing, replay), `repository.py` / `storage.py` (in-memory + SQLite),
`backends.py` (per-CLI command builders), `models.py` (Pydantic v2 schema),
`usage.py` (token accounting).

## API surface

All application endpoints are under `/v1` (health is also exposed unversioned).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/v1/health` | Liveness. |
| GET | `/v1/backends` | List available backends. |
| GET | `/v1/backends/{name}/models` | Backend model catalog (may be empty). |
| POST | `/v1/sessions` | Create a session. |
| GET | `/v1/sessions` | List sessions. |
| GET | `/v1/sessions/{id}` | Fetch a session. |
| PATCH | `/v1/sessions/{id}` | Update a session (e.g. title). |
| DELETE | `/v1/sessions/{id}` | Archive a session. |
| POST | `/v1/sessions/{id}/runs` | Start (or append) a run. |
| GET | `/v1/sessions/{id}/runs` | List runs. |
| GET | `/v1/sessions/{id}/runs/{run_id}` | Fetch a run. |
| DELETE | `/v1/sessions/{id}/runs/{run_id}` | Interrupt a run. |
| GET | `/v1/sessions/{id}/messages` | Normalized messages. |
| GET | `/v1/sessions/{id}/events` | This session's events. |
| GET | `/v1/events` | Global SSE event stream. |
| GET | `/v1/events/max-sequence` | Highest sequence number (for replay/resume). |

The ASGI application import string is `agent_harness.api:app`.

## Requirements

- **Python ≥ 3.12.**
- **The agent CLIs you intend to drive, on `PATH`** — `claude`, `codex`, and/or
  `pi`. The harness invokes them by bare name (no wrapper, no configurable
  binary path).
- **Linux preferred.** The harness runs on macOS too, but process-group reaping
  and transcript observation are exercised primarily on Linux.
- For real execution, whatever each agent CLI itself needs (auth, network).

## Running the service

```bash
uv run agent-harness serve \
  --host 127.0.0.1 \
  --port 8000 \
  --database .agent-harness.db
```

Key flags:

- `--execute-runs` — actually launch backend CLI processes for
  `POST /v1/sessions/{id}/runs`. **Without it, run creation records the input
  message and publishes API events but does not invoke any CLI** — useful for
  API-shape testing.
- `--database PATH` — use SQLite for durable state (default: in-memory).
- `--no-observer` — disable automatic observation of transcript roots.
- `--observe-root PATH` — watch an additional transcript root (repeatable).
- `--cors-origin ORIGIN` — allow a browser origin (repeatable).

By default the server observes the known Claude Code, Codex, and pi transcript
roots under `HOME` when those directories exist.

## Observing external sessions

The observer watches each backend's transcript files with `watchfiles` and
replays their records through the same event bus as harness-owned runs. To try
it against a safe fake transcript instead of your real sessions:

```bash
mkdir -p /tmp/agent-harness-observer-demo
uv run agent-harness serve --port 8000 \
  --observe-root /tmp/agent-harness-observer-demo
# in another shell:
python examples/python/observer_external.py \
  --base-url http://127.0.0.1:8000 --root /tmp/agent-harness-observer-demo
```

See [`examples/`](examples/) for curl and Python clients.

## Appending and resume semantics

Appending to an observed external session uses the same run endpoint. With
`--execute-runs` enabled:

- `codex_<uuid>` sessions launch `codex exec resume --json --model <model>
  <uuid> <message>`.
- `claude_<uuid>` sessions launch `claude --print --output-format stream-json
  --include-partial-messages --model <model> --resume <uuid> <message>`.

Harness-created sessions launch a fresh non-interactive backend process and
bind the backend's own session id from its output so subsequent turns resume
in-context.

## Installation

A standalone install runbook for the harness is planned but not yet written.
For now, the full-stack install guide — provisioning Mattermost, the harness,
and the bridge together — lives in mm-bridge's
[`INSTALL.md`](https://github.com/tijszwinkels/mm-bridge); its harness-side
steps stand on their own if you only want the service.

To install the CLI from a checkout:

```bash
uv tool install .        # provides the `agent-harness` command
agent-harness serve --help
```

## Development

```bash
uv run pytest
```

## License

[MIT](LICENSE) © 2026 Tijs Zwinkels
