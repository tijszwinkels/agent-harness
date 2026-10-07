# Hybrid Semantics

agent-harness v1 is both an orchestrator and an observer. Harness-launched runs
own their process lifecycle. External Claude Code and Codex conversations are
discovered from transcript files, normalized into the same Session, Run,
Message, and Event model, and exposed through the same HTTP and SSE API.

Sources consulted on 2026-05-08:

- OpenAPI Specification v3.1.0, `https://spec.openapis.org/oas/v3.1.0.html`,
  for contract structure and media type descriptions.
- MDN EventSource and Server-Sent Events documentation,
  `https://developer.mozilla.org/en-US/docs/Web/API/EventSource` and
  `https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events`,
  for the `text/event-stream` transport model.

Additional sources consulted on 2026-05-14:

- MDN Server-Sent Events documentation,
  `https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events`,
  for SSE `id` fields and reconnect framing.
- Prior WIP diff in
  `/home/claude/projects/agent-harness-echo/worktrees/durable-events`,
  for the first durable event-bus design sketch.

## Session Identity

External session ids are stable and backend-prefixed:

- Claude Code: `claude_<session_uuid>`
- Codex: `codex_<rollout_uuid>`

The suffix is the backend's durable conversation identifier from the transcript,
not a harness-generated alias. The same external transcript must always map to
the same harness session id across restarts.

Harness-origin sessions use harness-generated ids. They must not use the
reserved `claude_` or `codex_` prefixes.

## Origins

`origin` records who created the resource:

- `harness`: created by agent-harness through the API.
- `external`: discovered from backend transcript files.

A session keeps its origin forever. If a client starts a new run in an external
session, the session remains `origin: external`; only the new run is
`origin: harness`.

## Resume And Adoption

`POST /v1/sessions/{id}/runs` is the only v1 API for starting work inside an
existing session. Clients that need a new conversation first create a session
with `POST /v1/sessions`, then start its first run.

When `{id}` points at a harness-origin session, the harness starts a new run in
that session.

When `{id}` points at an external session, the harness adopts that
conversation only if the backend supports resume for the external id. Adoption
means the harness launches a new backend process with the backend's resume flag
or equivalent and associates the resulting work with a new harness-origin run.
The previously observed external messages remain part of the session history.

Current v1 command mappings:

- Codex external sessions must use the `codex_<uuid>` id shape. The harness
  launches `codex exec resume --json --model <model> <uuid> <message>`.
- Claude Code external sessions must use the `claude_<uuid>` id shape. The
  harness launches
  `claude --print --output-format stream-json --include-partial-messages --model <model> --resume <uuid> <message>`.
- pi external sessions use the canonical `ses_<32hex>` id shape (the transcript
  filename's UUID, dashes stripped). The harness launches
  `pi -p [--model <provider/model>] --session <transcript-path> <message>`, resuming
  the exact file the observer read (`Session.pi_transcript_path`) rather than a
  path rebuilt from the id. Unlike codex and claude, pi appends to the observed
  transcript itself: the external conversation and the harness-origin runs share
  one file. `model` is stored provider-qualified (`ollama/glm-5.2:cloud`) and
  `--provider` is never emitted — pi only honours a `provider/` prefix when
  `--provider` is absent, so a separately-pinned provider could silently outrank
  a freshly-chosen model.

Harness-origin sessions currently start a fresh non-interactive backend process.
They do not yet resume the backend's underlying conversation, because v1 does
not persist the backend-generated session id emitted by a launched process.

If the backend cannot resume the external conversation,
`POST /v1/sessions/{id}/runs` returns `409 Conflict` with a clear message.

"Cannot resume" includes the case where the backend would *silently succeed at
the wrong thing*. Both pi resume forms create rather than fail — an unknown
`--session-id`, or a `--session` path that doesn't exist — so a moved or deleted
transcript would otherwise produce a plausible reply with none of the session's
history. Before creating the run the harness checks the recorded transcript
exists and that its `session` header names this session, and 409s otherwise. The
header check, not just the path, is what stops a replaced or truncated file being
resumed under the wrong session's name.

Resume is always a **local** operation: the harness invokes the backend CLI on
its own host, against transcripts on its own disk. It cannot continue a session
whose transcript lives on another machine, and the observer cannot discover one
either — see the observer note in the README.

## Catch-Up Semantics

SSE streams follow new events by default:

```text
GET /v1/events
GET /v1/sessions/{session_id}/events
```

`?from=beginning` replays retained events for the selected scope before
following new events. `?after=<sequence>` replays retained events with a higher
sequence and then follows new events. If both are supplied, `after` is more
specific and wins.

Retention is implementation-defined in v1, but while an event is retained it
must keep its original sequence number. A client that reconnects with the last
seen SSE `id` as `after` must never receive an event with sequence less than or
equal to that value.

Message list endpoints return materialized message state. SSE endpoints return
the event log. Clients that need lossless incremental updates should use SSE.

When SQLite is configured, the SSE event log is durable. Every event published
through the harness event bus is assigned a monotonically increasing sequence,
inserted into the SQLite `events` table, and only then fanned out to live SSE
subscribers. Restarting the harness with the same database continues allocating
from the existing maximum sequence, so clients can resume with `after=<last id>`
without treating a restart as a sequence reset.

When SQLite is not configured, the same SSE contract is served by the in-memory
event bus. Replay works only for events retained by the current process.

Events are retained indefinitely in v1. The rows are small and the bridge needs
a continuous event log more than it needs automatic pruning. If this becomes a
storage issue, add an explicit max-age or max-row policy rather than silently
discarding resumable history.

## External Liveness

The harness does not own external processes and must not imply that it can prove
their process state.

An external session is `running` when transcript activity was observed in the
last 30 seconds (`DEFAULT_IDLE_AFTER_SECONDS`). Otherwise it is `idle`. The
timestamp source is the observer's last accepted transcript event, not process
table inspection.

Transcript freshness is only a liveness signal for sessions the harness is not
running. **While a session has a `queued` or `running` harness run, the run
lifecycle owns its status**: `create_run` sets it `running`, `finish_run` sets
it `idle` once no other run is queued or running. A harness run's rollout is
bound to its session, so its transcript also feeds the freshness map, but 30 s
of silence (a long tool call, extended thinking) is not the end of a run.

Rule (`models.observed_status`): an observation never decides the status of a
session that is archived (the user's action) or has an active run (the run
lifecycle's) — in either direction, so silence cannot demote it and activity
cannot override e.g. `waiting_for_input`. Enforced at three points:

- the observer skips status flips for sessions with an active run, and the
  freshness tick only demotes `running`;
- materializing an observed `session.updated` (`merge_observed_session`)
  keeps the stored status in those cases (other observer-owned fields still
  apply), checked under the repository lock;
- the announced event carries the same decision: the SQLite repository
  rewrites the `session.updated` status in `append_event` under the lock that
  also covers the materialization, so the durable log, replay and subscribers
  match the stored row even if a run started while the publication waited for
  the bus. On the in-memory bus the observer reconciles before publishing and,
  should the row still differ after materialization, publishes the stored
  session as a correction.

Once the run has finished, freshness applies again: a late transcript flush
may mark the session `running`, and the next tick returns it to `idle`.

Archiving an external session affects harness visibility only. It does not stop,
signal, delete, or otherwise mutate the external backend process or transcript.

## Filesystem Watcher

The v1 observer primitive is `watchfiles`.

Observers watch backend transcript roots, parse changed transcript files, and
upsert external sessions, runs, messages, and events. Parsing should be
idempotent: rescanning the same transcript content must not duplicate messages
or events.

Filesystem watcher failures should be reported as `observer.error` events and
logged with enough context to identify the backend, path, and exception.

## Ordering And Event Bus

All harness and observer publications go through one event bus instance.

The bus assigns a single monotonically increasing integer `sequence` to every
event after parsing and before fan-out. With SQLite, sequence allocation and
event insertion happen inside the repository transaction while the event bus
holds its publisher lock. This sequence is the authoritative ordering key for:

- SSE `id`
- event replay with `after`
- message ordering when messages are derived from events

Backend transcript timestamps may be retained as metadata, but they do not
define cross-backend ordering. If two events are observed in the same poll or
filesystem notification cycle, the bus publication order decides their relative
sequence.

## External Interrupts

In v1, interrupt support is lifecycle ownership based.

The harness accepts `DELETE /v1/sessions/{id}/runs/{run_id}` only for
harness-origin runs where it owns the launched process and the backend adapter
supports interruption.

Interrupting an external run or externally owned activity returns `409 Conflict`
with a clear error:

```json
{
  "error": {
    "code": "external_interrupt_unsupported",
    "message": "Cannot interrupt an external run in v1 because the harness does not own its process."
  }
}
```

The harness must not infer process ownership from a matching external session
id. A resumed/adopted conversation creates a new harness-origin run; that new
run may be interruptible if its backend adapter supports interruption.
