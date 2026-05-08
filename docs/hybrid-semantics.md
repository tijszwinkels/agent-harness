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

If the backend cannot resume the external conversation,
`POST /v1/sessions/{id}/runs` returns `409 Conflict` with an error code such as
`resume_unsupported`.

## Catch-Up Semantics

SSE streams follow new events by default:

```text
GET /v1/events
GET /v1/sessions/{session_id}/events
GET /v1/sessions/{id}/runs/{run_id}/events
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

## External Liveness

The harness does not own external processes and must not imply that it can prove
their process state.

An external session is `running` when transcript activity was observed in the
last 15 seconds. Otherwise it is `idle`. The timestamp source is the observer's
last accepted transcript event, not process table inspection.

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

All harness and observer publications go through one in-process event bus.

The bus assigns a single monotonically increasing integer `sequence` to every
event after parsing and before fan-out. This sequence is the authoritative
ordering key for:

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
