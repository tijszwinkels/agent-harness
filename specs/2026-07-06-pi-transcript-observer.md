# pi transcript observer — parser + watch wiring

Filed: 2026-07-06
Branch: `feat/pi-observer` (from `main` @ d170e77)
Closes: "Transcript observation — DEFERRED" in
`specs/2026-07-05-pi-backend-deviations.md` (pi backend blocker 2).

## Problem

pi-backend runs complete but the observer emits **nothing** — pi's rollout
schema is neither claude's nor codex's, so `transcript_identity_from_path`
rejects pi paths and no `message` events are produced. Downstream, mm-bridge's
`_on_harness_message` never fires, so pi replies post blank to Mattermost.

This note designs a pi rollout parser + observe-root that mirror the existing
claude/codex observer path so that pi assistant turns surface as `message`
events on `GET /v1/sessions/{id}/messages` and the SSE `/v1/events` stream,
with block shapes compatible with mm-bridge.

## pi rollout facts (verified empirically 2026-07-06)

Location: `~/.pi/agent/sessions/<sanitized-cwd>/<ISO-ts>_<session-uuid>.jsonl`
where `<sanitized-cwd>` is the abs cwd with `/`→`-`, wrapped in `--…--`, and the
filename UUID == `_harness_session_id_as_uuid(session.id)`.

Record types (outer `type`) seen across the local corpus (21.6k lines):
`message`, `session`, `model_change`, `thinking_level_change`, `custom`,
`session_info`, `compaction`, `reload`, `custom_message`, `branch_summary`,
`provider_transport_failure`.

- `session` (first line): `{type, version, id, timestamp, cwd}` — has **cwd**, no model.
- `model_change`: `{type, provider, modelId, …}` — has **model**, no cwd.
- `message`: `{type:"message", id, parentId, timestamp, message:{role, content, …}}`.

`message.role` ∈ {`user`, `assistant`, `toolResult`, `bashExecution`}.
Assistant messages additionally carry `model`, `usage`, `stopReason`.
`stopReason` ∈ {`toolUse`(8356), `stop`(991), `aborted`(115), `error`(31), `length`(11)}.

Content-block types inside `message.content`: `text`, `toolCall`, `thinking`,
`image`. Shapes:
- `{type:"text", text}` — matches claude; handled by shared `_blocks_from_content`.
- `{type:"thinking", thinking, thinkingSignature}` — matches claude; shared handler.
- `{type:"toolCall", id, name, arguments}` — **camelCase**, distinct from claude's
  `tool_use`. `arguments` (not `input`). `id` is `toolu_…` / provider id.
- `{type:"image", …}` — 1 occurrence; best-effort via shared handler.

Tool **results** are a separate top-level message record, not a content block:
`{type:"message", message:{role:"toolResult", toolCallId, toolName,
content:[{type:text,text}], isError, …}}`.

## Record → event mapping

| pi record | condition | emitted events |
|---|---|---|
| `session` / `model_change` / `thinking_level_change` / `custom` / `session_info` / `compaction` / `reload` / `custom_message` / `branch_summary` / `provider_transport_failure` | — | ignored (metadata; debug-log, no warning) |
| `message` role=`user` | content blocks | `message` (role=user) — user mirroring |
| `message` role=`assistant` | content: text/thinking/toolCall/image | `message` (role=assistant); + `run.usage` if `message.usage`; + `run.end_turn` if `stopReason=="stop"` |
| `message` role=`toolResult` | toolCallId, content, isError | `message` (role=user, tool_result block) — matches codex `function_call_output` |
| `message` role=`bashExecution` / other | — | ignored (debug-log) |

Block normalization (`_blocks_from_pi_content`): route `toolCall` →
`_tool_use_block` (reads `arguments`+`id`+`name`, already supported); delegate
`text`/`thinking`/`image` to the shared `_blocks_from_content`; skip unknowns.
Reuses `_message_event`, so emitted `message` events are byte-identical in shape
to claude/codex — mm-bridge posts them unchanged.

`stopReason=="stop"` is the claude-`end_turn` analog. `toolUse` continues the
turn; `aborted`/`error`/`length` are abnormal terminations — conservatively NOT
surfaced as `run.end_turn` (claude likewise emits end_turn only for `end_turn`,
not `max_tokens`). pi is single-shot `pi -p`, so process-exit already drives
`run.completed`/`failed`; `run.end_turn` is a watchdog-cleanup nicety here.

## Usage / cost

`parse_pi_usage({input, output, cacheRead, cacheWrite, cost:{total}})` →
`Usage(input, output, cache_read=cacheRead, cache_creation=cacheWrite,
cost_usd=cost.total)`. `parse_pi_context_snapshot` = `input + cacheRead +
cacheWrite` (loaded context, excludes output; all-zero → `0`, treated as
no-snapshot by `_run_usage_event`). pi's rollout DOES carry token+cost, so
`run.usage` (→ `Run.usage`, `Session.stats.tokens/cost/context_used`) is
implemented, unlike the deviations note's provisional "no `run.usage`".

## Watch wiring + session binding

- Add `~/.pi/agent/sessions` to `ObserverSettings.default_transcript_roots`
  (and thus `existing_default_transcript_roots`). `TranscriptWatchService`
  watches it recursively; every `.jsonl` change → `observer.tail_file`.
- `transcript_identity_from_path` recognizes pi paths (`.pi` + `agent` +
  `sessions` in parts) and derives the session id from the filename UUID via
  `external_session_id_from_pi_path` → `ses_<32hex>`.
- **Harness-origin binding is implicit** (like claude's external-id
  coincidence): `PiCommandBuilder` passes `_harness_session_id_as_uuid(ses_<hex>)`
  as `--session-id`, so pi writes `<ts>_<uuid>.jsonl` and path-derivation gives
  back the exact harness `ses_<hex>`. No orchestrator pre-bind, no expectation
  registry — nothing in `orchestrator.py` changes. `_resolve_identity` returns
  the base (path-derived) identity; `is_rebound=False`.

## Failure modes & guards

- **Malformed / partial lines**: handled by the existing `tail_file` machinery
  (unterminated line → break + retry next tick; bad JSON → warn + skip). The pi
  parser only sees whole, valid records.
- **Unknown record / role / block types**: debug-logged and skipped; unknown
  `message` roles fall through to a single "unsupported pi shape" warning path,
  never a crash.
- **External-origin pi sessions — DEFERRED (STRETCH).** pi splits cwd (`session`
  record) and model (`model_change` record) across two records, so a single
  record never carries both — `_session_event_if_complete` can't synthesize an
  external `session.updated` the way it does for claude/codex (whose every
  record carries cwd+model). Proper external discovery needs stateful cwd+model
  accumulation across records; deferred as a follow-up. **Guard:** to avoid
  orphaned events + an unbounded `_pending_materialization` buffer for the
  operator's many interactive pi sessions, the observer **skips** publishing a
  pi `message` when the target session doesn't exist in the repository. This
  scopes pi observation to harness-owned sessions (the MUST) without leaking.
  `run.usage`/`run.end_turn` are already dropped when there's no active run.
  When external synthesis lands, this guard is removed.
- **No claude/codex behavior change**: the pi path is additive — a new branch in
  `transcript_identity_from_path` / `parse_transcript_record`, a new
  `_parse_pi_record`, new pure helpers, and one default-root entry. Existing
  tests pass unmodified.

## Test plan (TDD)

1. Path/identity helpers: `external_session_id_from_pi_path`,
   `transcript_identity_from_path` for a pi path (golden filename).
2. Parser golden: parse the copied fixture
   (`tests/fixtures/pi_rollout_golden.jsonl`) → user + assistant `message`
   events with expected text/thinking blocks + `run.usage`.
3. One test per record type + edges: assistant text, thinking+text interleave,
   toolCall → tool_use block, toolResult → role=user tool_result block,
   `stopReason=="stop"` → run.end_turn, metadata ignored w/o warning,
   unknown role/blocks skipped, partial line, multi-turn resume in one file.
4. Observer integration: append to a temp pi rollout under a repo-owned harness
   session → `tail_file` emits `message` (materialized to `list_messages`);
   external (no session) → skipped.
5. Settings/CLI: pi root in defaults.
6. End-to-end: own harness (Node 24), POST pi session+run, assert `message`
   events on the API.
