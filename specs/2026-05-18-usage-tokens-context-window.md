# Usage Tokens & Context Window

Filed: 2026-05-18 — task `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.90198926-0cbf-4df7-978a-2b45dcea61d5`
Worktree: `worktrees/feat/usage-tokens-context-window`
Branch: `feat/usage-tokens-context-window`
Companion (blocked on this): command-bridge `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.2bcf6ae5-815b-48f6-bd66-0fc723483259`

## Problem

`Run.usage` (`models.py:130`) and `SessionStats.tokens` (`models.py:46`) are declared but never written. All 188 live sessions on `:8877` return zeros. There is no `context_window` field at all — only the model name string.

## Goal

Populate token usage + cost on every `Run`, aggregate them into `Session.stats.tokens`, and surface codex's `model_context_window` so command-bridge can render a fill-meter.

## Upstream event shapes (verified)

### claude-code, rollout JSONL (external observer path)

Each `type: "assistant"` record carries:

```json
"message": {
  "usage": {
    "input_tokens": 6,
    "cache_creation_input_tokens": 21949,
    "cache_read_input_tokens": 18976,
    "output_tokens": 221,
    "service_tier": "standard",
    "cache_creation": {...},
    "iterations": [...]
  }
}
```

No `total_cost_usd` in rollouts (claude doesn't store it server-side).

### claude-code, stream-json (orchestrator stdout, harness-origin runs)

`{"type":"result", "stop_reason":"end_turn", "usage": {...same shape as above...}, "total_cost_usd": 0.123}` at turn end. Cost is only available on this path.

### codex, rollout JSONL (external + orchestrator both observe this)

```json
{"type":"event_msg","payload":{
  "type":"token_count",
  "info":{
    "total_token_usage": {"input_tokens":4342,"cached_input_tokens":1024,"output_tokens":75,"reasoning_output_tokens":0,"total_tokens":4417},
    "last_token_usage": {...same shape, this turn only...},
    "model_context_window": 258400
  }
}}
```

**Important quirks**:

- The *first* `token_count` after session start has `"info": null` — must handle gracefully.
- `total_token_usage` is **cumulative since session start**, `last_token_usage` is just this turn.
- `reasoning_output_tokens` is *included in* `output_tokens` in codex's accounting (verified in their docs). Do not double-count.

## Model changes

### `Usage` (extend, keep backward-compatible defaults)

`Usage` already has `input`, `output`, `cache_read`, `cache_creation`, `cost_usd`. Keep these. No new fields on `Usage`.

Mapping:

| Source | Source field | Maps to |
|---|---|---|
| claude usage | `input_tokens` | `input` |
| claude usage | `output_tokens` | `output` |
| claude usage | `cache_read_input_tokens` | `cache_read` |
| claude usage | `cache_creation_input_tokens` | `cache_creation` |
| claude result | `total_cost_usd` | `cost_usd` (claude only; codex stays 0) |
| codex token_count | `last_token_usage.input_tokens` | `input` (per-turn delta) |
| codex token_count | `last_token_usage.output_tokens` | `output` |
| codex token_count | `last_token_usage.cached_input_tokens` | `cache_read` |
| codex | (no equivalent) | `cache_creation` stays 0 |

### `SessionStats` (extend)

Add **one** new optional field:

```python
class SessionStats(HarnessModel):
    tokens: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = Field(default=0, ge=0)
    context_window: int | None = Field(default=None, ge=1)  # NEW
```

Rationale for `int | None` at session level (not run level):

- Codex only. Claude doesn't expose a context-window number.
- Changes slowly within a session (only when codex compacts or upgrades model mid-session, which is rare).
- Command-bridge wants "current window for fill-meter" — session-level is the right granularity.
- Store `latest_seen` value. Don't track history.

### `SessionStats.tokens` aggregation strategy

Sum of all `Run.usage.{input, output, cache_read, cache_creation}` across runs in the session. Document this in the field docstring. Backend semantics differ slightly (codex's per-run delta vs claude's per-turn) but both end up as a meaningful sum of per-run usage.

Shape:

```python
stats.tokens == {
    "input": 12345,
    "output": 678,
    "cache_read": 100000,
    "cache_creation": 50000,
}
```

`stats.cost_usd` = sum of `Run.usage.cost_usd` (claude only; codex contributes 0).

## Code changes

### 1. `src/agent_harness/observer.py`

- Remove `"token_count"` from `_IGNORED_CODEX_PAYLOAD_TYPES`. Handle it before the generic message-event path: extract `last_token_usage` and `model_context_window`, route through a new observer-side `_apply_usage(run, usage, context_window=None)` helper that updates the active Run's `usage` and the parent Session's `stats`. Do **not** emit a `message` event for token_count (don't pollute chat stream).
- For claude rollouts: in the path that consumes `type: "assistant"` records, after creating/updating the corresponding Run, also call `_apply_usage(run, parse_claude_usage(record.message.usage))`.
- `_apply_usage` adds to current Run.usage (additive) and re-aggregates session stats.

### 2. `src/agent_harness/orchestrator.py` (stdout parsers)

- Claude stream-json parser: on `{"type":"result","usage":...,"total_cost_usd":...}`, write `Run.usage` and update session stats.
- Codex stdout parser: on each `token_count` event seen via stdout (`event_msg.payload.type == "token_count"`), apply same logic. (Note: codex stdout via `exec --json` should emit the same `event_msg` shape; verify on the harness sidecar before relying on it. If codex stdout doesn't emit `token_count`, the observer path covers it.)

### 3. `src/agent_harness/repository.py` and `storage.py`

- `update_run_usage(run_id, usage: Usage) -> Run` — replaces (not appends) the run's usage. SQLite column: serialize as JSON in `runs.usage_json` (or whatever existing column name is — inspect first; may need a migration).
- `update_session_stats(session_id, stats: SessionStats) -> Session` — same pattern.
- If a migration is required for new columns or for `context_window`, add it under the existing schema-migration mechanism (look for `schema_migrations` table usage).

### 4. `src/agent_harness/models.py`

- Add `context_window: int | None = Field(default=None, ge=1)` to `SessionStats`.

### 5. `specs/openapi.yaml`

- Update `SessionStats` schema to include `context_window`.
- The `Usage` schema is unchanged but ensure example values reflect non-zero realistic usage.

### 6. Tests (TDD; write the failing test first)

Required in `tests/`:

- `test_observer_claude_usage_from_rollout` — replay a fixture claude rollout, assert `Run.usage` populated correctly across multiple assistant records.
- `test_observer_codex_token_count` — replay codex rollout including the initial `info: null` event and several populated `token_count` events; assert per-turn `Run.usage` and session-level `context_window`.
- `test_observer_token_count_does_not_emit_message` — assert no SSE `message` event is emitted for `token_count` payloads.
- `test_session_stats_aggregation` — multi-run session, assert sums match individual runs.
- `test_orchestrator_claude_stream_json_result_usage` — feed a fake stream-json `result` line, assert `Run.usage.cost_usd` and tokens populated.
- `test_models_session_stats_context_window_field` — pydantic validation, including the `ge=1` constraint.
- `test_openapi_documents_context_window` — drift guard in the same style as `test_openapi.py`'s existing tests.

## Out of scope (do NOT do in this PR)

- Cost estimation for codex (no upstream cost data).
- Per-run `context_window`. Only session-level.
- Backfill of historical sessions. New data only — operator can re-run the harness to refresh.
- Token cost projection / fill-meter UI logic. That's command-bridge `2bcf6ae5`.
- Splitting `reasoning_output_tokens` out as a separate field.

## Verification on sidecar

After implementing:

1. Spin up a sidecar harness on port 8879 with a temp DB (do NOT touch :8877).
2. Run a real claude session via the harness; `GET /v1/runs/<id>` should show non-zero `usage`.
3. Run a codex session via `exec --json`; verify either stdout or observer path populates usage + context_window.
4. List sessions on the sidecar; `Session.stats.tokens` should sum across runs; `Session.stats.context_window` should be populated for codex sessions, null for claude.

## Self-review checklist

- [ ] All tests pass; no existing tests regressed.
- [ ] OpenAPI updated, drift test passes.
- [ ] `token_count` removed from `_IGNORED_CODEX_PAYLOAD_TYPES` AND the new handler runs before the generic ignored-types check.
- [ ] First codex `token_count` with `info: null` is silently skipped (no crash, no zeroing of existing usage).
- [ ] Codex `total_token_usage` is NOT used for `Run.usage` (would double-count across runs); only `last_token_usage`.
- [ ] Sidecar smoke test executed; non-zero numbers verified for both backends.
- [ ] PR description references this spec path and includes a sample API response showing non-zero usage.

## Out-of-band reminders for the dev sub-session

- Don't run any SSE `curl` without `-m N` timeout (orphan-bash circuit-breaker rule — see `~/.agents/echo/MEMORY.md`).
- Final completion: post `<persona>: DONE see ~<your-channel-slug>~ PR #<N>` in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca` after the PR opens.
- `/codex:review` for self-review pass before opening the PR.
