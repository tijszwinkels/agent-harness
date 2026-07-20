# Live context size — `Session.stats.context_used`

Filed: 2026-05-19 — Heron's task `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.0acee33a-a6ea-42de-8509-1f038be7eecd`
Worktree: `worktrees/feat/context-used`
Branch: `feat/context-used`

## Why

`Session.stats.tokens` ships today as the **cumulative cost-rollup** — sum of `Run.usage` across every run in the session. For claude-code sessions with long tool-use loops, `cache_read` can exceed 10M while the model's actual context window is 200K. The cumulative number is correct for billing but useless for a "how full is the context window?" indicator on command-bridge.

A separate per-session **live snapshot** is needed — overwrite-not-sum, representing "how many tokens are currently loaded into the context as of the most recent turn." VibeDeck implements this for both backends (verified paths in Heron's task body).

## Field

```python
class SessionStats(HarnessModel):
    tokens: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = Field(default=0, ge=0)
    context_window: int | None = Field(default=None, ge=1)
    context_used: int | None = Field(default=None, ge=0)  # NEW
```

Semantics:
- **Optional**: `None` until the first usage observation lands; observer-only sessions without a usage stream stay `None`.
- **Overwrite-not-sum**: each observed update REPLACES the previous value. Decreases (e.g., after context compaction) are valid and welcomed.
- **Per-session**: not per-run. A session with multiple runs reflects the latest turn's snapshot regardless of which run owned that turn.

## Sources per backend

### codex

The rollout's `event_msg/token_count` events carry `info.total_token_usage` as a cumulative sum since session start (NOT a per-turn delta — that's `last_token_usage`). After each turn, codex emits a fresh `token_count` whose `total_token_usage.total_tokens` reflects current loaded context.

Source field: `info.total_token_usage.total_tokens`.

Reference: VibeDeck `backends/codex/pricing.py:91`.

### claude-code

Claude rollout `assistant` records each carry a `message.usage` block. For an assistant turn:
```
context_used = usage.input_tokens
             + usage.cache_creation_input_tokens
             + usage.cache_read_input_tokens
```

Note: `output_tokens` is excluded. Output is the model's response; it's not part of the context that's loaded for the NEXT turn. The snapshot represents "what was loaded into context for THIS turn", which equals "what will be loaded for the next turn modulo growth".

The latest assistant record's value is the current snapshot; earlier records are stale.

Reference: VibeDeck `templates/static/js/utils.js:201`.

## Event model decision

Two reasonable options. Spec recommends **option B** but Solace may swap to **option A** after reading the materialization code if it's cleaner.

**Option A — extend `run.usage` events with an optional `context_used` field.**
- Pro: reuses existing event type and materialization branch.
- Pro: timing aligns — both backends emit usage updates at the same cadence we'd emit snapshot updates.
- Con: semantically conflates per-run-delta and per-session-snapshot on one event.

**Option B — new `session.stats_update` event with `{context_used: int | None}` payload (recommended).**
- Pro: clean separation; `run.usage` stays a pure per-run-delta event.
- Pro: extensible — future per-session stats (e.g., `last_turn_cost_usd`) ride the same event.
- Con: a new event type to add through bus, materializer, drift-guard.

The materializer for either path overwrites `Session.stats.context_used` (no aggregation).

## Code changes

### `src/agent_harness/usage.py`

For codex, add a helper that pulls the snapshot value:
```python
def parse_codex_context_snapshot(payload: dict) -> int | None:
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    total = info.get("total_token_usage")
    if not isinstance(total, dict):
        return None
    return int(total.get("total_tokens", 0)) or None
```

For claude, add:
```python
def parse_claude_context_snapshot(usage: dict) -> int:
    return (
        int(usage.get("input_tokens", 0))
        + int(usage.get("cache_creation_input_tokens", 0))
        + int(usage.get("cache_read_input_tokens", 0))
    )
```

Existing `parse_codex_token_count` and `parse_claude_usage` are unchanged — they continue to return per-run-delta usage.

### `src/agent_harness/observer.py`

Where the existing `run.usage` emission happens for each backend:

- codex: alongside computing the per-run delta from `last_token_usage`, also compute the snapshot from `total_token_usage`. Emit the snapshot via the chosen event shape (option A: include in run.usage; option B: emit separate session.stats_update).
- claude-code: from each `assistant` rollout record's `message.usage`, compute the snapshot. Emit the same way.

Skip snapshot emission if the value is `None` (codex's first `token_count` after session start has `info: null`).

### `src/agent_harness/models.py`

Add `context_used: int | None = Field(default=None, ge=0)` to `SessionStats`.

### `src/agent_harness/storage.py` and `repository.py`

Materializer for the chosen event applies `session.stats.context_used = new_value` (overwrite). Does NOT touch `tokens` or `cost_usd`.

### `specs/openapi.yaml`

Document `context_used` on `SessionStats`. Document the new event type if option B.

### Tests

In `tests/test_usage.py`:
- `parse_codex_context_snapshot` returns `total_tokens` when info present.
- `parse_codex_context_snapshot` returns `None` when info is `None` or missing.
- `parse_claude_context_snapshot` sums input + cache_creation + cache_read; ignores output.

In `tests/test_observer.py`:
- Drop a claude rollout with two assistant records; assert `Session.stats.context_used` reflects the LATEST record's snapshot, not a sum.
- Drop a codex rollout with two `token_count` events; assert same.
- Snapshot decrease (e.g., after compaction) is preserved.

In `tests/test_storage.py` (and `test_repository.py`):
- Materializer overwrites `context_used`; doesn't accumulate.
- `context_used` is independent of `tokens` (existing tokens behavior unchanged).

In `tests/test_openapi.py`:
- Drift-guard for the new field.

## What this is NOT

- A cost calculator. `cost_usd` stays zero per the maintainer's deferred-indefinitely call.
- A backfill for existing sessions. New data only; pre-existing sessions stay `context_used = None` until they next run.
- A change to the cumulative `tokens` semantics. That field continues to be sum-across-runs for cost rollup.
- A heuristic about "how close to context_window". Command-bridge computes `context_used / context_window` on its side; harness just ships the inputs.

## Sidecar verification

After implementation:

1. Spin up sidecar harness :8879+.
2. Harness-spawn a claude session with a multi-turn prompt that accumulates context (e.g., a few tool-use loops).
3. After each turn, `GET /v1/sessions/<id>` and assert `stats.context_used` is non-zero, in the same order of magnitude as the latest assistant message's input + cache_*.
4. Assert `stats.tokens.cache_read` is much larger than `stats.context_used` after several turns — that's the proof that the cumulative vs snapshot distinction works correctly.
5. Same flow for codex; additionally verify `context_used <= context_window`.

## Self-review checklist

- [ ] All tests pass; no regressions.
- [ ] `context_used` semantically NEVER sums; overwrites only.
- [ ] First codex `token_count` with `info: null` doesn't crash or zero out.
- [ ] Multi-turn flow: latest snapshot wins; earlier snapshots don't leak.
- [ ] OpenAPI updated; drift-guard passes.
- [ ] Sidecar smoke verifies the cumulative-vs-snapshot distinction on real data.
- [ ] PR title: `feat: Session.stats.context_used (live context snapshot)`
- [ ] PR body references this spec + Heron's task `0acee33a` + companion command-bridge task `2bcf6ae5`.

## Workflow

**Draft-PR-first** (the workflow that broke the stall pattern in Phase 4):
1. Commit this spec, push, open draft PR — before substantive work.
2. Push after each commit (PR auto-updates).
3. `gh pr ready` + DONE post at the end.

## Estimated effort

~1 dev day. Smaller than any phase of the refactor; well-scoped.

## Out-of-band reminders

- No SSE curl without `-m`; no `httpx.stream()` in tests.
- Sidecar only; don't touch `:8877`.
- `/codex:review` before flipping draft to ready.
- Final DONE in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`.
