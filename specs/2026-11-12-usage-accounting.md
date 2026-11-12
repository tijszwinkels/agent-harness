# Usage accounting and provenance

Issue: #48. Supersedes the codex source in `2026-05-19-context-used.md`.

## Why

API consumers (per-session usage views) need token totals and the current
context size they can trust. Three defects made them wrong:

1. **Claude totals inflated.** Claude Code writes one transcript record per
   content block (thinking, text, tool_use) of one API response, and every
   record repeats the response's `message.usage`. The harness added each
   record, so a three-block response counted three times. In a local corpus
   of Claude Code transcripts, 61,217 usage-bearing records described 30,534
   logical messages (about 2× inflation). Within one `message.id` the usage
   was identical, except that `output_tokens` sometimes grew from a partial
   value in early blocks to the final value in the last block (e.g. 8, 8, 375).
2. **Codex context was cumulative.** `stats.context_used` came from
   `info.total_token_usage.total_tokens`, the session's cumulative total. It
   grows past `model_context_window` on long sessions.
3. **Codex cached input counted twice.** Codex follows OpenAI's convention:
   `input_tokens` includes `cached_input_tokens` (codex's own
   `non_cached_input()` is `input_tokens - cached_input_tokens`). The harness
   stored `input = input_tokens` and `cache_read = cached_input_tokens`, so
   the sum of the four token fields counted cached input twice; on long
   sessions most input is cached. Codex also sometimes writes the same
   `token_count` twice; adding `last_token_usage` per record counted that call
   twice.

## Rules (`tokens_version = 2`)

`run.usage` events from claude and codex carry
`accounting: {"version": 2, "key": <identity or null>}`. Both repositories
keep a per-session ledger (`usage_ledger` table in SQLite) of the largest
usage counted per key. The materializer adds only the growth over that
entry, per field, in the same transaction as the event row.

- **Claude**: key `claude:<message.id>`. Repeated content blocks add nothing;
  a partial-then-final report adds the final value once; distinct messages
  with identical numbers each count. Sidechain (subagent) records count
  towards tokens but do not move `context_used`.
- **Codex**: key = the cumulative `total_token_usage` tuple, which identifies
  the call (`TokenUsageInfo::new_or_append` appends each call's
  `last_token_usage` to it). The counted usage is `last_token_usage` with
  `input` = non-cached input. A repeated `token_count`, and the post-compaction
  report (same cumulative total, zeroed `last_token_usage`), add nothing.
  `cache_write_input_tokens` is not mapped: its relation to `input_tokens` is
  undocumented and it has only been observed as 0.
- **Pi**: counting unchanged. Events carry
  `accounting: {"version": 2, "key": null, "additive": true}` and are added
  as before, because pi writes each message once. Unlike claude and codex,
  a pi line replayed after a crash can still count twice (pre-existing
  behaviour). Pi is the only backend that reports `cost_usd`.
- **Events without `accounting`** (stored by an older harness version) are
  added as before but were not de-duplicated. Non-zero usage from such an
  event sets `tokens_version` to `null`. It never promotes a session.
- **No identity** (claude record without `message.id`, codex `token_count`
  without `total_token_usage`): the usage is not counted, and
  `stats.skipped_usage_records` is incremented if it was non-zero. These
  shapes were not observed in practice.
- **Replay** (cursor reset, retried line, restart): the ledger is persisted,
  so re-reading a transcript adds nothing that was already counted. A report
  first read while no harness run was active was never counted. If a cursor
  reset re-reads it during a run, it is counted once and attributed to that
  run; this attribution behaviour already existed.
- **Snapshot freshness** (`context_used` and `context_window`) follows
  transcript order, not accounting keys. Accounting keys identify billable
  usage: a codex compaction report shares its predecessor's key but is a
  newer observation. Both repositories keep the offset of the newest applied
  observation per (session, transcript path); SQLite stores it in
  `usage_positions`, in the event's transaction. A snapshot applies only if
  its line is at a later offset of that transcript. A replay of an older
  line therefore cannot roll either value back. The stored offset only
  moves forward. A report missing from the ledger is not taken as evidence
  of a rewrite: usage read while no harness run is active is never recorded
  there. Notes:
  - Conservative limitation: if a transcript were rewritten or truncated in
    place at the same path, its snapshots would not update until it grew
    past the stored offset. Claude Code and Codex append to their
    transcripts. Token counting is not affected.
  - Different transcript files of one session are ordered independently.
  - An event without a position falls back to keys: it is stale only when
    its key is known and is not the latest one.

### Current context

- **Codex**: `info.last_token_usage.total_tokens`. Codex documents this value
  as "the latest active context size" and computes its own "context left"
  indicator from it. After a compaction codex reports a zeroed
  `last_token_usage` whose `total_tokens` is the compacted size, so the value
  may decrease. Missing or zero values leave the previous snapshot unchanged;
  nothing is derived from the window or the cumulative total.
- **Claude**: `input + cache_creation + cache_read` of the newest main-chain
  assistant record in transcript order (unchanged).
- **Pi**: `input + cacheRead + cacheWrite` (unchanged).

## Consumer contract

New `Session.stats` fields (additive; existing fields keep their shape):

| Field | Values | Meaning |
| --- | --- | --- |
| `tokens_version` | `2` or `null` | `2`: `tokens` and `cost_usd` were accumulated entirely under these rules. `null`: legacy totals, possibly inflated. Do not present them as accurate. |
| `skipped_usage_records` | integer ≥ 0 | > 0: some reports could not be counted, so `tokens` is a lower bound. |
| `context_used_source` | `claude.message.usage`, `codex.last_token_usage`, `pi.message.usage` or `null` | `null`: legacy or unknown provenance. Older versions stored codex's cumulative total here, so do not show `context_used` as the current context. |

Rules for a consumer:

- Show token totals only when `tokens_version >= 2`. Mark them as a lower
  bound when `skipped_usage_records > 0`. The four token fields are disjoint
  and may be summed. An empty `tokens` object (`{}`) means no usage was
  observed (typical for external sessions); it is not a measured zero.
- Show the context only when `context_used_source` and `context_used` are
  both non-null. `context_window` is codex-only.
- Show cost only for pi. `cost_usd == 0` elsewhere means "not reported".
- Coverage: usage is recorded only while a harness run is active. Activity in
  an external terminal is not counted, external-only sessions typically have
  no usage, and `context_used` is the latest snapshot seen during a run, so
  it may be stale after external use.

Example (`GET /v1/sessions/{id}`, abridged):

```json
"stats": {
  "messages": 14,
  "tokens": {"input": 1200, "output": 800, "cache_read": 50000, "cache_creation": 3000},
  "cost_usd": 0,
  "tokens_version": 2,
  "skipped_usage_records": 0,
  "context_window": null,
  "context_used": 54200,
  "context_used_source": "claude.message.usage"
}
```

A legacy session loads as `"tokens_version": null, "skipped_usage_records": 0,
"context_used_source": null` until the next corrected snapshot sets the source.

## Legacy data

- Sessions created by this version start with `tokens_version: 2` (the
  `Session` default). Rows saved earlier have no such key and load as `null`.
- A legacy session keeps `null` when new usage arrives, because its earlier
  totals stay mixed in. The one exception is a legacy session with nothing
  accumulated yet (all token fields and cost zero): it moves to `2` on its
  first verified usage (keyed or pi-additive), since nothing earlier can be
  wrong.
- `context_used` is a snapshot. The next corrected observation replaces the
  value and sets `context_used_source`.
- `Run.usage` has no marker. Runs that finished before this version carry
  legacy values.
- **No reconciliation.** Recomputing old totals would need the transcripts
  plus a mapping of which records fell inside harness runs, which the harness
  did not store. Rather than guess, old totals stay marked untrusted. An
  offline recount tool could be added separately if it is needed.

## Rollback

`SessionStats` validates with `extra="forbid"`, and this version writes
`tokens_version`, `skipped_usage_records` and `context_used_source` into every
session row it saves. The procedure from `2026-10-06-native-session-titles.md`
applies. Before rolling back, restore a pre-deploy database backup, or stop
the harness and strip the keys:

```sql
update sessions set payload = json_remove(payload,
  '$.stats.tokens_version', '$.stats.skipped_usage_records',
  '$.stats.context_used_source');
```

The `usage_ledger` and `usage_positions` tables are separate, and older
versions ignore them.

## Not in scope

No pricing engine or cost for claude/codex, no collection of usage outside
harness runs, no backfill or rewrite of stored totals, and no changes to the
popup or other consumers.
