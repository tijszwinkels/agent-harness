# Codex Session Deduplication

Filed: 2026-05-18 — task `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.6e22258a-1706-4506-9f4c-ded2c21768e3`
Worktree: `worktrees/feat/codex-session-dedupe`
Branch: `feat/codex-session-dedupe`
Reporter: Heron (command-bridge, channel `hbitur1fxtyxxk1h9j7zhd3drc`)

## Problem

Every codex harness spawn produces TWO session rows in the harness DB:

1. `ses_<hex>` with `origin=harness` — created by `POST /v1/sessions`. Gets the title we want, gets a handful of stdout-parsed messages, never receives the bulk of activity.
2. `codex_<rollout-uuid>` with `origin=external` — registered by the rollout observer when it sees the codex JSONL appear. Gets the full message volume; `title=null`.

Same backend process. Two DB rows. Confirmed via `GET /v1/backends` that `codex.session_id_choice=false` — the harness can't pin codex's rollout UUID at spawn time (codex `exec --json` has no `--session-id` flag).

Symptom in command-bridge: duplicated session lanes for every codex harness spawn.

## Reference architecture

- **claude-code path (works correctly today)**: `--session-id <ses_hex>` pins the rollout filename, observer recognizes it as belonging to the harness session, dedupes by exact id match. `session_id_choice=true`.
- **codex path (broken)**: no flag to pin id, observer creates a new external row keyed by codex's internal UUID. `session_id_choice=false`.

## Fix path (per agreement with Heron)

Add a reconcile pre-check in the observer's session-registration flow. Before registering a new `origin=external` codex session, look for an outstanding `origin=harness` codex session that this rollout almost certainly belongs to.

### Match criteria (all must hold)

- Same `project.path` (the rollout's cwd matches the harness session's stored project path).
- `Session.backend == "codex"`.
- Harness session's status is non-terminal (`running` or `idle`, NOT `completed`/`error`/`interrupted`).
- Time window: `|Session.created_at - rollout_earliest_event_ts| <= 30s`. Pick the rollout's earliest `timestamp` field (typically the `session_meta` record at index 0).

### On match

- Do NOT register a new external row.
- Bind the rollout file to the harness session in the observer's offset tracker (`set_observer_offset(path, ...)` keyed against the harness session id).
- All subsequent message events from the rollout attach to the harness session row.
- Session.updated_at flows as normal.
- If the harness session has a `codex_internal_id` field (check `models.py:Session`; add if not present), populate it with the rollout UUID so future restarts can rebind without re-scanning.

### On no match

- Existing path: register a new external session row keyed by `codex_<uuid>`.

### Terminal-state skip

If a candidate harness session matches by cwd+window but its status is already terminal, do NOT reconcile. Register the external row as today. Rationale: a post-mortem replay of a completed harness session is a legitimate independent observation, not a dupe.

## Code touch points (verify and refine in your investigation phase)

- `src/agent_harness/observer.py` — the `ExternalTranscriptObserver` and the spot where it registers a new session for a freshly-discovered rollout file. Likely in or near `TranscriptWatchService` / a method that handles "new rollout file appeared".
- `src/agent_harness/repository.py` / `storage.py` — may need a new query: `find_recent_harness_codex_session(cwd, ts_window) -> Session | None`.
- `src/agent_harness/models.py` — possibly add `Session.codex_internal_id: str | None` (only if existing data model can't tie back via the offset table alone).

**Read the existing code carefully before deciding on field additions.** It is possible the offset table is sufficient to keep the mapping and no new model field is needed. Prefer no schema change if you can avoid it.

## Coordination

- **Vega is currently editing `observer.py`** on branch `feat/usage-tokens-context-window`. Their PR is in flight (usage tokens + context window). Whoever lands second rebases on the first. Talk to Vega via Echo's channel if you anticipate a real conflict — most likely you'll edit different functions.
- DO NOT touch `_IGNORED_CODEX_PAYLOAD_TYPES` or message-routing logic. Vega owns that area.

## Tests

Required (TDD — failing test first):

- `test_observer_reconciles_codex_rollout_to_harness_session` — create a harness session via `POST /v1/sessions`, drop a fixture codex rollout in the watched dir with `cwd` matching and `session_meta.timestamp` within 30s of session creation. Assert: no `codex_<uuid>` row is created; the harness session receives the rollout's message events; offset tracker has the rollout path keyed against the harness session id.
- `test_observer_does_not_reconcile_when_cwd_differs` — same setup but rollout cwd differs. Assert: a new external row is registered as today.
- `test_observer_does_not_reconcile_outside_time_window` — rollout timestamp 60s after session creation. Assert: external row registered.
- `test_observer_does_not_reconcile_terminal_session` — harness session is `completed` before rollout appears. Assert: external row registered.
- `test_observer_does_not_reconcile_claude` — claude-code rollout with cwd-match doesn't trigger reconcile (only codex). The `session_id` match path still works as today.
- (If you add `codex_internal_id`) `test_session_codex_internal_id_persists_across_restart` — observer restart re-binds without rescanning.

## Out of scope

- Backfilling historical dupes (operator manual cleanup; document in PR).
- Anything in claude-code's flow.
- Anything in message-routing or payload parsing (Vega's territory).
- Renderer-side dedupe in command-bridge (Heron's call; held pending @tijs decision).

## Verification on sidecar

After implementing:

1. Spin up sidecar harness on port 8879+ with temp DB.
2. `POST /v1/sessions` with `backend=codex`, send a prompt that triggers a real codex run.
3. List sessions: only ONE row should appear (the harness one), with full message count.
4. Run the same scenario twice in a row — distinct sessions, no cross-pollination.
5. External codex session (drop a rollout file under a cwd that has no matching harness session) — verify it still registers as `codex_<uuid>` external row.

## Self-review checklist

- [ ] All new tests pass; existing tests not regressed.
- [ ] OpenAPI updated if any model fields changed; drift-guard test passes.
- [ ] Reconcile logic gated to `backend=codex` only (don't accidentally affect claude).
- [ ] Terminal-state skip implemented.
- [ ] Sidecar verification executed; counted sessions match expectations.
- [ ] PR description references this spec path + task ref `6e22258a` + Heron's report.

## Out-of-band reminders

- Don't run SSE `curl` without `-m N`. Don't use `httpx.stream()` in tests. (Orphan-bash circuit-breaker.)
- Don't touch the live :8877 daemon. Sidecar only.
- `/codex:review` before opening the PR.
- Final completion: post `<persona>: DONE see ~<your-channel-slug>~ PR #<N>` in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`.
