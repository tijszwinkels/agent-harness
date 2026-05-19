# Phase 4 — Watchdog rewires; supervisor stops parsing; final cleanup

Parent spec: `specs/2026-05-19-unified-ingestion.md`
Parent task: `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.a51a3f0c-c91f-4e57-a9a5-3eb8e416109d`
Worktree: `worktrees/feat/unified-phase-4`
Branch: `feat/unified-phase-4` (off main, post PR #15)

This is the final phase of the unified-ingestion refactor.

## Scope

The supervisor stops parsing stdout entirely. End-turn signaling moves to the observer. Naming cleanups, dead-code pruning, and a few minor noise reductions Orion's PR #15 review flagged as Phase 4 candidates.

After Phase 4 lands:

- The supervisor's stdout handling is **only** a heartbeat tick for the idle watchdog and a stderr → `process.stderr` event passthrough. No parsing, no end-turn detection, no message synthesis.
- The observer emits an `__END_TURN__` event when it sees:
  - claude rollout: `assistant` record with `.message.stop_reason == "end_turn"`
  - codex rollout: `event_msg.payload.type == "task_complete"`
- The watchdog (`RunProcess._watch_end_turn_cleanup`) subscribes to the observer's `__END_TURN__` instead of polling the supervisor's stdout-derived signal.
- `_IGNORED_*` lists no longer carry entries whose only purpose was avoiding dual-path duplication.
- `Session.codex_internal_id` is removed (no longer carries unique information after Phase 2's expectation registry replaced fd discovery).
- Origin tag cleanup: `_source_data` no longer stamps `origin: "external"` on every event — the field served the dual-path arbitration that no longer exists.
- The pre-existing "Unsupported Codex transcript shape" log spam for `turn_context` records (Orion's worth-noting #3) is suppressed.

## Detailed changes

### Observer emits `__END_TURN__` events

In `ExternalTranscriptObserver.publish_line`, when parsing rollout records:

- **claude path**: when the record is `type: "assistant"` AND `.message.stop_reason == "end_turn"`, emit a `run.end_turn` event with `{run_id: <active>, backend: "claude-code"}`. Skip if no active run for the session (pure-external observer-only case).
- **codex path**: when the record is `event_msg/task_complete`, emit `run.end_turn` with `{run_id: <active>, backend: "codex"}`.

The event name is `run.end_turn` (not `__END_TURN__`) so it survives serialization through the durable bus. The watchdog subscribes by event-name match.

### Watchdog rewire

In `RunProcess._watch_end_turn_cleanup`:

- Replace the existing "subscribe to internal `__END_TURN__` signal from stdout parser" with "subscribe to `run.end_turn` events on the event bus, filtered by `session_id == self.session.id` and `data.run_id == self.run_record.id`".
- The 20s SIGTERM / 20s SIGKILL grace cleanup logic stays unchanged.
- Latency: observer-driven end-turn is delayed by rollout flush cadence (usually <1s for claude, <2s for codex). Documented in code comment; acceptable.

### Remove `_detect_end_turn_in_line` and related supervisor stdout parsing

- Delete `_detect_end_turn_in_line` from orchestrator.py.
- Delete the parser-dispatch portion of `_stream_lines` — only the heartbeat tick (`self._last_activity_at = self._clock()`) and stderr forwarding remain.
- Delete `_end_turn_event` ferrying / queue / signal infrastructure that connected stdout-detector to watchdog.
- Tests asserting "stdout parser detects end-turn for watchdog" deleted; replaced by "observer-emitted end-turn drives watchdog" coverage.

### `_IGNORED_*` lists pruning

- `_IGNORED_CLAUDE_RECORD_TYPES`: review each entry. Remove any whose only purpose was "don't double-emit because stdout already did". After Phase 4 only entries with a genuine record-type-doesn't-apply reason remain (e.g., `"system"`, `"attachment"` may still belong).
- `_IGNORED_CODEX_PAYLOAD_TYPES`: same. `token_count` was already removed in Phase 3. Review remaining entries; remove unwarranted suppression. Note `agent_message`/`assistant_message` are duplicates of canonical `response_item/message` per the architecture memory — those entries stay.

Audit comment in code explains which entries survived and why.

### `Session.codex_internal_id` retire

- Removed from `Session` model.
- Removed from `repository.find_session_by_codex_internal_id` (entire helper deleted).
- SQLite migration: `codex_internal_id` column dropped. (Or kept-nullable for backward-compat with old DBs; lean toward "kept" for safety since the field is harmless when null.)
- Removed from OpenAPI schema for `Session`; drift-guard test confirms removal.

### Origin tag cleanup

Audit `_source_data` and related event-data construction:

- The `origin: "external"` tag was used by storage's append_event branches to gate "harness vs external" materialization paths. With Phase 3's single materialization point, this distinction is no longer load-bearing.
- Remove the unconditional `origin: "external"` stamp from `_source_data`.
- If any downstream consumer (mm-bridge, command-bridge) still inspects `origin`, that's their migration to handle — but a search of `bridge.py` / `store.ts` should confirm whether anyone reads it.
- Test coverage: `_source_data` returns expected payload shape without `origin` key.

### Suppress `turn_context` warning noise

- In observer's "Unsupported Codex transcript shape" log path, recognize `turn_context` as a known-skipped record type. Demote to debug-level log, or add to a "known-but-skipped" allowlist that suppresses the warning entirely.
- Same for any other recurrent shapes that emit warnings but are benign.

### Parent spec patch

After all changes, update `specs/2026-05-19-unified-ingestion.md` Phase 4 section to reflect what actually shipped. Mark the refactor complete with a "Status: Done as of 2026-05-19, see PRs #13, #14, #15, #16" header.

## Tests

In `tests/test_observer.py`:

- `test_observer_emits_run_end_turn_for_claude_assistant_with_end_turn_stop_reason`
- `test_observer_emits_run_end_turn_for_codex_task_complete`
- `test_observer_does_not_emit_run_end_turn_without_active_run` — pure-external case.

In `tests/test_orchestrator.py`:

- `test_watchdog_triggers_on_observer_run_end_turn` — watchdog subscribes to event bus; observer publishes `run.end_turn`; watchdog cleanup fires.
- `test_stream_lines_no_longer_runs_parser` — feed stdout lines; assert no `__END_TURN__` signal flows through the orchestrator's internal queue (the queue itself is removed).
- Delete tests that asserted `_detect_end_turn_in_line` produces the right signal — supervisor no longer has this responsibility.

In `tests/test_models.py`:

- `test_session_does_not_carry_codex_internal_id` — drift guard for the field removal.

In `tests/test_openapi.py`:

- `test_openapi_session_schema_does_not_document_codex_internal_id`.

Existing tests touching `_source_data` may need updates if they asserted the `origin` field — update to reflect the new shape.

## Sidecar verification

After implementation:

1. Spin up sidecar harness on port 8879+.
2. Harness-spawn a claude run that ends with `stop_reason == "end_turn"`. Verify watchdog grace fires within a few seconds of the rollout's last write.
3. Same for codex `task_complete`.
4. Idle watchdog still functional: long-silent stdout still ticks `_last_activity_at`; 30-min idle timeout still trips.
5. `GET /v1/sessions/{id}` response doesn't include `codex_internal_id`.
6. `_source_data` output (peek at a few events on the SSE wire) doesn't carry `origin` key.
7. Tail a codex rollout that has `turn_context` records; observer log doesn't fill with warnings.

## What is NOT in Phase 4

- Cost data for claude (deferred indefinitely per Tijs).
- Auto-creating Run records for pure-external sessions (not architecturally needed; revisit if a future feature requires it).
- Backfilling historical sessions.
- Bridge-side migrations (mm-bridge / command-bridge consume the new event shapes; their teams handle on their schedule).

## Self-review checklist

- [ ] All tests pass; no regressions across the refactor.
- [ ] `_detect_end_turn_in_line` and the stdout end-turn signal infrastructure fully removed.
- [ ] Observer emits `run.end_turn` for both backends; watchdog subscribes via event bus.
- [ ] `_IGNORED_*` audit complete; surviving entries documented in code comment.
- [ ] `Session.codex_internal_id` removed from model + OpenAPI + repository helpers; drift-guard test passes.
- [ ] `origin: "external"` tag removed from `_source_data`; no downstream code path requires it.
- [ ] `turn_context` warning suppressed.
- [ ] Parent spec patched with "Status: Done" header + PR list.
- [ ] PR title: `feat(unified-ingestion phase 4): watchdog rewires; supervisor stops parsing; final cleanup`
- [ ] Sidecar smoke executed end-to-end for all listed scenarios.

## Workflow change — DRAFT PR FIRST

To break the recurring "work done but not pushed" stall pattern (3x in Phases 1, 2, 3), this phase uses a different workflow:

1. **Before substantive work**: open a draft PR with just the spec file as the first commit. Push immediately.
   - `git add specs/2026-05-19-unified-ingestion-phase-4.md && git commit -m 'docs: Phase 4 spec'`
   - `git push -u origin feat/unified-phase-4`
   - `gh pr create --draft --title 'feat(unified-ingestion phase 4): watchdog rewires; supervisor stops parsing; final cleanup' --body '<reference parent spec + this Phase 4 spec + parent task>'`
2. **Each subsequent commit**: push immediately after committing. `git push` is part of the same hand motion as `git commit`.
3. **After last commit + self-review**: `gh pr ready` to convert the draft → ready-for-review. Then post DONE in Echo's channel.

This converts the wrap-up from a discrete end-of-task event into a per-commit habit. The push step happens four or five times instead of once; no single push-then-stall window can swallow it.

## Out-of-band reminders

- Orphan-bash rule: no SSE curl without `-m`, no `httpx.stream()` in tests.
- Sidecar only; don't touch `:8877`.
- `/codex:review` (or claude self-review pattern) before flipping the PR out of draft.
- Final DONE post in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`: `Solace: DONE see ~s-feefd6b41023~ PR #<N>`.

## Estimated effort

~1.5 dev days. Mostly deletes + a single non-trivial wire (`observer → event bus → watchdog` for end-turn). The cleanup items are mechanical.
