# Phase 3 — Single materialization path; observer materializes run.usage from rollouts

Parent spec: `specs/2026-05-19-unified-ingestion.md`
Parent task: `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.a51a3f0c-c91f-4e57-a9a5-3eb8e416109d`
Worktree: `worktrees/feat/unified-phase-3`
Branch: `feat/unified-phase-3` (off main, post PR #14)

## Scoping note

Parent spec's Phase 3 also included the watchdog rewire from stdout end-turn detection to observer-emitted end-turn events. That work has been **deferred to Phase 4** alongside the parent spec's planned cleanup items (`_IGNORED_*` lists, `Session.codex_internal_id`, dead origin tags). Reason: Phase 3 as originally specced was too large; the architectural change to bus.publish + the cherry-pick from PR #11 already make this a substantial PR. Splitting gives smaller, reviewable units.

So Phase 3 = data-plane unification. Phase 4 = supervisor cleanup + watchdog rewire + naming cleanup.

## Scope

Two reinforcing changes:

1. **`DurableEventBus.publish` becomes the single materialization point.** Today `repo.append_event` does insert + side-effect materialization (run.usage, run lifecycle), and the observer separately calls `repo.materialize_event` for events it itself published — risking double-application. Phase 2 removed the message carve-out; Phase 3 finishes the pattern by moving materialization from `append_event` into `bus.publish` (which always runs the side-effects in one place after the insert). Falcon's double-count bug (closed PR #11) becomes structurally impossible.
2. **`run.usage` is populated from rollout content** by the observer, using parsers cherry-picked from closed PR #11. `SessionStats.context_window` field is added.

After Phase 3, every event materializes exactly once, in one method (`bus.publish` for durable; `repository.materialize_event(store_event=True)` for in-memory). The "two-path materialize + origin carve-out" anti-pattern is gone.

## Architectural change: single materialization point

Today's flow (post-Phase-2):

```
supervisor._publish(event) ──► bus.publish(event) ──► repo.append_event(event)
                                                       │
                                                       ├── insert event row
                                                       ├── if event in RUN_LIFECYCLE_EVENTS: materialize lifecycle
                                                       └── if event == run.usage: materialize usage
                                                       (message carve-out already removed in Phase 2)

observer.tail_file ──► observer.bus.publish(event) ──► (same chain as above)
                  └──► observer._materialize_or_buffer(event)
                       └──► repo.materialize_event(event, store_event=False)
                            └── re-runs lifecycle + usage materialization
                            (this re-application was the carve-out's purpose; in Phase 3 we eliminate the double-call architecturally)
```

Phase 3 flow:

```
supervisor._publish(event) ──► bus.publish(event)
                               ├── repo.append_event(event)        # pure insert, no side effects
                               └── repo.materialize_event(event,   # all side effects
                                                          store_event=False)

observer.tail_file ──► observer.bus.publish(event) ──► (same chain)
                  └──► (no longer calls materialize_event for the same event)
                       (buffering on SessionNotFoundError moves to publish call-site)
```

Concrete changes:

- `SQLiteRepository.append_event`: strip the materialization branches. Becomes a pure event-row insert. Returns the published event with sequence assigned.
- `InMemoryRepository.append_event`: same.
- `DurableEventBus.publish`: call `append_event` then `materialize_event(store_event=False)`. Both wrapped in the existing lock so they're atomic from the subscriber's perspective.
- `repository.materialize_event(event, store_event=...)`: continues to be the pure materialization function. When called from `bus.publish` with `store_event=False`, runs all side-effects (lifecycle, usage, message, session.updated) WITHOUT re-inserting.
- `ExternalTranscriptObserver._materialize_or_buffer`: simplifies. Drops the `repository.materialize_event` call for events being published through the bus (the bus does it now). Keeps the buffering logic for `SessionNotFoundError` — but the exception now bubbles from `bus.publish`, so the catch moves to the publish call-site (`tail_file` / `publish_line`).
- In-memory test fixtures: the `materialize_event(store_event=True)` path is still used by tests that don't go through the bus. No change there.

This is the right shape to dissolve the original design smell Tijs flagged early in this refactor ("why is materialization invoked from two places?"). After Phase 3, the answer is: it isn't.

## run.usage materialization from rollouts

Cherry-pick from closed PR #11:

- `src/agent_harness/usage.py` — Vega's parser module:
  - `parse_claude_usage(record: dict) -> Usage` — parses `.message.usage` from claude rollout `assistant` records.
  - `parse_codex_token_count(payload: dict) -> tuple[Usage, int | None]` — parses `event_msg.payload.type=token_count`; returns `(per-turn usage, context_window)`. Uses `last_token_usage` for the Usage (NOT `total_token_usage` — that would double-count across runs in a session).
- `SessionStats.context_window: int | None` field — codex-only; updated on each `token_count` event (latest-seen value).
- OpenAPI documentation for `context_window` in `specs/openapi.yaml`.
- Rollout-fixture tests for both parsers.

Wire-up: in `ExternalTranscriptObserver.publish_line` (or wherever rollout records are parsed today), when the record is an assistant message with usage (claude) or an `event_msg/token_count` (codex):

- Compute Usage via the parser.
- Publish a `run.usage` event with the Usage payload AND the active run_id (resolved from the session's active run).
- For codex `token_count`, additionally publish a `session.stats_update` (or similar — see open question below) with `context_window` so the session row updates.

Skip the publish if no active run exists (pure-external sessions where the harness never created a Run record — same edge Falcon flagged on PR #11 as "worth-noting #1"). Document this limitation in a code comment; revisit in Phase 4 or later when we decide whether to auto-create runs for external sessions.

Cost data: not in this phase per Tijs's 2026-05-19 call. `Usage.cost_usd` stays zero. Future helper can compute cost from token counts × a price table — separate task.

## What disappears

- Materialization branches inside `SQLiteRepository.append_event` and `InMemoryRepository.append_event` (storage.py:497-512 region — already had message removed in Phase 2; now lifecycle + run.usage move out too).
- The `_materialize_or_buffer` call to `repository.materialize_event` for events going through the bus.
- The `store_event=not bus.stores_events` toggle complexity in observer's call (the path that bypassed the bus is gone).

## What stays

- `repository.materialize_event` itself (used by in-memory bus paths and tests; semantically pure now).
- The buffering logic for `SessionNotFoundError` — relocated to the publish call-site.
- `Run.usage` model field (already there).
- Phase 2's expectation registry, supervisor heartbeat tick, `process.stderr` event, observer being sole writer for messages.

## Tests

In `tests/test_storage.py`:

- `test_append_event_does_not_materialize_run_lifecycle` — append a `run.started` event; assert the Run table is NOT updated by `append_event` alone.
- `test_append_event_does_not_materialize_run_usage` — append a `run.usage` event; assert Run.usage is NOT updated by `append_event` alone.
- Replace existing tests that relied on `append_event` materializing — they should now exercise the `bus.publish` path.

In `tests/test_events.py` (or wherever DurableEventBus tests live):

- `test_durable_publish_materializes_run_lifecycle` — call `bus.publish(run.started)`; assert Run is now in `running` state.
- `test_durable_publish_materializes_run_usage` — assert Usage applied to Run.
- `test_durable_publish_propagates_session_not_found` — assert observer's publish call-site can catch and buffer.
- **Regression test**: `test_durable_publish_does_not_double_count_run_usage` — Falcon's repro: claude usage `(6/4/18/21)` → result `(6/4/18/21)`, NOT `(12/8/36/42)`. Locks in the structural fix.

In `tests/test_observer.py`:

- `test_observer_publishes_run_usage_from_claude_assistant_record` — drop a claude rollout with `.message.usage`; assert a `run.usage` event lands with the correct token counts.
- `test_observer_publishes_run_usage_from_codex_token_count` — drop a codex rollout with an `event_msg/token_count` record; assert `run.usage` lands AND `session.stats_update` (or equivalent) carries `context_window`.
- `test_observer_handles_initial_codex_token_count_with_null_info` — codex's first `token_count` after session start has `info: null`. Must not crash; must not emit a zero-usage `run.usage`.
- `test_observer_skips_usage_publish_when_no_active_run` — pure-external session with no Run; assert no `run.usage` is published. Document the gap.

In `tests/test_usage.py` (new file, from PR #11 cherry-pick):

- All of Vega's parser unit tests for claude `.message.usage` and codex `token_count`.

## Sidecar verification

After implementation:

1. Spin up sidecar harness on port 8879+ with temp DB.
2. `POST /v1/sessions` with backend=claude-code; run a real claude session.
3. After the run completes, `GET /v1/sessions/<id>` → `stats.tokens` has non-zero `input`/`output`/`cache_read`/`cache_creation`; `Run.usage` matches the `result` block from claude's last turn.
4. Same for backend=codex. Additionally: `session.stats.context_window` is populated (e.g., 258400 for opus or whatever codex's `model_context_window` reports).
5. Run a long multi-turn claude session; assert usage SUMS across turns rather than overwriting.
6. Trigger Falcon's double-count repro path manually (run both observer + supervisor against the same rollout) — verify usage is NOT doubled.

## Migration / consumer impact

- mm-bridge: no schema changes; `run.usage` events were already a defined shape, just zero-valued. mm-bridge already passes through. Nothing to do.
- command-bridge: same — it reads `Run.usage` from the GET endpoint. Will now show non-zero values automatically. Token-counts UI starts working for the first time.
- Heron's blocked command-bridge task `2bcf6ae5` unblocks after Phase 3 lands.

## What is NOT in Phase 3

- Watchdog rewire to observer-emitted end-turn (Phase 4).
- Removal of supervisor's `_detect_end_turn_in_line` (Phase 4 — kept in Phase 3 because watchdog still uses it).
- `_IGNORED_*` lists pruning (Phase 4).
- `Session.codex_internal_id` decision (Phase 4).
- Dead origin-tag cleanup (Phase 4).
- Cost data for claude (deferred indefinitely per Tijs).
- Auto-creating Run records for pure-external sessions (revisit in Phase 4+ or as a separate task).

## Self-review checklist

- [ ] All tests pass; no regressions.
- [ ] `bus.publish` calls `materialize_event` exactly once per event; observer no longer calls materialize_event for bus-published events.
- [ ] `append_event` in both SQLite and in-memory repos: pure insert; no side-effect dispatch.
- [ ] Falcon's double-count repro is a passing regression test.
- [ ] `usage.py` cherry-picked verbatim where possible; only adapt imports.
- [ ] `SessionStats.context_window` field on the model + OpenAPI; drift-guard test passes.
- [ ] Sidecar smoke executed; non-zero usage verified for both backends; context_window populated for codex.
- [ ] PR title: `feat(unified-ingestion phase 3): single materialization point; observer writes run.usage`
- [ ] PR body references parent spec, Phase 3 spec, parent task, and closes-out Falcon's PR #11 finding + Heron's command-bridge task `2bcf6ae5`.

## Out-of-band reminders

- Orphan-bash rule: no SSE curl without `-m`, no `httpx.stream()` in tests.
- Sidecar only; don't touch `:8877`.
- `/codex:review` (or claude self-review) before PR.
- **Wrap-up is a single non-interruptible sequence**: after the last commit, push → open PR → DONE post in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`. Don't pause between them — the 30-min idle watchdog will fire and we re-do this conversation.

## Estimated effort

~2-3 dev days. The architectural change is small in LOC but conceptually load-bearing; the cherry-pick is well-defined; the tests are the bulk of the work.

## Open question for Solace to call

The codex `token_count` event carries `context_window` as a session-level value. Should that propagate as:

- (a) A new `session.stats_update` event with `{context_window: int}` payload, materialized into `SessionStats`.
- (b) Embedded in the `run.usage` event with an optional `context_window` field, materialized into the parent session.
- (c) Both — a `run.usage` for the per-run delta AND a session-level event for the window.

My lean: (b). Avoids inventing a new event type; the materializer in `materialize_event` for `run.usage` can update both the Run and the Session in one pass. Solace's call after reading the existing event shape conventions; default to (b) unless it pushes back against the materializer's structure.
