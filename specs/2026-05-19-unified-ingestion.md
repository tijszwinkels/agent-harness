# Unified Ingestion: Rollout Files as the Single Data Source

Drafted: 2026-05-19 — Echo (PM)
**Status: Done as of 2026-05-19.** Shipped across four phased PRs:
- PR #13 (Phase 1): rollout discovery + observer pre-binding
- PR #14 (Phase 2): observer becomes sole writer for messages; codex expectation registry
- PR #15 (Phase 3): single materialization point; observer writes run.usage
- PR #16 (Phase 4): watchdog rewires; supervisor stops parsing; final cleanup

The dual-path architecture is fully gone. The original design smell
Tijs flagged ("why is materialization invoked from two places?") is
resolved at every layer — `bus.publish` is the single materialization
point, the observer is the sole writer for `message` / `run.usage` /
`run.end_turn`, and the supervisor's stdout pump is purely heartbeat
+ stderr passthrough.

Original worktree: `worktrees/spec/unified-ingestion` (spec drafting).
Implementation worktrees: `worktrees/feat/unified-phase-{1,2,3,4}`.

## Why

Two outstanding bugs share a root cause:

- **Falcon's must-fix on PR #11** — `run.usage` is double-counted under SQLite because materialization is invoked from both `SQLiteRepository.append_event` (storage.py:503) and `materialize_event` (storage.py:543). The existing carve-out at storage.py:505-511 covers `message` but not `run.usage`. The carve-out itself is the design smell.
- **Heron's PR #12** — codex harness spawns produce two session rows because stdout (orchestrator) and rollout (observer) both register sessions, with no way to pin codex's UUID at spawn time.

Both are symptoms of the same architecture: messages and turn data enter the harness through **two paths** (stdout parsing AND rollout tailing) and the system patches the duplication with origin-tagged carve-outs and post-hoc reconciliation.

The visual explainer at `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.a23bc11b-6ace-4a00-a811-c175182c8b72` shows all four current ingestion paths side-by-side.

## Architectural Decision

**Rollout files are the single source of truth for messages, turn events, and usage.** Stdout parsing exits the data plane entirely.

The orchestrator remains, but reduced to a process supervisor:
- spawn the CLI subprocess,
- discover and pre-bind the rollout file to the harness session id,
- emit `run.started` / `run.completed` / `run.failed` / `run.interrupted` from process exit status,
- forward stderr lines as `process.stderr` events (diagnostics, not parsed),
- enforce interrupts via `process.terminate()`,
- drive the watchdog from observer-emitted end-turn events.

The observer is the only writer for `message`, `message.delta`, `run.usage`, `session.updated`, and any backend-derived event we surface in the future.

### Guiding rule (Tijs, 2026-05-19)

> "Still acceptable to pull things from process invocation or even stderr/stdout if we need it, but don't duplicate where we get messages from. That gets messy quickly."

Stderr passthrough, exit codes, interrupt timing, and rollout-discovery probes are all fair game. The forbidden move is producing a `message` (or `run.usage`, or any data also present in the rollout) from anywhere but the observer.

## Components

### Supervisor (renamed-and-thinned `RunProcess`)

What it keeps:
- `_publish` for `run.started`, `run.completed`, `run.failed`, `run.interrupted` with `returncode` and `error_type`.
- Stderr → `process.stderr` event stream (string, one per line, not parsed). This is the diagnostic firehose; the rollout doesn't capture stderr.
- `interrupt()` → `process.terminate()` flow.
- Watchdog (post-end_turn 20s SIGTERM/20s SIGKILL + 30min idle from PR #10), now triggered by observer-emitted end-turn events rather than stdout-parsed `__END_TURN__`.

What it sheds:
- `parse_codex_stream_line`, `parse_claude_stream_line`.
- `_stream_lines` stdout path (parser dispatch, `message.delta` per-line publish, `__END_TURN__` emission).
- `default_stdout_parsers()`.
- Everything in storage.py:505-511 and storage.py:543 that exists to disambiguate dual-path materialization.

Approximate LOC delta: -250 in orchestrator.py, -40 across storage.py / events.py / repository.py for now-dead carve-outs.

### Rollout discovery

**Claude.** Deterministic at spawn time. Orchestrator passes `--session-id <dashed-uuid>` (the 8-4-4-4-12 form, derived from the harness's `ses_<hex>` session id); rollout filename is `~/.claude/projects/<slugified-cwd>/<dashed-uuid>.jsonl`. Supervisor pre-registers the path → harness-session-id binding with the observer immediately after spawn (the file may not exist yet, but the binding does; observer's tail-on-create handles the gap). Phase 1 deviation: Phase 1's first cut composed the path using the `ses_<hex>` form, which never matched the file claude actually writes; fixed in Phase 1's self-review pass (commit `d027228`).

**Codex.** Path is not predictable from the orchestrator's side. Phase 2 uses a content-based **expectation registry** on the observer instead of an fd probe: the orchestrator calls `observer.expect_codex_rollout(cwd, session_id)` synchronously at spawn time; the observer reads `session_meta` from each new codex rollout's first line and matches against active expectations (cwd-exact + ±30s window from `registered_at`, closest-time wins). Expectations have a 10s TTL so failed spawns don't leave permanent hints. Resolution is content-based, so a fast watchfiles inotify firing before/after the orchestrator's spawn-side handoff still routes correctly.

Phase 1 originally probed `psutil.Process(pid).open_files()` synchronously (poll up to 5s) — that approach landed in PR #13 but Sentry's review flagged a residual race between psutil polling and watchfiles. Phase 2 replaces it with the expectation registry; psutil and the lsof fallback are removed.

### Observer (essentially unchanged, but becomes the sole writer)

- `tail_file` / `publish_line` already handle both backends.
- `_materialize_or_buffer` already calls `repository.materialize_event` for harness-bound sessions. After this change, that becomes the only materialization path.
- New: an API or helper for the supervisor to call: `observer.bind_rollout(path, session_id)` — registers `(path → session_id)` in the offset tracker before the file may exist. When `tail_file` later sees the path, the binding is already in place.

### Repository / Storage

- `append_event` stops calling `_materialize_run_lifecycle_event`, `_materialize_run_usage_event`, `_materialize_message_event`. It becomes a pure event-row insert.
- `materialize_event` becomes the single materializer. Its `store_event=…` toggle still controls whether it ALSO writes the event row (when the bus is in-memory) or not (when the bus is durable and the row was inserted by `append_event` upstream).
- The origin-tagged carve-out at storage.py:505-511 is removed.
- `_materialize_run_usage_event` is no longer called from two places, so the carve-out gap that caused Falcon's bug ceases to exist.

### Watchdog source change

Today the watchdog observes `__END_TURN__` synthesized by the orchestrator's stdout parser. After this refactor:

- Watchdog subscribes to the observer's event stream (or reads from the in-process bus) for `session_id` == ours and either `result.stop_reason="end_turn"` (claude) or `event_msg.payload.type="turn_complete"` (codex) materializing into a `run.usage`-adjacent or dedicated end-turn signal.
- Cleanest expression: observer publishes a synthetic `__END_TURN__` event tagged with `session_id` and `run_id`. Watchdog subscribes by `session_id`.

Latency cost: end-turn timing now depends on rollout flush cadence. Empirically claude flushes promptly on `result`; codex flushes on `turn_complete`. Worst-case extra delay before the 20s SIGTERM grace starts: 1-2s. Acceptable.

## What disappears

- All four ingestion paths in the visual explainer collapse to **two**: claude rollout, codex rollout. Both go through the same observer.
- PR #11's design smell (storage.py carve-outs for `message`, soon `run.usage`) — gone.
- PR #12's reconcile-by-cwd-window heuristic — gone, replaced by direct fd-based binding at spawn time.
- The `_IGNORED_CODEX_PAYLOAD_TYPES`/`_IGNORED_CLAUDE_RECORD_TYPES` lists shrink: types whose only purpose was "don't double-emit because stdout already did" can be unsuppressed.
- `Session.codex_internal_id` (added by PR #12) becomes redundant unless we keep it for diagnostic display purposes.

## What stays / what we re-derive

| Concern | Source after refactor |
|---|---|
| Assistant messages | Rollout (observer) |
| User messages | Rollout (observer) |
| Tool use / tool results | Rollout (observer) |
| `run.usage` (tokens, cache) | Rollout (observer); both backends |
| Session-level context window | Rollout (observer); codex only |
| Run lifecycle events | Supervisor process status |
| Exit code / error type on failure | Supervisor `process.wait()` returncode |
| stderr lines (diagnostics) | Supervisor → `process.stderr` event |
| Interrupt | Supervisor `process.terminate()` |
| End-turn for watchdog | Observer derives from rollout; supervisor's watchdog subscribes |
| Cost for claude (`total_cost_usd`) | **Computed** from token counts × known price table; not from upstream (rollout has no cost field). See open question below. |

## Phased implementation

Stages, each landable independently. Hard cut-over per phase — no parallel ingestion paths during transition.

PRs #11 and #12 are **abandoned** rather than landed. The dual-path bugs they patch (Falcon's double-materialization, Heron's dupe-session) become structurally impossible under unified ingestion; treating them as artifacts of the architecture rather than as separable bugs avoids carrying band-aids forward.

**Cherry-pick from PR #11** during Phase 2 implementation:

- `usage.py` rollout parsers (`parse_claude_usage` from `.message.usage` records, `parse_codex_token_count` from `event_msg/token_count`).
- `SessionStats.context_window` model field + OpenAPI documentation.
- Rollout-fixture tests for the parsers.

Everything else in PR #11 (orchestrator stdout parsers, `_should_skip_parsed_event` deduper, double-materialization carve-out attempts) is thrown away because the dual path it serviced no longer exists.

**Cherry-pick from PR #12**: nothing required for correctness. `Session.codex_internal_id` may be worth keeping as a diagnostic field (the codex UUID is useful when humans cross-reference a session to a rollout filename); decide in Phase 1.

**Phase 1 — supervisor refactor, no behavioral change.** (Landed: PR #13, `d651b61`.)
- Added `observer.bind_rollout(path, session_id)` and `observer.unbind_rollout(path)`.
- Added `RolloutDiscovery` module: claude (deterministic path) + codex (psutil fd probe with lsof fallback). Unit-tested with fake processes.
- Supervisor calls `bind_rollout` after spawn; codex bind ran synchronously after Phase 1's self-review pass.
- Stdout parsing still active; observer ingests as before. Result: each event lands in the same session row via both paths; the carve-outs prevent double materialization. No functional change.
- Gated behind `AGENT_HARNESS_ROLLOUT_PRE_BIND=1` env var (default-off in Phase 1; Phase 2 removes the gate).
- Sentry's review of PR #13 flagged a residual race between psutil-polling and watchfiles inotify (worth-noting #2), motivating Phase 2's switch to the expectation registry.

**Phase 2 — observer becomes sole materializer for messages.**
- Remove `parse_codex_stream_line` / `parse_claude_stream_line`. Remove the `message.delta` per-line publish from stdout in the supervisor.
- Remove the message-event materialization from `append_event` (the carve-out at storage.py:505-511); `append_event` no longer materializes messages at all.
- Observer is now the only writer for `message` / `message.delta`.
- All existing message-related tests should still pass — they assert end state, not source.

**Phase 3 — observer becomes sole materializer for usage and lifecycle data, watchdog rewires.**
- Remove `_materialize_run_usage_event` invocation from `append_event`.
- Observer emits a `__END_TURN__` event (or equivalent). Watchdog subscribes.
- `__END_TURN__` stdout detection in supervisor is removed.
- Remove `_materialize_run_lifecycle_event` from `append_event` — supervisor still publishes run lifecycle events, but they materialize through the single `materialize_event` path.

**Phase 4 — cleanup.**
- Remove origin tags on `_source_data` that no longer carry meaning.
- Re-evaluate `Session.codex_internal_id` — keep for display or drop.
- `_IGNORED_*` lists pruned.

## Tests

For each phase:

- Phase 1: `test_rollout_discovery_claude_deterministic`, `test_observer_bind_rollout_predates_file_creation`. (The psutil-probe and lsof-fallback tests landed with Phase 1 and were retired by Phase 2 along with `discover_codex` itself.)
- Phase 2: existing message tests pass unchanged after stdout parser removal. New: `test_supervisor_does_not_emit_message_events`.
- Phase 3: `test_watchdog_triggers_from_observer_end_turn`, `test_run_usage_no_double_count_via_unified_path` (Falcon's repro becomes a regression guard).
- Phase 4: housekeeping.

Each phase's PR also runs the sidecar smoke (claude harness session + codex harness session, both producing non-zero usage on a single session row each).

## Open questions

1. **Cost data — deferred** (Tijs, 2026-05-19). Token counts only in this refactor. `Usage.cost_usd` remains zero for codex (no upstream); claude `total_cost_usd` is dropped from the data plane since it only exists in stream-json `result`. A future helper can derive cost from token counts × a price table once we identify a maintainable price source.

2. **psutil dependency — added in Phase 1, removed in Phase 2.** Phase 1 added `psutil` for the codex fd probe; Phase 2 retires that probe in favor of the observer-side expectation registry (content-based match on `session_meta`), and drops `psutil` along with the `lsof` fallback.

3. **`process.stderr` event shape — confirmed** (Tijs, 2026-05-19). New event name: `process.stderr` with `{text: str}`. SSE consumers (mm-bridge, command-bridge) update accordingly. Mute previous `message.delta` from supervisor.

4. **Migration window — hard cut-over** (Tijs, 2026-05-19). No parallel-path consistency-check overlay. Each phase removes its half cleanly; tests cover the new shape.

5. **`Session.codex_internal_id` — decide in Phase 1.** Optional. Useful for diagnostics (cross-reference session → rollout filename); not required for correctness. Default: drop unless we find a concrete debug-flow that needs it.

6. **External-only sessions (no harness spawn).** Already work via the observer; behavior unchanged after refactor. The architecture homogenizes "external" vs "harness-spawned" at the data-plane level; the supervisor's only added value is process lifecycle + stderr capture.

## Out of scope

- Backfilling historical sessions to recompute usage / cost.
- Anything in claude's `--session-id` semantics (already works).
- Windows support (we're Linux + macOS).
- Bringing PRs #11 and #12 forward — they ship as-is in Phase 0.

## Estimated effort

- Phase 0: PR #11 carve-out + PR #12 ship — already underway.
- Phase 1: ~1 day. New module, supervisor wiring, tests.
- Phase 2: ~1 day. Removal + test verification.
- Phase 3: ~1 day. Watchdog rewire + tests.
- Phase 4: ~0.5 day.

Total: ~4 dev days plus review cycles. Reasonable scope for a single sub-session if Tijs greenlights, but I'd prefer phased PRs reviewed independently — fault isolation is much better.

## Authoring notes

Once this spec lands, file a TaskFlow task as parent of the four phase tasks. Heron and Aster should be looped in: this changes the visible event stream shape for command-bridge and the mm-bridge.
