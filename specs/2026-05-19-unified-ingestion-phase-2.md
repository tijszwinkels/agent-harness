# Phase 2 — Observer becomes sole writer for messages; expectation registry

Parent spec: `specs/2026-05-19-unified-ingestion.md`
Parent task: `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.a51a3f0c-c91f-4e57-a9a5-3eb8e416109d`
Worktree: `worktrees/feat/unified-phase-2`
Branch: `feat/unified-phase-2` (off main, post PR #13)

## Scope

Two structural changes that ship together because they reinforce each other:

1. **Codex rollout discovery moves from fd-polling to a content-based expectation registry.** The orchestrator registers a hint ("I'm spawning codex from cwd X around timestamp T") and the observer matches it against `session_meta` peeked from the rollout file's first line. `RolloutDiscovery.discover_codex` and the `psutil` dependency retire.
2. **The supervisor stops emitting `message` and `message.delta` events.** The observer is the sole writer for both. The `message`-on-external carve-out in `storage.append_event` retires. stderr forwarding becomes a new `process.stderr` event.

Both changes work together to eliminate the dual-path duplication for the message data type. After Phase 2 lands, there is exactly one place in the system that writes `message` events: the observer's `_materialize_or_buffer` path.

## Why option 1 (content-match) won

Phase 1 introduced synchronous codex pre-bind via psutil fd-polling. Sentry's review flagged a residual race: even though spawn now blocks until `discover_codex` succeeds, watchfiles inotify can fire on the rollout file BEFORE `bind_rollout` lands, because watchfiles and discover poll independently.

Option 1 dissolves the race by inverting the resolver: instead of orchestrator → observer ("here's the path; bind it"), the observer reads `session_meta` from the file content and looks up the orchestrator's hint. Watchfiles can fire whenever; resolution is content-based, not timing-based.

Confirmed with the maintainer 2026-05-19: "yeah, do option 1."

## Code changes

### `src/agent_harness/observer.py`

Add an expectation registry to `ExternalTranscriptObserver`:

```python
@dataclass(frozen=True, slots=True)
class CodexRolloutExpectation:
    cwd: Path
    session_id: str
    registered_at: datetime
    expires_at: datetime
```

```python
class ExternalTranscriptObserver:
    def __init__(self, ..., expectation_ttl_seconds: float = 10.0) -> None:
        ...
        self._codex_expectations: list[CodexRolloutExpectation] = []
        self._expectation_ttl = timedelta(seconds=expectation_ttl_seconds)

    def expect_codex_rollout(self, *, cwd: Path, session_id: str) -> None:
        """Hint that a codex rollout matching ``cwd`` is about to appear.

        Called by the orchestrator at codex spawn time. The observer
        matches incoming codex rollouts against active expectations by
        cwd + session_meta.timestamp window. Expectations expire after
        ``expectation_ttl_seconds`` (default 10s).
        """
        now = self._clock()
        self._codex_expectations.append(CodexRolloutExpectation(
            cwd=cwd,
            session_id=session_id,
            registered_at=now,
            expires_at=now + self._expectation_ttl,
        ))
```

Identity resolution for unbound codex paths reads `session_meta`:

```python
def _resolve_codex_identity(self, path: Path) -> TranscriptIdentity | None:
    bound = self._path_to_session.get(path)
    if bound is not None:
        return TranscriptIdentity(backend="codex", path=path, session_id=bound, is_rebound=True)

    session_meta = self._peek_session_meta(path)  # returns None on partial flush
    if session_meta is None:
        return None  # caller retries on next tail_file tick

    self._purge_expired_expectations()
    expectation = self._find_matching_expectation(
        cwd=session_meta.cwd,
        timestamp=session_meta.timestamp,
    )
    if expectation is None:
        return None  # fall through to filename-pattern in caller

    # Consume the expectation and record the binding so subsequent
    # tail_file calls on this path don't re-peek.
    self._consume_expectation(expectation)
    self._path_to_session[path] = expectation.session_id
    return TranscriptIdentity(
        backend="codex",
        path=path,
        session_id=expectation.session_id,
        is_rebound=True,
    )
```

`_peek_session_meta`: open the file, read the first line; if no trailing newline, return None (partial flush — retry next tick); else parse as JSON, check `type == "session_meta"`, return `(cwd, timestamp)` as a small dataclass.

`_find_matching_expectation(cwd, timestamp)`: filter unexpired expectations where `cwd` equals (or is a path-equivalent to) the rollout's cwd AND `|timestamp - expectation.registered_at| <= matching_window` (default 30s — generous for codex startup delays). Among matches, return the closest-timestamp one (tiebreaker for the rare double-spawn case).

The existing filename-pattern path stays as the resolver of last resort for genuinely external codex sessions (no expectation registered).

### `src/agent_harness/rollout_discovery.py`

Delete `RolloutDiscovery.discover_codex`, the polling loop, and the lsof fallback. Keep `RolloutDiscovery.discover_claude` (deterministic, no race, used by claude pre-bind which is unchanged).

The module shrinks to maybe ~80 LOC.

### `src/agent_harness/orchestrator.py`

`_pre_bind_codex` reduces to:

```python
def _pre_bind_codex(self) -> None:
    self._observer.expect_codex_rollout(
        cwd=self.command.cwd,
        session_id=self.session.id,
    )
```

Not async anymore. Called BEFORE `await self._process_factory(self.command)` so the hint is registered before codex can possibly flush its first byte.

`_stream_lines` (the stdout/stderr pump) is restructured. Today it emits `message.delta` for every line and runs the parser for end-turn detection. After Phase 2:

- **stdout**: no `message.delta` publish. Only `_detect_end_turn_in_line` runs to emit the `__END_TURN__` signal (consumed by the watchdog).
- **stderr**: emit a `process.stderr` event with `{text: <line>}` payload. No `message.delta` wrapping.

`parse_codex_stream_line` and `parse_claude_stream_line` are replaced with a single backend-agnostic `_detect_end_turn_in_line(text: str, backend: str) -> bool` helper. It returns True when:
- claude: `{"type":"result","stop_reason":"end_turn"}`
- codex: `{"type":"event_msg","payload":{"type":"task_complete"}}` or `{"type":"turn.completed"}` (verify the exact codex stdout shape against a live sample before locking)

`default_stdout_parsers()` is deleted. `_stdout_parsers` field on `RunProcess` is deleted.

### `src/agent_harness/storage.py`

Remove lines 505-511:

```python
if (
    published.event == "message"
    and published.session_id
    and published.data.get("origin") != "external"
    and self._find_session_locked(published.session_id) is not None
):
    self._materialize_message_event(published, dedupe_by_content=False)
```

`append_event` no longer materializes messages. The observer's `materialize_event` is the sole path.

(Note: the `run.usage` materialization at storage.py:503 stays — Phase 3 removes that.)

### `src/agent_harness/cli.py` and `api.py`

Remove the `AGENT_HARNESS_ROLLOUT_PRE_BIND` env-var check. Pre-binding is unconditional.

### `pyproject.toml` + `uv.lock`

Remove `psutil` from dependencies. Run `uv lock` to regenerate.

### `specs/2026-05-19-unified-ingestion.md` (parent spec)

Patch two staleness points Sentry flagged:
1. Parent spec line ~95 said codex bind "runs in a background task so the spawn returns quickly". Phase 1 went synchronous. Phase 2 makes it a one-line registration (also synchronous, no await). Update the description.
2. Parent spec mentioned `psutil.Process(pid).open_files()` as the codex discovery mechanism. After Phase 2 this is gone. Update to reference the expectation registry.

## What disappears (concretely)

- `RolloutDiscovery.discover_codex`
- `psutil` dependency, `lsof` fallback path
- `RunProcess._stdout_parsers` field, `default_stdout_parsers()` function
- `parse_codex_stream_line`, `parse_claude_stream_line` (replaced by minimal end-turn detector)
- `_publish("message.delta", ...)` from supervisor's stdout pump
- `storage.append_event`'s message materialization branch (lines 505-511)
- `AGENT_HARNESS_ROLLOUT_PRE_BIND` env var
- Tests asserting supervisor emits `message.delta` events from stdout

## What stays / what's new

- `observer.bind_rollout` / `unbind_rollout` (Phase 1)
- `observer.expect_codex_rollout` (NEW in Phase 2)
- `observer._path_to_session` map (Phase 1)
- claude pre-bind path (deterministic, unchanged from Phase 1)
- `process.stderr` event (NEW): `{text: str, run_id: str}` — stderr lines from the supervisor
- Filename-pattern fallback for genuinely-external codex sessions (still useful when no expectation is registered)

## Tests

In `tests/test_observer.py`:

- `test_expect_codex_rollout_matches_by_cwd_and_timestamp` — register expectation; create rollout fixture with matching `session_meta`; assert resolved identity uses the expected session_id.
- `test_expect_codex_rollout_falls_through_when_cwd_differs` — expectation registered for cwd A; rollout has cwd B; identity falls through to filename pattern.
- `test_expect_codex_rollout_falls_through_when_timestamp_outside_window` — expectation timestamp far from session_meta timestamp; falls through.
- `test_expect_codex_rollout_closest_timestamp_wins` — two expectations with same cwd, different timestamps; assert the closer one matches.
- `test_expectation_expires_after_ttl` — expire and verify it doesn't match.
- `test_session_meta_peek_returns_none_for_partial_flush` — rollout file with no trailing newline on line 1; peek returns None.
- `test_session_meta_peek_skips_non_session_meta_first_line` — first line is something else; peek returns None.

In `tests/test_orchestrator.py`:

- `test_run_process_emits_no_message_delta_for_stdout` — feed fake stdout lines; assert no `message.delta` events on the bus.
- `test_run_process_emits_process_stderr_for_stderr_lines` — feed fake stderr lines; assert `process.stderr` events fire with the right text.
- `test_run_process_still_detects_end_turn_for_watchdog` — claude `result` line / codex `task_complete` line triggers the watchdog cleanup.
- `test_pre_bind_codex_registers_expectation` — assert `observer.expect_codex_rollout` is called with the right cwd + session_id.
- Delete tests asserting `message.delta` from stdout (they were Phase 1 behavior).

In `tests/test_storage.py`:

- `test_append_event_no_longer_materializes_message` — append a `message` event with origin=harness; assert messages table is NOT touched by `append_event` alone.
- Existing tests that exercise `materialize_event` for messages should still pass — that's the surviving path.

In `tests/test_rollout_discovery.py`:

- Delete `discover_codex` test class entirely.
- Keep `discover_claude` tests.

## Sidecar verification

After implementation:

1. Spin up sidecar harness on port 8879+ with temp DB. (No env var to set anymore; pre-bind is on by default.)
2. `POST /v1/sessions` with backend=codex; trigger a real codex run.
3. Verify in the events stream:
   - Exactly ONE session row for this run.
   - Messages flow as `message` events from the observer (with `session_id` = harness id, not phantom `codex_<uuid>`).
   - No `message.delta` events.
   - `process.stderr` events visible if codex writes anything to stderr.
   - `__END_TURN__` signal still fires (watchdog clean-up succeeds).
4. Same for backend=claude-code.
5. Drop a fixture codex rollout in `~/.codex/sessions/...` from a cwd the harness did NOT spawn for; observer registers it via filename pattern (no expectation match); resolution unchanged.

## Migration / consumer impact

- **mm-bridge**: today discards `message.delta` explicitly (`bridge.py:2243`). After Phase 2, those events don't fire. No code change required, but the `HARNESS_ACTIVITY_EVENTS` set in `bridge.py:39` can drop `message.delta` for clarity (small follow-up; not blocking Phase 2 merge).
- **command-bridge**: today has a `message.delta` handler at `store.ts:253`, but expects a structured shape (`message_id`, `delta.type`, etc.) that the orchestrator never emitted. Effectively dead code. Phase 2 makes it formally dead — safe to remove in a separate command-bridge PR.
- **agent-harness's own watchdog**: `HARNESS_ACTIVITY_EVENTS` in mm-bridge tracks `message.delta` for idle-timeout — but the harness's own watchdog is driven by `_ACTIVITY_EVENTS` in `orchestrator.py:28` which also lists `message.delta`. After Phase 2, the watchdog falls back on `message` and `tool_use` events (both still fire). Verify the 30-min idle threshold isn't tripped by normal turn latency. (Rollout flushes are per-turn, well under 30min.)

## What is NOT in Phase 2

- `_materialize_run_usage_event` invocation in `append_event` (Phase 3).
- Watchdog rewires to observer-emitted end-turn (Phase 3).
- usage.py cherry-pick from PR #11 (Phase 3 — needs to wire to observer-driven materialization).
- `Session.codex_internal_id` retire/keep decision (Phase 4).
- `process.stderr` consumer updates in mm-bridge / command-bridge (separate tasks; their teams handle).

## Self-review checklist

- [ ] All tests pass; no regressions.
- [ ] No `message.delta` events emitted from the supervisor under any code path.
- [ ] `psutil` removed from `pyproject.toml`; `uv lock` re-run.
- [ ] `discover_codex` and its tests deleted; no orphan references in observer/orchestrator/api/cli.
- [ ] `storage.py` carve-out (lines 505-511) removed; tests confirm messages don't materialize via `append_event`.
- [ ] Expectation registry purges expired entries; no memory growth across many runs.
- [ ] Closest-timestamp tiebreaker deterministic on equidistant case (e.g., earliest-registered wins).
- [ ] Parent spec patched for Phase 1 deviations + Phase 2 architecture.
- [ ] Sidecar smoke executed end-to-end for both backends.
- [ ] PR description references the parent spec, Phase 2 spec, parent task, AND Sentry's worth-noting #2 as the motivating issue.

## Out-of-band reminders

- Orphan-bash rule: no SSE curl without `-m`, no `httpx.stream()` in tests.
- Sidecar only; don't touch `:8877`.
- `/codex:review` (or claude self-review pattern) before opening the PR.
- Final completion: post `Solace: DONE see ~<your-channel-slug>~ PR #<N>` in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`. Don't skip this — same lesson from Phase 1.

## Estimated effort

~2 days of dev time. Code surface is smaller than Phase 1 (mostly deletes), but the expectation-registry + session_meta peek logic + tests are the substantive work.
