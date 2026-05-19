# Phase 1 — Rollout Discovery + Pre-Binding

Parent spec: `specs/2026-05-19-unified-ingestion.md`
Parent task: `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.a51a3f0c-c91f-4e57-a9a5-3eb8e416109d`
Worktree: `worktrees/feat/unified-phase-1`
Branch: `feat/unified-phase-1`

## Scope

Add the machinery to map a freshly-spawned CLI subprocess to its rollout file and pre-register that mapping with the observer **before** the observer naturally discovers the file. This is the foundation that allows later phases to remove dual-path materialization without losing harness-session binding.

**No behavioral change in this phase.** Both ingestion paths remain live; the observer keeps materializing as today; the existing carve-outs continue to prevent doubles. This phase is observable only via new tests + the existence of pre-bindings before the file appears on disk.

## Deliverables

### New module: `src/agent_harness/rollout_discovery.py`

Public API:

```python
class RolloutDiscovery:
    def __init__(self, *, claude_projects_root: Path | None = None, codex_sessions_root: Path | None = None) -> None: ...

    async def discover_claude(self, *, session_id: str, cwd: Path) -> Path:
        """Compute the deterministic rollout path for a claude session.

        The path is the slugified-cwd directory plus <session_id>.jsonl.
        Returns the path even if the file does not yet exist; caller is
        responsible for binding the path before the file is created.
        """

    async def discover_codex(self, *, pid: int, cwd: Path, timeout_seconds: float = 5.0, poll_interval: float = 0.05) -> Path:
        """Locate the rollout file the codex subprocess just opened.

        Uses psutil.Process(pid).open_files() to enumerate fds. Filters
        for paths matching `~/.codex/sessions/.../rollout-*.jsonl`. Polls
        at poll_interval until the matching fd appears or timeout_seconds
        elapses (default 5s, codex typically opens within a few hundred
        ms). Raises RolloutDiscoveryError on timeout.

        If multiple rollout fds match, narrow by:
          1. exact cwd match against the rollout's session_meta.cwd
             (read the first line of the file; if not yet flushed, accept
             the youngest match as a tiebreaker)
          2. youngest file mtime

        On platforms where psutil.open_files() is unsupported, falls back
        to `lsof -p <pid> -F n` and parses output.
        """
```

Module-level helpers (private but tested):

- `_claude_slug(cwd: Path) -> str` — leading-slash strip + `/ → -`. Must match what claude-code itself produces.
- `_match_codex_rollout(open_files: Iterable[psutil.popenfile], sessions_root: Path) -> list[Path]` — filter logic.
- `_narrow_by_cwd(paths: list[Path], cwd: Path) -> Path` — tiebreaker.

### Custom exception

```python
class RolloutDiscoveryError(Exception):
    """Raised when the rollout file cannot be located for a subprocess."""
```

### Observer extension: `observer.bind_rollout(path, session_id)`

```python
class ExternalTranscriptObserver:
    def bind_rollout(self, path: Path, session_id: str) -> None:
        """Pre-register a path -> harness session id binding.

        Called by the orchestrator at spawn time. When tail_file later
        encounters this path (whether immediately or after a watchfiles
        notification), it uses session_id instead of deriving one from
        the filename pattern.

        Idempotent. Overwrites any prior binding for the same path.
        """
```

Internally stored in a dict `_path_to_session: dict[Path, str]`. Consulted in `transcript_identity_from_path` (or wherever the identity is currently derived from the filename) as the first lookup; if a binding exists, use it. Fallback path: existing filename-pattern logic.

### Orchestrator wiring

In `RunProcess.run()` (or just after `_process_factory(self.command)` returns):

- For claude (`session.backend == "claude-code"` and `session.origin == "harness"`):
  - `path = await discovery.discover_claude(session_id=self.session.id, cwd=self.command.cwd)`
  - `observer.bind_rollout(path, self.session.id)`
  - This happens BEFORE the file exists on disk; that's fine.

- For codex (`session.backend == "codex"` and `session.origin == "harness"`):
  - `path = await discovery.discover_codex(pid=process.pid, cwd=self.command.cwd)`
  - `observer.bind_rollout(path, self.session.id)`
  - Runs in a background task so the spawn returns quickly. If discovery fails, log + emit a `run.warning` event but don't fail the run; the observer will fall back to its existing filename-pattern registration (and the dupe-session symptom persists until later phases).

The discovery+bind step is gated behind a feature flag (env var `AGENT_HARNESS_ROLLOUT_PRE_BIND=1`) defaulting to **off** for this phase. That way Phase 1 lands with the machinery in place, observable via tests, but the production runtime keeps its current behavior until Phase 2 explicitly enables and then depends on it. Phase 2 removes the flag.

### Dependency

Add `psutil` to `pyproject.toml`'s runtime dependencies. Existing `lsof` use is via subprocess; do not add a hard dependency on the `lsof` binary, only use it when psutil's `open_files()` raises `NotImplementedError` (rare; happens on a few BSDs and exotic platforms).

## What NOT to do in Phase 1

- Do NOT remove `parse_codex_stream_line` or `parse_claude_stream_line`. Phase 2.
- Do NOT touch `storage.py:505-511` carve-out. Phase 2.
- Do NOT change `_materialize_run_usage_event` invocation. Phase 3.
- Do NOT touch the watchdog. Phase 3.
- Do NOT remove or rename `Session.codex_internal_id`. Phase 4.
- Do NOT emit `process.stderr` events. The stderr → `message.delta` path stays. Phase 2.

## Tests

In `tests/test_rollout_discovery.py` (new file):

- `test_claude_slug_strips_leading_slash`
- `test_claude_slug_replaces_separators`
- `test_discover_claude_returns_deterministic_path` — no file required.
- `test_discover_claude_handles_nested_cwd`
- `test_discover_codex_finds_open_fd` — using a fake `psutil.Process` returning open files; assert correct path returned.
- `test_discover_codex_polls_until_fd_appears` — fake process that "opens" the file on the 3rd poll; assert returned path correctness.
- `test_discover_codex_raises_on_timeout`
- `test_discover_codex_narrows_by_cwd_when_multiple_match` — two rollout fds, only one with matching cwd; assert the right one.
- `test_discover_codex_lsof_fallback` — psutil raises NotImplementedError, lsof returns plausible output; assert correct path.
- `test_rollout_discovery_error_is_raised_with_meaningful_message`

In `tests/test_observer.py` (extend):

- `test_bind_rollout_predates_file_creation` — call `bind_rollout(path, "ses_abc")` BEFORE the file exists; then create the file with content; tail_file should attribute events to `ses_abc` not to a synthesized id.
- `test_bind_rollout_overwrites_prior_binding`
- `test_unbound_rollout_falls_back_to_filename_pattern` — regression guard for existing behavior.

In `tests/test_orchestrator.py` (extend, gated on the feature flag):

- `test_run_process_pre_binds_claude_rollout_when_flag_enabled`
- `test_run_process_pre_binds_codex_rollout_when_flag_enabled` — using a fake process factory + fake discovery.
- `test_run_process_does_not_pre_bind_when_flag_disabled` — default behavior.
- `test_run_process_continues_when_discover_codex_fails` — discovery raises; run still succeeds, warning event emitted.

## Sidecar verification

After implementation:

1. Spin up sidecar harness on port 8879+ with temp DB. Set `AGENT_HARNESS_ROLLOUT_PRE_BIND=1`.
2. `POST /v1/sessions` with backend=claude-code; trigger a real claude run. Inspect events: assistant messages should attach to the harness session id; pre-binding should not break existing behavior.
3. Same for backend=codex. Verify the dupe-session symptom is still present (no observer-side dedupe yet) BUT the bound path is now annotated correctly in observer state. List sessions; both rows still present (Phase 1 hasn't removed dual-path yet).
4. Sanity: run with the env var unset; behavior must match current main exactly.

## Out of scope (for the whole refactor, restated)

- Backfilling historical sessions.
- Windows support.
- Cost data computation.

## Self-review checklist

- [ ] psutil added to pyproject; lsof fallback wired only when psutil raises NotImplementedError.
- [ ] All new tests pass; existing tests not regressed.
- [ ] Feature flag default-off; production runtime unchanged.
- [ ] Sidecar smoke executed for both env-var-on and env-var-off cases.
- [ ] PR title: `feat(unified-ingestion phase 1): rollout discovery + observer pre-binding`
- [ ] PR body references the parent spec, the Phase 1 spec, and the parent task.

## Out-of-band reminders

- Orphan-bash rule: no SSE curl without `-m`, no `httpx.stream()` in tests.
- Sidecar only; don't touch :8877.
- `/codex:review` before opening the PR.
- Final completion: post `Solace: DONE see ~<your-channel-slug>~ PR #<N>` in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`.
