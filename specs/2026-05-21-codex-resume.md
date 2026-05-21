# Codex harness-origin session resume

Filed: 2026-05-21 — Aster's task `ApWJVWXvVKIIMnH6CP6u8HUyU2gLvyYGnwRlgrWAUwcP.c1add695-fb08-4bbf-a59d-99e7cd5c7ae1`
Worktree: `worktrees/feat/codex-resume`
Branch: `feat/codex-resume`

## Problem

`CodexCommandBuilder.build` (orchestrator.py:85-109) only emits a resume command when `session.origin == "external"`:

```python
if session.origin == "external":
    argv = ("codex", "exec", "resume", "--json", ..., _external_resume_id(...), text)
else:
    argv = ("codex", "exec", "--json", "--model", model, *bypass, text)  # ← harness-origin: fresh spawn every turn
```

Every harness-origin codex run launches a brand-new `codex exec --json` with no resume id → codex generates a fresh rollout per turn → the model sees zero prior context.

`ClaudeCodeCommandBuilder` (lines 137-177) has the correct pattern: `--session-id` on first run, `--resume <id>` on follow-up. The codex equivalent was never written.

**Verified in vivo by Aster 2026-05-21** (SLAM 4 session `ses_c3577e146acc4920ab9f590264328455`): two separate codex rollouts generated for two runs of the same logical session. Rollout 2's first user message was Tijs's bare `?` nudge with no prior conversation in scope; codex correctly responded "What would you like me to work on?" because it genuinely had no context. Latent since codex was first added.

## Architectural insight

The Phase 1 binding machinery already discovers the codex rollout UUID at spawn time:
- Orchestrator calls `observer.expect_codex_rollout(cwd, session_id)` before spawning codex.
- Observer matches the new rollout file by `session_meta.cwd` + timestamp window, calls `bind_rollout(path, session_id)`.
- Phase 4 retired `Session.codex_internal_id` (the old PR #12 field) — but the UUID is still extractable from the bound rollout path filename.

What's missing: persisting the UUID on the Session so it survives between runs, and using it on follow-up `CodexCommandBuilder.build` invocations.

## Field design

Add a new field to `Session`:

```python
class Session(HarnessModel):
    ...
    codex_resume_id: str | None = Field(default=None)  # NEW
```

Semantics:
- `None` for non-codex sessions and for codex sessions whose first run hasn't completed binding yet.
- Set ONCE by the observer when binding resolves (idempotent — subsequent observations of the same path leave it alone).
- Used by `CodexCommandBuilder.build` to decide between fresh-spawn and resume.
- This field's purpose is **resume**, not diagnostics. Don't conflate with Phase 4's retired `codex_internal_id`.

Why a separate field from the bound path:
- The orchestrator needs the rollout UUID, not the full file path, for `codex exec resume <uuid>`.
- Decoupling lets the observer move rollout files without breaking resume.

## Implementation

### 1. `models.py`

Add `codex_resume_id: str | None = Field(default=None)` to `Session`. Same field on response/event schemas.

### 2. `observer.py`

Extract the UUID from the bound rollout filename when binding completes. The filename pattern is `rollout-<ISO-timestamp>-<UUID>.jsonl` (use the existing `_CODEX_ROLLOUT_RE` regex at observer.py:30-ish for the UUID).

After `bind_rollout(path, session_id)` succeeds for a codex session, emit a `session.updated` event with the Session payload carrying `codex_resume_id=<extracted_uuid>`. The bus's single materialization point (Phase 3) applies it to the DB row.

Idempotency: if the Session already has a non-None `codex_resume_id`, don't re-emit. (Edge case: observer restart re-scans an already-bound rollout. Should not re-set a field that's already set.)

### 3. `orchestrator.py`

`CodexCommandBuilder.build` collapses to:

```python
def build(self, *, session: Session, is_first_run: bool, ...) -> ProcessCommand:
    bypass = ("--dangerously-bypass-approvals-and-sandbox",) if session.bypass_permissions else ()

    if session.codex_resume_id is not None:
        # Either harness-origin with completed first-run binding, OR external-origin.
        argv = (
            "codex", "exec", "resume", "--json",
            "--model", session.model, *bypass,
            session.codex_resume_id, text,
        )
    else:
        # First harness-origin run; observer will populate codex_resume_id after this turn.
        argv = (
            "codex", "exec", "--json",
            "--model", session.model, *bypass, text,
        )

    return ProcessCommand(argv=argv, cwd=session.project.path)
```

Delete `_external_resume_id` and the `origin == "external"` branch — the unified field replaces both code paths.

### 4. External-origin migration

External-origin codex sessions today have their UUID encoded in `session.id` (form `codex_<uuid>`). The new field needs to be populated for those too. Two options:

- **Option A (recommended)**: backfill on observer restart — when seeding `_last_event_at_from_repository` (existing observer logic for external sessions), also populate `codex_resume_id` from the session ID prefix for external codex sessions where the field is None. One-shot migration, no DB schema change.
- **Option B**: lazy migration in `CodexCommandBuilder` — if `codex_resume_id is None` AND `session.id.startswith("codex_")`, derive on the fly. Less clean; fallback persists indefinitely.

Lean A. Backfill happens once per observer process startup for existing rows.

### 5. Repository / storage

The `session.updated` materialization path already handles arbitrary Session model shapes (uses `Session.model_validate`). New field rides for free; verify nothing in the materializer is hardcoded against the old shape.

### 6. OpenAPI

Document `codex_resume_id` on Session schema. Drift-guard test confirms.

## Tests

In `tests/test_observer.py`:
- `test_observer_populates_codex_resume_id_on_binding` — drop a codex rollout matching an expectation; assert the resulting `session.updated` event carries `codex_resume_id=<filename_uuid>`.
- `test_observer_does_not_re_emit_codex_resume_id_when_already_set` — second tail of same path; no duplicate event.
- `test_observer_seeds_codex_resume_id_for_external_sessions_on_restart` — Option A backfill behavior.

In `tests/test_orchestrator.py`:
- `test_codex_command_uses_exec_resume_when_resume_id_present` — harness-origin session with codex_resume_id set; assert command uses `codex exec resume <id>`.
- `test_codex_command_uses_fresh_exec_when_resume_id_absent` — first run path.
- `test_codex_command_uses_resume_for_external_origin_via_unified_field` — external session with codex_resume_id (after Option A backfill).

In `tests/test_models.py`:
- `test_session_carries_codex_resume_id` — drift guard.

In `tests/test_openapi.py`:
- Drift guard for the new field.

## Sidecar verification

After implementation:

1. Spin up sidecar harness :8879+.
2. Harness-spawn a codex session, send prompt 1: "Remember the word PINEAPPLE."
3. Wait for run.completed. `GET /v1/sessions/<id>` should show `codex_resume_id` populated.
4. POST another run with prompt 2: "What word did I ask you to remember?"
5. Verify the resulting rollout file is the SAME file as run 1 (or a continuation), AND codex's response includes "PINEAPPLE".
6. Verify codex `exec resume <id>` was the actual command used (inspect the supervisor log or stub the process factory for assertions).

This is the canonical multi-turn-context-survives test. Should fail today, pass after this PR.

## What this is NOT

- A fix for the `image_generation_*` parser gap (separate task `ec7cb027`).
- A change to claude-code's resume flow (already correct).
- A migration to a single CommandBuilder abstraction (out of scope; CodexCommandBuilder stays a separate class).
- Codex rollout discovery changes — Phase 2's expectation registry stays as-is.

## Workflow — DRAFT-PR-FIRST + paired-final-action

The pattern that broke the wrap-up stall in Phase 4 was: open a draft PR with just the spec commit BEFORE substantive work, push after every subsequent commit, then `gh pr ready` + DONE post AT THE END.

Refinement from PR #17 retrospective: `gh pr ready` and the DONE post are ONE INSEPARABLE paired action — both happen in the same shell sequence immediately after self-review completes, before doing anything else. Don't pause between them.

## Self-review checklist

- [ ] All tests pass; no regressions.
- [ ] `_external_resume_id` removed; unified field replaces both code paths.
- [ ] Idempotent observer emission: re-tail of bound rollout doesn't re-emit.
- [ ] Option A backfill verified for an existing external-origin codex session.
- [ ] OpenAPI updated; drift-guard test passes.
- [ ] Sidecar smoke verifies PINEAPPLE multi-turn scenario.
- [ ] PR title: `feat: codex harness-origin session resume`.
- [ ] PR body references this spec + Aster's task `c1add695` + the SLAM 4 in-vivo evidence.

## Out-of-band reminders

- No SSE curl without `-m`; no `httpx.stream()` in tests.
- Sidecar only; don't touch `:8877`.
- `/codex:review` before flipping draft to ready.
- Final paired action: `gh pr ready N` then DONE post in Echo's channel `mjg461xsgbrn7gks7ftoqmf8ca`, both in the same shell sequence.

## Estimated effort

~1-1.5 days. Smaller than Phase 1, similar to context_used. The core change is small (one new field + builder collapse); migration backfill + tests are the bulk of the work.
