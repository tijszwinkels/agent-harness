# Issue: "codex forgets its personality" (harness-origin codex loses context across turns)

Filed: 2026-05-30
Branch: `feat/codex-resume`
Spec: `specs/2026-05-21-codex-resume.md`

## Symptom

A harness-origin codex session loses all prior conversation context between
turns. Most visibly, the persona/kickoff established on turn 1 (mm-bridge
sends the persona instruction as the *first user message only*) is gone by
turn 2 — codex behaves as a blank session.

Observed in vivo by Aster on 2026-05-21 (SLAM 4 session
`ses_c3577e146acc4920ab9f590264328455`): two separate codex rollouts were
generated for two runs of the same logical harness session. Turn 2's rollout
opened on the operator's bare `?` nudge with no prior conversation in scope, so codex
correctly replied "What would you like me to work on?" — it genuinely had no
context.

## Root cause

`CodexCommandBuilder.build` in `src/agent_harness/orchestrator.py` gated the
resume command on `session.origin == "external"`:

```python
# main (broken):
del run, is_first_run
...
if session.origin == "external":
    return ProcessCommand(argv=("codex","exec","resume","--json","--model",
        session.model, *bypass, _external_resume_id(session, prefix="codex_"), text), ...)
return ProcessCommand(argv=("codex","exec","--json","--model",
    session.model, *bypass, text), ...)   # ← harness-origin: NO resume id
```

(main `orchestrator.py:74-109`.) Harness-origin codex runs therefore ALWAYS
emitted a fresh `codex exec --json --model <m> <text>` with no resume token.
Codex writes a brand-new rollout per turn, so the model never sees prior
turns. `ClaudeCodeCommandBuilder` already had the correct pattern
(`--session-id` first run, `--resume <uuid>` follow-up); the codex equivalent
was simply never written.

### Why the persona lives only in turn 1

mm-bridge sends the persona/kickoff instruction as the *first user message*
of a spawned session and never replays it. With codex generating a fresh
rollout each turn, the persona prompt (in turn 1's rollout) is invisible to
every subsequent turn. The bug is latent for single-turn runs and surfaces
the moment a session takes a second turn.

## The fix (this branch, +9 commits)

1. `models.py:176` — add `Session.codex_resume_id: str | None = None`. The
   live resume key (distinct from the retired Phase-4 `codex_internal_id`
   diagnostic field). `None` for non-codex sessions and for codex sessions
   whose first run hasn't bound a rollout yet.
2. `orchestrator.py` — rewrite `CodexCommandBuilder.build` to gate on the
   field, not origin: `codex_resume_id is not None` → `codex exec resume
   <uuid>`, else fresh `codex exec`. The harness and external paths collapse
   into one origin-independent gate.
3. `observer.py` — persist `codex_resume_id` onto the session when binding
   completes (extracts the rollout UUID from the bound filename, harness
   origin), and a startup backfill for external sessions (UUID encoded in the
   `codex_<uuid>` id). Idempotent: re-tailing an already-bound rollout does
   not re-emit `session.updated`; the dedupe entry is evicted on unbind.
4. The materializer preserves `codex_resume_id` on a None-incoming
   harness→harness update so a later session payload can't clobber it.

End-to-end flow: orchestrator registers `observer.expect_codex_rollout(cwd,
session_id)` at spawn → observer matches the new rollout by `session_meta.cwd`
+ a ±30s timestamp window → `bind_rollout` → `session.updated` patches
`codex_resume_id` → next `CodexCommandBuilder.build` emits `codex exec resume
<uuid>`, restoring full context (and thus the turn-1 persona).

## How verified

- **Unit (deterministic):** `tests/test_orchestrator.py`
  - `test_codex_command_uses_exec_resume_when_resume_id_present` — resume id
    set → argv contains `resume` + the exact UUID.
  - `test_codex_command_uses_fresh_exec_when_resume_id_absent` — id None →
    fresh `codex exec`, no `resume`.
  - `test_codex_command_uses_resume_for_external_origin_via_unified_field`.
  - `test_codex_command_resumes_harness_origin_independent_of_origin_field`
    (added in this verification) — pins `session.origin == "harness"`
    explicitly so the gate can never silently re-couple to origin. Confirmed
    RED against main's builder (`'resume' not in argv`) before going GREEN
    against the fix.
- **Integration (full chain, deterministic):** `tests/test_observer.py::
  test_codex_multi_turn_resumes_after_observer_binding_pineapple` — drives
  the real `ExternalTranscriptObserver.tail_file` → `session.updated` →
  `CodexCommandBuilder` across two turns: turn 1 = fresh exec, turn 2 =
  `codex exec resume <uuid>`. This is the canonical PINEAPPLE proof.
- **Sidecar smoke:** `scripts/smoke_codex_resume.py` boots the daemon and
  proves the external-session backfill populates `codex_resume_id` over the
  wire after a restart. (Note: this script binds port 8879 and relocates via
  `HOME`, not `CODEX_HOME` — run it only outside the live-harness host.)
- **Full suite:** `uv run pytest -q` green (261 tests, including the new one);
  no regressions.
- **Real-codex multi-turn smoke (manual, for operator):** attempted in an
  isolated `CODEX_HOME` (copied `auth.json` + `config.toml`; verified codex
  writes rollouts under `$CODEX_HOME/sessions` and does NOT touch the live
  `~/.codex/sessions`). Real codex authenticated and turn 1 completed with a
  correctly-shaped rollout (session_meta `cwd` matched the project path).
  However the harness's own observer did not bind the rollout in this
  throwaway setup — `watchfiles`/inotify did not deliver the create event for
  a rollout written into a freshly-created nested date subdirectory under the
  isolated `--observe-root`, so `codex_resume_id` stayed None and turn 2 could
  not resume. This is an inotify-on-new-subdirectory delivery limitation of
  the ad-hoc isolation, NOT a defect in the resume logic — the binding chain
  it would exercise is already proven deterministically by the PINEAPPLE
  observer integration test above. Recommended manual smoke for the operator:
  run the real two-turn flow against a harness whose `--observe-root` already
  contains today's `YYYY/MM/DD` subtree (or rely on the live harness's
  default-root watch, which has long-lived descriptors), POST turn 1
  ("Your codeword is PINEAPPLE"), confirm `GET /v1/sessions/<id>` shows
  `codex_resume_id` populated, then POST turn 2 and confirm the reply recalls
  PINEAPPLE.

## Known follow-up (NOT fixed here)

Secondary persona-drop path in mm-bridge: a Purpose backend/model change
triggers `_restart_session_with_config` (`mm-bridge/src/mm_bridge/bridge.py`
~1173 → ~1294). It tears down the current harness session and creates a brand
new one, sending only `INVITE_PLACEHOLDER` as the kickoff run — it does NOT
replay the original persona/kickoff prompt. So switching backend/model mid
conversation drops persona regardless of codex-resume (new session id, no
prior rollout to resume). This lives in mm-bridge, not the harness; it should
be addressed there (replay the persona on restart, or carry it forward).
Tracked here only as a pointer — out of scope for `feat/codex-resume`.
