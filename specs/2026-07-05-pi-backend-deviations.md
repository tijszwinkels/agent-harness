# pi backend — deviations, feature-parity calls, and e2e verification

Filed: 2026-07-05 — AgentFlow "implement" EXECUTABLE
`A5RqS3ppN1w5CvSX_QhxDWLNTpoV8AD3AKAPND9blJoE.9d947f66-f9ed-48bc-91e7-1ea8a3be0e8a`
Branch: `feat/pi-backend` (from `fd6b937`)
Spec: `AjHvnCoWHxVqhWzn-ZxQdTtjv5GmFjlsrKpLjwgD0s6o.c7ccfe5a…` (pi-backend-spec.md)
Spec-gate remarks (part of the spec): `A5RqS3ppN1w5CvSX_QhxDWLNTpoV8AD3AKAPND9blJoE.0a2d6854…`

This note records where the implementation deviates from the letter of the
spec because the operator's spec-gate remarks overrode it, the feature-parity
calls made under remark 2, and the end-to-end evidence (R5/R6).

## Deviation 1 — no configurable pi executable (spec R5 → remark 1)

The spec proposed a configurable pi executable via `AGENT_HARNESS_PI_BIN`
(default `"pi"`) so an operator could point the harness at a Node ≥ 22.19
wrapper. **Remark 1 overrode this:** the operator wants the *machine default
Node* bumped so the systemd-launched harness gets Node ≥ 22.19 directly, and
"the harness code should therefore not carry a pi-specific Node workaround."

Implemented accordingly:
- `PiCommandBuilder` hardcodes the bare `"pi"` on PATH, exactly like
  `ClaudeCodeCommandBuilder`/`CodexCommandBuilder` hardcode `claude`/`codex`.
- No env var, no `--pi-bin` flag, no wrapper indirection.
- The R5 "configurable bin" sub-requirement and its tests were dropped.

**Open operational item (P10):** pi needs Node ≥ 22.19 (it crashes on Node 20 —
see R5 below). Delivering that is an out-of-code step: bump the default Node in
the harness's **systemd** environment (systemd services default to Node 20 on
pillar even though interactive shells get Node 24 via fnm). This executor did
**not** modify the live machine's Node — that is a production change outside the
task's repo scope and outside an unattended executor's remit. **Until the
systemd Node is ≥ 22.19, every pi run will exit non-zero (`failed`).**

## Deviation 2 — resume is mandatory and simpler than claude's (remark 2)

Spec R6 (resume) was optional; **remark 2 made it mandatory.** pi's
`--session-id <id>` "creates it if missing" (verified `pi --help`, v0.80.3),
so — unlike claude, which switches `--session-id` → `--resume` on later runs —
the pi builder passes the *same* deterministic UUID (derived from the harness
session id via the shared `_harness_session_id_as_uuid` helper) on **every**
run. pi creates the session on turn 1 and loads it (retaining context) after.
`is_first_run` is intentionally ignored by `PiCommandBuilder`.

## Feature-parity reconsideration (remark 2) — what was built vs. deferred

Remark 2: "make pi feature-complete relative to claude-code and codex to the
extent possible … only leave out what is genuinely infeasible, and state
plainly why." Item by item:

- **Run lifecycle events — DONE (already work).** `run.started` /
  `run.completed` / `run.failed` are driven by the foreground process exit in
  `RunProcess.run`, with no backend-specific code. Verified end-to-end (R4/R5).
- **Multi-turn resume — DONE.** See Deviation 2; verified end-to-end (R6).
- **Transcript observation — DEFERRED, with reason.** pi *does* write per-session
  rollouts (`~/.pi/agent/sessions/<sanitized-cwd>/<ts>_<uuid>.jsonl`), so this is
  *feasible*, not impossible. But pi's rollout schema is its own (`session` /
  `model_change` / `thinking_level_change` / `custom` / … records, distinct from
  both claude and codex). Emitting `message`/`thinking`/`tool_use` events for pi
  would require a new pi rollout parser **plus** rollout discovery, watchfiles
  wiring, session binding, and external backfill — i.e. re-creating the entire
  claude/codex observer path (observer.py is ~75 KB). That is a separate,
  spec-sized effort; folding it into this change would ship a large,
  under-tested surface, violating P3 (concise) and P7 (verify). Deferred as a
  well-scoped follow-up: because pi's session directory mirrors claude's
  cwd-keyed `.jsonl` layout, a future `parse_pi_*` + a pi observe-root is the
  natural shape.
- **Usage / cost (`run.usage`, `SessionStats`) — DEFERRED (depends on the above).**
  Token/cost numbers come from parsing the rollout; with no pi transcript parser
  there is no `run.usage`, so `SessionStats` stays at defaults for pi. Not
  independently buildable without the parser.
- **External-origin pi session discovery/backfill — DEFERRED (depends on the above).**
  The observer watches no pi directory, so externally-started pi sessions aren't
  discovered. Tied to the same parser/observer work.
- **End-turn watchdog — correctly N/A (not a gap).** It only arms on
  `run.end_turn`, which comes from the rollout parser. Single-shot `pi -p` exits
  when the turn is done, so lifecycle completion is bound to process exit; the
  30-min idle watchdog still protects pi via the stdout heartbeat + stderr
  signal.

## Capability flags (remark 3) — reconsidered per remark 2

`BackendCapabilities.fork` / `interactive_pty` are **declarative metadata that
describe the backend's own capability**, not the harness's plumbing. Neither is
read by any harness code for *any* backend today (`grep -rn "fork"/"pty" src/`
finds only the field declaration in `models.py` and the literal assignments in
`backends.py`), yet claude and codex both set both flags `True`. Setting them
`False` for pi while pi genuinely supports the underlying behavior would describe
pi as *less capable than it is* on the exact same "not-wired-through-the-harness"
footing — an inconsistency, not honesty. So, verifying empirically against
`pi --help` (v0.80.3, the version this branch targets):

- **`fork=True`** — `pi --help` lists `--fork <path|id>` ("Fork specific session
  file or partial UUID into a new session"). pi supports fork; the flag now says so.
- **`interactive_pty=True`** — pi's default mode is interactive (the `pi "prompt"`
  and bare-`pi` examples), and `-p/--print` is documented as "Non-interactive
  mode". pi has an interactive TUI; the flag now says so.
- **`mcp=False`** — kept `False`. This is a *genuine* pi limitation, not a
  plumbing gap: `pi --help` (v0.80.3) exposes no MCP flag or subcommand at all,
  unlike claude/codex. This is the one flag remark 2's "only leave out what is
  genuinely infeasible, and state plainly why" clause applies to.
- **`session_id_choice=True`** — backed by real resume (Deviation 2).

The test `test_pi_backend_registered_with_honest_capabilities` was updated (TDD)
to assert these values.

## End-to-end verification (P7)

Driven through the **real** `agent-harness serve --execute-runs` HTTP API
(POST session + POST run + poll), spawning real `pi` subprocesses via the
production `AsyncioProcessFactory`. Scripts kept at `~/tmp/pi-e2e-harness.sh`
and `~/tmp/pi-e2e-resume.sh`. Model: groq `llama-3.3-70b-versatile`.

- **R5 success (Node 24):** run resolved `completed`; pi used its write tool
  (`-a`, from `bypass_permissions`) to create `harness-e2e.txt` = `PONG`. Exit 0.
- **R5 failure (Node 20):** same harness path with `pi` forced onto
  `/usr/bin/node` v20.19.4 → run resolved `failed` (returncode 1); the crash
  is exactly `TypeError: webidl.util.markAsUncloneable is not a function`
  (undici/CacheStorage under Node 20). Proves the Node version is the
  determinant and the machine-default bump is the fix.
- **R6 resume:** two runs on one session. Turn 1 stated codeword `BANANA7`
  ("do not create files yet"); turn 2 recalled it and wrote `recall.txt` =
  `BANANA7`. pi's rollout was written under the derived UUID
  `…_287be53a-db40-4b7a-953a-8012be7c9031` matching
  `ses_287be53adb404b7a953a8012be7c9031`, confirming create-or-load.

### e2e environment caveat (P5/P10 — dev-machine only)

The e2e set `PI_CODING_AGENT_DIR` to a fresh dir **only** to sidestep a broken
*globally-installed* pi extension on this dev box (`pi-messenger-bridge` throws
`ExtensionRunner.assertActive` at teardown, forcing exit 1 even on a successful
turn). That is an environment defect, not a harness concern — the harness passes
no such env. **Production note:** the harness treats any non-zero pi exit as
`failed`, so the machine that runs the harness must have healthy pi extensions
(or none) for pi runs to succeed, in addition to Node ≥ 22.19.
