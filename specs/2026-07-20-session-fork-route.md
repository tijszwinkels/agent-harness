# Session fork route + optional session model

Filed: 2026-07-20 — found during mm-bridge latency analysis (plan: `~/projects/mm-bridge/RELEASE.md`, "Found during perf analysis").
Worktree: `worktrees/feat/session-fork-route`
Branch: `feat/session-fork-route`

## Problem

Two defects block the public co-release of agent-harness + mm-bridge:

1. **Missing fork route.** mm-bridge maps a Mattermost *thread* to a *forked* session and calls
   `POST /v1/sessions/{session_id}/forks` (`agent_harness_client.fork_session`). That route was
   never implemented → every thread reply 404s with `Couldn't fork (Not Found)`.

2. **pi session-create 422.** `CreateSessionRequest.model` is required (`min_length=1`), but pi
   callers (mm-bridge's pi purpose) don't send a model — the pi CLI has its own configured default.
   Every pi `POST /v1/sessions` fails with 422.

## Consumer contract (mm-bridge — must need zero changes)

`agent_harness_client.fork_session(session_id, *, message, title=None)`:

- `POST /v1/sessions/{session_id}/forks`, body `{"message"?: str, "title"?: str}`.
- **201** → `{"session": Session, "run": Run | None}`. Bridge reads `resp["session"]["id"]` and
  tracks `resp["run"]` via `run.get("run_id") or run.get("id")` — a full `Run` (has `id`) satisfies it.
- **404 or 409** → `HarnessForkUnsupported` (bridge dead-threads with the detail string). So the
  "cannot fork" status is **409** (404 stays reserved for "no such parent session").
- Synchronous — the bridge gets the child id immediately, does **not** call `create_run` afterwards,
  so the fork route must start the run itself when `message` is present.
- The bridge posts: *"the full history of the parent up to its current state is included"* — the
  fork must resume the parent's whole conversation, not a slice.

The OpenAPI already documents this route (`/v1/sessions/{id}/forks`, `ForkSessionRequest`,
`ForkSessionResponse`, 201/404/409) — the code just has to match it.

## Fork mechanics per backend (verified against the real CLIs 2026-07-20)

A fork = a new **harness-owned** child session whose **first run** resumes the parent's conversation
but writes to the child's *own* deterministic transcript, leaving the parent untouched. The child id
(`ses_<hex>`) derives a canonical UUID via `_harness_session_id_as_uuid`, so the observer binds the
child's transcript exactly as for a normal harness session.

| Backend | First-run fork argv | Verified |
| --- | --- | --- |
| **claude-code** | `claude … --resume <parent_uuid> --fork-session --session-id <child_uuid> -- <text>` | child written to the pinned id, parent `.jsonl` byte-identical before/after, child inherited context (PINEAPPLE) |
| **pi** | `pi -p --model M --fork <parent_uuid> --session-id <child_uuid> [-a] <text>` | child session file created under the pinned id, parent file byte-identical, child replied PINEAPPLE |
| **codex** | — none — | rejected: see below |

**Why codex is rejected (409).** Verified today:
- `codex exec resume <id>` **appends to the parent's own rollout** (parent grew 85 693 → 90 348 B,
  same file ended up holding both turns) → reusing it for a fork would mutate/interleave the parent.
- `codex fork <id>` is a **TUI-only** interactive subcommand — `codex exec` has no `fork`, and
  `codex fork` has no `--json`/non-interactive mode, so it's incompatible with the harness's
  `codex exec --json` orchestration pipeline.
- A file-copy fork (duplicate the rollout under a fresh UUID) is possible but couples the harness to
  codex's on-disk rollout format + `session_meta.id` internals — fragile, version-specific, and out
  of scope for this route. Filed as a follow-up.

So `codex` forks return **409** with a clear detail, even though the declarative `fork=True`
capability (which describes the *CLI's* own ability) stays as-is.

## Rejected states (reject cleanly, per the brief)

- **Unsupported backend** (codex / unknown / a parent id that can't derive a fork UUID) → **409**.
- **Parent with a live run** (any run `queued` or `running`) → **409** `"cannot fork a session with
  an in-progress run"`. The fork point would be a mid-write transcript; rejecting is honest and the
  bridge just dead-threads (rare: a thread reply while the channel session is mid-turn).
- **External (observed-only) parent** → **allowed** for claude/pi: the parent transcript on disk is
  read to seed a *new harness-owned* child; the external parent is never mutated. Only codex externals
  are rejected (same reason as codex harness). This is what lets a channel mapped to an *external*
  claude session still fork on a thread reply.
- **Archived parent** → allowed (read-only on the parent's transcript).

## Design

### 1. `models.py`

- `Session.model: str | None = Field(default=None, min_length=1)` — optional; `None` ⇒ "use the
  backend CLI's own default" (the builder omits `--model`).
- `Session.forked_from: str | None = Field(default=None)` — parent session id this was forked from.
  Set once at fork creation; consumed by the command builder on the **first** run to emit the fork
  argv (so a failed-then-retried first run still forks, not creates fresh). Lineage/observability
  otherwise.
- `CreateSessionRequest.model: str | None = Field(default=None, min_length=1)`.
- New `ForkSessionRequest{message?: str(min_len 1), title?: str(min_len 1)}`.
- New `ForkSessionResponse{session: Session, run: Run | None}`.

### 2. `orchestrator.py`

- `ClaudeCodeCommandBuilder` / `PiCommandBuilder`: when `is_first_run and session.forked_from`, emit
  the fork argv above (parent uuid from `_harness_session_id_as_uuid(session.forked_from)`). Otherwise
  unchanged. `--model` becomes conditional on `session.model is not None` in all three builders.
- `def session_supports_fork(backend: str) -> bool` → `{"claude-code", "pi"}`.
- `def validate_fork_source(parent: Session) -> None` — raises `CommandBuildError` when the backend
  can't fork or the parent id can't derive a fork UUID (mirrors `validate_session_resume_target`).

### 3. `repository.py` / `storage.py`

- `create_forked_session(parent, *, title) -> Session` (both repos): child inherits
  `backend/model/project/bypass_permissions`, `origin="harness"`, `forked_from=parent.id`,
  `title = title if title is not None else parent.title`. No schema change (Session is a JSON payload).

### 4. `api.py`

- Extract the create_run launch body (preflight → `repo.create_run` → build command → submit to
  `run_manager` → `on_start`/materialization) into an inner `_create_and_launch_run(session_id,
  request) -> CreateRunResponse`. `create_run` becomes a thin caller — **no behavior change**.
- `POST /v1/sessions/{session_id}/forks`:
  1. `repo.get_session` (404 if missing).
  2. `validate_fork_source(parent)` (409 on failure).
  3. reject if any run for the parent is `queued`/`running` (409).
  4. `repo.create_forked_session`, publish `session.updated` for the child (mirrors create_session).
  5. if `message` is not None → `_create_and_launch_run(child.id, CreateRunRequest(message=…))`,
     then `run = repo.get_run(child.id, resp.run_id)`; else `run = None`.
  6. return `ForkSessionResponse(session=child, run=run)`, **201**.

### 5. `specs/openapi.yaml`

- `Session`: drop `model` from `required`; `model: [string,"null"]`; add `forked_from: [string,"null"]`.
- `CreateSessionRequest`: drop `model` from `required`; `model: [string,"null"]`.
- Fork route + `ForkSessionRequest`/`ForkSessionResponse` already present — drift-guard test added.

## Tests (TDD — failing first)

- `test_models.py`: `forked_from` drift guard; model-optional accepted.
- `test_orchestrator.py`: claude fork argv (first run, forked_from) resumes parent + `--fork-session`
  + pins child `--session-id`; pi fork argv; `--model` omitted when `session.model is None`; a fork's
  *second* run resumes the child (not the parent); `session_supports_fork` / `validate_fork_source`.
- `test_api.py`: create session without model → 201; fork happy path (claude) 201 returns child +
  running Run; fork with no message → child, `run=None`; fork unknown parent → 404; fork codex → 409;
  fork while parent has a live run → 409; fork of an external claude parent → 201 harness child.
- `test_openapi.py`: fork request/response + optional-model drift guards.

## Verification

- Full suite green (baseline 320 + new).
- Manual E2E against a **local** sidecar harness (spare port, in-memory repo — never `:8877`):
  create claude session → fork → run in the fork → child transcript distinct from parent, context
  carried.

## Out of scope / follow-ups

- Codex file-copy fork (would make codex forkable non-interactively).
- Deploy: the running `:8877` daemon needs a restart to serve the new route — a deploy step, done
  separately, **not** here.

## Workflow

Draft-PR-first: open the draft with this spec commit, push after every commit, `gh pr ready` + DONE
post as one paired action at the end. Sidecar only; no `:8877`. No SSE curl without `-m`.
