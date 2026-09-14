# External pi sessions — discovery + headless continuation

Filed: 2026-09-14
Branch: `external-pi-mattermost` (from `origin/main` @ a2140ac)
Supersedes the "scoped to harness-owned sessions" carve-out in
`specs/2026-07-06-pi-transcript-observer.md`.
Companion: mm-bridge `specs/20260914-external-pi-sessions/design.md`.

## Problem

A pi session started in a terminal is invisible to the harness. The
observer parses its transcript fine, but `publish_line` **skipped** every
message whose session the repository didn't already own:

> External-origin pi sessions aren't synthesized yet: pi splits cwd
> (`session` record) and model (`model_change` record) across two
> records, so a single record can't build a `session.updated` the way
> claude/codex do. — `observer.py`, pre-change

So pi was the one backend where `.sessions` never listed your terminal
work and `.invite` refused outright ("it can't be resumed from
Mattermost"). This note makes external pi sessions discoverable and
continuable, and — the part that needed empirical work — makes
continuation either genuinely resume the conversation or say so.

## pi semantics (verified empirically against pi v0.84.2, 2026-09-14)

Probes ran in throwaway `/tmp` directories with a bogus `--api-key`
(`not-a-real-key`) against pre-seeded synthetic transcripts. No real
credentials were read and no personal transcript was touched.

1. **`--session <abs path>` appends to that exact transcript, from any
   cwd.** Verified: prior user *and* assistant turns reached the provider,
   the file grew in place, no second file appeared. This is the resume
   form the harness uses for observed sessions.

2. **Both resume forms are create-if-missing.** `--session-id <unknown>`
   prints `Warning: No project session found with id <uuid>; creating a
   new session with that id`; `--session <nonexistent path>` writes that
   exact file. Both exit 0 with a plausible reply and no history. There is
   no "resume or fail" mode to ask for. **This is the central hazard: the
   failure mode is a silent wrong answer, not an error** — and it is why
   every resume path here is fail-closed.

3. **The session directory is `$PI_CODING_AGENT_DIR|$HOME/.pi/agent/sessions/--<cwd
   with `/`→`-`>--/`.** The root moves with `PI_CODING_AGENT_DIR` and with
   the harness's own `--observe-root`, and the `/`→`-` mapping is lossy (a
   literal `-` is indistinguishable from a separator). So a transcript path
   **cannot be reconstructed** from a session id — it has to be recorded
   when observed.

4. **A model reference must be fully provider-qualified, and `:` in a
   model id is safe.** With a catalogue holding both `openrouter/free` and
   `qwen/qwen3-coder:free`, a *shortened* `--model openrouter/free` is
   fuzzy-matched to the **qwen** entry; the fully-qualified form
   (`openrouter/openrouter/free`) preserves the exact route (independent
   review, against the installed `resolveCliModel`). Qualification is
   therefore about naming the right model, not merely about tidiness. pi
   interprets the `provider/` prefix only when `--provider` is absent
   (`core/model-resolver.js`, guarded by `if (!provider)`), so the
   qualified form and the flag are alternatives, not complements. The `:`
   suffix is split only when it is a valid thinking level — verified,
   `--model review-probe/glm-5.2:cloud` with no `--provider` reached the
   provider as model `glm-5.2:cloud`.

5. **Assistant records carry `provider` and `model` inline** (observed in
   the probe output alongside `api`, `usage`, `stopReason`).

6. **Two live pi instances on one transcript produce sibling branches**
   (independent review, PTY probe): a headless run appends and receives the
   original context, but a still-open TUI keeps its own in-memory leaf. Its
   next turn chains onto the *same parent*, and whoever reopens the file
   sees only the branch appended last — the other side's turns are absent
   from the model's context. Not interleaving: silent divergence.

## Design

### 1. Accumulate, then synthesize (`src/agent_harness/pi_discovery.py`)

New module, because the state pi forces on us doesn't belong in a 2200-line
stream processor:

- `PiSessionFacts` — frozen `(cwd, provider, model, created_at)`, with a
  `qualified_model` property. `merged_with` lets a later record fill gaps
  but **never blank** a known value: an observation overwrites the stored
  Session's model, so announcing `model=None` after learning a model would
  wipe it.
- `pi_facts_from_record` — what one record discloses. `session` → cwd;
  `model_change` → provider + modelId; assistant `message` → provider +
  model (finding 5, the fallback when `model_change` is behind the
  observer's persisted offset).
- `read_pi_head_facts(path, max_lines=64)` — re-derive facts from a
  transcript's head. Never raises.
- `PiTranscriptRegistry` — LRU-bounded (512) per-transcript accumulator.
  `take_announcement` returns facts worth emitting a `session.updated`
  for: `None` until the cwd is known, and `None` again whenever facts are
  unchanged since the last announcement.

### 2. Parser + observer wiring

`parse_transcript_line` / `parse_transcript_record` take an optional
`pi_registry`. Omitting it preserves the previous stateless behavior
exactly (metadata records stay silent), so the parser remains usable as a
pure function.

`_parse_pi_record` folds every record into the registry and prepends at
most one synthesized `session.updated` (`origin=external`, `backend=pi`,
project from cwd, provider-qualified `model` and the observed transcript
path when known) to whatever the record itself produces.

`ExternalTranscriptObserver` owns the registry and:

- **hydrates once per transcript path, newest source first.** If the
  repository already holds the session, its facts are authoritative —
  they reflect every `model_change` the previous process saw. Only when
  the harness has never heard of the session does it fall back to peeking
  the transcript head, which by definition shows only the model the
  conversation *opened* with. Getting this order wrong dragged a
  long-running session back to its first model after a restart.
- filters two parser events in `_keep_pi_event`:
  - `session.updated` for a session the harness already **owns** — the
    repository's origin-downgrade guard would refuse the write anyway, but
    the event still travels the bus and would show a harness session
    flipping to `origin: external`.
  - `message` for a session that doesn't exist and couldn't be
    synthesized — i.e. the transcript never stated a usable cwd, so there
    is no project path to resume it from. **Skipped, never buffered**: an
    operator accumulates many interactive pi sessions and
    `_pending_materialization` would grow without bound. This is the only
    surviving piece of the old blanket guard, now with a truthful reason
    and a WARNING instead of a DEBUG.

### 3. `Session.pi_transcript_path`, and a single model field

The observer records the **absolute path of the transcript it actually
read**. Per finding 3 a path cannot be reconstructed from a session id, so
anything derived (`Path.home()` + sanitized cwd + a globbed timestamp)
breaks the moment `PI_CODING_AGENT_DIR` or `--observe-root` moves. It is
stored absolute because runs are launched with `cwd` set to the *project*
directory — a relative path would resolve against the project and address
a different file, which pi would then create.

The model is stored **provider-qualified** (`ollama/glm-5.2:cloud`) in the
existing `Session.model`. This identifies both provider and model through
one CLI argument; no separate `--provider` flag is needed. Model IDs
containing `:` remain intact, as verified in finding 4.

Qualification joins unconditionally, because a shortened reference names
the wrong model (finding 4): `openrouter` + `openrouter/free` must become
`openrouter/openrouter/free`. pi splits on the first slash only, which
resolves it correctly. A
`model_change` carrying a provider but *no* model id yields no facts at
all — on its own a provider can't name a model, and keeping it would
re-qualify an already-qualified value on the next merge.

### 4. Fail-closed resume, in two places

`pi_resume_target(session)` returns the transcript only when all three
hold: a path was recorded, the file exists and is non-empty, and its first
record is a `session` header whose `id` is this session's UUID. The header
check — not the filename — is what stops a moved, truncated or reused file
being resumed under the wrong session's name.

It is enforced **twice, deliberately**:

- `validate_session_resume_target` runs at preflight, so `POST /v1/runs`
  returns `409 Conflict` with the reason as `detail` before any run row is
  created.
- `PiCommandBuilder` re-checks at build time and raises rather than
  falling back to `--session-id`. Preflight and build are different
  moments — a transcript can vanish in between, and a caller could skip
  the preflight — and the fallback would be exactly the silent new
  conversation this design exists to prevent. There is no safe default,
  so there isn't one.

Harness-origin pi sessions are untouched: they own their id, keep
`--session-id`, and still create their conversation on the first run.

### 5. Preserving what an observation doesn't own

`merge_observed_session` (in `models.py`, shared by the in-memory and
SQLite repositories so the rules can't drift) defines what a
`session.updated` from the observer may overwrite: `backend`, `model`,
`project`, `status`, `updated_at`, plus `codex_resume_id` /
`pi_transcript_path` when non-None. Everything else — `title`, `effort`,
`bypass_permissions`, `stats`, `created_at`, `forked_from` — belongs to
the user or the harness and survives untouched. The allowlist is the
narrow side on purpose: a field added later defaults to "preserved".

`created_at` comes from the `session` record's own timestamp, so a
conversation started months ago isn't reported as new.

## Limitations (deliberate, documented)

- **Local filesystem only.** The observer is a `watchfiles` watcher over
  local paths and resume invokes the CLI locally. A laptop's pi sessions
  do not appear in a harness on another host; that needs a harness+bridge
  on the laptop, or transport landing the transcripts under an
  `--observe-root`. Stated in README.md and docs/hybrid-semantics.md.
- **Shared transcript, and divergence rather than interleaving.** Per
  finding 6, a still-open TUI and a headless run do not merge into one
  thread — they become sibling branches, and reopening keeps only the last
  one appended. pi has no locking protocol to coordinate this, so the
  mitigation is a procedure, not a guarantee: mm-bridge's notice tells the
  operator to **close the terminal session before continuing it elsewhere
  and reload it before going back**.
- **No Herdr adapter.** Driving a live pi/Herdr TUI is out of scope. This
  is headless continuation only.
- **A cwd-less transcript stays invisible.** Without a `session` record
  we cannot locate the conversation for resume, so mirroring its messages
  would produce a dead session. Skipped with a WARNING.

## Acceptance criteria

| # | Criterion | Test |
|---|---|---|
| 1 | A pi transcript with a cwd yields an `origin=external` pi session | `test_observer_discovers_an_external_pi_session_and_all_its_messages` |
| 2 | A user turn written *before* `model_change` is not lost | `test_observer_announces_a_live_pi_session_before_its_model_is_known` |
| 3 | Model stays `None` until pi discloses it, then fills in | same |
| 4 | Provider is preserved, always qualified into the model, never re-qualified | `test_qualified_model_never_strips_a_repeated_provider_segment`, `test_a_partial_model_change_cannot_double_qualify_the_model` |
| 5 | Harness-owned pi sessions are unchanged (no downgrade, no synthetic event) | `test_observer_leaves_a_harness_owned_pi_session_alone` |
| 6 | Restart / pre-existing transcript recovers via head peek | `test_observer_backfills_pi_facts_from_the_file_head_after_a_restart` |
| 7 | Duplicate ingestion emits one session row, not one per line | `test_observer_does_not_duplicate_session_updated_on_a_re_tail`, `test_registry_does_not_re_announce_unchanged_facts` |
| 8 | Malformed records never drop the session or crash the tail | `test_observer_tolerates_malformed_pi_records_without_dropping_the_session`, `test_facts_from_malformed_records_are_empty` |
| 9 | State is bounded | `test_registry_is_bounded_and_evicts_least_recently_used` |
| 10 | A cwd-less transcript is skipped, not buffered | `test_observer_skips_pi_messages_when_the_transcript_never_states_a_cwd` |
| 11 | `POST /v1/runs` resumes the *same* conversation headlessly | `test_run_on_an_external_pi_session_resumes_the_same_conversation` |
| 12 | A missing transcript 409s with a truthful reason, creating no run | `test_run_on_an_external_pi_session_409s_when_the_transcript_is_gone` |
| 13 | A transcript belonging to another conversation is refused | `test_resume_target_rejects_a_transcript_for_a_different_conversation`, `test_run_on_an_external_pi_session_409s_when_the_transcript_is_a_stranger` |
| 14 | The builder never falls back to create when the source vanishes | `test_the_builder_refuses_rather_than_creating_when_the_source_vanishes` |
| 14a | The builder resolves the transcript once, so a restored file can't yield `--session None` | `test_the_builder_resolves_the_transcript_exactly_once` |
| 15 | The resume path is absolute | `test_resume_target_is_absolute_even_for_a_relative_observation` |
| 16 | A refresh keeps title / effort / bypass_permissions / created_at | `test_metadata_refresh_keeps_user_owned_session_settings`, `test_session_keeps_the_conversations_original_creation_time` |
| 17 | Full stack against the real pi CLI | `scripts/pi_external_acceptance_probe.py` |
| 18 | claude/codex behavior unchanged | full suite green |

Independent review (2026-09-14) reproduced seven defects in the first
implementation — lost `created_at`, model regression after restart, wiped
user settings, a home-relative path lookup that missed custom
observe-roots, a stripped provider namespace, a relative resume path, and
a builder that fell back to create-if-missing when its source check
failed. Fixing the last of those introduced an eighth (a double
resolution that could emit `--session None`), also caught in review.
Their reproducers are integrated above; `scripts/pi_external_acceptance_probe.py`
is their end-to-end probe, kept in-tree.

Tests: `tests/test_pi_external_sessions.py` (62), plus the retargeted
`test_observer_skips_pi_message_for_unlocatable_session` in
`tests/test_pi_observer.py` — the old
`test_observer_skips_pi_message_for_unknown_session` asserted the blanket
skip this note removes; its transcript has no `session` record, so it now
documents the cwd-less case with an accurate name and comment.

```
uv run pytest -q
```
