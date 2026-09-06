# Per-session effort

Review of the existing effort feature, 2026-09-06. Integration base:
`737c4a9`; original commits: `cd9b0ac`, `27a1554`, `8bc09b1`.

## Contract

- `Session.effort` and create-request effort are optional, nullable, non-empty
  strings. Older stored sessions without the field load with `None`.
- Effort is deliberately free-form, like model. The harness neither translates
  levels nor guarantees that a given backend/model supports one.
- PATCH omission leaves effort unchanged. A string pins the new value; null
  clears the harness override. Null title still returns 422.
- The override is emitted for every newly built command, including resume and
  fork commands. Already-running and already-queued commands retain their argv.
- No override means no flag: CLI defaults and saved resume settings apply.
  Clearing the override does not erase the backend's own saved session settings.
- Forks inherit the parent's effort. The existing Claude/pi fork support and
  Codex fork rejection remain as described in `2026-07-20-session-fork-route.md`.
- Create/PATCH owns effort. Transcript events preserve the current stored value,
  including clears; observer events expose that value to subscribers.
- Every registered backend advertises `capabilities.effort = true`.

## Backend verification

Sources retrieved on 2026-09-06; CLI help inspected locally the same day:

- Claude Code **2.1.258**: `--effort <level>`; help lists `low`, `medium`,
  `high`, `xhigh`, `max`. The [official CLI reference](https://code.claude.com/docs/en/cli-reference)
  specifies a session override that does not persist, with model-dependent
  support. Emit it on every invocation.
- Codex CLI **0.153.4**: both `exec --help` and `exec resume --help` accept
  `-c/--config <key=value>`. Use `-c model_reasoning_effort=<level>` on both
  paths. The [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
  lists `minimal`, `low`, `medium`, `high`, `xhigh`; it does not promise Claude's
  `max`. The installed CLI passes arbitrary strings onward, so local parsing
  alone does not demonstrate that a model supports a value.
- pi **0.84.2**: `--thinking <level>`; installed help and
  `dist/cli/args.js` list `off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
  `max`. [Upstream argument parser](https://github.com/earendil-works/pi-mono/blob/main/packages/coding-agent/src/cli/args.ts).
  CLI acceptance does not guarantee that the provider/model uses the requested
  level unchanged.

## Review fixes and verification

Regression tests failed before fixing PATCH clearing, stale session payloads
overwriting effort in both repositories, and observer events publishing a null
override. OpenAPI regressions also exposed missing `minLength` constraints and
the non-nullable PATCH schema. The shared-value-space claim was removed.

Coverage includes create/PATCH/defaults, fork inheritance, new/resumed commands,
observer delivery, stale updates after clearing, and SQLite persistence. Tests
use temporary storage and synthetic transcripts; authenticated model execution
and production restart/deployment are outside this review's validation.
The final full suite passes on Python 3.12.3 and 3.13.13: **412 tests** on each,
with one existing FastAPI deprecation warning. `git diff --check` also passes.

Related specifications: `openapi.yaml`, `2026-05-21-codex-resume.md`,
`2026-07-20-session-fork-route.md`, `2026-07-05-pi-backend-deviations.md`.
