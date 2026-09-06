# Durable Codex rollout ownership

Issue: https://github.com/tijszwinkels/agent-harness/issues/37
Related spec: [Codex session resume](2026-05-21-codex-resume.md).

## Contract

Before resolving an unbound Codex rollout through spawn expectations or
the external-session fallback, look up its filename UUID in persisted
harness-origin Codex sessions' `codex_resume_id` fields. A unique owner
wins regardless of expectation age, cached misses, cwd, or session status.
Restore the path binding and retire that owner's spawn expectations.
Transcript activity must not unarchive an archived session.

This lookup survives run cleanup and observer/database restarts. It does
not require parsing `session_meta`; incomplete transcript lines still wait
for their terminating newline before ingestion. Unknown UUIDs continue
through the existing expectation and external-session paths.

An ambiguous owner or failed repository lookup logs the affected path
and defers ingestion without advancing the offset. An existing external
duplicate cannot win over a harness owner. Historical messages and rows
are not migrated; only subsequent ingestion is routed correctly.

SQLite schema version 4 adds a non-unique partial expression index on
`codex_resume_id` for harness-origin Codex sessions. The lookup reads only
matching rows; it does not deserialize the full session store per rollout.
The additive index is compatible with the previous application version.

Known `event_msg` metadata (`item_completed`, `token_usage_record`,
`world_state`) is logged at debug level. Unknown shapes retain warnings;
message, usage, and end-turn parsing keep their existing paths.

## Validation

`tests/test_codex_rollout_binding.py` covers a 31-minute delay, no active
expectation, run cleanup, competing same-cwd expectations, archived owners,
SQLite reopen with persisted offsets, cached misses, pre-existing external
duplicates, ambiguous owners, transient lookup failure/retry, unrelated
sessions, and metadata log levels. Ownership tests exercise both repositories;
the SQLite cases also check delivery through the durable session event stream.

Sources retrieved 2026-09-06: issue #37 above;
[SQLite expression indexes](https://www.sqlite.org/expridx.html);
[pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html).
