# Idle watchdog kills actively-working runs at start+31 min

Filed: 2026-07-23
Worktree: `worktrees/fix/idle-watchdog-heartbeat`
Branch: `fix/idle-watchdog-heartbeat`

## Symptom

Harness-owned claude runs get SIGTERM'd by the idle watchdog at almost exactly
`start + 31 min`, while the run's Mattermost channel shows narration and
Edit/Bash/Read tool activity every 1–2 minutes throughout the window.

Verified evidence (dispatching session, 2026-07-23):

- `ses_c010f19b01d1…` / `run_5b9cd4df`: started 20:53:15Z, `interrupted`
  21:24:18Z (= start + 31 min 03 s) — visibly active the whole time.
- Same signature earlier: `ses_d872bf01…` killed "after 30 minutes of
  inactivity" seconds after visibly active work (2026-07-21).

`31 min` is the tell: `IDLE_CHECK_INTERVAL_SECONDS = 60` × `IDLE_TIMEOUT_SECONDS
= 1800`. The first check past `1860 s` fires, and it fires only if
`_last_activity_at` was **never advanced from run start** (frozen within the
first minute).

## Root cause (empirically reproduced through the real spawn path)

Ruled out first: **not** a deployed-build mismatch. The live `:8877` daemon
(PID confirmed, editable install, `orchestrator.py` mtime predates process
start) runs the current Phase-4 stdout-heartbeat code. This is a real code bug.

The idle watchdog's only two activity sources are (a) the stdout heartbeat in
`_stream_lines` (every non-empty stdout line ticks `_last_activity_at`), and
(b) `process.stderr` events. Claude talks via its rollout file, rarely via
stderr, so the stdout heartbeat is the load-bearing signal for claude runs.

`AsyncioProcessFactory` spawns the subprocess with
`asyncio.create_subprocess_exec(...)` and **no `limit=`**, so stdout/stderr use
asyncio's default `StreamReader` limit of **64 KB**. claude
`--output-format stream-json` emits each protocol event as one line. A `user`
turn echoing a large `Read`/`Bash` tool_result — or the init event — routinely
exceeds 64 KB in a single line. When that happens `readline()` raises
`ValueError: Separator is not found, and chunk exceed the limit`
(`LimitOverrunError`). That raise is **uncaught** in `_stream_lines`, so the
stdout reader task dies permanently. The heartbeat never ticks again; ~31 min
later the idle watchdog SIGTERMs the (still working) run.

The observer keeps tailing the rollout **file** — a completely separate path —
so Mattermost keeps showing tool activity while the idle clock is frozen. That
mismatch is exactly the reported "active but killed" signature.

Reproduced deterministically through the real spawn path (probe script):

```
[   4.1s] line#  1 len=253    type=system
[   4.1s] line#  2 len=253    type=system
[   4.1s] line#  3 len=324    type=system
[   6.7s] readline() RAISED ValueError: Separator is not found, and chunk exceed the limit
```

The oversized line was the `user` turn echoing a >64 KB file Read. In the real
`_stream_lines` that ValueError propagates and kills the reader (see
`_finish_streams`, which only surfaces the exception at drain time — long after
the heartbeat has frozen).

## Fix — defense in depth

### Layer 1 — root cause: `_stream_lines` can't die on line size

- `AsyncioProcessFactory` passes `limit=STDOUT_STREAM_LIMIT_BYTES` (8 MB) so
  normal large lines (big tool_results, init event, base64 images) read whole,
  keeping `_note_stdout_task_event` intact.
- `_stream_lines` wraps `readline()` in `try/except ValueError`: an oversized
  line is genuine activity (the subprocess is producing output), so it ticks
  `_last_activity_at` and keeps reading instead of letting the reader die.
  Losing the oversized line's *content* is harmless — stdout is heartbeat +
  task-lifecycle only (task records are tiny), and message data flows via the
  observer.

### Layer 2 — makes the bug class un-reintroducible: watchdog subscribes to observer events

`RunProcess._watch_activity_events` subscribes to the shared event bus for the
run's session and ticks `_last_activity_at` on every observer-emitted activity
event (`message` / `message.delta` / `tool_use` / `run.usage`). Any visible
work keeps the run warm **regardless of what stdout does** — a second,
independent activity source. `after = max_sequence` at run start skips a prior
turn's replayed activity (same guard as `_wait_for_run_end_turn`); session
scoping is sufficient because the RunManager keeps at most one active
subprocess per session.

### Layer 3 — configurable timeout

`serve --idle-timeout-seconds` (default 1800) threads through
`RunManager → RunProcess._idle_timeout_seconds`. Resolved at construction so
`IDLE_TIMEOUT_SECONDS` stays monkeypatchable in tests. The `run.timed_out_idle`
event reports the configured value.

### Layer 4 (stretch, separate commit — droppable in review)

At timeout, before killing: a process-tree activity check. If live child
processes have accumulated CPU since the previous check, the run is in a single
long-but-silent tool call (e.g. a >30-min build or transcription where BOTH the
stdout heartbeat AND observer events are silent) — defer and re-check. A
quiescent tree is killed as today. A `run.idle_warning` event is emitted on the
first deferral so a frontend can notify. The hard timeout remains the backstop.

## Non-goals

- Changing end-turn cleanup (`_watch_end_turn_cleanup`) — unaffected.
- Message/usage materialization — still observer-owned; the stdout heartbeat
  and the new subscription are diagnostic-only (they never publish).

## Deploy note

The `:8877` daemon must be restarted (coordinated) after merge to pick this up.
