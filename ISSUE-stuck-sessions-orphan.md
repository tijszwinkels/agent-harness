# Stuck sessions: orphaned background task wedges the post-exit stream drain

## Symptom

Sessions occasionally "get stuck and stop responding": the agent finishes
its turn, but the harness never publishes a terminal `run.*` event. The
bridge therefore never sees the run end — the typing indicator stays on,
the run stays `running` forever, and the session can't accept the next
prompt (a session's runs are serialized; a never-finishing run blocks the
queue indefinitely).

## Root cause

File: `src/agent_harness/orchestrator.py`

The agent subprocess is spawned with `stdout=PIPE, stderr=PIPE,
start_new_session=True` (`AsyncioProcessFactory`, ~line 239-248). `RunProcess.run`
awaits the **foreground** child and then drains its pipes:

```
returncode = await wait_task                       # orchestrator.py:333  (foreground child exit)
await self._drain_streams_after_exit(stream_tasks) # orchestrator.py:334  (was: _finish_streams)
```

The pre-fix drain was an unbounded `asyncio.gather` over the two
`_stream_lines` reader tasks, each blocked on:

```
while line := await stream.readline():             # orchestrator.py:512
```

`readline()` only returns the empty sentinel at **pipe EOF**. The pipe
EOFs when *every* writer end is closed. If the agent (e.g. Claude Code)
promoted a **background** task — `curl /v1/events` with no `-m`, an
`httpx.stream(...)` with no bound, etc. — that descendant:

1. lives in the **same process group** (`start_new_session=True` group), and
2. **inherited the stdout/stderr pipe write-end** from the foreground child.

So even after the foreground child exits, the inherited write-end keeps the
pipe open. `readline()` never EOFs ⇒ the gather never returns ⇒
`_finish_streams` blocks forever ⇒ `run.completed` (~line 355) is never
published ⇒ the run hangs `running` permanently.

### Why the two existing safety nets didn't catch it

* `_watch_end_turn_cleanup` (~line 600-644) only **arms** when the observer
  publishes `run.end_turn` (claude `stop_reason=="end_turn"` / codex
  `task_complete`). If the turn didn't end cleanly (crash, partial output,
  no clean stop record), it never arms — no SIGTERM ladder, no cleanup.
* `_watch_idle_timeout` (~line 646-679, `IDLE_TIMEOUT_SECONDS = 30min`)
  `killpg`s the group on inactivity, but `_last_activity_at` is re-warmed by
  **every** non-empty stdout line (`_stream_lines`, ~line 521) and every
  stderr line (`process.stderr` ∈ `_ACTIVITY_EVENTS`, `_publish` ~line 791).
  A **chatty** orphan that keeps writing to the inherited pipe re-warms the
  timer on every line, so the 30-min idle kill is deferred indefinitely →
  permanent hang.

## The fix (chosen: A + B)

Centered on **(A) bound the post-exit drain**, plus **(B) reap the orphaned
process group** — both small and using existing machinery.

`run()` now calls `_drain_streams_after_exit` (orchestrator.py:536) instead
of the unbounded `_finish_streams`:

```python
_done, pending = await asyncio.wait(
    tasks,
    timeout=POST_EXIT_DRAIN_GRACE_SECONDS,        # = 5.0s, orchestrator.py:33
    return_when=asyncio.ALL_COMPLETED,
)
if not pending:
    await self._finish_streams(tasks)             # clean EOF: drain all output, surface reader errors
    return
# Grace expired with readers still blocked → an orphan holds the pipe.
self._reap_process_group(signal.SIGTERM)          # release the inherited write-end (B)
await self._cancel_streams(list(pending))         # stop waiting on the held pipe (A) — correctness backstop
```

Run completion is now bound to the **foreground child's exit** (`wait_task`),
not to inherited-pipe EOF. After the foreground child exits, the drain gets a
finite 5s grace; if EOF doesn't arrive, the run is logically over, so we reap
the group and abandon the readers and proceed to publish the terminal event.

`_reap_process_group` (orchestrator.py:762) is a deliberate sibling of the
existing `_signal_process_group`, differing in exactly the two ways the
orphan-reap path requires:

* **No `returncode is not None` short-circuit.** `_signal_process_group`
  bails once the foreground process has a returncode (correct for the
  watchdog paths, which signal a *live* process). At reap time the
  foreground child has *already* exited — the targets are the *other*
  members of its group.
* **Targets `process.pid` directly as the pgid** instead of
  `os.getpgid(process.pid)`. The child is the `start_new_session=True` group
  leader, so the pgid equals its pid; and once asyncio has reaped the exited
  leader, `os.getpgid(pid)` would raise `ProcessLookupError` even while
  orphan group members are still alive.

Correctness does **not** depend on the SIGTERM landing: even if the killpg
fails (group already gone, or a grandchild escaped the group),
`_cancel_streams` unblocks the readers so the run always completes.

The grace is a real wall-clock `asyncio.wait(timeout=...)` rather than the
injected `self._sleep`. This keeps the run hot-path's clean-completion case
cheap (no extra task/timer hops loaded onto every normal run) and is correct
because the timeout is small and bounded. The injected sleep stays reserved
for the watchdogs that genuinely need a controllable clock.

## Normal case is preserved

When there's no orphan, the foreground child's exit closes the only pipe
writer, both readers EOF within milliseconds, `ALL_COMPLETED` fires
immediately (`pending` empty), and `_finish_streams` drains every buffered
line exactly as before — no truncation, no added wall-clock latency.

## Alternatives considered / rejected

* **(C) alone — stop late stdout/stderr from re-warming `_last_activity_at`
  after foreground exit.** Would defang the chatty-orphan idle-defer, but
  the run would *still* hang up to 30 min (until the idle watchdog finally
  fires) for a *silent* orphan, and the completion is still gated on EOF.
  (A) makes completion bound to the foreground exit directly, which is the
  actual invariant we want. (C) is subsumed: once (A) abandons the readers,
  late lines can't re-warm anything because there's no reader left running.
* **Just `killpg` on the normal-exit path, keep the unbounded gather.**
  Reaping helps, but if a grandchild has escaped the process group (double
  fork, `setsid` in the child) the pipe still never EOFs and the gather
  still hangs. (A)'s cancellation is the guarantee; (B) is hygiene.
* **`asyncio.wait_for(gather, timeout=...)` wrapping a gather task.** Works,
  but adds ~6 event-loop turns to the clean-completion hot-path (measured),
  which surfaced as a turn-count regression in
  `test_api.py::test_rapid_back_to_back_creates_serialize_through_real_run_manager`.
  The single `asyncio.wait(tasks, timeout=..., ALL_COMPLETED)` is tighter
  (only the readers, no wrapper task, no separate sleep task) and within the
  existing turn budget.

## How verified (TDD)

`tests/test_orchestrator.py`, all driven by the `FakeProcess` /
`FakeStream` fakes (`close_stdout=False` / `close_stderr=False` model an
orphan: no EOF sentinel is ever pushed):

* `test_run_completes_when_orphan_holds_pipe_open_after_foreground_exit` —
  foreground `wait()` resolves, `readline()` never returns. Pre-fix:
  `RunProcess.run()` hangs (caught as `asyncio.wait_for` `TimeoutError`).
  Post-fix: completes and publishes `run.completed` within the grace.
* `test_run_reaps_process_group_when_orphan_wedges_drain` — asserts a
  SIGTERM lands on the group keyed by the child **pid** (not `os.getpgid`'s
  value) when the drain times out.
* `test_run_completes_with_chatty_orphan_after_foreground_exit` — the orphan
  keeps emitting lines after the foreground exit (re-warming
  `_last_activity_at`); the run still terminates via the bounded drain.
* `test_run_drains_full_output_when_streams_eof_promptly` — **mandatory
  regression**: no orphan, readers EOF promptly; every stderr line is
  drained and `run.completed` fires with the full output, exact event order
  unchanged. (Passes on both old and new code — proves the fix didn't
  truncate the clean path.)

RED was confirmed by running the new tests against `HEAD`'s orchestrator
(the three orphan tests time out / hang). GREEN with the fix. Full suite:
`uv run pytest -q` → **249 passed** (1 pre-existing unrelated FastAPI
deprecation warning).

## Residual risk

* **A grandchild that fully detaches from the group** (own `setsid` after a
  double fork) won't be reaped by `killpg`, and won't be killed — it lives
  on as a true orphan holding nothing of ours. The run still **completes**
  correctly (readers are cancelled regardless), so the session is no longer
  stuck; the cost is a leaked descendant process, which is out of scope for
  the stuck-session fix and was already possible before.
* **5s grace tradeoff.** A real run whose pipe legitimately takes >5s to
  flush its tail after the foreground process exits would have its trailing
  output truncated. In practice the pipe closes synchronously with the
  foreground child's exit when there's no orphan, so 5s is comfortable; the
  constant is module-level and easy to tune.

## Related bridge-side gap (follow-up, NOT fixed here)

`mm-bridge/src/mm_bridge/bridge.py` `_run_typing_watchdog` (lines 641-657)
stops the typing **indicator** after `typing_stop_after_silence_seconds` of
harness silence, but it never issues a terminal action against the stuck run
(no `DELETE`/interrupt to the harness). So before this harness fix, a wedged
run left the bridge's typing cleared but the harness-side run still
`running`. This harness fix resolves the wedge at the source. A defensive
follow-up on the bridge — have the typing watchdog also reconcile/cancel a
run that's been silent past the threshold — would harden against any *future*
harness-side hang, but belongs in the mm-bridge repo, not here.
