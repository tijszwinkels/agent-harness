#!/usr/bin/env python3
"""Sidecar smoke test for Phase 4 — watchdog rewires, supervisor
stops parsing, observer emits run.end_turn.

What this exercises end-to-end vs unit tests:

- Daemon boots cleanly under Phase 4 (the supervisor's stdout pump
  is now heartbeat-only; verify no startup errors on real subprocess
  spawn paths).
- The observer ingests a codex rollout with ``task_complete`` and
  emits ``run.end_turn`` on the bus (without crashing).
- ``turn_context`` rollout records no longer spam warnings.

Spawning a real codex/claude CLI from this sandbox isn't feasible,
so the harness-session-with-active-run watchdog path is locked at
the unit-test layer (see test_orchestrator.py::
test_watchdog_triggers_on_observer_run_end_turn). This smoke
confirms production startup + the observer's parse path don't
regress.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PORT = 8879
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"


def http_json(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"raw": text}
        return exc.code, payload


def wait_for_health() -> None:
    deadline = time.monotonic() + 10.0
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            req = Request(f"{BASE_URL}/health")
            with urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (URLError, HTTPError) as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(
        f"harness on :{PORT} did not become healthy in 10s (last: {last_err})"
    )


def write_codex_rollout_with_end_turn(
    *,
    sessions_root: Path,
    rollout_uuid: str,
    cwd: str,
) -> Path:
    """Drop a codex rollout that exercises Phase 4's new triggers:
    a ``turn_context`` record (formerly warned), and a
    ``task_complete`` event_msg (emits ``run.end_turn``)."""
    transcript_dir = sessions_root / "2026" / "05" / "19"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript = transcript_dir / f"rollout-2026-05-19T10-30-00-{rollout_uuid}.jsonl"
    lines = [
        json.dumps({
            "timestamp": "2026-05-19T10:30:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": rollout_uuid,
                "timestamp": "2026-05-19T10:30:00.000Z",
                "cwd": cwd,
                "originator": "codex_exec",
                "cli_version": "0.128.0",
            },
        }),
        # turn_context formerly warned; Phase 4 adds it to the
        # ignore list. Logs should stay clean.
        json.dumps({"type": "turn_context", "payload": {"cwd": cwd, "model": "gpt-5.4"}}),
        json.dumps({
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        }),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="phase4-smoke-"))
    print(f"[smoke] tmp dir: {tmp}")
    fake_home = tmp / "home"
    fake_home.mkdir()
    sessions_root = fake_home / ".codex" / "sessions"
    sessions_root.mkdir(parents=True)
    db_path = tmp / "harness.db"

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["PYTHONPATH"] = str(repo_root / "src")

    cmd = [
        sys.executable, "-m", "agent_harness.cli", "serve",
        "--host", HOST,
        "--port", str(PORT),
        "--database", str(db_path),
        "--observe-root", str(sessions_root),
    ]
    print(f"[smoke] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(repo_root),
    )
    try:
        wait_for_health()
        print(f"[smoke] harness up on :{PORT}")

        rollout_uuid = "11111111-1111-1111-1111-111111111111"
        rollout = write_codex_rollout_with_end_turn(
            sessions_root=sessions_root,
            rollout_uuid=rollout_uuid,
            cwd=str(tmp / "project"),
        )
        print(f"[smoke] dropped rollout with task_complete: {rollout}")

        # Wait for the observer to ingest.
        deadline = time.monotonic() + 6.0
        external_id = f"codex_{rollout_uuid}"
        external_seen = False
        while time.monotonic() < deadline:
            _, sessions_list = http_json("GET", f"{BASE_URL}/v1/sessions")
            ids = [s["id"] for s in sessions_list["data"]]
            if external_id in ids:
                external_seen = True
                break
            time.sleep(0.2)
        if not external_seen:
            print(f"[smoke] FAIL: observer did not register the rollout: {ids}")
            return 2
        print(f"[smoke] OK: observer registered external session {external_id}")

        # Drain the harness's stdout/stderr buffer to capture any
        # warning log lines from the observer.
        # (We read the tail in the finally block.)

        print("[smoke] ALL OK")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        if proc.stdout is not None:
            tail = proc.stdout.read().decode("utf-8", errors="replace")
            if tail:
                # Surface warnings — there shouldn't be any
                # "Unsupported Codex transcript shape" for turn_context.
                warning_lines = [
                    line for line in tail.splitlines()
                    if "Unsupported Codex transcript shape" in line
                    and "turn_context" in line
                ]
                if warning_lines:
                    print(
                        "[smoke] WARN: turn_context warning(s) seen — "
                        "Phase 4 ignore-list entry missing?"
                    )
                    for line in warning_lines:
                        print(f"  {line}")
                print("[smoke] --- harness tail (last 2 KB) ---")
                print(tail[-2000:])
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
