#!/usr/bin/env python3
"""Sidecar smoke test for the ``Session.stats.context_used`` feature.

What this exercises end-to-end vs unit tests:

- Daemon boots cleanly with the new field plumbed through OpenAPI,
  models, observer emission, and materialization.
- A multi-turn codex rollout with growing
  ``info.total_token_usage.total_tokens`` parses without warnings.
- The observer registers the rollout as an external session via
  ``session_meta``; nothing crashes when the new snapshot field is
  set on subsequent ``token_count`` events.

Spawning a real codex/claude CLI from this sandbox isn't feasible,
and the rollout-only path produces an external session WITHOUT a
harness Run (so the run.usage materializer drops the events — this
is intentional per Phase 3 design, locked in by
``test_observer_skips_usage_publish_when_no_active_run``). The
end-to-end cumulative-vs-snapshot proof runs as a unit-style
integration test in
``tests/test_observer.py::test_observer_cumulative_vs_snapshot_end_to_end_claude``
which exercises the real DurableEventBus + SQLite materializer on
a multi-turn rollout.

This smoke confirms the production startup + parse path don't
regress and that the new field is documented in the live API.
Spec: specs/2026-05-19-context-used.md
"""

from __future__ import annotations

import json
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


def write_codex_rollout_with_growing_snapshots(
    *,
    sessions_root: Path,
    rollout_uuid: str,
    cwd: str,
) -> Path:
    """Drop a codex rollout with three token_count events whose
    ``total_token_usage.total_tokens`` grows turn over turn — the
    minimum sequence that demonstrates the cumulative-vs-snapshot
    distinction at the parse path."""
    transcript_dir = sessions_root / "2026" / "05" / "19"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript = transcript_dir / f"rollout-2026-05-19T10-30-00-{rollout_uuid}.jsonl"

    def token_count(total: int, cached: int) -> dict:
        return {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cached_input_tokens": cached,
                    },
                    "total_token_usage": {"total_tokens": total},
                    "model_context_window": 258400,
                },
            },
        }

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
        json.dumps({"type": "turn_context", "payload": {"cwd": cwd, "model": "gpt-5.4"}}),
        json.dumps(token_count(total=10000, cached=2000)),
        json.dumps(token_count(total=25000, cached=15000)),
        json.dumps(token_count(total=42000, cached=35000)),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def assert_openapi_documents_context_used() -> None:
    """Sanity-check the live spec file the daemon was built from — the
    drift-guard unit test is authoritative, but it costs nothing to
    re-confirm the new field is documented before declaring the smoke
    green."""
    spec_path = Path(__file__).resolve().parent.parent / "specs" / "openapi.yaml"
    text = spec_path.read_text(encoding="utf-8")
    if "context_used:" not in text:
        raise RuntimeError(
            f"openapi.yaml does not document context_used: {spec_path}"
        )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="context-used-smoke-"))
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

    assert_openapi_documents_context_used()

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
    rc = 0
    body_failed = False
    warning_lines: list[str] = []
    try:
        wait_for_health()
        print(f"[smoke] harness up on :{PORT}")

        rollout_uuid = "22222222-2222-2222-2222-222222222222"
        rollout = write_codex_rollout_with_growing_snapshots(
            sessions_root=sessions_root,
            rollout_uuid=rollout_uuid,
            cwd=str(tmp / "project"),
        )
        print(f"[smoke] dropped multi-turn rollout: {rollout}")

        deadline = time.monotonic() + 6.0
        external_id = f"codex_{rollout_uuid}"
        external_seen = False
        ids: list[str] = []
        while time.monotonic() < deadline:
            _, sessions_list = http_json("GET", f"{BASE_URL}/v1/sessions")
            ids = [s["id"] for s in sessions_list["data"]]
            if external_id in ids:
                external_seen = True
                break
            time.sleep(0.2)
        if not external_seen:
            print(f"[smoke] FAIL: observer did not register the rollout: {ids}")
            body_failed = True
            rc = 2
        else:
            print(f"[smoke] OK: observer registered external session {external_id}")
            # GET on the session must succeed and the response must
            # accept the new optional ``context_used`` field. Since
            # the external session has no harness Run, the field stays
            # null (the materializer's run.usage branch is gated on
            # run_id); the goal here is API-shape compatibility, not
            # the cumulative-vs-snapshot proof (that lives in the
            # observer integration test).
            status, ext = http_json("GET", f"{BASE_URL}/v1/sessions/{external_id}")
            if status != 200:
                print(f"[smoke] FAIL: GET external session: status={status} body={ext}")
                body_failed = True
                rc = 4
            else:
                stats = ext.get("stats") or {}
                if "context_used" not in stats:
                    print(
                        f"[smoke] FAIL: stats payload missing context_used "
                        f"key: stats={stats}"
                    )
                    body_failed = True
                    rc = 5
                else:
                    print(
                        f"[smoke] OK: stats.context_used present "
                        f"(value={stats['context_used']!r})"
                    )
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
                warning_lines = [
                    line for line in tail.splitlines()
                    if "Unsupported Codex transcript shape" in line
                ]
                if warning_lines:
                    print("[smoke] FAIL: unexpected unsupported-shape warning(s):")
                    for line in warning_lines:
                        print(f"  {line}")
                    rc = 3
                print("[smoke] --- harness tail (last 2 KB) ---")
                print(tail[-2000:])
        shutil.rmtree(tmp, ignore_errors=True)
    if rc == 0 and not body_failed and not warning_lines:
        print("[smoke] ALL OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
