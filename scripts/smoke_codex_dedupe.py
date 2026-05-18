#!/usr/bin/env python3
"""Sidecar smoke test for codex rollout dedupe (spec 2026-05-18).

Spins up a harness on a non-default port + temp DB, with the observer
restricted to a fake transcript root under $HOME, then:

1. POST /v1/sessions with backend=codex, project.path=<tmp project dir>.
2. Drop a fixture codex rollout whose session_meta.cwd matches and whose
   timestamp lies within the 30s reconcile window.
3. Wait for the observer's watchfiles loop to pick it up.
4. Assert: GET /v1/sessions returns exactly ONE session row (the harness
   one), with origin=harness, codex_internal_id set, and the rollout's
   user message attached.

Also runs a negative case: a rollout under a DIFFERENT cwd is observed
and verified to register as a new external `codex_<uuid>` row.

Exits 0 on success; non-zero with a diagnostic on any failure.
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PORT = 8879
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"


def http_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict | list]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            payload = {"raw": body_text}
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
    raise RuntimeError(f"harness on :{PORT} did not become healthy in 10s (last: {last_err})")


def write_rollout(
    *,
    sessions_root: Path,
    rollout_uuid: str,
    cwd: str,
    session_meta_ts: datetime,
) -> Path:
    transcript_dir = sessions_root / "2026" / "05" / "18"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript = transcript_dir / f"rollout-2026-05-18T10-30-00-{rollout_uuid}.jsonl"
    ts_iso = session_meta_ts.isoformat().replace("+00:00", "Z")
    lines = [
        json.dumps({
            "timestamp": ts_iso,
            "type": "session_meta",
            "payload": {
                "id": rollout_uuid,
                "timestamp": ts_iso,
                "cwd": cwd,
                "originator": "codex_exec",
                "cli_version": "0.128.0",
            },
        }),
        json.dumps({
            "type": "turn_context",
            "payload": {"cwd": cwd, "model": "gpt-5.4"},
        }),
        json.dumps({
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "hello from rollout"},
        }),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="codex-dedupe-smoke-"))
    print(f"[smoke] tmp dir: {tmp}")
    fake_home = tmp / "home"
    fake_home.mkdir()
    sessions_root = fake_home / ".codex" / "sessions"
    sessions_root.mkdir(parents=True)
    db_path = tmp / "harness.db"
    project_dir_match = tmp / "project"
    project_dir_match.mkdir()
    project_dir_mismatch = tmp / "other-project"
    project_dir_mismatch.mkdir()

    repo_root = Path(__file__).resolve().parent
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

        # --- Positive case: matching cwd within window ---
        status, harness_session = http_json(
            "POST", f"{BASE_URL}/v1/sessions",
            {
                "backend": "codex",
                "model": "gpt-5.4",
                "project": {"path": str(project_dir_match), "name": "project"},
                "title": "smoke-test-codex",
            },
        )
        assert status in (200, 201), (status, harness_session)
        assert harness_session["origin"] == "harness", harness_session
        assert harness_session["codex_internal_id"] is None, harness_session
        harness_id = harness_session["id"]
        created_at = datetime.fromisoformat(
            harness_session["created_at"].replace("Z", "+00:00")
        )
        print(f"[smoke] harness session created: {harness_id}")

        # Drop a rollout whose session_meta timestamp is 5s after created_at
        # (well within the 30s window).
        rollout_uuid = "11111111-1111-1111-1111-111111111111"
        rollout = write_rollout(
            sessions_root=sessions_root,
            rollout_uuid=rollout_uuid,
            cwd=str(project_dir_match),
            session_meta_ts=created_at + timedelta(seconds=5),
        )
        print(f"[smoke] dropped rollout: {rollout}")

        # Give watchfiles + the observer time to ingest. Poll up to 8s.
        deadline = time.monotonic() + 8.0
        rebound = False
        while time.monotonic() < deadline:
            _, session = http_json("GET", f"{BASE_URL}/v1/sessions/{harness_id}")
            if session.get("codex_internal_id") == rollout_uuid:
                rebound = True
                break
            time.sleep(0.2)
        if not rebound:
            print(f"[smoke] FAIL: harness session did not rebind to rollout: {session}")
            return 2
        print(f"[smoke] OK: harness session bound to rollout {rollout_uuid}")

        # No phantom codex_<uuid> row.
        _, sessions_list = http_json("GET", f"{BASE_URL}/v1/sessions")
        ids = [s["id"] for s in sessions_list["data"]]
        if f"codex_{rollout_uuid}" in ids:
            print(f"[smoke] FAIL: phantom codex_<uuid> row present: {ids}")
            return 3
        print(f"[smoke] OK: no phantom row (sessions: {ids})")

        # The user message attached to the harness session.
        _, msgs = http_json("GET", f"{BASE_URL}/v1/sessions/{harness_id}/messages")
        msg_texts = [
            block.get("text")
            for msg in msgs["data"]
            for block in msg.get("blocks", [])
            if block.get("type") == "text"
        ]
        if "hello from rollout" not in msg_texts:
            print(f"[smoke] FAIL: rollout message missing from harness session: {msg_texts}")
            return 4
        print(f"[smoke] OK: rollout message attached to harness session")

        # --- Negative case: rollout under a different cwd → external row ---
        rollout_uuid_2 = "22222222-2222-2222-2222-222222222222"
        rollout_2 = write_rollout(
            sessions_root=sessions_root,
            rollout_uuid=rollout_uuid_2,
            cwd="/some/totally/unrelated/path",
            session_meta_ts=datetime.now(UTC),
        )
        print(f"[smoke] dropped unrelated rollout: {rollout_2}")

        deadline = time.monotonic() + 8.0
        external_seen = False
        while time.monotonic() < deadline:
            _, sessions_list = http_json("GET", f"{BASE_URL}/v1/sessions")
            ids = [s["id"] for s in sessions_list["data"]]
            if f"codex_{rollout_uuid_2}" in ids:
                external_seen = True
                break
            time.sleep(0.2)
        if not external_seen:
            print(f"[smoke] FAIL: unrelated rollout did not register as external row: {ids}")
            return 5
        print(f"[smoke] OK: unrelated rollout registered as codex_<uuid> external row")

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
            tail = proc.stdout.read().decode("utf-8", errors="replace")[-2000:]
            if tail:
                print("[smoke] --- harness tail ---")
                print(tail)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
