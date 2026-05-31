#!/usr/bin/env python3
"""Sidecar smoke test for codex harness-origin session resume.

What this exercises end-to-end vs unit tests:

- Daemon boots cleanly with the new field plumbed through OpenAPI,
  models, observer emission, materializer, and the unified
  CodexCommandBuilder branch.
- The observer's Option A backfill populates ``codex_resume_id`` on
  existing external-origin codex sessions after a restart, derived
  from the ``codex_<uuid>`` id prefix.
- ``GET /v1/sessions/<id>`` surfaces ``codex_resume_id`` over the
  wire — the field is part of the public Session schema.

What this does NOT exercise:

- The harness binding path (orchestrator → expect_codex_rollout →
  observer match → session.updated with codex_resume_id). That path
  is unit-test-only (no codex CLI in the sandbox); the canonical
  multi-turn PINEAPPLE proof lives in
  ``tests/test_observer.py::test_codex_multi_turn_resumes_after_observer_binding_pineapple``
  which drives the full observer + builder chain through
  InMemoryRepository.

Spec: specs/2026-05-21-codex-resume.md
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

ROLLOUT_UUID = "019e0bb0-0000-0000-0000-000000000000"
EXTERNAL_SESSION_ID = f"codex_{ROLLOUT_UUID}"


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


def write_codex_rollout(*, sessions_root: Path, cwd: str) -> Path:
    transcript_dir = sessions_root / "2026" / "05" / "21"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript = transcript_dir / f"rollout-2026-05-21T10-30-00-{ROLLOUT_UUID}.jsonl"
    lines = [
        json.dumps({
            "timestamp": "2026-05-21T10:30:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": ROLLOUT_UUID,
                "timestamp": "2026-05-21T10:30:00.000Z",
                "cwd": cwd,
                "originator": "codex_exec",
                "cli_version": "0.128.0",
            },
        }),
        json.dumps({"type": "turn_context", "payload": {"cwd": cwd, "model": "gpt-5.4"}}),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def launch_harness(
    *, env: dict[str, str], db_path: Path, sessions_root: Path, repo_root: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "agent_harness.cli", "serve",
        "--host", HOST,
        "--port", str(PORT),
        "--database", str(db_path),
        "--observe-root", str(sessions_root),
    ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(repo_root),
    )


def stop_harness(proc: subprocess.Popen) -> str:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    if proc.stdout is None:
        return ""
    return proc.stdout.read().decode("utf-8", errors="replace")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="codex-resume-smoke-"))
    print(f"[smoke] tmp dir: {tmp}")
    fake_home = tmp / "home"
    fake_home.mkdir()
    sessions_root = fake_home / ".codex" / "sessions"
    sessions_root.mkdir(parents=True)
    db_path = tmp / "harness.db"
    project_dir = tmp / "project"
    project_dir.mkdir()

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["PYTHONPATH"] = str(repo_root / "src")

    # --- Phase 1: boot harness, drop a rollout, observer registers
    # the external codex session. The session is materialized with
    # codex_resume_id=None at this stage because the observer
    # constructs a fresh Session payload from session_meta and only
    # the backfill (on the NEXT startup) derives the UUID.

    proc = launch_harness(
        env=env, db_path=db_path, sessions_root=sessions_root, repo_root=repo_root,
    )
    phase1_tail = ""
    try:
        wait_for_health()
        print(f"[smoke] phase 1: harness up on :{PORT}")

        rollout = write_codex_rollout(
            sessions_root=sessions_root, cwd=str(project_dir),
        )
        print(f"[smoke] dropped codex rollout: {rollout}")

        deadline = time.monotonic() + 6.0
        seen_external = False
        while time.monotonic() < deadline:
            _, listing = http_json("GET", f"{BASE_URL}/v1/sessions")
            ids = [s["id"] for s in listing["data"]]
            if EXTERNAL_SESSION_ID in ids:
                seen_external = True
                break
            time.sleep(0.2)
        if not seen_external:
            print(f"[smoke] FAIL: observer did not register external session")
            return 2
        print(f"[smoke] OK: observer registered {EXTERNAL_SESSION_ID}")

        # The observer's session.updated payload has the field set
        # to None (fresh Session() default). The backfill won't fire
        # here because it ran at construction time, BEFORE this
        # session existed.
        _, ext_phase1 = http_json("GET", f"{BASE_URL}/v1/sessions/{EXTERNAL_SESSION_ID}")
        cri_phase1 = ext_phase1.get("codex_resume_id")
        if cri_phase1 is not None:
            print(
                f"[smoke] WARN: codex_resume_id already set in phase 1 "
                f"(unexpected but not a failure): {cri_phase1!r}"
            )
    finally:
        phase1_tail = stop_harness(proc)

    # --- Phase 2: restart harness against the SAME DB. The backfill
    # now sees the external codex session with codex_resume_id=None
    # and a ``codex_<uuid>`` id; it patches the field. GET reflects
    # the change immediately.

    proc = launch_harness(
        env=env, db_path=db_path, sessions_root=sessions_root, repo_root=repo_root,
    )
    phase2_tail = ""
    rc = 0
    try:
        wait_for_health()
        print(f"[smoke] phase 2: restarted; backfill should have run")

        _, ext_phase2 = http_json("GET", f"{BASE_URL}/v1/sessions/{EXTERNAL_SESSION_ID}")
        cri_phase2 = ext_phase2.get("codex_resume_id")
        if cri_phase2 != ROLLOUT_UUID:
            print(
                f"[smoke] FAIL: codex_resume_id not backfilled. "
                f"got {cri_phase2!r}, expected {ROLLOUT_UUID!r}"
            )
            rc = 3
        else:
            print(
                f"[smoke] OK: backfill populated codex_resume_id="
                f"{cri_phase2!r} on restart"
            )
    finally:
        phase2_tail = stop_harness(proc)

    print("[smoke] --- phase 1 tail (last 1 KB) ---")
    print(phase1_tail[-1000:])
    print("[smoke] --- phase 2 tail (last 1 KB) ---")
    print(phase2_tail[-1000:])
    shutil.rmtree(tmp, ignore_errors=True)
    if rc == 0:
        print("[smoke] ALL OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
