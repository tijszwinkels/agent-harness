#!/usr/bin/env python3
"""Sidecar smoke test for Phase 1 — rollout discovery + observer pre-binding.

We don't have a real claude/codex CLI handy in this environment, so the
smoke test verifies the *observable side-effects* of Phase 1 via HTTP:

1. Boots a harness on :8879+ with a temp DB and a fake $HOME/.codex
   observe-root. Variant A: ``AGENT_HARNESS_ROLLOUT_PRE_BIND`` unset
   (default off). Variant B: set to "1".

2. For both variants: drops a synthetic codex rollout under the watched
   root with a stable cwd in session_meta. Asserts the harness comes
   up cleanly and the observer ingests the rollout in both cases.

3. With flag ON: also exercises the bind_rollout path directly by
   POSTing a session, then calling the in-process observer's bind via
   a synthetic helper request — we can't here easily because the
   bind_rollout method is in-process, not exposed over HTTP. So we
   only assert the env-var-on variant boots and ingests the rollout
   the same way as the env-var-off variant. The unit + observer tests
   are the substantive coverage; this script just confirms production
   wiring doesn't crash with the new env var set.

Exits 0 on success.
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


HOST = "127.0.0.1"
BASE_URL_TEMPLATE = "http://%s:%d"


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


def wait_for_health(base_url: str) -> None:
    deadline = time.monotonic() + 10.0
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            req = Request(f"{base_url}/health")
            with urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (URLError, HTTPError) as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"harness on {base_url} did not become healthy in 10s (last: {last_err})")


def write_rollout(sessions_root: Path, rollout_uuid: str, cwd: str) -> Path:
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
        json.dumps({"type": "turn_context", "payload": {"cwd": cwd, "model": "gpt-5.4"}}),
        json.dumps({
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "hello"},
        }),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def run_variant(*, port: int, flag_on: bool) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="phase1-smoke-"))
    print(f"[smoke] variant flag_on={flag_on} tmp={tmp}")
    fake_home = tmp / "home"
    fake_home.mkdir()
    sessions_root = fake_home / ".codex" / "sessions"
    sessions_root.mkdir(parents=True)
    db_path = tmp / "harness.db"

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["PYTHONPATH"] = str(repo_root / "src")
    if flag_on:
        env["AGENT_HARNESS_ROLLOUT_PRE_BIND"] = "1"
    else:
        env.pop("AGENT_HARNESS_ROLLOUT_PRE_BIND", None)

    cmd = [
        sys.executable, "-m", "agent_harness.cli", "serve",
        "--host", HOST,
        "--port", str(port),
        "--database", str(db_path),
        "--observe-root", str(sessions_root),
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(repo_root),
    )
    base_url = BASE_URL_TEMPLATE % (HOST, port)
    try:
        wait_for_health(base_url)

        # POST a harness session so the repo has at least one row.
        status, harness_session = http_json(
            "POST", f"{base_url}/v1/sessions",
            {
                "backend": "codex",
                "model": "gpt-5.4",
                "project": {"path": str(tmp / "project"), "name": "project"},
                "title": "phase1-smoke",
            },
        )
        assert status in (200, 201), (status, harness_session)
        print(f"[smoke] harness session: {harness_session['id']}")

        # Drop an unrelated codex rollout — the observer should ingest
        # it as today (filename-pattern → codex_<uuid> external row).
        rollout_uuid = "11111111-1111-2222-1111-111111111111"
        write_rollout(sessions_root, rollout_uuid, cwd=str(tmp / "some-other-cwd"))

        # Poll up to 8s for the observer to register the external row.
        deadline = time.monotonic() + 8.0
        external_seen = False
        while time.monotonic() < deadline:
            _, sessions_list = http_json("GET", f"{base_url}/v1/sessions")
            ids = [s["id"] for s in sessions_list["data"]]
            if f"codex_{rollout_uuid}" in ids:
                external_seen = True
                break
            time.sleep(0.2)
        if not external_seen:
            print(f"[smoke] FAIL: observer did not ingest external rollout: {ids}")
            return 2
        print(f"[smoke] OK: observer ingested external rollout under filename-pattern path")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        if proc.stdout is not None:
            tail = proc.stdout.read().decode("utf-8", errors="replace")[-1500:]
            if tail:
                print("[smoke] --- harness tail ---")
                print(tail)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    # Variant A: flag off (current main behavior).
    rc = run_variant(port=8879, flag_on=False)
    if rc != 0:
        print(f"[smoke] flag-off variant FAILED rc={rc}")
        return rc

    # Variant B: flag on (Phase 1 wiring active; should match A externally).
    rc = run_variant(port=8880, flag_on=True)
    if rc != 0:
        print(f"[smoke] flag-on variant FAILED rc={rc}")
        return rc

    print("[smoke] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
