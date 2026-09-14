"""End-to-end acceptance probe for external pi session continuation.

Exercises the real stack — the actual ``pi`` CLI, the filesystem watcher,
SQLite, the HTTP API, the SSE stream and the RunManager — rather than the
doubles the unit suite uses. It asserts the whole contract in one pass:

  real pi writes a transcript
    -> the watcher discovers it as an origin=external pi session
    -> GET /v1/sessions and /messages expose it, SSE replays it
    -> POST /v1/runs continues THE SAME conversation headlessly
       (prior user + assistant turns reach the provider; the transcript is
        appended to, not rewritten; no second transcript appears)
    -> the run completes as a harness-origin run
    -> with the transcript moved away, the next POST /v1/runs 409s
       instead of silently starting a new conversation

Safe to run anywhere: it talks to a loopback stub provider with a
generated throwaway token, and points ``PI_CODING_AGENT_DIR`` at a
scratch directory, so it never reads real credentials, never reaches the
network, and never touches a personal pi transcript.

Written by the independent reviewer of this feature and kept in-tree
because the unit suite cannot cover CLI behaviour. Slow (~30s) and
dependent on a working ``pi`` on PATH, so it is a script rather than a
pytest case.

    uv run python scripts/pi_external_acceptance_probe.py

Spec: specs/2026-09-14-external-pi-sessions.md
"""

import json, os, subprocess, threading, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile

ROOT = Path(tempfile.mkdtemp(prefix="harness-probe-", dir="/tmp"))
CONFIG = ROOT / ".pi" / "agent"
CONFIG.mkdir(parents=True)
CWD = ROOT / "project"
CWD.mkdir()
requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta, finish in [
            ({"role": "assistant", "content": f"PROBE_REPLY_{len(requests)}"}, None),
            ({}, "stop"),
        ]:
            chunk = {
                "id": "probe",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "probe",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
# A generated token is confined to this loopback fixture; no user credentials.
(CONFIG / "models.json").write_text(
    json.dumps(
        {
            "providers": {
                "review-probe": {
                    "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
                    "api": "openai-completions",
                    "apiKey": uuid.uuid4().hex,
                    "models": [
                        {
                            "id": "probe",
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 32000,
                            "maxTokens": 1000,
                        }
                    ],
                }
            }
        }
    )
)
ENV = {
    k: v
    for k, v in os.environ.items()
    if not any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
}
ENV.update(PI_CODING_AGENT_DIR=str(CONFIG), PI_OFFLINE="1", PI_TELEMETRY="0")
BASE = [
    "pi",
    "--offline",
    "-ne",
    "-ns",
    "-np",
    "--no-themes",
    "-nc",
    "-nt",
    "--provider",
    "review-probe",
    "--model",
    "probe",
]
sid = str(uuid.uuid4())


def run(args, prompt):
    # stdin=DEVNULL: inherited stdin makes pi block indefinitely in some
    # non-interactive contexts even with -p.
    p = subprocess.run(
        BASE + ["-p"] + args + [prompt],
        cwd=CWD,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=40,
        stdin=subprocess.DEVNULL,
    )
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout)
    assert "PROBE_REPLY_" in p.stdout, p.stdout


import socket, time
import httpx, uvicorn
from agent_harness.api import create_app
from agent_harness.events import DurableEventBus
from agent_harness.storage import open_sqlite_repository
from agent_harness.orchestrator import RunManager
from agent_harness.settings import ObserverSettings

run(["--session-id", sid], "Remember FIRST_PROBE_CONTEXT")
transcript = next(CONFIG.glob("sessions/**/*.jsonl"))
original = transcript.read_bytes()
# Only the isolated config is visible to spawned pi. Personal config stays untouched.
os.environ["PI_CODING_AGENT_DIR"] = str(CONFIG)
os.environ["PI_OFFLINE"] = "1"
os.environ["PI_TELEMETRY"] = "0"
repo = open_sqlite_repository(ROOT / "harness.sqlite")
bus = DurableEventBus(repo)
manager = RunManager(event_bus=bus, idle_timeout_seconds=20, max_run_seconds=30)
app = create_app(
    repository=repo,
    event_bus=bus,
    run_manager=manager,
    observer_settings=ObserverSettings.from_roots([CONFIG / "sessions"]),
)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
uvi = uvicorn.Server(uvicorn.Config(app, log_level="error"))
th = threading.Thread(target=lambda: uvi.run(sockets=[sock]), daemon=True)
th.start()
client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10)
session_id = "ses_" + sid.replace("-", "")


# 8s (the reviewer's original bound) proved tight under concurrent load —
# one run timed out waiting for the watcher to publish the session. The
# wait is a bound on flakiness, not part of the contract, so it costs
# nothing to be generous.
def eventually(fn, seconds=25):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            result = fn()
            if result:
                return result
        except (httpx.ConnectError, KeyError):
            pass
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for acceptance condition")


try:
    eventually(lambda: uvi.started)
    time.sleep(0.3)
    os.utime(transcript, None)  # Trigger a filesystem event, as a live pi turn would.
    sessions = eventually(lambda: client.get("/v1/sessions").json()["data"])
    session = next(s for s in sessions if s["id"] == session_id)
    assert session["backend"] == "pi" and session["origin"] == "external", session
    assert session["project"]["path"] == str(CWD)
    messages = client.get(f"/v1/sessions/{session_id}/messages").json()["data"]
    assert len(messages) == 2, messages
    observed = []
    with client.stream(
        "GET", f"/v1/sessions/{session_id}/events?from=beginning"
    ) as response:
        for line in response.iter_lines():
            if line.startswith("event: "):
                observed.append(line[7:])
            if observed.count("message") >= 2:
                break
    assert "session.updated" in observed, observed
    result = client.post(
        f"/v1/sessions/{session_id}/runs", json={"message": "HEADLESS_API_CONTINUATION"}
    )
    assert result.status_code == 202, result.text
    rid = result.json()["run_id"]

    def finished():
        run = client.get(f"/v1/sessions/{session_id}/runs/{rid}").json()
        return (
            run if run.get("status") in ("completed", "failed", "interrupted") else None
        )

    final = eventually(finished, 30)
    assert final["status"] == "completed", final
    assert final["origin"] == "harness", final
    assert len(requests) == 2, requests
    assert "FIRST_PROBE_CONTEXT" in json.dumps(requests[1]["messages"])
    assert "PROBE_REPLY_1" in json.dumps(requests[1]["messages"])
    assert transcript.read_bytes().startswith(original)
    assert len(list(CONFIG.glob("sessions/**/*.jsonl"))) == 1
    msgs = eventually(
        lambda: (
            m
            if len(
                m := client.get(f"/v1/sessions/{session_id}/messages").json()["data"]
            )
            >= 4
            else None
        )
    )
    assert any("PROBE_REPLY_2" in json.dumps(m) for m in msgs), msgs
    transcript.rename(transcript.with_suffix(".saved"))
    rejected = client.post(
        f"/v1/sessions/{session_id}/runs",
        json={"message": "MUST_NOT_CREATE_NEW_SESSION"},
    )
    assert rejected.status_code == 409, rejected.text
    assert len(requests) == 2
    print(
        "PASS: real pi -> watcher -> SQLite -> HTTP list/messages/SSE -> POST runs -> same transcript/context -> completed harness run -> missing transcript 409"
    )
    print("Fixture:", ROOT)
finally:
    client.close()
    uvi.should_exit = True
    th.join(timeout=8)
    server.shutdown()
