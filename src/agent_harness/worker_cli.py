"""``agent-harness worker …`` and ``agent-harness up/down``.

The worker calls are thin HTTP clients (stdlib only, so any agent's shell can
run them) over the harness mailbox API:

* ``spawn``  — create a fresh harness session and send it the brief. The
  harness starts the first turn; the call returns the new address at once.
* ``send``   — store a message for an address. A turn-based (harness-origin)
  recipient is resumed with it; an interactive (external) one keeps it in its
  inbox.
* ``inbox``  — read (and by default acknowledge) unread stored messages.

``up``/``down`` are for a task launcher: reuse a running loopback harness or
start one, and on close wait for pending deliveries, then stop only a harness
this launcher started.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from agent_harness.messaging import ENV_URL, harness_url, self_address

LOOPBACK = "127.0.0.1"


class HarnessError(RuntimeError):
    pass


# --- HTTP ------------------------------------------------------------------


def _request(url: str, method: str = "GET", body: Any = None, *, timeout: float = 15.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise HarnessError(f"{method} {url}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise HarnessError(f"cannot reach agent-harness at {url}: {exc}") from exc
    return json.loads(payload) if payload else None


def _quote(session_id: str) -> str:
    return urllib.parse.quote(session_id, safe="")


# --- worker calls ------------------------------------------------------------


def _read_text(value: str | None, file: str | None, what: str) -> str:
    if file is not None:
        text = sys.stdin.read() if file == "-" else Path(file).read_text()
    elif value == "-":
        text = sys.stdin.read()
    else:
        text = value or ""
    if not text.strip():
        raise HarnessError(f"{what} is empty")
    return text


def cmd_spawn(args: argparse.Namespace) -> dict[str, Any]:
    base = args.url or harness_url()
    brief = _read_text(args.brief, args.brief_file, "brief")
    cwd = str(Path(args.cwd or os.getcwd()).resolve())
    session_body: dict[str, Any] = {
        "backend": args.backend,
        "project": {"path": cwd, "name": Path(cwd).name or cwd},
        # A headless worker cannot answer permission prompts.
        "bypass_permissions": not args.no_bypass,
    }
    for key in ("model", "effort", "title"):
        if getattr(args, key):
            session_body[key] = getattr(args, key)
    session = _request(f"{base}/v1/sessions", "POST", session_body)
    sender = args.sender or self_address()
    sent = _send(base, session["id"], brief, sender)
    return {
        "address": session["id"],
        "backend": session["backend"],
        "status": sent["status"],
        "run_id": sent.get("run_id"),
        "reply_to": sender,
    }


def _send(base: str, address: str, text: str, sender: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"body": text}
    if sender:
        body["sender"] = sender
    return _request(f"{base}/v1/sessions/{_quote(address)}/mailbox", "POST", body)


def cmd_send(args: argparse.Namespace) -> dict[str, Any]:
    base = args.url or harness_url()
    text = _read_text(args.text, args.file, "message")
    sender = args.sender or self_address()
    sent = _send(base, args.address, text, sender)
    return {
        "message_id": sent["message"]["id"],
        "to": args.address,
        "status": sent["status"],
        "run_id": sent.get("run_id"),
    }


def cmd_inbox(args: argparse.Namespace) -> dict[str, Any]:
    base = args.url or harness_url()
    me = args.sender or self_address()
    if not me:
        raise HarnessError(
            "cannot tell which session you are; pass --as <address> "
            "(harness runs set AGENT_HARNESS_SESSION, Claude Code sets CLAUDE_CODE_SESSION_ID)"
        )
    query = "" if args.all else "?state=unread"
    mail = _request(f"{base}/v1/sessions/{_quote(me)}/mailbox{query}")["data"]
    if not args.all and not args.peek and mail:
        _request(
            f"{base}/v1/sessions/{_quote(me)}/mailbox/ack", "POST", {"ids": [m["id"] for m in mail]}
        )
    return {"address": me, "messages": mail}


def _print_inbox(result: dict[str, Any]) -> None:
    mail = result["messages"]
    if not mail:
        print(f"No messages for {result['address']}.")
        return
    for item in mail:
        print(f"--- {item['id']} from {item.get('sender') or 'anonymous'} at {item['created_at']} [{item['state']}]")
        print(item["body"].rstrip())
        print()


def cmd_whoami(args: argparse.Namespace) -> dict[str, Any]:
    return {"address": args.sender or self_address(), "url": args.url or harness_url()}


# --- launcher ----------------------------------------------------------------


def _state_dir(args: argparse.Namespace) -> Path:
    if args.state_dir:
        return Path(args.state_dir).expanduser()
    root = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(root) / "agent-harness"


def _state_file(args: argparse.Namespace) -> Path:
    return _state_dir(args) / f"launcher-{args.port}.json"


def _is_harness(url: str) -> bool:
    try:
        backends = _request(f"{url}/v1/backends", timeout=2.0)
    except HarnessError:
        return False
    return isinstance(backends, dict) and "data" in backends


def _pid_matches(pid: int, port: int) -> bool:
    """True when ``pid`` is still the harness we started (guards PID reuse)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True  # no /proc (macOS): trust the recorded pid
    words = [w.decode(errors="replace") for w in cmdline]
    return "serve" in words and str(port) in words and any("agent" in w for w in words)


def cmd_up(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{LOOPBACK}:{args.port}"
    state_file = _state_file(args)
    if _is_harness(url):
        return {"url": url, "started": False, "pid": None, "env": {ENV_URL: url}}

    state_dir = _state_dir(args)
    state_dir.mkdir(parents=True, exist_ok=True)
    database = args.database or str(state_dir / f"harness-{args.port}.db")
    log_path = state_dir / f"harness-{args.port}.log"
    argv = [
        sys.executable, "-m", "agent_harness.cli", "serve",
        "--host", LOOPBACK, "--port", str(args.port),
        "--execute-runs", "--database", database,
        *args.serve_arg,
    ]
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HarnessError(f"harness exited with {proc.returncode}; see {log_path}")
        if _is_harness(url):
            break
        time.sleep(0.2)
    else:
        proc.terminate()
        raise HarnessError(f"harness did not become ready within {args.wait}s; see {log_path}")
    state = {"url": url, "pid": proc.pid, "port": args.port, "task": args.task,
             "database": database, "log": str(log_path)}
    state_file.write_text(json.dumps(state))
    return {"url": url, "started": True, "pid": proc.pid, "database": database,
            "env": {ENV_URL: url}}


def _wait_drained(url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        status = _request(f"{url}/v1/mailbox/status")
        if status["pending"] == 0 and status["active_runs"] == 0:
            return status
        if time.monotonic() >= deadline:
            return status
        time.sleep(1.0)


def cmd_down(args: argparse.Namespace) -> dict[str, Any]:
    state_file = _state_file(args)
    url = f"http://{LOOPBACK}:{args.port}"
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = None
    if state is None or state.get("task") != args.task:
        # Reused (or someone else's) harness: it keeps running and keeps
        # delivering, so there is nothing for this launcher to stop.
        return {"url": url, "stopped": False, "reason": "not started by this launcher"}

    pid = int(state["pid"])
    if not _pid_matches(pid, args.port):
        state_file.unlink(missing_ok=True)
        return {"url": url, "stopped": False, "reason": "recorded harness is no longer running"}

    status = _wait_drained(url, args.timeout) if _is_harness(url) else {"pending": 0, "active_runs": 0}
    if (status["pending"] or status["active_runs"]) and not args.force:
        raise HarnessError(
            f"harness still busy after {args.timeout}s (pending={status['pending']}, "
            f"active_runs={status['active_runs']}); rerun down, or pass --force"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
    state_file.unlink(missing_ok=True)
    return {"url": url, "stopped": True, "pid": pid, "drained": status}


def _pid_alive(pid: int) -> bool:
    try:
        # Reap if it is our child (tests); harmless otherwise.
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --- argparse wiring -----------------------------------------------------------


def add_worker_parsers(subparsers: argparse._SubParsersAction) -> None:
    worker = subparsers.add_parser(
        "worker",
        help="Spawn workers and exchange messages through a local harness.",
        description=(
            "Messages to a harness-launched (turn-based) session resume it in a new "
            "process; messages to an observed interactive session are only stored "
            "and read with 'inbox'."
        ),
    )
    wsub = worker.add_subparsers(dest="worker_command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--url", help=f"Harness base URL (default: ${ENV_URL} or http://127.0.0.1:8000).")
        p.add_argument("--as", dest="sender", help="Your own address (default: from the environment).")

    spawn = wsub.add_parser("spawn", help="Start a new worker in a fresh session; prints its address.")
    common(spawn)
    spawn.add_argument("--backend", required=True, choices=["claude-code", "codex", "pi"])
    spawn.add_argument("--model")
    spawn.add_argument("--effort")
    spawn.add_argument("--cwd", help="Working directory (default: current).")
    spawn.add_argument("--title")
    spawn.add_argument("--no-bypass-permissions", dest="no_bypass", action="store_true")
    spawn.add_argument("brief", nargs="?", help="Brief text, or '-' for stdin.")
    spawn.add_argument("--brief-file", help="Read the brief from this file ('-' for stdin).")
    spawn.set_defaults(handler=cmd_spawn)

    send = wsub.add_parser("send", help="Send a message to an address; returns once accepted.")
    common(send)
    send.add_argument("address")
    send.add_argument("text", nargs="?", help="Message text, or '-' for stdin.")
    send.add_argument("--file", help="Read the message from this file ('-' for stdin).")
    send.set_defaults(handler=cmd_send)

    inbox = wsub.add_parser("inbox", help="Read unread messages stored for you.")
    common(inbox)
    inbox.add_argument("--all", action="store_true", help="List every message, read or not; marks nothing.")
    inbox.add_argument("--peek", action="store_true", help="Do not mark the listed messages read.")
    inbox.add_argument("--json", action="store_true")
    inbox.set_defaults(handler=cmd_inbox)

    whoami = wsub.add_parser("whoami", help="Print your own address and harness URL.")
    common(whoami)
    whoami.set_defaults(handler=cmd_whoami)

    for name, handler, helptext in (
        ("up", cmd_up, "Reuse a running loopback harness on --port, or start one."),
        ("down", cmd_down, "Deliver pending messages, then stop the harness only if 'up' started it."),
    ):
        p = subparsers.add_parser(name, help=helptext)
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--state-dir", help="Launcher state (default: $XDG_STATE_HOME/agent-harness).")
        p.add_argument("--task", default="default", help="Launcher identity; down stops only what up --task started.")
        p.set_defaults(handler=handler)
        if name == "up":
            p.add_argument("--database", help="SQLite path (default: <state-dir>/harness-<port>.db).")
            p.add_argument("--wait", type=float, default=20.0, help="Seconds to wait for readiness.")
            p.add_argument("--serve-arg", action="append", default=[],
                           help="Extra argument for 'serve' (repeatable), e.g. --serve-arg=--no-observer.")
        else:
            p.add_argument("--timeout", type=float, default=600.0, help="Seconds to wait for delivery.")
            p.add_argument("--force", action="store_true", help="Stop even if deliveries are still running.")


def run_handler(args: argparse.Namespace) -> int:
    try:
        result = args.handler(args)
    except HarnessError as exc:
        print(f"agent-harness: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "worker_command", None) == "inbox" and not args.json:
        _print_inbox(result)
    else:
        print(json.dumps(result))
    return 0
