#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8876"
DEFAULT_ROOT = "/tmp/agent-harness-observer-demo"
DEMO_ROLLOUT_UUID = "123e4567-e89b-12d3-a456-426614174000"
DEMO_TIMESTAMP = "2026-05-08T10-30-00"
DEMO_SESSION_ID = f"codex_{DEMO_ROLLOUT_UUID}"


def codex_demo_transcript_path(root: str | Path) -> Path:
    return (
        Path(root)
        / ".codex"
        / "sessions"
        / "2026"
        / "05"
        / "08"
        / f"rollout-{DEMO_TIMESTAMP}-{DEMO_ROLLOUT_UUID}.jsonl"
    )


def demo_transcript_lines(*, cwd: str, model: str) -> list[str]:
    records = [
        {"type": "turn_context", "payload": {"cwd": cwd, "model": model}},
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "role": "user",
                "content": "Observe this external user message",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "assistant_message",
                "role": "assistant",
                "content": "Observed external assistant response",
            },
        },
    ]
    return [json.dumps(record, separators=(",", ":")) for record in records]


def append_demo_transcript(*, root: str | Path, cwd: str, model: str, delay: float) -> Path:
    transcript = codex_demo_transcript_path(root)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("a", encoding="utf-8") as handle:
        for line in demo_transcript_lines(cwd=cwd, model=model):
            handle.write(line + "\n")
            handle.flush()
            time.sleep(delay)
    return transcript


def get_json(url: str) -> dict[str, Any]:
    with request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_observed_session(*, base_url: str, session_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return get_json(f"{base_url.rstrip('/')}/v1/sessions/{session_id}")
        except error.HTTPError as exc:
            if exc.code != 404:
                raise
            last_error = exc
        except error.URLError as exc:
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"session {session_id} was not observed before timeout") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Demonstrate external transcript observation with a fake Codex JSONL.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cwd", default="/workspace/observer-demo")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--write-only", action="store_true", help="Only write the transcript; do not call the API.")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    transcript = append_demo_transcript(root=root, cwd=args.cwd, model=args.model, delay=args.delay)
    print(f"Wrote demo transcript: {transcript}")

    if args.write_only:
        return 0

    session = wait_for_observed_session(base_url=args.base_url, session_id=DEMO_SESSION_ID, timeout=args.timeout)
    messages = get_json(f"{args.base_url.rstrip('/')}/v1/sessions/{DEMO_SESSION_ID}/messages")
    print(json.dumps({"session": session, "messages": messages["data"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
