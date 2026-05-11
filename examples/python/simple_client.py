#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import request


DEFAULT_BASE_URL = "http://127.0.0.1:8876"


def create_session_payload(*, backend: str, model: str, project_path: str, title: str | None) -> dict[str, Any]:
    path = Path(project_path)
    return {
        "backend": backend,
        "model": model,
        "project": {"path": project_path, "name": path.name or project_path},
        "title": title,
    }


def json_request(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if body is not None else {}
    http_request = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(http_request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an agent-harness session and run through the v1 API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--backend", default="codex", choices=["codex", "claude-code"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--project-path", default="/workspace/demo")
    parser.add_argument("--title", default="Python quickstart")
    parser.add_argument("--message", default="Say hello from the Python quickstart")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    session = json_request(
        "POST",
        f"{base_url}/v1/sessions",
        create_session_payload(
            backend=args.backend,
            model=args.model,
            project_path=args.project_path,
            title=args.title,
        ),
    )
    run = json_request("POST", f"{base_url}/v1/sessions/{session['id']}/runs", {"message": args.message})
    messages = json_request("GET", f"{base_url}/v1/sessions/{session['id']}/messages")

    print(json.dumps({"session": session, "run": run, "messages": messages["data"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
