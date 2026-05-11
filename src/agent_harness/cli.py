from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from agent_harness.api import create_app
from agent_harness.events import InMemoryEventBus
from agent_harness.orchestrator import RunManager
from agent_harness.settings import ObserverSettings
from agent_harness.storage import open_sqlite_repository


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the agent-harness HTTP/SSE service.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--database", help="SQLite database path for durable repository state.")
    serve.add_argument(
        "--execute-runs",
        action="store_true",
        help="Launch harness-owned runs with real backend subprocess adapters.",
    )
    serve.add_argument(
        "--observe-root",
        action="append",
        default=[],
        help="Transcript root to observe. Repeat to watch multiple roots.",
    )
    serve.add_argument(
        "--observe-default-roots",
        action="store_true",
        help="Observe Claude Code and Codex transcript roots under HOME.",
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        observer_settings = _observer_settings_from_args(args)
        app = _app_for_serve(args, observer_settings)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
        return 0

    parser.error(f"unsupported command: {args.command}")


def _observer_settings_from_args(args: argparse.Namespace) -> ObserverSettings:
    settings = ObserverSettings.from_roots(args.observe_root)
    if args.observe_default_roots:
        default_settings = ObserverSettings.default_transcript_roots()
        settings = ObserverSettings.from_roots([*settings.roots, *default_settings.roots])
    return settings


def _app_for_serve(args: argparse.Namespace, observer_settings: ObserverSettings):
    if not observer_settings.enabled and not args.database and not args.execute_runs:
        return "agent_harness.api:app"

    kwargs = {"observer_settings": observer_settings}
    repository = open_sqlite_repository(args.database) if args.database else None
    if repository is not None:
        kwargs["repository"] = repository
    if args.execute_runs:
        event_bus = InMemoryEventBus()
        kwargs["event_bus"] = event_bus
        kwargs["run_manager"] = RunManager(event_bus=event_bus)
    return create_app(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
