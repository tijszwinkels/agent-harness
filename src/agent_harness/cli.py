from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import uvicorn

from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.orchestrator import RunManager
from agent_harness.rollout_discovery import RolloutDiscovery
from agent_harness.settings import ObserverSettings
from agent_harness.storage import open_sqlite_repository

logger = logging.getLogger(__name__)

# Hosts that keep the service reachable only from the local machine. Binding
# anything else exposes the API on a network interface.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _warn_if_public_host(host: str) -> None:
    """Log a prominent warning when the harness binds a non-loopback host.

    The API is unauthenticated and spawns agent CLIs with
    ``--dangerously-skip-permissions``; exposing it on a routable interface
    (e.g. ``0.0.0.0`` on a public-IP box) is unauthenticated remote code
    execution. The bridge reaches the harness over localhost, so loopback is
    the correct default for the normal co-located topology.
    """
    if host in _LOOPBACK_HOSTS:
        return
    logger.warning(
        "agent-harness is binding %s, which is NOT loopback. The API is "
        "UNAUTHENTICATED and launches agent CLIs with "
        "--dangerously-skip-permissions; anyone who can reach this port has "
        "remote code execution. Bind 127.0.0.1, or firewall the port and put "
        "an authenticating proxy in front before exposing it.",
        host,
    )


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
        help="Observe Claude Code and Codex transcript roots under HOME. This is the default when those roots exist.",
    )
    serve.add_argument(
        "--no-observer",
        action="store_true",
        help="Disable automatic transcript observation.",
    )
    serve.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allow cross-origin browser access from this origin. Repeat for multiple. Off by default.",
    )
    serve.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Kill a harness-owned run after this many seconds with no "
            "visible activity (stdout/stderr heartbeat or observer "
            "message/tool_use/usage events). A CPU-busy process tree is "
            "deferred at expiry and re-checked. Default: 600. "
            "Only applies with --execute-runs."
        ),
    )
    serve.add_argument(
        "--max-run-seconds",
        type=float,
        default=None,
        help=(
            "Hard per-run wall-clock cap: kill a harness-owned run after "
            "this many seconds even if it is actively working. Bounds a "
            "long silent-but-CPU-busy tool call. Default: 3600. "
            "Only applies with --execute-runs."
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.idle_timeout_seconds is not None and args.idle_timeout_seconds <= 0:
            parser.error("--idle-timeout-seconds must be positive")
        if args.max_run_seconds is not None and args.max_run_seconds <= 0:
            parser.error("--max-run-seconds must be positive")
        _warn_if_public_host(args.host)
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
    if args.no_observer:
        return settings
    if args.observe_default_roots:
        default_settings = ObserverSettings.default_transcript_roots()
        settings = ObserverSettings.from_roots([*settings.roots, *default_settings.roots])
    else:
        default_settings = ObserverSettings.existing_default_transcript_roots()
        settings = ObserverSettings.from_roots([*settings.roots, *default_settings.roots])
    return settings


def _app_for_serve(args: argparse.Namespace, observer_settings: ObserverSettings):
    if (
        not observer_settings.enabled
        and not args.database
        and not args.execute_runs
        and not args.cors_origin
    ):
        # The import-string fast path returns the module-level default app,
        # which cannot carry per-invocation CORS config.
        return "agent_harness.api:app"

    kwargs = {"observer_settings": observer_settings}
    if args.cors_origin:
        kwargs["cors_origins"] = args.cors_origin
    repository = open_sqlite_repository(args.database) if args.database else None
    event_bus = DurableEventBus(repository) if repository is not None else InMemoryEventBus()
    if repository is not None:
        kwargs["repository"] = repository
        kwargs["event_bus"] = event_bus
    if args.execute_runs:
        kwargs["event_bus"] = event_bus
        # The observer is constructed by api.py's lifespan and
        # late-bound onto RunManager via ``set_observer``. A default
        # ``RolloutDiscovery`` is wired here so claude pre-bind (the
        # only RolloutDiscovery consumer post-Phase-2) is reachable
        # once the observer arrives. Codex pre-bind uses the
        # observer's expectation registry directly — no discovery
        # involvement.
        kwargs["run_manager"] = RunManager(
            event_bus=event_bus,
            rollout_discovery=RolloutDiscovery(),
            idle_timeout_seconds=args.idle_timeout_seconds,
            max_run_seconds=args.max_run_seconds,
        )
    return create_app(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
