from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Protocol

from agent_harness.models import Run, Session, SessionStats, Usage

logger = logging.getLogger(__name__)

TOKEN_KEYS = ("input", "output", "cache_read", "cache_creation")


class UsageRepository(Protocol):
    def get_session(self, session_id: str) -> Session: ...

    def list_runs(self, session_id: str) -> list[Run]: ...

    def get_run(self, session_id: str, run_id: str) -> Run: ...

    def update_run_usage(self, run_id: str, usage: Usage) -> Run: ...

    def update_session_stats(self, session_id: str, stats: SessionStats) -> Session: ...


def parse_claude_usage(data: object, *, cost_usd: object = None) -> Usage | None:
    if not isinstance(data, Mapping):
        return None
    return Usage(
        input=_nonnegative_int(data.get("input_tokens")),
        output=_nonnegative_int(data.get("output_tokens")),
        cache_read=_nonnegative_int(data.get("cache_read_input_tokens")),
        cache_creation=_nonnegative_int(data.get("cache_creation_input_tokens")),
        cost_usd=_nonnegative_float(cost_usd),
    )


def parse_codex_token_count(payload: object) -> tuple[Usage | None, int | None]:
    if not isinstance(payload, Mapping):
        return None, None
    info = payload.get("info")
    if info is None:
        return None, None
    if not isinstance(info, Mapping):
        logger.warning("Skipping Codex token_count with non-object info: type=%s", type(info).__name__)
        return None, None

    last = info.get("last_token_usage")
    context_window = _positive_int(info.get("model_context_window"))
    if not isinstance(last, Mapping):
        return (Usage(), context_window) if context_window is not None else (None, None)

    return (
        parse_codex_usage(last),
        context_window,
    )


def parse_codex_usage(data: object) -> Usage | None:
    if not isinstance(data, Mapping):
        return None
    return Usage(
        input=_nonnegative_int(data.get("input_tokens")),
        output=_nonnegative_int(data.get("output_tokens")),
        cache_read=_nonnegative_int(data.get("cached_input_tokens")),
        cache_creation=0,
    )


def add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_creation=left.cache_creation + right.cache_creation,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def aggregate_session_stats(
    runs: Iterable[Run],
    *,
    previous: SessionStats | None = None,
    context_window: int | None = None,
) -> SessionStats:
    totals = {key: 0 for key in TOKEN_KEYS}
    cost_usd = 0.0
    for run in runs:
        totals["input"] += run.usage.input
        totals["output"] += run.usage.output
        totals["cache_read"] += run.usage.cache_read
        totals["cache_creation"] += run.usage.cache_creation
        cost_usd += run.usage.cost_usd

    latest_context_window = context_window
    if latest_context_window is None and previous is not None:
        latest_context_window = previous.context_window

    return SessionStats(
        messages=previous.messages if previous is not None else 0,
        tokens=totals,
        cost_usd=cost_usd,
        context_window=latest_context_window,
    )


def apply_run_usage(
    repository: UsageRepository,
    *,
    session_id: str,
    run_id: str,
    usage: Usage,
    context_window: int | None = None,
) -> Run:
    run = repository.get_run(session_id, run_id)
    updated = repository.update_run_usage(run_id, add_usage(run.usage, usage))
    session = repository.get_session(session_id)
    stats = aggregate_session_stats(
        repository.list_runs(session_id),
        previous=session.stats,
        context_window=context_window,
    )
    repository.update_session_stats(session_id, stats)
    return updated


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _positive_int(value: object) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed >= 1 else None


def _nonnegative_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0 else 0.0
