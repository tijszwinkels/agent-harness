"""Parsers for backend-specific token-count records.

Cherry-picked from closed PR #11 (Vega's rollout-usage work). The
parsers translate claude's ``.message.usage`` blocks and codex's
``event_msg/token_count`` payloads into the unified ``Usage`` model;
codex additionally exposes ``model_context_window`` which the parser
returns alongside the per-turn usage for ``Session.stats.context_window``.

The aggregation helpers (``aggregate_session_stats``, ``apply_run_usage``)
and the ``UsageRepository`` protocol from PR #11 are intentionally
dropped — Phase 3's materialization runs through
``repository.materialize_event``'s ``run.usage`` branch, which applies
the usage directly to the Run + Session in one transactional pass.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from agent_harness.models import Usage

logger = logging.getLogger(__name__)


def parse_claude_usage(data: object, *, cost_usd: object = None) -> Usage | None:
    """Parse a claude rollout ``message.usage`` block into a ``Usage``.

    Maps claude's field names (``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens``, ``cache_creation_input_tokens``) to
    the unified Usage fields. ``cost_usd`` is optional — claude's
    rollouts don't include it; the harness's stream-json ``result``
    block carries it but that surface retires in Phase 3+.
    """
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
    """Parse a codex ``event_msg/token_count`` payload.

    Returns ``(usage, context_window)``:
    - ``usage`` is built from ``info.last_token_usage`` (per-turn).
      NOT from ``total_token_usage`` — that would double-count across
      runs in a session because codex reports cumulative totals.
    - ``context_window`` is ``info.model_context_window`` (session-scoped).

    Special cases:
    - Codex's first ``token_count`` after session start has
      ``info: null``; returns ``(None, None)``.
    - ``info`` present but no ``last_token_usage``: returns ``(Usage(),
      context_window)`` so the materializer can still update
      ``Session.stats.context_window``.
    """
    if not isinstance(payload, Mapping):
        return None, None
    info = payload.get("info")
    if info is None:
        return None, None
    if not isinstance(info, Mapping):
        logger.warning(
            "Skipping Codex token_count with non-object info: type=%s",
            type(info).__name__,
        )
        return None, None

    last = info.get("last_token_usage")
    context_window = _positive_int(info.get("model_context_window"))
    if not isinstance(last, Mapping):
        return (Usage(), context_window) if context_window is not None else (None, None)

    return parse_codex_usage(last), context_window


def parse_codex_usage(data: object) -> Usage | None:
    """Parse a single codex token-usage block.

    Maps codex's ``cached_input_tokens`` onto our ``cache_read`` field.
    Codex doesn't separately report cache-creation tokens, so
    ``cache_creation`` is always zero on this path.
    """
    if not isinstance(data, Mapping):
        return None
    return Usage(
        input=_nonnegative_int(data.get("input_tokens")),
        output=_nonnegative_int(data.get("output_tokens")),
        cache_read=_nonnegative_int(data.get("cached_input_tokens")),
        cache_creation=0,
    )


def parse_codex_context_snapshot(payload: object) -> int | None:
    """Pull the currently-loaded context size from a codex
    ``event_msg/token_count`` payload's
    ``info.total_token_usage.total_tokens``.

    Distinct from ``parse_codex_token_count``: that returns the
    per-turn delta from ``last_token_usage`` (additive on
    ``Session.stats.tokens``); this returns a snapshot suitable for
    overwrite onto ``Session.stats.context_used``. After every turn
    codex emits a fresh ``token_count`` whose total reflects the
    context loaded for the next turn — so the latest value wins.

    Returns ``None`` whenever the snapshot is unavailable: codex's
    first ``token_count`` after session start has ``info: null``; some
    payloads omit ``total_token_usage`` entirely; a zero total is
    treated as no-data (a real zero is indistinguishable from "we
    haven't observed yet" and we mustn't overwrite an earlier real
    snapshot with it).
    """
    if not isinstance(payload, Mapping):
        return None
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return None
    total = info.get("total_token_usage")
    if not isinstance(total, Mapping):
        return None
    snapshot = _nonnegative_int(total.get("total_tokens"))
    return snapshot if snapshot >= 1 else None


def parse_claude_context_snapshot(usage: object) -> int | None:
    """Compute the currently-loaded context size from a claude
    ``message.usage`` block.

    For an assistant turn the loaded context equals
    ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``.
    ``output_tokens`` is excluded — output is the model's response, not
    part of the context the model sees on the next turn.

    Returns ``None`` for a non-mapping input (matches
    ``parse_claude_usage``'s contract). An all-zero mapping returns
    ``0``, which the materializer treats as "no useful snapshot" — see
    ``observer._run_usage_event``'s skip-on-zero guard.
    """
    if not isinstance(usage, Mapping):
        return None
    return (
        _nonnegative_int(usage.get("input_tokens"))
        + _nonnegative_int(usage.get("cache_creation_input_tokens"))
        + _nonnegative_int(usage.get("cache_read_input_tokens"))
    )


def parse_pi_usage(data: object) -> Usage | None:
    """Parse a pi rollout ``message.usage`` block into a ``Usage``.

    pi reports ``{input, output, cacheRead, cacheWrite, reasoning,
    totalTokens, cost:{input, output, cacheRead, cacheWrite, total}}``
    (verified pi rollout corpus 2026-07-06). Maps camelCase token fields
    onto the unified Usage; ``cacheWrite`` is pi's cache-creation
    equivalent. ``cost.total`` carries the per-turn USD cost — unlike
    claude/codex rollouts, pi embeds cost directly, so ``cost_usd`` is
    populated rather than defaulted to zero.
    """
    if not isinstance(data, Mapping):
        return None
    cost = data.get("cost")
    cost_total = cost.get("total") if isinstance(cost, Mapping) else None
    return Usage(
        input=_nonnegative_int(data.get("input")),
        output=_nonnegative_int(data.get("output")),
        cache_read=_nonnegative_int(data.get("cacheRead")),
        cache_creation=_nonnegative_int(data.get("cacheWrite")),
        cost_usd=_nonnegative_float(cost_total),
    )


def parse_pi_context_snapshot(data: object) -> int | None:
    """Compute the currently-loaded context size from a pi
    ``message.usage`` block.

    Loaded context = ``input + cacheRead + cacheWrite`` (the prompt the
    model saw this turn); ``output`` is excluded — it's the response, not
    part of next-turn context. Mirrors ``parse_claude_context_snapshot``.

    Returns ``None`` for a non-mapping input (matches ``parse_pi_usage``'s
    contract). An all-zero mapping returns ``0``, which the materializer
    treats as "no useful snapshot" — see ``observer._run_usage_event``'s
    skip-on-zero guard.
    """
    if not isinstance(data, Mapping):
        return None
    return (
        _nonnegative_int(data.get("input"))
        + _nonnegative_int(data.get("cacheRead"))
        + _nonnegative_int(data.get("cacheWrite"))
    )


def add_usage(left: Usage, right: Usage) -> Usage:
    """Sum two ``Usage`` records component-wise. Used by the
    materializer to apply per-turn usage onto the run's running
    total."""
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_creation=left.cache_creation + right.cache_creation,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
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
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0 else 0.0
