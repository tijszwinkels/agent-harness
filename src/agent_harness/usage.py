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

Accounting (``TOKEN_ACCOUNTING_VERSION``): claude and codex ``run.usage``
events carry an ``accounting`` object with a stable identity ``key`` for
the native usage report. Both repositories keep a per-session ledger of
the largest usage seen per key and add only the growth, so a report that
is repeated (claude content blocks, a repeated codex ``token_count``,
replay after a restart) counts once — see ``plan_usage_application``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_harness.models import (
    TOKEN_ACCOUNTING_VERSION,
    ContextSource,
    SessionStats,
    Usage,
)

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

    Codex follows OpenAI's convention: ``input_tokens`` INCLUDES
    ``cached_input_tokens`` (codex's own ``non_cached_input()`` is
    ``input_tokens - cached_input_tokens``), and ``output_tokens``
    includes ``reasoning_output_tokens``. ``Usage`` uses claude's
    disjoint convention, so ``input`` is the non-cached part and
    ``cache_read`` the cached part — the four fields sum to the tokens
    processed instead of counting cached input twice.

    ``cache_write_input_tokens`` (newer codex) is not mapped: codex does
    not document whether it is part of ``input_tokens``, and it has only
    been observed as zero. ``cache_creation`` is therefore always zero.
    """
    if not isinstance(data, Mapping):
        return None
    input_tokens = _nonnegative_int(data.get("input_tokens"))
    cached = min(_nonnegative_int(data.get("cached_input_tokens")), input_tokens)
    return Usage(
        input=input_tokens - cached,
        output=_nonnegative_int(data.get("output_tokens")),
        cache_read=cached,
        cache_creation=0,
    )


def codex_usage_key(payload: object) -> str | None:
    """Stable identity of the model call a codex ``token_count`` reports.

    Codex appends each call's ``last_token_usage`` to the cumulative
    ``total_token_usage`` (``TokenUsageInfo::new_or_append``), so the
    cumulative value identifies the call: a repeated ``token_count``
    (same total) and the post-compaction report (same total, zeroed
    ``last_token_usage``) share the key of the call before them.
    ``None`` when the cumulative block is missing — such a report cannot
    be de-duplicated and is not counted.
    """
    if not isinstance(payload, Mapping):
        return None
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return None
    total = info.get("total_token_usage")
    if not isinstance(total, Mapping):
        return None
    parts = [
        _nonnegative_int(total.get(field))
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    ]
    if not any(parts):
        return None
    return "codex:total:" + ":".join(str(part) for part in parts)


def claude_usage_key(message: object) -> str | None:
    """Stable identity of a claude assistant message: ``message.id``.

    Claude Code writes one transcript record per content block (thinking,
    text, tool_use) of a single API response, each repeating the
    response's ``message.usage``; later records may carry a larger
    ``output_tokens`` (partial then final). ``None`` without an id."""
    if not isinstance(message, Mapping):
        return None
    message_id = message.get("id")
    if not isinstance(message_id, str) or not message_id.strip():
        return None
    return "claude:" + message_id


def parse_codex_context_snapshot(payload: object) -> int | None:
    """Pull the current context size from a codex
    ``event_msg/token_count`` payload's ``info.last_token_usage.total_tokens``.

    Codex documents ``last_token_usage``'s total as "the latest active
    context size" and computes its own "context left" indicator from it.
    ``total_token_usage`` is the CUMULATIVE session total (it grows past
    ``model_context_window`` on long sessions) and must not be used here.
    After a compaction codex reports a zeroed ``last_token_usage`` whose
    ``total_tokens`` is the compacted context estimate, so the snapshot
    may decrease.

    Returns ``None`` whenever the snapshot is unavailable: codex's
    first ``token_count`` after session start has ``info: null``; some
    payloads omit ``last_token_usage``; a zero total is treated as
    no-data (a real zero is indistinguishable from "we haven't observed
    yet" and we mustn't overwrite an earlier real snapshot with it).
    """
    if not isinstance(payload, Mapping):
        return None
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return None
    last = info.get("last_token_usage")
    if not isinstance(last, Mapping):
        return None
    snapshot = _nonnegative_int(last.get("total_tokens"))
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


def usage_is_zero(usage: Usage) -> bool:
    return not (
        usage.input or usage.output or usage.cache_read or usage.cache_creation or usage.cost_usd
    )


def _max_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=max(left.input, right.input),
        output=max(left.output, right.output),
        cache_read=max(left.cache_read, right.cache_read),
        cache_creation=max(left.cache_creation, right.cache_creation),
        cost_usd=max(left.cost_usd, right.cost_usd),
    )


def _subtract_usage(left: Usage, right: Usage) -> Usage:
    """``left - right`` for ``left >= right`` component-wise."""
    return Usage(
        input=left.input - right.input,
        output=left.output - right.output,
        cache_read=left.cache_read - right.cache_read,
        cache_creation=left.cache_creation - right.cache_creation,
        cost_usd=max(left.cost_usd - right.cost_usd, 0.0),
    )


@dataclass(frozen=True)
class UsageAccounting:
    """The ``accounting`` object of a ``run.usage`` event. ``key`` is the
    native report's identity; ``None`` means it had none."""

    key: str | None
    version: int = TOKEN_ACCOUNTING_VERSION

    def to_data(self) -> dict[str, Any]:
        return {"version": self.version, "key": self.key}


def usage_accounting_from(data: object) -> UsageAccounting | None:
    """The event's accounting, or ``None`` for an unkeyed (additive)
    event: pi usage, and events stored by older harness versions."""
    if not isinstance(data, Mapping):
        return None
    raw = data.get("accounting")
    if not isinstance(raw, Mapping):
        return None
    version = raw.get("version")
    if isinstance(version, bool) or version != TOKEN_ACCOUNTING_VERSION:
        return None
    key = raw.get("key")
    return UsageAccounting(key=key if isinstance(key, str) and key else None)


_CONTEXT_SOURCES: frozenset[str] = frozenset(ContextSource.__args__)  # type: ignore[attr-defined]


def context_source_from(data: object) -> ContextSource | None:
    if not isinstance(data, Mapping):
        return None
    raw = data.get("context_source")
    return raw if isinstance(raw, str) and raw in _CONTEXT_SOURCES else None  # type: ignore[return-value]


@dataclass(frozen=True)
class UsageApplication:
    """What one ``run.usage`` event changes. ``run_delta`` is added to
    ``Run.usage``; ``stats`` replaces ``Session.stats``; ``ledger`` is the
    ``(key, usage)`` row to store (``None``: no ledger change)."""

    run_delta: Usage
    stats: SessionStats
    ledger: tuple[str, Usage] | None


def plan_usage_application(
    stats: SessionStats,
    data: Mapping[str, Any],
    *,
    incoming: Usage,
    previous: Usage | None,
    latest_key: str | None,
    context_window: int | None,
    context_used: int | None,
) -> UsageApplication:
    """Decide how a ``run.usage`` event changes a session. Shared by the
    in-memory and SQLite repositories so their rules can't drift.

    ``previous`` is the ledger entry for the event's key (``None`` when
    the key is new or the event is unkeyed); ``latest_key`` is the key
    most recently added to this session's ledger.

    Tokens:
    - Unkeyed events (pi; older stored events) add ``incoming`` as before.
    - Keyed events add only the growth over the largest usage already
      recorded for that key, per field: a repeat adds nothing, a
      partial-then-final claude message adds the final value once, and
      distinct messages with identical numbers each count.
    - An event whose accounting has no key adds nothing; non-zero usage
      is counted in ``skipped_usage_records`` instead.

    Context: a snapshot is applied unless the event's key is already in
    the ledger and is not the latest one — that is a replay of an older
    report, which must not roll the snapshot back.

    ``tokens_version``: never changed, except that a legacy session
    (``None``) with nothing accumulated yet starts on the current rules —
    there is nothing earlier that could be wrong.
    """
    accounting = usage_accounting_from(data)
    skipped = 0
    ledger: tuple[str, Usage] | None = None
    fresh = True
    if accounting is None:
        delta = incoming
    elif accounting.key is None:
        delta = Usage()
        if not usage_is_zero(incoming):
            skipped = 1
    elif previous is None:
        delta = incoming
        ledger = (accounting.key, incoming)
    else:
        merged = _max_usage(previous, incoming)
        delta = _subtract_usage(merged, previous)
        if merged != previous:
            ledger = (accounting.key, merged)
        fresh = accounting.key == latest_key

    tokens = dict(stats.tokens or {})
    nothing_accumulated = not any(
        int(tokens.get(field, 0) or 0)
        for field in ("input", "output", "cache_read", "cache_creation")
    ) and not stats.cost_usd
    for field in ("input", "output", "cache_read", "cache_creation"):
        tokens[field] = int(tokens.get(field, 0)) + getattr(delta, field)
    update: dict[str, Any] = {
        "tokens": tokens,
        "cost_usd": stats.cost_usd + delta.cost_usd,
    }
    if stats.tokens_version is None and nothing_accumulated:
        update["tokens_version"] = TOKEN_ACCOUNTING_VERSION
    if skipped:
        update["skipped_usage_records"] = stats.skipped_usage_records + skipped
    if context_window is not None:
        update["context_window"] = context_window
    if context_used is not None and fresh:
        # SNAPSHOT semantics: overwrite, never sum. A None value means
        # "no fresh observation" (event omitted the field); leave the
        # prior snapshot — and its source — in place.
        update["context_used"] = context_used
        update["context_used_source"] = context_source_from(data)
    return UsageApplication(
        run_delta=delta,
        stats=stats.model_copy(update=update),
        ledger=ledger,
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
