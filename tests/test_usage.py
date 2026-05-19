"""Unit tests for ``agent_harness.usage`` parsers.

Cherry-picked from closed PR #11 (Vega's rollout-usage work). The
parsers translate backend-specific token-count records into the
unified ``Usage`` model and, for codex, surface the session-level
``context_window`` alongside the per-turn usage.
"""

from __future__ import annotations

from agent_harness.models import Usage
from agent_harness.usage import (
    parse_claude_context_snapshot,
    parse_claude_usage,
    parse_codex_context_snapshot,
    parse_codex_token_count,
    parse_codex_usage,
)


# --- claude .message.usage ---------------------------------------------------


def test_parse_claude_usage_extracts_all_token_kinds() -> None:
    usage = parse_claude_usage(
        {
            "input_tokens": 6,
            "output_tokens": 4,
            "cache_read_input_tokens": 18,
            "cache_creation_input_tokens": 21,
        }
    )
    assert usage == Usage(input=6, output=4, cache_read=18, cache_creation=21, cost_usd=0.0)


def test_parse_claude_usage_accepts_optional_cost_usd() -> None:
    usage = parse_claude_usage(
        {"input_tokens": 1, "output_tokens": 2},
        cost_usd=0.12,
    )
    assert usage is not None
    assert usage.cost_usd == 0.12


def test_parse_claude_usage_defaults_missing_fields_to_zero() -> None:
    usage = parse_claude_usage({"input_tokens": 100})
    assert usage == Usage(input=100, output=0, cache_read=0, cache_creation=0, cost_usd=0.0)


def test_parse_claude_usage_clamps_negative_to_zero() -> None:
    # Defensive: spec doesn't allow negative counts but the parser
    # mustn't propagate them into the model.
    usage = parse_claude_usage(
        {"input_tokens": -1, "output_tokens": -2, "cache_read_input_tokens": -3}
    )
    assert usage == Usage(input=0, output=0, cache_read=0, cache_creation=0, cost_usd=0.0)


def test_parse_claude_usage_returns_none_for_non_mapping_input() -> None:
    assert parse_claude_usage(None) is None
    assert parse_claude_usage("not a dict") is None
    assert parse_claude_usage([1, 2, 3]) is None


# --- codex event_msg/token_count ---------------------------------------------


def test_parse_codex_token_count_returns_usage_and_context_window() -> None:
    usage, context_window = parse_codex_token_count(
        {
            "info": {
                "last_token_usage": {
                    "input_tokens": 120,
                    "output_tokens": 300,
                    "cached_input_tokens": 50,
                },
                "model_context_window": 258400,
            }
        }
    )
    assert usage == Usage(input=120, output=300, cache_read=50, cache_creation=0, cost_usd=0.0)
    assert context_window == 258400


def test_parse_codex_token_count_handles_null_info() -> None:
    """Codex's first token_count event after session start has
    ``info: null``. Parser must return (None, None) — caller skips
    emitting a zero-usage run.usage event."""
    assert parse_codex_token_count({"info": None}) == (None, None)


def test_parse_codex_token_count_handles_missing_info_key() -> None:
    assert parse_codex_token_count({}) == (None, None)


def test_parse_codex_token_count_handles_non_object_info() -> None:
    """If ``info`` is not a mapping (defensive — never observed from
    real codex), parser warns and returns (None, None)."""
    usage, context_window = parse_codex_token_count({"info": "not-an-object"})
    assert usage is None
    assert context_window is None


def test_parse_codex_token_count_returns_context_window_alone_if_usage_missing() -> None:
    """``model_context_window`` is sometimes present even when
    ``last_token_usage`` is not (e.g. codex emits the window early in
    the session). Parser surfaces a zero Usage so the materializer
    can still update the session's context_window."""
    usage, context_window = parse_codex_token_count(
        {"info": {"model_context_window": 258400}}
    )
    assert usage == Usage()
    assert context_window == 258400


def test_parse_codex_token_count_returns_none_for_non_mapping_payload() -> None:
    assert parse_codex_token_count(None) == (None, None)


# --- codex usage record (per-turn) -------------------------------------------


def test_parse_codex_usage_maps_cached_input_to_cache_read() -> None:
    usage = parse_codex_usage(
        {"input_tokens": 100, "output_tokens": 200, "cached_input_tokens": 30}
    )
    assert usage == Usage(input=100, output=200, cache_read=30, cache_creation=0, cost_usd=0.0)


def test_parse_codex_usage_returns_none_for_non_mapping() -> None:
    assert parse_codex_usage(None) is None


# --- context-used snapshot helpers (Solace's context_used feature) -----------
#
# Distinct from the per-turn ``Usage`` parsers above: these return a
# SNAPSHOT of the currently-loaded context size, which the materializer
# OVERWRITES (not sums) onto ``Session.stats.context_used``. Decreases
# are valid (context compaction).


def test_parse_codex_context_snapshot_returns_total_tokens() -> None:
    snapshot = parse_codex_context_snapshot(
        {
            "info": {
                "total_token_usage": {"total_tokens": 12345},
                "last_token_usage": {"input_tokens": 50},
            }
        }
    )
    assert snapshot == 12345


def test_parse_codex_context_snapshot_returns_none_when_info_is_null() -> None:
    # Codex's first ``token_count`` after session start has ``info: null``
    # — the snapshot is not yet available; caller must skip emission.
    assert parse_codex_context_snapshot({"info": None}) is None


def test_parse_codex_context_snapshot_returns_none_when_info_missing() -> None:
    assert parse_codex_context_snapshot({}) is None


def test_parse_codex_context_snapshot_returns_none_when_total_usage_missing() -> None:
    assert (
        parse_codex_context_snapshot(
            {"info": {"last_token_usage": {"input_tokens": 50}}}
        )
        is None
    )


def test_parse_codex_context_snapshot_returns_none_when_total_tokens_is_zero() -> None:
    # A zero value is meaningless as a "currently-loaded" snapshot —
    # treat it as no-data so the materializer doesn't overwrite a real
    # earlier observation with 0.
    assert (
        parse_codex_context_snapshot(
            {"info": {"total_token_usage": {"total_tokens": 0}}}
        )
        is None
    )


def test_parse_codex_context_snapshot_returns_none_for_non_mapping() -> None:
    assert parse_codex_context_snapshot(None) is None
    assert parse_codex_context_snapshot("not-a-dict") is None


def test_parse_codex_context_snapshot_ignores_garbage_total_tokens() -> None:
    # Defensive: a non-numeric value mustn't crash the observer.
    assert (
        parse_codex_context_snapshot(
            {"info": {"total_token_usage": {"total_tokens": "abc"}}}
        )
        is None
    )


def test_parse_claude_context_snapshot_sums_input_and_cache_fields() -> None:
    snapshot = parse_claude_context_snapshot(
        {
            "input_tokens": 100,
            "output_tokens": 200,  # explicitly excluded
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 30,
        }
    )
    # output is NOT in the loaded context for the next turn.
    assert snapshot == 100 + 50 + 30


def test_parse_claude_context_snapshot_excludes_output_tokens() -> None:
    # Belt-and-suspenders: a usage block with ONLY output tokens
    # produces a zero snapshot.
    assert parse_claude_context_snapshot({"output_tokens": 9999}) == 0


def test_parse_claude_context_snapshot_handles_missing_fields() -> None:
    assert parse_claude_context_snapshot({"input_tokens": 42}) == 42
    assert parse_claude_context_snapshot({}) == 0


def test_parse_claude_context_snapshot_returns_none_for_non_mapping() -> None:
    assert parse_claude_context_snapshot(None) is None
    assert parse_claude_context_snapshot("not-a-dict") is None


def test_parse_claude_context_snapshot_clamps_negative_to_zero() -> None:
    # Defensive: negative components must not subtract from the snapshot.
    assert (
        parse_claude_context_snapshot(
            {
                "input_tokens": -5,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 3,
            }
        )
        == 13
    )
