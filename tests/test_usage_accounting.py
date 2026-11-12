"""Usage accounting semantics (issue #48).

Synthetic native-format fixtures reproduce two defects of the earlier
accounting and lock in the corrected rules:

- Claude Code writes one transcript record per content block of an API
  response, each repeating the response's ``message.usage``; summing the
  records inflated session totals. Usage now counts once per
  ``message.id`` (largest value per field, so a partial-then-final report
  adds the final value once).
- Codex ``stats.context_used`` came from the CUMULATIVE
  ``total_token_usage``; it now comes from ``last_token_usage`` (codex's
  own current-context source). Codex ``input_tokens`` includes cached
  input, so ``Usage.input`` is now the non-cached part.

Plus the provenance contract consumers use to tell corrected values from
legacy ones (``tokens_version``, ``skipped_usage_records``,
``context_used_source``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_harness.events import DurableEventBus
from agent_harness.models import (
    CreateRunRequest,
    CreateSessionRequest,
    Event,
    Project,
    Session,
    SessionStats,
)
from agent_harness.observer import (
    ExternalTranscriptObserver,
    TranscriptIdentity,
    parse_transcript_line,
)
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository


# --- native-format fixtures ---------------------------------------------------


def claude_record(
    message_id: str | None,
    *,
    input_tokens: int = 3,
    output_tokens: int = 10,
    cache_read: int = 20000,
    cache_creation: int = 500,
    block: dict | None = None,
    stop_reason: str | None = None,
    sidechain: bool = False,
) -> str:
    """One Claude Code assistant transcript record (one content block)."""
    message: dict = {
        "model": "claude-opus-4-7",
        "role": "assistant",
        "type": "message",
        "content": [block or {"type": "text", "text": "ok"}],
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
            "service_tier": "standard",
        },
    }
    if message_id is not None:
        message["id"] = message_id
    record = {
        "type": "assistant",
        "cwd": "/repo",
        "isSidechain": sidechain,
        "requestId": f"req_{message_id}",
        "message": message,
    }
    return json.dumps(record)


def codex_token_count(last: dict | None, total: dict | None, window: int | None = 258400) -> str:
    info: dict = {}
    if last is not None:
        info["last_token_usage"] = last
    if total is not None:
        info["total_token_usage"] = total
    if window is not None:
        info["model_context_window"] = window
    return json.dumps(
        {"type": "event_msg", "payload": {"type": "token_count", "info": info, "rate_limits": None}}
    )


def codex_usage(input_tokens: int, cached: int, output: int, reasoning: int = 0, total: int | None = None) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": input_tokens + output if total is None else total,
    }


def add_usage(left: dict, right: dict) -> dict:
    return {key: left[key] + right[key] for key in left}


# --- harness -------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def repository(request, tmp_path) -> Iterator[object]:
    if request.param == "memory":
        yield InMemoryRepository()
        return
    repo = open_sqlite_repository(tmp_path / "harness.db")
    try:
        yield repo
    finally:
        repo.close()


def make_session(repository, backend: str = "claude-code") -> tuple[Session, str]:
    session = repository.create_session(
        CreateSessionRequest(
            backend=backend,
            model="model-x",
            project=Project(path="/repo", name="repo"),
        )
    )
    return session, start_run(repository, session.id)


def start_run(repository, session_id: str) -> str:
    run = repository.create_run(session_id, CreateRunRequest(message="hi"))
    repository.start_run(session_id, run.id)
    return run.id


def usage_events(backend: str, session_id: str, lines: list[str]) -> list[Event]:
    identity = TranscriptIdentity(
        backend=backend,  # type: ignore[arg-type]
        path=Path("/synthetic/transcript.jsonl"),
        session_id=session_id,
        is_rebound=True,
    )
    events: list[Event] = []
    offset = 0
    for line in lines:
        events.extend(
            event
            for event in parse_transcript_line(line, identity=identity, offset=offset)
            if event.event == "run.usage"
        )
        offset += len(line) + 1
    return events


def apply(repository, events: list[Event], run_id: str) -> None:
    """Materialize like ``DurableEventBus`` does after ``publish_line``
    stamps the active run id."""
    for event in events:
        stamped = event.model_copy(update={"run_id": run_id})
        atomic = getattr(repository, "append_and_materialize_event", None)
        if callable(atomic):
            _, deferred = atomic(stamped)
            assert deferred is None
        else:
            repository.materialize_event(stamped)


def feed(repository, backend: str, session_id: str, run_id: str, lines: list[str]) -> None:
    apply(repository, usage_events(backend, session_id, lines), run_id)


def tokens(repository, session_id: str) -> dict:
    return repository.get_session(session_id).stats.tokens


def run_usage(repository, session_id: str, run_id: str) -> dict:
    run = next(run for run in repository.list_runs(session_id) if run.id == run_id)
    return run.usage.model_dump(exclude={"cost_usd"})


# --- Claude: once per logical message -------------------------------------------


def test_claude_repeated_content_block_usage_counts_once(repository) -> None:
    """Regression: three content blocks of ONE response used to count 3x."""
    session, run_id = make_session(repository)
    lines = [
        claude_record("msg_A", block={"type": "thinking", "thinking": "hmm", "signature": "s"}),
        claude_record("msg_A", block={"type": "text", "text": "Let me look."}),
        claude_record(
            "msg_A",
            block={"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            stop_reason="tool_use",
        ),
    ]
    feed(repository, "claude-code", session.id, run_id, lines)

    expected = {"input": 3, "output": 10, "cache_read": 20000, "cache_creation": 500}
    assert tokens(repository, session.id) == expected
    assert run_usage(repository, session.id, run_id) == expected
    stats = repository.get_session(session.id).stats
    assert stats.tokens_version == 2
    assert stats.skipped_usage_records == 0
    assert stats.context_used == 3 + 20000 + 500
    assert stats.context_used_source == "claude.message.usage"


def test_claude_partial_then_final_usage_counts_final_once(repository) -> None:
    """Native shape: earlier blocks carry a partial output_tokens, the last
    block the final one (observed: 8, 8, 375). Count 375 once."""
    session, run_id = make_session(repository)
    lines = [
        claude_record("msg_A", output_tokens=8),
        claude_record("msg_A", output_tokens=8),
        claude_record("msg_A", output_tokens=375, stop_reason="tool_use"),
    ]
    feed(repository, "claude-code", session.id, run_id, lines)
    assert tokens(repository, session.id) == {
        "input": 3, "output": 375, "cache_read": 20000, "cache_creation": 500,
    }


def test_claude_distinct_messages_with_identical_usage_both_count(repository) -> None:
    session, run_id = make_session(repository)
    lines = [claude_record("msg_A"), claude_record("msg_B")]
    feed(repository, "claude-code", session.id, run_id, lines)
    assert tokens(repository, session.id) == {
        "input": 6, "output": 20, "cache_read": 40000, "cache_creation": 1000,
    }


def test_claude_interleaved_messages_and_late_final_update(repository) -> None:
    """A final update to an older message (e.g. after a sidechain record in
    between) still adds only its growth, and does not roll the context
    snapshot back to the older message."""
    session, run_id = make_session(repository)
    lines = [
        claude_record("msg_A", output_tokens=5, cache_read=1000),
        claude_record("msg_B", output_tokens=7, cache_read=2000),
        claude_record("msg_A", output_tokens=50, cache_read=1000),
    ]
    feed(repository, "claude-code", session.id, run_id, lines)
    stats = repository.get_session(session.id).stats
    assert stats.tokens == {"input": 6, "output": 57, "cache_read": 3000, "cache_creation": 1000}
    assert stats.context_used == 3 + 2000 + 500


def test_claude_missing_message_id_is_skipped_not_counted(repository) -> None:
    """Without ``message.id`` a record cannot be de-duplicated: it is not
    counted, and the session says so instead of silently guessing."""
    session, run_id = make_session(repository)
    feed(repository, "claude-code", session.id, run_id, [claude_record("msg_A"), claude_record(None)])
    stats = repository.get_session(session.id).stats
    assert stats.tokens == {"input": 3, "output": 10, "cache_read": 20000, "cache_creation": 500}
    assert stats.skipped_usage_records == 1
    # The snapshot is still a valid observation of the current context.
    assert stats.context_used == 20503


def test_claude_all_zero_usage_without_id_is_not_a_skip(repository) -> None:
    session, run_id = make_session(repository)
    zero = claude_record(None, input_tokens=0, output_tokens=0, cache_read=0, cache_creation=0)
    feed(repository, "claude-code", session.id, run_id, [zero])
    assert repository.get_session(session.id).stats.skipped_usage_records == 0


def test_claude_sidechain_usage_counts_but_does_not_move_context(repository) -> None:
    session, run_id = make_session(repository)
    lines = [
        claude_record("msg_main", cache_read=50000),
        claude_record("msg_sub", cache_read=900, sidechain=True),
    ]
    feed(repository, "claude-code", session.id, run_id, lines)
    stats = repository.get_session(session.id).stats
    assert stats.tokens["cache_read"] == 50900
    assert stats.context_used == 3 + 50000 + 500


def test_claude_replay_does_not_inflate_or_roll_back_context(repository) -> None:
    """Re-reading the whole transcript (cursor reset, retried line) adds
    nothing and leaves the newest context snapshot in place."""
    session, run_id = make_session(repository)
    lines = [
        claude_record("msg_A", cache_read=1000),
        claude_record("msg_A", cache_read=1000),
        claude_record("msg_B", cache_read=9000),
    ]
    feed(repository, "claude-code", session.id, run_id, lines)
    before = repository.get_session(session.id).stats
    feed(repository, "claude-code", session.id, run_id, lines)
    feed(repository, "claude-code", session.id, run_id, lines[:1])
    after = repository.get_session(session.id).stats
    assert after.tokens == before.tokens
    assert after.context_used == before.context_used == 3 + 9000 + 500


def test_same_message_id_in_two_sessions_counts_in_each(repository) -> None:
    first, first_run = make_session(repository)
    second, second_run = make_session(repository)
    feed(repository, "claude-code", first.id, first_run, [claude_record("msg_A")])
    feed(repository, "claude-code", second.id, second_run, [claude_record("msg_A")])
    assert tokens(repository, first.id)["output"] == 10
    assert tokens(repository, second.id)["output"] == 10


def test_claude_ledger_survives_sqlite_restart(tmp_path) -> None:
    """A message whose blocks straddle a harness restart, and a replay of
    the whole transcript after it, still count once."""
    db = tmp_path / "harness.db"
    repo = open_sqlite_repository(db)
    session, run_id = make_session(repo)
    first_blocks = [claude_record("msg_A", output_tokens=8), claude_record("msg_B", output_tokens=4)]
    feed(repo, "claude-code", session.id, run_id, [first_blocks[0]])
    repo.close()

    repo = open_sqlite_repository(db)
    try:
        # The restart reconciles the dangling run; a new run continues.
        run_id = start_run(repo, session.id)
        rest = [claude_record("msg_A", output_tokens=375), first_blocks[1]]
        feed(repo, "claude-code", session.id, run_id, rest)
        feed(repo, "claude-code", session.id, run_id, [first_blocks[0], *rest])
        stats = repo.get_session(session.id).stats
        assert stats.tokens == {"input": 6, "output": 379, "cache_read": 40000, "cache_creation": 1000}
        assert stats.tokens_version == 2
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_claude_end_to_end_observer_counts_blocks_once(tmp_path) -> None:
    """Real ingestion path: observer tail → DurableEventBus → SQLite."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session, run_id = make_session(repository)
        raw = session.id.removeprefix("ses_")
        claude_uuid = f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"
        rollout = tmp_path / ".claude" / "projects" / "-repo" / f"{claude_uuid}.jsonl"
        rollout.parent.mkdir(parents=True)
        lines = [
            claude_record("msg_A", output_tokens=8, block={"type": "text", "text": "Looking."}),
            claude_record(
                "msg_A",
                output_tokens=120,
                block={"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "x"}},
                stop_reason="tool_use",
            ),
            claude_record("msg_B", output_tokens=30, cache_read=25000, stop_reason="end_turn"),
        ]
        rollout.write_text("\n".join(lines) + "\n", encoding="utf-8")

        observer = ExternalTranscriptObserver(DurableEventBus(repository), repository=repository)
        await observer.tail_file(rollout)

        stats = repository.get_session(session.id).stats
        assert stats.tokens == {"input": 6, "output": 150, "cache_read": 45000, "cache_creation": 1000}
        assert stats.context_used == 3 + 25000 + 500
        assert run_usage(repository, session.id, run_id)["output"] == 150
    finally:
        repository.close()


# --- Codex: current context vs cumulative totals ---------------------------------


def test_codex_context_is_current_not_cumulative(repository) -> None:
    """Regression: context_used used to be the cumulative total, which grew
    past the context window on long sessions."""
    session, run_id = make_session(repository, "codex")
    calls = [codex_usage(150_000, 140_000, 2_000, 800) for _ in range(5)]
    lines, total = [], None
    for call in calls:
        total = call if total is None else add_usage(total, call)
        lines.append(codex_token_count(call, total))
    feed(repository, "codex", session.id, run_id, lines)

    stats = repository.get_session(session.id).stats
    assert total["total_tokens"] > stats.context_window  # the old value
    assert stats.context_used == 152_000 <= stats.context_window
    assert stats.context_used_source == "codex.last_token_usage"
    # Cumulative totals remain cumulative, with cached input not doubled:
    # the four fields sum to codex's own cumulative total_tokens.
    assert stats.tokens == {"input": 50_000, "output": 10_000, "cache_read": 700_000, "cache_creation": 0}
    assert sum(stats.tokens.values()) == total["total_tokens"]


def test_codex_repeated_token_count_counts_once(repository) -> None:
    """Codex sometimes writes the same token_count twice (same cumulative
    total); summing last_token_usage counted that call twice."""
    session, run_id = make_session(repository, "codex")
    first = codex_usage(1000, 0, 100)
    second = codex_usage(1500, 1000, 50)
    total_1 = first
    total_2 = add_usage(first, second)
    lines = [
        codex_token_count(first, total_1),
        codex_token_count(second, total_2),
        codex_token_count(second, total_2),
    ]
    feed(repository, "codex", session.id, run_id, lines)
    assert tokens(repository, session.id) == {"input": 1500, "output": 150, "cache_read": 1000, "cache_creation": 0}


def test_codex_compaction_decreases_context_without_new_usage(repository) -> None:
    session, run_id = make_session(repository, "codex")
    call = codex_usage(200_000, 190_000, 1_000, 300)
    compacted = {
        "input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
        "output_tokens": 0, "reasoning_output_tokens": 0, "total_tokens": 16_369,
    }
    lines = [codex_token_count(call, call), codex_token_count(compacted, call)]
    feed(repository, "codex", session.id, run_id, lines)
    stats = repository.get_session(session.id).stats
    assert stats.context_used == 16_369
    assert stats.tokens == {"input": 10_000, "output": 1_000, "cache_read": 190_000, "cache_creation": 0}


def test_codex_stale_replay_does_not_roll_context_back(repository) -> None:
    session, run_id = make_session(repository, "codex")
    first = codex_usage(10_000, 0, 100)
    second = codex_usage(30_000, 9_000, 200)
    lines = [codex_token_count(first, first), codex_token_count(second, add_usage(first, second))]
    feed(repository, "codex", session.id, run_id, lines)
    feed(repository, "codex", session.id, run_id, lines[:1])
    stats = repository.get_session(session.id).stats
    assert stats.context_used == 30_200
    assert stats.tokens["output"] == 300


def test_codex_without_cumulative_total_is_skipped_not_guessed(repository) -> None:
    session, run_id = make_session(repository, "codex")
    feed(repository, "codex", session.id, run_id, [codex_token_count(codex_usage(500, 0, 20), None)])
    stats = repository.get_session(session.id).stats
    assert stats.tokens == {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    assert stats.skipped_usage_records == 1
    assert stats.context_used == 520


def test_codex_unknown_context_is_not_fabricated(repository) -> None:
    """No last_token_usage → context stays unknown (not 0, not the window,
    not the cumulative total); the window itself is still recorded."""
    session, run_id = make_session(repository, "codex")
    feed(repository, "codex", session.id, run_id, [codex_token_count(None, codex_usage(900_000, 0, 1))])
    stats = repository.get_session(session.id).stats
    assert stats.context_used is None
    assert stats.context_used_source is None
    assert stats.context_window == 258400


def test_codex_info_null_emits_nothing() -> None:
    line = json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}})
    assert usage_events("codex", "ses_x", [line]) == []


# --- Pi: unchanged ---------------------------------------------------------------


def test_pi_usage_stays_additive_with_cost_and_source(repository) -> None:
    session, run_id = make_session(repository, "pi")
    line = json.dumps(
        {
            "type": "message",
            "id": "a1b2c3d4",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input": 10, "output": 5, "cacheRead": 100, "cacheWrite": 7,
                    "cost": {"total": 0.25},
                },
                "stopReason": "stop",
            },
        }
    )
    events = usage_events("pi", session.id, [line])
    assert events and "accounting" not in events[0].data
    apply(repository, events, run_id)
    apply(repository, events, run_id)  # pi is per record, as before
    stats = repository.get_session(session.id).stats
    assert stats.tokens == {"input": 20, "output": 10, "cache_read": 200, "cache_creation": 14}
    assert stats.cost_usd == pytest.approx(0.5)
    assert stats.tokens_version == 2
    assert stats.context_used == 117
    assert stats.context_used_source == "pi.message.usage"


# --- provenance of legacy rows ---------------------------------------------------


def _legacy_session(backend: str, *, tokens: dict, context_used: int | None) -> Session:
    """A session row as an older harness stored it: no provenance keys."""
    payload = Session(
        backend=backend,  # type: ignore[arg-type]
        model="model-x",
        project=Project(path="/repo", name="repo"),
    ).model_dump(mode="json")
    stats = payload["stats"]
    for key in ("tokens_version", "skipped_usage_records", "context_used_source"):
        del stats[key]
    stats["tokens"] = tokens
    stats["context_used"] = context_used
    return Session.model_validate(payload)


def test_legacy_stats_load_as_unverified() -> None:
    legacy = _legacy_session("codex", tokens={"input": 5}, context_used=3_000_000)
    assert legacy.stats.tokens_version is None
    assert legacy.stats.context_used_source is None
    assert legacy.stats.skipped_usage_records == 0
    assert SessionStats().tokens_version is None
    assert Session(backend="codex", project=Project(path="/r", name="r")).stats.tokens_version == 2


def test_legacy_sqlite_row_without_provenance_stays_unverified(tmp_path) -> None:
    """A row written by an older harness keeps tokens_version=None after new
    usage arrives: its earlier (possibly inflated) totals are mixed in. The
    context snapshot, by contrast, is fully replaced by a corrected one."""
    db = tmp_path / "harness.db"
    repo = open_sqlite_repository(db)
    legacy = _legacy_session(
        "codex",
        tokens={"input": 4_000_000, "output": 9_000, "cache_read": 3_900_000, "cache_creation": 0},
        context_used=4_009_000,
    )
    repo.upsert_session(legacy)
    repo.close()
    # Simulate exactly what an older version wrote: strip the new keys.
    with sqlite3.connect(db) as connection:
        connection.execute(
            "update sessions set payload = json_remove(payload, '$.stats.tokens_version',"
            " '$.stats.skipped_usage_records', '$.stats.context_used_source') where id = ?",
            (legacy.id,),
        )

    repo = open_sqlite_repository(db)
    try:
        loaded = repo.get_session(legacy.id)
        assert loaded.stats.tokens_version is None
        assert loaded.stats.context_used_source is None

        run_id = start_run(repo, legacy.id)
        call = codex_usage(100_000, 90_000, 500)
        feed(repo, "codex", legacy.id, run_id, [codex_token_count(call, call)])
        after = repo.get_session(legacy.id).stats
        assert after.tokens_version is None  # not relabelled as corrected
        assert after.tokens["output"] == 9_500
        assert after.context_used == 100_500
        assert after.context_used_source == "codex.last_token_usage"
    finally:
        repo.close()


def test_legacy_session_without_accumulated_usage_is_promoted(repository) -> None:
    """Nothing was counted under the old rules, so nothing can be wrong."""
    legacy = _legacy_session("claude-code", tokens={}, context_used=None)
    repository.upsert_session(legacy)
    run_id = start_run(repository, legacy.id)
    feed(repository, "claude-code", legacy.id, run_id, [claude_record("msg_A"), claude_record("msg_A")])
    stats = repository.get_session(legacy.id).stats
    assert stats.tokens_version == 2
    assert stats.tokens["output"] == 10


def test_legacy_session_with_accumulated_usage_is_not_promoted(repository) -> None:
    legacy = _legacy_session("claude-code", tokens={"input": 1, "output": 0}, context_used=10)
    repository.upsert_session(legacy)
    run_id = start_run(repository, legacy.id)
    feed(repository, "claude-code", legacy.id, run_id, [claude_record("msg_A")])
    stats = repository.get_session(legacy.id).stats
    assert stats.tokens_version is None
    assert stats.context_used_source == "claude.message.usage"


def test_unversioned_stored_event_has_no_context_source(repository) -> None:
    """An event in the older format (no accounting/context_source) is
    applied additively and leaves context provenance unknown."""
    session, run_id = make_session(repository, "codex")
    event = Event(
        event="run.usage",
        session_id=session.id,
        data={"usage": {"input": 1, "output": 2}, "context_used": 3_000_000},
    )
    apply(repository, [event], run_id)
    stats = repository.get_session(session.id).stats
    assert stats.context_used == 3_000_000
    assert stats.context_used_source is None


def test_sqlite_schema_has_usage_ledger(tmp_path) -> None:
    repo = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session, run_id = make_session(repo)
        feed(repo, "claude-code", session.id, run_id, [claude_record("msg_A"), claude_record("msg_A")])
    finally:
        repo.close()
    with sqlite3.connect(tmp_path / "harness.db") as connection:
        rows = connection.execute("select session_id, usage_key from usage_ledger").fetchall()
    assert rows == [(session.id, "claude:msg_A")]
