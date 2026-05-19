import asyncio
import contextlib

import pytest

from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project
from agent_harness.storage import open_sqlite_repository


@pytest.mark.asyncio
async def test_event_bus_assigns_monotonic_sequences() -> None:
    bus = InMemoryEventBus()

    first = await bus.publish(Event(event="session.updated", data={"session_id": "a"}))
    second = await bus.publish(Event(event="run.started", data={"run_id": "r1"}))

    assert [first.sequence, second.sequence] == [1, 2]


@pytest.mark.asyncio
async def test_subscribe_replays_events_after_sequence() -> None:
    bus = InMemoryEventBus()
    await bus.publish(Event(event="session.updated", data={"session_id": "a"}))
    await bus.publish(Event(event="run.started", data={"run_id": "r1"}))

    subscription = bus.subscribe(after=0)
    replayed = [await subscription.__anext__(), await subscription.__anext__()]
    await subscription.aclose()

    assert [event.sequence for event in replayed] == [1, 2]
    assert [event.event for event in replayed] == ["session.updated", "run.started"]


@pytest.mark.asyncio
async def test_subscribe_receives_live_events_after_replay() -> None:
    bus = InMemoryEventBus()
    await bus.publish(Event(event="session.updated", data={"session_id": "a"}))

    subscription = bus.subscribe(after=1)
    next_event = asyncio.create_task(subscription.__anext__())
    published = await bus.publish(Event(event="run.started", data={"run_id": "r1"}))

    assert await asyncio.wait_for(next_event, timeout=1) == published
    await subscription.aclose()


@pytest.mark.asyncio
async def test_subscribe_emits_keepalive_under_filtered_traffic() -> None:
    """Regression: the keepalive deadline must track the SUBSCRIBER's
    last yield, not the bus's last publish. Otherwise high-volume traffic
    that ``_matches`` filters out (events for other sessions, or whose
    sequence is at/below ``after``) keeps ``queue.get`` returning before
    the timeout, and the keepalive never fires — the exact stale-cursor
    failure mode we're trying to surface."""
    bus = InMemoryEventBus()
    # ``after=100`` means every published event will be filtered out
    # (sequences start at 1). Pair with a tight keepalive window so the
    # test runs in well under a second.
    subscription = bus.subscribe(after=100, keepalive_seconds=0.05)

    async def flood_other_session():
        for _ in range(20):
            await bus.publish(
                Event(event="message", session_id="other", run_id="r", data={}),
            )
            await asyncio.sleep(0.01)

    flooder = asyncio.create_task(flood_other_session())
    try:
        sentinel = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)
        assert sentinel is None
    finally:
        flooder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await flooder
        await subscription.aclose()


@pytest.mark.asyncio
async def test_subscribe_emits_none_sentinel_on_keepalive_window() -> None:
    """``keepalive_seconds=N`` makes ``subscribe`` yield ``None`` whenever
    the bus stays silent for the interval. The SSE layer renders those
    as comment frames so clients can detect a dead/stuck stream within
    one missed-keepalive window."""
    bus = InMemoryEventBus()
    subscription = bus.subscribe(after=0, keepalive_seconds=0.01)
    try:
        # No events published — first pull should time out and yield None.
        sentinel = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)
        assert sentinel is None

        # The subscription must survive the timeout and still deliver real
        # events (regression guard: an earlier draft cancelled __anext__
        # from outside, tearing down the queue via the generator's finally).
        published = await bus.publish(
            Event(event="run.started", session_id="ses_a", run_id="run_a", data={}),
        )
        delivered = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)
        assert delivered == published
    finally:
        await subscription.aclose()


@pytest.mark.asyncio
async def test_replay_can_filter_by_session() -> None:
    bus = InMemoryEventBus()
    await bus.publish(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    await bus.publish(Event(event="run.started", session_id="ses_b", run_id="run_b", data={}))
    await bus.publish(Event(event="message", session_id="ses_a", run_id="run_a", data={}))

    session_events = await bus.replay(session_id="ses_a")

    assert [event.sequence for event in session_events] == [1, 3]


@pytest.mark.asyncio
async def test_inmemory_event_bus_reports_max_sequence() -> None:
    bus = InMemoryEventBus()

    assert await bus.max_sequence() == 0
    assert await bus.max_sequence(session_id="ses_a") == 0

    await bus.publish(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    await bus.publish(Event(event="run.started", session_id="ses_b", run_id="run_b", data={}))
    await bus.publish(Event(event="message", session_id="ses_a", run_id="run_a", data={}))

    assert await bus.max_sequence() == 3
    assert await bus.max_sequence(session_id="ses_a") == 3
    assert await bus.max_sequence(session_id="ses_b") == 2
    assert await bus.max_sequence(session_id="ses_missing") == 0


@pytest.mark.asyncio
async def test_durable_event_bus_continues_sequence_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    first_bus = DurableEventBus(repository)
    first = await first_bus.publish(Event(event="session.updated", session_id="ses_a", data={}))
    repository.close()

    reopened = open_sqlite_repository(db_path)
    second_bus = DurableEventBus(reopened)
    second = await second_bus.publish(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    replayed = await second_bus.replay()

    assert first.sequence == 1
    assert second.sequence == 2
    assert [event.sequence for event in replayed] == [1, 2]
    reopened.close()


@pytest.mark.asyncio
async def test_durable_lifecycle_publish_rolls_back_when_materialization_fails(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    session = repository.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
        )
    )
    run = repository.create_run(session.id, CreateRunRequest(message="finish atomically"))
    repository.start_run(session.id, run.id)
    bus = DurableEventBus(repository)

    def fail_materialization(event):
        if event.event == "run.completed":
            raise RuntimeError("simulated crash after append")
        return original_materialize(event)

    original_materialize = repository._materialize_run_lifecycle_event
    monkeypatch.setattr(repository, "_materialize_run_lifecycle_event", fail_materialization)

    with pytest.raises(RuntimeError, match="simulated crash"):
        await bus.publish(
            Event(
                event="run.completed",
                session_id=session.id,
                run_id=run.id,
                data={"returncode": 0},
            )
        )
    repository.close()

    reopened = open_sqlite_repository(db_path)
    try:
        events = reopened.list_events(session_id=session.id, run_id=run.id)
        assert [event.event for event in events] == []
        assert reopened.get_run(session.id, run.id).status == "failed"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_durable_event_bus_reports_max_sequence(tmp_path) -> None:
    repository = open_sqlite_repository(tmp_path / "harness.db")
    bus = DurableEventBus(repository)

    assert await bus.max_sequence() == 0
    assert await bus.max_sequence(session_id="ses_a") == 0

    await bus.publish(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    await bus.publish(Event(event="run.started", session_id="ses_b", run_id="run_b", data={}))
    await bus.publish(Event(event="message", session_id="ses_a", run_id="run_a", data={}))

    assert await bus.max_sequence() == 3
    assert await bus.max_sequence(session_id="ses_a") == 3
    assert await bus.max_sequence(session_id="ses_b") == 2
    assert await bus.max_sequence(session_id="ses_missing") == 0
    repository.close()


@pytest.mark.asyncio
async def test_durable_subscribe_has_no_gap_between_replay_and_live(tmp_path) -> None:
    repository = open_sqlite_repository(tmp_path / "harness.db")
    bus = DurableEventBus(repository)
    await bus.publish(Event(event="session.updated", session_id="ses_a", data={}))

    subscription = bus.subscribe(after=0, session_id="ses_a")
    replayed = await subscription.__anext__()
    live_event = asyncio.create_task(subscription.__anext__())
    published = await bus.publish(Event(event="message", session_id="ses_a", data={}))

    try:
        assert replayed.sequence == 1
        assert await asyncio.wait_for(live_event, timeout=1) == published
    finally:
        await subscription.aclose()
        repository.close()


# --- Phase 3: bus.publish becomes the single materialization point -----------


@pytest.mark.asyncio
async def test_durable_publish_materializes_run_lifecycle(tmp_path) -> None:
    """Phase 3: ``DurableEventBus.publish`` is the single materialization
    point. A ``run.started`` event published through the bus must flip
    the run's status to ``running`` without any other call doing it."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        bus = DurableEventBus(repository)
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        assert run.status == "queued"

        await bus.publish(
            Event(event="run.started", session_id=session.id, run_id=run.id, data={})
        )

        after = repository.get_run(session.id, run.id)
        assert after.status == "running"
        assert after.started_at is not None
    finally:
        repository.close()


@pytest.mark.asyncio
async def test_durable_publish_materializes_run_usage(tmp_path) -> None:
    """``run.usage`` published through the bus must apply additively to
    ``Run.usage`` and reflect in ``Session.stats.tokens``. The
    architectural unification means the side effect happens exactly
    once per publish — no double-counting."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        bus = DurableEventBus(repository)
        session = repository.create_session(
            CreateSessionRequest(
                backend="claude-code",
                model="claude-opus-4-7",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))

        await bus.publish(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={"usage": {"input": 6, "output": 4, "cache_read": 18, "cache_creation": 21}},
            )
        )

        after_run = repository.get_run(session.id, run.id)
        assert after_run.usage.input == 6
        assert after_run.usage.output == 4
        assert after_run.usage.cache_read == 18
        assert after_run.usage.cache_creation == 21

        after_session = repository.get_session(session.id)
        assert after_session.stats.tokens.get("input") == 6
        assert after_session.stats.tokens.get("output") == 4
    finally:
        repository.close()


@pytest.mark.asyncio
async def test_durable_publish_does_not_double_count_run_usage(tmp_path) -> None:
    """Regression for Falcon's PR #11 finding.

    Pre-Phase-3, ``run.usage`` was materialized in BOTH
    ``append_event`` and ``materialize_event``. The observer published
    a usage event through the bus (insert + append_event materialize)
    then separately called ``materialize_event`` for the same event,
    doubling the applied usage.

    Phase 3's structural fix: ``bus.publish`` is the single
    materialization point. One publish → one materialize → the
    advertised numbers land verbatim in ``Run.usage``."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        bus = DurableEventBus(repository)
        session = repository.create_session(
            CreateSessionRequest(
                backend="claude-code",
                model="claude-opus-4-7",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))

        # Single publish — should NOT double-apply.
        await bus.publish(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={"usage": {"input": 6, "output": 4, "cache_read": 18, "cache_creation": 21}},
            )
        )

        after = repository.get_run(session.id, run.id)
        # NOT (12, 8, 36, 42) — that was Falcon's bug.
        assert (
            after.usage.input,
            after.usage.output,
            after.usage.cache_read,
            after.usage.cache_creation,
        ) == (6, 4, 18, 21)
    finally:
        repository.close()


@pytest.mark.asyncio
async def test_durable_publish_propagates_session_not_found_for_buffering(tmp_path) -> None:
    """When ``bus.publish`` is asked to materialize a message event for
    a session that doesn't exist yet (rollout fired before
    ``POST /v1/sessions``), the bus inserts the event row (replay-safe)
    and raises so the caller can buffer for later flush. The exception
    must carry the published event so the buffer keeps the assigned
    sequence."""
    from agent_harness.repository import (
        MaterializationDeferred,
        SessionNotFoundError,
    )

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        bus = DurableEventBus(repository)
        message_event = Event(
            event="message",
            session_id="ses_does_not_exist",
            data={"message": {"role": "user", "blocks": [{"type": "text", "text": "hi"}]}},
        )

        with pytest.raises((MaterializationDeferred, SessionNotFoundError)) as excinfo:
            await bus.publish(message_event)

        # The exception carries the published event so the caller's
        # buffer keeps the sequence assigned by append_event.
        deferred = excinfo.value
        published = getattr(deferred, "event", None)
        assert published is not None
        assert published.sequence is not None
        assert published.event == "message"

        # Event row is in the durable store so replay still works.
        events = repository.list_events(session_id="ses_does_not_exist")
        assert any(e.event == "message" for e in events)
    finally:
        repository.close()
