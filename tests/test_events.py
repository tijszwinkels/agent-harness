import asyncio
import contextlib

import pytest

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event


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
