import asyncio

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
async def test_replay_can_filter_by_session_and_run() -> None:
    bus = InMemoryEventBus()
    await bus.publish(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    await bus.publish(Event(event="run.started", session_id="ses_b", run_id="run_b", data={}))
    await bus.publish(Event(event="message", session_id="ses_a", run_id="run_a", data={}))

    session_events = await bus.replay(session_id="ses_a")
    run_events = await bus.replay(run_id="run_b")

    assert [event.sequence for event in session_events] == [1, 3]
    assert [event.sequence for event in run_events] == [2]
