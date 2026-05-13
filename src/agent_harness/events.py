from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from agent_harness.models import Event


class InMemoryEventBus:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_seq = 1
        self._history: list[Event] = []
        self._subscribers: set[asyncio.Queue[Event]] = set()

    async def publish(self, event: Event) -> Event:
        async with self._lock:
            published = event.with_sequence(self._next_seq)
            self._next_seq += 1
            self._history.append(published)
            subscribers = tuple(self._subscribers)

        for subscriber in subscribers:
            subscriber.put_nowait(published)

        return published

    async def replay(self, after: int = 0, *, session_id: str | None = None) -> list[Event]:
        async with self._lock:
            return [
                event
                for event in self._history
                if _matches(event, after=after, session_id=session_id)
            ]

    async def _register(
        self,
        after: int,
        *,
        session_id: str | None,
    ) -> tuple[list[Event], asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        async with self._lock:
            replay = [
                event
                for event in self._history
                if _matches(event, after=after, session_id=session_id)
            ]
            self._subscribers.add(queue)
        return replay, queue

    async def _unregister(self, queue: asyncio.Queue[Event]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def subscribe(
        self,
        after: int = 0,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        replay, queue = await self._register(after, session_id=session_id)
        try:
            for event in replay:
                yield event

            while True:
                event = await queue.get()
                if _matches(event, after=after, session_id=session_id):
                    yield event
        finally:
            with suppress(RuntimeError):
                await self._unregister(queue)


def _matches(event: Event, *, after: int, session_id: str | None) -> bool:
    if event.sequence is None or event.sequence <= after:
        return False
    if session_id is not None and event.session_id != session_id:
        return False
    return True
