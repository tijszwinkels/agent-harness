from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol

from agent_harness.models import Event


class EventRepository(Protocol):
    def append_event(self, event: Event) -> Event:
        pass

    def list_events(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        after: int = 0,
    ) -> list[Event]:
        pass

    def max_sequence(self, *, session_id: str | None = None) -> int:
        pass

    def materialize_event(self, event: Event, *, store_event: bool = True) -> None:
        pass


class InMemoryEventBus:
    stores_events = False

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

    async def max_sequence(self, *, session_id: str | None = None) -> int:
        async with self._lock:
            if session_id is None:
                return self._next_seq - 1
            return max(
                (
                    event.sequence or 0
                    for event in self._history
                    if event.session_id == session_id
                ),
                default=0,
            )

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
        keepalive_seconds: float | None = None,
    ) -> AsyncIterator[Event | None]:
        # When ``keepalive_seconds`` is set the generator yields ``None`` after
        # each interval of bus silence, letting the SSE layer inject a comment
        # frame so the wire never goes longer than the interval without bytes.
        # The timeout lives INSIDE the generator on purpose: cancelling
        # ``__anext__`` from outside would propagate into the ``queue.get()``
        # await and tear down the subscription via the ``finally`` block.
        if keepalive_seconds is not None and keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be positive when provided")
        replay, queue = await self._register(after, session_id=session_id)
        loop = asyncio.get_running_loop()
        try:
            for event in replay:
                yield event

            deadline = (
                loop.time() + keepalive_seconds
                if keepalive_seconds is not None
                else None
            )
            while True:
                if deadline is None:
                    event = await queue.get()
                else:
                    # Keepalive timeout is bound to the SUBSCRIBER's last
                    # yield, not the bus's last publish. Without this,
                    # high-volume traffic filtered out by ``_matches``
                    # (other sessions / events at or below ``after``)
                    # would keep ``queue.get`` returning instantly and
                    # the keepalive would never fire — leaving the very
                    # stale-cursor failure mode we're trying to detect
                    # invisible to the client.
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        yield None
                        deadline = loop.time() + keepalive_seconds
                        continue
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        yield None
                        deadline = loop.time() + keepalive_seconds
                        continue
                if _matches(event, after=after, session_id=session_id):
                    if deadline is not None:
                        deadline = loop.time() + keepalive_seconds
                    yield event
        finally:
            with suppress(RuntimeError):
                await self._unregister(queue)


class DurableEventBus:
    stores_events = True

    def __init__(self, repository: EventRepository) -> None:
        self._repository = repository
        self._lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[Event]] = set()

    async def publish(self, event: Event) -> Event:
        async with self._lock:
            published = self._repository.append_event(event)
            subscribers = tuple(self._subscribers)

        for subscriber in subscribers:
            subscriber.put_nowait(published)

        return published

    async def replay(self, after: int = 0, *, session_id: str | None = None) -> list[Event]:
        async with self._lock:
            return self._repository.list_events(after=after, session_id=session_id)

    async def max_sequence(self, *, session_id: str | None = None) -> int:
        async with self._lock:
            return self._repository.max_sequence(session_id=session_id)

    async def _register(
        self,
        after: int,
        *,
        session_id: str | None,
    ) -> tuple[list[Event], asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        async with self._lock:
            replay = self._repository.list_events(after=after, session_id=session_id)
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
        keepalive_seconds: float | None = None,
    ) -> AsyncIterator[Event | None]:
        if keepalive_seconds is not None and keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be positive when provided")
        replay, queue = await self._register(after, session_id=session_id)
        loop = asyncio.get_running_loop()
        try:
            for event in replay:
                yield event

            deadline = (
                loop.time() + keepalive_seconds
                if keepalive_seconds is not None
                else None
            )
            while True:
                if deadline is None:
                    event = await queue.get()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        yield None
                        deadline = loop.time() + keepalive_seconds
                        continue
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        yield None
                        deadline = loop.time() + keepalive_seconds
                        continue
                if _matches(event, after=after, session_id=session_id):
                    if deadline is not None:
                        deadline = loop.time() + keepalive_seconds
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
