"""Transcript observation must survive bad records and watcher failures, and
say so honestly when it can't (issue #46).

All transcripts here are synthetic. The 2026-11-04 failure: a Pi tool result
carried an unpaired surrogate (``"\\ud83d"``, half an emoji), its event could
not be serialized, and the exception stopped the watcher for every file while
``/health`` kept answering ok.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import LiveStateRequest, replace_lone_surrogates
from agent_harness.native_titles import native_name
from agent_harness.observer import (
    ExternalTranscriptObserver,
    TranscriptWatchService,
    pi_transcript_path,
)
from agent_harness.settings import ObserverSettings
from agent_harness.storage import open_sqlite_repository

CWD = "/home/demo/work"
UUID_A = "11111111-2222-4333-8444-555555555555"
UUID_B = "66666666-7777-4888-9999-aaaaaaaaaaaa"
START = datetime(2026, 11, 4, 19, 0, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def sid(uuid: str) -> str:
    return "ses_" + uuid.replace("-", "")


def header(uuid: str) -> str:
    return json.dumps({"type": "session", "version": 3, "id": uuid, "cwd": CWD}) + "\n"


MODEL = json.dumps({"type": "model_change", "provider": "openai-codex", "modelId": "gpt-6-astra"}) + "\n"
OWNER = json.dumps({"type": "custom", "id": "o1", "customType": "agent-harness.live-state-owner",
                    "data": {"source": "companion"}}) + "\n"


def user(text: str) -> str:
    return json.dumps({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}) + "\n"


def tool_result(call_id: str, text_json: str) -> str:
    # ``text_json`` is spliced raw so a test can write an escaped lone
    # surrogate exactly as a JSON writer does (``json.dumps`` would too).
    return (
        '{"type":"message","message":{"role":"toolResult","toolCallId":"' + call_id
        + '","toolName":"web_search","content":[{"type":"text","text":' + text_json + '}],"isError":false}}\n'
    )


def transcript(home: Path, uuid: str, *lines: str) -> Path:
    path = pi_transcript_path(CWD, "2026-11-04T19-00-00-000Z", uuid, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header(uuid) + MODEL + "".join(lines)).encode("utf-8"))
    return path


def append(path: Path, *lines: str) -> None:
    with path.open("ab") as handle:
        handle.write("".join(lines).encode("utf-8"))


@pytest.fixture
def sqlite_harness(tmp_path):
    clock = Clock()
    repository = open_sqlite_repository(tmp_path / "harness.db")
    repository.clock = clock
    bus = DurableEventBus(repository)
    observer = ExternalTranscriptObserver(bus, repository=repository, clock=clock)
    return repository, bus, observer, clock


def message_texts(repository, session_id: str) -> list[str]:
    out = []
    for message in repository.list_messages(session_id):
        for block in message.blocks:
            if block.type in ("text", "tool_result"):
                out.append(getattr(block, "text", None) or block.content)
    return out


# ---------------------------------------------------------------- unicode


def test_replace_lone_surrogates_keeps_valid_text() -> None:
    assert replace_lone_surrogates("plain ascii") == "plain ascii"
    assert replace_lone_surrogates("naïve 日本 😀") == "naïve 日本 😀"
    assert replace_lone_surrogates("cut \ud83d here") == "cut \ufffd here"
    assert replace_lone_surrogates("\ude00 low first") == "\ufffd low first"
    assert replace_lone_surrogates("\ud83d\ude00") == "😀"  # a pair still in UTF-16 form
    assert replace_lone_surrogates("\udc00\ud800") == "\ufffd\ufffd"  # reversed: not a pair
    assert native_name("title \ud83d") == "title \ufffd"


@pytest.mark.asyncio
async def test_lone_surrogates_in_tool_output_are_replaced_and_valid_text_kept(sqlite_harness, tmp_path) -> None:
    repository, bus, observer, _ = sqlite_harness
    path = transcript(
        tmp_path, UUID_A,
        tool_result("call_1", '"result \\ud83d cut"'),          # lone high (the incident)
        tool_result("call_2", '"\\ude00 lone low"'),            # lone low
        tool_result("call_3", '"pair \\ud83d\\ude00 ok"'),      # escaped valid pair
        user("emoji 😀 naïve ascii"),                           # raw UTF-8, non-BMP
        '{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_4",'
        '"name":"bash","arguments":{"cmd":"echo \\ud83d","nested":[{"q":"\\udfff"}]}}]}}\n',
    )
    before = path.read_bytes()

    await observer.tail_file(path)

    texts = message_texts(repository, sid(UUID_A))
    assert texts == ["result \ufffd cut", "\ufffd lone low", "pair 😀 ok", "emoji 😀 naïve ascii"]
    tool_use = repository.list_messages(sid(UUID_A))[-1].blocks[0]
    assert tool_use.input == {"cmd": "echo \ufffd", "nested": [{"q": "\ufffd"}]}
    assert observer.observation_snapshot()["rejected_records"] == 0
    # Everything stored and replayable, and the transcript itself untouched.
    assert len(await bus.replay()) >= 6
    assert path.read_bytes() == before
    assert observer.tracked_offsets()[path] == len(before)


# ------------------------------------------------------- bad-record containment


def hostile(call_id: str) -> str:
    # A lone surrogate in a routing identifier (the tool call id) is not
    # rewritten; the record can't be stored and is rejected as a whole.
    return tool_result(call_id + "\\ud83d", '"payload"')


@pytest.mark.asyncio
async def test_bad_record_is_rejected_and_later_records_and_files_continue(sqlite_harness, tmp_path, caplog) -> None:
    repository, _, observer, _ = sqlite_harness
    path_a = transcript(tmp_path, UUID_A, hostile("call_x"), user("after the bad one"))
    path_b = transcript(tmp_path, UUID_B, user("other file"))

    with caplog.at_level(logging.WARNING, logger="agent_harness.observer"):
        await observer.tail_file(path_a)
        await observer.tail_file(path_b)

    assert message_texts(repository, sid(UUID_A)) == ["after the bad one"]
    assert message_texts(repository, sid(UUID_B)) == ["other file"]
    assert observer.tracked_offsets()[path_a] == path_a.stat().st_size
    snapshot = observer.observation_snapshot()
    assert snapshot["rejected_records"] == 1
    assert snapshot["retrying_files"] == 0
    rejected = [r for r in caplog.records if "Rejected transcript record" in r.getMessage()]
    assert len(rejected) == 1
    # Metadata only: never the record's content.
    assert "payload" not in rejected[0].getMessage()
    assert "call_x" not in rejected[0].getMessage()


@pytest.mark.asyncio
async def test_repeated_bad_records_are_rate_limited(sqlite_harness, tmp_path, caplog) -> None:
    repository, _, observer, clock = sqlite_harness
    path = transcript(tmp_path, UUID_A, *[hostile(f"c{i}") for i in range(40)], user("survivor"))

    with caplog.at_level(logging.WARNING, logger="agent_harness.observer"):
        await observer.tail_file(path)
        clock.advance(61)
        append(path, hostile("late"))
        await observer.tail_file(path)

    messages = [r.getMessage() for r in caplog.records]
    assert sum("Rejected transcript record" in m for m in messages) == 6  # 5 + 1 in the next window
    assert any("Suppressed 35 more" in m for m in messages)
    assert observer.observation_snapshot()["rejected_records"] == 41
    assert message_texts(repository, sid(UUID_A)) == ["survivor"]


# ------------------------------------------------------- cursor / retry semantics


@pytest.mark.asyncio
async def test_partial_line_keeps_its_cursor(sqlite_harness, tmp_path) -> None:
    repository, _, observer, _ = sqlite_harness
    path = transcript(tmp_path, UUID_A)
    await observer.tail_file(path)
    complete = path.stat().st_size
    line = user("split write")
    append(path, line[:15])
    await observer.tail_file(path)
    assert observer.tracked_offsets()[path] == complete
    append(path, line[15:])
    await observer.tail_file(path)
    assert message_texts(repository, sid(UUID_A)) == ["split write"]
    assert observer.tracked_offsets()[path] == path.stat().st_size


class FlakyBus:
    """Wraps a durable bus; fails chosen publish calls like a locked DB."""

    def __init__(self, inner: DurableEventBus, fail_on: set[int], error: BaseException | None = None) -> None:
        self._inner = inner
        self.stores_events = inner.stores_events
        self.fail_on = fail_on
        self.error = error or sqlite3.OperationalError("database is locked")
        self.calls = 0

    async def publish(self, event):
        self.calls += 1
        if self.calls in self.fail_on:
            raise self.error
        return await self._inner.publish(event)

    async def replay(self, *args, **kwargs):
        return await self._inner.replay(*args, **kwargs)


@pytest.mark.asyncio
async def test_storage_failure_keeps_cursor_and_retry_resumes_without_duplicates(tmp_path) -> None:
    clock = Clock()
    repository = open_sqlite_repository(tmp_path / "harness.db")
    repository.clock = clock
    durable = DurableEventBus(repository)
    # Calls: 1 = session.updated (header+model), 2 = user message,
    # 3 = second line's first event ... fail the multi-event line part-way.
    bus = FlakyBus(durable, fail_on=set())
    observer = ExternalTranscriptObserver(bus, repository=repository, clock=clock)
    path = transcript(tmp_path, UUID_A, user("first"))
    await observer.tail_file(path)
    cursor = observer.tracked_offsets()[path]
    events_before = len(await durable.replay())

    # A pi assistant record publishes a message and a run-less usage event is
    # dropped; give it a model change too so the line carries two events.
    assistant = json.dumps({"type": "message", "message": {
        "role": "assistant", "model": "gpt-6-astra-2", "provider": "openai-codex",
        "content": [{"type": "text", "text": "multi"}], "stopReason": "stop"}}) + "\n"
    append(path, assistant, user("next"))
    from agent_harness.observer import parse_transcript_line, transcript_identity_from_path
    parsed = parse_transcript_line(assistant, identity=transcript_identity_from_path(path, home=tmp_path))
    assert len(parsed) >= 2

    bus.fail_on = {bus.calls + 2}  # second event of the assistant line
    await observer.tail_file(path)
    assert observer.tracked_offsets()[path] == cursor  # not skipped
    assert observer.observation_snapshot()["retrying_files"] == 1
    assert observer.observation_snapshot()["last_error"] == "OperationalError"

    # Backing off: neither a new write nor an early retry tick re-reads it.
    published = await observer.tail_file(path)
    assert published == []
    assert await observer.retry_pending() == []

    clock.advance(5)
    await observer.retry_pending()
    assert observer.observation_snapshot()["retrying_files"] == 0
    assert observer.tracked_offsets()[path] == path.stat().st_size
    assert message_texts(repository, sid(UUID_A)) == ["first", "multi", "next"]
    replayed = await durable.replay()
    kinds = [(e.event, json.dumps(e.data.get("message", {}).get("blocks"), sort_keys=True)) for e in replayed[events_before:]]
    assert len(kinds) == len(set(kinds)), "an event of the retried line was published twice"


@pytest.mark.asyncio
async def test_retry_backoff_grows_and_never_skips(tmp_path) -> None:
    clock = Clock()
    repository = open_sqlite_repository(tmp_path / "harness.db")
    repository.clock = clock
    durable = DurableEventBus(repository)
    bus = FlakyBus(durable, fail_on=set(range(1, 100)))
    observer = ExternalTranscriptObserver(bus, repository=repository, clock=clock)
    path = transcript(tmp_path, UUID_A, user("waits"))

    await observer.tail_file(path)
    attempts = 1
    for delay in (5, 10, 20, 40, 80, 120, 120):
        clock.advance(delay - 1)
        await observer.retry_pending()
        assert bus.calls == attempts  # not due yet
        clock.advance(1)
        await observer.retry_pending()
        attempts += 1
        assert bus.calls == attempts
    assert observer.tracked_offsets().get(path, 0) == 0
    bus.fail_on = set()
    clock.advance(120)
    await observer.retry_pending()
    assert message_texts(repository, sid(UUID_A)) == ["waits"]


@pytest.mark.asyncio
async def test_cancellation_mid_line_does_not_lose_the_line(sqlite_harness, tmp_path) -> None:
    repository, bus, observer, _ = sqlite_harness
    path = transcript(tmp_path, UUID_A, user("kept"))
    flaky = FlakyBus(bus, fail_on={2}, error=asyncio.CancelledError())
    observer._event_bus = flaky  # noqa: SLF001 — inject a cancellation mid-line
    with pytest.raises(asyncio.CancelledError):
        await observer.tail_file(path)
    observer._event_bus = bus  # noqa: SLF001
    await observer.tail_file(path)
    assert message_texts(repository, sid(UUID_A)) == ["kept"]
    events = await bus.replay()
    assert sum(e.event == "message" for e in events) == 1


@pytest.mark.asyncio
async def test_restart_resumes_from_saved_cursor_without_republishing(tmp_path) -> None:
    db = tmp_path / "harness.db"
    repository = open_sqlite_repository(db)
    bus = DurableEventBus(repository)
    path = transcript(tmp_path, UUID_A, hostile("bad"), user("one"))
    await ExternalTranscriptObserver(bus, repository=repository).tail_file(path)
    count = len(await bus.replay())
    repository.close()

    repository = open_sqlite_repository(db)
    bus = DurableEventBus(repository)
    observer = ExternalTranscriptObserver(bus, repository=repository)
    assert await observer.tail_file(path) == []
    append(path, user("two"))
    await observer.tail_file(path)
    assert message_texts(repository, sid(UUID_A)) == ["one", "two"]
    assert len(await bus.replay()) == count + 1


# ---------------------------------------------------------- watch service


class ScriptedWatcher:
    """Each call is one watcher run: yields its batches, then raises,
    finishes, or waits for ``stop_event``."""

    def __init__(self, runs):
        self.runs = list(runs)
        self.calls = 0

    def __call__(self, *_roots, stop_event=None, **_kwargs):
        run = self.runs[min(self.calls, len(self.runs) - 1)]
        self.calls += 1
        return self._run(run, stop_event)

    async def _run(self, run, stop_event):
        for item in run:
            if isinstance(item, BaseException):
                raise item
            if item == "wait":
                await stop_event.wait()
                return
            if callable(item):
                item()
                continue
            yield item


def service_for(observer, roots, watcher, clock, **kwargs) -> TranscriptWatchService:
    return TranscriptWatchService(
        roots=roots, observer=observer, watcher=watcher, clock=clock,
        restart_base_seconds=0.01, restart_max_seconds=0.02, freshness_interval_seconds=3600, **kwargs,
    )


@pytest.mark.asyncio
async def test_watcher_failure_budget_ends_in_visible_failed_state(sqlite_harness, tmp_path, caplog) -> None:
    _, _, observer, clock = sqlite_harness
    watcher = ScriptedWatcher([[RuntimeError("inotify gone")]])
    service = service_for(observer, [tmp_path], watcher, clock)
    with caplog.at_level(logging.WARNING, logger="agent_harness.observer"):
        await asyncio.wait_for(service.watch_forever(stop_event=asyncio.Event()), 5)
    assert watcher.calls == 4  # first run + 3 restarts, then it gives up
    health = service.health()
    assert (health["state"], health["complete"], health["restarts"]) == ("failed", False, 3)
    assert health["last_error"] == "RuntimeError"
    assert sum("restarting in" in r.getMessage() for r in caplog.records) == 3
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_progress_resets_the_failure_budget(sqlite_harness, tmp_path) -> None:
    _, _, observer, clock = sqlite_harness
    path = transcript(tmp_path, UUID_A)
    runs = []
    for i in range(6):  # more failures than the budget, each after real progress
        runs.append([lambda i=i: append(path, user(f"m{i}")), {("modified", str(path))}, RuntimeError("blip")])
    runs.append(["wait"])
    watcher = ScriptedWatcher(runs)
    service = service_for(observer, [tmp_path], watcher, clock)
    stop = asyncio.Event()
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if watcher.calls == 7:
            break
    assert service.health()["state"] in ("healthy", "catching_up")
    stop.set()
    await asyncio.wait_for(task, 5)
    assert service.health()["state"] == "stopped"
    assert service.health()["restarts"] == 6


@pytest.mark.asyncio
async def test_restart_catches_up_missed_writes_from_cursors(sqlite_harness, tmp_path) -> None:
    repository, _, observer, clock = sqlite_harness
    old = transcript(tmp_path, UUID_A, user("seen"))
    await observer.tail_file(old)
    # Untracked history from long before the failure stays untouched.
    history_uuid = "99999999-8888-4777-8666-555555555555"
    history = transcript(tmp_path, history_uuid, user("ancient"))
    import os
    os.utime(history, (START.timestamp() - 86400, START.timestamp() - 86400))

    def gap_writes() -> None:
        append(old, user("missed while down"))
        transcript(tmp_path, UUID_B, OWNER, user("new during the gap"))

    stop = asyncio.Event()
    watcher = ScriptedWatcher([[gap_writes, RuntimeError("watch died")], ["wait"]])
    clock.now = datetime.fromtimestamp(old.stat().st_mtime, UTC)
    service = service_for(observer, [tmp_path], watcher, clock)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if service.health()["state"] == "healthy" and watcher.calls == 2:
            break
    assert service.health()["state"] == "healthy"
    assert service.health()["complete"] is True
    assert message_texts(repository, sid(UUID_A)) == ["seen", "missed while down"]
    assert message_texts(repository, sid(UUID_B)) == ["new during the gap"]
    assert repository.live_state_owner(sid(UUID_B)) == "companion"
    assert not repository.has_session(sid(history_uuid))
    stop.set()
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_quiet_watcher_is_healthy_and_stop_during_backoff_is_clean(sqlite_harness, tmp_path) -> None:
    _, _, observer, clock = sqlite_harness
    stop = asyncio.Event()
    service = service_for(observer, [tmp_path], ScriptedWatcher([["wait"]]), clock)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    await asyncio.sleep(0.05)
    clock.advance(3600)  # an hour without a single transcript write
    assert service.health()["state"] == "healthy"
    stop.set()
    await asyncio.wait_for(task, 5)

    stop = asyncio.Event()
    service = TranscriptWatchService(roots=[tmp_path], observer=observer, clock=clock,
                                     watcher=ScriptedWatcher([[RuntimeError("x")]]),
                                     restart_base_seconds=60, freshness_interval_seconds=3600)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    await asyncio.sleep(0.05)
    assert service.health()["state"] == "degraded"
    stop.set()
    await asyncio.wait_for(task, 5)
    assert service.health()["state"] == "stopped"


@pytest.mark.asyncio
async def test_retry_tick_runs_in_the_freshness_loop(sqlite_harness, tmp_path) -> None:
    repository, bus, observer, clock = sqlite_harness
    flaky = FlakyBus(bus, fail_on={1})
    observer._event_bus = flaky  # noqa: SLF001
    path = transcript(tmp_path, UUID_A, user("eventually"))
    stop = asyncio.Event()
    watcher = ScriptedWatcher([[{("modified", str(path))}, "wait"]])
    service = TranscriptWatchService(roots=[tmp_path], observer=observer, watcher=watcher, clock=clock,
                                     freshness_interval_seconds=0.01)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    await asyncio.sleep(0.05)
    assert service.health()["state"] == "degraded"
    assert service.health()["complete"] is False
    clock.advance(5)
    for _ in range(100):
        await asyncio.sleep(0.01)
        if service.health()["state"] == "healthy":
            break
    assert service.health()["state"] == "healthy"
    assert message_texts(repository, sid(UUID_A)) == ["eventually"]
    stop.set()
    await asyncio.wait_for(task, 5)


# ------------------------------------------------------------------ API


class FakeService:
    def __init__(self, state: str) -> None:
        self.state = state

    def health(self):
        return {"state": self.state, "complete": self.state == "healthy"}

    async def watch_forever(self, *, stop_event=None):
        await stop_event.wait()


@pytest.mark.parametrize(
    ("state", "code", "status"),
    [("healthy", 200, "ok"), ("catching_up", 200, "degraded"), ("degraded", 200, "degraded"),
     ("failed", 503, "failed")],
)
def test_health_reflects_configured_observer(tmp_path, state, code, status) -> None:
    service = FakeService(state)
    app = create_app(observer_settings=ObserverSettings.from_roots([tmp_path]),
                     watch_service_factory=lambda **_: service)
    with TestClient(app) as client:
        for url in ("/health", "/v1/health"):
            response = client.get(url)
            assert response.status_code == code
            assert response.json()["status"] == status
            assert response.json()["observation"]["state"] == state
        # Listing stays available whatever the observer does.
        listing = client.get("/v1/sessions")
        assert listing.status_code == 200
        assert listing.json()["observation"] == {"state": state, "complete": state == "healthy"}


def test_health_reports_a_dead_watch_task_as_failed(tmp_path) -> None:
    class Dies:
        def health(self):
            return {"state": "healthy", "complete": True}

        async def watch_forever(self, *, stop_event=None):
            await asyncio.sleep(0.01)
            raise RuntimeError("bug")

    app = create_app(observer_settings=ObserverSettings.from_roots([tmp_path]),
                     watch_service_factory=lambda **_: Dies())
    with pytest.raises(RuntimeError, match="bug"):  # shutdown re-raises, as before
        with TestClient(app) as client:
            import time
            time.sleep(0.1)
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["observation"] == {"state": "failed", "complete": False}


def test_disabled_observer_is_explicit_and_ok() -> None:
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok", "observation": {"state": "disabled", "complete": False}}
    body = client.get("/v1/sessions").json()
    assert body == {"data": [], "observation": {"state": "disabled", "complete": False}, "live_leases": {}}


@pytest.mark.asyncio
async def test_new_session_is_discovered_and_its_lease_counts_through_long_silence(tmp_path) -> None:
    """New Pi session with an owner entry → discovered → busy accepted (no
    404) → renewed past 120 s of transcript silence: listed as leased
    while ``updated_at`` stays old → idle / lapse removes the lease."""
    clock = Clock()
    repository = open_sqlite_repository(tmp_path / "harness.db")
    repository.clock = clock
    bus = DurableEventBus(repository)
    observer = ExternalTranscriptObserver(bus, repository=repository, clock=clock)
    path = transcript(tmp_path, UUID_A, OWNER, user("long task"))
    client = TestClient(create_app(repository=repository, event_bus=bus))
    url = f"/v1/sessions/{sid(UUID_A)}/live-state"
    body = {"source": "companion", "producer": "p1", "sequence": 1, "state": "busy", "lease_seconds": 45}

    assert client.put(url, json=body).status_code == 404  # not discovered yet
    await observer.tail_file(path)
    assert client.put(url, json=body).status_code == 200

    updated_at = client.get("/v1/sessions").json()["data"][0]["updated_at"]
    for sequence in range(2, 14):  # 3 minutes of 15 s heartbeats, no transcript writes
        clock.advance(15)
        await observer.freshness_tick()
        assert client.put(url, json={**body, "sequence": sequence}).json()["busy"] is True
    listing = client.get("/v1/sessions").json()
    row = listing["data"][0]
    assert row["status"] == "running"
    assert row["updated_at"] == updated_at  # renewals don't touch the row
    lease = listing["live_leases"][sid(UUID_A)]
    assert lease["state"] == "busy" and lease["source"] == "companion"
    assert datetime.fromisoformat(lease["expires_at"]) == clock.now + timedelta(seconds=45)

    # Producer disappears: the lease lapses and is no longer listed.
    clock.advance(46)
    assert client.get("/v1/sessions").json()["live_leases"] == {}

    # Idle settles the turn and lists no lease either.
    assert client.put(url, json={**body, "sequence": 30}).json()["busy"] is True
    assert client.put(url, json={**body, "sequence": 31, "state": "idle"}).json()["status"] == "idle"
    assert client.get("/v1/sessions").json()["live_leases"] == {}
    repository.close()


def test_live_leases_with_in_memory_repository() -> None:
    from agent_harness.models import Project, Session
    from agent_harness.repository import InMemoryRepository

    repository = InMemoryRepository()
    clock = Clock()
    repository.clock = clock
    repository.upsert_session(Session(id=sid(UUID_A), backend="pi", origin="external", status="idle",
                                      project=Project(path=CWD, name="work")))
    repository.note_live_state_owner(sid(UUID_A), "companion")
    repository.apply_live_state(sid(UUID_A), LiveStateRequest(source="companion", producer="p", sequence=1,
                                                              state="busy", lease_seconds=30))
    client = TestClient(create_app(repository=repository, event_bus=InMemoryEventBus()))
    assert set(client.get("/v1/sessions").json()["live_leases"]) == {sid(UUID_A)}
    clock.advance(31)
    assert client.get("/v1/sessions").json()["live_leases"] == {}


# ------------------------------------------------- review round 1 (Aster)


@pytest.mark.asyncio
async def test_materialization_failure_after_append_is_not_appended_twice(sqlite_harness, tmp_path, monkeypatch) -> None:
    """Finding 1: the durable bus used to commit the event row before
    materializing; a failed materialization left the row, and the retry
    appended it again."""
    repository, bus, observer, clock = sqlite_harness
    path = transcript(tmp_path, UUID_A, user("first"))
    await observer.tail_file(path)
    append(path, user("second"))
    original = repository._materialize_message_event
    failures = iter([True])

    def flaky(event, **kwargs):
        if next(failures, False):
            raise sqlite3.OperationalError("database is locked")
        return original(event, **kwargs)

    monkeypatch.setattr(repository, "_materialize_message_event", flaky)
    await observer.tail_file(path)
    assert observer.observation_snapshot()["retrying_files"] == 1
    clock.advance(5)
    await observer.retry_pending()

    messages = [e for e in await bus.replay() if e.event == "message"]
    ids = [e.data["message"]["id"] for e in messages]
    assert len(ids) == len(set(ids)) == 2
    assert message_texts(repository, sid(UUID_A)) == ["first", "second"]
    assert repository.get_session(sid(UUID_A)).stats.messages == 2


@pytest.mark.asyncio
async def test_failed_running_kick_is_retried_without_republishing(sqlite_harness, tmp_path, monkeypatch) -> None:
    """Finding 2: the event counted as done before its running kick; a
    failed kick advanced the cursor and left the session idle."""
    repository, bus, observer, clock = sqlite_harness
    path = transcript(tmp_path, UUID_A, user("first"))
    await observer.tail_file(path)
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(sid(UUID_A)).status == "idle"

    append(path, user("back to work"))
    original = observer._maybe_publish_status_flip
    failures = iter([True])

    async def flaky(*args, **kwargs):
        if next(failures, False):
            raise sqlite3.OperationalError("database is locked")
        return await original(*args, **kwargs)

    monkeypatch.setattr(observer, "_maybe_publish_status_flip", flaky)
    await observer.tail_file(path)
    assert repository.get_session(sid(UUID_A)).status == "idle"
    assert observer.observation_snapshot()["retrying_files"] == 1
    clock.advance(5)
    await observer.retry_pending()

    assert repository.get_session(sid(UUID_A)).status == "running"
    assert observer.observation_snapshot()["retrying_files"] == 0
    assert observer.tracked_offsets()[path] == path.stat().st_size
    assert [e.data["message"]["blocks"][0]["text"] for e in await bus.replay() if e.event == "message"] == [
        "first", "back to work"]


@pytest.mark.asyncio
async def test_deferred_materialization_failure_is_kept_and_retried(sqlite_harness, monkeypatch) -> None:
    repository, bus, observer, _ = sqlite_harness
    from agent_harness.models import Event, Message, Project, Session, TextBlock

    session_id = sid(UUID_A)
    message = Message(role="user", blocks=[TextBlock(text="early")])
    # Buffer an event as the observer does when the session doesn't exist yet.
    from agent_harness.repository import MaterializationDeferred
    try:
        await bus.publish(Event(event="message", session_id=session_id, data={"message": message.model_dump(mode="json")}))
    except MaterializationDeferred as deferred:
        observer._buffer_materialization(deferred.event)  # noqa: SLF001
    original = repository.materialize_event
    failures = iter([True])

    def flaky(event, **kwargs):
        if next(failures, False):
            raise sqlite3.OperationalError("database is locked")
        return original(event, **kwargs)

    monkeypatch.setattr(repository, "materialize_event", flaky)
    repository.upsert_session(Session(id=session_id, backend="pi", origin="external", status="idle",
                                      project=Project(path=CWD, name="work")))
    observer._flush_pending_materialization(session_id)  # noqa: SLF001 — what session.updated triggers
    assert observer.observation_snapshot()["retrying_files"] == 1
    assert message_texts(repository, session_id) == []
    await observer.retry_pending()
    assert observer.observation_snapshot()["retrying_files"] == 0
    assert message_texts(repository, session_id) == ["early"]


@pytest.mark.asyncio
async def test_stale_catch_up_cannot_clear_a_newer_failure_window(sqlite_harness, tmp_path) -> None:
    """Finding 3: a catch-up started for one failure kept running through
    the next failure's backoff and cleared the newer gap."""
    repository, _, observer, clock = sqlite_harness
    seen = transcript(tmp_path, UUID_A, user("seen"))
    await observer.tail_file(seen)
    clock.now = datetime.fromtimestamp(seen.stat().st_mtime, UTC)

    def write_new_session() -> None:
        transcript(tmp_path, UUID_B, OWNER, user("born between failures"))

    stop = asyncio.Event()
    watcher = ScriptedWatcher([
        [RuntimeError("first")],
        # Restarted, and fails again at once while its catch-up runs. That
        # catch-up used to finish during the backoff and clear the gap ...
        [RuntimeError("second")],
        # ... so a session written before the next watcher was up (modelled
        # here as no change event for it) was never caught up.
        [write_new_session, "wait"],
    ])
    service = TranscriptWatchService(roots=[tmp_path], observer=observer, watcher=watcher, clock=clock,
                                     restart_base_seconds=0.05, freshness_interval_seconds=3600)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    for _ in range(300):
        await asyncio.sleep(0.01)
        if watcher.calls == 3 and service.health()["state"] == "healthy":
            break
    assert service.health()["state"] == "healthy"
    assert message_texts(repository, sid(UUID_B)) == ["born between failures"]
    stop.set()
    await asyncio.wait_for(task, 5)


def test_catch_up_scan_reports_io_errors_but_not_vanished_entries(tmp_path) -> None:
    from agent_harness.observer import _catch_up_candidates

    (tmp_path / "gone.jsonl").symlink_to(tmp_path / "missing-target")  # stat: FileNotFoundError
    found, errors = _catch_up_candidates([tmp_path], {}, 0)
    assert (found, errors) == ([], [])

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "x.jsonl").write_text("{}\n")
    locked.chmod(0)
    try:
        _, errors = _catch_up_candidates([tmp_path], {}, 0)
    finally:
        locked.chmod(0o700)
    assert errors == ["PermissionError"]
    _, errors = _catch_up_candidates([tmp_path / "no-such-root"], {}, 0)
    assert errors == ["FileNotFoundError"]


@pytest.mark.asyncio
async def test_incomplete_catch_up_scan_stays_degraded_and_rescans(sqlite_harness, tmp_path, monkeypatch) -> None:
    """Finding 4: a scan error used to be swallowed, skipping a grown
    transcript while reporting complete coverage."""
    repository, _, observer, clock = sqlite_harness
    path = transcript(tmp_path, UUID_A, user("seen"))
    await observer.tail_file(path)

    real_stat = Path.stat
    failing = {"left": 1}

    def flaky_stat(self, *args, **kwargs):
        if self == path and failing["left"]:
            failing["left"] -= 1
            raise PermissionError(13, "synthetic", str(self))
        return real_stat(self, *args, **kwargs)

    def grow_then_fail() -> None:
        append(path, user("missed while down"))
        monkeypatch.setattr(Path, "stat", flaky_stat)

    stop = asyncio.Event()
    watcher = ScriptedWatcher([[grow_then_fail, RuntimeError("watch died")], ["wait"]])
    service = TranscriptWatchService(roots=[tmp_path], observer=observer, watcher=watcher, clock=clock,
                                     restart_base_seconds=0.05, freshness_interval_seconds=3600)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    states = set()
    for _ in range(300):
        await asyncio.sleep(0.01)
        health = service.health()
        states.add(health["state"])
        if watcher.calls == 2 and health["state"] == "healthy":
            break
    assert "degraded" in states  # the incomplete scan was never certified
    assert service.health()["state"] == "healthy"
    assert service.health()["last_error"] == "PermissionError"
    assert message_texts(repository, sid(UUID_A)) == ["seen", "missed while down"]
    stop.set()
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_stop_ends_a_catch_up_that_keeps_rescanning(sqlite_harness, tmp_path) -> None:
    """Round 2: a catch-up retrying a failing scan kept watch_forever from
    finishing after stop_event was set."""
    _, _, observer, clock = sqlite_harness
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    stop = asyncio.Event()
    watcher = ScriptedWatcher([[RuntimeError("watch died")], ["wait"]])
    service = TranscriptWatchService(roots=[tmp_path], observer=observer, watcher=watcher, clock=clock,
                                     restart_base_seconds=0.01, freshness_interval_seconds=3600)
    task = asyncio.create_task(service.watch_forever(stop_event=stop))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if watcher.calls == 2 and service.health()["state"] == "degraded":
                break
        assert service.health()["state"] == "degraded"
        assert service.health()["last_error"] == "PermissionError"
        stop.set()
        await asyncio.wait_for(task, 1)
        assert service.health()["state"] == "stopped"
    finally:
        locked.chmod(0o700)
        task.cancel()
