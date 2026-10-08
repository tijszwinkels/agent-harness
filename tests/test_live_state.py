"""Live busy/idle claims for external pi sessions (Companion driving Pi).

Reproduces the 2026-10-08 counterexample with synthetic data: Companion
reports a Pi conversation busy (thinking / a long tool call), the transcript
is silent for more than 30 s, and the harness used to show it idle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.live_state import MAX_CLAIMS, LiveStateRegistry, LiveStateRejected
from agent_harness.models import (
    CreateRunRequest,
    CreateSessionRequest,
    Event,
    LiveStateRequest,
    Project,
    Session,
)
from agent_harness.observer import ExternalTranscriptObserver, pi_transcript_path
from agent_harness.repository import InMemoryRepository, SessionNotFoundError
from agent_harness.storage import open_sqlite_repository

UUID = "0c0f1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"   # synthetic "Companion conversation"
SESSION_ID = "ses_" + UUID.replace("-", "")
CWD = "/home/demo/companion/agents"
START = datetime(2026, 10, 8, 9, 0, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture(params=["memory", "sqlite"])
def harness(request, tmp_path):
    clock = Clock()
    if request.param == "memory":
        repository = InMemoryRepository()
        bus = InMemoryEventBus()
    else:
        repository = open_sqlite_repository(tmp_path / "harness.db")
        bus = DurableEventBus(repository)
    repository.clock = clock
    observer = ExternalTranscriptObserver(bus, repository=repository, idle_after_seconds=30.0, clock=clock)
    return repository, bus, observer, clock


OWNER_LINE = '{"type":"custom","id":"o1","customType":"agent-harness.live-state-owner","data":{"source":"companion"}}\n'


def _transcript(tmp_path: Path, uuid: str = UUID, *, owner: bool = True) -> Path:
    path = pi_transcript_path(CWD, "2026-10-08T09-00-00-000Z", uuid, home=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"session","version":3,"id":"' + uuid + '","cwd":"' + CWD + '"}\n'
        + (OWNER_LINE if owner else "")
        + '{"type":"model_change","provider":"openai-codex","modelId":"gpt-6-astra"}\n'
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"plan the release"}]}}\n',
        encoding="utf-8",
    )
    return path


def _append_assistant(path: Path, text: str = "working on it") -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"'
            + text + '"}],"stopReason":"stop"}}\n'
        )


async def _observed(observer, tmp_path) -> Path:
    path = _transcript(tmp_path)
    await observer.tail_file(path)
    return path


def claim(state: str, sequence: int, *, producer: str = "companion-a", lease: int = 45) -> LiveStateRequest:
    return LiveStateRequest(source="companion", producer=producer, sequence=sequence, state=state, lease_seconds=lease)


def _statuses(events) -> list[str]:
    return [
        e.data["session"]["status"] for e in events
        if e.event == "session.updated" and isinstance(e.data.get("session"), dict)
    ]


@pytest.mark.asyncio
async def test_counterexample_busy_companion_turn_stays_working_past_30s_silence(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    session = repository.get_session(SESSION_ID)
    assert (session.backend, session.origin, session.status) == ("pi", "external", "running")

    # Without a claim: silence → idle (the old behaviour, still the fallback).
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "idle"

    # Companion says busy and keeps renewing (heartbeat every 15 s, 45 s lease)
    # through four minutes of silent thinking/tool calls.
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    assert repository.get_session(SESSION_ID).status == "running"
    for sequence in range(2, 18):
        clock.advance(15)
        repository.apply_live_state(SESSION_ID, claim("busy", sequence))
        await observer.freshness_tick()
        assert repository.get_session(SESSION_ID).status == "running"

    # Settled: one idle claim ends it promptly.
    repository.apply_live_state(SESSION_ID, claim("idle", 18))
    assert repository.get_session(SESSION_ID).status == "idle"


@pytest.mark.asyncio
async def test_expired_lease_falls_back_to_transcript_freshness(harness, tmp_path) -> None:
    """A producer that dies mid-turn must not leave the session running."""
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1, lease=45))
    clock.advance(44)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "running"
    assert repository.status_owned(SESSION_ID)
    clock.advance(2)                                   # lease over, transcript silent for 46 s
    assert not repository.status_owned(SESSION_ID)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "idle"


@pytest.mark.asyncio
async def test_expired_lease_with_fresh_transcript_stays_running_until_silent(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    path = await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1, lease=10))
    clock.advance(20)
    _append_assistant(path)
    await observer.tail_file(path)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "running"
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "idle"


@pytest.mark.asyncio
async def test_stale_duplicate_and_reordered_updates_are_ignored(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    _, accepted, _changed = repository.apply_live_state(SESSION_ID, claim("busy", 5))
    assert accepted is not None
    # Duplicate and older updates from the same producer change nothing.
    for stale in (claim("busy", 5), claim("idle", 4), claim("idle", 0)):
        _, accepted, _changed = repository.apply_live_state(SESSION_ID, stale)
        assert accepted is None
        assert repository.get_session(SESSION_ID).status == "running"
    # Reordered delivery: idle(7) arrives before busy(6) → busy(6) is stale.
    repository.apply_live_state(SESSION_ID, claim("idle", 7))
    _, accepted, _changed = repository.apply_live_state(SESSION_ID, claim("busy", 6))
    assert accepted is None
    assert repository.get_session(SESSION_ID).status == "idle"
    assert not repository.status_owned(SESSION_ID)


@pytest.mark.asyncio
async def test_new_producer_after_companion_restart_replaces_the_claim(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 900, producer="companion-old"))
    _, accepted, _changed = repository.apply_live_state(SESSION_ID, claim("idle", 1, producer="companion-new"))
    assert accepted is not None
    assert repository.get_session(SESSION_ID).status == "idle"
    assert not repository.status_owned(SESSION_ID)


@pytest.mark.asyncio
async def test_transcript_observations_cannot_change_status_while_busy(harness, tmp_path) -> None:
    """PR43's guard path covers live claims: neither a freshness flip nor a
    stale observation snapshot (stored or announced) demotes a busy session."""
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    stale = repository.get_session(SESSION_ID).model_copy(update={"status": "idle"})
    published = await observer._publish_via_bus(
        Event(event="session.updated", session_id=SESSION_ID, data={"session": stale.model_dump(mode="json")})
    )
    assert repository.get_session(SESSION_ID).status == "running"
    assert published.data["session"]["status"] == "running"
    assert _statuses(await bus.replay(session_id=SESSION_ID))[-1] == "running"
    clock.advance(40)
    repository.apply_live_state(SESSION_ID, claim("busy", 2))
    await observer.freshness_tick()
    assert "idle" not in _statuses(await bus.replay(session_id=SESSION_ID))[-1:]


@pytest.mark.asyncio
async def test_idle_claim_does_not_block_later_transcript_activity(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    path = await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("idle", 1))
    assert repository.get_session(SESSION_ID).status == "idle"
    clock.advance(5)
    _append_assistant(path, "someone continued in a terminal")
    await observer.tail_file(path)
    assert repository.get_session(SESSION_ID).status == "running"


@pytest.mark.asyncio
async def test_active_harness_run_keeps_ownership(harness, tmp_path) -> None:
    """A harness run resumed on the external session wins over claims."""
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    run = repository.create_run(SESSION_ID, CreateRunRequest(message="continue"))
    repository.start_run(SESSION_ID, run.id)
    repository.apply_live_state(SESSION_ID, claim("idle", 1))
    assert repository.get_session(SESSION_ID).status == "running"
    repository.finish_run(SESSION_ID, run.id, status="completed")
    assert repository.get_session(SESSION_ID).status == "idle"


@pytest.mark.asyncio
async def test_waiting_session_becomes_running_on_busy_claim(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    repository.upsert_session(repository.get_session(SESSION_ID).model_copy(update={"status": "waiting_for_input"}))
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    assert repository.get_session(SESSION_ID).status == "running"


def test_scope_only_external_pi_sessions(harness) -> None:
    repository, *_ = harness
    for sid in ("ses_" + "a" * 32, "codex_x"):
        repository.note_live_state_owner(sid, "companion")
    with pytest.raises(SessionNotFoundError):
        repository.apply_live_state("ses_" + "f" * 32, claim("busy", 1))
    harness_pi = repository.create_session(
        CreateSessionRequest(backend="pi", project=Project(path="/r", name="r"))
    )
    codex_external = Session(id="codex_x", backend="codex", origin="external", status="idle",
                             project=Project(path="/r", name="r"))
    repository.upsert_session(codex_external)
    archived = Session(id="ses_" + "a" * 32, backend="pi", origin="external", status="archived",
                       project=Project(path="/r", name="r"))
    repository.upsert_session(archived)
    for session_id in (harness_pi.id, codex_external.id, archived.id):
        with pytest.raises(LiveStateRejected):
            repository.apply_live_state(session_id, claim("busy", 1))
        assert not repository.status_owned(session_id)
    assert repository.get_session(archived.id).status == "archived"


def test_harness_restart_forgets_claims(tmp_path) -> None:
    """Claims live in memory: a restarted harness sees none and the session
    falls back to the transcript rule until the producer's next heartbeat."""
    clock = Clock()
    first = open_sqlite_repository(tmp_path / "harness.db")
    first.clock = clock
    first.upsert_session(Session(id=SESSION_ID, backend="pi", origin="external", status="idle",
                                 project=Project(path=CWD, name="agents")))
    first.note_live_state_owner(SESSION_ID, "companion")
    first.apply_live_state(SESSION_ID, claim("busy", 1))
    assert first.status_owned(SESSION_ID)
    first._connection.close()  # the old process is gone
    second = open_sqlite_repository(tmp_path / "harness.db")
    second.clock = clock
    assert second.get_session(SESSION_ID).status == "running"
    assert not second.status_owned(SESSION_ID)
    # Ownership is in memory too: after a restart the API re-learns it from
    # the transcript (see test_api_owner_scan_after_restart).
    second.note_live_state_owner(SESSION_ID, "companion")
    second.apply_live_state(SESSION_ID, claim("busy", 2))
    assert second.status_owned(SESSION_ID)


def test_registry_is_bounded() -> None:
    registry = LiveStateRegistry()
    now = START
    for index in range(MAX_CLAIMS + 50):
        registry.offer(f"ses_{index:032x}", claim("busy", 1), now + timedelta(milliseconds=index))
    assert len(registry._claims) == MAX_CLAIMS


def test_api_contract(tmp_path) -> None:
    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    transcript = _transcript(tmp_path)
    repository.upsert_session(Session(id=SESSION_ID, backend="pi", origin="external", status="idle",
                                      project=Project(path=CWD, name="agents"),
                                      pi_transcript_path=str(transcript)))
    client = TestClient(create_app(repository=repository, event_bus=bus))
    url = f"/v1/sessions/{SESSION_ID}/live-state"
    body = {"source": "companion", "producer": "p1", "sequence": 1, "state": "busy", "lease_seconds": 45}

    response = client.put(url, json=body)
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert response.json()["status"] == "running"
    assert response.json()["busy"] is True
    assert client.get(f"/v1/sessions/{SESSION_ID}").json()["status"] == "running"

    duplicate = client.put(url, json=body)
    assert duplicate.json()["accepted"] is False

    assert client.put(url, json={**body, "sequence": 2, "state": "idle"}).json()["status"] == "idle"
    assert client.put("/v1/sessions/ses_" + "e" * 32 + "/live-state", json=body).status_code == 404
    for bad in ({**body, "state": "done"}, {**body, "lease_seconds": 600}, {**body, "source": "Bad Source"},
                {**body, "producer": "../x"}, {**body, "sequence": -1}, {**body, "extra": 1}):
        assert client.put(url, json={**bad, "sequence": bad.get("sequence", 9)}).status_code == 422
    repository.upsert_session(Session(id="ses_" + "b" * 32, backend="pi", origin="harness", status="idle",
                                      project=Project(path="/r", name="r")))
    assert client.put("/v1/sessions/ses_" + "b" * 32 + "/live-state", json=body).status_code == 409


# --------------------------------------------------------------- review round 1


@pytest.mark.asyncio
async def test_unrelated_terminal_pi_session_cannot_be_claimed(harness, tmp_path) -> None:
    """Finding 5: the producer that owns one conversation cannot change a
    valid, external, unrelated Pi session (no owner entry in its transcript)."""
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    other_uuid = "9f9f9f9f-1111-4222-8333-444455556666"
    other_id = "ses_" + other_uuid.replace("-", "")
    await observer.tail_file(_transcript(tmp_path, other_uuid, owner=False))
    other = repository.get_session(other_id)
    assert (other.backend, other.origin) == ("pi", "external")
    clock.advance(31)
    await observer.freshness_tick()
    repository.apply_live_state(SESSION_ID, claim("busy", 1))     # own conversation: fine
    with pytest.raises(LiveStateRejected):
        repository.apply_live_state(other_id, claim("busy", 2))
    assert repository.get_session(other_id).status == "idle"
    assert not repository.status_owned(other_id)


@pytest.mark.asyncio
async def test_owner_entry_from_another_source_is_not_enough(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    await _observed(observer, tmp_path)
    with pytest.raises(LiveStateRejected):
        repository.apply_live_state(
            SESSION_ID,
            LiveStateRequest(source="other-tool", producer="x", sequence=1, state="busy"),
        )


@pytest.mark.asyncio
async def test_settled_turn_bytes_tailed_late_do_not_revive(harness, tmp_path) -> None:
    """Finding 2: busy → final assistant bytes written (not yet tailed) → idle →
    observer tails those bytes: the session stays idle in row, SSE and replay.
    A genuinely new line afterwards resumes best-effort tracking."""
    repository, bus, observer, clock = harness
    path = await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    _append_assistant(path, "final answer")
    from agent_harness.live_state import transcript_size
    repository.apply_live_state(SESSION_ID, claim("idle", 2), settled_offset=transcript_size(str(path)))
    assert repository.get_session(SESSION_ID).status == "idle"
    before = len(_statuses(await bus.replay(session_id=SESSION_ID)))
    clock.advance(1)
    await observer.tail_file(path)
    assert repository.get_session(SESSION_ID).status == "idle"
    assert "running" not in _statuses(await bus.replay(session_id=SESSION_ID))[before:]
    clock.advance(5)
    _append_assistant(path, "a new prompt was answered in a terminal")
    await observer.tail_file(path)
    assert repository.get_session(SESSION_ID).status == "running"


@pytest.mark.parametrize("terminal", ["completed", "failed", "interrupted"])
@pytest.mark.asyncio
async def test_finished_run_hands_back_to_a_valid_busy_claim(harness, tmp_path, terminal) -> None:
    """Finding 3: when the higher-priority run ends while Companion's busy
    claim is still valid, the session stays running (not idle under a claim)."""
    repository, bus, observer, clock = harness
    path = await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    run = repository.create_run(SESSION_ID, CreateRunRequest(message="continue"))
    repository.start_run(SESSION_ID, run.id)
    repository.finish_run(SESSION_ID, run.id, status=terminal)
    assert repository.get_session(SESSION_ID).status == "running"
    assert repository.status_owned(SESSION_ID)
    # And once the claim has lapsed, the run's end means idle as before.
    run2 = repository.create_run(SESSION_ID, CreateRunRequest(message="again"))
    repository.start_run(SESSION_ID, run2.id)
    clock.advance(60)
    repository.finish_run(SESSION_ID, run2.id, status=terminal)
    assert repository.get_session(SESSION_ID).status == "idle"


def test_api_owner_scan_after_restart(tmp_path) -> None:
    """Without the observer having seen the owner entry (harness restart), the
    API learns it from the session's own transcript; without an entry it 409s."""
    owned = _transcript(tmp_path)
    other_uuid = "9f9f9f9f-1111-4222-8333-444455556666"
    unowned = _transcript(tmp_path, other_uuid, owner=False)
    repository = InMemoryRepository()
    for sid, path in ((SESSION_ID, owned), ("ses_" + other_uuid.replace("-", ""), unowned)):
        repository.upsert_session(Session(id=sid, backend="pi", origin="external", status="idle",
                                          project=Project(path=CWD, name="agents"),
                                          pi_transcript_path=str(path)))
    client = TestClient(create_app(repository=repository, event_bus=InMemoryEventBus()))
    body = {"source": "companion", "producer": "p1", "sequence": 1, "state": "busy"}
    assert client.put(f"/v1/sessions/{SESSION_ID}/live-state", json=body).json()["status"] == "running"
    response = client.put("/v1/sessions/ses_" + other_uuid.replace("-", "") + "/live-state", json=body)
    assert response.status_code == 409
    assert "not owned" in response.json()["detail"]


def test_owner_scan_is_rate_limited() -> None:
    registry = LiveStateRegistry()
    assert registry.should_scan_owner("ses_x", START)
    assert not registry.should_scan_owner("ses_x", START + timedelta(seconds=5))
    assert registry.should_scan_owner("ses_x", START + timedelta(seconds=31))
    registry.mark_owner("ses_x", "companion")
    assert not registry.should_scan_owner("ses_x", START + timedelta(seconds=100))


def test_api_idle_records_the_settled_boundary(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    repository = InMemoryRepository()
    repository.upsert_session(Session(id=SESSION_ID, backend="pi", origin="external", status="idle",
                                      project=Project(path=CWD, name="agents"),
                                      pi_transcript_path=str(transcript)))
    client = TestClient(create_app(repository=repository, event_bus=InMemoryEventBus()))
    url = f"/v1/sessions/{SESSION_ID}/live-state"
    client.put(url, json={"source": "companion", "producer": "p", "sequence": 1, "state": "busy"})
    _append_assistant(transcript, "done")
    client.put(url, json={"source": "companion", "producer": "p", "sequence": 2, "state": "idle"})
    size = transcript.stat().st_size
    assert repository.observation_superseded(SESSION_ID, size - 10)
    assert not repository.observation_superseded(SESSION_ID, size)


# --------------------------------------------------------------- review round 2


def _append_model_change(path: Path, model: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"model_change","provider":"openai-codex","modelId":"' + model + '"}\n')


@pytest.mark.asyncio
async def test_settled_metadata_change_tailed_late_keeps_status_but_applies_metadata(harness, tmp_path) -> None:
    """Finding 1: a model_change written by the finished turn and tailed after
    its idle produces a full session.updated; it must not make the session
    running (row and replay) but its metadata still applies."""
    repository, bus, observer, clock = harness
    path = await _observed(observer, tmp_path)
    repository.apply_live_state(SESSION_ID, claim("busy", 1))
    _append_model_change(path, "gpt-6-sol")
    from agent_harness.live_state import transcript_size
    repository.apply_live_state(SESSION_ID, claim("idle", 2), settled_offset=transcript_size(str(path)))
    before = len(_statuses(await bus.replay(session_id=SESSION_ID)))
    clock.advance(1)
    await observer.tail_file(path)
    session = repository.get_session(SESSION_ID)
    assert session.status == "idle"
    assert "running" not in _statuses(await bus.replay(session_id=SESSION_ID))[before:]
    assert session.model is not None and "gpt-6-sol" in session.model


@pytest.mark.asyncio
async def test_api_publishes_when_status_changes_during_its_awaits(tmp_path, monkeypatch) -> None:
    """Finding 2: an observation sets running while the idle request waits on
    its transcript stat; the claim's change back to idle must be published."""
    import agent_harness.api as api_module
    from httpx import ASGITransport, AsyncClient

    for kind in ("memory", "sqlite"):
        transcript = _transcript(tmp_path / kind)
        if kind == "memory":
            repository = InMemoryRepository()
            bus = InMemoryEventBus()
        else:
            repository = open_sqlite_repository(tmp_path / kind / "harness.db")
            bus = DurableEventBus(repository)
        repository.upsert_session(Session(id=SESSION_ID, backend="pi", origin="external", status="idle",
                                          project=Project(path=CWD, name="agents"),
                                          pi_transcript_path=str(transcript)))
        repository.note_live_state_owner(SESSION_ID, "companion")
        app = create_app(repository=repository, event_bus=bus)

        original = api_module.transcript_size

        def slow_size(path):
            # The observation lands while the request is between its awaits.
            running = repository.get_session(SESSION_ID).model_copy(update={"status": "running"})
            repository.upsert_session(running)
            return original(path)

        monkeypatch.setattr(api_module, "transcript_size", slow_size)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            body = {"source": "companion", "producer": "p", "sequence": 1, "state": "idle"}
            response = await client.put(f"/v1/sessions/{SESSION_ID}/live-state", json=body)
        assert response.json()["status"] == "idle"
        assert repository.get_session(SESSION_ID).status == "idle"
        assert _statuses(await bus.replay(session_id=SESSION_ID))[-1] == "idle", kind
        monkeypatch.setattr(api_module, "transcript_size", original)


def test_owner_beyond_first_chunk_is_found_incrementally_and_survives_restart(tmp_path, monkeypatch) -> None:
    """Finding 5: an owner entry far into a long transcript (appended when the
    feature was activated) is found by successive bounded scans, persisted,
    and still known after the harness restarts."""
    import agent_harness.live_state as live_state

    monkeypatch.setattr(live_state, "OWNER_SCAN_CHUNK_BYTES", 64 * 1024)
    monkeypatch.setattr(live_state, "OWNER_SCAN_BLOCK_BYTES", 8 * 1024)
    path = _transcript(tmp_path, owner=False)
    with path.open("a", encoding="utf-8") as handle:
        for index in range(1500):  # ~200 KiB of history before the marker
            handle.write('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"history '
                         + str(index) + ' ' + "x" * 100 + '"}]}}\n')
        handle.write(OWNER_LINE)
    clock = Clock()
    repository = open_sqlite_repository(tmp_path / "harness.db")
    repository.clock = clock
    repository.upsert_session(Session(id=SESSION_ID, backend="pi", origin="external", status="idle",
                                      project=Project(path=CWD, name="agents"), pi_transcript_path=str(path)))
    client = TestClient(create_app(repository=repository, event_bus=DurableEventBus(repository)))
    url = f"/v1/sessions/{SESSION_ID}/live-state"
    statuses = []
    for sequence in range(1, 10):
        response = client.put(url, json={"source": "companion", "producer": "p", "sequence": sequence, "state": "busy"})
        statuses.append(response.status_code)
        if response.status_code == 200:
            break
        clock.advance(11)  # past the rescan rate limit
    assert statuses[0] == 409 and statuses[-1] == 200, statuses
    client.close()
    repository._connection.close()

    reopened = open_sqlite_repository(tmp_path / "harness.db")
    assert reopened.live_state_owner(SESSION_ID) == "companion"
    reopened.apply_live_state(SESSION_ID, claim("busy", 1, producer="after-restart"))
    assert reopened.status_owned(SESSION_ID)


def test_owner_scan_memory_is_bounded_for_one_huge_line(tmp_path, monkeypatch) -> None:
    """Finding 6: a single 8 MiB JSONL record is not read into memory whole."""
    import tracemalloc

    import agent_harness.live_state as live_state

    monkeypatch.setattr(live_state, "OWNER_SCAN_BLOCK_BYTES", 64 * 1024)
    path = tmp_path / "huge.jsonl"
    with path.open("wb") as handle:
        handle.write(b'{"type":"message","text":"' + b"y" * (8 * 1024 * 1024) + b'"}\n')
        handle.write(OWNER_LINE.encode())
    tracemalloc.start()
    owner, cursor = live_state.scan_owner_chunk(str(path), 0, budget=32 * 1024 * 1024)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert owner == "companion"
    assert peak < 1024 * 1024, peak


def test_owner_marker_inside_a_long_line_is_ignored(tmp_path) -> None:
    import agent_harness.live_state as live_state

    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"type":"message","text":"' + b"z" * 20000
                     + b' agent-harness.live-state-owner "agent-harness.live-state-owner" ' + b"z" * 20000 + b'"}\n')
    assert live_state.scan_owner_chunk(str(path), 0)[0] is None
