"""Regression: transcript silence must not demote a session whose harness run
is still queued or running (false idle, observed 2026-10-07).

A harness run's rollout is bound to the harness session, so its transcript
events feed the observer's freshness map. Before the fix, 30 s of silence
during one continuous run (a long tool call) published and materialized
``status=idle`` mid-run; the next transcript line flipped it back. The run
lifecycle (``create_run`` → running, ``finish_run`` → idle) owns the status
while a run is active; freshness stays the liveness signal for sessions the
harness is not running.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project, Session
from agent_harness.observer import ExternalTranscriptObserver
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository

START = datetime(2026, 10, 7, 14, 29, 48, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@pytest.fixture(params=["memory", "sqlite"])
def harness(request, tmp_path):
    """(repository, bus, observer, clock) on both repository implementations."""
    if request.param == "memory":
        repository = InMemoryRepository()
        bus = InMemoryEventBus()
    else:
        repository = open_sqlite_repository(tmp_path / "harness.db")
        bus = DurableEventBus(repository)
    clock = Clock()
    observer = ExternalTranscriptObserver(bus, repository=repository, idle_after_seconds=30.0, clock=clock)
    return repository, bus, observer, clock


def _rollout(tmp_path: Path, uuid: str = "019e00f1-0000-0000-0000-000000000000") -> Path:
    path = tmp_path / ".codex" / "sessions" / "2026" / "10" / "07" / f"rollout-2026-10-07T14-29-48-{uuid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append(path: Path, *lines: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


FIRST = (
    '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
    '{"type":"event_msg","payload":{"type":"user_message","message":"run the long build"}}',
)
LATER = '{"type":"event_msg","payload":{"type":"user_message","message":"still building"}}'


def _harness_session_with_run(repository, *, start: bool = True):
    session = repository.create_session(
        CreateSessionRequest(backend="codex", model="gpt-5.4", project=Project(path="/repo", name="repo"))
    )
    run = repository.create_run(session.id, CreateRunRequest(message="run the long build"))
    if start:
        repository.start_run(session.id, run.id)
    return session, run


async def _idle_events(bus, session_id: str) -> list[Event]:
    return [
        e for e in await bus.replay(session_id=session_id)
        if e.event == "session.updated" and e.data.get("session", {}).get("status") == "idle"
    ]


@pytest.mark.asyncio
async def test_running_harness_run_is_not_demoted_by_transcript_silence(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    assert repository.get_session(session.id).status == "running"

    # The reported incident: five silent stretches > 30 s within one run.
    for _ in range(5):
        clock.advance(95)
        await observer.freshness_tick()
        assert repository.get_session(session.id).status == "running"
        _append(rollout, LATER)
        clock.advance(1)
        await observer.tail_file(rollout)
        assert repository.get_session(session.id).status == "running"
    assert await _idle_events(bus, session.id) == []

    # Completion still clears promptly, through the run lifecycle.
    repository.finish_run(session.id, run.id, status="completed", stop_reason="end_turn")
    assert repository.get_session(session.id).status == "idle"
    clock.advance(60)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "idle"


@pytest.mark.parametrize("terminal", ["completed", "failed", "interrupted"])
@pytest.mark.asyncio
async def test_terminal_run_states_still_end_working(harness, tmp_path, terminal) -> None:
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    repository.finish_run(session.id, run.id, status=terminal)
    assert repository.get_session(session.id).status == "idle"
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "idle"


@pytest.mark.asyncio
async def test_queued_run_is_not_demoted(harness, tmp_path) -> None:
    """A follow-up waiting in the per-session queue is busy, not idle — and must
    not inherit the previous run's stale freshness timestamp."""
    repository, bus, observer, clock = harness
    session, first = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    repository.finish_run(session.id, first.id, status="completed")
    queued = repository.create_run(session.id, CreateRunRequest(message="follow-up"))
    assert repository.get_run(session.id, queued.id).status == "queued"
    assert repository.get_session(session.id).status == "running"
    clock.advance(120)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "running"
    assert await _idle_events(bus, session.id) == []


@pytest.mark.asyncio
async def test_late_transcript_after_run_end_still_heals_to_idle(harness, tmp_path) -> None:
    """A late flush after finish_run may kick the session to running; with no
    active run, freshness must still bring it back to idle."""
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    repository.finish_run(session.id, run.id, status="completed")
    _append(rollout, LATER)
    clock.advance(1)
    await observer.tail_file(rollout)
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "idle"


@pytest.mark.asyncio
async def test_adopted_external_session_with_active_run_is_not_demoted(harness, tmp_path) -> None:
    """An external conversation resumed through the harness keeps origin=external
    but its harness run owns the status while it is active."""
    repository, bus, observer, clock = harness
    rollout = _rollout(tmp_path, "019e00f2-0000-0000-0000-000000000000")
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    session_id = "codex_019e00f2-0000-0000-0000-000000000000"
    assert repository.get_session(session_id).origin == "external"
    run = repository.create_run(session_id, CreateRunRequest(message="continue"))
    repository.start_run(session_id, run.id)
    clock.advance(45)
    await observer.freshness_tick()
    assert repository.get_session(session_id).status == "running"
    repository.finish_run(session_id, run.id, status="completed")
    assert repository.get_session(session_id).status == "idle"


@pytest.mark.asyncio
async def test_pure_external_session_keeps_the_30s_heuristic(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    rollout = _rollout(tmp_path, "019e00f3-0000-0000-0000-000000000000")
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    session_id = "codex_019e00f3-0000-0000-0000-000000000000"
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(session_id).status == "idle"
    assert len(await _idle_events(bus, session_id)) == 1
    _append(rollout, LATER)
    clock.advance(1)
    await observer.tail_file(rollout)
    assert repository.get_session(session_id).status == "running"


@pytest.mark.asyncio
async def test_waiting_for_input_is_not_overwritten_by_freshness(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    rollout = _rollout(tmp_path, "019e00f4-0000-0000-0000-000000000000")
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    session_id = "codex_019e00f4-0000-0000-0000-000000000000"
    waiting = repository.get_session(session_id).model_copy(update={"status": "waiting_for_input"})
    repository.upsert_session(waiting)
    clock.advance(120)
    await observer.freshness_tick()
    assert repository.get_session(session_id).status == "waiting_for_input"


@pytest.mark.asyncio
async def test_archived_session_is_left_alone(harness, tmp_path) -> None:
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    repository.finish_run(session.id, run.id, status="completed")
    repository.archive_session(session.id)
    _append(rollout, LATER)
    clock.advance(31)
    await observer.tail_file(rollout)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "archived"


@pytest.mark.asyncio
async def test_stale_idle_observation_cannot_demote_an_active_run(harness, tmp_path) -> None:
    """Materialization guard: an idle observation decided before create_run
    (or replayed late) must not overwrite the run lifecycle's status."""
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    stale = repository.get_session(session.id).model_copy(update={"status": "idle"})
    repository.materialize_event(
        Event(event="session.updated", session_id=session.id, data={"session": stale.model_dump(mode="json")})
    )
    assert repository.get_session(session.id).status == "running"
    # Other observer-owned fields still apply.
    renamed = stale.model_copy(update={"model": "gpt-5.5"})
    repository.materialize_event(
        Event(event="session.updated", session_id=session.id, data={"session": renamed.model_dump(mode="json")})
    )
    after = repository.get_session(session.id)
    assert (after.status, after.model) == ("running", "gpt-5.5")
    # Once the run is over, observations own the status again.
    repository.finish_run(session.id, run.id, status="completed")
    running = after.model_copy(update={"status": "running"})
    repository.materialize_event(
        Event(event="session.updated", session_id=session.id, data={"session": running.model_dump(mode="json")})
    )
    assert repository.get_session(session.id).status == "running"
    repository.materialize_event(
        Event(event="session.updated", session_id=session.id, data={"session": stale.model_dump(mode="json")})
    )
    assert repository.get_session(session.id).status == "idle"


@pytest.mark.asyncio
async def test_race_between_freshness_check_and_create_run(harness, tmp_path, monkeypatch) -> None:
    """The observer's active-run check can be stale by the time its event is
    materialized; the repository guard keeps the new run's status."""
    repository, bus, observer, clock = harness
    session, run = _harness_session_with_run(repository)
    rollout = _rollout(tmp_path)
    observer.bind_rollout(rollout, session.id)
    _append(rollout, *FIRST)
    await observer.tail_file(rollout)
    # Simulate the check having run just before create_run made the run active.
    monkeypatch.setattr(observer, "_has_active_run", lambda session_id: False)
    clock.advance(31)
    await observer.freshness_tick()
    assert repository.get_session(session.id).status == "running"


def test_has_active_run_matches_run_states(harness) -> None:
    repository, *_ = harness
    session, run = _harness_session_with_run(repository, start=False)
    assert repository.has_active_run(session.id)          # queued
    repository.start_run(session.id, run.id)
    assert repository.has_active_run(session.id)          # running
    repository.finish_run(session.id, run.id, status="completed")
    assert not repository.has_active_run(session.id)
    assert not repository.has_active_run("ses_unknown")


def test_merge_observed_session_rule() -> None:
    from agent_harness.models import merge_observed_session

    base = Session(backend="codex", model="m", project=Project(path="/r", name="r"), status="running", origin="harness")
    idle = base.model_copy(update={"status": "idle"})
    assert merge_observed_session(idle, base, has_active_run=True).status == "running"
    assert merge_observed_session(idle, base, has_active_run=False).status == "idle"
    waiting = base.model_copy(update={"status": "waiting_for_input"})
    assert merge_observed_session(base, waiting, has_active_run=True).status == "running"
