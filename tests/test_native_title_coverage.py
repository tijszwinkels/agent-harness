"""Native title clearing, claude custom/generated titles and the codex
name index. Builds on ``test_native_titles.py`` (pi names, provenance).

All transcripts, indexes and databases here are synthetic.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import Project, Session
from agent_harness.native_titles import CLEARED
from agent_harness.observer import (
    ExternalTranscriptObserver,
    claude_transcript_path,
    pi_transcript_path,
)
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository

PI_UUID = "e5a93149-9a70-4aef-a189-2681b4e08525"
PI_ID = "ses_e5a931499a704aefa1892681b4e08525"
CLAUDE_UUID = "0f8fad5b-d9cb-469f-a165-70867728950e"
CLAUDE_ID = "ses_0f8fad5bd9cb469fa16570867728950e"
CWD = "/home/me/project"

PI_SESSION = {"type": "session", "version": 3, "id": PI_UUID, "cwd": CWD}
PI_USER = {"type": "message", "message": {"role": "user", "content": "q"}}
CLAUDE_USER = {
    "type": "user",
    "cwd": CWD,
    "message": {"role": "user", "model": "claude-x", "content": "hi"},
}


def _name(name):
    return {"type": "session_info", "id": "x", "parentId": None, "name": name}


def _custom(title, session_id=CLAUDE_UUID):
    record = {"type": "custom-title", "customTitle": title}
    if session_id is not None:
        record["sessionId"] = session_id
    return record


def _ai(title, session_id=CLAUDE_UUID):
    return {"type": "ai-title", "aiTitle": title, "sessionId": session_id}


def _write(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _append(path: Path, *records) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _pi(home: Path, records: list) -> Path:
    path = pi_transcript_path(CWD, "2026-10-07T10-00-00-000Z", PI_UUID, home=home)
    _write(path, records)
    return path


def _claude(home: Path, records: list) -> Path:
    path = claude_transcript_path(CWD, CLAUDE_UUID, home=home)
    _write(path, records)
    return path


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _repos(tmp_path):
    """Both repository/event-bus paths."""
    yield "memory", InMemoryRepository(), None
    sqlite = open_sqlite_repository(tmp_path / "harness.db")
    yield "sqlite", sqlite, DurableEventBus(sqlite)


def _observer(repository, bus=None, **kwargs) -> ExternalTranscriptObserver:
    return ExternalTranscriptObserver(
        bus or InMemoryEventBus(), repository=repository, idle_after_seconds=30.0, **kwargs
    )


def _session_payloads(events) -> list[tuple]:
    return [
        (e.data["session"]["title"], e.data["session"]["title_source"])
        for e in events
        if e.event == "session.updated" and "session" in e.data
    ]


def _title(repository, session_id) -> tuple:
    session = repository.get_session(session_id)
    return session.title, session.title_source


# --------------------------------------------------------------------------- #
# pi clearing                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pi_blank_name_clears_and_a_later_name_restores(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        transcript = _pi(tmp_path / kind, [PI_SESSION, _name("named")])
        observer = _observer(repository, bus)
        await observer.tail_file(transcript)
        assert _title(repository, PI_ID) == ("named", "native"), kind

        _append(transcript, _name("  "))
        cleared = await observer.tail_file(transcript)
        assert _session_payloads(cleared) == [(None, "native")], kind
        assert _title(repository, PI_ID) == (None, "native"), kind

        # Repeated clears and malformed names change nothing.
        _append(transcript, _name(""), _name(None), _name(3))
        assert await observer.tail_file(transcript) == [], kind

        _append(transcript, _name("again"))
        restored = await observer.tail_file(transcript)
        assert _session_payloads(restored) == [("again", "native")], kind
        assert _title(repository, PI_ID) == ("again", "native"), kind


@pytest.mark.asyncio
async def test_clearing_never_touches_an_explicit_title(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        transcript = _pi(tmp_path / kind, [PI_SESSION, _name("named")])
        observer = _observer(repository, bus)
        await observer.tail_file(transcript)
        repository.patch_session(PI_ID, {"title": "bridge", "title_source": None})

        _append(transcript, _name(""), PI_USER)
        published = await observer.tail_file(transcript)

        assert set(_session_payloads(published)) <= {("bridge", None)}, kind
        assert _title(repository, PI_ID) == ("bridge", None), kind


@pytest.mark.asyncio
async def test_clearing_never_touches_a_harness_session(tmp_path) -> None:
    repository = InMemoryRepository()
    repository.upsert_session(
        Session(id=PI_ID, backend="pi", project=Project(path=CWD, name="p"),
                origin="harness", title="owned", title_source="native")
    )
    transcript = _pi(tmp_path, [PI_SESSION, _name("")])

    assert await _observer(repository).tail_file(transcript) == []
    assert _title(repository, PI_ID) == ("owned", "native")


@pytest.mark.asyncio
async def test_a_clear_is_not_activity(tmp_path) -> None:
    transcript = _pi(tmp_path, [PI_SESSION, _name("named")])
    repository = InMemoryRepository()
    clock = _Clock()
    observer = _observer(repository, clock=clock)
    await observer.tail_file(transcript)
    clock.now += timedelta(seconds=31)
    await observer.freshness_tick()
    before = repository.get_session(PI_ID)
    last_seen = observer._last_event_at[PI_ID]

    clock.now += timedelta(minutes=1)
    _append(transcript, _name(""))
    await observer.tail_file(transcript)

    after = repository.get_session(PI_ID)
    assert (after.title, after.status) == (None, "idle")
    assert (after.updated_at, after.stats) == (before.updated_at, before.stats)
    assert observer._last_event_at[PI_ID] == last_seen


@pytest.mark.asyncio
async def test_a_clear_survives_restart_without_resurrecting(tmp_path) -> None:
    """The stored row still says ``named`` (cleared while the harness was
    down, consumed by a previous process); the backfill corrects it."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    transcript = _pi(tmp_path, [PI_SESSION, _name("named")])
    await _observer(repository, DurableEventBus(repository)).tail_file(transcript)
    stale = repository.get_session(PI_ID)
    _append(transcript, _name(""))
    repository.set_observer_offset(str(transcript), transcript.stat().st_size)
    repository.close()

    reopened = open_sqlite_repository(db_path)
    try:
        _observer(reopened, DurableEventBus(reopened))
        session = reopened.get_session(PI_ID)
        assert (session.title, session.title_source) == (None, "native")
        assert session.updated_at == stale.updated_at
    finally:
        reopened.close()


# --------------------------------------------------------------------------- #
# claude custom / generated titles                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_claude_title_before_discovery_is_applied_at_discovery(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        transcript = _claude(tmp_path / kind, [_ai("generated"), _custom("chosen"), CLAUDE_USER])

        published = await _observer(repository, bus).tail_file(transcript)

        assert _session_payloads(published)[-1] == ("chosen", "native"), kind
        assert _title(repository, CLAUDE_ID) == ("chosen", "native"), kind


@pytest.mark.asyncio
async def test_claude_generated_title_alone_is_used(tmp_path) -> None:
    repository = InMemoryRepository()
    transcript = _claude(tmp_path, [CLAUDE_USER, _ai("generated")])

    await _observer(repository).tail_file(transcript)

    assert _title(repository, CLAUDE_ID) == ("generated", "native")


@pytest.mark.asyncio
async def test_claude_custom_beats_generated_in_either_order(tmp_path) -> None:
    for order in ([_custom("chosen"), _ai("generated")], [_ai("generated"), _custom("chosen")]):
        home = tmp_path / str(len(list(home for home in tmp_path.iterdir())))
        repository = InMemoryRepository()
        transcript = _claude(home, [CLAUDE_USER, *order])
        await _observer(repository).tail_file(transcript)
        assert _title(repository, CLAUDE_ID) == ("chosen", "native"), order


@pytest.mark.asyncio
async def test_claude_clears_fall_back_between_custom_and_generated(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        transcript = _claude(tmp_path / kind, [CLAUDE_USER, _ai("generated"), _custom("chosen")])
        observer = _observer(repository, bus)
        await observer.tail_file(transcript)

        # Clearing the generated title keeps the custom one: nothing to publish.
        _append(transcript, _ai(" "))
        assert _session_payloads(await observer.tail_file(transcript)) == [], kind
        assert _title(repository, CLAUDE_ID) == ("chosen", "native"), kind

        _append(transcript, _ai("generated 2"), _custom(""))
        published = await observer.tail_file(transcript)
        assert _session_payloads(published)[-1] == ("generated 2", "native"), kind

        _append(transcript, _ai(""))
        published = await observer.tail_file(transcript)
        assert _session_payloads(published) == [(None, "native")], kind
        assert _title(repository, CLAUDE_ID) == (None, "native"), kind


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["11111111-2222-3333-4444-555555555555", 7, None])
async def test_claude_title_for_another_session_is_ignored(tmp_path, session_id) -> None:
    repository = InMemoryRepository()
    transcript = _claude(tmp_path, [CLAUDE_USER, _custom("mine")])
    observer = _observer(repository)
    await observer.tail_file(transcript)

    record = _custom("theirs", session_id="placeholder")
    record["sessionId"] = session_id
    _append(transcript, record, _custom("", session_id="also-not-mine"))
    await observer.tail_file(transcript)

    assert _title(repository, CLAUDE_ID) == ("mine", "native")


@pytest.mark.asyncio
async def test_claude_title_without_session_id_is_accepted(tmp_path) -> None:
    repository = InMemoryRepository()
    transcript = _claude(tmp_path, [CLAUDE_USER, _custom("legacy", session_id=None)])
    await _observer(repository).tail_file(transcript)
    assert _title(repository, CLAUDE_ID) == ("legacy", "native")


@pytest.mark.asyncio
async def test_claude_custom_preference_survives_restart(tmp_path) -> None:
    """After a restart a new generated title must not displace the custom
    one, although the stored row alone can't say which slot it came from."""
    for kind, repository, bus in _repos(tmp_path):
        transcript = _claude(tmp_path / kind, [CLAUDE_USER, _ai("generated"), _custom("chosen")])
        await _observer(repository, bus).tail_file(transcript)

        restarted = _observer(repository, bus)
        restarted._state.set_next_offset(transcript, transcript.stat().st_size)
        _append(transcript, _ai("regenerated"))
        await restarted.tail_file(transcript)
        assert _title(repository, CLAUDE_ID) == ("chosen", "native"), kind

        _append(transcript, _custom(""))
        await restarted.tail_file(transcript)
        assert _title(repository, CLAUDE_ID) == ("regenerated", "native"), kind


@pytest.mark.asyncio
async def test_claude_startup_backfill_reads_the_consumed_prefix_only(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    transcript = _claude(tmp_path, [CLAUDE_USER])
    await _observer(repository, DurableEventBus(repository)).tail_file(transcript)
    _append(transcript, _ai("generated"), _custom("chosen"))
    consumed = transcript.stat().st_size
    repository.set_observer_offset(str(transcript), consumed)
    _append(transcript, _custom("unread"))
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_custom("partial")))
    repository.close()

    reopened = open_sqlite_repository(db_path)
    try:
        restarted = _observer(reopened, DurableEventBus(reopened))
        assert _title(reopened, CLAUDE_ID) == ("chosen", "native")
        # The ordered tail then applies the unread record; the partial
        # line waits for its newline.
        await restarted.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == ("unread", "native")
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_claude_title_records_are_not_activity(tmp_path) -> None:
    repository = InMemoryRepository()
    clock = _Clock()
    transcript = _claude(tmp_path, [CLAUDE_USER])
    observer = _observer(repository, clock=clock)
    await observer.tail_file(transcript)
    clock.now += timedelta(seconds=31)
    await observer.freshness_tick()
    before = repository.get_session(CLAUDE_ID)

    clock.now += timedelta(minutes=1)
    _append(transcript, _ai("generated"), _custom("chosen"), _custom(""))
    await observer.tail_file(transcript)

    after = repository.get_session(CLAUDE_ID)
    assert (after.title, after.status) == ("generated", "idle")
    assert (after.updated_at, after.stats) == (before.updated_at, before.stats)


@pytest.mark.asyncio
async def test_claude_announcements_advertise_the_explicit_title(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        transcript = _claude(tmp_path / kind, [CLAUDE_USER, _custom("chosen")])
        observer = _observer(repository, bus)
        await observer.tail_file(transcript)
        repository.patch_session(CLAUDE_ID, {"title": "bridge", "title_source": None})

        _append(transcript, CLAUDE_USER, _ai("generated"), _custom(""))
        published = await observer.tail_file(transcript)

        assert set(_session_payloads(published)) == {("bridge", None)}, kind
        assert _title(repository, CLAUDE_ID) == ("bridge", None), kind


@pytest.mark.asyncio
async def test_unreadable_prefix_keeps_the_stored_title_until_it_recovers(tmp_path) -> None:
    """A failed startup scan must not hydrate empty slots: the custom slot
    would look absent and a generated title (or its removal) would win."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    transcript = _claude(tmp_path, [CLAUDE_USER, _ai("generated"), _custom("chosen")])
    await _observer(repository, DurableEventBus(repository)).tail_file(transcript)
    repository.set_observer_offset(str(transcript), transcript.stat().st_size)
    repository.close()

    transcript.chmod(0)
    reopened = open_sqlite_repository(db_path)
    try:
        if os.access(transcript, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        observer = _observer(reopened, DurableEventBus(reopened))
        assert _title(reopened, CLAUDE_ID) == ("chosen", "native")
        transcript.chmod(0o600)

        _append(transcript, _ai(""))
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == ("chosen", "native")
        _append(transcript, _ai("regenerated"))
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == ("chosen", "native")
    finally:
        transcript.chmod(0o600)
        reopened.close()


@pytest.mark.asyncio
async def test_recovered_prefix_is_reconciled_on_an_ordinary_line(tmp_path) -> None:
    """Upgrade path: the row is untitled (written before ai-titles were
    read) and the startup backfill can't read the transcript. The first
    readable new line is plain conversation; the recovered name must
    still land, without counting as activity."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    transcript = _claude(tmp_path, [CLAUDE_USER, _ai("generated")])
    await _observer(repository, DurableEventBus(repository)).tail_file(transcript)
    legacy = repository.get_session(CLAUDE_ID).model_copy(
        update={"title": None, "title_source": None}
    )
    repository.upsert_session(legacy)
    repository.set_observer_offset(str(transcript), transcript.stat().st_size)
    repository.close()

    transcript.chmod(0)
    reopened = open_sqlite_repository(db_path)
    try:
        if os.access(transcript, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        observer = _observer(reopened, DurableEventBus(reopened))
        assert _title(reopened, CLAUDE_ID) == (None, None)
        transcript.chmod(0o600)

        renames = []
        publish = observer._publish_native_name

        async def counting(*args, **kwargs):
            event = await publish(*args, **kwargs)
            if event is not None:
                renames.append(event)
            return event

        observer._publish_native_name = counting
        _append(transcript, CLAUDE_USER)
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == ("generated", "native")
        assert len(renames) == 1
        # Reconciled once; later ordinary lines don't re-check it.
        _append(transcript, CLAUDE_USER)
        await observer.tail_file(transcript)
        assert len(renames) == 1
    finally:
        transcript.chmod(0o600)
        reopened.close()


@pytest.mark.asyncio
async def test_truncated_transcript_does_not_resurrect_a_cleared_name(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    transcript = _claude(tmp_path, [CLAUDE_USER, _custom("old")])
    truncated_size = transcript.stat().st_size
    _append(transcript, _custom(""))
    await _observer(repository, DurableEventBus(repository)).tail_file(transcript)
    assert _title(repository, CLAUDE_ID) == (None, "native")
    repository.close()

    with transcript.open("r+b") as handle:
        handle.truncate(truncated_size)
    reopened = open_sqlite_repository(db_path)
    try:
        observer = _observer(reopened, DurableEventBus(reopened))
        assert _title(reopened, CLAUDE_ID) == (None, "native")
        _append(transcript, CLAUDE_USER)
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == (None, "native")
        # Regrown past the old offset: once the tail realigns, the prefix is
        # the file as it now is — the genuinely latest record wins, and the
        # truncated-away ``old`` never comes back.
        while transcript.stat().st_size <= observer._state.next_offset(transcript):
            _append(transcript, CLAUDE_USER)
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == (None, "native")
        _append(transcript, _custom("after truncation"))
        await observer.tail_file(transcript)
        assert _title(reopened, CLAUDE_ID) == ("after truncation", "native")
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_eviction_rehydrates_from_the_consumed_prefix(tmp_path) -> None:
    from agent_harness.native_titles import NativeTitleTracker

    repository = InMemoryRepository()
    transcript = _claude(tmp_path, [CLAUDE_USER, _ai("generated"), _custom("chosen")])
    observer = _observer(repository)
    await observer.tail_file(transcript)
    observer._native_titles = NativeTitleTracker(max_entries=1)  # forget everything

    _append(transcript, _ai("regenerated"))
    await observer.tail_file(transcript)
    assert _title(repository, CLAUDE_ID) == ("chosen", "native")
    _append(transcript, _custom(""))
    await observer.tail_file(transcript)
    assert _title(repository, CLAUDE_ID) == ("regenerated", "native")


def test_records_observed_before_hydration_say_nothing() -> None:
    from agent_harness.native_titles import NativeTitleTracker

    tracker = NativeTitleTracker()
    tracker.observe("t.jsonl", "ai", CLEARED)
    assert tracker.effective("t.jsonl", "claude-code") is None
    tracker.hydrate("t.jsonl", {"custom": "chosen"})
    assert tracker.effective("t.jsonl", "claude-code") == "chosen"


# --------------------------------------------------------------------------- #
# codex name index                                                             #
# --------------------------------------------------------------------------- #

from agent_harness.codex_names import CodexNameIndex, default_codex_name_index  # noqa: E402
from agent_harness.settings import ObserverSettings  # noqa: E402

T1 = "123e4567-e89b-12d3-a456-426614174000"
T2 = "223e4567-e89b-12d3-a456-426614174000"
CODEX_ID = f"codex_{T1}"


def _entry(thread_id, name, updated_at="2026-10-07T10:00:00Z"):
    return {"id": thread_id, "thread_name": name, "updated_at": updated_at}


def _replace(path: Path, records: list, *, raw: str = "") -> None:
    """codex's removal: write a temp file, rename it over the index."""
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(r) + "\n" for r in records) + raw, encoding="utf-8")
    os.replace(temp, path)


def _rollout(home: Path, thread_id: str = T1) -> Path:
    path = (
        home / ".codex" / "sessions" / "2026" / "10" / "07"
        / f"rollout-2026-10-07T10-00-00-{thread_id}.jsonl"
    )
    _write(
        path,
        [
            {"type": "turn_context", "payload": {"cwd": "/repo", "model": "gpt-x"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
        ],
    )
    return path


def _bump_mtime(path: Path) -> None:
    """Make a same-size rewrite visible to stat-based change detection.

    Filesystems with coarse timestamps (ZFS here) keep the mtime of a
    rewrite made within the same tick, so the test advances it explicitly.
    """
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_index_latest_entry_in_file_order_wins(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [
        _entry(T1, "newer by clock", "2030-01-01T00:00:00Z"),
        _entry(T1, "latest in file", "2001-01-01T00:00:00Z"),
        _entry(T2.upper(), "upper-case id"),
    ])
    index = CodexNameIndex(index_path)

    assert index.refresh() == {T1: "latest in file", T2: "upper-case id"}
    assert index.get(T1.upper()) == "latest in file"
    assert index.trusted


def test_index_ignores_invalid_entries_and_defers_partial_lines(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "kept")])
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write(json.dumps(_entry("not-a-uuid", "x")) + "\n")
        handle.write(json.dumps({"id": T1, "thread_name": 5}) + "\n")
        handle.write(json.dumps({"id": T1}) + "\n")
        handle.write(json.dumps(_entry(T1, "partial")))
    index = CodexNameIndex(index_path)

    assert index.refresh() == {T1: "kept"}
    assert not index.trusted

    with index_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert index.refresh() == {T1: "partial"}


def test_index_blank_name_is_a_clear(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "named")])
    index = CodexNameIndex(index_path)
    index.refresh()

    _append(index_path, _entry(T1, "  "))
    assert index.refresh() == {T1: CLEARED}
    _append(index_path, _entry(T1, "again"))
    assert index.refresh() == {T1: "again"}


def test_index_absent_then_created_then_appended(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    index = CodexNameIndex(index_path)
    assert index.refresh() == {}
    assert not index.trusted

    _write(index_path, [_entry(T1, "one")])
    assert index.refresh() == {T1: "one"}
    assert index.refresh() == {}  # unchanged file: nothing re-read

    _append(index_path, _entry(T2, "two"), _entry(T1, "one"))
    assert index.refresh() == {T2: "two"}


def test_index_atomic_removal_clears_the_removed_thread(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "one"), _entry(T2, "two")])
    index = CodexNameIndex(index_path)
    index.refresh()

    _replace(index_path, [_entry(T2, "two")])

    assert index.refresh() == {T1: CLEARED}
    assert index.get(T2) == "two"


def test_index_removal_keeps_pre_existing_malformed_lines(tmp_path) -> None:
    """codex's rewrite keeps lines it can't parse, so they don't block it."""
    index_path = tmp_path / "session_index.jsonl"
    index_path.write_text(
        "junk\n" + json.dumps(_entry(T1, "one")) + "\n" + json.dumps(_entry(T2, "two")) + "\n",
        encoding="utf-8",
    )
    index = CodexNameIndex(index_path)
    index.refresh()

    _replace(index_path, [_entry(T2, "two")], raw="junk\n")

    assert index.refresh() == {T1: CLEARED}


@pytest.mark.parametrize(
    "damage",
    ["new garbage", "partial tail", "in-place truncation", "in-place rewrite", "deleted"],
)
def test_index_untrustworthy_changes_never_infer_removals(tmp_path, damage) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "one"), _entry(T2, "two")])
    index = CodexNameIndex(index_path)
    index.refresh()

    if damage == "new garbage":
        _replace(index_path, [_entry(T2, "two")], raw="garbage\n")
    elif damage == "partial tail":
        _replace(index_path, [_entry(T2, "two")], raw='{"id": "')
    elif damage == "in-place truncation":
        with index_path.open("r+b") as handle:
            handle.truncate(0)
    elif damage == "in-place rewrite":
        original = index_path.read_bytes()
        rewritten = json.dumps(_entry(T2, "TWO")).encode() + b"\n"
        with index_path.open("r+b") as handle:
            handle.write(rewritten.ljust(len(original) - 1) + b"\n")
            handle.truncate()
        _bump_mtime(index_path)
    else:
        index_path.unlink()
        index.refresh()
        _write(index_path, [_entry(T2, "two")])  # recreated afresh

    index.refresh()

    assert index.get(T1) == "one"
    if damage == "in-place rewrite":
        assert index.get(T2) == "TWO"


def test_index_unreadable_file_changes_nothing(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "one")])
    index = CodexNameIndex(index_path)
    index.refresh()
    index_path.unlink()
    index_path.mkdir()  # exists, but cannot be read as a file

    assert index.refresh() == {}
    assert index.get(T1) == "one"


@pytest.mark.asyncio
async def test_codex_name_before_discovery_is_applied_at_discovery(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        home = tmp_path / kind
        index_path = home / ".codex" / "session_index.jsonl"
        _write(index_path, [_entry(T1, "named thread"), _entry(T2, "no rollout")])
        observer = _observer(repository, bus, codex_name_index=index_path)
        await observer.freshness_tick()
        assert not repository.has_session(CODEX_ID), kind  # no phantom session

        published = await observer.tail_file(_rollout(home))

        assert _session_payloads(published)[-1] == ("named thread", "native"), kind
        assert _title(repository, CODEX_ID) == ("named thread", "native"), kind
        assert not repository.has_session(f"codex_{T2}"), kind


@pytest.mark.asyncio
async def test_codex_live_rename_clear_and_removal(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        home = tmp_path / kind
        index_path = home / ".codex" / "session_index.jsonl"
        clock = _Clock()
        observer = _observer(repository, bus, codex_name_index=index_path, clock=clock)
        await observer.tail_file(_rollout(home))
        clock.now += timedelta(seconds=31)
        await observer.freshness_tick()
        before = repository.get_session(CODEX_ID)
        last_seen = observer._last_event_at[CODEX_ID]
        assert before.status == "idle", kind

        _write(index_path, [_entry(T1, "renamed")])  # index created late
        await observer.freshness_tick()
        assert _title(repository, CODEX_ID) == ("renamed", "native"), kind

        _append(index_path, _entry(T1, ""))
        await observer.freshness_tick()
        assert _title(repository, CODEX_ID) == (None, "native"), kind

        _append(index_path, _entry(T1, "back"), _entry(T2, "other"))
        await observer.freshness_tick()
        assert _title(repository, CODEX_ID) == ("back", "native"), kind

        _replace(index_path, [_entry(T2, "other")])
        published = await observer._refresh_codex_names()
        assert _session_payloads(published) == [(None, "native")], kind

        after = repository.get_session(CODEX_ID)
        assert (after.status, after.updated_at, after.stats) == (
            "idle", before.updated_at, before.stats,
        ), kind
        assert observer._last_event_at[CODEX_ID] == last_seen, kind


@pytest.mark.asyncio
async def test_codex_names_never_touch_explicit_or_harness_sessions(tmp_path) -> None:
    for kind, repository, bus in _repos(tmp_path):
        home = tmp_path / kind
        index_path = home / ".codex" / "session_index.jsonl"
        observer = _observer(repository, bus, codex_name_index=index_path)
        await observer.tail_file(_rollout(home))
        repository.patch_session(CODEX_ID, {"title": "bridge", "title_source": None})
        repository.upsert_session(
            Session(id="ses_harness", backend="codex", project=Project(path="/r", name="r"),
                    origin="harness", codex_resume_id=T2)
        )

        _write(index_path, [_entry(T1, "native"), _entry(T2, "native 2")])
        published = await observer._refresh_codex_names()
        _append(index_path, _entry(T1, ""))
        published += await observer._refresh_codex_names()

        assert published == [], kind
        assert _title(repository, CODEX_ID) == ("bridge", None), kind
        assert _title(repository, "ses_harness") == (None, None), kind


def _codex_db_with_named_session(tmp_path: Path, index_records, *, raw: str = "") -> Path:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    repository.upsert_session(
        Session(id=CODEX_ID, backend="codex", project=Project(path="/repo", name="repo"),
                origin="external", codex_resume_id=T1, title="stored", title_source="native",
                status="idle")
    )
    repository.close()
    index_path = tmp_path / "session_index.jsonl"
    if index_records is not None:
        _write(index_path, index_records)
        if raw:
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(raw)
    return db_path


@pytest.mark.parametrize(
    ("records", "raw", "expected"),
    [
        ([_entry(T1, "current")], "", ("current", "native")),
        ([_entry(T1, "")], "", (None, "native")),
        ([_entry(T2, "other")], "", (None, "native")),  # removed while down
        ([_entry(T2, "other")], "junk\n", ("stored", "native")),  # damaged: no inference
        ([_entry(T2, "other")], '{"id"', ("stored", "native")),  # partial: no inference
        (None, "", ("stored", "native")),  # missing index: no inference
    ],
    ids=["renamed", "cleared", "removed", "malformed", "partial", "missing"],
)
def test_codex_startup_reconcile(tmp_path, records, raw, expected) -> None:
    db_path = _codex_db_with_named_session(tmp_path, records, raw=raw)
    repository = open_sqlite_repository(db_path)
    try:
        before = repository.get_session(CODEX_ID)
        _observer(repository, DurableEventBus(repository),
                  codex_name_index=tmp_path / "session_index.jsonl")
        after = repository.get_session(CODEX_ID)
        assert (after.title, after.title_source) == expected
        assert (after.status, after.updated_at) == (before.status, before.updated_at)
    finally:
        repository.close()


def test_codex_index_is_off_without_configuration(tmp_path) -> None:
    db_path = _codex_db_with_named_session(tmp_path, [_entry(T1, "")])
    repository = open_sqlite_repository(db_path)
    try:
        _observer(repository, DurableEventBus(repository))
        assert _title(repository, CODEX_ID) == ("stored", "native")
    finally:
        repository.close()


def test_codex_index_path_follows_the_observed_codex_root(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    defaults = ObserverSettings.default_transcript_roots(home=tmp_path)
    assert defaults.codex_name_index_path() == tmp_path / ".codex" / "session_index.jsonl"

    no_codex = ObserverSettings.from_roots([tmp_path / ".claude" / "projects"])
    assert no_codex.codex_name_index_path() is None

    custom_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(custom_home))
    custom = ObserverSettings.from_roots([custom_home / "sessions"])
    assert custom.codex_name_index_path() == custom_home / "session_index.jsonl"
    assert default_codex_name_index() == custom_home / "session_index.jsonl"

    explicit = ObserverSettings.from_roots(
        [tmp_path / "elsewhere"], codex_name_index=tmp_path / "names.jsonl"
    )
    assert explicit.codex_name_index_path() == tmp_path / "names.jsonl"


def test_cli_passes_an_explicit_codex_name_index(monkeypatch, tmp_path) -> None:
    from agent_harness.cli import main

    root = tmp_path / "transcripts"
    root.mkdir()
    captured = {}
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "agent_harness.cli.create_app", lambda **kwargs: captured.update(kwargs) or object()
    )
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", lambda *a, **k: None)

    assert main(["serve", "--observe-root", str(root),
                 "--codex-name-index", str(tmp_path / "n.jsonl")]) == 0

    settings = captured["observer_settings"]
    assert settings.roots == (root,)
    assert settings.codex_name_index_path() == tmp_path / "n.jsonl"


def test_index_retained_names_are_not_removed_by_a_later_rewrite(tmp_path) -> None:
    """T1 survives a damaged replacement; a later clean rewrite that never
    contained T1 must not be read as removing it."""
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "one"), _entry(T2, "two")])
    index = CodexNameIndex(index_path)
    index.refresh()

    _replace(index_path, [_entry(T2, "two")], raw='{"id": "' + T1 + '", broken\n')
    assert index.refresh() == {}
    _replace(index_path, [_entry(T2, "two!")], raw='{"id": "' + T1 + '", broken\n')
    assert index.refresh() == {T2: "two!"}
    assert index.get(T1) == "one"


def test_index_recreated_file_starts_a_fresh_baseline(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "one"), _entry(T2, "two")])
    index = CodexNameIndex(index_path)
    index.refresh()
    index_path.unlink()
    index.refresh()
    _write(index_path, [_entry(T2, "two")])
    index.refresh()

    _replace(index_path, [_entry(T2, "two!")])
    assert index.refresh() == {T2: "two!"}
    assert index.get(T1) == "one"
    # ...while a thread present in that baseline is still removable.
    _replace(index_path, [])
    assert index.refresh() == {T2: CLEARED}


def test_index_in_place_rewrite_before_the_tail_probe_is_seen(tmp_path) -> None:
    index_path = tmp_path / "session_index.jsonl"
    _write(index_path, [_entry(T1, "old"), _entry(T2, "x" * 200)])
    index = CodexNameIndex(index_path)
    index.refresh()

    content = index_path.read_bytes().replace(b'"old"', b'"new"')
    with index_path.open("r+b") as handle:
        handle.write(content)
    _bump_mtime(index_path)

    assert index.refresh() == {T1: "new"}
