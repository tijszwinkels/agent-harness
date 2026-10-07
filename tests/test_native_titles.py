"""Backend-native conversation names → ``Session.title``.

pi records its conversation name as ``session_info`` records (``/name``,
``--name``, ``pi.setSessionName()``); claude records ``/rename`` as
``custom-title``. Both reach ``Session.title`` with
``title_source="native"`` on observed sessions, without ever overriding
an explicit (client-set) title and without reading as conversation
activity. All transcripts and databases here are synthetic.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import (
    Project,
    Session,
    merge_observed_session,
)
from agent_harness.native_titles import (
    CLEARED,
    effective_native_title,
    native_name,
    native_title_slot,
    scan_native_titles,
)
from agent_harness.observer import (
    ExternalTranscriptObserver,
    claude_transcript_path,
    pi_transcript_path,
)
from agent_harness.pi_discovery import PiTranscriptRegistry, read_pi_head_facts
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository

UUID = "e5a93149-9a70-4aef-a189-2681b4e08525"
SESSION_ID = "ses_e5a931499a704aefa1892681b4e08525"
CWD = "/home/me/project"
TS = "2026-09-14T11-25-51-562Z"

SESSION_RECORD = {"type": "session", "version": 3, "id": UUID, "cwd": CWD}
MODEL_CHANGE = {"type": "model_change", "provider": "ollama", "modelId": "glm-5.2:cloud"}
USER_RECORD = {
    "type": "message",
    "message": {"role": "user", "content": [{"type": "text", "text": "a question"}]},
}
ASSISTANT_RECORD = {
    "type": "message",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "an answer"}],
        "provider": "ollama",
        "model": "glm-5.2:cloud",
        "stopReason": "stop",
    },
}


def _name(name: object) -> dict:
    return {"type": "session_info", "id": "x", "parentId": None, "name": name}


def _transcript(home: Path) -> Path:
    path = pi_transcript_path(CWD, TS, UUID, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _append(path: Path, *records: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _observer(repository, *, clock=None, bus=None) -> ExternalTranscriptObserver:
    return ExternalTranscriptObserver(
        bus or InMemoryEventBus(),
        repository=repository,
        idle_after_seconds=30.0,
        **({"clock": clock} if clock else {}),
    )


def _titles(events) -> list[str | None]:
    return [
        e.data["session"].get("title")
        for e in events
        if e.event == "session.updated" and "session" in e.data
    ]


# --------------------------------------------------------------------------- #
# Extraction                                                                   #
# --------------------------------------------------------------------------- #


def test_session_info_name_fills_the_pi_slot() -> None:
    assert native_title_slot("pi", _name("Refactor auth")) == ("name", "Refactor auth")


@pytest.mark.parametrize("name", ["", "   ", "\n\t"], ids=repr)
def test_an_explicit_blank_name_is_a_removal(name) -> None:
    assert native_title_slot("pi", _name(name)) == ("name", CLEARED)


@pytest.mark.parametrize("name", [None, 42, ["x"], {"name": "x"}], ids=repr)
def test_missing_or_malformed_names_say_nothing(name) -> None:
    assert native_title_slot("pi", _name(name)) is None
    assert native_title_slot("pi", {"type": "session_info"}) is None
    assert native_title_slot("pi", ["session_info"]) is None


def test_names_are_normalized_to_one_line() -> None:
    assert native_name("  Fix\nthe   bug \r\n") == "Fix the bug"


def test_native_title_sources_per_backend() -> None:
    assert native_title_slot("claude-code", {"type": "custom-title", "customTitle": "c"}) == (
        "custom", "c",
    )
    assert native_title_slot("claude-code", {"type": "ai-title", "aiTitle": "g"}) == ("ai", "g")
    # A record type means nothing outside its own backend.
    assert native_title_slot("claude-code", _name("p")) is None
    assert native_title_slot("codex", _name("p")) is None


def test_slot_preference_and_fallback() -> None:
    assert effective_native_title("claude-code", {"ai": "g", "custom": "c"}) == "c"
    assert effective_native_title("claude-code", {"custom": CLEARED, "ai": "g"}) == "g"
    assert effective_native_title("claude-code", {"custom": "c", "ai": CLEARED}) == "c"
    assert effective_native_title("claude-code", {"custom": CLEARED}) is CLEARED
    assert effective_native_title("claude-code", {"ai": CLEARED}) is CLEARED
    assert effective_native_title("claude-code", {}) is None
    assert effective_native_title("pi", {"name": CLEARED}) is CLEARED


def test_pi_announcements_carry_no_title() -> None:
    """An announcement marks the session running; a rename is not activity."""
    registry = PiTranscriptRegistry()
    registry.observe("t.jsonl", SESSION_RECORD)
    assert registry.take_announcement("t.jsonl") is not None
    registry.observe("t.jsonl", _name("renamed"))
    assert registry.take_announcement("t.jsonl") is None


def test_title_scan_reads_complete_records_up_to_an_offset(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [SESSION_RECORD, _name("one"), USER_RECORD, _name("two"), _name(7)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ not json session_info\n")
    end_of_two = sum(len(line) for line in path.read_bytes().splitlines(keepends=True)[:4])
    _append(path, _name("  "))
    size = path.stat().st_size

    def scan(end):
        return scan_native_titles(path, "pi", end_offset=end)

    assert scan(size) == {"name": CLEARED}
    assert scan(end_of_two) == {"name": "two"}
    assert scan(0) == {}
    # A prefix that can't be recovered in full is unknown, not empty:
    # mid-line, beyond the end of the file, or unreadable.
    assert scan(10) is None
    assert scan(size + 1) is None
    assert scan_native_titles(tmp_path / "missing.jsonl", "pi", end_offset=99) is None


# --------------------------------------------------------------------------- #
# Provenance / precedence                                                      #
# --------------------------------------------------------------------------- #


def _external(**update) -> Session:
    return Session(
        id=SESSION_ID,
        backend="pi",
        project=Project(path=CWD, name="project"),
        origin="external",
        **update,
    )


def test_merge_applies_a_native_title_to_an_untitled_or_native_session() -> None:
    incoming = _external(title="native", title_source="native")
    assert merge_observed_session(incoming, _external()).title == "native"
    renamed = merge_observed_session(
        incoming, _external(title="older", title_source="native")
    )
    assert (renamed.title, renamed.title_source) == ("native", "native")


def test_merge_never_replaces_an_explicit_title() -> None:
    incoming = _external(title="native", title_source="native")
    kept = merge_observed_session(incoming, _external(title="bridge channel"))
    assert (kept.title, kept.title_source) == ("bridge channel", None)


def test_merge_ignores_untitled_observations() -> None:
    kept = merge_observed_session(_external(), _external(title="n", title_source="native"))
    assert (kept.title, kept.title_source) == ("n", "native")


def test_fork_inherits_provenance_only_with_the_title() -> None:
    parent = _external(title="n", title_source="native")
    assert Session.forked_child(parent).title_source == "native"
    assert Session.forked_child(parent, title="explicit").title_source is None


def test_patching_a_title_makes_it_explicit() -> None:
    repository = InMemoryRepository()
    repository.upsert_session(_external(title="native", title_source="native"))
    client = TestClient(create_app(repository=repository))

    response = client.patch(f"/v1/sessions/{SESSION_ID}", json={"title": "mine"})

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "mine"
    assert response.json()["title_source"] is None
    # Other fields leave provenance alone.
    repository.upsert_session(_external(title="native", title_source="native"))
    client.patch(f"/v1/sessions/{SESSION_ID}", json={"effort": "high"})
    assert repository.get_session(SESSION_ID).title_source == "native"


# --------------------------------------------------------------------------- #
# Observer: discovery and live renames                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_discovery_carries_the_latest_name(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(
        transcript,
        [SESSION_RECORD, _name("first"), USER_RECORD, MODEL_CHANGE, ASSISTANT_RECORD,
         _name("second"), _name(None), _name(7)],
    )
    repository = InMemoryRepository()

    await _observer(repository).tail_file(transcript)

    session = repository.get_session(SESSION_ID)
    assert (session.title, session.title_source) == ("second", "native")
    assert [m.role for m in repository.list_messages(SESSION_ID)] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_live_rename_is_not_activity(tmp_path) -> None:
    """A rename on an idle session updates the title and publishes it, but
    leaves status, ``updated_at`` and the freshness clock untouched."""
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, USER_RECORD, ASSISTANT_RECORD])
    repository = InMemoryRepository()
    clock = _Clock()
    observer = _observer(repository, clock=clock)
    await observer.tail_file(transcript)
    clock.now += timedelta(seconds=31)
    await observer.freshness_tick()
    before = repository.get_session(SESSION_ID)
    assert before.status == "idle"
    last_seen = observer._last_event_at[SESSION_ID]

    clock.now += timedelta(minutes=5)
    _append(transcript, _name("Renamed later"))
    published = await observer.tail_file(transcript)

    assert [e.event for e in published] == ["session.updated"]
    assert _titles(published) == ["Renamed later"]
    after = repository.get_session(SESSION_ID)
    assert (after.title, after.title_source) == ("Renamed later", "native")
    assert after.status == "idle"
    assert after.updated_at == before.updated_at
    assert after.stats == before.stats
    assert observer._last_event_at[SESSION_ID] == last_seen
    clock.now += timedelta(seconds=31)
    await observer.freshness_tick()
    assert repository.get_session(SESSION_ID).status == "idle"


@pytest.mark.asyncio
async def test_metadata_only_transcript_is_not_kept_working_by_renames(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD])
    repository = InMemoryRepository()
    clock = _Clock()
    observer = _observer(repository, clock=clock)
    await observer.tail_file(transcript)
    clock.now += timedelta(seconds=31)
    await observer.freshness_tick()

    for name in ("a", "b"):
        clock.now += timedelta(seconds=5)
        _append(transcript, _name(name))
        await observer.tail_file(transcript)

    session = repository.get_session(SESSION_ID)
    assert (session.title, session.status) == ("b", "idle")


@pytest.mark.asyncio
async def test_malformed_and_repeated_names_publish_nothing(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("same")])
    repository = InMemoryRepository()
    observer = _observer(repository)
    await observer.tail_file(transcript)

    _append(transcript, _name(5), _name(None), _name("same"), {"type": "session_info"})
    published = await observer.tail_file(transcript)

    assert published == []
    assert repository.get_session(SESSION_ID).title == "same"


@pytest.mark.asyncio
async def test_rename_never_overrides_an_explicit_title(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("native")])
    repository = InMemoryRepository()
    observer = _observer(repository)
    await observer.tail_file(transcript)
    repository.patch_session(SESSION_ID, {"title": "explicit", "title_source": None})

    _append(transcript, _name("native again"))
    renamed = await observer.tail_file(transcript)
    # A conversation announcement carrying the native name can't either.
    _append(transcript, {"type": "model_change", "provider": "p", "modelId": "m"})
    await observer.tail_file(transcript)

    assert _titles(renamed) == []
    session = repository.get_session(SESSION_ID)
    assert (session.title, session.title_source) == ("explicit", None)
    assert session.model == "p/m"


@pytest.mark.asyncio
async def test_harness_owned_sessions_keep_their_title(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("native")])
    repository = InMemoryRepository()
    repository.upsert_session(
        Session(
            id=SESSION_ID,
            backend="pi",
            project=Project(path=CWD, name="project"),
            origin="harness",
        )
    )

    published = await _observer(repository).tail_file(transcript)

    assert [e.event for e in published] == []
    session = repository.get_session(SESSION_ID)
    assert (session.title, session.origin) == (None, "harness")


# --------------------------------------------------------------------------- #
# Restart: hydration and startup backfill                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_native_title_survives_restart_and_tracks_later_renames(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, _name("before")])
    repository = InMemoryRepository()
    first = _observer(repository)
    await first.tail_file(transcript)

    restarted = _observer(repository)
    restarted._state.set_next_offset(transcript, transcript.stat().st_size)
    # A model switch re-announces with the hydrated native name intact...
    _append(transcript, {"type": "model_change", "provider": "p", "modelId": "m"})
    await restarted.tail_file(transcript)
    assert repository.get_session(SESSION_ID).title == "before"
    # ...and a later rename still lands.
    _append(transcript, _name("after"))
    await restarted.tail_file(transcript)
    assert repository.get_session(SESSION_ID).title == "after"


@pytest.mark.asyncio
async def test_unknown_session_resumed_mid_file_recovers_its_name(tmp_path) -> None:
    """Offset persisted, session row missing: the name behind the offset
    (beyond the bounded head peek) is still recovered."""
    transcript = _transcript(tmp_path)
    filler = [{"type": "custom", "n": i} for i in range(80)]
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, *filler, _name("deep name")])
    repository = InMemoryRepository()
    observer = _observer(repository)
    observer._state.set_next_offset(transcript, transcript.stat().st_size)

    _append(transcript, USER_RECORD)
    await observer.tail_file(transcript)

    session = repository.get_session(SESSION_ID)
    assert (session.title, session.title_source) == ("deep name", "native")


@pytest.mark.asyncio
async def test_startup_backfill_names_pre_existing_sessions(tmp_path) -> None:
    """Rows written before native titles were read: the name sits behind
    the persisted offset and would never be tailed again."""
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, MODEL_CHANGE, USER_RECORD, _name("old name")])
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    observer = _observer(repository, bus=DurableEventBus(repository))
    await observer.tail_file(transcript)
    legacy = repository.get_session(SESSION_ID).model_copy(
        update={"title": None, "title_source": None, "status": "idle"}
    )
    repository.upsert_session(legacy)
    # Written while the harness was down: still ahead of the offset.
    _append(transcript, _name("newer name"))
    repository.close()

    reopened = open_sqlite_repository(db_path)
    try:
        restarted = _observer(reopened, bus=DurableEventBus(reopened))
        session = reopened.get_session(SESSION_ID)
        assert (session.title, session.title_source) == ("old name", "native")
        assert session.status == "idle"
        assert session.updated_at == legacy.updated_at

        published = await restarted.tail_file(transcript)
        assert _titles(published) == ["newer name"]
        assert reopened.get_session(SESSION_ID).title == "newer name"
        assert reopened.get_session(SESSION_ID).status == "idle"
    finally:
        reopened.close()


def _backfill_repository(transcript: Path, *, offset: int | None, **update):
    # SQLite: the offset store lives there, as in production.
    repository = open_sqlite_repository(transcript.with_suffix(".db"))
    if offset is not None:
        repository.set_observer_offset(str(transcript), offset)
    repository.upsert_session(
        Session(
            id=SESSION_ID,
            backend="pi",
            project=Project(path=CWD, name="project"),
            pi_transcript_path=str(transcript.resolve()),
            **{"origin": "external", **update},
        )
    )
    return repository


@pytest.mark.parametrize(
    "update",
    [{"title": "explicit"}, {"origin": "harness"}, {"origin": "harness", "title": "bridge"}],
    ids=["explicit", "harness", "harness-titled"],
)
def test_startup_backfill_leaves_explicit_and_harness_sessions_alone(tmp_path, update) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("native")])
    repository = _backfill_repository(transcript, offset=transcript.stat().st_size, **update)
    before = repository.get_session(SESSION_ID)

    _observer(repository)

    assert repository.get_session(SESSION_ID) == before


def test_startup_backfill_names_an_untitled_external_session(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("native")])
    repository = _backfill_repository(transcript, offset=transcript.stat().st_size)

    _observer(repository)

    session = repository.get_session(SESSION_ID)
    assert (session.title, session.title_source) == ("native", "native")


@pytest.mark.asyncio
async def test_startup_backfill_never_reads_ahead_of_an_unconsumed_transcript(tmp_path) -> None:
    """No consumed prefix (offset missing or zero): the tail replays every
    name in order, so the backfill must not jump ahead to the last one."""
    for offset in (None, 0):
        home = tmp_path / f"home-{offset}"
        transcript = _transcript(home)
        _write(transcript, [SESSION_RECORD, _name("first"), _name("second")])
        repository = _backfill_repository(transcript, offset=offset)
        _observer(repository)
        assert repository.get_session(SESSION_ID).title is None


def test_startup_backfill_uses_the_watched_spelling_of_the_path(tmp_path) -> None:
    """Offsets are keyed by the watched spelling (here under a symlinked
    root); ``pi_transcript_path`` is stored resolved."""
    real = tmp_path / "real"
    transcript = _transcript(real)
    _write(transcript, [SESSION_RECORD, _name("first")])
    end_of_first = transcript.stat().st_size
    _append(transcript, _name("unread"))
    link = tmp_path / "link"
    link.symlink_to(real)
    watched = link / transcript.relative_to(real)
    repository = _backfill_repository(transcript, offset=None)
    repository.set_observer_offset(str(watched), end_of_first)

    _observer(repository)

    assert repository.get_session(SESSION_ID).title == "first"


# --------------------------------------------------------------------------- #
# claude ``custom-title``                                                      #
# --------------------------------------------------------------------------- #

CLAUDE_UUID = "0f8fad5b-d9cb-469f-a165-70867728950e"
CLAUDE_SESSION_ID = "ses_0f8fad5bd9cb469fa16570867728950e"


def _claude_transcript(home: Path) -> Path:
    path = claude_transcript_path(CWD, CLAUDE_UUID, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(
        path,
        [{"type": "user", "cwd": CWD, "message": {"role": "user", "model": "claude-x",
                                                  "content": "hi"}}],
    )
    return path


@pytest.mark.asyncio
async def test_claude_rename_titles_an_external_session(tmp_path) -> None:
    transcript = _claude_transcript(tmp_path)
    repository = InMemoryRepository()
    observer = _observer(repository)
    await observer.tail_file(transcript)

    _append(
        transcript,
        {"type": "ai-title", "aiTitle": "generated", "sessionId": CLAUDE_UUID},
        {"type": "custom-title", "customTitle": "Chosen\nname", "sessionId": CLAUDE_UUID},
        {"type": "ai-title", "aiTitle": "regenerated", "sessionId": CLAUDE_UUID},
    )
    published = await observer.tail_file(transcript)

    # The custom title outranks a later generated one.
    assert _titles(published) == ["generated", "Chosen name"]
    session = repository.get_session(CLAUDE_SESSION_ID)
    assert (session.title, session.title_source) == ("Chosen name", "native")


@pytest.mark.asyncio
async def test_claude_rename_leaves_harness_sessions_alone(tmp_path) -> None:
    transcript = _claude_transcript(tmp_path)
    repository = InMemoryRepository()
    repository.upsert_session(
        Session(
            id=CLAUDE_SESSION_ID,
            backend="claude-code",
            project=Project(path=CWD, name="project"),
            origin="harness",
            title="bridge channel",
        )
    )
    observer = _observer(repository)
    observer.bind_rollout(transcript, CLAUDE_SESSION_ID)

    _append(transcript, {"type": "custom-title", "customTitle": "native"})
    await observer.tail_file(transcript)

    assert repository.get_session(CLAUDE_SESSION_ID).title == "bridge channel"


# --------------------------------------------------------------------------- #
# Review regressions                                                           #
# --------------------------------------------------------------------------- #


def _repositories(tmp_path):
    yield "memory", InMemoryRepository(), None
    sqlite = open_sqlite_repository(tmp_path / "harness.db")
    yield "sqlite", sqlite, DurableEventBus(sqlite)


@pytest.mark.asyncio
async def test_published_payloads_respect_an_explicit_title(tmp_path) -> None:
    """Announcements after a PATCH must advertise what the row stores, not
    the transcript's name: subscribers use the payload as the label."""
    for kind, repository, bus in _repositories(tmp_path):
        transcript = _transcript(tmp_path / kind)
        _write(transcript, [SESSION_RECORD, _name("native")])
        observer = _observer(repository, bus=bus)
        await observer.tail_file(transcript)
        repository.patch_session(SESSION_ID, {"title": "channel", "title_source": None})

        _append(transcript, MODEL_CHANGE, ASSISTANT_RECORD, _name("native 2"))
        published = await observer.tail_file(transcript)

        sessions = [e.data["session"] for e in published if e.event == "session.updated"]
        assert sessions, kind
        assert {(s["title"], s["title_source"]) for s in sessions} == {("channel", None)}, kind
        stored = repository.get_session(SESSION_ID)
        assert (stored.title, stored.title_source) == ("channel", None), kind


@pytest.mark.asyncio
async def test_published_payloads_carry_a_native_title_once_accepted(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("native")])
    observer = _observer(InMemoryRepository())
    await observer.tail_file(transcript)

    _append(transcript, MODEL_CHANGE)
    published = await observer.tail_file(transcript)

    assert [(e.data["session"]["title"], e.data["session"]["title_source"])
            for e in published] == [("native", "native")]


@pytest.mark.asyncio
async def test_discovery_publishes_names_in_transcript_order(tmp_path) -> None:
    for kind, repository, bus in _repositories(tmp_path):
        transcript = _transcript(tmp_path / kind)
        _write(transcript, [SESSION_RECORD, _name("first"), _name("second")])

        published = await _observer(repository, bus=bus).tail_file(transcript)

        assert _titles(published) == [None, "first", "second"], kind
        assert repository.get_session(SESSION_ID).title == "second", kind


@pytest.mark.asyncio
async def test_an_unterminated_name_is_not_applied_early(tmp_path) -> None:
    transcript = _transcript(tmp_path)
    _write(transcript, [SESSION_RECORD, _name("done")])
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_name("in flight")))
    complete = transcript.read_bytes().rindex(b"\n") + 1
    assert scan_native_titles(transcript, "pi", end_offset=complete) == {"name": "done"}
    assert scan_native_titles(transcript, "pi", end_offset=transcript.stat().st_size) is None
    repository = InMemoryRepository()
    observer = _observer(repository)

    await observer.tail_file(transcript)
    assert repository.get_session(SESSION_ID).title == "done"

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    await observer.tail_file(transcript)
    assert repository.get_session(SESSION_ID).title == "in flight"


def test_head_peek_carries_no_title(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [SESSION_RECORD, _name("ahead")])
    assert not hasattr(read_pi_head_facts(path), "title")
