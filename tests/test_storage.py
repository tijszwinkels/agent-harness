import sqlite3

from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Message, Project, Session
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import SCHEMA_VERSION, open_sqlite_repository


def test_sqlite_repository_persists_sessions_runs_and_messages_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    session = repository.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
            title="Durable work",
        )
    )
    run = repository.create_run(session.id, CreateRunRequest(message="persist me"))
    completed = repository.finish_run(session.id, run.id, status="completed", stop_reason="end_turn")
    archived = repository.archive_session(session.id)
    repository.close()

    reopened = open_sqlite_repository(db_path)

    assert reopened.has_session(session.id)
    assert not reopened.has_session("ses_missing")
    assert reopened.get_session(session.id).status == archived.status == "archived"
    assert reopened.list_sessions()[0].id == session.id
    reopened_run = reopened.get_run(session.id, run.id)
    assert reopened_run.status == completed.status == "completed"
    assert reopened_run.stop_reason == "end_turn"
    assert reopened_run.completed_at is not None
    assert [item.id for item in reopened.list_runs(session.id)] == [run.id]
    messages = reopened.list_messages(session.id)
    assert [message.role for message in messages] == ["user"]
    assert messages[0].blocks[0].text == "persist me"
    reopened.close()


def test_sqlite_repository_persists_materialized_external_events_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    session = Session(
        id="codex_external",
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/repo", name="repo"),
        origin="external",
    )
    message = Message.user("observed")

    repository.materialize_event(
        Event(sequence=1, event="session.updated", session_id=session.id, data={"session": session.model_dump(mode="json")})
    )
    repository.materialize_event(
        Event(sequence=2, event="message", session_id=session.id, data={"message": message.model_dump(mode="json")})
    )
    repository.close()

    reopened = open_sqlite_repository(db_path)

    assert reopened.get_session(session.id).origin == "external"
    assert [item.id for item in reopened.list_messages(session.id)] == [message.id]
    assert [event.sequence for event in reopened.list_events(session_id=session.id)] == [1, 2]
    reopened.close()


def test_sqlite_repository_append_event_allocates_next_sequence_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    first = repository.append_event(Event(event="session.updated", session_id="ses_a", data={}))
    repository.close()

    reopened = open_sqlite_repository(db_path)
    second = reopened.append_event(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))

    assert [first.sequence, second.sequence] == [1, 2]
    assert [event.sequence for event in reopened.list_events()] == [1, 2]
    reopened.close()


def test_sqlite_repository_reports_max_event_sequence(tmp_path) -> None:
    repository = open_sqlite_repository(tmp_path / "harness.db")

    assert repository.max_sequence() == 0
    assert repository.max_sequence(session_id="ses_a") == 0

    repository.append_event(Event(event="run.started", session_id="ses_a", run_id="run_a", data={}))
    repository.append_event(Event(event="run.started", session_id="ses_b", run_id="run_b", data={}))
    repository.append_event(Event(event="message", session_id="ses_a", run_id="run_a", data={}))

    assert repository.max_sequence() == 3
    assert repository.max_sequence(session_id="ses_a") == 3
    assert repository.max_sequence(session_id="ses_b") == 2
    assert repository.max_sequence(session_id="ses_missing") == 0
    repository.close()


def test_sqlite_repository_skips_duplicate_observed_messages(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    session = repository.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
        )
    )
    message = Message.user("same")

    repository.add_message(session.id, message)
    repository.add_message(session.id, Message.user("between"))
    repository.materialize_event(
        Event(sequence=1, event="message", session_id=session.id, data={"message": message.model_dump(mode="json")})
    )

    assert [item.blocks[0].text for item in repository.list_messages(session.id)] == ["same", "between"]
    assert repository.get_session(session.id).stats.messages == 2
    repository.close()


def test_sqlite_repository_propagates_bypass_permissions(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    session = repository.create_session(
        CreateSessionRequest(
            backend="claude-code",
            model="claude-opus-4-7",
            project=Project(path="/repo", name="repo"),
            bypass_permissions=True,
        )
    )
    assert session.bypass_permissions is True

    repository.close()
    reopened = open_sqlite_repository(db_path)
    assert reopened.get_session(session.id).bypass_permissions is True
    reopened.close()


def test_materialize_session_updated_preserves_harness_origin(tmp_path) -> None:
    """The external transcript observer always emits session.updated with
    origin=external. When a session was originally created via the
    harness-spawn path (origin=harness) — and after the ses_<hex>
    canonicalization, that record shares the canonical id with whatever
    the observer scans — the observer's event must NOT downgrade the
    record's origin to external. Otherwise the bridge sees the next MM
    user post in that channel as targeting an "external" session, fires
    _replace_external_session, and the user's harness-spawned session
    gets adopted/replaced on every bridge restart cycle."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    harness_session = repository.create_session(
        CreateSessionRequest(
            backend="claude-code",
            model="claude-opus-4-7",
            project=Project(path="/repo", name="repo"),
            bypass_permissions=True,
        )
    )
    assert harness_session.origin == "harness"
    assert harness_session.bypass_permissions is True

    # Observer-style event for the *same* canonical id, with the always-
    # external payload it emits.
    observer_payload = harness_session.model_copy(
        update={
            "origin": "external",
            "bypass_permissions": False,
            "model": "claude-opus-4-7",
        }
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=harness_session.id,
            data={"session": observer_payload.model_dump(mode="json")},
        )
    )

    after = repository.get_session(harness_session.id)
    assert after.origin == "harness", "observer must not downgrade harness origin"
    assert after.bypass_permissions is True, "observer must not clear bypass_permissions"
    repository.close()


def test_materialize_session_updated_creates_when_session_absent(tmp_path) -> None:
    """For a brand-new external claude session the observer is the only
    source — its session.updated event must create the record."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    external = Session(
        id="ses_3eb0e45b9d724deabdc3b472e0c4c2fc",
        backend="claude-code",
        model="claude-opus-4-7",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=external.id,
            data={"session": external.model_dump(mode="json")},
        )
    )

    after = repository.get_session(external.id)
    assert after.origin == "external"
    assert after.model == "claude-opus-4-7"
    repository.close()


def test_materialize_harness_to_harness_preserves_codex_resume_id_on_none_incoming() -> None:
    """Halcyon's NEEDS-FIX on PR #18: a status-flip event whose
    payload was built from a stale snapshot (read BEFORE
    ``_maybe_publish_codex_resume_id`` wrote the field) carries
    ``codex_resume_id=None``. The harness→harness materializer
    branch must NOT clobber an already-set codex_resume_id when the
    incoming payload's field is None — mirrors the existing
    ``stats``-preservation pattern.

    Concrete race: ``_maybe_publish_status_flip`` (running↔idle
    transition triggered by ``freshness_tick``) emits a
    ``session.updated`` after the resume-id event landed; without
    this guard the status-flip wipes the just-written
    codex_resume_id back to None."""
    from agent_harness.repository import InMemoryRepository

    repo = InMemoryRepository()
    session_id = "ses_019e0e0000000000000000000000000a"
    # Existing harness session has codex_resume_id set (observer's
    # binding emission already landed).
    existing = Session(
        id=session_id,
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="harness",
        codex_resume_id="019e0e00-0000-0000-0000-000000000000",
    )
    repo.upsert_session(existing)

    # Incoming harness session.updated event (e.g. a status flip)
    # was built from a snapshot taken BEFORE the resume-id event
    # landed — so it carries codex_resume_id=None.
    incoming = existing.model_copy(
        update={"status": "idle", "codex_resume_id": None}
    )
    repo.materialize_event(
        Event(
            event="session.updated",
            session_id=session_id,
            data={"session": incoming.model_dump(mode="json")},
        )
    )

    after = repo.get_session(session_id)
    # The status flip MUST land (the incoming event's purpose).
    assert after.status == "idle"
    # But the just-set codex_resume_id MUST survive.
    assert after.codex_resume_id == "019e0e00-0000-0000-0000-000000000000"


def test_materialize_harness_to_harness_preserves_codex_resume_id_sqlite(tmp_path) -> None:
    """Parallel SQLite coverage: both materializer paths must
    implement identical preservation semantics — otherwise a
    DurableEventBus deployment would have the leak the in-memory
    test guards against."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    try:
        session_id = "ses_019e0e0100000000000000000000000b"
        existing = Session(
            id=session_id,
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
            status="running",
            origin="harness",
            codex_resume_id="019e0e01-0000-0000-0000-000000000000",
        )
        repository.upsert_session(existing)

        incoming = existing.model_copy(
            update={"status": "idle", "codex_resume_id": None}
        )
        repository.materialize_event(
            Event(
                event="session.updated",
                session_id=session_id,
                data={"session": incoming.model_dump(mode="json")},
            ),
            store_event=False,
        )

        after = repository.get_session(session_id)
        assert after.status == "idle"
        assert after.codex_resume_id == "019e0e01-0000-0000-0000-000000000000"
    finally:
        repository.close()


def test_materialize_session_updated_allows_observer_updates_to_external(tmp_path) -> None:
    """For an existing origin=external session, the observer is the source
    of truth — later session.updated events should still flow through
    (e.g. if model changes mid-session)."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)

    initial = Session(
        id="ses_3eb0e45b9d724deabdc3b472e0c4c2fc",
        backend="claude-code",
        model="claude-opus-4-6",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=initial.id,
            data={"session": initial.model_dump(mode="json")},
        )
    )
    updated = initial.model_copy(update={"model": "claude-opus-4-7"})
    repository.materialize_event(
        Event(
            sequence=2,
            event="session.updated",
            session_id=initial.id,
            data={"session": updated.model_dump(mode="json")},
        )
    )

    after = repository.get_session(initial.id)
    assert after.model == "claude-opus-4-7"
    assert after.origin == "external"
    repository.close()


def _seed_session_with_runs(repo, session_id_suffix: str = "session"):
    """Create a session + two runs (one running, one queued) for race tests."""
    session = repo.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
        )
    )
    run_a = repo.create_run(session.id, CreateRunRequest(message="first"))
    run_b = repo.create_run(session.id, CreateRunRequest(message="second"))
    repo.start_run(session.id, run_a.id)
    repo.start_run(session.id, run_b.id)
    return session, run_a, run_b


def test_finish_run_does_not_set_session_idle_when_another_run_is_active_inmemory() -> None:
    """Regression: ``finish_run`` used to unconditionally set
    ``session.status = "idle"`` even when a queued/running successor
    existed. That contradicted the actual repo state — the second run
    was running, but the session looked idle — and surfaced via the
    ``GET /v1/sessions/{id}`` endpoint.

    Behavior: when another non-terminal run exists for the session,
    finish_run leaves session.status alone.
    """
    repo = InMemoryRepository()
    session, run_a, run_b = _seed_session_with_runs(repo)

    repo.finish_run(session.id, run_a.id, status="completed", stop_reason="end_turn")

    assert repo.get_run(session.id, run_a.id).status == "completed"
    assert repo.get_run(session.id, run_b.id).status == "running"
    # Session should still appear active because run_b is running.
    assert repo.get_session(session.id).status != "idle", (
        "session flipped to idle while run_b is still running"
    )

    # Now finish the last run — session should finally go idle.
    repo.finish_run(session.id, run_b.id, status="completed", stop_reason="end_turn")
    assert repo.get_session(session.id).status == "idle"


def test_finish_run_does_not_set_session_idle_when_another_run_is_active_sqlite(tmp_path) -> None:
    """SQLite counterpart of the previous test — same invariant must
    hold across persistent storage."""
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    try:
        session, run_a, run_b = _seed_session_with_runs(repo)

        repo.finish_run(session.id, run_a.id, status="completed", stop_reason="end_turn")

        assert repo.get_run(session.id, run_a.id).status == "completed"
        assert repo.get_run(session.id, run_b.id).status == "running"
        assert repo.get_session(session.id).status != "idle", (
            "session flipped to idle while run_b is still running"
        )

        repo.finish_run(session.id, run_b.id, status="completed", stop_reason="end_turn")
        assert repo.get_session(session.id).status == "idle"
    finally:
        repo.close()


def test_interrupt_run_is_noop_on_terminal_inmemory() -> None:
    """Once a run reaches a terminal state (completed/failed/interrupted),
    a later ``interrupt_run`` must not rewrite its status, stop_reason,
    or completed_at — first terminal wins.

    Regression: a late DELETE /runs/<id> on an already-completed run was
    flipping ``status=completed stop_reason=end_turn`` to
    ``status=interrupted stop_reason=interrupted``, corrupting the
    historical record (see investigation 2026-05-15)."""
    repo = InMemoryRepository()
    session, run_a, _run_b = _seed_session_with_runs(repo)

    completed = repo.finish_run(
        session.id, run_a.id, status="completed", stop_reason="end_turn",
    )

    returned = repo.interrupt_run(session.id, run_a.id)

    # Unchanged on disk + in the return value.
    assert returned.status == "completed"
    assert returned.stop_reason == "end_turn"
    assert returned.completed_at == completed.completed_at
    refreshed = repo.get_run(session.id, run_a.id)
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "end_turn"


def test_interrupt_run_is_noop_on_terminal_sqlite(tmp_path) -> None:
    """SQLite counterpart — same first-terminal-wins invariant."""
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    try:
        session, run_a, _run_b = _seed_session_with_runs(repo)

        completed = repo.finish_run(
            session.id, run_a.id, status="completed", stop_reason="end_turn",
        )

        returned = repo.interrupt_run(session.id, run_a.id)

        assert returned.status == "completed"
        assert returned.stop_reason == "end_turn"
        assert returned.completed_at == completed.completed_at
        refreshed = repo.get_run(session.id, run_a.id)
        assert refreshed.status == "completed"
        assert refreshed.stop_reason == "end_turn"
    finally:
        repo.close()


def test_materialize_run_interrupted_event_skips_terminal_inmemory() -> None:
    """When the API endpoint publishes ``run.interrupted`` for a target
    that the repo already marked terminal, the materializer must NOT
    rewrite its status. Same first-terminal-wins invariant via the
    materialization path."""
    from agent_harness.models import utc_now

    repo = InMemoryRepository()
    session, run_a, _run_b = _seed_session_with_runs(repo)
    repo.finish_run(session.id, run_a.id, status="completed", stop_reason="end_turn")

    repo.materialize_event(
        Event(
            event="run.interrupted",
            session_id=session.id,
            run_id=run_a.id,
            sequence=999,
            created_at=utc_now(),
        ),
        store_event=False,
    )

    refreshed = repo.get_run(session.id, run_a.id)
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "end_turn"


def test_materialize_run_interrupted_event_skips_terminal_sqlite(tmp_path) -> None:
    """SQLite counterpart."""
    from agent_harness.models import utc_now

    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    try:
        session, run_a, _run_b = _seed_session_with_runs(repo)
        repo.finish_run(session.id, run_a.id, status="completed", stop_reason="end_turn")

        repo.materialize_event(
            Event(
                event="run.interrupted",
                session_id=session.id,
                run_id=run_a.id,
                sequence=999,
                created_at=utc_now(),
            ),
            store_event=False,
        )

        refreshed = repo.get_run(session.id, run_a.id)
        assert refreshed.status == "completed"
        assert refreshed.stop_reason == "end_turn"
    finally:
        repo.close()


def test_open_sqlite_repository_initializes_schema(tmp_path) -> None:
    db_path = tmp_path / "harness.db"

    repository = open_sqlite_repository(db_path)
    repository.close()

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table' and name not like 'sqlite_%'"
            )
        }

    assert {"schema_migrations", "sessions", "runs", "messages", "events"} <= tables


def test_open_sqlite_repository_records_current_schema_version(tmp_path) -> None:
    db_path = tmp_path / "harness.db"

    repository = open_sqlite_repository(db_path)
    repository.close()

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "select version from schema_migrations order by version"
        ).fetchall()

    assert rows == [(SCHEMA_VERSION,)]


def test_open_sqlite_repository_bumps_legacy_schema_version(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "create table schema_migrations(version integer primary key, applied_at text not null)"
        )
        connection.execute(
            "insert into schema_migrations(version, applied_at) values (?, ?)",
            (SCHEMA_VERSION - 1, "2000-01-01T00:00:00+00:00"),
        )

    repository = open_sqlite_repository(db_path)
    repository.close()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("select max(version) from schema_migrations").fetchone()

    assert row == (SCHEMA_VERSION,)


def test_open_sqlite_repository_replaces_schema_migration_metadata(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    stale_applied_at = "2000-01-01T00:00:00+00:00"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "create table schema_migrations(version integer primary key, applied_at text not null)"
        )
        connection.execute(
            "insert into schema_migrations(version, applied_at) values (?, ?)",
            (SCHEMA_VERSION, stale_applied_at),
        )

    repository = open_sqlite_repository(db_path)
    repository.close()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "select applied_at from schema_migrations where version = ?",
            (SCHEMA_VERSION,),
        ).fetchone()

    assert row is not None
    assert row[0] != stale_applied_at


def test_startup_reconciles_dangling_running_runs_and_sessions(tmp_path) -> None:
    """On a harness crash, runs and sessions persist in SQLite with status
    ``running``/``queued`` while the in-memory RunManager state evaporates.
    Reopening the repository must flip those rows to terminal/idle so the
    next process can start fresh runs without the bridge perceiving them
    as still alive."""
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    session, run_a, run_b = _seed_session_with_runs(repo)
    # run_a is running (via start_run), run_b is queued in the same call
    # path; tweak run_b back to queued explicitly so we cover both states.
    queued = repo.get_run(session.id, run_b.id).model_copy(
        update={"status": "queued", "started_at": None},
    )
    repo._upsert_run(queued)
    # Sanity preconditions.
    assert repo.get_session(session.id).status == "running"
    assert repo.get_run(session.id, run_a.id).status == "running"
    assert repo.get_run(session.id, run_b.id).status == "queued"
    repo.close()

    reopened = open_sqlite_repository(db_path)
    try:
        assert reopened.get_session(session.id).status == "idle"
        assert reopened.get_run(session.id, run_a.id).status == "failed"
        assert reopened.get_run(session.id, run_a.id).completed_at is not None
        assert reopened.get_run(session.id, run_b.id).status == "failed"
    finally:
        reopened.close()


def test_startup_reconcile_preserves_external_running_sessions(tmp_path) -> None:
    """External sessions (transcript-observer-owned) are managed by the
    actual CLI process, not the harness's RunManager. A live external
    session can legitimately stay ``running`` across harness restarts;
    the startup sweep must leave it alone."""
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    external = Session(
        id="ses_external_live_one",
        backend="claude-code",
        model="claude-opus-4-7",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
    )
    repo.upsert_session(external)
    repo.close()

    reopened = open_sqlite_repository(db_path)
    try:
        after = reopened.get_session(external.id)
        assert after.status == "running", "external session was wrongly idled"
        assert after.origin == "external"
    finally:
        reopened.close()


def test_startup_reconcile_leaves_terminal_runs_untouched(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)
    session = repo.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
        )
    )
    run = repo.create_run(session.id, CreateRunRequest(message="done"))
    repo.finish_run(session.id, run.id, status="completed", stop_reason="end_turn")
    repo.close()

    reopened = open_sqlite_repository(db_path)
    try:
        finished = reopened.get_run(session.id, run.id)
        assert finished.status == "completed"
        assert finished.stop_reason == "end_turn"
        # Session went idle naturally when its only run completed.
        assert reopened.get_session(session.id).status == "idle"
    finally:
        reopened.close()


def test_observer_offsets_persist_across_reopen(tmp_path) -> None:
    """Transcript tail offsets must survive a harness restart — otherwise the
    observer re-reads every transcript from the start on each boot and
    re-emits every historical event into the durable event bus (the
    2026-05-15 message-flood). Upsert is keyed by path."""
    db_path = tmp_path / "harness.db"
    repo = open_sqlite_repository(db_path)

    assert repo.get_observer_offsets() == {}

    repo.set_observer_offset("/transcripts/a.jsonl", 256)
    repo.set_observer_offset("/transcripts/b.jsonl", 128)
    # Upsert: later write to the same key replaces the earlier value.
    repo.set_observer_offset("/transcripts/a.jsonl", 512)
    repo.close()

    reopened = open_sqlite_repository(db_path)
    try:
        assert reopened.get_observer_offsets() == {
            "/transcripts/a.jsonl": 512,
            "/transcripts/b.jsonl": 128,
        }
    finally:
        reopened.close()


def test_observer_offsets_table_creation_is_idempotent_on_legacy_db(tmp_path) -> None:
    """The migration must not destroy data on a DB that predates this
    change. Simulate an old DB (sessions + events but no
    ``observer_offsets`` table) and verify a fresh open backfills the
    table without touching existing rows."""
    db_path = tmp_path / "harness.db"
    legacy = sqlite3.connect(db_path)
    try:
        legacy.executescript(
            """
            create table sessions (
                id text primary key,
                payload text not null,
                created_at text not null,
                updated_at text not null
            );
            create table events (
                row_id integer primary key autoincrement,
                sequence integer,
                event text not null,
                session_id text,
                run_id text,
                payload text not null,
                created_at text not null
            );
            insert into sessions(id, payload, created_at, updated_at)
                values ('keep-me', '{}', '2026-05-15T00:00:00Z', '2026-05-15T00:00:00Z');
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    repo = open_sqlite_repository(db_path)
    try:
        assert repo.get_observer_offsets() == {}
        repo.set_observer_offset("/transcripts/a.jsonl", 42)
        survivor = repo._connection.execute("select id from sessions").fetchone()
        assert survivor["id"] == "keep-me"
        assert repo.get_observer_offsets() == {"/transcripts/a.jsonl": 42}
    finally:
        repo.close()


def test_sqlite_repository_append_event_no_longer_materializes_message(tmp_path) -> None:
    """Phase 2: ``append_event`` is a pure event-row insert. The
    message-materialization carve-out (previously gated on
    ``data.origin != "external"``) is removed; the observer's
    ``materialize_event`` path is now the sole writer to the messages
    table. The event row itself is still inserted, but the messages
    table stays empty until materialize_event runs."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        message = Message(role="assistant", blocks=[{"type": "text", "text": "should not materialize"}])

        published = repository.append_event(
            Event(
                event="message",
                session_id=session.id,
                data={"message": message.model_dump(mode="json")},
            )
        )

        # Event row was inserted (the durable bus relies on this).
        assert published.sequence is not None
        events = repository.list_events(session_id=session.id)
        assert any(e.event == "message" for e in events)

        # But the message did NOT land in the messages table — that's
        # the observer's job now via ``materialize_event``. The session
        # also kept its baseline (one user message from create_run-less
        # path; here zero because we didn't call create_run).
        messages = repository.list_messages(session.id)
        assert messages == []
        # The default stats.messages from create_session is 0.
        assert repository.get_session(session.id).stats.messages == 0
    finally:
        repository.close()


# --- Phase 3: append_event is a pure insert; all side effects move to bus ----


def test_sqlite_repository_append_event_does_not_materialize_run_lifecycle(tmp_path) -> None:
    """Phase 3: ``append_event`` is a pure event-row insert. Even run
    lifecycle events (which Phase 2 still materialized) move out — the
    ``DurableEventBus.publish`` path orchestrates both calls so there's
    a single materialization point per event."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        assert run.status == "queued"

        # Append a run.started event WITHOUT going through the bus.
        published = repository.append_event(
            Event(event="run.started", session_id=session.id, run_id=run.id, data={})
        )

        # Event row inserted; run status did NOT change.
        assert published.sequence is not None
        assert repository.get_run(session.id, run.id).status == "queued"
    finally:
        repository.close()


def test_sqlite_repository_append_event_does_not_materialize_run_usage(tmp_path) -> None:
    """Same contract for ``run.usage``: pure insert, no side effects."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="claude-code",
                model="claude-opus-4-7",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))

        repository.append_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={"usage": {"input": 6, "output": 4, "cache_read": 18, "cache_creation": 21}},
            )
        )

        # Run.usage stays at its default zero — no materialization
        # happened via append_event alone.
        after = repository.get_run(session.id, run.id)
        assert (
            after.usage.input,
            after.usage.output,
            after.usage.cache_read,
            after.usage.cache_creation,
        ) == (0, 0, 0, 0)
    finally:
        repository.close()


def test_materialize_event_preserves_existing_session_stats_on_external_upsert(tmp_path) -> None:
    """Regression for Phase 3 review's must-fix #1.

    The observer's parser builds a freshly-constructed ``Session``
    payload for ``session.updated`` events whose ``stats`` field
    defaults to a zero-valued ``SessionStats()``. Phase 3's
    ``run.usage`` materializer rolls token deltas into
    ``Session.stats``; if a subsequent ``session.updated`` upsert
    overwrites the existing row wholesale (including its accumulated
    stats), every assistant / turn_context line in a multi-turn
    external-resume session would wipe prior turns' accumulated
    tokens. PR #11 originally guarded against this; the Phase 3
    cherry-pick of the parsers dropped the guard.

    Verify: an external session with non-zero stats receives a fresh
    session.updated and KEEPS its accumulated stats.
    """
    from agent_harness.models import Event, Project, Session, SessionStats

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        # Seed an external session with non-trivial stats (as if a
        # prior ``run.usage`` had already aggregated tokens).
        seeded = Session(
            id="codex_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
            origin="external",
            stats=SessionStats(
                messages=3,
                tokens={"input": 120, "output": 300, "cache_read": 50, "cache_creation": 0},
                cost_usd=0.0,
                context_window=258400,
            ),
        )
        repository.upsert_session(seeded)

        # Now materialize a fresh session.updated event (as the
        # observer would emit on a later turn_context line). The
        # payload's stats is zero-valued by construction.
        fresh = Session(
            id=seeded.id,
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/repo", name="repo"),
            origin="external",
        )
        assert fresh.stats == SessionStats()  # zeroed by default

        repository.materialize_event(
            Event(
                event="session.updated",
                session_id=seeded.id,
                data={"session": fresh.model_dump(mode="json")},
            ),
            store_event=False,
        )

        after = repository.get_session(seeded.id)
        assert after.stats.messages == 3
        assert after.stats.tokens == {
            "input": 120,
            "output": 300,
            "cache_read": 50,
            "cache_creation": 0,
        }
        assert after.stats.context_window == 258400
    finally:
        repository.close()


def test_materialize_run_usage_sets_session_context_window(tmp_path) -> None:
    """End-to-end coverage for the Phase 3 spec's open-question (b)
    resolution: codex's ``context_window`` rides on the ``run.usage``
    event and the materializer applies it to ``Session.stats.context_window``
    in the same pass that updates ``Run.usage`` and ``Session.stats.tokens``.

    Falcon's worth-noting + Aegis-style coverage gap: parser and
    observer emission were tested in isolation; this nails the
    materializer's session-stats write."""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        repository.start_run(session.id, run.id)

        repository.materialize_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={
                    "usage": {
                        "input": 120,
                        "output": 300,
                        "cache_read": 50,
                        "cache_creation": 0,
                    },
                    "context_window": 258400,
                },
            ),
            store_event=False,
        )

        after = repository.get_session(session.id)
        assert after.stats.context_window == 258400
        assert after.stats.tokens.get("input") == 120
        assert after.stats.tokens.get("output") == 300
    finally:
        repository.close()


def test_materialize_run_usage_sets_session_context_used(tmp_path) -> None:
    """``context_used`` rides on the ``run.usage`` event alongside
    ``usage`` and ``context_window`` (option A from the
    Session.stats.context_used spec). Materializer overwrites
    ``Session.stats.context_used`` in the same pass that adds usage
    onto ``Session.stats.tokens`` and overwrites ``context_window``.
    Spec: specs/2026-05-19-context-used.md"""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        repository.start_run(session.id, run.id)

        repository.materialize_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={
                    "usage": {"input": 50, "output": 20, "cache_read": 0, "cache_creation": 0},
                    "context_window": 258400,
                    "context_used": 17000,
                },
            ),
            store_event=False,
        )
        assert repository.get_session(session.id).stats.context_used == 17000
    finally:
        repository.close()


def test_materialize_run_usage_overwrites_context_used_does_not_sum(tmp_path) -> None:
    """SNAPSHOT semantics: a later observation REPLACES the earlier
    value (never sums). Two ``run.usage`` events with different
    context_used must leave the session at the latter — but tokens
    (additive) keeps accumulating, proving the two writes are
    independent."""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="claude-code",
                model="claude-4-7-sonnet",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        repository.start_run(session.id, run.id)

        for usage, snapshot in [
            ({"input": 100, "output": 50, "cache_read": 10000, "cache_creation": 0}, 10500),
            ({"input": 80, "output": 40, "cache_read": 30000, "cache_creation": 0}, 30200),
        ]:
            repository.materialize_event(
                Event(
                    event="run.usage",
                    session_id=session.id,
                    run_id=run.id,
                    data={"usage": usage, "context_used": snapshot},
                ),
                store_event=False,
            )

        after = repository.get_session(session.id)
        # Latest snapshot wins (overwrite-not-sum).
        assert after.stats.context_used == 30200
        # Tokens keep summing per existing contract.
        assert after.stats.tokens["input"] == 100 + 80
        assert after.stats.tokens["cache_read"] == 10000 + 30000


    finally:
        repository.close()


def test_materialize_run_usage_preserves_context_used_when_event_omits_it(tmp_path) -> None:
    """A ``run.usage`` event WITHOUT a ``context_used`` key must not
    clear the existing snapshot. Earlier observation stays put; a
    snapshot only changes when the rollout produces a fresh value."""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="codex",
                model="gpt-5.4",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        repository.start_run(session.id, run.id)

        # First event seeds context_used.
        repository.materialize_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={
                    "usage": {"input": 10, "output": 5, "cache_read": 0, "cache_creation": 0},
                    "context_used": 4096,
                },
            ),
            store_event=False,
        )
        assert repository.get_session(session.id).stats.context_used == 4096

        # Second event omits context_used (e.g. codex token_count
        # event with info: null mid-session — only the per-turn delta
        # is meaningful). The snapshot must survive.
        repository.materialize_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={
                    "usage": {"input": 3, "output": 2, "cache_read": 0, "cache_creation": 0},
                },
            ),
            store_event=False,
        )
        assert repository.get_session(session.id).stats.context_used == 4096
    finally:
        repository.close()


def test_materialize_run_usage_overwrites_context_used_inmemory() -> None:
    """Parallel coverage for ``InMemoryRepository`` — both materializers
    (SQLite + in-memory) implement the same ``run.usage`` branch, so the
    overwrite-not-sum invariant must hold on both. Without this, a code
    path that exercises only the in-memory repo (the FastAPI factory's
    default) could silently regress."""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repo = InMemoryRepository()
    session = repo.create_session(
        CreateSessionRequest(
            backend="claude-code",
            model="claude-4-7-sonnet",
            project=Project(path="/repo", name="repo"),
        )
    )
    run = repo.create_run(session.id, CreateRunRequest(message="hi"))
    repo.start_run(session.id, run.id)

    for snapshot in [4096, 9000]:
        repo.materialize_event(
            Event(
                event="run.usage",
                session_id=session.id,
                run_id=run.id,
                data={
                    "usage": {"input": 1, "output": 1, "cache_read": 0, "cache_creation": 0},
                    "context_used": snapshot,
                },
            ),
            store_event=False,
        )
    assert repo.get_session(session.id).stats.context_used == 9000


def test_materialize_run_usage_accepts_context_used_decrease(tmp_path) -> None:
    """After context compaction the snapshot legitimately decreases —
    the materializer must apply a smaller value just as readily as a
    larger one."""
    from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project

    repository = open_sqlite_repository(tmp_path / "harness.db")
    try:
        session = repository.create_session(
            CreateSessionRequest(
                backend="claude-code",
                model="claude-4-7-sonnet",
                project=Project(path="/repo", name="repo"),
            )
        )
        run = repository.create_run(session.id, CreateRunRequest(message="hi"))
        repository.start_run(session.id, run.id)

        for snapshot in [180000, 12000]:  # pre-compact, post-compact
            repository.materialize_event(
                Event(
                    event="run.usage",
                    session_id=session.id,
                    run_id=run.id,
                    data={
                        "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
                        "context_used": snapshot,
                    },
                ),
                store_event=False,
            )
        assert repository.get_session(session.id).stats.context_used == 12000
    finally:
        repository.close()


def _forkable_parent(repo, *, backend: str = "claude-code", title: str | None = "Parent") -> Session:
    return repo.create_session(
        CreateSessionRequest(
            backend=backend,
            model="claude-4-7-sonnet",
            project=Project(path="/repo", name="repo"),
            title=title,
            bypass_permissions=True,
            effort="high",
        )
    )


def test_inmemory_create_forked_session_inherits_fields_and_records_parent() -> None:
    repo = InMemoryRepository()
    parent = _forkable_parent(repo)

    child = repo.create_forked_session(parent, title=None)

    assert child.id != parent.id
    assert child.origin == "harness"
    assert child.forked_from == parent.id
    assert child.backend == parent.backend
    assert child.model == parent.model
    assert child.project == parent.project
    assert child.bypass_permissions is True
    assert child.effort == parent.effort == "high"
    # No title given → inherit the parent's.
    assert child.title == parent.title
    # Child is a real, retrievable session with its own (empty) message log.
    assert repo.get_session(child.id).id == child.id
    assert repo.list_messages(child.id) == []


def test_inmemory_create_forked_session_uses_explicit_title() -> None:
    repo = InMemoryRepository()
    parent = _forkable_parent(repo)

    child = repo.create_forked_session(parent, title="Thread reply")

    assert child.title == "Thread reply"


def test_sqlite_create_forked_session_persists_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    parent = _forkable_parent(repository)
    child = repository.create_forked_session(parent, title=None)
    repository.close()

    reopened = open_sqlite_repository(db_path)
    stored = reopened.get_session(child.id)
    assert stored.forked_from == parent.id
    assert stored.origin == "harness"
    assert stored.model == parent.model
    assert stored.effort == "high"
    reopened.close()


def test_sqlite_create_session_persists_effort_across_reopen(tmp_path) -> None:
    """``effort`` lands in the session JSON blob — no schema/migration work.
    Rows written before the field existed deserialize on the pydantic default
    (None), which is why SCHEMA_VERSION is untouched."""
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    with_effort = repository.create_session(
        CreateSessionRequest(
            backend="pi",
            project=Project(path="/repo", name="repo"),
            effort="medium",
        )
    )
    without = repository.create_session(
        CreateSessionRequest(backend="pi", project=Project(path="/repo", name="repo"))
    )
    repository.close()

    reopened = open_sqlite_repository(db_path)
    assert reopened.get_session(with_effort.id).effort == "medium"
    assert reopened.get_session(without.id).effort is None
    reopened.close()
