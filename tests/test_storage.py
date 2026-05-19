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
