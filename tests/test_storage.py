import sqlite3

from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Message, Project, Session
from agent_harness.storage import open_sqlite_repository


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
