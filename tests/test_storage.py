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


def test_sqlite_repository_skips_consecutive_duplicate_messages(tmp_path) -> None:
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
    repository.materialize_event(
        Event(sequence=1, event="message", session_id=session.id, data={"message": message.model_dump(mode="json")})
    )

    assert [item.blocks[0].text for item in repository.list_messages(session.id)] == ["same"]
    assert repository.get_session(session.id).stats.messages == 1
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
