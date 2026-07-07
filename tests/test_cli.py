from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.cli import main


def test_serve_command_runs_uvicorn_with_import_string_when_no_observer_roots_exist(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)

    assert main(["serve", "--host", "0.0.0.0", "--port", "8765", "--reload"]) == 0

    assert calls == [
        (
            ("agent_harness.api:app",),
            {"host": "0.0.0.0", "port": 8765, "reload": True, "log_level": "info"},
        )
    ]


def test_serve_command_builds_app_for_database(monkeypatch, tmp_path) -> None:
    calls = []
    opened = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    def fake_open(path):
        opened.append(path)
        return object()

    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)
    monkeypatch.setattr("agent_harness.cli.open_sqlite_repository", fake_open)

    database = tmp_path / "agent-harness.db"
    assert main(["serve", "--database", str(database)]) == 0

    assert opened == [str(database)]
    assert calls[0][0][0] != "agent_harness.api:app"


def test_serve_command_uses_durable_bus_for_database_and_execute_runs(monkeypatch, tmp_path) -> None:
    calls = []
    repository = object()
    app = object()

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return app

    monkeypatch.setattr("agent_harness.cli.create_app", fake_create_app)
    monkeypatch.setattr("agent_harness.cli.open_sqlite_repository", lambda _path: repository)
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", lambda *_args, **_kwargs: None)

    assert main(["serve", "--database", str(tmp_path / "harness.db"), "--execute-runs"]) == 0

    event_bus = calls[0]["event_bus"]
    assert calls[0]["repository"] is repository
    assert isinstance(event_bus, DurableEventBus)
    assert calls[0]["run_manager"]._event_bus is event_bus


def test_serve_command_keeps_in_memory_bus_for_execute_runs_without_database(monkeypatch) -> None:
    calls = []
    app = object()

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return app

    monkeypatch.setattr("agent_harness.cli.create_app", fake_create_app)
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", lambda *_args, **_kwargs: None)

    assert main(["serve", "--execute-runs"]) == 0

    assert isinstance(calls[0]["event_bus"], InMemoryEventBus)


def test_serve_command_builds_app_for_real_run_execution(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)

    assert main(["serve", "--execute-runs"]) == 0

    assert calls[0][0][0] != "agent_harness.api:app"


def test_serve_command_configures_observer_roots(monkeypatch, tmp_path) -> None:
    calls = []
    root = tmp_path / "transcripts"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    app = object()

    def fake_create_app(**kwargs):
        calls.append(("create_app", kwargs))
        return app

    def fake_run(*args, **kwargs):
        calls.append(("run", args, kwargs))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("agent_harness.cli.create_app", fake_create_app)
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)

    assert main(["serve", "--observe-root", str(root)]) == 0

    assert calls == [
        ("create_app", {"observer_settings": calls[0][1]["observer_settings"]}),
        ("run", (app,), {"host": "127.0.0.1", "port": 8000, "reload": False, "log_level": "info"}),
    ]
    assert calls[0][1]["observer_settings"].roots == (root,)


def test_serve_command_auto_observes_existing_default_transcript_roots(monkeypatch, tmp_path) -> None:
    calls = []
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    (tmp_path / ".codex" / "sessions").mkdir(parents=True)
    (tmp_path / ".pi" / "agent" / "sessions").mkdir(parents=True)

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_harness.cli.create_app", fake_create_app)
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", lambda *_args, **_kwargs: None)

    assert main(["serve", "--observe-default-roots"]) == 0

    assert calls[0]["observer_settings"].roots == (
        tmp_path / ".claude" / "projects",
        tmp_path / ".codex" / "sessions",
        tmp_path / ".pi" / "agent" / "sessions",
    )


def test_serve_command_can_disable_auto_observer(monkeypatch, tmp_path) -> None:
    calls = []
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    (tmp_path / ".codex" / "sessions").mkdir(parents=True)

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)

    assert main(["serve", "--no-observer"]) == 0

    assert calls[0][0] == ("agent_harness.api:app",)


def test_serve_command_wires_cors_origins(monkeypatch, tmp_path) -> None:
    calls = []
    app = object()

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_harness.cli.create_app", fake_create_app)
    monkeypatch.setattr("agent_harness.cli.uvicorn.run", lambda *_args, **_kwargs: None)

    assert (
        main(
            [
                "serve",
                "--cors-origin",
                "https://a.example",
                "--cors-origin",
                "https://b.example",
            ]
        )
        == 0
    )

    assert calls[0]["cors_origins"] == ["https://a.example", "https://b.example"]
