from agent_harness.cli import main


def test_serve_command_runs_uvicorn_with_import_string(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("agent_harness.cli.uvicorn.run", fake_run)

    assert main(["serve", "--host", "0.0.0.0", "--port", "8765", "--reload"]) == 0

    assert calls == [
        (
            ("agent_harness.api:app",),
            {"host": "0.0.0.0", "port": 8765, "reload": True, "log_level": "info"},
        )
    ]
