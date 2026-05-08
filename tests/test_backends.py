from agent_harness.backends import default_backend_registry


def test_default_backend_registry_lists_claude_code_and_codex() -> None:
    registry = default_backend_registry()

    backends = registry.list()
    backend_names = {backend.name for backend in backends}

    assert backend_names == {"claude-code", "codex"}
    assert all(backend.capabilities.stream_json for backend in backends)
    assert registry.get("codex").display_name == "Codex"
