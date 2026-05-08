from agent_harness.backends import default_backend_registry


def test_default_backend_registry_lists_claude_code_and_codex() -> None:
    registry = default_backend_registry()

    backends = registry.list()
    backend_ids = {backend.id for backend in backends}

    assert backend_ids == {"claude-code", "codex"}
    assert all(backend.capabilities.supports_sse for backend in backends)
    assert registry.get("codex").display_name == "Codex"
