from agent_harness.backends import default_backend_registry


def test_default_backend_registry_lists_claude_code_codex_and_pi() -> None:
    registry = default_backend_registry()

    backends = registry.list()
    backend_names = {backend.name for backend in backends}

    assert backend_names == {"claude-code", "codex", "pi"}
    assert all(backend.capabilities.stream_json for backend in backends)
    assert registry.get("codex").display_name == "Codex"


def test_pi_backend_registered_with_honest_capabilities() -> None:
    """pi's capability flags are honest current values (spec Design §3,
    confirmed by the spec-gate remarks): headless-only, no per-action
    permission prompt, session-id resume wired, no harness fork/pty/mcp
    plumbing yet."""
    registry = default_backend_registry()

    assert registry.has("pi")
    pi = registry.get("pi")
    assert pi.display_name == "pi"
    assert pi.available is True

    caps = pi.capabilities
    assert caps.fork is False
    assert caps.subagents is False
    assert caps.permission_detection is False
    assert caps.interactive_pty is False
    assert caps.stream_json is True
    assert caps.structured_output is True
    assert caps.session_id_choice is True
    assert caps.max_budget is False
    assert caps.mcp is False
    assert caps.sandbox is None
    assert caps.tools == "granular"
    assert caps.interrupt_external_runs is False
