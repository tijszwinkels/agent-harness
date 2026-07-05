from agent_harness.backends import default_backend_registry


def test_default_backend_registry_lists_claude_code_codex_and_pi() -> None:
    registry = default_backend_registry()

    backends = registry.list()
    backend_names = {backend.name for backend in backends}

    assert backend_names == {"claude-code", "codex", "pi"}
    assert all(backend.capabilities.stream_json for backend in backends)
    assert registry.get("codex").display_name == "Codex"


def test_pi_backend_registered_with_honest_capabilities() -> None:
    """pi's capability flags describe pi's own capabilities, on the same
    "declarative metadata" footing as claude/codex (none of fork/interactive_pty
    are plumbed through the harness for any backend). Verified against pi v0.80.3
    ``pi --help``: ``--fork`` exists (fork True), the default mode is interactive
    while ``-p`` is "Non-interactive mode" (interactive_pty True), and there is no
    MCP flag (mcp False — a genuine pi limitation). Resume is wired via
    ``--session-id`` (session_id_choice True)."""
    registry = default_backend_registry()

    assert registry.has("pi")
    pi = registry.get("pi")
    assert pi.display_name == "pi"
    assert pi.available is True

    caps = pi.capabilities
    assert caps.fork is True
    assert caps.subagents is False
    assert caps.permission_detection is False
    assert caps.interactive_pty is True
    assert caps.stream_json is True
    assert caps.structured_output is True
    assert caps.session_id_choice is True
    assert caps.max_budget is False
    assert caps.mcp is False
    assert caps.sandbox is None
    assert caps.tools == "granular"
    assert caps.interrupt_external_runs is False
