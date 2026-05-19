"""Unit tests for ``agent_harness.rollout_discovery``.

Phase 2 scope: claude-only. The codex path moved to the observer's
expectation registry (see ``tests/test_observer.py``); the
psutil-based ``discover_codex`` and its lsof fallback have been
removed along with the psutil dependency.
"""

from __future__ import annotations

from pathlib import Path

from agent_harness.rollout_discovery import RolloutDiscovery, _claude_slug


def test_claude_slug_strips_leading_slash() -> None:
    assert _claude_slug(Path("/home/me/project")) == "-home-me-project"


def test_claude_slug_replaces_separators() -> None:
    assert _claude_slug(Path("/a/b/c")) == "-a-b-c"


def test_claude_slug_handles_no_leading_slash() -> None:
    # Relative paths shouldn't crash; claude-code itself anchors to abs
    # paths but the helper must be robust.
    assert _claude_slug(Path("a/b")) == "a-b"


def test_discover_claude_returns_deterministic_path(tmp_path) -> None:
    discovery = RolloutDiscovery(claude_projects_root=tmp_path / ".claude" / "projects")
    cwd = Path("/home/me/project")
    # Claude is invoked with ``--session-id <8-4-4-4-12 uuid>``; the
    # rollout file's stem is that exact dashed UUID. The discovery
    # helper must compose the same shape, otherwise the path-keyed
    # binding map would never match the file claude actually writes.
    claude_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    path = discovery.discover_claude(session_id=claude_uuid, cwd=cwd)
    assert path == (
        tmp_path / ".claude" / "projects" / "-home-me-project"
        / f"{claude_uuid}.jsonl"
    )


def test_discover_claude_matches_real_filename_shape(tmp_path) -> None:
    """The bound path must equal what claude actually writes to disk.

    Regression guard for the spec-described case: orchestrator passes
    ``--session-id <dashed-uuid>``, so the rollout file appears under
    that dashed-uuid stem (not the harness's ``ses_<32hex>`` form).
    """
    discovery = RolloutDiscovery(claude_projects_root=tmp_path / ".claude" / "projects")
    cwd = Path("/repo")
    claude_uuid = "12345678-1234-5678-1234-567812345678"
    expected = (
        tmp_path / ".claude" / "projects" / "-repo" / f"{claude_uuid}.jsonl"
    )
    # Simulate the real file claude would create after spawn.
    expected.parent.mkdir(parents=True)
    expected.write_text("", encoding="utf-8")

    path = discovery.discover_claude(session_id=claude_uuid, cwd=cwd)
    assert path == expected
    assert path.exists()


def test_discover_claude_handles_nested_cwd(tmp_path) -> None:
    discovery = RolloutDiscovery(claude_projects_root=tmp_path / ".claude" / "projects")
    cwd = Path("/home/me/projects/agent-harness-echo/worktrees/feat/x")
    claude_uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    path = discovery.discover_claude(session_id=claude_uuid, cwd=cwd)
    assert path.name == f"{claude_uuid}.jsonl"
    assert path.parent.name == "-home-me-projects-agent-harness-echo-worktrees-feat-x"


def test_discover_claude_returns_path_even_when_file_missing(tmp_path) -> None:
    discovery = RolloutDiscovery(claude_projects_root=tmp_path / ".claude" / "projects")
    path = discovery.discover_claude(
        session_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        cwd=Path("/repo"),
    )
    assert not path.exists()
