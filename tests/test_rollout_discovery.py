"""Unit tests for ``agent_harness.rollout_discovery``.

Covers the deterministic claude path computation and the codex
psutil-based fd probe with lsof fallback. Real subprocesses are never
spawned — ``psutil.Process`` and ``subprocess.run`` are stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.rollout_discovery import (
    RolloutDiscovery,
    RolloutDiscoveryError,
    _claude_slug,
)


class _FakeFd:
    """Minimal stand-in for ``psutil._common.popenfile`` — only ``.path``
    is read by the discovery module."""

    def __init__(self, path: str) -> None:
        self.path = path


class _FakeProcess:
    """Stub for ``psutil.Process`` used by the codex fd probe.

    The list of open files returned by each ``open_files()`` call cycles
    through ``open_files_sequence``; the discovery module polls until a
    matching path appears, so the sequence simulates the codex subprocess
    opening its rollout fd on the Nth poll. ``error`` can be set to a
    callable that, if non-None, is raised by ``open_files()``.
    """

    def __init__(
        self,
        *,
        open_files_sequence: list[list[_FakeFd]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._sequence = open_files_sequence or [[]]
        self._index = 0
        self._error = error
        self.calls = 0

    def open_files(self) -> list[_FakeFd]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        snapshot = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return snapshot


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


@pytest.mark.asyncio
async def test_discover_codex_finds_open_fd(tmp_path) -> None:
    sessions_root = tmp_path / ".codex" / "sessions"
    rollout = sessions_root / "2026" / "05" / "19" / "rollout-2026-05-19T10-30-00-019e0000-0000-0000-0000-000000000000.jsonl"

    open_files = [[_FakeFd(str(rollout))]]
    fake_process = _FakeProcess(open_files_sequence=open_files)
    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    path = await discovery.discover_codex(
        pid=12345, cwd=Path("/repo"), timeout_seconds=1.0, poll_interval=0.01
    )
    assert path == rollout
    assert fake_process.calls == 1


@pytest.mark.asyncio
async def test_discover_codex_polls_until_fd_appears(tmp_path) -> None:
    """psutil reports no rollout fd for the first two polls; the third
    poll reveals it. discover_codex must keep polling and return the path."""
    sessions_root = tmp_path / ".codex" / "sessions"
    rollout = sessions_root / "2026" / "05" / "19" / "rollout-2026-05-19T10-30-00-019e0001-0000-0000-0000-000000000000.jsonl"

    fake_process = _FakeProcess(
        open_files_sequence=[
            [],
            [_FakeFd("/tmp/some-other-file.log")],
            [_FakeFd(str(rollout))],
        ]
    )
    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    path = await discovery.discover_codex(
        pid=12345, cwd=Path("/repo"), timeout_seconds=1.0, poll_interval=0.01
    )
    assert path == rollout
    assert fake_process.calls == 3


@pytest.mark.asyncio
async def test_discover_codex_raises_on_timeout(tmp_path) -> None:
    sessions_root = tmp_path / ".codex" / "sessions"
    fake_process = _FakeProcess(open_files_sequence=[[]])
    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    with pytest.raises(RolloutDiscoveryError) as excinfo:
        await discovery.discover_codex(
            pid=12345, cwd=Path("/repo"), timeout_seconds=0.1, poll_interval=0.02
        )

    msg = str(excinfo.value)
    assert "12345" in msg
    assert "0.1" in msg or "timeout" in msg.lower()


@pytest.mark.asyncio
async def test_discover_codex_narrows_by_cwd_when_multiple_match(tmp_path) -> None:
    """Two rollout fds match the codex root filter; the tiebreaker reads
    session_meta.cwd from each and picks the one whose cwd matches."""
    sessions_root = tmp_path / ".codex" / "sessions"
    day_dir = sessions_root / "2026" / "05" / "19"
    day_dir.mkdir(parents=True)
    correct_cwd = "/repo/correct"
    wrong_cwd = "/repo/other"

    correct = day_dir / "rollout-2026-05-19T10-30-00-019e0002-0000-0000-0000-000000000000.jsonl"
    wrong = day_dir / "rollout-2026-05-19T10-30-00-019e0003-0000-0000-0000-000000000000.jsonl"

    correct.write_text(
        '{"timestamp":"2026-05-19T10:30:00Z","type":"session_meta",'
        '"payload":{"id":"019e0002-0000-0000-0000-000000000000",'
        '"timestamp":"2026-05-19T10:30:00Z","cwd":"' + correct_cwd + '"}}\n',
        encoding="utf-8",
    )
    wrong.write_text(
        '{"timestamp":"2026-05-19T10:30:00Z","type":"session_meta",'
        '"payload":{"id":"019e0003-0000-0000-0000-000000000000",'
        '"timestamp":"2026-05-19T10:30:00Z","cwd":"' + wrong_cwd + '"}}\n',
        encoding="utf-8",
    )

    fake_process = _FakeProcess(
        open_files_sequence=[[_FakeFd(str(correct)), _FakeFd(str(wrong))]]
    )
    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    path = await discovery.discover_codex(
        pid=12345, cwd=Path(correct_cwd), timeout_seconds=1.0, poll_interval=0.01
    )
    assert path == correct


@pytest.mark.asyncio
async def test_discover_codex_lsof_fallback(tmp_path, monkeypatch) -> None:
    """psutil.Process.open_files() raises NotImplementedError on some BSDs
    and exotic platforms. Fall back to ``lsof -p <pid> -F n``."""
    sessions_root = tmp_path / ".codex" / "sessions"
    rollout = sessions_root / "2026" / "05" / "19" / "rollout-2026-05-19T10-30-00-019e0004-0000-0000-0000-000000000000.jsonl"

    fake_process = _FakeProcess(error=NotImplementedError("not on this platform"))

    # lsof output format with -F n: blocks separated by file entry markers.
    # We only care about the ``n<path>`` lines.
    lsof_output = (
        "p12345\n"
        "n/dev/null\n"
        "n" + str(rollout) + "\n"
        "n/tmp/other.log\n"
    )

    class _FakeLsof:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, argv: list[str], *, capture_output: bool, text: bool, timeout: float | None = None):
            self.calls.append(argv)

            class _Result:
                returncode = 0
                stdout = lsof_output

            return _Result()

    fake_lsof = _FakeLsof()
    monkeypatch.setattr("agent_harness.rollout_discovery.subprocess.run", fake_lsof)

    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    path = await discovery.discover_codex(
        pid=12345, cwd=Path("/repo"), timeout_seconds=1.0, poll_interval=0.01
    )
    assert path == rollout
    assert fake_lsof.calls and fake_lsof.calls[0][0] == "lsof"


@pytest.mark.asyncio
async def test_rollout_discovery_error_has_meaningful_message(tmp_path) -> None:
    """Timeout errors should mention pid and timeout so operators can
    correlate a failure to a specific subprocess in logs."""
    sessions_root = tmp_path / ".codex" / "sessions"
    fake_process = _FakeProcess(open_files_sequence=[[]])
    discovery = RolloutDiscovery(
        codex_sessions_root=sessions_root,
        process_factory=lambda pid: fake_process,
    )

    with pytest.raises(RolloutDiscoveryError) as excinfo:
        await discovery.discover_codex(
            pid=99999,
            cwd=Path("/repo"),
            timeout_seconds=0.05,
            poll_interval=0.02,
        )
    assert "99999" in str(excinfo.value)
