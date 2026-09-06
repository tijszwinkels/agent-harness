from __future__ import annotations

from collections.abc import Iterable

from agent_harness.models import Backend, BackendCapabilities, BackendName


class BackendRegistry:
    def __init__(self, backends: Iterable[Backend], models: dict[str, list[str]] | None = None) -> None:
        self._backends = {backend.name: backend for backend in backends}
        self._models = models or {}

    def list(self) -> list[Backend]:
        return list(self._backends.values())

    def get(self, name: BackendName) -> Backend:
        return self._backends[name]

    def has(self, name: str) -> bool:
        return name in self._backends

    def models(self, name: str) -> list[str]:
        if name not in self._backends:
            raise KeyError(name)
        return list(self._models.get(name, []))


def default_backend_registry() -> BackendRegistry:
    claude = BackendCapabilities(
        fork=True,
        subagents=True,
        permission_detection=True,
        interactive_pty=True,
        stream_json=True,
        structured_output=True,
        session_id_choice=True,
        max_budget=True,
        mcp=True,
        effort=True,
        sandbox=None,
        tools="granular",
        interrupt_external_runs=False,
    )
    codex = BackendCapabilities(
        fork=True,
        subagents=False,
        permission_detection=False,
        interactive_pty=True,
        stream_json=True,
        structured_output=True,
        session_id_choice=False,
        max_budget=False,
        mcp=True,
        effort=True,
        sandbox=["read-only", "workspace-write", "danger"],
        tools="granular",
        interrupt_external_runs=False,
    )
    # pi (github.com/parkerhutchinson/pi-cli, verified pi v0.80.3): headless
    # ``pi -p`` for harness runs, so no per-action permission prompt
    # (``permission_detection`` False). Resume is wired via ``--session-id``
    # (``session_id_choice`` True). ``fork`` and ``interactive_pty`` describe pi's
    # own capabilities on the same declarative footing as claude/codex — neither
    # flag is backed by harness code for *any* backend, yet both are True there.
    # Verified against ``pi --help`` (v0.80.3): ``--fork <path|id>`` exists (fork
    # True), and pi's default mode is interactive while ``-p/--print`` is
    # explicitly "Non-interactive mode" (interactive_pty True). ``mcp`` stays
    # False — a genuine pi limitation (no MCP flag in ``pi --help``), unlike
    # claude/codex. pi has no sandbox levels (unlike codex) and no budget flag;
    # its tool allow/deny lists make ``tools`` granular. ``effort`` is True on all
    # three backends — pi spells it ``--thinking <level>`` (verified 2026-09-06).
    pi = BackendCapabilities(
        fork=True,
        subagents=False,
        permission_detection=False,
        interactive_pty=True,
        stream_json=True,
        structured_output=True,
        session_id_choice=True,
        max_budget=False,
        mcp=False,
        effort=True,
        sandbox=None,
        tools="granular",
        interrupt_external_runs=False,
    )
    return BackendRegistry(
        [
            Backend(name="claude-code", display_name="Claude Code", capabilities=claude),
            Backend(name="codex", display_name="Codex", capabilities=codex),
            Backend(name="pi", display_name="pi", capabilities=pi),
        ]
    )
