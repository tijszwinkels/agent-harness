from __future__ import annotations

from collections.abc import Iterable

from agent_harness.models import Backend, BackendCapabilities, BackendName


class BackendRegistry:
    def __init__(self, backends: Iterable[Backend]) -> None:
        self._backends = {backend.name: backend for backend in backends}

    def list(self) -> list[Backend]:
        return list(self._backends.values())

    def get(self, name: BackendName) -> Backend:
        return self._backends[name]

    def has(self, name: str) -> bool:
        return name in self._backends


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
        sandbox=["read-only", "workspace-write", "danger"],
        tools="granular",
        interrupt_external_runs=False,
    )
    return BackendRegistry(
        [
            Backend(name="claude-code", display_name="Claude Code", capabilities=claude),
            Backend(name="codex", display_name="Codex", capabilities=codex),
        ]
    )
