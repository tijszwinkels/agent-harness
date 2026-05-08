from __future__ import annotations

from collections.abc import Iterable

from agent_harness.models import Backend, BackendCapabilities, BackendId


class BackendRegistry:
    def __init__(self, backends: Iterable[Backend]) -> None:
        self._backends = {backend.id: backend for backend in backends}

    def list(self) -> list[Backend]:
        return list(self._backends.values())

    def get(self, backend_id: BackendId) -> Backend:
        return self._backends[backend_id]

    def has(self, backend_id: str) -> bool:
        return backend_id in self._backends


def default_backend_registry() -> BackendRegistry:
    common = BackendCapabilities(
        launch_modes=["orchestrated", "observed"],
        supports_sse=True,
        supports_transcript_observation=True,
        supports_working_directory=True,
        supports_resume=True,
    )
    return BackendRegistry(
        [
            Backend(id="claude-code", display_name="Claude Code", capabilities=common),
            Backend(id="codex", display_name="Codex", capabilities=common),
        ]
    )
