"""Locate the rollout file a freshly-spawned CLI subprocess will write.

Used by the orchestrator at run-spawn time to pre-register a
path → harness-session-id binding with the observer, so the observer
attributes incoming events to the correct row instead of falling back
to a filename-derived synthesized id.

Phase 2 scope:
- claude-code: rollout filename is deterministic from the dashed UUID
  the orchestrator passes via ``--session-id`` + the slugified cwd.
  No probe required; the discovery just composes the path.
- codex: handled by the observer's expectation registry
  (``ExternalTranscriptObserver.expect_codex_rollout``). The
  orchestrator registers a hint at spawn time and the observer matches
  by ``session_meta`` content. ``RolloutDiscovery`` is no longer
  involved on the codex path — no psutil/fd probe, no lsof fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _claude_slug(cwd: Path) -> str:
    """Compute the claude-code project-dir slug for ``cwd``.

    Mirrors what claude-code itself produces: replace each ``/`` (and
    the leading separator) with ``-``. Relative paths are slugged as-is
    so the helper is robust; production callers should pass absolute
    paths.
    """
    return cwd.as_posix().replace("/", "-")


class RolloutDiscovery:
    """Locate claude-code rollout files for newly-spawned subprocesses."""

    def __init__(
        self,
        *,
        claude_projects_root: Path | None = None,
    ) -> None:
        home = Path.home()
        self._claude_projects_root = (
            claude_projects_root or home / ".claude" / "projects"
        )

    def discover_claude(self, *, session_id: str, cwd: Path) -> Path:
        """Compute the deterministic claude-code rollout path.

        The path is ``<projects_root>/<slugified-cwd>/<session_id>.jsonl``,
        where ``session_id`` is the claude session id **as it appears in
        the rollout filename** — i.e. the 8-4-4-4-12 dashed UUID that
        the orchestrator passes via ``--session-id``. The harness's own
        ``ses_<hex>`` form must be converted by the caller before
        calling here (see ``orchestrator._pre_bind_claude_if_claude``).

        Returns the path even if the file does not yet exist — the
        observer's binding map records the path; ``tail_file`` will pick
        up content once claude flushes its first line.
        """
        return self._claude_projects_root / _claude_slug(cwd) / f"{session_id}.jsonl"
