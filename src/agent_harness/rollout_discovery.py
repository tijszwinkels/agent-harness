"""Locate the rollout file a freshly-spawned CLI subprocess is writing to.

Used by the orchestrator at run-spawn time to pre-register the
path → harness-session-id binding with the observer, so the observer
attributes incoming events to the correct row instead of falling back
to a filename-derived synthesized id.

Two backends:
- claude-code: rollout filename is deterministic from
  ``--session-id <ses_hex>`` + the slugified cwd; no probe needed.
- codex: rollout path is opaque from the harness's side. We probe
  the subprocess's open fds via ``psutil.Process.open_files()`` and
  filter for paths under ``~/.codex/sessions/``. psutil dispatches to
  ``/proc/<pid>/fd/`` on Linux and to ``libproc`` on macOS. The codex
  process opens its rollout within the first few hundred milliseconds
  of spawn but not strictly synchronously, so we poll briefly. On
  platforms where ``open_files()`` raises ``NotImplementedError``, we
  fall back to invoking ``lsof -p <pid> -F n``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RolloutDiscoveryError(Exception):
    """Raised when the rollout file cannot be located for a subprocess.

    Carries enough context (pid, timeout, paths considered) for an
    operator to correlate a failure to a specific run in the harness
    logs.
    """


@dataclass(frozen=True, slots=True)
class _CodexFd:
    """Minimal projection of a psutil ``popenfile`` (or an lsof line) —
    only the path matters for our filter logic."""

    path: Path


ProcessFactory = Callable[[int], Any]
"""Factory that returns a psutil.Process-shaped object for a given pid.

Pluggable for testing — tests pass a fake that exposes ``open_files()``.
In production this is ``psutil.Process``.
"""


def _claude_slug(cwd: Path) -> str:
    """Compute the claude-code project-dir slug for ``cwd``.

    Mirrors what claude-code itself produces: replace each ``/`` (and
    the leading separator) with ``-``. Relative paths are slugged as-is
    so the helper is robust; production callers should pass absolute
    paths.
    """
    text = cwd.as_posix()
    return text.replace("/", "-")


def _default_process_factory(pid: int) -> Any:
    # Imported lazily so importing this module is cheap and so that
    # environments without psutil installed can still load the module
    # for the deterministic claude path (psutil is only required for
    # the codex probe).
    import psutil  # type: ignore[import-not-found]

    return psutil.Process(pid)


class RolloutDiscovery:
    """Locate rollout files for newly-spawned CLI subprocesses."""

    def __init__(
        self,
        *,
        claude_projects_root: Path | None = None,
        codex_sessions_root: Path | None = None,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        home = Path.home()
        self._claude_projects_root = (
            claude_projects_root or home / ".claude" / "projects"
        )
        self._codex_sessions_root = codex_sessions_root or home / ".codex" / "sessions"
        self._process_factory = process_factory or _default_process_factory

    def discover_claude(self, *, session_id: str, cwd: Path) -> Path:
        """Compute the deterministic claude-code rollout path.

        The path is ``<projects_root>/<slugified-cwd>/<session_id>.jsonl``,
        where ``session_id`` is the claude session id **as it appears in
        the rollout filename** — i.e. the 8-4-4-4-12 dashed UUID that
        the orchestrator passes via ``--session-id``. The harness's own
        ``ses_<hex>`` form must be converted by the caller before
        calling here (see ``orchestrator._pre_bind_claude``).

        Returns the path even if the file does not yet exist — the
        observer's binding map records the path; ``tail_file`` will pick
        up content once claude flushes its first line.
        """
        return self._claude_projects_root / _claude_slug(cwd) / f"{session_id}.jsonl"

    async def discover_codex(
        self,
        *,
        pid: int,
        cwd: Path,
        timeout_seconds: float = 5.0,
        poll_interval: float = 0.05,
    ) -> Path:
        """Locate the rollout file the codex subprocess just opened.

        Polls the subprocess's open fds (via psutil, lsof fallback)
        until a path matching ``codex_sessions_root`` appears. When
        multiple match, narrows by reading each candidate's
        ``session_meta.cwd`` and preferring an exact match against
        ``cwd``; if no candidate has flushed session_meta yet, the
        youngest by mtime wins.

        Raises ``RolloutDiscoveryError`` if no rollout fd appears within
        ``timeout_seconds``.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout_seconds, 0.0)
        matches: list[Path] = []

        while True:
            try:
                process = self._process_factory(pid)
                open_files = list(process.open_files())
            except NotImplementedError:
                open_files = self._lsof_open_files(pid)
            except Exception:
                # Process likely exited before we probed; let the timeout
                # logic decide whether to keep retrying or raise.
                logger.debug("psutil probe raised for pid=%s; retrying", pid)
                open_files = []

            matches = list(self._filter_codex_rollouts(open_files))
            if matches:
                break

            if loop.time() >= deadline:
                raise RolloutDiscoveryError(
                    f"Codex rollout fd did not appear within {timeout_seconds}s "
                    f"for pid={pid} (cwd={cwd}); checked under "
                    f"{self._codex_sessions_root}"
                )
            await asyncio.sleep(poll_interval)

        if len(matches) == 1:
            return matches[0]
        return _narrow_by_cwd(matches, cwd)

    def _filter_codex_rollouts(self, open_files: Iterable[Any]) -> Iterable[Path]:
        root = self._codex_sessions_root
        for entry in open_files:
            raw_path = getattr(entry, "path", None)
            if raw_path is None:
                continue
            path = Path(raw_path)
            if path.suffix != ".jsonl":
                continue
            if not path.name.startswith("rollout-"):
                continue
            try:
                path.relative_to(root)
            except ValueError:
                continue
            yield path

    def _lsof_open_files(self, pid: int) -> list[_CodexFd]:
        """Defensive fallback for platforms where psutil.open_files()
        raises NotImplementedError.

        ``lsof -p <pid> -F n`` emits a record per fd; ``n<path>`` lines
        carry the path. We parse and project to the same shape as the
        psutil result.
        """
        try:
            result = subprocess.run(
                ["lsof", "-p", str(pid), "-F", "n"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except FileNotFoundError:
            logger.warning(
                "lsof not available for fallback codex fd probe (pid=%s); "
                "returning empty fd list",
                pid,
            )
            return []
        except subprocess.TimeoutExpired:
            logger.warning("lsof probe timed out for pid=%s", pid)
            return []

        # lsof exits non-zero when a process has no open files of a
        # given kind — that's not a hard error for our purposes.
        if result.returncode not in (0, 1):
            logger.warning(
                "lsof probe exited unexpectedly for pid=%s: returncode=%s stderr=%s",
                pid,
                result.returncode,
                result.stderr[-200:] if result.stderr else "",
            )

        out: list[_CodexFd] = []
        for line in (result.stdout or "").splitlines():
            if line.startswith("n"):
                out.append(_CodexFd(path=Path(line[1:])))
        return out


def _narrow_by_cwd(paths: list[Path], cwd: Path) -> Path:
    """Tiebreaker when more than one rollout fd matches the codex root.

    Read each candidate's first line (typically ``session_meta``) and
    prefer one whose ``payload.cwd`` matches ``cwd`` exactly. If none of
    the candidates has flushed session_meta yet, fall back to the
    youngest file by mtime.
    """
    cwd_str = cwd.as_posix()
    cwd_match: Path | None = None
    for candidate in paths:
        meta_cwd = _read_session_meta_cwd(candidate)
        if meta_cwd is not None and meta_cwd == cwd_str:
            cwd_match = candidate
            break
    if cwd_match is not None:
        return cwd_match

    # No cwd-match — youngest mtime wins.
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return max(paths, key=_mtime)


def _read_session_meta_cwd(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            first = fh.readline()
    except OSError:
        return None
    if not first.endswith(b"\n"):
        # Partial flush — caller will pick the mtime fallback.
        return None
    try:
        record = json.loads(first.decode("utf-8", errors="replace").strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(record, Mapping) or record.get("type") != "session_meta":
        return None
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    cwd = payload.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else None
