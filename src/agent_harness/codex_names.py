"""Codex thread names from ``$CODEX_HOME/session_index.jsonl``.

Codex keeps thread names outside its rollouts, in an append-only JSONL
index of ``{"id": <thread uuid>, "thread_name": ..., "updated_at": ...}``
entries. A rename appends an entry; the latest valid entry for an id *in
file order* is current (``updated_at`` is not consulted — it is codex's
metadata, not conversation activity). Removing a thread's names rewrites
the file without that id's entries and renames it into place
(``remove_thread_name_entries``, codex rust-v0.159.0), so the file's
identity changes.

This reader caches the latest name per id and reconciles incrementally:
appended complete lines are read from the last offset; a replaced,
truncated or rewritten file is re-read whole. A blank ``thread_name`` is a
deliberate removal (:data:`~agent_harness.native_titles.CLEARED`).
Removal by rewrite is inferred only from a trustworthy snapshot (see
:meth:`CodexNameIndex.refresh`); a missing, unreadable or partially written
index never clears anything.

Codex's paginated history keeps names primarily in its SQLite thread
store and writes this index best-effort, so a name that only exists in
that database is not visible here. See
specs/2026-10-07-native-title-coverage.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from agent_harness.native_titles import CLEARED, NativeName, native_name

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = "session_index.jsonl"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
# Bytes remembered just before the consumed offset, to notice a file that
# was rewritten in place (same inode) rather than appended to.
_TAIL_PROBE_BYTES = 64


def default_codex_name_index(codex_home: str | Path | None = None) -> Path:
    """``$CODEX_HOME/session_index.jsonl``, defaulting to ``~/.codex``."""
    home = codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex"
    return Path(home).expanduser() / INDEX_FILE_NAME


def parse_index_entry(line: bytes | str) -> tuple[str, NativeName] | None:
    """``(uuid, name)`` for a valid entry line, else ``None``."""
    try:
        record = json.loads(line)
    except (JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    thread_id = record.get("id")
    if not isinstance(thread_id, str) or not _UUID_RE.match(thread_id.lower()):
        return None
    name = native_name(record.get("thread_name"))
    if name is None:
        return None
    return thread_id.lower(), name


@dataclass(frozen=True, slots=True)
class _Snapshot:
    names: dict[str, NativeName]
    offset: int
    malformed: int
    complete: bool


def _read(path: Path, start: int = 0) -> _Snapshot:
    """Entries from ``start``, stopping before an unterminated last line."""
    names: dict[str, NativeName] = {}
    malformed = 0
    offset = start
    complete = True
    with path.open("rb") as handle:
        handle.seek(start)
        for line in handle:
            if not line.endswith(b"\n"):
                complete = False
                break
            offset += len(line)
            if not line.strip():
                continue
            entry = parse_index_entry(line)
            if entry is None:
                malformed += 1
                continue
            names[entry[0]] = entry[1]
    return _Snapshot(names=names, offset=offset, malformed=malformed, complete=complete)


class CodexNameIndex:
    """Cached view of one codex name index file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._names: dict[str, NativeName] = {}
        # Ids with a valid entry in the file as last read. Distinct from
        # ``_names``, which also keeps last-known names for ids no longer in
        # the file: only an id that was *present* and is gone after a
        # trustworthy replacement counts as removed.
        self._present: set[str] = set()
        self._identity: tuple[int, int] | None = None
        self._stamp: tuple[int, int] | None = None  # (size, mtime_ns)
        self._offset = 0
        self._tail_probe = b""
        self._malformed = 0
        self._trusted = False

    def get(self, thread_id: str) -> NativeName | None:
        return self._names.get(thread_id.lower())

    @property
    def trusted(self) -> bool:
        """Whether the last full read was a complete, fully valid snapshot.

        Startup uses this to treat a stored native title whose id is
        absent from the index as removed. A missing file is not trusted.
        """
        return self._trusted

    def refresh(self) -> dict[str, NativeName]:
        """Reconcile with the file; return ids whose current name changed.

        Removal inference — an id with a valid entry in the previous read
        and none after the file was replaced — requires: a new file
        identity (codex's write-then-rename), a successful read ending in
        a complete line, and no more malformed lines than the previous
        read (codex's rewrite keeps lines it can't parse; garbage is not a
        removal). Names cached from older reads but absent from the
        previous one are never inferred removed. A truncated or
        rewritten-in-place file is re-read but never used to infer
        removals; a missing or unreadable one changes nothing, and a file
        that reappears starts a fresh baseline.
        """
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            # Codex's rewrite renames over the file, so it is never absent
            # mid-replacement. Forget the identity: whatever appears next
            # is a fresh start, not a trustworthy replacement to infer
            # removals from. Cached names stay.
            self._identity = None
            self._stamp = None
            self._present = set()
            self._offset = 0
            self._tail_probe = b""
            self._trusted = False
            return {}
        except OSError:
            logger.debug("codex name index: cannot stat %s", self.path, exc_info=True)
            return {}
        identity = (stat.st_dev, stat.st_ino)
        stamp = (stat.st_size, stat.st_mtime_ns)
        if identity == self._identity and stamp == self._stamp:
            return {}
        try:
            # Fast path only for growth of the same file whose consumed
            # tail is unchanged. A same-size change (a rewrite in place)
            # or a shrink is re-read whole. The tail sample can't see a
            # rewrite further back combined with growth; codex never does
            # that (it appends, or renames a rewritten file into place).
            if (
                identity == self._identity
                and self._stamp is not None
                and stat.st_size > self._stamp[0]
                and self._tail_probe == self._probe(self._offset)
            ):
                return self._append(self._read_from(self._offset), stamp)
            snapshot = self._read_from(0)
        except OSError:
            logger.debug("codex name index: cannot read %s", self.path, exc_info=True)
            return {}
        replaced = self._identity is not None and identity != self._identity
        may_infer_removals = (
            replaced and snapshot.complete and snapshot.malformed <= self._malformed
        )
        changes: dict[str, NativeName] = {}
        names = dict(snapshot.names)
        for thread_id, previous in self._names.items():
            if thread_id in names:
                continue
            removed = may_infer_removals and thread_id in self._present
            names[thread_id] = CLEARED if removed and isinstance(previous, str) else previous
        for thread_id, name in names.items():
            if self._names.get(thread_id) != name:
                changes[thread_id] = name
        self._names = names
        self._present = set(snapshot.names)
        self._identity = identity
        self._stamp = stamp
        self._offset = snapshot.offset
        self._tail_probe = self._probe(snapshot.offset)
        self._malformed = snapshot.malformed
        self._trusted = snapshot.complete and snapshot.malformed == 0
        return changes

    def _read_from(self, start: int) -> _Snapshot:
        return _read(self.path, start)

    def _append(self, snapshot: _Snapshot, stamp: tuple[int, int]) -> dict[str, NativeName]:
        changes = {
            thread_id: name
            for thread_id, name in snapshot.names.items()
            if self._names.get(thread_id) != name
        }
        self._names.update(snapshot.names)
        self._present.update(snapshot.names)
        self._offset = snapshot.offset
        self._tail_probe = self._probe(snapshot.offset)
        self._malformed += snapshot.malformed
        self._stamp = stamp
        if snapshot.malformed:
            self._trusted = False
        return changes

    def _probe(self, offset: int) -> bytes:
        start = max(0, offset - _TAIL_PROBE_BYTES)
        try:
            with self.path.open("rb") as handle:
                handle.seek(start)
                return handle.read(offset - start)
        except OSError:
            return b""


__all__ = [
    "CodexNameIndex",
    "INDEX_FILE_NAME",
    "default_codex_name_index",
    "parse_index_entry",
]
