"""Backend-native conversation names, read from transcript metadata records.

pi appends ``{"type":"session_info","name":...}`` on every ``/name``.
claude appends ``custom-title`` (``/rename``) and ``ai-title`` (its own
generated title) records, and displays the custom title in preference to
the generated one. Each source is a *slot*. Within a slot the latest
complete record wins; across slots the backend's preference order decides,
regardless of which record came last.

A record whose name is an explicit blank string is a deliberate removal
(:data:`CLEARED`). Anything else that isn't a usable name — a missing
field, a non-string, malformed JSON, a record for another session — says
nothing and leaves the slot as it was.

The observer applies the resulting name to ``Session.title``; the
provenance and precedence rules live in ``agent_harness.models``. See
specs/2026-10-07-native-title-coverage.md.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


class _Cleared:
    """Sentinel for "the backend's name was deliberately removed"."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "CLEARED"


CLEARED: Final = _Cleared()
NativeName = str | _Cleared

# Slot preference per backend, most preferred first.
SLOT_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "pi": ("name",),
    "claude-code": ("custom", "ai"),
}

# (record type, name field, slot) per backend.
_RECORD_SLOTS: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "pi": (("session_info", "name", "name"),),
    "claude-code": (
        ("custom-title", "customTitle", "custom"),
        ("ai-title", "aiTitle", "ai"),
    ),
}

# Byte markers for a cheap pre-filter before any JSON decoding.
_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    backend: tuple(record_type for record_type, _, _ in slots)
    for backend, slots in _RECORD_SLOTS.items()
}

DEFAULT_MAX_TRACKED_TRANSCRIPTS = 512


def native_name(value: object) -> NativeName | None:
    """A record's name field as a title, :data:`CLEARED`, or ``None``.

    Whitespace runs (including newlines) collapse to one space, as pi
    does. A string that is blank after that is an explicit removal; a
    non-string carries no information.
    """
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or CLEARED


def native_title_slot(
    backend: str,
    record: object,
    *,
    transcript_uuid: str | None = None,
) -> tuple[str, NativeName] | None:
    """``(slot, name)`` for a native-title record, else ``None``.

    claude stamps a ``sessionId`` on its title records. When present it
    must match the transcript (``transcript_uuid``); a record naming some
    other conversation is ignored rather than trusted.
    """
    if not isinstance(record, Mapping):
        return None
    record_type = record.get("type")
    for expected_type, name_field, slot in _RECORD_SLOTS.get(backend, ()):
        if record_type != expected_type:
            continue
        if "sessionId" in record and not _same_session(
            record.get("sessionId"), transcript_uuid
        ):
            return None
        name = native_name(record.get(name_field))
        return None if name is None else (slot, name)
    return None


def native_title_slot_from_line(
    backend: str,
    line: str | bytes,
    *,
    transcript_uuid: str | None = None,
) -> tuple[str, NativeName] | None:
    markers = _MARKERS.get(backend)
    if not markers:
        return None
    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
    if not any(marker in text for marker in markers):
        return None
    try:
        record = json.loads(text)
    except JSONDecodeError:
        return None
    return native_title_slot(backend, record, transcript_uuid=transcript_uuid)


def effective_native_title(
    backend: str, slots: Mapping[str, NativeName]
) -> NativeName | None:
    """The name a backend would display for these slots.

    The first slot (in preference order) holding a name wins. With no name
    left anywhere, a deliberately cleared slot means "no native title";
    with nothing known at all, ``None``.
    """
    order = SLOT_ORDER.get(backend, ())
    for slot in order:
        value = slots.get(slot)
        if isinstance(value, str):
            return value
    if any(slots.get(slot) is CLEARED for slot in order):
        return CLEARED
    return None


def scan_native_titles(
    path: str | Path,
    backend: str,
    *,
    end_offset: int,
    transcript_uuid: str | None = None,
) -> dict[str, NativeName] | None:
    """Slots recovered from the first ``end_offset`` bytes of a transcript.

    Only complete (newline-terminated) lines inside the bound count: the
    observer has consumed exactly that prefix, and reading further would
    apply names ahead of the ordered tail. Never raises.

    ``None`` when the prefix can't be recovered in full — the file is
    unreadable, or shorter than (or no longer line-aligned at) the
    consumed offset, e.g. after truncation. Callers then keep the stored
    title and try again later rather than trusting partial slots: a
    missing custom slot would otherwise let a generated title (or its
    removal) take over.
    """
    slots: dict[str, NativeName] = {}
    if end_offset <= 0:
        return slots
    markers = tuple(marker.encode() for marker in _MARKERS.get(backend, ()))
    consumed = 0
    try:
        with Path(path).open("rb") as handle:
            for line in handle:
                if consumed + len(line) > end_offset or not line.endswith(b"\n"):
                    break
                consumed += len(line)
                if not any(marker in line for marker in markers):
                    continue
                found = native_title_slot_from_line(
                    backend, line, transcript_uuid=transcript_uuid
                )
                if found is not None:
                    slots[found[0]] = found[1]
    except OSError:
        logger.debug("native title scan: cannot read %s", path, exc_info=True)
        return None
    if consumed != end_offset:
        logger.debug(
            "native title scan: %s has %d of %d consumed bytes", path, consumed, end_offset
        )
        return None
    return slots


def _same_session(value: object, transcript_uuid: str | None) -> bool:
    if not isinstance(value, str) or transcript_uuid is None:
        return False
    return value.replace("-", "").lower() == transcript_uuid.replace("-", "").lower()


@dataclass(slots=True)
class _Entry:
    slots: dict[str, NativeName] = field(default_factory=dict)
    hydrated: bool = False


class NativeTitleTracker:
    """Bounded per-transcript slot state.

    LRU-capped like ``PiTranscriptRegistry``: an evicted transcript is
    simply re-hydrated from its consumed prefix the next time it changes.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_TRACKED_TRANSCRIPTS) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[Path, _Entry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def _entry(self, path: Path) -> _Entry:
        entry = self._entries.get(path)
        if entry is None:
            entry = _Entry()
            self._entries[path] = entry
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(path)
        return entry

    def needs_hydration(self, path: str | Path) -> bool:
        entry = self._entries.get(Path(path))
        return entry is None or not entry.hydrated

    def hydrate(self, path: str | Path, slots: Mapping[str, NativeName]) -> None:
        entry = self._entry(Path(path))
        entry.slots = dict(slots)
        entry.hydrated = True

    def observe(self, path: str | Path, slot: str, name: NativeName) -> None:
        self._entry(Path(path)).slots[slot] = name

    def effective(self, path: str | Path, backend: str) -> NativeName | None:
        """The current name, or ``None`` until the slots are trustworthy.

        Records observed before a successful hydration are kept, but say
        nothing on their own: the prefix they follow is still unknown.
        """
        entry = self._entries.get(Path(path))
        if entry is None or not entry.hydrated:
            return None
        return effective_native_title(backend, entry.slots)


__all__ = [
    "CLEARED",
    "NativeName",
    "NativeTitleTracker",
    "SLOT_ORDER",
    "effective_native_title",
    "native_name",
    "native_title_slot",
    "native_title_slot_from_line",
    "scan_native_titles",
]
