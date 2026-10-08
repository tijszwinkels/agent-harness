"""Live busy/idle claims from a process that runs an external session.

Transcript freshness is only a guess for sessions the harness does not run:
an external agent that is thinking or running a long tool writes nothing for
a while. A process that drives such a session live (Companion driving Pi over
RPC) knows better, and reports it through ``PUT /v1/sessions/{id}/live-state``.

A claim is deliberately small, metadata only, and short-lived:

- ``busy`` holds a lease. While it is unexpired the session counts as working
  and transcript observations cannot change its status. The producer renews
  it with heartbeats; if the producer dies, the lease runs out and the usual
  30 s transcript rule takes over again, so nothing sticks.
- ``idle`` ends the turn: the session is set idle once (unless a harness run
  owns it) and transcript activity may move it again afterwards.

Claims are kept in memory only. A harness restart forgets them and the
producer re-asserts on its next heartbeat; nothing is added to the persisted
session rows, so a rollback to a version without this module stays safe.

Ownership: a producer may only report sessions it drives. Companion's Pi
process writes a custom transcript entry (``OWNER_ENTRY_TYPE``, data
``{"source": "companion"}``) into each conversation it runs; the harness
learns the owner from that entry (while tailing, or by an incremental,
bounded scan of the session's own recorded transcript path), persists it
(SQLite: a separate ``live_state_owners`` table, which older harness versions
simply ignore), and accepts claims only from that source. A terminal Pi session has no such entry and cannot be claimed.

Settled boundary: an ``idle`` claim records how far the session's transcript
reached when it arrived. Lines before that offset were written by the turn
that just ended; observing them later (the watcher is asynchronous) must not
change the session's status — whether through a running-kick or a metadata
``session.updated`` — though non-status metadata still applies. Lines after
it are new activity.

Ordering: each producer (one per producer process lifetime) numbers its
updates. A claim whose sequence is not newer than the last accepted claim of
the same producer is ignored, so duplicated or reordered requests cannot
resurrect an old state. A different producer id (the producer restarted)
replaces the previous claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from agent_harness.models import LiveStateRequest, Session

# Bound on remembered claims. Each is a handful of fields; the cap only
# matters if a misbehaving producer floods distinct session ids.
MAX_CLAIMS = 1024
MAX_OWNERS = 4096

# Custom Pi transcript entry that names the producer allowed to report a session.
OWNER_ENTRY_TYPE = "agent-harness.live-state-owner"
OWNER_MARKER = f'"{OWNER_ENTRY_TYPE}"'.encode()
# A session without a known owner is scanned (one more chunk) at most this often.
OWNER_RESCAN_SECONDS = 10


class LiveStateRejected(Exception):
    """The session cannot take live-state claims (wrong kind, archived, or
    not owned by the reporting source)."""


def owner_from_entry(line: str | bytes) -> str | None:
    """The source named by an owner entry line, or ``None``."""
    import json

    try:
        record = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict) or record.get("type") != "custom":
        return None
    if record.get("customType") != OWNER_ENTRY_TYPE:
        return None
    data = record.get("data")
    source = data.get("source") if isinstance(data, dict) else None
    if isinstance(source, str) and 0 < len(source) <= 32:
        return source
    return None


# Owner scans read the transcript in blocks of this size, a bounded number
# of bytes per call (``OWNER_SCAN_CHUNK_BYTES``), continuing where the last
# call stopped, so even a very long transcript is covered eventually.
OWNER_SCAN_BLOCK_BYTES = 1024 * 1024
OWNER_SCAN_CHUNK_BYTES = 16 * 1024 * 1024
# The owner entry is a short line; a "line" around a marker match longer
# than this is ignored rather than read.
OWNER_LINE_WINDOW = 4096


def event_offset(event: object) -> int | None:
    """Transcript offset an observer event came from, if any."""
    data = getattr(event, "data", None)
    offset = data.get("offset") if isinstance(data, dict) else None
    return offset if isinstance(offset, int) and offset >= 0 else None


def _owner_at(handle, position: int) -> str | None:
    """Parse the (short) line around ``position`` as an owner entry."""
    window_start = max(0, position - OWNER_LINE_WINDOW)
    handle.seek(window_start)
    window = handle.read(2 * OWNER_LINE_WINDOW)
    relative = position - window_start
    newline_before = window.rfind(b"\n", 0, relative)
    if newline_before == -1 and window_start > 0:
        return None  # longer than the window: not an owner entry
    line_start = newline_before + 1
    line_end = window.find(b"\n", relative)
    if line_end == -1:
        return None  # incomplete or too long
    return owner_from_entry(window[line_start:line_end])


def scan_owner_chunk(path: str | None, start: int, budget: int | None = None) -> tuple[str | None, int]:
    """Look for the owner entry in ``[start, start + budget)`` of a session's
    own transcript. Returns ``(owner, next_start)``.

    ``path`` comes from the harness's session record (never from a request).
    Memory is bounded by one block plus a line window, however long the
    transcript's lines are; the caller keeps ``next_start`` to continue.
    """
    import os
    from pathlib import Path

    if not path:
        return None, start
    if budget is None:
        budget = OWNER_SCAN_CHUNK_BYTES
    try:
        with Path(path).open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if start > size:
                start = 0  # truncated or replaced: start over
            end = min(size, start + budget)
            # Re-read a marker's length before ``start`` so a marker split by
            # the previous chunk boundary is still found.
            position = max(0, start - len(OWNER_MARKER))
            carry = b""
            while position < end:
                handle.seek(position)
                block = handle.read(min(OWNER_SCAN_BLOCK_BYTES, end - position))
                if not block:
                    break
                buffer = carry + block
                buffer_start = position - len(carry)
                found = buffer.find(OWNER_MARKER)
                while found != -1:
                    owner = _owner_at(handle, buffer_start + found)
                    if owner is not None:
                        return owner, end
                    found = buffer.find(OWNER_MARKER, found + 1)
                position += len(block)
                carry = buffer[-len(OWNER_MARKER):]
            return None, end
    except OSError:
        return None, start


def transcript_size(path: str | None) -> int | None:
    """Current size of a session's own transcript, or ``None``."""
    import os

    if not path:
        return None
    try:
        return os.stat(path).st_size
    except OSError:
        return None


@dataclass(frozen=True)
class LiveClaim:
    source: str
    producer: str
    sequence: int
    state: str
    received_at: datetime
    expires_at: datetime
    # For ``idle``: transcript size when the turn ended (see module docs).
    settled_offset: int | None = None

    def busy_at(self, now: datetime) -> bool:
        return self.state == "busy" and now < self.expires_at


def ensure_accepts_live_state(session: Session, *, source: str, owner: str | None) -> None:
    """Only external pi sessions take claims: those are the ones a live
    producer (Companion) drives and the harness can only observe. Harness-run
    sessions are owned by their runs; archived ones by the user. And only the
    producer named in the session's own transcript may report it."""
    if session.backend != "pi" or session.origin != "external":
        raise LiveStateRejected("live state is only accepted for external pi sessions")
    if session.status == "archived":
        raise LiveStateRejected("session is archived")
    if owner != source:
        raise LiveStateRejected(f"session is not owned by {source}")


def status_for_claim(session: Session, claim: LiveClaim, *, has_active_run: bool) -> str:
    """The status a newly accepted claim gives ``session``. A queued or
    running harness run keeps ownership (it sets running/idle itself)."""
    if has_active_run:
        return session.status
    return "running" if claim.state == "busy" else "idle"


class LiveStateRegistry:
    """Latest claim per session. Not thread-safe on its own: repositories
    call it under their own lock, together with the session row they guard."""

    def __init__(self) -> None:
        self._claims: dict[str, LiveClaim] = {}
        self._owners: dict[str, str] = {}
        self._owner_scans: dict[str, datetime] = {}
        self._scan_cursors: dict[str, int] = {}

    # ---- ownership

    def owner(self, session_id: str) -> str | None:
        return self._owners.get(session_id)

    def mark_owner(self, session_id: str, source: str) -> None:
        self._owners[session_id] = source
        self._owner_scans.pop(session_id, None)
        while len(self._owners) > MAX_OWNERS:
            del self._owners[next(iter(self._owners))]

    def scan_cursor(self, session_id: str) -> int:
        return self._scan_cursors.get(session_id, 0)

    def set_scan_cursor(self, session_id: str, offset: int) -> None:
        self._scan_cursors[session_id] = offset
        while len(self._scan_cursors) > MAX_OWNERS:
            del self._scan_cursors[next(iter(self._scan_cursors))]

    def should_scan_owner(self, session_id: str, now: datetime) -> bool:
        """Rate-limit transcript scans for sessions without a known owner."""
        if session_id in self._owners:
            return False
        last = self._owner_scans.get(session_id)
        if last is not None and now - last < timedelta(seconds=OWNER_RESCAN_SECONDS):
            return False
        self._owner_scans[session_id] = now
        while len(self._owner_scans) > MAX_OWNERS:
            del self._owner_scans[next(iter(self._owner_scans))]
        return True

    # ---- claims

    def superseded(self, session_id: str, offset: int) -> bool:
        """True for a transcript line written before the session's last
        settled ``idle`` (see module docs)."""
        claim = self._claims.get(session_id)
        return (
            claim is not None
            and claim.state == "idle"
            and claim.settled_offset is not None
            and offset < claim.settled_offset
        )

    def busy(self, session_id: str, now: datetime) -> bool:
        claim = self._claims.get(session_id)
        return claim is not None and claim.busy_at(now)

    def get(self, session_id: str) -> LiveClaim | None:
        return self._claims.get(session_id)

    def offer(
        self,
        session_id: str,
        request: LiveStateRequest,
        now: datetime,
        *,
        settled_offset: int | None = None,
    ) -> LiveClaim | None:
        """Record ``request`` if it is newer than what we have; return the new
        claim, or ``None`` when it is a stale/duplicate update."""
        current = self._claims.get(session_id)
        if current is not None and current.producer == request.producer and request.sequence <= current.sequence:
            return None
        claim = LiveClaim(
            source=request.source,
            producer=request.producer,
            sequence=request.sequence,
            state=request.state,
            received_at=now,
            expires_at=now + timedelta(seconds=request.lease_seconds),
            settled_offset=settled_offset if request.state == "idle" else None,
        )
        self._claims[session_id] = claim
        self._evict(now)
        return claim

    def forget(self, session_id: str) -> None:
        self._claims.pop(session_id, None)

    def _evict(self, now: datetime) -> None:
        if len(self._claims) <= MAX_CLAIMS:
            return
        # Expired busy claims and old idle claims carry no authority any more.
        for session_id, claim in list(self._claims.items()):
            if not claim.busy_at(now):
                del self._claims[session_id]
        while len(self._claims) > MAX_CLAIMS:
            oldest = min(self._claims, key=lambda sid: self._claims[sid].received_at)
            del self._claims[oldest]
