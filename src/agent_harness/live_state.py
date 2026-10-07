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


class LiveStateRejected(Exception):
    """The session cannot take live-state claims (wrong kind, or archived)."""


@dataclass(frozen=True)
class LiveClaim:
    source: str
    producer: str
    sequence: int
    state: str
    received_at: datetime
    expires_at: datetime

    def busy_at(self, now: datetime) -> bool:
        return self.state == "busy" and now < self.expires_at


def ensure_accepts_live_state(session: Session) -> None:
    """Only external pi sessions take claims: those are the ones a live
    producer (Companion) drives and the harness can only observe. Harness-run
    sessions are owned by their runs; archived ones by the user."""
    if session.backend != "pi" or session.origin != "external":
        raise LiveStateRejected("live state is only accepted for external pi sessions")
    if session.status == "archived":
        raise LiveStateRejected("session is archived")


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

    def busy(self, session_id: str, now: datetime) -> bool:
        claim = self._claims.get(session_id)
        return claim is not None and claim.busy_at(now)

    def get(self, session_id: str) -> LiveClaim | None:
        return self._claims.get(session_id)

    def offer(self, session_id: str, request: LiveStateRequest, now: datetime) -> LiveClaim | None:
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
