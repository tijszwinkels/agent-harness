from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Mapping, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import (
    BackendName,
    Event,
    ImageBlock,
    Message,
    MessageBlock,
    MessageRole,
    Project,
    Session,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    utc_now,
)
from agent_harness.repository import (
    InMemoryRepository,
    MaterializationDeferred,
    SessionNotFoundError,
)
from agent_harness.usage import (
    parse_claude_context_snapshot,
    parse_claude_usage,
    parse_codex_context_snapshot,
    parse_codex_token_count,
)

try:
    from watchfiles import awatch
except ImportError:  # pragma: no cover - dependency is declared, this protects embedded use.
    awatch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CODEX_ROLLOUT_RE = re.compile(
    r"^rollout-.+-(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)
# Claude rollout record-types we intentionally drop on the floor.
# Phase 4 audit: every entry here has a "record genuinely doesn't
# carry message-shaped content for us" rationale — not "avoid
# double-emit because stdout already did it" (the dual-path
# motivation went away with Phase 2).
#
# - ``attachment``: claude's image/file-attachment marker — payload
#   shape is image-only; the canonical ``user``/``assistant`` record
#   carries the image content blocks if they're conversation
#   participants.
# - ``last-prompt``: claude's bookkeeping for the most-recent user
#   prompt; duplicate of the matching ``user`` record.
# - ``pr-link``: claude-code's GitHub PR-link metadata; not
#   conversational.
# - ``queue-operation``: claude-code's internal queue housekeeping.
# - ``system``: claude system-message hooks; not conversational.
_IGNORED_CLAUDE_RECORD_TYPES = {
    "attachment",
    "last-prompt",
    "pr-link",
    "queue-operation",
    "system",
}

# Codex rollout record-types (outer ``type``) we drop on the floor.
# Phase 4 audit:
#
# - ``compacted``: codex's compaction snapshot — observability
#   without a data-plane consumer.
# - ``session_meta``: parsed separately (cwd + timestamp lookups for
#   the expectation registry); ignoring here prevents duplicate
#   session.updated emission.
# - ``turn_context``: NEW in Phase 4 — codex emits this as a
#   per-turn context-only marker. It carries cwd + model which
#   ``_session_event_if_complete`` already extracts from
#   ``payload``; without this entry the parser falls through to the
#   "Unsupported Codex transcript shape" warning, spamming logs
#   (Orion's worth-noting #3 on PR #15).
_IGNORED_CODEX_RECORD_TYPES = {"compacted", "session_meta", "turn_context"}
_IGNORED_CODEX_PAYLOAD_TYPES = {
    # ``agent_message`` and ``assistant_message`` are codex's
    # ``event_msg`` form of the assistant turn; the canonical
    # representation is ``response_item/message`` which carries the
    # full content blocks. Ignoring the event_msg form is what stops
    # the observer from emitting the same assistant message twice.
    "agent_message",
    "assistant_message",
    # ``context_compacted`` carries a codex-side summary of the
    # compaction the runtime just performed; observability without a
    # data-plane consumer yet. Demoted to "metadata", revisit if
    # mm-bridge / command-bridge grow a need for it.
    "context_compacted",
    # ``task_started`` is the matching bookend to ``task_complete``;
    # ``task_complete`` is now surfaced as ``run.end_turn`` (Phase 4).
    # ``task_started`` stays ignored — the harness's own
    # ``run.started`` event covers the same lifecycle moment from
    # the supervisor's side.
    "task_started",
    # ``web_search_call`` is the standalone codex search affordance
    # (no harness consumer; the response_item/function_call form is
    # what tool-use parsers care about).
    "web_search_call",
    #
    # ``token_count`` was removed from this set in Phase 3 — surfaced
    # as ``run.usage``. ``task_complete`` was removed in Phase 4 —
    # surfaced as ``run.end_turn``. Both have an explicit branch in
    # ``_parse_codex_record`` before this gate fires.
}


@dataclass(frozen=True, slots=True)
class TranscriptIdentity:
    backend: BackendName
    path: Path
    session_id: str
    # True when ``session_id`` came from an orchestrator-side
    # ``bind_rollout`` (i.e. routes to a harness session id) rather
    # than from the filename pattern. Suppresses the synthetic
    # ``session.updated`` (origin=external) the parser would otherwise
    # emit — that session already exists and is harness-owned. The
    # repository's origin-downgrade guard would skip the upsert, but
    # the event still travels the bus and confuses subscribers.
    is_rebound: bool = False


@dataclass(frozen=True, slots=True)
class CodexRolloutExpectation:
    """A hint from the orchestrator that a codex rollout from ``cwd``
    is about to appear, and should route to ``session_id`` rather than
    the synthesized ``codex_<uuid>`` external row.

    The observer matches incoming codex rollouts against active
    expectations by ``session_meta.cwd`` + a ±30s timestamp window —
    content-based, not timing-based, so a watchfiles inotify firing
    BEFORE the orchestrator's spawn-side handoff can still resolve
    correctly. Expectations have a TTL so a failed/cancelled spawn
    doesn't leave a permanent ghost hint.
    """

    cwd: Path
    session_id: str
    registered_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class _SessionMetaPeek:
    """Outcome of peeking the first line of a codex rollout file.

    ``retry`` distinguishes "partial flush — try again next tick" from
    "we read enough to know this isn't a codex rollout with a
    session_meta head"; only the latter gets memoized as "no match"
    by the caller.
    """

    cwd: str | None
    timestamp: datetime | None
    retry: bool = False


class ObserverOffsetStore(Protocol):
    def get_observer_offsets(self) -> dict[str, int]:
        pass

    def set_observer_offset(self, path: str, next_offset: int) -> None:
        pass


def _offset_store_from(repository: object) -> "ObserverOffsetStore | None":
    # Duck-type: SQLite-backed repositories expose persistence; the
    # in-memory repository does not. We don't synthesize an in-memory
    # store because the existing ``next_offsets`` dict already plays
    # that role for the observer's lifetime.
    if repository is None:
        return None
    get = getattr(repository, "get_observer_offsets", None)
    setter = getattr(repository, "set_observer_offset", None)
    if callable(get) and callable(setter):
        return repository  # type: ignore[return-value]
    return None


@dataclass(slots=True)
class ObserverState:
    seen_offsets: dict[Path, set[int]] = field(default_factory=dict)
    next_offsets: dict[Path, int] = field(default_factory=dict)
    # Write-through store for ``next_offsets``. Without it the observer
    # loses its place on harness restart and re-emits every transcript
    # line — the 2026-05-15 MM flood.
    offset_store: ObserverOffsetStore | None = None

    @classmethod
    def from_store(cls, store: ObserverOffsetStore) -> "ObserverState":
        persisted = store.get_observer_offsets()
        return cls(
            next_offsets={Path(p): int(off) for p, off in persisted.items()},
            offset_store=store,
        )

    def mark_seen(self, path: Path, offset: int) -> bool:
        offsets = self.seen_offsets.setdefault(path, set())
        if offset in offsets:
            return False
        offsets.add(offset)
        return True

    def next_offset(self, path: Path) -> int:
        return self.next_offsets.get(path, 0)

    def set_next_offset(self, path: Path, offset: int) -> None:
        self.next_offsets[path] = offset
        if self.offset_store is not None:
            try:
                self.offset_store.set_observer_offset(str(path), offset)
            except Exception:
                logger.exception(
                    "Failed to persist observer offset for %s; in-memory "
                    "state is up to date but a restart will re-tail.",
                    path,
                )


DEFAULT_IDLE_AFTER_SECONDS = 30.0
# Codex rollout expectation knobs (two dimensions, two purposes):
#
# - ``DEFAULT_EXPECTATION_TTL_SECONDS`` (60s) — how long a registered
#   expectation stays eligible to match before ``_purge_expired_expectations``
#   evicts it.
# - ``_EXPECTATION_TIMESTAMP_WINDOW`` (30s) — how far the rollout's
#   ``session_meta.timestamp`` may diverge from the expectation's
#   ``registered_at`` for them to match.
#
# TTL > window deliberately. A slow codex spawn (>10s before
# session_meta is flushed) is normal; if TTL ≤ window the expectation
# would be evicted while still timing-eligible to match, silently
# falling through to the filename-pattern path and re-instating the
# Heron-PR-#12 dupe-session symptom. The TTL provides a generous
# buffer for codex startup jitter; the window enforces the
# tight-coupling guarantee at match time.
DEFAULT_EXPECTATION_TTL_SECONDS = 60.0
_EXPECTATION_TIMESTAMP_WINDOW = timedelta(seconds=30)


class ExternalTranscriptObserver:
    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        repository: InMemoryRepository | None = None,
        state: ObserverState | None = None,
        idle_after_seconds: float = DEFAULT_IDLE_AFTER_SECONDS,
        clock: Callable[[], datetime] | None = None,
        expectation_ttl_seconds: float = DEFAULT_EXPECTATION_TTL_SECONDS,
    ) -> None:
        self._event_bus = event_bus
        self._repository = repository
        if state is None:
            offset_store = _offset_store_from(repository)
            state = (
                ObserverState.from_store(offset_store)
                if offset_store is not None
                else ObserverState()
            )
        self._state = state
        self._pending_materialization: dict[str, list[Event]] = {}
        self._idle_after = timedelta(seconds=idle_after_seconds)
        self._clock = clock or utc_now
        # Pre-bindings populated by the orchestrator's spawn path (via
        # ``bind_rollout``) — claude pre-bind path. When the observer
        # later sees a rollout path, a binding here wins over the
        # filename-pattern lookup. Pre-binding before the file exists
        # is supported: the binding is consulted only when ``tail_file``
        # encounters the path.
        self._path_to_session: dict[Path, str] = {}
        # Phase 2: codex rollout expectations. The orchestrator hints
        # ("I'm spawning codex from cwd X at time T"); the observer
        # consumes by reading session_meta from the rollout's first
        # line and matching against active expectations. Content-based
        # so the watchfiles inotify ↔ orchestrator-side race that
        # plagued Phase 1's psutil fd probe dissolves.
        self._codex_expectations: list[CodexRolloutExpectation] = []
        self._expectation_ttl = timedelta(seconds=expectation_ttl_seconds)
        # Memoize "we already peeked and this rollout doesn't have a
        # session_meta head" so we don't re-read the file on every
        # tail_file tick for the same path. Distinct from
        # ``_path_to_session``: that's "matched, route to session_id";
        # this is "checked, falls through to filename pattern".
        #
        # Phase 4 hardening (Aegis-flagged worth-noting on PR #14):
        # the cache is cleared by ``expect_codex_rollout`` on every
        # new spawn, so a tail_file that races ahead of expectation
        # registration can't poison the entry. For genuinely-external
        # rollouts (no expectation ever fires for them), the cache
        # accumulates one entry per rollout the observer encounters —
        # bounded by the rollout's own lifecycle; if the harness
        # processes thousands of distinct external rollouts across a
        # long uptime, a periodic sweep could be added (not load-bearing
        # today).
        self._codex_resolution_cache: dict[Path, bool] = {}
        # Per-session last-seen-transcript-event timestamp. Drives the
        # bidirectional running ↔ idle status transitions: a tick past the
        # threshold flips silent sessions to idle; a fresh event on an idle
        # session kicks it back to running.
        self._last_event_at: dict[str, datetime] = {}
        # Seed last-seen timestamps from the repository so freshness_tick
        # can heal records that pre-date this process. Without this seed,
        # restarts leave already-stale running sessions stranded — the
        # observer offset persistence (d7658fd) means existing transcripts
        # are not re-scanned, so _last_event_at would stay empty for them.
        self._seed_last_event_at_from_repository()

    def _seed_last_event_at_from_repository(self) -> None:
        if self._repository is None:
            return
        try:
            sessions = self._repository.list_sessions()
        except Exception:
            logger.exception("Failed to seed observer last-event map from repository")
            return
        for session in sessions:
            if session.origin != "external":
                continue
            # Use the session's updated_at as a proxy for the last transcript
            # event. It may be earlier than the actual rollout mtime, but the
            # freshness tick only cares about the running→idle threshold —
            # any timestamp older than the threshold flips the session
            # exactly once, which is the desired migration behavior.
            self._last_event_at[session.id] = session.updated_at

    def bind_rollout(self, path: str | Path, session_id: str) -> None:
        """Pre-register a path → harness-session-id mapping.

        Called by the orchestrator immediately after (or before) a CLI
        subprocess spawns. When ``tail_file`` later encounters
        ``path``, it uses ``session_id`` instead of deriving one from
        the filename pattern. Idempotent: a second call for the same
        path overwrites the prior binding.
        """
        self._path_to_session[Path(path)] = session_id

    def unbind_rollout(self, path: str | Path) -> None:
        """Drop a previously-bound path from the resolver map.

        Used for explicit path-based eviction (claude pre-bind knows
        the exact path it bound). Codex matches go through
        ``unbind_session`` instead — the orchestrator doesn't know
        which rollout path the observer ended up matching. Idempotent.
        """
        self._path_to_session.pop(Path(path), None)
        # Forget any "checked-no-match" memoization too, so a fresh
        # session_meta peek can run if the rollout reappears.
        self._codex_resolution_cache.pop(Path(path), None)

    def unbind_session(self, session_id: str) -> None:
        """Evict every path binding pointing at ``session_id``.

        Called by the orchestrator's ``RunProcess.run()`` finally
        block unconditionally for harness-origin sessions. Walks
        ``_path_to_session`` and removes any entry whose value matches.
        This is the cleanup hook for codex (where the observer
        records the binding from the content-matched rollout path
        the orchestrator never sees), but it also covers the claude
        path uniformly so the two backends converge on one eviction
        codepath.

        Without this hook ``_path_to_session`` would grow by one
        entry per codex harness run across the whole harness lifetime
        — each codex run opens a fresh rollout file and the binding
        for the finished run is dead weight. Same growth pattern
        Phase 1 fixed for claude; Phase 2 introduces the
        codex-side variant that needed its own eviction. Idempotent:
        no matching paths is a no-op.
        """
        to_remove = [
            path
            for path, bound_sid in self._path_to_session.items()
            if bound_sid == session_id
        ]
        for path in to_remove:
            self._path_to_session.pop(path, None)
            self._codex_resolution_cache.pop(path, None)

    def expect_codex_rollout(self, *, cwd: Path | str, session_id: str) -> None:
        """Register a hint that a codex rollout matching ``cwd`` is about
        to appear and should route to ``session_id``.

        Called by the orchestrator at codex spawn time. The observer
        matches incoming codex rollouts against active expectations by
        ``session_meta.cwd`` + a ±30s timestamp window from the
        expectation's registration time. Expectations expire after
        ``expectation_ttl_seconds`` (default 60s — TTL > window so a
        slow codex spawn doesn't lose its expectation; see the
        constant definitions for the rationale) so a failed spawn
        doesn't strand a permanent ghost hint.
        """
        now = self._clock()
        self._codex_expectations.append(
            CodexRolloutExpectation(
                cwd=Path(cwd),
                session_id=session_id,
                registered_at=now,
                expires_at=now + self._expectation_ttl,
            )
        )
        # Phase 4: invalidate the "checked-no-match" memo. Without
        # this, if ``tail_file`` peeked a rollout BEFORE this
        # expectation was registered (e.g. watchfiles fires fast on
        # a brand-new codex spawn), the path would be permanently
        # cached as "no match" — even though the just-registered
        # expectation would match it. The cache is a small dict; a
        # full clear on each spawn is cheap and corrects the race
        # cleanly. Phase 3 left this as a TODO; Phase 4 closes it.
        self._codex_resolution_cache.clear()

    def _resolve_identity(self, transcript_path: Path) -> TranscriptIdentity | None:
        """Resolve the identity that owns events from ``transcript_path``.

        Returns ``None`` when codex's session_meta is in a partial
        flush — caller should retry on the next tail tick rather than
        attribute events to the wrong session id.
        """
        base = transcript_identity_from_path(transcript_path)
        bound = self._path_to_session.get(transcript_path)
        if bound is not None and bound != base.session_id:
            return TranscriptIdentity(
                backend=base.backend,
                path=transcript_path,
                session_id=bound,
                is_rebound=True,
            )
        if base.backend == "codex" and bound is None:
            return self._resolve_codex_identity(transcript_path, base)
        return base

    def _resolve_codex_identity(
        self, transcript_path: Path, base: TranscriptIdentity
    ) -> TranscriptIdentity | None:
        # Memoized "we already checked this path and there's no
        # matching expectation". Falls through to the filename-pattern
        # base identity (external codex_<uuid> row) — the desired
        # behavior for genuinely-external codex sessions.
        if self._codex_resolution_cache.get(transcript_path) is True:
            return base

        peek = self._peek_session_meta(transcript_path)
        if peek.retry:
            # Partial flush — return None so the caller skips this
            # tail. The expectation stays alive for the next tick.
            return None

        if peek.cwd is None or peek.timestamp is None:
            # First line isn't a session_meta record (or no timestamp).
            # Memoize so we don't re-read on every event line.
            self._codex_resolution_cache[transcript_path] = True
            return base

        self._purge_expired_expectations()
        expectation = self._find_matching_expectation(
            cwd=peek.cwd, timestamp=peek.timestamp
        )
        if expectation is None:
            self._codex_resolution_cache[transcript_path] = True
            return base

        self._consume_expectation(expectation)
        self._path_to_session[transcript_path] = expectation.session_id
        return TranscriptIdentity(
            backend="codex",
            path=transcript_path,
            session_id=expectation.session_id,
            is_rebound=True,
        )

    def _purge_expired_expectations(self) -> None:
        now = self._clock()
        self._codex_expectations = [
            e for e in self._codex_expectations if e.expires_at > now
        ]

    def _find_matching_expectation(
        self, *, cwd: str, timestamp: datetime
    ) -> CodexRolloutExpectation | None:
        """Among unexpired expectations whose cwd matches and whose
        registered_at lies within the timestamp window, return the one
        closest in time. Closest-wins disambiguates the rare
        back-to-back same-cwd spawn race.
        """
        cwd_path = Path(cwd)
        candidates: list[tuple[timedelta, CodexRolloutExpectation]] = []
        for exp in self._codex_expectations:
            if exp.cwd != cwd_path:
                continue
            delta = abs(exp.registered_at - timestamp)
            if delta > _EXPECTATION_TIMESTAMP_WINDOW:
                continue
            candidates.append((delta, exp))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _consume_expectation(self, expectation: CodexRolloutExpectation) -> None:
        try:
            self._codex_expectations.remove(expectation)
        except ValueError:
            # Already removed by a concurrent purge — fine.
            pass

    def _peek_session_meta(self, path: Path) -> _SessionMetaPeek:
        """Read the first record of a codex rollout file.

        Returns ``retry=True`` for partial-flush cases (no trailing
        newline on the first line, OSError opening the file). Returns
        ``cwd``/``timestamp`` only for a confirmed ``session_meta``
        record with both fields. Anything else (non-session_meta first
        line, malformed JSON) returns ``cwd=None`` and ``timestamp=None``
        with ``retry=False`` — the caller memoizes that result.
        """
        try:
            with path.open("rb") as fh:
                first = fh.readline()
        except OSError:
            logger.debug("session_meta peek: cannot open %s yet", path)
            return _SessionMetaPeek(cwd=None, timestamp=None, retry=True)
        if not first.endswith(b"\n"):
            return _SessionMetaPeek(cwd=None, timestamp=None, retry=True)
        try:
            record = json.loads(first.decode("utf-8", errors="replace").strip())
        except JSONDecodeError:
            logger.debug("session_meta peek: malformed first line in %s", path)
            return _SessionMetaPeek(cwd=None, timestamp=None)
        if not isinstance(record, Mapping):
            return _SessionMetaPeek(cwd=None, timestamp=None)
        if record.get("type") != "session_meta":
            return _SessionMetaPeek(cwd=None, timestamp=None)
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            cwd = None
        # Prefer the outer timestamp on the session_meta record (the
        # rollout's earliest event timestamp); fall back to the inner
        # payload timestamp for older codex versions.
        ts_str = record.get("timestamp") or payload.get("timestamp")
        if not isinstance(ts_str, str) or not ts_str:
            return _SessionMetaPeek(cwd=cwd, timestamp=None)
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("session_meta peek: unparseable timestamp %s in %s", ts_str, path)
            return _SessionMetaPeek(cwd=cwd, timestamp=None)
        return _SessionMetaPeek(cwd=cwd, timestamp=ts)

    async def tail_file(self, path: str | Path) -> list[Event]:
        transcript_path = Path(path)
        try:
            identity = self._resolve_identity(transcript_path)
        except ValueError:
            logger.warning("Unsupported transcript path: %s", transcript_path)
            return []
        # Codex partial-flush case: session_meta hasn't been newline-
        # terminated yet, so we can't confidently route events. Skip
        # this tick; watchfiles will fire again once the writer flushes.
        if identity is None:
            return []

        published: list[Event] = []
        try:
            with transcript_path.open("rb") as transcript:
                transcript.seek(self._state.next_offset(transcript_path))
                while line := transcript.readline():
                    # Lines without a trailing newline are partial flushes by
                    # the writer — leave the offset put so the next tail_file
                    # invocation re-reads from the start of the partial line
                    # once the rest (and the newline) arrives. Without this,
                    # the offset would advance past half a JSON object and
                    # the assistant turn would be lost.
                    if not line.endswith(b"\n"):
                        break
                    start_offset = transcript.tell() - len(line)
                    end_offset = transcript.tell()
                    published.extend(
                        await self.publish_line(
                            transcript_path,
                            line.decode("utf-8", errors="replace"),
                            offset=start_offset,
                            identity=identity,
                        )
                    )
                    self._state.set_next_offset(transcript_path, end_offset)
        except OSError:
            logger.exception("Failed to read transcript file: %s", transcript_path)
        return published

    async def publish_line(
        self,
        path: str | Path,
        line: str,
        *,
        offset: int,
        identity: TranscriptIdentity | None = None,
    ) -> list[Event]:
        transcript_path = Path(path)
        if not self._state.mark_seen(transcript_path, offset):
            logger.debug("Skipping duplicate transcript offset: path=%s offset=%s", transcript_path, offset)
            return []

        resolved_identity = identity or self._resolve_identity(transcript_path)
        if resolved_identity is None:
            # Codex partial-flush — defer this line to a later tick.
            return []
        published: list[Event] = []
        for event in parse_transcript_line(line, identity=resolved_identity, offset=offset):
            if event.event in ("run.usage", "run.end_turn"):
                # Parsers emit run.usage / run.end_turn with
                # ``run_id=None`` (no repository access from the pure
                # parser). Resolve the active run for the session here
                # and drop the event if no active run exists.
                resolved = self._resolve_active_run_id(event)
                if resolved is None:
                    continue
                event = resolved
            published_event = await self._publish_via_bus(event)
            if published_event is None:
                continue
            published.append(published_event)
            if published_event.session_id:
                self._last_event_at[published_event.session_id] = self._clock()
                # Fresh transcript event on a session that was idled
                # by a previous freshness tick → flip back to running.
                kick = await self._maybe_publish_status_flip(
                    published_event.session_id,
                    target_status="running",
                    identity=resolved_identity,
                    offset=offset,
                )
                if kick is not None:
                    published.append(kick)
        return published

    async def _publish_via_bus(self, event: Event) -> Event | None:
        """Publish through the bus and route side effects.

        DurableEventBus does the materialize internally and may raise
        ``MaterializationDeferred`` carrying the published event when
        the referenced session doesn't yet exist (rollout-before-POST
        race). InMemoryEventBus doesn't touch the repo, so the
        observer still calls ``materialize_event(store_event=True)``
        separately for the test path that pairs an in-memory bus with
        a SQLite repo. Both paths share the same buffering on
        ``SessionNotFoundError``.

        Returns the published event in both the success and the
        deferred-materialization case — the event row WAS inserted
        and subscribers WERE notified; only the side-effect application
        is pending. Buffered events are replayed via
        ``_flush_pending_materialization`` once the session arrives.
        """
        published_event: Event
        try:
            published_event = await self._event_bus.publish(event)
        except MaterializationDeferred as deferred:
            # Durable bus path: event row is inserted and subscribers
            # were notified; materialization is pending. Buffer the
            # published event (with its assigned sequence) so the
            # session.updated flush hook can re-materialize it once
            # the session appears.
            if self._repository is not None:
                self._buffer_materialization(deferred.event)
            return deferred.event
        if self._repository is not None and not self._event_bus.stores_events:
            # In-memory bus path: the bus stores nothing in the repo;
            # the observer drives materialization explicitly. Tests
            # use this combo with a SQLite repo.
            try:
                self._repository.materialize_event(published_event, store_event=True)
            except SessionNotFoundError:
                if published_event.event in {"message", "run.usage"} and published_event.session_id:
                    self._buffer_materialization(published_event)
                    return published_event
                raise
        # When a session arrives (either freshly registered by the
        # rollout's session_meta event or the explicit
        # ``session.updated`` carrying its row), flush any events that
        # were buffered while the session didn't exist yet.
        if (
            published_event.event == "session.updated"
            and published_event.session_id
            and self._repository is not None
        ):
            self._flush_pending_materialization(published_event.session_id)
        return published_event

    async def freshness_tick(self) -> None:
        """Flip running sessions to idle when their last transcript event is
        older than the configured threshold. Idempotent — only emits a
        ``session.updated`` event on the first flip.

        Called periodically by :class:`TranscriptWatchService` and also
        suitable for direct invocation in tests with a fake clock.
        """
        if self._repository is None:
            return
        now = self._clock()
        for session_id, last_seen in list(self._last_event_at.items()):
            if now - last_seen <= self._idle_after:
                continue
            await self._maybe_publish_status_flip(
                session_id,
                target_status="idle",
                identity=None,
                offset=None,
            )

    async def _maybe_publish_status_flip(
        self,
        session_id: str,
        *,
        target_status: str,
        identity: TranscriptIdentity | None,
        offset: int | None,
    ) -> Event | None:
        if self._repository is None:
            return None
        try:
            session = self._repository.get_session(session_id)
        except SessionNotFoundError:
            return None
        if session.status == target_status:
            return None

        updated = session.model_copy(update={"status": target_status, "updated_at": self._clock()})
        data: dict[str, Any] = {"session": updated.model_dump(mode="json")}
        if identity is not None:
            data = {**_source_data(identity, offset=offset), **data}
        # Phase 3: ``_publish_via_bus`` handles materialization on the
        # durable path (via the bus) and on the in-memory path (via
        # the explicit ``materialize_event`` call). The session is
        # known to exist (we just fetched it), so MaterializationDeferred
        # won't fire here — but use the same wrapper for uniformity.
        published = await self._publish_via_bus(
            Event(event="session.updated", session_id=session.id, data=data)
        )
        return published

    def _resolve_active_run_id(self, event: Event) -> Event | None:
        """Stamp the session's active run id onto an event whose
        parser-emitted ``run_id`` is ``None``.

        Used for ``run.usage`` (Phase 3) and ``run.end_turn`` (Phase 4):
        both are observer-synthesized events that need to attribute
        rollout-derived activity to the harness's running Run. The
        pure parser has no repository access, so resolution happens
        here.

        Returns ``None`` when no active run exists — pure-external
        sessions or runs that have already reached terminal state.
        Falcon's worth-noting #1 on PR #11 documents the gap; the
        synthesized event is dropped rather than published as a
        phantom keyed to nothing.
        """
        if event.run_id is not None:
            return event
        if event.session_id is None or self._repository is None:
            return None
        active_run_id = self._active_run_id_for_session(event.session_id)
        if active_run_id is None:
            logger.debug(
                "Skipping %s: no active run for session=%s",
                event.event,
                event.session_id,
            )
            return None
        return event.model_copy(update={"run_id": active_run_id})

    def _active_run_id_for_session(self, session_id: str) -> str | None:
        """Return the id of the session's currently-running harness
        run, or ``None`` if no run is in ``running`` status.

        Used by ``_resolve_active_run_id`` for both ``run.usage``
        (Phase 3) and ``run.end_turn`` (Phase 4): the caller drops
        the event when this returns ``None``. Two reasons not to fall
        back to the most-recently-started non-running run:

        - ``run.usage``: applying usage to a completed run silently
          rewrites its outcome.
        - ``run.end_turn``: a stale end-turn either no-ops against an
          already-exited watchdog OR would falsely arm cleanup for an
          unrelated future run (the watchdog's session-scoped
          subscription would receive it).

        Walks ``list_runs`` in reverse so the most-recently-started
        running run wins. The per-session FIFO queue serializes the
        harness to one running run per session, so the reverse order
        is a defensive no-op in practice — but cheap.
        """
        if self._repository is None:
            return None
        try:
            runs = self._repository.list_runs(session_id)
        except SessionNotFoundError:
            return None
        for run in reversed(runs):
            if run.status == "running":
                return run.id
        return None

    def _session_exists(self, session_id: str) -> bool:
        if self._repository is None:
            return False
        has_session = getattr(self._repository, "has_session", None)
        if callable(has_session):
            return bool(has_session(session_id))
        try:
            self._repository.get_session(session_id)
        except SessionNotFoundError:
            return False
        return True

    def _buffer_materialization(self, event: Event) -> None:
        if event.session_id is None:
            return
        logger.debug(
            "Buffering observed message until session exists: session=%s event=%s",
            event.session_id,
            event.sequence,
        )
        self._pending_materialization.setdefault(event.session_id, []).append(event)

    def _flush_pending_materialization(self, session_id: str) -> None:
        if self._repository is None:
            return

        pending = self._pending_materialization.pop(session_id, [])
        still_pending: list[Event] = []
        for event in pending:
            # Buffered events have already been inserted via the bus
            # (the durable path) or are about to be inserted via the
            # ``store_event=True`` materialize (the in-memory bus path).
            # On the durable path, ``store_event=False`` avoids
            # reinserting the existing row.
            store_event = not self._event_bus.stores_events
            try:
                self._repository.materialize_event(event, store_event=store_event)
            except SessionNotFoundError:
                still_pending.append(event)

        if still_pending:
            self._pending_materialization[session_id] = still_pending


Watcher = Callable[..., AsyncIterator[Iterable[tuple[object, str]]]]


class TranscriptWatchService:
    def __init__(
        self,
        *,
        roots: Iterable[str | Path],
        observer: ExternalTranscriptObserver,
        watcher: Watcher | None = None,
        freshness_interval_seconds: float = 10.0,
    ) -> None:
        self._roots = tuple(Path(root) for root in roots)
        self._observer = observer
        if watcher is not None:
            self._watcher = watcher
        elif awatch is not None:
            self._watcher = awatch
        else:
            raise RuntimeError("watchfiles is required for live transcript watching")
        self._freshness_interval_seconds = freshness_interval_seconds

    async def watch_forever(self, *, stop_event: object | None = None) -> None:
        freshness_task = asyncio.create_task(self._freshness_loop())
        try:
            async for changes in self._watcher(*self._roots, stop_event=stop_event):
                for _change, changed_path in changes:
                    path = Path(changed_path)
                    if path.suffix == ".jsonl":
                        await self._observer.tail_file(path)
        finally:
            freshness_task.cancel()
            try:
                await freshness_task
            except asyncio.CancelledError:
                pass

    async def _freshness_loop(self) -> None:
        # Walks the observer's last-event map and flips silent sessions to
        # idle. Errors are logged and swallowed so a transient repository
        # blip can't take down the whole watch service.
        while True:
            try:
                await asyncio.sleep(self._freshness_interval_seconds)
                await self._observer.freshness_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Observer freshness tick failed")


def claude_project_dir_name(cwd: str | Path) -> str:
    return Path(cwd).expanduser().as_posix().replace("/", "-")


def claude_transcript_path(cwd: str | Path, session_uuid: str, *, home: str | Path | None = None) -> Path:
    home_path = Path(home).expanduser() if home is not None else Path.home()
    return home_path / ".claude" / "projects" / claude_project_dir_name(cwd) / f"{session_uuid}.jsonl"


def codex_transcript_path(
    *,
    year: int | str,
    month: int | str,
    day: int | str,
    timestamp: str,
    rollout_uuid: str,
    home: str | Path | None = None,
) -> Path:
    home_path = Path(home).expanduser() if home is not None else Path.home()
    return (
        home_path
        / ".codex"
        / "sessions"
        / str(year)
        / str(month).zfill(2)
        / str(day).zfill(2)
        / f"rollout-{timestamp}-{rollout_uuid}.jsonl"
    )


def external_session_id_from_claude_path(path: str | Path) -> str:
    # Canonical form: ``ses_<32hex>``. This matches the shape harness-origin
    # claude sessions use (see ``_harness_session_id_as_uuid``), so an
    # external observation and a harness spawn of the *same* claude session
    # UUID produce the *same* session id — no more duplicate session
    # records / duplicate MM channels per terminal session.
    transcript_path = Path(path)
    if transcript_path.suffix != ".jsonl" or not transcript_path.stem:
        raise ValueError(f"Not a Claude Code transcript path: {transcript_path}")
    hex_part = transcript_path.stem.replace("-", "")
    if len(hex_part) != 32 or not all(c in "0123456789abcdef" for c in hex_part.lower()):
        raise ValueError(
            f"Claude Code transcript stem is not a UUID: {transcript_path.stem!r}",
        )
    return f"ses_{hex_part.lower()}"


def external_session_id_from_codex_path(path: str | Path) -> str:
    transcript_path = Path(path)
    match = _CODEX_ROLLOUT_RE.match(transcript_path.name)
    if match is None:
        raise ValueError(f"Not a Codex rollout transcript path: {transcript_path}")
    return f"codex_{match.group('uuid')}"


def transcript_identity_from_path(path: str | Path) -> TranscriptIdentity:
    transcript_path = Path(path)
    parts = transcript_path.parts
    if ".claude" in parts and "projects" in parts:
        return TranscriptIdentity(
            backend="claude-code",
            path=transcript_path,
            session_id=external_session_id_from_claude_path(transcript_path),
        )
    if ".codex" in parts and "sessions" in parts:
        return TranscriptIdentity(
            backend="codex",
            path=transcript_path,
            session_id=external_session_id_from_codex_path(transcript_path),
        )
    raise ValueError(f"Unsupported transcript path: {transcript_path}")


def parse_transcript_line(line: str, *, identity: TranscriptIdentity, offset: int | None = None) -> list[Event]:
    stripped = line.strip()
    if not stripped:
        return []

    try:
        record = json.loads(stripped)
    except JSONDecodeError:
        logger.warning("Malformed transcript JSONL line: backend=%s path=%s", identity.backend, identity.path)
        return []

    return parse_transcript_record(record, identity=identity, offset=offset)


def parse_transcript_record(
    record: object,
    *,
    identity: TranscriptIdentity,
    offset: int | None = None,
) -> list[Event]:
    if not isinstance(record, Mapping):
        logger.warning(
            "Unsupported transcript JSONL shape: backend=%s path=%s shape=%s",
            identity.backend,
            identity.path,
            type(record).__name__,
        )
        return []

    if identity.backend == "claude-code":
        return _parse_claude_record(record, identity=identity, offset=offset)
    if identity.backend == "codex":
        return _parse_codex_record(record, identity=identity, offset=offset)

    logger.warning("Unsupported transcript backend: %s", identity.backend)
    return []


def _parse_claude_record(
    record: Mapping[str, Any],
    *,
    identity: TranscriptIdentity,
    offset: int | None,
) -> list[Event]:
    record_type = _string_value(record.get("type"))
    message = record.get("message") if isinstance(record.get("message"), Mapping) else {}
    events = _session_event_if_complete(
        identity=identity,
        cwd=_string_value(record.get("cwd")),
        model=_string_value(message.get("model")) or _string_value(record.get("model")),
        offset=offset,
    )

    role = _role_from_value(message.get("role")) or _role_from_value(record_type)
    if record_type in _IGNORED_CLAUDE_RECORD_TYPES:
        logger.debug(
            "Ignoring Claude Code transcript metadata: path=%s type=%s",
            identity.path,
            record_type,
        )
        return events

    if record_type in {"user", "assistant"} and role is not None:
        message_event = _message_event(
            identity=identity,
            role=role,
            source_type=record_type,
            model=_string_value(message.get("model")),
            blocks=_blocks_from_claude_message(message),
            offset=offset,
        )
        if message_event is not None:
            events.append(message_event)
        # Phase 3: claude's ``assistant`` record carries
        # ``message.usage`` with per-turn token counts. Emit a
        # ``run.usage`` event with ``run_id=None`` — ``publish_line``
        # resolves the active run id before publishing.
        if record_type == "assistant":
            raw_usage = message.get("usage")
            usage = parse_claude_usage(raw_usage)
            if usage is not None:
                # Snapshot rides on the same run.usage event (option A
                # from the spec; matches the existing context_window
                # pattern). ``parse_claude_context_snapshot`` returns
                # 0 for an all-zero usage block — skip emission in
                # that case so the materializer doesn't overwrite an
                # earlier real snapshot with a meaningless zero.
                snapshot = parse_claude_context_snapshot(raw_usage)
                context_used = snapshot if snapshot else None
                events.append(
                    _run_usage_event(
                        identity=identity,
                        usage=usage,
                        context_window=None,
                        context_used=context_used,
                        offset=offset,
                    )
                )
            # Phase 4: when the assistant message ends the turn
            # (``stop_reason == "end_turn"``), emit ``run.end_turn``.
            # Drives the watchdog's post-end_turn cleanup via the
            # event bus — replaces the supervisor's stdout-derived
            # detector that Phase 4 retires.
            if message.get("stop_reason") == "end_turn":
                events.append(
                    _run_end_turn_event(
                        identity=identity,
                        backend="claude-code",
                        offset=offset,
                    )
                )
        return events

    logger.warning(
        "Unsupported Claude Code transcript shape: path=%s type=%s keys=%s",
        identity.path,
        record_type,
        sorted(str(key) for key in record.keys()),
    )
    return events


def _parse_codex_record(
    record: Mapping[str, Any],
    *,
    identity: TranscriptIdentity,
    offset: int | None,
) -> list[Event]:
    record_type = _string_value(record.get("type"))
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    events = _session_event_if_complete(
        identity=identity,
        cwd=_string_value(payload.get("cwd")) or _string_value(record.get("cwd")),
        model=_string_value(payload.get("model")) or _string_value(record.get("model")),
        offset=offset,
    )

    payload_type = _string_value(payload.get("type"))

    # Phase 3: codex's ``event_msg/token_count`` carries per-turn
    # usage in ``payload.info.last_token_usage`` and the
    # session-scoped ``model_context_window``. Emit a ``run.usage``
    # event with ``run_id=None`` (resolved by ``publish_line`` against
    # the session's active harness run before publish) — codex's
    # first ``token_count`` has ``info: null``; the parser returns
    # ``(None, None)`` for that case and we skip emission.
    if record_type == "event_msg" and payload_type == "token_count":
        usage, context_window = parse_codex_token_count(payload)
        # Snapshot from total_token_usage rides on the same run.usage
        # event (option A; matches the existing context_window
        # pattern). Returns None when info is missing/null/zero — the
        # _run_usage_event omission then preserves any earlier
        # snapshot on the session.
        context_used = parse_codex_context_snapshot(payload)
        if usage is not None or context_window is not None or context_used is not None:
            events.append(
                _run_usage_event(
                    identity=identity,
                    usage=usage or Usage(),
                    context_window=context_window,
                    context_used=context_used,
                    offset=offset,
                )
            )
        return events

    # Phase 4: codex's ``event_msg/task_complete`` signals end of
    # turn. Surface as ``run.end_turn`` (run_id resolved in
    # publish_line) — drives the watchdog's post-end_turn cleanup
    # grace via the event bus, replacing the supervisor's removed
    # stdout-derived signal.
    if record_type == "event_msg" and payload_type == "task_complete":
        events.append(
            _run_end_turn_event(
                identity=identity,
                backend="codex",
                offset=offset,
            )
        )
        return events

    if record_type in _IGNORED_CODEX_RECORD_TYPES or payload_type in _IGNORED_CODEX_PAYLOAD_TYPES:
        logger.debug(
            "Ignoring Codex transcript metadata: path=%s type=%s payload_type=%s",
            identity.path,
            record_type,
            payload_type,
        )
        return events

    role = _role_from_value(payload.get("role")) or _role_from_codex_payload_type(payload_type)
    if record_type == "response_item" and payload_type == "message" and role == "user":
        logger.debug(
            "Ignoring Codex response_item user context: path=%s",
            identity.path,
        )
        return events

    if record_type == "response_item" and payload_type == "message" and role is None:
        logger.debug(
            "Ignoring unsupported Codex message role: path=%s role=%s",
            identity.path,
            payload.get("role"),
        )
        return events

    if record_type == "response_item" and payload_type == "reasoning":
        message_event = _message_event(
            identity=identity,
            role="assistant",
            source_type=payload_type,
            model=_string_value(payload.get("model")),
            blocks=_blocks_from_codex_reasoning(payload),
            offset=offset,
        )
        if message_event is not None:
            events.append(message_event)
        return events

    if record_type == "response_item" and payload_type in {"function_call", "custom_tool_call"}:
        message_event = _message_event(
            identity=identity,
            role="assistant",
            source_type=payload_type,
            model=_string_value(payload.get("model")),
            blocks=_blocks_from_codex_tool_call(payload),
            offset=offset,
        )
        if message_event is not None:
            events.append(message_event)
        return events

    if record_type == "response_item" and payload_type in {"function_call_output", "custom_tool_call_output"}:
        message_event = _message_event(
            identity=identity,
            role="user",
            source_type=payload_type,
            model=_string_value(payload.get("model")),
            blocks=_blocks_from_codex_tool_result(payload),
            offset=offset,
        )
        if message_event is not None:
            events.append(message_event)
        return events

    if record_type in {"event_msg", "response_item"} and role is not None:
        message_event = _message_event(
            identity=identity,
            role=role,
            source_type=payload_type or record_type or "unknown",
            model=_string_value(payload.get("model")),
            blocks=_blocks_from_codex_payload(payload),
            offset=offset,
        )
        if message_event is not None:
            events.append(message_event)
        return events

    if events:
        return events

    logger.warning(
        "Unsupported Codex transcript shape: path=%s type=%s payload_type=%s keys=%s",
        identity.path,
        record_type,
        payload_type,
        sorted(str(key) for key in record.keys()),
    )
    return []


def _session_event_if_complete(
    *,
    identity: TranscriptIdentity,
    cwd: str | None,
    model: str | None,
    offset: int | None,
) -> list[Event]:
    if not cwd or not model:
        return []
    # When the identity was rebound to a harness session id via
    # ``observer.bind_rollout``, the harness session already exists and
    # is owned by the orchestrator. Synthesizing an ``origin=external``
    # session.updated event would be a redundant write at materialize
    # time (the origin-downgrade guard skips it) but the event still
    # flows through the bus to bridge subscribers — who'd see the
    # harness session flip to ``origin=external``. Skip emission.
    if identity.is_rebound:
        return []

    session = Session(
        id=identity.session_id,
        backend=identity.backend,
        model=model,
        project=Project(path=cwd, name=Path(cwd).name or cwd),
        status="running",
        origin="external",
    )
    return [
        Event(
            event="session.updated",
            session_id=session.id,
            data={
                **_source_data(identity, offset=offset),
                "session": session.model_dump(mode="json"),
            },
        )
    ]


def _message_event(
    *,
    identity: TranscriptIdentity,
    role: MessageRole,
    source_type: str,
    model: str | None,
    blocks: list[MessageBlock],
    offset: int | None,
) -> Event | None:
    if not blocks:
        logger.debug(
            "Skipping observed message without normalized blocks: path=%s role=%s source_type=%s",
            identity.path,
            role,
            source_type,
        )
        return None

    message = Message(
        role=role,
        blocks=blocks,
        model=model,
    )
    data: dict[str, Any] = {
        **_source_data(identity, offset=offset),
        "message": message.model_dump(mode="json"),
        "source_type": source_type,
    }
    return Event(event="message", session_id=identity.session_id, data=data)


def _run_usage_event(
    *,
    identity: TranscriptIdentity,
    usage: Usage,
    context_window: int | None,
    context_used: int | None,
    offset: int | None,
) -> Event:
    """Build a ``run.usage`` event with ``run_id=None``.

    The parser has no repository access; ``publish_line`` resolves the
    active run id for the session and stamps it on the event before
    publishing. The event carries three sibling fields that the
    materializer applies in one pass:

    - ``usage`` (additive on Run.usage + Session.stats.tokens)
    - ``context_window`` (overwrite Session.stats.context_window;
      codex-only)
    - ``context_used`` (overwrite Session.stats.context_used; both
      backends — option A from specs/2026-05-19-context-used.md)

    Omitting an optional field from event.data preserves any prior
    value on Session.stats — only writes that observe a fresh value
    move the snapshot.
    """
    data: dict[str, Any] = {
        **_source_data(identity, offset=offset),
        "usage": usage.model_dump(mode="json"),
    }
    if context_window is not None:
        data["context_window"] = context_window
    if context_used is not None:
        data["context_used"] = context_used
    return Event(
        event="run.usage",
        session_id=identity.session_id,
        run_id=None,
        data=data,
    )


def _run_end_turn_event(
    *,
    identity: TranscriptIdentity,
    backend: BackendName,
    offset: int | None,
) -> Event:
    """Build a ``run.end_turn`` event with ``run_id=None``.

    Phase 4: published whenever the rollout signals the end of an
    assistant turn (claude ``stop_reason == "end_turn"`` or codex
    ``event_msg/task_complete``). ``publish_line`` resolves the
    active run id before publishing. The supervisor's
    ``_watch_end_turn_cleanup`` watchdog subscribes to this event
    on the bus and arms its SIGTERM grace timer when ``run_id``
    matches its own. Replaces the supervisor-side stdout detector
    that Phase 4 retires.
    """
    return Event(
        event="run.end_turn",
        session_id=identity.session_id,
        run_id=None,
        data={
            **_source_data(identity, offset=offset),
            "backend": backend,
        },
    )


def blocks_from_claude_message(message: Mapping[str, Any]) -> list[MessageBlock]:
    return _blocks_from_content(message.get("content"))


_blocks_from_claude_message = blocks_from_claude_message


def _blocks_from_codex_payload(payload: Mapping[str, Any]) -> list[MessageBlock]:
    for key in ("message", "content", "text_elements"):
        blocks = _blocks_from_content(payload.get(key))
        if blocks:
            return blocks
    return []


def _blocks_from_codex_reasoning(payload: Mapping[str, Any]) -> list[MessageBlock]:
    blocks: list[MessageBlock] = []
    for key in ("summary", "content", "text"):
        for text in _text_parts_from_content(payload.get(key)):
            blocks.append(ThinkingBlock(text=text))
    return blocks


def _blocks_from_codex_tool_call(payload: Mapping[str, Any]) -> list[MessageBlock]:
    block = _tool_use_block(payload)
    return [block] if block is not None else []


def _blocks_from_codex_tool_result(payload: Mapping[str, Any]) -> list[MessageBlock]:
    block = _tool_result_block(
        tool_use_id=_string_value(payload.get("call_id")) or _string_value(payload.get("id")),
        content=payload.get("output"),
        is_error=payload.get("is_error"),
    )
    return [block] if block is not None else []


def _blocks_from_content(content: object) -> list[MessageBlock]:
    if isinstance(content, str):
        return [TextBlock(text=content)] if content else []

    if not isinstance(content, list):
        return []

    blocks: list[MessageBlock] = []
    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append(TextBlock(text=item))
            continue

        if not isinstance(item, Mapping):
            continue

        item_type = _string_value(item.get("type"))
        if item_type in {"text", "input_text", "output_text", "summary_text"}:
            text = _string_value(item.get("text"))
            if text:
                blocks.append(TextBlock(text=text))
            continue

        if item_type == "thinking":
            text = _string_value(item.get("thinking")) or _string_value(item.get("text"))
            if text:
                blocks.append(ThinkingBlock(text=text))
            continue

        if item_type == "tool_use":
            block = _tool_use_block(item)
            if block is not None:
                blocks.append(block)
            continue

        if item_type == "tool_result":
            block = _tool_result_block(
                tool_use_id=_string_value(item.get("tool_use_id")),
                content=item.get("content"),
                is_error=item.get("is_error"),
            )
            if block is not None:
                blocks.append(block)
            continue

        if item_type in {"image", "input_image", "output_image"}:
            block = _image_block(item)
            if block is not None:
                blocks.append(block)

    return blocks


def _tool_use_block(item: Mapping[str, Any]) -> ToolUseBlock | None:
    name = _string_value(item.get("name"))
    tool_id = _string_value(item.get("id")) or _string_value(item.get("call_id"))
    if not name or not tool_id:
        return None

    input_data = _tool_input_from_value(
        item.get("input")
        if "input" in item
        else item.get("arguments")
        if "arguments" in item
        else item.get("args")
    )
    return ToolUseBlock(name=name, input=input_data, id=tool_id)


def _tool_input_from_value(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except JSONDecodeError:
            return {"input": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"input": parsed}
    return {}


def _tool_result_block(
    *,
    tool_use_id: str | None,
    content: object,
    is_error: object,
) -> ToolResultBlock | None:
    if not tool_use_id:
        return None
    return ToolResultBlock(
        tool_use_id=tool_use_id,
        content=_text_from_content(content) or "",
        is_error=is_error is True,
    )


def _image_block(item: Mapping[str, Any]) -> ImageBlock | None:
    source = item.get("source")
    if isinstance(source, Mapping):
        media_type = _string_value(source.get("media_type"))
        data = _string_value(source.get("data"))
        if media_type and data:
            return ImageBlock(media_type=media_type, data=data)

    image_url = _string_value(item.get("image_url")) or _string_value(item.get("url"))
    if image_url and image_url.startswith("data:"):
        return _image_block_from_data_url(image_url)

    return None


def _image_block_from_data_url(value: str) -> ImageBlock | None:
    header, separator, data = value.partition(",")
    if not separator or not data:
        return None
    media_prefix = "data:"
    media_type = header[len(media_prefix) :].split(";", 1)[0] if header.startswith(media_prefix) else ""
    if not media_type:
        return None
    return ImageBlock(media_type=media_type, data=data)


def _text_from_content(content: object) -> str | None:
    joined = "\n".join(_text_parts_from_content(content))
    return joined or None


def _text_parts_from_content(content: object) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = _string_value(item.get("text"))
                if text:
                    parts.append(text)
        return [part for part in parts if part]

    if isinstance(content, Mapping):
        text = _string_value(content.get("text")) or _string_value(content.get("summary"))
        return [text] if text else []

    return []



def _source_data(identity: TranscriptIdentity, *, offset: int | None) -> dict[str, Any]:
    # Phase 4: the unconditional ``origin: "external"`` stamp was
    # load-bearing under Phase 2's dual-path materialization — the
    # storage carve-outs gated on it to disambiguate "harness vs
    # external" message handling. Phase 3's single materialization
    # point retired those carve-outs; the tag is now dead metadata
    # and would actually mislead consumers (an observer-emitted
    # event for a harness-bound rollout carries the harness session
    # id but a misleading ``origin: external`` source tag).
    data: dict[str, Any] = {
        "backend": identity.backend,
        "transcript_path": str(identity.path),
    }
    if offset is not None:
        data["offset"] = offset
    return data


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _role_from_value(value: object) -> MessageRole | None:
    return value if value in {"user", "assistant"} else None


def _role_from_codex_payload_type(value: str | None) -> MessageRole | None:
    if value == "user_message":
        return "user"
    if value in {"assistant_message", "agent_message"}:
        return "assistant"
    return None
