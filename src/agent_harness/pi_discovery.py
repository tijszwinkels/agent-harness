"""Accumulate the facts needed to synthesize an external pi session.

claude and codex each stamp cwd *and* model on records the observer
already parses, so a single record is enough to synthesize an
``origin=external`` session. pi does not: its ``session`` record carries
the cwd, its ``model_change`` record carries provider + model id, and its
``message`` records carry neither (only the assistant's own
provider/model echo). Synthesis therefore needs memory across records —
this module holds it, bounded, so the observer stays a stream processor.

Facts are **sticky**: once learned, a value is never blanked by a later
record that simply doesn't repeat it. The repository's external-origin
upsert overwrites the stored Session wholesale, so an announcement
carrying ``model=None`` would wipe a model we had already discovered.

See specs/2026-09-14-external-pi-sessions.md.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

# One entry per pi transcript the observer has seen. An operator's
# ``~/.pi/agent/sessions`` accumulates a file per terminal conversation
# and the observer watches the tree recursively, so the map has to be
# capped; eviction is harmless because ``read_pi_head_facts`` can always
# re-derive an entry from the transcript itself.
DEFAULT_MAX_TRACKED_TRANSCRIPTS = 512

# How far into a transcript to look when re-deriving facts for a file the
# registry has no entry for. The ``session`` record is line 1 and
# ``model_change`` is typically line 2; the allowance covers transcripts
# that open with a few metadata records without reading a 20MB rollout.
DEFAULT_HEAD_PEEK_LINES = 64


@dataclass(frozen=True, slots=True)
class PiSessionFacts:
    """Session-level facts scavenged from a pi transcript.

    ``cwd`` is the only one that gates synthesis: pi resolves
    ``--session-id`` *within the project's session directory*, so without
    the cwd the harness cannot locate — or resume — the conversation.
    ``model``/``provider`` stay optional; a pi run with neither flag falls
    back to the CLI's own configured default.
    """

    cwd: str | None = None
    provider: str | None = None
    model: str | None = None
    # The ``session`` record's own timestamp — when the human actually
    # started this conversation. Without it a rediscovered session claims
    # to have been created the moment the harness first noticed it, which
    # reorders ``.sessions`` and misreports the session's age.
    created_at: datetime | None = None

    @property
    def is_announceable(self) -> bool:
        return bool(self.cwd)

    @property
    def qualified_model(self) -> str | None:
        """``provider/id`` when the provider is known, else the bare id.

        This is what goes into ``Session.model``, and it is the *whole*
        model representation — the harness never emits ``--provider``.
        pi only interprets a ``provider/`` prefix when ``--provider`` is
        absent (``core/model-resolver.js``: the inference is guarded by
        ``if (!provider)``), so pinning both would let a stale provider
        silently outrank a freshly-chosen ``provider/model``. One field
        cannot go stale against itself.

        Safe despite pi ids containing ``:`` (``glm-5.2:cloud``): pi
        splits a ``:`` suffix only when it is a valid thinking level.
        Verified end-to-end 2026-09-14 — ``--model review-probe/glm-5.2:cloud``
        with no ``--provider`` reached the provider as ``glm-5.2:cloud``.
        """
        if self.model is None:
            return None
        if self.provider is None:
            return self.model
        # Unconditional join — never "skip the prefix if it looks present".
        # pi model ids carry their own namespaces: openrouter's catalogue is
        # full of ids like ``openrouter/free``, and stripping the repeated
        # segment would name a model that doesn't exist. pi splits only on
        # the FIRST slash, so ``openrouter/openrouter/free`` resolves to
        # provider=openrouter, id=openrouter/free — exactly right.
        return f"{self.provider}/{self.model}"

    def merged_with(self, other: "PiSessionFacts") -> "PiSessionFacts":
        """``other``'s known values win; its unknowns leave ours intact."""
        return PiSessionFacts(
            cwd=other.cwd or self.cwd,
            provider=other.provider or self.provider,
            model=other.model or self.model,
            created_at=other.created_at or self.created_at,
        )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _timestamp(value: object) -> datetime | None:
    """Parse pi's ISO-8601 record timestamp; ``None`` on anything else."""
    text = _text(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable pi timestamp: %r", text)
        return None


def pi_facts_from_record(record: object) -> PiSessionFacts:
    """Whatever session-level facts a single pi record happens to carry.

    Tolerant by construction: anything that isn't the shape we expect
    yields empty facts rather than raising, because transcripts are
    written by a third-party CLI whose schema moves.
    """
    if not isinstance(record, Mapping):
        return PiSessionFacts()

    record_type = _text(record.get("type"))
    if record_type == "session":
        return PiSessionFacts(
            cwd=_text(record.get("cwd")),
            created_at=_timestamp(record.get("timestamp")),
        )
    if record_type == "model_change":
        return _model_facts(record.get("provider"), record.get("modelId"))
    if record_type == "message":
        # pi stamps ``provider`` and ``model`` on each assistant record.
        # That is the only fact source left for a transcript whose
        # ``model_change`` sits behind the observer's persisted offset.
        message = record.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            return PiSessionFacts()
        return _model_facts(message.get("provider"), message.get("model"))
    return PiSessionFacts()


def _model_facts(provider: object, model_id: object) -> PiSessionFacts:
    """Model facts from a record, or nothing at all.

    A provider without a model id is dropped rather than kept: on its own
    it can't name a model, and carrying it forward would re-qualify a
    model we already know — turning a stored ``provider/id`` into
    ``provider/provider/id`` on the next merge. Provider and model id are
    only ever meaningful together.
    """
    model = _text(model_id)
    if model is None:
        return PiSessionFacts()
    return PiSessionFacts(provider=_text(provider), model=model)


def read_pi_head_facts(
    path: str | Path,
    *,
    max_lines: int = DEFAULT_HEAD_PEEK_LINES,
) -> PiSessionFacts:
    """Re-derive facts by reading the first ``max_lines`` of a transcript.

    The registry is in-memory while observer offsets are persisted, so
    after a restart the observer resumes mid-file with no idea what the
    session's cwd is. Rather than drop those turns, peek the head — the
    same move ``_peek_session_meta`` makes for codex. Never raises.
    """
    facts = PiSessionFacts()
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= max_lines:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except JSONDecodeError:
                    logger.debug("pi head peek: malformed line %s in %s", index, path)
                    continue
                facts = facts.merged_with(pi_facts_from_record(record))
    except OSError:
        logger.debug("pi head peek: cannot read %s", path, exc_info=True)
    return facts


@dataclass(slots=True)
class _Entry:
    facts: PiSessionFacts = PiSessionFacts()
    announced: PiSessionFacts | None = None
    hydrated: bool = False


class PiTranscriptRegistry:
    """Bounded per-transcript store of :class:`PiSessionFacts`.

    LRU-capped: touching an entry (observe/facts/hydrate) moves it to the
    front, and the least recently touched transcript is dropped once
    ``max_entries`` is exceeded.
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
                evicted, _ = self._entries.popitem(last=False)
                logger.debug("Evicting pi transcript facts for %s", evicted)
        else:
            self._entries.move_to_end(path)
        return entry

    def observe(self, path: str | Path, record: object) -> None:
        """Fold whatever ``record`` discloses into ``path``'s facts."""
        entry = self._entry(Path(path))
        entry.facts = entry.facts.merged_with(pi_facts_from_record(record))

    def hydrate(
        self,
        path: str | Path,
        facts: PiSessionFacts,
        *,
        already_announced: bool = False,
    ) -> None:
        """Seed facts recovered outside the record stream.

        Two sources, and the difference matters. Facts read back from a
        session the repository already holds are ``already_announced`` —
        the row exists, so re-emitting it would be noise. Facts scavenged
        from a transcript's head describe a session nobody knows about
        yet, and must still be announced.

        Idempotent per entry: only the first hydration lands, so a
        transcript whose head genuinely has no cwd isn't re-read on every
        tick.
        """
        entry = self._entry(Path(path))
        if entry.hydrated:
            return
        entry.hydrated = True
        entry.facts = facts.merged_with(entry.facts)
        if already_announced:
            entry.announced = entry.facts

    def needs_hydration(self, path: str | Path) -> bool:
        entry = self._entries.get(Path(path))
        return entry is None or not entry.hydrated

    def facts(self, path: str | Path) -> PiSessionFacts:
        entry = self._entries.get(Path(path))
        return entry.facts if entry is not None else PiSessionFacts()

    def take_announcement(self, path: str | Path) -> PiSessionFacts | None:
        """Facts worth emitting a ``session.updated`` for, else ``None``.

        ``None`` until the cwd is known, and again whenever the facts are
        unchanged since the last announcement — so a re-tail, a duplicate
        watchfiles event or a long run of message records produces exactly
        one session row, not one per line.
        """
        entry = self._entries.get(Path(path))
        if entry is None or not entry.facts.is_announceable:
            return None
        if entry.announced == entry.facts:
            return None
        entry.announced = entry.facts
        return entry.facts

    def forget(self, path: str | Path) -> None:
        self._entries.pop(Path(path), None)


__all__ = [
    "DEFAULT_HEAD_PEEK_LINES",
    "DEFAULT_MAX_TRACKED_TRANSCRIPTS",
    "PiSessionFacts",
    "PiTranscriptRegistry",
    "pi_facts_from_record",
    "read_pi_head_facts",
]
