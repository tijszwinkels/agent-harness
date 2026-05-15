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
    utc_now,
)
from agent_harness.repository import InMemoryRepository, SessionNotFoundError

try:
    from watchfiles import awatch
except ImportError:  # pragma: no cover - dependency is declared, this protects embedded use.
    awatch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CODEX_ROLLOUT_RE = re.compile(
    r"^rollout-.+-(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)
_IGNORED_CLAUDE_RECORD_TYPES = {"attachment", "last-prompt", "pr-link", "queue-operation", "system"}
_IGNORED_CODEX_RECORD_TYPES = {"compacted", "session_meta"}
_IGNORED_CODEX_PAYLOAD_TYPES = {
    "agent_message",
    "assistant_message",
    "context_compacted",
    "task_complete",
    "task_started",
    "token_count",
    "web_search_call",
}


@dataclass(frozen=True, slots=True)
class TranscriptIdentity:
    backend: BackendName
    path: Path
    session_id: str


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


class ExternalTranscriptObserver:
    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        repository: InMemoryRepository | None = None,
        state: ObserverState | None = None,
        idle_after_seconds: float = DEFAULT_IDLE_AFTER_SECONDS,
        clock: Callable[[], datetime] | None = None,
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
        # Per-session last-seen-transcript-event timestamp. Drives the
        # bidirectional running ↔ idle status transitions: a tick past the
        # threshold flips silent sessions to idle; a fresh event on an idle
        # session kicks it back to running.
        self._last_event_at: dict[str, datetime] = {}

    async def tail_file(self, path: str | Path) -> list[Event]:
        transcript_path = Path(path)
        try:
            identity = transcript_identity_from_path(transcript_path)
        except ValueError:
            logger.warning("Unsupported transcript path: %s", transcript_path)
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

        resolved_identity = identity or transcript_identity_from_path(transcript_path)
        published: list[Event] = []
        for event in parse_transcript_line(line, identity=resolved_identity, offset=offset):
            published_event = await self._event_bus.publish(event)
            if self._repository is not None:
                self._materialize_or_buffer(published_event)
            published.append(published_event)
            if published_event.session_id:
                self._last_event_at[published_event.session_id] = self._clock()
                # If a fresh transcript event arrives for a session that was
                # marked idle by a previous freshness tick, flip it back to
                # running so subscribers see the resumption.
                kick = await self._maybe_publish_status_flip(
                    published_event.session_id,
                    target_status="running",
                    identity=resolved_identity,
                    offset=offset,
                )
                if kick is not None:
                    published.append(kick)
        return published

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
        event = await self._event_bus.publish(
            Event(event="session.updated", session_id=session.id, data=data)
        )
        # Materialize into the repository so subsequent ``get_session`` calls
        # reflect the new status. Skip buffering (session is known to exist).
        if self._repository is not None:
            try:
                self._repository.materialize_event(
                    event, store_event=not self._event_bus.stores_events
                )
            except SessionNotFoundError:
                pass
        return event

    def _materialize_or_buffer(self, event: Event) -> None:
        if self._repository is None:
            return

        if event.event == "message" and event.session_id and not self._session_exists(event.session_id):
            self._buffer_materialization(event)
            return

        try:
            self._repository.materialize_event(event, store_event=not self._event_bus.stores_events)
        except SessionNotFoundError:
            if event.event == "message" and event.session_id:
                self._buffer_materialization(event)
                return
            raise

        if event.event == "session.updated" and event.session_id:
            self._flush_pending_materialization(event.session_id)

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
            try:
                self._repository.materialize_event(event, store_event=not self._event_bus.stores_events)
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
    data: dict[str, Any] = {
        "backend": identity.backend,
        "origin": "external",
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
