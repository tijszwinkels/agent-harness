from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

from agent_harness.events import InMemoryEventBus
from agent_harness.models import BackendName, Event, Message, MessageRole, Project, Session, TextBlock

logger = logging.getLogger(__name__)

_CODEX_ROLLOUT_RE = re.compile(
    r"^rollout-.+-(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)


@dataclass(frozen=True, slots=True)
class TranscriptIdentity:
    backend: BackendName
    path: Path
    session_id: str


@dataclass(slots=True)
class ObserverState:
    seen_offsets: dict[Path, set[int]] = field(default_factory=dict)
    next_offsets: dict[Path, int] = field(default_factory=dict)

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


class ExternalTranscriptObserver:
    def __init__(self, event_bus: InMemoryEventBus, *, state: ObserverState | None = None) -> None:
        self._event_bus = event_bus
        self._state = state or ObserverState()

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
            published.append(await self._event_bus.publish(event))
        return published


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
    transcript_path = Path(path)
    if transcript_path.suffix != ".jsonl" or not transcript_path.stem:
        raise ValueError(f"Not a Claude Code transcript path: {transcript_path}")
    return f"claude_{transcript_path.stem}"


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
    if record_type in {"user", "assistant"} and role is not None:
        events.append(
            _message_event(
                identity=identity,
                role=role,
                source_type=record_type,
                model=_string_value(message.get("model")),
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
    role = _role_from_value(payload.get("role")) or _role_from_codex_payload_type(payload_type)
    if record_type in {"event_msg", "response_item"} and role is not None:
        events.append(
            _message_event(
                identity=identity,
                role=role,
                source_type=payload_type or record_type or "unknown",
                model=_string_value(payload.get("model")),
                offset=offset,
            )
        )
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
    offset: int | None,
) -> Event:
    message = Message(
        role=role,
        blocks=[TextBlock(text=f"Observed external {role} message")],
        model=model,
    )
    data: dict[str, Any] = {
        **_source_data(identity, offset=offset),
        "message": message.model_dump(mode="json"),
        "source_type": source_type,
    }
    return Event(event="message", session_id=identity.session_id, data=data)


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
