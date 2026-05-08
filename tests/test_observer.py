import logging

import pytest

from agent_harness.events import InMemoryEventBus
from agent_harness.observer import (
    ExternalTranscriptObserver,
    codex_transcript_path,
    external_session_id_from_codex_path,
    external_session_id_from_claude_path,
    claude_project_dir_name,
    claude_transcript_path,
    parse_transcript_record,
    parse_transcript_line,
    transcript_identity_from_path,
)


def test_claude_transcript_path_and_external_id_helpers() -> None:
    path = claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")

    assert path.as_posix() == (
        "/tmp/home/.claude/projects/-home-me-project/123e4567-e89b-12d3-a456-426614174000.jsonl"
    )
    assert claude_project_dir_name("/home/me/project") == "-home-me-project"
    assert external_session_id_from_claude_path(path) == "claude_123e4567-e89b-12d3-a456-426614174000"
    assert transcript_identity_from_path(path).session_id == "claude_123e4567-e89b-12d3-a456-426614174000"


def test_codex_transcript_path_and_external_id_helpers() -> None:
    path = codex_transcript_path(
        year=2026,
        month=5,
        day=8,
        timestamp="2026-05-08T10-30-00",
        rollout_uuid="123e4567-e89b-12d3-a456-426614174000",
        home="/tmp/home",
    )

    assert path.as_posix() == (
        "/tmp/home/.codex/sessions/2026/05/08/"
        "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    )
    assert external_session_id_from_codex_path(path) == "codex_123e4567-e89b-12d3-a456-426614174000"
    assert transcript_identity_from_path(path).session_id == "codex_123e4567-e89b-12d3-a456-426614174000"


def test_parser_logs_malformed_json_without_crashing(caplog: pytest.LogCaptureFixture) -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    with caplog.at_level(logging.WARNING):
        observations = parse_transcript_line("{not-json", identity=identity)

    assert observations == []
    assert "Malformed transcript JSONL line" in caplog.text


def test_parser_logs_unsupported_dict_shape_without_crashing(caplog: pytest.LogCaptureFixture) -> None:
    identity = transcript_identity_from_path(
        codex_transcript_path(
            year=2026,
            month=5,
            day=8,
            timestamp="2026-05-08T10-30-00",
            rollout_uuid="123e4567-e89b-12d3-a456-426614174000",
            home="/tmp/home",
        )
    )

    with caplog.at_level(logging.WARNING):
        observations = parse_transcript_record({"type": "unexpected"}, identity=identity)

    assert observations == []
    assert "Unsupported Codex transcript shape" in caplog.text


def test_parser_materializes_external_session_for_supported_claude_line() -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    observations = parse_transcript_line(
        (
            '{"type":"assistant","cwd":"/home/me/project","sessionId":"123e4567-e89b-12d3-a456-426614174000",'
            '"message":{"role":"assistant","model":"claude-opus","content":[{"type":"text","text":"done"}]}}'
        ),
        identity=identity,
    )

    assert [event.event for event in observations] == ["session.updated", "message"]
    assert observations[0].data["session"]["origin"] == "external"
    assert observations[0].session_id == identity.session_id
    assert observations[1].data["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_observer_deduplicates_by_file_offset(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"event_msg","payload":{"type":"user_message","role":"user","content":"hello"}}\n',
        encoding="utf-8",
    )
    observer = ExternalTranscriptObserver(InMemoryEventBus())

    first = await observer.tail_file(transcript)
    second = await observer.tail_file(transcript)

    assert len(first) == 1
    assert second == []


@pytest.mark.asyncio
async def test_observer_publishes_event_sequence_through_bus(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
                '{"type":"event_msg","payload":{"type":"user_message","role":"user","content":"hello"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bus = InMemoryEventBus()
    observer = ExternalTranscriptObserver(bus)

    published = await observer.tail_file(transcript)

    assert [event.sequence for event in published] == [1, 2]
    assert [event.event for event in published] == ["session.updated", "message"]
    assert published[0].data["session"]["origin"] == "external"
    assert [event.sequence for event in await bus.replay()] == [1, 2]
