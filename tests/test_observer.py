import logging
from pathlib import Path

import pytest

from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.observer import (
    ExternalTranscriptObserver,
    ObserverState,
    TranscriptWatchService,
    codex_transcript_path,
    external_session_id_from_codex_path,
    external_session_id_from_claude_path,
    claude_project_dir_name,
    claude_transcript_path,
    parse_transcript_record,
    parse_transcript_line,
    transcript_identity_from_path,
)
from agent_harness.models import Event, Message, Project, Session
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository


def test_claude_transcript_path_and_external_id_helpers() -> None:
    path = claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")

    assert path.as_posix() == (
        "/tmp/home/.claude/projects/-home-me-project/123e4567-e89b-12d3-a456-426614174000.jsonl"
    )
    assert claude_project_dir_name("/home/me/project") == "-home-me-project"
    # Canonical form: ses_<32hex>. Same shape as harness-origin session ids,
    # so the external observer and harness-spawn paths produce equivalent ids
    # for the same underlying claude session UUID (no more dual session records
    # / duplicate MM channels for one terminal claude session).
    assert external_session_id_from_claude_path(path) == "ses_123e4567e89b12d3a456426614174000"
    assert transcript_identity_from_path(path).session_id == "ses_123e4567e89b12d3a456426614174000"


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
    assert observations[1].data["message"]["blocks"][0]["text"] == "done"


def test_parser_extracts_text_from_supported_claude_user_line() -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    observations = parse_transcript_line(
        (
            '{"type":"user","cwd":"/home/me/project",'
            '"message":{"role":"user","content":"hello from claude"}}'
        ),
        identity=identity,
    )

    assert observations[0].data["message"]["blocks"][0]["text"] == "hello from claude"


def test_parser_normalizes_claude_content_blocks() -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    events = parse_transcript_record(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "checking constraints"},
                    {"type": "text", "text": "running tests"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest"}},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "aW1n"},
                    },
                ],
            },
        },
        identity=identity,
    )

    blocks = events[0].data["message"]["blocks"]
    assert [block["type"] for block in blocks] == ["thinking", "text", "tool_use", "image"]
    assert blocks[0]["text"] == "checking constraints"
    assert blocks[2]["id"] == "toolu_1"
    assert blocks[2]["input"] == {"command": "pytest"}
    assert blocks[3]["media_type"] == "image/png"
    assert blocks[3]["data"] == "aW1n"


def test_parser_normalizes_claude_tool_result_blocks() -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    events = parse_transcript_record(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "ok"}],
                        "is_error": False,
                    }
                ],
            },
        },
        identity=identity,
    )

    blocks = events[0].data["message"]["blocks"]
    assert blocks == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok", "is_error": False}]


def test_parser_ignores_known_claude_metadata_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    identity = transcript_identity_from_path(
        claude_transcript_path("/home/me/project", "123e4567-e89b-12d3-a456-426614174000", home="/tmp/home")
    )

    with caplog.at_level(logging.WARNING):
        events = [
            *parse_transcript_record({"type": "attachment", "cwd": "/repo"}, identity=identity),
            *parse_transcript_record({"type": "last-prompt", "lastPrompt": "hello"}, identity=identity),
            *parse_transcript_record({"type": "pr-link", "prUrl": "https://example.test/pr/1"}, identity=identity),
            *parse_transcript_record({"type": "queue-operation", "operation": "push"}, identity=identity),
            *parse_transcript_record({"type": "system", "subtype": "hook"}, identity=identity),
        ]

    assert events == []
    assert caplog.text == ""


@pytest.mark.asyncio
async def test_observer_deduplicates_by_file_offset(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n',
        encoding="utf-8",
    )
    observer = ExternalTranscriptObserver(InMemoryEventBus())

    first = await observer.tail_file(transcript)
    second = await observer.tail_file(transcript)

    assert len(first) == 1
    assert second == []


@pytest.mark.asyncio
async def test_observer_holds_offset_until_partial_line_completes(tmp_path) -> None:
    """Regression: ``tail_file`` used to advance ``next_offset`` past a
    line that was only partially flushed by the writer, which caused the
    next read to start mid-record and lose the assistant turn.

    Now: a line without a trailing newline is treated as incomplete; the
    offset is held until the rest of the line (with newline) lands, and
    the complete line is then published exactly once.
    """
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    # First flush: one complete record + the first half of a second record.
    full_first = '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n'
    partial_second = '{"type":"event_msg","payload":{"type":"user_message","message":"hal'
    transcript.write_bytes((full_first + partial_second).encode("utf-8"))
    observer = ExternalTranscriptObserver(InMemoryEventBus())

    first = await observer.tail_file(transcript)
    # Only the complete record is published; the partial tail is held.
    assert len(first) == 1
    assert first[0].event == "message"

    # Writer flushes the rest of the second record with a final newline.
    with transcript.open("ab") as f:
        f.write(b'f-finished"}}\n')

    second = await observer.tail_file(transcript)
    # The second record is delivered exactly once, with its content intact.
    assert len(second) == 1, "partial line was skipped or duplicated"
    assert second[0].event == "message"
    assert second[0].data["message"]["blocks"][0]["text"] == "half-finished"


@pytest.mark.asyncio
async def test_observer_publishes_event_sequence_through_bus(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
                '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
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


@pytest.mark.asyncio
async def test_observer_materializes_external_session_and_message(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
                '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    await observer.tail_file(transcript)

    session_id = "codex_123e4567-e89b-12d3-a456-426614174000"
    session = repository.get_session(session_id)
    messages = repository.list_messages(session_id)
    assert session.origin == "external"
    assert session.backend == "codex"
    assert messages[0].role == "user"


@pytest.mark.asyncio
async def test_observer_deduplicates_codex_event_echoes_and_repository_echoes(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
                '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
                '{"type":"event_msg","payload":{"type":"agent_message","message":"hi"}}',
                '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = InMemoryRepository()
    session_id = "codex_123e4567-e89b-12d3-a456-426614174000"
    repository.upsert_session(
        Session(
            id=session_id,
            backend="codex",
            model="gpt-5.4",
            project={"path": "/repo", "name": "repo"},
            origin="external",
        )
    )
    repository.add_message(session_id, Message.user("hello"))
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    await observer.tail_file(transcript)

    messages = repository.list_messages(session_id)
    assert [(message.role, message.blocks[0].text) for message in messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


@pytest.mark.asyncio
async def test_observer_buffers_messages_until_external_session_exists(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    await observer.tail_file(transcript)

    messages = repository.list_messages("codex_123e4567-e89b-12d3-a456-426614174000")
    assert [message.role for message in messages] == ["user"]
    assert messages[0].blocks[0].text == "hello"


@pytest.mark.asyncio
async def test_durable_observer_buffers_messages_until_external_session_exists(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
                '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = open_sqlite_repository(tmp_path / "harness.db")
    observer = ExternalTranscriptObserver(DurableEventBus(repository), repository=repository)

    try:
        await observer.tail_file(transcript)

        messages = repository.list_messages("codex_123e4567-e89b-12d3-a456-426614174000")
        assert [message.role for message in messages] == ["user"]
        assert messages[0].blocks[0].text == "hello"
    finally:
        repository.close()


def test_parser_skips_message_records_without_text() -> None:
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

    events = parse_transcript_record(
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": []}},
        identity=identity,
    )

    assert events == []


def test_parser_does_not_emit_observed_external_placeholder_for_no_text_records() -> None:
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

    events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_image", "image_url": "/tmp/not-read.png"}],
            },
        },
        identity=identity,
    )

    assert events == []
    assert "Observed external" not in repr(events)


def test_parser_ignores_known_codex_metadata_without_warning(caplog: pytest.LogCaptureFixture) -> None:
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
        events = [
            *parse_transcript_record({"type": "session_meta", "payload": {"cwd": "/repo"}}, identity=identity),
            *parse_transcript_record(
                {"type": "event_msg", "payload": {"type": "agent_message", "message": "hello"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn_1"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "response_item", "payload": {"type": "custom_tool_call_output", "output": "..."}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "compacted", "payload": {"message": "summary"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "event_msg", "payload": {"type": "context_compacted"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "event_msg", "payload": {"type": "task_complete"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "response_item", "payload": {"type": "web_search_call", "id": "ws_1"}},
                identity=identity,
            ),
            *parse_transcript_record(
                {"type": "response_item", "payload": {"type": "message", "role": "developer", "content": []}},
                identity=identity,
            ),
        ]

    assert events == []
    assert caplog.text == ""


def test_parser_extracts_text_from_supported_codex_payloads() -> None:
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

    user_events = parse_transcript_record(
        {"type": "event_msg", "payload": {"type": "user_message", "message": "hello from codex"}},
        identity=identity,
    )
    ignored_context_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello from codex"}],
            },
        },
        identity=identity,
    )
    assistant_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "assistant reply"}],
            },
        },
        identity=identity,
    )

    assert ignored_context_events == []
    assert user_events[0].data["message"]["blocks"][0]["text"] == "hello from codex"
    assert assistant_events[0].data["message"]["blocks"][0]["text"] == "assistant reply"


def test_parser_uses_codex_event_user_messages_and_response_item_assistant_messages() -> None:
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

    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "hi"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        },
    ]

    messages = [
        event.data["message"]
        for record in records
        for event in parse_transcript_record(record, identity=identity)
        if event.event == "message"
    ]

    assert [(message["role"], message["blocks"][0]["text"]) for message in messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


def test_parser_normalizes_codex_reasoning_and_tool_items() -> None:
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

    reasoning_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "considered repo state"}],
            },
        },
        identity=identity,
    )
    call_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": "{\"cmd\":\"pytest\"}",
            },
        },
        identity=identity,
    )
    result_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call_1", "output": "passed"},
        },
        identity=identity,
    )

    assert reasoning_events[0].data["message"]["blocks"] == [
        {"type": "thinking", "text": "considered repo state"}
    ]
    assert call_events[0].data["message"]["blocks"] == [
        {"type": "tool_use", "name": "shell", "input": {"cmd": "pytest"}, "id": "call_1"}
    ]
    assert result_events[0].data["message"]["blocks"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "passed", "is_error": False}
    ]


def test_parser_normalizes_codex_custom_tool_items() -> None:
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

    call_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call_custom",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            },
        },
        identity=identity,
    )
    result_events = parse_transcript_record(
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_custom",
                "output": [{"type": "output_text", "text": "patched"}],
            },
        },
        identity=identity,
    )

    assert call_events[0].data["message"]["blocks"] == [
        {"type": "tool_use", "name": "apply_patch", "input": {"input": "*** Begin Patch"}, "id": "call_custom"}
    ]
    assert result_events[0].data["message"]["blocks"] == [
        {"type": "tool_result", "tool_use_id": "call_custom", "content": "patched", "is_error": False}
    ]


@pytest.mark.asyncio
async def test_watch_service_uses_changed_jsonl_paths(tmp_path) -> None:
    path = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = path / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n',
        encoding="utf-8",
    )

    async def fake_watcher(*_roots, **_kwargs):
        yield {("modified", str(transcript)), ("modified", str(path / "ignore.txt"))}

    bus = InMemoryEventBus()
    service = TranscriptWatchService(
        roots=[tmp_path],
        observer=ExternalTranscriptObserver(bus),
        watcher=fake_watcher,
    )

    await service.watch_forever()

    assert [event.event for event in await bus.replay()] == ["session.updated"]


def test_inmemory_repo_materialize_preserves_harness_origin() -> None:
    """Mirror of the storage.py test for the in-memory path: observer's
    session.updated event must not downgrade a harness-origin session to
    external (otherwise the bridge replaces it on the next MM post)."""
    repository = InMemoryRepository()
    from agent_harness.models import CreateSessionRequest

    harness_session = repository.create_session(
        CreateSessionRequest(
            backend="claude-code",
            model="claude-opus-4-7",
            project=Project(path="/repo", name="repo"),
            bypass_permissions=True,
        )
    )
    assert harness_session.origin == "harness"

    observer_payload = harness_session.model_copy(
        update={"origin": "external", "bypass_permissions": False}
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=harness_session.id,
            data={"session": observer_payload.model_dump(mode="json")},
        )
    )

    after = repository.get_session(harness_session.id)
    assert after.origin == "harness"
    assert after.bypass_permissions is True


def test_inmemory_repo_materialize_creates_when_absent() -> None:
    repository = InMemoryRepository()
    external = Session(
        id="ses_3eb0e45b9d724deabdc3b472e0c4c2fc",
        backend="claude-code",
        model="claude-opus-4-7",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=external.id,
            data={"session": external.model_dump(mode="json")},
        )
    )
    after = repository.get_session(external.id)
    assert after.origin == "external"


@pytest.mark.asyncio
async def test_observer_offsets_survive_repository_reopen(tmp_path) -> None:
    """Root cause of the 2026-05-15 MM flood: ``ObserverState`` was
    in-memory only, so after a harness restart the observer re-read every
    transcript from offset 0 and re-published every line into the durable
    event bus. With offsets persisted via the repository, a re-opened
    observer must skip already-processed lines."""
    transcript_dir = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    transcript = transcript_dir / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"world"}}\n',
        encoding="utf-8",
    )

    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    bus = DurableEventBus(repository)
    observer = ExternalTranscriptObserver(bus, repository=repository)

    first_run = await observer.tail_file(transcript)
    assert [event.event for event in first_run] == ["session.updated", "message", "message"]
    repository.close()

    reopened = open_sqlite_repository(db_path)
    try:
        reopened_bus = DurableEventBus(reopened)
        # Fresh observer — fresh in-memory state. Without persistence this
        # would re-emit both transcript lines as new bus events.
        reopened_observer = ExternalTranscriptObserver(reopened_bus, repository=reopened)
        second_run = await reopened_observer.tail_file(transcript)
        assert second_run == [], "observer re-emitted events after restart"

        # Appending a *new* line must still be picked up — offsets shouldn't
        # block legitimate tail forward.
        with transcript.open("a", encoding="utf-8") as f:
            f.write('{"type":"event_msg","payload":{"type":"user_message","message":"third"}}\n')
        forward = await reopened_observer.tail_file(transcript)
        assert [event.event for event in forward] == ["message"]
    finally:
        reopened.close()


def test_observer_state_from_store_loads_existing_offsets() -> None:
    """``ObserverState.from_store`` primes ``next_offsets`` from the store
    so a freshly-constructed state behaves as if the previous lifetime had
    already advanced to the persisted positions."""
    class FakeStore:
        def __init__(self) -> None:
            self.writes: list[tuple[str, int]] = []

        def get_observer_offsets(self) -> dict[str, int]:
            return {"/transcripts/a.jsonl": 128, "/transcripts/b.jsonl": 256}

        def set_observer_offset(self, path: str, next_offset: int) -> None:
            self.writes.append((path, next_offset))

    store = FakeStore()
    state = ObserverState.from_store(store)

    assert state.next_offset(Path("/transcripts/a.jsonl")) == 128
    assert state.next_offset(Path("/transcripts/b.jsonl")) == 256
    assert state.next_offset(Path("/transcripts/unknown.jsonl")) == 0

    state.set_next_offset(Path("/transcripts/a.jsonl"), 512)
    assert state.next_offset(Path("/transcripts/a.jsonl")) == 512
    assert ("/transcripts/a.jsonl", 512) in store.writes


def test_inmemory_repo_materialize_allows_updates_to_external() -> None:
    repository = InMemoryRepository()
    initial = Session(
        id="ses_3eb0e45b9d724deabdc3b472e0c4c2fc",
        backend="claude-code",
        model="claude-opus-4-6",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
    )
    repository.materialize_event(
        Event(
            sequence=1,
            event="session.updated",
            session_id=initial.id,
            data={"session": initial.model_dump(mode="json")},
        )
    )
    repository.materialize_event(
        Event(
            sequence=2,
            event="session.updated",
            session_id=initial.id,
            data={
                "session": initial.model_copy(
                    update={"model": "claude-opus-4-7"}
                ).model_dump(mode="json")
            },
        )
    )
    after = repository.get_session(initial.id)
    assert after.model == "claude-opus-4-7"
    assert after.origin == "external"
