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


# -----------------------------------------------------------------------------
# Observer session-status freshness (2026-05-15)
# -----------------------------------------------------------------------------
# External (observer-tracked) sessions get stamped ``status="running"`` at
# synthesis and never transition back. ``ExternalTranscriptObserver`` now
# tracks per-session last-event time and flips status both ways: silent for
# >threshold → idle, fresh event on an idle session → running again.


def _make_codex_transcript(tmp_path: Path) -> tuple[Path, str]:
    """Return (transcript_path, session_id) primed with a minimal codex
    rollout that already includes cwd/model so the first tail synthesizes
    a session record."""
    transcript = (
        tmp_path
        / ".codex" / "sessions" / "2026" / "05" / "08"
        / "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n',
        encoding="utf-8",
    )
    return transcript, "codex_123e4567-e89b-12d3-a456-426614174000"


@pytest.mark.asyncio
async def test_observer_freshness_tick_flips_silent_session_to_idle(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    transcript, session_id = _make_codex_transcript(tmp_path)
    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731 — small fake-clock closure
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=clock,
    )

    await observer.tail_file(transcript)
    assert repository.get_session(session_id).status == "running"

    # Advance 31s past the last transcript event and tick.
    now = now + timedelta(seconds=31)
    await observer.freshness_tick()

    assert repository.get_session(session_id).status == "idle"
    # A session.updated event must be published so SSE subscribers see the flip.
    events = await bus.replay(session_id=session_id)
    status_events = [
        e for e in events
        if e.event == "session.updated"
        and e.data.get("session", {}).get("status") == "idle"
    ]
    assert len(status_events) == 1


@pytest.mark.asyncio
async def test_observer_freshness_tick_leaves_fresh_session_running(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    transcript, session_id = _make_codex_transcript(tmp_path)
    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=clock,
    )

    await observer.tail_file(transcript)

    # Advance only 10s — well under the threshold.
    now = now + timedelta(seconds=10)
    await observer.freshness_tick()

    assert repository.get_session(session_id).status == "running"


@pytest.mark.asyncio
async def test_observer_kicks_idle_session_back_to_running_on_new_event(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    transcript, session_id = _make_codex_transcript(tmp_path)
    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=clock,
    )

    await observer.tail_file(transcript)
    # Move past threshold, tick → idle.
    now = now + timedelta(seconds=31)
    await observer.freshness_tick()
    assert repository.get_session(session_id).status == "idle"

    # Append a new transcript line — the observer should flip the session
    # back to "running" and publish a session.updated.
    with transcript.open("a", encoding="utf-8") as f:
        f.write('{"type":"event_msg","payload":{"type":"user_message","message":"second"}}\n')
    now = now + timedelta(seconds=1)
    await observer.tail_file(transcript)

    assert repository.get_session(session_id).status == "running"
    running_flips = [
        e for e in await bus.replay(session_id=session_id)
        if e.event == "session.updated"
        and e.data.get("session", {}).get("status") == "running"
    ]
    # Two: the original creation + the kick-back.
    assert len(running_flips) >= 2


@pytest.mark.asyncio
async def test_observer_freshness_tick_does_not_emit_duplicate_idle_events(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    transcript, session_id = _make_codex_transcript(tmp_path)
    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=clock,
    )

    await observer.tail_file(transcript)
    now = now + timedelta(seconds=31)
    await observer.freshness_tick()
    await observer.freshness_tick()  # second tick on already-idle session

    idle_events = [
        e for e in await bus.replay(session_id=session_id)
        if e.event == "session.updated"
        and e.data.get("session", {}).get("status") == "idle"
    ]
    assert len(idle_events) == 1, "freshness tick must be idempotent on already-idle sessions"


@pytest.mark.asyncio
async def test_observer_freshness_tick_only_flips_silent_sessions(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    # Two distinct codex rollouts, only one goes silent.
    a_dir = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08"
    a_dir.mkdir(parents=True)
    transcript_a = a_dir / "rollout-2026-05-08T10-30-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
    transcript_b = a_dir / "rollout-2026-05-08T10-30-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
    init_payload = '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
    transcript_a.write_text(init_payload, encoding="utf-8")
    transcript_b.write_text(init_payload, encoding="utf-8")
    session_a = "codex_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    session_b = "codex_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    repository = InMemoryRepository()
    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=clock,
    )

    await observer.tail_file(transcript_a)
    await observer.tail_file(transcript_b)

    # 20s later, only B receives a new line — A stays silent.
    now = now + timedelta(seconds=20)
    with transcript_b.open("a", encoding="utf-8") as f:
        f.write('{"type":"event_msg","payload":{"type":"user_message","message":"keepalive"}}\n')
    await observer.tail_file(transcript_b)

    # 20s further (40s since A's last activity, 20s since B's) — tick.
    now = now + timedelta(seconds=20)
    await observer.freshness_tick()

    assert repository.get_session(session_a).status == "idle"
    assert repository.get_session(session_b).status == "running"


@pytest.mark.asyncio
async def test_observer_seeds_last_event_at_from_existing_running_sessions() -> None:
    """A fresh observer process (post-restart) must heal pre-existing stale
    running sessions on its first freshness tick. The 2026-05-15 deployment
    of #7 left 82 records stuck on running because the observer's
    ``_last_event_at`` map only populated from NEW transcript events — old
    sessions whose transcripts hadn't been written to since restart were
    never tracked. Seed the map from the repository at construction."""
    from datetime import UTC, datetime

    repository = InMemoryRepository()
    stale_external = Session(
        id="codex_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        backend="codex",
        model="gpt-5.4",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="external",
        updated_at=datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC),  # >1 week old
    )
    repository.upsert_session(stale_external)

    # External-origin only — harness-origin sessions are managed by
    # finish_run, not the observer freshness tick.
    harness_running = Session(
        id="ses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        backend="claude-code",
        model="claude-opus-4-7",
        project=Project(path="/repo", name="repo"),
        status="running",
        origin="harness",
        updated_at=datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC),
    )
    repository.upsert_session(harness_running)

    bus = InMemoryEventBus()
    now = datetime(2026, 5, 15, 21, 36, 30, tzinfo=UTC)
    observer = ExternalTranscriptObserver(
        bus, repository=repository, idle_after_seconds=30.0, clock=lambda: now,
    )

    await observer.freshness_tick()

    assert repository.get_session(stale_external.id).status == "idle"
    assert repository.get_session(harness_running.id).status == "running"


@pytest.mark.asyncio
async def test_watch_service_runs_freshness_tick_alongside_watcher(tmp_path) -> None:
    """Integration: ``TranscriptWatchService`` must spawn the freshness loop
    while the file watcher is alive. A regression that drops the
    ``asyncio.create_task(self._freshness_loop())`` would otherwise go
    unnoticed because the unit tests call ``freshness_tick`` directly."""
    import asyncio

    transcript = tmp_path / ".codex" / "sessions" / "2026" / "05" / "08" / (
        "rollout-2026-05-08T10-30-00-eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n',
        encoding="utf-8",
    )

    tick_calls = 0

    class CountingObserver(ExternalTranscriptObserver):
        async def freshness_tick(self) -> None:  # type: ignore[override]
            nonlocal tick_calls
            tick_calls += 1
            await super().freshness_tick()

    watcher_done = asyncio.Event()

    async def slow_watcher(*_roots, **_kwargs):
        # Yield one batch so the file is ingested, then hold the loop open
        # long enough for the freshness loop to fire at least once.
        yield {("modified", str(transcript))}
        await watcher_done.wait()

    observer = CountingObserver(InMemoryEventBus(), repository=InMemoryRepository())
    service = TranscriptWatchService(
        roots=[tmp_path],
        observer=observer,
        watcher=slow_watcher,
        freshness_interval_seconds=0.05,  # ~1/200th of production interval
    )

    task = asyncio.create_task(service.watch_forever())
    try:
        # Allow the freshness loop ample wall-clock time to tick.
        await asyncio.sleep(0.25)
        assert tick_calls >= 2, f"expected ≥2 freshness ticks, got {tick_calls}"
    finally:
        watcher_done.set()
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_bind_rollout_predates_file_creation(tmp_path) -> None:
    """The orchestrator pre-binds a rollout path to a harness session id
    BEFORE the CLI subprocess has created the file. When ``tail_file``
    later sees the path, events must attach to the bound session id —
    not to the synthesized ``ses_<hex>``/``codex_<uuid>`` derived from
    the filename pattern."""
    bus = InMemoryEventBus()
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(bus, repository=repository)

    # Pre-bind a codex rollout path to a specific harness session id.
    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0010-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    harness_session_id = "ses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    observer.bind_rollout(rollout, harness_session_id)

    # Now write the file's content. tail_file should attribute events
    # to the bound id.
    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)

    assert published, "expected at least one event from a complete rollout"
    for event in published:
        assert event.session_id == harness_session_id, event


@pytest.mark.asyncio
async def test_bind_rollout_overwrites_prior_binding(tmp_path) -> None:
    bus = InMemoryEventBus()
    observer = ExternalTranscriptObserver(bus)

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0011-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )

    observer.bind_rollout(rollout, "ses_first")
    observer.bind_rollout(rollout, "ses_second")

    published = await observer.tail_file(rollout)
    for event in published:
        assert event.session_id == "ses_second", event


@pytest.mark.asyncio
async def test_bind_rollout_suppresses_synthetic_session_updated(tmp_path) -> None:
    """A bound codex rollout must NOT emit the synthetic
    ``session.updated`` (origin=external) event that the parser would
    normally produce for a fresh codex transcript. The harness session
    is already in the repository under the bound id; emitting an
    origin=external session.updated against it would travel the bus to
    bridge subscribers and confuse them, even though the materialize
    layer's origin-downgrade guard prevents the actual upsert."""
    bus = InMemoryEventBus()
    repository = InMemoryRepository()
    observer = ExternalTranscriptObserver(bus, repository=repository)

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0020-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    harness_session_id = "ses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    observer.bind_rollout(rollout, harness_session_id)
    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)

    # No synthetic session.updated for the bound path's first parse.
    assert [e.event for e in published] == ["message"]
    assert published[0].session_id == harness_session_id


@pytest.mark.asyncio
async def test_unbind_rollout_removes_mapping(tmp_path) -> None:
    """After ``unbind_rollout``, the resolver must fall through to the
    filename-pattern path on subsequent tails. This is how the
    orchestrator keeps ``_path_to_session`` bounded — one entry per
    *active* harness run, not per all runs ever."""
    bus = InMemoryEventBus()
    observer = ExternalTranscriptObserver(bus)

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0021-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    observer.bind_rollout(rollout, "ses_bound")
    observer.unbind_rollout(rollout)
    # Idempotent: a second unbind on a missing path is a no-op.
    observer.unbind_rollout(rollout)

    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )
    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0021-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event


@pytest.mark.asyncio
async def test_unbound_rollout_falls_back_to_filename_pattern(tmp_path) -> None:
    """Regression guard: when no pre-binding exists, the observer must
    keep deriving the session id from the rollout filename as today."""
    bus = InMemoryEventBus()
    observer = ExternalTranscriptObserver(bus)

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0012-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0012-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event


# --- Phase 2: codex rollout expectation registry -------------------------


def _write_codex_rollout_with_session_meta(
    base_dir: Path,
    *,
    rollout_uuid: str,
    cwd: str,
    session_meta_ts: str,
    body_lines: list[str] | None = None,
) -> Path:
    transcript = (
        base_dir / ".codex" / "sessions" / "2026" / "05" / "19"
        / f"rollout-2026-05-19T10-30-00-{rollout_uuid}.jsonl"
    )
    transcript.parent.mkdir(parents=True, exist_ok=True)
    session_meta_line = (
        '{"timestamp":"' + session_meta_ts + '","type":"session_meta",'
        '"payload":{"id":"' + rollout_uuid + '","timestamp":"' + session_meta_ts + '",'
        '"cwd":"' + cwd + '"}}'
    )
    default_body = [
        '{"type":"turn_context","payload":{"cwd":"' + cwd + '","model":"gpt-5.4"}}',
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}',
    ]
    lines = [session_meta_line] + (body_lines if body_lines is not None else default_body)
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


@pytest.mark.asyncio
async def test_expect_codex_rollout_matches_by_cwd_and_timestamp(tmp_path) -> None:
    """The orchestrator registers an expectation at codex spawn time;
    the observer matches incoming rollouts against active expectations
    by ``session_meta.cwd`` + a ±30s timestamp window."""
    from datetime import UTC, datetime

    bus = InMemoryEventBus()
    fixed_now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    observer = ExternalTranscriptObserver(bus, clock=lambda: fixed_now)

    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_harness_match")

    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0100-0000-0000-0000-000000000000",
        cwd="/repo",
        session_meta_ts="2026-05-19T10:30:05.000Z",
    )

    published = await observer.tail_file(rollout)
    assert published, "expected events to be published from the matched rollout"
    for event in published:
        assert event.session_id == "ses_harness_match", event


@pytest.mark.asyncio
async def test_expect_codex_rollout_falls_through_when_cwd_differs(tmp_path) -> None:
    """Expectation for cwd A; rollout under cwd B. The observer must
    fall back to the filename-pattern path (external codex_<uuid> row),
    leaving the expectation unconsumed for the real rollout that may
    still arrive."""
    from datetime import UTC, datetime

    bus = InMemoryEventBus()
    fixed_now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    observer = ExternalTranscriptObserver(bus, clock=lambda: fixed_now)
    observer.expect_codex_rollout(cwd=Path("/repo-a"), session_id="ses_harness_a")

    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0101-0000-0000-0000-000000000000",
        cwd="/repo-b",
        session_meta_ts="2026-05-19T10:30:05.000Z",
    )

    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0101-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event


@pytest.mark.asyncio
async def test_expect_codex_rollout_falls_through_when_timestamp_outside_window(tmp_path) -> None:
    """Expectation registered now; rollout's session_meta timestamp is
    far in the future (well past the ±30s window). Falls through."""
    from datetime import UTC, datetime

    bus = InMemoryEventBus()
    fixed_now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    observer = ExternalTranscriptObserver(bus, clock=lambda: fixed_now)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_harness_stale")

    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0102-0000-0000-0000-000000000000",
        cwd="/repo",
        # 5 minutes after the expectation was registered — outside the
        # 30s match window.
        session_meta_ts="2026-05-19T10:35:00.000Z",
    )

    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0102-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event


@pytest.mark.asyncio
async def test_expect_codex_rollout_closest_timestamp_wins(tmp_path) -> None:
    """Two expectations with the same cwd, registered a few seconds
    apart; one rollout arrives whose session_meta timestamp is closer
    to the second expectation. The second one must win."""
    from datetime import UTC, datetime, timedelta

    bus = InMemoryEventBus()
    base = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    now = base

    def clock() -> datetime:
        return now

    observer = ExternalTranscriptObserver(bus, clock=clock)

    # First spawn at t=0
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_first")
    # Second spawn at t=+10s
    now = base + timedelta(seconds=10)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_second")

    # Rollout arrives with session_meta ts at t=+11s — much closer to
    # the second expectation than the first.
    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0103-0000-0000-0000-000000000000",
        cwd="/repo",
        session_meta_ts="2026-05-19T10:30:11.000Z",
    )

    published = await observer.tail_file(rollout)
    assert published
    for event in published:
        assert event.session_id == "ses_second", event


@pytest.mark.asyncio
async def test_expect_codex_rollout_equidistant_tiebreaker(tmp_path) -> None:
    """When two expectations sit equidistant from the rollout's
    session_meta timestamp, the earliest-registered one wins.

    Python's stable sort gives us this for free: in
    ``_find_matching_expectation`` we sort by absolute delta, and ties
    preserve insertion order. This test locks the contract in so a
    future refactor (e.g. switching to a min-heap) can't silently
    flip the tiebreaker behavior — which would steal codex rollouts
    from the first spawn in a back-to-back same-cwd race.
    """
    from datetime import UTC, datetime, timedelta

    bus = InMemoryEventBus()
    base = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    now = base

    def clock() -> datetime:
        return now

    observer = ExternalTranscriptObserver(bus, clock=clock)

    # Both expectations registered at the SAME instant. (Same clock
    # value; no advance between calls.)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_earliest")
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_later")

    # Rollout's session_meta ts is exactly the registration instant —
    # both expectations are equidistant (delta = 0).
    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0108-0000-0000-0000-000000000000",
        cwd="/repo",
        session_meta_ts="2026-05-19T10:30:00.000Z",
    )

    published = await observer.tail_file(rollout)
    assert published
    for event in published:
        assert event.session_id == "ses_earliest", event


@pytest.mark.asyncio
async def test_expectation_expires_after_ttl(tmp_path) -> None:
    """An expectation that's been sitting for longer than the TTL must
    NOT match a freshly-arriving rollout. Falls through to filename."""
    from datetime import UTC, datetime, timedelta

    bus = InMemoryEventBus()
    base = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    now = base

    def clock() -> datetime:
        return now

    observer = ExternalTranscriptObserver(
        bus, clock=clock, expectation_ttl_seconds=10.0
    )
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_expired")

    # 60s later — well past the 10s TTL.
    now = base + timedelta(seconds=60)
    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0104-0000-0000-0000-000000000000",
        cwd="/repo",
        session_meta_ts="2026-05-19T10:31:00.000Z",
    )

    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0104-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event


@pytest.mark.asyncio
async def test_expectation_matches_after_partial_ttl_elapsed(tmp_path) -> None:
    """Regression guard for the TTL/window asymmetry. The expectation
    TTL (60s default) must be strictly greater than the timestamp
    match window (30s) — otherwise a slow codex spawn whose
    ``session_meta.timestamp`` is still within the window from
    ``registered_at`` would find the expectation already evicted,
    silently fall through to filename-pattern, and re-instate the
    Heron-PR-#12 dupe-session symptom.

    Scenario: expectation registered at t=0; rollout flushed at t=25s
    with ``session_meta.timestamp`` matching the registration time.
    25s is past the (insufficient) old 10s TTL but well inside the
    timestamp-window — match must succeed under Phase 2's 60s TTL.
    """
    from datetime import UTC, datetime, timedelta

    bus = InMemoryEventBus()
    base = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    now = base

    def clock() -> datetime:
        return now

    # Use the production default TTL (60s) — this is the contract
    # under test.
    observer = ExternalTranscriptObserver(bus, clock=clock)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_slow_codex")

    # 25s later: slow codex spawn has finally flushed session_meta.
    # Within the 60s TTL; the rollout's session_meta timestamp matches
    # the original registration time (well within the 30s window).
    now = base + timedelta(seconds=25)
    rollout = _write_codex_rollout_with_session_meta(
        tmp_path,
        rollout_uuid="019e0107-0000-0000-0000-000000000000",
        cwd="/repo",
        session_meta_ts="2026-05-19T10:30:00.000Z",
    )

    published = await observer.tail_file(rollout)
    assert published
    for event in published:
        assert event.session_id == "ses_slow_codex", event


@pytest.mark.asyncio
async def test_session_meta_peek_returns_none_for_partial_flush(tmp_path) -> None:
    """First line of the rollout has no trailing newline yet (codex is
    still writing). The observer must not match an expectation against
    incomplete data — it returns no events and waits for the next
    tail_file tick. The expectation stays alive for that retry."""
    from datetime import UTC, datetime

    bus = InMemoryEventBus()
    fixed_now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    observer = ExternalTranscriptObserver(bus, clock=lambda: fixed_now)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_partial")

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0105-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    # First line is partial (no trailing newline).
    rollout.write_text(
        '{"timestamp":"2026-05-19T10:30:00Z","type":"session_meta","payload":'
        '{"id":"019e0105-0000-0000-0000-000000000000","cwd":"/re',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)
    assert published == []

    # Now the writer flushes the rest. tail_file picks up where it left
    # off; the expectation should still match.
    rollout.write_text(
        '{"timestamp":"2026-05-19T10:30:00Z","type":"session_meta","payload":'
        '{"id":"019e0105-0000-0000-0000-000000000000",'
        '"timestamp":"2026-05-19T10:30:00Z","cwd":"/repo"}}\n'
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)
    assert published
    for event in published:
        assert event.session_id == "ses_partial", event


@pytest.mark.asyncio
async def test_session_meta_peek_skips_non_session_meta_first_line(tmp_path) -> None:
    """When the first complete line is not a ``session_meta`` record,
    the observer can't extract cwd/timestamp from it, so the
    expectation can't be matched. Behavior falls back to filename."""
    from datetime import UTC, datetime

    bus = InMemoryEventBus()
    fixed_now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=UTC)
    observer = ExternalTranscriptObserver(bus, clock=lambda: fixed_now)
    observer.expect_codex_rollout(cwd=Path("/repo"), session_id="ses_no_meta")

    rollout = (
        tmp_path / ".codex" / "sessions" / "2026" / "05" / "19"
        / "rollout-2026-05-19T10-30-00-019e0106-0000-0000-0000-000000000000.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        '{"type":"turn_context","payload":{"cwd":"/repo","model":"gpt-5.4"}}\n'
        '{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}\n',
        encoding="utf-8",
    )

    published = await observer.tail_file(rollout)
    expected_id = "codex_019e0106-0000-0000-0000-000000000000"
    assert published
    for event in published:
        assert event.session_id == expected_id, event
