"""pi transcript observer — parser, identity, usage, integration.

Mirrors the claude/codex coverage in test_observer.py for the pi backend.
See specs/2026-07-06-pi-transcript-observer.md.
"""

import logging
from pathlib import Path

import pytest

from agent_harness.events import InMemoryEventBus
from agent_harness.models import (
    CreateRunRequest,
    CreateSessionRequest,
    Project,
    Session,
)
from agent_harness.observer import (
    ExternalTranscriptObserver,
    external_session_id_from_pi_path,
    parse_transcript_line,
    parse_transcript_record,
    pi_session_dir_name,
    pi_transcript_path,
    transcript_identity_from_path,
)
from agent_harness.repository import InMemoryRepository

FIXTURE = Path(__file__).parent / "fixtures" / "pi_rollout_golden.jsonl"
# The golden fixture's session uuid == this harness id under
# _harness_session_id_as_uuid (verified: strip dashes, prepend ses_).
GOLDEN_UUID = "e5a93149-9a70-4aef-a189-2681b4e08525"
GOLDEN_SESSION_ID = "ses_e5a931499a704aefa1892681b4e08525"


def _pi_identity(cwd: str = "/home/me/project", uuid: str = GOLDEN_UUID):
    return transcript_identity_from_path(
        pi_transcript_path(cwd, "2026-07-06T11-25-51-562Z", uuid, home="/tmp/home")
    )


# --------------------------------------------------------------------------- #
# Path / identity helpers                                                      #
# --------------------------------------------------------------------------- #


def test_pi_transcript_path_and_dir_name() -> None:
    path = pi_transcript_path(
        "/home/claude/projects/dataverse",
        "2026-07-06T11-25-51-562Z",
        GOLDEN_UUID,
        home="/tmp/home",
    )
    assert path.as_posix() == (
        "/tmp/home/.pi/agent/sessions/--home-claude-projects-dataverse--/"
        "2026-07-06T11-25-51-562Z_e5a93149-9a70-4aef-a189-2681b4e08525.jsonl"
    )
    assert pi_session_dir_name("/home/claude/projects/dataverse") == (
        "--home-claude-projects-dataverse--"
    )


def test_external_session_id_from_pi_path_is_canonical() -> None:
    path = pi_transcript_path(
        "/home/me/project", "2026-07-06T11-25-51-562Z", GOLDEN_UUID, home="/tmp/home"
    )
    # Canonical ses_<32hex> — equals the harness id derived from the same uuid,
    # so harness-origin pi rollouts bind implicitly (no orchestrator pre-bind).
    assert external_session_id_from_pi_path(path) == GOLDEN_SESSION_ID
    assert transcript_identity_from_path(path).session_id == GOLDEN_SESSION_ID
    assert transcript_identity_from_path(path).backend == "pi"


def test_external_session_id_from_pi_path_rejects_non_uuid() -> None:
    bad = Path("/tmp/home/.pi/agent/sessions/--x--/2026-07-06T00-00-00Z_not-a-uuid.jsonl")
    with pytest.raises(ValueError):
        external_session_id_from_pi_path(bad)


# --------------------------------------------------------------------------- #
# Parser — one test per record type + edges                                   #
# --------------------------------------------------------------------------- #


def test_parser_extracts_pi_assistant_text_and_usage() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
                "model": "glm-5.2:cloud",
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 5,
                    "cacheWrite": 3,
                    "cost": {"total": 0.012},
                },
                "stopReason": "stop",
            },
        },
        identity=identity,
    )
    kinds = [e.event for e in events]
    assert kinds == ["message", "run.usage", "run.end_turn"]
    msg = events[0].data["message"]
    assert msg["role"] == "assistant"
    assert msg["model"] == "glm-5.2:cloud"
    assert msg["blocks"] == [{"type": "text", "text": "Hi there!"}]
    usage = events[1].data["usage"]
    assert usage["input"] == 100
    assert usage["output"] == 20
    assert usage["cache_read"] == 5
    assert usage["cache_creation"] == 3
    assert usage["cost_usd"] == pytest.approx(0.012)
    assert events[1].data["context_used"] == 108  # input + cacheRead + cacheWrite
    assert events[1].run_id is None  # resolved by publish_line
    assert events[2].data["backend"] == "pi"


def test_parser_extracts_pi_user_message() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {"role": "user", "content": [{"type": "text", "text": "Hi! Are you there?"}]},
        },
        identity=identity,
    )
    assert [e.event for e in events] == ["message"]
    assert events[0].data["message"]["role"] == "user"
    assert events[0].data["message"]["blocks"][0]["text"] == "Hi! Are you there?"


def test_parser_normalizes_pi_thinking_and_text() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think", "thinkingSignature": "sig"},
                    {"type": "text", "text": "the answer"},
                ],
                "stopReason": "stop",
            },
        },
        identity=identity,
    )
    blocks = events[0].data["message"]["blocks"]
    assert [b["type"] for b in blocks] == ["thinking", "text"]
    assert blocks[0]["text"] == "let me think"
    assert blocks[1]["text"] == "the answer"


def test_parser_normalizes_pi_toolcall_block() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "toolu_018Pkh",
                        "name": "name_session",
                        "arguments": {"name": "Terminal toggle"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
        identity=identity,
    )
    # toolUse does NOT end the turn — no run.end_turn.
    assert [e.event for e in events] == ["message"]
    block = events[0].data["message"]["blocks"][0]
    assert block == {
        "type": "tool_use",
        "name": "name_session",
        "input": {"name": "Terminal toggle"},
        "id": "toolu_018Pkh",
    }


def test_parser_maps_pi_toolresult_to_user_tool_result_block() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "toolu_018Pkh",
                "toolName": "name_session",
                "content": [{"type": "text", "text": "Session named: X"}],
                "isError": False,
            },
        },
        identity=identity,
    )
    # Matches codex: tool result surfaces as a role=user message.
    assert [e.event for e in events] == ["message"]
    assert events[0].data["message"]["role"] == "user"
    assert events[0].data["message"]["blocks"] == [
        {"type": "tool_result", "tool_use_id": "toolu_018Pkh", "content": "Session named: X", "is_error": False}
    ]


def test_parser_marks_pi_tool_error() -> None:
    identity = _pi_identity()
    events = parse_transcript_record(
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "toolu_err",
                "content": [{"type": "text", "text": "boom"}],
                "isError": True,
            },
        },
        identity=identity,
    )
    block = events[0].data["message"]["blocks"][0]
    assert block["is_error"] is True
    assert block["content"] == "boom"


def test_parser_only_ends_turn_on_stop_reason_stop() -> None:
    identity = _pi_identity()
    for stop_reason in ("toolUse", "aborted", "error", "length"):
        events = parse_transcript_record(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "x"}],
                    "stopReason": stop_reason,
                },
            },
            identity=identity,
        )
        assert "run.end_turn" not in [e.event for e in events], stop_reason


@pytest.mark.parametrize(
    "record",
    [
        {"type": "session", "version": 3, "id": "u", "cwd": "/repo"},
        {"type": "model_change", "provider": "ollama", "modelId": "glm-5.2:cloud"},
        {"type": "thinking_level_change", "thinkingLevel": "high"},
        {"type": "custom", "customType": "plannotator", "data": {}},
        {"type": "session_info", "name": "x"},
        {"type": "compaction"},
    ],
)
def test_parser_ignores_pi_metadata_without_warning(record, caplog) -> None:
    identity = _pi_identity()
    with caplog.at_level(logging.WARNING):
        events = parse_transcript_record(record, identity=identity)
    assert events == []
    assert caplog.records == []


def test_parser_ignores_unsupported_pi_role(caplog) -> None:
    identity = _pi_identity()
    with caplog.at_level(logging.WARNING):
        events = parse_transcript_record(
            {"type": "message", "message": {"role": "bashExecution", "content": []}},
            identity=identity,
        )
    assert events == []
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Golden fixture                                                               #
# --------------------------------------------------------------------------- #


def test_parser_golden_fixture() -> None:
    identity = _pi_identity(cwd="/home/claude/projects/dataverse")
    events = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        events.extend(parse_transcript_line(line, identity=identity))
    kinds = [e.event for e in events]
    # 5 metadata/user records + 1 assistant (message + run.usage + run.end_turn).
    assert kinds == ["message", "message", "run.usage", "run.end_turn"]
    assert events[0].data["message"]["role"] == "user"
    assert events[0].data["message"]["blocks"][0]["text"] == "Hi! - Are you there now?"
    assistant = events[1].data["message"]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["blocks"]] == ["thinking", "text"]
    assert "ready to help" in assistant["blocks"][1]["text"]


# --------------------------------------------------------------------------- #
# Observer integration                                                          #
# --------------------------------------------------------------------------- #


def _harness_pi_session(repository: InMemoryRepository, session_id: str, cwd: str) -> None:
    repository.upsert_session(
        Session(
            id=session_id,
            backend="pi",
            model="glm-5.2:cloud",
            project=Project(path=cwd, name=Path(cwd).name),
            origin="harness",
            status="running",
        )
    )


@pytest.mark.asyncio
async def test_observer_materializes_pi_assistant_message(tmp_path) -> None:
    cwd = "/home/me/project"
    transcript = pi_transcript_path(cwd, "2026-07-06T11-25-51-562Z", GOLDEN_UUID, home=tmp_path)
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                '{"type":"session","version":3,"id":"' + GOLDEN_UUID + '","cwd":"/home/me/project"}',
                '{"type":"model_change","provider":"ollama","modelId":"glm-5.2:cloud"}',
                '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hello pi"}]}}',
                '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"hi from pi"}],'
                '"model":"glm-5.2:cloud","usage":{"input":10,"output":2,"cacheRead":0,"cacheWrite":0,"cost":{"total":0}},'
                '"stopReason":"stop"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = InMemoryRepository()
    _harness_pi_session(repository, GOLDEN_SESSION_ID, cwd)
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    published = await observer.tail_file(transcript)

    kinds = [e.event for e in published]
    assert "message" in kinds
    messages = repository.list_messages(GOLDEN_SESSION_ID)
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant"]
    assert messages[1].blocks[0].text == "hi from pi"


@pytest.mark.asyncio
async def test_observer_skips_pi_message_for_unlocatable_session(tmp_path) -> None:
    # A transcript that never states its cwd can't be synthesized into a
    # session (pi resolves --session-id per project dir, so there'd be no
    # way to resume it) — the observer skips rather than orphans/buffers.
    # External pi sessions WITH a cwd are discovered and mirrored; see
    # tests/test_pi_external_sessions.py.
    cwd = "/home/me/project"
    transcript = pi_transcript_path(cwd, "2026-07-06T11-25-51-562Z", GOLDEN_UUID, home=tmp_path)
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"orphan"}],"stopReason":"stop"}}\n',
        encoding="utf-8",
    )
    repository = InMemoryRepository()  # no session upserted
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    published = await observer.tail_file(transcript)

    assert [e.event for e in published if e.event == "message"] == []


@pytest.mark.asyncio
async def test_observer_materializes_pi_usage_and_cost_into_session_stats(tmp_path) -> None:
    """With an ACTIVE running run, a pi assistant record's usage must
    resolve run_id and materialize into Session.stats (tokens + cost +
    context_used). Guards the SHOULD claim that pi carries token/cost —
    the parser-only tests never exercise the run.usage materialization
    path, and pi is single-shot so this is its only usage record.
    """
    repository = InMemoryRepository()
    session = repository.create_session(
        CreateSessionRequest(
            backend="pi",
            model="glm-5.2:cloud",
            project=Project(path="/repo", name="repo"),
        )
    )
    # create_run flips the session running; start_run makes the run the
    # active one that run.usage / run.end_turn resolve against.
    run = repository.create_run(session.id, CreateRunRequest(message="hi"))
    repository.start_run(session.id, run.id)

    raw = session.id.removeprefix("ses_")
    pi_uuid = f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"
    rollout = pi_transcript_path("/repo", "2026-07-06T11-25-51-562Z", pi_uuid, home=tmp_path)
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],'
        '"model":"glm-5.2:cloud","usage":{"input":100,"output":50,"cacheRead":30,"cacheWrite":4,'
        '"cost":{"total":0.0125}},"stopReason":"stop"}}\n',
        encoding="utf-8",
    )

    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)
    published = await observer.tail_file(rollout)

    usage_events = [e for e in published if e.event == "run.usage"]
    assert usage_events, [e.event for e in published]
    assert usage_events[0].run_id == run.id
    assert usage_events[0].data["usage"] == {
        "input": 100,
        "output": 50,
        "cache_read": 30,
        "cache_creation": 4,
        "cost_usd": 0.0125,
    }
    assert usage_events[0].data["context_used"] == 134  # input + cacheRead + cacheWrite
    # run.end_turn resolves to the active run too (stopReason == "stop").
    end_turn = [e for e in published if e.event == "run.end_turn"]
    assert end_turn and end_turn[0].run_id == run.id

    # Materialized into Session.stats (the real payoff of run.usage).
    stats = repository.get_session(session.id).stats
    assert stats.tokens["input"] == 100
    assert stats.tokens["output"] == 50
    assert stats.tokens["cache_read"] == 30
    assert stats.tokens["cache_creation"] == 4
    assert stats.cost_usd == pytest.approx(0.0125)
    assert stats.context_used == 134


@pytest.mark.asyncio
async def test_observer_pi_multi_turn_resume_in_one_file(tmp_path) -> None:
    cwd = "/home/me/project"
    transcript = pi_transcript_path(cwd, "2026-07-06T11-25-51-562Z", GOLDEN_UUID, home=tmp_path)
    transcript.parent.mkdir(parents=True)
    repository = InMemoryRepository()
    _harness_pi_session(repository, GOLDEN_SESSION_ID, cwd)
    observer = ExternalTranscriptObserver(InMemoryEventBus(), repository=repository)

    transcript.write_text(
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"turn one"}],"stopReason":"stop"}}\n',
        encoding="utf-8",
    )
    await observer.tail_file(transcript)
    # Append a second turn to the SAME file (pi --session-id resume).
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(
            '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"turn two"}],"stopReason":"stop"}}\n'
        )
    await observer.tail_file(transcript)

    texts = [m.blocks[0].text for m in repository.list_messages(GOLDEN_SESSION_ID)]
    assert texts == ["turn one", "turn two"]
