from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

BackendName = Literal["claude-code", "codex"]
Origin = Literal["harness", "external"]
SessionStatus = Literal["idle", "running", "waiting_for_input", "archived"]
RunStatus = Literal["queued", "running", "completed", "failed", "interrupted"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "interrupted"]
MessageRole = Literal["user", "assistant"]
ToolMode = Literal["granular", "name-list", "none"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Project(HarnessModel):
    path: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Usage(HarnessModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_creation: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class SessionStats(HarnessModel):
    messages: int = Field(default=0, ge=0)
    tokens: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = Field(default=0, ge=0)


class TextBlock(HarnessModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ThinkingBlock(HarnessModel):
    type: Literal["thinking"] = "thinking"
    text: str = Field(min_length=1)


class ToolUseBlock(HarnessModel):
    type: Literal["tool_use"] = "tool_use"
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: new_id("tool"), min_length=1)


class ToolResultBlock(HarnessModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1)
    content: str = ""
    is_error: bool = False


class ImageBlock(HarnessModel):
    type: Literal["image"] = "image"
    media_type: str = Field(min_length=1)
    data: str = Field(min_length=1)


MessageBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock,
    Field(discriminator="type"),
]


class Message(HarnessModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    role: MessageRole
    timestamp: datetime = Field(default_factory=utc_now)
    blocks: list[MessageBlock] = Field(min_length=1)
    model: str | None = None
    run_id: str | None = None

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", blocks=[TextBlock(text=text)])


class BackendCapabilities(HarnessModel):
    fork: bool
    subagents: bool
    permission_detection: bool
    interactive_pty: bool
    stream_json: bool
    structured_output: bool
    session_id_choice: bool
    max_budget: bool
    mcp: bool
    sandbox: list[Literal["read-only", "workspace-write", "danger"]] | None
    tools: ToolMode
    interrupt_external_runs: bool = False


class Backend(HarnessModel):
    name: BackendName
    display_name: str = Field(min_length=1)
    available: bool = True
    capabilities: BackendCapabilities


class Run(HarnessModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    session_id: str = Field(min_length=1)
    status: RunStatus = "queued"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_message_id: str | None = None
    stop_reason: StopReason | None = None
    origin: Origin = "harness"
    usage: Usage = Field(default_factory=Usage)


class Session(HarnessModel):
    id: str = Field(default_factory=lambda: new_id("ses"))
    backend: BackendName
    model: str = Field(min_length=1)
    project: Project
    title: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: SessionStatus = "idle"
    origin: Origin = "harness"
    stats: SessionStats = Field(default_factory=SessionStats)


class Event(HarnessModel):
    sequence: int | None = Field(default=None, ge=1)
    event: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    def with_sequence(self, sequence: int) -> "Event":
        return self.model_copy(update={"sequence": sequence})


class CreateSessionRequest(HarnessModel):
    backend: BackendName
    model: str = Field(min_length=1)
    project: Project
    title: str | None = None


class CreateRunRequest(HarnessModel):
    message: str = Field(min_length=1)
    model: str | None = None


class CreateRunResponse(HarnessModel):
    session_id: str
    run_id: str


class DataList(HarnessModel):
    data: list[Any]
