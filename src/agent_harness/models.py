from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

BackendId = Literal["claude-code", "codex"]
LaunchMode = Literal["orchestrated", "observed"]
SessionStatus = Literal["active", "archived"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
MessageRole = Literal["system", "user", "assistant", "tool", "observer"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(HarnessModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ToolCallBlock(HarnessModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(HarnessModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1)
    output: Any
    is_error: bool = False


MessageBlock = Annotated[TextBlock | ToolCallBlock | ToolResultBlock, Field(discriminator="type")]


class Message(HarnessModel):
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    blocks: list[MessageBlock] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", blocks=[TextBlock(text=text)])


class BackendCapabilities(HarnessModel):
    launch_modes: list[LaunchMode] = Field(min_length=1)
    supports_sse: bool
    supports_transcript_observation: bool
    supports_working_directory: bool
    supports_resume: bool


class Backend(HarnessModel):
    id: BackendId
    display_name: str = Field(min_length=1)
    capabilities: BackendCapabilities


class Run(HarnessModel):
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    backend_id: BackendId
    status: RunStatus = "queued"
    working_directory: str | None = None
    command: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Session(HarnessModel):
    id: UUID = Field(default_factory=uuid4)
    backend_id: BackendId
    title: str | None = None
    status: SessionStatus = "active"
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Event(HarnessModel):
    seq: int | None = Field(default=None, ge=1)
    type: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: UUID | None = None
    session_id: UUID | None = None
    created_at: datetime = Field(default_factory=utc_now)

    def with_sequence(self, seq: int) -> "Event":
        return self.model_copy(update={"seq": seq})


class CreateSessionRequest(HarnessModel):
    backend_id: BackendId
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(HarnessModel):
    session: Session


class SessionListResponse(HarnessModel):
    sessions: list[Session]


class BackendListResponse(HarnessModel):
    backends: list[Backend]
