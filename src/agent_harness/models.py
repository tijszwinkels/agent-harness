from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

BackendName = Literal["claude-code", "codex", "pi"]
Origin = Literal["harness", "external"]
SessionStatus = Literal["idle", "running", "waiting_for_input", "archived"]
RunStatus = Literal["queued", "running", "completed", "failed", "interrupted"]
RUN_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "interrupted"})
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
    tokens: dict[str, Any] = Field(
        default_factory=dict,
        description="Sum of input, output, cache_read, and cache_creation token usage across runs in the session.",
    )
    cost_usd: float = Field(default=0, ge=0)
    # Codex-only: latest ``model_context_window`` observed in a
    # ``token_count`` payload. ``None`` until first observed (e.g.
    # claude sessions never set it). The observer updates this each
    # time the rollout carries a fresh context-window value.
    context_window: int | None = Field(default=None, ge=1)
    # Per-session SNAPSHOT of currently-loaded context tokens
    # (overwrite-not-sum). Distinct from cumulative ``tokens``:
    # for long claude tool-use loops, ``cache_read`` can exceed 10M
    # while the actual loaded context is bounded by ``context_window``.
    # Sourced from codex ``info.total_token_usage.total_tokens`` and
    # claude ``message.usage.input + cache_creation + cache_read``.
    # ``None`` until the first observation lands; decreases (after
    # context compaction) are valid.
    context_used: int | None = Field(default=None, ge=0)


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
    # True when the backend CLI accepts a per-invocation reasoning-effort
    # level (see ``Session.effort``). All three do today, each spelling it
    # differently; declared per backend rather than assumed by callers so a
    # future backend without one has to say so explicitly.
    effort: bool
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
    # Optional: ``None`` means "use the backend CLI's own configured default".
    # pi callers omit the model; the command builder then omits ``--model`` so
    # the CLI falls back. A given value must still be non-empty.
    model: str | None = Field(default=None, min_length=1)
    # Reasoning/thinking level for every run in this session. ``None`` means
    # "emit no flag", so each backend CLI falls back to its own configured
    # default. Free-form and deliberately unvalidated here, exactly like
    # ``model`` — the caller owns the value space (``low|medium|high|xhigh|max``
    # is valid on all three backends; each CLI spells the flag differently, see
    # ``orchestrator._effort_flag``). Mutable mid-session via
    # ``PATCH /v1/sessions/{id}``: argv is rebuilt per run, so a change lands on
    # the next turn without recreating the session.
    effort: str | None = Field(default=None, min_length=1)
    project: Project
    title: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: SessionStatus = "idle"
    origin: Origin = "harness"
    stats: SessionStats = Field(default_factory=SessionStats)
    # When True, builders pass the backend's permission-bypass flag
    # (`--dangerously-skip-permissions` for claude-code,
    # `--dangerously-bypass-approvals-and-sandbox` for codex).
    bypass_permissions: bool = False
    # Codex rollout UUID used by CodexCommandBuilder to select
    # ``codex exec resume <uuid>`` over a fresh ``codex exec``. Set
    # once by the observer when it extracts the UUID from a bound
    # rollout's filename (harness origin) or by the startup backfill
    # (external origin — UUID is encoded in ``codex_<uuid>`` ids).
    # ``None`` for non-codex sessions and for codex sessions whose
    # first run hasn't completed binding yet. The field is mutated
    # exactly once; subsequent observations of the same rollout leave
    # it alone. Distinct from the retired ``codex_internal_id``
    # diagnostic field (PR #12 / Phase 4) — this one is the live
    # resume key.
    codex_resume_id: str | None = None
    # Parent session id this session was forked from
    # (``POST /v1/sessions/{id}/forks``). ``None`` for non-fork sessions.
    # Set once at fork creation; the command builder consumes it on the
    # first run to resume the parent's conversation into this child's own
    # transcript (claude ``--fork-session``, pi ``--fork``). After the first
    # run the child has its own transcript and resumes itself normally, so
    # this field is thereafter lineage/observability only.
    forked_from: str | None = None

    @classmethod
    def forked_child(cls, parent: "Session", *, title: str | None = None) -> "Session":
        """A new harness-owned child that forks ``parent``.

        Inherits the parent's backend / model / effort / project /
        bypass_permissions, records ``forked_from``, and keeps the parent's
        title unless overridden. Always ``origin="harness"`` — the fork is
        harness-owned even when the parent is an observed (external) session.
        """
        return cls(
            backend=parent.backend,
            model=parent.model,
            effort=parent.effort,
            project=parent.project.model_copy(deep=True),
            title=title if title is not None else parent.title,
            bypass_permissions=parent.bypass_permissions,
            origin="harness",
            forked_from=parent.id,
        )


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
    # Optional: omit to let the backend CLI use its own configured default
    # (pi callers rely on this). A given value must be non-empty.
    model: str | None = Field(default=None, min_length=1)
    # Optional reasoning/thinking level; omit to let the backend CLI use its
    # own configured default. Unvalidated free-form, like ``model``.
    effort: str | None = Field(default=None, min_length=1)
    project: Project
    title: str | None = None
    bypass_permissions: bool = False


class PatchSessionRequest(HarnessModel):
    # Patch payload for ``PATCH /v1/sessions/{id}``. Only the listed fields
    # are user-mutable; everything else on Session is either derived
    # (status, stats, updated_at) or immutable (id, backend, origin,
    # created_at). Unset fields are left untouched. Explicit ``null`` is
    # rejected at the route layer to avoid a silent "clear field" path
    # that no current consumer wants — omit the field instead.
    title: str | None = Field(default=None, min_length=1)
    # Changing the reasoning level must NOT cost the conversation: the command
    # builder rebuilds argv per run, so the next turn picks the new level up
    # in the same session.
    effort: str | None = Field(default=None, min_length=1)


class ForkSessionRequest(HarnessModel):
    # Fork payload for ``POST /v1/sessions/{id}/forks``. Both fields are
    # optional. ``message`` — when present — starts a first run in the fork
    # (the bridge relies on this synchronous launch; it does not call
    # ``create_run`` afterwards). ``title`` overrides the inherited parent
    # title. Empty strings are rejected (omit the field instead), matching
    # ``PatchSessionRequest``.
    message: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)


class CreateRunRequest(HarnessModel):
    message: str = Field(min_length=1)
    model: str | None = None


class CreateRunResponse(HarnessModel):
    session_id: str
    run_id: str
    # "running" when the harness spawned the subprocess immediately, "queued"
    # when the session already had an in-flight run and this one is waiting.
    status: RunStatus = "running"


class ForkSessionResponse(HarnessModel):
    # ``session`` is the new harness-owned child. ``run`` is the first run
    # started in the fork when a ``message`` was supplied, else ``None``.
    session: Session
    run: Run | None = None


class InterruptRunResponse(HarnessModel):
    # ``run`` is the run targeted by the DELETE — interrupted whether it was
    # actively running or merely queued. ``dropped_queued`` lists *other*
    # runs that were sitting in the session's queue and got dropped as a
    # side-effect (a DELETE on any run empties the per-session queue, since
    # the user is signalling they want this conversation flow stopped).
    run: Run
    dropped_queued: list[Run] = Field(default_factory=list)


class DataList(HarnessModel):
    data: list[Any]
