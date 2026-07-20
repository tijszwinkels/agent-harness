from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_harness.events import InMemoryEventBus
from agent_harness.models import Event, Message, Run, RunStatus, Session, TextBlock, utc_now
from agent_harness.rollout_discovery import RolloutDiscovery

logger = logging.getLogger(__name__)

END_TURN_GRACE_SECONDS = 20.0
END_TURN_HARD_KILL_AFTER_SECONDS = 40.0
IDLE_TIMEOUT_SECONDS = 30 * 60
IDLE_HARD_KILL_GRACE_SECONDS = 30.0
IDLE_CHECK_INTERVAL_SECONDS = 60.0
# After the FOREGROUND child exits, the stdout/stderr drain gets this
# many seconds to reach EOF before we stop waiting on the readers. A
# background orphan spawned by the agent (same process group, inherited
# the pipe write-end) keeps the pipe open indefinitely, so EOF never
# arrives and the run would otherwise hang "running" forever. Run
# completion must be bound to the foreground child's exit, not to
# inherited-pipe EOF. The normal case (no orphan) EOFs well within this
# window, so clean output still drains fully and fast.
POST_EXIT_DRAIN_GRACE_SECONDS = 5.0

# Steering prompt appended to every harness claude run. Verified
# empirically (claude v2.1.200): ``claude --print`` KILLS background
# Bash tasks (``task_type: local_bash``) at turn teardown —
# task_updated status "killed" + task_notification "stopped" right
# after the result record — while the model believes it will be
# notified on completion, so the work is silently dropped. There is
# no CLI flag to disable backgrounding; the system prompt is the only
# steering channel. Async Task-tool subagents are the explicit
# exception: those DO survive the turn (claude stays alive and
# re-invokes the model — see ``_watch_end_turn_cleanup``).
CLAUDE_PRINT_MODE_SYSTEM_PROMPT = (
    "You are running non-interactively under `claude --print` inside an "
    "automated harness. Bash tool calls with `run_in_background: true` (and "
    "commands auto-promoted to background on timeout) are killed when your "
    "turn ends — you will never receive their completion notification. Run "
    "commands synchronously (raise the Bash timeout if needed), or for work "
    "that must outlive the turn, detach it on the host "
    "(`setsid cmd > /tmp/log 2>&1 < /dev/null &`) and check the log on a "
    "later turn. Subagents launched via the Task tool DO keep running after "
    "your turn ends and will re-invoke you on completion."
)

# Terminal statuses for async-task lifecycle records on claude's
# stream-json stdout (see ``RunProcess._note_stdout_task_event``).
# ``task_updated`` patches carrying one of these mean the async task
# is done. Observed values (claude v2.1.200): "completed", "killed";
# "failed" and "stopped" are included as the same family of terminal
# states. Patches without a status (or with a non-terminal one) are
# progress/rename updates, NOT lifecycle transitions.
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "killed", "stopped"})

# Phase 4: events that tick ``_last_activity_at`` via ``RunProcess._publish``,
# keeping the 30-min idle watchdog warm. The observer's ``message`` /
# ``tool_use`` / ``run.usage`` / ``run.end_turn`` events flow through
# the shared bus but DON'T go through ``RunProcess._publish``, so they
# don't tick this map. The two activity signals the supervisor still
# publishes are ``process.stderr`` (stderr-active runs) and the
# stdout heartbeat (which sets ``_last_activity_at`` directly in
# ``_stream_lines``, bypassing _publish). Idle-watchdog rewire to
# subscribe to observer events is a future-phase candidate; for now
# stdout heartbeat + stderr keep claude / codex runs warm.
_ACTIVITY_EVENTS = frozenset({
    "process.stderr",
})


class CommandBuildError(ValueError):
    """Raised when a session cannot be mapped to a runnable backend command."""


def _model_flag(session: Session) -> tuple[str, ...]:
    # ``--model M`` when the session pins a model; empty when it doesn't, so
    # the backend CLI falls back to its own configured default. All three CLIs
    # accept an absent ``--model`` (verified 2026-07-20).
    return ("--model", session.model) if session.model is not None else ()


@dataclass(frozen=True)
class ProcessCommand:
    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("ProcessCommand requires at least one argv item")


class BackendCommandBuilder(Protocol):
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        """Build a non-interactive CLI command for a harness-owned run.

        ``is_first_run`` is ``True`` when no prior run exists for this
        session. Builders use it to pick between session-creation and
        session-resume flags (e.g. ``claude --session-id`` vs
        ``claude --resume``).
        """


class CodexCommandBuilder:
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        del run, is_first_run
        text = _message_text(message)
        bypass: tuple[str, ...] = (
            ("--dangerously-bypass-approvals-and-sandbox",)
            if session.bypass_permissions
            else ()
        )
        # Unified resume gate: present codex_resume_id → resume,
        # absent → fresh exec. The field is populated by the observer
        # when binding extracts the UUID from a bound rollout's
        # filename (harness origin) or by the startup backfill from
        # ``codex_<uuid>`` ids (external origin). Replaces the prior
        # split between an origin==external branch and a no-resume
        # harness branch — the harness side never resumed at all,
        # which is why multi-turn codex runs lost context across
        # turns (spec: 2026-05-21-codex-resume.md).
        if session.codex_resume_id is not None:
            return ProcessCommand(
                argv=(
                    "codex",
                    "exec",
                    "resume",
                    "--json",
                    *_model_flag(session),
                    *bypass,
                    session.codex_resume_id,
                    # ``--`` ends option parsing so a ``-``-prefixed prompt is
                    # read as the positional prompt, not an unknown flag (codex
                    # exits 2 otherwise). Same guard as the claude builder.
                    "--",
                    text,
                ),
                cwd=session.project.path,
            )

        return ProcessCommand(
            argv=("codex", "exec", "--json", *_model_flag(session), *bypass, "--", text),
            cwd=session.project.path,
        )


def _harness_session_id_as_uuid(session_id: str) -> str:
    # Accepts any of the session-id shapes the harness or observer can
    # produce for a claude session and returns the canonical
    # 8-4-4-4-12 UUID string (claude --session-id / --resume validates UUID
    # format).
    #
    #   * ``ses_<32hex>``               — canonical harness + external form
    #   * ``claude_<uuid-with-dashes>`` — legacy external-observer form,
    #                                     kept for records persisted before
    #                                     the canonicalization.
    #   * bare 32-hex or dashed UUID    — best-effort fallback.
    if session_id.startswith("ses_"):
        body = session_id.removeprefix("ses_")
    elif session_id.startswith("claude_"):
        body = session_id.removeprefix("claude_")
    else:
        body = session_id
    hex_part = body.replace("-", "").lower()
    if len(hex_part) != 32 or not all(c in "0123456789abcdef" for c in hex_part):
        raise CommandBuildError(
            f"Cannot derive claude session UUID from harness session id {session_id!r}",
        )
    return f"{hex_part[0:8]}-{hex_part[8:12]}-{hex_part[12:16]}-{hex_part[16:20]}-{hex_part[20:32]}"


def claude_conversation_exists(session: Session, *, home: str | Path | None = None) -> bool:
    """True when claude has already written this session's transcript to disk.

    Drives the ``--session-id`` (create) vs ``--resume`` choice off whether the
    conversation actually exists, rather than "has any prior run". A run that
    failed *before* claude created the conversation (e.g. an arg-parse error on
    a ``-``-prefixed prompt) is then retried as a create — otherwise every later
    run ``--resume``s a session that was never born and fails forever ("No
    conversation found with session ID"). Returns False for ids that aren't
    claude UUIDs (nothing to resume).
    """
    try:
        claude_uuid = _harness_session_id_as_uuid(session.id)
    except CommandBuildError:
        return False
    # Local import avoids any orchestrator<->observer import-order coupling.
    from agent_harness.observer import claude_transcript_path

    return claude_transcript_path(session.project.path, claude_uuid, home=home).exists()


class ClaudeCodeCommandBuilder:
    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        del run
        text = _message_text(message)
        argv = (
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            CLAUDE_PRINT_MODE_SYSTEM_PROMPT,
            *_model_flag(session),
        )
        if session.bypass_permissions:
            argv = (*argv, "--dangerously-skip-permissions")
        claude_uuid = _harness_session_id_as_uuid(session.id)
        if session.origin == "external":
            # External claude sessions are already running under this UUID
            # (claude wrote the .jsonl with it). We can only --resume — we
            # don't own session creation.
            argv = (*argv, "--resume", claude_uuid)
        elif is_first_run and session.forked_from is not None:
            # Fork: resume the PARENT's conversation but --fork-session so
            # claude writes to a NEW id, pinned to this child's own UUID.
            # The parent transcript is read-only (verified 2026-07-20). Only
            # the first run forks; once the child transcript exists, follow-up
            # runs resume the child normally via the branch below.
            parent_uuid = _harness_session_id_as_uuid(session.forked_from)
            argv = (*argv, "--resume", parent_uuid, "--fork-session", "--session-id", claude_uuid)
        else:
            # Harness-origin: pin a deterministic claude session UUID so
            # subsequent runs can --resume and retain conversation context.
            # claude --session-id creates the session on first use and
            # errors on duplicate, so we switch to --resume for follow-up.
            flag = "--session-id" if is_first_run else "--resume"
            argv = (*argv, flag, claude_uuid)

        # ``--`` terminates option parsing so a prompt that begins with ``-``
        # (e.g. a chat message starting with a dash) is taken as the positional
        # prompt rather than an unknown flag. Without it claude exits 1
        # ("error: unknown option '-…'") before creating the conversation,
        # which strands the session on a phantom --resume target.
        return ProcessCommand(
            argv=(*argv, "--", text),
            cwd=session.project.path,
        )


class PiCommandBuilder:
    """Headless pi runs: ``pi -p --model M --session-id <uuid> [-a] <text>``.

    Modelled on ``CodexCommandBuilder`` (the simpler template — no
    stdout/rollout coupling), with one simplification: pi's
    ``--session-id <id>`` "creates it if missing" (verified pi v0.80.3
    ``--help``), so the SAME flag serves the first run and every resume.
    There is no first-run/resume flag switch like claude's
    ``--session-id`` -> ``--resume`` — passing a deterministic UUID
    derived from the harness session id on every run makes pi create
    the session on the first turn and load it (retaining context) on the
    next. Empirically verified for multi-turn context before landing
    (see specs/2026-07-05-pi-backend-deviations.md).

    The executable is the bare ``pi`` on PATH — matching how the
    claude/codex builders hardcode their binaries. Node >= 22.19 (pi's
    hard requirement; it crashes on Node 20 with ``markAsUncloneable``)
    is delivered operationally by bumping the machine's default Node for
    the harness's systemd environment, NOT by a pi-specific env-var/
    wrapper indirection in the harness code (spec-gate remark 1).
    """

    def build(
        self,
        *,
        session: Session,
        run: Run,
        message: Message,
        is_first_run: bool = True,
    ) -> ProcessCommand:
        del run
        text = _message_text(message)
        # Reuse the claude UUID derivation: harness ``ses_<hex>`` ->
        # canonical 8-4-4-4-12 UUID. pi accepts any string id, but a
        # UUID keeps the on-disk session filenames uniform and lets the
        # same helper (and its non-hex guard) cover both backends.
        session_uuid = _harness_session_id_as_uuid(session.id)
        approve: tuple[str, ...] = ("-a",) if session.bypass_permissions else ()
        # Fork: on the first run, ``pi --fork <parent> --session-id <child>``
        # forks the parent session file into the child's pinned id, leaving
        # the parent untouched (verified 2026-07-20). Follow-up runs drop
        # ``--fork`` and just load the child via its idempotent --session-id.
        fork: tuple[str, ...] = ()
        if is_first_run and session.forked_from is not None:
            fork = ("--fork", _harness_session_id_as_uuid(session.forked_from))
        return ProcessCommand(
            argv=(
                "pi",
                "-p",
                *_model_flag(session),
                *fork,
                "--session-id",
                session_uuid,
                *approve,
                text,
            ),
            cwd=session.project.path,
        )


def default_command_builders() -> dict[str, BackendCommandBuilder]:
    return {
        "claude-code": ClaudeCodeCommandBuilder(),
        "codex": CodexCommandBuilder(),
        "pi": PiCommandBuilder(),
    }


# Phase 4 retired the supervisor's stdout end-turn detector
# (``_detect_end_turn_in_line``). End-turn signaling now flows from
# the observer's rollout parser via the ``run.end_turn`` event bus
# subscription (see ``RunProcess._wait_for_run_end_turn``). The
# supervisor's stdout pump is now purely a heartbeat tick + stderr
# passthrough (see ``_stream_lines``).


# Backends whose non-interactive CLI can fork a session cleanly (resume the
# parent's history into a NEW transcript, leaving the parent untouched):
#   * claude-code — ``--resume <parent> --fork-session --session-id <child>``
#   * pi          — ``--fork <parent> --session-id <child>``
# codex is excluded: ``codex exec resume`` appends to the parent's own rollout
# (would mutate it), and ``codex fork`` is a TUI-only subcommand with no
# ``--json`` mode — incompatible with the harness's ``codex exec --json``
# pipeline. Verified against the real CLIs 2026-07-20.
_FORKABLE_BACKENDS: frozenset[str] = frozenset({"claude-code", "pi"})


def session_supports_fork(backend: str) -> bool:
    return backend in _FORKABLE_BACKENDS


def validate_fork_source(parent: Session) -> None:
    """Raise ``CommandBuildError`` when ``parent`` can't seed a fork.

    Mirrors ``validate_session_resume_target``: the API layer catches the
    error and maps it to a 409 so the bridge surfaces its "cannot fork"
    message.
    """
    if not session_supports_fork(parent.backend):
        raise CommandBuildError(
            f"Backend {parent.backend} does not support forking",
        )
    # Both forkable backends derive the parent's fork UUID from its session id
    # (claude/pi share the ``ses_<hex>`` / legacy / raw-UUID derivation). A
    # parent whose id can't produce a UUID can't be a fork source.
    _harness_session_id_as_uuid(parent.id)


def validate_session_resume_target(session: Session) -> None:
    if session.origin != "external":
        return
    if session.backend == "codex":
        # External codex sessions must carry codex_resume_id (set by
        # the startup backfill from the ``codex_<uuid>`` id). Falling
        # through to fresh exec on an external session would silently
        # discard the rollout's prior context.
        if session.codex_resume_id is None:
            raise CommandBuildError(
                f"Cannot resume external codex session from id {session.id}",
            )
        return
    if session.backend == "claude-code":
        # Validates session.id resolves to a claude UUID under any of the
        # accepted shapes (ses_<hex>, legacy claude_<uuid>, raw UUID).
        _harness_session_id_as_uuid(session.id)
        return
    raise CommandBuildError(f"Cannot resume external session for backend {session.backend}")


class AsyncLineReader(Protocol):
    async def readline(self) -> bytes:
        pass


class ManagedProcess(Protocol):
    stdout: AsyncLineReader | None
    stderr: AsyncLineReader | None
    returncode: int | None
    pid: int

    async def wait(self) -> int:
        pass

    def terminate(self) -> None:
        pass


class ProcessFactory(Protocol):
    async def __call__(self, command: ProcessCommand) -> ManagedProcess:
        pass


class AsyncioProcessFactory:
    async def __call__(self, command: ProcessCommand) -> ManagedProcess:
        env = None if not command.env else os.environ | dict(command.env)
        return await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=command.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )


@dataclass(frozen=True)
class RunProcessResult:
    run_id: str
    status: RunStatus
    returncode: int | None = None
    error: str | None = None


class RunProcess:
    def __init__(
        self,
        *,
        session: Session,
        run: Run,
        command: ProcessCommand,
        event_bus: InMemoryEventBus,
        process_factory: ProcessFactory | None = None,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        observer: Any | None = None,
        rollout_discovery: Any | None = None,
    ) -> None:
        self.session = session
        self.run_record = run
        self.command = command
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._process: ManagedProcess | None = None
        self._interrupted = False
        self._clock = clock
        self._sleep = sleep
        # Phase 4: end-turn signaling moved to the event bus
        # (``_wait_for_run_end_turn`` subscribes by session_id).
        # The internal ``_end_turn_event`` asyncio.Event is retired.
        self._last_activity_at = self._clock()
        self._last_activity_event: str | None = None
        self._watchdog_termination_in_progress = False
        self._watchdog_termination_task: asyncio.Task[None] | None = None
        # claude v2.x async subagents: task lifecycle records
        # (task_started / task_updated / task_notification) appear
        # ONLY on stdout — never in the rollout the observer tails
        # (verified empirically, claude v2.1.200) — so the supervisor
        # mirrors the pending-task set here for the end-turn
        # watchdog's kill/defer decision. Maps task_id → task_type
        # ("local_agent", "local_bash", or "" when absent).
        self._pending_tasks: dict[str, str] = {}
        # Phase 2: rollout pre-binding is unconditional for harness
        # sessions. claude uses the deterministic ``discover_claude``
        # path; codex uses the expectation registry (no fd probe).
        # ``observer`` and ``rollout_discovery`` may be ``None`` for
        # tests that don't exercise the pre-bind path; production wires
        # both via api.py's lifespan + cli.py.
        self._observer = observer
        self._rollout_discovery = rollout_discovery
        # Set to the bound rollout path once claude pre-bind succeeds,
        # so ``run()``'s finally block can unbind on terminal-state
        # cleanup. Codex uses the expectation registry instead; no
        # bound path to clean up there (the registry self-purges via
        # TTL + consume-on-match).
        self._bound_rollout_path: Path | None = None

    async def run(self) -> RunProcessResult:
        await self._publish("run.started", {})

        # Phase 2: codex pre-bind registers an expectation BEFORE the
        # spawn so a fast watchfiles fire on codex's first byte can
        # still resolve correctly. Claude pre-bind runs post-spawn
        # (deterministic; no race against the file existing).
        self._pre_register_codex_expectation_if_codex()

        try:
            self._process = await self._process_factory(self.command)
        except Exception as exc:
            logger.exception("Failed to start run process: session=%s run=%s", self.session.id, self.run_record.id)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))

        # Claude pre-bind (deterministic, post-spawn).
        self._pre_bind_claude_if_claude()

        stream_tasks = self._stream_tasks(self._process)
        wait_task = asyncio.create_task(self._process.wait())
        watchdog_tasks = [
            asyncio.create_task(self._watch_end_turn_cleanup(wait_task)),
            asyncio.create_task(self._watch_idle_timeout(wait_task)),
        ]
        try:
            returncode = await wait_task
            await self._drain_streams_after_exit(stream_tasks)
        except Exception as exc:
            logger.exception("Run process failed while active: session=%s run=%s", self.session.id, self.run_record.id)
            await self._cancel_streams(stream_tasks)
            await self._publish("run.failed", {"error": str(exc), "error_type": type(exc).__name__})
            return RunProcessResult(run_id=self.run_record.id, status="failed", error=str(exc))
        finally:
            await self._finish_watchdogs(watchdog_tasks)
            # Drop every rollout binding pointing at this session —
            # keeps ``observer._path_to_session`` bounded across the
            # harness lifetime (one entry per *active* harness run,
            # not per all runs ever). Covers claude (path-bound at
            # spawn) and codex (path bound by the observer at
            # expectation-match time) through a single eviction call.
            self._unbind_rollouts_for_session()

        if self._interrupted:
            await self._publish("run.interrupted", {"returncode": returncode})
            return RunProcessResult(run_id=self.run_record.id, status="interrupted", returncode=returncode)

        if returncode == 0:
            await self._publish("run.completed", {"returncode": returncode})
            return RunProcessResult(run_id=self.run_record.id, status="completed", returncode=returncode)

        logger.warning(
            "Run process exited with non-zero status: session=%s run=%s returncode=%s",
            self.session.id,
            self.run_record.id,
            returncode,
        )
        await self._publish("run.failed", {"returncode": returncode})
        return RunProcessResult(run_id=self.run_record.id, status="failed", returncode=returncode)

    async def interrupt(self) -> bool:
        if self.run_record.origin != "harness":
            return False
        if self._process is None:
            return False

        self._interrupted = True
        self._process.terminate()
        return True

    def _pre_bind_enabled(self) -> bool:
        if self.session.origin != "harness":
            return False
        if self._observer is None:
            return False
        return True

    def _pre_register_codex_expectation_if_codex(self) -> None:
        """Phase 2 codex hand-off. Replaces Phase 1's psutil-based fd
        probe with a one-line expectation registration the observer
        will match content-side via ``session_meta`` peek. No spawn-
        time blocking, no race with watchfiles."""
        if self.session.backend != "codex":
            return
        if not self._pre_bind_enabled():
            return
        cwd_str = self.command.cwd
        if cwd_str is None:
            logger.warning(
                "Skipping codex expectation registration: command has no cwd; session=%s",
                self.session.id,
            )
            return
        try:
            self._observer.expect_codex_rollout(
                cwd=Path(cwd_str), session_id=self.session.id
            )
        except Exception:
            logger.exception(
                "Codex expectation registration failed: session=%s",
                self.session.id,
            )
            return
        logger.debug(
            "Codex rollout expectation registered: session=%s cwd=%s",
            self.session.id,
            cwd_str,
        )

    def _pre_bind_claude_if_claude(self) -> None:
        if self.session.backend != "claude-code":
            return
        if not self._pre_bind_enabled():
            return
        if self._rollout_discovery is None:
            return
        cwd_str = self.command.cwd
        if cwd_str is None:
            logger.warning(
                "Skipping claude rollout pre-bind: command has no cwd; session=%s",
                self.session.id,
            )
            return
        # Claude writes the rollout under the dashed-UUID stem it was
        # invoked with (orchestrator builds ``--session-id <uuid>``).
        # Bind under that exact filename — using the harness's
        # ``ses_<hex>`` form here would silently miss the file the
        # observer actually sees on disk.
        try:
            claude_uuid = _harness_session_id_as_uuid(self.session.id)
        except CommandBuildError as exc:
            logger.warning(
                "Skipping claude rollout pre-bind: cannot derive UUID; session=%s error=%s",
                self.session.id,
                exc,
            )
            return
        try:
            path = self._rollout_discovery.discover_claude(
                session_id=claude_uuid, cwd=Path(cwd_str)
            )
            self._observer.bind_rollout(path, self.session.id)
        except Exception:
            logger.exception(
                "Claude rollout pre-bind raised: session=%s",
                self.session.id,
            )
            return
        self._bound_rollout_path = path
        logger.debug(
            "Claude rollout pre-bound: session=%s path=%s",
            self.session.id,
            path,
        )

    def _unbind_rollouts_for_session(self) -> None:
        """Terminal-state cleanup hook.

        Calls ``observer.unbind_session`` unconditionally for
        harness-origin sessions so both claude (path-bound at spawn,
        ``_bound_rollout_path`` tracked here) and codex (path bound by
        the observer from the matched expectation, NOT visible to the
        orchestrator) converge on the same eviction path. The
        observer walks ``_path_to_session`` for entries whose value
        equals ``self.session.id`` and removes them.

        Idempotent; falls back to ``unbind_rollout`` for the
        deprecated path-keyed cleanup if the observer doesn't support
        the session-keyed API (test fakes pre-this-fix).
        """
        if self._observer is None or self.session.origin != "harness":
            self._bound_rollout_path = None
            return
        try:
            unbind_session = getattr(self._observer, "unbind_session", None)
            if callable(unbind_session):
                unbind_session(self.session.id)
            elif self._bound_rollout_path is not None:
                self._observer.unbind_rollout(self._bound_rollout_path)
        except Exception:
            logger.exception(
                "Failed to unbind rollouts for session=%s", self.session.id
            )
        finally:
            self._bound_rollout_path = None

    def _stream_tasks(self, process: ManagedProcess) -> list[asyncio.Task[None]]:
        tasks: list[asyncio.Task[None]] = []
        if process.stdout is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stdout", process.stdout)))
        if process.stderr is not None:
            tasks.append(asyncio.create_task(self._stream_lines("stderr", process.stderr)))
        return tasks

    async def _stream_lines(self, stream_name: Literal["stdout", "stderr"], stream: AsyncLineReader) -> None:
        # Phase 4: the supervisor's stdout pump is heartbeat + stderr
        # passthrough. End-turn signaling moved to the observer
        # (rollout-parser → ``run.end_turn`` event → watchdog
        # subscription); message data has flowed through the observer
        # since Phase 2. One narrow exception to "no parsing":
        # ``_note_stdout_task_event`` mirrors async-task lifecycle
        # records, which exist ONLY on stdout — never in the rollout —
        # so the observer cannot supply them. End-turn detection does
        # NOT move back here.
        #
        # Heartbeat: every non-empty stdout line ticks
        # ``_last_activity_at`` directly (NOT via ``_publish``),
        # keeping the idle watchdog warm for long-running harness
        # runs that emit no stderr (claude talks only via rollout).
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                continue
            if stream_name == "stderr":
                await self._publish("process.stderr", {"text": text})
                continue
            # Stdout heartbeat (no publish).
            self._last_activity_at = self._clock()
            self._last_activity_event = "stdout"
            self._note_stdout_task_event(text)

    def _note_stdout_task_event(self, text: str) -> None:
        """Track async-task lifecycle records from stream-json stdout.

        claude v2.x async subagents (Task tool, ``task_type``
        "local_agent") and background Bash ("local_bash") emit
        ``{"type": "system", "subtype": "task_started" | "task_updated"
        | "task_progress" | "task_notification", "task_id": ..., ...}``
        records on stdout ONLY — verified empirically (claude
        v2.1.200): the rollout transcript carries zero task lifecycle
        records, so the observer can never see them. The end-turn
        watchdog consults ``_pending_tasks`` to distinguish a process
        that is legitimately alive past end_turn (waiting on a
        subagent that will re-invoke the model) from a wedged one.
        """
        # Cheap prefilter: all four subtypes contain ``"task_`` — the
        # vast majority of stream lines (partial-message deltas etc.)
        # skip the json.loads entirely.
        if '"task_' not in text:
            return
        try:
            record = json.loads(text)
        except ValueError:
            logger.debug(
                "Ignoring unparseable stdout line with task marker: session=%s run=%s",
                self.session.id,
                self.run_record.id,
            )
            return
        if not isinstance(record, dict) or record.get("type") != "system":
            return
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return
        subtype = record.get("subtype")
        if subtype == "task_started":
            self._pending_tasks[task_id] = str(record.get("task_type") or "")
            logger.debug(
                "run %s: async task started: task_id=%s task_type=%s (%d pending)",
                self.run_record.id,
                task_id,
                self._pending_tasks[task_id],
                len(self._pending_tasks),
            )
            return
        if subtype == "task_updated":
            patch = record.get("patch")
            status = patch.get("status") if isinstance(patch, dict) else None
            if status not in _TERMINAL_TASK_STATUSES:
                return
        elif subtype != "task_notification":
            # task_progress (and anything else): progress-only, never
            # a lifecycle transition.
            return
        # Terminal task_updated or task_notification (always terminal).
        if self._pending_tasks.pop(task_id, None) is not None:
            logger.debug(
                "run %s: async task finished: task_id=%s subtype=%s (%d pending)",
                self.run_record.id,
                task_id,
                subtype,
                len(self._pending_tasks),
            )

    async def _finish_streams(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.exception(
                    "Run stream reader failed: session=%s run=%s",
                    self.session.id,
                    self.run_record.id,
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def _drain_streams_after_exit(self, tasks: list[asyncio.Task[None]]) -> None:
        """Bound the post-exit stdout/stderr drain to a finite grace.

        Called once the FOREGROUND child has exited. In the normal case
        the readers reach EOF within milliseconds and we drain every
        line exactly as ``_finish_streams`` always did. But if the agent
        promoted a BACKGROUND task into its own process group, that
        orphan inherited the pipe write-end and the pipe never EOFs — the
        readers would block on ``readline()`` forever and ``run()`` would
        never publish a terminal event (the stuck-session bug).

        Resolution: give the drain ``POST_EXIT_DRAIN_GRACE_SECONDS`` to
        finish; if it doesn't, the foreground child is already gone so
        the run is logically over. Reap the process group (releases the
        orphan's inherited pipe write-end via SIGTERM) and then abandon
        the readers, so completion is bound to the foreground exit rather
        than to inherited-pipe EOF.
        """
        if not tasks:
            return
        # ALL_COMPLETED + a wall-clock timeout: returns the instant every
        # reader has EOF'd (the normal, near-instant path) OR once the
        # grace elapses with readers still blocked. The timeout is real
        # wall-clock on purpose — it is bounded and small, and keeping it
        # off the injected ``self._sleep`` avoids loading extra event-loop
        # hops onto the run hot-path's clean-completion case.
        _done, pending = await asyncio.wait(
            tasks,
            timeout=POST_EXIT_DRAIN_GRACE_SECONDS,
            return_when=asyncio.ALL_COMPLETED,
        )
        if not pending:
            # Clean EOF within grace — normal path. Surface any reader
            # exception exactly as ``_finish_streams`` did.
            await self._finish_streams(tasks)
            return

        # Grace expired with readers still blocked: an orphan is holding
        # the pipe open. Reap the group to release it, then stop waiting.
        logger.warning(
            "Stream drain exceeded %.0fs after foreground exit; reaping process "
            "group and abandoning readers (likely an orphaned background task "
            "holding the pipe): session=%s run=%s",
            POST_EXIT_DRAIN_GRACE_SECONDS,
            self.session.id,
            self.run_record.id,
        )
        self._reap_process_group(signal.SIGTERM)
        # Cancel the still-blocked readers so the run proceeds even if the
        # killpg above couldn't release the pipe (e.g. the group was
        # already gone, or a grandchild escaped the group). Correctness
        # does not depend on the SIGTERM landing.
        await self._cancel_streams(list(pending))

    async def _cancel_streams(self, tasks: list[asyncio.Task[None]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _finish_watchdogs(self, tasks: list[asyncio.Task[None]]) -> None:
        if self._watchdog_termination_in_progress:
            terminator = self._watchdog_termination_task
            to_cancel = [task for task in tasks if task is not terminator]
            await self._cancel_watchdogs(to_cancel)
            if terminator is not None:
                await asyncio.gather(terminator, return_exceptions=True)
            return
        await self._cancel_watchdogs(tasks)

    async def _cancel_watchdogs(self, tasks: list[asyncio.Task[None]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _wait_for_run_end_turn(self) -> bool:
        """Block until the observer publishes a ``run.end_turn`` event
        for this RunProcess's session AND run id.

        Returns ``True`` when the matching event arrived (caller
        proceeds to the grace + SIGTERM ladder); returns ``False`` if
        the subscription is cancelled while we wait. Subscribes
        filtered by ``session_id`` (cheap filter built into
        bus.subscribe) and double-checks ``event.run_id`` in the
        handler so a same-session parallel run can't arm our cleanup.

        ``after=current_max_sequence`` skips historical events: the
        watchdog only cares about end-turn signals fired AFTER this
        RunProcess started. Without this filter, the observer's first
        tail of a never-observed rollout (e.g. a fresh harness
        restart picking up a long-lived external session) would
        replay historical ``run.end_turn`` records — and
        ``_resolve_active_run_id`` would stamp them with THIS run's
        id, arming the watchdog SIGTERM within the 20s grace window
        for a brand-new run.
        """
        after = await self._event_bus.max_sequence(session_id=self.session.id)
        subscription = self._event_bus.subscribe(
            after=after, session_id=self.session.id
        )
        try:
            async for event in subscription:
                if event is None:
                    continue
                if event.event != "run.end_turn":
                    continue
                # ``run_id`` is stamped at the top level by the
                # observer's ``_resolve_active_run_id`` (a model_copy
                # update on ``Event.run_id``); the parser-built event
                # carries ``run_id=None`` until that resolution.
                if event.run_id == self.run_record.id:
                    return True
        except asyncio.CancelledError:
            raise
        finally:
            await subscription.aclose()
        return False

    async def _watch_end_turn_cleanup(self, wait_task: asyncio.Task[int]) -> None:
        # Phase 4: subscribe to observer-emitted ``run.end_turn``
        # events on the bus instead of polling an internal
        # stdout-derived signal. Filter by session_id (subscription
        # parameter) AND ``data.run_id`` (handler-side check) so a
        # parallel run for the same session doesn't arm our cleanup.
        #
        # Latency: end-turn timing now depends on rollout flush
        # cadence — empirically <1s for claude (flushes promptly on
        # ``stop_reason``) and <2s for codex (flushes on
        # ``task_complete``). Acceptable for a 20s grace window.
        #
        # Async-subagent awareness (claude v2.x, verified v2.1.200):
        # the rollout records ``stop_reason=end_turn`` even while an
        # async Task-tool subagent is still running. ``claude --print``
        # legitimately stays alive, streams task_progress on stdout,
        # RE-INVOKES the model when the subagent finishes, and only
        # exits after a turn ends with no pending tasks. Killing at
        # first end_turn destroys the subagent's work (same process
        # group). So a kill requires genuine post-end-turn quiescence:
        #
        #   (a) no pending async tasks (``_pending_tasks``, mirrored
        #       from the stdout-only task lifecycle records) —
        #       otherwise defer and re-arm on the NEXT end_turn, which
        #       fires after the post-subagent model re-invocation;
        #   (b) stdout/stderr silent for a full grace window — output
        #       since the end_turn means the process is NOT in
        #       post-turn quiescence (model re-invoked / teardown
        #       flushing), so re-check after the remaining quiet time.
        #       A wedged process goes silent, so the kill still lands
        #       ~grace seconds after its last output.
        #
        # Known accepted corner: if a NEW end_turn fires while we're
        # inside the quiescence loop and we then defer back to
        # re-subscribe, that end_turn may be missed (the subscription
        # starts after ``max_sequence``); a wedge in that narrow
        # window is reaped by the 30-min idle watchdog instead.
        while True:
            if not await self._wait_for_run_end_turn():
                return
            if self._process_exited(wait_task):
                return

            await self._sleep(END_TURN_GRACE_SECONDS)
            while True:
                if self._process_exited(wait_task):
                    return
                if self._pending_tasks:
                    logger.info(
                        "run %s: end_turn with %d async task(s) pending — "
                        "deferring end-turn cleanup, waiting for next end_turn",
                        self.run_record.id,
                        len(self._pending_tasks),
                    )
                    break  # → outer loop: wait for the NEXT end_turn.
                quiet_seconds = (self._clock() - self._last_activity_at).total_seconds()
                if quiet_seconds < END_TURN_GRACE_SECONDS:
                    # Activity since the end_turn — not quiescent yet.
                    await self._sleep(END_TURN_GRACE_SECONDS - quiet_seconds)
                    continue

                self._watchdog_termination_in_progress = True
                self._watchdog_termination_task = asyncio.current_task()
                self._interrupted = True
                if not self._signal_process_group(signal.SIGTERM):
                    return

                hard_kill = False
                remaining = max(0.0, END_TURN_HARD_KILL_AFTER_SECONDS - END_TURN_GRACE_SECONDS)
                if remaining:
                    await self._wait_for_process_or_sleep(wait_task, remaining)

                if not self._process_exited(wait_task):
                    hard_kill = self._signal_process_group(signal.SIGKILL)
                if hard_kill:
                    await wait_task

                await self._publish(
                    "run.terminated_after_end_turn",
                    {
                        "grace_seconds": int(END_TURN_GRACE_SECONDS),
                        "hard_kill": hard_kill,
                        "returncode": self._returncode(wait_task),
                        "reason": "subprocess_did_not_exit_after_end_turn",
                    },
                )
                return

    async def _watch_idle_timeout(self, wait_task: asyncio.Task[int]) -> None:
        while True:
            await self._sleep(IDLE_CHECK_INTERVAL_SECONDS)
            if self._process_exited(wait_task):
                return

            idle_seconds = (self._clock() - self._last_activity_at).total_seconds()
            if idle_seconds <= IDLE_TIMEOUT_SECONDS:
                continue

            self._watchdog_termination_in_progress = True
            self._watchdog_termination_task = asyncio.current_task()
            self._interrupted = True
            if not self._signal_process_group(signal.SIGTERM):
                return

            hard_kill = False
            await self._wait_for_process_or_sleep(wait_task, IDLE_HARD_KILL_GRACE_SECONDS)
            if not self._process_exited(wait_task):
                hard_kill = self._signal_process_group(signal.SIGKILL)
                if hard_kill:
                    await wait_task

            await self._publish(
                "run.timed_out_idle",
                {
                    "idle_seconds": int(IDLE_TIMEOUT_SECONDS),
                    "last_activity_event": self._last_activity_event,
                    "last_activity_at": _format_timestamp(self._last_activity_at),
                    "hard_kill": hard_kill,
                    "reason": "no_activity_within_threshold",
                },
            )
            return

    async def _wait_for_process_or_sleep(self, wait_task: asyncio.Task[int], seconds: float) -> None:
        if wait_task.done():
            return
        sleep_task = asyncio.create_task(self._sleep(seconds))
        done, pending = await asyncio.wait({wait_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
        del done
        if sleep_task in pending:
            sleep_task.cancel()
            await asyncio.gather(sleep_task, return_exceptions=True)

    def _process_exited(self, wait_task: asyncio.Task[int]) -> bool:
        process = self._process
        return wait_task.done() or process is None or process.returncode is not None

    def _returncode(self, wait_task: asyncio.Task[int]) -> int | None:
        if wait_task.done() and not wait_task.cancelled():
            try:
                return wait_task.result()
            except Exception:
                logger.exception(
                    "Run process wait task failed while reading returncode: session=%s run=%s",
                    self.session.id,
                    self.run_record.id,
                )
        if self._process is not None:
            return self._process.returncode
        return None

    def _reap_process_group(self, sig: signal.Signals) -> bool:
        """Signal the run's process group AFTER the foreground child has
        exited, to clean up orphaned background descendants.

        Distinct from ``_signal_process_group`` in two ways the orphan-
        reap path needs:

        * No ``returncode is not None`` short-circuit. The foreground
          child has *already* exited here (that's the whole point); the
          orphans we're reaping are other members of its group.
        * Targets ``process.pid`` directly as the pgid instead of
          ``os.getpgid(pid)``. The child was launched with
          ``start_new_session=True``, so it is the group leader and the
          pgid equals its pid — and once asyncio has reaped the leader,
          ``os.getpgid(pid)`` would raise ``ProcessLookupError`` even
          while orphan group members are still alive.
        """
        process = self._process
        if process is None:
            return False
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            logger.info(
                "Run process group already empty at orphan reap: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False
        except Exception:
            logger.exception(
                "Failed to reap run process group: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        process = self._process
        if process is None or process.returncode is not None:
            return False
        try:
            # POSIX-only by design: production harness runs on Linux, and the
            # subprocess is launched with start_new_session=True to isolate a
            # process group for watchdog cleanup.
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            logger.info(
                "Run process group no longer exists before watchdog signal: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False
        except Exception:
            logger.exception(
                "Failed to signal run process group: session=%s run=%s pid=%s signal=%s",
                self.session.id,
                self.run_record.id,
                process.pid,
                sig.name,
            )
            return False

    async def _publish(self, event: str, data: dict[str, object]) -> Event:
        if event in _ACTIVITY_EVENTS:
            self._last_activity_at = self._clock()
            self._last_activity_event = event
        return await self._event_bus.publish(
            Event(event=event, session_id=self.session.id, run_id=self.run_record.id, data=data)
        )


RUN_QUEUE_MAX_PER_SESSION = 16


@dataclass(frozen=True)
class SubmitResult:
    # ``status`` is "running" or "queued" on accept, None on reject. Reject
    # currently only happens when the per-session queue cap is exceeded —
    # ``reason`` carries the machine-readable code (``"queue_full"``) so
    # callers can map it to a transport-specific failure (api.py → 429).
    accepted: bool
    status: RunStatus | None
    reason: str | None = None


@dataclass(frozen=True)
class _QueuedRun:
    session: Session
    run: Run
    command: ProcessCommand
    on_start: Callable[[], None] | None


class RunManager:
    def __init__(
        self,
        *,
        event_bus: InMemoryEventBus,
        process_factory: ProcessFactory | None = None,
        queue_max_per_session: int = RUN_QUEUE_MAX_PER_SESSION,
        observer: Any | None = None,
        rollout_discovery: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._process_factory = process_factory or AsyncioProcessFactory()
        self._queue_max_per_session = queue_max_per_session
        self._active: dict[str, RunProcess] = {}
        self._tasks: dict[str, asyncio.Task[RunProcessResult]] = {}
        # Tracks which run_id currently owns the subprocess for each session
        # (at most one). New submits for an already-owned session land on the
        # per-session FIFO in ``_queues`` instead of spawning concurrently —
        # which would race on the shared claude-code rollout JSONL.
        self._active_run_by_session: dict[str, str] = {}
        self._queues: dict[str, deque[_QueuedRun]] = {}
        # Optional wiring for the rollout pre-binding path. The production
        # cli wires only ``rollout_discovery`` here — the observer is
        # constructed later inside FastAPI's lifespan and late-bound via
        # ``set_observer`` (see api.py). Tests usually pass both directly.
        # Pre-bind is unconditional in Phase 2 (no env-var gate); the
        # actual work skips itself when ``observer`` is unset.
        self._observer = observer
        self._rollout_discovery = rollout_discovery

    def set_observer(self, observer: Any) -> None:
        """Late binder used by api.py's lifespan: observer is constructed
        when the FastAPI app starts up, after RunManager already exists.
        Idempotent. Called once at most in normal flow."""
        self._observer = observer

    def submit(
        self,
        *,
        session: Session,
        run: Run,
        command: ProcessCommand,
        on_start: Callable[[], None] | None = None,
    ) -> SubmitResult:
        # Decide whether to spawn the run immediately or queue it. Must be
        # called from the event loop thread (it may schedule asyncio tasks);
        # FastAPI's single-loop model satisfies this.
        if session.id not in self._active_run_by_session:
            self._start_now(session, run, command, on_start)
            return SubmitResult(accepted=True, status="running")

        queue = self._queues.setdefault(session.id, deque())
        if len(queue) >= self._queue_max_per_session:
            return SubmitResult(accepted=False, status=None, reason="queue_full")
        queue.append(_QueuedRun(session=session, run=run, command=command, on_start=on_start))
        return SubmitResult(accepted=True, status="queued")

    def start(self, *, session: Session, run: Run, command: ProcessCommand) -> RunProcess:
        # Back-compat shim for callers and tests that pre-date ``submit``. It
        # raises on rejection — pre-queue callers had no concept of "queue full".
        result = self.submit(session=session, run=run, command=command)
        if not result.accepted:
            raise RuntimeError(f"RunManager.start cannot accept run: {result.reason}")
        # When the submit landed on the queue rather than spawning, there is
        # no RunProcess instance yet. Existing call sites only consult the
        # return value in tests; surface a clear error rather than a None.
        run_process = self._active.get(run.id)
        if run_process is None:
            raise RuntimeError(
                "RunManager.start returned a queued submit; callers needing the "
                "RunProcess object must use submit() and handle status='queued'.",
            )
        return run_process

    def drop_queued(self, session_id: str) -> list[str]:
        # Pop every queued entry for ``session_id`` and return their run ids.
        # The caller is responsible for reflecting the drop in the repository
        # (we deliberately don't reach into the repo from here). Idempotent:
        # returns [] when there's nothing queued.
        queue = self._queues.pop(session_id, None)
        if not queue:
            return []
        return [entry.run.id for entry in queue]

    async def interrupt(self, session_id: str, run_id: str) -> bool:
        run_process = self._active.get(run_id)
        if run_process is None or run_process.session.id != session_id:
            return False
        return await run_process.interrupt()

    async def wait(self, run_id: str) -> RunProcessResult:
        task = self._tasks[run_id]
        return await task

    def _start_now(
        self,
        session: Session,
        run: Run,
        command: ProcessCommand,
        on_start: Callable[[], None] | None,
    ) -> RunProcess:
        # ``on_start`` runs before the subprocess is launched so the caller
        # (typically api.py) can flip the run's repo status to "running" and
        # schedule materialization in lock-step with the actual spawn. We log
        # and swallow callback failures rather than abort the spawn — losing
        # a status update is preferable to leaving the user's prompt dropped.
        if on_start is not None:
            try:
                on_start()
            except Exception:
                logger.exception(
                    "RunManager on_start callback raised: session=%s run=%s",
                    session.id,
                    run.id,
                )

        run_process = RunProcess(
            session=session,
            run=run,
            command=command,
            event_bus=self._event_bus,
            process_factory=self._process_factory,
            observer=self._observer,
            rollout_discovery=self._rollout_discovery,
        )
        self._active[run.id] = run_process
        self._active_run_by_session[session.id] = run.id
        task = asyncio.create_task(self._run_and_forget(run_process))
        self._tasks[run.id] = task
        return run_process

    async def _run_and_forget(self, run_process: RunProcess) -> RunProcessResult:
        try:
            return await run_process.run()
        finally:
            run_id = run_process.run_record.id
            session_id = run_process.session.id
            self._active.pop(run_id, None)
            if self._active_run_by_session.get(session_id) == run_id:
                self._active_run_by_session.pop(session_id, None)
            self._spawn_next_queued(session_id)

    def _spawn_next_queued(self, session_id: str) -> None:
        queue = self._queues.get(session_id)
        if not queue:
            return
        nxt = queue.popleft()
        if not queue:
            self._queues.pop(session_id, None)
        self._start_now(nxt.session, nxt.run, nxt.command, nxt.on_start)


def _message_text(message: Message) -> str:
    text = "\n".join(block.text for block in message.blocks if isinstance(block, TextBlock))
    if not text:
        raise ValueError("Message must include at least one text block for CLI launch")
    return text


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
