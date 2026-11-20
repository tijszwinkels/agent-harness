"""Worker messaging helpers shared by the API and the worker CLI.

The harness exposes a per-session mailbox (``/v1/sessions/{id}/mailbox``).
This module holds the pieces both sides agree on: the environment variables a
harness-launched run receives, how a caller finds its own address, and how
pending mail is rendered into the user turn that resumes a worker.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

from agent_harness.models import MailMessage, Session

# Set on every harness-launched run (see ``api._create_and_launch_run``).
ENV_URL = "AGENT_HARNESS_URL"
ENV_SESSION = "AGENT_HARNESS_SESSION"
DEFAULT_URL = "http://127.0.0.1:8000"

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")


def harness_env(*, session_id: str, url: str | None) -> dict[str, str]:
    env = {ENV_SESSION: session_id}
    if url:
        env[ENV_URL] = url
    return env


def self_address(env: Mapping[str, str] | None = None) -> str | None:
    """The calling session's address, from its environment.

    1. ``AGENT_HARNESS_SESSION`` — set by the harness on every run it launches
       (turn-based workers, chat-driven sessions).
    2. ``CLAUDE_CODE_SESSION_ID`` — set by Claude Code for its own tool calls,
       including an interactive TUI. The harness observer files a Claude
       conversation ``<uuid>`` as ``ses_<uuid hex>``.

    Codex and pi TUIs expose no such variable; they pass ``--as``.
    """
    env = os.environ if env is None else env
    harness = env.get(ENV_SESSION, "").strip()
    if harness:
        return harness
    claude = env.get("CLAUDE_CODE_SESSION_ID", "").strip()
    if claude and _UUID_RE.match(claude):
        return "ses_" + claude.replace("-", "").lower()
    return None


def harness_url(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(ENV_URL, "").strip() or DEFAULT_URL


def format_delivery_turn(session: Session, mail: Sequence[MailMessage]) -> str:
    """Render pending mail as the next user turn of a resumed worker."""
    count = len(mail)
    lines = [
        f"[agent-harness] {count} message{'s' if count != 1 else ''} for you "
        f"(your address: {session.id}).",
    ]
    for item in mail:
        sender = item.sender or "anonymous"
        lines += ["", f"--- message {item.id} from {sender} ---", item.body.rstrip()]
    senders = list(dict.fromkeys(m.sender for m in mail if m.sender))
    lines.append("")
    if senders:
        lines.append(
            "To reply, run: agent-harness worker send "
            f"{senders[-1]} \"<your reply>\"  (returns at once; then end your turn — "
            "a reply to you resumes this conversation)."
        )
    else:
        lines.append("The sender left no reply address.")
    return "\n".join(lines)
