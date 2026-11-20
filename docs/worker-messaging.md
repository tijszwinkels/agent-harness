# Worker messaging (one machine)

Coding agents (Claude Code, Codex, pi) can delegate to each other through
three calls that are the same for every backend. Delivery goes through the
local harness; no chat frontend, terminal multiplexer or agent hook is
involved.

```bash
# In the implementer (any agent's shell):
agent-harness worker spawn --backend codex --model gpt-6.2-astra --effort high \
    "Review the diff on this branch; reply with findings."
# -> {"address": "ses_9c…", "status": "delivered", "reply_to": "ses_41…", ...}
# The implementer now ends its turn; its process exits.

# Later, in the reviewer:
agent-harness worker send ses_41… "Two findings: …"
# -> the implementer's conversation is resumed in a new process with that text.

# In an interactive TUI session (never resumed):
agent-harness worker inbox
```

## The calls

| Call | What it does | Returns |
|---|---|---|
| `worker spawn --backend B [--model M] [--effort E] [--cwd DIR] [--title T] BRIEF` | Creates a **fresh** harness session (not a fork) and sends it `BRIEF` as its first message. | At once, with the new `address`. |
| `worker send ADDRESS TEXT` | Stores a message for `ADDRESS`. | At once, once stored: `delivered` (a run is starting), `waiting` (recipient busy), or `stored` (interactive recipient). |
| `worker inbox [--peek] [--all] [--json]` | Lists unread stored messages for you and marks them read (`--peek`: don't). | The messages. |
| `worker whoami` | Prints your address and the harness URL. | |

`BRIEF`/`TEXT` may be `-` to read stdin; `--brief-file`/`--file` read a file.
Spawned workers run with the backend's permission bypass (a headless worker
cannot answer prompts); `--no-bypass-permissions` turns that off.

## Addresses

An address is the harness session id (`ses_<32 hex>`, or a legacy observed id
such as `codex_<uuid>`). It is backend-neutral. The form `<id>@<host>` is
reserved for cross-machine routing and is refused for now.

A caller's own address (used as the reply-to of `spawn`/`send` and as the
mailbox of `inbox`) comes from, in order:

1. `--as ADDRESS`;
2. `AGENT_HARNESS_SESSION`, which the harness sets on **every** run it
   launches (together with `AGENT_HARNESS_URL`);
3. `CLAUDE_CODE_SESSION_ID`, which Claude Code sets for its own tools, also in
   an interactive TUI (mapped to the observer's `ses_<hex>` id).

Interactive Codex and pi sessions expose no such variable and pass `--as`
(their observed id is listed by `GET /v1/sessions`).

The harness URL comes from `--url`, else `AGENT_HARNESS_URL`, else
`http://127.0.0.1:8000`.

## Delivery rule: turn-based vs interactive

The harness decides from what it knows, not from process sniffing:

* **Turn-based (`origin = harness`)** — sessions the harness launched (spawned
  workers, chat-driven sessions). Their CLI process exits after each turn.
  Incoming mail is `pending`; when the session has no queued or running run,
  the harness resumes the **same conversation in a new process** with all
  pending mail as one user turn (oldest first, each with its sender and a
  reply hint). If the session is busy, the mail waits; when that run ends the
  harness delivers it. Each message is claimed by exactly one run (at most
  once).
* **Interactive (`origin = external`)** — sessions the harness only observes
  from their transcript files, typically a person typing in the agent's own
  TUI. The harness never starts a turn there. Mail is stored as `unread` and
  read with `worker inbox`.

Delivery also runs at harness startup, so mail sent while the harness was down
or not executing runs (no `--execute-runs`) is delivered when it can be.

Durability: with `--database`, mail is stored in SQLite and survives restarts.
A message already handed to a run is never handed out again, even if that run
died with the harness (at most once, not exactly once).

Known limits:

* A harness-origin session that someone has also opened in a TUI
  (`claude --resume …`) is still treated as turn-based.
* Mail to an archived session stays pending.
* A worker that should be woken must have been launched by this harness. An
  agent started outside it (e.g. a Codex TUI) is interactive by this rule.

## Task launcher: `up` / `down`

```bash
agent-harness up --port 8000        # prints {"url", "started", "pid", ...}
export AGENT_HARNESS_URL=http://127.0.0.1:8000
… run the task (agents call agent-harness worker …) …
agent-harness down --port 8000
```

* `up` reuses a harness already answering on `127.0.0.1:PORT`, or starts
  `agent-harness serve --host 127.0.0.1 --execute-runs --database
  <state-dir>/harness-PORT.db` detached and waits until it answers. It always
  binds loopback. State lives in `$XDG_STATE_HOME/agent-harness`
  (`--state-dir`).
* `down` stops the harness **only if `up` with the same `--task` started it**
  (the recorded pid is checked against its command line). Before stopping it
  waits until no deliverable mail is pending and no runs are active
  (`--timeout`, default 600 s); if still busy it refuses unless `--force`.
  A reused harness keeps running and keeps delivering.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/sessions/{id}/mailbox` | Send (`{"body", "sender"?}`) → `{message, status, run_id}`. |
| GET | `/v1/sessions/{id}/mailbox?state=unread` | List mail (filter repeatable). |
| POST | `/v1/sessions/{id}/mailbox/ack` | Mark unread mail read (`{"ids": [...]}`). |
| GET | `/v1/mailbox/status` | `{pending, active_runs, executes_runs}`. |

A `mail.received` event is published on the event stream for every message.

## Follow-up: other machines

Not built here. The pieces meant to carry over: an address can grow an
`@host` part; messages are self-contained records (`id`, `to`, `sender`,
`body`, `created_at`) that an outbox/relay can forward and a remote harness
can insert with the same `POST …/mailbox`; delivery state stays local to the
recipient's harness.
