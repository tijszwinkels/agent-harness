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
A message body is at most 30,000 characters and may not contain NUL: it is
handed to the worker's CLI as one argument. Put larger content in a file and
send its path.

Permissions: a headless worker cannot answer permission prompts, but a spawn
must not widen what its spawner may do. By default a worker **inherits** the
permission bypass of its spawner when the spawner is a harness session, and is
restricted otherwise (an interactive session, or no known spawner). Pass
`--bypass-permissions` or `--no-bypass-permissions` to choose explicitly. A
task launcher that starts the first worker usually passes
`--bypass-permissions` once; its descendants inherit it.

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
  once). A batch is capped at 96 KiB, counting each message's rendered header,
  so it fits one CLI argument; the rest is delivered in the following turn.
  If storage fails before the delivery run exists, the batch returns to
  pending for the next trigger.
* **Interactive (`origin = external`)** — sessions the harness only observes
  from their transcript files, typically a person typing in the agent's own
  TUI. The harness never starts a turn there. Mail is stored as `unread` and
  read with `worker inbox`.

Delivery also runs at harness startup, so mail sent while the harness was down
or not executing runs (no `--execute-runs`) is delivered when it can be.

Durability: with `--database`, mail is stored in SQLite and survives restarts.
A message is durably claimed before its run starts and is never handed out
again: not after a restart, not by a second harness process on the same
database (claims are conditional updates). If the harness dies between the
claim and the process start, that message is lost rather than delivered twice
(at most once, not exactly once).

Codex workers resume by the thread id the transcript observer binds from their
first run, so the harness must observe the Codex sessions root (the default
`serve` does; with `--no-observer` add `--observe-root <CODEX_HOME>/sessions`).
The observer recognizes Codex transcripts only under a `.codex/sessions` path,
so a custom `CODEX_HOME` must itself be a directory named `.codex`.
Until that id is bound, mail for a Codex worker that already ran stays
pending instead of starting a new conversation; it is delivered as soon as the
observer announces the id.

Known limits:

* A harness-origin session that someone has also opened in a TUI
  (`claude --resume …`) is still treated as turn-based.
* Mail to an archived session stays pending.
* Crash recovery: if the harness process is killed hard while a worker run is
  in progress, that worker's CLI process can survive (it runs in its own
  process group). On restart the harness marks the run failed and may resume
  the same conversation for pending mail while the survivor still runs. This
  is the harness's existing restart behaviour for every run, not specific to
  messaging.
* A Codex worker whose first run never produced a rollout keeps its mail
  pending.
* A worker that should be woken must have been launched by this harness. An
  agent started outside it (e.g. a Codex TUI) is interactive by this rule.

## Task launcher: `up` / `down`

```bash
agent-harness up --port 8000        # prints {"url", "started", "task", "pid", ...}
export AGENT_HARNESS_URL=http://127.0.0.1:8000
… run the task (agents call agent-harness worker …) …
agent-harness down --port 8000 --task <task printed by up>
```

* `up` reuses a harness already answering on `127.0.0.1:PORT`, or starts
  `agent-harness serve --host 127.0.0.1 --execute-runs --database
  <state-dir>/harness-PORT.db` detached and waits until it answers. It always
  binds loopback. State lives in `$XDG_STATE_HOME/agent-harness`
  (`--state-dir`).
  Forwarded `--serve-arg` options cannot override the loopback bind, port or
  database. Reuse requires a harness that executes runs and has worker
  messaging and was started bound to loopback only (`/v1/mailbox/status`
  reports `loopback_only`); anything else on the port is an error. A new
  harness is started only when the port refuses connections: a timeout or
  other unclear answer is reported, never treated as a free port. `up` and
  `down` for one port take the same lock file. The default state directory
  is kept 0700 and the launcher's own state, log and default database files
  0600, including files left by an earlier version; an operator-supplied
  `--state-dir` or `--database` keeps its permissions.
* `up` prints a `task` id (random unless `--task` is given). `down --task ID`
  stops the harness **only if `up` with that task started it**. The pid is
  checked against its recorded kernel start time before every signal; without
  `/proc` (e.g. macOS) `down` refuses to signal at all. Before stopping it
  waits until no deliverable mail is pending and no runs are active
  (`--timeout`, default 600 s). If the harness is still busy or does not
  answer, it refuses unless `--force`. A reused harness keeps running and
  keeps delivering. Stopping a harness that another task reuses ends that
  task's deliveries too; the stored mail is delivered when it starts again.

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
