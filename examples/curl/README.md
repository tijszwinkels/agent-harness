# curl Quickstart

Start the service:

```bash
uv run agent-harness serve \
  --host 0.0.0.0 \
  --port 8876 \
  --database .agent-harness.db
```

Create a session:

```bash
curl -sS http://127.0.0.1:8876/v1/sessions \
  -H 'content-type: application/json' \
  -d '{
    "backend": "codex",
    "model": "gpt-5.4-mini",
    "project": {"path": "/workspace/demo", "name": "demo"},
    "title": "curl quickstart"
  }'
```

Start a run:

```bash
curl -sS http://127.0.0.1:8876/v1/sessions/SESSION_ID/runs \
  -H 'content-type: application/json' \
  -d '{"message": "Say hello from the quickstart"}'
```

List sessions:

```bash
curl -sS http://127.0.0.1:8876/v1/sessions
```

Stream events from the beginning of this process:

```bash
curl -N 'http://127.0.0.1:8876/v1/events?from=beginning'
```

To exercise the real subprocess adapter, start the service with `--execute-runs`. That may invoke local Codex or Claude Code credentials, so the basic quickstart leaves it off.
