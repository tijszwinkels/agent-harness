# Agent Harness Examples

These examples assume the service is running locally.

```bash
uv run agent-harness serve \
  --host 0.0.0.0 \
  --port 8876 \
  --database .agent-harness.db
```

The service automatically observes existing Claude Code and Codex transcript roots under `HOME`. Use `--observe-root PATH` only when you want to add a specific directory, such as the safe fake transcript root used by `python/observer_external.py`.

Use the Tailscale URL from another machine:

```text
http://pillar.tail72f2bc.ts.net:8876
```

## Examples

- `curl/README.md` shows the v1 API with plain curl.
- `python/simple_client.py` creates a session, starts a run, and reads the session back.
- `python/observer_external.py` demonstrates the external transcript observer with a safe fake Codex transcript.

The fake transcript has the same path shape and small JSONL records that the observer expects. It exists only for repeatable demos and tests; production observation watches real Claude Code and Codex transcript roots.
