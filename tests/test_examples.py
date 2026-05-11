from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_example(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observer_external_builds_codex_demo_transcript(tmp_path: Path) -> None:
    example = _load_example("examples/python/observer_external.py")

    transcript = example.codex_demo_transcript_path(tmp_path)
    lines = example.demo_transcript_lines(cwd="/workspace/demo", model="gpt-5.4-mini")

    assert transcript.as_posix().endswith(
        "/.codex/sessions/2026/05/08/"
        "rollout-2026-05-08T10-30-00-123e4567-e89b-12d3-a456-426614174000.jsonl"
    )
    assert [json.loads(line)["type"] for line in lines] == ["turn_context", "event_msg", "event_msg"]
    assert json.loads(lines[0])["payload"]["cwd"] == "/workspace/demo"


def test_simple_client_builds_v1_payloads() -> None:
    example = _load_example("examples/python/simple_client.py")

    payload = example.create_session_payload(
        backend="codex",
        model="gpt-5.4-mini",
        project_path="/workspace/demo",
        title="Example session",
    )

    assert payload == {
        "backend": "codex",
        "model": "gpt-5.4-mini",
        "project": {"path": "/workspace/demo", "name": "demo"},
        "title": "Example session",
    }
