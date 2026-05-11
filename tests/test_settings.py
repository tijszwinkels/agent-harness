from pathlib import Path

import pytest

from agent_harness.settings import ObserverConfigurationError, ObserverSettings


def test_observer_settings_default_to_disabled() -> None:
    settings = ObserverSettings()

    assert settings.roots == ()
    assert not settings.enabled


def test_observer_settings_builds_default_transcript_roots(tmp_path) -> None:
    settings = ObserverSettings.default_transcript_roots(home=tmp_path)

    assert settings.roots == (
        tmp_path / ".claude" / "projects",
        tmp_path / ".codex" / "sessions",
    )
    assert settings.enabled


def test_observer_settings_normalize_and_deduplicate_roots(tmp_path) -> None:
    root = tmp_path / "transcripts"

    settings = ObserverSettings.from_roots([root, Path(root)])

    assert settings.roots == (root,)


def test_observer_settings_validate_configured_roots_exist(tmp_path) -> None:
    missing = tmp_path / "missing"
    settings = ObserverSettings.from_roots([missing])

    with pytest.raises(ObserverConfigurationError, match="Observer root does not exist"):
        settings.validate()
