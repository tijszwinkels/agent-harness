from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent_harness.codex_names import INDEX_FILE_NAME


class ObserverConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ObserverSettings:
    roots: tuple[Path, ...] = ()
    # Explicit codex name index (``session_index.jsonl``). ``None`` derives
    # it from an observed codex sessions root; see ``codex_name_index_path``.
    codex_name_index: Path | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.roots)

    def codex_name_index_path(self) -> Path | None:
        """The codex name index to read, or ``None`` when codex isn't observed.

        Codex keeps thread names in ``$CODEX_HOME/session_index.jsonl``,
        beside its ``sessions`` root. The index is read only alongside an
        observed codex root (``<CODEX_HOME>/sessions``, recognized as a
        ``.codex`` directory or ``$CODEX_HOME``), so disabling codex
        observation disables it too. ``codex_name_index`` overrides the
        derivation for custom homes/roots.
        """
        if self.codex_name_index is not None:
            return self.codex_name_index
        codex_home = os.environ.get("CODEX_HOME")
        for root in self.roots:
            if root.name != "sessions":
                continue
            parent = root.parent
            if parent.name == ".codex" or (
                codex_home and parent == Path(codex_home).expanduser()
            ):
                return parent / INDEX_FILE_NAME
        return None

    @classmethod
    def from_roots(
        cls,
        roots: Iterable[str | Path],
        *,
        codex_name_index: str | Path | None = None,
    ) -> ObserverSettings:
        normalized: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            path = Path(root).expanduser()
            if path in seen:
                continue
            normalized.append(path)
            seen.add(path)
        return cls(
            roots=tuple(normalized),
            codex_name_index=Path(codex_name_index).expanduser() if codex_name_index else None,
        )

    @classmethod
    def default_transcript_roots(cls, *, home: str | Path | None = None) -> ObserverSettings:
        home_path = Path(home).expanduser() if home is not None else Path.home()
        return cls.from_roots(
            [
                home_path / ".claude" / "projects",
                home_path / ".codex" / "sessions",
                home_path / ".pi" / "agent" / "sessions",
            ]
        )

    @classmethod
    def existing_default_transcript_roots(cls, *, home: str | Path | None = None) -> ObserverSettings:
        settings = cls.default_transcript_roots(home=home)
        return cls.from_roots(root for root in settings.roots if root.exists())

    def validate(self) -> None:
        missing_roots = [root for root in self.roots if not root.exists()]
        if missing_roots:
            formatted = ", ".join(str(root) for root in missing_roots)
            raise ObserverConfigurationError(f"Observer root does not exist: {formatted}")
