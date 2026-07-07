from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ObserverConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ObserverSettings:
    roots: tuple[Path, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.roots)

    @classmethod
    def from_roots(cls, roots: Iterable[str | Path]) -> ObserverSettings:
        normalized: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            path = Path(root).expanduser()
            if path in seen:
                continue
            normalized.append(path)
            seen.add(path)
        return cls(roots=tuple(normalized))

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
