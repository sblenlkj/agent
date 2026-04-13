from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PathConfig:
    root: Path
    data_dir: Path
    output_dir: Path
    pyproject_path: Path
    run_path: Path
    env_path: Path

    @classmethod
    def from_project_root(
        cls,
        *,
        root: Path | None = None
    ) -> "PathConfig":
        """
        Build a singleton-like path config from project root.

        Expected file location:
            src/common/paths.py

        Therefore:
            Path(__file__).resolve().parents[2] -> project root
        """
        resolved_root = root or Path(__file__).resolve().parents[2]

        return cls(
            root=resolved_root,
            data_dir=resolved_root / "data",
            output_dir=resolved_root / "output",
            pyproject_path=resolved_root / "pyproject.toml",
            run_path=resolved_root / "run.py",
            env_path=resolved_root / ".env",
        )

PATHS = PathConfig.from_project_root()