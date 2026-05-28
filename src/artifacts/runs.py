from __future__ import annotations

from pathlib import Path


def find_latest_run(prefix: str, base_dir: str | Path = "runs") -> Path:
    """
    Find the latest run directory matching a prefix.

    Example:
        prefix="fiveg_nidd_v1_"
        returns latest directory under runs/fiveg_nidd_v1_*
    """
    runs_path = Path(base_dir)

    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_path}")

    candidates = [
        path
        for path in runs_path.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No run directories found in {runs_path}/ with prefix '{prefix}'"
        )

    return sorted(candidates)[-1]