from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch


def create_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """
    Create a timestamped run directory.

    Example:
        runs/nsl_kdd_v1_20260528_153012
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: str | Path) -> None:
    """Save a Python object as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_model_checkpoint(
    model_state_dict: Mapping[str, Any],
    run_dir: str | Path,
    round_idx: int,
    client_id: int,
) -> Path:
    """
    Save a client model checkpoint for a given federated round.

    The checkpoint path follows:

        runs/<run_id>/checkpoints/round_003/client_000.pt
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"round_{int(round_idx):03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"client_{int(client_id):03d}.pt"
    torch.save(dict(model_state_dict), ckpt_path)

    return ckpt_path