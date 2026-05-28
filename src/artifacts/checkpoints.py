from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.factory import build_model


def format_round_dir(round_idx: int) -> str:
    """Format a federated round index as a checkpoint directory name."""
    round_idx = int(round_idx)

    if round_idx < 0:
        raise ValueError("round_idx must be >= 0.")

    return f"round_{round_idx:03d}"


def format_client_checkpoint(client_id: int) -> str:
    """Format a client ID as a checkpoint filename."""
    client_id = int(client_id)

    if client_id < 0:
        raise ValueError("client_id must be >= 0.")

    return f"client_{client_id:03d}.pt"


def get_checkpoint_path(
    run_dir: str | Path,
    round_idx: int,
    client_id: int,
) -> Path:
    """
    Return the checkpoint path for a given run, round, and client.

    Expected layout:
        runs/<run_id>/checkpoints/round_003/client_000.pt
    """
    run_dir = Path(run_dir)
    ckpt_path = (
        run_dir
        / "checkpoints"
        / format_round_dir(round_idx)
        / format_client_checkpoint(client_id)
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


def list_available_rounds(run_dir: str | Path) -> list[int]:
    """
    List available checkpoint rounds for a run directory.

    The function scans:
        runs/<run_id>/checkpoints/round_001/
        runs/<run_id>/checkpoints/round_002/
        ...

    Returns:
        Sorted list of integer round indices.
    """
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    rounds: list[int] = []

    for path in checkpoints_dir.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if not name.startswith("round_"):
            continue

        try:
            round_idx = int(name.replace("round_", ""))
        except ValueError:
            continue

        rounds.append(round_idx)

    rounds = sorted(rounds)

    if not rounds:
        raise FileNotFoundError(
            f"No round checkpoint directories found in: {checkpoints_dir}"
        )

    return rounds


def resolve_rounds_to_analyze(
    run_dir: str | Path,
    default_round: int,
    requested_round: int | None = None,
    all_rounds: bool = False,
) -> list[int]:
    """
    Resolve which rounds should be analyzed.

    Behavior:
        - default: use the final/default round from the config.
        - --round N: analyze only round N.
        - --all-rounds: analyze all available checkpoint rounds.
    """
    if requested_round is not None and all_rounds:
        raise ValueError("Use either --round or --all-rounds, not both.")

    available_rounds = list_available_rounds(run_dir)

    if all_rounds:
        return available_rounds

    if requested_round is not None:
        requested_round = int(requested_round)

        if requested_round not in available_rounds:
            raise FileNotFoundError(
                f"Requested round {requested_round} was not found in checkpoints. "
                f"Available rounds: {available_rounds}"
            )

        return [requested_round]

    default_round = int(default_round)

    if default_round not in available_rounds:
        raise FileNotFoundError(
            f"Default round {default_round} from config was not found in checkpoints. "
            f"Available rounds: {available_rounds}"
        )

    return [default_round]


def load_model_from_checkpoint(
    config: dict,
    input_dim: int,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """
    Rebuild a model from config and load a saved state_dict checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        config=config,
        input_dim=int(input_dim),
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model