from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from _path import add_project_root_to_path

add_project_root_to_path()

from src.artifacts.checkpoints import (
    get_checkpoint_path,
    load_model_from_checkpoint,
    resolve_rounds_to_analyze,
)
from src.artifacts.runs import find_latest_run
from src.config import load_config
from src.data.loading import load_dataset_from_config
from src.data.partition import transform_features
from src.training.eval import evaluate_model
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved client checkpoint on the global test set.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nsl_kdd.yaml",
        help="Path to the experiment YAML configuration file.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Federated round to evaluate. Example: --round 1",
    )
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Evaluate all available checkpoint rounds.",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client checkpoint to evaluate. Default: 0.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit run directory. "
            "If omitted, the latest run matching the dataset prefix is used."
        ),
    )
    return parser.parse_args()


def evaluate_checkpoint_round(
    config: dict[str, Any],
    data,
    run_dir: Path,
    round_idx: int,
    client_id: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Load one client checkpoint for one round and evaluate it on the global test set.
    """
    print("\n" + "=" * 90)
    print(f"CHECKPOINT EVALUATION | round={round_idx:03d} | client={client_id:03d}")

    checkpoint_path = get_checkpoint_path(
        run_dir=run_dir,
        round_idx=round_idx,
        client_id=client_id,
    )

    print("Checkpoint path:", checkpoint_path)

    model = load_model_from_checkpoint(
        config=config,
        input_dim=int(data.input_dim),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    X_test = transform_features(
        data.X_test_raw,
        data.transformer,
    )
    y_test = data.y_test.to_numpy()

    metrics = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=int(config["training"]["batch_size"]),
    )

    print(
        f"loss={metrics['loss']:.6f} "
        f"accuracy={metrics['accuracy']:.6f}"
    )

    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["runtime"]["seed"])
    device = get_device(config["runtime"]["device"])

    loaded_dataset = load_dataset_from_config(config)
    data = loaded_dataset.data

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(
            prefix=loaded_dataset.run_prefix,
            base_dir=config["artifacts"]["run_dir"],
        )

    default_round = int(config["federated"]["num_rounds"])

    rounds_to_analyze = resolve_rounds_to_analyze(
        run_dir=run_dir,
        default_round=default_round,
        requested_round=args.round,
        all_rounds=args.all_rounds,
    )

    print("Dataset:", config["dataset"]["name"])
    print("Run directory:", run_dir)
    print("Rounds to evaluate:", rounds_to_analyze)
    print("Client ID:", int(args.client_id))
    print("Device:", device)

    for round_idx in rounds_to_analyze:
        evaluate_checkpoint_round(
            config=config,
            data=data,
            run_dir=run_dir,
            round_idx=round_idx,
            client_id=int(args.client_id),
            device=device,
        )


if __name__ == "__main__":
    main()