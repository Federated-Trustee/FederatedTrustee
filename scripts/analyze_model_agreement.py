from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _path import add_project_root_to_path

add_project_root_to_path()

from src.analysis.agreement import (
    compute_agreement_matrix,
    compute_mean_agreement,
    save_agreement_outputs,
)
from src.artifacts.checkpoints import (
    get_checkpoint_path,
    load_model_from_checkpoint,
    resolve_rounds_to_analyze,
)
from src.artifacts.runs import find_latest_run
from src.config import load_config
from src.data.loading import load_dataset_from_config
from src.data.partition import transform_features
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prediction agreement between local client models.",
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
        help="Federated round to analyze. Example: --round 1",
    )
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Analyze all available checkpoint rounds.",
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


def predict_labels(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict class labels for a dense feature matrix."""
    predictions: list[np.ndarray] = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            end = start + int(batch_size)

            batch_X = torch.as_tensor(
                X[start:end],
                dtype=torch.float32,
                device=device,
            )

            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.append(preds)

    return np.concatenate(predictions, axis=0)


def print_agreement_matrix(
    matrix: np.ndarray,
    client_ids: list[int],
) -> None:
    """Print the pairwise agreement matrix in a compact text format."""
    header = "      " + " ".join([f"C{int(cid):02d}" for cid in client_ids])
    print(header)

    for i, client_id in enumerate(client_ids):
        row = " ".join(
            [f"{matrix[i, j]:.4f}" for j in range(len(client_ids))]
        )
        print(f"C{int(client_id):02d}  {row}")


def run_model_agreement_for_round(
    config: dict[str, Any],
    data,
    run_dir: Path,
    round_idx: int,
    num_clients: int,
    device: torch.device,
) -> Path:
    """
    Compute agreement between all local client models for one federated round.
    """
    print("\n" + "=" * 90)
    print(f"LOCAL MODEL AGREEMENT | round={round_idx:03d}")

    X_ref = transform_features(
        data.X_test_raw,
        data.transformer,
    )
    print("Reference set shape:", X_ref.shape)

    predictions: dict[int, np.ndarray] = {}

    for client_id in range(int(num_clients)):
        checkpoint_path = get_checkpoint_path(
            run_dir=run_dir,
            round_idx=round_idx,
            client_id=client_id,
        )

        model = load_model_from_checkpoint(
            config=config,
            input_dim=int(data.input_dim),
            checkpoint_path=checkpoint_path,
            device=device,
        )

        preds = predict_labels(
            model=model,
            X=X_ref,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )

        predictions[int(client_id)] = preds

        print(
            f"Loaded client {client_id:03d} "
            f"from {checkpoint_path.name} "
            f"-> predictions shape: {preds.shape}"
        )

    client_ids, agreement_matrix = compute_agreement_matrix(predictions)

    print("\nAgreement matrix:")
    print_agreement_matrix(agreement_matrix, client_ids)

    summary_df = compute_mean_agreement(
        matrix=agreement_matrix,
        client_ids=client_ids,
    )

    print("\nMean agreement ranking:")
    for _, row in summary_df.iterrows():
        print(
            f"Client {int(row['client_id']):03d}: "
            f"mean={row['mean_agreement']:.6f} "
            f"min={row['min_agreement']:.6f} "
            f"max={row['max_agreement']:.6f}"
        )

    out_dir = save_agreement_outputs(
        run_dir=run_dir,
        round_idx=round_idx,
        client_ids=client_ids,
        matrix=agreement_matrix,
        summary_df=summary_df,
        output_group="local_models",
        extra_metadata={
            "dataset": config["dataset"]["name"],
            "attack_enabled": config.get("attack", {}).get("enabled", False),
            "attack_type": config.get("attack", {}).get("type"),
            "malicious_client_ids": config.get("attack", {}).get(
                "malicious_client_ids",
                [],
            ),
            "reference_size": int(X_ref.shape[0]),
            "source": "local_models",
        },
    )

    print(f"\nSaved local model agreement outputs to: {out_dir}")

    return out_dir


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
    num_clients = int(config["federated"]["num_clients"])

    rounds_to_analyze = resolve_rounds_to_analyze(
        run_dir=run_dir,
        default_round=default_round,
        requested_round=args.round,
        all_rounds=args.all_rounds,
    )

    print("Dataset:", config["dataset"]["name"])
    print("Run directory:", run_dir)
    print("Rounds to analyze:", rounds_to_analyze)
    print("Number of clients:", num_clients)
    print("Device:", device)

    for round_idx in rounds_to_analyze:
        run_model_agreement_for_round(
            config=config,
            data=data,
            run_dir=run_dir,
            round_idx=round_idx,
            num_clients=num_clients,
            device=device,
        )


if __name__ == "__main__":
    main()