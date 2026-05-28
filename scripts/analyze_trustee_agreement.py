from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _path import add_project_root_to_path

add_project_root_to_path()

from src.analysis.agreement import (
    compute_agreement_matrix,
    compute_mean_agreement,
    save_agreement_outputs,
)
from src.analysis.trustee import extract_trustee_surrogate, trustee_result_to_row
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
        description="Extract TRUSTEE trees and compute agreement between pruned surrogates.",
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
        help="Federated round to analyze. Example: --round 1.",
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
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="TRUSTEE num_iter. If omitted, uses config['trustee']['num_iter'].",
    )
    parser.add_argument(
        "--num-stability-iter",
        type=int,
        default=None,
        help=(
            "TRUSTEE num_stability_iter. "
            "If omitted, uses config['trustee']['num_stability_iter']."
        ),
    )
    parser.add_argument(
        "--samples-size",
        type=float,
        default=None,
        help="TRUSTEE samples_size. If omitted, uses config['trustee']['samples_size'].",
    )
    return parser.parse_args()


def get_trustee_params(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trustee_cfg = config.get("trustee", {})

    return {
        "num_iter": int(
            args.num_iter
            if args.num_iter is not None
            else trustee_cfg.get("num_iter", 20)
        ),
        "num_stability_iter": int(
            args.num_stability_iter
            if args.num_stability_iter is not None
            else trustee_cfg.get("num_stability_iter", 5)
        ),
        "samples_size": float(
            args.samples_size
            if args.samples_size is not None
            else trustee_cfg.get("samples_size", 0.3)
        ),
    }


def save_trustee_outputs(
    run_dir: str | Path,
    round_idx: int,
    trustee_df: pd.DataFrame,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save per-client TRUSTEE extraction metrics."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "trustee" / f"round_{int(round_idx):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trustee_df.to_csv(out_dir / "trustee_summary.csv", index=False)

    metadata: dict[str, Any] = {
        "round_idx": int(round_idx),
        "num_clients": int(len(trustee_df)),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return out_dir


def print_agreement_matrix(
    matrix: np.ndarray,
    client_ids: list[int],
) -> None:
    """Print the pairwise surrogate agreement matrix."""
    header = "      " + " ".join([f"C{int(cid):02d}" for cid in client_ids])
    print(header)

    for i, client_id in enumerate(client_ids):
        row = " ".join(
            [f"{matrix[i, j]:.4f}" for j in range(len(client_ids))]
        )
        print(f"C{int(client_id):02d}  {row}")


def run_trustee_agreement_for_round(
    config: dict[str, Any],
    data,
    run_dir: Path,
    round_idx: int,
    num_clients: int,
    device: torch.device,
    trustee_params: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Extract TRUSTEE trees for all clients in one round and compute agreement
    between the pruned surrogate trees.
    """
    print("\n" + "=" * 90)
    print(f"TRUSTEE AGREEMENT | round={round_idx:03d}")

    X_ref = transform_features(
        data.X_test_raw,
        data.transformer,
    )
    y_true = data.y_test.to_numpy()

    print("Reference set shape:", X_ref.shape)
    print("TRUSTEE params:", trustee_params)

    tree_predictions: dict[int, np.ndarray] = {}
    trustee_rows: list[dict[str, Any]] = []

    for client_id in range(int(num_clients)):
        print("\n" + "=" * 90)
        print(f"ROUND {round_idx:03d} | CLIENT {client_id:03d}")

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

        result = extract_trustee_surrogate(
            model=model,
            X_ref=X_ref,
            y_true=y_true,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            num_iter=int(trustee_params["num_iter"]),
            num_stability_iter=int(trustee_params["num_stability_iter"]),
            samples_size=float(trustee_params["samples_size"]),
            verbose=False,
        )

        tree_predictions[int(client_id)] = result.pruned_predictions

        trustee_rows.append(
            trustee_result_to_row(
                result=result,
                client_id=client_id,
                checkpoint_path=checkpoint_path,
            )
        )

        print(
            f"teacher_acc={result.teacher_accuracy:.6f} | "
            f"pruned_fidelity={result.pruned_fidelity:.6f} | "
            f"pruned_acc={result.pruned_accuracy:.6f} | "
            f"pruned_depth={result.pruned_depth} | "
            f"pruned_leaves={result.pruned_leaves}"
        )

    trustee_df = (
        pd.DataFrame(trustee_rows)
        .sort_values("client_id")
        .reset_index(drop=True)
    )

    print("\nTRUSTEE summary:")
    print(
        trustee_df[
            [
                "client_id",
                "teacher_accuracy",
                "pruned_fidelity",
                "pruned_accuracy",
                "pruned_depth",
                "pruned_leaves",
            ]
        ].to_string(index=False)
    )

    client_ids, agreement_matrix = compute_agreement_matrix(tree_predictions)

    print("\nAgreement matrix between pruned TRUSTEE trees:")
    print_agreement_matrix(agreement_matrix, client_ids)

    summary_df = compute_mean_agreement(
        matrix=agreement_matrix,
        client_ids=client_ids,
    )

    print("\nMean agreement ranking - pruned TRUSTEE trees:")
    for _, row in summary_df.iterrows():
        print(
            f"Client {int(row['client_id']):03d}: "
            f"mean={row['mean_agreement']:.6f} "
            f"min={row['min_agreement']:.6f} "
            f"max={row['max_agreement']:.6f}"
        )

    common_metadata = {
        "dataset": config["dataset"]["name"],
        "reference_size": int(X_ref.shape[0]),
        "attack_enabled": config.get("attack", {}).get("enabled", False),
        "attack_type": config.get("attack", {}).get("type"),
        "malicious_client_ids": config.get("attack", {}).get(
            "malicious_client_ids",
            [],
        ),
        "trustee_params": trustee_params,
    }

    trustee_out_dir = save_trustee_outputs(
        run_dir=run_dir,
        round_idx=round_idx,
        trustee_df=trustee_df,
        extra_metadata=common_metadata,
    )

    agreement_out_dir = save_agreement_outputs(
        run_dir=run_dir,
        round_idx=round_idx,
        client_ids=client_ids,
        matrix=agreement_matrix,
        summary_df=summary_df,
        output_group="trustee_pruned_trees",
        extra_metadata={
            **common_metadata,
            "source": "trustee_pruned_trees",
        },
    )

    print(f"\nSaved TRUSTEE outputs to: {trustee_out_dir}")
    print(f"Saved TRUSTEE agreement outputs to: {agreement_out_dir}")

    return trustee_out_dir, agreement_out_dir


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
    trustee_params = get_trustee_params(config, args)

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
    print("TRUSTEE params:", trustee_params)

    for round_idx in rounds_to_analyze:
        run_trustee_agreement_for_round(
            config=config,
            data=data,
            run_dir=run_dir,
            round_idx=round_idx,
            num_clients=num_clients,
            device=device,
            trustee_params=trustee_params,
        )


if __name__ == "__main__":
    main()