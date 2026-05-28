from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_agreement_matrix(
    predictions: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """
    Compute pairwise agreement between clients.

    Agreement is the fraction of reference samples for which two clients
    produce the same predicted label.
    """
    if not predictions:
        raise ValueError("predictions must not be empty.")

    client_ids = sorted(int(client_id) for client_id in predictions.keys())

    lengths = {
        int(client_id): len(predictions[client_id])
        for client_id in client_ids
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            f"All prediction arrays must have the same length. Got lengths: {lengths}"
        )

    n_clients = len(client_ids)
    matrix = np.zeros((n_clients, n_clients), dtype=np.float64)

    for i, cid_i in enumerate(client_ids):
        for j, cid_j in enumerate(client_ids):
            pred_i = np.asarray(predictions[cid_i])
            pred_j = np.asarray(predictions[cid_j])

            agreement = np.mean(pred_i == pred_j)
            matrix[i, j] = float(agreement)

    return client_ids, matrix


def compute_mean_agreement(
    matrix: np.ndarray,
    client_ids: list[int],
) -> pd.DataFrame:
    """
    Compute mean, minimum, and maximum agreement for each client against others.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    if matrix.shape[0] != len(client_ids):
        raise ValueError(
            "matrix size must match the number of client IDs. "
            f"matrix.shape={matrix.shape}, len(client_ids)={len(client_ids)}"
        )

    rows = []

    for i, client_id in enumerate(client_ids):
        others = [
            matrix[i, j]
            for j in range(len(client_ids))
            if j != i
        ]

        if others:
            mean_agreement = float(np.mean(others))
            min_agreement = float(np.min(others))
            max_agreement = float(np.max(others))
        else:
            mean_agreement = 1.0
            min_agreement = 1.0
            max_agreement = 1.0

        rows.append(
            {
                "client_id": int(client_id),
                "mean_agreement": mean_agreement,
                "min_agreement": min_agreement,
                "max_agreement": max_agreement,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean_agreement", ascending=True)
        .reset_index(drop=True)
    )


def save_agreement_outputs(
    run_dir: str | Path,
    round_idx: int,
    client_ids: list[int],
    matrix: np.ndarray,
    summary_df: pd.DataFrame,
    extra_metadata: dict | None = None,
    output_group: str | None = None,
) -> Path:
    """
    Save agreement matrix, summary ranking, and metadata.

    Default layout:
        runs/<run_id>/agreement/round_003/

    Grouped layout:
        runs/<run_id>/agreement/local_models/round_003/
        runs/<run_id>/agreement/trustee_pruned_trees/round_003/
    """
    run_dir = Path(run_dir)
    round_idx = int(round_idx)

    if output_group:
        out_dir = run_dir / "agreement" / output_group / f"round_{round_idx:03d}"
    else:
        out_dir = run_dir / "agreement" / f"round_{round_idx:03d}"

    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(
        np.asarray(matrix, dtype=np.float64),
        index=[f"client_{int(cid):03d}" for cid in client_ids],
        columns=[f"client_{int(cid):03d}" for cid in client_ids],
    )
    matrix_df.to_csv(out_dir / "agreement_matrix.csv", index=True)

    summary_df.to_csv(out_dir / "agreement_summary.csv", index=False)

    metadata = {
        "round_idx": round_idx,
        "num_clients": int(len(client_ids)),
    }

    if output_group:
        metadata["output_group"] = output_group

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