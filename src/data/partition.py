from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClientPartition:
    """Raw data assigned to one federated client."""

    client_id: int
    X_raw: pd.DataFrame
    y: pd.Series


def transform_features(
    X_raw: pd.DataFrame,
    transformer,
) -> np.ndarray:
    """
    Transform raw features using a fitted sklearn transformer.

    Sparse outputs, such as one-hot encoded matrices, are converted to dense
    NumPy arrays because the PyTorch model expects dense float32 tensors.
    """
    X = transformer.transform(X_raw)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


def make_iid_partitions(
    X_raw: pd.DataFrame,
    y: pd.Series,
    num_clients: int,
    seed: int = 42,
) -> list[ClientPartition]:
    """
    Split a dataset into IID client partitions.

    Samples are shuffled once using the provided seed and then split into
    approximately equal-sized partitions.
    """
    num_clients = int(num_clients)

    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    n_samples = len(X_raw)

    if n_samples != len(y):
        raise ValueError("X_raw and y must have the same number of rows.")

    if num_clients > n_samples:
        raise ValueError(
            f"num_clients={num_clients} cannot exceed number of samples={n_samples}."
        )

    rng = np.random.default_rng(int(seed))
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_indices = np.array_split(indices, num_clients)

    partitions: list[ClientPartition] = []

    for client_id, idx in enumerate(split_indices):
        partitions.append(
            ClientPartition(
                client_id=int(client_id),
                X_raw=X_raw.iloc[idx].reset_index(drop=True),
                y=y.iloc[idx].reset_index(drop=True),
            )
        )

    return partitions