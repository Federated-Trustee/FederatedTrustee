from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "Attack Type"

DROP_COLUMNS = [
    "Unnamed: 0",
    "Label",
    "Attack Type",
    "Attack Tool",
]


@dataclass
class FiveGNIDDData:
    """Container for the binary UDPFlood 5G-NIDD experiment data."""

    X_train_raw: pd.DataFrame
    y_train: pd.Series
    X_test_raw: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler
    input_dim: int
    feature_names: list[str]
    removed_columns: list[str]

    @property
    def transformer(self) -> StandardScaler:
        """Common transformer interface used by the FL pipeline."""
        return self.scaler


def _load_raw_file(file_path: str | Path) -> pd.DataFrame:
    """Load the 5G-NIDD CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"5G-NIDD file not found: {path}")

    return pd.read_csv(path)


def _to_binary_udpflood(attack_type: str) -> int:
    """
    Convert 5G-NIDD labels to the binary UDPFlood task.

    Mapping:
        Benign -> 0
        UDPFlood -> 1
        all other attack categories -> -1, later filtered out
    """
    attack_type = str(attack_type).strip()

    if attack_type == "Benign":
        return 0

    if attack_type == "UDPFlood":
        return 1

    return -1


def _coerce_numeric_features(X_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert all feature columns to numeric and drop columns with invalid values.

    Columns containing NaN or infinite values after conversion are removed to
    keep the pipeline deterministic and compatible with StandardScaler.
    """
    X_raw = X_raw.copy()

    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    removed_columns = X_raw.columns[X_raw.isna().any()].tolist()

    if removed_columns:
        X_raw = X_raw.drop(columns=removed_columns)

    if X_raw.empty:
        raise ValueError("No 5G-NIDD feature columns remain after numeric cleaning.")

    return X_raw, removed_columns


def load_fiveg_nidd_binary_udpflood(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> FiveGNIDDData:
    """
    Load 5G-NIDD as a binary Benign-vs-UDPFlood classification task.

    The scaler is fitted only on the training split and then reused for all
    transformations in the federated pipeline.
    """
    from sklearn.model_selection import train_test_split

    df = _load_raw_file(csv_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected column '{LABEL_COLUMN}' not found in 5G-NIDD CSV.")

    df = df.copy()
    df["target"] = df[LABEL_COLUMN].apply(_to_binary_udpflood)
    df = df[df["target"] != -1].copy()

    if df.empty:
        raise ValueError("No 5G-NIDD samples remain after UDPFlood filtering.")

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]

    X_raw = df.drop(columns=existing_drop_cols + ["target"]).copy()
    y = df["target"].astype(int)

    X_raw, removed_columns = _coerce_numeric_features(X_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    input_dim = int(X_train_raw.shape[1])
    feature_names = list(X_train_raw.columns)

    return FiveGNIDDData(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_test_raw=X_test_raw,
        y_test=y_test,
        scaler=scaler,
        input_dim=input_dim,
        feature_names=feature_names,
        removed_columns=removed_columns,
    )