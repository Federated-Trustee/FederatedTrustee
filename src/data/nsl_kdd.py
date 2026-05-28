from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

DOS_LABELS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "mailbomb",
    "apache2",
    "processtable",
    "udpstorm",
}

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
DROP_COLUMNS = ["label", "difficulty", "target"]


@dataclass
class NSLKDDData:
    """Container for the binary DoS NSL-KDD experiment data."""

    X_train_raw: pd.DataFrame
    y_train: pd.Series
    X_test_raw: pd.DataFrame
    y_test: pd.Series
    preprocessor: ColumnTransformer
    input_dim: int
    feature_names: list[str]

    @property
    def transformer(self) -> ColumnTransformer:
        """Common transformer interface used by the FL pipeline."""
        return self.preprocessor


def _load_raw_file(file_path: str | Path) -> pd.DataFrame:
    """Load an NSL-KDD raw text file with predefined column names."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"NSL-KDD file not found: {path}")

    return pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)


def _to_binary_dos(label: str) -> int:
    """
    Convert NSL-KDD labels to the binary DoS task.

    Mapping:
        normal -> 0
        DoS attack labels -> 1
        all other attack categories -> -1, later filtered out
    """
    label = str(label).strip()

    if label == "normal":
        return 0

    if label in DOS_LABELS:
        return 1

    return -1


def _build_preprocessor(X_train_raw: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline for NSL-KDD tabular features."""
    numerical_cols = [
        col for col in X_train_raw.columns
        if col not in CATEGORICAL_COLUMNS
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                numerical_cols,
            ),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def _prepare_binary_dos_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter non-DoS attacks and return features plus binary target."""
    df = df.copy()
    df["target"] = df["label"].apply(_to_binary_dos)

    df = df[df["target"] != -1].copy()

    if df.empty:
        raise ValueError("No NSL-KDD samples remain after binary DoS filtering.")

    X_raw = df.drop(columns=DROP_COLUMNS)
    y = df["target"].astype(int)

    return X_raw.reset_index(drop=True), y.reset_index(drop=True)


def load_nsl_kdd_binary_dos(
    train_path: str | Path,
    test_path: str | Path,
) -> NSLKDDData:
    """
    Load NSL-KDD as a binary normal-vs-DoS classification task.

    The preprocessor is fitted only on the training split and then reused for
    the test split, avoiding information leakage.
    """
    train_df = _load_raw_file(train_path)
    test_df = _load_raw_file(test_path)

    X_train_raw, y_train = _prepare_binary_dos_split(train_df)
    X_test_raw, y_test = _prepare_binary_dos_split(test_df)

    preprocessor = _build_preprocessor(X_train_raw)

    X_train_processed = preprocessor.fit_transform(X_train_raw)
    _ = preprocessor.transform(X_test_raw)

    feature_names = list(preprocessor.get_feature_names_out())
    input_dim = int(X_train_processed.shape[1])

    return NSLKDDData(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_test_raw=X_test_raw,
        y_test=y_test,
        preprocessor=preprocessor,
        input_dim=input_dim,
        feature_names=feature_names,
    )