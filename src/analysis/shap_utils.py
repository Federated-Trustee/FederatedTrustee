from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch


class TorchProbabilityWrapper:
    """
    Wrapper that exposes a PyTorch classifier through a predict_proba API.

    SHAP KernelExplainer expects a callable that receives a 2D NumPy array and
    returns model outputs. This wrapper returns softmax probabilities.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 512,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for input samples."""
        X = np.asarray(X, dtype=np.float32)

        probabilities: list[np.ndarray] = []

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size

                batch_X = torch.as_tensor(
                    X[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                logits = self.model(batch_X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(probs)

        return np.concatenate(probabilities, axis=0)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


@dataclass
class ShapExplanationResult:
    """Container for a single local SHAP explanation."""

    client_id: int
    role: str
    predicted_label: int
    explained_class: int
    explained_class_probability: float
    probabilities: np.ndarray
    base_value: float
    shap_values: np.ndarray
    feature_values: np.ndarray


def clean_feature_name(name: str, index: int) -> str:
    """
    Clean feature names for plotting.

    Some datasets contain symbols or short labels that can render poorly in
    matplotlib. This function keeps names readable and falls back to stable
    feature IDs when needed.
    """
    clean = str(name).strip()

    clean = clean.replace("num__", "")
    clean = clean.replace("cat__", "")
    clean = clean.replace("remainder__", "")

    # Avoid matplotlib/mathtext rendering issues.
    clean = clean.replace("$", "")
    clean = clean.replace("\\", "")
    clean = clean.replace("{", "")
    clean = clean.replace("}", "")
    clean = clean.replace("^", "")
    clean = clean.replace("_", " ")

    clean = re.sub(r"[\t\r\n]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Keep common readable characters and remove isolated noise.
    clean = re.sub(r"[^A-Za-z0-9À-ÿ .:/()+\-&%]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    alnum_count = sum(ch.isalnum() for ch in clean)

    if not clean or alnum_count <= 1:
        return f"feature_{index:03d}"

    return clean


def sanitize_feature_names(feature_names: list[str]) -> tuple[list[str], pd.DataFrame]:
    """
    Clean feature names and return a mapping table.

    Returns:
        clean_feature_names, feature_name_mapping_df
    """
    clean_names: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for idx, original in enumerate(feature_names):
        clean = clean_feature_name(str(original), idx)

        if clean in seen:
            seen[clean] += 1
            clean_unique = f"{clean} ({seen[clean]})"
        else:
            seen[clean] = 0
            clean_unique = clean

        clean_names.append(clean_unique)
        rows.append(
            {
                "feature_index": int(idx),
                "original_feature_name": str(original),
                "clean_feature_name": clean_unique,
            }
        )

    return clean_names, pd.DataFrame(rows)


def choose_sample_index(
    wrappers: dict[int, TorchProbabilityWrapper],
    benign_client: int,
    malicious_clients: list[int],
    X_ref: np.ndarray,
    y_true: np.ndarray,
    target_class: int,
) -> int:
    """
    Choose a reference sample for SHAP comparison.

    Priority:
        1. a sample with true label == target_class and prediction disagreement
           between benign and at least one malicious client;
        2. any sample with disagreement;
        3. any sample with true label == target_class;
        4. sample 0.
    """
    benign_preds = wrappers[int(benign_client)].predict_label(X_ref)
    malicious_preds = {
        int(client_id): wrappers[int(client_id)].predict_label(X_ref)
        for client_id in malicious_clients
    }

    for idx in range(len(X_ref)):
        has_disagreement = any(
            malicious_preds[int(client_id)][idx] != benign_preds[idx]
            for client_id in malicious_clients
        )

        if has_disagreement and int(y_true[idx]) == int(target_class):
            return int(idx)

    for idx in range(len(X_ref)):
        has_disagreement = any(
            malicious_preds[int(client_id)][idx] != benign_preds[idx]
            for client_id in malicious_clients
        )

        if has_disagreement:
            return int(idx)

    for idx in range(len(X_ref)):
        if int(y_true[idx]) == int(target_class):
            return int(idx)

    return 0


def get_class_shap_values(shap_values, class_id: int) -> np.ndarray:
    """
    Extract SHAP values for one class.

    Supports SHAP outputs as:
        - list[class] -> array(n_samples, n_features)
        - array(n_samples, n_features, n_classes)
        - array(n_samples, n_features)
    """
    class_id = int(class_id)

    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_id])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, class_id]

    if shap_values.ndim == 2:
        return shap_values

    raise ValueError(f"Unsupported SHAP values shape: {shap_values.shape}")


def get_expected_value(expected_value, class_id: int) -> float:
    """Extract SHAP expected value for one class."""
    class_id = int(class_id)

    if isinstance(expected_value, list):
        return float(expected_value[class_id])

    expected_value = np.asarray(expected_value)

    if expected_value.ndim == 0:
        return float(expected_value)

    return float(expected_value[class_id])


def explain_single_client(
    wrapper: TorchProbabilityWrapper,
    sample: np.ndarray,
    client_id: int,
    role: str,
    explain_mode: str,
    target_class: int,
    background: np.ndarray,
    nsamples: int,
) -> ShapExplanationResult:
    """
    Compute a local SHAP explanation for one client and one sample.

    explain_mode:
        - "target": explain the fixed target_class for all clients.
        - "predicted": explain the class predicted by each client.
    """
    probs = wrapper.predict_proba(sample)[0]
    predicted_label = int(np.argmax(probs))

    if explain_mode == "predicted":
        explained_class = predicted_label
    elif explain_mode == "target":
        explained_class = int(target_class)
    else:
        raise ValueError("explain_mode must be either 'target' or 'predicted'.")

    explainer = shap.KernelExplainer(
        wrapper.predict_proba,
        background,
    )

    shap_values = explainer.shap_values(
        sample,
        nsamples=int(nsamples),
    )

    class_shap_values = get_class_shap_values(
        shap_values=shap_values,
        class_id=explained_class,
    )

    shap_values_1d = np.asarray(class_shap_values[0], dtype=float)
    feature_values_1d = np.asarray(sample[0], dtype=float)

    base_value = get_expected_value(
        expected_value=explainer.expected_value,
        class_id=explained_class,
    )

    return ShapExplanationResult(
        client_id=int(client_id),
        role=str(role),
        predicted_label=predicted_label,
        explained_class=int(explained_class),
        explained_class_probability=float(probs[explained_class]),
        probabilities=np.asarray(probs, dtype=float),
        base_value=float(base_value),
        shap_values=shap_values_1d,
        feature_values=feature_values_1d,
    )


def build_plot_dataframe(
    shap_values_1d: np.ndarray,
    feature_values_1d: np.ndarray,
    feature_names: list[str],
    top_k: int,
    min_abs_shap: float,
) -> pd.DataFrame:
    """Build a sorted dataframe for local SHAP bar plots."""
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": feature_values_1d,
            "shap_value": shap_values_1d,
            "abs_shap": np.abs(shap_values_1d),
        }
    )

    df = df[df["abs_shap"] >= float(min_abs_shap)].copy()
    df = df.sort_values("abs_shap", ascending=False).head(int(top_k))

    if df.empty:
        df = pd.DataFrame(
            {
                "feature": ["No feature above threshold"],
                "feature_value": [0.0],
                "shap_value": [0.0],
                "abs_shap": [0.0],
            }
        )

    return df.iloc[::-1].reset_index(drop=True)


def save_local_bar_plot(
    shap_values_1d: np.ndarray,
    feature_values_1d: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
    top_k: int,
    min_abs_shap: float,
) -> None:
    """Save a local SHAP bar plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = build_plot_dataframe(
        shap_values_1d=shap_values_1d,
        feature_values_1d=feature_values_1d,
        feature_names=feature_names,
        top_k=top_k,
        min_abs_shap=min_abs_shap,
    )

    fig_height = max(4.5, 0.45 * len(plot_df) + 1.5)

    plt.figure(figsize=(11, fig_height))
    plt.barh(plot_df["feature"], plot_df["shap_value"])
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(
    shap_values_1d: np.ndarray,
    base_value: float,
    feature_values_1d: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    """Save a local SHAP waterfall plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explanation = shap.Explanation(
        values=shap_values_1d,
        base_values=base_value,
        data=feature_values_1d,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(
        explanation,
        max_display=int(top_k),
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_shap_values_csv(
    result: ShapExplanationResult,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save per-feature SHAP values for one explanation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature": feature_names,
            "feature_value": result.feature_values,
            "shap_value": result.shap_values,
            "abs_shap_value": np.abs(result.shap_values),
        }
    ).sort_values("abs_shap_value", ascending=False)

    shap_df.to_csv(output_path, index=False)