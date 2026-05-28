from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from trustee import ClassificationTrustee


class TorchExpertAdapter:
    """
    Adapter that exposes a PyTorch classifier through a scikit-like predict API.

    TRUSTEE expects an expert model with a `predict(X)` method. This adapter
    wraps a PyTorch model and returns class labels from raw logits.
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        predictions: list[np.ndarray] = []

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
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.append(preds)

        return np.concatenate(predictions, axis=0)


@dataclass
class TrusteeExtractionResult:
    """Outputs and metrics produced by a TRUSTEE surrogate extraction."""

    dt: Any
    pruned_dt: Any
    teacher_predictions: np.ndarray
    unpruned_predictions: np.ndarray
    pruned_predictions: np.ndarray
    teacher_accuracy: float
    trustee_agreement: float
    trustee_reward: float
    unpruned_fidelity: float
    pruned_fidelity: float
    unpruned_accuracy: float
    pruned_accuracy: float
    unpruned_depth: int
    unpruned_leaves: int
    pruned_depth: int
    pruned_leaves: int


def extract_trustee_surrogate(
    model: torch.nn.Module,
    X_ref: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_iter: int = 20,
    num_stability_iter: int = 5,
    samples_size: float = 0.3,
    verbose: bool = False,
) -> TrusteeExtractionResult:
    """
    Extract TRUSTEE surrogate trees for a trained PyTorch model.

    The PyTorch model is treated as the expert/teacher. TRUSTEE fits a decision
    tree surrogate to mimic the expert predictions on a reference set.
    """
    expert = TorchExpertAdapter(
        model=model,
        device=device,
        batch_size=batch_size,
    )

    teacher_preds = expert.predict(X_ref)
    teacher_acc = float(np.mean(teacher_preds == y_true))

    trustee = ClassificationTrustee(expert=expert)
    trustee.fit(
        X_ref,
        y_true,
        num_iter=int(num_iter),
        num_stability_iter=int(num_stability_iter),
        samples_size=float(samples_size),
        verbose=verbose,
    )

    dt, pruned_dt, agreement, reward = trustee.explain()

    dt_preds = dt.predict(X_ref)
    pruned_dt_preds = pruned_dt.predict(X_ref)

    dt_fidelity = float(np.mean(dt_preds == teacher_preds))
    pruned_dt_fidelity = float(np.mean(pruned_dt_preds == teacher_preds))

    dt_acc = float(np.mean(dt_preds == y_true))
    pruned_dt_acc = float(np.mean(pruned_dt_preds == y_true))

    return TrusteeExtractionResult(
        dt=dt,
        pruned_dt=pruned_dt,
        teacher_predictions=teacher_preds,
        unpruned_predictions=dt_preds,
        pruned_predictions=pruned_dt_preds,
        teacher_accuracy=teacher_acc,
        trustee_agreement=float(agreement),
        trustee_reward=float(reward),
        unpruned_fidelity=dt_fidelity,
        pruned_fidelity=pruned_dt_fidelity,
        unpruned_accuracy=dt_acc,
        pruned_accuracy=pruned_dt_acc,
        unpruned_depth=int(dt.get_depth()),
        unpruned_leaves=int(dt.get_n_leaves()),
        pruned_depth=int(pruned_dt.get_depth()),
        pruned_leaves=int(pruned_dt.get_n_leaves()),
    )


def trustee_result_to_row(
    result: TrusteeExtractionResult,
    client_id: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Convert a TRUSTEE extraction result into a CSV-friendly row."""
    return {
        "client_id": int(client_id),
        "checkpoint_path": str(checkpoint_path),
        "teacher_accuracy": result.teacher_accuracy,
        "trustee_agreement": result.trustee_agreement,
        "trustee_reward": result.trustee_reward,
        "unpruned_fidelity": result.unpruned_fidelity,
        "pruned_fidelity": result.pruned_fidelity,
        "unpruned_accuracy": result.unpruned_accuracy,
        "pruned_accuracy": result.pruned_accuracy,
        "unpruned_depth": result.unpruned_depth,
        "unpruned_leaves": result.unpruned_leaves,
        "pruned_depth": result.pruned_depth,
        "pruned_leaves": result.pruned_leaves,
    }