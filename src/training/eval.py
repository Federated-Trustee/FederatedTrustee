from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_eval_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """
    Create a DataLoader for evaluation.

    Evaluation loaders do not shuffle samples.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    X_tensor = torch.as_tensor(np.asarray(X).copy(), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y).copy(), dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """
    Evaluate a model using cross-entropy loss and accuracy.

    Returns:
        Dictionary with:
        - loss
        - accuracy
    """
    model.to(device)
    model.eval()

    loader = make_eval_dataloader(
        X=X,
        y=y,
        batch_size=int(batch_size),
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == batch_y).sum().item())

        batch_size_now = int(batch_X.size(0))
        total_loss += float(loss.item()) * batch_size_now
        total_samples += batch_size_now

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
    }