from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from NumPy arrays.

    Args:
        X: Feature matrix.
        y: Target labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.

    Returns:
        DataLoader with float32 features and long integer labels.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
    )


def train_one_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    """
    Train one model locally using standard supervised learning.

    The model outputs raw logits and is optimized using CrossEntropyLoss.
    """
    epochs = int(epochs)

    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    model.to(device)
    model.train()

    loader = make_dataloader(
        X=X_train,
        y=y_train,
        batch_size=int(batch_size),
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    last_loss = 0.0

    for _ in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_now = int(batch_X.size(0))
            running_loss += float(loss.item()) * batch_size_now
            total_samples += batch_size_now

        last_loss = running_loss / max(total_samples, 1)

    return {"train_loss": float(last_loss)}