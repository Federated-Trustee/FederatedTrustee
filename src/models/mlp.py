from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """
    Multilayer perceptron for tabular binary/multiclass classification.

    The model expects preprocessed numerical input features and outputs raw
    logits. Loss functions and metrics should apply softmax/argmax externally
    when needed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        input_dim = int(input_dim)
        hidden_dims = [int(dim) for dim in hidden_dims]
        num_classes = int(num_classes)
        dropout = float(dropout)

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be > 0")

        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits."""
        return self.network(x)