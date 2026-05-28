from __future__ import annotations

from torch import nn

from src.models.mlp import TabularMLP


def build_model(config: dict, input_dim: int) -> nn.Module:
    """
    Build a model from the experiment configuration.

    Currently supported models:
        - mlp: tabular multilayer perceptron.
    """
    model_cfg = config["model"]

    model_name = str(model_cfg["name"]).lower().strip()

    if model_name != "mlp":
        raise ValueError(f"Unsupported model name: {model_name}")

    return TabularMLP(
        input_dim=int(input_dim),
        hidden_dims=[int(dim) for dim in model_cfg["hidden_dims"]],
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
    )