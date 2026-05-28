from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """
    Convert a PyTorch model state_dict into Flower-compatible NumPy arrays.
    """
    return [
        tensor.detach().cpu().numpy()
        for tensor in model.state_dict().values()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: list[np.ndarray],
) -> None:
    """
    Load Flower-compatible NumPy arrays into a PyTorch model.

    The order of arrays must match the model state_dict order.
    """
    model_keys = list(model.state_dict().keys())

    if len(model_keys) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model expects {len(model_keys)} tensors, "
            f"but received {len(parameters)}."
        )

    state_dict = OrderedDict(
        {
            key: torch.as_tensor(value)
            for key, value in zip(model_keys, parameters)
        }
    )

    model.load_state_dict(state_dict, strict=True)