from __future__ import annotations

import torch


def get_device(requested_device: str = "cpu") -> torch.device:
    """
    Resolve the computation device.

    Supported values:
        - "cpu": always use CPU.
        - "cuda": use CUDA if available; otherwise fall back to CPU.
        - "auto": use CUDA if available; otherwise CPU.
    """
    requested_device = requested_device.lower().strip()

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        print("[WARN] CUDA requested, but not available. Falling back to CPU.")
        return torch.device("cpu")

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{requested_device}'. "
        "Expected one of: 'cpu', 'cuda', or 'auto'."
    )