from __future__ import annotations

import numpy as np


def apply_label_flip(
    y: np.ndarray,
    source_label: int,
    target_label: int,
    flip_probability: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply probabilistic label flipping to a label array.

    Only labels equal to `source_label` are eligible for flipping. Each eligible
    sample is independently flipped to `target_label` with probability
    `flip_probability`.

    Args:
        y: Original label array.
        source_label: Label value eligible for flipping.
        target_label: New label assigned after flipping.
        flip_probability: Probability of flipping each eligible sample.
        seed: Optional random seed.

    Returns:
        A poisoned copy of the original label array.
    """
    flip_probability = float(flip_probability)
    source_label = int(source_label)
    target_label = int(target_label)

    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must be between 0.0 and 1.0.")

    if source_label == target_label:
        raise ValueError("source_label and target_label must be different.")

    y_poisoned = np.asarray(y).copy()

    source_mask = y_poisoned == source_label
    source_indices = np.where(source_mask)[0]

    if len(source_indices) == 0 or flip_probability == 0.0:
        return y_poisoned

    rng = np.random.default_rng(seed)
    random_values = rng.random(len(source_indices))
    flip_mask = random_values < flip_probability

    indices_to_flip = source_indices[flip_mask]
    y_poisoned[indices_to_flip] = target_label

    return y_poisoned