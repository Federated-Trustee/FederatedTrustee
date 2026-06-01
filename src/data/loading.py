from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.fiveg_nidd import load_fiveg_nidd_binary_udpflood
from src.data.nsl_kdd import load_nsl_kdd_binary_dos


@dataclass
class LoadedDataset:
    """Dataset bundle plus metadata used by experiment and analysis scripts."""

    data: Any
    run_prefix: str


def get_run_prefix_from_config(config: dict) -> str:
    """
    Build the run directory prefix from the experiment name.

    Example:
        experiment_name: paper_exp1_fiveg_nidd_p060_m2_seed021
        run_prefix:      paper_exp1_fiveg_nidd_p060_m2_seed021_
    """
    experiment_name = str(config.get("experiment_name", "")).strip()

    if not experiment_name:
        dataset_name = str(config["dataset"]["name"]).strip()
        experiment_name = f"{dataset_name}_v1"

    return f"{experiment_name}_"


def load_dataset_from_config(config: dict) -> LoadedDataset:
    """
    Load the dataset specified by a configuration dictionary.

    Supported datasets:
        - nsl_kdd
        - fiveg_nidd
    """
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    run_prefix = get_run_prefix_from_config(config)

    if dataset_name == "nsl_kdd":
        data = load_nsl_kdd_binary_dos(
            train_path=dataset_cfg["train_path"],
            test_path=dataset_cfg["test_path"],
        )

        return LoadedDataset(
            data=data,
            run_prefix=run_prefix,
        )

    if dataset_name == "fiveg_nidd":
        data = load_fiveg_nidd_binary_udpflood(
            csv_path=dataset_cfg["csv_path"],
            test_size=dataset_cfg["test_size"],
            random_state=dataset_cfg["random_state"],
        )

        return LoadedDataset(
            data=data,
            run_prefix=run_prefix,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")