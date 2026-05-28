from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _path import add_project_root_to_path

add_project_root_to_path()

from src.analysis.shap_utils import (
    TorchProbabilityWrapper,
    choose_sample_index,
    explain_single_client,
    sanitize_feature_names,
    save_local_bar_plot,
    save_shap_values_csv,
    save_waterfall_plot,
)
from src.artifacts.checkpoints import get_checkpoint_path, load_model_from_checkpoint
from src.config import load_config
from src.data.loading import load_dataset_from_config
from src.data.partition import transform_features
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local SHAP explanations for the same sample across "
            "benign and malicious federated clients."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML configuration file.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing saved client checkpoints.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Federated round to analyze. Default: 3.",
    )
    parser.add_argument(
        "--benign-client",
        type=int,
        required=True,
        help="Benign client ID used as comparison reference.",
    )
    parser.add_argument(
        "--malicious-clients",
        type=int,
        nargs="+",
        required=True,
        help="Malicious client IDs to compare against the benign client.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Optional test sample index to explain. If omitted, the script "
            "selects a sample with prediction disagreement when possible."
        ),
    )
    parser.add_argument(
        "--explain",
        type=str,
        choices=["target", "predicted"],
        default="target",
        help=(
            "'target' explains --target-class for all clients. "
            "'predicted' explains each client's predicted class."
        ),
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Target class used when --explain target is selected. Default: 1.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of background samples used by Kernel SHAP. Default: 100.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Number of SHAP samples used by KernelExplainer. Default: 200.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of features shown in plots. Default: 12.",
    )
    parser.add_argument(
        "--min-abs-shap",
        type=float,
        default=1e-8,
        help="Features below this absolute SHAP value are omitted from bar plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting SHAP background samples. Default: 42.",
    )

    return parser.parse_args()


def build_output_dir(
    run_dir: Path,
    round_idx: int,
    sample_index: int,
    explain_mode: str,
) -> Path:
    """Build the SHAP output directory."""
    return (
        run_dir
        / "figures"
        / "shap"
        / f"round_{int(round_idx):03d}"
        / f"sample_{int(sample_index):05d}"
        / f"explain_{explain_mode}"
    )


def build_client_roles(
    benign_client: int,
    malicious_clients: list[int],
) -> dict[int, str]:
    """Build a client_id -> role mapping."""
    roles = {int(benign_client): "benign"}

    for client_id in malicious_clients:
        roles[int(client_id)] = "malicious"

    return roles


def load_client_wrappers(
    config: dict[str, Any],
    data,
    run_dir: Path,
    round_idx: int,
    client_ids: list[int],
    device,
) -> tuple[dict[int, TorchProbabilityWrapper], dict[int, str]]:
    """
    Load client checkpoints and wrap them for SHAP probability prediction.
    """
    wrappers: dict[int, TorchProbabilityWrapper] = {}
    checkpoint_paths: dict[int, str] = {}

    for client_id in client_ids:
        checkpoint_path = get_checkpoint_path(
            run_dir=run_dir,
            round_idx=round_idx,
            client_id=client_id,
        )

        model = load_model_from_checkpoint(
            config=config,
            input_dim=int(data.input_dim),
            checkpoint_path=checkpoint_path,
            device=device,
        )

        wrappers[int(client_id)] = TorchProbabilityWrapper(
            model=model,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )
        checkpoint_paths[int(client_id)] = str(checkpoint_path)

    return wrappers, checkpoint_paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["runtime"]["seed"])
    device = get_device(config["runtime"]["device"])

    rng = np.random.default_rng(int(args.seed))

    run_dir = Path(args.run_dir)
    round_idx = int(args.round)

    loaded_dataset = load_dataset_from_config(config)
    data = loaded_dataset.data

    X_train = transform_features(
        data.X_train_raw,
        data.transformer,
    )
    X_ref = transform_features(
        data.X_test_raw,
        data.transformer,
    )
    y_true = data.y_test.to_numpy()

    feature_names, feature_map_df = sanitize_feature_names(data.feature_names)

    if len(feature_names) != X_ref.shape[1]:
        raise ValueError(
            f"Feature name count does not match input dimension. "
            f"len(feature_names)={len(feature_names)}, X_ref.shape[1]={X_ref.shape[1]}"
        )

    client_roles = build_client_roles(
        benign_client=int(args.benign_client),
        malicious_clients=[int(cid) for cid in args.malicious_clients],
    )
    all_clients = list(client_roles.keys())

    wrappers, checkpoint_paths = load_client_wrappers(
        config=config,
        data=data,
        run_dir=run_dir,
        round_idx=round_idx,
        client_ids=all_clients,
        device=device,
    )

    if args.sample_index is None:
        sample_index = choose_sample_index(
            wrappers=wrappers,
            benign_client=int(args.benign_client),
            malicious_clients=[int(cid) for cid in args.malicious_clients],
            X_ref=X_ref,
            y_true=y_true,
            target_class=int(args.target_class),
        )
    else:
        sample_index = int(args.sample_index)

    if sample_index < 0 or sample_index >= len(X_ref):
        raise ValueError(
            f"Invalid sample index {sample_index}. "
            f"Valid range: 0 to {len(X_ref) - 1}."
        )

    sample = X_ref[sample_index : sample_index + 1]
    sample_raw = data.X_test_raw.iloc[[sample_index]].copy()

    background_size = min(int(args.background_size), len(X_train))
    background_indices = rng.choice(
        np.arange(len(X_train)),
        size=background_size,
        replace=False,
    )
    background = X_train[background_indices]

    output_dir = build_output_dir(
        run_dir=run_dir,
        round_idx=round_idx,
        sample_index=sample_index,
        explain_mode=args.explain,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:", config["dataset"]["name"])
    print("Run directory:", run_dir)
    print("Round:", round_idx)
    print("Client roles:", client_roles)
    print("Explain mode:", args.explain)
    print("Target class:", int(args.target_class))
    print("Selected sample index:", sample_index)
    print("True label:", int(y_true[sample_index]))
    print("Background size:", background_size)
    print("Output directory:", output_dir)

    sample_raw.to_csv(output_dir / "selected_sample_raw.csv", index=False)
    feature_map_df.to_csv(output_dir / "feature_name_mapping.csv", index=False)

    rows: list[dict[str, Any]] = []

    for client_id in all_clients:
        role = client_roles[int(client_id)]
        wrapper = wrappers[int(client_id)]

        print("\n" + "=" * 90)
        print(f"COMPUTING SHAP | client={client_id:03d} | role={role}")

        result = explain_single_client(
            wrapper=wrapper,
            sample=sample,
            client_id=int(client_id),
            role=role,
            explain_mode=args.explain,
            target_class=int(args.target_class),
            background=background,
            nsamples=int(args.nsamples),
        )

        prefix = (
            f"client_{client_id:03d}_{role}_"
            f"explained_class_{result.explained_class}_"
            f"pred_{result.predicted_label}"
        )

        bar_path = output_dir / f"{prefix}_shap_bar.png"
        waterfall_path = output_dir / f"{prefix}_shap_waterfall.png"
        csv_path = output_dir / f"{prefix}_shap_values.csv"

        title = (
            f"SHAP local explanation | client={client_id:03d} ({role}) | "
            f"round={round_idx:03d} | sample={sample_index} | "
            f"explained class={result.explained_class} | "
            f"pred={result.predicted_label}"
        )

        save_local_bar_plot(
            shap_values_1d=result.shap_values,
            feature_values_1d=result.feature_values,
            feature_names=feature_names,
            output_path=bar_path,
            title=title,
            top_k=int(args.top_k),
            min_abs_shap=float(args.min_abs_shap),
        )

        save_waterfall_plot(
            shap_values_1d=result.shap_values,
            base_value=float(result.base_value),
            feature_values_1d=result.feature_values,
            feature_names=feature_names,
            output_path=waterfall_path,
            title=title,
            top_k=int(args.top_k),
        )

        save_shap_values_csv(
            result=result,
            feature_names=feature_names,
            output_path=csv_path,
        )

        rows.append(
            {
                "client_id": int(client_id),
                "role": role,
                "checkpoint_path": checkpoint_paths[int(client_id)],
                "sample_index": int(sample_index),
                "true_label": int(y_true[sample_index]),
                "predicted_label": int(result.predicted_label),
                "explain_mode": args.explain,
                "target_class_argument": int(args.target_class),
                "explained_class": int(result.explained_class),
                "explained_class_probability": float(result.explained_class_probability),
                "prob_class_0": float(result.probabilities[0]),
                "prob_class_1": float(result.probabilities[1]),
                "base_value": float(result.base_value),
                "bar_plot": str(bar_path),
                "waterfall_plot": str(waterfall_path),
                "shap_values_csv": str(csv_path),
            }
        )

        print("Predicted label:", result.predicted_label)
        print("Explained class:", result.explained_class)
        print("Probabilities:", result.probabilities)
        print(
            f"Explained class probability: "
            f"{result.explained_class_probability:.6f}"
        )
        print("Saved bar plot:", bar_path)
        print("Saved waterfall plot:", waterfall_path)
        print("Saved SHAP values:", csv_path)

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "shap_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "dataset": config["dataset"]["name"],
        "run_dir": str(run_dir),
        "round_idx": int(round_idx),
        "client_roles": client_roles,
        "all_clients": all_clients,
        "sample_index": int(sample_index),
        "true_label": int(y_true[sample_index]),
        "explain_mode": args.explain,
        "target_class": int(args.target_class),
        "background_size": int(background_size),
        "nsamples": int(args.nsamples),
        "top_k": int(args.top_k),
        "min_abs_shap": float(args.min_abs_shap),
        "output_dir": str(output_dir),
    }

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\nSaved summary:", summary_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()