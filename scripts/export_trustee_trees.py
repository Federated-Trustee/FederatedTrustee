from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_text, plot_tree

from _path import add_project_root_to_path

add_project_root_to_path()

from src.analysis.trustee import extract_trustee_surrogate, trustee_result_to_row
from src.artifacts.checkpoints import get_checkpoint_path, load_model_from_checkpoint
from src.config import load_config
from src.data.loading import load_dataset_from_config
from src.data.partition import transform_features
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TRUSTEE surrogate trees as figures and text files.",
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
        help="Run directory to analyze.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Federated round to analyze. Default: 3.",
    )
    parser.add_argument(
        "--client-ids",
        type=int,
        nargs="+",
        required=True,
        help="Client IDs to export trees for. Example: --client-ids 0 1 2.",
    )
    parser.add_argument(
        "--tree-kind",
        type=str,
        choices=["pruned", "unpruned", "both"],
        default="both",
        help="Which TRUSTEE tree to export.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional max depth for plotting only. It does not change the extracted tree.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=22,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=12,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="TRUSTEE num_iter. If omitted, uses config['trustee']['num_iter'].",
    )
    parser.add_argument(
        "--num-stability-iter",
        type=int,
        default=None,
        help=(
            "TRUSTEE num_stability_iter. "
            "If omitted, uses config['trustee']['num_stability_iter']."
        ),
    )
    parser.add_argument(
        "--samples-size",
        type=float,
        default=None,
        help="TRUSTEE samples_size. If omitted, uses config['trustee']['samples_size'].",
    )

    return parser.parse_args()


def get_trustee_params(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trustee_cfg = config.get("trustee", {})

    return {
        "num_iter": int(
            args.num_iter
            if args.num_iter is not None
            else trustee_cfg.get("num_iter", 20)
        ),
        "num_stability_iter": int(
            args.num_stability_iter
            if args.num_stability_iter is not None
            else trustee_cfg.get("num_stability_iter", 5)
        ),
        "samples_size": float(
            args.samples_size
            if args.samples_size is not None
            else trustee_cfg.get("samples_size", 0.3)
        ),
    }


def sanitize_feature_names(feature_names: list[str]) -> list[str]:
    """Clean feature names for tree plots."""
    clean_names: list[str] = []

    for name in feature_names:
        clean = str(name)
        clean = clean.replace("num__", "")
        clean = clean.replace("cat__", "")
        clean = clean.replace("remainder__", "")
        clean_names.append(clean)

    return clean_names


def save_tree_figure(
    tree,
    feature_names: list[str],
    class_names: list[str],
    output_path: Path,
    title: str,
    max_depth: int | None,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Save a sklearn decision tree as a PNG figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(float(fig_width), float(fig_height)))

    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
        max_depth=max_depth,
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()


def save_tree_text(
    tree,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save a sklearn decision tree as a text representation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree_text = export_text(
        tree,
        feature_names=feature_names,
        decimals=4,
        spacing=3,
    )

    output_path.write_text(tree_text, encoding="utf-8")


def get_trees_to_save(tree_kind: str, unpruned_tree, pruned_tree) -> list[tuple[str, Any]]:
    """Select which trees should be exported."""
    trees_to_save: list[tuple[str, Any]] = []

    if tree_kind in {"unpruned", "both"}:
        trees_to_save.append(("unpruned", unpruned_tree))

    if tree_kind in {"pruned", "both"}:
        trees_to_save.append(("pruned", pruned_tree))

    return trees_to_save


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["runtime"]["seed"])
    device = get_device(config["runtime"]["device"])

    run_dir = Path(args.run_dir)
    round_idx = int(args.round)

    loaded_dataset = load_dataset_from_config(config)
    data = loaded_dataset.data

    X_ref = transform_features(
        data.X_test_raw,
        data.transformer,
    )
    y_true = data.y_test.to_numpy()

    feature_names = sanitize_feature_names(data.feature_names)

    if len(feature_names) != X_ref.shape[1]:
        raise ValueError(
            f"Feature name count does not match transformed input dimension. "
            f"len(feature_names)={len(feature_names)}, X_ref.shape[1]={X_ref.shape[1]}"
        )

    trustee_params = get_trustee_params(config, args)

    class_names = ["Benign/Normal", "Attack"]

    output_base = (
        run_dir
        / "figures"
        / "trustee_trees"
        / f"round_{round_idx:03d}"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print("Dataset:", config["dataset"]["name"])
    print("Run directory:", run_dir)
    print("Round:", round_idx)
    print("Client IDs:", args.client_ids)
    print("Reference set shape:", X_ref.shape)
    print("TRUSTEE params:", trustee_params)
    print("Output directory:", output_base)

    for client_id in args.client_ids:
        client_id = int(client_id)

        print("\n" + "=" * 90)
        print(f"EXPORTING TRUSTEE TREES | client={client_id:03d}")

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

        result = extract_trustee_surrogate(
            model=model,
            X_ref=X_ref,
            y_true=y_true,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            num_iter=int(trustee_params["num_iter"]),
            num_stability_iter=int(trustee_params["num_stability_iter"]),
            samples_size=float(trustee_params["samples_size"]),
            verbose=False,
        )

        trees_to_save = get_trees_to_save(
            tree_kind=args.tree_kind,
            unpruned_tree=result.dt,
            pruned_tree=result.pruned_dt,
        )

        for kind, tree in trees_to_save:
            image_path = output_base / f"client_{client_id:03d}_{kind}_tree.png"
            text_path = output_base / f"client_{client_id:03d}_{kind}_tree.txt"

            title = (
                f"TRUSTEE {kind} tree | "
                f"client={client_id:03d} | "
                f"round={round_idx:03d}"
            )

            save_tree_figure(
                tree=tree,
                feature_names=feature_names,
                class_names=class_names,
                output_path=image_path,
                title=title,
                max_depth=args.max_depth,
                fig_width=float(args.fig_width),
                fig_height=float(args.fig_height),
                dpi=int(args.dpi),
            )

            save_tree_text(
                tree=tree,
                feature_names=feature_names,
                output_path=text_path,
            )

            print(f"Saved {kind} tree image:", image_path)
            print(f"Saved {kind} tree text:", text_path)

        rows.append(
            trustee_result_to_row(
                result=result,
                client_id=client_id,
                checkpoint_path=checkpoint_path,
            )
        )

        print(
            f"teacher_acc={result.teacher_accuracy:.6f} | "
            f"trustee_agreement={result.trustee_agreement:.6f} | "
            f"reward={result.trustee_reward:.6f} | "
            f"pruned_depth={result.pruned_depth} | "
            f"pruned_leaves={result.pruned_leaves}"
        )

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("client_id")
        .reset_index(drop=True)
    )

    summary_path = output_base / "exported_tree_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "dataset": config["dataset"]["name"],
        "run_dir": str(run_dir),
        "round_idx": round_idx,
        "client_ids": [int(client_id) for client_id in args.client_ids],
        "tree_kind": args.tree_kind,
        "trustee_params": trustee_params,
        "max_depth_for_plot": args.max_depth,
        "reference_size": int(X_ref.shape[0]),
        "feature_count": int(X_ref.shape[1]),
    }

    metadata_path = output_base / "metadata.json"

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