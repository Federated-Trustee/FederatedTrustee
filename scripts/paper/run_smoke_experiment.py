from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIGS = {
    "fiveg_nidd": PROJECT_ROOT / "configs" / "fiveg_nidd.yaml",
    "nsl_kdd": PROJECT_ROOT / "configs" / "nsl_kdd.yaml",
}

TMP_CONFIG_DIR = PROJECT_ROOT / "tmp" / "paper_configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small end-to-end smoke test for the FederatedTrustee pipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fiveg_nidd", "nsl_kdd"],
        default="fiveg_nidd",
        help="Dataset to use in the smoke test. Default: fiveg_nidd.",
    )
    parser.add_argument(
        "--round-mode",
        type=str,
        choices=["final", "all-rounds"],
        default="all-rounds",
        help="Pipeline round mode. Default: all-rounds.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of federated clients. Default: 2.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of federated rounds. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed. Default: 1.",
    )
    parser.add_argument(
        "--flip-probability",
        type=float,
        default=1.0,
        help="Label-flipping probability. Default: 1.0.",
    )
    parser.add_argument(
        "--trustee-num-iter",
        type=int,
        default=2,
        help="TRUSTEE num_iter for a quick smoke test. Default: 2.",
    )
    parser.add_argument(
        "--trustee-num-stability-iter",
        type=int,
        default=1,
        help="TRUSTEE num_stability_iter for a quick smoke test. Default: 1.",
    )
    parser.add_argument(
        "--trustee-samples-size",
        type=float,
        default=0.1,
        help="TRUSTEE samples_size for a quick smoke test. Default: 0.1.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def build_smoke_config(
    base_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # Keep the original experiment_name so the existing pipeline can find
    # the latest run using the current dataset prefix logic.
    cfg["experiment_name"] = base_cfg["experiment_name"]

    cfg["federated"]["num_clients"] = int(args.num_clients)
    cfg["federated"]["num_rounds"] = int(args.num_rounds)
    cfg["federated"]["local_epochs"] = 1
    cfg["federated"]["fraction_fit"] = 1.0
    cfg["federated"]["fraction_evaluate"] = 1.0

    cfg["runtime"]["seed"] = int(args.seed)

    cfg["attack"]["enabled"] = True
    cfg["attack"]["type"] = "label_flip"
    cfg["attack"]["malicious_client_ids"] = [0]
    cfg["attack"]["source_label"] = 1
    cfg["attack"]["target_label"] = 0
    cfg["attack"]["flip_probability"] = float(args.flip_probability)

    cfg.setdefault("trustee", {})
    cfg["trustee"]["enabled"] = True
    cfg["trustee"]["round_to_analyze"] = int(args.num_rounds)
    cfg["trustee"]["num_iter"] = int(args.trustee_num_iter)
    cfg["trustee"]["num_stability_iter"] = int(args.trustee_num_stability_iter)
    cfg["trustee"]["samples_size"] = float(args.trustee_samples_size)

    cfg["smoke_test"] = {
        "enabled": True,
        "description": "Small end-to-end local pipeline test.",
        "dataset": args.dataset,
        "num_clients": int(args.num_clients),
        "num_rounds": int(args.num_rounds),
        "seed": int(args.seed),
        "flip_probability": float(args.flip_probability),
    }

    return cfg


def find_latest_run(config: dict[str, Any]) -> Path | None:
    run_base = PROJECT_ROOT / config["artifacts"]["run_dir"]
    prefix = f"{config['experiment_name']}_"

    if not run_base.exists():
        return None

    candidates = [
        path
        for path in run_base.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]

    if not candidates:
        return None

    return sorted(candidates)[-1]


def run_pipeline(config_path: Path, round_mode: str) -> int:
    cmd = [
        "bash",
        "run_pipeline.sh",
        str(config_path),
        round_mode,
    ]

    print("\n" + "=" * 90)
    print("[RUN]", " ".join(cmd))
    print("=" * 90 + "\n")

    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
    )

    return int(completed.returncode)


def main() -> int:
    args = parse_args()

    base_config_path = BASE_CONFIGS[args.dataset]
    base_cfg = load_yaml(base_config_path)

    cfg = build_smoke_config(
        base_cfg=base_cfg,
        args=args,
    )

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    smoke_config_path = TMP_CONFIG_DIR / f"smoke_{args.dataset}.yaml"
    save_yaml(smoke_config_path, cfg)

    print("=" * 90)
    print("FederatedTrustee smoke test")
    print("=" * 90)
    print(f"Dataset       : {args.dataset}")
    print(f"Config        : {smoke_config_path}")
    print(f"Round mode    : {args.round_mode}")
    print(f"Clients       : {args.num_clients}")
    print(f"Rounds        : {args.num_rounds}")
    print(f"Seed          : {args.seed}")
    print(f"Flip prob.    : {args.flip_probability}")
    print("=" * 90)

    before_run = find_latest_run(cfg)

    exit_code = run_pipeline(
        config_path=smoke_config_path,
        round_mode=args.round_mode,
    )

    after_run = find_latest_run(cfg)

    print("\n" + "=" * 90)
    print("Smoke test summary")
    print("=" * 90)
    print(f"Exit code     : {exit_code}")
    print(f"Config used   : {smoke_config_path}")

    if after_run is not None and after_run != before_run:
        print(f"New run       : {after_run}")
    elif after_run is not None:
        print(f"Latest run    : {after_run}")
    else:
        print("Latest run    : not found")

    if exit_code == 0:
        print("[OK] Smoke test completed successfully.")
    else:
        print("[FAIL] Smoke test failed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())