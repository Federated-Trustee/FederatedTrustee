from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIGS = {
    "nsl_kdd": PROJECT_ROOT / "configs" / "nsl_kdd.yaml",
    "fiveg_nidd": PROJECT_ROOT / "configs" / "fiveg_nidd.yaml",
}

TMP_CONFIG_ROOT = PROJECT_ROOT / "tmp" / "paper_configs"


@dataclass(frozen=True)
class ExperimentSpec:
    group: str
    dataset: str
    malicious_client_ids: list[int]
    flip_probability: float
    seed: int

    @property
    def num_malicious(self) -> int:
        return len(self.malicious_client_ids)

    @property
    def experiment_name(self) -> str:
        return (
            f"paper_{self.group}_"
            f"{self.dataset}_"
            f"{format_probability(self.flip_probability)}_"
            f"m{self.num_malicious}_"
            f"seed{self.seed:03d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FederatedTrustee paper experiment grids.",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["exp1", "exp2", "round-analysis", "all"],
        required=True,
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nsl_kdd", "fiveg_nidd", "all"],
        default="all",
        help="Dataset filter. Default: all.",
    )
    parser.add_argument(
        "--round-mode",
        type=str,
        choices=["final", "all-rounds"],
        default="all-rounds",
        help="Pipeline round mode. Default: all-rounds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned experiments without executing them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of experiments to run.",
    )
    return parser.parse_args()


def format_probability(p: float) -> str:
    return f"p{int(round(float(p) * 100)):03d}"


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


def selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return ["nsl_kdd", "fiveg_nidd"]

    return [dataset_arg]


def build_exp1_specs(datasets: list[str]) -> list[ExperimentSpec]:
    """
    Experiment 1:
    Variation of flip probability with 2 malicious clients.

    Total expected:
        2 datasets x 4 p values x 5 seeds = 40 experiments
    """
    specs: list[ExperimentSpec] = []

    flip_probabilities = [1.0, 0.8, 0.6, 0.4]
    seeds = [1, 7, 21, 42, 84]

    for dataset in datasets:
        for p in flip_probabilities:
            for seed in seeds:
                specs.append(
                    ExperimentSpec(
                        group="exp1",
                        dataset=dataset,
                        malicious_client_ids=[0, 1],
                        flip_probability=p,
                        seed=seed,
                    )
                )

    return specs


def build_exp2_specs(datasets: list[str]) -> list[ExperimentSpec]:
    """
    Experiment 2:
    Variation of the number of malicious clients.

    Total expected:
        2 datasets x 2 malicious scenarios x 3 p values x 3 seeds = 36 experiments
    """
    specs: list[ExperimentSpec] = []

    malicious_scenarios = [
        [0],
        [0, 1, 2, 3],
    ]
    flip_probabilities = [1.0, 0.8, 0.6]
    seeds = [1, 21, 84]

    for dataset in datasets:
        for malicious_client_ids in malicious_scenarios:
            for p in flip_probabilities:
                for seed in seeds:
                    specs.append(
                        ExperimentSpec(
                            group="exp2",
                            dataset=dataset,
                            malicious_client_ids=malicious_client_ids,
                            flip_probability=p,
                            seed=seed,
                        )
                    )

    return specs


def build_round_analysis_specs(datasets: list[str]) -> list[ExperimentSpec]:
    """
    Experiment 3:
    Round analysis.

    This is a subset of Experiment 1:
        malicious_client_ids = [0, 1]
        p in {1.0, 0.6}
        seeds in {1, 7, 21, 42, 84}

    Total expected:
        2 datasets x 2 p values x 5 seeds = 20 experiments

    If Experiment 1 was already run with --round-mode all-rounds, this group
    does not need separate training.
    """
    specs: list[ExperimentSpec] = []

    flip_probabilities = [1.0, 0.6]
    seeds = [1, 7, 21, 42, 84]

    for dataset in datasets:
        for p in flip_probabilities:
            for seed in seeds:
                specs.append(
                    ExperimentSpec(
                        group="round_analysis",
                        dataset=dataset,
                        malicious_client_ids=[0, 1],
                        flip_probability=p,
                        seed=seed,
                    )
                )

    return specs


def build_specs(group: str, datasets: list[str]) -> list[ExperimentSpec]:
    if group == "exp1":
        return build_exp1_specs(datasets)

    if group == "exp2":
        return build_exp2_specs(datasets)

    if group == "round-analysis":
        return build_round_analysis_specs(datasets)

    if group == "all":
        return build_exp1_specs(datasets) + build_exp2_specs(datasets)

    raise ValueError(f"Unsupported group: {group}")


def build_config(base_cfg: dict[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    cfg["experiment_name"] = spec.experiment_name

    cfg["federated"]["num_clients"] = 10
    cfg["federated"]["num_rounds"] = 3
    cfg["federated"]["fraction_fit"] = 1.0
    cfg["federated"]["fraction_evaluate"] = 1.0
    cfg["federated"]["local_epochs"] = 1

    cfg["runtime"]["seed"] = int(spec.seed)

    cfg["attack"]["enabled"] = True
    cfg["attack"]["type"] = "label_flip"
    cfg["attack"]["malicious_client_ids"] = [int(cid) for cid in spec.malicious_client_ids]
    cfg["attack"]["source_label"] = 1
    cfg["attack"]["target_label"] = 0
    cfg["attack"]["flip_probability"] = float(spec.flip_probability)

    cfg.setdefault("trustee", {})
    cfg["trustee"].setdefault("enabled", True)
    cfg["trustee"].setdefault("round_to_analyze", 3)
    cfg["trustee"].setdefault("num_iter", 20)
    cfg["trustee"].setdefault("num_stability_iter", 5)
    cfg["trustee"].setdefault("samples_size", 0.3)

    cfg["paper"] = {
        "experiment_group": spec.group,
        "dataset": spec.dataset,
        "num_clients": 10,
        "num_rounds": 3,
        "num_malicious": spec.num_malicious,
        "malicious_client_ids": [int(cid) for cid in spec.malicious_client_ids],
        "flip_probability": float(spec.flip_probability),
        "seed": int(spec.seed),
    }

    return cfg


def config_path_for_spec(spec: ExperimentSpec) -> Path:
    return (
        TMP_CONFIG_ROOT
        / spec.group
        / spec.dataset
        / f"{spec.experiment_name}.yaml"
    )


def run_pipeline(config_path: Path, round_mode: str) -> int:
    cmd = [
        "bash",
        "run_pipeline.sh",
        str(config_path),
        round_mode,
    ]

    print("\n" + "=" * 100)
    print("[RUN]", " ".join(cmd))
    print("=" * 100)

    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
    )

    return int(completed.returncode)


def print_plan(specs: list[ExperimentSpec], round_mode: str) -> None:
    print("=" * 100)
    print("FederatedTrustee paper experiment plan")
    print("=" * 100)
    print(f"Total experiments: {len(specs)}")
    print(f"Round mode       : {round_mode}")
    print("=" * 100)

    for idx, spec in enumerate(specs, start=1):
        print(
            f"[{idx:03d}/{len(specs):03d}] "
            f"group={spec.group} | "
            f"dataset={spec.dataset} | "
            f"p={spec.flip_probability} | "
            f"m={spec.num_malicious} | "
            f"malicious={spec.malicious_client_ids} | "
            f"seed={spec.seed} | "
            f"experiment_name={spec.experiment_name}"
        )


def main() -> int:
    args = parse_args()

    datasets = selected_datasets(args.dataset)
    specs = build_specs(args.group, datasets)

    if args.limit is not None:
        specs = specs[: int(args.limit)]

    print_plan(specs, args.round_mode)

    if args.dry_run:
        print("\n[DRY RUN] No experiments were executed.")
        return 0

    failures: list[ExperimentSpec] = []

    for idx, spec in enumerate(specs, start=1):
        print("\n" + "#" * 100)
        print(f"[{idx}/{len(specs)}] Running {spec.experiment_name}")
        print("#" * 100)

        base_cfg = load_yaml(BASE_CONFIGS[spec.dataset])
        cfg = build_config(base_cfg=base_cfg, spec=spec)

        config_path = config_path_for_spec(spec)
        save_yaml(config_path, cfg)

        exit_code = run_pipeline(
            config_path=config_path,
            round_mode=args.round_mode,
        )

        if exit_code != 0:
            failures.append(spec)
            print(f"[FAIL] {spec.experiment_name} | exit_code={exit_code}")
        else:
            print(f"[OK] {spec.experiment_name}")

    print("\n" + "=" * 100)
    print("Final summary")
    print("=" * 100)
    print(f"Total planned : {len(specs)}")
    print(f"Failures      : {len(failures)}")
    print(f"Temp configs  : {TMP_CONFIG_ROOT}")

    if failures:
        print("\nFailed experiments:")
        for spec in failures:
            print(f" - {spec.experiment_name}")
        return 1

    print("[OK] All experiments completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())