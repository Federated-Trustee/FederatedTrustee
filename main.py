from __future__ import annotations

import os

# Prevents Ray/Flower simulation warnings related to accelerator environment
# variables when running CPU-only experiments.
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import argparse
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.artifacts.io import create_run_dir, save_json
from src.config import load_config
from src.data.loading import load_dataset_from_config
from src.data.partition import make_iid_partitions
from src.fl.server import build_strategy, create_client_fn
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a federated learning experiment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nsl_kdd.yaml",
        help="Path to the experiment YAML configuration file.",
    )
    return parser.parse_args()


def build_server_app(strategy, num_rounds: int) -> ServerApp:
    """Build the Flower ServerApp using the provided strategy."""

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )

    return ServerApp(server_fn=server_fn)


def print_run_header(
    cfg: dict[str, Any],
    device,
    input_dim: int,
    num_clients: int,
    run_dir: Path,
) -> None:
    """Print a compact summary of the experiment before execution."""
    attack_cfg = cfg.get("attack", {})
    attack_enabled = bool(attack_cfg.get("enabled", False))

    print("=" * 72)
    print("Federated Learning Experiment")
    print("=" * 72)
    print(f"Dataset        : {cfg['dataset']['name']}")
    print(f"Device         : {device}")
    print(f"Input dim      : {input_dim}")
    print(f"Clients        : {num_clients}")
    print(f"Rounds         : {cfg['federated']['num_rounds']}")
    print(f"Attack enabled : {attack_enabled}")

    if attack_enabled:
        print(f"Attack type    : {attack_cfg.get('type')}")
        print(f"Malicious IDs  : {attack_cfg.get('malicious_client_ids', [])}")
        print(f"Flip prob.     : {attack_cfg.get('flip_probability')}")

    print(f"Run directory  : {run_dir}")
    print("=" * 72)


def main() -> None:
    """Run the full federated learning experiment."""
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["runtime"]["seed"])
    device = get_device(cfg["runtime"]["device"])

    loaded_dataset = load_dataset_from_config(cfg)
    data_bundle = loaded_dataset.data

    client_partitions = make_iid_partitions(
        X_raw=data_bundle.X_train_raw,
        y=data_bundle.y_train,
        num_clients=int(cfg["federated"]["num_clients"]),
        seed=int(cfg["runtime"]["seed"]),
    )

    run_dir = create_run_dir(
        base_dir=cfg["artifacts"]["run_dir"],
        experiment_name=cfg["experiment_name"],
    )

    save_json(cfg, run_dir / "config.json")

    print_run_header(
        cfg=cfg,
        device=device,
        input_dim=int(data_bundle.input_dim),
        num_clients=len(client_partitions),
        run_dir=run_dir,
    )

    client_fn = create_client_fn(
        config=cfg,
        data_bundle=data_bundle,
        client_partitions=client_partitions,
        device=device,
        run_dir=str(run_dir),
    )

    strategy = build_strategy(
        config=cfg,
        data_bundle=data_bundle,
        device=device,
    )

    server_app = build_server_app(
        strategy=strategy,
        num_rounds=int(cfg["federated"]["num_rounds"]),
    )
    client_app = fl.client.ClientApp(client_fn=client_fn)

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(cfg["federated"]["num_clients"]),
    )

    save_json(
        {
            "status": "completed",
            "dataset": cfg["dataset"]["name"],
            "num_rounds": int(cfg["federated"]["num_rounds"]),
            "num_clients": int(cfg["federated"]["num_clients"]),
            "device": str(device),
            "input_dim": int(data_bundle.input_dim),
            "run_dir": str(run_dir),
        },
        run_dir / "run_summary.json",
    )

    print("[DONE] Run finished successfully.")
    print(f"[DONE] Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()