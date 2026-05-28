from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np
import torch

from src.artifacts.io import save_model_checkpoint
from src.attacks.label_flip import apply_label_flip
from src.data.partition import transform_features
from src.models.factory import build_model
from src.models.parameters import get_model_parameters, set_model_parameters
from src.training.eval import evaluate_model
from src.training.train import train_one_model


class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing one federated client.

    Each client:
    - receives the current global model parameters;
    - trains locally on its own partition;
    - optionally applies label-flip poisoning if marked as malicious;
    - saves a checkpoint after each federated round;
    - evaluates using the global test set.
    """

    def __init__(
        self,
        client_id: int,
        config: dict[str, Any],
        data_bundle: Any,
        client_partition: Any,
        device: torch.device,
        run_dir: str,
    ) -> None:
        self.client_id = int(client_id)
        self.config = config
        self.data_bundle = data_bundle
        self.client_partition = client_partition
        self.device = device
        self.run_dir = run_dir

        self.model = build_model(
            config=config,
            input_dim=int(data_bundle.input_dim),
        )

        self.X_train = transform_features(
            client_partition.X_raw,
            data_bundle.transformer,
        )
        self.y_train_clean = client_partition.y.to_numpy()

        self.attack_stats = {
            "attack_applied": 0.0,
            "eligible_samples": 0.0,
            "flipped_samples": 0.0,
        }
        self.y_train = self._maybe_poison_labels(self.y_train_clean)

        self.X_test = transform_features(
            data_bundle.X_test_raw,
            data_bundle.transformer,
        )
        self.y_test = data_bundle.y_test.to_numpy()

    def _maybe_poison_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Apply label-flip poisoning if this client is configured as malicious.
        """
        attack_cfg = self.config.get("attack", {})

        if not attack_cfg.get("enabled", False):
            return y.copy()

        if attack_cfg.get("type") != "label_flip":
            return y.copy()

        malicious_ids = {
            int(client_id)
            for client_id in attack_cfg.get("malicious_client_ids", [])
        }

        if self.client_id not in malicious_ids:
            return y.copy()

        source_label = int(attack_cfg["source_label"])
        target_label = int(attack_cfg["target_label"])
        flip_probability = float(attack_cfg.get("flip_probability", 1.0))

        y_poisoned = apply_label_flip(
            y=y,
            source_label=source_label,
            target_label=target_label,
            flip_probability=flip_probability,
            seed=int(self.config["runtime"]["seed"]) + self.client_id,
        )

        flipped = int(np.sum(y != y_poisoned))
        eligible = int(np.sum(y == source_label))

        self.attack_stats = {
            "attack_applied": 1.0,
            "eligible_samples": float(eligible),
            "flipped_samples": float(flipped),
        }

        return y_poisoned

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return current local model parameters."""
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """
        Train the local model for one federated round.
        """
        set_model_parameters(self.model, parameters)

        metrics = train_one_model(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            device=self.device,
            epochs=int(self.config["federated"]["local_epochs"]),
            batch_size=int(self.config["training"]["batch_size"]),
            learning_rate=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )

        round_idx = int(config.get("server_round", 0))

        save_model_checkpoint(
            model_state_dict=self.model.state_dict(),
            run_dir=self.run_dir,
            round_idx=round_idx,
            client_id=self.client_id,
        )

        metrics["checkpoint_saved"] = 1.0
        metrics["attack_applied"] = self.attack_stats["attack_applied"]
        metrics["eligible_samples"] = self.attack_stats["eligible_samples"]
        metrics["flipped_samples"] = self.attack_stats["flipped_samples"]

        updated_parameters = get_model_parameters(self.model)

        return updated_parameters, len(self.y_train), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """
        Evaluate the current global model on the global test set.
        """
        set_model_parameters(self.model, parameters)

        metrics = evaluate_model(
            model=self.model,
            X=self.X_test,
            y=self.y_test,
            device=self.device,
            batch_size=int(self.config["training"]["batch_size"]),
        )

        return (
            float(metrics["loss"]),
            len(self.y_test),
            {"accuracy": float(metrics["accuracy"])},
        )