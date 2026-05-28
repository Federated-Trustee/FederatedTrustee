from __future__ import annotations

from typing import Any, Callable

import flwr as fl
import numpy as np
import torch

from src.data.partition import transform_features
from src.fl.client import FlowerClient
from src.models.factory import build_model
from src.models.parameters import set_model_parameters
from src.training.eval import evaluate_model


def create_client_fn(
    config: dict[str, Any],
    data_bundle: Any,
    client_partitions: list[Any],
    device: torch.device,
    run_dir: str,
) -> Callable:
    """
    Create a Flower client function.

    Flower uses this function to instantiate clients during simulation.
    """
    num_clients = len(client_partitions)

    def client_fn(context: fl.common.Context):
        client_id = int(context.node_config["partition-id"])

        if client_id < 0 or client_id >= num_clients:
            raise ValueError(
                f"Invalid client_id={client_id}. "
                f"Expected a value in [0, {num_clients - 1}]."
            )

        partition = client_partitions[client_id]

        client = FlowerClient(
            client_id=client_id,
            config=config,
            data_bundle=data_bundle,
            client_partition=partition,
            device=device,
            run_dir=run_dir,
        )

        return client.to_client()

    return client_fn


def create_centralized_evaluate_fn(
    config: dict[str, Any],
    data_bundle: Any,
    device: torch.device,
):
    """
    Create the centralized evaluation function used by the Flower server.
    """
    X_test = transform_features(
        data_bundle.X_test_raw,
        data_bundle.transformer,
    )
    y_test = data_bundle.y_test.to_numpy()

    def evaluate(
        server_round: int,
        parameters: list[np.ndarray],
        config_dict: dict[str, Any],
    ):
        model = build_model(
            config=config,
            input_dim=int(data_bundle.input_dim),
        )
        set_model_parameters(model, parameters)

        metrics = evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )

        print(
            f"[CENTRAL EVAL] round={server_round:02d} "
            f"loss={metrics['loss']:.6f} "
            f"accuracy={metrics['accuracy']:.6f}"
        )

        return (
            float(metrics["loss"]),
            {"centralized_accuracy": float(metrics["accuracy"])},
        )

    return evaluate


def weighted_average(
    metrics: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """
    Compute a weighted average of client metrics.

    Each metric is weighted by the number of examples reported by the client.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    metric_keys = sorted(
        {
            key
            for _, metric_dict in metrics
            for key in metric_dict.keys()
        }
    )

    aggregated: dict[str, float] = {}

    for key in metric_keys:
        weighted_sum = 0.0
        used_examples = 0

        for num_examples, metric_dict in metrics:
            if key not in metric_dict:
                continue

            weighted_sum += num_examples * float(metric_dict[key])
            used_examples += num_examples

        if used_examples > 0:
            aggregated[key] = weighted_sum / used_examples

    return aggregated


def fit_config(server_round: int) -> dict[str, Any]:
    """
    Send the current federated round index to each client.
    """
    return {"server_round": int(server_round)}


def build_strategy(
    config: dict[str, Any],
    data_bundle: Any,
    device: torch.device,
) -> fl.server.strategy.FedAvg:
    """
    Build the FedAvg strategy used in the simulation.
    """
    fed_cfg = config["federated"]
    num_clients = int(fed_cfg["num_clients"])

    return fl.server.strategy.FedAvg(
        fraction_fit=float(fed_cfg["fraction_fit"]),
        fraction_evaluate=float(fed_cfg["fraction_evaluate"]),
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        evaluate_fn=create_centralized_evaluate_fn(
            config=config,
            data_bundle=data_bundle,
            device=device,
        ),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )