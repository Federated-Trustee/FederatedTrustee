from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consolidated CSV with local-model and TRUSTEE agreement "
            "scores per client, round, and run."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing experiment runs. Default: runs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="client_round_agreements.csv",
        help="Output CSV path. Default: client_round_agreements.csv",
    )
    parser.add_argument(
        "--run-name-contains",
        type=str,
        default=None,
        help=(
            "Optional substring filter for run directory names. "
            "Example: --run-name-contains 20260518"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from disk, returning None if unavailable or invalid."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def read_agreement_summary(path: Path) -> dict[int, float]:
    """Read agreement_summary.csv as client_id -> mean_agreement."""
    out: dict[int, float] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            out[int(row["client_id"])] = float(row["mean_agreement"])

    return out


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Extract experiment metadata from a run's config.json."""
    config = load_json(run_dir / "config.json")

    if config is None:
        return {
            "dataset": None,
            "seed": None,
            "flip_probability": None,
            "num_malicious": None,
            "malicious_clients": [],
        }

    malicious_clients = config.get("attack", {}).get("malicious_client_ids", [])

    if not isinstance(malicious_clients, list):
        malicious_clients = []

    return {
        "dataset": config.get("dataset", {}).get("name"),
        "seed": config.get("runtime", {}).get("seed"),
        "flip_probability": config.get("attack", {}).get("flip_probability"),
        "num_malicious": len(malicious_clients),
        "malicious_clients": [int(client_id) for client_id in malicious_clients],
    }


def parse_round_name(path: Path) -> int | None:
    """Parse round index from a directory name such as round_003."""
    name = path.name

    if not name.startswith("round_"):
        return None

    try:
        return int(name.replace("round_", ""))
    except ValueError:
        return None


def collect_agreement_rounds(run_dir: Path, output_group: str) -> dict[int, Path]:
    """
    Collect agreement_summary.csv files for one output group.

    New layout:
        agreement/<output_group>/round_003/agreement_summary.csv

    Backward compatibility for local models:
        agreement/round_003/agreement_summary.csv
    """
    out: dict[int, Path] = {}

    grouped_dir = run_dir / "agreement" / output_group

    if grouped_dir.exists():
        for round_dir in sorted(grouped_dir.glob("round_*")):
            round_idx = parse_round_name(round_dir)
            summary_path = round_dir / "agreement_summary.csv"

            if round_idx is not None and summary_path.exists():
                out[round_idx] = summary_path

    if output_group == "local_models":
        legacy_dir = run_dir / "agreement"

        for round_dir in sorted(legacy_dir.glob("round_*")):
            round_idx = parse_round_name(round_dir)
            summary_path = round_dir / "agreement_summary.csv"

            if round_idx is not None and summary_path.exists() and round_idx not in out:
                out[round_idx] = summary_path

    return out


def iter_run_dirs(
    runs_dir: Path,
    run_name_contains: str | None,
) -> list[Path]:
    """Return sorted run directories, optionally filtered by substring."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    ]

    if run_name_contains:
        run_dirs = [
            path
            for path in run_dirs
            if run_name_contains in path.name
        ]

    return sorted(run_dirs)


def build_rows(
    runs_dir: Path,
    run_name_contains: str | None,
) -> list[dict[str, Any]]:
    """Build consolidated agreement rows across runs."""
    rows: list[dict[str, Any]] = []

    for run_dir in iter_run_dirs(runs_dir, run_name_contains):
        metadata = load_run_metadata(run_dir)

        malicious_set = set(metadata["malicious_clients"])

        local_rounds = collect_agreement_rounds(
            run_dir=run_dir,
            output_group="local_models",
        )
        trustee_rounds = collect_agreement_rounds(
            run_dir=run_dir,
            output_group="trustee_pruned_trees",
        )

        available_rounds = sorted(
            set(local_rounds.keys()) | set(trustee_rounds.keys())
        )

        for round_idx in available_rounds:
            local_map = (
                read_agreement_summary(local_rounds[round_idx])
                if round_idx in local_rounds
                else {}
            )
            trustee_map = (
                read_agreement_summary(trustee_rounds[round_idx])
                if round_idx in trustee_rounds
                else {}
            )

            client_ids = sorted(set(local_map.keys()) | set(trustee_map.keys()))

            for client_id in client_ids:
                rows.append(
                    {
                        "dataset": metadata["dataset"],
                        "seed": metadata["seed"],
                        "flip_probability": metadata["flip_probability"],
                        "num_malicious": metadata["num_malicious"],
                        "malicious_clients": str(metadata["malicious_clients"]),
                        "round": int(round_idx),
                        "client_id": int(client_id),
                        "is_malicious": int(client_id) in malicious_set,
                        "local_mean_agreement": local_map.get(client_id, ""),
                        "trustee_mean_agreement": trustee_map.get(client_id, ""),
                        "run_dir": str(run_dir),
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            float(row["flip_probability"])
            if row["flip_probability"] is not None
            else -1.0,
            int(row["num_malicious"])
            if row["num_malicious"] is not None
            else -1,
            int(row["seed"])
            if row["seed"] is not None
            else -1,
            int(row["round"]),
            int(row["client_id"]),
        )
    )

    return rows


def save_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Save consolidated rows as CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "seed",
        "flip_probability",
        "num_malicious",
        "malicious_clients",
        "round",
        "client_id",
        "is_malicious",
        "local_mean_agreement",
        "trustee_mean_agreement",
        "run_dir",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    output_csv = Path(args.output_csv)

    rows = build_rows(
        runs_dir=runs_dir,
        run_name_contains=args.run_name_contains,
    )

    save_rows(rows, output_csv)

    print(f"[OK] Output CSV: {output_csv}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()