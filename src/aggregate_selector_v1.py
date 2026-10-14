"""Aggregate frozen Selector v1 results across benchmark models.

This module reads existing coordinate-selection artifacts and PySR candidate
pools. It does not run inference, validation, held-out testing, or inspect
ground-truth SBML equations.

The resulting table provides a frozen summary of Selector v1 behavior before
held-out test evaluation or mechanistic-recovery analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("results")
EXPERIMENT_NAME = "generic_exact_experiment"
TRAIN_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def discover_models(results_root: Path) -> list[str]:
    """Return model IDs containing generic exact experiment results."""

    model_ids = []

    if not results_root.exists():
        return model_ids

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue

        experiment_dir = model_dir / EXPERIMENT_NAME

        if experiment_dir.is_dir():
            model_ids.append(model_dir.name)

    return model_ids


def candidate_files(train_dir: Path) -> list[Path]:
    """Return candidate-pool files for all states in deterministic order."""

    return sorted(
        train_dir.glob("state_*/top_8_by_loss.csv"),
        key=lambda path: path.parent.name,
    )


def candidate_counts(train_dir: Path) -> list[int]:
    """Return the number of retained PySR candidates for every state."""

    files = candidate_files(train_dir)

    return [
        len(pd.read_csv(path))
        for path in files
    ]


def candidate_space_size(counts: list[int]) -> int:
    """Return Cartesian candidate-system space size."""

    if not counts:
        return 0

    return math.prod(counts)


def load_coordinate_summary(train_dir: Path) -> pd.DataFrame | None:
    """Load the frozen coordinate-selection summary for one training size."""

    path = (
        train_dir
        / "coordinate_selection"
        / "coordinate_selected_system.csv"
    )

    if not path.exists():
        return None

    frame = pd.read_csv(path)

    if frame.empty:
        return None

    return frame


def first_available(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    default=None,
):
    """Return the first available value among alternative column names."""

    for column in columns:
        if column in frame.columns:
            return frame.iloc[0][column]

    return default


def aggregate_training_result(
    *,
    model_id: str,
    train_name: str,
    train_dir: Path,
) -> dict | None:
    """Aggregate one model/training-size Selector v1 result."""

    selection = load_coordinate_summary(train_dir)

    if selection is None:
        return None

    counts = candidate_counts(train_dir)

    if not counts:
        raise RuntimeError(
            f"No candidate pools found for {model_id}/{train_name}"
        )

    n_states = len(counts)
    space_size = candidate_space_size(counts)

    systems_evaluated = first_available(
        selection,
        (
            "unique_systems_evaluated",
            "systems_evaluated",
        ),
    )

    rounds = first_available(
        selection,
        (
            "coordinate_rounds",
            "rounds",
        ),
    )

    validation_nrmse = first_available(
        selection,
        (
            "validation_macro_nrmse",
            "macro_nrmse",
        ),
    )

    validation_status = first_available(
        selection,
        (
            "validation_status",
            "status",
        ),
    )

    system_id = first_available(
        selection,
        (
            "system_id",
            "canonical_system_id",
        ),
    )

    if systems_evaluated is None:
        search_fraction = float("nan")
        search_percent = float("nan")
    else:
        systems_evaluated = int(systems_evaluated)
        search_fraction = systems_evaluated / space_size
        search_percent = 100.0 * search_fraction

    selected_ranks = first_available(
        selection,
        (
            "candidate_rank_signature",
        ),
        default="",
    )

    return {
        "model_id": model_id,
        "n_states": n_states,
        "train_dataset": train_name,
        "train_trajectories": int(train_name.split("_")[1]),
        "candidate_counts": ",".join(str(value) for value in counts),
        "candidate_space": space_size,
        "systems_evaluated": systems_evaluated,
        "search_fraction": search_fraction,
        "search_percent": search_percent,
        "coordinate_rounds": rounds,
        "canonical_system_id": system_id,
        "selected_ranks": selected_ranks,
        "validation_macro_nrmse": validation_nrmse,
        "validation_status": validation_status,
    }


def build_aggregate_report(
    *,
    results_root: Path,
) -> pd.DataFrame:
    """Build the frozen Selector v1 aggregate table."""

    rows = []

    for model_id in discover_models(results_root):
        experiment_dir = (
            results_root
            / model_id
            / EXPERIMENT_NAME
        )

        for train_name in TRAIN_DATASETS:
            train_dir = experiment_dir / train_name

            if not train_dir.is_dir():
                continue

            row = aggregate_training_result(
                model_id=model_id,
                train_name=train_name,
                train_dir=train_dir,
            )

            if row is not None:
                rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame = frame.sort_values(
        by=[
            "n_states",
            "model_id",
            "train_trajectories",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return frame


def print_report(frame: pd.DataFrame) -> None:
    """Print a compact Selector v1 report."""

    print("=" * 100)
    print("FROZEN SELECTOR V1 AGGREGATE REPORT")
    print("=" * 100)

    if frame.empty:
        print("No coordinate-selection results found.")
        return

    print(f"Models: {frame['model_id'].nunique()}")
    print(f"Model/training results: {len(frame)}")
    print(
        "State-count range:",
        f"{frame['n_states'].min()}-{frame['n_states'].max()}",
    )

    print()
    print(
        frame[
            [
                "model_id",
                "n_states",
                "train_dataset",
                "candidate_space",
                "systems_evaluated",
                "search_percent",
                "coordinate_rounds",
                "validation_macro_nrmse",
                "validation_status",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Aggregate frozen Selector v1 benchmark results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/selector_v1_aggregate_report.csv"
        ),
    )

    args = parser.parse_args()

    frame = build_aggregate_report(
        results_root=args.results_root,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        args.output,
        index=False,
    )

    print_report(frame)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
