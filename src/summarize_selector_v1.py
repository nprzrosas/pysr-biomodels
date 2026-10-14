"""Summarize the frozen Selector v1 aggregate report by model.

This module reads the frozen 40-row Selector v1 aggregate report and creates
a model-level summary. It does not run PySR, system selection, held-out
testing, or ground-truth evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/selector_v1_aggregate_report.csv")
DEFAULT_OUTPUT = Path("results/selector_v1_model_summary.csv")

TRAIN_SIZES = (4, 8, 16, 32)


def validation_value(
    group: pd.DataFrame,
    train_size: int,
) -> float:
    """Return validation macro NRMSE for one training size."""

    rows = group.loc[
        group["train_trajectories"] == train_size,
        "validation_macro_nrmse",
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for train size {train_size}; "
            f"found {len(rows)}."
        )

    return float(rows.iloc[0])


def summarize_model(group: pd.DataFrame) -> dict:
    """Summarize all Selector v1 training conditions for one model."""

    model_ids = group["model_id"].unique()

    if len(model_ids) != 1:
        raise ValueError("Model group contains multiple model IDs.")

    model_id = model_ids[0]

    state_counts = group["n_states"].unique()

    if len(state_counts) != 1:
        raise ValueError(
            f"Inconsistent state counts for {model_id}: "
            f"{state_counts.tolist()}"
        )

    observed_train_sizes = set(
        group["train_trajectories"].astype(int)
    )

    expected_train_sizes = set(TRAIN_SIZES)

    if observed_train_sizes != expected_train_sizes:
        raise ValueError(
            f"{model_id} has training sizes "
            f"{sorted(observed_train_sizes)}, expected "
            f"{sorted(expected_train_sizes)}."
        )

    validation = {
        train_size: validation_value(group, train_size)
        for train_size in TRAIN_SIZES
    }

    finite_validation = {
        train_size: value
        for train_size, value in validation.items()
        if np.isfinite(value)
    }

    if finite_validation:
        best_train_size = min(
            finite_validation,
            key=finite_validation.get,
        )
        best_validation = finite_validation[best_train_size]
    else:
        best_train_size = np.nan
        best_validation = np.inf

    train04 = validation[4]
    train32 = validation[32]

    if (
        np.isfinite(train04)
        and np.isfinite(train32)
        and train04 != 0.0
    ):
        change_04_to_32_percent = (
            (train32 - train04)
            / train04
            * 100.0
        )
        improvement_04_to_32_percent = (
            (train04 - train32)
            / train04
            * 100.0
        )
    else:
        change_04_to_32_percent = np.nan
        improvement_04_to_32_percent = np.nan

    successful_selections = int(
        (group["validation_status"] == "success").sum()
    )
    failed_selections = int(
        (group["validation_status"] != "success").sum()
    )

    return {
        "model_id": model_id,
        "n_states": int(state_counts[0]),
        "validation_nrmse_train04": validation[4],
        "validation_nrmse_train08": validation[8],
        "validation_nrmse_train16": validation[16],
        "validation_nrmse_train32": validation[32],
        "best_validation_nrmse": best_validation,
        "best_train_trajectories": best_train_size,
        "change_04_to_32_percent": change_04_to_32_percent,
        "improvement_04_to_32_percent": (
            improvement_04_to_32_percent
        ),
        "successful_selections": successful_selections,
        "failed_selections": failed_selections,
        "max_candidate_space": int(
            group["candidate_space"].max()
        ),
        "min_candidate_space": int(
            group["candidate_space"].min()
        ),
        "total_systems_evaluated": int(
            group["systems_evaluated"].sum()
        ),
        "max_systems_evaluated": int(
            group["systems_evaluated"].max()
        ),
        "min_search_fraction": float(
            group["search_fraction"].min()
        ),
        "max_search_fraction": float(
            group["search_fraction"].max()
        ),
        "max_coordinate_rounds": int(
            group["coordinate_rounds"].max()
        ),
    }


def build_model_summary(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Build one Selector v1 summary row per model."""

    rows = []

    for _, group in aggregate.groupby(
        "model_id",
        sort=False,
    ):
        rows.append(
            summarize_model(group)
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=[
            "n_states",
            "model_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print a compact model-level summary."""

    print("=" * 110)
    print("SELECTOR V1 MODEL-LEVEL SUMMARY")
    print("=" * 110)

    print(f"Models: {len(summary)}")

    print()

    columns = [
        "model_id",
        "n_states",
        "validation_nrmse_train04",
        "validation_nrmse_train08",
        "validation_nrmse_train16",
        "validation_nrmse_train32",
        "best_validation_nrmse",
        "best_train_trajectories",
        "successful_selections",
        "failed_selections",
    ]

    print(
        summary[columns].to_string(
            index=False,
        )
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a model-level summary from the frozen "
            "Selector v1 aggregate report."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    aggregate = pd.read_csv(
        args.input,
        dtype={"selected_ranks": str},
    )

    summary = build_model_summary(
        aggregate
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.output,
        index=False,
    )

    print_summary(summary)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
