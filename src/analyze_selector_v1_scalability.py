"""Analyze computational scalability of frozen Selector v1 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("results/selector_v1_aggregate_report.csv")
DEFAULT_OUTPUT = Path("results/selector_v1_scalability.csv")


def build_scalability_table(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Build scalability metrics for every model/training condition."""

    required = {
        "model_id",
        "n_states",
        "train_dataset",
        "train_trajectories",
        "candidate_space",
        "systems_evaluated",
        "search_fraction",
        "coordinate_rounds",
        "validation_macro_nrmse",
        "validation_status",
    }

    missing = required - set(aggregate.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    table = aggregate[
        [
            "model_id",
            "n_states",
            "train_dataset",
            "train_trajectories",
            "candidate_space",
            "systems_evaluated",
            "search_fraction",
            "search_percent",
            "coordinate_rounds",
            "validation_macro_nrmse",
            "validation_status",
        ]
    ].copy()

    table["space_per_evaluated_system"] = (
        table["candidate_space"]
        / table["systems_evaluated"]
    )

    table = table.sort_values(
        by=[
            "n_states",
            "model_id",
            "train_trajectories",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return table


def print_scalability_summary(
    table: pd.DataFrame,
) -> None:
    """Print key Selector v1 scalability statistics."""

    largest_space_row = table.loc[
        table["candidate_space"].idxmax()
    ]

    smallest_fraction_row = table.loc[
        table["search_fraction"].idxmin()
    ]

    most_evaluated_row = table.loc[
        table["systems_evaluated"].idxmax()
    ]

    print("=" * 100)
    print("SELECTOR V1 SCALABILITY ANALYSIS")
    print("=" * 100)

    print(f"Conditions: {len(table)}")
    print(
        f"Models: {table['model_id'].nunique()}"
    )

    print()
    print("Largest candidate space:")
    print(
        f"  {largest_space_row['model_id']} "
        f"{largest_space_row['train_dataset']}"
    )
    print(
        f"  candidate_space = "
        f"{int(largest_space_row['candidate_space']):,}"
    )
    print(
        f"  systems_evaluated = "
        f"{int(largest_space_row['systems_evaluated']):,}"
    )

    print()
    print("Smallest explored fraction:")
    print(
        f"  {smallest_fraction_row['model_id']} "
        f"{smallest_fraction_row['train_dataset']}"
    )
    print(
        f"  search_fraction = "
        f"{smallest_fraction_row['search_fraction']:.6e}"
    )
    print(
        f"  search_percent = "
        f"{smallest_fraction_row['search_percent']:.6e}%"
    )
    print(
        "  approximately 1 evaluated system per "
        f"{smallest_fraction_row['space_per_evaluated_system']:,.0f} "
        "possible systems"
    )

    print()
    print("Largest number of evaluated systems:")
    print(
        f"  {most_evaluated_row['model_id']} "
        f"{most_evaluated_row['train_dataset']}"
    )
    print(
        f"  systems_evaluated = "
        f"{int(most_evaluated_row['systems_evaluated']):,}"
    )

    print()
    print("By state count:")
    print(
        table.groupby("n_states")
        .agg(
            max_candidate_space=(
                "candidate_space",
                "max",
            ),
            max_systems_evaluated=(
                "systems_evaluated",
                "max",
            ),
            min_search_fraction=(
                "search_fraction",
                "min",
            ),
        )
        .to_string()
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze computational scalability of frozen "
            "Selector v1 results."
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

    table = build_scalability_table(
        aggregate
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        args.output,
        index=False,
    )

    print_scalability_summary(table)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
