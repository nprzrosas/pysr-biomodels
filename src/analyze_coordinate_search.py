"""Retrospectively compare coordinate search with exhaustive system selection."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def ranking_columns():
    """Return the frozen validation ranking columns."""
    return [
        "validation_macro_nrmse",
        "complexity_total",
        "training_loss_total",
        "system_id",
    ]


def candidate_rank_columns(frame):
    """Return candidate-rank columns in their stored state order."""
    return [
        column
        for column in frame.columns
        if column.startswith("candidate_rank_")
    ]


def system_tuple(
    row,
    rank_columns,
):
    """Convert one system row into its candidate-rank tuple."""
    return tuple(
        int(row[column])
        for column in rank_columns
    )


def build_system_lookup(
    frame,
    rank_columns,
):
    """Map candidate-rank tuples to rows in the exhaustive table."""
    lookup = {}

    for _, row in frame.iterrows():
        ranks = system_tuple(
            row,
            rank_columns,
        )

        lookup[ranks] = row

    return lookup


def row_sort_key(row):
    """Return the frozen validation ranking key."""
    return (
        float(row["validation_macro_nrmse"]),
        int(row["complexity_total"]),
        float(row["training_loss_total"]),
        int(row["system_id"]),
    )


def coordinate_search(
    frame,
    rank_columns,
):
    """Run deterministic coordinate search on the validation landscape."""
    lookup = build_system_lookup(
        frame,
        rank_columns,
    )

    maximum_ranks = tuple(
        int(frame[column].max())
        for column in rank_columns
    )

    current = tuple(
        1
        for _ in rank_columns
    )

    if current not in lookup:
        raise RuntimeError(
            f"Initial system {current} does not exist."
        )

    evaluated = {
        current
    }

    rounds = 0

    while True:
        rounds += 1

        changed = False

        for state_index in range(
            len(rank_columns)
        ):
            proposals = []

            for candidate_rank in range(
                1,
                maximum_ranks[state_index] + 1,
            ):
                proposal = list(
                    current
                )

                proposal[state_index] = candidate_rank

                proposal = tuple(
                    proposal
                )

                if proposal not in lookup:
                    continue

                evaluated.add(
                    proposal
                )

                proposals.append(
                    proposal
                )

            if not proposals:
                continue

            best_proposal = min(
                proposals,
                key=lambda ranks: row_sort_key(
                    lookup[ranks]
                ),
            )

            if best_proposal != current:
                current = best_proposal
                changed = True

        if not changed:
            break

    best_row = lookup[
        current
    ]

    return (
        best_row,
        current,
        rounds,
        evaluated,
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively compare coordinate search with exhaustive "
            "validation for one model."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels identifier, for example BIOMD0000000156.",
    )

    return parser.parse_args()


def main():
    """Compare coordinate search against exhaustive validation."""
    args = parse_arguments()

    model_id = args.model_id

    results_root = (
        PROJECT_ROOT
        / "results"
        / model_id
        / "generic_exact_experiment"
    )

    print("=" * 100)
    print("COORDINATE SEARCH VS EXHAUSTIVE")
    print("=" * 100)
    print(
        "Model:",
        model_id,
    )

    for dataset_name in DATASETS:
        path = (
            results_root
            / dataset_name
            / "validation_system_ranking.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Validation ranking not found: {path}"
            )

        frame = pd.read_csv(
            path
        )

        rank_columns = candidate_rank_columns(
            frame
        )

        exhaustive = (
            frame
            .sort_values(
                ranking_columns(),
                kind="stable",
            )
            .iloc[0]
        )

        exhaustive_system = int(
            exhaustive["system_id"]
        )

        exhaustive_nrmse = float(
            exhaustive["validation_macro_nrmse"]
        )

        exhaustive_ranks = system_tuple(
            exhaustive,
            rank_columns,
        )

        (
            best,
            best_ranks,
            rounds,
            evaluated,
        ) = coordinate_search(
            frame,
            rank_columns,
        )

        best_system = int(
            best["system_id"]
        )

        best_nrmse = float(
            best["validation_macro_nrmse"]
        )

        relative_gap = (
            best_nrmse
            / exhaustive_nrmse
            - 1.0
        )

        search_fraction = (
            len(evaluated)
            / len(frame)
        )

        print()
        print("=" * 100)
        print(dataset_name)
        print("=" * 100)

        print(
            "Exhaustive system:",
            exhaustive_system,
        )

        print(
            "Exhaustive ranks:",
            exhaustive_ranks,
        )

        print(
            "Exhaustive NRMSE:",
            exhaustive_nrmse,
        )

        print(
            "Coordinate system:",
            best_system,
        )

        print(
            "Coordinate ranks:",
            best_ranks,
        )

        print(
            "Coordinate NRMSE:",
            best_nrmse,
        )

        print(
            "Relative gap:",
            f"{100.0 * relative_gap:.3f}%",
        )

        print(
            "Rounds:",
            rounds,
        )

        print(
            "Unique systems examined:",
            f"{len(evaluated)} / {len(frame)}",
        )

        print(
            "Search fraction:",
            f"{100.0 * search_fraction:.1f}%",
        )

        print(
            "Matches exhaustive:",
            best_system
            == exhaustive_system,
        )


if __name__ == "__main__":
    main()