"""Retrospectively compare beam search with exhaustive system selection."""

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

BEAM_WIDTHS = (
    2,
    4,
    8,
    16,
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


def expand_system(
    ranks,
    maximum_ranks,
):
    """Generate all one-coordinate substitutions of one system."""
    neighbors = set()

    for state_index in range(len(ranks)):
        for candidate_rank in range(
            1,
            maximum_ranks[state_index] + 1,
        ):
            if candidate_rank == ranks[state_index]:
                continue

            neighbor = list(ranks)

            neighbor[state_index] = candidate_rank

            neighbors.add(
                tuple(neighbor)
            )

    return neighbors


def beam_search(
    frame,
    rank_columns,
    beam_width,
):
    """Run deterministic beam search over the exhaustive validation landscape."""
    lookup = build_system_lookup(
        frame,
        rank_columns,
    )

    maximum_ranks = tuple(
        int(frame[column].max())
        for column in rank_columns
    )

    start = tuple(
        1
        for _ in rank_columns
    )

    if start not in lookup:
        raise RuntimeError(
            f"Initial system {start} does not exist."
        )

    beam = [
        start
    ]

    evaluated = {
        start
    }

    rounds = 0

    while True:
        rounds += 1

        proposed = set(
            beam
        )

        for ranks in beam:
            proposed.update(
                expand_system(
                    ranks,
                    maximum_ranks,
                )
            )

        proposed = {
            ranks
            for ranks in proposed
            if ranks in lookup
        }

        evaluated.update(
            proposed
        )

        ordered = sorted(
            proposed,
            key=lambda ranks: row_sort_key(
                lookup[ranks]
            ),
        )

        new_beam = ordered[
            :beam_width
        ]

        if new_beam == beam:
            break

        beam = new_beam

    best_ranks = beam[
        0
    ]

    best_row = lookup[
        best_ranks
    ]

    return (
        best_row,
        best_ranks,
        rounds,
        evaluated,
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively compare beam search with exhaustive "
            "validation for one model."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels identifier, for example BIOMD0000000079.",
    )

    return parser.parse_args()


def main():
    """Compare several beam widths against exhaustive validation."""
    args = parse_arguments()

    model_id = args.model_id

    results_root = (
        PROJECT_ROOT
        / "results"
        / model_id
        / "generic_exact_experiment"
    )

    print("=" * 100)
    print("BEAM SEARCH VS EXHAUSTIVE")
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
            "Total systems:",
            len(frame),
        )

        for beam_width in BEAM_WIDTHS:
            (
                best,
                best_ranks,
                rounds,
                evaluated,
            ) = beam_search(
                frame,
                rank_columns,
                beam_width,
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
            print(
                f"Beam width: {beam_width}"
            )

            print(
                "  system:",
                best_system,
            )

            print(
                "  ranks:",
                best_ranks,
            )

            print(
                "  NRMSE:",
                best_nrmse,
            )

            print(
                "  relative gap:",
                f"{100.0 * relative_gap:.3f}%",
            )

            print(
                "  rounds:",
                rounds,
            )

            print(
                "  unique systems examined:",
                f"{len(evaluated)} / {len(frame)}",
            )

            print(
                "  search fraction:",
                f"{100.0 * search_fraction:.1f}%",
            )

            print(
                "  matches exhaustive:",
                best_system
                == exhaustive_system,
            )


if __name__ == "__main__":
    main()