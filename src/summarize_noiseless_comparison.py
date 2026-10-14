"""Summarize exact and estimated derivative held-out test experiments."""

from pathlib import Path

import pandas as pd

from .simulate import MODEL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
)

EXACT_PATH = (
    RESULT_ROOT
    / "training_size_experiment"
    / "test_learning_curve.csv"
)

ESTIMATED_PATH = (
    RESULT_ROOT
    / "estimated_derivative_experiment"
    / "test_learning_curve.csv"
)

OUTPUT_PATH = (
    RESULT_ROOT
    / "exact_vs_estimated_noiseless.csv"
)


def load_results(
    path,
    derivative_method,
):
    """Load one held-out test learning curve."""
    frame = pd.read_csv(
        path
    )

    frame = frame.copy()

    frame.insert(
        0,
        "derivative_method",
        derivative_method,
    )

    return frame


def main():
    """Combine exact and estimated derivative test results."""
    exact = load_results(
        EXACT_PATH,
        "exact",
    )

    estimated = load_results(
        ESTIMATED_PATH,
        "savitzky_golay",
    )

    combined = pd.concat(
        [
            exact,
            estimated,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "training_trajectories",
            "derivative_method",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "derivative_method",
        "training_trajectories",
        "pysr_loss_rank",
        "complexity",
        "mean_rmse",
        "worst_rmse",
        "mean_relative_rmse",
        "mean_r2",
    ]

    print()
    print(
        "EXACT VS ESTIMATED DERIVATIVES"
    )
    print(
        "NOISELESS HELD-OUT TEST"
    )
    print()

    print(
        combined[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()