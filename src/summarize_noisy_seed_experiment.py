"""Summarize multi-seed noisy-derivative held-out test results."""

from pathlib import Path

import pandas as pd

from .simulate import MODEL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "noisy_seed_experiment"
)

TEST_PATH = (
    EXPERIMENT_ROOT
    / "test_learning_curve.csv"
)

OUTPUT_PATH = (
    EXPERIMENT_ROOT
    / "test_summary_by_noise_and_training_size.csv"
)


def load_test_results():
    """Load the frozen multi-seed held-out test results."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing held-out test results: {TEST_PATH}"
        )

    frame = pd.read_csv(
        TEST_PATH
    )

    required_columns = {
        "noise_seed",
        "noise_label",
        "noise_fraction",
        "training_trajectories",
        "pysr_loss_rank",
        "complexity",
        "successful_test_trajectories",
        "failed_test_trajectories",
        "mean_rmse",
        "worst_rmse",
        "mean_relative_rmse",
        "mean_r2",
    }

    missing = required_columns - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing required test columns: "
            f"{sorted(missing)}"
        )

    return frame


def summarize_results(
    frame,
):
    """Aggregate held-out test performance across noise seeds."""
    summary = (
        frame.groupby(
            [
                "noise_label",
                "noise_fraction",
                "training_trajectories",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=(
                "noise_seed",
                "nunique",
            ),
            mean_test_rmse=(
                "mean_rmse",
                "mean",
            ),
            median_test_rmse=(
                "mean_rmse",
                "median",
            ),
            sd_test_rmse=(
                "mean_rmse",
                "std",
            ),
            min_test_rmse=(
                "mean_rmse",
                "min",
            ),
            max_test_rmse=(
                "mean_rmse",
                "max",
            ),
            mean_worst_rmse=(
                "worst_rmse",
                "mean",
            ),
            mean_relative_rmse=(
                "mean_relative_rmse",
                "mean",
            ),
            median_relative_rmse=(
                "mean_relative_rmse",
                "median",
            ),
            mean_r2=(
                "mean_r2",
                "mean",
            ),
            mean_complexity=(
                "complexity",
                "mean",
            ),
            median_complexity=(
                "complexity",
                "median",
            ),
            mean_loss_rank=(
                "pysr_loss_rank",
                "mean",
            ),
            successful_test_trajectories=(
                "successful_test_trajectories",
                "sum",
            ),
            failed_test_trajectories=(
                "failed_test_trajectories",
                "sum",
            ),
        )
    )

    summary = summary.sort_values(
        [
            "noise_fraction",
            "training_trajectories",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return summary


def main():
    """Create the final descriptive summary for the multi-seed experiment."""
    frame = load_test_results()

    summary = summarize_results(
        frame
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "noise_label",
        "training_trajectories",
        "n_seeds",
        "mean_test_rmse",
        "median_test_rmse",
        "sd_test_rmse",
        "min_test_rmse",
        "max_test_rmse",
        "mean_complexity",
        "median_complexity",
        "mean_loss_rank",
        "failed_test_trajectories",
    ]

    print()
    print(
        "MULTI-SEED NOISY-DERIVATIVE EXPERIMENT"
    )
    print(
        "HELD-OUT TEST SUMMARY"
    )
    print()

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Input conditions:",
        len(frame),
    )

    print(
        "Summary conditions:",
        len(summary),
    )

    print(
        "Unique noise seeds:",
        frame["noise_seed"].nunique(),
    )

    print(
        "Total successful test trajectories:",
        summary[
            "successful_test_trajectories"
        ].sum(),
    )

    print(
        "Total failed test trajectories:",
        summary[
            "failed_test_trajectories"
        ].sum(),
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()