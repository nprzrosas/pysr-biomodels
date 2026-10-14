"""Summarize the exact-derivative training-size experiment."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "BIOMD0000000414"
    / "training_size_experiment"
)

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def build_summary():
    """Combine validation and held-out test results."""
    rows = []

    for dataset_name in TRAINING_DATASETS:
        validation = pd.read_csv(
            EXPERIMENT_ROOT
            / dataset_name
            / "selected_by_validation.csv"
        ).iloc[0]

        test = pd.read_csv(
            EXPERIMENT_ROOT
            / dataset_name
            / "test_summary.csv"
        ).iloc[0]

        rows.append(
            {
                "training_dataset": dataset_name,
                "training_trajectories": int(
                    dataset_name.split("_")[1]
                ),
                "pysr_loss_rank": int(
                    validation["pysr_loss_rank"]
                ),
                "complexity": int(
                    validation["complexity"]
                ),
                "validation_mean_rmse": validation[
                    "mean_rmse"
                ],
                "test_mean_rmse": test[
                    "mean_rmse"
                ],
                "test_worst_rmse": test[
                    "worst_rmse"
                ],
                "test_mean_relative_rmse": test[
                    "mean_relative_rmse"
                ],
                "test_mean_r2": test[
                    "mean_r2"
                ],
                "equation_original_units": validation[
                    "equation_original_units"
                ],
            }
        )

    return pd.DataFrame(rows)


def main():
    """Write the frozen exact-derivative experiment summary."""
    summary = build_summary()

    output_path = (
        EXPERIMENT_ROOT
        / "exact_derivative_learning_curve.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print()
    print("EXACT-DERIVATIVE LEARNING CURVE")
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()