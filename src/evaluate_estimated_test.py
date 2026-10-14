"""Evaluate validation-selected equations on the held-out test set."""

from pathlib import Path

import pandas as pd

from .evaluate_trajectories import (
    calculate_metrics,
    compile_rhs,
    integrate_candidate,
)
from .simulate import (
    MODEL_ID,
    build_experimental_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "estimated_derivative_experiment"
)

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def evaluate_selected_candidate(
    *,
    dataset_name,
    test_frame,
):
    """Evaluate one validation-selected equation on held-out test data."""
    result_dir = (
        EXPERIMENT_ROOT
        / dataset_name
    )

    selected_path = (
        result_dir
        / "selected_by_validation.csv"
    )

    selected = pd.read_csv(
        selected_path
    )

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected candidate for {dataset_name}."
        )

    candidate = selected.iloc[0]

    expression = candidate[
        "equation_original_units"
    ]

    rhs_function = compile_rhs(
        expression
    )

    metric_rows = []
    prediction_rows = []

    for trajectory_id, trajectory in (
        test_frame.groupby(
            "trajectory_id",
            sort=False,
        )
    ):
        trajectory = trajectory.sort_values(
            "time"
        )

        time_values = trajectory[
            "time"
        ].to_numpy(
            dtype=float
        )

        observed = trajectory[
            "VENUS"
        ].to_numpy(
            dtype=float
        )

        predicted, nfev = integrate_candidate(
            rhs_function=rhs_function,
            time_values=time_values,
            initial_value=observed[0],
        )

        metrics = calculate_metrics(
            observed=observed,
            predicted=predicted,
        )

        metric_rows.append(
            {
                "training_dataset": dataset_name,
                "pysr_loss_rank": int(
                    candidate["pysr_loss_rank"]
                ),
                "complexity": int(
                    candidate["complexity"]
                ),
                "trajectory_id": trajectory_id,
                "equation_original_units": expression,
                "nfev": nfev,
                **metrics,
            }
        )

        for time, original, inferred in zip(
            time_values,
            observed,
            predicted,
        ):
            prediction_rows.append(
                {
                    "training_dataset": dataset_name,
                    "trajectory_id": trajectory_id,
                    "time": time,
                    "VENUS_original": original,
                    "VENUS_inferred": inferred,
                    "error": inferred - original,
                }
            )

    metrics_frame = pd.DataFrame(
        metric_rows
    )

    predictions_frame = pd.DataFrame(
        prediction_rows
    )

    summary = pd.DataFrame(
        [
            {
                "training_dataset": dataset_name,
                "training_trajectories": int(
                    dataset_name.split("_")[1]
                ),
                "pysr_loss_rank": int(
                    candidate["pysr_loss_rank"]
                ),
                "complexity": int(
                    candidate["complexity"]
                ),
                "mean_rmse": (
                    metrics_frame["rmse"].mean()
                ),
                "worst_rmse": (
                    metrics_frame["rmse"].max()
                ),
                "mean_relative_rmse": (
                    metrics_frame[
                        "relative_rmse"
                    ].mean()
                ),
                "mean_r2": (
                    metrics_frame["r2"].mean()
                ),
                "equation_original_units": expression,
            }
        ]
    )

    metrics_frame.to_csv(
        result_dir
        / "test_per_trajectory.csv",
        index=False,
    )

    predictions_frame.to_csv(
        result_dir
        / "test_predictions.csv",
        index=False,
    )

    summary.to_csv(
        result_dir
        / "test_summary.csv",
        index=False,
    )

    return summary


def main():
    """Evaluate all frozen validation-selected equations on test data."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    test_frame = datasets[
        "test"
    ]

    summaries = []

    for dataset_name in TRAINING_DATASETS:
        summary = evaluate_selected_candidate(
            dataset_name=dataset_name,
            test_frame=test_frame,
        )

        summaries.append(
            summary
        )

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    combined.to_csv(
        EXPERIMENT_ROOT
        / "test_learning_curve.csv",
        index=False,
    )

    print()
    print("FINAL HELD-OUT TEST RESULTS")
    print()

    print(
        combined[
            [
                "training_trajectories",
                "pysr_loss_rank",
                "complexity",
                "mean_rmse",
                "worst_rmse",
                "mean_relative_rmse",
                "mean_r2",
                "equation_original_units",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        EXPERIMENT_ROOT
        / "test_learning_curve.csv",
    )


if __name__ == "__main__":
    main()