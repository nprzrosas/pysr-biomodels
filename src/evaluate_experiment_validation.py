"""Select PySR candidates using fixed validation trajectories."""

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
    / "training_size_experiment"
)

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def evaluate_candidate_on_validation(
    *,
    pysr_loss_rank,
    candidate_row,
    validation_frame,
):
    """Evaluate one candidate on all fixed validation trajectories."""
    expression = candidate_row[
        "equation_original_units"
    ]

    rhs_function = compile_rhs(
        expression
    )

    metric_rows = []
    prediction_rows = []

    for trajectory_id, trajectory in (
        validation_frame.groupby(
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
                "pysr_loss_rank": pysr_loss_rank,
                "complexity": int(
                    candidate_row["complexity"]
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
                    "pysr_loss_rank": pysr_loss_rank,
                    "complexity": int(
                        candidate_row["complexity"]
                    ),
                    "trajectory_id": trajectory_id,
                    "time": time,
                    "VENUS_original": original,
                    "VENUS_inferred": inferred,
                    "error": inferred - original,
                }
            )

    return metric_rows, prediction_rows


def evaluate_training_dataset(
    *,
    dataset_name,
    validation_frame,
):
    """Evaluate and select candidates for one training-set size."""
    result_dir = (
        EXPERIMENT_ROOT
        / dataset_name
    )

    candidate_path = (
        result_dir
        / "top_8_by_loss.csv"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing PySR candidates: {candidate_path}"
        )

    candidates = pd.read_csv(
        candidate_path
    )

    required_columns = {
        "complexity",
        "loss",
        "equation_original_units",
    }

    missing = (
        required_columns
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            "Missing required candidate columns: "
            f"{sorted(missing)}"
        )

    all_metrics = []
    all_predictions = []

    for pysr_loss_rank, (_, candidate) in enumerate(
        candidates.iterrows(),
        start=1,
    ):
        (
            metrics,
            predictions,
        ) = evaluate_candidate_on_validation(
            pysr_loss_rank=pysr_loss_rank,
            candidate_row=candidate,
            validation_frame=validation_frame,
        )

        all_metrics.extend(
            metrics
        )

        all_predictions.extend(
            predictions
        )

    metrics_frame = pd.DataFrame(
        all_metrics
    )

    predictions_frame = pd.DataFrame(
        all_predictions
    )

    summary = (
        metrics_frame
        .groupby(
            [
                "pysr_loss_rank",
                "complexity",
                "equation_original_units",
            ],
            as_index=False,
        )
        .agg(
            mean_relative_rmse=(
                "relative_rmse",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
        )
    )

    summary = summary.sort_values(
        [
            "mean_rmse",
            "complexity",
            "pysr_loss_rank",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "validation_rank",
        range(
            1,
            len(summary) + 1,
        ),
    )

    selected = summary.iloc[
        [0]
    ].copy()

    summary.to_csv(
        result_dir
        / "validation_evaluation.csv",
        index=False,
    )

    metrics_frame.to_csv(
        result_dir
        / "validation_per_trajectory.csv",
        index=False,
    )

    predictions_frame.to_csv(
        result_dir
        / "validation_predictions.csv",
        index=False,
    )

    selected.to_csv(
        result_dir
        / "selected_by_validation.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print(
        f"Validation selection: {dataset_name}"
    )
    print("=" * 70)
    print()

    display_columns = [
        "validation_rank",
        "pysr_loss_rank",
        "complexity",
        "mean_relative_rmse",
        "mean_rmse",
        "mean_r2",
        "equation_original_units",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("SELECTED CANDIDATE")
    print()

    print(
        selected[
            display_columns
        ].to_string(
            index=False
        )
    )

    return selected


def main():
    """Select candidates using validation data only."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    validation_frame = datasets[
        "validation"
    ]

    for dataset_name in TRAINING_DATASETS:
        candidate_path = (
            EXPERIMENT_ROOT
            / dataset_name
            / "top_8_by_loss.csv"
        )

        if not candidate_path.exists():
            continue

        evaluate_training_dataset(
            dataset_name=dataset_name,
            validation_frame=validation_frame,
        )


if __name__ == "__main__":
    main()