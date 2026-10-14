"""Evaluate multi-seed noisy-training equations on held-out test data."""

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
    / "noisy_seed_experiment"
)

NOISE_SEEDS = (
    4040,
    4041,
    4042,
    4043,
    4044,
)

NOISE_LEVELS = (
    ("noise_0p5pct", 0.005),
    ("noise_1pct", 0.01),
    ("noise_2pct", 0.02),
    ("noise_5pct", 0.05),
)

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def evaluate_selected_candidate(
    *,
    noise_seed,
    noise_label,
    noise_fraction,
    dataset_name,
    test_frame,
):
    """Evaluate one validation-selected equation on held-out clean test data."""
    seed_label = f"seed_{noise_seed}"

    result_dir = (
        EXPERIMENT_ROOT
        / seed_label
        / noise_label
        / dataset_name
    )

    selected_path = (
        result_dir
        / "selected_by_validation.csv"
    )

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing validation-selected candidate: {selected_path}"
        )

    selected = pd.read_csv(
        selected_path
    )

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one selected candidate for "
            f"seed={noise_seed} / {noise_label} / {dataset_name}."
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

        try:
            predicted, nfev = integrate_candidate(
                rhs_function=rhs_function,
                time_values=time_values,
                initial_value=observed[0],
            )

        except RuntimeError as error:
            metric_rows.append(
                {
                    "noise_seed": noise_seed,
                    "noise_label": noise_label,
                    "noise_fraction": noise_fraction,
                    "training_dataset": dataset_name,
                    "pysr_loss_rank": int(
                        candidate["pysr_loss_rank"]
                    ),
                    "complexity": int(
                        candidate["complexity"]
                    ),
                    "trajectory_id": trajectory_id,
                    "equation_original_units": expression,
                    "integration_success": False,
                    "integration_error": str(error),
                    "nfev": None,
                    "mse": float("inf"),
                    "rmse": float("inf"),
                    "mae": float("inf"),
                    "relative_rmse": float("inf"),
                    "r2": float("-inf"),
                }
            )

            continue

        metrics = calculate_metrics(
            observed=observed,
            predicted=predicted,
        )

        metric_rows.append(
            {
                "noise_seed": noise_seed,
                "noise_label": noise_label,
                "noise_fraction": noise_fraction,
                "training_dataset": dataset_name,
                "pysr_loss_rank": int(
                    candidate["pysr_loss_rank"]
                ),
                "complexity": int(
                    candidate["complexity"]
                ),
                "trajectory_id": trajectory_id,
                "equation_original_units": expression,
                "integration_success": True,
                "integration_error": "",
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
                    "noise_seed": noise_seed,
                    "noise_label": noise_label,
                    "noise_fraction": noise_fraction,
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

    successful_trajectories = int(
        metrics_frame[
            "integration_success"
        ].sum()
    )

    failed_trajectories = int(
        (
            ~metrics_frame[
                "integration_success"
            ]
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "noise_seed": noise_seed,
                "noise_label": noise_label,
                "noise_fraction": noise_fraction,
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
                "successful_test_trajectories": (
                    successful_trajectories
                ),
                "failed_test_trajectories": (
                    failed_trajectories
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
    """Evaluate all frozen multi-seed equations on held-out test data."""
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

    for noise_seed in NOISE_SEEDS:
        for (
            noise_label,
            noise_fraction,
        ) in NOISE_LEVELS:
            for dataset_name in TRAINING_DATASETS:
                summary = evaluate_selected_candidate(
                    noise_seed=noise_seed,
                    noise_label=noise_label,
                    noise_fraction=noise_fraction,
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
    print("=" * 70)
    print("FINAL HELD-OUT MULTI-SEED NOISY TEST RESULTS")
    print("=" * 70)
    print()

    print(
        combined[
            [
                "noise_seed",
                "noise_label",
                "training_trajectories",
                "pysr_loss_rank",
                "complexity",
                "successful_test_trajectories",
                "failed_test_trajectories",
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
        "Conditions evaluated:",
        len(combined),
    )

    print(
        "Saved:",
        EXPERIMENT_ROOT
        / "test_learning_curve.csv",
    )


if __name__ == "__main__":
    main()