"""Select noisy-training PySR candidates across multiple noise seeds."""

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


def evaluate_candidate_on_validation(
    *,
    pysr_loss_rank,
    candidate_row,
    validation_frame,
):
    """Evaluate one candidate on all fixed clean validation trajectories."""
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

        try:
            predicted, nfev = integrate_candidate(
                rhs_function=rhs_function,
                time_values=time_values,
                initial_value=observed[0],
            )

        except RuntimeError as error:
            metric_rows.append(
                {
                    "pysr_loss_rank": pysr_loss_rank,
                    "complexity": int(
                        candidate_row["complexity"]
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
                "pysr_loss_rank": pysr_loss_rank,
                "complexity": int(
                    candidate_row["complexity"]
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
    noise_seed,
    noise_label,
    noise_fraction,
    dataset_name,
    validation_frame,
):
    """Select one candidate for one seed, noise, and training-size condition."""
    seed_label = f"seed_{noise_seed}"

    result_dir = (
        EXPERIMENT_ROOT
        / seed_label
        / noise_label
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

    summary.insert(
        0,
        "noise_fraction",
        noise_fraction,
    )

    summary.insert(
        0,
        "noise_label",
        noise_label,
    )

    summary.insert(
        0,
        "noise_seed",
        noise_seed,
    )

    summary.insert(
        0,
        "training_dataset",
        dataset_name,
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

    metrics_frame.insert(
        0,
        "noise_fraction",
        noise_fraction,
    )

    metrics_frame.insert(
        0,
        "noise_label",
        noise_label,
    )

    metrics_frame.insert(
        0,
        "noise_seed",
        noise_seed,
    )

    metrics_frame.insert(
        0,
        "training_dataset",
        dataset_name,
    )

    predictions_frame.insert(
        0,
        "noise_fraction",
        noise_fraction,
    )

    predictions_frame.insert(
        0,
        "noise_label",
        noise_label,
    )

    predictions_frame.insert(
        0,
        "noise_seed",
        noise_seed,
    )

    predictions_frame.insert(
        0,
        "training_dataset",
        dataset_name,
    )

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
        f"Validation: seed={noise_seed} / "
        f"{noise_label} / {dataset_name}"
    )
    print("=" * 70)

    display_columns = [
        "validation_rank",
        "pysr_loss_rank",
        "complexity",
        "mean_rmse",
        "mean_relative_rmse",
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
    print("SELECTED")
    print(
        selected[
            display_columns
        ].to_string(
            index=False
        )
    )

    return selected


def main():
    """Select candidates for all seeds, noise levels, and training sizes."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    validation_frame = datasets[
        "validation"
    ]

    selected_frames = []

    for noise_seed in NOISE_SEEDS:
        for (
            noise_label,
            noise_fraction,
        ) in NOISE_LEVELS:
            for dataset_name in TRAINING_DATASETS:
                selected = evaluate_training_dataset(
                    noise_seed=noise_seed,
                    noise_label=noise_label,
                    noise_fraction=noise_fraction,
                    dataset_name=dataset_name,
                    validation_frame=validation_frame,
                )

                selected_frames.append(
                    selected
                )

    combined = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    combined.to_csv(
        EXPERIMENT_ROOT
        / "validation_selected_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("ALL MULTI-SEED VALIDATION-SELECTED CANDIDATES")
    print("=" * 70)
    print()

    print(
        combined[
            [
                "noise_seed",
                "noise_label",
                "training_dataset",
                "pysr_loss_rank",
                "complexity",
                "mean_rmse",
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
        "Selected candidates:",
        len(combined),
    )

    print(
        "Saved:",
        EXPERIMENT_ROOT
        / "validation_selected_summary.csv",
    )


if __name__ == "__main__":
    main()