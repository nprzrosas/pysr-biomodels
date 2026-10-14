"""Evaluate coordinate-selected multistate systems on held-out test data."""

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_generic_validation import (
    calculate_state_metrics,
    compile_rhs_system,
    integrate_system,
)
from .model_spec import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "generic_exact_experiment"

SELECTION_METHOD = "coordinate_search"

SELECTION_DIRECTORY = "coordinate_selection"

SELECTED_SYSTEM_FILENAME = "coordinate_selected_system.csv"

TEST_DIRECTORY = "test"

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def load_selected_system(
    *,
    experiment_dir,
    spec: ModelSpec,
):
    """Load exactly one system selected by coordinate validation search."""
    selection_dir = (
        Path(experiment_dir)
        / SELECTION_DIRECTORY
    )

    selected_path = (
        selection_dir
        / SELECTED_SYSTEM_FILENAME
    )

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing coordinate-selected system: {selected_path}"
        )

    selected = pd.read_csv(
        selected_path
    )

    if len(selected) != 1:
        raise ValueError(
            f"{SELECTED_SYSTEM_FILENAME} must contain exactly one system."
        )

    row = selected.iloc[0]

    if "selection_method" not in selected.columns:
        raise ValueError(
            "Selected system is missing selection_method provenance."
        )

    selection_method = str(
        row["selection_method"]
    )

    if selection_method != SELECTION_METHOD:
        raise ValueError(
            "Unexpected selection method: "
            f"{selection_method}. "
            f"Expected {SELECTION_METHOD}."
        )

    if "model_id" not in selected.columns:
        raise ValueError(
            "Selected system is missing model_id provenance."
        )

    selected_model_id = str(
        row["model_id"]
    )

    if selected_model_id != spec.model_id:
        raise ValueError(
            "Selected system model_id does not match requested model: "
            f"{selected_model_id} != {spec.model_id}."
        )

    expressions = {}

    for state_id in spec.state_ids:
        column = f"equation_{state_id}"

        if column not in selected.columns:
            raise ValueError(
                f"Missing selected equation column: {column}"
            )

        expressions[state_id] = str(
            row[column]
        )

    return row, expressions


def load_state_scales(
    *,
    experiment_dir,
    spec: ModelSpec,
):
    """Load state scales derived exclusively from the training data."""
    metadata_path = (
        Path(experiment_dir)
        / "run_metadata.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing run metadata: {metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    if len(metadata) != 1:
        raise ValueError(
            "Expected exactly one metadata row."
        )

    row = metadata.iloc[0]

    state_scales = {}

    for state_id in spec.state_ids:
        column = f"state_scale_{state_id}"

        if column not in metadata.columns:
            raise ValueError(
                f"Missing state scale column: {column}"
            )

        scale = float(
            row[column]
        )

        if (
            not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError(
                f"Invalid state scale for {state_id}: {scale}"
            )

        state_scales[state_id] = scale

    return state_scales


def evaluate_selected_system_on_test(
    *,
    spec: ModelSpec,
    dataset_name,
    test_frame,
):
    """Evaluate one coordinate-selected system on held-out test trajectories."""
    experiment_dir = (
        PROJECT_ROOT
        / "results"
        / spec.model_id
        / EXPERIMENT_NAME
        / dataset_name
    )

    selection_dir = (
        experiment_dir
        / SELECTION_DIRECTORY
    )

    test_output_dir = (
        selection_dir
        / TEST_DIRECTORY
    )

    test_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_row, expressions = load_selected_system(
        experiment_dir=experiment_dir,
        spec=spec,
    )

    state_scales = load_state_scales(
        experiment_dir=experiment_dir,
        spec=spec,
    )

    rhs_functions = compile_rhs_system(
        spec=spec,
        expressions=expressions,
    )

    expected_trajectories = test_frame[
        "trajectory_id"
    ].nunique()

    metric_rows = []
    prediction_rows = []
    trajectory_rows = []

    failed_trajectories = []
    total_rhs_evaluations = 0

    for trajectory_id, trajectory in test_frame.groupby(
        "trajectory_id",
        sort=False,
    ):
        trajectory = (
            trajectory
            .sort_values("time")
            .reset_index(drop=True)
        )

        time_values = trajectory[
            "time"
        ].to_numpy(
            dtype=float
        )

        observed_matrix = trajectory[
            list(spec.state_ids)
        ].to_numpy(
            dtype=float
        )

        initial_values = observed_matrix[
            0,
            :
        ]

        try:
            predicted_matrix, rhs_evaluations = integrate_system(
                rhs_functions=rhs_functions,
                time_values=time_values,
                initial_values=initial_values,
            )

        except Exception as error:
            failed_trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

            continue

        total_rhs_evaluations += rhs_evaluations

        trajectory_nrmse_values = []

        for state_index, state_id in enumerate(
            spec.state_ids
        ):
            observed = observed_matrix[
                :,
                state_index,
            ]

            predicted = predicted_matrix[
                :,
                state_index,
            ]

            metrics = calculate_state_metrics(
                observed=observed,
                predicted=predicted,
                state_scale=state_scales[state_id],
            )

            trajectory_nrmse_values.append(
                metrics["normalized_rmse"]
            )

            metric_rows.append(
                {
                    "model_id": spec.model_id,
                    "dataset": dataset_name,
                    "selection_method": SELECTION_METHOD,
                    "system_id": int(
                        selected_row["system_id"]
                    ),
                    "trajectory_id": trajectory_id,
                    "state_id": state_id,
                    "state_scale": state_scales[
                        state_id
                    ],
                    **metrics,
                }
            )

            for (
                time_value,
                observed_value,
                predicted_value,
            ) in zip(
                time_values,
                observed,
                predicted,
            ):
                prediction_rows.append(
                    {
                        "model_id": spec.model_id,
                        "dataset": dataset_name,
                        "selection_method": SELECTION_METHOD,
                        "system_id": int(
                            selected_row["system_id"]
                        ),
                        "trajectory_id": trajectory_id,
                        "state_id": state_id,
                        "time": time_value,
                        "observed": observed_value,
                        "predicted": predicted_value,
                        "error": (
                            predicted_value
                            - observed_value
                        ),
                    }
                )

        trajectory_rows.append(
            {
                "model_id": spec.model_id,
                "dataset": dataset_name,
                "selection_method": SELECTION_METHOD,
                "system_id": int(
                    selected_row["system_id"]
                ),
                "trajectory_id": trajectory_id,
                "macro_nrmse": float(
                    np.mean(
                        trajectory_nrmse_values
                    )
                ),
                "rhs_evaluations": rhs_evaluations,
            }
        )

    successful_trajectories = len(
        trajectory_rows
    )

    failed_count = len(
        failed_trajectories
    )

    if (
        failed_count > 0
        or successful_trajectories
        != expected_trajectories
    ):
        test_macro_nrmse = np.inf
        test_status = "failed"

    else:
        test_macro_nrmse = float(
            np.mean(
                [
                    row["macro_nrmse"]
                    for row in trajectory_rows
                ]
            )
        )

        test_status = "success"

    metrics_frame = pd.DataFrame(
        metric_rows
    )

    predictions_frame = pd.DataFrame(
        prediction_rows
    )

    trajectories_frame = pd.DataFrame(
        trajectory_rows
    )

    failures_frame = pd.DataFrame(
        failed_trajectories
    )

    summary = {
        "model_id": spec.model_id,
        "dataset": dataset_name,
        "selection_method": SELECTION_METHOD,
        "system_id": int(
            selected_row["system_id"]
        ),
        "validation_macro_nrmse": float(
            selected_row[
                "validation_macro_nrmse"
            ]
        ),
        "test_macro_nrmse": test_macro_nrmse,
        "test_status": test_status,
        "expected_trajectories": (
            expected_trajectories
        ),
        "successful_trajectories": (
            successful_trajectories
        ),
        "failed_trajectories": (
            failed_count
        ),
        "total_rhs_evaluations": (
            total_rhs_evaluations
        ),
    }

    if "candidate_rank_signature" in selected_row.index:
        summary[
            "candidate_rank_signature"
        ] = str(
            selected_row[
                "candidate_rank_signature"
            ]
        )

    for state_id in spec.state_ids:
        summary[
            f"equation_{state_id}"
        ] = expressions[state_id]

        candidate_rank_column = (
            f"candidate_rank_{state_id}"
        )

        if candidate_rank_column in selected_row.index:
            summary[
                candidate_rank_column
            ] = int(
                selected_row[
                    candidate_rank_column
                ]
            )

        state_metrics = pd.DataFrame()

        if not metrics_frame.empty:
            state_metrics = metrics_frame[
                metrics_frame["state_id"]
                == state_id
            ]

        if not state_metrics.empty:
            summary[
                f"mean_rmse_{state_id}"
            ] = float(
                state_metrics["rmse"].mean()
            )

            summary[
                f"mean_normalized_rmse_{state_id}"
            ] = float(
                state_metrics[
                    "normalized_rmse"
                ].mean()
            )

            summary[
                f"mean_r2_{state_id}"
            ] = float(
                state_metrics["r2"].mean()
            )

        else:
            summary[
                f"mean_rmse_{state_id}"
            ] = np.inf

            summary[
                f"mean_normalized_rmse_{state_id}"
            ] = np.inf

            summary[
                f"mean_r2_{state_id}"
            ] = np.nan

    pd.DataFrame(
        [summary]
    ).to_csv(
        test_output_dir
        / "test_summary.csv",
        index=False,
    )

    metrics_frame.to_csv(
        test_output_dir
        / "test_state_metrics.csv",
        index=False,
    )

    predictions_frame.to_csv(
        test_output_dir
        / "test_predictions.csv",
        index=False,
    )

    trajectories_frame.to_csv(
        test_output_dir
        / "test_trajectory_metrics.csv",
        index=False,
    )

    if not failures_frame.empty:
        failures_frame.to_csv(
            test_output_dir
            / "test_failures.csv",
            index=False,
        )

    return summary


def evaluate_all_selected_systems(
    *,
    spec: ModelSpec,
    test_frame,
):
    """Evaluate every frozen coordinate-selected system on held-out test data."""
    summaries = []

    for dataset_name in TRAINING_DATASETS:
        print()
        print("=" * 70)
        print(
            f"TEST: {dataset_name}"
        )
        print("=" * 70)

        summary = evaluate_selected_system_on_test(
            spec=spec,
            dataset_name=dataset_name,
            test_frame=test_frame,
        )

        summaries.append(
            summary
        )

        print(
            "Selection method:",
            summary[
                "selection_method"
            ],
        )

        print(
            "Selected system:",
            summary["system_id"],
        )

        if "candidate_rank_signature" in summary:
            print(
                "Selected ranks:",
                summary[
                    "candidate_rank_signature"
                ],
            )

        print(
            "Validation macro NRMSE:",
            summary[
                "validation_macro_nrmse"
            ],
        )

        print(
            "Test macro NRMSE:",
            summary[
                "test_macro_nrmse"
            ],
        )

        print(
            "Status:",
            summary[
                "test_status"
            ],
        )

        print(
            "Successful trajectories:",
            summary[
                "successful_trajectories"
            ],
            "/",
            summary[
                "expected_trajectories"
            ],
        )

        for state_id in spec.state_ids:
            print(
                f"Mean normalized RMSE {state_id}:",
                summary[
                    f"mean_normalized_rmse_{state_id}"
                ],
            )

    summary_frame = pd.DataFrame(
        summaries
    )

    output_dir = (
        PROJECT_ROOT
        / "results"
        / spec.model_id
        / EXPERIMENT_NAME
        / SELECTION_DIRECTORY
        / TEST_DIRECTORY
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "test_comparison.csv"
    )

    summary_frame.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("HELD-OUT TEST SUMMARY")
    print("=" * 70)

    display_columns = [
        "dataset",
        "selection_method",
        "system_id",
        "validation_macro_nrmse",
        "test_macro_nrmse",
        "test_status",
    ]

    for state_id in spec.state_ids:
        display_columns.append(
            f"mean_normalized_rmse_{state_id}"
        )

    print(
        summary_frame[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved comparison:",
        output_path,
    )

    return summary_frame