"""Plot trajectories for human-readable model recovery reports.

This module reconstructs frozen SBML training and validation trajectories
solely for visualization. Validation trajectories are reconstructed using the
same numerical integration functions as the frozen validation pipeline.

Held-out test figures are different: they are generated directly from the
already frozen test prediction, summary, and failure files. Test systems are
not reintegrated for reporting.

This module does not rerun PySR, perform model selection, calculate new
scientific metrics, modify frozen scientific results, or use held-out test
results for tuning.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluate_generic_validation import (
    compile_rhs_system,
    integrate_system,
)
from ..experimental_design_generic import (
    build_experimental_design,
)
from ..model_compatibility import (
    require_model_compatibility,
)
from ..model_spec import ModelSpec
from ..preflight import (
    run_preflight,
)
from ..simulate_generic import (
    build_generic_development_dataset,
    simulate_nominal_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "generic_exact_experiment"
REPORT_DIRECTORY = "model_recovery_report"
FIGURE_DIRECTORY = "figures"

SELECTION_DIRECTORY = "coordinate_selection"
SELECTED_SYSTEM_FILENAME = "coordinate_selected_system.csv"

TEST_DIRECTORY = "test"
TEST_SUMMARY_FILENAME = "test_summary.csv"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
TEST_FAILURES_FILENAME = "test_failures.csv"

POINTS_PER_TRAJECTORY = 101
EXPECTED_VALIDATION_TRAJECTORIES = 8
EXPECTED_TEST_TRAJECTORIES = 8

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

TRAINING_TRAJECTORIES = {
    "train_04": 4,
    "train_08": 8,
    "train_16": 16,
    "train_32": 32,
}



@dataclass(frozen=True)
class ValidationPrediction:
    """Visualization-only prediction for one frozen validation trajectory."""

    dataset_name: str
    trajectory_id: str
    time_values: np.ndarray
    observed_matrix: np.ndarray
    predicted_matrix: np.ndarray | None
    failure_message: str | None


def report_figure_directory(
    *,
    model_id: str,
) -> Path:
    """Return and create the figure directory for one model report."""
    directory = (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / REPORT_DIRECTORY
        / FIGURE_DIRECTORY
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def experiment_directory(
    *,
    model_id: str,
    dataset_name: str,
) -> Path:
    """Return one frozen training-size experiment directory."""
    return (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / dataset_name
    )


def selected_system_path(
    *,
    model_id: str,
    dataset_name: str,
) -> Path:
    """Return the frozen coordinate-selected system path."""
    return (
        experiment_directory(
            model_id=model_id,
            dataset_name=dataset_name,
        )
        / SELECTION_DIRECTORY
        / SELECTED_SYSTEM_FILENAME
    )


def test_directory(
    *,
    model_id: str,
    dataset_name: str,
) -> Path:
    """Return the frozen held-out test directory."""
    return (
        experiment_directory(
            model_id=model_id,
            dataset_name=dataset_name,
        )
        / SELECTION_DIRECTORY
        / TEST_DIRECTORY
    )


def validate_training_dataset(
    *,
    frame: pd.DataFrame,
    expected_trajectories: int,
    state_ids,
) -> None:
    """Validate a reconstructed training dataset before plotting."""
    required_columns = {
        "trajectory_id",
        "time",
        *state_ids,
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    observed_trajectories = frame[
        "trajectory_id"
    ].nunique()

    if observed_trajectories != expected_trajectories:
        raise ValueError(
            "Training dataset contains "
            f"{observed_trajectories} trajectories; expected "
            f"{expected_trajectories}."
        )

    if frame.empty:
        raise ValueError(
            "Training dataset is empty."
        )


def validate_nested_training_datasets(
    *,
    dataset,
) -> None:
    """Verify that the four reporting datasets preserve the frozen nesting."""
    previous_ids = None

    for dataset_name in TRAINING_DATASETS:
        frame = getattr(
            dataset,
            dataset_name,
        )

        expected_trajectories = (
            TRAINING_TRAJECTORIES[
                dataset_name
            ]
        )

        trajectory_ids = list(
            frame[
                "trajectory_id"
            ].drop_duplicates()
        )

        if len(trajectory_ids) != expected_trajectories:
            raise ValueError(
                f"{dataset_name} contains {len(trajectory_ids)} "
                f"trajectories; expected {expected_trajectories}."
            )

        if previous_ids is not None:
            current_prefix = trajectory_ids[
                :len(previous_ids)
            ]

            if current_prefix != previous_ids:
                raise ValueError(
                    "Training datasets do not preserve the expected "
                    "nested trajectory ordering."
                )

        previous_ids = trajectory_ids


def validate_validation_dataset(
    *,
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> None:
    """Validate the reconstructed frozen validation dataset."""
    required_columns = {
        "trajectory_id",
        "time",
        *spec.state_ids,
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Validation dataset is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    trajectory_count = frame[
        "trajectory_id"
    ].nunique()

    if trajectory_count != EXPECTED_VALIDATION_TRAJECTORIES:
        raise ValueError(
            "Frozen validation dataset should contain "
            f"{EXPECTED_VALIDATION_TRAJECTORIES} trajectories; "
            f"found {trajectory_count}."
        )

    expected_rows = (
        trajectory_count
        * POINTS_PER_TRAJECTORY
    )

    if len(frame) != expected_rows:
        raise ValueError(
            "Frozen validation dataset contains "
            f"{len(frame)} rows; expected {expected_rows}."
        )


def state_colors(
    *,
    state_ids,
):
    """Assign one consistent Matplotlib color to each dynamic state."""
    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ]

    return {
        state_id: color_cycle[
            index
            % len(color_cycle)
        ]
        for index, state_id in enumerate(
            state_ids
        )
    }


def load_coordinate_selected_system(
    *,
    model_id: str,
    dataset_name: str,
    spec: ModelSpec,
) -> pd.Series:
    """Load and validate one frozen Selector v1 system."""
    path = selected_system_path(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen selected system: {path}"
        )

    frame = pd.read_csv(
        path
    )

    if len(frame) != 1:
        raise ValueError(
            f"{path} contains {len(frame)} rows; expected exactly one."
        )

    row = frame.iloc[
        0
    ]

    if str(
        row[
            "model_id"
        ]
    ) != model_id:
        raise ValueError(
            f"Model ID mismatch in {path}."
        )

    if str(
        row[
            "dataset_name"
        ]
    ) != dataset_name:
        raise ValueError(
            f"Dataset mismatch in {path}."
        )

    if str(
        row[
            "selection_method"
        ]
    ) != "coordinate_search":
        raise ValueError(
            "Reporting expected a frozen coordinate-search selection; "
            f"found {row['selection_method']}."
        )

    for state_id in spec.state_ids:
        equation_column = (
            f"equation_{state_id}"
        )

        if equation_column not in row.index:
            raise ValueError(
                f"Selected system is missing {equation_column}."
            )

        equation = row[
            equation_column
        ]

        if pd.isna(
            equation
        ):
            raise ValueError(
                f"Selected equation for {state_id} is missing."
            )

    return row


def selected_expressions(
    *,
    selected_row: pd.Series,
    spec: ModelSpec,
) -> dict[str, str]:
    """Extract frozen selected equations in dynamic-state order."""
    return {
        state_id: str(
            selected_row[
                f"equation_{state_id}"
            ]
        )
        for state_id in spec.state_ids
    }


def reconstruct_validation_predictions(
    *,
    spec: ModelSpec,
    validation_frame: pd.DataFrame,
    dataset_name: str,
    selected_row: pd.Series,
) -> list[ValidationPrediction]:
    """Integrate one frozen selected system over validation trajectories."""
    expressions = selected_expressions(
        selected_row=selected_row,
        spec=spec,
    )

    rhs_functions = compile_rhs_system(
        spec=spec,
        expressions=expressions,
    )

    predictions = []

    for trajectory_id, trajectory in validation_frame.groupby(
        "trajectory_id",
        sort=False,
    ):
        trajectory = (
            trajectory
            .sort_values(
                "time"
            )
            .reset_index(
                drop=True
            )
        )

        time_values = trajectory[
            "time"
        ].to_numpy(
            dtype=float
        )

        observed_matrix = trajectory[
            list(
                spec.state_ids
            )
        ].to_numpy(
            dtype=float
        )

        initial_values = observed_matrix[
            0,
            :
        ]

        predicted_matrix = None
        failure_message = None

        try:
            (
                predicted_matrix,
                _rhs_evaluations,
            ) = integrate_system(
                rhs_functions=rhs_functions,
                time_values=time_values,
                initial_values=initial_values,
            )

        except Exception as error:
            failure_message = (
                f"{type(error).__name__}: {error}"
            )

        predictions.append(
            ValidationPrediction(
                dataset_name=dataset_name,
                trajectory_id=str(
                    trajectory_id
                ),
                time_values=time_values,
                observed_matrix=observed_matrix,
                predicted_matrix=predicted_matrix,
                failure_message=failure_message,
            )
        )

    return predictions


def validate_reconstructed_status(
    *,
    selected_row: pd.Series,
    predictions,
    dataset_name: str,
) -> None:
    """Check reconstructed integration status against frozen validation."""
    frozen_status = str(
        selected_row[
            "validation_status"
        ]
    ).strip().lower()

    frozen_successes = int(
        selected_row[
            "successful_trajectories"
        ]
    )

    frozen_failures = int(
        selected_row[
            "failed_trajectories"
        ]
    )

    reconstructed_successes = sum(
        prediction.predicted_matrix
        is not None
        for prediction in predictions
    )

    reconstructed_failures = sum(
        prediction.predicted_matrix
        is None
        for prediction in predictions
    )

    reconstructed_status = (
        "success"
        if reconstructed_failures == 0
        else "failed"
    )

    if (
        reconstructed_successes
        != frozen_successes
        or reconstructed_failures
        != frozen_failures
        or reconstructed_status
        != frozen_status
    ):
        raise RuntimeError(
            "Reporting reconstruction does not reproduce the frozen "
            f"validation status for {dataset_name}. "
            f"Frozen: status={frozen_status}, "
            f"successes={frozen_successes}, failures={frozen_failures}. "
            f"Reconstructed: status={reconstructed_status}, "
            f"successes={reconstructed_successes}, "
            f"failures={reconstructed_failures}."
        )


def load_test_summary(
    *,
    model_id: str,
    dataset_name: str,
) -> pd.Series:
    """Load and validate one frozen held-out test summary."""
    path = (
        test_directory(
            model_id=model_id,
            dataset_name=dataset_name,
        )
        / TEST_SUMMARY_FILENAME
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen test summary: {path}"
        )

    frame = pd.read_csv(
        path
    )

    if len(frame) != 1:
        raise ValueError(
            f"{path} contains {len(frame)} rows; expected exactly one."
        )

    row = frame.iloc[
        0
    ]

    if str(
        row[
            "model_id"
        ]
    ) != model_id:
        raise ValueError(
            f"Model ID mismatch in {path}."
        )

    if str(
        row[
            "dataset"
        ]
    ) != dataset_name:
        raise ValueError(
            f"Dataset mismatch in {path}."
        )

    if str(
        row[
            "selection_method"
        ]
    ) != "coordinate_search":
        raise ValueError(
            "Frozen test result does not use coordinate_search."
        )

    expected_trajectories = int(
        row[
            "expected_trajectories"
        ]
    )

    if expected_trajectories != EXPECTED_TEST_TRAJECTORIES:
        raise ValueError(
            "Frozen test summary expected "
            f"{expected_trajectories} trajectories; expected "
            f"{EXPECTED_TEST_TRAJECTORIES}."
        )

    successful = int(
        row[
            "successful_trajectories"
        ]
    )

    failed = int(
        row[
            "failed_trajectories"
        ]
    )

    if (
        successful
        + failed
        != expected_trajectories
    ):
        raise ValueError(
            "Frozen test success/failure counts do not sum to the "
            "expected trajectory count."
        )

    return row


def load_test_predictions(
    *,
    model_id: str,
    dataset_name: str,
    spec: ModelSpec,
    summary_row: pd.Series,
) -> pd.DataFrame:
    """Load and validate frozen held-out test predictions."""
    path = (
        test_directory(
            model_id=model_id,
            dataset_name=dataset_name,
        )
        / TEST_PREDICTIONS_FILENAME
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen test predictions: {path}"
        )
    required_columns = {
        "model_id",
        "dataset",
        "selection_method",
        "system_id",
        "trajectory_id",
        "state_id",
        "time",
        "observed",
        "predicted",
        "error",
    }

    try:
        frame = pd.read_csv(
            path
        )
    except pd.errors.EmptyDataError:
        successful = int(
            summary_row[
                "successful_trajectories"
            ]
        )

        if successful != 0:
            raise ValueError(
                "Frozen test predictions contain no CSV data despite "
                "successful test trajectories."
            )

        frame = pd.DataFrame(
            columns=sorted(
                required_columns
            )
        )
    

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Frozen test predictions are missing columns: "
            f"{sorted(missing_columns)}."
        )

    if frame.empty:
        successful = int(
            summary_row[
                "successful_trajectories"
            ]
        )

        if successful != 0:
            raise ValueError(
                "Test predictions are empty despite successful "
                "test trajectories."
            )

        return frame

    model_ids = set(
        frame[
            "model_id"
        ].astype(
            str
        )
    )

    if model_ids != {
        model_id
    }:
        raise ValueError(
            f"Unexpected model IDs in {path}: {sorted(model_ids)}."
        )

    datasets = set(
        frame[
            "dataset"
        ].astype(
            str
        )
    )

    if datasets != {
        dataset_name
    }:
        raise ValueError(
            f"Unexpected datasets in {path}: {sorted(datasets)}."
        )

    methods = set(
        frame[
            "selection_method"
        ].astype(
            str
        )
    )

    if methods != {
        "coordinate_search"
    }:
        raise ValueError(
            f"Unexpected selection methods in {path}: {sorted(methods)}."
        )

    system_ids = set(
        frame[
            "system_id"
        ].astype(
            int
        )
    )

    expected_system_id = int(
        summary_row[
            "system_id"
        ]
    )

    if system_ids != {
        expected_system_id
    }:
        raise ValueError(
            f"Unexpected system IDs in {path}: {sorted(system_ids)}."
        )

    observed_states = set(
        frame[
            "state_id"
        ].astype(
            str
        )
    )

    expected_states = set(
        spec.state_ids
    )

    if observed_states != expected_states:
        raise ValueError(
            "Frozen test predictions contain an unexpected state set. "
            f"Observed={sorted(observed_states)}, "
            f"expected={sorted(expected_states)}."
        )

    trajectory_ids = list(
        frame[
            "trajectory_id"
        ].drop_duplicates()
    )

    expected_successes = int(
        summary_row[
            "successful_trajectories"
        ]
    )

    if len(trajectory_ids) != expected_successes:
        raise ValueError(
            "Frozen test predictions contain "
            f"{len(trajectory_ids)} successful trajectories; "
            f"summary reports {expected_successes}."
        )

    for trajectory_id in trajectory_ids:
        trajectory = frame[
            frame[
                "trajectory_id"
            ]
            == trajectory_id
        ]

        for state_id in spec.state_ids:
            state_frame = trajectory[
                trajectory[
                    "state_id"
                ]
                == state_id
            ]

            if len(state_frame) != POINTS_PER_TRAJECTORY:
                raise ValueError(
                    f"{dataset_name}/{trajectory_id}/{state_id} "
                    f"contains {len(state_frame)} points; expected "
                    f"{POINTS_PER_TRAJECTORY}."
                )

    return frame


def load_test_failures(
    *,
    model_id: str,
    dataset_name: str,
    summary_row: pd.Series,
) -> pd.DataFrame:
    """Load and validate frozen held-out test failures."""
    directory = test_directory(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    path = (
        directory
        / TEST_FAILURES_FILENAME
    )

    expected_failures = int(
        summary_row[
            "failed_trajectories"
        ]
    )

    if not path.exists():
        if expected_failures == 0:
            return pd.DataFrame(
                columns=[
                    "trajectory_id",
                    "error_type",
                    "error_message",
                ]
            )

        raise FileNotFoundError(
            f"Missing frozen test failure file: {path}"
        )

    frame = pd.read_csv(
        path
    )

    required_columns = {
        "trajectory_id",
        "error_type",
        "error_message",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Frozen test failure file is missing columns: "
            f"{sorted(missing_columns)}."
        )

    if len(frame) != expected_failures:
        raise ValueError(
            f"{path} contains {len(frame)} failures; "
            f"summary reports {expected_failures}."
        )

    return frame


def validate_test_result_consistency(
    *,
    summary_row: pd.Series,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate consistency of frozen test summary and detailed outputs."""
    successful = int(
        summary_row[
            "successful_trajectories"
        ]
    )

    failed = int(
        summary_row[
            "failed_trajectories"
        ]
    )

    prediction_trajectories = (
        predictions[
            "trajectory_id"
        ].nunique()
        if not predictions.empty
        else 0
    )

    if prediction_trajectories != successful:
        raise ValueError(
            f"{dataset_name}: prediction trajectories "
            f"({prediction_trajectories}) do not match frozen "
            f"successful trajectories ({successful})."
        )

    if len(failures) != failed:
        raise ValueError(
            f"{dataset_name}: failure rows ({len(failures)}) "
            f"do not match frozen failed trajectories ({failed})."
        )

    status = str(
        summary_row[
            "test_status"
        ]
    ).strip().lower()

    expected_status = (
        "success"
        if failed == 0
        else "failed"
    )

    if status != expected_status:
        raise ValueError(
            f"{dataset_name}: frozen test status is {status}, "
            f"but success/failure counts imply {expected_status}."
        )

    test_nrmse = float(
        summary_row[
            "test_macro_nrmse"
        ]
    )

    if status == "failed":
        if not np.isinf(
            test_nrmse
        ):
            raise ValueError(
                f"{dataset_name}: failed frozen test system should "
                "have infinite test_macro_nrmse."
            )

    elif not np.isfinite(
        test_nrmse
    ):
        raise ValueError(
            f"{dataset_name}: successful frozen test system should "
            "have finite test_macro_nrmse."
        )


def plot_reference_sbml_trajectory(
    *,
    frame: pd.DataFrame,
    model_id: str,
    state_ids,
    output_directory: Path,
    colors,
) -> Path:
    """Plot the nominal trajectory generated by the original SBML model."""
    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    for state_id in state_ids:
        axis.plot(
            frame[
                "time"
            ],
            frame[
                state_id
            ],
            label=state_id,
            linewidth=2.0,
            color=colors[
                state_id
            ],
        )

    axis.set_xlabel(
        "Time"
    )

    axis.set_ylabel(
        "State value"
    )

    axis.set_title(
        f"{model_id}: nominal SBML trajectory"
    )

    axis.legend(
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "reference_sbml_trajectory.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def plot_state_training_panel(
    *,
    axis,
    frame: pd.DataFrame,
    state_id: str,
    dataset_name: str,
    color,
) -> None:
    """Plot all training trajectories for one state and training-set size."""
    trajectory_ids = list(
        frame[
            "trajectory_id"
        ].drop_duplicates()
    )

    for trajectory_id in trajectory_ids:
        trajectory = frame[
            frame[
                "trajectory_id"
            ]
            == trajectory_id
        ]

        axis.plot(
            trajectory[
                "time"
            ],
            trajectory[
                state_id
            ],
            linewidth=0.9,
            alpha=0.55,
            color=color,
        )

    axis.grid(
        alpha=0.2
    )

    axis.margins(
        x=0.0
    )

    expected_trajectories = (
        TRAINING_TRAJECTORIES[
            dataset_name
        ]
    )

    axis.set_title(
        f"{expected_trajectories} trajectories"
    )


def plot_training_trajectories(
    *,
    dataset,
    model_id: str,
    state_ids,
    output_directory: Path,
    colors,
) -> Path:
    """Plot nested training trajectories as states by training-set size."""
    validate_nested_training_datasets(
        dataset=dataset,
    )

    number_of_states = len(
        state_ids
    )

    number_of_datasets = len(
        TRAINING_DATASETS
    )

    figure_width = (
        3.6
        * number_of_datasets
    )

    figure_height = (
        2.6
        * number_of_states
    )

    figure, axes = plt.subplots(
        nrows=number_of_states,
        ncols=number_of_datasets,
        figsize=(
            figure_width,
            figure_height,
        ),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for column_index, dataset_name in enumerate(
        TRAINING_DATASETS
    ):
        frame = getattr(
            dataset,
            dataset_name,
        )

        validate_training_dataset(
            frame=frame,
            expected_trajectories=(
                TRAINING_TRAJECTORIES[
                    dataset_name
                ]
            ),
            state_ids=state_ids,
        )

        for row_index, state_id in enumerate(
            state_ids
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            plot_state_training_panel(
                axis=axis,
                frame=frame,
                state_id=state_id,
                dataset_name=dataset_name,
                color=colors[
                    state_id
                ],
            )

            if column_index == 0:
                axis.set_ylabel(
                    f"{state_id}\nState value"
                )

            if row_index == (
                number_of_states
                - 1
            ):
                axis.set_xlabel(
                    "Time"
                )

    figure.suptitle(
        (
            f"{model_id}: nested training trajectories\n"
            "Nested design: 4 ⊂ 8 ⊂ 16 ⊂ 32 "
            "training trajectories"
        ),
        fontsize=14,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    output_path = (
        output_directory
        / "training_trajectories.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def plot_validation_panel(
    *,
    axis,
    predictions,
    state_index: int,
    color,
) -> None:
    """Plot SBML truth and inferred validation trajectories."""
    for prediction in predictions:
        axis.plot(
            prediction.time_values,
            prediction.observed_matrix[
                :,
                state_index,
            ],
            linewidth=1.1,
            alpha=0.55,
            color=color,
            linestyle="-",
        )

        if prediction.predicted_matrix is not None:
            axis.plot(
                prediction.time_values,
                prediction.predicted_matrix[
                    :,
                    state_index,
                ],
                linewidth=1.1,
                alpha=0.8,
                color=color,
                linestyle="--",
            )

    failed_count = sum(
        prediction.predicted_matrix
        is None
        for prediction in predictions
    )

    if failed_count:
        axis.text(
            0.5,
            0.92,
            (
                f"FAILED: {failed_count}/"
                f"{EXPECTED_VALIDATION_TRAJECTORIES} trajectories"
            ),
            horizontalalignment="center",
            verticalalignment="top",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )

    axis.grid(
        alpha=0.2
    )

    axis.margins(
        x=0.0
    )


def plot_validation_truth_vs_inferred(
    *,
    spec: ModelSpec,
    validation_frame: pd.DataFrame,
    model_id: str,
    output_directory: Path,
    colors,
) -> Path:
    """Plot frozen validation truth against selected inferred ODE systems."""
    validate_validation_dataset(
        frame=validation_frame,
        spec=spec,
    )

    number_of_states = spec.n_states

    number_of_datasets = len(
        TRAINING_DATASETS
    )

    figure, axes = plt.subplots(
        nrows=number_of_states,
        ncols=number_of_datasets,
        figsize=(
            3.8 * number_of_datasets,
            2.7 * number_of_states,
        ),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for column_index, dataset_name in enumerate(
        TRAINING_DATASETS
    ):
        selected_row = load_coordinate_selected_system(
            model_id=model_id,
            dataset_name=dataset_name,
            spec=spec,
        )

        predictions = reconstruct_validation_predictions(
            spec=spec,
            validation_frame=validation_frame,
            dataset_name=dataset_name,
            selected_row=selected_row,
        )

        validate_reconstructed_status(
            selected_row=selected_row,
            predictions=predictions,
            dataset_name=dataset_name,
        )

        train_count = (
            TRAINING_TRAJECTORIES[
                dataset_name
            ]
        )

        validation_nrmse = float(
            selected_row[
                "validation_macro_nrmse"
            ]
        )

        validation_status = str(
            selected_row[
                "validation_status"
            ]
        )

        for row_index, state_id in enumerate(
            spec.state_ids
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            plot_validation_panel(
                axis=axis,
                predictions=predictions,
                state_index=row_index,
                color=colors[
                    state_id
                ],
            )

            if row_index == 0:
                if np.isfinite(
                    validation_nrmse
                ):
                    metric_text = (
                        f"NRMSE = {validation_nrmse:.4f}"
                    )

                else:
                    metric_text = "NRMSE = FAIL"

                axis.set_title(
                    (
                        f"{train_count} training trajectories\n"
                        f"{validation_status}, {metric_text}"
                    )
                )

            if column_index == 0:
                axis.set_ylabel(
                    f"{state_id}\nState value"
                )

            if row_index == (
                number_of_states
                - 1
            ):
                axis.set_xlabel(
                    "Time"
                )

    truth_handle = plt.Line2D(
        [],
        [],
        color="black",
        linewidth=1.8,
        linestyle="-",
        label="Original SBML",
    )

    inferred_handle = plt.Line2D(
        [],
        [],
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="Selected inferred ODE",
    )

    figure.suptitle(
        (
            f"{model_id}: validation — "
            "original SBML vs selected inferred ODE"
        ),
        fontsize=14,
        y=0.985,
    )

    figure.legend(
        handles=[
            truth_handle,
            inferred_handle,
        ],
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.955,
        ),
        ncol=2,
        frameon=False,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.90,
        )
    )

    output_path = (
        output_directory
        / "validation_truth_vs_inferred.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def plot_test_panel(
    *,
    axis,
    predictions: pd.DataFrame,
    state_id: str,
    color,
    failed_count: int,
) -> None:
    """Plot frozen successful held-out test trajectories for one state."""
    state_frame = predictions[
        predictions[
            "state_id"
        ]
        == state_id
    ]

    trajectory_ids = list(
        state_frame[
            "trajectory_id"
        ].drop_duplicates()
    )

    for trajectory_id in trajectory_ids:
        trajectory = (
            state_frame[
                state_frame[
                    "trajectory_id"
                ]
                == trajectory_id
            ]
            .sort_values(
                "time"
            )
        )

        axis.plot(
            trajectory[
                "time"
            ],
            trajectory[
                "observed"
            ],
            linewidth=1.1,
            alpha=0.55,
            color=color,
            linestyle="-",
        )

        axis.plot(
            trajectory[
                "time"
            ],
            trajectory[
                "predicted"
            ],
            linewidth=1.1,
            alpha=0.8,
            color=color,
            linestyle="--",
        )

    if failed_count:
        axis.text(
            0.5,
            0.92,
            (
                f"FAILED: {failed_count}/"
                f"{EXPECTED_TEST_TRAJECTORIES} trajectories"
            ),
            horizontalalignment="center",
            verticalalignment="top",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )

    axis.grid(
        alpha=0.2
    )

    axis.margins(
        x=0.0
    )


def plot_test_truth_vs_inferred(
    *,
    spec: ModelSpec,
    model_id: str,
    output_directory: Path,
    colors,
) -> Path:
    """Plot frozen held-out test truth against frozen inferred predictions."""
    number_of_states = spec.n_states

    number_of_datasets = len(
        TRAINING_DATASETS
    )

    figure, axes = plt.subplots(
        nrows=number_of_states,
        ncols=number_of_datasets,
        figsize=(
            3.8 * number_of_datasets,
            2.7 * number_of_states,
        ),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for column_index, dataset_name in enumerate(
        TRAINING_DATASETS
    ):
        summary_row = load_test_summary(
            model_id=model_id,
            dataset_name=dataset_name,
        )

        predictions = load_test_predictions(
            model_id=model_id,
            dataset_name=dataset_name,
            spec=spec,
            summary_row=summary_row,
        )

        failures = load_test_failures(
            model_id=model_id,
            dataset_name=dataset_name,
            summary_row=summary_row,
        )

        validate_test_result_consistency(
            summary_row=summary_row,
            predictions=predictions,
            failures=failures,
            dataset_name=dataset_name,
        )

        train_count = (
            TRAINING_TRAJECTORIES[
                dataset_name
            ]
        )

        test_status = str(
            summary_row[
                "test_status"
            ]
        ).strip().lower()

        test_nrmse = float(
            summary_row[
                "test_macro_nrmse"
            ]
        )

        successful_count = int(
            summary_row[
                "successful_trajectories"
            ]
        )

        failed_count = int(
            summary_row[
                "failed_trajectories"
            ]
        )

        for row_index, state_id in enumerate(
            spec.state_ids
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            plot_test_panel(
                axis=axis,
                predictions=predictions,
                state_id=state_id,
                color=colors[
                    state_id
                ],
                failed_count=failed_count,
            )

            if row_index == 0:
                if test_status == "success":
                    metric_text = (
                        f"test NRMSE = {test_nrmse:.4f}"
                    )

                else:
                    metric_text = "test NRMSE = FAIL"

                axis.set_title(
                    (
                        f"{train_count} training trajectories\n"
                        f"{test_status}, {metric_text}\n"
                        f"{successful_count}/"
                        f"{EXPECTED_TEST_TRAJECTORIES} completed"
                    )
                )

            if column_index == 0:
                axis.set_ylabel(
                    f"{state_id}\nState value"
                )

            if row_index == (
                number_of_states
                - 1
            ):
                axis.set_xlabel(
                    "Time"
                )

    truth_handle = plt.Line2D(
        [],
        [],
        color="black",
        linewidth=1.8,
        linestyle="-",
        label="Original SBML",
    )

    inferred_handle = plt.Line2D(
        [],
        [],
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="Selected inferred ODE",
    )

    figure.suptitle(
        (
            f"{model_id}: held-out test — "
            "original SBML vs selected inferred ODE"
        ),
        fontsize=14,
        y=0.985,
    )

    figure.legend(
        handles=[
            truth_handle,
            inferred_handle,
        ],
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.955,
        ),
        ncol=2,
        frameon=False,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.88,
        )
    )

    output_path = (
        output_directory
        / "test_truth_vs_inferred.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def generate_trajectory_figures(
    *,
    model_id: str,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
]:
    """Generate all trajectory figures for one model recovery report."""
    result = run_preflight(
        model_id=model_id,
    )

    require_model_compatibility(
        result.spec
    )

    if result.simulation_time is None:
        raise ValueError(
            "No curated simulation time is available for "
            f"{model_id}."
        )

    design = build_experimental_design(
        spec=result.spec,
    )

    nominal_frame = simulate_nominal_trajectory(
        spec=result.spec,
        duration=result.simulation_time,
        points=POINTS_PER_TRAJECTORY,
    )

    development_dataset = (
        build_generic_development_dataset(
            spec=result.spec,
            design=design,
            duration=result.simulation_time,
            points=POINTS_PER_TRAJECTORY,
        )
    )

    if not development_dataset.test.empty:
        raise ValueError(
            "Development reconstruction unexpectedly materialized "
            "held-out test data."
        )

    output_directory = report_figure_directory(
        model_id=model_id,
    )

    colors = state_colors(
        state_ids=result.spec.state_ids,
    )

    reference_path = (
        plot_reference_sbml_trajectory(
            frame=nominal_frame,
            model_id=model_id,
            state_ids=result.spec.state_ids,
            output_directory=output_directory,
            colors=colors,
        )
    )

    training_path = (
        plot_training_trajectories(
            dataset=development_dataset,
            model_id=model_id,
            state_ids=result.spec.state_ids,
            output_directory=output_directory,
            colors=colors,
        )
    )

    validation_path = (
        plot_validation_truth_vs_inferred(
            spec=result.spec,
            validation_frame=development_dataset.validation,
            model_id=model_id,
            output_directory=output_directory,
            colors=colors,
        )
    )

    test_path = (
        plot_test_truth_vs_inferred(
            spec=result.spec,
            model_id=model_id,
            output_directory=output_directory,
            colors=colors,
        )
    )

    return (
        reference_path,
        training_path,
        validation_path,
        test_path,
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate trajectory figures for a frozen "
            "model-recovery report."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels model identifier.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    (
        reference_path,
        training_path,
        validation_path,
        test_path,
    ) = generate_trajectory_figures(
        model_id=arguments.model_id,
    )

    print()
    print("=" * 70)
    print("MODEL RECOVERY TRAJECTORY FIGURES")
    print("=" * 70)

    print(
        "Model:",
        arguments.model_id,
    )

    print(
        "Reference SBML trajectory:",
        reference_path,
    )

    print(
        "Training trajectories:",
        training_path,
    )

    print(
        "Validation truth vs inferred:",
        validation_path,
    )

    print(
        "Held-out test truth vs inferred:",
        test_path,
    )

    print()
    print(
        "No PySR inference, model selection, scientific metric "
        "calculation, or held-out test reintegration was performed."
    )


if __name__ == "__main__":
    main()