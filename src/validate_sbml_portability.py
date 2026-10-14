"""Validate numerical portability of frozen inferred ODE systems in SBML.

Purpose
-------
Verify that an inferred ODE system exported to SBML produces numerically
equivalent trajectories in RoadRunner when compared with the same frozen
equations integrated through the existing solve_ivp validation
implementation.

This is a representation-portability test. It does not compare the inferred
system against the original biological SBML model.

For each frozen validation trajectory:

    frozen inferred equations
        -> existing SymPy/solve_ivp implementation
        -> exported SBML/RoadRunner implementation

The two resulting trajectories are compared point by point whenever both
integrators complete successfully.

Integration failures are scientific outputs of this validation and are
recorded rather than terminating the complete experiment.

This module does not:

- run PySR;
- perform candidate selection;
- refit equations;
- modify frozen selection outputs;
- use held-out test trajectories;
- compare against the original biological SBML model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner

from .evaluate_generic_validation import (
    compile_rhs_system,
    integrate_system,
)
from .experimental_design_generic import build_experimental_design
from .export_inferred_sbml import (
    TRAINING_DATASETS,
    inferred_sbml_path,
    load_model_spec,
    load_selected_system,
    selected_expressions,
)
from .simulate_generic import build_generic_development_dataset
from .simulation_metadata import load_simulation_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PORTABILITY_SUMMARY_FILENAME = "portability_summary.csv"

PORTABILITY_TRAJECTORY_METRICS_FILENAME = (
    "portability_trajectory_metrics.csv"
)

PORTABILITY_PREDICTIONS_FILENAME = (
    "portability_predictions.csv"
)

PORTABILITY_COMPARISON_FILENAME = (
    "portability_comparison.csv"
)

POINTS = 101

ABSOLUTE_TOLERANCE = 1.0e-6
RELATIVE_TOLERANCE = 1.0e-5

PASS_STATUS = "pass"
FAIL_STATUS = "fail"
NOT_EVALUABLE_STATUS = "not_evaluable"


@dataclass(frozen=True)
class PortabilityResult:
    """Paths and status for one training-condition portability validation."""

    model_id: str
    training_dataset: str
    status: str
    summary_path: Path
    trajectory_metrics_path: Path
    predictions_path: Path


def output_directory(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the SBML-validation directory for one training condition."""

    return inferred_sbml_path(
        model_id=model_id,
        training_dataset=training_dataset,
    ).parent


def summary_path(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the portability-summary path."""

    return (
        output_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / PORTABILITY_SUMMARY_FILENAME
    )


def trajectory_metrics_path(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the trajectory-level portability metrics path."""

    return (
        output_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / PORTABILITY_TRAJECTORY_METRICS_FILENAME
    )


def predictions_path(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the pointwise portability-prediction path."""

    return (
        output_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / PORTABILITY_PREDICTIONS_FILENAME
    )


def aggregate_output_directory(
    model_id: str,
) -> Path:
    """Return the model-level aggregate SBML-validation directory."""

    path = (
        PROJECT_ROOT
        / "results"
        / model_id
        / "generic_exact_experiment"
        / "coordinate_selection"
        / "sbml_validation"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def comparison_path(
    model_id: str,
) -> Path:
    """Return the aggregate portability-comparison path."""

    return (
        aggregate_output_directory(
            model_id=model_id,
        )
        / PORTABILITY_COMPARISON_FILENAME
    )


def validation_dataset(
    model_id: str,
):
    """Reconstruct frozen development data and return validation only."""

    spec = load_model_spec(
        model_id
    )

    duration = load_simulation_time(
        model_id=model_id,
    )

    design = build_experimental_design(
        spec=spec,
    )

    dataset = build_generic_development_dataset(
        spec=spec,
        design=design,
        duration=duration,
        points=POINTS,
    )

    if not dataset.test.empty:
        raise RuntimeError(
            "Development dataset unexpectedly contains held-out test data."
        )

    if dataset.validation.empty:
        raise RuntimeError(
            f"Validation dataset is empty for {model_id}."
        )

    return (
        spec,
        float(duration),
        dataset.validation.copy(),
    )


def validation_trajectory_ids(
    validation: pd.DataFrame,
) -> tuple[str, ...]:
    """Return validation trajectory IDs in frozen dataset order."""

    if "trajectory_id" not in validation.columns:
        raise ValueError(
            "Validation dataset is missing trajectory_id."
        )

    return tuple(
        validation[
            "trajectory_id"
        ]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )


def selected_equations(
    model_id: str,
    training_dataset: str,
) -> dict[str, str]:
    """Load frozen selected equations for one training condition."""

    spec = load_model_spec(
        model_id
    )

    selected_system = load_selected_system(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    return selected_expressions(
        spec=spec,
        selected_system=selected_system,
    )


def compile_selected_system(
    *,
    spec,
    expressions: dict[str, str],
):
    """Compile frozen selected equations using existing validation code."""

    return compile_rhs_system(
        spec=spec,
        expressions=expressions,
    )


def state_matrix(
    frame: pd.DataFrame,
    state_ids: tuple[str, ...],
) -> np.ndarray:
    """Extract a state matrix in canonical ModelSpec order."""

    missing = [
        state_id
        for state_id in state_ids
        if state_id not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Trajectory frame is missing state columns: "
            + ", ".join(
                missing
            )
        )

    return frame.loc[
        :,
        list(state_ids),
    ].to_numpy(
        dtype=float
    )


def simulate_solve_ivp(
    *,
    rhs_functions,
    time_values: np.ndarray,
    initial_values: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Integrate frozen equations using the existing solve_ivp protocol."""

    values, rhs_evaluations = integrate_system(
        rhs_functions=rhs_functions,
        time_values=time_values,
        initial_values=initial_values,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    expected_shape = (
        len(time_values),
        len(initial_values),
    )

    if values.shape != expected_shape:
        raise RuntimeError(
            "Unexpected solve_ivp trajectory shape. "
            f"Expected {expected_shape}, found {values.shape}."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "solve_ivp produced non-finite values."
        )

    if not isinstance(
        rhs_evaluations,
        (int, np.integer),
    ):
        raise RuntimeError(
            "Existing integrate_system returned an unexpected "
            "RHS-evaluation count type: "
            f"{type(rhs_evaluations).__name__}."
        )

    if rhs_evaluations < 0:
        raise RuntimeError(
            "Existing integrate_system returned a negative "
            "RHS-evaluation count."
        )

    return (
        values,
        int(rhs_evaluations),
    )


def configure_roadrunner_initial_conditions(
    *,
    rr: roadrunner.RoadRunner,
    state_ids: tuple[str, ...],
    initial_values: np.ndarray,
) -> None:
    """Set one validation initial condition in the inferred SBML model."""

    if len(state_ids) != len(initial_values):
        raise ValueError(
            "State and initial-value dimensions do not match."
        )

    rr.resetAll()

    for state_id, value in zip(
        state_ids,
        initial_values,
        strict=True,
    ):
        rr[f"init({state_id})"] = float(
            value
        )

    rr.reset()


def simulate_roadrunner(
    *,
    sbml_path: Path,
    state_ids: tuple[str, ...],
    time_values: np.ndarray,
    initial_values: np.ndarray,
) -> np.ndarray:
    """Integrate exported inferred SBML on the validation time grid."""

    if not sbml_path.exists():
        raise FileNotFoundError(
            f"Exported inferred SBML not found: {sbml_path}"
        )

    if len(time_values) < 2:
        raise ValueError(
            "At least two time points are required."
        )

    time_differences = np.diff(
        time_values
    )

    if not np.all(
        time_differences > 0.0
    ):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    expected_differences = np.full(
        len(time_differences),
        time_differences[0],
        dtype=float,
    )

    if not np.allclose(
        time_differences,
        expected_differences,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Current RoadRunner portability validator expects "
            "an evenly spaced validation time grid."
        )

    rr = roadrunner.RoadRunner(
        str(sbml_path)
    )

    observed_state_ids = tuple(
        rr.model.getFloatingSpeciesIds()
    )

    if set(observed_state_ids) != set(
        state_ids
    ):
        raise RuntimeError(
            "Exported SBML floating species do not match ModelSpec. "
            f"Expected={state_ids}, "
            f"observed={observed_state_ids}."
        )

    configure_roadrunner_initial_conditions(
        rr=rr,
        state_ids=state_ids,
        initial_values=initial_values,
    )

    selections = [
        "time",
        *state_ids,
    ]

    result = rr.simulate(
        float(
            time_values[0]
        ),
        float(
            time_values[-1]
        ),
        len(
            time_values
        ),
        selections=selections,
    )

    result_array = np.asarray(
        result,
        dtype=float,
    )

    expected_shape = (
        len(time_values),
        len(state_ids) + 1,
    )

    if result_array.shape != expected_shape:
        raise RuntimeError(
            "Unexpected RoadRunner trajectory shape. "
            f"Expected {expected_shape}, "
            f"found {result_array.shape}."
        )

    simulated_time = result_array[
        :,
        0,
    ]

    if not np.allclose(
        simulated_time,
        time_values,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "RoadRunner returned a time grid different from "
            "the requested validation grid."
        )

    values = result_array[
        :,
        1:,
    ]

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "RoadRunner produced non-finite values."
        )

    return values


def pointwise_error_metrics(
    solve_ivp_values: np.ndarray,
    roadrunner_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pointwise absolute and scale-aware relative errors."""

    absolute_error = np.abs(
        solve_ivp_values
        - roadrunner_values
    )

    scale = np.maximum(
        np.maximum(
            np.abs(
                solve_ivp_values
            ),
            np.abs(
                roadrunner_values
            ),
        ),
        ABSOLUTE_TOLERANCE,
    )

    relative_error = (
        absolute_error
        / scale
    )

    return (
        absolute_error,
        relative_error,
    )


def trajectory_passes(
    *,
    solve_ivp_values: np.ndarray,
    roadrunner_values: np.ndarray,
) -> bool:
    """Apply the predefined numerical-portability tolerance."""

    return bool(
        np.allclose(
            solve_ivp_values,
            roadrunner_values,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
    )


def error_message(
    error: Exception,
) -> str:
    """Return a compact reproducible exception description."""

    return (
        f"{type(error).__name__}: "
        f"{error}"
    )


def base_trajectory_row(
    *,
    model_id: str,
    training_dataset: str,
    trajectory_id: str,
    state_id: str,
    solve_ivp_status: str,
    roadrunner_status: str,
    equivalence_status: str,
    solve_ivp_rhs_evaluations,
    solve_ivp_error: str,
    roadrunner_error: str,
) -> dict:
    """Create common fields for one trajectory/state metrics row."""

    return {
        "model_id": model_id,
        "dataset_name": training_dataset,
        "trajectory_id": trajectory_id,
        "state_id": state_id,
        "solve_ivp_status": solve_ivp_status,
        "roadrunner_status": roadrunner_status,
        "equivalence_status": equivalence_status,
        "solve_ivp_rhs_evaluations": solve_ivp_rhs_evaluations,
        "solve_ivp_error": solve_ivp_error,
        "roadrunner_error": roadrunner_error,
    }


def evaluate_one_trajectory(
    *,
    model_id: str,
    training_dataset: str,
    trajectory_id: str,
    trajectory_frame: pd.DataFrame,
    state_ids: tuple[str, ...],
    rhs_functions,
    sbml_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Evaluate portability for one validation trajectory.

    Integration failures are recorded rather than propagated so that all
    validation trajectories and all training conditions can be evaluated.
    """

    ordered = (
        trajectory_frame
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )

    time_values = ordered[
        "time"
    ].to_numpy(
        dtype=float
    )

    observed_values = state_matrix(
        frame=ordered,
        state_ids=state_ids,
    )

    initial_values = observed_values[
        0,
        :
    ].copy()

    solve_ivp_values = None
    roadrunner_values = None

    solve_ivp_rhs_evaluations = np.nan

    solve_ivp_status = FAIL_STATUS
    roadrunner_status = FAIL_STATUS

    solve_ivp_error = ""
    roadrunner_error = ""

    try:
        (
            solve_ivp_values,
            solve_ivp_rhs_evaluations,
        ) = simulate_solve_ivp(
            rhs_functions=rhs_functions,
            time_values=time_values,
            initial_values=initial_values,
        )

        solve_ivp_status = PASS_STATUS

    except Exception as error:
        solve_ivp_error = error_message(
            error
        )

    try:
        roadrunner_values = simulate_roadrunner(
            sbml_path=sbml_path,
            state_ids=state_ids,
            time_values=time_values,
            initial_values=initial_values,
        )

        roadrunner_status = PASS_STATUS

    except Exception as error:
        roadrunner_error = error_message(
            error
        )

    both_integrated = (
        solve_ivp_status == PASS_STATUS
        and roadrunner_status == PASS_STATUS
    )

    metric_rows = []
    prediction_rows = []

    if not both_integrated:
        for state_id in state_ids:
            row = base_trajectory_row(
                model_id=model_id,
                training_dataset=training_dataset,
                trajectory_id=trajectory_id,
                state_id=state_id,
                solve_ivp_status=solve_ivp_status,
                roadrunner_status=roadrunner_status,
                equivalence_status=NOT_EVALUABLE_STATUS,
                solve_ivp_rhs_evaluations=solve_ivp_rhs_evaluations,
                solve_ivp_error=solve_ivp_error,
                roadrunner_error=roadrunner_error,
            )

            row.update(
                {
                    "max_absolute_error": np.nan,
                    "mean_absolute_error": np.nan,
                    "max_relative_error": np.nan,
                    "mean_relative_error": np.nan,
                    "rmse": np.nan,
                }
            )

            metric_rows.append(
                row
            )

        return (
            metric_rows,
            prediction_rows,
        )

    absolute_error, relative_error = pointwise_error_metrics(
        solve_ivp_values=solve_ivp_values,
        roadrunner_values=roadrunner_values,
    )

    passed = trajectory_passes(
        solve_ivp_values=solve_ivp_values,
        roadrunner_values=roadrunner_values,
    )

    equivalence_status = (
        PASS_STATUS
        if passed
        else FAIL_STATUS
    )

    for state_index, state_id in enumerate(
        state_ids
    ):
        state_absolute_error = absolute_error[
            :,
            state_index,
        ]

        state_relative_error = relative_error[
            :,
            state_index,
        ]

        state_difference = (
            solve_ivp_values[
                :,
                state_index,
            ]
            - roadrunner_values[
                :,
                state_index,
            ]
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        state_difference
                    )
                )
            )
        )

        row = base_trajectory_row(
            model_id=model_id,
            training_dataset=training_dataset,
            trajectory_id=trajectory_id,
            state_id=state_id,
            solve_ivp_status=solve_ivp_status,
            roadrunner_status=roadrunner_status,
            equivalence_status=equivalence_status,
            solve_ivp_rhs_evaluations=solve_ivp_rhs_evaluations,
            solve_ivp_error=solve_ivp_error,
            roadrunner_error=roadrunner_error,
        )

        row.update(
            {
                "max_absolute_error": float(
                    np.max(
                        state_absolute_error
                    )
                ),
                "mean_absolute_error": float(
                    np.mean(
                        state_absolute_error
                    )
                ),
                "max_relative_error": float(
                    np.max(
                        state_relative_error
                    )
                ),
                "mean_relative_error": float(
                    np.mean(
                        state_relative_error
                    )
                ),
                "rmse": rmse,
            }
        )

        metric_rows.append(
            row
        )

        for time_index, time_value in enumerate(
            time_values
        ):
            prediction_rows.append(
                {
                    "model_id": model_id,
                    "dataset_name": training_dataset,
                    "trajectory_id": trajectory_id,
                    "time": float(
                        time_value
                    ),
                    "state_id": state_id,
                    "initial_value": float(
                        initial_values[
                            state_index
                        ]
                    ),
                    "solve_ivp_value": float(
                        solve_ivp_values[
                            time_index,
                            state_index,
                        ]
                    ),
                    "roadrunner_value": float(
                        roadrunner_values[
                            time_index,
                            state_index,
                        ]
                    ),
                    "absolute_error": float(
                        absolute_error[
                            time_index,
                            state_index,
                        ]
                    ),
                    "relative_error": float(
                        relative_error[
                            time_index,
                            state_index,
                        ]
                    ),
                    "equivalence_status": equivalence_status,
                }
            )

    return (
        metric_rows,
        prediction_rows,
    )


def unique_trajectory_statuses(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one integration/equivalence status row per trajectory."""

    columns = [
        "trajectory_id",
        "solve_ivp_status",
        "roadrunner_status",
        "equivalence_status",
    ]

    statuses = (
        metrics[
            columns
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    duplicated = statuses[
        "trajectory_id"
    ].duplicated(
        keep=False
    )

    if duplicated.any():
        duplicate_ids = (
            statuses.loc[
                duplicated,
                "trajectory_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "A trajectory has inconsistent status rows: "
            + ", ".join(
                duplicate_ids
            )
        )

    return statuses


def finite_metric_max(
    series: pd.Series,
) -> float:
    """Return maximum finite metric or NaN when no finite values exist."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(finite) == 0:
        return float(
            "nan"
        )

    return float(
        np.max(
            finite
        )
    )


def finite_metric_mean(
    series: pd.Series,
) -> float:
    """Return mean finite metric or NaN when no finite values exist."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(finite) == 0:
        return float(
            "nan"
        )

    return float(
        np.mean(
            finite
        )
    )


def build_summary(
    *,
    model_id: str,
    training_dataset: str,
    metrics: pd.DataFrame,
    trajectory_ids: tuple[str, ...],
    sbml_path: Path,
) -> pd.DataFrame:
    """Build one-row portability summary for one selected system."""

    if metrics.empty:
        raise RuntimeError(
            "Cannot summarize empty portability metrics."
        )

    statuses = unique_trajectory_statuses(
        metrics
    )

    expected_trajectories = len(
        trajectory_ids
    )

    if len(statuses) != expected_trajectories:
        raise RuntimeError(
            "Portability metrics do not contain exactly one "
            "status per validation trajectory. "
            f"Expected {expected_trajectories}, "
            f"found {len(statuses)}."
        )

    solve_ivp_successes = int(
        (
            statuses[
                "solve_ivp_status"
            ]
            == PASS_STATUS
        ).sum()
    )

    solve_ivp_failures = int(
        (
            statuses[
                "solve_ivp_status"
            ]
            == FAIL_STATUS
        ).sum()
    )

    roadrunner_successes = int(
        (
            statuses[
                "roadrunner_status"
            ]
            == PASS_STATUS
        ).sum()
    )

    roadrunner_failures = int(
        (
            statuses[
                "roadrunner_status"
            ]
            == FAIL_STATUS
        ).sum()
    )

    comparable_trajectories = int(
        (
            statuses[
                "equivalence_status"
            ]
            != NOT_EVALUABLE_STATUS
        ).sum()
    )

    equivalent_trajectories = int(
        (
            statuses[
                "equivalence_status"
            ]
            == PASS_STATUS
        ).sum()
    )

    nonequivalent_trajectories = int(
        (
            statuses[
                "equivalence_status"
            ]
            == FAIL_STATUS
        ).sum()
    )

    not_evaluable_trajectories = int(
        (
            statuses[
                "equivalence_status"
            ]
            == NOT_EVALUABLE_STATUS
        ).sum()
    )

    all_integrated = (
        solve_ivp_successes == expected_trajectories
        and roadrunner_successes == expected_trajectories
    )

    all_equivalent = (
        comparable_trajectories == expected_trajectories
        and equivalent_trajectories == expected_trajectories
        and nonequivalent_trajectories == 0
        and not_evaluable_trajectories == 0
    )

    overall_pass = (
        all_integrated
        and all_equivalent
    )

    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "dataset_name": training_dataset,
                "validation_trajectories": expected_trajectories,
                "solve_ivp_successes": solve_ivp_successes,
                "solve_ivp_failures": solve_ivp_failures,
                "roadrunner_successes": roadrunner_successes,
                "roadrunner_failures": roadrunner_failures,
                "comparable_trajectories": comparable_trajectories,
                "equivalent_trajectories": equivalent_trajectories,
                "nonequivalent_trajectories": nonequivalent_trajectories,
                "not_evaluable_trajectories": not_evaluable_trajectories,
                "max_absolute_error": finite_metric_max(
                    metrics[
                        "max_absolute_error"
                    ]
                ),
                "mean_absolute_error": finite_metric_mean(
                    metrics[
                        "mean_absolute_error"
                    ]
                ),
                "max_relative_error": finite_metric_max(
                    metrics[
                        "max_relative_error"
                    ]
                ),
                "mean_relative_error": finite_metric_mean(
                    metrics[
                        "mean_relative_error"
                    ]
                ),
                "max_rmse": finite_metric_max(
                    metrics[
                        "rmse"
                    ]
                ),
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "all_solve_ivp_integrated": (
                    solve_ivp_successes
                    == expected_trajectories
                ),
                "all_roadrunner_integrated": (
                    roadrunner_successes
                    == expected_trajectories
                ),
                "all_comparable_trajectories_equivalent": (
                    comparable_trajectories > 0
                    and equivalent_trajectories
                    == comparable_trajectories
                    and nonequivalent_trajectories == 0
                ),
                "portability_status": (
                    PASS_STATUS
                    if overall_pass
                    else FAIL_STATUS
                ),
                "sbml_path": str(
                    sbml_path
                ),
            }
        ]
    )


def validate_training_dataset(
    *,
    model_id: str,
    training_dataset: str,
    spec,
    validation: pd.DataFrame,
) -> PortabilityResult:
    """Validate one frozen training-size selected system."""

    expressions = selected_equations(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    rhs_functions = compile_selected_system(
        spec=spec,
        expressions=expressions,
    )

    sbml_path = inferred_sbml_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    if not sbml_path.exists():
        raise FileNotFoundError(
            "Run src.export_inferred_sbml first. "
            f"Missing: {sbml_path}"
        )

    trajectory_ids = validation_trajectory_ids(
        validation
    )

    metric_rows = []
    prediction_rows = []

    for trajectory_id in trajectory_ids:
        trajectory_frame = validation.loc[
            validation[
                "trajectory_id"
            ].astype(str)
            == trajectory_id
        ].copy()

        current_metrics, current_predictions = (
            evaluate_one_trajectory(
                model_id=model_id,
                training_dataset=training_dataset,
                trajectory_id=trajectory_id,
                trajectory_frame=trajectory_frame,
                state_ids=tuple(
                    spec.state_ids
                ),
                rhs_functions=rhs_functions,
                sbml_path=sbml_path,
            )
        )

        metric_rows.extend(
            current_metrics
        )

        prediction_rows.extend(
            current_predictions
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    summary = build_summary(
        model_id=model_id,
        training_dataset=training_dataset,
        metrics=metrics,
        trajectory_ids=trajectory_ids,
        sbml_path=sbml_path,
    )

    destination = output_directory(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_summary_path = summary_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    current_metrics_path = trajectory_metrics_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    current_predictions_path = predictions_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    summary.to_csv(
        current_summary_path,
        index=False,
    )

    metrics.to_csv(
        current_metrics_path,
        index=False,
    )

    predictions.to_csv(
        current_predictions_path,
        index=False,
    )

    status = str(
        summary.iloc[
            0
        ][
            "portability_status"
        ]
    )

    return PortabilityResult(
        model_id=model_id,
        training_dataset=training_dataset,
        status=status,
        summary_path=current_summary_path,
        trajectory_metrics_path=current_metrics_path,
        predictions_path=current_predictions_path,
    )


def write_aggregate_comparison(
    *,
    model_id: str,
    results: tuple[PortabilityResult, ...],
) -> Path:
    """Combine training-condition summaries into one model-level CSV."""

    frames = []

    for result in results:
        frame = pd.read_csv(
            result.summary_path
        )

        frames.append(
            frame
        )

    comparison = pd.concat(
        frames,
        ignore_index=True,
    )

    training_order = {
        training_dataset: index
        for index, training_dataset in enumerate(
            TRAINING_DATASETS
        )
    }

    comparison[
        "_training_order"
    ] = comparison[
        "dataset_name"
    ].map(
        training_order
    )

    comparison = (
        comparison
        .sort_values(
            "_training_order"
        )
        .drop(
            columns=[
                "_training_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    path = comparison_path(
        model_id=model_id,
    )

    comparison.to_csv(
        path,
        index=False,
    )

    return path


def print_training_summary(
    *,
    training_dataset: str,
    summary: pd.Series,
) -> None:
    """Print compact human-readable portability results."""

    print(
        f"  portability status: "
        f"{summary['portability_status']}"
    )

    print(
        "  solve_ivp integration: "
        f"{int(summary['solve_ivp_successes'])}/"
        f"{int(summary['validation_trajectories'])}"
    )

    print(
        "  RoadRunner integration: "
        f"{int(summary['roadrunner_successes'])}/"
        f"{int(summary['validation_trajectories'])}"
    )

    print(
        "  comparable trajectories: "
        f"{int(summary['comparable_trajectories'])}"
    )

    print(
        "  equivalent trajectories: "
        f"{int(summary['equivalent_trajectories'])}"
    )

    print(
        "  nonequivalent trajectories: "
        f"{int(summary['nonequivalent_trajectories'])}"
    )

    print(
        "  not evaluable trajectories: "
        f"{int(summary['not_evaluable_trajectories'])}"
    )

    max_absolute_error = summary[
        "max_absolute_error"
    ]

    max_relative_error = summary[
        "max_relative_error"
    ]

    if pd.notna(
        max_absolute_error
    ):
        print(
            "  max absolute error: "
            f"{float(max_absolute_error):.6e}"
        )

    else:
        print(
            "  max absolute error: n/a"
        )

    if pd.notna(
        max_relative_error
    ):
        print(
            "  max relative error: "
            f"{float(max_relative_error):.6e}"
        )

    else:
        print(
            "  max relative error: n/a"
        )

    print(
        f"  summary: {summary_path(summary['model_id'], training_dataset)}"
    )


def validate_model(
    model_id: str,
) -> tuple[PortabilityResult, ...]:
    """Validate SBML portability for all four frozen training sizes."""

    spec, duration, validation = validation_dataset(
        model_id
    )

    print(
        f"Model: {model_id}"
    )

    print(
        f"States: {', '.join(spec.state_ids)}"
    )

    print(
        f"Simulation duration: {duration}"
    )

    print(
        "Validation trajectories: "
        f"{validation['trajectory_id'].nunique()}"
    )

    print(
        "Held-out test trajectories used: 0"
    )

    print(
        "Absolute tolerance: "
        f"{ABSOLUTE_TOLERANCE:.1e}"
    )

    print(
        "Relative tolerance: "
        f"{RELATIVE_TOLERANCE:.1e}"
    )

    print()

    results = []

    for training_dataset in TRAINING_DATASETS:
        print(
            "=" * 70
        )

        print(
            f"Validating {training_dataset}..."
        )

        result = validate_training_dataset(
            model_id=model_id,
            training_dataset=training_dataset,
            spec=spec,
            validation=validation,
        )

        results.append(
            result
        )

        summary = pd.read_csv(
            result.summary_path
        ).iloc[
            0
        ]

        print_training_summary(
            training_dataset=training_dataset,
            summary=summary,
        )

        print()

    result_tuple = tuple(
        results
    )

    aggregate_path = write_aggregate_comparison(
        model_id=model_id,
        results=result_tuple,
    )

    print(
        "=" * 70
    )

    print(
        "SBML portability validation complete."
    )

    print()

    for result in result_tuple:
        print(
            f"{result.training_dataset}: "
            f"{result.status}"
        )

    print()

    print(
        "Aggregate comparison:"
    )

    print(
        aggregate_path
    )

    print()

    print(
        "No held-out test trajectories were used."
    )

    print(
        "No PySR inference, candidate selection, or "
        "refitting was performed."
    )

    return result_tuple


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate numerical portability between frozen "
            "inferred equations integrated with solve_ivp and "
            "their exported SBML representation integrated "
            "with RoadRunner."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000079."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run SBML portability validation."""

    args = parse_arguments()

    validate_model(
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()