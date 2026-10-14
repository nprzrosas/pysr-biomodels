"""Compare original and inferred SBML models dynamically.

Purpose
-------
Compare the frozen original biological SBML model against each frozen
PySR-inferred SBML model using RoadRunner for both systems.

For each validation initial condition:

    original SBML -> RoadRunner -> reference trajectory
    inferred SBML -> RoadRunner -> inferred trajectory

Both models are simulated:

- from the same initial conditions;
- on the same time interval;
- at the same output times;
- using RoadRunner.

The resulting trajectories are compared using RMSE, normalized RMSE,
R², and absolute-error diagnostics.

This is a dynamic-fidelity evaluation. It is distinct from:

- symbolic/mechanistic recovery;
- exact algebraic recovery;
- SBML representation portability;
- PySR training or model selection.

This module does not:

- run PySR;
- perform candidate selection;
- refit equations;
- modify frozen selection outputs;
- use held-out test trajectories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner

from .experimental_design_generic import build_experimental_design
from .export_inferred_sbml import (
    TRAINING_DATASETS,
    inferred_sbml_path,
    load_model_spec,
)
from .model_registry import resolve_model_path
from .simulate_generic import build_generic_development_dataset
from .simulation_metadata import load_simulation_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "generic_exact_experiment"

SELECTION_DIRECTORY = "coordinate_selection"

SBML_VALIDATION_DIRECTORY = "sbml_validation"

DYNAMIC_FIDELITY_DIRECTORY = "dynamic_fidelity"

SUMMARY_FILENAME = "dynamic_fidelity_summary.csv"

TRAJECTORY_METRICS_FILENAME = (
    "dynamic_fidelity_trajectory_metrics.csv"
)

PREDICTIONS_FILENAME = (
    "dynamic_fidelity_predictions.csv"
)

COMPARISON_FILENAME = (
    "dynamic_fidelity_comparison.csv"
)

POINTS = 101

PASS_STATUS = "pass"
FAIL_STATUS = "fail"

SUCCESS_STATUS = "success"
NOT_EVALUABLE_STATUS = "not_evaluable"


@dataclass(frozen=True)
class DynamicFidelityResult:
    """Outputs for one inferred training condition."""

    model_id: str
    training_dataset: str
    status: str
    summary_path: Path
    trajectory_metrics_path: Path
    predictions_path: Path


def original_sbml_path(
    model_id: str,
) -> Path:
    """Resolve the canonical original SBML through the model registry."""

    path = resolve_model_path(
        model_id=model_id,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Original SBML not found: {path}"
        )

    return path


def output_directory(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return output directory for one training condition."""

    path = (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / training_dataset
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
        / DYNAMIC_FIDELITY_DIRECTORY
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def aggregate_output_directory(
    model_id: str,
) -> Path:
    """Return model-level aggregate output directory."""

    path = (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
        / DYNAMIC_FIDELITY_DIRECTORY
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


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
        dataset,
    )


def validate_state_ids(
    *,
    rr: roadrunner.RoadRunner,
    expected_state_ids: tuple[str, ...],
    model_label: str,
) -> None:
    """Check that RoadRunner exposes the expected dynamic states."""

    observed_state_ids = tuple(
        rr.model.getFloatingSpeciesIds()
    )

    if set(observed_state_ids) != set(
        expected_state_ids
    ):
        raise RuntimeError(
            f"{model_label} floating species do not match ModelSpec. "
            f"Expected={expected_state_ids}, "
            f"observed={observed_state_ids}."
        )


def configure_initial_conditions(
    *,
    rr: roadrunner.RoadRunner,
    state_ids: tuple[str, ...],
    initial_values: np.ndarray,
) -> None:
    """Set identical floating-species initial conditions."""

    if len(state_ids) != len(
        initial_values
    ):
        raise ValueError(
            "State and initial-value dimensions do not match."
        )

    rr.resetAll()

    for state_id, value in zip(
        state_ids,
        initial_values,
        strict=True,
    ):
        rr[
            f"init({state_id})"
        ] = float(
            value
        )

    rr.reset()


def simulate_sbml(
    *,
    sbml_path: Path,
    state_ids: tuple[str, ...],
    time_values: np.ndarray,
    initial_values: np.ndarray,
    model_label: str,
) -> np.ndarray:
    """Simulate one SBML model with RoadRunner."""

    if not sbml_path.exists():
        raise FileNotFoundError(
            f"{model_label} SBML not found: {sbml_path}"
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
            "Dynamic-fidelity comparison expects an evenly "
            "spaced time grid."
        )

    rr = roadrunner.RoadRunner(
        str(sbml_path)
    )

    validate_state_ids(
        rr=rr,
        expected_state_ids=state_ids,
        model_label=model_label,
    )

    configure_initial_conditions(
        rr=rr,
        state_ids=state_ids,
        initial_values=initial_values,
    )

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
        selections=[
            "time",
            *state_ids,
        ],
    )

    values = np.asarray(
        result,
        dtype=float,
    )

    expected_shape = (
        len(time_values),
        len(state_ids) + 1,
    )

    if values.shape != expected_shape:
        raise RuntimeError(
            f"{model_label} returned unexpected trajectory shape. "
            f"Expected {expected_shape}, "
            f"found {values.shape}."
        )

    simulated_time = values[
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
            f"{model_label} returned a different time grid."
        )

    state_values = values[
        :,
        1:,
    ]

    if not np.all(
        np.isfinite(
            state_values
        )
    ):
        raise RuntimeError(
            f"{model_label} produced non-finite values."
        )

    return state_values


def training_frame(
    *,
    dataset,
    training_dataset: str,
) -> pd.DataFrame:
    """Return one frozen nested training dataset."""

    frames = {
        "train_04": dataset.train_04,
        "train_08": dataset.train_08,
        "train_16": dataset.train_16,
        "train_32": dataset.train_32,
    }

    if training_dataset not in frames:
        raise ValueError(
            f"Unknown training dataset: {training_dataset}"
        )

    frame = frames[
        training_dataset
    ]

    if frame.empty:
        raise RuntimeError(
            f"Training dataset is empty: {training_dataset}"
        )

    return frame
    

def training_state_scales(
    *,
    spec,
    dataset,
    training_dataset: str,
) -> dict[str, float]:
    """Return frozen training-data scales used for NRMSE.

    Each state is normalized by max(abs(x)) over the corresponding
    frozen training dataset. Validation outcomes are therefore not
    used to define the normalization scale.
    """

    training = training_frame(
        dataset=dataset,
        training_dataset=training_dataset,
    )

    scales = {}

    for state_id in spec.state_ids:
        values = training[
            state_id
        ].to_numpy(
            dtype=float
        )

        scale = float(
            np.max(
                np.abs(
                    values
                )
            )
        )

        if not np.isfinite(
            scale
        ):
            raise RuntimeError(
                f"Non-finite training scale for {state_id}."
            )

        if scale <= 0.0:
            raise RuntimeError(
                f"Non-positive training scale for {state_id}."
            )

        scales[
            state_id
        ] = scale

    return scales


def r_squared(
    *,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate R² relative to the original SBML trajectory."""

    residual_sum_squares = float(
        np.sum(
            np.square(
                reference
                - prediction
            )
        )
    )

    centered = (
        reference
        - np.mean(
            reference
        )
    )

    total_sum_squares = float(
        np.sum(
            np.square(
                centered
            )
        )
    )

    if total_sum_squares <= 0.0:
        if residual_sum_squares <= 0.0:
            return 1.0

        return float(
            "nan"
        )

    return float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def error_message(
    error: Exception,
) -> str:
    """Return compact exception information."""

    return (
        f"{type(error).__name__}: "
        f"{error}"
    )


def evaluate_trajectory(
    *,
    model_id: str,
    training_dataset: str,
    trajectory_id: str,
    trajectory: pd.DataFrame,
    state_ids: tuple[str, ...],
    state_scales: dict[str, float],
    original_path: Path,
    inferred_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Compare original and inferred SBML for one validation IC."""

    ordered = (
        trajectory
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

    initial_values = ordered.loc[
        0,
        list(
            state_ids
        ),
    ].to_numpy(
        dtype=float
    )

    original_values = None
    inferred_values = None

    original_status = FAIL_STATUS
    inferred_status = FAIL_STATUS

    original_error = ""
    inferred_error = ""

    try:
        original_values = simulate_sbml(
            sbml_path=original_path,
            state_ids=state_ids,
            time_values=time_values,
            initial_values=initial_values,
            model_label="original",
        )

        original_status = PASS_STATUS

    except Exception as error:
        original_error = error_message(
            error
        )

    try:
        inferred_values = simulate_sbml(
            sbml_path=inferred_path,
            state_ids=state_ids,
            time_values=time_values,
            initial_values=initial_values,
            model_label="inferred",
        )

        inferred_status = PASS_STATUS

    except Exception as error:
        inferred_error = error_message(
            error
        )

    comparable = (
        original_status == PASS_STATUS
        and inferred_status == PASS_STATUS
    )

    metric_rows = []
    prediction_rows = []

    if not comparable:
        for state_id in state_ids:
            metric_rows.append(
                {
                    "model_id": model_id,
                    "dataset_name": training_dataset,
                    "trajectory_id": trajectory_id,
                    "state_id": state_id,
                    "original_status": original_status,
                    "inferred_status": inferred_status,
                    "comparison_status": NOT_EVALUABLE_STATUS,
                    "rmse": np.nan,
                    "normalized_rmse": np.nan,
                    "r2": np.nan,
                    "max_absolute_error": np.nan,
                    "mean_absolute_error": np.nan,
                    "original_error": original_error,
                    "inferred_error": inferred_error,
                }
            )

        return (
            metric_rows,
            prediction_rows,
        )

    for state_index, state_id in enumerate(
        state_ids
    ):
        original_state = original_values[
            :,
            state_index,
        ]

        inferred_state = inferred_values[
            :,
            state_index,
        ]

        difference = (
            original_state
            - inferred_state
        )

        absolute_error = np.abs(
            difference
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        difference
                    )
                )
            )
        )

        normalized_rmse = float(
            rmse
            / state_scales[
                state_id
            ]
        )

        r2 = r_squared(
            reference=original_state,
            prediction=inferred_state,
        )

        metric_rows.append(
            {
                "model_id": model_id,
                "dataset_name": training_dataset,
                "trajectory_id": trajectory_id,
                "state_id": state_id,
                "original_status": original_status,
                "inferred_status": inferred_status,
                "comparison_status": SUCCESS_STATUS,
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "r2": r2,
                "max_absolute_error": float(
                    np.max(
                        absolute_error
                    )
                ),
                "mean_absolute_error": float(
                    np.mean(
                        absolute_error
                    )
                ),
                "original_error": "",
                "inferred_error": "",
            }
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
                    "original_value": float(
                        original_state[
                            time_index
                        ]
                    ),
                    "inferred_value": float(
                        inferred_state[
                            time_index
                        ]
                    ),
                    "absolute_error": float(
                        absolute_error[
                            time_index
                        ]
                    ),
                }
            )

    return (
        metric_rows,
        prediction_rows,
    )


def trajectory_status_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one status row per validation trajectory."""

    columns = [
        "trajectory_id",
        "original_status",
        "inferred_status",
        "comparison_status",
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

    if statuses[
        "trajectory_id"
    ].duplicated().any():
        raise RuntimeError(
            "Inconsistent trajectory status rows."
        )

    return statuses


def finite_mean(
    values: pd.Series,
) -> float:
    """Return mean of finite numeric values."""

    array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = array[
        np.isfinite(
            array
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


def finite_max(
    values: pd.Series,
) -> float:
    """Return maximum of finite numeric values."""

    array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = array[
        np.isfinite(
            array
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


def summarize(
    *,
    model_id: str,
    training_dataset: str,
    metrics: pd.DataFrame,
    original_path: Path,
    inferred_path: Path,
) -> pd.DataFrame:
    """Create system-level dynamic-fidelity summary."""

    if metrics.empty:
        raise RuntimeError(
            "Cannot summarize empty dynamic-fidelity metrics."
        )

    statuses = trajectory_status_table(
        metrics
    )

    total_trajectories = len(
        statuses
    )

    original_successes = int(
        (
            statuses[
                "original_status"
            ]
            == PASS_STATUS
        ).sum()
    )

    inferred_successes = int(
        (
            statuses[
                "inferred_status"
            ]
            == PASS_STATUS
        ).sum()
    )

    comparable_trajectories = int(
        (
            statuses[
                "comparison_status"
            ]
            == SUCCESS_STATUS
        ).sum()
    )

    failed_trajectories = (
        total_trajectories
        - comparable_trajectories
    )

    successful_metrics = metrics[
        metrics[
            "comparison_status"
        ]
        == SUCCESS_STATUS
    ].copy()

    if successful_metrics.empty:
        macro_nrmse = float(
            "inf"
        )

    elif failed_trajectories > 0:
        macro_nrmse = float(
            "inf"
        )

    else:
        trajectory_macro = (
            successful_metrics
            .groupby(
                "trajectory_id",
                sort=False,
            )[
                "normalized_rmse"
            ]
            .mean()
        )

        macro_nrmse = float(
            trajectory_macro.mean()
        )

    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "dataset_name": training_dataset,
                "validation_trajectories": total_trajectories,
                "original_successes": original_successes,
                "original_failures": (
                    total_trajectories
                    - original_successes
                ),
                "inferred_successes": inferred_successes,
                "inferred_failures": (
                    total_trajectories
                    - inferred_successes
                ),
                "comparable_trajectories": comparable_trajectories,
                "failed_trajectories": failed_trajectories,
                "validation_macro_nrmse": macro_nrmse,
                "mean_state_nrmse": finite_mean(
                    successful_metrics[
                        "normalized_rmse"
                    ]
                ),
                "max_state_nrmse": finite_max(
                    successful_metrics[
                        "normalized_rmse"
                    ]
                ),
                "mean_state_rmse": finite_mean(
                    successful_metrics[
                        "rmse"
                    ]
                ),
                "mean_state_r2": finite_mean(
                    successful_metrics[
                        "r2"
                    ]
                ),
                "dynamic_fidelity_status": (
                    SUCCESS_STATUS
                    if failed_trajectories == 0
                    else FAIL_STATUS
                ),
                "original_sbml_path": str(
                    original_path
                ),
                "inferred_sbml_path": str(
                    inferred_path
                ),
            }
        ]
    )


def validate_training_dataset(
    *,
    model_id: str,
    training_dataset: str,
    spec,
    dataset,
    validation: pd.DataFrame,
    original_path: Path,
) -> DynamicFidelityResult:
    """Compare one inferred SBML system with the original SBML."""

    inferred_path = inferred_sbml_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    if not inferred_path.exists():
        raise FileNotFoundError(
            "Inferred SBML not found. "
            "Run src.export_inferred_sbml first. "
            f"Missing: {inferred_path}"
        )

    state_scales = training_state_scales(
        spec=spec,
        dataset=dataset,
        training_dataset=training_dataset,
    )

    trajectory_ids = tuple(
        validation[
            "trajectory_id"
        ]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    metric_rows = []
    prediction_rows = []

    for trajectory_id in trajectory_ids:
        trajectory = validation.loc[
            validation[
                "trajectory_id"
            ].astype(str)
            == trajectory_id
        ].copy()

        current_metrics, current_predictions = (
            evaluate_trajectory(
                model_id=model_id,
                training_dataset=training_dataset,
                trajectory_id=trajectory_id,
                trajectory=trajectory,
                state_ids=tuple(
                    spec.state_ids
                ),
                state_scales=state_scales,
                original_path=original_path,
                inferred_path=inferred_path,
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

    summary = summarize(
        model_id=model_id,
        training_dataset=training_dataset,
        metrics=metrics,
        original_path=original_path,
        inferred_path=inferred_path,
    )

    destination = output_directory(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    current_summary_path = (
        destination
        / SUMMARY_FILENAME
    )

    current_metrics_path = (
        destination
        / TRAJECTORY_METRICS_FILENAME
    )

    current_predictions_path = (
        destination
        / PREDICTIONS_FILENAME
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
            "dynamic_fidelity_status"
        ]
    )

    return DynamicFidelityResult(
        model_id=model_id,
        training_dataset=training_dataset,
        status=status,
        summary_path=current_summary_path,
        trajectory_metrics_path=current_metrics_path,
        predictions_path=current_predictions_path,
    )


def write_comparison(
    *,
    model_id: str,
    results: tuple[DynamicFidelityResult, ...],
) -> Path:
    """Write aggregate comparison across training sizes."""

    frames = [
        pd.read_csv(
            result.summary_path
        )
        for result in results
    ]

    comparison = pd.concat(
        frames,
        ignore_index=True,
    )

    order = {
        dataset_name: index
        for index, dataset_name in enumerate(
            TRAINING_DATASETS
        )
    }

    comparison[
        "_order"
    ] = comparison[
        "dataset_name"
    ].map(
        order
    )

    if comparison[
        "_order"
    ].isna().any():
        unknown = (
            comparison.loc[
                comparison[
                    "_order"
                ].isna(),
                "dataset_name",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Unexpected training datasets in summaries: "
            + ", ".join(
                unknown
            )
        )

    comparison = (
        comparison
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    path = (
        aggregate_output_directory(
            model_id=model_id,
        )
        / COMPARISON_FILENAME
    )

    comparison.to_csv(
        path,
        index=False,
    )

    return path


def print_training_summary(
    *,
    summary: pd.Series,
) -> None:
    """Print one training-condition dynamic-fidelity summary."""

    print(
        "  original integration: "
        f"{int(summary['original_successes'])}/"
        f"{int(summary['validation_trajectories'])}"
    )

    print(
        "  inferred integration: "
        f"{int(summary['inferred_successes'])}/"
        f"{int(summary['validation_trajectories'])}"
    )

    print(
        "  comparable trajectories: "
        f"{int(summary['comparable_trajectories'])}"
    )

    macro_nrmse = float(
        summary[
            "validation_macro_nrmse"
        ]
    )

    if np.isfinite(
        macro_nrmse
    ):
        print(
            "  validation macro NRMSE: "
            f"{macro_nrmse:.8g}"
        )

    else:
        print(
            "  validation macro NRMSE: inf"
        )

    mean_state_nrmse = float(
        summary[
            "mean_state_nrmse"
        ]
    )

    if np.isfinite(
        mean_state_nrmse
    ):
        print(
            "  mean state NRMSE: "
            f"{mean_state_nrmse:.8g}"
        )

    else:
        print(
            "  mean state NRMSE: n/a"
        )

    mean_state_r2 = float(
        summary[
            "mean_state_r2"
        ]
    )

    if np.isfinite(
        mean_state_r2
    ):
        print(
            "  mean state R2: "
            f"{mean_state_r2:.8g}"
        )

    else:
        print(
            "  mean state R2: n/a"
        )

    print(
        "  status: "
        f"{summary['dynamic_fidelity_status']}"
    )


def compare_model(
    model_id: str,
) -> tuple[DynamicFidelityResult, ...]:
    """Compare original and inferred SBML for all training sizes."""

    (
        spec,
        duration,
        validation,
        dataset,
    ) = validation_dataset(
        model_id
    )

    original_path = original_sbml_path(
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
        f"Original SBML: {original_path}"
    )

    print()

    results = []

    for training_dataset in TRAINING_DATASETS:
        print(
            "=" * 70
        )

        print(
            f"Comparing {training_dataset}..."
        )

        result = validate_training_dataset(
            model_id=model_id,
            training_dataset=training_dataset,
            spec=spec,
            dataset=dataset,
            validation=validation,
            original_path=original_path,
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
            summary=summary,
        )

        print(
            f"  summary: {result.summary_path}"
        )

        print()

    result_tuple = tuple(
        results
    )

    aggregate_path = write_comparison(
        model_id=model_id,
        results=result_tuple,
    )

    print(
        "=" * 70
    )

    print(
        "Original-vs-inferred SBML comparison complete."
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
        "Held-out test trajectories used: 0"
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
            "Compare original biological SBML with frozen "
            "PySR-inferred SBML using RoadRunner."
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
    """Run original-vs-inferred SBML comparison."""

    args = parse_arguments()

    compare_model(
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()