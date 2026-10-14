"""Evaluate multistate PySR systems on held-out validation trajectories."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    DerivativeMethod,
    validate_derivative_provenance,
)
from .model_spec import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

SOLVER_METHOD = "LSODA"
RTOL = 1e-7
ATOL = 1e-9
MAX_RHS_EVALUATIONS = 8000


def resolve_validation_experiment_root(
    *,
    model_id: str,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
    project_root: Path | None = None,
) -> Path:
    """Resolve the candidate and validation root for one benchmark variant.

    Args:
        model_id: BioModels identifier used as the model directory.
        variant: Validated derivative method and protected namespace.
        project_root: Optional project root override for tests.

    Returns:
        Variant-specific model experiment directory.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")

    root = PROJECT_ROOT if project_root is None else Path(project_root)

    return root / "results" / model_id / variant.experiment_namespace


def _read_single_metadata_row(path: Path) -> dict[str, object]:
    """Read one required provenance metadata row.

    Args:
        path: CSV metadata path.

    Returns:
        The sole metadata row as a dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing derivative provenance metadata: {path}"
        )

    metadata = pd.read_csv(path)

    if len(metadata) != 1:
        raise ValueError(f"Expected exactly one metadata row in {path}.")

    return metadata.iloc[0].to_dict()


def _validate_candidate_metadata_row(
    *,
    metadata: dict[str, object],
    metadata_path: Path,
    experiment_root: Path,
    variant: BenchmarkVariant,
) -> None:
    """Validate candidate metadata against its path and benchmark variant.

    Frozen exact artifacts predate the ``experiment_namespace`` metadata
    column. For those artifacts only, the already-protected directory name is
    used as namespace provenance. New variants must record the field.

    Args:
        metadata: One experiment- or training-level metadata row.
        metadata_path: Source path used in validation errors.
        experiment_root: Resolved variant experiment root.
        variant: Requested benchmark variant.
    """
    provenance = {
        key: metadata[key]
        for key in (
            "derivative_method",
            "experiment_namespace",
        )
        if key in metadata and not pd.isna(metadata[key])
    }

    if (
        "experiment_namespace" not in provenance
        and variant.derivative_method is DerivativeMethod.EXACT
    ):
        provenance["experiment_namespace"] = experiment_root.name

    try:
        validate_derivative_provenance(
            provenance=provenance,
            variant=variant,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Candidate provenance mismatch in {metadata_path}: {error}"
        ) from error


def validate_candidate_pool_provenance(
    *,
    experiment_root: Path,
    variant: BenchmarkVariant,
    dataset_names: tuple[str, ...] = TRAINING_DATASETS,
) -> None:
    """Validate all candidate metadata before selection can write outputs.

    Args:
        experiment_root: Resolved model experiment directory.
        variant: Requested benchmark variant.
        dataset_names: Training conditions whose metadata must agree.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")

    experiment_root = Path(experiment_root)

    validate_derivative_provenance(
        provenance={
            "derivative_method": variant.derivative_method.value,
            "experiment_namespace": experiment_root.name,
        },
        variant=variant,
    )

    metadata_paths = [
        experiment_root / "experiment_metadata.csv",
        *(
            experiment_root / dataset_name / "run_metadata.csv"
            for dataset_name in dataset_names
        ),
    ]

    for metadata_path in metadata_paths:
        metadata = _read_single_metadata_row(metadata_path)
        _validate_candidate_metadata_row(
            metadata=metadata,
            metadata_path=metadata_path,
            experiment_root=experiment_root,
            variant=variant,
        )


def validate_selection_development_dataset(
    *,
    dataset,
    spec: ModelSpec,
    variant: BenchmarkVariant,
) -> None:
    """Validate derivative lineage and the selection information boundary.

    Args:
        dataset: In-memory generic development dataset.
        spec: Ordered dynamic-state metadata.
        variant: Requested benchmark variant.
    """
    validate_derivative_provenance(
        provenance=dataset.derivative_provenance,
        variant=variant,
    )

    if not dataset.test.empty:
        raise ValueError(
            "Held-out test data must remain unmaterialized during validation."
        )

    if variant.derivative_method is DerivativeMethod.SAVGOL:
        expected_columns = (
            "trajectory_id",
            "time",
            *spec.state_ids,
        )

        if tuple(dataset.validation.columns) != expected_columns:
            raise ValueError(
                "Savitzky-Golay validation must contain only trajectory_id, "
                "time, and ordered state columns."
            )


def compile_rhs_system(
    *,
    spec: ModelSpec,
    expressions,
):
    """Compile one symbolic RHS expression for every dynamic state."""
    symbols = [
        sp.Symbol(state_id)
        for state_id in spec.state_ids
    ]

    symbol_map = {
        state_id: symbol
        for state_id, symbol in zip(
            spec.state_ids,
            symbols,
        )
    }

    functions = []

    for state_id in spec.state_ids:
        if state_id not in expressions:
            raise ValueError(
                f"Missing RHS expression for state {state_id}."
            )

        expression = sp.sympify(
            expressions[state_id],
            locals=symbol_map,
        )

        function = sp.lambdify(
            symbols,
            expression,
            modules="numpy",
        )

        functions.append(
            function
        )

    return functions


def integrate_system(
    *,
    rhs_functions,
    time_values,
    initial_values,
):
    """Integrate one coupled candidate ODE system over one trajectory."""
    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    initial_values = np.asarray(
        initial_values,
        dtype=float,
    )

    if time_values.ndim != 1:
        raise ValueError(
            "Time values must be one-dimensional."
        )

    if initial_values.ndim != 1:
        raise ValueError(
            "Initial values must be one-dimensional."
        )

    if len(rhs_functions) != len(
        initial_values
    ):
        raise ValueError(
            "Number of RHS functions does not match "
            "number of initial states."
        )

    if not np.all(
        np.isfinite(
            initial_values
        )
    ):
        raise ValueError(
            "Initial values must be finite."
        )

    rhs_evaluations = 0


    def ode_system(
        _time,
        state,
    ):
        nonlocal rhs_evaluations

        rhs_evaluations += 1

        if rhs_evaluations > MAX_RHS_EVALUATIONS:
            raise RuntimeError(
                "Maximum number of RHS evaluations exceeded."
            )

        arguments = [
            float(value)
            for value in state
        ]

        derivatives = []

        for rhs_function in rhs_functions:
            value = rhs_function(
                *arguments
            )

            value = float(
                np.asarray(
                    value
                )
            )

            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    "Candidate produced a non-finite derivative."
                )

            derivatives.append(
                value
            )

        return derivatives


    solution = solve_ivp(
        fun=ode_system,
        t_span=(
            float(
                time_values[0]
            ),
            float(
                time_values[-1]
            ),
        ),
        y0=initial_values,
        t_eval=time_values,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
    )

    if not solution.success:
        raise RuntimeError(
            solution.message
        )

    if solution.y.shape != (
        len(initial_values),
        len(time_values),
    ):
        raise RuntimeError(
            "Integration did not return the expected state-time matrix."
        )

    if not np.all(
        np.isfinite(
            solution.y
        )
    ):
        raise RuntimeError(
            "Integration returned non-finite state values."
        )

    return (
        solution.y.T,
        rhs_evaluations,
    )


def calculate_state_metrics(
    *,
    observed,
    predicted,
    state_scale,
):
    """Calculate original-unit and training-scale-normalized errors."""
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted state arrays must have equal shape."
        )

    if state_scale <= 0.0:
        raise ValueError(
            "State scale must be positive."
        )

    residual = (
        predicted
        - observed
    )

    mse = float(
        np.mean(
            residual**2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )

    normalized_rmse = (
        rmse
        / float(
            state_scale
        )
    )

    ss_res = float(
        np.sum(
            residual**2
        )
    )

    ss_tot = float(
        np.sum(
            (
                observed
                - np.mean(
                    observed
                )
            )
            ** 2
        )
    )

    if ss_tot > 0.0:
        r2 = (
            1.0
            - ss_res
            / ss_tot
        )
    else:
        r2 = np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "normalized_rmse": normalized_rmse,
        "r2": r2,
    }


def load_state_candidates(
    *,
    experiment_dir,
    spec: ModelSpec,
):
    """Load top PySR candidates for every dynamic state."""
    candidates = {}

    required_columns = {
        "complexity",
        "loss",
        "equation_original_units",
    }

    for state_id in spec.state_ids:
        candidate_path = (
            Path(experiment_dir)
            / f"state_{state_id}"
            / "top_8_by_loss.csv"
        )

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Missing candidate file: {candidate_path}"
            )

        frame = pd.read_csv(
            candidate_path
        )

        missing_columns = (
            required_columns
            - set(
                frame.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"Candidate file for {state_id} "
                "is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if frame.empty:
            raise ValueError(
                f"No candidates available for state {state_id}."
            )

        frame = (
            frame
            .reset_index(
                drop=True
            )
            .copy()
        )

        frame[
            "candidate_rank"
        ] = (
            np.arange(
                len(frame)
            )
            + 1
        )

        candidates[
            state_id
        ] = frame

    return candidates


def build_candidate_systems(
    *,
    spec: ModelSpec,
    candidates,
):
    """Enumerate candidate RHS combinations for the current state system."""
    candidate_ranges = [
        range(
            len(
                candidates[
                    state_id
                ]
            )
        )
        for state_id in spec.state_ids
    ]

    systems = []

    for system_index, combination in enumerate(
        product(
            *candidate_ranges
        ),
        start=1,
    ):
        state_rows = {}

        for state_id, candidate_index in zip(
            spec.state_ids,
            combination,
        ):
            state_rows[
                state_id
            ] = (
                candidates[
                    state_id
                ]
                .iloc[
                    candidate_index
                ]
            )

        systems.append(
            (
                system_index,
                state_rows,
            )
        )

    return systems


def evaluate_system(
    *,
    spec: ModelSpec,
    system_id,
    state_rows,
    validation_frame,
    state_scales,
):
    """Evaluate one coupled candidate system across validation trajectories."""
    expressions = {
        state_id: state_rows[
            state_id
        ][
            "equation_original_units"
        ]
        for state_id in spec.state_ids
    }

    rhs_functions = compile_rhs_system(
        spec=spec,
        expressions=expressions,
    )

    metric_rows = []
    prediction_rows = []

    successful_trajectories = 0
    failed_trajectories = 0
    total_rhs_evaluations = 0

    failure_messages = []

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

        initial_values = (
            observed_matrix[
                0,
                :
            ]
        )

        try:
            (
                predicted_matrix,
                rhs_evaluations,
            ) = integrate_system(
                rhs_functions=rhs_functions,
                time_values=time_values,
                initial_values=initial_values,
            )

        except Exception as error:
            failed_trajectories += 1

            failure_messages.append(
                f"{trajectory_id}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            continue

        successful_trajectories += 1

        total_rhs_evaluations += (
            rhs_evaluations
        )

        trajectory_normalized_errors = []

        for state_index, state_id in enumerate(
            spec.state_ids
        ):
            observed = (
                observed_matrix[
                    :,
                    state_index,
                ]
            )

            predicted = (
                predicted_matrix[
                    :,
                    state_index,
                ]
            )

            metrics = calculate_state_metrics(
                observed=observed,
                predicted=predicted,
                state_scale=state_scales[
                    state_id
                ],
            )

            trajectory_normalized_errors.append(
                metrics[
                    "normalized_rmse"
                ]
            )

            metric_rows.append(
                {
                    "system_id": system_id,
                    "trajectory_id": trajectory_id,
                    "state_id": state_id,
                    "candidate_rank": int(
                        state_rows[
                            state_id
                        ][
                            "candidate_rank"
                        ]
                    ),
                    "complexity": int(
                        state_rows[
                            state_id
                        ][
                            "complexity"
                        ]
                    ),
                    "training_loss": float(
                        state_rows[
                            state_id
                        ][
                            "loss"
                        ]
                    ),
                    "equation_original_units": (
                        expressions[
                            state_id
                        ]
                    ),
                    "rhs_evaluations": (
                        rhs_evaluations
                    ),
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
                        "system_id": system_id,
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

        trajectory_macro_nrmse = float(
            np.mean(
                trajectory_normalized_errors
            )
        )

        for row in metric_rows[
            -spec.n_states:
        ]:
            row[
                "trajectory_macro_nrmse"
            ] = trajectory_macro_nrmse

    return {
        "metric_rows": metric_rows,
        "prediction_rows": prediction_rows,
        "successful_trajectories": successful_trajectories,
        "failed_trajectories": failed_trajectories,
        "total_rhs_evaluations": total_rhs_evaluations,
        "failure_messages": failure_messages,
    }


def summarize_system(
    *,
    spec: ModelSpec,
    system_id,
    state_rows,
    evaluation,
    expected_trajectories,
):
    """Create one validation summary row for a candidate ODE system."""
    complexity_total = sum(
        int(
            state_rows[
                state_id
            ][
                "complexity"
            ]
        )
        for state_id in spec.state_ids
    )

    training_loss_total = sum(
        float(
            state_rows[
                state_id
            ][
                "loss"
            ]
        )
        for state_id in spec.state_ids
    )

    summary_row = {
        "system_id": system_id,
        "successful_trajectories": (
            evaluation[
                "successful_trajectories"
            ]
        ),
        "failed_trajectories": (
            evaluation[
                "failed_trajectories"
            ]
        ),
        "total_rhs_evaluations": (
            evaluation[
                "total_rhs_evaluations"
            ]
        ),
        "complexity_total": (
            complexity_total
        ),
        "training_loss_total": (
            training_loss_total
        ),
    }

    for state_id in spec.state_ids:
        state_row = state_rows[
            state_id
        ]

        summary_row[
            f"candidate_rank_{state_id}"
        ] = int(
            state_row[
                "candidate_rank"
            ]
        )

        summary_row[
            f"complexity_{state_id}"
        ] = int(
            state_row[
                "complexity"
            ]
        )

        summary_row[
            f"training_loss_{state_id}"
        ] = float(
            state_row[
                "loss"
            ]
        )

        summary_row[
            f"equation_{state_id}"
        ] = state_row[
            "equation_original_units"
        ]

    if (
        evaluation[
            "failed_trajectories"
        ]
        > 0
        or evaluation[
            "successful_trajectories"
        ]
        != expected_trajectories
    ):
        summary_row[
            "validation_macro_nrmse"
        ] = np.inf

        summary_row[
            "validation_status"
        ] = "failed"

    else:
        metrics_frame = pd.DataFrame(
            evaluation[
                "metric_rows"
            ]
        )

        trajectory_errors = (
            metrics_frame[
                [
                    "trajectory_id",
                    "trajectory_macro_nrmse",
                ]
            ]
            .drop_duplicates()
        )

        summary_row[
            "validation_macro_nrmse"
        ] = float(
            trajectory_errors[
                "trajectory_macro_nrmse"
            ].mean()
        )

        summary_row[
            "validation_status"
        ] = "success"

        for state_id in spec.state_ids:
            state_metrics = (
                metrics_frame[
                    metrics_frame[
                        "state_id"
                    ]
                    == state_id
                ]
            )

            summary_row[
                f"mean_rmse_{state_id}"
            ] = float(
                state_metrics[
                    "rmse"
                ].mean()
            )

            summary_row[
                f"mean_normalized_rmse_{state_id}"
            ] = float(
                state_metrics[
                    "normalized_rmse"
                ].mean()
            )

            summary_row[
                f"mean_r2_{state_id}"
            ] = float(
                state_metrics[
                    "r2"
                ].mean()
            )

    summary_row[
        "failure_messages"
    ] = " | ".join(
        evaluation[
            "failure_messages"
        ]
    )

    return summary_row


def load_state_scales(
    *,
    experiment_dir,
    spec: ModelSpec,
):
    """Load training-only state scales from the PySR run metadata."""
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

    state_scales = {}

    for state_id in spec.state_ids:
        column = (
            f"state_scale_{state_id}"
        )

        if column not in metadata.columns:
            raise ValueError(
                f"Missing state scale column: {column}"
            )

        scale = float(
            metadata.iloc[
                0
            ][
                column
            ]
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= 0.0
        ):
            raise ValueError(
                f"Invalid state scale for {state_id}: {scale}"
            )

        state_scales[
            state_id
        ] = scale

    return state_scales


def evaluate_validation_dataset(
    *,
    spec: ModelSpec,
    dataset_name,
    validation_frame,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
    project_root: Path | None = None,
):
    """Evaluate all candidate systems for one training-size experiment."""
    experiment_root = resolve_validation_experiment_root(
        model_id=spec.model_id,
        variant=variant,
        project_root=project_root,
    )

    validate_candidate_pool_provenance(
        experiment_root=experiment_root,
        variant=variant,
        dataset_names=(dataset_name,),
    )

    experiment_dir = (
        experiment_root
        / dataset_name
    )

    candidates = load_state_candidates(
        experiment_dir=experiment_dir,
        spec=spec,
    )

    state_scales = load_state_scales(
        experiment_dir=experiment_dir,
        spec=spec,
    )

    systems = build_candidate_systems(
        spec=spec,
        candidates=candidates,
    )

    expected_trajectories = (
        validation_frame[
            "trajectory_id"
        ].nunique()
    )

    print()
    print("=" * 70)
    print(
        f"Validation: {dataset_name}"
    )
    print("=" * 70)
    print(
        "States:",
        spec.state_ids,
    )
    print(
        "State scales:",
        state_scales,
    )
    print(
        "Validation trajectories:",
        expected_trajectories,
    )
    print(
        "Candidate systems:",
        len(systems),
    )

    summary_rows = []
    all_metric_rows = []
    all_prediction_rows = []

    for (
        system_id,
        state_rows,
    ) in systems:
        evaluation = evaluate_system(
            spec=spec,
            system_id=system_id,
            state_rows=state_rows,
            validation_frame=validation_frame,
            state_scales=state_scales,
        )

        summary_row = summarize_system(
            spec=spec,
            system_id=system_id,
            state_rows=state_rows,
            evaluation=evaluation,
            expected_trajectories=expected_trajectories,
        )

        summary_rows.append(
            summary_row
        )

        all_metric_rows.extend(
            evaluation[
                "metric_rows"
            ]
        )

        all_prediction_rows.extend(
            evaluation[
                "prediction_rows"
            ]
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = (
        summary
        .sort_values(
            [
                "validation_macro_nrmse",
                "complexity_total",
                "training_loss_total",
                "system_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "validation_rank"
    ] = (
        np.arange(
            len(summary)
        )
        + 1
    )

    selected = (
        summary
        .iloc[
            [
                0
            ]
        ]
        .copy()
    )

    metrics_frame = pd.DataFrame(
        all_metric_rows
    )

    predictions_frame = pd.DataFrame(
        all_prediction_rows
    )

    summary.to_csv(
        experiment_dir
        / "validation_system_ranking.csv",
        index=False,
    )

    selected.to_csv(
        experiment_dir
        / "selected_system.csv",
        index=False,
    )

    metrics_frame.to_csv(
        experiment_dir
        / "validation_state_metrics.csv",
        index=False,
    )

    predictions_frame.to_csv(
        experiment_dir
        / "validation_predictions.csv",
        index=False,
    )

    print()
    print("Top validation systems:")
    print()

    display_columns = [
        "validation_rank",
        "system_id",
        "validation_status",
        "validation_macro_nrmse",
        "complexity_total",
        "training_loss_total",
    ]

    for state_id in spec.state_ids:
        display_columns.append(
            f"candidate_rank_{state_id}"
        )

        display_columns.append(
            f"mean_normalized_rmse_{state_id}"
        )

    available_columns = [
        column
        for column in display_columns
        if column in summary.columns
    ]

    print(
        summary[
            available_columns
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Selected system:",
        int(
            selected.iloc[
                0
            ][
                "system_id"
            ]
        ),
    )

    print(
        "Validation macro NRMSE:",
        selected.iloc[
            0
        ][
            "validation_macro_nrmse"
        ],
    )

    print()
    print(
        "Saved:",
        experiment_dir
        / "validation_system_ranking.csv",
    )

    print(
        "Saved:",
        experiment_dir
        / "selected_system.csv",
    )

    return summary, selected
