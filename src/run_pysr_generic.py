"""Generic multistate PySR inference for ODE systems."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor

from .model_spec import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

TOP_K = 8

NITERATIONS = 60
MAXSIZE = 20
RANDOM_SEED = 1010


@dataclass(frozen=True)
class TrainingMatrices:
    """Normalized feature and derivative matrices for one training dataset."""

    x_values: np.ndarray
    y_values: dict[str, np.ndarray]
    state_scales: dict[str, float]


@dataclass(frozen=True)
class StateFitResult:
    """Results and timing for one inferred state equation."""

    state_id: str
    fit_seconds: float
    equations: pd.DataFrame
    top_candidates: pd.DataFrame


@dataclass(frozen=True)
class GenericFitResult:
    """Results and timing for one multistate PySR inference run."""

    model_id: str
    dataset_name: str
    state_scales: dict[str, float]
    state_results: dict[str, StateFitResult]
    dataset_generation_seconds: float
    training_preparation_seconds: float
    pysr_fit_seconds: float
    orchestration_seconds: float
    total_inference_seconds: float


def build_training_matrices(
    *,
    spec: ModelSpec,
    train_frame,
):
    """Normalize multistate features and derivative targets using train only."""
    state_columns = list(
        spec.state_ids
    )

    missing_state_columns = [
        state_id
        for state_id in state_columns
        if state_id not in train_frame.columns
    ]

    if missing_state_columns:
        raise ValueError(
            "Missing state columns: "
            f"{missing_state_columns}"
        )

    state_scales = {
        state_id: float(
            train_frame[
                state_id
            ]
            .abs()
            .max()
        )
        for state_id in state_columns
    }

    for state_id, scale in state_scales.items():
        if not np.isfinite(
            scale
        ):
            raise ValueError(
                f"Non-finite training scale for state {state_id}."
            )

        if scale <= 0.0:
            raise ValueError(
                f"Training scale for state {state_id} "
                "must be positive."
            )

    x_frame = train_frame[
        state_columns
    ].copy()

    for state_id in state_columns:
        x_frame[
            state_id
        ] = (
            x_frame[
                state_id
            ]
            / state_scales[
                state_id
            ]
        )

    x_values = x_frame.to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            x_values
        )
    ):
        raise ValueError(
            "Normalized feature matrix contains non-finite values."
        )

    y_values = {}

    for state_id in state_columns:
        target_column = (
            f"d_{state_id}_dt"
        )

        if target_column not in train_frame.columns:
            raise ValueError(
                f"Missing derivative target: {target_column}"
            )

        target_scale = state_scales[
            state_id
        ]

        target_values = (
            train_frame[
                target_column
            ]
            .to_numpy(
                dtype=float
            )
            / target_scale
        )

        if not np.all(
            np.isfinite(
                target_values
            )
        ):
            raise ValueError(
                "Normalized derivative target contains "
                f"non-finite values for state {state_id}."
            )

        y_values[
            state_id
        ] = target_values

    return TrainingMatrices(
        x_values=x_values,
        y_values=y_values,
        state_scales=state_scales,
    )


def build_pysr_model(
    *,
    output_dir,
):
    """Construct PySR using the frozen historical inference configuration."""
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return PySRRegressor(
        niterations=NITERATIONS,
        binary_operators=[
            "+",
            "-",
            "*",
            "/",
        ],
        unary_operators=[],
        maxsize=MAXSIZE,
        parsimony=0.0,
        weight_optimize=0.0,
        should_optimize_constants=True,
        optimizer_algorithm="BFGS",
        model_selection="best",
        random_state=RANDOM_SEED,
        progress=False,
        verbosity=0,
        parallelism="serial",
        deterministic=True,
        output_directory=str(
            output_dir
            / "pysr_internal"
        ),
    )


def select_top_candidates(
    equations,
    top_k=TOP_K,
):
    """Select the lowest-training-loss PySR candidates."""
    if equations.empty:
        return equations.copy()

    sorted_equations = equations.copy()

    if "loss" in sorted_equations.columns:
        sorted_equations = (
            sorted_equations.sort_values(
                "loss",
                kind="stable",
            )
        )

    return sorted_equations.head(
        top_k
    ).copy()


def back_transform_equation(
    *,
    expression,
    target_state,
    state_ids,
    state_scales,
):
    """Transform one normalized RHS back to original state coordinates."""
    scaled_expression = sp.sympify(
        str(expression)
    )

    substitutions = {}

    for index, state_id in enumerate(
        state_ids
    ):
        scaled_symbol = sp.Symbol(
            f"x{index}"
        )

        original_symbol = sp.Symbol(
            state_id
        )

        substitutions[
            scaled_symbol
        ] = (
            original_symbol
            / state_scales[
                state_id
            ]
        )

    original_expression = (
        state_scales[
            target_state
        ]
        * scaled_expression.subs(
            substitutions,
            simultaneous=True,
        )
    )

    return sp.simplify(
        original_expression
    )


def rename_scaled_equation(
    *,
    expression,
    state_ids,
):
    """Replace generic PySR feature names with scientific state names."""
    scaled_expression = sp.sympify(
        str(expression)
    )

    substitutions = {
        sp.Symbol(
            f"x{index}"
        ): sp.Symbol(
            state_id
        )
        for index, state_id in enumerate(
            state_ids
        )
    }

    renamed_expression = (
        scaled_expression.subs(
            substitutions,
            simultaneous=True,
        )
    )

    return sp.simplify(
        renamed_expression
    )


def fit_state_equation(
    *,
    spec: ModelSpec,
    state_id,
    matrices: TrainingMatrices,
    output_dir,
):
    """Fit one PySR RHS using all dynamic states as features."""
    state_output_dir = (
        Path(output_dir)
        / f"state_{state_id}"
    )

    model = build_pysr_model(
        output_dir=state_output_dir,
    )

    variable_names = [
        f"x{index}"
        for index in range(
            spec.n_states
        )
    ]

    target_values = matrices.y_values[
        state_id
    ]

    start_time = perf_counter()

    model.fit(
        matrices.x_values,
        target_values,
        variable_names=variable_names,
        X_units=None,
        y_units=None,
    )

    fit_seconds = (
        perf_counter()
        - start_time
    )

    equations = model.equations_

    if isinstance(
        equations,
        list,
    ):
        if len(equations) != 1:
            raise ValueError(
                "Expected one PySR equation table "
                f"for state {state_id}."
            )

        equations = equations[
            0
        ]

    equations = equations.copy()

    equations[
        "equation_scaled"
    ] = [
        str(
            rename_scaled_equation(
                expression=expression,
                state_ids=spec.state_ids,
            )
        )
        for expression in equations[
            "equation"
        ]
    ]

    equations[
        "equation_original_units"
    ] = [
        str(
            back_transform_equation(
                expression=expression,
                target_state=state_id,
                state_ids=spec.state_ids,
                state_scales=matrices.state_scales,
            )
        )
        for expression in equations[
            "equation"
        ]
    ]

    top_candidates = select_top_candidates(
        equations,
        top_k=TOP_K,
    )

    equations.to_csv(
        state_output_dir
        / "all_equations.csv",
        index=False,
    )

    top_candidates.to_csv(
        state_output_dir
        / "top_8_by_loss.csv",
        index=False,
    )

    return StateFitResult(
        state_id=state_id,
        fit_seconds=fit_seconds,
        equations=equations,
        top_candidates=top_candidates,
    )


def run_generic_training_experiment(
    *,
    spec: ModelSpec,
    dataset_name,
    train_frame,
    dataset_generation_seconds=0.0,
):
    """Fit one independent PySR RHS per dynamic state."""
    inference_start_time = perf_counter()

    experiment_root = (
        PROJECT_ROOT
        / "results"
        / spec.model_id
        / "generic_exact_experiment"
    )

    output_dir = (
        experiment_root
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preparation_start_time = perf_counter()

    matrices = build_training_matrices(
        spec=spec,
        train_frame=train_frame,
    )

    training_preparation_seconds = (
        perf_counter()
        - preparation_start_time
    )

    print()
    print("=" * 70)
    print(
        "Generic multistate PySR inference"
    )
    print("=" * 70)
    print(
        "Model:",
        spec.model_id,
    )
    print(
        "Dataset:",
        dataset_name,
    )
    print(
        "States:",
        spec.state_ids,
    )
    print(
        "Trajectories:",
        train_frame[
            "trajectory_id"
        ].nunique(),
    )
    print(
        "Training rows:",
        len(train_frame),
    )
    print(
        "State scales:",
        matrices.state_scales,
    )
    print(
        "X shape:",
        matrices.x_values.shape,
    )
    print("=" * 70)

    state_results = {}

    for state_id in spec.state_ids:
        print()
        print(
            f"Fitting RHS for {state_id}..."
        )

        result = fit_state_equation(
            spec=spec,
            state_id=state_id,
            matrices=matrices,
            output_dir=output_dir,
        )

        state_results[
            state_id
        ] = result

        print(
            f"PySR fit seconds "
            f"({state_id}): "
            f"{result.fit_seconds:.6f}"
        )

        display_columns = [
            "complexity",
            "loss",
            "score",
            "equation_scaled",
            "equation_original_units",
        ]

        available_columns = [
            column
            for column in display_columns
            if column
            in result.top_candidates.columns
        ]

        print()
        print(
            result.top_candidates[
                available_columns
            ].to_string(
                index=False
            )
        )

    pysr_fit_seconds = sum(
        result.fit_seconds
        for result in state_results.values()
    )

    elapsed_before_metadata = (
        perf_counter()
        - inference_start_time
    )

    orchestration_seconds = max(
        0.0,
        (
            elapsed_before_metadata
            - training_preparation_seconds
            - pysr_fit_seconds
        ),
    )

    total_inference_seconds = (
        elapsed_before_metadata
    )

    metadata_row = {
        "model_id": spec.model_id,
        "dataset": dataset_name,
        "number_of_states": spec.n_states,
        "trajectory_count": (
            train_frame[
                "trajectory_id"
            ].nunique()
        ),
        "training_rows": len(
            train_frame
        ),
        "dataset_generation_seconds": float(
            dataset_generation_seconds
        ),
        "training_preparation_seconds": (
            training_preparation_seconds
        ),
        "pysr_fit_seconds": (
            pysr_fit_seconds
        ),
        "orchestration_seconds": (
            orchestration_seconds
        ),
        "total_inference_seconds": (
            total_inference_seconds
        ),
        "niterations": NITERATIONS,
        "maxsize": MAXSIZE,
        "top_k": TOP_K,
        "random_seed": RANDOM_SEED,
        "derivative_method": "exact",
    }

    for state_id in spec.state_ids:
        metadata_row[
            f"state_scale_{state_id}"
        ] = matrices.state_scales[
            state_id
        ]

        metadata_row[
            f"pysr_fit_seconds_{state_id}"
        ] = state_results[
            state_id
        ].fit_seconds

    metadata = pd.DataFrame(
        [
            metadata_row
        ]
    )

    metadata.to_csv(
        output_dir
        / "run_metadata.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("Runtime summary")
    print("=" * 70)

    print(
        "Dataset generation seconds:",
        f"{dataset_generation_seconds:.6f}",
    )

    print(
        "Training preparation seconds:",
        f"{training_preparation_seconds:.6f}",
    )

    for state_id in spec.state_ids:
        print(
            f"PySR fit seconds "
            f"({state_id}):",
            f"{state_results[state_id].fit_seconds:.6f}",
        )

    print(
        "Total PySR fit seconds:",
        f"{pysr_fit_seconds:.6f}",
    )

    print(
        "Inference orchestration seconds:",
        f"{orchestration_seconds:.6f}",
    )

    print(
        "Total inference seconds:",
        f"{total_inference_seconds:.6f}",
    )

    print()
    print(
        "Dataset generation is reported separately "
        "and is not included in total inference time."
    )

    return GenericFitResult(
        model_id=spec.model_id,
        dataset_name=dataset_name,
        state_scales=matrices.state_scales,
        state_results=state_results,
        dataset_generation_seconds=float(
            dataset_generation_seconds
        ),
        training_preparation_seconds=(
            training_preparation_seconds
        ),
        pysr_fit_seconds=pysr_fit_seconds,
        orchestration_seconds=(
            orchestration_seconds
        ),
        total_inference_seconds=(
            total_inference_seconds
        ),
    )