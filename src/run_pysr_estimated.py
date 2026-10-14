"""Run PySR using derivatives estimated from time-series observations."""

from pathlib import Path

import pandas as pd
import sympy as sp
from pysr import PySRRegressor

from .estimate_derivatives import estimate_dataset_derivatives
from .simulate import MODEL_ID, build_experimental_dataset


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


def build_training_matrices(train_frame):
    """Normalize states and estimated derivatives using training data only."""
    state_columns = ["VENUS"]
    target_columns = ["d_VENUS_dt_estimated"]

    state_scales = {
        state: float(train_frame[state].abs().max())
        for state in state_columns
    }

    x_frame = train_frame[state_columns].copy()
    y_frame = train_frame[target_columns].copy()

    for state in state_columns:
        x_frame[state] = (
            x_frame[state]
            / state_scales[state]
        )

    target_state = "VENUS"
    target_scale = state_scales[target_state]

    y_frame["d_VENUS_dt_estimated"] = (
        y_frame["d_VENUS_dt_estimated"]
        / target_scale
    )

    x_values = x_frame.to_numpy(
        dtype=float
    )

    y_values = y_frame.to_numpy(
        dtype=float
    )

    return (
        x_values,
        y_values,
        state_scales,
    )


def build_pysr_model(output_dir):
    """Construct PySR using the frozen historical inference configuration."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PySRRegressor(
        niterations=60,
        binary_operators=[
            "+",
            "-",
            "*",
            "/",
        ],
        unary_operators=[],
        maxsize=20,
        parsimony=0.0,
        weight_optimize=0.0,
        should_optimize_constants=True,
        optimizer_algorithm="BFGS",
        model_selection="best",
        random_state=1010,
        progress=False,
        verbosity=0,
        parallelism="serial",
        deterministic=True,
        output_directory=str(
            output_dir / "pysr_internal"
        ),
    )

    return model


def select_top_candidates(
    equations,
    top_k=8,
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
    expression,
    state_scale,
):
    """Transform a one-state RHS from scaled to original coordinates."""
    scaled_state = sp.Symbol("x0")
    original_state = sp.Symbol("VENUS")

    scaled_expression = sp.sympify(
        str(expression)
    )

    original_expression = (
        state_scale
        * scaled_expression.subs(
            scaled_state,
            original_state / state_scale,
        )
    )

    return sp.simplify(
        original_expression
    )


def run_training_experiment(
    *,
    dataset_name,
    train_frame,
):
    """Fit PySR using Savitzky-Golay estimated derivatives."""
    output_dir = (
        EXPERIMENT_ROOT
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimated_frame = estimate_dataset_derivatives(
        train_frame
    )

    (
        x_values,
        y_values,
        state_scales,
    ) = build_training_matrices(
        estimated_frame
    )

    print()
    print("=" * 70)
    print(
        f"Estimated-derivative dataset: {dataset_name}"
    )
    print(
        "Trajectories:",
        estimated_frame["trajectory_id"].nunique(),
    )
    print(
        "Training rows:",
        len(estimated_frame),
    )
    print(
        "Derivative target:",
        "d_VENUS_dt_estimated",
    )
    print(
        "State scales:",
        state_scales,
    )
    print(
        "X shape:",
        x_values.shape,
    )
    print(
        "Y shape:",
        y_values.shape,
    )
    print("=" * 70)
    print()

    model = build_pysr_model(
        output_dir
    )

    model.fit(
        x_values,
        y_values,
        variable_names=["x0"],
        X_units=None,
        y_units=None,
    )

    equations = model.equations_

    if isinstance(equations, list):
        equations = equations[0]

    equations = equations.copy()

    equations["equation_scaled"] = (
        equations["equation"]
        .astype(str)
        .str.replace(
            r"\bx0\b",
            "VENUS",
            regex=True,
        )
    )

    state_scale = state_scales[
        "VENUS"
    ]

    equations[
        "equation_original_units"
    ] = [
        str(
            back_transform_equation(
                expression,
                state_scale,
            )
        )
        for expression in equations[
            "equation"
        ]
    ]

    equations = equations.sort_values(
        "loss",
        kind="stable",
    ).reset_index(
        drop=True
    )

    equations[
        "pysr_loss_rank"
    ] = range(
        1,
        len(equations) + 1,
    )

    top_candidates = select_top_candidates(
        equations,
        top_k=8,
    )

    equations.to_csv(
        output_dir
        / "all_equations.csv",
        index=False,
    )

    top_candidates.to_csv(
        output_dir
        / "top_8_by_loss.csv",
        index=False,
    )

    metadata = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "derivative_method": (
                    "savitzky_golay"
                ),
                "derivative_column": (
                    "d_VENUS_dt_estimated"
                ),
                "savgol_window_length": 11,
                "savgol_polyorder": 3,
                "training_trajectories": (
                    estimated_frame[
                        "trajectory_id"
                    ].nunique()
                ),
                "training_rows": len(
                    estimated_frame
                ),
                "state_scale_VENUS": (
                    state_scale
                ),
                "pysr_niterations": 60,
                "pysr_maxsize": 20,
                "pysr_random_seed": 1010,
            }
        ]
    )

    metadata.to_csv(
        output_dir
        / "run_metadata.csv",
        index=False,
    )

    print(
        "Top 8 candidates by training loss:"
    )

    display_columns = [
        "pysr_loss_rank",
        "complexity",
        "loss",
        "score",
        "equation_scaled",
        "equation_original_units",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in top_candidates.columns
    ]

    print(
        top_candidates[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Results:",
        output_dir,
    )

    return top_candidates


def main():
    """Run the estimated-derivative experiment for all training sizes."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    for dataset_name in TRAINING_DATASETS:
        run_training_experiment(
            dataset_name=dataset_name,
            train_frame=datasets[
                dataset_name
            ],
        )


if __name__ == "__main__":
    main()