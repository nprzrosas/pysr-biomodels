"""Run the PySR training-size experiment on nested trajectory datasets."""

from pathlib import Path

import pandas as pd
import sympy as sp
from pysr import PySRRegressor

from .simulate import MODEL_ID, build_experimental_dataset


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


def build_training_matrices(train_frame):
    """Normalize one training dataset using training data only."""
    state_columns = ["VENUS"]
    target_columns = ["d_VENUS_dt"]

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

    y_frame["d_VENUS_dt"] = (
        y_frame["d_VENUS_dt"]
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
    """Construct PySR using the historical inference configuration."""
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
    """Fit one PySR model for a specified nested training dataset."""
    output_dir = (
        EXPERIMENT_ROOT
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        x_values,
        y_values,
        state_scales,
    ) = build_training_matrices(
        train_frame
    )

    print()
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(
        "Trajectories:",
        train_frame["trajectory_id"].nunique(),
    )
    print(
        "Training rows:",
        len(train_frame),
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

    top_candidates = (
        select_top_candidates(
            equations,
            top_k=8,
        )
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
                "model_id": MODEL_ID,
                "dataset": dataset_name,
                "trajectory_count": (
                    train_frame[
                        "trajectory_id"
                    ].nunique()
                ),
                "row_count": len(
                    train_frame
                ),
                "state_scale_VENUS": (
                    state_scale
                ),
                "niterations": 60,
                "maxsize": 20,
                "random_seed": 1010,
            }
        ]
    )

    metadata.to_csv(
        output_dir
        / "run_metadata.csv",
        index=False,
    )

    display_columns = [
        "complexity",
        "loss",
        "score",
        "equation_scaled",
        "equation_original_units",
    ]

    print(
        "Top 8 candidates "
        "by training loss:"
    )

    print(
        top_candidates[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Results: {output_dir}"
    )

    return top_candidates


def main():
    """Run PySR independently for all nested training-set sizes."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_name in TRAINING_DATASETS:
        run_training_experiment(
            dataset_name=dataset_name,
            train_frame=datasets[
                dataset_name
            ],
        )


if __name__ == "__main__":
    main()