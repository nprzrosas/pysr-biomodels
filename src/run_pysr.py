"""Run PySR on the reconstructed historical benchmark dataset."""

from pathlib import Path

import pandas as pd
import sympy as sp
from pysr import PySRRegressor

from .simulate import MODEL_ID, build_benchmark_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "pysr_historical"
)


def build_training_matrices(datasets):
    """Build the normalized matrices used by the historical PySR run."""
    train = datasets["train"]

    state_columns = ["VENUS"]
    target_columns = ["d_VENUS_dt"]

    state_scales = {
        state: float(train[state].abs().max())
        for state in state_columns
    }

    x_frame = train[state_columns].copy()
    y_frame = train[target_columns].copy()

    for state in state_columns:
        x_frame[state] = x_frame[state] / state_scales[state]

    target_state = "VENUS"
    target_scale = state_scales[target_state]
    y_frame["d_VENUS_dt"] = y_frame["d_VENUS_dt"] / target_scale

    x_values = x_frame.to_numpy(dtype=float)
    y_values = y_frame.to_numpy(dtype=float)

    return x_values, y_values, state_scales


def build_pysr_model():
    """Construct PySR using the reconstructed historical configuration."""
    OUTPUT_DIR.mkdir(
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
            OUTPUT_DIR / "pysr_internal"
        ),
    )

    return model


def select_top_candidates(
    equations,
    top_k=8,
):
    """Select candidates using the historical lowest-loss criterion."""
    if equations.empty:
        return equations.copy()

    sorted_equations = equations.copy()

    if "loss" in sorted_equations.columns:
        sorted_equations = sorted_equations.sort_values(
            "loss",
            kind="stable",
        )

    return sorted_equations.head(top_k).copy()


def back_transform_equation(
    expression,
    state_scale,
):
    """Transform a one-state RHS from scaled to original coordinates."""
    scaled_state = sp.Symbol("x0")
    original_state = sp.Symbol("VENUS")

    scaled_expression = sp.sympify(str(expression))

    original_expression = state_scale * scaled_expression.subs(
        scaled_state,
        original_state / state_scale,
    )

    return sp.simplify(original_expression)


def main():
    """Fit PySR to the reconstructed training data."""
    datasets, _, _ = build_benchmark_dataset()

    train_frame = datasets["train"]

    x_values, y_values, state_scales = build_training_matrices(datasets)
    print("State scales:", state_scales)

    print(f"Model: {MODEL_ID}")
    print(f"Training rows: {len(train_frame)}")
    print(f"X shape: {x_values.shape}")
    print(f"Y shape: {y_values.shape}")
    print("Variable mapping: x0 -> VENUS")
    print()

    model = build_pysr_model()

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

    state_scale = state_scales["VENUS"]

    equations["equation_original_units"] = [
        str(
            back_transform_equation(
                expression,
                state_scale,
            )
        )
        for expression in equations["equation"]
    ]

    top_candidates = select_top_candidates(
        equations,
        top_k=8,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    equations.to_csv(
        OUTPUT_DIR / "all_equations.csv",
        index=False,
    )

    top_candidates.to_csv(
        OUTPUT_DIR / "top_8_by_loss.csv",
        index=False,
    )

    display_columns = [
        "complexity",
        "loss",
        "score",
        "equation_scaled",
        "equation_original_units",
    ]

    print("All equations:")
    print(
        equations[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("Top 8 candidates by training loss:")
    print(
        top_candidates[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()