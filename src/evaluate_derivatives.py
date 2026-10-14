"""Evaluate inferred PySR equations against exact derivative targets."""

from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from .simulate import MODEL_ID, build_benchmark_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "pysr_historical"
)

CANDIDATES_PATH = RESULTS_DIR / "top_8_by_loss.csv"

OUTPUT_PATH = RESULTS_DIR / "derivative_evaluation.csv"


def load_candidates():
    """Load the top PySR candidates in original physical units."""
    candidates = pd.read_csv(CANDIDATES_PATH)

    required_columns = {
        "complexity",
        "loss",
        "equation_original_units",
    }

    missing_columns = required_columns - set(candidates.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return candidates


def build_derivative_function(equation):
    """Convert a symbolic VENUS RHS into a numerical function."""
    venus = sp.Symbol("VENUS")

    expression = sp.sympify(equation)

    function = sp.lambdify(
        venus,
        expression,
        modules="numpy",
    )

    return function


def evaluate_candidate(
    equation,
    dataset,
):
    """Evaluate one inferred equation against exact derivatives."""
    function = build_derivative_function(equation)

    venus = dataset["VENUS"].to_numpy(dtype=float)

    true_derivative = dataset["d_VENUS_dt"].to_numpy(
        dtype=float
    )

    predicted_derivative = np.asarray(
        function(venus),
        dtype=float,
    )

    if predicted_derivative.ndim == 0:
        predicted_derivative = np.full(
            venus.shape,
            float(predicted_derivative),
            dtype=float,
        )

    predicted_derivative = np.broadcast_to(
        predicted_derivative,
        venus.shape,
    ).astype(float)

    residual = predicted_derivative - true_derivative

    mse = float(
        np.mean(
            residual ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def evaluate_all_candidates(
    candidates,
    datasets,
):
    """Evaluate every candidate separately on train and validation."""
    records = []

    for candidate_rank, (_, candidate) in enumerate(
        candidates.iterrows(),
        start=1,
    ):
        equation = candidate["equation_original_units"]

        for split in [
            "train",
            "validation",
        ]:
            metrics = evaluate_candidate(
                equation,
                datasets[split],
            )

            records.append(
                {
                    "candidate_rank": candidate_rank,
                    "complexity": int(
                        candidate["complexity"]
                    ),
                    "pysr_training_loss_scaled": float(
                        candidate["loss"]
                    ),
                    "split": split,
                    "equation_original_units": equation,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def main():
    """Evaluate saved PySR candidates on reconstructed datasets."""
    print(f"Model: {MODEL_ID}")
    print(f"Candidates: {CANDIDATES_PATH}")
    print()

    candidates = load_candidates()

    datasets, _, _ = build_benchmark_dataset()

    print(f"Loaded candidates: {len(candidates)}")
    print(f"Train rows: {len(datasets['train'])}")
    print(f"Validation rows: {len(datasets['validation'])}")
    print()

    evaluation = evaluate_all_candidates(
        candidates,
        datasets,
    )

    evaluation.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "candidate_rank",
        "complexity",
        "split",
        "mse",
        "rmse",
        "mae",
    ]

    print("Derivative evaluation:")
    print(
        evaluation[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()