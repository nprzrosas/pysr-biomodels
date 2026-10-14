"""Evaluate PySR candidate equations by forward ODE integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp

from .simulate import MODEL_ID, build_benchmark_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYSR_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "pysr_historical"
)

CANDIDATE_PATH = PYSR_RESULTS_DIR / "top_8_by_loss.csv"

OUTPUT_PATH = PYSR_RESULTS_DIR / "trajectory_evaluation.csv"

TRAJECTORY_OUTPUT_PATH = (
    PYSR_RESULTS_DIR / "trajectory_predictions.csv"
)

SOLVER_METHOD = "LSODA"
RTOL = 1e-7
ATOL = 1e-9


def compile_rhs(expression_text):
    """Compile an original-unit symbolic expression into a numerical RHS."""
    venus = sp.Symbol("VENUS")

    expression = sp.sympify(expression_text)

    function = sp.lambdify(
        venus,
        expression,
        modules="numpy",
    )

    return function


def integrate_candidate(rhs_function, time_values, initial_value):
    """Integrate one candidate equation over one trajectory."""
    rhs_evaluations = 0


    def ode_system(_time, state):
        nonlocal rhs_evaluations

        rhs_evaluations += 1

        if rhs_evaluations > 8000:
            raise RuntimeError(
                "Maximum number of RHS evaluations exceeded."
            )

        value = rhs_function(state[0])

        value = float(np.asarray(value))

        if not np.isfinite(value):
            raise RuntimeError(
                "Candidate produced a non-finite derivative."
            )

        return [value]


    solution = solve_ivp(
        fun=ode_system,
        t_span=(
            float(time_values[0]),
            float(time_values[-1]),
        ),
        y0=[float(initial_value)],
        t_eval=time_values,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    if solution.y.shape[1] != len(time_values):
        raise RuntimeError(
            "Integration did not return all requested time points."
        )

    return solution.y[0], solution.nfev


def calculate_metrics(observed, predicted):
    """Calculate trajectory metrics in original state units."""
    residual = predicted - observed

    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    dynamic_range = float(
        np.max(observed) - np.min(observed)
    )

    if dynamic_range > 0.0:
        relative_rmse = rmse / dynamic_range
    else:
        relative_rmse = np.inf

    ss_res = float(
        np.sum((observed - predicted) ** 2)
    )

    ss_tot = float(
        np.sum((observed - np.mean(observed)) ** 2)
    )

    if ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": relative_rmse,
        "r2": r2,
        "dynamic_range": dynamic_range,
    }


def evaluate_candidate(
    candidate_rank,
    candidate_row,
    datasets,
):
    """Evaluate one PySR candidate across train and validation trajectories."""
    expression = candidate_row["equation_original_units"]

    rhs_function = compile_rhs(expression)

    metric_rows = []
    prediction_rows = []

    for split in ("train", "validation"):
        dataset = datasets[split]

        for trajectory_id, trajectory in dataset.groupby(
            "trajectory_id",
            sort=False,
        ):
            trajectory = trajectory.sort_values("time")

            time_values = trajectory["time"].to_numpy(
                dtype=float
            )

            observed = trajectory["VENUS"].to_numpy(
                dtype=float
            )

            predicted, nfev = integrate_candidate(
                rhs_function=rhs_function,
                time_values=time_values,
                initial_value=observed[0],
            )

            metrics = calculate_metrics(
                observed=observed,
                predicted=predicted,
            )

            metric_rows.append(
                {
                    "candidate_rank": candidate_rank,
                    "complexity": int(
                        candidate_row["complexity"]
                    ),
                    "split": split,
                    "trajectory_id": trajectory_id,
                    "equation_original_units": expression,
                    "nfev": nfev,
                    **metrics,
                }
            )

            for time, original, inferred in zip(
                time_values,
                observed,
                predicted,
            ):
                prediction_rows.append(
                    {
                        "candidate_rank": candidate_rank,
                        "complexity": int(
                            candidate_row["complexity"]
                        ),
                        "split": split,
                        "trajectory_id": trajectory_id,
                        "time": time,
                        "VENUS_original": original,
                        "VENUS_inferred": inferred,
                        "error": inferred - original,
                    }
                )

    return metric_rows, prediction_rows


def main():
    """Run forward trajectory evaluation for the top PySR candidates."""
    datasets, _, _ = build_benchmark_dataset()

    candidates = pd.read_csv(CANDIDATE_PATH)

    required_columns = {
        "complexity",
        "equation_original_units",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing required candidate columns: {sorted(missing)}"
        )

    all_metrics = []
    all_predictions = []

    for candidate_rank, (_, candidate) in enumerate(
        candidates.iterrows(),
        start=1,
    ):
        metrics, predictions = evaluate_candidate(
            candidate_rank=candidate_rank,
            candidate_row=candidate,
            datasets=datasets,
        )

        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_frame = pd.DataFrame(all_metrics)

    predictions_frame = pd.DataFrame(all_predictions)

    summary = (
        metrics_frame
        .groupby(
            [
                "candidate_rank",
                "complexity",
                "split",
            ],
            as_index=False,
        )
        .agg(
            mean_relative_rmse=(
                "relative_rmse",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    predictions_frame.to_csv(
        TRAJECTORY_OUTPUT_PATH,
        index=False,
    )

    print()
    print("Trajectory evaluation:")
    print()

    print(
        summary[
            [
                "candidate_rank",
                "complexity",
                "split",
                "mean_relative_rmse",
                "mean_rmse",
                "mean_r2",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved summary: {OUTPUT_PATH}")
    print(
        f"Saved trajectories: {TRAJECTORY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()