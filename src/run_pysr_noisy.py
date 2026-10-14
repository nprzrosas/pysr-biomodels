"""Run PySR using derivatives estimated from noisy time-series observations."""

from pathlib import Path

import pandas as pd
import sympy as sp
from pysr import PySRRegressor

from .add_measurement_noise import (
    add_measurement_noise,
    calculate_training_state_scale,
)
from .estimate_derivatives import estimate_dataset_derivatives
from .simulate import MODEL_ID, build_experimental_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / MODEL_ID
    / "noisy_derivative_experiment"
)

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

NOISE_LEVELS = (
    0.005,
    0.01,
    0.02,
    0.05,
)

NOISE_SEED = 4040

SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 3


def noise_label(
    noise_fraction,
):
    """Convert a noise fraction to a stable directory label."""
    percent = (
        100.0
        * noise_fraction
    )

    return (
        f"noise_{percent:g}pct"
        .replace(
            ".",
            "p",
        )
    )


def build_training_matrices(
    train_frame,
):
    """Normalize observed states and estimated derivatives."""
    state_column = "VENUS_observed"
    target_column = (
        "d_VENUS_observed_dt_estimated"
    )

    state_scale = float(
        train_frame[
            state_column
        ].abs().max()
    )

    if state_scale <= 0:
        raise ValueError(
            "Observed training state scale must be positive."
        )

    x_frame = train_frame[
        [state_column]
    ].copy()

    y_frame = train_frame[
        [target_column]
    ].copy()

    x_frame[
        state_column
    ] = (
        x_frame[state_column]
        / state_scale
    )

    y_frame[
        target_column
    ] = (
        y_frame[target_column]
        / state_scale
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
        state_scale,
    )


def build_pysr_model(
    output_dir,
):
    """Construct PySR using the frozen inference configuration."""
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
            output_dir
            / "pysr_internal"
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
    scaled_state = sp.Symbol(
        "x0"
    )

    original_state = sp.Symbol(
        "VENUS"
    )

    scaled_expression = sp.sympify(
        str(expression)
    )

    original_expression = (
        state_scale
        * scaled_expression.subs(
            scaled_state,
            original_state
            / state_scale,
        )
    )

    return sp.simplify(
        original_expression
    )


def prepare_noisy_training_data(
    *,
    train_frame,
    noise_fraction,
):
    """Create noisy observations and estimate their derivatives."""
    clean_training_scale = (
        calculate_training_state_scale(
            train_frame
        )
    )

    noisy_frame = add_measurement_noise(
        train_frame,
        state_scale=clean_training_scale,
        noise_fraction=noise_fraction,
        seed=NOISE_SEED,
        state_column="VENUS",
        output_column="VENUS_observed",
    )

    estimated_frame = (
        estimate_dataset_derivatives(
            noisy_frame,
            state_column="VENUS_observed",
            window_length=SAVGOL_WINDOW_LENGTH,
            polyorder=SAVGOL_POLYORDER,
        )
    )

    return (
        estimated_frame,
        clean_training_scale,
    )


def run_training_experiment(
    *,
    dataset_name,
    train_frame,
    noise_fraction,
):
    """Fit PySR to derivatives estimated from noisy observations."""
    label = noise_label(
        noise_fraction
    )

    output_dir = (
        EXPERIMENT_ROOT
        / label
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        estimated_frame,
        clean_training_scale,
    ) = prepare_noisy_training_data(
        train_frame=train_frame,
        noise_fraction=noise_fraction,
    )

    (
        x_values,
        y_values,
        observed_state_scale,
    ) = build_training_matrices(
        estimated_frame
    )

    print()
    print("=" * 70)
    print(
        f"Noisy dataset: {dataset_name}"
    )
    print(
        "Noise fraction:",
        noise_fraction,
    )
    print(
        "Noise percent:",
        100.0 * noise_fraction,
    )
    print(
        "Trajectories:",
        estimated_frame[
            "trajectory_id"
        ].nunique(),
    )
    print(
        "Training rows:",
        len(estimated_frame),
    )
    print(
        "Feature:",
        "VENUS_observed",
    )
    print(
        "Derivative target:",
        "d_VENUS_observed_dt_estimated",
    )
    print(
        "Clean training scale used for noise:",
        clean_training_scale,
    )
    print(
        "Observed state scale used for PySR normalization:",
        observed_state_scale,
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
        variable_names=[
            "x0"
        ],
        X_units=None,
        y_units=None,
    )

    equations = model.equations_

    if isinstance(
        equations,
        list,
    ):
        equations = equations[
            0
        ]

    equations = equations.copy()

    equations[
        "equation_scaled"
    ] = (
        equations[
            "equation"
        ]
        .astype(str)
        .str.replace(
            r"\bx0\b",
            "VENUS",
            regex=True,
        )
    )

    equations[
        "equation_original_units"
    ] = [
        str(
            back_transform_equation(
                expression,
                observed_state_scale,
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
                "noise_fraction": noise_fraction,
                "noise_percent": (
                    100.0
                    * noise_fraction
                ),
                "noise_seed": NOISE_SEED,
                "noise_scale_definition": (
                    "max_abs_clean_training_state"
                ),
                "clean_training_state_scale": (
                    clean_training_scale
                ),
                "feature_column": (
                    "VENUS_observed"
                ),
                "derivative_method": (
                    "savitzky_golay"
                ),
                "derivative_column": (
                    "d_VENUS_observed_dt_estimated"
                ),
                "savgol_window_length": (
                    SAVGOL_WINDOW_LENGTH
                ),
                "savgol_polyorder": (
                    SAVGOL_POLYORDER
                ),
                "training_trajectories": (
                    estimated_frame[
                        "trajectory_id"
                    ].nunique()
                ),
                "training_rows": len(
                    estimated_frame
                ),
                "pysr_normalization_state_scale": (
                    observed_state_scale
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
        top_candidates[
            [
                "pysr_loss_rank",
                "complexity",
                "loss",
                "equation_original_units",
            ]
        ].to_string(
            index=False
        )
    )


def main():
    """Run all frozen noisy-training experiments."""
    (
        datasets,
        _,
        _,
        _,
    ) = build_experimental_dataset()

    for noise_fraction in NOISE_LEVELS:
        for dataset_name in TRAINING_DATASETS:
            run_training_experiment(
                dataset_name=dataset_name,
                train_frame=datasets[
                    dataset_name
                ],
                noise_fraction=noise_fraction,
            )


if __name__ == "__main__":
    main()