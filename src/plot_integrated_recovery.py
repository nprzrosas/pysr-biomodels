"""Plot frozen integrated dynamic and mechanistic recovery results.

This module is visualization-only. It reads the frozen integrated recovery
tables and their frozen reporting summary. It does not recompute inference,
selection, trajectory evaluation, or mechanistic recovery metrics.

Primary mechanistic figures use only models whose ground-truth RHS is
compatible with the states-only inference feature space.

Outputs
-------
results/integrated_recovery_figures/
    mechanistic_recovery_vs_training.pdf
    mechanistic_recovery_vs_training.png
    held_out_robustness_vs_training.pdf
    held_out_robustness_vs_training.png
    held_out_nrmse_vs_training.pdf
    held_out_nrmse_vs_training.png
    dynamic_vs_mechanistic_recovery.pdf
    dynamic_vs_mechanistic_recovery.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate_dependency_recovery import (
    RESULTS_ROOT,
    TRAIN_DATASETS,
)


INTEGRATED_PATH = (
    RESULTS_ROOT
    / "integrated_recovery.csv"
)

COMPATIBLE_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_compatible.csv"
)

SUMMARY_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_summary.csv"
)

OUTPUT_DIRECTORY = (
    RESULTS_ROOT
    / "integrated_recovery_figures"
)

EXPECTED_COMPATIBLE_MODELS = 8
EXPECTED_COMPATIBLE_CONDITIONS = 32

TRAINING_ORDER = [
    4,
    8,
    16,
    32,
]

DPI = 300


def require_file(
    path,
):
    """Require one frozen reporting input."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen result was not found: {path}"
        )


def require_columns(
    frame,
    required_columns,
    *,
    source_name,
):
    """Require columns needed by the plotting layer."""
    missing_columns = (
        set(
            required_columns
        )
        - set(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing_columns)}."
        )


def validate_inputs(
    integrated,
    compatible,
    summary,
):
    """Validate frozen plotting inputs."""
    require_columns(
        integrated,
        {
            "model_id",
            "train_dataset",
            "train_trajectories",
            "test_status",
            "test_macro_nrmse",
            "micro_dependency_f1",
            "macro_numerator_monomial_f1",
            "macro_interaction_f1",
            "rational_structure_accuracy",
            "mechanistic_comparability",
        },
        source_name="Integrated recovery",
    )

    require_columns(
        compatible,
        {
            "model_id",
            "train_dataset",
            "train_trajectories",
            "test_status",
            "test_macro_nrmse",
            "micro_dependency_f1",
            "macro_numerator_monomial_f1",
            "macro_interaction_f1",
            "rational_structure_accuracy",
            "mechanistic_comparability",
        },
        source_name="Compatible integrated recovery",
    )

    require_columns(
        summary,
        {
            "train_dataset",
            "train_trajectories",
            "n_models",
            "n_test_success",
            "n_test_failed",
            "test_success_rate",
            "median_test_macro_nrmse_successful",
            "micro_dependency_f1",
            "mean_model_numerator_monomial_f1",
            "mean_model_interaction_f1",
            "mean_model_rational_structure_accuracy",
            "micro_numerical_equivalence_fraction",
        },
        source_name="Integrated recovery summary",
    )

    if len(
        compatible
    ) != EXPECTED_COMPATIBLE_CONDITIONS:
        raise ValueError(
            "Compatible integrated recovery must contain "
            f"{EXPECTED_COMPATIBLE_CONDITIONS} conditions."
        )

    if compatible[
        "model_id"
    ].nunique() != EXPECTED_COMPATIBLE_MODELS:
        raise ValueError(
            "Compatible integrated recovery must contain "
            f"{EXPECTED_COMPATIBLE_MODELS} models."
        )

    observed_datasets = set(
        summary[
            "train_dataset"
        ].astype(
            str
        )
    )

    if observed_datasets != set(
        TRAIN_DATASETS
    ):
        raise ValueError(
            "Integrated recovery summary contains unexpected "
            "training conditions."
        )

    if len(
        summary
    ) != len(
        TRAIN_DATASETS
    ):
        raise ValueError(
            "Integrated recovery summary must contain exactly "
            "one row per training condition."
        )

    if not (
        compatible[
            "mechanistic_comparability"
        ]
        == "compatible"
    ).all():
        raise ValueError(
            "Compatible plotting input contains "
            "non-compatible conditions."
        )


def ordered_summary(
    summary,
):
    """Return summary rows ordered by number of training trajectories."""
    ordered = summary.copy()

    ordered[
        "train_trajectories"
    ] = pd.to_numeric(
        ordered[
            "train_trajectories"
        ],
        errors="raise",
    ).astype(
        int
    )

    observed_training = set(
        ordered[
            "train_trajectories"
        ]
    )

    if observed_training != set(
        TRAINING_ORDER
    ):
        raise ValueError(
            "Unexpected training trajectory counts. "
            f"Observed={sorted(observed_training)}, "
            f"expected={TRAINING_ORDER}."
        )

    return (
        ordered.sort_values(
            "train_trajectories"
        )
        .reset_index(
            drop=True
        )
    )


def save_figure(
    figure,
    *,
    stem,
):
    """Save one figure as PDF and high-resolution PNG."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / f"{stem}.pdf"
    )

    png_path = (
        OUTPUT_DIRECTORY
        / f"{stem}.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {pdf_path}"
    )

    print(
        f"Saved: {png_path}"
    )


def plot_mechanistic_recovery_vs_training(
    summary,
):
    """Plot mechanistic recovery metrics versus training trajectories."""
    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.5,
        )
    )

    x = summary[
        "train_trajectories"
    ].to_numpy()

    series = [
        (
            "Dependency F1",
            summary[
                "micro_dependency_f1"
            ].to_numpy(),
            "o",
        ),
        (
            "Numerator monomial F1",
            summary[
                "mean_model_numerator_monomial_f1"
            ].to_numpy(),
            "s",
        ),
        (
            "Interaction F1",
            summary[
                "mean_model_interaction_f1"
            ].to_numpy(),
            "^",
        ),
        (
            "Rational-structure accuracy",
            summary[
                "mean_model_rational_structure_accuracy"
            ].to_numpy(),
            "D",
        ),
        (
            "Numerical equivalence",
            summary[
                "micro_numerical_equivalence_fraction"
            ].to_numpy(),
            "v",
        ),
    ]

    for (
        label,
        values,
        marker,
    ) in series:
        axis.plot(
            x,
            values,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=label,
        )

    axis.set_xlabel(
        "Training trajectories"
    )

    axis.set_ylabel(
        "Recovery metric"
    )

    axis.set_title(
        "Mechanistic recovery across training-set sizes"
    )

    axis.set_xticks(
        TRAINING_ORDER
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
    )

    figure.tight_layout()

    save_figure(
        figure,
        stem="mechanistic_recovery_vs_training",
    )


def plot_held_out_robustness_vs_training(
    summary,
):
    """Plot held-out integration success rate versus training size."""
    figure, axis = plt.subplots(
        figsize=(
            7.0,
            5.0,
        )
    )

    x = summary[
        "train_trajectories"
    ].to_numpy()

    success_rate = (
        summary[
            "test_success_rate"
        ].to_numpy()
        * 100.0
    )

    axis.plot(
        x,
        success_rate,
        marker="o",
        linewidth=2,
        markersize=8,
    )

    for (
        x_value,
        rate,
        successes,
        models,
    ) in zip(
        x,
        success_rate,
        summary[
            "n_test_success"
        ],
        summary[
            "n_models"
        ],
    ):
        axis.annotate(
            f"{int(successes)}/{int(models)}",
            (
                x_value,
                rate,
            ),
            xytext=(
                0,
                9,
            ),
            textcoords="offset points",
            ha="center",
        )

    axis.set_xlabel(
        "Training trajectories"
    )

    axis.set_ylabel(
        "Held-out integration success (%)"
    )

    axis.set_title(
        "Held-out integration robustness"
    )

    axis.set_xticks(
        TRAINING_ORDER
    )

    axis.set_ylim(
        0.0,
        105.0,
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    save_figure(
        figure,
        stem="held_out_robustness_vs_training",
    )


def plot_held_out_nrmse_vs_training(
    summary,
):
    """Plot held-out NRMSE conditional on successful integration."""
    figure, axis = plt.subplots(
        figsize=(
            7.0,
            5.0,
        )
    )

    x = summary[
        "train_trajectories"
    ].to_numpy()

    y = summary[
        "median_test_macro_nrmse_successful"
    ].to_numpy()

    axis.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=8,
    )

    axis.set_xlabel(
        "Training trajectories"
    )

    axis.set_ylabel(
        "Median held-out macro NRMSE"
    )

    axis.set_title(
        "Held-out error among successfully integrated systems"
    )

    axis.set_xticks(
        TRAINING_ORDER
    )

    axis.grid(
        alpha=0.25
    )

    figure.text(
        0.5,
        0.01,
        (
            "NRMSE is conditional on successful integration of all "
            "held-out trajectories."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(
            0,
            0.04,
            1,
            1,
        )
    )

    save_figure(
        figure,
        stem="held_out_nrmse_vs_training",
    )


def plot_dynamic_vs_mechanistic_recovery(
    compatible,
):
    """Plot held-out error against dependency recovery for successful systems."""
    successful = compatible[
        compatible[
            "test_status"
        ]
        == "success"
    ].copy()

    successful[
        "test_macro_nrmse"
    ] = pd.to_numeric(
        successful[
            "test_macro_nrmse"
        ],
        errors="coerce",
    )

    successful[
        "micro_dependency_f1"
    ] = pd.to_numeric(
        successful[
            "micro_dependency_f1"
        ],
        errors="coerce",
    )

    successful[
        "train_trajectories"
    ] = pd.to_numeric(
        successful[
            "train_trajectories"
        ],
        errors="raise",
    ).astype(
        int
    )

    finite = (
        np.isfinite(
            successful[
                "test_macro_nrmse"
            ]
        )
        & np.isfinite(
            successful[
                "micro_dependency_f1"
            ]
        )
        & (
            successful[
                "test_macro_nrmse"
            ]
            > 0
        )
    )

    successful = successful[
        finite
    ].copy()

    if successful.empty:
        raise ValueError(
            "No finite successful held-out conditions are "
            "available for the dynamic-versus-mechanistic plot."
        )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.8,
        )
    )

    marker_by_training = {
        4: "o",
        8: "s",
        16: "^",
        32: "D",
    }

    for train_trajectories in TRAINING_ORDER:
        subset = successful[
            successful[
                "train_trajectories"
            ]
            == train_trajectories
        ]

        if subset.empty:
            continue

        axis.scatter(
            subset[
                "micro_dependency_f1"
            ],
            subset[
                "test_macro_nrmse"
            ],
            marker=marker_by_training[
                train_trajectories
            ],
            s=65,
            label=(
                f"{train_trajectories} training trajectories"
            ),
        )

    axis.set_xlabel(
        "Dependency recovery F1"
    )

    axis.set_ylabel(
        "Held-out macro NRMSE"
    )

    axis.set_title(
        "Dynamic generalization versus dependency recovery"
    )

    axis.set_yscale(
        "log"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        frameon=False,
        loc="best",
    )

    figure.text(
        0.5,
        0.01,
        (
            "Only systems that successfully integrated all held-out "
            "trajectories are shown."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(
            0,
            0.04,
            1,
            1,
        )
    )

    save_figure(
        figure,
        stem="dynamic_vs_mechanistic_recovery",
    )


def main():
    """Generate figures from frozen integrated recovery results."""
    require_file(
        INTEGRATED_PATH
    )

    require_file(
        COMPATIBLE_PATH
    )

    require_file(
        SUMMARY_PATH
    )

    integrated = pd.read_csv(
        INTEGRATED_PATH
    )

    compatible = pd.read_csv(
        COMPATIBLE_PATH
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    validate_inputs(
        integrated,
        compatible,
        summary,
    )

    summary = ordered_summary(
        summary
    )

    print()
    print("=" * 80)
    print(
        "PLOTTING FROZEN INTEGRATED RECOVERY RESULTS"
    )
    print("=" * 80)

    print(
        "Compatible models:",
        compatible[
            "model_id"
        ].nunique(),
    )

    print(
        "Compatible conditions:",
        len(
            compatible
        ),
    )

    print(
        "Output directory:",
        OUTPUT_DIRECTORY,
    )

    print()

    plot_mechanistic_recovery_vs_training(
        summary
    )

    plot_held_out_robustness_vs_training(
        summary
    )

    plot_held_out_nrmse_vs_training(
        summary
    )

    plot_dynamic_vs_mechanistic_recovery(
        compatible
    )

    print()
    print(
        "Integrated recovery plotting complete."
    )


if __name__ == "__main__":
    main()