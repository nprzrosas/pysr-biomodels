"""Plot global dynamic benchmark results.

This script is reporting-only. It reads the frozen benchmark summary and
does not rerun PySR, simulation, model selection, or evaluation.

Input
-----
results/benchmark_summary.csv

Outputs
-------
results/benchmark_analysis/figures/
    validation_nrmse_by_training_size.png
    test_nrmse_by_training_size.png
    test_success_by_training_size.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmark_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmark_analysis"
    / "figures"
)

TRAINING_SIZES = [
    4,
    8,
    16,
    32,
]


def load_data() -> pd.DataFrame:
    """Load and validate the frozen benchmark summary."""

    frame = pd.read_csv(INPUT_PATH)

    required = {
        "model_id",
        "training_trajectories",
        "validation_macro_nrmse",
        "test_macro_nrmse",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if frame.duplicated(
        ["model_id", "training_trajectories"]
    ).any():
        raise ValueError(
            "Duplicate model/training-size rows detected."
        )

    return frame


def short_model_id(model_id: str) -> str:
    """Return compact BioModels identifier for plot labels."""

    return model_id.replace(
        "BIOMD000000",
        "",
    )


def plot_validation(
    frame: pd.DataFrame,
) -> None:
    """Plot validation NRMSE learning curves."""

    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    for model_id, group in frame.groupby(
        "model_id",
        sort=True,
    ):
        group = group.sort_values(
            "training_trajectories"
        )

        x = group[
            "training_trajectories"
        ].to_numpy()

        y = pd.to_numeric(
            group["validation_macro_nrmse"],
            errors="coerce",
        ).to_numpy()

        finite = np.isfinite(y)

        ax.plot(
            x[finite],
            y[finite],
            marker="o",
            linewidth=1.2,
            alpha=0.55,
            label=short_model_id(model_id),
        )

    medians = []

    for training_size in TRAINING_SIZES:
        values = pd.to_numeric(
            frame.loc[
                frame["training_trajectories"]
                == training_size,
                "validation_macro_nrmse",
            ],
            errors="coerce",
        )

        values = values[
            np.isfinite(values)
        ]

        medians.append(
            values.median()
        )

    ax.plot(
        TRAINING_SIZES,
        medians,
        marker="o",
        linewidth=3,
        label="Median",
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_yscale(
        "log",
    )

    ax.set_xticks(
        TRAINING_SIZES,
        labels=[
            str(value)
            for value in TRAINING_SIZES
        ],
    )

    ax.set_xlabel(
        "Training trajectories"
    )

    ax.set_ylabel(
        "Validation macro NRMSE"
    )

    ax.set_title(
        "Validation performance vs. training trajectories"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "validation_nrmse_by_training_size.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_test(
    frame: pd.DataFrame,
) -> None:
    """Plot finite held-out test NRMSE and mark integration failures."""

    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    all_test_values = pd.to_numeric(
        frame["test_macro_nrmse"],
        errors="coerce",
    )

    finite_positive_values = all_test_values[
        np.isfinite(all_test_values)
        & (all_test_values > 0)
    ]

    if finite_positive_values.empty:
        raise ValueError(
            "No finite positive held-out test values."
        )

    failure_y = (
        finite_positive_values.max()
        * 2.0
    )

    for model_id, group in frame.groupby(
        "model_id",
        sort=True,
    ):
        group = group.sort_values(
            "training_trajectories"
        )

        x = group[
            "training_trajectories"
        ].to_numpy()

        y = pd.to_numeric(
            group["test_macro_nrmse"],
            errors="coerce",
        ).to_numpy()

        finite = np.isfinite(y)
        failed = ~finite

        ax.plot(
            x,
            np.where(
                finite,
                y,
                np.nan,
            ),
            marker="o",
            linewidth=1.2,
            alpha=0.55,
            label=short_model_id(model_id),
        )

        if failed.any():
            ax.scatter(
                x[failed],
                np.full(
                    failed.sum(),
                    failure_y,
                ),
                marker="x",
                s=70,
                linewidths=2,
            )

    ax.axhline(
        failure_y,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.text(
        TRAINING_SIZES[0],
        failure_y * 1.08,
        "Integration failure",
        fontsize=9,
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_yscale(
        "log",
    )

    ax.set_xticks(
        TRAINING_SIZES,
        labels=[
            str(value)
            for value in TRAINING_SIZES
        ],
    )

    ax.set_xlabel(
        "Training trajectories"
    )

    ax.set_ylabel(
        "Held-out test macro NRMSE"
    )

    ax.set_title(
        "Held-out performance vs. training trajectories"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "test_nrmse_by_training_size.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_test_success(
    frame: pd.DataFrame,
) -> None:
    """Plot held-out integration success rate."""

    success_rates = []
    success_counts = []

    for training_size in TRAINING_SIZES:
        values = pd.to_numeric(
            frame.loc[
                frame["training_trajectories"]
                == training_size,
                "test_macro_nrmse",
            ],
            errors="coerce",
        )

        finite = np.isfinite(values)

        success_rates.append(
            finite.mean()
        )

        success_counts.append(
            int(finite.sum())
        )

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.plot(
        TRAINING_SIZES,
        success_rates,
        marker="o",
        linewidth=2,
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xticks(
        TRAINING_SIZES,
        labels=[
            str(value)
            for value in TRAINING_SIZES
        ],
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Training trajectories"
    )

    ax.set_ylabel(
        "Held-out integration success fraction"
    )

    ax.set_title(
        "Held-out integration robustness"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    for x, value, count in zip(
        TRAINING_SIZES,
        success_rates,
        success_counts,
    ):
        ax.annotate(
            f"{count}/10",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "test_success_by_training_size.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main() -> None:
    """Generate global dynamic benchmark figures."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_data()

    plot_validation(
        frame
    )

    plot_test(
        frame
    )

    plot_test_success(
        frame
    )

    print(
        "Generated benchmark figures:"
    )

    for filename in (
        "validation_nrmse_by_training_size.png",
        "test_nrmse_by_training_size.png",
        "test_success_by_training_size.png",
    ):
        print(
            OUTPUT_DIR / filename
        )

    print()
    print(
        "No scientific computation was rerun."
    )


if __name__ == "__main__":
    main()