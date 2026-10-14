"""Plot mechanistic-recovery learning curves for the PySR benchmark.

This script is reporting-only. It reads the frozen benchmark summary and
does not rerun PySR, simulation, model selection, mechanistic evaluation,
or any other scientific computation.

Only systems with:

    mechanistic_comparability == "compatible"

are included in the primary mechanistic analysis.

Input
-----
results/benchmark_summary.csv

Outputs
-------
results/benchmark_analysis/figures/
    mechanistic_dependency_f1.png
    mechanistic_monomial_f1.png
    mechanistic_interaction_f1.png
    mechanistic_rational_accuracy.png
    mechanistic_denominator_f1.png
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

EXPECTED_COMPATIBLE_MODELS = 8
EXPECTED_COMPATIBLE_ROWS = 32

METRICS = {
    "macro_dependency_f1": {
        "ylabel": "Macro dependency F1",
        "title": "Dependency recovery vs. training trajectories",
        "filename": "mechanistic_dependency_f1.png",
    },
    "macro_numerator_monomial_f1": {
        "ylabel": "Macro numerator monomial F1",
        "title": "Numerator monomial recovery vs. training trajectories",
        "filename": "mechanistic_monomial_f1.png",
    },
    "macro_interaction_f1": {
        "ylabel": "Macro interaction F1",
        "title": "Interaction recovery vs. training trajectories",
        "filename": "mechanistic_interaction_f1.png",
    },
    "rational_structure_accuracy": {
        "ylabel": "Rational-structure accuracy",
        "title": "Rational-structure recovery vs. training trajectories",
        "filename": "mechanistic_rational_accuracy.png",
    },
    "macro_denominator_dependency_f1": {
        "ylabel": "Macro denominator dependency F1",
        "title": "Denominator recovery vs. training trajectories",
        "filename": "mechanistic_denominator_f1.png",
    },
}


def load_compatible_data() -> pd.DataFrame:
    """Load and validate the mechanistically comparable benchmark cohort."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Benchmark summary not found: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH
    )

    required = {
        "model_id",
        "training_trajectories",
        "mechanistic_comparability",
        *METRICS.keys(),
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    compatible = frame.loc[
        frame["mechanistic_comparability"]
        == "compatible"
    ].copy()

    n_models = compatible[
        "model_id"
    ].nunique()

    if n_models != EXPECTED_COMPATIBLE_MODELS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_MODELS} "
            "mechanistically compatible models, "
            f"found {n_models}."
        )

    if len(compatible) != EXPECTED_COMPATIBLE_ROWS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_ROWS} "
            "compatible model/training rows, "
            f"found {len(compatible)}."
        )

    if compatible.duplicated(
        [
            "model_id",
            "training_trajectories",
        ]
    ).any():
        raise ValueError(
            "Duplicate model/training-size rows "
            "detected in compatible cohort."
        )

    observed_sizes = set(
        compatible[
            "training_trajectories"
        ]
    )

    if observed_sizes != set(TRAINING_SIZES):
        raise ValueError(
            "Unexpected training sizes. "
            f"Expected {TRAINING_SIZES}, "
            f"found {sorted(observed_sizes)}."
        )

    counts = (
        compatible[
            "training_trajectories"
        ]
        .value_counts()
        .to_dict()
    )

    for training_size in TRAINING_SIZES:
        count = counts.get(
            training_size,
            0,
        )

        if count != EXPECTED_COMPATIBLE_MODELS:
            raise ValueError(
                f"Expected "
                f"{EXPECTED_COMPATIBLE_MODELS} "
                f"rows for training size "
                f"{training_size}, "
                f"found {count}."
            )

    for metric in METRICS:
        values = pd.to_numeric(
            compatible[metric],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Missing or nonnumeric values "
                f"detected for {metric}."
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Nonfinite values detected "
                f"for {metric}."
            )

        outside_range = (
            (values < 0)
            | (values > 1)
        )

        if outside_range.any():
            raise ValueError(
                f"Values outside [0, 1] "
                f"detected for {metric}."
            )

    return compatible


def short_model_id(
    model_id: str,
) -> str:
    """Return a compact BioModels identifier for plot labels."""

    return model_id.replace(
        "BIOMD000000",
        "",
    )


def calculate_medians(
    frame: pd.DataFrame,
    metric: str,
) -> list[float]:
    """Calculate the cohort median at each training size."""

    medians = []

    for training_size in TRAINING_SIZES:
        values = pd.to_numeric(
            frame.loc[
                frame["training_trajectories"]
                == training_size,
                metric,
            ],
            errors="coerce",
        )

        if len(values) != EXPECTED_COMPATIBLE_MODELS:
            raise ValueError(
                f"Expected "
                f"{EXPECTED_COMPATIBLE_MODELS} "
                f"values for {metric} at "
                f"train={training_size}, "
                f"found {len(values)}."
            )

        medians.append(
            float(values.median())
        )

    return medians


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot one mechanistic-recovery metric across training sizes."""

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
            group[metric],
            errors="coerce",
        ).to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.2,
            alpha=0.55,
            label=short_model_id(
                model_id
            ),
        )

    medians = calculate_medians(
        frame,
        metric,
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

    ax.set_xticks(
        TRAINING_SIZES,
        labels=[
            str(value)
            for value in TRAINING_SIZES
        ],
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.set_xlabel(
        "Training trajectories"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
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
        / filename
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main() -> None:
    """Generate mechanistic-recovery benchmark figures."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_compatible_data()

    print(
        "Mechanistically compatible cohort:"
    )
    print(
        f"  models: "
        f"{frame['model_id'].nunique()}"
    )
    print(
        f"  rows: {len(frame)}"
    )

    print()
    print(
        "Excluded feature-space-mismatch models:"
    )

    full_frame = pd.read_csv(
        INPUT_PATH
    )

    excluded_models = sorted(
        full_frame.loc[
            full_frame[
                "mechanistic_comparability"
            ]
            != "compatible",
            "model_id",
        ].unique()
    )

    for model_id in excluded_models:
        print(
            "  "
            + short_model_id(
                model_id
            )
        )

    print()
    print(
        "Generating mechanistic figures:"
    )

    for metric, settings in METRICS.items():
        plot_metric(
            frame=frame,
            metric=metric,
            ylabel=settings["ylabel"],
            title=settings["title"],
            filename=settings["filename"],
        )

        print(
            "  "
            + str(
                OUTPUT_DIR
                / settings["filename"]
            )
        )

    print()
    print(
        "No scientific computation was rerun."
    )


if __name__ == "__main__":
    main()
