#!/usr/bin/env python3
"""Generate the audited figures for the GSoC 2026 final report."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "docs" / "figures"
SUMMARY_PATH = REPOSITORY_ROOT / "analysis" / "expanded100_final_scientific_summary.csv"
CASE_STUDY_PATH = REPOSITORY_ROOT / "analysis" / "expanded100_case_study_candidates.csv"
TEST_PREDICTIONS_TRAIN32_PATH = (
    REPOSITORY_ROOT / "analysis" / "test_predictions_train32.csv"
)
TEST_TRAJECTORY_METRICS_TRAIN32_PATH = (
    REPOSITORY_ROOT / "analysis" / "test_trajectory_metrics_train32.csv"
)
MECHANISTIC_RECOVERY_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "expanded100_mechanistic_recovery"
    / "mechanistic_recovery.csv"
)
BIOMD_0156_TEST_COMPARISON_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "expanded100_test_comparisons"
    / "BIOMD0000000156"
    / "generic_exact_experiment"
    / "coordinate_selection"
    / "test"
    / "test_comparison.csv"
)
FUNNEL_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "model_selection_audit"
    / "biomodels_selection_funnel.csv"
)

PNG_DPI = 180
FONT_FAMILY = "DejaVu Sans"

NAVY = "#17324D"
BLUE = "#2F6B9A"
LIGHT_BLUE = "#DCEAF5"
TEAL = "#1B8A89"
LIGHT_TEAL = "#D9EFEC"
GREEN = "#4C956C"
LIGHT_GREEN = "#E1F0E6"
ORANGE = "#D17A22"
LIGHT_ORANGE = "#F8E6D2"
RED = "#B64E4E"
LIGHT_RED = "#F5DEDE"
PURPLE = "#7656A3"
LIGHT_PURPLE = "#E9E1F3"
GRAY = "#5E6973"
LIGHT_GRAY = "#EEF1F3"
WHITE = "#FFFFFF"

HELDOUT_EXAMPLE_MODEL_ID = "BIOMD0000000156"
HELDOUT_EXAMPLE_DATASET = "train_32"
HELDOUT_EXAMPLE_TRAJECTORY_ID = "test_003"
HELDOUT_EXAMPLE_STATES = ("x", "y", "y0")


def configure_style() -> None:
    """Configure shared, publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB3BA",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9DEE2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def save_figure(figure: Figure, output_directory: Path, stem: str) -> None:
    """Save one figure as SVG and high-resolution PNG.

    Args:
        figure: Matplotlib figure to save.
        output_directory: Destination directory.
        stem: Filename stem without extension.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_directory / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Creator": "pysr_only GSoC report figure generator"},
    )
    figure.savefig(
        output_directory / f"{stem}.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        metadata={"Creator": "pysr_only GSoC report figure generator"},
    )
    plt.close(figure)


def add_box(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = NAVY,
    fontsize: float = 9.5,
    linewidth: float = 1.2,
) -> None:
    """Add a rounded labeled box in axes-fraction coordinates.

    Args:
        axes: Target axes with a 0--1 coordinate system.
        x: Left position.
        y: Bottom position.
        width: Box width.
        height: Box height.
        text: Label text.
        facecolor: Box fill color.
        edgecolor: Box edge color.
        fontsize: Label font size.
        linewidth: Box edge width.
    """
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axes.add_patch(patch)
    axes.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        linespacing=1.2,
    )


def add_arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    style: str = "-",
    width: float = 1.4,
) -> None:
    """Add a directional arrow between two axes-fraction positions.

    Args:
        axes: Target axes.
        start: Arrow start coordinate.
        end: Arrow end coordinate.
        color: Arrow color.
        style: Line style.
        width: Line width.
    """
    axes.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": width,
            "linestyle": style,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def prepare_diagram_axes(figure: Figure) -> Axes:
    """Create a normalized axes for a box-and-arrow diagram.

    Args:
        figure: Parent figure.

    Returns:
        Axes configured with a 0--1 coordinate range and no frame.
    """
    axes = figure.add_subplot(111)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")
    return axes


def generate_corpus_figure(output_directory: Path) -> None:
    """Generate the historical screening and cohort-expansion diagram.

    Args:
        output_directory: Destination directory.
    """
    funnel = pd.read_csv(FUNNEL_PATH)
    expected = [1075, 497, 326, 303, 66, 43]
    observed = funnel["models_retained"].astype(int).tolist()
    if observed != expected:
        raise ValueError(f"Unexpected historical funnel counts: {observed}")

    figure = plt.figure(figsize=(12, 7))
    axes = prepare_diagram_axes(figure)
    axes.set_title(
        "Historical corpus screening and filename-aware expansion",
        color=NAVY,
        fontweight="bold",
        pad=12,
    )

    axes.text(
        0.23,
        0.92,
        "HISTORICAL SCREENING SEQUENCE",
        ha="center",
        color=GRAY,
        fontweight="bold",
        fontsize=10,
    )
    labels = [
        "1,075  Local curated model directories",
        "497  Passed no-variable-parameter criterion",
        "326  Found by canonical-filename resolver",
        "303  Passed load and preflight",
        "66  Passed historical compatibility stage",
        "43  Had required simulation-time metadata",
    ]
    y_positions = [0.81, 0.69, 0.57, 0.45, 0.33, 0.21]
    for index, (label, y_position) in enumerate(zip(labels, y_positions)):
        add_box(axes, 0.04, y_position, 0.38, 0.075, label, LIGHT_GRAY, fontsize=9)
        if index:
            add_arrow(
                axes,
                (0.23, y_positions[index - 1]),
                (0.23, y_position + 0.075),
            )

    axes.text(
        0.72,
        0.92,
        "FINAL EXPANDED COHORT",
        ha="center",
        color=GRAY,
        fontweight="bold",
        fontsize=10,
    )
    add_box(
        axes,
        0.53,
        0.67,
        0.38,
        0.11,
        "43 previously benchmark-ready models",
        LIGHT_BLUE,
        edgecolor=BLUE,
        fontsize=10,
    )
    add_box(
        axes,
        0.53,
        0.48,
        0.38,
        0.13,
        "57 additional models admitted after\nrobust SBML filename resolution",
        LIGHT_ORANGE,
        edgecolor=ORANGE,
        fontsize=10,
    )
    axes.text(0.72, 0.63, "+", ha="center", va="center", fontsize=19, color=NAVY)
    add_box(
        axes,
        0.57,
        0.23,
        0.30,
        0.13,
        "100 expanded\nbenchmark-ready models",
        LIGHT_GREEN,
        edgecolor=GREEN,
        fontsize=12,
        linewidth=1.8,
    )
    add_arrow(axes, (0.72, 0.48), (0.72, 0.36), color=GREEN, width=1.8)
    add_arrow(axes, (0.72, 0.67), (0.72, 0.61), color=BLUE)
    axes.text(
        0.5,
        0.08,
        "The left sequence documents historical screening. It is not a fully reconstructed,\n"
        "mutually exclusive post-refactor funnel and does not assess scientific validity.",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=9.5,
        style="italic",
    )
    save_figure(figure, output_directory, "01_corpus_screening_and_expansion")


def generate_pipeline_figure(output_directory: Path) -> None:
    """Generate the refactored training, selection, and evaluation workflow.

    Args:
        output_directory: Destination directory.
    """
    figure = plt.figure(figsize=(14, 7))
    axes = prepare_diagram_axes(figure)
    axes.set_title(
        "Refactored PySR benchmark pipeline and data boundaries",
        color=NAVY,
        fontweight="bold",
        pad=12,
    )

    labels = [
        "SBML\nmodel",
        "Standardized\ninitial-condition\ndesign",
        "Simulation +\nexact simulator\nderivatives",
        "Nested training\nsets\n4 / 8 / 16 / 32",
        "PySR per-RHS\ncandidate\ninference",
        "Coupled-system\nselection",
        "Freeze selected\nsystem",
        "Held-out\ntesting",
    ]
    x_positions = [0.015, 0.14, 0.27, 0.40, 0.53, 0.66, 0.79, 0.90]
    widths = [0.09, 0.105, 0.105, 0.105, 0.105, 0.105, 0.09, 0.085]
    fills = [
        LIGHT_GRAY,
        LIGHT_BLUE,
        LIGHT_BLUE,
        LIGHT_TEAL,
        LIGHT_ORANGE,
        LIGHT_PURPLE,
        LIGHT_GREEN,
        LIGHT_GREEN,
    ]
    edges = [GRAY, BLUE, BLUE, TEAL, ORANGE, PURPLE, GREEN, GREEN]
    for index, (label, x_position, width, fill, edge) in enumerate(
        zip(labels, x_positions, widths, fills, edges)
    ):
        add_box(
            axes,
            x_position,
            0.59,
            width,
            0.18,
            label,
            fill,
            edgecolor=edge,
            fontsize=8.5,
        )
        if index:
            add_arrow(
                axes,
                (x_positions[index - 1] + widths[index - 1], 0.68),
                (x_position, 0.68),
            )

    add_box(
        axes,
        0.64,
        0.31,
        0.15,
        0.10,
        "8 validation trajectories",
        LIGHT_PURPLE,
        edgecolor=PURPLE,
        fontsize=9,
    )
    add_arrow(axes, (0.715, 0.41), (0.715, 0.59), color=PURPLE, style="--")
    axes.text(0.715, 0.44, "used for selection", ha="center", color=PURPLE, fontsize=8)

    add_box(
        axes,
        0.83,
        0.31,
        0.14,
        0.10,
        "8 held-out trajectories",
        LIGHT_GREEN,
        edgecolor=GREEN,
        fontsize=9,
    )
    add_arrow(axes, (0.90, 0.41), (0.94, 0.59), color=GREEN, style="--")
    axes.text(
        0.90, 0.27, "never used for selection", ha="center", color=GREEN, fontsize=8
    )

    add_box(
        axes,
        0.56,
        0.08,
        0.19,
        0.11,
        "Symbolic ground-truth RHSs",
        LIGHT_RED,
        edgecolor=RED,
        fontsize=9,
    )
    add_box(
        axes,
        0.79,
        0.07,
        0.18,
        0.13,
        "Post-hoc mechanistic\nrecovery analysis",
        LIGHT_RED,
        edgecolor=RED,
        fontsize=9.5,
    )
    add_arrow(axes, (0.75, 0.135), (0.79, 0.135), color=RED, style="--")
    add_arrow(axes, (0.835, 0.59), (0.88, 0.20), color=RED, style="--")
    axes.text(
        0.83,
        0.49,
        "selected inferred\nsystem",
        ha="center",
        va="center",
        color=RED,
        fontsize=7.8,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    axes.text(
        0.66,
        0.035,
        "Ground truth is not used for candidate inference or selection",
        ha="center",
        color=RED,
        fontsize=8.5,
        fontweight="bold",
    )
    axes.text(
        0.49,
        0.88,
        "Development and selection",
        ha="center",
        color=GRAY,
        fontsize=10,
        fontweight="bold",
    )
    axes.text(
        0.91,
        0.88,
        "Evaluation only",
        ha="center",
        color=GREEN,
        fontsize=10,
        fontweight="bold",
    )
    axes.plot([0.875, 0.875], [0.51, 0.85], color=GREEN, linewidth=1.2, linestyle=":")
    save_figure(figure, output_directory, "02_refactored_pipeline")


def generate_execution_figure(output_directory: Path) -> None:
    """Generate the expanded-100 execution-outcome hierarchy.

    Args:
        output_directory: Destination directory.
    """
    figure = plt.figure(figsize=(11, 6.5))
    axes = prepare_diagram_axes(figure)
    axes.set_title(
        "Expanded-100 execution outcomes",
        color=NAVY,
        fontweight="bold",
        pad=12,
    )

    add_box(
        axes,
        0.35,
        0.78,
        0.30,
        0.12,
        "100 benchmark-ready models",
        LIGHT_BLUE,
        BLUE,
        fontsize=13,
        linewidth=1.8,
    )
    add_box(
        axes,
        0.12,
        0.52,
        0.30,
        0.13,
        "75 workflow-complete\nthrough held-out testing",
        LIGHT_GREEN,
        GREEN,
        fontsize=12,
        linewidth=1.7,
    )
    add_box(
        axes,
        0.58,
        0.52,
        0.30,
        0.13,
        "25 incomplete",
        LIGHT_ORANGE,
        ORANGE,
        fontsize=12,
        linewidth=1.7,
    )
    add_arrow(axes, (0.43, 0.78), (0.29, 0.65), color=GREEN, width=1.7)
    add_arrow(axes, (0.57, 0.78), (0.73, 0.65), color=ORANGE, width=1.7)

    child_specs = [
        (0.44, "12 timeout-censored", LIGHT_ORANGE, ORANGE),
        (0.66, "2 memory-censored", LIGHT_RED, RED),
        (0.87, "11 unresolved candidate-stage\nworker exits", LIGHT_GRAY, GRAY),
    ]
    widths = [0.18, 0.14, 0.20]
    for (x_position, label, fill, edge), width in zip(child_specs, widths):
        add_box(
            axes,
            x_position - width / 2,
            0.22,
            width,
            0.13,
            label,
            fill,
            edge,
            fontsize=8.8,
        )
        add_arrow(axes, (0.73, 0.52), (x_position, 0.35), color=edge)

    axes.text(
        0.5,
        0.08,
        "Incomplete runs are execution outcomes, not a single scientific failure category.",
        ha="center",
        color=GRAY,
        fontsize=10,
        style="italic",
    )
    save_figure(figure, output_directory, "03_expanded100_execution_outcomes")


def generate_performance_figure(output_directory: Path) -> None:
    """Generate predictive and mechanistic training-size panels.

    Args:
        output_directory: Destination directory.
    """
    summary = pd.read_csv(SUMMARY_PATH).sort_values("training_trajectories")
    training_sizes = summary["training_trajectories"].to_numpy(dtype=int)

    figure = plt.figure(figsize=(12, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.34, wspace=0.24)
    error_axes = figure.add_subplot(grid[0, 0])
    success_axes = figure.add_subplot(grid[0, 1])
    structure_axes = figure.add_subplot(grid[1, :])
    figure.suptitle(
        "Predictive and mechanistic performance by training-set size",
        color=NAVY,
        fontweight="bold",
        fontsize=15,
        y=0.98,
    )

    medians = summary["median_test_nrmse"].to_numpy(dtype=float)
    error_axes.plot(
        training_sizes, medians, color=BLUE, marker="o", linewidth=2.2, markersize=7
    )
    for x_value, y_value in zip(training_sizes, medians):
        error_axes.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=NAVY,
        )
    error_axes.set_title("A. Median held-out macro-NRMSE")
    error_axes.set_xlabel("Training trajectories")
    error_axes.set_ylabel("Median macro-NRMSE")
    error_axes.set_xticks(training_sizes)
    error_axes.set_ylim(0, max(medians) * 1.28)
    error_axes.text(
        0.02,
        0.94,
        "Full held-out integration systems only",
        transform=error_axes.transAxes,
        va="top",
        color=GRAY,
        fontsize=9,
    )

    successes = summary["full_test_success"].to_numpy(dtype=int)
    completed = summary["completed_models"].to_numpy(dtype=int)
    proportions = successes / completed
    success_axes.bar(
        training_sizes,
        proportions,
        width=3.0,
        color=TEAL,
        edgecolor=NAVY,
        linewidth=0.8,
    )
    for x_value, proportion, numerator, denominator in zip(
        training_sizes, proportions, successes, completed
    ):
        success_axes.text(
            x_value,
            proportion + 0.025,
            f"{numerator}/{denominator}",
            ha="center",
            color=NAVY,
            fontweight="bold",
        )
    success_axes.set_title("B. Full held-out integration success")
    success_axes.set_xlabel("Training trajectories")
    success_axes.set_ylabel("Proportion of models")
    success_axes.set_xticks(training_sizes)
    success_axes.set_ylim(0, 1.08)
    success_axes.axhline(1, color=GRAY, linewidth=0.8, linestyle=":")
    success_axes.text(
        0.02,
        0.07,
        "n = 75 workflow-complete models",
        transform=success_axes.transAxes,
        color=GRAY,
        fontsize=9,
    )

    metric_specs = [
        ("mean_dependency_f1", "Dependency F1", BLUE, "o"),
        ("mean_monomial_f1", "Numerator monomial F1", ORANGE, "s"),
        ("mean_interaction_f1", "Interaction F1", TEAL, "^"),
        ("rational_structure_accuracy", "Rational-structure accuracy", PURPLE, "D"),
    ]
    for column, label, color, marker in metric_specs:
        structure_axes.plot(
            training_sizes,
            summary[column],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
    structure_axes.set_title("C. Model-level mechanistic metrics")
    structure_axes.set_xlabel("Training trajectories")
    structure_axes.set_ylabel("Mean model-level score")
    structure_axes.set_xticks(training_sizes)
    structure_axes.set_ylim(0.45, 0.90)
    structure_axes.legend(ncol=2, loc="lower center")
    structure_axes.text(
        0.02,
        0.95,
        "n = 66 compatible models",
        transform=structure_axes.transAxes,
        va="top",
        color=GRAY,
        fontsize=9,
    )
    save_figure(figure, output_directory, "04_predictive_and_mechanistic_metrics")


def generate_case_study_figure(output_directory: Path) -> None:
    """Generate three illustrative predictive/structural case-study panels.

    Args:
        output_directory: Destination directory.
    """
    cases = pd.read_csv(CASE_STUDY_PATH)
    model_ids = ["BIOMD0000000156", "BIOMD0000000319", "BIOMD0000000728"]
    selected = cases[cases["model_id"].isin(model_ids)].copy()
    selected["training_trajectories"] = (
        selected["dataset"].str.removeprefix("train_").astype(int)
    )
    if len(selected) != 12:
        raise ValueError(f"Expected 12 case-study rows, found {len(selected)}")

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col")
    figure.suptitle(
        "Illustrative case studies: prediction and structural components",
        color=NAVY,
        fontweight="bold",
        fontsize=15,
        y=0.99,
    )
    metric_specs = [
        ("macro_dependency_f1", "Dependency F1", BLUE, "o"),
        ("macro_numerator_monomial_f1", "Monomial F1", ORANGE, "s"),
        ("macro_interaction_f1", "Interaction F1", TEAL, "^"),
        ("rational_structure_accuracy", "Rational accuracy", PURPLE, "D"),
    ]
    for column_index, model_id in enumerate(model_ids):
        model = selected[selected["model_id"] == model_id].sort_values(
            "training_trajectories"
        )
        training_sizes = model["training_trajectories"].to_numpy(dtype=int)
        nrmse = model["test_macro_nrmse"].to_numpy(dtype=float)

        prediction_axes = axes[0, column_index]
        prediction_axes.plot(
            training_sizes, nrmse, color=GREEN, marker="o", linewidth=2.2, markersize=7
        )
        prediction_axes.set_yscale("log")
        prediction_axes.set_ylim(1e-7, 1e4)
        prediction_axes.set_title(model_id, fontweight="bold", color=NAVY)
        prediction_axes.set_xticks(training_sizes)
        prediction_axes.text(
            0.5,
            -0.06,
            "All 8 test trajectories integrated",
            transform=prediction_axes.transAxes,
            va="top",
            ha="center",
            color=GREEN,
            fontsize=8,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        if column_index == 0:
            prediction_axes.set_ylabel("Held-out macro-NRMSE\n(log scale)")

        structure_axes = axes[1, column_index]
        for column, label, color, marker in metric_specs:
            structure_axes.plot(
                training_sizes,
                model[column],
                label=label,
                color=color,
                marker=marker,
                linewidth=1.7,
                markersize=5,
            )
        structure_axes.set_ylim(-0.04, 1.08)
        structure_axes.set_xticks(training_sizes)
        structure_axes.set_xlabel("Training trajectories")
        if column_index == 0:
            structure_axes.set_ylabel("Model-level score")

        train_32 = model[model["training_trajectories"] == 32].iloc[0]
        numerical = int(train_32["numerical_equivalence_matches"])
        numerical_denominator = int(train_32["n_numerically_evaluated_rhs"])
        exact = int(train_32["exact_algebraic_matches"])
        exact_denominator = int(train_32["n_algebraically_evaluated_rhs"])
        structure_axes.text(
            0.04,
            0.06,
            f"Train 32: numerical {numerical}/{numerical_denominator}; exact {exact}/{exact_denominator}",
            transform=structure_axes.transAxes,
            fontsize=8,
            color=NAVY,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )

    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005)
    )
    figure.text(
        0.5,
        0.955,
        "Examples are illustrative and do not estimate population frequencies.",
        ha="center",
        color=GRAY,
        fontsize=9,
        style="italic",
    )
    figure.subplots_adjust(bottom=0.13, top=0.90, hspace=0.25, wspace=0.20)
    save_figure(figure, output_directory, "06_case_studies")


def generate_heldout_trajectory_figure(output_directory: Path) -> None:
    """Generate a held-out trajectory and equation comparison.

    Args:
        output_directory: Destination directory.
    """
    predictions = pd.read_csv(TEST_PREDICTIONS_TRAIN32_PATH)
    selected_predictions = predictions[
        (predictions["model_id"] == HELDOUT_EXAMPLE_MODEL_ID)
        & (predictions["dataset"] == HELDOUT_EXAMPLE_DATASET)
        & (predictions["trajectory_id"] == HELDOUT_EXAMPLE_TRAJECTORY_ID)
    ].copy()

    state_counts = selected_predictions.groupby("state_id").size().to_dict()
    expected_state_counts = {state_id: 101 for state_id in HELDOUT_EXAMPLE_STATES}
    if len(selected_predictions) != 303 or state_counts != expected_state_counts:
        raise ValueError(
            "Held-out example must contain 303 rows: 101 for each of x, y, and y0."
        )
    time_counts = selected_predictions.groupby("state_id")["time"].nunique().to_dict()
    if time_counts != expected_state_counts:
        raise ValueError("Held-out example must contain 101 distinct time points.")
    if set(selected_predictions["selection_method"]) != {"coordinate_search"} or set(
        selected_predictions["system_id"]
    ) != {1}:
        raise ValueError("Held-out example does not use the frozen selected system.")

    trajectory_metrics = pd.read_csv(TEST_TRAJECTORY_METRICS_TRAIN32_PATH)
    model_trajectory_metrics = trajectory_metrics[
        (trajectory_metrics["model_id"] == HELDOUT_EXAMPLE_MODEL_ID)
        & (trajectory_metrics["dataset"] == HELDOUT_EXAMPLE_DATASET)
    ].sort_values("macro_nrmse")
    if len(model_trajectory_metrics) != 8:
        raise ValueError("Expected eight official held-out trajectory metrics.")

    central_pair = model_trajectory_metrics.iloc[3:5]
    if central_pair["trajectory_id"].tolist() != ["test_003", "test_000"]:
        raise ValueError("test_003 is not the lower-error member of the central pair.")

    displayed_metric = model_trajectory_metrics[
        model_trajectory_metrics["trajectory_id"] == HELDOUT_EXAMPLE_TRAJECTORY_ID
    ]
    if len(displayed_metric) != 1:
        raise ValueError("Expected one official metric for test_003.")
    trajectory_macro_nrmse = float(displayed_metric.iloc[0]["macro_nrmse"])
    if not math.isclose(
        trajectory_macro_nrmse,
        3.763991365656924e-06,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("Unexpected test_003 macro-NRMSE.")

    test_comparison = pd.read_csv(BIOMD_0156_TEST_COMPARISON_PATH)
    selected_system = test_comparison[
        test_comparison["dataset"] == HELDOUT_EXAMPLE_DATASET
    ]
    if len(selected_system) != 1:
        raise ValueError("Expected one train_32 system-level test row.")
    selected_system_row = selected_system.iloc[0]
    validation_macro_nrmse = float(selected_system_row["validation_macro_nrmse"])
    test_macro_nrmse = float(selected_system_row["test_macro_nrmse"])
    expected_system_values = (
        math.isclose(
            validation_macro_nrmse,
            4.017627269111215e-06,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        and math.isclose(
            test_macro_nrmse,
            3.6540933364804454e-06,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        and selected_system_row["test_status"] == "success"
        and int(selected_system_row["expected_trajectories"]) == 8
        and int(selected_system_row["successful_trajectories"]) == 8
    )
    if not expected_system_values:
        raise ValueError("Unexpected train_32 system-level held-out metrics.")

    recovery = pd.read_csv(MECHANISTIC_RECOVERY_PATH)
    selected_recovery = recovery[
        (recovery["model_id"] == HELDOUT_EXAMPLE_MODEL_ID)
        & (recovery["dataset_name"] == HELDOUT_EXAMPLE_DATASET)
    ].copy()
    if set(selected_recovery["state_id"]) != set(HELDOUT_EXAMPLE_STATES):
        raise ValueError("Expected mechanistic-recovery rows for x, y, and y0.")
    if len(selected_recovery) != 3:
        raise ValueError("Expected exactly three mechanistic-recovery rows.")

    unit_score_columns = (
        "dependency_f1",
        "numerator_monomial_f1",
        "interaction_f1",
    )
    for column in unit_score_columns:
        if not selected_recovery[column].eq(1.0).all():
            raise ValueError(f"Expected {column} = 1.0 for all three RHSs.")
    boolean_expectations = {
        "rational_structure_match": "true",
        "numerical_equivalence_match": "true",
        "exact_algebraic_match": "false",
    }
    for column, expected_value in boolean_expectations.items():
        normalized_values = selected_recovery[column].astype(str).str.lower()
        if not normalized_values.eq(expected_value).all():
            raise ValueError(f"Unexpected {column} values for held-out example.")

    equation_rows = selected_recovery.set_index("state_id")
    figure = plt.figure(figsize=(14, 9.5))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.05],
        hspace=0.30,
        wspace=0.22,
    )
    trajectory_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    equation_axes = figure.add_subplot(grid[1, :])

    figure.suptitle(
        "Held-out trajectory reconstruction: BIOMD0000000156",
        color=NAVY,
        fontweight="bold",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.949,
        (f"train_32 | held-out test_003 macro-NRMSE = {trajectory_macro_nrmse:.12g}"),
        ha="center",
        color=NAVY,
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"System validation macro-NRMSE = {validation_macro_nrmse:.12g}; "
            f"held-out macro-NRMSE = {test_macro_nrmse:.12g} (8/8 trajectories)"
        ),
        ha="center",
        color=GRAY,
        fontsize=9.5,
    )

    for axes, state_id in zip(trajectory_axes, HELDOUT_EXAMPLE_STATES):
        state_rows = selected_predictions[
            selected_predictions["state_id"] == state_id
        ].sort_values("time")
        axes.plot(
            state_rows["time"],
            state_rows["observed"],
            color=BLUE,
            linewidth=2.8,
            label="SBML reference",
        )
        axes.plot(
            state_rows["time"],
            state_rows["predicted"],
            color=ORANGE,
            linestyle="--",
            linewidth=1.8,
            label="Selected inferred system",
        )
        axes.set_title(f"{state_id}(t)", color=NAVY, fontweight="bold")
        axes.set_xlabel("Time")
        axes.set_ylabel("State value")
        axes.margins(x=0.01)

    handles, labels = trajectory_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.905),
    )

    equation_axes.set_xlim(0, 1)
    equation_axes.set_ylim(0, 1)
    equation_axes.axis("off")
    equation_axes.text(
        0.5,
        0.97,
        (
            "All three RHSs: dependency F1 = 1.0; numerator monomial F1 = 1.0; "
            "interaction F1 = 1.0; rational-structure match = true"
        ),
        ha="center",
        va="top",
        color=NAVY,
        fontsize=9.5,
        fontweight="bold",
    )
    equation_axes.text(
        0.5,
        0.87,
        "Numerical equivalence: 3/3 RHSs | Exact algebraic equivalence: 0/3 RHSs",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=9.5,
    )

    box_positions = (0.63, 0.41, 0.19)
    box_colors = (LIGHT_BLUE, LIGHT_TEAL, LIGHT_PURPLE)
    box_edges = (BLUE, TEAL, PURPLE)
    for state_id, y_position, facecolor, edgecolor in zip(
        HELDOUT_EXAMPLE_STATES,
        box_positions,
        box_colors,
        box_edges,
    ):
        row = equation_rows.loc[state_id]
        equation_text = (
            f"{state_id} RHS\n"
            f"Ground truth: {row['canonical_true_equation']}\n"
            f"Inferred: {row['canonical_inferred_equation']}"
        )
        add_box(
            equation_axes,
            0.02,
            y_position,
            0.96,
            0.16,
            equation_text,
            facecolor,
            edgecolor,
            fontsize=8.5,
        )

    equation_axes.text(
        0.5,
        0.055,
        (
            "test_003 is the lower-error member of the two central held-out "
            "trajectories; it was not selected for visual appearance."
        ),
        ha="center",
        va="center",
        color=GRAY,
        fontsize=9,
        style="italic",
    )
    figure.subplots_adjust(top=0.85, bottom=0.06)
    save_figure(figure, output_directory, "05_heldout_trajectory_example")


def generate_docker_figure(output_directory: Path) -> None:
    """Generate the verified Docker build/runtime architecture diagram.

    Args:
        output_directory: Destination directory.
    """
    figure = plt.figure(figsize=(13, 7))
    axes = prepare_diagram_axes(figure)
    axes.set_title(
        "Docker build, immutable benchmark inputs, and writable runtime boundary",
        color=NAVY,
        fontweight="bold",
        pad=12,
    )

    axes.text(0.16, 0.90, "BUILD INPUTS", ha="center", color=GRAY, fontweight="bold")
    input_labels = [
        "Project source",
        "pyproject.toml + uv.lock",
        "Expanded-100 SBML bundle",
        "Simulation metadata +\ncohort manifest",
        "Network access during build",
    ]
    input_y = [0.75, 0.62, 0.49, 0.34, 0.19]
    for label, y_position in zip(input_labels, input_y):
        fill = LIGHT_ORANGE if "Network" in label else LIGHT_GRAY
        edge = ORANGE if "Network" in label else GRAY
        add_box(
            axes, 0.04, y_position, 0.24, 0.09, label, fill, edgecolor=edge, fontsize=9
        )

    add_box(
        axes,
        0.36,
        0.25,
        0.27,
        0.56,
        "BUILT CONTAINER IMAGE\n\n"
        "linux/amd64\n"
        "Locked Python environment\n"
        "Julia/PySR initialized at build\n"
        "Bundled SBML + metadata\n"
        "Validated benchmark manifest\n\n"
        "Julia runtime not completely\nversion-pinned",
        LIGHT_BLUE,
        edgecolor=BLUE,
        fontsize=10,
        linewidth=1.8,
    )
    for y_position in input_y:
        add_arrow(axes, (0.28, y_position + 0.045), (0.36, 0.53), color=BLUE)

    add_box(
        axes,
        0.70,
        0.48,
        0.25,
        0.27,
        "RUNTIME\n\nNon-root UID 10001\nBenchmark execution\nWritable Julia environment copy\nNo live BioModels download",
        LIGHT_GREEN,
        edgecolor=GREEN,
        fontsize=10,
        linewidth=1.7,
    )
    add_arrow(axes, (0.63, 0.57), (0.70, 0.61), color=GREEN, width=1.8)

    add_box(
        axes,
        0.70,
        0.16,
        0.25,
        0.18,
        "HOST-MOUNTED /output\n\nresults  •  runs  •  telemetry\nlocks  •  mutable caches",
        LIGHT_TEAL,
        edgecolor=TEAL,
        fontsize=10,
        linewidth=1.7,
    )
    add_arrow(axes, (0.825, 0.48), (0.825, 0.34), color=TEAL, width=1.8)
    axes.text(
        0.50,
        0.08,
        "The container improves environmental reproducibility; it does not claim bitwise reproducibility.",
        ha="center",
        color=GRAY,
        fontsize=10,
        style="italic",
    )
    save_figure(figure, output_directory, "07_docker_reproducibility_architecture")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for generated SVG and PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate every final-report figure from audited inputs."""
    arguments = parse_arguments()
    output_directory = arguments.output_dir.resolve()
    configure_style()
    generate_corpus_figure(output_directory)
    generate_pipeline_figure(output_directory)
    generate_execution_figure(output_directory)
    generate_performance_figure(output_directory)
    generate_heldout_trajectory_figure(output_directory)
    generate_case_study_figure(output_directory)
    generate_docker_figure(output_directory)
    print(f"Generated report figures in {output_directory}")


if __name__ == "__main__":
    main()
