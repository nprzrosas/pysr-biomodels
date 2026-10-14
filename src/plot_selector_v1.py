"""Plot characterization results for frozen Selector v1.

The figures are derived exclusively from the frozen Selector v1 aggregate
report. This module does not run PySR, candidate selection, held-out testing,
or ground-truth evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/selector_v1_aggregate_report.csv")
DEFAULT_OUTPUT_DIR = Path("results/selector_v1_figures")

TRAIN_SIZES = (4, 8, 16, 32)


def load_aggregate(path: Path) -> pd.DataFrame:
    """Load and validate the frozen Selector v1 aggregate report."""

    frame = pd.read_csv(
        path,
        dtype={"selected_ranks": str},
    )

    required = {
        "model_id",
        "n_states",
        "train_trajectories",
        "candidate_space",
        "systems_evaluated",
        "search_fraction",
        "validation_macro_nrmse",
        "validation_status",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return frame


def model_labels(frame: pd.DataFrame) -> dict[str, str]:
    """Build compact labels containing model ID and state count."""

    labels = {}

    for model_id, group in frame.groupby("model_id"):
        n_states = int(group["n_states"].iloc[0])
        short_id = model_id.replace("BIOMD000000", "")
        labels[model_id] = f"{short_id} ({n_states})"

    return labels


def plot_candidate_space(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot candidate-system space versus number of states."""

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    for train_size in TRAIN_SIZES:
        subset = frame[
            frame["train_trajectories"] == train_size
        ].sort_values(
            ["n_states", "model_id"]
        )

        ax.scatter(
            subset["n_states"],
            subset["candidate_space"],
            s=60,
            label=f"{train_size} trajectories",
        )

    ax.set_yscale("log")

    ax.set_xlabel("Number of dynamic states")
    ax.set_ylabel("Candidate system space")
    ax.set_title(
        "Selector v1: combinatorial candidate space"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "candidate_space_vs_states.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "candidate_space_vs_states.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_systems_evaluated(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot actual systems evaluated versus number of states."""

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    for train_size in TRAIN_SIZES:
        subset = frame[
            frame["train_trajectories"] == train_size
        ].sort_values(
            ["n_states", "model_id"]
        )

        ax.scatter(
            subset["n_states"],
            subset["systems_evaluated"],
            s=60,
            label=f"{train_size} trajectories",
        )

    ax.set_xlabel("Number of dynamic states")
    ax.set_ylabel("Systems evaluated")
    ax.set_title(
        "Selector v1: actual search effort"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "systems_evaluated_vs_states.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "systems_evaluated_vs_states.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_validation_vs_training(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot validation macro NRMSE across training sizes by model."""

    fig, ax = plt.subplots(
        figsize=(9.0, 6.0)
    )

    labels = model_labels(frame)

    for model_id, group in frame.groupby(
        "model_id",
        sort=False,
    ):
        subset = group.sort_values(
            "train_trajectories"
        ).copy()

        values = subset[
            "validation_macro_nrmse"
        ].to_numpy(
            dtype=float
        )

        finite_values = np.where(
            np.isfinite(values),
            values,
            np.nan,
        )

        ax.plot(
            subset["train_trajectories"],
            finite_values,
            marker="o",
            linewidth=1.5,
            label=labels[model_id],
        )

    ax.set_yscale("log")

    ax.set_xticks(
        list(TRAIN_SIZES)
    )

    ax.set_xlabel(
        "Training trajectories"
    )
    ax.set_ylabel(
        "Validation macro NRMSE"
    )
    ax.set_title(
        "Selector v1: validation error vs number of training trajectories"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend(
        title="Model (states)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "validation_vs_training.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "validation_vs_training.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_search_efficiency(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot actual search effort against candidate-system space."""

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    for train_size in TRAIN_SIZES:
        subset = frame[
            frame["train_trajectories"] == train_size
        ]

        ax.scatter(
            subset["candidate_space"],
            subset["systems_evaluated"],
            s=60,
            label=f"{train_size} trajectories",
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Candidate system space"
    )
    ax.set_ylabel(
        "Systems evaluated"
    )
    ax.set_title(
        "Selector v1: search effort vs. candidate space"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "search_effort_vs_candidate_space.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "search_effort_vs_candidate_space.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_search_fraction(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot explored fraction of candidate space versus number of states."""

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    for train_size in TRAIN_SIZES:
        subset = frame[
            frame["train_trajectories"] == train_size
        ].sort_values(
            ["n_states", "model_id"]
        )

        ax.scatter(
            subset["n_states"],
            subset["search_fraction"],
            s=60,
            label=f"{train_size} trajectories",
        )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Number of dynamic states"
    )
    ax.set_ylabel(
        "Fraction of candidate space evaluated"
    )
    ax.set_title(
        "Selector v1: fraction of candidate space explored"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "search_fraction_vs_states.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "search_fraction_vs_states.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Create characterization figures from the "
            "frozen Selector v1 aggregate report."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    frame = load_aggregate(
        args.input
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_candidate_space(
        frame=frame,
        output_dir=args.output_dir,
    )

    plot_systems_evaluated(
        frame=frame,
        output_dir=args.output_dir,
    )

    plot_search_efficiency(
        frame=frame,
        output_dir=args.output_dir,
    )

    plot_validation_vs_training(
        frame=frame,
        output_dir=args.output_dir,
    )

    plot_search_fraction(
        frame=frame,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("SELECTOR V1 FIGURES")
    print("=" * 80)
    print(
        "Created:",
        args.output_dir / "candidate_space_vs_states.pdf",
    )
    print(
        "Created:",
        args.output_dir / "systems_evaluated_vs_states.pdf",
    )
    print(
        "Created:",
        args.output_dir / "validation_vs_training.pdf",
    )
    print(
        "Created:",
        args.output_dir / "search_effort_vs_candidate_space.pdf",
    )
    print(
        "Created:",
        args.output_dir / "search_fraction_vs_states.pdf",
    )
    print()
    print("PNG versions were also created at 300 dpi.")


if __name__ == "__main__":
    main()
