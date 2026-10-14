"""Plot original-SBML versus inferred-SBML validation trajectories.

Purpose
-------
Visualize the frozen dynamic-fidelity results produced by
``compare_original_inferred_sbml.py``.

The plotting code reads only existing CSV outputs. It does not:

- simulate either SBML model;
- run PySR;
- select candidates;
- refit equations;
- modify frozen outputs;
- use held-out test trajectories.

For each training condition, the figure contains:

    rows    = dynamic states
    columns = validation trajectories

Each successful panel shows:

    original SBML trajectory
    inferred SBML trajectory

If either model failed to integrate for a validation trajectory,
the corresponding column is explicitly marked as an integration failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..export_inferred_sbml import (
    TRAINING_DATASETS,
    load_model_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "generic_exact_experiment"

SELECTION_DIRECTORY = "coordinate_selection"

SBML_VALIDATION_DIRECTORY = "sbml_validation"

DYNAMIC_FIDELITY_DIRECTORY = "dynamic_fidelity"

PREDICTIONS_FILENAME = (
    "dynamic_fidelity_predictions.csv"
)

TRAJECTORY_METRICS_FILENAME = (
    "dynamic_fidelity_trajectory_metrics.csv"
)

FIGURE_FILENAME = (
    "original_vs_inferred_sbml_validation.png"
)

FIGURE_DPI = 180


def dynamic_fidelity_directory(
    *,
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return dynamic-fidelity output directory."""

    return (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / training_dataset
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
        / DYNAMIC_FIDELITY_DIRECTORY
    )


def predictions_path(
    *,
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return frozen dynamic-fidelity predictions path."""

    return (
        dynamic_fidelity_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / PREDICTIONS_FILENAME
    )


def trajectory_metrics_path(
    *,
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return frozen trajectory metrics path."""

    return (
        dynamic_fidelity_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / TRAJECTORY_METRICS_FILENAME
    )


def figure_path(
    *,
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return destination figure path."""

    return (
        dynamic_fidelity_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / FIGURE_FILENAME
    )


def load_predictions(
    *,
    model_id: str,
    training_dataset: str,
) -> pd.DataFrame:
    """Load frozen pointwise dynamic-fidelity predictions."""

    path = predictions_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Dynamic-fidelity predictions not found. "
            "Run src.compare_original_inferred_sbml first. "
            f"Missing: {path}"
        )

    frame = pd.read_csv(
        path
    )

    required_columns = {
        "model_id",
        "dataset_name",
        "trajectory_id",
        "time",
        "state_id",
        "initial_value",
        "original_value",
        "inferred_value",
        "absolute_error",
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Predictions file is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    unexpected_models = set(
        frame[
            "model_id"
        ].astype(str)
    ) - {
        model_id
    }

    if unexpected_models:
        raise ValueError(
            "Predictions contain unexpected model IDs: "
            + ", ".join(
                sorted(
                    unexpected_models
                )
            )
        )

    unexpected_datasets = set(
        frame[
            "dataset_name"
        ].astype(str)
    ) - {
        training_dataset
    }

    if unexpected_datasets:
        raise ValueError(
            "Predictions contain unexpected dataset names: "
            + ", ".join(
                sorted(
                    unexpected_datasets
                )
            )
        )

    return frame


def load_trajectory_metrics(
    *,
    model_id: str,
    training_dataset: str,
) -> pd.DataFrame:
    """Load frozen trajectory-level dynamic-fidelity metrics."""

    path = trajectory_metrics_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Dynamic-fidelity trajectory metrics not found. "
            "Run src.compare_original_inferred_sbml first. "
            f"Missing: {path}"
        )

    frame = pd.read_csv(
        path
    )

    required_columns = {
        "model_id",
        "dataset_name",
        "trajectory_id",
        "state_id",
        "original_status",
        "inferred_status",
        "comparison_status",
        "normalized_rmse",
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Trajectory metrics file is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    unexpected_models = set(
        frame[
            "model_id"
        ].astype(str)
    ) - {
        model_id
    }

    if unexpected_models:
        raise ValueError(
            "Trajectory metrics contain unexpected model IDs: "
            + ", ".join(
                sorted(
                    unexpected_models
                )
            )
        )

    unexpected_datasets = set(
        frame[
            "dataset_name"
        ].astype(str)
    ) - {
        training_dataset
    }

    if unexpected_datasets:
        raise ValueError(
            "Trajectory metrics contain unexpected dataset names: "
            + ", ".join(
                sorted(
                    unexpected_datasets
                )
            )
        )

    return frame


def trajectory_statuses(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly one status row per validation trajectory."""

    status_columns = [
        "trajectory_id",
        "original_status",
        "inferred_status",
        "comparison_status",
    ]

    statuses = (
        metrics[
            status_columns
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    duplicated = statuses[
        "trajectory_id"
    ].duplicated(
        keep=False
    )

    if duplicated.any():
        duplicate_ids = (
            statuses.loc[
                duplicated,
                "trajectory_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Inconsistent trajectory status information for: "
            + ", ".join(
                duplicate_ids
            )
        )

    return statuses


def ordered_trajectory_ids(
    metrics: pd.DataFrame,
) -> tuple[str, ...]:
    """Return validation trajectories in their frozen order."""

    return tuple(
        metrics[
            "trajectory_id"
        ]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )


def trajectory_status(
    *,
    statuses: pd.DataFrame,
    trajectory_id: str,
) -> pd.Series:
    """Return status information for one trajectory."""

    rows = statuses.loc[
        statuses[
            "trajectory_id"
        ].astype(str)
        == trajectory_id
    ]

    if len(
        rows
    ) != 1:
        raise RuntimeError(
            "Expected exactly one status row for "
            f"{trajectory_id}; found {len(rows)}."
        )

    return rows.iloc[
        0
    ]


def state_nrmse(
    *,
    metrics: pd.DataFrame,
    trajectory_id: str,
    state_id: str,
) -> float:
    """Return frozen NRMSE for one trajectory/state pair."""

    rows = metrics.loc[
        (
            metrics[
                "trajectory_id"
            ].astype(str)
            == trajectory_id
        )
        & (
            metrics[
                "state_id"
            ].astype(str)
            == state_id
        )
    ]

    if len(
        rows
    ) != 1:
        return float(
            "nan"
        )

    value = rows.iloc[
        0
    ][
        "normalized_rmse"
    ]

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return float(
            "nan"
        )


def plot_failed_panel(
    *,
    axis,
    status: pd.Series,
) -> None:
    """Mark a panel whose trajectory could not be compared."""

    axis.set_xticks(
        []
    )

    axis.set_yticks(
        []
    )

    original_status = str(
        status[
            "original_status"
        ]
    )

    inferred_status = str(
        status[
            "inferred_status"
        ]
    )

    if (
        original_status == "pass"
        and inferred_status != "pass"
    ):
        message = (
            "INFERRED\n"
            "INTEGRATION\n"
            "FAILED"
        )

    elif (
        original_status != "pass"
        and inferred_status == "pass"
    ):
        message = (
            "ORIGINAL\n"
            "INTEGRATION\n"
            "FAILED"
        )

    elif (
        original_status != "pass"
        and inferred_status != "pass"
    ):
        message = (
            "BOTH\n"
            "INTEGRATIONS\n"
            "FAILED"
        )

    else:
        message = (
            "TRAJECTORY\n"
            "NOT\n"
            "EVALUABLE"
        )

    axis.text(
        0.5,
        0.5,
        message,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axis.transAxes,
        fontsize=8,
    )


def plot_successful_panel(
    *,
    axis,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    trajectory_id: str,
    state_id: str,
) -> None:
    """Plot original and inferred trajectories for one panel."""

    panel = predictions.loc[
        (
            predictions[
                "trajectory_id"
            ].astype(str)
            == trajectory_id
        )
        & (
            predictions[
                "state_id"
            ].astype(str)
            == state_id
        )
    ].copy()

    panel = (
        panel
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )

    if panel.empty:
        raise RuntimeError(
            "Missing successful prediction data for "
            f"{trajectory_id}, state {state_id}."
        )

    time_values = panel[
        "time"
    ].to_numpy(
        dtype=float
    )

    original_values = panel[
        "original_value"
    ].to_numpy(
        dtype=float
    )

    inferred_values = panel[
        "inferred_value"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            time_values
        )
    ):
        raise RuntimeError(
            "Non-finite time values found for "
            f"{trajectory_id}, state {state_id}."
        )

    if not np.all(
        np.isfinite(
            original_values
        )
    ):
        raise RuntimeError(
            "Non-finite original values found for "
            f"{trajectory_id}, state {state_id}."
        )

    if not np.all(
        np.isfinite(
            inferred_values
        )
    ):
        raise RuntimeError(
            "Non-finite inferred values found for "
            f"{trajectory_id}, state {state_id}."
        )

    axis.plot(
        time_values,
        original_values,
        linewidth=1.4,
        label="Original SBML",
    )

    axis.plot(
        time_values,
        inferred_values,
        linewidth=1.2,
        linestyle="--",
        label="Inferred SBML",
    )

    nrmse = state_nrmse(
        metrics=metrics,
        trajectory_id=trajectory_id,
        state_id=state_id,
    )

    if np.isfinite(
        nrmse
    ):
        axis.text(
            0.97,
            0.95,
            f"NRMSE={nrmse:.3g}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
            fontsize=7,
        )

    axis.tick_params(
        axis="both",
        labelsize=7,
    )


def normalize_axes(
    axes,
    number_of_states: int,
    number_of_trajectories: int,
):
    """Return axes as a two-dimensional NumPy array."""

    axes = np.asarray(
        axes,
        dtype=object,
    )

    if (
        number_of_states == 1
        and number_of_trajectories == 1
    ):
        return axes.reshape(
            1,
            1,
        )

    if number_of_states == 1:
        return axes.reshape(
            1,
            number_of_trajectories,
        )

    if number_of_trajectories == 1:
        return axes.reshape(
            number_of_states,
            1,
        )

    return axes


def plot_training_dataset(
    *,
    model_id: str,
    training_dataset: str,
) -> Path:
    """Generate one original-vs-inferred validation figure."""

    spec = load_model_spec(
        model_id
    )

    predictions = load_predictions(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    metrics = load_trajectory_metrics(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    statuses = trajectory_statuses(
        metrics
    )

    trajectory_ids = ordered_trajectory_ids(
        metrics
    )

    state_ids = tuple(
        spec.state_ids
    )

    number_of_states = len(
        state_ids
    )

    number_of_trajectories = len(
        trajectory_ids
    )

    if number_of_states == 0:
        raise RuntimeError(
            "Model contains no dynamic states."
        )

    if number_of_trajectories == 0:
        raise RuntimeError(
            "No validation trajectories found."
        )

    figure_width = max(
        12.0,
        2.3
        * number_of_trajectories,
    )

    figure_height = max(
        4.0,
        2.2
        * number_of_states,
    )

    figure, axes = plt.subplots(
        number_of_states,
        number_of_trajectories,
        figsize=(
            figure_width,
            figure_height,
        ),
        squeeze=False,
    )

    axes = normalize_axes(
        axes,
        number_of_states=number_of_states,
        number_of_trajectories=number_of_trajectories,
    )

    for column_index, trajectory_id in enumerate(
        trajectory_ids
    ):
        status = trajectory_status(
            statuses=statuses,
            trajectory_id=trajectory_id,
        )

        comparison_status = str(
            status[
                "comparison_status"
            ]
        )

        for row_index, state_id in enumerate(
            state_ids
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            if comparison_status == "success":
                plot_successful_panel(
                    axis=axis,
                    predictions=predictions,
                    metrics=metrics,
                    trajectory_id=trajectory_id,
                    state_id=state_id,
                )

            else:
                plot_failed_panel(
                    axis=axis,
                    status=status,
                )

            if row_index == 0:
                axis.set_title(
                    trajectory_id.replace(
                        "validation_",
                        "Val ",
                    ),
                    fontsize=9,
                )

            if column_index == 0:
                axis.set_ylabel(
                    state_id,
                    fontsize=10,
                )

            if row_index == (
                number_of_states - 1
            ):
                axis.set_xlabel(
                    "Time",
                    fontsize=8,
                )

    first_success_axis = None

    for column_index, trajectory_id in enumerate(
        trajectory_ids
    ):
        status = trajectory_status(
            statuses=statuses,
            trajectory_id=trajectory_id,
        )

        if str(
            status[
                "comparison_status"
            ]
        ) == "success":
            first_success_axis = axes[
                0,
                column_index,
            ]

            break

    if first_success_axis is not None:
        handles, labels = (
            first_success_axis
            .get_legend_handles_labels()
        )

        if handles:
            figure.legend(
                handles,
                labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                fontsize=9,
                bbox_to_anchor=(
                    0.5,
                    0.965,
                ),
            )

    figure.suptitle(
        (
            f"{model_id}: original SBML vs inferred SBML "
            f"on validation trajectories ({training_dataset})"
        ),
        fontsize=12,
        y=0.995,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    destination = figure_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        destination,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return destination


def plot_model(
    model_id: str,
) -> tuple[Path, ...]:
    """Generate validation figures for all frozen training sizes."""

    print(
        f"Model: {model_id}"
    )

    print(
        "Source: frozen original-vs-inferred SBML "
        "dynamic-fidelity CSV files"
    )

    print(
        "Held-out test trajectories used: 0"
    )

    print()

    paths = []

    for training_dataset in TRAINING_DATASETS:
        print(
            f"Plotting {training_dataset}..."
        )

        path = plot_training_dataset(
            model_id=model_id,
            training_dataset=training_dataset,
        )

        paths.append(
            path
        )

        print(
            f"  {path}"
        )

    print()

    print(
        "Generated "
        f"{len(paths)} validation figures."
    )

    print(
        "No simulation, PySR inference, candidate selection, "
        "or refitting was performed."
    )

    return tuple(
        paths
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot frozen original-SBML versus inferred-SBML "
            "validation trajectories."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000079."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate dynamic-fidelity figures."""

    args = parse_arguments()

    plot_model(
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()