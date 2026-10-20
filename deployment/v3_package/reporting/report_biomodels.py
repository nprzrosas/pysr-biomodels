"""Generate generic human-readable reports from completed PySR BioModels runs.

This reporting layer reads existing inference and validation artifacts only.
It does not run PySR, perform model selection, integrate ODE systems, or
access held-out test data.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "generic_exact_experiment"

TRAIN_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)


def discover_models(results_root):
    """Return model IDs having an exact-experiment result directory."""
    models = []

    if not results_root.exists():
        return models

    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue

        experiment = (
            path
            / EXPERIMENT_NAME
        )

        if experiment.is_dir():
            models.append(
                path.name
            )

    return models


def selected_system_path(
    results_root,
    model_id,
    dataset_name,
):
    """Return the authoritative coordinate-selected system path."""
    return (
        results_root
        / model_id
        / EXPERIMENT_NAME
        / dataset_name
        / "coordinate_selection"
        / "coordinate_selected_system.csv"
    )


def validation_predictions_path(
    results_root,
    model_id,
    dataset_name,
):
    """Return the validation-prediction path."""
    return (
        results_root
        / model_id
        / EXPERIMENT_NAME
        / dataset_name
        / "validation_predictions.csv"
    )


def load_selected_system(
    results_root,
    model_id,
    dataset_name,
):
    """Load one authoritative coordinate-selected system."""
    path = selected_system_path(
        results_root,
        model_id,
        dataset_name,
    )

    if not path.exists():
        return None

    frame = pd.read_csv(
        path
    )

    if len(frame) != 1:
        raise ValueError(
            f"{path} contains {len(frame)} rows; expected one."
        )

    row = frame.iloc[0]

    if (
        "selection_method" in row.index
        and str(row["selection_method"])
        != "coordinate_search"
    ):
        raise ValueError(
            f"{path} is not a coordinate-search selection."
        )

    return row


def equation_columns(selected_row):
    """Return selected equation columns in stored state order."""
    return [
        column
        for column in selected_row.index
        if column.startswith(
            "equation_"
        )
    ]


def candidate_rank_columns(selected_row):
    """Return candidate-rank columns in stored state order."""
    return [
        column
        for column in selected_row.index
        if column.startswith(
            "candidate_rank_"
        )
        and column
        != "candidate_rank_signature"
    ]


def state_ids_from_selected_row(
    selected_row,
):
    """Recover dynamic state IDs from equation columns."""
    return [
        column[
            len("equation_"):
        ]
        for column in equation_columns(
            selected_row
        )
    ]


def safe_value(
    row,
    column,
    default=np.nan,
):
    """Return a selected-row value when present."""
    if column not in row.index:
        return default

    value = row[
        column
    ]

    if pd.isna(
        value
    ):
        return default

    return value


def build_summary_row(
    model_id,
    dataset_name,
    selected_row,
):
    """Build one model/dataset summary record."""
    state_ids = state_ids_from_selected_row(
        selected_row
    )

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "train_trajectories": int(
            dataset_name.split(
                "_"
            )[-1]
        ),
        "n_states": len(
            state_ids
        ),
        "state_ids": ";".join(
            state_ids
        ),
        "selected_system": int(
            selected_row[
                "system_id"
            ]
        ),
        "selection_method": safe_value(
            selected_row,
            "selection_method",
            "",
        ),
        "validation_status": safe_value(
            selected_row,
            "validation_status",
            "",
        ),
        "validation_macro_nrmse": safe_value(
            selected_row,
            "validation_macro_nrmse",
        ),
        "successful_trajectories": safe_value(
            selected_row,
            "successful_trajectories",
        ),
        "failed_trajectories": safe_value(
            selected_row,
            "failed_trajectories",
        ),
        "complexity_total": safe_value(
            selected_row,
            "complexity_total",
        ),
        "training_loss_total": safe_value(
            selected_row,
            "training_loss_total",
        ),
        "total_rhs_evaluations": safe_value(
            selected_row,
            "total_rhs_evaluations",
        ),
        "coordinate_rounds": safe_value(
            selected_row,
            "coordinate_rounds",
        ),
        "systems_evaluated": safe_value(
            selected_row,
            "systems_evaluated",
        ),
        "failure_messages": safe_value(
            selected_row,
            "failure_messages",
            "",
        ),
    }


def build_equation_rows(
    model_id,
    dataset_name,
    selected_row,
):
    """Build one output row for every winning state equation."""
    rows = []

    system_id = int(
        selected_row[
            "system_id"
        ]
    )

    validation_macro_nrmse = safe_value(
        selected_row,
        "validation_macro_nrmse",
    )

    validation_status = safe_value(
        selected_row,
        "validation_status",
        "",
    )

    for state_id in state_ids_from_selected_row(
        selected_row
    ):
        rows.append(
            {
                "model_id": model_id,
                "dataset_name": dataset_name,
                "train_trajectories": int(
                    dataset_name.split(
                        "_"
                    )[-1]
                ),
                "selected_system": system_id,
                "state_id": state_id,
                "candidate_rank": safe_value(
                    selected_row,
                    f"candidate_rank_{state_id}",
                ),
                "complexity": safe_value(
                    selected_row,
                    f"complexity_{state_id}",
                ),
                "training_loss": safe_value(
                    selected_row,
                    f"training_loss_{state_id}",
                ),
                "equation": str(
                    selected_row[
                        f"equation_{state_id}"
                    ]
                ),
                "validation_status": validation_status,
                "validation_macro_nrmse": (
                    validation_macro_nrmse
                ),
                "mean_rmse": safe_value(
                    selected_row,
                    f"mean_rmse_{state_id}",
                ),
                "mean_normalized_rmse": safe_value(
                    selected_row,
                    f"mean_normalized_rmse_{state_id}",
                ),
                "mean_r2": safe_value(
                    selected_row,
                    f"mean_r2_{state_id}",
                ),
            }
        )

    return rows


def load_selected_validation_predictions(
    results_root,
    model_id,
    dataset_name,
    system_id,
):
    """Load validation predictions for the selected system only."""
    path = validation_predictions_path(
        results_root,
        model_id,
        dataset_name,
    )

    if not path.exists():
        return None

    frame = pd.read_csv(
        path
    )

    required = {
        "system_id",
        "trajectory_id",
        "state_id",
        "time",
        "observed",
        "predicted",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    frame = frame[
        frame["system_id"]
        == system_id
    ].copy()

    if frame.empty:
        return None

    return frame


def make_validation_overview(
    predictions,
    model_id,
    dataset_name,
    system_id,
    validation_macro_nrmse,
    output_path,
):
    """Plot all validation trajectories for one selected system."""
    state_ids = list(
        predictions[
            "state_id"
        ].drop_duplicates()
    )

    trajectory_ids = list(
        predictions[
            "trajectory_id"
        ].drop_duplicates()
    )

    if not state_ids or not trajectory_ids:
        raise ValueError(
            "Validation predictions are empty."
        )

    columns_per_page = min(
        4,
        len(
            trajectory_ids
        ),
    )

    rows_per_state = int(
        np.ceil(
            len(
                trajectory_ids
            )
            / columns_per_page
        )
    )

    total_rows = (
        len(
            state_ids
        )
        * rows_per_state
    )

    fig, axes = plt.subplots(
        total_rows,
        columns_per_page,
        figsize=(
            4.0
            * columns_per_page,
            2.7
            * total_rows,
        ),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_visible(
            False
        )

    for state_index, state_id in enumerate(
        state_ids
    ):
        for trajectory_index, trajectory_id in enumerate(
            trajectory_ids
        ):
            page_row = (
                trajectory_index
                // columns_per_page
            )

            column = (
                trajectory_index
                % columns_per_page
            )

            row = (
                state_index
                * rows_per_state
                + page_row
            )

            ax = axes[
                row
            ][
                column
            ]

            ax.set_visible(
                True
            )

            subset = predictions[
                (
                    predictions["state_id"]
                    == state_id
                )
                & (
                    predictions["trajectory_id"]
                    == trajectory_id
                )
            ].sort_values(
                "time"
            )

            ax.plot(
                subset["time"],
                subset["observed"],
                linewidth=1.8,
                label="Observed",
            )

            ax.plot(
                subset["time"],
                subset["predicted"],
                linewidth=1.8,
                linestyle="--",
                label="PySR inferred",
            )

            ax.set_title(
                f"{trajectory_id} | {state_id}",
                fontsize=9,
            )

            ax.set_xlabel(
                "Time"
            )

            ax.set_ylabel(
                state_id
            )

            ax.grid(
                alpha=0.2
            )

    visible_axes = [
        ax
        for ax in axes.flat
        if ax.get_visible()
    ]

    handles, labels = (
        visible_axes[
            0
        ].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
    )

    nrmse_text = (
        "NA"
        if pd.isna(
            validation_macro_nrmse
        )
        else f"{float(validation_macro_nrmse):.4g}"
    )

    fig.suptitle(
        f"{model_id} — {dataset_name}\n"
        f"Selected system {system_id} | "
        f"Validation macro NRMSE = {nrmse_text}",
        y=0.995,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.95,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def report_one_dataset(
    results_root,
    reports_root,
    model_id,
    dataset_name,
):
    """Report one completed model/training-size result."""
    selected_row = load_selected_system(
        results_root,
        model_id,
        dataset_name,
    )

    if selected_row is None:
        return {
            "summary": None,
            "equations": [],
            "failure": {
                "model_id": model_id,
                "dataset_name": dataset_name,
                "failure_type": "MISSING_SELECTION",
                "message": (
                    "coordinate_selected_system.csv "
                    "was not found."
                ),
            },
        }

    summary = build_summary_row(
        model_id,
        dataset_name,
        selected_row,
    )

    equations = build_equation_rows(
        model_id,
        dataset_name,
        selected_row,
    )

    failure = None

    validation_status = str(
        summary[
            "validation_status"
        ]
    )

    if validation_status != "success":
        failure = {
            "model_id": model_id,
            "dataset_name": dataset_name,
            "failure_type": "VALIDATION_FAILED",
            "message": str(
                summary[
                    "failure_messages"
                ]
            ),
        }

    predictions = (
        load_selected_validation_predictions(
            results_root,
            model_id,
            dataset_name,
            summary[
                "selected_system"
            ],
        )
    )

    if predictions is None:
        if failure is None:
            failure = {
                "model_id": model_id,
                "dataset_name": dataset_name,
                "failure_type": (
                    "MISSING_VALIDATION_PREDICTIONS"
                ),
                "message": (
                    "Selected validation predictions "
                    "were not found."
                ),
            }

    else:
        figure_path = (
            reports_root
            / "models"
            / model_id
            / "figures"
            / (
                f"{dataset_name}_"
                "validation_overview.png"
            )
        )

        make_validation_overview(
            predictions,
            model_id,
            dataset_name,
            summary[
                "selected_system"
            ],
            summary[
                "validation_macro_nrmse"
            ],
            figure_path,
        )

    return {
        "summary": summary,
        "equations": equations,
        "failure": failure,
    }


def write_model_reports(
    reports_root,
    model_id,
    summary_rows,
    equation_rows,
):
    """Write per-model CSV reports."""
    model_directory = (
        reports_root
        / "models"
        / model_id
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if summary_rows:
        pd.DataFrame(
            summary_rows
        ).to_csv(
            model_directory
            / "model_summary.csv",
            index=False,
        )

    if equation_rows:
        pd.DataFrame(
            equation_rows
        ).to_csv(
            model_directory
            / "winning_equations.csv",
            index=False,
        )


def generate_reports(
    results_root,
    reports_root,
    requested_models=None,
):
    """Generate generic reports for available model results."""
    available_models = discover_models(
        results_root
    )

    if requested_models:
        models = list(
            requested_models
        )
    else:
        models = available_models

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    all_equations = []
    all_failures = []

    for model_id in models:
        model_summaries = []
        model_equations = []

        if model_id not in available_models:
            all_failures.append(
                {
                    "model_id": model_id,
                    "dataset_name": "",
                    "failure_type": "MISSING_MODEL_RESULTS",
                    "message": (
                        "No exact-experiment result "
                        "directory was found."
                    ),
                }
            )

            continue

        print(
            f"Reporting {model_id}"
        )

        for dataset_name in TRAIN_DATASETS:
            try:
                result = report_one_dataset(
                    results_root,
                    reports_root,
                    model_id,
                    dataset_name,
                )

            except Exception as error:
                all_failures.append(
                    {
                        "model_id": model_id,
                        "dataset_name": dataset_name,
                        "failure_type": "REPORTING_ERROR",
                        "message": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

                print(
                    f"  {dataset_name}: "
                    f"REPORTING_ERROR: {error}"
                )

                continue

            summary = result[
                "summary"
            ]

            equations = result[
                "equations"
            ]

            failure = result[
                "failure"
            ]

            if summary is not None:
                model_summaries.append(
                    summary
                )

                all_summaries.append(
                    summary
                )

            model_equations.extend(
                equations
            )

            all_equations.extend(
                equations
            )

            if failure is not None:
                all_failures.append(
                    failure
                )

            print(
                f"  {dataset_name}: "
                f"{'OK' if failure is None else failure['failure_type']}"
            )

        write_model_reports(
            reports_root,
            model_id,
            model_summaries,
            model_equations,
        )

    pd.DataFrame(
        all_summaries
    ).to_csv(
        reports_root
        / "benchmark_summary.csv",
        index=False,
    )

    pd.DataFrame(
        all_equations
    ).to_csv(
        reports_root
        / "winning_equations.csv",
        index=False,
    )

    failure_columns = [
        "model_id",
        "dataset_name",
        "failure_type",
        "message",
    ]

    pd.DataFrame(
        all_failures,
        columns=failure_columns,
    ).to_csv(
        reports_root
        / "failures.csv",
        index=False,
    )

    print()
    print(
        "Models requested:",
        len(
            models
        ),
    )
    print(
        "Model/dataset summaries:",
        len(
            all_summaries
        ),
    )
    print(
        "Winning equations:",
        len(
            all_equations
        ),
    )
    print(
        "Failures/incomplete datasets:",
        len(
            all_failures
        ),
    )
    print(
        "Reports:",
        reports_root,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reports-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Optional model ID. Repeat to report "
            "multiple specific models."
        ),
    )

    args = parser.parse_args()

    generate_reports(
        args.results_root,
        args.reports_root,
        requested_models=args.models,
    )


if __name__ == "__main__":
    main()
