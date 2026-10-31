"""Generate readable cohort and per-model figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inventory import (
    TRAIN_DATASETS,
    load_selected_row,
    prediction_path,
    selection_path,
)

STATUS_COLORS = {
    "pretest_complete": "#2a9d8f",
    "selector_partial": "#e9c46a",
    "selector_interrupted": "#f4a261",
    "candidate_partial": "#e76f51",
    "candidate_failed": "#c1121f",
}


def load_validation_predictions(path: Path) -> pd.DataFrame:
    """Read validation predictions while accepting empty failure artifacts.

    Args:
        path: Coordinate-validation prediction CSV.

    Returns:
        Parsed predictions or an empty frame when the selector intentionally
        produced an empty artifact after all integrations failed.
    """
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_coverage_figure(
    model_frame: pd.DataFrame,
    condition_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot model- and condition-level pipeline coverage.

    Args:
        model_frame: Model pipeline inventory.
        condition_frame: Complete condition inventory.
        output_path: Destination PNG.
    """
    requested = len(model_frame)
    candidates = int(model_frame["candidate_complete"].sum())
    full_selection = int(model_frame["pipeline_status"].eq("pretest_complete").sum())
    selected_conditions = int(condition_frame["selection_present"].sum())
    expected_conditions = len(condition_frame)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    model_labels = ["Attempted", "PySR candidates", "Complete selection"]
    model_values = [requested, candidates, full_selection]
    axes[0].bar(model_labels, model_values, color=["#457b9d", "#2a9d8f", "#264653"])
    axes[0].set_ylim(0, max(model_values) * 1.18)
    axes[0].set_ylabel("Models")
    axes[0].set_title("Coverage by pipeline stage")
    axes[0].tick_params(axis="x", rotation=15)
    for index, value in enumerate(model_values):
        axes[0].text(index, value + 0.6, f"{value}/{requested}", ha="center")

    condition_values = [expected_conditions, selected_conditions]
    axes[1].bar(
        ["Expected", "Selected"],
        condition_values,
        color=["#457b9d", "#2a9d8f"],
    )
    axes[1].set_ylim(0, expected_conditions * 1.18)
    axes[1].set_ylabel("Model x training size")
    axes[1].set_title("Coverage by condition")
    for index, value in enumerate(condition_values):
        axes[1].text(
            index,
            value + 2,
            f"{value}/{expected_conditions}",
            ha="center",
        )
    fig.suptitle("PySR BioModels batch coverage", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_scalability_figure(model_frame: pd.DataFrame, output_path: Path) -> None:
    """Plot observed wall time and memory against model size.

    Args:
        model_frame: Model pipeline inventory with telemetry.
        output_path: Destination PNG.
    """
    frame = model_frame.copy()
    frame["observed_hours"] = (
        pd.to_numeric(frame.get("observed_wall_seconds"), errors="coerce") / 3600.0
    )
    frame["peak_rss_gib"] = pd.to_numeric(
        frame.get("peak_process_tree_rss_bytes"), errors="coerce"
    ) / (1024**3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for status, group in frame.groupby("pipeline_status", sort=False):
        color = STATUS_COLORS.get(str(status), "#6c757d")
        axes[0].scatter(
            group["n_states"],
            group["observed_hours"],
            label=status,
            color=color,
            alpha=0.85,
        )
        axes[1].scatter(
            group["n_states"],
            group["peak_rss_gib"],
            label=status,
            color=color,
            alpha=0.85,
        )
    axes[0].set_xlabel("Dynamic states")
    axes[0].set_ylabel("Observed wall time (hours)")
    axes[0].set_title("Runtime vs. model size")
    axes[1].set_xlabel("Dynamic states")
    axes[1].set_ylabel("Peak process-tree RSS (GiB)")
    axes[1].set_title("Memory vs. model size")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Observed scalability in one 60-worker run", fontsize=15)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_validation_heatmap(condition_frame: pd.DataFrame, output_path: Path) -> None:
    """Plot validation macro NRMSE across available conditions.

    Args:
        condition_frame: Complete condition inventory.
        output_path: Destination PNG.
    """
    frame = condition_frame.copy()
    frame["numeric_nrmse"] = pd.to_numeric(
        frame["validation_macro_nrmse"], errors="coerce"
    )
    finite_positive = frame["numeric_nrmse"].where(
        np.isfinite(frame["numeric_nrmse"]) & (frame["numeric_nrmse"] > 0)
    )
    frame["log10_nrmse"] = np.log10(finite_positive)
    model_order = (
        frame[["model_id", "n_states"]]
        .groupby("model_id", as_index=False)["n_states"]
        .max()
        .sort_values(["n_states", "model_id"])["model_id"]
        .tolist()
    )
    matrix = frame.pivot(
        index="model_id", columns="dataset_name", values="log10_nrmse"
    ).reindex(index=model_order, columns=TRAIN_DATASETS)

    fig_height = max(7.0, 0.28 * len(matrix) + 2.2)
    fig, axis = plt.subplots(figsize=(8.5, fig_height))
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = axis.imshow(masked, aspect="auto", cmap="viridis", interpolation="none")
    image.cmap.set_bad("#e5e7eb")
    axis.set_xticks(range(len(TRAIN_DATASETS)), TRAIN_DATASETS)
    axis.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    axis.set_title("Validation error (log10 NRMSE); gray = missing or non-finite")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("log10(NRMSE)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_mechanistic_summary_figure(
    mechanistic_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mechanistic recovery across training sizes.

    Args:
        mechanistic_summary: Per-condition mechanistic recovery summaries.
        output_path: Destination PNG.
    """
    if mechanistic_summary.empty:
        return
    compatible = mechanistic_summary.loc[
        mechanistic_summary["mechanistic_comparability"] == "compatible"
    ].copy()
    metrics = [
        ("macro_dependency_f1", "Dependency F1"),
        ("exact_algebraic_fraction", "Algebraically exact fraction"),
        ("numerical_equivalence_fraction", "Numerically equivalent fraction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, (column, title) in zip(axes, metrics, strict=True):
        values = []
        counts = []
        for dataset_name in TRAIN_DATASETS:
            series = pd.to_numeric(
                compatible.loc[compatible["dataset_name"] == dataset_name, column],
                errors="coerce",
            ).dropna()
            values.append(series.mean() if len(series) else np.nan)
            counts.append(len(series))
        bars = axis.bar(TRAIN_DATASETS, values, color="#2a9d8f")
        axis.set_title(title)
        axis.set_ylim(0, 1.08)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        for bar, count in zip(bars, counts, strict=True):
            height = bar.get_height()
            if np.isfinite(height):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.025,
                    f"n={count}",
                    ha="center",
                    fontsize=8,
                )
    axes[0].set_ylabel("Mean across evaluable conditions")
    fig.suptitle(
        "Mechanistic recovery by training size\n"
        "Models compatible with the inference state space only",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def select_representative_states(
    equation_frame: pd.DataFrame,
    maximum_states: int,
) -> list[str]:
    """Choose best and worst states by validation NRMSE.

    Args:
        equation_frame: Selected equation rows for one model.
        maximum_states: Maximum states to display.

    Returns:
        Ordered representative state identifiers.
    """
    available = equation_frame.dropna(subset=["mean_normalized_rmse"]).copy()
    if available.empty:
        return (
            equation_frame["state_id"].drop_duplicates().head(maximum_states).tolist()
        )
    preferred_dataset = next(
        (
            dataset
            for dataset in reversed(TRAIN_DATASETS)
            if dataset in set(available["dataset_name"])
        ),
        TRAIN_DATASETS[0],
    )
    available = available.loc[available["dataset_name"] == preferred_dataset]
    state_order = available["state_id"].astype(str).drop_duplicates().tolist()
    available["mean_normalized_rmse"] = pd.to_numeric(
        available["mean_normalized_rmse"], errors="coerce"
    )
    available = available.dropna(subset=["mean_normalized_rmse"]).sort_values(
        "mean_normalized_rmse"
    )
    if len(available) <= maximum_states:
        return state_order
    half = maximum_states // 2
    best = available.head(half)["state_id"].tolist()
    worst = available.tail(maximum_states - half)["state_id"].tolist()
    selected_states = set(best + worst)
    return [state_id for state_id in state_order if state_id in selected_states]


def _selected_system_mask(
    system_ids: pd.Series,
    selected_system_id: object,
) -> pd.Series:
    """Identify rows belonging to one frozen selected system.

    Args:
        system_ids: System identifiers read from an artifact.
        selected_system_id: Identifier stored in the selected-system row.

    Returns:
        Boolean mask aligned with ``system_ids``.
    """
    stored_ids = (
        system_ids.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    selected_id = str(selected_system_id).strip()
    if selected_id.endswith(".0"):
        selected_id = selected_id.rstrip("0").rstrip(".")
    return stored_ids == selected_id


def load_selected_validation_predictions(
    path: Path,
    selected_system_id: object,
) -> pd.DataFrame:
    """Load only predictions for the frozen selected system.

    The coordinate-search artifact may contain predictions for hundreds of
    candidate coupled systems. Reading in chunks prevents those candidates
    from being retained in memory or accidentally drawn in the report.

    Args:
        path: Coordinate-validation prediction CSV.
        selected_system_id: Frozen selected-system identifier.

    Returns:
        Pointwise predictions for the selected system only.
    """
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    columns = [
        "system_id",
        "trajectory_id",
        "state_id",
        "time",
        "observed",
        "predicted",
    ]
    selected_chunks = []
    try:
        chunks = pd.read_csv(path, usecols=columns, chunksize=200_000)
        for chunk in chunks:
            selected = chunk.loc[
                _selected_system_mask(chunk["system_id"], selected_system_id)
            ]
            if not selected.empty:
                selected_chunks.append(selected)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if not selected_chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(selected_chunks, ignore_index=True)


def load_selected_validation_metrics(
    path: Path,
    selected_system_id: object,
) -> pd.DataFrame:
    """Load trajectory-state NRMSE values for one selected system.

    Args:
        path: Coordinate-validation state-metrics CSV.
        selected_system_id: Frozen selected-system identifier.

    Returns:
        Metrics for the selected system only.
    """
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    columns = ["system_id", "trajectory_id", "state_id", "normalized_rmse"]
    try:
        frame = pd.read_csv(path, usecols=columns)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()
    return frame.loc[
        _selected_system_mask(frame["system_id"], selected_system_id)
    ].copy()


def _panel_nrmse(
    metrics: pd.DataFrame,
    trajectory_id: str,
    state_id: str,
) -> float:
    """Return one stored trajectory-state NRMSE value.

    Args:
        metrics: Selected-system validation metrics.
        trajectory_id: Frozen validation trajectory identifier.
        state_id: Dynamic state identifier.

    Returns:
        NRMSE or NaN when the panel was not evaluable.
    """
    if metrics.empty:
        return float("nan")
    rows = metrics.loc[
        metrics["trajectory_id"].astype(str).eq(trajectory_id)
        & metrics["state_id"].astype(str).eq(state_id),
        "normalized_rmse",
    ]
    if len(rows) != 1:
        return float("nan")
    return float(pd.to_numeric(rows, errors="coerce").iloc[0])


def make_model_validation_figures(
    results_root: Path,
    model_id: str,
    equation_frame: pd.DataFrame,
    output_directory: Path,
    maximum_states: int = 10,
) -> tuple[list[str], list[str]]:
    """Plot selected-system validation trajectories in readable grids.

    Args:
        results_root: Scientific result root.
        model_id: BioModels identifier.
        equation_frame: Selected equations for this model.
        output_directory: Destination figure directory.
        maximum_states: Maximum representative states to plot.

    Returns:
        State identifiers and training datasets with generated figures.
    """
    state_ids = select_representative_states(equation_frame, maximum_states)
    if not state_ids:
        return [], []

    output_directory.mkdir(parents=True, exist_ok=True)
    generated_datasets = []
    expected_trajectories = tuple(f"validation_{index:03d}" for index in range(8))
    for dataset_name in TRAIN_DATASETS:
        path = prediction_path(results_root, model_id, dataset_name)
        selection = selection_path(results_root, model_id, dataset_name)
        if not selection.is_file():
            continue
        selected_row = load_selected_row(selection)
        selected_system_id = selected_row["system_id"]
        predictions = load_selected_validation_predictions(path, selected_system_id)
        metrics = load_selected_validation_metrics(
            path.with_name("coordinate_validation_state_metrics.csv"),
            selected_system_id,
        )
        stored_trajectories = tuple(
            predictions.get("trajectory_id", pd.Series(dtype=str))
            .astype(str)
            .drop_duplicates()
        )
        trajectory_ids = tuple(
            dict.fromkeys(expected_trajectories + stored_trajectories)
        )
        fig, axes = plt.subplots(
            len(state_ids),
            len(trajectory_ids),
            figsize=(2.3 * len(trajectory_ids), 2.15 * len(state_ids) + 1.4),
            squeeze=False,
            sharex=False,
        )
        legend_handles = None
        for row_index, state_id in enumerate(state_ids):
            for column, trajectory_id in enumerate(trajectory_ids):
                axis = axes[row_index, column]
                if predictions.empty:
                    trajectory = pd.DataFrame()
                else:
                    trajectory = predictions.loc[
                        predictions["state_id"].astype(str).eq(state_id)
                        & predictions["trajectory_id"].astype(str).eq(trajectory_id)
                    ].sort_values("time")
                if trajectory.empty:
                    axis.text(
                        0.5,
                        0.5,
                        "INTEGRATION\nFAILED OR\nNOT EVALUABLE",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=7,
                    )
                    axis.set_xticks([])
                    axis.set_yticks([])
                else:
                    original = axis.plot(
                        trajectory["time"],
                        trajectory["observed"],
                        color="#1f77b4",
                        linewidth=1.35,
                        label="Original dynamics",
                    )[0]
                    inferred = axis.plot(
                        trajectory["time"],
                        trajectory["predicted"],
                        color="#ff7f0e",
                        linewidth=1.15,
                        linestyle="--",
                        label="Selected inferred ODE",
                    )[0]
                    legend_handles = (original, inferred)
                    nrmse = _panel_nrmse(metrics, trajectory_id, state_id)
                    if np.isfinite(nrmse):
                        axis.text(
                            0.97,
                            0.95,
                            f"NRMSE={nrmse:.3g}",
                            ha="right",
                            va="top",
                            transform=axis.transAxes,
                            fontsize=7,
                        )
                    axis.tick_params(axis="both", labelsize=7)
                if row_index == 0:
                    axis.set_title(
                        trajectory_id.replace("validation_", "Val "), fontsize=9
                    )
                if column == 0:
                    axis.set_ylabel(state_id)
                if row_index == len(state_ids) - 1:
                    axis.set_xlabel("Time")
                axis.grid(alpha=0.18)

        macro_nrmse = pd.to_numeric(
            pd.Series([selected_row.get("validation_macro_nrmse")]), errors="coerce"
        ).iloc[0]
        macro_text = "NA" if pd.isna(macro_nrmse) else f"{float(macro_nrmse):.3g}"
        if legend_handles:
            fig.legend(
                legend_handles,
                ["Original dynamics", "Selected inferred ODE"],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=2,
            )
        fig.suptitle(
            f"{model_id}: original vs. inferred validation trajectories "
            f"({dataset_name})\nSelected-system macro NRMSE={macro_text}",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        output_path = output_directory / f"validation_{dataset_name}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated_datasets.append(dataset_name)

    return state_ids, generated_datasets
