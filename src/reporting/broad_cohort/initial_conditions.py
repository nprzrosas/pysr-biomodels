"""Reconstruct development initial conditions without opening held-out data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experimental_design_generic import (
    DEFAULT_POOL_SIZE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VALIDATION_SEED,
    generate_log_lhs_pool,
    maximin_order,
    select_maximin_against_reference,
)
from src.model_spec import load_model_spec


def _condition_frame(
    matrix: np.ndarray,
    state_ids: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    """Convert one initial-condition matrix to a readable table.

    Args:
        matrix: Trajectories by dynamic states.
        state_ids: Dynamic-state order.
        prefix: Stable trajectory prefix.

    Returns:
        Table with one row per initial condition.
    """
    rows = []
    for index, values in enumerate(matrix):
        row = {
            "trajectory_index": index,
            "trajectory_id": f"{prefix}_{index:03d}",
        }
        row.update(
            {
                f"{state_id}_0": float(value)
                for state_id, value in zip(state_ids, values, strict=True)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_development_initial_conditions(
    model_id: str,
    model_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Reconstruct frozen training and validation initial conditions.

    This intentionally stops after validation and never generates the
    held-out initial-condition pool.

    Args:
        model_id: BioModels identifier.
        model_path: Original curated SBML file.

    Returns:
        Training conditions, validation conditions, and nominal state values.
    """
    spec = load_model_spec(model_id=model_id, model_path=model_path)
    nominal = np.asarray(
        [spec.initial_values[state_id] for state_id in spec.state_ids],
        dtype=float,
    )
    if np.any(nominal <= 0.0):
        raise ValueError(f"Model {model_id} has non-positive nominal initial values.")

    training_pool = generate_log_lhs_pool(
        nominal_initial_values=nominal,
        pool_size=DEFAULT_POOL_SIZE,
        seed=DEFAULT_TRAIN_SEED,
    )
    training = maximin_order(
        pool=training_pool,
        nominal_initial_values=nominal,
        count=32,
    )
    validation_pool = generate_log_lhs_pool(
        nominal_initial_values=nominal,
        pool_size=DEFAULT_POOL_SIZE,
        seed=DEFAULT_VALIDATION_SEED,
    )
    validation = select_maximin_against_reference(
        pool=validation_pool,
        reference=training,
        nominal_initial_values=nominal,
        count=8,
    )

    training_frame = _condition_frame(training, spec.state_ids, "train")
    training_frame["in_train_04"] = training_frame["trajectory_index"] < 4
    training_frame["in_train_08"] = training_frame["trajectory_index"] < 8
    training_frame["in_train_16"] = training_frame["trajectory_index"] < 16
    training_frame["in_train_32"] = True
    validation_frame = _condition_frame(validation, spec.state_ids, "validation")
    nominal_series = pd.Series(nominal, index=spec.state_ids, name="nominal")
    return training_frame, validation_frame, nominal_series


def make_initial_condition_figure(
    model_id: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    nominal: pd.Series,
    output_path: Path,
) -> None:
    """Plot initial conditions as log10 fold changes from nominal values.

    Args:
        model_id: BioModels identifier.
        training: Frozen 32-condition training design.
        validation: Frozen eight-condition validation design.
        nominal: Nominal value indexed by state identifier.
        output_path: Destination PNG.
    """
    state_ids = nominal.index.astype(str).tolist()
    state_columns = [f"{state_id}_0" for state_id in state_ids]
    train_values = training[state_columns].to_numpy(dtype=float)
    validation_values = validation[state_columns].to_numpy(dtype=float)
    nominal_values = nominal.to_numpy(dtype=float)
    train_log_fold = np.log10(train_values / nominal_values[np.newaxis, :])
    validation_log_fold = np.log10(validation_values / nominal_values[np.newaxis, :])

    width = min(24.0, max(10.0, 0.38 * len(state_ids) + 5.0))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(width, 10.5),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    color_options = {
        "aspect": "auto",
        "cmap": "coolwarm",
        "vmin": -1.0,
        "vmax": 1.0,
        "interpolation": "none",
    }
    training_image = axes[0].imshow(train_log_fold, **color_options)
    axes[1].imshow(validation_log_fold, **color_options)
    for axis, labels, title in (
        (axes[0], training["trajectory_id"], "Training initial conditions"),
        (axes[1], validation["trajectory_id"], "Validation initial conditions"),
    ):
        axis.set_xticks(range(len(state_ids)), state_ids, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set_title(title)
        axis.set_xlabel("Dynamic state")
    for boundary in (3.5, 7.5, 15.5):
        axes[0].axhline(boundary, color="black", linewidth=1.2)
    axes[0].set_ylabel("Nested trajectory order")
    colorbar = figure.colorbar(training_image, ax=axes, shrink=0.72, pad=0.02)
    colorbar.set_label("log10(initial value / nominal value)")
    figure.suptitle(
        f"{model_id}: frozen development initial-condition design\n"
        "Training is nested: first 4, 8, 16, and 32 rows",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
