"""Generate generic multistate derivative targets with a strict boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .benchmark_variant import (
    BenchmarkVariant,
    DerivativeMethod,
    SavgolDerivativeConfig,
)

if TYPE_CHECKING:
    from .model_spec import ModelSpec


UNIFORM_TIME_RTOL = 1e-8
UNIFORM_TIME_ATOL = 1e-12


@dataclass(frozen=True)
class DerivativeGenerationResult:
    """Derivative-enriched frame and reproducibility provenance."""

    frame: pd.DataFrame
    provenance: dict[str, object]


def state_only_columns(spec: "ModelSpec") -> tuple[str, ...]:
    """Return the only columns permitted at the SG information boundary.

    Args:
        spec: Model metadata defining dynamic-state order.

    Returns:
        Ordered trajectory, time, and state column names.
    """
    return (
        "trajectory_id",
        "time",
        *spec.state_ids,
    )


def derivative_columns(spec: "ModelSpec") -> tuple[str, ...]:
    """Return ordered generic derivative-target column names.

    Args:
        spec: Model metadata defining dynamic-state order.

    Returns:
        Ordered derivative-target column names.
    """
    return tuple(f"d_{state_id}_dt" for state_id in spec.state_ids)


def _validate_state_only_frame(
    *,
    frame: pd.DataFrame,
    spec: "ModelSpec",
) -> None:
    """Validate the strict state-only derivative-generation input schema."""
    expected_columns = state_only_columns(spec)
    observed_columns = tuple(str(column) for column in frame.columns)

    unexpected_derivatives = [
        column
        for column in observed_columns
        if column not in expected_columns and column.startswith("d_")
    ]

    if unexpected_derivatives:
        raise ValueError(
            "Derivative-generation input must not contain derivative "
            f"columns: {unexpected_derivatives}."
        )

    if observed_columns != expected_columns:
        raise ValueError(
            "Derivative-generation input columns must be exactly "
            f"{list(expected_columns)} in that order; received "
            f"{list(observed_columns)}."
        )

    if frame.empty:
        raise ValueError("Derivative-generation input must not be empty.")

    if frame["trajectory_id"].isna().any():
        raise ValueError("trajectory_id values must not be missing.")

    numeric_columns = ["time", *spec.state_ids]
    numeric_values = frame[numeric_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("Time and state inputs must all be finite.")


def _estimate_savgol_trajectory(
    *,
    trajectory: pd.DataFrame,
    spec: "ModelSpec",
    config: SavgolDerivativeConfig,
) -> pd.DataFrame:
    """Estimate every state derivative within one isolated trajectory."""
    ordered = trajectory.sort_values("time", kind="stable").reset_index(drop=True)

    if len(ordered) < config.window_length:
        raise ValueError(
            "Trajectory is shorter than the frozen Savitzky-Golay "
            f"window ({len(ordered)} < {config.window_length})."
        )

    time_values = ordered["time"].to_numpy(dtype=float)
    time_steps = np.diff(time_values)

    if np.any(time_steps <= 0.0):
        raise ValueError("Trajectory times must be unique and strictly increasing.")

    delta = float(np.mean(time_steps))

    if not np.allclose(
        time_steps,
        delta,
        rtol=UNIFORM_TIME_RTOL,
        atol=UNIFORM_TIME_ATOL,
    ):
        raise ValueError(
            "Savitzky-Golay derivative estimation requires uniform "
            "time sampling within each trajectory."
        )

    for state_id in spec.state_ids:
        state_values = ordered[state_id].to_numpy(dtype=float)
        estimates = savgol_filter(
            state_values,
            window_length=config.window_length,
            polyorder=config.polyorder,
            deriv=config.deriv,
            delta=delta,
            mode=config.mode,
        )

        if not np.all(np.isfinite(estimates)):
            raise ValueError(
                f"Non-finite Savitzky-Golay estimates for state {state_id}."
            )

        ordered[f"d_{state_id}_dt"] = estimates

    return ordered


def generate_savgol_derivatives(
    *,
    frame: pd.DataFrame,
    spec: "ModelSpec",
    variant: BenchmarkVariant,
) -> DerivativeGenerationResult:
    """Generate SG targets from only trajectory IDs, time, and states.

    The function has no RoadRunner dependency. Its strict input schema prevents
    exact derivative columns from crossing the Savitzky-Golay information
    boundary.

    Args:
        frame: State-only trajectories in the required column order.
        spec: Model metadata defining state names and order.
        variant: Validated frozen Savitzky-Golay benchmark variant.

    Returns:
        Derivative-enriched trajectories and serializable provenance.
    """
    if variant.derivative_method is not DerivativeMethod.SAVGOL:
        raise ValueError("generate_savgol_derivatives requires SG variant.")

    if not isinstance(variant.derivative_config, SavgolDerivativeConfig):
        raise TypeError("SG variant has an invalid derivative configuration.")

    _validate_state_only_frame(frame=frame, spec=spec)

    estimated_trajectories = []

    for _, trajectory in frame.groupby("trajectory_id", sort=False):
        estimated_trajectories.append(
            _estimate_savgol_trajectory(
                trajectory=trajectory,
                spec=spec,
                config=variant.derivative_config,
            )
        )

    result_frame = pd.concat(estimated_trajectories, ignore_index=True)
    expected_output_columns = (
        *state_only_columns(spec),
        *derivative_columns(spec),
    )

    if tuple(result_frame.columns) != expected_output_columns:
        raise RuntimeError(
            "Savitzky-Golay output schema does not match the generic derivative schema."
        )

    provenance = variant.derivative_provenance(
        input_columns=state_only_columns(spec),
        state_order=spec.state_ids,
    )

    return DerivativeGenerationResult(
        frame=result_frame,
        provenance=provenance,
    )


def generate_derivatives(
    *,
    frame: pd.DataFrame,
    spec: "ModelSpec",
    variant: BenchmarkVariant,
    exact_generator: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> DerivativeGenerationResult:
    """Generate exact or SG targets through one validated interface.

    Args:
        frame: State-only trajectories.
        spec: Model metadata defining state names and order.
        variant: Validated benchmark derivative variant.
        exact_generator: Exact-rate callback supplied only by simulation code.

    Returns:
        Derivative-enriched trajectories and serializable provenance.
    """
    _validate_state_only_frame(frame=frame, spec=spec)

    if variant.derivative_method is DerivativeMethod.SAVGOL:
        if exact_generator is not None:
            raise ValueError("An exact-rate generator must not be exposed to SG mode.")

        return generate_savgol_derivatives(
            frame=frame,
            spec=spec,
            variant=variant,
        )

    if variant.derivative_method is not DerivativeMethod.EXACT:
        raise ValueError(f"Unsupported derivative method: {variant.derivative_method}.")

    if exact_generator is None:
        raise ValueError("Exact derivative generation requires a callback.")

    result_frame = exact_generator(frame.copy())
    expected_output_columns = (
        *state_only_columns(spec),
        *derivative_columns(spec),
    )

    if tuple(result_frame.columns) != expected_output_columns:
        raise ValueError(
            "Exact derivative generator returned columns "
            f"{list(result_frame.columns)}; expected "
            f"{list(expected_output_columns)}."
        )

    target_values = result_frame[list(derivative_columns(spec))].to_numpy(dtype=float)

    if not np.all(np.isfinite(target_values)):
        raise ValueError("Exact derivative targets must all be finite.")

    provenance = variant.derivative_provenance(
        input_columns=state_only_columns(spec),
        state_order=spec.state_ids,
    )

    return DerivativeGenerationResult(
        frame=result_frame,
        provenance=provenance,
    )
