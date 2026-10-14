"""Estimate state derivatives from simulated time-series observations."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


DEFAULT_WINDOW_LENGTH = 11
DEFAULT_POLYORDER = 3


def estimate_trajectory_derivative(
    trajectory,
    *,
    state_column="VENUS",
    window_length=DEFAULT_WINDOW_LENGTH,
    polyorder=DEFAULT_POLYORDER,
):
    """Estimate d(state)/dt for one uniformly sampled trajectory."""
    trajectory = trajectory.sort_values(
        "time"
    ).copy()

    time_values = trajectory[
        "time"
    ].to_numpy(
        dtype=float
    )

    state_values = trajectory[
        state_column
    ].to_numpy(
        dtype=float
    )

    if len(time_values) < window_length:
        raise ValueError(
            "Trajectory is shorter than the Savitzky-Golay window."
        )

    time_steps = np.diff(
        time_values
    )

    delta_t = float(
        np.mean(time_steps)
    )

    if not np.allclose(
        time_steps,
        delta_t,
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError(
            "Savitzky-Golay derivative estimation requires "
            "uniform time sampling."
        )

    estimated_derivative = savgol_filter(
        state_values,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta_t,
        mode="interp",
    )

    trajectory[
        f"d_{state_column}_dt_estimated"
    ] = estimated_derivative

    return trajectory


def estimate_dataset_derivatives(
    frame,
    *,
    state_column="VENUS",
    window_length=DEFAULT_WINDOW_LENGTH,
    polyorder=DEFAULT_POLYORDER,
):
    """Estimate derivatives independently for every trajectory."""
    estimated_frames = []

    for _, trajectory in frame.groupby(
        "trajectory_id",
        sort=False,
    ):
        estimated_frames.append(
            estimate_trajectory_derivative(
                trajectory,
                state_column=state_column,
                window_length=window_length,
                polyorder=polyorder,
            )
        )

    return pd.concat(
        estimated_frames,
        ignore_index=True,
    )