"""Add controlled measurement noise to simulated state trajectories."""

import numpy as np
import pandas as pd


DEFAULT_NOISE_SEED = 4040


def calculate_training_state_scale(
    train_frame,
    state_column="VENUS",
):
    """Calculate the noise scale using training data only."""
    values = train_frame[
        state_column
    ].to_numpy(
        dtype=float
    )

    scale = np.max(
        np.abs(values)
    )

    if not np.isfinite(scale):
        raise ValueError(
            "Training state scale is not finite."
        )

    if scale <= 0:
        raise ValueError(
            "Training state scale must be positive."
        )

    return float(scale)


def add_measurement_noise(
    frame,
    *,
    state_scale,
    noise_fraction,
    seed=DEFAULT_NOISE_SEED,
    state_column="VENUS",
    output_column="VENUS_observed",
):
    """Add Gaussian measurement noise to one state variable."""
    if noise_fraction < 0:
        raise ValueError(
            "noise_fraction must be non-negative."
        )

    if state_scale <= 0:
        raise ValueError(
            "state_scale must be positive."
        )

    result = frame.copy()

    clean_values = result[
        state_column
    ].to_numpy(
        dtype=float
    )

    noise_std = (
        float(noise_fraction)
        * float(state_scale)
    )

    if noise_fraction == 0:
        noise = np.zeros_like(
            clean_values
        )
    else:
        rng = np.random.default_rng(
            seed
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(clean_values),
        )

    result[
        output_column
    ] = clean_values + noise

    result[
        "measurement_noise"
    ] = noise

    result[
        "noise_fraction"
    ] = float(
        noise_fraction
    )

    result[
        "noise_std"
    ] = noise_std

    return result