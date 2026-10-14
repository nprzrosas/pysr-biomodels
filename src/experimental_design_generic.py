"""Generic multidimensional initial-condition experimental design."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .model_spec import ModelSpec


DEFAULT_LOG10_LOW = -1.0
DEFAULT_LOG10_HIGH = 1.0
DEFAULT_POOL_SIZE = 256

DEFAULT_TRAIN_SIZE = 32
DEFAULT_VALIDATION_SIZE = 8
DEFAULT_TEST_SIZE = 8

DEFAULT_TRAIN_SEED = 1010
DEFAULT_VALIDATION_SEED = 2020
DEFAULT_TEST_SEED = 3030


@dataclass(frozen=True)
class ExperimentalDesign:
    """Initial-condition design for train, validation, and test."""

    state_ids: tuple[str, ...]
    nominal_initial_values: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def generate_log_lhs_pool(
    *,
    nominal_initial_values,
    pool_size,
    seed,
    log10_low=DEFAULT_LOG10_LOW,
    log10_high=DEFAULT_LOG10_HIGH,
):
    """Generate multiplicative initial conditions using log-space LHS."""
    nominal = np.asarray(
        nominal_initial_values,
        dtype=float,
    )

    if nominal.ndim != 1:
        raise ValueError(
            "Nominal initial values must be one-dimensional."
        )

    if np.any(nominal <= 0.0):
        raise ValueError(
            "Log-multiplicative initial-condition design requires "
            "strictly positive nominal initial values."
        )

    sampler = qmc.LatinHypercube(
        d=len(nominal),
        seed=seed,
    )

    unit_samples = sampler.random(
        n=pool_size
    )

    log_factors = qmc.scale(
        unit_samples,
        [log10_low] * len(nominal),
        [log10_high] * len(nominal),
    )

    factors = np.power(
        10.0,
        log_factors,
    )

    return (
        nominal[np.newaxis, :]
        * factors
    )


def log10_coordinates(
    *,
    points,
    nominal_initial_values,
):
    """Express initial conditions as log10 fold changes from nominal."""
    points = np.asarray(
        points,
        dtype=float,
    )

    nominal = np.asarray(
        nominal_initial_values,
        dtype=float,
    )

    if np.any(points <= 0.0):
        raise ValueError(
            "Log-distance calculations require positive values."
        )

    return np.log10(
        points
        / nominal[np.newaxis, :]
    )


def minimum_distances(
    *,
    candidates,
    selected,
    nominal_initial_values,
):
    """Calculate each candidate's minimum log-space distance to selected points."""
    candidate_coordinates = log10_coordinates(
        points=candidates,
        nominal_initial_values=nominal_initial_values,
    )

    selected_coordinates = log10_coordinates(
        points=selected,
        nominal_initial_values=nominal_initial_values,
    )

    differences = (
        candidate_coordinates[:, np.newaxis, :]
        - selected_coordinates[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    return distances.min(
        axis=1
    )


def maximin_order(
    *,
    pool,
    nominal_initial_values,
    count,
):
    """Select and order points greedily using log-space maximin distance."""
    pool = np.asarray(
        pool,
        dtype=float,
    )

    nominal = np.asarray(
        nominal_initial_values,
        dtype=float,
    )

    if count > len(pool):
        raise ValueError(
            "Requested more points than available in the pool."
        )

    nominal_coordinates = log10_coordinates(
        points=nominal[np.newaxis, :],
        nominal_initial_values=nominal,
    )[0]

    pool_coordinates = log10_coordinates(
        points=pool,
        nominal_initial_values=nominal,
    )

    first_distances = np.linalg.norm(
        pool_coordinates
        - nominal_coordinates[np.newaxis, :],
        axis=1,
    )

    first_index = int(
        np.argmax(first_distances)
    )

    selected_indices = [
        first_index
    ]

    available_indices = [
        index
        for index in range(len(pool))
        if index != first_index
    ]

    while len(selected_indices) < count:
        selected = pool[
            selected_indices
        ]

        candidates = pool[
            available_indices
        ]

        distances = minimum_distances(
            candidates=candidates,
            selected=selected,
            nominal_initial_values=nominal,
        )

        best_position = int(
            np.argmax(distances)
        )

        best_index = available_indices.pop(
            best_position
        )

        selected_indices.append(
            best_index
        )

    return pool[
        selected_indices
    ]


def select_maximin_against_reference(
    *,
    pool,
    reference,
    nominal_initial_values,
    count,
):
    """Select points maximizing distance from a fixed and growing reference set."""
    pool = np.asarray(
        pool,
        dtype=float,
    )

    reference = np.asarray(
        reference,
        dtype=float,
    )

    selected = []
    available = pool.copy()

    while len(selected) < count:
        if selected:
            comparison = np.vstack(
                [
                    reference,
                    np.asarray(selected),
                ]
            )
        else:
            comparison = reference

        distances = minimum_distances(
            candidates=available,
            selected=comparison,
            nominal_initial_values=nominal_initial_values,
        )

        best_index = int(
            np.argmax(distances)
        )

        selected.append(
            available[best_index].copy()
        )

        available = np.delete(
            available,
            best_index,
            axis=0,
        )

    return np.asarray(
        selected
    )


def build_experimental_design(
    *,
    spec: ModelSpec,
):
    """Build frozen multidimensional train, validation, and test IC sets."""
    nominal = np.asarray(
        [
            spec.initial_values[state_id]
            for state_id in spec.state_ids
        ],
        dtype=float,
    )

    if np.any(nominal <= 0.0):
        raise ValueError(
            f"Model {spec.model_id} contains zero or negative "
            "nominal state values and is not supported by the "
            "current log-multiplicative design."
        )

    train_pool = generate_log_lhs_pool(
        nominal_initial_values=nominal,
        pool_size=DEFAULT_POOL_SIZE,
        seed=DEFAULT_TRAIN_SEED,
    )

    train = maximin_order(
        pool=train_pool,
        nominal_initial_values=nominal,
        count=DEFAULT_TRAIN_SIZE,
    )

    validation_pool = generate_log_lhs_pool(
        nominal_initial_values=nominal,
        pool_size=DEFAULT_POOL_SIZE,
        seed=DEFAULT_VALIDATION_SEED,
    )

    validation = select_maximin_against_reference(
        pool=validation_pool,
        reference=train,
        nominal_initial_values=nominal,
        count=DEFAULT_VALIDATION_SIZE,
    )

    test_pool = generate_log_lhs_pool(
        nominal_initial_values=nominal,
        pool_size=DEFAULT_POOL_SIZE,
        seed=DEFAULT_TEST_SEED,
    )

    test = select_maximin_against_reference(
        pool=test_pool,
        reference=np.vstack(
            [
                train,
                validation,
            ]
        ),
        nominal_initial_values=nominal,
        count=DEFAULT_TEST_SIZE,
    )

    return ExperimentalDesign(
        state_ids=spec.state_ids,
        nominal_initial_values=nominal,
        train=train,
        validation=validation,
        test=test,
    )