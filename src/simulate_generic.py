"""Generic SBML simulation utilities for multistate ODE inference."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import roadrunner

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    DerivativeMethod,
)
from .derivative_generation_generic import (
    generate_derivatives,
    state_only_columns,
)
from .experimental_design_generic import ExperimentalDesign
from .model_spec import ModelSpec


@dataclass(frozen=True)
class GenericDataset:
    """Simulated train, validation, and test datasets."""

    train_04: pd.DataFrame
    train_08: pd.DataFrame
    train_16: pd.DataFrame
    train_32: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    preprocessing_seconds: float
    derivative_provenance: dict[str, object] | None = None


def _calculate_exact_derivatives(
    *,
    rr,
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    """Calculate exact RoadRunner derivatives for simulated state rows."""
    time_values = frame[
        "time"
    ].to_numpy(
        dtype=float
    )

    state_matrix = frame[
        list(spec.state_ids)
    ].to_numpy(
        dtype=float
    )

    if state_matrix.shape != (
        len(frame),
        spec.n_states,
    ):
        raise ValueError(
            "State matrix has unexpected shape: "
            f"{state_matrix.shape}."
        )

    derivative_rows = []

    for row_index, time_value in enumerate(
        time_values
    ):
        rr.time = float(
            time_value
        )

        for state_index, state_id in enumerate(
            spec.state_ids
        ):
            rr[state_id] = float(
                state_matrix[
                    row_index,
                    state_index,
                ]
            )

        rates = np.asarray(
            rr.getRatesOfChange(),
            dtype=float,
        )

        if rates.shape != (
            spec.n_states,
        ):
            raise ValueError(
                "Number of RoadRunner rates does not "
                "match the number of dynamic states. "
                f"Expected {spec.n_states}, "
                f"received shape {rates.shape}."
            )

        if not np.all(
            np.isfinite(rates)
        ):
            raise ValueError(
                "RoadRunner returned non-finite "
                f"derivatives at row {row_index}."
            )

        derivative_rows.append(
            rates
        )

    return np.vstack(
        derivative_rows
    )


def _add_exact_derivatives(
    *,
    rr,
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Add exact derivative columns to a simulated trajectory."""
    derivative_matrix = _calculate_exact_derivatives(
        rr=rr,
        frame=frame,
        spec=spec,
    )

    for state_index, state_id in enumerate(
        spec.state_ids
    ):
        frame[
            f"d_{state_id}_dt"
        ] = derivative_matrix[
            :,
            state_index,
        ]

    return frame


def simulate_nominal_trajectory(
    *,
    spec: ModelSpec,
    duration,
    points=101,
):
    """Simulate one nominal trajectory with exact state derivatives."""
    if duration <= 0:
        raise ValueError(
            "Simulation duration must be positive."
        )

    if points < 2:
        raise ValueError(
            "Simulation requires at least two time points."
        )

    rr = roadrunner.RoadRunner(
        str(spec.model_path)
    )

    rr.resetAll()

    selections = [
        "time",
        *spec.state_ids,
    ]

    result = rr.simulate(
        0.0,
        float(duration),
        int(points),
        selections=selections,
    )

    frame = pd.DataFrame(
        np.asarray(
            result,
            dtype=float,
        ),
        columns=selections,
    )

    frame = _add_exact_derivatives(
        rr=rr,
        frame=frame,
        spec=spec,
    )

    frame.insert(
        0,
        "trajectory_id",
        "nominal_000",
    )

    return frame


def _simulate_state_frame(
    *,
    spec: ModelSpec,
    initial_values,
    duration,
    points=101,
    trajectory_id="trajectory_000",
):
    """Simulate one state-only trajectory and retain its simulator."""
    initial_values = np.asarray(
        initial_values,
        dtype=float,
    )

    if initial_values.shape != (
        spec.n_states,
    ):
        raise ValueError(
            "Initial-condition vector has shape "
            f"{initial_values.shape}; expected "
            f"({spec.n_states},)."
        )

    if not np.all(
        np.isfinite(initial_values)
    ):
        raise ValueError(
            "Initial conditions must be finite."
        )

    if duration <= 0:
        raise ValueError(
            "Simulation duration must be positive."
        )

    if points < 2:
        raise ValueError(
            "Simulation requires at least two time points."
        )

    rr = roadrunner.RoadRunner(
        str(spec.model_path)
    )

    rr.resetAll()

    for state_id, initial_value in zip(
        spec.state_ids,
        initial_values,
    ):
        rr[
            f"init({state_id})"
        ] = float(
            initial_value
        )

    rr.reset()

    selections = [
        "time",
        *spec.state_ids,
    ]

    result = rr.simulate(
        0.0,
        float(duration),
        int(points),
        selections=selections,
    )

    frame = pd.DataFrame(
        np.asarray(
            result,
            dtype=float,
        ),
        columns=selections,
    )

    frame.insert(
        0,
        "trajectory_id",
        trajectory_id,
    )

    return rr, frame


def simulate_state_only_trajectory(
    *,
    spec: ModelSpec,
    initial_values,
    duration,
    points=101,
    trajectory_id="trajectory_000",
):
    """Simulate states without calculating or exposing exact derivatives.

    Args:
        spec: Model metadata and source SBML path.
        initial_values: Initial dynamic-state values in ModelSpec order.
        duration: Positive simulation horizon.
        points: Number of output samples.
        trajectory_id: Stable trajectory identifier.

    Returns:
        A state-only table containing trajectory ID, time, and states.
    """
    _, frame = _simulate_state_frame(
        spec=spec,
        initial_values=initial_values,
        duration=duration,
        points=points,
        trajectory_id=trajectory_id,
    )

    return frame


def simulate_trajectory(
    *,
    spec: ModelSpec,
    initial_values,
    duration,
    points=101,
    trajectory_id="trajectory_000",
):
    """Simulate one trajectory with exact derivatives for compatibility.

    Args:
        spec: Model metadata and source SBML path.
        initial_values: Initial dynamic-state values in ModelSpec order.
        duration: Positive simulation horizon.
        points: Number of output samples.
        trajectory_id: Stable trajectory identifier.

    Returns:
        The historical generic trajectory schema with exact targets.
    """
    rr, state_frame = _simulate_state_frame(
        spec=spec,
        initial_values=initial_values,
        duration=duration,
        points=points,
        trajectory_id=trajectory_id,
    )

    generated = generate_derivatives(
        frame=state_frame,
        spec=spec,
        variant=EXACT_BENCHMARK_VARIANT,
        exact_generator=lambda frame: _add_exact_derivatives(
            rr=rr,
            frame=frame,
            spec=spec,
        ),
    )

    return generated.frame


def _simulate_initial_condition_set(
    *,
    spec: ModelSpec,
    initial_conditions,
    duration,
    points,
    trajectory_prefix,
    state_only,
):
    """Simulate an initial-condition matrix with a selected output schema."""
    initial_conditions = np.asarray(
        initial_conditions,
        dtype=float,
    )

    if initial_conditions.ndim != 2:
        raise ValueError(
            "Initial-condition set must be a two-dimensional array."
        )

    expected_shape = (
        len(initial_conditions),
        spec.n_states,
    )

    if initial_conditions.shape != expected_shape:
        raise ValueError(
            "Initial-condition matrix has an unexpected shape."
        )

    frames = []

    for index, initial_values in enumerate(
        initial_conditions
    ):
        trajectory_id = (
            f"{trajectory_prefix}_{index:03d}"
        )

        simulation_function = (
            simulate_state_only_trajectory
            if state_only
            else simulate_trajectory
        )

        frame = simulation_function(
            spec=spec,
            initial_values=initial_values,
            duration=duration,
            points=points,
            trajectory_id=trajectory_id,
        )

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def simulate_state_only_initial_condition_set(
    *,
    spec: ModelSpec,
    initial_conditions,
    duration,
    points=101,
    trajectory_prefix="trajectory",
):
    """Simulate multiple trajectories without calculating exact rates.

    Args:
        spec: Model metadata and source SBML path.
        initial_conditions: Two-dimensional initial-condition matrix.
        duration: Positive simulation horizon.
        points: Number of output samples per trajectory.
        trajectory_prefix: Prefix used to build trajectory identifiers.

    Returns:
        Concatenated state-only trajectories.
    """
    return _simulate_initial_condition_set(
        spec=spec,
        initial_conditions=initial_conditions,
        duration=duration,
        points=points,
        trajectory_prefix=trajectory_prefix,
        state_only=True,
    )


def simulate_initial_condition_set(
    *,
    spec: ModelSpec,
    initial_conditions,
    duration,
    points=101,
    trajectory_prefix="trajectory",
):
    """Simulate trajectories with exact derivatives for compatibility.

    Args:
        spec: Model metadata and source SBML path.
        initial_conditions: Two-dimensional initial-condition matrix.
        duration: Positive simulation horizon.
        points: Number of output samples per trajectory.
        trajectory_prefix: Prefix used to build trajectory identifiers.

    Returns:
        Concatenated trajectories with exact derivative targets.
    """
    return _simulate_initial_condition_set(
        spec=spec,
        initial_conditions=initial_conditions,
        duration=duration,
        points=points,
        trajectory_prefix=trajectory_prefix,
        state_only=False,
    )


def build_generic_development_dataset(
    *,
    spec: ModelSpec,
    design: ExperimentalDesign,
    duration,
    points=101,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
):
    """Build development data for a validated derivative variant.

    Savitzky-Golay mode simulates only time and state columns, generates
    targets once for the complete 32-trajectory training design, and never
    exposes exact rates to the estimator. Exact mode preserves the historical
    generic dataset behavior for callers that omit ``variant``.

    Args:
        spec: Model metadata and source SBML path.
        design: Frozen generic train, validation, and test IC design.
        duration: Positive simulation horizon.
        points: Number of output samples per trajectory.
        variant: Validated exact or Savitzky-Golay benchmark variant.

    Returns:
        Nested training data, validation data, an empty test frame, and
        derivative provenance.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")

    start_time = perf_counter()

    if variant.derivative_method is DerivativeMethod.EXACT:
        train_32 = simulate_initial_condition_set(
            spec=spec,
            initial_conditions=design.train,
            duration=duration,
            points=points,
            trajectory_prefix="train",
        )

        validation = simulate_initial_condition_set(
            spec=spec,
            initial_conditions=design.validation,
            duration=duration,
            points=points,
            trajectory_prefix="validation",
        )

        derivative_provenance = variant.derivative_provenance(
            input_columns=state_only_columns(spec),
            state_order=spec.state_ids,
        )
    elif variant.derivative_method is DerivativeMethod.SAVGOL:
        train_state_frame = simulate_state_only_initial_condition_set(
            spec=spec,
            initial_conditions=design.train,
            duration=duration,
            points=points,
            trajectory_prefix="train",
        )

        validation = simulate_state_only_initial_condition_set(
            spec=spec,
            initial_conditions=design.validation,
            duration=duration,
            points=points,
            trajectory_prefix="validation",
        )

        generated = generate_derivatives(
            frame=train_state_frame,
            spec=spec,
            variant=variant,
        )

        train_32 = generated.frame
        derivative_provenance = generated.provenance
    else:
        raise ValueError(
            f"Unsupported derivative method: {variant.derivative_method}."
        )

    train_ids = list(
        train_32[
            "trajectory_id"
        ].drop_duplicates()
    )

    def subset_training(
        count,
    ):
        selected_ids = set(
            train_ids[:count]
        )

        return (
            train_32[
                train_32[
                    "trajectory_id"
                ].isin(
                    selected_ids
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    train_04 = subset_training(
        4
    )

    train_08 = subset_training(
        8
    )

    train_16 = subset_training(
        16
    )

    preprocessing_seconds = (
        perf_counter()
        - start_time
    )

    return GenericDataset(
        train_04=train_04,
        train_08=train_08,
        train_16=train_16,
        train_32=train_32,
        validation=validation,
        test=pd.DataFrame(),
        preprocessing_seconds=preprocessing_seconds,
        derivative_provenance=derivative_provenance,
    )


def build_generic_dataset(
    *,
    spec: ModelSpec,
    design: ExperimentalDesign,
    duration,
    points=101,
):
    """Build all generic train, validation, and test trajectory datasets."""
    start_time = perf_counter()

    train_32 = simulate_initial_condition_set(
        spec=spec,
        initial_conditions=design.train,
        duration=duration,
        points=points,
        trajectory_prefix="train",
    )

    validation = simulate_initial_condition_set(
        spec=spec,
        initial_conditions=design.validation,
        duration=duration,
        points=points,
        trajectory_prefix="validation",
    )

    test = simulate_initial_condition_set(
        spec=spec,
        initial_conditions=design.test,
        duration=duration,
        points=points,
        trajectory_prefix="test",
    )

    train_ids = list(
        train_32[
            "trajectory_id"
        ].drop_duplicates()
    )

    def subset_training(
        count,
    ):
        selected_ids = set(
            train_ids[:count]
        )

        return (
            train_32[
                train_32[
                    "trajectory_id"
                ].isin(
                    selected_ids
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    train_04 = subset_training(
        4
    )

    train_08 = subset_training(
        8
    )

    train_16 = subset_training(
        16
    )

    preprocessing_seconds = (
        perf_counter()
        - start_time
    )

    return GenericDataset(
        train_04=train_04,
        train_08=train_08,
        train_16=train_16,
        train_32=train_32,
        validation=validation,
        test=test,
        preprocessing_seconds=preprocessing_seconds,
        derivative_provenance=(
            EXACT_BENCHMARK_VARIANT.derivative_provenance(
                input_columns=state_only_columns(spec),
                state_order=spec.state_ids,
            )
        ),
    )


def summarize_trajectory(
    *,
    frame,
    spec: ModelSpec,
):
    """Print a compact summary of a simulated trajectory."""
    print()
    print("=" * 70)
    print("GENERIC NOMINAL TRAJECTORY")
    print("=" * 70)

    print(
        "Model:",
        spec.model_id,
    )

    print(
        "States:",
        spec.state_ids,
    )

    print(
        "Rows:",
        len(frame),
    )

    print()

    print(
        frame.head().to_string(
            index=False
        )
    )

    print()
    print("State ranges:")

    for state_id in spec.state_ids:
        values = frame[
            state_id
        ]

        print(
            f"  {state_id}: "
            f"{values.min():.8g} .. "
            f"{values.max():.8g}"
        )

    print()
    print("Derivative ranges:")

    for state_id in spec.state_ids:
        derivative_column = (
            f"d_{state_id}_dt"
        )

        values = frame[
            derivative_column
        ]

        print(
            f"  {derivative_column}: "
            f"{values.min():.8g} .. "
            f"{values.max():.8g}"
        )
