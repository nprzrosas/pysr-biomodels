"""Numerically validate symbolic SBML ground truth against RoadRunner.

This module verifies that the symbolic ODE right-hand sides reconstructed by
``extract_mechanistic_ground_truth`` reproduce RoadRunner's numerical rates of
change at states visited by simulated trajectories.

Both autonomous systems,

    dX/dt = f(X),

and explicitly time-dependent systems,

    dX/dt = f(t, X),

are supported.

This validation is strictly post-hoc. It must not influence PySR fitting,
candidate generation, Selector v1, or held-out test selection.
"""

from dataclasses import dataclass

import numpy as np
import roadrunner
import sympy as sp

from src.extract_mechanistic_ground_truth import (
    MechanisticGroundTruth,
    extract_mechanistic_ground_truth,
)
from src.model_registry import resolve_model_path
from src.simulation_metadata import load_simulation_time


DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12
DEFAULT_SIMULATION_POINTS = 101
DEFAULT_VALIDATION_POINTS = 11


@dataclass(frozen=True)
class ValidationPointResult:
    """Comparison between symbolic and RoadRunner rates at one state."""

    point_index: int
    time: float
    state_values: tuple[float, ...]
    roadrunner_rates: tuple[float, ...]
    symbolic_rates: tuple[float, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    all_close: bool


@dataclass(frozen=True)
class MechanisticValidationResult:
    """Summary of symbolic ground-truth numerical validation."""

    model_id: str
    state_ids: tuple[str, ...]
    extraction_status: str
    numerical_validation_status: str
    n_points_checked: int
    maximum_absolute_error: float
    maximum_relative_error: float
    rtol: float
    atol: float
    point_results: tuple[ValidationPointResult, ...]


def get_roadrunner_state_ids(
    rr,
):
    """Return RoadRunner floating-species IDs in numerical state order."""
    return tuple(
        str(state_id)
        for state_id
        in rr.model.getFloatingSpeciesIds()
    )


def validate_state_order(
    *,
    ground_truth,
    roadrunner_state_ids,
):
    """Require symbolic and RoadRunner dynamic-state sets to agree."""
    symbolic_state_ids = tuple(
        ground_truth.state_ids
    )

    roadrunner_state_ids = tuple(
        roadrunner_state_ids
    )

    if symbolic_state_ids == roadrunner_state_ids:
        return

    if set(
        symbolic_state_ids
    ) != set(
        roadrunner_state_ids
    ):
        raise ValueError(
            "Symbolic and RoadRunner dynamic-state sets differ.\n"
            f"Symbolic: {symbolic_state_ids}\n"
            f"RoadRunner: {roadrunner_state_ids}"
        )


def compile_symbolic_rhs(
    *,
    ground_truth,
    state_ids,
):
    """Compile symbolic RHS expressions as functions of time and state."""
    time_symbol = sp.Symbol(
        "time"
    )

    state_symbols = tuple(
        sp.Symbol(
            state_id
        )
        for state_id in state_ids
    )

    arguments = (
        time_symbol,
        *state_symbols,
    )

    allowed_symbols = set(
        state_symbols
    ) | {
        time_symbol,
    }

    functions = {}

    for state_id in state_ids:
        if state_id not in ground_truth.rhs_by_state:
            raise ValueError(
                f"No symbolic RHS found for state {state_id}."
            )

        expression = ground_truth.rhs_by_state[
            state_id
        ]

        unsupported_symbols = (
            expression.free_symbols
            - allowed_symbols
        )

        if unsupported_symbols:
            raise ValueError(
                f"Symbolic RHS for {state_id} contains "
                "unsupported free symbols: "
                + ", ".join(
                    sorted(
                        str(symbol)
                        for symbol
                        in unsupported_symbols
                    )
                )
            )

        functions[
            state_id
        ] = sp.lambdify(
            arguments,
            expression,
            modules="numpy",
        )

    return functions


def evaluate_symbolic_rates(
    *,
    symbolic_functions,
    state_ids,
    time,
    state_values,
):
    """Evaluate every symbolic RHS at one time and numerical state."""
    arguments = (
        float(
            time
        ),
        *(
            float(value)
            for value in state_values
        ),
    )

    rates = []

    for state_id in state_ids:
        value = symbolic_functions[
            state_id
        ](
            *arguments
        )

        value = float(
            np.asarray(
                value
            )
        )

        rates.append(
            value
        )

    return np.asarray(
        rates,
        dtype=float,
    )


def set_roadrunner_state(
    *,
    rr,
    state_ids,
    state_values,
):
    """Set RoadRunner floating-species concentrations."""
    if len(
        state_ids
    ) != len(
        state_values
    ):
        raise ValueError(
            "State ID and state-value lengths differ."
        )

    for state_id, value in zip(
        state_ids,
        state_values,
    ):
        rr[
            state_id
        ] = float(
            value
        )


def compute_relative_errors(
    *,
    reference,
    candidate,
    atol,
):
    """Compute stable element-wise relative errors."""
    reference = np.asarray(
        reference,
        dtype=float,
    )

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    denominator = np.maximum(
        np.abs(
            reference
        ),
        float(
            atol
        ),
    )

    return (
        np.abs(
            reference - candidate
        )
        / denominator
    )


def choose_validation_indices(
    *,
    number_of_rows,
    number_of_validation_points,
):
    """Choose approximately evenly spaced trajectory rows."""
    if number_of_rows < 1:
        raise ValueError(
            "Trajectory contains no rows."
        )

    if number_of_validation_points < 1:
        raise ValueError(
            "number_of_validation_points must be positive."
        )

    number_of_validation_points = min(
        number_of_validation_points,
        number_of_rows,
    )

    indices = np.linspace(
        0,
        number_of_rows - 1,
        num=number_of_validation_points,
        dtype=int,
    )

    return np.unique(
        indices
    )


def simulate_nominal_trajectory(
    *,
    rr,
    duration,
    simulation_points,
):
    """Simulate one nominal trajectory with RoadRunner."""
    if duration <= 0:
        raise ValueError(
            f"Duration must be positive, received {duration}."
        )

    if simulation_points < 2:
        raise ValueError(
            "simulation_points must be at least 2."
        )

    rr.reset()

    result = rr.simulate(
        0.0,
        float(
            duration
        ),
        int(
            simulation_points
        ),
    )

    return np.asarray(
        result,
        dtype=float,
    )


def extract_simulation_columns(
    *,
    rr,
    simulation,
    state_ids,
):
    """Extract time and dynamic states from a RoadRunner simulation."""
    column_names = [
        str(name)
        for name in rr.timeCourseSelections
    ]

    if simulation.ndim != 2:
        raise ValueError(
            "RoadRunner simulation result is not two-dimensional."
        )

    if len(
        column_names
    ) != simulation.shape[1]:
        raise ValueError(
            "RoadRunner selection count does not match "
            "simulation column count."
        )

    if "time" not in column_names:
        raise ValueError(
            "RoadRunner simulation does not contain time."
        )

    time_index = column_names.index(
        "time"
    )

    state_indices = []

    for state_id in state_ids:
        concentration_name = (
            f"[{state_id}]"
        )

        if concentration_name in column_names:
            state_indices.append(
                column_names.index(
                    concentration_name
                )
            )

        elif state_id in column_names:
            state_indices.append(
                column_names.index(
                    state_id
                )
            )

        else:
            raise ValueError(
                f"State {state_id} not found in RoadRunner "
                f"simulation selections: {column_names}"
            )

    times = simulation[
        :,
        time_index,
    ]

    states = simulation[
        :,
        state_indices,
    ]

    return (
        times,
        states,
    )


def compare_one_state(
    *,
    rr,
    ground_truth,
    symbolic_functions,
    state_ids,
    point_index,
    time,
    state_values,
    rtol,
    atol,
):
    """Compare symbolic and RoadRunner rates at one time and state."""
    rr.model.setTime(
        float(
            time
        )
    )

    set_roadrunner_state(
        rr=rr,
        state_ids=state_ids,
        state_values=state_values,
    )

    roadrunner_rates = np.asarray(
        rr.getRatesOfChange(),
        dtype=float,
    )

    symbolic_rates = evaluate_symbolic_rates(
        symbolic_functions=symbolic_functions,
        state_ids=state_ids,
        time=time,
        state_values=state_values,
    )

    if roadrunner_rates.shape != symbolic_rates.shape:
        raise ValueError(
            "RoadRunner and symbolic rate vectors have "
            "different shapes."
        )

    if not np.all(
        np.isfinite(
            roadrunner_rates
        )
    ):
        raise ValueError(
            f"Non-finite RoadRunner rate at validation point "
            f"{point_index}."
        )

    if not np.all(
        np.isfinite(
            symbolic_rates
        )
    ):
        raise ValueError(
            f"Non-finite symbolic rate at validation point "
            f"{point_index}."
        )

    absolute_errors = np.abs(
        roadrunner_rates
        - symbolic_rates
    )

    relative_errors = compute_relative_errors(
        reference=roadrunner_rates,
        candidate=symbolic_rates,
        atol=atol,
    )

    maximum_absolute_error = float(
        np.max(
            absolute_errors
        )
    )

    maximum_relative_error = float(
        np.max(
            relative_errors
        )
    )

    all_close = bool(
        np.allclose(
            roadrunner_rates,
            symbolic_rates,
            rtol=rtol,
            atol=atol,
        )
    )

    return ValidationPointResult(
        point_index=int(
            point_index
        ),
        time=float(
            time
        ),
        state_values=tuple(
            float(value)
            for value in state_values
        ),
        roadrunner_rates=tuple(
            float(value)
            for value in roadrunner_rates
        ),
        symbolic_rates=tuple(
            float(value)
            for value in symbolic_rates
        ),
        maximum_absolute_error=(
            maximum_absolute_error
        ),
        maximum_relative_error=(
            maximum_relative_error
        ),
        all_close=all_close,
    )


def validate_ground_truth_on_trajectory(
    *,
    model_id,
    ground_truth,
    rr,
    duration,
    simulation_points=DEFAULT_SIMULATION_POINTS,
    validation_points=DEFAULT_VALIDATION_POINTS,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    """Validate symbolic RHS at times and states from a nominal trajectory."""
    state_ids = get_roadrunner_state_ids(
        rr
    )

    validate_state_order(
        ground_truth=ground_truth,
        roadrunner_state_ids=state_ids,
    )

    symbolic_functions = compile_symbolic_rhs(
        ground_truth=ground_truth,
        state_ids=state_ids,
    )

    simulation = simulate_nominal_trajectory(
        rr=rr,
        duration=duration,
        simulation_points=simulation_points,
    )

    (
        times,
        states,
    ) = extract_simulation_columns(
        rr=rr,
        simulation=simulation,
        state_ids=state_ids,
    )

    validation_indices = (
        choose_validation_indices(
            number_of_rows=len(
                times
            ),
            number_of_validation_points=(
                validation_points
            ),
        )
    )

    point_results = []

    for point_number, row_index in enumerate(
        validation_indices,
        start=1,
    ):
        point_result = compare_one_state(
            rr=rr,
            ground_truth=ground_truth,
            symbolic_functions=symbolic_functions,
            state_ids=state_ids,
            point_index=point_number,
            time=times[
                row_index
            ],
            state_values=states[
                row_index
            ],
            rtol=rtol,
            atol=atol,
        )

        point_results.append(
            point_result
        )

    all_close = all(
        result.all_close
        for result in point_results
    )

    maximum_absolute_error = max(
        result.maximum_absolute_error
        for result in point_results
    )

    maximum_relative_error = max(
        result.maximum_relative_error
        for result in point_results
    )

    numerical_validation_status = (
        "passed"
        if all_close
        else "failed"
    )

    return MechanisticValidationResult(
        model_id=model_id,
        state_ids=state_ids,
        extraction_status="success",
        numerical_validation_status=(
            numerical_validation_status
        ),
        n_points_checked=len(
            point_results
        ),
        maximum_absolute_error=float(
            maximum_absolute_error
        ),
        maximum_relative_error=float(
            maximum_relative_error
        ),
        rtol=float(
            rtol
        ),
        atol=float(
            atol
        ),
        point_results=tuple(
            point_results
        ),
    )


def validate_mechanistic_ground_truth(
    *,
    model_id,
    simulation_points=DEFAULT_SIMULATION_POINTS,
    validation_points=DEFAULT_VALIDATION_POINTS,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    """Extract and numerically validate mechanistic ground truth."""
    model_path = resolve_model_path(
        model_id
    )

    ground_truth = (
        extract_mechanistic_ground_truth(
            model_id=model_id,
            model_path=model_path,
        )
    )

    rr = roadrunner.RoadRunner(
        str(
            model_path
        )
    )

    duration = load_simulation_time(
        model_id
    )

    validation = (
        validate_ground_truth_on_trajectory(
            model_id=model_id,
            ground_truth=ground_truth,
            rr=rr,
            duration=duration,
            simulation_points=(
                simulation_points
            ),
            validation_points=(
                validation_points
            ),
            rtol=rtol,
            atol=atol,
        )
    )

    return (
        ground_truth,
        validation,
    )


def print_validation_result(
    validation,
):
    """Print a human-readable validation report."""
    print(
        "MODEL"
    )

    print(
        validation.model_id
    )

    print(
        "\nSTATE IDS"
    )

    print(
        list(
            validation.state_ids
        )
    )

    print(
        "\nVALIDATION SUMMARY"
    )

    print(
        "Extraction status:",
        validation.extraction_status,
    )

    print(
        "Numerical validation status:",
        validation.numerical_validation_status,
    )

    print(
        "Points checked:",
        validation.n_points_checked,
    )

    print(
        "Maximum absolute error:",
        f"{validation.maximum_absolute_error:.6e}",
    )

    print(
        "Maximum relative error:",
        f"{validation.maximum_relative_error:.6e}",
    )

    print(
        "rtol:",
        validation.rtol,
    )

    print(
        "atol:",
        validation.atol,
    )

    print(
        "\nPOINTS"
    )

    for result in validation.point_results:
        print()

        print(
            f"Point {result.point_index}"
        )

        print(
            "  time:",
            f"{result.time:.16g}",
        )

        print(
            "  max abs error:",
            f"{result.maximum_absolute_error:.6e}",
        )

        print(
            "  max rel error:",
            f"{result.maximum_relative_error:.6e}",
        )

        print(
            "  all close:",
            result.all_close,
        )


def main():
    """Run one model from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate symbolic mechanistic ground truth "
            "against RoadRunner."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels model identifier, for example "
            "BIOMD0000000079."
        ),
    )

    parser.add_argument(
        "--simulation-points",
        type=int,
        default=DEFAULT_SIMULATION_POINTS,
    )

    parser.add_argument(
        "--validation-points",
        type=int,
        default=DEFAULT_VALIDATION_POINTS,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
    )

    args = parser.parse_args()

    (
        _ground_truth,
        validation,
    ) = validate_mechanistic_ground_truth(
        model_id=args.model_id,
        simulation_points=(
            args.simulation_points
        ),
        validation_points=(
            args.validation_points
        ),
        rtol=args.rtol,
        atol=args.atol,
    )

    print_validation_result(
        validation
    )

    if (
        validation.numerical_validation_status
        != "passed"
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()