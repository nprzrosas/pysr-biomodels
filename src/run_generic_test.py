"""Evaluate frozen Selector v1 systems on held-out test trajectories."""

import argparse

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    DerivativeMethod,
    get_benchmark_variant,
)
from .evaluate_generic_test import (
    evaluate_all_selected_systems,
    load_frozen_selected_systems,
)
from .experimental_design_generic import (
    build_experimental_design,
)
from .model_compatibility import (
    require_model_compatibility,
)
from .preflight import (
    print_preflight,
    run_preflight,
)
from .simulate_generic import (
    simulate_initial_condition_set,
    simulate_state_only_initial_condition_set,
)

POINTS_PER_TRAJECTORY = 101
EXPECTED_TEST_TRAJECTORIES = 8


def validate_test_frame(
    *,
    test_frame,
    spec,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> None:
    """Verify the materialized held-out test dataset.

    Args:
        test_frame: Held-out trajectories to validate.
        spec: Ordered model metadata.
        variant: Validated derivative method and protected namespace.
    """
    trajectory_count = test_frame[
        "trajectory_id"
    ].nunique()

    expected_rows = (
        EXPECTED_TEST_TRAJECTORIES
        * POINTS_PER_TRAJECTORY
    )

    if trajectory_count != EXPECTED_TEST_TRAJECTORIES:
        raise ValueError(
            "Held-out test contains "
            f"{trajectory_count} trajectories; expected "
            f"{EXPECTED_TEST_TRAJECTORIES}."
        )

    if len(test_frame) != expected_rows:
        raise ValueError(
            "Held-out test contains "
            f"{len(test_frame)} rows; expected "
            f"{expected_rows}."
        )

    if variant.derivative_method is DerivativeMethod.SAVGOL:
        expected_columns = (
            "trajectory_id",
            "time",
            *spec.state_ids,
        )

        if tuple(test_frame.columns) != expected_columns:
            raise ValueError(
                "Savitzky-Golay held-out test data must contain only "
                "trajectory_id, time, and ordered state columns."
            )


def materialize_test_frame(
    *,
    spec,
    initial_conditions,
    duration,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
):
    """Materialize held-out trajectories for one benchmark variant.

    Savitzky-Golay testing uses only state trajectories and therefore never
    invokes exact-rate generation. The default exact path preserves the
    historical test-data schema for existing callers.

    Args:
        spec: Ordered model metadata.
        initial_conditions: Frozen held-out initial-condition matrix.
        duration: Curated simulation horizon.
        variant: Validated derivative method and protected namespace.

    Returns:
        Held-out trajectory table for dynamic evaluation.
    """
    if variant.derivative_method is DerivativeMethod.SAVGOL:
        return simulate_state_only_initial_condition_set(
            spec=spec,
            initial_conditions=initial_conditions,
            duration=duration,
            points=POINTS_PER_TRAJECTORY,
            trajectory_prefix="test",
        )

    if variant.derivative_method is DerivativeMethod.EXACT:
        return simulate_initial_condition_set(
            spec=spec,
            initial_conditions=initial_conditions,
            duration=duration,
            points=POINTS_PER_TRAJECTORY,
            trajectory_prefix="test",
        )

    raise ValueError(
        f"Unsupported derivative method: {variant.derivative_method}."
    )


def run_model(
    *,
    model_id: str,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> None:
    """Evaluate all frozen Selector v1 systems for one BioModel.

    Args:
        model_id: BioModels identifier to evaluate.
        variant: Validated derivative method and protected namespace.
    """
    result = run_preflight(
        model_id=model_id,
    )

    print_preflight(
        result
    )

    require_model_compatibility(
        result.spec
    )

    if result.simulation_time is None:
        raise ValueError(
            f"No curated simulation time available for {model_id}."
        )

    design = build_experimental_design(
        spec=result.spec,
    )

    if len(design.test) != EXPECTED_TEST_TRAJECTORIES:
        raise ValueError(
            "Experimental design contains "
            f"{len(design.test)} test trajectories; expected "
            f"{EXPECTED_TEST_TRAJECTORIES}."
        )

    frozen_selections = load_frozen_selected_systems(
        spec=result.spec,
        variant=variant,
    )

    print()
    print("Frozen coordinate selections:")

    for dataset_name, (selected_row, _) in frozen_selections.items():
        print(
            f"  {dataset_name}: system {int(selected_row['system_id'])}, "
            f"ranks {selected_row['candidate_rank_signature']}"
        )

    print()
    print("=" * 70)
    print("MATERIALIZING HELD-OUT TEST")
    print("=" * 70)

    print(
        "Model:",
        result.model_id,
    )

    print(
        "Simulation time:",
        result.simulation_time,
    )

    print(
        "Test trajectories:",
        len(design.test),
    )

    print(
        "Points per trajectory:",
        POINTS_PER_TRAJECTORY,
    )

    test_frame = materialize_test_frame(
        spec=result.spec,
        initial_conditions=design.test,
        duration=result.simulation_time,
        variant=variant,
    )

    validate_test_frame(
        test_frame=test_frame,
        spec=result.spec,
        variant=variant,
    )

    print(
        "Test rows:",
        len(test_frame),
    )

    print()
    print("=" * 70)
    print("EVALUATING FROZEN SELECTOR V1 SYSTEMS")
    print("=" * 70)

    summary_frame = evaluate_all_selected_systems(
        spec=result.spec,
        test_frame=test_frame,
        variant=variant,
        frozen_selections=frozen_selections,
    )

    print()
    print("=" * 70)
    print("HELD-OUT TEST COMPLETE")
    print("=" * 70)

    print(
        "Model:",
        result.model_id,
    )

    print(
        "Systems evaluated:",
        len(summary_frame),
    )


def parse_benchmark_variant(value: str) -> BenchmarkVariant:
    """Resolve one CLI derivative-method value to a benchmark variant.

    Args:
        value: Derivative-method CLI value.

    Returns:
        Registered immutable benchmark variant.

    Raises:
        argparse.ArgumentTypeError: If the derivative method is unknown.
    """
    try:
        return get_benchmark_variant(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        arguments: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        Parsed model identifier and validated benchmark variant.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Selector v1 systems on "
            "held-out test trajectories."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModel identifier, for example "
            "BIOMD0000000079."
        ),
    )

    parser.add_argument(
        "--derivative-method",
        default=EXACT_BENCHMARK_VARIANT,
        type=parse_benchmark_variant,
        metavar="{exact,savgol}",
        help=(
            "Derivative-target method. Defaults to exact; "
            "savgol selects the frozen Savitzky-Golay v1 variant."
        ),
    )

    return parser.parse_args(arguments)


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    run_model(
        model_id=arguments.model_id,
        variant=arguments.derivative_method,
    )


if __name__ == "__main__":
    main()
