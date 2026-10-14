"""Evaluate frozen Selector v1 systems on held-out test trajectories."""

import argparse

from .evaluate_generic_test import (
    evaluate_all_selected_systems,
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
)


POINTS_PER_TRAJECTORY = 101
EXPECTED_TEST_TRAJECTORIES = 8


def validate_test_frame(
    *,
    test_frame,
) -> None:
    """Verify the materialized held-out test dataset."""
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


def run_model(
    *,
    model_id: str,
) -> None:
    """Evaluate all frozen Selector v1 systems for one BioModel."""
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

    test_frame = simulate_initial_condition_set(
        spec=result.spec,
        initial_conditions=design.test,
        duration=result.simulation_time,
        points=POINTS_PER_TRAJECTORY,
        trajectory_prefix="test",
    )

    validate_test_frame(
        test_frame=test_frame,
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


def parse_arguments():
    """Parse command-line arguments."""
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

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    run_model(
        model_id=arguments.model_id,
    )


if __name__ == "__main__":
    main()
