"""Run coupled-system validation for one generic benchmark variant."""

import argparse

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    get_benchmark_variant,
)
from .evaluate_generic_validation import (
    TRAINING_DATASETS,
    evaluate_validation_dataset,
    resolve_validation_experiment_root,
    validate_candidate_pool_provenance,
    validate_selection_development_dataset,
)
from .experimental_design_generic import (
    build_experimental_design,
)
from .model_compatibility import (
    require_model_compatibility,
)
from .preflight import (
    run_preflight,
)
from .simulate_generic import (
    build_generic_development_dataset,
)


def run_validation(
    *,
    model_id: str,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> None:
    """Validate all PySR candidate systems without materializing test data.

    Args:
        model_id: BioModels identifier to validate.
        variant: Candidate derivative method and protected result namespace.
    """
    result = run_preflight(
        model_id=model_id,
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

    experiment_root = resolve_validation_experiment_root(
        model_id=result.model_id,
        variant=variant,
    )

    validate_candidate_pool_provenance(
        experiment_root=experiment_root,
        variant=variant,
        dataset_names=TRAINING_DATASETS,
    )

    dataset = build_generic_development_dataset(
        spec=result.spec,
        design=design,
        duration=result.simulation_time,
        points=101,
        variant=variant,
    )

    validate_selection_development_dataset(
        dataset=dataset,
        spec=result.spec,
        variant=variant,
    )

    print()
    print("=" * 70)
    print("GENERIC COUPLED-SYSTEM VALIDATION")
    print("=" * 70)
    print(
        "Model:",
        result.model_id,
    )
    print(
        "States:",
        result.spec.state_ids,
    )
    print(
        "Validation trajectories:",
        dataset.validation[
            "trajectory_id"
        ].nunique(),
    )
    print(
        "Validation rows:",
        len(
            dataset.validation
        ),
    )
    print(
        "Held-out test rows:",
        len(
            dataset.test
        ),
    )

    selected_rows = []

    for dataset_name in TRAINING_DATASETS:
        summary, selected = (
            evaluate_validation_dataset(
                spec=result.spec,
                dataset_name=dataset_name,
                validation_frame=dataset.validation,
                variant=variant,
            )
        )

        selected_row = (
            selected
            .iloc[
                0
            ]
            .copy()
        )

        selected_row[
            "dataset"
        ] = dataset_name

        selected_rows.append(
            selected_row
        )

    print()
    print("=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    for selected_row in selected_rows:
        print()
        print(
            selected_row[
                "dataset"
            ]
        )

        print(
            "  system:",
            int(
                selected_row[
                    "system_id"
                ]
            ),
        )

        print(
            "  macro NRMSE:",
            float(
                selected_row[
                    "validation_macro_nrmse"
                ]
            ),
        )

        print(
            "  complexity:",
            int(
                selected_row[
                    "complexity_total"
                ]
            ),
        )

        for state_id in result.spec.state_ids:
            print(
                f"  candidate {state_id}:",
                int(
                    selected_row[
                        f"candidate_rank_{state_id}"
                    ]
                ),
            )

            print(
                f"  equation {state_id}:",
                selected_row[
                    f"equation_{state_id}"
                ],
            )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(
        "Held-out test remains unopened."
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
            "Run coupled-system validation for one "
            "generic benchmark variant."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModel identifier, for example "
            "BIOMD0000000036."
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

    run_validation(
        model_id=arguments.model_id,
        variant=arguments.derivative_method,
    )


if __name__ == "__main__":
    main()
