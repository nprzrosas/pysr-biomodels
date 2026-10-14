"""Run coupled-system validation for one generic exact-derivative experiment."""

import argparse

from .evaluate_generic_validation import (
    evaluate_validation_dataset,
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
from .run_pysr_generic import (
    TRAINING_DATASETS,
)
from .simulate_generic import (
    build_generic_development_dataset,
)


def run_validation(
    *,
    model_id: str,
) -> None:
    """Validate all PySR candidate systems without materializing test data."""
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

    dataset = build_generic_development_dataset(
        spec=result.spec,
        design=design,
        duration=result.simulation_time,
        points=101,
    )

    if not dataset.test.empty:
        raise ValueError(
            "Held-out test data must remain unmaterialized "
            "during validation."
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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run coupled-system validation for one "
            "generic exact-derivative experiment."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModel identifier, for example "
            "BIOMD0000000036."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    run_validation(
        model_id=arguments.model_id,
    )


if __name__ == "__main__":
    main()