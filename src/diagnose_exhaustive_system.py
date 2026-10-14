"""Diagnose one training dataset with exhaustive coupled-system validation."""

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
from .simulate_generic import (
    build_generic_development_dataset,
)


def run_diagnostic(
    *,
    model_id: str,
    dataset_name: str,
) -> None:
    """Run exhaustive validation for exactly one training dataset."""
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
    print("=" * 80)
    print("EXHAUSTIVE SYSTEM DIAGNOSTIC")
    print("=" * 80)
    print("Model:", result.model_id)
    print("States:", result.spec.state_ids)
    print("Dataset:", dataset_name)
    print(
        "Validation trajectories:",
        dataset.validation["trajectory_id"].nunique(),
    )
    print("Validation rows:", len(dataset.validation))
    print("Held-out test rows:", len(dataset.test))
    print()
    print("Running exhaustive Cartesian system search...")

    summary, selected = evaluate_validation_dataset(
        spec=result.spec,
        dataset_name=dataset_name,
        validation_frame=dataset.validation,
    )

    selected_row = selected.iloc[0]

    print()
    print("=" * 80)
    print("BEST EXHAUSTIVE SYSTEM")
    print("=" * 80)
    print(
        "System ID:",
        int(selected_row["system_id"]),
    )
    print(
        "Validation macro NRMSE:",
        float(selected_row["validation_macro_nrmse"]),
    )
    print(
        "Complexity:",
        int(selected_row["complexity_total"]),
    )

    ranks = []

    for state_id in result.spec.state_ids:
        rank = int(
            selected_row[
                f"candidate_rank_{state_id}"
            ]
        )
        ranks.append(rank)

        print()
        print(
            f"{state_id} candidate rank:",
            rank,
        )
        print(
            f"{state_id} equation:",
            selected_row[
                f"equation_{state_id}"
            ],
        )

    print()
    print("Selected ranks:", tuple(ranks))
    print(
        "Systems evaluated:",
        len(summary),
    )

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("Held-out test remains unopened.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run exhaustive coupled-system validation "
            "for exactly one training dataset."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels model identifier.",
    )

    parser.add_argument(
        "dataset_name",
        help=(
            "Training dataset, for example train_16."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    run_diagnostic(
        model_id=arguments.model_id,
        dataset_name=arguments.dataset_name,
    )


if __name__ == "__main__":
    main()
