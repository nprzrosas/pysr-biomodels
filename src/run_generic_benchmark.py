"""Run generic exact-derivative PySR inference for one BioModel."""

import argparse
from pathlib import Path

import pandas as pd

from .benchmark_variant import EXACT_BENCHMARK_VARIANT
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
from .run_pysr_generic import (
    TRAINING_DATASETS,
    run_generic_training_experiment,
)
from .simulate_generic import (
    build_generic_development_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = EXACT_BENCHMARK_VARIANT.experiment_namespace


def validate_development_dataset(
    *,
    dataset,
) -> None:
    """Verify that train and validation exist while held-out test is absent."""
    expected_rows = {
        "train_04": 404,
        "train_08": 808,
        "train_16": 1616,
        "train_32": 3232,
        "validation": 808,
    }

    for dataset_name, expected_count in expected_rows.items():
        frame = getattr(
            dataset,
            dataset_name,
        )

        if len(frame) != expected_count:
            raise ValueError(
                f"{dataset_name} has {len(frame)} rows; "
                f"expected {expected_count}."
            )

    if not dataset.test.empty:
        raise ValueError(
            "Held-out test data must not be materialized "
            "during development inference."
        )


def write_experiment_metadata(
    *,
    result,
    design,
    dataset,
) -> Path:
    """Write model-level metadata describing the frozen development run."""
    experiment_root = (
        PROJECT_ROOT
        / "results"
        / result.model_id
        / EXPERIMENT_NAME
    )

    experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "model_id": result.model_id,
        "model_path": str(
            result.spec.model_path
        ),
        "number_of_states": (
            result.spec.n_states
        ),
        "state_ids": ";".join(
            result.spec.state_ids
        ),
        "simulation_time": (
            result.simulation_time
        ),
        "points_per_trajectory": 101,
        "train_trajectories_max": len(
            design.train
        ),
        "validation_trajectories": len(
            design.validation
        ),
        "test_trajectories_designed": len(
            design.test
        ),
        "test_materialized": False,
        "derivative_method": (
            EXACT_BENCHMARK_VARIANT.derivative_method.value
        ),
        "dataset_generation_seconds": (
            dataset.preprocessing_seconds
        ),
    }

    metadata_path = (
        experiment_root
        / "experiment_metadata.csv"
    )

    pd.DataFrame(
        [metadata]
    ).to_csv(
        metadata_path,
        index=False,
    )

    return metadata_path


def run_model(
    *,
    model_id: str,
) -> None:
    """Run all frozen training sizes for one compatible BioModel."""
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

    print()
    print("=" * 70)
    print("BUILDING DEVELOPMENT DATASET")
    print("=" * 70)
    print(
        "Held-out test trajectories will NOT be materialized."
    )

    dataset = build_generic_development_dataset(
        spec=result.spec,
        design=design,
        duration=result.simulation_time,
        points=101,
        variant=EXACT_BENCHMARK_VARIANT,
    )

    validate_development_dataset(
        dataset=dataset,
    )

    print(
        "Dataset generation seconds:",
        f"{dataset.preprocessing_seconds:.6f}",
    )

    print(
        "Validation trajectories:",
        dataset.validation[
            "trajectory_id"
        ].nunique(),
    )

    print(
        "Test rows:",
        len(dataset.test),
    )

    metadata_path = write_experiment_metadata(
        result=result,
        design=design,
        dataset=dataset,
    )

    print(
        "Experiment metadata:",
        metadata_path,
    )

    for dataset_name in TRAINING_DATASETS:
        train_frame = getattr(
            dataset,
            dataset_name,
        )

        run_generic_training_experiment(
            spec=result.spec,
            dataset_name=dataset_name,
            train_frame=train_frame,
            dataset_generation_seconds=(
                dataset.preprocessing_seconds
            ),
        )

    print()
    print("=" * 70)
    print("GENERIC INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Model:",
        result.model_id,
    )

    print(
        "Training datasets:",
        TRAINING_DATASETS,
    )

    print(
        "Validation generated:",
        not dataset.validation.empty,
    )

    print(
        "Held-out test materialized:",
        not dataset.test.empty,
    )

    print()
    print(
        "Next stage: coupled-system validation."
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run exact-derivative generic PySR inference "
            "for one compatible BioModel."
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

    run_model(
        model_id=arguments.model_id,
    )


if __name__ == "__main__":
    main()
