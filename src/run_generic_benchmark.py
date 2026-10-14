"""Run generic variant-aware PySR inference for one BioModel."""

import argparse
from pathlib import Path

import pandas as pd

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    get_benchmark_variant,
    validate_derivative_provenance,
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
from .run_pysr_generic import (
    TRAINING_DATASETS,
    resolve_experiment_root,
    run_generic_training_experiment,
)
from .simulate_generic import (
    build_generic_development_dataset,
)

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
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> Path:
    """Write model-level metadata for one validated benchmark variant.

    Args:
        result: Successful generic preflight result.
        design: Frozen generic initial-condition design.
        dataset: In-memory development dataset.
        variant: Validated derivative method and result namespace.

    Returns:
        Path to the written experiment metadata table.
    """
    validate_derivative_provenance(
        provenance=dataset.derivative_provenance,
        variant=variant,
    )

    experiment_root = resolve_experiment_root(
        model_id=result.model_id,
        variant=variant,
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
        "derivative_method": variant.derivative_method.value,
        "experiment_namespace": variant.experiment_namespace,
        "dataset_generation_seconds": (
            dataset.preprocessing_seconds
        ),
    }

    identity_fields = {
        "derivative_method",
        "experiment_namespace",
    }
    metadata.update(
        {
            key: value
            for key, value in dataset.derivative_provenance.items()
            if key not in identity_fields
        }
    )

    if (
        metadata["derivative_method"]
        != variant.derivative_method.value
        or metadata["experiment_namespace"]
        != variant.experiment_namespace
    ):
        raise RuntimeError("Authoritative benchmark identity changed in metadata.")

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
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> None:
    """Run all frozen training sizes for one benchmark variant.

    Args:
        model_id: BioModels identifier to infer.
        variant: Validated derivative method and result namespace.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")

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
    print(
        "Derivative method:",
        variant.derivative_method.value,
    )
    print(
        "Experiment namespace:",
        variant.experiment_namespace,
    )

    dataset = build_generic_development_dataset(
        spec=result.spec,
        design=design,
        duration=result.simulation_time,
        points=101,
        variant=variant,
    )

    validate_derivative_provenance(
        provenance=dataset.derivative_provenance,
        variant=variant,
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
        variant=variant,
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
            variant=variant,
            derivative_provenance=dataset.derivative_provenance,
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


def parse_benchmark_variant(value: str) -> BenchmarkVariant:
    """Resolve a CLI derivative method through the variant registry.

    Args:
        value: User-supplied derivative-method identifier.

    Returns:
        Validated frozen benchmark variant.

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
            "Run generic PySR inference "
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
