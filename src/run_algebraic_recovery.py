"""Run frozen Algebraic Recovery v1 across the benchmark cohort."""

import pandas as pd

from src.evaluate_algebraic_recovery import (
    ALGEBRAIC_RECOVERY_VERSION,
    NUMERICAL_EQUIVALENCE_ATOL,
    NUMERICAL_EQUIVALENCE_POINTS,
    NUMERICAL_EQUIVALENCE_RTOL,
    NUMERICAL_EQUIVALENCE_SEED,
    evaluate_model,
)
from src.evaluate_dependency_recovery import (
    RESULTS_ROOT,
    TRAIN_DATASETS,
)


FROZEN_MODEL_IDS = (
    "BIOMD0000000079",
    "BIOMD0000000156",
    "BIOMD0000000174",
    "BIOMD0000000322",
    "BIOMD0000000039",
    "BIOMD0000000116",
    "BIOMD0000000229",
    "BIOMD0000000010",
    "BIOMD0000000022",
    "BIOMD0000000476",
)

EXPECTED_MODEL_COUNT = 10
EXPECTED_DATASETS_PER_MODEL = 4
EXPECTED_CONDITION_COUNT = 40

AGGREGATE_OUTPUT_PATH = (
    RESULTS_ROOT
    / "algebraic_recovery_summary.csv"
)


def validate_model_summary(
    *,
    model_id,
    summary,
):
    """Validate one model-level Algebraic Recovery v1 summary."""
    if len(summary) != EXPECTED_DATASETS_PER_MODEL:
        raise ValueError(
            f"{model_id}: expected "
            f"{EXPECTED_DATASETS_PER_MODEL} rows, "
            f"found {len(summary)}."
        )

    required_columns = {
        "algebraic_recovery_version",
        "model_id",
        "dataset_name",
        "n_states",
        "ground_truth_is_autonomous",
        "mechanistic_comparability",
        "n_exact_algebraic_matches",
        "exact_algebraic_recovery_fraction",
        "all_rhs_exact_algebraic_match",
        "n_numerically_evaluated_rhs",
        "n_numerically_equivalent_rhs",
        "numerical_equivalence_fraction",
        "all_evaluable_rhs_numerically_equivalent",
        "numerical_equivalence_rtol",
        "numerical_equivalence_atol",
        "numerical_equivalence_requested_points",
        "numerical_equivalence_seed",
    }

    missing_columns = (
        required_columns
        - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{model_id}: missing columns "
            f"{sorted(missing_columns)}."
        )

    observed_versions = set(
        summary[
            "algebraic_recovery_version"
        ].astype(str)
    )

    if observed_versions != {
        ALGEBRAIC_RECOVERY_VERSION
    }:
        raise ValueError(
            f"{model_id}: unexpected Algebraic Recovery "
            f"versions {sorted(observed_versions)}."
        )

    observed_models = set(
        summary[
            "model_id"
        ].astype(str)
    )

    if observed_models != {
        model_id
    }:
        raise ValueError(
            f"{model_id}: summary contains unexpected "
            f"model IDs {sorted(observed_models)}."
        )

    observed_datasets = set(
        summary[
            "dataset_name"
        ].astype(str)
    )

    if observed_datasets != set(
        TRAIN_DATASETS
    ):
        raise ValueError(
            f"{model_id}: training-dataset mismatch. "
            f"Observed={sorted(observed_datasets)}, "
            f"expected={sorted(TRAIN_DATASETS)}."
        )

    duplicated = summary.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            f"{model_id}: duplicate model/dataset "
            "conditions detected."
        )

    if not (
        summary[
            "numerical_equivalence_rtol"
        ]
        == NUMERICAL_EQUIVALENCE_RTOL
    ).all():
        raise ValueError(
            f"{model_id}: numerical-equivalence rtol "
            "does not match frozen v1."
        )

    if not (
        summary[
            "numerical_equivalence_atol"
        ]
        == NUMERICAL_EQUIVALENCE_ATOL
    ).all():
        raise ValueError(
            f"{model_id}: numerical-equivalence atol "
            "does not match frozen v1."
        )

    if not (
        summary[
            "numerical_equivalence_requested_points"
        ]
        == NUMERICAL_EQUIVALENCE_POINTS
    ).all():
        raise ValueError(
            f"{model_id}: numerical-equivalence point count "
            "does not match frozen v1."
        )

    if not (
        summary[
            "numerical_equivalence_seed"
        ]
        == NUMERICAL_EQUIVALENCE_SEED
    ).all():
        raise ValueError(
            f"{model_id}: numerical-equivalence seed "
            "does not match frozen v1."
        )


def validate_aggregate(
    aggregate,
):
    """Validate the complete frozen Algebraic Recovery v1 aggregate."""
    if len(
        FROZEN_MODEL_IDS
    ) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            "Frozen cohort does not contain exactly "
            f"{EXPECTED_MODEL_COUNT} models."
        )

    if len(
        set(
            FROZEN_MODEL_IDS
        )
    ) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            "Frozen cohort contains duplicate model IDs."
        )

    if len(
        aggregate
    ) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CONDITION_COUNT} conditions, "
            f"found {len(aggregate)}."
        )

    if aggregate[
        "model_id"
    ].nunique() != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} unique models, "
            f"found {aggregate['model_id'].nunique()}."
        )

    duplicated = aggregate.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "Aggregate contains duplicate model/dataset "
            "conditions."
        )

    counts = (
        aggregate.groupby(
            "model_id"
        )[
            "dataset_name"
        ]
        .nunique()
    )

    if not (
        counts
        == EXPECTED_DATASETS_PER_MODEL
    ).all():
        raise ValueError(
            "Each model must contain exactly four "
            "training conditions."
        )

    observed_datasets = set(
        aggregate[
            "dataset_name"
        ].astype(str)
    )

    if observed_datasets != set(
        TRAIN_DATASETS
    ):
        raise ValueError(
            "Aggregate training-dataset mismatch."
        )


def print_aggregate_summary(
    aggregate,
):
    """Print compact frozen-cohort Algebraic Recovery results."""
    print()
    print("=" * 80)
    print("ALGEBRAIC RECOVERY v1 AGGREGATE")
    print("=" * 80)

    print(
        "Version:",
        ALGEBRAIC_RECOVERY_VERSION,
    )

    print(
        "Models:",
        aggregate[
            "model_id"
        ].nunique(),
    )

    print(
        "Conditions:",
        len(
            aggregate
        ),
    )

    print(
        "Training datasets:",
        ", ".join(
            TRAIN_DATASETS
        ),
    )

    print()
    print(
        "Conditions by mechanistic comparability:"
    )

    print(
        aggregate[
            "mechanistic_comparability"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Mean system-level recovery by training dataset:"
    )

    columns = [
        "exact_algebraic_recovery_fraction",
        "numerical_equivalence_fraction",
    ]

    by_training = (
        aggregate.groupby(
            "dataset_name",
            sort=False,
        )[
            columns
        ]
        .mean()
        .reset_index()
    )

    print(
        by_training.to_string(
            index=False
        )
    )

    print()
    print(
        "Complete exact system recovery:"
    )

    exact_systems = (
        aggregate.groupby(
            "dataset_name",
            sort=False,
        )[
            "all_rhs_exact_algebraic_match"
        ]
        .agg(
            [
                "sum",
                "count",
            ]
        )
        .reset_index()
    )

    print(
        exact_systems.to_string(
            index=False
        )
    )

    print()
    print(
        "Complete numerical-equivalence recovery:"
    )

    numerical_systems = (
        aggregate[
            aggregate[
                "n_numerically_evaluated_rhs"
            ]
            > 0
        ]
        .groupby(
            "dataset_name",
            sort=False,
        )[
            "all_evaluable_rhs_numerically_equivalent"
        ]
        .agg(
            [
                "sum",
                "count",
            ]
        )
        .reset_index()
    )

    print(
        numerical_systems.to_string(
            index=False
        )
    )

    print()
    print(
        "Aggregate output:",
        AGGREGATE_OUTPUT_PATH,
    )


def run_all_models():
    """Run frozen Algebraic Recovery v1 for all cohort models."""
    summaries = []

    print()
    print("=" * 80)
    print("RUNNING ALGEBRAIC RECOVERY")
    print("=" * 80)

    print(
        "Version:",
        ALGEBRAIC_RECOVERY_VERSION,
    )

    print(
        "Frozen models:",
        len(
            FROZEN_MODEL_IDS
        ),
    )

    print(
        "Expected conditions:",
        EXPECTED_CONDITION_COUNT,
    )

    print(
        "Frozen numerical diagnostic:",
        f"rtol={NUMERICAL_EQUIVALENCE_RTOL}, "
        f"atol={NUMERICAL_EQUIVALENCE_ATOL}, "
        f"points={NUMERICAL_EQUIVALENCE_POINTS}, "
        f"seed={NUMERICAL_EQUIVALENCE_SEED}",
    )

    for model_index, model_id in enumerate(
        FROZEN_MODEL_IDS,
        start=1,
    ):
        print()
        print("#" * 80)

        print(
            f"MODEL {model_index}/{EXPECTED_MODEL_COUNT}: "
            f"{model_id}"
        )

        print("#" * 80)

        (
            _ground_truth,
            _result,
            summary,
            _output_path,
            _summary_path,
        ) = evaluate_model(
            model_id=model_id,
        )

        validate_model_summary(
            model_id=model_id,
            summary=summary,
        )

        summaries.append(
            summary
        )

    aggregate = pd.concat(
        summaries,
        ignore_index=True,
    )

    validate_aggregate(
        aggregate
    )

    AGGREGATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate.to_csv(
        AGGREGATE_OUTPUT_PATH,
        index=False,
    )

    print_aggregate_summary(
        aggregate
    )

    return aggregate


def main():
    """Run and validate Algebraic Recovery v1."""
    run_all_models()


if __name__ == "__main__":
    main()