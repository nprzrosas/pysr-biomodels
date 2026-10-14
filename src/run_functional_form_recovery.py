"""Run frozen functional-form recovery v1 across the benchmark cohort.

This module evaluates the frozen Functional-form Recovery v1 definition
for the same ten models used in the frozen Selector v1, held-out test,
and dependency-recovery analyses.

The runner does not modify inference, candidate generation, model
selection, held-out testing, mechanistic ground truth, or the
functional-form recovery definition.

Expected cohort:
    10 models
    4 training conditions per model
    40 model-training conditions total

Outputs:
    Per-model functional-form recovery files are written by
    evaluate_functional_form_recovery.evaluate_model.

    The aggregate summary is written to:
        results/functional_form_recovery_summary.csv
"""

import pandas as pd

from src.evaluate_dependency_recovery import (
    TRAIN_DATASETS,
)
from src.evaluate_functional_form_recovery import (
    FUNCTIONAL_FORM_RECOVERY_VERSION,
    RESULTS_ROOT,
    evaluate_model,
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

EXPECTED_MODEL_COUNT = (
    10
)

EXPECTED_DATASETS_PER_MODEL = (
    4
)

EXPECTED_CONDITION_COUNT = (
    EXPECTED_MODEL_COUNT
    * EXPECTED_DATASETS_PER_MODEL
)

AGGREGATE_OUTPUT_PATH = (
    RESULTS_ROOT
    / "functional_form_recovery_summary.csv"
)


def validate_model_summary(
    *,
    model_id,
    summary,
):
    """Validate one model-level functional-form recovery summary."""
    if len(summary) != EXPECTED_DATASETS_PER_MODEL:
        raise ValueError(
            f"{model_id}: expected "
            f"{EXPECTED_DATASETS_PER_MODEL} summary rows, "
            f"found {len(summary)}."
        )

    if "functional_form_recovery_version" not in summary.columns:
        raise ValueError(
            f"{model_id}: summary is missing "
            "functional_form_recovery_version."
        )

    observed_versions = set(
        summary[
            "functional_form_recovery_version"
        ].astype(str)
    )

    expected_versions = {
        FUNCTIONAL_FORM_RECOVERY_VERSION
    }

    if observed_versions != expected_versions:
        raise ValueError(
            f"{model_id}: unexpected functional-form recovery "
            f"versions {sorted(observed_versions)}; expected "
            f"{sorted(expected_versions)}."
        )

    observed_model_ids = set(
        summary[
            "model_id"
        ].astype(str)
    )

    if observed_model_ids != {
        model_id
    }:
        raise ValueError(
            f"{model_id}: summary contains unexpected model IDs "
            f"{sorted(observed_model_ids)}."
        )

    observed_datasets = set(
        summary[
            "dataset_name"
        ].astype(str)
    )

    expected_datasets = set(
        TRAIN_DATASETS
    )

    if observed_datasets != expected_datasets:
        missing = (
            expected_datasets
            - observed_datasets
        )

        unexpected = (
            observed_datasets
            - expected_datasets
        )

        raise ValueError(
            f"{model_id}: training-dataset mismatch. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    duplicated = summary.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = summary.loc[
            duplicated,
            [
                "model_id",
                "dataset_name",
            ],
        ]

        raise ValueError(
            f"{model_id}: duplicate model/dataset conditions:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def validate_aggregate(
    aggregate,
):
    """Validate the complete frozen 10-model aggregate."""
    if len(FROZEN_MODEL_IDS) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            "Frozen model cohort definition contains "
            f"{len(FROZEN_MODEL_IDS)} models; expected "
            f"{EXPECTED_MODEL_COUNT}."
        )

    if len(set(FROZEN_MODEL_IDS)) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            "Frozen model cohort contains duplicate model IDs."
        )

    if len(aggregate) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            "Aggregate contains "
            f"{len(aggregate)} conditions; expected "
            f"{EXPECTED_CONDITION_COUNT}."
        )

    required_columns = {
        "functional_form_recovery_version",
        "model_id",
        "dataset_name",
        "n_states",
        "ground_truth_is_autonomous",
        "mechanistic_comparability",
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "macro_denominator_dependency_f1",
        "mean_absolute_numerator_degree_error",
        "mean_absolute_denominator_degree_error",
    }

    missing_columns = (
        required_columns
        - set(
            aggregate.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Aggregate is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    observed_versions = set(
        aggregate[
            "functional_form_recovery_version"
        ].astype(str)
    )

    expected_versions = {
        FUNCTIONAL_FORM_RECOVERY_VERSION
    }

    if observed_versions != expected_versions:
        raise ValueError(
            "Aggregate contains unexpected functional-form "
            f"recovery versions {sorted(observed_versions)}; "
            f"expected {sorted(expected_versions)}."
        )

    observed_models = set(
        aggregate[
            "model_id"
        ].astype(str)
    )

    expected_models = set(
        FROZEN_MODEL_IDS
    )

    if observed_models != expected_models:
        missing_models = (
            expected_models
            - observed_models
        )

        unexpected_models = (
            observed_models
            - expected_models
        )

        raise ValueError(
            "Aggregate model cohort mismatch. "
            f"Missing={sorted(missing_models)}, "
            f"unexpected={sorted(unexpected_models)}."
        )

    observed_datasets = set(
        aggregate[
            "dataset_name"
        ].astype(str)
    )

    expected_datasets = set(
        TRAIN_DATASETS
    )

    if observed_datasets != expected_datasets:
        missing_datasets = (
            expected_datasets
            - observed_datasets
        )

        unexpected_datasets = (
            observed_datasets
            - expected_datasets
        )

        raise ValueError(
            "Aggregate training-dataset mismatch. "
            f"Missing={sorted(missing_datasets)}, "
            f"unexpected={sorted(unexpected_datasets)}."
        )

    duplicated = aggregate.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = aggregate.loc[
            duplicated,
            [
                "model_id",
                "dataset_name",
            ],
        ]

        raise ValueError(
            "Aggregate contains duplicate model/dataset "
            "conditions:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    counts_by_model = (
        aggregate.groupby(
            "model_id"
        )[
            "dataset_name"
        ]
        .nunique()
    )

    invalid_counts = counts_by_model[
        counts_by_model
        != EXPECTED_DATASETS_PER_MODEL
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Some models do not contain exactly "
            f"{EXPECTED_DATASETS_PER_MODEL} training conditions:\n"
            f"{invalid_counts.to_string()}"
        )


def print_aggregate_summary(
    aggregate,
):
    """Print a compact summary of the complete aggregate."""
    print()
    print("=" * 80)
    print("FUNCTIONAL-FORM RECOVERY v1 AGGREGATE")
    print("=" * 80)

    print(
        "Version:",
        FUNCTIONAL_FORM_RECOVERY_VERSION,
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
    print("Conditions by mechanistic comparability:")

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
    print("Mean metrics by training dataset:")

    metric_columns = [
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "macro_denominator_dependency_f1",
        "mean_absolute_numerator_degree_error",
        "mean_absolute_denominator_degree_error",
    ]

    by_training = (
        aggregate.groupby(
            "dataset_name",
            sort=False,
        )[
            metric_columns
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
        "Aggregate output:",
        AGGREGATE_OUTPUT_PATH,
    )


def run_all_models():
    """Run frozen functional-form recovery v1 for all cohort models."""
    summaries = []

    print()
    print("=" * 80)
    print("RUNNING FUNCTIONAL-FORM RECOVERY")
    print("=" * 80)

    print(
        "Version:",
        FUNCTIONAL_FORM_RECOVERY_VERSION,
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
    """Run and validate the frozen 10-model functional-form analysis."""
    run_all_models()


if __name__ == "__main__":
    main()