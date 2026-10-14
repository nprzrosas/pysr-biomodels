"""Summarize frozen Functional-form Recovery v1 results.

This module performs reporting only. It does not recompute functional-form
recovery metrics and does not modify any frozen inference, selection,
validation, held-out test, or mechanistic-recovery result.

The primary scientific summary is restricted to conditions with:

    mechanistic_comparability == "compatible"

Conditions with feature-space mismatch are retained and reported separately.

Input:
    results/functional_form_recovery_summary.csv

Outputs:
    results/functional_form_recovery_compatible_summary.csv
    results/functional_form_recovery_comparability_summary.csv
"""

import pandas as pd

from src.evaluate_functional_form_recovery import (
    FUNCTIONAL_FORM_RECOVERY_VERSION,
    RESULTS_ROOT,
)


INPUT_PATH = (
    RESULTS_ROOT
    / "functional_form_recovery_summary.csv"
)

COMPATIBLE_OUTPUT_PATH = (
    RESULTS_ROOT
    / "functional_form_recovery_compatible_summary.csv"
)

COMPARABILITY_OUTPUT_PATH = (
    RESULTS_ROOT
    / "functional_form_recovery_comparability_summary.csv"
)

EXPECTED_TOTAL_MODELS = (
    10
)

EXPECTED_TOTAL_CONDITIONS = (
    40
)

EXPECTED_COMPATIBLE_MODELS = (
    8
)

EXPECTED_COMPATIBLE_CONDITIONS = (
    32
)

EXPECTED_MISMATCH_MODELS = (
    2
)

EXPECTED_MISMATCH_CONDITIONS = (
    8
)

EXPECTED_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

METRIC_COLUMNS = (
    "macro_numerator_monomial_f1",
    "macro_interaction_f1",
    "rational_structure_accuracy",
    "macro_denominator_dependency_f1",
    "mean_absolute_numerator_degree_error",
    "mean_absolute_denominator_degree_error",
)


def validate_input(
    frame,
):
    """Validate the frozen Functional-form Recovery v1 aggregate."""
    if len(frame) != EXPECTED_TOTAL_CONDITIONS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_TOTAL_CONDITIONS} total conditions, "
            f"found {len(frame)}."
        )

    if frame[
        "model_id"
    ].nunique() != EXPECTED_TOTAL_MODELS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_TOTAL_MODELS} unique models, found "
            f"{frame['model_id'].nunique()}."
        )

    observed_versions = set(
        frame[
            "functional_form_recovery_version"
        ].astype(str)
    )

    expected_versions = {
        FUNCTIONAL_FORM_RECOVERY_VERSION
    }

    if observed_versions != expected_versions:
        raise ValueError(
            "Unexpected functional-form recovery versions: "
            f"{sorted(observed_versions)}. Expected "
            f"{sorted(expected_versions)}."
        )

    observed_datasets = set(
        frame[
            "dataset_name"
        ].astype(str)
    )

    expected_datasets = set(
        EXPECTED_DATASETS
    )

    if observed_datasets != expected_datasets:
        raise ValueError(
            "Unexpected training datasets. "
            f"Observed={sorted(observed_datasets)}, "
            f"expected={sorted(expected_datasets)}."
        )

    duplicated = frame.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = frame.loc[
            duplicated,
            [
                "model_id",
                "dataset_name",
            ],
        ]

        raise ValueError(
            "Duplicate model/dataset conditions found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    allowed_comparability = {
        "compatible",
        "feature_space_mismatch",
    }

    observed_comparability = set(
        frame[
            "mechanistic_comparability"
        ].astype(str)
    )

    unexpected_comparability = (
        observed_comparability
        - allowed_comparability
    )

    if unexpected_comparability:
        raise ValueError(
            "Unexpected mechanistic comparability values: "
            f"{sorted(unexpected_comparability)}."
        )


def validate_partition(
    *,
    compatible,
    mismatch,
):
    """Validate the compatible and feature-space-mismatch partitions."""
    if len(compatible) != EXPECTED_COMPATIBLE_CONDITIONS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_CONDITIONS} compatible conditions, "
            f"found {len(compatible)}."
        )

    if compatible[
        "model_id"
    ].nunique() != EXPECTED_COMPATIBLE_MODELS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_MODELS} compatible models, "
            f"found {compatible['model_id'].nunique()}."
        )

    if len(mismatch) != EXPECTED_MISMATCH_CONDITIONS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_MISMATCH_CONDITIONS} feature-space-mismatch "
            f"conditions, found {len(mismatch)}."
        )

    if mismatch[
        "model_id"
    ].nunique() != EXPECTED_MISMATCH_MODELS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_MISMATCH_MODELS} feature-space-mismatch models, "
            f"found {mismatch['model_id'].nunique()}."
        )

    compatible_counts = (
        compatible.groupby(
            "model_id"
        )[
            "dataset_name"
        ]
        .nunique()
    )

    if not (
        compatible_counts
        == len(
            EXPECTED_DATASETS
        )
    ).all():
        raise ValueError(
            "Each compatible model must contain all four "
            "training conditions."
        )

    mismatch_counts = (
        mismatch.groupby(
            "model_id"
        )[
            "dataset_name"
        ]
        .nunique()
    )

    if not (
        mismatch_counts
        == len(
            EXPECTED_DATASETS
        )
    ).all():
        raise ValueError(
            "Each feature-space-mismatch model must contain all four "
            "training conditions."
        )


def summarize_by_training(
    frame,
):
    """Calculate reporting summaries by training condition."""
    rows = []

    for dataset_name in EXPECTED_DATASETS:
        dataset_frame = frame[
            frame[
                "dataset_name"
            ]
            == dataset_name
        ]

        row = {
            "functional_form_recovery_version": (
                FUNCTIONAL_FORM_RECOVERY_VERSION
            ),
            "dataset_name": dataset_name,
            "n_models": dataset_frame[
                "model_id"
            ].nunique(),
            "n_conditions": len(
                dataset_frame
            ),
        }

        for metric in METRIC_COLUMNS:
            values = dataset_frame[
                metric
            ]

            row[
                f"mean_{metric}"
            ] = values.mean()

            row[
                f"median_{metric}"
            ] = values.median()

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_comparability(
    frame,
):
    """Summarize cohort size and metrics by comparability class."""
    rows = []

    for comparability, group in frame.groupby(
        "mechanistic_comparability",
        sort=False,
    ):
        row = {
            "functional_form_recovery_version": (
                FUNCTIONAL_FORM_RECOVERY_VERSION
            ),
            "mechanistic_comparability": comparability,
            "n_models": group[
                "model_id"
            ].nunique(),
            "n_conditions": len(
                group
            ),
        }

        for metric in METRIC_COLUMNS:
            row[
                f"mean_{metric}"
            ] = group[
                metric
            ].mean()

            row[
                f"median_{metric}"
            ] = group[
                metric
            ].median()

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def print_model_partition(
    *,
    compatible,
    mismatch,
):
    """Print the model IDs in each mechanistic-comparability class."""
    compatible_models = sorted(
        compatible[
            "model_id"
        ].unique()
    )

    mismatch_models = sorted(
        mismatch[
            "model_id"
        ].unique()
    )

    print()
    print("Compatible models:")

    for model_id in compatible_models:
        print(
            " ",
            model_id,
        )

    print()
    print("Feature-space-mismatch models:")

    for model_id in mismatch_models:
        print(
            " ",
            model_id,
        )


def main():
    """Generate reporting summaries from frozen v1 results."""
    frame = pd.read_csv(
        INPUT_PATH
    )

    validate_input(
        frame
    )

    compatible = frame[
        frame[
            "mechanistic_comparability"
        ]
        == "compatible"
    ].copy()

    mismatch = frame[
        frame[
            "mechanistic_comparability"
        ]
        == "feature_space_mismatch"
    ].copy()

    validate_partition(
        compatible=compatible,
        mismatch=mismatch,
    )

    compatible_summary = summarize_by_training(
        compatible
    )

    comparability_summary = summarize_comparability(
        frame
    )

    compatible_summary.to_csv(
        COMPATIBLE_OUTPUT_PATH,
        index=False,
    )

    comparability_summary.to_csv(
        COMPARABILITY_OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("FUNCTIONAL-FORM RECOVERY v1 REPORTING SUMMARY")
    print("=" * 80)

    print(
        "Input:",
        INPUT_PATH,
    )

    print(
        "Version:",
        FUNCTIONAL_FORM_RECOVERY_VERSION,
    )

    print()
    print(
        "Total cohort:",
        f"{frame['model_id'].nunique()} models,",
        f"{len(frame)} conditions",
    )

    print(
        "Compatible cohort:",
        f"{compatible['model_id'].nunique()} models,",
        f"{len(compatible)} conditions",
    )

    print(
        "Feature-space mismatch:",
        f"{mismatch['model_id'].nunique()} models,",
        f"{len(mismatch)} conditions",
    )

    print_model_partition(
        compatible=compatible,
        mismatch=mismatch,
    )

    print()
    print("=" * 80)
    print("PRIMARY COMPATIBLE-COHORT SUMMARY")
    print("=" * 80)

    print(
        compatible_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("COMPARABILITY-CLASS SUMMARY")
    print("=" * 80)

    print(
        comparability_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Compatible summary output:",
        COMPATIBLE_OUTPUT_PATH,
    )

    print(
        "Comparability summary output:",
        COMPARABILITY_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()