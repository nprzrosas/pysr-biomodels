"""Summarize frozen integrated dynamic and mechanistic recovery results.

This module is reporting-only. It reads the frozen integrated recovery tables
and summarizes recovery as a function of the number of training trajectories.

Primary mechanistic reporting is restricted to models whose ground-truth RHS
is compatible with the states-only inference feature space.

Dynamic held-out performance is reported in two complementary ways:

1. Held-out integration success rate across all compatible systems.
2. Test NRMSE and test/validation ratios among systems that successfully
   integrated all held-out trajectories.

Mechanistic recovery is summarized using the already-frozen metrics from:

- Dependency Recovery.
- Functional-form Recovery v1.
- Algebraic Recovery v1.

No inference, candidate selection, trajectory simulation, held-out testing,
or mechanistic recovery metric is recomputed here.
"""

import numpy as np
import pandas as pd

from src.evaluate_algebraic_recovery import (
    ALGEBRAIC_RECOVERY_VERSION,
)
from src.evaluate_dependency_recovery import (
    RESULTS_ROOT,
    TRAIN_DATASETS,
)
from src.evaluate_functional_form_recovery import (
    FUNCTIONAL_FORM_RECOVERY_VERSION,
)


INPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery.csv"
)

COMPATIBLE_INPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_compatible.csv"
)

OUTPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_summary.csv"
)

COMPARABILITY_OUTPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_comparability_summary.csv"
)

EXPECTED_MODEL_COUNT = 10
EXPECTED_CONDITION_COUNT = 40

EXPECTED_COMPATIBLE_MODEL_COUNT = 8
EXPECTED_COMPATIBLE_CONDITION_COUNT = 32

EXPECTED_MISMATCH_MODEL_COUNT = 2
EXPECTED_MISMATCH_CONDITION_COUNT = 8

COMPATIBLE_LABEL = (
    "compatible"
)

MISMATCH_LABEL = (
    "feature_space_mismatch"
)

SUCCESS_LABEL = (
    "success"
)

FAILED_LABEL = (
    "failed"
)


def require_file(
    path,
):
    """Require one frozen integrated result file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen result was not found: {path}"
        )


def require_columns(
    frame,
    required_columns,
    *,
    source_name,
):
    """Require a set of columns in one frozen result table."""
    missing_columns = (
        set(
            required_columns
        )
        - set(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing_columns)}."
        )


def safe_fraction(
    numerator,
    denominator,
):
    """Return numerator / denominator, or NaN for a zero denominator."""
    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
    )


def finite_series(
    values,
):
    """Return numeric finite values only."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    return numeric[
        np.isfinite(
            numeric
        )
    ]


def validate_condition_keys(
    frame,
    *,
    source_name,
    expected_models,
    expected_conditions,
):
    """Validate model x training-condition uniqueness and completeness."""
    require_columns(
        frame,
        {
            "model_id",
            "train_dataset",
        },
        source_name=source_name,
    )

    if len(
        frame
    ) != expected_conditions:
        raise ValueError(
            f"{source_name}: expected "
            f"{expected_conditions} conditions, "
            f"found {len(frame)}."
        )

    if frame[
        "model_id"
    ].nunique() != expected_models:
        raise ValueError(
            f"{source_name}: expected "
            f"{expected_models} models, found "
            f"{frame['model_id'].nunique()}."
        )

    observed_datasets = set(
        frame[
            "train_dataset"
        ].astype(
            str
        )
    )

    expected_datasets = set(
        TRAIN_DATASETS
    )

    if observed_datasets != expected_datasets:
        raise ValueError(
            f"{source_name}: training-condition mismatch. "
            f"Observed={sorted(observed_datasets)}, "
            f"expected={sorted(expected_datasets)}."
        )

    duplicated = frame.duplicated(
        subset=[
            "model_id",
            "train_dataset",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = frame.loc[
            duplicated,
            [
                "model_id",
                "train_dataset",
            ],
        ]

        raise ValueError(
            f"{source_name}: duplicate conditions detected:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    dataset_counts = (
        frame.groupby(
            "model_id"
        )[
            "train_dataset"
        ]
        .nunique()
    )

    if not (
        dataset_counts
        == len(
            TRAIN_DATASETS
        )
    ).all():
        raise ValueError(
            f"{source_name}: every model must have exactly "
            f"{len(TRAIN_DATASETS)} training conditions."
        )


def validate_frozen_inputs(
    integrated,
    compatible,
):
    """Validate the two frozen integrated recovery inputs."""
    required_columns = {
        "model_id",
        "n_states",
        "train_dataset",
        "train_trajectories",
        "validation_macro_nrmse",
        "validation_status",
        "test_macro_nrmse",
        "test_status",
        "validation_test_ratio",
        "test_minus_validation",
        "ground_truth_is_autonomous",
        "ground_truth_independent_variables",
        "inference_feature_space",
        "mechanistic_comparability",
        "macro_dependency_f1",
        "micro_dependency_precision",
        "micro_dependency_recall",
        "micro_dependency_f1",
        "total_true_positive",
        "total_false_positive",
        "total_false_negative",
        "functional_form_recovery_version",
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "macro_denominator_dependency_f1",
        "mean_absolute_numerator_degree_error",
        "mean_absolute_denominator_degree_error",
        "algebraic_recovery_version",
        "n_exact_algebraic_matches",
        "exact_algebraic_recovery_fraction",
        "n_numerically_evaluated_rhs",
        "n_numerically_equivalent_rhs",
        "numerical_equivalence_fraction",
    }

    require_columns(
        integrated,
        required_columns,
        source_name="Integrated recovery",
    )

    require_columns(
        compatible,
        required_columns,
        source_name="Compatible integrated recovery",
    )

    validate_condition_keys(
        integrated,
        source_name="Integrated recovery",
        expected_models=EXPECTED_MODEL_COUNT,
        expected_conditions=EXPECTED_CONDITION_COUNT,
    )

    validate_condition_keys(
        compatible,
        source_name="Compatible integrated recovery",
        expected_models=EXPECTED_COMPATIBLE_MODEL_COUNT,
        expected_conditions=EXPECTED_COMPATIBLE_CONDITION_COUNT,
    )

    integrated_compatible = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == COMPATIBLE_LABEL
    ].copy()

    mismatch = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == MISMATCH_LABEL
    ].copy()

    if len(
        integrated_compatible
    ) != EXPECTED_COMPATIBLE_CONDITION_COUNT:
        raise ValueError(
            "Integrated recovery contains an unexpected number "
            "of compatible conditions."
        )

    if integrated_compatible[
        "model_id"
    ].nunique() != EXPECTED_COMPATIBLE_MODEL_COUNT:
        raise ValueError(
            "Integrated recovery contains an unexpected number "
            "of compatible models."
        )

    if len(
        mismatch
    ) != EXPECTED_MISMATCH_CONDITION_COUNT:
        raise ValueError(
            "Integrated recovery contains an unexpected number "
            "of feature-space mismatch conditions."
        )

    if mismatch[
        "model_id"
    ].nunique() != EXPECTED_MISMATCH_MODEL_COUNT:
        raise ValueError(
            "Integrated recovery contains an unexpected number "
            "of feature-space mismatch models."
        )

    compatible_keys = set(
        zip(
            compatible[
                "model_id"
            ].astype(
                str
            ),
            compatible[
                "train_dataset"
            ].astype(
                str
            ),
        )
    )

    integrated_compatible_keys = set(
        zip(
            integrated_compatible[
                "model_id"
            ].astype(
                str
            ),
            integrated_compatible[
                "train_dataset"
            ].astype(
                str
            ),
        )
    )

    if compatible_keys != integrated_compatible_keys:
        raise ValueError(
            "Compatible integrated recovery does not match "
            "the compatible subset of the complete integrated table."
        )

    observed_functional_versions = set(
        integrated[
            "functional_form_recovery_version"
        ].astype(
            str
        )
    )

    if observed_functional_versions != {
        FUNCTIONAL_FORM_RECOVERY_VERSION
    }:
        raise ValueError(
            "Functional-form Recovery version mismatch. "
            f"Observed={sorted(observed_functional_versions)}, "
            f"expected={FUNCTIONAL_FORM_RECOVERY_VERSION}."
        )

    observed_algebraic_versions = set(
        integrated[
            "algebraic_recovery_version"
        ].astype(
            str
        )
    )

    if observed_algebraic_versions != {
        ALGEBRAIC_RECOVERY_VERSION
    }:
        raise ValueError(
            "Algebraic Recovery version mismatch. "
            f"Observed={sorted(observed_algebraic_versions)}, "
            f"expected={ALGEBRAIC_RECOVERY_VERSION}."
        )

    allowed_test_statuses = {
        SUCCESS_LABEL,
        FAILED_LABEL,
    }

    observed_test_statuses = set(
        integrated[
            "test_status"
        ].astype(
            str
        )
    )

    if not observed_test_statuses.issubset(
        allowed_test_statuses
    ):
        raise ValueError(
            "Unexpected held-out test status values: "
            f"{sorted(observed_test_statuses)}."
        )


def summarize_compatible_dataset(
    frame,
    *,
    dataset_name,
):
    """Summarize one training size for the compatible model cohort."""
    subset = frame[
        frame[
            "train_dataset"
        ]
        == dataset_name
    ].copy()

    if len(
        subset
    ) != EXPECTED_COMPATIBLE_MODEL_COUNT:
        raise ValueError(
            f"{dataset_name}: expected "
            f"{EXPECTED_COMPATIBLE_MODEL_COUNT} compatible models, "
            f"found {len(subset)}."
        )

    train_trajectory_values = set(
        subset[
            "train_trajectories"
        ].astype(
            int
        )
    )

    if len(
        train_trajectory_values
    ) != 1:
        raise ValueError(
            f"{dataset_name}: inconsistent train_trajectories values: "
            f"{sorted(train_trajectory_values)}."
        )

    train_trajectories = next(
        iter(
            train_trajectory_values
        )
    )

    successful = subset[
        subset[
            "test_status"
        ]
        == SUCCESS_LABEL
    ].copy()

    failed = subset[
        subset[
            "test_status"
        ]
        == FAILED_LABEL
    ].copy()

    successful_test_nrmse = finite_series(
        successful[
            "test_macro_nrmse"
        ]
    )

    successful_validation_test_ratio = finite_series(
        successful[
            "validation_test_ratio"
        ]
    )

    total_rhs = int(
        subset[
            "n_states"
        ].sum()
    )

    total_true_positive = int(
        subset[
            "total_true_positive"
        ].sum()
    )

    total_false_positive = int(
        subset[
            "total_false_positive"
        ].sum()
    )

    total_false_negative = int(
        subset[
            "total_false_negative"
        ].sum()
    )

    micro_dependency_precision = safe_fraction(
        total_true_positive,
        (
            total_true_positive
            + total_false_positive
        ),
    )

    micro_dependency_recall = safe_fraction(
        total_true_positive,
        (
            total_true_positive
            + total_false_negative
        ),
    )

    if (
        np.isnan(
            micro_dependency_precision
        )
        or np.isnan(
            micro_dependency_recall
        )
        or (
            micro_dependency_precision
            + micro_dependency_recall
        )
        == 0
    ):
        micro_dependency_f1 = (
            0.0
            if (
                not np.isnan(
                    micro_dependency_precision
                )
                and not np.isnan(
                    micro_dependency_recall
                )
            )
            else np.nan
        )

    else:
        micro_dependency_f1 = (
            2.0
            * micro_dependency_precision
            * micro_dependency_recall
            / (
                micro_dependency_precision
                + micro_dependency_recall
            )
        )

    n_exact_algebraic_matches = int(
        subset[
            "n_exact_algebraic_matches"
        ].sum()
    )

    n_numerically_evaluated_rhs = int(
        subset[
            "n_numerically_evaluated_rhs"
        ].sum()
    )

    n_numerically_equivalent_rhs = int(
        subset[
            "n_numerically_equivalent_rhs"
        ].sum()
    )

    row = {
        "train_dataset": (
            dataset_name
        ),
        "train_trajectories": (
            train_trajectories
        ),
        "n_models": (
            len(
                subset
            )
        ),
        "total_rhs": (
            total_rhs
        ),
        "n_test_success": (
            len(
                successful
            )
        ),
        "n_test_failed": (
            len(
                failed
            )
        ),
        "test_success_rate": safe_fraction(
            len(
                successful
            ),
            len(
                subset
            ),
        ),
        "median_test_macro_nrmse_successful": (
            successful_test_nrmse.median()
            if len(
                successful_test_nrmse
            )
            else np.nan
        ),
        "mean_test_macro_nrmse_successful": (
            successful_test_nrmse.mean()
            if len(
                successful_test_nrmse
            )
            else np.nan
        ),
        "median_validation_test_ratio_successful": (
            successful_validation_test_ratio.median()
            if len(
                successful_validation_test_ratio
            )
            else np.nan
        ),
        "mean_validation_test_ratio_successful": (
            successful_validation_test_ratio.mean()
            if len(
                successful_validation_test_ratio
            )
            else np.nan
        ),
        "mean_model_macro_dependency_f1": (
            subset[
                "macro_dependency_f1"
            ].mean()
        ),
        "median_model_macro_dependency_f1": (
            subset[
                "macro_dependency_f1"
            ].median()
        ),
        "micro_dependency_precision": (
            micro_dependency_precision
        ),
        "micro_dependency_recall": (
            micro_dependency_recall
        ),
        "micro_dependency_f1": (
            micro_dependency_f1
        ),
        "mean_model_numerator_monomial_f1": (
            subset[
                "macro_numerator_monomial_f1"
            ].mean()
        ),
        "median_model_numerator_monomial_f1": (
            subset[
                "macro_numerator_monomial_f1"
            ].median()
        ),
        "mean_model_interaction_f1": (
            subset[
                "macro_interaction_f1"
            ].mean()
        ),
        "median_model_interaction_f1": (
            subset[
                "macro_interaction_f1"
            ].median()
        ),
        "mean_model_rational_structure_accuracy": (
            subset[
                "rational_structure_accuracy"
            ].mean()
        ),
        "median_model_rational_structure_accuracy": (
            subset[
                "rational_structure_accuracy"
            ].median()
        ),
        "mean_model_denominator_dependency_f1": (
            subset[
                "macro_denominator_dependency_f1"
            ].mean()
        ),
        "median_model_denominator_dependency_f1": (
            subset[
                "macro_denominator_dependency_f1"
            ].median()
        ),
        "mean_absolute_numerator_degree_error": (
            subset[
                "mean_absolute_numerator_degree_error"
            ].mean()
        ),
        "median_absolute_numerator_degree_error": (
            subset[
                "mean_absolute_numerator_degree_error"
            ].median()
        ),
        "mean_absolute_denominator_degree_error": (
            subset[
                "mean_absolute_denominator_degree_error"
            ].mean()
        ),
        "median_absolute_denominator_degree_error": (
            subset[
                "mean_absolute_denominator_degree_error"
            ].median()
        ),
        "n_exact_algebraic_matches": (
            n_exact_algebraic_matches
        ),
        "micro_exact_algebraic_recovery_fraction": safe_fraction(
            n_exact_algebraic_matches,
            total_rhs,
        ),
        "mean_model_exact_algebraic_recovery_fraction": (
            subset[
                "exact_algebraic_recovery_fraction"
            ].mean()
        ),
        "n_numerically_evaluated_rhs": (
            n_numerically_evaluated_rhs
        ),
        "n_numerically_equivalent_rhs": (
            n_numerically_equivalent_rhs
        ),
        "micro_numerical_equivalence_fraction": safe_fraction(
            n_numerically_equivalent_rhs,
            n_numerically_evaluated_rhs,
        ),
        "mean_model_numerical_equivalence_fraction": (
            subset[
                "numerical_equivalence_fraction"
            ].mean()
        ),
        "functional_form_recovery_version": (
            FUNCTIONAL_FORM_RECOVERY_VERSION
        ),
        "algebraic_recovery_version": (
            ALGEBRAIC_RECOVERY_VERSION
        ),
        "mechanistic_comparability": (
            COMPATIBLE_LABEL
        ),
    }

    return row


def summarize_compatible(
    compatible,
):
    """Summarize the primary compatible cohort by training size."""
    rows = []

    for dataset_name in TRAIN_DATASETS:
        rows.append(
            summarize_compatible_dataset(
                compatible,
                dataset_name=dataset_name,
            )
        )

    return pd.DataFrame(
        rows
    )


def summarize_comparability_group(
    frame,
    *,
    comparability,
):
    """Summarize one mechanistic comparability class across all conditions."""
    subset = frame[
        frame[
            "mechanistic_comparability"
        ]
        == comparability
    ].copy()

    total_rhs = int(
        subset[
            "n_states"
        ].sum()
    )

    n_exact = int(
        subset[
            "n_exact_algebraic_matches"
        ].sum()
    )

    n_numerically_evaluated = int(
        subset[
            "n_numerically_evaluated_rhs"
        ].sum()
    )

    n_numerically_equivalent = int(
        subset[
            "n_numerically_equivalent_rhs"
        ].sum()
    )

    return {
        "mechanistic_comparability": (
            comparability
        ),
        "n_models": (
            subset[
                "model_id"
            ].nunique()
        ),
        "n_conditions": (
            len(
                subset
            )
        ),
        "total_rhs": (
            total_rhs
        ),
        "mean_model_macro_dependency_f1": (
            subset[
                "macro_dependency_f1"
            ].mean()
        ),
        "mean_model_numerator_monomial_f1": (
            subset[
                "macro_numerator_monomial_f1"
            ].mean()
        ),
        "mean_model_interaction_f1": (
            subset[
                "macro_interaction_f1"
            ].mean()
        ),
        "mean_model_rational_structure_accuracy": (
            subset[
                "rational_structure_accuracy"
            ].mean()
        ),
        "mean_model_denominator_dependency_f1": (
            subset[
                "macro_denominator_dependency_f1"
            ].mean()
        ),
        "n_exact_algebraic_matches": (
            n_exact
        ),
        "micro_exact_algebraic_recovery_fraction": safe_fraction(
            n_exact,
            total_rhs,
        ),
        "n_numerically_evaluated_rhs": (
            n_numerically_evaluated
        ),
        "n_numerically_equivalent_rhs": (
            n_numerically_equivalent
        ),
        "micro_numerical_equivalence_fraction": safe_fraction(
            n_numerically_equivalent,
            n_numerically_evaluated,
        ),
    }


def summarize_comparability(
    integrated,
):
    """Summarize compatible and feature-space mismatch groups."""
    rows = []

    for comparability in (
        COMPATIBLE_LABEL,
        MISMATCH_LABEL,
    ):
        rows.append(
            summarize_comparability_group(
                integrated,
                comparability=comparability,
            )
        )

    return pd.DataFrame(
        rows
    )


def print_primary_summary(
    summary,
):
    """Print the main integrated recovery summary."""
    print()
    print("=" * 100)
    print(
        "INTEGRATED RECOVERY — PRIMARY COMPATIBLE COHORT"
    )
    print("=" * 100)

    display_columns = [
        "train_dataset",
        "train_trajectories",
        "n_test_success",
        "n_test_failed",
        "test_success_rate",
        "median_test_macro_nrmse_successful",
        "micro_dependency_f1",
        "mean_model_numerator_monomial_f1",
        "mean_model_interaction_f1",
        "mean_model_rational_structure_accuracy",
        "micro_numerical_equivalence_fraction",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


def print_algebraic_summary(
    summary,
):
    """Print the strict and approximate complete-function recovery summary."""
    print()
    print("=" * 100)
    print(
        "COMPLETE-FUNCTION RECOVERY — COMPATIBLE COHORT"
    )
    print("=" * 100)

    display_columns = [
        "train_dataset",
        "total_rhs",
        "n_exact_algebraic_matches",
        "micro_exact_algebraic_recovery_fraction",
        "n_numerically_equivalent_rhs",
        "n_numerically_evaluated_rhs",
        "micro_numerical_equivalence_fraction",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


def print_comparability_summary(
    summary,
):
    """Print the reporting split by mechanistic comparability."""
    print()
    print("=" * 100)
    print(
        "INTEGRATED RECOVERY — MECHANISTIC COMPARABILITY"
    )
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )


def main():
    """Create frozen integrated recovery reporting summaries."""
    require_file(
        INPUT_PATH
    )

    require_file(
        COMPATIBLE_INPUT_PATH
    )

    integrated = pd.read_csv(
        INPUT_PATH
    )

    compatible = pd.read_csv(
        COMPATIBLE_INPUT_PATH
    )

    validate_frozen_inputs(
        integrated,
        compatible,
    )

    primary_summary = summarize_compatible(
        compatible
    )

    comparability_summary = summarize_comparability(
        integrated
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    comparability_summary.to_csv(
        COMPARABILITY_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Frozen integrated input:",
        INPUT_PATH,
    )

    print(
        "Frozen compatible input:",
        COMPATIBLE_INPUT_PATH,
    )

    print(
        "Compatible models:",
        compatible[
            "model_id"
        ].nunique(),
    )

    print(
        "Compatible conditions:",
        len(
            compatible
        ),
    )

    print_primary_summary(
        primary_summary
    )

    print_algebraic_summary(
        primary_summary
    )

    print_comparability_summary(
        comparability_summary
    )

    print()
    print(
        "Primary summary output:",
        OUTPUT_PATH,
    )

    print(
        "Comparability output:",
        COMPARABILITY_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()