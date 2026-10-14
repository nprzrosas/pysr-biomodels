"""Integrate frozen dynamic and mechanistic recovery results.

This module creates a condition-level master table joining the frozen results
from:

1. Selector v1 and held-out trajectory evaluation.
2. Dependency Recovery.
3. Functional-form Recovery v1.
4. Algebraic Recovery v1.

Each output row represents one frozen experimental condition:

    model_id x train_dataset

The integration is reporting-only. No PySR fitting, candidate selection,
trajectory simulation, held-out testing, or mechanistic recovery metric is
recomputed here.

Two outputs are produced:

    results/integrated_recovery.csv

        All 10 frozen models x 4 training conditions = 40 rows.

    results/integrated_recovery_compatible.csv

        The 8 models whose true RHS equations are compatible with the
        states-only inference feature space x 4 training conditions = 32 rows.

The two non-autonomous models remain in the complete table with
mechanistic_comparability="feature_space_mismatch", but they are excluded
from the primary mechanistically comparable table.
"""

from pathlib import Path

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

SELECTOR_PATH = (
    RESULTS_ROOT
    / "selector_v1_aggregate_report.csv"
)

HELD_OUT_TEST_PATH = (
    RESULTS_ROOT
    / "selector_v1_held_out_test.csv"
)

DEPENDENCY_PATH = (
    RESULTS_ROOT
    / "dependency_recovery_summary.csv"
)

FUNCTIONAL_FORM_PATH = (
    RESULTS_ROOT
    / "functional_form_recovery_summary.csv"
)

ALGEBRAIC_PATH = (
    RESULTS_ROOT
    / "algebraic_recovery_summary.csv"
)

OUTPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery.csv"
)

COMPATIBLE_OUTPUT_PATH = (
    RESULTS_ROOT
    / "integrated_recovery_compatible.csv"
)


def require_file(
    path,
):
    """Require one frozen input file."""
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


def normalize_dataset_column(
    frame,
    *,
    source_name,
):
    """Normalize the frozen training-condition column to train_dataset."""
    frame = frame.copy()

    candidate_columns = [
        column
        for column in (
            "train_dataset",
            "dataset_name",
            "dataset",
        )
        if column in frame.columns
    ]

    if not candidate_columns:
        raise ValueError(
            f"{source_name} has none of the recognized "
            "training-condition columns: "
            "train_dataset, dataset_name, dataset."
        )

    reference_column = candidate_columns[
        0
    ]

    reference_values = frame[
        reference_column
    ].astype(
        str
    )

    for column in candidate_columns[
        1:
    ]:
        comparison_values = frame[
            column
        ].astype(
            str
        )

        if not reference_values.equals(
            comparison_values
        ):
            raise ValueError(
                f"{source_name} contains conflicting "
                "training-condition columns: "
                f"{reference_column} and {column}."
            )

    if reference_column != "train_dataset":
        frame = frame.rename(
            columns={
                reference_column: (
                    "train_dataset"
                ),
            }
        )

    redundant_columns = [
        column
        for column in candidate_columns
        if column != reference_column
    ]

    if redundant_columns:
        frame = frame.drop(
            columns=redundant_columns
        )

    return frame


def validate_condition_keys(
    frame,
    *,
    source_name,
):
    """Validate the frozen 10-model x 4-condition key space."""
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
    ) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"{source_name}: expected "
            f"{EXPECTED_CONDITION_COUNT} conditions, "
            f"found {len(frame)}."
        )

    if frame[
        "model_id"
    ].nunique() != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"{source_name}: expected "
            f"{EXPECTED_MODEL_COUNT} models, found "
            f"{frame['model_id'].nunique()}."
        )

    observed_models = set(
        frame[
            "model_id"
        ].astype(
            str
        )
    )

    expected_models = set(
        FROZEN_MODEL_IDS
    )

    if observed_models != expected_models:
        raise ValueError(
            f"{source_name}: frozen model cohort mismatch. "
            f"Observed={sorted(observed_models)}, "
            f"expected={sorted(expected_models)}."
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
        == EXPECTED_DATASETS_PER_MODEL
    ).all():
        raise ValueError(
            f"{source_name}: every model must have exactly "
            f"{EXPECTED_DATASETS_PER_MODEL} training conditions."
        )


def parse_train_trajectories(
    train_dataset,
):
    """Convert train_04/train_08/train_16/train_32 to trajectory counts."""
    train_dataset = str(
        train_dataset
    )

    prefix = (
        "train_"
    )

    if not train_dataset.startswith(
        prefix
    ):
        raise ValueError(
            "Unexpected training-dataset label: "
            f"{train_dataset}"
        )

    value = train_dataset[
        len(
            prefix
        ):
    ]

    try:
        return int(
            value
        )

    except ValueError as error:
        raise ValueError(
            "Could not parse training trajectory count from "
            f"{train_dataset}."
        ) from error


def load_selector_results():
    """Load the frozen Selector v1 aggregate."""
    require_file(
        SELECTOR_PATH
    )

    frame = pd.read_csv(
        SELECTOR_PATH
    )

    frame = normalize_dataset_column(
        frame,
        source_name="Selector v1",
    )

    validate_condition_keys(
        frame,
        source_name="Selector v1",
    )

    require_columns(
        frame,
        {
            "model_id",
            "n_states",
            "train_dataset",
            "train_trajectories",
            "candidate_space",
            "systems_evaluated",
            "search_fraction",
            "coordinate_rounds",
            "canonical_system_id",
            "selected_ranks",
            "validation_macro_nrmse",
            "validation_status",
        },
        source_name="Selector v1",
    )

    columns = [
        "model_id",
        "n_states",
        "train_dataset",
        "train_trajectories",
        "candidate_space",
        "systems_evaluated",
        "search_fraction",
        "coordinate_rounds",
        "canonical_system_id",
        "selected_ranks",
        "validation_macro_nrmse",
        "validation_status",
    ]

    return frame[
        columns
    ].copy()


def load_held_out_test_results():
    """Load the frozen held-out trajectory evaluation."""
    require_file(
        HELD_OUT_TEST_PATH
    )

    frame = pd.read_csv(
        HELD_OUT_TEST_PATH
    )

    frame = normalize_dataset_column(
        frame,
        source_name="Held-out test",
    )

    validate_condition_keys(
        frame,
        source_name="Held-out test",
    )

    require_columns(
        frame,
        {
            "model_id",
            "train_dataset",
            "validation_macro_nrmse",
            "test_macro_nrmse",
            "test_status",
            "validation_test_ratio",
            "test_minus_validation",
        },
        source_name="Held-out test",
    )

    columns = [
        "model_id",
        "train_dataset",
        "validation_macro_nrmse",
        "test_macro_nrmse",
        "test_status",
        "validation_test_ratio",
        "test_minus_validation",
    ]

    frame = frame[
        columns
    ].copy()

    frame = frame.rename(
        columns={
            "validation_macro_nrmse": (
                "held_out_source_validation_macro_nrmse"
            ),
        }
    )

    return frame


def load_dependency_results():
    """Load frozen Dependency Recovery results."""
    require_file(
        DEPENDENCY_PATH
    )

    frame = pd.read_csv(
        DEPENDENCY_PATH
    )

    frame = normalize_dataset_column(
        frame,
        source_name="Dependency Recovery",
    )

    validate_condition_keys(
        frame,
        source_name="Dependency Recovery",
    )

    require_columns(
        frame,
        {
            "model_id",
            "train_dataset",
            "n_states",
            "ground_truth_is_autonomous",
            "ground_truth_independent_variables",
            "inference_feature_space",
            "mechanistic_comparability",
            "total_true_positive",
            "total_false_positive",
            "total_false_negative",
            "macro_dependency_precision",
            "macro_dependency_recall",
            "macro_dependency_f1",
            "micro_dependency_precision",
            "micro_dependency_recall",
            "micro_dependency_f1",
        },
        source_name="Dependency Recovery",
    )

    columns = [
        "model_id",
        "train_dataset",
        "n_states",
        "ground_truth_is_autonomous",
        "ground_truth_independent_variables",
        "inference_feature_space",
        "mechanistic_comparability",
        "total_true_positive",
        "total_false_positive",
        "total_false_negative",
        "macro_dependency_precision",
        "macro_dependency_recall",
        "macro_dependency_f1",
        "micro_dependency_precision",
        "micro_dependency_recall",
        "micro_dependency_f1",
    ]

    return frame[
        columns
    ].copy()


def load_functional_form_results():
    """Load frozen Functional-form Recovery v1 results."""
    require_file(
        FUNCTIONAL_FORM_PATH
    )

    frame = pd.read_csv(
        FUNCTIONAL_FORM_PATH
    )

    frame = normalize_dataset_column(
        frame,
        source_name="Functional-form Recovery",
    )

    validate_condition_keys(
        frame,
        source_name="Functional-form Recovery",
    )

    require_columns(
        frame,
        {
            "functional_form_recovery_version",
            "model_id",
            "train_dataset",
            "n_states",
            "ground_truth_is_autonomous",
            "mechanistic_comparability",
            "macro_numerator_monomial_f1",
            "macro_interaction_f1",
            "rational_structure_accuracy",
            "macro_denominator_dependency_f1",
            "mean_absolute_numerator_degree_error",
            "mean_absolute_denominator_degree_error",
        },
        source_name="Functional-form Recovery",
    )

    observed_versions = set(
        frame[
            "functional_form_recovery_version"
        ].astype(
            str
        )
    )

    if observed_versions != {
        FUNCTIONAL_FORM_RECOVERY_VERSION
    }:
        raise ValueError(
            "Functional-form Recovery version mismatch. "
            f"Observed={sorted(observed_versions)}, "
            f"expected={FUNCTIONAL_FORM_RECOVERY_VERSION}."
        )

    columns = [
        "model_id",
        "train_dataset",
        "functional_form_recovery_version",
        "n_states",
        "ground_truth_is_autonomous",
        "mechanistic_comparability",
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "macro_denominator_dependency_f1",
        "mean_absolute_numerator_degree_error",
        "mean_absolute_denominator_degree_error",
    ]

    return frame[
        columns
    ].copy()


def load_algebraic_results():
    """Load frozen Algebraic Recovery v1 results."""
    require_file(
        ALGEBRAIC_PATH
    )

    frame = pd.read_csv(
        ALGEBRAIC_PATH
    )

    frame = normalize_dataset_column(
        frame,
        source_name="Algebraic Recovery",
    )

    validate_condition_keys(
        frame,
        source_name="Algebraic Recovery",
    )

    require_columns(
        frame,
        {
            "algebraic_recovery_version",
            "model_id",
            "train_dataset",
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
        },
        source_name="Algebraic Recovery",
    )

    observed_versions = set(
        frame[
            "algebraic_recovery_version"
        ].astype(
            str
        )
    )

    if observed_versions != {
        ALGEBRAIC_RECOVERY_VERSION
    }:
        raise ValueError(
            "Algebraic Recovery version mismatch. "
            f"Observed={sorted(observed_versions)}, "
            f"expected={ALGEBRAIC_RECOVERY_VERSION}."
        )

    columns = [
        "model_id",
        "train_dataset",
        "algebraic_recovery_version",
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
    ]

    return frame[
        columns
    ].copy()


def require_matching_values(
    frame,
    *,
    left_column,
    right_column,
    label,
):
    """Require two joined provenance columns to agree."""
    left = frame[
        left_column
    ]

    right = frame[
        right_column
    ]

    both_missing = (
        left.isna()
        & right.isna()
    )

    equal = (
        left.eq(
            right
        )
        | both_missing
    )

    if not equal.all():
        mismatch = frame.loc[
            ~equal,
            [
                "model_id",
                "train_dataset",
                left_column,
                right_column,
            ],
        ]

        raise ValueError(
            f"Joined provenance mismatch for {label}:\n"
            f"{mismatch.to_string(index=False)}"
        )


def integrate_results():
    """Join all frozen recovery layers into one condition-level table."""
    selector = load_selector_results()

    held_out = load_held_out_test_results()

    dependency = load_dependency_results()

    functional = load_functional_form_results()

    algebraic = load_algebraic_results()

    integrated = selector.merge(
        held_out,
        on=[
            "model_id",
            "train_dataset",
        ],
        how="inner",
        validate="one_to_one",
    )

    require_matching_values(
        integrated,
        left_column="validation_macro_nrmse",
        right_column="held_out_source_validation_macro_nrmse",
        label="validation_macro_nrmse",
    )

    integrated = integrated.drop(
        columns=[
            "held_out_source_validation_macro_nrmse",
        ]
    )

    integrated = integrated.merge(
        dependency,
        on=[
            "model_id",
            "train_dataset",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_dependency",
        ),
    )

    require_matching_values(
        integrated,
        left_column="n_states",
        right_column="n_states_dependency",
        label="n_states",
    )

    integrated = integrated.drop(
        columns=[
            "n_states_dependency",
        ]
    )

    integrated = integrated.merge(
        functional,
        on=[
            "model_id",
            "train_dataset",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_functional",
        ),
    )

    require_matching_values(
        integrated,
        left_column="n_states",
        right_column="n_states_functional",
        label="n_states",
    )

    require_matching_values(
        integrated,
        left_column="ground_truth_is_autonomous",
        right_column="ground_truth_is_autonomous_functional",
        label="ground_truth_is_autonomous",
    )

    require_matching_values(
        integrated,
        left_column="mechanistic_comparability",
        right_column="mechanistic_comparability_functional",
        label="mechanistic_comparability",
    )

    integrated = integrated.drop(
        columns=[
            "n_states_functional",
            "ground_truth_is_autonomous_functional",
            "mechanistic_comparability_functional",
        ]
    )

    integrated = integrated.merge(
        algebraic,
        on=[
            "model_id",
            "train_dataset",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_algebraic",
        ),
    )

    require_matching_values(
        integrated,
        left_column="n_states",
        right_column="n_states_algebraic",
        label="n_states",
    )

    require_matching_values(
        integrated,
        left_column="ground_truth_is_autonomous",
        right_column="ground_truth_is_autonomous_algebraic",
        label="ground_truth_is_autonomous",
    )

    require_matching_values(
        integrated,
        left_column="mechanistic_comparability",
        right_column="mechanistic_comparability_algebraic",
        label="mechanistic_comparability",
    )

    integrated = integrated.drop(
        columns=[
            "n_states_algebraic",
            "ground_truth_is_autonomous_algebraic",
            "mechanistic_comparability_algebraic",
        ]
    )

    expected_train_trajectories = integrated[
        "train_dataset"
    ].map(
        parse_train_trajectories
    )

    observed_train_trajectories = integrated[
        "train_trajectories"
    ].astype(
        int
    )

    if not observed_train_trajectories.equals(
        expected_train_trajectories.astype(
            int
        )
    ):
        mismatch = integrated.loc[
            observed_train_trajectories
            != expected_train_trajectories,
            [
                "model_id",
                "train_dataset",
                "train_trajectories",
            ],
        ]

        raise ValueError(
            "train_trajectories does not agree with "
            "train_dataset:\n"
            f"{mismatch.to_string(index=False)}"
        )

    validate_condition_keys(
        integrated,
        source_name="Integrated recovery",
    )

    return integrated


def validate_integrated_results(
    integrated,
):
    """Validate the final integrated recovery table."""
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
        "mechanistic_comparability",
        "macro_dependency_f1",
        "micro_dependency_precision",
        "micro_dependency_recall",
        "micro_dependency_f1",
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "macro_denominator_dependency_f1",
        "mean_absolute_numerator_degree_error",
        "mean_absolute_denominator_degree_error",
        "exact_algebraic_recovery_fraction",
        "numerical_equivalence_fraction",
    }

    require_columns(
        integrated,
        required_columns,
        source_name="Integrated recovery",
    )

    compatible = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == COMPATIBLE_LABEL
    ]

    mismatch = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == MISMATCH_LABEL
    ]

    if compatible[
        "model_id"
    ].nunique() != EXPECTED_COMPATIBLE_MODEL_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_MODEL_COUNT} compatible "
            "models, found "
            f"{compatible['model_id'].nunique()}."
        )

    if len(
        compatible
    ) != EXPECTED_COMPATIBLE_CONDITION_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMPATIBLE_CONDITION_COUNT} compatible "
            f"conditions, found {len(compatible)}."
        )

    if mismatch[
        "model_id"
    ].nunique() != EXPECTED_MISMATCH_MODEL_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_MISMATCH_MODEL_COUNT} feature-space "
            "mismatch models, found "
            f"{mismatch['model_id'].nunique()}."
        )

    if len(
        mismatch
    ) != EXPECTED_MISMATCH_CONDITION_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_MISMATCH_CONDITION_COUNT} feature-space "
            f"mismatch conditions, found {len(mismatch)}."
        )

    if not (
        compatible[
            "ground_truth_is_autonomous"
        ].astype(
            bool
        )
    ).all():
        raise ValueError(
            "Compatible cohort unexpectedly contains "
            "non-autonomous ground truth."
        )

    if (
        mismatch[
            "ground_truth_is_autonomous"
        ].astype(
            bool
        )
    ).any():
        raise ValueError(
            "Feature-space mismatch cohort unexpectedly "
            "contains autonomous ground truth."
        )


def order_integrated_results(
    integrated,
):
    """Return the master table in a stable scientific reporting order."""
    model_order = {
        model_id: index
        for index, model_id in enumerate(
            FROZEN_MODEL_IDS
        )
    }

    dataset_order = {
        dataset_name: index
        for index, dataset_name in enumerate(
            TRAIN_DATASETS
        )
    }

    ordered = integrated.copy()

    ordered[
        "_model_order"
    ] = ordered[
        "model_id"
    ].map(
        model_order
    )

    ordered[
        "_dataset_order"
    ] = ordered[
        "train_dataset"
    ].map(
        dataset_order
    )

    ordered = ordered.sort_values(
        by=[
            "_model_order",
            "_dataset_order",
        ],
        kind="stable",
    )

    ordered = ordered.drop(
        columns=[
            "_model_order",
            "_dataset_order",
        ]
    )

    ordered = ordered.reset_index(
        drop=True
    )

    return ordered


def print_summary(
    integrated,
    compatible,
):
    """Print a compact integrity summary for the integrated outputs."""
    print()
    print("=" * 80)
    print("INTEGRATED RECOVERY RESULTS")
    print("=" * 80)

    print(
        "All models:",
        integrated[
            "model_id"
        ].nunique(),
    )

    print(
        "All conditions:",
        len(
            integrated
        ),
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

    mismatch = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == MISMATCH_LABEL
    ]

    print(
        "Feature-space mismatch models:",
        mismatch[
            "model_id"
        ].nunique(),
    )

    print(
        "Feature-space mismatch conditions:",
        len(
            mismatch
        ),
    )

    print()
    print(
        "Conditions by training size:"
    )

    print(
        integrated[
            "train_dataset"
        ]
        .value_counts()
        .reindex(
            TRAIN_DATASETS
        )
        .to_string()
    )

    print()
    print(
        "Held-out test status:"
    )

    print(
        integrated[
            "test_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Mechanistic comparability:"
    )

    print(
        integrated[
            "mechanistic_comparability"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Primary compatible cohort by training size:"
    )

    display_columns = [
        "train_dataset",
        "model_id",
        "n_states",
        "test_status",
        "test_macro_nrmse",
        "micro_dependency_f1",
        "macro_numerator_monomial_f1",
        "macro_interaction_f1",
        "rational_structure_accuracy",
        "numerical_equivalence_fraction",
    ]

    print(
        compatible[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Complete output:",
        OUTPUT_PATH,
    )

    print(
        "Compatible output:",
        COMPATIBLE_OUTPUT_PATH,
    )


def main():
    """Build the frozen integrated recovery tables."""
    integrated = integrate_results()

    validate_integrated_results(
        integrated
    )

    integrated = order_integrated_results(
        integrated
    )

    compatible = integrated[
        integrated[
            "mechanistic_comparability"
        ]
        == COMPATIBLE_LABEL
    ].copy()

    compatible = compatible.reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    compatible.to_csv(
        COMPATIBLE_OUTPUT_PATH,
        index=False,
    )

    print_summary(
        integrated,
        compatible,
    )


if __name__ == "__main__":
    main()