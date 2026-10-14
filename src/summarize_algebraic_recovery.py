"""Summarize frozen Algebraic Recovery v1 results.

This module performs reporting only.

It reads the frozen cohort-level Algebraic Recovery v1 output produced by
``src.run_algebraic_recovery`` and creates:

1. A primary summary restricted to models whose mechanistic ground truth is
   compatible with the states-only PySR feature space.

2. A secondary summary stratified by mechanistic comparability.

No symbolic equations are recomputed here. No inference, selection,
trajectory evaluation, held-out testing, dependency recovery,
functional-form recovery, or algebraic-recovery definitions are modified.

Primary RHS-level recovery is reported using micro aggregation:

    sum(recovered RHS) / sum(evaluable RHS)

This prevents small models and large models from receiving equal weight when
the scientific quantity of interest is the fraction of individual RHS
equations recovered.

Model-level means and medians are also retained as descriptive statistics.
"""

import numpy as np
import pandas as pd

from src.evaluate_algebraic_recovery import (
    ALGEBRAIC_RECOVERY_VERSION,
    NUMERICAL_EQUIVALENCE_ATOL,
    NUMERICAL_EQUIVALENCE_POINTS,
    NUMERICAL_EQUIVALENCE_RTOL,
    NUMERICAL_EQUIVALENCE_SEED,
)
from src.evaluate_dependency_recovery import (
    RESULTS_ROOT,
    TRAIN_DATASETS,
)


INPUT_PATH = (
    RESULTS_ROOT
    / "algebraic_recovery_summary.csv"
)

COMPATIBLE_OUTPUT_PATH = (
    RESULTS_ROOT
    / "algebraic_recovery_compatible_summary.csv"
)

COMPARABILITY_OUTPUT_PATH = (
    RESULTS_ROOT
    / "algebraic_recovery_comparability_summary.csv"
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


def safe_fraction(
    numerator,
    denominator,
):
    """Return a fraction or NaN when the denominator is zero."""
    if denominator == 0:
        return float(
            "nan"
        )

    return (
        float(
            numerator
        )
        / float(
            denominator
        )
    )


def validate_frozen_input(
    frame,
):
    """Validate the frozen Algebraic Recovery v1 aggregate."""
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
        - set(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Frozen Algebraic Recovery aggregate is missing "
            f"columns: {sorted(missing_columns)}."
        )

    if len(
        frame
    ) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CONDITION_COUNT} frozen "
            f"conditions, found {len(frame)}."
        )

    if frame[
        "model_id"
    ].nunique() != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} models, found "
            f"{frame['model_id'].nunique()}."
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
            "Duplicate model/dataset conditions detected:\n"
            f"{duplicate_rows.to_string(index=False)}"
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
            "Unexpected Algebraic Recovery versions: "
            f"{sorted(observed_versions)}."
        )

    observed_datasets = set(
        frame[
            "dataset_name"
        ].astype(
            str
        )
    )

    expected_datasets = set(
        TRAIN_DATASETS
    )

    if observed_datasets != expected_datasets:
        raise ValueError(
            "Training-dataset mismatch. "
            f"Observed={sorted(observed_datasets)}, "
            f"expected={sorted(expected_datasets)}."
        )

    dataset_counts = (
        frame.groupby(
            "model_id"
        )[
            "dataset_name"
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
            "Every model must contain exactly one condition "
            "for each frozen training dataset."
        )

    observed_comparability = set(
        frame[
            "mechanistic_comparability"
        ].astype(
            str
        )
    )

    expected_comparability = {
        COMPATIBLE_LABEL,
        MISMATCH_LABEL,
    }

    if observed_comparability != expected_comparability:
        raise ValueError(
            "Unexpected mechanistic comparability classes. "
            f"Observed={sorted(observed_comparability)}, "
            f"expected={sorted(expected_comparability)}."
        )

    if not np.allclose(
        frame[
            "numerical_equivalence_rtol"
        ].astype(
            float
        ),
        NUMERICAL_EQUIVALENCE_RTOL,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Numerical-equivalence rtol does not match "
            "the frozen Algebraic Recovery v1 definition."
        )

    if not np.allclose(
        frame[
            "numerical_equivalence_atol"
        ].astype(
            float
        ),
        NUMERICAL_EQUIVALENCE_ATOL,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Numerical-equivalence atol does not match "
            "the frozen Algebraic Recovery v1 definition."
        )

    if not (
        frame[
            "numerical_equivalence_requested_points"
        ].astype(
            int
        )
        == NUMERICAL_EQUIVALENCE_POINTS
    ).all():
        raise ValueError(
            "Numerical-equivalence point count does not match "
            "the frozen Algebraic Recovery v1 definition."
        )

    if not (
        frame[
            "numerical_equivalence_seed"
        ].astype(
            int
        )
        == NUMERICAL_EQUIVALENCE_SEED
    ).all():
        raise ValueError(
            "Numerical-equivalence seed does not match "
            "the frozen Algebraic Recovery v1 definition."
        )

    compatible = frame[
        frame[
            "mechanistic_comparability"
        ]
        == COMPATIBLE_LABEL
    ]

    mismatch = frame[
        frame[
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


def summarize_group(
    frame,
):
    """Summarize one set of model/dataset conditions."""
    n_conditions = len(
        frame
    )

    n_models = frame[
        "model_id"
    ].nunique()

    total_rhs = int(
        frame[
            "n_states"
        ].sum()
    )

    exact_matches = int(
        frame[
            "n_exact_algebraic_matches"
        ].sum()
    )

    numerically_evaluated_rhs = int(
        frame[
            "n_numerically_evaluated_rhs"
        ].sum()
    )

    numerically_equivalent_rhs = int(
        frame[
            "n_numerically_equivalent_rhs"
        ].sum()
    )

    exact_complete_systems = int(
        frame[
            "all_rhs_exact_algebraic_match"
        ]
        .astype(
            bool
        )
        .sum()
    )

    numerical_complete_systems = int(
        frame[
            "all_evaluable_rhs_numerically_equivalent"
        ]
        .astype(
            bool
        )
        .sum()
    )

    exact_model_fractions = frame[
        "exact_algebraic_recovery_fraction"
    ].astype(
        float
    )

    numerical_model_fractions = frame[
        "numerical_equivalence_fraction"
    ].astype(
        float
    )

    return {
        "algebraic_recovery_version": (
            ALGEBRAIC_RECOVERY_VERSION
        ),
        "n_models": n_models,
        "n_conditions": n_conditions,
        "total_rhs": total_rhs,
        "n_exact_algebraic_matches": (
            exact_matches
        ),
        "micro_exact_algebraic_recovery_fraction": (
            safe_fraction(
                exact_matches,
                total_rhs,
            )
        ),
        "mean_model_exact_algebraic_recovery_fraction": (
            exact_model_fractions.mean()
        ),
        "median_model_exact_algebraic_recovery_fraction": (
            exact_model_fractions.median()
        ),
        "n_complete_exact_systems": (
            exact_complete_systems
        ),
        "complete_exact_system_fraction": (
            safe_fraction(
                exact_complete_systems,
                n_conditions,
            )
        ),
        "n_numerically_evaluated_rhs": (
            numerically_evaluated_rhs
        ),
        "n_numerically_equivalent_rhs": (
            numerically_equivalent_rhs
        ),
        "micro_numerical_equivalence_fraction": (
            safe_fraction(
                numerically_equivalent_rhs,
                numerically_evaluated_rhs,
            )
        ),
        "mean_model_numerical_equivalence_fraction": (
            numerical_model_fractions.mean()
        ),
        "median_model_numerical_equivalence_fraction": (
            numerical_model_fractions.median()
        ),
        "n_complete_numerically_equivalent_systems": (
            numerical_complete_systems
        ),
        "complete_numerical_equivalence_system_fraction": (
            safe_fraction(
                numerical_complete_systems,
                n_conditions,
            )
        ),
        "numerical_equivalence_rtol": (
            NUMERICAL_EQUIVALENCE_RTOL
        ),
        "numerical_equivalence_atol": (
            NUMERICAL_EQUIVALENCE_ATOL
        ),
        "numerical_equivalence_requested_points": (
            NUMERICAL_EQUIVALENCE_POINTS
        ),
        "numerical_equivalence_seed": (
            NUMERICAL_EQUIVALENCE_SEED
        ),
    }


def summarize_compatible(
    frame,
):
    """Create the primary summary for feature-space-compatible models."""
    compatible = frame[
        frame[
            "mechanistic_comparability"
        ]
        == COMPATIBLE_LABEL
    ].copy()

    rows = []

    for dataset_name in TRAIN_DATASETS:
        dataset_frame = compatible[
            compatible[
                "dataset_name"
            ]
            == dataset_name
        ]

        if len(
            dataset_frame
        ) != EXPECTED_COMPATIBLE_MODEL_COUNT:
            raise ValueError(
                f"{dataset_name}: expected "
                f"{EXPECTED_COMPATIBLE_MODEL_COUNT} compatible "
                f"models, found {len(dataset_frame)}."
            )

        row = summarize_group(
            dataset_frame
        )

        row[
            "dataset_name"
        ] = dataset_name

        row[
            "mechanistic_comparability"
        ] = COMPATIBLE_LABEL

        rows.append(
            row
        )

    columns = [
        "algebraic_recovery_version",
        "dataset_name",
        "mechanistic_comparability",
        "n_models",
        "n_conditions",
        "total_rhs",
        "n_exact_algebraic_matches",
        "micro_exact_algebraic_recovery_fraction",
        "mean_model_exact_algebraic_recovery_fraction",
        "median_model_exact_algebraic_recovery_fraction",
        "n_complete_exact_systems",
        "complete_exact_system_fraction",
        "n_numerically_evaluated_rhs",
        "n_numerically_equivalent_rhs",
        "micro_numerical_equivalence_fraction",
        "mean_model_numerical_equivalence_fraction",
        "median_model_numerical_equivalence_fraction",
        "n_complete_numerically_equivalent_systems",
        "complete_numerical_equivalence_system_fraction",
        "numerical_equivalence_rtol",
        "numerical_equivalence_atol",
        "numerical_equivalence_requested_points",
        "numerical_equivalence_seed",
    ]

    return pd.DataFrame(
        rows
    )[
        columns
    ]


def summarize_comparability(
    frame,
):
    """Create an overall summary stratified by comparability class."""
    rows = []

    for comparability in (
        COMPATIBLE_LABEL,
        MISMATCH_LABEL,
    ):
        subset = frame[
            frame[
                "mechanistic_comparability"
            ]
            == comparability
        ]

        row = summarize_group(
            subset
        )

        row[
            "mechanistic_comparability"
        ] = comparability

        rows.append(
            row
        )

    columns = [
        "algebraic_recovery_version",
        "mechanistic_comparability",
        "n_models",
        "n_conditions",
        "total_rhs",
        "n_exact_algebraic_matches",
        "micro_exact_algebraic_recovery_fraction",
        "mean_model_exact_algebraic_recovery_fraction",
        "median_model_exact_algebraic_recovery_fraction",
        "n_complete_exact_systems",
        "complete_exact_system_fraction",
        "n_numerically_evaluated_rhs",
        "n_numerically_equivalent_rhs",
        "micro_numerical_equivalence_fraction",
        "mean_model_numerical_equivalence_fraction",
        "median_model_numerical_equivalence_fraction",
        "n_complete_numerically_equivalent_systems",
        "complete_numerical_equivalence_system_fraction",
        "numerical_equivalence_rtol",
        "numerical_equivalence_atol",
        "numerical_equivalence_requested_points",
        "numerical_equivalence_seed",
    ]

    return pd.DataFrame(
        rows
    )[
        columns
    ]


def print_compatible_summary(
    summary,
):
    """Print the primary compatible-model summary."""
    print()
    print("=" * 80)
    print(
        "ALGEBRAIC RECOVERY v1 — COMPATIBLE MODELS"
    )
    print("=" * 80)

    display_columns = [
        "dataset_name",
        "n_models",
        "total_rhs",
        "n_exact_algebraic_matches",
        "micro_exact_algebraic_recovery_fraction",
        "n_numerically_equivalent_rhs",
        "n_numerically_evaluated_rhs",
        "micro_numerical_equivalence_fraction",
        "n_complete_numerically_equivalent_systems",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Compatible output:",
        COMPATIBLE_OUTPUT_PATH,
    )


def print_comparability_summary(
    summary,
):
    """Print the secondary comparability-stratified summary."""
    print()
    print("=" * 80)
    print(
        "ALGEBRAIC RECOVERY v1 — COMPARABILITY"
    )
    print("=" * 80)

    display_columns = [
        "mechanistic_comparability",
        "n_models",
        "n_conditions",
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

    print()
    print(
        "Comparability output:",
        COMPARABILITY_OUTPUT_PATH,
    )


def main():
    """Generate frozen Algebraic Recovery v1 reporting summaries."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Frozen Algebraic Recovery aggregate was not found: "
            f"{INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH
    )

    validate_frozen_input(
        frame
    )

    compatible_summary = summarize_compatible(
        frame
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
    print(
        "Frozen input:",
        INPUT_PATH,
    )

    print(
        "Algebraic Recovery version:",
        ALGEBRAIC_RECOVERY_VERSION,
    )

    print(
        "Models:",
        frame[
            "model_id"
        ].nunique(),
    )

    print(
        "Conditions:",
        len(
            frame
        ),
    )

    print(
        "Compatible models:",
        frame.loc[
            frame[
                "mechanistic_comparability"
            ]
            == COMPATIBLE_LABEL,
            "model_id",
        ].nunique(),
    )

    print(
        "Feature-space mismatch models:",
        frame.loc[
            frame[
                "mechanistic_comparability"
            ]
            == MISMATCH_LABEL,
            "model_id",
        ].nunique(),
    )

    print_compatible_summary(
        compatible_summary
    )

    print_comparability_summary(
        comparability_summary
    )


if __name__ == "__main__":
    main()