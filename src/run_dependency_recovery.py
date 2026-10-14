"""Run frozen state-dependency recovery across the benchmark suite.

This module executes the already-defined dependency-recovery analysis for
the frozen 10-model benchmark suite and aggregates the per-model summaries.

The analysis is strictly post-hoc. It does not modify PySR inference,
candidate generation, Selector v1, validation selection, or held-out test
evaluation.
"""

from pathlib import Path

import pandas as pd

from src.evaluate_dependency_recovery import (
    TRAIN_DATASETS,
    evaluate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dependency_recovery_summary.csv"
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

EXPECTED_MODELS = len(
    FROZEN_MODEL_IDS
)

EXPECTED_DATASETS_PER_MODEL = len(
    TRAIN_DATASETS
)

EXPECTED_ROWS = (
    EXPECTED_MODELS
    * EXPECTED_DATASETS_PER_MODEL
)


def validate_model_summary(
    *,
    model_id,
    summary,
):
    """Validate one model-level dependency-recovery summary."""
    if len(
        summary
    ) != EXPECTED_DATASETS_PER_MODEL:
        raise ValueError(
            f"{model_id}: expected "
            f"{EXPECTED_DATASETS_PER_MODEL} summary rows, "
            f"found {len(summary)}."
        )

    observed_models = set(
        summary[
            "model_id"
        ].astype(
            str
        )
    )

    if observed_models != {
        model_id
    }:
        raise ValueError(
            f"{model_id}: unexpected model IDs in summary: "
            f"{sorted(observed_models)}."
        )

    observed_datasets = set(
        summary[
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
            f"{model_id}: dataset mismatch. "
            f"Expected {sorted(expected_datasets)}, "
            f"found {sorted(observed_datasets)}."
        )

    duplicate_mask = summary.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = summary.loc[
            duplicate_mask,
            [
                "model_id",
                "dataset_name",
            ],
        ]

        raise ValueError(
            f"{model_id}: duplicate summary rows:\n"
            f"{duplicates.to_string(index=False)}"
        )


def validate_aggregate(
    frame,
):
    """Validate the complete frozen dependency-recovery aggregate."""
    if len(
        frame
    ) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} aggregate rows, "
            f"found {len(frame)}."
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
        missing_models = (
            expected_models
            - observed_models
        )

        unexpected_models = (
            observed_models
            - expected_models
        )

        raise ValueError(
            "Aggregate model mismatch. "
            f"Missing={sorted(missing_models)}, "
            f"unexpected={sorted(unexpected_models)}."
        )

    duplicate_mask = frame.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = frame.loc[
            duplicate_mask,
            [
                "model_id",
                "dataset_name",
            ],
        ]

        raise ValueError(
            "Duplicate model/dataset rows detected:\n"
            f"{duplicates.to_string(index=False)}"
        )

    for model_id in FROZEN_MODEL_IDS:
        model_frame = frame[
            frame[
                "model_id"
            ]
            == model_id
        ]

        observed_datasets = set(
            model_frame[
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
                f"{model_id}: aggregate dataset mismatch. "
                f"Expected {sorted(expected_datasets)}, "
                f"found {sorted(observed_datasets)}."
            )


def run_dependency_recovery():
    """Run dependency recovery for all frozen benchmark models."""
    summaries = []

    for model_index, model_id in enumerate(
        FROZEN_MODEL_IDS,
        start=1,
    ):
        print()
        print("#" * 80)

        print(
            "FROZEN MODEL",
            f"{model_index}/{EXPECTED_MODELS}",
        )

        print(
            "Model:",
            model_id,
        )

        (
            _ground_truth,
            _detailed,
            summary,
            _detailed_path,
            _summary_path,
        ) = evaluate_model(
            model_id=model_id,
        )

        validate_model_summary(
            model_id=model_id,
            summary=summary,
        )

        summaries.append(
            summary.copy()
        )

    aggregate = pd.concat(
        summaries,
        ignore_index=True,
    )

    validate_aggregate(
        aggregate
    )

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

    aggregate[
        "_model_order"
    ] = aggregate[
        "model_id"
    ].map(
        model_order
    )

    aggregate[
        "_dataset_order"
    ] = aggregate[
        "dataset_name"
    ].map(
        dataset_order
    )

    aggregate = aggregate.sort_values(
        by=[
            "_model_order",
            "_dataset_order",
        ]
    ).drop(
        columns=[
            "_model_order",
            "_dataset_order",
        ]
    ).reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return aggregate


def print_aggregate_summary(
    frame,
):
    """Print compact diagnostics for the complete frozen aggregate."""
    compatible_count = int(
        (
            frame[
                "mechanistic_comparability"
            ]
            == "compatible"
        ).sum()
    )

    mismatch_count = int(
        (
            frame[
                "mechanistic_comparability"
            ]
            == "feature_space_mismatch"
        ).sum()
    )

    autonomous_models = (
        frame.loc[
            frame[
                "ground_truth_is_autonomous"
            ]
            == True,
            "model_id",
        ]
        .drop_duplicates()
        .nunique()
    )

    non_autonomous_models = (
        frame.loc[
            frame[
                "ground_truth_is_autonomous"
            ]
            == False,
            "model_id",
        ]
        .drop_duplicates()
        .nunique()
    )

    print()
    print("=" * 80)
    print("FROZEN DEPENDENCY RECOVERY COMPLETE")

    print(
        "Models:",
        frame[
            "model_id"
        ].nunique(),
    )

    print(
        "Training conditions:",
        frame[
            "dataset_name"
        ].nunique(),
    )

    print(
        "Model × training conditions:",
        len(
            frame
        ),
    )

    print(
        "Autonomous models:",
        autonomous_models,
    )

    print(
        "Non-autonomous models:",
        non_autonomous_models,
    )

    print(
        "Compatible conditions:",
        compatible_count,
    )

    print(
        "Feature-space-mismatch conditions:",
        mismatch_count,
    )

    print()
    print(
        "Output:",
        OUTPUT_PATH,
    )

    print()
    print(
        frame[
            [
                "model_id",
                "dataset_name",
                "n_states",
                "mechanistic_comparability",
                "total_true_positive",
                "total_false_positive",
                "total_false_negative",
                "macro_dependency_f1",
                "micro_dependency_f1",
            ]
        ].to_string(
            index=False
        )
    )


def main():
    """Run and aggregate frozen dependency-recovery evaluation."""
    frame = run_dependency_recovery()

    print_aggregate_summary(
        frame
    )


if __name__ == "__main__":
    main()