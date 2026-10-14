"""Aggregate frozen Selector v1 held-out test results."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "results"

SELECTOR_AGGREGATE_PATH = (
    RESULTS_ROOT
    / "selector_v1_aggregate_report.csv"
)

OUTPUT_PATH = (
    RESULTS_ROOT
    / "selector_v1_held_out_test.csv"
)

EXPERIMENT_NAME = "generic_exact_experiment"

TEST_RELATIVE_PATH = (
    Path("coordinate_selection")
    / "test"
    / "test_comparison.csv"
)


def discover_test_results():
    """Discover all model-level held-out test comparison files."""
    pattern = (
        f"BIOMD*/{EXPERIMENT_NAME}/"
        "coordinate_selection/test/"
        "test_comparison.csv"
    )

    return sorted(
        RESULTS_ROOT.glob(
            pattern
        )
    )


def load_test_results(
    test_paths,
):
    """Load and concatenate model-level held-out test summaries."""
    frames = []

    for test_path in test_paths:
        frame = pd.read_csv(
            test_path
        )

        if "model_id" not in frame.columns:
            raise ValueError(
                f"Missing model_id in {test_path}"
            )

        if "dataset" not in frame.columns:
            raise ValueError(
                f"Missing dataset in {test_path}"
            )

        frames.append(
            frame
        )

    if not frames:
        raise ValueError(
            "No held-out test results were found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def validate_test_results(
    test_results,
) -> None:
    """Validate the expected frozen held-out evaluation cohort."""
    expected_models = 10
    expected_conditions = 40
    expected_datasets = {
        "train_04",
        "train_08",
        "train_16",
        "train_32",
    }

    model_count = test_results[
        "model_id"
    ].nunique()

    if model_count != expected_models:
        raise ValueError(
            f"Found {model_count} models; "
            f"expected {expected_models}."
        )

    if len(test_results) != expected_conditions:
        raise ValueError(
            f"Found {len(test_results)} conditions; "
            f"expected {expected_conditions}."
        )

    observed_datasets = set(
        test_results[
            "dataset"
        ].unique()
    )

    if observed_datasets != expected_datasets:
        raise ValueError(
            "Unexpected training datasets: "
            f"{sorted(observed_datasets)}"
        )

    duplicate_mask = test_results.duplicated(
        subset=[
            "model_id",
            "dataset",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = test_results.loc[
            duplicate_mask,
            [
                "model_id",
                "dataset",
            ],
        ]

        raise ValueError(
            "Duplicate model/dataset conditions:\n"
            + duplicates.to_string(
                index=False
            )
        )


def build_aggregate(
    *,
    selector_results,
    test_results,
):
    """Join frozen Selector v1 metadata with held-out test outcomes."""
    selector = selector_results.copy()

    selector = selector.rename(
        columns={
            "train_dataset": "dataset",
        }
    )

    required_selector_columns = [
        "model_id",
        "n_states",
        "dataset",
        "train_trajectories",
        "candidate_space",
        "systems_evaluated",
        "search_fraction",
        "coordinate_rounds",
        "canonical_system_id",
        "selected_ranks",
    ]

    missing_columns = [
        column
        for column in required_selector_columns
        if column not in selector.columns
    ]

    if missing_columns:
        raise ValueError(
            "Selector aggregate is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    selector = selector[
        required_selector_columns
    ].copy()

    merged = test_results.merge(
        selector,
        on=[
            "model_id",
            "dataset",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged[
        "n_states"
    ].isna().any():
        missing = merged.loc[
            merged["n_states"].isna(),
            [
                "model_id",
                "dataset",
            ],
        ]

        raise ValueError(
            "Held-out conditions missing from Selector v1 aggregate:\n"
            + missing.to_string(
                index=False
            )
        )

    merged[
        "validation_test_ratio"
    ] = np.where(
        np.isfinite(
            merged[
                "validation_macro_nrmse"
            ]
        )
        & (
            merged[
                "validation_macro_nrmse"
            ]
            > 0.0
        )
        & np.isfinite(
            merged[
                "test_macro_nrmse"
            ]
        ),
        (
            merged[
                "test_macro_nrmse"
            ]
            / merged[
                "validation_macro_nrmse"
            ]
        ),
        np.nan,
    )

    merged[
        "test_minus_validation"
    ] = np.where(
        np.isfinite(
            merged[
                "validation_macro_nrmse"
            ]
        )
        & np.isfinite(
            merged[
                "test_macro_nrmse"
            ]
        ),
        (
            merged[
                "test_macro_nrmse"
            ]
            - merged[
                "validation_macro_nrmse"
            ]
        ),
        np.nan,
    )

    preferred_columns = [
        "model_id",
        "n_states",
        "dataset",
        "train_trajectories",
        "selection_method",
        "system_id",
        "canonical_system_id",
        "candidate_rank_signature",
        "selected_ranks",
        "validation_macro_nrmse",
        "test_macro_nrmse",
        "validation_test_ratio",
        "test_minus_validation",
        "test_status",
        "expected_trajectories",
        "successful_trajectories",
        "failed_trajectories",
        "candidate_space",
        "systems_evaluated",
        "search_fraction",
        "coordinate_rounds",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in merged.columns
    ]

    remaining_columns = [
        column
        for column in merged.columns
        if column not in available_columns
    ]

    return merged[
        available_columns
        + remaining_columns
    ].sort_values(
        [
            "n_states",
            "model_id",
            "train_trajectories",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    """Aggregate and validate the frozen held-out test results."""
    if not SELECTOR_AGGREGATE_PATH.exists():
        raise FileNotFoundError(
            "Missing Selector v1 aggregate: "
            f"{SELECTOR_AGGREGATE_PATH}"
        )

    test_paths = discover_test_results()

    print(
        "Held-out test files:",
        len(test_paths),
    )

    test_results = load_test_results(
        test_paths
    )

    validate_test_results(
        test_results
    )

    selector_results = pd.read_csv(
        SELECTOR_AGGREGATE_PATH
    )

    aggregate = build_aggregate(
        selector_results=selector_results,
        test_results=test_results,
    )

    aggregate.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Models:",
        aggregate[
            "model_id"
        ].nunique(),
    )

    print(
        "Conditions:",
        len(aggregate),
    )

    print(
        "Successful conditions:",
        (
            aggregate[
                "test_status"
            ]
            == "success"
        ).sum(),
    )

    print(
        "Failed conditions:",
        (
            aggregate[
                "test_status"
            ]
            == "failed"
        ).sum(),
    )

    print()
    print(
        aggregate[
            [
                "model_id",
                "n_states",
                "dataset",
                "validation_macro_nrmse",
                "test_macro_nrmse",
                "test_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
