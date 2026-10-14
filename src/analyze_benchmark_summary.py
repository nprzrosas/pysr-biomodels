"""Analyze the frozen master PySR benchmark summary.

Purpose
-------
Summarize how increasing the number of training trajectories affects:

1. validation trajectory error,
2. held-out test integration success,
3. held-out test error among successful systems,
4. mechanistic recovery for feature-space-compatible models.

Input
-----
results/benchmark_summary.csv

Outputs
-------
results/benchmark_analysis/
    dynamic_summary_by_training_size.csv
    mechanistic_summary_by_training_size.csv
    model_level_changes_04_to_32.csv
    benchmark_analysis_summary.txt

This script is analysis/reporting-only. It does not rerun any scientific
computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

DEFAULT_INPUT = RESULTS_ROOT / "benchmark_summary.csv"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "benchmark_analysis"

TRAINING_SIZES = (
    4,
    8,
    16,
    32,
)

MECHANISTIC_METRICS = (
    "macro_dependency_f1",
    "macro_numerator_monomial_f1",
    "macro_interaction_f1",
    "rational_structure_accuracy",
    "macro_denominator_dependency_f1",
)

REQUIRED_COLUMNS = (
    "model_id",
    "n_states",
    "dataset_name",
    "training_trajectories",
    "validation_macro_nrmse",
    "test_macro_nrmse",
    "mechanistic_comparability",
    "macro_dependency_f1",
    "macro_numerator_monomial_f1",
    "macro_interaction_f1",
    "rational_structure_accuracy",
    "macro_denominator_dependency_f1",
    "exact_algebraic_recovery_fraction",
    "numerical_equivalence_fraction",
    "dynamic_fidelity_macro_nrmse",
    "portability_status",
)


def require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Require the master table to contain the expected columns."""

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Master benchmark table is missing required columns: "
            + ", ".join(missing)
        )


def load_master_table(
    path: Path,
) -> pd.DataFrame:
    """Load and validate the frozen master benchmark table."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Master benchmark table not found: {path}"
        )

    frame = pd.read_csv(path)

    require_columns(
        frame,
        REQUIRED_COLUMNS,
    )

    if len(frame) != 40:
        raise ValueError(
            f"Expected 40 benchmark rows, found {len(frame)}."
        )

    duplicates = frame.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate model/dataset rows found in master table."
        )

    observed_sizes = set(
        frame["training_trajectories"]
    )

    if observed_sizes != set(TRAINING_SIZES):
        raise ValueError(
            "Unexpected training sizes. "
            f"Expected {TRAINING_SIZES}, "
            f"found {sorted(observed_sizes)}."
        )

    counts = (
        frame["training_trajectories"]
        .value_counts()
        .to_dict()
    )

    for training_size in TRAINING_SIZES:
        if counts.get(training_size, 0) != 10:
            raise ValueError(
                f"Expected 10 rows for training size "
                f"{training_size}, found "
                f"{counts.get(training_size, 0)}."
            )

    return frame


def finite_values(
    series: pd.Series,
) -> pd.Series:
    """Return finite numeric values from a series."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric[
        np.isfinite(numeric)
    ]


def summarize_numeric(
    series: pd.Series,
    prefix: str,
) -> dict:
    """Calculate robust descriptive statistics for finite values."""

    values = finite_values(series)

    if values.empty:
        return {
            f"{prefix}_n_finite": 0,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    return {
        f"{prefix}_n_finite": len(values),
        f"{prefix}_median": values.median(),
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_mean": values.mean(),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
    }


def build_dynamic_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize validation and held-out behavior by training size."""

    rows = []

    for training_size in TRAINING_SIZES:
        group = frame.loc[
            frame["training_trajectories"]
            == training_size
        ].copy()

        validation = pd.to_numeric(
            group["validation_macro_nrmse"],
            errors="coerce",
        )

        test = pd.to_numeric(
            group["test_macro_nrmse"],
            errors="coerce",
        )

        finite_validation_mask = np.isfinite(
            validation
        )
        finite_test_mask = np.isfinite(
            test
        )

        row = {
            "training_trajectories": training_size,
            "n_models": group["model_id"].nunique(),
            "validation_successful_systems":
                int(finite_validation_mask.sum()),
            "validation_failed_systems":
                int((~finite_validation_mask).sum()),
            "validation_success_fraction":
                float(finite_validation_mask.mean()),
            "test_successful_systems":
                int(finite_test_mask.sum()),
            "test_failed_systems":
                int((~finite_test_mask).sum()),
            "test_success_fraction":
                float(finite_test_mask.mean()),
            "portability_passes":
                int(
                    group["portability_status"]
                    .eq("pass")
                    .sum()
                ),
            "portability_pass_fraction":
                float(
                    group["portability_status"]
                    .eq("pass")
                    .mean()
                ),
        }

        row.update(
            summarize_numeric(
                group["validation_macro_nrmse"],
                "validation_nrmse",
            )
        )

        row.update(
            summarize_numeric(
                group["test_macro_nrmse"],
                "test_nrmse",
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_mechanistic_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mechanistic recovery for comparable models only."""

    compatible = frame.loc[
        frame["mechanistic_comparability"]
        == "compatible"
    ].copy()

    if compatible["model_id"].nunique() != 8:
        raise ValueError(
            "Expected 8 feature-space-compatible models, "
            f"found {compatible['model_id'].nunique()}."
        )

    if len(compatible) != 32:
        raise ValueError(
            "Expected 32 compatible model/training rows, "
            f"found {len(compatible)}."
        )

    rows = []

    for training_size in TRAINING_SIZES:
        group = compatible.loc[
            compatible["training_trajectories"]
            == training_size
        ].copy()

        row = {
            "training_trajectories": training_size,
            "n_models": group["model_id"].nunique(),
        }

        for metric in MECHANISTIC_METRICS:
            row.update(
                summarize_numeric(
                    group[metric],
                    metric,
                )
            )

        row.update(
            summarize_numeric(
                group[
                    "exact_algebraic_recovery_fraction"
                ],
                "exact_algebraic_recovery_fraction",
            )
        )

        row.update(
            summarize_numeric(
                group[
                    "numerical_equivalence_fraction"
                ],
                "numerical_equivalence_fraction",
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def safe_relative_change(
    start: float,
    end: float,
) -> float:
    """Calculate relative change when the baseline is finite and nonzero."""

    if not np.isfinite(start):
        return np.nan

    if not np.isfinite(end):
        return np.nan

    if start == 0:
        return np.nan

    return (end - start) / abs(start)


def build_model_level_changes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare train_04 and train_32 within each model."""

    rows = []

    for model_id, model_frame in frame.groupby(
        "model_id",
        sort=True,
    ):
        start_rows = model_frame.loc[
            model_frame["training_trajectories"] == 4
        ]
        end_rows = model_frame.loc[
            model_frame["training_trajectories"] == 32
        ]

        if len(start_rows) != 1 or len(end_rows) != 1:
            raise ValueError(
                f"{model_id} does not contain exactly one "
                "train_04 and one train_32 row."
            )

        start = start_rows.iloc[0]
        end = end_rows.iloc[0]

        start_validation = float(
            start["validation_macro_nrmse"]
        )
        end_validation = float(
            end["validation_macro_nrmse"]
        )

        start_test = float(
            start["test_macro_nrmse"]
        )
        end_test = float(
            end["test_macro_nrmse"]
        )

        row = {
            "model_id": model_id,
            "n_states": int(start["n_states"]),
            "mechanistic_comparability":
                start["mechanistic_comparability"],

            "validation_nrmse_train04":
                start_validation,
            "validation_nrmse_train32":
                end_validation,
            "validation_absolute_change":
                end_validation - start_validation
                if (
                    np.isfinite(start_validation)
                    and np.isfinite(end_validation)
                )
                else np.nan,
            "validation_relative_change":
                safe_relative_change(
                    start_validation,
                    end_validation,
                ),
            "validation_improved_04_to_32":
                bool(
                    np.isfinite(start_validation)
                    and np.isfinite(end_validation)
                    and end_validation < start_validation
                ),

            "test_nrmse_train04":
                start_test,
            "test_nrmse_train32":
                end_test,
            "test_finite_train04":
                bool(np.isfinite(start_test)),
            "test_finite_train32":
                bool(np.isfinite(end_test)),
            "test_absolute_change":
                end_test - start_test
                if (
                    np.isfinite(start_test)
                    and np.isfinite(end_test)
                )
                else np.nan,
            "test_relative_change":
                safe_relative_change(
                    start_test,
                    end_test,
                ),
            "test_improved_04_to_32":
                bool(
                    np.isfinite(start_test)
                    and np.isfinite(end_test)
                    and end_test < start_test
                ),

            "dependency_f1_train04":
                start["macro_dependency_f1"],
            "dependency_f1_train32":
                end["macro_dependency_f1"],
            "dependency_f1_change":
                end["macro_dependency_f1"]
                - start["macro_dependency_f1"],

            "monomial_f1_train04":
                start[
                    "macro_numerator_monomial_f1"
                ],
            "monomial_f1_train32":
                end[
                    "macro_numerator_monomial_f1"
                ],
            "monomial_f1_change":
                end[
                    "macro_numerator_monomial_f1"
                ]
                - start[
                    "macro_numerator_monomial_f1"
                ],

            "interaction_f1_train04":
                start["macro_interaction_f1"],
            "interaction_f1_train32":
                end["macro_interaction_f1"],
            "interaction_f1_change":
                end["macro_interaction_f1"]
                - start["macro_interaction_f1"],

            "rational_accuracy_train04":
                start["rational_structure_accuracy"],
            "rational_accuracy_train32":
                end["rational_structure_accuracy"],
            "rational_accuracy_change":
                end["rational_structure_accuracy"]
                - start["rational_structure_accuracy"],

            "denominator_f1_train04":
                start[
                    "macro_denominator_dependency_f1"
                ],
            "denominator_f1_train32":
                end[
                    "macro_denominator_dependency_f1"
                ],
            "denominator_f1_change":
                end[
                    "macro_denominator_dependency_f1"
                ]
                - start[
                    "macro_denominator_dependency_f1"
                ],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def format_number(
    value: float,
) -> str:
    """Format a numeric result for the text report."""

    if pd.isna(value):
        return "NA"

    if np.isposinf(value):
        return "inf"

    if np.isneginf(value):
        return "-inf"

    return f"{value:.6g}"


def build_text_report(
    master: pd.DataFrame,
    dynamic: pd.DataFrame,
    mechanistic: pd.DataFrame,
    changes: pd.DataFrame,
) -> str:
    """Build a compact human-readable benchmark summary."""

    lines = []

    lines.append("PySR benchmark analysis")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"Master rows: {len(master)}"
    )
    lines.append(
        f"Models: {master['model_id'].nunique()}"
    )
    lines.append(
        "Feature-space-compatible models: "
        f"{master.loc[master['mechanistic_comparability'] == 'compatible', 'model_id'].nunique()}"
    )
    lines.append("")

    lines.append(
        "Dynamic performance by training size"
    )
    lines.append("-" * 70)

    for _, row in dynamic.iterrows():
        lines.append(
            "train={:>2}: "
            "validation median={} "
            "[IQR {}, {}], "
            "test success={}/{}, "
            "finite-test median={}".format(
                int(
                    row[
                        "training_trajectories"
                    ]
                ),
                format_number(
                    row[
                        "validation_nrmse_median"
                    ]
                ),
                format_number(
                    row[
                        "validation_nrmse_q25"
                    ]
                ),
                format_number(
                    row[
                        "validation_nrmse_q75"
                    ]
                ),
                int(
                    row[
                        "test_successful_systems"
                    ]
                ),
                int(row["n_models"]),
                format_number(
                    row[
                        "test_nrmse_median"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "Mechanistic recovery "
        "(feature-space-compatible models only)"
    )
    lines.append("-" * 70)

    for _, row in mechanistic.iterrows():
        lines.append(
            "train={:>2}: "
            "dependency F1 median={}, "
            "monomial F1 median={}, "
            "interaction F1 median={}, "
            "rational accuracy median={}, "
            "denominator F1 median={}".format(
                int(
                    row[
                        "training_trajectories"
                    ]
                ),
                format_number(
                    row[
                        "macro_dependency_f1_median"
                    ]
                ),
                format_number(
                    row[
                        "macro_numerator_monomial_f1_median"
                    ]
                ),
                format_number(
                    row[
                        "macro_interaction_f1_median"
                    ]
                ),
                format_number(
                    row[
                        "rational_structure_accuracy_median"
                    ]
                ),
                format_number(
                    row[
                        "macro_denominator_dependency_f1_median"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "Within-model train_04 -> train_32"
    )
    lines.append("-" * 70)

    validation_improved = int(
        changes[
            "validation_improved_04_to_32"
        ].sum()
    )

    test_finite_both = changes.loc[
        changes["test_finite_train04"]
        & changes["test_finite_train32"]
    ]

    test_improved = int(
        test_finite_both[
            "test_improved_04_to_32"
        ].sum()
    )

    lines.append(
        "Validation NRMSE improved in "
        f"{validation_improved}/{len(changes)} models."
    )

    lines.append(
        "Among models with finite held-out test "
        "at both train_04 and train_32, "
        f"test NRMSE improved in "
        f"{test_improved}/{len(test_finite_both)} models."
    )

    compatible_changes = changes.loc[
        changes["mechanistic_comparability"]
        == "compatible"
    ]

    for metric in (
        "dependency_f1_change",
        "monomial_f1_change",
        "interaction_f1_change",
        "rational_accuracy_change",
        "denominator_f1_change",
    ):
        improved = int(
            (
                compatible_changes[metric]
                > 0
            ).sum()
        )

        unchanged = int(
            np.isclose(
                compatible_changes[metric],
                0.0,
            ).sum()
        )

        worsened = int(
            (
                compatible_changes[metric]
                < 0
            ).sum()
        )

        lines.append(
            f"{metric}: "
            f"improved={improved}, "
            f"unchanged={unchanged}, "
            f"worsened={worsened}"
        )

    lines.append("")
    lines.append(
        "Important interpretation constraints"
    )
    lines.append("-" * 70)
    lines.append(
        "Mechanistic recovery summaries exclude "
        "feature_space_mismatch systems."
    )
    lines.append(
        "Finite held-out test summaries condition "
        "error statistics on successful integration; "
        "failure counts are reported separately."
    )
    lines.append(
        "SBML portability is implementation equivalence, "
        "not biological/mechanistic recovery."
    )
    lines.append(
        "Dynamic-fidelity validation and selected-system "
        "validation are related measurements and should "
        "not be treated as independent evidence."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run the reporting-only benchmark analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the frozen master PySR benchmark table."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Master benchmark CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived analysis tables.",
    )

    args = parser.parse_args()

    master = load_master_table(
        args.input
    )

    dynamic = build_dynamic_summary(
        master
    )

    mechanistic = build_mechanistic_summary(
        master
    )

    changes = build_model_level_changes(
        master
    )

    report = build_text_report(
        master,
        dynamic,
        mechanistic,
        changes,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dynamic_path = (
        args.output_dir
        / "dynamic_summary_by_training_size.csv"
    )

    mechanistic_path = (
        args.output_dir
        / "mechanistic_summary_by_training_size.csv"
    )

    changes_path = (
        args.output_dir
        / "model_level_changes_04_to_32.csv"
    )

    report_path = (
        args.output_dir
        / "benchmark_analysis_summary.txt"
    )

    dynamic.to_csv(
        dynamic_path,
        index=False,
    )

    mechanistic.to_csv(
        mechanistic_path,
        index=False,
    )

    changes.to_csv(
        changes_path,
        index=False,
    )

    report_path.write_text(
        report + "\n"
    )

    print(report)

    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(dynamic_path.resolve())
    print(mechanistic_path.resolve())
    print(changes_path.resolve())
    print(report_path.resolve())

    print()
    print(
        "No scientific computation was rerun."
    )


if __name__ == "__main__":
    main()
