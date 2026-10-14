"""Build the master summary for the frozen PySR benchmark cohort.

Purpose
-------
Aggregate already-computed scientific results into one row per
(model_id, training-set size).

The main benchmark contains:

    10 models x 4 training sizes = 40 selected systems

This script is reporting-only. It does not:

- simulate SBML models,
- run PySR,
- perform candidate selection,
- refit equations,
- integrate validation or held-out test trajectories,
- recompute mechanistic-recovery metrics,
- recompute SBML portability,
- recompute SBML dynamic fidelity.

All scientific quantities are read from frozen result artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model_registry import resolve_model_path
from src.model_spec import load_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

MODEL_IDS = (
    "BIOMD0000000079",
    "BIOMD0000000156",
    "BIOMD0000000322",
    "BIOMD0000000174",
    "BIOMD0000000039",
    "BIOMD0000000116",
    "BIOMD0000000229",
    "BIOMD0000000010",
    "BIOMD0000000022",
    "BIOMD0000000476",
)

DATASET_NAMES = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

TRAINING_TRAJECTORIES = {
    "train_04": 4,
    "train_08": 8,
    "train_16": 16,
    "train_32": 32,
}

OUTPUT_COLUMNS = (
    # Identity.
    "model_id",
    "n_states",
    "dataset_name",
    "training_trajectories",

    # Selected-system validation.
    "validation_macro_nrmse",

    # Held-out test.
    "test_macro_nrmse",
    "test_status",
    "test_successful_trajectories",
    "test_failed_trajectories",

    # Dependency recovery.
    "mechanistic_comparability",
    "macro_dependency_precision",
    "macro_dependency_recall",
    "macro_dependency_f1",
    "micro_dependency_precision",
    "micro_dependency_recall",
    "micro_dependency_f1",

    # Functional-form recovery.
    "macro_numerator_monomial_f1",
    "macro_interaction_f1",
    "rational_structure_accuracy",
    "macro_denominator_dependency_f1",
    "mean_absolute_numerator_degree_error",
    "mean_absolute_denominator_degree_error",

    # Algebraic / numerical recovery.
    "n_exact_algebraic_matches",
    "exact_algebraic_recovery_fraction",
    "all_rhs_exact_algebraic_match",
    "n_numerically_evaluated_rhs",
    "n_numerically_equivalent_rhs",
    "numerical_equivalence_fraction",
    "all_evaluable_rhs_numerically_equivalent",

    # Original SBML vs inferred SBML.
    "dynamic_fidelity_macro_nrmse",
    "dynamic_fidelity_mean_state_r2",
    "dynamic_fidelity_status",

    # solve_ivp inferred ODE vs RoadRunner inferred SBML.
    "portability_status",
    "portability_solve_ivp_successes",
    "portability_roadrunner_successes",
    "portability_comparable_trajectories",
    "portability_equivalent_trajectories",
    "portability_nonequivalent_trajectories",
    "portability_not_evaluable_trajectories",
    "portability_max_absolute_error",
    "portability_max_relative_error",
)


def read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required frozen CSV artifact."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required frozen result is missing: {path}"
        )

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Required frozen result is empty: {path}"
        ) from exc


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Read an optional frozen CSV artifact when present."""

    if not path.is_file():
        return None

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def select_dataset_row(
    frame: pd.DataFrame,
    dataset_name: str,
    source_path: Path,
) -> pd.Series:
    """Select exactly one row for a training dataset."""

    if "dataset_name" not in frame.columns:
        raise ValueError(
            f"{source_path} does not contain dataset_name."
        )

    matches = frame.loc[
        frame["dataset_name"] == dataset_name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for {dataset_name} in "
            f"{source_path}, found {len(matches)}."
        )

    return matches.iloc[0]


def require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source_path: Path,
) -> None:
    """Require a frozen CSV to contain the expected columns."""

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{source_path} is missing required columns: "
            f"{', '.join(missing)}"
        )


def first_present(
    row: pd.Series | None,
    names: tuple[str, ...],
    default=pd.NA,
):
    """Return the first available non-null field from a row."""

    if row is None:
        return default

    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]

    return default


def load_model_state_count(model_id: str) -> int:
    """Load the canonical number of inferred dynamic states."""

    model_path = resolve_model_path(model_id)
    spec = load_model_spec(model_id, model_path)

    return len(spec.state_ids)


def load_validation_row(
    experiment_root: Path,
    dataset_name: str,
) -> pd.Series | None:
    """Load frozen selected-system validation information."""

    candidates = (
        experiment_root
        / dataset_name
        / "coordinate_selection"
        / "coordinate_selected_system.csv",
        experiment_root
        / dataset_name
        / "coordinate_selection"
        / "selection_summary.csv",
    )

    for path in candidates:
        frame = read_optional_csv(path)

        if frame is None or frame.empty:
            continue

        if len(frame) == 1:
            return frame.iloc[0]

    return None


def load_test_row(
    experiment_root: Path,
    dataset_name: str,
) -> pd.Series | None:
    """Load the frozen held-out test summary."""

    path = (
        experiment_root
        / dataset_name
        / "coordinate_selection"
        / "test"
        / "test_summary.csv"
    )

    frame = read_optional_csv(path)

    if frame is None or frame.empty:
        return None

    if len(frame) != 1:
        raise ValueError(
            f"Expected one held-out test summary row in "
            f"{path}, found {len(frame)}."
        )

    return frame.iloc[0]


def validate_mechanistic_tables(
    dependency: pd.DataFrame,
    dependency_path: Path,
    functional: pd.DataFrame,
    functional_path: Path,
    algebraic: pd.DataFrame,
    algebraic_path: Path,
) -> None:
    """Validate contracts for frozen mechanistic summaries."""

    require_columns(
        dependency,
        (
            "model_id",
            "dataset_name",
            "n_states",
            "mechanistic_comparability",
            "macro_dependency_precision",
            "macro_dependency_recall",
            "macro_dependency_f1",
            "micro_dependency_precision",
            "micro_dependency_recall",
            "micro_dependency_f1",
        ),
        dependency_path,
    )

    require_columns(
        functional,
        (
            "model_id",
            "dataset_name",
            "n_states",
            "mechanistic_comparability",
            "macro_numerator_monomial_f1",
            "macro_interaction_f1",
            "rational_structure_accuracy",
            "macro_denominator_dependency_f1",
            "mean_absolute_numerator_degree_error",
            "mean_absolute_denominator_degree_error",
        ),
        functional_path,
    )

    require_columns(
        algebraic,
        (
            "model_id",
            "dataset_name",
            "n_states",
            "mechanistic_comparability",
            "n_exact_algebraic_matches",
            "exact_algebraic_recovery_fraction",
            "all_rhs_exact_algebraic_match",
            "n_numerically_evaluated_rhs",
            "n_numerically_equivalent_rhs",
            "numerical_equivalence_fraction",
            "all_evaluable_rhs_numerically_equivalent",
        ),
        algebraic_path,
    )


def build_model_rows(model_id: str) -> list[dict]:
    """Build the four benchmark rows for one model."""

    experiment_root = (
        RESULTS_ROOT
        / model_id
        / "generic_exact_experiment"
    )

    sbml_root = (
        experiment_root
        / "coordinate_selection"
        / "sbml_validation"
    )

    mechanistic_root = (
        experiment_root
        / "mechanistic_recovery"
    )

    portability_path = (
        sbml_root
        / "portability_comparison.csv"
    )

    fidelity_path = (
        sbml_root
        / "dynamic_fidelity"
        / "dynamic_fidelity_comparison.csv"
    )

    dependency_path = (
        mechanistic_root
        / "dependency_recovery_summary.csv"
    )

    functional_path = (
        mechanistic_root
        / "functional_form_recovery_summary.csv"
    )

    algebraic_path = (
        mechanistic_root
        / "algebraic_recovery_summary.csv"
    )

    portability = read_required_csv(
        portability_path
    )
    fidelity = read_required_csv(
        fidelity_path
    )
    dependency = read_required_csv(
        dependency_path
    )
    functional = read_required_csv(
        functional_path
    )
    algebraic = read_required_csv(
        algebraic_path
    )

    validate_mechanistic_tables(
        dependency,
        dependency_path,
        functional,
        functional_path,
        algebraic,
        algebraic_path,
    )

    n_states = load_model_state_count(
        model_id
    )

    rows = []

    for dataset_name in DATASET_NAMES:
        portability_row = select_dataset_row(
            portability,
            dataset_name,
            portability_path,
        )

        fidelity_row = select_dataset_row(
            fidelity,
            dataset_name,
            fidelity_path,
        )

        dependency_row = select_dataset_row(
            dependency,
            dataset_name,
            dependency_path,
        )

        functional_row = select_dataset_row(
            functional,
            dataset_name,
            functional_path,
        )

        algebraic_row = select_dataset_row(
            algebraic,
            dataset_name,
            algebraic_path,
        )

        validation_row = load_validation_row(
            experiment_root,
            dataset_name,
        )

        test_row = load_test_row(
            experiment_root,
            dataset_name,
        )

        comparability_values = {
            str(
                dependency_row[
                    "mechanistic_comparability"
                ]
            ),
            str(
                functional_row[
                    "mechanistic_comparability"
                ]
            ),
            str(
                algebraic_row[
                    "mechanistic_comparability"
                ]
            ),
        }

        if len(comparability_values) != 1:
            raise ValueError(
                f"Inconsistent mechanistic comparability for "
                f"{model_id} {dataset_name}: "
                f"{sorted(comparability_values)}"
            )

        reported_state_counts = {
            int(dependency_row["n_states"]),
            int(functional_row["n_states"]),
            int(algebraic_row["n_states"]),
        }

        if reported_state_counts != {n_states}:
            raise ValueError(
                f"State-count mismatch for "
                f"{model_id} {dataset_name}: "
                f"canonical={n_states}, "
                f"mechanistic={sorted(reported_state_counts)}"
            )

        row = {
            "model_id": model_id,
            "n_states": n_states,
            "dataset_name": dataset_name,
            "training_trajectories":
                TRAINING_TRAJECTORIES[dataset_name],

            "validation_macro_nrmse": first_present(
                validation_row,
                (
                    "validation_macro_nrmse",
                    "macro_nrmse",
                    "validation_nrmse",
                ),
            ),

            "test_macro_nrmse": first_present(
                test_row,
                (
                    "test_macro_nrmse",
                    "macro_nrmse",
                    "mean_macro_nrmse",
                ),
            ),
            "test_status": first_present(
                test_row,
                (
                    "test_status",
                    "status",
                    "integration_status",
                ),
            ),
            "test_successful_trajectories":
                first_present(
                    test_row,
                    (
                        "successful_trajectories",
                        "test_successful_trajectories",
                    ),
                ),
            "test_failed_trajectories":
                first_present(
                    test_row,
                    (
                        "failed_trajectories",
                        "test_failed_trajectories",
                    ),
                ),

            "mechanistic_comparability":
                dependency_row[
                    "mechanistic_comparability"
                ],

            "macro_dependency_precision":
                dependency_row[
                    "macro_dependency_precision"
                ],
            "macro_dependency_recall":
                dependency_row[
                    "macro_dependency_recall"
                ],
            "macro_dependency_f1":
                dependency_row[
                    "macro_dependency_f1"
                ],
            "micro_dependency_precision":
                dependency_row[
                    "micro_dependency_precision"
                ],
            "micro_dependency_recall":
                dependency_row[
                    "micro_dependency_recall"
                ],
            "micro_dependency_f1":
                dependency_row[
                    "micro_dependency_f1"
                ],

            "macro_numerator_monomial_f1":
                functional_row[
                    "macro_numerator_monomial_f1"
                ],
            "macro_interaction_f1":
                functional_row[
                    "macro_interaction_f1"
                ],
            "rational_structure_accuracy":
                functional_row[
                    "rational_structure_accuracy"
                ],
            "macro_denominator_dependency_f1":
                functional_row[
                    "macro_denominator_dependency_f1"
                ],
            "mean_absolute_numerator_degree_error":
                functional_row[
                    "mean_absolute_numerator_degree_error"
                ],
            "mean_absolute_denominator_degree_error":
                functional_row[
                    "mean_absolute_denominator_degree_error"
                ],

            "n_exact_algebraic_matches":
                algebraic_row[
                    "n_exact_algebraic_matches"
                ],
            "exact_algebraic_recovery_fraction":
                algebraic_row[
                    "exact_algebraic_recovery_fraction"
                ],
            "all_rhs_exact_algebraic_match":
                algebraic_row[
                    "all_rhs_exact_algebraic_match"
                ],
            "n_numerically_evaluated_rhs":
                algebraic_row[
                    "n_numerically_evaluated_rhs"
                ],
            "n_numerically_equivalent_rhs":
                algebraic_row[
                    "n_numerically_equivalent_rhs"
                ],
            "numerical_equivalence_fraction":
                algebraic_row[
                    "numerical_equivalence_fraction"
                ],
            "all_evaluable_rhs_numerically_equivalent":
                algebraic_row[
                    "all_evaluable_rhs_numerically_equivalent"
                ],

            "dynamic_fidelity_macro_nrmse":
                fidelity_row[
                    "validation_macro_nrmse"
                ],
            "dynamic_fidelity_mean_state_r2":
                fidelity_row[
                    "mean_state_r2"
                ],
            "dynamic_fidelity_status":
                fidelity_row[
                    "dynamic_fidelity_status"
                ],

            "portability_status":
                portability_row[
                    "portability_status"
                ],
            "portability_solve_ivp_successes":
                portability_row[
                    "solve_ivp_successes"
                ],
            "portability_roadrunner_successes":
                portability_row[
                    "roadrunner_successes"
                ],
            "portability_comparable_trajectories":
                portability_row[
                    "comparable_trajectories"
                ],
            "portability_equivalent_trajectories":
                portability_row[
                    "equivalent_trajectories"
                ],
            "portability_nonequivalent_trajectories":
                portability_row[
                    "nonequivalent_trajectories"
                ],
            "portability_not_evaluable_trajectories":
                portability_row[
                    "not_evaluable_trajectories"
                ],
            "portability_max_absolute_error":
                portability_row[
                    "max_absolute_error"
                ],
            "portability_max_relative_error":
                portability_row[
                    "max_relative_error"
                ],
        }

        rows.append(row)

    return rows


def validate_summary(
    summary: pd.DataFrame,
) -> None:
    """Validate the complete benchmark-table contract."""

    expected_rows = (
        len(MODEL_IDS)
        * len(DATASET_NAMES)
    )

    if len(summary) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(summary)}."
        )

    duplicated = summary.duplicated(
        subset=[
            "model_id",
            "dataset_name",
        ]
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
            "Duplicate model/dataset rows found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    observed_models = set(
        summary["model_id"]
    )
    expected_models = set(
        MODEL_IDS
    )

    if observed_models != expected_models:
        raise ValueError(
            "Model cohort mismatch. "
            f"Expected {sorted(expected_models)}, "
            f"found {sorted(observed_models)}."
        )

    for model_id in MODEL_IDS:
        model_rows = summary.loc[
            summary["model_id"] == model_id
        ]

        observed_datasets = set(
            model_rows["dataset_name"]
        )

        if observed_datasets != set(DATASET_NAMES):
            raise ValueError(
                f"{model_id} does not contain "
                f"exactly {DATASET_NAMES}."
            )

        if len(model_rows) != 4:
            raise ValueError(
                f"{model_id} should have 4 rows, "
                f"found {len(model_rows)}."
            )


def build_summary() -> pd.DataFrame:
    """Build the complete 40-row benchmark summary."""

    rows = []

    for model_id in MODEL_IDS:
        print(f"Reading {model_id}...")

        rows.extend(
            build_model_rows(model_id)
        )

    summary = pd.DataFrame(rows)
    summary = summary.loc[
        :,
        OUTPUT_COLUMNS,
    ]

    validate_summary(
        summary
    )

    return summary


def main() -> None:
    """Build and save the master benchmark summary."""

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate frozen scientific results for "
            "the 10-model PySR benchmark cohort."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            RESULTS_ROOT
            / "benchmark_summary.csv"
        ),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    summary = build_summary()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.output,
        index=False,
    )

    portability_passes = (
        summary["portability_status"]
        .eq("pass")
        .sum()
    )

    comparable_rows = (
        summary[
            "mechanistic_comparability"
        ]
        .ne("feature_space_mismatch")
        .sum()
    )

    mismatched_rows = (
        summary[
            "mechanistic_comparability"
        ]
        .eq("feature_space_mismatch")
        .sum()
    )

    exact_systems = (
        summary[
            "all_rhs_exact_algebraic_match"
        ]
        .eq(True)
        .sum()
    )

    numerical_systems = (
        summary[
            "all_evaluable_rhs_numerically_equivalent"
        ]
        .eq(True)
        .sum()
    )

    print()
    print("=" * 70)
    print("MASTER BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"Models: "
        f"{summary['model_id'].nunique()}"
    )
    print(
        f"Rows: {len(summary)}"
    )
    print(
        "Training sizes: "
        + ", ".join(
            str(value)
            for value in sorted(
                summary[
                    "training_trajectories"
                ].unique()
            )
        )
    )
    print(
        "Mechanistically comparable rows: "
        f"{comparable_rows}/{len(summary)}"
    )
    print(
        "Feature-space mismatch rows: "
        f"{mismatched_rows}/{len(summary)}"
    )
    print(
        "Exact full-system algebraic matches: "
        f"{exact_systems}/{len(summary)}"
    )
    print(
        "Full-system numerical equivalence: "
        f"{numerical_systems}/{len(summary)}"
    )
    print(
        "Strict SBML portability passes: "
        f"{portability_passes}/{len(summary)}"
    )
    print(
        f"Columns: {len(summary.columns)}"
    )
    print(
        f"Output: {args.output.resolve()}"
    )

    print()
    print(
        "Frozen scientific results were read "
        "but not modified."
    )
    print(
        "No simulation, PySR inference, candidate "
        "selection, refitting, or trajectory "
        "integration was performed."
    )


if __name__ == "__main__":
    main()