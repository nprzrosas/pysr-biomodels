"""Evaluate state-dependency recovery of frozen PySR systems.

This module compares the state-variable dependencies present in the
mechanistic SBML ground-truth ODEs with those present in the equations
selected by frozen Selector v1.

The analysis is strictly post-hoc. It must not influence PySR fitting,
candidate generation, validation selection, Selector v1, or held-out
test evaluation.

For non-autonomous ground-truth models, explicit independent variables
such as time are recorded separately. Dependency precision, recall, and
F1 evaluate only dependencies among dynamic state variables because the
frozen PySR inference feature space contained states only.
"""

import argparse
from pathlib import Path

import pandas as pd
import sympy as sp

from src.extract_mechanistic_ground_truth import (
    extract_mechanistic_ground_truth,
)
from src.model_registry import (
    resolve_model_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
)

EXPERIMENT_NAME = (
    "generic_exact_experiment"
)

SELECTION_DIRECTORY = (
    "coordinate_selection"
)

SELECTED_SYSTEM_FILENAME = (
    "coordinate_selected_system.csv"
)

OUTPUT_DIRECTORY = (
    "mechanistic_recovery"
)

TRAIN_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

INFERENCE_FEATURE_SPACE = (
    "states_only"
)


def parse_inferred_equation(
    equation_text,
    *,
    state_ids,
):
    """Parse one frozen PySR equation using explicit state symbols."""
    state_symbols = {
        state_id: sp.Symbol(
            state_id
        )
        for state_id in state_ids
    }

    expression = sp.sympify(
        str(
            equation_text
        ),
        locals=state_symbols,
    )

    return (
        expression,
        state_symbols,
    )


def state_dependencies(
    expression,
    *,
    state_symbols,
):
    """Return dynamic-state symbols appearing in one expression."""
    dynamic_symbols = set(
        state_symbols.values()
    )

    dependencies = (
        expression.free_symbols
        & dynamic_symbols
    )

    return {
        str(
            symbol
        )
        for symbol in dependencies
    }


def dependency_metrics(
    *,
    true_dependencies,
    inferred_dependencies,
):
    """Calculate precision, recall, and F1 for state dependencies."""
    true_dependencies = set(
        true_dependencies
    )

    inferred_dependencies = set(
        inferred_dependencies
    )

    true_positive = (
        true_dependencies
        & inferred_dependencies
    )

    false_positive = (
        inferred_dependencies
        - true_dependencies
    )

    false_negative = (
        true_dependencies
        - inferred_dependencies
    )

    n_true_positive = len(
        true_positive
    )

    n_false_positive = len(
        false_positive
    )

    n_false_negative = len(
        false_negative
    )

    precision_denominator = (
        n_true_positive
        + n_false_positive
    )

    if precision_denominator > 0:
        precision = (
            n_true_positive
            / precision_denominator
        )

    else:
        precision = float(
            "nan"
        )

    recall_denominator = (
        n_true_positive
        + n_false_negative
    )

    if recall_denominator > 0:
        recall = (
            n_true_positive
            / recall_denominator
        )

    else:
        recall = float(
            "nan"
        )

    f1_denominator = (
        2
        * n_true_positive
        + n_false_positive
        + n_false_negative
    )

    if f1_denominator > 0:
        f1 = (
            2
            * n_true_positive
            / f1_denominator
        )

    else:
        f1 = 1.0

    return {
        "true_positive_dependencies": (
            true_positive
        ),
        "false_positive_dependencies": (
            false_positive
        ),
        "false_negative_dependencies": (
            false_negative
        ),
        "n_true_dependencies": len(
            true_dependencies
        ),
        "n_inferred_dependencies": len(
            inferred_dependencies
        ),
        "n_true_positive": (
            n_true_positive
        ),
        "n_false_positive": (
            n_false_positive
        ),
        "n_false_negative": (
            n_false_negative
        ),
        "dependency_precision": (
            precision
        ),
        "dependency_recall": (
            recall
        ),
        "dependency_f1": (
            f1
        ),
    }


def serialize_dependencies(
    dependencies,
):
    """Serialize a dependency set deterministically for CSV output."""
    return ",".join(
        sorted(
            dependencies
        )
    )


def selected_system_path(
    *,
    model_id,
    dataset_name,
):
    """Return the frozen Selector v1 system path."""
    return (
        RESULTS_ROOT
        / model_id
        / EXPERIMENT_NAME
        / dataset_name
        / SELECTION_DIRECTORY
        / SELECTED_SYSTEM_FILENAME
    )


def load_selected_system(
    *,
    model_id,
    dataset_name,
):
    """Load and validate one frozen Selector v1 selected system."""
    path = selected_system_path(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Frozen selected system does not exist: "
            f"{path}"
        )

    frame = pd.read_csv(
        path
    )

    if len(
        frame
    ) != 1:
        raise ValueError(
            "Expected exactly one selected-system row "
            f"in {path}, found {len(frame)}."
        )

    row = frame.iloc[
        0
    ]

    recorded_model_id = str(
        row[
            "model_id"
        ]
    )

    if recorded_model_id != model_id:
        raise ValueError(
            "Selected-system model mismatch: "
            f"expected {model_id}, "
            f"found {recorded_model_id}."
        )

    recorded_dataset = str(
        row[
            "dataset_name"
        ]
    )

    if recorded_dataset != dataset_name:
        raise ValueError(
            "Selected-system dataset mismatch: "
            f"expected {dataset_name}, "
            f"found {recorded_dataset}."
        )

    selection_method = str(
        row[
            "selection_method"
        ]
    )

    if selection_method != "coordinate_search":
        raise ValueError(
            "Expected frozen coordinate_search selection, "
            f"found {selection_method}."
        )

    return (
        path,
        row,
    )


def mechanistic_comparability(
    ground_truth,
):
    """Classify compatibility of truth with frozen inference features."""
    if ground_truth.is_autonomous:
        return "compatible"

    return "feature_space_mismatch"


def evaluate_one_rhs(
    *,
    model_id,
    dataset_name,
    state_id,
    ground_truth,
    selected_row,
):
    """Evaluate state-dependency recovery for one inferred RHS."""
    equation_column = (
        f"equation_{state_id}"
    )

    if (
        equation_column
        not in selected_row.index
    ):
        raise KeyError(
            "Selected system is missing equation column "
            f"{equation_column}."
        )

    inferred_equation_text = str(
        selected_row[
            equation_column
        ]
    )

    (
        inferred_expression,
        state_symbols,
    ) = parse_inferred_equation(
        inferred_equation_text,
        state_ids=ground_truth.state_ids,
    )

    true_expression = (
        ground_truth.rhs_by_state[
            state_id
        ]
    )

    true_dependencies = state_dependencies(
        true_expression,
        state_symbols=state_symbols,
    )

    inferred_dependencies = state_dependencies(
        inferred_expression,
        state_symbols=state_symbols,
    )

    metrics = dependency_metrics(
        true_dependencies=true_dependencies,
        inferred_dependencies=inferred_dependencies,
    )

    independent_variables = set(
        ground_truth.explicit_independent_variables
    )

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "state_id": state_id,
        "ground_truth_is_autonomous": (
            ground_truth.is_autonomous
        ),
        "ground_truth_independent_variables": (
            serialize_dependencies(
                independent_variables
            )
        ),
        "inference_feature_space": (
            INFERENCE_FEATURE_SPACE
        ),
        "mechanistic_comparability": (
            mechanistic_comparability(
                ground_truth
            )
        ),
        "true_equation": str(
            true_expression
        ),
        "inferred_equation": (
            inferred_equation_text
        ),
        "true_state_dependencies": (
            serialize_dependencies(
                true_dependencies
            )
        ),
        "inferred_state_dependencies": (
            serialize_dependencies(
                inferred_dependencies
            )
        ),
        "true_positive_dependencies": (
            serialize_dependencies(
                metrics[
                    "true_positive_dependencies"
                ]
            )
        ),
        "false_positive_dependencies": (
            serialize_dependencies(
                metrics[
                    "false_positive_dependencies"
                ]
            )
        ),
        "false_negative_dependencies": (
            serialize_dependencies(
                metrics[
                    "false_negative_dependencies"
                ]
            )
        ),
        "n_true_dependencies": (
            metrics[
                "n_true_dependencies"
            ]
        ),
        "n_inferred_dependencies": (
            metrics[
                "n_inferred_dependencies"
            ]
        ),
        "n_true_positive": (
            metrics[
                "n_true_positive"
            ]
        ),
        "n_false_positive": (
            metrics[
                "n_false_positive"
            ]
        ),
        "n_false_negative": (
            metrics[
                "n_false_negative"
            ]
        ),
        "dependency_precision": (
            metrics[
                "dependency_precision"
            ]
        ),
        "dependency_recall": (
            metrics[
                "dependency_recall"
            ]
        ),
        "dependency_f1": (
            metrics[
                "dependency_f1"
            ]
        ),
    }


def evaluate_dataset(
    *,
    model_id,
    dataset_name,
    ground_truth,
):
    """Evaluate all state equations for one frozen selected system."""
    (
        selection_path,
        selected_row,
    ) = load_selected_system(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    rows = []

    for state_id in ground_truth.state_ids:
        row = evaluate_one_rhs(
            model_id=model_id,
            dataset_name=dataset_name,
            state_id=state_id,
            ground_truth=ground_truth,
            selected_row=selected_row,
        )

        rows.append(
            row
        )

    frame = pd.DataFrame(
        rows
    )

    return (
        selection_path,
        frame,
    )


def summarize_dependency_recovery(
    frame,
):
    """Summarize dependency recovery by training dataset."""
    rows = []

    for dataset_name, dataset_frame in frame.groupby(
        "dataset_name",
        sort=False,
    ):
        total_true_positive = int(
            dataset_frame[
                "n_true_positive"
            ].sum()
        )

        total_false_positive = int(
            dataset_frame[
                "n_false_positive"
            ].sum()
        )

        total_false_negative = int(
            dataset_frame[
                "n_false_negative"
            ].sum()
        )

        micro_precision_denominator = (
            total_true_positive
            + total_false_positive
        )

        if micro_precision_denominator > 0:
            micro_precision = (
                total_true_positive
                / micro_precision_denominator
            )

        else:
            micro_precision = float(
                "nan"
            )

        micro_recall_denominator = (
            total_true_positive
            + total_false_negative
        )

        if micro_recall_denominator > 0:
            micro_recall = (
                total_true_positive
                / micro_recall_denominator
            )

        else:
            micro_recall = float(
                "nan"
            )

        micro_f1_denominator = (
            2
            * total_true_positive
            + total_false_positive
            + total_false_negative
        )

        if micro_f1_denominator > 0:
            micro_f1 = (
                2
                * total_true_positive
                / micro_f1_denominator
            )

        else:
            micro_f1 = 1.0

        rows.append(
            {
                "model_id": (
                    dataset_frame[
                        "model_id"
                    ].iloc[0]
                ),
                "dataset_name": (
                    dataset_name
                ),
                "n_states": len(
                    dataset_frame
                ),
                "ground_truth_is_autonomous": (
                    dataset_frame[
                        "ground_truth_is_autonomous"
                    ].iloc[0]
                ),
                "ground_truth_independent_variables": (
                    dataset_frame[
                        "ground_truth_independent_variables"
                    ].iloc[0]
                ),
                "inference_feature_space": (
                    dataset_frame[
                        "inference_feature_space"
                    ].iloc[0]
                ),
                "mechanistic_comparability": (
                    dataset_frame[
                        "mechanistic_comparability"
                    ].iloc[0]
                ),
                "total_true_positive": (
                    total_true_positive
                ),
                "total_false_positive": (
                    total_false_positive
                ),
                "total_false_negative": (
                    total_false_negative
                ),
                "macro_dependency_precision": (
                    dataset_frame[
                        "dependency_precision"
                    ].mean()
                ),
                "macro_dependency_recall": (
                    dataset_frame[
                        "dependency_recall"
                    ].mean()
                ),
                "macro_dependency_f1": (
                    dataset_frame[
                        "dependency_f1"
                    ].mean()
                ),
                "micro_dependency_precision": (
                    micro_precision
                ),
                "micro_dependency_recall": (
                    micro_recall
                ),
                "micro_dependency_f1": (
                    micro_f1
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_model(
    *,
    model_id,
):
    """Evaluate dependency recovery across all frozen training sizes."""
    model_path = resolve_model_path(
        model_id
    )

    ground_truth = (
        extract_mechanistic_ground_truth(
            model_id=model_id,
            model_path=model_path,
        )
    )

    frames = []

    print()
    print("=" * 80)
    print("DEPENDENCY RECOVERY")
    print("Model:", model_id)

    print(
        "States:",
        len(
            ground_truth.state_ids
        ),
    )

    print(
        "Ground truth autonomous:",
        ground_truth.is_autonomous,
    )

    print(
        "Ground truth independent variables:",
        ground_truth.explicit_independent_variables,
    )

    print(
        "Inference feature space:",
        INFERENCE_FEATURE_SPACE,
    )

    print(
        "Mechanistic comparability:",
        mechanistic_comparability(
            ground_truth
        ),
    )

    for dataset_name in TRAIN_DATASETS:
        (
            selection_path,
            dataset_frame,
        ) = evaluate_dataset(
            model_id=model_id,
            dataset_name=dataset_name,
            ground_truth=ground_truth,
        )

        frames.append(
            dataset_frame
        )

        print()
        print(
            "Dataset:",
            dataset_name,
        )

        print(
            "Frozen selection:",
            selection_path,
        )

        print(
            "Mean dependency precision:",
            f"{dataset_frame['dependency_precision'].mean():.6f}",
        )

        print(
            "Mean dependency recall:",
            f"{dataset_frame['dependency_recall'].mean():.6f}",
        )

        print(
            "Mean dependency F1:",
            f"{dataset_frame['dependency_f1'].mean():.6f}",
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    output_directory = (
        RESULTS_ROOT
        / model_id
        / EXPERIMENT_NAME
        / OUTPUT_DIRECTORY
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "dependency_recovery.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    summary = summarize_dependency_recovery(
        result
    )

    summary_path = (
        output_directory
        / "dependency_recovery_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "Detailed output:",
        output_path,
    )

    print(
        "Summary output:",
        summary_path,
    )

    print()
    print("=" * 80)
    print("DEPENDENCY RECOVERY SUMMARY")

    print(
        summary[
            [
                "dataset_name",
                "total_true_positive",
                "total_false_positive",
                "total_false_negative",
                "macro_dependency_f1",
                "micro_dependency_precision",
                "micro_dependency_recall",
                "micro_dependency_f1",
            ]
        ].to_string(
            index=False
        )
    )

    return (
        ground_truth,
        result,
        summary,
        output_path,
        summary_path,
    )


def print_detailed_results(
    frame,
):
    """Print compact per-RHS dependency-recovery results."""
    columns = [
        "dataset_name",
        "state_id",
        "true_state_dependencies",
        "inferred_state_dependencies",
        "false_positive_dependencies",
        "false_negative_dependencies",
        "dependency_precision",
        "dependency_recall",
        "dependency_f1",
    ]

    print()
    print("=" * 80)
    print("PER-RHS RESULTS")

    print(
        frame[
            columns
        ].to_string(
            index=False
        )
    )


def main():
    """Run dependency recovery for one frozen model."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate state-dependency recovery "
            "for one frozen Selector v1 model."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000079."
        ),
    )

    arguments = parser.parse_args()

    (
        _ground_truth,
        frame,
        _summary,
        _output_path,
        _summary_path,
    ) = evaluate_model(
        model_id=arguments.model_id,
    )

    print_detailed_results(
        frame
    )


if __name__ == "__main__":
    main()