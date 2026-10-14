"""Inspect algebraic structure of true and inferred ODE expressions.

This diagnostic script compares several SymPy representations of the
mechanistic ground-truth RHS equations and frozen Selector v1 PySR
equations.

It is intentionally diagnostic only. It does not define term-recovery
metrics and must not modify any frozen inference or evaluation result.
"""

import argparse

import sympy as sp

from src.evaluate_dependency_recovery import (
    TRAIN_DATASETS,
    load_selected_system,
    mechanistic_comparability,
    parse_inferred_equation,
)
from src.extract_mechanistic_ground_truth import (
    extract_mechanistic_ground_truth,
)
from src.model_registry import (
    resolve_model_path,
)


def expression_representations(
    expression,
):
    """Return useful algebraic representations of one SymPy expression."""
    return {
        "original": expression,
        "expanded": sp.expand(
            expression
        ),
        "factored": sp.factor(
            expression
        ),
        "cancelled": sp.cancel(
            expression
        ),
        "together": sp.together(
            expression
        ),
    }


def additive_terms(
    expression,
):
    """Return top-level additive terms after algebraic expansion."""
    expanded = sp.expand(
        expression
    )

    return sp.Add.make_args(
        expanded
    )


def print_expression(
    *,
    label,
    expression,
):
    """Print several structural views of one expression."""
    representations = expression_representations(
        expression
    )

    print()
    print(
        label
    )

    for representation_name, representation in (
        representations.items()
    ):
        print(
            f"  {representation_name}:"
        )

        print(
            "   ",
            representation,
        )

    print(
        "  expanded additive terms:"
    )

    for index, term in enumerate(
        additive_terms(
            expression
        ),
        start=1,
    ):
        print(
            f"    [{index}] {term}"
        )


def inspect_dataset(
    *,
    model_id,
    dataset_name,
    ground_truth,
):
    """Inspect true and inferred RHS structure for one training dataset."""
    (
        selection_path,
        selected_row,
    ) = load_selected_system(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    print()
    print("=" * 80)
    print(
        "DATASET:",
        dataset_name,
    )

    print(
        "Frozen selection:",
        selection_path,
    )

    for state_id in ground_truth.state_ids:
        equation_column = (
            f"equation_{state_id}"
        )

        if equation_column not in selected_row.index:
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
            _state_symbols,
        ) = parse_inferred_equation(
            inferred_equation_text,
            state_ids=ground_truth.state_ids,
        )

        true_expression = (
            ground_truth.rhs_by_state[
                state_id
            ]
        )

        print()
        print("-" * 80)

        print(
            "STATE:",
            state_id,
        )

        print_expression(
            label="GROUND TRUTH",
            expression=true_expression,
        )

        print_expression(
            label="INFERRED",
            expression=inferred_expression,
        )


def run_inspection(
    *,
    model_id,
):
    """Inspect term structure for one frozen model."""
    model_path = resolve_model_path(
        model_id
    )

    ground_truth = (
        extract_mechanistic_ground_truth(
            model_id=model_id,
            model_path=model_path,
        )
    )

    print()
    print("=" * 80)
    print("TERM-STRUCTURE INSPECTION")
    print("=" * 80)

    print(
        "Model:",
        model_id,
    )

    print(
        "States:",
        ground_truth.state_ids,
    )

    print(
        "Ground truth autonomous:",
        ground_truth.is_autonomous,
    )

    print(
        "Mechanistic comparability:",
        mechanistic_comparability(
            ground_truth
        ),
    )

    for dataset_name in TRAIN_DATASETS:
        inspect_dataset(
            model_id=model_id,
            dataset_name=dataset_name,
            ground_truth=ground_truth,
        )


def main():
    """Run algebraic-structure inspection for one frozen model."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect algebraic structure of ground-truth "
            "and frozen PySR equations."
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

    run_inspection(
        model_id=arguments.model_id,
    )


if __name__ == "__main__":
    main()