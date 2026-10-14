"""Evaluate functional-form recovery of frozen PySR systems.

This module compares structural properties of mechanistic SBML
ground-truth ODEs with those present in equations selected by frozen
Selector v1.

The analysis is strictly post-hoc. It must not influence PySR fitting,
candidate generation, validation selection, Selector v1, or held-out
test evaluation.

Functional-form recovery is evaluated using separate interpretable
components rather than a single composite mechanistic score:

1. Numerator monomial recovery.
2. Multiplicative interaction recovery.
3. Rational state-dependence agreement.
4. Denominator state-dependency recovery.
5. Numerator and denominator polynomial-degree differences.

Constant monomials are recorded but excluded from monomial-recovery
precision, recall, and F1.

Functional-form recovery v1 was frozen after diagnostic validation on
BIOMD0000000079. Subsequent model-level evaluations must use these
definitions unchanged.
"""

import argparse

import pandas as pd

from src.evaluate_dependency_recovery import (
    EXPERIMENT_NAME,
    INFERENCE_FEATURE_SPACE,
    RESULTS_ROOT,
    TRAIN_DATASETS,
    load_selected_system,
    mechanistic_comparability,
    parse_inferred_equation,
)
from src.extract_mechanistic_ground_truth import (
    extract_mechanistic_ground_truth,
)
from src.inspect_functional_structure import (
    functional_fingerprint,
)
from src.model_registry import (
    resolve_model_path,
)


OUTPUT_DIRECTORY = (
    "mechanistic_recovery"
)

FUNCTIONAL_FORM_RECOVERY_VERSION = (
    "v1"
)


def serialize_items(
    values,
):
    """Serialize structural items deterministically for CSV output."""
    return ",".join(
        sorted(
            str(value)
            for value in values
        )
    )


def serialize_pairs(
    pairs,
):
    """Serialize state-interaction pairs deterministically."""
    return ",".join(
        sorted(
            f"{first}*{second}"
            for first, second in pairs
        )
    )


def set_recovery_metrics(
    *,
    true_items,
    inferred_items,
):
    """Calculate set-based precision, recall, and F1."""
    true_items = set(
        true_items
    )

    inferred_items = set(
        inferred_items
    )

    true_positive = (
        true_items
        & inferred_items
    )

    false_positive = (
        inferred_items
        - true_items
    )

    false_negative = (
        true_items
        - inferred_items
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
        "true_positive": (
            true_positive
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "n_true": len(
            true_items
        ),
        "n_inferred": len(
            inferred_items
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
        "precision": (
            precision
        ),
        "recall": (
            recall
        ),
        "f1": (
            f1
        ),
    }


def mechanistic_monomials(
    signatures,
):
    """Remove constant-only signatures from monomial recovery."""
    return {
        signature
        for signature in signatures
        if signature != "constant"
    }


def degree_difference(
    *,
    true_degree,
    inferred_degree,
):
    """Return inferred-minus-true polynomial degree."""
    if (
        true_degree is None
        or inferred_degree is None
    ):
        return float(
            "nan"
        )

    return (
        inferred_degree
        - true_degree
    )


def evaluate_one_rhs(
    *,
    model_id,
    dataset_name,
    state_id,
    ground_truth,
    selected_row,
):
    """Evaluate functional-form recovery for one inferred RHS."""
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

    true_fingerprint = functional_fingerprint(
        true_expression,
        state_symbols=state_symbols,
    )

    inferred_fingerprint = functional_fingerprint(
        inferred_expression,
        state_symbols=state_symbols,
    )

    true_monomials = mechanistic_monomials(
        true_fingerprint[
            "numerator_signatures"
        ]
    )

    inferred_monomials = mechanistic_monomials(
        inferred_fingerprint[
            "numerator_signatures"
        ]
    )

    monomial_metrics = set_recovery_metrics(
        true_items=true_monomials,
        inferred_items=inferred_monomials,
    )

    true_interactions = set(
        true_fingerprint[
            "numerator_interactions"
        ]
    )

    inferred_interactions = set(
        inferred_fingerprint[
            "numerator_interactions"
        ]
    )

    interaction_metrics = set_recovery_metrics(
        true_items=true_interactions,
        inferred_items=inferred_interactions,
    )

    true_denominator_dependencies = set(
        true_fingerprint[
            "denominator_dependencies"
        ]
    )

    inferred_denominator_dependencies = set(
        inferred_fingerprint[
            "denominator_dependencies"
        ]
    )

    denominator_metrics = set_recovery_metrics(
        true_items=true_denominator_dependencies,
        inferred_items=inferred_denominator_dependencies,
    )

    true_rational = bool(
        true_fingerprint[
            "has_rational_state_dependence"
        ]
    )

    inferred_rational = bool(
        inferred_fingerprint[
            "has_rational_state_dependence"
        ]
    )

    rational_structure_match = (
        true_rational
        == inferred_rational
    )

    independent_variables = set(
        ground_truth.explicit_independent_variables
    )

    return {
        "functional_form_recovery_version": (
            FUNCTIONAL_FORM_RECOVERY_VERSION
        ),
        "model_id": model_id,
        "dataset_name": dataset_name,
        "state_id": state_id,
        "ground_truth_is_autonomous": (
            ground_truth.is_autonomous
        ),
        "ground_truth_independent_variables": (
            serialize_items(
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
        "true_numerator_monomials": (
            serialize_items(
                true_monomials
            )
        ),
        "inferred_numerator_monomials": (
            serialize_items(
                inferred_monomials
            )
        ),
        "true_positive_numerator_monomials": (
            serialize_items(
                monomial_metrics[
                    "true_positive"
                ]
            )
        ),
        "false_positive_numerator_monomials": (
            serialize_items(
                monomial_metrics[
                    "false_positive"
                ]
            )
        ),
        "false_negative_numerator_monomials": (
            serialize_items(
                monomial_metrics[
                    "false_negative"
                ]
            )
        ),
        "n_true_numerator_monomials": (
            monomial_metrics[
                "n_true"
            ]
        ),
        "n_inferred_numerator_monomials": (
            monomial_metrics[
                "n_inferred"
            ]
        ),
        "n_true_positive_numerator_monomials": (
            monomial_metrics[
                "n_true_positive"
            ]
        ),
        "n_false_positive_numerator_monomials": (
            monomial_metrics[
                "n_false_positive"
            ]
        ),
        "n_false_negative_numerator_monomials": (
            monomial_metrics[
                "n_false_negative"
            ]
        ),
        "numerator_monomial_precision": (
            monomial_metrics[
                "precision"
            ]
        ),
        "numerator_monomial_recall": (
            monomial_metrics[
                "recall"
            ]
        ),
        "numerator_monomial_f1": (
            monomial_metrics[
                "f1"
            ]
        ),
        "true_interactions": (
            serialize_pairs(
                true_interactions
            )
        ),
        "inferred_interactions": (
            serialize_pairs(
                inferred_interactions
            )
        ),
        "true_positive_interactions": (
            serialize_pairs(
                interaction_metrics[
                    "true_positive"
                ]
            )
        ),
        "false_positive_interactions": (
            serialize_pairs(
                interaction_metrics[
                    "false_positive"
                ]
            )
        ),
        "false_negative_interactions": (
            serialize_pairs(
                interaction_metrics[
                    "false_negative"
                ]
            )
        ),
        "n_true_interactions": (
            interaction_metrics[
                "n_true"
            ]
        ),
        "n_inferred_interactions": (
            interaction_metrics[
                "n_inferred"
            ]
        ),
        "n_true_positive_interactions": (
            interaction_metrics[
                "n_true_positive"
            ]
        ),
        "n_false_positive_interactions": (
            interaction_metrics[
                "n_false_positive"
            ]
        ),
        "n_false_negative_interactions": (
            interaction_metrics[
                "n_false_negative"
            ]
        ),
        "interaction_precision": (
            interaction_metrics[
                "precision"
            ]
        ),
        "interaction_recall": (
            interaction_metrics[
                "recall"
            ]
        ),
        "interaction_f1": (
            interaction_metrics[
                "f1"
            ]
        ),
        "true_rational_state_dependence": (
            true_rational
        ),
        "inferred_rational_state_dependence": (
            inferred_rational
        ),
        "rational_structure_match": (
            rational_structure_match
        ),
        "true_denominator_dependencies": (
            serialize_items(
                true_denominator_dependencies
            )
        ),
        "inferred_denominator_dependencies": (
            serialize_items(
                inferred_denominator_dependencies
            )
        ),
        "true_positive_denominator_dependencies": (
            serialize_items(
                denominator_metrics[
                    "true_positive"
                ]
            )
        ),
        "false_positive_denominator_dependencies": (
            serialize_items(
                denominator_metrics[
                    "false_positive"
                ]
            )
        ),
        "false_negative_denominator_dependencies": (
            serialize_items(
                denominator_metrics[
                    "false_negative"
                ]
            )
        ),
        "n_true_denominator_dependencies": (
            denominator_metrics[
                "n_true"
            ]
        ),
        "n_inferred_denominator_dependencies": (
            denominator_metrics[
                "n_inferred"
            ]
        ),
        "n_true_positive_denominator_dependencies": (
            denominator_metrics[
                "n_true_positive"
            ]
        ),
        "n_false_positive_denominator_dependencies": (
            denominator_metrics[
                "n_false_positive"
            ]
        ),
        "n_false_negative_denominator_dependencies": (
            denominator_metrics[
                "n_false_negative"
            ]
        ),
        "denominator_dependency_precision": (
            denominator_metrics[
                "precision"
            ]
        ),
        "denominator_dependency_recall": (
            denominator_metrics[
                "recall"
            ]
        ),
        "denominator_dependency_f1": (
            denominator_metrics[
                "f1"
            ]
        ),
        "true_numerator_degree": (
            true_fingerprint[
                "numerator_degree"
            ]
        ),
        "inferred_numerator_degree": (
            inferred_fingerprint[
                "numerator_degree"
            ]
        ),
        "numerator_degree_difference": (
            degree_difference(
                true_degree=true_fingerprint[
                    "numerator_degree"
                ],
                inferred_degree=inferred_fingerprint[
                    "numerator_degree"
                ],
            )
        ),
        "true_denominator_degree": (
            true_fingerprint[
                "denominator_degree"
            ]
        ),
        "inferred_denominator_degree": (
            inferred_fingerprint[
                "denominator_degree"
            ]
        ),
        "denominator_degree_difference": (
            degree_difference(
                true_degree=true_fingerprint[
                    "denominator_degree"
                ],
                inferred_degree=inferred_fingerprint[
                    "denominator_degree"
                ],
            )
        ),
    }


def evaluate_dataset(
    *,
    model_id,
    dataset_name,
    ground_truth,
):
    """Evaluate all RHS equations for one frozen selected system."""
    (
        selection_path,
        selected_row,
    ) = load_selected_system(
        model_id=model_id,
        dataset_name=dataset_name,
    )

    rows = []

    for state_id in ground_truth.state_ids:
        rows.append(
            evaluate_one_rhs(
                model_id=model_id,
                dataset_name=dataset_name,
                state_id=state_id,
                ground_truth=ground_truth,
                selected_row=selected_row,
            )
        )

    return (
        selection_path,
        pd.DataFrame(
            rows
        ),
    )


def summarize_functional_form_recovery(
    frame,
):
    """Summarize functional-form recovery by training dataset."""
    rows = []

    for dataset_name, dataset_frame in frame.groupby(
        "dataset_name",
        sort=False,
    ):
        rows.append(
            {
                "functional_form_recovery_version": (
                    FUNCTIONAL_FORM_RECOVERY_VERSION
                ),
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
                "mechanistic_comparability": (
                    dataset_frame[
                        "mechanistic_comparability"
                    ].iloc[0]
                ),
                "macro_numerator_monomial_f1": (
                    dataset_frame[
                        "numerator_monomial_f1"
                    ].mean()
                ),
                "macro_interaction_f1": (
                    dataset_frame[
                        "interaction_f1"
                    ].mean()
                ),
                "rational_structure_accuracy": (
                    dataset_frame[
                        "rational_structure_match"
                    ].mean()
                ),
                "macro_denominator_dependency_f1": (
                    dataset_frame[
                        "denominator_dependency_f1"
                    ].mean()
                ),
                "mean_absolute_numerator_degree_error": (
                    dataset_frame[
                        "numerator_degree_difference"
                    ].abs().mean()
                ),
                "mean_absolute_denominator_degree_error": (
                    dataset_frame[
                        "denominator_degree_difference"
                    ].abs().mean()
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
    """Evaluate functional-form recovery across frozen training sizes."""
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
    print("FUNCTIONAL-FORM RECOVERY")
    print(
        "Version:",
        FUNCTIONAL_FORM_RECOVERY_VERSION,
    )
    print(
        "Model:",
        model_id,
    )

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
            "Mean numerator monomial F1:",
            f"{dataset_frame['numerator_monomial_f1'].mean():.6f}",
        )

        print(
            "Mean interaction F1:",
            f"{dataset_frame['interaction_f1'].mean():.6f}",
        )

        print(
            "Rational structure accuracy:",
            f"{dataset_frame['rational_structure_match'].mean():.6f}",
        )

        print(
            "Mean denominator dependency F1:",
            f"{dataset_frame['denominator_dependency_f1'].mean():.6f}",
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = summarize_functional_form_recovery(
        result
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
        / "functional_form_recovery.csv"
    )

    summary_path = (
        output_directory
        / "functional_form_recovery_summary.csv"
    )

    result.to_csv(
        output_path,
        index=False,
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
    print("FUNCTIONAL-FORM RECOVERY SUMMARY")

    print(
        summary.to_string(
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
    """Print compact per-RHS functional-form results."""
    columns = [
        "dataset_name",
        "state_id",
        "true_numerator_monomials",
        "inferred_numerator_monomials",
        "numerator_monomial_f1",
        "true_interactions",
        "inferred_interactions",
        "interaction_f1",
        "true_rational_state_dependence",
        "inferred_rational_state_dependence",
        "rational_structure_match",
        "true_denominator_dependencies",
        "inferred_denominator_dependencies",
        "denominator_dependency_f1",
        "true_numerator_degree",
        "inferred_numerator_degree",
        "true_denominator_degree",
        "inferred_denominator_degree",
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
    """Run functional-form recovery for one frozen model."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen functional-form recovery v1 "
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