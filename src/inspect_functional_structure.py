"""Inspect functional structure of true and inferred ODE expressions.

This diagnostic module extracts structural properties from mechanistic
SBML ground-truth equations and frozen Selector v1 PySR equations.

The purpose is to determine a suitable representation for subsequent
functional-form recovery evaluation.

No recovery score is calculated here. The definitions in this file are
diagnostic and are not yet frozen evaluation metrics.
"""

import argparse
from itertools import combinations

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


def dynamic_dependencies(
    expression,
    *,
    state_symbols,
):
    """Return dynamic states occurring in an expression."""
    dynamic_symbols = set(
        state_symbols.values()
    )

    dependencies = (
        expression.free_symbols
        & dynamic_symbols
    )

    return tuple(
        sorted(
            str(symbol)
            for symbol in dependencies
        )
    )


def rational_components(
    expression,
):
    """Return canonical numerator and denominator expressions."""
    cancelled = sp.cancel(
        expression
    )

    numerator, denominator = sp.fraction(
        cancelled
    )

    return (
        sp.expand(
            numerator
        ),
        sp.expand(
            denominator
        ),
    )


def polynomial_degree(
    expression,
    *,
    state_symbols,
):
    """Return total polynomial degree or None for non-polynomial input."""
    dynamic_symbols = tuple(
        state_symbols.values()
    )

    if not dynamic_symbols:
        return 0

    try:
        polynomial = sp.Poly(
            expression,
            *dynamic_symbols,
        )

    except sp.PolynomialError:
        return None

    return int(
        polynomial.total_degree()
    )


def state_exponent_map(
    term,
    *,
    state_symbols,
):
    """Return nonzero integer powers of dynamic states in one monomial."""
    powers = term.as_powers_dict()

    result = {}

    for state_id, symbol in state_symbols.items():
        exponent = powers.get(
            symbol,
            sp.Integer(
                0
            ),
        )

        if (
            exponent.is_integer
            and exponent.is_number
            and exponent != 0
        ):
            result[
                state_id
            ] = int(
                exponent
            )

    return result


def monomial_signatures(
    expression,
    *,
    state_symbols,
):
    """Return state-only signatures of additive polynomial terms."""
    expanded = sp.expand(
        expression
    )

    signatures = set()

    for term in sp.Add.make_args(
        expanded
    ):
        exponent_map = state_exponent_map(
            term,
            state_symbols=state_symbols,
        )

        if not exponent_map:
            signatures.add(
                "constant"
            )

            continue

        pieces = []

        for state_id in sorted(
            exponent_map
        ):
            exponent = exponent_map[
                state_id
            ]

            if exponent == 1:
                pieces.append(
                    state_id
                )

            else:
                pieces.append(
                    f"{state_id}^{exponent}"
                )

        signatures.add(
            "*".join(
                pieces
            )
        )

    return tuple(
        sorted(
            signatures
        )
    )


def interaction_pairs(
    expression,
    *,
    state_symbols,
):
    """Return state pairs co-occurring multiplicatively in monomials."""
    expanded = sp.expand(
        expression
    )

    pairs = set()

    for term in sp.Add.make_args(
        expanded
    ):
        exponent_map = state_exponent_map(
            term,
            state_symbols=state_symbols,
        )

        states = sorted(
            state_id
            for state_id, exponent in exponent_map.items()
            if exponent > 0
        )

        for first_state, second_state in combinations(
            states,
            2,
        ):
            pairs.add(
                (
                    first_state,
                    second_state,
                )
            )

    return tuple(
        sorted(
            pairs
        )
    )


def serialize_items(
    values,
):
    """Serialize a sequence compactly for diagnostic output."""
    if not values:
        return "-"

    return ",".join(
        str(value)
        for value in values
    )


def serialize_pairs(
    pairs,
):
    """Serialize state-interaction pairs."""
    if not pairs:
        return "-"

    return ",".join(
        f"{first}*{second}"
        for first, second in pairs
    )


def functional_fingerprint(
    expression,
    *,
    state_symbols,
):
    """Extract a diagnostic functional-form fingerprint."""
    numerator, denominator = rational_components(
        expression
    )

    denominator_is_constant = not bool(
        denominator.free_symbols
        & set(
            state_symbols.values()
        )
    )

    numerator_dependencies = dynamic_dependencies(
        numerator,
        state_symbols=state_symbols,
    )

    denominator_dependencies = dynamic_dependencies(
        denominator,
        state_symbols=state_symbols,
    )

    numerator_degree = polynomial_degree(
        numerator,
        state_symbols=state_symbols,
    )

    denominator_degree = polynomial_degree(
        denominator,
        state_symbols=state_symbols,
    )

    numerator_signatures = monomial_signatures(
        numerator,
        state_symbols=state_symbols,
    )

    denominator_signatures = monomial_signatures(
        denominator,
        state_symbols=state_symbols,
    )

    numerator_interactions = interaction_pairs(
        numerator,
        state_symbols=state_symbols,
    )

    denominator_interactions = interaction_pairs(
        denominator,
        state_symbols=state_symbols,
    )

    return {
        "dependencies": dynamic_dependencies(
            expression,
            state_symbols=state_symbols,
        ),
        "has_rational_state_dependence": (
            not denominator_is_constant
        ),
        "numerator_dependencies": (
            numerator_dependencies
        ),
        "denominator_dependencies": (
            denominator_dependencies
        ),
        "numerator_degree": (
            numerator_degree
        ),
        "denominator_degree": (
            denominator_degree
        ),
        "numerator_signatures": (
            numerator_signatures
        ),
        "denominator_signatures": (
            denominator_signatures
        ),
        "numerator_interactions": (
            numerator_interactions
        ),
        "denominator_interactions": (
            denominator_interactions
        ),
    }


def print_fingerprint(
    *,
    label,
    fingerprint,
):
    """Print one functional fingerprint."""
    print(
        label
    )

    print(
        "  dependencies:",
        serialize_items(
            fingerprint[
                "dependencies"
            ]
        ),
    )

    print(
        "  rational state dependence:",
        fingerprint[
            "has_rational_state_dependence"
        ],
    )

    print(
        "  numerator dependencies:",
        serialize_items(
            fingerprint[
                "numerator_dependencies"
            ]
        ),
    )

    print(
        "  denominator dependencies:",
        serialize_items(
            fingerprint[
                "denominator_dependencies"
            ]
        ),
    )

    print(
        "  numerator degree:",
        fingerprint[
            "numerator_degree"
        ],
    )

    print(
        "  denominator degree:",
        fingerprint[
            "denominator_degree"
        ],
    )

    print(
        "  numerator signatures:",
        serialize_items(
            fingerprint[
                "numerator_signatures"
            ]
        ),
    )

    print(
        "  denominator signatures:",
        serialize_items(
            fingerprint[
                "denominator_signatures"
            ]
        ),
    )

    print(
        "  numerator interactions:",
        serialize_pairs(
            fingerprint[
                "numerator_interactions"
            ]
        ),
    )

    print(
        "  denominator interactions:",
        serialize_pairs(
            fingerprint[
                "denominator_interactions"
            ]
        ),
    )


def inspect_dataset(
    *,
    model_id,
    dataset_name,
    ground_truth,
):
    """Inspect functional structure for one frozen training dataset."""
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

        print()
        print("-" * 80)

        print(
            "STATE:",
            state_id,
        )

        print()
        print(
            "TRUE EQUATION:"
        )

        print(
            " ",
            true_expression,
        )

        print()
        print(
            "INFERRED EQUATION:"
        )

        print(
            " ",
            inferred_expression,
        )

        print()

        print_fingerprint(
            label="GROUND-TRUTH FINGERPRINT",
            fingerprint=true_fingerprint,
        )

        print()

        print_fingerprint(
            label="INFERRED FINGERPRINT",
            fingerprint=inferred_fingerprint,
        )


def run_inspection(
    *,
    model_id,
):
    """Inspect functional structure for one frozen model."""
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
    print("FUNCTIONAL-STRUCTURE INSPECTION")
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
    """Run functional-structure inspection for one frozen model."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect functional structure of ground-truth "
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