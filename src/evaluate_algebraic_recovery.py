"""Evaluate exact algebraic recovery of frozen PySR systems.

This module compares each frozen PySR-selected RHS equation against its
mechanistic SBML ground-truth RHS.

The analysis is strictly post-hoc. It must not influence PySR fitting,
candidate generation, validation selection, Selector v1, held-out test
evaluation, dependency recovery, or functional-form recovery.

Algebraic Recovery v1 reports two distinct concepts:

1. exact_algebraic_match
   Strict symbolic equivalence. The difference between the true and inferred
   RHS must simplify exactly to zero.

2. numerical_equivalence
   Diagnostic approximate equivalence evaluated at deterministic state-space
   points. This does not count as exact algebraic recovery.

Models whose true equations contain independent variables unavailable to the
PySR feature space remain marked as feature_space_mismatch.

Frozen numerical-equivalence diagnostic:
    rtol = 1e-6
    atol = 1e-9
    points = 32
    seed = 4040
"""

import argparse

import numpy as np
import pandas as pd
import sympy as sp

from src.evaluate_dependency_recovery import (
    EXPERIMENT_NAME,
    INFERENCE_FEATURE_SPACE,
    RESULTS_ROOT,
    TRAIN_DATASETS,
    load_selected_system,
    mechanistic_comparability,
)
from src.extract_mechanistic_ground_truth import (
    extract_mechanistic_ground_truth,
)
from src.model_registry import (
    resolve_model_path,
)


OUTPUT_DIRECTORY = (
    "mechanistic_recovery"
)

ALGEBRAIC_RECOVERY_VERSION = (
    "v1"
)

NUMERICAL_EQUIVALENCE_RTOL = (
    1e-6
)

NUMERICAL_EQUIVALENCE_ATOL = (
    1e-9
)

NUMERICAL_EQUIVALENCE_POINTS = (
    32
)

NUMERICAL_EQUIVALENCE_SEED = (
    4040
)


def canonical_expression(
    expression,
):
    """Return a deterministic simplified rational representation."""
    expression = sp.sympify(
        expression
    )

    expression = sp.cancel(
        expression
    )

    expression = sp.together(
        expression
    )

    expression = sp.cancel(
        expression
    )

    return expression


def algebraic_difference(
    *,
    true_expression,
    inferred_expression,
):
    """Return a simplified symbolic difference between two expressions."""
    difference = (
        true_expression
        - inferred_expression
    )

    difference = sp.together(
        difference
    )

    difference = sp.cancel(
        difference
    )

    difference = sp.simplify(
        difference
    )

    return difference


def exact_algebraic_match(
    *,
    true_expression,
    inferred_expression,
):
    """Test whether two expressions are symbolically equivalent."""
    difference = algebraic_difference(
        true_expression=true_expression,
        inferred_expression=inferred_expression,
    )

    if difference == 0:
        return (
            True,
            difference,
        )

    equals_zero = difference.equals(
        0
    )

    return (
        bool(
            equals_zero
        )
        if equals_zero is not None
        else False,
        difference,
    )


def deterministic_state_points(
    *,
    n_states,
):
    """Generate deterministic positive points for numerical diagnostics."""
    rng = np.random.default_rng(
        NUMERICAL_EQUIVALENCE_SEED
    )

    log_values = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            NUMERICAL_EQUIVALENCE_POINTS,
            n_states,
        ),
    )

    return np.power(
        10.0,
        log_values,
    )


def compile_state_expression(
    *,
    expression,
    state_symbols,
):
    """Compile an expression that depends only on dynamic state variables."""
    state_symbol_set = set(
        state_symbols
    )

    unsupported_symbols = (
        expression.free_symbols
        - state_symbol_set
    )

    if unsupported_symbols:
        return (
            None,
            unsupported_symbols,
        )

    function = sp.lambdify(
        state_symbols,
        expression,
        modules="numpy",
    )

    return (
        function,
        set(),
    )


def evaluate_compiled_expression(
    *,
    function,
    state_values,
):
    """Evaluate one compiled symbolic expression at one state vector."""
    value = function(
        *[
            float(value)
            for value in state_values
        ]
    )

    value = np.asarray(
        value,
        dtype=float,
    )

    if value.size != 1:
        raise ValueError(
            "Expected scalar RHS evaluation, "
            f"received shape {value.shape}."
        )

    return float(
        value.reshape(
            -1
        )[0]
    )


def numerical_equivalence(
    *,
    true_expression,
    inferred_expression,
    state_symbols,
):
    """Evaluate approximate equivalence at deterministic state-space points."""
    (
        true_function,
        true_unsupported_symbols,
    ) = compile_state_expression(
        expression=true_expression,
        state_symbols=state_symbols,
    )

    (
        inferred_function,
        inferred_unsupported_symbols,
    ) = compile_state_expression(
        expression=inferred_expression,
        state_symbols=state_symbols,
    )

    unsupported_symbols = (
        true_unsupported_symbols
        | inferred_unsupported_symbols
    )

    if unsupported_symbols:
        return {
            "status": (
                "not_applicable_unsupported_symbols"
            ),
            "match": False,
            "n_points": 0,
            "n_valid_points": 0,
            "max_absolute_error": float(
                "nan"
            ),
            "max_relative_error": float(
                "nan"
            ),
            "unsupported_symbols": (
                ",".join(
                    sorted(
                        str(symbol)
                        for symbol in unsupported_symbols
                    )
                )
            ),
        }

    points = deterministic_state_points(
        n_states=len(
            state_symbols
        )
    )

    absolute_errors = []
    relative_errors = []

    n_valid_points = 0
    all_close = True

    for state_values in points:
        try:
            true_value = evaluate_compiled_expression(
                function=true_function,
                state_values=state_values,
            )

            inferred_value = evaluate_compiled_expression(
                function=inferred_function,
                state_values=state_values,
            )

        except (
            ArithmeticError,
            FloatingPointError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ):
            all_close = False
            continue

        if (
            not np.isfinite(
                true_value
            )
            or not np.isfinite(
                inferred_value
            )
        ):
            all_close = False
            continue

        n_valid_points += 1

        absolute_error = abs(
            inferred_value
            - true_value
        )

        relative_denominator = max(
            abs(
                true_value
            ),
            NUMERICAL_EQUIVALENCE_ATOL,
        )

        relative_error = (
            absolute_error
            / relative_denominator
        )

        absolute_errors.append(
            absolute_error
        )

        relative_errors.append(
            relative_error
        )

        if not np.isclose(
            inferred_value,
            true_value,
            rtol=NUMERICAL_EQUIVALENCE_RTOL,
            atol=NUMERICAL_EQUIVALENCE_ATOL,
        ):
            all_close = False

    if n_valid_points == 0:
        status = (
            "failed_no_valid_points"
        )

        match = False

        max_absolute_error = float(
            "nan"
        )

        max_relative_error = float(
            "nan"
        )

    else:
        status = (
            "evaluated"
        )

        match = (
            all_close
            and n_valid_points
            == NUMERICAL_EQUIVALENCE_POINTS
        )

        max_absolute_error = max(
            absolute_errors
        )

        max_relative_error = max(
            relative_errors
        )

    return {
        "status": status,
        "match": match,
        "n_points": (
            NUMERICAL_EQUIVALENCE_POINTS
        ),
        "n_valid_points": (
            n_valid_points
        ),
        "max_absolute_error": (
            max_absolute_error
        ),
        "max_relative_error": (
            max_relative_error
        ),
        "unsupported_symbols": "",
    }


def evaluate_one_rhs(
    *,
    model_id,
    dataset_name,
    state_id,
    ground_truth,
    selected_row,
):
    """Evaluate algebraic recovery for one frozen inferred RHS."""
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

    canonical_state_symbols = {
        state_name: sp.Symbol(
            state_name
        )
        for state_name in ground_truth.state_ids
    }

    state_symbols = [
        canonical_state_symbols[
            state_name
        ]
        for state_name in ground_truth.state_ids
    ]

    inferred_expression = sp.sympify(
        inferred_equation_text,
        locals=canonical_state_symbols,
    )

    true_expression_original = (
        ground_truth.rhs_by_state[
            state_id
        ]
    )

    true_expression = sp.sympify(
        str(
            true_expression_original
        ),
        locals=canonical_state_symbols,
    )

    canonical_true = canonical_expression(
        true_expression
    )

    canonical_inferred = canonical_expression(
        inferred_expression
    )

    (
        is_exact_match,
        symbolic_difference,
    ) = exact_algebraic_match(
        true_expression=canonical_true,
        inferred_expression=canonical_inferred,
    )

    numerical_result = numerical_equivalence(
        true_expression=canonical_true,
        inferred_expression=canonical_inferred,
        state_symbols=state_symbols,
    )

    return {
        "algebraic_recovery_version": (
            ALGEBRAIC_RECOVERY_VERSION
        ),
        "model_id": model_id,
        "dataset_name": dataset_name,
        "state_id": state_id,
        "ground_truth_is_autonomous": (
            ground_truth.is_autonomous
        ),
        "ground_truth_independent_variables": (
            ",".join(
                sorted(
                    str(value)
                    for value in (
                        ground_truth.explicit_independent_variables
                    )
                )
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
        "canonical_true_equation": str(
            canonical_true
        ),
        "canonical_inferred_equation": str(
            canonical_inferred
        ),
        "symbolic_difference": str(
            symbolic_difference
        ),
        "exact_algebraic_match": (
            is_exact_match
        ),
        "numerical_equivalence_status": (
            numerical_result[
                "status"
            ]
        ),
        "numerical_equivalence_match": (
            numerical_result[
                "match"
            ]
        ),
        "numerical_equivalence_n_points": (
            numerical_result[
                "n_points"
            ]
        ),
        "numerical_equivalence_n_valid_points": (
            numerical_result[
                "n_valid_points"
            ]
        ),
        "numerical_equivalence_max_absolute_error": (
            numerical_result[
                "max_absolute_error"
            ]
        ),
        "numerical_equivalence_max_relative_error": (
            numerical_result[
                "max_relative_error"
            ]
        ),
        "numerical_equivalence_unsupported_symbols": (
            numerical_result[
                "unsupported_symbols"
            ]
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


def evaluate_dataset(
    *,
    model_id,
    dataset_name,
    ground_truth,
):
    """Evaluate algebraic recovery for one frozen selected system."""
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


def summarize_algebraic_recovery(
    frame,
):
    """Summarize exact and numerical recovery by training dataset."""
    rows = []

    for dataset_name, dataset_frame in frame.groupby(
        "dataset_name",
        sort=False,
    ):
        exact_matches = dataset_frame[
            "exact_algebraic_match"
        ].astype(
            bool
        )

        numerical_evaluated = (
            dataset_frame[
                "numerical_equivalence_status"
            ]
            == "evaluated"
        )

        numerical_matches = dataset_frame.loc[
            numerical_evaluated,
            "numerical_equivalence_match",
        ].astype(
            bool
        )

        if numerical_evaluated.any():
            numerical_match_fraction = (
                numerical_matches.mean()
            )

        else:
            numerical_match_fraction = float(
                "nan"
            )

        rows.append(
            {
                "algebraic_recovery_version": (
                    ALGEBRAIC_RECOVERY_VERSION
                ),
                "model_id": (
                    dataset_frame[
                        "model_id"
                    ].iloc[0]
                ),
                "dataset_name": dataset_name,
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
                "n_exact_algebraic_matches": int(
                    exact_matches.sum()
                ),
                "exact_algebraic_recovery_fraction": (
                    exact_matches.mean()
                ),
                "all_rhs_exact_algebraic_match": (
                    bool(
                        exact_matches.all()
                    )
                ),
                "n_numerically_evaluated_rhs": int(
                    numerical_evaluated.sum()
                ),
                "n_numerically_equivalent_rhs": int(
                    numerical_matches.sum()
                ),
                "numerical_equivalence_fraction": (
                    numerical_match_fraction
                ),
                "all_evaluable_rhs_numerically_equivalent": (
                    bool(
                        numerical_matches.all()
                    )
                    if len(
                        numerical_matches
                    ) > 0
                    else False
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
        )

    return pd.DataFrame(
        rows
    )


def evaluate_model(
    *,
    model_id,
):
    """Evaluate algebraic recovery across all frozen training sizes."""
    model_path = resolve_model_path(
        model_id
    )

    ground_truth = extract_mechanistic_ground_truth(
        model_id=model_id,
        model_path=model_path,
    )

    frames = []

    print()
    print("=" * 80)
    print("ALGEBRAIC RECOVERY")

    print(
        "Version:",
        ALGEBRAIC_RECOVERY_VERSION,
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

    print(
        "Numerical equivalence:",
        f"rtol={NUMERICAL_EQUIVALENCE_RTOL}, "
        f"atol={NUMERICAL_EQUIVALENCE_ATOL}, "
        f"points={NUMERICAL_EQUIVALENCE_POINTS}, "
        f"seed={NUMERICAL_EQUIVALENCE_SEED}",
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
            "Exact algebraic matches:",
            f"{int(dataset_frame['exact_algebraic_match'].sum())}"
            f"/{len(dataset_frame)}",
        )

        numerical_evaluated = (
            dataset_frame[
                "numerical_equivalence_status"
            ]
            == "evaluated"
        )

        numerical_matches = dataset_frame.loc[
            numerical_evaluated,
            "numerical_equivalence_match",
        ]

        print(
            "Numerically equivalent RHS:",
            f"{int(numerical_matches.sum())}"
            f"/{int(numerical_evaluated.sum())}",
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = summarize_algebraic_recovery(
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
        / "algebraic_recovery.csv"
    )

    summary_path = (
        output_directory
        / "algebraic_recovery_summary.csv"
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
    print("ALGEBRAIC RECOVERY SUMMARY")

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
    """Print compact per-RHS algebraic-recovery results."""
    columns = [
        "dataset_name",
        "state_id",
        "exact_algebraic_match",
        "numerical_equivalence_status",
        "numerical_equivalence_match",
        "numerical_equivalence_max_absolute_error",
        "numerical_equivalence_max_relative_error",
    ]

    print()
    print("=" * 80)
    print("PER-RHS ALGEBRAIC RECOVERY")

    print(
        frame[
            columns
        ].to_string(
            index=False
        )
    )


def main():
    """Run Algebraic Recovery v1 for one frozen model."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact algebraic recovery for one "
            "frozen Selector v1 model."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000156."
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