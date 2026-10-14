"""Export frozen inferred ODE systems to SBML using rate rules.

Purpose
-------
Represent a coordinate-selected PySR ODE system as a portable SBML model
without changing the inferred equations or assigning reaction-network
interpretations that were not inferred.

This module is representation-only. It does not:

- run PySR;
- perform model selection;
- refit equations;
- evaluate against the original SBML;
- modify frozen scientific outputs;
- reconstruct biochemical reactions.

Each inferred state equation

    dX_i / dt = f_i(X)

is represented as an SBML rateRule for species X_i.

The exported SBML is intended for subsequent portability validation:

    inferred equations + solve_ivp

versus

    exported inferred SBML + RoadRunner

before comparison against the original SBML model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import libsbml
import pandas as pd
import sympy as sp

from .model_spec import ModelSpec
from .preflight import run_preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "generic_exact_experiment"

SELECTION_DIRECTORY = "coordinate_selection"
SELECTED_SYSTEM_FILENAME = "coordinate_selected_system.csv"

SBML_VALIDATION_DIRECTORY = "sbml_validation"
INFERRED_SBML_FILENAME = "inferred_model.xml"

TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

SBML_LEVEL = 3
SBML_VERSION = 2

DEFAULT_COMPARTMENT_ID = "inferred_compartment"

EXPECTED_SELECTION_METHOD = "coordinate_search"


@dataclass(frozen=True)
class ExportedSBML:
    """Metadata describing one exported inferred SBML model."""

    model_id: str
    training_dataset: str
    source_selection_path: Path
    output_path: Path
    state_ids: tuple[str, ...]


def experiment_directory(
    model_id: str,
) -> Path:
    """Return the frozen generic exact experiment directory."""

    return (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
    )


def selected_system_path(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the frozen coordinate-selected system path."""

    return (
        experiment_directory(model_id)
        / training_dataset
        / SELECTION_DIRECTORY
        / SELECTED_SYSTEM_FILENAME
    )


def sbml_validation_directory(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the SBML portability output directory."""

    return (
        experiment_directory(model_id)
        / training_dataset
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
    )


def inferred_sbml_path(
    model_id: str,
    training_dataset: str,
) -> Path:
    """Return the destination path for an inferred SBML model."""

    return (
        sbml_validation_directory(
            model_id=model_id,
            training_dataset=training_dataset,
        )
        / INFERRED_SBML_FILENAME
    )


def load_model_spec(
    model_id: str,
) -> ModelSpec:
    """Load the original model specification through existing preflight."""

    result = run_preflight(
        model_id=model_id,
    )

    if result.spec is None:
        raise RuntimeError(
            f"Preflight returned no ModelSpec for {model_id}."
        )

    if not result.compatibility.compatible:
        raise RuntimeError(
            f"Model {model_id} is not compatible with the current "
            "benchmark assumptions."
        )

    return result.spec


def load_selected_system(
    model_id: str,
    training_dataset: str,
) -> pd.DataFrame:
    """Load and validate one frozen coordinate-selected system."""

    path = selected_system_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen selected system not found: {path}"
        )

    frame = pd.read_csv(
        path
    )

    if frame.empty:
        raise ValueError(
            f"Selected-system file is empty: {path}"
        )

    if len(frame) != 1:
        raise ValueError(
            "Expected exactly one coordinate-selected system, "
            f"but found {len(frame)} rows in {path}."
        )

    required_columns = {
        "model_id",
        "dataset_name",
        "selection_method",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Selected-system file is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    observed_model_id = str(
        frame.iloc[0]["model_id"]
    )

    if observed_model_id != model_id:
        raise ValueError(
            f"Expected model_id={model_id}, "
            f"found {observed_model_id}."
        )

    observed_dataset_name = str(
        frame.iloc[0]["dataset_name"]
    )

    if observed_dataset_name != training_dataset:
        raise ValueError(
            f"Expected dataset_name={training_dataset}, "
            f"found {observed_dataset_name}."
        )

    observed_selection_method = str(
        frame.iloc[0]["selection_method"]
    )

    if observed_selection_method != EXPECTED_SELECTION_METHOD:
        raise ValueError(
            f"Expected selection_method={EXPECTED_SELECTION_METHOD}, "
            f"found {observed_selection_method}."
        )

    return frame


def validate_selected_states(
    spec: ModelSpec,
    selected_system: pd.DataFrame,
) -> None:
    """Require one frozen inferred equation column for every dynamic state."""

    if len(selected_system) != 1:
        raise ValueError(
            "Expected exactly one selected system row."
        )

    missing_equation_columns = []

    for state_id in spec.state_ids:
        column = f"equation_{state_id}"

        if column not in selected_system.columns:
            missing_equation_columns.append(
                column
            )

    if missing_equation_columns:
        raise ValueError(
            "Selected-system file is missing inferred equation "
            "columns required by ModelSpec: "
            + ", ".join(
                missing_equation_columns
            )
        )

    row = selected_system.iloc[0]

    for state_id in spec.state_ids:
        column = f"equation_{state_id}"
        value = row[column]

        if pd.isna(value):
            raise ValueError(
                f"Inferred equation is missing for state {state_id} "
                f"in column {column}."
            )

        expression = str(
            value
        ).strip()

        if not expression:
            raise ValueError(
                f"Inferred equation is empty for state {state_id}."
            )


def selected_expressions(
    spec: ModelSpec,
    selected_system: pd.DataFrame,
) -> dict[str, str]:
    """Return frozen inferred expressions keyed by state identifier."""

    validate_selected_states(
        spec=spec,
        selected_system=selected_system,
    )

    row = selected_system.iloc[0]

    return {
        state_id: str(
            row[f"equation_{state_id}"]
        ).strip()
        for state_id in spec.state_ids
    }


def sympy_expression(
    expression: str,
    state_ids: tuple[str, ...],
) -> sp.Expr:
    """Parse an inferred expression using explicit state symbols."""

    symbol_map = {
        state_id: sp.Symbol(state_id)
        for state_id in state_ids
    }

    parsed = sp.sympify(
        expression,
        locals=symbol_map,
    )

    unknown_symbols = {
        str(symbol)
        for symbol in parsed.free_symbols
        if str(symbol) not in symbol_map
    }

    if unknown_symbols:
        raise ValueError(
            "Inferred expression contains symbols outside "
            "the supported state feature space: "
            f"{sorted(unknown_symbols)}"
        )

    return parsed


def make_real_ast(
    value: float,
) -> libsbml.ASTNode:
    """Create a libSBML real-number AST node."""

    node = libsbml.ASTNode(
        libsbml.AST_REAL
    )

    code = node.setValue(
        float(value)
    )

    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(
            f"Could not create SBML real AST node for {value}."
        )

    return node


def make_integer_ast(
    value: int,
) -> libsbml.ASTNode:
    """Create a libSBML integer AST node."""

    node = libsbml.ASTNode(
        libsbml.AST_INTEGER
    )

    code = node.setValue(
        int(value)
    )

    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(
            f"Could not create SBML integer AST node for {value}."
        )

    return node


def make_name_ast(
    name: str,
) -> libsbml.ASTNode:
    """Create a libSBML variable-name AST node."""

    node = libsbml.ASTNode(
        libsbml.AST_NAME
    )

    code = node.setName(
        name
    )

    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(
            f"Could not create SBML name AST node for {name}."
        )

    return node


def make_operator_ast(
    node_type: int,
    children: list[libsbml.ASTNode],
) -> libsbml.ASTNode:
    """Create an operator AST node and attach its children."""

    node = libsbml.ASTNode(
        node_type
    )

    for child in children:
        code = node.addChild(
            child
        )

        if code != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(
                "Could not attach child to SBML AST operator."
            )

    return node


def sympy_to_sbml_ast(
    expression: sp.Expr,
) -> libsbml.ASTNode:
    """Convert a supported SymPy expression directly to a libSBML AST.

    Supported expression classes are deliberately limited to the frozen
    PySR grammar used by this benchmark:

        +, -, *, /, powers, state symbols, and numeric constants.

    SymPy canonicalizes subtraction and division into addition,
    multiplication, and negative powers. The resulting AST therefore uses
    equivalent SBML arithmetic rather than attempting textual conversion.
    """

    if isinstance(
        expression,
        sp.Symbol,
    ):
        return make_name_ast(
            str(expression)
        )

    if isinstance(
        expression,
        sp.Integer,
    ):
        return make_integer_ast(
            int(expression)
        )

    if isinstance(
        expression,
        sp.Rational,
    ):
        numerator = make_integer_ast(
            int(expression.p)
        )

        denominator = make_integer_ast(
            int(expression.q)
        )

        return make_operator_ast(
            libsbml.AST_DIVIDE,
            [
                numerator,
                denominator,
            ],
        )

    if isinstance(
        expression,
        sp.Float,
    ):
        return make_real_ast(
            float(expression)
        )

    if expression.is_Number:
        return make_real_ast(
            float(expression)
        )

    if isinstance(
        expression,
        sp.Add,
    ):
        children = [
            sympy_to_sbml_ast(argument)
            for argument in expression.args
        ]

        return make_operator_ast(
            libsbml.AST_PLUS,
            children,
        )

    if isinstance(
        expression,
        sp.Mul,
    ):
        numerator_factors = []
        denominator_factors = []

        for argument in expression.args:
            if (
                isinstance(argument, sp.Pow)
                and argument.exp.is_number
                and argument.exp.is_negative
            ):
                denominator_factors.append(
                    sp.Pow(
                        argument.base,
                        -argument.exp,
                        evaluate=False,
                    )
                )
            else:
                numerator_factors.append(
                    argument
                )

        if denominator_factors:
            if numerator_factors:
                numerator_expression = sp.Mul(
                    *numerator_factors,
                    evaluate=False,
                )
            else:
                numerator_expression = sp.Integer(
                    1
                )

            if len(denominator_factors) == 1:
                denominator_expression = denominator_factors[0]
            else:
                denominator_expression = sp.Mul(
                    *denominator_factors,
                    evaluate=False,
                )

            return make_operator_ast(
                libsbml.AST_DIVIDE,
                [
                    sympy_to_sbml_ast(
                        numerator_expression
                    ),
                    sympy_to_sbml_ast(
                        denominator_expression
                    ),
                ],
            )

        children = [
            sympy_to_sbml_ast(argument)
            for argument in expression.args
        ]

        return make_operator_ast(
            libsbml.AST_TIMES,
            children,
        )

    if isinstance(
        expression,
        sp.Pow,
    ):
        base = expression.base
        exponent = expression.exp

        if (
            exponent.is_number
            and exponent.is_negative
        ):
            positive_exponent = -exponent

            denominator = sp.Pow(
                base,
                positive_exponent,
                evaluate=False,
            )

            return make_operator_ast(
                libsbml.AST_DIVIDE,
                [
                    make_integer_ast(
                        1
                    ),
                    sympy_to_sbml_ast(
                        denominator
                    ),
                ],
            )

        return make_operator_ast(
            libsbml.AST_POWER,
            [
                sympy_to_sbml_ast(
                    base
                ),
                sympy_to_sbml_ast(
                    exponent
                ),
            ],
        )

    raise TypeError(
        "Unsupported SymPy expression encountered during "
        "SBML export: "
        f"type={type(expression).__name__}, "
        f"expression={expression}"
    )


def inferred_expression_to_sbml_ast(
    expression: str,
    state_ids: tuple[str, ...],
) -> libsbml.ASTNode:
    """Parse one frozen equation and convert it to a libSBML AST."""

    parsed = sympy_expression(
        expression=expression,
        state_ids=state_ids,
    )

    return sympy_to_sbml_ast(
        parsed
    )


def check_libsbml_code(
    code: int,
    operation: str,
) -> None:
    """Raise when a libSBML setter reports failure."""

    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(
            "libSBML operation failed during "
            f"{operation}: code={code}"
        )


def create_document() -> libsbml.SBMLDocument:
    """Create an empty SBML Level 3 Version 2 document."""

    document = libsbml.SBMLDocument(
        SBML_LEVEL,
        SBML_VERSION,
    )

    if document is None:
        raise RuntimeError(
            "Could not create SBMLDocument."
        )

    return document


def create_model(
    document: libsbml.SBMLDocument,
    model_id: str,
    training_dataset: str,
) -> libsbml.Model:
    """Create the inferred SBML model container."""

    model = document.createModel()

    if model is None:
        raise RuntimeError(
            "Could not create SBML model."
        )

    sbml_model_id = (
        f"{model_id}_{training_dataset}_pysr_inferred"
    )

    check_libsbml_code(
        model.setId(
            sbml_model_id
        ),
        "setting model id",
    )

    check_libsbml_code(
        model.setName(
            f"{model_id} PySR inferred ODE system "
            f"({training_dataset})"
        ),
        "setting model name",
    )

    return model


def create_compartment(
    model: libsbml.Model,
) -> str:
    """Create a unit-size constant compartment for inferred states."""

    compartment = model.createCompartment()

    if compartment is None:
        raise RuntimeError(
            "Could not create SBML compartment."
        )

    check_libsbml_code(
        compartment.setId(
            DEFAULT_COMPARTMENT_ID
        ),
        "setting compartment id",
    )

    check_libsbml_code(
        compartment.setConstant(
            True
        ),
        "setting compartment constant flag",
    )

    check_libsbml_code(
        compartment.setSize(
            1.0
        ),
        "setting compartment size",
    )

    check_libsbml_code(
        compartment.setSpatialDimensions(
            3.0
        ),
        "setting compartment spatial dimensions",
    )

    return DEFAULT_COMPARTMENT_ID


def create_species(
    model: libsbml.Model,
    *,
    state_id: str,
    initial_value: float,
    compartment_id: str,
) -> None:
    """Create one dynamic SBML species."""

    species = model.createSpecies()

    if species is None:
        raise RuntimeError(
            f"Could not create SBML species {state_id}."
        )

    check_libsbml_code(
        species.setId(
            state_id
        ),
        f"setting species id for {state_id}",
    )

    check_libsbml_code(
        species.setCompartment(
            compartment_id
        ),
        f"setting compartment for {state_id}",
    )

    check_libsbml_code(
        species.setInitialConcentration(
            float(initial_value)
        ),
        f"setting initial concentration for {state_id}",
    )

    check_libsbml_code(
        species.setBoundaryCondition(
            False
        ),
        f"setting boundaryCondition for {state_id}",
    )

    check_libsbml_code(
        species.setConstant(
            False
        ),
        f"setting constant flag for {state_id}",
    )

    check_libsbml_code(
        species.setHasOnlySubstanceUnits(
            False
        ),
        f"setting substance-units flag for {state_id}",
    )


def create_rate_rule(
    model: libsbml.Model,
    *,
    state_id: str,
    expression: str,
    state_ids: tuple[str, ...],
) -> None:
    """Create one SBML rateRule from a frozen inferred RHS."""

    rule = model.createRateRule()

    if rule is None:
        raise RuntimeError(
            f"Could not create rateRule for {state_id}."
        )

    check_libsbml_code(
        rule.setVariable(
            state_id
        ),
        f"setting rateRule variable for {state_id}",
    )

    math_ast = inferred_expression_to_sbml_ast(
        expression=expression,
        state_ids=state_ids,
    )

    check_libsbml_code(
        rule.setMath(
            math_ast
        ),
        f"setting rateRule math for {state_id}",
    )


def validate_sbml_document(
    document: libsbml.SBMLDocument,
) -> None:
    """Run libSBML consistency checks and reject validation errors."""

    document.checkConsistency()

    errors: list[str] = []

    for index in range(
        document.getNumErrors()
    ):
        error = document.getError(
            index
        )

        if (
            error.getSeverity()
            >= libsbml.LIBSBML_SEV_ERROR
        ):
            errors.append(
                f"[{error.getErrorId()}] "
                f"{error.getMessage().strip()}"
            )

    if errors:
        formatted = "\n".join(
            f"  - {message}"
            for message in errors
        )

        raise RuntimeError(
            "Exported inferred SBML failed "
            "consistency validation:\n"
            f"{formatted}"
        )


def build_inferred_sbml_document(
    *,
    spec: ModelSpec,
    selected_system: pd.DataFrame,
    training_dataset: str,
) -> libsbml.SBMLDocument:
    """Build SBML representing the frozen inferred ODE system."""

    expressions = selected_expressions(
        spec=spec,
        selected_system=selected_system,
    )

    document = create_document()

    model = create_model(
        document=document,
        model_id=spec.model_id,
        training_dataset=training_dataset,
    )

    compartment_id = create_compartment(
        model
    )

    for state_id in spec.state_ids:
        create_species(
            model,
            state_id=state_id,
            initial_value=float(
                spec.initial_values[state_id]
            ),
            compartment_id=compartment_id,
        )

    state_ids = tuple(
        spec.state_ids
    )

    for state_id in state_ids:
        create_rate_rule(
            model,
            state_id=state_id,
            expression=expressions[state_id],
            state_ids=state_ids,
        )

    validate_sbml_document(
        document
    )

    return document


def write_sbml_document(
    document: libsbml.SBMLDocument,
    output_path: Path,
) -> None:
    """Write SBML without touching frozen source artifacts."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = libsbml.SBMLWriter()

    success = writer.writeSBMLToFile(
        document,
        str(output_path),
    )

    if success != 1:
        raise RuntimeError(
            "Failed to write inferred SBML: "
            f"{output_path}"
        )


def export_selected_system(
    *,
    model_id: str,
    training_dataset: str,
) -> ExportedSBML:
    """Export one frozen selected PySR system as rate-rule SBML."""

    if training_dataset not in TRAINING_DATASETS:
        raise ValueError(
            f"Unsupported training dataset: "
            f"{training_dataset}. "
            f"Expected one of {TRAINING_DATASETS}."
        )

    spec = load_model_spec(
        model_id
    )

    selection_path = selected_system_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    selected_system = load_selected_system(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    document = build_inferred_sbml_document(
        spec=spec,
        selected_system=selected_system,
        training_dataset=training_dataset,
    )

    output_path = inferred_sbml_path(
        model_id=model_id,
        training_dataset=training_dataset,
    )

    write_sbml_document(
        document=document,
        output_path=output_path,
    )

    return ExportedSBML(
        model_id=model_id,
        training_dataset=training_dataset,
        source_selection_path=selection_path,
        output_path=output_path,
        state_ids=tuple(
            spec.state_ids
        ),
    )


def export_all_training_sizes(
    model_id: str,
) -> tuple[ExportedSBML, ...]:
    """Export all four frozen training-size selections."""

    exports = []

    for training_dataset in TRAINING_DATASETS:
        exported = export_selected_system(
            model_id=model_id,
            training_dataset=training_dataset,
        )

        exports.append(
            exported
        )

    return tuple(
        exports
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Export frozen coordinate-selected PySR ODE "
            "systems to SBML rate-rule models."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000079."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the inferred-SBML export."""

    args = parse_arguments()

    exports = export_all_training_sizes(
        model_id=args.model_id,
    )

    print(
        f"Exported {len(exports)} inferred SBML models "
        f"for {args.model_id}."
    )

    for exported in exports:
        print(
            f"  {exported.training_dataset}: "
            f"{exported.output_path}"
        )

    print()

    print(
        "Frozen PySR equations and model-selection outputs "
        "were read but not modified."
    )

    print(
        "No PySR inference, selection, refitting, or "
        "scientific evaluation was performed."
    )


if __name__ == "__main__":
    main()