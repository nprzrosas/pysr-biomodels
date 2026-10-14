"""Extract symbolic mechanistic ground truth from an SBML model.

This module reconstructs the ODE right-hand side for every dynamic species
directly from the original SBML reaction network.

The reconstruction is intended exclusively for post-hoc mechanistic recovery
evaluation. It must not be used during PySR fitting, candidate generation,
validation selection, or held-out test selection.
"""

from dataclasses import dataclass
from pathlib import Path

import libsbml
import sympy as sp


@dataclass(frozen=True)
class ReactionGroundTruth:
    """Symbolic representation of one SBML reaction."""

    reaction_id: str
    kinetic_law_original: str
    kinetic_law_expanded: sp.Expr
    stoichiometry: dict[str, sp.Expr]


@dataclass(frozen=True)
class MechanisticGroundTruth:
    """Symbolic ground truth reconstructed from an SBML model."""

    model_id: str
    model_path: Path
    state_ids: tuple[str, ...]
    reactions: tuple[ReactionGroundTruth, ...]
    rhs_by_state: dict[str, sp.Expr]
    is_autonomous: bool
    explicit_independent_variables: tuple[str, ...]


def read_sbml_document(
    model_path,
):
    """Read one SBML document and fail on parser errors."""
    model_path = Path(
        model_path
    ).resolve()

    document = libsbml.readSBMLFromFile(
        str(model_path)
    )

    if document is None:
        raise ValueError(
            f"Could not read SBML file: {model_path}"
        )

    severe_errors = []

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
            severe_errors.append(
                error.getMessage()
            )

    if severe_errors:
        raise ValueError(
            "SBML parser reported error(s):\n"
            + "\n".join(
                severe_errors
            )
        )

    model = document.getModel()

    if model is None:
        raise ValueError(
            f"SBML file contains no model: {model_path}"
        )

    return (
        model_path,
        document,
        model,
    )


def formula_from_ast(
    ast_node,
):
    """Convert a libSBML AST node to an SBML formula string."""
    if ast_node is None:
        raise ValueError(
            "Cannot convert an empty SBML AST node."
        )

    formula = libsbml.formulaToL3String(
        ast_node
    )

    if formula is None:
        raise ValueError(
            "libSBML could not convert AST node to a formula."
        )

    return str(
        formula
    )


def build_function_definitions(
    model,
):
    """Extract SBML user-defined functions as symbolic templates."""
    definitions = {}

    for index in range(
        model.getNumFunctionDefinitions()
    ):
        definition = model.getFunctionDefinition(
            index
        )

        function_id = definition.getId()
        math_node = definition.getMath()

        if math_node is None:
            raise ValueError(
                f"Function {function_id} "
                "has no MathML definition."
            )

        if (
            math_node.getType()
            != libsbml.AST_LAMBDA
        ):
            raise ValueError(
                f"Function {function_id} "
                "is not represented as a lambda."
            )

        number_of_children = (
            math_node.getNumChildren()
        )

        if number_of_children < 1:
            raise ValueError(
                f"Function {function_id} "
                "has an empty lambda."
            )

        argument_names = []

        for child_index in range(
            number_of_children - 1
        ):
            argument_node = (
                math_node.getChild(
                    child_index
                )
            )

            argument_name = (
                argument_node.getName()
            )

            if not argument_name:
                raise ValueError(
                    f"Function {function_id} "
                    "contains an unnamed argument."
                )

            argument_names.append(
                argument_name
            )

        body_node = math_node.getChild(
            number_of_children - 1
        )

        body_formula = formula_from_ast(
            body_node
        )

        definitions[
            function_id
        ] = {
            "arguments": tuple(
                argument_names
            ),
            "body_formula": body_formula,
        }

    return definitions


def build_numeric_constants(
    model,
):
    """Collect fixed SBML compartments, parameters, and constant species."""
    constants = {}

    compartment_sizes = {}

    for index in range(
        model.getNumCompartments()
    ):
        compartment = model.getCompartment(
            index
        )

        if compartment.isSetSize():
            compartment_size = float(
                compartment.getSize()
            )

            constants[
                compartment.getId()
            ] = compartment_size

            compartment_sizes[
                compartment.getId()
            ] = compartment_size

    for index in range(
        model.getNumParameters()
    ):
        parameter = model.getParameter(
            index
        )

        if parameter.isSetValue():
            constants[
                parameter.getId()
            ] = float(
                parameter.getValue()
            )

    for index in range(
        model.getNumSpecies()
    ):
        species = model.getSpecies(
            index
        )

        if not species.getBoundaryCondition():
            continue

        if not species.getConstant():
            continue

        species_id = species.getId()

        if species.isSetInitialConcentration():
            constants[
                species_id
            ] = float(
                species.getInitialConcentration()
            )

            continue

        if not species.isSetInitialAmount():
            raise ValueError(
                "Constant boundary species "
                f"{species_id} has neither "
                "initial concentration nor "
                "initial amount."
            )

        initial_amount = float(
            species.getInitialAmount()
        )

        if species.getHasOnlySubstanceUnits():
            constants[
                species_id
            ] = initial_amount

            continue

        compartment_id = (
            species.getCompartment()
        )

        if (
            compartment_id
            not in compartment_sizes
        ):
            raise ValueError(
                "Cannot convert initial amount "
                "to concentration for constant "
                f"boundary species {species_id}: "
                f"compartment {compartment_id} "
                "has no numeric size."
            )

        compartment_size = (
            compartment_sizes[
                compartment_id
            ]
        )

        if compartment_size == 0.0:
            raise ValueError(
                "Cannot convert initial amount "
                "to concentration for constant "
                f"boundary species {species_id}: "
                f"compartment {compartment_id} "
                "has zero size."
            )

        constants[
            species_id
        ] = (
            initial_amount
            / compartment_size
        )

    return constants


def build_sympy_locals(
    *,
    state_ids,
    symbol_names,
    function_definitions,
):
    """Create explicit SymPy symbols and functions for SBML identifiers."""
    locals_dictionary = {}

    function_ids = set(
        function_definitions
    )

    for name in sorted(
        set(symbol_names)
        | set(state_ids)
    ):
        if name in function_ids:
            locals_dictionary[
                name
            ] = sp.Function(
                name
            )

        else:
            locals_dictionary[
                name
            ] = sp.Symbol(
                name
            )

    return locals_dictionary


def normalize_formula_text(
    formula,
):
    """Translate basic SBML formula syntax into SymPy syntax."""
    normalized = str(
        formula
    )

    normalized = normalized.replace(
        "^",
        "**",
    )

    return normalized


def parse_formula(
    formula,
    *,
    locals_dictionary,
):
    """Parse one SBML formula as a SymPy expression."""
    normalized = normalize_formula_text(
        formula
    )

    return sp.sympify(
        normalized,
        locals=locals_dictionary,
    )


def expand_user_functions(
    expression,
    *,
    function_definitions,
    locals_dictionary,
):
    """Recursively expand user-defined SBML functions."""
    expanded = expression

    maximum_iterations = max(
        1,
        len(
            function_definitions
        )
        + 1,
    )

    for _ in range(
        maximum_iterations
    ):
        changed = False

        function_calls = list(
            expanded.atoms(
                sp.Function
            )
        )

        for function_call in function_calls:
            function_id = str(
                function_call.func
            )

            if (
                function_id
                not in function_definitions
            ):
                continue

            definition = (
                function_definitions[
                    function_id
                ]
            )

            arguments = definition[
                "arguments"
            ]

            if (
                len(
                    function_call.args
                )
                != len(
                    arguments
                )
            ):
                raise ValueError(
                    f"Function {function_id} "
                    f"expects {len(arguments)} "
                    "argument(s), "
                    f"received "
                    f"{len(function_call.args)}."
                )

            body = parse_formula(
                definition[
                    "body_formula"
                ],
                locals_dictionary=(
                    locals_dictionary
                ),
            )

            substitutions = {
                locals_dictionary[
                    argument_name
                ]: argument_value
                for (
                    argument_name,
                    argument_value,
                ) in zip(
                    arguments,
                    function_call.args,
                )
            }

            body = body.subs(
                substitutions,
                simultaneous=True,
            )

            expanded = expanded.xreplace(
                {
                    function_call: body,
                }
            )

            changed = True

        if not changed:
            break

    remaining_functions = []

    for function_call in expanded.atoms(
        sp.Function
    ):
        function_id = str(
            function_call.func
        )

        if (
            function_id
            in function_definitions
        ):
            remaining_functions.append(
                function_call
            )

    if remaining_functions:
        raise ValueError(
            "Could not fully expand SBML "
            "user-defined functions: "
            + ", ".join(
                str(function_call)
                for function_call
                in remaining_functions
            )
        )

    return expanded


def local_parameters_for_reaction(
    reaction,
):
    """Return numeric local parameters from one kinetic law."""
    kinetic_law = (
        reaction.getKineticLaw()
    )

    if kinetic_law is None:
        raise ValueError(
            f"Reaction {reaction.getId()} "
            "has no kinetic law."
        )

    parameters = {}

    for index in range(
        kinetic_law.getNumParameters()
    ):
        parameter = (
            kinetic_law.getParameter(
                index
            )
        )

        if not parameter.isSetValue():
            raise ValueError(
                f"Local parameter "
                f"{parameter.getId()} "
                f"in reaction "
                f"{reaction.getId()} "
                "has no value."
            )

        parameters[
            parameter.getId()
        ] = float(
            parameter.getValue()
        )

    return parameters


def species_reference_stoichiometry(
    species_reference,
):
    """Return the stoichiometric coefficient of a species reference."""
    if (
        species_reference
        .isSetStoichiometry()
    ):
        return sp.Float(
            species_reference
            .getStoichiometry()
        )

    return sp.Integer(
        1
    )


def reaction_stoichiometry(
    reaction,
    *,
    state_ids,
):
    """Return net stoichiometric coefficients for dynamic species."""
    coefficients = {
        state_id: sp.Integer(
            0
        )
        for state_id in state_ids
    }

    for index in range(
        reaction.getNumReactants()
    ):
        reference = reaction.getReactant(
            index
        )

        species_id = (
            reference.getSpecies()
        )

        if species_id not in coefficients:
            continue

        coefficients[
            species_id
        ] -= (
            species_reference_stoichiometry(
                reference
            )
        )

    for index in range(
        reaction.getNumProducts()
    ):
        reference = reaction.getProduct(
            index
        )

        species_id = (
            reference.getSpecies()
        )

        if species_id not in coefficients:
            continue

        coefficients[
            species_id
        ] += (
            species_reference_stoichiometry(
                reference
            )
        )

    return coefficients


def collect_symbol_names(
    *,
    model,
    state_ids,
    function_definitions,
):
    """Collect identifiers needed when parsing SBML formulas."""
    names = set(
        state_ids
    )

    for index in range(
        model.getNumSpecies()
    ):
        species = model.getSpecies(
            index
        )

        names.add(
            species.getId()
        )

    for index in range(
        model.getNumCompartments()
    ):
        names.add(
            model.getCompartment(
                index
            ).getId()
        )

    for index in range(
        model.getNumParameters()
    ):
        names.add(
            model.getParameter(
                index
            ).getId()
        )

    for (
        function_id,
        definition,
    ) in function_definitions.items():
        names.add(
            function_id
        )

        names.update(
            definition[
                "arguments"
            ]
        )

    for index in range(
        model.getNumReactions()
    ):
        reaction = model.getReaction(
            index
        )

        kinetic_law = (
            reaction.getKineticLaw()
        )

        if kinetic_law is None:
            continue

        for parameter_index in range(
            kinetic_law.getNumParameters()
        ):
            names.add(
                kinetic_law.getParameter(
                    parameter_index
                ).getId()
            )

    return names


def extract_dynamic_state_ids(
    model,
):
    """Return species represented as dynamic ODE states."""
    state_ids = []

    for index in range(
        model.getNumSpecies()
    ):
        species = model.getSpecies(
            index
        )

        if species.getBoundaryCondition():
            continue

        if species.getConstant():
            continue

        state_ids.append(
            species.getId()
        )

    return tuple(
        state_ids
    )


def find_explicit_independent_variables(
    *,
    rhs_by_state,
    state_ids,
):
    """Find free RHS symbols that are not dynamic state variables."""
    state_symbols = {
        sp.Symbol(
            state_id
        )
        for state_id in state_ids
    }

    independent_symbols = set()

    for expression in rhs_by_state.values():
        independent_symbols.update(
            expression.free_symbols
            - state_symbols
        )

    return tuple(
        sorted(
            str(symbol)
            for symbol in independent_symbols
        )
    )


def extract_mechanistic_ground_truth(
    *,
    model_id,
    model_path,
):
    """Reconstruct symbolic dynamic-state RHS equations from SBML."""
    (
        resolved_model_path,
        _document,
        model,
    ) = read_sbml_document(
        model_path
    )

    state_ids = extract_dynamic_state_ids(
        model
    )

    if not state_ids:
        raise ValueError(
            f"No dynamic species found "
            f"for {model_id}."
        )

    function_definitions = (
        build_function_definitions(
            model
        )
    )

    numeric_constants = (
        build_numeric_constants(
            model
        )
    )

    symbol_names = collect_symbol_names(
        model=model,
        state_ids=state_ids,
        function_definitions=(
            function_definitions
        ),
    )

    locals_dictionary = (
        build_sympy_locals(
            state_ids=state_ids,
            symbol_names=symbol_names,
            function_definitions=(
                function_definitions
            ),
        )
    )

    constant_substitutions = {
        locals_dictionary[
            name
        ]: value
        for (
            name,
            value,
        ) in numeric_constants.items()
        if (
            name
            in locals_dictionary
        )
    }

    rhs_by_state = {
        state_id: sp.Integer(
            0
        )
        for state_id in state_ids
    }

    reactions = []

    for reaction_index in range(
        model.getNumReactions()
    ):
        reaction = model.getReaction(
            reaction_index
        )

        reaction_id = (
            reaction.getId()
        )

        kinetic_law = (
            reaction.getKineticLaw()
        )

        if kinetic_law is None:
            raise ValueError(
                f"Reaction {reaction_id} "
                "has no kinetic law."
            )

        kinetic_math = (
            kinetic_law.getMath()
        )

        if kinetic_math is None:
            raise ValueError(
                f"Reaction {reaction_id} "
                "has no kinetic MathML."
            )

        original_formula = (
            formula_from_ast(
                kinetic_math
            )
        )

        expression = parse_formula(
            original_formula,
            locals_dictionary=(
                locals_dictionary
            ),
        )

        expression = (
            expand_user_functions(
                expression,
                function_definitions=(
                    function_definitions
                ),
                locals_dictionary=(
                    locals_dictionary
                ),
            )
        )

        local_parameters = (
            local_parameters_for_reaction(
                reaction
            )
        )

        local_substitutions = {
            locals_dictionary[
                name
            ]: value
            for (
                name,
                value,
            ) in local_parameters.items()
            if (
                name
                in locals_dictionary
            )
        }

        expression = expression.subs(
            constant_substitutions,
            simultaneous=True,
        )

        expression = expression.subs(
            local_substitutions,
            simultaneous=True,
        )

        expression = sp.factor(
            sp.cancel(
                expression
            )
        )

        stoichiometry = (
            reaction_stoichiometry(
                reaction,
                state_ids=state_ids,
            )
        )

        for state_id in state_ids:
            coefficient = (
                stoichiometry[
                    state_id
                ]
            )

            if coefficient == 0:
                continue

            rhs_by_state[
                state_id
            ] += (
                coefficient
                * expression
            )

        reactions.append(
            ReactionGroundTruth(
                reaction_id=reaction_id,
                kinetic_law_original=(
                    original_formula
                ),
                kinetic_law_expanded=(
                    expression
                ),
                stoichiometry=(
                    stoichiometry
                ),
            )
        )

    rhs_by_state = {
        state_id: sp.factor(
            sp.cancel(
                expression
            )
        )
        for (
            state_id,
            expression,
        ) in rhs_by_state.items()
    }

    explicit_independent_variables = (
        find_explicit_independent_variables(
            rhs_by_state=rhs_by_state,
            state_ids=state_ids,
        )
    )

    is_autonomous = (
        len(
            explicit_independent_variables
        )
        == 0
    )

    return MechanisticGroundTruth(
        model_id=model_id,
        model_path=(
            resolved_model_path
        ),
        state_ids=state_ids,
        reactions=tuple(
            reactions
        ),
        rhs_by_state=(
            rhs_by_state
        ),
        is_autonomous=is_autonomous,
        explicit_independent_variables=(
            explicit_independent_variables
        ),
    )


def print_mechanistic_ground_truth(
    ground_truth,
):
    """Print reaction-level and state-level mechanistic ground truth."""
    print(
        "MODEL"
    )

    print(
        ground_truth.model_id
    )

    print(
        "\nDYNAMIC STATES"
    )

    print(
        list(
            ground_truth.state_ids
        )
    )

    print(
        "\nAUTONOMY"
    )

    print(
        "Autonomous:",
        ground_truth.is_autonomous,
    )

    print(
        "Explicit independent variables:",
        ground_truth.explicit_independent_variables,
    )

    print(
        "\nREACTIONS"
    )

    for reaction in ground_truth.reactions:
        print()

        print(
            "=" * 70
        )

        print(
            "Reaction:",
            reaction.reaction_id,
        )

        print(
            "Original kinetic law:",
            reaction.kinetic_law_original,
        )

        print(
            "Expanded kinetic law:",
            reaction.kinetic_law_expanded,
        )

        nonzero_stoichiometry = {
            state_id: coefficient
            for (
                state_id,
                coefficient,
            ) in (
                reaction
                .stoichiometry
                .items()
            )
            if coefficient != 0
        }

        print(
            "Stoichiometry:",
            nonzero_stoichiometry,
        )

    print()

    print(
        "=" * 70
    )

    print(
        "GROUND-TRUTH RHS"
    )

    for state_id in ground_truth.state_ids:
        print(
            f"d{state_id}/dt = "
            f"{ground_truth.rhs_by_state[state_id]}"
        )