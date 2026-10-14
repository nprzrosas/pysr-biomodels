"""Generate a human-readable model recovery report.

This reporting module reads frozen inference, selection, validation, test,
and mechanistic-recovery outputs. It may reconstruct the original SBML
trajectory and frozen training trajectories solely for visualization.

It does not rerun PySR, change Selector v1 decisions, redefine scientific
metrics, or modify frozen scientific results.
"""

import argparse
import math
import re
import shutil
from pathlib import Path

import pandas as pd
import sympy as sp

from .evaluate_dependency_recovery import (
    RESULTS_ROOT,
    TRAIN_DATASETS,
)
from .extract_mechanistic_ground_truth import (
    extract_mechanistic_ground_truth,
)
from .model_registry import (
    resolve_model_path,
)
from .reporting.plot_model_trajectories import (
    generate_trajectory_figures,
)


EXPERIMENT_NAME = "generic_exact_experiment"

SELECTION_DIRECTORY = "coordinate_selection"
SELECTED_SYSTEM_FILENAME = "coordinate_selected_system.csv"

MECHANISTIC_RECOVERY_DIRECTORY = "mechanistic_recovery"
DEPENDENCY_FILENAME = "dependency_recovery.csv"
FUNCTIONAL_FORM_FILENAME = "functional_form_recovery.csv"
ALGEBRAIC_FILENAME = "algebraic_recovery.csv"

REPORT_DIRECTORY = "model_recovery_report"
REPORT_FILENAME = "report.md"
FIGURE_DIRECTORY = "figures"

REFERENCE_TRAJECTORY_FILENAME = "reference_sbml_trajectory.png"
TRAINING_TRAJECTORIES_FILENAME = "training_trajectories.png"
VALIDATION_TRAJECTORIES_FILENAME = "validation_truth_vs_inferred.png"
TEST_TRAJECTORIES_FILENAME = "test_truth_vs_inferred.png"

INTEGRATED_RECOVERY_FILENAME = "integrated_recovery.csv"

REPORTING_INPUT_DIRECTORY = "reporting"
VALIDATION_INITIAL_CONDITIONS_FILENAME = (
    "validation_initial_conditions.csv"
)

SBML_VALIDATION_DIRECTORY = "sbml_validation"
DYNAMIC_FIDELITY_DIRECTORY = "dynamic_fidelity"
DYNAMIC_FIDELITY_COMPARISON_FILENAME = (
    "dynamic_fidelity_comparison.csv"
)
SBML_DYNAMIC_FIDELITY_SOURCE_FILENAME = (
    "original_vs_inferred_sbml_validation.png"
)


DISPLAY_SIGNIFICANT_DIGITS = 4

TRAINING_TRAJECTORIES = {
    "train_04": 4,
    "train_08": 8,
    "train_16": 16,
    "train_32": 32,
}


def require_file(
    path: Path,
) -> Path:
    """Require one reporting input file to exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required reporting input does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but found: {path}"
        )

    return path


def experiment_directory(
    *,
    model_id: str,
) -> Path:
    """Return the frozen exact-derivative experiment directory."""
    return (
        RESULTS_ROOT
        / model_id
        / EXPERIMENT_NAME
    )


def selected_system_path(
    *,
    model_id: str,
    dataset_name: str,
) -> Path:
    """Return the frozen selected-system path for one training size."""
    return (
        experiment_directory(
            model_id=model_id,
        )
        / dataset_name
        / SELECTION_DIRECTORY
        / SELECTED_SYSTEM_FILENAME
    )


def mechanistic_recovery_directory(
    *,
    model_id: str,
) -> Path:
    """Return the frozen mechanistic-recovery directory."""
    return (
        experiment_directory(
            model_id=model_id,
        )
        / MECHANISTIC_RECOVERY_DIRECTORY
    )


def report_directory(
    *,
    model_id: str,
) -> Path:
    """Return and create the report output directory."""
    directory = (
        experiment_directory(
            model_id=model_id,
        )
        / REPORT_DIRECTORY
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def report_figure_directory(
    *,
    model_id: str,
) -> Path:
    """Return the report figure directory."""
    return (
        report_directory(
            model_id=model_id,
        )
        / FIGURE_DIRECTORY
    )


def normalize_dataset_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common training-dataset column names."""
    frame = frame.copy()

    if (
        "dataset_name" not in frame.columns
        and "train_dataset" in frame.columns
    ):
        frame = frame.rename(
            columns={
                "train_dataset": "dataset_name",
            }
        )

    return frame


def normalize_state_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common state identifier column names."""
    frame = frame.copy()

    candidate_columns = (
        "state_id",
        "state",
        "rhs_state",
        "target_state",
    )

    if "state_id" in frame.columns:
        return frame

    for column in candidate_columns:
        if column in frame.columns:
            return frame.rename(
                columns={
                    column: "state_id",
                }
            )

    return frame


def read_csv(
    path: Path,
) -> pd.DataFrame:
    """Read one required CSV file."""
    require_file(
        path
    )

    return pd.read_csv(
        path
    )


def load_selected_systems(
    *,
    model_id: str,
) -> dict[str, pd.Series]:
    """Load one frozen selected system for every training-set size."""
    selected = {}

    for dataset_name in TRAIN_DATASETS:
        path = selected_system_path(
            model_id=model_id,
            dataset_name=dataset_name,
        )

        frame = read_csv(
            path
        )

        if len(frame) != 1:
            raise ValueError(
                f"{path} contains {len(frame)} rows; expected exactly one."
            )

        row = frame.iloc[
            0
        ]

        if (
            "model_id" in row.index
            and str(row["model_id"]) != model_id
        ):
            raise ValueError(
                f"Selected-system model mismatch in {path}."
            )

        if (
            "dataset_name" in row.index
            and str(row["dataset_name"]) != dataset_name
        ):
            raise ValueError(
                f"Selected-system dataset mismatch in {path}."
            )

        if (
            "selection_method" in row.index
            and str(row["selection_method"]) != "coordinate_search"
        ):
            raise ValueError(
                f"Unexpected selection method in {path}: "
                f"{row['selection_method']}."
            )

        selected[
            dataset_name
        ] = row

    return selected


def load_detailed_recovery(
    *,
    model_id: str,
    filename: str,
) -> pd.DataFrame:
    """Load one frozen equation-level mechanistic-recovery result family."""
    path = (
        mechanistic_recovery_directory(
            model_id=model_id,
        )
        / filename
    )

    frame = read_csv(
        path
    )

    frame = normalize_dataset_column(
        frame
    )

    frame = normalize_state_column(
        frame
    )

    if "model_id" in frame.columns:
        observed_models = set(
            frame[
                "model_id"
            ].astype(str)
        )

        if observed_models != {
            model_id
        }:
            raise ValueError(
                f"{filename} contains unexpected model IDs: "
                f"{sorted(observed_models)}."
            )

    return frame


def load_integrated_condition_results(
    *,
    model_id: str,
) -> pd.DataFrame:
    """Load the frozen integrated recovery rows for one model."""
    path = (
        RESULTS_ROOT
        / INTEGRATED_RECOVERY_FILENAME
    )

    frame = read_csv(
        path
    )

    frame = normalize_dataset_column(
        frame
    )

    if "model_id" not in frame.columns:
        raise ValueError(
            f"{path} does not contain model_id."
        )

    frame = frame[
        frame[
            "model_id"
        ].astype(str)
        == model_id
    ].copy()

    if len(frame) != len(
        TRAIN_DATASETS
    ):
        raise ValueError(
            f"Integrated recovery contains {len(frame)} rows for "
            f"{model_id}; expected {len(TRAIN_DATASETS)}."
        )

    return frame


def format_number(
    value,
    *,
    digits=DISPLAY_SIGNIFICANT_DIGITS,
) -> str:
    """Format one scalar compactly for a human-readable report."""
    if value is None:
        return "NA"

    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if math.isnan(
        numeric
    ):
        return "NA"

    if math.isinf(
        numeric
    ):
        return "FAIL" if numeric > 0 else "-INF"

    if numeric == 0.0:
        return "0"

    return f"{numeric:.{digits}f}"


def format_boolean(
    value,
) -> str:
    """Format one boolean-like value."""
    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
        }:
            return "yes"

        if normalized in {
            "false",
            "0",
            "no",
        }:
            return "no"

    if pd.isna(
        value
    ):
        return "NA"

    return (
        "yes"
        if bool(value)
        else "no"
    )


def format_match_symbol(
    value,
) -> str:
    """Format a boolean recovery criterion using a compact symbol."""
    if pd.isna(
        value
    ):
        return "NA"

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
        }:
            return "✓"

        if normalized in {
            "false",
            "0",
            "no",
        }:
            return "✗"

    return (
        "✓"
        if bool(value)
        else "✗"
    )


def escape_markdown_table(
    value,
) -> str:
    """Escape one value for use inside a Markdown table cell."""
    return (
        str(value)
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\n",
            " ",
        )
    )


def build_symbol_map(
    *,
    state_ids,
) -> dict[str, sp.Symbol]:
    """Build explicit SymPy symbols for all dynamic states."""
    return {
        state_id: sp.Symbol(
            state_id
        )
        for state_id in state_ids
    }


def normalize_expression_text(
    expression_text,
) -> str:
    """Normalize common textual operators before SymPy parsing."""
    text = str(
        expression_text
    ).strip()

    text = text.replace(
        "^",
        "**",
    )

    return text


def parse_report_expression(
    expression_text,
    *,
    state_ids,
):
    """Parse one equation using an explicit dynamic-state symbol namespace."""
    symbol_map = build_symbol_map(
        state_ids=state_ids,
    )

    normalized = normalize_expression_text(
        expression_text
    )

    return sp.sympify(
        normalized,
        locals=symbol_map,
    )


def round_numeric_atom(
    value: sp.Float,
):
    """Round one SymPy Float for display without changing stored results."""
    numeric = float(
        value
    )

    if numeric == 0.0:
        return sp.Float(
            0.0
        )

    rounded_text = (
        f"{numeric:.{DISPLAY_SIGNIFICANT_DIGITS}g}"
    )

    return sp.Float(
        rounded_text
    )


def display_expression(
    expression,
):
    """Return a display-only expression with compact numeric coefficients."""
    replacements = {
        atom: round_numeric_atom(
            atom
        )
        for atom in expression.atoms(
            sp.Float
        )
    }

    return expression.xreplace(
        replacements
    )


def expression_to_latex(
    expression_text,
    *,
    state_ids,
) -> str:
    """Convert one stored equation to compact LaTeX for reporting."""
    try:
        expression = parse_report_expression(
            expression_text,
            state_ids=state_ids,
        )

        expression = display_expression(
            expression
        )

        return sp.latex(
            expression
        )

    except Exception:
        escaped = (
            str(expression_text)
            .replace(
                "\\",
                "\\\\",
            )
            .replace(
                "_",
                "\\_",
            )
        )

        return (
            "\\text{"
            + escaped
            + "}"
        )


def selected_equation(
    *,
    selected_row: pd.Series,
    state_id: str,
):
    """Read one selected RHS equation from a frozen selected-system row."""
    column = (
        f"equation_{state_id}"
    )

    if column not in selected_row.index:
        raise KeyError(
            f"Selected system does not contain {column}."
        )

    return selected_row[
        column
    ]


def find_metric_row(
    *,
    frame: pd.DataFrame,
    dataset_name: str,
    state_id: str,
) -> pd.Series:
    """Find one equation-level recovery row."""
    if "dataset_name" not in frame.columns:
        raise KeyError(
            "Recovery table does not contain dataset_name."
        )

    if "state_id" not in frame.columns:
        raise KeyError(
            "Recovery table does not contain state_id."
        )

    subset = frame[
        (
            frame[
                "dataset_name"
            ].astype(str)
            == dataset_name
        )
        & (
            frame[
                "state_id"
            ].astype(str)
            == state_id
        )
    ]

    if len(subset) != 1:
        raise ValueError(
            "Expected exactly one recovery row for "
            f"{dataset_name}/{state_id}; found {len(subset)}."
        )

    return subset.iloc[
        0
    ]


def first_available(
    row: pd.Series,
    names,
):
    """Return the first available non-null field from a row."""
    for name in names:
        if (
            name in row.index
            and not pd.isna(
                row[name]
            )
        ):
            return row[
                name
            ]

    return None


def metric_text(
    row: pd.Series,
    names,
) -> str:
    """Format the first available numeric metric."""
    value = first_available(
        row,
        names,
    )

    return format_number(
        value
    )


def match_metric_text(
    row: pd.Series,
    names,
) -> str:
    """Format the first available boolean recovery metric."""
    value = first_available(
        row,
        names,
    )

    if value is None:
        return "NA"

    return format_match_symbol(
        value
    )


def integrated_row(
    *,
    integrated: pd.DataFrame,
    dataset_name: str,
) -> pd.Series:
    """Return one integrated system-level result row."""
    subset = integrated[
        integrated[
            "dataset_name"
        ].astype(str)
        == dataset_name
    ]

    if len(subset) != 1:
        raise ValueError(
            f"Expected one integrated row for {dataset_name}; "
            f"found {len(subset)}."
        )

    return subset.iloc[
        0
    ]


def reaction_rate_names(
    *,
    truth,
) -> dict[str, str]:
    """Assign compact v0, v1, ... labels to ground-truth reactions."""
    return {
        reaction.reaction_id: (
            f"v{index}"
        )
        for index, reaction in enumerate(
            truth.reactions
        )
    }


def reaction_balance_terms(
    *,
    truth,
    state_id: str,
    rate_names,
):
    """Return signed reaction-rate contributions for one state."""
    terms = []

    for reaction in truth.reactions:
        coefficient = reaction.stoichiometry.get(
            state_id,
            0.0,
        )

        try:
            coefficient = float(
                coefficient
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if coefficient == 0.0:
            continue

        terms.append(
            (
                coefficient,
                rate_names[
                    reaction.reaction_id
                ],
            )
        )

    return terms


def reaction_balance_latex(
    *,
    truth,
    state_id: str,
    rate_names,
) -> str:
    """Format the reaction-level balance for one state."""
    terms = reaction_balance_terms(
        truth=truth,
        state_id=state_id,
        rate_names=rate_names,
    )

    if not terms:
        return "0"

    pieces = []

    for index, (
        coefficient,
        rate_name,
    ) in enumerate(
        terms
    ):
        magnitude = abs(
            coefficient
        )

        if math.isclose(
            magnitude,
            1.0,
        ):
            body = rate_name
        else:
            body = (
                f"{format_number(magnitude)} "
                f"{rate_name}"
            )

        if index == 0:
            if coefficient < 0.0:
                pieces.append(
                    f"-{body}"
                )
            else:
                pieces.append(
                    body
                )
        else:
            operator = (
                "-"
                if coefficient < 0.0
                else "+"
            )

            pieces.append(
                f" {operator} {body}"
            )

    return "".join(
        pieces
    )


def overview_section(
    *,
    model_id: str,
    truth,
    integrated: pd.DataFrame,
) -> list[str]:
    """Build the report overview."""
    first_row = integrated.iloc[
        0
    ]

    comparability = first_available(
        first_row,
        (
            "mechanistic_comparability",
        ),
    )

    independent_variables = (
        truth.explicit_independent_variables
    )

    independent_text = (
        ", ".join(
            independent_variables
        )
        if independent_variables
        else "none"
    )

    lines = [
        f"# Model Recovery Report: `{model_id}`",
        "",
        "## 1. Overview",
        "",
        (
            "This report summarizes the frozen exact-derivative symbolic "
            "recovery experiment for this SBML model."
        ),
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Model | `{model_id}` |",
        f"| Dynamic states | {len(truth.state_ids)} |",
        f"| Reactions | {len(truth.reactions)} |",
        (
            "| Ground-truth autonomous | "
            f"{format_boolean(truth.is_autonomous)} |"
        ),
        (
            "| Explicit independent variables | "
            f"{escape_markdown_table(independent_text)} |"
        ),
        (
            "| Inference feature space | "
            f"{escape_markdown_table(first_available(first_row, ('inference_feature_space',)) or 'states_only')} |"
        ),
        (
            "| Mechanistic comparability | "
            f"{escape_markdown_table(comparability or 'NA')} |"
        ),
        "",
    ]

    return lines


def protocol_section() -> list[str]:
    """Build a compact description of the frozen inference protocol."""
    return [
        "## 2. Frozen inference protocol",
        "",
        (
            "The original SBML model was simulated under a frozen "
            "multidimensional initial-condition design. PySR received "
            "dynamic state values and exact RoadRunner derivatives."
        ),
        "",
        (
            "The training sets are nested: **4 ⊂ 8 ⊂ 16 ⊂ 32 "
            "trajectories**. Validation trajectories were used for "
            "coupled-system selection. Held-out test trajectories were "
            "not used during inference or selection."
        ),
        "",
        (
            "One symbolic regression problem was solved for each dynamic "
            "state, using all dynamic states as candidate input features. "
            "The selected equations were then evaluated as a coupled ODE "
            "system."
        ),
        "",
        (
            "All equations, selection decisions, validation results, "
            "held-out results, and mechanistic-recovery metrics shown "
            "below are read from frozen outputs."
        ),
        "",
    ]


def reaction_section(
    *,
    truth,
) -> list[str]:
    """Build the original SBML reaction-law section."""
    rate_names = reaction_rate_names(
        truth=truth,
    )

    lines = [
        "## 3. Original SBML model",
        "",
        "### Reaction rate laws",
        "",
        (
            "The following rate laws are reconstructed from the original "
            "SBML after expansion of supported user-defined functions and "
            "substitution of numeric constants."
        ),
        "",
    ]

    for reaction in truth.reactions:
        rate_name = rate_names[
            reaction.reaction_id
        ]

        latex = expression_to_latex(
            reaction.kinetic_law_expanded,
            state_ids=truth.state_ids,
        )

        lines.extend(
            [
                (
                    f"**{rate_name} — "
                    f"`{reaction.reaction_id}`**"
                ),
                "",
                "$$",
                f"{rate_name} = {latex}",
                "$$",
                "",
            ]
        )

    return lines


def ground_truth_equation_section(
    *,
    truth,
) -> list[str]:
    """Build the ground-truth ODE section."""
    rate_names = reaction_rate_names(
        truth=truth,
    )

    lines = [
        "### Ground-truth ODE system",
        "",
        (
            "For each state, the reaction-level balance is shown first, "
            "followed by the canonical total right-hand side used for "
            "mechanistic comparison."
        ),
        "",
    ]

    for state_id in truth.state_ids:
        balance = reaction_balance_latex(
            truth=truth,
            state_id=state_id,
            rate_names=rate_names,
        )

        rhs_latex = expression_to_latex(
            truth.rhs_by_state[
                state_id
            ],
            state_ids=truth.state_ids,
        )

        lines.extend(
            [
                f"#### State `{state_id}`",
                "",
                "**Reaction form**",
                "",
                "$$",
                (
                    f"\\frac{{d\\,{state_id}}}{{dt}} "
                    f"= {balance}"
                ),
                "$$",
                "",
                "**Canonical total RHS**",
                "",
                "$$",
                (
                    f"\\frac{{d\\,{state_id}}}{{dt}} "
                    f"= {rhs_latex}"
                ),
                "$$",
                "",
            ]
        )

    return lines


def validation_initial_conditions_path(
    *,
    model_id: str,
) -> Path:
    """Return the frozen validation initial-condition reporting table."""
    return (
        experiment_directory(
            model_id=model_id,
        )
        / REPORTING_INPUT_DIRECTORY
        / VALIDATION_INITIAL_CONDITIONS_FILENAME
    )


def load_validation_initial_conditions(
    *,
    model_id: str,
) -> pd.DataFrame:
    """Load the frozen validation initial conditions for reporting."""
    frame = read_csv(
        validation_initial_conditions_path(
            model_id=model_id,
        )
    )

    required_columns = {
        "trajectory_index",
        "trajectory_id",
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Validation initial-condition table is missing required "
            f"columns: {sorted(missing)}."
        )

    if len(frame) != 8:
        raise ValueError(
            "Validation initial-condition table contains "
            f"{len(frame)} rows; expected 8."
        )

    return frame.sort_values(
        "trajectory_index"
    ).reset_index(
        drop=True
    )


def validation_initial_conditions_section(
    *,
    validation_initial_conditions: pd.DataFrame,
    state_ids,
) -> list[str]:
    """Build the validation initial-condition table."""
    lines = [
        "### Validation initial conditions",
        "",
        (
            "The eight validation trajectories correspond to eight "
            "distinct initial conditions from the frozen experimental "
            "design. These are the conditions labeled `Val 000` through "
            "`Val 007` in the validation figures."
        ),
        "",
        (
            "These trajectories were used during coupled-system selection "
            "and therefore are validation data, not held-out test data."
        ),
        "",
    ]

    header = [
        "Validation trajectory",
    ]

    header.extend(
        f"{state_id}₀"
        for state_id in state_ids
    )

    lines.append(
        "| "
        + " | ".join(
            header
        )
        + " |"
    )

    lines.append(
        "|"
        + "|".join(
            [
                "---",
                *(
                    "---:"
                    for _ in state_ids
                ),
            ]
        )
        + "|"
    )

    for _, row in validation_initial_conditions.iterrows():
        trajectory_index = int(
            row[
                "trajectory_index"
            ]
        )

        values = []

        for state_id in state_ids:
            column = f"{state_id}_0"

            if column not in row.index:
                raise ValueError(
                    "Validation initial-condition table does not contain "
                    f"required state column {column}."
                )

            values.append(
                format_number(
                    row[
                        column
                    ],
                    digits=6,
                )
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    f"Val {trajectory_index:03d}",
                    *values,
                ]
            )
            + " |"
        )

    lines.append(
        ""
    )

    return lines


def dynamic_fidelity_comparison_path(
    *,
    model_id: str,
) -> Path:
    """Return the frozen aggregate SBML dynamic-fidelity table."""
    return (
        experiment_directory(
            model_id=model_id,
        )
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
        / DYNAMIC_FIDELITY_DIRECTORY
        / DYNAMIC_FIDELITY_COMPARISON_FILENAME
    )


def load_dynamic_fidelity_comparison(
    *,
    model_id: str,
) -> pd.DataFrame:
    """Load the frozen original-SBML versus inferred-SBML comparison."""
    frame = normalize_dataset_column(
        read_csv(
            dynamic_fidelity_comparison_path(
                model_id=model_id,
            )
        )
    )

    if "dataset_name" not in frame.columns:
        raise ValueError(
            "Dynamic-fidelity comparison does not contain dataset_name."
        )

    frame = frame[
        frame[
            "dataset_name"
        ].astype(str).isin(
            TRAIN_DATASETS
        )
    ].copy()

    if len(frame) != len(TRAIN_DATASETS):
        raise ValueError(
            "Dynamic-fidelity comparison contains "
            f"{len(frame)} training rows; expected {len(TRAIN_DATASETS)}."
        )

    return frame


def dynamic_fidelity_source_figure_path(
    *,
    model_id: str,
    dataset_name: str,
) -> Path:
    """Return one frozen SBML dynamic-fidelity figure."""
    return (
        experiment_directory(
            model_id=model_id,
        )
        / dataset_name
        / SELECTION_DIRECTORY
        / SBML_VALIDATION_DIRECTORY
        / DYNAMIC_FIDELITY_DIRECTORY
        / SBML_DYNAMIC_FIDELITY_SOURCE_FILENAME
    )


def dynamic_fidelity_report_filename(
    *,
    dataset_name: str,
) -> str:
    """Return the report-local filename for one SBML comparison figure."""
    return f"sbml_dynamic_fidelity_{dataset_name}.png"


def prepare_sbml_dynamic_fidelity_figures(
    *,
    model_id: str,
) -> None:
    """Copy frozen SBML dynamic-fidelity figures into the report directory."""
    destination_directory = report_figure_directory(
        model_id=model_id,
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_name in TRAIN_DATASETS:
        source = require_file(
            dynamic_fidelity_source_figure_path(
                model_id=model_id,
                dataset_name=dataset_name,
            )
        )

        destination = (
            destination_directory
            / dynamic_fidelity_report_filename(
                dataset_name=dataset_name,
            )
        )

        shutil.copy2(
            source,
            destination,
        )


def dynamic_fidelity_row(
    *,
    dynamic_fidelity: pd.DataFrame,
    dataset_name: str,
) -> pd.Series:
    """Return one training-size row from the dynamic-fidelity table."""
    subset = dynamic_fidelity[
        dynamic_fidelity[
            "dataset_name"
        ].astype(str)
        == dataset_name
    ]

    if len(subset) != 1:
        raise ValueError(
            "Expected exactly one dynamic-fidelity row for "
            f"{dataset_name}; found {len(subset)}."
        )

    return subset.iloc[0]


def completed_trajectory_text(
    row: pd.Series,
    names,
) -> str:
    """Format a completed-trajectory count as n/8 when available."""
    value = first_available(
        row,
        names,
    )

    if value is None:
        return "NA"

    try:
        return f"{int(float(value))}/8"
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def sbml_dynamic_fidelity_section(
    *,
    dynamic_fidelity: pd.DataFrame,
) -> list[str]:
    """Build the original-SBML versus inferred-SBML validation section."""
    lines = [
        "### Original SBML versus inferred SBML",
        "",
        (
            "This comparison evaluates dynamic fidelity after exporting the "
            "frozen selected inferred ODE system to SBML. Both the original "
            "biological SBML model and the inferred SBML model are simulated "
            "with RoadRunner/CVODE from the same frozen validation initial "
            "conditions and on the same time grid."
        ),
        "",
        (
            "This is distinct from mechanistic recovery: close trajectories "
            "do not imply that the original reaction mechanism or exact "
            "functional form was recovered. It is also distinct from SBML "
            "portability, which asks whether the exported inferred SBML "
            "reproduces the inferred ODE implementation itself."
        ),
        "",
        (
            "Validation data participated in Selector v1 system selection, "
            "so these results are not an unbiased held-out generalization "
            "estimate."
        ),
        "",
        (
            "| Train | Original completed | Inferred completed | "
            "Comparable | Status | Validation macro NRMSE | "
            "Mean state NRMSE | Mean state R² |"
        ),
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]

    for dataset_name in TRAIN_DATASETS:
        row = dynamic_fidelity_row(
            dynamic_fidelity=dynamic_fidelity,
            dataset_name=dataset_name,
        )

        train_count = TRAINING_TRAJECTORIES[
            dataset_name
        ]

        original_completed = completed_trajectory_text(
            row,
            (
                "original_trajectories_completed",
                "original_successes",
                "original_completed_trajectories",
                "original_completed",
                "n_original_completed",
            ),
        )

        inferred_completed = completed_trajectory_text(
            row,
            (
                "inferred_trajectories_completed",
                "inferred_successes",
                "inferred_completed_trajectories",
                "inferred_completed",
                "n_inferred_completed",
            ),
        )

        comparable = completed_trajectory_text(
            row,
            (
                "comparable_trajectories",
                "n_comparable_trajectories",
                "trajectories_compared",
            ),
        )

        status = str(
            first_available(
                row,
                (
                    "dynamic_fidelity_status",
                    "status",
                ),
            )
            or "NA"
        )

        validation_nrmse = format_number(
            first_available(
                row,
                (
                    "validation_macro_nrmse",
                    "macro_nrmse",
                ),
            ),
            digits=6,
        )

        mean_state_nrmse = format_number(
            first_available(
                row,
                (
                    "mean_state_nrmse",
                    "mean_nrmse",
                ),
            ),
            digits=6,
        )

        mean_state_r2 = format_number(
            first_available(
                row,
                (
                    "mean_state_r2",
                    "mean_r2",
                ),
            ),
            digits=6,
        )

        lines.append(
            (
                f"| {train_count} "
                f"| {original_completed} "
                f"| {inferred_completed} "
                f"| {comparable} "
                f"| {escape_markdown_table(status)} "
                f"| {validation_nrmse} "
                f"| {mean_state_nrmse} "
                f"| {mean_state_r2} |"
            )
        )

    lines.append("")

    for dataset_name in TRAIN_DATASETS:
        train_count = TRAINING_TRAJECTORIES[
            dataset_name
        ]

        filename = dynamic_fidelity_report_filename(
            dataset_name=dataset_name,
        )

        lines.extend(
            [
                f"#### {train_count} training trajectories",
                "",
                (
                    "Solid curves show the original SBML model; dashed "
                    "curves show the exported inferred SBML model. Each "
                    "column corresponds to one frozen validation initial "
                    "condition listed above."
                ),
                "",
                (
                    f"![Original versus inferred SBML: {dataset_name}]"
                    f"({FIGURE_DIRECTORY}/{filename})"
                ),
                "",
            ]
        )

    return lines


def trajectory_figure_section() -> list[str]:
    """Build the original-dynamics and training-data figure section."""
    return [
        "## 4. Original dynamics and training data",
        "",
        "### Nominal SBML trajectory",
        "",
        (
            "The nominal trajectory below is generated directly from the "
            "original SBML model using the curated simulation horizon."
        ),
        "",
        (
            f"![Nominal SBML trajectory]"
            f"({FIGURE_DIRECTORY}/{REFERENCE_TRAJECTORY_FILENAME})"
        ),
        "",
        "### Nested training trajectories",
        "",
        (
            "The symbolic-regression experiments use a frozen nested "
            "initial-condition design. The 4-trajectory training set is "
            "contained in the 8-trajectory set, which is contained in the "
            "16-trajectory set, which is contained in the "
            "32 training trajectories."
        ),
        "",
        (
            f"![Nested training trajectories]"
            f"({FIGURE_DIRECTORY}/{TRAINING_TRAJECTORIES_FILENAME})"
        ),
        "",
    ]


def dynamic_evaluation_section(
    *,
    dynamic_fidelity: pd.DataFrame,
) -> list[str]:
    """Build the validation, SBML-fidelity, and held-out evaluation section."""
    lines = [
        "## 7. Dynamic validation and held-out evaluation",
        "",
        "### Validation trajectories: original SBML versus inferred ODE",
        "",
        (
            "The figure below compares trajectories generated by the "
            "original SBML model with trajectories generated by each frozen "
            "selected inferred ODE system on the validation initial "
            "conditions. Solid lines represent the original SBML model and "
            "dashed lines represent the selected inferred ODE system."
        ),
        "",
        (
            "Validation trajectories were used during coupled-system "
            "selection. Therefore, validation performance measures selection "
            "performance rather than fully held-out generalization."
        ),
        "",
        (
            f"![Validation: original SBML versus selected inferred ODE]"
            f"({FIGURE_DIRECTORY}/{VALIDATION_TRAJECTORIES_FILENAME})"
        ),
        "",
    ]

    lines.extend(
        sbml_dynamic_fidelity_section(
            dynamic_fidelity=dynamic_fidelity,
        )
    )

    lines.extend(
        [
            "### Held-out test trajectories",
            "",
            (
                "The figure below shows the frozen held-out evaluation. "
                "These test trajectories were not used for symbolic "
                "regression or coupled-system selection. The figure is "
                "generated directly from the previously frozen test "
                "predictions and failure records; the inferred systems are "
                "not reintegrated for this report."
            ),
            "",
            (
                "A test condition is reported as failed when at least one "
                "of the eight held-out trajectories fails to complete under "
                "the frozen integration protocol. Successful trajectories "
                "remain visible, but a partial result is not reported as a "
                "successful held-out test."
            ),
            "",
            (
                f"![Held-out test: original SBML versus selected inferred ODE]"
                f"({FIGURE_DIRECTORY}/{TEST_TRAJECTORIES_FILENAME})"
            ),
            "",
        ]
    )

    return lines


def recovered_system_section(
    *,
    truth,
    selected_systems,
) -> list[str]:
    """Build the recovered coupled-system equations for all training sizes."""
    lines = [
        "## 5. Recovered ODE systems",
        "",
        (
            "Each system below is the frozen Selector v1 system selected "
            "after PySR candidate generation and validation-based "
            "coordinate search."
        ),
        "",
    ]

    for dataset_name in TRAIN_DATASETS:
        selected_row = selected_systems[
            dataset_name
        ]

        trajectory_count = (
            TRAINING_TRAJECTORIES[
                dataset_name
            ]
        )

        lines.extend(
            [
                (
                    f"### {trajectory_count} training "
                    "trajectories"
                ),
                "",
            ]
        )

        for state_id in truth.state_ids:
            equation = selected_equation(
                selected_row=selected_row,
                state_id=state_id,
            )

            latex = expression_to_latex(
                equation,
                state_ids=truth.state_ids,
            )

            lines.extend(
                [
                    "$$",
                    (
                        f"\\frac{{d\\,{state_id}}}{{dt}} "
                        f"= {latex}"
                    ),
                    "$$",
                    "",
                ]
            )

    return lines


def equation_metric_values(
    *,
    dependency_row: pd.Series,
    functional_row: pd.Series,
    algebraic_row: pd.Series,
) -> list[tuple[str, str]]:
    """Collect the frozen equation-level recovery metrics for one RHS."""
    return [
        (
            "Dependency F1",
            metric_text(
                dependency_row,
                (
                    "dependency_f1",
                    "f1",
                ),
            ),
        ),
        (
            "Numerator monomial F1",
            metric_text(
                functional_row,
                (
                    "numerator_monomial_f1",
                    "monomial_f1",
                ),
            ),
        ),
        (
            "Interaction F1",
            metric_text(
                functional_row,
                (
                    "interaction_f1",
                ),
            ),
        ),
        (
            "Rational structure",
            match_metric_text(
                functional_row,
                (
                    "rational_structure_match",
                    "rational_match",
                ),
            ),
        ),
        (
            "Denominator dependency F1",
            metric_text(
                functional_row,
                (
                    "denominator_dependency_f1",
                    "denominator_f1",
                ),
            ),
        ),
        (
            "Numerical equivalence",
            match_metric_text(
                algebraic_row,
                (
                    "numerical_equivalence_match",
                    "numerically_equivalent",
                    "numerical_equivalent",
                    "numerical_equivalence",
                ),
            ),
        ),
        (
            "Exact algebraic match",
            match_metric_text(
                algebraic_row,
                (
                    "exact_algebraic_match",
                    "exact_match",
                ),
            ),
        ),
    ]


def equation_comparison_section(
    *,
    truth,
    selected_systems,
    dependency: pd.DataFrame,
    functional: pd.DataFrame,
    algebraic: pd.DataFrame,
) -> list[str]:
    """Build equation-by-equation truth, inference, and recovery comparisons."""
    lines = [
        "## 6. Equation-by-equation recovery",
        "",
        (
            "This section compares each true SBML right-hand side with "
            "the frozen inferred equation at each training-set size."
        ),
        "",
    ]

    for state_id in truth.state_ids:
        true_latex = expression_to_latex(
            truth.rhs_by_state[
                state_id
            ],
            state_ids=truth.state_ids,
        )

        lines.extend(
            [
                f"### State `{state_id}`",
                "",
                "**Ground truth**",
                "",
                "$$",
                (
                    f"\\frac{{d\\,{state_id}}}{{dt}} "
                    f"= {true_latex}"
                ),
                "$$",
                "",
            ]
        )

        for dataset_name in TRAIN_DATASETS:
            trajectory_count = (
                TRAINING_TRAJECTORIES[
                    dataset_name
                ]
            )

            selected_row = selected_systems[
                dataset_name
            ]

            inferred_equation = selected_equation(
                selected_row=selected_row,
                state_id=state_id,
            )

            inferred_latex = expression_to_latex(
                inferred_equation,
                state_ids=truth.state_ids,
            )

            dependency_row = find_metric_row(
                frame=dependency,
                dataset_name=dataset_name,
                state_id=state_id,
            )

            functional_row = find_metric_row(
                frame=functional,
                dataset_name=dataset_name,
                state_id=state_id,
            )

            algebraic_row = find_metric_row(
                frame=algebraic,
                dataset_name=dataset_name,
                state_id=state_id,
            )

            metrics = equation_metric_values(
                dependency_row=dependency_row,
                functional_row=functional_row,
                algebraic_row=algebraic_row,
            )

            lines.extend(
                [
                    (
                        f"#### {trajectory_count} training "
                        "trajectories"
                    ),
                    "",
                    "$$",
                    (
                        f"\\frac{{d\\,{state_id}}}{{dt}} "
                        f"= {inferred_latex}"
                    ),
                    "$$",
                    "",
                    "| Recovery criterion | Result |",
                    "|---|---:|",
                ]
            )

            for label, value in metrics:
                lines.append(
                    (
                        f"| {escape_markdown_table(label)} "
                        f"| {escape_markdown_table(value)} |"
                    )
                )

            lines.append(
                ""
            )

    return lines


def system_metrics_section(
    *,
    integrated: pd.DataFrame,
) -> list[str]:
    """Build the system-level frozen recovery summary."""
    lines = [
        "## 8. System-level recovery summary",
        "",
        (
            "This table summarizes the recovery hierarchy for each "
            "training-set size. These metrics should be interpreted "
            "together rather than as interchangeable definitions of "
            "success."
        ),
        "",
        (
            "| Train | Validation NRMSE | Test status | Test NRMSE | "
            "Dependency F1 | Monomial F1 | Interaction F1 | "
            "Rational accuracy | Denominator F1 | Numerical eq. | "
            "Exact algebraic |"
        ),
        (
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for dataset_name in TRAIN_DATASETS:
        row = integrated_row(
            integrated=integrated,
            dataset_name=dataset_name,
        )

        train_count = (
            TRAINING_TRAJECTORIES[
                dataset_name
            ]
        )

        validation_nrmse = format_number(
            first_available(
                row,
                (
                    "validation_macro_nrmse",
                ),
            )
        )

        test_status = str(
            first_available(
                row,
                (
                    "test_status",
                    "held_out_test_status",
                ),
            )
            or "NA"
        )

        test_nrmse = format_number(
            first_available(
                row,
                (
                    "test_macro_nrmse",
                    "held_out_test_macro_nrmse",
                ),
            )
        )

        dependency_f1 = format_number(
            first_available(
                row,
                (
                    "micro_dependency_f1",
                ),
            )
        )

        monomial_f1 = format_number(
            first_available(
                row,
                (
                    "macro_numerator_monomial_f1",
                ),
            )
        )

        interaction_f1 = format_number(
            first_available(
                row,
                (
                    "macro_interaction_f1",
                ),
            )
        )

        rational_accuracy = format_number(
            first_available(
                row,
                (
                    "rational_structure_accuracy",
                ),
            )
        )

        denominator_f1 = format_number(
            first_available(
                row,
                (
                    "macro_denominator_dependency_f1",
                ),
            )
        )

        numerical_equivalent = format_number(
            first_available(
                row,
                (
                    "n_numerically_equivalent_rhs",
                    "numerically_equivalent_rhs",
                    "numerically_equivalent_rhs_count",
                    "numerical_equivalent_rhs",
                    "numerical_equivalent_rhs_count",
                    "numerically_equivalent_count",
                    "numerical_equivalent_count",
                ),
            )
        )

        exact_algebraic = format_number(
            first_available(
                row,
                (
                    "n_exact_algebraic_matches",
                    "exact_algebraic_rhs",
                    "exact_algebraic_rhs_count",
                    "exact_algebraic_matches",
                    "exact_algebraic_match_count",
                    "exact_algebraic_count",
                ),
            )
        )

        lines.append(
            (
                f"| {train_count} "
                f"| {validation_nrmse} "
                f"| {escape_markdown_table(test_status)} "
                f"| {test_nrmse} "
                f"| {dependency_f1} "
                f"| {monomial_f1} "
                f"| {interaction_f1} "
                f"| {rational_accuracy} "
                f"| {denominator_f1} "
                f"| {numerical_equivalent} "
                f"| {exact_algebraic} |"
            )
        )

    lines.append(
        ""
    )

    return lines


def recovery_hierarchy_section(
    *,
    integrated: pd.DataFrame,
) -> list[str]:
    """Build a compact progressively stricter recovery table."""
    lines = [
        "## 9. Recovery hierarchy",
        "",
        (
            "The same results can be read as progressively stricter "
            "levels of recovery."
        ),
        "",
        (
            "| Recovery level | Train 4 | Train 8 | "
            "Train 16 | Train 32 |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    rows = []

    held_out = []
    dependency = []
    monomial = []
    interaction = []
    rational = []
    denominator = []
    numerical = []
    exact = []

    for dataset_name in TRAIN_DATASETS:
        row = integrated_row(
            integrated=integrated,
            dataset_name=dataset_name,
        )

        status = str(
            first_available(
                row,
                (
                    "test_status",
                    "held_out_test_status",
                ),
            )
            or ""
        ).lower()

        held_out.append(
            "✓"
            if status == "success"
            else "✗"
        )

        dependency.append(
            format_number(
                first_available(
                    row,
                    (
                        "micro_dependency_f1",
                    ),
                )
            )
        )

        monomial.append(
            format_number(
                first_available(
                    row,
                    (
                        "macro_numerator_monomial_f1",
                    ),
                )
            )
        )

        interaction.append(
            format_number(
                first_available(
                    row,
                    (
                        "macro_interaction_f1",
                    ),
                )
            )
        )

        rational.append(
            format_number(
                first_available(
                    row,
                    (
                        "rational_structure_accuracy",
                    ),
                )
            )
        )

        denominator.append(
            format_number(
                first_available(
                    row,
                    (
                        "macro_denominator_dependency_f1",
                    ),
                )
            )
        )

        numerical.append(
            format_number(
                first_available(
                    row,
                    (
                        "n_numerically_equivalent_rhs",
                        "numerically_equivalent_rhs",
                        "numerical_equivalent_rhs",
                        "numerical_equivalent_count",
                    ),
                )
            )
        )

        exact.append(
            format_number(
                first_available(
                    row,
                    (
                        "n_exact_algebraic_matches",
                        "exact_algebraic_rhs",
                        "exact_algebraic_matches",
                        "exact_algebraic_count",
                    ),
                )
            )
        )

    rows.extend(
        [
            (
                "Held-out integration",
                held_out,
            ),
            (
                "Dependency F1",
                dependency,
            ),
            (
                "Numerator monomial F1",
                monomial,
            ),
            (
                "Interaction F1",
                interaction,
            ),
            (
                "Rational accuracy",
                rational,
            ),
            (
                "Denominator F1",
                denominator,
            ),
            (
                "Numerical equivalence",
                numerical,
            ),
            (
                "Exact algebraic recovery",
                exact,
            ),
        ]
    )

    for label, values in rows:
        lines.append(
            (
                f"| {label} "
                f"| {values[0]} "
                f"| {values[1]} "
                f"| {values[2]} "
                f"| {values[3]} |"
            )
        )

    lines.append(
        ""
    )

    return lines


def factual_summary_section(
    *,
    model_id: str,
    integrated: pd.DataFrame,
) -> list[str]:
    """Build a deterministic factual summary from frozen results."""
    successful_tests = 0

    dependency_values = []
    interaction_values = []

    numerical_conditions = 0
    exact_conditions = 0

    for dataset_name in TRAIN_DATASETS:
        row = integrated_row(
            integrated=integrated,
            dataset_name=dataset_name,
        )

        status = str(
            first_available(
                row,
                (
                    "test_status",
                    "held_out_test_status",
                ),
            )
            or ""
        ).lower()

        if status == "success":
            successful_tests += 1

        dependency_value = first_available(
            row,
            (
                "micro_dependency_f1",
            ),
        )

        if dependency_value is not None:
            dependency_values.append(
                (
                    float(
                        dependency_value
                    ),
                    TRAINING_TRAJECTORIES[
                        dataset_name
                    ],
                )
            )

        interaction_value = first_available(
            row,
            (
                "macro_interaction_f1",
            ),
        )

        if interaction_value is not None:
            interaction_values.append(
                (
                    float(
                        interaction_value
                    ),
                    TRAINING_TRAJECTORIES[
                        dataset_name
                    ],
                )
            )

        numerical_value = first_available(
            row,
            (
                "n_numerically_equivalent_rhs",
                "numerically_equivalent_rhs",
                "numerical_equivalent_rhs",
                "numerical_equivalent_count",
            ),
        )

        if (
            numerical_value is not None
            and float(
                numerical_value
            ) > 0.0
        ):
            numerical_conditions += 1

        exact_value = first_available(
            row,
            (
                "n_exact_algebraic_matches",
                "exact_algebraic_rhs",
                "exact_algebraic_matches",
                "exact_algebraic_count",
            ),
        )

        if (
            exact_value is not None
            and float(
                exact_value
            ) > 0.0
        ):
            exact_conditions += 1

    best_dependency = max(
        dependency_values,
        default=(
            float("nan"),
            0,
        ),
    )

    best_interaction = max(
        interaction_values,
        default=(
            float("nan"),
            0,
        ),
    )

    return [
        "## 10. Factual summary",
        "",
        (
            f"- Model `{model_id}` successfully integrated all held-out "
            f"trajectories for **{successful_tests}/{len(TRAIN_DATASETS)}** "
            "training-set sizes."
        ),
        (
            "- The highest system-level dependency F1 was "
            f"**{format_number(best_dependency[0])}** with "
            f"**{best_dependency[1]}** training trajectories."
        ),
        (
            "- The highest system-level interaction F1 was "
            f"**{format_number(best_interaction[0])}** with "
            f"**{best_interaction[1]}** training trajectories."
        ),
        (
            "- At least one RHS was numerically equivalent to its "
            "ground-truth counterpart in "
            f"**{numerical_conditions}/{len(TRAIN_DATASETS)}** "
            "training-set sizes."
        ),
        (
            "- At least one RHS was exactly algebraically recovered in "
            f"**{exact_conditions}/{len(TRAIN_DATASETS)}** "
            "training-set sizes."
        ),
        "",
        (
            "The equation-level sections above should be used to "
            "interpret these aggregate values. In particular, strong "
            "dependency or interaction recovery does not by itself imply "
            "recovery of the complete SBML functional form."
        ),
        "",
    ]


def provenance_section(
    *,
    model_id: str,
) -> list[str]:
    """Build the report provenance section."""
    return [
        "## 11. Report provenance",
        "",
        "This report reads the following frozen result families:",
        "",
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/train_*/"
            f"{SELECTION_DIRECTORY}/{SELECTED_SYSTEM_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/"
            f"{MECHANISTIC_RECOVERY_DIRECTORY}/{DEPENDENCY_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/"
            f"{MECHANISTIC_RECOVERY_DIRECTORY}/{FUNCTIONAL_FORM_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/"
            f"{MECHANISTIC_RECOVERY_DIRECTORY}/{ALGEBRAIC_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/"
            f"{REPORTING_INPUT_DIRECTORY}/"
            f"{VALIDATION_INITIAL_CONDITIONS_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/"
            f"{SELECTION_DIRECTORY}/{SBML_VALIDATION_DIRECTORY}/"
            f"{DYNAMIC_FIDELITY_DIRECTORY}/"
            f"{DYNAMIC_FIDELITY_COMPARISON_FILENAME}`"
        ),
        (
            f"- `{model_id}/{EXPERIMENT_NAME}/train_*/"
            f"{SELECTION_DIRECTORY}/{SBML_VALIDATION_DIRECTORY}/"
            f"{DYNAMIC_FIDELITY_DIRECTORY}/"
            f"{SBML_DYNAMIC_FIDELITY_SOURCE_FILENAME}`"
        ),
        f"- `results/{INTEGRATED_RECOVERY_FILENAME}`",
        "",
        (
            "Ground-truth equations are reconstructed from the original "
            "SBML using the already validated mechanistic-ground-truth "
            "extraction code."
        ),
        "",
        (
            "The nominal and training trajectory figures are reconstructed "
            "from the original SBML using the frozen experimental design "
            "and the same generic simulation utilities used by the "
            "benchmark. These simulations are performed solely for "
            "reporting and visualization."
        ),
        "",
        (
            "The validation initial-condition table is read from the frozen "
            "reporting table generated from the deterministic experimental "
            "design."
        ),
        "",
        (
            "The validation ODE figure reconstructs the frozen selected "
            "systems on the frozen validation initial conditions using the "
            "same ODE integration implementation and tolerances as the "
            "validation pipeline. Validation is shown for visualization "
            "only and is not re-used to change the frozen selection."
        ),
        "",
        (
            "The original-SBML versus inferred-SBML dynamic-fidelity table "
            "and figures are read from frozen RoadRunner/CVODE comparison "
            "outputs. The report copies the existing PNG files into its "
            "figure directory; it does not rerun either SBML model."
        ),
        "",
        (
            "The held-out test figure is read directly from the frozen "
            "`test_predictions.csv`, `test_summary.csv`, and "
            "`test_failures.csv` outputs. Held-out test systems are not "
            "reintegrated by this reporting script."
        ),
        "",
        (
            "No PySR inference, Selector v1 search, held-out model "
            "selection, SBML dynamic-fidelity simulation, or scientific "
            "metric is recomputed or redefined by this reporting script."
        ),
        "",
    ]


def validate_report_figures(
    *,
    model_id: str,
) -> None:
    """Require every trajectory figure referenced by the report."""
    figure_directory = report_figure_directory(
        model_id=model_id,
    )

    required_figures = (
        REFERENCE_TRAJECTORY_FILENAME,
        TRAINING_TRAJECTORIES_FILENAME,
        VALIDATION_TRAJECTORIES_FILENAME,
        TEST_TRAJECTORIES_FILENAME,
        *(
            dynamic_fidelity_report_filename(
                dataset_name=dataset_name,
            )
            for dataset_name in TRAIN_DATASETS
        ),
    )

    for filename in required_figures:
        require_file(
            figure_directory
            / filename
        )


def build_report(
    *,
    model_id: str,
) -> Path:
    """Build the complete human-readable report for one frozen model."""
    model_path = resolve_model_path(
        model_id
    )

    truth = extract_mechanistic_ground_truth(
        model_id=model_id,
        model_path=model_path,
    )

    selected_systems = load_selected_systems(
        model_id=model_id,
    )

    dependency = load_detailed_recovery(
        model_id=model_id,
        filename=DEPENDENCY_FILENAME,
    )

    functional = load_detailed_recovery(
        model_id=model_id,
        filename=FUNCTIONAL_FORM_FILENAME,
    )

    algebraic = load_detailed_recovery(
        model_id=model_id,
        filename=ALGEBRAIC_FILENAME,
    )

    integrated = load_integrated_condition_results(
        model_id=model_id,
    )

    validation_initial_conditions = (
        load_validation_initial_conditions(
            model_id=model_id,
        )
    )

    dynamic_fidelity = load_dynamic_fidelity_comparison(
        model_id=model_id,
    )

    generate_trajectory_figures(
        model_id=model_id,
    )

    prepare_sbml_dynamic_fidelity_figures(
        model_id=model_id,
    )

    validate_report_figures(
        model_id=model_id,
    )

    lines = []

    lines.extend(
        overview_section(
            model_id=model_id,
            truth=truth,
            integrated=integrated,
        )
    )

    lines.extend(
        protocol_section()
    )

    lines.extend(
        reaction_section(
            truth=truth,
        )
    )

    lines.extend(
        ground_truth_equation_section(
            truth=truth,
        )
    )

    lines.extend(
        trajectory_figure_section()
    )

    lines.extend(
        validation_initial_conditions_section(
            validation_initial_conditions=(
                validation_initial_conditions
            ),
            state_ids=truth.state_ids,
        )
    )

    lines.extend(
        recovered_system_section(
            truth=truth,
            selected_systems=selected_systems,
        )
    )

    lines.extend(
        equation_comparison_section(
            truth=truth,
            selected_systems=selected_systems,
            dependency=dependency,
            functional=functional,
            algebraic=algebraic,
        )
    )

    lines.extend(
        dynamic_evaluation_section(
            dynamic_fidelity=dynamic_fidelity,
        )
    )

    lines.extend(
        system_metrics_section(
            integrated=integrated,
        )
    )

    lines.extend(
        recovery_hierarchy_section(
            integrated=integrated,
        )
    )

    lines.extend(
        factual_summary_section(
            model_id=model_id,
            integrated=integrated,
        )
    )

    lines.extend(
        provenance_section(
            model_id=model_id,
        )
    )

    output_directory = report_directory(
        model_id=model_id,
    )

    output_path = (
        output_directory
        / REPORT_FILENAME
    )

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return output_path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a human-readable report from frozen "
            "symbolic-recovery results."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels model identifier.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    output_path = build_report(
        model_id=arguments.model_id,
    )

    print()
    print("=" * 70)
    print("MODEL RECOVERY REPORT")
    print("=" * 70)

    print(
        "Model:",
        arguments.model_id,
    )

    print(
        "Report:",
        output_path,
    )

    print(
        "Figures:",
        report_figure_directory(
            model_id=arguments.model_id,
        ),
    )

    print()
    print(
        "Frozen scientific results were read but not modified."
    )


if __name__ == "__main__":
    main()