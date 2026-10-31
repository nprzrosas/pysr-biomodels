"""Render a navigable static HTML report for human readers."""

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; color: #17202a; }
main { max-width: 1180px; margin: auto; padding: 2rem; }
h1, h2, h3 { color: #17324d; }
.muted { color: #5f6b76; }
.notice { background: #fff4d6; border-left: 5px solid #e9a23b; padding: 1rem; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0; }
.card { border: 1px solid #d8dee4; border-radius: 8px; padding: 1rem;
        min-width: 150px; background: #f8fafc; }
.card strong { display: block; font-size: 1.7rem; color: #0f766e; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border-bottom: 1px solid #d8dee4; padding: 0.55rem; text-align: left;
         vertical-align: top; }
th { background: #edf2f7; position: sticky; top: 0; }
tr:hover { background: #f8fafc; }
img.figure { max-width: 100%; height: auto; border: 1px solid #d8dee4; }
.figure-block { margin: 1.5rem 0 2.5rem; }
.figure-block img { cursor: zoom-in; }
code { background: #eef2f5; padding: 0.1rem 0.25rem; border-radius: 3px; }
.equation { font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
.status-pretest_complete { color: #087f5b; font-weight: 700; }
.status-candidate_failed, .status-candidate_partial { color: #c92a2a; font-weight: 700; }
.status-selector_partial, .status-selector_interrupted { color: #b35c00; font-weight: 700; }
details { margin: 0.8rem 0; }
nav a { margin-right: 1rem; }
.success { color: #087f5b; font-weight: 700; }
.failure { color: #c92a2a; font-weight: 700; }
.unknown { color: #6b7280; font-weight: 700; }
.stage { border-top: 3px solid #d8dee4; margin-top: 2rem; padding-top: 0.6rem; }
.compact { font-size: 0.78rem; }
.interpretation { background: #edf7f6; border-left: 5px solid #2a9d8f;
                  padding: 1rem; }
.model-section { break-before: page; }
@media print {
  main { max-width: none; padding: 0; }
  body { font-size: 9pt; }
  table { font-size: 7pt; }
  th { position: static; }
  details > * { display: block !important; }
  .figure-block { break-before: page; break-inside: avoid; }
  .model-section { break-before: page; }
  .raw-data { display: none; }
  a { color: inherit; text-decoration: none; }
}
"""


def _value(value: Any, digits: int = 4) -> str:
    """Format a scalar for HTML display.

    Args:
        value: Scalar value.
        digits: Significant digits for floating values.

    Returns:
        Escaped display text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return html.escape(str(value))


def _page(title: str, body: str, navigation: str = "") -> str:
    """Wrap report content in a standalone HTML document.

    Args:
        title: Browser and page title.
        body: HTML body fragment.
        navigation: Optional navigation fragment.

    Returns:
        Complete HTML document.
    """
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title><style>{STYLE}</style></head>
<body><main><nav>{navigation}</nav>{body}</main></body></html>
"""


def _model_narrative(
    model: pd.Series,
    conditions: pd.DataFrame,
    mechanistic: pd.DataFrame,
) -> str:
    """Create a factual English summary for one model.

    Args:
        model: Model inventory row.
        conditions: This model's four expected conditions.
        mechanistic: Available mechanistic summaries.

    Returns:
        Human-readable paragraph.
    """
    selected = int(conditions["selection_present"].sum())
    successful = int(conditions["validation_status"].eq("success").sum())
    text = (
        f"The model has {int(model['n_states'])} dynamic states. "
        f"Selected systems were available for {selected} of 4 training sizes; "
        f"{successful} completed all eight validation trajectories without "
        "integration failures."
    )
    if not mechanistic.empty:
        comparable = mechanistic["mechanistic_comparability"].iloc[0]
        dependency = pd.to_numeric(
            mechanistic["macro_dependency_f1"], errors="coerce"
        ).mean()
        text += (
            f" Mechanistic comparability is {comparable}; mean dependency F1 "
            f"across evaluated conditions is {dependency:.3f}."
        )
        if "exact_algebraic_fraction" in mechanistic.columns:
            exact = pd.to_numeric(
                mechanistic["exact_algebraic_fraction"], errors="coerce"
            ).mean()
            text += f" The mean algebraically exact RHS fraction is {exact:.3f}."
    return text


def _match_text(value: Any) -> str:
    """Format a nullable mechanistic-match result.

    Args:
        value: Boolean-like match value.

    Returns:
        Human-readable match label.
    """
    if pd.isna(value):
        return "Not evaluated"
    return "Yes" if bool(value) else "No"


def _condition_outcome(condition: pd.Series) -> tuple[str, str]:
    """Describe inference and integration status for one training size.

    Args:
        condition: Condition inventory row.

    Returns:
        CSS class and human-readable status.
    """
    if not bool(condition["selection_present"]):
        return "unknown", "No selected inferred system"
    successful = pd.to_numeric(
        pd.Series([condition.get("successful_trajectories")]), errors="coerce"
    ).iloc[0]
    failed = pd.to_numeric(
        pd.Series([condition.get("failed_trajectories")]), errors="coerce"
    ).iloc[0]
    if str(condition.get("validation_status")) == "success":
        return "success", "Dynamic validation succeeded (8/8 trajectories)"
    successful_text = "?" if pd.isna(successful) else str(int(successful))
    failed_text = "?" if pd.isna(failed) else str(int(failed))
    return (
        "failure",
        (
            "Dynamic validation failed "
            f"({successful_text} successful, {failed_text} failed)"
        ),
    )


def _initial_condition_table(frame: pd.DataFrame) -> str:
    """Render an exact initial-condition table.

    Args:
        frame: Initial conditions with stable trajectory identifiers.

    Returns:
        HTML table fragment.
    """
    state_columns = [column for column in frame.columns if column.endswith("_0")]
    headers = "".join(
        f"<th>{html.escape(column.removesuffix('_0'))}<sub>0</sub></th>"
        for column in state_columns
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['trajectory_id']))}</td>"
        + "".join(
            f"<td>{_value(row[column], digits=6)}</td>" for column in state_columns
        )
        + "</tr>"
        for row in frame.to_dict(orient="records")
    )
    return f"""<div style="overflow-x:auto"><table class="compact"><thead><tr>
<th>Trajectory</th>{headers}</tr></thead><tbody>{rows}</tbody></table></div>"""


def _equation_stage_sections(
    conditions: pd.DataFrame,
    equations: pd.DataFrame,
    mechanistic_detail: pd.DataFrame,
) -> str:
    """Render original and inferred equations for every training size.

    Args:
        conditions: Four condition rows for one model.
        equations: Frozen selected equations.
        mechanistic_detail: Available original-equation comparisons.

    Returns:
        Four human-readable stage sections.
    """
    original_rows = (
        mechanistic_detail[["state_id", "true_equation"]].drop_duplicates("state_id")
        if not mechanistic_detail.empty
        else pd.DataFrame(columns=["state_id", "true_equation"])
    )
    original_by_state = {
        str(row["state_id"]): row["true_equation"]
        for row in original_rows.to_dict(orient="records")
    }
    sections = []
    if mechanistic_detail.empty:
        sections.append(
            "<div class='notice'><strong>Original-equation comparison "
            "unavailable.</strong> The post-hoc SBML ground-truth extraction "
            "did not produce evaluable RHS equations for this model. Inferred "
            "equations and dynamic outcomes are still reported when available."
            "</div>"
        )
    for condition in conditions.to_dict(orient="records"):
        dataset_name = str(condition["dataset_name"])
        condition_series = pd.Series(condition)
        css_class, outcome = _condition_outcome(condition_series)
        stage_equations = equations.loc[equations["dataset_name"] == dataset_name]
        stage_mechanistic = (
            mechanistic_detail.loc[mechanistic_detail["dataset_name"] == dataset_name]
            if not mechanistic_detail.empty
            else pd.DataFrame()
        )
        state_ids = list(original_by_state)
        state_ids.extend(
            state
            for state in stage_equations["state_id"].astype(str).tolist()
            if state not in state_ids
        )
        rows = []
        for state_id in state_ids:
            inferred_rows = stage_equations.loc[
                stage_equations["state_id"].astype(str) == state_id
            ]
            inferred = (
                inferred_rows.iloc[0]["equation"] if len(inferred_rows) else np.nan
            )
            comparison = (
                stage_mechanistic.loc[
                    stage_mechanistic["state_id"].astype(str) == state_id
                ]
                if "state_id" in stage_mechanistic.columns
                else pd.DataFrame()
            )
            comparison_row = comparison.iloc[0] if len(comparison) else pd.Series()
            rows.append(
                f"<tr><td>{html.escape(state_id)}</td>"
                f"<td class='equation'>{_value(original_by_state.get(state_id))}</td>"
                f"<td class='equation'>{_value(inferred)}</td>"
                f"<td>{_match_text(comparison_row.get('exact_algebraic_match', np.nan))}</td>"
                f"<td>{_match_text(comparison_row.get('numerical_equivalence_match', np.nan))}</td>"
                f"<td>{_value(comparison_row.get('dependency_f1', np.nan))}</td></tr>"
            )
        if rows:
            equation_table = f"""<div style="overflow-x:auto"><table><thead><tr>
<th>State</th><th>Original SBML RHS</th><th>Inferred RHS</th>
<th>Exact</th><th>Numerically equivalent</th><th>Dependency F1</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""
        else:
            equation_table = (
                "<p>No inferred equations were produced for this training size.</p>"
            )
        nrmse = _value(condition.get("validation_macro_nrmse"))
        sections.append(
            f"""<section class="stage"><h3>{html.escape(dataset_name)}: """
            f"{int(condition['train_trajectories'])} training trajectories</h3>"
            f'<p class="{css_class}">{html.escape(outcome)}</p>'
            f"<p>Selected-system validation macro NRMSE: <code>{nrmse}</code>. "
            "Validation was used by the selector; it is not held-out evidence.</p>"
            f"{equation_table}</section>"
        )
    return "".join(sections)


def _inventory_training_cell(condition: pd.Series) -> str:
    """Render a compact training-stage outcome for the cohort index.

    Args:
        condition: One model-training condition.

    Returns:
        Compact HTML status cell.
    """
    css_class, _outcome = _condition_outcome(condition)
    if not bool(condition["selection_present"]):
        return f'<span class="{css_class}">No selection</span>'
    nrmse = _value(condition.get("validation_macro_nrmse"))
    short_outcome = "8/8 integrated" if css_class == "success" else "Integration failed"
    return (
        f'<span class="{css_class}">{short_outcome}</span><br>'
        f'<span class="muted">NRMSE {nrmse}</span>'
    )


def write_model_page(
    output_root: Path,
    model: pd.Series,
    conditions: pd.DataFrame,
    equations: pd.DataFrame,
    mechanistic_summary: pd.DataFrame,
    mechanistic_detail: pd.DataFrame,
    representative_states: list[str],
    validation_figure_datasets: list[str],
    training_initial_conditions: pd.DataFrame,
    validation_initial_conditions: pd.DataFrame,
) -> None:
    """Write one model-specific report page.

    Args:
        output_root: Report destination root.
        model: Model inventory row.
        conditions: Model condition rows.
        equations: Model selected equations.
        mechanistic_summary: Model mechanistic condition summaries.
        mechanistic_detail: Model per-RHS mechanistic records.
        representative_states: States displayed in the trajectory figure.
        validation_figure_datasets: Training sizes with validation figures.
        training_initial_conditions: Frozen nested training conditions.
        validation_initial_conditions: Frozen validation conditions.
    """
    model_id = str(model["model_id"])
    model_root = output_root / "models" / model_id
    model_root.mkdir(parents=True, exist_ok=True)
    condition_rows = "".join(
        f"<tr><td>{_value(row['dataset_name'])}</td>"
        f"<td>{_value(row['condition_status'])}</td>"
        f"<td>{_value(row['validation_macro_nrmse'])}</td>"
        f"<td>{_value(row['successful_trajectories'])}</td>"
        f"<td>{_value(row['failed_trajectories'])}</td>"
        f"<td>{_value(row['systems_evaluated'])}</td></tr>"
        for row in conditions.to_dict(orient="records")
    )

    trajectory_section = "<p>No selected validation predictions were available.</p>"
    if representative_states and validation_figure_datasets:
        states = ", ".join(html.escape(value) for value in representative_states)
        figure_blocks = "".join(
            f"""<div class="figure-block"><h3>{html.escape(dataset_name)}</h3>
<a href="figures/validation_{html.escape(dataset_name)}.png" target="_blank">
<img class="figure" src="figures/validation_{html.escape(dataset_name)}.png"
alt="Original and inferred validation trajectories for {html.escape(dataset_name)}"></a></div>"""
            for dataset_name in validation_figure_datasets
        )
        trajectory_section = f"""
<p>Rows are dynamic states and columns are the eight validation initial
conditions. Blue lines show the original simulated dynamics; dashed orange
lines show the frozen selected inferred ODE. Panel NRMSE values belong to that
state and validation trajectory. Click an image to open it at full size.</p>
<p>States shown: {states}.</p>{figure_blocks}"""

    initial_condition_section = (
        "<p>Initial-condition reconstruction was unavailable for this model.</p>"
    )
    if (
        not training_initial_conditions.empty
        and not validation_initial_conditions.empty
    ):
        validation_table = _initial_condition_table(validation_initial_conditions)
        training_table = _initial_condition_table(training_initial_conditions)
        initial_condition_section = f"""
<p>The training design contains 32 ordered initial conditions. The experiments
are nested: <code>train_04</code> uses rows 0-3, <code>train_08</code> uses rows
0-7, <code>train_16</code> uses rows 0-15, and <code>train_32</code> uses all
rows 0-31. Validation uses eight separate conditions, labeled
<code>Val 000</code> through <code>Val 007</code> in the trajectory figures.</p>
<p>The heatmap reports log10 fold change from each state's nominal initial
value. Horizontal lines mark the nested training boundaries.</p>
<a href="figures/initial_condition_design.png" target="_blank">
<img class="figure" src="figures/initial_condition_design.png"
alt="Training and validation initial-condition design"></a>
<p><a href="training_initial_conditions.csv">Download exact training initial conditions</a> |
<a href="validation_initial_conditions.csv">Download exact validation initial conditions</a></p>
<details class="raw-data"><summary>Show the 8 exact validation initial conditions</summary>
{validation_table}</details>
<details class="raw-data"><summary>Show the 32 exact nested training initial conditions</summary>
{training_table}</details>"""

    equation_stages = _equation_stage_sections(
        conditions,
        equations,
        mechanistic_detail,
    )

    status = str(model["pipeline_status"])
    body = f"""
<h1>{html.escape(model_id)}</h1>
<p><a href="../../index.html">&lt;- Back to the cohort report</a></p>
<p class="status-{html.escape(status)}">Pipeline status: {html.escape(status)}</p>
<p>{html.escape(_model_narrative(model, conditions, mechanistic_summary))}</p>
<div class="cards"><div class="card"><span>States</span><strong>{int(model["n_states"])}</strong></div>
<div class="card"><span>Selections</span><strong>{int(model["selected_conditions"])}/4</strong></div>
<div class="card"><span>Successful validations</span><strong>{int(model["validation_success_conditions"])}/4</strong></div></div>
<h2>Experimental conditions</h2>
<table><thead><tr><th>Dataset</th><th>Status</th><th>NRMSE</th><th>Successful trajectories</th>
<th>Failed</th><th>Systems evaluated</th></tr></thead><tbody>{condition_rows}</tbody></table>
<h2>Initial conditions and nested training design</h2>{initial_condition_section}
<h2>Original and inferred equations at each training size</h2>
<div class="interpretation"><strong>How success is defined here.</strong>
Dynamic success means that the selected coupled ODE integrated all eight
validation trajectories. It does not mean that the original SBML equations
were recovered. The Exact, Numerically equivalent, and Dependency F1 columns
evaluate mechanistic recovery separately.</div>
{equation_stages}
<h2>Validation trajectories at each training size</h2>{trajectory_section}
"""
    (model_root / "index.html").write_text(
        _page(f"{model_id} -- PySR report", body), encoding="utf-8"
    )


def write_index_page(
    output_root: Path,
    model_frame: pd.DataFrame,
    condition_frame: pd.DataFrame,
    mechanistic_summary: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    """Write the broad-cohort landing page.

    Args:
        output_root: Report destination root.
        model_frame: Model pipeline inventory.
        condition_frame: Complete condition inventory.
        mechanistic_summary: Mechanistic condition summaries.
        provenance: Snapshot provenance fields.
    """
    requested = len(model_frame)
    candidates = int(model_frame["candidate_complete"].sum())
    complete = int(model_frame["pipeline_status"].eq("pretest_complete").sum())
    selected = int(condition_frame["selection_present"].sum())
    validation_success = int(condition_frame["validation_status"].eq("success").sum())
    model_rows = []
    for row in model_frame.to_dict(orient="records"):
        model_id = str(row["model_id"])
        model_conditions = condition_frame.loc[
            condition_frame["model_id"] == model_id
        ].set_index("dataset_name")
        stage_cells = "".join(
            f"<td>{_inventory_training_cell(model_conditions.loc[dataset])}</td>"
            for dataset in ("train_04", "train_08", "train_16", "train_32")
        )
        model_rows.append(
            f"<tr><td><a href='models/{html.escape(model_id)}/index.html'>"
            f"{html.escape(model_id)}</a></td>"
            f"<td>{int(row['n_states'])}</td>"
            f"<td class='status-{html.escape(str(row['pipeline_status']))}'>"
            f"{html.escape(str(row['pipeline_status']))}</td>"
            f"{stage_cells}</tr>"
        )
    model_rows_html = "".join(model_rows)
    mechanism_text = (
        f"{len(mechanistic_summary)} conditions were evaluated against original "
        "equations reconstructed from SBML."
        if not mechanistic_summary.empty
        else "Mechanistic comparison was not run."
    )
    mechanistic_figure = (
        """
<h2>Mechanistic recovery</h2>
<p>Algebraic and numerical fractions use only right-hand sides whose comparison
completed; timeouts are reported as not evaluated. The primary figure excludes
models whose ground truth requires variables that PySR did not receive.</p>
<img class="figure" src="figures/mechanistic_recovery.png" alt="Mechanistic recovery">
"""
        if not mechanistic_summary.empty
        else ""
    )
    body = f"""
<h1>Broad PySR BioModels report -- 2026-10-21</h1>
<p class="muted">Coverage, scalability, failures, validation, and mechanistic recovery.</p>
<div class="notice"><strong>Held-out data remain closed.</strong> This report uses training,
validation, logs, and original SBML equations. It neither materializes nor inspects test trajectories.</div>
<div class="cards">
<div class="card"><span>Models attempted</span><strong>{requested}</strong></div>
<div class="card"><span>Complete candidates</span><strong>{candidates}</strong></div>
<div class="card"><span>Complete selection</span><strong>{complete}</strong></div>
<div class="card"><span>Selected conditions</span><strong>{selected}/{len(condition_frame)}</strong></div>
<div class="card"><span>Successful validation</span><strong>{validation_success}/{selected}</strong></div>
</div>
<h2>How to read this report</h2>
<p>The benchmark asks whether symbolic regression can reconstruct a coupled ODE
system from simulated state values and exact derivatives. For every dynamic
state, PySR generates candidate right-hand sides. A coordinate selector then
chooses one candidate per state using eight validation trajectories.</p>
<p>Training sets are nested: the same first 4 trajectories are contained in the
8-, 16-, and 32-trajectory sets. Each model page shows the exact development
initial conditions, original SBML equations, inferred equations at all four
training sizes, mechanistic comparisons, and the eight validation time courses.
{html.escape(mechanism_text)}</p>
<div class="interpretation"><strong>Important distinction.</strong> A dynamically
successful condition integrated all eight validation trajectories. This does
not imply recovery of the original equations. Algebraic equality, numerical
equivalence, and dependency recovery are reported separately. Validation was
used for system selection, so it is not held-out generalization evidence.</div>
<p><a href="print_report.html"><strong>Open the complete printable HTML report</strong></a> |
<a href="biomodels41_20261021_report.pdf"><strong>Download the complete PDF</strong></a></p>
<h2>Coverage</h2><img class="figure" src="figures/coverage.png" alt="Coverage">
<h2>Observed scalability</h2><img class="figure" src="figures/scalability.png" alt="Scalability">
<p class="muted">This figure describes a single run with 60 workers; it is not a
controlled parallel-speedup curve.</p>
<h2>Validation</h2><img class="figure" src="figures/validation_heatmap.png" alt="Validation NRMSE">
{mechanistic_figure}
<h2>Model inventory</h2><div style="overflow-x:auto"><table><thead><tr>
<th>Model</th><th>States</th><th>Pipeline status</th><th>train_04</th>
<th>train_08</th><th>train_16</th><th>train_32</th>
</tr></thead><tbody>{model_rows_html}</tbody></table></div>
<h2>Downloadable data</h2><ul>
<li><a href="data/model_pipeline_status.csv">Model-level status</a></li>
<li><a href="data/model_training_condition_summary.csv">Condition-level summary</a></li>
<li><a href="data/selected_equations.csv">Selected equations</a></li>
<li><a href="data/failure_inventory.csv">Failure inventory</a></li>
<li><a href="data/mechanistic_recovery.csv">Per-RHS mechanistic recovery</a></li>
<li><a href="data/mechanistic_errors.csv">Mechanistic-analysis limitations</a></li>
<li><a href="data/sbml_input_audit.csv">SBML SHA-256 audit</a></li>
</ul>
<h2>Provenance</h2><p>Archive SHA-256: <code>{html.escape(str(provenance.get("archive_sha256", "not provided")))}</code>.<br>
Run ID: <code>{html.escape(str(provenance.get("run_id", "unknown")))}</code>.</p>
"""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.html").write_text(
        _page("Broad PySR BioModels report", body), encoding="utf-8"
    )


def _page_body(path: Path) -> str:
    """Extract the report body fragment from one generated HTML page.

    Args:
        path: Standalone report page.

    Returns:
        Content following navigation and preceding the main closing tag.
    """
    document = path.read_text(encoding="utf-8")
    start = document.index("</nav>") + len("</nav>")
    end = document.index("</main></body>")
    return document[start:end]


def write_print_report(output_root: Path, model_ids: tuple[str, ...]) -> Path:
    """Combine the landing page and model pages into printable HTML.

    Args:
        output_root: Generated report root.
        model_ids: Ordered cohort identifiers.

    Returns:
        Path to the complete printable HTML report.
    """
    overview = _page_body(output_root / "index.html")
    model_sections = []
    for model_id in model_ids:
        model_page = output_root / "models" / model_id / "index.html"
        body = _page_body(model_page)
        body = body.replace(
            '<p><a href="../../index.html">&lt;- Back to the cohort report</a></p>',
            "",
        )
        prefix = f"models/{model_id}/"
        body = body.replace('href="figures/', f'href="{prefix}figures/')
        body = body.replace('src="figures/', f'src="{prefix}figures/')
        body = body.replace(
            'href="training_initial_conditions.csv"',
            f'href="{prefix}training_initial_conditions.csv"',
        )
        body = body.replace(
            'href="validation_initial_conditions.csv"',
            f'href="{prefix}validation_initial_conditions.csv"',
        )
        body = body.replace("<details>", "<details open>")
        model_sections.append(f'<section class="model-section">{body}</section>')

    complete_body = (
        overview + "<section class='model-section'><h1>Per-model inference details</h1>"
        "<p>The following sections preserve the cohort order and compare all "
        "four training sizes for every model.</p></section>" + "".join(model_sections)
    )
    path = output_root / "print_report.html"
    path.write_text(
        _page("Complete PySR BioModels report", complete_body),
        encoding="utf-8",
    )
    return path
