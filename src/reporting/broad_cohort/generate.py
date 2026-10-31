"""Generate the broad 41-model PySR report from immutable artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .html_report import write_index_page, write_model_page, write_print_report
from .initial_conditions import (
    build_development_initial_conditions,
    make_initial_condition_figure,
)
from .inventory import (
    build_condition_inventory,
    build_failure_inventory,
    build_model_inventory,
    build_telemetry_summary,
    load_cohort,
    load_run_manifest,
)
from .mechanistic import audit_sbml_inputs, evaluate_mechanistic_recovery
from .plots import (
    make_coverage_figure,
    make_mechanistic_summary_figure,
    make_model_validation_figures,
    make_scalability_figure,
    make_validation_heatmap,
    select_representative_states,
)

DEFAULT_RUN_ID = "biomodels41_standard_exact"


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading it into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def held_out_artifacts(results_root: Path) -> list[str]:
    """Discover any held-out artifacts without opening their contents.

    Args:
        results_root: Scientific result root.

    Returns:
        Relative paths for discovered test files.
    """
    return sorted(
        str(path.relative_to(results_root))
        for path in results_root.glob(
            "BIOMD*/generic_exact_experiment/**/test/test_*.csv"
        )
    )


def _write_data(
    output_root: Path,
    model_frame: pd.DataFrame,
    condition_frame: pd.DataFrame,
    equation_frame: pd.DataFrame,
    failure_frame: pd.DataFrame,
    telemetry_frame: pd.DataFrame,
    mechanistic_detail: pd.DataFrame,
    mechanistic_summary: pd.DataFrame,
    mechanistic_errors: pd.DataFrame,
    sbml_audit: pd.DataFrame,
) -> None:
    """Write machine-readable report tables.

    Args:
        output_root: Report destination root.
        model_frame: Model pipeline inventory.
        condition_frame: Complete condition inventory.
        equation_frame: Selected equation rows.
        failure_frame: Failure ledger.
        telemetry_frame: Per-model telemetry summaries.
        mechanistic_detail: Per-RHS recovery results.
        mechanistic_summary: Per-condition recovery summaries.
        mechanistic_errors: Isolated mechanistic evaluation errors.
        sbml_audit: Local-versus-inference SBML checksum audit.
    """
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "model_pipeline_status.csv": model_frame,
        "model_training_condition_summary.csv": condition_frame,
        "selected_equations.csv": equation_frame,
        "failure_inventory.csv": failure_frame,
        "telemetry_summary.csv": telemetry_frame,
        "mechanistic_recovery.csv": mechanistic_detail,
        "mechanistic_recovery_summary.csv": mechanistic_summary,
        "mechanistic_errors.csv": mechanistic_errors,
        "sbml_input_audit.csv": sbml_audit,
    }
    for filename, frame in tables.items():
        frame.to_csv(data_root / filename, index=False)


def generate_report(
    *,
    results_root: Path,
    runs_root: Path,
    cohort_path: Path,
    output_root: Path,
    model_root: Path | None,
    archive_path: Path | None,
    run_id: str,
    mechanistic_mode: str,
    maximum_states_per_figure: int,
    reuse_existing_figures: bool,
    mechanistic_timeout_seconds: float,
) -> dict[str, object]:
    """Generate all broad-cohort tables, figures, and HTML pages.

    Args:
        results_root: Read-only scientific result root.
        runs_root: Read-only orchestration run root.
        cohort_path: Ordered cohort JSON.
        output_root: Generated report destination.
        model_root: Optional curated SBML root.
        archive_path: Optional source archive used for provenance.
        run_id: Batch run identifier.
        mechanistic_mode: ``none``, ``dependencies``, or ``full``.
        maximum_states_per_figure: Representative states per model figure.
        reuse_existing_figures: Reuse figures generated from this immutable
            snapshot when their target files already exist.
        mechanistic_timeout_seconds: Symbolic-operation limit per RHS.

    Returns:
        Report provenance and headline counts.
    """
    model_ids = load_cohort(cohort_path)
    condition_frame, equation_frame = build_condition_inventory(results_root, model_ids)
    model_frame = build_model_inventory(
        results_root, runs_root, run_id, model_ids, condition_frame
    )
    telemetry_frame = build_telemetry_summary(runs_root, run_id)
    failure_frame = build_failure_inventory(model_frame, condition_frame)

    mechanistic_detail = pd.DataFrame()
    mechanistic_summary = pd.DataFrame()
    mechanistic_errors = pd.DataFrame()
    sbml_audit = pd.DataFrame()
    if mechanistic_mode != "none":
        if model_root is None:
            raise ValueError("--model-root is required for mechanistic analysis.")
        sbml_audit = audit_sbml_inputs(
            model_root,
            model_ids,
            load_run_manifest(runs_root, run_id),
        )
        mismatches = sbml_audit.loc[sbml_audit["status"] == "mismatch"]
        if not mismatches.empty:
            mismatch_ids = ", ".join(mismatches["model_id"].astype(str))
            raise ValueError(
                f"Local SBML inputs differ from the inference snapshot: {mismatch_ids}"
            )
        (
            mechanistic_detail,
            mechanistic_summary,
            mechanistic_errors,
        ) = evaluate_mechanistic_recovery(
            results_root,
            model_root,
            condition_frame,
            mode=mechanistic_mode,
            timeout_seconds=mechanistic_timeout_seconds,
            cache_root=output_root / ".cache" / "mechanistic",
        )
        if not mechanistic_errors.empty:
            mechanistic_failures = mechanistic_errors.copy()
            mechanistic_failures["stage"] = "mechanistic_analysis"
            mechanistic_failures["scientific_or_operational"] = "analysis_limitation"
            mechanistic_failures["dataset_name"] = mechanistic_failures[
                "dataset_name"
            ].fillna("")
            failure_frame = pd.concat(
                [
                    failure_frame,
                    mechanistic_failures[
                        [
                            "model_id",
                            "dataset_name",
                            "stage",
                            "failure_type",
                            "scientific_or_operational",
                            "message",
                        ]
                    ],
                ],
                ignore_index=True,
            )

    figures_root = output_root / "figures"
    coverage_path = figures_root / "coverage.png"
    scalability_path = figures_root / "scalability.png"
    validation_heatmap_path = figures_root / "validation_heatmap.png"
    if not reuse_existing_figures or not coverage_path.is_file():
        make_coverage_figure(model_frame, condition_frame, coverage_path)
    if not reuse_existing_figures or not scalability_path.is_file():
        make_scalability_figure(model_frame, scalability_path)
    if not reuse_existing_figures or not validation_heatmap_path.is_file():
        make_validation_heatmap(condition_frame, validation_heatmap_path)
    mechanistic_figure_path = figures_root / "mechanistic_recovery.png"
    if not mechanistic_summary.empty and (
        not reuse_existing_figures or not mechanistic_figure_path.is_file()
    ):
        make_mechanistic_summary_figure(mechanistic_summary, mechanistic_figure_path)

    for model_id in model_ids:
        model = model_frame.loc[model_frame["model_id"] == model_id].iloc[0]
        conditions = condition_frame.loc[condition_frame["model_id"] == model_id]
        equations = equation_frame.loc[equation_frame["model_id"] == model_id]
        model_mechanistic_summary = (
            mechanistic_summary.loc[mechanistic_summary["model_id"] == model_id]
            if not mechanistic_summary.empty
            else pd.DataFrame()
        )
        model_mechanistic_detail = (
            mechanistic_detail.loc[mechanistic_detail["model_id"] == model_id]
            if not mechanistic_detail.empty
            else pd.DataFrame()
        )
        model_output = output_root / "models" / model_id
        legacy_figure = model_output / "figures" / "validation_overview.png"
        if legacy_figure.is_file():
            legacy_figure.unlink()
        representative_states = []
        validation_figure_datasets = []
        training_initial_conditions = pd.DataFrame()
        validation_initial_conditions = pd.DataFrame()
        if not equations.empty:
            figure_directory = model_output / "figures"
            expected_figure_datasets = (
                conditions.loc[conditions["selection_present"], "dataset_name"]
                .astype(str)
                .tolist()
            )
            expected_figure_paths = [
                figure_directory / f"validation_{dataset_name}.png"
                for dataset_name in expected_figure_datasets
            ]
            if (
                reuse_existing_figures
                and expected_figure_paths
                and all(path.is_file() for path in expected_figure_paths)
            ):
                representative_states = select_representative_states(
                    equations, maximum_states_per_figure
                )
                validation_figure_datasets = expected_figure_datasets
            else:
                (
                    representative_states,
                    validation_figure_datasets,
                ) = make_model_validation_figures(
                    results_root,
                    model_id,
                    equations,
                    figure_directory,
                    maximum_states=maximum_states_per_figure,
                )
        if model_root is not None:
            source_model_path = model_root / model_id / f"{model_id}_url.xml"
            try:
                (
                    training_initial_conditions,
                    validation_initial_conditions,
                    nominal_initial_conditions,
                ) = build_development_initial_conditions(
                    model_id,
                    source_model_path,
                )
                initial_condition_path = (
                    model_output / "figures" / "initial_condition_design.png"
                )
                if not reuse_existing_figures or not initial_condition_path.is_file():
                    make_initial_condition_figure(
                        model_id,
                        training_initial_conditions,
                        validation_initial_conditions,
                        nominal_initial_conditions,
                        initial_condition_path,
                    )
            except Exception as error:  # noqa: BLE001 - isolate report metadata.
                print(
                    f"Initial-condition reporting failed for {model_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
        model_output.mkdir(parents=True, exist_ok=True)
        conditions.to_csv(model_output / "condition_summary.csv", index=False)
        equations.to_csv(model_output / "selected_equations.csv", index=False)
        model_mechanistic_detail.to_csv(
            model_output / "mechanistic_recovery.csv", index=False
        )
        training_initial_conditions.to_csv(
            model_output / "training_initial_conditions.csv", index=False
        )
        validation_initial_conditions.to_csv(
            model_output / "validation_initial_conditions.csv", index=False
        )
        write_model_page(
            output_root,
            model,
            conditions,
            equations,
            model_mechanistic_summary,
            model_mechanistic_detail,
            representative_states,
            validation_figure_datasets,
            training_initial_conditions,
            validation_initial_conditions,
        )

    test_paths = held_out_artifacts(results_root)
    provenance: dict[str, object] = {
        "archive_path": str(archive_path.resolve()) if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
        "results_root": str(results_root.resolve()),
        "runs_root": str(runs_root.resolve()),
        "model_root": str(model_root.resolve()) if model_root else None,
        "cohort_path": str(cohort_path.resolve()),
        "run_id": run_id,
        "requested_models": len(model_ids),
        "models_with_complete_candidates": int(model_frame["candidate_complete"].sum()),
        "models_with_complete_selection": int(
            model_frame["pipeline_status"].eq("pretest_complete").sum()
        ),
        "expected_conditions": len(condition_frame),
        "selected_conditions": int(condition_frame["selection_present"].sum()),
        "held_out_status": "not_opened" if not test_paths else "artifacts_present",
        "held_out_artifact_count": len(test_paths),
        "mechanistic_mode": mechanistic_mode,
        "sbml_hash_matches": int(sbml_audit["status"].eq("match").sum())
        if not sbml_audit.empty
        else 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_data(
        output_root,
        model_frame,
        condition_frame,
        equation_frame,
        failure_frame,
        telemetry_frame,
        mechanistic_detail,
        mechanistic_summary,
        mechanistic_errors,
        sbml_audit,
    )
    write_index_page(
        output_root,
        model_frame,
        condition_frame,
        mechanistic_summary,
        provenance,
    )
    write_print_report(output_root, model_ids)
    return provenance


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Validated CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate a human-readable broad PySR cohort report."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--mechanistic-mode",
        choices=("none", "dependencies", "full"),
        default="full",
    )
    parser.add_argument("--maximum-states-per-figure", type=int, default=10)
    parser.add_argument("--reuse-existing-figures", action="store_true")
    parser.add_argument("--mechanistic-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """Run broad-cohort reporting from the command line."""
    arguments = parse_arguments()
    provenance = generate_report(
        results_root=arguments.results_root,
        runs_root=arguments.runs_root,
        cohort_path=arguments.cohort,
        output_root=arguments.output_root,
        model_root=arguments.model_root,
        archive_path=arguments.archive_path,
        run_id=arguments.run_id,
        mechanistic_mode=arguments.mechanistic_mode,
        maximum_states_per_figure=arguments.maximum_states_per_figure,
        reuse_existing_figures=arguments.reuse_existing_figures,
        mechanistic_timeout_seconds=arguments.mechanistic_timeout_seconds,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
