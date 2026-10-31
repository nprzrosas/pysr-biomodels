"""Run post-hoc mechanistic comparisons for a broad result cohort."""

import hashlib
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate_algebraic_recovery import evaluate_one_rhs as evaluate_algebraic_rhs
from src.evaluate_dependency_recovery import evaluate_one_rhs as evaluate_dependency_rhs
from src.evaluate_functional_form_recovery import (
    evaluate_one_rhs as evaluate_functional_rhs,
)
from src.extract_mechanistic_ground_truth import extract_mechanistic_ground_truth

from .inventory import load_selected_row, selection_path, state_ids_from_selection


class MechanisticEvaluationTimeoutError(TimeoutError):
    """Raised when one symbolic comparison exceeds its time budget."""


@contextmanager
def evaluation_timeout(seconds: float):
    """Limit one symbolic evaluation on POSIX systems.

    Args:
        seconds: Positive wall-clock limit. Non-positive values disable it.

    Yields:
        Control to the protected symbolic operation.
    """
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise MechanisticEvaluationTimeoutError(
            f"Symbolic evaluation exceeded {seconds:g} seconds."
        )

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def model_path(model_root: Path, model_id: str) -> Path:
    """Return the canonical local SBML path for one model.

    Args:
        model_root: Root containing one directory per BioModels identifier.
        model_id: BioModels identifier.

    Returns:
        Expected curated SBML path.
    """
    return model_root / model_id / f"{model_id}_url.xml"


def audit_sbml_inputs(
    model_root: Path,
    model_ids: tuple[str, ...],
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Verify local SBML files against hashes captured during inference.

    Args:
        model_root: Curated SBML input root.
        model_ids: Ordered cohort identifiers.
        manifest: Batch run manifest containing expected SBML hashes.

    Returns:
        One input-provenance row per requested model.
    """
    manifest_models = manifest.get("models", {})
    rows = []
    for model_id in model_ids:
        path = model_path(model_root, model_id)
        expected = manifest_models.get(model_id, {}).get("sbml_sha256", "")
        observed = ""
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            observed = digest.hexdigest()
        if not path.is_file():
            status = "missing"
        elif not expected:
            status = "expected_hash_missing"
        elif observed == expected:
            status = "match"
        else:
            status = "mismatch"
        rows.append(
            {
                "model_id": model_id,
                "sbml_path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _merge_selected_metrics(
    dependency: dict[str, Any],
    functional: dict[str, Any] | None,
    algebraic: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge stable mechanistic fields into one RHS record.

    Args:
        dependency: Dependency-recovery result.
        functional: Optional functional-form result.
        algebraic: Optional algebraic-recovery result.

    Returns:
        Unified human- and machine-readable record.
    """
    row = dict(dependency)
    if functional is not None:
        for key in (
            "numerator_monomial_precision",
            "numerator_monomial_recall",
            "numerator_monomial_f1",
            "interaction_precision",
            "interaction_recall",
            "interaction_f1",
            "rational_structure_match",
            "denominator_dependency_f1",
            "true_numerator_degree",
            "inferred_numerator_degree",
            "true_denominator_degree",
            "inferred_denominator_degree",
        ):
            row[key] = functional.get(key, np.nan)
    if algebraic is not None:
        for key in (
            "canonical_true_equation",
            "canonical_inferred_equation",
            "symbolic_difference",
            "exact_algebraic_match",
            "numerical_equivalence_status",
            "numerical_equivalence_match",
            "numerical_equivalence_n_valid_points",
            "numerical_equivalence_max_absolute_error",
            "numerical_equivalence_max_relative_error",
        ):
            row[key] = algebraic.get(key, np.nan)
    return row


def evaluate_mechanistic_recovery(
    results_root: Path,
    model_root: Path,
    condition_frame: pd.DataFrame,
    mode: str = "full",
    timeout_seconds: float = 10.0,
    cache_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare selected equations with SBML ground-truth equations.

    Args:
        results_root: Read-only broad-cohort result root.
        model_root: Curated SBML input root.
        condition_frame: Complete model-by-training condition inventory.
        mode: ``dependencies`` for structural dependencies only or ``full``
            for functional, exact algebraic, and numerical comparisons.
        timeout_seconds: Per-operation symbolic time limit.
        cache_root: Optional directory for completed per-model checkpoints.

    Returns:
        Per-RHS results, condition summaries, and mechanistic errors.
    """
    if mode not in {"dependencies", "full"}:
        raise ValueError(f"Unknown mechanistic mode: {mode}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    selected_conditions = condition_frame.loc[condition_frame["selection_present"]]

    for model_id, model_conditions in selected_conditions.groupby(
        "model_id", sort=False
    ):
        model_id = str(model_id)
        cache_path = (
            cache_root / f"{model_id}.{mode}.csv" if cache_root is not None else None
        )
        error_cache_path = (
            cache_root / f"{model_id}.{mode}.errors.csv"
            if cache_root is not None
            else None
        )
        if cache_path is not None and cache_path.is_file():
            print(f"Mechanistic recovery {model_id}: cached", flush=True)
            try:
                cached_rows = pd.read_csv(cache_path)
            except pd.errors.EmptyDataError:
                cached_rows = pd.DataFrame()
            rows.extend(cached_rows.to_dict(orient="records"))
            if error_cache_path is not None and error_cache_path.is_file():
                try:
                    cached_errors = pd.read_csv(error_cache_path)
                except pd.errors.EmptyDataError:
                    cached_errors = pd.DataFrame()
                errors.extend(cached_errors.to_dict(orient="records"))
            continue

        print(f"Mechanistic recovery {model_id}: running", flush=True)
        model_rows: list[dict[str, Any]] = []
        model_errors: list[dict[str, str]] = []
        sbml_path = model_path(model_root, str(model_id))
        if not sbml_path.is_file():
            model_errors.append(
                {
                    "model_id": str(model_id),
                    "dataset_name": "",
                    "state_id": "",
                    "failure_type": "missing_sbml",
                    "message": str(sbml_path),
                }
            )
        else:
            try:
                ground_truth_timeout = max(30.0, timeout_seconds * 6.0)
                with evaluation_timeout(ground_truth_timeout):
                    ground_truth = extract_mechanistic_ground_truth(
                        model_id=str(model_id), model_path=sbml_path
                    )
            except Exception as error:  # noqa: BLE001 - isolate model extraction.
                model_errors.append(
                    {
                        "model_id": str(model_id),
                        "dataset_name": "",
                        "state_id": "",
                        "failure_type": "ground_truth_extraction_failed",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
                ground_truth = None

        if sbml_path.is_file() and ground_truth is not None:
            for condition in model_conditions.to_dict(orient="records"):
                dataset_name = str(condition["dataset_name"])
                selected_row = load_selected_row(
                    selection_path(results_root, str(model_id), dataset_name)
                )
                for state_id in state_ids_from_selection(selected_row):
                    try:
                        dependency = evaluate_dependency_rhs(
                            model_id=str(model_id),
                            dataset_name=dataset_name,
                            state_id=state_id,
                            ground_truth=ground_truth,
                            selected_row=selected_row,
                        )
                    except Exception as error:  # noqa: BLE001 - isolate one RHS.
                        model_errors.append(
                            {
                                "model_id": str(model_id),
                                "dataset_name": dataset_name,
                                "state_id": state_id,
                                "failure_type": "dependency_evaluation_failed",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue

                    functional = None
                    algebraic = None
                    if mode == "full":
                        try:
                            with evaluation_timeout(timeout_seconds):
                                functional = evaluate_functional_rhs(
                                    model_id=str(model_id),
                                    dataset_name=dataset_name,
                                    state_id=state_id,
                                    ground_truth=ground_truth,
                                    selected_row=selected_row,
                                )
                        except Exception as error:  # noqa: BLE001 - isolate one RHS.
                            model_errors.append(
                                {
                                    "model_id": str(model_id),
                                    "dataset_name": dataset_name,
                                    "state_id": state_id,
                                    "failure_type": "functional_evaluation_failed",
                                    "message": f"{type(error).__name__}: {error}",
                                }
                            )
                        try:
                            with evaluation_timeout(timeout_seconds):
                                algebraic = evaluate_algebraic_rhs(
                                    model_id=str(model_id),
                                    dataset_name=dataset_name,
                                    state_id=state_id,
                                    ground_truth=ground_truth,
                                    selected_row=selected_row,
                                )
                        except Exception as error:  # noqa: BLE001 - isolate one RHS.
                            model_errors.append(
                                {
                                    "model_id": str(model_id),
                                    "dataset_name": dataset_name,
                                    "state_id": state_id,
                                    "failure_type": "algebraic_evaluation_failed",
                                    "message": f"{type(error).__name__}: {error}",
                                }
                            )
                    model_rows.append(
                        _merge_selected_metrics(dependency, functional, algebraic)
                    )

        rows.extend(model_rows)
        errors.extend(model_errors)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(model_rows).to_csv(cache_path, index=False)
            pd.DataFrame(
                model_errors,
                columns=[
                    "model_id",
                    "dataset_name",
                    "state_id",
                    "failure_type",
                    "message",
                ],
            ).to_csv(error_cache_path, index=False)
        print(
            f"Mechanistic recovery {model_id}: "
            f"{len(model_rows)} RHS, {len(model_errors)} issues",
            flush=True,
        )

    detail = pd.DataFrame(rows)
    error_frame = pd.DataFrame(
        errors,
        columns=[
            "model_id",
            "dataset_name",
            "state_id",
            "failure_type",
            "message",
        ],
    )
    return detail, summarize_mechanistic_recovery(detail), error_frame


def summarize_mechanistic_recovery(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-RHS mechanistic recovery by condition.

    Args:
        detail: Per-RHS mechanistic results.

    Returns:
        One summary row per evaluated model and training condition.
    """
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (model_id, dataset_name), frame in detail.groupby(
        ["model_id", "dataset_name"], sort=False
    ):
        row: dict[str, Any] = {
            "model_id": model_id,
            "dataset_name": dataset_name,
            "n_rhs_evaluated": len(frame),
            "mechanistic_comparability": frame["mechanistic_comparability"].iloc[0],
            "macro_dependency_f1": pd.to_numeric(
                frame["dependency_f1"], errors="coerce"
            ).mean(),
        }
        if "exact_algebraic_match" in frame.columns:
            exact = frame["exact_algebraic_match"].dropna().astype(bool)
            numerical = frame["numerical_equivalence_match"].dropna().astype(bool)
            row.update(
                {
                    "n_algebraically_evaluated_rhs": len(exact),
                    "exact_algebraic_matches": int(exact.sum()),
                    "exact_algebraic_fraction": (
                        float(exact.mean()) if len(exact) else np.nan
                    ),
                    "n_numerically_evaluated_rhs": len(numerical),
                    "numerical_equivalence_matches": int(numerical.sum()),
                    "numerical_equivalence_fraction": (
                        float(numerical.mean()) if len(numerical) else np.nan
                    ),
                    "macro_numerator_monomial_f1": pd.to_numeric(
                        frame["numerator_monomial_f1"], errors="coerce"
                    ).mean(),
                    "macro_interaction_f1": pd.to_numeric(
                        frame["interaction_f1"], errors="coerce"
                    ).mean(),
                    "rational_structure_accuracy": pd.to_numeric(
                        frame["rational_structure_match"], errors="coerce"
                    ).mean(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
