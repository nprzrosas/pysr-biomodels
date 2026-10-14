"""Variant-aware orchestration for one generic benchmark model.

This module contains stage ordering and artifact gates only. Scientific work
is delegated to the existing production command-line modules in isolated
subprocesses.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    get_benchmark_variant,
)
from .evaluate_generic_validation import validate_candidate_pool_provenance
from .preflight import PreflightResult, run_preflight
from .run_manifest import (
    ArtifactCheckpoint,
    RunContext,
    checkpoint_as_dict,
    checkpoint_paths,
)

TRAINING_DATASETS = ("train_04", "train_08", "train_16", "train_32")
EXHAUSTIVE_STATE_LIMIT = 3
EVENT_PREFIX = "@@PYSR_BATCH_EVENT@@"

PHASES = (
    "preflight",
    "candidates",
    "optional_exhaustive",
    "selector",
    "pretest",
    "test",
)

CommandRunner = Callable[[tuple[str, ...]], None]
EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ModelRunResult:
    """Orchestration outcome for one model invocation."""

    model_id: str
    derivative_method: str
    experiment_namespace: str
    phase: str
    state_ids: tuple[str, ...]
    state_count: int
    exhaustive_permitted: bool
    stages: dict[str, str]
    stage_checkpoints: dict[str, ArtifactCheckpoint]
    pretest_checkpoint: ArtifactCheckpoint | None
    selection_identity: dict[str, dict[str, Any]] | None


def experiment_root(
    *,
    model_id: str,
    variant: BenchmarkVariant,
    run_context: RunContext,
) -> Path:
    """Resolve a model's protected scientific namespace.

    Args:
        model_id: BioModels identifier.
        variant: Authoritative derivative method and namespace.
        run_context: Configured filesystem roots.

    Returns:
        Variant-specific scientific result directory.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")
    return run_context.results_root / model_id / variant.experiment_namespace


def candidate_artifact_paths(
    *, root: Path, state_ids: Sequence[str]
) -> tuple[Path, ...]:
    """Return required candidate-generation artifacts.

    Args:
        root: Variant-specific experiment directory.
        state_ids: Ordered dynamic state identifiers.

    Returns:
        Authoritative candidate artifact paths.
    """
    paths = [root / "experiment_metadata.csv"]
    for dataset_name in TRAINING_DATASETS:
        dataset_root = root / dataset_name
        paths.append(dataset_root / "run_metadata.csv")
        for state_id in state_ids:
            state_root = dataset_root / f"state_{state_id}"
            paths.extend(
                (
                    state_root / "all_equations.csv",
                    state_root / "top_8_by_loss.csv",
                )
            )
    return tuple(paths)


def exhaustive_artifact_paths(*, root: Path) -> tuple[Path, ...]:
    """Return required exhaustive-validation artifacts.

    Args:
        root: Variant-specific experiment directory.

    Returns:
        Exhaustive output paths for all training sizes.
    """
    filenames = (
        "selected_system.csv",
        "validation_predictions.csv",
        "validation_state_metrics.csv",
        "validation_system_ranking.csv",
    )
    return tuple(
        root / dataset_name / filename
        for dataset_name in TRAINING_DATASETS
        for filename in filenames
    )


def selector_artifact_paths(*, root: Path) -> tuple[Path, ...]:
    """Return required coordinate Selector v1 artifacts.

    Args:
        root: Variant-specific experiment directory.

    Returns:
        Selector output paths for all training sizes.
    """
    filenames = (
        "coordinate_search_trace.csv",
        "coordinate_selected_system.csv",
        "coordinate_validation_predictions.csv",
        "coordinate_validation_ranking.csv",
        "coordinate_validation_state_metrics.csv",
    )
    return tuple(
        root / dataset_name / "coordinate_selection" / filename
        for dataset_name in TRAINING_DATASETS
        for filename in filenames
    )


def test_artifact_paths(*, root: Path) -> tuple[Path, ...]:
    """Return required held-out test artifacts.

    Args:
        root: Variant-specific experiment directory.

    Returns:
        Test outputs for all training sizes and the aggregate comparison.
    """
    paths = [root / "coordinate_selection" / "test" / "test_comparison.csv"]
    filenames = (
        "test_predictions.csv",
        "test_state_metrics.csv",
        "test_summary.csv",
        "test_trajectory_metrics.csv",
    )
    for dataset_name in TRAINING_DATASETS:
        directory = root / dataset_name / "coordinate_selection" / "test"
        paths.extend(directory / filename for filename in filenames)
    return tuple(paths)


def artifact_state(paths: Sequence[Path]) -> str:
    """Classify required artifacts without treating partial output as complete.

    Args:
        paths: Complete required path set for one stage.

    Returns:
        ``absent``, ``complete``, or ``partial``.
    """
    existence = [Path(path).is_file() for path in paths]
    if all(existence):
        return "complete"
    if any(existence):
        return "partial"
    return "absent"


def production_command(
    *, module: str, model_id: str, variant: BenchmarkVariant
) -> tuple[str, ...]:
    """Build a canonical package-execution command for a scientific stage.

    Args:
        module: Existing production module name.
        model_id: BioModels identifier.
        variant: Authoritative benchmark variant.

    Returns:
        Immutable subprocess argument tuple.
    """
    return (
        sys.executable,
        "-m",
        module,
        model_id,
        "--derivative-method",
        variant.derivative_method.value,
    )


def default_command_runner(command: tuple[str, ...]) -> None:
    """Run a production stage and propagate a nonzero exit status.

    Args:
        command: Canonical package-execution command.
    """
    subprocess.run(command, check=True)


def _emit(
    callback: EventCallback | None,
    *,
    model_id: str,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    """Emit one orchestration event to a caller or structured stdout."""
    event = {
        "model_id": model_id,
        "stage": stage,
        "status": status,
        **details,
    }
    if callback is None:
        print(f"{EVENT_PREFIX}{json.dumps(event, sort_keys=True)}", flush=True)
    else:
        callback(event)


def _require_clean_stage(
    *,
    stage: str,
    paths: tuple[Path, ...],
    resume: bool,
) -> bool:
    """Return whether a stage should run, rejecting unsafe artifact states."""
    state = artifact_state(paths)
    if state == "partial":
        raise RuntimeError(
            f"Stage '{stage}' has partial artifacts and will not be overwritten."
        )
    if state == "complete":
        if resume:
            return False
        raise RuntimeError(
            f"Stage '{stage}' is already complete and overwrite is disabled."
        )
    return True


def _validate_candidate_lineage(*, root: Path, variant: BenchmarkVariant) -> None:
    """Validate candidate provenance before any downstream scientific stage."""
    validate_candidate_pool_provenance(
        experiment_root=root,
        variant=variant,
        dataset_names=TRAINING_DATASETS,
    )


def _run_stage(
    *,
    stage: str,
    module: str,
    model_id: str,
    variant: BenchmarkVariant,
    paths: tuple[Path, ...],
    resume: bool,
    command_runner: CommandRunner,
    event_callback: EventCallback | None,
    checkpoint_root: Path,
    expected_checkpoint: ArtifactCheckpoint | None = None,
) -> tuple[str, ArtifactCheckpoint]:
    """Run one delegated stage after strict artifact-state checks."""
    should_run = _require_clean_stage(stage=stage, paths=paths, resume=resume)
    if not should_run:
        checkpoint = checkpoint_paths(list(paths), relative_to=checkpoint_root)
        if expected_checkpoint is not None and checkpoint != expected_checkpoint:
            raise RuntimeError(
                f"Completed stage '{stage}' differs from its manifest checkpoint."
            )
        _emit(event_callback, model_id=model_id, stage=stage, status="skipped")
        return "skipped", checkpoint

    _emit(event_callback, model_id=model_id, stage=stage, status="running")
    command_runner(
        production_command(module=module, model_id=model_id, variant=variant)
    )
    if artifact_state(paths) != "complete":
        raise RuntimeError(
            f"Stage '{stage}' returned without its complete artifact contract."
        )
    checkpoint = checkpoint_paths(list(paths), relative_to=checkpoint_root)
    _emit(
        event_callback,
        model_id=model_id,
        stage=stage,
        status="completed",
        artifact_checkpoint=checkpoint_as_dict(checkpoint),
    )
    return "completed", checkpoint


def selection_identity(
    *, root: Path, state_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Read frozen coordinate-selected system identities.

    Args:
        root: Variant-specific experiment directory.
        state_ids: Ordered state identifiers.

    Returns:
        Selected system ID and state ranks by training condition.
    """
    identities: dict[str, dict[str, Any]] = {}
    for dataset_name in TRAINING_DATASETS:
        path = (
            root
            / dataset_name
            / "coordinate_selection"
            / "coordinate_selected_system.csv"
        )
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(f"Expected one selected-system row in {path}.")
        row = rows[0]
        identities[dataset_name] = {
            "system_id": int(float(row["system_id"])),
            "candidate_ranks": {
                state_id: int(float(row[f"candidate_rank_{state_id}"]))
                for state_id in state_ids
            },
        }
    return identities


def pretest_artifact_paths(
    *, root: Path, state_ids: Sequence[str], exhaustive_permitted: bool
) -> tuple[Path, ...]:
    """Return all authoritative scientific artifacts frozen before test."""
    paths = list(candidate_artifact_paths(root=root, state_ids=state_ids))
    if exhaustive_permitted:
        paths.extend(exhaustive_artifact_paths(root=root))
    paths.extend(selector_artifact_paths(root=root))
    return tuple(paths)


def verify_pretest_checkpoint(
    *,
    root: Path,
    state_ids: Sequence[str],
    exhaustive_permitted: bool,
    expected: ArtifactCheckpoint,
) -> None:
    """Reject held-out evaluation if frozen pre-test artifacts changed."""
    current = checkpoint_paths(
        pretest_artifact_paths(
            root=root,
            state_ids=state_ids,
            exhaustive_permitted=exhaustive_permitted,
        ),
        relative_to=root,
    )
    if current != expected:
        raise RuntimeError(
            "Pre-test artifact checkpoint changed; held-out test is blocked. "
            f"Expected {asdict(expected)}, observed {asdict(current)}."
        )


def _require_preflight(result: PreflightResult) -> None:
    """Reject a model that is incompatible or lacks a simulation horizon."""
    if not result.compatibility.compatible:
        raise ValueError(
            f"Model {result.model_id} is incompatible: {result.compatibility.reasons}"
        )
    if result.simulation_time is None:
        raise ValueError(f"No curated simulation time for {result.model_id}.")


def run_model(
    model_id: str,
    benchmark_variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
    phase: str = "preflight",
    run_context: RunContext | None = None,
    resume: bool = True,
    *,
    expected_pretest_checkpoint: ArtifactCheckpoint | None = None,
    held_out_test_authorized: bool = False,
    command_runner: CommandRunner = default_command_runner,
    preflight_runner: Callable[[str], PreflightResult] = run_preflight,
    event_callback: EventCallback | None = None,
    expected_stage_checkpoints: dict[str, ArtifactCheckpoint] | None = None,
) -> ModelRunResult:
    """Orchestrate one model while delegating all scientific computation.

    Args:
        model_id: BioModels identifier.
        benchmark_variant: Authoritative derivative method and namespace.
        phase: Explicit orchestration phase; pre-test never implies test.
        run_context: Non-scientific run and filesystem context.
        resume: Skip only stages whose complete artifact contract exists.
        expected_pretest_checkpoint: Frozen checkpoint required for test.
        held_out_test_authorized: Explicit authorization to open held-out data.
        command_runner: Injectable subprocess runner used by tests.
        preflight_runner: Injectable existing preflight implementation.
        event_callback: Optional structured stage-event sink.
        expected_stage_checkpoints: Prior manifest checkpoints used on resume.

    Returns:
        Stage outcomes and optional frozen selection checkpoint.
    """
    if not isinstance(benchmark_variant, BenchmarkVariant):
        raise TypeError("benchmark_variant must be a BenchmarkVariant.")
    if phase not in PHASES:
        raise ValueError(f"Unknown phase '{phase}'. Expected one of {PHASES}.")

    context = run_context or RunContext(run_id="standalone")
    result = preflight_runner(model_id)
    _require_preflight(result)
    root = experiment_root(
        model_id=model_id,
        variant=benchmark_variant,
        run_context=context,
    )
    state_ids = tuple(result.spec.state_ids)
    exhaustive_permitted = result.spec.n_states <= EXHAUSTIVE_STATE_LIMIT
    stage_status: dict[str, str] = {"preflight": "completed"}
    stage_checkpoints: dict[str, ArtifactCheckpoint] = {}
    expected_checkpoints = expected_stage_checkpoints or {}
    _emit(event_callback, model_id=model_id, stage="preflight", status="completed")

    candidate_paths = candidate_artifact_paths(root=root, state_ids=state_ids)
    exhaustive_paths = exhaustive_artifact_paths(root=root)
    selector_paths = selector_artifact_paths(root=root)

    def ensure_candidates() -> None:
        status, checkpoint = _run_stage(
            stage="candidates",
            module="src.run_generic_benchmark",
            model_id=model_id,
            variant=benchmark_variant,
            paths=candidate_paths,
            resume=resume,
            command_runner=command_runner,
            event_callback=event_callback,
            expected_checkpoint=expected_checkpoints.get("candidates"),
            checkpoint_root=root,
        )
        stage_status["candidates"] = status
        stage_checkpoints["candidates"] = checkpoint
        _validate_candidate_lineage(root=root, variant=benchmark_variant)

    def ensure_exhaustive() -> None:
        if not exhaustive_permitted:
            raise RuntimeError(
                "Full Cartesian validation is prohibited for models with "
                f"more than {EXHAUSTIVE_STATE_LIMIT} states."
            )
        ensure_candidates_if_needed()
        status, checkpoint = _run_stage(
            stage="optional_exhaustive",
            module="src.run_generic_validation",
            model_id=model_id,
            variant=benchmark_variant,
            paths=exhaustive_paths,
            resume=resume,
            command_runner=command_runner,
            event_callback=event_callback,
            expected_checkpoint=expected_checkpoints.get("optional_exhaustive"),
            checkpoint_root=root,
        )
        stage_status["optional_exhaustive"] = status
        stage_checkpoints["optional_exhaustive"] = checkpoint

    def ensure_selector() -> None:
        ensure_candidates_if_needed()
        status, checkpoint = _run_stage(
            stage="selector",
            module="src.select_generic_system",
            model_id=model_id,
            variant=benchmark_variant,
            paths=selector_paths,
            resume=resume,
            command_runner=command_runner,
            event_callback=event_callback,
            expected_checkpoint=expected_checkpoints.get("selector"),
            checkpoint_root=root,
        )
        stage_status["selector"] = status
        stage_checkpoints["selector"] = checkpoint

    def ensure_candidates_if_needed() -> None:
        if "candidates" not in stage_status:
            ensure_candidates()

    if phase == "candidates":
        ensure_candidates()
    elif phase == "optional_exhaustive":
        ensure_exhaustive()
    elif phase == "selector":
        ensure_selector()
    elif phase == "pretest":
        ensure_candidates()
        if exhaustive_permitted:
            ensure_exhaustive()
        else:
            stage_status["optional_exhaustive"] = "not_permitted"
        ensure_selector()
    elif phase == "test":
        if not held_out_test_authorized:
            raise RuntimeError("Held-out test requires explicit authorization.")
        if expected_pretest_checkpoint is None:
            raise RuntimeError("Held-out test requires a verified pre-test checkpoint.")
        if artifact_state(candidate_paths) != "complete":
            raise RuntimeError("Candidate stage is not complete; test is blocked.")
        _validate_candidate_lineage(root=root, variant=benchmark_variant)
        if artifact_state(selector_paths) != "complete":
            raise RuntimeError("Selector stage is not complete; test is blocked.")
        verify_pretest_checkpoint(
            root=root,
            state_ids=state_ids,
            exhaustive_permitted=exhaustive_permitted,
            expected=expected_pretest_checkpoint,
        )
        status, checkpoint = _run_stage(
            stage="test",
            module="src.run_generic_test",
            model_id=model_id,
            variant=benchmark_variant,
            paths=test_artifact_paths(root=root),
            resume=resume,
            command_runner=command_runner,
            event_callback=event_callback,
            expected_checkpoint=expected_checkpoints.get("test"),
            checkpoint_root=root,
        )
        stage_status["test"] = status
        stage_checkpoints["test"] = checkpoint

    checkpoint = None
    identities = None
    if artifact_state(selector_paths) == "complete":
        identities = selection_identity(root=root, state_ids=state_ids)
    if (
        artifact_state(candidate_paths) == "complete"
        and artifact_state(selector_paths) == "complete"
        and (not exhaustive_permitted or artifact_state(exhaustive_paths) == "complete")
    ):
        checkpoint = checkpoint_paths(
            pretest_artifact_paths(
                root=root,
                state_ids=state_ids,
                exhaustive_permitted=exhaustive_permitted,
            ),
            relative_to=root,
        )

    return ModelRunResult(
        model_id=model_id,
        derivative_method=benchmark_variant.derivative_method.value,
        experiment_namespace=benchmark_variant.experiment_namespace,
        phase=phase,
        state_ids=state_ids,
        state_count=result.spec.n_states,
        exhaustive_permitted=exhaustive_permitted,
        stages=stage_status,
        stage_checkpoints=stage_checkpoints,
        pretest_checkpoint=checkpoint,
        selection_identity=identities,
    )


def parse_checkpoint(value: str | None) -> ArtifactCheckpoint | None:
    """Parse a JSON checkpoint supplied by the batch coordinator.

    Args:
        value: JSON object containing checkpoint fields.

    Returns:
        Parsed checkpoint or ``None``.
    """
    if value is None:
        return None
    payload = json.loads(value)
    return ArtifactCheckpoint(**payload)


def parse_stage_checkpoints(value: str | None) -> dict[str, ArtifactCheckpoint]:
    """Parse manifest stage checkpoints supplied by the coordinator.

    Args:
        value: JSON mapping of stage names to checkpoint objects.

    Returns:
        Parsed stage checkpoint mapping.
    """
    if value is None:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise TypeError("Stage checkpoints must be a JSON object.")
    return {
        str(stage): ArtifactCheckpoint(**checkpoint)
        for stage, checkpoint in payload.items()
    }


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the standalone model-orchestration CLI.

    Args:
        arguments: Optional argument list for tests.

    Returns:
        Parsed orchestration arguments.
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate one generic benchmark model by explicit phase."
    )
    parser.add_argument("model_id", help="BioModels identifier.")
    parser.add_argument(
        "--derivative-method",
        choices=("exact", "savgol"),
        default="exact",
        help="Benchmark derivative variant; defaults to exact.",
    )
    parser.add_argument("--phase", choices=PHASES, default="preflight")
    parser.add_argument("--run-id", default="standalone")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--authorize-held-out-test", action="store_true")
    parser.add_argument("--expected-pretest-checkpoint")
    parser.add_argument("--expected-stage-checkpoints")
    return parser.parse_args(arguments)


def main() -> None:
    """Run the package CLI and emit a structured completion event."""
    arguments = parse_arguments()
    variant = get_benchmark_variant(arguments.derivative_method)
    result = run_model(
        model_id=arguments.model_id,
        benchmark_variant=variant,
        phase=arguments.phase,
        run_context=RunContext(run_id=arguments.run_id),
        resume=not arguments.no_resume,
        expected_pretest_checkpoint=parse_checkpoint(
            arguments.expected_pretest_checkpoint
        ),
        held_out_test_authorized=arguments.authorize_held_out_test,
        expected_stage_checkpoints=parse_stage_checkpoints(
            arguments.expected_stage_checkpoints
        ),
    )
    print(
        f"{EVENT_PREFIX}{json.dumps({'event': 'result', **asdict(result)}, default=str)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
