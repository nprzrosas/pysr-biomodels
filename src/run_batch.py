"""Bounded subprocess orchestration for independent benchmark models."""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import psutil

from .benchmark_variant import BenchmarkVariant, get_benchmark_variant
from .preflight import PreflightResult, run_preflight
from .resource_monitor import (
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    PlainTextProgressReporter,
    ProcessTreeSampler,
    TelemetryWriter,
    WorkerTelemetryContext,
    discover_resources,
)
from .run_manifest import (
    ArtifactCheckpoint,
    ModelVariantLock,
    RunContext,
    create_run_manifest,
    load_run_manifest,
    utc_timestamp,
)
from .run_model import EVENT_PREFIX, PHASES

THREAD_CONTROL_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

POINTS_PER_TRAJECTORY = 101
EXHAUSTIVE_STATE_LIMIT = 3


@dataclass(frozen=True)
class WorkerOutcome:
    """Terminal outcome from one isolated model worker."""

    model_id: str
    worker_id: str
    status: str
    exit_code: int
    result: dict[str, Any] | None = None
    failure: str | None = None
    global_invariant_failure: bool = False


@dataclass(frozen=True)
class ActiveWorker:
    """Live subprocess metadata used by telemetry and interruption."""

    worker_id: str
    model_id: str
    stage: str
    state: str
    process: subprocess.Popen[str]


class ActiveProcessRegistry:
    """Thread-safe registry of live worker subprocesses."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._workers: dict[str, ActiveWorker] = {}

    def add(self, worker: ActiveWorker) -> None:
        """Register one live subprocess.

        Args:
            worker: Live worker description.
        """
        with self._lock:
            self._workers[worker.worker_id] = worker

    def update_stage(self, worker_id: str, stage: str, state: str) -> None:
        """Update a registered worker's current stage labels."""
        with self._lock:
            current = self._workers.get(worker_id)
            if current is not None:
                self._workers[worker_id] = ActiveWorker(
                    worker_id=current.worker_id,
                    model_id=current.model_id,
                    stage=stage,
                    state=state,
                    process=current.process,
                )

    def remove(self, worker_id: str) -> None:
        """Remove a completed worker."""
        with self._lock:
            self._workers.pop(worker_id, None)

    def snapshot(self) -> tuple[ActiveWorker, ...]:
        """Return an immutable snapshot of active workers."""
        with self._lock:
            return tuple(self._workers.values())

    def terminate_all(self) -> None:
        """Request termination of every active process group."""
        for worker in self.snapshot():
            process = worker.process
            if process.poll() is not None:
                continue
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                continue


def validate_model_ids(model_ids: Sequence[str]) -> tuple[str, ...]:
    """Reject an empty cohort and duplicate scheduling.

    Args:
        model_ids: Requested model identifiers in deterministic order.

    Returns:
        Validated immutable model sequence.
    """
    normalized = tuple(str(model_id) for model_id in model_ids)
    if not normalized:
        raise ValueError("The cohort manifest contains no models.")
    invalid = [
        model_id
        for model_id in normalized
        if not model_id or model_id in {".", ".."} or Path(model_id).name != model_id
    ]
    if invalid:
        raise ValueError(f"Model IDs must be safe path components: {invalid}")
    duplicates = sorted(
        model_id for model_id in set(normalized) if normalized.count(model_id) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate model scheduling is prohibited: {duplicates}")
    return normalized


def load_cohort_manifest(path: Path) -> tuple[str, ...]:
    """Load a list of model IDs from a JSON cohort manifest.

    Args:
        path: JSON file containing a list or a ``models`` list.

    Returns:
        Validated model identifiers.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list) or not all(
        isinstance(item, str) for item in models
    ):
        raise ValueError("Cohort manifest must contain a JSON list of model IDs.")
    return validate_model_ids(models)


def worker_environment(
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a worker environment with explicit single-thread controls.

    Args:
        base_environment: Environment to copy; defaults to ``os.environ``.

    Returns:
        Isolated subprocess environment.
    """
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(THREAD_CONTROL_ENVIRONMENT)
    return environment


def model_worker_command(
    *,
    model_id: str,
    variant: BenchmarkVariant,
    phase: str,
    run_context: RunContext,
    held_out_test_authorized: bool = False,
    expected_pretest_checkpoint: ArtifactCheckpoint | None = None,
    expected_stage_checkpoints: dict[str, ArtifactCheckpoint] | None = None,
) -> tuple[str, ...]:
    """Build the isolated generic model-worker command.

    Args:
        model_id: BioModels identifier.
        variant: Authoritative benchmark variant.
        phase: Explicit orchestration phase.
        run_context: Run identity and roots.
        held_out_test_authorized: Explicit held-out authorization.
        expected_pretest_checkpoint: Frozen checkpoint required for test.
        expected_stage_checkpoints: Prior verified stage checkpoints.

    Returns:
        Canonical package-execution command.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase '{phase}'.")
    command = [
        sys.executable,
        "-m",
        "src.run_model",
        model_id,
        "--derivative-method",
        variant.derivative_method.value,
        "--phase",
        phase,
        "--run-id",
        run_context.run_id,
    ]
    if held_out_test_authorized:
        command.append("--authorize-held-out-test")
    if expected_pretest_checkpoint is not None:
        command.extend(
            (
                "--expected-pretest-checkpoint",
                json.dumps(asdict(expected_pretest_checkpoint), sort_keys=True),
            )
        )
    if expected_stage_checkpoints:
        command.extend(
            (
                "--expected-stage-checkpoints",
                json.dumps(
                    {
                        stage: asdict(checkpoint)
                        for stage, checkpoint in expected_stage_checkpoints.items()
                    },
                    sort_keys=True,
                ),
            )
        )
    return tuple(command)


def schedule_models(
    *,
    model_ids: Sequence[str],
    workers: int,
    worker: Callable[[str, str], WorkerOutcome],
    stop_event: threading.Event | None = None,
    on_tick: Callable[[dict[str, Future[WorkerOutcome]], list[WorkerOutcome]], None]
    | None = None,
    poll_interval_seconds: float = 0.05,
) -> list[WorkerOutcome]:
    """Run independent model jobs with bounded lazy scheduling.

    Args:
        model_ids: Unique model identifiers.
        workers: Maximum number of active model subprocesses.
        worker: Function accepting model ID and worker ID.
        stop_event: Shared interruption marker that stops new scheduling.
        on_tick: Optional coordinator callback for monitoring and events.
        poll_interval_seconds: Scheduler polling interval.

    Returns:
        Outcomes in cohort order for models that started.
    """
    models = validate_model_ids(model_ids)
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    stopped = stop_event or threading.Event()
    pending = iter(models)
    active: dict[Future[WorkerOutcome], tuple[str, str]] = {}
    completed: list[WorkerOutcome] = []
    worker_counter = 0
    global_failure = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal worker_counter
        if stopped.is_set() or global_failure:
            return False
        try:
            model_id = next(pending)
        except StopIteration:
            return False
        worker_counter += 1
        worker_id = str(worker_counter)
        future = executor.submit(worker, model_id, worker_id)
        active[future] = (model_id, worker_id)
        return True

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="model-worker"
    ) as executor:
        for _ in range(min(workers, len(models))):
            submit_next(executor)

        while active:
            done, _ = wait(
                tuple(active),
                timeout=poll_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                model_id, worker_id = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:  # noqa: BLE001 - isolate worker failures.
                    outcome = WorkerOutcome(
                        model_id=model_id,
                        worker_id=worker_id,
                        status="failed",
                        exit_code=1,
                        failure=f"{type(error).__name__}: {error}",
                    )
                completed.append(outcome)
                if outcome.global_invariant_failure:
                    global_failure = True

            if on_tick is not None:
                on_tick(
                    {active[future][0]: future for future in active},
                    list(completed),
                )

            while len(active) < workers and submit_next(executor):
                pass

    order = {model_id: index for index, model_id in enumerate(models)}
    return sorted(completed, key=lambda outcome: order[outcome.model_id])


class SubprocessModelWorker:
    """Launch one isolated ``src.run_model`` subprocess per invocation."""

    def __init__(
        self,
        *,
        context: RunContext,
        variant: BenchmarkVariant,
        phase: str,
        registry: ActiveProcessRegistry,
        event_queue: queue.Queue[dict[str, Any]],
        expected_checkpoints: dict[str, ArtifactCheckpoint] | None = None,
        expected_stage_checkpoints: dict[str, dict[str, ArtifactCheckpoint]]
        | None = None,
        held_out_test_authorized: bool = False,
    ) -> None:
        """Configure an isolated production worker.

        Args:
            context: Run filesystem context.
            variant: Authoritative benchmark variant.
            phase: Explicit requested phase.
            registry: Live process registry.
            event_queue: Thread-to-coordinator event queue.
            expected_checkpoints: Pre-test checkpoints keyed by model.
            expected_stage_checkpoints: Verified stage checkpoints by model.
            held_out_test_authorized: Explicit held-out authorization.
        """
        self.context = context
        self.variant = variant
        self.phase = phase
        self.registry = registry
        self.event_queue = event_queue
        self.expected_checkpoints = expected_checkpoints or {}
        self.expected_stage_checkpoints = expected_stage_checkpoints or {}
        self.held_out_test_authorized = held_out_test_authorized

    def __call__(self, model_id: str, worker_id: str) -> WorkerOutcome:
        """Run one model and capture its structured events and log."""
        lock = ModelVariantLock(
            context=self.context,
            model_id=model_id,
            variant=self.variant,
        )
        try:
            lock.acquire()
        except RuntimeError as error:
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status="failed",
                exit_code=1,
                failure=str(error),
            )

        try:
            self.context.logs_directory.mkdir(parents=True, exist_ok=True)
            log_path = self.context.logs_directory / f"{model_id}.log"
            command = model_worker_command(
                model_id=model_id,
                variant=self.variant,
                phase=self.phase,
                run_context=self.context,
                held_out_test_authorized=self.held_out_test_authorized,
                expected_pretest_checkpoint=self.expected_checkpoints.get(model_id),
                expected_stage_checkpoints=self.expected_stage_checkpoints.get(
                    model_id
                ),
            )
            result_payload: dict[str, Any] | None = None
            environment = worker_environment()

            with log_path.open("a", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=self.context.project_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=(os.name == "posix"),
                )
                self.registry.add(
                    ActiveWorker(
                        worker_id=worker_id,
                        model_id=model_id,
                        stage="starting",
                        state="running",
                        process=process,
                    )
                )
                self.event_queue.put(
                    {
                        "model_id": model_id,
                        "worker_id": worker_id,
                        "stage": self.phase,
                        "status": "running",
                        "pid": process.pid,
                    }
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_stream.write(line)
                    log_stream.flush()
                    if not line.startswith(EVENT_PREFIX):
                        continue
                    try:
                        event = json.loads(line[len(EVENT_PREFIX) :])
                    except json.JSONDecodeError:
                        continue
                    event.update({"worker_id": worker_id})
                    self.event_queue.put(event)
                    if event.get("event") == "result":
                        result_payload = event
                    elif "stage" in event:
                        self.registry.update_stage(
                            worker_id,
                            str(event["stage"]),
                            str(event.get("status", "running")),
                        )
                exit_code = process.wait()

            status = "completed" if exit_code == 0 else "failed"
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status=status,
                exit_code=exit_code,
                result=result_payload,
                failure=None if exit_code == 0 else f"Worker exited {exit_code}.",
            )
        finally:
            self.registry.remove(worker_id)
            lock.release()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def preflight_model_record(*, result: PreflightResult, phase: str) -> dict[str, Any]:
    """Build a complete manifest record from existing preflight metadata.

    Args:
        result: Existing generic preflight result.
        phase: Requested orchestration phase.

    Returns:
        JSON-safe per-model run metadata.
    """
    if not result.compatibility.compatible:
        raise ValueError(
            f"Model {result.model_id} is incompatible: {result.compatibility.reasons}"
        )
    if result.simulation_time is None:
        raise ValueError(f"No curated simulation time for {result.model_id}.")
    state_count = result.spec.n_states
    return {
        "status": "pending",
        "model_id": result.model_id,
        "sbml_source_path": str(result.spec.model_path),
        "sbml_sha256": _sha256(result.spec.model_path),
        "state_ids": list(result.spec.state_ids),
        "state_count": state_count,
        "simulation_horizon": result.simulation_time,
        "sampling_count": POINTS_PER_TRAJECTORY,
        "preflight_compatible": result.compatibility.compatible,
        "preflight_reasons": list(result.compatibility.reasons),
        "exhaustive_permitted": state_count <= EXHAUSTIVE_STATE_LIMIT,
        "exhaustive_planned": (
            phase in {"optional_exhaustive", "pretest"}
            and state_count <= EXHAUSTIVE_STATE_LIMIT
        ),
        "held_out_test_authorized": False,
        "artifact_checkpoints": {},
        "selection_identity": None,
        "selection_checkpoint": None,
        "failures": [],
        "exit_code": None,
        "stages": {
            "preflight": {
                "status": "completed",
                "updated_at_utc": utc_timestamp(),
            }
        },
    }


def _drain_events(event_queue: queue.Queue[dict[str, Any]], manifest: Any) -> None:
    """Apply worker events to the manifest on the coordinator thread."""
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            return
        if event.get("event") == "result":
            continue
        if all(key in event for key in ("model_id", "stage", "status")):
            details = {
                key: value
                for key, value in event.items()
                if key not in {"model_id", "stage", "status", "event"}
            }
            manifest.update_stage(
                model_id=str(event["model_id"]),
                stage=str(event["stage"]),
                status=str(event["status"]),
                **details,
            )
            if event["status"] == "running":
                model = manifest.payload["models"][str(event["model_id"])]
                model.setdefault("started_at_utc", utc_timestamp())
                manifest.update_model(str(event["model_id"]), status="running")
            checkpoint = event.get("artifact_checkpoint")
            if isinstance(checkpoint, dict):
                model = manifest.payload["models"][str(event["model_id"])]
                model.setdefault("artifact_checkpoints", {})[str(event["stage"])] = (
                    checkpoint
                )
                if event["stage"] == "selector":
                    model["selection_checkpoint"] = checkpoint
                manifest.write()


def execute_batch(
    *,
    model_ids: Sequence[str],
    variant: BenchmarkVariant,
    workers: int,
    phase: str,
    context: RunContext,
    telemetry_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    held_out_test_authorized: bool = False,
    preflight_runner: Callable[[str], PreflightResult] = run_preflight,
    output_stream: Any = sys.stdout,
) -> list[WorkerOutcome]:
    """Preflight and execute a bounded model-level subprocess batch.

    Args:
        model_ids: Unique model identifiers.
        variant: Authoritative benchmark variant.
        workers: Maximum active model subprocesses.
        phase: Explicit orchestration phase.
        context: Run filesystem context.
        telemetry_interval_seconds: Persistent monitoring interval.
        held_out_test_authorized: Explicit held-out authorization.
        preflight_runner: Existing preflight logic, injectable for tests.
        output_stream: Non-TTY-safe progress stream.

    Returns:
        Terminal outcomes for every started model.
    """
    models = validate_model_ids(model_ids)
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if telemetry_interval_seconds <= 0.0:
        raise ValueError("telemetry interval must be positive.")
    if phase == "test" and not held_out_test_authorized:
        raise RuntimeError("Batch held-out test requires explicit authorization.")

    resources = discover_resources()
    effective_workers = min(workers, len(models))
    model_records = {
        model_id: preflight_model_record(
            result=preflight_runner(model_id),
            phase=phase,
        )
        for model_id in models
    }
    for record in model_records.values():
        record["held_out_test_authorized"] = held_out_test_authorized

    if context.manifest_path.exists():
        manifest = load_run_manifest(context=context)
        if (
            manifest.payload.get("derivative_method") != variant.derivative_method.value
            or manifest.payload.get("experiment_namespace")
            != variant.experiment_namespace
        ):
            raise ValueError("Existing run manifest has a different benchmark variant.")
        existing_models = manifest.payload.get("models", {})
        if set(existing_models) != set(models):
            raise ValueError("Existing run manifest has a different model cohort.")
        invariant_fields = (
            "sbml_source_path",
            "sbml_sha256",
            "state_ids",
            "state_count",
            "simulation_horizon",
            "sampling_count",
            "preflight_compatible",
            "exhaustive_permitted",
        )
        for model_id in models:
            changed_fields = [
                field
                for field in invariant_fields
                if existing_models[model_id].get(field)
                != model_records[model_id].get(field)
            ]
            if changed_fields:
                raise ValueError(
                    f"Preflight invariants changed for {model_id}: "
                    f"{changed_fields}. Resume is blocked."
                )
        manifest.payload["requested_workers"] = workers
        manifest.payload["effective_workers"] = effective_workers
        manifest.payload["requested_phase"] = phase
        for record in existing_models.values():
            record["held_out_test_authorized"] = held_out_test_authorized
        manifest.write()
        model_records = existing_models
    else:
        manifest = create_run_manifest(
            context=context,
            variant=variant,
            requested_workers=workers,
            effective_workers=effective_workers,
            requested_phase=phase,
            resources=resources,
            thread_environment=THREAD_CONTROL_ENVIRONMENT,
            model_records=model_records,
        )
    registry = ActiveProcessRegistry()
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    reporter = PlainTextProgressReporter(output_stream)
    started_at = monotonic()
    samplers: dict[tuple[str, int], ProcessTreeSampler] = {}
    last_sample = 0.0

    expected_checkpoints: dict[str, ArtifactCheckpoint] = {}
    expected_stage_checkpoints: dict[str, dict[str, ArtifactCheckpoint]] = {}
    for model_id, record in model_records.items():
        raw_checkpoints = record.get("artifact_checkpoints", {})
        if isinstance(raw_checkpoints, dict):
            expected_stage_checkpoints[model_id] = {
                stage: ArtifactCheckpoint(**checkpoint)
                for stage, checkpoint in raw_checkpoints.items()
            }
    if phase == "test":
        for model_id, record in model_records.items():
            stored = record.get("pretest_checkpoint")
            if stored is None:
                raise RuntimeError(
                    f"Manifest lacks a pre-test checkpoint for {model_id}."
                )
            if not isinstance(stored, dict):
                raise TypeError(
                    f"Manifest pre-test checkpoint for {model_id} must be an object."
                )
            expected_checkpoints[model_id] = ArtifactCheckpoint(**stored)

    worker = SubprocessModelWorker(
        context=context,
        variant=variant,
        phase=phase,
        registry=registry,
        event_queue=events,
        expected_checkpoints=expected_checkpoints,
        expected_stage_checkpoints=expected_stage_checkpoints,
        held_out_test_authorized=held_out_test_authorized,
    )

    def tick(
        _active_futures: dict[str, Future[WorkerOutcome]],
        completed: list[WorkerOutcome],
    ) -> None:
        nonlocal last_sample
        _drain_events(events, manifest)
        elapsed = monotonic() - started_at
        if elapsed - last_sample < telemetry_interval_seconds:
            return
        last_sample = elapsed
        rows: list[dict[str, Any]] = []
        for active in registry.snapshot():
            key = (active.worker_id, active.process.pid)
            try:
                sampler = samplers.get(key)
                if sampler is None:
                    sampler = ProcessTreeSampler(
                        parent_pid=active.process.pid,
                        effective_cpu_capacity=float(
                            resources["effective_cpu_capacity"]
                        ),
                    )
                    sampler.prime()
                    samplers[key] = sampler
                row = sampler.sample(
                    context=WorkerTelemetryContext(
                        run_id=context.run_id,
                        worker_id=active.worker_id,
                        model_id=active.model_id,
                        stage=active.stage,
                        worker_state=active.state,
                    ),
                    elapsed_seconds=elapsed,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            telemetry.write(row)
            rows.append(row)
        failed = sum(outcome.status == "failed" for outcome in completed)
        completed_count = sum(outcome.status == "completed" for outcome in completed)
        reporter.report(
            completed=completed_count,
            total=len(models),
            failed=failed,
            active_rows=rows,
        )

    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)

        def handle_sigint(_signum: int, _frame: Any) -> None:
            stop_event.set()
            registry.terminate_all()

        signal.signal(signal.SIGINT, handle_sigint)

    try:
        with TelemetryWriter(context.telemetry_path) as telemetry:
            outcomes = schedule_models(
                model_ids=models,
                workers=effective_workers,
                worker=worker,
                stop_event=stop_event,
                on_tick=tick,
                poll_interval_seconds=min(0.25, telemetry_interval_seconds),
            )
            _drain_events(events, manifest)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)

    for outcome in outcomes:
        model_status = "failed"
        if outcome.status == "completed":
            if phase == "test":
                model_status = "test_complete"
            elif outcome.result and outcome.result.get("pretest_checkpoint"):
                model_status = "pretest_complete"
            else:
                model_status = "pending"
        updates: dict[str, Any] = {
            "status": model_status,
            "exit_code": outcome.exit_code,
            "ended_at_utc": utc_timestamp(),
        }
        if outcome.failure:
            updates["failures"] = [outcome.failure]
        if outcome.result:
            updates["worker_result"] = outcome.result
            checkpoint = outcome.result.get("pretest_checkpoint")
            if checkpoint:
                updates["pretest_checkpoint"] = checkpoint
            stage_checkpoints = outcome.result.get("stage_checkpoints")
            if stage_checkpoints:
                updates["artifact_checkpoints"] = stage_checkpoints
                if "selector" in stage_checkpoints:
                    updates["selection_checkpoint"] = stage_checkpoints["selector"]
            identity = outcome.result.get("selection_identity")
            if identity:
                updates["selection_identity"] = identity
        manifest.update_model(outcome.model_id, **updates)

    # Cohort-level aggregation belongs here only after every worker stopped.
    manifest.payload["batch_status"] = (
        "interrupted"
        if stop_event.is_set()
        else (
            "completed"
            if all(outcome.status == "completed" for outcome in outcomes)
            else "completed_with_failures"
        )
    )
    manifest.write()
    completed_count = sum(outcome.status == "completed" for outcome in outcomes)
    failed_count = sum(outcome.status == "failed" for outcome in outcomes)
    reporter.report(
        completed=completed_count,
        total=len(models),
        failed=failed_count,
        active_rows=[],
    )
    return outcomes


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the generic batch CLI.

    Args:
        arguments: Optional argument list for tests.

    Returns:
        Parsed batch configuration.
    """
    parser = argparse.ArgumentParser(
        description="Run independent benchmark models in bounded subprocesses."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--derivative-method",
        choices=("exact", "savgol"),
        default="exact",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=DEFAULT_SAMPLING_INTERVAL_SECONDS,
    )
    parser.add_argument("--authorize-held-out-test", action="store_true")
    return parser.parse_args(arguments)


def main() -> None:
    """Execute the generic package CLI."""
    arguments = parse_arguments()
    variant = get_benchmark_variant(arguments.derivative_method)
    run_id = arguments.run_id or (
        f"{variant.derivative_method.value}-"
        f"{utc_timestamp().replace(':', '').replace('+00:00', 'Z')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    context_arguments: dict[str, Any] = {"run_id": run_id}
    if arguments.runs_root is not None:
        context_arguments["runs_root"] = arguments.runs_root.resolve()
    context = RunContext(**context_arguments)
    model_ids = load_cohort_manifest(arguments.manifest)
    outcomes = execute_batch(
        model_ids=model_ids,
        variant=variant,
        workers=arguments.workers,
        phase=arguments.phase,
        context=context,
        telemetry_interval_seconds=arguments.telemetry_interval,
        held_out_test_authorized=arguments.authorize_held_out_test,
    )
    if len(outcomes) != len(model_ids) or any(
        outcome.status != "completed" for outcome in outcomes
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
