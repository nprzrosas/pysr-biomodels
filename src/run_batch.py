"""Bounded subprocess orchestration for independent benchmark models."""

import argparse
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import psutil

from .benchmark_variant import BenchmarkVariant, get_benchmark_variant
from .preflight import PreflightResult, run_preflight
from .resource_monitor import (
    BYTES_PER_GIBIBYTE,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    MemoryPressureGuard,
    MemorySafetyPolicy,
    PlainTextProgressReporter,
    ProcessTreeSampler,
    TelemetryWriter,
    WorkerTelemetryContext,
    discover_resources,
    memory_snapshot,
    validate_prelaunch_memory,
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
class PerModelResourcePolicy:
    """Operational resource limits applied independently to each worker."""

    maximum_process_tree_rss_bytes: int = 0
    memory_consecutive_samples: int = 2
    wall_clock_timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate disabled-or-positive limits and the persistence count."""
        if (
            not isinstance(self.maximum_process_tree_rss_bytes, int)
            or isinstance(self.maximum_process_tree_rss_bytes, bool)
            or self.maximum_process_tree_rss_bytes < 0
        ):
            raise ValueError("Per-model RSS limit must be a non-negative integer.")
        if (
            not isinstance(self.memory_consecutive_samples, int)
            or isinstance(self.memory_consecutive_samples, bool)
            or self.memory_consecutive_samples < 1
        ):
            raise ValueError(
                "Per-model memory sample count must be a positive integer."
            )
        if (
            not isinstance(self.wall_clock_timeout_seconds, (int, float))
            or isinstance(self.wall_clock_timeout_seconds, bool)
            or not math.isfinite(self.wall_clock_timeout_seconds)
            or self.wall_clock_timeout_seconds < 0.0
        ):
            raise ValueError("Per-model timeout must be finite and non-negative.")


@dataclass(frozen=True)
class WorkerFailureInfo:
    """Structured operational details for one worker failure."""

    failure_kind: str
    last_stage: str | None = None
    peak_process_tree_rss_bytes: int | None = None
    configured_process_tree_rss_limit_bytes: int | None = None
    consecutive_over_limit_samples: int | None = None
    elapsed_worker_seconds: float | None = None
    configured_timeout_seconds: float | None = None


@dataclass
class _WorkerResourceState:
    """Mutable sampling state isolated to one worker identifier."""

    started_at_monotonic: float
    peak_process_tree_rss_bytes: int = 0
    consecutive_over_limit_samples: int = 0
    failure_info: WorkerFailureInfo | None = None


class PerModelResourceGuard:
    """Evaluate per-model operational limits using coordinator telemetry."""

    def __init__(self, policy: PerModelResourcePolicy) -> None:
        """Initialize independent worker counters.

        Args:
            policy: Per-model operational limits.
        """
        self.policy = policy
        self._lock = threading.Lock()
        self._states: dict[str, _WorkerResourceState] = {}
        self._finished_worker_ids: set[str] = set()

    def register_worker(self, worker_id: str, started_at_monotonic: float) -> None:
        """Record the actual launch time for a newly created subprocess.

        Args:
            worker_id: Batch-unique worker identifier.
            started_at_monotonic: Monotonic timestamp captured after launch.
        """
        with self._lock:
            self._finished_worker_ids.discard(worker_id)
            self._states[worker_id] = _WorkerResourceState(
                started_at_monotonic=started_at_monotonic
            )

    def observe(
        self,
        *,
        worker_id: str,
        stage: str,
        process_tree_rss_bytes: int | None,
        observed_at_monotonic: float,
        started_at_monotonic: float,
    ) -> WorkerFailureInfo | None:
        """Evaluate one worker sample and return a newly triggered failure.

        Args:
            worker_id: Batch-unique worker identifier.
            stage: Current operational stage label.
            process_tree_rss_bytes: Recursive RSS measurement, when available.
            observed_at_monotonic: Monotonic time of this telemetry tick.
            started_at_monotonic: Actual worker launch time.

        Returns:
            Newly triggered resource failure, or ``None``.
        """
        with self._lock:
            if worker_id in self._finished_worker_ids:
                return None
            state = self._states.setdefault(
                worker_id,
                _WorkerResourceState(started_at_monotonic=started_at_monotonic),
            )
            if state.failure_info is not None:
                return None

            if process_tree_rss_bytes is not None:
                rss_bytes = max(0, int(process_tree_rss_bytes))
                state.peak_process_tree_rss_bytes = max(
                    state.peak_process_tree_rss_bytes,
                    rss_bytes,
                )
                rss_limit = self.policy.maximum_process_tree_rss_bytes
                if rss_limit > 0 and rss_bytes > rss_limit:
                    state.consecutive_over_limit_samples += 1
                else:
                    state.consecutive_over_limit_samples = 0

                if (
                    rss_limit > 0
                    and state.consecutive_over_limit_samples
                    >= self.policy.memory_consecutive_samples
                ):
                    state.failure_info = WorkerFailureInfo(
                        failure_kind="memory_limit_exceeded",
                        last_stage=stage,
                        peak_process_tree_rss_bytes=(state.peak_process_tree_rss_bytes),
                        configured_process_tree_rss_limit_bytes=rss_limit,
                        consecutive_over_limit_samples=(
                            state.consecutive_over_limit_samples
                        ),
                        elapsed_worker_seconds=max(
                            0.0,
                            observed_at_monotonic - state.started_at_monotonic,
                        ),
                        configured_timeout_seconds=(
                            self.policy.wall_clock_timeout_seconds
                        ),
                    )
                    return state.failure_info

            elapsed_seconds = max(
                0.0,
                observed_at_monotonic - state.started_at_monotonic,
            )
            timeout_seconds = self.policy.wall_clock_timeout_seconds
            if timeout_seconds > 0.0 and elapsed_seconds >= timeout_seconds:
                state.failure_info = WorkerFailureInfo(
                    failure_kind="timeout_exceeded",
                    last_stage=stage,
                    peak_process_tree_rss_bytes=(state.peak_process_tree_rss_bytes),
                    configured_process_tree_rss_limit_bytes=(
                        self.policy.maximum_process_tree_rss_bytes
                    ),
                    consecutive_over_limit_samples=(
                        state.consecutive_over_limit_samples
                    ),
                    elapsed_worker_seconds=elapsed_seconds,
                    configured_timeout_seconds=timeout_seconds,
                )
                return state.failure_info
            return None

    def finish_worker(self, worker_id: str) -> WorkerFailureInfo | None:
        """Remove one worker's counters and return any censoring record.

        Args:
            worker_id: Completed worker identifier.

        Returns:
            Recorded resource failure, if the worker was censored.
        """
        with self._lock:
            state = self._states.pop(worker_id, None)
            self._finished_worker_ids.add(worker_id)
        return None if state is None else state.failure_info


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
    failure_info: WorkerFailureInfo | None = None


@dataclass(frozen=True)
class ActiveWorker:
    """Live subprocess metadata used by telemetry and interruption."""

    worker_id: str
    model_id: str
    stage: str
    state: str
    process: subprocess.Popen[str]
    started_at_monotonic: float = field(default_factory=monotonic)


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
                    started_at_monotonic=current.started_at_monotonic,
                )

    def remove(self, worker_id: str) -> None:
        """Remove a completed worker."""
        with self._lock:
            self._workers.pop(worker_id, None)

    def snapshot(self) -> tuple[ActiveWorker, ...]:
        """Return an immutable snapshot of active workers."""
        with self._lock:
            return tuple(self._workers.values())

    def terminate_worker(
        self,
        worker_id: str,
        grace_seconds: float = 10.0,
    ) -> None:
        """Terminate only one owned worker process group.

        Args:
            worker_id: Registered worker identifier to terminate.
            grace_seconds: Time allowed for graceful termination before kill.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
        if worker is not None:
            self._terminate_workers((worker,), grace_seconds)

    def terminate_all(self, grace_seconds: float = 10.0) -> None:
        """Terminate all owned process groups, escalating after a grace period.

        Args:
            grace_seconds: Time allowed for graceful termination before kill.
        """
        self._terminate_workers(self.snapshot(), grace_seconds)

    @staticmethod
    def _terminate_workers(
        workers: Sequence[ActiveWorker],
        grace_seconds: float,
    ) -> None:
        """Apply shared TERM/grace/KILL behavior to explicit owned workers.

        Args:
            workers: Exact registered workers whose process groups are owned.
            grace_seconds: Time allowed for graceful termination before kill.
        """
        workers = tuple(workers)
        for worker in workers:
            process = worker.process
            if not ActiveProcessRegistry._worker_group_is_alive(worker):
                continue
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                continue

        deadline = monotonic() + max(0.0, grace_seconds)
        while monotonic() < deadline:
            if all(
                not ActiveProcessRegistry._worker_group_is_alive(worker)
                for worker in workers
            ):
                return
            sleep(0.05)

        for worker in workers:
            process = worker.process
            if not ActiveProcessRegistry._worker_group_is_alive(worker):
                continue
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                continue

    @staticmethod
    def _worker_group_is_alive(worker: ActiveWorker) -> bool:
        """Report whether an owned worker or POSIX process group remains.

        Args:
            worker: Registered worker whose launch created the process group.

        Returns:
            Whether the owned termination target still exists.
        """
        if os.name != "posix":
            return worker.process.poll() is None
        try:
            os.killpg(worker.process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


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
        resource_guard: PerModelResourceGuard | None = None,
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
            resource_guard: Independent per-model operational limit guard.
            expected_checkpoints: Pre-test checkpoints keyed by model.
            expected_stage_checkpoints: Verified stage checkpoints by model.
            held_out_test_authorized: Explicit held-out authorization.
        """
        self.context = context
        self.variant = variant
        self.phase = phase
        self.registry = registry
        self.resource_guard = resource_guard or PerModelResourceGuard(
            PerModelResourcePolicy()
        )
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
            last_stage = self.phase

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
                launched_at = monotonic()
                self.resource_guard.register_worker(worker_id, launched_at)
                self.registry.add(
                    ActiveWorker(
                        worker_id=worker_id,
                        model_id=model_id,
                        stage="starting",
                        state="running",
                        process=process,
                        started_at_monotonic=launched_at,
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
                        last_stage = str(event["stage"])
                        self.registry.update_stage(
                            worker_id,
                            str(event["stage"]),
                            str(event.get("status", "running")),
                        )
                exit_code = process.wait()

            resource_failure = self.resource_guard.finish_worker(worker_id)
            if resource_failure is not None:
                return WorkerOutcome(
                    model_id=model_id,
                    worker_id=worker_id,
                    status="resource_censored",
                    exit_code=exit_code,
                    result=result_payload,
                    failure=(
                        "Worker was resource-censored: "
                        f"{resource_failure.failure_kind}."
                    ),
                    failure_info=resource_failure,
                )
            status = "completed" if exit_code == 0 else "failed"
            failure_info = None
            if exit_code != 0:
                failure_info = WorkerFailureInfo(
                    failure_kind="worker_nonzero_exit",
                    last_stage=last_stage,
                    elapsed_worker_seconds=max(0.0, monotonic() - launched_at),
                )
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status=status,
                exit_code=exit_code,
                result=result_payload,
                failure=None if exit_code == 0 else f"Worker exited {exit_code}.",
                failure_info=failure_info,
            )
        finally:
            self.registry.remove(worker_id)
            self.resource_guard.finish_worker(worker_id)
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
    memory_safety_policy: MemorySafetyPolicy | None = None,
    per_model_resource_policy: PerModelResourcePolicy | None = None,
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
        memory_safety_policy: Optional pre-launch and runtime memory gates.
        per_model_resource_policy: Independent operational worker limits.
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

    memory_safety_policy = memory_safety_policy or MemorySafetyPolicy()
    per_model_resource_policy = per_model_resource_policy or PerModelResourcePolicy()
    resources = discover_resources()
    validate_prelaunch_memory(memory_safety_policy, resources)
    effective_workers = min(workers, len(models))
    model_records = {
        model_id: preflight_model_record(
            result=preflight_runner(model_id),
            phase=phase,
        )
        for model_id in models
    }
    for record in model_records.values():
        record["held_out_test_authorized"] = False

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
        if phase != "test":
            for record in existing_models.values():
                record["held_out_test_authorized"] = False
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
    memory_safety = manifest.payload.setdefault("memory_safety", {})
    memory_safety.update(
        {
            "minimum_launch_available_bytes": (
                memory_safety_policy.minimum_launch_available_bytes
            ),
            "emergency_available_bytes": (
                memory_safety_policy.emergency_available_bytes
            ),
            "emergency_consecutive_samples": (
                memory_safety_policy.emergency_consecutive_samples
            ),
            "prelaunch_snapshot": resources,
        }
    )
    memory_safety.setdefault("events", [])
    manifest.payload["per_model_resource_policy"] = asdict(per_model_resource_policy)
    manifest.write()
    registry = ActiveProcessRegistry()
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    reporter = PlainTextProgressReporter(output_stream)
    started_at = monotonic()
    samplers: dict[tuple[str, int], ProcessTreeSampler] = {}
    last_sample = 0.0
    memory_guard = MemoryPressureGuard(memory_safety_policy)
    per_model_resource_guard = PerModelResourceGuard(per_model_resource_policy)
    memory_emergency_triggered = False

    expected_checkpoints: dict[str, ArtifactCheckpoint] = {}
    expected_stage_checkpoints: dict[str, dict[str, ArtifactCheckpoint]] = {}
    for model_id, record in model_records.items():
        raw_checkpoints = record.get("artifact_checkpoints", {})
        if isinstance(raw_checkpoints, dict):
            expected_stage_checkpoints[model_id] = {
                stage: ArtifactCheckpoint(**checkpoint)
                for stage, checkpoint in raw_checkpoints.items()
            }

    runnable_models = list(models)

    if phase == "test":
        runnable_models = []
        for model_id in models:
            record = model_records[model_id]
            stored = record.get("pretest_checkpoint")

            if (
                record.get("status") == "pretest_complete"
                and isinstance(stored, dict)
            ):
                expected_checkpoints[model_id] = ArtifactCheckpoint(**stored)
                runnable_models.append(model_id)

        if not runnable_models:
            raise RuntimeError(
                "Held-out test has no eligible models with a completed "
                "pre-test checkpoint."
            )

        manifest.payload["held_out_test_eligibility"] = {
            "original_model_count": len(models),
            "eligible_model_count": len(runnable_models),
            "excluded_model_count": len(models) - len(runnable_models),
            "eligible_models": list(runnable_models),
            "excluded_models": [
                model_id
                for model_id in models
                if model_id not in runnable_models
            ],
        }

        runnable_model_set = set(runnable_models)
        for model_id in models:
            model_records[model_id]["held_out_test_authorized"] = (
                model_id in runnable_model_set
            )

    effective_workers = min(workers, len(runnable_models))
    manifest.payload["effective_workers"] = effective_workers
    manifest.write()

    worker = SubprocessModelWorker(
        context=context,
        variant=variant,
        phase=phase,
        registry=registry,
        resource_guard=per_model_resource_guard,
        event_queue=events,
        expected_checkpoints=expected_checkpoints,
        expected_stage_checkpoints=expected_stage_checkpoints,
        held_out_test_authorized=held_out_test_authorized,
    )

    def tick(
        _active_futures: dict[str, Future[WorkerOutcome]],
        completed: list[WorkerOutcome],
    ) -> None:
        nonlocal last_sample, memory_emergency_triggered
        _drain_events(events, manifest)
        elapsed = monotonic() - started_at
        if elapsed - last_sample < telemetry_interval_seconds:
            return
        last_sample = elapsed
        rows: list[dict[str, Any]] = []
        sampled_workers: list[tuple[ActiveWorker, int | None]] = []
        for active in registry.snapshot():
            key = (active.worker_id, active.process.pid)
            process_tree_rss_bytes: int | None = None
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
                pass
            else:
                telemetry.write(row)
                rows.append(row)
                process_tree_rss_bytes = int(row["process_tree_rss_bytes"])
            sampled_workers.append((active, process_tree_rss_bytes))
        failed = sum(outcome.status != "completed" for outcome in completed)
        completed_count = sum(outcome.status == "completed" for outcome in completed)
        reporter.report(
            completed=completed_count,
            total=len(models),
            failed=failed,
            active_rows=rows,
        )
        current_memory = memory_snapshot()
        available = int(current_memory["effective_memory_available_bytes"] or 0)
        if not memory_emergency_triggered and memory_guard.observe(available):
            memory_emergency_triggered = True
            stop_event.set()
            event = {
                "timestamp_utc": utc_timestamp(),
                "event": "persistent_low_available_memory",
                "effective_memory_available_bytes": available,
                "emergency_available_bytes": (
                    memory_safety_policy.emergency_available_bytes
                ),
                "consecutive_samples": memory_guard.consecutive_low_samples,
                "action": "stop_scheduling_and_terminate_owned_workers",
            }
            manifest.payload["memory_safety"]["events"].append(event)
            manifest.payload["batch_status"] = "memory_safety_stopping"
            manifest.write()
            registry.terminate_all()
            return

        if memory_emergency_triggered:
            return
        for active, process_tree_rss_bytes in sampled_workers:
            resource_failure = per_model_resource_guard.observe(
                worker_id=active.worker_id,
                stage=active.stage,
                process_tree_rss_bytes=process_tree_rss_bytes,
                observed_at_monotonic=monotonic(),
                started_at_monotonic=active.started_at_monotonic,
            )
            if resource_failure is not None:
                registry.terminate_worker(active.worker_id)

    previous_handlers: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():

        def handle_termination(_signum: int, _frame: Any) -> None:
            """Stop scheduling and terminate only this batch's workers."""
            stop_event.set()
            registry.terminate_all()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_termination)

    try:
        with TelemetryWriter(context.telemetry_path) as telemetry:
            outcomes = schedule_models(
                model_ids=runnable_models,
                workers=effective_workers,
                worker=worker,
                stop_event=stop_event,
                on_tick=tick,
                poll_interval_seconds=min(0.25, telemetry_interval_seconds),
            )
            _drain_events(events, manifest)
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)

    for outcome in outcomes:
        model_status = "failed"
        if outcome.status == "resource_censored":
            model_status = "resource_censored"
        elif outcome.status == "completed":
            if phase == "test":
                model_status = "test_complete"
            elif outcome.result and outcome.result.get("pretest_checkpoint"):
                model_status = "pretest_complete"
            else:
                model_status = "pending"
        manifest.update_stage(
            model_id=outcome.model_id,
            stage=phase,
            status=outcome.status,
            worker_id=outcome.worker_id,
            exit_code=outcome.exit_code,
            failure_kind=(
                outcome.failure_info.failure_kind
                if outcome.failure_info is not None
                else None
            ),
        )

        if (
            outcome.status == "resource_censored"
            and outcome.failure_info is not None
            and outcome.failure_info.last_stage
            and outcome.failure_info.last_stage != phase
        ):
            manifest.update_stage(
                model_id=outcome.model_id,
                stage=outcome.failure_info.last_stage,
                status="resource_censored",
                worker_id=outcome.worker_id,
                exit_code=outcome.exit_code,
                failure_kind=outcome.failure_info.failure_kind,
            )

        updates: dict[str, Any] = {
            "status": model_status,
            "exit_code": outcome.exit_code,
            "ended_at_utc": utc_timestamp(),
        }
        if outcome.failure:
            if outcome.status == "resource_censored":
                updates["operational_failures"] = [outcome.failure]
            else:
                updates["failures"] = [outcome.failure]
        if outcome.failure_info is not None:
            updates["failure_info"] = asdict(outcome.failure_info)
            updates["failure_kind"] = outcome.failure_info.failure_kind
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
        "memory_safety_stopped"
        if memory_emergency_triggered
        else (
            "interrupted"
            if stop_event.is_set()
            else (
                "orchestration_failed"
                if len(outcomes) != len(runnable_models)
                or any(outcome.global_invariant_failure for outcome in outcomes)
                else (
                    "completed"
                    if all(outcome.status == "completed" for outcome in outcomes)
                    else "completed_with_failures"
                )
            )
        )
    )
    manifest.write()
    completed_count = sum(outcome.status == "completed" for outcome in outcomes)
    failed_count = sum(outcome.status != "completed" for outcome in outcomes)
    reporter.report(
        completed=completed_count,
        total=len(runnable_models),
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
    parser.add_argument(
        "--min-available-memory-gib",
        type=_nonnegative_float,
        default=0.0,
        help="Block launch below this effective available-memory floor.",
    )
    parser.add_argument(
        "--emergency-available-memory-gib",
        type=_nonnegative_float,
        default=0.0,
        help="Gracefully stop after persistent runtime memory pressure.",
    )
    parser.add_argument(
        "--memory-emergency-samples",
        type=_positive_int,
        default=2,
        help="Consecutive low-memory samples required before stopping.",
    )
    parser.add_argument(
        "--per-model-memory-gib",
        type=_nonnegative_float,
        default=0.0,
        help="Maximum recursive RSS per model; zero disables the limit.",
    )
    parser.add_argument(
        "--per-model-memory-samples",
        type=_positive_int,
        default=2,
        help="Consecutive over-limit RSS samples required per model.",
    )
    parser.add_argument(
        "--per-model-timeout-minutes",
        type=_nonnegative_float,
        default=0.0,
        help="Wall-clock timeout from each model's launch; zero disables it.",
    )
    parser.add_argument(
        "--model-failure-exit-policy",
        choices=("error", "success"),
        default="error",
        help=(
            "Use 'success' to return zero for completed_with_failures; "
            "interrupted and memory-safety-stopped batches remain nonzero."
        ),
    )
    return parser.parse_args(arguments)


def _nonnegative_float(value: str) -> float:
    """Parse one finite non-negative CLI number.

    Args:
        value: Raw command-line value.

    Returns:
        Validated floating-point value.

    Raises:
        argparse.ArgumentTypeError: If the value is invalid.
    """
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_int(value: str) -> int:
    """Parse one positive CLI integer.

    Args:
        value: Raw command-line value.

    Returns:
        Validated integer value.

    Raises:
        argparse.ArgumentTypeError: If the value is below one.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def batch_exit_code(batch_status: str, model_failure_exit_policy: str) -> int:
    """Map an authoritative manifest status to the explicit CLI exit policy.

    Args:
        batch_status: Final coordinator status from the run manifest.
        model_failure_exit_policy: ``error`` or ``success`` for model failures.

    Returns:
        Process exit code zero or one.
    """
    if model_failure_exit_policy not in {"error", "success"}:
        raise ValueError("Unknown model failure exit policy.")
    if batch_status == "completed":
        return 0
    if (
        batch_status == "completed_with_failures"
        and model_failure_exit_policy == "success"
    ):
        return 0
    return 1


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
    execute_batch(
        model_ids=model_ids,
        variant=variant,
        workers=arguments.workers,
        phase=arguments.phase,
        context=context,
        telemetry_interval_seconds=arguments.telemetry_interval,
        held_out_test_authorized=arguments.authorize_held_out_test,
        memory_safety_policy=MemorySafetyPolicy(
            minimum_launch_available_bytes=int(
                arguments.min_available_memory_gib * BYTES_PER_GIBIBYTE
            ),
            emergency_available_bytes=int(
                arguments.emergency_available_memory_gib * BYTES_PER_GIBIBYTE
            ),
            emergency_consecutive_samples=arguments.memory_emergency_samples,
        ),
        per_model_resource_policy=PerModelResourcePolicy(
            maximum_process_tree_rss_bytes=int(
                arguments.per_model_memory_gib * BYTES_PER_GIBIBYTE
            ),
            memory_consecutive_samples=arguments.per_model_memory_samples,
            wall_clock_timeout_seconds=(arguments.per_model_timeout_minutes * 60.0),
        ),
    )
    manifest = load_run_manifest(context=context)
    manifest.payload["cli_model_failure_exit_policy"] = (
        arguments.model_failure_exit_policy
    )
    manifest.write()
    status = str(manifest.payload.get("batch_status", "unknown"))
    exit_code = batch_exit_code(status, arguments.model_failure_exit_policy)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
