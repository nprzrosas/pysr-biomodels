"""Tests for bounded generic batch scheduling and isolation."""

import io
import json
import os
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
)
from src.resource_monitor import MemorySafetyPolicy
from src.run_batch import (
    THREAD_CONTROL_ENVIRONMENT,
    ActiveProcessRegistry,
    ActiveWorker,
    PerModelResourceGuard,
    PerModelResourcePolicy,
    WorkerFailureInfo,
    WorkerOutcome,
    batch_exit_code,
    execute_batch,
    model_worker_command,
    parse_arguments,
    schedule_models,
    validate_model_ids,
    worker_environment,
)
from src.run_manifest import ModelVariantLock, RunContext


class BatchSchedulerTests(unittest.TestCase):
    """Exercise concurrency, failure isolation, and interruption semantics."""

    def test_workers_one_runs_every_model_serially(self) -> None:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        outcomes = schedule_models(model_ids=("A", "B", "C"), workers=1, worker=worker)
        self.assertEqual(maximum, 1)
        self.assertEqual([item.model_id for item in outcomes], ["A", "B", "C"])

    def test_parallel_scheduler_never_exceeds_worker_bound(self) -> None:
        active = 0
        maximum = 0
        observed_overlap = threading.Event()
        lock = threading.Lock()

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active > 1:
                    observed_overlap.set()
            time.sleep(0.08)
            with lock:
                active -= 1
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        schedule_models(model_ids=("A", "B", "C", "D"), workers=2, worker=worker)
        self.assertTrue(observed_overlap.is_set())
        self.assertLessEqual(maximum, 2)

    def test_worker_count_does_not_change_deterministic_outcomes(self) -> None:
        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status="completed",
                exit_code=0,
                result={"artifact": f"frozen-{model_id}"},
            )

        models = ("A", "B", "C")
        serial = schedule_models(model_ids=models, workers=1, worker=worker)
        parallel = schedule_models(model_ids=models, workers=3, worker=worker)
        self.assertEqual(
            [(item.model_id, item.result) for item in serial],
            [(item.model_id, item.result) for item in parallel],
        )

    def test_one_failed_model_does_not_stop_unrelated_models(self) -> None:
        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            if model_id == "B":
                return WorkerOutcome(model_id, worker_id, "failed", 7)
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        outcomes = schedule_models(model_ids=("A", "B", "C"), workers=2, worker=worker)
        self.assertEqual(
            [item.status for item in outcomes],
            [
                "completed",
                "failed",
                "completed",
            ],
        )

    def test_global_invariant_failure_stops_new_scheduling(self) -> None:
        started: list[str] = []

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            started.append(model_id)
            return WorkerOutcome(
                model_id,
                worker_id,
                "failed",
                1,
                global_invariant_failure=True,
            )

        schedule_models(model_ids=("A", "B", "C"), workers=1, worker=worker)
        self.assertEqual(started, ["A"])

    def test_stop_event_prevents_new_scheduling(self) -> None:
        stop_event = threading.Event()
        started: list[str] = []

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            started.append(model_id)
            stop_event.set()
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        schedule_models(
            model_ids=("A", "B", "C"),
            workers=1,
            worker=worker,
            stop_event=stop_event,
        )
        self.assertEqual(started, ["A"])

    def test_duplicate_model_scheduling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_model_ids(("A", "A"))

    def test_model_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe path components"):
            validate_model_ids(("../results",))

    def test_single_worker_termination_targets_only_requested_group(self) -> None:
        registry = ActiveProcessRegistry()
        first = SimpleNamespace(pid=101, poll=lambda: None)
        second = SimpleNamespace(pid=202, poll=lambda: None)
        for worker_id, process in (("1", first), ("2", second)):
            registry.add(
                ActiveWorker(
                    worker_id=worker_id,
                    model_id=f"MODEL_{worker_id}",
                    stage="candidates",
                    state="running",
                    process=process,
                    started_at_monotonic=time.monotonic(),
                )
            )

        with patch("src.run_batch.os.killpg") as kill_process_group:
            registry.terminate_worker("1", grace_seconds=0.0)

        if os.name == "posix":
            signal_calls = [
                item for item in kill_process_group.call_args_list if item.args[1] != 0
            ]
            self.assertEqual(
                signal_calls,
                [call(101, signal.SIGTERM), call(101, signal.SIGKILL)],
            )


class PerModelResourceGuardTests(unittest.TestCase):
    """Verify independent persistence counters and disabled limits."""

    @staticmethod
    def observe(
        guard: PerModelResourceGuard,
        worker_id: str,
        rss_bytes: int,
        observed_at: float,
    ) -> WorkerFailureInfo | None:
        """Send one deterministic worker sample to a guard."""
        return guard.observe(
            worker_id=worker_id,
            stage="candidates",
            process_tree_rss_bytes=rss_bytes,
            observed_at_monotonic=observed_at,
            started_at_monotonic=0.0,
        )

    def test_one_sample_spike_does_not_trigger_two_sample_policy(self) -> None:
        guard = PerModelResourceGuard(
            PerModelResourcePolicy(
                maximum_process_tree_rss_bytes=100,
                memory_consecutive_samples=2,
            )
        )
        self.assertIsNone(self.observe(guard, "1", 101, 1.0))
        self.assertIsNone(self.observe(guard, "1", 100, 2.0))

    def test_below_limit_sample_resets_only_that_worker_counter(self) -> None:
        guard = PerModelResourceGuard(
            PerModelResourcePolicy(
                maximum_process_tree_rss_bytes=100,
                memory_consecutive_samples=2,
            )
        )
        self.assertIsNone(self.observe(guard, "1", 101, 1.0))
        self.assertIsNone(self.observe(guard, "2", 101, 1.0))
        self.assertIsNone(self.observe(guard, "1", 50, 2.0))
        second_failure = self.observe(guard, "2", 102, 2.0)
        self.assertIsNotNone(second_failure)
        self.assertIsNone(self.observe(guard, "1", 103, 3.0))
        first_failure = self.observe(guard, "1", 104, 4.0)
        self.assertEqual(first_failure.failure_kind, "memory_limit_exceeded")

    def test_disabled_limits_never_trigger(self) -> None:
        guard = PerModelResourceGuard(PerModelResourcePolicy())
        self.assertIsNone(self.observe(guard, "1", 10**15, 10**9))

    def test_policy_rejects_negative_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            PerModelResourcePolicy(maximum_process_tree_rss_bytes=-1)
        with self.assertRaises(ValueError):
            PerModelResourcePolicy(memory_consecutive_samples=0)
        with self.assertRaises(ValueError):
            PerModelResourcePolicy(wall_clock_timeout_seconds=-1.0)


class BatchIsolationTests(unittest.TestCase):
    """Verify lock, namespace, command, and environment isolation."""

    def setUp(self) -> None:
        """Create an isolated run context."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.context = RunContext(
            run_id="batch-test",
            project_root=root,
            results_root=root / "results",
            runs_root=root / "runs",
            cache_root=root / "cache",
        )

    def _execute_resource_limited_batch(
        self,
        *,
        run_id: str,
        policy: PerModelResourcePolicy,
        rss_bytes: int,
    ) -> tuple[list[WorkerOutcome], dict[str, object], list[str], list[str]]:
        """Execute two fake workers through the real coordinator tick.

        Args:
            run_id: Isolated run identifier.
            policy: Per-model limits under test.
            rss_bytes: Recursive RSS returned for the first model.

        Returns:
            Outcomes, persisted manifest, launch order, and terminated workers.
        """
        sbml_path = Path(self.temporary_directory.name) / f"{run_id}.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")
        context = RunContext(
            run_id=run_id,
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )
        started: list[str] = []
        terminated: list[str] = []
        termination_events: dict[str, threading.Event] = {}

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A",),
                    n_states=1,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=1.0,
            )

        def worker_factory(**keywords: object):
            registry = keywords["registry"]
            resource_guard = keywords["resource_guard"]

            def worker(model_id: str, worker_id: str) -> WorkerOutcome:
                started.append(model_id)
                launched_at = time.monotonic()
                termination_event = threading.Event()
                termination_events[worker_id] = termination_event
                process = SimpleNamespace(pid=1000 + int(worker_id))
                registry.add(
                    ActiveWorker(
                        worker_id=worker_id,
                        model_id=model_id,
                        stage="candidates",
                        state="running",
                        process=process,
                        started_at_monotonic=launched_at,
                    )
                )
                resource_guard.register_worker(worker_id, launched_at)
                if model_id == "MODEL_A":
                    if not termination_event.wait(timeout=2.0):
                        registry.remove(worker_id)
                        resource_guard.finish_worker(worker_id)
                        return WorkerOutcome(
                            model_id,
                            worker_id,
                            "failed",
                            1,
                            failure="Coordinator did not terminate worker.",
                        )
                    failure_info = resource_guard.finish_worker(worker_id)
                    registry.remove(worker_id)
                    return WorkerOutcome(
                        model_id=model_id,
                        worker_id=worker_id,
                        status="resource_censored",
                        exit_code=-signal.SIGTERM,
                        failure="Worker was resource-censored.",
                        failure_info=failure_info,
                    )
                registry.remove(worker_id)
                resource_guard.finish_worker(worker_id)
                return WorkerOutcome(model_id, worker_id, "completed", 0)

            return worker

        class FakeSampler:
            """Return deterministic telemetry without allocating memory."""

            def __init__(self, *, parent_pid: int, **_keywords: object) -> None:
                self.parent_pid = parent_pid

            def prime(self) -> None:
                """Match the production sampler interface."""

            def sample(self, *, context: object, elapsed_seconds: float):
                """Build the reporter and CSV fields used by the coordinator."""
                worker_rss = rss_bytes if context.model_id == "MODEL_A" else 0
                return {
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "elapsed_seconds": elapsed_seconds,
                    "run_id": context.run_id,
                    "worker_id": context.worker_id,
                    "model_id": context.model_id,
                    "stage": context.stage,
                    "worker_state": context.worker_state,
                    "parent_pid": self.parent_pid,
                    "child_pids": "",
                    "process_tree_cpu_percent_raw": 0.0,
                    "process_tree_cpu_percent_capacity_normalized": 0.0,
                    "process_tree_rss_bytes": worker_rss,
                    "system_cpu_percent": 0.0,
                    "system_memory_used_bytes": 0,
                    "system_memory_available_bytes": 1_000_000,
                    "effective_memory_available_bytes": 1_000_000,
                    "system_memory_percent": 0.0,
                    "system_swap_total_bytes": 0,
                    "system_swap_used_bytes": 0,
                }

        def terminate_worker(
            _registry: ActiveProcessRegistry,
            worker_id: str,
            _grace_seconds: float = 10.0,
        ) -> None:
            terminated.append(worker_id)
            termination_events[worker_id].set()

        resources = {
            "effective_memory_available_bytes": 1_000_000,
            "effective_cpu_capacity": 1.0,
        }
        with (
            patch("src.run_batch.discover_resources", return_value=resources),
            patch("src.run_batch.ProcessTreeSampler", FakeSampler),
            patch(
                "src.run_batch.SubprocessModelWorker",
                side_effect=worker_factory,
            ),
            patch(
                "src.run_batch.ActiveProcessRegistry.terminate_worker",
                autospec=True,
                side_effect=terminate_worker,
            ),
        ):
            outcomes = execute_batch(
                model_ids=("MODEL_A", "MODEL_B"),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="preflight",
                context=context,
                telemetry_interval_seconds=0.001,
                per_model_resource_policy=policy,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )
        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        return outcomes, payload, started, terminated

    def test_memory_censoring_terminates_one_worker_and_continues(self) -> None:
        outcomes, payload, started, terminated = self._execute_resource_limited_batch(
            run_id="per-model-memory-test",
            policy=PerModelResourcePolicy(
                maximum_process_tree_rss_bytes=100,
                memory_consecutive_samples=1,
            ),
            rss_bytes=101,
        )
        self.assertEqual(started, ["MODEL_A", "MODEL_B"])
        self.assertEqual(terminated, ["1"])
        self.assertEqual(
            [outcome.status for outcome in outcomes],
            ["resource_censored", "completed"],
        )
        self.assertEqual(outcomes[0].exit_code, -signal.SIGTERM)
        self.assertEqual(payload["batch_status"], "completed_with_failures")
        self.assertEqual(
            payload["per_model_resource_policy"]["maximum_process_tree_rss_bytes"],
            100,
        )
        record = payload["models"]["MODEL_A"]
        self.assertEqual(record["status"], "resource_censored")
        self.assertEqual(record["failure_kind"], "memory_limit_exceeded")
        self.assertEqual(
            record["failure_info"]["peak_process_tree_rss_bytes"],
            101,
        )
        self.assertIn("operational_failures", record)
        self.assertEqual(record["failures"], [])

    def test_timeout_terminates_one_worker_and_continues(self) -> None:
        outcomes, payload, started, terminated = self._execute_resource_limited_batch(
            run_id="per-model-timeout-test",
            policy=PerModelResourcePolicy(
                wall_clock_timeout_seconds=0.01,
            ),
            rss_bytes=0,
        )
        self.assertEqual(started, ["MODEL_A", "MODEL_B"])
        self.assertEqual(terminated, ["1"])
        self.assertEqual(outcomes[0].status, "resource_censored")
        failure_info = payload["models"]["MODEL_A"]["failure_info"]
        self.assertEqual(failure_info["failure_kind"], "timeout_exceeded")
        self.assertGreaterEqual(failure_info["elapsed_worker_seconds"], 0.01)
        self.assertEqual(failure_info["configured_timeout_seconds"], 0.01)

    def test_same_model_variant_concurrent_lock_is_rejected(self) -> None:
        first = ModelVariantLock(
            context=self.context,
            model_id="MODEL",
            variant=SAVGOL_BENCHMARK_VARIANT,
        )
        second = ModelVariantLock(
            context=self.context,
            model_id="MODEL",
            variant=SAVGOL_BENCHMARK_VARIANT,
        )
        with first, self.assertRaisesRegex(RuntimeError, "writer lock"):
            second.acquire()
        self.assertFalse(first.path.exists())

    def test_different_variants_use_different_locks(self) -> None:
        exact = ModelVariantLock(
            context=self.context,
            model_id="MODEL",
            variant=EXACT_BENCHMARK_VARIANT,
        )
        savgol = ModelVariantLock(
            context=self.context,
            model_id="MODEL",
            variant=SAVGOL_BENCHMARK_VARIANT,
        )
        with exact, savgol:
            self.assertNotEqual(exact.path, savgol.path)

    def test_worker_thread_environment_is_capped(self) -> None:
        environment = worker_environment({"KEEP": "yes", "OMP_NUM_THREADS": "9"})
        self.assertEqual(environment["KEEP"], "yes")
        for name, expected in THREAD_CONTROL_ENVIRONMENT.items():
            self.assertEqual(environment[name], expected)

    def test_batch_registers_and_restores_sigterm_handler(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "signal-model.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A",),
                    n_states=1,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=1.0,
            )

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        with (
            patch("src.run_batch.SubprocessModelWorker", return_value=worker),
            patch("src.run_batch.signal.signal") as register_signal,
        ):
            execute_batch(
                model_ids=("MODEL",),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="preflight",
                context=RunContext(
                    run_id="signal-handler-test",
                    project_root=self.context.project_root,
                    results_root=self.context.results_root,
                    runs_root=self.context.runs_root,
                    cache_root=self.context.cache_root,
                ),
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        registered_signals = [call.args[0] for call in register_signal.call_args_list]
        self.assertGreaterEqual(registered_signals.count(signal.SIGINT), 2)
        self.assertGreaterEqual(registered_signals.count(signal.SIGTERM), 2)

    def test_worker_command_uses_variant_and_never_implicitly_opens_test(self) -> None:
        command = model_worker_command(
            model_id="MODEL",
            variant=SAVGOL_BENCHMARK_VARIANT,
            phase="pretest",
            run_context=self.context,
        )
        self.assertEqual(command[1:3], ("-m", "src.run_model"))
        self.assertIn("savgol", command)
        self.assertNotIn("--authorize-held-out-test", command)

    def test_exact_command_remains_default_namespace_variant(self) -> None:
        command = model_worker_command(
            model_id="MODEL",
            variant=EXACT_BENCHMARK_VARIANT,
            phase="candidates",
            run_context=self.context,
        )
        method_index = command.index("--derivative-method") + 1
        self.assertEqual(command[method_index], "exact")

    def test_plain_output_does_not_require_tty(self) -> None:
        from src.resource_monitor import PlainTextProgressReporter

        stream = io.StringIO()
        PlainTextProgressReporter(stream).report(
            completed=1,
            total=2,
            failed=0,
            active_rows=[],
        )
        self.assertIn("completed=1/2", stream.getvalue())

    def test_mocked_batch_writes_complete_external_manifest(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "model.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")
        project_root = Path(__file__).resolve().parents[1]
        context = RunContext(
            run_id="manifest-test",
            project_root=project_root,
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A", "B", "C", "D"),
                    n_states=4,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=20.0,
            )

        def synthetic_worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status="completed",
                exit_code=0,
                result={
                    "stage_checkpoints": {},
                    "pretest_checkpoint": None,
                    "selection_identity": None,
                },
            )

        output = io.StringIO()
        with patch(
            "src.run_batch.SubprocessModelWorker",
            return_value=synthetic_worker,
        ):
            outcomes = execute_batch(
                model_ids=("MODEL_A", "MODEL_B"),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=2,
                phase="preflight",
                context=context,
                telemetry_interval_seconds=0.01,
                preflight_runner=preflight,
                output_stream=output,
            )

        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(payload["derivative_method"], "savgol")
        self.assertEqual(payload["experiment_namespace"], "generic_savgol_experiment")
        self.assertEqual(payload["requested_workers"], 2)
        self.assertEqual(payload["effective_workers"], 2)
        self.assertEqual(payload["thread_environment"], THREAD_CONTROL_ENVIRONMENT)
        self.assertEqual(payload["models"]["MODEL_A"]["state_count"], 4)
        self.assertFalse(payload["models"]["MODEL_A"]["exhaustive_permitted"])
        self.assertEqual(
            payload["per_model_resource_policy"],
            {
                "maximum_process_tree_rss_bytes": 0,
                "memory_consecutive_samples": 2,
                "wall_clock_timeout_seconds": 0.0,
            },
        )
        self.assertTrue(context.telemetry_path.exists())
        self.assertIn("completed=2/2", output.getvalue())

    def test_ordinary_nonzero_exit_is_not_resource_censoring(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "nonzero-model.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")
        context = RunContext(
            run_id="ordinary-nonzero-test",
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A",),
                    n_states=1,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=1.0,
            )

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(
                model_id=model_id,
                worker_id=worker_id,
                status="failed",
                exit_code=7,
                failure="Worker exited 7.",
                failure_info=WorkerFailureInfo(
                    failure_kind="worker_nonzero_exit",
                    last_stage="candidates",
                ),
            )

        with patch("src.run_batch.SubprocessModelWorker", return_value=worker):
            outcomes = execute_batch(
                model_ids=("MODEL",),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="preflight",
                context=context,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        record = payload["models"]["MODEL"]
        self.assertEqual(outcomes[0].exit_code, 7)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_kind"], "worker_nonzero_exit")
        self.assertEqual(record["failures"], ["Worker exited 7."])
        self.assertNotIn("operational_failures", record)

    def test_memory_gate_blocks_before_manifest_or_worker_creation(self) -> None:
        context = RunContext(
            run_id="memory-block-test",
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )
        resources = {
            "effective_memory_available_bytes": 500,
        }
        with (
            patch("src.run_batch.discover_resources", return_value=resources),
            patch("src.run_batch.SubprocessModelWorker") as worker_factory,
            self.assertRaisesRegex(MemoryError, "blocked the batch"),
        ):
            execute_batch(
                model_ids=("MODEL",),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=4,
                phase="pretest",
                context=context,
                memory_safety_policy=MemorySafetyPolicy(
                    minimum_launch_available_bytes=600
                ),
                preflight_runner=lambda _model_id: self.fail(
                    "Preflight must not run after a failed memory gate."
                ),
                output_stream=io.StringIO(),
            )
        self.assertFalse(context.run_directory.exists())
        worker_factory.assert_not_called()

    def test_persistent_runtime_pressure_stops_new_model_scheduling(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "memory-model.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")
        context = RunContext(
            run_id="memory-runtime-test",
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )
        started: list[str] = []

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A", "B", "C", "D"),
                    n_states=4,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=20.0,
            )

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            started.append(model_id)
            time.sleep(0.05)
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        resources = {
            "effective_memory_available_bytes": 1_000,
            "effective_cpu_capacity": 1.0,
        }
        low_memory = {"effective_memory_available_bytes": 100}
        with (
            patch("src.run_batch.discover_resources", return_value=resources),
            patch("src.run_batch.memory_snapshot", return_value=low_memory),
            patch("src.run_batch.SubprocessModelWorker", return_value=worker),
            patch("src.run_batch.ActiveProcessRegistry.terminate_all") as terminate,
        ):
            outcomes = execute_batch(
                model_ids=("MODEL_A", "MODEL_B"),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="pretest",
                context=context,
                telemetry_interval_seconds=0.01,
                memory_safety_policy=MemorySafetyPolicy(
                    emergency_available_bytes=200,
                    emergency_consecutive_samples=2,
                ),
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(started, ["MODEL_A"])
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(payload["batch_status"], "memory_safety_stopped")
        self.assertEqual(len(payload["memory_safety"]["events"]), 1)
        terminate.assert_called_once()

    def test_resume_rejects_changed_sbml_before_worker_creation(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "model.xml"
        sbml_path.write_text("first\n", encoding="utf-8")
        context = RunContext(
            run_id="resume-test",
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A",),
                    n_states=1,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=5.0,
            )

        def worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(model_id, worker_id, "completed", 0)

        with patch("src.run_batch.SubprocessModelWorker", return_value=worker):
            execute_batch(
                model_ids=("MODEL",),
                variant=EXACT_BENCHMARK_VARIANT,
                workers=1,
                phase="preflight",
                context=context,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        sbml_path.write_text("changed\n", encoding="utf-8")
        with (
            patch("src.run_batch.SubprocessModelWorker") as worker_factory,
            self.assertRaisesRegex(ValueError, "Preflight invariants changed"),
        ):
            execute_batch(
                model_ids=("MODEL",),
                variant=EXACT_BENCHMARK_VARIANT,
                workers=1,
                phase="preflight",
                context=context,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )
        worker_factory.assert_not_called()

    def test_successful_resume_marks_requested_phase_complete(self) -> None:
        sbml_path = Path(self.temporary_directory.name) / "model.xml"
        sbml_path.write_text("<sbml/>\n", encoding="utf-8")
        context = RunContext(
            run_id="stage-resume-test",
            project_root=Path(__file__).resolve().parents[1],
            results_root=self.context.results_root,
            runs_root=self.context.runs_root,
            cache_root=self.context.cache_root,
        )

        def preflight(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id=model_id,
                spec=SimpleNamespace(
                    model_id=model_id,
                    model_path=sbml_path,
                    state_ids=("A", "B", "C"),
                    n_states=3,
                ),
                compatibility=SimpleNamespace(compatible=True, reasons=()),
                simulation_time=5.0,
            )

        def failed_worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(
                model_id,
                worker_id,
                "failed",
                1,
                failure="synthetic failure",
            )

        checkpoint = {
            "file_count": 1,
            "path_hash": "path",
            "content_hash": "content",
        }

        def successful_worker(model_id: str, worker_id: str) -> WorkerOutcome:
            return WorkerOutcome(
                model_id,
                worker_id,
                "completed",
                0,
                result={
                    "stage_checkpoints": {},
                    "pretest_checkpoint": checkpoint,
                    "selection_identity": {},
                },
            )

        with patch(
            "src.run_batch.SubprocessModelWorker",
            return_value=failed_worker,
        ):
            execute_batch(
                model_ids=("MODEL",),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="pretest",
                context=context,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        with patch(
            "src.run_batch.SubprocessModelWorker",
            return_value=successful_worker,
        ):
            execute_batch(
                model_ids=("MODEL",),
                variant=SAVGOL_BENCHMARK_VARIANT,
                workers=1,
                phase="pretest",
                context=context,
                preflight_runner=preflight,
                output_stream=io.StringIO(),
            )

        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        record = payload["models"]["MODEL"]
        self.assertEqual(record["status"], "pretest_complete")
        self.assertEqual(record["stages"]["pretest"]["status"], "completed")
        self.assertEqual(record["stages"]["pretest"]["exit_code"], 0)


class BatchCliTests(unittest.TestCase):
    """Verify explicit resource units and model-failure exit semantics."""

    def test_per_model_cli_defaults_disable_limits(self) -> None:
        arguments = parse_arguments(
            ["--manifest", "cohort.json", "--phase", "preflight"]
        )
        self.assertEqual(arguments.per_model_memory_gib, 0.0)
        self.assertEqual(arguments.per_model_memory_samples, 2)
        self.assertEqual(arguments.per_model_timeout_minutes, 0.0)
        self.assertEqual(arguments.model_failure_exit_policy, "error")

    def test_per_model_cli_values_and_invalid_inputs(self) -> None:
        arguments = parse_arguments(
            [
                "--manifest",
                "cohort.json",
                "--phase",
                "preflight",
                "--per-model-memory-gib",
                "1.5",
                "--per-model-memory-samples",
                "3",
                "--per-model-timeout-minutes",
                "4.5",
            ]
        )
        self.assertEqual(arguments.per_model_memory_gib, 1.5)
        self.assertEqual(arguments.per_model_memory_samples, 3)
        self.assertEqual(arguments.per_model_timeout_minutes, 4.5)
        with (
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_arguments(
                [
                    "--manifest",
                    "cohort.json",
                    "--phase",
                    "preflight",
                    "--per-model-memory-samples",
                    "0",
                ]
            )

    def test_exit_policy_only_allows_completed_model_failures(self) -> None:
        self.assertEqual(batch_exit_code("completed", "error"), 0)
        self.assertEqual(batch_exit_code("completed_with_failures", "error"), 1)
        self.assertEqual(
            batch_exit_code("completed_with_failures", "success"),
            0,
        )
        self.assertEqual(batch_exit_code("interrupted", "success"), 1)
        self.assertEqual(batch_exit_code("orchestration_failed", "success"), 1)
        self.assertEqual(batch_exit_code("memory_safety_stopped", "success"), 1)


if __name__ == "__main__":
    unittest.main()
