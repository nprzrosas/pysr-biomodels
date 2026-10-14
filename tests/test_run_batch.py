"""Tests for bounded generic batch scheduling and isolation."""

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
)
from src.run_batch import (
    THREAD_CONTROL_ENVIRONMENT,
    WorkerOutcome,
    execute_batch,
    model_worker_command,
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
        self.assertTrue(context.telemetry_path.exists())
        self.assertIn("completed=2/2", output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
