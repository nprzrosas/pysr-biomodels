"""Tests for cgroup discovery and recursive process telemetry."""

import csv
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.resource_monitor import (
    MemoryPressureGuard,
    MemorySafetyPolicy,
    PlainTextProgressReporter,
    ProcessTreeSampler,
    TelemetryWriter,
    WorkerTelemetryContext,
    discover_resources,
    memory_snapshot,
    read_cgroup_v2_cpu_capacity,
    read_cgroup_v2_memory_current,
    read_cgroup_v2_memory_limit,
    validate_prelaunch_memory,
)
from src.run_batch import (
    ActiveProcessRegistry,
    ActiveWorker,
    PerModelResourceGuard,
    PerModelResourcePolicy,
    WorkerFailureInfo,
)


class ResourceDiscoveryTests(unittest.TestCase):
    """Verify bounded-resource discovery with and without cgroup limits."""

    def test_cgroup_v2_limits_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "cpu.max").write_text("250000 100000\n", encoding="utf-8")
            (root / "memory.max").write_text("1073741824\n", encoding="utf-8")
            self.assertEqual(read_cgroup_v2_cpu_capacity(root), 2.5)
            self.assertEqual(read_cgroup_v2_memory_limit(root), 1073741824)
            resources = discover_resources(root)
            self.assertLessEqual(resources["effective_cpu_capacity"], 2.5)
            self.assertLessEqual(resources["effective_memory_limit_bytes"], 1073741824)

    def test_unlimited_or_missing_cgroup_files_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
            (root / "memory.max").write_text("max\n", encoding="utf-8")
            self.assertIsNone(read_cgroup_v2_cpu_capacity(root))
            self.assertIsNone(read_cgroup_v2_memory_limit(root))

    def test_cgroup_memory_availability_limits_host_availability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "memory.max").write_text("1073741824\n", encoding="utf-8")
            (root / "memory.current").write_text("805306368\n", encoding="utf-8")
            self.assertEqual(read_cgroup_v2_memory_current(root), 805306368)
            snapshot = memory_snapshot(root)
            self.assertEqual(snapshot["effective_memory_available_bytes"], 268435456)


class MemorySafetyTests(unittest.TestCase):
    """Verify launch and persistent runtime memory protection."""

    def test_prelaunch_gate_rejects_insufficient_memory(self) -> None:
        policy = MemorySafetyPolicy(minimum_launch_available_bytes=600)
        with self.assertRaisesRegex(MemoryError, "blocked the batch"):
            validate_prelaunch_memory(
                policy,
                {"effective_memory_available_bytes": 599},
            )

    def test_runtime_gate_requires_consecutive_low_samples(self) -> None:
        guard = MemoryPressureGuard(
            MemorySafetyPolicy(
                emergency_available_bytes=200,
                emergency_consecutive_samples=2,
            )
        )
        self.assertFalse(guard.observe(199))
        self.assertFalse(guard.observe(201))
        self.assertFalse(guard.observe(199))
        self.assertTrue(guard.observe(198))


class ProcessTreeTelemetryTests(unittest.TestCase):
    """Verify child discovery and machine-readable telemetry output."""

    def test_recursive_child_process_is_reported(self) -> None:
        parent_code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(5)']); "
            "print(p.pid,flush=True); time.sleep(5)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.addCleanup(self._terminate, process)
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        sampler = ProcessTreeSampler(
            parent_pid=process.pid,
            effective_cpu_capacity=1.0,
        )
        sampler.prime()
        time.sleep(0.05)
        row = sampler.sample(
            context=WorkerTelemetryContext(
                run_id="run",
                worker_id="1",
                model_id="MODEL",
                stage="candidates",
                worker_state="running",
            ),
            elapsed_seconds=0.05,
        )
        self.assertIn(str(child_pid), row["child_pids"].split(";"))
        self.assertEqual(row["model_id"], "MODEL")
        self.assertEqual(row["stage"], "candidates")

    def test_recursive_child_cpu_is_measured_after_priming(self) -> None:
        child_code = """import time
end = time.monotonic() + 2
while time.monotonic() < end:
    pass
"""
        parent_code = (
            "import subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
            "print(p.pid,flush=True); time.sleep(3)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.addCleanup(self._terminate, process)
        assert process.stdout is not None
        process.stdout.readline()
        sampler = ProcessTreeSampler(
            parent_pid=process.pid,
            effective_cpu_capacity=1.0,
        )
        sampler.prime()
        time.sleep(0.2)
        row = sampler.sample(
            context=WorkerTelemetryContext(
                run_id="run",
                worker_id="1",
                model_id="MODEL",
                stage="candidates",
                worker_state="running",
            ),
            elapsed_seconds=0.2,
        )
        self.assertGreater(row["process_tree_cpu_percent_raw"], 0.0)

    def test_telemetry_csv_has_model_worker_and_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "telemetry.csv"
            row = {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "elapsed_seconds": 1.0,
                "run_id": "run",
                "worker_id": "2",
                "model_id": "MODEL",
                "stage": "selector",
                "worker_state": "running",
                "parent_pid": 100,
                "child_pids": "101;102",
                "process_tree_cpu_percent_raw": 150.0,
                "process_tree_cpu_percent_capacity_normalized": 75.0,
                "process_tree_rss_bytes": 1024,
                "system_cpu_percent": 50.0,
                "system_memory_used_bytes": 2048,
                "system_memory_available_bytes": 4096,
                "effective_memory_available_bytes": 4096,
                "system_memory_percent": 25.0,
                "system_swap_total_bytes": 8192,
                "system_swap_used_bytes": 1024,
            }
            with TelemetryWriter(path) as writer:
                writer.write(row)
            with path.open("r", encoding="utf-8", newline="") as stream:
                loaded = list(csv.DictReader(stream))
            self.assertEqual(loaded[0]["worker_id"], "2")
            self.assertEqual(loaded[0]["model_id"], "MODEL")
            self.assertEqual(loaded[0]["stage"], "selector")

    def test_plain_progress_includes_resource_and_process_context(self) -> None:
        stream = io.StringIO()
        row = {
            "elapsed_seconds": 12.5,
            "worker_id": "2",
            "model_id": "MODEL",
            "stage": "candidates",
            "parent_pid": 123,
            "process_tree_cpu_percent_raw": 250.0,
            "process_tree_cpu_percent_capacity_normalized": 15.625,
            "process_tree_rss_bytes": 2 * 1024**3,
            "system_cpu_percent": 40.0,
            "system_memory_percent": 55.0,
            "effective_memory_available_bytes": 4 * 1024**3,
        }
        PlainTextProgressReporter(stream).report(
            completed=0,
            total=2,
            failed=0,
            active_rows=[row],
        )
        output = stream.getvalue()
        self.assertIn("elapsed=12.5s", output)
        self.assertIn("system_cpu=40.0%", output)
        self.assertIn("system_ram=55.0%", output)
        self.assertIn("available=4.00GiB", output)
        self.assertIn("pid=123", output)
        self.assertIn("cpu=250.0%", output)
        self.assertIn("capacity=15.6%", output)
        self.assertIn("rss=2.00GiB", output)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        """Terminate a synthetic process and its direct child."""
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


@unittest.skipUnless(os.name == "posix", "Process-group integration requires POSIX.")
class PerModelResourceIntegrationTests(unittest.TestCase):
    """Exercise real Linux sampling, resource decisions, and termination."""

    def test_real_rss_limit_terminates_worker_and_preserves_isolation(self) -> None:
        allocation_bytes = 48 * 1024**2
        memory_limit_bytes = 16 * 1024**2
        worker_code = f"""import time
payload = bytearray({allocation_bytes})
for index in range(0, len(payload), 4096):
    payload[index] = 1
print("ready", flush=True)
time.sleep(10)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        started_at = time.monotonic()
        self.addCleanup(ProcessTreeTelemetryTests._terminate, process)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(), "ready")

        worker_id = "real-memory-worker"
        stage = "real_memory_stage"
        registry = ActiveProcessRegistry()
        registry.add(
            ActiveWorker(
                worker_id=worker_id,
                model_id="MEMORY_MODEL",
                stage=stage,
                state="running",
                process=process,
                started_at_monotonic=started_at,
            )
        )
        guard = PerModelResourceGuard(
            PerModelResourcePolicy(
                maximum_process_tree_rss_bytes=memory_limit_bytes,
                memory_consecutive_samples=2,
            )
        )
        guard.register_worker(worker_id, started_at)
        sampler = ProcessTreeSampler(
            parent_pid=process.pid,
            effective_cpu_capacity=1.0,
        )
        sampler.prime()

        failure_info: WorkerFailureInfo | None = None
        deadline = time.monotonic() + 2.0
        try:
            while failure_info is None and time.monotonic() < deadline:
                observed_at = time.monotonic()
                row = sampler.sample(
                    context=WorkerTelemetryContext(
                        run_id="integration-run",
                        worker_id=worker_id,
                        model_id="MEMORY_MODEL",
                        stage=stage,
                        worker_state="running",
                    ),
                    elapsed_seconds=observed_at - started_at,
                )
                failure_info = guard.observe(
                    worker_id=worker_id,
                    stage=stage,
                    process_tree_rss_bytes=int(row["process_tree_rss_bytes"]),
                    observed_at_monotonic=observed_at,
                    started_at_monotonic=started_at,
                )
                if failure_info is None:
                    time.sleep(0.02)

            self.assertIsNotNone(failure_info)
            assert failure_info is not None
            self.assertEqual(failure_info.failure_kind, "memory_limit_exceeded")
            self.assertGreater(
                failure_info.peak_process_tree_rss_bytes,
                memory_limit_bytes,
            )
            self.assertEqual(failure_info.last_stage, stage)
        finally:
            registry.terminate_worker(worker_id, grace_seconds=0.5)
            process.wait(timeout=2.0)
            registry.remove(worker_id)
            guard.finish_worker(worker_id)

        self.assertIsNotNone(process.poll())
        self._assert_successor_process_runs()

    def test_real_timeout_terminates_worker_and_preserves_isolation(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        started_at = time.monotonic()
        self.addCleanup(ProcessTreeTelemetryTests._terminate, process)
        worker_id = "real-timeout-worker"
        timeout_seconds = 0.2
        stage = "real_timeout_stage"
        registry = ActiveProcessRegistry()
        registry.add(
            ActiveWorker(
                worker_id=worker_id,
                model_id="TIMEOUT_MODEL",
                stage=stage,
                state="running",
                process=process,
                started_at_monotonic=started_at,
            )
        )
        guard = PerModelResourceGuard(
            PerModelResourcePolicy(
                wall_clock_timeout_seconds=timeout_seconds,
            )
        )
        guard.register_worker(worker_id, started_at)

        failure_info: WorkerFailureInfo | None = None
        deadline = time.monotonic() + 2.0
        try:
            while failure_info is None and time.monotonic() < deadline:
                failure_info = guard.observe(
                    worker_id=worker_id,
                    stage=stage,
                    process_tree_rss_bytes=None,
                    observed_at_monotonic=time.monotonic(),
                    started_at_monotonic=started_at,
                )
                if failure_info is None:
                    time.sleep(0.02)

            self.assertIsNotNone(failure_info)
            assert failure_info is not None
            self.assertEqual(failure_info.failure_kind, "timeout_exceeded")
            self.assertGreaterEqual(
                failure_info.elapsed_worker_seconds,
                timeout_seconds,
            )
        finally:
            registry.terminate_worker(worker_id, grace_seconds=0.5)
            process.wait(timeout=2.0)
            registry.remove(worker_id)
            guard.finish_worker(worker_id)

        self.assertIsNotNone(process.poll())
        self._assert_successor_process_runs()

    def _assert_successor_process_runs(self) -> None:
        """Prove prior process-group termination spared subsequent work."""
        successor = subprocess.run(
            [sys.executable, "-c", "print('successor-ok')"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
            start_new_session=True,
        )
        self.assertEqual(successor.stdout.strip(), "successor-ok")


if __name__ == "__main__":
    unittest.main()
