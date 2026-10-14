"""Tests for cgroup discovery and recursive process telemetry."""

import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.resource_monitor import (
    ProcessTreeSampler,
    TelemetryWriter,
    WorkerTelemetryContext,
    discover_resources,
    read_cgroup_v2_cpu_capacity,
    read_cgroup_v2_memory_limit,
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
                "system_memory_percent": 25.0,
            }
            with TelemetryWriter(path) as writer:
                writer.write(row)
            with path.open("r", encoding="utf-8", newline="") as stream:
                loaded = list(csv.DictReader(stream))
            self.assertEqual(loaded[0]["worker_id"], "2")
            self.assertEqual(loaded[0]["model_id"], "MODEL")
            self.assertEqual(loaded[0]["stage"], "selector")

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


if __name__ == "__main__":
    unittest.main()
