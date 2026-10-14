"""Tests for atomic manifests and scientific-artifact checkpoints."""

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.run_manifest import (
    RunContext,
    RunManifest,
    atomic_write_json,
    checkpoint_paths,
)


class RunManifestTests(unittest.TestCase):
    """Verify atomic persistence and coordinator ownership."""

    def test_atomic_json_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "manifest.json"
            atomic_write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_manifest_rejects_noncoordinator_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = RunManifest(
                path=Path(temporary_directory) / "manifest.json",
                payload={"models": {}},
            )
            errors: list[Exception] = []

            def mutate() -> None:
                try:
                    manifest.update_model("MODEL", status="running")
                except Exception as error:  # noqa: BLE001 - assertion capture.
                    errors.append(error)

            thread = threading.Thread(target=mutate)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)

    def test_stage_records_separate_start_and_end_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = RunManifest(
                path=Path(temporary_directory) / "manifest.json",
                payload={"models": {"MODEL": {}}},
            )
            manifest.update_stage(
                model_id="MODEL", stage="preflight", status="running", pid=123
            )
            manifest.update_stage(
                model_id="MODEL", stage="preflight", status="completed"
            )
            record = manifest.payload["models"]["MODEL"]["stages"]["preflight"]
            self.assertIn("started_at_utc", record)
            self.assertIn("ended_at_utc", record)
            self.assertEqual(record["pid"], 123)

    def test_checkpoint_changes_when_artifact_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "artifact.csv"
            path.write_text("first\n", encoding="utf-8")
            before = checkpoint_paths([path], relative_to=root)
            path.write_text("second\n", encoding="utf-8")
            after = checkpoint_paths([path], relative_to=root)
            self.assertEqual(before.path_hash, after.path_hash)
            self.assertNotEqual(before.content_hash, after.content_hash)

    def test_run_id_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "path component"):
            RunContext(run_id="../results")


if __name__ == "__main__":
    unittest.main()
