"""Tests for the frozen expanded-cohort Docker input bundle."""

import json
import stat
import tempfile
import unittest
from pathlib import Path

from src.docker_input_bundle import (
    EXPECTED_MODEL_COUNT,
    load_metadata_records,
    validate_bundle,
    validate_source_inputs,
)
from src.model_registry import resolve_model_path
from src.run_batch import load_cohort_manifest
from src.simulation_metadata import load_simulation_time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT / "docker" / "expanded_100_inputs"


class FrozenDockerBundleTests(unittest.TestCase):
    """Validate every checked-in model through production runtime loaders."""

    def test_all_100_bundle_models_resolve_with_authoritative_time(self) -> None:
        summary = validate_bundle(BUNDLE_ROOT)
        self.assertTrue(summary["all_models_valid"])
        self.assertEqual(summary["model_count"], EXPECTED_MODEL_COUNT)

        manifest_path = BUNDLE_ROOT / "metadata" / "benchmark_ready_100.json"
        metadata_path = BUNDLE_ROOT / "metadata" / "model_time.jsonl"
        model_root = BUNDLE_ROOT / "models"
        model_ids = load_cohort_manifest(manifest_path)
        self.assertEqual(len(model_ids), EXPECTED_MODEL_COUNT)
        self.assertEqual(
            {path.name for path in model_root.iterdir() if path.is_dir()},
            set(model_ids),
        )

        for model_id in model_ids:
            sbml_path = resolve_model_path(model_id, model_root)
            simulation_time = load_simulation_time(
                model_id=model_id,
                sbml_filename=sbml_path.name,
                metadata_path=metadata_path,
            )
            self.assertGreater(simulation_time, 0.0)

    def test_bundle_contains_only_100_sbml_files_and_100_metadata_rows(self) -> None:
        model_files = tuple(
            path for path in (BUNDLE_ROOT / "models").glob("*/*") if path.is_file()
        )
        records = load_metadata_records(BUNDLE_ROOT / "metadata" / "model_time.jsonl")
        self.assertEqual(len(model_files), EXPECTED_MODEL_COUNT)
        self.assertEqual(len(records), EXPECTED_MODEL_COUNT)
        self.assertTrue(BUNDLE_ROOT.stat().st_mode & stat.S_IROTH)
        self.assertTrue(BUNDLE_ROOT.stat().st_mode & stat.S_IXOTH)


class DockerBundleValidationFailureTests(unittest.TestCase):
    """Exercise strict source validation without repairing malformed inputs."""

    def setUp(self) -> None:
        """Create a minimal two-model source inventory."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.model_root = self.root / "models"
        self.model_root.mkdir()
        self.manifest_path = self.root / "cohort.json"
        self.metadata_path = self.root / "model_time.jsonl"
        self._write_manifest(["MODEL_A", "MODEL_B"])
        for model_id in ("MODEL_A", "MODEL_B"):
            model_directory = self.model_root / model_id
            model_directory.mkdir()
            (model_directory / f"{model_id}_url.xml").write_text(
                "<sbml/>\n",
                encoding="utf-8",
            )
        self._write_metadata(
            [
                {
                    "model_id": "MODEL_A",
                    "sbml_file": "MODEL_A_url.xml",
                    "simulation_time": "1",
                },
                {
                    "model_id": "MODEL_B",
                    "sbml_file": "MODEL_B_url.xml",
                    "simulation_time": "2",
                },
            ]
        )

    def _write_manifest(self, model_ids: list[str]) -> None:
        """Write a deterministic test cohort.

        Args:
            model_ids: Ordered model identifiers.
        """
        self.manifest_path.write_text(
            f"{json.dumps({'models': model_ids}, indent=2)}\n",
            encoding="utf-8",
        )

    def _write_metadata(self, rows: list[dict[str, str]]) -> None:
        """Write deterministic model-aware JSONL rows.

        Args:
            rows: Metadata objects in file order.
        """
        self.metadata_path.write_text(
            "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
            encoding="utf-8",
        )

    def _validate(self) -> None:
        """Validate the two-model fixture with production source checks."""
        validate_source_inputs(
            cohort_manifest=self.manifest_path,
            model_root=self.model_root,
            metadata_path=self.metadata_path,
            expected_model_count=2,
        )

    def test_duplicate_manifest_model_is_rejected(self) -> None:
        self._write_manifest(["MODEL_A", "MODEL_A"])
        with self.assertRaisesRegex(ValueError, "Duplicate model"):
            self._validate()

    def test_missing_or_ambiguous_model_is_rejected(self) -> None:
        (self.model_root / "MODEL_B" / "MODEL_B_url.xml").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No SBML"):
            self._validate()
        (self.model_root / "MODEL_B" / "first.xml").write_text(
            "<sbml/>\n",
            encoding="utf-8",
        )
        (self.model_root / "MODEL_B" / "second.xml").write_text(
            "<sbml/>\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Multiple SBML"):
            self._validate()

    def test_missing_or_duplicate_metadata_is_rejected(self) -> None:
        row = {
            "model_id": "MODEL_A",
            "sbml_file": "MODEL_A_url.xml",
            "simulation_time": "1",
        }
        self._write_metadata([row])
        with self.assertRaisesRegex(ValueError, "Missing metadata"):
            self._validate()
        self._write_metadata([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            self._validate()

    def test_filename_mismatch_and_invalid_time_are_rejected(self) -> None:
        first = {
            "model_id": "MODEL_A",
            "sbml_file": "wrong.xml",
            "simulation_time": "1",
        }
        second = {
            "model_id": "MODEL_B",
            "sbml_file": "MODEL_B_url.xml",
            "simulation_time": "2",
        }
        self._write_metadata([first, second])
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            self._validate()
        first["sbml_file"] = "MODEL_A_url.xml"
        first["simulation_time"] = "0"
        self._write_metadata([first, second])
        with self.assertRaisesRegex(ValueError, "Invalid curated simulation time"):
            self._validate()

    def test_bundle_source_count_must_be_exactly_100(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            validate_source_inputs(
                cohort_manifest=self.manifest_path,
                model_root=self.model_root,
                metadata_path=self.metadata_path,
            )


if __name__ == "__main__":
    unittest.main()
