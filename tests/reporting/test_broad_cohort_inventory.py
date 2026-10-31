"""Tests for artifact-driven broad-cohort inventory."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.reporting.broad_cohort.initial_conditions import (
    build_development_initial_conditions,
)
from src.reporting.broad_cohort.inventory import (
    TRAIN_DATASETS,
    build_condition_inventory,
    build_failure_inventory,
    build_model_inventory,
    load_cohort,
)
from src.reporting.broad_cohort.mechanistic import (
    MechanisticEvaluationTimeoutError,
    evaluation_timeout,
)
from src.reporting.broad_cohort.plots import (
    load_selected_validation_predictions,
    load_validation_predictions,
)


class BroadCohortInventoryTests(unittest.TestCase):
    """Verify complete denominators and explicit incomplete states."""

    def setUp(self) -> None:
        """Create an isolated result and run tree."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.results_root = self.root / "results"
        self.runs_root = self.root / "runs"
        self.run_id = "batch"
        run_root = self.runs_root / self.run_id
        (run_root / "logs").mkdir(parents=True)
        (self.runs_root / "locks").mkdir(parents=True)
        manifest = {
            "models": {
                "MODEL_A": {
                    "state_count": 1,
                    "status": "running",
                    "held_out_test_authorized": False,
                },
                "MODEL_B": {
                    "state_count": 2,
                    "status": "failed",
                    "held_out_test_authorized": False,
                },
            }
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        telemetry = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-01-01T00:00:00Z",
                    "model_id": "MODEL_A",
                    "process_tree_rss_bytes": 100,
                    "process_tree_cpu_percent_capacity_normalized": 2,
                    "effective_memory_available_bytes": 1000,
                    "stage": "selector",
                },
                {
                    "timestamp_utc": "2026-01-01T00:01:00Z",
                    "model_id": "MODEL_A",
                    "process_tree_rss_bytes": 200,
                    "process_tree_cpu_percent_capacity_normalized": 4,
                    "effective_memory_available_bytes": 900,
                    "stage": "selector",
                },
            ]
        )
        telemetry.to_csv(run_root / "telemetry.csv", index=False)
        (run_root / "logs" / "MODEL_A.log").write_text(
            '"stage": "selector", "status": "running"', encoding="utf-8"
        )
        (run_root / "logs" / "MODEL_B.log").write_text(
            "CV_CONV_FAILURE", encoding="utf-8"
        )

    def tearDown(self) -> None:
        """Remove the isolated fixture tree."""
        self.temporary_directory.cleanup()

    def _write_candidates(self) -> None:
        """Write complete one-state candidate artifacts for MODEL_A."""
        experiment = self.results_root / "MODEL_A" / "generic_exact_experiment"
        for dataset_name in TRAIN_DATASETS:
            state_root = experiment / dataset_name / "state_x"
            state_root.mkdir(parents=True)
            (state_root / "all_equations.csv").write_text("loss\n1\n")
            (state_root / "top_8_by_loss.csv").write_text("loss\n1\n")

    def test_cohort_rejects_duplicate_models(self) -> None:
        """Reject ambiguous cohort denominators."""
        path = self.root / "cohort.json"
        path.write_text(json.dumps(["MODEL_A", "MODEL_A"]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_cohort(path)

    def test_empty_prediction_artifact_is_a_valid_empty_frame(self) -> None:
        """Treat newline-only predictions as an integration-failure artifact."""
        path = self.root / "predictions.csv"
        path.write_text("\n", encoding="utf-8")
        self.assertTrue(load_validation_predictions(path).empty)

    def test_validation_plot_loader_keeps_only_selected_system(self) -> None:
        """Exclude coordinate-search candidates from report figures."""
        path = self.root / "predictions.csv"
        unselected_id = 147573952589676412928
        selected_id = 147573952589676412929
        predictions = pd.DataFrame(
            [
                {
                    "system_id": unselected_id,
                    "trajectory_id": "validation_000",
                    "state_id": "x",
                    "time": 0.0,
                    "observed": 1.0,
                    "predicted": 99.0,
                },
                {
                    "system_id": selected_id,
                    "trajectory_id": "validation_000",
                    "state_id": "x",
                    "time": 0.0,
                    "observed": 1.0,
                    "predicted": 1.1,
                },
            ]
        )
        predictions.to_csv(path, index=False)
        selected = load_selected_validation_predictions(path, selected_id)
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected.iloc[0]["system_id"]), selected_id)
        self.assertAlmostEqual(float(selected.iloc[0]["predicted"]), 1.1)

    def test_initial_condition_report_reconstructs_development_only(self) -> None:
        """Build nested training and validation conditions without test data."""
        spec = SimpleNamespace(
            state_ids=("x", "y"),
            initial_values={"x": 1.0, "y": 2.0},
        )
        with patch(
            "src.reporting.broad_cohort.initial_conditions.load_model_spec",
            return_value=spec,
        ):
            training, validation, nominal = build_development_initial_conditions(
                "MODEL_A",
                self.root / "model.xml",
            )
        self.assertEqual(len(training), 32)
        self.assertEqual(len(validation), 8)
        self.assertEqual(int(training["in_train_04"].sum()), 4)
        self.assertEqual(int(training["in_train_08"].sum()), 8)
        self.assertEqual(int(training["in_train_16"].sum()), 16)
        self.assertEqual(int(training["in_train_32"].sum()), 32)
        self.assertEqual(nominal.to_dict(), {"x": 1.0, "y": 2.0})
        self.assertFalse(any("test" in column for column in training.columns))

    def test_symbolic_evaluation_timeout_is_bounded(self) -> None:
        """Interrupt a pathological symbolic operation without ending a run."""
        with (
            self.assertRaises(MechanisticEvaluationTimeoutError),
            evaluation_timeout(0.01),
        ):
            time.sleep(0.1)

    def test_inventory_keeps_all_expected_conditions(self) -> None:
        """Represent missing selections instead of dropping them."""
        conditions, equations = build_condition_inventory(
            self.results_root, ("MODEL_A", "MODEL_B")
        )
        self.assertEqual(len(conditions), 8)
        self.assertEqual(int(conditions["selection_present"].sum()), 0)
        self.assertTrue(equations.empty)

    def test_model_and_failure_classification_uses_artifacts(self) -> None:
        """Distinguish complete candidates from simulation failure."""
        self._write_candidates()
        conditions, _equations = build_condition_inventory(
            self.results_root, ("MODEL_A", "MODEL_B")
        )
        models = build_model_inventory(
            self.results_root,
            self.runs_root,
            self.run_id,
            ("MODEL_A", "MODEL_B"),
            conditions,
        ).set_index("model_id")
        self.assertEqual(
            models.loc["MODEL_A", "pipeline_status"], "selector_interrupted"
        )
        self.assertEqual(models.loc["MODEL_B", "pipeline_status"], "candidate_failed")
        self.assertEqual(
            models.loc["MODEL_B", "log_category"],
            "simulation_cvode_convergence",
        )
        failures = build_failure_inventory(models.reset_index(), conditions)
        self.assertIn("candidate_failed", set(failures["failure_type"]))
        self.assertEqual(
            len(failures.loc[failures["failure_type"] == "missing_selection"]), 8
        )


if __name__ == "__main__":
    unittest.main()
