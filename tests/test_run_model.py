"""Tests for generic single-model orchestration guards."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
)
from src.run_manifest import RunContext, checkpoint_paths
from src.run_model import (
    TRAINING_DATASETS,
    candidate_artifact_paths,
    exhaustive_artifact_paths,
    experiment_root,
    pretest_artifact_paths,
    run_model,
    selector_artifact_paths,
    test_artifact_paths,
)

MODEL_ID = "BIOMD_TEST"


def synthetic_preflight(state_ids: tuple[str, ...]) -> SimpleNamespace:
    """Build a compatible preflight fixture."""
    return SimpleNamespace(
        model_id=MODEL_ID,
        spec=SimpleNamespace(
            model_id=MODEL_ID,
            state_ids=state_ids,
            n_states=len(state_ids),
        ),
        compatibility=SimpleNamespace(compatible=True, reasons=()),
        simulation_time=10.0,
    )


def write_csv(path: Path, row: dict[str, object]) -> None:
    """Write one-row CSV fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def create_candidates(
    root: Path,
    state_ids: tuple[str, ...],
    method: str,
    namespace: str,
) -> None:
    """Create a complete synthetic candidate artifact contract."""
    provenance = {
        "derivative_method": method,
        "experiment_namespace": namespace,
    }
    write_csv(root / "experiment_metadata.csv", provenance)
    for dataset_name in TRAINING_DATASETS:
        write_csv(root / dataset_name / "run_metadata.csv", provenance)
        for state_id in state_ids:
            state_root = root / dataset_name / f"state_{state_id}"
            write_csv(state_root / "all_equations.csv", {"equation": state_id})
            write_csv(state_root / "top_8_by_loss.csv", {"equation": state_id})


def create_plain_artifacts(paths: tuple[Path, ...]) -> None:
    """Create nonempty placeholder artifacts for an orchestration contract."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n", encoding="utf-8")


def create_selector(root: Path, state_ids: tuple[str, ...]) -> None:
    """Create selector outputs with a readable frozen identity."""
    paths = selector_artifact_paths(root=root)
    create_plain_artifacts(paths)
    for dataset_name in TRAINING_DATASETS:
        row: dict[str, object] = {"system_id": 1}
        row.update({f"candidate_rank_{state_id}": 1 for state_id in state_ids})
        write_csv(
            root
            / dataset_name
            / "coordinate_selection"
            / "coordinate_selected_system.csv",
            row,
        )


class RunModelTests(unittest.TestCase):
    """Verify stage ordering, namespace isolation, and held-out gates."""

    def setUp(self) -> None:
        """Create isolated orchestration roots."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.context = RunContext(
            run_id="test-run",
            project_root=root,
            results_root=root / "results",
            runs_root=root / "runs",
            cache_root=root / "cache",
        )

    def test_exact_default_namespace_is_unchanged(self) -> None:
        root = experiment_root(
            model_id=MODEL_ID,
            variant=EXACT_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        result = run_model(
            MODEL_ID,
            phase="preflight",
            run_context=self.context,
            preflight_runner=lambda _model_id: synthetic_preflight(("A", "B")),
            event_callback=lambda _event: None,
        )
        self.assertEqual(root.name, "generic_exact_experiment")
        self.assertEqual(result.experiment_namespace, "generic_exact_experiment")

    def test_preflight_does_not_adopt_existing_downstream_artifacts(self) -> None:
        state_ids = ("A", "B", "C")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        create_candidates(root, state_ids, "savgol", "generic_savgol_experiment")
        create_plain_artifacts(exhaustive_artifact_paths(root=root))
        create_selector(root, state_ids)

        result = run_model(
            MODEL_ID,
            SAVGOL_BENCHMARK_VARIANT,
            "preflight",
            self.context,
            command_runner=self.fail,
            preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
            event_callback=lambda _event: None,
        )
        self.assertIsNone(result.pretest_checkpoint)
        self.assertIsNone(result.selection_identity)
        self.assertEqual(result.stages, {"preflight": "completed"})

    def test_savgol_namespace_is_isolated(self) -> None:
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        self.assertEqual(root.name, "generic_savgol_experiment")
        self.assertNotEqual(root.name, "generic_exact_experiment")

    def test_more_than_three_states_cannot_run_exhaustive(self) -> None:
        calls: list[tuple[str, ...]] = []
        with self.assertRaisesRegex(RuntimeError, "prohibited"):
            run_model(
                MODEL_ID,
                SAVGOL_BENCHMARK_VARIANT,
                "optional_exhaustive",
                self.context,
                command_runner=calls.append,
                preflight_runner=lambda _model_id: synthetic_preflight(
                    ("A", "B", "C", "D")
                ),
                event_callback=lambda _event: None,
            )
        self.assertEqual(calls, [])

    def test_three_states_pretest_runs_candidates_validation_selector_only(
        self,
    ) -> None:
        state_ids = ("A", "B", "C")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        modules: list[str] = []

        def runner(command: tuple[str, ...]) -> None:
            module = command[2]
            modules.append(module)
            if module == "src.run_generic_benchmark":
                create_candidates(
                    root,
                    state_ids,
                    "savgol",
                    "generic_savgol_experiment",
                )
            elif module == "src.run_generic_validation":
                create_plain_artifacts(exhaustive_artifact_paths(root=root))
            elif module == "src.select_generic_system":
                create_selector(root, state_ids)

        result = run_model(
            MODEL_ID,
            SAVGOL_BENCHMARK_VARIANT,
            "pretest",
            self.context,
            command_runner=runner,
            preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
            event_callback=lambda _event: None,
        )
        self.assertEqual(
            modules,
            [
                "src.run_generic_benchmark",
                "src.run_generic_validation",
                "src.select_generic_system",
            ],
        )
        self.assertNotIn("src.run_generic_test", modules)
        self.assertIsNotNone(result.pretest_checkpoint)

    def test_partial_candidate_stage_is_not_complete_or_overwritten(self) -> None:
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        write_csv(
            root / "experiment_metadata.csv",
            {
                "derivative_method": "savgol",
                "experiment_namespace": "generic_savgol_experiment",
            },
        )
        calls: list[tuple[str, ...]] = []
        with self.assertRaisesRegex(RuntimeError, "partial artifacts"):
            run_model(
                MODEL_ID,
                SAVGOL_BENCHMARK_VARIANT,
                "candidates",
                self.context,
                command_runner=calls.append,
                preflight_runner=lambda _model_id: synthetic_preflight(("A", "B")),
                event_callback=lambda _event: None,
            )
        self.assertEqual(calls, [])

    def test_test_requires_explicit_authorization_and_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "explicit authorization"):
            run_model(
                MODEL_ID,
                SAVGOL_BENCHMARK_VARIANT,
                "test",
                self.context,
                preflight_runner=lambda _model_id: synthetic_preflight(("A", "B")),
                event_callback=lambda _event: None,
            )

        with self.assertRaisesRegex(RuntimeError, "pre-test checkpoint"):
            run_model(
                MODEL_ID,
                SAVGOL_BENCHMARK_VARIANT,
                "test",
                self.context,
                held_out_test_authorized=True,
                preflight_runner=lambda _model_id: synthetic_preflight(("A", "B")),
                event_callback=lambda _event: None,
            )

    def test_changed_pretest_artifact_blocks_test_before_command(self) -> None:
        state_ids = ("A", "B", "C", "D")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        create_candidates(root, state_ids, "savgol", "generic_savgol_experiment")
        create_selector(root, state_ids)
        paths = pretest_artifact_paths(
            root=root,
            state_ids=state_ids,
            exhaustive_permitted=False,
        )
        checkpoint = checkpoint_paths(paths, relative_to=root)
        paths[-1].write_text("changed\n", encoding="utf-8")
        calls: list[tuple[str, ...]] = []

        with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
            run_model(
                MODEL_ID,
                SAVGOL_BENCHMARK_VARIANT,
                "test",
                self.context,
                expected_pretest_checkpoint=checkpoint,
                held_out_test_authorized=True,
                command_runner=calls.append,
                preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
                event_callback=lambda _event: None,
            )
        self.assertEqual(calls, [])

    def test_valid_checkpoint_allows_only_test_stage(self) -> None:
        state_ids = ("A", "B", "C", "D")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=SAVGOL_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        create_candidates(root, state_ids, "savgol", "generic_savgol_experiment")
        create_selector(root, state_ids)
        checkpoint = checkpoint_paths(
            pretest_artifact_paths(
                root=root,
                state_ids=state_ids,
                exhaustive_permitted=False,
            ),
            relative_to=root,
        )
        modules: list[str] = []

        def runner(command: tuple[str, ...]) -> None:
            modules.append(command[2])
            create_plain_artifacts(test_artifact_paths(root=root))

        run_model(
            MODEL_ID,
            SAVGOL_BENCHMARK_VARIANT,
            "test",
            self.context,
            expected_pretest_checkpoint=checkpoint,
            held_out_test_authorized=True,
            command_runner=runner,
            preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
            event_callback=lambda _event: None,
        )
        self.assertEqual(modules, ["src.run_generic_test"])

    def test_resume_rejects_changed_completed_stage(self) -> None:
        state_ids = ("A", "B")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=EXACT_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        create_candidates(root, state_ids, "exact", "generic_exact_experiment")
        paths = candidate_artifact_paths(root=root, state_ids=state_ids)
        checkpoint = checkpoint_paths(paths, relative_to=root)
        paths[-1].write_text("changed\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "manifest checkpoint"):
            run_model(
                MODEL_ID,
                EXACT_BENCHMARK_VARIANT,
                "candidates",
                self.context,
                expected_stage_checkpoints={"candidates": checkpoint},
                command_runner=self.fail,
                preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
                event_callback=lambda _event: None,
            )

    def test_resume_skips_only_verified_completed_stage(self) -> None:
        state_ids = ("A", "B")
        root = experiment_root(
            model_id=MODEL_ID,
            variant=EXACT_BENCHMARK_VARIANT,
            run_context=self.context,
        )
        create_candidates(root, state_ids, "exact", "generic_exact_experiment")
        checkpoint = checkpoint_paths(
            candidate_artifact_paths(root=root, state_ids=state_ids),
            relative_to=root,
        )
        result = run_model(
            MODEL_ID,
            EXACT_BENCHMARK_VARIANT,
            "candidates",
            self.context,
            expected_stage_checkpoints={"candidates": checkpoint},
            command_runner=self.fail,
            preflight_runner=lambda _model_id: synthetic_preflight(state_ids),
            event_callback=lambda _event: None,
        )
        self.assertEqual(result.stages["candidates"], "skipped")
        self.assertEqual(result.stage_checkpoints["candidates"], checkpoint)


if __name__ == "__main__":
    unittest.main()
