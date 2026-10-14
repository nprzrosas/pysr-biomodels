"""Tests for variant-aware generic held-out evaluation."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import run_generic_test
from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
)
from src.evaluate_generic_test import (
    TRAINING_DATASETS,
    evaluate_all_selected_systems,
    evaluate_selected_system_on_test,
    load_frozen_selected_systems,
    load_selected_system,
    resolve_test_experiment_root,
)

STATE_IDS = ("P", "Q", "R")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_spec() -> SimpleNamespace:
    """Build minimal ordered model metadata for held-out tests.

    Returns:
        A ModelSpec-compatible synthetic object.
    """
    return SimpleNamespace(
        model_id="BIOMD0000000079",
        state_ids=STATE_IDS,
        n_states=len(STATE_IDS),
    )


def write_metadata(
    *,
    experiment_root: Path,
    derivative_method: str,
    experiment_namespace: str,
    dataset_names: tuple[str, ...] = TRAINING_DATASETS,
) -> None:
    """Write minimal variant provenance and training scales.

    Args:
        experiment_root: Temporary variant-specific experiment root.
        derivative_method: Derivative method recorded in metadata.
        experiment_namespace: Namespace recorded in metadata.
        dataset_names: Training conditions to create.
    """
    provenance = {
        "derivative_method": derivative_method,
        "experiment_namespace": experiment_namespace,
    }
    experiment_root.mkdir(parents=True)
    pd.DataFrame([provenance]).to_csv(
        experiment_root / "experiment_metadata.csv",
        index=False,
    )

    for dataset_name in dataset_names:
        dataset_root = experiment_root / dataset_name
        dataset_root.mkdir()
        pd.DataFrame(
            [
                {
                    **provenance,
                    "state_scale_P": 1.0,
                    "state_scale_Q": 1.0,
                    "state_scale_R": 1.0,
                }
            ]
        ).to_csv(dataset_root / "run_metadata.csv", index=False)


def write_coordinate_selection(
    *,
    experiment_root: Path,
    dataset_name: str,
    system_id: int,
    ranks: tuple[int, int, int],
) -> None:
    """Write one synthetic frozen coordinate-selected system.

    Args:
        experiment_root: Temporary variant-specific experiment root.
        dataset_name: Training-size condition.
        system_id: Canonical Cartesian system identifier.
        ranks: Candidate ranks in P, Q, R order.
    """
    selection_dir = experiment_root / dataset_name / "coordinate_selection"
    selection_dir.mkdir()
    row = {
        "system_id": system_id,
        "model_id": "BIOMD0000000079",
        "dataset_name": dataset_name,
        "selection_method": "coordinate_search",
        "candidate_rank_signature": ",".join(str(rank) for rank in ranks),
        "validation_macro_nrmse": 0.25,
    }

    for state_id, rank in zip(STATE_IDS, ranks):
        row[f"candidate_rank_{state_id}"] = rank
        row[f"equation_{state_id}"] = state_id

    pd.DataFrame([row]).to_csv(
        selection_dir / "coordinate_selected_system.csv",
        index=False,
    )


def build_state_only_test_frame() -> pd.DataFrame:
    """Build eight state-only trajectories with 101 samples each.

    Returns:
        Synthetic held-out state trajectories in model state order.
    """
    frames = []

    for trajectory_index in range(8):
        time_values = np.linspace(0.0, 1.0, 101)
        frames.append(
            pd.DataFrame(
                {
                    "trajectory_id": f"test_{trajectory_index:03d}",
                    "time": time_values,
                    "P": time_values,
                    "Q": time_values + 1.0,
                    "R": time_values + 2.0,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


class HeldOutVariantResolutionTests(unittest.TestCase):
    """Verify CLI defaults and protected result namespaces."""

    def test_default_remains_exact_and_savgol_is_explicit(self) -> None:
        exact = run_generic_test.parse_arguments(["BIOMD0000000079"])
        savgol = run_generic_test.parse_arguments(
            ["BIOMD0000000079", "--derivative-method", "savgol"]
        )

        self.assertIs(exact.derivative_method, EXACT_BENCHMARK_VARIANT)
        self.assertIs(savgol.derivative_method, SAVGOL_BENCHMARK_VARIANT)

    def test_variants_resolve_to_distinct_namespaces(self) -> None:
        exact = resolve_test_experiment_root(
            model_id="BIOMD0000000079",
            variant=EXACT_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )
        savgol = resolve_test_experiment_root(
            model_id="BIOMD0000000079",
            variant=SAVGOL_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )

        self.assertEqual(
            exact,
            Path("/project/results/BIOMD0000000079/generic_exact_experiment"),
        )
        self.assertEqual(
            savgol,
            Path("/project/results/BIOMD0000000079/generic_savgol_experiment"),
        )

    def test_package_entrypoint_help_supports_both_variants(self) -> None:
        for derivative_method in ("exact", "savgol"):
            with self.subTest(derivative_method=derivative_method):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.run_generic_test",
                        "BIOMD0000000079",
                        "--derivative-method",
                        derivative_method,
                        "--help",
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    check=False,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0, msg=completed.stderr)
                self.assertIn("--derivative-method", completed.stdout)


class HeldOutInformationBoundaryTests(unittest.TestCase):
    """Protect the SG state-only held-out information boundary."""

    def test_savgol_materialization_never_calls_exact_simulation(self) -> None:
        expected = build_state_only_test_frame()

        with (
            patch(
                "src.run_generic_test.simulate_state_only_initial_condition_set",
                return_value=expected,
            ) as state_only,
            patch("src.run_generic_test.simulate_initial_condition_set") as exact,
        ):
            observed = run_generic_test.materialize_test_frame(
                spec=build_spec(),
                initial_conditions=np.ones((8, 3)),
                duration=40.0,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        self.assertIs(observed, expected)
        state_only.assert_called_once()
        exact.assert_not_called()

    def test_default_exact_materialization_is_unchanged(self) -> None:
        expected = pd.DataFrame()

        with (
            patch(
                "src.run_generic_test.simulate_initial_condition_set",
                return_value=expected,
            ) as exact,
            patch(
                "src.run_generic_test.simulate_state_only_initial_condition_set"
            ) as state_only,
        ):
            observed = run_generic_test.materialize_test_frame(
                spec=build_spec(),
                initial_conditions=np.ones((8, 3)),
                duration=40.0,
            )

        self.assertIs(observed, expected)
        exact.assert_called_once()
        state_only.assert_not_called()

    def test_savgol_test_schema_rejects_derivative_columns(self) -> None:
        frame = build_state_only_test_frame()
        frame["d_P_dt"] = 0.0

        with self.assertRaisesRegex(ValueError, "must contain only"):
            run_generic_test.validate_test_frame(
                test_frame=frame,
                spec=build_spec(),
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_selection_validation_precedes_test_materialization(self) -> None:
        preflight = SimpleNamespace(
            model_id="BIOMD0000000079",
            spec=build_spec(),
            simulation_time=40.0,
        )
        design = SimpleNamespace(test=np.ones((8, 3)))

        with (
            patch("src.run_generic_test.run_preflight", return_value=preflight),
            patch("src.run_generic_test.print_preflight"),
            patch("src.run_generic_test.require_model_compatibility"),
            patch(
                "src.run_generic_test.build_experimental_design",
                return_value=design,
            ),
            patch(
                "src.run_generic_test.load_frozen_selected_systems",
                side_effect=ValueError("selection provenance mismatch"),
            ),
            patch("src.run_generic_test.materialize_test_frame") as materialize,
            self.assertRaisesRegex(ValueError, "selection provenance mismatch"),
        ):
            run_generic_test.run_model(
                model_id="BIOMD0000000079",
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        materialize.assert_not_called()

    def test_runner_passes_prevalidated_selections_to_evaluator(self) -> None:
        preflight = SimpleNamespace(
            model_id="BIOMD0000000079",
            spec=build_spec(),
            simulation_time=40.0,
        )
        design = SimpleNamespace(test=np.ones((8, 3)))
        test_frame = build_state_only_test_frame()
        selections = {
            dataset_name: (
                pd.Series(
                    {
                        "system_id": system_id,
                        "candidate_rank_signature": signature,
                    }
                ),
                {},
            )
            for dataset_name, system_id, signature in (
                ("train_04", 440, "7,7,8"),
                ("train_08", 28, "1,5,4"),
                ("train_16", 93, "3,3,3"),
                ("train_32", 238, "5,8,4"),
            )
        }

        with (
            patch("src.run_generic_test.run_preflight", return_value=preflight),
            patch("src.run_generic_test.print_preflight"),
            patch("src.run_generic_test.require_model_compatibility"),
            patch(
                "src.run_generic_test.build_experimental_design",
                return_value=design,
            ),
            patch(
                "src.run_generic_test.load_frozen_selected_systems",
                return_value=selections,
            ),
            patch(
                "src.run_generic_test.materialize_test_frame",
                return_value=test_frame,
            ),
            patch(
                "src.run_generic_test.evaluate_all_selected_systems",
                return_value=pd.DataFrame(index=range(4)),
            ) as evaluate,
        ):
            run_generic_test.run_model(
                model_id="BIOMD0000000079",
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        self.assertIs(evaluate.call_args.kwargs["frozen_selections"], selections)
        self.assertIs(
            evaluate.call_args.kwargs["variant"],
            SAVGOL_BENCHMARK_VARIANT,
        )


class FrozenSelectionTests(unittest.TestCase):
    """Verify frozen systems and output paths are variant isolated."""

    def test_loader_reads_coordinate_selection_not_exhaustive_optimum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = Path(temporary_directory)
            dataset_root = experiment_root / "train_04"
            dataset_root.mkdir()
            write_coordinate_selection(
                experiment_root=experiment_root,
                dataset_name="train_04",
                system_id=440,
                ranks=(7, 7, 8),
            )
            pd.DataFrame([{"system_id": 39}]).to_csv(
                dataset_root / "selected_system.csv",
                index=False,
            )

            selected, _ = load_selected_system(
                experiment_dir=dataset_root,
                spec=build_spec(),
            )

            self.assertEqual(int(selected["system_id"]), 440)
            self.assertEqual(selected["candidate_rank_signature"], "7,7,8")

    def test_all_frozen_selections_are_loaded_from_savgol_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            experiment_root = (
                project_root
                / "results"
                / "BIOMD0000000079"
                / "generic_savgol_experiment"
            )
            write_metadata(
                experiment_root=experiment_root,
                derivative_method="savgol",
                experiment_namespace="generic_savgol_experiment",
            )

            expected_ids = (440, 28, 93, 238)
            expected_ranks = ((7, 7, 8), (1, 5, 4), (3, 3, 3), (5, 8, 4))

            for dataset_name, system_id, ranks in zip(
                TRAINING_DATASETS,
                expected_ids,
                expected_ranks,
            ):
                write_coordinate_selection(
                    experiment_root=experiment_root,
                    dataset_name=dataset_name,
                    system_id=system_id,
                    ranks=ranks,
                )

            selections = load_frozen_selected_systems(
                spec=build_spec(),
                variant=SAVGOL_BENCHMARK_VARIANT,
                project_root=project_root,
            )

            self.assertEqual(
                [int(selections[name][0]["system_id"]) for name in TRAINING_DATASETS],
                list(expected_ids),
            )

    def test_provenance_mismatch_fails_before_test_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            experiment_root = (
                project_root
                / "results"
                / "BIOMD0000000079"
                / "generic_savgol_experiment"
            )
            write_metadata(
                experiment_root=experiment_root,
                derivative_method="exact",
                experiment_namespace="generic_savgol_experiment",
                dataset_names=("train_04",),
            )
            output_dir = experiment_root / "train_04" / "coordinate_selection" / "test"

            with (
                patch("src.evaluate_generic_test.compile_rhs_system") as compile_rhs,
                self.assertRaisesRegex(ValueError, "Candidate provenance mismatch"),
            ):
                evaluate_selected_system_on_test(
                    spec=build_spec(),
                    dataset_name="train_04",
                    test_frame=build_state_only_test_frame(),
                    variant=SAVGOL_BENCHMARK_VARIANT,
                    project_root=project_root,
                )

            self.assertFalse(output_dir.exists())
            compile_rhs.assert_not_called()

    def test_aggregate_output_uses_savgol_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            experiment_root = (
                project_root
                / "results"
                / "BIOMD0000000079"
                / "generic_savgol_experiment"
            )
            write_metadata(
                experiment_root=experiment_root,
                derivative_method="savgol",
                experiment_namespace="generic_savgol_experiment",
            )

            for dataset_name in TRAINING_DATASETS:
                write_coordinate_selection(
                    experiment_root=experiment_root,
                    dataset_name=dataset_name,
                    system_id=1,
                    ranks=(1, 1, 1),
                )

            summary = {
                "dataset": "train_04",
                "selection_method": "coordinate_search",
                "system_id": 1,
                "validation_macro_nrmse": 0.1,
                "test_macro_nrmse": 0.2,
                "test_status": "success",
                "successful_trajectories": 8,
                "expected_trajectories": 8,
                "mean_normalized_rmse_P": 0.1,
                "mean_normalized_rmse_Q": 0.2,
                "mean_normalized_rmse_R": 0.3,
            }

            with patch(
                "src.evaluate_generic_test.evaluate_selected_system_on_test",
                return_value=summary,
            ):
                evaluate_all_selected_systems(
                    spec=build_spec(),
                    test_frame=build_state_only_test_frame(),
                    variant=SAVGOL_BENCHMARK_VARIANT,
                    project_root=project_root,
                )

            expected_output = (
                experiment_root
                / "coordinate_selection"
                / "test"
                / "test_comparison.csv"
            )
            exact_output = (
                project_root
                / "results"
                / "BIOMD0000000079"
                / "generic_exact_experiment"
                / "coordinate_selection"
                / "test"
                / "test_comparison.csv"
            )
            self.assertTrue(expected_output.exists())
            self.assertFalse(exact_output.exists())


if __name__ == "__main__":
    unittest.main()
