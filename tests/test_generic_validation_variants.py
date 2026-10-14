"""Regression tests for variant-aware validation and Selector v1."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src import run_generic_validation
from src import select_generic_system
from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
)
from src.evaluate_generic_validation import (
    ATOL,
    MAX_RHS_EVALUATIONS,
    RTOL,
    SOLVER_METHOD,
    TRAINING_DATASETS,
    build_candidate_systems,
    evaluate_validation_dataset,
    resolve_validation_experiment_root,
    validate_candidate_pool_provenance,
    validate_selection_development_dataset,
)
from src.simulate_generic import build_generic_development_dataset


STATE_IDS = ("P", "Q", "R")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_spec() -> SimpleNamespace:
    """Build minimal ordered model metadata for validation tests.

    Returns:
        A ModelSpec-compatible synthetic object.
    """
    return SimpleNamespace(
        model_id="BIOMD0000000079",
        model_path=Path("synthetic.xml"),
        state_ids=STATE_IDS,
        n_states=len(STATE_IDS),
    )


def write_candidate_metadata(
    *,
    experiment_root: Path,
    derivative_method: str,
    experiment_namespace: str | None,
    dataset_names=TRAINING_DATASETS,
) -> None:
    """Write minimal synthetic candidate provenance files.

    Args:
        experiment_root: Temporary experiment directory.
        derivative_method: Recorded candidate derivative method.
        experiment_namespace: Recorded namespace, or ``None`` for legacy exact.
        dataset_names: Training conditions to represent.
    """
    row = {
        "derivative_method": derivative_method,
    }

    if experiment_namespace is not None:
        row["experiment_namespace"] = experiment_namespace

    experiment_root.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(
        experiment_root / "experiment_metadata.csv",
        index=False,
    )

    for dataset_name in dataset_names:
        dataset_root = experiment_root / dataset_name
        dataset_root.mkdir()
        pd.DataFrame([row]).to_csv(
            dataset_root / "run_metadata.csv",
            index=False,
        )


def selected_system_frame() -> pd.DataFrame:
    """Build one selected-row fixture used by runner orchestration tests.

    Returns:
        A one-row selected-system table.
    """
    row = {
        "system_id": 1,
        "validation_macro_nrmse": 0.0,
        "complexity_total": 3,
    }

    for state_id in STATE_IDS:
        row[f"candidate_rank_{state_id}"] = 1
        row[f"equation_{state_id}"] = state_id

    return pd.DataFrame([row])


class VariantResolutionTests(unittest.TestCase):
    """Verify validation and Selector CLI defaults and namespaces."""

    def test_validation_cli_default_remains_exact(self) -> None:
        arguments = run_generic_validation.parse_arguments(
            ["BIOMD0000000079"]
        )

        self.assertIs(arguments.derivative_method, EXACT_BENCHMARK_VARIANT)

    def test_selector_cli_default_remains_exact(self) -> None:
        arguments = select_generic_system.parse_arguments(
            ["BIOMD0000000079"]
        )

        self.assertIs(arguments.derivative_method, EXACT_BENCHMARK_VARIANT)

    def test_exact_resolves_to_exact_namespace(self) -> None:
        root = resolve_validation_experiment_root(
            model_id="BIOMD0000000079",
            variant=EXACT_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )

        self.assertEqual(
            root,
            Path("/project/results/BIOMD0000000079/generic_exact_experiment"),
        )

    def test_savgol_resolves_to_savgol_namespace(self) -> None:
        root = resolve_validation_experiment_root(
            model_id="BIOMD0000000079",
            variant=SAVGOL_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )

        self.assertEqual(
            root,
            Path("/project/results/BIOMD0000000079/generic_savgol_experiment"),
        )

    def test_validation_cli_resolves_savgol(self) -> None:
        arguments = run_generic_validation.parse_arguments(
            ["BIOMD0000000079", "--derivative-method", "savgol"]
        )

        self.assertIs(arguments.derivative_method, SAVGOL_BENCHMARK_VARIANT)


class PackageEntrypointTests(unittest.TestCase):
    """Ensure documented package CLIs import and expose argument help."""

    def test_validation_and_selector_package_entrypoints(self) -> None:
        for module_name in (
            "src.run_generic_validation",
            "src.select_generic_system",
        ):
            for derivative_method in ("exact", "savgol"):
                with self.subTest(
                    module_name=module_name,
                    derivative_method=derivative_method,
                ):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            module_name,
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

                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=completed.stderr,
                    )
                    self.assertIn(
                        "--derivative-method",
                        completed.stdout,
                    )


class CandidateProvenanceTests(unittest.TestCase):
    """Require candidate lineage to match the protected namespace."""

    def test_consistent_savgol_candidate_provenance_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / SAVGOL_BENCHMARK_VARIANT.experiment_namespace
            )
            write_candidate_metadata(
                experiment_root=experiment_root,
                derivative_method="savgol",
                experiment_namespace="generic_savgol_experiment",
            )

            validate_candidate_pool_provenance(
                experiment_root=experiment_root,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_frozen_legacy_exact_metadata_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / EXACT_BENCHMARK_VARIANT.experiment_namespace
            )
            write_candidate_metadata(
                experiment_root=experiment_root,
                derivative_method="exact",
                experiment_namespace=None,
            )

            validate_candidate_pool_provenance(
                experiment_root=experiment_root,
                variant=EXACT_BENCHMARK_VARIANT,
            )

    def test_variant_namespace_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / EXACT_BENCHMARK_VARIANT.experiment_namespace
            )
            output_path = experiment_root / "selection_must_not_exist.csv"

            with self.assertRaisesRegex(
                ValueError,
                "expected experiment_namespace='generic_savgol_experiment'",
            ):
                validate_candidate_pool_provenance(
                    experiment_root=experiment_root,
                    variant=SAVGOL_BENCHMARK_VARIANT,
                )

            self.assertFalse(output_path.exists())
            self.assertFalse(experiment_root.exists())

    def test_candidate_method_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / SAVGOL_BENCHMARK_VARIANT.experiment_namespace
            )
            write_candidate_metadata(
                experiment_root=experiment_root,
                derivative_method="exact",
                experiment_namespace="generic_savgol_experiment",
            )

            with self.assertRaisesRegex(
                ValueError,
                "observed derivative_method='exact'",
            ):
                validate_candidate_pool_provenance(
                    experiment_root=experiment_root,
                    variant=SAVGOL_BENCHMARK_VARIANT,
                )

    def test_evaluator_mismatch_fails_before_candidate_or_csv_work(self) -> None:
        spec = build_spec()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            experiment_root = resolve_validation_experiment_root(
                model_id=spec.model_id,
                variant=SAVGOL_BENCHMARK_VARIANT,
                project_root=project_root,
            )
            write_candidate_metadata(
                experiment_root=experiment_root,
                derivative_method="exact",
                experiment_namespace="generic_savgol_experiment",
                dataset_names=("train_04",),
            )
            output_path = (
                experiment_root / "train_04" / "validation_system_ranking.csv"
            )

            with (
                patch(
                    "src.evaluate_generic_validation.load_state_candidates"
                ) as load_candidates,
                self.assertRaisesRegex(
                    ValueError,
                    "observed derivative_method='exact'",
                ),
            ):
                evaluate_validation_dataset(
                    spec=spec,
                    dataset_name="train_04",
                    validation_frame=pd.DataFrame(),
                    variant=SAVGOL_BENCHMARK_VARIANT,
                    project_root=project_root,
                )

            load_candidates.assert_not_called()
            self.assertFalse(output_path.exists())

    def test_missing_candidate_provenance_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / SAVGOL_BENCHMARK_VARIANT.experiment_namespace
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing derivative provenance metadata",
            ):
                validate_candidate_pool_provenance(
                    experiment_root=experiment_root,
                    variant=SAVGOL_BENCHMARK_VARIANT,
                )

    def test_savgol_missing_namespace_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_root = (
                Path(temporary_directory)
                / SAVGOL_BENCHMARK_VARIANT.experiment_namespace
            )
            write_candidate_metadata(
                experiment_root=experiment_root,
                derivative_method="savgol",
                experiment_namespace=None,
            )

            with self.assertRaisesRegex(
                ValueError,
                "observed experiment_namespace='<missing>'",
            ):
                validate_candidate_pool_provenance(
                    experiment_root=experiment_root,
                    variant=SAVGOL_BENCHMARK_VARIANT,
                )

    def test_runner_provenance_failure_precedes_dataset_and_output(self) -> None:
        candidate_validation = Mock(
            side_effect=ValueError("candidate provenance mismatch")
        )

        with (
            patch(
                "src.run_generic_validation.run_preflight",
                return_value=SimpleNamespace(
                    model_id="BIOMD0000000079",
                    spec=build_spec(),
                    simulation_time=40.0,
                ),
            ),
            patch("src.run_generic_validation.require_model_compatibility"),
            patch("src.run_generic_validation.build_experimental_design"),
            patch(
                "src.run_generic_validation.validate_candidate_pool_provenance",
                candidate_validation,
            ),
            patch(
                "src.run_generic_validation.build_generic_development_dataset"
            ) as build_dataset,
            patch(
                "src.run_generic_validation.evaluate_validation_dataset"
            ) as evaluate,
            self.assertRaisesRegex(ValueError, "candidate provenance mismatch"),
        ):
            run_generic_validation.run_validation(
                model_id="BIOMD0000000079",
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        build_dataset.assert_not_called()
        evaluate.assert_not_called()

    def test_selector_provenance_failure_precedes_dataset_and_output(
        self,
    ) -> None:
        spec = build_spec()
        preflight = SimpleNamespace(
            spec=spec,
            simulation_time=40.0,
            compatibility=SimpleNamespace(
                compatible=True,
                reasons=(),
            ),
        )

        with (
            patch(
                "src.select_generic_system.run_preflight",
                return_value=preflight,
            ),
            patch("src.select_generic_system.build_experimental_design"),
            patch(
                "src.select_generic_system.validate_candidate_pool_provenance",
                side_effect=ValueError("candidate provenance mismatch"),
            ),
            patch(
                "src.select_generic_system.build_generic_development_dataset"
            ) as build_dataset,
            patch(
                "src.select_generic_system.write_selection_results"
            ) as write_results,
            self.assertRaisesRegex(ValueError, "candidate provenance mismatch"),
        ):
            select_generic_system.run_selection(
                model_id=spec.model_id,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        build_dataset.assert_not_called()
        write_results.assert_not_called()


class SavgolValidationBoundaryTests(unittest.TestCase):
    """Exercise the real SG dataset builder through validation orchestration."""

    def test_savgol_validation_is_state_only_without_exact_rates(self) -> None:
        spec = build_spec()
        design = SimpleNamespace(
            train=np.ones((32, len(STATE_IDS))),
            validation=np.ones((8, len(STATE_IDS))),
            test=np.ones((8, len(STATE_IDS))),
        )
        preflight = SimpleNamespace(
            model_id=spec.model_id,
            spec=spec,
            simulation_time=40.0,
        )
        captured = {}

        class FakeRoadRunner:
            """Provide state trajectories and reject exact-rate access."""

            def __init__(self, _model_path: str) -> None:
                return None

            def resetAll(self) -> None:
                return None

            def reset(self) -> None:
                return None

            def __setitem__(self, _key: str, _value: float) -> None:
                return None

            def simulate(
                self,
                start: float,
                stop: float,
                points: int,
                *,
                selections: list[str],
            ) -> np.ndarray:
                self.assert_state_selections(selections)
                time_values = np.linspace(start, stop, points)
                return np.column_stack(
                    [
                        time_values,
                        time_values + 1.0,
                        time_values**2 + 1.0,
                        time_values**3 + 1.0,
                    ]
                )

            @staticmethod
            def assert_state_selections(selections: list[str]) -> None:
                """Reject any simulation selection beyond time and states.

                Args:
                    selections: RoadRunner output selections.
                """
                if selections != ["time", "P", "Q", "R"]:
                    raise AssertionError(f"Unexpected selections: {selections}")

            def getRatesOfChange(self) -> np.ndarray:
                raise AssertionError("getRatesOfChange must not be called.")

        def capture_dataset(**kwargs):
            dataset = build_generic_development_dataset(**kwargs)
            captured["dataset"] = dataset
            return dataset

        selected = selected_system_frame()

        with (
            patch(
                "src.run_generic_validation.run_preflight",
                return_value=preflight,
            ),
            patch("src.run_generic_validation.require_model_compatibility"),
            patch(
                "src.run_generic_validation.build_experimental_design",
                return_value=design,
            ),
            patch(
                "src.run_generic_validation.validate_candidate_pool_provenance"
            ),
            patch(
                "src.run_generic_validation.build_generic_development_dataset",
                side_effect=capture_dataset,
            ),
            patch(
                "src.run_generic_validation.evaluate_validation_dataset",
                return_value=(pd.DataFrame(), selected),
            ) as evaluate,
            patch(
                "src.simulate_generic.roadrunner.RoadRunner",
                FakeRoadRunner,
            ),
            patch(
                "src.simulate_generic._calculate_exact_derivatives",
                side_effect=AssertionError("Exact derivative generator called."),
            ) as exact_derivatives,
            patch("builtins.print"),
        ):
            run_generic_validation.run_validation(
                model_id=spec.model_id,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        dataset = captured["dataset"]
        expected_columns = ["trajectory_id", "time", "P", "Q", "R"]
        self.assertEqual(list(dataset.validation.columns), expected_columns)
        self.assertFalse(
            any(column.startswith("d_") for column in dataset.validation)
        )
        self.assertTrue(dataset.test.empty)
        self.assertEqual(evaluate.call_count, 4)
        exact_derivatives.assert_not_called()

        for call in evaluate.call_args_list:
            self.assertIs(call.kwargs["variant"], SAVGOL_BENCHMARK_VARIANT)
            self.assertEqual(
                list(call.kwargs["validation_frame"].columns),
                expected_columns,
            )

    def test_savgol_validation_rejects_derivative_columns(self) -> None:
        validation = pd.DataFrame(
            columns=["trajectory_id", "time", "P", "Q", "R", "d_P_dt"]
        )
        dataset = SimpleNamespace(
            validation=validation,
            test=pd.DataFrame(),
            derivative_provenance={
                "derivative_method": "savgol",
                "experiment_namespace": "generic_savgol_experiment",
            },
        )

        with self.assertRaisesRegex(ValueError, "only trajectory_id"):
            validate_selection_development_dataset(
                dataset=dataset,
                spec=build_spec(),
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_exact_validation_dataset_remains_compatible(self) -> None:
        dataset = SimpleNamespace(
            validation=pd.DataFrame(
                columns=[
                    "trajectory_id",
                    "time",
                    "P",
                    "Q",
                    "R",
                    "d_P_dt",
                    "d_Q_dt",
                    "d_R_dt",
                ]
            ),
            test=pd.DataFrame(),
            derivative_provenance={
                "derivative_method": "exact",
                "experiment_namespace": "generic_exact_experiment",
            },
        )

        validate_selection_development_dataset(
            dataset=dataset,
            spec=build_spec(),
            variant=EXACT_BENCHMARK_VARIANT,
        )


class SelectorV1RegressionTests(unittest.TestCase):
    """Lock the frozen scientific constants and search behavior."""

    def test_selector_scientific_constants_are_unchanged(self) -> None:
        self.assertEqual(SOLVER_METHOD, "LSODA")
        self.assertEqual(RTOL, 1e-7)
        self.assertEqual(ATOL, 1e-9)
        self.assertEqual(MAX_RHS_EVALUATIONS, 8000)

    def test_selector_sort_key_is_unchanged(self) -> None:
        summary = {
            "validation_macro_nrmse": 0.25,
            "complexity_total": 11,
            "training_loss_total": 0.01,
            "system_id": 7,
        }

        self.assertEqual(
            select_generic_system.validation_sort_key(summary),
            (0.25, 11, 0.01, 7),
        )

    def test_coordinate_search_preserves_fixed_state_order(self) -> None:
        spec = SimpleNamespace(
            model_id="SYNTHETIC",
            state_ids=("state_b", "state_a"),
            n_states=2,
        )
        candidates = {
            state_id: pd.DataFrame(
                {
                    "candidate_rank": [1, 2],
                    "complexity": [1, 1],
                    "loss": [0.0, 0.0],
                    "equation_original_units": ["0", "1"],
                }
            )
            for state_id in spec.state_ids
        }
        validation = pd.DataFrame(
            {
                "trajectory_id": ["validation_000"],
                "time": [0.0],
                "state_b": [0.0],
                "state_a": [0.0],
            }
        )

        def fake_evaluate_system(**kwargs):
            ranks = tuple(
                int(kwargs["state_rows"][state_id]["candidate_rank"])
                for state_id in spec.state_ids
            )
            return {"ranks": ranks}

        def fake_summarize_system(**kwargs):
            ranks = kwargs["evaluation"]["ranks"]
            return {
                "system_id": kwargs["system_id"],
                "validation_macro_nrmse": float(
                    sum((rank - 2) ** 2 for rank in ranks)
                ),
                "complexity_total": 2,
                "training_loss_total": 0.0,
            }

        with (
            patch(
                "src.select_generic_system.evaluate_system",
                side_effect=fake_evaluate_system,
            ),
            patch(
                "src.select_generic_system.summarize_system",
                side_effect=fake_summarize_system,
            ),
        ):
            selection = select_generic_system.run_coordinate_selection(
                spec=spec,
                dataset_name="train_04",
                candidates=candidates,
                validation_frame=validation,
                state_scales={"state_b": 1.0, "state_a": 1.0},
            )

        self.assertEqual(selection["selected"]["ranks"], (2, 2))
        self.assertEqual(
            selection["evaluation_order"],
            [(1, 1), (2, 1), (2, 2), (1, 2)],
        )

    def test_three_state_exhaustive_cartesian_order_is_unchanged(self) -> None:
        spec = build_spec()
        candidates = {
            state_id: pd.DataFrame(
                {"marker": [f"{state_id}1", f"{state_id}2"]}
            )
            for state_id in spec.state_ids
        }

        systems = build_candidate_systems(
            spec=spec,
            candidates=candidates,
        )

        self.assertEqual(len(systems), 8)
        self.assertEqual(systems[0][0], 1)
        self.assertEqual(systems[-1][0], 8)
        self.assertEqual(
            tuple(systems[0][1][state_id]["marker"] for state_id in STATE_IDS),
            ("P1", "Q1", "R1"),
        )
        self.assertEqual(
            tuple(systems[-1][1][state_id]["marker"] for state_id in STATE_IDS),
            ("P2", "Q2", "R2"),
        )

    def test_canonical_cartesian_system_id_is_unchanged(self) -> None:
        spec = build_spec()
        candidates = {
            "P": pd.DataFrame(index=range(2)),
            "Q": pd.DataFrame(index=range(3)),
            "R": pd.DataFrame(index=range(4)),
        }

        self.assertEqual(
            select_generic_system.canonical_system_id(
                spec=spec,
                candidates=candidates,
                ranks=(1, 1, 1),
            ),
            1,
        )
        self.assertEqual(
            select_generic_system.canonical_system_id(
                spec=spec,
                candidates=candidates,
                ranks=(2, 3, 4),
            ),
            24,
        )


if __name__ == "__main__":
    unittest.main()
