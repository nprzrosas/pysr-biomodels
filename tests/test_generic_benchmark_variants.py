"""Tests for variant-aware generic benchmark orchestration."""

import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    SAVGOL_BENCHMARK_VARIANT,
    validate_derivative_provenance,
)


STATE_IDS = ("P", "Q", "R")
RUNNER_MODULE_NAMES = (
    "src.run_generic_benchmark",
    "src.run_pysr_generic",
)
MISSING_MODULE = object()


class UnavailablePySRRegressor:
    """Fail if an unmocked test attempts to construct PySR."""

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("Tests must mock PySRRegressor construction.")


@contextlib.contextmanager
def isolated_runner_modules():
    """Import runner modules with a temporary, fully restored PySR stub.

    Yields:
        The generic benchmark and generic PySR runner modules.
    """
    import src

    saved_modules = {
        name: sys.modules.get(name, MISSING_MODULE)
        for name in RUNNER_MODULE_NAMES
    }
    saved_package_attributes = {
        name.rsplit(".", maxsplit=1)[1]: getattr(
            src,
            name.rsplit(".", maxsplit=1)[1],
            MISSING_MODULE,
        )
        for name in RUNNER_MODULE_NAMES
    }
    saved_pysr = sys.modules.get("pysr", MISSING_MODULE)

    for name in RUNNER_MODULE_NAMES:
        sys.modules.pop(name, None)

    pysr_stub = ModuleType("pysr")
    pysr_stub.PySRRegressor = UnavailablePySRRegressor
    sys.modules["pysr"] = pysr_stub

    try:
        benchmark_module = importlib.import_module(
            "src.run_generic_benchmark"
        )
        pysr_module = importlib.import_module("src.run_pysr_generic")
        yield benchmark_module, pysr_module
    finally:
        for name in RUNNER_MODULE_NAMES:
            sys.modules.pop(name, None)

        for name, module in saved_modules.items():
            if module is not MISSING_MODULE:
                sys.modules[name] = module

        for attribute, value in saved_package_attributes.items():
            if value is MISSING_MODULE:
                if hasattr(src, attribute):
                    delattr(src, attribute)
            else:
                setattr(src, attribute, value)

        if saved_pysr is MISSING_MODULE:
            sys.modules.pop("pysr", None)
        else:
            sys.modules["pysr"] = saved_pysr


class IsolatedRunnerTestCase(unittest.TestCase):
    """Provide runner modules without importing or retaining real PySR."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner_context = isolated_runner_modules()
        cls.benchmark_module, cls.pysr_module = cls.runner_context.__enter__()
        cls.addClassCleanup(
            cls.runner_context.__exit__,
            None,
            None,
            None,
        )


def build_state_frame(
    *,
    trajectory_count: int,
    point_count: int,
    trajectory_prefix: str,
) -> pd.DataFrame:
    """Build deterministic state-only trajectories for runner tests.

    Args:
        trajectory_count: Number of synthetic trajectories.
        point_count: Number of samples in each trajectory.
        trajectory_prefix: Prefix for stable trajectory identifiers.

    Returns:
        State-only multitrajectory table in ModelSpec state order.
    """
    frames = []

    for index in range(trajectory_count):
        time_values = np.linspace(0.0, 10.0, point_count)
        offset = float(index)
        frames.append(
            pd.DataFrame(
                {
                    "trajectory_id": f"{trajectory_prefix}_{index:03d}",
                    "time": time_values,
                    "P": time_values + offset,
                    "Q": time_values**2 + offset,
                    "R": time_values**3 + offset,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def build_provenance(variant) -> dict[str, object]:
    """Build the minimum valid lineage mapping for a benchmark variant.

    Args:
        variant: Frozen exact or Savitzky-Golay benchmark variant.

    Returns:
        Matching derivative-method and experiment-namespace provenance.
    """
    return {
        "derivative_method": variant.derivative_method.value,
        "experiment_namespace": variant.experiment_namespace,
    }


class VariantCliAndPathTests(IsolatedRunnerTestCase):
    """Validate CLI resolution and protected result namespaces."""

    def test_exact_resolves_to_exact_namespace(self) -> None:
        arguments = self.benchmark_module.parse_arguments(
            ["BIOMD0000000079", "--derivative-method", "exact"]
        )
        root = self.pysr_module.resolve_experiment_root(
            model_id=arguments.model_id,
            variant=arguments.derivative_method,
            project_root=Path("/project"),
        )

        self.assertIs(arguments.derivative_method, EXACT_BENCHMARK_VARIANT)
        self.assertEqual(
            root,
            Path("/project/results/BIOMD0000000079/generic_exact_experiment"),
        )

    def test_savgol_resolves_to_savgol_namespace(self) -> None:
        arguments = self.benchmark_module.parse_arguments(
            ["BIOMD0000000079", "--derivative-method", "savgol"]
        )
        root = self.pysr_module.resolve_experiment_root(
            model_id=arguments.model_id,
            variant=arguments.derivative_method,
            project_root=Path("/project"),
        )

        self.assertIs(arguments.derivative_method, SAVGOL_BENCHMARK_VARIANT)
        self.assertEqual(
            root,
            Path("/project/results/BIOMD0000000079/generic_savgol_experiment"),
        )

    def test_cli_default_remains_exact(self) -> None:
        arguments = self.benchmark_module.parse_arguments(
            ["BIOMD0000000079"]
        )

        self.assertIs(arguments.derivative_method, EXACT_BENCHMARK_VARIANT)

    def test_unknown_derivative_method_fails_clearly(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.benchmark_module.parse_arguments(
                ["BIOMD0000000079", "--derivative-method", "unknown"]
            )

        self.assertIn("Unknown derivative method 'unknown'", stderr.getvalue())

    def test_variants_resolve_to_distinct_namespaces(self) -> None:
        exact_root = self.pysr_module.resolve_experiment_root(
            model_id="BIOMD0000000079",
            variant=EXACT_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )
        savgol_root = self.pysr_module.resolve_experiment_root(
            model_id="BIOMD0000000079",
            variant=SAVGOL_BENCHMARK_VARIANT,
            project_root=Path("/project"),
        )

        self.assertNotEqual(exact_root, savgol_root)


class LineageValidationTests(IsolatedRunnerTestCase):
    """Reject dataset and benchmark-variant lineage inconsistencies."""

    def setUp(self) -> None:
        self.spec = SimpleNamespace(
            model_id="BIOMD0000000079",
            state_ids=STATE_IDS,
            n_states=len(STATE_IDS),
        )

    def test_exact_provenance_with_savgol_variant_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "expected derivative_method='savgol'.*observed "
            "derivative_method='exact'",
        ):
            validate_derivative_provenance(
                provenance=build_provenance(EXACT_BENCHMARK_VARIANT),
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_savgol_provenance_with_exact_variant_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "expected derivative_method='exact'.*observed "
            "derivative_method='savgol'",
        ):
            validate_derivative_provenance(
                provenance=build_provenance(SAVGOL_BENCHMARK_VARIANT),
                variant=EXACT_BENCHMARK_VARIANT,
            )

    def test_inconsistent_namespace_raises(self) -> None:
        provenance = build_provenance(SAVGOL_BENCHMARK_VARIANT)
        provenance["experiment_namespace"] = "generic_exact_experiment"

        with self.assertRaisesRegex(
            ValueError,
            "expected experiment_namespace='generic_savgol_experiment'.*"
            "observed experiment_namespace='generic_exact_experiment'",
        ):
            validate_derivative_provenance(
                provenance=provenance,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_missing_derivative_method_raises(self) -> None:
        provenance = {
            "experiment_namespace": "generic_exact_experiment",
        }

        with self.assertRaisesRegex(
            ValueError,
            "observed derivative_method='<missing>'",
        ):
            validate_derivative_provenance(
                provenance=provenance,
                variant=EXACT_BENCHMARK_VARIANT,
            )

    def test_missing_experiment_namespace_raises(self) -> None:
        provenance = {
            "derivative_method": "exact",
        }

        with self.assertRaisesRegex(
            ValueError,
            "observed experiment_namespace='<missing>'",
        ):
            validate_derivative_provenance(
                provenance=provenance,
                variant=EXACT_BENCHMARK_VARIANT,
            )

    def test_metadata_mismatch_fails_before_directory_creation(self) -> None:
        dataset = SimpleNamespace(
            derivative_provenance=build_provenance(EXACT_BENCHMARK_VARIANT),
        )
        result = SimpleNamespace(model_id=self.spec.model_id)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "must_not_exist"

            with (
                patch.object(
                    self.pysr_module,
                    "PROJECT_ROOT",
                    project_root,
                ),
                self.assertRaises(ValueError),
            ):
                self.benchmark_module.write_experiment_metadata(
                    result=result,
                    design=SimpleNamespace(),
                    dataset=dataset,
                    variant=SAVGOL_BENCHMARK_VARIANT,
                )

            self.assertFalse(project_root.exists())

    def test_training_mismatch_fails_before_pysr_construction(self) -> None:
        constructor = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "must_not_exist"

            with (
                patch.object(
                    self.pysr_module,
                    "PROJECT_ROOT",
                    project_root,
                ),
                patch.object(
                    self.pysr_module,
                    "PySRRegressor",
                    constructor,
                ),
                self.assertRaises(ValueError),
            ):
                self.pysr_module.run_generic_training_experiment(
                    spec=self.spec,
                    dataset_name="train_04",
                    train_frame=pd.DataFrame(),
                    variant=SAVGOL_BENCHMARK_VARIANT,
                    derivative_provenance=build_provenance(
                        EXACT_BENCHMARK_VARIANT
                    ),
                )

            self.assertFalse(project_root.exists())

        constructor.assert_not_called()

    def test_consistent_exact_lineage_succeeds(self) -> None:
        validate_derivative_provenance(
            provenance=build_provenance(EXACT_BENCHMARK_VARIANT),
            variant=EXACT_BENCHMARK_VARIANT,
        )

    def test_consistent_savgol_lineage_succeeds(self) -> None:
        provenance = {
            "derivative_method": SAVGOL_BENCHMARK_VARIANT.derivative_method,
            "experiment_namespace": (
                SAVGOL_BENCHMARK_VARIANT.experiment_namespace
            ),
        }

        validate_derivative_provenance(
            provenance=provenance,
            variant=SAVGOL_BENCHMARK_VARIANT,
        )

    def test_metadata_identity_conflict_raises_instead_of_overwriting(
        self,
    ) -> None:
        provenance = build_provenance(EXACT_BENCHMARK_VARIANT)
        provenance["experiment_namespace"] = "generic_savgol_experiment"
        dataset = SimpleNamespace(derivative_provenance=provenance)

        with self.assertRaises(ValueError):
            self.benchmark_module.write_experiment_metadata(
                result=SimpleNamespace(model_id=self.spec.model_id),
                design=SimpleNamespace(),
                dataset=dataset,
                variant=EXACT_BENCHMARK_VARIANT,
            )

    def test_frozen_pysr_constructor_configuration_and_patch_restoration(
        self,
    ) -> None:
        original_constructor = self.pysr_module.PySRRegressor
        constructor = Mock(return_value=object())

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                self.pysr_module,
                "PySRRegressor",
                constructor,
            ):
                model = self.pysr_module.build_pysr_model(
                    output_dir=Path(temporary_directory) / "state_P",
                )

            self.assertIsNotNone(model)

        self.assertIs(
            self.pysr_module.PySRRegressor,
            original_constructor,
        )
        constructor.assert_called_once()
        constructor_arguments = constructor.call_args.kwargs
        expected_arguments = {
            "niterations": 60,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [],
            "maxsize": 20,
            "parsimony": 0.0,
            "weight_optimize": 0.0,
            "should_optimize_constants": True,
            "optimizer_algorithm": "BFGS",
            "model_selection": "best",
            "random_state": 1010,
            "parallelism": "serial",
            "deterministic": True,
        }

        for key, value in expected_arguments.items():
            self.assertEqual(constructor_arguments[key], value)


class VariantAwareRunnerTests(IsolatedRunnerTestCase):
    """Validate runner data flow without invoking PySR."""

    def setUp(self) -> None:
        self.spec = SimpleNamespace(
            model_id="BIOMD0000000079",
            state_ids=STATE_IDS,
            n_states=len(STATE_IDS),
        )
        self.preflight = SimpleNamespace(
            model_id=self.spec.model_id,
            spec=self.spec,
            simulation_time=40.0,
        )
        self.design = SimpleNamespace(
            train=np.ones((32, len(STATE_IDS))),
            validation=np.ones((8, len(STATE_IDS))),
            test=np.ones((8, len(STATE_IDS))),
        )

    @staticmethod
    def fake_state_only_set(**kwargs) -> pd.DataFrame:
        """Return state-only data for a mocked generic simulator.

        Args:
            **kwargs: Generic simulation arguments.

        Returns:
            Synthetic state-only trajectories.
        """
        return build_state_frame(
            trajectory_count=len(kwargs["initial_conditions"]),
            point_count=kwargs["points"],
            trajectory_prefix=kwargs["trajectory_prefix"],
        )

    def test_savgol_runner_passes_targets_without_exact_or_test_data(
        self,
    ) -> None:
        with (
            patch(
                "src.run_generic_benchmark.run_preflight",
                return_value=self.preflight,
            ),
            patch("src.run_generic_benchmark.print_preflight"),
            patch("src.run_generic_benchmark.require_model_compatibility"),
            patch(
                "src.run_generic_benchmark.build_experimental_design",
                return_value=self.design,
            ),
            patch(
                "src.simulate_generic.simulate_state_only_initial_condition_set",
                side_effect=self.fake_state_only_set,
            ),
            patch(
                "src.simulate_generic.simulate_initial_condition_set",
                side_effect=AssertionError("Exact simulation path called."),
            ) as exact_simulation,
            patch(
                "src.simulate_generic._calculate_exact_derivatives",
                side_effect=AssertionError("Exact derivative generator called."),
            ) as exact_derivatives,
            patch(
                "src.run_generic_benchmark.write_experiment_metadata"
            ) as metadata,
            patch(
                "src.run_generic_benchmark.run_generic_training_experiment"
            ) as training,
            patch("builtins.print"),
        ):
            self.benchmark_module.run_model(
                model_id=self.spec.model_id,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        exact_simulation.assert_not_called()
        exact_derivatives.assert_not_called()
        self.assertEqual(training.call_count, 4)

        expected_training_columns = [
            "trajectory_id",
            "time",
            "P",
            "Q",
            "R",
            "d_P_dt",
            "d_Q_dt",
            "d_R_dt",
        ]

        for call in training.call_args_list:
            self.assertIs(call.kwargs["variant"], SAVGOL_BENCHMARK_VARIANT)
            self.assertEqual(
                call.kwargs["derivative_provenance"]["derivative_method"],
                "savgol",
            )
            self.assertEqual(
                list(call.kwargs["train_frame"].columns),
                expected_training_columns,
            )

        development_dataset = metadata.call_args.kwargs["dataset"]
        self.assertEqual(
            list(development_dataset.validation.columns),
            ["trajectory_id", "time", "P", "Q", "R"],
        )
        self.assertTrue(development_dataset.test.empty)
        self.assertIs(
            metadata.call_args.kwargs["variant"],
            SAVGOL_BENCHMARK_VARIANT,
        )

    def test_run_model_default_remains_exact(self) -> None:
        train_frame = pd.DataFrame(
            {
                "trajectory_id": ["train_000"],
                "time": [0.0],
                "P": [1.0],
                "Q": [1.0],
                "R": [1.0],
                "d_P_dt": [0.0],
                "d_Q_dt": [0.0],
                "d_R_dt": [0.0],
            }
        )
        dataset = SimpleNamespace(
            train_04=train_frame,
            train_08=train_frame,
            train_16=train_frame,
            train_32=train_frame,
            validation=pd.DataFrame(
                {"trajectory_id": ["validation_000"]}
            ),
            test=pd.DataFrame(),
            preprocessing_seconds=0.0,
            derivative_provenance={
                "derivative_method": "exact",
                "experiment_namespace": "generic_exact_experiment",
            },
        )

        with (
            patch(
                "src.run_generic_benchmark.run_preflight",
                return_value=self.preflight,
            ),
            patch("src.run_generic_benchmark.print_preflight"),
            patch("src.run_generic_benchmark.require_model_compatibility"),
            patch(
                "src.run_generic_benchmark.build_experimental_design",
                return_value=self.design,
            ),
            patch(
                "src.run_generic_benchmark.build_generic_development_dataset",
                return_value=dataset,
            ) as build_dataset,
            patch("src.run_generic_benchmark.validate_development_dataset"),
            patch("src.run_generic_benchmark.write_experiment_metadata"),
            patch(
                "src.run_generic_benchmark.run_generic_training_experiment"
            ) as training,
            patch("builtins.print"),
        ):
            self.benchmark_module.run_model(model_id=self.spec.model_id)

        self.assertIs(
            build_dataset.call_args.kwargs["variant"],
            EXACT_BENCHMARK_VARIANT,
        )
        self.assertEqual(training.call_count, 4)
        for call in training.call_args_list:
            self.assertIs(call.kwargs["variant"], EXACT_BENCHMARK_VARIANT)
            self.assertEqual(
                call.kwargs["derivative_provenance"],
                dataset.derivative_provenance,
            )

    def test_run_model_rejects_lineage_mismatch_before_output_calls(
        self,
    ) -> None:
        dataset = SimpleNamespace(
            derivative_provenance=build_provenance(EXACT_BENCHMARK_VARIANT),
        )

        with (
            patch(
                "src.run_generic_benchmark.run_preflight",
                return_value=self.preflight,
            ),
            patch("src.run_generic_benchmark.print_preflight"),
            patch("src.run_generic_benchmark.require_model_compatibility"),
            patch(
                "src.run_generic_benchmark.build_experimental_design",
                return_value=self.design,
            ),
            patch(
                "src.run_generic_benchmark.build_generic_development_dataset",
                return_value=dataset,
            ),
            patch(
                "src.run_generic_benchmark.validate_development_dataset"
            ) as validate_dataset,
            patch(
                "src.run_generic_benchmark.write_experiment_metadata"
            ) as write_metadata,
            patch(
                "src.run_generic_benchmark.run_generic_training_experiment"
            ) as training,
            patch("builtins.print"),
            self.assertRaises(ValueError),
        ):
            self.benchmark_module.run_model(
                model_id=self.spec.model_id,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        validate_dataset.assert_not_called()
        write_metadata.assert_not_called()
        training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
