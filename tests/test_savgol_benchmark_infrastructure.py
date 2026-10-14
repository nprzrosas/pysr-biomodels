"""Synthetic tests for the frozen Savitzky-Golay benchmark-v1 path."""

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    EXACT_EXPERIMENT_NAMESPACE,
    SAVGOL_BENCHMARK_VARIANT,
    SAVGOL_EXPERIMENT_NAMESPACE,
    BenchmarkVariant,
    DerivativeMethod,
    ExactDerivativeConfig,
    SavgolDerivativeConfig,
    get_benchmark_variant,
)
from src.derivative_generation_generic import (
    derivative_columns,
    generate_derivatives,
    generate_savgol_derivatives,
    state_only_columns,
)
from src.simulate_generic import (
    build_generic_development_dataset,
    simulate_state_only_trajectory,
    simulate_trajectory,
)

STATE_IDS = ("state_b", "state_a")


def build_spec() -> SimpleNamespace:
    """Build minimal ordered model metadata for synthetic tests.

    Returns:
        A ModelSpec-compatible synthetic object.
    """
    return SimpleNamespace(
        model_id="SYNTHETIC",
        model_path=Path("synthetic.xml"),
        state_ids=STATE_IDS,
        n_states=len(STATE_IDS),
    )


def build_state_frame(
    *,
    trajectory_count: int = 2,
    point_count: int = 101,
    shuffle_rows: bool = False,
    trajectory_prefix: str = "train",
) -> pd.DataFrame:
    """Build noiseless multistate polynomial trajectories.

    Args:
        trajectory_count: Number of independent trajectories.
        point_count: Uniformly spaced samples per trajectory.
        shuffle_rows: Whether to reverse each trajectory's row order.
        trajectory_prefix: Prefix for trajectory identifiers.

    Returns:
        A state-only synthetic trajectory table.
    """
    frames = []

    for index in range(trajectory_count):
        time_values = np.linspace(0.0, 10.0, point_count)
        offset = float(index * 1000)

        frame = pd.DataFrame(
            {
                "trajectory_id": f"{trajectory_prefix}_{index:03d}",
                "time": time_values,
                "state_b": time_values**2 + offset,
                "state_a": 2.0 * time_values**3 - time_values + offset,
            }
        )

        if shuffle_rows:
            frame = frame.iloc[::-1].reset_index(drop=True)

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


class BenchmarkVariantTests(unittest.TestCase):
    """Validate protected derivative-method and namespace associations."""

    def test_frozen_variants_are_resolved_centrally(self) -> None:
        self.assertIs(
            get_benchmark_variant("exact"),
            EXACT_BENCHMARK_VARIANT,
        )
        self.assertIs(
            get_benchmark_variant("savgol"),
            SAVGOL_BENCHMARK_VARIANT,
        )

    def test_savgol_cannot_use_exact_namespace(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkVariant(
                derivative_method=DerivativeMethod.SAVGOL,
                experiment_namespace=EXACT_EXPERIMENT_NAMESPACE,
                derivative_config=SavgolDerivativeConfig(),
            )

    def test_exact_cannot_use_savgol_namespace(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkVariant(
                derivative_method=DerivativeMethod.EXACT,
                experiment_namespace=SAVGOL_EXPERIMENT_NAMESPACE,
                derivative_config=ExactDerivativeConfig(),
            )

    def test_savgol_configuration_is_frozen(self) -> None:
        with self.assertRaises(ValueError):
            SavgolDerivativeConfig(window_length=13)

    def test_savgol_provenance_is_complete(self) -> None:
        provenance = SAVGOL_BENCHMARK_VARIANT.derivative_provenance(
            input_columns=("trajectory_id", "time", *STATE_IDS),
            state_order=STATE_IDS,
        )

        expected = {
            "derivative_method": "savgol",
            "experiment_namespace": "generic_savgol_experiment",
            "derivative_input_columns": ("trajectory_id;time;state_b;state_a"),
            "derivative_state_order": "state_b;state_a",
            "savgol_window_length": 11,
            "savgol_polyorder": 3,
            "savgol_deriv": 1,
            "savgol_mode": "interp",
            "savgol_delta_policy": "uniform_per_trajectory_time_step",
        }

        for key, value in expected.items():
            self.assertEqual(provenance[key], value)


class SavgolDerivativeGenerationTests(unittest.TestCase):
    """Validate multistate SG generation using analytical fixtures."""

    def setUp(self) -> None:
        self.spec = build_spec()

    def test_multistate_schema_order_finiteness_and_derivative_values(
        self,
    ) -> None:
        frame = build_state_frame()
        generated = generate_savgol_derivatives(
            frame=frame,
            spec=self.spec,
            variant=SAVGOL_BENCHMARK_VARIANT,
        ).frame

        expected_columns = (
            *state_only_columns(self.spec),
            *derivative_columns(self.spec),
        )
        self.assertEqual(tuple(generated.columns), expected_columns)
        self.assertTrue(np.isfinite(generated.iloc[:, 1:].to_numpy()).all())

        first = generated[generated["trajectory_id"] == "train_000"]
        time_values = first["time"].to_numpy()

        np.testing.assert_allclose(
            first["d_state_b_dt"].to_numpy(),
            2.0 * time_values,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            first["d_state_a_dt"].to_numpy(),
            6.0 * time_values**2 - 1.0,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_boundary_rows_are_preserved(self) -> None:
        frame = build_state_frame(trajectory_count=1)
        generated = generate_savgol_derivatives(
            frame=frame,
            spec=self.spec,
            variant=SAVGOL_BENCHMARK_VARIANT,
        ).frame

        self.assertEqual(len(generated), len(frame))
        self.assertEqual(generated.iloc[0]["time"], frame.iloc[0]["time"])
        self.assertEqual(generated.iloc[-1]["time"], frame.iloc[-1]["time"])
        self.assertTrue(np.isfinite(generated.iloc[0]["d_state_b_dt"]))
        self.assertTrue(np.isfinite(generated.iloc[-1]["d_state_b_dt"]))

    def test_each_trajectory_is_sorted_by_time(self) -> None:
        frame = build_state_frame(shuffle_rows=True)
        generated = generate_savgol_derivatives(
            frame=frame,
            spec=self.spec,
            variant=SAVGOL_BENCHMARK_VARIANT,
        ).frame

        for _, trajectory in generated.groupby("trajectory_id", sort=False):
            self.assertTrue(np.all(np.diff(trajectory["time"]) > 0.0))

    def test_trajectories_are_filtered_independently(self) -> None:
        combined = build_state_frame(trajectory_count=2)
        together = generate_savgol_derivatives(
            frame=combined,
            spec=self.spec,
            variant=SAVGOL_BENCHMARK_VARIANT,
        ).frame

        for trajectory_id, trajectory in combined.groupby("trajectory_id", sort=False):
            separately = generate_savgol_derivatives(
                frame=trajectory.reset_index(drop=True),
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            ).frame
            from_combined = together[
                together["trajectory_id"] == trajectory_id
            ].reset_index(drop=True)

            pd.testing.assert_frame_equal(from_combined, separately)

    def test_nonuniform_time_spacing_is_rejected(self) -> None:
        frame = build_state_frame(trajectory_count=1)
        frame.loc[5, "time"] += 0.01

        with self.assertRaisesRegex(ValueError, "uniform time sampling"):
            generate_savgol_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_short_trajectory_is_rejected(self) -> None:
        frame = build_state_frame(trajectory_count=1, point_count=10)

        with self.assertRaisesRegex(ValueError, "shorter"):
            generate_savgol_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_derivative_input_column_is_rejected(self) -> None:
        frame = build_state_frame(trajectory_count=1)
        frame["d_state_b_dt"] = 0.0

        with self.assertRaisesRegex(ValueError, "must not contain"):
            generate_savgol_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_state_column_order_is_enforced(self) -> None:
        frame = build_state_frame(trajectory_count=1)
        frame = frame[["trajectory_id", "time", "state_a", "state_b"]]

        with self.assertRaisesRegex(ValueError, "exactly"):
            generate_savgol_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_nonfinite_state_is_rejected(self) -> None:
        frame = build_state_frame(trajectory_count=1)
        frame.loc[4, "state_a"] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            generate_savgol_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

    def test_sg_interface_rejects_exact_generator(self) -> None:
        frame = build_state_frame(trajectory_count=1)

        with self.assertRaisesRegex(ValueError, "must not be exposed"):
            generate_derivatives(
                frame=frame,
                spec=self.spec,
                variant=SAVGOL_BENCHMARK_VARIANT,
                exact_generator=lambda value: value,
            )

    def test_exact_interface_uses_explicit_generator(self) -> None:
        frame = build_state_frame(trajectory_count=1)

        def exact_generator(state_frame: pd.DataFrame) -> pd.DataFrame:
            result = state_frame.copy()
            result["d_state_b_dt"] = 1.0
            result["d_state_a_dt"] = 2.0
            return result

        generated = generate_derivatives(
            frame=frame,
            spec=self.spec,
            variant=EXACT_BENCHMARK_VARIANT,
            exact_generator=exact_generator,
        )

        self.assertEqual(
            tuple(generated.frame.columns),
            (*state_only_columns(self.spec), *derivative_columns(self.spec)),
        )
        self.assertEqual(generated.provenance["derivative_method"], "exact")

    def test_sg_module_has_no_roadrunner_dependency(self) -> None:
        import src.derivative_generation_generic as module

        syntax_tree = ast.parse(inspect.getsource(module))
        imported_roots = set()

        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        self.assertNotIn("roadrunner", imported_roots)
        self.assertNotIn("roadrunner", module.__dict__)


class StateOnlySimulationTests(unittest.TestCase):
    """Prove that state-only simulation does not request exact rates."""

    def test_state_only_trajectory_never_calls_exact_rates(self) -> None:
        spec = build_spec()

        class FakeRoadRunner:
            """Minimal state-only simulator that rejects rate requests."""

            def __init__(self, _model_path: str) -> None:
                self.initial_values = {}

            def resetAll(self) -> None:
                return None

            def reset(self) -> None:
                return None

            def __setitem__(self, key: str, value: float) -> None:
                self.initial_values[key] = value

            def simulate(
                self,
                start: float,
                stop: float,
                points: int,
                *,
                selections: list[str],
            ) -> np.ndarray:
                time_values = np.linspace(start, stop, points)
                return np.column_stack([time_values, time_values**2, time_values**3])

            def getRatesOfChange(self) -> np.ndarray:
                raise AssertionError("Exact rates must not be requested.")

        with patch(
            "src.simulate_generic.roadrunner.RoadRunner",
            FakeRoadRunner,
        ):
            frame = simulate_state_only_trajectory(
                spec=spec,
                initial_values=np.array([1.0, 2.0]),
                duration=10.0,
                points=101,
                trajectory_id="train_000",
            )

        self.assertEqual(tuple(frame.columns), state_only_columns(spec))
        self.assertEqual(len(frame), 101)

    def test_existing_exact_trajectory_schema_is_preserved(self) -> None:
        spec = build_spec()
        instances = []

        class FakeRoadRunner:
            """Minimal simulator returning deterministic synthetic rates."""

            def __init__(self, _model_path: str) -> None:
                self.rate_calls = 0
                self.time = 0.0
                instances.append(self)

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
                time_values = np.linspace(start, stop, points)
                return np.column_stack([time_values, time_values**2, time_values**3])

            def getRatesOfChange(self) -> np.ndarray:
                self.rate_calls += 1
                return np.array([1.0, 2.0])

        with patch(
            "src.simulate_generic.roadrunner.RoadRunner",
            FakeRoadRunner,
        ):
            frame = simulate_trajectory(
                spec=spec,
                initial_values=np.array([1.0, 2.0]),
                duration=10.0,
                points=101,
                trajectory_id="train_000",
            )

        self.assertEqual(
            tuple(frame.columns),
            (*state_only_columns(spec), *derivative_columns(spec)),
        )
        self.assertEqual(instances[0].rate_calls, 101)
        self.assertTrue((frame["d_state_b_dt"] == 1.0).all())
        self.assertTrue((frame["d_state_a_dt"] == 2.0).all())


class SavgolDevelopmentDatasetTests(unittest.TestCase):
    """Validate SG preparation without invoking scientific simulation."""

    def setUp(self) -> None:
        self.spec = build_spec()
        self.design = SimpleNamespace(
            train=np.ones((32, len(STATE_IDS))),
            validation=np.ones((8, len(STATE_IDS))),
            test=np.ones((8, len(STATE_IDS))),
        )

    @staticmethod
    def fake_state_only_set(**kwargs) -> pd.DataFrame:
        """Return synthetic state-only trajectories for a mocked simulator."""
        return build_state_frame(
            trajectory_count=len(kwargs["initial_conditions"]),
            point_count=kwargs["points"],
            trajectory_prefix=kwargs["trajectory_prefix"],
        )

    def test_sg_preparation_is_nested_and_excludes_validation_and_test(
        self,
    ) -> None:
        with (
            patch(
                "src.simulate_generic.simulate_state_only_initial_condition_set",
                side_effect=self.fake_state_only_set,
            ),
            patch(
                "src.simulate_generic.simulate_initial_condition_set",
                side_effect=AssertionError("Exact simulation path called."),
            ),
            patch(
                "src.simulate_generic._calculate_exact_derivatives",
                side_effect=AssertionError("Exact-rate generator called."),
            ),
            patch(
                "src.simulate_generic.generate_derivatives",
                wraps=generate_derivatives,
            ) as derivative_mock,
        ):
            dataset = build_generic_development_dataset(
                spec=self.spec,
                design=self.design,
                duration=10.0,
                points=101,
                variant=SAVGOL_BENCHMARK_VARIANT,
            )

        derivative_mock.assert_called_once()
        derivative_input = derivative_mock.call_args.kwargs["frame"]
        self.assertEqual(derivative_input["trajectory_id"].nunique(), 32)
        self.assertTrue(
            derivative_input["trajectory_id"].str.startswith("train_").all()
        )
        self.assertEqual(
            dataset.train_04["trajectory_id"].nunique(),
            4,
        )
        self.assertEqual(
            dataset.train_08["trajectory_id"].nunique(),
            8,
        )
        self.assertEqual(
            dataset.train_16["trajectory_id"].nunique(),
            16,
        )
        self.assertEqual(
            dataset.train_32["trajectory_id"].nunique(),
            32,
        )
        self.assertEqual(
            tuple(dataset.validation.columns), state_only_columns(self.spec)
        )
        self.assertTrue(dataset.test.empty)
        self.assertEqual(
            dataset.derivative_provenance["derivative_method"],
            "savgol",
        )

        shared_from_four = dataset.train_04.reset_index(drop=True)
        shared_from_thirty_two = dataset.train_32[
            dataset.train_32["trajectory_id"].isin(
                shared_from_four["trajectory_id"].unique()
            )
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            shared_from_four,
            shared_from_thirty_two,
        )

    def test_default_development_variant_remains_exact(self) -> None:
        exact_columns = (
            *state_only_columns(self.spec),
            *derivative_columns(self.spec),
        )

        def fake_exact_set(**kwargs) -> pd.DataFrame:
            frame = build_state_frame(
                trajectory_count=len(kwargs["initial_conditions"]),
                point_count=kwargs["points"],
                trajectory_prefix=kwargs["trajectory_prefix"],
            )
            frame["d_state_b_dt"] = 1.0
            frame["d_state_a_dt"] = 2.0
            return frame[list(exact_columns)]

        with patch(
            "src.simulate_generic.simulate_initial_condition_set",
            side_effect=fake_exact_set,
        ) as exact_mock:
            dataset = build_generic_development_dataset(
                spec=self.spec,
                design=self.design,
                duration=10.0,
                points=101,
            )

        self.assertEqual(exact_mock.call_count, 2)
        self.assertEqual(tuple(dataset.train_32.columns), exact_columns)
        self.assertEqual(tuple(dataset.validation.columns), exact_columns)
        self.assertEqual(
            dataset.derivative_provenance["derivative_method"],
            "exact",
        )


if __name__ == "__main__":
    unittest.main()
