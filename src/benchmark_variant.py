"""Validated benchmark variants for generic derivative generation."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

EXACT_EXPERIMENT_NAMESPACE = "generic_exact_experiment"
SAVGOL_EXPERIMENT_NAMESPACE = "generic_savgol_experiment"

SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 3
SAVGOL_DERIVATIVE_ORDER = 1
SAVGOL_MODE = "interp"
SAVGOL_DELTA_POLICY = "uniform_per_trajectory_time_step"


class DerivativeMethod(str, Enum):
    """Supported generic derivative-target generation methods."""

    EXACT = "exact"
    SAVGOL = "savgol"


@dataclass(frozen=True)
class ExactDerivativeConfig:
    """Provenance for exact derivative targets."""

    rate_source: str = "RoadRunner.getRatesOfChange"


@dataclass(frozen=True)
class SavgolDerivativeConfig:
    """Frozen Savitzky-Golay benchmark-v1 configuration."""

    window_length: int = SAVGOL_WINDOW_LENGTH
    polyorder: int = SAVGOL_POLYORDER
    deriv: int = SAVGOL_DERIVATIVE_ORDER
    mode: str = SAVGOL_MODE
    delta_policy: str = SAVGOL_DELTA_POLICY

    def __post_init__(self) -> None:
        """Reject changes to the frozen Savitzky-Golay v1 protocol."""
        observed = (
            self.window_length,
            self.polyorder,
            self.deriv,
            self.mode,
            self.delta_policy,
        )
        expected = (
            SAVGOL_WINDOW_LENGTH,
            SAVGOL_POLYORDER,
            SAVGOL_DERIVATIVE_ORDER,
            SAVGOL_MODE,
            SAVGOL_DELTA_POLICY,
        )

        if observed != expected:
            raise ValueError(
                "Savitzky-Golay benchmark v1 is frozen at "
                "window_length=11, polyorder=3, deriv=1, "
                "mode='interp', and a uniform per-trajectory delta."
            )


DerivativeConfig = ExactDerivativeConfig | SavgolDerivativeConfig


@dataclass(frozen=True)
class BenchmarkVariant:
    """Associate a derivative method with its protected result namespace."""

    derivative_method: DerivativeMethod
    experiment_namespace: str
    derivative_config: DerivativeConfig

    def __post_init__(self) -> None:
        """Validate method, namespace, and configuration as one unit."""
        if not isinstance(self.derivative_method, DerivativeMethod):
            raise TypeError("derivative_method must be a DerivativeMethod.")

        expected_namespace = {
            DerivativeMethod.EXACT: EXACT_EXPERIMENT_NAMESPACE,
            DerivativeMethod.SAVGOL: SAVGOL_EXPERIMENT_NAMESPACE,
        }[self.derivative_method]

        if self.experiment_namespace != expected_namespace:
            raise ValueError(
                f"Derivative method '{self.derivative_method.value}' must use "
                f"namespace '{expected_namespace}', not "
                f"'{self.experiment_namespace}'."
            )

        expected_config_type = {
            DerivativeMethod.EXACT: ExactDerivativeConfig,
            DerivativeMethod.SAVGOL: SavgolDerivativeConfig,
        }[self.derivative_method]

        if not isinstance(self.derivative_config, expected_config_type):
            raise TypeError(
                f"Derivative method '{self.derivative_method.value}' requires "
                f"{expected_config_type.__name__}."
            )

    def derivative_provenance(
        self,
        *,
        input_columns: tuple[str, ...],
        state_order: tuple[str, ...],
    ) -> dict[str, object]:
        """Build serializable derivative-generation provenance.

        Args:
            input_columns: Ordered columns exposed to target generation.
            state_order: Ordered dynamic-state identifiers.

        Returns:
            A dictionary suitable for inclusion in run metadata.
        """
        provenance: dict[str, object] = {
            "derivative_method": self.derivative_method.value,
            "experiment_namespace": self.experiment_namespace,
            "derivative_input_columns": ";".join(input_columns),
            "derivative_state_order": ";".join(state_order),
        }

        if isinstance(self.derivative_config, ExactDerivativeConfig):
            provenance["derivative_rate_source"] = self.derivative_config.rate_source
        else:
            provenance.update(
                {
                    "savgol_window_length": (self.derivative_config.window_length),
                    "savgol_polyorder": self.derivative_config.polyorder,
                    "savgol_deriv": self.derivative_config.deriv,
                    "savgol_mode": self.derivative_config.mode,
                    "savgol_delta_policy": (self.derivative_config.delta_policy),
                    "savgol_parameter_origin": (
                        "prespecified_historical_transfer_BIOMD0000000414"
                    ),
                    "savgol_parameter_claim": "not_optimized_across_models",
                }
            )

        return provenance


def validate_derivative_provenance(
    *,
    provenance: Mapping[str, object] | None,
    variant: BenchmarkVariant,
) -> None:
    """Validate dataset lineage against one benchmark variant.

    Args:
        provenance: Derivative-generation provenance stored with a dataset.
        variant: Authoritative derivative method and namespace selection.

    Raises:
        TypeError: If the variant or provenance has an invalid container type.
        ValueError: If identity fields are absent or conflict with the variant.
    """
    if not isinstance(variant, BenchmarkVariant):
        raise TypeError("variant must be a validated BenchmarkVariant.")

    if provenance is None or not isinstance(provenance, Mapping):
        raise TypeError("Derivative provenance must be a mapping.")

    missing_marker = "<missing>"
    observed_method_value = provenance.get(
        "derivative_method",
        missing_marker,
    )
    observed_namespace = provenance.get(
        "experiment_namespace",
        missing_marker,
    )

    if isinstance(observed_method_value, DerivativeMethod):
        observed_method = observed_method_value.value
    else:
        observed_method = observed_method_value

    expected_method = variant.derivative_method.value
    expected_namespace = variant.experiment_namespace

    if (
        observed_method != expected_method
        or observed_namespace != expected_namespace
    ):
        raise ValueError(
            "Derivative provenance does not match the benchmark variant: "
            f"expected derivative_method='{expected_method}', "
            f"observed derivative_method='{observed_method}'; "
            f"expected experiment_namespace='{expected_namespace}', "
            f"observed experiment_namespace='{observed_namespace}'."
        )


EXACT_BENCHMARK_VARIANT = BenchmarkVariant(
    derivative_method=DerivativeMethod.EXACT,
    experiment_namespace=EXACT_EXPERIMENT_NAMESPACE,
    derivative_config=ExactDerivativeConfig(),
)

SAVGOL_BENCHMARK_VARIANT = BenchmarkVariant(
    derivative_method=DerivativeMethod.SAVGOL,
    experiment_namespace=SAVGOL_EXPERIMENT_NAMESPACE,
    derivative_config=SavgolDerivativeConfig(),
)


def get_benchmark_variant(
    derivative_method: str | DerivativeMethod,
) -> BenchmarkVariant:
    """Return the frozen variant for a derivative method.

    Args:
        derivative_method: Exact or Savitzky-Golay method identifier.

    Returns:
        The validated immutable benchmark variant.
    """
    try:
        method = DerivativeMethod(derivative_method)
    except ValueError as error:
        valid_methods = ", ".join(method.value for method in DerivativeMethod)
        raise ValueError(
            f"Unknown derivative method '{derivative_method}'. "
            f"Expected one of: {valid_methods}."
        ) from error

    return {
        DerivativeMethod.EXACT: EXACT_BENCHMARK_VARIANT,
        DerivativeMethod.SAVGOL: SAVGOL_BENCHMARK_VARIANT,
    }[method]
