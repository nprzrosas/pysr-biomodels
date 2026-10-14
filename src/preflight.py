"""Preflight checks for models entering the generic PySR benchmark."""

from dataclasses import dataclass

from .model_compatibility import (
    CompatibilityReport,
    check_model_compatibility,
)
from .model_registry import resolve_model_path
from .model_spec import (
    ModelSpec,
    load_model_spec,
)
from .simulation_metadata import (
    load_simulation_time,
)


@dataclass(frozen=True)
class PreflightResult:
    """Resolved model information required before benchmark execution."""

    model_id: str
    spec: ModelSpec
    compatibility: CompatibilityReport
    simulation_time: float | None

    @property
    def ready(self) -> bool:
        """Return whether the model is ready for the current benchmark."""
        return (
            self.compatibility.compatible
            and self.simulation_time is not None
        )


def run_preflight(
    model_id: str,
) -> PreflightResult:
    """Resolve and validate one model before benchmark execution."""
    model_path = resolve_model_path(
        model_id=model_id,
    )

    spec = load_model_spec(
        model_id=model_id,
        model_path=model_path,
    )

    compatibility = check_model_compatibility(
        spec
    )

    simulation_time = None

    if compatibility.compatible:
        try:
            simulation_time = load_simulation_time(
                model_id=model_id,
            )
        except ValueError:
            simulation_time = None

    return PreflightResult(
        model_id=model_id,
        spec=spec,
        compatibility=compatibility,
        simulation_time=simulation_time,
    )


def print_preflight(
    result: PreflightResult,
) -> None:
    """Print a human-readable preflight report."""
    print()
    print("=" * 70)
    print("GENERIC BENCHMARK PREFLIGHT")
    print("=" * 70)

    print(
        "Model:",
        result.model_id,
    )

    print(
        "SBML:",
        result.spec.model_path,
    )

    print(
        "States:",
        result.spec.state_ids,
    )

    print(
        "Number of states:",
        result.spec.n_states,
    )

    print(
        "Initial values:",
        result.spec.initial_values,
    )

    print(
        "Boundary species:",
        result.spec.boundary_species_ids,
    )

    print(
        "Reactions:",
        len(
            result.spec.reaction_ids
        ),
    )

    print(
        "Compatible:",
        result.compatibility.compatible,
    )

    if result.compatibility.reasons:
        print("Compatibility reasons:")

        for reason in result.compatibility.reasons:
            print(
                " -",
                reason,
            )

    if result.simulation_time is None:
        print(
            "Simulation time:",
            "UNAVAILABLE",
        )
    else:
        print(
            "Simulation time:",
            result.simulation_time,
        )

    print(
        "Ready:",
        result.ready,
    )