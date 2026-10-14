"""Generic model specification utilities for PySR ODE inference."""

from dataclasses import dataclass
from pathlib import Path

import roadrunner


@dataclass(frozen=True)
class ModelSpec:
    """Scientific metadata required to infer one dynamical model."""

    model_id: str
    model_path: Path
    state_ids: tuple[str, ...]
    initial_values: dict[str, float]
    boundary_species_ids: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    compartment_ids: tuple[str, ...]
    reaction_ids: tuple[str, ...]

    @property
    def n_states(self):
        """Return the number of inferred dynamical states."""
        return len(self.state_ids)


def load_model_spec(
    model_id,
    model_path,
):
    """Load an SBML model and identify its dynamical floating species."""
    model_path = Path(
        model_path
    ).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"SBML model not found: {model_path}"
        )

    rr = roadrunner.RoadRunner(
        str(model_path)
    )

    state_ids = tuple(
        rr.model.getFloatingSpeciesIds()
    )

    if not state_ids:
        raise ValueError(
            f"Model {model_id} has no floating species."
        )

    initial_values = {
        species_id: float(
            rr[f"init({species_id})"]
        )
        for species_id in state_ids
    }

    boundary_species_ids = tuple(
        rr.model.getBoundarySpeciesIds()
    )

    parameter_ids = tuple(
        rr.model.getGlobalParameterIds()
    )

    compartment_ids = tuple(
        rr.model.getCompartmentIds()
    )

    reaction_ids = tuple(
        rr.model.getReactionIds()
    )

    return ModelSpec(
        model_id=model_id,
        model_path=model_path,
        state_ids=state_ids,
        initial_values=initial_values,
        boundary_species_ids=boundary_species_ids,
        parameter_ids=parameter_ids,
        compartment_ids=compartment_ids,
        reaction_ids=reaction_ids,
    )


def print_model_spec(
    spec,
):
    """Print a human-readable summary of a model specification."""
    print()
    print("=" * 70)
    print("MODEL SPECIFICATION")
    print("=" * 70)

    print(
        "Model ID:",
        spec.model_id,
    )

    print(
        "SBML:",
        spec.model_path,
    )

    print(
        "Number of states:",
        spec.n_states,
    )

    print()

    print("Dynamic states:")

    for state_id in spec.state_ids:
        print(
            f"  {state_id}: "
            f"initial={spec.initial_values[state_id]}"
        )

    print()

    print(
        "Boundary species:",
        spec.boundary_species_ids,
    )

    print(
        "Global parameters:",
        spec.parameter_ids,
    )

    print(
        "Compartments:",
        spec.compartment_ids,
    )

    print(
        "Reactions:",
        spec.reaction_ids,
    )