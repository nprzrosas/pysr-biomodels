"""Inspect the SBML model used for the initial PySR-only reconstruction."""

from pathlib import Path

import roadrunner


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters"
    / "BIOMD0000000414"
    / "BIOMD0000000414_url.xml"
)


def main():
    """Load the SBML model and print its basic scientific components."""
    print(f"Loading model: {MODEL_PATH}")

    rr = roadrunner.RoadRunner(str(MODEL_PATH))

    print("\nFloating species:")
    for species_id in rr.model.getFloatingSpeciesIds():
        value = rr[species_id]
        initial_value = rr[f"init({species_id})"]
        print(
            f"  {species_id}: "
            f"initial={initial_value}, current={value}"
        )

    print("\nBoundary species:")
    for species_id in rr.model.getBoundarySpeciesIds():
        value = rr[species_id]
        print(f"  {species_id}: {value}")

    print("\nGlobal parameters:")
    for parameter_id in rr.model.getGlobalParameterIds():
        value = rr[parameter_id]
        print(f"  {parameter_id}: {value}")

    print("\nCompartments:")
    for compartment_id in rr.model.getCompartmentIds():
        value = rr[compartment_id]
        print(f"  {compartment_id}: {value}")

    print("\nReactions:")
    for reaction_id in rr.model.getReactionIds():
        rate = rr[reaction_id]
        print(f"  {reaction_id}: rate={rate}")

    print("\nInitial state derivative:")
    rates = rr.getRatesOfChange()

    for species_id, rate in zip(
        rr.model.getFloatingSpeciesIds(),
        rates,
    ):
        print(f"  d{species_id}/dt = {rate}")


if __name__ == "__main__":
    main()