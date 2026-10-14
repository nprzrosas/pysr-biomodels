"""Read curated simulation times for BioModels models."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CURATED_TIMES_PATH = (
    PROJECT_ROOT
    / "data2sbml"
    / "output"
    / "time_curated_images.csv"
)


def get_simulation_time(model_id):
    """Return the curated simulation end time for a BioModels model."""
    with CURATED_TIMES_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            sbml_file = row["sbml_file"]

            if sbml_file.startswith(model_id):
                return float(row["simulation_time"])

    raise ValueError(
        f"No curated simulation time found for model {model_id}"
    )


def main():
    """Test curated simulation-time retrieval."""
    model_id = "BIOMD0000000414"
    simulation_time = get_simulation_time(model_id)

    print(f"Model: {model_id}")
    print(f"Curated simulation time: {simulation_time}")


if __name__ == "__main__":
    main()