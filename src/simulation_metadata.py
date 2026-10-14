"""Load curated simulation metadata for BioModels."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SIMULATION_METADATA_PATH = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "time_curated_images.csv"
)


def load_simulation_time(
    model_id: str,
    metadata_path: Path | None = None,
) -> float:
    """Return the curated simulation horizon for one model."""
    if metadata_path is None:
        metadata_path = DEFAULT_SIMULATION_METADATA_PATH

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Simulation metadata file not found: {metadata_path}"
        )

    frame = pd.read_csv(metadata_path)

    required_columns = {
        "sbml_file",
        "simulation_time",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Simulation metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    expected_filename = (
        f"{model_id}_url.xml"
    )

    matches = frame[
        frame["sbml_file"]
        == expected_filename
    ]

    if len(matches) == 0:
        raise ValueError(
            "No curated simulation time found for "
            f"{model_id} ({expected_filename})."
        )

    if len(matches) > 1:
        raise ValueError(
            "Multiple simulation metadata rows found for "
            f"{model_id} ({expected_filename})."
        )

    simulation_time = float(
        matches.iloc[0][
            "simulation_time"
        ]
    )

    if (
        not np.isfinite(simulation_time)
        or simulation_time <= 0.0
    ):
        raise ValueError(
            "Invalid curated simulation time for "
            f"{model_id}: {simulation_time}"
        )

    return simulation_time