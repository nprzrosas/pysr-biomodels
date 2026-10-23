"""Load curated model-aware simulation metadata for BioModels."""

import json
from pathlib import Path

import numpy as np

from .runtime_paths import (
    NATIVE_MODEL_TIME_METADATA_PATH,
    model_time_metadata_path,
)


DEFAULT_SIMULATION_METADATA_PATH = NATIVE_MODEL_TIME_METADATA_PATH


def load_simulation_time(
    model_id: str,
    sbml_filename: str | None = None,
    metadata_path: Path | None = None,
) -> float:
    """Return the curated simulation horizon for one BioModels model.

    Metadata is resolved by model_id. If sbml_filename is supplied, it is
    checked against the filename recorded for that model to prevent accidental
    cross-model assignment when different models share the same basename.
    """
    if metadata_path is None:
        metadata_path = model_time_metadata_path()

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model-time metadata file not found: {metadata_path}"
        )

    matches = []

    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in model-time metadata "
                    f"{metadata_path} at line {line_number}."
                ) from exc

            if row.get("model_id") == model_id:
                matches.append(row)

    if len(matches) == 0:
        raise ValueError(
            f"No curated simulation-time metadata found for {model_id}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple model-time metadata rows found for {model_id}."
        )

    row = matches[0]
    recorded_filename = row.get("sbml_file")

    if sbml_filename is not None and recorded_filename != sbml_filename:
        raise ValueError(
            "SBML filename mismatch for "
            f"{model_id}: metadata records {recorded_filename!r}, "
            f"but resolved model file is {sbml_filename!r}."
        )

    raw_simulation_time = row.get("simulation_time")

    if raw_simulation_time is None or str(raw_simulation_time).strip() == "":
        raise ValueError(
            f"No usable simulation time found for {model_id}."
        )

    try:
        simulation_time = float(raw_simulation_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid curated simulation time for "
            f"{model_id}: {raw_simulation_time!r}"
        ) from exc

    if (
        not np.isfinite(simulation_time)
        or simulation_time <= 0.0
    ):
        raise ValueError(
            "Invalid curated simulation time for "
            f"{model_id}: {simulation_time}"
        )

    return simulation_time
