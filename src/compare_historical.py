"""Compare reconstructed datasets against the historical benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from .initial_conditions import build_log_grid_initial_candidates
from .simulate import (
    MODEL_ID,
    MODEL_PATH,
    build_combined_datasets,
    select_trajectories,
)
from .simulation_time import get_simulation_time


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = (
    PROJECT_ROOT
    / "LLM-SR"
    / "analysis"
    / "mechanistic_pipeline"
    / MODEL_ID
    / "simple100_targeted_terms_v1"
    / "dynamic_data"
    / f"{MODEL_ID}_log_ic_exact_t120_train4_val4"
)


def reconstruct_datasets():
    """Reconstruct the train and validation datasets independently."""
    duration = get_simulation_time(MODEL_ID)

    species_ids = ["VENUS"]
    nominal_values = np.array([1.0])

    candidates = build_log_grid_initial_candidates(
        nominal_values=nominal_values,
        train_count=4,
        validation_count=4,
        zero_initial_value=1.0,
        log_scale_factors=(
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
        ),
        validation_min_log_distance=0.5,
        candidate_pool_multiplier=3,
        max_initial_candidates=24,
        seed=1010,
    )

    accepted, _ = select_trajectories(
        sbml_path=MODEL_PATH,
        species_ids=species_ids,
        nominal_values=nominal_values,
        candidates=candidates,
        start=0.0,
        duration=duration,
        points=101,
        train_count=4,
        validation_count=4,
        zero_initial_value=1.0,
    )

    return build_combined_datasets(
        accepted=accepted,
        species_ids=species_ids,
    )


def load_historical_split(split):
    """Load historical states and exact derivatives for one split."""
    prefix = "log_ic_exact_t120_train4_val4"

    timecourse_path = (
        HISTORICAL_DIR
        / f"original_timecourse_{prefix}_{split}.csv"
    )

    derivatives_path = (
        HISTORICAL_DIR
        / f"exact_derivatives_{prefix}_{split}.csv"
    )

    timecourse = pd.read_csv(timecourse_path)
    derivatives = pd.read_csv(derivatives_path)

    historical = timecourse.copy()

    historical["d_VENUS_dt"] = (
        derivatives["d_VENUS_dt"].to_numpy()
    )

    return historical


def compare_split(reconstructed, historical, split):
    """Report numerical differences for one data split."""
    if len(reconstructed) != len(historical):
        raise ValueError(
            f"{split}: row count differs: "
            f"{len(reconstructed)} vs {len(historical)}"
        )

    trajectory_match = np.array_equal(
        reconstructed["trajectory_id"].to_numpy(),
        historical["trajectory_id"].to_numpy(),
    )

    split_match = np.array_equal(
        reconstructed["split"].to_numpy(),
        historical["split"].to_numpy(),
    )

    time_error = np.max(
        np.abs(
            reconstructed["time"].to_numpy(dtype=float)
            - historical["time"].to_numpy(dtype=float)
        )
    )

    state_error = np.max(
        np.abs(
            reconstructed["VENUS"].to_numpy(dtype=float)
            - historical["VENUS"].to_numpy(dtype=float)
        )
    )

    derivative_error = np.max(
        np.abs(
            reconstructed["d_VENUS_dt"].to_numpy(dtype=float)
            - historical["d_VENUS_dt"].to_numpy(dtype=float)
        )
    )

    print(f"{split.upper()}")
    print(f"Rows: {len(reconstructed)}")
    print(f"Trajectory IDs match: {trajectory_match}")
    print(f"Split labels match: {split_match}")
    print(f"Max time error:       {time_error:.17g}")
    print(f"Max VENUS error:      {state_error:.17g}")
    print(f"Max derivative error: {derivative_error:.17g}")
    print()


def main():
    """Compare the complete reconstructed benchmark with historical data."""
    reconstructed = reconstruct_datasets()

    for split in (
        "train",
        "validation",
    ):
        historical = load_historical_split(split)

        compare_split(
            reconstructed=reconstructed[split],
            historical=historical,
            split=split,
        )


if __name__ == "__main__":
    main()