"""Simulate SBML trajectories and compute exact derivatives for PySR inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner

from .experimental_design import build_experimental_initial_conditions
from .initial_conditions import build_log_grid_initial_candidates
from .simulation_time import get_simulation_time


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "BIOMD0000000414"

MODEL_PATH = (
    PROJECT_ROOT
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters"
    / MODEL_ID
    / f"{MODEL_ID}_url.xml"
)


def simulate_timecourse(
    sbml_path,
    species_ids,
    initial_values,
    start,
    duration,
    points,
):
    """Simulate one SBML trajectory from specified initial concentrations."""
    simulator = roadrunner.RoadRunner(str(sbml_path))

    simulator.model.setFloatingSpeciesInitConcentrations(
        np.asarray(initial_values, dtype=float)
    )
    simulator.resetAll()

    simulator.selections = ["time", *species_ids]

    result = simulator.simulate(
        start,
        start + duration,
        points,
    )

    return pd.DataFrame(
        result,
        columns=["time", *species_ids],
    )


def compute_exact_derivatives(
    sbml_path,
    species_ids,
    timecourse,
):
    """Evaluate the exact SBML rates at every sampled state."""
    simulator = roadrunner.RoadRunner(str(sbml_path))

    derivative_data = {
        "time": timecourse["time"].astype(float).tolist()
    }

    for species_id in species_ids:
        derivative_data[f"d_{species_id}_dt"] = []

    for _, row in timecourse.iterrows():
        simulator.time = float(row["time"])

        simulator.model.setFloatingSpeciesConcentrations(
            [
                float(row[species_id])
                for species_id in species_ids
            ]
        )

        rates = (
            simulator.model
            .getFloatingSpeciesConcentrationRates()
        )

        for species_id, rate in zip(species_ids, rates):
            derivative_data[
                f"d_{species_id}_dt"
            ].append(float(rate))

    return pd.DataFrame(derivative_data)


def trajectory_quality(
    timecourse,
    species_ids,
    nominal_values,
    zero_initial_value,
    allow_negative_trajectories=False,
    negative_tolerance=1e-10,
    max_abs_relative_value=1e6,
    min_dynamic_range_relative=1e-6,
    is_nominal=False,
):
    """Evaluate whether a simulated trajectory is suitable for inference."""
    values = timecourse[species_ids].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        return False, "nonfinite"

    baseline = np.where(
        np.abs(nominal_values) > 0.0,
        np.abs(nominal_values),
        zero_initial_value,
    )

    state_scale = max(
        float(np.max(baseline)),
        1.0,
    )

    minimum_value = float(np.min(values))

    maximum_absolute_value = float(
        np.max(np.abs(values))
    )

    dynamic_ranges = np.ptp(
        values,
        axis=0,
    )

    maximum_dynamic_range = float(
        np.max(dynamic_ranges)
    )

    maximum_absolute_relative_value = (
        maximum_absolute_value / state_scale
    )

    maximum_dynamic_range_relative = (
        maximum_dynamic_range / state_scale
    )

    if (
        not allow_negative_trajectories
        and minimum_value < -abs(negative_tolerance)
    ):
        return False, "negative"

    if (
        maximum_absolute_relative_value
        > max_abs_relative_value
    ):
        return False, "too_large"

    if (
        not is_nominal
        and maximum_dynamic_range_relative
        < min_dynamic_range_relative
    ):
        return False, "insufficient_dynamic_range"

    return True, "accepted"


def select_trajectories(
    sbml_path,
    species_ids,
    nominal_values,
    candidates,
    start,
    duration,
    points,
    train_count,
    validation_count,
    zero_initial_value,
):
    """Simulate, quality-filter, and assign candidate trajectories."""
    accepted = []
    rejected = []

    split_counts = {
        "train": 0,
        "validation": 0,
    }

    for candidate_index, candidate in enumerate(candidates):
        if (
            split_counts["train"] >= train_count
            and split_counts["validation"] >= validation_count
        ):
            break

        frame = simulate_timecourse(
            sbml_path=sbml_path,
            species_ids=species_ids,
            initial_values=candidate["values"],
            start=start,
            duration=duration,
            points=points,
        )

        is_nominal = candidate["design"] == "nominal"

        passed, reason = trajectory_quality(
            timecourse=frame,
            species_ids=species_ids,
            nominal_values=nominal_values,
            zero_initial_value=zero_initial_value,
            is_nominal=is_nominal,
        )

        if not passed:
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "reason": reason,
                    "candidate": candidate,
                }
            )
            continue

        preferred_split = candidate["split_preference"]

        if split_counts[preferred_split] < (
            train_count
            if preferred_split == "train"
            else validation_count
        ):
            split = preferred_split

        elif (
            preferred_split == "validation"
            and split_counts["train"] < train_count
        ):
            split = "train"

        elif (
            preferred_split == "train"
            and split_counts["validation"] < validation_count
        ):
            split = "validation"

        else:
            continue

        trajectory_number = split_counts[split]

        trajectory_id = (
            f"{split}_{trajectory_number:03d}"
        )

        split_counts[split] += 1

        derivatives = compute_exact_derivatives(
            sbml_path=sbml_path,
            species_ids=species_ids,
            timecourse=frame,
        )

        accepted.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "candidate_index": candidate_index,
                "candidate": candidate,
                "timecourse": frame,
                "derivatives": derivatives,
            }
        )

    if split_counts["train"] < train_count:
        raise RuntimeError(
            "Not enough accepted training trajectories."
        )

    if split_counts["validation"] < validation_count:
        raise RuntimeError(
            "Not enough accepted validation trajectories."
        )

    return accepted, rejected


def build_combined_datasets(
    accepted,
    species_ids,
):
    """Combine accepted trajectories into train and validation datasets."""
    combined = {
        "train": [],
        "validation": [],
    }

    derivative_columns = [
        f"d_{species_id}_dt"
        for species_id in species_ids
    ]

    for trajectory in accepted:
        timecourse = trajectory["timecourse"]
        derivatives = trajectory["derivatives"]

        frame = timecourse.copy()

        for column in derivative_columns:
            frame[column] = derivatives[
                column
            ].to_numpy()

        frame.insert(
            1,
            "trajectory_id",
            trajectory["trajectory_id"],
        )

        frame.insert(
            2,
            "split",
            trajectory["split"],
        )

        combined[
            trajectory["split"]
        ].append(frame)

    datasets = {}

    for split in (
        "train",
        "validation",
    ):
        datasets[split] = pd.concat(
            combined[split],
            ignore_index=True,
        )

    return datasets


def build_benchmark_dataset():
    """Build the historical PySR benchmark dataset for the current model."""
    duration = get_simulation_time(MODEL_ID)

    species_ids = ["VENUS"]
    nominal_values = np.array([1.0])

    train_count = 4
    validation_count = 4

    zero_initial_value = 1.0

    candidates = build_log_grid_initial_candidates(
        nominal_values=nominal_values,
        train_count=train_count,
        validation_count=validation_count,
        zero_initial_value=zero_initial_value,
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

    accepted, rejected = select_trajectories(
        sbml_path=MODEL_PATH,
        species_ids=species_ids,
        nominal_values=nominal_values,
        candidates=candidates,
        start=0.0,
        duration=duration,
        points=101,
        train_count=train_count,
        validation_count=validation_count,
        zero_initial_value=zero_initial_value,
    )

    datasets = build_combined_datasets(
        accepted=accepted,
        species_ids=species_ids,
    )

    return datasets, accepted, rejected


def simulate_experimental_split(
    *,
    split,
    initial_values,
    sbml_path,
    species_ids,
    nominal_values,
    start,
    duration,
    points,
    zero_initial_value,
):
    """Simulate one fixed experimental split without reassigning trajectories."""
    accepted = []
    rejected = []

    for trajectory_number, values in enumerate(initial_values):
        trajectory_id = (
            f"{split}_{trajectory_number:03d}"
        )

        frame = simulate_timecourse(
            sbml_path=sbml_path,
            species_ids=species_ids,
            initial_values=values,
            start=start,
            duration=duration,
            points=points,
        )

        passed, reason = trajectory_quality(
            timecourse=frame,
            species_ids=species_ids,
            nominal_values=nominal_values,
            zero_initial_value=zero_initial_value,
            is_nominal=False,
        )

        if not passed:
            rejected.append(
                {
                    "trajectory_id": trajectory_id,
                    "split": split,
                    "initial_values": np.asarray(
                        values,
                        dtype=float,
                    ),
                    "reason": reason,
                }
            )
            continue

        derivatives = compute_exact_derivatives(
            sbml_path=sbml_path,
            species_ids=species_ids,
            timecourse=frame,
        )

        accepted.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "initial_values": np.asarray(
                    values,
                    dtype=float,
                ),
                "timecourse": frame,
                "derivatives": derivatives,
            }
        )

    return accepted, rejected


def combine_experimental_trajectories(
    *,
    trajectories,
    species_ids,
):
    """Combine experimental trajectories into one tabular dataset."""
    derivative_columns = [
        f"d_{species_id}_dt"
        for species_id in species_ids
    ]

    frames = []

    for trajectory in trajectories:
        frame = trajectory["timecourse"].copy()

        for column in derivative_columns:
            frame[column] = trajectory[
                "derivatives"
            ][column].to_numpy()

        frame.insert(
            1,
            "trajectory_id",
            trajectory["trajectory_id"],
        )

        frame.insert(
            2,
            "split",
            trajectory["split"],
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_experimental_dataset():
    """Build the new nested train/validation/test experimental dataset."""
    duration = get_simulation_time(MODEL_ID)

    species_ids = ["VENUS"]
    nominal_values = np.array([1.0])

    zero_initial_value = 1.0

    designs = build_experimental_initial_conditions(
        nominal_values=nominal_values,
        train_count=32,
        validation_count=8,
        test_count=8,
        train_seed=1010,
        validation_seed=2020,
        test_seed=3030,
    )

    accepted_by_split = {}
    rejected_by_split = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        accepted, rejected = simulate_experimental_split(
            split=split,
            initial_values=designs[split]["values"],
            sbml_path=MODEL_PATH,
            species_ids=species_ids,
            nominal_values=nominal_values,
            start=0.0,
            duration=duration,
            points=101,
            zero_initial_value=zero_initial_value,
        )

        accepted_by_split[split] = accepted
        rejected_by_split[split] = rejected

        expected_count = len(
            designs[split]["values"]
        )

        if len(accepted) != expected_count:
            raise RuntimeError(
                f"Experimental split '{split}' expected "
                f"{expected_count} accepted trajectories, "
                f"but only {len(accepted)} passed quality control."
            )

    full_train = accepted_by_split["train"]

    datasets = {
        "train_04": combine_experimental_trajectories(
            trajectories=full_train[:4],
            species_ids=species_ids,
        ),
        "train_08": combine_experimental_trajectories(
            trajectories=full_train[:8],
            species_ids=species_ids,
        ),
        "train_16": combine_experimental_trajectories(
            trajectories=full_train[:16],
            species_ids=species_ids,
        ),
        "train_32": combine_experimental_trajectories(
            trajectories=full_train[:32],
            species_ids=species_ids,
        ),
        "validation": combine_experimental_trajectories(
            trajectories=accepted_by_split["validation"],
            species_ids=species_ids,
        ),
        "test": combine_experimental_trajectories(
            trajectories=accepted_by_split["test"],
            species_ids=species_ids,
        ),
    }

    return (
        datasets,
        designs,
        accepted_by_split,
        rejected_by_split,
    )


def main():
    """Reconstruct and summarize the historical benchmark dataset."""
    datasets, accepted, rejected = build_benchmark_dataset()

    duration = get_simulation_time(MODEL_ID)

    print(f"Model: {MODEL_ID}")
    print(f"Duration: {duration}")
    print(f"Accepted trajectories: {len(accepted)}")
    print(f"Rejected trajectories: {len(rejected)}")
    print()

    print("Accepted trajectories:")

    for trajectory in accepted:
        candidate = trajectory["candidate"]

        print(
            f"{trajectory['trajectory_id']:14s} "
            f"candidate={trajectory['candidate_index']:02d}  "
            f"VENUS={candidate['values'][0]:.15f}  "
            f"design={candidate['design']:15s} "
            f"preferred={candidate['split_preference']}"
        )

    print()
    print("Combined datasets:")
    print(f"Train rows: {len(datasets['train'])}")
    print(f"Validation rows: {len(datasets['validation'])}")
    print()

    print("Train columns:")
    print(datasets["train"].columns.tolist())
    print()

    print("First train rows:")
    print(datasets["train"].head())
    print()

    print("Last train rows:")
    print(datasets["train"].tail())


if __name__ == "__main__":
    main()