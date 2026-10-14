"""Experimental initial-condition designs for PySR inference."""

import numpy as np
from scipy.stats import qmc


DEFAULT_LOG10_MIN = -1.0
DEFAULT_LOG10_MAX = 1.0
DEFAULT_CANDIDATE_POOL_SIZE = 256


def latin_hypercube_log_scales(
    *,
    sample_count,
    dimensions,
    seed,
    log10_min=DEFAULT_LOG10_MIN,
    log10_max=DEFAULT_LOG10_MAX,
):
    """Generate Latin-hypercube scale factors uniformly in log10 space."""
    sampler = qmc.LatinHypercube(
        d=dimensions,
        seed=seed,
    )

    unit_samples = sampler.random(n=sample_count)

    log_samples = qmc.scale(
        unit_samples,
        l_bounds=np.full(dimensions, log10_min),
        u_bounds=np.full(dimensions, log10_max),
    )

    scales = np.power(10.0, log_samples)

    return scales


def order_scales_for_nested_coverage(scales):
    """Order LHS points for progressively space-filling nested subsets."""
    scales = np.asarray(
        scales,
        dtype=float,
    )

    log_scales = np.log10(scales)

    dimensions = log_scales.shape[1]

    nominal_log_scale = np.zeros(
        dimensions,
        dtype=float,
    )

    distance_to_nominal = np.linalg.norm(
        log_scales - nominal_log_scale,
        axis=1,
    )

    first_index = int(
        np.argmin(distance_to_nominal)
    )

    selected = [first_index]

    remaining = set(range(len(scales)))
    remaining.remove(first_index)

    while remaining:
        best_index = None
        best_distance = -np.inf

        selected_points = log_scales[selected]

        for candidate_index in sorted(remaining):
            candidate = log_scales[candidate_index]

            distances = np.linalg.norm(
                selected_points - candidate,
                axis=1,
            )

            minimum_distance = float(
                np.min(distances)
            )

            if minimum_distance > best_distance:
                best_distance = minimum_distance
                best_index = candidate_index

        selected.append(best_index)
        remaining.remove(best_index)

    return scales[selected]


def select_maximin_scales(
    *,
    candidate_scales,
    reference_scales,
    sample_count,
):
    """Select points maximizing separation from reference and selected points."""
    candidate_scales = np.asarray(
        candidate_scales,
        dtype=float,
    )

    reference_scales = np.asarray(
        reference_scales,
        dtype=float,
    )

    candidate_log_scales = np.log10(
        candidate_scales
    )

    reference_log_scales = np.log10(
        reference_scales
    )

    selected_indices = []
    remaining_indices = set(
        range(len(candidate_scales))
    )

    while len(selected_indices) < sample_count:
        best_index = None
        best_distance = -np.inf

        if selected_indices:
            selected_log_scales = candidate_log_scales[
                selected_indices
            ]

            comparison_points = np.vstack(
                [
                    reference_log_scales,
                    selected_log_scales,
                ]
            )
        else:
            comparison_points = reference_log_scales

        for candidate_index in sorted(remaining_indices):
            candidate = candidate_log_scales[
                candidate_index
            ]

            distances = np.linalg.norm(
                comparison_points - candidate,
                axis=1,
            )

            minimum_distance = float(
                np.min(distances)
            )

            if minimum_distance > best_distance:
                best_distance = minimum_distance
                best_index = candidate_index

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return candidate_scales[selected_indices]


def scales_to_initial_conditions(
    *,
    nominal_values,
    scales,
    zero_initial_value=1.0,
):
    """Convert multiplicative scale factors to state initial conditions."""
    nominal_values = np.asarray(
        nominal_values,
        dtype=float,
    )

    baseline = np.where(
        nominal_values != 0.0,
        nominal_values,
        zero_initial_value,
    )

    return scales * baseline


def build_experimental_initial_conditions(
    *,
    nominal_values,
    train_count=32,
    validation_count=8,
    test_count=8,
    train_seed=1010,
    validation_seed=2020,
    test_seed=3030,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """Build nested train and separated validation/test IC designs."""
    nominal_values = np.asarray(
        nominal_values,
        dtype=float,
    )

    dimensions = len(nominal_values)

    train_scales = latin_hypercube_log_scales(
        sample_count=train_count,
        dimensions=dimensions,
        seed=train_seed,
    )

    train_scales = order_scales_for_nested_coverage(
        train_scales
    )

    validation_candidates = latin_hypercube_log_scales(
        sample_count=candidate_pool_size,
        dimensions=dimensions,
        seed=validation_seed,
    )

    validation_scales = select_maximin_scales(
        candidate_scales=validation_candidates,
        reference_scales=train_scales,
        sample_count=validation_count,
    )

    test_candidates = latin_hypercube_log_scales(
        sample_count=candidate_pool_size,
        dimensions=dimensions,
        seed=test_seed,
    )

    test_reference_scales = np.vstack(
        [
            train_scales,
            validation_scales,
        ]
    )

    test_scales = select_maximin_scales(
        candidate_scales=test_candidates,
        reference_scales=test_reference_scales,
        sample_count=test_count,
    )

    split_scales = {
        "train": train_scales,
        "validation": validation_scales,
        "test": test_scales,
    }

    split_seeds = {
        "train": train_seed,
        "validation": validation_seed,
        "test": test_seed,
    }

    designs = {}

    for split, scales in split_scales.items():
        values = scales_to_initial_conditions(
            nominal_values=nominal_values,
            scales=scales,
        )

        designs[split] = {
            "scales": scales,
            "values": values,
            "seed": split_seeds[split],
        }

    return designs


def minimum_cross_split_distance(
    left_scales,
    right_scales,
):
    """Return the minimum pairwise distance between splits in log10 space."""
    left_log_scales = np.log10(
        np.asarray(left_scales, dtype=float)
    )

    right_log_scales = np.log10(
        np.asarray(right_scales, dtype=float)
    )

    distances = np.linalg.norm(
        left_log_scales[:, None, :]
        - right_log_scales[None, :, :],
        axis=2,
    )

    left_index, right_index = np.unravel_index(
        np.argmin(distances),
        distances.shape,
    )

    return (
        float(distances[left_index, right_index]),
        left_index,
        right_index,
    )


def main():
    """Display and diagnose the experimental IC design."""
    nominal_values = np.array([1.0])

    designs = build_experimental_initial_conditions(
        nominal_values=nominal_values,
    )

    for split, design in designs.items():
        print()
        print(split.upper())
        print(f"seed: {design['seed']}")
        print(f"count: {len(design['values'])}")
        print()

        for index, (scales, values) in enumerate(
            zip(
                design["scales"],
                design["values"],
            )
        ):
            print(
                f"{index:02d}  "
                f"scale={scales[0]:.8f}  "
                f"VENUS={values[0]:.8f}"
            )

    train_values = designs["train"]["values"]

    print()
    print("NESTED TRAINING SETS")
    print()

    for train_count in (4, 8, 16, 32):
        subset = train_values[:train_count]

        print(
            f"train_{train_count:02d}: "
            f"{len(subset)} trajectories"
        )

        print(
            "  VENUS range: "
            f"{subset[:, 0].min():.8f} "
            f"to "
            f"{subset[:, 0].max():.8f}"
        )

    print()
    print("CROSS-SPLIT LOG10 DISTANCES")
    print()

    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        distance, left_index, right_index = (
            minimum_cross_split_distance(
                designs[left]["scales"],
                designs[right]["scales"],
            )
        )

        left_scale = designs[left]["scales"][
            left_index
        ]

        right_scale = designs[right]["scales"][
            right_index
        ]

        print(
            f"{left:10s} vs {right:10s}: "
            f"min distance = {distance:.6f}, "
            f"scales = {left_scale[0]:.6f} "
            f"vs {right_scale[0]:.6f}"
        )


if __name__ == "__main__":
    main()