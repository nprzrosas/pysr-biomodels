"""Generate reproducible initial-condition candidates for PySR inference."""

import numpy as np


def scale_distance(scales):
    """Return the maximum absolute log10 distance from the nominal scale."""
    return float(
        np.max(
            np.abs(
                np.log10(
                    np.maximum(scales, 1e-300)
                )
            )
        )
    )


def candidate_values(nominal_values, zero_initial_value, scales):
    """Convert multiplicative scales into initial concentrations."""
    baseline = np.where(
        nominal_values != 0.0,
        nominal_values,
        zero_initial_value,
    )

    values = baseline * scales

    zero_mask = (
        (nominal_values == 0.0)
        & (zero_initial_value == 0.0)
    )
    values[zero_mask] = 0.0

    return values.astype(float)


def append_unique_candidate(
    candidates,
    seen,
    *,
    scales,
    nominal_values,
    zero_initial_value,
    design,
    split_preference,
):
    """Append a candidate if its scale vector has not been seen before."""
    key = tuple(np.round(scales.astype(float), 12))

    if key in seen:
        return

    seen.add(key)

    candidates.append(
        {
            "values": candidate_values(
                nominal_values,
                zero_initial_value,
                scales,
            ),
            "scales": scales.astype(float),
            "design": design,
            "split_preference": split_preference,
            "log_distance": scale_distance(scales),
        }
    )


def build_log_grid_initial_candidates(
    *,
    nominal_values,
    train_count,
    validation_count,
    zero_initial_value,
    log_scale_factors,
    validation_min_log_distance,
    candidate_pool_multiplier,
    max_initial_candidates,
    seed,
):
    """Build an ordered pool of log-scale initial-condition candidates."""
    dimensions = len(nominal_values)

    rng = np.random.default_rng(seed)

    total_count = train_count + validation_count

    target_pool_size = min(
        max_initial_candidates,
        max(
            total_count,
            total_count * max(1, candidate_pool_multiplier),
        ),
    )

    candidates = []
    seen = set()

    def has_capacity():
        return len(candidates) < target_pool_size

    # Nominal initial condition.
    nominal_scales = np.ones(dimensions, dtype=float)

    append_unique_candidate(
        candidates,
        seen,
        scales=nominal_scales,
        nominal_values=nominal_values,
        zero_initial_value=zero_initial_value,
        design="nominal",
        split_preference="train",
    )

    # Change one state at a time using the predefined log-scale factors.
    for dimension in range(dimensions):
        for factor in log_scale_factors:
            if not has_capacity():
                break

            if factor == 1.0:
                continue

            scales = np.ones(dimensions, dtype=float)
            scales[dimension] = factor

            split_preference = (
                "validation"
                if scale_distance(scales)
                >= validation_min_log_distance
                else "train"
            )

            append_unique_candidate(
                candidates,
                seen,
                scales=scales,
                nominal_values=nominal_values,
                zero_initial_value=zero_initial_value,
                design="one_at_a_time",
                split_preference=split_preference,
            )

    # Pairwise perturbations matter when the model has at least two states.
    extreme_factors = (
        min(log_scale_factors),
        max(log_scale_factors),
    )

    moderate_factors = tuple(
        factor
        for factor in log_scale_factors
        if min(log_scale_factors)
        < factor
        < max(log_scale_factors)
    ) or log_scale_factors

    for left in range(dimensions):
        for right in range(left + 1, dimensions):
            if not has_capacity():
                break

            for factor_left in extreme_factors:
                for factor_right in extreme_factors:
                    if not has_capacity():
                        break

                    scales = np.ones(dimensions, dtype=float)
                    scales[left] = factor_left
                    scales[right] = factor_right

                    append_unique_candidate(
                        candidates,
                        seen,
                        scales=scales,
                        nominal_values=nominal_values,
                        zero_initial_value=zero_initial_value,
                        design="pairwise_extreme",
                        split_preference="validation",
                    )

            for factor_left in moderate_factors:
                for factor_right in moderate_factors:
                    if not has_capacity():
                        break

                    if (
                        factor_left == 1.0
                        and factor_right == 1.0
                    ):
                        continue

                    scales = np.ones(dimensions, dtype=float)
                    scales[left] = factor_left
                    scales[right] = factor_right

                    append_unique_candidate(
                        candidates,
                        seen,
                        scales=scales,
                        nominal_values=nominal_values,
                        zero_initial_value=zero_initial_value,
                        design="pairwise_moderate",
                        split_preference="train",
                    )

    # Fill the remaining pool with reproducible log-uniform perturbations.
    log_factors = np.log10(
        np.asarray(log_scale_factors, dtype=float)
    )

    log_min = float(np.min(log_factors))
    log_max = float(np.max(log_factors))

    random_attempts = 0
    max_random_attempts = max(
        target_pool_size * 20,
        100,
    )

    while (
        len(candidates) < target_pool_size
        and random_attempts < max_random_attempts
    ):
        random_attempts += 1

        scales = np.power(
            10.0,
            rng.uniform(
                log_min,
                log_max,
                size=dimensions,
            ),
        )

        split_preference = (
            "validation"
            if scale_distance(scales)
            >= validation_min_log_distance
            else "train"
        )

        append_unique_candidate(
            candidates,
            seen,
            scales=scales,
            nominal_values=nominal_values,
            zero_initial_value=zero_initial_value,
            design="random_log_grid",
            split_preference=split_preference,
        )

    # Reproduce the historical ordering.
    candidates.sort(
        key=lambda item: (
            0 if item["design"] == "nominal" else 1,
            0 if item["split_preference"] == "train" else 1,
            float(item["log_distance"]),
        )
    )

    return candidates[:max_initial_candidates]


def main():
    nominal_values = np.array([1.0])

    candidates = build_log_grid_initial_candidates(
        nominal_values=nominal_values,
        train_count=4,
        validation_count=4,
        zero_initial_value=1.0,
        log_scale_factors=(0.1, 0.3, 1.0, 3.0, 10.0),
        validation_min_log_distance=0.5,
        candidate_pool_multiplier=3,
        max_initial_candidates=24,
        seed=1010,
    )

    print(f"Generated candidates: {len(candidates)}")
    print()

    for index, candidate in enumerate(candidates):
        print(
            f"{index:02d}  "
            f"VENUS={candidate['values'][0]:.15f}  "
            f"scale={candidate['scales'][0]:.15f}  "
            f"distance={candidate['log_distance']:.15f}  "
            f"design={candidate['design']}  "
            f"preferred={candidate['split_preference']}"
        )


if __name__ == "__main__":
    main()