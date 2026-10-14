"""Select a coupled PySR ODE system using deterministic coordinate search."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark_variant import (
    EXACT_BENCHMARK_VARIANT,
    BenchmarkVariant,
    get_benchmark_variant,
)
from src.evaluate_generic_validation import (
    evaluate_system,
    load_state_candidates,
    load_state_scales,
    resolve_validation_experiment_root,
    summarize_system,
    validate_candidate_pool_provenance,
    validate_selection_development_dataset,
)
from src.experimental_design_generic import build_experimental_design
from src.preflight import run_preflight
from src.simulate_generic import build_generic_development_dataset


TRAINING_DATASETS = (
    "train_04",
    "train_08",
    "train_16",
    "train_32",
)

SELECTION_METHOD = "coordinate_search"


def validation_sort_key(summary):
    """Return the frozen validation ranking key."""
    return (
        float(summary["validation_macro_nrmse"]),
        int(summary["complexity_total"]),
        float(summary["training_loss_total"]),
        int(summary["system_id"]),
    )


def canonical_system_id(
    *,
    spec,
    candidates,
    ranks,
):
    """Return the canonical Cartesian system ID for candidate ranks."""
    if len(ranks) != spec.n_states:
        raise ValueError(
            "Number of candidate ranks does not match number of states."
        )

    candidate_counts = [
        len(candidates[state_id])
        for state_id in spec.state_ids
    ]

    zero_based_system_id = 0

    for state_index, candidate_rank in enumerate(
        ranks
    ):
        candidate_rank = int(
            candidate_rank
        )

        candidate_count = candidate_counts[
            state_index
        ]

        if (
            candidate_rank < 1
            or candidate_rank > candidate_count
        ):
            raise ValueError(
                f"Invalid candidate rank {candidate_rank} "
                f"for state {spec.state_ids[state_index]}."
            )

        trailing_product = int(
            np.prod(
                candidate_counts[
                    state_index + 1:
                ],
                dtype=int,
            )
        )

        zero_based_system_id += (
            candidate_rank - 1
        ) * trailing_product

    return zero_based_system_id + 1


def build_state_rows(
    *,
    spec,
    candidates,
    ranks,
):
    """Build candidate rows from one-based candidate ranks."""
    if len(ranks) != spec.n_states:
        raise ValueError(
            "Number of candidate ranks does not match number of states."
        )

    state_rows = {}

    for state_id, candidate_rank in zip(
        spec.state_ids,
        ranks,
    ):
        frame = candidates[state_id]

        candidate_index = (
            int(candidate_rank)
            - 1
        )

        if (
            candidate_index < 0
            or candidate_index >= len(frame)
        ):
            raise ValueError(
                f"Invalid candidate rank {candidate_rank} "
                f"for state {state_id}."
            )

        state_rows[state_id] = frame.iloc[
            candidate_index
        ]

    return state_rows


def add_provenance(
    *,
    summary,
    model_id,
    dataset_name,
    ranks,
    spec,
):
    """Add stable model and candidate-system provenance."""
    result = dict(summary)

    result["model_id"] = model_id
    result["dataset_name"] = dataset_name
    result["selection_method"] = SELECTION_METHOD

    result["candidate_rank_signature"] = ",".join(
        str(int(rank))
        for rank in ranks
    )

    for state_id, candidate_rank in zip(
        spec.state_ids,
        ranks,
    ):
        result[
            f"candidate_rank_{state_id}"
        ] = int(candidate_rank)

    return result


def run_coordinate_selection(
    *,
    spec,
    dataset_name,
    candidates,
    validation_frame,
    state_scales,
):
    """Run deterministic single-start coordinate search."""
    expected_trajectories = (
        validation_frame[
            "trajectory_id"
        ].nunique()
    )

    if expected_trajectories <= 0:
        raise ValueError(
            "Validation dataset contains no trajectories."
        )

    cache = {}
    evaluation_order = []


    def evaluate_ranks(ranks):
        """Evaluate a candidate-rank tuple once and cache it."""
        ranks = tuple(
            int(rank)
            for rank in ranks
        )

        if ranks in cache:
            return cache[ranks]

        state_rows = build_state_rows(
            spec=spec,
            candidates=candidates,
            ranks=ranks,
        )

        system_id = canonical_system_id(
            spec=spec,
            candidates=candidates,
            ranks=ranks,
        )

        evaluation = evaluate_system(
            spec=spec,
            system_id=system_id,
            state_rows=state_rows,
            validation_frame=validation_frame,
            state_scales=state_scales,
        )

        summary = summarize_system(
            spec=spec,
            system_id=system_id,
            state_rows=state_rows,
            evaluation=evaluation,
            expected_trajectories=expected_trajectories,
        )

        summary = add_provenance(
            summary=summary,
            model_id=spec.model_id,
            dataset_name=dataset_name,
            ranks=ranks,
            spec=spec,
        )

        result = {
            "ranks": ranks,
            "state_rows": state_rows,
            "evaluation": evaluation,
            "summary": summary,
        }

        cache[ranks] = result

        evaluation_order.append(
            ranks
        )

        return result

    current_ranks = tuple(
        1
        for _ in spec.state_ids
    )

    evaluate_ranks(
        current_ranks
    )

    rounds = 0

    while True:
        rounds += 1

        changed = False

        for state_index, state_id in enumerate(
            spec.state_ids
        ):
            proposals = []

            number_of_candidates = len(
                candidates[state_id]
            )

            for candidate_rank in range(
                1,
                number_of_candidates + 1,
            ):
                proposal = list(
                    current_ranks
                )

                proposal[
                    state_index
                ] = candidate_rank

                proposal = tuple(
                    proposal
                )

                proposals.append(
                    evaluate_ranks(
                        proposal
                    )
                )

            best_result = min(
                proposals,
                key=lambda result: validation_sort_key(
                    result["summary"]
                ),
            )

            best_ranks = best_result[
                "ranks"
            ]

            if best_ranks != current_ranks:
                current_ranks = best_ranks
                changed = True

        if not changed:
            break

    return {
        "selected": cache[current_ranks],
        "cache": cache,
        "evaluation_order": evaluation_order,
        "rounds": rounds,
    }


def write_selection_results(
    *,
    output_dir,
    spec,
    dataset_name,
    selection,
):
    """Write coordinate-search selection artifacts."""
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    metric_rows = []
    prediction_rows = []
    trace_rows = []

    for evaluation_index, ranks in enumerate(
        selection["evaluation_order"],
        start=1,
    ):
        result = selection[
            "cache"
        ][
            ranks
        ]

        summary = dict(
            result["summary"]
        )

        summary[
            "evaluation_index"
        ] = evaluation_index

        summary_rows.append(
            summary
        )

        for metric_row in result[
            "evaluation"
        ][
            "metric_rows"
        ]:
            row = dict(
                metric_row
            )

            row["model_id"] = spec.model_id
            row["dataset_name"] = dataset_name
            row["selection_method"] = SELECTION_METHOD

            metric_rows.append(
                row
            )

        for prediction_row in result[
            "evaluation"
        ][
            "prediction_rows"
        ]:
            row = dict(
                prediction_row
            )

            row["model_id"] = spec.model_id
            row["dataset_name"] = dataset_name
            row["selection_method"] = SELECTION_METHOD

            prediction_rows.append(
                row
            )

        trace_row = {
            "model_id": spec.model_id,
            "dataset_name": dataset_name,
            "selection_method": SELECTION_METHOD,
            "evaluation_index": evaluation_index,
            "system_id": result[
                "summary"
            ][
                "system_id"
            ],
        }

        for state_id, candidate_rank in zip(
            spec.state_ids,
            ranks,
        ):
            trace_row[
                f"candidate_rank_{state_id}"
            ] = int(candidate_rank)

        trace_rows.append(
            trace_row
        )

    ranking = pd.DataFrame(
        summary_rows
    )

    ranking = (
        ranking
        .sort_values(
            [
                "validation_macro_nrmse",
                "complexity_total",
                "training_loss_total",
                "system_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "coordinate_validation_rank"
    ] = (
        np.arange(
            len(ranking)
        )
        + 1
    )

    selected_summary = dict(
        selection[
            "selected"
        ][
            "summary"
        ]
    )

    selected_summary[
        "coordinate_rounds"
    ] = selection[
        "rounds"
    ]

    selected_summary[
        "systems_evaluated"
    ] = len(
        selection[
            "cache"
        ]
    )

    ranking.to_csv(
        output_dir
        / "coordinate_validation_ranking.csv",
        index=False,
    )

    pd.DataFrame(
        [
            selected_summary
        ]
    ).to_csv(
        output_dir
        / "coordinate_selected_system.csv",
        index=False,
    )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        output_dir
        / "coordinate_validation_state_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        output_dir
        / "coordinate_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        trace_rows
    ).to_csv(
        output_dir
        / "coordinate_search_trace.csv",
        index=False,
    )


def parse_benchmark_variant(value: str) -> BenchmarkVariant:
    """Resolve one CLI derivative-method value to a benchmark variant.

    Args:
        value: Derivative-method CLI value.

    Returns:
        Registered immutable benchmark variant.

    Raises:
        argparse.ArgumentTypeError: If the derivative method is unknown.
    """
    try:
        return get_benchmark_variant(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        arguments: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        Parsed model identifier and validated benchmark variant.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select coupled PySR equations using deterministic "
            "coordinate search on validation trajectories."
        )
    )

    parser.add_argument(
        "model_id",
        help="BioModels identifier.",
    )

    parser.add_argument(
        "--derivative-method",
        default=EXACT_BENCHMARK_VARIANT,
        type=parse_benchmark_variant,
        metavar="{exact,savgol}",
        help=(
            "Derivative-target method. Defaults to exact; "
            "savgol selects the frozen Savitzky-Golay v1 variant."
        ),
    )

    return parser.parse_args(arguments)


def run_selection(
    *,
    model_id: str,
    variant: BenchmarkVariant = EXACT_BENCHMARK_VARIANT,
) -> None:
    """Run coordinate-search validation selection for one benchmark variant.

    Args:
        model_id: BioModels identifier to select.
        variant: Candidate derivative method and protected result namespace.
    """
    preflight = run_preflight(
        model_id
    )

    if not preflight.compatibility.compatible:
        raise RuntimeError(
            f"Model {model_id} is not compatible: "
            f"{preflight.compatibility.reasons}"
        )

    spec = preflight.spec

    design = build_experimental_design(
        spec=spec,
    )

    model_results_root = resolve_validation_experiment_root(
        model_id=spec.model_id,
        variant=variant,
    )

    validate_candidate_pool_provenance(
        experiment_root=model_results_root,
        variant=variant,
        dataset_names=TRAINING_DATASETS,
    )

    dataset = build_generic_development_dataset(
        spec=spec,
        design=design,
        duration=preflight.simulation_time,
        variant=variant,
    )

    validate_selection_development_dataset(
        dataset=dataset,
        spec=spec,
        variant=variant,
    )

    print("=" * 100)
    print("GENERIC COORDINATE SYSTEM SELECTION")
    print("=" * 100)
    print(
        "BioModels model ID:",
        spec.model_id,
    )
    print(
        "States:",
        spec.state_ids,
    )
    print(
        "Validation trajectories:",
        dataset.validation[
            "trajectory_id"
        ].nunique(),
    )
    print(
        "Held-out test rows:",
        len(
            dataset.test
        ),
    )

    for dataset_name in TRAINING_DATASETS:
        experiment_dir = (
            model_results_root
            / dataset_name
        )

        candidates = load_state_candidates(
            experiment_dir=experiment_dir,
            spec=spec,
        )

        state_scales = load_state_scales(
            experiment_dir=experiment_dir,
            spec=spec,
        )

        selection = run_coordinate_selection(
            spec=spec,
            dataset_name=dataset_name,
            candidates=candidates,
            validation_frame=dataset.validation,
            state_scales=state_scales,
        )

        output_dir = (
            experiment_dir
            / "coordinate_selection"
        )

        write_selection_results(
            output_dir=output_dir,
            spec=spec,
            dataset_name=dataset_name,
            selection=selection,
        )

        selected = selection[
            "selected"
        ]

        summary = selected[
            "summary"
        ]

        print()
        print("=" * 100)
        print(
            dataset_name
        )
        print("=" * 100)

        print(
            "BioModels model ID:",
            spec.model_id,
        )
        print(
            "Selected ranks:",
            selected[
                "ranks"
            ],
        )
        print(
            "Canonical system ID:",
            summary[
                "system_id"
            ],
        )
        print(
            "Validation macro NRMSE:",
            summary[
                "validation_macro_nrmse"
            ],
        )
        print(
            "Coordinate rounds:",
            selection[
                "rounds"
            ],
        )
        print(
            "Unique systems evaluated:",
            len(
                selection[
                    "cache"
                ]
            ),
        )
        print(
            "Validation status:",
            summary[
                "validation_status"
            ],
        )

        for state_id in spec.state_ids:
            print(
                f"{state_id}:",
                summary[
                    f"equation_{state_id}"
                ],
            )


def main() -> None:
    """Run coordinate-search validation selection for one BioModels model."""
    arguments = parse_arguments()

    run_selection(
        model_id=arguments.model_id,
        variant=arguments.derivative_method,
    )


if __name__ == "__main__":
    main()
