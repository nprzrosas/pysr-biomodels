"""Inventory BioModels supported by the generic PySR benchmark."""

from pathlib import Path

import pandas as pd

from .model_registry import DEFAULT_MODEL_ROOT, resolve_model_path
from .preflight import run_preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "generic_model_inventory.csv"
)


def discover_model_ids(
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> list[str]:
    """Discover BioModel identifiers available under the model root."""
    model_ids = []

    for model_dir in sorted(
        Path(model_root).glob(
            "BIOMD*"
        )
    ):
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name

        try:
            resolve_model_path(
                model_id=model_id,
                model_root=model_root,
            )
        except (FileNotFoundError, ValueError):
            continue

        model_ids.append(
            model_id
        )

    return model_ids


def inventory_models() -> pd.DataFrame:
    """Run generic benchmark preflight across all discovered BioModels."""
    model_ids = discover_model_ids()

    rows = []

    print(
        "Discovered models:",
        len(model_ids),
    )

    for index, model_id in enumerate(
        model_ids,
        start=1,
    ):
        try:
            result = run_preflight(
                model_id=model_id,
            )

            row = {
                "model_id": model_id,
                "preflight_success": True,
                "ready": result.ready,
                "compatible": (
                    result.compatibility.compatible
                ),
                "n_states": result.spec.n_states,
                "state_ids": ";".join(
                    result.spec.state_ids
                ),
                "initial_values": ";".join(
                    f"{state_id}="
                    f"{result.spec.initial_values[state_id]:.17g}"
                    for state_id
                    in result.spec.state_ids
                ),
                "n_boundary_species": len(
                    result.spec.boundary_species_ids
                ),
                "boundary_species_ids": ";".join(
                    result.spec.boundary_species_ids
                ),
                "n_reactions": len(
                    result.spec.reaction_ids
                ),
                "simulation_time": (
                    result.simulation_time
                ),
                "rejection_reasons": " | ".join(
                    result.compatibility.reasons
                ),
                "error_type": "",
                "error_message": "",
            }

        except Exception as error:
            row = {
                "model_id": model_id,
                "preflight_success": False,
                "ready": False,
                "compatible": False,
                "n_states": None,
                "state_ids": "",
                "initial_values": "",
                "n_boundary_species": None,
                "boundary_species_ids": "",
                "n_reactions": None,
                "simulation_time": None,
                "rejection_reasons": "",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        rows.append(
            row
        )

        if (
            index % 100 == 0
            or index == len(model_ids)
        ):
            print(
                f"Processed {index}/{len(model_ids)}"
            )

    frame = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return frame


def print_inventory_summary(
    frame: pd.DataFrame,
) -> None:
    """Print a compact summary of the model inventory."""
    print()
    print("=" * 70)
    print("GENERIC MODEL INVENTORY")
    print("=" * 70)

    print(
        "Models:",
        len(frame),
    )

    print(
        "Preflight successful:",
        int(
            frame[
                "preflight_success"
            ].sum()
        ),
    )

    print(
        "Preflight errors:",
        int(
            (
                ~frame[
                    "preflight_success"
                ]
            ).sum()
        ),
    )

    print(
        "Ready:",
        int(
            frame[
                "ready"
            ].sum()
        ),
    )

    ready = frame[
        frame["ready"]
    ]

    print()
    print("Ready models by number of states:")

    if ready.empty:
        print(
            "  None"
        )

    else:
        counts = (
            ready[
                "n_states"
            ]
            .value_counts()
            .sort_index()
        )

        for n_states, count in counts.items():
            print(
                f"  {int(n_states)} states: "
                f"{int(count)}"
            )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


def main() -> None:
    """Build and summarize the generic model inventory."""
    frame = inventory_models()

    print_inventory_summary(
        frame
    )


if __name__ == "__main__":
    main()