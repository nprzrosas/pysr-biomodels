"""Audit symbolic mechanistic ground-truth extraction across frozen models.

Each model is independently extracted and numerically validated against
RoadRunner. Failures are recorded rather than stopping the complete audit.

This audit is strictly post-hoc and must not influence PySR inference,
Selector v1, or held-out test selection.
"""

from pathlib import Path
import traceback

import pandas as pd

from src.validate_mechanistic_ground_truth import (
    validate_mechanistic_ground_truth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "mechanistic_ground_truth_audit.csv"
)

FROZEN_MODEL_IDS = (
    "BIOMD0000000079",
    "BIOMD0000000156",
    "BIOMD0000000174",
    "BIOMD0000000322",
    "BIOMD0000000039",
    "BIOMD0000000116",
    "BIOMD0000000229",
    "BIOMD0000000010",
    "BIOMD0000000022",
    "BIOMD0000000476",
)


def audit_one_model(
    model_id,
):
    """Extract and validate mechanistic ground truth for one model."""
    print()
    print("=" * 80)
    print("MODEL")
    print(model_id)

    try:
        (
            ground_truth,
            validation,
        ) = validate_mechanistic_ground_truth(
            model_id=model_id,
        )

        print(
            "States:",
            len(
                ground_truth.state_ids
            ),
        )

        print(
            "Reactions:",
            len(
                ground_truth.reactions
            ),
        )

        print(
            "Autonomous:",
            ground_truth.is_autonomous,
        )

        print(
            "Explicit independent variables:",
            ground_truth.explicit_independent_variables,
        )

        print(
            "Extraction:",
            validation.extraction_status,
        )

        print(
            "Numerical validation:",
            validation.numerical_validation_status,
        )

        print(
            "Points checked:",
            validation.n_points_checked,
        )

        print(
            "Maximum absolute error:",
            f"{validation.maximum_absolute_error:.6e}",
        )

        print(
            "Maximum relative error:",
            f"{validation.maximum_relative_error:.6e}",
        )

        return {
            "model_id": model_id,
            "n_states": len(
                ground_truth.state_ids
            ),
            "n_reactions": len(
                ground_truth.reactions
            ),
            "ground_truth_is_autonomous": (
                ground_truth.is_autonomous
            ),
            "ground_truth_independent_variables": (
                ",".join(
                    ground_truth.explicit_independent_variables
                )
            ),
            "extraction_status": (
                validation.extraction_status
            ),
            "numerical_validation_status": (
                validation.numerical_validation_status
            ),
            "n_points_checked": (
                validation.n_points_checked
            ),
            "maximum_absolute_error": (
                validation.maximum_absolute_error
            ),
            "maximum_relative_error": (
                validation.maximum_relative_error
            ),
            "error_type": "",
            "error_message": "",
        }

    except Exception as error:
        print(
            "FAILED"
        )

        print(
            "Error type:",
            type(
                error
            ).__name__,
        )

        print(
            "Error message:",
            str(
                error
            ),
        )

        traceback.print_exc()

        return {
            "model_id": model_id,
            "n_states": None,
            "n_reactions": None,
            "ground_truth_is_autonomous": None,
            "ground_truth_independent_variables": "",
            "extraction_status": "failed",
            "numerical_validation_status": (
                "not_run"
            ),
            "n_points_checked": 0,
            "maximum_absolute_error": None,
            "maximum_relative_error": None,
            "error_type": type(
                error
            ).__name__,
            "error_message": str(
                error
            ),
        }


def run_audit():
    """Audit all frozen mechanistic-recovery models."""
    rows = []

    for model_id in FROZEN_MODEL_IDS:
        row = audit_one_model(
            model_id
        )

        rows.append(
            row
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


def print_audit_summary(
    frame,
):
    """Print compact aggregate audit results."""
    total_models = len(
        frame
    )

    extraction_successes = int(
        (
            frame[
                "extraction_status"
            ]
            == "success"
        ).sum()
    )

    numerical_passes = int(
        (
            frame[
                "numerical_validation_status"
            ]
            == "passed"
        ).sum()
    )

    numerical_failures = int(
        (
            frame[
                "numerical_validation_status"
            ]
            == "failed"
        ).sum()
    )

    not_run = int(
        (
            frame[
                "numerical_validation_status"
            ]
            == "not_run"
        ).sum()
    )

    autonomous_models = int(
        (
            frame[
                "ground_truth_is_autonomous"
            ]
            == True
        ).sum()
    )

    non_autonomous_models = int(
        (
            frame[
                "ground_truth_is_autonomous"
            ]
            == False
        ).sum()
    )

    print()
    print("=" * 80)
    print("AUDIT SUMMARY")

    print(
        "Models:",
        total_models,
    )

    print(
        "Extraction successes:",
        extraction_successes,
    )

    print(
        "Numerical validation passes:",
        numerical_passes,
    )

    print(
        "Numerical validation failures:",
        numerical_failures,
    )

    print(
        "Numerical validation not run:",
        not_run,
    )

    print(
        "Autonomous ground-truth models:",
        autonomous_models,
    )

    print(
        "Non-autonomous ground-truth models:",
        non_autonomous_models,
    )

    print()
    print(
        "Output:",
        OUTPUT_PATH,
    )

    print()
    print(
        frame[
            [
                "model_id",
                "n_states",
                "n_reactions",
                "ground_truth_is_autonomous",
                "ground_truth_independent_variables",
                "extraction_status",
                "numerical_validation_status",
                "maximum_absolute_error",
                "maximum_relative_error",
                "error_type",
            ]
        ].to_string(
            index=False
        )
    )


def main():
    """Run the frozen-model mechanistic ground-truth audit."""
    frame = run_audit()

    print_audit_summary(
        frame
    )


if __name__ == "__main__":
    main()