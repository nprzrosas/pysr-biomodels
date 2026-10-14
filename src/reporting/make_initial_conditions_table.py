"""Generate human-readable initial-condition tables for reporting.

Purpose
-------
Extract the frozen deterministic experimental design used by the generic
PySR benchmark and write human-readable tables describing the initial
conditions assigned to:

- training trajectories;
- validation trajectories;
- held-out test trajectories.

This is a reporting-only utility.

It does not:

- simulate SBML models;
- run RoadRunner;
- run PySR;
- select candidate equations;
- refit equations;
- evaluate trajectories;
- read held-out test outcomes.

The test table contains only the predetermined initial conditions from the
experimental design. No held-out test predictions or performance metrics are
read.

Outputs
-------
For each model:

results/<model_id>/generic_exact_experiment/reporting/
    training_initial_conditions.csv
    validation_initial_conditions.csv
    test_initial_conditions.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..experimental_design_generic import build_experimental_design
from ..export_inferred_sbml import load_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "generic_exact_experiment"

REPORTING_DIRECTORY = "reporting"

TRAINING_FILENAME = "training_initial_conditions.csv"

VALIDATION_FILENAME = "validation_initial_conditions.csv"

TEST_FILENAME = "test_initial_conditions.csv"


@dataclass(frozen=True)
class InitialConditionTablePaths:
    """Paths to generated reporting tables."""

    training: Path
    validation: Path
    test: Path


def reporting_directory(
    model_id: str,
) -> Path:
    """Return model-level reporting directory."""

    path = (
        PROJECT_ROOT
        / "results"
        / model_id
        / EXPERIMENT_NAME
        / REPORTING_DIRECTORY
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def validate_initial_condition_matrix(
    *,
    matrix: np.ndarray,
    state_ids: tuple[str, ...],
    expected_rows: int,
    design_name: str,
) -> None:
    """Validate one experimental-design initial-condition matrix."""

    if matrix.ndim != 2:
        raise ValueError(
            f"{design_name} initial conditions must be a "
            "two-dimensional matrix."
        )

    expected_shape = (
        expected_rows,
        len(state_ids),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{design_name} initial-condition matrix has shape "
            f"{matrix.shape}; expected {expected_shape}."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            f"{design_name} initial conditions contain "
            "non-finite values."
        )

    if np.any(
        matrix <= 0.0
    ):
        raise ValueError(
            f"{design_name} initial conditions contain "
            "non-positive values."
        )


def make_initial_condition_frame(
    *,
    matrix: np.ndarray,
    state_ids: tuple[str, ...],
    trajectory_prefix: str,
    split_name: str,
) -> pd.DataFrame:
    """Convert an initial-condition matrix to a reporting table."""

    rows = []

    for index, initial_values in enumerate(
        matrix
    ):
        row = {
            "split": split_name,
            "trajectory_index": index,
            "trajectory_id": (
                f"{trajectory_prefix}_{index:03d}"
            ),
        }

        for state_id, value in zip(
            state_ids,
            initial_values,
            strict=True,
        ):
            row[
                f"{state_id}_0"
            ] = float(
                value
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def add_training_membership_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate nested training-set membership.

    The generic benchmark uses nested training sets:

    train_04 ⊂ train_08 ⊂ train_16 ⊂ train_32.

    Therefore the first four trajectories belong to all four training
    conditions, the first eight belong to train_08/train_16/train_32,
    and so on.
    """

    result = frame.copy()

    trajectory_index = result[
        "trajectory_index"
    ].to_numpy(
        dtype=int
    )

    result[
        "in_train_04"
    ] = (
        trajectory_index
        < 4
    )

    result[
        "in_train_08"
    ] = (
        trajectory_index
        < 8
    )

    result[
        "in_train_16"
    ] = (
        trajectory_index
        < 16
    )

    result[
        "in_train_32"
    ] = (
        trajectory_index
        < 32
    )

    return result


def validate_nested_training_membership(
    frame: pd.DataFrame,
) -> None:
    """Validate expected nested training-set sizes."""

    expected_counts = {
        "in_train_04": 4,
        "in_train_08": 8,
        "in_train_16": 16,
        "in_train_32": 32,
    }

    for column, expected_count in expected_counts.items():
        if column not in frame.columns:
            raise RuntimeError(
                f"Missing training membership column: {column}"
            )

        observed_count = int(
            frame[
                column
            ].sum()
        )

        if observed_count != expected_count:
            raise RuntimeError(
                f"{column} contains {observed_count} trajectories; "
                f"expected {expected_count}."
            )

    train_04 = frame[
        "in_train_04"
    ].to_numpy(
        dtype=bool
    )

    train_08 = frame[
        "in_train_08"
    ].to_numpy(
        dtype=bool
    )

    train_16 = frame[
        "in_train_16"
    ].to_numpy(
        dtype=bool
    )

    train_32 = frame[
        "in_train_32"
    ].to_numpy(
        dtype=bool
    )

    if np.any(
        train_04
        & ~train_08
    ):
        raise RuntimeError(
            "train_04 is not nested inside train_08."
        )

    if np.any(
        train_08
        & ~train_16
    ):
        raise RuntimeError(
            "train_08 is not nested inside train_16."
        )

    if np.any(
        train_16
        & ~train_32
    ):
        raise RuntimeError(
            "train_16 is not nested inside train_32."
        )


def write_initial_condition_tables(
    model_id: str,
) -> InitialConditionTablePaths:
    """Generate all initial-condition reporting tables."""

    spec = load_model_spec(
        model_id
    )

    design = build_experimental_design(
        spec=spec,
    )

    state_ids = tuple(
        spec.state_ids
    )

    training_matrix = np.asarray(
        design.train,
        dtype=float,
    )

    validation_matrix = np.asarray(
        design.validation,
        dtype=float,
    )

    test_matrix = np.asarray(
        design.test,
        dtype=float,
    )

    validate_initial_condition_matrix(
        matrix=training_matrix,
        state_ids=state_ids,
        expected_rows=32,
        design_name="training",
    )

    validate_initial_condition_matrix(
        matrix=validation_matrix,
        state_ids=state_ids,
        expected_rows=8,
        design_name="validation",
    )

    validate_initial_condition_matrix(
        matrix=test_matrix,
        state_ids=state_ids,
        expected_rows=8,
        design_name="test",
    )

    training = make_initial_condition_frame(
        matrix=training_matrix,
        state_ids=state_ids,
        trajectory_prefix="train",
        split_name="training",
    )

    training = add_training_membership_columns(
        training
    )

    validate_nested_training_membership(
        training
    )

    validation = make_initial_condition_frame(
        matrix=validation_matrix,
        state_ids=state_ids,
        trajectory_prefix="validation",
        split_name="validation",
    )

    test = make_initial_condition_frame(
        matrix=test_matrix,
        state_ids=state_ids,
        trajectory_prefix="test",
        split_name="test",
    )

    destination = reporting_directory(
        model_id
    )

    training_path = (
        destination
        / TRAINING_FILENAME
    )

    validation_path = (
        destination
        / VALIDATION_FILENAME
    )

    test_path = (
        destination
        / TEST_FILENAME
    )

    training.to_csv(
        training_path,
        index=False,
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
    )

    return InitialConditionTablePaths(
        training=training_path,
        validation=validation_path,
        test=test_path,
    )


def print_table(
    *,
    title: str,
    path: Path,
) -> None:
    """Print one generated initial-condition table."""

    frame = pd.read_csv(
        path
    )

    print(
        title
    )

    print(
        "-" * len(
            title
        )
    )

    print(
        frame.to_string(
            index=False
        )
    )

    print()


def generate_report_tables(
    model_id: str,
) -> InitialConditionTablePaths:
    """Generate and display initial-condition reporting tables."""

    spec = load_model_spec(
        model_id
    )

    paths = write_initial_condition_tables(
        model_id
    )

    print(
        f"Model: {model_id}"
    )

    print(
        "States: "
        + ", ".join(
            spec.state_ids
        )
    )

    print()

    print_table(
        title="Training initial conditions",
        path=paths.training,
    )

    print_table(
        title="Validation initial conditions",
        path=paths.validation,
    )

    print_table(
        title="Held-out test initial conditions",
        path=paths.test,
    )

    print(
        "Generated reporting tables:"
    )

    print(
        f"  training:   {paths.training}"
    )

    print(
        f"  validation: {paths.validation}"
    )

    print(
        f"  test:       {paths.test}"
    )

    print()

    print(
        "No SBML simulation, PySR inference, candidate selection, "
        "refitting, or performance evaluation was performed."
    )

    print(
        "The held-out test table contains initial conditions only; "
        "no held-out test outcomes were read."
    )

    return paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate human-readable training, validation, and "
            "held-out test initial-condition tables."
        )
    )

    parser.add_argument(
        "model_id",
        help=(
            "BioModels identifier, for example "
            "BIOMD0000000079."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate initial-condition reporting tables."""

    args = parse_arguments()

    generate_report_tables(
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()