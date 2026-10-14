"""Migrate legacy coordinate-search system IDs to canonical Cartesian IDs."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "results"

EXPERIMENT_NAME = "generic_exact_experiment"

COORDINATE_DIRECTORY = "coordinate_selection"

TRACE_FILENAME = "coordinate_search_trace.csv"

COORDINATE_FILES = (
    "coordinate_search_trace.csv",
    "coordinate_selected_system.csv",
    "coordinate_validation_predictions.csv",
    "coordinate_validation_ranking.csv",
    "coordinate_validation_state_metrics.csv",
)

BACKUP_SUFFIX = ".pre_canonical_id.bak"


def discover_coordinate_directories():
    """Return all existing coordinate-selection result directories."""
    pattern = (
        f"*/{EXPERIMENT_NAME}/"
        f"train_*/{COORDINATE_DIRECTORY}"
    )

    return sorted(
        RESULTS_ROOT.glob(pattern)
    )


def get_rank_columns(trace):
    """Return candidate-rank columns in trace column order."""
    rank_columns = [
        column
        for column in trace.columns
        if column.startswith("candidate_rank_")
    ]

    if not rank_columns:
        raise ValueError(
            "No candidate-rank columns found in coordinate trace."
        )

    return rank_columns


def get_candidate_counts(
    *,
    coordinate_dir,
    rank_columns,
):
    """Load candidate counts from the PySR top-candidate files."""
    experiment_dir = coordinate_dir.parent

    candidate_counts = []

    for rank_column in rank_columns:
        state_id = rank_column.removeprefix(
            "candidate_rank_"
        )

        candidate_path = (
            experiment_dir
            / f"state_{state_id}"
            / "top_8_by_loss.csv"
        )

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Missing candidate file: {candidate_path}"
            )

        candidates = pd.read_csv(
            candidate_path
        )

        candidate_count = len(candidates)

        if candidate_count <= 0:
            raise ValueError(
                f"No candidates found for state {state_id}."
            )

        candidate_counts.append(
            candidate_count
        )

    return candidate_counts


def calculate_canonical_system_id(
    *,
    ranks,
    candidate_counts,
):
    """Calculate one-based Cartesian system ID from candidate ranks."""
    if len(ranks) != len(candidate_counts):
        raise ValueError(
            "Candidate ranks and candidate counts have different lengths."
        )

    zero_based_system_id = 0

    for state_index, candidate_rank in enumerate(ranks):
        candidate_rank = int(candidate_rank)

        candidate_count = int(
            candidate_counts[state_index]
        )

        if (
            candidate_rank < 1
            or candidate_rank > candidate_count
        ):
            raise ValueError(
                f"Invalid candidate rank {candidate_rank} "
                f"for candidate count {candidate_count}."
            )

        trailing_product = int(
            np.prod(
                candidate_counts[state_index + 1:],
                dtype=int,
            )
        )

        zero_based_system_id += (
            candidate_rank - 1
        ) * trailing_product

    return zero_based_system_id + 1


def build_id_mapping(
    *,
    coordinate_dir,
    trace_path=None,
):
    """Build legacy-to-canonical system-ID mapping from a coordinate trace."""
    if trace_path is None:
        trace_path = (
            coordinate_dir
            / TRACE_FILENAME
        )

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Missing coordinate trace: {trace_path}"
        )

    trace = pd.read_csv(
        trace_path
    )

    if "system_id" not in trace.columns:
        raise ValueError(
            f"Missing system_id column in {trace_path}"
        )

    if trace["system_id"].duplicated().any():
        raise ValueError(
            f"System IDs are not unique in {trace_path}"
        )

    rank_columns = get_rank_columns(
        trace
    )

    candidate_counts = get_candidate_counts(
        coordinate_dir=coordinate_dir,
        rank_columns=rank_columns,
    )

    mapping = {}
    canonical_ids = []

    for row in trace.itertuples(index=False):
        source_system_id = int(
            getattr(
                row,
                "system_id",
            )
        )

        ranks = tuple(
            int(
                getattr(
                    row,
                    rank_column,
                )
            )
            for rank_column in rank_columns
        )

        canonical_system_id = calculate_canonical_system_id(
            ranks=ranks,
            candidate_counts=candidate_counts,
        )

        mapping[source_system_id] = (
            canonical_system_id
        )

        canonical_ids.append(
            canonical_system_id
        )

    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError(
            "Canonical system-ID collision detected."
        )

    return {
        "mapping": mapping,
        "rank_columns": rank_columns,
        "candidate_counts": candidate_counts,
        "trace": trace,
    }


def backup_path_for(path):
    """Return the backup path associated with a coordinate CSV."""
    return Path(
        str(path) + BACKUP_SUFFIX
    )


def existing_coordinate_files(
    *,
    coordinate_dir,
):
    """Return coordinate CSV files that currently exist."""
    return [
        coordinate_dir / filename
        for filename in COORDINATE_FILES
        if (
            coordinate_dir
            / filename
        ).exists()
    ]


def classify_migration_state(
    *,
    coordinate_dir,
):
    """Classify a directory as legacy, migrated, or inconsistent."""
    files = existing_coordinate_files(
        coordinate_dir=coordinate_dir,
    )

    if not files:
        raise ValueError(
            f"No coordinate result files found in {coordinate_dir}"
        )

    backup_flags = [
        backup_path_for(path).exists()
        for path in files
    ]

    if not any(backup_flags):
        return "legacy"

    if not all(backup_flags):
        return "inconsistent"

    backup_trace_path = backup_path_for(
        coordinate_dir
        / TRACE_FILENAME
    )

    if not backup_trace_path.exists():
        return "inconsistent"

    migration_info = build_id_mapping(
        coordinate_dir=coordinate_dir,
        trace_path=backup_trace_path,
    )

    mapping = migration_info["mapping"]

    current_trace = pd.read_csv(
        coordinate_dir
        / TRACE_FILENAME
    )

    backup_trace = pd.read_csv(
        backup_trace_path
    )

    if len(current_trace) != len(backup_trace):
        return "inconsistent"

    if "system_id" not in current_trace.columns:
        return "inconsistent"

    expected_ids = backup_trace[
        "system_id"
    ].map(mapping)

    if expected_ids.isna().any():
        return "inconsistent"

    expected_ids = expected_ids.astype(int)

    current_ids = current_trace[
        "system_id"
    ].astype(int)

    if current_ids.equals(expected_ids):
        return "migrated"

    return "inconsistent"


def make_backup(
    *,
    path,
):
    """Create a pre-migration backup and reject existing backups."""
    backup_path = backup_path_for(
        path
    )

    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_path}"
        )

    shutil.copy2(
        path,
        backup_path,
    )

    return backup_path


def migrate_csv(
    *,
    path,
    mapping,
):
    """Replace legacy system IDs in one coordinate-result CSV."""
    frame = pd.read_csv(
        path
    )

    if "system_id" not in frame.columns:
        return {
            "filename": path.name,
            "rows": len(frame),
            "changed_rows": 0,
            "status": "no_system_id_column",
        }

    original_ids = frame[
        "system_id"
    ].copy()

    unknown_ids = sorted(
        set(
            int(value)
            for value in original_ids.dropna().unique()
        )
        - set(mapping)
    )

    if unknown_ids:
        raise ValueError(
            f"{path} contains system IDs absent from trace: "
            f"{unknown_ids}"
        )

    migrated_ids = original_ids.map(
        mapping
    )

    if migrated_ids.isna().any():
        raise ValueError(
            f"Failed to migrate every system ID in {path}"
        )

    migrated_ids = migrated_ids.astype(int)

    changed_rows = int(
        (
            migrated_ids
            != original_ids.astype(int)
        ).sum()
    )

    frame["system_id"] = migrated_ids

    frame.to_csv(
        path,
        index=False,
    )

    return {
        "filename": path.name,
        "rows": len(frame),
        "changed_rows": changed_rows,
        "status": "migrated",
    }


def verify_migration(
    *,
    coordinate_dir,
    mapping,
):
    """Verify that migrated coordinate files use canonical IDs."""
    canonical_ids = set(
        mapping.values()
    )

    for filename in COORDINATE_FILES:
        path = (
            coordinate_dir
            / filename
        )

        if not path.exists():
            continue

        frame = pd.read_csv(
            path
        )

        if "system_id" not in frame.columns:
            continue

        file_ids = set(
            int(value)
            for value in frame[
                "system_id"
            ].dropna().unique()
        )

        invalid_ids = (
            file_ids
            - canonical_ids
        )

        if invalid_ids:
            raise ValueError(
                f"Post-migration verification failed for {path}: "
                f"{sorted(invalid_ids)}"
            )


def migrate_coordinate_directory(
    *,
    coordinate_dir,
):
    """Migrate one coordinate-selection directory safely."""
    migration_state = classify_migration_state(
        coordinate_dir=coordinate_dir,
    )

    if migration_state == "migrated":
        return {
            "coordinate_dir": coordinate_dir,
            "status": "already_migrated",
            "results": [],
        }

    if migration_state == "inconsistent":
        raise RuntimeError(
            "Migration state is inconsistent for "
            f"{coordinate_dir}. Refusing to modify files."
        )

    migration_info = build_id_mapping(
        coordinate_dir=coordinate_dir,
    )

    mapping = migration_info[
        "mapping"
    ]

    files = existing_coordinate_files(
        coordinate_dir=coordinate_dir,
    )

    for path in files:
        make_backup(
            path=path,
        )

    results = []

    for path in files:
        result = migrate_csv(
            path=path,
            mapping=mapping,
        )

        results.append(
            result
        )

    verify_migration(
        coordinate_dir=coordinate_dir,
        mapping=mapping,
    )

    final_state = classify_migration_state(
        coordinate_dir=coordinate_dir,
    )

    if final_state != "migrated":
        raise RuntimeError(
            "Migration completed but final verification "
            f"failed for {coordinate_dir}."
        )

    return {
        "coordinate_dir": coordinate_dir,
        "status": "migrated",
        "mapping": mapping,
        "rank_columns": migration_info[
            "rank_columns"
        ],
        "candidate_counts": migration_info[
            "candidate_counts"
        ],
        "results": results,
    }


def main():
    """Migrate all legacy coordinate-search result IDs."""
    coordinate_directories = (
        discover_coordinate_directories()
    )

    if not coordinate_directories:
        raise RuntimeError(
            "No coordinate-selection directories found."
        )

    print("=" * 100)
    print("COORDINATE SYSTEM-ID MIGRATION")
    print("=" * 100)

    print(
        "Coordinate directories found:",
        len(coordinate_directories),
    )

    total_files = 0
    total_changed_rows = 0
    migrated_directories = 0
    already_migrated_directories = 0

    for coordinate_dir in coordinate_directories:
        print()
        print("-" * 100)

        print(
            coordinate_dir.relative_to(
                PROJECT_ROOT
            )
        )

        print("-" * 100)

        migration = migrate_coordinate_directory(
            coordinate_dir=coordinate_dir,
        )

        if migration["status"] == "already_migrated":
            already_migrated_directories += 1

            print(
                "Status: already migrated; no files modified."
            )

            continue

        migrated_directories += 1

        print("Status: migrated")

        print(
            "Candidate counts:",
            migration[
                "candidate_counts"
            ],
        )

        print(
            "Systems in trace:",
            len(
                migration[
                    "mapping"
                ]
            ),
        )

        for result in migration["results"]:
            total_files += 1

            total_changed_rows += result[
                "changed_rows"
            ]

            print(
                f"{result['filename']}: "
                f"{result['status']}, "
                f"rows={result['rows']}, "
                f"changed_rows={result['changed_rows']}"
            )

    print()
    print("=" * 100)
    print("MIGRATION COMPLETE")
    print("=" * 100)

    print(
        "Directories migrated:",
        migrated_directories,
    )

    print(
        "Directories already migrated:",
        already_migrated_directories,
    )

    print(
        "Files processed:",
        total_files,
    )

    print(
        "Rows with updated system_id:",
        total_changed_rows,
    )

    print(
        "Backup suffix:",
        BACKUP_SUFFIX,
    )


if __name__ == "__main__":
    main()