"""Build and validate the frozen expanded-cohort Docker input bundle."""

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_registry import resolve_model_path
from .run_batch import load_cohort_manifest
from .simulation_metadata import load_simulation_time

EXPECTED_MODEL_COUNT = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT_MANIFEST = (
    PROJECT_ROOT / "results" / "expanded_model_inventory" / "benchmark_ready_100.json"
)
DEFAULT_MODEL_ROOT = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters"
)
DEFAULT_METADATA_PATH = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters_model_time.jsonl"
)
DEFAULT_BUNDLE_ROOT = PROJECT_ROOT / "docker" / "expanded_100_inputs"
BUNDLE_MANIFEST_RELATIVE_PATH = Path("metadata/benchmark_ready_100.json")
BUNDLE_METADATA_RELATIVE_PATH = Path("metadata/model_time.jsonl")
BUNDLE_PROVENANCE_RELATIVE_PATH = Path("metadata/bundle_provenance.json")


@dataclass(frozen=True)
class MetadataRecord:
    """One parsed model-aware metadata row with original serialized content."""

    payload: dict[str, Any]
    serialized_line: str


@dataclass(frozen=True)
class ValidatedModelInput:
    """Validated source artifacts for one expanded-cohort model."""

    model_id: str
    sbml_path: Path
    metadata_record: MetadataRecord
    simulation_time: float


def set_bundle_permissions(bundle_root: Path) -> None:
    """Normalize generated inputs for deterministic non-root read access.

    Args:
        bundle_root: Generated staging or final bundle root.
    """
    bundle_root = Path(bundle_root)
    for path in (bundle_root, *sorted(bundle_root.rglob("*"))):
        path.chmod(0o755 if path.is_dir() else 0o444)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_metadata_records(path: Path) -> dict[str, MetadataRecord]:
    """Load model-aware JSONL metadata and reject duplicate model IDs.

    Args:
        path: Source or bundled JSONL metadata path.

    Returns:
        Records keyed by unique model identifier.
    """
    records: dict[str, MetadataRecord] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            serialized = line.rstrip("\r\n")
            if not serialized.strip():
                continue
            try:
                payload = json.loads(serialized)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON metadata at {path}:{line_number}."
                ) from error
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Metadata row at {path}:{line_number} must be an object."
                )
            model_id = payload.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(
                    f"Metadata row at {path}:{line_number} lacks model_id."
                )
            if model_id in records:
                raise ValueError(f"Duplicate metadata model_id: {model_id}")
            records[model_id] = MetadataRecord(payload, serialized)
    return records


def validate_source_inputs(
    *,
    cohort_manifest: Path,
    model_root: Path,
    metadata_path: Path,
    expected_model_count: int = EXPECTED_MODEL_COUNT,
) -> tuple[ValidatedModelInput, ...]:
    """Resolve and validate every source required by one bundle build.

    Args:
        cohort_manifest: Ordered JSON cohort manifest.
        model_root: Root containing source model directories.
        metadata_path: Authoritative model-aware JSONL metadata.
        expected_model_count: Required manifest size.

    Returns:
        Validated source inputs in cohort order.
    """
    model_ids = load_cohort_manifest(Path(cohort_manifest))
    if len(model_ids) != expected_model_count:
        raise ValueError(
            "Expanded Docker cohort must contain exactly "
            f"{expected_model_count} models; found {len(model_ids)}."
        )

    records = load_metadata_records(Path(metadata_path))
    validated: list[ValidatedModelInput] = []
    for model_id in model_ids:
        sbml_path = resolve_model_path(model_id, Path(model_root))
        record = records.get(model_id)
        if record is None:
            raise ValueError(f"Missing metadata row for {model_id}.")
        simulation_time = load_simulation_time(
            model_id=model_id,
            sbml_filename=sbml_path.name,
            metadata_path=Path(metadata_path),
        )
        validated.append(
            ValidatedModelInput(
                model_id=model_id,
                sbml_path=sbml_path,
                metadata_record=record,
                simulation_time=simulation_time,
            )
        )
    return tuple(validated)


def build_bundle(
    *,
    cohort_manifest: Path = DEFAULT_COHORT_MANIFEST,
    model_root: Path = DEFAULT_MODEL_ROOT,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
) -> dict[str, Any]:
    """Create a new deterministic expanded-cohort input bundle.

    Args:
        cohort_manifest: Ordered validated 100-model JSON manifest.
        model_root: Source root containing the larger curated model inventory.
        metadata_path: Authoritative model-aware JSONL metadata.
        bundle_root: New bundle destination, which must not already exist.

    Returns:
        Validation summary for the completed bundle.
    """
    cohort_manifest = Path(cohort_manifest).resolve()
    model_root = Path(model_root).resolve()
    metadata_path = Path(metadata_path).resolve()
    bundle_root = Path(bundle_root).resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"Bundle destination already exists; refusing to overwrite: {bundle_root}"
        )

    validated = validate_source_inputs(
        cohort_manifest=cohort_manifest,
        model_root=model_root,
        metadata_path=metadata_path,
    )
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".expanded_100_inputs.", dir=bundle_root.parent)
    )
    try:
        models_destination = staging_root / "models"
        metadata_destination = staging_root / "metadata"
        models_destination.mkdir()
        metadata_destination.mkdir()

        shutil.copyfile(
            cohort_manifest,
            staging_root / BUNDLE_MANIFEST_RELATIVE_PATH,
        )
        bundled_metadata = staging_root / BUNDLE_METADATA_RELATIVE_PATH
        with bundled_metadata.open("w", encoding="utf-8", newline="\n") as stream:
            for item in validated:
                stream.write(f"{item.metadata_record.serialized_line}\n")

        provenance_models: list[dict[str, Any]] = []
        for item in validated:
            destination_directory = models_destination / item.model_id
            destination_directory.mkdir()
            destination_path = destination_directory / item.sbml_path.name
            shutil.copyfile(item.sbml_path, destination_path)
            provenance_models.append(
                {
                    "model_id": item.model_id,
                    "sbml_file": item.sbml_path.name,
                    "sbml_sha256": sha256_file(item.sbml_path),
                    "metadata_line_sha256": hashlib.sha256(
                        item.metadata_record.serialized_line.encode("utf-8")
                    ).hexdigest(),
                }
            )

        provenance = {
            "artifact_type": "expanded_100_docker_input_bundle",
            "schema_version": 1,
            "model_count": len(validated),
            "cohort_manifest_sha256": sha256_file(cohort_manifest),
            "source_model_time_metadata_sha256": sha256_file(metadata_path),
            "models": provenance_models,
        }
        provenance_path = staging_root / BUNDLE_PROVENANCE_RELATIVE_PATH
        provenance_path.write_text(
            f"{json.dumps(provenance, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        set_bundle_permissions(staging_root)
        staging_root.rename(bundle_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return validate_bundle(bundle_root)


def validate_bundle(
    bundle_root: Path,
    *,
    expected_model_count: int = EXPECTED_MODEL_COUNT,
) -> dict[str, Any]:
    """Validate bundle layout, checksums, resolution, and simulation times.

    Args:
        bundle_root: Frozen bundle root.
        expected_model_count: Required bundle size.

    Returns:
        JSON-safe validation summary.
    """
    bundle_root = Path(bundle_root).resolve()
    manifest_path = bundle_root / BUNDLE_MANIFEST_RELATIVE_PATH
    metadata_path = bundle_root / BUNDLE_METADATA_RELATIVE_PATH
    provenance_path = bundle_root / BUNDLE_PROVENANCE_RELATIVE_PATH
    model_root = bundle_root / "models"

    model_ids = load_cohort_manifest(manifest_path)
    if len(model_ids) != expected_model_count:
        raise ValueError(
            f"Bundle manifest must contain {expected_model_count} models; "
            f"found {len(model_ids)}."
        )
    records = load_metadata_records(metadata_path)
    if set(records) != set(model_ids):
        missing = sorted(set(model_ids) - set(records))
        extra = sorted(set(records) - set(model_ids))
        raise ValueError(
            f"Bundled metadata cohort mismatch; missing={missing}, extra={extra}."
        )

    directory_names = {path.name for path in model_root.iterdir() if path.is_dir()}
    non_directories = sorted(
        path.name for path in model_root.iterdir() if not path.is_dir()
    )
    if directory_names != set(model_ids) or non_directories:
        missing = sorted(set(model_ids) - directory_names)
        extra = sorted(directory_names - set(model_ids))
        raise ValueError(
            "Bundled model-directory mismatch; "
            f"missing={missing}, extra={extra}, non_directories={non_directories}."
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("model_count") != expected_model_count:
        raise ValueError("Bundle provenance model count is not exactly 100.")
    provenance_models = provenance.get("models")
    if not isinstance(provenance_models, list):
        raise TypeError("Bundle provenance models must be a list.")
    provenance_by_id = {
        row["model_id"]: row
        for row in provenance_models
        if isinstance(row, dict) and isinstance(row.get("model_id"), str)
    }
    if len(provenance_by_id) != expected_model_count:
        raise ValueError("Bundle provenance does not contain 100 unique models.")
    if sha256_file(manifest_path) != provenance.get("cohort_manifest_sha256"):
        raise ValueError("Bundled cohort manifest checksum does not match provenance.")

    for model_id in model_ids:
        sbml_path = resolve_model_path(model_id, model_root)
        model_files = tuple(
            path for path in sbml_path.parent.iterdir() if path.is_file()
        )
        if model_files != (sbml_path,):
            raise ValueError(
                f"Bundled model directory for {model_id} must contain only its SBML."
            )
        load_simulation_time(
            model_id=model_id,
            sbml_filename=sbml_path.name,
            metadata_path=metadata_path,
        )
        provenance_row = provenance_by_id.get(model_id)
        if provenance_row is None:
            raise ValueError(f"Bundle provenance lacks {model_id}.")
        if provenance_row.get("sbml_file") != sbml_path.name:
            raise ValueError(f"Bundle provenance filename mismatch for {model_id}.")
        if provenance_row.get("sbml_sha256") != sha256_file(sbml_path):
            raise ValueError(f"Bundle SBML checksum mismatch for {model_id}.")
        metadata_digest = hashlib.sha256(
            records[model_id].serialized_line.encode("utf-8")
        ).hexdigest()
        if provenance_row.get("metadata_line_sha256") != metadata_digest:
            raise ValueError(f"Bundle metadata checksum mismatch for {model_id}.")

    return {
        "bundle_root": str(bundle_root),
        "model_count": len(model_ids),
        "manifest_path": str(manifest_path),
        "model_root": str(model_root),
        "metadata_path": str(metadata_path),
        "all_models_valid": True,
    }


def main(arguments: list[str] | None = None) -> None:
    """Build or validate the expanded-cohort Docker input bundle.

    Args:
        arguments: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build or validate the frozen 100-model Docker input bundle."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args(arguments)
    if options.validate_only:
        summary = validate_bundle(options.output)
    else:
        summary = build_bundle(
            cohort_manifest=options.manifest,
            model_root=options.model_root,
            metadata_path=options.metadata,
            bundle_root=options.output,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
