"""Atomic run manifests, artifact checkpoints, and model writer locks."""

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from . import runtime_paths
from .benchmark_variant import BenchmarkVariant

PROJECT_ROOT = runtime_paths.PROJECT_ROOT
DEFAULT_RESULTS_ROOT = runtime_paths.NATIVE_RESULTS_ROOT
DEFAULT_RUNS_ROOT = runtime_paths.NATIVE_RUNS_ROOT
DEFAULT_CACHE_ROOT = runtime_paths.NATIVE_CACHE_ROOT
DEFAULT_MODEL_ROOT = runtime_paths.NATIVE_MODEL_ROOT
DEFAULT_SIMULATION_METADATA_PATH = runtime_paths.NATIVE_SIMULATION_METADATA_PATH


def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp.

    Returns:
        An ISO-8601 timestamp using the UTC timezone.
    """
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ArtifactCheckpoint:
    """Stable inventory for a collection of scientific artifacts."""

    file_count: int
    path_hash: str
    content_hash: str


@dataclass(frozen=True)
class RunContext:
    """Filesystem and identity context shared by orchestration modules."""

    run_id: str
    project_root: Path = PROJECT_ROOT
    results_root: Path = field(default_factory=runtime_paths.results_root)
    runs_root: Path = field(default_factory=runtime_paths.runs_root)
    cache_root: Path = field(default_factory=runtime_paths.cache_root)
    model_input_root: Path = field(default_factory=runtime_paths.model_input_root)
    simulation_metadata_path: Path = field(
        default_factory=runtime_paths.simulation_metadata_path
    )

    def __post_init__(self) -> None:
        """Reject run identifiers that could escape the configured run root."""
        if (
            not self.run_id
            or self.run_id in {".", ".."}
            or Path(self.run_id).name != self.run_id
        ):
            raise ValueError("run_id must be one safe filesystem path component.")

    @property
    def run_directory(self) -> Path:
        """Return the run-specific, non-scientific metadata directory."""
        return self.runs_root / self.run_id

    @property
    def manifest_path(self) -> Path:
        """Return the run manifest path."""
        return self.run_directory / "manifest.json"

    @property
    def telemetry_path(self) -> Path:
        """Return the persistent telemetry path."""
        return self.run_directory / "telemetry.csv"

    @property
    def logs_directory(self) -> Path:
        """Return the per-model log directory."""
        return self.run_directory / "logs"

    @property
    def locks_directory(self) -> Path:
        """Return the cross-run model-lock directory.

        Locks deliberately live directly under ``runs/locks`` rather than
        inside one run directory so concurrent run IDs cannot write the same
        model and benchmark variant.
        """
        return self.runs_root / "locks"


def checkpoint_paths(
    paths: tuple[Path, ...] | list[Path],
    *,
    relative_to: Path,
) -> ArtifactCheckpoint:
    """Hash existing files by relative path and content.

    Args:
        paths: Files included in the checkpoint.
        relative_to: Root used to create stable relative paths.

    Returns:
        A deterministic artifact inventory.
    """
    root = Path(relative_to).resolve()
    normalized = sorted({Path(path).resolve() for path in paths})
    path_hasher = hashlib.sha256()
    content_hasher = hashlib.sha256()

    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint artifact is missing: {path}")

        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Checkpoint artifact is outside its declared root: {path}"
            ) from error

        encoded_path = relative_path.encode("utf-8")
        digest = hashlib.sha256(path.read_bytes()).digest()
        path_hasher.update(encoded_path)
        path_hasher.update(b"\0")
        content_hasher.update(encoded_path)
        content_hasher.update(b"\0")
        content_hasher.update(digest)

    return ArtifactCheckpoint(
        file_count=len(normalized),
        path_hash=path_hasher.hexdigest(),
        content_hash=content_hasher.hexdigest(),
    )


def checkpoint_tree(root: Path) -> ArtifactCheckpoint:
    """Checkpoint all files below a directory.

    Args:
        root: Directory whose complete file tree is inventoried.

    Returns:
        A deterministic artifact inventory.
    """
    root = Path(root)
    paths = tuple(path for path in root.rglob("*") if path.is_file())
    return checkpoint_paths(paths, relative_to=root)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON document in its destination directory.

    Args:
        path: Destination JSON path.
        payload: Serializable object to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def git_identity(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read the current Git commit and dirty state without changing Git.

    Args:
        project_root: Repository root.

    Returns:
        Commit hash and a Boolean dirty-state marker.
    """
    embedded_commit = os.environ.get("PYSR_GIT_COMMIT")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": embedded_commit or "unknown",
            "dirty": None,
        }
    return {"commit": commit, "dirty": bool(status.strip())}


class RunManifest:
    """Coordinator-owned atomic manifest for one orchestration run."""

    def __init__(self, *, path: Path, payload: dict[str, Any]) -> None:
        """Initialize a manifest without writing it.

        Args:
            path: Manifest destination.
            payload: Initial manifest document.
        """
        self.path = Path(path)
        self.payload = payload
        self._owner_thread = threading.get_ident()

    def _require_owner(self) -> None:
        """Reject manifest mutation outside the coordinator thread."""
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Only the coordinator may update the run manifest.")

    def write(self) -> None:
        """Atomically persist the current manifest payload."""
        self._require_owner()
        self.payload["updated_at_utc"] = utc_timestamp()
        atomic_write_json(self.path, self.payload)

    def update_model(self, model_id: str, **updates: Any) -> None:
        """Update one model record and persist atomically.

        Args:
            model_id: Model record to update.
            **updates: Fields merged into the model record.
        """
        self._require_owner()
        models = self.payload.setdefault("models", {})
        record = models.setdefault(model_id, {})
        record.update(updates)
        self.write()

    def update_stage(
        self,
        *,
        model_id: str,
        stage: str,
        status: str,
        **updates: Any,
    ) -> None:
        """Update a model stage with status and timestamp metadata.

        Args:
            model_id: Model record to update.
            stage: Explicit orchestration stage.
            status: New stage state.
            **updates: Additional stage fields.
        """
        self._require_owner()
        model = self.payload.setdefault("models", {}).setdefault(model_id, {})
        stages = model.setdefault("stages", {})
        stage_record = stages.setdefault(stage, {})
        stage_record.update(updates)
        stage_record["status"] = status
        timestamp = utc_timestamp()
        if status == "running":
            stage_record.setdefault("started_at_utc", timestamp)
        if status in {"completed", "failed", "skipped"}:
            stage_record["ended_at_utc"] = timestamp
        stage_record["updated_at_utc"] = timestamp
        self.write()


def create_run_manifest(
    *,
    context: RunContext,
    variant: BenchmarkVariant,
    requested_workers: int,
    effective_workers: int,
    requested_phase: str,
    resources: dict[str, Any],
    thread_environment: dict[str, str],
    model_records: dict[str, dict[str, Any]],
) -> RunManifest:
    """Create and persist a new run manifest.

    Args:
        context: Run filesystem context.
        variant: Authoritative benchmark variant.
        requested_workers: User-requested model worker count.
        effective_workers: Scheduler worker bound after cohort limits.
        requested_phase: Explicit orchestration phase.
        resources: Host and cgroup resource discovery.
        thread_environment: Thread limits applied to workers.
        model_records: Preflight-derived model metadata.

    Returns:
        The coordinator-owned manifest object.
    """
    if context.manifest_path.exists():
        raise FileExistsError(
            f"Run manifest already exists; load it to resume: {context.manifest_path}"
        )

    git = git_identity(context.project_root)
    payload: dict[str, Any] = {
        "run_id": context.run_id,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "derivative_method": variant.derivative_method.value,
        "experiment_namespace": variant.experiment_namespace,
        "scientific_namespace": variant.experiment_namespace,
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "requested_phase": requested_phase,
        "active_worker_count_at_startup": 0,
        "resources": resources,
        "thread_environment": thread_environment,
        "paths": {
            "project_root": str(context.project_root),
            "results_root": str(context.results_root),
            "runs_root": str(context.runs_root),
            "cache_root": str(context.cache_root),
            "model_input_root": str(context.model_input_root),
            "simulation_metadata_path": str(context.simulation_metadata_path),
        },
        "models": model_records,
    }
    manifest = RunManifest(path=context.manifest_path, payload=payload)
    manifest.write()
    return manifest


def load_run_manifest(*, context: RunContext) -> RunManifest:
    """Load an existing run manifest for coordinator-owned resume.

    Args:
        context: Run context whose manifest is loaded.

    Returns:
        A coordinator-owned manifest object.
    """
    with context.manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("run_id") != context.run_id:
        raise ValueError("Run manifest identity does not match the run context.")
    return RunManifest(path=context.manifest_path, payload=payload)


class ModelVariantLock:
    """Cross-run exclusive writer lock for one model and variant."""

    def __init__(
        self,
        *,
        context: RunContext,
        model_id: str,
        variant: BenchmarkVariant,
    ) -> None:
        """Configure a lock without acquiring it.

        Args:
            context: Run filesystem context.
            model_id: BioModels identifier.
            variant: Protected result namespace identity.
        """
        filename = f"{model_id}.{variant.derivative_method.value}.lock"
        self.path = context.locks_directory / filename
        self.run_id = context.run_id
        self._acquired = False

    def acquire(self) -> None:
        """Acquire the lock atomically or reject a concurrent writer."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

        try:
            descriptor = os.open(self.path, flags, 0o644)
        except FileExistsError as error:
            raise RuntimeError(
                f"Concurrent model/variant writer lock exists: {self.path}"
            ) from error

        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "created_at_utc": utc_timestamp(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
        self._acquired = True

    def release(self) -> None:
        """Release a lock acquired by this object."""
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self) -> Self:
        """Acquire and return the lock."""
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        """Release the lock on context-manager exit."""
        self.release()


def checkpoint_as_dict(checkpoint: ArtifactCheckpoint) -> dict[str, Any]:
    """Convert an artifact checkpoint to a JSON-safe dictionary.

    Args:
        checkpoint: Artifact checkpoint to serialize.

    Returns:
        A JSON-safe dictionary.
    """
    return asdict(checkpoint)
