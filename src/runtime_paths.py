"""Centralized native and container filesystem configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NATIVE_MODEL_ROOT = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters"
)
NATIVE_SIMULATION_METADATA_PATH = (
    PROJECT_ROOT.parent / "data2sbml" / "output" / "time_curated_images.csv"
)
NATIVE_MODEL_TIME_METADATA_PATH = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters_model_time.jsonl"
)
NATIVE_RESULTS_ROOT = PROJECT_ROOT / "results"
NATIVE_RUNS_ROOT = PROJECT_ROOT / "runs"
NATIVE_CACHE_ROOT = PROJECT_ROOT / ".cache"


def configured_path(environment_name: str, native_default: Path) -> Path:
    """Resolve one optional environment path without Docker-specific logic.

    Args:
        environment_name: Environment variable containing an optional path.
        native_default: Existing native path used when the variable is absent.

    Returns:
        Absolute configured path with user markers expanded.
    """
    configured = os.environ.get(environment_name)
    selected = Path(configured) if configured else Path(native_default)
    return selected.expanduser().resolve()


def model_input_root() -> Path:
    """Return the configured BioModels input root.

    Returns:
        Read-only model-tree root used by model resolution.
    """
    return configured_path("PYSR_MODEL_ROOT", NATIVE_MODEL_ROOT)


def simulation_metadata_path() -> Path:
    """Return the configured simulation-metadata CSV path.

    Returns:
        Curated simulation metadata path.
    """
    return configured_path(
        "PYSR_SIMULATION_METADATA_PATH",
        NATIVE_SIMULATION_METADATA_PATH,
    )


def model_time_metadata_path() -> Path:
    """Return the configured model-aware simulation-time metadata path."""
    return configured_path(
        "PYSR_MODEL_TIME_METADATA_PATH",
        NATIVE_MODEL_TIME_METADATA_PATH,
    )


def results_root() -> Path:
    """Return the configured scientific-results root.

    Returns:
        Scientific output root.
    """
    return configured_path("PYSR_RESULTS_ROOT", NATIVE_RESULTS_ROOT)


def runs_root() -> Path:
    """Return the configured orchestration-run root.

    Returns:
        Manifest, log, lock, and telemetry root.
    """
    return configured_path("PYSR_RUNS_ROOT", NATIVE_RUNS_ROOT)


def cache_root() -> Path:
    """Return the configured application cache root.

    Returns:
        Writable application cache root.
    """
    return configured_path("PYSR_CACHE_ROOT", NATIVE_CACHE_ROOT)
