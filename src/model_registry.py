"""Resolve source SBML models used by the generic benchmark."""

from pathlib import Path

from .runtime_paths import NATIVE_MODEL_ROOT, model_input_root

DEFAULT_MODEL_ROOT = NATIVE_MODEL_ROOT


def resolve_model_path(
    model_id: str,
    model_root: Path | None = None,
) -> Path:
    """Resolve and validate the source SBML path for one BioModel.

    Prefer the historical ``<MODEL_ID>_url.xml`` filename when present.
    Otherwise, fall back to the unique XML/SBML file in the model
    directory. Reject missing or ambiguous model directories explicitly.
    """
    if model_root is None:
        model_root = model_input_root()

    model_root = Path(model_root)
    model_dir = model_root / model_id

    if not model_dir.exists():
        raise FileNotFoundError(
            f"BioModel directory not found: {model_dir}"
        )

    if not model_dir.is_dir():
        raise ValueError(
            f"BioModel path is not a directory: {model_dir}"
        )

    canonical_path = (
        model_dir
        / f"{model_id}_url.xml"
    )

    if canonical_path.is_file():
        return canonical_path.resolve()

    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in {".xml", ".sbml"}
        )
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No SBML model file found for "
            f"{model_id} in {model_dir}"
        )

    if len(candidates) > 1:
        candidate_names = ", ".join(
            path.name
            for path in candidates
        )
        raise ValueError(
            "Multiple SBML model files found for "
            f"{model_id}: {candidate_names}"
        )

    return candidates[0].resolve()
