"""Resolve source SBML models used by the generic benchmark."""

from pathlib import Path

from .runtime_paths import NATIVE_MODEL_ROOT, model_input_root

DEFAULT_MODEL_ROOT = NATIVE_MODEL_ROOT


def resolve_model_path(
    model_id: str,
    model_root: Path | None = None,
) -> Path:
    """Resolve and validate the source SBML path for one BioModel."""
    if model_root is None:
        model_root = model_input_root()

    model_root = Path(
        model_root
    )

    model_path = (
        model_root
        / model_id
        / f"{model_id}_url.xml"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"SBML model not found: {model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"SBML path is not a file: {model_path}"
        )

    return model_path.resolve()
