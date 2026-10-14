"""Resolve source SBML models used by the generic benchmark."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ROOT = (
    PROJECT_ROOT.parent
    / "data2sbml"
    / "output"
    / "all_curated_biomodels_no_variable_parameters"
)


def resolve_model_path(
    model_id: str,
    model_root: Path | None = None,
) -> Path:
    """Resolve and validate the source SBML path for one BioModel."""
    if model_root is None:
        model_root = DEFAULT_MODEL_ROOT

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