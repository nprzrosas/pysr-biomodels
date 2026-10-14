"""Compatibility checks for the generic PySR benchmark."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .model_spec import ModelSpec


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of checking whether a model fits the current benchmark."""

    model_id: str
    compatible: bool
    reasons: tuple[str, ...]


def _local_name(tag: str) -> str:
    """Return an XML tag name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _count_sbml_elements(
    model_path: Path,
) -> dict[str, int]:
    """Count SBML constructs relevant to the current compatibility policy."""
    root = ET.parse(
        model_path
    ).getroot()

    counts = {
        "event": 0,
        "rateRule": 0,
        "assignmentRule": 0,
        "algebraicRule": 0,
    }

    for element in root.iter():
        name = _local_name(
            element.tag
        )

        if name in counts:
            counts[name] += 1

    return counts


def check_model_compatibility(
    spec: ModelSpec,
) -> CompatibilityReport:
    """Check whether a model is supported by the current generic benchmark."""
    reasons = []

    if spec.n_states < 1:
        reasons.append(
            "Model has no floating dynamic species."
        )

    for state_id in spec.state_ids:
        initial_value = float(
            spec.initial_values[state_id]
        )

        if not np.isfinite(
            initial_value
        ):
            reasons.append(
                f"State {state_id} has a non-finite "
                f"initial value: {initial_value}."
            )

        elif initial_value <= 0.0:
            reasons.append(
                f"State {state_id} has initial value "
                f"{initial_value}; the current log-space "
                "experimental design requires strictly "
                "positive initial values."
            )

    counts = _count_sbml_elements(
        spec.model_path
    )

    if counts["event"] > 0:
        reasons.append(
            "Model contains "
            f"{counts['event']} event(s); events are "
            "not supported by the current benchmark."
        )

    if counts["rateRule"] > 0:
        reasons.append(
            "Model contains "
            f"{counts['rateRule']} rate rule(s); rate "
            "rules are not supported by the current benchmark."
        )

    if counts["assignmentRule"] > 0:
        reasons.append(
            "Model contains "
            f"{counts['assignmentRule']} assignment rule(s); "
            "assignment rules are not supported by the "
            "current benchmark."
        )

    if counts["algebraicRule"] > 0:
        reasons.append(
            "Model contains "
            f"{counts['algebraicRule']} algebraic rule(s); "
            "algebraic rules are not supported by the "
            "current benchmark."
        )

    return CompatibilityReport(
        model_id=spec.model_id,
        compatible=len(reasons) == 0,
        reasons=tuple(reasons),
    )


def require_model_compatibility(
    spec: ModelSpec,
) -> None:
    """Raise an error when a model is outside the supported benchmark."""
    report = check_model_compatibility(
        spec
    )

    if report.compatible:
        return

    reason_text = "\n".join(
        f"  - {reason}"
        for reason in report.reasons
    )

    raise ValueError(
        f"Model {spec.model_id} is not compatible "
        "with the current generic benchmark:\n"
        f"{reason_text}"
    )