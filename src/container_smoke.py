"""Lightweight container-runtime diagnostics without benchmark execution."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from .preflight import run_preflight
from .resource_monitor import discover_resources
from .runtime_paths import (
    cache_root,
    model_input_root,
    results_root,
    runs_root,
    simulation_metadata_path,
)

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
)


def package_version(distribution_name: str) -> str:
    """Return one installed Python distribution version.

    Args:
        distribution_name: Distribution name understood by importlib.metadata.

    Returns:
        Installed version string.
    """
    return importlib.metadata.version(distribution_name)


def writable_root_check(root: Path) -> bool:
    """Verify one configured root with a temporary harmless marker.

    Args:
        root: Writable directory to exercise.

    Returns:
        ``True`` after a successful create/write/delete cycle.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".container_write_smoke"
    marker.write_text("ok\n", encoding="utf-8")
    marker.unlink()
    return True


def resource_report() -> dict[str, Any]:
    """Build a JSON-safe resource, path, and thread report.

    Returns:
        Container runtime observations without initializing Julia.
    """
    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "resources": discover_resources(),
        "paths": {
            "model_input_root": str(model_input_root()),
            "simulation_metadata_path": str(simulation_metadata_path()),
            "results_root": str(results_root()),
            "runs_root": str(runs_root()),
            "cache_root": str(cache_root()),
        },
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def full_runtime_report(
    *,
    model_id: str | None,
    simulate: bool,
    check_writable: bool,
) -> dict[str, Any]:
    """Import scientific dependencies and optionally inspect one SBML model.

    Args:
        model_id: Optional mounted model identifier for preflight.
        simulate: Whether to perform a three-point RoadRunner smoke simulation.
        check_writable: Whether to exercise configured writable roots.

    Returns:
        Complete runtime version and smoke-test report.
    """
    import libsbml
    import numpy
    import pandas
    import psutil
    import pysr
    import roadrunner
    import scipy
    import sympy
    from juliacall import Main as julia_main

    report = resource_report()
    report["versions"] = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "sympy": sympy.__version__,
        "psutil": psutil.__version__,
        "pysr": pysr.__version__,
        "juliacall": package_version("juliacall"),
        "julia": str(julia_main.seval("VERSION")),
        "roadrunner": roadrunner.__version__,
        "libsbml": libsbml.getLibSBMLDottedVersion(),
    }

    if model_id is not None:
        preflight = run_preflight(model_id)
        report["preflight"] = {
            "model_id": preflight.model_id,
            "ready": preflight.ready,
            "model_path": str(preflight.spec.model_path),
            "state_ids": list(preflight.spec.state_ids),
            "simulation_horizon": preflight.simulation_time,
        }
        if simulate:
            simulator = roadrunner.RoadRunner(str(preflight.spec.model_path))
            horizon = min(float(preflight.simulation_time or 1.0), 1.0)
            values = simulator.simulate(0.0, horizon, 3)
            report["roadrunner_smoke"] = {
                "rows": int(values.shape[0]),
                "columns": int(values.shape[1]),
                "finite": bool(numpy.isfinite(numpy.asarray(values)).all()),
            }

    if check_writable:
        report["writable_roots"] = {
            "results": writable_root_check(results_root()),
            "runs": writable_root_check(runs_root()),
            "cache": writable_root_check(cache_root()),
        }

    return report


def main(arguments: list[str] | None = None) -> None:
    """Run container diagnostics and print one JSON document.

    Args:
        arguments: Optional argument vector for tests.
    """
    parser = argparse.ArgumentParser(
        description="Inspect the container runtime without running a benchmark."
    )
    parser.add_argument("--resources-only", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--check-writable", action="store_true")
    options = parser.parse_args(arguments)
    if options.simulate and options.model_id is None:
        parser.error("--simulate requires --model-id")

    if options.resources_only:
        report = resource_report()
    else:
        report = full_runtime_report(
            model_id=options.model_id,
            simulate=options.simulate,
            check_writable=options.check_writable,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
