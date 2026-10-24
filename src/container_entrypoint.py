"""Initialize writable container state and execute the requested Python command."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


OUTPUT_ROOT = Path("/output")
JULIA_ENV_TEMPLATE = Path("/opt/pysr/julia_env_template")
JULIA_ENV_RUNTIME = Path("/output/cache/julia_env")

RUNTIME_DIRECTORIES = (
    Path("/output/results"),
    Path("/output/runs"),
    Path("/output/cache/project"),
    Path("/output/cache/uv"),
    Path("/output/cache/julia"),
)


def initialize_runtime() -> None:
    """Create writable runtime state on the host-backed /output filesystem."""
    for path in RUNTIME_DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)

    if not JULIA_ENV_RUNTIME.exists():
        if not JULIA_ENV_TEMPLATE.is_dir():
            raise RuntimeError(
                f"Missing immutable Julia environment template: "
                f"{JULIA_ENV_TEMPLATE}"
            )

        JULIA_ENV_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(JULIA_ENV_TEMPLATE, JULIA_ENV_RUNTIME)

        # The immutable template is intentionally read-only.  The runtime
        # copy belongs to the invoking host user and must be writable because
        # juliapkg uses lock.pid and may update its project state.
        for path in JULIA_ENV_RUNTIME.rglob("*"):
            path.chmod(path.stat().st_mode | 0o200)
        JULIA_ENV_RUNTIME.chmod(JULIA_ENV_RUNTIME.stat().st_mode | 0o200)

    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(JULIA_ENV_RUNTIME)


def main() -> None:
    initialize_runtime()

    if len(sys.argv) < 2:
        args = ["-m", "src.container_smoke", "--resources-only"]
    else:
        args = sys.argv[1:]

    os.execv(sys.executable, [sys.executable, *args])


if __name__ == "__main__":
    main()
