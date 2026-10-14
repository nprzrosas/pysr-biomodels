"""Tests for container-neutral paths and Docker foundation files."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.container_smoke import main as container_smoke_main
from src.evaluate_generic_validation import resolve_validation_experiment_root
from src.run_manifest import RunContext, git_identity
from src.runtime_paths import (
    NATIVE_MODEL_ROOT,
    NATIVE_RESULTS_ROOT,
    model_input_root,
    results_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimePathTests(unittest.TestCase):
    """Verify environment overrides without changing native defaults."""

    def test_native_defaults_remain_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_input_root(), NATIVE_MODEL_ROOT.resolve())
            self.assertEqual(results_root(), NATIVE_RESULTS_ROOT.resolve())

    def test_environment_paths_feed_context_and_generic_scientific_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = {
                "PYSR_MODEL_ROOT": str(root / "models"),
                "PYSR_SIMULATION_METADATA_PATH": str(root / "metadata.csv"),
                "PYSR_RESULTS_ROOT": str(root / "results"),
                "PYSR_RUNS_ROOT": str(root / "runs"),
                "PYSR_CACHE_ROOT": str(root / "cache"),
            }
            with patch.dict(os.environ, environment, clear=False):
                context = RunContext(run_id="container-test")
                self.assertEqual(context.model_input_root, root / "models")
                self.assertEqual(context.results_root, root / "results")
                self.assertEqual(context.runs_root, root / "runs")
                self.assertEqual(context.cache_root, root / "cache")
                self.assertEqual(
                    resolve_validation_experiment_root(model_id="MODEL"),
                    root / "results" / "MODEL" / "generic_exact_experiment",
                )

    def test_embedded_git_commit_supports_images_without_dot_git(self) -> None:
        failure = subprocess.CalledProcessError(128, ("git", "rev-parse", "HEAD"))
        with (
            patch.dict(os.environ, {"PYSR_GIT_COMMIT": "abc123"}),
            patch("src.run_manifest.subprocess.run", side_effect=failure),
        ):
            self.assertEqual(
                git_identity(PROJECT_ROOT),
                {"commit": "abc123", "dirty": None},
            )


class DockerDefinitionTests(unittest.TestCase):
    """Guard security, mount, architecture, and entrypoint contracts."""

    def test_dockerfile_uses_nonroot_runtime_and_configurable_mount_roots(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER appuser", dockerfile)
        self.assertIn("/app/.venv/julia_env", dockerfile)
        self.assertIn("libexpat1", dockerfile)
        self.assertIn("PYSR_MODEL_ROOT=/inputs/models", dockerfile)
        self.assertIn("PYSR_RESULTS_ROOT=/results", dockerfile)
        self.assertIn("PYSR_RUNS_ROOT=/runs", dockerfile)
        self.assertIn("JULIA_DEPOT_PATH=/cache/julia:/opt/julia-depot", dockerfile)
        self.assertIn('ENTRYPOINT ["/app/.venv/bin/python"]', dockerfile)
        self.assertIn('"${TARGETARCH}" != "amd64"', dockerfile)

    def test_build_context_excludes_science_and_runtime_artifacts(self) -> None:
        entries = {
            line.strip()
            for line in (PROJECT_ROOT / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        self.assertTrue(
            {".git", ".venv", "results", "runs", "project_backups"} <= entries
        )

    def test_resource_smoke_cli_does_not_initialize_scientific_runtime(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output):
            container_smoke_main(["--resources-only"])
        payload = json.loads(output.getvalue())
        self.assertIn("resources", payload)
        self.assertIn("thread_environment", payload)
        self.assertNotIn("versions", payload)


if __name__ == "__main__":
    unittest.main()
