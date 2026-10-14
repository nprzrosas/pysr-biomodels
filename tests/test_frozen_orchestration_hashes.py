"""Regression guards for scientific namespaces frozen before orchestration."""

import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
EXACT_CONTENT_HASH = "3289165db06136f6b7cecc412c0918d802de591aa626990ef9f96918b56e30f2"
SAVGOL_0079_CONTENT_HASH = (
    "be4377c2ea16f6524f09bb1c5b99358fbdd23861db15c43ce452afeb627ba8d1"
)


def sha256sum_inventory(paths: list[Path]) -> str:
    """Reproduce the frozen sha256sum path/content inventory hash."""
    lines = []
    for path in sorted(paths):
        content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"{content_digest}  {relative_path}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


class FrozenOrchestrationHashTests(unittest.TestCase):
    """Ensure infrastructure tests cannot alter frozen scientific outputs."""

    def test_all_exact_namespaces_remain_frozen(self) -> None:
        paths = [
            path
            for path in RESULTS_ROOT.rglob("*")
            if path.is_file() and "generic_exact_experiment" in path.parts
        ]
        self.assertEqual(len(paths), 2536)
        self.assertEqual(sha256sum_inventory(paths), EXACT_CONTENT_HASH)

    def test_savgol_0079_namespace_remains_frozen(self) -> None:
        root = RESULTS_ROOT / "BIOMD0000000079" / "generic_savgol_experiment"
        paths = [path for path in root.rglob("*") if path.is_file()]
        self.assertEqual(len(paths), 118)
        self.assertEqual(sha256sum_inventory(paths), SAVGOL_0079_CONTENT_HASH)

    def test_savgol_0156_has_not_been_created(self) -> None:
        self.assertFalse(
            (RESULTS_ROOT / "BIOMD0000000156" / "generic_savgol_experiment").exists()
        )


if __name__ == "__main__":
    unittest.main()
