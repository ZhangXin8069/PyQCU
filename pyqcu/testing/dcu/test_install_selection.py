#!/usr/bin/env python3
"""CUDA-free contracts for the DCU build-selection path."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqcu._build_backend import normalize_build_backend  # noqa: E402


class DcuInstallSelectionTests(unittest.TestCase):
    def test_backend_normalization_accepts_dcu(self) -> None:
        self.assertEqual(normalize_build_backend(" DCU "), "dcu")
        self.assertEqual(normalize_build_backend(None), "cuda")
        with self.assertRaisesRegex(ValueError, "cuda, dcu, kupl"):
            normalize_build_backend("rocm")

    def test_install_help_lists_dcu(self) -> None:
        completed = subprocess.run(
            ["bash", str(ROOT / "install.sh"), "help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("dcu", output)
        self.assertIn("QCU_DCU_CUDA_COMPILER", output)

    def test_make_dcu_check_uses_explicit_compiler(self) -> None:
        env = os.environ.copy()
        env["QCU_DCU_CUDA_COMPILER"] = sys.executable
        completed = subprocess.run(
            ["bash", str(ROOT / "cpp/cuda/qcu/make-dcu.sh"), "--check"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("PYQCU DCU backend: ready", output)
        self.assertIn(sys.executable, output)

    def test_make_dcu_rejects_missing_explicit_compiler(self) -> None:
        env = os.environ.copy()
        env["QCU_DCU_CUDA_COMPILER"] = str(ROOT / "missing-dcu-compiler")
        env.pop("ROCM_PATH", None)
        completed = subprocess.run(
            ["bash", str(ROOT / "cpp/cuda/qcu/make-dcu.sh"), "--check"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 2, output)
        self.assertIn("no DTK nvcc-compatible compiler found", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
