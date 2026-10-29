#!/usr/bin/env python3
"""Contracts for a CUDA-free KUPL installation on older PyTorch builds."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyqcu.cann as cann  # noqa: E402


class PureKuplInstallTests(unittest.TestCase):
    def test_real_dtype_does_not_require_torch_dtype_to_real(self) -> None:
        source = inspect.getsource(cann.real_dtype)
        calls = [
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_real"
        ]
        self.assertEqual(calls, [])
        self.assertIs(cann.real_dtype(torch.complex64), torch.float32)
        self.assertIs(cann.real_dtype(torch.complex128), torch.float64)
        self.assertIs(cann.real_dtype(torch.float32), torch.float32)

    def test_split_complex_creation_uses_portable_real_mapping(self) -> None:
        previous = cann.force_use_npu
        cann.force_use_npu = True
        try:
            value = cann.zeros(
                (2,), dtype=torch.complex64, device=torch.device("cpu"))
        finally:
            cann.force_use_npu = previous
        self.assertEqual(value.dtype, torch.complex64)
        self.assertEqual(value.tolist(), [0j, 0j])

    def test_setup_kupl_skips_cuda_native_and_cython_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            env = os.environ.copy()
            env["PYQCU_BUILD_BACKEND"] = "kupl"
            env["PYTHONPATH"] = (
                str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "setup.py"),
                    "build_ext",
                    "--build-temp", str(temp / "temp"),
                    "--build-lib", str(temp / "lib"),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("PYQCU build backend: kupl", output)
        self.assertNotIn("build-nv.sh", output)
        self.assertNotIn("Failed to find nvcc", output)

    def test_topology_cli_json_does_not_initialize_mpi(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "pyqcu/testing/kupl/kunpeng_runtime.py"),
                "inspect",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("MPI_Init", completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertIn("affinity", payload)
        self.assertIn("nodes", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
