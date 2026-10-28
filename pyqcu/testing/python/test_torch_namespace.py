"""Canonical backend namespace and compatibility-facade contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pyqcu.dslash as compatibility_dslash
import pyqcu.lattice as common_lattice
import pyqcu.smear as compatibility_smear
import pyqcu.solver as compatibility_solver
import pyqcu.tools as compatibility_tools
import pyqcu.torch as torch_backend
from pyqcu.solver.multigrid import Multigrid
from pyqcu.torch import dslash as torch_dslash
from pyqcu.torch import smear as torch_smear
from pyqcu.torch import solver as torch_solver
from pyqcu.torch import tools as torch_tools


REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_under(path: str, relative: str) -> bool:
    try:
        Path(path).resolve().relative_to(REPO_ROOT / relative)
    except ValueError:
        return False
    return True


def test_torch_implementation_packages_live_under_pyqcu_torch():
    assert _is_under(torch_backend.__file__, "pyqcu/torch")
    assert _is_under(torch_dslash.__file__, "pyqcu/torch/dslash")
    assert _is_under(torch_smear.__file__, "pyqcu/torch/smear")
    assert _is_under(torch_solver.__file__, "pyqcu/torch/solver")
    assert _is_under(torch_tools.__file__, "pyqcu/torch/tools")


def test_common_namespaces_remain_at_top_level():
    assert _is_under(common_lattice.__file__, "pyqcu/lattice")
    assert _is_under(compatibility_tools.__file__, "pyqcu/tools")
    assert isinstance(Multigrid, type)


def test_compatibility_facades_export_the_torch_implementations():
    assert compatibility_dslash.operator is torch_dslash.operator
    assert compatibility_smear.stout_smear is torch_smear.stout_smear
    assert compatibility_solver.bistabcg is torch_solver.bistabcg
    assert compatibility_solver.StrictMultigrid is torch_solver.StrictMultigrid
    assert compatibility_tools.norm is torch_tools.norm
    assert compatibility_tools.vdot is torch_tools.vdot


def test_torch_solver_import_does_not_require_cuda_binding():
    code = """
import importlib.abc
import sys

class BlockCuda(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyqcu.cuda' or fullname.startswith('pyqcu.cuda.'):
            raise RuntimeError(f'blocked eager import: {fullname}')
        return None

sys.meta_path.insert(0, BlockCuda())
import pyqcu.torch.solver._legacy_multigrid as legacy
assert legacy.qcu._module is None
assert legacy.define._module is None
assert 'pyqcu.cuda.qcu' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
