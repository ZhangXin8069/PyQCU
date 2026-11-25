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


def test_complex_vdot_does_not_inherit_the_platform_vdot(monkeypatch):
    """A complex inner product must not be delegated to ``torch.vdot``.

    The cluster toolchain is torch 1.12.1, where ``torch.vdot`` on complex tensors
    returns a silent zero: ``[1+2j, 3-1j] . [2, 1+1j]`` is 0 instead of 4,
    measured in place.  ``pyqcu.cann.vdot`` is the single choke point every torch
    solver reaches through ``pyqcu.tools.vdot``, and a zero inner product raises
    nothing - BiCGStab reports "breakdown at iter 0: r_tilde orthogonal to r",
    which is how the defect actually surfaced (a Legacy-MG inner-convergence gate
    refused a good parent on a compute node).

    The platform ``torch.vdot`` is stubbed to reproduce 1.12.1's behaviour, so the
    regression stays visible on a modern torch where the real call happens to work.
    """
    import torch

    import pyqcu.cann as cann
    from pyqcu.torch.tools import _linalg

    a = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex64)
    b = torch.tensor([2 + 0j, 1 + 1j], dtype=torch.complex64)

    def silent_zero(input, other, *args, **kwargs):  # torch 1.12.1 behaviour
        return torch.zeros((), dtype=input.dtype)

    monkeypatch.setattr(torch, "vdot", silent_zero)
    assert complex(cann.vdot(a, b)) == 4 + 0j
    assert complex(_linalg.vdot(a, b)) == 4 + 0j
    # A real inner product keeps the platform path, so this asserts the branch.
    real = torch.tensor([1.0, 2.0])
    assert complex(cann.vdot(real, real)) == 0 + 0j
