"""Canonical backend namespace and compatibility-facade contract."""

from __future__ import annotations

from pathlib import Path

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
    return Path(path).resolve().is_relative_to(REPO_ROOT / relative)


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
