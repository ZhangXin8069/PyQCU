# BUGFIX 2026-07-28 R3: pyqcu/cuda/__init__.py was missing, preventing the package
# from being discovered by find_packages() in non-editable (pip install) mode.
# Without this file, `from pyqcu.cuda import qcu, define` would raise ImportError
# because define.py is not included in the wheel and the package is undiscoverable.
#
# This file makes pyqcu.cuda a proper Python package. The public API is:
#   from pyqcu.cuda import qcu       # Cython bridge to libqcu.so
#   from pyqcu.cuda import define    # Parameter constants and dtype helpers
#   from pyqcu.cuda import schur_op  # CudaSchurOp (multi-thread-safe Schur op)
#   from pyqcu.cuda import multi_gpu # MultiGpuMultigrid (one-thread-one-GPU driver)

_LAZY_EXPORTS = {}


def __getattr__(name):
    if name == "CudaStrictMultigridSolver":
        if name not in _LAZY_EXPORTS:
            from ._strict_multigrid import CudaStrictMultigridSolver
            _LAZY_EXPORTS[name] = CudaStrictMultigridSolver
        return _LAZY_EXPORTS[name]
    raise AttributeError(f"module 'pyqcu.cuda' has no attribute {name!r}")
from argparse import Namespace
Namespace.__module__ = "pyqcu.cuda"
