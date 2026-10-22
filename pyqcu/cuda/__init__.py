# BUGFIX 2026-07-28 R3: pyqcu/cuda/__init__.py was missing, preventing the package
# from being discovered by find_packages() in non-editable (pip install) mode.
# Without this file, `from pyqcu.cuda import qcu, define` would raise ImportError
# because define.py is not included in the wheel and the package is undiscoverable.
#
# This file makes pyqcu.cuda a proper Python package. The public API is:
#   from pyqcu.cuda import qcu       # Cython bridge to libqcu.so
#   from pyqcu.cuda import define    # Parameter constants and dtype helpers
#   from pyqcu.cuda import schur_op  # CudaSchurOp (multi-thread-safe Schur op)
#   from pyqcu.cuda.multigrid import StrictMultigrid, LegacyMultigrid
#   from pyqcu.cuda._multi_gpu import MultiGpuMultigrid
#       TEST-ONLY, explicit opt-in. Production CUDA topology is one Python
#       process = one MPI rank = one GPU/DCU/GPGPU; importing this package does
#       not import, construct, or start the multi-GPU test driver.

_LAZY_EXPORTS = {}
__all__ = ["LegacyMultigrid", "QcuRuntime", "StrictMultigrid"]


def __getattr__(name):
    if name == "QcuRuntime":
        if name not in _LAZY_EXPORTS:
            from ._runtime import QcuRuntime
            _LAZY_EXPORTS[name] = QcuRuntime
        return _LAZY_EXPORTS[name]
    if name == "StrictMultigrid":
        if name not in _LAZY_EXPORTS:
            from .multigrid.strict_multigrid import StrictMultigrid
            _LAZY_EXPORTS[name] = StrictMultigrid
        return _LAZY_EXPORTS[name]
    if name == "LegacyMultigrid":
        if name not in _LAZY_EXPORTS:
            from .multigrid.legacy_multigrid import LegacyMultigrid
            _LAZY_EXPORTS[name] = LegacyMultigrid
        return _LAZY_EXPORTS[name]
    raise AttributeError(f"module 'pyqcu.cuda' has no attribute {name!r}")
from argparse import Namespace
Namespace.__module__ = "pyqcu.cuda"
