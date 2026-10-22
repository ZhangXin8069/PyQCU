"""PyTorch/CANN implementation backend for PyQCU.

Common lattice conventions, MPI/HDF5 utilities, backend selection, CUDA and
KUPL remain in their top-level namespaces.  This package owns executable
Torch implementations of operators, solvers, smearing, and Torch-only tools.
"""

from __future__ import annotations

from importlib import import_module

_SUBPACKAGES = {"dslash", "smear", "solver", "tools"}
_LAZY_EXPORTS = {}


def __getattr__(name):
    if name in _SUBPACKAGES:
        if name not in _LAZY_EXPORTS:
            _LAZY_EXPORTS[name] = import_module(f"{__name__}.{name}")
        return _LAZY_EXPORTS[name]
    raise AttributeError(f"module 'pyqcu.torch' has no attribute {name!r}")


__all__ = ["dslash", "smear", "solver", "tools"]
