"""Formal ``pyqcu.test`` port: shipped, stdlib-only, quick correctness gates.

``pyqcu.test`` is the **formal** cross-backend test port:

* it ships in the sdist and the wheel (``pyqcu/testing`` does not);
* importing it pulls in no backend - no ``torch``, ``numpy``, ``mpi4py``,
  ``pytest`` and no ``pyqcu.testing`` - so ``python -m pyqcu.test --list``
  works on a machine where only the package is installed;
* it answers "are the torch/KUPL/CUDA code paths still correct at the project
  correctness lattice", not "how fast are they".

``pyqcu/testing`` remains the informal, checkout-only tree for research,
performance, QUDA/HPCKit assets and historical data.  The two trees do not
import each other.

Public surface
--------------
:class:`Test`      selected cases plus the run configuration
:class:`Case`      one declarative case (runner string + ``expect`` gates)
:class:`Result`    one case's verdict, metrics and timing
:class:`TestReport` aggregate, JSON-safe report
:data:`QUICK_LATTICE` the ``--quick`` / tier-0 lattice, ``(8, 8, 8, 16)``
:data:`__version__` the distribution version (same as :mod:`pyqcu`)

Quick start::

    python -m pyqcu.test --list
    python -m pyqcu.test --quick
    python -m pyqcu.test --quick --json -
    python -m pyqcu.test.selfcheck
"""

from __future__ import annotations

from .. import __version__ as _PACKAGE_VERSION
from .common import Case, Result, Test, TestReport
from .policy import QUICK_LATTICE

__version__ = _PACKAGE_VERSION

__all__ = [
    "Test",
    "Case",
    "Result",
    "TestReport",
    "QUICK_LATTICE",
    "__version__",
]
