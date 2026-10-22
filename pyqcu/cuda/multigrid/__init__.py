"""Canonical CUDA multigrid namespace.

Legacy-MultiGrid is the QCU default and development/debug target;
Strict-MultiGrid is explicit comparison/reference/special-fallback only.
"""

from .legacy_multigrid import LegacyMultigrid
from .strict_multigrid import StrictMultigrid

__all__ = ["LegacyMultigrid", "StrictMultigrid"]
