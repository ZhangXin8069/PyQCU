"""Compatibility import for the backend-neutral MultiGrid depth warning."""

from pyqcu._multigrid_warnings import (
    MultigridDepthWarning,
    warn_excessive_multigrid_levels,
)

__all__ = ["MultigridDepthWarning", "warn_excessive_multigrid_levels"]
