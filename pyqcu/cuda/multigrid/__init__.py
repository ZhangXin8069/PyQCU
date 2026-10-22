"""Canonical CUDA multigrid namespace."""

from .legacy_multigrid import LegacyMultigrid
from .strict_multigrid import StrictMultigrid

__all__ = ["LegacyMultigrid", "StrictMultigrid"]
