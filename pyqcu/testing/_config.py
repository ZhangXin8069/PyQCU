"""Shared test configuration for PyQCU correctness checks."""

from __future__ import annotations


# All backend correctness gates use the same non-hypercubic test volume.
# Larger or hypercubic volumes remain valid only for performance/capacity runs.
CORRECTNESS_LATTICE: tuple[int, int, int, int] = (8, 8, 8, 16)
