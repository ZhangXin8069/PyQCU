"""Geometry regression for the simple KUPL two-level MG builder."""

import torch

from pyqcu import tools


def test_local_orthogonalize_uses_full_coarse_lattice() -> None:
    lattice = (8, 8, 8, 8)
    mg_grid = (2, 2, 2, 2)
    coarse = tuple(
        lattice[index] // mg_grid[index] for index in range(4))
    null_vectors = torch.randn((12, 12, *lattice), dtype=torch.complex64)

    localized = tools.local_orthogonalize(
        null_vectors, coarse_lat_size=list(coarse), verbose=False)

    assert localized.shape == (
        12, 12, coarse[0], 2, coarse[1], 2,
        coarse[2], 2, coarse[3], 2)
    coarse_vector = torch.randn(
        (12, *coarse), dtype=torch.complex64)
    fine_vector = tools.prolong(localized, coarse_vector)
    assert fine_vector.shape == (12, *lattice)
