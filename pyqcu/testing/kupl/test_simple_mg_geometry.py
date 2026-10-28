"""Geometry regression for the simple KUPL two-level MG builder."""

import numpy as np
import torch

from pyqcu import tools
from pyqcu.testing.simple.kupl_multigrid import (
    _rank_geometry, load_coarse_assets, write_coarse_assets,
)


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


def test_local_orthogonalize_cgs2_is_orthonormal_and_preserves_span() -> None:
    lattice = (4, 4, 4, 4)
    coarse = (2, 2, 2, 2)
    E, e = 6, 12
    generator = torch.Generator().manual_seed(17)
    null_vectors = torch.randn(
        (E, e, *lattice), dtype=torch.complex64, generator=generator)

    localized = tools.local_orthogonalize(
        null_vectors, coarse_lat_size=list(coarse), verbose=False)
    X, Y, Z, T = coarse
    x, y, z, t = (lattice[index] // coarse[index] for index in range(4))
    columns = localized.reshape(
        E, e, X, x, Y, y, Z, z, T, t).permute(
            2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                X * Y * Z * T, E, e * x * y * z * t).transpose(-2, -1)
    gram = columns.conj().transpose(-2, -1) @ columns
    identity = torch.eye(E, dtype=gram.dtype).expand_as(gram)
    torch.testing.assert_close(gram, identity, rtol=2.0e-5, atol=2.0e-5)

    original = null_vectors.reshape(
        E, e, X, x, Y, y, Z, z, T, t).permute(
            2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
                X * Y * Z * T, E, e * x * y * z * t).transpose(-2, -1)
    reference, _ = torch.linalg.qr(original)
    expected_projector = reference @ reference.conj().transpose(-2, -1)
    actual_projector = columns @ columns.conj().transpose(-2, -1)
    torch.testing.assert_close(
        actual_projector, expected_projector, rtol=2.0e-4, atol=2.0e-4)


def test_coarse_cache_loader_returns_one_sequence_per_level(tmp_path) -> None:
    arrays = tuple(
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
        for _ in range(4)
    )
    path = tmp_path / "coarse.h5"
    write_coarse_assets(path, *arrays)

    loaded = load_coarse_assets(path, arrays[0].dtype)

    assert len(loaded) == 4
    for levels, expected in zip(loaded, arrays, strict=True):
        assert isinstance(levels, tuple)
        assert len(levels) == 1
        assert levels[0].shape == expected.shape


def test_coarse_cache_loader_slices_interleaved_null_vectors(tmp_path) -> None:
    lonv = torch.arange(
        2 * 3 * 4 * 2 * 5 * 2 * 6 * 2 * 7 * 2,
        dtype=torch.float32).reshape(2, 3, 4, 2, 5, 2, 6, 2, 7, 2).numpy()
    hnn = torch.arange(2 * 1 * 3 * 3 * 4 * 5 * 6 * 7,
                       dtype=torch.float32).reshape(2, 1, 3, 3, 4, 5, 6, 7).numpy()
    hdg = torch.arange(2 * 1 * 2 * 3 * 3 * 4 * 5 * 6 * 7,
                       dtype=torch.float32).reshape(
                           2, 1, 2, 3, 3, 4, 5, 6, 7).numpy()
    sitting = torch.arange(3 * 3 * 4 * 5 * 6 * 7,
                           dtype=torch.float32).reshape(3, 3, 4, 5, 6, 7).numpy()
    path = tmp_path / "coarse_slice.h5"
    write_coarse_assets(path, lonv, hnn, hdg, sitting)

    starts = (1, 2, 1, 2)
    local = (2, 2, 3, 2)
    null_vectors, hop_nn, hop_diag, onsite = load_coarse_assets(
        path, lonv.dtype, starts=starts, local_shape=local)

    expected_null = lonv[
        :, :, 1:3, :, 2:4, :, 1:4, :, 2:4, :]
    np.testing.assert_array_equal(null_vectors[0], expected_null)
    np.testing.assert_array_equal(hop_nn[0], hnn[..., 1:3, 2:4, 1:4, 2:4])
    np.testing.assert_array_equal(hop_diag[0], hdg[..., 1:3, 2:4, 1:4, 2:4])
    np.testing.assert_array_equal(onsite[0], sitting[..., 1:3, 2:4, 1:4, 2:4])


def test_rank_geometry_uses_c_order_process_coordinates() -> None:
    coordinate, local, starts = _rank_geometry(
        (24, 24, 24, 64), (2, 2, 1, 1), 1)

    assert coordinate == (0, 1, 0, 0)
    assert local == (12, 12, 24, 64)
    assert starts == (0, 12, 0, 0)
