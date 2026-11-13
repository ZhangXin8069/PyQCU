"""Contract tests for the shared Legacy-MG setup helper."""

from __future__ import annotations

import numpy as np

from pyqcu.kupl import qcu
from pyqcu.testing.simple.kupl_multigrid import build_coarse_assets


def test_reused_clovers_keep_odd_target_parity(monkeypatch) -> None:
    lattice = (4, 4, 4, 4)
    fine_odd = (4, 4, 4, 2)
    coarse = (2, 2, 2, 2)
    mg_grid = [2, 2, 2, 1]
    dtype = np.dtype(np.complex64)
    block = tuple(fine_odd[i] // coarse[i] for i in range(4))
    parity_shapes_seen: list[int] = []

    monkeypatch.setattr(qcu, "applyInitQcu", lambda *args: None)
    monkeypatch.setattr(qcu, "applyEndQcu", lambda *args: None)

    def build_null(null_arg, *args):
        E = int(args[7])
        e = int(args[8])
        Xc, Yc, Zc, Tc = (int(value) for value in args[13:17])
        null_arg.array.fill(0)
        view = null_arg.array.reshape(
            E, e, Xc, block[0], Yc, block[1], Zc, block[2], Tc, block[3])
        block_sites = int(np.prod(block))
        for component in range(E):
            view[component, component, ...] = 1.0 / np.sqrt(block_sites)
        return 0

    def build_stencil(sitting_arg, hop_nn_arg, hop_diag_arg, *args):
        params_arg = args[7]
        parity_shapes_seen.append(int(params_arg.array[16]))
        sitting_arg.array.fill(1)
        hop_nn_arg.array.fill(1)
        hop_diag_arg.array.fill(1)
        return 0

    monkeypatch.setattr(qcu, "applyMultigridBuildNullVectorsQcu", build_null)
    monkeypatch.setattr(
        qcu, "applyMultigridBuildStencilLocalQcu", build_stencil)

    onsite_shape = (4, 3, 4, 3, *fine_odd)
    clovers = tuple(
        np.zeros(onsite_shape, dtype=dtype) for _ in range(4))
    build_coarse_assets(
        np.zeros((3, 3, 4, *lattice), dtype=dtype),
        lattice,
        0.05,
        1.0 / (2.0 * 0.05 + 8.0),
        1.0,
        dtype,
        2,
        [12, 2],
        mg_grid,
        1,
        42,
        device_name="cpu",
        validate_galerkin=False,
        gauge_parity=np.zeros(
            (2, 3, 3, 4, *fine_odd), dtype=dtype),
        clover_assets=clovers,
    )

    assert parity_shapes_seen == [1]
