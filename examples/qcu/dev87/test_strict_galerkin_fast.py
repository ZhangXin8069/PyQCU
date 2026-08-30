"""CPU equivalence test for batched/local strict Galerkin construction."""

from __future__ import annotations

from math import ceil, prod
from time import perf_counter

import pytest
import torch

from pyqcu.solver import QudaCoarseOperator, QudaMatPCOperator, QudaTransfer
from pyqcu.tools._strict_galerkin import (
    STRICT_GALERKIN_SCHEMA,
    build_strict_galerkin,
)


DTYPE = torch.complex64
FINE_SHAPE = (4, 4, 4, 4)
BLOCK = (1, 2, 2, 2)


def _nearest_operator(seed=20260830):
    generator = torch.Generator().manual_seed(seed)
    dof = 4
    identity = torch.eye(dof, dtype=DTYPE).reshape(
        dof, dof, 1, 1, 1, 1)
    diagonal = (2.0 * identity.expand(dof, dof, *FINE_SHAPE).clone() +
                0.01 * torch.randn(
                    dof, dof, *FINE_SHAPE, dtype=DTYPE,
                    generator=generator))
    forward = [0.01 * torch.randn(
        dof, dof, *FINE_SHAPE, dtype=DTYPE, generator=generator)
        for _ in range(4)]
    backward = [0.01 * torch.randn(
        dof, dof, *FINE_SHAPE, dtype=DTYPE, generator=generator)
        for _ in range(4)]

    def kernel(value):
        out = torch.einsum("ijxyzt,bjxyzt->bixyzt", diagonal, value)
        for dim in range(4):
            out = out + torch.einsum(
                "ijxyzt,bjxyzt->bixyzt", forward[dim],
                torch.roll(value, shifts=-1, dims=2 + dim))
            out = out + torch.einsum(
                "ijxyzt,bjxyzt->bixyzt", backward[dim],
                torch.roll(value, shifts=1, dims=2 + dim))
        return out

    return kernel


def test_strict_galerkin_fast_matches_columns_assets_and_matpc():
    torch.manual_seed(20260831)
    null_vectors = torch.randn(1, 4, 1, *FINE_SHAPE, dtype=DTYPE)
    transfer = QudaTransfer(
        null_vectors, FINE_SHAPE, fine_spin=4, fine_color=1,
        coarse_spin=2, block_size=BLOCK, verbose=False)
    assert transfer.coarse_spin == 2
    assert transfer.coarse_dof == 2 * transfer.nvec == 2

    kernel = _nearest_operator()
    calls = {"batch": 0, "single": 0}

    def batch_matvec(value):
        calls["batch"] += 1
        return kernel(value)

    def single_matvec(value):
        calls["single"] += 1
        return kernel(value.unsqueeze(0))[0]

    t0 = perf_counter()
    fast = build_strict_galerkin(
        transfer, batch_matvec, site_batch_size=4,
        retain_blocks=True, verbose=False)
    fast_seconds = perf_counter() - t0

    t0 = perf_counter()
    reference = QudaCoarseOperator(
        transfer, single_matvec, materialize=True, verbose=False)
    reference_seconds = perf_counter() - t0

    coarse_sites = prod(transfer.coarse_shape)
    assert calls["single"] == coarse_sites * transfer.coarse_dof
    assert calls["batch"] == ceil(coarse_sites / 4)
    assert fast.stats["scalar_columns"] == calls["single"]
    assert fast.stats["operator_calls"] == calls["batch"]
    assert fast.stats["worst_fine_support_leakage"] == 0.0

    assert fast.blocks is not None and reference.blocks is not None
    assert set(fast.blocks) == set(reference.blocks)
    for key in reference.blocks:
        assert torch.allclose(
            fast.blocks[key], reference.blocks[key], rtol=3e-5, atol=3e-6)

    reference_assets = reference.to_qcu_strict_assets()
    assert tuple(fast.raw_links.shape) == (
        2, 4, transfer.coarse_dof, transfer.coarse_dof,
        *transfer.coarse_shape)
    assert tuple(fast.preconditioned_links.shape) == tuple(fast.raw_links.shape)
    assert tuple(fast.onsite_pair.shape) == (
        2, transfer.coarse_dof, transfer.coarse_dof,
        *transfer.coarse_shape)
    assert torch.allclose(
        fast.raw_links, reference_assets["raw_links"],
        rtol=3e-5, atol=3e-6)
    assert torch.allclose(
        fast.preconditioned_links,
        reference_assets["preconditioned_links"],
        rtol=5e-5, atol=5e-6)
    assert torch.allclose(
        fast.onsite_pair, reference_assets["onsite_pair"],
        rtol=5e-5, atol=5e-6)

    trial = torch.randn(
        transfer.coarse_dof, *transfer.coarse_shape, dtype=DTYPE)
    exact = transfer.restrict(kernel(transfer.prolong(trial).unsqueeze(0))[0])
    assert torch.allclose(fast.apply_raw(trial), exact, rtol=3e-5, atol=3e-6)

    # Installing into a lazy reference object exercises the minimal future
    # integration boundary without modifying the solver module.
    installed = QudaCoarseOperator(transfer, single_matvec, materialize=False)
    fast.install(installed)
    assert torch.allclose(installed.apply(trial), exact, rtol=3e-5, atol=3e-6)
    compact = torch.randn(
        transfer.coarse_dof, prod(transfer.coarse_shape) // 2,
        dtype=DTYPE)
    matpc_reference = QudaMatPCOperator(installed, parity=1).apply(compact)
    assert torch.allclose(
        fast.apply_matpc(compact, parity=1), matpc_reference,
        rtol=5e-5, atol=5e-6)

    # Parity is a view of P/R and MATPC only: the constructed assets retain
    # the full 4x2x2x2 coarse geometry and coarse_spin=2.
    parity_prolong = transfer.prolong_parity(trial, parity=1)
    parity_restrict = transfer.restrict_parity(parity_prolong, parity=1)
    assert tuple(parity_restrict.shape) == (
        transfer.coarse_dof, *transfer.coarse_shape)
    assert tuple(fast.X.shape[-4:]) == transfer.coarse_shape

    payload = fast.cache_payload(transfer.to_qcu_blocked())
    assert payload["schema"]["name"] == STRICT_GALERKIN_SCHEMA
    assert payload["schema"]["coarse_spin"] == 2
    assert "target_parity" not in payload["schema"]
    assert payload["schema"]["parity_scope"].startswith("R/P views")

    # A diagonal two-block displacement is outside strict X/Y support.  The
    # strong fine-support guard must reject it before assets are produced.
    def wider_batch(value):
        return kernel(value) + 0.1 * torch.roll(
            value, shifts=(2, 2), dims=(2, 3))

    with pytest.raises(ValueError, match="nearest-neighbour"):
        build_strict_galerkin(
            transfer, wider_batch, site_batch_size=4,
            include_raw_links=False, retain_blocks=False)

    print(
        "PYQCU::TOOLS::STRICT_GALERKIN:\n "
        f"fast={fast_seconds:.3f}s/{calls['batch']} calls, "
        f"column_ref={reference_seconds:.3f}s/{calls['single']} calls")
