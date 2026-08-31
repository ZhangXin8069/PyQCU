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
    build_strict_galerkin_colored,
    strict_galerkin_colored_memory_model,
    strict_galerkin_memory_model,
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
    calls = {"batch": 0, "colored": 0, "single": 0}

    def batch_matvec(value):
        calls["batch"] += 1
        return kernel(value)

    def single_matvec(value):
        calls["single"] += 1
        return kernel(value.unsqueeze(0))[0]

    def colored_matvec(value):
        calls["colored"] += 1
        return kernel(value)

    t0 = perf_counter()
    fast = build_strict_galerkin(
        transfer, batch_matvec, site_batch_size=4,
        retain_blocks=True, verbose=False)
    fast_seconds = perf_counter() - t0

    colored = build_strict_galerkin_colored(
        transfer, colored_matvec, column_batch_size=2,
        projection_site_batch_size=4, check_fine_support=False,
        retain_blocks=True, verbose=False)

    t0 = perf_counter()
    reference = QudaCoarseOperator(
        transfer, single_matvec, materialize=True, verbose=False)
    reference_seconds = perf_counter() - t0

    coarse_sites = prod(transfer.coarse_shape)
    assert calls["single"] == coarse_sites * transfer.coarse_dof
    assert calls["batch"] == ceil(coarse_sites / 4)
    assert calls["colored"] == colored.stats["operator_calls"]
    assert calls["colored"] < calls["single"]
    assert fast.stats["scalar_columns"] == calls["single"]
    assert fast.stats["operator_calls"] == calls["batch"]
    assert fast.stats["worst_fine_support_leakage"] == 0.0
    assert not colored.stats["support_checked"]
    assert (colored.stats["memory"]["workspace_upper_bytes"] <
            fast.stats["memory"]["workspace_upper_bytes"])

    assert fast.blocks is not None and reference.blocks is not None
    assert set(fast.blocks) == set(reference.blocks)
    for key in reference.blocks:
        assert torch.allclose(
            fast.blocks[key], reference.blocks[key], rtol=3e-5, atol=3e-6)
        assert colored.blocks is not None
        assert torch.allclose(
            colored.blocks[key], reference.blocks[key],
            rtol=3e-5, atol=3e-6)

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
        colored.raw_links, reference_assets["raw_links"],
        rtol=3e-5, atol=3e-6)
    assert torch.allclose(
        fast.preconditioned_links,
        reference_assets["preconditioned_links"],
        rtol=5e-5, atol=5e-6)
    assert torch.allclose(
        colored.preconditioned_links,
        reference_assets["preconditioned_links"],
        rtol=5e-5, atol=5e-6)
    assert torch.allclose(
        fast.onsite_pair, reference_assets["onsite_pair"],
        rtol=5e-5, atol=5e-6)
    assert torch.allclose(
        colored.onsite_pair, reference_assets["onsite_pair"],
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


def test_colored_memory_model_bounds_large_e48_block4_geometry_workspace():
    """较大 E48/block4 几何必须选 colored path，不能分配 K×E 全场。"""
    common = {
        "coarse_dof": 48,
        "fine_dof": 12,
        "fine_shape": (16, 32, 32, 48),
        "coarse_shape": (4, 8, 8, 12),
        "block_size": (4, 4, 4, 4),
        "element_size": 8,
        "include_raw_links": False,
        "retain_blocks": False,
    }
    site = strict_galerkin_memory_model(
        **common, site_batch_size=4)
    colored = strict_galerkin_colored_memory_model(
        **common, column_batch_size=1,
        projection_site_batch_size=4)
    assert site["workspace_upper_bytes"] > 20 * (1 << 30)
    assert colored["workspace_upper_bytes"] < 512 * (1 << 20)
    assert colored["workspace_upper_bytes"] < site["workspace_upper_bytes"] // 40


def test_e24_formal_geometry_workspace_tradeoff_is_exact():
    common = {
        "coarse_dof": 24,
        "fine_dof": 12,
        "fine_shape": (16, 32, 32, 48),
        "coarse_shape": (8, 16, 16, 24),
        "block_size": (2, 2, 2, 2),
        "element_size": 8,
        "include_raw_links": False,
        "retain_blocks": False,
        "projection_site_batch_size": 4,
    }
    expected = {
        1: (306524928, 528),
        4: (1212681216, 132),
        8: (2420889600, 66),
        12: (3629097984, 44),
        24: (7253723136, 22),
    }
    for columns, (workspace, calls) in expected.items():
        model = strict_galerkin_colored_memory_model(
            **common, column_batch_size=columns)
        assert model["workspace_upper_bytes"] == workspace
        assert model["operator_calls"] == calls

    assert expected[1][0] <= 512 * (1 << 20)
    assert expected[4][0] > 512 * (1 << 20)
    assert expected[12][0] < 4 * (1 << 30)
    assert expected[24][0] > 6 * (1 << 30)
