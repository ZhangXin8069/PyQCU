"""Contracts for the backend-neutral PyQCU API."""

from __future__ import annotations

import math

import pytest
import torch

from pyqcu.api import (
    AssetManifest,
    BackendCapabilities,
    BiCGStabControl,
    CGControl,
    CGOperator,
    CoarsenConfig,
    CycleConfig,
    FGMRESControl,
    LinearOperator,
    LatticeGeometry,
    MRConfig,
    PeriodicCorrectionConfig,
    ResidualScope,
    SolveResult,
    SolveStatus,
    StrictMultigridConfig,
)
from pyqcu.solver.multigrid import Multigrid


class _FakeOperator:
    def __init__(self):
        self.backend = "test"
        self.geometry = LatticeGeometry(
            local_shape=(4, 4, 4, 8), dtype="c64")
        self.residual_scope = ResidualScope.FULL
        self.capabilities = {"adjoint": True, "matpc": True}

    def apply(self, value, *, parity=None, adjoint=False):
        return value

    def apply_adjoint(self, value, *, parity=None):
        return value

    def diagonal(self, parity=None):
        return None

    def apply_normal(self, value, *, parity=None):
        return value

    def apply_matpc(self, value, *, parity=None):
        return value


class _MetadataMultigrid:
    backend = "test"
    variant = "strict"
    residual_scope = ResidualScope.MATPC

    def __init__(self):
        self.calls = []
        self.converged = True
        self.outer_iterations = 7
        self.final_residual = 2.5e-8
        self.last_solve_seconds = 0.25
        self.fused_workspace_planned_bytes = 4096
        self.correction_count = 3

    def setup(self):
        self.calls.append("setup")
        return self

    def capabilities(self):
        return BackendCapabilities(
            backend="test", variant="strict", full_coarse=True, matpc=True)

    def solve(self, rhs, x0=None):
        self.calls.append(("solve", x0))
        return rhs + 1

    def close(self):
        self.calls.append("close")


class _MinimalMultigrid:
    backend = "test"
    variant = "legacy"

    def setup(self):
        return self

    def solve(self, rhs, x0=None):
        return rhs


def test_lattice_geometry_and_config_validate_public_contract():
    geometry = LatticeGeometry(
        local_shape=(2, 4, 6, 8), dtype="c64", parity=1)
    assert geometry.volume == 384
    assert geometry.local_sites == 192
    assert geometry.full_shape == (2, 4, 6, 8)

    control = BiCGStabControl(rtol=1e-8, max_iter=80)
    assert control.convergence_threshold(10.0) == pytest.approx(
        1e-7 + control.atol)

    fgmres = FGMRESControl(rtol=1e-8, max_iter=100, restart=20)
    assert fgmres.restart == 20
    assert CGControl(operator_kind=CGOperator.NORMAL).operator_kind is (
        CGOperator.NORMAL)

    config = StrictMultigridConfig(
        coarsen=CoarsenConfig(nvec=8, block_size=(2, 2, 2, 2), levels=2),
        smoother=MRConfig(steps=2),
        cycle=CycleConfig(kind="v", pre=2, post=1),
        outer=fgmres,
    )
    assert config.cycle.kind == "V"

    with pytest.raises(ValueError, match="block_size"):
        CoarsenConfig(nvec=8, block_size=(2, 0, 2, 2), levels=2)
    with pytest.raises(ValueError, match="rtol"):
        BiCGStabControl(rtol=math.inf)
    with pytest.raises(ValueError, match="restart"):
        FGMRESControl(restart=101, max_iter=100)
    with pytest.raises(ValueError, match="interval"):
        PeriodicCorrectionConfig(interval=0)

    capabilities = BackendCapabilities(
        backend="test",
        variant="strict",
        full_coarse=True,
        matpc=True,
        persistent_hierarchy=True,
        hpd_cg=True,
        supported_precisions=("c64", "c128"),
        max_levels=4,
    )
    assert capabilities.supports_precision("c64")
    assert not capabilities.supports_precision("c32")
    assert capabilities.require("matpc") is capabilities
    with pytest.raises(ValueError, match="compact_coarse"):
        capabilities.require("compact_coarse")
    with pytest.raises(ValueError, match="unknown capability"):
        capabilities.require("backend")


def test_solve_result_rejects_inconsistent_status_and_nonfinite_metrics():
    result = SolveResult(
        x=torch.zeros(1),
        backend="test",
        variant="strict",
        method="multigrid",
        converged=True,
        krylov_iterations=4,
        correction_count=1,
        cycle_count=1,
        final_residual=1e-8,
        residual_scope=ResidualScope.MATPC,
        matpc_true_residual=1e-8,
        diagnostics={"cache": "hit"},
    )
    assert result.status is SolveStatus.CONVERGED
    assert result.relative_true_residual is None
    assert result.full_true_residual is None
    with pytest.raises(TypeError):
        result.diagnostics["cache"] = "miss"

    with pytest.raises(ValueError, match="converged"):
        SolveResult(
            x=None,
            backend="test",
            variant="strict",
            method="multigrid",
            status=SolveStatus.CONVERGED,
            converged=False,
        )
    with pytest.raises(ValueError, match="final_residual"):
        SolveResult(
            x=None,
            backend="test",
            variant="strict",
            method="multigrid",
            final_residual=math.inf,
        )
    with pytest.raises(ValueError, match="residual_scope"):
        SolveResult(
            x=None,
            backend="test",
            variant="strict",
            method="multigrid",
            final_residual=1e-8,
        )


def test_asset_manifest_identity_is_order_independent_and_sensitive():
    common = dict(
        kind="strict_coarse_links",
        backend="test",
        variant="strict",
        precision="c64",
        lattice_shape=(4, 4, 4, 8),
        process_grid=(1, 1, 1, 1),
        block_size=(2, 2, 2, 2),
        level_count=2,
        algorithm_version="strict-v1",
    )
    first = AssetManifest(
        **common, metadata={"b": 2, "a": 1}, logical_bytes=1024)
    second = AssetManifest(
        **common, metadata={"a": 1, "b": 2}, logical_bytes=1024)
    changed = AssetManifest(
        **common, metadata={"a": 1, "b": 3}, logical_bytes=1024)
    assert first.identity == second.identity
    assert first.identity != changed.identity
    assert len(first.identity) == 64


def test_multigrid_solve_result_preserves_legacy_solve_return():
    wrapper = object.__new__(Multigrid)
    wrapper.variant = "strict"
    wrapper.backend = "test"
    wrapper._implementation = _MetadataMultigrid()
    rhs = torch.zeros(12, 4, 4, 4, 8, dtype=torch.complex64)

    raw = wrapper.solve(rhs)
    assert torch.equal(raw, rhs + 1)
    result = wrapper.solve_result(rhs)

    assert torch.equal(result.x, rhs + 1)
    assert result.backend == "test"
    assert result.variant == "strict"
    assert result.method == "multigrid"
    assert result.status is SolveStatus.CONVERGED
    assert result.converged is True
    assert result.krylov_iterations == 7
    assert result.correction_count == 3
    assert result.final_residual == pytest.approx(2.5e-8)
    assert result.matpc_true_residual == pytest.approx(2.5e-8)
    assert result.full_true_residual is None
    assert result.residual_scope is ResidualScope.MATPC
    assert result.workspace_bytes == 4096
    assert result.setup_seconds is not None
    assert result.solve_seconds is not None
    assert wrapper.implementation.calls[-2:] == [
        "setup", ("solve", None)]
    capabilities = wrapper.capabilities()
    assert capabilities.full_coarse
    assert capabilities.matpc


def test_multigrid_solve_result_does_not_fabricate_unknown_metrics():
    wrapper = object.__new__(Multigrid)
    wrapper.variant = "legacy"
    wrapper.backend = "test"
    wrapper._implementation = _MinimalMultigrid()
    rhs = torch.ones(1)
    result = wrapper.solve_result(rhs)
    assert torch.equal(result.x, rhs)
    assert result.status is SolveStatus.UNKNOWN
    assert result.converged is None
    assert result.krylov_iterations is None
    assert result.final_residual is None
    assert result.residual_scope is None
    capabilities = wrapper.capabilities()
    assert capabilities.backend == "test"
    assert not capabilities.full_coarse


def test_linear_operator_protocol_is_runtime_checkable():
    assert isinstance(_FakeOperator(), LinearOperator)


def test_strict_backends_declare_compact_matpc_residual_scope():
    from pyqcu.cuda.multigrid import StrictMultigrid as CudaStrict
    from pyqcu.kupl.multigrid import StrictMultigrid as KuplStrict
    from pyqcu.torch.solver import StrictMultigrid as TorchStrict

    assert CudaStrict.residual_scope == "matpc"
    assert KuplStrict.residual_scope == "matpc"
    assert TorchStrict.residual_scope == "matpc"
