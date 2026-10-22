"""Public naming and lifecycle contract for the unified multigrid facade."""

from __future__ import annotations

import importlib

import pytest
import torch

from pyqcu import solver
from pyqcu.solver import Multigrid
from pyqcu.solver.multigrid import (
    LegacyMultigrid as LegacyFromPackage,
    StrictMultigrid as StrictFromPackage,
)
from pyqcu.torch.solver import LegacyMultigrid, StrictMultigrid
from pyqcu.torch.solver import _legacy_multigrid as legacy_module
from pyqcu.kupl.multigrid import legacy_multigrid as kupl_legacy_module
from pyqcu.kupl.multigrid import strict_multigrid as kupl_strict_module
from pyqcu.cuda.multigrid import legacy_multigrid as cuda_legacy_module


def test_multigrid_package_is_the_canonical_hierarchy_namespace():
    assert LegacyFromPackage is LegacyMultigrid
    assert StrictFromPackage is StrictMultigrid

    from pyqcu.cuda.multigrid import LegacyMultigrid as CudaLegacy
    from pyqcu.cuda.multigrid import StrictMultigrid as CudaStrict
    from pyqcu.kupl.multigrid import LegacyMultigrid as KuplLegacy
    from pyqcu.kupl.multigrid import StrictMultigrid as KuplStrict

    assert (CudaLegacy.variant, CudaLegacy.backend) == ("legacy", "cuda")
    assert (CudaStrict.variant, CudaStrict.backend) == ("strict", "cuda")
    assert (KuplLegacy.variant, KuplLegacy.backend) == ("legacy", "kupl")
    assert (KuplStrict.variant, KuplStrict.backend) == ("strict", "kupl")
    assert not issubclass(CudaLegacy, LegacyMultigrid)
    public_mg_names = {
        name for name in dir(solver)
        if "multigrid" in name.lower() and not name.startswith("_")
    }
    assert all("quda" not in name.lower() for name in public_mg_names)


def test_multigrid_selector_defaults_to_legacy_and_validates_names(monkeypatch):
    module = importlib.import_module("pyqcu.solver.multigrid")

    class ProbeLegacy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        module, "_implementation_type",
        lambda backend, variant: ProbeLegacy)
    wrapper = Multigrid()
    assert Multigrid.DEFAULT_VARIANT == "legacy"
    assert (wrapper.variant, wrapper.backend) == ("legacy", "torch")
    assert wrapper.implementation.kwargs == {}

    with pytest.raises(ValueError, match="variant"):
        Multigrid(variant="unknown")
    with pytest.raises(ValueError, match="backend"):
        Multigrid(variant="strict", backend="unknown")


def test_multigrid_selector_builds_torch_strict_hierarchy():
    shape = (4, 4, 4, 4)
    diagonal = torch.eye(12, dtype=torch.complex64).reshape(
        12, 12, 1, 1, 1, 1).expand(12, 12, *shape).clone()
    null = torch.randn(2, 4, 3, *shape, dtype=torch.complex64)
    wrapper = Multigrid(
        variant="strict",
        backend="torch",
        fine_matvec=lambda value: value,
        fine_diagonal=diagonal,
        fine_adjoint=lambda value: value,
        lat_size=shape,
        null_vectors=[null],
        dof_list=[12, 4],
        block_size=(2, 2, 2, 2),
        max_level=2,
        setup_iters=0,
        verbose=False,
    )
    assert isinstance(wrapper.implementation, StrictMultigrid)
    assert (wrapper.variant, wrapper.backend) == ("strict", "torch")


def test_torch_legacy_lifecycle_uses_setup_and_close_aliases():
    class ProbeLegacy(LegacyMultigrid):
        def __init__(self):
            self.calls = []

        def init(self):
            self.calls.append("init")
            return self

        def end(self):
            self.calls.append("end")

    with ProbeLegacy() as mg:
        assert isinstance(mg, LegacyMultigrid)
    assert mg.calls == ["init", "end"]


def test_torch_legacy_init_builds_every_configured_level(monkeypatch):
    shape = (4, 4, 4, 4)
    U = torch.zeros(3, 3, 4, *shape, dtype=torch.complex64)
    for color in range(3):
        U[color, color] = 1.0
    clover = torch.zeros(4, 3, 4, 3, *shape, dtype=torch.complex64)
    mg = LegacyMultigrid(
        [torch.complex64, torch.complex64],
        [torch.device("cpu"), torch.device("cpu")],
        U,
        clover,
        kappa=torch.tensor([0.125]),
        min_size=2,
        max_level=2,
        dof_list=[12, 2],
        mg_grid_size=[2, 2, 2, 2],
        verbose=False,
    )
    monkeypatch.setattr(
        legacy_module.tools, "give_null_vecs",
        lambda *, null_vecs, **kwargs: null_vecs)
    monkeypatch.setattr(
        legacy_module.tools, "local_orthogonalize",
        lambda *, null_vecs, **kwargs: null_vecs)
    monkeypatch.setattr(
        legacy_module.dslash, "operator",
        lambda **kwargs: object())
    with mg:
        assert len(mg.op_list) == 2
        assert len(mg.b_list) == 2
        assert len(mg.lonv_list) == 1


def test_kupl_legacy_lifecycle_delegates_to_canonical_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kupl_legacy_module.qcu, "applyInitQcu",
        lambda *args: calls.append("init"))
    monkeypatch.setattr(
        kupl_legacy_module.qcu, "applyEndQcu",
        lambda *args: calls.append("end"))
    monkeypatch.setattr(
        kupl_legacy_module.qcu, "applyCloverMultigridQcu",
        lambda *args: calls.append("solve"))
    monkeypatch.setattr(
        kupl_legacy_module.qcu, "verifyCloverMultigridQcu",
        lambda *args: 0)
    state = (
        torch.zeros(58, dtype=torch.int32),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros(100, dtype=torch.int64),
        torch.zeros(1, dtype=torch.complex64),
    )
    rhs = torch.randn(12, 4, 4, 4, 4, dtype=torch.complex64)
    mg = kupl_legacy_module.LegacyMultigrid(
        *state, torch.zeros_like(rhs), torch.zeros_like(rhs),
        torch.zeros_like(rhs), torch.zeros_like(rhs))
    with mg:
        result = mg.solve(rhs)
    assert result.shape == rhs.shape
    assert calls == ["init", "solve", "end"]
    assert mg.closed


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 7,
    reason="CUDA legacy lifecycle probe requires a PyTorch-supported device",
)
def test_cuda_legacy_lifecycle_delegates_to_monolithic_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cuda_legacy_module.qcu, "applyInitQcu",
        lambda *args: calls.append("init"))
    monkeypatch.setattr(
        cuda_legacy_module.qcu, "applyEndQcu",
        lambda *args: calls.append("end"))
    monkeypatch.setattr(
        cuda_legacy_module.qcu, "applyCloverMultigridQcu",
        lambda *args: calls.append("solve"))
    state = (
        torch.zeros(58, dtype=torch.int32),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros(100, dtype=torch.int64),
    )
    asset = torch.zeros(1, dtype=torch.complex64, device="cuda")
    rhs = torch.zeros(12, 4, 4, 4, 4, dtype=torch.complex64, device="cuda")
    mg = cuda_legacy_module.LegacyMultigrid(
        *state, *((asset,) * 5))
    with mg:
        result = mg.solve(rhs)
    assert result.shape == rhs.shape
    assert calls == ["init", "solve", "end"]
    assert mg.closed


def test_kupl_strict_lifecycle_delegates_to_canonical_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kupl_strict_module.qcu, "applyInitQcu",
        lambda *args: calls.append("init"))
    monkeypatch.setattr(
        kupl_strict_module.qcu, "applyEndQcu",
        lambda *args: calls.append("end"))
    monkeypatch.setattr(
        kupl_strict_module.qcu, "applyMultigridStrictInitQcu",
        lambda *args: (calls.append("strict-init") or 0))
    monkeypatch.setattr(
        kupl_strict_module.qcu, "applyMultigridStrictEndQcu",
        lambda *args: (calls.append("strict-end") or 0))

    def fake_solve(*args):
        calls.append("solve")
        args[-4][0] = 1
        args[-3][0] = 1
        args[-2][0] = 0.0
        args[-1][0] = 256
        return 0

    monkeypatch.setattr(
        kupl_strict_module.qcu, "applyMultigridStrictFgmresQcu", fake_solve)
    state = (
        torch.zeros(58, dtype=torch.int32),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros(100, dtype=torch.int64),
        torch.zeros(1, dtype=torch.complex64),
    )
    dims = (4, 4, 4, 4)
    source = torch.randn(12, *dims, dtype=torch.complex64)
    mg = kupl_strict_module.StrictMultigrid(
        *state, torch.zeros_like(state[3]), torch.zeros_like(state[3]),
        torch.zeros_like(state[3]), torch.zeros_like(state[3]),
        torch.zeros_like(state[3]), dims, (2, 2, 2, 2))
    with mg:
        result = mg.solve(source)
    assert result.shape == (2, 4, 3, 4, 4, 4, 2)
    assert calls == ["init", "strict-init", "solve", "strict-end", "end"]
    assert mg.closed
