"""Fast equivalence gates for the vectorized strict fine-operator batch."""

from __future__ import annotations

import torch

from pyqcu import dslash
from pyqcu.solver import QudaStrictMultigrid


def _identity_gauge(shape=(2, 2, 2, 2), dtype=torch.complex64):
    gauge = torch.zeros((3, 3, 4, *shape), dtype=dtype)
    diagonal = torch.arange(3)
    gauge[diagonal, diagonal] = 1
    return gauge


def test_fine_wilson_clover_batch_matches_scalar_stack():
    torch.manual_seed(8701)
    shape = (2, 2, 2, 2)
    gauge = _identity_gauge(shape)
    clover = 0.01 * torch.randn(
        (4, 3, 4, 3, *shape), dtype=torch.complex64)
    op = dslash.operator(
        U=gauge, clover_term=clover,
        kappa=torch.tensor([0.12]), verbose=False)
    source = torch.randn((3, 4, 3, *shape), dtype=torch.complex64)

    expected = torch.stack([op.matvec(item) for item in source])
    actual = op.matvec_batch(source)

    torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_fine_batch_preserves_scalar_mixed_precision_contract():
    torch.manual_seed(8702)
    shape = (2, 2, 2, 2)
    gauge = _identity_gauge(shape, dtype=torch.complex64)
    clover = 0.01 * torch.randn(
        (4, 3, 4, 3, *shape), dtype=torch.complex64)
    op = dslash.operator(
        U=gauge, clover_term=clover,
        kappa=torch.tensor([0.12]), verbose=False)
    source = torch.randn((2, 12, *shape), dtype=torch.complex128)

    expected = torch.stack([op.matvec(item) for item in source])
    actual = op.matvec_batch(source)

    assert actual.dtype == source.dtype
    assert actual.device == source.device
    torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_strict_hierarchy_auto_wires_vectorized_fine_batch():
    # strict level-1 checkerboarding requires every coarse extent to stay even.
    shape = (4, 4, 4, 4)
    gauge = _identity_gauge(shape)
    hierarchy = QudaStrictMultigrid(
        U=gauge,
        kappa=torch.tensor([0.12]),
        null_vectors=[torch.ones((1, 4, 3, *shape), dtype=torch.complex64)],
        dof_list=[12, 2],
        block_size=(2, 2, 2, 2),
        max_level=2,
        setup_method="random",
        setup_iters=0,
        materialize_coarse=True,
        verbose=False,
    )

    wired = hierarchy._fine._batch_matvec
    assert wired.__self__ is hierarchy.fine_dslash
    assert wired.__func__ is hierarchy.fine_dslash.matvec_batch.__func__
