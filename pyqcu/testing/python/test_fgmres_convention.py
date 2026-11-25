"""Pin the Arnoldi projection convention the KUPL Legacy-MG FGMRES fix depends on.

The KUPL Legacy-MG's first FGMRES(m=10) run failed because the Arnoldi
projection was written as ``image^H v_i`` instead of ``v_i^H w``: every
Hessenberg entry came out conjugated, the least-squares correction pointed the
wrong way, and the solve left the residual above the tolerance the C++ then
refuses to publish.  This module pins the convention from Python, where it can
be broken on purpose:

* ``pyqcu.tools.vdot(a, b)`` is the Hermitian inner product ``a^H b`` - it
  conjugates its FIRST argument, which is the slot the basis vector goes in.
* A numpy right-preconditioned FGMRES(m) reference built on that convention
  converges on a tiny Hermitian system; the same reference with the projection
  flipped to ``w^H v_i`` does not.
* The live Python solver (``pyqcu.torch.solver._gmres.fgmres``) loses
  convergence when ``pyqcu.tools.vdot`` returns the conjugated scalar, and
  recovers as soon as the flip is undone.

The tests are deliberately small (n = 8, one restart cycle) so the separation is
a numerical fact rather than a statistical one.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyqcu.cann as _torch
import pyqcu.tools as tools
from pyqcu.torch.solver._gmres import fgmres


# ---------------------------------------------------------------------------
# the tiny Hermitian system
# ---------------------------------------------------------------------------
def _hermitian_system(n: int = 8, seed: int = 5):
    """Return ``(A, b, M^-1)``: A Hermitian positive definite, M^-1 = diag(A)^-1.

    The off-diagonal phases are genuinely complex on purpose: a real-symmetric
    matrix would make ``conj`` a no-op and the convention test meaningless.
    """
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = dense @ dense.conj().T + 3.0 * np.eye(n)
    A = 0.5 * (A + A.conj().T)
    diagonal = np.diag(A).real
    rhs = np.random.default_rng(seed + 1)
    b = rhs.normal(size=n) + 1j * rhs.normal(size=n)
    return A, b, lambda value: value / diagonal


# ---------------------------------------------------------------------------
# a numpy right-preconditioned FGMRES(m) reference
# ---------------------------------------------------------------------------
def _givens(a: complex, b: complex) -> tuple[complex, complex]:
    """Unitary complex Givens pair ``(cos, sin)`` annihilating ``b``.

    Same convention as ``pyqcu.torch.solver._gmres._givens_rotation`` (and
    QUDA's FlexArnoldiProcedure): ``cos = a/r``, ``sin = b/r`` with
    ``r = hypot(|a|, |b|)``, applied as ``[conj(cos) conj(sin); -sin cos]``.
    """
    radius = float(np.hypot(abs(a), abs(b)))
    if radius == 0.0:
        return 1.0 + 0.0j, 0.0 + 0.0j
    return a / radius, b / radius


def _apply_rotation(H, g, cos, sin, j: int) -> None:
    for i in range(j):
        upper, lower = H[i, j], H[i + 1, j]
        H[i, j] = np.conj(cos[i]) * upper + np.conj(sin[i]) * lower
        H[i + 1, j] = -sin[i] * upper + cos[i] * lower
    upper, lower = H[j, j], H[j + 1, j]
    cos[j], sin[j] = _givens(upper, lower)
    H[j, j] = np.conj(cos[j]) * upper + np.conj(sin[j]) * lower
    H[j + 1, j] = 0.0
    first, second = g[j], g[j + 1]
    g[j] = np.conj(cos[j]) * first + np.conj(sin[j]) * second
    g[j + 1] = -sin[j] * first + cos[j] * second


def _reference_fgmres(A, b, *, m: int, precond, conjugate_first: bool = True):
    """One FGMRES(m) cycle; returns ``(estimate, true_relative, orthonormal)``.

    ``conjugate_first`` selects the Arnoldi projection:

    * ``True``  - ``h_ij = v_i^H w``  (``numpy.vdot(V[i], w)``), the convention
      the C++ fix implements and the one every FGMRES needs.
    * ``False`` - ``h_ij = w^H v_i``, i.e. the conjugated Hessenberg of the
      KUPL bug.

    ``estimate`` is the Givens residual estimate ``|g[m]| / ||b||``, ``true``
    is the true ``||b - A x|| / ||b||``, and ``orthonormal`` is the
    orthonormality defect ``||V^H V - I||`` of the Arnoldi basis.  The relation
    ``A Z = V H`` holds by construction for any ``h`` (the basis is defined by
    subtracting whatever ``h`` says), so orthonormality - not that relation -
    is the invariant a wrong projection destroys: ``v_i^H w`` is exactly the
    coefficient that makes ``w`` orthogonal to ``v_i``.
    """
    n = b.size
    x = np.zeros(n, dtype=complex)
    r = b - A @ x
    beta = float(np.linalg.norm(r))
    basis = np.zeros((m + 1, n), dtype=complex)
    directions = np.zeros((m, n), dtype=complex)
    hessenberg = np.zeros((m + 1, m), dtype=complex)
    g = np.zeros(m + 1, dtype=complex)
    cos = np.zeros(m, dtype=complex)
    sin = np.zeros(m, dtype=complex)
    basis[0] = r / beta
    g[0] = beta
    used = 0
    for j in range(m):
        directions[j] = precond(basis[j])
        image = A @ directions[j]
        for i in range(j + 1):
            h = (np.vdot(basis[i], image) if conjugate_first
                 else np.vdot(image, basis[i]))
            hessenberg[i, j] = h
            image = image - h * basis[i]
        hessenberg[j + 1, j] = np.linalg.norm(image)
        used = j + 1
        if hessenberg[j + 1, j] > 0.0:
            basis[j + 1] = image / hessenberg[j + 1, j]
    column = basis[:used]
    gram = column.conj() @ column.T
    orthonormal = float(np.linalg.norm(gram - np.eye(used)))
    for j in range(used):
        _apply_rotation(hessenberg, g, cos, sin, j)
    y = np.zeros(used, dtype=complex)
    for i in range(used - 1, -1, -1):
        total = g[i]
        for k in range(i + 1, used):
            total -= hessenberg[i, k] * y[k]
        y[i] = total / hessenberg[i, i] if abs(hessenberg[i, i]) > 0.0 else 0.0
    for i in range(used):
        x = x + y[i] * directions[i]
    return (abs(g[used]) / beta, float(np.linalg.norm(b - A @ x)) / beta,
            orthonormal)


# ---------------------------------------------------------------------------
# 1. the inner product convention
# ---------------------------------------------------------------------------
def test_tools_vdot_conjugates_its_first_argument():
    """``pyqcu.tools.vdot(a, b)`` must be ``a^H b = sum(conj(a) * b)``.

    ``numpy.vdot`` itself conjugates its first argument, so the Hermitian inner
    product is ``numpy.vdot(a, b)``; ``numpy.vdot(conj(a), b)`` is
    ``sum(a * b)``, the flipped (non-Hermitian) convention, and the last
    assertion pins that we are not that.
    """
    a = np.array([1.0 + 2.0j, 3.0 - 1.0j, -0.5 + 0.25j])
    b = np.array([2.0 + 0.0j, 1.0 + 1.0j, -1.5 + 0.5j])
    left = _torch.tensor(a, dtype=_torch.torch.complex128)
    right = _torch.tensor(b, dtype=_torch.torch.complex128)
    value = complex(tools.vdot(left, right))

    hermitian = complex(np.vdot(a, b))
    assert hermitian == pytest.approx(complex(np.sum(np.conj(a) * b)))
    assert value == pytest.approx(hermitian)

    flipped = complex(np.vdot(np.conj(a), b))       # == np.sum(a * b)
    assert abs(flipped - hermitian) > 1.0e-6
    assert abs(value - flipped) > 1.0e-6
    # A Hermitian form is real and non-negative on the diagonal.
    assert complex(tools.vdot(left, left)).imag == pytest.approx(0.0, abs=1e-12)
    assert complex(tools.vdot(left, left)).real == pytest.approx(
        float(np.sum(np.abs(a) ** 2)))


# ---------------------------------------------------------------------------
# 2. the reference converges - and only with the right projection
# ---------------------------------------------------------------------------
def test_reference_fgmres_converges_on_a_tiny_hermitian_system():
    A, b, precond = _hermitian_system()
    estimate, true, orthonormal = _reference_fgmres(
        A, b, m=b.size, precond=precond, conjugate_first=True)
    # m = n makes the first cycle exact, so both the estimate and the true
    # residual have to hit machine precision - and the basis has to stay
    # orthonormal.
    assert true < 1.0e-12
    assert estimate < 1.0e-12
    assert orthonormal < 1.0e-12


def test_the_flipped_projection_leaves_the_reference_unconverged():
    """The destruction proof: ``w^H v_i`` must not converge on the same system."""
    A, b, precond = _hermitian_system()
    correct = _reference_fgmres(
        A, b, m=b.size, precond=precond, conjugate_first=True)
    flipped = _reference_fgmres(
        A, b, m=b.size, precond=precond, conjugate_first=False)
    assert flipped[1] > 1.0e-4          # no solution: true residual ~1e-2
    assert flipped[1] > 1.0e6 * correct[1]
    assert flipped[2] > 1.0e-3          # the Arnoldi basis lost orthonormality
    # The estimate stays close to the true residual, so a caller that trusts it
    # would report a converged solve while x is wrong - exactly the C++
    # failure mode ("did not reach the full true residual").
    assert flipped[0] == pytest.approx(flipped[1], rel=1.0e-1)


# ---------------------------------------------------------------------------
# 3. the same proof against the live Python solver
# ---------------------------------------------------------------------------
def _solve_live(A, b, n, *, precond_diagonal, flip: bool):
    operator = _torch.tensor(A, dtype=_torch.torch.complex128)
    rhs = _torch.tensor(b, dtype=_torch.torch.complex128)
    diagonal = _torch.tensor(precond_diagonal, dtype=_torch.torch.float64)

    def matvec(value):
        return (operator @ value.reshape(-1)).reshape(value.shape)

    def precond(value):
        return (value.reshape(-1) / (diagonal + 0.0j)).reshape(value.shape)

    stats: dict = {}
    x = fgmres(rhs, matvec, tol=1.0e-10, max_iter=n, restart=n,
               precond=precond, if_rtol=True, verbose=False, stats=stats)
    relative = float(
        _torch.norm(rhs - matvec(x)).item() / _torch.norm(rhs).item())
    return relative, stats


def test_the_live_solver_loses_convergence_when_vdot_is_flipped(monkeypatch):
    """Flip ``pyqcu.tools.vdot`` in place, show the failure, then restore."""
    A, b, precond = _hermitian_system()
    diagonal = np.diag(A).real
    n = b.size

    baseline, baseline_stats = _solve_live(
        A, b, n, precond_diagonal=diagonal, flip=False)
    assert baseline < 1.0e-12
    assert baseline_stats["stop_reason"] == "native_tolerance"

    original = tools.vdot

    def flipped(a, c):
        return original(a, c).conj()

    with monkeypatch.context() as patch:
        patch.setattr(tools, "vdot", flipped)
        broken, broken_stats = _solve_live(
            A, b, n, precond_diagonal=diagonal, flip=True)
    assert broken > 1.0e-4
    assert broken > 1.0e6 * baseline
    assert broken_stats["stop_reason"] == "iteration_limit"

    # Restored: the same call converges again.
    assert tools.vdot is original
    restored, restored_stats = _solve_live(
        A, b, n, precond_diagonal=diagonal, flip=False)
    assert restored == pytest.approx(baseline, rel=1.0e-6, abs=1.0e-15)
    assert restored_stats["stop_reason"] == "native_tolerance"
