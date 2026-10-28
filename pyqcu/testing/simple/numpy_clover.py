"""NumPy Wilson-Clover construction for independent residual checks."""

from __future__ import annotations

import numpy as np


_GAMMA = np.array(
    [
        [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
        [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    ],
    dtype=np.complex128,
)
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_CPP_SPIN_BLOCKS = (
    ((0, 0, -1j), (1, 1, 1j), (2, 2, -1j), (3, 3, 1j)),
    ((0, 1, -1), (1, 0, 1), (2, 1, -1), (3, 2, 1)),
    ((0, 1, 1j), (1, 0, 1j), (2, 1, -1j), (3, 2, -1j)),
    ((0, 1, -1j), (1, 0, -1j), (2, 1, -1j), (3, 2, -1j)),
    ((0, 1, -1), (1, 0, 1), (2, 1, 1), (3, 2, -1)),
    ((0, 0, 1j), (1, 1, -1j), (2, 2, -1j), (3, 3, 1j)),
)


def make_clover(
    gauge: np.ndarray, kappa: float, csw: float, *,
    convention: str = "torch",
) -> np.ndarray:
    """Return the QCU Clover term ``[4,3,4,3,x,y,z,t]``.

    The convention matches ``pyqcu.torch.dslash.make_clover`` and the CUDA
    ``applyCloversQcu`` path: a periodic four-plaquette field contraction with
    coefficient ``-kappa*csw/8``.
    """
    U = np.asarray(gauge)
    if U.ndim != 7 or U.shape[:3] != (3, 3, 4):
        raise ValueError(f"gauge must be [3,3,4,x,y,z,t], got {U.shape}")
    if not np.isfinite(kappa) or not np.isfinite(csw):
        raise ValueError("kappa and csw must be finite")
    if convention not in ("torch", "qcu_cpp"):
        raise ValueError(f"unsupported Clover convention {convention!r}")
    dtype = np.complex64 if U.dtype == np.dtype(np.complex64) else np.complex128
    U = np.ascontiguousarray(U, dtype=dtype)
    U_dag = np.ascontiguousarray(U.transpose(1, 0, 2, 3, 4, 5, 6).conj())
    spatial = U.shape[-4:]
    clover = np.zeros((4, 3, 4, 3, *spatial), dtype=dtype)
    coefficient = -0.125 * float(kappa) * float(csw)

    for pair_index, (mu, nu) in enumerate(_PAIRS):
        # QCU wards use negative indices for the four trailing lattice axes.
        axis_mu = mu - 4
        axis_nu = nu - 4
        U_mu = U[:, :, mu]
        U_nu = U[:, :, nu]
        U_dag_mu = U_dag[:, :, mu]
        U_dag_nu = U_dag[:, :, nu]
        F = np.zeros_like(U_mu)

        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", U_mu, np.roll(U_nu, -1, axis=axis_mu))
        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(U_dag_mu, -1, axis=axis_nu))
        F += np.einsum("abxyzt,bcxyzt->acxyzt", term, U_dag_nu)

        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", U_nu,
            np.roll(np.roll(U_dag_mu, 1, axis=axis_mu), -1, axis=axis_nu))
        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(U_dag_nu, 1, axis=axis_mu))
        F += np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(U_mu, 1, axis=axis_mu))

        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt",
            np.roll(U_dag_mu, 1, axis=axis_mu),
            np.roll(np.roll(U_dag_nu, 1, axis=axis_mu), 1, axis=axis_nu))
        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(np.roll(U_mu, 1, axis=axis_mu), 1, axis=axis_nu))
        F += np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(U_nu, 1, axis=axis_nu))

        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt",
            np.roll(U_dag_nu, 1, axis=axis_nu),
            np.roll(U_mu, 1, axis=axis_nu))
        term = np.einsum(
            "abxyzt,bcxyzt->acxyzt", term,
            np.roll(np.roll(U_nu, -1, axis=axis_mu), 1, axis=axis_nu))
        F += np.einsum("abxyzt,bcxyzt->acxyzt", term, U_dag_mu)

        F -= F.transpose(1, 0, 2, 3, 4, 5).conj()
        if convention == "torch":
            sigma = _GAMMA[mu] @ _GAMMA[nu]
            clover += coefficient * np.einsum(
                "Ss,Ccxyzt->SCscxyzt", sigma, F).astype(dtype, copy=False)
        else:
            for row, column, phase in _CPP_SPIN_BLOCKS[pair_index]:
                clover[row, :, column, :] += (
                    coefficient * phase * F).astype(dtype, copy=False)
    return clover


def clover_to_kupl(clover: np.ndarray) -> np.ndarray:
    """Convert full QCU Clover data to ``[2,12,12,x,y,z,t/2]`` parity blocks."""
    array = np.asarray(clover)
    if array.ndim != 8 or array.shape[:4] != (4, 3, 4, 3):
        raise ValueError("Clover must be [4,3,4,3,x,y,z,t]")
    x, y, z, t = array.shape[-4:]
    if t % 2:
        raise ValueError("Clover parity compression requires even time extent")
    xs, ys, zs, t2s = np.indices((x, y, z, t // 2))
    output = np.empty((2, 12, 12, x, y, z, t // 2), dtype=array.dtype)
    matrix = array.reshape(12, 12, x, y, z, t)
    for parity in range(2):
        times = 2 * t2s + (((xs + ys + zs) & 1) ^ parity)
        output[parity] = matrix[:, :, xs, ys, zs, times]
    return np.ascontiguousarray(output)
