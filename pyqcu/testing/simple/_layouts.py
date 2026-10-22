"""Cross-backend gauge, fermion, and Clover layout conversions."""

from __future__ import annotations

import numpy as np


def qcu_gauge_to_qdp(gauge: np.ndarray) -> np.ndarray:
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[:3] != (3, 3, 4):
        raise ValueError("QCU gauge must be [color,color,dir,x,y,z,t]")
    return np.ascontiguousarray(array.transpose(2, 6, 5, 4, 3, 0, 1))


def qdp_gauge_to_qcu(gauge: np.ndarray) -> np.ndarray:
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[0] != 4 or array.shape[-2:] != (3, 3):
        raise ValueError("QDP gauge must be [dir,t,z,y,x,color,color]")
    return np.ascontiguousarray(array.transpose(5, 6, 0, 4, 3, 2, 1))


def qcu_fermion_to_qdp(fermion: np.ndarray) -> np.ndarray:
    """Convert full QCU ``[s,c,x,y,z,t]`` to compressed QDP ``[q,t,z,y,x/2,s,c]``."""
    array = np.asarray(fermion)
    if array.ndim != 6 or array.shape[:2] != (4, 3):
        raise ValueError("QCU fermion must be [spin,color,x,y,z,t]")
    _, _, x, y, z, t = array.shape
    source = array.transpose(5, 4, 3, 2, 0, 1)  # t,z,y,x,s,c
    rows = np.empty((2, t, z, y, x // 2), dtype=np.int64)
    tt = np.arange(t)[:, None, None, None]
    zz = np.arange(z)[None, :, None, None]
    yy = np.arange(y)[None, None, :, None]
    for q in range(2):
        rows[q] = ((((tt * z + zz) * y + yy) * x) +
                   ((q - (tt + zz + yy)) & 1) +
                   2 * np.arange(x // 2)[None, None, None, :])
    return np.ascontiguousarray(
        source.reshape(-1, 12)[rows].reshape(2, t, z, y, x // 2, 4, 3))


def qdp_fermion_to_qcu(fermion: np.ndarray) -> np.ndarray:
    """Convert compressed QDP ``[q,t,z,y,x/2,s,c]`` to full QCU ``[s,c,x,y,z,t]``."""
    array = np.asarray(fermion)
    if array.ndim != 7 or array.shape[0] != 2 or array.shape[-2:] != (4, 3):
        raise ValueError("QDP fermion must be [q,t,z,y,x/2,spin,color]")
    _, t, z, y, xh, spin, color = array.shape
    x = 2 * xh
    output = np.empty((t, z, y, x, spin, color), dtype=array.dtype)
    tt = np.arange(t)[:, None, None, None]
    zz = np.arange(z)[None, :, None, None]
    yy = np.arange(y)[None, None, :, None]
    for q in range(2):
        rows = ((((tt * z + zz) * y + yy) * x) +
                ((q - (tt + zz + yy)) & 1) +
                2 * np.arange(xh)[None, None, None, :])
        output.reshape(-1, spin * color)[rows] = array[q].reshape(
            t, z, y, xh, spin * color)
    return np.ascontiguousarray(output.transpose(4, 5, 3, 2, 1, 0))


def qcu_gauge_to_kupl(gauge: np.ndarray) -> np.ndarray:
    """Convert full QCU gauge data to KUPL's parity-compressed layout."""
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[:3] != (3, 3, 4):
        raise ValueError("QCU gauge must be [color,color,dir,x,y,z,t]")
    x, y, z, t = array.shape[-4:]
    if t % 2:
        raise ValueError("KUPL gauge requires an even time extent")
    xs, ys, zs, t2s = np.indices((x, y, z, t // 2))
    output = np.empty((2, 3, 3, 4, x, y, z, t // 2), dtype=array.dtype)
    for parity in range(2):
        times = 2 * t2s + (((xs + ys + zs) & 1) ^ parity)
        output[parity] = array[..., xs, ys, zs, times]
    return np.ascontiguousarray(output)


def kupl_gauge_to_qcu(gauge: np.ndarray) -> np.ndarray:
    """Convert KUPL parity-compressed gauge data to full QCU layout."""
    array = np.asarray(gauge)
    if array.ndim != 8 or array.shape[:4] != (2, 3, 3, 4):
        raise ValueError("KUPL gauge must be [parity,color,color,dir,x,y,z,t/2]")
    _, _, _, _, x, y, z, ht = array.shape
    output = np.empty((3, 3, 4, x, y, z, 2 * ht), dtype=array.dtype)
    xs, ys, zs, t2s = np.indices((x, y, z, ht))
    for parity in range(2):
        times = 2 * t2s + (((xs + ys + zs) & 1) ^ parity)
        output[..., xs, ys, zs, times] = array[parity]
    return np.ascontiguousarray(output)


def qcu_fermion_to_kupl(fermion: np.ndarray) -> np.ndarray:
    """Convert full ``[spin,color,x,y,z,t]`` fermions to KUPL parity blocks."""
    array = np.asarray(fermion)
    if array.ndim != 6 or array.shape[:2] != (4, 3):
        raise ValueError("QCU fermion must be [spin,color,x,y,z,t]")
    x, y, z, t = array.shape[-4:]
    if t % 2:
        raise ValueError("KUPL fermion requires an even time extent")
    xs, ys, zs, t2s = np.indices((x, y, z, t // 2))
    output = np.empty((2, 12, x, y, z, t // 2), dtype=array.dtype)
    for parity in range(2):
        times = 2 * t2s + (((xs + ys + zs) & 1) ^ parity)
        selected = array[:, :, xs, ys, zs, times]
        selected = np.moveaxis(selected, (0, 1), (-2, -1))
        output[parity] = selected.reshape(x, y, z, t // 2, 12).transpose(
            4, 0, 1, 2, 3)
    return np.ascontiguousarray(output)


def kupl_fermion_to_qcu(fermion: np.ndarray) -> np.ndarray:
    """Convert KUPL parity blocks to full ``[spin,color,x,y,z,t]`` fermions."""
    array = np.asarray(fermion)
    if array.ndim != 6 or array.shape[0] != 2 or array.shape[1] != 12:
        raise ValueError("KUPL fermion must be [parity,12,x,y,z,t/2]")
    _, _, x, y, z, ht = array.shape
    output = np.empty((4, 3, x, y, z, 2 * ht), dtype=array.dtype)
    xs, ys, zs, t2s = np.indices((x, y, z, ht))
    for parity in range(2):
        times = 2 * t2s + (((xs + ys + zs) & 1) ^ parity)
        selected = array[parity].reshape(12, x, y, z, ht).transpose(
            1, 2, 3, 4, 0).reshape(x, y, z, ht, 4, 3)
        selected = np.moveaxis(selected, (-2, -1), (0, 1))
        output[:, :, xs, ys, zs, times] = selected
    return np.ascontiguousarray(output)


def _clover_chiral_spins(chi: int) -> tuple[int, int]:
    return (0, 3) if chi == 0 else (1, 2)


def _clover_offdiag_index(row: int, col: int) -> int:
    """Match QUDA's packed 6x6 upper-triangle index (excluding diagonal)."""
    n = 6
    return n * (n - 1) // 2 - (n - row) * (n - row - 1) // 2 + col - row - 1


def qcu_clover_to_quda(clover: np.ndarray) -> np.ndarray:
    """Convert QCU ``[4,3,4,3,x,y,z,t]`` to QUDA packed real ``[q,t,z,y,x/2,2,36]``."""
    array = np.asarray(clover)
    if array.ndim != 8 or array.shape[:4] != (4, 3, 4, 3):
        raise ValueError("QCU clover must be [4,3,4,3,x,y,z,t]")
    _, _, _, _, x, y, z, t = array.shape
    matrices = array.transpose(7, 6, 5, 4, 0, 1, 2, 3).reshape(
        t, z, y, x, 12, 12)
    packed_full = np.zeros((t, z, y, x, 2, 36), dtype=np.float64)
    for chi in range(2):
        spins = _clover_chiral_spins(chi)
        for lr, sr in enumerate(spins):
            for cr in range(3):
                row = sr * 3 + cr
                packed_full[..., chi, lr * 3 + cr] = matrices[
                    ..., row, row].real
        for row in range(6):
            sr, cr = divmod(row, 3)
            for col in range(row + 1, 6):
                sc, cc = divmod(col, 3)
                r_phys = spins[sr] * 3 + cr
                c_phys = spins[sc] * 3 + cc
                k = _clover_offdiag_index(row, col)
                packed_full[..., chi, 6 + 2 * k] = matrices[
                    ..., r_phys, c_phys].real
                packed_full[..., chi, 6 + 2 * k + 1] = matrices[
                    ..., r_phys, c_phys].imag
    q = (np.indices((t, z, y, x)).sum(axis=0) & 1)
    xh = np.arange(x)[None, None, None, :] // 2
    site_index = (((np.arange(t)[:, None, None, None] * z +
                    np.arange(z)[None, :, None, None]) * y +
                   np.arange(y)[None, None, :, None]) * (x // 2) + xh)
    destination = q * (t * z * y * (x // 2)) + site_index
    source = np.arange(t * z * y * x).reshape(t, z, y, x)
    output = np.empty((2, t, z, y, x // 2, 2, 36), dtype=np.float64)
    output.reshape(-1, 2, 36)[destination.reshape(-1)] = \
        packed_full.reshape(-1, 2, 36)[source.reshape(-1)]
    return np.ascontiguousarray(output)


def quda_clover_to_qcu(clover: np.ndarray) -> np.ndarray:
    """Convert QUDA packed real ``[q,t,z,y,x/2,2,36]`` to QCU clover."""
    array = np.asarray(clover)
    if array.ndim != 7 or array.shape[0] != 2 or array.shape[-2:] != (2, 36):
        raise ValueError("QUDA clover must be [q,t,z,y,x/2,2,36]")
    q_dim, t, z, y, xh, _, _ = array.shape
    assert q_dim == 2
    x = 2 * xh
    packed_full = np.zeros((t, z, y, x, 2, 36), dtype=np.float64)
    q = (np.indices((t, z, y, x)).sum(axis=0) & 1)
    xh_index = np.arange(x)[None, None, None, :] // 2
    site_index = (((np.arange(t)[:, None, None, None] * z +
                    np.arange(z)[None, :, None, None]) * y +
                   np.arange(y)[None, None, :, None]) * xh + xh_index)
    destination = q * (t * z * y * xh) + site_index
    source = np.arange(t * z * y * x).reshape(t, z, y, x)
    packed_full.reshape(-1, 2, 36)[source.reshape(-1)] = \
        array.reshape(-1, 2, 36)[destination.reshape(-1)]
    matrices = np.zeros((t, z, y, x, 12, 12), dtype=np.complex128)
    for chi in range(2):
        spins = _clover_chiral_spins(chi)
        for lr, sr in enumerate(spins):
            for cr in range(3):
                row = sr * 3 + cr
                matrices[..., row, row] = packed_full[..., chi, lr * 3 + cr]
        for row in range(6):
            sr, cr = divmod(row, 3)
            for col in range(row + 1, 6):
                sc, cc = divmod(col, 3)
                r_phys = spins[sr] * 3 + cr
                c_phys = spins[sc] * 3 + cc
                k = _clover_offdiag_index(row, col)
                value = (packed_full[..., chi, 6 + 2 * k] +
                         1j * packed_full[..., chi, 6 + 2 * k + 1])
                matrices[..., r_phys, c_phys] = value
                matrices[..., c_phys, r_phys] = np.conj(value)
    output = matrices.reshape(t, z, y, x, 4, 3, 4, 3)
    return np.ascontiguousarray(
        output.transpose(4, 5, 6, 7, 3, 2, 1, 0))


__all__ = [
    "kupl_fermion_to_qcu",
    "kupl_gauge_to_qcu",
    "qcu_clover_to_quda",
    "qcu_fermion_to_kupl",
    "qcu_fermion_to_qdp",
    "qcu_gauge_to_kupl",
    "qcu_gauge_to_qdp",
    "qdp_fermion_to_qcu",
    "qdp_gauge_to_qcu",
    "quda_clover_to_qcu",
]
