#!/usr/bin/env python3
"""Distributed, independent true residual for multi-rank KUPL fields.

KUPL distributes a periodic 4D lattice over an MPI process grid.  Each rank
owns a local sub-volume and stores fermions/gauge links in the parity-compressed
layout ``[parity, ..., x, y, z, t/2]``.  This module reconstructs the full
Wilson-Clover operator on every rank using only NumPy: it assembles the local
full-site fields, exchanges the one-site boundary planes and the backward gauge
links through MPI, applies the operator in a vectorised way, and finally
Allreduces the local residual norms.

The operator convention matches ``kupl_bistabcg.relative_full_residual`` and
therefore the single-rank independent check::

    M psi = C psi - kappa * D_bare psi,
    D_bare psi = sum_mu [(I - gamma_mu) U_mu(x) psi(x+mu)
                         + (I + gamma_mu) U_mu(x-mu)^dag psi(x-mu)].

All directions are periodic, as in the single-rank reference.  The half-time
indexing and parity selection require even local extents along every axis;
``kupl_multigrid._rank_geometry`` already enforces that for KUPL.
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import kupl_fermion_to_qcu, kupl_gauge_to_qcu
    from .kupl_bistabcg import gamma_matrix
except ImportError:  # pragma: no cover - direct script execution
    from _common import kupl_fermion_to_qcu, kupl_gauge_to_qcu
    from kupl_bistabcg import gamma_matrix


SPIN_AXIS = 0
LATTICE_AXES = (1, 2, 3, 4)
# Gauge links carry two colour axes before the four lattice axes.
LINK_AXES = (2, 3, 4, 5)


def _neighbor(coordinate: tuple[int, int, int, int],
              grid: tuple[int, int, int, int], axis: int,
              offset: int) -> int:
    target = list(coordinate)
    target[axis] = (target[axis] + offset) % grid[axis]
    return int(np.ravel_multi_index(target, grid, order="C"))


def _plane(array: np.ndarray, axis: int, side: int,
           axes: tuple[int, int, int, int] = LATTICE_AXES) -> np.ndarray:
    """Return the first (``side=0``) or last (``side=-1``) lattice plane."""
    index = [slice(None)] * array.ndim
    index[axes[axis]] = side
    return np.ascontiguousarray(array[tuple(index)])


def _shift_plus(array: np.ndarray, axis: int, halo: np.ndarray,
                axes: tuple[int, int, int, int] = LATTICE_AXES) -> np.ndarray:
    """Shift ``array`` one site towards +axis, inserting a neighbour plane."""
    out = np.empty_like(array)
    index = [slice(None)] * array.ndim
    source = [slice(None)] * array.ndim
    index[axes[axis]] = slice(0, -1)
    source[axes[axis]] = slice(1, None)
    out[tuple(index)] = array[tuple(source)]
    index[axes[axis]] = -1
    out[tuple(index)] = halo
    return out


def _shift_minus(array: np.ndarray, axis: int, halo: np.ndarray,
                 axes: tuple[int, int, int, int] = LATTICE_AXES) -> np.ndarray:
    """Shift ``array`` one site towards -axis, inserting a neighbour plane."""
    out = np.empty_like(array)
    index = [slice(None)] * array.ndim
    source = [slice(None)] * array.ndim
    index[axes[axis]] = slice(1, None)
    source[axes[axis]] = slice(0, -1)
    out[tuple(index)] = array[tuple(source)]
    index[axes[axis]] = 0
    out[tuple(index)] = halo
    return out


def _exchange_planes(array: np.ndarray, axis: int,
                     coordinate: tuple[int, int, int, int],
                     grid: tuple[int, int, int, int],
                     comm,
                     axes: tuple[int, int, int, int] = LATTICE_AXES
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Exchange the first and last planes with the two axis neighbours.

    Returns ``(halo_plus, halo_minus)`` where ``halo_plus`` is the first plane
    of the +axis neighbour and ``halo_minus`` is the last plane of the -axis
    neighbour.  With a single rank along ``axis`` the local periodic planes are
    returned directly.
    """
    first = _plane(array, axis, 0, axes)
    last = _plane(array, axis, -1, axes)
    if grid[axis] == 1:
        return first, last
    plus = _neighbor(coordinate, grid, axis, +1)
    minus = _neighbor(coordinate, grid, axis, -1)
    # Our first plane belongs to the - neighbour; receive the + neighbour's
    # first plane.  Then our last plane belongs to the + neighbour; receive the
    # - neighbour's last plane.
    halo_plus = comm.sendrecv(first, dest=minus, source=plus)
    halo_minus = comm.sendrecv(last, dest=plus, source=minus)
    return np.ascontiguousarray(halo_plus), np.ascontiguousarray(halo_minus)


def _apply_hop(psi: np.ndarray, gauge: np.ndarray,
               coordinate: tuple[int, int, int, int],
               grid: tuple[int, int, int, int], comm) -> np.ndarray:
    """Apply ``D_bare`` to a local full-site field ``[12,x,y,z,t]``."""
    if psi.shape[0] != 12:
        raise ValueError(f"expected 12 spin-color components, got {psi.shape}")
    dims = psi.shape[1:]
    output = np.zeros((12, *dims), dtype=np.complex128)

    for direction in range(4):
        # ``gauge`` is [color_out,color_in,dir,x,y,z,t].
        link = gauge[:, :, direction]
        psi_plus, psi_minus = _exchange_planes(
            psi, direction, coordinate, grid, comm)
        link_minus = _exchange_planes(
            link, direction, coordinate, grid, comm, LINK_AXES)[1]

        # Forward: (I - gamma_mu) U_mu(x) psi(x+mu).
        shifted = _shift_plus(psi, direction, psi_plus)
        transported = np.einsum(
            "abn,sbn->san", link.reshape(3, 3, -1),
            shifted.reshape(12, -1).reshape(4, 3, -1))
        gamma = np.eye(4, dtype=np.complex128) - gamma_matrix(direction)
        output += np.einsum(
            "st,tcn->scn", gamma, transported).reshape(12, *dims)

        # Backward: (I + gamma_mu) U_mu(x-mu)^dag psi(x-mu).
        shifted = _shift_minus(psi, direction, psi_minus)
        link_b = _shift_minus(link, direction, link_minus, LINK_AXES)
        transported = np.einsum(
            "ban,sbn->san", np.conj(link_b.reshape(3, 3, -1)),
            shifted.reshape(12, -1).reshape(4, 3, -1))
        gamma = np.eye(4, dtype=np.complex128) + gamma_matrix(direction)
        output += np.einsum(
            "st,tcn->scn", gamma, transported).reshape(12, *dims)
    return output


def distributed_relative_residual(
        solution: np.ndarray, source: np.ndarray, gauge: np.ndarray,
        clover_even: np.ndarray, clover_odd: np.ndarray, kappa: float,
        comm, coordinate: tuple[int, int, int, int],
        grid: tuple[int, int, int, int]) -> float:
    """Return the global relative true residual on an MPI process grid.

    ``solution``/``source`` are KUPL parity blocks
    ``[2,12,lx,ly,lz,lt/2]``; ``gauge`` is
    ``[2,3,3,4,lx,ly,lz,lt/2]``; the clover arrays are
    ``[4,3,4,3,lx,ly,lz,lt/2]``.  The result is identical to the single-rank
    NumPy reference up to floating-point reduction order.
    """
    psi = kupl_fermion_to_qcu(solution).astype(np.complex128)
    rhs = kupl_fermion_to_qcu(source).astype(np.complex128)
    links = kupl_gauge_to_qcu(gauge).astype(np.complex128)
    dims = psi.shape[2:]
    psi = np.ascontiguousarray(psi.reshape(12, *dims))
    rhs = np.ascontiguousarray(rhs.reshape(12, *dims))
    links = np.ascontiguousarray(links)

    dslash = _apply_hop(psi, links, coordinate, grid, comm)

    xs, ys, zs, ts = np.indices(dims, dtype=np.int16)
    parity = ((xs + ys + zs + ts) & 1).reshape(-1)
    flat_psi = psi.reshape(12, -1)
    flat_rhs = rhs.reshape(12, -1)
    flat_dslash = dslash.reshape(12, -1)

    numerator = 0.0
    denominator = 0.0
    for target in range(2):
        mask = parity == target
        onsite = (clover_even if target == 0 else clover_odd).reshape(12, 12, -1)
        matrix = np.ascontiguousarray(onsite.astype(np.complex128))
        solution_block = np.ascontiguousarray(flat_psi[:, mask])
        source_block = flat_rhs[:, mask]
        residual = (np.einsum("rsn,sn->rn", matrix, solution_block) -
                    kappa * flat_dslash[:, mask] - source_block)
        numerator += float(np.linalg.norm(residual) ** 2)
        denominator += float(np.linalg.norm(source_block) ** 2)
    if comm is not None:
        numerator = float(comm.allreduce(numerator))
        denominator = float(comm.allreduce(denominator))
    return float(np.sqrt(numerator) /
                 max(np.sqrt(denominator), np.finfo(float).tiny))


def gather_solution(solution: np.ndarray, lattice: tuple[int, int, int, int],
                    grid: tuple[int, int, int, int],
                    starts: tuple[int, int, int, int], comm):
    """Gather local KUPL solution blocks into one full QCU field on rank 0."""
    local = kupl_fermion_to_qcu(solution)
    local_shape = local.shape[-4:]
    if comm is None:
        return local
    parts = comm.gather(local, root=0)
    if comm.Get_rank() != 0:
        return None
    full = np.zeros((4, 3, *lattice), dtype=local.dtype)
    for rank, part in enumerate(parts):
        coordinate = tuple(
            int(value) for value in np.unravel_index(rank, grid, order="C"))
        rank_starts = tuple(
            coordinate[axis] * local_shape[axis] for axis in range(4))
        slices = tuple(
            slice(rank_starts[axis], rank_starts[axis] + local_shape[axis])
            for axis in range(4))
        full[(slice(None), slice(None)) + slices] = part
    return full
