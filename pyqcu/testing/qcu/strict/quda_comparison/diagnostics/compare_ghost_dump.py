"""Compare the dumped C++ coarse halos against a Python reference.

Run with::

    PYQCU_STRICT_DEBUG_GHOST=/root/PyQCU/data/diag/ghost \
        mpirun -np 2 python data/diag/compare_ghost_dump.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

ROOT = Path("/root/PyQCU")
sys.path.insert(
    0, str(ROOT / "pyqcu" / "testing" / "qcu" / "strict" /
          "quda_comparison"))

from common import grid_index_for_rank, process_grid  # noqa: E402
from pyqcu.cuda import define, qcu  # noqa: E402
from pyqcu.solver._quda_multigrid import Checkerboard, QudaMatPCOperator  # noqa: E402
from strict_mpi_primitive_probe import (  # noqa: E402
    _localise_compact,
    _make_operator,
    _params,
    _slice_local,
)

SHAPE = (8, 8, 8, 8)
GRID = (2, 1, 1, 1)
DOF = 4
PARITY = 1
SEED = 20260917
PREFIX = Path("/root/PyQCU/data/diag/ghost")


def _compact_t(x, y, z, th, parity):
    return 2 * th + ((parity - x - y - z) & 1)


def hop_oracle(value, links, target_parity, shape):
    """C++ strict_hopping_parity kernel, evaluated on the global lattice."""
    E = int(value.shape[0])
    X, Y, Z, T = shape
    th_extent = T // 2
    out = torch.zeros_like(value)
    source_parity = 1 - target_parity
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                for th in range(th_extent):
                    t = _compact_t(x, y, z, th, target_parity)
                    total = torch.zeros(E, dtype=value.dtype)
                    for dim in range(4):
                        coords = [x, y, z, t]
                        for forward in (True, False):
                            neighbour = list(coords)
                            if dim == 3:
                                neighbour[3] = (
                                    coords[3] + (1 if forward else -1)) % T
                            else:
                                neighbour[dim] = (
                                    coords[dim] + (1 if forward else -1)) % [
                                    X, Y, Z, T][dim]
                            nx, ny, nz, nt = neighbour
                            offset = source_parity ^ ((nx + ny + nz) & 1)
                            if (nt - offset) % 2:
                                raise AssertionError("parity mismatch")
                            source_th = (nt - offset) // 2
                            source = value[:, nx, ny, nz, source_th]
                            link = links[0 if forward else 1, dim, :, :,
                                         x, y, z, t]
                            if forward:
                                total = total + torch.einsum(
                                    "rc,c->r", link, source)
                            else:
                                total = total + torch.einsum(
                                    "cr,c->r", link.conj(), source)
                    out[:, x, y, z, th] = total
    return out


def packed_face(compact_field, links, parity, shape, side=1):
    """Reproduce strict_launch_pack_compact_faces for one face (dim 0)."""
    E = int(compact_field.shape[0])
    X, Y, Z, T = shape
    face_count = Y * Z * T
    packed = np.zeros((E, face_count), dtype=np.complex128)
    x = 0 if side == 0 else X - 1
    for y in range(Y):
        for z in range(Z):
            for th in range(T // 2):
                t = _compact_t(x, y, z, th, parity)
                face = y * Z * T + z * T + t
                packed[:, face] = compact_field[:, x, y, z, th].numpy()
    return packed.reshape(-1)


def packed_link_face(links, shape, side=1):
    E = int(links.shape[2])
    X, Y, Z, T = shape
    face_count = Y * Z * T
    packed = np.zeros((E, E, face_count), dtype=np.complex128)
    x = 0 if side == 0 else X - 1
    for y in range(Y):
        for z in range(Z):
            for t in range(T):
                face = y * Z * T + z * T + t
                packed[(slice(None), slice(None), face)] = links[
                    :, :, x, y, z, t].numpy()
    return packed.reshape(-1)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid = process_grid(GRID)
    local_shape = tuple(SHAPE[axis] // grid[axis] for axis in range(4))
    coordinate = grid_index_for_rank(grid, rank)
    starts = [coordinate[axis] * local_shape[axis] for axis in range(4)]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    operator = _make_operator(
        SHAPE, DOF, SEED, "backward-x", dtype=torch.complex128)
    assets = operator.to_qcu_strict_assets(
        dtype=torch.complex128, device="cpu", include_raw_links=True)
    torch.manual_seed(SEED + 1)
    global_full = torch.randn(DOF, *SHAPE, dtype=torch.complex128)
    layout = Checkerboard(SHAPE)
    compact_global = layout.extract(global_full, PARITY)
    compact_full = layout.embed(compact_global, PARITY, DOF)
    compact_parity_field = torch.zeros(
        DOF, *SHAPE[:3], SHAPE[3] // 2, dtype=torch.complex128)
    for x in range(SHAPE[0]):
        for y in range(SHAPE[1]):
            for z in range(SHAPE[2]):
                for th in range(SHAPE[3] // 2):
                    t = _compact_t(x, y, z, th, PARITY)
                    compact_parity_field[:, x, y, z, th] = compact_full[
                        :, x, y, z, t]

    local_compact = _localise_compact(
        compact_full, starts, local_shape, PARITY).to(device)
    local_links = _slice_local(
        assets["preconditioned_links"], starts, local_shape).to(device)
    params, argv = _params(local_shape, GRID, rank, DOF, PARITY,
                           data_type=define._LAT_C128_)
    set_ptrs = define.set_ptrs.clone()
    qcu.applyInitQcu(set_ptrs, params, argv)
    out = torch.empty_like(local_compact)
    scratch = torch.empty_like(local_compact)
    qcu.applyMultigridStrictMatPCQcu(
        out, local_compact, local_links, scratch, set_ptrs, params, PARITY)
    torch.cuda.synchronize()
    comm.Barrier()

    maximum = int(np.prod(GRID))
    peer = (rank + 1) % maximum

    def peer_slice(tensor, peer_rank):
        coordination = grid_index_for_rank(grid, peer_rank)
        peer_starts = [
            coordination[axis] * local_shape[axis] for axis in range(4)]
        return _slice_local(
            tensor, peer_starts, local_shape)

    # hop1: in = local parity PARITY field, target parity = 1 - PARITY.
    neighbour_input = peer_slice(compact_parity_field, peer)
    expected_vector = packed_face(
        neighbour_input, None, 1 - PARITY, local_shape, side=1)
    dumped = np.fromfile(
        f"{PREFIX}.{rank}.000.vec-ghost.bin", dtype=np.complex128)
    slot = int(dumped.size // 8)
    backward_slot = dumped[slot:2 * slot]
    print({
        "rank": rank,
        "hop1_vector_ghost_max": float(
            np.abs(backward_slot - expected_vector).max()),
    }, flush=True)

    neighbour_links = peer_slice(assets["raw_links"][1, 0], peer)
    expected_link = packed_link_face(neighbour_links, local_shape, side=1)
    dumped_link = np.fromfile(
        f"{PREFIX}.{rank}.001.link-ghost.bin", dtype=np.complex128)
    print({
        "rank": rank,
        "hop1_link_ghost_max": float(
            np.abs(dumped_link - expected_link).max()),
    }, flush=True)

    local_links_cpu = _slice_local(
        assets["preconditioned_links"], starts, local_shape)
    local_compact_cpu = _localise_compact(
        compact_full, starts, local_shape, PARITY)
    oracle_hop1 = hop_oracle(
        local_compact_cpu, local_links_cpu, 1 - PARITY, local_shape)
    dumped_hop1 = np.fromfile(
        f"{PREFIX}.{rank}.002.hop1-out.bin", dtype=np.complex128).reshape(
            np.asarray(oracle_hop1).shape)
    print({
        "rank": rank,
        "hop1_out_max": float(np.abs(dumped_hop1 - oracle_hop1.numpy()).max()),
    }, flush=True)

    neighbour_scratch = peer_slice(
        layout.embed(compact_global, PARITY, DOF), peer)
    neighbour_scratch = _localise_compact(
        layout.embed(compact_global, PARITY, DOF),
        [grid_index_for_rank(grid, peer)[axis] * local_shape[axis]
         for axis in range(4)], local_shape, PARITY)
    neighbour_scratch_links = _slice_local(
        assets["preconditioned_links"],
        [grid_index_for_rank(grid, peer)[axis] * local_shape[axis]
         for axis in range(4)], local_shape)
    oracle_peer_scratch = hop_oracle(
        neighbour_scratch, neighbour_scratch_links, 1 - PARITY, local_shape)
    expected_vector2 = packed_face(
        oracle_peer_scratch, None, 1 - PARITY, local_shape, side=1)
    dumped2 = np.fromfile(
        f"{PREFIX}.{rank}.003.vec-ghost.bin", dtype=np.complex128)
    slot2 = int(dumped2.size // 8)
    print({
        "rank": rank,
        "hop2_vector_ghost_max": float(
            np.abs(dumped2[slot2:2 * slot2] - expected_vector2).max()),
    }, flush=True)

    oracle_hop2 = hop_oracle(
        oracle_hop1, local_links_cpu, PARITY, local_shape)
    oracle_out = local_compact_cpu - oracle_hop2
    dumped_hop2 = np.fromfile(
        f"{PREFIX}.{rank}.005.hop2-out.bin", dtype=np.complex128).reshape(
            np.asarray(oracle_out).shape)
    print({
        "rank": rank,
        "hop2_out_vs_oracle": float(
            np.abs(dumped_hop2 - oracle_out.numpy()).max()),
        "oracle_vs_reference": float(np.abs(
            oracle_out.numpy()
            - np.asarray(_localise_compact(
                layout.embed(QudaMatPCOperator(operator, parity=PARITY).apply(
                    compact_global), PARITY, DOF),
                starts, local_shape, PARITY))).max()),
    }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
