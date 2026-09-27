"""Identify which term the compact cross-rank backward correction actually adds.

Run with::

    mpirun -np 2 python data/diag/diag_compact_correction.py

It reuses the primitive probe helpers, computes the *expected* remote
backward term with the global Python operator, and compares it with the
observed C++/reference difference at the rank boundary plane.
"""

from __future__ import annotations

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
from pyqcu.solver._quda_multigrid import (  # noqa: E402
    Checkerboard,
    QudaMatPCOperator,
)
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


def _canonical_forms():
    torch.manual_seed(SEED + 1)
    random = torch.randn(DOF, *SHAPE, dtype=torch.complex128)
    face = random * torch.zeros(DOF, *SHAPE, dtype=torch.complex128)
    face[..., SHAPE[0] - 1, :, :, :] = random[..., SHAPE[0] - 1, :, :, :]
    two_delta = torch.zeros(DOF, *SHAPE, dtype=torch.complex128)
    two_delta[0, 7, 2, 3, 4] = 1.0
    two_delta[1, 7, 2, 3, 4] = 1.0
    two_sites = torch.zeros(DOF, *SHAPE, dtype=torch.complex128)
    two_sites[0, 7, 2, 3, 4] = 1.0
    two_sites[0, 7, 5, 6, 2] = 1.0
    low_half = torch.zeros(DOF, *SHAPE, dtype=torch.complex128)
    low_half[..., : SHAPE[0] // 2, :, :, :] = random[
        ..., : SHAPE[0] // 2, :, :, :]
    high_half = torch.zeros(DOF, *SHAPE, dtype=torch.complex128)
    high_half[..., SHAPE[0] // 2:, :, :, :] = random[
        ..., SHAPE[0] // 2:, :, :, :]
    return {
        "random": random,
        "rank0-random": low_half,
        "rank1-random": high_half,
        "face-random": face,
        "two-components": two_delta,
        "two-sites": two_sites,
    }


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
    layout = Checkerboard(SHAPE)
    params, argv = _params(local_shape, GRID, rank, DOF, PARITY,
                           data_type=define._LAT_C128_)
    set_ptrs = define.set_ptrs.clone()
    qcu.applyInitQcu(set_ptrs, params, argv)
    reports = []
    for name, global_full in _canonical_forms().items():
        compact_global = layout.extract(global_full, PARITY)
        expected_compact = QudaMatPCOperator(
            operator, parity=PARITY).apply(compact_global)
        expected_full = layout.embed(expected_compact, PARITY, DOF)
        expected_local = _localise_compact(
            expected_full, starts, local_shape, PARITY).to(device)
        compact_full = layout.embed(compact_global, PARITY, DOF)
        local_compact = _localise_compact(
            compact_full, starts, local_shape, PARITY).to(device)
        local_links = _slice_local(
            assets["preconditioned_links"], starts, local_shape).to(device)
        out = torch.empty_like(local_compact)
        scratch = torch.empty_like(local_compact)
        qcu.applyMultigridStrictMatPCQcu(
            out, local_compact, local_links, scratch, set_ptrs, params,
            PARITY)
        torch.cuda.synchronize()
        difference = (out - expected_local).cpu()
        magnitude = max(float(expected_local.abs().max()), 1.0e-30)
        if name == "random" and rank == 0:
            flat = difference.abs().reshape(-1)
            order = torch.argsort(flat, descending=True)[:5]
            top = []
            for position in order.tolist():
                index = np.unravel_index(position, difference.shape)
                top.append({
                    "index": [int(value) for value in index],
                    "actual": complex(out.cpu()[(slice(None),) + index[1:]]
                                      .reshape(-1)[0]) if False else complex(
                        out.cpu()[index]),
                    "expected": complex(expected_local.cpu()[index]),
                })
            nonzero = int((flat > 1.0e-6).sum())
            print({
                "rank": rank,
                "top_errors": top,
                "cells_over_1e-6": nonzero,
                "cells_total": int(flat.numel()),
            }, flush=True)
        reports.append({
            "form": name,
            "rank": rank,
            "max_abs_error": float(difference.abs().max()),
            "scaled_error": float(difference.abs().max()) / magnitude,
            "input_scale": float(global_full.abs().max()),
        })
    gathered = comm.gather(reports, root=0)
    if rank == 0:
        for group in gathered:
            print(group, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
