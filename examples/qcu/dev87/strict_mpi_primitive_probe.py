#!/usr/bin/env python3
"""Element-wise MPI validation for the strict full/compact coarse operators.

The probe builds one deterministic global nearest-neighbour operator, slices
its ``X/Y/Yhat`` assets by rank, and compares the C++ strict primitives with
the Python periodic reference.  It intentionally bypasses the production
solver-capability gate so a halo regression is reported independently of
setup/cache/FGMRES integration.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import grid_index_for_rank, process_grid
from pyqcu.cuda import define, qcu
from pyqcu.solver._quda_multigrid import (
    Checkerboard,
    QudaCoarseOperator,
    QudaMatPCOperator,
    QudaTransfer,
)


def _slice_local(tensor: torch.Tensor, starts, local_shape):
    slices = [slice(None)] * (tensor.ndim - 4)
    slices.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4)
    )
    return tensor[tuple(slices)].contiguous()


def _localise_compact(
        value: torch.Tensor, starts, local_shape, parity: int) -> torch.Tensor:
    """Slice a global parity-packed field into the C++ rank-local layout."""
    x, y, z = local_shape[:3]
    th = local_shape[3] // 2
    local = torch.empty(
        (*value.shape[:-4], x, y, z, th), dtype=value.dtype,
        device=value.device)
    for xl in range(x):
        for yl in range(y):
            for zl in range(z):
                t_parity = (parity - xl - yl - zl) & 1
                for tc in range(th):
                    global_t = starts[3] + 2 * tc + t_parity
                    local[(..., xl, yl, zl, tc)] = value[
                        (..., starts[0] + xl, starts[1] + yl,
                         starts[2] + zl, global_t)]
    return local.contiguous()


def _make_operator(
        shape, dof: int, seed: int, link_mode: str = "both",
        dtype: torch.dtype = torch.complex64
) -> QudaCoarseOperator:
    torch.manual_seed(seed)
    metadata_null = torch.randn(
        2, 4, *shape, dtype=dtype)
    metadata_transfer = QudaTransfer(
        metadata_null, shape, fine_spin=2, fine_color=dof // 2,
        coarse_spin=2, block_size=(1, 1, 1, 1))
    operator = QudaCoarseOperator(metadata_transfer, lambda value: value)
    identity = torch.eye(dof, dtype=dtype).reshape(
        dof, dof, 1, 1, 1, 1).expand(dof, dof, *shape).clone()
    operator.blocks = {(0, 0, 0, 0): 1.7 * identity}
    for dim in range(4):
        generator = torch.Generator().manual_seed(seed + 17 * dim)
        plus = tuple(1 if axis == dim else 0 for axis in range(4))
        minus = tuple(-1 if axis == dim else 0 for axis in range(4))
        plus_link = (
            0.003 * torch.randn(
                dof, dof, *shape, generator=generator,
                dtype=dtype)
        )
        if link_mode in ("both", "forward", "forward-x"):
            if link_mode == "forward-x" and dim != 0:
                continue
            operator.blocks[plus] = plus_link
        if (link_mode in ("both", "backward", "backward-x") and
                shape[dim] > 2):
            if link_mode == "backward-x" and dim != 0:
                continue
            operator.blocks[minus] = (
                0.0025 * torch.randn(
                    dof, dof, *shape, generator=generator,
                    dtype=dtype)
            )
    operator.X = operator.blocks[(0, 0, 0, 0)]
    operator._build_links()
    return operator


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual - expected).reshape(-1)
    denominator = max(float(torch.linalg.norm(expected).item()), 1e-30)
    return float(torch.linalg.norm(difference).item()) / denominator


def _params(local_shape, grid, rank, dof: int, parity: int,
            data_type: int = define._LAT_C64_):
    params = define.params.clone()
    params[define._LAT_X_:define._LAT_T_ + 1] = torch.tensor(local_shape)
    params[define._LAT_XYZT_] = int(np.prod(local_shape))
    params[define._GRID_X_:define._GRID_T_ + 1] = torch.tensor(grid)
    params[define._NODE_RANK_] = rank
    params[define._NODE_SIZE_] = int(np.prod(grid))
    params[define._DATA_TYPE_] = data_type
    params[define._PARITY_] = parity
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = define._SET_PLAN1_
    params[define._MG_NUM_LEVEL_] = 2
    params[define._MG_LEVEL1_E_] = dof
    params[define._MG_LEVEL1_X_:define._MG_LEVEL1_T_ + 1] = torch.tensor(
        local_shape)
    return params, define.argv.clone()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=[4, 4, 4, 4])
    parser.add_argument("--grid", type=int, nargs=4, default=[2, 1, 1, 1])
    parser.add_argument("--dof", type=int, default=4)
    parser.add_argument("--parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--only-matpc", action="store_true")
    parser.add_argument(
        "--link-mode",
        choices=("both", "forward", "backward", "forward-x", "backward-x",
                 "none"),
        default="both")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--delta-source", action="store_true")
    parser.add_argument("--delta-coordinate", type=int, nargs=4)
    parser.add_argument("--delta-component", type=int, default=0)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != size:
        raise ValueError(f"grid={grid} does not match MPI size={size}")
    if any(args.shape[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"shape={args.shape} is not divisible by grid={grid}")
    local_shape = tuple(
        args.shape[axis] // grid[axis] for axis in range(4))
    if any(extent % 2 for extent in local_shape):
        raise ValueError(
            f"local shape={local_shape} must be even for checkerboarding")
    coordinate = grid_index_for_rank(grid, rank)
    starts = [coordinate[axis] * local_shape[axis] for axis in range(4)]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict MPI primitive probe")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    if args.dtype == "c128":
        dtype = torch.complex128
        data_type = define._LAT_C128_
    else:
        dtype = torch.complex64
        data_type = define._LAT_C64_
    operator = _make_operator(
        tuple(args.shape), int(args.dof), int(args.seed), args.link_mode,
        dtype=dtype)
    assets = operator.to_qcu_strict_assets(
        dtype=dtype, device="cpu", include_raw_links=True)
    torch.manual_seed(args.seed + 1)
    global_full = torch.randn(
        args.dof, *args.shape, dtype=dtype)
    if args.delta_source:
        global_full.zero_()
        target = (list(args.delta_coordinate)
                  if args.delta_coordinate is not None else [0, 0, 0, 0])
        global_full[int(args.delta_component), target[0], target[1],
                    target[2], target[3]] = 1.0
    reference_full = operator.apply(global_full).contiguous()

    local_full = _slice_local(
        global_full, starts, local_shape).to(device)
    local_raw_links = _slice_local(
        assets["raw_links"], starts, local_shape).to(device)
    local_onsite = _slice_local(
        assets["onsite_pair"], starts, local_shape).to(device)
    local_preconditioned_links = _slice_local(
        assets["preconditioned_links"], starts, local_shape).to(device)
    expected_full = _slice_local(
        reference_full, starts, local_shape).to(device)

    layout = Checkerboard(tuple(args.shape))
    compact_global = layout.extract(global_full, args.parity)
    expected_compact = QudaMatPCOperator(
        operator, parity=args.parity).apply(compact_global)
    expected_compact_full = layout.embed(
        expected_compact, args.parity, args.dof)
    expected_compact_local = _localise_compact(
        expected_compact_full, starts, local_shape, args.parity).to(device)
    local_compact_full = layout.embed(
        compact_global, args.parity, args.dof)
    local_compact = _localise_compact(
        local_compact_full, starts, local_shape, args.parity).to(device)
    expected_compact_local = expected_compact_local.reshape_as(local_compact)

    params, argv = _params(
        local_shape, grid, rank, int(args.dof), args.parity,
        data_type=data_type)
    set_ptrs = define.set_ptrs.clone()
    initialized = False
    try:
        qcu.applyInitQcu(set_ptrs, params, argv)
        initialized = True
        full_rel = None
        full_out = None
        if not args.only_matpc:
            full_out = torch.empty_like(local_full)
            qcu.applyMultigridStrictCoarseQcu(
                full_out, local_full, local_raw_links, local_onsite,
                set_ptrs, params, 0)
            torch.cuda.synchronize()
            full_rel = _relative(full_out.cpu(), expected_full.cpu())

        compact_out = torch.empty_like(local_compact)
        scratch = torch.empty_like(local_compact)
        qcu.applyMultigridStrictMatPCQcu(
            compact_out, local_compact, local_preconditioned_links,
            scratch, set_ptrs, params, args.parity)
        torch.cuda.synchronize()
        compact_rel = _relative(
            compact_out.cpu(), expected_compact_local.cpu())
        if args.diagnose:
            for axis, name in enumerate("xyzt"):
                difference = (
                    compact_out - expected_compact_local).abs()
                low = difference.select(axis + 1, 0).max().item()
                high = difference.select(axis + 1, -1).max().item()
                print(
                    f"compact rank={rank} axis={name} "
                    f"low={low:.6e} high={high:.6e}",
                    flush=True)
                profile = difference.movedim(axis + 1, 0).reshape(
                    difference.shape[axis + 1], -1).max(dim=1).values
                print(
                    f"compact rank={rank} axis={name} profile="
                    + ",".join(f"{value:.3e}" for value in profile.tolist()),
                    flush=True)
        if full_out is not None:
            for axis, name in enumerate("xyzt"):
                difference = (full_out - expected_full).abs()
                low = difference.select(axis + 1, 0).max().item()
                high = difference.select(axis + 1, -1).max().item()
                print(
                    f"rank={rank} axis={name} low={low:.6e} "
                    f"high={high:.6e}",
                    flush=True)
            if args.diagnose:
                target = [starts[0], starts[1], starts[2], starts[3]]
                expected_value = reference_full[
                    0, target[0], target[1], target[2], target[3]]
                actual_value = full_out[
                    0, 0, 0, 0, 0]
                onsite_input = local_full[0, 0, 0, 0, 0]
                expected_backward = (
                    expected_value - 1.7 * global_full[
                        0, target[0], target[1], target[2], target[3]])
                actual_backward = actual_value - 1.7 * onsite_input
                own_max = [
                    starts[0] + local_shape[0] - 1,
                    starts[1] + local_shape[1] - 1,
                    starts[2] + local_shape[2] - 1,
                    starts[3] + local_shape[3] - 1,
                ]
                own_link = assets["raw_links"][1, 0]
                own_candidate = sum(
                    own_link[:, 0, own_max[0], target[1],
                              target[2], target[3]].conj() *
                    global_full[:, own_max[0], target[1],
                                target[2], target[3]]
                )
                neighbor_x = (starts[0] - 1) % args.shape[0]
                remote_candidate = sum(
                    assets["raw_links"][1, 0][
                        :, 0, neighbor_x, target[1],
                        target[2], target[3]].conj() *
                    global_full[:, neighbor_x, target[1],
                                target[2], target[3]]
                )
                print(
                    "diagnose",
                    {"rank": rank,
                     "expected": complex(expected_value),
                     "actual": complex(actual_value),
                     "expected_backward": complex(expected_backward),
                     "actual_backward": complex(actual_backward),
                     "backward_delta": complex(
                         actual_backward - expected_backward),
                     "local_wrap_candidate": complex(own_candidate),
                     "remote_candidate": complex(remote_candidate)},
                    flush=True)
                print(
                    "link-face",
                    {"rank": rank,
                     "own_max_sum": complex(own_link[
                         :, :, own_max[0], own_max[1],
                         own_max[2], own_max[3]].sum()),
                     "remote_max_sum": complex(
                         assets["raw_links"][1, 0][
                             :, :, neighbor_x, own_max[1],
                             own_max[2], own_max[3]].sum())},
                    flush=True)
        payload = {
            "rank": rank,
            "coordinate": coordinate,
            "full_relative": full_rel,
            "compact_relative": compact_rel,
        }
        print(payload, flush=True)
        tolerance = 3e-5
        if ((full_rel is not None and full_rel >= tolerance) or
                compact_rel >= tolerance):
            raise AssertionError(
                f"rank {rank}: strict MPI primitive mismatch: {payload}")
    finally:
        if initialized:
            qcu.applyEndQcu(set_ptrs, params)
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
