#!/usr/bin/env python3
"""Compare distributed Strict-MG Galerkin setup assets with a global reference.

The probe is intentionally setup-only.  It constructs the same random global
gauge field and null-vector basis on every rank, slices the local block, and
builds the strict hierarchy with ``process_grid``.  A separate one-rank run
on the complete lattice writes a reference HDF5 file.  The distributed run
loads that file and compares the rank-local coarse assets, reporting maxima
for interior coarse sites and for sites touching a decomposed rank boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import grid_index_for_rank, process_grid
from pyqcu import lattice, tools
from pyqcu.solver import QudaStrictMultigrid

REPO = HERE.parents[2]


def _shape4(value: Sequence[int], name: str) -> Tuple[int, int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 4 or any(item <= 0 for item in result):
        raise ValueError(f"{name} 必须是四个正数，得到 {result}")
    return result  # type: ignore[return-value]


def _slice_local(tensor: torch.Tensor, starts: Sequence[int],
                 local_shape: Sequence[int]) -> torch.Tensor:
    slices = [slice(None)] * (tensor.ndim - 4)
    slices.extend(
        slice(int(starts[axis]), int(starts[axis] + local_shape[axis]))
        for axis in range(4)
    )
    return tensor[tuple(slices)].contiguous()


def _global_gauge(shape: Sequence[int], dtype: torch.dtype,
                  seed: int, sigma: float) -> torch.Tensor:
    gauge = torch.zeros(3, 3, 4, *shape, dtype=dtype)
    for color in range(3):
        gauge[color, color] = 1.0
    return lattice.generate_gauge_field(
        gauge, sigma=float(sigma), seed=int(seed), verbose=False)


def _global_null(shape: Sequence[int], dtype: torch.dtype,
                 seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    return torch.randn(
        2, 4, 3, *shape, dtype=dtype)


def _dtype(value: str) -> Tuple[torch.dtype, np.dtype]:
    name = str(value).lower()
    if name in {"c64", "complex64"}:
        return torch.complex64, np.complex64
    if name in {"c128", "complex128"}:
        return torch.complex128, np.complex128
    raise ValueError("--dtype 必须是 c64 或 c128")


def _coarse_slices(local_shape: Sequence[int], origin: Sequence[int],
                   axes: Sequence[int]) -> Tuple[slice, ...]:
    slices: list[slice] = [slice(None)] * (max(axes) + 1)
    for axis, extent in enumerate(local_shape):
        slices[int(axes[axis])] = slice(
            int(origin[axis]), int(origin[axis] + extent))
    return tuple(slices)


def _asset_axes(name: str) -> Tuple[int, ...]:
    if name == "V":
        # [E,e,Xc,bx,Yc,by,Zc,bz,Tc,bt]
        return (2, 4, 6, 8)
    if name in {"raw_links", "preconditioned_links"}:
        # [2,4,E,E,Xc,Yc,Zc,Tc]
        return (4, 5, 6, 7)
    if name == "onsite_pair":
        # [2,E,E,Xc,Yc,Zc,Tc]
        return (3, 4, 5, 6)
    raise KeyError(name)


def _asset_mask(name: str, local_coarse: Sequence[int],
                grid: Sequence[int], ndim: int) -> torch.Tensor:
    axes = _asset_axes(name)
    mask = torch.zeros(
        [1] * int(ndim), dtype=torch.bool)
    for spatial_index, (axis, extent) in enumerate(
            zip(axes, local_coarse)):
        axis_mask = torch.zeros(int(extent), dtype=torch.bool)
        if int(grid[spatial_index]) > 1:
            axis_mask[0] = True
            axis_mask[-1] = True
        shape = [1] * int(ndim)
        shape[axis] = int(extent)
        mask = mask | axis_mask.reshape(shape)
    return mask


def _relative_error(local: torch.Tensor, reference: torch.Tensor,
                    mask: torch.Tensor | None) -> Dict[str, float]:
    if tuple(local.shape) != tuple(reference.shape):
        raise ValueError(
            f"asset shape mismatch: local={tuple(local.shape)} "
            f"reference={tuple(reference.shape)}")
    difference = (local - reference).abs()
    magnitude = reference.abs()
    if mask is not None:
        selected = mask.expand_as(difference)
        difference = difference[selected]
        magnitude = magnitude[selected]
    if difference.numel() == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "count": 0}
    scale = float(magnitude.max().item())
    max_abs = float(difference.max().item())
    max_rel = max_abs / max(scale, 1.0e-30)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "count": int(difference.numel()),
    }


def _build_hierarchy(args: argparse.Namespace, comm: Any,
                     shape: Sequence[int], grid: Sequence[int],
                     local_shape: Sequence[int], starts: Sequence[int],
                     dtype: torch.dtype, device: torch.device):
    global_gauge = _global_gauge(
        shape, dtype, seed=int(args.seed), sigma=float(args.sigma))
    global_null = _global_null(
        shape, dtype, seed=int(args.seed) + 1)
    local_gauge = _slice_local(
        global_gauge, starts, local_shape).to(device).contiguous()
    local_null = _slice_local(
        global_null, starts, local_shape).to(device).contiguous()
    scalar_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    kappa = torch.tensor(
        [1.0 / (2.0 * float(args.mass) + 8.0)],
        dtype=scalar_dtype, device=device)
    u_0 = torch.ones(1, dtype=scalar_dtype, device=device)
    block_sizes = ((2, 2, 2, 2),) * (int(args.levels) - 1)
    hierarchy = QudaStrictMultigrid(
        U=local_gauge,
        clover_term=None,
        kappa=kappa,
        u_0=u_0,
        lat_size=local_shape,
        null_vectors=[local_null],
        nvec_list=[2] * (int(args.levels) - 1),
        dof_list=[12, 4],
        block_size=(
            block_sizes[0] if len(block_sizes) == 1 else block_sizes),
        max_level=int(args.levels),
        materialize_coarse=True,
        use_parity=True,
        target_parity=1,
        setup_iters=0,
        strict_galerkin_mode=args.mode,
        strict_galerkin_include_raw_links=True,
        strict_galerkin_check_support=not bool(args.skip_support_check),
        propagate_null_vectors=True,
        verbose=bool(args.verbose),
        process_grid=grid if any(int(x) > 1 for x in grid) else None,
        comm=comm if any(int(x) > 1 for x in grid) else None,
    )
    hierarchy.setup()
    return hierarchy, local_gauge, local_null


def _export_assets(hierarchy: Any) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    for level, transfer in enumerate(hierarchy.transfers):
        coarse = hierarchy.operators[level + 1]
        strict = coarse.to_qcu_strict_assets(include_raw_links=True)
        assets[f"level{level}_V"] = transfer.to_qcu_blocked().detach().cpu().numpy()
        for name in ("raw_links", "preconditioned_links", "onsite_pair"):
            value = strict[name]
            if value is None:
                continue
            assets[f"level{level}_{name}"] = value.detach().cpu().numpy()
    return assets


def _reference_path(args: argparse.Namespace,
                    shape: Sequence[int], grid: Sequence[int]) -> Path:
    if args.reference_file:
        return Path(args.reference_file)
    return (
        REPO / "data" /
        f"strict_setup_ref_{'x'.join(str(x) for x in shape)}_"
        f"{args.dtype}.h5"
    )


def _run_reference(args: argparse.Namespace, comm: Any) -> int:
    if int(comm.Get_size()) != 1:
        raise ValueError("--reference 必须在单 rank 上运行")
    shape = _shape4(args.shape, "--shape")
    grid = (1, 1, 1, 1)
    local_shape = shape
    dtype, _ = _dtype(args.dtype)
    device = torch.device(args.device)
    hierarchy, _, _ = _build_hierarchy(
        args, comm, shape, grid, local_shape, (0, 0, 0, 0), dtype, device)
    assets = _export_assets(hierarchy)
    payload = {
        "shape": list(shape),
        "grid": list(grid),
        "block_size": [2, 2, 2, 2],
        "levels": int(args.levels),
        "assets": assets,
    }
    path = _reference_path(args, shape, grid)
    tools.save_dict_h5(str(path), payload, verbose=bool(args.verbose))
    print(json.dumps({
        "status": "finished",
        "mode": "reference",
        "path": str(path),
        "shape": list(shape),
        "levels": int(args.levels),
        "assets": sorted(assets),
    }, ensure_ascii=False), flush=True)
    return 0


def _run_distributed(args: argparse.Namespace, comm: Any) -> int:
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != size:
        raise ValueError(f"grid={grid} does not match MPI size={size}")
    shape = _shape4(args.shape, "--shape")
    if any(shape[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"shape={shape} is not divisible by grid={grid}")
    local_shape = tuple(
        shape[axis] // grid[axis] for axis in range(4))
    if any(extent % 2 for extent in local_shape):
        raise ValueError(f"local_shape={local_shape} 必须各维为偶数")
    coordinate = grid_index_for_rank(grid, rank)
    starts = tuple(
        coordinate[axis] * local_shape[axis] for axis in range(4))
    dtype, _ = _dtype(args.dtype)
    device = torch.device(args.device)

    hierarchy, _, _ = _build_hierarchy(
        args, comm, shape, grid, local_shape, starts, dtype, device)
    local_assets = _export_assets(hierarchy)
    reference_path = _reference_path(args, shape, grid)
    if not Path(reference_path).exists():
        raise FileNotFoundError(
            f"reference file does not exist: {reference_path}")
    reference = tools.load_dict_h5(
        str(reference_path), verbose=bool(args.verbose))
    if list(reference.get("shape", [])) != list(shape):
        raise ValueError(
            f"reference shape={reference.get('shape')} 与请求 shape={shape} 不一致")
    if int(reference.get("levels", -1)) != int(args.levels):
        raise ValueError(
            f"reference levels={reference.get('levels')} 与请求 "
            f"levels={args.levels} 不一致")
    reference_assets = reference["assets"]
    report: Dict[str, Any] = {
        "status": "finished",
        "mode": "distributed",
        "rank": rank,
        "shape": list(shape),
        "grid": list(grid),
        "local_shape": list(local_shape),
        "start": list(starts),
        "levels": int(args.levels),
        "setup_modes": [
            stat.get("effective_probe_mode", stat.get("probe_mode"))
            for stat in hierarchy.strict_setup_stats],
        "assets": {},
    }
    for name, local in local_assets.items():
        if name not in reference_assets:
            raise KeyError(f"reference asset missing: {name}")
        local = torch.as_tensor(local, dtype=dtype, device=device)
        ref = torch.as_tensor(
            reference_assets[name], dtype=dtype, device=device)
        level = int(name.split("_", 1)[0][5:])
        local_coarse = tuple(
            int(x) for x in hierarchy.transfers[level].coarse_shape)
        origin = tuple(
            coordinate[axis] * local_coarse[axis] for axis in range(4))
        axes = _asset_axes(name.split("_", 1)[1])
        slice_spec = _coarse_slices(local_coarse, origin, axes)
        expected = ref[slice_spec]
        if tuple(local.shape) != tuple(expected.shape):
            raise ValueError(
                f"{name} shape local={tuple(local.shape)} "
                f"reference slice={tuple(expected.shape)}")
        asset_mask = _asset_mask(
            name.split("_", 1)[1], local_coarse, grid, local.ndim)
        report["assets"][name] = {
            "all": _relative_error(local, expected, None),
            "interior": _relative_error(local, expected, ~asset_mask),
            "boundary": _relative_error(local, expected, asset_mask),
        }

    gathered = comm.gather(report, root=0)
    if rank == 0:
        all_reports = [item for item in gathered if item is not None]
        summary: Dict[str, Any] = {
            "status": "finished",
            "mode": "distributed",
            "reference": str(reference_path),
            "ranks": len(all_reports),
            "assets": {},
        }
        for report_item in all_reports:
            summary.setdefault("setup_modes", {}).setdefault(
                str(report_item.get("rank")),
                report_item.get("setup_modes", []))
            for name, metrics in report_item["assets"].items():
                current = summary["assets"].setdefault(name, {
                    "interior_max_rel": 0.0,
                    "boundary_max_rel": 0.0,
                    "all_max_rel": 0.0,
                })
                current["interior_max_rel"] = max(
                    current["interior_max_rel"],
                    metrics["interior"]["max_rel"])
                current["boundary_max_rel"] = max(
                    current["boundary_max_rel"],
                    metrics["boundary"]["max_rel"])
                current["all_max_rel"] = max(
                    current["all_max_rel"], metrics["all"]["max_rel"])
        worst = max(
            (max(metric["interior_max_rel"], metric["boundary_max_rel"])
             for metric in summary["assets"].values()),
            default=0.0,
        )
        summary["worst_max_rel"] = worst
        summary["max_rel_limit"] = float(args.max_rel)
        summary["passed"] = bool(worst <= float(args.max_rel))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        passed = bool(summary["passed"])
    else:
        passed = False
    passed = bool(comm.bcast(passed, root=0))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=[8, 8, 8, 16])
    parser.add_argument("--grid", type=int, nargs=4, default=[1, 1, 1, 1])
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--mode",
                        choices=("auto", "column", "site-batch", "colored"),
                        default="site-batch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-file")
    parser.add_argument("--max-rel", type=float, default=1.0e-5)
    parser.add_argument("--skip-support-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.levels < 2:
        raise ValueError("--levels 必须 >= 2")
    return _run_reference(args, MPI.COMM_WORLD) if args.reference else \
        _run_distributed(args, MPI.COMM_WORLD)


if __name__ == "__main__":
    raise SystemExit(main())
