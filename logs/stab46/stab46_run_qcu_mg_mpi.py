"""dev87 MPI 多 rank MG 冒烟：验证分布式下相对停机/真残差刷新路径。

用法（source ./env.sh 后）：
  mpirun --allow-run-as-root -np 2 python examples/qcu/dev87/run_qcu_mg_mpi.py
两 rank 可共享同一 GPU（仅正确性冒烟，不做性能声明）。
"""
import os
import sys
import time
import argparse
from pathlib import Path

os.environ.setdefault("QCU_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "dev87_mpi"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, LAT_DEFAULT, MASS_DEFAULT, SEED_DEFAULT,
                    global_parity_to_local, load_local_gauge_h5,
                    load_stencil_local, local_geometry, parse_complex_dtype,
                    process_grid, gauge_tag)
from pyqcu.cuda._multi_gpu import _SET_PTRS_COARSE_BASE_

from pyqcu import tools
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda import qcu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=[8, 8, 8, 16])
    ap.add_argument("--mass", type=float, default=MASS_DEFAULT)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--E", type=int, default=12)
    ap.add_argument("--nvi", type=int, default=1)
    ap.add_argument("--mg-grid", type=int, nargs=4, default=[2, 2, 2, 2])
    ap.add_argument("--grid", type=int, nargs=4, default=None,
                    help="MPI 进程网格 [x y z t]；默认使用 tools.give_grid_size()")
    ap.add_argument("--fine-dtype", default="c64",
                    choices=("c64", "c128"))
    ap.add_argument("--coarse-dtype", default=None,
                    choices=("c64", "c128"),
                    help="level-1 粗格精度；默认与 fine-dtype 相同")
    ap.add_argument("--coarse-max-iter", type=int, default=200)
    ap.add_argument("--restart", type=int, default=5)
    ap.add_argument("--tol-factor", type=float, default=3000.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--mu-pre", type=int, default=4)
    ap.add_argument("--smoother", choices=("cg", "mr", "chebyshev", "ca-gcr"),
                    default="cg",
                    help="粗层固定步平滑器；ca-gcr 同时选择 CA-GCR 外层")
    ap.add_argument("--solver", choices=("bicgstab", "bicgstab-l", "gcr", "ca-gcr"),
                    default=None, help="外层求解器")
    ap.add_argument("--gcr", action="store_true",
                    help="启用 FGMRES(10)+MG 预条件外层")
    ap.add_argument("--bicgstab-l", action="store_true",
                    help="启用固定 L=2 的 BiCGStab(L) 外层")
    ap.add_argument("--cycle", choices=("v", "w", "f", "k"), default="v",
                    help="递归粗网格 cycle 类型")
    ap.add_argument("--deflate", action="store_true",
                    help="启用一次初始粗网格校正")
    args = ap.parse_args()

    lat = [int(v) for v in args.lat]
    mass = args.mass
    if any(v <= 0 for v in lat + args.mg_grid):
        raise ValueError("--lat 与 --mg-grid 必须为正整数")
    if args.E <= 0 or args.nvi <= 0:
        raise ValueError("--E 与 --nvi 必须为正整数")
    if args.mu_pre < 0 or args.coarse_max_iter <= 0 or args.max_iter <= 0:
        raise ValueError("--mu-pre 必须非负，迭代上限必须为正整数")
    selected = args.solver
    if args.gcr:
        if selected is not None and selected != "gcr":
            raise ValueError("--gcr 与 --solver 的选择冲突")
        selected = "gcr"
    if args.bicgstab_l:
        if selected is not None and selected != "bicgstab-l":
            raise ValueError("--bicgstab-l 与 --solver 的选择冲突")
        selected = "bicgstab-l"
    if selected is None:
        selected = "ca-gcr" if args.smoother == "ca-gcr" else "bicgstab"
    if selected == "gcr" and args.smoother == "ca-gcr":
        raise ValueError("FGMRES 与 --smoother ca-gcr 不能同时选择")
    if selected == "bicgstab-l" and args.smoother == "ca-gcr":
        raise ValueError("BiCGStabL 与 --smoother ca-gcr 不能同时选择")
    grid = process_grid(args.grid)
    if int(np.prod(grid)) != define.size:
        raise ValueError(
            f"MPI grid {grid} has {int(np.prod(grid))} ranks, "
            f"but MPI world size is {define.size}")
    local_lat, _, _ = local_geometry(lat, grid=grid, rank=define.rank,
                                      require_even=True)
    fine_dtype, fine_code = parse_complex_dtype(args.fine_dtype)
    coarse_dtype, coarse_code = parse_complex_dtype(
        args.coarse_dtype or args.fine_dtype)
    Lx, Ly, Lz, Lt = local_lat

    # The local physical dimensions are passed to LatticeSet.  It performs
    # the parity Lt/2 conversion internally, so the Python RHS/gauge below
    # must already be rank-local but must retain the physical Lt here.
    p = mod_params.clone(); av = mod_argv.clone(); s = mod_set_ptrs.clone()
    dt = fine_code
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = grid
    p[define._NODE_RANK_] = define.rank
    p[define._NODE_SIZE_] = define.size
    p[define._DATA_TYPE_] = dt
    av = av.to(dtype=torch.float64 if fine_dtype == torch.complex128 else torch.float32)
    av[define._MASS_] = mass; av[define._ATOL_] = args.atol; av[define._SIGMA_] = 0.1

    # Only the local gauge and local coarse assets are retained on the GPU.
    # The cache is global, hence load_stencil_local performs a rank-coordinate
    # hyperslab selection before the H2D copy.
    g = load_local_gauge_h5(lat, mass, seed=SEED_DEFAULT, grid=grid,
                            rank=define.rank, device="cuda", dtype=fine_dtype)
    lonv, hnn, hdg, sit = load_stencil_local(
        lat, args.E, args.nvi, grid=grid, rank=define.rank, device="cuda",
        dtype=coarse_dtype, level=1)

    from common import make_clover_tensors
    ce, cei, coo, coi, s, p, av = make_clover_tensors(
        g, local_lat, mass, grid=grid, rank=define.rank, dtype=fine_dtype,
        data_type=fine_code)
    # make_clover_tensors returns a fresh set_ptrs clone after its two Clover
    # lifetimes; bind the coarse assets to that final clone, otherwise the
    # solver would see four null pointers.
    s[_SET_PTRS_COARSE_BASE_ + 0] = lonv.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 1] = hnn.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 2] = hdg.contiguous().data_ptr()
    s[_SET_PTRS_COARSE_BASE_ + 3] = sit.contiguous().data_ptr()

    # Prefer the exact global RHS exported by run_qcu_ops when it matches the
    # requested lattice.  A repository gauge file also carries a parity RHS;
    # use it as a deterministic fallback so an MPI launch never recursively
    # invokes a global single-rank setup under MPI_COMM_WORLD.
    expected = (2, 4, 3, *lat[:3], lat[3] // 2)
    npz_path = Path(__file__).resolve().parent / "out" / "qcu_clover_solve.npz"
    b_global = None
    if npz_path.exists():
        with np.load(npz_path) as npz:
            if tuple(npz["b_eo"].shape) == expected:
                b_global = torch.from_numpy(np.asarray(npz["b_eo"]))
    if b_global is None:
        import h5py
        gauge_path = DATA_DIR / gauge_tag(lat, mass, SEED_DEFAULT)
        with h5py.File(str(gauge_path), "r") as f:
            if "fi" not in f:
                raise KeyError(f"{gauge_path} does not contain parity RHS 'fi'")
            fi = np.asarray(f["fi"][...])
        if tuple(fi.shape) != expected:
            raise ValueError(
                f"no global RHS for lattice {lat}: expected {expected}, "
                f"found {tuple(fi.shape)} in {gauge_path}")
        b_global = torch.from_numpy(fi)
    b_eo = global_parity_to_local(
        b_global, lat, grid=grid, rank=define.rank, device="cuda",
        dtype=fine_dtype)
    expected_local = (2, 4, 3, Lx, Ly, Lz, Lt // 2)
    if tuple(b_eo.shape) != expected_local:
        raise AssertionError(
            f"local RHS shape mismatch: got {tuple(b_eo.shape)}, "
            f"expected {expected_local}")

    # The coarse cache uses [E,E,Xc,Yc,Zc,Tc], so derive the local coarse
    # geometry from the actual local asset rather than reusing global values.
    local_coarse = [int(lonv.shape[2]), int(lonv.shape[4]),
                    int(lonv.shape[6]), int(lonv.shape[8])]
    idx = int(p[define._SET_INDEX_].item())
    p[define._SET_INDEX_] = idx; p[define._SET_PLAN_] = 1
    p[define._PARITY_] = 0; p[define._MAX_ITER_] = args.max_iter; p[define._VERBOSE_] = 1
    p[define._MG_NUM_LEVEL_] = 2
    p[define._MG_LEVEL1_E_] = args.E
    p[define._MG_LEVEL1_X_] = local_coarse[0]
    p[define._MG_LEVEL1_Y_] = local_coarse[1]
    p[define._MG_LEVEL1_Z_] = local_coarse[2]
    p[define._MG_LEVEL1_T_] = local_coarse[3]
    p[define._MG_LEVEL1_MAX_ITER_] = args.coarse_max_iter
    p[define._MG_LEVEL1_DATA_TYPE_] = coarse_code
    p[define._MG_LEVEL1_NUM_RESTART_] = args.restart
    mode = 0
    if selected == "gcr":
        mode |= define._MG_MODE_GCR_
    elif selected == "bicgstab-l":
        mode |= define._MG_MODE_BICGSTABL_
    elif selected == "ca-gcr":
        mode |= define._MG_MODE_CA_GCR_
    if args.smoother == "mr":
        mode |= define._MG_MODE_MR_SMOOTHER_
    elif args.smoother == "chebyshev":
        mode |= define._MG_MODE_CHEBYSHEV_
    cycle_bits = {
        "v": 0,
        "w": define._MG_MODE_W_CYCLE_,
        "f": define._MG_MODE_F_CYCLE_,
        "k": define._MG_MODE_K_CYCLE_,
    }
    mode |= cycle_bits[args.cycle]
    p[define._MG_USE_GCR_] = mode
    p[define._MG_USE_DEFLATE_] = int(args.deflate)
    p[define._MG_MU_PRE_] = args.mu_pre
    av[define._ATOL_] = args.atol
    av[define._MG_LEVEL1_ATOL_] = args.atol * args.tol_factor

    qcu.applyInitQcu(s, p, av)
    x_mg = torch.zeros_like(b_eo)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    qcu.applyCloverMultigridQcu(x_mg, b_eo, g, ce, coo, cei, coi, s, p)
    torch.cuda.synchronize()
    if define.rank == 0:
        print(f"[mpi-mg] grid={grid} local={local_lat} coarse={local_coarse} "
              f"fine={args.fine_dtype} coarse={args.coarse_dtype or args.fine_dtype} "
              f"wall={time.perf_counter()-t0:.3f}s", flush=True)
    p[define._SET_INDEX_] = idx
    try:
        qcu.applyEndQcu(s, p)
    except Exception:
        pass


if __name__ == "__main__":
    main()
