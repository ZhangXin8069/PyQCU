"""阶段 2（pyquda 进程）：读 h5 -> PyQUDA 侧 dslash/solver -> h5/json。

独立进程运行（dev87 F2：pyquda 与 pyqcu 不得同进程加载 libquda.so/libqcu.so）。
本进程不 import pyqcu。用法：python pyqcu/testing/pyquda/run_pyquda.py [--lat 8 8 8 16]
      [--mass 0.05] [--tol 1e-8] [--max-iter 2000] [--csw 1.0] [--dslash-only]
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import DATA_DIR, LAT_DEFAULT, MASS, quda_fermion_to_pyqcu, save_h5, save_json
from _qmp import initialize_qmp


def make_wilson(latt_info, mass, tol, maxiter, verbosity=None, csw=None):
    from pyquda.dirac import CloverWilsonDirac, WilsonDirac
    from pyquda.enum_quda import (
        QudaMassNormalization, QudaPrecision, QudaVerbosity,
    )

    if verbosity is None:
        verbosity = QudaVerbosity.QUDA_SUMMARIZE
    # LatticeInfo 默认 t_boundary=+1，与 PyQCU 的周期 T 边界一致。
    cls = CloverWilsonDirac if csw is not None else WilsonDirac
    kwargs = {"clover_csw": csw} if csw is not None else {}
    d = cls(latt_info, mass, tol, maxiter, **kwargs)
    d.setVerbosity(verbosity)
    d.setPrecision(
        cuda=QudaPrecision.QUDA_SINGLE_PRECISION,
        sloppy=QudaPrecision.QUDA_SINGLE_PRECISION,
        precondition=QudaPrecision.QUDA_SINGLE_PRECISION,
        refinement_sloppy=QudaPrecision.QUDA_SINGLE_PRECISION,
        eigensolver=QudaPrecision.QUDA_SINGLE_PRECISION,
    )
    d.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
    d.invert_param.compute_true_res = 1
    return d


def solve_quda(d, b_lf, capture=False):
    """Solve with the current PyQUDA Dirac API and capture iteration logs."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        x = d.invert(b_lf)
    ip = d.invert_param
    log = stdout.getvalue() + stderr.getvalue()
    return x, ip, log if capture else ""


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        # QUDA exposes per-shift arrays for multi-shift capable fields.  This
        # runner performs one solve, which is stored in slot zero.
        return value[0]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--csw", type=float, default=1.0)
    ap.add_argument("--dslash-only", action="store_true")
    args = ap.parse_args()

    lat = list(args.lat)
    tag = "x".join(map(str, lat))
    data_dir = DATA_DIR / tag

    qmp_runtime = initialize_qmp((1, 1, 1, 1))
    import pyquda
    from pyquda.enum_quda import QudaVerbosity
    from pyquda.field import LatticeFermion, LatticeGauge, LatticeInfo

    pyquda.init(
        grid_size=[1, 1, 1, 1], latt_size=lat, enable_nvshmem=False)
    print("[pyquda] QUDA init OK")
    print(f"[pyquda] QMP={qmp_runtime['library']}")
    latt_info = LatticeInfo(lat)

    in_h5 = load_input(data_dir)
    U_q, b_q = in_h5["U_q"], in_h5["b_q"]
    # value 须为 device（cupy）数组：QUDA 按 CUDA_FIELD_LOCATION 取指针，
    # 传 numpy（host）会导致 NaN/发散（实测）。
    import cupy as cp

    # PyQUDA field 的 host storage 使用 complex128；device precision 由
    # make_wilson() 显式设为 single，与 PyQCU 的 canonical c64 数据对齐。
    U = LatticeGauge(
        latt_info, cp.asarray(U_q.astype(np.complex128)))
    b_lf = LatticeFermion(
        latt_info, cp.asarray(b_q.astype(np.complex128)))
    print(f"[pyquda] lat={lat} mass={args.mass} tol={args.tol} kappa(quda)="
          f"{1.0/(2*(args.mass+1)):.8f}（PyQUDA kappa 定义，对齐时改用 mass 归一化）")

    d = make_wilson(latt_info, args.mass, args.tol, args.max_iter,
                    verbosity=QudaVerbosity.QUDA_VERBOSE)
    d.loadGauge(U)

    if not args.dslash_only:
        # ---- solver（CG/NORMOP，mass 归一化解 D_mass x = b）
        t0 = time.perf_counter()
        x, ip, log = solve_quda(d, b_lf, capture=True)
        wall_s = time.perf_counter() - t0
        iters = int(_scalar(ip.iter, 0))
        secs = float(_scalar(ip.secs, 0.0))
        true_res = float(_scalar(ip.true_res, float("nan")))
        print(f"[pyquda] CG done: iters={iters} secs={secs:.4f} wall={wall_s:.3f}s "
              f"true_res={true_res:.3e}")
        x_q = x.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
        save_h5(data_dir / "pyquda.h5",
                x_q=x_q, iter_hist=np.array(parse_cg_iters(log), dtype=np.float64))
        save_json(f"pyquda_{tag}", {
            "lat": lat, "mass": args.mass, "tol": args.tol,
            "iters": iters, "secs": secs, "wall_s": wall_s,
            "true_res": true_res,
            "n_cg_rows": len(parse_cg_iters(log)),
        })

        # ---- 干净计时（SUMMARIZE，无逐迭代打印开销）
        d2 = make_wilson(latt_info, args.mass, args.tol, args.max_iter)
        d2.loadGauge(U)
        t0 = time.perf_counter()
        x2, ip2, _ = solve_quda(d2, b_lf)
        wall2 = time.perf_counter() - t0
        save_json(f"pyquda_perf_{tag}", {
            "iters": int(_scalar(ip2.iter, 0)),
            "secs": float(_scalar(ip2.secs, 0.0)),
            "wall_s": wall2,
            "true_res": float(_scalar(ip2.true_res, float("nan"))),
        })
        d2.freeGauge()

        # ---- Clover solver（csw）
        dc = make_wilson(
            latt_info, args.mass, args.tol, args.max_iter, csw=args.csw)
        t0 = time.perf_counter()
        dc.loadGauge(U)
        xc, ipc, logc = solve_quda(dc, b_lf, capture=True)
        wall_c = time.perf_counter() - t0
        print(f"[pyquda] Clover CG done: iters={int(_scalar(ipc.iter, 0))} "
              f"secs={float(_scalar(ipc.secs, 0.0)):.4f} "
              f"wall={wall_c:.3f}s "
              f"true_res={float(_scalar(ipc.true_res, float('nan'))):.3e}")
        xc_q = xc.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
        save_h5(data_dir / "pyquda_clover.h5",
                xc_q=xc_q, iter_hist=np.array(parse_cg_iters(logc), dtype=np.float64))
        save_json(f"pyquda_clover_{tag}", {
            "lat": lat, "mass": args.mass, "tol": args.tol, "csw": args.csw,
            "iters": int(_scalar(ipc.iter, 0)),
            "secs": float(_scalar(ipc.secs, 0.0)),
            "wall_s": wall_c,
            "true_res": float(_scalar(ipc.true_res, float("nan"))),
        })
        dc.freeGauge()

    # ---- dslash 单步中间量（跳跃部分，kappa 归一化；out/in 均为对应奇偶半场）
    d.loadGauge(U)
    y = d.dslash(b_lf)
    y_q = y.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
    save_h5(data_dir / "pyquda_dslash.h5", y_q=y_q)
    print(f"[pyquda] dslash hop b norm = {np.linalg.norm(y_q):.6e}")
    d.freeGauge()


def load_input(data_dir: Path):
    import h5py

    path = data_dir / "input.h5"
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]) for k in ("U_q", "b_q")}


def parse_cg_iters(log: str):
    import re

    rows = []
    for line in log.splitlines():
        m = re.search(r"CG:\s*(\d+)\s+iterations.*?\|r\|/\|b\|\s*=\s*([0-9.eE+-]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    return rows


if __name__ == "__main__":
    main()
