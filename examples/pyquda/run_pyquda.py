"""阶段 2（pyquda 进程）：读 h5 -> PyQuda-0.3.2（QUDA 1.1.0）侧 dslash/solver -> h5/json。

独立进程运行（dev87 F2：pyquda 与 pyqcu 不得同进程加载 libquda.so/libqcu.so）。
本进程不 import pyqcu。用法：python examples/pyquda/run_pyquda.py [--lat 8 8 8 16]
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

import pyquda  # noqa: E402
from pyquda.pyquda import dslashQuda, invertQuda  # noqa: E402
from pyquda.field import LatticeGauge, LatticeFermion  # noqa: E402
from pyquda.core import getDslash  # noqa: E402
from pyquda.enum_quda import (  # noqa: E402
    QudaMassNormalization, QudaParity, QudaVerbosity,
)


def make_wilson(lat, mass, tol, maxiter, verbosity=QudaVerbosity.QUDA_SUMMARIZE, csw=None):
    # anti_periodic_t=False：T 周期边界，与 pyqcu（torch.roll）一致
    d = (getDslash(lat, mass, tol, maxiter, clover_coeff_t=csw, anti_periodic_t=False)
         if csw else getDslash(lat, mass, tol, maxiter, anti_periodic_t=False))
    d.invert_param.verbosity = verbosity
    d.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
    d.invert_param.compute_true_res = 1
    return d


def solve_quda(d, b_lf, capture=False):
    """invertQuda 直接求解（mass 归一化，不乘 2kappa —— 对齐 pyqcu D_mass=(m+4)D_kappa）。"""
    x = LatticeFermion(b_lf.latt_size)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        invertQuda(x.data_ptr, b_lf.data_ptr, d.invert_param)
    ip = d.invert_param
    return x, ip, buf.getvalue() if capture else ""


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

    pyquda.init(grid_size=[1, 1, 1, 1])
    print("[pyquda] QUDA init OK")

    in_h5 = load_input(data_dir)
    U_q, b_q = in_h5["U_q"], in_h5["b_q"]
    # value 须为 device（cupy）数组：QUDA 按 CUDA_FIELD_LOCATION 取指针，
    # 传 numpy（host）会导致 NaN/发散（实测）。
    import cupy as cp

    # QUDA cpu_prec/cuda_prec=DOUBLE（general.py），gauge/fermion 须 complex128
    U = LatticeGauge(lat, cp.asarray(U_q.astype(np.complex128)))
    b_lf = LatticeFermion(lat, cp.asarray(b_q.astype(np.complex128)))
    print(f"[pyquda] lat={lat} mass={args.mass} tol={args.tol} kappa(quda)="
          f"{1.0/(2*(args.mass+1)):.8f}（0.3.2 kappa 定义，对齐时改用 mass 归一化）")

    d = make_wilson(lat, args.mass, args.tol, args.max_iter,
                    verbosity=QudaVerbosity.QUDA_VERBOSE)
    d.loadGauge(U)

    if not args.dslash_only:
        # ---- solver（CG/NORMOP，mass 归一化解 D_mass x = b）
        t0 = time.perf_counter()
        x, ip, log = solve_quda(d, b_lf, capture=True)
        wall_s = time.perf_counter() - t0
        iters = int(ip.iter)
        print(f"[pyquda] CG done: iters={iters} secs={ip.secs:.4f} wall={wall_s:.3f}s "
              f"true_res={ip.true_res:.3e}")
        x_q = x.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
        save_h5(data_dir / "pyquda.h5",
                x_q=x_q, iter_hist=np.array(parse_cg_iters(log), dtype=np.float64))
        save_json(f"pyquda_{tag}", {
            "lat": lat, "mass": args.mass, "tol": args.tol,
            "iters": iters, "secs": float(ip.secs), "wall_s": wall_s,
            "true_res": float(ip.true_res),
            "n_cg_rows": len(parse_cg_iters(log)),
        })

        # ---- 干净计时（SUMMARIZE，无逐迭代打印开销）
        d2 = make_wilson(lat, args.mass, args.tol, args.max_iter)
        d2.loadGauge(U)
        t0 = time.perf_counter()
        x2, ip2, _ = solve_quda(d2, b_lf)
        wall2 = time.perf_counter() - t0
        save_json(f"pyquda_perf_{tag}", {
            "iters": int(ip2.iter), "secs": float(ip2.secs), "wall_s": wall2,
            "true_res": float(ip2.true_res),
        })
        d2.destroy()

        # ---- Clover solver（csw）
        dc = make_wilson(lat, args.mass, args.tol, args.max_iter, csw=args.csw)
        t0 = time.perf_counter()
        dc.loadGauge(U)
        xc, ipc, logc = solve_quda(dc, b_lf, capture=True)
        wall_c = time.perf_counter() - t0
        print(f"[pyquda] Clover CG done: iters={int(ipc.iter)} secs={ipc.secs:.4f} "
              f"wall={wall_c:.3f}s true_res={ipc.true_res:.3e}")
        xc_q = xc.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
        save_h5(data_dir / "pyquda_clover.h5",
                xc_q=xc_q, iter_hist=np.array(parse_cg_iters(logc), dtype=np.float64))
        save_json(f"pyquda_clover_{tag}", {
            "lat": lat, "mass": args.mass, "tol": args.tol, "csw": args.csw,
            "iters": int(ipc.iter), "secs": float(ipc.secs), "wall_s": wall_c,
            "true_res": float(ipc.true_res),
        })
        dc.destroy()

    # ---- dslash 单步中间量（跳跃部分，kappa 归一化；out/in 均为对应奇偶半场）
    y = LatticeFermion(lat)
    dslashQuda(y.odd_ptr, b_lf.even_ptr, d.invert_param, QudaParity.QUDA_ODD_PARITY)
    dslashQuda(y.even_ptr, b_lf.odd_ptr, d.invert_param, QudaParity.QUDA_EVEN_PARITY)
    y_q = y.data.get().reshape(2, lat[3], lat[2], lat[1], lat[0] // 2, 4, 3)
    save_h5(data_dir / "pyquda_dslash.h5", y_q=y_q)
    print(f"[pyquda] dslash hop b norm = {np.linalg.norm(y_q):.6e}")
    d.destroy()


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