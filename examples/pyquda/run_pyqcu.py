"""阶段 1（pyqcu 进程）：生成规范场/源 -> h5；pyqcu 侧 dslash/solver 计算 -> h5/json。

独立进程运行（dev87 F2：pyqcu 与 pyquda 不得同进程加载 libqcu.so/libquda.so）。
用法：python examples/pyquda/run_pyqcu.py [--lat 8 8 8 16] [--mass 0.05] [--tol 1e-8]
      [--max-iter 2000] [--device cuda] [--dslash-only]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import numpy as np

from common import (
    DATA_DIR, KAPPA_PYQCU, LAT_DEFAULT, MASS,
    pyqcu_gauge_to_quda, pyqcu_fermion_to_quda,
    save_h5, save_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqcu import dslash, lattice, solver, tools  # noqa: E402
import pyqcu.cann as _torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dslash-only", action="store_true")
    args = ap.parse_args()

    lat = list(args.lat)
    kappa = torch.Tensor([KAPPA_PYQCU])
    device = torch.device(args.device)
    dtype = torch.complex64
    tag = "x".join(map(str, lat))
    data_dir = DATA_DIR / tag
    data_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    U = torch.zeros([3, 3, 4] + lat, dtype=dtype, device=device)
    lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    b = _torch.randn([4, 3] + lat, dtype=dtype, device=device)

    save_h5(
        data_dir / "input.h5",
        U_p=U.detach().cpu().numpy(),
        b_p=b.detach().cpu().numpy(),
        U_q=pyqcu_gauge_to_quda(U.detach().cpu().numpy(), lat),
        b_q=pyqcu_fermion_to_quda(b.detach().cpu().numpy(), lat),
        lat=np.array(lat), mass=np.array([args.mass]),
    )
    print(f"[pyqcu] kappa = {KAPPA_PYQCU:.8f} (1/(2m+8), m={args.mass})")

    _mv_time = [0.0]
    _mv_count = [0]

    def matvec(psi):
        _t = time.perf_counter()
        out = dslash.give_wilson(psi, U, kappa, verbose=False)
        _mv_time[0] += time.perf_counter() - _t
        _mv_count[0] += 1
        return out

    # ---- dslash 单步中间量（D b）
    y_p = matvec(b)
    print(f"[pyqcu] D b : norm = {_torch.norm(y_p).item():.6e}")

    # ---- solver（BiCGStab，解 D x = b）
    x_p, hist = None, []
    if not args.dslash_only:
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t_start = time.perf_counter()
        if device.type == "cuda":
            t0.record()
        x_p = solver.bistabcg(
            b, matvec, tol=args.tol, max_iter=args.max_iter,
            if_rtol=True, x0=torch.zeros_like(b), verbose=False, history=hist,
        )
        if device.type == "cuda":
            t1 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()
            wall_s = t0.elapsed_time(t1) / 1000.0
        else:
            wall_s = time.perf_counter() - t_start
        print(f"[pyqcu] BiCGStab done, iters={len(hist)}, wall={wall_s:.3f}s")
        r = matvec(x_p) - b
        rel_res = _torch.norm(r).item() / _torch.norm(b).item()
        y_hop = dslash.give_wilson(b, U, kappa, with_I=False, verbose=False)
        save_h5(data_dir / "pyqcu.h5",
                x_p=x_p.detach().cpu().numpy(),
                y_p=y_p.detach().cpu().numpy(),
                y_hop=y_hop.detach().cpu().numpy())
        save_json(f"pyqcu_{tag}", {
            "lat": lat, "mass": args.mass, "kappa": KAPPA_PYQCU,
            "tol": args.tol, "iters": len(hist), "wall_s": wall_s,
            "rel_res_full": rel_res,
            "hist": hist,
            "y_p_norm": float(_torch.norm(y_p).item()),
            "y_hop_norm": float(_torch.norm(y_hop).item()),
            "avg_matvec_s": _mv_time[0] / _mv_count[0] if _mv_count[0] else 0.0,
        })
        print(f"[pyqcu] rel_res(D x_p - b) = {rel_res:.3e}")

        # ---- Clover solver（D_cl = give_wilson + give_clover，csw=1）
        cl = dslash.make_clover(U, kappa, verbose=False)
        matvec_cl = lambda psi: (  # noqa: E731
            dslash.give_wilson(psi, U, kappa, verbose=False)
            + dslash.give_clover(psi, cl, verbose=False)
        )
        hist_cl = []
        x_cl = solver.bistabcg(
            b, matvec_cl, tol=args.tol, max_iter=args.max_iter,
            if_rtol=True, x0=torch.zeros_like(b), verbose=False, history=hist_cl,
        )
        r_cl = matvec_cl(x_cl) - b
        rel_res_cl = _torch.norm(r_cl).item() / _torch.norm(b).item()
        save_h5(data_dir / "pyqcu_clover.h5", x_cl_p=x_cl.detach().cpu().numpy())
        save_json(f"pyqcu_clover_{tag}", {
            "lat": lat, "mass": args.mass, "kappa": KAPPA_PYQCU,
            "tol": args.tol, "iters": len(hist_cl),
            "rel_res_full": rel_res_cl, "hist": hist_cl,
        })
        print(f"[pyqcu] Clover BiCGStab done, iters={len(hist_cl)}, "
              f"rel_res={rel_res_cl:.3e}")
    else:
        save_h5(data_dir / "pyqcu.h5", y_p=y_p.detach().cpu().numpy())
        save_json(f"pyqcu_{tag}", {
            "lat": lat, "mass": args.mass, "kappa": KAPPA_PYQCU,
            "y_p_norm": float(_torch.norm(y_p).item()),
        })


if __name__ == "__main__":
    main()