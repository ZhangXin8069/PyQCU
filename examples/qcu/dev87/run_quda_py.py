"""dev87 quda/PyQUDA 侧对照运行器（目标b 辅助目标a）。

前置：quda 编译安装至 $QUDA_INSTALL，PyQUDA 以 QUDA_PATH 指向该安装重装成功。
用法：
  python examples/qcu/dev87/run_quda_py.py --case solve|mg|all [--nvec 24]
对照输入：out/gauge_qdp_c64.npy 与 out/qcu_clover_solve.npz（由 run_qcu_ops.py 产出）。
"""
import argparse
import os
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def save_result(name, payload):
    path = OUT / f"{name}.json"
    payload = dict(payload)
    payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    print(f"[result] {path}")
    return path


def build_core(lat_xyzt):
    import pyquda
    import pyquda_utils.core as core
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=list(lat_xyzt), backend="torch",
                backend_target="cuda", enable_nvshmem=False)
    return core


def make_info(core, lat_xyzt):
    # t_boundary=+1（周期）与 PyQCU 一致；各向同性
    return core.LatticeInfo(list(lat_xyzt), 1, 1.0)


def to_gauge_field(core, info, qdp_npy):
    """qdp (mu,t,z,y,row,col) -> LatticeGauge（内部 eo 压缩）。"""
    import torch
    from pyquda.field import LatticeGauge
    eo = info.evenodd(np.ascontiguousarray(qdp_npy), True)
    return LatticeGauge(info, 4, torch.from_numpy(eo.astype(np.complex128)).to("cuda"))


def to_fermion_field(core, info, scxyzt):
    """PyQCU 全格点 (s,c,x,y,z,t) -> LatticeFermion。"""
    import torch
    from pyquda.field import LatticeFermion
    tzyxsc = np.ascontiguousarray(np.transpose(scxyzt.astype(np.complex128),
                                              (5, 4, 3, 2, 0, 1)))
    eo = info.evenodd(tzyxsc, False)
    return LatticeFermion(info, torch.from_numpy(eo.astype(np.complex128)).to("cuda"))


def field_to_scxyzt(info, f):
    """LatticeFermion -> (s,c,x,y,z,t) numpy。"""
    arr = np.asarray(info.lexico(np.asarray(f.data), False))  # (t,z,y,x,s,c)
    return np.ascontiguousarray(np.transpose(arr, (4, 5, 3, 2, 1, 0)))


def reconstruct_full_b(b_eo):
    """PyQCU [2,4,3,x,y,z,T/2] -> (s,c,x,y,z,t)。"""
    import torch
    from pyqcu import tools
    t_dev = torch.from_numpy(b_eo).to("cuda")
    return tools.poooxyzt2oooxyzt(t_dev).cpu().numpy()


def rel_diff(a, b):
    na = np.linalg.norm(a.ravel())
    nb = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (nb if nb > 0 else 1.0)), float(na), float(nb)


def case_solve(core, lat, mass, tol, maxiter):
    qdp = np.load(OUT / "gauge_qdp_c64.npy")
    npz = np.load(OUT / "qcu_clover_solve.npz")
    b_full = reconstruct_full_b(npz["b_eo"])
    x_qcu = reconstruct_full_b(npz["x_eo"])  # 同一布局恢复管线

    info = make_info(core, lat)
    b = to_fermion_field(core, info, b_full)
    dirac = core.getClover(info, mass, tol, maxiter, clover_csw_t=1.0)
    try:
        from pyquda.enum_quda import QudaInverterType
        dirac.invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
    except Exception as e:
        print("[solve] keep default inv_type:", e)
    if os.environ.get("DEV87_NO_CLOVER"):
        # Wilson 纯 dslash 对照：绕开 loadClover 挂点（pyquda 0.10.54 的
        # CloverWilsonDirac.loadClover 在本机构造即卡死，根因待查上游）
        from pyquda.dirac.wilson import WilsonDirac
        wd = WilsonDirac(info, mass, tol, maxiter)
        wd.loadGauge(to_gauge_field(core, info, qdp))
        t0 = time.perf_counter()
        x = wd.invert(b)
    else:
        dirac.loadGauge(to_gauge_field(core, info, qdp))
        t0 = time.perf_counter()
        x = dirac.invert(b)
    t0 = time.perf_counter()
    x = dirac.invert(b)
    torch.cuda.synchronize() if hasattr(torch, "cuda") else None
    solve_s = time.perf_counter() - t0
    x_q = field_to_scxyzt(info, x)
    rd, na, nb = rel_diff(x_q, x_qcu)
    ip = dirac.invert_param
    res = save_result("quda_clover_solve", {
        "lat": lat, "mass": mass, "tol": tol,
        "iters": int(ip.iter), "secs": float(ip.secs), "wall_s": solve_s,
        "gflops": float(ip.gflops),
        "rel_diff_vs_qcu": rd, "norm_quda": na, "norm_qcu": nb,
    })
    np.savez_compressed(OUT / "quda_clover_solve.npz", x_scxyzt=x_q)
    try:
        dirac.freeGauge()
    except Exception:
        pass
    return res


def case_mg(core, lat, mass, tol, maxiter, nvec=24, block=(2, 2, 2, 2)):
    qdp = np.load(OUT / "gauge_qdp_c64.npy")
    npz = np.load(OUT / "qcu_clover_solve.npz")
    b_full = reconstruct_full_b(npz["b_eo"])

    info = make_info(core, lat)
    b = to_fermion_field(core, info, b_full)
    dirac = core.getClover(info, mass, tol, maxiter, clover_csw_t=1.0,
                           multigrid=[list(block), [nvec]])
    t_setup0 = time.perf_counter()
    dirac.loadGauge(to_gauge_field(core, info, qdp))
    setup_s = time.perf_counter() - t_setup0
    t0 = time.perf_counter()
    x = dirac.invert(b)
    solve_s = time.perf_counter() - t0
    ip = dirac.invert_param
    res = save_result("quda_clover_mg", {
        "lat": lat, "mass": mass, "tol": tol, "nvec": nvec, "block": list(block),
        "setup_s": setup_s, "iters": int(ip.iter), "secs": float(ip.secs),
        "wall_s": solve_s, "gflops": float(ip.gflops),
    })
    try:
        dirac.multigrid.destroy()
    except Exception as e:
        print("[mg] destroy:", e)
    try:
        dirac.freeGauge()
    except Exception:
        pass
    return res


def main():
    import torch  # noqa: F401  (backend=torch 时 pyquda 内部使用)
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["solve", "mg", "all"])
    ap.add_argument("--lat", type=int, nargs=4, default=[16, 32, 32, 48])
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--nvec", type=int, default=24)
    args = ap.parse_args()
    core = build_core(args.lat)
    print("[run_quda_py] init ok", flush=True)
    if args.case in ("solve", "all"):
        print(case_solve(core, args.lat, args.mass, args.tol, args.maxiter))
    if args.case in ("mg", "all"):
        print(case_mg(core, args.lat, args.mass, args.tol, args.maxiter, args.nvec))


if __name__ == "__main__":
    main()
