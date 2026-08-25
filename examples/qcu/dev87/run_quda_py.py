"""dev87 quda/PyQUDA 侧对照运行器（目标b 辅助目标a）。

前置：quda 编译安装至 $QUDA_INSTALL，PyQUDA 以 QUDA_PATH 指向该安装重装成功。
用法：
  python examples/qcu/dev87/run_quda_py.py --case solve|mg|all [--nvec 24]
对照输入：out/gauge_qdp_c64.npy 与 out/qcu_clover_solve.npz（由 run_qcu_ops.py 产出）。
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

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
    data = f.data.cpu().numpy() if hasattr(f.data, "cpu") else np.asarray(f.data)
    arr = np.asarray(info.lexico(data, False))  # (t,z,y,x,s,c)
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


def _force_double(dirac):
    """WSL2 映射原子受限：全链路 double，规避 clover 降精度 max_element 路径。"""
    from pyquda.enum_quda import QudaPrecision as P
    dirac.setPrecision(cuda=P.QUDA_DOUBLE_PRECISION, sloppy=P.QUDA_DOUBLE_PRECISION,
                       precondition=P.QUDA_DOUBLE_PRECISION,
                       refinement_sloppy=P.QUDA_DOUBLE_PRECISION,
                       eigensolver=P.QUDA_DOUBLE_PRECISION)


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
        # gdb 定位: invertQuda 尾部 checkClover 因 inv_param 仍为 clover 型而
        # 解引用未加载的 clover 指针 → SIGSEGV。显式降级为 WILSON 型跳过。
        from pyquda.enum_quda import QudaDiracType as _QDT
        wd.invert_param.dslash_type = _QDT.QUDA_WILSON_DIRAC
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
    scale = mass + 4.0
    rd_scaled = rel_diff(x_q * scale, x_qcu)[0]
    ip = dirac.invert_param
    res = save_result("quda_clover_solve", {
        "lat": lat, "mass": mass, "tol": tol,
        "iters": int(ip.iter), "secs": float(ip.secs), "wall_s": solve_s,
        "gflops": float(ip.gflops),
        "rel_diff_vs_qcu_raw": rd,
        "rel_diff_vs_qcu": rd_scaled, "norm_scale": scale,
        "norm_quda": na, "norm_qcu": nb,
    })
    np.savez_compressed(OUT / "quda_clover_solve.npz", x_scxyzt=x_q)
    try:
        dirac.freeGauge()
    except Exception:
        pass
    return res


def case_mg(core, lat, mass, tol, maxiter, nvec=24, block=(2, 2, 2, 2)):
    qdp_path = OUT / "gauge_qdp_c64.npy"
    npz_path = OUT / "qcu_clover_solve.npz"
    meta = json.loads((OUT / "qdp_gauge_meta.json").read_text()) \
        if (OUT / "qdp_gauge_meta.json").exists() else {"lat_xyzt": [16, 32, 32, 48]}
    if list(lat) == list(meta.get("lat_xyzt", [])) and qdp_path.exists() and npz_path.exists():
        qdp = np.load(qdp_path)
        b_full = reconstruct_full_b(np.load(npz_path)["b_eo"])
    else:
        # 格子不匹配（如 smoke）：内部生成随机规范+源，保持流程自洽
        rng = np.random.default_rng(42)
        X, Y, Z, T = lat
        e3 = np.zeros((3, 3), dtype=np.complex128); e3[:] = np.eye(3)
        u = np.tile(e3.reshape(1, 1, 1, 1, 3, 3), (4, T, Z, Y, X, 1, 1)) * 0.0
        u += rng.normal(size=(4, T, Z, Y, X, 3, 3)) + 1j * rng.normal(size=(4, T, Z, Y, X, 3, 3))
        u /= np.linalg.norm(u, axis=(-2, -1), keepdims=True)
        qdp = np.ascontiguousarray(u)
        b_full = None

    info = make_info(core, lat)
    if b_full is not None:
        b = to_fermion_field(core, info, b_full)
    else:
        from pyquda.field import LatticeFermion
        X, Y, Z, T = lat
        rng = np.random.default_rng(7)
        tzyxsc = rng.normal(size=(T, Z, Y, X, 4, 3)) + 1j * rng.normal(size=(T, Z, Y, X, 4, 3))
        b = LatticeFermion(info, torch.from_numpy(
            info.evenodd(np.ascontiguousarray(tzyxsc.astype(np.complex128)), False)).to("cuda"))
    geo_levels = [list(block)]
    if all(l // b // b >= 2 for l, b in zip(lat, block)):
        geo_levels.append(list(block))
    dirac = core.getClover(info, mass, tol, maxiter, clover_csw_t=1.0,
                           multigrid=geo_levels)
    try:
        # nvec 经 Multigrid.setParam 覆盖为指定值（默认工厂为 24）
        mg_obj = dirac.multigrid
        if hasattr(mg_obj, "setParam"):
            for lv in range(len(mg_obj.param.n_vec)):
                mg_obj.setParam(n_vec=nvec, level=lv)
    except Exception as e:
        print("[mg] setParam(nvec) fallback to default:", e)
    # 注意：MG 层精度保持 quda 默认（本快照未启用 GPU_MULTIGRID_DOUBLE，
    # 强推 double 会触发 block_orthogonalize 编译期禁用分支）
    t_setup0 = time.perf_counter()
    dirac.loadGauge(to_gauge_field(core, info, qdp))
    setup_s = time.perf_counter() - t_setup0
    t0 = time.perf_counter()
    x = dirac.invert(b)
    solve_s = time.perf_counter() - t0
    ip = dirac.invert_param
    # 解一致性导出：与 PyQCU 同 b 的 MG 解对照（归一化 m+4）
    try:
        yd = x.data.cpu().numpy() if hasattr(x.data, "cpu") else np.asarray(x.data)
        y_lex = np.asarray(info.lexico(np.ascontiguousarray(yd), False))
        x_q_mg = np.ascontiguousarray(np.transpose(y_lex, (4, 5, 3, 2, 1, 0)))
        np.savez_compressed(OUT / "quda_clover_mg.npz", x_scxyzt=x_q_mg)
    except Exception as e:
        print("[mg] export x failed:", e)
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


def case_opcmp(core, lat, mass, random_gauge=False):
    """单位/随机规范下 M 算子级对比（G2/G3 锚点）：quda MatQuda vs PyQCU give_wilson(+clover)。"""
    import torch
    from pyqcu import dslash, tools
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import load_gauge_h5
    kappa = 1.0 / (2 * mass + 8)
    X, Y, Z, T = lat
    if random_gauge:
        U = tools.poooxyzt2oooxyzt(load_gauge_h5(lat, mass, device="cuda")).contiguous()
        tag = "random"
    else:
        e = torch.eye(3, dtype=torch.complex64)
        U = e.view(3, 3, 1, 1, 1, 1, 1).expand(3, 3, 4, X, Y, Z, T).contiguous().cuda()
        tag = "unit"
    gen = torch.Generator(device="cpu").manual_seed(7)
    src = torch.randn([4, 3] + list(lat), generator=gen, dtype=torch.float32,
                      device="cpu").to(torch.complex64).cuda()
    cl = dslash.make_clover(U, kappa=kappa)
    part_src = src.clone()
    part_hop = dslash.give_wilson(src, U, torch.Tensor([kappa]), torch.Tensor([1.0]),
                                  with_I=False)
    part_clov = dslash.give_clover(src, cl)
    y_pyqcu = part_src + part_hop + part_clov

    info = make_info(core, lat)
    x = to_fermion_field(core, info, src.cpu().numpy())
    u_np = np.ascontiguousarray(np.transpose(U.double().cpu().numpy(), (2, 6, 5, 4, 3, 0, 1)))
    from pyquda.field import LatticeGauge
    g = LatticeGauge(info, 4, torch.from_numpy(info.evenodd(u_np, True).astype(np.complex128)).to("cuda"))
    dw = core.getClover(info, mass, 1e-12, 100, clover_csw_t=1.0)
    _force_double(dw)
    dw.loadGauge(g)
    y = dw.mat(x)
    yd = y.data.cpu().numpy() if hasattr(y.data, "cpu") else np.asarray(y.data)
    y_lex = np.asarray(info.lexico(np.ascontiguousarray(yd), False))  # (t,z,y,x,s,c)
    y_np = np.transpose(y_lex, (4, 5, 3, 2, 1, 0))
    y_quda = torch.from_numpy(np.ascontiguousarray(y_np)).to("cuda").to(torch.complex64)
    nb = float(tools.norm(y_quda))
    rd = float(tools.norm((y_pyqcu - y_quda).ravel()) / nb)
    # 线性回归 y_q ?= c1*y_p + c2*src（判约定差：对角/跃迁系数）
    Ap = np.stack([part_src.ravel().cpu().numpy(),
                   part_hop.ravel().cpu().numpy(),
                   part_clov.ravel().cpu().numpy()], axis=1)
    bv = y_quda.ravel().cpu().numpy()
    G = Ap.conj().T @ Ap
    rhs = Ap.conj().T @ bv
    solv = np.linalg.solve(G, rhs)
    fit_rel = float(np.linalg.norm(Ap @ solv - bv) / np.linalg.norm(bv))
    sol = solv
    print(f"[opcmp] coeffs src={complex(sol[0]):.6f} hop={complex(sol[1]):.6f} "
          f"clover={complex(sol[2]):.6f} fit_rel={fit_rel:.3e}")
    res = save_result(f"opcmp_{tag}_gauge", {
        "lat": list(lat), "mass": mass, "norm_quda_y": nb, "rel_diff": rd,
        "coeff_src_re": float(np.real(sol[0])), "coeff_src_im": float(np.imag(sol[0])),
        "coeff_hop_re": float(np.real(sol[1])), "coeff_hop_im": float(np.imag(sol[1])),
        "coeff_clover_re": float(np.real(sol[2])), "coeff_clover_im": float(np.imag(sol[2])),
        "expect_hop_coeff": float(mass + 4), "lstsq_fit_rel": fit_rel})
    try:
        dw.freeGauge()
    except Exception:
        pass
    return res


def main():
    import torch  # noqa: F401  (backend=torch 时 pyquda 内部使用)
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["solve", "mg", "opcmp", "all"])
    ap.add_argument("--lat", type=int, nargs=4, default=[16, 32, 32, 48])
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--nvec", type=int, default=24)
    ap.add_argument("--block", type=int, nargs=4, default=[2,2,2,2])
    ap.add_argument("--random-gauge", action="store_true")
    args = ap.parse_args()
    core = build_core(args.lat)
    print("[run_quda_py] init ok", flush=True)
    if args.case == "opcmp":
        print(case_opcmp(core, args.lat, args.mass, random_gauge=args.random_gauge))
        return
    if args.case in ("solve", "all"):
        print(case_solve(core, args.lat, args.mass, args.tol, args.maxiter))
    if args.case in ("mg", "all"):
        print(case_mg(core, args.lat, args.mass, args.tol, args.maxiter, args.nvec, tuple(args.block)))


if __name__ == "__main__":
    main()
