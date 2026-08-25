"""dev87 G3 向量级 Clover 隔离：MatQuda(csw=1) − MatWilson(csw=0) 与 PyQCU give_clover 对照。

两库各自对同一随机源作用，取差分得纯 Clover 贡献；扫描变换（恒等/共轭/厄米/转置）
并做复标量最小二乘，报告余弦与残差。双进程阶段隔离（libqcu/libquda 不同进程）。
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
LAT = [8, 8, 8, 16]
MASS = 0.05
KAPPA = 1.0 / (2 * MASS + 8)


def phase_a():
    """PyQCU：gauge/clover 差分基向量落盘。"""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import MASS_DEFAULT
    from pyqcu import dslash, tools
    import pyqcu.cuda.define as define
    from pyqcu.cuda.define import params as mp, argv as ma, set_ptrs as ms
    from pyqcu.cuda import qcu
    X, Y, Z, T = LAT
    p = mp.clone(); av = ma.clone(); s = ms.clone()
    dt = define._LAT_C64_
    p[define._LAT_X_] = X; p[define._LAT_Y_] = Y; p[define._LAT_Z_] = Z; p[define._LAT_T_] = T
    p[define._LAT_XYZT_] = X * Y * Z * T
    p[define._GRID_X_] = p[define._GRID_Y_] = p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = dt
    av[define._MASS_] = MASS; av[define._SIGMA_] = 0.1
    g_eo = torch.empty([2, 3, 3, 4] + define.lat_shape(p), dtype=torch.complex64, device="cuda")
    p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = -1; p[define._SEED_] = 42
    qcu.applyInitQcu(s, p, av)
    qcu.applyGaussGaugeQcu(g_eo, s, p)
    p[define._SET_INDEX_] = 0
    qcu.applyEndQcu(s, p)
    U = tools.poooxyzt2oooxyzt(g_eo).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(11)
    src = torch.randn([4, 3] + LAT, generator=gen, dtype=torch.float32,
                      device="cpu").to(torch.complex64).cuda()
    cl = dslash.make_clover(U, kappa=KAPPA)
    y_clov = dslash.give_clover(src, cl)
    y_wilson = dslash.give_wilson(src, U, torch.Tensor([KAPPA]), torch.Tensor([1.0]))
    np.savez_compressed(OUT / "_clv_phaseA.npz",
                        u=U.cpu().numpy(), src=src.cpu().numpy(),
                        y_clov=y_clov.cpu().numpy(), y_wilson=y_wilson.cpu().numpy())
    print("[phaseA] saved", flush=True)


def phase_b():
    import torch
    import pyquda
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=LAT, backend="torch", backend_target="cuda",
                enable_nvshmem=False, enable_tuning=False,
                resource_path="/tmp/opencode/quda_resource",
                enable_device_memory_pool=False, enable_pinned_memory_pool=False)
    import pyquda_utils.core as core
    from pyquda.field import LatticeFermion, LatticeGauge

    z = np.load(OUT / "_clv_phaseA.npz")
    u_np64 = z["u"].astype(np.complex128)
    src_np = z["src"]
    info = core.LatticeInfo(list(LAT), 1, 1.0)

    def to_tzyxsc(v):
        return np.ascontiguousarray(np.transpose(v.astype(np.complex128), (5, 4, 3, 2, 0, 1)))

    u_qdp = np.ascontiguousarray(np.transpose(u_np64, (2, 6, 5, 4, 3, 0, 1)))
    g = LatticeGauge(info, 4, torch.from_numpy(info.evenodd(u_qdp, True)).to("cuda"))
    x = LatticeFermion(info, torch.from_numpy(info.evenodd(to_tzyxsc(src_np), False)).to("cuda"))

    dw1 = core.getClover(info, MASS, 1e-12, 100, clover_csw_t=1.0)
    dw1.loadGauge(g)
    y1 = dw1.mat(x)
    yd1 = y1.data.cpu().numpy() if hasattr(y1.data, "cpu") else np.asarray(y1.data)
    y1_full = np.asarray(info.lexico(np.ascontiguousarray(yd1), False))
    try:
        dw1.freeGauge()
    except Exception:
        pass

    dw0 = core.getWilson(info, MASS, 1e-12, 100)
    dw0.loadGauge(g)
    y0 = dw0.mat(x)
    yd0 = y0.data.cpu().numpy() if hasattr(y0.data, "cpu") else np.asarray(y0.data)
    y0_full = np.asarray(info.lexico(np.ascontiguousarray(yd0), False))
    try:
        dw0.freeGauge()
    except Exception:
        pass

    # quda clover 差分（t,z,y,x,s,c -> s,c,x,y,z,t）
    d_q = np.transpose(y1_full - y0_full, (4, 5, 3, 2, 1, 0))
    d_q_t = torch.from_numpy(np.ascontiguousarray(d_q)).to("cuda")

    from pyqcu import tools
    y_clov = torch.from_numpy(z["y_clov"]).to("cuda")
    nb = float(tools.norm(d_q_t))
    print(f"[clv] ||quda clover diff||={nb:.4e}  ||pyqcu give_clover||={float(tools.norm(y_clov)):.4e}")

    def cos(a, b):
        na = float(tools.norm(a)); nbb = float(tools.norm(b))
        return float((torch.sum(torch.conj(a.ravel()) * b.ravel()) / (na * nbb + 1e-30)).real)

    c01 = cos(y_clov, d_q_t)
    print(f"[clv] cosine(pyqcu_clover, quda_diff)={c01:.6f}")
    variants = {
        "identity": y_clov,
        "conj": y_clov.conj(),
        "negate": -y_clov,
    }
    best = {}
    for name, yy in variants.items():
        cc = cos(yy, d_q_t)
        ls = complex((torch.sum(torch.conj(yy.ravel()) * d_q_t.ravel())
                      / (float(tools.norm(yy)) ** 2 + 1e-30)).item())
        res = float(tools.norm((d_q_t - ls * yy).ravel()) / nb)
        best[name] = {"cos": cc, "scale": ls, "rel_res": res}
        print(f"[clv] {name:14s} cos={cc:+.6f} ls_scale={ls:+.5f} rel_after_scale={res:.4e}")

    import json
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "cmp_clover_vec.json").write_text(json.dumps(
        {"norm_quda": nb, "cos_identity": c01, **best}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase-b":
        phase_b()
    else:
        phase_a()
        import subprocess
        import os as _os
        r = subprocess.run([sys.executable, __file__, "--phase-b"], env=dict(_os.environ))
        raise SystemExit(r.returncode)
