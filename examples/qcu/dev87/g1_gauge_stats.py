"""dev87 G1 规范场生成统计对照：PyQCU gauss vs quda gaussGaugeQuda。

RNG 算法不同 ⇒ 不做位等比较；对比统计性质：
  - 链接单位性缺陷 max|U†U − I|
  - 平均 1x1 plaquette（实部）与标准差
双进程阶段隔离（libqcu/libquda 不同进程）。
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
LAT = [8, 8, 8, 16]
SIGMA = 0.1
SEEDS = [43, 44, 45]


def plaquette_stats(u):
    """u: [c,c,mu,x,y,z,t] -> (max_unitarity_deficit, mean_plaq, std_plaq)。"""
    Uc = np.asarray(u, dtype=np.complex128)
    # 单位性（逐方向）
    Um = np.moveaxis(Uc, 2, 0)  # (mu,c,c,...)
    G = np.einsum("mij...,mik...->mjk...", Um.conj(), Um)
    eye = np.eye(3, dtype=np.complex128)[:, :, None, None, None, None]
    deficit = float(np.abs(G - eye).max())
    def mul(A, B):
        return np.einsum("ij...,jk...->ik...", A, B)
    U0 = Uc[:, :, 0]; U1 = Uc[:, :, 1]
    A = U0
    B = np.roll(U1, -1, axis=1)              # x+mu
    C = np.roll(U0.conj(), -1, axis=2)       # x+nu
    D = U1.conj()
    P = mul(mul(mul(A, B), C), D)
    tr = np.einsum("ii...->...", P) / 3.0
    return deficit, float(tr.real.mean()), float(tr.real.std())


def phase_a():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import MASS_DEFAULT
    import pyqcu.cuda.define as define
    from pyqcu.cuda.define import params as mp, argv as ma, set_ptrs as ms
    from pyqcu.cuda import qcu
    from pyqcu import tools
    X, Y, Z, T = LAT
    p = mp.clone(); av = ma.clone(); s = ms.clone()
    dt = define._LAT_C64_
    p[define._LAT_X_] = X; p[define._LAT_Y_] = Y; p[define._LAT_Z_] = Z; p[define._LAT_T_] = T
    p[define._LAT_XYZT_] = X * Y * Z * T
    p[define._GRID_X_] = p[define._GRID_Y_] = p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = dt
    av[define._SIGMA_] = SIGMA
    out = {}
    for seed in SEEDS:
        g_eo = torch.empty([2, 3, 3, 4] + define.lat_shape(p), dtype=torch.complex64, device="cuda")
        p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = -1; p[define._SEED_] = seed
        qcu.applyInitQcu(s, p, av)
        qcu.applyGaussGaugeQcu(g_eo, s, p)
        p[define._SET_INDEX_] = 0
        qcu.applyEndQcu(s, p)
        U = tools.poooxyzt2oooxyzt(g_eo).contiguous()
        np.save(OUT / f"_g1_pyqcu_{seed}.npy", U.cpu().numpy())
        out[seed] = True
    print("[g1A] saved", sorted(out), flush=True)


def phase_b():
    import torch
    import pyquda
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=LAT, backend="torch", backend_target="cuda",
                enable_nvshmem=False, enable_tuning=False,
                resource_path="/tmp/opencode/quda_resource",
                enable_device_memory_pool=False, enable_pinned_memory_pool=False)
    from pyquda.field import LatticeGauge
    info = __import__("pyquda_utils.core", fromlist=["LatticeInfo"]).LatticeInfo(list(LAT), 1, 1.0)
    res = {}
    for seed in SEEDS:
        g = LatticeGauge(info)
        g.gauss(seed, SIGMA)
        d = g.data
        dnp = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
        # eo 压缩 -> lexico(t,z,y,x) 全格点，轴 (mu,2,Xh,Y,Z,T,r,c)
        lex = info.lexico(np.ascontiguousarray(dnp), True)  # (4,t,z,y,x,r,c)
        u_xyzt = np.transpose(lex, (0, 4, 3, 2, 1, 5, 6))   # (mu,x,y,z,t,r,c)->转成(c,c,mu,x,y,z,t)
        u = np.ascontiguousarray(np.transpose(u_xyzt, (5, 6, 0, 1, 2, 3, 4)))
        np.save(OUT / f"_g1_quda_{seed}.npy", u)
        res[seed] = True
    print("[g1B] saved", sorted(res), flush=True)


def analyze():
    rows = {}
    for tag in ("pyqcu", "quda"):
        rows[tag] = {}
        for seed in SEEDS:
            u = np.load(OUT / f"_g1_{tag}_{seed}.npy")
            rows[tag][seed] = plaquette_stats(u)
    summary = {}
    for tag, dd in rows.items():
        defs = [dd[s][0] for s in SEEDS]
        means = [dd[s][1] for s in SEEDS]
        stds = [dd[s][2] for s in SEEDS]
        summary[tag] = {"unitarity_deficit_max": max(defs),
                        "plaq_mean_range": [min(means), max(means)],
                        "plaq_std_range": [min(stds), max(stds)]}
        print(f"[g1] {tag}: {summary[tag]}")
    (OUT / "cmp_gauge_stats.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase-a":
        phase_a(); analyze() if False else None
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase-b":
        phase_b(); analyze()
    else:
        phase_a()
        import subprocess
        import os as _os
        env = dict(_os.environ)
        r = subprocess.run([sys.executable, __file__, "--phase-b"], env=env)
        raise SystemExit(r.returncode)
