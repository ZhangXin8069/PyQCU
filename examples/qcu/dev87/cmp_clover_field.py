"""dev87 G3 Clover 场级锚定：同一随机规范下逐点比较 quda resident clover 与 PyQCU make_clover。

quda 侧经 saveClover 取 PACKED_CLOVER_ORDER host 缓冲，按 clover_field_order.h:377 解码为
每站点 6x6 复矩阵（chirality 块 {0,3}/{1,2}）。PyQCU 侧 make_clover -> [4,3,4,3,X,Y,Z,T]。
扫描变换（恒等/共轭/厄米/转置/chiral 块交换）并做全局最小二乘尺度拟合。
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
LAT = [8, 8, 8, 16]
MASS = 0.05


def phase_a():
    """独立进程：仅 PyQCU（libqcu），产出 U/cl numpy。"""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import MASS_DEFAULT
    from pyqcu import dslash, tools
    import pyqcu.cuda.define as define
    from pyqcu.cuda.define import params as mp, argv as ma, set_ptrs as ms
    from pyqcu.cuda import qcu

    """纯 PyQCU：生成规范场与 clover 项，落盘 numpy 后退出（隔离 CUDA 上下文）。"""
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
    cl = dslash.make_clover(U, kappa=1.0 / (2 * MASS + 8))
    np.savez_compressed(OUT / "_clover_phaseA.npz",
                        u=U.cpu().numpy(), cl=cl.cpu().numpy())
    print("[phaseA] saved", flush=True)


def phase_b():
    """独立进程：仅 quda/PyQUDA，绝不 import pyqcu。"""
    import torch
    import pyquda
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=LAT, backend="torch", backend_target="cuda",
                enable_nvshmem=False, enable_tuning=False,
                resource_path="/tmp/opencode/quda_resource",
                enable_device_memory_pool=False, enable_pinned_memory_pool=False)
    import pyquda_utils.core as core
    from pyquda.field import LatticeGauge

    kappa = 1.0 / (2 * MASS + 8)
    z_ = np.load(OUT / "_clover_phaseA.npz")
    U = torch.from_numpy(z_["u"])
    cl_np_full = z_["cl"]  # [4,3,4,3,X,Y,Z,T]
    X, Y, Z, T = LAT

    # ---- quda 侧 ----
    info = core.LatticeInfo(list(LAT), 1, 1.0)
    u_np = np.ascontiguousarray(np.transpose(U.double().cpu().numpy(), (2, 6, 5, 4, 3, 0, 1)))
    gq = LatticeGauge(info, 4,
                      torch.from_numpy(info.evenodd(u_np, True).astype(np.complex128)).to("cuda"))
    dw = core.getClover(info, MASS, 1e-12, 100, clover_csw_t=1.0)
    dw.loadGauge(gq)
    dw.saveClover(gq)
    data = dw.clover.data
    data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    print("[clover] quda saved buffer shape:", data.shape, data.dtype,
          "absmax=", float(np.abs(data).max()), "nnz=", int(np.count_nonzero(data)))
    # 形状 (p, T, Z, Y, X/2, chi, 36)，eo 压缩沿 x；PACKED 每块 72 doubles
    N = 6

    def unpack_full(buf2):
        """buf (2,36) -> 12x12 complex；chiral 块 {0,3}/{1,2}，物理行=s_phys*3+c。"""
        M = np.zeros((12, 12), dtype=np.complex128)
        for chi in range(2):
            blk = buf2[chi]
            phys = (0, 3) if chi == 0 else (1, 2)
            def pr(sl, cr):
                return phys[sl] * 3 + cr
            for row in range(N):
                sr, cr = divmod(row, 3)
                M[pr(sr, cr), pr(sr, cr)] += blk[row]
            done = {(pr(*divmod(i, 3)),) * 2 for i in range(N)}
            for row in range(N):
                for col in range(row + 1, N):
                    k = N * (N - 1) // 2 - (N - col) * (N - col - 1) // 2 + row - col - 1
                    re_, im_ = blk[N + 2 * k], blk[N + 2 * k + 1]
                    r_ = pr(*divmod(row, 3)); c_ = pr(*divmod(col, 3))
                    M[r_, c_] = re_ + 1j * im_
                    M[c_, r_] = re_ - 1j * im_
        return M

    cl_np = cl_np_full  # [4,3,4,3,X,Y,Z,T]

    def pyqcu_matrix(x, y, z, t):
        return cl_np[:, :, :, :, x, y, z, t].reshape(12, 12)

    def pack_expected(P):
        """PyQCU 12x12 -> (2,36)，与 quda accessor 索引一致。"""
        out = np.zeros((2, 36))
        for chi in range(2):
            phys = (0, 3) if chi == 0 else (1, 2)
            blk = out[chi]
            for lr in range(2):
                for cr in range(3):
                    r_ = phys[lr] * 3 + cr
                    blk[lr * 3 + cr] = P[r_, r_].real
            pairs = [(lr, lc) for lr in range(2) for lc in range(lr + 1, 2)]
            kk = 0
            for lr in range(2):
                for lc in range(lr + 1, 2):
                    for a_ in range(3):
                        for b_ in range(3):
                            r_ = phys[lr] * 3 + a_
                            c_ = phys[lc] * 3 + b_
                            k = N * (N - 1) // 2 - (N - (lc * 3 + b_)) * (N - (lc * 3 + b_) - 1) // 2 + (lr * 3 + a_) - (lc * 3 + b_) - 1
                            idx = N + 2 * k
                            blk[idx] = P[r_, c_].real
                            blk[idx + 1] = P[r_, c_].imag
                            kk += 1
        return out

    t0c, z0c, y0c, x0c = 0, 0, 0, 0
    p0 = (t0c + z0c + y0c + x0c) % 2
    xh0 = x0c // 2
    act = np.asarray(data[p0, t0c, z0c, y0c, xh0], dtype=np.float64)
    exp = pack_expected(pyqcu_matrix(x0c, y0c, z0c, t0c)).ravel()
    actf = act.ravel()
    print("[dbg] exp[:10] ", np.round(exp[:10], 5))
    print("[dbg] act[:10] ", np.round(actf[:10], 5))
    print("[dbg] exp norm", float(np.linalg.norm(exp)), "act norm", float(np.linalg.norm(actf)))
    cosv = float(np.vdot(exp, actf) / (np.linalg.norm(exp) * np.linalg.norm(actf) + 1e-30))
    print("[dbg] cosine(exp,act@site)=", cosv)

    samples = []
    for t_ in range(LAT[3]):
        for z_ in range(LAT[2]):
            for y_ in range(LAT[1]):
                for xh in range(LAT[0] // 2):
                    ssum = t_ + z_ + y_ + xh * 2
                    p = ssum % 2
                    x_ = xh * 2 if (ssum % 2 == 0) else xh * 2 + 1
                    pp = (x_ + y_ + z_ + t_) % 2
                    buf2 = data[p, t_, z_, y_, xh]
                    samples.append((unpack_full(np.asarray(buf2, dtype=np.float64)),
                                    pyqcu_matrix(x_, y_, z_, t_)))
                    if len(samples) >= 64:
                        break
                if len(samples) >= 64:
                    break
            if len(samples) >= 64:
                break
        if len(samples) >= 64:
            break

    Qm = np.stack([q.ravel() for q, _ in samples])
    transforms = {
        "identity": lambda A: A,
        "conj": lambda A: np.conj(A),
        "dagger": lambda A: np.conj(A.T),
        "transpose": lambda A: A.T,
        "swap_chiral": lambda A: np.eye(12)[np.array([0,9,3,6,1,10,4,7,2,11,5,8])][:, :] @ A @ np.eye(12)[np.array([0,9,3,6,1,10,4,7,2,11,5,8])].T,
        "negate": lambda A: -A,
    }
    results = {}
    for name, f in transforms.items():
        Pm = np.stack([f(p).ravel() for _, p in samples])
        a_ls = np.vdot(Pm.ravel(), Qm.ravel()) / np.vdot(Pm.ravel(), Pm.ravel())
        rel_ls = float(np.linalg.norm(Qm - a_ls * Pm) / np.linalg.norm(Qm))
        results[name] = (a_ls, rel_ls)
        print(f"[clover] {name:11s} scale={complex(a_ls):.5f} rel_after_scale={rel_ls:.4e}")
    import json
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "cmp_clover.json").write_text(json.dumps(
        {k: {"scale_re": float(np.real(v[1])), "scale_im": float(np.imag(v[1])),
             "rel": v[2]} for k, v in results.items()}, indent=2))
    try:
        dw.freeGauge()
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase-b":
        phase_b()
    else:
        phase_a()
        import subprocess, os as _os
        env = dict(_os.environ)
        r = subprocess.run([sys.executable, __file__, "--phase-b"], env=env)
        raise SystemExit(r.returncode)
