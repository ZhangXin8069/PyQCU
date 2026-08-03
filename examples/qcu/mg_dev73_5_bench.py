#!/usr/bin/env python3
"""dev73_5 —— MultiGrid 扩展性能基准（精度 / 格子 / 求解器参数 × BiStabCG 对照）。

保持与 mg-v4-report-2026-08-02 相同的变量与最终结果约定：
  * 参考求解器 = applyCloverBistabCgQcu（奇偶预条件 Schur BiStabCG, VERBOSE=0）
  * MultiGrid   = applyCloverMultigridQcu（Schur-一致，多层粗层，null_vecs 缓存）
  * mass=0.05, atol=1e-6, gauge_seed=42, kappa=1/(2m+8), sigma=0.1
扫描轴：
  * 精度 precision : c64（默认单精度）/ c128（双精度）
  * 格子 lattice   : {8,16,16,16}（默认，各向均不小于）、{16,16,16,16}、{8,16,16,32}
  * 求解器参数     :
      - V-cycle 频率 r（_MG_LEVEL1_NUM_RESTART_）          —— “是否进入下一层”
      - 最粗层收敛条件 ct（_MG_LEVEL1_ATOL_ = ATOL*ct）      —— “最粗一层的收敛条件”
      - 最粗层最大迭代 cmi（_MG_LEVEL1_MAX_ITER_）          —— 平滑/求解迭代上限
      - 层数 levels（2L / 3L）
每配置记录：ref 与 mg 耗时（中位数）、加速比、细层迭代数、收敛残差、MG 收敛历史
（CONVERGENCE_HISTORY）、计算热点（PROF_SECTIONS: fine_iter/vcycle/coarse_solve/
coarse_dslash），并附参考 BiStabCG 的逐迭代收敛历史（同算子 Python 复现）。

用法（在 /root/PyQCU 下运行以保证 logs/clover_multigrid.log 落到 logs/）：
    source ./env.sh && CUDA_VISIBLE_DEVICES=2 \
        python examples/qcu/mg_dev73_5_bench.py [--only prefix ...]
输出：logs/dev73_5_bench.json
"""
import torch, os, sys, time, json, re
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_csm = _load("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "conftest.schur.multigrid.py"))
build_config = _csm.build_config
from mg_nullvec_cache import build_or_load_coarse_ops

LOG_DIR = "/root/PyQCU/logs"
LOG_PATH = os.path.join(LOG_DIR, "clover_multigrid.log")

# 参考收敛历史缓存：key = (lattice_tuple, precision)
_REF_HIST_CACHE = {}


# ----------------------------------------------------------------------
# log 解析
# ----------------------------------------------------------------------
def parse_mg_log(path=LOG_PATH):
    """返回 (conv_history, prof_sections, total_iterations)."""
    conv, prof, n_iter = [], None, None
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
            if m:
                conv = [float(x) for x in m.group(1).split(",") if x.strip()]
            m = re.search(r'PROF_SECTIONS:\s*(.*)', line)
            if m:
                prof = m.group(1).strip()
            m = re.search(r'Total iterations:\s*(\d+)', line)
            if m:
                n_iter = int(m.group(1))
    prof_d = {}
    if prof:
        for tok in prof.split():
            if "=" in tok:
                k, v = tok.split("=")
                v = v.rstrip("ms")
                prof_d[k] = float(v)
    return conv, prof_d, n_iter


# ----------------------------------------------------------------------
# 参考 BiStabCG 逐迭代收敛历史（同 Schur 算子，Python 复现，仅用于画图）
# ----------------------------------------------------------------------
def bistabcg_history(b, matvec, tol, max_iter=2000):
    x = torch.zeros_like(b)
    r = b - matvec(x)
    r_norm = float(tools.norm(r))
    hist = [r_norm]
    if r_norm < tol:
        return x, hist
    r_tilde = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    s = torch.zeros_like(b)
    t = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    for i in range(max_iter):
        rho = tools.vdot(r_tilde, r)
        if abs(rho) < 1e-30:
            break
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = matvec(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv) < 1e-30:
            break
        alpha = rho / rtv
        s = r - alpha * v
        t = matvec(s)
        tts = tools.vdot(t, t)
        if abs(tts) < 1e-30:
            break
        omega = tools.vdot(t, s) / tts
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = float(tools.norm(r))
        hist.append(r_norm)
        if r_norm < tol:
            break
    return x, hist


def ref_conv_history(op, qcu_src, tol):
    """在奇偶 Schur 系统上复现参考 BiStabCG，返回逐迭代残差历史。"""
    ls_full = list(qcu_src.shape[2:])        # [X,Y,Z,T]
    ls_odd = ls_full.copy()
    ls_odd[3] //= 2
    eo = tools.oooxyzt2poooxyzt(qcu_src)     # [2,4,3,X,Y,Z,T//2]
    b_e = eo[0].reshape([12] + ls_odd)
    b_o = eo[1].reshape([12] + ls_odd)
    b = op.give_b_parity(b_e, b_o)
    _, hist = bistabcg_history(b, op.matvec_parity, tol)
    return hist


# ----------------------------------------------------------------------
# 单一配置基准
# ----------------------------------------------------------------------
def bench_one(label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
              MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
              DT=define._LAT_C64_, NV_ITERS=2, gauge_seed=42,
              ntrials_mg=3, ntrials_ref=3):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)            # [X,Y,Z,T//2]
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)

    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    # ---- 参考 BiStabCG（多次计时，取中位数）----
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    ref_times = []
    for _ in range(ntrials_ref):
        fo_ref.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi,
                                   set_ptrs, params)
        torch.cuda.synchronize()
        ref_times.append(time.perf_counter() - t0)
    ref_times_sorted = sorted(ref_times)
    ref_time = ref_times_sorted[len(ref_times_sorted) // 2]

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)

    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity

    # ---- 参考 BiStabCG 收敛历史（同算子复现；按 lattice/precision 缓存）----
    ref_key = (tuple([Lx, Ly, Lz, Lt]), "c128" if dt == torch.complex128 else "c64")
    if ref_key not in _REF_HIST_CACHE:
        _REF_HIST_CACHE[ref_key] = ref_conv_history(op, qcu_src, ATOL)
    ref_hist = _REF_HIST_CACHE[ref_key]
    ref_iters = len(ref_hist) - 1

    # ---- Schur 粗算子（缓存）----
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // MG_GRID[d] for d in range(4)]
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=False)
        set_ptrs[30 + 4 * (lvl - 1) + 0] = lonv.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 1] = hnn.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 2] = hdg.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 3] = sit.contiguous().data_ptr()
        from mg_stencil_build import apply_stencil
        prev = S
        S = lambda v, hnn_i=hnn, hdg_i=hdg, sit_i=sit: apply_stencil(hnn_i, hdg_i, sit_i, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd

    # ---- MG 求解（多次计时，取中位数）----
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    # 清空 C++ 日志，保证本次收敛历史干净
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    mg_times = []
    torch.cuda.synchronize()
    for _ in range(ntrials_mg):
        fo_mg.zero_()
        t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi,
                                    set_ptrs, params)
        torch.cuda.synchronize()
        mg_times.append(time.perf_counter() - t0)
    mg_times_sorted = sorted(mg_times)
    mg_time = mg_times_sorted[len(mg_times_sorted) // 2]

    conv, prof, n_iter = parse_mg_log(LOG_PATH)

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    mg_iters = len(conv) - 1 if conv else (n_iter or 0)

    res = {
        "label": label,
        "lattice": [Lx, Ly, Lz, Lt],
        "precision": "c128" if dt == torch.complex128 else "c64",
        "mass": MASS, "atol": ATOL,
        "levels": NUM_LEVELS, "dof": DOF_LIST, "restart": NUM_RESTART,
        "coarse_tol_factor": COARSE_TOL_FACTOR, "coarse_max_iter": COARSE_MAX_ITER,
        "ref_ms": ref_time * 1000, "ref_iters": ref_iters,
        "ref_res": float(ref_res),
        "mg_ms": mg_time * 1000, "mg_iters": mg_iters,
        "speedup": ref_time / mg_time if mg_time > 0 else 0.0,
        "mg_res": float(mg_res), "vs_ref": float(mg_vs_ref),
        "conv_mg": conv,
        "ref_hist": ref_hist,
        "prof": prof,
    }
    print(f"[{label}] ref={res['ref_ms']:.0f}ms({ref_iters}it) "
          f"mg={res['mg_ms']:.0f}ms({mg_iters}it) "
          f"speedup={res['speedup']:.3f}x mg_res={mg_res:.2e} "
          f"vs_ref={mg_vs_ref:.2e} "
          f"prof={prof}")
    return res


# ----------------------------------------------------------------------
# 扫描配置
# ----------------------------------------------------------------------
CONFIGS = [
    # ---------------- 精度（默认格子 {8,16,16,16}, 2L, r=10, ct=1e5, cmi=15）----------------
    dict(label="8x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c128_2L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C128_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r12_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=12, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    # ---------------- 格子大小（c64, 2L, r=10, ct=1e5, cmi=15）----------------
    dict(label="16x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x32_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=32,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    # ---------------- 求解器参数扫描（默认格子, c64, 2L）----------------
    #  V-cycle 频率 r
    dict(label="8x16x16x16_c64_2L_r5_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=5, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r15_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=15, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r20_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=20, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    #  最粗层收敛条件 ct
    dict(label="8x16x16x16_c64_2L_r10_ct1e2_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e2,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r10_ct1e3_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e3,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r10_ct1e4_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e4,
         DT=define._LAT_C64_, NV_ITERS=2),
    #  最粗层最大迭代 cmi（平滑器/求解迭代上限）
    dict(label="8x16x16x16_c64_2L_r10_ct1e5_cmi50", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=50, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    dict(label="8x16x16x16_c64_2L_r10_ct1e5_cmi200", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=200, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
    #  层数（3L）
    dict(label="8x16x16x16_c64_3L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16,
         MASS=0.05, ATOL=1e-6, NUM_LEVELS=3, DOF_LIST=[12, 48, 48], MG_GRID=[2, 2, 2, 2],
         NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5,
         DT=define._LAT_C64_, NV_ITERS=2),
]


def main():
    only = []
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1:]
    results, ref_conv = [], {}
    for cfg in CONFIGS:
        label = cfg["label"]
        if only and not any(label.startswith(p) for p in only):
            print(f"[skip] {label}")
            continue
        try:
            r = bench_one(**cfg)
            results.append(r)
            # 参考收敛历史按 (lattice,precision) 记录一份即可
            key = (tuple(r["lattice"]), r["precision"])
            if key not in ref_conv:
                # 重新生成（同一 lattice/precision 下相同）
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{label}] FAILED: {e}")

    # 参考收敛历史：每个 (lattice, precision) 保存一份
    for key, hist in _REF_HIST_CACHE.items():
        ref_conv[str(key)] = {"hist": hist, "iters": len(hist) - 1}
    with open(os.path.join(LOG_DIR, "dev73_5_ref_conv.json"), "w") as f:
        json.dump(ref_conv, f, indent=2)

    out = {"results": results}
    with open(os.path.join(LOG_DIR, "dev73_5_bench.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['label']}: {r['speedup']:.3f}x  mg={r['mg_ms']:.0f}ms "
              f"ref={r['ref_ms']:.0f}ms iters={r['mg_iters']}/{r['ref_iters']} "
              f"vs_ref={r['vs_ref']:.2e}")


if __name__ == "__main__":
    main()
