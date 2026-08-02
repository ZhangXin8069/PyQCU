#!/usr/bin/env python3
"""PyQCU CUDA-C++ Clover MultiGrid 性能基准（v4）。

在给定 lattice / precision / MG 参数下，并行测量
  * 参考求解器 : applyCloverBistabCgQcu（奇偶预条件 Clover BiStabCG）
  * MultiGrid   : applyCloverMultigridQcu（Schur-一致 MG，含多层粗层）
输出 ref_time / mg_time / speedup / 细层迭代数 / 收敛残差，并落盘到 logs/。

null_vecs 缓存默认开启（mg_nullvec_cache），对同一 gauge（seed=42）与配置
只计算一次。
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


def bench_one(label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
              NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
              DT=define._LAT_C64_, NV_ITERS=2, gauge_seed=42, ntrials=3):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
                      DT)
    KAPPA = 1.0/(2*MASS+8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2,3,3,4]+ls, dtype=dt, device=device)
    fi = torch.randn([2,4,3]+ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi); fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4,3,4,3]+ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)

    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    # ---- reference BiStabCG ----
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
    torch.cuda.synchronize(); ref_time = time.perf_counter()-t0

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)

    # ---- Schur coarse operators (cached) ----
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    lat_fine_odd = [Lx, Ly, Lz, Lt//2]
    E_prev = 12
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d]//MG_GRID[d] for d in range(4)]
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx,Ly,Lz,Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=True)
        set_ptrs[30+4*(lvl-1)+0] = lonv.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+1] = hnn.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+2] = hdg.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+3] = sit.contiguous().data_ptr()
        # For the NEXT level the "fine operator" is the materialized A_c.
        from mg_stencil_build import apply_stencil
        prev = S
        def make_A(hnn_i, hdg_i, sit_i):
            return lambda v: apply_stencil(hnn_i, hdg_i, sit_i, v)
        S = make_A(hnn, hdg, sit)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd

    # ---- MG solve (multiple trials) ----
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs, params, av)
    mg_times = []
    torch.cuda.synchronize()
    for _ in range(ntrials):
        fo_mg.zero_()
        t0=time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs, params)
        torch.cuda.synchronize()
        mg_times.append(time.perf_counter()-t0)
    mg_time = min(mg_times)

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src)/tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref)/tools.norm(qcu_ref)
    conv = []
    lp = os.path.join(LOG_DIR, "clover_multigrid.log")
    if os.path.exists(lp):
        for line in open(lp):
            m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
            if m: conv = [float(x) for x in m.group(1).split(",") if x.strip()]
    iters = len([c for c in conv if c > ATOL])
    res = {"label": label, "lattice": [Lx,Ly,Lz,Lt], "mass": MASS,
           "levels": NUM_LEVELS, "dof": DOF_LIST,
           "ref_ms": ref_time*1000, "mg_ms": mg_time*1000,
           "speedup": ref_time/mg_time, "iters": iters,
           "mg_res": float(mg_res), "vs_ref": float(mg_vs_ref)}
    print(f"[{label}] ref={res['ref_ms']:.0f}ms mg={res['mg_ms']:.0f}ms "
          f"speedup={res['speedup']:.3f}x iters={iters} "
          f"mg_res={mg_res:.2e} vs_ref={mg_vs_ref:.2e}")
    return res


if __name__ == "__main__":
    # ---- default lattice {8,16,16,16} ----
    results = []
    for cfg in [
        ("8x16x16x16_c64_2L_r10", 8,16,16,16, 0.05, 1e-6, 2, [12,48],  [2,2,2,2], 10, 15, 1e5),
        ("8x16x16x16_c64_3L_r10", 8,16,16,16, 0.05, 1e-6, 3, [12,48,48],[2,2,2,2], 10, 15, 1e5),
        ("8x8x8x16_c64_2L_r10",  8,8,8,16,   0.05, 1e-6, 2, [12,48],  [2,2,2,2], 10, 15, 1e5),
    ]:
        try:
            results.append(bench_one(*cfg))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{cfg[0]}] FAILED: {e}")
    with open(os.path.join(LOG_DIR, "mg_v4_bench_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['label']}: {r['speedup']:.3f}x  mg={r['mg_ms']:.0f}ms "
              f"ref={r['ref_ms']:.0f}ms iters={r['iters']} vs_ref={r['vs_ref']:.2e}")
