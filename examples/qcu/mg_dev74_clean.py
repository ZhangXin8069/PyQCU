#!/usr/bin/env python3
"""dev74 —— 干净（独占 GPU、独立进程）性能测量 + 资源占用统计。

dev73_5_clean 协议（ref/mg 交叉计时，min of pairs，独立进程）基础上：
  * 资源统计：峰值显存（torch max_memory_allocated）、进程峰值 RSS、
    nullvec 缓存磁盘占用（每配置独立进程 → 每配置 RSS 干净可归因）
  * --build cpp：粗算子构建用 C++ CUDA Schur 算子（applyCloverBistabCgDslashQcu）
  * warm/cold：nullvec 缓存命中（warm）时峰值显存显著低于 cold（首次构建）

用法（每次调用测一个配置，进程隔离；null_vecs 缓存复用）：
    source ./env.sh && CUDA_VISIBLE_DEVICES=2 \
        python examples/qcu/mg_dev74_clean.py \
            --lattice 8 8 8 16 --prec c64 --levels 2 --restart 10 \
            --ct 1e5 --cmi 15 --pairs 5 [--build cpp]
输出：logs/dev74_clean_<label>.json
"""
import torch, os, sys, time, json, resource
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
from mg_dev73_5_bench import ref_conv_history, parse_mg_log
from mg_dev74_dslash import make_cuda_schur_ops

LOG_DIR = "/root/PyQCU/logs"
LOG_PATH = os.path.join(LOG_DIR, "clover_multigrid.log")


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cache_disk_mb():
    d = "/root/PyQCU/logs/nullvec_cache"
    total = 0
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def measure_one(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
                NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT,
                NV_ITERS=2, gauge_seed=42, pairs=5, build_mode="py"):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, NUM_RESTART, COARSE_MAX_ITER,
                      COARSE_TOL_FACTOR, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)
    torch.cuda.reset_peak_memory_stats()

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

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity

    # ---- Schur 粗算子（缓存；--build cpp 用 C++ dslash）----
    if build_mode == "cpp":
        cpp_ops = make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1)
        S_build = cpp_ops[0].matvec
    else:
        cpp_ops = None
        S_build = None
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    t_build = 0.0
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // MG_GRID[d] for d in range(4)]
        t0 = time.perf_counter()
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S_build or S, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=False)
        t_build += time.perf_counter() - t0
        set_ptrs[30 + 4 * (lvl - 1) + 0] = lonv.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 1] = hnn.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 2] = hdg.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 3] = sit.contiguous().data_ptr()
        from mg_stencil_build import apply_stencil
        S = lambda v, hnn_i=hnn, hdg_i=hdg, sit_i=sit: apply_stencil(hnn_i, hdg_i, sit_i, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    peak_cold = torch.cuda.max_memory_allocated() / 1e6

    # ---- 交叉计时：ref, mg, ref, mg, ... ----
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    ref_times, mg_times = [], []
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize()
    for i in range(pairs):
        fo_ref.zero_(); fo_mg.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi,
                                   set_ptrs, params)
        torch.cuda.synchronize()
        ref_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi,
                                    set_ptrs, params)
        torch.cuda.synchronize()
        mg_times.append(time.perf_counter() - t0)
    peak_warm = torch.cuda.max_memory_allocated() / 1e6
    conv_mg, prof, _n_iter = parse_mg_log(LOG_PATH)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
    rmin = min(ref_times); mmin = min(mg_times)
    rmed = sorted(ref_times)[len(ref_times) // 2]
    mmed = sorted(mg_times)[len(mg_times) // 2]
    ref_hist = ref_conv_history(op, qcu_src, ATOL)
    if cpp_ops:
        for o in cpp_ops:
            o.release()
    res = {"lattice": [Lx, Ly, Lz, Lt],
           "precision": "c128" if dt == torch.complex128 else "c64",
           "levels": NUM_LEVELS, "dof": DOF_LIST,
           "restart": NUM_RESTART, "ct": COARSE_TOL_FACTOR,
           "cmi": COARSE_MAX_ITER,
           "ref_times_ms": [t * 1000 for t in ref_times],
           "mg_times_ms": [t * 1000 for t in mg_times],
           "ref_min_ms": rmin * 1000, "mg_min_ms": mmin * 1000,
           "speedup_min": rmin / mmin,
           "ref_med_ms": rmed * 1000, "mg_med_ms": mmed * 1000,
           "speedup_med": rmed / mmed,
           "vs_ref": float(vs_ref), "mg_res": float(mg_res),
           "ref_res": float(ref_res),
           "mg_iters": len(conv_mg) - 1 if conv_mg else 0,
           "ref_iters": len(ref_hist) - 1,
           "conv_mg": conv_mg, "ref_hist": ref_hist, "prof": prof,
           # ---- dev74 资源统计 ----
           "build_mode": build_mode,
           "build_s": t_build,
           "peak_vram_cold_mb": round(peak_cold, 1),
           "peak_vram_warm_mb": round(peak_warm, 1),
           "rss_kb": rss_kb(),
           "disk_mb": round(cache_disk_mb(), 1)}
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", nargs=4, type=int, default=[8, 16, 16, 16])
    ap.add_argument("--prec", default="c64")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--dof", nargs="+", type=int, default=None)
    ap.add_argument("--restart", type=int, default=10)
    ap.add_argument("--ct", type=float, default=1e5)
    ap.add_argument("--cmi", type=int, default=15)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--build", default="py", choices=["py", "cpp"])
    args = ap.parse_args()
    Lx, Ly, Lz, Lt = args.lattice
    DT = define._LAT_C128_ if args.prec == "c128" else define._LAT_C64_
    dof = args.dof or ([12, 48] if args.levels == 2 else [12, 48, 48])
    res = measure_one(Lx, Ly, Lz, Lt, 0.05, 1e-6, args.levels, dof,
                      [2, 2, 2, 2], args.restart, args.cmi, args.ct, DT,
                      NV_ITERS=2, gauge_seed=42, pairs=args.pairs,
                      build_mode=args.build)
    label = (f"L{'x'.join(map(str,[Lx,Ly,Lz,Lt]))}_{args.prec}_"
             f"L{args.levels}_r{args.restart}_ct{args.ct:.0e}_cmi{args.cmi}"
             f"_{args.build}")
    res["label"] = label
    out_path = os.path.join(LOG_DIR, f"dev74_clean_{label}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    try:
        print(json.dumps(res))
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
