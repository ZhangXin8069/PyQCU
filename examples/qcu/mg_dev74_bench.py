#!/usr/bin/env python3
"""dev74 —— MultiGrid 扩展性能基准（dev73_5 进阶版）。

在 dev73_5 协议（参考 BiStabCG vs MultiGrid，精度/格子/求解器参数扫描）基础上：
  * 格子扩展逼近硬件极限：
      - 本地验证（默认）：8x8x8x16 / 8x16x16x16 / 16x16x16x16（小格子）
      - 集群（--cluster）：16x32x32x32 / 16x32x32x64 / 24x32x32x64
        （目标 512G 内存 / 32G 显存服务器；本机仅生成预算预测）
  * 资源占用统计：峰值显存（torch max_memory_allocated）、进程峰值内存（RSS）、
    nullvec 缓存磁盘占用，逐配置记录
  * 粗算子构建可选 C++ CUDA Schur 算子（applyCloverBistabCgDslashQcu，
    --build cpp）：单线程实测比 Python matvec_parity 快 ~2x（多线程仅多卡有效）

用法（${HOME}/PyQCU 下）：
    source ./env.sh && python examples/qcu/mg_dev74_bench.py [--cluster] [--build py|cpp]
输出：logs/dev74_bench.json, logs/dev74_budget_*.json（预算预测）
"""
import torch, os, sys, time, json, re, resource
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
from mg_dev73_5_bench import parse_mg_log, ref_conv_history
from mg_dev74_dslash import make_cuda_schur_ops
from mg_dev74_budget import vram_model, rss_model, disk_cache_bytes

LOG_DIR = os.path.expanduser("~/PyQCU/logs")
LOG_PATH = os.path.join(LOG_DIR, "clover_multigrid.log")

_REF_HIST_CACHE = {}


# ----------------------------------------------------------------------
# 资源统计
# ----------------------------------------------------------------------
def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cache_disk_mb():
    d = os.path.expanduser("~/PyQCU/logs/nullvec_cache")
    total = 0
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


# ----------------------------------------------------------------------
# 单一配置基准（dev73_5 协议 + 资源统计）
# ----------------------------------------------------------------------
def bench_one(label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
              MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
              DT=define._LAT_C64_, NV_ITERS=2, gauge_seed=42,
              ntrials_mg=3, ntrials_ref=3, build_mode="py"):
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
    peak_ref = torch.cuda.max_memory_allocated() / 1e6

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)

    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity

    ref_key = (tuple([Lx, Ly, Lz, Lt]), "c128" if dt == torch.complex128 else "c64")
    if ref_key not in _REF_HIST_CACHE:
        _REF_HIST_CACHE[ref_key] = ref_conv_history(op, qcu_src, ATOL)
    ref_hist = _REF_HIST_CACHE[ref_key]
    ref_iters = len(ref_hist) - 1

    # ---- Schur 粗算子（缓存；build_mode=cpp 时用 C++ dslash 构建）----
    if build_mode == "cpp":
        cpp_ops = make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1)
        S_cpp = cpp_ops[0].matvec
    else:
        cpp_ops = None
        S_cpp = None
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    t_build = 0.0
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // MG_GRID[d] for d in range(4)]
        t0 = time.perf_counter()
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S_cpp or S, dt, device, NV_ITERS, use_cache=True,
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
    peak_build = torch.cuda.max_memory_allocated() / 1e6

    # ---- MG 求解（多次计时，取中位数）----
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
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
    peak_mg = torch.cuda.max_memory_allocated() / 1e6

    conv, prof, n_iter = parse_mg_log(LOG_PATH)

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    mg_iters = len(conv) - 1 if conv else (n_iter or 0)

    if cpp_ops:
        for o in cpp_ops:
            o.release()

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
        # ---- dev74 资源统计 ----
        "build_mode": build_mode,
        "build_s": t_build,
        "peak_vram_ref_mb": round(peak_ref, 1),
        "peak_vram_build_mb": round(peak_build, 1),
        "peak_vram_mg_mb": round(peak_mg, 1),
        "peak_vram_mb": round(peak_mg, 1),
        "rss_kb": rss_kb(),
        "disk_mb": round(cache_disk_mb(), 1),
    }
    print(f"[{label}] ref={res['ref_ms']:.0f}ms({ref_iters}it) "
          f"mg={res['mg_ms']:.0f}ms({mg_iters}it) "
          f"speedup={res['speedup']:.3f}x mg_res={mg_res:.2e} "
          f"vs_ref={mg_vs_ref:.2e} "
          f"vram={res['peak_vram_mb']:.0f}MB rss={res['rss_kb']/1e3:.0f}MB "
          f"disk={res['disk_mb']:.0f}MB build={t_build:.0f}s")
    return res


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
_BASE = dict(MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48],
             MG_GRID=[2, 2, 2, 2], NUM_RESTART=10, COARSE_MAX_ITER=15,
             COARSE_TOL_FACTOR=1e5, DT=define._LAT_C64_, NV_ITERS=2)

LOCAL_CONFIGS = [
    dict(label="8x8x8x16_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=8, Lz=8, Lt=16, **_BASE),
    dict(label="8x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16, **_BASE),
    dict(label="16x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=16, Lz=16, Lt=16, **_BASE),
]

CLUSTER_CONFIGS = [
    dict(label="16x32x32x32_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=32, **_BASE),
    dict(label="16x32x32x64_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=64, **_BASE),
    dict(label="24x32x32x64_c64_2L_r10_ct1e5_cmi15", Lx=24, Ly=32, Lz=32, Lt=64, **_BASE),
]


def main():
    cluster = "--cluster" in sys.argv
    only = []
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1:]
    build_mode = "py"
    if "--build" in sys.argv:
        build_mode = sys.argv[sys.argv.index("--build") + 1]
        assert build_mode in ("py", "cpp")

    # 本机显存探测
    gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
    cfgs = CLUSTER_CONFIGS if cluster else LOCAL_CONFIGS
    results = []
    for cfg in cfgs:
        label = cfg["label"]
        if only and not any(label.startswith(p) for p in only):
            print(f"[skip] {label}")
            continue
        V = cfg["Lx"] * cfg["Ly"] * cfg["Lz"] * cfg["Lt"]
        pred = vram_model(V)
        if pred > gpu_mem_mb * 0.9:
            print(f"[skip] {label}: 预算预测 {pred:.0f}MB > 本机显存 "
                  f"{gpu_mem_mb:.0f}MB×0.9，仅在集群运行（--cluster 服务器）")
            results.append({"label": label, "lattice": [cfg["Lx"], cfg["Ly"],
                            cfg["Lz"], cfg["Lt"]], "skipped": "vram-budget",
                            "pred_vram_mb": round(pred)})
            continue
        try:
            r = bench_one(**cfg, build_mode=build_mode)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{label}] FAILED: {e}")
            results.append({"label": label, "lattice": [cfg["Lx"], cfg["Ly"],
                           cfg["Lz"], cfg["Lt"]], "failed": str(e)})

    for key, hist in _REF_HIST_CACHE.items():
        with open(os.path.join(LOG_DIR, "dev74_ref_conv.json"), "w") as f:
            json.dump({str(key): {"hist": hist, "iters": len(hist) - 1}}, f, indent=2)

    out = {"results": results, "mode": "cluster" if cluster else "local",
           "build_mode": build_mode}
    out_path = os.path.join(LOG_DIR, "dev74_bench.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY ({'CLUSTER' if cluster else 'LOCAL'}, build={build_mode}) ===")
    for r in results:
        if "skipped" in r or "failed" in r:
            print(f"  {r['label']}: {r.get('skipped') or r.get('failed')}")
            continue
        print(f"{r['label']}: {r['speedup']:.3f}x  mg={r['mg_ms']:.0f}ms "
              f"ref={r['ref_ms']:.0f}ms iters={r['mg_iters']}/{r['ref_iters']} "
              f"vs_ref={r['vs_ref']:.2e} vram={r['peak_vram_mb']:.0f}MB "
              f"rss={r['rss_kb']/1e3:.0f}MB")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
