#!/usr/bin/env python3
"""
dev84 — 16×32×32×48 统一格子 CUDA C++ 多线程 MultiGrid 稳定 >2 真实加速比套件

判据: speedup_vs_L1 = t(MG_L1) / t(MG_多层) > 2.0 (solve 阶段墙钟, setup 全缓存)
对照: BiStabCG 正确性 | MG 单线程 vs P100x2 并行 | Python 全算子残差
器件: V100-32GB = torch cuda:0 单卡; P100-16GB*2 = torch cuda:1,2 多卡
数据: gauge/nullvec 统一 data/ (seed=42 一一对应), 见 README.md

子命令:
  setup    生成/校验统一 gauge + 高质量 nullvec 缓存 (data/)
  bench    基准: BiStabCG / MG_L1 / MG_多层 (参数化 rs/cf/cmi/nvi/E)
  verify   正确性: rel_vs_ref + Python 全算子残差
  multi    单线程 V100 vs P100x2 双线程并行 + 一致性
  hotspot  torch.profiler 热点剖析
  check    加速比断言 gate>2
  report   汇总报告 + 图表

产物: examples/qcu/dev84/out/ (镜像 logs/dev84/)
"""
import os
import sys
import time
import json
import re
import argparse
import traceback
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(__file__).resolve().parent
OUT_DIR = WORKDIR / "out"
DATA_DIR = ROOT / "data"
LOG_DIR_MIRROR = ROOT / "logs" / "dev84"
for _d in (OUT_DIR, DATA_DIR, LOG_DIR_MIRROR):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["QCU_LOG_DIR"] = str(OUT_DIR)

import torch
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda._multi_gpu import build_schur_levels, MultiGpuMultigrid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

LAT_DEFAULT = [16, 32, 32, 48]
MASS = 0.05
ATOL = 1e-6
DT = define._LAT_C64_
DT_C32 = define._LAT_C32_
MG_GRID = [2, 2, 2, 2]


def gauge_tag(lat, mass, seed):
    return f"gauge_{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_m{mass}_seed{seed}_c64.h5"


def nv_tag(lat, E, nvi, suf=""):
    return f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv1_E{E}_nvi{nvi}{suf}_t1e-2.h5"


def load_nullvec(lat, E, nvi, device, suf=""):
    """加载 data/ 的 33-tensor 粗算子缓存 (lonv/hnn/hdg/sit)。"""
    cf = DATA_DIR / nv_tag(lat, E, nvi, suf)
    if not cf.exists():
        raise RuntimeError(f"nullvec cache missing: {cf} (run setup)")
    return tuple(tools.load_tensor_h5(str(cf), dataset=k, device=device)
                 for k in ("lonv", "hnn", "hdg", "sit"))


def load_gauge(lat, mass, atol, seed=42, verbose=False):
    """加载(或生成)统一 gauge -> data/, 返回 dev80_3 兼容元组。生成设备=当前。"""
    tag = gauge_tag(lat, mass, seed)
    gpath = DATA_DIR / tag
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    if not gpath.exists():
        if verbose:
            print(f"[gauge] MISS {gpath} -> generating")
        p = mod_params.clone(); a = mod_argv.clone(); s = mod_set_ptrs.clone()
        Lx, Ly, Lz, Lt = lat
        p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
        p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
        p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = 1, 1, 1, 1
        p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
        p[define._DATA_TYPE_] = DT
        av = a.to(dtype=define.dtype(DT).to_real())
        av[define._MASS_] = mass; av[define._ATOL_] = atol; av[define._SIGMA_] = 0.1
        ls = define.lat_shape(p)
        dt = define.dtype(DT)
        g = torch.empty([2, 3, 3, 4] + ls, dtype=dt, device=device)
        fi = torch.randn([2, 4, 3] + ls, dtype=dt, device='cpu').to(device)
        p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = -1; p[define._VERBOSE_] = 0; p[define._SEED_] = seed
        qcu.applyInitQcu(s, p, av)
        qcu.applyGaussGaugeQcu(g, s, p)
        p[define._SET_INDEX_] = 0
        qcu.applyEndQcu(s, p)
        import h5py
        with h5py.File(str(gpath), 'w') as f:
            f.create_dataset('g', data=g.detach().cpu().contiguous().numpy())
            f.create_dataset('fi', data=fi.detach().cpu().contiguous().numpy())
        del g, fi
        torch.cuda.empty_cache()
    import h5py
    with h5py.File(str(gpath), 'r') as f:
        g_np = f['g'][...]; fi_np = f['fi'][...]
    if verbose:
        print(f"[gauge] CACHE hit {gpath}")
    g = torch.from_numpy(g_np).to(device=device)
    fi = torch.from_numpy(fi_np).to(device=device)
    del g_np, fi_np
    p = mod_params.clone(); a = mod_argv.clone()
    Lx, Ly, Lz, Lt = lat
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = 1, 1, 1, 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = DT
    p[define._SEED_] = seed
    av = a.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_] = mass; av[define._ATOL_] = atol; av[define._SIGMA_] = 0.1
    ls = define.lat_shape(p)
    s = mod_set_ptrs.clone()
    ce = torch.empty([4, 3, 4, 3] + ls, dtype=torch.complex64, device=device)
    cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
    p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = 2; p[define._PARITY_] = 0
    qcu.applyInitQcu(s, p, av)
    qcu.applyCloversQcu(ce, cei, g, s, p)
    p[define._SET_INDEX_] = 1; p[define._SET_PLAN_] = 2; p[define._PARITY_] = 1
    qcu.applyInitQcu(s, p, av)
    qcu.applyCloversQcu(coo, coi, g, s, p)
    for idx in (0, 1):
        p[define._SET_INDEX_] = idx
        qcu.applyEndQcu(s, p)
    U_full = tools.poooxyzt2oooxyzt(g)
    b_full = tools.poooxyzt2oooxyzt(fi)
    kappa = 1.0 / (2 * mass + 8)
    clover_full = dslash.make_clover(U_full, kappa=kappa)
    return g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av, p


def ensure_nullvec(lat, E, nvi, op, S, device, verbose=False, timeout_s=600,
                   cpp_ctx=None, gen="invit", suf=""):
    """确保 data/ 有高质量 nullvec 缓存 (33-tensor stencil), miss 则构建。

    cpp_ctx 给定时走 C++ Schur matvec 逐向量路径 (CudaSchurOp, ~13ms/matvec,
    分钟级)；否则批量 Python einsum 路径 (大格子 ~108s/迭代, 仅小格子可用)。
    gen="invit": 原逆迭代语义 (nv_tol 为绝对容差 — 2026-08-22 dev84 诊断发现
    该路径在本格子生成近似随机噪声向量: ‖Sv‖/‖v‖≈0.4≈谱 RMS, ρ_V≈0.976)。
    gen="ddamg": DDalphaAMG 式测试向量 — 松相对容差 (1e-1 rtol) 单次近似逆
    S x = randn, x≈S⁻¹r 按谱放大低模, 归一化后即近零模富集向量。
    """
    dof_list = [12, E]
    lat_fine_odd = [lat[0], lat[1], lat[2], lat[3] // 2]
    tag = nv_tag(lat, E, nvi, suf)
    cf = DATA_DIR / tag
    if cf.exists() and verbose:
        print(f"[nullvec] CACHE hit {cf}", flush=True)
        return
    t0 = time.perf_counter()
    print(f"[nullvec] MISS {tag} -> building gen={gen} nvi={nvi} "
          f"({'C++ matvec' if cpp_ctx else 'batch einsum'})", flush=True)
    import h5py
    if cpp_ctx is not None and gen == "ddamg":
        from pyqcu.tools._multigrid import local_orthogonalize
        from pyqcu.cuda._schur_op import CudaSchurOp
        from pyqcu import solver as _sol
        torch.cuda.set_device(0)
        cu_op = CudaSchurOp(cpp_ctx["av"], cpp_ctx["g"], cpp_ctx["ce"],
                            cpp_ctx["coo"], cpp_ctx["cei"], cpp_ctx["coi"],
                            params=cpp_ctx["params"])
        shape = [12] + list(lat_fine_odd)
        dt = define.dtype(DT)
        genv = torch.Generator(device=device).manual_seed(42)
        vs = []
        for i in range(E):
            r = torch.randn(shape, dtype=dt, device=device, generator=genv)
            x = _sol.bistabcg(b=r.contiguous(), matvec=cu_op.matvec,
                              tol=1e-3, if_rtol=True, max_iter=600,
                              verbose=False)
            vs.append(x / torch.linalg.norm(x))
            if verbose:
                sr = float(tools.norm(cu_op.matvec(vs[-1])))
                print(f"  [ddamg] vec{i}: ||Sv||/||v||={sr:.3e}", flush=True)
        _null = torch.stack(vs).contiguous()
        cu_op.release()
    elif cpp_ctx is not None:
        from pyqcu.tools._multigrid import local_orthogonalize
        from pyqcu.tools import give_null_vecs_mt as _gnv_mt_wrap
        from pyqcu.cuda._schur_op import CudaSchurOp
        torch.cuda.set_device(0)
        cu_op = CudaSchurOp(cpp_ctx["av"], cpp_ctx["g"], cpp_ctx["ce"],
                            cpp_ctx["coo"], cpp_ctx["cei"], cpp_ctx["coi"],
                            params=cpp_ctx["params"])
        _null = _gnv_mt_wrap([cu_op], E, 12, lat_fine_odd, define.dtype(DT),
                             device, nv_iters=nvi, nthreads=1, verbose=False,
                             nv_tol=3e-2)
        cu_op.release()
    else:
        from pyqcu.tools._multigrid import (give_null_vecs_mt, BatchedLocalSchur,
                                            _schur_matvec_batch)
        batch_mv = lambda x, _op=op: _schur_matvec_batch(_op, x)
        _null = give_null_vecs_mt(None, E, 12, lat_fine_odd, define.dtype(DT),
                                  device, nv_iters=nvi, nthreads=1, verbose=False,
                                  nv_tol=1e-2, batch_matvec=batch_mv)
    from pyqcu.tools._multigrid import local_orthogonalize, build_stencil_local, BatchedLocalSchur
    lonv = local_orthogonalize(null_vecs=_null, coarse_lat_size=[d // 2 for d in lat_fine_odd],
                               verbose=False)
    del _null
    lsch = BatchedLocalSchur(op, *lat_fine_odd, W=10)
    hnn, hdg, sit = build_stencil_local(lsch, lonv, E, lat_fine_odd,
                                        [d // 2 for d in lat_fine_odd],
                                        define.dtype(DT), device, verbose=verbose)
    tmp = str(cf) + ".tmp.h5"
    with h5py.File(tmp, 'w') as f:
        for key, tt in (("lonv", lonv), ("hnn", hnn), ("hdg", hdg), ("sit", sit)):
            f.create_dataset(key, data=tt.detach().cpu().contiguous().numpy())
    os.replace(tmp, str(cf))
    # 兼容 build_schur_levels/MultiGpuMultigrid 两套 tag 命名 (t0.01 vs t1e-2)
    alt = DATA_DIR / f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv1_E{E}_nvi{nvi}{suf}_t0.01.h5"
    try:
        if not alt.exists():
            os.symlink(str(cf), str(alt))
    except Exception:
        pass
    print(f"[nullvec] built in {time.perf_counter()-t0:.1f}s -> {cf}", flush=True)
    del lsch
    torch.cuda.empty_cache()


def solve_bistabcg(g, fi, ce, cei, coo, coi, lat, mass, atol, timeout=300, max_iter=1000):
    """参考求解器 applyCloverBistabCgQcu。返回 (fo, t, stat, err)。"""
    def _run():
        p = mod_params.clone(); a = mod_argv.clone(); s = mod_set_ptrs.clone()
        Lx, Ly, Lz, Lt = lat
        p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
        p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
        p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = 1, 1, 1, 1
        p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
        p[define._DATA_TYPE_] = DT
        p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = 1; p[define._VERBOSE_] = 0
        p[define._MAX_ITER_] = max_iter
        av = a.to(dtype=define.dtype(DT).to_real())
        av[define._MASS_] = mass; av[define._ATOL_] = atol; av[define._SIGMA_] = 0.1
        qcu.applyInitQcu(s, p, av)
        fo = torch.empty_like(fi)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo, fi, g, ce, coo, cei, coi, s, p)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        p[define._SET_INDEX_] = 0
        qcu.applyEndQcu(s, p)
        return fo, t
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            fo, t = fut.result(timeout=timeout)
            return fo, t, "OK", None
        except TimeoutError:
            return None, float(timeout), "TIMEOUT", "BiStabCG timeout >%ds" % timeout
        except Exception:
            return None, 0.0, "FAIL", traceback.format_exc()


def parse_last_log(patterns=("CONVERGENCE_HISTORY", "PROF_SECTIONS")):
    """解析 out/clover_multigrid.log 最后一条匹配行。"""
    lp = OUT_DIR / "clover_multigrid.log"
    found = {k: None for k in patterns}
    if not lp.exists():
        return found
    try:
        with open(lp) as f:
            for line in f:
                for k in patterns:
                    m = re.search(k + r':\s*(.*)', line)
                    if m:
                        found[k] = m.group(1).strip()
    except Exception:
        pass
    conv = []
    if found["CONVERGENCE_HISTORY"]:
        try:
            inner = found["CONVERGENCE_HISTORY"].strip().strip("[]")
            conv = [float(x) for x in inner.split(",") if x.strip()]
        except Exception:
            pass
    prof = {}
    if found["PROF_SECTIONS"]:
        for kv in re.findall(r'(\w+)=([0-9.]+)', found["PROF_SECTIONS"]):
            try:
                prof[kv[0]] = float(kv[1])
            except ValueError:
                pass
        mn = re.search(r'n_vcycles=(\d+)', found["PROF_SECTIONS"])
        if mn:
            prof["n_vcycles"] = int(mn.group(1))
    return {"conv": conv, "prof": prof}


def solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, mass, atol,
             num_levels, E, device_gen, timeout=300, verbose=False,
             rs=5, cf=3e3, cmi=200, nvi=20, mp=False, gcr=False, deflate=False,
             nvsuf=""):
    """applyCloverMultigridQcu (num_levels 层)。setup 走 data/ 缓存, 只计 solve 墙钟。
    gcr=True 时 params[_MG_USE_GCR_]=1 → C++ run_gcr (FGMRES(10)⊕V-cycle 预条件子, quda 式)。
    返回 (fo, t_solve, conv, prof, stat, err)。"""
    def _run():
        nonlocal U_full, clover_full
        p = mod_params.clone(); a = mod_argv.clone(); s = mod_set_ptrs.clone()
        Lx, Ly, Lz, Lt = lat
        p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
        p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
        p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = 1, 1, 1, 1
        p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
        p[define._DATA_TYPE_] = DT
        av = a.to(dtype=define.dtype(DT).to_real())
        av[define._MASS_] = mass; av[define._ATOL_] = atol; av[define._SIGMA_] = 0.1
        p[define._MG_NUM_LEVEL_] = num_levels
        p[define._MG_USE_GCR_] = 1 if gcr else 0
        p[define._MG_USE_DEFLATE_] = 1 if deflate else 0
        p[define._MG_MU_PRE_] = 4
        if num_levels >= 2:
            p[define._MG_LEVEL1_E_] = E
            p[define._MG_LEVEL1_X_] = Lx // MG_GRID[0]
            p[define._MG_LEVEL1_Y_] = Ly // MG_GRID[1]
            p[define._MG_LEVEL1_Z_] = Lz // MG_GRID[2]
            p[define._MG_LEVEL1_T_] = Lt // (2 * MG_GRID[3])
            p[define._MG_LEVEL1_MAX_ITER_] = cmi
            p[define._MG_LEVEL1_DATA_TYPE_] = DT_C32 if mp else DT
            p[define._MG_LEVEL1_NUM_RESTART_] = rs
            av[define._MG_LEVEL1_ATOL_] = atol * cf
        # --- setup: nullvec/stencil 缓存命中或构建 (不计入 solve 时间) ---
        if num_levels >= 2:
            lonv, hnn, hdg, sit = load_nullvec(lat, E, nvi, torch.device("cuda:0"), suf=nvsuf)
            del U_full, clover_full
            import gc
            gc.collect(); torch.cuda.empty_cache()
            s[30 + 0] = lonv.contiguous().data_ptr()
            s[30 + 1] = hnn.contiguous().data_ptr()
            s[30 + 2] = hdg.contiguous().data_ptr()
            s[30 + 3] = sit.contiguous().data_ptr()
        # --- solve (计时段) ---
        p[define._SET_INDEX_] = 0; p[define._SET_PLAN_] = 1
        p[define._VERBOSE_] = 1 if verbose else 0
        qcu.applyInitQcu(s, p, av)
        fo = torch.empty_like(fi)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo, fi, g, ce, coo, cei, coi, s, p)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        parsed = parse_last_log()
        p[define._SET_INDEX_] = 0
        qcu.applyEndQcu(s, p)
        return fo, t, parsed["conv"], parsed["prof"]
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            fo, t, conv, prof = fut.result(timeout=timeout)
            return fo, t, conv, prof, "OK", None
        except TimeoutError:
            return None, float(timeout), [], {}, "TIMEOUT", "MG %dL timeout >%ds" % (num_levels, timeout)
        except Exception:
            return None, 0.0, [], {}, "FAIL", traceback.format_exc()


def full_residual_rel(x_full_dev, g, fi, kappa):
    """Python 全算子相对残差 ||D x - b||/||b|| (奇偶掩码布局)。"""
    U = tools.poooxyzt2oooxyzt(g)
    src = tools.poooxyzt2oooxyzt(fi)
    out = tools.poooxyzt2oooxyzt(x_full_dev)
    cl = dslash.make_clover(U, kappa=kappa)
    r = tools.norm(dslash.give_wilson(out, U, kappa, True) + dslash.give_clover(out, cl) - src)
    return float(r / tools.norm(src))


def cmd_setup(args):
    lat = [int(x) for x in args.lat.split(",")]
    g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av, p = \
        load_gauge(lat, args.mass, args.atol, verbose=True)
    if args.E > 0:
        device = torch.device("cuda:0")
        # dev84 指令23: 大体量 setup 先释放非 CudaSchurOp 所需资产
        del b_full
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
        op = dslash.operator(U=U_full, clover_term=clover_full,
                             kappa=torch.Tensor([kappa]), support_parity=True, verbose=False)
        S = op.matvec_parity
        del fi
        _gc.collect(); torch.cuda.empty_cache()
        ensure_nullvec(lat, args.E, args.nvi, op, S, device, verbose=True,
                       cpp_ctx=dict(av=av, g=g, ce=ce, coo=coo, cei=cei, coi=coi, params=p),
                       gen=args.gen, suf=args.nvsuf)
        del op, S
    print("[setup] done")


def cmd_bench(args):
    lat = [int(x) for x in args.lat.split(",")]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    torch.cuda.set_device(0)
    print(f"=== dev84 bench {lat} levels={levels} rs={args.rs} cf={args.cf} "
          f"cmi={args.cmi} nvi={args.nvi} E={args.E} ===")
    print(f"OUT_DIR={OUT_DIR} DATA_DIR={DATA_DIR}")
    print(f"device0 = {torch.cuda.get_device_name(0)}")
    logp = OUT_DIR / "clover_multigrid.log"
    if logp.exists():
        logp.unlink()
    g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av, p = \
        load_gauge(lat, args.mass, args.atol, verbose=True)
    results = []
    # 1) BiStabCG 参考
    print("\n[1/3] BiStabCG reference ...")
    fo_ref, t_ref, st_ref, err_ref = solve_bistabcg(g, fi, ce, cei, coo, coi,
                                                    lat, args.mass, args.atol, timeout=args.timeout)
    ref_res = None
    if fo_ref is not None:
        ref_res = full_residual_rel(fo_ref, g, fi, kappa)
        print(f"  BiStabCG t={t_ref:.3f}s full_res={ref_res:.2e}")
        results.append({"label": "BiStabCG", "t": t_ref, "res": ref_res, "stat": st_ref})
    else:
        print(f"  BiStabCG {st_ref}: {err_ref}")
        results.append({"label": "BiStabCG", "t": None, "stat": st_ref, "err": err_ref})
    # 2) MG 各层
    for nl in levels:
        dof = [12] + [args.E] * (nl - 1)
        print(f"\n[{nl}] MG_{nl}L dof={dof} ...")
        fo_mg, t_mg, conv, prof, st_mg, err_mg = solve_mg(
            g, fi, ce, cei, coo, coi, U_full, clover_full, lat, args.mass, args.atol,
            nl, args.E, torch.device("cuda:0"), timeout=args.timeout, verbose=args.verbose,
            rs=args.rs, cf=args.cf, cmi=args.cmi, nvi=args.nvi,
            mp=args.mp, gcr=args.gcr, deflate=args.deflate, nvsuf=args.nvsuf)
        label = f"MG_{nl}L" + ("_GCR" if args.gcr else "") + ("_D" if args.deflate else "")
        if fo_mg is None:
            print(f"  {label} {st_mg}: {err_mg}")
            results.append({"label": label, "t": None, "stat": st_mg, "err": err_mg})
            continue
        mg_res = full_residual_rel(fo_mg, g, fi, kappa)
        rel = None
        if fo_ref is not None:
            rel = float(tools.norm(tools.poooxyzt2oooxyzt(fo_mg) -
                                   tools.poooxyzt2oooxyzt(fo_ref)) /
                       tools.norm(tools.poooxyzt2oooxyzt(fo_ref)))
        print(f"  {label} t={t_mg:.3f}s iters={len(conv)} full_res={mg_res:.2e} "
              f"rel_vs_ref={rel if rel is None else format(rel, '.2e')} "
              f"prof={prof}")
        entry = {"label": label, "t": t_mg, "iters": len(conv), "res": mg_res,
                 "rel_vs_ref": rel, "stat": st_mg, "prof": prof}
        suffix = "_GCR" if args.gcr else ""
        if nl == 1 and len(conv) > 0:
            np.savetxt(OUT_DIR / f"conv_1L{suffix}.txt", np.array(conv))
        if nl >= 2 and len(conv) > 0:
            np.savetxt(OUT_DIR / f"conv_{nl}L{suffix}.txt", np.array(conv))
        results.append(entry)
    # 3) 加速比汇总 (基准=L1)
    t_l1 = next((r["t"] for r in results if r["label"] == "MG_1L" and r["t"]), None)
    print("\n=== SUMMARY ===")
    best = 0.0
    if t_l1:
        for r in results:
            if r.get("t") and r["label"].startswith("MG"):
                sp = t_l1 / r["t"]
                r["speedup_vs_L1"] = sp
                print(f"  {r['label']:8s} t={r['t']:.3f}s speedup_vs_L1={sp:.3f}x")
                if r["label"] != "MG_1L":
                    best = max(best, sp)
    if fo_ref is not None:
        for r in results:
            if r.get("t") and r["label"].startswith("MG"):
                r["speedup_vs_BiStabCG"] = t_ref / r["t"]
    verdict = "PASS" if best > 2.0 else "FAIL"
    print(f"\nbest speedup_vs_L1 = {best:.3f}  target>2 -> {verdict}")
    report = {"lat": lat, "mass": args.mass, "atol": args.atol, "E": args.E,
              "nvi": args.nvi, "rs": args.rs, "cf": args.cf, "cmi": args.cmi,
              "results": results, "best_speedup_vs_L1": best, "verdict": verdict,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(OUT_DIR / "bench_out.txt", "w") as f:
        f.write(f"{json.dumps({k: report[k] for k in ('lat','rs','cf','cmi','nvi','E','best_speedup_vs_L1','verdict')})}\n")
        for r in results:
            f.write(json.dumps(r) + "\n")


def cmd_verify(args):
    """正确性: 重跑 bench 的解对比已在 bench 内做; 此处对已有 report.json 断言。"""
    rp = OUT_DIR / "report.json"
    if not rp.exists():
        print("missing report.json — run bench first"); sys.exit(2)
    j = json.load(open(rp))
    ok = True
    for r in j["results"]:
        if r["label"] == "BiStabCG" or not r.get("res"):
            continue
        res_ok = r["res"] < 10 * ATOL
        rel_ok = r.get("rel_vs_ref") is None or r["rel_vs_ref"] < 1e-4
        print(f"{r['label']}: full_res={r.get('res'):.2e} (<{10*ATOL:.0e} {res_ok}) "
              f"rel_vs_ref={r.get('rel_vs_ref')} ({rel_ok})")
        ok &= res_ok and rel_ok
    print("VERIFY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_multi(args):
    lat = [int(x) for x in args.lat.split(",")]
    nl = int(args.levels)
    dof = [12] + [args.E] * (nl - 1)
    common = dict(lat_size=lat, mass=args.mass, atol=args.atol, num_levels=nl,
                  dof_list=dof, mg_grid=MG_GRID, num_restart=args.rs,
                  coarse_max_iter=args.cmi, coarse_tol_factor=args.cf,
                  nv_iters=args.nvi, use_cache=True, cache_dir=str(DATA_DIR), verbose=False)
    print(f"=== dev84 multi: single V100 vs P100x2 ({lat}, {nl}L, dof={dof}) ===")
    mg_s = MultiGpuMultigrid(nthreads=1, device_ids=[0], **common)
    rs_ = mg_s.solve()
    t_single = max(t['mg_time'] for t in rs_['threads'])
    t_single_ref = max(t['ref_time'] for t in rs_['threads'])
    cons_s = mg_s.verify_consistency(tol=1e-5)
    print(f"single V100 : mg_wall={t_single:.3f}s bistabcg_wall={t_single_ref:.3f}s "
          f"consistency={'PASS' if cons_s['all_pass'] else 'FAIL'}")
    row = {"single_v100_mg": t_single, "single_v100_bistabcg": t_single_ref,
           "single_consistency": cons_s}
    try:
        mg_m = MultiGpuMultigrid(nthreads=2, device_ids=[1, 2], **common)
        rm = mg_m.solve()
        t_multi = max(t['mg_time'] for t in rm['threads'])
        t_multi_ref = max(t['ref_time'] for t in rm['threads'])
        cons_m = mg_m.verify_consistency(tol=1e-5)
        par = t_single / t_multi
        print(f"multi P100x2: mg_wall={t_multi:.3f}s bistabcg_wall={t_multi_ref:.3f}s "
              f"consistency={'PASS' if cons_m['all_pass'] else 'FAIL'}")
        print(f"parallel speedup (single/multi) = {par:.3f}x")
        row.update({"multi_p100x2_mg": t_multi, "multi_p100x2_bistabcg": t_multi_ref,
                    "multi_consistency": cons_m, "parallel_speedup_mg": par})
    except Exception as e:
        traceback.print_exc()
        row.update({"multi_error": str(e)})
    with open(OUT_DIR / "multi_report.json", "w") as f:
        json.dump(row, f, indent=2, default=str)


def cmd_hotspot(args):
    lat = [int(x) for x in args.lat.split(",")]
    torch.cuda.set_device(0)
    import torch.profiler as profiler
    print(f"=== dev84 hotspot {lat} on {torch.cuda.get_device_name(0)} ===")
    g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av, p = \
        load_gauge(lat, args.mass, args.atol)
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU,
                                      profiler.ProfilerActivity.CUDA],
                          record_shapes=True) as prof:
        solve_bistabcg(g, fi, ce, cei, coo, coi, lat, args.mass, args.atol, timeout=300)
    prof.export_chrome_trace(str(OUT_DIR / "trace_bistabcg.json"))
    print(f"[hotspot] trace_bistabcg.json saved")
    g2, fi2, ce2, cei2, coo2, coi2, U2, b2, cl2, k2, av2, p2 = \
        load_gauge(lat, args.mass, args.atol)
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU,
                                      profiler.ProfilerActivity.CUDA],
                          record_shapes=True) as prof2:
        solve_mg(g2, fi2, ce2, cei2, coo2, coi2, U2, cl2, lat, args.mass, args.atol,
                 2, args.E, torch.device("cuda:0"), timeout=600,
                 rs=args.rs, cf=args.cf, cmi=args.cmi, nvi=args.nvi, verbose=False)
    prof2.export_chrome_trace(str(OUT_DIR / "trace_mg2l.json"))
    print("[hotspot] trace_mg2l.json saved")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                              "--format=csv"], capture_output=True, text=True, timeout=10)
        (OUT_DIR / "hotspot_smi.txt").write_text(out.stdout)
    except Exception as e:
        print(f"nvidia-smi failed {e}")


def cmd_check(args):
    rp = OUT_DIR / "report.json"
    if not rp.exists():
        print(f"missing {rp}, run bench first"); sys.exit(2)
    j = json.load(open(rp))
    best = j.get("best_speedup_vs_L1", 0)
    gate = args.gate
    print(f"best speedup_vs_L1={best:.3f} gate={gate} -> {'PASS' if best >= gate else 'FAIL'}")
    sys.exit(0 if best >= gate else 1)


def cmd_report(args):
    rp = OUT_DIR / "report.json"
    if rp.exists():
        print(json.dumps(json.load(open(rp)), indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rp.exists():
            j = json.load(open(rp))
            items = [(r["label"], r["t"]) for r in j["results"] if r.get("t")]
            plt.figure(figsize=(8, 4))
            plt.bar([k for k, _ in items], [v for _, v in items])
            plt.ylabel("time (s)")
            plt.title(f"dev84 {j['lat']} best_vs_L1 {j['best_speedup_vs_L1']:.2f}x [{j['verdict']}]")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bench_bar.png", dpi=150)
        for cf in OUT_DIR.glob("conv_*.txt"):
            d = np.loadtxt(cf)
            plt.figure()
            plt.semilogy(np.maximum(d, 1e-12))
            plt.title(cf.stem)
            plt.savefig(OUT_DIR / (cf.stem + ".png"), dpi=150)
        print("[report] plots saved to", OUT_DIR)
    except Exception as e:
        print(f"plot failed {e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    for name, extra in (("setup", []), ("bench", []), ("verify", []), ("multi", []),
                        ("hotspot", []), ("check", []), ("report", [])):
        sp = sub.add_parser(name)
        sp.add_argument("--lat", type=str, default="16,32,32,48")
        sp.add_argument("--mass", type=float, default=MASS)
        sp.add_argument("--atol", type=float, default=ATOL)
        sp.add_argument("--timeout", type=int, default=300)
        sp.add_argument("--levels", type=str, default="1,2")
        sp.add_argument("--E", type=int, default=12)
        sp.add_argument("--nvi", type=int, default=20)
        sp.add_argument("--rs", type=int, default=5)
        sp.add_argument("--cf", type=float, default=3e3)
        sp.add_argument("--cmi", type=int, default=200)
        sp.add_argument("--gcr", action="store_true",
                        help="C++ run_gcr: FGMRES(10) ⊕ V-cycle 预条件子 (quda 式)")
        sp.add_argument("--mp", action="store_true",
                        help="粗层 fp32 (c32), 减半粗算子带宽")
        sp.add_argument("--gen", type=str, default="invit",
                        choices=["invit", "ddamg"],
                        help="nullvec 生成配方: invit=原逆迭代(绝对tol,噪声) / "
                             "ddamg=DDalphaAMG式松相对容差近似逆 (dev84)")
        sp.add_argument("--nvsuf", type=str, default="",
                        help="nullvec 缓存 tag 后缀 (如 _dd)")
        sp.add_argument("--deflate", action="store_true",
                        help="收缩启动: 一次粗校正作初值, 不重置 Krylov (dev84 R5)")
        if name == "check":
            sp.add_argument("--gate", type=float, default=2.0)
        sp.add_argument("--verbose", action="store_true")
        sp.set_defaults(func=globals()[f"cmd_{name}"])
    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
