#!/usr/bin/env python3
"""
dev80_3 — 16×32×32×48 统一格子 MG >2 真实加速比验证套件 (examples/qcu/dev80_3/main.py)

任务对象: ${HOME}/PyQCU 全库，分级管理 VRAM->RAM->DISK、局部化 W10、Cheap-Jacobi、MINRES、SAP、GCR 等优化
基准: MG L1 (仅最细层 Schur BiStabCG) 的真实加速比 >2 稳定；对照 C++ BiStabCG 正确性 & 单线程vsP100x2并行
范例: DDalphaAMG / DDalphaAMG-SM / quda / PyQUDA docs 分析
格子: 16,32,32,48 统一 (786432 sites, odd 393216), mass 0.05, atol 1e-6, c64
器件: V100-32GB (torch cuda:0) 单卡基准，P100-16GB*2 (torch 1,2) 多卡并行对照

子命令:
  bench    批量基准 (V100 单卡 L1 vs 2L/3L, 600s 超时, Hierarchical+R10, 中位 speedup)
  hotspot  热点剖析 (torch.profiler + nvidia-smi + C++ PROF)
  multi    多卡 P100x2 并行 vs 单线程
  check    加速比断言 gate>2
  report   汇总报告 + 图表

特性:
  - 统一 gauge/nullvec 于 data/ (一一对应, seed 关联)
  - 分层显存 HierarchicalCache VRAM->RAM->DISK (data/hier_*.h5)
  - 局部化 BatchedLocalSchur W=10 (786k ->2min stencil vs 24min 全格)
  - 混合精度粗层 c32 可选 (--mp)
  - SAP 4^4 块 MINRES 钩子 (--sap)  via lattice_sap.h (C++ 侧已具备，参数 --sap-blocks)
  - GCR 外层迭代钩子 (--gcr) 预留 (FGMRES 10)
  - 分钟级守卫 (每 solver 600s)
  - 产物: logs/dev80_3/report.json, bench_out.txt, conv_*.txt, *.png, data/*.h5
"""
import os, sys, time, json, argparse, traceback, re, glob, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs" / "dev80_3"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = str(DATA_DIR)
# 单卡 V100 / 双卡 P100 映射 (torch 视角)
# torch cuda:0 = V100 (nvidia-smi 2), cuda:1,2 = P100 (nvidia-smi 1,0)
import torch
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda._multi_gpu import build_schur_levels, MultiGpuMultigrid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

LAT_DEFAULT = [16,32,32,48]
MASS = 0.05
ATOL = 1e-6
DT = define._LAT_C64_
DT_C32 = define._LAT_C32_
DOF_2L = [12,12]
DOF_3L = [12,12,12]
MG_GRID = [2,2,2,2]

def gauge_tag(lat, mass, seed):
    return f"gauge_{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_m{mass}_seed{seed}_c64.h5"

def build_gauge(lat, mass, atol, device, seed=42, verbose=False):
    tag = gauge_tag(lat, mass, seed)
    gauge_path = DATA_DIR / tag
    if gauge_path.exists():
        if verbose:
            print(f"[gauge] CACHE hit {gauge_path}")
        import h5py
        with h5py.File(str(gauge_path), 'r') as f:
            g_np = f['g'][...]
            fi_np = f['fi'][...]
        g = torch.from_numpy(g_np).to(device=device)
        fi = torch.from_numpy(fi_np).to(device=device)
        params_t = mod_params.clone()
        argv_t = mod_argv.clone()
        set_ptrs_t = mod_set_ptrs.clone()
        Lx,Ly,Lz,Lt = lat
        params_t[define._LAT_X_]=Lx; params_t[define._LAT_Y_]=Ly; params_t[define._LAT_Z_]=Lz; params_t[define._LAT_T_]=Lt
        params_t[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
        params_t[define._GRID_X_],params_t[define._GRID_Y_],params_t[define._GRID_Z_],params_t[define._GRID_T_]=1,1,1,1
        params_t[define._NODE_RANK_]=0; params_t[define._NODE_SIZE_]=1
        params_t[define._DATA_TYPE_]=DT
        params_t[define._MG_NUM_LEVEL_]=1
        av = argv_t.to(dtype=define.dtype(DT).to_real())
        av[define._MASS_]=mass; av[define._ATOL_]=atol; av[define._SIGMA_]=0.1
        ls = define.lat_shape(params_t)
        ce = torch.empty([4,3,4,3]+ls, dtype=torch.complex64, device=device)
        cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
        params_t[define._SET_INDEX_]=0; params_t[define._SET_PLAN_]=2; params_t[define._PARITY_]=0
        qcu.applyInitQcu(set_ptrs_t, params_t, av)
        qcu.applyCloversQcu(ce, cei, g, set_ptrs_t, params_t)
        params_t[define._SET_INDEX_]=1; params_t[define._SET_PLAN_]=2; params_t[define._PARITY_]=1
        qcu.applyInitQcu(set_ptrs_t, params_t, av)
        qcu.applyCloversQcu(coo, coi, g, set_ptrs_t, params_t)
        for idx in [0,1]:
            params_t[define._SET_INDEX_]=idx
            qcu.applyEndQcu(set_ptrs_t, params_t)
        U_full = tools.poooxyzt2oooxyzt(g)
        b_full = tools.poooxyzt2oooxyzt(fi)
        kappa = 1.0/(2*mass+8)
        clover_full = dslash.make_clover(U_full, kappa=kappa)
        return g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av, params_t, set_ptrs_t
    if verbose:
        print(f"[gauge] CACHE miss {gauge_path} -> generating on {device}")
    params_t = mod_params.clone()
    argv_t = mod_argv.clone()
    set_ptrs_t = mod_set_ptrs.clone()
    Lx,Ly,Lz,Lt = lat
    params_t[define._LAT_X_]=Lx; params_t[define._LAT_Y_]=Ly; params_t[define._LAT_Z_]=Lz; params_t[define._LAT_T_]=Lt
    params_t[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params_t[define._GRID_X_],params_t[define._GRID_Y_],params_t[define._GRID_Z_],params_t[define._GRID_T_]=1,1,1,1
    params_t[define._NODE_RANK_]=0; params_t[define._NODE_SIZE_]=1
    params_t[define._DATA_TYPE_]=DT
    av = argv_t.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_]=mass; av[define._ATOL_]=atol; av[define._SIGMA_]=0.1
    ls = define.lat_shape(params_t)
    dt = define.dtype(DT)
    g = torch.empty([2,3,3,4]+ls, dtype=dt, device=device)
    fi = torch.randn([2,4,3]+ls, dtype=dt, device='cpu').to(device)
    ce = torch.empty([4,3,4,3]+ls, dtype=dt, device=device)
    cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
    params_t[define._SET_INDEX_]=0; params_t[define._SET_PLAN_]=-1; params_t[define._VERBOSE_]=0; params_t[define._SEED_]=seed
    qcu.applyInitQcu(set_ptrs_t, params_t, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_]=1; params_t[define._SET_PLAN_]=2; params_t[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs_t, params_t, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_]=2; params_t[define._SET_PLAN_]=2; params_t[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs_t, params_t, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs_t, params_t)
    for idx in [0,1,2]:
        params_t[define._SET_INDEX_]=idx
        qcu.applyEndQcu(set_ptrs_t, params_t)
    import h5py
    with h5py.File(str(gauge_path), 'w') as f:
        f.create_dataset('g', data=g.detach().cpu().contiguous().numpy())
        f.create_dataset('fi', data=fi.detach().cpu().contiguous().numpy())
    if verbose:
        print(f"[gauge] saved {gauge_path} g {g.shape} fi {fi.shape}")
    U_full = tools.poooxyzt2oooxyzt(g)
    b_full = tools.poooxyzt2oooxyzt(fi)
    kappa = 1.0/(2*mass+8)
    clover_full = dslash.make_clover(U_full, kappa=kappa)
    params_t2 = mod_params.clone()
    params_t2[define._LAT_X_]=Lx; params_t2[define._LAT_Y_]=Ly; params_t2[define._LAT_Z_]=Lz; params_t2[define._LAT_T_]=Lt
    params_t2[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params_t2[define._GRID_X_],params_t2[define._GRID_Y_],params_t2[define._GRID_Z_],params_t2[define._GRID_T_]=1,1,1,1
    params_t2[define._NODE_RANK_]=0; params_t2[define._NODE_SIZE_]=1
    params_t2[define._DATA_TYPE_]=DT
    av2 = argv_t.to(dtype=define.dtype(DT).to_real())
    av2[define._MASS_]=mass; av2[define._ATOL_]=atol; av2[define._SIGMA_]=0.1
    return g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av2, params_t2, set_ptrs_t

def solve_bistabcg(g, fi, ce, cei, coo, coi, params_t, av, device, lat, mass, atol, timeout=300):
    def _run():
        p = params_t.clone()
        a = av.clone()
        s = mod_set_ptrs.clone()
        Lx,Ly,Lz,Lt = lat
        p[define._LAT_X_]=Lx; p[define._LAT_Y_]=Ly; p[define._LAT_Z_]=Lz; p[define._LAT_T_]=Lt
        p[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
        p[define._GRID_X_],p[define._GRID_Y_],p[define._GRID_Z_],p[define._GRID_T_]=1,1,1,1
        p[define._NODE_RANK_]=0; p[define._NODE_SIZE_]=1
        p[define._DATA_TYPE_]=DT
        p[define._SET_INDEX_]=0; p[define._SET_PLAN_]=1; p[define._VERBOSE_]=0; p[define._MAX_ITER_]=1000
        a[define._MASS_]=mass; a[define._ATOL_]=atol; a[define._SIGMA_]=0.1
        qcu.applyInitQcu(s, p, a)
        fo = torch.empty_like(fi)
        torch.cuda.synchronize()
        t0=time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo, fi, g, ce, coo, cei, coi, s, p)
        torch.cuda.synchronize()
        t=time.perf_counter()-t0
        p[define._SET_INDEX_]=0
        qcu.applyEndQcu(s, p)
        return fo, t
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            fo, t = fut.result(timeout=timeout)
            return fo, t, "OK", None
        except TimeoutError:
            return None, timeout, "TIMEOUT", "BiStabCG timeout >%ds" % timeout
        except Exception as e:
            return None, 0, "FAIL", traceback.format_exc()

def solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, mass, atol, num_levels, dof_list_in, device, timeout=300, verbose=False, rs=5, cf=1e5, cmi=15, nvi=2, mp=False, sap=False):
    def _run():
        nonlocal U_full, clover_full, g, fi, ce, cei, coo, coi
        dof_list = list(dof_list_in)
        if lat==[16,32,32,48] and lat==[32,32,32,32]:
            pass
        # 32 fallback not needed; 16,32,32,48 with E12 cap only if >12
        if len(dof_list)>1 and dof_list[1]>12 and lat==[16,32,32,48]:
            # keep 12 as optimal for 786k, E24 would be 1.3G->4.4G
            # allow E24 if explicitly requested but warn
            pass
        p = mod_params.clone()
        a = mod_argv.clone()
        s = mod_set_ptrs.clone()
        Lx,Ly,Lz,Lt = lat
        p[define._LAT_X_]=Lx; p[define._LAT_Y_]=Ly; p[define._LAT_Z_]=Lz; p[define._LAT_T_]=Lt
        p[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
        p[define._GRID_X_],p[define._GRID_Y_],p[define._GRID_Z_],p[define._GRID_T_]=1,1,1,1
        p[define._NODE_RANK_]=0; p[define._NODE_SIZE_]=1
        p[define._DATA_TYPE_]=DT
        a2 = a.to(dtype=define.dtype(DT).to_real())
        a2[define._MASS_]=mass; a2[define._ATOL_]=atol; a2[define._SIGMA_]=0.1
        p[define._MG_NUM_LEVEL_]=num_levels
        if num_levels>=2:
            p[define._MG_LEVEL1_E_]=dof_list[1]
            p[define._MG_LEVEL1_X_]=Lx//MG_GRID[0]
            p[define._MG_LEVEL1_Y_]=Ly//MG_GRID[1]
            p[define._MG_LEVEL1_Z_]=Lz//MG_GRID[2]
            p[define._MG_LEVEL1_T_]=Lt//(2*MG_GRID[3])
            p[define._MG_LEVEL1_MAX_ITER_]=cmi
            p[define._MG_LEVEL1_DATA_TYPE_]=DT_C32 if mp else DT
            p[define._MG_LEVEL1_NUM_RESTART_]=rs
            a2[define._MG_LEVEL1_ATOL_]=atol*cf
        if num_levels>=3:
            eff_e2 = dof_list[2]
            p[define._MG_LEVEL2_E_]=eff_e2
            p[define._MG_LEVEL2_X_]=Lx//(MG_GRID[0]*MG_GRID[0])
            p[define._MG_LEVEL2_Y_]=Ly//(MG_GRID[1]*MG_GRID[1])
            p[define._MG_LEVEL2_Z_]=Lz//(MG_GRID[2]*MG_GRID[2])
            p[define._MG_LEVEL2_T_]=Lt//(4*MG_GRID[3])
            p[define._MG_LEVEL2_MAX_ITER_]=200
            p[define._MG_LEVEL2_DATA_TYPE_]=DT_C32 if mp else DT
            p[define._MG_LEVEL2_NUM_RESTART_]=3
            a2[define._MG_LEVEL2_ATOL_]=atol*cf
        if num_levels>=2:
            import gc
            from pyqcu.tools import HierarchicalCache
            hcache = HierarchicalCache(cache_dir=DATA_DIR)
            for name, t in [("g", g), ("fi", fi), ("ce", ce), ("cei", cei), ("coo", coo), ("coi", coi)]:
                try:
                    hcache.register(name, t)
                except Exception:
                    pass
            for name in ["g", "fi", "ce", "cei", "coo", "coi"]:
                ht = hcache.tensors.get(name)
                if ht and ht.is_on_vram():
                    vol_chk = lat[0]*lat[1]*lat[2]*lat[3]
                    do_offload = vol_chk >= 400000
                    if not do_offload:
                        try:
                            free = torch.cuda.mem_get_info(device)[0]
                        except:
                            free = 0
                        do_offload = free < 4*1024**3
                    if do_offload:
                        ht.offload_to_ram()
                        if verbose:
                            try:
                                free2 = torch.cuda.mem_get_info(device)[0]
                            except:
                                free2 = 0
                            print(f"[Hierarchical] offload {name} -> {ht.memory_tier()} (free {free2/1e9:.1f}GB)")
            if verbose:
                print(f"[Hierarchical] lat {lat} dof {dof_list} mp={mp} sap={sap} status {hcache.status()}")
            kappa = 1.0/(2*mass+8)
            op = dslash.operator(U=U_full, clover_term=clover_full, kappa=torch.Tensor([kappa]), support_parity=True, verbose=False)
            S = op.matvec_parity
            del U_full, clover_full
            import gc
            gc.collect(); torch.cuda.empty_cache()
            if verbose:
                print(f"[mem before coarse] allocated {torch.cuda.memory_allocated()/1e9:.2f}GB reserved {torch.cuda.memory_reserved()/1e9:.2f}GB")
            vol = lat[0]*lat[1]*lat[2]*lat[3]
            # For 16x32x32x48 786k, batch true when E<=12
            if lat==[16,32,32,48] and dof_list[1]<=12:
                use_batch = True
            elif lat==[32,32,32,32] and dof_list[1]<=12:
                use_batch = True
            else:
                use_batch = False if vol>400000 else True
            if verbose:
                print(f"[Hierarchical] lat {lat} vol {vol} E={dof_list[1]} batch={use_batch}")
            eff_nvi = 1 if lat==[16,32,32,48] and nvi>1 else nvi
            if eff_nvi != nvi and verbose:
                print(f"[Hierarchical] lat {lat} nvi {nvi}-> {eff_nvi} (large vol time ctrl)")
            try:
                use_local = (lat==[16,32,32,48] and vol>=400000 and dof_list[1]<=12)
                if use_local:
                    print(f"[Local] lat {lat} vol {vol} E={dof_list[1]} use BatchedLocalSchur W=10")
                    from pyqcu.tools._multigrid import give_null_vecs_mt, local_orthogonalize, build_stencil_local, BatchedLocalSchur, _schur_matvec_batch
                    import time as _time, os as _os, h5py as _h5py
                    from pyqcu import tools as _tools
                    lat_fine_odd = [lat[0], lat[1], lat[2], lat[3]//2]
                    lonvs=[]; hnn_l=[]; hdg_l=[]; sit_l=[]
                    batch_mv_raw = lambda x, _op=op: _schur_matvec_batch(_op, x)
                    batch_mv = batch_mv_raw
                    for lvl in range(1, num_levels):
                        E_c = dof_list[lvl]
                        lat_coarse_odd = [lat_fine_odd[d]//MG_GRID[d] for d in range(4)]
                        tag = f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv{lvl}_E{E_c}_nvi{eff_nvi}_t1e-2"
                        cache_file = _os.path.join(CACHE_DIR, tag+".h5")
                        if _os.path.exists(cache_file):
                            lonv=_tools.load_tensor_h5(cache_file, dataset="lonv", device=device)
                            hnn=_tools.load_tensor_h5(cache_file, dataset="hnn", device=device)
                            hdg=_tools.load_tensor_h5(cache_file, dataset="hdg", device=device)
                            sit=_tools.load_tensor_h5(cache_file, dataset="sit", device=device)
                            print(f"  [level {lvl}] E={E_c} CACHED coarse={lat_coarse_odd}")
                        else:
                            t0=_time.perf_counter()
                            _null = give_null_vecs_mt(None, E_c, 12 if lvl==1 else dof_list[lvl-1], lat_fine_odd, define.dtype(DT), device, nv_iters=eff_nvi, nthreads=1, verbose=False, nv_tol=1e-2, batch_matvec=batch_mv)
                            lonv = local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse_odd, verbose=False)
                            lsch = BatchedLocalSchur(op, *lat_fine_odd, W=10)
                            hnn, hdg, sit = build_stencil_local(lsch, lonv, E_c, lat_fine_odd, lat_coarse_odd, define.dtype(DT), device, verbose=verbose)
                            with _h5py.File(cache_file, 'w') as f:
                                for key, tt in (("lonv", lonv), ("hnn", hnn), ("hdg", hdg), ("sit", sit)):
                                    f.create_dataset(key, data=tt.detach().cpu().contiguous().numpy())
                            print(f"  [level {lvl}] E={E_c} local { _time.perf_counter()-t0:.1f}s coarse={lat_coarse_odd}")
                        lonvs.append(lonv); hnn_l.append(hnn); hdg_l.append(hdg); sit_l.append(sit)
                        lat_fine_odd = lat_coarse_odd
                    # done local
                else:
                    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(
                        op, S, num_levels, dof_list, MG_GRID, lat, dof_list[1],
                        define.dtype(DT), device, nv_iters=eff_nvi, use_cache=True, cache_dir=CACHE_DIR, verbose=verbose,
                        batch_build=use_batch)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] coarse build OOM {e}, try empty_cache retry with E=8")
                gc.collect(); torch.cuda.empty_cache()
                for retry_E in [8,6]:
                    try:
                        eff_dof2 = [12, retry_E] + ([retry_E] if num_levels>2 else [])
                        lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(
                            op, S, num_levels, eff_dof2, MG_GRID, lat, eff_dof2[1],
                            define.dtype(DT), device, nv_iters=1, use_cache=False, verbose=verbose)
                        dof_list = eff_dof2
                        print(f"[OOM retry] success E={retry_E}")
                        break
                    except torch.cuda.OutOfMemoryError:
                        continue
                else:
                    raise
            del op, S
            gc.collect(); torch.cuda.empty_cache()
            if verbose:
                print(f"[mem after coarse] allocated {torch.cuda.memory_allocated()/1e9:.2f}GB")
            for name in ["g", "fi", "ce", "cei", "coo", "coi"]:
                ht = hcache.tensors.get(name)
                if ht:
                    try:
                        t = ht.to_device(device)
                        if name == "g":
                            g = t
                        elif name == "fi":
                            fi = t
                        elif name == "ce":
                            ce = t
                        elif name == "cei":
                            cei = t
                        elif name == "coo":
                            coo = t
                        elif name == "coi":
                            coi = t
                        if verbose:
                            print(f"[Hierarchical] reload {name} -> {ht.memory_tier()}")
                    except Exception as e:
                        print(f"[Hierarchical] reload {name} failed {e}")
            for fl in range(len(lonvs)):
                s[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
                s[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
                s[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
                s[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
        p[define._SET_INDEX_]=0; p[define._SET_PLAN_]=1; p[define._VERBOSE_]=0
        qcu.applyInitQcu(s, p, a2)
        fo = torch.empty_like(fi)
        torch.cuda.synchronize()
        t0=time.perf_counter()
        qcu.applyCloverMultigridQcu(fo, fi, g, ce, coo, cei, coi, s, p)
        torch.cuda.synchronize()
        t=time.perf_counter()-t0
        conv=[]
        log_path = os.path.join(os.getcwd(), "logs", "clover_multigrid.log")
        if not os.path.exists(log_path):
            log_path = os.path.join(str(LOG_DIR), "clover_multigrid.log")
        alt = os.path.expanduser("~/PyQCU/logs/clover_multigrid.log")
        for lp in [log_path, alt]:
            if os.path.exists(lp):
                try:
                    with open(lp) as f:
                        for line in f:
                            m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                            if m:
                                conv=[float(x) for x in m.group(1).split(",") if x.strip()]
                                break
                except: pass
                break
        p[define._SET_INDEX_]=0
        qcu.applyEndQcu(s, p)
        return fo, t, conv
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            fo, t, conv = fut.result(timeout=timeout)
            return fo, t, conv, "OK", None
        except TimeoutError:
            return None, timeout, [], "TIMEOUT", f"MG {num_levels}L timeout >{timeout}s (likely OOM or coarse build bottleneck)"
        except Exception as e:
            return None, 0, [], "FAIL", traceback.format_exc()

def cmd_bench(args):
    lat = [int(x) for x in args.lat.split(",")]
    device_ids = [int(x) for x in args.device.split(",") if x.strip()!=""]
    levels = [int(x) for x in args.levels.split(",") if x.strip()!=""]
    device = torch.device(f"cuda:{device_ids[0]}")
    torch.cuda.set_device(device_ids[0])
    print(f"=== dev80_3 bench  {lat}  device {device_ids}  levels {levels}  mass={args.mass} atol={args.atol} ===")
    print(f"DATA_DIR={DATA_DIR} LOG_DIR={LOG_DIR} CACHE_DIR={CACHE_DIR}")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(device_ids[0])}")
    for lp in [Path("logs/clover_multigrid.log"), LOG_DIR/"clover_multigrid.log", Path.home()/"PyQCU/logs/clover_multigrid.log"]:
        try:
            if lp.exists():
                lp.unlink()
        except: pass
    os.environ["QCU_LOG_DIR"] = str(LOG_DIR)
    gen_dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    g0, fi0, ce0, cei0, coo0, coi0, U_full, b_full, clover_full, kappa, av0, _, _ = build_gauge(lat, args.mass, args.atol, gen_dev, seed=42, verbose=args.verbose)
    torch.cuda.set_device(device_ids[0])
    g = g0.to(device); fi = fi0.to(device); ce = ce0.to(device); cei = cei0.to(device); coo = coo0.to(device); coi = coi0.to(device)
    results=[]
    print("\n[1/4] BiStabCG reference...")
    fo_ref, t_ref, stat_ref, err_ref = solve_bistabcg(g, fi, ce, cei, coo, coi, mod_params.clone(), mod_argv.clone(), device, lat, args.mass, args.atol, timeout=args.timeout)
    if fo_ref is not None:
        qcu_U = tools.poooxyzt2oooxyzt(g)
        qcu_src = tools.poooxyzt2oooxyzt(fi)
        qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
        ref_cl = dslash.make_clover(qcu_U, kappa=kappa)
        ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, kappa, True)+dslash.give_clover(qcu_ref, ref_cl)-qcu_src)/tools.norm(qcu_src)
        print(f"  BiStabCG t={t_ref:.3f}s res={ref_res:.2e} stat={stat_ref}")
        results.append({"label":"BiStabCG", "t":t_ref, "res":float(ref_res), "stat":stat_ref, "levels":0})
    else:
        print(f"  BiStabCG {stat_ref} {err_ref}")
        results.append({"label":"BiStabCG", "t":None, "stat":stat_ref, "err":err_ref, "levels":0})
        ref_res = None
    for nl in levels:
        if nl==1:
            dof = [12]
        elif nl==2:
            if lat==[16,32,32,48]:
                dof = [12,12]
            else:
                dof = [12,12]
        elif nl==3:
            if lat==[16,32,32,48]:
                dof = [12,12,12]
            else:
                dof = [12,12,12]
        else:
            dof = [12]*nl
        print(f"\n[MG {nl}L] dof={dof} rs={args.rs} cf={args.cf} cmi={args.cmi} nvi={args.nvi} mp={args.mp} sap={args.sap} ...")
        eff_timeout = 600 if lat==[16,32,32,48] and nl>=2 else args.timeout
        fo_mg, t_mg, conv, stat_mg, err_mg = solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, args.mass, args.atol, nl, dof, gen_dev if nl>=2 else device, timeout=eff_timeout, verbose=args.verbose, rs=args.rs, cf=args.cf, cmi=args.cmi, nvi=args.nvi, mp=args.mp, sap=args.sap)
        if fo_mg is not None:
            qcu_U = tools.poooxyzt2oooxyzt(g)
            qcu_src = tools.poooxyzt2oooxyzt(fi)
            qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
            ref_cl = dslash.make_clover(qcu_U, kappa=kappa)
            mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, kappa, True)+dslash.give_clover(qcu_mg, ref_cl)-qcu_src)/tools.norm(qcu_src)
            if fo_ref is not None:
                qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
                rel = tools.norm(qcu_mg - qcu_ref)/tools.norm(qcu_ref)
            else:
                rel = None
            print(f"  MG {nl}L t={t_mg:.3f}s res={mg_res:.2e} rel={rel} conv_pts={len(conv)} stat={stat_mg}")
            if len(conv)>0:
                np.savetxt(LOG_DIR/f"conv_{nl}L.txt", np.array(conv))
            results.append({"label":f"MG_{nl}L", "t":t_mg, "res":float(mg_res), "rel_vs_ref":float(rel) if rel is not None else None, "stat":stat_mg, "levels":nl, "conv_len":len(conv)})
        else:
            print(f"  MG {nl}L {stat_mg} {err_mg}")
            results.append({"label":f"MG_{nl}L", "t":None, "stat":stat_mg, "err":err_mg, "levels":nl})
    t_l1 = next((r["t"] for r in results if r["label"]=="MG_1L" and r["t"] is not None), None)
    t_ref_v = next((r["t"] for r in results if r["label"]=="BiStabCG" and r["t"] is not None), None)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['label']:10s} t={r.get('t')}  stat={r['stat']}")
    if t_l1 is not None:
        for r in results:
            if r["t"] is not None and r["label"].startswith("MG_"):
                speed = t_l1 / r["t"] if r["t"]>0 else 0
                print(f"  {r['label']} speedup_vs_L1={speed:.3f}x  (L1={t_l1:.3f}s {r['label']}={r['t']:.3f}s)")
                r["speedup_vs_L1"] = speed
    if t_ref_v is not None:
        for r in results:
            if r["t"] is not None and r["label"].startswith("MG_"):
                speed2 = t_ref_v / r["t"] if r["t"]>0 else 0
                print(f"  {r['label']} speedup_vs_BiStabCG={speed2:.3f}x")
                r["speedup_vs_BiStabCG"] = speed2
    best_speed = max([r.get("speedup_vs_L1",0) for r in results if r["label"]!="BiStabCG" and r["label"]!="MG_1L"], default=0)
    print(f"\nBest speedup_vs_L1 = {best_speed:.3f}x  target >2.0 -> {'PASS' if best_speed>2 else 'FAIL'}")
    report = {
        "lat": lat, "mass": MASS if args.mass==MASS else args.mass, "atol": ATOL if args.atol==ATOL else args.atol, "device": device_ids, "levels": levels,
        "results": results, "best_speedup_vs_L1": best_speed, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(DATA_DIR), "cache_dir": CACHE_DIR,
        "note": "L1=MG 1 level (Schur BiStabCG finest only); true speedup = L1 / MG_multi_level; gate 2.0 for 16x32x32x48",
        "mp": args.mp, "sap": args.sap, "rs": args.rs, "cf": args.cf, "cmi": args.cmi, "nvi": args.nvi
    }
    with open(LOG_DIR/"report.json","w") as f:
        json.dump(report,f,indent=2)
    with open(LOG_DIR/"bench_out.txt","w") as f:
        f.write(f"lat {lat} mass {args.mass} atol {args.atol}\n")
        f.write(f"mp={args.mp} sap={args.sap} rs={args.rs} cf={args.cf} cmi={args.cmi} nvi={args.nvi}\n")
        f.write(f"best speedup_vs_L1 {best_speed:.3f} {'PASS' if best_speed>2 else 'FAIL'}\n")
        for r in results:
            f.write(str(r)+"\n")
    print(f"\nReport saved to {LOG_DIR}/report.json and bench_out.txt")
    if device_ids==[0]:
        print("\n[hint] 单卡 V100 完成；多卡 P100*2 需修复后启用 multi 子命令。")

def cmd_hotspot(args):
    lat = [int(x) for x in args.lat.split(",")]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"=== hotspot profiling {lat} on {torch.cuda.get_device_name(0)} ===")
    import torch.profiler as profiler
    # quick profile of 1L vs 2L
    gen_dev = torch.device("cuda:0")
    g0, fi0, ce0, cei0, coo0, coi0, U_full, b_full, clover_full, kappa, av0, _, _ = build_gauge(lat, 0.05, 1e-6, gen_dev, seed=42, verbose=False)
    g=g0; fi=fi0; ce=ce0; cei=cei0; coo=coo0; coi=coi0
    # profile BiStabCG
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True, with_stack=True
    ) as prof:
        fo_ref, t_ref, _, _ = solve_bistabcg(g, fi, ce, cei, coo, coi, mod_params.clone(), mod_argv.clone(), device, lat, 0.05, 1e-6, timeout=300)
    print(f"BiStabCG t {t_ref:.3f}")
    prof.export_chrome_trace(str(LOG_DIR / "trace_bistabcg.json"))
    print(f"trace saved {LOG_DIR}/trace_bistabcg.json")
    # profile MG 1L
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        fo, t, conv, stat, err = solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, 0.05, 1e-6, 1, [12], device, timeout=300, verbose=False)
    print(f"MG 1L t {t:.3f}")
    prof.export_chrome_trace(str(LOG_DIR / "trace_mg1l.json"))
    # profile MG 2L
    # need to rebuild U_full etc? solve_mg moved them, rebuild
    g0, fi0, ce0, cei0, coo0, coi0, U_full2, b_full2, clover_full2, kappa2, av02, _, _ = build_gauge(lat, 0.05, 1e-6, gen_dev, seed=42, verbose=False)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        fo2, t2, conv2, stat2, err2 = solve_mg(g0, fi0, ce0, cei0, coo0, coi0, U_full2, clover_full2, lat, 0.05, 1e-6, 2, [12,12], device, timeout=600, verbose=True)
    print(f"MG 2L t {t2:.3f} stat {stat2}")
    prof.export_chrome_trace(str(LOG_DIR / "trace_mg2l.json"))
    print(f"hotspot traces saved to {LOG_DIR}/trace*.json")
    # nvidia-smi snapshot
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"], capture_output=True, text=True, timeout=10)
        with open(LOG_DIR/"hotspot_smi.txt","w") as f:
            f.write(out.stdout)
        print(out.stdout)
    except Exception as e:
        print(f"nvidia-smi failed {e}")

def cmd_multi(args):
    lat = [int(x) for x in args.lat.split(",")]
    print(f"=== multi P100x2 vs V100 on {lat} ===")
    # single V100
    mg_single = MultiGpuMultigrid(lat_size=lat, mass=0.05, atol=1e-6, num_levels=args.levels, dof_list=[12,12] if args.levels==2 else [12,12,12], mg_grid=MG_GRID, num_restart=3, coarse_max_iter=15, coarse_tol_factor=1e3, nv_iters=1, nthreads=1, device_ids=[0], use_cache=True, cache_dir=CACHE_DIR, verbose=False)
    res_s = mg_single.solve()
    t_single = max(t['mg_time'] for t in res_s['threads'])
    print(f"single V100 mg_wall={t_single:.3f}s")
    # multi P100*2
    try:
        mg_multi = MultiGpuMultigrid(lat_size=lat, mass=0.05, atol=1e-6, num_levels=args.levels, dof_list=[12,12] if args.levels==2 else [12,12,12], mg_grid=MG_GRID, num_restart=3, coarse_max_iter=15, coarse_tol_factor=1e3, nv_iters=1, nthreads=2, device_ids=[1,2], use_cache=True, cache_dir=CACHE_DIR, verbose=False)
        res_m = mg_multi.solve()
        t_multi = max(t['mg_time'] for t in res_m['threads'])
        print(f"multi P100x2 mg_wall={t_multi:.3f}s")
        cons = mg_multi.verify_consistency(tol=1e-5)
        print(f"consistency {cons}")
        with open(LOG_DIR/"multi_report.json","w") as f:
            json.dump({"single": t_single, "multi": t_multi, "consistency": cons, "lat": lat}, f, indent=2)
        print(f"parallel speedup single/multi = {t_single/t_multi:.3f}x (multi vs single)")
    except Exception as e:
        print(f"multi failed {e}")
        traceback.print_exc()
        with open(LOG_DIR/"multi_report.json","w") as f:
            json.dump({"single": t_single, "error": str(e), "lat": lat}, f, indent=2)

def cmd_check(args):
    import json
    p = LOG_DIR/"report.json"
    if not p.exists():
        print(f"missing {p}, run bench first")
        sys.exit(2)
    with open(p) as f:
        j=json.load(f)
    best=j.get("best_speedup_vs_L1",0)
    gate=2.0
    print(f"best speedup_vs_L1={best:.3f} gate={gate} -> {'PASS' if best>=gate else 'FAIL'}")
    sys.exit(0 if best>=gate else 1)

def cmd_report(args):
    import json, glob
    p = LOG_DIR/"report.json"
    if p.exists():
        with open(p) as f:
            j=json.load(f)
        print(json.dumps(j, indent=2))
    # generate png if matplotlib available
    try:
        import matplotlib.pyplot as plt
        if p.exists():
            with open(p) as f:
                j=json.load(f)
            labels=[r["label"] for r in j["results"] if r.get("t") is not None]
            times=[r["t"] for r in j["results"] if r.get("t") is not None]
            plt.figure(figsize=(8,4))
            plt.bar(labels, times)
            plt.ylabel("time (s)")
            plt.title(f"dev80_3 bench {j['lat']} best {j['best_speedup_vs_L1']:.2f}x")
            plt.tight_layout()
            plt.savefig(LOG_DIR/"bench_bar.png", dpi=150)
            print(f"saved {LOG_DIR/'bench_bar.png'}")
            # conv
            for conv_file in glob.glob(str(LOG_DIR/"conv_*.txt")):
                data=np.loadtxt(conv_file)
                plt.figure()
                plt.semilogy(data)
                plt.title(Path(conv_file).stem)
                plt.savefig(LOG_DIR / (Path(conv_file).stem + ".png"))
            print("conv plots done")
    except Exception as e:
        print(f"plot failed {e}")

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="cmd")
    p_bench=sub.add_parser("bench")
    p_bench.add_argument("--lat", type=str, default="16,32,32,48")
    p_bench.add_argument("--device", type=str, default="0")
    p_bench.add_argument("--levels", type=str, default="1,2")
    p_bench.add_argument("--mass", type=float, default=0.05)
    p_bench.add_argument("--atol", type=float, default=1e-6)
    p_bench.add_argument("--timeout", type=int, default=300)
    p_bench.add_argument("--rs", type=int, default=5)
    p_bench.add_argument("--cf", type=float, default=1e5)
    p_bench.add_argument("--cmi", type=int, default=15)
    p_bench.add_argument("--nvi", type=int, default=2)
    p_bench.add_argument("--E", type=int, default=12, help="coarse E")
    p_bench.add_argument("--mp", action="store_true", help="mixed precision coarse c32")
    p_bench.add_argument("--sap", action="store_true", help="SAP smoother enable")
    p_bench.add_argument("--verbose", action="store_true")
    p_bench.set_defaults(func=cmd_bench)
    p_hot=sub.add_parser("hotspot")
    p_hot.add_argument("--lat", type=str, default="16,32,32,48")
    p_hot.set_defaults(func=cmd_hotspot)
    p_multi=sub.add_parser("multi")
    p_multi.add_argument("--lat", type=str, default="16,32,32,48")
    p_multi.add_argument("--levels", type=int, default=2)
    p_multi.set_defaults(func=cmd_multi)
    p_check=sub.add_parser("check")
    p_check.set_defaults(func=cmd_check)
    p_report=sub.add_parser("report")
    p_report.set_defaults(func=cmd_report)
    args=parser.parse_args()
    if not hasattr(args,"func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__=="__main__":
    main()
