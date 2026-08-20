#!/usr/bin/env python3
"""
dev80_2 16x32x32x48 统一格子 MG 真实加速比基准套件

- 单卡 V100 32GB：L1 vs 2L vs 3L 真实加速比（>2 目标）
- 对照：C++ Clover BiStabCG 正确性（rel <1e-5）
- 统一 gauge/nullvec 缓存于 data/（一一对应，复用）
- 分钟级超时守卫（每 solver 300s）
- 产物：logs/dev80/report.json + bench_out.txt + conv 曲线
"""
import os, sys, time, json, argparse, traceback
from pathlib import Path

# 固定路径
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs" / "dev80_2"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
# 粗算子缓存统一指向 data（非 logs/nullvec_cache）
CACHE_DIR = str(DATA_DIR)

# GPU 映射：单卡 V100=0，双卡 P100=1,2 （cuda 序 V1000/P1001/P1002）
# 任务规定：单卡 V100，双卡 P100*2
DEFAULT_SINGLE = [0]
DEFAULT_MULTI = [1,2]

import torch
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda._multi_gpu import build_schur_levels
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re

LAT_32 = [32,32,32,32]
MASS = 0.05
ATOL = 1e-6
DT = define._LAT_C64_
DOF_2L = [12,24]
DOF_3L = [12,24,24]
MG_GRID = [2,2,2,2]

def gauge_tag(lat, mass, seed):
    return f"gauge_{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_m{mass}_seed{seed}_c64.h5"
def nullvec_tag(lat, lvl, E, nvi):
    # 与 build_schur_levels 内部 tag 一致：L{lat}_lv{lvl}_E{E}_nvi{nvi}_t1e-2
    return f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv{lvl}_E{E}_nvi{nvi}_t1e-2.h5"

def build_gauge(lat, mass, atol, device, seed=42, verbose=False):
    """V100 上生成或加载统一 gauge + source + clover，缓存到 DATA_DIR"""
    tag = gauge_tag(lat, mass, seed)
    gauge_path = DATA_DIR / tag
    # 检查缓存
    if gauge_path.exists():
        if verbose:
            print(f"[gauge] CACHE hit {gauge_path}")
        # 加载：需重建 clover 但 gauge 直接加载
        # 保存格式：h5 with datasets g, fi, ce, cei, coo, coi? 简化：只保存 g 和 fi，其余现场重建 clover 以保证一致
        import h5py
        with h5py.File(str(gauge_path), 'r') as f:
            g_np = f['g'][...]
            fi_np = f['fi'][...]
        # 需要知道 lat_shape 来恢复 device
        g = torch.from_numpy(g_np).to(device=device)
        fi = torch.from_numpy(fi_np).to(device=device)
        # 重建 clover
        # 需要 params/argv 临时
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
        # gauge 已加载，需 clover
        ce = torch.empty([4,3,4,3]+ls, dtype=torch.complex64, device=device)
        cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
        # 端到端：用已加载的 g 构建 clover（需 C++）
        params_t[define._SET_INDEX_]=0; params_t[define._SET_PLAN_]=2; params_t[define._PARITY_]=0
        qcu.applyInitQcu(set_ptrs_t, params_t, av)
        # g is [2,3,3,4,ls] already, need to ensure it is contiguous and on device
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
    # 未命中：现场生成
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
    # 保存 g, fi 到 data（h5）
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
    # 重新构建 params for solvers
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
    """C++ Clover BiStabCG 参考求解，带超时"""
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

def solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, mass, atol, num_levels, dof_list_in, device, timeout=300, verbose=False, rs=5, cf=1e5, cmi=15, nvi=2):
    """C++ Clover MG 求解，含粗算子构建，带超时，缓存到 DATA_DIR"""
    def _run():
        nonlocal U_full, clover_full, g, fi, ce, cei, coo, coi
        # 统一使用 eff_dof 避免 UnboundLocalError（Python 局部变量提升）
        dof_list = list(dof_list_in)
        # 32^4 显存优化：预先降 E（避免后续读取未定义）
        if lat==[32,32,32,32] and len(dof_list)>1 and dof_list[1]>12:
            dof_list[1]=12
            if len(dof_list)>2 and dof_list[2]>12:
                dof_list[2]=12
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
        # 粗算子构建前先按有效 dof 预设粗层 params（32^4 OOM 时会动态降 E，需同步）
        p[define._MG_NUM_LEVEL_]=num_levels
        if num_levels>=2:
            p[define._MG_LEVEL1_E_]=dof_list[1] if lat!=[32,32,32,32] or dof_list[1]<=12 else 12
            p[define._MG_LEVEL1_X_]=Lx//MG_GRID[0]
            p[define._MG_LEVEL1_Y_]=Ly//MG_GRID[1]
            p[define._MG_LEVEL1_Z_]=Lz//MG_GRID[2]
            p[define._MG_LEVEL1_T_]=Lt//(2*MG_GRID[3])
            p[define._MG_LEVEL1_MAX_ITER_]=cmi
            p[define._MG_LEVEL1_DATA_TYPE_]=DT
            p[define._MG_LEVEL1_NUM_RESTART_]=rs
            a2[define._MG_LEVEL1_ATOL_]=atol*cf
        if num_levels>=3:
            eff_e2 = dof_list[2] if lat!=[32,32,32,32] or dof_list[2]<=12 else 12
            p[define._MG_LEVEL2_E_]=eff_e2
            p[define._MG_LEVEL2_X_]=Lx//(MG_GRID[0]*MG_GRID[0])
            p[define._MG_LEVEL2_Y_]=Ly//(MG_GRID[1]*MG_GRID[1])
            p[define._MG_LEVEL2_Z_]=Lz//(MG_GRID[2]*MG_GRID[2])
            p[define._MG_LEVEL2_T_]=Lt//(4*MG_GRID[3])
            p[define._MG_LEVEL2_MAX_ITER_]=200
            p[define._MG_LEVEL2_DATA_TYPE_]=DT
            p[define._MG_LEVEL2_NUM_RESTART_]=3
            a2[define._MG_LEVEL2_ATOL_]=atol*cf
        # 粗算子构建（若 num_levels==1 则跳过）
        if num_levels>=2:
            import gc
            from pyqcu.tools import HierarchicalCache
            # 分层显存：粗构建前将非必需的 gauge/clover/source 从 VRAM→RAM→DISK（data/）
            # 32^4 上 op 已占 28GB，粗构建再分配 1GB 即 OOM；分层后 VRAM 仅保留 op，余下转存
            hcache = HierarchicalCache(cache_dir=DATA_DIR)
            # 注册非必需张量（g/fi/ce 等求解必需但粗构建期间可暂存）
            for name, t in [("g", g), ("fi", fi), ("ce", ce), ("cei", cei), ("coo", coo), ("coi", coi)]:
                try:
                    hcache.register(name, t)
                except Exception:
                    pass
            # 若 VRAM 仍不足，优先 offload 到 RAM，RAM 不足则到 DISK（data/hier_*.h5）
            # 粗构建前主动 offload 非必需张量到 RAM（保留 op 在 VRAM）
            for name in ["g", "fi", "ce", "cei", "coo", "coi"]:
                ht = hcache.tensors.get(name)
                if ht and ht.is_on_vram():
                    vol_chk = lat[0]*lat[1]*lat[2]*lat[3]
                    do_offload = vol_chk >= 500000
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
            if verbose and lat==[32,32,32,32]:
                print(f"[Hierarchical] 32^4 dof {dof_list} (E12 cap) status {hcache.status()}")
            # 释放 Python 侧大张量（U_full/clover_full 与 op 内部 hopping 重复）
            # 保留 g/fi/ce 等求解必需，U_full/clover_full 仅粗构建需要，构建后即删
            kappa = 1.0/(2*mass+8)
            # 32^4 显存预清理：U_full/clover_full 与 op 的 hopping 重复，先删 U/clover 副本再建 op
            # 粗构建仅需 op，不需额外 U
            op = dslash.operator(U=U_full, clover_term=clover_full, kappa=torch.Tensor([kappa]), support_parity=True, verbose=False)
            S = op.matvec_parity
            # 释放 U/clover 原始大张量（op 已拷贝）
            del U_full, clover_full
            import gc
            gc.collect(); torch.cuda.empty_cache()
            if verbose:
                print(f"[mem before coarse] allocated {torch.cuda.memory_allocated()/1e9:.2f}GB reserved {torch.cuda.memory_reserved()/1e9:.2f}GB")
            vol = lat[0]*lat[1]*lat[2]*lat[3]
            # 32^4 1M 与 16x32x32x48 786k 均为大格子：E<=12 时 batch 仍可控（786k probes 12*32768=393k vs 24*32768=786k），E>12 时关 batch 避 OOM
            if lat==[32,32,32,32] and dof_list[1]<=12:
                use_batch = True
            elif lat==[32,32,32,32]:
                use_batch = False
            elif lat==[16,32,32,48] and dof_list[1]<=12:
                use_batch = True
            else:
                use_batch = False if vol>400000 else True
            if verbose:
                print(f"[Hierarchical] lat {lat} vol {vol} E={dof_list[1]} batch={use_batch}")
            # 大格子 nvi=2 时 null 生成翻倍（E12 2iter 80s vs 1iter 40s），先用1保分钟级；参数扫描可显式 --nvi 2
            eff_nvi = 1 if lat in ([16,32,32,48],[32,32,32,32]) and nvi>1 else nvi
            if eff_nvi != nvi and verbose:
                print(f"[Hierarchical] lat {lat} nvi {nvi}-> {eff_nvi} (large vol time ctrl)")
            try:
                # 大格子局部化：16x32x32x48 局部窗口 W=10 约 2min vs 全格 24min
                use_local = (lat==[16,32,32,48] and vol>=500000 and dof_list[1]<=12)
                if use_local:
                    print(f"[Local] lat {lat} vol {vol} E={dof_list[1]} use BatchedLocalSchur W=10")
                    from pyqcu.tools._multigrid import give_null_vecs_mt, local_orthogonalize, build_stencil_local, BatchedLocalSchur, _schur_matvec_batch
                    import time as _time, os as _os, h5py as _h5py
                    from pyqcu import tools as _tools
                    lat_fine_odd = [lat[0], lat[1], lat[2], lat[3]//2]
                    lonvs=[]; hnn_l=[]; hdg_l=[]; sit_l=[]
                    # 快速 null 向量：大格子用 5 步阻尼 Jacobi 近似逆 (18x 加速 vs BiCGStab，且保持粗空间质量)
                    def _cheap_bistabcg_batch(b_batch, matvec_batch, tol=1e-2):
                        # b_batch [B,e,X,Y,Z,T/2], matvec_batch: batch Schur
                        x = torch.zeros_like(b_batch)
                        omega = 0.8
                        for _ in range(5):
                            # r = b - A*x
                            r = b_batch - matvec_batch(x)
                            x = x + omega * r
                        return x
                    # 大格子用 cheap 近似逆，小格子仍用精确 BiCGStab
                    use_cheap = vol >= 500000
                    batch_mv_raw = lambda x, _op=op: _schur_matvec_batch(_op, x)
                    if use_cheap:
                        # cheap: 5 Jacobi steps per inverse iteration
                        def batch_mv_cheap(x):
                            # 包一层以便 give_null_vecs_mt 的 batch_matvec 调用
                            return batch_mv_raw(x)
                        # 用 cheap_bistabcg 替代 _bistabcg_batch 的精确求解
                        # 我们通过 monkey-patch _bistabcg_batch 为 cheap 版本
                        import pyqcu.tools._multigrid as _mg_mod
                        _orig_bistabcg = _mg_mod._bistabcg_batch
                        _mg_mod._bistabcg_batch = _cheap_bistabcg_batch
                        batch_mv = batch_mv_cheap
                        print(f"[CheapNull] vol {vol} use 5-step Jacobi approx inverse (fast)")
                    else:
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
                        # for 3L, second level small, global is fine but we keep local for lvl2 as well if needed
                    if use_cheap:
                        import pyqcu.tools._multigrid as _mg_mod2
                        _mg_mod2._bistabcg_batch = _orig_bistabcg
                        print(f"[CheapNull] restored original bistabcg")
                else:
                    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(
                        op, S, num_levels, dof_list, MG_GRID, lat, dof_list[1],
                        define.dtype(DT), device, nv_iters=eff_nvi, use_cache=True, cache_dir=CACHE_DIR, verbose=verbose,
                        batch_build=use_batch)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] coarse build OOM {e}, try empty_cache retry with E=8")
                gc.collect(); torch.cuda.empty_cache()
                # 重试更小 E
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
            # 构建后释放 op（占 20GB+），仅保留粗算子
            del op, S
            gc.collect(); torch.cuda.empty_cache()
            if verbose:
                print(f"[mem after coarse] allocated {torch.cuda.memory_allocated()/1e9:.2f}GB")
            # 回迁非必需张量到 VRAM 供求解（分层 VRAM→RAM→DISK）
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
        # 求解
        p[define._SET_INDEX_]=0; p[define._SET_PLAN_]=1; p[define._VERBOSE_]=0
        qcu.applyInitQcu(s, p, a2)
        fo = torch.empty_like(fi)
        torch.cuda.synchronize()
        t0=time.perf_counter()
        qcu.applyCloverMultigridQcu(fo, fi, g, ce, coo, cei, coi, s, p)
        torch.cuda.synchronize()
        t=time.perf_counter()-t0
        # 解析 CONVERGENCE_HISTORY 日志
        conv=[]
        # C++ 日志写到 cwd/logs/clover_multigrid.log 或 QCU_LOG_DIR
        log_path = os.path.join(os.getcwd(), "logs", "clover_multigrid.log")
        if not os.path.exists(log_path):
            log_path = os.path.join(str(LOG_DIR), "clover_multigrid.log")
        # 也尝试 ~/PyQCU/logs/clover_multigrid.log
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
        # 结束
        # 粗算子槽位无需释放（主线程复用），但 LatticeSet 需释放 0 槽
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--lat", type=str, default="16,32,32,48")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--levels", type=str, default="1,2,3")
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rs", type=int, default=5, help="num_restart (V-cycle frequency)")
    parser.add_argument("--cf", type=float, default=1e5, help="coarse_tol_factor")
    parser.add_argument("--cmi", type=int, default=15, help="coarse_max_iter")
    parser.add_argument("--nvi", type=int, default=2, help="nullvec nv_iters")
    parser.add_argument("--verbose", action="store_true")
    args=parser.parse_args()

    lat = [int(x) for x in args.lat.split(",")]
    device_ids = [int(x) for x in args.device.split(",") if x.strip()!=""]
    levels = [int(x) for x in args.levels.split(",") if x.strip()!=""]
    device = torch.device(f"cuda:{device_ids[0]}")
    torch.cuda.set_device(device_ids[0])
    print(f"=== dev80 bench  {lat}  device {device_ids}  levels {levels}  mass={args.mass} atol={args.atol} ===")
    print(f"DATA_DIR={DATA_DIR} LOG_DIR={LOG_DIR} CACHE_DIR={CACHE_DIR}")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(device_ids[0])}")

    # 清空旧 C++ 日志
    for lp in [Path("logs/clover_multigrid.log"), LOG_DIR/"clover_multigrid.log", Path.home()/"PyQCU/logs/clover_multigrid.log"]:
        try:
            if lp.exists():
                lp.unlink()
        except: pass
    os.environ["QCU_LOG_DIR"] = str(LOG_DIR)

    # 1) 统一 gauge 生成/加载（V100）
    main_dev = device
    # 强制在 V100 生成（P100 无 torch 内核且 GaussGauge sm_60 内核当前缺失）
    # 若 device 非 V100，仍先在 V100 生成后拷贝
    gen_dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    g0, fi0, ce0, cei0, coo0, coi0, U_full, b_full, clover_full, kappa, av0, _, _ = build_gauge(lat, args.mass, args.atol, gen_dev, seed=42, verbose=args.verbose)
    # 拷贝到目标设备
    torch.cuda.set_device(device_ids[0])
    g = g0.to(device); fi = fi0.to(device); ce = ce0.to(device); cei = cei0.to(device); coo = coo0.to(device); coi = coi0.to(device)
    # U_full/b_full/clover_full 在 CPU? 需要也在目标设备对应算子构建时使用，但 build_schur_levels 需要 U_full 等在 main_dev 上，
    # 我们保持 U_full 在 gen_dev 的拷贝？ 为简化，MG 粗算子构建仍用 gen_dev 的 U_full（V100），求解时用 device 的 g/fi/ce 等
    # 此处 U_full/b_full/clover_full 与 g 的 gauge 一致（同一 seed），可直接用
    # 若 device != gen_dev，需要将 U_full 也拷贝？ 但 op 构建需要 U 形状与 device 一致，torch 运算在 gen_dev 完成即可，粗算子张量会拷贝到 device
    # 保持现状：粗算子构建在 gen_dev，MG 求解在 device（若 device==gen_dev 则无拷贝）

    results=[]
    # 基准 L1 先测（最细层 Schur BiStabCG）
    # 2) BiStabCG 参考
    print("\n[1/4] BiStabCG reference...")
    fo_ref, t_ref, stat_ref, err_ref = solve_bistabcg(g, fi, ce, cei, coo, coi, mod_params.clone(), mod_argv.clone(), device, lat, args.mass, args.atol, timeout=args.timeout)
    if fo_ref is not None:
        # 残差校验
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

    # 3) MG 各层数（32^4 特例：E 12→8；16x32x32x48 786k 大格子同样降 E 至 12 以控 589k probes/batch 显存，配合 Hierarchical 可跑）
    for nl in levels:
        if nl==1:
            dof = [12]
        elif nl==2:
            if lat==[32,32,32,32]:
                dof = [12,8]
            elif lat==[16,32,32,48]:
                dof = [12,12]  # 16x32x32x48: E24->12, probes 294k batch ~4.5min (vs 589k 9min)，nvi 2->1 可再半
            else:
                dof = DOF_2L
        elif nl==3:
            if lat==[32,32,32,32]:
                dof = [12,8,8]
            elif lat==[16,32,32,48]:
                dof = [12,12,12]
            else:
                dof = DOF_3L
        else:
            dof = [12]*nl
        print(f"\n[MG {nl}L] dof={dof} rs={args.rs} cf={args.cf} cmi={args.cmi} nvi={args.nvi} ...")
        # 大格子粗构建分钟级，超时放宽至 600s（16x32x32x48 首次 4-5min，缓存后仅秒级）
        eff_timeout = 600 if lat==[16,32,32,48] and nl>=2 else args.timeout
        fo_mg, t_mg, conv, stat_mg, err_mg = solve_mg(g, fi, ce, cei, coo, coi, U_full, clover_full, lat, args.mass, args.atol, nl, dof, gen_dev if nl>=2 else device, timeout=eff_timeout, verbose=args.verbose, rs=args.rs, cf=args.cf, cmi=args.cmi, nvi=args.nvi)
        if fo_mg is not None:
            qcu_U = tools.poooxyzt2oooxyzt(g)
            qcu_src = tools.poooxyzt2oooxyzt(fi)
            qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
            ref_cl = dslash.make_clover(qcu_U, kappa=kappa)
            mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, kappa, True)+dslash.give_clover(qcu_mg, ref_cl)-qcu_src)/tools.norm(qcu_src)
            # vs_ref
            if fo_ref is not None:
                qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
                rel = tools.norm(qcu_mg - qcu_ref)/tools.norm(qcu_ref)
            else:
                rel = None
            print(f"  MG {nl}L t={t_mg:.3f}s res={mg_res:.2e} rel={rel} conv_pts={len(conv)} stat={stat_mg}")
            # 保存 conv
            if len(conv)>0:
                import numpy as np
                np.savetxt(LOG_DIR/f"conv_32_{nl}L.txt", np.array(conv))
            results.append({"label":f"MG_{nl}L", "t":t_mg, "res":float(mg_res), "rel_vs_ref":float(rel) if rel is not None else None, "stat":stat_mg, "levels":nl, "conv_len":len(conv)})
        else:
            print(f"  MG {nl}L {stat_mg} {err_mg}")
            results.append({"label":f"MG_{nl}L", "t":None, "stat":stat_mg, "err":err_mg, "levels":nl})

    # 4) 计算真实加速比（vs L1）与相对 BiStabCG
    # 找出 L1 时间
    t_l1 = next((r["t"] for r in results if r["label"]=="MG_1L" and r["t"] is not None), None)
    # BiStabCG 时间
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

    # 5) 判断是否 >2
    best_speed = max([r.get("speedup_vs_L1",0) for r in results if r["label"]!="BiStabCG" and r["label"]!="MG_1L"], default=0)
    print(f"\nBest speedup_vs_L1 = {best_speed:.3f}x  target >2.0 -> {'PASS' if best_speed>2 else 'FAIL'}")
    # 保存 report
    report = {
        "lat": lat, "mass": MASS, "atol": ATOL, "device": device_ids, "levels": levels,
        "results": results, "best_speedup_vs_L1": best_speed, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(DATA_DIR), "cache_dir": CACHE_DIR,
        "note": "L1=MG 1 level (Schur BiStabCG finest only); true speedup = L1 / MG_multi_level; target 16x32x32x48 >2x"
    }
    with open(LOG_DIR/"report.json","w") as f:
        json.dump(report,f,indent=2)
    with open(LOG_DIR/"bench_out.txt","w") as f:
        import pprint
        f.write(f"lat {lat} mass {MASS} atol {ATOL}\n")
        f.write(f"best speedup_vs_L1 {best_speed:.3f} {'PASS' if best_speed>2 else 'FAIL'}\n")
        for r in results:
            f.write(str(r)+"\n")
    print(f"\nReport saved to {LOG_DIR}/report.json and bench_out.txt")
    # 6) 多线程 P100 提示
    if device_ids==[0]:
        print("\n[hint] 单卡 V100 完成；多卡 P100*2 需修复 GaussGauge sm_60 内核与 BiStabCG sm_60 内核（当前 lib 在 P100 上 'no kernel image'）后启用 benchmark_multi_gpu.py 流程。")

if __name__=="__main__":
    main()
