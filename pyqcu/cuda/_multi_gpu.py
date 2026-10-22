"""测试专用：单进程多线程多 GPU 的 Legacy Clover-MG 驱动。

生产 CUDA 拓扑固定为一个 Python 进程对应一个 MPI rank、对应一张
GPU/DCU/GPGPU。本模块中的 ``MultiGpuMultigrid`` 是该规则的显式测试例外：
``pyqcu`` 导入时不会创建实例、启动线程或选择多张设备；只有调用方显式导入
并构造该类时才启用。``build_schur_levels`` 与 cache helper 不启动多设备
worker，仍可供单进程、单 rank、单设备的 Legacy MG 程序使用。

模式：N 个线程并行，每个线程绑定一张 GPU（device_id = tid % num_gpus），
各自在绑定的设备上运行完整的 C++ 后端流程：
  生成/拷贝规范场与 Clover 项 → 参考 BiStabCG（applyCloverBistabCgQcu）
  → 完整 Clover Multigrid V-cycle（applyCloverMultigridQcu）。

线程隔离约定：
  * 每线程持独立 params/argv/set_ptrs 副本（int32[58]/float[8]/int64[100]），
    _SET_INDEX_ 在各自副本中从 0 起独立计数 —— 无共享写竞争。
  * 每线程独立 CUDA 上下文：worker 内先 torch.cuda.set_device(dev_id)
    （CUDA 运行时 API 的 current device 是线程局部的）。
  * Cython 桥（pyqcu.cuda.qcu）所有函数均以 with nogil 释放 GIL 后调用
    C++ 后端 —— 多线程可真正并行进入 libqcu.so，一线程一卡才有效。
  * 粗网格 Schur 算子（33-tensor）在主线程构建一次（确定性探测，写集
    不相交），各线程拷贝到本卡后填入自己的 set_ptrs 槽位（只读共享）。

该驱动不是单次 solve 的跨 GPU 格点分域实现，不用于生产 ``solve_case``。
需要多个 GPGPU 协同求解同一格点场时，必须使用 MPI 多进程、一 rank 一设备。

验证语义：默认所有线程求解同一问题（同 seed 生成的规范场/源），
solve() 返回各线程解，verify_consistency() 校验线程间一致性
（与主线程单线程参考解相对误差 < tol）—— 多卡正确性的等价判据。
"""

import hashlib
import math
import os
import time
from typing import List, Optional

import torch

from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as _module_params
from pyqcu.cuda.define import argv as _module_argv
from pyqcu.cuda.define import set_ptrs as _module_set_ptrs
from pyqcu.cuda._schur_op import CudaSchurOp

_SET_PTRS_COARSE_BASE_ = define._SET_PTRS_COARSE_BASE_


def _coarse_cache_paths(cache_dir, lat_full, level, E, nv_iters, nv_tol,
                        mass=None, csw=None, seed=None, cache_tag=None):
    """Return equivalent cache names, preferring the repository spelling.

    Older runs formatted ``1e-2`` as ``t0.01`` while the shared ``data/``
    assets use ``t1e-2``.  Treat the names as aliases for reads so a cache hit
    never silently turns into a multi-minute stencil rebuild.  New writes use
    the first (canonical) name.
    """
    mass_label = "unknown" if mass is None else f"{float(mass):.12g}"
    csw_label = "unknown" if csw is None else f"{float(csw):.12g}"
    seed_label = "unknown" if seed is None else str(int(seed))
    tag_label = ""
    if cache_tag:
        digest = hashlib.sha256(str(cache_tag).encode("utf-8")).hexdigest()[:12]
        tag_label = f"_k{digest}"
    base = (f"L{lat_full[0]}x{lat_full[1]}x{lat_full[2]}x{lat_full[3]}"
            f"_m{mass_label}_c{csw_label}_s{seed_label}{tag_label}"
            f"_lv{level}_E{E}_nvi{nv_iters}_t")
    value = float(nv_tol)
    if abs(value - 1e-2) <= 1e-12:
        labels = ("1e-2", "0.01")
    else:
        labels = (f"{nv_tol}", f"{value:g}")
    return [os.path.join(cache_dir, base + label + ".h5") for label in labels]


def _clone_state():
    """每线程独立 params/argv/set_ptrs 副本。"""
    return (_module_params.clone(), _module_argv.clone(), _module_set_ptrs.clone())


def build_schur_levels(op, S, num_levels, dof_list, mg_grid, lat_full, E, dt, device,
                       nv_iters=2, use_cache=True, cache_dir=None, verbose=True,
                       matvec_ops=None, nthreads=None, av=None, params_template=None,
                       nv_tol=1e-2, batch_build=True, mass=None, csw=None,
                       seed=None, cache_tag=None, local_stencil=None,
                       local_stencil_batch_size=4):
    """构建 S 的 null 向量 + 33-tensor Galerkin 粗算子 A_c = P^T S P。

    返回 (lonvs, hnn_l, hdg_l, sit_l) 列表（每粗层一项）。
    结果以 h5py 缓存到磁盘（keyed by lattice/dof/nv_iters/nv_tol），
    重复运行跳过昂贵的 setup（与 h5py 多线程读写约定一致）。

    matvec_ops: 可选，每线程一个 matvec 算子（如 CudaSchurOp 实例列表）——
                给定时 null 向量生成（give_null_vecs_mt）与 stencil 构建
                （build_stencil_mt）均多线程并行，适合多卡/多线程加速；
                否则用单线程 Python matvec S。
    2026-08-15 扩展：lvl>=2 同样走多线程 C++ matvec。每层构建完成后用
                本层 33-tensor stencil 创建 CudaCoarseSchurOp 列表（宽版
                粗层 Schur 算子，任意 DOF E），替换细层 CudaSchurOp 继续
                下一层 —— 消除 lvl>=2 的单线程 Python stencil 瓶颈。
                av/params_template 给定时才启用；否则 lvl>=2 回退单线程。

    nv_tol: null 向量 BiCGStab 解容差（1e-2 默认，5e-5 在粗层大系统迭代爆炸）。

    batch_build=True（默认）：stencil 探测批量化 —— 固定 c_idx 一次批量全部
                E 探针（_probe_point_batch，torch 批量 matvec）。细层用
                _schur_matvec_batch(op)（dslash.operator 组件 einsum），
                lvl>=2 用 _stencil_matvec_batch（本层 stencil 批量版），
                消除逐探针 C++ 调用+同步开销。要求 op 为 dslash.operator
                （V100 主线程 torch 可用；P100 sm_60 无 torch kernel 时
                构建本就在主线程 V100 完成）。实测 8x8x8x16 lv1：
                135s → ~15s（10 倍），16x16x16x32 lv1（196608 probes）~36min → ~3min。
    """
    from pyqcu.tools import build_stencil, apply_stencil, build_stencil_mt, give_null_vecs_mt
    from pyqcu.tools._multigrid import (
        BatchedLocalSchur, _schur_matvec_batch, _stencil_matvec_batch,
        build_stencil_local,
    )
    from pyqcu.cuda._schur_op import CudaCoarseSchurOp
    progress = os.environ.get("PYQCU_MG_BUILD_PROGRESS") == "1"

    def report(message):
        if progress:
            print(f"PYQCU::CUDA::MG_BUILD:\n {message}", flush=True)

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~/PyQCU/logs"), "nullvec_cache")
    os.makedirs(cache_dir, exist_ok=True)
    lonvs, hnn_l, hdg_l, sit_l = [], [], [], []
    lat_fine_odd = [lat_full[0], lat_full[1], lat_full[2], lat_full[3] // 2]
    E_prev = 12
    coarse_ops = matvec_ops  # 当前层 matvec 算子（细层 CudaSchurOp / 粗层 CudaCoarseSchurOp）
    created_ops = []         # 本函数创建的粗层 ops（结束时统一 release）
    # 批量 matvec（batch_build 模式）：细层用 dslash.operator 的批量 Schur，
    # lvl>=2 用本层 stencil 的批量版（_stencil_matvec_batch）
    if batch_build and op is not None:
        batch_mv = lambda x, _op=op: _schur_matvec_batch(_op, x)
    else:
        batch_mv = None
    for lvl in range(1, num_levels):
        E_c = dof_list[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // mg_grid[d] for d in range(4)]
        cache_paths = _coarse_cache_paths(
            cache_dir, lat_full, lvl, E_c, nv_iters, nv_tol,
            mass=mass, csw=csw, seed=seed, cache_tag=cache_tag)
        cache_file = next((p for p in cache_paths if use_cache and os.path.exists(p)),
                          cache_paths[0])
        if use_cache and os.path.exists(cache_file):
            report(f"level {lvl} cache hit {cache_file}")
            lonv = tools.load_tensor_h5(cache_file, dataset="lonv", device=device)
            hnn = tools.load_tensor_h5(cache_file, dataset="hnn", device=device)
            hdg = tools.load_tensor_h5(cache_file, dataset="hdg", device=device)
            sit = tools.load_tensor_h5(cache_file, dataset="sit", device=device)
            if verbose:
                print(f"  [level {lvl}] E={E_c} CACHED coarse={lat_coarse_odd}")
        else:
            t0 = time.perf_counter()
            if local_stencil is None:
                requested_local = os.environ.get("PYQCU_MG_LOCAL_STENCIL")
                if requested_local is None:
                    use_local_stencil = (
                        lvl == 1 and op is not None and
                        E_c * math.prod(lat_coarse_odd) >= 200000)
                else:
                    use_local_stencil = (
                        requested_local == "1" and lvl == 1 and op is not None)
            else:
                use_local_stencil = bool(local_stencil) and lvl == 1 and op is not None
            report(f"level {lvl} null-space start E={E_c} shape={lat_fine_odd}")
            if batch_build and batch_mv is not None:
                # 批量 null 向量生成（torch 批量 BiCGStab）+ 批量探测
                _null = give_null_vecs_mt(
                    coarse_ops, E_c, E_prev, lat_fine_odd, dt, device,
                    nv_iters=nv_iters, nthreads=nthreads, verbose=False,
                    nv_tol=nv_tol, batch_matvec=batch_mv)
            elif coarse_ops is not None:
                # 多线程 C++ matvec（细层 CudaSchurOp；粗层 CudaCoarseSchurOp）
                _null = give_null_vecs_mt(
                    coarse_ops, E_c, E_prev, lat_fine_odd, dt, device,
                    nv_iters=nv_iters, nthreads=nthreads, verbose=False,
                    nv_tol=nv_tol)
            else:
                _null = torch.randn([E_c, E_prev] + lat_fine_odd, dtype=dt, device=device)
                for _ in range(nv_iters):
                    _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None,
                                                 verbose=False)
            lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse_odd,
                                             verbose=False)
            report(
                f"level {lvl} null-space done {time.perf_counter() - t0:.3f}s; "
                "stencil start")
            if use_local_stencil:
                # W=8 is the validated production window: it covers the
                # center block plus both coarse neighbours, while W=10 adds
                # 2.44x more W^4 temporary volume without changing any of the
                # 33 stencil tensors.
                lsch = BatchedLocalSchur(op, *lat_fine_odd, W=8)
                hnn, hdg, sit = build_stencil_local(
                    lsch, lonv, E_c, lat_fine_odd, lat_coarse_odd, dt,
                    device, verbose=verbose,
                    batch_size=local_stencil_batch_size)
            elif batch_build and batch_mv is not None:
                # 批量探测（torch einsum，主线程 V100）
                hnn, hdg, sit = build_stencil_mt(
                    [batch_mv], lonv, E_c, E_prev, lat_fine_odd, lat_coarse_odd,
                    dt, device, nthreads=1, verbose=verbose, batch=True)
            elif coarse_ops is not None:
                hnn, hdg, sit = build_stencil_mt(
                    coarse_ops, lonv, E_c, E_prev, lat_fine_odd, lat_coarse_odd,
                    dt, device, nthreads=nthreads or len(coarse_ops), verbose=verbose)
            else:
                hnn, hdg, sit = build_stencil(S, lonv, E_c, E_prev,
                                              lat_fine_odd, lat_coarse_odd,
                                              dt=dt, device=device, verbose=verbose)
            # 单句柄一次写入全部 dataset（避免逐 dataset 覆盖重建文件）
            import h5py
            with h5py.File(cache_file, 'w') as f:
                for key, t in (("lonv", lonv), ("hnn", hnn), ("hdg", hdg), ("sit", sit)):
                    f.create_dataset(key, data=t.detach().cpu().contiguous().numpy())
            if verbose:
                print(f"  [level {lvl}] E={E_c} nv_time={time.perf_counter()-t0:.1f}s "
                      f"coarse={lat_coarse_odd}")
            report(f"level {lvl} stencil done {time.perf_counter() - t0:.3f}s")
        lonvs.append(lonv); hnn_l.append(hnn); hdg_l.append(hdg); sit_l.append(sit)
        S = lambda v, _hnn=hnn, _hdg=hdg, _sit=sit: apply_stencil(_hnn, _hdg, _sit, v)
        # 下一层批量 matvec：本层 stencil 的批量版（任意 E）
        if batch_build and op is not None:
            st = (sit, hnn, hdg)  # _stencil_matvec_batch 约定 (sit, hop_nn, hop_diag)
            batch_mv = lambda x, _st=st: _stencil_matvec_batch(_st, x)
        # 下一层 matvec：本层 stencil 的 C++ 宽版算子（任意 E），
        # 未传 av/params_template 时回退单线程 Python（coarse_ops=None）。
        if coarse_ops is not None and av is not None and params_template is not None:
            coarse_ops = [
                CudaCoarseSchurOp(av, E_c, lat_coarse_odd,
                                  (hnn.to(device), hdg.to(device), sit.to(device)),
                                  device=device, params=params_template)
                for _ in range(len(matvec_ops))]
            created_ops.extend(coarse_ops)
        else:
            coarse_ops = None
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    for o in created_ops:
        o.release()
    return lonvs, hnn_l, hdg_l, sit_l


def _setup_gpu_tensors(params_t, argv_t, set_ptrs_t, device, mass, atol, csw,
                       dt, seed=42, verbose=False, gauge=None, source=None):
    """在指定设备上生成规范场/Clover 项/源（C++ 后端）。

    返回 (g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa)。
    """
    params_t[define._PARITY_] = 0
    params_t[define._NODE_RANK_] = 0
    params_t[define._NODE_SIZE_] = 1
    params_t[define._DAGGER_] = 0
    params_t[define._MAX_ITER_] = 1000
    params_t[define._VERBOSE_] = 1 if verbose else 0
    params_t[define._SEED_] = seed
    params_t[define._TEST_IN_CPU_] = 0
    av = argv_t.to(dtype=define.dtype(define.epytd(dt)).to_real())
    av[define._MASS_] = mass
    av[define._ATOL_] = atol
    av[define._SIGMA_] = 0.1
    av[define._CSW_] = csw
    ls = define.lat_shape(params_t)
    # P100(sm_60) 兼容：torch 2.10 无 sm_60 kernel image，zeros/randn 填充内核
    # 在 P100 上失败并污染 CUDA 错误状态；empty（纯 cudaMalloc）与 CPU 生成 +
    # H2D 拷贝（driver memcpy）均可用。C++ 后端会写满 g/ce/coo 等输出缓冲。
    g = torch.empty([2, 3, 3, 4] + ls, dtype=dt, device=device)
    # 与规范场 seed 一样固定源场的随机流；这样单卡/多卡性能和数值对照
    # 使用完全相同的输入，而不是被全局 RNG 的调用顺序悄悄改变。
    if source is None:
        fi_gen = torch.Generator(device='cpu').manual_seed(int(seed) + 1)
        fi = torch.randn([2, 4, 3] + ls, dtype=dt, device='cpu',
                         generator=fi_gen).to(device)
    else:
        fi = torch.empty([2, 4, 3] + ls, dtype=dt, device=device)
    ce = torch.empty([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
    if gauge is not None:
        gauge_t = gauge.to(device=device, dtype=dt).contiguous()
        if tuple(gauge_t.shape) != tuple(g.shape):
            raise ValueError(
                f"external gauge shape {tuple(gauge_t.shape)} does not match {tuple(g.shape)}")
        g.copy_(gauge_t)
    if source is not None:
        source_t = source.to(device=device, dtype=dt).contiguous()
        if tuple(source_t.shape) != tuple(fi.shape):
            raise ValueError(
                f"external source shape {tuple(source_t.shape)} does not match {tuple(fi.shape)}")
        fi.copy_(source_t)
    params_t[define._SET_INDEX_] = 0; params_t[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs_t, params_t, av)
    if gauge is None:
        qcu.applyGaussGaugeQcu(g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2; params_t[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs_t, params_t, av); qcu.applyCloversQcu(ce, cei, g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2; params_t[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs_t, params_t, av); qcu.applyCloversQcu(coo, coi, g, set_ptrs_t, params_t)
    U_full = tools.poooxyzt2oooxyzt(g)
    b_full = tools.poooxyzt2oooxyzt(fi)
    kappa = 1.0 / (2 * mass + 8)
    clover_full = dslash.make_clover(U_full, kappa=kappa, csw=csw)
    return g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av


class MultiGpuMultigrid(object):
    """测试专用：单进程多线程多 GPU 的 Legacy Clover-MG 驱动。

    该类只在显式构造时启用，不属于 CUDA 生产执行拓扑。生产程序必须保持
    ``一个 Python 进程 = 一个 MPI rank = 一张 GPU/DCU/GPGPU``；多设备
    协同通过 MPI 格点分域完成，而不是在一个进程内创建多个整题求解 worker。

    构造参数：
        lat_size: [Lx, Ly, Lz, Lt]
        mass / atol: 求解质量参数与容差
        num_levels / dof_list / mg_grid / num_restart / coarse_max_iter /
        coarse_tol_factor / nv_iters: 粗网格配置（同 conftest.schur.multigrid）
        nthreads: 线程数（默认 = 设备数）
        device_ids: 显式设备 id 列表（默认 range(可见设备数)；
            线程 tid 绑定 device_ids[tid % len(device_ids)]）
        use_cache / cache_dir: 粗网格算子 h5py 缓存
        verbose: 详细输出；seed: 规范场与源场的可复现实验种子
        run_reference: 是否在线计算 BiStabCG 参考解；一致性验证需要开启，
            纯 MG 吞吐模式可关闭以省去一次完整求解和独立 LatticeSet 初始化

    方法：
        solve(): 并行求解（每线程一个完整 MG 流程），返回结果字典
        verify_consistency(tol): 线程间解与主线程参考解的一致性校验
    """

    def __init__(self, lat_size=None, mass=0.05, atol=1e-6,
                 num_levels=2,
                 dof_list=None, mg_grid=None, num_restart=5, coarse_max_iter=15,
                 coarse_tol_factor=1e5, nv_iters=2, nthreads=None, device_ids=None,
                 use_cache=True, cache_dir=None, verbose=True,
                 independent_problems=False, seed=42, csw=1.0,
                 run_reference=True, gauge=None, source=None,
                 input_tag=None, verify_residual=False, use_deflate=False):
        from mpi4py import MPI
        # C++ LatticeSet::init 用 MPI_COMM_WORLD 真实 rank 覆盖 _NODE_RANK_，
        # 多线程独立实例语义要求单 rank（多进程分布走每进程单线程实例路径）。
        if MPI.COMM_WORLD.Get_size() > 1:
            raise RuntimeError(
                "PYQCU::CUDA::MULTI_GPU:\n MultiGpuMultigrid requires a single "
                "MPI rank (mpirun -np 1): multi-process distribution uses "
                "per-rank single-thread instances; multi-GPU parallelism inside "
                "one process uses one-thread-one-GPU.")
        if lat_size is None:
            lat_size = [8, 8, 8, 16]
        if mg_grid is None:
            mg_grid = [2, 2, 2, 2]
        if not math.isfinite(float(mass)) or 2.0 * float(mass) + 8.0 == 0.0:
            raise ValueError("mass must be finite and must not make kappa infinite")
        if not math.isfinite(float(atol)) or float(atol) <= 0.0:
            raise ValueError("atol must be finite and positive")
        if not math.isfinite(float(csw)):
            raise ValueError("csw must be finite")
        self.lat_size = list(lat_size)
        self.mass = mass
        self.atol = atol
        self.csw = float(csw)
        self.num_levels = num_levels
        if dof_list is None:
            dof_list = [12, 48, 48, 48, 48]
        self.dof_list = dof_list[:num_levels]
        self.mg_grid = mg_grid
        self.num_restart = num_restart
        self.coarse_max_iter = coarse_max_iter
        self.coarse_tol_factor = coarse_tol_factor
        self.nv_iters = nv_iters
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus == 0:
            raise RuntimeError("PYQCU::CUDA::MULTI_GPU:\n no CUDA device available")
        if device_ids is None:
            device_ids = list(range(num_gpus))
        self.device_ids = [int(d) for d in device_ids]
        if nthreads is None:
            nthreads = len(self.device_ids)
        self.nthreads = nthreads
        self.dt = torch.complex64
        self.independent_problems = independent_problems
        self.run_reference = bool(run_reference)
        self.seed = int(seed)
        if independent_problems and (gauge is not None or source is not None):
            raise ValueError(
                "external gauge/source are supported only for a shared problem")
        self.gauge = gauge
        self.source = source
        self.input_tag = input_tag
        self.verify_residual = bool(verify_residual)
        self.use_deflate = bool(use_deflate)
        self._results = None
        self._shared = None
        self.last_timings = None

    def _build_coarse_ops(self, S, E, device):
        """主线程构建 Schur 粗网格算子（h5py 缓存）。"""
        return build_schur_levels(
            op=None, S=S, num_levels=self.num_levels, dof_list=self.dof_list,
            mg_grid=self.mg_grid, lat_full=self.lat_size, E=E, dt=self.dt,
            device=device, nv_iters=self.nv_iters, use_cache=self.use_cache,
            cache_dir=self.cache_dir, verbose=self.verbose,
            mass=self.mass, csw=self.csw, seed=self.seed,
            cache_tag=self.input_tag or f"generated:{self.seed}")

    def _config_params(self, params_t, argv_t, set_ptrs_t, av):
        Lx, Ly, Lz, Lt = self.lat_size
        params_t[define._LAT_X_] = Lx; params_t[define._LAT_Y_] = Ly
        params_t[define._LAT_Z_] = Lz; params_t[define._LAT_T_] = Lt
        params_t[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
        # 每线程/每进程独立完整实例：单 rank 网格（无 MPI 分布 halo），
        # 多卡并行通过线程×卡绑定实现，不依赖 MPI 通信。
        params_t[define._GRID_X_], params_t[define._GRID_Y_], \
            params_t[define._GRID_Z_], params_t[define._GRID_T_] = 1, 1, 1, 1
        params_t[define._NODE_RANK_] = 0
        params_t[define._NODE_SIZE_] = 1
        params_t[define._DATA_TYPE_] = define.epytd(self.dt)
        # mass/atol/sigma 必须显式设置（模块级 argv 默认 mass=-3.5 会奇异）
        av[define._MASS_] = self.mass
        av[define._ATOL_] = self.atol
        av[define._SIGMA_] = 0.1
        av[define._CSW_] = self.csw
        params_t[define._MG_NUM_LEVEL_] = self.num_levels
        params_t[define._MG_USE_DEFLATE_] = int(self.use_deflate)
        if self.num_levels >= 2:
            params_t[define._MG_LEVEL1_E_] = self.dof_list[1]
            params_t[define._MG_LEVEL1_X_] = Lx // self.mg_grid[0]
            params_t[define._MG_LEVEL1_Y_] = Ly // self.mg_grid[1]
            params_t[define._MG_LEVEL1_Z_] = Lz // self.mg_grid[2]
            params_t[define._MG_LEVEL1_T_] = Lt // (2 * self.mg_grid[3])
            params_t[define._MG_LEVEL1_MAX_ITER_] = self.coarse_max_iter
            params_t[define._MG_LEVEL1_DATA_TYPE_] = define.epytd(self.dt)
            params_t[define._MG_LEVEL1_NUM_RESTART_] = self.num_restart
        if self.num_levels >= 3:
            params_t[define._MG_LEVEL2_E_] = self.dof_list[2]
            params_t[define._MG_LEVEL2_X_] = Lx // (self.mg_grid[0] * self.mg_grid[0])
            params_t[define._MG_LEVEL2_Y_] = Ly // (self.mg_grid[1] * self.mg_grid[1])
            params_t[define._MG_LEVEL2_Z_] = Lz // (self.mg_grid[2] * self.mg_grid[2])
            params_t[define._MG_LEVEL2_T_] = Lt // (4 * self.mg_grid[3])
            params_t[define._MG_LEVEL2_MAX_ITER_] = 200
            params_t[define._MG_LEVEL2_DATA_TYPE_] = define.epytd(self.dt)
            params_t[define._MG_LEVEL2_NUM_RESTART_] = 3
        av[define._MG_LEVEL1_ATOL_] = self.atol * self.coarse_tol_factor
        if self.num_levels >= 3:
            av[define._MG_LEVEL2_ATOL_] = self.atol * self.coarse_tol_factor

    def _worker(self, tid, shared):
        """单线程完整 MG 流程（绑定 device_ids[tid % n]）。

        independent_problems=True 时每线程独立生成规范场/Clover/源与粗算子
        （每线程不同 seed），无共享状态 —— 多卡吞吐并行模式；
        否则共享输入 + 每线程拷贝（一致性验证模式）。
        """
        dev_id = self.device_ids[tid % len(self.device_ids)]
        torch.cuda.set_device(dev_id)
        dev = torch.device(f'cuda:{dev_id}')
        params_t, argv_t, set_ptrs_t = _clone_state()
        self._config_params(params_t, argv_t, set_ptrs_t, argv_t)
        if self.independent_problems:
            # 独立问题：规范场/Clover/源与粗算子由主线程在 V100 预构建
            # （P100 无 torch kernel image，粗算子构建的 torch 运算只能在 V100
            # 完成），本线程只做 D2D 拷贝 + C++ 求解。
            st = shared['t'][tid]
            g = st['g'].to(dev); fi = st['fi'].to(dev)
            ce = st['ce'].to(dev); cei = st['cei'].to(dev)
            coo = st['coo'].to(dev); coi = st['coi'].to(dev)
            coarse = st['coarse']
        else:
            # 共享输入（规范场/Clover/源）拷贝到本卡
            g = shared['g'].to(dev); fi = shared['fi'].to(dev)
            ce = shared['ce'].to(dev); cei = shared['cei'].to(dev)
            coo = shared['coo'].to(dev); coi = shared['coi'].to(dev)
            coarse = shared['coarse']
        # 2026-08-20 dev80 fix: 共享/独立模式均由主线程在 V100 预生成 gauge/clover 并拷贝，
        # worker 不再本地重新生成（P100 sm_60 上 GaussGauge/BiStabCG 内核缺 image 且浪费显存）。
        # 原逻辑共享模式每 worker 重跑 GaussGauge（seed 一致）+ 2 次 Clovers（各需 LatticeSet scratch），
        # 在 32^4 上每 worker 额外 ~3GB scratch 且 P100 上直接 no kernel image。
        # 现改为零拷贝复用：g/ce/coo 均为 V100 预生成后 D2D 拷贝到本卡，worker 仅需 BiStabCG/MG 的 plan1 LatticeSet。
        # 若需独立 gauge 演化，改为主线程按 seed=42+tid 预生成（solve() 独立分支已做）。
        pass  # gauge/clover 已由主线程生成并拷贝，无需 worker 本地重建
        # 保留占位以保持 set_ptrs 槽位计数与旧逻辑兼容（但不再 init 0/1/2）
        # 粗网格算子拷贝到本卡，填入 set_ptrs 槽位
        # 注意：.to(dev) 在跨设备时产生新张量，必须保留引用（_coarse_dev），
        # 否则临时对象被 GC 回收后 data_ptr 悬垂，C++ 求解读到垃圾（nan）。
        _coarse_dev = []
        for fl, (lonv, hnn, hdg, sit) in enumerate(coarse):
            base = _SET_PTRS_COARSE_BASE_ + 4 * fl
            lonv_d, hnn_d, hdg_d, sit_d = (lonv.to(dev).contiguous(),
                                           hnn.to(dev).contiguous(),
                                           hdg.to(dev).contiguous(),
                                           sit.to(dev).contiguous())
            _coarse_dev.extend([lonv_d, hnn_d, hdg_d, sit_d])
            set_ptrs_t[base + 0] = lonv_d.data_ptr()
            set_ptrs_t[base + 1] = hnn_d.data_ptr()
            set_ptrs_t[base + 2] = hdg_d.data_ptr()
            set_ptrs_t[base + 3] = sit_d.data_ptr()
        initialized_slots = []
        try:
            fo_ref = None
            ref_time = None
            if self.run_reference:
                params_t[define._SET_INDEX_] = 0
                params_t[define._SET_PLAN_] = 1
                params_t[define._VERBOSE_] = 0
                qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
                initialized_slots.append(0)
                fo_ref = torch.empty_like(fi)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                qcu.applyCloverBistabCgQcu(
                    fo_ref, fi, g, ce, coo, cei, coi, set_ptrs_t, params_t)
                torch.cuda.synchronize()
                ref_time = time.perf_counter() - t0

            mg_slot = 1 if self.run_reference else 0
            fo_mg = torch.empty_like(fi)
            params_t[define._SET_INDEX_] = mg_slot
            params_t[define._SET_PLAN_] = 1
            params_t[define._VERBOSE_] = 0
            qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
            initialized_slots.append(mg_slot)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            qcu.applyCloverMultigridQcu(
                fo_mg, fi, g, ce, coo, cei, coi, set_ptrs_t, params_t)
            torch.cuda.synchronize()
            mg_time = time.perf_counter() - t0
            mg_residual = None
            mg_residual_seconds = None
            if self.verify_residual:
                residual_started = time.perf_counter()
                full_g = tools.poooxyzt2oooxyzt(g)
                full_x = tools.poooxyzt2oooxyzt(fo_mg)
                full_b = tools.poooxyzt2oooxyzt(fi)
                full_clover = tools.poooxyzt2oooxyzt(
                    torch.stack((ce, coo), dim=0))
                # applyClovers returns A=I+T while the independent full
                # operator expects the additive clover term T because
                # give_wilson already supplies the identity.
                full_clover = dslash.cut_I(full_clover)
                kappa_t = torch.tensor(
                    [1.0 / (2.0 * self.mass + 8.0)], dtype=full_g.real.dtype,
                    device=dev)
                u0 = torch.ones([1], dtype=full_g.real.dtype, device=dev)
                residual = (
                    dslash.give_wilson(
                        full_x, full_g, kappa_t, u0, with_I=True) +
                    dslash.give_clover(full_x, full_clover) - full_b)
                mg_residual = float(
                    torch.linalg.vector_norm(residual).item() /
                    torch.linalg.vector_norm(full_b).item())
                mg_residual_seconds = time.perf_counter() - residual_started
        finally:
            for slot in reversed(initialized_slots):
                params_t[define._SET_INDEX_] = slot
                qcu.applyEndQcu(set_ptrs_t, params_t)
        return {'tid': tid, 'device': dev_id, 'mg': fo_mg.cpu(),
                'ref': None if fo_ref is None else fo_ref.cpu(),
                'ref_time': ref_time, 'mg_time': mg_time,
                'mg_residual': mg_residual,
                'mg_residual_seconds': mg_residual_seconds}

    def solve(self):
        """并行求解：每线程一个完整 C++ MG 流程。返回 {'threads': [...], 'results': ...}。"""
        from concurrent.futures import ThreadPoolExecutor
        solve_started = time.perf_counter()
        # 主线程 Python 层运算（粗算子构建、poooxyzt2oooxyzt、make_clover 等）
        # 统一在 cuda:0（V100, sm_70+）：P100(sm_60) 无 torch kernel image，
        # 任何 torch 内核在 P100 上都会报 cudaErrorNoKernelImageForDevice；
        # workers 各卡只做 D2D 拷贝 + C++ 求解（libqcu.so 含 sm_60 SASS）。
        main_dev = torch.device('cuda:0')
        # BUGFIX 2026-08-15: 主线程 current device 必须锁定 V100。上一实例
        # （或本实例共享模式结尾）可能把主线程 current device 切到 P100
        # （torch.cuda.set_device 是进程级、C++ 线程局部 current device 跟随），
        # 导致后续实例主线程的 C++ 调用（applyInitQcu/applyGaussGaugeQcu 等）
        # 在 P100 上执行而张量在 V100 → illegal memory access。
        # （CUDA current device 是线程局部的：主线程显式设定 V100 后，
        #   worker 线程各自 set_device 不受影响。）
        torch.cuda.set_device(int(main_dev.index))
        # 主线程预热 torch lazy 初始化（clover inverse 等）：worker 线程并发
        # 首次触发 torch.linalg.inv 等 lazy backend 会报 "lazy wrapper should
        # be called at most once"，预热后在主线程完成初始化。
        try:
            _w = torch.randn([4, 4], dtype=self.dt, device=main_dev)
            torch.linalg.inv(_w)
            torch.cuda.synchronize()
        except Exception:
            pass
        if self.independent_problems:
            # 每线程独立问题：主线程在 V100 上为每线程预构建规范场/Clover/源与
            # 粗算子（P100 无 torch kernel image，粗算子构建的 torch 运算只能在
            # V100 完成；线程只在各自卡上做 D2D 拷贝 + C++ 求解）。
            # 各线程 seed 不同 → 缓存 key 相同 → 必须按线程分目录，否则多线程
            # 写同一缓存文件互相污染。
            shared_t = {}
            for tid in range(self.nthreads):
                pt, at, st = _clone_state()
                self._config_params(pt, at, st, at)
                g, fi, ce, cei, coo, coi, _, _, _, _, av = _setup_gpu_tensors(
                    pt, at, st, main_dev, self.mass, self.atol, self.csw,
                    self.dt, seed=self.seed + tid, verbose=False)
                cache_t = (os.path.join(self.cache_dir, f"tid{tid}")
                           if self.cache_dir else None)
                ops_t = [CudaSchurOp(av, g, ce, coo, cei, coi, params=pt)]
                coarse = list(zip(*build_schur_levels(
                    None, None, self.num_levels, self.dof_list, self.mg_grid,
                    self.lat_size, self.dof_list[1], self.dt, main_dev,
                    nv_iters=self.nv_iters, use_cache=self.use_cache,
                    cache_dir=cache_t, verbose=False,
                    matvec_ops=ops_t, nthreads=1, av=av, params_template=pt,
                    mass=self.mass, csw=self.csw, seed=self.seed + tid,
                    cache_tag=f"generated:{self.seed + tid}")))
                for o in ops_t:
                    o.release()
                for _idx in (0, 1, 2):
                    pt[define._SET_INDEX_] = _idx
                    qcu.applyEndQcu(st, pt)
                shared_t[tid] = {'g': g, 'fi': fi, 'ce': ce, 'cei': cei,
                                 'coo': coo, 'coi': coi, 'coarse': coarse}
            threads = []
            from concurrent.futures import ThreadPoolExecutor as _TPE
            setup_seconds = time.perf_counter() - solve_started
            with _TPE(max_workers=self.nthreads) as ex:
                for r in ex.map(lambda tid: self._worker(tid, {'t': shared_t}),
                                range(self.nthreads)):
                    threads.append(r)
            results = {'threads': threads, 'lat_size': self.lat_size,
                       'mass': self.mass, 'atol': self.atol, 'csw': self.csw,
                       'nthreads': self.nthreads, 'device_ids': self.device_ids,
                       'num_levels': self.num_levels, 'dof_list': self.dof_list,
                       'independent_problems': True,
                       'run_reference': self.run_reference}
            self._results = results
            self.last_timings = {
                'setup_seconds': setup_seconds,
                'wall_seconds': time.perf_counter() - solve_started,
            }
            return results
        # Keep the prepared gauge/Clover/coarse hierarchy resident across solves.
        # Rebuilding it for every warmup/repeat both dominated latency and raced
        # PyTorch's cached allocator against the C++ cudaMallocAsync pool.
        setup_reused = self._shared is not None
        if setup_reused:
            shared = self._shared
            setup_seconds = 0.0
        else:
            params_t, argv_t, set_ptrs_t = _clone_state()
            self._config_params(params_t, argv_t, set_ptrs_t, argv_t)
            g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av = \
                _setup_gpu_tensors(params_t, argv_t, set_ptrs_t, main_dev,
                                   self.mass, self.atol, self.csw, self.dt,
                                   seed=self.seed, verbose=self.verbose,
                                   gauge=self.gauge, source=self.source)
            op = dslash.operator(U=U_full, clover_term=clover_full,
                                 kappa=torch.Tensor([kappa]), support_parity=True,
                                 verbose=False)
            S = op.matvec_parity
            # 粗算子构建：缓存命中时免 C++ 构建（24³×72 大格子显存优化）。
            cache_hit = False
            if self.use_cache and self.cache_dir is not None:
                try:
                    lat_fine_odd = [self.lat_size[0], self.lat_size[1],
                                    self.lat_size[2], self.lat_size[3] // 2]
                    all_hit = True
                    for lvl in range(1, self.num_levels):
                        E_c = self.dof_list[lvl]
                        lat_coarse_odd = [
                            lat_fine_odd[d] // self.mg_grid[d] for d in range(4)]
                        paths = _coarse_cache_paths(
                            self.cache_dir, self.lat_size, lvl, E_c,
                            self.nv_iters, 1e-2, mass=self.mass, csw=self.csw,
                            seed=self.seed,
                            cache_tag=self.input_tag or f"generated:{self.seed}")
                        if not any(os.path.exists(cf) for cf in paths):
                            all_hit = False
                            break
                        lat_fine_odd = lat_coarse_odd
                    cache_hit = all_hit
                except Exception:
                    cache_hit = False
            if self.num_levels == 1:
                coarse = ([], [], [], [])
                ops_build = []
            elif cache_hit:
                ops_build = []
                coarse = build_schur_levels(
                    op, S, self.num_levels, self.dof_list, self.mg_grid,
                    self.lat_size, self.dof_list[1], self.dt, main_dev,
                    nv_iters=self.nv_iters, use_cache=self.use_cache,
                    cache_dir=self.cache_dir, verbose=False, matvec_ops=None,
                    nthreads=1, av=None, params_template=None, mass=self.mass,
                    csw=self.csw, seed=self.seed,
                    cache_tag=self.input_tag or f"generated:{self.seed}")
            else:
                ops_build = [
                    CudaSchurOp(av, g, ce, coo, cei, coi, params=params_t)
                    for _ in range(max(1, min(self.nthreads, 4)))]
                coarse = build_schur_levels(
                    op, S, self.num_levels, self.dof_list, self.mg_grid,
                    self.lat_size, self.dof_list[1], self.dt, main_dev,
                    nv_iters=self.nv_iters, use_cache=self.use_cache,
                    cache_dir=self.cache_dir, verbose=False,
                    matvec_ops=ops_build, nthreads=len(ops_build), av=av,
                    params_template=params_t, mass=self.mass, csw=self.csw,
                    seed=self.seed,
                    cache_tag=self.input_tag or f"generated:{self.seed}")
                for o in ops_build:
                    o.release()
            shared = {
                'g': g, 'fi': fi, 'ce': ce, 'cei': cei,
                'coo': coo, 'coi': coi,
                'coarse': list(zip(*coarse)) if coarse and coarse[0] else [],
            }
            self._shared = shared
            self.gauge = None
            self.source = None
            # The full-site Python operator and clover tensors are setup-only.
            # C++ workers consume the packed parity fields and HDF5 coarse
            # assets, so retaining these duplicates needlessly approaches the
            # V100 capacity limit.
            del U_full, b_full, clover_full, op, S
            torch.cuda.empty_cache()
            # 清理主线程临时 LatticeSet（setup 用），避免槽位与工作线程混淆
            params_t[define._SET_INDEX_] = 0
            params_t[define._SET_PLAN_] = -1
            qcu.applyEndQcu(set_ptrs_t, params_t)
            params_t[define._SET_INDEX_] = 1
            qcu.applyEndQcu(set_ptrs_t, params_t)
            params_t[define._SET_INDEX_] = 2
            qcu.applyEndQcu(set_ptrs_t, params_t)
            setup_seconds = time.perf_counter() - solve_started
        threads = []
        with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
            for r in ex.map(lambda tid: self._worker(tid, shared), range(self.nthreads)):
                threads.append(r)
        # BUGFIX 2026-08-15: 移除污染主线程 current device 的 set_device
        # （历史残留：无主线程参考解计算）。主线程 device 已由 solve() 开头
        # 锁定 V100，连续多实例安全（否则下一实例主线程 C++ 在 P100 上写
        # V100 张量 → illegal memory access）。
        results = {'threads': threads, 'lat_size': self.lat_size,
                   'mass': self.mass, 'atol': self.atol, 'csw': self.csw,
                   'nthreads': self.nthreads, 'device_ids': self.device_ids,
                   'num_levels': self.num_levels, 'dof_list': self.dof_list,
                   'run_reference': self.run_reference,
                   'verify_residual': self.verify_residual,
                   'input_tag': self.input_tag}
        self._results = results
        self.last_timings = {
            'setup_seconds': setup_seconds,
            'setup_reused': setup_reused,
            'wall_seconds': time.perf_counter() - solve_started,
        }
        return results

    def close(self):
        """Release resident input/coarse tensors held for repeated solves."""
        self._shared = None
        self._results = None
        self.last_timings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_report(self, path: str):
        """导出求解报告（JSON：配置/每线程耗时/残差/一致性）。"""
        import json
        if self._results is None:
            raise RuntimeError("PYQCU::CUDA::MULTI_GPU:\n call solve() first")
        rep = dict(self._results)
        rep['threads'] = [{k: (float(v) if torch.is_tensor(v) and v.numel() == 1
                               else (v.tolist() if torch.is_tensor(v) and v.numel() <= 16
                                     else (f"<tensor{v.shape}>" if torch.is_tensor(v) else v)))
                           for k, v in t.items()} for t in rep['threads']]
        rep['device_ids'] = list(rep['device_ids'])
        rep['dof_list'] = list(rep['dof_list'])
        rep['lat_size'] = list(rep['lat_size'])
        with open(path, 'w') as f:
            json.dump(rep, f, indent=2)
        print(f"PYQCU::CUDA::MULTI_GPU:\n report saved to {path}")

    def verify_consistency(self, tol: float = 1e-5) -> dict:
        """校验各线程解与线程 0 解（参考）的一致性（相对误差 < tol）。"""
        if self._results is None:
            raise RuntimeError("PYQCU::CUDA::MULTI_GPU:\n call solve() first")
        threads = self._results['threads']
        ref = threads[0]['mg']
        checks = []
        for t in threads:
            rel = float((t['mg'] - ref).abs().max() / (ref.abs().max() + 1e-30))
            checks.append({'tid': t['tid'], 'device': t['device'],
                           'rel_max_diff': rel, 'pass': rel < tol})
        all_pass = all(c['pass'] for c in checks)
        summary = {'tol': tol, 'checks': checks, 'all_pass': all_pass}
        if self.verbose:
            print(f"PYQCU::CUDA::MULTI_GPU:\n consistency verify (tol={tol}): "
                  f"{'PASS' if all_pass else 'FAIL'}")
            for c in checks:
                print(f"PYQCU::CUDA::MULTI_GPU:\n   tid={c['tid']} device={c['device']} "
                      f"rel_max_diff={c['rel_max_diff']:.3e}")
        return summary


def verify_multi_gpu_mg(**kwargs):
    """一行式验证入口（供 pyqcu.testing 使用）。

    kwargs 透传 MultiGpuMultigrid（lat_size/mass/atol/nthreads/...）。
    返回 (results, consistency)。
    """
    mg = MultiGpuMultigrid(**kwargs)
    results = mg.solve()
    consistency = mg.verify_consistency(tol=kwargs.get('tol', 1e-5))
    return results, consistency
