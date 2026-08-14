"""多线程多卡 C++ Clover Multigrid 驱动（一线程一卡）。

模式：N 个线程并行，每个线程绑定一张 GPU（device_id = tid % num_gpus），
各自在绑定的设备上运行完整的 C++ 后端流程：
  生成/拷贝规范场与 Clover 项 → 参考 BiStabCG（applyCloverBistabCgQcu）
  → 完整 Clover Multigrid V-cycle（applyCloverMultigridQcu）。

线程隔离约定：
  * 每线程持独立 params/argv/set_ptrs 副本（int32[54]/float[7]/int64[100]），
    _SET_INDEX_ 在各自副本中从 0 起独立计数 —— 无共享写竞争。
  * 每线程独立 CUDA 上下文：worker 内先 torch.cuda.set_device(dev_id)
    （CUDA 运行时 API 的 current device 是线程局部的）。
  * Cython 桥（pyqcu.cuda.qcu）所有函数均以 with nogil 释放 GIL 后调用
    C++ 后端 —— 多线程可真正并行进入 libqcu.so，一线程一卡才有效。
  * 粗网格 Schur 算子（33-tensor）在主线程构建一次（确定性探测，写集
    不相交），各线程拷贝到本卡后填入自己的 set_ptrs 槽位（只读共享）。

验证语义：默认所有线程求解同一问题（同 seed 生成的规范场/源），
solve() 返回各线程解，verify_consistency() 校验线程间一致性
（与主线程单线程参考解相对误差 < tol）—— 多卡正确性的等价判据。
"""

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

_SET_PTRS_COARSE_BASE_ = 30


def _clone_state():
    """每线程独立 params/argv/set_ptrs 副本。"""
    return (_module_params.clone(), _module_argv.clone(), _module_set_ptrs.clone())


def build_schur_levels(op, S, num_levels, dof_list, mg_grid, lat_full, E, dt, device,
                       nv_iters=2, use_cache=True, cache_dir=None, verbose=True,
                       matvec_ops=None, nthreads=None):
    """构建 S 的 null 向量 + 33-tensor Galerkin 粗算子 A_c = P^T S P。

    返回 (lonvs, hnn_l, hdg_l, sit_l) 列表（每粗层一项）。
    结果以 h5py 缓存到磁盘（keyed by lattice/dof/nv_iters），
    重复运行跳过昂贵的 setup（与任务②的 h5py 多线程读写约定一致）。

    matvec_ops: 可选，每线程一个 matvec 算子（如 CudaSchurOp 实例列表）——
                给定时 null 向量生成（give_null_vecs_mt）与 stencil 构建
                （build_stencil_mt）均多线程并行，适合多卡/多线程加速；
                否则用单线程 Python matvec S。
    """
    from pyqcu.tools import build_stencil, apply_stencil, build_stencil_mt, give_null_vecs_mt
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~/PyQCU/logs"), "nullvec_cache")
    os.makedirs(cache_dir, exist_ok=True)
    lonvs, hnn_l, hdg_l, sit_l = [], [], [], []
    lat_fine_odd = [lat_full[0], lat_full[1], lat_full[2], lat_full[3] // 2]
    E_prev = 12
    for lvl in range(1, num_levels):
        E_c = dof_list[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // mg_grid[d] for d in range(4)]
        tag = f"L{lat_full[0]}x{lat_full[1]}x{lat_full[2]}x{lat_full[3]}_lv{lvl}_E{E_c}_nvi{nv_iters}"
        cache_file = os.path.join(cache_dir, tag + ".h5")
        if use_cache and os.path.exists(cache_file):
            lonv = tools.load_tensor_h5(cache_file, dataset="lonv", device=device)
            hnn = tools.load_tensor_h5(cache_file, dataset="hnn", device=device)
            hdg = tools.load_tensor_h5(cache_file, dataset="hdg", device=device)
            sit = tools.load_tensor_h5(cache_file, dataset="sit", device=device)
            if verbose:
                print(f"  [level {lvl}] E={E_c} CACHED coarse={lat_coarse_odd}")
        else:
            t0 = time.perf_counter()
            if matvec_ops is not None:
                _null = give_null_vecs_mt(
                    matvec_ops, E_c, E_prev, lat_fine_odd, dt, device,
                    nv_iters=nv_iters, nthreads=nthreads, verbose=False)
            else:
                _null = torch.randn([E_c, E_prev] + lat_fine_odd, dtype=dt, device=device)
                for _ in range(nv_iters):
                    _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None,
                                                 verbose=False)
            lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse_odd,
                                             verbose=False)
            if matvec_ops is not None:
                hnn, hdg, sit = build_stencil_mt(
                    matvec_ops, lonv, E_c, E_prev, lat_fine_odd, lat_coarse_odd,
                    dt, device, nthreads=nthreads or len(matvec_ops), verbose=verbose)
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
        lonvs.append(lonv); hnn_l.append(hnn); hdg_l.append(hdg); sit_l.append(sit)
        S = lambda v, _hnn=hnn, _hdg=hdg, _sit=sit: apply_stencil(_hnn, _hdg, _sit, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    return lonvs, hnn_l, hdg_l, sit_l


def _setup_gpu_tensors(params_t, argv_t, set_ptrs_t, device, mass, atol, dt, seed=42,
                       verbose=False):
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
    ls = define.lat_shape(params_t)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)
    params_t[define._SET_INDEX_] = 0; params_t[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs_t, params_t, av); qcu.applyGaussGaugeQcu(g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2; params_t[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs_t, params_t, av); qcu.applyCloversQcu(ce, cei, g, set_ptrs_t, params_t)
    params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2; params_t[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs_t, params_t, av); qcu.applyCloversQcu(coo, coi, g, set_ptrs_t, params_t)
    U_full = tools.poooxyzt2oooxyzt(g)
    b_full = tools.poooxyzt2oooxyzt(fi)
    kappa = 1.0 / (2 * mass + 8)
    clover_full = dslash.make_clover(U_full, kappa=kappa)
    return g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av


class MultiGpuMultigrid(object):
    """多线程多卡 C++ Clover Multigrid 驱动（一线程一卡）。

    构造参数：
        lat_size: [Lx, Ly, Lz, Lt]
        mass / atol: 求解质量参数与容差
        num_levels / dof_list / mg_grid / num_restart / coarse_max_iter /
        coarse_tol_factor / nv_iters: 粗网格配置（同 conftest.schur.multigrid）
        nthreads: 线程数（默认 = 设备数）
        device_ids: 显式设备 id 列表（默认 range(可见设备数)；
            线程 tid 绑定 device_ids[tid % len(device_ids)]）
        use_cache / cache_dir: 粗网格算子 h5py 缓存
        verbose: 详细输出

    方法：
        solve(): 并行求解（每线程一个完整 MG 流程），返回结果字典
        verify_consistency(tol): 线程间解与主线程参考解的一致性校验
    """

    def __init__(self, lat_size=None, mass=0.05, atol=1e-6, num_levels=2,
                 dof_list=None, mg_grid=None, num_restart=10, coarse_max_iter=200,
                 coarse_tol_factor=1e4, nv_iters=2, nthreads=None, device_ids=None,
                 use_cache=True, cache_dir=None, verbose=True,
                 independent_problems=False):
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
        self.lat_size = list(lat_size)
        self.mass = mass
        self.atol = atol
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
        self._results = None

    def _build_coarse_ops(self, S, E, device):
        """主线程构建 Schur 粗网格算子（h5py 缓存）。"""
        return build_schur_levels(
            op=None, S=S, num_levels=self.num_levels, dof_list=self.dof_list,
            mg_grid=self.mg_grid, lat_full=self.lat_size, E=E, dt=self.dt,
            device=device, nv_iters=self.nv_iters, use_cache=self.use_cache,
            cache_dir=self.cache_dir, verbose=self.verbose)

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
        params_t[define._MG_NUM_LEVEL_] = self.num_levels
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
            g, fi, ce, cei, coo, coi, U_f, b_f, cl_f, kap, av = _setup_gpu_tensors(
                params_t, argv_t, set_ptrs_t, dev, self.mass, self.atol,
                self.dt, seed=42 + tid, verbose=False)
            # 粗算子构建直接用 C++ matvec（避免 worker 内 Python clover inverse
            # 的 torch lazy 初始化并发冲突；也更快）。
            # 独立问题各线程 seed 不同 → 缓存 key 相同 → 必须按线程分目录，
            # 否则多线程写同一缓存文件互相污染。
            cache_t = (os.path.join(self.cache_dir, f"tid{tid}")
                       if self.cache_dir else None)
            ops_t = [CudaSchurOp(av, g, ce, coo, cei, coi, params=params_t)]
            coarse = list(zip(*build_schur_levels(
                None, None, self.num_levels, self.dof_list, self.mg_grid,
                self.lat_size, self.dof_list[1], self.dt, dev,
                nv_iters=self.nv_iters, use_cache=self.use_cache,
                cache_dir=cache_t, verbose=False,
                matvec_ops=ops_t, nthreads=1)))
            for o in ops_t: o.release()
        else:
            # 共享输入（规范场/Clover/源）拷贝到本卡
            g = shared['g'].to(dev); fi = shared['fi'].to(dev)
            ce = shared['ce'].to(dev); cei = shared['cei'].to(dev)
            coo = shared['coo'].to(dev); coi = shared['coi'].to(dev)
            coarse = shared['coarse']
        # 独立 LatticeSet 槽位（每线程从自己的 0 开始）
        params_t[define._SET_INDEX_] = 0; params_t[define._SET_PLAN_] = -1
        qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
        qcu.applyGaussGaugeQcu(g, set_ptrs_t, params_t)
        params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2
        params_t[define._PARITY_] = 0
        qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
        qcu.applyCloversQcu(ce, cei, g, set_ptrs_t, params_t)
        params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 2
        params_t[define._PARITY_] = 1
        qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
        qcu.applyCloversQcu(coo, coi, g, set_ptrs_t, params_t)
        # 粗网格算子拷贝到本卡，填入 set_ptrs 槽位
        for fl, (lonv, hnn, hdg, sit) in enumerate(coarse):
            base = _SET_PTRS_COARSE_BASE_ + 4 * fl
            set_ptrs_t[base + 0] = lonv.to(dev).contiguous().data_ptr()
            set_ptrs_t[base + 1] = hnn.to(dev).contiguous().data_ptr()
            set_ptrs_t[base + 2] = hdg.to(dev).contiguous().data_ptr()
            set_ptrs_t[base + 3] = sit.to(dev).contiguous().data_ptr()
        # 参考 BiStabCG
        params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 1
        params_t[define._VERBOSE_] = 0
        qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
        fo_ref = torch.zeros_like(fi)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs_t, params_t)
        torch.cuda.synchronize(); ref_time = time.perf_counter() - t0
        # C++ Clover Multigrid
        fo_mg = torch.zeros_like(fi)
        params_t[define._SET_INDEX_] += 1; params_t[define._SET_PLAN_] = 1
        params_t[define._VERBOSE_] = 0
        qcu.applyInitQcu(set_ptrs_t, params_t, argv_t)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs_t, params_t)
        torch.cuda.synchronize(); mg_time = time.perf_counter() - t0
        return {'tid': tid, 'device': dev_id, 'mg': fo_mg.cpu(), 'ref': fo_ref.cpu(),
                'ref_time': ref_time, 'mg_time': mg_time}

    def solve(self):
        """并行求解：每线程一个完整 C++ MG 流程。返回 {'threads': [...], 'results': ...}。"""
        from concurrent.futures import ThreadPoolExecutor
        main_dev = torch.device(f'cuda:{self.device_ids[0]}')
        if self.independent_problems:
            # 每线程独立问题：主线程仅建公共状态，不做共享构建
            threads = []
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=self.nthreads) as ex:
                for r in ex.map(lambda tid: self._worker(tid, None), range(self.nthreads)):
                    threads.append(r)
            results = {'threads': threads, 'lat_size': self.lat_size,
                       'mass': self.mass, 'atol': self.atol,
                       'nthreads': self.nthreads, 'device_ids': self.device_ids,
                       'num_levels': self.num_levels, 'dof_list': self.dof_list,
                       'independent_problems': True}
            self._results = results
            return results
        # 主线程生成共享输入（规范场/Clover/源）与粗网格算子
        params_t, argv_t, set_ptrs_t = _clone_state()
        self._config_params(params_t, argv_t, set_ptrs_t, argv_t)
        g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av = \
            _setup_gpu_tensors(params_t, argv_t, set_ptrs_t, main_dev, self.mass,
                               self.atol, self.dt, verbose=self.verbose)
        op = dslash.operator(U=U_full, clover_term=clover_full,
                             kappa=torch.Tensor([kappa]), support_parity=True,
                             verbose=False)
        S = op.matvec_parity
        # 粗算子构建默认用多线程路径（C++ matvec，加速 ~2x；每线程一个 op）
        if self.nthreads > 1:
            ops_build = [CudaSchurOp(av, g, ce, coo, cei, coi, params=params_t)
                         for _ in range(min(self.nthreads, 4))]
            coarse = build_schur_levels(
                op, S, self.num_levels, self.dof_list, self.mg_grid, self.lat_size,
                self.dof_list[1], self.dt, main_dev, nv_iters=self.nv_iters,
                use_cache=self.use_cache, cache_dir=self.cache_dir, verbose=False,
                matvec_ops=ops_build, nthreads=len(ops_build))
            for o in ops_build:
                o.release()
        else:
            coarse = self._build_coarse_ops(S, self.dof_list[1], main_dev)
        shared = {'g': g, 'fi': fi, 'ce': ce, 'cei': cei, 'coo': coo, 'coi': coi,
                  'coarse': list(zip(*coarse))}  # [(lonv,hnn,hdg,sit)] per coarse level
        # 清理主线程临时 LatticeSet（setup 用），避免槽位与工作线程混淆
        params_t[define._SET_INDEX_] = 0; params_t[define._SET_PLAN_] = -1
        qcu.applyEndQcu(set_ptrs_t, params_t)
        params_t[define._SET_INDEX_] = 1
        qcu.applyEndQcu(set_ptrs_t, params_t)
        params_t[define._SET_INDEX_] = 2
        qcu.applyEndQcu(set_ptrs_t, params_t)
        threads = []
        with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
            for r in ex.map(lambda tid: self._worker(tid, shared), range(self.nthreads)):
                threads.append(r)
        # 主线程参考解（设备 0，单线程，与线程 0 相同输入）
        torch.cuda.set_device(self.device_ids[0])
        results = {'threads': threads, 'lat_size': self.lat_size,
                   'mass': self.mass, 'atol': self.atol,
                   'nthreads': self.nthreads, 'device_ids': self.device_ids,
                   'num_levels': self.num_levels, 'dof_list': self.dof_list}
        self._results = results
        return results

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
    """一行式验证入口（供 pyqcu.testing / examples 使用）。

    kwargs 透传 MultiGpuMultigrid（lat_size/mass/atol/nthreads/...）。
    返回 (results, consistency)。
    """
    mg = MultiGpuMultigrid(**kwargs)
    results = mg.solve()
    consistency = mg.verify_consistency(tol=kwargs.get('tol', 1e-5))
    return results, consistency
