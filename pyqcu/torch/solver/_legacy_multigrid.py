from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from pyqcu import tools
from pyqcu.torch import dslash
from pyqcu.torch.tools._multigrid import load_legacy_mg_assets
import pyqcu.cann as _torch
import mpi4py.MPI as MPI
from time import perf_counter
from pyqcu.cuda import qcu, define
from pyqcu._multigrid_warnings import warn_excessive_multigrid_levels


class _CachedLegacyCoarseOperator:
    """Torch action of a cached 33-tensor Legacy coarse operator."""

    def __init__(self, hop_nn: torch.Tensor, hop_diag: torch.Tensor,
                 sitting: torch.Tensor):
        self.hop_nn = hop_nn
        self.hop_diag = hop_diag
        self.sitting = SimpleNamespace(M=sitting)
        self.hopping = SimpleNamespace(
            M_plus_list=[hop_nn[0, direction] for direction in range(4)],
            M_minus_list=[hop_nn[1, direction] for direction in range(4)],
        )

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        return tools.apply_stencil(
            self.hop_nn, self.hop_diag, self.sitting.M, src)


class LegacyMultigrid:
    """Legacy 33-stencil 自适应多重网格实现。"""

    variant = "legacy"

    def __init__(self, dtype_list: List[torch.dtype], device_list: List[torch.device],  U: torch.Tensor, clover_term: torch.Tensor, kappa: Optional[torch.Tensor] = torch.Tensor([0.1]), u_0: Optional[torch.Tensor] = torch.Tensor([1.0]), clover_ee_inv: Optional[torch.Tensor] = None, clover_oo_inv: Optional[torch.Tensor] = None, min_size: int = 4, max_level: int = 4, mg_grid_size: List[int] = [2, 2, 2, 2], num_convergence_sample: int = 50, dof_list: List[int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], tol: float = 1e-6, max_iter: int = 1000, num_restart: int = 5, root: int = 0, support_parity: bool = False, verbose: bool = True, seed: Optional[int] = None, coarse_assets: Optional[Sequence[Union[str, Path, Mapping[str, Any]]]] = None, coarse_tol: float = 1e-4, coarse_max_iter: int = 200, deflate: bool = False, adaptive_correction: bool = True, correction_calibration: int = 30, correction_window: int = 12):
        """自适应多重网格 V-cycle 求解器。

        seed: 可选确定性种子 —— 给定时初始 b/x0 用独立本地 Generator 生成
        （多线程实例并行时消除全局 RNG 竞争，残差可复现）；None 保持全局 RNG 行为。
        coarse_assets: 每个 coarse transition 一个 HDF5 路径或已加载资产映射。
        传入 CUDA Legacy ``lonv/hnn/hdg/sit`` 缓存时，Torch 直接消费同一
        null basis 与 33-tensor Galerkin 算子，不再生成另一套层级。
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lat_size = list(U.shape[-4:])  # xyzt
        self.min_size = min_size
        self.max_level = max_level
        self.kappa = kappa if kappa is not None else torch.Tensor([0.1])
        self.clover_ee_inv = clover_ee_inv
        self.clover_oo_inv = clover_oo_inv
        self.mass = (1/self.kappa - 8)/2  # just for plot......
        self.u_0 = u_0
        self.tol = tol
        self.max_iter = max_iter
        self.num_restart = num_restart
        self.coarse_tol = float(coarse_tol)
        self.coarse_max_iter = int(coarse_max_iter)
        self.deflate = bool(deflate)
        self.adaptive_correction = bool(adaptive_correction)
        self.correction_calibration = max(0, int(correction_calibration))
        self.correction_window = max(1, int(correction_window))
        self.coarse_assets: List[Union[str, Path, Mapping[str, Any]]] = (
            list(coarse_assets) if coarse_assets is not None else [])
        self.root = root
        self.with_cuda_qcu = True if self.clover_ee_inv is not None and self.clover_oo_inv is not None else False
        self.support_parity = support_parity
        self.verbose = verbose
        # Build grid list
        self.lat_size_list = []
        self.mg_grid_size = mg_grid_size
        _lat_size = self.lat_size
        while all(_ >= self.min_size for _ in _lat_size) and len(self.lat_size_list) < self.max_level:
            self.lat_size_list.append(_lat_size)
            _lat_size = [_lat_size[d] // self.mg_grid_size[d]
                         for d in range(4)]
        self.num_level = len(self.lat_size_list)
        self.multigrid_depth_warning = warn_excessive_multigrid_levels(
            self.num_level, variant="legacy", backend="torch",
            context="Torch LegacyMultigrid", lattice=self.lat_size,
            block=self.mg_grid_size)
        self.dof_list = dof_list[:self.num_level]
        self.dtype_list = dtype_list[:self.num_level]
        self.device_list = device_list[:self.num_level]
        if self.rank == self.root:
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n self.dof_list:{self.dof_list}")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n self.dtype_list:{self.dtype_list}")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n self.device_list:{self.device_list}")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n self.lat_size_list:{self.lat_size_list}")
        for device in self.device_list:
            tools.set_device(device=device)
        self.op_list = [dslash.operator(U=U,
                                        clover_term=clover_term, verbose=self.verbose, kappa=self.kappa, u_0=self.u_0, support_parity=self.support_parity, clover_ee_inv=self.clover_ee_inv, clover_oo_inv=self.clover_oo_inv,)]
        if seed is not None:
            # bug40 后续: 多线程实例并行时全局 RNG 竞争使初始 b/x0 非确定;
            # 显式 seed 时用本地 Generator(每实例独立流),残差可复现。
            # 实数 randn 无需 cann 复数分解; cann.randn 不透传 generator 故直用 torch
            g_b = torch.Generator(device=self.device_list[0]).manual_seed(seed)
            self.b = torch.randn(size=[12]+self.lat_size,
                                 dtype=self.dtype_list[0],
                                 device=self.device_list[0],
                                 generator=g_b)
            g_x = torch.Generator(device=self.device_list[0]).manual_seed(seed + 1_000_000)
            self.x0 = torch.randn(
                size=[12]+self.lat_size, dtype=self.dtype_list[0],
                device=self.device_list[0], generator=g_x)
        else:
            self.b = _torch.randn(size=[12]+self.lat_size,
                                  dtype=self.dtype_list[0], device=self.device_list[0])
            self.x0 = _torch.randn(
                size=[12]+self.lat_size, dtype=self.dtype_list[0], device=self.device_list[0])
        self.b_list = [self.b.clone()]
        self.nv_list = []  # null_vecs_list
        self.lonv_list = []  # local_ortho_null_vecs_list
        self.num_convergence_sample = num_convergence_sample
        self.convergence_history = []
        self.convergence_tol = 0
        # Pre-allocated CUDA tensors for restrict/prolong (level 0 -> 1)
        self._mg_restrict_out = None
        self._mg_prolong_out = None
        self._mg_coarse_dslash_out = None
        self._mg_coarse_hopping_packed = None
        self._mg_coarse_hopping_packed_level = None  # track which level is cached
        self._max_set_index = -1  # init 创建的 C++ LatticeSet 最大槽位（end 释放用）
        self._setup_done = False
        self.asset_identities: List[Dict[str, Any]] = []
        self.last_solve_report: Dict[str, Any] = {}

    def _restrict_cuda(self, fine_vec: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated restriction using C++ backend."""
        lonv = self.lonv_list[level]
        E = lonv.shape[0]
        e = fine_vec.shape[0]
        Xf, Yf, Zf, Tf = fine_vec.shape[1:]
        Xc, Yc, Zc, Tc = self.lat_size_list[level+1]
        # Set multigrid params
        self.params[define._LAT_X_] = Xf
        self.params[define._LAT_Y_] = Yf
        self.params[define._LAT_Z_] = Zf
        self.params[define._LAT_T_] = Tf
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_LEVEL1_E_] = E
        self.params[define._MG_NUM_LEVEL_] = e  # fine DOF
        # Allocate or reuse output tensor
        if self._mg_restrict_out is None or self._mg_restrict_out.shape != torch.Size([E, Xc, Yc, Zc, Tc]):
            self._mg_restrict_out = torch.zeros(
                [E, Xc, Yc, Zc, Tc], dtype=fine_vec.dtype, device=fine_vec.device)
        qcu.applyMultigridRestrictQcu(
            self._mg_restrict_out, fine_vec, lonv, self.set_ptrs, self.params)
        torch.cuda.synchronize()  # C++ 私有流完成后再返回（固定缓冲复用）
        return self._mg_restrict_out

    def _prolong_cuda(self, coarse_vec: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated prolongation using C++ backend."""
        lonv = self.lonv_list[level]
        E = coarse_vec.shape[0]
        e = lonv.shape[1]
        Xc, Yc, Zc, Tc = coarse_vec.shape[1:]
        Xf = Xc * self.mg_grid_size[0]
        Yf = Yc * self.mg_grid_size[1]
        Zf = Zc * self.mg_grid_size[2]
        Tf = Tc * self.mg_grid_size[3]
        # Set multigrid params
        self.params[define._LAT_X_] = Xf
        self.params[define._LAT_Y_] = Yf
        self.params[define._LAT_Z_] = Zf
        self.params[define._LAT_T_] = Tf
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_LEVEL1_E_] = E
        self.params[define._MG_NUM_LEVEL_] = e  # fine DOF
        # Allocate or reuse output tensor
        if self._mg_prolong_out is None or self._mg_prolong_out.shape != torch.Size([e, Xf, Yf, Zf, Tf]):
            self._mg_prolong_out = torch.zeros(
                [e, Xf, Yf, Zf, Tf], dtype=coarse_vec.dtype, device=coarse_vec.device)
        qcu.applyMultigridProLongQcu(
            self._mg_prolong_out, coarse_vec, lonv, self.set_ptrs, self.params)
        torch.cuda.synchronize()
        return self._mg_prolong_out

    def _coarse_dslash_cuda(self, src: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated coarse-grid dslash using C++ backend.

        Applies the coarse-grid Dirac operator D_coarse = sitting + hopping_plus + hopping_minus.
        Data layout:
          - fermion:  [E, Xc, Yc, Zc, Tc]
          - hopping:  [2, 4, E, E, Xc, Yc, Zc, Tc] (pm × dir × Eout × Ein × XYZT)
          - sitting:  [E, E, Xc, Yc, Zc, Tc]
        """
        op = self.op_list[level]
        E, Xc, Yc, Zc, Tc = src.shape
        # NOTE: SET_INDEX_ is reset to 0 for coarse dslash. This is safe because
        # the coarse dslash operates at a different multigrid level and does not
        # overlap with fine-level BiCGStab iterations that increment SET_INDEX_.
        # If coarse dslash were ever called concurrently with fine-level ops,
        # a dedicated SET_INDEX_ per level would be required.
        self.params[define._SET_INDEX_] = 0
        # Pack hopping matrices: [2, 4, E, E, Xc, Yc, Zc, Tc]
        # dim0: 0=plus, 1=minus; dim1: 0=X, 1=Y, 2=Z, 3=T
        target_shape = torch.Size([2, 4, E, E, Xc, Yc, Zc, Tc])
        if (self._mg_coarse_hopping_packed is None or
                self._mg_coarse_hopping_packed.shape != target_shape or
                self._mg_coarse_hopping_packed.device != src.device or
                self._mg_coarse_hopping_packed_level != level):
            hopping_packed = torch.zeros(
                target_shape, dtype=src.dtype, device=src.device)
            for ward in range(4):
                hopping_packed[0, ward] = op.hopping.M_plus_list[ward].to(
                    dtype=src.dtype, device=src.device)
                hopping_packed[1, ward] = op.hopping.M_minus_list[ward].to(
                    dtype=src.dtype, device=src.device)
            self._mg_coarse_hopping_packed = hopping_packed
            self._mg_coarse_hopping_packed_level = level
        # Set multigrid params for the C++ backend
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_NUM_LEVEL_] = E  # coarse DOF (repurposed)
        # Ensure sitting tensor is on the correct device and dtype
        sitting = op.sitting.M.to(dtype=src.dtype, device=src.device)
        # Allocate or reuse output tensor
        if (self._mg_coarse_dslash_out is None or
                self._mg_coarse_dslash_out.shape != src.shape or
                self._mg_coarse_dslash_out.device != src.device):
            self._mg_coarse_dslash_out = torch.zeros_like(src)
        qcu.applyMultigridCoarseDslashQcu(
            self._mg_coarse_dslash_out, src,
            self._mg_coarse_hopping_packed, sitting,
            self.set_ptrs, self.params)
        torch.cuda.synchronize()
        return self._mg_coarse_dslash_out

    def _verify_coarse_dslash(self, level: int = 1, tol: float = 1e-3) -> bool:
        """Debug helper: verify CUDA coarse dslash against Python einsum version.

        Generates random input, compares CUDA output with Python reference.
        Returns True if max absolute difference < tol.
        """
        op = self.op_list[level]
        Xc, Yc, Zc, Tc = self.lat_size_list[level]
        E = self.dof_list[level]
        dtype = self.dtype_list[level]
        device = self.device_list[level]
        src = torch.randn([E, Xc, Yc, Zc, Tc], dtype=dtype, device=device)
        # Python reference (einsum-based)
        ref = op.matvec(src)
        # CUDA path
        cu = self._coarse_dslash_cuda(src, level=level)
        diff = (ref - cu).abs().max().item()
        if self.rank == self.root:
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID::VERIFY:\n "
                  f"level={level}, max|ref-cuda| = {diff:.6e}")
        return diff < tol

    def init(self):
        if self._setup_done:
            return self
        # Build local-orthonormal near-null space vectors
        comm = MPI.COMM_WORLD
        comm.Barrier()
        if self.with_cuda_qcu:
            from pyqcu.cuda.define import params, argv, set_ptrs
            params[define._LAT_X_] = self.lat_size[define._LAT_X_]
            params[define._LAT_Y_] = self.lat_size[define._LAT_Y_]
            params[define._LAT_Z_] = self.lat_size[define._LAT_Z_]
            params[define._LAT_T_] = self.lat_size[define._LAT_T_]
            params[define._LAT_XYZT_] = params[define._LAT_X_] * \
                params[define._LAT_Y_]*params[define._LAT_Z_] * \
                params[define._LAT_T_]
            params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
                define._GRID_T_] = tools.give_grid_size()
            params[define._PARITY_] = 0
            params[define._NODE_RANK_] = self.comm.Get_rank()
            params[define._NODE_SIZE_] = self.comm.Get_size()
            params[define._DAGGER_] = 0
            params[define._MAX_ITER_] = self.max_iter
            params[define._DATA_TYPE_] = define.epytd(
                torch_dtype=self.dtype_list[0])
            params[define._SET_INDEX_] = 0
            params[define._SET_PLAN_] = 1
            params[define._VERBOSE_] = 0
            params[define._SEED_] = 42
            argv = argv.to(dtype=define.dtype(
                params[define._DATA_TYPE_]).to_real())
            argv[define._MASS_] = self.mass
            argv[define._ATOL_] = self.tol
            argv[define._SIGMA_] = 0.1
            self.set_ptrs = set_ptrs.clone()
            self.params = params.clone()
            self.argv = argv.clone()
            qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            self._max_set_index = max(
                self._max_set_index, int(self.params[define._SET_INDEX_]))
            self.gauge_eo = tools.oooxyzt2poooxyzt(
                input_array=self.op_list[0].hopping.U)  # type: ignore
            self.clover_ee = self.op_list[0].sitting.M_e
            self.clover_oo = self.op_list[0].sitting.M_o
            self.fermion_in_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
                dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
            self.fermion_out_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
                dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
            self.dest_o = self.fermion_out_eo[1]

        transitions = len(self.lat_size_list) - 1
        if self.coarse_assets and len(self.coarse_assets) < transitions:
            raise ValueError(
                f"coarse_assets provides {len(self.coarse_assets)} transitions, "
                f"but the hierarchy requires {transitions}")
        self._use_cached_assets = bool(self.coarse_assets)

        for i in range(1, len(self.lat_size_list)):
            if self.coarse_assets:
                asset = load_legacy_mg_assets(
                    self.coarse_assets[i - 1],
                    dtype=self.dtype_list[i], device=self.device_list[i])
                expected_shape = tuple(self.lat_size_list[i - 1][axis] // self.mg_grid_size[axis]
                                       for axis in range(4))
                expected_shape = (*expected_shape[:3], expected_shape[3] // 2)
                if tuple(asset["coarse_shape"]) != expected_shape:
                    raise ValueError(
                        f"Legacy cache level {i} coarse shape "
                        f"{asset['coarse_shape']} != expected {expected_shape}")
                if int(asset["lonv"].shape[1]) != int(self.dof_list[i - 1]):
                    raise ValueError(
                        f"Legacy cache level {i} input dof "
                        f"{asset['lonv'].shape[1]} != {self.dof_list[i - 1]}")
                _null_vecs = None
                _local_ortho_null_vecs = asset["lonv"]
                self.nv_list.append(None)
                self.lonv_list.append(_local_ortho_null_vecs)
                self.b_list.append(torch.zeros(
                    size=[asset["lonv"].shape[0], *expected_shape],
                    dtype=self.dtype_list[i], device=self.device_list[i]))
                self.op_list.append(_CachedLegacyCoarseOperator(
                    asset["hnn"], asset["hdg"], asset["sit"]))
                self.asset_identities.append({
                    key: asset[key]
                    for key in ("path", "file_sha256", "dataset_sha256",
                                "coarse_shape")
                })
                continue

            _null_vecs = _torch.randn(size=[self.dof_list[i], self.dof_list[i-1]] +
                                      self.lat_size_list[i-1], dtype=self.dtype_list[i-1], device=self.device_list[i-1])
            if self.with_cuda_qcu and i == 1:
                def _bistabcg(b: torch.Tensor, tol: float = 1e-6, max_iter: int = 1000, x0:  Optional[torch.Tensor] = None, if_rtol: bool = False, verbose: bool = True) -> torch.Tensor:
                    if if_rtol:
                        _tol = tools.norm(b)*tol
                    else:
                        _tol = tol
                    self.fermion_in_eo = tools.oooxyzt2poooxyzt(input_array=b)
                    self.params[define._MAX_ITER_] = max_iter
                    self.params[define._SET_INDEX_] += 1
                    self.argv[define._ATOL_] = _tol
                    qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
                    self._max_set_index = max(
                        self._max_set_index, int(self.params[define._SET_INDEX_]))
                    qcu.applyCloverBistabCgQcu(self.fermion_out_eo, self.fermion_in_eo, self.gauge_eo, self.clover_ee,  # type: ignore
                                               self.clover_oo, self.clover_ee_inv, self.clover_oo_inv,  self.set_ptrs, self.params)  # type: ignore
                    return tools.poooxyzt2oooxyzt(input_array=self.fermion_out_eo).reshape(b.shape)
                bistabcg = _bistabcg
            else:
                bistabcg = None
            _null_vecs = tools.give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                bistabcg=bistabcg,
                verbose=self.verbose).to(dtype=self.dtype_list[i], device=self.device_list[i])
            self.nv_list.append(_null_vecs)
            _local_ortho_null_vecs = tools.local_orthogonalize(
                null_vecs=_null_vecs,
                coarse_lat_size=self.lat_size_list[i],
                verbose=self.verbose)
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.zeros(
                size=[self.dof_list[i]]+self.lat_size_list[i], dtype=self.dtype_list[i], device=self.device_list[i]))
            self.op_list.append(dslash.operator(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs,  verbose=self.verbose))
        if not self.coarse_assets:
            self._use_cached_assets = False
        comm.Barrier()
        self._setup_done = True
        return self

    def end(self):
        """释放 C++ LatticeSet 资源（防多实例/重复调用显存泄漏 OOM）。

        init 创建的 LatticeSet（槽位 0.._max_set_index，含 _bistabcg 的
        逆迭代实例）全部 applyEndQcu 释放；调用后实例不可再 solve。
        """
        if self.with_cuda_qcu and self._max_set_index >= 0:
            for _idx in range(self._max_set_index + 1):
                self.params[define._SET_INDEX_] = _idx
                try:
                    qcu.applyEndQcu(self.set_ptrs, self.params)
                except Exception:
                    pass
            self._max_set_index = -1
        self._setup_done = False

    def setup(self):
        """Build the hierarchy; canonical lifecycle name for all backends."""
        return self.init()

    def close(self) -> None:
        """Release backend resources; safe to call more than once."""
        self.end()

    def __enter__(self) -> "LegacyMultigrid":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        # 隐式 GC 释放时机不可控：先同步设备再释放，减少与并发 C++ 操作的干扰。
        # 多实例循环场景请显式 end() + del + gc.collect()（见 AGENTS.md）。
        try:
            torch.cuda.synchronize()
            self.end()
        except Exception:
            pass

    def levels_back(self):
        self.convergence_tol = 0
        self.num_level = len(self.op_list) if len(
            self.op_list) <= self.max_level else self.max_level

    def adaptive(self, iter: int = 0):
        if self.convergence_tol > 3:
            self.num_level = 1
        if (iter+1)*2 < self.num_convergence_sample+1:
            return
        convergence_now = self.convergence_history[-1]
        convergence_sample = self.convergence_history[-(
            self.num_convergence_sample+1):-1]
        count = 0
        for convergence in convergence_sample:
            if convergence < convergence_now:
                count += 1
        if count >= self.num_convergence_sample//2:
            self.convergence_tol += 1

    def cycle(self, level: int = 0) -> torch.Tensor:
        """Run the CUDA-aligned Legacy BiCGStab/V-cycle control path."""
        matvec = self.op_list[level].matvec
        if self.with_cuda_qcu and level == 1 and not self._use_cached_assets:
            def _coarse_matvec(src: torch.Tensor) -> torch.Tensor:
                return self._coarse_dslash_cuda(src=src, level=level)
            matvec = _coarse_matvec

        b = self.b_list[level].clone()
        b_origin = None
        b_e = None
        if level == 0 and self.support_parity:
            b_origin = b.clone()
            b_eo = tools.oooxyzt2poooxyzt(b)
            b_e = b_eo[0]
            b = self.op_list[0].give_b_parity(b_e=b_e, b_o=b_eo[1])
            if not self.with_cuda_qcu:
                matvec = self.op_list[0].matvec_parity
            else:
                def _matvec(src_o: torch.Tensor) -> torch.Tensor:
                    self.params[define._SET_INDEX_] = 0
                    qcu.applyCloverBistabCgDslashQcu(
                        self.dest_o, src_o, self.gauge_eo,
                        self.clover_ee, self.clover_oo,
                        self.clover_ee_inv, self.clover_oo_inv,
                        self.set_ptrs, self.params)  # type: ignore
                    torch.cuda.synchronize()
                    return self.dest_o.reshape(src_o.shape)
                matvec = _matvec

        if level == 0:
            self._cycle_stats = {
                "iterations": 0,
                "vcycles": 0,
                "corrections": [],
                "deflation": None,
                "iteration_samples": [],
                "stop_reason": None,
            }
            self._coarse_r0_ref = {}

        x = torch.zeros_like(b)
        r = b - matvec(x)
        b_norm = tools.norm(b)
        r_norm = tools.norm(r)
        if level == 0:
            # Match CUDA Legacy's relative Schur-residual stopping rule.
            tolerance = self.tol * max(float(b_norm), 1e-30)
            iteration_limit = self.max_iter
            self.convergence_history.append(r_norm)
        else:
            # CUDA anchors a level's target to its first incoming RHS norm.
            self._coarse_r0_ref.setdefault(level, float(r_norm))
            tolerance = self.coarse_tol * max(
                float(self._coarse_r0_ref[level]), 1e-30)
            iteration_limit = self.coarse_max_iter
            if r_norm < 1e-4:
                return x

        if self.verbose:
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of b:{b_norm}")
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of r:{r_norm}")
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of x0:{tools.norm(x)}")
        if r_norm < tolerance:
            if level == 0:
                self._cycle_stats["stop_reason"] = "initial_residual"
            return x

        def coarse_correct(current_x: torch.Tensor, current_r: torch.Tensor,
                           reason: str) -> tuple[torch.Tensor, torch.Tensor]:
            pre_correction = float(tools.norm(current_r))
            self._cycle_stats["vcycles"] += 1
            if self.with_cuda_qcu and level == 0:
                restricted = self._restrict_cuda(
                    fine_vec=current_r, level=level)
            else:
                restricted = tools.restrict(
                    local_ortho_null_vecs=self.lonv_list[level],
                    fine_vec=current_r)
            self.b_list[level + 1] = restricted.to(
                dtype=self.dtype_list[level + 1],
                device=self.device_list[level + 1])
            coarse_error = self.cycle(level=level + 1).to(
                dtype=self.dtype_list[level],
                device=self.device_list[level])
            if self.with_cuda_qcu and level == 0:
                fine_error = self._prolong_cuda(
                    coarse_vec=coarse_error, level=level)
            else:
                fine_error = tools.prolong(
                    local_ortho_null_vecs=self.lonv_list[level],
                    coarse_vec=coarse_error)
            if (level == 0 and self.support_parity and
                    not self._use_cached_assets):
                fine_error = tools.oooxyzt2poooxyzt(fine_error)[1]
            updated_x = current_x + fine_error
            updated_r = b - matvec(updated_x)
            post_correction = float(tools.norm(updated_r))
            record = {
                "reason": reason,
                "pre_residual": pre_correction,
                "post_residual": post_correction,
                "accepted": post_correction < pre_correction,
                "applied": False,
                "acceptance_policy": "post_residual_must_improve",
            }
            self._cycle_stats["corrections"].append(record)
            return updated_x, updated_r

        def full_residual_relative() -> tuple[float, float]:
            x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
            x_origin = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([x_e, x], dim=0))
            residual = b_origin - self.op_list[0].matvec(src=x_origin)
            absolute = float(tools.norm(residual))
            relative = absolute / max(float(tools.norm(b_origin)), 1e-30)
            return absolute, relative

        corrections_enabled = (
            level < self.num_level - 1 and self.num_restart > 0)

        if (level == 0 and self.deflate and corrections_enabled and
                r_norm > tolerance):
            previous_x, previous_r = x, r
            candidate_x, candidate_r = coarse_correct(x, r, "deflation")
            record = self._cycle_stats["corrections"][-1]
            if record["accepted"]:
                x, r = candidate_x, candidate_r
                record["applied"] = True
            else:
                corrections_enabled = False
                x, r = previous_x, previous_r
            r_norm = tools.norm(r)
            self._cycle_stats["deflation"] = dict(
                self._cycle_stats["corrections"][-1])

        r_tilde = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        s = torch.zeros_like(b)
        t = torch.zeros_like(b)
        rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        start_time = perf_counter()
        iter_times: List[float] = []
        since_correction = 0
        calibration_done = not (level == 0 and self.adaptive_correction)
        check_stride = 4
        stopped_by_tolerance = False
        full_gate_checks = 0
        full_gate_passed = False
        near_tolerance = False
        breakdown_restarts = 0
        if self.verbose:
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Starting Iterations")

        for i in range(iteration_limit):
            iter_start_time = perf_counter()
            try:
                rho = tools.vdot(r_tilde, r)
                if abs(rho) < 1e-20:
                    raise RuntimeError("rho")
                beta = (rho / rho_prev) * (alpha / omega)
                rho_prev = rho
                p = r + beta * (p - omega * v)
                v = matvec(p)
                rtv = tools.vdot(r_tilde, v)
                if abs(rtv) < 1e-20:
                    raise RuntimeError("rtv")
                alpha = rho / rtv
                if not torch.isfinite(alpha):
                    raise RuntimeError("alpha non-finite")
                s = r - alpha * v
                t = matvec(s)
                tts = tools.vdot(t, t)
                if abs(tts) < 1e-20:
                    raise RuntimeError("tts")
                omega = tools.vdot(t, s) / tts
                if not torch.isfinite(omega):
                    raise RuntimeError("omega non-finite")
            except RuntimeError:
                breakdown_restarts += 1
                if self.verbose:
                    print(
                        f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:"
                        f"BiCGStab breakdown at iter {i}, restarting...")
                r_tilde = r.clone()
                p = torch.zeros_like(b)
                v = torch.zeros_like(b)
                s = torch.zeros_like(b)
                t = torch.zeros_like(b)
                rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                continue

            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = tools.norm(r)
            since_correction += 1
            total = i + 1
            if level == 0:
                self.convergence_history.append(r_norm)
                self._cycle_stats["iterations"] = total
                self._cycle_stats["iteration_samples"].append({
                    "iteration": total,
                    "residual": float(r_norm),
                    "relative_residual": float(
                        r_norm / max(float(b_norm), 1e-30)),
                    "since_correction": since_correction,
                })
            if self.verbose:
                print(
                    f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n "
                    f"B-{level}-bistabcg-Iteration {i}: Residual = {r_norm:.6e}")

            # CUDA's single-rank path performs a reliable update every 48
            # Schur iterations.  It is part of the convergence control flow,
            # not just diagnostics: the true residual replaces the recurrence
            # residual and all BiCGStab scalars are reset.
            if (level == 0 and total % 48 == 0 and
                    self.comm.Get_size() == 1):
                r = b - matvec(x)
                r_norm = tools.norm(r)
                r_tilde = r.clone()
                p = torch.zeros_like(b)
                v = torch.zeros_like(b)
                s = torch.zeros_like(b)
                t = torch.zeros_like(b)
                rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                since_correction = 0
                self.convergence_history[-1] = r_norm
                sample = self._cycle_stats["iteration_samples"][-1]
                sample["residual"] = float(r_norm)
                sample["relative_residual"] = float(
                    r_norm / max(float(b_norm), 1e-30))
                sample["reliable_refresh"] = True

            near_tolerance = (
                near_tolerance or r_norm <= 4.0 * tolerance)
            do_check = near_tolerance or total % check_stride == 0
            if level == 0 and corrections_enabled and not calibration_done:
                if total >= self.correction_calibration:
                    calibration_done = True

            if do_check and r_norm <= tolerance:
                if level == 0 and self.support_parity:
                    full_absolute, full_relative = full_residual_relative()
                    full_gate_checks += 1
                    self._cycle_stats["last_full_gate"] = {
                        "iteration": total,
                        "absolute_residual": full_absolute,
                        "relative_residual": full_relative,
                    }
                    if full_relative <= self.tol:
                        full_gate_passed = True
                        stopped_by_tolerance = True
                        self._cycle_stats["stop_reason"] = "full_residual"
                        break
                else:
                    stopped_by_tolerance = True
                    self._cycle_stats["stop_reason"] = "native_tolerance"
                    break

            correction_due = (
                corrections_enabled and do_check and
                since_correction >= self.num_restart and
                r_norm > 100.0 * tolerance)
            if level == 0 and self.adaptive_correction:
                correction_due = correction_due and calibration_done
            if correction_due:
                previous_x, previous_r = x, r
                candidate_x, candidate_r = coarse_correct(
                    x, r, "periodic")
                record = self._cycle_stats["corrections"][-1]
                if record["accepted"]:
                    x, r = candidate_x, candidate_r
                    record["applied"] = True
                    r_tilde = r.clone()
                    p = torch.zeros_like(b)
                    v = torch.zeros_like(b)
                    s = torch.zeros_like(b)
                    t = torch.zeros_like(b)
                    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                else:
                    corrections_enabled = False
                    x, r = previous_x, previous_r
                r_norm = tools.norm(r)
                since_correction = 0
                if level == 0:
                    self.convergence_history.append(r_norm)
                    self._cycle_stats["iteration_samples"][-1][
                        "post_correction_residual"] = float(r_norm)

            iter_times.append(perf_counter() - iter_start_time)
            if self.verbose:
                print(
                    f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n "
                    f"F-{level}-bistabcg-Iteration {i}: Residual = "
                    f"{r_norm:.6e}, Time = {iter_times[-1]:.6f} s")

        if level == 0:
            if self._cycle_stats["stop_reason"] is None:
                self._cycle_stats["stop_reason"] = (
                    "full_residual" if stopped_by_tolerance else
                    "iteration_limit")
            corrections = self._cycle_stats["corrections"]
            self._cycle_stats["correction_gate_disabled"] = not corrections_enabled
            self._cycle_stats["calibration_done"] = calibration_done
            self._cycle_stats["full_gate_checks"] = full_gate_checks
            self._cycle_stats["full_gate_passed"] = full_gate_passed
            self._cycle_stats["corrections_accepted"] = sum(
                bool(item.get("accepted")) for item in corrections)
            self._cycle_stats["corrections_rejected"] = sum(
                not bool(item.get("accepted")) for item in corrections)
            self._cycle_stats["breakdown_restarts"] = breakdown_restarts
            self._cycle_stats["native_residual"] = float(r_norm)
            self._cycle_stats["native_relative_residual"] = float(
                r_norm / max(float(b_norm), 1e-30))

        total_time = perf_counter() - start_time
        if self.verbose:
            avg_iter_time = sum(iter_times) / len(iter_times) if iter_times else 0.0
            print("PYQCU::SOLVER::LEGACY_MULTIGRID:\n Performance Statistics:")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Total iterations: {len(iter_times)}")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Total time: {total_time:.6f} seconds")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Average time per iteration: {avg_iter_time:.6f} s")
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Final residual: {r_norm:.2e}")

        if level == 0 and self.support_parity:
            x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
            x_origin = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([x_e, x], dim=0))
            r_origin = b_origin - self.op_list[0].matvec(src=x_origin)
            self._cycle_stats["full_residual_norm"] = float(tools.norm(r_origin))
            self._cycle_stats["full_relative_residual"] = float(
                tools.norm(r_origin) /
                max(float(tools.norm(b_origin)), 1e-30))
            if self.verbose:
                print(
                    f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n "
                    f"Final residual(origin): {tools.norm(r_origin):.2e}")
            x = x_origin.clone()
        return x

    def _cycle_absolute_tolerance_reference(self, level: int = 0) -> torch.Tensor:
        """Regression oracle for the pre-parity absolute-tolerance control path."""
        matvec = self.op_list[level].matvec
        if self.with_cuda_qcu and level == 1:
            def _coarse_matvec(src: torch.Tensor) -> torch.Tensor:
                return self._coarse_dslash_cuda(src=src, level=level)
            matvec = _coarse_matvec
        b = self.b_list[level].clone()
        if level == 0 and self.support_parity:
            b_origin = b.clone()
            b_eo = tools.oooxyzt2poooxyzt(b)
            b_e = b_eo[0]
            b_o = b_eo[1]
            b = self.op_list[0].give_b_parity(b_e=b_e, b_o=b_o)
            if self.with_cuda_qcu is False:
                matvec = self.op_list[0].matvec_parity
            else:
                def _matvec(src_o: torch.Tensor) -> torch.Tensor:
                    self.params[define._SET_INDEX_] = 0
                    qcu.applyCloverBistabCgDslashQcu(self.dest_o, src_o, self.gauge_eo,
                                                     self.clover_ee, self.clover_oo, self.clover_ee_inv, self.clover_oo_inv,  self.set_ptrs, self.params)  # type: ignore
                    torch.cuda.synchronize()  # C++ 私有流完成后再读固定缓冲
                    return self.dest_o.reshape(src_o.shape)
                matvec = _matvec
        x = torch.zeros_like(b)
        r = b - matvec(x)
        r_norm = tools.norm(r)
        _tol = r_norm*0.5 if level != self.num_level - 1 else r_norm*0.1
        if self.verbose:
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of b:{tools.norm(b)}")
            print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of r:{r_norm}")
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Norm of x0:{tools.norm(x)}")
        if level == 0:
            self.convergence_history.append(r_norm)
            _tol = self.tol
        if r_norm < _tol:
            print("PYQCU::SOLVER::LEGACY_MULTIGRID:\n x0 is just right!")
            return x
        r_tilde = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        s = torch.zeros_like(b)
        t = torch.zeros_like(b)
        rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        start_time = perf_counter()
        iter_times = []
        count_restart = 0
        print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:Starting Iterations")
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            # BUGFIX 2026-07-28 R2: BiCGStab breakdown detection (same as _bistabcg.py).
            # 2026-08-14: breakdown 改为自动重启（保留当前 x/r，重置影子向量与系数）
            # 而非抛错 —— BiCGStab 对某些右端项在迭代中会合法 breakdown，
            # 重启后通常继续收敛；迭代预算不变。
            try:
                rho = tools.vdot(r_tilde, r)
                if abs(rho) < 1e-20:
                    raise RuntimeError("rho")
                beta = (rho / rho_prev) * (alpha / omega)
                rho_prev = rho
                p = r + beta * (p - omega * v)
                v = matvec(p)
                rtv = tools.vdot(r_tilde, v)
                if abs(rtv) < 1e-20:
                    raise RuntimeError("rtv")
                alpha = rho / rtv
                if not torch.isfinite(alpha):
                    raise RuntimeError("alpha non-finite")
                s = r - alpha * v
                t = matvec(s)
                tts = tools.vdot(t, t)
                if abs(tts) < 1e-20:
                    raise RuntimeError("tts")
                omega = tools.vdot(t, s) / tts
                if not torch.isfinite(omega):
                    raise RuntimeError("omega non-finite")
            except RuntimeError:
                if self.verbose:
                    print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n {level}:BiCGStab breakdown "
                          f"at iter {i}, restarting...")
                r_tilde = r.clone()
                p = torch.zeros_like(b)
                v = torch.zeros_like(b)
                s = torch.zeros_like(b)
                t = torch.zeros_like(b)
                rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                r_norm = tools.norm(r)
                if level == 0:
                    self.convergence_history.append(r_norm)
                continue
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = tools.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n B-{level}-bistabcg-Iteration {i}: Residual = {r_norm:.6e}")
            count_restart += 1
            # cycle start
            if level < self.num_level-1 and count_restart > self.num_restart:
                if level == 0 and self.support_parity:
                    x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
                    x_origin = tools.poooxyzt2oooxyzt(
                        input_array=torch.stack([x_e, x], dim=0))
                    r = b_origin-self.op_list[0].matvec(src=x_origin)
                if self.with_cuda_qcu and level == 0:
                    r_coarse = self._restrict_cuda(fine_vec=r, level=level)
                else:
                    r_coarse = tools.restrict(
                        local_ortho_null_vecs=self.lonv_list[level], fine_vec=r)
                self.b_list[level+1] = r_coarse.to(dtype=self.dtype_list[level+1],
                                                   device=self.device_list[level+1])
                e_coarse = self.cycle(level=level+1).to(dtype=self.dtype_list[level],
                                                        device=self.device_list[level])
                if self.with_cuda_qcu and level == 0:
                    e_fine = self._prolong_cuda(coarse_vec=e_coarse, level=level)
                else:
                    e_fine = tools.prolong(
                        local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse)
                if level == 0 and self.support_parity:
                    e_fine_eo = tools.oooxyzt2poooxyzt(e_fine)
                    e_fine = e_fine_eo[1]
                x = x + e_fine
                r = b - matvec(x)
                count_restart = 0
                # BUGFIX 2026-07-28 R3: BiCGStab state must be fully reset after
                # a coarse-grid correction. The residual r has changed, so the
                # shadow vector r_tilde, the search directions p/v/s/t, and the
                # scalar coefficients rho_prev/alpha/omega must all be reinitialized.
                # Without this, rho = vdot(r_tilde_old, r_new) gives meaningless
                # results, causing convergence degradation or divergence.
                r_tilde = r.clone()
                p = torch.zeros_like(b)
                v = torch.zeros_like(b)
                s = torch.zeros_like(b)
                t = torch.zeros_like(b)
                rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
                omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
            r_norm = tools.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
                self.adaptive(iter=i)
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n F-{level}-bistabcg-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
            if r_norm < _tol:
                if self.verbose:
                    print(
                        f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Converged at iteration {i} with residual {r_norm:.6e}")
                break
        else:
            print(
                "PYQCU::SOLVER::LEGACY_MULTIGRID:\n Warning: Maximum iterations reached, may not have converged")
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Performance Statistics:")
        print(
            f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Total iterations: {len(iter_times)}")
        print(
            f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Total time: {total_time:.6f} seconds")
        print(
            f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Average time per iteration: {avg_iter_time:.6f} s")
        print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Final residual: {r_norm:.2e}")
        if level == 0 and self.support_parity:
            x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
            x_origin = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([x_e, x], dim=0))
            r_origin = b_origin-self.op_list[0].matvec(src=x_origin)
            r_norm_origin = tools.norm(r_origin)
            print(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Final residual(origin): {r_norm_origin:.2e}")
            x = x_origin.clone()
        return x

    def solve(self, b: Optional[torch.Tensor] = None, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if b is not None:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.b_list[0] = self.b.clone()
        if x0 is not None:
            self.x0 = x0.reshape([12]+list(x0.shape)[2:]).clone()  # sc->e
        self.levels_back()
        start_time = perf_counter()
        x = self.cycle()
        total_time = perf_counter() - start_time
        self.last_solve_report = {
            **self._cycle_stats,
            "backend": "torch",
            "variant": "legacy",
            "device": str(self.device_list[0]),
            "outer_solver": "BiCGStab",
            "residual_scope": "schur",
            "stop_criterion": "relative_schur_residual",
            "tolerance": float(self.tol),
            "restart": int(self.num_restart),
            "coarse_max_iter": int(self.coarse_max_iter),
            "coarse_tolerance": float(self.coarse_tol),
            "initial_guess": "zero",
            "trace": False,
            "asset_identities": list(self.asset_identities),
            "solve_seconds": total_time,
        }
        print(f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Performance Statistics:")
        print(
            f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Total time: {total_time:.6f} seconds")
        print(
            f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n Final residual: {self.convergence_history[-1]:.2e}")
        return x.reshape([4, 3]+list(x.shape[-4:]))

    def plot(self, save_path=None):
        if self.rank == self.root:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(10, 6))
            plt.title(
                f"PYQCU::SOLVER::LEGACY_MULTIGRID:\n convergence_history(mass={self.mass})\n self.dof_list:{self.dof_list}\n self.lat_size_list:{self.lat_size_list}\n self.dtype_list:{self.dtype_list}\n self.device_list:{self.device_list}", fontsize=12)
            plt.semilogy(range(1, len(self.convergence_history) + 1),
                         self.convergence_history, 'b-o', markersize=4, linewidth=2)
            plt.xlabel(
                f"Iteration*2(record two r_norm between x changes in one iteration)", fontsize=12)
            plt.ylabel('Residual Norm', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path is None:
                save_path = "convergence_history.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
