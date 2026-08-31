"""显存有界的 strict QUDA 风格 Clover MultiGrid CUDA 求解器。

该路径与旧 ``applyCloverMultigridQcu`` 并列：fine 层求解目标奇偶的归一化
Clover Schur 系统，预条件器执行 fine MR -> parity R -> strict coarse V-cycle ->
parity P -> fine MR，外层使用 C++ fused 右预条件 FGMRES。粗层 hierarchy
常驻，Krylov workspace 在首次求解时一次分配并复用，迭代内不扩张显存。
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

import torch

from pyqcu.cuda import define, qcu
from pyqcu.cuda._schur_op import CudaSchurOp
from pyqcu.cuda._strict_mpi import (
    StrictMpiPreflightResult,
    collective_validate_strict_runtime,
    strict_mpi_world_communicator,
)
from pyqcu.solver._quda_multigrid import QcuStrictAssetBinding


def _strict_cuda_runtime_descriptor(
        hierarchy: Any, params: torch.Tensor) -> Dict[str, Any]:
    """提取 setup 前即可获得的 rank-local hierarchy/params 几何。"""

    if (hasattr(params, "device") and
            (params.device.type != "cpu" or not params.is_contiguous())):
        raise ValueError("params 必须是连续 CPU 张量")
    if not getattr(hierarchy, "_strict_quda", False):
        raise ValueError("CudaStrictMultigridSolver 要求 strict hierarchy")
    if int(hierarchy.target_parity) not in (0, 1):
        raise ValueError(
            "fine Clover CUDA 路径要求 hierarchy.target_parity 为 0 或 1")

    fine_shape = tuple(int(value) for value in hierarchy.fine_shape)
    params_shape = tuple(int(params[index]) for index in (
        define._LAT_X_, define._LAT_Y_, define._LAT_Z_, define._LAT_T_))
    if fine_shape != params_shape:
        raise ValueError(
            "strict hierarchy 与 params 的 rank-local fine shape 不一致："
            f"hierarchy={fine_shape}, params={params_shape}")

    transitions = int(hierarchy._transition_count)
    blocks = tuple(
        tuple(int(value) for value in block)
        for block in hierarchy._block_sizes[:transitions]
    )
    if len(blocks) != transitions:
        raise ValueError("strict hierarchy 的 block_size 数量少于层间过渡数")
    local_shapes = [fine_shape]
    for level, block in enumerate(blocks):
        current = local_shapes[-1]
        if len(block) != 4 or any(width <= 0 for width in block):
            raise ValueError(
                f"strict level {level} block_size 必须包含四个正整数")
        if any(current[axis] % block[axis] for axis in range(4)):
            raise ValueError(
                f"strict level {level} local_shape={current} 不能被 "
                f"block_size={block} 整除")
        local_shapes.append(tuple(
            current[axis] // block[axis] for axis in range(4)))

    return {
        "process_grid": tuple(int(params[index]) for index in (
            define._GRID_X_, define._GRID_Y_,
            define._GRID_Z_, define._GRID_T_)),
        "node_size": int(params[define._NODE_SIZE_]),
        "node_rank": int(params[define._NODE_RANK_]),
        "local_shapes": tuple(local_shapes),
        "block_sizes": blocks,
    }


def _collective_strict_cuda_preflight(
        hierarchy: Any, params: torch.Tensor) -> StrictMpiPreflightResult:
    """在 hierarchy setup/CUDA 分配前执行 production fail-closed。"""

    comm = strict_mpi_world_communicator()
    return collective_validate_strict_runtime(
        comm,
        lambda: _strict_cuda_runtime_descriptor(hierarchy, params),
        require_backend_ready=True,
    )


class CudaStrictMultigridSolver:
    """Clover odd-Schur + strict coarse hierarchy 的 CUDA 求解器。

    ``hierarchy`` 必须是已配置为 ``target_parity=0/1`` 的
    :class:`QudaStrictMultigrid`；fine 层使用对应目标奇偶的归一化
    Clover MATPC。``params`` 提供 fine lattice/MPI/dtype
    协议；粗层几何由 hierarchy 覆盖写入。默认将 Krylov arena 限制在
    512 MiB，并按 fused 总 workspace（含两份 coarse transfer field）自动
    缩短 restart，而不是超预算分配。
    """

    def __init__(
            self, hierarchy: Any, argv: torch.Tensor, gauge: torch.Tensor,
            clover_ee: torch.Tensor, clover_oo: torch.Tensor,
            clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
            params: torch.Tensor, restart: Optional[int] = None,
            max_krylov_bytes: Optional[int] = 512 << 20,
            retain_raw_links: bool = False,
            release_setup_assets: bool = True,
            verbose: bool = False):
        strict_mpi_preflight = _collective_strict_cuda_preflight(
            hierarchy, params)
        hierarchy.setup()
        if not getattr(hierarchy, "_strict_quda", False):
            raise ValueError("CudaStrictMultigridSolver 要求 strict hierarchy")
        if int(hierarchy.target_parity) not in (0, 1):
            raise ValueError(
                "fine Clover CUDA 路径要求 hierarchy.target_parity 为 0 或 1")
        if not hierarchy.transfers:
            raise ValueError("strict CUDA solver 至少需要一个 coarse 层")
        if len(hierarchy.operators) > 5:
            raise ValueError("strict CUDA solver 当前最多支持 5 层")
        fields = (gauge, clover_ee, clover_oo,
                  clover_ee_inv, clover_oo_inv)
        if not all(field.is_cuda and field.is_contiguous() for field in fields):
            raise ValueError("Gauge/Clover 资产必须是同设备连续 CUDA 张量")
        if len({field.device for field in fields}) != 1:
            raise ValueError("Gauge/Clover 资产必须位于同一 CUDA 设备")
        if not params.is_contiguous() or params.device.type != "cpu":
            raise ValueError("params 必须是连续 CPU 张量")
        if not argv.is_contiguous() or argv.device.type != "cpu":
            raise ValueError("argv 必须是连续 CPU 张量")

        self.hierarchy = hierarchy
        self.strict_mpi_preflight = strict_mpi_preflight
        self._default_tol = float(hierarchy.tol)
        self._default_max_iter = int(hierarchy.max_iter)
        self.release_setup_assets = bool(release_setup_assets)
        self._setup_release_report: Dict[str, Any] = {
            "sealed": False,
            "detached_setup_storage_bytes": 0,
            "detached_setup_storage_count": 0,
            "allocator_released_bytes": 0,
        }
        self.device = gauge.device
        self.verbose = bool(verbose)
        self.gauge = gauge
        self.clover_ee = clover_ee
        self.clover_oo = clover_oo
        self.clover_ee_inv = clover_ee_inv
        self.clover_oo_inv = clover_oo_inv
        self.argv = argv.clone().contiguous()
        configured = params.clone().contiguous()
        configured[define._PARITY_] = int(hierarchy.target_parity)
        configured[define._MG_NUM_LEVEL_] = len(hierarchy.operators)
        configured[define._MG_MU_PRE_] = max(
            1, int(max(hierarchy.nu_pre, hierarchy.nu_post)))
        for level, operator in enumerate(hierarchy.operators[1:], start=1):
            base = define._MG_LEVEL1_E_ + (level - 1) * define._MG_PARAMS_SIZE_
            configured[base:base + define._MG_PARAMS_SIZE_] = torch.tensor(
                [operator.dof, *operator.shape, hierarchy.coarse_max_iter,
                 int(configured[define._DATA_TYPE_]), hierarchy.restart],
                dtype=configured.dtype)
            self.argv[define._MG_LEVEL1_ATOL_ + level - 1] = (
                hierarchy.coarse_tol)

        self.requested_restart = max(
            1, int(hierarchy.restart if restart is None else restart))
        self.max_krylov_bytes = (
            None if max_krylov_bytes is None else int(max_krylov_bytes))
        if self.max_krylov_bytes is not None and self.max_krylov_bytes <= 0:
            raise ValueError("max_krylov_bytes 必须为正数或 None")
        self.nu_pre = max(0, int(hierarchy.nu_pre))
        self.nu_post = max(0, int(hierarchy.nu_post))
        self._closed = False
        self._binding: Optional[QcuStrictAssetBinding] = None
        self._coarse_initialized = False
        self.convergence_history: List[float] = []
        self.iterations = 0
        self.converged = False
        self.final_residual = float("inf")
        self.last_restart = 0
        self.last_solve_seconds = 0.0
        self._fused_workspace_resident_bytes = 0

        self.schur: Optional[CudaSchurOp] = None
        self.params = self.set_ptrs = None
        self.fine_null_vectors = None
        try:
            with torch.cuda.device(self.device):
                self.schur = CudaSchurOp(
                    self.argv, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv, device=self.device,
                    params=configured)
                self.params = self.schur.params
                self.set_ptrs = self.schur.set_ptrs
                self.fine_null_vectors = hierarchy.transfers[0].to_qcu_blocked(
                    dtype=gauge.dtype, device=self.device).contiguous()
                assets = hierarchy.qcu_strict_transition_assets(
                    dtype=gauge.dtype, device=self.device,
                    include_raw_links=retain_raw_links,
                    runtime_start_level=1)
                self._binding = QcuStrictAssetBinding(
                    self.set_ptrs, assets, start_level=1,
                    retain_raw_links=retain_raw_links)
                self.coarse_workspace_bytes = qcu.applyMultigridStrictInitQcu(
                    self.set_ptrs, self.params, 1)
                self._coarse_initialized = True

                first_coarse = hierarchy.operators[1]
                self.compact_shape = (
                    hierarchy.fine_dof, *hierarchy.fine_shape[:3],
                    hierarchy.fine_shape[3] // 2)
                self.full_shape = (
                    2, hierarchy.fine_spin, hierarchy.fine_color,
                    *hierarchy.fine_shape[:3], hierarchy.fine_shape[3] // 2)
                self._configure_fused_workspace(first_coarse)
                if self.release_setup_assets:
                    allocated_before = int(torch.cuda.memory_allocated(self.device))
                    release_report = hierarchy.seal_cuda_runtime(
                        runtime_assets_bound=True)
                    allocated_after = int(torch.cuda.memory_allocated(self.device))
                    release_report["allocator_released_bytes"] = max(
                        0, allocated_before - allocated_after)
                    self._setup_release_report = release_report
                    self.hierarchy = None
        except Exception:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def _configure_fused_workspace(self, first_coarse: Any) -> None:
        vector_elements = 1
        for extent in self.compact_shape:
            vector_elements *= int(extent)
        vector_bytes = vector_elements * self.gauge.element_size()
        coarse_elements = int(first_coarse.dof)
        for extent in first_coarse.shape:
            coarse_elements *= int(extent)
        coarse_vector_bytes = coarse_elements * self.gauge.element_size()
        effective = min(self.requested_restart, self._default_max_iter)
        if self.max_krylov_bytes is not None:
            fixed_bytes = 5 * vector_bytes + 2 * coarse_vector_bytes
            if self.max_krylov_bytes < fixed_bytes + 2 * vector_bytes:
                raise MemoryError(
                    "Krylov 预算不足：restart=1 需要 7 个 compact fine "
                    "向量和 2 个 first-coarse full 向量")
            budget_restart = (
                (self.max_krylov_bytes - fixed_bytes) // (2 * vector_bytes))
            effective = min(effective, int(budget_restart))
        if effective < 1:
            raise ValueError("strict FGMRES 要求 max_iter 和有效 restart 均为正数")
        self.restart = effective
        self.vector_bytes = vector_bytes
        self.coarse_vector_bytes = coarse_vector_bytes
        self.outer_arena_bytes = (2 * effective + 5) * vector_bytes
        self.coarse_io_bytes = 2 * coarse_vector_bytes
        self.fused_workspace_planned_bytes = (
            self.outer_arena_bytes + self.coarse_io_bytes)

    def _workspace_bytes(self, restart: int) -> int:
        return ((2 * int(restart) + 5) * int(self.vector_bytes) +
                2 * int(self.coarse_vector_bytes))

    def solve(self, full_rhs: torch.Tensor, x0: Optional[torch.Tensor] = None,
              out: Optional[torch.Tensor] = None, tol: Optional[float] = None,
              max_iter: Optional[int] = None) -> torch.Tensor:
        """求解完整 parity-split Clover 系统并重构 even 分量。"""
        if self._closed:
            raise RuntimeError("CudaStrictMultigridSolver 已关闭")
        if tuple(full_rhs.shape) != self.full_shape or not full_rhs.is_contiguous():
            raise ValueError(f"full_rhs 必须是连续张量，shape={self.full_shape}")
        if full_rhs.device != self.device or full_rhs.dtype != self.gauge.dtype:
            raise ValueError("full_rhs 的设备/精度必须与 Gauge 一致")
        if out is None:
            out = torch.empty_like(full_rhs)
        elif tuple(out.shape) != self.full_shape or not out.is_contiguous():
            raise ValueError("out 必须与 full_rhs 同形且连续")
        if x0 is not None and (tuple(x0.shape) != self.full_shape or
                               not x0.is_contiguous()):
            raise ValueError("x0 必须与 full_rhs 同形且连续")
        if out.device != self.device or out.dtype != self.gauge.dtype:
            raise ValueError("out 的设备/精度必须与 full_rhs 一致")
        if x0 is not None and (
                x0.device != self.device or x0.dtype != self.gauge.dtype):
            raise ValueError("x0 的设备/精度必须与 full_rhs 一致")

        tolerance = float(self._default_tol if tol is None else tol)
        iteration_budget = int(
            self._default_max_iter if max_iter is None else max_iter)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tol 必须位于 (0, 1)")
        if iteration_budget < 1:
            raise ValueError("max_iter 必须为正数")
        solve_restart = min(self.restart, iteration_budget)
        expected_workspace = self._workspace_bytes(solve_restart)
        if (self.max_krylov_bytes is not None and
                expected_workspace > self.max_krylov_bytes):
            raise MemoryError("有效 strict FGMRES workspace 超过显存预算")
        if x0 is None:
            self.params[define._MG_USE_INIT_GUESS_] = 0
        else:
            out.copy_(x0)
            self.params[define._MG_USE_INIT_GUESS_] = 1

        start = perf_counter()
        result = qcu.applyMultigridStrictFgmresQcu(
            out, full_rhs, self.gauge, self.clover_ee, self.clover_oo,
            self.clover_ee_inv, self.clover_oo_inv, self.fine_null_vectors,
            self.set_ptrs, self.params, solve_restart, iteration_budget,
            tolerance, self.nu_pre, self.nu_post, self.max_krylov_bytes)
        elapsed = perf_counter() - start
        allocated_bytes = int(result["allocated_bytes"])
        if allocated_bytes != expected_workspace:
            raise RuntimeError(
                "strict fused workspace 账本漂移："
                f"expected={expected_workspace}, actual={allocated_bytes}")
        self.iterations = int(result["iterations"])
        self.converged = bool(result["converged"])
        self.final_residual = float(result["final_true_residual"])
        self.convergence_history = [self.final_residual]
        self.last_restart = solve_restart
        self.last_solve_seconds = elapsed
        self._fused_workspace_resident_bytes = allocated_bytes
        if self.verbose:
            print(
                "PYQCU::SOLVER::STRICT_MG::FGMRES:\n "
                f"iterations={self.iterations} restart={solve_restart} "
                f"converged={self.converged} "
                f"true_residual={self.final_residual:.6e} "
                f"workspace={allocated_bytes} time={elapsed:.6f} sec")
        return out

    def memory_report(self) -> Dict[str, Any]:
        """返回本求解器可精确归属的显存字节账本。"""
        if self._closed:
            raise RuntimeError("CudaStrictMultigridSolver 已关闭")
        binding = ({"resident_bytes": 0, "omitted_raw_bytes": 0}
                   if self._binding is None else self._binding.memory_report())
        assert self.fine_null_vectors is not None
        fine_transfer = int(
            self.fine_null_vectors.numel() *
            self.fine_null_vectors.element_size())
        coarse_io = int(self.coarse_io_bytes)
        borrowed_storages: Dict[int, int] = {}
        for field in (self.gauge, self.clover_ee, self.clover_oo,
                      self.clover_ee_inv, self.clover_oo_inv):
            storage = field.untyped_storage()
            borrowed_storages[int(storage.data_ptr())] = int(storage.nbytes())
        borrowed_inputs = sum(borrowed_storages.values())
        _, x, y, z, t_half = self.compact_shape
        surface = (
            y * z * t_half + x * z * t_half +
            x * y * t_half + x * y * z)
        element_size = int(self.gauge.element_size())
        lattice_scratch = 3 * int(self.vector_bytes)
        lattice_device_halo = 24 * int(surface) * element_size
        lattice_pinned_halo = lattice_device_halo
        lattice_params_scalars = 5 * 58 * 4 + 11 * element_size
        lattice_known = (
            lattice_scratch + lattice_device_halo + lattice_params_scalars)
        fused_resident = int(self._fused_workspace_resident_bytes)
        total = (int(binding["resident_bytes"]) + fine_transfer +
                 int(self.coarse_workspace_bytes) + fused_resident)
        planned_total = (
            total - fused_resident + int(self.fused_workspace_planned_bytes))
        return {
            "requested_restart": int(self.requested_restart),
            "effective_restart": int(self.restart),
            "vector_bytes": int(self.vector_bytes),
            "outer_arena_bytes": int(self.outer_arena_bytes),
            "coarse_io_bytes": coarse_io,
            "python_outer_arena_bytes": 0,
            "python_coarse_io_bytes": 0,
            "fused_workspace_planned_bytes": int(
                self.fused_workspace_planned_bytes),
            "fused_workspace_resident_bytes": fused_resident,
            "fused_workspace_budget_bytes": self.max_krylov_bytes,
            "last_solve_restart": int(self.last_restart),
            "coarse_workspace_bytes": int(self.coarse_workspace_bytes),
            "asset_resident_bytes": int(binding["resident_bytes"]),
            "fine_transfer_bytes": fine_transfer,
            "omitted_raw_bytes": int(binding["omitted_raw_bytes"]),
            "accounted_owned_bytes": total,
            "planned_accounted_owned_bytes": planned_total,
            "hierarchy_sealed": bool(
                self._setup_release_report.get("sealed", False)),
            "setup_detached_storage_bytes": int(
                self._setup_release_report.get(
                    "detached_setup_storage_bytes", 0)),
            "setup_detached_storage_count": int(
                self._setup_release_report.get(
                    "detached_setup_storage_count", 0)),
            "setup_allocator_released_bytes": int(
                self._setup_release_report.get(
                    "allocator_released_bytes", 0)),
            "borrowed_gauge_clover_bytes": int(borrowed_inputs),
            "lattice_scratch_requested_bytes": int(lattice_scratch),
            "lattice_device_halo_requested_bytes": int(lattice_device_halo),
            "lattice_pinned_host_halo_requested_bytes": int(
                lattice_pinned_halo),
            "lattice_params_scalars_requested_bytes": int(
                lattice_params_scalars),
            "lattice_known_device_bytes": int(lattice_known),
            "known_live_device_bytes": int(
                total + borrowed_inputs + lattice_known),
            "planned_known_device_bytes": int(
                planned_total + borrowed_inputs + lattice_known),
            "accounting_scope": (
                "runtime-owned; setup hierarchy detached" if
                self._setup_release_report.get("sealed", False) else
                "runtime-owned only; live setup hierarchy excluded"),
            "unaccounted_runtime": (
                "cuBLAS handles/workspace, CUDA context, allocator rounding/"
                "reserve, caller rhs/out"),
        }

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        with torch.cuda.device(self.device):
            if self._coarse_initialized and self.schur is not None:
                try:
                    qcu.applyMultigridStrictEndQcu(
                        self.schur.set_ptrs, self.schur.params)
                except Exception as error:  # pragma: no cover - cleanup path
                    first_error = error
                self._coarse_initialized = False
            if self._binding is not None:
                self._binding.close()
                self._binding = None
            if self.schur is not None:
                try:
                    self.schur.release()
                except Exception as error:  # pragma: no cover - cleanup path
                    if first_error is None:
                        first_error = error
                self.schur = None

        self._fused_workspace_resident_bytes = 0
        self.fine_null_vectors = None
        self.gauge = self.clover_ee = self.clover_oo = None
        self.clover_ee_inv = self.clover_oo_inv = None
        self.hierarchy = None
        self.params = self.set_ptrs = None
        self.argv = None
        self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "CudaStrictMultigridSolver":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CudaStrictMultigridSolver"]
