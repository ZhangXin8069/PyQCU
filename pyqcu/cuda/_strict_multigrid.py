"""显存有界的 strict QUDA 风格 Clover MultiGrid CUDA 求解器。

该路径与旧 ``applyCloverMultigridQcu`` 并列：fine 层求解 odd Clover
Schur 系统，预条件器执行 fine MR -> parity R -> strict coarse V-cycle ->
parity P -> fine MR，外层使用右预条件 FGMRES。粗层 hierarchy 和 Krylov
张量均在求解前一次分配并复用，运行期不逐迭代扩张显存。
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

import torch

from pyqcu import tools
from pyqcu.cuda import define, qcu
from pyqcu.cuda._schur_op import CudaSchurOp
from pyqcu.solver._gmres import _givens_rotation, _solve_upper_triangular
from pyqcu.solver._quda_multigrid import QcuStrictAssetBinding


class CudaStrictMultigridSolver:
    """Clover odd-Schur + strict coarse hierarchy 的 CUDA 求解器。

    ``hierarchy`` 必须是已配置为 ``target_parity=1`` 的
    :class:`QudaStrictMultigrid`。``params`` 提供 fine lattice/MPI/dtype
    协议；粗层几何由 hierarchy 覆盖写入。默认将 Krylov arena 限制在
    512 MiB，并自动缩短 restart，而不是超预算分配。
    """

    def __init__(
            self, hierarchy: Any, argv: torch.Tensor, gauge: torch.Tensor,
            clover_ee: torch.Tensor, clover_oo: torch.Tensor,
            clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
            params: torch.Tensor, restart: Optional[int] = None,
            max_krylov_bytes: Optional[int] = 512 << 20,
            retain_raw_links: bool = False, verbose: bool = False):
        hierarchy.setup()
        if not getattr(hierarchy, "_strict_quda", False):
            raise ValueError("CudaStrictMultigridSolver 要求 strict hierarchy")
        if int(hierarchy.target_parity) != 1:
            raise ValueError(
                "fine Clover CUDA 路径固定 odd Schur；hierarchy.target_parity 必须为 1")
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
        self.device = gauge.device
        self.verbose = bool(verbose)
        self.gauge = gauge
        self.clover_ee = clover_ee
        self.clover_oo = clover_oo
        self.clover_ee_inv = clover_ee_inv
        self.clover_oo_inv = clover_oo_inv
        self.argv = argv.clone().contiguous()
        configured = params.clone().contiguous()
        configured[define._PARITY_] = 1
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
        self._arena: Optional[torch.Tensor] = None
        self._V = self._Z = self._b = self._r = self._w = self._x = None
        self.convergence_history: List[float] = []
        self.iterations = 0
        self.converged = False
        self.final_residual = float("inf")

        self.schur: Optional[CudaSchurOp] = None
        self.params = self.set_ptrs = None
        self.fine_null_vectors = None
        self._coarse_rhs = self._coarse_out = None
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
                coarse_shape = (first_coarse.dof, *first_coarse.shape)
                self._coarse_rhs = torch.empty(
                    coarse_shape, dtype=gauge.dtype, device=self.device)
                self._coarse_out = torch.empty_like(self._coarse_rhs)
                self.compact_shape = (
                    hierarchy.fine_dof, *hierarchy.fine_shape[:3],
                    hierarchy.fine_shape[3] // 2)
                self.full_shape = (
                    2, hierarchy.fine_spin, hierarchy.fine_color,
                    *hierarchy.fine_shape[:3], hierarchy.fine_shape[3] // 2)
                self._ensure_arena()
        except Exception:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_arena(self) -> None:
        vector_elements = 1
        for extent in self.compact_shape:
            vector_elements *= int(extent)
        vector_bytes = vector_elements * self.gauge.element_size()
        effective = self.requested_restart
        if self.max_krylov_bytes is not None:
            available_vectors = self.max_krylov_bytes // vector_bytes
            if available_vectors < 7:
                raise MemoryError(
                    "Krylov 预算不足：restart=1 也需要 7 个 compact fine 向量")
            effective = min(effective, max(1, (available_vectors - 5) // 2))
        self.restart = effective
        # V[m+1] + Z[m] + prepared rhs/r/w/x，共 2m+5 个向量。
        self._arena = torch.empty(
            (2 * effective + 5, *self.compact_shape),
            dtype=self.gauge.dtype, device=self.device)
        cursor = 0
        self._V = self._arena[cursor:cursor + effective + 1]
        cursor += effective + 1
        self._Z = self._arena[cursor:cursor + effective]
        cursor += effective
        self._b = self._arena[cursor]
        self._r = self._arena[cursor + 1]
        self._w = self._arena[cursor + 2]
        self._x = self._arena[cursor + 3]
        self.vector_bytes = vector_bytes
        self.outer_arena_bytes = int(
            self._arena.numel() * self._arena.element_size())

    def _matvec_into(self, out: torch.Tensor, source: torch.Tensor) -> None:
        assert self.schur is not None
        self.schur.matvec_into(out, source)

    def _smooth_into(self, solution: torch.Tensor, residual: torch.Tensor,
                     image: torch.Tensor, steps: int) -> None:
        for _ in range(steps):
            self._matvec_into(image, residual)
            denominator = tools.vdot(image.reshape(-1), image.reshape(-1))
            if float(torch.abs(denominator).item()) <= 1.0e-20:
                break
            alpha = (tools.vdot(image.reshape(-1), residual.reshape(-1)) /
                     denominator).item()
            solution.add_(residual, alpha=alpha)
            residual.add_(image, alpha=-alpha)

    def _precondition_into(self, out: torch.Tensor,
                           source: torch.Tensor) -> None:
        assert self._r is not None and self._w is not None
        out.zero_()
        self._r.copy_(source)
        self._smooth_into(out, self._r, self._w, self.nu_pre)
        qcu.applyMultigridStrictRestrictQcu(
            self._coarse_rhs, self._r, self.fine_null_vectors,
            self.set_ptrs, self.params, 1)
        workspace = qcu.applyMultigridStrictVCycleQcu(
            self._coarse_out, self._coarse_rhs,
            self.set_ptrs, self.params, 1)
        if workspace != self.coarse_workspace_bytes:
            raise RuntimeError("strict coarse workspace 字节数在运行期发生变化")
        qcu.applyMultigridStrictProLongQcu(
            self._w, self._coarse_out, self.fine_null_vectors,
            self.set_ptrs, self.params, 1)
        out.add_(self._w)
        self._matvec_into(self._w, out)
        self._r.copy_(source).sub_(self._w)
        self._smooth_into(out, self._r, self._w, self.nu_post)

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
        assert all(value is not None for value in
                   (self._V, self._Z, self._b, self._r, self._w, self._x))

        qcu.applyCloverBistabCgPrepareQcu(
            self._b, full_rhs, self.gauge, self.clover_ee, self.clover_oo,
            self.clover_ee_inv, self.clover_oo_inv,
            self.set_ptrs, self.params)
        if x0 is None:
            self._x.zero_()
        else:
            self._x.copy_(x0[1].reshape(self.compact_shape))
        self._matvec_into(self._w, self._x)
        self._r.copy_(self._b).sub_(self._w)
        b_norm = tools.norm(self._b.reshape(-1))
        error = tools.norm(self._r.reshape(-1))
        threshold = float(self.hierarchy.tol if tol is None else tol) * b_norm
        iteration_budget = int(
            self.hierarchy.max_iter if max_iter is None else max_iter)
        self.convergence_history = [float(error)]
        self.iterations = 0
        self.converged = (error == 0.0 or b_norm == 0.0 or error < threshold)
        start = perf_counter()

        while self.iterations < iteration_budget and not self.converged:
            beta = error
            self._V[0].copy_(self._r).div_(beta)
            m = min(self.restart, iteration_budget - self.iterations)
            H = [[0.0 + 0.0j] * m for _ in range(m + 1)]
            g = [0.0 + 0.0j] * (m + 1)
            g[0] = complex(beta)
            sn = [0.0 + 0.0j] * m
            cs = [0.0 + 0.0j] * m
            inner = 0
            for j in range(m):
                self._precondition_into(self._Z[j], self._V[j])
                self._matvec_into(self._w, self._Z[j])
                for i in range(j + 1):
                    coefficient = tools.vdot(
                        self._V[i].reshape(-1), self._w.reshape(-1)).item()
                    H[i][j] = coefficient
                    self._w.add_(self._V[i], alpha=-coefficient)
                h_next = tools.norm(self._w.reshape(-1))
                H[j + 1][j] = h_next
                if h_next > 0.0:
                    self._V[j + 1].copy_(self._w).div_(h_next)
                _givens_rotation(H, g, cs, sn, j)
                self.iterations += 1
                inner = j + 1
                estimate = float(abs(g[j + 1]))
                self.convergence_history.append(estimate)
                if estimate < threshold:
                    break
            eta = _solve_upper_triangular(H, g, inner)
            for j in range(inner):
                self._x.add_(self._Z[j], alpha=eta[j])
            self._matvec_into(self._w, self._x)
            self._r.copy_(self._b).sub_(self._w)
            error = tools.norm(self._r.reshape(-1))
            self.convergence_history.append(float(error))
            self.converged = error < threshold

        qcu.applyCloverBistabCgReconstructQcu(
            out, full_rhs, self._x, self.gauge, self.clover_ee,
            self.clover_oo, self.clover_ee_inv, self.clover_oo_inv,
            self.set_ptrs, self.params)
        self.final_residual = float(error)
        if self.verbose:
            elapsed = perf_counter() - start
            print(
                "PYQCU::SOLVER::STRICT_MG::FGMRES:\n "
                f"iterations={self.iterations} restart={self.restart} "
                f"residual={error:.6e} time={elapsed:.6f} sec")
        return out

    def memory_report(self) -> Dict[str, int]:
        """返回本求解器可精确归属的显存字节账本。"""
        if self._closed:
            raise RuntimeError("CudaStrictMultigridSolver 已关闭")
        binding = ({"resident_bytes": 0, "omitted_raw_bytes": 0}
                   if self._binding is None else self._binding.memory_report())
        assert self.fine_null_vectors is not None
        assert self._coarse_rhs is not None and self._coarse_out is not None
        fine_transfer = int(
            self.fine_null_vectors.numel() *
            self.fine_null_vectors.element_size())
        coarse_io = int(
            (self._coarse_rhs.numel() + self._coarse_out.numel()) *
            self._coarse_rhs.element_size())
        total = (int(binding["resident_bytes"]) + fine_transfer + coarse_io +
                 int(self.coarse_workspace_bytes) + self.outer_arena_bytes)
        return {
            "requested_restart": int(self.requested_restart),
            "effective_restart": int(self.restart),
            "vector_bytes": int(self.vector_bytes),
            "outer_arena_bytes": int(self.outer_arena_bytes),
            "coarse_io_bytes": coarse_io,
            "coarse_workspace_bytes": int(self.coarse_workspace_bytes),
            "asset_resident_bytes": int(binding["resident_bytes"]),
            "fine_transfer_bytes": fine_transfer,
            "omitted_raw_bytes": int(binding["omitted_raw_bytes"]),
            "accounted_owned_bytes": total,
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

        # Views keep their base storage alive, so clear every view before the arena.
        self._V = self._Z = self._b = self._r = self._w = self._x = None
        self._arena = None
        self._coarse_rhs = self._coarse_out = None
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
