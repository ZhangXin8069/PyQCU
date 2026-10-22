"""Shared compact-Schur adapter for the CUDA/KUPL flat QCU ABI."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional

import pyqcu.cann as _torch
from pyqcu.api import (
    BackendCapabilities,
    BiCGStabControl,
    CGControl,
    CGOperator,
    LatticeGeometry,
    NativeSolve,
    ResidualScope,
    SolverControl,
    SolveStatus,
)
from pyqcu.cuda import define


class FlatQcuLinearOperator:
    """Compact parity-Schur operator plus whole-solve native access."""

    def __init__(
            self, *, backend: str, runtime: Any, gauge: Any,
            parity: int, schur_bicgstab_operation: str,
            normal_cg_operation: Optional[str],
            solver_bicgstab: str, solver_cg: Optional[str],
            iteration_getter: Optional[str],
            clover_assets: Optional[tuple[Any, ...]] = None):
        if not hasattr(runtime, "operation") or not hasattr(runtime, "call"):
            raise TypeError("runtime must provide call() and operation()")
        parity_value = int(parity)
        if parity_value not in (0, 1):
            raise ValueError("parity must be 0 or 1")
        shape = tuple(int(value) for value in runtime.lattice)
        if len(shape) != 4 or any(value <= 0 or value % 2 for value in shape):
            raise ValueError("QCU lattice requires four positive even extents")
        compact_shape = (*shape[:3], shape[3] // 2)
        expected_gauge = (2, 3, 3, 4, *compact_shape)
        if tuple(int(value) for value in gauge.shape) != expected_gauge:
            raise ValueError(
                "gauge must have shape [2,3,3,4,X,Y,Z,T/2], "
                f"got {tuple(gauge.shape)}")
        if not bool(gauge.is_contiguous()):
            raise ValueError("gauge must be contiguous")
        if gauge.device != runtime.device:
            raise ValueError("gauge device must match runtime device")
        self.backend = str(backend).strip().lower()
        self.runtime = runtime
        self.gauge = gauge
        self.parity = parity_value
        self._compact_shape = compact_shape
        self._schur_bicgstab_operation = schur_bicgstab_operation
        self._normal_cg_operation = normal_cg_operation
        self._solver_bicgstab = solver_bicgstab
        self._solver_cg = solver_cg
        self._iteration_getter = iteration_getter
        self._clover_assets = tuple(clover_assets or ())
        if self._clover_assets and len(self._clover_assets) != 4:
            raise ValueError(
                "clover_assets must contain ce, coo, ce_inv, coo_inv")
        expected_clover = (4, 3, 4, 3, *compact_shape)
        for index, asset in enumerate(self._clover_assets):
            if tuple(int(value) for value in asset.shape) != expected_clover:
                raise ValueError(
                    f"clover_assets[{index}] must have shape {expected_clover}")
            if not bool(asset.is_contiguous()):
                raise ValueError(f"clover_assets[{index}] must be contiguous")
            if asset.dtype != gauge.dtype or asset.device != gauge.device:
                raise ValueError(
                    f"clover_assets[{index}] dtype/device must match gauge")
        self.geometry = LatticeGeometry(
            local_shape=shape,
            dtype=_dtype_name(gauge.dtype),
            process_grid=_process_grid(runtime.params),
            parity=parity_value,
            layout="poooxyzt",
        )
        self.residual_scope = ResidualScope.SCHUR
        self.capabilities = MappingProxyType({
            "adjoint": False,
            "bicgstab": True,
            "cg": self._solver_cg is not None,
            "matpc": False,
            "native_krylov": True,
            "python_krylov": False,
            "normal_cg": self._solver_cg is not None,
        })

    def apply(self, x: Any, *, parity: Optional[int] = None,
              adjoint: bool = False) -> Any:
        if adjoint:
            return self.apply_adjoint(x, parity=parity)
        self._validate_compact(x, "x")
        self._validate_parity(parity)
        output = _torch.torch.empty_like(x)
        self.runtime.params[define._SET_PLAN_] = define._SET_PLAN1_
        self.runtime.params[define._PARITY_] = self.parity
        self.runtime.call(
            self._schur_bicgstab_operation, output, x, self.gauge,
            *self._clover_assets)
        return output

    def apply_adjoint(self, x: Any, *, parity: Optional[int] = None) -> Any:
        del x, parity
        raise NotImplementedError(
            f"{self.backend} compact Schur adapter has no adjoint ABI")

    def diagonal(self, parity: Optional[int] = None) -> Any:
        del parity
        raise NotImplementedError(
            "flat QCU compact Schur adapter has no diagonal ABI")

    def apply_normal(self, x: Any, *, parity: Optional[int] = None) -> Any:
        if self._normal_cg_operation is None:
            raise NotImplementedError(
                f"{self.backend} compact Schur adapter has no normal apply ABI")
        self._validate_compact(x, "x")
        self._validate_parity(parity)
        output = _torch.torch.empty_like(x)
        self.runtime.params[define._SET_PLAN_] = define._SET_PLAN1_
        self.runtime.params[define._PARITY_] = self.parity
        self.runtime.call(
            self._normal_cg_operation, output, x, self.gauge,
            *self._clover_assets)
        return output

    def apply_matpc(self, x: Any, *, parity: Optional[int] = None) -> Any:
        del x, parity
        raise NotImplementedError(
            "flat QCU compact Schur adapter does not expose normalized MATPC")

    def capability_report(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend,
            parity=True,
            normal_cg=self._solver_cg is not None,
            native_krylov=True,
            python_krylov=False,
            supported_precisions=("c64", "c128"),
        )

    def solve_native(
            self, method: str, rhs: Any, x0: Any,
            control: SolverControl) -> NativeSolve:
        method_name = str(method).strip().lower().replace("-", "_")
        aliases = {"bicgstab": "bicgstab", "bistabcg": "bicgstab", "cg": "cg"}
        if method_name not in aliases:
            raise ValueError("native method must be bicgstab or cg")
        method_name = aliases[method_name]
        self._validate_full_field(rhs, "rhs")
        if x0 is not None:
            self._validate_full_field(x0, "x0")
        self._validate_control(method_name, control)
        operation_name = (
            self._solver_bicgstab if method_name == "bicgstab"
            else self._solver_cg)
        if operation_name is None:
            raise NotImplementedError(
                f"{self.backend} does not provide a native CG adapter")

        rhs_norm = float(_torch.norm(rhs))
        if rhs_norm == 0.0:
            solution = _torch.zeros_like(rhs)
            return NativeSolve(
                x=solution,
                status=SolveStatus.CONVERGED,
                converged=True,
                iterations=0,
                residual_scope=self._method_scope(method_name),
                diagnostics={"zero_rhs": True, "native_call_count": 0},
            )

        self.runtime.argv[define._ATOL_] = control.rtol
        self.runtime.params[define._MAX_ITER_] = control.max_iter
        self.runtime.params[define._SET_PLAN_] = define._SET_PLAN1_
        self.runtime.params[define._PARITY_] = self.parity
        solution = _torch.torch.empty_like(rhs)
        if x0 is None:
            solution.zero_()
            self.runtime.params[define._MG_USE_INIT_GUESS_] = 0
        else:
            solution.copy_(x0)
            self.runtime.params[define._MG_USE_INIT_GUESS_] = 1

        with self.runtime.operation(operation_name) as operation:
            operation(solution, rhs, self.gauge, *self._clover_assets)
            iterations = (
                None if self._iteration_getter is None
                else int(operation.read(self._iteration_getter)))
        return NativeSolve(
            x=solution,
            status=SolveStatus.CONVERGED,
            converged=True,
            iterations=iterations,
            residual_scope=self._method_scope(method_name),
            diagnostics={
                "abi_symbol": operation_name,
                "effective_rtol": control.rtol,
                "native_call_count": 1,
                "native_normal_return": True,
                "rhs_norm": rhs_norm,
                "warm_start": x0 is not None,
            },
        )

    def _validate_compact(self, value: Any, name: str) -> None:
        shape = tuple(int(item) for item in value.shape)
        valid = (
            len(shape) == 5 and shape[0] == 12 and shape[1:] == self._compact_shape
        ) or (
            len(shape) == 6 and shape[:2] == (4, 3) and
            shape[2:] == self._compact_shape)
        if not valid:
            raise ValueError(
                f"{name} must be [12,X,Y,Z,T/2] or [4,3,X,Y,Z,T/2]")
        if not bool(value.is_contiguous()):
            raise ValueError(f"{name} must be contiguous")
        if value.dtype != self.gauge.dtype:
            raise ValueError(f"{name} dtype must match gauge")
        if value.device != self.gauge.device:
            raise ValueError(f"{name} device must match gauge")

    def _validate_full_field(self, value: Any, name: str) -> None:
        shape = tuple(int(item) for item in value.shape)
        expected = (2, 4, 3, *self._compact_shape)
        if shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {shape}")
        if not bool(value.is_contiguous()):
            raise ValueError(f"{name} must be contiguous")
        if value.dtype != self.gauge.dtype:
            raise ValueError(f"{name} dtype must match gauge")
        if value.device != self.gauge.device:
            raise ValueError(f"{name} device must match gauge")

    def _validate_parity(self, parity: Optional[int]) -> None:
        if parity is not None and int(parity) != self.parity:
            raise ValueError(
                f"operator parity={self.parity}, got parity={parity}")

    @staticmethod
    def _validate_control(method: str, control: SolverControl) -> None:
        if method == "bicgstab" and not isinstance(control, BiCGStabControl):
            raise TypeError("bicgstab requires BiCGStabControl")
        if method == "cg" and not isinstance(control, CGControl):
            raise TypeError("cg requires CGControl")
        if control.atol != 0:
            raise ValueError("native QCU drivers support relative tolerance only")
        if control.min_iter != 0:
            raise ValueError("native QCU drivers do not expose min_iter")
        if control.true_residual_every is not None:
            raise ValueError(
                "native QCU drivers do not expose periodic residual checks")
        if control.residual_scope is not ResidualScope.FULL:
            raise ValueError(
                "native QCU control must target the final full residual")
        if isinstance(control, BiCGStabControl):
            if control.breakdown_restarts != 0:
                raise ValueError(
                    "native QCU BiCGStab does not expose breakdown restarts")
        elif isinstance(control, CGControl):
            if control.operator_kind is not CGOperator.NORMAL:
                raise ValueError(
                    "native QCU CG solves the normal operator and requires "
                    "CGOperator.NORMAL")
        else:
            raise TypeError("native QCU requires BiCGStabControl or CGControl")

    @staticmethod
    def _method_scope(method: str) -> ResidualScope:
        return ResidualScope.SCHUR if method == "bicgstab" else ResidualScope.NORMAL


def _process_grid(params: Any) -> tuple[int, int, int, int]:
    grid = tuple(int(params[index]) for index in (
        define._GRID_X_, define._GRID_Y_, define._GRID_Z_, define._GRID_T_))
    if any(value <= 0 for value in grid):
        return (1, 1, 1, 1)
    return grid  # type: ignore[return-value]


def _dtype_name(dtype: Any) -> str:
    names = {
        _torch.torch.complex64: "c64",
        _torch.torch.complex128: "c128",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise TypeError("flat QCU operator supports c64/c128") from error


__all__ = ["FlatQcuLinearOperator"]
