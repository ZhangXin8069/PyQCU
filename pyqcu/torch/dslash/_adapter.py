"""Torch adapter for the backend-neutral LinearOperator protocol."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional

import pyqcu.cann as _torch
from pyqcu import lattice, tools
from pyqcu.api import BackendCapabilities, LatticeGeometry, ResidualScope


class TorchLinearOperator:
    """Expose a PyQCU Torch Wilson/Clover operator through pyqcu.api."""

    backend = "torch"

    def __init__(self, operator: Any, *, mode: str = "full"):
        mode_name = str(mode).strip().lower().replace("-", "_")
        aliases = {"full": "full", "schur": "schur", "parity": "schur"}
        if mode_name not in aliases:
            raise ValueError("mode must be full, schur, or parity")
        self._mode = aliases[mode_name]
        self.operator = operator
        gauge = getattr(getattr(operator, "hopping", None), "U", None)
        if gauge is None:
            raise ValueError("TorchLinearOperator requires a physical gauge field")
        self._shape = tuple(int(value) for value in gauge.shape[-4:])
        self._dtype = _dtype_name(gauge.dtype)
        self.geometry = LatticeGeometry(
            local_shape=self._shape,
            dtype=self._dtype,
            process_grid=tuple(int(value) for value in tools.give_grid_size()),
            parity=(1 if self._mode == "schur" else None),
            layout="xyzt",
        )
        self.residual_scope = (
            ResidualScope.FULL if self._mode == "full"
            else ResidualScope.SCHUR)
        self.capabilities = MappingProxyType({
            "adjoint": True,
            "parity": bool(getattr(operator, "hopping", None) is not None),
            "matpc": False,
            "python_krylov": True,
            "native_krylov": False,
            "hpd_cg": False,
        })

    def apply(self, x: Any, *, parity: Optional[int] = None,
              adjoint: bool = False) -> Any:
        if adjoint:
            return self.apply_adjoint(x, parity=parity)
        if self._mode == "full":
            if parity is not None:
                raise ValueError("full TorchLinearOperator does not accept parity")
            return self.operator.matvec(x)
        if parity not in (None, 1):
            raise ValueError("Torch Schur adapter acts on parity=1")
        if x.ndim >= 2 and int(x.shape[0]) == 4 and int(x.shape[1]) == 3:
            return self.operator.matvec_parity4fermion(x)
        return self.operator.matvec_parity(x)

    def apply_adjoint(self, x: Any, *, parity: Optional[int] = None) -> Any:
        transformed = _gamma5_apply(x)
        image = self.apply(transformed, parity=parity)
        return _gamma5_apply(image)

    def diagonal(self, parity: Optional[int] = None) -> Any:
        if self._mode != "full" or parity is not None:
            raise NotImplementedError("Schur adapter does not provide a diagonal")
        gauge = self.operator.hopping.U
        if getattr(self.operator.sitting, "clover_term", None) is None:
            return _torch.eye(
                12, dtype=gauge.dtype, device=gauge.device).reshape(
                    12, 12, 1, 1, 1, 1).expand(
                        12, 12, *self._shape).clone()
        return _torch.einsum("iixyzt->ixyzt", self.operator.sitting.M)

    def apply_normal(self, x: Any, *, parity: Optional[int] = None) -> Any:
        return self.apply_adjoint(self.apply(x, parity=parity), parity=parity)

    def apply_matpc(self, x: Any, *, parity: Optional[int] = None) -> Any:
        raise NotImplementedError(
            "TorchLinearOperator does not expose a normalized MATPC view")

    def capability_report(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="torch",
            parity=bool(self.capabilities["parity"]),
            adjoint=True,
            python_krylov=True,
            supported_precisions=("c64", "c128"),
        )


def _gamma5_apply(value: Any) -> Any:
    original_shape = tuple(value.shape)
    if len(original_shape) < 2:
        raise ValueError("fermion field must have spin/color dimensions")
    if int(original_shape[0]) == 12:
        structured = value.reshape(4, 3, *original_shape[1:])
        flattened = True
    elif int(original_shape[0]) == 4 and int(original_shape[1]) == 3:
        structured = value
        flattened = False
    else:
        raise ValueError("fermion field must be [12,...] or [4,3,...]")
    matrix = lattice.gamma_5.to(
        device=value.device, dtype=value.dtype)
    result = _torch.einsum("ab,bcxyzt->acxyzt", matrix, structured)
    if flattened:
        return result.reshape(original_shape)
    return result


def _dtype_name(dtype: Any) -> str:
    names = {
        _torch.torch.complex64: "c64",
        _torch.torch.complex128: "c128",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise TypeError("operator supports complex64 or complex128 only") from error


__all__ = ["TorchLinearOperator"]
