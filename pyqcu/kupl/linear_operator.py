"""KUPL LinearOperator adapter over the flat QCU ABI."""

from __future__ import annotations

from typing import Any

from pyqcu.solver._native_qcu import FlatQcuLinearOperator


class KuplLinearOperator(FlatQcuLinearOperator):
    """One-shot KUPL Wilson operator and whole native Krylov adapter."""

    def __init__(
            self, *, runtime: Any, gauge: Any, parity: int = 1,
            clover: Any = None):
        clover_assets = None if clover is None else tuple(clover)
        clover_mode = clover_assets is not None
        super().__init__(
            backend="kupl",
            runtime=runtime,
            gauge=gauge,
            parity=parity,
            schur_bicgstab_operation=(
                "applyCloverBistabCgDslashQcu" if clover_mode
                else "applyWilsonBistabCgDslashQcu"),
            normal_cg_operation=(None if clover_mode else "applyWilsonCgDslashQcu"),
            solver_bicgstab=(
                "applyCloverBistabCgQcu" if clover_mode
                else "applyWilsonBistabCgQcu"),
            solver_cg=None if clover_mode else "applyWilsonCgQcu",
            iteration_getter="getCloverBistabCgIterationsQcu",
            clover_assets=clover_assets,
        )


__all__ = ["KuplLinearOperator"]
