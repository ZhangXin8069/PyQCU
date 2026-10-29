"""Backend-neutral multigrid selector.

Concrete Torch, CUDA and KUPL hierarchies remain in their backend packages;
this module only normalizes the public selection and lifecycle interface.
"""

from __future__ import annotations

from importlib import import_module
from time import perf_counter
from typing import Any

from pyqcu.api import BackendCapabilities, SolveResult
from pyqcu.api._compat import compatibility_result
from pyqcu.torch.solver.multigrid import LegacyMultigrid, StrictMultigrid
from .cache import (
    LegacyMultigridCache,
    MultigridCache,
    MultigridCacheLoadResult,
    StrictMultigridCache,
)


class Multigrid:
    """Backend-neutral wrapper around a strict or legacy hierarchy.

    Legacy-MultiGrid is the default QCU multigrid solver and the canonical
    development/debug/optimization target. Strict-MultiGrid is retained for
    explicit comparison, algorithm reference, and special fallback use.

    Depth warning policy: Legacy is validated at two levels and emits
    ``MultigridDepthWarning`` above that depth. Strict is validated at three
    levels and emits the same warning above three. A warning is not a
    correctness gate; experimental hierarchies still require independent
    residual validation.
    """

    _BACKENDS = {"torch", "cuda", "kupl"}
    _VARIANTS = {"strict", "legacy"}
    DEFAULT_VARIANT = "legacy"

    def __init__(self, *, variant: str = DEFAULT_VARIANT,
                 backend: str = "torch", **kwargs: Any):
        # Do not make Strict an implicit default: comparison/reference paths
        # must be selected by name so production and tuning target one solver.
        variant_name = str(variant).strip().lower().replace("-", "_")
        backend_name = str(backend).strip().lower()
        if variant_name not in self._VARIANTS:
            raise ValueError(
                f"variant 必须是 legacy（默认）或 strict，得到 {variant!r}")
        if backend_name not in self._BACKENDS:
            raise ValueError(
                f"backend 必须是 torch、cuda 或 kupl，得到 {backend!r}")

        implementation_type = _implementation_type(
            backend_name, variant_name)
        self.variant = variant_name
        self.backend = backend_name
        self._implementation = implementation_type(**kwargs)

    @property
    def implementation(self) -> Any:
        """Concrete backend object wrapped by this selector."""
        return self._implementation

    def setup(self) -> "Multigrid":
        """Build or validate the selected hierarchy."""
        setup = getattr(self._implementation, "setup", None)
        if setup is None:
            raise TypeError(
                f"{type(self._implementation).__name__} 未实现 setup()")
        setup()
        return self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        """Solve ``D x = rhs`` with the selected hierarchy."""
        solve = getattr(self._implementation, "solve", None)
        if solve is None:
            raise TypeError(
                f"{type(self._implementation).__name__} 未实现 solve()")
        if x0 is None:
            return solve(rhs)
        return solve(rhs, x0=x0)

    def solve_result(self, rhs: Any, x0: Any = None) -> SolveResult:
        """Solve once and return the backend-neutral evidence envelope.

        Existing ``solve`` remains unchanged and continues to return only the
        backend field. Metrics that an implementation does not expose remain
        unknown instead of being inferred from unrelated configuration.
        """

        setup_start = perf_counter()
        self.setup()
        setup_seconds = perf_counter() - setup_start
        solve_start = perf_counter()
        result = self.solve(rhs, x0=x0)
        solve_seconds = perf_counter() - solve_start
        return compatibility_result(
            implementation=self._implementation,
            x=result,
            backend=self.backend,
            variant=self.variant,
            setup_seconds=setup_seconds,
            solve_seconds=solve_seconds,
        )

    def capabilities(self) -> BackendCapabilities:
        """Return explicit capabilities or a conservative empty declaration."""

        probe = getattr(self._implementation, "capabilities", None)
        if callable(probe):
            result = probe()
            if not isinstance(result, BackendCapabilities):
                raise TypeError(
                    "backend capabilities() must return BackendCapabilities")
            return result
        return BackendCapabilities(
            backend=self.backend,
            variant=self.variant,
        )

    def close(self) -> None:
        """Release hierarchy resources if the backend owns any."""
        close = getattr(self._implementation, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "Multigrid":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        if name == "_implementation":
            raise AttributeError(name)
        return getattr(self._implementation, name)


def _implementation_type(backend: str, variant: str):
    if backend == "torch":
        return StrictMultigrid if variant == "strict" else LegacyMultigrid
    module = import_module(f"pyqcu.{backend}.multigrid")
    return getattr(module, "StrictMultigrid" if variant == "strict" else
                   "LegacyMultigrid")


__all__ = [
    "LegacyMultigrid",
    "LegacyMultigridCache",
    "Multigrid",
    "MultigridCache",
    "MultigridCacheLoadResult",
    "StrictMultigrid",
    "StrictMultigridCache",
]
