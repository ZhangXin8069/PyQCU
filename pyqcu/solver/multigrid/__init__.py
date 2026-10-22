"""Unified multigrid entry point and hierarchy implementations.

``Multigrid`` is the backend-neutral selector.  The concrete hierarchy lives
in ``StrictMultigrid`` or ``LegacyMultigrid`` and is selected explicitly with
``variant=...``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .legacy_multigrid import LegacyMultigrid
from .strict_multigrid import StrictMultigrid


class Multigrid:
    """Backend-neutral wrapper around a strict or legacy hierarchy."""

    _BACKENDS = {"torch", "cuda", "kupl"}
    _VARIANTS = {"strict", "legacy"}

    def __init__(self, *, variant: str, backend: str = "torch", **kwargs: Any):
        variant_name = str(variant).strip().lower().replace("-", "_")
        backend_name = str(backend).strip().lower()
        if variant_name not in self._VARIANTS:
            raise ValueError(
                f"variant 必须是 strict 或 legacy，得到 {variant!r}")
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
        result = getattr(self._implementation, "setup", None)
        if result is None:
            raise TypeError(
                f"{type(self._implementation).__name__} 未实现 setup()")
        returned = result()
        return self if returned is None else self

    def solve(self, rhs: Any, x0: Any = None) -> Any:
        """Solve ``D x = rhs`` with the selected hierarchy."""
        solver = getattr(self._implementation, "solve", None)
        if solver is None:
            raise TypeError(
                f"{type(self._implementation).__name__} 未实现 solve()")
        if x0 is None:
            return solver(rhs)
        return solver(rhs, x0=x0)

    def close(self) -> None:
        """Release hierarchy resources if the backend owns any."""
        closer = getattr(self._implementation, "close", None)
        if closer is not None:
            closer()

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


__all__ = ["LegacyMultigrid", "Multigrid", "StrictMultigrid"]
