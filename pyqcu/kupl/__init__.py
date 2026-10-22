"""KUPL CPU backend Python bindings."""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {}


def __getattr__(name):
    if name in {"qcu", "define", "multigrid"}:
        if name not in _LAZY_EXPORTS:
            value = importlib.import_module(f"{__name__}.{name}")
            _LAZY_EXPORTS[name] = value
        return _LAZY_EXPORTS[name]
    if name in {"LegacyMultigrid", "StrictMultigrid"}:
        if name not in _LAZY_EXPORTS:
            module = importlib.import_module(f"{__name__}.multigrid")
            _LAZY_EXPORTS[name] = getattr(module, name)
        return _LAZY_EXPORTS[name]
    raise AttributeError(f"module 'pyqcu.kupl' has no attribute {name!r}")


__all__ = ["LegacyMultigrid", "StrictMultigrid", "define", "multigrid", "qcu"]
