"""KUPL CPU backend Python bindings."""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {}


def __getattr__(name):
    if name in {"qcu", "define"}:
        if name not in _LAZY_EXPORTS:
            value = importlib.import_module(f"{__name__}.{name}")
            _LAZY_EXPORTS[name] = value
        return _LAZY_EXPORTS[name]
    raise AttributeError(f"module 'pyqcu.kupl' has no attribute {name!r}")


__all__ = ["define", "qcu"]
