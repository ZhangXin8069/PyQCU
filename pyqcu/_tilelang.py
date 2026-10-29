"""Lazy optional TileLang detection with one process-wide warning."""

from __future__ import annotations

import warnings
from threading import Lock
from types import ModuleType
from typing import Optional

_TILELANG: Optional[ModuleType] = None
_CHECKED = False
_LOCK = Lock()


class TileLangUnavailable(RuntimeError):
    """Raised only when a caller explicitly requests a TileLang-only path."""


def import_tilelang() -> Optional[ModuleType]:
    """Return TileLang when available, otherwise warn once and return None."""
    global _CHECKED, _TILELANG
    if _CHECKED:
        return _TILELANG
    with _LOCK:
        if _CHECKED:
            return _TILELANG
        _CHECKED = True
        try:
            import tilelang  # type: ignore[import-not-found]
        except Exception as error:
            warnings.warn(
                "TileLang is unavailable; TileLang-specific acceleration and "
                f"tests will be skipped ({type(error).__name__}: {error})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        _TILELANG = tilelang
    return _TILELANG


def tilelang_available() -> bool:
    """Return whether TileLang can be imported."""
    return import_tilelang() is not None


__all__ = ["TileLangUnavailable", "import_tilelang", "tilelang_available"]
