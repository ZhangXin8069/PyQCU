"""Lazy optional TileLang detection with one process-wide warning.

TileLang is an optional dependency: ``pyqcu`` imports and runs without it by
falling back to ``torch.einsum`` (``Eexyzt_exyzt2Exyzt``) or by raising
:class:`TileLangUnavailable` from the TileLang-only program factories
(``matmul_gpu``/``matmul_cpu``).

Warning contract
----------------
* The probe runs at most once per process and its outcome (``_TILELANG`` /
  ``_CHECKED``, plus the captured import error) is cached, so every caller
  sees the same answer and a silent probe followed by an eager one still
  reports the original import error.
* Import-time probes never warn.  A module that touches the optional
  dependency while it is *being imported* -- ``pyqcu.tools._define`` with
  ``warn=False``, ``pyqcu.torch.tools._einsum``/``_matul`` with the default --
  is a probe, not a request, so ``import pyqcu`` and every
  ``python -m pyqcu.test`` run that pulls in the torch tools stay quiet.
* The first *explicit* request made from running code -- ``import_tilelang()``
  called after import, e.g. from a test or a ``__main__`` entry point --
  emits exactly one ``RuntimeWarning`` and latches ``_WARNED`` so no later
  request warns again.  A silent probe does not consume that one warning.
* ``tilelang_available()`` never warns: it is a predicate, not a request.
"""

from __future__ import annotations

import sys
import warnings
from threading import Lock
from types import ModuleType
from typing import Optional

_TILELANG: Optional[ModuleType] = None
_CHECKED = False
_WARNED = False
_ERROR: Optional[str] = None
_LOCK = Lock()


class TileLangUnavailable(RuntimeError):
    """Raised only when a caller explicitly requests a TileLang-only path."""


def _probe() -> Optional[ModuleType]:
    """Import TileLang at most once per process; never warns."""
    global _CHECKED, _ERROR, _TILELANG
    if _CHECKED:
        return _TILELANG
    with _LOCK:
        if _CHECKED:
            return _TILELANG
        try:
            import tilelang  # type: ignore[import-not-found]
        except Exception as error:  # any failure means "unavailable"
            _ERROR = f"{type(error).__name__}: {error}"
            _TILELANG = None
        else:
            _ERROR = None
            _TILELANG = tilelang
        _CHECKED = True
    return _TILELANG


def _at_import_time() -> bool:
    """Whether the caller is module-level code of a module being imported.

    ``pyqcu.tools._define`` and ``pyqcu.torch.tools._einsum``/``_matul`` touch
    TileLang at import time, which must stay silent.  Module-level code of an
    *imported* module is identified by ``co_name == "<module>"`` plus a
    ``__spec__`` (the import system assigns one); code that merely runs at top
    level, such as a ``python -c``/``python -m`` script or ``runpy.run_path``
    (``__spec__ is None``), is a real request and still warns.
    """
    try:
        frame = sys._getframe(1)
        own_globals = globals()
        while frame is not None and frame.f_globals is own_globals:
            frame = frame.f_back  # skip this module's own helper frames
    except (AttributeError, ValueError):  # pragma: no cover - exotic runtime
        return False
    if frame is None or frame.f_code.co_name != "<module>":
        return False
    return getattr(frame.f_globals.get("__spec__"), "name", None) is not None


def _claim_warning() -> bool:
    """Reserve the single process-wide warning; ``True`` for the winner."""
    global _WARNED
    with _LOCK:
        if _WARNED:
            return False
        _WARNED = True
        return True


def import_tilelang(*, warn: bool = True) -> Optional[ModuleType]:
    """Return TileLang when available, otherwise return ``None``.

    Parameters
    ----------
    warn
        ``True`` (default) requests the one-time diagnostic when TileLang
        cannot be imported; it is emitted at most once per process
        (``_WARNED``) and never for a call made while a module is being
        imported, so import-time probes stay silent.  ``False`` is the pure
        probe used by ``pyqcu.tools._define`` and :func:`tilelang_available`:
        it never warns and never consumes the one warning.

    Notes
    -----
    The import result is cached, so a silent probe followed by an explicit
    probe still reports the cached failure -- and still warns exactly once.
    """
    module = _probe()
    if module is not None or not warn:
        return module
    if _at_import_time():
        return None
    if _claim_warning():
        warnings.warn(
            "TileLang is unavailable; TileLang-specific acceleration and "
            f"tests will be skipped ({_ERROR})",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def tilelang_available() -> bool:
    """Return whether TileLang can be imported, without a warning."""
    return import_tilelang(warn=False) is not None


__all__ = ["TileLangUnavailable", "import_tilelang", "tilelang_available"]
