"""Validated build-backend names shared by packaging and tests."""

from __future__ import annotations

SUPPORTED_BUILD_BACKENDS = frozenset({"cuda", "dcu", "kupl"})
GPU_BUILD_BACKENDS = frozenset({"cuda", "dcu"})
DEFAULT_BUILD_BACKEND = "cuda"


def normalize_build_backend(value: str | None = None) -> str:
    """Return a canonical build-backend name or fail with the valid choices."""
    name = str(value or DEFAULT_BUILD_BACKEND).strip().lower()
    if name not in SUPPORTED_BUILD_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_BUILD_BACKENDS))
        raise ValueError(
            f"PYQCU_BUILD_BACKEND must be one of {choices}, got {name!r}")
    return name


__all__ = [
    "DEFAULT_BUILD_BACKEND",
    "GPU_BUILD_BACKENDS",
    "SUPPORTED_BUILD_BACKENDS",
    "normalize_build_backend",
]
