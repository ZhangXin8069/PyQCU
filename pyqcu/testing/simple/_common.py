"""Minimal shared helpers for the QUDA/QCU examples.

The file formats are intentionally explicit:

* QCU gauge: HDF5 ``[color_out,color_in,dir,x,y,z,t]``.
* QUDA/QDP gauge: ``[dir,t,z,y,x,color_out,color_in]`` in a NumPy file,
  or the same array packed in a single-file SciDAC/LIME ``.lime`` file.
* QCU fermion: HDF5 ``[spin,color,x,y,z,t]``.
* QUDA fermion: parity-compressed ``[q,t,z,y,x/2,spin,color]``.
* QUDA clover: packed real ``[q,t,z,y,x/2,chirality,36]``.

The helpers stay NumPy-only so conversion scripts do not initialize CUDA or
either solver library.
"""

from __future__ import annotations

import ctypes
import os
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

try:
    from ._lime import lime_metadata, read_lime_gauge, write_lime_gauge
    from ._layouts import (
        kupl_fermion_to_qcu,
        kupl_gauge_to_qcu,
        qcu_clover_to_quda,
        qcu_fermion_to_kupl,
        qcu_fermion_to_qdp,
        qcu_gauge_to_kupl,
        qcu_gauge_to_qdp,
        qdp_fermion_to_qcu,
        qdp_gauge_to_qcu,
        quda_clover_to_qcu,
    )
except ImportError:  # direct script execution from this directory
    from _lime import lime_metadata, read_lime_gauge, write_lime_gauge
    from _layouts import (
        kupl_fermion_to_qcu,
        kupl_gauge_to_qcu,
        qcu_clover_to_quda,
        qcu_fermion_to_kupl,
        qcu_fermion_to_qdp,
        qcu_gauge_to_kupl,
        qcu_gauge_to_qdp,
        qdp_fermion_to_qcu,
        qdp_gauge_to_qcu,
        quda_clover_to_qcu,
    )


REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("PYQCU_DATA_DIR", str(REPO / "data"))).expanduser()

DEFAULT_LIME = Path(os.environ.get(
    "PYQCU_REFERENCE_LIME",
    str(DATA_DIR / "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime"))).expanduser()
DEFAULT_MASS = 0.05
DEFAULT_CSW = 1.160920226
DEFAULT_ATOL = 1.0e-6
DEFAULT_MAX_ITER = 1000
DEFAULT_NVEC = 24
DEFAULT_BLOCK = (2, 2, 2, 2)

_QUDA_LIBRARY_HANDLE = None


def configure_quda_runtime(*, preload: bool = True) -> Path | None:
    """Select a repository or explicitly configured QUDA installation.

    PyQUDA's extension has a stale RUNPATH, while some hosts expose an old
    QUDA build through LD_LIBRARY_PATH. Loading the selected library by its
    absolute path before importing PyQUDA pins the intended ABI and GPU
    architecture.
    """
    global _QUDA_LIBRARY_HANDLE
    candidates = []
    if os.environ.get("QUDA_INSTALL"):
        candidates.append(Path(os.environ["QUDA_INSTALL"]).expanduser())
    candidates.extend((
        REPO / "data/quda-sm70-install",
        REPO / "data/quda-double-install",
    ))
    selected = None
    for candidate in candidates:
        library = candidate / "lib" / "libquda.so"
        if library.is_file():
            selected = candidate.resolve()
            break
    if selected is None:
        return None

    os.environ["QUDA_INSTALL"] = str(selected)
    os.environ["QUDA_PATH"] = str(selected)
    search_path = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in search_path.split(":") if entry]
    library_dir = str(selected / "lib")
    if library_dir not in entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join([library_dir, *entries])
    if preload and _QUDA_LIBRARY_HANDLE is None:
        _QUDA_LIBRARY_HANDLE = ctypes.CDLL(
            str(selected / "lib" / "libquda.so"), mode=ctypes.RTLD_GLOBAL)
    return selected


def resolve_csw(explicit: float | None,
                metadata: Mapping[str, Any] | None = None) -> float:
    """Resolve PyQUDA's ``csw`` precedence: CLI, file metadata, default."""
    if explicit is not None:
        value = float(explicit)
    elif metadata is not None and metadata.get("csw") is not None:
        value = float(metadata["csw"])
    else:
        value = float(DEFAULT_CSW)
    if not np.isfinite(value):
        raise ValueError("csw must be finite")
    return value


def csw_matches(left: float, right: float | None, *,
                rtol: float = 2.0e-7) -> bool:
    """Compare clover coefficients with a tolerance suitable for LIME XML."""
    return right is None or bool(np.isclose(float(left), float(right), rtol=rtol,
                                            atol=0.0))


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve a CLI path relative to the repository data directory."""
    path = Path(value).expanduser() if value is not None else Path(default)
    if path.is_absolute():
        return path
    # Accept both ``--input data/foo`` from the repository root and the
    # documented ``--input foo`` shorthand rooted at ``data/``.
    if path.parts and path.parts[0] == "data":
        return (REPO / path).resolve()
    if path.exists():
        return path.resolve()
    return DATA_DIR / path


def _attrs(path: Path) -> dict[str, Any]:
    return {
        "format": "pyqcu.simple.v1",
        "source_path": str(path),
    }


def read_h5(path: str | os.PathLike[str], dataset: str = "data") -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset][...])


def read_h5_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read HDF5 attributes, decoding array-valued attributes (e.g. lattice)."""
    metadata: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            array = np.asarray(value)
            metadata[key] = (array.tolist() if array.ndim and array.size > 1
                             else array.item())
    return metadata


def write_h5(
    path: str | os.PathLike[str],
    array: np.ndarray,
    *,
    dataset: str = "data",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attributes = _attrs(path)
    if metadata:
        attributes.update(metadata)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=np.asarray(array))
        for key, value in attributes.items():
            if value is not None:
                handle.attrs[key] = value


def read_array(path: str | os.PathLike[str], dataset: str = "data") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            return np.asarray(handle[dataset])
    return read_h5(path, dataset)


def write_array(
    path: str | os.PathLike[str],
    array: np.ndarray,
    *,
    dataset: str = "data",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array), allow_pickle=False)
    elif path.suffix == ".npz":
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {dataset: np.asarray(array)}
        if metadata:
            payload["metadata"] = np.asarray(
                [str({key: str(value) for key, value in metadata.items()})])
        np.savez_compressed(path, **payload)
    else:
        write_h5(path, array, dataset=dataset, metadata=metadata)


def json_safe(value: Any, *, omit_none: bool = False) -> Any:
    """Convert a report to JSON-native values, optionally dropping nulls."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            converted = json_safe(item, omit_none=omit_none)
            if omit_none and converted is None:
                continue
            result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        converted = [json_safe(item, omit_none=omit_none) for item in value]
        if omit_none:
            converted = [item for item in converted if item is not None]
        return converted
    return value


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any],
               *, omit_none: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
                        json_safe(payload, omit_none=omit_none), indent=2,
                        ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def cuda_memory_snapshot(device: str | int = 0) -> dict[str, Any]:
    """Return process-local CUDA allocation and device-wide free-memory data."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    dev = torch.device(device if isinstance(device, int) else str(device))
    torch.cuda.synchronize(dev)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    return {
        "available": True,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev),
        "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
        "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def estimated_clover_gflops(lattice: tuple[int, int, int, int], iterations: int,
                             wall_seconds: float,
                             applications_per_iteration: int = 2,
                             operator: str = "schur") -> dict[str, Any]:
    """Estimate operator throughput from native iteration count and wall time.

    ``schur`` counts the two bare hops and two onsite 12x12 products on the
    two parity halves used by each preconditioned Clover application.  A
    direct parity dslash counts one bare hop and one onsite product on half
    the lattice.  Both are converted to a full-lattice equivalent.  This is a
    documented FLOP model, not a hardware counter.
    """
    sites = int(np.prod(lattice))
    if operator == "schur":
        flops_per_site = (288 + 144) * 8
    elif operator == "dslash":
        flops_per_site = (288 + 144) * 8 // 2
    else:
        raise ValueError(f"unsupported Clover operator model: {operator}")
    flops_per_matvec = sites * flops_per_site
    applications = max(0, int(iterations)) * int(applications_per_iteration)
    flops = applications * flops_per_matvec
    return {
        "model": ("parity Schur Wilson-Clover, complex MAC=8 flop"
                  if operator == "schur" else
                  "parity Wilson-Clover dslash, complex MAC=8 flop"),
        "operator": operator,
        "flops_per_full_site": int(flops_per_site),
        "applications_per_iteration": int(applications_per_iteration),
        "operator_applications": applications,
        "estimated_flops": int(flops),
        "estimated_gflops": float(flops / wall_seconds / 1.0e9) if wall_seconds > 0 else None,
    }


def solver_acceptance(
        atol: float, *, dtype: np.dtype = np.dtype(np.complex64)) -> float:
    """Return the relative residual acceptance used by the simple examples."""
    real_dtype = np.empty((), dtype=dtype).real.dtype
    multiplier = 2048 if real_dtype.itemsize <= 4 else 128
    return max(float(atol), float(np.finfo(real_dtype).eps) * multiplier)


def convergence_status(
        residual: float, atol: float, *, comparison_valid: bool = True,
        dtype: np.dtype = np.dtype(np.complex64)) -> tuple[bool, str, float]:
    """Classify a relative true residual without hard-coded 1e-2 success."""
    acceptance = solver_acceptance(atol, dtype=dtype)
    converged = bool(np.isfinite(residual) and residual <= acceptance)
    if not converged:
        return False, "FAIL", acceptance
    return True, "PASS" if comparison_valid else "WARN", acceptance


def read_qdp_gauge(path: str | os.PathLike[str]) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix == ".lime":
        return read_lime_gauge(path)
    array = read_array(path)
    if array.ndim != 7 or array.shape[0] != 4 or array.shape[-2:] != (3, 3):
        raise ValueError("QUDA gauge must be [dir,t,z,y,x,3,3]")
    return np.asarray(array)


def write_qdp_gauge(path: str | os.PathLike[str], gauge: np.ndarray,
                    metadata: Mapping[str, Any] | None = None) -> None:
    path = Path(path).expanduser()
    if path.suffix == ".lime":
        write_lime_gauge(path, gauge, metadata)
        return
    if gauge.ndim != 7 or gauge.shape[0] != 4 or gauge.shape[-2:] != (3, 3):
        raise ValueError("QUDA gauge must be [dir,t,z,y,x,3,3]")
    write_array(path, np.ascontiguousarray(gauge), metadata=metadata)
