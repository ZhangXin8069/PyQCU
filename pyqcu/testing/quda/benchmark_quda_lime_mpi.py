#!/usr/bin/env python3
"""Benchmark QUDA Clover gauge setup and BiCGStab on one distributed LIME input.

The same global gauge configuration and deterministic global right-hand side
are read/split for 1, 2, or 4 MPI ranks.  During runs, ``pyquda.mpi.init`` is
intentionally called before any local gauge slab is read.  This script does
not create Slurm jobs and must only be launched on a CUDA compute node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "pyqcu.quda-lime-mpi.v1"
LIME_MAGIC = b"\x45\x67\x89\xab"
DEFAULT_LAT = (24, 24, 24, 64)
DEFAULT_GRID = (1, 1, 1, 1)
DEFAULT_TOL = 1.0e-6
DEFAULT_MAX_ITER = 1000
DEFAULT_WARMUPS = 1
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42
RHS_MPI_TAG = 41021


class BenchmarkFailure(RuntimeError):
    """A fail-closed benchmark error with per-rank details."""

    def __init__(self, message: str, errors: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.errors = [dict(item) for item in errors]


@dataclass(frozen=True)
class LimeMetadata:
    path: Path
    lattice_xyzt: tuple[int, int, int, int]
    precision_bits: int
    binary_dtype: np.dtype
    binary_offset: int
    binary_length: int
    csw: float | None
    reference_mass: float | None
    checksum_present: bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gauge", required=True, help="global LIME/QIO gauge file")
    parser.add_argument(
        "--lat", nargs=4, type=_positive_int, default=DEFAULT_LAT,
        metavar=("X", "Y", "Z", "T"), help="global lattice dimensions",
    )
    parser.add_argument(
        "--grid", nargs=4, type=_positive_int, default=DEFAULT_GRID,
        metavar=("GX", "GY", "GZ", "GT"),
        help="4D MPI process grid; product must equal MPI_COMM_WORLD size",
    )
    parser.add_argument(
        "--mass", type=_finite_float, default=None,
        help="QCU/KUPL bare mass; defaults to LIME record XML Mass",
    )
    parser.add_argument(
        "--csw", type=_positive_float, default=None,
        help="Clover csw; defaults to LIME record XML clovCoeff",
    )
    parser.add_argument(
        "--dtype", choices=("c64", "c128"), default="c64",
        help="QUDA device precision; host QDP staging remains complex128",
    )
    parser.add_argument(
        "--solver", choices=("bicgstab", "mg"), default="bicgstab",
        help="plain BiCGStab reference or recursive multigrid",
    )
    parser.add_argument(
        "--levels", type=_positive_int, default=2,
        help="MG levels including the fine level",
    )
    parser.add_argument(
        "--nvec", type=_positive_int, default=6,
        help="MG near-null vectors; installed build supports 6,24,32",
    )
    parser.add_argument(
        "--mg-block", nargs=4, type=_positive_int, default=(2, 2, 2, 2),
        metavar=("BX", "BY", "BZ", "BT"), help="MG block size",
    )
    parser.add_argument("--tol", type=_positive_float, default=DEFAULT_TOL)
    parser.add_argument("--max-iter", type=_positive_int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--warmups", type=_nonnegative_int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--source", default=None,
        help="optional global RHS (.npy/.npz/.h5) in [spin,color,x,y,z,t]",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="seed for the deterministic global RHS when --source is absent",
    )
    parser.add_argument(
        "--report", default="quda_lime_mpi_report.json",
        help="JSON report path written by rank 0; use '-' for stdout only",
    )
    return parser


def _lime_records(path: Path) -> dict[str, tuple[int, int]]:
    records: dict[str, tuple[int, int]] = {}
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        while True:
            magic = handle.read(4)
            if not magic or magic.startswith((b"\n", b"\r\n")):
                break
            if magic.strip(b"\0 \t\r\n") == b"":
                break
            if magic != LIME_MAGIC:
                raise ValueError(f"not a LIME file at byte {handle.tell() - 4}: {path}")
            if len(handle.read(4)) != 4:
                raise ValueError("truncated LIME header")
            raw_length = handle.read(8)
            raw_name = handle.read(128)
            if len(raw_length) != 8 or len(raw_name) != 128:
                raise ValueError("truncated LIME record header")
            length = struct.unpack(">Q", raw_length)[0]
            name = raw_name.split(b"\0", 1)[0].decode("utf-8")
            offset = handle.tell()
            padded_length = ((length + 7) // 8) * 8
            if offset + padded_length > file_size:
                raise ValueError(f"LIME record {name!r} exceeds file size")
            records[name] = (offset, length)
            handle.seek(padded_length, os.SEEK_CUR)
    if not records:
        raise ValueError(f"LIME file contains no records: {path}")
    return records


def _lime_record_bytes(
    path: Path, records: Mapping[str, tuple[int, int]], name: str,
) -> bytes:
    try:
        offset, length = records[name]
    except KeyError as exc:
        raise ValueError(f"LIME file is missing required record {name!r}: {path}") from exc
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise ValueError(f"truncated LIME record {name!r}")
    return payload


def _xml_root(path: Path, records: Mapping[str, tuple[int, int]], name: str) -> ET.Element:
    payload = _lime_record_bytes(path, records, name).rstrip(b"\0 \t\r\n")
    try:
        return ET.fromstring(payload.decode("utf-8"))
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise ValueError(f"invalid XML in LIME record {name!r}") from exc


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_element(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _find_text(root: ET.Element, name: str) -> str | None:
    element = _find_element(root, name)
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text or None


def _find_float(root: ET.Element, name: str) -> float | None:
    text = _find_text(root, name)
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid floating-point {name}={text!r} in LIME XML") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} in LIME XML")
    return value


def _find_float_in_roots(roots: Sequence[ET.Element], name: str) -> float | None:
    for root in roots:
        value = _find_float(root, name)
        if value is not None:
            return value
    return None


def read_lime_metadata(path: str | os.PathLike[str]) -> LimeMetadata:
    """Read gauge dimensions, precision, clover coefficient, and reference mass."""
    input_path = Path(path).expanduser().resolve()
    records = _lime_records(input_path)

    ildg_root: ET.Element | None = None
    ildg_precision: str | None = None
    if "ildg-format" in records:
        ildg_root = _xml_root(input_path, records, "ildg-format")
        ildg_dims = tuple(
            int(_find_text(ildg_root, axis) or "0") for axis in ("lx", "ly", "lz", "lt")
        )
        ildg_precision = _find_text(ildg_root, "precision")

    if "scidac-private-file-xml" in records:
        file_root = _xml_root(input_path, records, "scidac-private-file-xml")
        dims_text = _find_text(file_root, "dims")
        if dims_text is None:
            raise ValueError("LIME private file XML has no <dims>")
        try:
            dims = tuple(int(item) for item in dims_text.split())
        except ValueError as exc:
            raise ValueError(f"invalid LIME <dims>: {dims_text!r}") from exc
    elif ildg_root is not None:
        dims = ildg_dims
    else:
        raise ValueError("LIME has neither private-file dimensions nor ildg-format")
    if len(dims) != 4 or any(item <= 0 for item in dims):
        raise ValueError(f"invalid LIME lattice dimensions: {dims}")
    if ildg_root is not None and ildg_dims != dims:
        raise ValueError(
            f"LIME dimension mismatch: private XML {dims}, ILDG XML {ildg_dims}"
        )

    record_name = next(
        (name for name in ("scidac-private-record-xml", "scidac-record-xml")
         if name in records),
        None,
    )
    if record_name is None:
        raise ValueError("LIME file has no SciDAC record XML for mass/csw metadata")
    record_root = _xml_root(input_path, records, record_name)
    metadata_roots = [record_root]
    other_record_name = (
        "scidac-record-xml"
        if record_name == "scidac-private-record-xml"
        else "scidac-private-record-xml"
    )
    if other_record_name in records:
        metadata_roots.append(_xml_root(input_path, records, other_record_name))

    ildg_is_gauge = ildg_root is not None and _find_text(ildg_root, "field") == "su3gauge"
    datacount = int(_find_text(record_root, "datacount") or ("4" if ildg_is_gauge else "0"))
    colors = int(_find_text(record_root, "colors") or ("3" if ildg_is_gauge else "0"))
    if datacount != 4:
        raise ValueError(f"LIME gauge has datacount={datacount}, expected 4")
    if colors != 3:
        raise ValueError(f"LIME gauge has colors={colors}, expected 3")

    record_precision = _find_text(record_root, "precision")
    precision_text = record_precision or ildg_precision
    if precision_text in ("D", "64"):
        precision_bits, dtype = 64, np.dtype(">c16")
    elif precision_text in ("F", "32"):
        precision_bits, dtype = 32, np.dtype(">c8")
    else:
        raise ValueError(f"unsupported LIME gauge precision: {precision_text!r}")

    try:
        binary_offset, binary_length = records["ildg-binary-data"]
    except KeyError as exc:
        raise ValueError("LIME file has no ildg-binary-data record") from exc
    expected_length = int(np.prod(dims, dtype=np.int64)) * 4 * 3 * 3 * dtype.itemsize
    if binary_length != expected_length:
        raise ValueError(
            f"LIME gauge payload size mismatch: {binary_length} != {expected_length}"
        )

    return LimeMetadata(
        path=input_path,
        lattice_xyzt=dims,
        precision_bits=precision_bits,
        binary_dtype=dtype,
        binary_offset=binary_offset,
        binary_length=binary_length,
        csw=_find_float_in_roots(metadata_roots, "clovCoeff"),
        reference_mass=_find_float_in_roots(metadata_roots, "Mass"),
        checksum_present="scidac-checksum" in records,
    )


def _process_coord(rank: int, grid: Sequence[int]) -> tuple[int, int, int, int]:
    _, gy, gz, gt = (int(value) for value in grid)
    return (
        rank // gt // gz // gy,
        rank // gt // gz % gy,
        rank // gt % gz,
        rank % gt,
    )


def _local_lattice(
    global_lattice: Sequence[int], grid: Sequence[int],
) -> tuple[int, int, int, int]:
    return tuple(
        int(length) // int(processes)
        for length, processes in zip(global_lattice, grid, strict=True)
    )


def _local_bounds(
    global_lattice: Sequence[int], grid: Sequence[int], coord: Sequence[int],
) -> tuple[slice, slice, slice, slice]:
    local = _local_lattice(global_lattice, grid)
    return tuple(
        slice(int(c) * size, (int(c) + 1) * size)
        for size, c in zip(local, coord, strict=True)
    )


def _validate_geometry(
    mpi_size: int, global_lattice: Sequence[int], grid: Sequence[int],
) -> tuple[int, int, int, int]:
    if len(global_lattice) != 4 or len(grid) != 4:
        raise ValueError("both --lat and --grid need four dimensions")
    if math.prod(grid) != mpi_size:
        raise ValueError(
            f"MPI rank count {mpi_size} does not match --grid product {math.prod(grid)}"
        )
    for axis, (length, processes) in enumerate(zip(global_lattice, grid, strict=True)):
        if length % processes:
            raise ValueError(
                f"global axis {axis} length {length} is not divisible by grid {processes}"
            )
    local = _local_lattice(global_lattice, grid)
    if any(extent <= 0 or extent % 2 for extent in local):
        raise ValueError(
            f"all local extents must be positive and even for parity layouts, got {local}"
        )
    return local


def read_lime_local_gauge(
    metadata: LimeMetadata,
    grid: Sequence[int],
    coord: Sequence[int],
) -> np.ndarray:
    """Read one global LIME memmap and copy only this rank's QDP local slab."""
    lx, ly, lz, lt = metadata.lattice_xyzt
    raw = np.memmap(
        metadata.path,
        dtype=metadata.binary_dtype,
        mode="r",
        offset=metadata.binary_offset,
        shape=(lt, lz, ly, lx, 4, 3, 3),
    )
    x_slice, y_slice, z_slice, t_slice = _local_bounds(
        metadata.lattice_xyzt, grid, coord,
    )
    slab = np.array(raw[t_slice, z_slice, y_slice, x_slice], copy=True)
    del raw
    return np.ascontiguousarray(
        slab.transpose(4, 0, 1, 2, 3, 5, 6), dtype=np.complex128,
    )


def _resolve_mass_and_csw(
    explicit_mass: float | None,
    explicit_csw: float | None,
    metadata: LimeMetadata,
) -> tuple[float, str, float, str]:
    if explicit_mass is not None:
        mass, mass_source = float(explicit_mass), "cli"
    elif metadata.reference_mass is not None:
        mass, mass_source = float(metadata.reference_mass), "lime"
    else:
        raise BenchmarkFailure(
            "LIME has no <Mass>; pass --mass explicitly rather than using a hidden default"
        )
    if explicit_csw is not None:
        csw, csw_source = float(explicit_csw), "cli"
    elif metadata.csw is not None:
        csw, csw_source = float(metadata.csw), "lime"
    else:
        raise BenchmarkFailure(
            "LIME has no <clovCoeff>; pass --csw explicitly rather than using a hidden default"
        )
    if not math.isfinite(mass) or mass <= -4.0:
        raise BenchmarkFailure(f"mass={mass} must be finite and greater than -4")
    if not math.isfinite(csw) or csw <= 0.0:
        raise BenchmarkFailure(f"csw={csw} must be finite and positive for a Clover benchmark")
    return mass, mass_source, csw, csw_source


def kappa_from_mass(mass: float) -> float:
    """QCU/KUPL bare-mass convention used by PyQuda's getDslash."""
    return 1.0 / (2.0 * (float(mass) + 4.0))


def median_and_mad(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.size == 0:
        raise ValueError("median/MAD requires at least one sample")
    if not np.isfinite(samples).all():
        raise ValueError("median/MAD samples must all be finite")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return {
        "count": int(samples.size),
        "samples": [float(item) for item in samples],
        "median_seconds": median,
        "mad_seconds": mad,
    }


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _field_dtype(dtype: str | np.dtype) -> np.dtype:
    name = str(dtype)
    if name in ("c64", "complex64"):
        return np.dtype(np.complex64)
    if name in ("c128", "complex128"):
        return np.dtype(np.complex128)
    raise ValueError(f"unsupported complex field dtype: {dtype!r}")


def _global_rhs_canonical(
    lattice: Sequence[int], seed: int, dtype: str | np.dtype = "c64",
) -> np.ndarray:
    shape = (4, 3, *(int(item) for item in lattice))
    rng = np.random.default_rng(int(seed))
    # Match the float64 RNG stream used by the QCU/KUPL LIME benchmarks.
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return np.ascontiguousarray(real + 1j * imag, dtype=_field_dtype(dtype))


def _read_global_rhs(
    path: Path, lattice: Sequence[int], dtype: str | np.dtype,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(f"{path} has no 'data' array")
            array = np.asarray(archive["data"])
    elif suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for --source HDF5 input") from exc
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"{path} has no /data dataset")
            array = np.asarray(handle["data"])
    else:
        raise ValueError("--source must be .npy, .npz, .h5, or .hdf5")
    expected_shape = (4, 3, *(int(item) for item in lattice))
    if array.shape != expected_shape:
        raise ValueError(f"source shape {array.shape} does not match {expected_shape}")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"source dtype must be complex, got {array.dtype}")
    array = np.ascontiguousarray(array, dtype=_field_dtype(dtype))
    if not np.isfinite(array).all():
        raise ValueError("source contains non-finite values")
    return array


def slice_global_rhs(
    global_rhs: np.ndarray,
    lattice: Sequence[int],
    grid: Sequence[int],
    coord: Sequence[int],
) -> np.ndarray:
    if global_rhs.shape != (4, 3, *(int(item) for item in lattice)):
        raise ValueError("global RHS shape does not match --lat")
    x_slice, y_slice, z_slice, t_slice = _local_bounds(lattice, grid, coord)
    return np.ascontiguousarray(
        global_rhs[:, :, x_slice, y_slice, z_slice, t_slice], dtype=np.complex64,
    )


def _canonical_fermion_to_qdp(local_rhs: np.ndarray) -> np.ndarray:
    if local_rhs.ndim != 6 or local_rhs.shape[:2] != (4, 3):
        raise ValueError("local RHS must have shape [spin,color,x,y,z,t]")
    return np.ascontiguousarray(
        local_rhs.transpose(5, 4, 3, 2, 0, 1), dtype=np.complex128,
    )


def _broadcast_root_result(
    comm: Any,
    label: str,
    function: Callable[[], Any],
) -> Any:
    payload: tuple[bool, Any] | None
    if comm.Get_rank() == 0:
        try:
            payload = (True, function())
        except Exception:
            payload = (False, traceback.format_exc())
    else:
        payload = None
    ok, value = comm.bcast(payload, root=0)
    if not ok:
        raise BenchmarkFailure(f"{label} failed on rank 0:\n{value}")
    return value


def _distribute_global_rhs(
    comm: Any,
    lattice: Sequence[int],
    grid: Sequence[int],
    coord: Sequence[int],
    source_path: str | None,
    seed: int,
    dtype: str | np.dtype,
) -> tuple[np.ndarray, dict[str, Any]]:
    def prepare_root() -> tuple[np.ndarray, dict[str, Any]]:
        if source_path is None:
            global_rhs = _global_rhs_canonical(lattice, seed, dtype)
            metadata = {
                "kind": "seeded-global",
                "seed": int(seed),
                "dtype": str(_field_dtype(dtype)),
                "global_shape": list(global_rhs.shape),
                "global_sha256": _sha256_array(global_rhs),
            }
        else:
            path = Path(source_path).expanduser().resolve()
            global_rhs = _read_global_rhs(path, lattice, dtype)
            metadata = {
                "kind": "file",
                "path": str(path),
                "dtype": str(_field_dtype(dtype)),
                "global_shape": list(global_rhs.shape),
                "global_sha256": _sha256_array(global_rhs),
                "file_sha256": _sha256_file(path),
            }
        return global_rhs, metadata

    root_result: tuple[np.ndarray, dict[str, Any]] | None
    error: str | None
    if comm.Get_rank() == 0:
        try:
            root_result = prepare_root()
            error = None
        except Exception:
            root_result = None
            error = traceback.format_exc()
    else:
        root_result, error = None, None
    shared = comm.bcast(
        {
            "ok": error is None,
            "error": error,
            "metadata": None if root_result is None else root_result[1],
        },
        root=0,
    )
    if not shared["ok"]:
        raise BenchmarkFailure(f"global RHS preparation failed on rank 0:\n{shared['error']}")

    local_shape = (4, 3, *_local_lattice(lattice, grid))
    if comm.Get_rank() == 0:
        assert root_result is not None
        global_rhs = root_result[0]
        local_rhs = slice_global_rhs(global_rhs, lattice, grid, coord)
        for destination in range(1, comm.Get_size()):
            destination_coord = _process_coord(destination, grid)
            send_slab = slice_global_rhs(
                global_rhs, lattice, grid, destination_coord,
            )
            comm.Send(np.ascontiguousarray(send_slab), dest=destination, tag=RHS_MPI_TAG)
        del global_rhs
    else:
        local_rhs = np.empty(local_shape, dtype=_field_dtype(dtype))
        comm.Recv(local_rhs, source=0, tag=RHS_MPI_TAG)
    return np.ascontiguousarray(local_rhs), dict(shared["metadata"])


def _configure_cuda_device(comm: Any) -> dict[str, Any]:
    """Set the rank-local CUDA device before the first ``import pyquda``."""
    from mpi4py import MPI
    import cupy as cp
    # PyQuda's pointer helper lazily imports torch while constructing host
    # gauge pointers.  Loading torch after libquda can crash in the Torch
    # protobuf initializers, so pin the shared-library order here.
    import torch  # noqa: F401

    shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = int(shared_comm.Get_rank())
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count <= 0:
        raise BenchmarkFailure("CUDA reports no visible devices")
    effective_device = local_rank
    mps_enabled = os.environ.get("QUDA_ENABLE_MPS") == "1"
    if effective_device >= device_count:
        raise BenchmarkFailure(
            f"local rank {local_rank} exceeds {device_count} visible CUDA devices; "
            "strong-scaling runs require one visible GPU per rank"
        )
    # This must remain before ``import pyquda`` in _load_pyquda_modules().
    cp.cuda.Device(effective_device).use()
    return {
        "local_rank": local_rank,
        "device": effective_device,
        "visible_device_count": device_count,
        "mps_enabled": mps_enabled,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _gpu_info(cp: Any, device: int) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(int(device))
    name = properties.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    uuid = properties.get("uuid")
    if isinstance(uuid, bytes):
        uuid = uuid.hex()
    return {
        "device": int(device),
        "name": str(name),
        "compute_capability": f"{properties.get('major')}.{properties.get('minor')}",
        "multiprocessor_count": int(properties.get("multiProcessorCount", 0)),
        "total_global_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "uuid": uuid,
    }


def _load_pyquda_modules(
) -> tuple[Any, Any, Any, Any, Any, Any, Any, str]:
    # Device selection in _configure_cuda_device() has already happened.
    import pyquda
    try:
        import pyquda_utils.core as core
        api = "pyquda_utils"
    except ImportError:
        import pyquda.core as core
        api = "pyquda"
    from pyquda.enum_quda import (
        QudaDslashType,
        QudaInverterType,
        QudaMassNormalization,
        QudaPrecision,
        QudaResidualType,
        QudaSolveType,
        QudaUseInitGuess,
    )
    from pyquda.field import LatticeFermion, LatticeGauge, cb2

    enums = {
        "QudaDslashType": QudaDslashType,
        "QudaInverterType": QudaInverterType,
        "QudaMassNormalization": QudaMassNormalization,
        "QudaPrecision": QudaPrecision,
        "QudaResidualType": QudaResidualType,
        "QudaSolveType": QudaSolveType,
        "QudaUseInitGuess": QudaUseInitGuess,
    }
    try:
        from pyquda.enum_quda import QudaTune
    except ImportError:
        pass
    else:
        enums["QudaTune"] = QudaTune

    return (
        pyquda,
        core,
        LatticeFermion,
        LatticeGauge,
        cb2,
        enums,
        pyquda if api == "pyquda_utils" else pyquda.mpi,
        api,
    )


def _activate_pyquda(
    pyquda: Any,
    pyquda_mpi: Any,
    grid: Sequence[int],
    expected_device: int,
    lattice: Sequence[int],
    api: str,
) -> None:
    if api == "pyquda_utils":
        if not hasattr(pyquda_mpi, "init"):
            raise BenchmarkFailure("PyQuda API mismatch: pyquda.init is unavailable")
        os.environ["LOCAL_RANK"] = str(int(expected_device))
        pyquda_mpi.init(
            grid_size=[int(item) for item in grid],
            latt_size=[int(item) for item in lattice],
            backend="torch",
            backend_target="cuda",
            enable_nvshmem=False,
            enable_tuning=False,
            enable_device_memory_pool=False,
            enable_pinned_memory_pool=False,
        )
        if tuple(int(item) for item in pyquda.getGridSize()) != tuple(grid):
            raise BenchmarkFailure(
                f"PyQuda grid {pyquda.getGridSize()} does not match requested "
                f"{tuple(grid)}"
            )
        import torch
        if int(torch.cuda.current_device()) != int(expected_device):
            raise BenchmarkFailure(
                f"PyQuda selected device {torch.cuda.current_device()}, "
                f"expected {expected_device}"
            )
        return

    if not hasattr(pyquda, "mpi") or not hasattr(pyquda.mpi, "init"):
        raise BenchmarkFailure("PyQuda API mismatch: pyquda.mpi.init is unavailable")
    # All LIME/local-gauge I/O in this script occurs after this call.
    pyquda_mpi.init(list(int(item) for item in grid))
    if int(pyquda_mpi.gpuid) != int(expected_device):
        raise BenchmarkFailure(
            f"PyQuda selected device {pyquda_mpi.gpuid}, expected {expected_device}"
        )
    if tuple(pyquda_mpi.grid) != tuple(int(item) for item in grid):
        raise BenchmarkFailure(
            f"PyQuda grid {pyquda_mpi.grid} does not match requested {tuple(grid)}"
        )


def _loaded_libquda_path() -> Path:
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) != 6:
                continue
            candidate = fields[5].replace("\\040", " ")
            if candidate.endswith(" (deleted)"):
                continue
            name = Path(candidate).name
            if name == "libquda.so" or name.startswith("libquda.so."):
                return Path(candidate).resolve()
    candidates: list[Path] = []
    for variable in ("QUDA_INSTALL", "QUDA_BUILD_DIR"):
        raw = os.environ.get(variable)
        if not raw:
            continue
        root = Path(raw).expanduser()
        candidates.extend((
            root / "lib" / "libquda.so",
            root / "lib64" / "libquda.so",
            root / "libquda.so",
            root / "usqcd" / "lib" / "libquda.so",
        ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BenchmarkFailure(
        "cannot identify the loaded libquda.so; set QUDA_INSTALL or QUDA_BUILD_DIR"
    )


def _verify_injected_quda(loaded: Path) -> dict[str, Any]:
    resolved_loaded = loaded.resolve()
    injected_candidates: list[Path] = []
    for variable in ("QUDA_INSTALL", "QUDA_BUILD_DIR"):
        raw = os.environ.get(variable)
        if not raw:
            continue
        root = Path(raw).expanduser()
        injected_candidates.extend((
            root / "lib" / "libquda.so",
            root / "lib64" / "libquda.so",
            root / "libquda.so",
            root / "usqcd" / "lib" / "libquda.so",
        ))
    existing = {path.resolve() for path in injected_candidates if path.is_file()}
    if existing and resolved_loaded not in existing:
        raise BenchmarkFailure(
            f"loaded libquda {resolved_loaded} does not match injected candidates "
            f"{sorted(str(path) for path in existing)}"
        )
    return {
        "path": str(resolved_loaded),
        "size_bytes": resolved_loaded.stat().st_size,
    }


def _build_fields(
    metadata: LimeMetadata,
    local_rhs: np.ndarray,
    grid: Sequence[int],
    coord: Sequence[int],
    LatticeGauge: Any,
    LatticeFermion: Any,
    cb2: Any,
    core: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    global_qdp_gauge = read_lime_local_gauge(metadata, grid, coord)
    local_lattice = _local_lattice(metadata.lattice_xyzt, grid)
    qdp_rhs = _canonical_fermion_to_qdp(local_rhs)
    if hasattr(core, "LatticeInfo"):
        import torch
        info = core.LatticeInfo(list(metadata.lattice_xyzt), 1, 1.0)
        parity_gauge = info.evenodd(global_qdp_gauge.astype(np.complex128), True)
        parity_rhs = info.evenodd(qdp_rhs.astype(np.complex128), False)
        gauge = LatticeGauge(
            info, 4, torch.from_numpy(parity_gauge).to("cuda"))
        rhs = LatticeFermion(
            info, torch.from_numpy(parity_rhs).to("cuda"))
    else:
        parity_gauge = cb2(global_qdp_gauge, [1, 2, 3, 4])
        gauge = LatticeGauge(
            list(local_lattice), parity_gauge,
            t_boundary=int(coord[3]) == int(grid[3]) - 1,
        )
        parity_rhs = cb2(qdp_rhs, [0, 1, 2, 3])
        rhs = LatticeFermion(list(local_lattice), parity_rhs)
    rhs.toDevice()

    return gauge, rhs, {
        "qdp_gauge_shape": list(global_qdp_gauge.shape),
        "parity_gauge_shape": list(gauge.data.shape),
        "qdp_rhs_shape": list(qdp_rhs.shape),
        "parity_rhs_shape": list(rhs.data.shape),
    }


def _enum_name(value: Any, enum_type: Any = None) -> str:
    if enum_type is not None:
        try:
            return enum_type(int(value)).name
        except (TypeError, ValueError):
            pass
    return getattr(value, "name", str(value))


def _build_dirac(
    core: Any,
    enum_quda: Mapping[str, Any],
    local_lattice: Sequence[int],
    mass: float,
    csw: float,
    tol: float,
    max_iter: int,
    dtype: str,
    solver: str,
    levels: int,
    nvec: int,
    mg_block: Sequence[int],
    global_lattice: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    if not hasattr(core, "getDslash"):
        raise BenchmarkFailure("PyQuda API mismatch: pyquda.core.getDslash is unavailable")
    modern_api = hasattr(core, "LatticeInfo")
    if modern_api:
        from pyquda.dirac import general as dirac_general
    else:
        from pyquda.dslash import general as dirac_general
    from pyquda.enum_quda import QudaPrecision

    device_precision = (
        QudaPrecision.QUDA_SINGLE_PRECISION
        if dtype == "c64" else QudaPrecision.QUDA_DOUBLE_PRECISION
    )
    multigrid = None
    if solver == "mg":
        if levels < 2:
            raise BenchmarkFailure("MG requires at least two levels")
        multigrid = [list(int(item) for item in mg_block)
                     for _ in range(levels - 1)]
    quda_lattice = global_lattice if modern_api else local_lattice
    dirac = core.getDslash(
        list(int(item) for item in quda_lattice),
        float(mass),
        float(tol),
        int(max_iter),
        clover_coeff_t=float(csw),
        anti_periodic_t=False,
        multigrid=False if multigrid is None else multigrid,
    )
    if modern_api:
        dirac.setPrecision(
            cuda=device_precision,
            sloppy=device_precision,
            refinement_sloppy=device_precision,
            precondition=device_precision,
            eigensolver=device_precision,
        )
    else:
        for name in (
            "cuda_prec",
            "cuda_prec_sloppy",
            "cuda_prec_precondition",
            "cuda_prec_eigensolver",
        ):
            setattr(dirac_general, name, device_precision)
    invert = dirac.invert_param
    invert.mass_normalization = enum_quda["QudaMassNormalization"].QUDA_KAPPA_NORMALIZATION
    if solver == "bicgstab":
        invert.inv_type = enum_quda["QudaInverterType"].QUDA_BICGSTAB_INVERTER
    invert.solve_type = enum_quda["QudaSolveType"].QUDA_DIRECT_PC_SOLVE
    invert.use_init_guess = enum_quda["QudaUseInitGuess"].QUDA_USE_INIT_GUESS_NO
    invert.compute_true_res = 1
    invert.tol = float(tol)
    invert.maxiter = int(max_iter)
    tune_enum = enum_quda.get("QudaTune")
    invert_tune_exposed = hasattr(invert, "tune")
    if invert_tune_exposed:
        if tune_enum is None:
            raise BenchmarkFailure(
                "PyQuda exposes QudaInvertParam.tune without QudaTune")
        invert.tune = tune_enum.QUDA_TUNE_NO
    gauge_tune_exposed = hasattr(dirac.gauge_param, "tune")
    if gauge_tune_exposed:
        if tune_enum is None:
            raise BenchmarkFailure(
                "PyQuda exposes QudaGaugeParam.tune without QudaTune")
        dirac.gauge_param.tune = tune_enum.QUDA_TUNE_NO

    mg_param = getattr(dirac, "mg_param", None)
    if mg_param is None:
        mg_param = getattr(getattr(dirac, "multigrid", None), "param", None)
    if mg_param is None:
        mg_param = getattr(getattr(dirac, "mg", None), "param", None)
    nvec_applied = False
    nvec_resolved: list[int] | None = None
    if solver == "mg":
        if mg_param is None:
            raise BenchmarkFailure("MG Dirac has no QudaMultigridParam")
        current_nvec = list(getattr(mg_param, "n_vec"))
        if not current_nvec:
            raise BenchmarkFailure("QudaMultigridParam.n_vec is empty")
        mg_param.n_vec = [int(nvec)] * len(current_nvec)
        nvec_resolved = [int(value) for value in mg_param.n_vec]
        nvec_applied = all(value == int(nvec) for value in nvec_resolved)
        if not nvec_applied:
            raise BenchmarkFailure(
                f"QUDA n_vec assignment did not stick: {nvec_resolved}")

    expected_kappa = kappa_from_mass(mass)
    expected_clover_coeff = expected_kappa * float(csw)
    checks = {
        "dslash_is_clover": invert.dslash_type == enum_quda["QudaDslashType"].QUDA_CLOVER_WILSON_DSLASH,
        "solve_type_is_direct_pc": invert.solve_type == enum_quda["QudaSolveType"].QUDA_DIRECT_PC_SOLVE,
        "mass_normalization_is_kappa": (
            invert.mass_normalization
            == enum_quda["QudaMassNormalization"].QUDA_KAPPA_NORMALIZATION
        ),
        "kappa_matches_qcu": math.isclose(
            float(invert.kappa), expected_kappa, rel_tol=1.0e-12, abs_tol=1.0e-15,
        ),
        "clover_coeff_matches_kappa_csw": math.isclose(
            float(invert.clover_coeff), expected_clover_coeff,
            rel_tol=1.0e-12, abs_tol=1.0e-15,
        ),
        "cuda_precision_matches_request": (
            int(invert.cuda_prec) == int(device_precision)),
        "cuda_sloppy_precision_matches_request": (
            int(invert.cuda_prec_sloppy) == int(device_precision)),
        "cuda_refinement_precision_matches_request": (
            int(invert.cuda_prec_refinement_sloppy)
            == int(device_precision)),
        "cuda_precondition_precision_matches_request": (
            int(invert.cuda_prec_precondition) == int(device_precision)),
        "cuda_eigensolver_precision_matches_request": (
            int(invert.cuda_prec_eigensolver) == int(device_precision)),
        "environment_tuning_disabled": (
            os.environ.get("QUDA_ENABLE_TUNING") == "0"),
    }
    if invert_tune_exposed:
        checks["invert_tuning_disabled"] = (
            invert.tune == tune_enum.QUDA_TUNE_NO)
    if gauge_tune_exposed:
        checks["gauge_tuning_disabled"] = (
            dirac.gauge_param.tune == tune_enum.QUDA_TUNE_NO)
    if solver == "bicgstab":
        checks["inverter_is_bicgstab"] = (
            invert.inv_type ==
            enum_quda["QudaInverterType"].QUDA_BICGSTAB_INVERTER)
    else:
        checks["multigrid_attached"] = mg_param is not None
        checks["nvec_applied"] = nvec_applied
    if not all(checks.values()):
        raise BenchmarkFailure(f"PyQuda/QUDA parameter contract failed: {checks}")
    config = {
        **checks,
        "dslash_type": _enum_name(
            invert.dslash_type, enum_quda["QudaDslashType"]),
        "inverter_type": _enum_name(
            invert.inv_type, enum_quda["QudaInverterType"]),
        "solve_type": _enum_name(
            invert.solve_type, enum_quda["QudaSolveType"]),
        "mass_normalization": _enum_name(
            invert.mass_normalization,
            enum_quda["QudaMassNormalization"]),
        "residual_type": _enum_name(
            invert.residual_type, enum_quda["QudaResidualType"]),
        "compute_true_res": int(invert.compute_true_res),
        "tuning_enabled": False,
        "invert_tune": (
            _enum_name(invert.tune, tune_enum)
            if invert_tune_exposed else "not_exposed_by_pyquda"),
        "gauge_tune": (
            _enum_name(dirac.gauge_param.tune, tune_enum)
            if gauge_tune_exposed else "not_exposed_by_pyquda"),
        "invert_tune_field_exposed": invert_tune_exposed,
        "gauge_tune_field_exposed": gauge_tune_exposed,
        "quda_enable_tuning_environment": os.environ.get(
            "QUDA_ENABLE_TUNING"),
        "solver": solver,
        "levels": int(levels),
        "nvec_requested": int(nvec),
        "nvec_resolved": nvec_resolved,
        "mg_block": [int(item) for item in mg_block],
        "tol": float(invert.tol),
        "maxiter": int(invert.maxiter),
        "requested_dtype": dtype,
        "qcu_mass": float(mass),
        "pyquda_getdslash_mass": float(mass),
        "kappa": float(invert.kappa),
        "quda_clover_coeff": float(invert.clover_coeff),
        "cpu_precision": _enum_name(invert.cpu_prec, enum_quda["QudaPrecision"]),
        "cuda_precision": _enum_name(
            invert.cuda_prec, enum_quda["QudaPrecision"]),
        "cuda_sloppy_precision": _enum_name(
            invert.cuda_prec_sloppy, enum_quda["QudaPrecision"]),
        "cuda_refinement_sloppy_precision": _enum_name(
            invert.cuda_prec_refinement_sloppy,
            enum_quda["QudaPrecision"]),
        "cuda_precondition_precision": _enum_name(
            invert.cuda_prec_precondition,
            enum_quda["QudaPrecision"]),
        "cuda_eigensolver_precision": _enum_name(
            invert.cuda_prec_eigensolver,
            enum_quda["QudaPrecision"]),
    }
    return dirac, config


def _synchronize_gpu(cp: Any) -> None:
    cp.cuda.runtime.deviceSynchronize()


def _timed_gpu_operation(
    comm: Any, cp: Any, operation: Callable[[], Any],
) -> tuple[Any, float]:
    comm.Barrier()
    _synchronize_gpu(cp)
    started = time.perf_counter()
    try:
        result = operation()
        _synchronize_gpu(cp)
    except Exception:
        try:
            _synchronize_gpu(cp)
        except Exception:
            pass
        raise
    return result, time.perf_counter() - started


def _collective_guard(
    comm: Any, stage: str, operation: Callable[[], Any],
) -> Any:
    result: Any = None
    local_error: str | None = None
    try:
        result = operation()
    except Exception:
        local_error = traceback.format_exc()
    errors = comm.allgather({
        "rank": int(comm.Get_rank()),
        "stage": stage,
        "error": local_error,
    })
    failures = [item for item in errors if item["error"] is not None]
    if failures:
        raise BenchmarkFailure(f"stage {stage!r} failed", failures)
    return result


def _solve_record(
    invert: Any,
    wall_seconds: float,
    kappa_normalization: Any,
) -> dict[str, Any]:
    def scalar(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            return value
        try:
            return value[0]
        except (IndexError, KeyError, TypeError):
            return value

    iterations = int(scalar(invert.iter))
    max_iter = int(scalar(invert.maxiter))
    solver_seconds = float(scalar(invert.secs))
    true_res = float(scalar(invert.true_res))
    gflops = float(scalar(invert.gflops))
    if int(scalar(invert.mass_normalization)) != int(scalar(kappa_normalization)):
        raise BenchmarkFailure(
            "mass normalization changed during solve: "
            f"{int(invert.mass_normalization)} != {int(kappa_normalization)}"
        )
    if int(scalar(invert.compute_true_res)) != 1:
        raise BenchmarkFailure("QUDA compute_true_res was not enabled during solve")
    if iterations <= 0:
        raise BenchmarkFailure(f"QUDA returned non-positive iteration count: {iterations}")
    if iterations > max_iter:
        raise BenchmarkFailure(
            f"QUDA iteration count {iterations} exceeds max_iter={max_iter}"
        )
    if not math.isfinite(true_res) or true_res < 0.0:
        raise BenchmarkFailure(f"QUDA returned invalid true_res: {true_res}")
    if not math.isfinite(solver_seconds) or solver_seconds < 0.0:
        raise BenchmarkFailure(f"QUDA returned invalid solver seconds: {solver_seconds}")
    if not math.isfinite(gflops) or gflops < 0.0:
        raise BenchmarkFailure(f"QUDA returned invalid GFLOP/s: {gflops}")
    return {
        "wall_seconds": float(wall_seconds),
        "quda_solver_seconds": solver_seconds,
        "iterations": iterations,
        "true_res": true_res,
        "gflops": gflops,
    }


def _metadata_json(metadata: LimeMetadata) -> dict[str, Any]:
    return {
        "path": str(metadata.path),
        "lattice_xyzt": list(metadata.lattice_xyzt),
        "precision_bits": metadata.precision_bits,
        "binary_dtype": metadata.binary_dtype.str,
        "binary_length": metadata.binary_length,
        "checksum_present": metadata.checksum_present,
        "lime_csw": metadata.csw,
        "lime_reference_mass": metadata.reference_mass,
    }


def _config_payload(
    args: argparse.Namespace,
    metadata: LimeMetadata,
    mpi_size: int,
    mass: float,
    mass_source: str,
    csw: float,
    csw_source: str,
) -> dict[str, Any]:
    local_lattice = _validate_geometry(mpi_size, args.lat, args.grid)
    kappa = kappa_from_mass(mass)
    return {
        "gauge": str(metadata.path),
        "global_lattice_xyzt": list(args.lat),
        "process_grid_xyzt": list(args.grid),
        "local_lattice_xyzt": list(local_lattice),
        "mpi_rank_count": int(mpi_size),
        "mass": float(mass),
        "mass_source": mass_source,
        "mass_semantics": "QCU/KUPL bare mass in kappa = 1 / (2 * (mass + 4))",
        "mass_normalization": "QUDA_KAPPA_NORMALIZATION",
        "dtype": args.dtype,
        "kappa": kappa,
        "pyquda_solution_postscale": 2.0 * kappa,
        "pyquda_solution_postscale_formula": "2 * kappa = 1 / (mass + 4)",
        "csw": float(csw),
        "csw_source": csw_source,
        "quda_clover_coeff": kappa * float(csw),
        "quda_clover_coeff_formula": "kappa * csw",
        "onsite_coefficient_formula": "-kappa * csw / 8",
        "tol": float(args.tol),
        "max_iter": int(args.max_iter),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
    }


def _phase_samples(rank_records: Sequence[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
    per_rank = [list(record[phase]) for record in rank_records]
    counts = {len(samples) for samples in per_rank}
    if len(counts) != 1:
        raise BenchmarkFailure(f"rank sample counts differ for phase {phase}: {sorted(counts)}")
    count = counts.pop()
    samples: list[dict[str, Any]] = []
    for index in range(count):
        rank_samples = [samples_for_rank[index] for samples_for_rank in per_rank]
        wall_seconds = [float(sample["wall_seconds"]) for sample in rank_samples]
        true_res = [float(sample["true_res"]) for sample in rank_samples]
        iterations = [int(sample["iterations"]) for sample in rank_samples]
        solver_seconds = [float(sample["quda_solver_seconds"]) for sample in rank_samples]
        samples.append({
            "sample": index,
            "max_wall_seconds": max(wall_seconds),
            "wall_seconds_by_rank": wall_seconds,
            "quda_solver_seconds_by_rank": solver_seconds,
            "iterations_by_rank": iterations,
            "true_res_by_rank": true_res,
            "true_res_max": max(true_res),
        })
    return samples


def _summarize_phases(
    rank_records: Sequence[Mapping[str, Any]], tol: float,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    setup_seconds = [float(record["gauge_load_setup_seconds"]) for record in rank_records]
    setup_summary = median_and_mad(setup_seconds)
    setup_summary["max_seconds"] = max(setup_seconds)
    setup_summary["seconds_by_rank"] = setup_seconds

    warmups = _phase_samples(rank_records, "warmups")
    steady = _phase_samples(rank_records, "steady")
    if not steady:
        raise BenchmarkFailure("steady phase contains no samples")

    warmup_max = [sample["max_wall_seconds"] for sample in warmups]
    steady_max = [sample["max_wall_seconds"] for sample in steady]
    warmup_wall = median_and_mad(warmup_max) if warmup_max else None
    steady_wall = median_and_mad(steady_max)

    for phase, samples in (("warmup", warmups), ("steady", steady)):
        for sample in samples:
            if sample["true_res_max"] > tol:
                errors.append(
                    f"{phase} sample {sample['sample']} true_res "
                    f"{sample['true_res_max']:.9g} exceeds tol {tol:.9g}"
                )
            if min(sample["iterations_by_rank"]) <= 0:
                errors.append(f"{phase} sample {sample['sample']} has non-positive iterations")

    return {
        "gauge_load_setup": {
            **setup_summary,
            "timing_scope": "CloverWilson.loadGauge including clover construction/load",
        },
        "warmups": {
            "distributed_wall": warmup_wall,
            "samples": warmups,
        },
        "steady": {
            "distributed_wall": steady_wall,
            "samples": steady,
        },
        "timing_semantics": {
            "clock": "time.perf_counter with CUDA device synchronize",
            "mpi_aggregation": "maximum rank wall time per sample, then median/MAD",
            "mad_definition": "median(abs(sample - median(sample)))",
            "warmup_excluded_from_steady": True,
        },
    }, errors


def _rank_json_fingerprint(rank_records: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            record["process_grid_xyzt"],
            record["local_lattice_xyzt"],
            record["libquda"]["sha256"],
            record["parameter_contract"]["mass_normalization"],
            record["parameter_contract"]["kappa"],
            record["parameter_contract"]["quda_clover_coeff"],
        )
        for record in rank_records
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(path_value: str, report: Mapping[str, Any]) -> None:
    payload = json.dumps(
        _json_safe(report), indent=2, ensure_ascii=False, allow_nan=False,
    )
    if path_value == "-":
        print(payload)
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_benchmark(
    args: argparse.Namespace,
    comm: Any,
) -> tuple[int, dict[str, Any]]:
    # QUDA reads this process-global switch before its tuning singleton is
    # initialized.  Set it before importing PyQuda on every rank.
    os.environ["QUDA_ENABLE_TUNING"] = "0"
    rank = int(comm.Get_rank())
    mpi_size = int(comm.Get_size())
    grid = tuple(int(item) for item in args.grid)
    lattice = tuple(int(item) for item in args.lat)
    _validate_geometry(mpi_size, lattice, grid)
    coord = _process_coord(rank, grid)

    device_record = _collective_guard(
        comm, "select_cuda_device", lambda: _configure_cuda_device(comm),
    )
    modules = _collective_guard(comm, "import_pyquda", _load_pyquda_modules)
    (pyquda, core, LatticeFermion, LatticeGauge, cb2, enum_quda,
     pyquda_mpi, pyquda_api) = modules
    _collective_guard(
        comm,
        "initialize_pyquda_mpi",
        lambda: _activate_pyquda(
            pyquda, pyquda_mpi, grid, int(device_record["device"]),
            lattice, pyquda_api,
        ),
    )
    import cupy as cp

    gpu = _collective_guard(
        comm, "read_gpu_info", lambda: _gpu_info(cp, int(device_record["device"])),
    )
    libquda_identity = _collective_guard(
        comm, "fingerprint_libquda",
        lambda: _verify_injected_quda(_loaded_libquda_path()),
    )
    libquda_paths = comm.allgather(libquda_identity["path"])
    if len(set(libquda_paths)) != 1:
        raise BenchmarkFailure(f"ranks loaded libquda from different paths: {libquda_paths}")
    libquda_sizes = comm.allgather(libquda_identity["size_bytes"])
    if len(set(libquda_sizes)) != 1:
        raise BenchmarkFailure(f"ranks loaded different libquda sizes: {libquda_sizes}")
    libquda_sha256 = _broadcast_root_result(
        comm, "hash_libquda", lambda: _sha256_file(libquda_identity["path"]),
    )
    libquda = {**libquda_identity, "sha256": libquda_sha256}
    metadata = _collective_guard(
        comm, "read_lime_metadata",
        lambda: read_lime_metadata(args.gauge),
    )
    if metadata.lattice_xyzt != lattice:
        raise BenchmarkFailure(
            f"LIME lattice {metadata.lattice_xyzt} does not match --lat {lattice}"
        )
    mass, mass_source, csw, csw_source = _resolve_mass_and_csw(
        args.mass, args.csw, metadata,
    )
    config = _config_payload(
        args, metadata, mpi_size, mass, mass_source, csw, csw_source,
    )
    configs = comm.allgather(config)
    if any(item != configs[0] for item in configs[1:]):
        raise BenchmarkFailure("rank configurations differ")

    # The file is hashed once on rank 0, outside all timed phases.
    gauge_sha256 = _broadcast_root_result(
        comm, "hash_lime_gauge", lambda: _sha256_file(metadata.path),
    )
    local_rhs, source_metadata = _collective_guard(
        comm,
        "prepare_global_rhs",
        lambda: _distribute_global_rhs(
            comm, lattice, grid, coord, args.source, args.seed, args.dtype,
        ),
    )
    gauge, rhs, layout = _collective_guard(
        comm,
        "build_local_fields",
        lambda: _build_fields(
            metadata, local_rhs, grid, coord,
            LatticeGauge, LatticeFermion, cb2, core,
        ),
    )
    del local_rhs
    dirac, parameter_contract = _collective_guard(
        comm,
        "construct_clover_bicgstab",
        lambda: _build_dirac(
            core, enum_quda, _local_lattice(lattice, grid),
            mass, csw, args.tol, args.max_iter, args.dtype, args.solver,
            args.levels, args.nvec, args.mg_block, lattice,
        ),
    )

    _, setup_seconds = _collective_guard(
        comm,
        "gauge_load_setup",
        lambda: _timed_gpu_operation(comm, cp, lambda: dirac.loadGauge(gauge)),
    )

    warmups: list[dict[str, Any]] = []
    for _ in range(int(args.warmups)):
        def warmup_once() -> dict[str, Any]:
            solution, seconds = _timed_gpu_operation(comm, cp, lambda: dirac.invert(rhs))
            del solution
            return _solve_record(
                dirac.invert_param, seconds,
                enum_quda["QudaMassNormalization"].QUDA_KAPPA_NORMALIZATION)

        warmups.append(_collective_guard(comm, "warmup_solve", warmup_once))

    steady: list[dict[str, Any]] = []
    for _ in range(int(args.repeats)):
        def steady_once() -> dict[str, Any]:
            solution, seconds = _timed_gpu_operation(comm, cp, lambda: dirac.invert(rhs))
            del solution
            return _solve_record(
                dirac.invert_param, seconds,
                enum_quda["QudaMassNormalization"].QUDA_KAPPA_NORMALIZATION)

        steady.append(_collective_guard(comm, "steady_solve", steady_once))

    local_record = {
        "rank": rank,
        "coord_xyzt": list(coord),
        "process_grid_xyzt": list(grid),
        "local_lattice_xyzt": list(_local_lattice(lattice, grid)),
        "host": platform.node(),
        "device_selection": device_record,
        "gpu": gpu,
        "libquda": libquda,
        "layout": layout,
        "parameter_contract": parameter_contract,
        "gauge_load_setup_seconds": float(setup_seconds),
        "warmups": warmups,
        "steady": steady,
    }
    rank_records = comm.gather(local_record, root=0)

    def finalize_root() -> tuple[int, dict[str, Any]]:
        records = [dict(item) for item in rank_records if item is not None]
        if len(records) != mpi_size:
            raise BenchmarkFailure(
                f"received {len(records)} rank records, expected {mpi_size}"
            )
        hashes = {record["libquda"]["sha256"] for record in records}
        if len(hashes) != 1:
            raise BenchmarkFailure(f"ranks loaded different libquda SHA256 values: {hashes}")
        fingerprints = _rank_json_fingerprint(records)
        if any(item != fingerprints[0] for item in fingerprints[1:]):
            raise BenchmarkFailure("rank parameter/layout contracts differ")
        phases, residual_errors = _summarize_phases(records, float(args.tol))
        status = "PASS" if not residual_errors else "FAIL"
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "errors": residual_errors,
            "config": config,
            "input": {
                "lime": _metadata_json(metadata),
                "lime_sha256": gauge_sha256,
                "source": source_metadata,
            },
            "environment": {
                "python": sys.version,
                "pyquda_module": str(getattr(pyquda, "__file__", None)),
                "quda_install": os.environ.get("QUDA_INSTALL"),
                "quda_build_dir": os.environ.get("QUDA_BUILD_DIR"),
                "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "libquda": {
                "path": records[0]["libquda"]["path"],
                "sha256": records[0]["libquda"]["sha256"],
                "size_bytes": records[0]["libquda"]["size_bytes"],
            },
            "mpi": {
                "rank_count": mpi_size,
                "process_grid_xyzt": list(grid),
                "ranks": records,
            },
            "phases": phases,
            "solver": {
                "implementation": (
                    f"{pyquda_api}.core.getDslash -> Dirac.invert"),
                "inverter": records[0]["parameter_contract"][
                    "inverter_type"],
                "true_res_source": "QudaInvertParam.true_res (compute_true_res=1)",
                "true_res_tolerance": float(args.tol),
            },
            "fairness": {
                "same_lime_sha256": True,
                "same_rhs_sha256": True,
                "same_libquda_sha256": True,
                "same_mass_csw_and_normalization": True,
                "all_reported_solves_converged": not residual_errors,
            },
        }
        _write_report(args.report, report)
        return (0 if status == "PASS" else 1), report

    payload: dict[str, Any] | None
    if rank == 0:
        try:
            status, report = finalize_root()
            payload = {"ok": True, "status": status, "report": report}
        except Exception:
            payload = {"ok": False, "error": traceback.format_exc()}
    else:
        payload = None
    shared = comm.bcast(payload, root=0)
    if not shared["ok"]:
        raise BenchmarkFailure(f"rank 0 finalization failed:\n{shared['error']}")
    if rank == 0 and args.report != "-":
        print(json.dumps(_json_safe(shared["report"]), ensure_ascii=False))
    elif rank != 0:
        print(json.dumps({
            "rank": rank,
            "status": shared["report"]["status"],
            "grid": list(grid),
            "local_shape_xyzt": list(_local_lattice(lattice, grid)),
        }))
    return int(shared["status"]), dict(shared["report"])


def _failure_report(
    args: argparse.Namespace,
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "errors": list(errors),
        "config": {
            "gauge": str(Path(args.gauge).expanduser()),
            "global_lattice_xyzt": list(args.lat),
            "process_grid_xyzt": list(args.grid),
            "mass": args.mass,
            "csw": args.csw,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from mpi4py import MPI
    except ImportError as exc:
        print(f"ERROR: mpi4py is required to launch this benchmark: {exc}", file=sys.stderr)
        return 1

    comm = MPI.COMM_WORLD
    try:
        status, _ = _run_benchmark(args, comm)
        return status
    except KeyboardInterrupt:
        print(f"ERROR: interrupted on rank {comm.Get_rank()}", file=sys.stderr)
        return 130
    except Exception as exc:
        local_error = {
            "rank": int(comm.Get_rank()),
            "stage": "main",
            "error": traceback.format_exc(),
        }
        try:
            gathered = comm.allgather(local_error)
        except Exception:
            gathered = [local_error]
        if isinstance(exc, BenchmarkFailure) and exc.errors:
            gathered = list(exc.errors)
        if comm.Get_rank() == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
            try:
                _write_report(args.report, _failure_report(args, gathered))
            except Exception as report_error:
                print(f"ERROR: cannot write failure report: {report_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
