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

import os
import struct
import xml.etree.ElementTree as ET
import zlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("PYQCU_DATA_DIR", str(REPO / "data"))).expanduser()

DEFAULT_LIME = DATA_DIR / "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime"
DEFAULT_MASS = -0.2770
DEFAULT_CSW = 1.160920226
DEFAULT_ATOL = 1.0e-6
DEFAULT_MAX_ITER = 1000
DEFAULT_NVEC = 24
DEFAULT_BLOCK = (2, 2, 2, 2)


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve a CLI path relative to the repository data directory."""
    path = Path(value).expanduser() if value is not None else Path(default)
    if path.is_absolute():
        return path
    # Accept both ``--input data/foo`` from the repository root and the
    # documented ``--input foo`` shorthand rooted at ``data/``.
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
    with h5py.File(path, "r") as handle:
        return {key: value.item() if hasattr(value, "item") else value
                for key, value in handle.attrs.items()}


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


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
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
                             applications_per_iteration: int = 2) -> dict[str, Any]:
    """Estimate operator throughput from native iteration count and wall time.

    The model counts one Clover-Wilson matvec as 4 directions x 12 x 12
    complex multiply-add plus one onsite 12 x 12 complex multiply-add.
    It is an order-of-magnitude estimate, not a hardware counter.
    """
    sites = int(np.prod(lattice))
    flops_per_matvec = sites * (4 * 12 * 12 * 8 + 12 * 12 * 8)
    applications = max(0, int(iterations)) * int(applications_per_iteration)
    flops = applications * flops_per_matvec
    return {
        "model": "4-direction Wilson hopping + onsite Clover, complex MAC=8 flop",
        "applications_per_iteration": int(applications_per_iteration),
        "operator_applications": applications,
        "estimated_flops": int(flops),
        "estimated_gflops": float(flops / wall_seconds / 1.0e9) if wall_seconds > 0 else None,
    }


def _lime_records(path: Path) -> dict[str, tuple[int, int]]:
    """Read the LIME record directory without loading the binary payload."""
    records: dict[str, tuple[int, int]] = {}
    with path.open("rb") as handle:
        while True:
            magic = handle.read(4)
            if not magic or magic == b"\n":
                break
            if magic != b"\x45\x67\x89\xab":
                raise ValueError(f"not a LIME file: {path}")
            handle.read(4)  # version and flags
            length = struct.unpack(">Q", handle.read(8))[0]
            name = handle.read(128).split(b"\0", 1)[0].decode("utf-8")
            offset = handle.tell()
            records[name] = (offset, length)
            handle.seek(offset + ((length + 7) // 8) * 8)
    return records


def _lime_xml(path: Path, records: Mapping[str, tuple[int, int]], name: str) -> ET.Element:
    offset, length = records[name]
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length).rstrip(b"\0")
    return ET.fromstring(payload.decode("utf-8"))


def lime_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return gauge dimensions and the Chroma clover coefficient if present."""
    path = Path(path).expanduser()
    records = _lime_records(path)
    file_xml = _lime_xml(path, records, "scidac-private-file-xml")
    dims = tuple(int(value) for value in file_xml.findtext("dims", "").split())
    if len(dims) != 4:
        raise ValueError(f"invalid LIME lattice dimensions in {path}: {dims}")
    meta: dict[str, Any] = {"lattice_xyzt": dims, "file": str(path)}
    if "scidac-record-xml" in records:
        record_xml = _lime_xml(path, records, "scidac-record-xml")
        coefficient = record_xml.find(".//clovCoeff")
        if coefficient is not None and coefficient.text:
            meta["csw"] = float(coefficient.text)
    return meta


def read_lime_gauge(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a single-file USQCD/QIO gauge field as QDP ``[d,t,z,y,x,3,3]``."""
    path = Path(path).expanduser()
    records = _lime_records(path)
    file_xml = _lime_xml(path, records, "scidac-private-file-xml")
    record_xml = _lime_xml(path, records, "scidac-private-record-xml")
    dims = tuple(int(value) for value in file_xml.findtext("dims", "").split())
    if len(dims) != 4:
        raise ValueError(f"invalid LIME lattice dimensions: {dims}")
    if int(record_xml.findtext("datacount", "0")) != 4:
        raise ValueError("only four-link gauge fields are supported")
    if int(record_xml.findtext("colors", "0")) != 3:
        raise ValueError("only Nc=3 gauge fields are supported")
    precision = record_xml.findtext("precision", "D")
    dtype = np.dtype(">c16" if precision == "D" else ">c8")
    offset, length = records["ildg-binary-data"]
    lx, ly, lz, lt = dims
    expected = lx * ly * lz * lt * 4 * 3 * 3 * dtype.itemsize
    if length != expected:
        raise ValueError(f"LIME payload size mismatch: {length} != {expected}")
    raw = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                    shape=(lt, lz, ly, lx, 4, 3, 3))
    gauge = np.ascontiguousarray(raw.transpose(4, 0, 1, 2, 3, 5, 6),
                                 dtype=np.complex64 if precision == "F" else np.complex128)
    if "scidac-checksum" in records:
        checksum = _lime_xml(path, records, "scidac-checksum")
        expected = (int(checksum.findtext("suma", "0"), 16),
                    int(checksum.findtext("sumb", "0"), 16))
        if _lime_checksum(gauge) != expected:
            raise ValueError(f"SciDAC checksum mismatch in {path}")
    return gauge


def _lime_record(handle: Any, name: str, payload: bytes) -> None:
    handle.write(b"\x45\x67\x89\xab")
    handle.write(struct.pack(">HHQ", 1, 0, len(payload)))
    encoded = name.encode("ascii")
    if len(encoded) > 128:
        raise ValueError(f"LIME record name too long: {name}")
    handle.write(encoded.ljust(128, b"\0"))
    handle.write(payload)
    handle.write(b"\0" * ((-len(payload)) % 8))


def _lime_checksum(gauge: np.ndarray) -> tuple[int, int]:
    """SciDAC checksum for a single-file gauge field, one rank."""
    # SciDAC checksums operate on the big-endian bytes actually stored by QIO.
    raw = np.ascontiguousarray(np.transpose(gauge, (1, 2, 3, 4, 0, 5, 6)),
                               dtype=">c16")
    rows = raw.reshape(-1, 36).view(np.uint8)
    work = np.empty(rows.shape[0], dtype="<u4")
    for index in range(rows.shape[0]):
        work[index] = zlib.crc32(rows[index])
    rank = np.arange(rows.shape[0], dtype="<u4")
    result = []
    for modulus in (29, 31):
        shift = rank % modulus
        rotated = np.where(shift == 0, work, (work << shift) | (work >> (32 - shift)))
        result.append(int(np.bitwise_xor.reduce(rotated)))
    return result[0], result[1]


def write_lime_gauge(path: str | os.PathLike[str], gauge: np.ndarray,
                     metadata: Mapping[str, Any] | None = None) -> None:
    """Write a single-file SciDAC/LIME gauge field with a verified checksum.

    The layout is the one accepted by ``read_lime_gauge`` and PyQUDA's
    ``readQIOGauge``: the binary record is ``[t,z,y,x,dir,row,col]`` with
    big-endian complex128 values.  This intentionally covers the single-rank
    data files used by this directory; it is not an MPI/QIO part-file writer.
    """
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[0] != 4 or array.shape[-2:] != (3, 3):
        raise ValueError("QDP gauge must be [dir,t,z,y,x,3,3]")
    _, t, z, y, x, _, _ = array.shape
    if min(t, z, y, x) <= 0:
        raise ValueError("gauge dimensions must be positive")
    csum = _lime_checksum(array)
    private_file = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<scidacFile><version>1.1</version><spacetime>4</spacetime>'
        f'<dims>{x} {y} {z} {t} </dims><volfmt>0</volfmt></scidacFile>'
    ).encode("utf-8")
    private_record = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<scidacRecord><version>1.1</version><recordtype>0</recordtype>'
        '<datatype>QDP_D3_ColorMatrix</datatype><precision>D</precision>'
        '<colors>3</colors><spins>4</spins><typesize>144</typesize>'
        '<datacount>4</datacount></scidacRecord>'
    ).encode("utf-8")
    checksum = (
        '<scidacChecksum><version>1.0</version>'
        f'<suma>{csum[0]:08x}</suma><sumb>{csum[1]:08x}</sumb></scidacChecksum>'
    ).encode("utf-8")
    ildg_format = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ildgFormat xmlns="http://www.lqcd.org/ildg"><version>1.0</version>'
        '<field>su3gauge</field><precision>64</precision>'
        f'<lx>{x}</lx><ly>{y}</ly><lz>{z}</lz><lt>{t}</lt></ildgFormat>'
    ).encode("utf-8")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.ascontiguousarray(np.transpose(array, (1, 2, 3, 4, 0, 5, 6)),
                               dtype=">c16")
    with path.open("wb") as handle:
        _lime_record(handle, "scidac-private-file-xml", private_file)
        _lime_record(handle, "scidac-private-record-xml", private_record)
        _lime_record(handle, "ildg-format", ildg_format)
        _lime_record(handle, "ildg-binary-data", raw.tobytes(order="C"))
        _lime_record(handle, "scidac-checksum", checksum)


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


def qcu_gauge_to_qdp(gauge: np.ndarray) -> np.ndarray:
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[:3] != (3, 3, 4):
        raise ValueError("QCU gauge must be [color,color,dir,x,y,z,t]")
    return np.ascontiguousarray(array.transpose(2, 6, 5, 4, 3, 0, 1))


def qdp_gauge_to_qcu(gauge: np.ndarray) -> np.ndarray:
    array = np.asarray(gauge)
    if array.ndim != 7 or array.shape[0] != 4 or array.shape[-2:] != (3, 3):
        raise ValueError("QDP gauge must be [dir,t,z,y,x,color,color]")
    return np.ascontiguousarray(array.transpose(5, 6, 0, 4, 3, 2, 1))


def qcu_fermion_to_qdp(fermion: np.ndarray) -> np.ndarray:
    """Convert full QCU ``[s,c,x,y,z,t]`` to compressed QDP ``[q,t,z,y,x/2,s,c]``."""
    array = np.asarray(fermion)
    if array.ndim != 6 or array.shape[:2] != (4, 3):
        raise ValueError("QCU fermion must be [spin,color,x,y,z,t]")
    _, _, x, y, z, t = array.shape
    source = array.transpose(5, 4, 3, 2, 0, 1)  # t,z,y,x,s,c
    rows = np.empty((2, t, z, y, x // 2), dtype=np.int64)
    tt = np.arange(t)[:, None, None, None]
    zz = np.arange(z)[None, :, None, None]
    yy = np.arange(y)[None, None, :, None]
    for q in range(2):
        rows[q] = ((((tt * z + zz) * y + yy) * x) +
                   ((q - (tt + zz + yy)) & 1) + 2 * np.arange(x // 2)[None, None, None, :])
    return np.ascontiguousarray(source.reshape(-1, 12)[rows].reshape(2, t, z, y, x // 2, 4, 3))


def qdp_fermion_to_qcu(fermion: np.ndarray) -> np.ndarray:
    """Convert compressed QDP ``[q,t,z,y,x/2,s,c]`` to full QCU ``[s,c,x,y,z,t]``."""
    array = np.asarray(fermion)
    if array.ndim != 7 or array.shape[0] != 2 or array.shape[-2:] != (4, 3):
        raise ValueError("QDP fermion must be [q,t,z,y,x/2,spin,color]")
    _, t, z, y, xh, spin, color = array.shape
    x = 2 * xh
    output = np.empty((t, z, y, x, spin, color), dtype=array.dtype)
    tt = np.arange(t)[:, None, None, None]
    zz = np.arange(z)[None, :, None, None]
    yy = np.arange(y)[None, None, :, None]
    for q in range(2):
        rows = ((((tt * z + zz) * y + yy) * x) +
                ((q - (tt + zz + yy)) & 1) + 2 * np.arange(xh)[None, None, None, :])
        output.reshape(-1, spin * color)[rows] = array[q].reshape(t, z, y, xh, spin * color)
    return np.ascontiguousarray(output.transpose(4, 5, 3, 2, 1, 0))


def _clover_chiral_spins(chi: int) -> tuple[int, int]:
    return (0, 3) if chi == 0 else (1, 2)


def _clover_offdiag_index(row: int, col: int) -> int:
    """Match QUDA's packed 6x6 upper-triangle index (excluding diagonal)."""
    n = 6
    return n * (n - 1) // 2 - (n - row) * (n - row - 1) // 2 + col - row - 1


def qcu_clover_to_quda(clover: np.ndarray) -> np.ndarray:
    """Convert QCU ``[4,3,4,3,x,y,z,t]`` to QUDA packed real ``[q,t,z,y,x/2,2,36]``."""
    array = np.asarray(clover)
    if array.ndim != 8 or array.shape[:4] != (4, 3, 4, 3):
        raise ValueError("QCU clover must be [4,3,4,3,x,y,z,t]")
    _, _, _, _, x, y, z, t = array.shape
    matrices = array.transpose(7, 6, 5, 4, 0, 1, 2, 3).reshape(t, z, y, x, 12, 12)
    packed_full = np.zeros((t, z, y, x, 2, 36), dtype=np.float64)
    for chi in range(2):
        spins = _clover_chiral_spins(chi)
        for lr, sr in enumerate(spins):
            for cr in range(3):
                row = sr * 3 + cr
                packed_full[..., chi, lr * 3 + cr] = matrices[..., row, row].real
        for row in range(6):
            sr, cr = divmod(row, 3)
            for col in range(row + 1, 6):
                sc, cc = divmod(col, 3)
                r_phys = spins[sr] * 3 + cr
                c_phys = spins[sc] * 3 + cc
                k = _clover_offdiag_index(row, col)
                packed_full[..., chi, 6 + 2 * k] = matrices[..., r_phys, c_phys].real
                packed_full[..., chi, 6 + 2 * k + 1] = matrices[..., r_phys, c_phys].imag
    q = (np.indices((t, z, y, x)).sum(axis=0) & 1)
    xh = np.arange(x)[None, None, None, :] // 2
    site_index = (((np.arange(t)[:, None, None, None] * z +
                    np.arange(z)[None, :, None, None]) * y +
                   np.arange(y)[None, None, :, None]) * (x // 2) + xh)
    destination = q * (t * z * y * (x // 2)) + site_index
    source = np.arange(t * z * y * x).reshape(t, z, y, x)
    output = np.empty((2, t, z, y, x // 2, 2, 36), dtype=np.float64)
    output.reshape(-1, 2, 36)[destination.reshape(-1)] = \
        packed_full.reshape(-1, 2, 36)[source.reshape(-1)]
    return np.ascontiguousarray(output)


def quda_clover_to_qcu(clover: np.ndarray) -> np.ndarray:
    """Convert QUDA packed real ``[q,t,z,y,x/2,2,36]`` to QCU clover."""
    array = np.asarray(clover)
    if array.ndim != 7 or array.shape[0] != 2 or array.shape[-2:] != (2, 36):
        raise ValueError("QUDA clover must be [q,t,z,y,x/2,2,36]")
    q_dim, t, z, y, xh, _, _ = array.shape
    assert q_dim == 2
    x = 2 * xh
    packed_full = np.zeros((t, z, y, x, 2, 36), dtype=np.float64)
    q = (np.indices((t, z, y, x)).sum(axis=0) & 1)
    xh_index = np.arange(x)[None, None, None, :] // 2
    site_index = (((np.arange(t)[:, None, None, None] * z +
                    np.arange(z)[None, :, None, None]) * y +
                   np.arange(y)[None, None, :, None]) * xh + xh_index)
    destination = q * (t * z * y * xh) + site_index
    source = np.arange(t * z * y * x).reshape(t, z, y, x)
    packed_full.reshape(-1, 2, 36)[source.reshape(-1)] = \
        array.reshape(-1, 2, 36)[destination.reshape(-1)]
    matrices = np.zeros((t, z, y, x, 12, 12), dtype=np.complex128)
    for chi in range(2):
        spins = _clover_chiral_spins(chi)
        for lr, sr in enumerate(spins):
            for cr in range(3):
                row = sr * 3 + cr
                matrices[..., row, row] = packed_full[..., chi, lr * 3 + cr]
        for row in range(6):
            sr, cr = divmod(row, 3)
            for col in range(row + 1, 6):
                sc, cc = divmod(col, 3)
                r_phys = spins[sr] * 3 + cr
                c_phys = spins[sc] * 3 + cc
                k = _clover_offdiag_index(row, col)
                value = (packed_full[..., chi, 6 + 2 * k] +
                         1j * packed_full[..., chi, 6 + 2 * k + 1])
                matrices[..., r_phys, c_phys] = value
                matrices[..., c_phys, r_phys] = np.conj(value)
    output = matrices.reshape(t, z, y, x, 4, 3, 4, 3)
    return np.ascontiguousarray(output.transpose(4, 5, 6, 7, 3, 2, 1, 0))
