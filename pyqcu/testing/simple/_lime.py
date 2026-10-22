"""Single-file SciDAC/LIME gauge metadata and payload helpers."""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _lime_records(path: Path,
                  wanted: set[str] | None = None,
                  any_of: set[str] | None = None) -> dict[str, tuple[int, int]]:
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
            if wanted is not None and not wanted.issubset(records):
                handle.seek(offset + ((length + 7) // 8) * 8)
                continue
            if any_of is not None and records.keys().isdisjoint(any_of):
                handle.seek(offset + ((length + 7) // 8) * 8)
                continue
            if wanted is not None or any_of is not None:
                break
            handle.seek(offset + ((length + 7) // 8) * 8)
    return records


def _lime_xml(path: Path, records: Mapping[str, tuple[int, int]],
              name: str) -> ET.Element:
    offset, length = records[name]
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length).rstrip(b"\0")
    return ET.fromstring(payload.decode("utf-8"))


def _lime_record_name(records: Mapping[str, tuple[int, int]],
                      *names: str) -> str:
    for name in names:
        if name in records:
            return name
    raise KeyError(f"missing LIME record; expected one of {names}")


def lime_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return gauge dimensions and the Chroma clover coefficient if present."""
    path = Path(path).expanduser()
    records = _lime_records(path)
    file_name = _lime_record_name(
        records, "scidac-private-file-xml", "scidac-file-xml")
    file_xml = _lime_xml(path, records, file_name)
    dims = tuple(int(value) for value in file_xml.findtext("dims", "").split())
    if len(dims) != 4:
        raise ValueError(f"invalid LIME lattice dimensions in {path}: {dims}")
    meta: dict[str, Any] = {"lattice_xyzt": dims, "file": str(path)}
    if "scidac-record-xml" not in records:
        records.update(_lime_records(
            path, any_of={"scidac-private-record-xml"}))
    record_name = (
        "scidac-record-xml" if "scidac-record-xml" in records
        else "scidac-private-record-xml"
        if "scidac-private-record-xml" in records else None)
    if record_name is not None:
        record_xml = _lime_xml(path, records, record_name)
        coefficient = record_xml.find(".//clovCoeff")
        if coefficient is not None and coefficient.text:
            meta["csw"] = float(coefficient.text)
        mass = record_xml.find(".//Mass")
        if mass is not None and mass.text:
            meta["reference_mass"] = float(mass.text)
    return meta


def read_lime_gauge(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a single-file USQCD/QIO gauge field as QDP ``[d,t,z,y,x,3,3]``."""
    path = Path(path).expanduser()
    records = _lime_records(path)
    file_name = _lime_record_name(
        records, "scidac-private-file-xml", "scidac-file-xml")
    record_name = _lime_record_name(
        records, "scidac-private-record-xml", "scidac-record-xml")
    file_xml = _lime_xml(path, records, file_name)
    record_xml = _lime_xml(path, records, record_name)
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
    gauge = np.ascontiguousarray(
        raw.transpose(4, 0, 1, 2, 3, 5, 6),
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
    raw = np.ascontiguousarray(
        np.transpose(gauge, (1, 2, 3, 4, 0, 5, 6)), dtype=">c16")
    rows = raw.reshape(-1, 36).view(np.uint8)
    work = np.empty(rows.shape[0], dtype="<u4")
    for index in range(rows.shape[0]):
        work[index] = zlib.crc32(rows[index])
    rank = np.arange(rows.shape[0], dtype="<u4")
    result = []
    for modulus in (29, 31):
        shift = rank % modulus
        rotated = np.where(
            shift == 0, work, (work << shift) | (work >> (32 - shift)))
        result.append(int(np.bitwise_xor.reduce(rotated)))
    return result[0], result[1]


def write_lime_gauge(path: str | os.PathLike[str], gauge: np.ndarray,
                     metadata: Mapping[str, Any] | None = None) -> None:
    """Write a single-file SciDAC/LIME gauge field with a verified checksum."""
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
    metadata = metadata or {}
    clover_xml = ""
    if metadata.get("csw") is not None:
        clover_xml += f"<clovCoeff>{float(metadata['csw']):.12g}</clovCoeff>"
    if metadata.get("reference_mass") is not None:
        clover_xml += f"<Mass>{float(metadata['reference_mass']):.12g}</Mass>"
    private_record = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<scidacRecord><version>1.1</version><recordtype>0</recordtype>'
        '<datatype>QDP_D3_ColorMatrix</datatype><precision>D</precision>'
        '<colors>3</colors><spins>4</spins><typesize>144</typesize>'
        f'<datacount>4</datacount>{clover_xml}</scidacRecord>'
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
    raw = np.ascontiguousarray(
        np.transpose(array, (1, 2, 3, 4, 0, 5, 6)), dtype=">c16")
    with path.open("wb") as handle:
        _lime_record(handle, "scidac-private-file-xml", private_file)
        _lime_record(handle, "scidac-private-record-xml", private_record)
        _lime_record(handle, "ildg-format", ildg_format)
        _lime_record(handle, "ildg-binary-data", raw.tobytes(order="C"))
        _lime_record(handle, "scidac-checksum", checksum)


__all__ = [
    "_lime_checksum",
    "_lime_record",
    "_lime_record_name",
    "_lime_records",
    "_lime_xml",
    "lime_metadata",
    "read_lime_gauge",
    "write_lime_gauge",
]
