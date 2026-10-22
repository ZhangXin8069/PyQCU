#!/usr/bin/env python3
"""Prepare a production QCU case from a LIME configuration.

The script extracts either the full lattice or a same-mass/same-Csw
sub-volume, converts it to the QCU HDF5 layout, generates the shared
complex64 source, and writes a manifest with the exact paths, shapes and
SHA256 digests used by every backend.

Example::

    source ./env.sh
    python pyqcu/testing/simple/prepare_case.py \\
        --lat 24 24 24 64 --out-dir data/production_L24

Sub-volume at the same mass and Csw (seams are not periodic, so use it for
controlled size/parallel studies, not physics)::

    python pyqcu/testing/simple/prepare_case.py --lat 16 16 16 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    from ._common import (DEFAULT_CSW, DEFAULT_MASS, lime_metadata,
                          qdp_gauge_to_qcu, read_h5_metadata, read_qdp_gauge,
                          resolve_path, write_h5, write_json)
except ImportError:
    from _common import (DEFAULT_CSW, DEFAULT_MASS, lime_metadata,
                         qdp_gauge_to_qcu, read_h5_metadata, read_qdp_gauge,
                         resolve_path, write_h5, write_json)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_LIME = Path.home() / "data" / (
    "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_gauge(lime: Path, lat, out_dir: Path, mass: float,
                  csw: float) -> Path:
    """Write the requested (sub-)volume as a QCU HDF5 gauge."""
    meta = lime_metadata(lime)
    full = tuple(meta["lattice_xyzt"])
    target = tuple(int(item) for item in lat)
    if not all(item <= extent for item, extent in zip(target, full)):
        raise ValueError(f"requested lattice {target} exceeds LIME {full}")
    path = out_dir / f"gauge_{'x'.join(map(str, target))}_c64.h5"
    if path.is_file():
        # Never silently reuse a cache written for another mass/Csw/shape.
        existing = read_h5_metadata(path)
        if (tuple(int(v) for v in existing.get("lattice", ())) != target or
                abs(float(existing.get("mass", mass)) - float(mass)) > 1e-12 or
                abs(float(existing.get("csw", csw)) - float(csw)) > 1e-12):
            raise ValueError(
                f"{path} exists but does not match lattice={target}, "
                f"mass={mass}, csw={csw}; remove it or choose another "
                "--out-dir")
        return path
    gauge = read_qdp_gauge(lime)
    gauge = gauge[:, :target[3], :target[2], :target[1], :target[0]]
    write_h5(path, qdp_gauge_to_qcu(gauge), metadata={
        "mass": mass, "csw": csw, "lattice": list(target),
        "provenance": (
            f"sub-volume of {lime.name} at fixed mass and Csw; "
            "seams are not periodic"),
    })
    return path


def prepare_source(lat, out_dir: Path, seed: int) -> Path:
    """Write the PCG64 source shared by CUDA QCU, KUPL QCU and QUDA."""
    target = tuple(int(item) for item in lat)
    path = out_dir / f"source_{'x'.join(map(str, target))}_c64.h5"
    if path.is_file():
        existing = read_h5_metadata(path)
        if int(existing.get("seed", seed)) != int(seed):
            raise ValueError(
                f"{path} was generated with seed {existing.get('seed')}, "
                f"requested {seed}; remove it or choose another --out-dir")
        return path
    rng = np.random.default_rng(seed)
    array = (rng.normal(size=(4, 3, *target)) +
             1j * rng.normal(size=(4, 3, *target))).astype(np.complex64)
    write_h5(path, array, metadata={
        "seed": seed,
        "generator": "numpy PCG64 normal float64, cast to complex64",
        "layout": "[spin,color,x,y,z,t]",
    })
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lime", default=str(DEFAULT_LIME))
    parser.add_argument("--lat", nargs=4, type=int, default=(24, 24, 24, 64))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    lime = resolve_path(args.lime, Path(args.lime).expanduser())
    if not lime.is_file():
        raise FileNotFoundError(lime)
    meta = lime_metadata(lime)
    full = tuple(meta["lattice_xyzt"])
    if not all(target <= extent for target, extent in zip(args.lat, full)):
        parser.error(f"requested lattice {tuple(args.lat)} exceeds LIME {full}")
    mass = (float(args.mass) if args.mass is not None else
            float(meta.get("reference_mass", DEFAULT_MASS)))
    csw = (float(args.csw) if args.csw is not None else
           float(meta.get("csw", DEFAULT_CSW)))
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        REPO / "data" / f"production_{'x'.join(map(str, args.lat))}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gauge = prepare_gauge(lime, args.lat, out_dir, mass, csw)
    source = prepare_source(args.lat, out_dir, args.seed)
    manifest = {
        "lime": str(lime),
        "lime_metadata": meta,
        "lattice": list(args.lat),
        "full_lattice": list(full),
        "sub_volume": list(args.lat) != list(full),
        "mass": mass, "csw": csw, "seed": args.seed,
        "dtype": "complex64",
        "gauge": {"path": str(gauge), "sha256": _sha256(gauge),
                  "size_bytes": gauge.stat().st_size},
        "source": {"path": str(source), "sha256": _sha256(source),
                   "size_bytes": source.stat().st_size},
        "note": ("sub-volume keeps the original gauge links; seams are not "
                 "periodic" if list(args.lat) != list(full) else
                 "full reference lattice"),
    }
    manifest_path = out_dir / "case_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"gauge": str(gauge), "source": str(source),
                      "manifest": str(manifest_path),
                      "mass": mass, "csw": csw,
                      "sub_volume": manifest["sub_volume"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
