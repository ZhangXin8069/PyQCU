#!/usr/bin/env python3
"""Prepare contiguous rank-local HDF5 slabs for a fixed MPI process grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from ._common import resolve_path
except ImportError:
    from _common import resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry(lat: tuple[int, int, int, int],
              grid: tuple[int, int, int, int], rank: int):
    if any(extent <= 0 for extent in lat + grid):
        raise ValueError("lattice and grid extents must be positive")
    if any(lat[axis] % grid[axis] for axis in range(4)):
        raise ValueError(f"lattice {lat} is not divisible by grid {grid}")
    local = tuple(lat[axis] // grid[axis] for axis in range(4))
    coordinate = tuple(int(value) for value in np.unravel_index(
        rank, grid, order="C"))
    starts = tuple(coordinate[axis] * local[axis] for axis in range(4))
    return coordinate, local, starts


def _write_slab(path: Path, data: np.ndarray, *, kind: str,
                input_path: Path, lat: tuple[int, int, int, int],
                grid: tuple[int, int, int, int], rank: int,
                coordinate: tuple[int, int, int, int],
                local: tuple[int, int, int, int],
                starts: tuple[int, int, int, int]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{rank}.partial")
    try:
        with h5py.File(partial, "w") as handle:
            handle.create_dataset("data", data=np.ascontiguousarray(data))
            handle.attrs["kind"] = kind
            handle.attrs["source"] = str(input_path)
            handle.attrs["lattice"] = np.asarray(lat, dtype=np.int64)
            handle.attrs["grid"] = np.asarray(grid, dtype=np.int64)
            handle.attrs["rank"] = int(rank)
            handle.attrs["coordinate"] = np.asarray(coordinate, dtype=np.int64)
            handle.attrs["local"] = np.asarray(local, dtype=np.int64)
            handle.attrs["starts"] = np.asarray(starts, dtype=np.int64)
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lat", nargs=4, type=int, required=True)
    parser.add_argument("--grid", nargs=4, type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--kind", choices=("gauge", "source"), required=True)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--hash-input", action="store_true",
                        help="hash the complete input after writing all slabs")
    args = parser.parse_args(argv)

    import h5py

    lat = tuple(int(value) for value in args.lat)
    grid = tuple(int(value) for value in args.grid)
    ranks = int(np.prod(grid))
    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    input_path = resolve_path(args.input, args.input.expanduser())
    output_dir = resolve_path(
        args.output_dir, args.output_dir.expanduser())
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing slab manifest {manifest_path}")

    records = []
    with h5py.File(input_path, "r") as source:
        if args.dataset not in source:
            raise ValueError(f"{input_path} has no dataset {args.dataset!r}")
        dataset = source[args.dataset]
        if tuple(dataset.shape[-4:]) != lat:
            raise ValueError(
                f"input trailing lattice {dataset.shape[-4:]} != {lat}")
        prefix = tuple(int(value) for value in dataset.shape[:-4])
        expected_prefix = (3, 3, 4) if args.kind == "gauge" else (4, 3)
        if prefix != expected_prefix:
            raise ValueError(
                f"{args.kind} prefix {prefix} != {expected_prefix}")
        for rank in range(ranks):
            coordinate, local, starts = _geometry(lat, grid, rank)
            selection = (Ellipsis,) + tuple(
                slice(starts[axis], starts[axis] + local[axis])
                for axis in range(4))
            value = np.ascontiguousarray(
                dataset[selection], dtype=dtype)
            target = output_dir / f"rank_{rank:04d}.h5"
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing slab {target}")
            _write_slab(
                target, value, kind=args.kind, input_path=input_path,
                lat=lat, grid=grid, rank=rank, coordinate=coordinate,
                local=local, starts=starts)
            records.append({
                "rank": rank,
                "coordinate": list(coordinate),
                "local": list(local),
                "starts": list(starts),
                "path": str(target),
                "shape": list(value.shape),
                "size_bytes": int(target.stat().st_size),
            })
            print(
                f"slab rank={rank}/{ranks} shape={value.shape} "
                f"bytes={target.stat().st_size}",
                flush=True)
            del value

    manifest = {
        "schema": "pyqcu.mpi-slab-cache.v1",
        "kind": args.kind,
        "input": str(input_path),
        "dataset": args.dataset,
        "prefix_shape": list(prefix),
        "lattice": list(lat),
        "grid": list(grid),
        "ranks": ranks,
        "dtype": str(dtype),
        "slabs": records,
    }
    if args.hash_input:
        manifest["input_sha256"] = _sha256(input_path)
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "ranks": ranks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
