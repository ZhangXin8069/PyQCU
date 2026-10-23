#!/usr/bin/env python3
"""Validate coarse-cache rank slabs, metadata, and digests under MPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyqcu.testing.simple._common import write_json
from pyqcu.testing.simple.kupl_multigrid import write_coarse_assets
from pyqcu.testing.simple.mpi._common import (
    load_coarse_assets,
    local_geometry,
    resolve_grid,
)
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count


ASSET_NAMES = ("lonv", "hnn", "hdg", "sit")


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _synthetic_assets() -> dict[str, np.ndarray]:
    x, y, z, t = (4, 4, 4, 4)
    e, k = 2, 2
    lonv_axes = (e, x, x, y, y, z, z, t, t)
    values = {
        "lonv": np.arange(
            int(np.prod(lonv_axes)), dtype=np.float32
        ).reshape(lonv_axes).astype(np.complex64),
        "hnn": np.arange(k**4 * x * y * z * t, dtype=np.float32
                        ).reshape(k, k, k, k, x, y, z, t).astype(np.complex64),
        "hdg": np.arange(k**5 * x * y * z * t, dtype=np.float32
                        ).reshape(k, k, k, k, k, x, y, z, t).astype(np.complex64),
        "sit": np.arange(k**2 * x * y * z * t, dtype=np.float32
                        ).reshape(k, k, x, y, z, t).astype(np.complex64),
    }
    for index, name in enumerate(ASSET_NAMES, start=1):
        values[name] += 1j * index
    return values


def _read_full(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        missing = sorted(set(ASSET_NAMES).difference(handle.keys()))
        if missing:
            raise ValueError(f"coarse cache is missing datasets {missing}")
        return {name: np.asarray(handle[name][...]) for name in ASSET_NAMES}


def _lattice_axes(name: str, dimensions: int) -> tuple[int, ...]:
    if name == "lonv":
        return (2, 4, 6, 8)
    return tuple(range(dimensions - 4, dimensions))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=None,
                        help="global coarse cache; omitted with --synthetic")
    parser.add_argument("--synthetic", action="store_true",
                        help="create a tiny cache on rank 0 for a smoke test")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        help="process grid over the coarse lattice")
    parser.add_argument("--ranks", type=int, default=None,
                        help="expected MPI_COMM_WORLD size")
    parser.add_argument("--report", default="coarse_cache_mpi.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cache is None and not args.synthetic:
        parser.error("either --cache or --synthetic is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from mpi4py import MPI

    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    try:
        size = validate_rank_count(int(comm.Get_size()))
        if args.ranks is not None and args.ranks != size:
            raise ValueError(
                f"--ranks={args.ranks} does not match MPI_COMM_WORLD size {size}")
    except ValueError as exc:
        if rank == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    scratch: tempfile.TemporaryDirectory[str] | None = None
    cache_location = args.cache
    coarse_lattice: tuple[int, int, int, int] | None = None
    if args.synthetic and rank == 0:
        scratch = tempfile.TemporaryDirectory(prefix="pyqcu-coarse-mpi-")
        cache_path = Path(scratch.name) / "coarse_synthetic.h5"
        assets = _synthetic_assets()
        write_coarse_assets(cache_path, *(assets[name] for name in ASSET_NAMES))
        cache_location = str(cache_path)
        coarse_lattice = tuple(int(value) for value in assets["sit"].shape[-4:])
    shared = comm.bcast((cache_location, coarse_lattice), root=0)
    cache_location, coarse_lattice = shared
    if cache_location is None:
        raise RuntimeError("rank 0 did not publish the coarse-cache path")
    metadata_error = None
    if coarse_lattice is None and rank == 0:
        try:
            fixture = _read_full(Path(cache_location))
            coarse_lattice = tuple(
                int(value) for value in fixture["sit"].shape[-4:])
        except Exception:
            import traceback

            metadata_error = traceback.format_exc()
    coarse_lattice, metadata_error = comm.bcast(
        (coarse_lattice, metadata_error), root=0)
    if metadata_error is not None or coarse_lattice is None:
        if rank == 0:
            output = Path(args.report).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(
                f".{output.name}.{os.getpid()}.partial")
            write_json(temporary, {
                "status": "FAIL",
                "mode": "rank-slab",
                "cache": cache_location,
                "ranks": size,
                "errors": [metadata_error or "coarse-cache metadata unavailable"],
            })
            os.replace(temporary, output)
            print(json.dumps({
                "status": "FAIL",
                "cache": cache_location,
                "report": str(output),
                "error": "coarse-cache metadata unavailable",
            }, ensure_ascii=False))
        return 1

    try:
        grid = resolve_grid(coarse_lattice, size, args.grid)
        local_shape, starts, coordinate = local_geometry(
            coarse_lattice, grid, rank)
    except ValueError as exc:
        if rank == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    if args.dry_run:
        if rank == 0:
            print(json.dumps({
                "status": "DRY-RUN",
                "mode": "rank-slab",
                "cache": cache_location,
                "coarse_lattice": list(coarse_lattice),
                "grid": list(grid),
                "ranks": size,
                "local_shape": list(local_shape),
                "report": str(args.report),
                "synthetic": args.synthetic,
            }, ensure_ascii=False))
        if scratch is not None:
            scratch.cleanup()
        return 0

    errors: list[str] = []
    local_assets: dict[str, np.ndarray] = {}
    local_records: dict[str, dict[str, object]] = {}
    try:
        loaded = load_coarse_assets(
            cache_location, dtype, starts=starts, local_shape=local_shape)
        for name, value in zip(ASSET_NAMES, loaded, strict=True):
            local_assets[name] = np.asarray(value)
            if value.dtype != dtype:
                errors.append(
                    f"{name}: dtype {value.dtype} != requested {dtype}")
            axes = _lattice_axes(name, value.ndim)
            selected_shape = tuple(value.shape[axis] for axis in axes)
            if selected_shape != local_shape:
                errors.append(
                    f"{name}: rank-local shape {selected_shape} != {local_shape}")
            if not np.isfinite(value).all():
                errors.append(f"{name}: non-finite values")
            local_records[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _digest(value),
            }
    except Exception:
        import traceback

        errors.append(traceback.format_exc())

    payload = {
        "rank": rank,
        "coordinate": list(coordinate),
        "starts": list(starts),
        "local_shape": list(local_shape),
        "assets": local_records,
        "errors": errors,
        "local_assets": local_assets,
    }
    gathered = comm.gather(payload, root=0)
    final_failure = int(comm.allreduce(int(bool(errors)), op=MPI.MAX))
    if rank == 0:
        assert gathered is not None
        all_errors = [
            f"rank {item['rank']}: {error}"
            for item in gathered for error in item["errors"]]
        reconstructed_ok: dict[str, bool] = {}
        global_hashes: dict[str, str] = {}
        if not final_failure and not all_errors:
            global_arrays = _read_full(Path(cache_location))
            for name in ASSET_NAMES:
                global_value = np.asarray(global_arrays[name], dtype=dtype)
                global_hashes[name] = _digest(global_value)
                rebuilt = np.empty_like(global_value)
                try:
                    axes = _lattice_axes(name, rebuilt.ndim)
                    for item in gathered:
                        selection = [slice(None)] * rebuilt.ndim
                        for axis, h5_axis in enumerate(axes):
                            selection[h5_axis] = slice(
                                item["starts"][axis],
                                item["starts"][axis] + item["local_shape"][axis])
                        rebuilt[tuple(selection)] = item["local_assets"][name]
                    reconstructed_ok[name] = (
                        _digest(rebuilt) == global_hashes[name])
                except Exception:
                    reconstructed_ok[name] = False
                if not reconstructed_ok[name]:
                    all_errors.append(
                        f"{name}: gathered rank slabs do not match cache")
        else:
            reconstructed_ok = {name: False for name in ASSET_NAMES}
        status = "PASS" if not final_failure and not all_errors and all(
            reconstructed_ok.values()) else "FAIL"
        report = {
            "status": status,
            "mode": "rank-slab",
            "cache": cache_location,
            "synthetic": bool(args.synthetic),
            "ranks": size,
            "grid": list(grid),
            "coarse_lattice": list(coarse_lattice),
            "local_shape": list(local_shape),
            "expected_dtype": str(dtype),
            "global_sha256": global_hashes,
            "reconstructed_from_slabs": reconstructed_ok,
            "rank_records": [
                {key: value for key, value in item.items()
                 if key != "local_assets"}
                for item in gathered],
            "errors": all_errors,
        }
        output = Path(args.report).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.partial")
        write_json(temporary, report)
        os.replace(temporary, output)
        print(json.dumps(report, ensure_ascii=False))
    else:
        status = "FAIL" if final_failure else "PASS"
    final = int(comm.bcast(int(status == "FAIL"), root=0))
    if scratch is not None:
        scratch.cleanup()
    return 1 if final else 0


if __name__ == "__main__":
    raise SystemExit(main())
