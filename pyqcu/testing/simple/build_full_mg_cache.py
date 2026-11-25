#!/usr/bin/env python3
"""Build one Legacy-MG cache root: the fine 33-tensor stencil or a Galerkin step.

Without ``--parent-cache`` the fine Wilson-Clover 33-tensor stencil is built from
the gauge on CPU or CUDA.  With ``--parent-cache`` the child level of one
Galerkin transition is built by
:func:`pyqcu.testing.simple.mg_transition.build_transition`, which probes the
parent root's stencil instead of the gauge: a level-N hierarchy is
``op(L) = R_L . op(L-1) . P_L``, so transition 1 is gauge-built and every later
transition consumes the root published by the transition before it.  One cache
root per transition, in order.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    file_sha256,
    legacy_mg_cache_identity,
    logical_array_sha256,
    save_legacy_mg_cache,
    write_legacy_mg_cache_manifest,
)

try:
    from ._common import (
        lime_metadata, qdp_gauge_to_qcu, read_h5, read_h5_metadata,
        read_qdp_gauge, resolve_path,
    )
    from .kupl_multigrid import build_coarse_assets
except ImportError:
    from _common import (
        lime_metadata, qdp_gauge_to_qcu, read_h5, read_h5_metadata,
        read_qdp_gauge, resolve_path,
    )
    from kupl_multigrid import build_coarse_assets


GALERKIN_TRANSITION_MODULE = "pyqcu.testing.simple.mg_transition"
FINE_BUILDER = "cxx-null-vector+local-33-stencil"
TRANSITION_BUILDER = "galerkin-transition-probe"


def _transition_builder():
    """Return ``(kind, callable)`` of the Galerkin bridge builder.

    ``pyqcu.testing.simple.mg_transition`` publishes
    ``build_transition_cache(parent_root, output_root, parent_lattice, block,
    dof, ...)``, the build/gate/publish facade for one transition.  A module
    trimmed to a bare
    ``build_transition(parent_root, parent_lattice, block, E_child, out_root)``
    that publishes the root itself is accepted as well, so this dispatcher does
    not pin the bridge builder's internal decomposition.
    """
    try:
        from . import mg_transition
    except ImportError:
        try:
            import mg_transition  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                f"{GALERKIN_TRANSITION_MODULE} is not available; it is required "
                "to build the root of a Galerkin transition (--parent-cache) "
                "from a parent cache root") from exc
    facade = getattr(mg_transition, "build_transition_cache", None)
    if callable(facade):
        return "cache", facade
    builder = getattr(mg_transition, "build_transition", None)
    if callable(builder):
        return "bare", builder
    raise RuntimeError(
        f"{GALERKIN_TRANSITION_MODULE} exposes neither build_transition_cache "
        "nor build_transition; cannot build a Galerkin transition root")


def _transition_dof(dof: Sequence[int], level: int) -> tuple[int, int]:
    """Return ``(E_parent, E_child)`` for transition ``level``.

    ``--dof`` carries either this transition's pair ``[E_parent, E_child]``
    (what :func:`legacy_mg_cache_shapes` consumes, and what ``solve_case.py``
    passes) or the full per-level list, in which case the pair around
    ``--level`` is used.  Both spellings agree for every chain this repo builds.
    """
    values = [int(value) for value in dof]
    if len(values) >= int(level) + 1:
        return values[int(level) - 1], values[int(level)]
    return values[0], values[-1]


def _call_transition_builder(builder: tuple[str, Any], parent_cache: Path,
                             parent_lattice: Sequence[int],
                             block: Sequence[int], parent_e: int, child_e: int,
                             output: Path, *, level: int, dtype: str,
                             device: str, seed: int) -> Any:
    """Call the bridge builder, forwarding only the keywords it declares.

    ``build_transition_cache`` needs the parent lattice and block and derives
    everything else from the parent root, but a 5-positional
    ``build_transition`` takes ``E_child`` instead of the pair; both are called
    with their documented argument order and with the provenance keywords each
    one declares (``dtype``, ``device``, ``seed``, ``verbose``, ...).
    """
    kind, function = builder
    candidates: dict[str, Any] = {
        "dtype": dtype,
        "device": device,
        "seed": int(seed),
        "level": int(level),
        "parent_e": int(parent_e),
        "verbose": True,
    }
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    variadic = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                   for parameter in parameters.values())
    keywords = {
        name: value for name, value in candidates.items()
        if variadic or name in parameters
    }
    if kind == "cache":
        return function(
            parent_cache, output, list(parent_lattice), list(block),
            [int(parent_e), int(child_e)], **keywords)
    return function(
        parent_cache, list(parent_lattice), list(block), int(child_e), output,
        **keywords)


def _publish_report(report: dict[str, Any], output: Path,
                    report_path: Path | None) -> None:
    """Write the transition run report; never the cache manifest itself."""
    if report_path is None:
        return
    if report_path.resolve() == output.with_suffix(".json").resolve():
        raise RuntimeError(
            "--report must not equal the cache manifest path; the bridge "
            "builder owns that file")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _build_transition(args, parser: argparse.ArgumentParser) -> int:
    """Build one Galerkin transition root from its parent root."""
    level = int(args.level)
    lattice = tuple(int(value) for value in args.lat)
    block = [int(value) for value in args.mg_grid]
    dof = [int(value) for value in args.dof]
    if len(dof) < 2:
        parser.error(
            "--dof must give the parent and child basis dimensions "
            "([E_parent, E_child] or the full per-level list)")
    parent_e, child_e = _transition_dof(dof, level)
    parent_cache = resolve_path(
        args.parent_cache, Path(args.parent_cache).expanduser())
    output = resolve_path(args.output, Path(args.output).expanduser())
    report_path = (
        resolve_path(args.report, Path(args.report).expanduser())
        if args.report is not None else None)
    if report_path is not None and report_path.resolve() == \
            output.with_suffix(".json").resolve():
        parser.error(
            "--report must not equal the cache manifest path")
    if args.dry_run:
        print(json.dumps({
            "status": "DRY-RUN",
            "builder": TRANSITION_BUILDER,
            "level": level,
            "parent_cache": str(parent_cache),
            "parent_cache_exists": parent_cache.is_file(),
            "parent_lattice": list(lattice),
            "block": block,
            "parent_e": parent_e,
            "child_e": child_e,
            "output": str(output),
            "report": None if report_path is None else str(report_path),
        }, ensure_ascii=False))
        return 0
    if not parent_cache.is_file():
        parser.error(
            f"--parent-cache {parent_cache} does not exist: the root of "
            f"transition {level - 1} is required to build transition {level}")

    builder = _transition_builder()
    started = time.perf_counter()
    result = _call_transition_builder(
        builder, parent_cache, lattice, block, parent_e, child_e, output,
        level=level, dtype=args.dtype, device=args.build_device, seed=args.seed)
    elapsed = time.perf_counter() - started
    if isinstance(result, (str, Path)) and \
            Path(result).expanduser().resolve() != output.resolve():
        raise RuntimeError(
            f"the bridge builder published {result} instead of the requested "
            f"transition root {output}")
    if not output.is_file():
        raise RuntimeError(
            f"the bridge builder did not publish the root of transition "
            f"{level}: {output}")

    report: dict[str, Any] = {}
    if isinstance(result, dict):
        report.update(result)
    sidecar = output.with_suffix(".json")
    if sidecar.is_file():
        try:
            declared = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            declared = {}
        if isinstance(declared, dict):
            for key, value in declared.items():
                report.setdefault(key, value)
    report.update({
        "builder": TRANSITION_BUILDER,
        "level": level,
        "parent_cache": str(parent_cache),
        "parent_lattice": list(lattice),
        "block": block,
        "parent_e": parent_e,
        "child_e": child_e,
        "output": str(output),
        "build_device": args.build_device,
        "setup_seconds": elapsed,
    })
    report.setdefault("status", "PASS")
    if args.grid is not None:
        report.setdefault("grid", [int(value) for value in args.grid])
    _publish_report(report, output, report_path)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", default=None,
                        help="LIME/HDF5 gauge; required unless --parent-cache")
    parser.add_argument("--lat", nargs=4, type=int, required=True,
                        help="level lattice (even time); the parent lattice when "
                             "--parent-cache is given")
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--csw", type=float, required=True)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--dof", nargs="+", type=int, default=(12, 12))
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--nv-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--parent-cache", type=Path, default=None,
                        help="parent transition root; selects the Galerkin "
                             "bridge builder instead of the gauge stencil")
    parser.add_argument("--level", type=int, default=None,
                        help="1-based transition index of the root being built "
                             "(1 = gauge-built fine stencil)")
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        help="MPI process grid forwarded by solve_case.py; the "
                             "single-process builder always builds the global "
                             "cache and only records the grid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved dispatch and exit")
    args = parser.parse_args(argv)

    lattice = tuple(int(value) for value in args.lat)
    output = resolve_path(args.output, Path(args.output).expanduser())
    if args.parent_cache is not None:
        if args.level is None:
            parser.error("--parent-cache requires --level <transition index>")
        if int(args.level) < 2:
            parser.error(
                "--level counts Galerkin transitions and starts at 2; level 1 "
                "is the gauge-built fine stencil")
        return _build_transition(args, parser)
    if args.gauge is None:
        parser.error("--gauge is required unless --parent-cache is given")
    if args.level is not None and int(args.level) != 1:
        parser.error(
            f"--level {args.level} needs --parent-cache; without it this "
            "builder publishes the gauge-built root of transition 1")

    gauge_path = resolve_path(args.gauge, Path(args.gauge).expanduser())
    report_path = (
        resolve_path(args.report, Path(args.report).expanduser())
        if args.report is not None else None)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY-RUN",
            "builder": FINE_BUILDER,
            "level": 1,
            "gauge": str(gauge_path),
            "gauge_exists": gauge_path.is_file(),
            "lattice": list(lattice),
            "block": [int(value) for value in args.mg_grid],
            "dof": [int(value) for value in args.dof],
            "build_device": args.build_device,
            "output": str(output),
            "report": None if report_path is None else str(report_path),
        }, ensure_ascii=False))
        return 0

    if gauge_path.suffix.lower() == ".lime":
        metadata = lime_metadata(gauge_path)
        gauge = qdp_gauge_to_qcu(read_qdp_gauge(gauge_path))
    else:
        metadata = read_h5_metadata(gauge_path)
        gauge = read_h5(gauge_path)
    if tuple(gauge.shape[-4:]) != lattice:
        raise ValueError(
            f"gauge lattice {tuple(gauge.shape[-4:])} does not match {lattice}")

    dtype = np.dtype(np.complex64 if args.dtype == "c64" else np.complex128)
    gauge = np.ascontiguousarray(gauge, dtype=dtype)
    kappa = 1.0 / (2.0 * float(args.mass) + 8.0)
    started = time.perf_counter()
    setup_stats: dict[str, object] = {}
    assets = build_coarse_assets(
        gauge, lattice, args.mass, kappa, args.csw, dtype, 2,
        list(args.dof), list(args.mg_grid), args.nv_iters, args.seed,
        verbose=True, local_stencil=True,
        local_stencil_batch_size=args.batch_size,
        device_name=args.build_device, setup_stats=setup_stats)
    elapsed = time.perf_counter() - started
    datasets = {
        name: value[0] for name, value in zip(
            ("lonv", "hnn", "hdg", "sit"), assets)
    }
    identity = legacy_mg_cache_identity(
        gauge_canonical_sha256=logical_array_sha256(gauge),
        gauge_file_sha256=file_sha256(gauge_path),
        lattice=lattice,
        dtype=dtype,
        mass=args.mass,
        csw=args.csw,
        dof=args.dof,
        mg_grid=args.mg_grid,
        nv_iters=args.nv_iters,
        seed=args.seed,
    )
    report = save_legacy_mg_cache(
        output, datasets, identity=identity,
        setup_method=FINE_BUILDER,
        setup_backend=(
            "cuda-qcu-cxx" if args.build_device == "cuda"
            else "kupl-qcu-cxx"),
        setup_validation=setup_stats,
        source_metadata=metadata,
        setup_seconds=elapsed)
    report.update({
        "builder": FINE_BUILDER,
        "level": 1,
        "gauge": str(gauge_path),
        "build_device": args.build_device,
        "local_stencil": True,
        "local_stencil_batch_size": int(args.batch_size),
        "output": str(output),
    })
    if args.grid is not None:
        report["grid"] = [int(value) for value in args.grid]
    if report_path is not None:
        write_legacy_mg_cache_manifest(output, report)
        if report_path.resolve() != output.with_suffix(".json").resolve():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
