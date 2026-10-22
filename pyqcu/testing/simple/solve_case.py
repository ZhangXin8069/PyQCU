#!/usr/bin/env python3
"""Production single-solve driver for CUDA QCU, KUPL QCU and QUDA.

``compare_three_way.py`` exists for measurements; this script is the
production entry point.  It prepares input if needed, runs exactly one cold
solve on the selected backend, writes the solution in the QCU HDF5 layout and
returns a JSON report with an independent true residual.

Examples::

    source ./env.sh

    # Prepare once (also done automatically when --lime is used)
    python pyqcu/testing/simple/prepare_case.py --lat 24 24 24 64 \\
        --out-dir data/production_L24

    # CUDA QCU on the V100 (physical PCI index 2)
    python pyqcu/testing/simple/solve_case.py --backend cuda \\
        --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24

    # KUPL QCU on 8 ranks x 2 OpenMP threads, 1x4x2x1 grid
    python pyqcu/testing/simple/solve_case.py --backend kupl \\
        --lat 24 24 24 64 --kupl-ranks 8 --kupl-threads 2 \\
        --out-dir data/production_L24

    # QUDA
    python pyqcu/testing/simple/solve_case.py --backend quda \\
        --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from ._common import (DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                          lime_metadata, read_h5_metadata, resolve_path,
                          write_json)
    from ._runner import (REPO, cuda_environment, default_kupl_grid,
                          kupl_command, kupl_environment,
                          libqcu_environment, quda_environment, run_report)
    from .prepare_case import prepare_gauge, prepare_source
except ImportError:
    from _common import (DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                         lime_metadata, read_h5_metadata, resolve_path,
                         write_json)
    from _runner import (REPO, cuda_environment, default_kupl_grid,
                         kupl_command, kupl_environment,
                         libqcu_environment, quda_environment, run_report)
    from prepare_case import prepare_gauge, prepare_source

SIMPLE = Path(__file__).resolve().parent
DEFAULT_LIME = Path.home() / "data" / (
    "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime")


def _gauge_metadata(path: Path) -> dict:
    if path.suffix == ".lime":
        return lime_metadata(path)
    if path.suffix in (".h5", ".hdf5"):
        return read_h5_metadata(path)
    return {}


def _resolve_case(args, out_dir: Path):
    if args.gauge is not None:
        gauge = resolve_path(args.gauge, Path(args.gauge).expanduser())
        if not gauge.is_file():
            raise FileNotFoundError(gauge)
        meta = _gauge_metadata(gauge)
        if tuple(meta.get("lattice", ())):
            gauge_lat = tuple(int(v) for v in meta["lattice"])
            if gauge_lat != tuple(args.lat):
                raise ValueError(
                    f"gauge lattice {gauge_lat} does not match --lat "
                    f"{tuple(args.lat)}")
        mass = (float(args.mass) if args.mass is not None else
                float(meta.get("reference_mass", meta.get("mass",
                                                         DEFAULT_MASS))))
        csw = (float(args.csw) if args.csw is not None else
               float(meta.get("csw", DEFAULT_CSW)))
        source = (resolve_path(args.source, Path(args.source).expanduser())
                  if args.source is not None else
                  prepare_source(args.lat, out_dir, args.seed))
        if not source.is_file():
            raise FileNotFoundError(source)
        return gauge, source, mass, csw, meta

    lime = resolve_path(args.lime, Path(args.lime).expanduser())
    if not lime.is_file():
        raise FileNotFoundError(lime)
    meta = lime_metadata(lime)
    full = tuple(meta["lattice_xyzt"])
    if not all(a <= b for a, b in zip(args.lat, full)):
        raise ValueError(f"requested lattice {tuple(args.lat)} exceeds "
                         f"LIME {full}")
    mass = (float(args.mass) if args.mass is not None else
            float(meta.get("reference_mass", DEFAULT_MASS)))
    csw = (float(args.csw) if args.csw is not None else
           float(meta.get("csw", DEFAULT_CSW)))
    gauge = prepare_gauge(lime, args.lat, out_dir, mass, csw)
    source = (resolve_path(args.source, Path(args.source).expanduser())
              if args.source is not None else
              prepare_source(args.lat, out_dir, args.seed))
    return gauge, source, mass, csw, meta


def _solution_path(out_dir: Path, backend: str, lat) -> Path:
    return out_dir / f"solution_{'x'.join(map(str, lat))}_{backend}_c64.h5"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cuda", "quda", "kupl"),
                        required=True)
    parser.add_argument("--lime", default=str(DEFAULT_LIME))
    parser.add_argument("--gauge", default=None,
                        help="QCU HDF5 or .lime; overrides --lime")
    parser.add_argument("--source", default=None,
                        help="full QCU fermion HDF5 [spin,color,x,y,z,t]")
    parser.add_argument("--lat", nargs=4, type=int, default=(24, 24, 24, 64))
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--libqcu", choices=("auto", "sm60", "sm70", "default"),
                        default="auto",
                        help="native libqcu variant for --backend cuda; "
                             "auto selects by GPU arch")
    parser.add_argument("--quda-variant",
                        choices=("auto", "sm60", "sm70", "default"),
                        default="auto",
                        help="QUDA install variant for --backend quda")
    parser.add_argument("--kupl-ranks", type=int, default=4)
    parser.add_argument("--kupl-threads", type=int, default=4)
    parser.add_argument("--kupl-grid", nargs=4, type=int, default=None)
    parser.add_argument("--kupl-lib", default="default",
                        help="KUPL library: default, lto (build-lto), or a "
                             "path to libqcu.so")
    parser.add_argument("--skip-independent-residual", action="store_true",
                        help="KUPL only: trust the backend fail-closed gate")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    args.lat = [int(value) for value in args.lat]

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        REPO / "data" / f"production_{'x'.join(map(str, args.lat))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output).expanduser() if args.output else _solution_path(
        out_dir, args.backend, args.lat)
    report = Path(args.report).expanduser() if args.report else (
        out_dir / f"report_{args.backend}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"note: overwriting existing solution {output}", file=sys.stderr)
    # Write next to the target and publish atomically only on PASS, so a crash
    # never leaves a truncated HDF5 under the production name.
    partial = output.parent / (output.name + ".partial")

    gauge, source, mass, csw, meta = _resolve_case(args, out_dir)
    print(json.dumps({"backend": args.backend, "gauge": str(gauge),
                      "source": str(source), "solution": str(output),
                      "lattice": args.lat, "mass": mass, "csw": csw,
                      "atol": args.atol}, ensure_ascii=False), flush=True)

    libqcu_variant = None
    quda_variant = None
    kupl_variant = None
    if args.backend in ("cuda", "quda"):
        env, runtime_device = cuda_environment(str(args.device))
        if args.backend == "cuda":
            env = libqcu_environment(env, runtime_device, args.libqcu)
            libqcu_variant = env.get("QCU_LIBCU_VARIANT", "default")
        else:
            env = quda_environment(env, runtime_device, args.quda_variant)
            quda_variant = env.get("QCU_QUDA_VARIANT", "default")
        script = SIMPLE / ("qcu_bistabcg.py" if args.backend == "cuda"
                           else "quda_bistabcg.py")
        command = [sys.executable, str(script),
                   "--lat", *args.lat, "--mass", mass, "--csw", csw,
                   "--gauge", gauge, "--source", source,
                   "--seed", args.seed, "--warmup", "0", "--repeats", "1",
                   "--output", partial, "--report", report]
        if args.backend == "cuda":
            command.extend(["--atol", args.atol, "--device", runtime_device])
        else:
            command.extend(["--tol", args.atol])
            env.setdefault("DEV87_REDUCE_SYNC", "1")
        data = run_report(command, env, report)
    else:
        ranks = int(args.kupl_ranks)
        grid = (tuple(args.kupl_grid) if args.kupl_grid else
                default_kupl_grid(tuple(args.lat), ranks))
        if grid is None:
            parser.error(f"{ranks} ranks cannot split {tuple(args.lat)} into "
                         "even local dimensions; pass --kupl-grid")
        if int(np.prod(grid)) != ranks:
            parser.error(f"--kupl-grid {grid} needs {int(np.prod(grid))} "
                         f"ranks, got {ranks}")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.kupl_threads)
        env = kupl_environment(env, args.kupl_lib)
        kupl_variant = env.get("QCU_KUPL_VARIANT", "default")
        command = kupl_command(args, grid, args.lat, mass, csw, gauge,
                               source, partial, report)
        if args.skip_independent_residual:
            command.append("--skip-independent-residual")
        data = run_report(command, env, report)

    if data.get("status") == "PASS" and partial.is_file():
        os.replace(partial, output)
    else:
        if data.get("status") == "PASS":
            data = dict(data, status="FAIL",
                        error="backend reported PASS but wrote no solution")
        try:
            partial.unlink()
        except FileNotFoundError:
            pass

    summary = {
        "backend": args.backend,
        "solution": str(output),
        "report": str(report),
        "status": data.get("status"),
        "iterations": data.get("iterations"),
        "wall_seconds": data.get("wall_seconds"),
        "solver_seconds": data.get("solver_seconds"),
        "true_residual": data.get("final_true_residual",
                                  data.get("true_residual")),
        "residual_source": data.get("residual_source",
                                    data.get("residual_history_source")),
        "mpi_size": data.get("mpi_size"),
        "grid": data.get("grid"),
        "libqcu_variant": libqcu_variant,
        "quda_variant": quda_variant,
        "kupl_variant": kupl_variant,
        "error": data.get("error"),
    }
    write_json(out_dir / f"last_solve_{args.backend}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
