#!/usr/bin/env python3
"""Measure CUDA QCU, KUPL QCU and QUDA on the same Clover BiCGStab case.

This is the *benchmark* entry point.  For production solves use
``solve_case.py``; both scripts share ``prepare_case.py`` and ``_runner.py``.

The benchmark

* prepares the gauge and source with ``prepare_case`` (same manifest/SHA256),
* runs CUDA QCU (standalone BiCGStab and fine-only legacy MG),
  KUPL QCU (optimized parity-Schur BiCGStab, ``--restart 0``) and
  QUDA (Clover BiCGStab),
* warms the GPU clocks, discards one warm-up solve and reports the median of
  ``--repeats`` timed solves with independent true residuals.

Example::

    source ./env.sh
    python pyqcu/testing/simple/compare_three_way.py \\
        --lat 24 24 24 64 --kupl-ranks 4 --kupl-threads 4

Sub-volume at the same mass and Csw (controlled size/parallel study; the
seams are not periodic, so this is not a physics ensemble)::

    python pyqcu/testing/simple/compare_three_way.py \\
        --lat 16 16 16 32 --kupl-ranks 4 --kupl-threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import median

import numpy as np

try:
    from ._common import (DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                          lime_metadata, resolve_path)
    from ._runner import (REPO, cuda_environment, default_kupl_grid,
                          gpu_clock_warmup, kupl_command, kupl_environment,
                          libqcu_environment, quda_environment, run_report)
    from .prepare_case import prepare_gauge, prepare_source
except ImportError:
    from _common import (DEFAULT_ATOL, DEFAULT_CSW, DEFAULT_MASS,
                         lime_metadata, resolve_path)
    from _runner import (REPO, cuda_environment, default_kupl_grid,
                         gpu_clock_warmup, kupl_command, kupl_environment,
                         libqcu_environment, quda_environment, run_report)
    from prepare_case import prepare_gauge, prepare_source

SIMPLE = Path(__file__).resolve().parent
DEFAULT_LIME = Path.home() / "data" / (
    "beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime")


def _lattice(value) -> tuple[int, int, int, int]:
    dims = tuple(int(item) for item in value)
    if len(dims) != 4:
        raise ValueError("lattice must have four dimensions")
    return dims


def run_cuda(args, out_dir: Path, gauge: Path, source: Path, mass: float,
             csw: float) -> list[dict]:
    env, device = cuda_environment(str(args.device))
    env = libqcu_environment(env, device, args.libqcu)
    command = [sys.executable, str(SIMPLE / "qcu_bistabcg.py"),
               "--lat", *args.lat, "--mass", mass, "--csw", csw,
               "--atol", args.atol, "--gauge", gauge, "--source", source,
               "--device", device, "--seed", args.seed,
               "--warmup", args.warmup, "--repeats", args.repeats]
    results = []
    report = out_dir / "cuda_bicg.json"
    data = run_report(command + ["--report", report], env, report)
    data["label"] = "CUDA QCU BiCGStab"
    data["libqcu_variant"] = env.get("QCU_LIBCU_VARIANT", "default")
    results.append(data)
    mg_command = [sys.executable, str(SIMPLE / "qcu_multigrid.py"),
                  "--lat", *args.lat, "--mass", mass, "--csw", csw,
                  "--atol", args.atol, "--levels", "1", "--gauge", gauge,
                  "--source", source, "--device", device,
                  "--warmup", args.warmup, "--repeats", args.repeats]
    report = out_dir / "cuda_mg_fine_only.json"
    data = run_report(mg_command + ["--report", report], env, report)
    data["label"] = "CUDA QCU MG (fine-only)"
    data["libqcu_variant"] = env.get("QCU_LIBCU_VARIANT", "default")
    results.append(data)
    return results


def run_quda(args, out_dir: Path, gauge: Path, source: Path, mass: float,
             csw: float) -> list[dict]:
    env, device = cuda_environment(str(args.device))
    env = quda_environment(env, device, args.quda_variant)
    env.setdefault("DEV87_REDUCE_SYNC", "1")
    command = [sys.executable, str(SIMPLE / "quda_bistabcg.py"),
               "--lat", *args.lat, "--mass", mass, "--csw", csw,
               "--tol", args.atol, "--gauge", gauge, "--source", source,
               "--seed", args.seed, "--warmup", args.warmup,
               "--repeats", args.repeats]
    report = out_dir / "quda_bicg.json"
    data = run_report(command + ["--report", report], env, report)
    data["label"] = "QUDA Clover BiCGStab"
    data["quda_variant"] = env.get("QCU_QUDA_VARIANT", "default")
    return [data]


def run_kupl(args, out_dir: Path, gauge: Path, source: Path, mass: float,
             csw: float) -> list[dict]:
    ranks = int(args.kupl_ranks)
    grid = (tuple(args.kupl_grid) if args.kupl_grid else
            default_kupl_grid(tuple(args.lat), ranks))
    if grid is None:
        print(f"KUPL: {ranks} ranks cannot split {tuple(args.lat)} into even "
              "local dimensions; falling back to 1 rank")
        ranks, grid = 1, (1, 1, 1, 1)
    if int(np.prod(grid)) != ranks:
        raise ValueError(f"KUPL grid {grid} needs {np.prod(grid)} ranks, "
                         f"got --kupl-ranks {ranks}")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.kupl_threads)
    env = kupl_environment(env, args.kupl_lib)
    samples = []
    last = None
    for repeat in range(args.warmup + args.repeats):
        report = out_dir / f"kupl_bicg_r{repeat + 1}.json"
        command = kupl_command(args, grid, args.lat, mass, csw, gauge,
                               source, None, report)
        # Keep the independent distributed full residual in the timed report.
        # Its current 4-rank cost is small, and a null residual makes a formal
        # solver comparison numerically incomplete.
        data = run_report(command, env, report)
        if data.get("status") == "FAIL":
            data["label"] = "KUPL QCU parity-Schur BiCGStab"
            return [data]
        last = data
        if repeat >= args.warmup:
            samples.append(float(data["wall_seconds"]))
    last = dict(last)
    last["label"] = "KUPL QCU parity-Schur BiCGStab"
    last["wall_seconds"] = median(samples)
    last["wall_samples"] = samples
    last["kupl_variant"] = env.get("QCU_KUPL_VARIANT", "default")
    return [last]


def summarize(results: list[dict]) -> list[dict]:
    quda = next((item for item in results
                 if item.get("label", "").startswith("QUDA")), None)
    baseline = None
    if quda is not None and quda.get("status") != "FAIL":
        baseline = float(quda.get("solver_seconds") or
                         quda.get("wall_seconds"))
    table = []
    for item in results:
        if item.get("status") == "FAIL":
            table.append({"label": item.get("label", "unknown"),
                          "status": "FAIL", "error": item.get("error")})
            continue
        wall = float(item.get("wall_seconds", float("nan")))
        # CUDA and KUPL report solve-only wall time; QUDA additionally
        # exposes its internal solver timer.  Use the solver timer when
        # present so the speed-up is computed on the same footing.
        effective = float(item.get("solver_seconds") or wall)
        row = {
            "label": item["label"],
            "wall_seconds": wall,
            "solver_seconds": item.get("solver_seconds"),
            "effective_seconds": effective,
            "iterations": item.get("iterations"),
            "true_residual": item.get("final_true_residual",
                                      item.get("true_residual")),
            "status": item.get("status"),
            "speedup_vs_quda": (baseline / effective
                                if baseline and effective > 0 else None),
        }
        table.append(row)
    return table


def markdown_table(rows: list[dict]) -> str:
    lines = ["| backend | solve (s) | wall (s) | iterations | true residual | vs QUDA |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] == "FAIL":
            lines.append(f"| {row['label']} | FAIL | - | - | - | - |")
            continue
        residual = row.get("true_residual")
        residual = "-" if residual is None else f"{residual:.3e}"
        speedup = row.get("speedup_vs_quda")
        speedup = "-" if speedup is None else f"{speedup:.3f}x"
        lines.append(
            f"| {row['label']} | {row['effective_seconds']:.6f} | "
            f"{row['wall_seconds']:.6f} | {row['iterations']} | "
            f"{residual} | {speedup} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", nargs=4, type=int, default=(24, 24, 24, 64))
    parser.add_argument("--lime", default=str(DEFAULT_LIME))
    parser.add_argument("--gauge", default=None,
                        help="pre-made QCU HDF5 gauge; overrides --lime")
    parser.add_argument("--source", default=None,
                        help="pre-made source HDF5 [4,3,x,y,z,t]")
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--csw", type=float, default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--libqcu", choices=("auto", "sm60", "sm70", "default"),
                        default="auto",
                        help="native libqcu variant; auto selects by GPU arch")
    parser.add_argument("--quda-variant",
                        choices=("auto", "sm60", "sm70", "default"),
                        default="auto",
                        help="QUDA install variant; auto selects by GPU arch")
    parser.add_argument("--gpu-warmup-seconds", type=float, default=3.0,
                        help="GPU clock-ramp burn before timed solves; 0 disables")
    parser.add_argument("--backends", default="cuda,quda,kupl",
                        help="comma-separated subset of cuda,quda,kupl")
    parser.add_argument("--kupl-ranks", type=int, default=4)
    parser.add_argument("--kupl-threads", type=int, default=4)
    parser.add_argument("--kupl-grid", nargs=4, type=int, default=None)
    parser.add_argument("--kupl-lib", default="default",
                        help="KUPL library: default, lto (build-lto), or a "
                             "path to libqcu.so")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    args.lat = list(_lattice(args.lat))
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats >= 1")

    lime = resolve_path(args.lime, Path(args.lime).expanduser())
    meta = lime_metadata(lime) if lime.is_file() else {}
    mass = (float(args.mass) if args.mass is not None else
            float(meta.get("reference_mass", DEFAULT_MASS)))
    csw = (float(args.csw) if args.csw is not None else
           float(meta.get("csw", DEFAULT_CSW)))
    # Child runners resolve relative report paths against DATA_DIR.  Pass an
    # absolute output directory so a caller-supplied repository-relative path
    # such as ``logs/...`` is not silently rewritten to ``data/logs/...``.
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else REPO / "data" / "benchmark_three_way")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gauge is not None:
        gauge = resolve_path(args.gauge, Path(args.gauge).expanduser())
        if not gauge.is_file():
            raise FileNotFoundError(gauge)
    else:
        if not lime.is_file():
            raise FileNotFoundError(lime)
        gauge = prepare_gauge(lime, args.lat, out_dir, mass, csw)
    if args.source is not None:
        source = resolve_path(args.source, Path(args.source).expanduser())
        if not source.is_file():
            raise FileNotFoundError(source)
    else:
        source = prepare_source(args.lat, out_dir, args.seed)
    print(f"gauge: {gauge}\nsource: {source}", flush=True)

    selected = {name.strip() for name in args.backends.split(",") if name.strip()}
    if "all" in selected:
        selected = {"cuda", "quda", "kupl"}
    unknown = selected - {"cuda", "quda", "kupl"}
    if unknown:
        parser.error(f"unknown backends: {', '.join(sorted(unknown))}")
    if selected & {"cuda", "quda"}:
        env, burn_device = cuda_environment(str(args.device))
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"):
                if env.get(key):
                    os.environ[key] = env[key]
        gpu_clock_warmup(args.gpu_warmup_seconds, burn_device)
        # The burn exported the physical -> logical mapping into os.environ;
        # later ``cuda_environment`` calls must not remap the already logical
        # index (that would ask for an ordinal outside the visible set).
        args.device = burn_device
    results: list[dict] = []
    if "cuda" in selected:
        results.extend(run_cuda(args, out_dir, gauge, source, mass, csw))
    if "quda" in selected:
        results.extend(run_quda(args, out_dir, gauge, source, mass, csw))
    if "kupl" in selected:
        results.extend(run_kupl(args, out_dir, gauge, source, mass, csw))

    rows = summarize(results)
    summary = {
        "lattice": args.lat, "mass": mass, "csw": csw,
        "atol": args.atol, "seed": args.seed,
        "warmup": args.warmup, "repeats": args.repeats,
        "gauge": str(gauge), "source": str(source),
        "gauge_note": (
            "user-provided gauge file" if args.gauge is not None else
            "reference LIME" if tuple(args.lat) ==
            tuple(meta.get("lattice_xyzt", ())) else
            "sub-volume of the reference LIME; seams are not periodic"),
        "device": args.device,
        "kupl": {"ranks": args.kupl_ranks, "threads": args.kupl_threads,
                 "lib": next((row.get("kupl_variant") for row in results
                              if row.get("kupl_variant")), "default")},
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    table = markdown_table(rows)
    (out_dir / "summary.md").write_text(
        f"# Three-way Clover BiCGStab benchmark\n\n"
        f"- lattice: `{'x'.join(map(str, args.lat))}`\n"
        f"- mass: `{mass}`, csw: `{csw}`, atol: `{args.atol}`, seed: `{args.seed}`\n"
        f"- warmup: {args.warmup}, timed repeats: {args.repeats}\n"
        f"- gauge: `{gauge}`\n\n{table}\n", encoding="utf-8")
    print(table)
    print(f"\nsummary: {summary_path}")
    return 0 if all(row["status"] != "FAIL" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
