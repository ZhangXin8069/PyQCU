#!/usr/bin/env python3
"""Collect an honest per-outer-iteration Strict/QUDA Clover-MG trace.

The ordinary formal benchmark remains the performance source of truth.  This
diagnostic runner enables optional solver logging, runs the same formal input
bundle, selects every measured solve after the two warmups, and writes the
raw traces plus a small self-contained SVG comparison plot.  The extra solver
logging means the diagnostic wall times must not be used as the performance
baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCHMARK = HERE / "bench_strict_vs_quda.py"
DEFAULT_OUTPUT = REPO / "data" / "strict_trace_20260902.json"
DEFAULT_PLOT = REPO / "data" / "strict_trace_20260902.svg"
DEFAULT_BENCHMARK_OUTPUT = REPO / "data" / "strict_trace_benchmark_20260902.json"
REFERENCE_BENCHMARK = REPO / "data" / "strict_vs_quda_formal_20260902.json"
WARMUPS = 2


def _next_path(path: Path) -> Path:
    """Do not overwrite an existing diagnostic artifact."""
    path = path.resolve()
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot choose a fresh output path near {path}")


def _json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _run_formal_benchmark(
        benchmark_output: Path, pyqcu_trace: Path, quda_trace: Path,
        repeats: int, timeout: float) -> Dict[str, Any]:
    benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYQCU_STRICT_TRACE_FILE"] = str(pyqcu_trace.resolve())
    env["PYQCU_QUDA_TRACE_FILE"] = str(quda_trace.resolve())
    env.setdefault("QUDA_INSTALL", str(REPO / "data" / "quda-qio-install"))
    env.setdefault("QUDA_PATH", env["QUDA_INSTALL"])
    quda_lib = str(Path(env["QUDA_INSTALL"]) / "lib")
    env["LD_LIBRARY_PATH"] = quda_lib + os.pathsep + env.get(
        "LD_LIBRARY_PATH", "")
    command = [
        sys.executable, "-B", str(BENCHMARK),
        "--profile", "formal", "--side", "both",
        "--cache-expect", "hit", "--repeats", str(repeats),
        "--quda-nullvec-prefix",
        str(REPO / "data" / "L16x32x32x48_nvec12_quda"),
        "--quda-nullvec-manifest",
        str(REPO / "data" / "L16x32x32x48_nvec12_quda.conversion.json"),
        "--output", str(benchmark_output),
    ]
    completed = subprocess.run(
        command, cwd=str(REPO), env=env, text=True,
        capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "formal trace benchmark failed with code "
            f"{completed.returncode}\nstdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}")
    if not benchmark_output.is_file():
        raise RuntimeError(f"benchmark did not produce {benchmark_output}")
    return _json(benchmark_output)


def _parse_pyqcu_trace(path: Path) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("\t")
        if not fields:
            continue
        kind = fields[0]
        if kind == "solve_begin":
            if current is not None:
                raise ValueError(f"nested solve_begin in {path}")
            current = {"rhs_norm": float(fields[1]), "events": []}
        elif kind == "initial_residual":
            if current is None:
                raise ValueError(f"initial residual before solve_begin in {path}")
            current["initial"] = {
                "iteration": int(fields[1]),
                "absolute": float(fields[2]),
                "relative": float(fields[3]),
                "elapsed_seconds": float(fields[4]),
            }
            current["events"].append({"kind": kind, **current["initial"]})
        elif kind == "iteration":
            if current is None:
                raise ValueError(f"iteration before solve_begin in {path}")
            event = {
                "kind": kind,
                "iteration": int(fields[1]),
                "cycle_iteration": int(fields[2]),
                "estimate_absolute": float(fields[3]),
                "estimate_relative": float(fields[4]),
                "arnoldi_next_norm": float(fields[5]),
                "elapsed_seconds": float(fields[6]),
            }
            current["events"].append(event)
        elif kind == "restart_residual":
            if current is None:
                raise ValueError(f"restart residual before solve_begin in {path}")
            event = {
                "kind": kind,
                "iteration": int(fields[1]),
                "absolute": float(fields[2]),
                "relative": float(fields[3]),
                "elapsed_seconds": float(fields[4]),
            }
            current.setdefault("restart_residuals", []).append(event)
            current["events"].append(event)
        elif kind == "solve_end":
            if current is None:
                raise ValueError(f"solve_end before solve_begin in {path}")
            current["end"] = {
                "iterations": int(fields[1]),
                "converged": bool(int(fields[2])),
                "absolute": float(fields[3]),
                "relative": float(fields[4]),
                "elapsed_seconds": float(fields[5]),
            }
            sections.append(current)
            current = None
        elif kind == "trace_version":
            continue
        elif kind.strip():
            raise ValueError(f"unknown Strict trace record {kind!r} in {path}")
    if current is not None:
        raise ValueError(f"unterminated Strict trace section in {path}")
    return sections


_QUDA_ITERATION = re.compile(
    r"GCR:\s*(?P<iteration>\d+)\s+iterations,.*?"
    r"\|r\|/\|b\|\s*=\s*(?P<relative>[0-9.eE+\-]+)")


def _parse_quda_trace(path: Path) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _QUDA_ITERATION.search(line)
        if match is None:
            continue
        iteration = int(match.group("iteration"))
        event = {
            "kind": "iteration",
            "iteration": iteration,
            "relative": float(match.group("relative")),
            "raw": line,
        }
        if iteration == 0:
            if current is not None:
                sections.append(current)
            current = {"events": []}
        if current is None:
            raise ValueError(f"QUDA iteration trace has no iteration zero in {path}")
        current["events"].append(event)
    if current is not None:
        sections.append(current)
    return sections


def _curve_from_pyqcu(section: Mapping[str, Any]) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    initial = section.get("initial")
    if isinstance(initial, Mapping):
        curve.append({
            "iteration": 0,
            "relative": float(initial["relative"]),
            "kind": "initial_true_residual",
        })
    for event in section.get("events", []):
        if event.get("kind") == "iteration":
            curve.append({
                "iteration": int(event["iteration"]),
                "relative": float(event["estimate_relative"]),
                "kind": "arnoldi_estimate",
            })
    return curve


def _curve_from_quda(section: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "iteration": int(event["iteration"]),
            "relative": float(event["relative"]),
            "kind": "gcr_iterated_residual",
        }
        for event in section.get("events", [])
    ]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take median of empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _side_trace(
        side: str, benchmark: Mapping[str, Any], sections: List[Dict[str, Any]],
        repeats: int, *, pyqcu: bool) -> Dict[str, Any]:
    record = benchmark["sides"][side]
    if record.get("status") != "ok":
        raise RuntimeError(f"{side} diagnostic benchmark status={record.get('status')}")
    expected = list(record["iterations"]["samples"])
    if len(sections) < WARMUPS + repeats:
        raise RuntimeError(
            f"{side} trace has {len(sections)} solves, expected at least "
            f"{WARMUPS + repeats}")
    steady_sections = sections[WARMUPS:WARMUPS + repeats]
    steady: List[Dict[str, Any]] = []
    for index, section in enumerate(steady_sections):
        curve = _curve_from_pyqcu(section) if pyqcu else _curve_from_quda(section)
        if not curve:
            raise RuntimeError(f"{side} steady trace {index} has no residual events")
        observed_iterations = (
            int(section.get("end", {}).get("iterations", -1))
            if pyqcu else int(curve[-1]["iteration"]))
        if observed_iterations != int(expected[index]):
            raise RuntimeError(
                f"{side} trace iteration mismatch at steady solve {index}: "
                f"trace={observed_iterations}, benchmark={expected[index]}")
        item: Dict[str, Any] = {
            "solve_index": WARMUPS + index,
            "iterations": observed_iterations,
            "residual_curve": curve,
            "events": section.get("events", []),
        }
        if pyqcu:
            item["trace_elapsed_seconds"] = float(
                section["end"]["elapsed_seconds"])
            item["restart_residuals"] = section.get("restart_residuals", [])
        steady.append(item)
    timing = record["timing"]["steady"]
    timing_samples = [float(value) for value in timing["samples_seconds"]]
    iteration_samples = [int(value) for value in expected]
    median_seconds = float(timing["median_seconds"])
    median_iterations = _median([float(value) for value in iteration_samples])
    return {
        "side": side,
        "diagnostic_trace_sections": len(sections),
        "warmups_excluded": WARMUPS,
        "steady": steady,
        "timing_reference": {
            "samples_seconds": timing_samples,
            "median_seconds": median_seconds,
            "mad_seconds": float(timing["mad_seconds"]),
            "median_iterations": median_iterations,
            "average_seconds_per_outer_iteration": (
                median_seconds / median_iterations),
            "average_milliseconds_per_outer_iteration": (
                1000.0 * median_seconds / median_iterations),
            "source": "un-instrumented formal benchmark document",
        },
    }


def _svg_text(x: float, y: float, value: str, *, size: int = 14,
              fill: str = "#202124", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" '
        f'fill="{fill}" text-anchor="{anchor}">{html.escape(value)}</text>')


def _make_plot(document: Mapping[str, Any], path: Path) -> None:
    width, height = 1100, 760
    left, right, top = 90, 35, 65
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, 480
    max_iteration = max(
        int(point["iteration"])
        for side in ("pyqcu", "quda")
        for solve in document["sides"][side]["steady"]
        for point in solve["residual_curve"])
    min_positive = min(
        max(float(point["relative"]), 1.0e-16)
        for side in ("pyqcu", "quda")
        for solve in document["sides"][side]["steady"]
        for point in solve["residual_curve"])
    y_max_log = 0.1
    y_min_log = int(min(-8.0, math.floor(math.log10(min_positive)) - 1))

    def xcoord(iteration: int) -> float:
        return plot_left + (plot_right - plot_left) * iteration / max_iteration

    def ycoord(value: float) -> float:
        log_value = math.log10(max(value, 10.0 ** y_min_log))
        return plot_top + (plot_bottom - plot_top) * (
            (y_max_log - log_value) / (y_max_log - y_min_log))

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, "Strict Clover-MG / QUDA outer residual trace",
                  size=20, anchor="middle"),
    ]
    for exponent in range(math.floor(y_min_log), 1):
        value = 10.0 ** exponent
        y = ycoord(value)
        pieces.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(_svg_text(plot_left - 10, y + 5, f"1e{exponent}",
                                 size=12, anchor="end"))
    for iteration in range(0, max_iteration + 1, max(1, max_iteration // 8)):
        x = xcoord(iteration)
        pieces.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
            f'y2="{plot_bottom}" stroke="#f3f4f6"/>')
        pieces.append(_svg_text(x, plot_bottom + 22, str(iteration),
                                 size=12, anchor="middle"))
    pieces.extend([
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
        f'y2="{plot_bottom}" stroke="#374151"/>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
        f'y2="{plot_bottom}" stroke="#374151"/>',
        _svg_text(18, (plot_top + plot_bottom) / 2, "相对迭代残差",
                  size=14),
        _svg_text((plot_left + plot_right) / 2, plot_bottom + 45,
                  "外层迭代编号", size=14, anchor="middle"),
    ])
    colors = {"pyqcu": "#1769aa", "quda": "#c2410c"}
    labels = {"pyqcu": "PyQCU Strict (Arnoldi estimate)",
              "quda": "QUDA GCR (iterated residual)"}
    for side in ("pyqcu", "quda"):
        for solve in document["sides"][side]["steady"]:
            points = solve["residual_curve"]
            coords = " ".join(
                f"{xcoord(int(point['iteration'])):.1f},{ycoord(float(point['relative'])):.1f}"
                for point in points)
            pieces.append(
                f'<polyline points="{coords}" fill="none" stroke="{colors[side]}" '
                'stroke-width="2" opacity="0.35"/>')
        pieces.append(
            f'<line x1="{plot_right - 270}" y1="{top - 25}" '
            f'x2="{plot_right - 240}" y2="{top - 25}" stroke="{colors[side]}" '
            'stroke-width="3"/>')
        pieces.append(_svg_text(plot_right - 230, top - 20, labels[side],
                                 size=13))

    bar_top, bar_bottom = 580, 705
    bar_values = [
        ("PyQCU", float(document["sides"]["pyqcu"]["timing_reference"]["median_seconds"]), colors["pyqcu"]),
        ("QUDA", float(document["sides"]["quda"]["timing_reference"]["median_seconds"]), colors["quda"]),
    ]
    max_time = max(value for _, value, _ in bar_values) * 1.25
    pieces.append(_svg_text(plot_left, bar_top - 18,
                            "无 trace 正式 steady median wall time（秒）", size=15))
    for index, (label, value, color) in enumerate(bar_values):
        bar_width = 230
        x = plot_left + 100 + index * 340
        bar_height = (bar_bottom - bar_top) * value / max_time
        y = bar_bottom - bar_height
        pieces.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" '
            f'height="{bar_height:.1f}" fill="{color}" opacity="0.85"/>')
        pieces.append(_svg_text(x + bar_width / 2, bar_bottom + 22, label,
                                 size=14, anchor="middle"))
        pieces.append(_svg_text(x + bar_width / 2, y - 8,
                                f"{value:.6f} s", size=14, anchor="middle"))
    pieces.append(_svg_text(
        plot_left + 690, bar_top + 25,
        "每外层迭代平均：PyQCU "
        f"{document['sides']['pyqcu']['timing_reference']['average_milliseconds_per_outer_iteration']:.3f} ms；"
        " QUDA "
        f"{document['sides']['quda']['timing_reference']['average_milliseconds_per_outer_iteration']:.3f} ms",
        size=13))
    pieces.append(_svg_text(
        plot_left + 690, bar_top + 50,
        "曲线取 5 次 steady trace；性能柱取同输入无 trace 正式 benchmark",
        size=13))
    pieces.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and parse a detailed Clover MultiGrid iteration trace")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--benchmark-output", type=Path,
                        default=DEFAULT_BENCHMARK_OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_BENCHMARK)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-run", action="store_true",
                        help="parse existing trace files beside --benchmark-output")
    parser.add_argument("--pyqcu-trace", type=Path, default=None)
    parser.add_argument("--quda-trace", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    output = _next_path(args.output)
    plot = _next_path(args.plot)
    benchmark_output = (args.benchmark_output.resolve() if args.no_run else
                        _next_path(args.benchmark_output))
    pyqcu_trace = (args.pyqcu_trace or output.with_name(
        output.stem + "_pyqcu.tsv")).resolve()
    quda_trace = (args.quda_trace or output.with_name(
        output.stem + "_quda.log")).resolve()
    if not args.no_run:
        benchmark = _run_formal_benchmark(
            benchmark_output, pyqcu_trace, quda_trace,
            args.repeats, args.timeout)
    else:
        benchmark = _json(benchmark_output)
    if args.reference.is_file():
        reference = _json(args.reference)
    else:
        reference = benchmark
    if reference.get("protocol", {}).get("config_hash") != benchmark.get(
            "protocol", {}).get("config_hash"):
        raise RuntimeError("reference and diagnostic benchmark config hashes differ")
    if reference.get("input_fingerprints", {}).get("bundle_hash") != benchmark.get(
            "input_fingerprints", {}).get("bundle_hash"):
        raise RuntimeError("reference and diagnostic benchmark input bundles differ")
    pyqcu_sections = _parse_pyqcu_trace(pyqcu_trace)
    quda_sections = _parse_quda_trace(quda_trace)
    sides = {
        "pyqcu": _side_trace(
            "pyqcu", reference, pyqcu_sections, args.repeats, pyqcu=True),
        "quda": _side_trace(
            "quda", reference, quda_sections, args.repeats, pyqcu=False),
    }
    document: Dict[str, Any] = {
        "schema": {"name": "pyqcu.strict-vs-quda.iteration-trace", "version": 1},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "benchmark_output": str(benchmark_output.resolve()),
            "reference_benchmark": str(args.reference.resolve()),
            "trace_logging": "enabled; diagnostic wall times excluded from performance claims",
            "repeats": args.repeats,
            "warmups": WARMUPS,
            "pyqcu_trace": str(pyqcu_trace),
            "quda_trace": str(quda_trace),
        },
        "protocol": benchmark.get("protocol"),
        "input_fingerprints": benchmark.get("input_fingerprints"),
        # The no-trace reference is the only performance comparison.  Keep the
        # verbose diagnostic benchmark separate because logging changes its
        # wall time.
        "comparison": reference.get("comparison"),
        "diagnostic_comparison": benchmark.get("comparison"),
        "sides": sides,
        "provenance": {
            "collector_commit": (
                benchmark.get("collector", {}).get("git", {}).get("commit")),
            "pyqcu_runtime": (
                reference.get("sides", {}).get("pyqcu", {}).get("provenance", {}).get("runtime")),
            "quda_runtime": (
                reference.get("sides", {}).get("quda", {}).get("provenance", {}).get("runtime")),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    _make_plot(document, plot)
    print(json.dumps({
        "trace": str(output.resolve()),
        "plot": str(plot.resolve()),
        "benchmark": str(benchmark_output.resolve()),
        "pyqcu_sections": len(pyqcu_sections),
        "quda_sections": len(quda_sections),
        "pyqcu_iterations": [item["iterations"] for item in sides["pyqcu"]["steady"]],
        "quda_iterations": [item["iterations"] for item in sides["quda"]["steady"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"trace_strict_vs_quda: {exc}", file=sys.stderr)
        raise SystemExit(1)
