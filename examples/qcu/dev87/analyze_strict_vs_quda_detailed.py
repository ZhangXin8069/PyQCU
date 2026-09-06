#!/usr/bin/env python3
"""Validate and expand a formal Strict/QUDA MultiGrid iteration trace.

The formal benchmark is the timing source of truth.  The trace collector adds
solver logging, so its wall times are used only for diagnostic event timing.
This script joins the two artifacts, checks their protocol and input identity,
and emits one row per outer iteration together with a compact SVG comparison.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_TRACE = REPO / "data" / "strict_trace_20260906.json"
DEFAULT_BENCHMARK = REPO / "data" / "strict_vs_quda_formal_20260906.json"
DEFAULT_OUTPUT = REPO / "data" / "strict_trace_detailed_20260906.json"
DEFAULT_CSV = REPO / "data" / "strict_trace_detailed_20260906.csv"
DEFAULT_PLOT = REPO / "data" / "strict_trace_detailed_20260906.svg"
DEFAULT_SUMMARY = REPO / "data" / "strict_trace_detailed_20260906.md"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是 dict: {path}")
    return value


def _positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} 不是有限非负数: {value!r}")
    return result


def _median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("空序列不能求 median")
    return float(median(items))


def _curve_points(section: Mapping[str, Any]) -> List[Dict[str, Any]]:
    points = section.get("residual_curve")
    if not isinstance(points, list) or not points:
        raise ValueError("steady trace 缺少 residual_curve")
    result = []
    previous = -1
    for point in points:
        iteration = int(point["iteration"])
        relative = _positive(point["relative"], "relative residual")
        if iteration <= previous:
            raise ValueError("residual_curve 的 iteration 必须严格递增")
        if relative == 0.0:
            relative = 1.0e-300
        result.append({
            "iteration": iteration,
            "relative": relative,
            "kind": str(point.get("kind", "unknown")),
        })
        previous = iteration
    if result[0]["iteration"] != 0:
        raise ValueError("residual_curve 必须从 iteration=0 开始")
    return result


def _trace_sections(trace: Mapping[str, Any], side: str) -> List[Mapping[str, Any]]:
    record = trace.get("sides", {}).get(side)
    if not isinstance(record, Mapping):
        raise ValueError(f"trace 缺少 side={side}")
    steady = record.get("steady")
    if not isinstance(steady, list) or not steady:
        raise ValueError(f"trace side={side} 缺少 steady sections")
    return steady


def _validate_identity(trace: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    trace_protocol = trace.get("protocol", {})
    bench_protocol = benchmark.get("protocol", {})
    if trace_protocol.get("config_hash") != bench_protocol.get("config_hash"):
        raise ValueError("trace 与正式 benchmark 的 config_hash 不一致")
    trace_inputs = trace.get("input_fingerprints", {})
    bench_inputs = benchmark.get("input_fingerprints", {})
    if trace_inputs.get("bundle_hash") != bench_inputs.get("bundle_hash"):
        raise ValueError("trace 与正式 benchmark 的 bundle_hash 不一致")
    comparison = benchmark.get("comparison", {})
    if comparison.get("status") != "pass" or comparison.get("fair") is not True:
        raise ValueError(f"正式 benchmark 不是 fair pass: {comparison!r}")
    protocol = bench_protocol
    expected = {
        "lattice_xyzt": [16, 32, 32, 48],
        "mass": 0.05,
        "nvec": 12,
        "coarse_spin": 2,
        "target_parity": 1,
        "levels": 2,
    }
    for key, wanted in expected.items():
        if protocol.get(key) != wanted:
            raise ValueError(f"正式协议 {key}={protocol.get(key)!r}，期望 {wanted!r}")


def _join(trace: Mapping[str, Any], benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_identity(trace, benchmark)
    sides: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for side in ("pyqcu", "quda"):
        trace_steady = _trace_sections(trace, side)
        bench_side = benchmark.get("sides", {}).get(side)
        if not isinstance(bench_side, Mapping) or bench_side.get("status") != "ok":
            raise ValueError(f"正式 benchmark side={side} 不是 ok")
        bench_timing = bench_side["timing"]["steady"]
        timing_samples = [float(x) for x in bench_timing["samples_seconds"]]
        iteration_samples = [int(x) for x in bench_side["iterations"]["samples"]]
        if len(trace_steady) != len(timing_samples):
            raise ValueError(
                f"{side} trace steady 数={len(trace_steady)}，timing 样本数={len(timing_samples)}")
        if len(iteration_samples) != len(trace_steady):
            raise ValueError(f"{side} iteration 样本数与 trace 数不一致")

        solves: List[Dict[str, Any]] = []
        for solve_index, (section, wall, expected_iterations) in enumerate(
                zip(trace_steady, timing_samples, iteration_samples)):
            curve = _curve_points(section)
            observed_iterations = int(section["iterations"])
            if observed_iterations != expected_iterations:
                raise ValueError(
                    f"{side} solve={solve_index} trace iterations={observed_iterations} "
                    f"benchmark iterations={expected_iterations}")
            if curve[-1]["iteration"] != observed_iterations:
                raise ValueError(f"{side} solve={solve_index} 曲线终点与迭代数不一致")
            average_ms = 1000.0 * wall / observed_iterations
            decrease = curve[0]["relative"] / curve[-1]["relative"]
            log10_per_iter = (
                math.log10(curve[-1]["relative"]) -
                math.log10(curve[0]["relative"])) / observed_iterations
            solve_record = {
                "solve_index": solve_index,
                "wall_seconds": wall,
                "iterations": observed_iterations,
                "average_milliseconds_per_outer_iteration": average_ms,
                "residual_reduction_factor": decrease,
                "log10_residual_change_per_iteration": log10_per_iter,
                "residual_kind": curve[0]["kind"] if len(curve) == 1 else curve[1]["kind"],
                "residual_curve": curve,
                "diagnostic_trace_elapsed_seconds": section.get("trace_elapsed_seconds"),
                "restart_residuals": section.get("restart_residuals", []),
            }
            solves.append(solve_record)
            for point in curve:
                event = {
                    "side": side,
                    "solve_index": solve_index,
                    "iteration": point["iteration"],
                    "relative_residual": point["relative"],
                    "residual_kind": point["kind"],
                    "wall_seconds": wall,
                    "iterations_total": observed_iterations,
                    "average_milliseconds_per_outer_iteration": average_ms,
                    "cycle_iteration": None,
                    "diagnostic_elapsed_seconds": None,
                }
                for raw in section.get("events", []):
                    if (raw.get("kind") == "iteration" and
                            int(raw.get("iteration", -1)) == point["iteration"]):
                        event["cycle_iteration"] = raw.get("cycle_iteration")
                        event["diagnostic_elapsed_seconds"] = raw.get("elapsed_seconds")
                        break
                rows.append(event)

        residual_values = [
            float(point["relative"])
            for solve in solves
            for point in solve["residual_curve"]
        ]
        sides[side] = {
            "steady_solve_count": len(solves),
            "timing_source": "strict_vs_quda_formal_20260906.json (no trace)",
            "residual_source": "strict_trace_20260906.json (trace logging enabled)",
            "median_wall_seconds": float(bench_timing["median_seconds"]),
            "mad_wall_seconds": float(bench_timing["mad_seconds"]),
            "median_iterations": _median(iteration_samples),
            "median_average_milliseconds_per_outer_iteration": _median(
                item["average_milliseconds_per_outer_iteration"] for item in solves),
            "iterations": iteration_samples,
            "full_operator_true_residual_rel": list(
                bench_side["true_residual"]["samples_rel"]),
            "solves": solves,
            "reported_residual_semantics": (
                "PyQCU Arnoldi least-squares estimate" if side == "pyqcu" else
                "QUDA GCR iterated residual"),
            "residual_min": min(residual_values),
            "residual_max": max(residual_values),
        }
    return {
        "schema": {"name": "pyqcu.strict-vs-quda.detailed-trace", "version": 1},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": benchmark["protocol"],
        "input_fingerprints": benchmark["input_fingerprints"],
        "comparison": benchmark["comparison"],
        "sources": {
            "trace": str(DEFAULT_TRACE.resolve()),
            "benchmark": str(DEFAULT_BENCHMARK.resolve()),
            "timing_rule": "仅使用无 trace benchmark 的 steady samples",
            "residual_rule": "仅使用 trace 的逐外层迭代曲线；最终正确性使用 full-op true residual",
        },
        "sides": sides,
        "rows": rows,
    }


def _write_csv(document: Mapping[str, Any], path: Path) -> None:
    fields = [
        "side", "solve_index", "iteration", "relative_residual", "residual_kind",
        "wall_seconds", "iterations_total",
        "average_milliseconds_per_outer_iteration", "cycle_iteration",
        "diagnostic_elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(document["rows"])


def _svg_text(x: float, y: float, value: str, *, size: int = 14,
              anchor: str = "start", fill: str = "#202124") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" '
            f'fill="{fill}" text-anchor="{anchor}">{html.escape(value)}</text>')


def _plot(document: Mapping[str, Any], path: Path) -> None:
    width, height = 1200, 820
    left, right, top, bottom = 95, 40, 70, 505
    plot_right = width - right
    colors = {"pyqcu": "#1769aa", "quda": "#c2410c"}
    labels = {"pyqcu": "PyQCU Strict（Arnoldi estimate）",
              "quda": "QUDA（GCR iterated residual）"}
    all_points = [
        point
        for side in ("pyqcu", "quda")
        for solve in document["sides"][side]["solves"]
        for point in solve["residual_curve"]
    ]
    max_iteration = max(int(point["iteration"]) for point in all_points)
    min_residual = min(float(point["relative"]) for point in all_points)
    y_min = min(-8.0, math.floor(math.log10(min_residual)) - 1)
    y_max = 0.1

    def xcoord(iteration: int) -> float:
        return left + (plot_right - left) * iteration / max(1, max_iteration)

    def ycoord(value: float) -> float:
        log_value = math.log10(max(value, 10.0 ** y_min))
        return top + (bottom - top) * (y_max - log_value) / (y_max - y_min)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 32, "Formal Strict Clover-MG / QUDA 逐外层迭代对照",
                  size=21, anchor="middle"),
    ]
    for exponent in range(math.floor(y_min), 1):
        value = 10.0 ** exponent
        y = ycoord(value)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
                      'stroke="#e5e7eb"/>')
        pieces.append(_svg_text(left - 10, y + 5, f"1e{exponent}", size=12, anchor="end"))
    for iteration in range(0, max_iteration + 1, max(1, max_iteration // 8)):
        x = xcoord(iteration)
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                      'stroke="#f3f4f6"/>')
        pieces.append(_svg_text(x, bottom + 22, str(iteration), size=12, anchor="middle"))
    pieces.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{plot_right}" y2="{bottom}" stroke="#374151"/>',
        _svg_text(20, (top + bottom) / 2, "报告残差（对数轴）", size=14),
        _svg_text((left + plot_right) / 2, bottom + 45, "外层迭代编号", size=14, anchor="middle"),
    ])
    for side in ("pyqcu", "quda"):
        for solve in document["sides"][side]["solves"]:
            points = solve["residual_curve"]
            coords = " ".join(
                f'{xcoord(int(point["iteration"])):.1f},{ycoord(float(point["relative"])):.1f}'
                for point in points)
            pieces.append(f'<polyline points="{coords}" fill="none" stroke="{colors[side]}" '
                          'stroke-width="2" opacity="0.30"/>')
        pieces.append(f'<line x1="{plot_right - 330 if side == "pyqcu" else plot_right - 140}" '
                      f'y1="{top - 25}" x2="{plot_right - 300 if side == "pyqcu" else plot_right - 110}" '
                      f'y2="{top - 25}" stroke="{colors[side]}" stroke-width="3"/>')
        pieces.append(_svg_text(plot_right - 290 if side == "pyqcu" else plot_right - 100,
                                top - 20, labels[side], size=13))

    bar_top, bar_bottom = 635, 770
    values = [
        ("PyQCU", float(document["sides"]["pyqcu"]["median_average_milliseconds_per_outer_iteration"]), colors["pyqcu"]),
        ("QUDA", float(document["sides"]["quda"]["median_average_milliseconds_per_outer_iteration"]), colors["quda"]),
    ]
    max_value = max(value for _, value, _ in values) * 1.25
    pieces.append(_svg_text(left, bar_top - 20, "无 trace 正式 benchmark：单次外层迭代平均时间（ms）", size=15))
    for index, (label, value, color) in enumerate(values):
        bar_width = 230
        x = left + 125 + index * 360
        bar_height = (bar_bottom - bar_top) * value / max_value
        y = bar_bottom - bar_height
        pieces.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
                      f'fill="{color}" opacity="0.85"/>')
        pieces.append(_svg_text(x + bar_width / 2, bar_bottom + 22, label, size=14, anchor="middle"))
        pieces.append(_svg_text(x + bar_width / 2, y - 8, f"{value:.3f} ms", size=14, anchor="middle"))
    pieces.append(_svg_text(left + 690, bar_top + 28,
                            "计时：无日志正式 benchmark；曲线：独立 trace 运行", size=13))
    pieces.append(_svg_text(left + 690, bar_top + 53,
                            "PyQCU/QUDA 的残差标量定义不同，最终正确性看 full-op true residual", size=13))
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _write_summary(document: Mapping[str, Any], path: Path) -> None:
    protocol = document["protocol"]
    lines = [
        "# 20260906 Strict MultiGrid 逐迭代对照摘要",
        "",
        "本文件由 `analyze_strict_vs_quda_detailed.py` 生成。性能时间来自无 trace 正式 benchmark；",
        "逐迭代残差来自开启日志的独立 trace。两者通过 `config_hash` 与 `bundle_hash` 校验。",
        "",
        f"- 格点：`{protocol['lattice_xyzt']}`；mass=`{protocol['mass']}`；nvec=`{protocol['nvec']}`；"
        f"coarse_spin=`{protocol['coarse_spin']}`；target_parity=`{protocol['target_parity']}`。",
        f"- comparison：`{document['comparison']}`。",
        "",
        "| 侧 | steady 迭代数 | median solve (s) | MAD (s) | median ms/外层迭代 | full-op 真残差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side, label in (("pyqcu", "PyQCU Strict"), ("quda", "QUDA")):
        item = document["sides"][side]
        residuals = ", ".join(f"{x:.4e}" for x in item["full_operator_true_residual_rel"])
        lines.append(
            f"| {label} | {item['iterations']} | {item['median_wall_seconds']:.6f} | "
            f"{item['mad_wall_seconds']:.6f} | "
            f"{item['median_average_milliseconds_per_outer_iteration']:.3f} | `{residuals}` |")
    lines.extend([
        "",
        "逐外层迭代数据见同目录 CSV；图表见同目录 SVG。PyQCU 曲线是 Arnoldi estimate，",
        "QUDA 曲线是 GCR iterated residual，不能把两条曲线的每个浮点值当成同一递推量；",
        "最终收敛判据由 full Wilson/Clover operator 的 true residual 给出。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="展开并验证 Strict/QUDA formal iteration trace")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trace = _load(args.trace.resolve())
    benchmark = _load(args.benchmark.resolve())
    document = _join(trace, benchmark)
    document["sources"]["trace"] = str(args.trace.resolve())
    document["sources"]["benchmark"] = str(args.benchmark.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(document, args.csv)
    _plot(document, args.plot)
    _write_summary(document, args.summary)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "csv": str(args.csv.resolve()),
        "plot": str(args.plot.resolve()),
        "summary": str(args.summary.resolve()),
        "config_hash": document["protocol"]["config_hash"],
        "bundle_hash": document["input_fingerprints"]["bundle_hash"],
        "comparison": document["comparison"],
        "iterations": {side: document["sides"][side]["iterations"] for side in ("pyqcu", "quda")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
