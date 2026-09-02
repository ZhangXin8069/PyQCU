#!/usr/bin/env python3
"""重复测量 dev87 的单 P100 / 双 P100 Clover MultiGrid 吞吐。

该程序复用 ``run_multigpu._run_one`` 的真实 C++ 后端路径。每个 repeat
先运行一张 P100，再运行两张 P100；两者都在同一正式大格、同一 Gauge/Clover、
同一 coarse cache 和同一参数下求解。双卡模式是线程×GPU 的独立完整问题
并发，不是一个问题的空间域分解，因此结果只解释为并行吞吐/线程隔离。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = REPO / "data" / "multigpu_formal_20260902.json"
DEFAULT_PLOT = REPO / "data" / "multigpu_formal_20260902.svg"


def _median(values: Sequence[float]) -> float:
    return float(statistics.median([float(value) for value in values]))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _summary(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "samples_seconds": values,
        "median_seconds": _median(values),
        "mad_seconds": _mad(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _svg(document: Dict[str, Any], path: Path) -> None:
    """Write a dependency-free comparison plot for the measured MG times."""
    single = document["summary"]["single_p100"]["median_seconds"]
    dual = document["summary"]["dual_p100"]["median_seconds"]
    width, height = 980, 560
    left, right, top, bottom = 90, 35, 65, 390
    chart_width = width - left - right
    chart_height = bottom - top
    scale = max(single, dual) * 1.25

    def bar_x(index: int) -> float:
        return left + 120 + index * 360

    def bar_y(value: float) -> float:
        return bottom - chart_height * value / scale

    def text(x: float, y: float, value: str, size: int = 15,
             anchor: str = "start") -> str:
        escaped = (value.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;"))
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" '
                f'text-anchor="{anchor}" fill="#202124">{escaped}</text>')

    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" '
        'viewBox="0 0 980 560">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 30,
             "Clover MultiGrid: single P100 vs dual P100 parallel wall time",
             size=20, anchor="middle"),
        text(left, 48,
             "取每个 repeat 中所有线程的最大 MG solve time；越低越好",
             size=13),
    ]
    for tick in range(0, 6):
        value = scale * tick / 5.0
        y = bottom - chart_height * tick / 5.0
        pieces.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(text(left - 10, y + 5, f"{value:.1f}s", size=12,
                           anchor="end"))
    colors = ("#1769aa", "#c2410c")
    labels = ("single P100", "dual P100")
    for index, (label, value, color) in enumerate(
            zip(labels, (single, dual), colors)):
        x = bar_x(index)
        y = bar_y(value)
        pieces.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="220" '
            f'height="{bottom-y:.1f}" fill="{color}" opacity="0.85"/>')
        pieces.append(text(x + 110, bottom + 25, label, anchor="middle"))
        pieces.append(text(x + 110, y - 10, f"{value:.6f} s",
                           anchor="middle"))
    pieces.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{width-right}" '
        f'y2="{bottom}" stroke="#374151"/>',
        text(left, bottom + 65,
             f"median throughput ratio = {single / dual:.6f}x; "
             f"two-card efficiency = {single / dual / 2.0:.6f}", size=14),
        text(left, bottom + 88,
             "同型号基线；双卡为两个独立完整问题并发，不是域分解", size=13),
        "</svg>",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeat the formal single-P100/dual-P100 MultiGrid test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--lat", type=int, nargs=4, default=[16, 32, 32, 48])
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--num-restart", type=int, default=3)
    parser.add_argument("--coarse-max-iter", type=int, default=15)
    parser.add_argument("--coarse-tol-factor", type=float, default=1e3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consistency-tol", type=float, default=1e-5)
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if len(args.devices) != 2 or args.devices[0] == args.devices[1]:
        raise ValueError("--devices must contain two distinct GPU ids")

    import sys
    sys.path.insert(0, str(HERE))
    from run_multigpu import _find_devices, _run_one

    v100, p100 = _find_devices()
    if any(device not in p100 for device in args.devices):
        raise RuntimeError(
            f"--devices={args.devices} 不是两张可见 P100；P100={p100}, V100={v100}")

    run_args = SimpleNamespace(
        lat=list(args.lat), mass=args.mass, atol=args.atol,
        num_restart=args.num_restart,
        coarse_max_iter=args.coarse_max_iter,
        coarse_tol_factor=args.coarse_tol_factor,
        seed=args.seed, consistency_tol=args.consistency_tol,
        cache_dir=args.cache_dir, verbose=False,
    )
    single_rows: List[Dict[str, Any]] = []
    dual_rows: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        single = _run_one("single_p100", [args.devices[0]], run_args)
        dual = _run_one("dual_p100", list(args.devices), run_args)
        single["repeat"] = repeat + 1
        dual["repeat"] = repeat + 1
        single_rows.append(single)
        dual_rows.append(dual)
        print(json.dumps({
            "repeat": repeat + 1,
            "single_p100_mg_s": single["mg_parallel_wall_s"],
            "dual_p100_mg_s": dual["mg_parallel_wall_s"],
            "dual_consistency": dual["consistency"],
        }, ensure_ascii=False), flush=True)

    single_summary = _summary(single_rows, "mg_parallel_wall_s")
    dual_summary = _summary(dual_rows, "mg_parallel_wall_s")
    ratio = single_summary["median_seconds"] / dual_summary["median_seconds"]
    all_consistent = all(
        row["consistency"]["all_pass"] for row in single_rows + dual_rows)
    document: Dict[str, Any] = {
        "schema": {"name": "pyqcu.multigpu.single-vs-dual", "version": 1},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "lat": list(args.lat), "mass": args.mass, "atol": args.atol,
            "num_levels": 2, "dof_list": [12, 12],
            "mg_grid": [2, 2, 2, 2], "num_restart": args.num_restart,
            "coarse_max_iter": args.coarse_max_iter,
            "coarse_tol_factor": args.coarse_tol_factor,
            "seed": args.seed, "cache_dir": str(args.cache_dir.resolve()),
            "repeats": args.repeats,
            "warmup_policy": "each _run_one performs one complete setup and solve; no timing warmup",
            "parallel_semantics": "one thread per GPU; each thread solves a complete copied problem",
        },
        "devices": {"requested": list(args.devices), "p100": p100,
                    "v100": v100},
        "samples": {"single_p100": single_rows, "dual_p100": dual_rows},
        "summary": {"single_p100": single_summary, "dual_p100": dual_summary},
        "comparison": {
            "single_p100_over_dual_p100": ratio,
            "dual_p100_efficiency_vs_two_single": ratio / 2.0,
            "all_consistency_pass": all_consistent,
            "wall_time_definition": "max thread MG solve time",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    _svg(document, args.plot)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "plot": str(args.plot.resolve()),
        "single_median_s": single_summary["median_seconds"],
        "dual_median_s": dual_summary["median_seconds"],
        "single_over_dual": ratio,
        "dual_efficiency": ratio / 2.0,
        "all_consistency_pass": all_consistent,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if all_consistent else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"bench_multigpu_repeat: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
