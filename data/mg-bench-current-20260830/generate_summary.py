#!/usr/bin/env python3
"""汇总本目录中的 PyQCU CUDA-C++ MultiGrid 实测结果。"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "summary.csv"

COLUMNS = [
    "run_group", "label", "lattice", "backend", "devices", "ranks",
    "levels", "dof", "fine_dtype", "coarse_dtype", "solver", "smoother",
    "cycle", "restart", "coarse_max_iter", "coarse_tol_factor", "mu_pre",
    "deflate", "warm_start", "samples", "timing_scope", "wall_s", "solve_s",
    "ref_s", "iterations", "true_residual_rel", "rel_diff_vs_ref",
    "speedup_vs_l1_wall", "speedup_vs_bistabcg_solve",
    "throughput_scale_vs_single_p100", "peak_mib", "delta_mib",
    "aggregate_peak_mib", "aggregate_delta_mib", "memory_breakdown", "status",
    "source",
]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def fmt_lat(values):
    return "x".join(str(int(v)) for v in values)


def embedded_report(path: Path):
    text = path.read_text(errors="replace")
    start = text.find('{\n  "lat": [')
    if start < 0:
        raise ValueError(f"未找到 JSON 报告：{path}")
    return json.JSONDecoder().raw_decode(text[start:])[0], text


def cpp_stats(text: str):
    times = [float(v) for v in re.findall(r"Total time: ([0-9.]+) seconds", text)]
    solve_ms = [float(v) / 1000.0 for v in re.findall(
        r"MG init=[0-9.]+ms end=[0-9.]+ms solve=([0-9.]+)ms", text)]
    iterations = [int(v) for v in re.findall(r"Total iterations: ([0-9]+)", text)]
    if not iterations:
        iterations = [int(v) for v in re.findall(
            r"Mixed-precision BiCGStab: ([0-9]+) fine iterations", text)]
    # 标准性能块的 ``Total time`` 精度更高；混合精度路径没有该行时，
    # 再回退到 ``MG ... solve=...ms``。
    solves = times or solve_ms
    return solves, iterations


def old_memory(path: Path):
    if not path.exists():
        return None
    text = path.read_text().strip()
    values = {k: v for k, v in re.findall(r"([a-z_]+)=([0-9.]+)", text)}
    if "peak_mib" not in values:
        return None
    return {
        "peak_mib": int(float(values["peak_mib"])),
        "delta_mib": int(float(values["delta_mib"])),
        "memory_breakdown": "V100=" + values["peak_mib"] + "MiB",
    }


def tsv_memory(path: Path, target_indices=None):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            row["gpu_index"] = int(row["gpu_index"])
            for key in ("baseline_mib", "peak_mib", "delta_mib"):
                row[key] = int(row[key])
            rows.append(row)
    selected = rows
    if target_indices is not None:
        selected = [r for r in rows if r["gpu_index"] in target_indices]
    elif any("V100" in r["name"] for r in rows):
        selected = [r for r in rows if "V100" in r["name"]]
    peak = max((r["peak_mib"] for r in selected), default=None)
    delta = max((r["delta_mib"] for r in selected), default=None)
    breakdown = ";".join(
        f"gpu{r['gpu_index']}:{r['name']}={r['peak_mib']}MiB(+{r['delta_mib']})"
        for r in rows
    )
    return {"peak_mib": peak, "delta_mib": delta,
            "memory_breakdown": breakdown, "rows": rows}


def aggregate_memory(directory: Path):
    base = 0
    with (directory / "baseline.csv").open() as stream:
        for line in stream:
            base += int(line.split(",")[5].strip())
    peak = 0
    current = 0
    have = False
    with (directory / "nvidia-smi.csv").open() as stream:
        for line in stream:
            fields = line.split(",")
            index = int(fields[1].strip())
            if index == 0 and have:
                peak = max(peak, current)
                current = 0
            current += int(fields[5].strip())
            have = True
    peak = max(peak, current)
    return peak, peak - base


def status(report, warm=False):
    residual = report.get("warm_true_residual_rel" if warm else "true_residual_rel")
    atol = report.get("atol", 1e-6)
    if not finite(residual):
        return "失败_NaN"
    if residual > atol:
        return "精度不足"
    return "通过_热启动" if warm else "通过"


rows = []

memory_map = {
    "current_1l_c64": "mg_1l",
    "current_2l_cg_v": "mg_2l_cg",
    "current_2l_mr": "mg_2l_mr",
    "current_2l_gcr": "mg_2l_gcr",
    "current_2l_bicgstab_l": "mg_2l_bicgstab_l",
    "current_2l_ca_gcr": "mg_2l_ca_gcr",
    "current_2l_c128": "mg_2l_c128",
    "current_2l_c64_c128": "mg_2l_c64_c128",
    "current_2l_c128_c64": "mg_2l_c128_c64",
    "current_2l_deflate": "mg_2l_deflate_warm",
    "current_2l_warm": "mg_2l_deflate_warm",
    "current_3l_f": "mg_3l_f",
}

# 单次参数矩阵。
for path in sorted(ROOT.glob("current_*.stdout")):
    if path.name in {"current_bistabcg.stdout", "current_multigpu.stdout",
                     "current_mpi_np2.stdout", "current_mpi_np4.stdout"}:
        continue
    report, text = embedded_report(path)
    solves, iterations = cpp_stats(text)
    row = {
        "run_group": "scan_small_single", "label": report["label"],
        "lattice": fmt_lat(report["lat"]), "backend": "CUDA-C++",
        "devices": "V100", "ranks": 1, "levels": report["levels"],
        "dof": "/".join(str(v) for v in report["E"]),
        "fine_dtype": report["fine_dtype"],
        "coarse_dtype": "/".join(report["coarse_dtypes"]),
        "solver": report["solver"], "smoother": report["smoother"],
        "cycle": report["cycle"], "restart": report["restart"],
        "coarse_max_iter": report["coarse_max_iter"],
        "coarse_tol_factor": report["coarse_tol_factor"],
        "mu_pre": report["mu_pre"], "deflate": report["deflate"],
        "warm_start": False, "samples": 1,
        "timing_scope": "同步调用墙钟；solve_s 为后端纯 solve（若可解析）",
        "wall_s": report["mg_wall_s"],
        "solve_s": solves[0] if solves else report["mg_wall_s"],
        "iterations": iterations[0] if iterations else "",
        "true_residual_rel": report["true_residual_rel"],
        "rel_diff_vs_ref": report["rel_diff_vs_bistabcg"],
        "status": status(report), "source": str(path.relative_to(REPO)),
    }
    if report["label"] in memory_map:
        mpath = ROOT / "peak_v100_8x8x8x16" / memory_map[report["label"]] / "memory.tsv"
        mem = old_memory(mpath)
        if mem:
            row.update(mem)
            row["source"] += ";" + str(mpath.relative_to(REPO))
    rows.append(row)
    if report.get("warm_requested") and "warm_wall_s" in report:
        warm_row = dict(row)
        warm_row.update({
            "label": report["label"] + "_reuse_x0", "warm_start": True,
            "wall_s": report["warm_wall_s"],
            "solve_s": solves[1] if len(solves) > 1 else report["warm_wall_s"],
            "iterations": iterations[1] if len(iterations) > 1 else "",
            "true_residual_rel": report["warm_true_residual_rel"],
            "rel_diff_vs_ref": report["warm_rel_diff_vs_bistabcg"],
            "timing_scope": "已有解 x0 的重复求解；不是 cold solve",
            "status": status(report, warm=True),
        })
        rows.append(warm_row)

small_l1 = next(r["wall_s"] for r in rows if r["label"] == "current_1l_c64")
for row in rows:
    if row["run_group"] == "scan_small_single" and not row["warm_start"]:
        row["speedup_vs_l1_wall"] = small_l1 / row["wall_s"]

# 固定 cmi=15 的三次严格重复。
strict_root = ROOT / "strict_cmi15_8x8x8x16"
strict_labels = {"l1": "strict_l1", "2l_mr": "strict_2l_mr", "3l_f": "strict_3l_f"}
strict_rows = []
for config, label in strict_labels.items():
    reps = []
    for directory in sorted((strict_root / config).glob("rep*")):
        report = json.loads((directory / "result.json").read_text())
        text = (directory / "stdout.txt").read_text(errors="replace")
        solves, iterations = cpp_stats(text)
        mem = tsv_memory(directory / "memory.tsv")
        reps.append((report, solves[-1], iterations[-1], mem))
    walls = [v[0]["mg_wall_s"] for v in reps]
    solves = [v[1] for v in reps]
    report = reps[0][0]
    peak_values = [v[3]["peak_mib"] for v in reps]
    delta_values = [v[3]["delta_mib"] for v in reps]
    row = {
        "run_group": "strict_repeat_small", "label": label,
        "lattice": fmt_lat(report["lat"]), "backend": "CUDA-C++",
        "devices": "V100", "ranks": 1, "levels": report["levels"],
        "dof": "/".join(str(v) for v in report["E"]),
        "fine_dtype": report["fine_dtype"],
        "coarse_dtype": "/".join(report["coarse_dtypes"]),
        "solver": report["solver"], "smoother": report["smoother"],
        "cycle": report["cycle"], "restart": report["restart"],
        "coarse_max_iter": report["coarse_max_iter"],
        "coarse_tol_factor": report["coarse_tol_factor"],
        "mu_pre": report["mu_pre"], "deflate": report["deflate"],
        "warm_start": False, "samples": len(reps),
        "timing_scope": "三次同步墙钟/后端 solve 中位数",
        "wall_s": statistics.median(walls), "solve_s": statistics.median(solves),
        "iterations": int(statistics.median(v[2] for v in reps)),
        "true_residual_rel": max(v[0]["true_residual_rel"] for v in reps),
        "rel_diff_vs_ref": max(v[0]["rel_diff_vs_bistabcg"] for v in reps),
        "peak_mib": int(statistics.median(peak_values)),
        "delta_mib": int(statistics.median(delta_values)),
        "memory_breakdown": f"V100稳态中位={int(statistics.median(peak_values))}MiB;"
                            f"保守最大={max(peak_values)}MiB(+{max(delta_values)})",
        "status": status(report),
        "source": str((strict_root / config).relative_to(REPO)) + "/rep{1,2,3}",
    }
    strict_rows.append(row)
rows.extend(strict_rows)
strict_l1_wall = next(r["wall_s"] for r in strict_rows if r["label"] == "strict_l1")
strict_l1_solve = next(r["solve_s"] for r in strict_rows if r["label"] == "strict_l1")
for row in strict_rows:
    row["speedup_vs_l1_wall"] = strict_l1_wall / row["wall_s"]
    row["speedup_vs_bistabcg_solve"] = strict_l1_solve / row["solve_s"]

# 多格点单次运行；显存来自同配置独立监测运行。
lattice_rows = []
for path in sorted(ROOT.glob("lattice_*/*/stdout.txt")):
    if path.parent.name == "bistabcg":
        continue
    report, text = embedded_report(path)
    solves, iterations = cpp_stats(text)
    lat = fmt_lat(report["lat"])
    memory_name = None
    if lat == "16x16x16x16" and report["levels"] in (1, 2, 3):
        memory_name = f"lat16_{report['levels']}l"
    if lat == "16x32x32x48" and report["levels"] in (1, 2):
        memory_name = f"latlarge_{report['levels']}l"
    row = {
        "run_group": "lattice_scan", "label": report["label"], "lattice": lat,
        "backend": "CUDA-C++", "devices": "V100", "ranks": 1,
        "levels": report["levels"], "dof": "/".join(str(v) for v in report["E"]),
        "fine_dtype": report["fine_dtype"],
        "coarse_dtype": "/".join(report["coarse_dtypes"]),
        "solver": report["solver"], "smoother": report["smoother"],
        "cycle": report["cycle"], "restart": report["restart"],
        "coarse_max_iter": report["coarse_max_iter"],
        "coarse_tol_factor": report["coarse_tol_factor"], "mu_pre": report["mu_pre"],
        "deflate": report["deflate"], "warm_start": False, "samples": 1,
        "timing_scope": "同步调用墙钟；solve_s 为后端纯 solve",
        "wall_s": report["mg_wall_s"], "solve_s": solves[0] if solves else "",
        "iterations": iterations[0] if iterations else "",
        "true_residual_rel": report["true_residual_rel"],
        "rel_diff_vs_ref": report["rel_diff_vs_bistabcg"], "status": status(report),
        "source": str(path.relative_to(REPO)),
    }
    if memory_name:
        mpath = ROOT / "peak_v100_large" / memory_name / "memory.tsv"
        mem = old_memory(mpath)
        if mem:
            row.update(mem)
            row["source"] += ";" + str(mpath.relative_to(REPO))
    lattice_rows.append(row)
rows.extend(lattice_rows)
for lat in sorted({r["lattice"] for r in lattice_rows}):
    candidates = [r for r in lattice_rows if r["lattice"] == lat and r["levels"] == 1]
    if not candidates:
        continue
    baseline = candidates[0]["wall_s"]
    for row in lattice_rows:
        if row["lattice"] == lat:
            row["speedup_vs_l1_wall"] = baseline / row["wall_s"]

# 独立 BiStabCG 基线（仅用其后端计时；未把它混成 L1 算法基线）。
baseline_specs = [
    ("8x8x8x16", "small_bistabcg", ROOT / "peak_v100_8x8x8x16/bistabcg"),
    ("16x16x16x16", "lat16_bistabcg", ROOT / "peak_v100_large/lat16_bistabcg"),
    ("16x32x32x48", "latlarge_bistabcg", ROOT / "peak_v100_large/latlarge_bistabcg"),
]
bistab_times = {}
for lat, label, directory in baseline_specs:
    text = (directory / "stdout.txt").read_text(errors="replace")
    match = re.search(r"bistabcg total time: .*:([0-9.]+) sec", text)
    solve = float(match.group(1)) if match else None
    bistab_times[lat] = solve
    mem = old_memory(directory / "memory.tsv") or {}
    rows.append({
        "run_group": "bistabcg_baseline", "label": label, "lattice": lat,
        "backend": "CUDA-C++", "devices": "V100", "ranks": 1, "levels": 0,
        "solver": "bicgstab", "samples": 1,
        "timing_scope": "后端不含 malloc/free/memcpy 的 solve 计时", "solve_s": solve,
        "status": "计时基线", "source": str((directory / "stdout.txt").relative_to(REPO)),
        **mem,
    })
for row in rows:
    baseline = bistab_times.get(row.get("lattice"))
    solve = row.get("solve_s")
    if baseline and finite(solve) and row.get("levels", 0) > 0:
        row["speedup_vs_bistabcg_solve"] = baseline / solve

# MultiGpu：目标卡峰值与全机同时峰值都记录。
multi_specs = [
    ("single_v100", "V100", {2}),
    ("single_p100", "P100", {0}),
    ("p100x2", "P100x2", {0, 1}),
]
multi_rows = []
for name, devices, target_indices in multi_specs:
    directory = ROOT / "multigpu" / name
    report = json.loads((directory / "result.json").read_text())
    result = next(iter(report["results"].values()))
    mem = tsv_memory(directory / "memory.tsv", target_indices)
    aggregate_peak, aggregate_delta = aggregate_memory(directory)
    text = (directory / "stdout.txt").read_text(errors="replace")
    true_res = [float(v) for v in re.findall(
        r"FINAL TRUE residual \(full-op\) = [^\n]+ relative = ([0-9.eE+-]+)", text)]
    row = {
        "run_group": "multigpu", "label": name, "lattice": fmt_lat(report["lat"]),
        "backend": "CUDA-C++ MultiGpuMultigrid", "devices": devices, "ranks": 1,
        "levels": 2, "dof": "12", "fine_dtype": "c64", "coarse_dtype": "c64",
        "solver": "bicgstab", "smoother": "cg", "cycle": "v",
        "restart": report["num_restart"], "coarse_max_iter": report["coarse_max_iter"],
        "coarse_tol_factor": report["coarse_tol_factor"], "samples": 1,
        "timing_scope": "wall 含准备+参考解+MG；solve 为最慢线程 MG",
        "wall_s": result["wall_s"], "solve_s": result["mg_parallel_wall_s"],
        "ref_s": result["ref_parallel_wall_s"], "iterations": 72,
        "true_residual_rel": max(true_res) if true_res else "",
        "rel_diff_vs_ref": max(t["mg_vs_ref_rel_max"] for t in result["threads"]),
        "speedup_vs_bistabcg_solve": result["ref_parallel_wall_s"] / result["mg_parallel_wall_s"],
        "peak_mib": mem["peak_mib"], "delta_mib": mem["delta_mib"],
        "aggregate_peak_mib": aggregate_peak, "aggregate_delta_mib": aggregate_delta,
        "memory_breakdown": mem["memory_breakdown"],
        "status": "通过_独立任务吞吐测试",
        "source": str((directory / "result.json").relative_to(REPO)) + ";" +
                  str((directory / "memory.tsv").relative_to(REPO)),
    }
    multi_rows.append(row)
rows.extend(multi_rows)
single_p100 = next(r for r in multi_rows if r["label"] == "single_p100")
p100x2 = next(r for r in multi_rows if r["label"] == "p100x2")
p100x2["throughput_scale_vs_single_p100"] = 2 * single_p100["solve_s"] / p100x2["solve_s"]
p100x2["speedup_vs_l1_wall"] = single_p100["solve_s"] / p100x2["solve_s"]

# MPI 路径：rank 共享同一 V100，故明确标为非扩展性数据。
single_rank_wall = next(r["wall_s"] for r in rows if r["label"] == "current_2l_cg_v")
for np, grid in ((2, "1x1x1x2"), (4, "1x1x2x2")):
    directory = ROOT / "mpi" / f"np{np}"
    text = (directory / "stdout.txt").read_text(errors="replace")
    solves, iterations = cpp_stats(text)
    wall = float(re.search(r"\[mpi-mg\].* wall=([0-9.]+)s", text).group(1))
    residual = float(re.search(r"Relative residual \|D\*x - b\|/\|b\|: ([0-9.eE+-]+)", text).group(1))
    mem = tsv_memory(directory / "memory.tsv")
    aggregate_peak, aggregate_delta = aggregate_memory(directory)
    rows.append({
        "run_group": "mpi_shared_gpu", "label": f"mpi_np{np}_{grid}",
        "lattice": "8x8x8x16", "backend": "CUDA-C++ MPI", "devices": "V100(shared)",
        "ranks": np, "levels": 2, "dof": "12", "fine_dtype": "c64",
        "coarse_dtype": "c64", "solver": "bicgstab", "smoother": "cg",
        "cycle": "v", "restart": 5, "coarse_max_iter": 15,
        "coarse_tol_factor": 3000, "mu_pre": 4, "samples": 1,
        "timing_scope": "MPI rank 共享单 GPU；仅路径/显存对照", "wall_s": wall,
        "solve_s": solves[0], "iterations": iterations[0], "true_residual_rel": residual,
        "speedup_vs_l1_wall": single_rank_wall / wall,
        "peak_mib": mem["peak_mib"], "delta_mib": mem["delta_mib"],
        "aggregate_peak_mib": aggregate_peak, "aggregate_delta_mib": aggregate_delta,
        "memory_breakdown": mem["memory_breakdown"], "status": "通过_非扩展性数据",
        "source": str((directory / "stdout.txt").relative_to(REPO)) + ";" +
                  str((directory / "memory.tsv").relative_to(REPO)),
    })

# logs/*.txt 指向的历史性能证据：保留版本演进，不与本轮中位数混算。
for filename in ("logs/dev80_2/bench_out.txt", "logs/dev80_3/bench_out.txt"):
    path = REPO / filename
    lines = path.read_text().splitlines()
    lat = ast.literal_eval(next(line[4:].split(" mass", 1)[0] for line in lines if line.startswith("lat ")))
    for line in lines:
        if not line.startswith("{'label'"):
            continue
        item = ast.literal_eval(line)
        rows.append({
            "run_group": "historical_log", "label": path.parent.name + "_" + item["label"],
            "lattice": fmt_lat(lat), "backend": "历史 CUDA-C++", "devices": "V100",
            "ranks": 1, "levels": item["levels"], "samples": 1,
            "timing_scope": "历史日志墙钟", "wall_s": item["t"],
            "true_residual_rel": item.get("res", ""),
            "rel_diff_vs_ref": item.get("rel_vs_ref", ""),
            "speedup_vs_l1_wall": item.get("speedup_vs_L1", ""),
            "speedup_vs_bistabcg_solve": item.get("speedup_vs_BiStabCG", ""),
            "status": "历史_" + item["stat"], "source": filename,
        })

with OUT.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in COLUMNS})

print(f"rows={len(rows)} output={OUT}")
