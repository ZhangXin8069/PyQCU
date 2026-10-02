#!/usr/bin/env python3
"""Build the 2026-10-02 PyQCU/QUDA architecture report data.

The input is the immutable ``git tag test27`` combined matrix.  This script
only reads those records; every table, summary and figure is regenerated under
``data/report_multigrid_comprehensive_20261002/report``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


TEST27_COMMIT = "04382801c003353ecd1bc6b9bc10f6db750bf74d"
REPO = Path(__file__).resolve().parents[5]
DEFAULT_INPUT_DIR = (
    REPO
    / "data"
    / "report_multigrid_comprehensive_20260928"
    / "final_protocol"
    / "combined"
)
DEFAULT_MEMORY_CSV = (
    REPO
    / "data"
    / "report_multigrid_comprehensive_20260928"
    / "final_protocol"
    / "report"
    / "mg_memory.csv"
)
DEFAULT_REFERENCE_DIR = (
    REPO
    / "data"
    / "report_multigrid_comprehensive_20261001"
    / "report"
)
DEFAULT_OUTDIR = (
    REPO
    / "data"
    / "report_multigrid_comprehensive_20261002"
    / "report"
)

UNIT_RE = re.compile(
    r"(?P<device>v100|p100)__"
    r"(?P<lattice>[0-9]+x[0-9]+x[0-9]+x[0-9]+)__"
    r"(?P<precision>c64|c128)__l(?P<levels>[123])__"
    r"trace-(?P<trace>on|off)"
)

SIDE_COLORS = {"pyqcu": "#245ba3", "quda": "#be4137"}
SIDE_LABELS = {"pyqcu": "PyQCU", "quda": "QUDA"}
DEVICE_MARKERS = {
    ("v100", "c64"): "o",
    ("v100", "c128"): "s",
    ("p100", "c64"): "^",
    ("p100", "c128"): "D",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _median(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("median of an empty sequence")
    return float(statistics.median(items))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_unit_path(path: Path) -> dict[str, Any]:
    match = UNIT_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"unexpected combined filename: {path.name}")
    dimensions = match.groupdict()
    levels = int(dimensions["levels"])
    return {
        **dimensions,
        "case_id": path.stem,
        "levels": levels,
        "solver": "bicgstab" if levels == 1 else f"mg-{levels}",
        "path": path,
    }


def load_units(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no combined JSON records under {input_dir}")
    units = []
    for path in paths:
        unit = _parse_unit_path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        unit["document"] = document
        units.append(unit)
    return units


def load_memory(memory_csv: Path) -> dict[tuple[str, str], dict[str, str]]:
    with memory_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["case_id"], row["side"]): row for row in rows}


def _active_side(unit: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    side_document = unit["document"]["sides"][side]
    if side_document.get("status") != "ok":
        raise ValueError(f"{unit['case_id']} has non-ok {side} side")
    return side_document


def _side_iterations(unit: Mapping[str, Any], side: str) -> float:
    side_document = _active_side(unit, side)
    if int(unit["levels"]) == 1:
        values = side_document["reference_solver"]["iterations"]
    else:
        values = side_document["iterations"]
    return _float(values.get("median", values.get("median_seconds")))


def _side_seconds(unit: Mapping[str, Any], side: str) -> float:
    side_document = _active_side(unit, side)
    if int(unit["levels"]) == 1:
        return _float(side_document["reference_solver"]["steady"]["median_seconds"])
    return _float(side_document["timing"]["steady"]["median_seconds"])


def _side_setup_seconds(unit: Mapping[str, Any], side: str) -> float:
    side_document = _active_side(unit, side)
    if int(unit["levels"]) == 1:
        return float("nan")
    return _float(side_document["timing"]["setup_seconds"])


def _side_true_residual(unit: Mapping[str, Any], side: str) -> float:
    residual = _active_side(unit, side).get("true_residual") or {}
    samples = [_float(value) for value in residual.get("samples_rel") or []]
    if samples:
        return max(samples)
    return _float(residual.get("max_rel"))


def ratio_components(
    pyqcu_seconds: float,
    quda_seconds: float,
    pyqcu_iterations: float,
    quda_iterations: float,
) -> tuple[float, float, float]:
    """Return ``(R, iteration_ratio, per_iteration_cost_ratio)``.

    The three quantities obey exactly

    ``R = (N_Q/N_P) * ((t_Q/N_Q)/(t_P/N_P))``.
    """

    if min(pyqcu_seconds, quda_seconds, pyqcu_iterations, quda_iterations) <= 0:
        raise ValueError("time and iteration counts must be positive")
    ratio = quda_seconds / pyqcu_seconds
    iteration_ratio = quda_iterations / pyqcu_iterations
    cost_ratio = (quda_seconds / quda_iterations) / (
        pyqcu_seconds / pyqcu_iterations
    )
    if not math.isclose(
        ratio,
        iteration_ratio * cost_ratio,
        rel_tol=2.0e-14,
        abs_tol=2.0e-14,
    ):
        raise AssertionError("ratio decomposition is not algebraically closed")
    return ratio, iteration_ratio, cost_ratio


def audit_units(
    units: Sequence[Mapping[str, Any]],
    memory: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, Any]:
    if len(units) != 66:
        raise ValueError(f"expected 66 combined units, found {len(units)}")
    level_counts = Counter(int(unit["levels"]) for unit in units)
    if level_counts != Counter({1: 22, 2: 22, 3: 22}):
        raise ValueError(f"unexpected level counts: {dict(level_counts)}")
    if len(memory) != 132:
        raise ValueError(f"expected 132 side memory records, found {len(memory)}")

    side_records = 0
    for unit in units:
        document = unit["document"]
        case_id = str(unit["case_id"])
        if document.get("state") != "complete":
            raise ValueError(f"{case_id}: state is not complete")
        if document.get("selected_sides") != ["pyqcu", "quda"]:
            raise ValueError(f"{case_id}: selected sides changed")
        config_hashes = set()
        bundle_hashes = set()
        for side in ("pyqcu", "quda"):
            side_document = _active_side(unit, side)
            side_records += 1
            config_hashes.add(side_document.get("config_hash"))
            bundle_hashes.add(side_document.get("input_bundle_hash"))
            phase_results = side_document.get("phase_results") or {}
            if phase_results.get("warmup_count") != 2:
                raise ValueError(f"{case_id}/{side}: warmup count changed")
            if phase_results.get("steady_count") != 5:
                raise ValueError(f"{case_id}/{side}: steady count changed")
            if (case_id, side) not in memory:
                raise ValueError(f"{case_id}/{side}: missing memory record")
            if int(unit["levels"]) > 1:
                gate = _float(
                    (side_document.get("true_residual") or {}).get("gate")
                )
                if not gate or _side_true_residual(unit, side) > gate:
                    raise ValueError(f"{case_id}/{side}: true residual failed")
        if len(config_hashes) != 1 or None in config_hashes:
            raise ValueError(f"{case_id}: config hash mismatch")
        if len(bundle_hashes) != 1 or None in bundle_hashes:
            raise ValueError(f"{case_id}: input bundle mismatch")

    if side_records != 132:
        raise ValueError(f"expected 132 side records, found {side_records}")
    return {
        "combined_units": len(units),
        "side_records": side_records,
        "level_counts": dict(sorted(level_counts.items())),
        "memory_records": len(memory),
        "cold_warmup_steady": {
            "units": 132,
            "cold": 1,
            "warmup": 2,
            "steady": 5,
        },
    }


def _unit_rows(
    units: Sequence[Mapping[str, Any]],
    memory: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        case_id = str(unit["case_id"])
        pyqcu_seconds = _side_seconds(unit, "pyqcu")
        quda_seconds = _side_seconds(unit, "quda")
        pyqcu_iterations = _side_iterations(unit, "pyqcu")
        quda_iterations = _side_iterations(unit, "quda")
        ratio, iteration_ratio, cost_ratio = ratio_components(
            pyqcu_seconds,
            quda_seconds,
            pyqcu_iterations,
            quda_iterations,
        )
        pyqcu_memory = memory[(case_id, "pyqcu")]
        quda_memory = memory[(case_id, "quda")]
        rows.append(
            {
                "unit_id": case_id,
                "device": unit["device"],
                "precision": unit["precision"],
                "lattice": unit["lattice"],
                "levels": int(unit["levels"]),
                "solver": unit["solver"],
                "trace": unit["trace"],
                "pyqcu_steady_seconds": pyqcu_seconds,
                "quda_steady_seconds": quda_seconds,
                "ratio_quda_over_pyqcu": ratio,
                "pyqcu_outer_iterations": pyqcu_iterations,
                "quda_outer_iterations": quda_iterations,
                "iteration_ratio_quda_over_pyqcu": iteration_ratio,
                "per_iteration_cost_ratio_quda_over_pyqcu": cost_ratio,
                "pyqcu_setup_seconds": _side_setup_seconds(unit, "pyqcu"),
                "quda_setup_seconds": _side_setup_seconds(unit, "quda"),
                "setup_ratio_quda_over_pyqcu": (
                    _side_setup_seconds(unit, "quda")
                    / _side_setup_seconds(unit, "pyqcu")
                    if int(unit["levels"]) > 1
                    else float("nan")
                ),
                "pyqcu_true_residual_max": _side_true_residual(unit, "pyqcu"),
                "quda_true_residual_max": _side_true_residual(unit, "quda"),
                "pyqcu_steady_sampler_peak_bytes": _float(
                    pyqcu_memory.get("steady_sampler_peak_bytes")
                ),
                "quda_steady_sampler_peak_bytes": _float(
                    quda_memory.get("steady_sampler_peak_bytes")
                ),
                "pyqcu_asset_resident_bytes": _float(
                    pyqcu_memory.get("asset_resident_bytes")
                ),
                "pyqcu_fused_workspace_bytes": _float(
                    pyqcu_memory.get("fused_workspace_bytes")
                ),
            }
        )
    return rows


def _group_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "samples": len(values),
        "median_ratio": _median(values),
        "min_ratio": min(values),
        "max_ratio": max(values),
        "wins": sum(value > 1.0 for value in values),
        "losses": sum(value < 1.0 for value in values),
    }


def _group_rows(unit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        key = (
            str(row["device"]),
            str(row["precision"]),
            str(row["solver"]),
            str(row["trace"]),
        )
        grouped[key].append(row)
    output = []
    for (device, precision, solver, trace), rows in sorted(grouped.items()):
        ratios = [float(row["ratio_quda_over_pyqcu"]) for row in rows]
        stats = _group_stats(ratios)
        output.append(
            {
                "device": device,
                "precision": precision,
                "solver": solver,
                "trace": trace,
                **stats,
                "pyqcu_median_seconds": _median(
                    float(row["pyqcu_steady_seconds"]) for row in rows
                ),
                "quda_median_seconds": _median(
                    float(row["quda_steady_seconds"]) for row in rows
                ),
                "pyqcu_median_iterations": _median(
                    float(row["pyqcu_outer_iterations"]) for row in rows
                ),
                "quda_median_iterations": _median(
                    float(row["quda_outer_iterations"]) for row in rows
                ),
                "median_iteration_ratio": _median(
                    float(row["iteration_ratio_quda_over_pyqcu"])
                    for row in rows
                ),
                "median_per_iteration_cost_ratio": _median(
                    float(row["per_iteration_cost_ratio_quda_over_pyqcu"])
                    for row in rows
                ),
            }
        )
    return output


def _algorithm_rows(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        if int(unit["levels"]) == 1:
            continue
        protocol = unit["document"]["protocol"]
        pyqcu = _active_side(unit, "pyqcu")
        quda = _active_side(unit, "quda")
        actual = (quda.get("quda_parameters") or {}).get("actual") or {}
        invert = actual.get("invert") or {}
        multigrid = actual.get("multigrid") or {}
        levels = multigrid.get("levels") or {}
        transition = multigrid.get("transition") or {}
        rows.append(
            {
                "case_id": unit["case_id"],
                "device": unit["device"],
                "precision": unit["precision"],
                "lattice": unit["lattice"],
                "levels": int(unit["levels"]),
                "trace": unit["trace"],
                "block_xyzt": "x".join(
                    str(value) for value in protocol["block_xyzt"]
                ),
                "nvec": int(protocol["nvec"]),
                "coarse_dof": int(protocol["coarse_dof"]),
                "nu_pre": int(protocol["nu_pre"]),
                "nu_post": int(protocol["nu_post"]),
                "outer_restart": int(protocol["restart_effective"]),
                "outer_tolerance": _float(protocol["tolerance"]),
                "coarse_tolerance": _float(protocol["coarse_tolerance"]),
                "pyqcu_iteration_kind": pyqcu.get("iteration_kind", ""),
                "pyqcu_smoother": "MR",
                "pyqcu_coarse_solver": "fused cooperative BiCGStab",
                "quda_outer_solver": invert.get("inv_type", ""),
                "quda_smoother": ",".join(
                    str(value) for value in levels.get("smoother", [])
                ),
                "quda_coarse_solver": ",".join(
                    str(value) for value in levels.get("coarse_solver", [])
                ),
                "quda_coarse_precision": protocol["quda_coarse_precision"][
                    "effective"
                ],
                "quda_setup_use_mma": ",".join(
                    str(value)
                    for value in transition.get("setup_use_mma", [])
                ),
                "quda_dslash_use_mma": ",".join(
                    str(value)
                    for value in transition.get("dslash_use_mma", [])
                ),
                "quda_transfer_use_mma": ",".join(
                    str(value)
                    for value in transition.get("transfer_use_mma", [])
                ),
            }
        )
    return rows


def _reference_rows(reference_dir: Path) -> dict[str, dict[str, str]]:
    path = reference_dir / "unit_times.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["unit_id"]: row for row in csv.DictReader(handle)}


def validate_reference(
    unit_rows: Sequence[Mapping[str, Any]],
    reference_dir: Path,
) -> int:
    reference = _reference_rows(reference_dir)
    if set(reference) != {str(row["unit_id"]) for row in unit_rows}:
        raise ValueError("20261001 reference matrix has a different unit set")
    checked = 0
    for row in unit_rows:
        old = reference[str(row["unit_id"])]
        pairs = (
            (
                "pyqcu_steady_seconds",
                float(old["pyqcu_steady_seconds"]),
                float(row["pyqcu_steady_seconds"]),
            ),
            (
                "quda_steady_seconds",
                float(old["quda_steady_seconds"]),
                float(row["quda_steady_seconds"]),
            ),
            (
                "ratio_quda_over_pyqcu",
                float(old["ratio_quda_over_pyqcu"]),
                float(row["ratio_quda_over_pyqcu"]),
            ),
        )
        for name, expected, actual in pairs:
            if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    f"{row['unit_id']}/{name}: 20261001={expected} "
                    f"recomputed={actual}"
                )
        checked += 1
    return checked


def _summary(
    units: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    def subset(levels: tuple[int, ...], trace: str | None = None) -> list[float]:
        return [
            float(row["ratio_quda_over_pyqcu"])
            for row in unit_rows
            if int(row["levels"]) in levels
            and (trace is None or row["trace"] == trace)
        ]

    mg_off = subset((2, 3), "off")
    mg_all = subset((2, 3))
    all_ratios = subset((1, 2, 3))
    return {
        "source_commit": TEST27_COMMIT,
        "audit": dict(audit),
        "ratios": {
            "all_66_median": _median(all_ratios),
            "mg_44_median": _median(mg_all),
            "mg_trace_off_22_median": _median(mg_off),
            "mg_trace_off_wins": sum(value > 1.0 for value in mg_off),
            "mg_trace_off_losses": sum(value < 1.0 for value in mg_off),
            "reference_22_median": _median(subset((1,))),
            "reference_22_wins": sum(value > 1.0 for value in subset((1,))),
            "mg_trace_off_iteration_ratio_median": _median(
                float(row["iteration_ratio_quda_over_pyqcu"])
                for row in unit_rows
                if row["levels"] in (2, 3) and row["trace"] == "off"
            ),
            "mg_trace_off_per_iteration_cost_ratio_median": _median(
                float(row["per_iteration_cost_ratio_quda_over_pyqcu"])
                for row in unit_rows
                if row["levels"] in (2, 3) and row["trace"] == "off"
            ),
        },
        "counts": {
            "units": len(unit_rows),
            "mg_units": sum(int(row["levels"]) in (2, 3) for row in unit_rows),
            "reference_units": sum(int(row["levels"]) == 1 for row in unit_rows),
        },
    }


def _memory_summary(memory: Mapping[tuple[str, str], Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_, side), row in memory.items():
        device = str(row["device"]).lower()
        device_tag = "p100" if "p100" in device else "v100"
        value = _float(row.get("steady_sampler_peak_bytes"))
        if value > 0:
            grouped[(side, device_tag, str(row["precision"]))].append(value)
    rows = []
    for (side, device, precision), values in sorted(grouped.items()):
        values = sorted(values)
        index = min(
            len(values) - 1,
            max(0, math.ceil(0.95 * len(values)) - 1),
        )
        rows.append(
            {
                "side": side,
                "device": device,
                "precision": precision,
                "samples": len(values),
                "median_bytes": _median(values),
                "p95_bytes": values[index],
                "max_bytes": max(values),
            }
        )
    all_values = sorted(
        _float(row.get("steady_sampler_peak_bytes"))
        for row in memory.values()
        if _float(row.get("steady_sampler_peak_bytes")) > 0
    )
    index = min(
        len(all_values) - 1,
        max(0, math.ceil(0.95 * len(all_values)) - 1),
    )
    rows.append(
        {
            "side": "all",
            "device": "all",
            "precision": "all",
            "samples": len(all_values),
            "median_bytes": _median(all_values),
            "p95_bytes": all_values[index],
            "max_bytes": max(all_values),
        }
    )
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
        }
    )


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", metadata={"Creator": "PyQCU"})
    fig.savefig(outdir / f"{stem}.svg", metadata={"Creator": "PyQCU"})
    fig.savefig(outdir / f"{stem}.png", dpi=160, metadata={"Creator": "PyQCU"})
    plt.close(fig)


def plot_ratio_decomposition(
    unit_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> None:
    rows = [
        row
        for row in unit_rows
        if int(row["levels"]) in (2, 3) and row["trace"] == "off"
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    xlim = (1.6, 4.2)
    ylim = (0.2, 5.3)
    x_grid = [xlim[0] * (xlim[1] / xlim[0]) ** (index / 300) for index in range(301)]
    for ratio, alpha in ((1.0, 0.45), (2.0, 0.35), (4.0, 0.3), (8.0, 0.25)):
        ax.plot(
            x_grid,
            [ratio / value for value in x_grid],
            color="#7a7a7a",
            linewidth=0.8,
            alpha=alpha,
            zorder=1,
        )
        y = ratio / xlim[1]
        if ylim[0] < y < ylim[1]:
            ax.text(
                xlim[1],
                y,
                f"R={ratio:g}",
                ha="right",
                va="bottom",
                color="#5d5d5d",
                fontsize=7,
            )
    for device in ("v100", "p100"):
        for precision in ("c64", "c128"):
            selected = [
                row
                for row in rows
                if row["device"] == device and row["precision"] == precision
            ]
            if not selected:
                continue
            ax.scatter(
                [float(row["iteration_ratio_quda_over_pyqcu"]) for row in selected],
                [
                    float(row["per_iteration_cost_ratio_quda_over_pyqcu"])
                    for row in selected
                ],
                s=[28.0 + 13.0 * float(row["ratio_quda_over_pyqcu"]) for row in selected],
                marker=DEVICE_MARKERS[(device, precision)],
                color=SIDE_COLORS["quda"] if device == "v100" else "#2f7d69",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.88,
                label=f"{device.upper()} {precision}",
                zorder=3,
            )
    ax.axvline(1.0, color="#222222", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([2.0, 3.0, 4.0])
    ax.set_xticklabels(["2", "3", "4"])
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel(r"Outer-iteration ratio  $N_Q/N_P$")
    ax.set_ylabel(
        "Per-iteration cost ratio\n"
        r"$(t_Q/N_Q)/(t_P/N_P)$"
    )
    ax.set_title("Why the trace-off MG wins decompose into iterations and cost")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    _save(fig, outdir, "mg_ratio_decomposition_20261002")


def plot_win_matrix(
    unit_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> None:
    rows = [
        row
        for row in unit_rows
        if int(row["levels"]) in (2, 3) and row["trace"] == "off"
    ]
    row_keys = [
        ("v100", "c64"),
        ("v100", "c128"),
        ("p100", "c64"),
        ("p100", "c128"),
    ]
    col_keys = [
        "8x8x8x16",
        "16x16x16x16",
        "16x16x32x32",
        "16x32x32x48",
    ]
    values = [[float("nan")] * len(col_keys) for _ in row_keys]
    annotations = [[""] * len(col_keys) for _ in row_keys]
    for row_index, (device, precision) in enumerate(row_keys):
        for col_index, lattice in enumerate(col_keys):
            selected = [
                row
                for row in rows
                if row["device"] == device
                and row["precision"] == precision
                and row["lattice"] == lattice
            ]
            if not selected:
                continue
            ratios = [float(row["ratio_quda_over_pyqcu"]) for row in selected]
            wins = sum(value > 1.0 for value in ratios)
            values[row_index][col_index] = _median(ratios)
            annotations[row_index][col_index] = (
                f"{_median(ratios):.2f}\n{wins}/{len(ratios)}"
            )
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=4.0)
    image = ax.imshow(values, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(
        [value.replace("x", r"$\times$") for value in col_keys],
        rotation=22,
        ha="right",
    )
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{device.upper()} / {prec}" for device, prec in row_keys])
    ax.set_xlabel("Lattice")
    ax.set_ylabel("Device / precision")
    ax.set_title("Trace-off MG median $R$ (upper) and PyQCU wins (lower)")
    for row_index in range(len(row_keys)):
        for col_index in range(len(col_keys)):
            text = annotations[row_index][col_index]
            if not text:
                continue
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                linespacing=1.15,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(r"$R=t_Q/t_P$")
    _save(fig, outdir, "mg_win_matrix_20261002")


def plot_iteration_compare(
    unit_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> None:
    rows = [
        row
        for row in unit_rows
        if int(row["levels"]) in (2, 3) and row["trace"] == "off"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 9.0), constrained_layout=True)
    for ax, levels in zip(axes, (2, 3)):
        selected = [row for row in rows if int(row["levels"]) == levels]
        selected.sort(
            key=lambda row: (
                str(row["device"]),
                str(row["precision"]),
                str(row["lattice"]),
            )
        )
        y = list(range(len(selected)))
        height = 0.36
        ax.barh(
            [value + height / 2 for value in y],
            [float(row["pyqcu_outer_iterations"]) for row in selected],
            height=height,
            color=SIDE_COLORS["pyqcu"],
            label="PyQCU",
        )
        ax.barh(
            [value - height / 2 for value in y],
            [float(row["quda_outer_iterations"]) for row in selected],
            height=height,
            color=SIDE_COLORS["quda"],
            label="QUDA",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [
                f"{row['device'].upper()} {row['precision']} "
                f"{str(row['lattice']).replace('x', 'x')}"
                for row in selected
            ]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Outer solver iterations")
        ax.set_title(f"MG-{levels}: outer iteration count")
        ax.legend(loc="lower right", frameon=False)
    _save(fig, outdir, "mg_outer_iterations_20261002")


def plot_setup_compare(
    unit_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> None:
    rows = [
        row
        for row in unit_rows
        if int(row["levels"]) in (2, 3) and row["trace"] == "off"
    ]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = [float(row["pyqcu_setup_seconds"]) for row in rows]
    y = [float(row["quda_setup_seconds"]) for row in rows]
    lower = min(min(x), min(y)) * 0.75
    upper = max(max(x), max(y)) * 1.3
    ax.plot([lower, upper], [lower, upper], color="#555555", linewidth=0.8)
    for device in ("v100", "p100"):
        for precision in ("c64", "c128"):
            selected = [
                row
                for row in rows
                if row["device"] == device and row["precision"] == precision
            ]
            if not selected:
                continue
            ax.scatter(
                [float(row["pyqcu_setup_seconds"]) for row in selected],
                [float(row["quda_setup_seconds"]) for row in selected],
                marker=DEVICE_MARKERS[(device, precision)],
                color=SIDE_COLORS["quda"] if device == "v100" else "#2f7d69",
                edgecolor="white",
                linewidth=0.6,
                s=44,
                label=f"{device.upper()} {precision}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("PyQCU setup time (s)")
    ax.set_ylabel("QUDA setup time (s)")
    ax.set_title("Setup comparison is context, not part of solve speedup")
    ax.legend(frameon=False, ncol=2)
    _save(fig, outdir, "mg_setup_context_20261002")


def plot_memory_distribution(
    memory: Mapping[tuple[str, str], Mapping[str, str]],
    outdir: Path,
) -> None:
    groups = [
        ("pyqcu", "v100", "c64"),
        ("quda", "v100", "c64"),
        ("pyqcu", "v100", "c128"),
        ("quda", "v100", "c128"),
        ("pyqcu", "p100", "c64"),
        ("quda", "p100", "c64"),
        ("pyqcu", "p100", "c128"),
        ("quda", "p100", "c128"),
    ]
    values = []
    labels = []
    point_colors = []
    for side, device, precision in groups:
        selected = []
        for (_, row_side), row in memory.items():
            row_device = str(row["device"]).lower()
            row_device_tag = "p100" if "p100" in row_device else "v100"
            value = _float(row.get("steady_sampler_peak_bytes"))
            if (
                row_side == side
                and row_device_tag == device
                and row["precision"] == precision
                and value > 0
            ):
                selected.append(value / (1 << 20))
        values.append(selected)
        labels.append(f"{SIDE_LABELS[side]}\n{device.upper()} {precision}")
        point_colors.append(SIDE_COLORS[side])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], point_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
    for index, (selected, color) in enumerate(zip(values, point_colors), start=1):
        offsets = [
            (point_index - (len(selected) - 1) / 2) * 0.022
            for point_index in range(len(selected))
        ]
        ax.scatter(
            [index + offset for offset in offsets],
            selected,
            color=color,
            s=10,
            alpha=0.7,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Device-wide steady peak (MiB)")
    ax.set_title("Observed steady memory includes allocator/context effects")
    ax.tick_params(axis="x", labelrotation=25)
    _save(fig, outdir, "mg_memory_distribution_20261002")


def render_plots(
    unit_rows: Sequence[Mapping[str, Any]],
    memory: Mapping[tuple[str, str], Mapping[str, str]],
    outdir: Path,
) -> None:
    _style()
    plot_ratio_decomposition(unit_rows, outdir)
    plot_win_matrix(unit_rows, outdir)
    plot_iteration_compare(unit_rows, outdir)
    plot_setup_compare(unit_rows, outdir)
    plot_memory_distribution(memory, outdir)


def build_data(
    input_dir: Path,
    memory_csv: Path,
    reference_dir: Path,
    outdir: Path,
) -> dict[str, Any]:
    units = load_units(input_dir)
    memory = load_memory(memory_csv)
    audit = audit_units(units, memory)
    unit_rows = _unit_rows(units, memory)
    group_rows = _group_rows(unit_rows)
    algorithm_rows = _algorithm_rows(units)
    reference_count = validate_reference(unit_rows, reference_dir)
    summary = _summary(units, unit_rows, audit)
    summary["audit"]["reference_comparison_cells"] = reference_count
    summary["ratios"]["memory"] = _memory_summary(memory)

    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "unit_analysis.csv", unit_rows)
    _write_csv(outdir / "group_analysis.csv", group_rows)
    _write_csv(outdir / "algorithm_profiles.csv", algorithm_rows)
    _write_json(outdir / "summary.json", summary)
    _write_json(
        outdir / "source_hashes.json",
        {
            "source_commit": TEST27_COMMIT,
            "combined": {
                unit["case_id"]: _sha256(unit["path"]) for unit in units
            },
            "memory_csv": _sha256(memory_csv),
            "reference_unit_times_csv": _sha256(reference_dir / "unit_times.csv"),
        },
    )
    return summary


def build(
    input_dir: Path,
    memory_csv: Path,
    reference_dir: Path,
    outdir: Path,
) -> dict[str, Any]:
    summary = build_data(input_dir, memory_csv, reference_dir, outdir)
    units = load_units(input_dir)
    memory = load_memory(memory_csv)
    unit_rows = _unit_rows(units, memory)
    render_plots(unit_rows, memory, outdir)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the 2026-10-02 MG architecture report data"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--memory-csv", type=Path, default=DEFAULT_MEMORY_CSV)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="build data and figures, then print the key audit counts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build(
        args.input_dir,
        args.memory_csv,
        args.reference_dir,
        args.outdir,
    )
    if args.check:
        ratios = summary["ratios"]
        audit = summary["audit"]
        print(
            f"units={audit['combined_units']} sides={audit['side_records']} "
            f"memory={audit['memory_records']}"
        )
        print(
            "R_all={:.6f} R_mg_off={:.6f} wins_mg_off={}/{}".format(
                ratios["all_66_median"],
                ratios["mg_trace_off_22_median"],
                ratios["mg_trace_off_wins"],
                ratios["mg_trace_off_wins"] + ratios["mg_trace_off_losses"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
