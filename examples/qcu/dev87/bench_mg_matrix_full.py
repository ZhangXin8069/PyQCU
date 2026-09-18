#!/usr/bin/env python3
"""Resumable full Cartesian launcher for PyQCU/QUDA MultiGrid benchmarks.

The matrix is intentionally represented independently from the collector CLI.
All collector compatibility decisions live in ``_collector_command``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_COLLECTOR = HERE / "bench_strict_vs_quda.py"
DEFAULT_OUTPUT_DIR = REPO / "data" / "mg_matrix_full"
DEFAULT_ASSET_ROOT = REPO / "data"
DEFAULT_QIO_ROOT = DEFAULT_ASSET_ROOT / "qio-matrix"

SIDES = ("pyqcu", "quda")
PRECISIONS = ("c64", "c128")
LATTICES = (
    (8, 8, 8, 16),
    (16, 16, 16, 16),
    (16, 32, 32, 48),
)
TRACES = ("off", "on")
LEVELS = (1, 2, 3)
DEVICES = ("v100", "p100")
BLOCK = (2, 2, 2, 2)
PHASE_SPECS = (("cold", 1), ("warmup", 2), ("steady", 5))
PHASE_CATEGORIES = (
    "pre_smoother",
    "post_smoother",
    "restrict",
    "prolongate",
    "coarse_solver",
    "other",
)
TRACE_ENV_NAMES = (
    "PYQCU_STRICT_TRACE_FILE",
    "PYQCU_QUDA_TRACE_FILE",
    "QUDA_MG_TRACE_FILE",
)


@dataclass(frozen=True)
class MatrixUnit:
    side: str
    precision: str
    lattice: tuple[int, int, int, int]
    trace: str
    levels: int
    device: str

    @property
    def lattice_tag(self) -> str:
        return "x".join(str(value) for value in self.lattice)

    @property
    def mpi_ranks(self) -> int:
        return 1 if self.device == "v100" else 2

    @property
    def process_grid(self) -> tuple[int, int, int, int]:
        return (1, 1, 1, 1) if self.mpi_ranks == 1 else (2, 1, 1, 1)

    @property
    def unit_id(self) -> str:
        return "__".join((
            self.side,
            self.device,
            self.lattice_tag,
            self.precision,
            f"l{self.levels}",
            f"trace-{self.trace}",
        ))

    @property
    def phase_records(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        ordinal = 0
        for phase, count in PHASE_SPECS:
            for repeat in range(count):
                ordinal += 1
                records.append({
                    "ordinal": ordinal,
                    "phase": phase,
                    "repeat": repeat,
                })
        return tuple(records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "side": self.side,
            "precision": self.precision,
            "lattice": list(self.lattice),
            "lattice_tag": self.lattice_tag,
            "block": list(BLOCK),
            "trace": self.trace,
            "levels": self.levels,
            "device": self.device,
            "mpi_ranks": self.mpi_ranks,
            "process_grid": list(self.process_grid),
            "phase_records": [dict(value) for value in self.phase_records],
        }


@dataclass(frozen=True)
class Selection:
    sides: tuple[str, ...] = SIDES
    levels: tuple[int, ...] = LEVELS
    precisions: tuple[str, ...] = PRECISIONS
    lattices: tuple[tuple[int, int, int, int], ...] = LATTICES
    traces: tuple[str, ...] = TRACES
    devices: tuple[str, ...] = DEVICES

    def as_dict(self) -> dict[str, Any]:
        return {
            "sides": list(self.sides),
            "levels": list(self.levels),
            "precisions": list(self.precisions),
            "lattices": [list(value) for value in self.lattices],
            "traces": list(self.traces),
            "devices": list(self.devices),
        }


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class AssetRoots:
    asset_root: Path = DEFAULT_ASSET_ROOT
    gauge_root: Path | None = None
    nullvec_root: Path | None = None
    qio_root: Path | None = None
    cache_root: Path | None = None

    @property
    def gauge_directory(self) -> Path:
        return self.gauge_root or self.asset_root

    @property
    def nullvec_directory(self) -> Path:
        return self.nullvec_root or self.asset_root

    @property
    def qio_directory(self) -> Path:
        if self.qio_root is not None:
            return self.qio_root
        if self.asset_root == DEFAULT_ASSET_ROOT:
            return DEFAULT_QIO_ROOT
        return self.asset_root / "qio-matrix"

    def cache_directory(self, output_dir: Path, unit: "MatrixUnit") -> Path:
        return (self.cache_root or (output_dir / "cache")) / unit.unit_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_root": str(self.asset_root),
            "gauge_root": str(self.gauge_directory),
            "nullvec_root": str(self.nullvec_directory),
            "qio_root": str(self.qio_directory),
            "cache_root": (
                None if self.cache_root is None else str(self.cache_root)),
        }


Runner = Callable[
    [Sequence[str], Path, Mapping[str, str], float],
    RunResult,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_units() -> tuple[MatrixUnit, ...]:
    """Return the deterministic full Cartesian product."""
    units: list[MatrixUnit] = []
    for side in SIDES:
        for precision in PRECISIONS:
            for lattice in LATTICES:
                for trace in TRACES:
                    for levels in LEVELS:
                        for device in DEVICES:
                            units.append(MatrixUnit(
                                side=side,
                                precision=precision,
                                lattice=lattice,
                                trace=trace,
                                levels=levels,
                                device=device,
                            ))
    return tuple(units)


def select_units(
        units: Sequence[MatrixUnit], selection: Selection,
) -> tuple[MatrixUnit, ...]:
    return tuple(
        unit for unit in units
        if unit.side in selection.sides
        and unit.levels in selection.levels
        and unit.precision in selection.precisions
        and unit.lattice in selection.lattices
        and unit.trace in selection.traces
        and unit.device in selection.devices
    )


def _flatten_filter(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        result.extend(
            token.strip() for token in str(value).split(",") if token.strip())
    return tuple(result)


def _parse_lattice(value: str) -> tuple[int, int, int, int]:
    normalized = value.lower().replace(" ", "").replace("*", "x")
    for lattice in LATTICES:
        tag = "x".join(str(extent) for extent in lattice)
        if normalized in (tag, f"L{tag}"):
            return lattice
    raise argparse.ArgumentTypeError(
        f"unknown lattice {value!r}; expected one of "
        + ", ".join("x".join(str(extent) for extent in lat) for lat in LATTICES))


def _selection_from_args(args: argparse.Namespace) -> Selection:
    sides = _flatten_filter(args.only_side) or SIDES
    precisions = _flatten_filter(args.only_precision) or PRECISIONS
    traces = _flatten_filter(args.only_trace) or TRACES
    devices = _flatten_filter(args.only_device) or DEVICES

    levels = tuple(int(value) for value in _flatten_filter(args.only_levels))
    lattices = tuple(_parse_lattice(value)
                     for value in _flatten_filter(args.only_lattice))

    invalid_sides = set(sides) - set(SIDES)
    invalid_precisions = set(precisions) - set(PRECISIONS)
    invalid_traces = set(traces) - set(TRACES)
    invalid_devices = set(devices) - set(DEVICES)
    invalid_levels = set(levels) - set(LEVELS)
    if invalid_sides:
        raise ValueError(f"unknown sides: {sorted(invalid_sides)}")
    if invalid_precisions:
        raise ValueError(f"unknown precisions: {sorted(invalid_precisions)}")
    if invalid_traces:
        raise ValueError(f"unknown trace values: {sorted(invalid_traces)}")
    if invalid_devices:
        raise ValueError(f"unknown devices: {sorted(invalid_devices)}")
    if invalid_levels:
        raise ValueError(f"unknown levels: {sorted(invalid_levels)}")

    return Selection(
        sides=tuple(dict.fromkeys(sides)),
        levels=tuple(dict.fromkeys(levels)) or LEVELS,
        precisions=tuple(dict.fromkeys(precisions)),
        lattices=tuple(dict.fromkeys(lattices)) or LATTICES,
        traces=tuple(dict.fromkeys(traces)),
        devices=tuple(dict.fromkeys(devices)),
    )


def unit_output_path(output_dir: Path, unit: MatrixUnit) -> Path:
    return output_dir / "units" / f"{unit.unit_id}.json"


def trace_paths(output_dir: Path, unit: MatrixUnit) -> dict[str, str]:
    directory = output_dir / "traces" / unit.unit_id
    return {
        "pyqcu": str(directory / "pyqcu.tsv"),
        "quda": str(directory / "quda.tsv"),
        "quda_log": str(directory / "quda.log"),
    }


def resolve_unit_assets(
        unit: MatrixUnit, roots: AssetRoots, output_dir: Path,
) -> dict[str, Any]:
    tag = unit.lattice_tag
    # The formal protocol stores canonical inputs in single precision and runs
    # the c128 column with a double-precision solver (QUDA_MULTIGRID_DOUBLE
    # mixed precision).  Prefer a genuine ``_c128`` asset when it exists, but
    # fall back to the canonical ``_c64`` input and record that choice instead
    # of silently declaring the whole c128 column unavailable.
    gauge_native = (
        roots.gauge_directory
        / f"gauge_{tag}_m0.05_seed42_{unit.precision}.h5")
    nullvec_native = (
        roots.nullvec_directory
        / f"L{tag}_nvec12_full_{unit.precision}.h5")
    gauge_fallback = (
        roots.gauge_directory / f"gauge_{tag}_m0.05_seed42_c64.h5")
    nullvec_fallback = (
        roots.nullvec_directory / f"L{tag}_nvec12_full_c64.h5")
    notes: list[str] = []
    if unit.precision == "c128" and not (
            gauge_native.is_file() and nullvec_native.is_file()):
        gauge, nullvec = gauge_fallback, nullvec_fallback
        input_storage_precision = "c64"
        notes.append(
            "c128 precedence uses the canonical c64 gauge/null vectors with "
            "double-precision solver arithmetic (QUDA_MULTIGRID_DOUBLE mixed "
            "protocol); no c128 canonical asset is required")
    else:
        gauge, nullvec = gauge_native, nullvec_native
        input_storage_precision = unit.precision
    qio_prefix = roots.qio_directory / f"L{tag}_nvec12_quda"
    qio_manifest = roots.qio_directory / f"L{tag}_nvec12_quda.v1.json"
    cache_dir = roots.cache_directory(output_dir, unit)
    return {
        "gauge_path": str(gauge.resolve()),
        "nullvec_path": str(nullvec.resolve()),
        "input_storage_precision": input_storage_precision,
        "quda_nullvec_prefix": str(qio_prefix.resolve()),
        "quda_nullvec_manifest": str(qio_manifest.resolve()),
        "strict_cache_dir": str(cache_dir.resolve()),
        "roots": roots.as_dict(),
        "notes": notes,
    }


def _qio_artifact_paths(
        manifest_path: Path, qio_prefix: Path,
) -> tuple[list[Path], list[str]]:
    if not manifest_path.is_file():
        return [], [f"missing manifest={manifest_path}"]
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"invalid manifest={manifest_path}: {exc}"]
    artifacts = value.get("artifacts") if isinstance(value, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        return [], [f"manifest has no artifacts={manifest_path}"]
    paths: list[Path] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        raw = artifact.get("path")
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw)
        paths.append(path if path.is_absolute() else qio_prefix.parent / path)
    if not paths:
        return [], [f"manifest artifacts have no path={manifest_path}"]
    return paths, []


def validate_unit_assets(
        unit: MatrixUnit, assets: Mapping[str, Any],
) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    incompatible: list[dict[str, str]] = []
    required_files = (
        ("gauge", Path(str(assets["gauge_path"]))),
        ("nullvec", Path(str(assets["nullvec_path"]))),
    )
    for kind, path in required_files:
        if not path.is_file():
            missing.append({"kind": kind, "path": str(path)})

    qio_prefix = Path(str(assets["quda_nullvec_prefix"]))
    qio_manifest = Path(str(assets["quda_nullvec_manifest"]))
    qio_artifacts: list[Path] = []
    notes: list[str] = list(assets.get("notes") or [])
    if unit.precision != "c64" and unit.side == "quda":
        notes.append(
            "QUDA side consumes the single-precision QIO fine null vectors "
            "while the coarse hierarchy is built in double precision")
    if unit.side == "quda":
        qio_artifacts, manifest_errors = _qio_artifact_paths(
            qio_manifest, qio_prefix)
        for detail in manifest_errors:
            missing.append({
                "kind": "qio_manifest",
                "path": detail.split("=", 1)[-1],
            })
        for artifact in qio_artifacts:
            if not artifact.is_file():
                missing.append({
                    "kind": "qio_artifact",
                    "path": str(artifact),
                })

    result = {
        "ok": not missing and not incompatible,
        "assets": dict(assets),
        "qio_artifacts": [str(path) for path in qio_artifacts],
        "missing": missing,
        "incompatible": incompatible,
        "notes": notes,
    }
    return result


def _asset_error_reason(resolution: Mapping[str, Any]) -> str:
    details = [
        f"missing {item['kind']}={item['path']}"
        for item in resolution.get("missing", [])
    ]
    details.extend(
        f"incompatible {item['kind']}={item['path']} ({item['reason']})"
        for item in resolution.get("incompatible", [])
    )
    return "input_asset_missing: " + "; ".join(details)


def _option_segments(argv: Sequence[str]) -> dict[str, list[str]]:
    segments: dict[str, list[str]] = {}
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if not token.startswith("--"):
            index += 1
            continue
        name = token.split("=", 1)[0]
        segment = [token]
        index += 1
        while index < len(argv) and not str(argv[index]).startswith("--"):
            segment.append(str(argv[index]))
            index += 1
        segments[name] = segment
    return segments


def _merge_extra_args(
        unit_argv: Sequence[str], extra_args: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    unit_segments = _option_segments(unit_argv)
    extra_segments = _option_segments(extra_args)
    conflicts = [
        {
            "option": name,
            "resolution": "unit_specific",
            "unit_argv": unit_segments[name],
            "extra_argv": extra_segments[name],
        }
        for name in sorted(set(unit_segments) & set(extra_segments))
    ]
    filtered_extra: list[str] = []
    index = 0
    while index < len(extra_args):
        token = str(extra_args[index])
        if not token.startswith("--"):
            filtered_extra.append(token)
            index += 1
            continue
        name = token.split("=", 1)[0]
        index += 1
        while index < len(extra_args) and not str(extra_args[index]).startswith("--"):
            index += 1
        if name not in unit_segments:
            filtered_extra.extend(extra_segments[name])
    return [*unit_argv, *filtered_extra], conflicts


def _collector_command_report(
        unit: MatrixUnit,
        *,
        collector: Path,
        output_dir: Path,
        collector_interface: str,
        extra_args: Sequence[str] = (),
        assets: Mapping[str, Any] | None = None,
        asset_roots: AssetRoots | None = None,
) -> dict[str, Any]:
    roots = asset_roots or AssetRoots()
    resolved_assets = (
        dict(assets) if assets is not None
        else resolve_unit_assets(unit, roots, output_dir)
    )
    output = unit_output_path(output_dir, unit)
    unit_argv = [
        sys.executable,
        "-B",
        str(collector),
        "--side", unit.side,
        "--lattice", *[str(value) for value in unit.lattice],
        "--block", *[str(value) for value in BLOCK],
        "--levels", str(unit.levels),
        "--precision", unit.precision,
        "--gauge-path", str(resolved_assets["gauge_path"]),
        "--nullvec-path", str(resolved_assets["nullvec_path"]),
        "--strict-cache-dir", str(resolved_assets["strict_cache_dir"]),
        "--cache-expect", "miss",
        "--output", str(output),
    ]
    # Matrix-level resume already skips successful units.  Let the collector
    # start fresh for failed units so a corrected execution configuration is
    # not rejected against the failed record's old config hash.
    if unit.side == "quda":
        unit_argv.extend([
            "--quda-nullvec-prefix",
            str(resolved_assets["quda_nullvec_prefix"]),
            "--quda-nullvec-manifest",
            str(resolved_assets["quda_nullvec_manifest"]),
        ])
    if collector_interface == "frozen":
        unit_argv.extend([
            "--phases", "cold,warmup,steady",
            "--mpi-ranks", str(unit.mpi_ranks),
            "--process-grid", *[str(value) for value in unit.process_grid],
            "--device", unit.device,
        ])
        if unit.trace == "on":
            unit_argv.append("--allow-trace")
    elif collector_interface == "legacy":
        unit_argv.extend([
            "--profile", "smoke",
            "--repeats", "5",
        ])
        if unit.trace == "on":
            unit_argv.append("--allow-trace")
        if unit.mpi_ranks > 1:
            unit_argv = [
                "mpirun", "-np", str(unit.mpi_ranks), *unit_argv,
            ]
    else:
        raise ValueError(f"unknown collector interface {collector_interface!r}")
    command, conflicts = _merge_extra_args(unit_argv, extra_args)
    return {
        "argv": command,
        "argument_conflicts": conflicts,
        "assets": resolved_assets,
    }


def _collector_command(
        unit: MatrixUnit,
        *,
        collector: Path,
        output_dir: Path,
        collector_interface: str,
        extra_args: Sequence[str] = (),
        assets: Mapping[str, Any] | None = None,
        asset_roots: AssetRoots | None = None,
) -> list[str]:
    """Return the collector argv while keeping all mapping centralized."""
    return _collector_command_report(
        unit,
        collector=collector,
        output_dir=output_dir,
        collector_interface=collector_interface,
        extra_args=extra_args,
        assets=assets,
        asset_roots=asset_roots,
    )["argv"]


def collector_environment(
        unit: MatrixUnit, output_dir: Path, *,
        base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_environment is None else base_environment)
    paths = trace_paths(output_dir, unit)
    env["PYQCU_MG_MATRIX_UNIT_ID"] = unit.unit_id
    env["PYQCU_MG_MATRIX_MPI_RANKS"] = str(unit.mpi_ranks)
    env["PYQCU_MG_MATRIX_PROCESS_GRID"] = ",".join(
        str(value) for value in unit.process_grid)
    env["PYQCU_MG_MATRIX_DEVICE"] = unit.device
    if unit.trace == "on":
        env["PYQCU_STRICT_TRACE_FILE"] = paths["pyqcu"]
        env["PYQCU_QUDA_TRACE_FILE"] = paths["quda_log"]
        env["QUDA_MG_TRACE_FILE"] = paths["quda"]
    else:
        for name in TRACE_ENV_NAMES:
            env.pop(name, None)
    return env


def _state_path(output_dir: Path) -> Path:
    return output_dir / "matrix_state.jsonl"


def _read_state(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"state line {line_number} is not an object")
        records.append(value)
    return records


def _latest_entries(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        unit_id = record.get("unit_id")
        if isinstance(unit_id, str):
            latest[unit_id] = dict(record)
    return latest


def _append_state(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _git_describe(repository: Path = REPO) -> str:
    try:
        completed = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repository,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unknown"


def _subprocess_runner(
        argv: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float,
) -> RunResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            reason=f"timeout after {timeout:g}s",
        )
    except OSError as exc:
        return RunResult(
            returncode=127,
            stderr=repr(exc),
            blocked=True,
            reason=f"runner unavailable: {exc}",
        )
    return RunResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _blocked_pattern(text: str) -> str | None:
    lowered = text.lower()
    patterns = (
        ("p100", "P100 unavailable or unsupported"),
        ("sm_60", "sm_60/P100 build unavailable"),
        ("no kernel image", "no compatible GPU kernel image"),
        ("v100_unavailable", "V100 unavailable or unsupported"),
        ("requires a v100", "V100 unavailable or unsupported"),
        ("wrong_gpu", "requested GPU model unavailable"),
        ("cuda_unavailable", "CUDA unavailable"),
        ("mpi_abort", "MPI/QUDA collective abort"),
        ("mpirun", "MPI launcher unavailable"),
        ("qmp", "QMP/MPI runtime unavailable"),
        ("not supported", "requested device/rank mode unsupported"),
    )
    for needle, reason in patterns:
        if needle in lowered:
            return reason
    return None


def _read_collector_output(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _side_failure(document: Mapping[str, Any], side: str) -> tuple[str,
                                                                   str] | None:
    sides = document.get("sides")
    if not isinstance(sides, Mapping):
        return None
    record = sides.get(side)
    if not isinstance(record, Mapping):
        return None
    status = record.get("status")
    reason = record.get("reason")
    if status in ("ok", None):
        return None
    if isinstance(reason, Mapping):
        detail = str(reason.get("detail") or reason.get("code") or status)
    else:
        detail = str(reason or status)
    return str(status), detail


def _classify_result(
        unit: MatrixUnit,
        result: RunResult,
        *,
        output_path: Path,
        output_updated: bool,
) -> tuple[str, str | None, dict[str, Any] | None]:
    document = _read_collector_output(output_path) if output_updated else None
    combined = "\n".join((result.stdout, result.stderr))
    blocked_reason = _blocked_pattern(combined)

    if result.timed_out:
        return "timeout", result.reason or "collector timed out", document
    if result.blocked:
        return "blocked", result.reason or "collector preflight blocked", document
    if document is not None:
        failure = _side_failure(document, unit.side)
        if failure is not None:
            status, detail = failure
            if status in ("skipped", "blocked", "side_unsupported"):
                return "blocked", detail, document
            if status == "timeout":
                return "timeout", detail, document
            return "failed", detail, document
    if result.returncode != 0:
        if blocked_reason:
            return "blocked", blocked_reason, document
        return "failed", f"collector exited with {result.returncode}", document
    if document is None:
        return "failed", "collector returned success without fresh JSON output", None
    return "ok", None, document


def run_matrix(
        units: Sequence[MatrixUnit],
        *,
        output_dir: Path,
        collector: Path = DEFAULT_COLLECTOR,
        collector_interface: str = "frozen",
        extra_args: Sequence[str] = (),
        timeout: float = 7200.0,
        resume: bool = False,
        fail_fast: bool = False,
        runner: Runner = _subprocess_runner,
        source_version: str | None = None,
        asset_roots: AssetRoots | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(output_dir)
    latest = _latest_entries(_read_state(state_path))
    source_version = source_version or _git_describe()
    roots = asset_roots or AssetRoots()
    results: list[dict[str, Any]] = []

    for unit in units:
        previous = latest.get(unit.unit_id)
        if resume and previous is not None and previous.get("status") == "ok":
            skipped = dict(previous)
            skipped["git_describe"] = (
                skipped.get("git_describe") or source_version)
            results.append(skipped)
            continue

        output_path = unit_output_path(output_dir, unit)
        resolved_assets = resolve_unit_assets(unit, roots, output_dir)
        asset_resolution = validate_unit_assets(unit, resolved_assets)
        command_report = _collector_command_report(
            unit,
            collector=collector,
            output_dir=output_dir,
            collector_interface=collector_interface,
            extra_args=extra_args,
            assets=resolved_assets,
            asset_roots=roots,
        )
        argv = command_report["argv"]
        env = collector_environment(unit, output_dir)
        paths = trace_paths(output_dir, unit)
        started_at = _utc_now()
        started_ns = time.time_ns()
        common = {
            "unit_id": unit.unit_id,
            "unit": unit.as_dict(),
            "argv": argv,
            "started_at": started_at,
            "output_path": str(output_path),
            "trace_files": paths,
            "git_describe": source_version,
            "collector_interface": collector_interface,
            "assets": resolved_assets,
            "asset_resolution": asset_resolution,
            "argument_conflicts": command_report["argument_conflicts"],
        }
        _append_state(state_path, {
            **common,
            "ended_at": None,
            "status": "running",
            "error_code": None,
            "error_reason": None,
            "returncode": None,
        })

        if not asset_resolution["ok"]:
            record = {
                **common,
                "ended_at": _utc_now(),
                "status": "blocked",
                "error_code": "input_asset_missing",
                "error_reason": _asset_error_reason(asset_resolution),
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
            }
            _append_state(state_path, record)
            latest[unit.unit_id] = record
            results.append(record)
            if fail_fast:
                break
            continue

        try:
            result = runner(argv, REPO, env, timeout)
        except Exception as exc:  # The state must retain unexpected runner errors.
            result = RunResult(
                returncode=1,
                stderr=repr(exc),
                reason=f"runner exception: {exc}",
            )

        try:
            output_updated = (
                output_path.is_file()
                and output_path.stat().st_mtime_ns >= started_ns - 1_000_000
            )
        except OSError:
            output_updated = False
        status, error_reason, _document = _classify_result(
            unit,
            result,
            output_path=output_path,
            output_updated=output_updated,
        )
        record = {
            **common,
            "ended_at": _utc_now(),
            "status": status,
            "error_code": None,
            "error_reason": error_reason,
            "returncode": int(result.returncode),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        _append_state(state_path, record)
        latest[unit.unit_id] = record
        results.append(record)
        if fail_fast and status != "ok":
            break
    return results


def _walk_stage_events(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if ("seconds" in value and "level" in value
                and ("name" in value or "phase" in value)):
            yield value
            return
        for child in value.values():
            yield from _walk_stage_events(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_stage_events(child)


def _walk_residual_events(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "relative" in value:
            yield value
            return
        for child in value.values():
            yield from _walk_residual_events(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_residual_events(child)


def _parse_trace_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    stages: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split("\t")
        if not fields or fields[0] != "stage":
            continue
        try:
            if len(fields) >= 10:
                stages.append({
                    "level": int(fields[2]),
                    "name": fields[3],
                    "seconds": float(fields[4]),
                    "pre_iterations": int(fields[5]),
                    "post_iterations": int(fields[6]),
                    "coarse_iterations": int(fields[7]),
                })
            elif len(fields) >= 6:
                stages.append({
                    "level": int(fields[2]),
                    "name": fields[3],
                    "seconds": float(fields[4]),
                })
        except ValueError:
            continue
    return stages


def _phase_category(name: str) -> str:
    lowered = name.lower()
    if "pre_smoother" in lowered:
        return "pre_smoother"
    if "post_smoother" in lowered:
        return "post_smoother"
    if "restrict" in lowered:
        return "restrict"
    if "prolong" in lowered:
        return "prolongate"
    if "coarse_solver" in lowered or "bicgstab" in lowered:
        return "coarse_solver"
    return "other"


def _empty_layer() -> dict[str, Any]:
    return {
        "total_seconds": 0.0,
        "outer_iteration_seconds": 0.0,
        "iterations": None,
        "phases": {
            name: {"seconds": 0.0, "events": 0, "iterations": None}
            for name in PHASE_CATEGORIES
        },
    }


def _iteration_delta(event: Mapping[str, Any], category: str) -> int | None:
    field = {
        "pre_smoother": "pre_iterations",
        "post_smoother": "post_iterations",
        "coarse_solver": "coarse_iterations",
    }.get(category)
    if field is None:
        return None
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _side_record(document: Mapping[str, Any], side: str) -> Mapping[str, Any] | None:
    sides = document.get("sides")
    if not isinstance(sides, Mapping):
        return None
    value = sides.get(side)
    return value if isinstance(value, Mapping) else None


def _outer_iterations(document: Mapping[str, Any], side: str) -> dict[str, Any]:
    record = _side_record(document, side)
    if record is None:
        return {"samples": None, "median": None, "semantics": None}
    iterations = record.get("iterations")
    if isinstance(iterations, Mapping):
        return {
            "samples": iterations.get("samples"),
            "median": iterations.get("median"),
            "semantics": record.get("iteration_scope"),
        }
    return {"samples": None, "median": None, "semantics": None}


def _finest_level_residual(
        document: Mapping[str, Any], side: str,
) -> dict[str, Any] | None:
    record = _side_record(document, side)
    if record is not None:
        true_residual = record.get("true_residual")
        if isinstance(true_residual, Mapping):
            value = true_residual.get("max_rel")
            if isinstance(value, (int, float)):
                return {
                    "level": 0,
                    "value": float(value),
                    "kind": "full_operator_true_residual",
                    "source": f"sides.{side}.true_residual.max_rel",
                }
        steady = record.get("steady")
        if isinstance(steady, list):
            candidates: list[Mapping[str, Any]] = []
            for solve in steady:
                if not isinstance(solve, Mapping):
                    continue
                curve = solve.get("residual_curve")
                if isinstance(curve, list):
                    candidates.extend(
                        point for point in curve if isinstance(point, Mapping))
            if candidates:
                point = max(
                    candidates,
                    key=lambda item: int(item.get("iteration", -1)),
                )
                value = point.get("relative")
                if isinstance(value, (int, float)):
                    return {
                        "level": 0,
                        "value": float(value),
                        "kind": str(point.get("kind", "reported_residual")),
                        "source": f"sides.{side}.steady[*].residual_curve",
                    }

    residuals = list(_walk_residual_events(document))
    if residuals:
        point = residuals[-1]
        value = point.get("relative")
        if isinstance(value, (int, float)):
            return {
                "level": int(point.get("level", 0)),
                "value": float(value),
                "kind": str(point.get("name", point.get("kind", "reported_residual"))),
                "source": "trace residual events",
            }
    return None


def _extract_unit_metrics(
        document: Mapping[str, Any],
        unit: MatrixUnit,
        *,
        trace_files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stages = [
        dict(item) for item in _walk_stage_events(document)
        if isinstance(item.get("seconds"), (int, float))
    ]
    if trace_files:
        for value in trace_files.values():
            if isinstance(value, str):
                stages.extend(_parse_trace_tsv(Path(value)))

    layers = {str(level): _empty_layer() for level in range(unit.levels)}
    for stage in stages:
        try:
            level = int(stage["level"])
            seconds = float(stage["seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if seconds < 0.0:
            continue
        layer = layers.setdefault(str(level), _empty_layer())
        name = str(stage.get("name", stage.get("phase", "unknown")))
        if name == "outer_iteration":
            layer["outer_iteration_seconds"] += seconds
            continue
        category = _phase_category(name)
        phase = layer["phases"][category]
        phase["seconds"] += seconds
        phase["events"] += 1
        delta = _iteration_delta(stage, category)
        if delta is not None:
            if phase["iterations"] is None:
                phase["iterations"] = 0
            phase["iterations"] += delta
            if layer["iterations"] is None:
                layer["iterations"] = 0
            layer["iterations"] += delta
        layer["total_seconds"] += seconds

    record = _side_record(document, unit.side) or {}
    timing = record.get("timing")
    setup_seconds = None
    if isinstance(timing, Mapping):
        value = timing.get("setup_seconds")
        if isinstance(value, (int, float)):
            setup_seconds = float(value)

    warnings: list[str] = []
    if not stages:
        warnings.append("no stage timing events found")
    if any(float(layer["total_seconds"]) == 0.0 for layer in layers.values()):
        warnings.append("one or more levels have no observed stage timings")
    iterations = _outer_iterations(document, unit.side)
    if iterations["median"] is None:
        warnings.append("outer iterations missing")
    residual = _finest_level_residual(document, unit.side)
    if residual is None:
        warnings.append("finest-level residual missing")
    return {
        "setup_seconds": setup_seconds,
        "outer_iterations": iterations,
        "layers": layers,
        "phase_categories": list(PHASE_CATEGORIES),
        "finest_level_residual": residual,
        "timing_semantics": (
            "sum of observed trace stage intervals; nested stages are not additive"
        ),
        "warnings": warnings,
    }


def summarize_matrix(
        units: Sequence[MatrixUnit],
        *,
        output_dir: Path,
        full_expected_units: int = 144,
) -> dict[str, Any]:
    state_path = _state_path(output_dir)
    latest = _latest_entries(_read_state(state_path))
    unit_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    blocked: list[str] = []
    failed: list[str] = []
    status_counts: dict[str, int] = {}

    for unit in units:
        entry = latest.get(unit.unit_id)
        status = str(entry.get("status")) if entry else "missing"
        status_counts[status] = status_counts.get(status, 0) + 1
        output_path = (
            Path(str(entry["output_path"]))
            if entry and entry.get("output_path")
            else unit_output_path(output_dir, unit)
        )
        metrics: dict[str, Any] | None = None
        error_reason = None if entry is None else entry.get("error_reason")
        error_code = None if entry is None else entry.get("error_code")
        if status == "ok" and output_path.is_file():
            document = _read_collector_output(output_path)
            if document is None:
                status = "failed"
                status_counts["ok"] -= 1
                status_counts[status] = status_counts.get(status, 0) + 1
                error_reason = "output JSON is missing or malformed"
            else:
                trace_files = (
                    entry.get("trace_files")
                    if isinstance(entry.get("trace_files"), Mapping)
                    else None
                )
                metrics = _extract_unit_metrics(
                    document, unit, trace_files=trace_files)
        elif status == "ok":
            status = "failed"
            status_counts["ok"] -= 1
            status_counts[status] = status_counts.get(status, 0) + 1
            error_reason = "completed state has no output JSON"

        if status != "ok":
            missing.append(unit.unit_id)
            if status == "blocked":
                blocked.append(unit.unit_id)
            elif status in ("failed", "timeout", "missing"):
                failed.append(unit.unit_id)

        unit_rows.append({
            "unit_id": unit.unit_id,
            "unit": unit.as_dict(),
            "status": status,
            "error_code": error_code,
            "error_reason": error_reason,
            "assets": (
                None if entry is None else entry.get("assets")),
            "asset_resolution": (
                None if entry is None else entry.get("asset_resolution")),
            "argument_conflicts": (
                [] if entry is None else entry.get("argument_conflicts") or []),
            "output_path": str(output_path),
            "setup_seconds": None if metrics is None else metrics["setup_seconds"],
            "outer_iterations": (
                None if metrics is None else metrics["outer_iterations"]),
            "layers": None if metrics is None else metrics["layers"],
            "phase_categories": list(PHASE_CATEGORIES),
            "finest_level_residual": (
                None if metrics is None else metrics["finest_level_residual"]),
            "timing_semantics": (
                None if metrics is None else metrics["timing_semantics"]),
            "metric_warnings": [] if metrics is None else metrics["warnings"],
        })

    expected = len(units)
    completed = expected - len(missing)
    return {
        "schema": {"name": "pyqcu.mg-matrix-full.summary", "version": 1},
        "created_at": _utc_now(),
        "matrix_status": "complete" if not missing else "incomplete",
        "full_matrix_expected_units": int(full_expected_units),
        "scope_expected_units": expected,
        "scope_completed_units": completed,
        "outside_scope_units": int(full_expected_units) - expected,
        "coverage": {
            "expected_units": expected,
            "completed_units": completed,
            "missing_units": len(missing),
            "completed_percent": (
                100.0 * completed / expected if expected else 100.0),
            "status_counts": dict(sorted(status_counts.items())),
            "missing_unit_ids": missing,
            "blocked_unit_ids": blocked,
            "failed_unit_ids": failed,
        },
        "units": unit_rows,
    }


def _count_payload(units: Sequence[MatrixUnit]) -> dict[str, Any]:
    return {
        "side_case_count": len(units),
        "phase_record_count": sum(len(unit.phase_records) for unit in units),
        "full_matrix_side_case_count": len(build_units()),
        "full_matrix_phase_record_count": sum(
            len(unit.phase_records) for unit in build_units()),
    }


def _list_payload(
        units: Sequence[MatrixUnit], selection: Selection,
) -> dict[str, Any]:
    return {
        "schema": {"name": "pyqcu.mg-matrix-full.list", "version": 1},
        "selection": selection.as_dict(),
        "counts": _count_payload(units),
        "units": [unit.as_dict() for unit in units],
    }


def _dry_run_payload(
        units: Sequence[MatrixUnit],
        selection: Selection,
        *,
        output_dir: Path,
        collector: Path,
        collector_interface: str,
        extra_args: Sequence[str],
        asset_roots: AssetRoots,
) -> dict[str, Any]:
    plan_units = []
    for unit in units:
        resolved_assets = resolve_unit_assets(unit, asset_roots, output_dir)
        resolution = validate_unit_assets(unit, resolved_assets)
        report = _collector_command_report(
            unit,
            collector=collector,
            output_dir=output_dir,
            collector_interface=collector_interface,
            extra_args=extra_args,
            assets=resolved_assets,
            asset_roots=asset_roots,
        )
        command = report["argv"]
        env = collector_environment(unit, output_dir, base_environment={})
        plan_units.append({
            **unit.as_dict(),
            "command": command,
            "command_display": shlex.join(command),
            "environment_overrides": {
                key: env[key]
                for key in (
                    "PYQCU_MG_MATRIX_UNIT_ID",
                    "PYQCU_MG_MATRIX_MPI_RANKS",
                    "PYQCU_MG_MATRIX_PROCESS_GRID",
                    "PYQCU_MG_MATRIX_DEVICE",
                    *TRACE_ENV_NAMES,
                )
                if key in env
            },
            "output_path": str(unit_output_path(output_dir, unit)),
            "trace_files": trace_paths(output_dir, unit),
            "asset_resolution": resolution,
            "argument_conflicts": report["argument_conflicts"],
            "predicted_status": "ready" if resolution["ok"] else "blocked",
        })
    return {
        "schema": {"name": "pyqcu.mg-matrix-full.dry-run", "version": 1},
        "selection": selection.as_dict(),
        "counts": _count_payload(units),
        "collector": str(collector),
        "collector_interface": collector_interface,
        "output_dir": str(output_dir),
        "asset_roots": asset_roots.as_dict(),
        "units": plan_units,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable full Cartesian PyQCU/QUDA MultiGrid matrix")
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--gauge-root", type=Path, default=None)
    parser.add_argument("--nullvec-root", type=Path, default=None)
    parser.add_argument("--qio-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--collector-interface", choices=("legacy", "frozen"), default="frozen")
    parser.add_argument("--extra-args", action="append", default=[],
                        help="extra collector arguments; repeat or quote a shell string")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--only-side", action="append", default=[])
    parser.add_argument("--only-levels", action="append", default=[])
    parser.add_argument("--only-precision", action="append", default=[])
    parser.add_argument("--only-lattice", action="append", default=[])
    parser.add_argument("--only-trace", action="append", default=[])
    parser.add_argument("--only-device", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    selection = _selection_from_args(args)
    units = select_units(build_units(), selection)
    asset_roots = AssetRoots(
        asset_root=args.asset_root,
        gauge_root=args.gauge_root,
        nullvec_root=args.nullvec_root,
        qio_root=args.qio_root,
        cache_root=args.cache_root,
    )
    extra_args: list[str] = []
    for value in args.extra_args:
        extra_args.extend(shlex.split(value))

    if args.summarize:
        summary = summarize_matrix(units, output_dir=args.output_dir)
        summary_path = args.output_dir / "matrix_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "output": str(summary_path.resolve()),
            "matrix_status": summary["matrix_status"],
            "coverage": summary["coverage"],
        }, ensure_ascii=False, indent=2))
        return 0 if summary["matrix_status"] == "complete" else 2

    if args.dry_run:
        print(json.dumps(_dry_run_payload(
            units,
            selection,
            output_dir=args.output_dir,
            collector=args.collector,
            collector_interface=args.collector_interface,
            extra_args=extra_args,
            asset_roots=asset_roots,
        ), ensure_ascii=False, indent=2))
        return 0

    if args.list or not any((args.dry_run, args.summarize, args.execute)):
        print(json.dumps(_list_payload(units, selection),
                         ensure_ascii=False, indent=2))
        return 0

    results = run_matrix(
        units,
        output_dir=args.output_dir,
        collector=args.collector,
        collector_interface=args.collector_interface,
        extra_args=extra_args,
        timeout=args.timeout,
        resume=args.resume,
        fail_fast=args.fail_fast,
        asset_roots=asset_roots,
    )
    statuses = [record["status"] for record in results]
    if "failed" in statuses:
        return 1
    if any(status != "ok" for status in statuses):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"bench_mg_matrix_full: {exc}", file=sys.stderr)
        raise SystemExit(1)
