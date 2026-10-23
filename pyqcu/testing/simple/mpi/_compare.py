"""Sequential, isolated backend orchestration for simple-MPI comparisons."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyqcu.testing.simple._common import REPO, write_json
from pyqcu.testing.simple.mpi._common import resolve_grid
from pyqcu.testing.simple.mpi._entrypoint import benchmark_environment
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count


ENTRYPOINT_DIR = Path(__file__).resolve().parent


def validate_rank_grid(parser, lattice: Sequence[int], ranks: int,
                       grid: Sequence[int] | None) -> tuple[int, list[int]]:
    try:
        count = validate_rank_count(int(ranks))
        selected = list(resolve_grid(lattice, count, grid))
    except ValueError as exc:
        parser.error(str(exc))
    return count, selected


def child_command(entry: str, arguments: Sequence[str]) -> list[str]:
    return [sys.executable, "-B", str(ENTRYPOINT_DIR / entry), *arguments]


def child_report(report: str | Path, label: str) -> str:
    path = Path(report)
    return str(path.with_name(f"{path.stem}.{label}.json"))


def require_orchestrator_singleton(parser) -> None:
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            parser.error(f"invalid {name}={value!r}")
        if size > 1:
            parser.error(
                "compare_* is an orchestrator and must run outside MPI; "
                "each backend child launches its own MPI job")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return "" if value is None else str(value)


def run_jobs(
    jobs: Sequence[tuple[str, Sequence[str]]],
    *,
    report: str | Path,
    dry_run: bool,
    timeout: float | None,
) -> int:
    """Run MPI launchers one at a time so backend shared libraries never mix."""
    environment = benchmark_environment()
    records: list[dict[str, Any]] = []
    for label, command in jobs:
        record: dict[str, Any] = {"label": label, "command": list(command)}
        if dry_run:
            record["status"] = "DRY-RUN"
            records.append(record)
            continue
        try:
            completed = subprocess.run(
                command, cwd=REPO, env=environment,
                text=True, capture_output=True, timeout=timeout, check=False)
            record.update({
                "returncode": int(completed.returncode),
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
        except subprocess.TimeoutExpired as exc:
            record.update({
                "returncode": 124,
                "status": "FAIL",
                "stdout": _text(exc.stdout),
                "stderr": f"timeout after {timeout:g} seconds",
            })
        records.append(record)

    status = "DRY-RUN" if dry_run else (
        "PASS" if all(item["status"] == "PASS" for item in records)
        else "FAIL")
    report_payload = {
        "status": status,
        "mode": "isolated-mpi-orchestrator",
        "mixed_backend_mpi_job": False,
        "trace_enabled": False,
        "quda_enable_tuning": "0",
        "jobs": records,
    }
    output = Path(report).expanduser() if not dry_run else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial")
        write_json(temporary, report_payload)
        os.replace(temporary, output)
        report_payload["report"] = str(output)
    print(json.dumps(report_payload, ensure_ascii=False))
    return 0 if status in ("PASS", "DRY-RUN") else 1
