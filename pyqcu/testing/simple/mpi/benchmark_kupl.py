#!/usr/bin/env python3
"""Launch the standalone KUPL C++ operator benchmark as one MPI job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pyqcu.testing.simple._common import REPO, write_json
from pyqcu.testing.simple.benchmark_kupl import parse_output
from pyqcu.testing.simple.mpi._entrypoint import (
    add_mpi_job_options,
    benchmark_environment,
    finalize_mpi_job,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    add_mpi_job_options(
        parser, lattice=(16, 16, 16, 32), ranks=2,
        report="benchmark_kupl_mpi.json")
    args = finalize_mpi_job(parser, parser.parse_args(argv))
    if args.repeats <= 0 or args.iterations <= 0 or args.threads <= 0:
        parser.error("--repeats, --iterations and --threads must be positive")
    binary = (Path(args.binary).expanduser() if args.binary else
              REPO / "cpp/kupl/qcu/build/kupl_qcu_bench")
    module_arguments = [
        "--lattice", *(str(value) for value in args.lat),
        "--grid", *(str(value) for value in args.grid),
        "--iterations", str(args.iterations),
    ]
    command = [str(args.mpirun)]
    if args.allow_run_as_root:
        command.append("--allow-run-as-root")
    if args.oversubscribe:
        command.append("--oversubscribe")
    command.extend(("--bind-to", str(args.bind_to)))
    if args.map_by:
        command.extend(("--map-by", str(args.map_by)))
    command.extend(("-np", str(args.ranks), str(binary), *module_arguments))
    if args.dry_run:
        print(json.dumps({
            "status": "DRY-RUN",
            "mode": "delegated-launcher",
            "ranks": args.ranks,
            "grid": args.grid,
            "command": command,
            "trace_enabled": False,
            "quda_enable_tuning": "0",
            "report": str(args.report),
        }, ensure_ascii=False))
        return 0
    if not binary.is_file():
        output = Path(args.report).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        write_json(temporary, {
            "status": "SKIP",
            "backend": "kupl",
            "reason": f"missing {binary}",
            "report": str(output),
        })
        os.replace(temporary, output)
        print(json.dumps({
            "status": "SKIP", "reason": f"missing {binary}",
            "report": str(output),
        }))
        return 0

    environment = benchmark_environment(updates={
        "KUPL_OMP_THREADS": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
        "KUPL_NUMA_BIND": "0",
    })
    runs = []
    for repeat in range(args.repeats):
        try:
            completed = subprocess.run(
                command, cwd=REPO, env=environment, text=True,
                capture_output=True, timeout=args.timeout, check=False)
            run = {
                "repeat": repeat + 1,
                "returncode": int(completed.returncode),
                "parsed": parse_output(completed.stdout),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = exc.stdout or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode(errors="replace")
            run = {
                "repeat": repeat + 1,
                "returncode": 124,
                "parsed": {},
                "stdout": timeout_stdout,
                "stderr": f"timeout after {args.timeout:g} seconds",
            }
        runs.append(run)
        if run["returncode"] != 0:
            break
    status = "PASS" if len(runs) == args.repeats and all(
        item["returncode"] == 0 for item in runs) else "FAIL"
    report = {
        "status": status,
        "backend": "kupl",
        "binary": str(binary),
        "command": command,
        "ranks": args.ranks,
        "grid": args.grid,
        "threads": args.threads,
        "trace_enabled": False,
        "quda_enable_tuning": "0",
        "runs": runs,
    }
    output = Path(args.report).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    write_json(temporary, report)
    os.replace(temporary, output)
    print(json.dumps({
        "status": status,
        "report": str(output),
        "ranks": args.ranks,
        "grid": args.grid,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
