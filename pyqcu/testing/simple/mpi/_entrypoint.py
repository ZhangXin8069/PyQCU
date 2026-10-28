"""Shared execution helpers for the flat simple-MPI entry points.

The wrappers in this directory deliberately keep backend work in the existing
simple/QCU/QUDA/KUPL implementations.  This module only supplies MPI
collective symmetry, rank-count and geometry validation, trace-env isolation,
and atomic publication of rank-zero outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyqcu.testing.simple._common import REPO, resolve_path, write_json
from pyqcu.testing.simple.mpi._common import resolve_grid
from pyqcu.testing.simple.mpi.mpi_ranks_power_of_two import validate_rank_count
from pyqcu.tools._argparse import BooleanOptionalAction


TRACE_ENV_VARS = (
    "PYQCU_STRICT_TRACE_FILE",
    "PYQCU_QUDA_TRACE_FILE",
    "QUDA_MG_TRACE",
    "QUDA_MG_TRACE_FILE",
)


def benchmark_environment(
    base: Mapping[str, str] | None = None,
    *,
    updates: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with tracing and QUDA tuning disabled."""
    environment = dict(os.environ if base is None else base)
    for name in list(environment):
        if (name in TRACE_ENV_VARS or
                ("TRACE" in name and
                 name.startswith(("PYQCU_", "QUDA_")))):
            environment.pop(name, None)
    environment["QUDA_ENABLE_TUNING"] = "0"
    if updates:
        environment.update({str(key): str(value) for key, value in updates.items()})
    return environment


def require_launcher_singleton(parser: argparse.ArgumentParser) -> None:
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
                "this delegated launcher must run as one process outside MPI; "
                "it creates its own mpirun job")


def _comm_world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _collective_result(comm, function) -> tuple[int, str]:
    """Run ``function`` on rank zero and broadcast its exit state."""
    rank = int(comm.Get_rank())
    if rank == 0:
        try:
            result = function()
            state = (int(result or 0), "")
        except SystemExit as exc:
            state = (int(exc.code or 0), "")
        except BaseException:
            state = (1, traceback.format_exc())
    else:
        state = (0, "")
    status, error = comm.bcast(state, root=0)
    if rank == 0 and error:
        print(error, file=sys.stderr, end="" if error.endswith("\n") else "\n")
    return int(status), str(error)


def _wrapper_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        add_help=False,
        epilog=(
            "Remaining arguments are forwarded unchanged. --dry-run prints "
            "the auditable MPI command without executing the parent entry."),
    )
    parser.add_argument(
        "--mpi-ranks", "--ranks", dest="mpi_ranks", type=int, default=None,
        help="expected MPI_COMM_WORLD size; must match when launched by mpirun")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def _replace_option(
    arguments: Sequence[str], option: str, replacement: Path
) -> tuple[list[str], Path | None]:
    result: list[str] = []
    original: Path | None = None
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a path")
            original = Path(arguments[index + 1]).expanduser()
            result.extend((value, str(replacement)))
            index += 2
            continue
        prefix = option + "="
        if value.startswith(prefix):
            original = Path(value[len(prefix):]).expanduser()
            result.append(prefix + str(replacement))
            index += 1
            continue
        result.append(value)
        index += 1
    return result, original


def _resolved_output(value: Path) -> Path:
    return resolve_path(value, value)


def _stage_outputs(
    arguments: Sequence[str], flags: Iterable[str]
) -> tuple[list[str], list[tuple[Path, Path]]]:
    staged_arguments = list(arguments)
    staged: list[tuple[Path, Path]] = []
    token = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    for flag in flags:
        original: Path | None = None
        for index in range(len(staged_arguments)):
            if staged_arguments[index] == flag or staged_arguments[index].startswith(
                    flag + "="):
                _, original = _replace_option(
                    staged_arguments, flag, Path("__placeholder__"))
                break
        if original is None:
            continue
        final = _resolved_output(original)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(
            f".{final.stem}.{token}.partial{final.suffix}")
        staged_arguments, _ = _replace_option(
            staged_arguments, flag, temporary)
        staged.append((temporary, final))
    return staged_arguments, staged


def _publish_staged(staged: Sequence[tuple[Path, Path]]) -> None:
    replacements = {
        str(temporary): str(final) for temporary, final in staged}

    def replace_paths(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [replace_paths(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_paths(item) for key, item in value.items()}
        return value

    for temporary, _ in staged:
        if not temporary.is_file():
            raise FileNotFoundError(
                f"rank-zero parent entry did not produce {temporary}")
        if temporary.suffix.lower() == ".json":
            try:
                payload = json.loads(temporary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            else:
                write_json(temporary, replace_paths(payload))
    for temporary, final in staged:
        os.replace(temporary, final)


def _discard_staged(staged: Sequence[tuple[Path, Path]]) -> None:
    for temporary, _ in staged:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_staged_text(
    text: str, staged: Sequence[tuple[Path, Path]]
) -> str:
    for temporary, final in staged:
        text = text.replace(str(temporary), str(final))
    return text


def run_rank0_module(
    module_name: str,
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
    atomic_output_flags: Sequence[str] = (),
) -> int:
    """Run a serial parent entry only on rank zero under MPI symmetry."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _wrapper_parser(description)
    options, forwarded = parser.parse_known_args(arguments)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    module = importlib.import_module(module_name)
    if options.help:
        return int(module.main(["--help"]) or 0)

    comm = _comm_world()
    try:
        size = validate_rank_count(int(comm.Get_size()))
        if options.mpi_ranks is not None and int(options.mpi_ranks) != size:
            raise ValueError(
                f"--mpi-ranks={options.mpi_ranks} does not match "
                f"MPI_COMM_WORLD size {size}")
    except ValueError as exc:
        rank = int(comm.Get_rank())
        if rank == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if options.dry_run:
        payload = {
            "status": "DRY-RUN",
            "mode": "rank0-root",
            "parent": module_name,
            "mpi_size": size,
            "rank_zero_only": True,
            "atomic_output_flags": list(atomic_output_flags),
            "arguments": forwarded,
        }

        def emit() -> int:
            print(json.dumps(payload, ensure_ascii=False))
            return 0

        return _collective_result(comm, emit)[0]

    def execute() -> int:
        staged_args, staged = _stage_outputs(
            forwarded, atomic_output_flags)
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                result = module.main(staged_args)
            _publish_staged(staged)
            text = _replace_staged_text(output.getvalue(), staged)
            if text:
                print(text, end="" if text.endswith("\n") else "\n")
            return int(result or 0)
        except BaseException:
            _discard_staged(staged)
            text = output.getvalue()
            if text:
                print(text, file=sys.stderr,
                      end="" if text.endswith("\n") else "\n")
            raise

    return _collective_result(comm, execute)[0]


def run_native_module(
    module_name: str,
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
    atomic_output_flags: Sequence[str] = (),
) -> int:
    """Run an already-native MPI module on every rank after validation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _wrapper_parser(description)
    options, forwarded = parser.parse_known_args(arguments)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    module = importlib.import_module(module_name)
    if options.help:
        return int(module.main(["--help"]) or 0)

    comm = _comm_world()
    rank = int(comm.Get_rank())
    try:
        size = validate_rank_count(int(comm.Get_size()))
        if options.mpi_ranks is not None and int(options.mpi_ranks) != size:
            raise ValueError(
                f"--mpi-ranks={options.mpi_ranks} does not match "
                f"MPI_COMM_WORLD size {size}")
    except ValueError as exc:
        if rank == 0:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if options.dry_run:
        if rank == 0:
            print(json.dumps({
                "status": "DRY-RUN",
                "mode": "native-mpi",
                "module": module_name,
                "mpi_size": size,
                "arguments": forwarded,
            }, ensure_ascii=False))
        return 0

    local_status = 0
    local_error = ""
    staged: list[tuple[Path, Path]] = []
    run_arguments = list(forwarded)
    output = io.StringIO()
    try:
        if rank == 0 and atomic_output_flags:
            run_arguments, staged = _stage_outputs(
                forwarded, atomic_output_flags)
        if rank == 0:
            with contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                local_status = int(module.main(run_arguments) or 0)
        else:
            local_status = int(module.main(run_arguments) or 0)
    except BaseException:
        local_status = 1
        local_error = traceback.format_exc()
    gathered = comm.allgather((local_status, local_error))
    failed = [(index, item) for index, item in enumerate(gathered)
              if item[0] != 0]
    status = failed[0][1][0] if failed else 0
    if rank == 0:
        if status == 0 and staged:
            try:
                _publish_staged(staged)
            except BaseException:
                status = 1
                local_error = traceback.format_exc()
        elif status != 0:
            _discard_staged(staged)
        text = _replace_staged_text(output.getvalue(), staged)
        if text:
            print(text, end="" if text.endswith("\n") else "\n")
    if failed and rank == 0:
        failed_rank, failed_state = failed[0]
        print(
            f"error: native MPI module failed on rank {failed_rank}",
            file=sys.stderr)
        if failed_state[1]:
            print(failed_state[1], file=sys.stderr, end="")
    if rank == 0 and local_error and not failed:
        print(local_error, file=sys.stderr, end="")
    return int(comm.bcast(int(status), root=0))


def add_mpi_job_options(
    parser: argparse.ArgumentParser,
    *,
    lattice: Sequence[int],
    ranks: int,
    report: str,
) -> None:
    parser.add_argument("--lat", nargs=4, type=int, default=tuple(lattice))
    parser.add_argument("--grid", nargs=4, type=int, default=None)
    parser.add_argument("--ranks", type=int, default=ranks)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--bind-to", choices=("none", "numa"), default="none")
    parser.add_argument("--map-by", default=None)
    parser.add_argument("--oversubscribe", action=BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allow-run-as-root", action=BooleanOptionalAction,
                        default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--report", default=report)
    parser.add_argument("--dry-run", action="store_true")


def finalize_mpi_job(parser: argparse.ArgumentParser, args: argparse.Namespace):
    require_launcher_singleton(parser)
    try:
        args.ranks = validate_rank_count(args.ranks)
        args.lat = [int(value) for value in args.lat]
        args.grid = list(resolve_grid(args.lat, args.ranks, args.grid))
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout is not None and args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.allow_run_as_root is None:
        args.allow_run_as_root = hasattr(os, "geteuid") and os.geteuid() == 0
    return args


def build_mpirun_command(
    args: argparse.Namespace,
    module_name: str,
    module_arguments: Sequence[str],
) -> list[str]:
    command = [str(args.mpirun)]
    if args.allow_run_as_root:
        command.append("--allow-run-as-root")
    if args.oversubscribe:
        command.append("--oversubscribe")
    command.extend(("--bind-to", str(args.bind_to)))
    if args.map_by:
        command.extend(("--map-by", str(args.map_by)))
    command.extend((
        "-np", str(args.ranks), sys.executable, "-B", "-m", module_name,
        *module_arguments,
    ))
    return command


def run_delegated_mpi(
    module_name: str,
    args: argparse.Namespace,
    module_arguments: Sequence[str],
    *,
    environment_updates: Mapping[str, str] | None = None,
) -> int:
    """Run one isolated MPI child job and atomically publish its report."""
    final_report = resolve_path(args.report, Path(args.report).expanduser())
    final_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = final_report.with_name(
        f".{final_report.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.partial")
    child_report = final_report if args.dry_run else temporary_report
    child_arguments = [
        *module_arguments,
        "--grid", *(str(value) for value in args.grid),
        "--report", str(child_report),
    ]
    command = build_mpirun_command(args, module_name, child_arguments)
    dry_payload = {
        "status": "DRY-RUN",
        "mode": "delegated-launcher",
        "backend_module": module_name,
        "ranks": args.ranks,
        "grid": args.grid,
        "device_policy": "one Python process = one MPI rank",
        "trace_enabled": False,
        "quda_enable_tuning": "0",
        "report": str(final_report),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(dry_payload, ensure_ascii=False))
        return 0

    environment = benchmark_environment(updates=environment_updates)
    try:
        completed = subprocess.run(
            command, cwd=REPO, env=environment, text=True,
            capture_output=True, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        try:
            temporary_report.unlink()
        except FileNotFoundError:
            pass
        print(json.dumps({
            "status": "FAIL",
            "backend_module": module_name,
            "error": f"timeout after {args.timeout:g} seconds",
            "report_published": False,
        }, ensure_ascii=False), file=sys.stderr)
        return 124
    if completed.returncode != 0:
        report_published = False
        if temporary_report.is_file():
            try:
                failure_payload = json.loads(
                    temporary_report.read_text(encoding="utf-8"))
                failure_payload["status"] = "FAIL"
                failure_payload["child_returncode"] = int(completed.returncode)
                write_json(temporary_report, failure_payload)
                os.replace(temporary_report, final_report)
                report_published = True
            except (OSError, json.JSONDecodeError, TypeError):
                try:
                    temporary_report.unlink()
                except FileNotFoundError:
                    pass
        if completed.stdout:
            print(completed.stdout, file=sys.stderr,
                  end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr,
                  end="" if completed.stderr.endswith("\n") else "\n")
        print(json.dumps({
            "status": "FAIL",
            "backend_module": module_name,
            "returncode": int(completed.returncode),
            "error": "MPI child failed",
            "report": str(final_report) if report_published else None,
            "report_published": report_published,
        }, ensure_ascii=False))
        return int(completed.returncode) or 1
    if not temporary_report.is_file():
        try:
            temporary_report.unlink()
        except FileNotFoundError:
            pass
        if completed.stdout:
            print(completed.stdout, file=sys.stderr,
                  end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr,
                  end="" if completed.stderr.endswith("\n") else "\n")
        print(json.dumps({
            "status": "FAIL",
            "backend_module": module_name,
            "returncode": int(completed.returncode),
            "error": "child did not publish a report",
            "report_published": False,
        }, ensure_ascii=False))
        return 1

    os.replace(temporary_report, final_report)
    report: dict[str, Any] = {}
    try:
        report = json.loads(final_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    print(json.dumps({
        "status": report.get("status", "PASS"),
        "backend_module": module_name,
        "ranks": args.ranks,
        "grid": args.grid,
        "report": str(final_report),
        "trace_enabled": False,
        "quda_enable_tuning": "0",
    }, ensure_ascii=False))
    return 0 if report.get("status") != "FAIL" else 1
