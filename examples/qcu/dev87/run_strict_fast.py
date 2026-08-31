#!/usr/bin/env python3
"""Fast, tiered regression runner for the QUDA-strict MultiGrid path.

The tiers are cumulative: ``--tier 1`` runs tier0 and tier1, while tier2 is
reachable only through an explicit ``--tier 2``.  The default run stays on
small synthetic CPU tests and never starts the real-gauge setup.

Examples::

    python examples/qcu/dev87/run_strict_fast.py --list
    python examples/qcu/dev87/run_strict_fast.py
    python examples/qcu/dev87/run_strict_fast.py --tier 1 --fail-fast
    python examples/qcu/dev87/run_strict_fast.py --only cpu-smoke --fail-fast
    python examples/qcu/dev87/run_strict_fast.py --tier 1 --json result.json

Human progress and child output go to stderr.  Unless ``--json`` names a
file, stdout contains only the final machine-readable JSON document.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ENV_SH = REPO_ROOT / "env.sh"
QUDA_ENV_SH = HERE / "quda_env.sh"
JSON_CAPTURE_CHARS = 20_000

CPU_TEST = "examples/pyqcu/test_quda_multigrid.py"
CUDA_TEST = "examples/qcu/dev87/test_quda_transfer_cuda.py"
FUSED_CUDA_TEST = "examples/qcu/dev87/test_strict_fused_cuda.py"
GALERKIN_TEST = "examples/qcu/dev87/test_strict_galerkin_fast.py"
STRICT_BENCH = "examples/qcu/dev87/bench_strict_vs_quda.py"
STRICT_BENCH_OUTPUT = "examples/qcu/dev87/out/strict_vs_quda_benchmark.json"
STRICT_QUDA_NULLVEC = "data/L16x32x32x48_nvec12_quda_level_0_nvec_12"
STRICT_QUDA_MANIFEST = "data/L16x32x32x48_nvec12_quda.conversion.json"


@dataclass(frozen=True)
class CommandSpec:
    """One independently timed gate command."""

    name: str
    tier: int
    description: str
    argv: tuple[str, ...]
    timeout_s: float
    kind: str
    requirement: str


def _node(path: str, test: str) -> str:
    return f"{path}::{test}"


def _pytest(*nodes: str) -> tuple[str, ...]:
    # Disable pytest's cache so tier0/tier1 do not write into the repository.
    return (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "-p",
        "no:cacheprovider",
        *nodes,
    )


COMMANDS = (
    CommandSpec(
        name="cpu-smoke",
        tier=0,
        description=(
            "19 focused CPU checks for gamma basis, exports, FGMRES edge cases, strict "
            "mode/geometry guards, "
            "parity transfer/MATPC, strict assets/layouts, matrix-free mode and "
            "colored Galerkin batching"
        ),
        argv=_pytest(
            _node(CPU_TEST, "test_pyqcu_gamma_basis_matches_quda_degrand_rossi_table"),
            _node(CPU_TEST, "test_new_and_legacy_multigrid_are_parallel_exports"),
            _node(CPU_TEST, "test_fgmres_zero_rhs_stops_without_nan"),
            _node(CPU_TEST, "test_fgmres_breakdown_keeps_singular_case_finite"),
            _node(
                CPU_TEST,
                "test_complex_givens_is_unitary_under_phase_cancellation",
            ),
            _node(
                CPU_TEST,
                "test_strict_rejects_unimplemented_modes_and_odd_coarse_extent",
            ),
            _node(
                CPU_TEST,
                "test_single_parity_transfer_is_adjoint_and_keeps_full_coarse_geometry",
            ),
            _node(
                CPU_TEST,
                "test_quda_matpc_matches_left_preconditioned_block_elimination",
            ),
            _node(
                CPU_TEST,
                "test_transfer_galerkin_and_quda_yhat_conventions",
            ),
            _node(
                CPU_TEST,
                "test_strict_quda_hierarchy_coarsens_full_preconditioned_operator",
            ),
            _node(
                CPU_TEST,
                "test_strict_qcu_assets_preserve_quda_y_yhat_storage_and_actions",
            ),
            _node(
                CPU_TEST,
                "test_qcu_blocked_transfer_layout_matches_python_transfer",
            ),
            _node(
                CPU_TEST,
                "test_qcu_stencil_pack_matches_all_33_periodic_slots",
            ),
            _node(
                CPU_TEST,
                "test_qcu_stencil_degenerate_extent_and_strict_support_guard",
            ),
            _node(
                CPU_TEST,
                "test_compact_parity_layout_roundtrip_matches_qcu_mapping",
            ),
            _node(CPU_TEST, "test_matrix_free_mode_keeps_coarse_operator_lazy"),
            _node(
                GALERKIN_TEST,
                "test_strict_galerkin_fast_matches_columns_assets_and_matpc",
            ),
            _node(
                GALERKIN_TEST,
                "test_colored_memory_model_bounds_large_e48_block4_geometry_workspace",
            ),
            _node(
                GALERKIN_TEST,
                "test_e24_formal_geometry_workspace_tradeoff_is_exact",
            ),
        ),
        timeout_s=45.0,
        kind="pytest",
        requirement="CPU only; small synthetic fields",
    ),
    CommandSpec(
        name="cuda-strict",
        tier=1,
        description=(
            "CUDA P/R, 33-point stencil, strict X/Y/Yhat, MATPC, prepare and "
            "reconstruct primitives, nontrivial Clover MATPC on both parities, "
            "persistent recursive V-cycle and bounded complete solve"
        ),
        argv=_pytest(
            _node(
                CUDA_TEST,
                "test_quda_transfer_and_stencil_match_qcu_kernels",
            ),
            _node(
                CUDA_TEST,
                "test_quda_strict_transfer_coarse_and_matpc_match_cuda_kernels",
            ),
            _node(
                CUDA_TEST,
                "test_quda_strict_recursive_vcycle_matches_reference_and_arena",
            ),
            _node(
                CUDA_TEST,
                "test_fine_clover_prepare_and_reconstruct_match_existing_solver",
            ),
            _node(
                CUDA_TEST,
                "test_strict_fine_matpc_nontrivial_clover_matches_python_both_parities",
            ),
            _node(
                CUDA_TEST,
                "test_strict_fused_nontrivial_clover_matches_python_matpc_both_parities",
            ),
            _node(
                CUDA_TEST,
                "test_cuda_strict_solver_converges_with_bounded_krylov_arena",
            ),
        ),
        timeout_s=120.0,
        kind="pytest",
        requirement="CUDA device and importable libqcu/Cython backend",
    ),
    CommandSpec(
        name="cuda-fused-fgmres",
        tier=1,
        description=(
            "Fused C++ right-FGMRES integration, persistent workspace reuse, "
            "warm x0, budget/descriptor guards and complex128 dispatch"
        ),
        argv=_pytest(
            _node(
                FUSED_CUDA_TEST,
                "test_strict_fused_solver_api_reuses_arena_and_warm_x0",
            ),
            _node(
                FUSED_CUDA_TEST,
                "test_strict_fused_fgmres_rejects_budget_restart_shape_and_dtype",
            ),
            _node(
                FUSED_CUDA_TEST,
                "test_strict_fused_fgmres_complex128_dispatch_and_residual",
            ),
        ),
        timeout_s=120.0,
        kind="pytest",
        requirement="CUDA device and rebuilt libqcu/Cython fused-FGMRES ABI",
    ),
    CommandSpec(
        name="real-gauge-quda-comparison",
        tier=2,
        description=(
            "canonical dev87 real-gauge Strict PyQCU solve/MG and external "
            "QUDA/PyQUDA scaled-solution comparison"
        ),
        argv=(
            sys.executable,
            STRICT_BENCH,
            "--profile",
            "formal",
            "--side",
            "both",
            "--cache-expect",
            "hit",
            "--quda-nullvec-prefix",
            STRICT_QUDA_NULLVEC,
            "--quda-nullvec-manifest",
            STRICT_QUDA_MANIFEST,
            "--output",
            STRICT_BENCH_OUTPUT,
        ),
        timeout_s=1_200.0,
        kind="benchmark",
        requirement=(
            "CUDA, dev87 real gauge/null-vector data, Strict runtime cache, "
            "QUDA/PyQUDA and QUDA runtime libraries"
        ),
    ),
)


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and > 0")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative fast gates for QUDA-strict MultiGrid. Default: "
            "tier0 only; tier2 requires explicit --tier 2."
        )
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="highest cumulative tier to run (default: 0)",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(spec.name for spec in COMMANDS),
        metavar="GATE",
        help=(
            "run only the named gate; repeat for multiple gates and bypass cumulative "
            "selection; tier2 still requires --tier 2 (use --list to see names)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="list all commands and exit without sourcing env.sh or running setup",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        metavar="SECONDS",
        help="override each selected command's independent timeout",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed/timed-out command; also pass -x to pytest",
    )
    parser.add_argument(
        "--json",
        default="-",
        metavar="PATH",
        help="write final JSON to PATH; '-' keeps stdout JSON-only (default: -)",
    )
    return parser


def _display_timeout(seconds: float) -> str:
    return f"{seconds:g}s"


def _list_commands() -> None:
    print("QUDA-strict fast gates (cumulative; default tier0)")
    print("tier2 is never selected unless --tier 2 is passed explicitly.\n")
    for spec in COMMANDS:
        print(
            f"tier{spec.tier}  {spec.name}  "
            f"timeout={_display_timeout(spec.timeout_s)}"
        )
        print(f"  {spec.description}")
        print(f"  requires: {spec.requirement}")
        print(f"  command: {shlex.join(spec.argv)}")
    print("\n--timeout SECONDS overrides the timeout separately for every command.")
    print(
        "--only GATE runs one named gate without cumulative tier selection; "
        "repeat it for multiple gates."
    )


def _effective_argv(spec: CommandSpec, fail_fast: bool) -> list[str]:
    argv = list(spec.argv)
    if fail_fast and spec.kind == "pytest":
        # Insert immediately after ``python -m pytest``; node ids stay exact.
        argv.insert(3, "-x")
    return argv


def _env_wrapped_argv(
    argv: Sequence[str], *, source_quda_env: bool = False
) -> list[str]:
    # Positional forwarding avoids interpolating paths or node ids into shell code.
    if source_quda_env:
        script = 'set -e\nsource "$1"\nshift\nsource "$1"\nshift\nexec "$@"'
        return [
            "bash",
            "-c",
            script,
            "strict-fast-env",
            str(ENV_SH),
            str(QUDA_ENV_SH),
            *argv,
        ]
    script = 'set -e\nsource "$1"\nshift\nexec "$@"'
    return [
        "bash",
        "-c",
        script,
        "strict-fast-env",
        str(ENV_SH),
        *argv,
    ]


def _pump(stream, chunks: list[str]) -> None:
    """Mirror one child stream to stderr while retaining it for JSON."""

    try:
        for line in iter(stream.readline, ""):
            chunks.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()
    finally:
        stream.close()


def _stop_process_group(process: subprocess.Popen[str]) -> int:
    """Stop a timed-out command and all descendants started by run_all.py."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        return process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


_PYTEST_COUNTS = re.compile(
    r"(?P<count>\d+) (?P<outcome>passed|failed|skipped|errors?|xfailed|xpassed)"
)


def _pytest_counts(stdout: str, stderr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _PYTEST_COUNTS.finditer(stdout + "\n" + stderr):
        outcome = match.group("outcome")
        # The final pytest summary is the last occurrence of each outcome.
        counts[outcome] = int(match.group("count"))
    return counts


def _json_tail(value: str) -> tuple[str, bool, int]:
    length = len(value)
    if length <= JSON_CAPTURE_CHARS:
        return value, False, length
    marker = f"[... {length - JSON_CAPTURE_CHARS} earlier characters omitted ...]\n"
    return marker + value[-JSON_CAPTURE_CHARS:], True, length


def _run_command(
    spec: CommandSpec,
    timeout_s: float,
    fail_fast: bool,
) -> dict[str, object]:
    argv = _effective_argv(spec, fail_fast)
    child_env = os.environ.copy()
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    if spec.kind == "pytest":
        # These exact-node gates use only pytest core.  Avoid importing every
        # globally installed plugin on each tier's single pytest startup.
        child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if spec.tier < 2:
        # Keep backend logs from quick synthetic checks outside the repository.
        child_env["QCU_LOG_DIR"] = tempfile.gettempdir()

    started = time.perf_counter()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    timed_out = False
    launch_error: str | None = None
    returncode: int | None = None

    print(
        f"\n[RUN ] tier{spec.tier}/{spec.name} "
        f"timeout={_display_timeout(timeout_s)}",
        file=sys.stderr,
        flush=True,
    )
    print(f"       {shlex.join(argv)}", file=sys.stderr, flush=True)

    try:
        process = subprocess.Popen(
            _env_wrapped_argv(argv, source_quda_env=spec.kind == "benchmark"),
            cwd=REPO_ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        threads = (
            threading.Thread(
                target=_pump, args=(process.stdout, stdout_chunks), daemon=True
            ),
            threading.Thread(
                target=_pump, args=(process.stderr, stderr_chunks), daemon=True
            ),
        )
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _stop_process_group(process)
        for thread in threads:
            thread.join()
    except (OSError, subprocess.SubprocessError) as exc:
        launch_error = f"{type(exc).__name__}: {exc}"

    elapsed_s = time.perf_counter() - started
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    counts = _pytest_counts(stdout, stderr) if spec.kind == "pytest" else {}

    pytest_all_skipped = (
        spec.kind == "pytest"
        and counts.get("skipped", 0) > 0
        and counts.get("passed", 0) == 0
        and counts.get("failed", 0) == 0
        and counts.get("error", 0) == 0
        and counts.get("errors", 0) == 0
    )

    if launch_error is not None:
        status = "failed"
    elif timed_out:
        status = "timeout"
    # An allow_module_level CUDA skip plus an exact node id makes pytest return
    # rc=4 ("no collectors") even though its summary is unambiguously skipped.
    # Keep real missing-node errors fatal whenever the module can be collected.
    elif pytest_all_skipped:
        status = "skipped"
    elif spec.kind == "benchmark" and returncode == 2:
        # bench_strict_vs_quda reserves rc=2 for an explicit environment/input
        # skip.  Do not turn that into a false pass in the aggregate gate.
        status = "skipped"
    elif returncode != 0:
        status = "failed"
    else:
        status = "passed"

    label = {
        "passed": "PASS",
        "skipped": "SKIP",
        "failed": "FAIL",
        "timeout": "TIME",
    }[status]
    print(
        f"[{label:4}] tier{spec.tier}/{spec.name} in {elapsed_s:.3f}s",
        file=sys.stderr,
        flush=True,
    )
    if launch_error:
        print(f"       {launch_error}", file=sys.stderr, flush=True)

    stdout_tail, stdout_truncated, stdout_chars = _json_tail(stdout)
    stderr_tail, stderr_truncated, stderr_chars = _json_tail(stderr)
    return {
        "name": spec.name,
        "tier": spec.tier,
        "description": spec.description,
        "requirement": spec.requirement,
        "kind": spec.kind,
        "command": argv,
        "timeout_s": timeout_s,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 3),
        "timed_out": timed_out,
        "launch_error": launch_error,
        "pytest_counts": counts,
        "stdout": stdout_tail,
        "stdout_truncated": stdout_truncated,
        "stdout_chars": stdout_chars,
        "stderr": stderr_tail,
        "stderr_truncated": stderr_truncated,
        "stderr_chars": stderr_chars,
    }


def _not_run(spec: CommandSpec, timeout_s: float, reason: str) -> dict[str, object]:
    return {
        "name": spec.name,
        "tier": spec.tier,
        "description": spec.description,
        "requirement": spec.requirement,
        "kind": spec.kind,
        "command": list(spec.argv),
        "timeout_s": timeout_s,
        "status": "not_run",
        "returncode": None,
        "elapsed_s": 0.0,
        "timed_out": False,
        "launch_error": None,
        "pytest_counts": {},
        "stdout": "",
        "stdout_truncated": False,
        "stdout_chars": 0,
        "stderr": "",
        "stderr_truncated": False,
        "stderr_chars": 0,
        "reason": reason,
    }


def _write_json(summary: dict[str, object], destination: str) -> None:
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if destination == "-":
        sys.stdout.write(payload)
        sys.stdout.flush()
        return
    path = Path(destination).expanduser()
    path.write_text(payload, encoding="utf-8")
    print(f"JSON: {path.resolve()}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_only:
        _list_commands()
        return 0

    if args.only:
        selected_names = set(args.only)
        selected = [spec for spec in COMMANDS if spec.name in selected_names]
        selection_mode = "only"
    else:
        selected = [spec for spec in COMMANDS if spec.tier <= args.tier]
        selection_mode = "tier"

    if args.only and args.tier != 2 and any(spec.tier == 2 for spec in selected):
        parser.error(
            "--only 选择 tier 2 gate 时必须显式传入 --tier 2"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    stop_reason: str | None = None

    print(
        (
            f"QUDA-strict fast gate: only={','.join(args.only)}"
            if args.only
            else f"QUDA-strict fast gate: cumulative tier0..tier{args.tier}"
        )
        + f", {len(selected)} command(s); env={ENV_SH}",
        file=sys.stderr,
        flush=True,
    )

    if not ENV_SH.is_file():
        stop_reason = f"required environment script does not exist: {ENV_SH}"
        print(f"[FAIL] {stop_reason}", file=sys.stderr, flush=True)
        for spec in selected:
            timeout_s = args.timeout if args.timeout is not None else spec.timeout_s
            results.append(_not_run(spec, timeout_s, stop_reason))
    else:
        for index, spec in enumerate(selected):
            timeout_s = args.timeout if args.timeout is not None else spec.timeout_s
            result = _run_command(spec, timeout_s, args.fail_fast)
            results.append(result)
            if args.fail_fast and result["status"] in {"failed", "timeout"}:
                stop_reason = f"fail-fast after {spec.name}"
                for remaining in selected[index + 1 :]:
                    remaining_timeout = (
                        args.timeout
                        if args.timeout is not None
                        else remaining.timeout_s
                    )
                    results.append(
                        _not_run(remaining, remaining_timeout, stop_reason)
                    )
                break

    total_s = time.perf_counter() - started
    statuses = [str(result["status"]) for result in results]
    failed = any(status in {"failed", "timeout"} for status in statuses)
    skipped = any(status in {"skipped", "not_run"} for status in statuses)
    if not ENV_SH.is_file():
        failed = True
    status_counts = {
        status: statuses.count(status)
        for status in ("passed", "skipped", "failed", "timeout", "not_run")
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "runner": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "environment_script": str(ENV_SH),
        "tier_semantics": "cumulative",
        "selection_mode": selection_mode,
        "requested_tier": args.tier,
        "selected_tiers": sorted({spec.tier for spec in selected}),
        "selected_names": [spec.name for spec in selected],
        "only": list(args.only or []),
        "fail_fast": args.fail_fast,
        "timeout_override_s": args.timeout,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_s": round(total_s, 3),
        "outcome": "failed" if failed else "skipped" if skipped else "passed",
        "status_counts": status_counts,
        "stop_reason": stop_reason,
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "commands": results,
    }

    print(
        f"\nTOTAL {summary['outcome'].upper()} tier0..tier{args.tier}: "
        f"{total_s:.3f}s "
        f"(pass={status_counts['passed']}, skip={status_counts['skipped']}, "
        f"fail={status_counts['failed']}, timeout={status_counts['timeout']}, "
        f"not_run={status_counts['not_run']})",
        file=sys.stderr,
        flush=True,
    )
    try:
        _write_json(summary, args.json)
    except OSError as exc:
        print(f"[FAIL] cannot write JSON output: {exc}", file=sys.stderr)
        return 2
    if failed:
        return 1
    return 2 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
