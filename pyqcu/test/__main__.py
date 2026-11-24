"""``python -m pyqcu.test``: formal cross-backend correctness gates.

Nothing here imports a backend: ``--list`` prints the selected matrix and
exits before any runner module is resolved, so it works in an interpreter
where torch/KUPL/CUDA are absent.

Configuration vocabulary
------------------------
Every flag is normalised once, by :mod:`pyqcu.test.policy`, before it reaches
a case: ``--lattice-size``/``--lattice``, ``--lattice-grid`` and ``--mg-grid``
accept ``8 8 8 16``, ``8,8,8,16`` and ``[8,8,8,16]``; ``--dtype`` accepts
``c64``/``c128`` plus the ``complex64``/``complex128``/``float32``/``float64``
aliases; ``--backend`` accepts ``cpu`` (torch+kupl), ``dcu`` (cuda) and ``npu``
(torch); ``--device`` maps onto the runner-facing device string.  ``--list
--json`` and the ``--json`` report print the *effective* configuration, so a
run can be reproduced from its own output.

Expanded features
-----------------
``--profile NAME`` overlays a named option set from :mod:`pyqcu.test.profiles`
(repeatable; later names win, explicit flags win over every profile).
``--sweep``/``--matrix SPEC`` expands a bounded, deterministic matrix through
:mod:`pyqcu.test.matrix`; ``--list --json`` emits the expanded rows and a run
executes them sequentially.  ``--junit``/``--csv``/``--markdown`` write the
same report payload atomically through :mod:`pyqcu.test.report`; ``--json``
remains the native report.

Full modes (work ledger)
------------------------
``--mode {list,prepare,prewarm,solve,merge,run,validate}`` switches from "run
the selected cases once" to the seven-step work ledger the informal
``pyqcu/testing/kupl/test_kunpeng_all.py`` driver established:

``list``      enumerate the rows (no execution, no backend import)
``prepare``   create ``rows/reports/solutions/caches``, validate the inputs
``prewarm``   read each requested input once (page-cache warm) and hash it
``solve``     execute every row, write ``rows/<run_id>.json`` + the CSV
``merge``     rebuild the CSV and ``reports/merged.json`` from the row files
``run``       prepare -> prewarm -> solve -> merge
``validate``  fail unless rows, per-row reports and the CSV agree

The row is the unit: one case (per ``--sweep`` combination) at its effective
lattice/dtype/solver/seed and the full-mode fields (``--ranks``/``--threads``/
``--shm``/``--shared``/``--gauge-file``/``--source-file``).  ``run_id`` is the
SHA-1 of that configuration plus ``--run-label``, so ``--resume`` (skip
terminal rows) and ``--skip-failed`` (also skip ``FAIL``) are exact.
``--lattice-preset L24|L32`` names the release-acceptance latices.  A full mode
is explicit and selects tier 1; ``--quick`` keeps its tier-0 ``8 8 8 16`` c64
meaning, and neither ever turns into a heavy run silently.

``--ledger-export {none,csv,excel,both}`` additionally writes the
reference-compatible ledger the informal ``test_kunpeng_all.py`` driver owns -
its 100-column CSV and/or its 21-sheet workbook - under
``<run root>/ledger/``.  ``reports/rows.csv`` keeps its 30-column
``ROW_FIELDS`` semantics and is never replaced; ``--mode list`` and
``--mode validate`` are rejected (a plan has no status, and validate's rows
were produced by solve).

Multi-rank: one logical row
---------------------------
``mpirun -np 32``/``drun`` runs this command on 32 processes of the same row.
A full mode therefore behaves as **one** logical row: every rank executes and
reports through the per-row allgather
(:func:`pyqcu.test.common.gather_rank_outcome`), the row keeps the worst outcome
any rank reported, and only rank 0 (:func:`pyqcu.test.common.ledger_owner`)
writes ``rows/<run_id>.json``, ``reports/<run_id>.json``, the CSV, the solution
copies, stdout and the ``--json``/``--outdir`` payload.  Replica ranks print one
stderr line and nothing else, so 32 processes produce one report and one CSV
line per row instead of 32 racing writers.  The gathered verdict is identical on
every rank, which is what keeps ``--fail-fast`` from leaving 31 ranks inside a
collective.

``pyqcu.test.io``/``cache``/``native``/``torch_matrix``/``driver`` are wired
lazily: ``--list`` only checks that they exist (``find_spec``), nothing heavy
is imported before a mode actually needs it, and a present-but-raising module
is a non-empty failure reason - never a silent fallback or a fabricated PASS.

Exit codes
----------
``0``  no ``FAIL`` / ``ERROR`` / ``TIMEOUT``
``1``  any ``FAIL`` / ``ERROR`` / ``TIMEOUT``, or every selected case skipped
       (``--allow-all-skipped`` turns the all-skipped run into ``0``)
``2``  usage or environment error (bad flag, unwritable ``--json`` target)

``prepare``/``prewarm``/``merge``/``validate`` map "the operation could not
produce a consistent ledger" to ``1`` and a usage error to ``2``; a ``SKIP``
input (``--gauge-file`` not given) is a no-op there, not a failure.
``pyqcu --selftest`` uses ``2`` for a numerical failure; this port separates
"the numbers are wrong" (1) from "the command is wrong" (2), like
``python -m pyqcd.einsum.test``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import __version__
from . import ledger_export, l48, matrix, policy, profiles
from .common import (CASE_MODE, FULL_MODE_TIER, LATTICE_PRESETS, MODES,
                     FullModeOptions, ModePaths, Result, Test, TestReport,
                     UsageError, build_rows, grid_product, json_safe,
                     ledger_owner, module_availability, mpi_info, probe_env,
                     run_mode, summarize, write_json_atomic)

__all__ = ["build_parser", "main"]

#: ``--json -`` writes the report to stdout instead of a file.
JSON_STDOUT = "-"

#: Full modes that run through :func:`_run_mode` and therefore reach the export
#: hook: ``--mode list`` is a plan emitted by :func:`_emit_listing` and never
#: exports, and a plain case run has no row ledger at all.
#: ``PYQCU_TEST_LEDGER_EXPORT`` is validated for every mode listed here, so a
#: typo is the documented usage error instead of a late failure.  ``validate``
#: belongs to the list even though the *flag* is rejected for it (its rows were
#: produced by solve): the hook still runs and still reads the variable.
_EXPORT_HOOK_MODES: tuple[str, ...] = ("prepare", "prewarm", "solve", "merge",
                                       "validate", "run")

_EPILOG = """\
examples:
  python -m pyqcu.test --list                     # matrix, no backend import
  python -m pyqcu.test --quick                    # tier 0, 8 8 8 16, c64, seed 42
  python -m pyqcu.test --quick --json -           # machine-readable report
  python -m pyqcu.test --profile cpu --only torch/lattice
  python -m pyqcu.test --only torch/lattice,torch/gauge/random --list
  python -m pyqcu.test --sweep "dtype=c64,c128" --only torch/dslash --list
  python -m pyqcu.test --backend torch,kupl --repeat 5
  python -m pyqcu.test --backend cpu --device gpu --dtype complex128
  python -m pyqcu.test --lattice-size 8,8,8,16 --lattice-grid 1,1,1,2
  python -m pyqcu.test --solver cg --dslash clover
  python -m pyqcu.test --only torch/dslash --verbose
  python -m pyqcu.test --only torch/lattice --junit report.xml --csv report.csv
  python -m pyqcu.test --check                    # packaging/import self-check

full modes (work ledger; explicit, tier 1):
  python -m pyqcu.test --mode list --json -       # rows + effective paths
  python -m pyqcu.test --quick --mode list        # the tier-0 rows, no heavy run
  python -m pyqcu.test --mode prepare --outdir runs/l24 --lattice-preset L24
  python -m pyqcu.test --mode prewarm --gauge-file data/gauge_l24.h5
  python -m pyqcu.test --mode solve --resume --skip-failed --run-label cold
  python -m pyqcu.test --mode merge --csv runs/l24/reports/rows.csv
  python -m pyqcu.test --mode merge --outdir runs/l24 --ledger-export both
  python -m pyqcu.test --mode validate
  python -m pyqcu.test --mode run --lattice-preset L24 --ranks 16 --shm
  python -m pyqcu.test --mode solve --native solve --lattice-preset L48 \\
      --gauge-file data/gauge_l48_c64.h5 --source-file data/source_l48_c64.h5
"""


@dataclass(frozen=True)
class _Config:
    """Validated CLI configuration (the single source for a run)."""

    backends: tuple[str, ...]
    tags: tuple[str, ...]
    tier: int | None
    lattice: tuple[int, int, int, int]
    lattice_grid: tuple[int, int, int, int]
    mg_grid: tuple[int, int, int, int]
    dtype: str
    seed: int
    repeat: int
    warmup: int
    atol: float | None
    timeout: float | None
    quick: bool
    source: str
    gauge: str
    gauge_sigma: float
    mass: float | None
    kappa: float | None
    csw: float
    solver: str
    dslash: str
    device: str
    max_iter: int | None
    restart: int | None
    mg_levels: int | None
    coarse_tol_factor: float | None
    #: ``--reference-budget`` in **bytes** (already resolved from ``8GiB``/
    #: ``8e9``/a plain int by :func:`_budget_flag`), or ``None`` for the runner's
    #: own :data:`pyqcu.test.native.REFERENCE_BUDGET_BYTES` default.
    reference_budget: int | None
    mg_nu_pre: int | None
    mg_nu_post: int | None
    threads: int | None
    device_id: int | None
    #: ``--profile`` names in the order given (later names win; explicit flags
    #: win over every profile).  Empty when no profile was requested.
    profiles: tuple[str, ...] = ()
    #: Raw ``--sweep``/``--matrix`` chunks (validated by :mod:`pyqcu.test.matrix`
    #: before a listing or a run); empty for a single-configuration run.
    sweep: tuple[str, ...] = ()
    #: ``--native`` case kinds appended to the matrix for a full mode.
    native: tuple[str, ...] = ()
    #: ``--l48-two-node``: append the tier-1 L48 two-node matrix.
    l48_two_node: bool = False
    #: Effective full mode (``None`` for the plain "run the cases once" path).
    mode: str | None = None
    #: Full-mode configuration: work-ledger paths, ranks/shm/inputs/resume.
    full: FullModeOptions = field(default_factory=FullModeOptions)
    #: ``--ledger-export`` selection of the reference CSV/workbook export
    #: (``None`` when the flag was not given: ``PYQCU_TEST_LEDGER_EXPORT`` then
    #: decides, and unset means "no export").
    ledger_export: str | None = None


@dataclass(frozen=True)
class _SweepRun:
    """One expanded sweep row: the axis values and the ``Test`` they select."""

    index: int
    total: int
    combination: dict[str, Any]
    kwargs: dict[str, Any]
    test: Test

    @property
    def label(self) -> str:
        return f"[{self.index}/{self.total}]"


def _tier_value(text: str) -> int | None:
    """Parse ``--tier``: an int from :data:`policy.TIERS` or ``all``."""
    if str(text).strip().lower() == "all":
        return None
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"tier must be one of {policy.TIERS} or 'all', got {text!r}") from error
    if value not in policy.TIERS:
        raise argparse.ArgumentTypeError(
            f"tier must be one of {policy.TIERS} or 'all', got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (no backend import, no side effect)."""
    parser = argparse.ArgumentParser(
        prog="python -m pyqcu.test",
        description=(
            "Formal cross-backend correctness port for PyQCU: torch, KUPL and "
            "CUDA tier-0 cases at the 8x8x8x16 project lattice."),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--backend", nargs="+", default=None, metavar="LIST",
        help="backends, as a comma-separated list or several words: "
             "torch,cuda,kupl plus the aliases cpu (torch+kupl), dcu (cuda) "
             "and npu (torch), or all (default: all; a --profile may pin it)")
    parser.add_argument(
        "--profile", action="append", default=None, metavar="NAME",
        help="named run profile: quick, cpu, gpu, c128, clover, mg, full "
             "(default: none); repeatable, later names win, and an explicit "
             "flag always wins over a profile value")
    parser.add_argument(
        "--lattice-size", "--lattice", dest="lattice_size", nargs="+",
        default=None, metavar="N",
        help="lattice extents x y z t: '8 8 8 16', '8,8,8,16' or "
             "'[8,8,8,16]' (default: 8 8 8 16; extents even and >= 4)")
    parser.add_argument(
        "--lattice-grid", nargs="+", default=None, metavar="N",
        help="MPI rank grid x y z t, same list syntax (default: 1 1 1 1); a "
             "product that does not match the MPI size is a runner SKIP "
             "(mpi_grid), never a usage error")
    parser.add_argument(
        "--mg-grid", nargs="+", default=None, metavar="N",
        help="multigrid block grid x y z t, same list syntax "
             "(default: 2 2 2 2)")
    parser.add_argument(
        "--mode", choices=MODES, default=None, metavar="{list,prepare,prewarm,"
        "solve,merge,run,validate}",
        help="full-mode work ledger instead of one case run (default: none = "
             "run the selected cases once): list rows, prepare inputs, prewarm "
             "them, solve rows, merge the CSV, run the whole pipeline, or "
             "validate the ledger; selects tier 1 unless --quick/--tier")
    parser.add_argument(
        "--lattice-preset", type=lambda text: str(text).strip().upper(),
        choices=tuple(LATTICE_PRESETS), default=None, metavar="{L24,L32,L48}",
        help="release-acceptance lattice preset: L24 = 24 24 24 64, "
             "L32 = 32 32 32 64, L48 = 48 48 48 96 (default: none; cannot be "
             "combined with --lattice-size/--lattice or --quick)")
    parser.add_argument(
        "--quick", action="store_true",
        help="lock tier 0, lattice 8 8 8 16, dtype c64, seed 42, repeat 1")
    parser.add_argument(
        "--tier", type=_tier_value, default=None, metavar="{0,1,all}",
        help="run cases with tier <= N (default: all declared tiers)")
    parser.add_argument(
        "--native", nargs="+", default=None,
        choices=("dslash", "solve", "mg"), metavar="{dslash,solve,mg}",
        help="full modes: also plan native rows (pyqcu.test.native, tag "
             "<backend>/native/<kind>) for these case kinds on every selected "
             "native backend; those rows are driven by pyqcu.test.driver and "
             "consume --gauge-file/--source-file as real inputs (default: "
             "none = only the tier-0 matrix)")
    parser.add_argument(
        "--l48-two-node", dest="l48_two_node", action="store_true",
        help="also plan the tier-1 L48 two-node matrix (48 48 48 96, "
             "2x2x2x4, 32 ranks / 2 nodes): BiCGStab c64+c128, Legacy-MG "
             "c64+c128 (UNVERIFIED at 32 ranks), input/IO + independent "
             "residual; no Strict-MG and no CG (the L48 CG group is retired). "
             "Tier 1, so --quick still selects only the 19 tier-0 cases; "
             "needs a full mode (--mode list is enough) or --list")
    parser.add_argument(
        "--only", nargs="+", action="extend", default=None, metavar="TAG",
        help="run only these tags: a full tag (torch/dslash/wilson), a "
             "backend, a kind segment (dslash) or a tag prefix "
             "(torch/dslash); repeatable, and the values may be comma- or "
             "space-separated (--only a,b == --only a b == --only a --only b)")
    parser.add_argument(
        "--sweep", "--matrix", dest="sweep", action="append", default=None,
        metavar="SPEC",
        help="parameter sweep: key=value pairs separated by ';' (dtype=c64,"
             "c128;backend=torch,kupl;lattice=[8,8,8,16];[16,16,16,16]); "
             "repeatable, expanded in pyqcu.test.matrix AXIS_NAMES order; "
             "--list lists every expanded row, execution runs them in order "
             "(hard cap 512)")
    parser.add_argument(
        "--junit", default=None, metavar="PATH",
        help="also write the report as JUnit XML atomically (PATH, or '-' "
             "for stdout); requires a run, not --list")
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="also write the report as CSV atomically (PATH, or '-' for "
             "stdout); requires a run, not --list")
    parser.add_argument(
        "--markdown", default=None, metavar="PATH",
        help="also write the report as Markdown atomically (PATH, or '-' for "
             "stdout); requires a run, not --list")
    parser.add_argument(
        "--solver", choices=policy.SOLVER_CHOICES, default=None,
        help="solver selection (default: all): bicgstab -> solve/bistabcg, "
             "cg -> solve/cg, multigrid|mg -> mg/; cases that are neither a "
             "solver nor a multigrid check stay selected")
    parser.add_argument(
        "--dslash", choices=policy.DSLASH_CHOICES, default=None,
        help="dslash selection (default: all): wilson -> dslash/wilson, "
             "clover -> dslash/clover; all|both keep every dslash case and a "
             "non-dslash case is never filtered")
    parser.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="device selection (default: auto): cpu, gpu (cuda:0), dcu "
             "(dcu:0), npu (npu:0), or a concrete string such as cuda:3; auto "
             "forwards nothing and lets the backend choose, an unavailable "
             "device is SKIP (device_unavailable)")
    parser.add_argument(
        "--source", choices=policy.SOURCE_CHOICES, default=None,
        help="fermion source (default: point): point, ones or random")
    parser.add_argument(
        "--gauge", choices=policy.GAUGE_CHOICES, default=None,
        help="background gauge field (default: random): random, unit or ones")
    parser.add_argument(
        "--list", action="store_true",
        help="print the selected cases and exit (imports no backend)")
    parser.add_argument(
        "--json", nargs="?", const=JSON_STDOUT, default=None, metavar="PATH",
        help="write the JSON report: PATH, or '-' for stdout "
             "(also the value when the flag is given without a path)")
    parser.add_argument(
        "--repeat", type=int, default=None, metavar="N",
        help="timing samples per case (default: 5, --quick: 1)")
    parser.add_argument(
        "--warmup", type=int, default=None, metavar="N",
        help="discarded warmup calls per case (default: 1; 0 is rejected)")
    parser.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="deterministic gauge/source seed (default: 42)")
    parser.add_argument(
        "--dtype", choices=policy.DTYPE_INPUT_CHOICES, default=None,
        help="lattice dtype (default: c64): c64/c128, or the "
             "complex64/complex128/float32/float64 aliases, all normalised "
             "onto c64/c128")
    parser.add_argument(
        "--gauge-sigma", "--guage-sigma", dest="gauge_sigma", type=float,
        default=None, metavar="F",
        help="Gaussian width of the random SU(3) field (default: 0.1; "
             "--guage-sigma is accepted as a historical typo)")
    parser.add_argument(
        "--mass", type=float, default=None, metavar="F",
        help="fermion mass forwarded to the runners (default: runner's own)")
    parser.add_argument(
        "--kappa", type=float, default=None, metavar="F",
        help="hopping parameter forwarded to the runners (default: runner's "
             "own, kappa = 1/(2m+8))")
    parser.add_argument(
        "--csw", type=float, default=None, metavar="F",
        help="clover coefficient (default: 1.0; 0 is the plain Wilson limit)")
    parser.add_argument(
        "--max-iter", type=int, default=None, metavar="N",
        help="solver iteration budget forwarded to the runners")
    parser.add_argument(
        "--restart", type=int, default=None, metavar="N",
        help="Krylov restart length forwarded to the runners")
    parser.add_argument(
        "--coarse-tol-factor", type=float, default=None, metavar="F",
        help="Legacy-MG coarse-level tolerance multiplier: the level-1"
             " atol is atol*F*3^(level-1) (default 1e4, i.e. a very"
             " loose 1e-1 coarse target at atol=1e-5)")
    parser.add_argument(
        "--reference-budget", dest="reference_budget", type=str, default=None,
        metavar="BYTES",
        help="per-rank workspace budget of the independent residual reference, "
             "as a byte count with an optional unit: 8589934592, 8GiB, 8e9 "
             "(default: the runner's own 4GiB). The reference SKIPs with "
             "reference_workspace when its plan does not fit, so raise this "
             "only together with a matching memory plan")
    parser.add_argument(
        "--mg-levels", type=int, default=None, metavar="N",
        help="multigrid level count forwarded to the runners")
    parser.add_argument(
        "--mg-nu-pre", type=int, default=None, metavar="N",
        help="multigrid pre-smoothing steps forwarded to the runners")
    parser.add_argument(
        "--mg-nu-post", type=int, default=None, metavar="N",
        help="multigrid post-smoothing steps forwarded to the runners")
    parser.add_argument(
        "--threads", type=int, default=None, metavar="N",
        help="native thread count forwarded to the CPU runners")
    parser.add_argument(
        "--device-id", type=int, default=None, metavar="N",
        help="accelerator index forwarded to the native runners")
    parser.add_argument(
        "--atol", type=float, default=None, metavar="F",
        help="relative comparison tolerance; overrides the dtype's 'compare' "
             "gate only, never the solver residual gates")
    parser.add_argument(
        "--timeout", type=float, default=None, metavar="S",
        help="per-case wall-clock budget in seconds (default: per case)")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop after the first FAIL/ERROR/TIMEOUT (a SKIP never stops)")
    parser.add_argument(
        "--allow-all-skipped", action="store_true",
        help="exit 0 when every selected case was skipped")
    parser.add_argument(
        "--outdir", default=None, metavar="DIR",
        help="also write the JSON report to DIR/report.json (atomic); in a "
             "full mode this is the default work root for --rows/--reports/"
             "--solutions/--cache-dir")
    parser.add_argument(
        "--rows", default=None, metavar="DIR",
        help="full-mode row directory (default: <outdir>/rows, outdir default "
             "'pyqcu-test-runs'): rows/<run_id>.json is the source of truth")
    parser.add_argument(
        "--reports", default=None, metavar="DIR",
        help="full-mode report directory (default: <outdir>/reports): per-row "
             "reports, prepare/prewarm/merged.json")
    parser.add_argument(
        "--solutions", default=None, metavar="DIR",
        help="full-mode solution directory (default: <outdir>/solutions): "
             "runner-produced solution files are copied here")
    parser.add_argument(
        "--cache-dir", default=None, metavar="DIR",
        help="full-mode cache directory or Legacy-MG cache artifact (default: "
             "<outdir>/caches); an existing artifact is probed with "
             "pyqcu.test.cache, never rebuilt here")
    parser.add_argument(
        "--ledger-export", default=None,
        choices=ledger_export.EXPORT_SELECTIONS,
        metavar="{none,csv,excel,both}",
        help="full modes: also write the reference-compatible ledger export "
             "(the 100-column CSV and/or the 21-sheet workbook) under "
             "<run root>/ledger/, atomically; reports/rows.csv keeps its "
             "30-column pyqcu.test semantics and is never replaced "
             "(default: none; PYQCU_TEST_LEDGER_EXPORT is honoured when the "
             "flag is not given, --mode list/validate are rejected)")
    parser.add_argument(
        "--run-label", default="", metavar="LABEL",
        help="suffix appended to every run_id (full modes): keeps cold/warm or "
             "before/after runs of one configuration apart")
    parser.add_argument(
        "--resume", action="store_true",
        help="full modes: reuse rows whose status is PASS/SKIP instead of "
             "re-running them")
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="with --resume: also skip rows already marked FAIL (they are then "
             "never retried)")
    parser.add_argument(
        "--ranks", type=int, default=None, metavar="N",
        help="full-mode MPI rank count recorded in every row (default: the "
             "product of --lattice-grid)")
    parser.add_argument(
        "--shm", dest="shm", action="store_const", const=True, default=None,
        help="full modes: record shared-memory halo = true (default: unset)")
    parser.add_argument(
        "--no-shm", dest="shm", action="store_const", const=False,
        help="full modes: record shared-memory halo = false")
    parser.add_argument(
        "--shared", action="store_true",
        help="full modes: request the shared (single-node) execution path")
    parser.add_argument(
        "--gauge-file", default=None, metavar="PATH",
        help="full modes: HDF5 gauge artifact to prepare/prewarm (validated by "
             "pyqcu.test.io when present; not a generated field)")
    parser.add_argument(
        "--source-file", default=None, metavar="PATH",
        help="full modes: HDF5 source artifact to prepare/prewarm (validated by "
             "pyqcu.test.io when present)")
    parser.add_argument(
        "--validate", action="store_true",
        help="validate the row ledger: with --mode solve/run it is a post-step, "
             "alone it is --mode validate")
    parser.add_argument(
        "--verbose", action="store_true",
        help="print per-case metrics and failed gates to stderr")
    parser.add_argument(
        "--check", action="store_true",
        help="run the pyqcu.test self-check (import/packaging/policy/3.10 "
             "syntax) and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (0/1/2)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check:
        if _report_exports(args) or args.list or args.sweep or args.profile \
                or args.outdir is not None or args.mode is not None \
                or args.rows is not None or args.reports is not None \
                or args.solutions is not None or args.cache_dir is not None \
                or args.resume or args.skip_failed or args.validate \
                or args.native or args.l48_two_node \
                or args.ledger_export is not None:
            print("pyqcu.test: error: --check runs no numerics and cannot be "
                  "combined with --list/--sweep/--profile/--mode/--native/"
                  "--l48-two-node/--rows/--reports/--solutions/--cache-dir/"
                  "--resume/--skip-failed/--validate/--junit/--csv/--markdown/"
                  "--ledger-export/--outdir", file=sys.stderr)
            return policy.EXIT_CODES["usage"]
        from . import selfcheck

        return selfcheck.main(
            [] if args.json is None else ["--json", args.json])
    try:
        return _run(args, argv)
    except UsageError as error:
        print(f"pyqcu.test: error: {error}", file=sys.stderr)
        return policy.EXIT_CODES["usage"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _parse_backends(value: Any) -> tuple[str, ...]:
    """Normalize ``--backend`` into a canonical, duplicate-free tuple."""
    try:
        return policy.expand_backends(value)
    except ValueError as error:
        raise UsageError(str(error)) from error


def _parse_only(values: Any) -> tuple[str, ...]:
    """Normalize ``--only`` into a comma-and-space aware tag tuple.

    ``--only`` is declared as ``nargs="+"`` with ``action="extend"``, so the
    repeatable form (``--only a --only b``) and the space-separated form
    (``--only a b``) both arrive as several argv words, while the documented
    comma-separated form (``--only a,b``) arrives as a single word.  Splitting
    every word on commas *before* :meth:`pyqcu.test.common.Test.select` sees the
    patterns makes the three spellings select exactly the same cases; without
    it the comma form is passed through as one literal tag and is rejected as
    an unknown tag.

    Empty words are dropped (``--only ""`` keeps the default selection) and the
    order of first appearance is preserved, so ``--list`` reports the same
    filter the run uses.
    """
    if values is None:
        return ()
    tags: list[str] = []
    for value in values:
        for part in str(value).split(","):
            tag = part.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tuple(tags)


def _parse_shape(value: Any, flag: str, validate: Any) -> Any:
    """Parse a four-value list flag and validate it (None stays None)."""
    if value is None:
        return None
    try:
        return validate(policy.parse_int_list(value))
    except ValueError as error:
        raise UsageError(f"{flag}: {error}") from error


def _int_flag(value: int | None, flag: str, *, minimum: int = 1) -> int | None:
    """Validate an optional integer flag."""
    if value is None:
        return None
    if value < minimum:
        raise UsageError(f"{flag} must be >= {minimum}, got {value}")
    return int(value)


def _float_flag(value: float | None, flag: str, *,
                minimum: float | None = None,
                exclusive: bool = False) -> float | None:
    """Validate an optional float flag (finite, and above the bound if any)."""
    if value is None:
        return None
    if not math.isfinite(float(value)):
        raise UsageError(f"{flag} must be finite, got {value!r}")
    if minimum is not None:
        too_small = value <= minimum if exclusive else value < minimum
        if too_small:
            comparison = ">" if exclusive else ">="
            raise UsageError(
                f"{flag} must be {comparison} {minimum:g}, got {value!r}")
    return float(value)


def _budget_flag(value: Any, flag: str) -> int | None:
    """Parse ``--reference-budget`` into bytes (``None`` when not given).

    The accepted spellings are exactly the ones
    :func:`pyqcu.test.native._resolve_reference_budget` documents - ``8589934592``,
    ``"8GiB"``, ``"8e9"`` - so the CLI cannot drift from the runner: the flag
    hands the raw text to that parser and keeps its error text.  The module is
    imported lazily (it is stdlib-only, but ``--list`` must stay import-free) and
    a bad value is the documented usage error, never a late ``ValueError`` from
    inside a case.
    """
    if value is None:
        return None
    from . import native

    try:
        return int(native._resolve_reference_budget(value))
    except ValueError as error:
        raise UsageError(f"{flag}: {error}") from error


def _option_values(args: argparse.Namespace) -> dict[str, Any]:
    """The CLI destinations a :mod:`pyqcu.test.profiles` profile may pin.

    The two vocabularies differ in one place (``--lattice-size`` stores
    ``lattice_size``), so the translation lives here and nowhere else.
    ``None`` is "not given" and is exactly the value ``merge_profile`` treats
    as unset; a real ``0`` (``--tier 0``, ``--csw 0``) is kept.
    """
    return {
        "tier": args.tier,
        "lattice": args.lattice_size,
        "backend": args.backend,
        "dtype": args.dtype,
        "seed": args.seed,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "device": args.device,
        "solver": args.solver,
        "dslash": args.dslash,
        "mg_grid": args.mg_grid,
        "lattice_grid": args.lattice_grid,
        "gauge_sigma": args.gauge_sigma,
        "mass": args.mass,
        "csw": args.csw,
        "source": args.source,
        "gauge": args.gauge,
    }


def _resolve_profile_values(
        args: argparse.Namespace) -> tuple[dict[str, Any], frozenset[str],
                                           tuple[str, ...]]:
    """Overlay ``--profile`` names on the CLI values ("explicit wins").

    The profiles are merged in reverse command-line order with
    :func:`pyqcu.test.profiles.merge_profile`, which fills only unset values:
    the *last* named profile therefore supplies a value first, and an earlier
    profile (or, before all of them, the explicit flag) never overwrites it.
    The returned ``explicit`` set is what lets ``--quick`` distinguish "the
    user asked for ``--dtype c128``" (a conflict) from "the profile pinned
    ``c128``" (a default ``--quick`` replaces).
    """
    values = _option_values(args)
    explicit = frozenset(name for name, value in values.items()
                         if value is not None)
    names = tuple(str(item).strip() for item in (args.profile or ())
                  if str(item).strip())
    for name in reversed(names):
        try:
            values = profiles.merge_profile(values, name)
        except (TypeError, ValueError) as error:
            raise UsageError(f"--profile: {error}") from error
    return values, explicit, names


def _resolved_profile(names: Sequence[str]) -> dict[str, Any]:
    """The effective profile values for ``names`` (later names win)."""
    resolved: dict[str, Any] = {}
    for name in reversed(tuple(names)):
        try:
            resolved = profiles.merge_profile(resolved, name)
        except (TypeError, ValueError) as error:
            raise UsageError(f"--profile: {error}") from error
    return resolved


def _parse_sweep(values: Any) -> tuple[str, ...]:
    """Validate ``--sweep``/``--matrix`` chunks; ``()`` when not requested."""
    if values is None:
        return ()
    chunks = tuple(str(item) for item in values if str(item).strip())
    if not chunks:
        return ()
    try:
        matrix.expand_matrix(matrix.parse_spec(chunks))
    except ValueError as error:
        raise UsageError(f"--sweep: {error}") from error
    return chunks


def _ledger_export_environment() -> str:
    """The validated ``PYQCU_TEST_LEDGER_EXPORT`` value (``""`` when unset).

    The environment variable is the second gate of the reference-ledger export,
    next to ``--ledger-export``, and
    :func:`pyqcu.test.ledger_export.export_formats` raises :class:`ValueError` on
    a value it does not know.  Validating it here turns a typo into the
    documented usage error (exit code 2) *before* the mode runs, instead of a
    traceback from inside the exporter after the work is done.
    """
    value = os.environ.get(ledger_export.EXPORT_ENVIRONMENT, "")
    if not str(value).strip():
        return ""
    try:
        ledger_export.export_formats(value)
    except ValueError as error:
        raise UsageError(str(error)) from error
    return str(value).strip()


def _ledger_export_selection(args: argparse.Namespace,
                             mode: str | None) -> str | None:
    """Validated ``--ledger-export`` value (``None`` when the flag is absent).

    The flag selects the reference-compatible CSV/workbook export of a *full
    mode*.  It is rejected - not ignored - without ``--mode`` (a plain case run
    has no row ledger to export), with ``--mode list`` (a plan has no status
    and the reference vocabulary has no "planned" phase) and with ``--mode
    validate`` (its rows were produced by solve; ``--mode merge``/``run``
    rebuild the same table without mislabelling the phase).

    Without the flag the environment variable decides, so it is validated here
    for every mode that reaches the export hook
    (:func:`_ledger_export_environment`): a typo is a usage error either way,
    never a traceback.  A mode that does not export (``list``, a plain case
    run) does not consult it, and an explicit ``--ledger-export none`` keeps
    overriding it.
    """
    choice = getattr(args, "ledger_export", None)
    if choice is None:
        if mode in _EXPORT_HOOK_MODES:
            _ledger_export_environment()
        return None
    text = str(choice).strip().lower()
    if text not in ledger_export.EXPORT_SELECTIONS:
        raise UsageError(
            f"--ledger-export must be one of "
            f"{', '.join(ledger_export.EXPORT_SELECTIONS)}, got {choice!r}")
    if mode is None:
        raise UsageError(
            "--ledger-export writes the reference CSV/workbook of a full "
            "mode; add --mode solve|run|merge|prepare|prewarm "
            "(a plain case run has no row ledger)")
    if mode == "list":
        raise UsageError(
            "--ledger-export needs executed or merged rows; --mode list is a "
            "plan without a status (export the row files with --mode merge)")
    if mode == "validate":
        raise UsageError(
            "--ledger-export does not export a --mode validate document: its "
            "rows were produced by solve, and labelling them phase=validate "
            "would misstate their provenance; use --mode merge, which rebuilds "
            "the same table from the row files")
    return text


def _resolve_config(args: argparse.Namespace) -> _Config:
    """Validate the namespace and resolve the effective run configuration."""
    values, explicit, profile_names = _resolve_profile_values(args)
    sweep = _parse_sweep(args.sweep)
    quick = bool(args.quick)
    tier = values["tier"]
    preset = args.lattice_preset
    preset_lattice = LATTICE_PRESETS[preset] if preset is not None else None
    lattice = _parse_shape(values["lattice"], "--lattice-size",
                           policy.validate_lattice)
    if preset_lattice is not None:
        if lattice is not None:
            raise UsageError(
                f"--lattice-preset {preset} pins the lattice to "
                f"{list(preset_lattice)}; drop --lattice-size/--lattice (or "
                "the --profile value that pins it)")
        lattice = preset_lattice
    dtype = (policy.normalize_dtype(values["dtype"])
             if values["dtype"] is not None else None)
    seed = values["seed"]
    repeat = values["repeat"]
    warmup = values["warmup"]

    if quick:
        defaults = policy.quick_defaults()
        profile_label = ",".join(profile_names)
        conflicts = []
        if lattice is not None and lattice != defaults["lattice"]:
            conflicts.append(f"--lattice-size {list(lattice)}"
                             if "lattice" in explicit else
                             f"--profile {profile_label} (lattice="
                             f"{list(lattice)})")
        if dtype is not None and dtype != defaults["dtype"]:
            conflicts.append(f"--dtype {values['dtype']}"
                             if "dtype" in explicit else
                             f"--profile {profile_label} (dtype="
                             f"{values['dtype']})")
        if seed is not None and int(seed) != defaults["seed"]:
            conflicts.append(f"--seed {seed}"
                             if "seed" in explicit else
                             f"--profile {profile_label} (seed={seed})")
        if "tier" in explicit and tier is not None \
                and tier != defaults["tier"]:
            conflicts.append(f"--tier {args.tier}")
        if "repeat" in explicit and int(repeat) != defaults["repeat"]:
            conflicts.append(f"--repeat {repeat}")
        if preset is not None:
            conflicts.append(f"--lattice-preset {preset}")
        # A profile can pin a locked value too (c128 pins the dtype); quick
        # must not silently downgrade it to c64.  ``tier=None`` (the "full"
        # profile's every-tier value) is indistinguishable from "unset" in
        # merge_profile and is simply replaced by quick's tier 0.
        if sweep:
            axes = matrix.parse_spec(sweep)
            locked = [
                name for name, locked_value in (
                    ("lattice", defaults["lattice"]),
                    ("dtype", defaults["dtype"]),
                    ("seed", defaults["seed"]))
                if name in axes and any(
                    value != locked_value for value in axes[name])
            ]
            if locked:
                conflicts.append(
                    "--sweep " + matrix.format_spec(
                        {name: axes[name] for name in locked}))
        if conflicts:
            raise UsageError(
                "--quick locks tier=0, lattice=8 8 8 16, dtype=c64, seed=42 "
                f"and repeat=1; conflicting flag(s): {', '.join(conflicts)}")
        lattice = defaults["lattice"]
        dtype = defaults["dtype"]
        seed = defaults["seed"]
        tier = defaults["tier"]
        repeat = defaults["repeat"]
        # Every built-in profile pins warmup 1, but a custom profile must not
        # smuggle a different warmup into a --quick run; an explicit
        # --warmup is still honoured (the pre-profile behaviour).
        if "warmup" not in explicit:
            warmup = defaults["warmup"]
    else:
        lattice = policy.DEFAULT_LATTICE if lattice is None else lattice
        dtype = policy.DEFAULT_DTYPE if dtype is None else dtype
        seed = policy.DEFAULT_SEED if seed is None else seed

    # ``--validate`` alone is the validate mode; a full mode selects tier 1
    # unless the user pinned a tier or asked for --quick (which keeps tier 0).
    mode = args.mode or ("validate" if args.validate else None)
    if mode is not None and not quick and tier is None:
        tier = FULL_MODE_TIER
    ledger_export_choice = _ledger_export_selection(args, mode)
    native_kinds = tuple(str(item).strip().lower() for item in (args.native or ()))
    if native_kinds and mode is None:
        raise UsageError(
            "--native plans native rows and needs a full mode (--mode "
            "list|prepare|prewarm|solve|merge|run); a plain case run uses the "
            "tier-0 matrix")
    l48_two_node = bool(getattr(args, "l48_two_node", False))
    if l48_two_node and mode is None and not args.list:
        raise UsageError(
            "--l48-two-node plans the L48 two-node ledger rows and needs a "
            "full mode (--mode list is enough) or --list; a plain case run "
            "would execute L48 at the wrong rank count")
    if l48_two_node and sweep:
        raise UsageError(
            "--l48-two-node declares each case's dtype and geometry, so it "
            "cannot be combined with --sweep/--matrix")

    if int(seed) < 0:
        raise UsageError(f"--seed must be >= 0, got {seed}")
    device = (policy.DEFAULT_DEVICE if values["device"] is None
              else str(values["device"]).strip().lower())
    try:
        policy.device_for_runner(device)
    except ValueError as error:
        raise UsageError(f"--device: {error}") from error
    mass = _float_flag(values["mass"], "--mass")
    if mass is not None and 2.0 * float(mass) + 8.0 == 0.0:
        raise UsageError(
            f"--mass {mass:g} is degenerate: kappa = 1/(2*mass + 8) would be "
            "infinite")
    warmup = policy.DEFAULT_WARMUP if warmup is None else int(warmup)
    if warmup < 1:
        raise UsageError(
            f"--warmup must be >= 1, got {warmup}: the first call carries plan "
            "caches, native library binding and JIT, so warmup 0 measures a "
            "cold start, not the operator")
    repeat = policy.DEFAULT_REPEAT if repeat is None else int(repeat)
    if repeat < 1:
        raise UsageError(f"--repeat must be >= 1, got {repeat}")
    if args.timeout is not None and float(args.timeout) <= 0.0:
        raise UsageError(f"--timeout must be > 0, got {args.timeout}")
    if args.atol is not None and float(args.atol) <= 0.0:
        raise UsageError(f"--atol must be > 0, got {args.atol}")
    tags = _parse_only(args.only)

    lattice_grid = _parse_shape(values["lattice_grid"], "--lattice-grid",
                                policy.validate_lattice_grid)
    mg_grid = _parse_shape(values["mg_grid"], "--mg-grid",
                           policy.validate_mg_grid)
    backends = (("all",) if values["backend"] is None
                else _parse_backends(values["backend"]))
    resolved_grid = (policy.DEFAULT_LATTICE_GRID if lattice_grid is None
                     else lattice_grid)
    threads = _int_flag(args.threads, "--threads")
    ranks = _int_flag(args.ranks, "--ranks")
    if ranks is None:
        # ``--ranks`` defaults to the process grid the run already declares,
        # so a row never claims a rank count the lattice grid contradicts.
        ranks = grid_product(resolved_grid)
    if mode is not None and args.csv is not None \
            and str(args.csv).strip() == JSON_STDOUT:
        raise UsageError(
            "--csv - writes a report to stdout and cannot hold the full-mode "
            "row ledger; point --csv at a file (or drop it and use "
            "<reports>/rows.csv)")
    # Parsed once: the same value goes to the run options (FullModeOptions, the
    # mapping driver.run_row reads) and to the Test (so --list --json and the
    # report show the budget a row was planned against).
    reference_budget = _budget_flag(args.reference_budget, "--reference-budget")
    try:
        full = FullModeOptions(
            mode=mode,
            paths=ModePaths.resolve(
                root=args.outdir, rows=args.rows, reports=args.reports,
                solutions=args.solutions, cache=args.cache_dir, csv=args.csv),
            run_label=args.run_label,
            resume=bool(args.resume),
            skip_failed=bool(args.skip_failed),
            shared=bool(args.shared),
            ranks=ranks,
            threads=threads,
            shm=args.shm,
            gauge_file=args.gauge_file,
            source_file=args.source_file,
            validate=bool(args.validate),
            lattice_preset=preset,
            # Only an *explicit* ``--lattice-grid`` (or one a profile sets) is
            # forwarded: ``None`` means "no grid was requested", which the
            # native rows translate into a grid derived from ``--ranks``.
            # Passing the resolved default here would turn every multi-rank
            # run into a ``SKIP(mpi_grid)``.
            lattice_grid=lattice_grid,
            # Run-level solver knobs, so a full-mode row can vary them.
            restart=_int_flag(args.restart, "--restart"),
            mg_levels=_int_flag(args.mg_levels, "--mg-levels"),
            mg_grid=mg_grid,
            coarse_tol_factor=_float_flag(args.coarse_tol_factor,
                                          "--coarse-tol-factor",
                                          exclusive=True),
            # The independent residual's per-rank workspace budget.  It is a
            # *run* option (every native solve/MG row plans the reference
            # against it), so it travels through FullModeOptions.as_dict() into
            # driver.run_row exactly like the MG knobs above.
            reference_budget=reference_budget,
            # ``args.atol`` and not ``config.atol``: the config is resolved
            # below, after this construction.  An unset flag stays ``None`` so
            # the native runner keeps its own per-kind default.
            atol=_float_flag(args.atol, "--atol", minimum=0.0, exclusive=True),
            tier=(FULL_MODE_TIER if tier is None else int(tier)),
            repeat=int(repeat),
            warmup=int(warmup),
            timeout=None if args.timeout is None else float(args.timeout),
            allow_all_skipped=bool(args.allow_all_skipped),
            fail_fast=bool(args.fail_fast),
        )
    except (TypeError, ValueError) as error:
        raise UsageError(f"--mode: {error}") from error
    return _Config(
        backends=backends,
        tags=tags,
        tier=tier,
        lattice=lattice,
        lattice_grid=resolved_grid,
        mg_grid=policy.DEFAULT_MG_GRID if mg_grid is None else mg_grid,
        dtype=dtype,
        seed=int(seed),
        repeat=int(repeat),
        warmup=int(warmup),
        atol=None if args.atol is None else float(args.atol),
        timeout=None if args.timeout is None else float(args.timeout),
        quick=quick,
        source=(policy.DEFAULT_SOURCE if values["source"] is None
                else values["source"]),
        gauge=(policy.DEFAULT_GAUGE if values["gauge"] is None
               else values["gauge"]),
        gauge_sigma=(policy.DEFAULT_GAUGE_SIGMA
                     if values["gauge_sigma"] is None
                     else _float_flag(values["gauge_sigma"], "--gauge-sigma",
                                      minimum=0.0)),
        mass=mass,
        kappa=_float_flag(args.kappa, "--kappa", minimum=0.0, exclusive=True),
        csw=(policy.DEFAULT_CSW if values["csw"] is None
             else _float_flag(values["csw"], "--csw", minimum=0.0)),
        solver=(policy.DEFAULT_SOLVER if values["solver"] is None
                else values["solver"]),
        dslash=(policy.DEFAULT_DSLASH if values["dslash"] is None
                else values["dslash"]),
        device=device,
        max_iter=_int_flag(args.max_iter, "--max-iter"),
        restart=_int_flag(args.restart, "--restart"),
        mg_levels=_int_flag(args.mg_levels, "--mg-levels"),
        coarse_tol_factor=_float_flag(args.coarse_tol_factor,
                                     "--coarse-tol-factor",
                                     exclusive=True),
        reference_budget=reference_budget,
        # Also hand it to the Test so Options carries it into the runner;
        # _native_mg_overrides alone only reached the Case and was dropped.
        mg_nu_pre=_int_flag(args.mg_nu_pre, "--mg-nu-pre"),
        mg_nu_post=_int_flag(args.mg_nu_post, "--mg-nu-post"),
        threads=threads,
        device_id=_int_flag(args.device_id, "--device-id", minimum=0),
        profiles=profile_names,
        sweep=sweep,
        native=native_kinds,
        l48_two_node=l48_two_node,
        mode=mode,
        full=full,
        ledger_export=ledger_export_choice,
    )


def _native_mg_overrides(config: _Config) -> dict[str, Any]:
    """Only the MG knobs the caller actually set, for ``native.case``."""
    overrides: dict[str, Any] = {}
    for field in ("restart", "mg_levels", "mg_grid", "coarse_tol_factor"):
        value = getattr(config, field, None)
        if value is not None:
            overrides[field] = value
    return overrides


def _native_cases(config: _Config) -> tuple[Any, ...]:
    """Native full-mode cases requested by ``--native`` (full modes only).

    ``pyqcu.test.native`` is imported lazily, and only when ``--native`` asked
    for native rows, so a plain ``--list``/``--quick`` never imports it and the
    tier-0 matrix is untouched.  The cases are tagged ``<backend>/native/<kind>``
    (which is exactly what :func:`pyqcu.test.driver.run_row` claims) and are
    declared tier 0 so ``--quick --mode solve --native solve`` stays a cheap
    8x8x8x16 smoke instead of silently selecting nothing.

    A request that matches no selected native backend is a usage error: a run
    that silently planned fewer rows than asked for is indistinguishable from a
    clean one.
    """
    kinds = tuple(config.native)
    if not kinds:
        return ()
    from . import native

    backends = (tuple(native.BACKENDS) if "all" in config.backends
                else tuple(name for name in config.backends
                           if name in native.BACKENDS))
    if not backends:
        raise UsageError(
            f"--native needs a native backend ({'/'.join(native.BACKENDS)}); "
            f"the selection has none ({', '.join(config.backends) or 'none'})")
    try:
        return tuple(
            native.case(
                kind, backend=backend, tier=0, lattice=config.lattice,
                dtype=config.dtype, mass=config.mass, kappa=config.kappa,
                csw=config.csw, threads=config.threads, shm=config.full.shm,
                max_iter=config.max_iter, device=config.device,
                lattice_grid=config.lattice_grid,
                # These were silently dropped before 2026-11-24: every
                # native run used the NativeCase defaults whatever the
                # command line said, which made an MG parameter sweep
                # impossible (all points identical).
                **_native_mg_overrides(config))
            for backend in backends
            for kind in kinds
        )
    except ValueError as error:
        raise UsageError(f"--native: {error}") from error


def _build_test(config: _Config) -> Test:
    """Select the cases for ``config`` (raises ``UsageError`` when hopeless)."""
    base = Test.default()
    cases = base.cases + _native_cases(config)
    if config.l48_two_node:
        # Tier 1: ``--quick`` (tier 0) filters them out again, so the tier-0
        # 19-case matrix is unchanged.  The cases are ``pinned``: their L48
        # geometry and per-case dtype survive --lattice-size/--dtype.
        cases += l48.cases()
    test = replace(base, cases=cases).select(
        backends=config.backends,
        tags=config.tags,
        tier=config.tier,
        quick=config.quick,
        lattice=config.lattice,
        dtype=config.dtype,
        seed=config.seed,
        solver=config.solver,
        dslash=config.dslash,
        lattice_grid=config.lattice_grid,
        mg_grid=config.mg_grid,
        device=config.device,
        source=config.source,
        gauge=config.gauge,
        mass=config.mass,
        gauge_sigma=config.gauge_sigma,
        csw=config.csw,
        kappa=config.kappa,
        max_iter=config.max_iter,
        restart=config.restart,
        mg_levels=config.mg_levels,
        mg_nu_pre=config.mg_nu_pre,
        mg_nu_post=config.mg_nu_post,
        threads=config.threads,
        device_id=config.device_id,
    )
    if config.atol is not None:
        test = replace(test, atol=config.atol)
    if config.coarse_tol_factor is not None:
        # Options.as_kwargs only forwards non-None values, so this is the
        # only way the Legacy-MG coarse tolerance reaches a runner.  The
        # Case attribute alone was dropped by the harness.
        test = replace(test, coarse_tol_factor=config.coarse_tol_factor)
    if config.reference_budget is not None:
        # Same hop as coarse_tol_factor, for the same reason: the value must
        # survive ``select()`` so ``--list --json``/the report show the budget
        # the row was planned against (a full-mode native row reads it from
        # FullModeOptions.as_dict() in driver.run_row).
        test = replace(test, reference_budget=config.reference_budget)
    return test


# ---------------------------------------------------------------------------
# sweep planning
# ---------------------------------------------------------------------------
def _sweep_plan(config: _Config, test: Test) -> tuple[_SweepRun, ...]:
    """Expand ``--sweep`` into one validated :class:`Test` per combination.

    ``matrix.parse_spec``/``expand_matrix`` produce the deterministic
    Cartesian product; ``matrix.select_kwargs`` turns each combination into
    :meth:`Test.select` keywords, so a sweep runs through exactly the same
    validators as the equivalent flags.  Both the listing and the execution
    use this one plan: what ``--list --json`` shows is what will run.

    The hard cap is :data:`pyqcu.test.matrix.MAX_CASES` combinations **and**
    that many case runs in total (a combination may select several cases).
    Exceeding it is a usage error before anything executes or is written - a
    typo must not launch a day of compute.
    """
    if not config.sweep:
        return ()
    try:
        axes = matrix.parse_spec(config.sweep)
        combinations = matrix.expand_matrix(axes)
    except ValueError as error:
        raise UsageError(f"--sweep: {error}") from error
    total = len(combinations)
    plan: list[_SweepRun] = []
    for index, combination in enumerate(combinations, 1):
        try:
            kwargs = matrix.select_kwargs(combination)
        except ValueError as error:
            raise UsageError(f"--sweep row {index}/{total}: {error}") from error
        try:
            selected = test.select(**kwargs)
        except UsageError as error:
            raise UsageError(
                f"--sweep row {index}/{total} "
                f"({matrix.format_spec(combination)}): {error}") from error
        plan.append(_SweepRun(index=index, total=total,
                              combination=dict(combination),
                              kwargs=dict(kwargs), test=selected))
    runs = sum(len(run.test.cases) for run in plan)
    if runs > matrix.MAX_CASES:
        raise UsageError(
            f"--sweep: {runs} case runs ({total} combinations x selected "
            f"cases) exceed the hard cap of {matrix.MAX_CASES}; narrow "
            "--sweep or --only")
    return tuple(plan)


def _sweep_listing_payload(config: _Config, test: Test,
                           plan: Sequence[_SweepRun]) -> dict[str, Any]:
    """The expanded rows document of ``--sweep --list --json``.

    Rows are rendered per combination with that combination's effective
    :meth:`Test.config`, so a row's ``device``/``solver``/``dslash`` is what
    the run will use rather than an untouched CLI default.  ``runs`` keeps the
    grouping (which combination selected which tags) without adding fields to
    the closed :data:`pyqcu.test.matrix.ROW_FIELDS` row vocabulary.
    """
    axes = matrix.parse_spec(config.sweep)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for run in plan:
        case_rows = matrix.render_matrix(run.test.cases,
                                         config=run.test.config())
        rows.extend(case_rows)
        runs.append({
            "index": run.index,
            "configuration": json_safe(run.combination),
            "select": json_safe(run.kwargs),
            "tags": [case.tag for case in run.test.cases],
            "count": len(case_rows),
        })
    counts = [run["count"] for run in runs]
    per_combination = counts[0] if counts and len(set(counts)) == 1 else None
    return {
        "spec": matrix.format_spec(axes),
        "order": list(axes),
        "axes": {name: json_safe(list(values))
                 for name, values in axes.items()},
        "limit": matrix.MAX_CASES,
        "combinations": len(plan),
        "per_combination": per_combination,
        "count": len(rows),
        "cases": rows,
        "runs": runs,
        "selected": len(test.cases),
        "selected_cases": [case.as_dict() for case in test.cases],
    }


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------
def _listing_payload(config: _Config, test: Test,
                     plan: Sequence[_SweepRun] = ()) -> dict[str, Any]:
    """JSON-safe description of the selected matrix (``--list --json``).

    Carries the *effective* configuration (:meth:`Test.config`): every
    normalised value the run would use, including the defaults the user did
    not spell out, plus the CLI-only timing/selection details and the resolved
    profile (if any).  ``mode``/``rows``/``paths`` are always present: a reader
    can tell a plain case run from a full mode, see how many ledger rows the
    configuration expands to, and where they would be written.
    """
    payload: dict[str, Any] = {
        "package": policy.PACKAGE,
        "version": __version__,
        "quick": config.quick,
        "tier": config.tier,
        "backends": list(config.backends),
        "tags_filter": list(config.tags),
        "repeat": config.repeat,
        "warmup": config.warmup,
        "timeout": config.timeout,
        "count": len(test.cases),
    }
    payload.update(test.config())
    payload["cases"] = [case.as_dict() for case in test.cases]
    if config.l48_two_node:
        payload["l48_two_node"] = l48.plan_payload()
    if config.l48_two_node or l48.is_l48_lattice(test.lattice):
        # The retirement is part of the reported policy, not only of the docs:
        # a reader of --list --json must see that L48 has no CG group.
        payload["retired_groups"] = [dict(l48.RETIREMENT)]
    payload["mode"] = config.mode or CASE_MODE
    payload["paths"] = config.full.paths.as_dict()
    payload["modules"] = module_availability()
    payload["full_mode"] = config.full.as_dict()
    payload["rows"] = {
        "dir": str(config.full.paths.rows),
        "count": _mode_row_count(config, test, plan),
        "csv": str(config.full.paths.csv),
    }
    if config.profiles:
        payload["profile"] = list(config.profiles)
        payload["profile_resolved"] = json_safe(
            _resolved_profile(config.profiles))
    if config.sweep:
        payload["sweep_spec"] = list(config.sweep)
    return payload


def _lattice_text(value: Any) -> str:
    """Render a lattice/grid list as ``8x8x8x16`` for the human listing."""
    return "x".join(str(item) for item in value)


def _profile_text(config: _Config) -> str:
    """One-line resolved profile for the human ``--list`` output."""
    resolved = _resolved_profile(config.profiles)
    parts = []
    for key in ("backend", "device", "dtype", "lattice", "seed", "repeat",
                "solver", "dslash"):
        if key not in resolved:
            continue
        value = resolved[key]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value) \
                if key == "backend" else _lattice_text(value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _mode_row_count(config: _Config, test: Test,
                    plan: Sequence[_SweepRun] = ()) -> int:
    """How many ledger rows this configuration expands to (no execution)."""
    if plan:
        return sum(len(run.test.cases) for run in plan)
    return len(test.cases)


def _mode_rows(config: _Config, test: Test,
               plan: Sequence[_SweepRun] = ()) -> list[dict[str, Any]]:
    """Ledger rows of the effective configuration (one per case/combination)."""
    rows: list[dict[str, Any]] = []
    if plan:
        for run in plan:
            rows.extend(build_rows(run.test, config.full,
                                   combination=run.combination))
    else:
        rows.extend(build_rows(test, config.full))
    return rows


def _format_plan_row(row: dict[str, Any]) -> str:
    """Stable one-line rendering of a planned row (``--mode list``)."""
    return (f"row {str(row.get('run_id')):<24} {str(row.get('tag')):<24} "
            f"lattice={_lattice_text(row.get('lattice') or ())} "
            f"dtype={row.get('dtype')} ranks={row.get('ranks')} "
            f"threads={row.get('threads')} shm={row.get('shm')} "
            f"label={row.get('run_label') or '-'}")


def _emit_listing(config: _Config, test: Test, args: argparse.Namespace,
                  plan: Sequence[_SweepRun] = ()) -> int:
    """Print or write the matrix and return ``0`` (no backend import).

    Only rank 0 prints or writes the listing: a 32-process ``--mode list`` must
    not emit 32 copies of the same plan (and ``--json -`` stays one document).
    """
    owner = ledger_owner()
    payload = _listing_payload(config, test, plan)
    rows: list[dict[str, Any]] = []
    if config.mode == "list":
        rows = _mode_rows(config, test, plan)
        payload["rows"] = {**payload["rows"],
                           "items": [json_safe(row) for row in rows]}
    if plan:
        payload.update(_sweep_listing_payload(config, test, plan))
    if args.json == JSON_STDOUT:
        if owner:
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 0
    if not owner:
        return 0
    if args.json is not None:
        print(f"wrote {_write_json(args.json, payload)}", file=sys.stderr)
    if config.profiles:
        print(f"profile: {','.join(config.profiles)} "
              f"(resolved: {_profile_text(config)})")
    if config.mode == "list":
        print(f"mode: list rows={len(rows)} dir={payload['rows']['dir']} "
              f"csv={payload['rows']['csv']}")
        for row in rows:
            print(_format_plan_row(row))
    elif plan:
        print(f"sweep: {payload['spec']} -> {payload['combinations']} "
              f"combination(s), {payload['count']} row(s), "
              f"limit {payload['limit']}")
        for run in payload["runs"]:
            print(f"[{run['index']}/{payload['combinations']}] "
                  f"{matrix.format_spec(run['configuration'])} "
                  f"({run['count']} case(s): {','.join(run['tags']) or '-'})")
        for row in payload["cases"]:
            print(f"{row['tag']:<24} backend={row['backend']:<6} "
                  f"dtype={row['dtype']:<5} "
                  f"lattice={_lattice_text(row['lattice'])} "
                  f"device={row['device']} solver={row['solver']} "
                  f"dslash={row['dslash']}")
    else:
        for case in test.cases:
            requires = ",".join(case.requires) or "-"
            budget = "none" if case.timeout is None else f"{case.timeout:g}s"
            print(f"{case.tag:<24} {case.kind:<8} {case.runner:<32} "
                  f"timeout={budget} requires={requires}")
            if args.verbose:
                for key, spec in case.expect.items():
                    print(f"    expect {key}: {dict(spec)}", file=sys.stderr)
    if config.l48_two_node:
        manifest = payload["l48_two_node"]
        print(f"l48-two-node: lattice={_lattice_text(manifest['lattice'])} "
              f"ranks={manifest['ranks']} nodes={manifest['nodes']} "
              f"grid={_lattice_text(manifest['grid'])} tier={manifest['tier']} "
              f"status={manifest['status']}")
        for group in manifest["groups"]:
            tags = ",".join(case["tag"] for case in group["cases"]) or "-"
            print(f"  {group['name']:<13} {group['kind']:<8} "
                  f"evidence={group['evidence']:<10} {tags}")
        print("  retired: l48/cg (the tier-0 CG cases stay on 8 8 8 16)")
    if args.verbose:
        print(f"mode: {payload['mode']} rows={payload['rows']['count']} "
              f"dir={payload['rows']['dir']}", file=sys.stderr)
        print(f"config: {json.dumps(json_safe(payload), sort_keys=True)}",
              file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _one_line(value: Any, limit: int = 140) -> str:
    """Collapse whitespace and truncate for the one-line-per-case stdout."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _case_seconds(result: Result, *names: str) -> float | None:
    """First numeric timing field present in a result's metrics."""
    for name in names:
        value = result.metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _format_result(result: Result) -> str:
    """Stable one-line rendering: ``STATUS  tag  detail``.

    Timings are deliberately explicit (``core``, ``wall``, ``prep``,
    ``solve``): a bare ``0.010s`` is ambiguous about whether it is the timed
    kernel, the native setup or the whole case.
    """
    detail = ""
    if result.status == "PASS":
        parts: list[str] = []
        if result.seconds is not None:
            parts.append(f"core={result.seconds:.3f}s")
        wall = _case_seconds(result, "elapsed_seconds", "wall_seconds")
        if wall is not None:
            parts.append(f"wall={wall:.3f}s")
        prepare = _case_seconds(result, "prepare_seconds")
        if prepare is not None:
            parts.append(f"prep={prepare:.3f}s")
        solve = _case_seconds(result, "solve_seconds")
        if solve is not None:
            parts.append(f"solve={solve:.3f}s")
        for key in ("relative_error", "relative_residual", "gamma_deviation"):
            value = result.metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f"{key}={float(value):.2e}")
                break
        iterations = result.metrics.get("iterations")
        if isinstance(iterations, int) and not isinstance(iterations, bool):
            parts.append(f"it={iterations}")
        detail = " ".join(parts)
    elif result.status == "SKIP":
        token = f"{result.skip_token}: " if result.skip_token else ""
        detail = token + (result.reason or "")
    else:
        detail = result.reason or "; ".join(result.failed)
    return f"{result.status:<7} {result.tag:<24} {_one_line(detail)}"


def _print_detail(result: Result) -> None:
    """Verbose per-case detail; always stderr so stdout stays parseable."""
    print(f"  {result.tag}: status={result.status} runner={result.runner} "
          f"dtype={result.dtype} seed={result.seed} "
          f"mpi_size={result.mpi_size}", file=sys.stderr)
    for key in sorted(result.metrics):
        print(f"    {key}={result.metrics[key]}", file=sys.stderr)
    for line in result.failed:
        print(f"    failed: {line}", file=sys.stderr)
    if result.reason:
        print(f"    reason: {result.reason}", file=sys.stderr)


def _timing_totals(cases: Sequence[Result]
                  ) -> tuple[float, float, float, float, int, list[float],
                             list[float]]:
    """Aggregate timing fields across cases (missing fields count as zero).

    ``core`` is the timed operation, ``wall`` the per-case elapsed time,
    ``prepare``/``solve`` the native setup and solve phases when a runner
    reports them.  Min/max are the per-case ``seconds_min``/``seconds_max``
    samples; they are empty for a single-sample run.
    """
    core = wall = prepare = solve = 0.0
    samples = 0
    minima: list[float] = []
    maxima: list[float] = []
    for case in cases:
        if case.seconds is not None:
            core += float(case.seconds)
        wall += _case_seconds(
            case, "elapsed_seconds", "wall_seconds") or 0.0
        prepare += _case_seconds(case, "prepare_seconds") or 0.0
        solve += _case_seconds(case, "solve_seconds") or 0.0
        value = case.metrics.get("samples")
        if isinstance(value, int) and not isinstance(value, bool):
            samples += value
        else:
            repeat = case.metrics.get("repeat")
            samples += (repeat if isinstance(repeat, int)
                        and not isinstance(repeat, bool) and repeat > 0
                        else 1)
        low = _case_seconds(case, "seconds_min")
        high = _case_seconds(case, "seconds_max")
        if low is not None:
            minima.append(low)
        if high is not None:
            maxima.append(high)
    return core, wall, prepare, solve, samples, minima, maxima


def _print_summary(result: Any, config: _Config,
                   stream: Any = None) -> None:
    """Print the aggregate line, timing breakdown and skip hint.

    ``stream`` is stderr when the JSON report owns stdout (``--json -``), so a
    pipe always receives exactly one JSON document.  Timing is always printed:
    a bare total has repeatedly been mistaken for the operator time, while
    ``core``, ``wall``, ``prep`` and ``solve`` answer different questions.
    """
    stream = sys.stdout if stream is None else stream
    counts = result.counts
    print(f"summary: {len(result.cases)} cases, "
          f"{counts['PASS']} pass, {counts['SKIP']} skip, "
          f"{counts['FAIL']} fail, {counts['ERROR']} error, "
          f"{counts['TIMEOUT']} timeout, "
          f"{float(result.duration_seconds):.2f}s "
          f"({result.dtype}, {'x'.join(str(n) for n in result.lattice)})",
          file=stream)
    print(f"overall: {result.status}", file=stream)
    core, wall, prepare, solve, samples, minima, maxima = _timing_totals(
        result.cases)
    print(f"timing: total={float(result.duration_seconds):.3f}s "
          f"core={core:.3f}s wall={wall:.3f}s "
          f"prepare={prepare:.3f}s solve={solve:.3f}s "
          f"samples={samples}", file=stream)
    if minima and maxima:
        print(f"timing: core_min={min(minima):.3f}s "
              f"core_max={max(maxima):.3f}s", file=stream)
    if config.quick and config.repeat <= 1:
        print("note: --quick makes no performance claim "
              "(single sample, no median)", file=sys.stderr)
    if result.all_skipped and not result.allow_all_skipped:
        print("pyqcu.test: every selected case was skipped; "
              "pass --allow-all-skipped to exit 0", file=sys.stderr)


def _write_json(path: str | Path, payload: Any) -> Path:
    """Atomic JSON write; a write failure is an environment error (exit 2)."""
    try:
        return write_json_atomic(path, payload)
    except (OSError, TypeError, ValueError) as error:
        raise UsageError(
            f"cannot write JSON report to {str(path)!r}: "
            f"{type(error).__name__}: {error}") from error


def _report_exports(args: argparse.Namespace, *,
                    include_csv: bool = True) -> tuple[tuple[str, str], ...]:
    """``(format, target)`` pairs from ``--junit``/``--csv``/``--markdown``.

    In a full mode ``--csv`` is the *row ledger*, not a report export, so the
    caller passes ``include_csv=False`` and ``--csv`` never gets overwritten by
    a second CSV format.
    """
    exports: list[tuple[str, str]] = []
    for name in ("junit", "csv", "markdown"):
        if name == "csv" and not include_csv:
            continue
        target = getattr(args, name, None)
        if target is not None:
            exports.append((name, str(target)))
    return tuple(exports)


def _validate_export_targets(
        args: argparse.Namespace, *,
        json_to_stdout: bool,
        exports: Sequence[tuple[str, str]] | None = None
) -> tuple[tuple[str, str], ...]:
    """Reject ambiguous export targets before any case runs or file is written.

    Checking first turns ``--json - --csv -`` and ``--junit p --csv p`` into a
    usage error (exit 2) without paying for a run or leaving a partial stdout
    document behind.
    """
    exports = _report_exports(args) if exports is None else tuple(exports)
    owners: dict[str, str] = {}
    for fmt, target in exports:
        if target in owners:
            raise UsageError(
                f"--{owners[target]} and --{fmt} both target {target!r}; one "
                "path cannot hold two report formats")
        owners[target] = fmt
        if target == JSON_STDOUT and json_to_stdout:
            raise UsageError(
                f"--{fmt} - would share stdout with --json -; write one of "
                "them to a file")
    return exports


def _write_exports(payload: Any, exports: Sequence[tuple[str, str]], *,
                   json_to_stdout: bool) -> None:
    """Write the extra report formats with :func:`pyqcu.test.report.write_report`.

    ``write_report`` is atomic (temporary sibling + ``fsync`` + ``os.replace``)
    and infers nothing here: the format is exactly the flag that asked for it.
    A ``-`` target is rendered to stdout - ``write_report`` deliberately does
    not special-case it.  Targets are validated by
    :func:`_validate_export_targets` before the run; the checks are repeated
    here so the helper is safe to call on its own.
    """
    if not exports:
        return
    from .report import render, write_report

    owners: dict[str, str] = {}
    for fmt, target in exports:
        if target in owners:
            raise UsageError(
                f"--{owners[target]} and --{fmt} both target {target!r}; one "
                "path cannot hold two report formats")
        owners[target] = fmt
        if target == JSON_STDOUT and json_to_stdout:
            raise UsageError(
                f"--{fmt} - would share stdout with --json -; write one "
                "of them to a file")
    for fmt, target in exports:
        if target == JSON_STDOUT:
            sys.stdout.write(render(payload, fmt))
            continue
        try:
            written = write_report(target, payload, fmt)
        except (OSError, TypeError, ValueError) as error:
            raise UsageError(
                f"--{fmt}: cannot write {target!r}: "
                f"{type(error).__name__}: {error}") from error
        print(f"wrote {written} ({fmt})", file=sys.stderr)


def _rank_fields() -> dict[str, Any]:
    """``mpi`` block of every report payload: who wrote it, and in what world.

    A 32-process ``--mode solve`` is **one logical row**: every rank executes and
    reports through the per-row allgather
    (:func:`pyqcu.test.common.gather_rank_outcome`), while rank 0 alone writes
    ``rows/``, ``reports/``, the CSV and this payload.  Recording it makes the
    report reproducible and tells a reader whether a document came from a
    single process or from the owning rank of an MPI world.
    """
    size, rank = mpi_info()
    return {
        "size": size,
        "rank": rank,
        "ledger_writer": ledger_owner(rank=rank),
        "row_reduction": "allgather",
    }


def _print_rank_note() -> None:
    """One stderr line when this process is a replica of an MPI world.

    stdout is the report stream (``--json -`` is one JSON document), so the
    notice goes to stderr and a replica rank prints nothing else: 32 processes
    must not produce 32 reports or 32 summaries.
    """
    size, rank = mpi_info()
    if size <= 1:
        return
    print(f"pyqcu.test: rank {rank}/{size}: one logical row - every rank "
          "computes and reports through the per-row allgather, rank 0 owns "
          "rows/reports/CSV, stdout and the report payload",
          file=sys.stderr)


def _emit_payload(payload: Any, args: argparse.Namespace, *,
                  to_stdout: bool,
                  exports: Sequence[tuple[str, str]] | None = None) -> None:
    """Emit ``--json``, ``--outdir`` and the extra formats for one run.

    Only the ledger owner (rank 0) emits: a replica rank of a 32-process run
    would otherwise write the same ``--json``/``--outdir``/``--junit`` target 31
    more times, and ``--json -`` would print 32 documents that no parser
    accepts.
    """
    if not ledger_owner():
        return
    if to_stdout:
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    elif args.json is not None:
        print(f"wrote {_write_json(args.json, payload)}", file=sys.stderr)
    if args.outdir is not None:
        target = Path(args.outdir).expanduser() / "report.json"
        print(f"wrote {_write_json(target, payload)}", file=sys.stderr)
    _write_exports(payload,
                   _report_exports(args) if exports is None else exports,
                   json_to_stdout=to_stdout)


def _mode_fields(config: _Config, rows: int) -> dict[str, Any]:
    """``mode``/``rows``/``paths``/``full_mode`` for any report payload.

    The same block is merged into ``TestReport.options`` and into the top level
    of the ``--json`` document, so "which mode ran, how many rows, where would
    they be written" is answerable from one run's own output.
    """
    document = config.full.as_dict()
    return {
        "mode": config.mode or CASE_MODE,
        "paths": document["paths"],
        "rows": {
            "dir": str(config.full.paths.rows),
            "count": int(rows),
            "csv": str(config.full.paths.csv),
        },
        "full_mode": document,
        "mpi": _rank_fields(),
        "modules": module_availability(),
    }


def _run_single(config: _Config, test: Test, args: argparse.Namespace,
                argv: Sequence[str] | None) -> int:
    """Execute one configuration and emit its report.

    In a multi-process launch only rank 0 prints and writes; the other ranks
    still run every case (they are the MPI peers the native calls need) but
    stay silent, so 32 processes produce one report, not 32.
    """
    mpi_size, _ = mpi_info()
    owner = ledger_owner()
    grid_size = grid_product(config.lattice_grid)
    if owner and grid_size != mpi_size and any(
            case.backend in ("kupl", "cuda") for case in test.cases):
        print(f"note: lattice_grid {list(config.lattice_grid)} "
              f"(product {grid_size}) does not match MPI size {mpi_size}; "
              "native cases will SKIP with mpi_grid", file=sys.stderr)

    to_stdout = args.json == JSON_STDOUT
    stream = sys.stderr if to_stdout else sys.stdout

    def on_result(result: Result) -> None:
        if not owner:
            return
        print(_format_result(result), file=stream, flush=True)
        if args.verbose:
            _print_detail(result)

    report = test.run(
        repeat=config.repeat,
        warmup=config.warmup,
        seed=config.seed,
        timeout=config.timeout,
        fail_fast=args.fail_fast,
        allow_all_skipped=args.allow_all_skipped,
        on_result=on_result,
    )

    payload = report.as_dict()
    fields = _mode_fields(config, len(test.cases))
    options = dict(payload.get("options") or {})
    options["quick"] = config.quick
    options["backend_filter"] = list(config.backends)
    options["tier"] = config.tier
    options["argv"] = list(sys.argv[1:] if argv is None else argv)
    options["profile"] = list(config.profiles)
    options.update(fields)
    if config.profiles:
        options["profile_resolved"] = json_safe(
            _resolved_profile(config.profiles))
    payload["options"] = options
    # Top-level mirrors: "which mode / how many rows / which paths" must be
    # readable without unwrapping `options`.
    payload.update(fields)

    _emit_payload(payload, args, to_stdout=to_stdout)
    if owner:
        _print_summary(report, config, stream=sys.stderr if to_stdout else None)
    return report.exit_code()


def _sweep_options(config: _Config, test: Test, plan: Sequence[_SweepRun],
                   args: argparse.Namespace, argv: Sequence[str] | None,
                   mpi_size: int) -> dict[str, Any]:
    """``TestReport.options`` for a sweep: base config plus sweep metadata."""
    options = dict(test.config())
    cases = [case for run in plan for case in run.test.cases]
    options.update({
        "repeat": config.repeat,
        "warmup": config.warmup,
        "timeout": config.timeout,
        "fail_fast": bool(args.fail_fast),
        "allow_all_skipped": bool(args.allow_all_skipped),
        "mpi_grid_matches": grid_product(config.lattice_grid) == mpi_size,
        "backends": sorted({case.backend for case in cases}),
        "tags": sorted({case.tag for case in cases}),
        "quick": config.quick,
        "backend_filter": list(config.backends),
        "tier": config.tier,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "profile": list(config.profiles),
    })
    options.update(_mode_fields(config, len(cases)))
    if config.profiles:
        options["profile_resolved"] = json_safe(
            _resolved_profile(config.profiles))
    return options


def _sweep_document(config: _Config, plan: Sequence[_SweepRun],
                    runs: Sequence[dict[str, Any]],
                    skipped: Sequence[dict[str, Any]],
                    stopped: bool) -> dict[str, Any]:
    """The ``sweep`` header of the aggregate report (one row source of truth)."""
    axes = matrix.parse_spec(config.sweep)
    executed = len(runs)
    return {
        "spec": matrix.format_spec(axes),
        "order": list(axes),
        "axes": {name: json_safe(list(values))
                 for name, values in axes.items()},
        "limit": matrix.MAX_CASES,
        "planned": len(plan),
        "executed": executed,
        "skipped_combinations": list(skipped),
        "fail_fast_stopped": stopped,
        "not_started": len(plan) - executed - len(skipped),
        "runs": list(runs),
    }


def _run_sweep(config: _Config, test: Test, plan: Sequence[_SweepRun],
               args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Run every expanded row sequentially and emit one aggregate report.

    Each combination runs through :meth:`Test.run`, so the verdicts, timing,
    ``--fail-fast`` and ``--allow-all-skipped`` semantics are the single-run
    ones - a sweep is a loop over configurations, not a second runner.  A
    combination that selects no case is reported as a skip with its reason
    instead of being dropped (no fake rows, no fabricated PASS).
    """
    mpi_size, mpi_rank = mpi_info()
    owner = ledger_owner()
    grid_size = grid_product(config.lattice_grid)
    if owner and grid_size != mpi_size and any(
            case.backend in ("kupl", "cuda")
            for run in plan for case in run.test.cases):
        print(f"note: lattice_grid {list(config.lattice_grid)} "
              f"(product {grid_size}) does not match MPI size {mpi_size}; "
              "native cases will SKIP with mpi_grid", file=sys.stderr)

    to_stdout = args.json == JSON_STDOUT
    stream = sys.stderr if to_stdout else sys.stdout
    started = time.perf_counter()
    results: list[Result] = []
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stopped = False

    for run in plan:
        if not run.test.cases:
            reason = (f"sweep configuration "
                      f"{matrix.format_spec(run.combination)} selects no case")
            if owner:
                print(f"{run.label} SKIP {reason}", file=stream, flush=True)
            skipped.append({
                "index": run.index,
                "configuration": json_safe(run.combination),
                "status": "SKIP",
                "reason": reason,
            })
            continue
        if owner:
            print(f"{run.label} {matrix.format_spec(run.combination)} "
                  f"({len(run.test.cases)} case(s))", file=stream, flush=True)

        def on_result(result: Result, label: str = run.label) -> None:
            if not owner:
                return
            print(f"{label} {_format_result(result)}", file=stream, flush=True)
            if args.verbose:
                _print_detail(result)

        row = run.test.run(
            repeat=config.repeat,
            warmup=config.warmup,
            seed=None,
            timeout=config.timeout,
            fail_fast=args.fail_fast,
            allow_all_skipped=args.allow_all_skipped,
            on_result=on_result,
        )
        results.extend(row.cases)
        runs.append({
            "index": run.index,
            "configuration": json_safe(run.combination),
            "select": json_safe(run.kwargs),
            "status": row.status,
            "counts": json_safe(row.counts),
            "exit_code": row.exit_code(),
            "duration_seconds": round(float(row.duration_seconds), 6),
            "cases": [result.tag for result in row.cases],
        })
        if args.fail_fast and row.exit_code() != policy.EXIT_CODES["pass"]:
            stopped = True
            remaining = len(plan) - run.index
            if owner:
                print(f"{run.label} stop: --fail-fast after {row.status}; "
                      f"{remaining} combination(s) not started",
                      file=sys.stderr)
            break

    duration = time.perf_counter() - started
    counts, status = summarize(results)
    all_skipped = bool(results) and all(
        result.status == "SKIP" for result in results)
    options = _sweep_options(config, test, plan, args, argv, mpi_size)
    document = _sweep_document(config, plan, runs, skipped, stopped)
    options["sweep"] = {
        "spec": document["spec"],
        "planned": document["planned"],
        "executed": document["executed"],
        "fail_fast_stopped": stopped,
    }
    report = TestReport(
        status=status,
        lattice=config.lattice,
        dtype=config.dtype,
        seed=config.seed,
        cases=tuple(results),
        counts=counts,
        duration_seconds=duration,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank,
        env=probe_env(),
        options=options,
        allow_all_skipped=args.allow_all_skipped,
    )
    payload = report.as_dict()
    payload["sweep"] = document
    payload.update(_mode_fields(config, len(results)))

    _emit_payload(payload, args, to_stdout=to_stdout)
    if not owner:
        # A replica rank prints nothing: the sweep report and the summary
        # belong to rank 0 (see pyqcu.test.common.ledger_owner).
        return report.exit_code()

    print(f"summary: {len(results)} case run(s), "
          f"{counts['PASS']} pass, {counts['SKIP']} skip, "
          f"{counts['FAIL']} fail, {counts['ERROR']} error, "
          f"{counts['TIMEOUT']} timeout, {duration:.2f}s "
          f"({report.dtype}, {_lattice_text(report.lattice)})", file=stream)
    print(f"sweep: {document['planned']} combination(s), "
          f"{document['executed']} executed, "
          f"{len(skipped)} without a case, "
          f"{document['not_started']} not started "
          f"(spec {document['spec']}, limit {document['limit']})",
          file=stream)
    print(f"overall: {report.status}", file=stream)
    if config.quick and config.repeat <= 1:
        print("note: --quick makes no performance claim "
              "(single sample, no median)", file=sys.stderr)
    if report.status == "SKIP" and not args.allow_all_skipped:
        print("pyqcu.test: no case ran successfully (every selected case "
              "skipped or no combination selected one); pass "
              "--allow-all-skipped to exit 0", file=sys.stderr)
    return report.exit_code()


def _mode_executor(config: _Config,
                   test: Test) -> Callable[[Any], Result]:
    """Execute one row's case through the ordinary :meth:`Test.run` path.

    A full mode orchestrates; it is not a second runner.  The repeat/warmup/
    timeout/seed and the PASS/FAIL judgement stay the ones of a plain run, so a
    row's verdict means exactly what the equivalent single-case run means.
    """
    def execute(case: Any) -> Result:
        single = replace(test, cases=(case,))
        report = single.run(
            repeat=config.repeat,
            warmup=config.warmup,
            seed=config.seed,
            timeout=config.timeout,
            fail_fast=False,
            allow_all_skipped=True,
        )
        if not report.cases:
            raise RuntimeError(f"case {getattr(case, 'tag', case)!r} produced "
                               "no result")
        return report.cases[0]

    return execute


def _format_mode_entry(entry: dict[str, Any]) -> str:
    """Stable one-line rendering of a full-mode entry (row or input)."""
    status = str(entry.get("status") or "PENDING")
    detail = _one_line(entry.get("reason") or "")
    if not detail:
        seconds = entry.get("seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            detail = f"{float(seconds):.3f}s"
    if "run_id" in entry:
        return (f"{status:<7} {str(entry.get('run_id')):<24} "
                f"{str(entry.get('tag') or '-'):<24} {detail}")
    role = str(entry.get("role") or "input")
    path = str(entry.get("path") or "-")
    if not detail and entry.get("bytes") is not None:
        detail = f"{entry['bytes']} bytes"
    return f"{status:<7} {role:<8} {_one_line(path, 56):<56} {detail}"


def _export_reference_ledger(config: _Config, test: Test,
                             payload: dict[str, Any]) -> str | None:
    """Write the reference CSV/workbook when ``--ledger-export`` asks for it.

    Delegates to :func:`pyqcu.test.ledger_export.export_from_payload`, which is
    inert unless the flag or ``PYQCU_TEST_LEDGER_EXPORT`` selects a format, and
    which writes into ``<run root>/ledger/`` - never over ``reports/rows.csv``
    (the 30-column :data:`pyqcu.test.common.ROW_FIELDS` table keeps its
    semantics).  The outcome is recorded in the document under
    ``ledger_export`` so a ``--json`` report states what was written; a failure
    is returned to the caller and turned into exit code 2 after the normal
    summary instead of being swallowed or raised as a traceback.
    """
    result = ledger_export.export_from_payload(
        payload, run_options=test.config(),
        selection=config.ledger_export, stream=sys.stderr)
    if result is None:
        return None
    payload["ledger_export"] = result.as_dict()
    return result.error


def _run_mode(config: _Config, test: Test, plan: Sequence[_SweepRun],
              args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Execute one full mode, emit its document and return its exit code.

    One logical row over N processes: every rank runs the mode and reports
    through the per-row allgather, and only rank 0 prints, writes the ledger and
    emits the report payload (:func:`pyqcu.test.common.ledger_owner`).  The exit
    code is the gathered verdict, so all ranks agree on it.
    """
    owner = ledger_owner()
    rows = _mode_rows(config, test, plan)
    to_stdout = args.json == JSON_STDOUT
    stream = sys.stderr if to_stdout else sys.stdout
    exports = _report_exports(args, include_csv=False)
    _validate_export_targets(args, json_to_stdout=to_stdout, exports=exports)
    if owner and config.mode in ("solve", "run") and not config.quick:
        # Explicit, never silent: this is the tier-1 shape of the run.
        print(f"note: --mode {config.mode} is an explicit tier-{config.full.tier} "
              f"run at {_lattice_text(config.lattice)} {config.dtype} "
              f"(ranks={config.full.ranks}, shm={config.full.shm}); pass "
              "--quick for the tier-0 8x8x8x16 gate", file=sys.stderr)

    def on_entry(entry: dict[str, Any]) -> None:
        if not owner:
            return
        print(_format_mode_entry(entry), file=stream, flush=True)
        if args.verbose:
            print(f"  entry: {json.dumps(json_safe(entry), sort_keys=True)}",
                  file=sys.stderr)

    payload, code = run_mode(
        config.mode, test, config.full, rows=rows,
        execute=_mode_executor(config, test), on_entry=on_entry,
        argv=list(sys.argv[1:] if argv is None else argv))
    payload["quick"] = config.quick
    payload["mpi"] = _rank_fields()
    export_error = (_export_reference_ledger(config, test, payload)
                    if owner else None)
    _emit_payload(payload, args, to_stdout=to_stdout, exports=exports)
    if not owner:
        # A replica rank writes nothing, so it must not raise on an export the
        # owner already reported either: rank 0's exit code is the run's.
        return code

    counts = payload.get("counts") or {}
    rows_summary = payload.get("rows") or {}
    print(f"summary: mode={payload.get('mode')} "
          f"rows={rows_summary.get('count', 0)} "
          f"{counts.get('PASS', 0)} pass, {counts.get('SKIP', 0)} skip, "
          f"{counts.get('FAIL', 0)} fail, {counts.get('ERROR', 0)} error, "
          f"{counts.get('TIMEOUT', 0)} timeout, "
          f"{float(payload.get('duration_seconds') or 0.0):.2f}s",
          file=stream)
    problems = payload.get("problems") or []
    if problems:
        print(f"problems: {len(problems)}", file=stream)
        for problem in problems[:10]:
            print(f"  - {_one_line(problem, 200)}", file=stream)
    print(f"overall: {payload.get('status')}", file=stream)
    if config.quick and config.repeat <= 1:
        print("note: --quick makes no performance claim "
              "(single sample, no median)", file=sys.stderr)
    if export_error:
        raise UsageError(f"--ledger-export: {export_error}")
    return code


def _run(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Execute the selected cases or one full mode; return the exit code."""
    config = _resolve_config(args)
    test = _build_test(config)
    plan = _sweep_plan(config, test) if config.sweep else ()
    _print_rank_note()
    if args.list or config.mode == "list":
        if _report_exports(args):
            raise UsageError(
                "--list does not execute cases; --junit/--csv/--markdown need "
                "a run (drop --list or the export flag)")
        if config.ledger_export is not None:
            raise UsageError(
                "--list does not execute a full mode; --ledger-export needs "
                "--mode solve|run|merge (drop --list or --ledger-export)")
        return _emit_listing(config, test, args, plan)
    if config.mode is not None:
        return _run_mode(config, test, plan, args, argv)
    _validate_export_targets(args, json_to_stdout=args.json == JSON_STDOUT)
    if plan:
        return _run_sweep(config, test, plan, args, argv)
    if not test.cases:
        known = ", ".join(case.tag for case in Test.default().cases)
        raise UsageError(
            "no case selected: check --backend/--only/--tier/--solver/"
            f"--dslash against --list (known tags: {known})")
    return _run_single(config, test, args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
