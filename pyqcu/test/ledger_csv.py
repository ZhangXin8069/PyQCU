"""Reference-compatible ledger CSV for the formal :mod:`pyqcu.test` port.

The informal KUPL-Kunpeng driver
(``pyqcu/testing/kupl/test_kunpeng_all.py``) owns a *published* 100-column CSV
contract: ``CSV_COLUMNS`` plus ``append_csv``/``merge_rows``/``upgrade_csv_schema``.
Downstream spreadsheets, ``awk`` pipelines and the 2026-11-15 workbook
(``kupl_kunpeng_performance_20261115.xlsx``) read that table by column *name and
position*, so replacing the driver means reproducing the table, not inventing a
second one.

This module is that replacement: it renders :mod:`pyqcu.test` full-mode rows
(``rows/<run_id>.json`` / ``payload["rows"]["items"]``) and their prepare/prewarm
input entries into the reference's column order, with the reference's own cell
serialisation, line endings and header-upgrade rule.

What is reproduced exactly
--------------------------
* :data:`REFERENCE_CSV_COLUMNS` - the same 100 names in the same order
  (a frozen copy, not an import of the informal tree: ``pyqcu.test`` must not
  import ``pyqcu.testing``).
* **Cell serialisation**: the driver writes ``row.get(column)`` straight into
  :class:`csv.DictWriter`, so ``None`` becomes an empty field, ``True`` becomes
  ``True``, ``0.0`` becomes ``0.0`` and a list becomes its Python ``repr``
  (``[24, 24, 24, 64]``).  :func:`csv_cell` documents the same rule and
  :func:`reference_csv_text` uses a real ``csv.writer`` so quoting is identical.
* **Line endings**: CRLF, because the driver relies on ``csv.writer``'s default
  ``lineterminator`` while opening the file with ``newline=""``.  A reference
  CSV is *not* LF-per-record.
* **Header upgrade**: an existing table with fewer columns is rewritten with the
  reference header plus any legacy-only column, preserving every value
  (:func:`upgrade_columns`, the counterpart of ``upgrade_csv_schema``).
* **Row order**: sorted by ``run_id`` (:func:`ledger_records`), which is what
  ``merge_rows`` does, so two runs over the same rows produce the same file.

What is *not* reproduced (and is never faked)
---------------------------------------------
The reference table mixes two kinds of information:

1. facts about a case run that :mod:`pyqcu.test` also records (case identity,
   dtype, lattice, lattice grid, mg grid/levels/restart/dof, tolerance,
   iterations, residual and its provenance, the convergence flag, the MG
   verification status, cache load/build time, timings, report/solution paths);
2. facts only the HPCKit/mpirun driver can observe (per-rank OPT_PROFILE
   counters and phase timers, ``node``/``freq_khz``/``governor``,
   ``returncode``/``out_path``/``err_path``, the cache global/ranked split, the
   reference driver's cold/warm read modes, HDF5 MPIO, the review flag).

Many of the "first kind" values only exist for *native* rows, and they live in
``row["metrics"]`` rather than at the top level: :func:`pyqcu.test.common._apply_driver`
merges every native observation there, so ``metrics['atol']``, ``['dof']``,
``['nv_iters']``, ``['coarse_max_iter']``, ``['coarse_tol_factor']``,
``['deflate']``, ``['verify_status']``, ``['converged']``,
``['residual_source']``, ``['reference_seconds']``, ``['cache_seconds']`` and
``['case_seconds']`` are real measurements, not applied defaults.  They are read
through :func:`_metric`; a tier-0 row that recorded none of them stays empty.

The same rule applies to the *input and cache evidence* a native row carries:
``metrics['gauge_path']``/``['gauge_sha256']``/``['gauge_file_dtype']`` (and the
``source_*`` pair), ``metrics['cache_path']``, ``['cache_file_sha256']``,
``['cache_identity_sha256']``, ``['cache_manifest_path']`` and
``['cache_setup_seconds']`` are what the runner actually consumed or loaded, so
they are written in preference to the run-level ``--gauge-file``/``--cache-dir``
context (which only names the *request*).  Without that preference a native MG
row would export an empty ``cache_path`` next to a non-empty cache timing.

Every column of the second kind is listed in :data:`COLUMN_RULES` with
``source="missing"`` and a reason; it is written as an *empty field*, exactly
like a value the driver did not observe.  Nothing is defaulted from the
reference snapshot (no invented ``-0.277`` mass, no copied ``1e-6`` atol, no
"the export machine's" hostname) and no column is filled from a different
quantity (the compare-gate ``--atol`` is *not* written into the solver-tolerance
``atol`` column).

Programmatic use::

    from pyqcu.test import ledger_csv

    records = ledger_csv.ledger_records(payload)          # payload = --mode solve --json
    ledger_csv.write_ledger_csv("rows.csv", records)      # atomic, CRLF, 100 columns
    ledger_csv.missing_columns()                          # documented gaps

Import rules: standard library only (``csv``, ``io``, ``json``, ``hashlib``,
``os``), plus :mod:`pyqcu.test.policy` and the pure helpers
``row_id``/``json_safe`` of :mod:`pyqcu.test.common`.  No backend import, no
``pyqcu.testing``, no HDF5: reading a cache manifest is the caller's job and is
passed in through :class:`LedgerContext`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import policy
from .common import json_safe

__all__ = [
    "COLUMN_RULES",
    "DERIVED_COLUMNS",
    "LINE_TERMINATOR",
    "MISSING_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "REFERENCE_CSV_COLUMNS",
    "REFERENCE_PHASES",
    "REFERENCE_SOLVERS",
    "SOLVER_COLUMN_BY_VARIANT",
    "ColumnRule",
    "LedgerContext",
    "csv_cell",
    "input_records",
    "ledger_records",
    "missing_columns",
    "performance_record",
    "reference_csv_text",
    "reference_record",
    "reference_records",
    "upgrade_columns",
    "write_ledger_csv",
]

# ---------------------------------------------------------------------------
# the reference header
# ---------------------------------------------------------------------------
#: The informal driver's ``CSV_COLUMNS``, verbatim and in order.  This is a
#: frozen copy on purpose: importing ``pyqcu.testing`` is forbidden for this
#: package, and the list is a published contract that must not drift silently.
#: ``selfcheck``/the tests assert that it still has 100 unique names.
REFERENCE_CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "phase", "solver", "dtype", "atol",
    "lattice", "lat_x", "lat_y", "lat_z", "lat_t",
    "mass", "csw", "kappa",
    "ranks", "threads", "rank_x_thread", "grid", "grid_mode",
    "shm", "source_kind", "source_path", "source_file_dtype", "source_sha256",
    "seed",
    "gauge_kind", "gauge_path", "gauge_file_dtype", "gauge_sha256",
    "cache_kind", "cache_path", "cache_sha256", "cache_identity_sha256",
    "cache_manifest_path", "cache_manifest_sha256",
    "mg_levels", "mg_restart", "mg_grid", "mg_dof", "nv_iters",
    "coarse_max_iter", "coarse_tol_factor", "deflate", "max_iter",
    "verify_status", "iterations",
    "wall_seconds", "builder_seconds", "build_mg_cache_seconds",
    "final_true_residual", "relative_full_residual",
    "distributed_relative_residual", "residual_source", "converged", "status",
    "node", "freq_khz", "freq_max_khz", "governor",
    "profile_rank",
    "dslash_seconds", "dslash_calls", "dslash_bytes_mb",
    "dslash_gbps_rank", "dslash_gbps_total",
    "clover_seconds", "clover_calls", "clover_bytes_mb",
    "clover_gbps_rank", "clover_gbps_total",
    "reduce_seconds", "reduce_calls",
    "halo_pack_seconds", "halo_sync_seconds", "halo_copy_seconds",
    "phase_prepare_seconds", "phase_solve_seconds", "phase_reconstruct_seconds",
    "input_prebind_seconds", "gauge_read_seconds", "source_read_seconds",
    "gauge_generate_seconds", "source_generate_seconds",
    "gauge_prewarm_seconds", "source_prewarm_seconds",
    "gauge_read_mode", "source_read_mode",
    "hdf5_mpio_read", "residual_seconds",
    "report_path", "report_sha256", "solution_path", "solution_sha256",
    "out_path", "err_path", "returncode", "elapsed_seconds",
    "review", "timeout", "error", "notes",
)

#: The reference's ``phase`` vocabulary.  ``solve``/``prepare``/``prewarm`` are
#: the three the driver writes; ``validate`` appears only in the ``--mode
#: validate`` failure row.
REFERENCE_PHASES: tuple[str, ...] = ("solve", "prepare", "prewarm", "validate")

#: The reference's ``solver`` vocabulary for solve rows (plus the two synthetic
#: values of the prepare/prewarm rows).  ``cg`` and ``mg`` are *extensions*: the
#: formal port has cases the driver never ran, and renaming them onto
#: ``bistabcg``/``mg2`` would claim a measurement that was never taken.
REFERENCE_SOLVERS: tuple[str, ...] = (
    "bistabcg", "mg1", "mg2", "cache_build", "input_prewarm")

#: ``<backend>/<area>/<variant>`` variant -> the reference ``solver`` value.
#: ``mg`` is deliberately not ``mg1``/``mg2``: the formal port's MG cases are
#: variant checks (``legacy``/``strict``), not the driver's 1-/2-level runs.
SOLVER_COLUMN_BY_VARIANT: Mapping[str, str] = {
    "bistabcg": "bistabcg",
    "cg": "cg",
    "legacy": "mg",
    "strict": "mg",
    "multigrid": "mg",
}

#: ``solver`` value of the prepare/prewarm rows the reference writes.
PREPARE_SOLVER = "cache_build"
PREWARM_SOLVER = "input_prewarm"

#: The driver opens the CSV with ``newline=""`` and lets :mod:`csv` use its
#: default line terminator, so a reference table is CRLF - one CRLF per record.
LINE_TERMINATOR = "\r\n"

#: Number of columns of the reference header (a documented constant because the
#: reports quote it: "142 rows / 100 columns").
REFERENCE_COLUMN_COUNT = 100


# ---------------------------------------------------------------------------
# per-column provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnRule:
    """Where one reference column comes from, and what it means.

    ``source`` is one of

    ``row``      a field of a :mod:`pyqcu.test` ledger row (``field``);
    ``option``   a run-level ``FullModeOptions`` value (``field``);
    ``derived``  computed from the row/options/inputs by this module;
    ``missing``  not observable by the formal port - always an empty field,
                 with ``note`` saying why and what would be needed.

    ``note`` is machine-readable documentation: :func:`column_note` surfaces it
    and the tests assert that every ``missing`` column carries a non-empty one.
    """

    name: str
    source: str
    field: str = ""
    note: str = ""


#: Column rules in reference order.  Built from literal tuples (rather than
#: decorated per column) so the order can be diffed against
#: :data:`REFERENCE_CSV_COLUMNS` by eye and by the tests.
COLUMN_RULES: tuple[ColumnRule, ...] = (
    ColumnRule("run_id", "row", "run_id"),
    ColumnRule("phase", "derived",
               note="the full-mode mode that produced the row: solve/prepare/"
                    "prewarm/validate; 'list' has no verdict and is not "
                    "exported"),
    ColumnRule("solver", "derived",
               note="algorithm from the case tag variant (bistabcg/cg/mg). A "
                    "'<backend>/native/solve' row is the native Bistab-CG "
                    "runner (applyCloverBistabCgQcu) and '<backend>/native/mg' "
                    "the native Legacy-MG runner, which is reported as 'mg' - "
                    "an extension: the reference's mg1/mg2 name a 1-/2-level "
                    "*configuration* of its driver, not this runner"),
    ColumnRule("dtype", "row", "dtype"),
    ColumnRule("atol", "derived",
               note="row metrics['atol'] (native runner) or "
                    "metrics['solver_tolerance'] (torch solver rows) - the "
                    "tolerance the runner validated and passed to the solver. "
                    "Empty for rows whose backend recorded no tolerance. The "
                    "compare gate policy.RESIDUAL_GATES[dtype]['solve'] (the "
                    "torch rows' metrics['tolerance']) is a different quantity "
                    "and is never copied here"),
    ColumnRule("lattice", "row", "lattice"),
    ColumnRule("lat_x", "derived", note="lattice[0]"),
    ColumnRule("lat_y", "derived", note="lattice[1]"),
    ColumnRule("lat_z", "derived", note="lattice[2]"),
    ColumnRule("lat_t", "derived", note="lattice[3]"),
    ColumnRule("mass", "derived",
               note="row 'mass'; falls back to metrics['mass'], the effective "
                    "value the backend ran with, when the run left the mass to "
                    "the runner's default (the formal port has no CLI default)"),
    ColumnRule("csw", "row", "csw"),
    ColumnRule("kappa", "derived",
               note="row 'kappa'; falls back to metrics['kappa'], the hopping "
                    "parameter the backend derived from the effective mass"),
    ColumnRule("ranks", "derived",
               note="row 'ranks' (the requested rank count), falling back to "
                    "metrics['mpi_size'] - the size of the communicator the "
                    "native runner actually ran on (driver.run_row)"),
    ColumnRule("threads", "derived",
               note="row 'threads'; falls back to metrics['threads'], the "
                    "thread count the native runner applied (only torch rows "
                    "apply it otherwise, so a KUPL/CUDA tier-0 row records the "
                    "request, not the effect)"),
    ColumnRule("rank_x_thread", "derived", note="'<ranks>x<threads>'"),
    ColumnRule("grid", "derived",
               note="per-row metrics['lattice_grid'] - the grid the row "
                    "actually ran with - falling back to the run's "
                    "lattice_grid when the backend recorded none (a --mode "
                    "merge document carries no run options, so the per-row "
                    "value is what keeps a merged table complete)"),
    ColumnRule("grid_mode", "derived",
               note="'t-finest' when grid == (1,1,1,ranks), else 'balanced' "
                    "(the reference driver's own rule)"),
    ColumnRule("shm", "derived",
               note="row 'shm' as 0/1; falls back to metrics['shm'], the "
                    "shared-halo decision the native runner applied"),
    ColumnRule("source_kind", "derived",
               note="row 'source' (the requested generator), falling back to "
                    "metrics['source_kind'] - the kind the runner was actually "
                    "given. The reference vocabulary also has 'real1', which "
                    "the formal port never generates"),
    ColumnRule("source_path", "derived",
               note="row 'source_file' (--source-file), falling back to "
                    "metrics['source_path'] - the artifact the native runner "
                    "actually consumed (driver._validated_input)"),
    ColumnRule("source_file_dtype", "derived",
               note="metrics['source_file_dtype'] of a native row, falling back "
                    "to the io.describe() dtype of the prepare/prewarm entry"),
    ColumnRule("source_sha256", "derived",
               note="metrics['source_sha256'] (the streamed SHA256 the native "
                    "runner recorded for the bytes it read), falling back to "
                    "the prewarm hash of the input file"),
    ColumnRule("seed", "row", "seed"),
    ColumnRule("gauge_kind", "derived",
               note="row 'gauge' (the requested generator), falling back to "
                    "metrics['gauge_kind']. The reference vocabulary is "
                    "real/random and 'real' (an on-disk production gauge) does "
                    "not exist here"),
    ColumnRule("gauge_path", "derived",
               note="row 'gauge_file' (--gauge-file), falling back to "
                    "metrics['gauge_path'] - the artifact the native runner "
                    "actually consumed"),
    ColumnRule("gauge_file_dtype", "derived",
               note="metrics['gauge_file_dtype'] of a native row, falling back "
                    "to the io.describe() dtype of the prepare/prewarm entry"),
    ColumnRule("gauge_sha256", "derived",
               note="metrics['gauge_sha256'] (the streamed SHA256 the native "
                    "runner recorded), falling back to the prewarm hash"),
    ColumnRule("cache_kind", "missing",
               note="the formal port has one Legacy-MG cache kind and does not "
                    "classify it as global/ranked (that split belongs to the "
                    "KUPL driver's cache_key)"),
    ColumnRule("cache_path", "derived",
               note="metrics['cache_path'] of a native MG row (cache.CacheEntry."
                    "as_dict), falling back to --cache-dir when it names an "
                    "artifact file"),
    ColumnRule("cache_sha256", "derived",
               note="metrics['cache_file_sha256'] of a native MG row, falling "
                    "back to manifest 'file_sha256' when a cache manifest was "
                    "supplied"),
    ColumnRule("cache_identity_sha256", "derived",
               note="metrics['cache_identity_sha256'] of a native MG row, "
                    "falling back to manifest 'identity_sha256' when a cache "
                    "manifest was supplied"),
    ColumnRule("cache_manifest_path", "derived",
               note="metrics['cache_manifest_path'] of a native MG row, falling "
                    "back to the manifest path implied by --cache-dir"),
    ColumnRule("cache_manifest_sha256", "derived",
               note="hashing the manifest file (metrics['cache_manifest_path'] "
                    "or the --cache-dir manifest) is opt-in "
                    "(LedgerContext.hash_files) because it makes the exporter "
                    "read the disk"),
    ColumnRule("mg_levels", "derived",
               note="per-row metrics['mg_levels'] (the level count the runner "
                    "configured), falling back to the run's --mg-levels"),
    ColumnRule("mg_restart", "derived",
               note="the run's --restart, else per-row metrics['restart'] - but "
                    "only for a multigrid row: a CG/BiCGStab row records its "
                    "own solver restart period under that key, which is not "
                    "the MG restart the column names"),
    ColumnRule("mg_grid", "derived",
               note="per-row metrics['mg_grid'] (the block grid the runner "
                    "configured), falling back to the run's --mg-grid"),
    ColumnRule("mg_dof", "derived",
               note="row metrics['dof'] truncated to metrics['mg_levels'] "
                    "entries - the components each level in use actually "
                    "consumed (native uses option.dof[:mg_levels] for its cache "
                    "identity; the reference writes args.dof[:levels])"),
    ColumnRule("nv_iters", "derived",
               note="row metrics['nv_iters'] - the null-vector iteration count "
                    "the native MG runner used to build its coarse space"),
    ColumnRule("coarse_max_iter", "derived",
               note="row metrics['coarse_max_iter'] - the coarse-solver "
                    "iteration cap the native MG runner configured"),
    ColumnRule("coarse_tol_factor", "derived",
               note="row metrics['coarse_tol_factor'] - the coarse tolerance "
                    "factor the native MG runner configured"),
    ColumnRule("deflate", "derived",
               note="row metrics['deflate'] - the deflation flag the native MG "
                    "runner wrote into params[_MG_USE_DEFLATE_]"),
    ColumnRule("max_iter", "derived",
               note="per-row metrics['max_iter'] (the cap the runner used), "
                    "falling back to the run's --max-iter; empty when neither "
                    "was given (the runner default applies)"),
    ColumnRule("verify_status", "derived",
               note="row metrics['verify_status'] - the return value of the "
                    "native verifyCloverMultigridQcu call (--verify); empty "
                    "when the row did not request verification"),
    ColumnRule("iterations", "derived",
               note="row metrics['iterations'] (aliases iters/iteration)"),
    ColumnRule("wall_seconds", "row", "seconds",
               note="median of --repeat samples of the case (warmup discarded); "
                    "the reference measures one mpirun wall clock - same unit, "
                    "different protocol"),
    ColumnRule("builder_seconds", "derived",
               note="row metrics['cache_seconds'] - the native MG cache "
                    "load/build time the runner measured (the reference calls "
                    "the same quantity cache load/build time)"),
    ColumnRule("build_mg_cache_seconds", "derived",
               note="metrics['cache_setup_seconds'] of a native MG row (the "
                    "recorded build time of the artifact it loaded), falling "
                    "back to manifest 'setup_seconds' when a cache manifest was "
                    "supplied"),
    ColumnRule("final_true_residual", "derived",
               note="row metrics['relative_residual'] (aliases rel_res/"
                    "true_residual) - the independently reconstructed "
                    "||D x - b|| / ||b||"),
    ColumnRule("relative_full_residual", "derived",
               note="same value: the formal port reconstructs the residual on "
                    "the full operator in one process, so the reference's "
                    "full/distributed split is decided by construction"),
    ColumnRule("distributed_relative_residual", "missing",
               note="the formal port has no MPI-halo residual reconstruction, "
                    "so it must not claim a distributed number"),
    ColumnRule("residual_source", "derived",
               note="row metrics['residual_source'] - how the runner computed "
                    "the residual it reports (native._true_residual)"),
    ColumnRule("converged", "derived",
               note="row metrics['converged'] - the solver's own convergence "
                    "flag (relative_residual <= atol, and verify_status 0 for "
                    "MG); empty for rows whose backend reports no flag"),
    ColumnRule("status", "row", "status",
               note="PASS/FAIL/SKIP/TIMEOUT/ERROR; the reference vocabulary is "
                    "PASS/FAIL/SKIP/REUSE"),
    ColumnRule("node", "missing",
               note="the formal port records no host per row; using the export "
                    "machine's hostname would be a false locality claim"),
    ColumnRule("freq_khz", "missing",
               note="cpufreq sampling belongs to the KUPL driver"),
    ColumnRule("freq_max_khz", "missing", note="cpufreq sampling belongs to the KUPL driver"),
    ColumnRule("governor", "missing", note="cpufreq sampling belongs to the KUPL driver"),
    ColumnRule("profile_rank", "missing",
               note="the KUPL operator profiler is not part of the formal port"),
    ColumnRule("dslash_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("dslash_calls", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("dslash_bytes_mb", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("dslash_gbps_rank", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("dslash_gbps_total", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("clover_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("clover_calls", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("clover_bytes_mb", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("clover_gbps_rank", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("clover_gbps_total", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("reduce_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("reduce_calls", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("halo_pack_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("halo_sync_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("halo_copy_seconds", "missing", note="KUPL OPT_PROFILE counter"),
    ColumnRule("phase_prepare_seconds", "missing", note="KUPL OPT_PROFILE phase timer"),
    ColumnRule("phase_solve_seconds", "missing", note="KUPL OPT_PROFILE phase timer"),
    ColumnRule("phase_reconstruct_seconds", "missing", note="KUPL OPT_PROFILE phase timer"),
    ColumnRule("input_prebind_seconds", "missing",
               note="device pre-binding is a KUPL driver step"),
    ColumnRule("gauge_read_seconds", "missing",
               note="in-solver HDF5 read time is reported by the KUPL driver; "
                    "the formal port's prewarm seconds are a full-file "
                    "read+SHA256 pass and are written to gauge_prewarm_seconds"),
    ColumnRule("source_read_seconds", "missing",
               note="see gauge_read_seconds"),
    ColumnRule("gauge_generate_seconds", "missing",
               note="the formal port never generates gauge artifacts"),
    ColumnRule("source_generate_seconds", "missing",
               note="the formal port never generates source artifacts"),
    ColumnRule("gauge_prewarm_seconds", "derived",
               note="prepare/prewarm input entry 'seconds' for role=gauge (one "
                    "read pass that also hashes the file)"),
    ColumnRule("source_prewarm_seconds", "derived",
               note="prepare/prewarm input entry 'seconds' for role=source"),
    ColumnRule("gauge_read_mode", "missing",
               note="which HDF5 reader produced the rank block: 'mpio' = "
                    "collective parallel read (driver='mpio' on the world "
                    "communicator), 'slab' = independent per-rank hyperslab "
                    "read. A native row carries it as "
                    "metrics['gauge_read_mode']; the reference driver writes "
                    "cold/warm/pre-generated into the same column, and this "
                    "export leaves the column empty"),
    ColumnRule("source_read_mode", "missing",
               note="see gauge_read_mode"),
    ColumnRule("hdf5_mpio_read", "missing",
               note="a native row records the effective reader in "
                    "gauge_read_mode ('mpio'/'slab'), not a separate "
                    "h5py-MPI flag, so this boolean stays empty"),
    ColumnRule("residual_seconds", "derived",
               note="row metrics['reference_seconds'] - the time the runner "
                    "spent reconstructing the true residual it reports"),
    ColumnRule("report_path", "row", "report_path"),
    ColumnRule("report_sha256", "derived",
               note="opt-in (hash_files): SHA256 of reports/<run_id>.json"),
    ColumnRule("solution_path", "row", "solution_path"),
    ColumnRule("solution_sha256", "derived",
               note="opt-in (hash_files): SHA256 of the solution file"),
    ColumnRule("out_path", "missing",
               note="the formal port runs in-process and writes no per-row "
                    "stdout/stderr log"),
    ColumnRule("err_path", "missing", note="see out_path"),
    ColumnRule("returncode", "missing",
               note="there is no per-row subprocess; the full-mode exit code is "
                    "per invocation, not per row"),
    ColumnRule("elapsed_seconds", "derived",
               note="row metrics['elapsed_seconds'] (tier-0 executors) or "
                    "metrics['case_seconds'] (native runner) - the wall clock "
                    "of the whole row's work. The formal port starts no "
                    "subprocess, so this excludes the mpirun/import startup the "
                    "reference's value includes; wall_seconds is the timed "
                    "region alone"),
    ColumnRule("review", "missing",
               note="the review-threshold flag belongs to the KUPL driver"),
    ColumnRule("timeout", "derived",
               note="True when status == TIMEOUT (the formal port's per-case "
                    "timeout produces that status)"),
    ColumnRule("error", "derived",
               note="the row's reason when status == ERROR (a harness-level "
                    "failure); FAIL/SKIP/TIMEOUT reasons go to notes so the two "
                    "kinds stay distinguishable"),
    ColumnRule("notes", "derived",
               note="ordered list of 'key=value' provenance notes (see "
                    "reference_record): backend, tag, tier, device, filters, "
                    "failed gates, skip token, non-PASS reason"),
)

#: Columns the formal port cannot observe (always empty).
MISSING_COLUMNS: tuple[str, ...] = tuple(
    rule.name for rule in COLUMN_RULES if rule.source == "missing")

#: Columns this module computes from rows/options/inputs.
DERIVED_COLUMNS: tuple[str, ...] = tuple(
    rule.name for rule in COLUMN_RULES if rule.source == "derived")

_RULES_BY_NAME: dict[str, ColumnRule] = {rule.name: rule for rule in COLUMN_RULES}


def column_note(name: str) -> str:
    """The documented provenance of one reference column (empty if unknown)."""
    rule = _RULES_BY_NAME.get(str(name))
    return "" if rule is None else rule.note


def missing_columns() -> tuple[str, ...]:
    """Reference columns with no :mod:`pyqcu.test` counterpart, in order."""
    return MISSING_COLUMNS


def derived_columns() -> tuple[str, ...]:
    """Reference columns computed by :func:`reference_record`, in order."""
    return DERIVED_COLUMNS


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerContext:
    """Run-level information a single row cannot carry.

    ``options`` is the payload's ``options`` block (``FullModeOptions.as_dict()``,
    i.e. mode/labels/ranks/shm/inputs/paths).  ``run_options`` is the *run*
    configuration - ``Test.config()`` at the integration point - and it is what
    carries ``lattice``, ``lattice_grid``, ``mg_grid``, ``dtype``, ``seed``,
    ``max_iter``, ``restart`` and ``mg_levels``: a full-mode document does not
    repeat them, and guessing them from a row would silently drop the grid.

    ``inputs`` are the prepare/prewarm input entries of the same invocation
    (used for file dtype/SHA256/prewarm seconds).  ``cache_manifest`` is an
    already-read Legacy-MG cache manifest (``pyqcu.test.cache.read_manifest``),
    passed in by the caller so this module never touches HDF5.  ``hash_files``
    turns on the expensive column derivations
    (``report_sha256``/``solution_sha256``/``cache_manifest_sha256``).
    """

    mode: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    run_options: Mapping[str, Any] = field(default_factory=dict)
    inputs: tuple[Mapping[str, Any], ...] = ()
    cache_manifest: Mapping[str, Any] | None = None
    hash_files: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *,
                     run_options: Mapping[str, Any] | None = None,
                     cache_manifest: Mapping[str, Any] | None = None,
                     hash_files: bool = False) -> "LedgerContext":
        """Build the context from a full-mode document (``--json`` payload).

        ``run_options`` should be ``Test.config()``: the payload's own
        ``options`` block describes the *mode* (paths, ranks, shm, inputs), not
        the case configuration, and the reference columns need both.
        """
        inputs = payload.get("inputs")
        options = payload.get("options")
        return cls(
            mode=(None if payload.get("mode") is None
                  else str(payload.get("mode"))),
            options=options if isinstance(options, Mapping) else {},
            run_options=run_options if isinstance(run_options, Mapping) else {},
            inputs=tuple(entry for entry in (inputs or ())
                         if isinstance(entry, Mapping)),
            cache_manifest=cache_manifest,
            hash_files=bool(hash_files),
        )

    def option(self, name: str) -> Any:
        """Run-level value of ``name`` (run configuration wins over the mode)."""
        if name in self.run_options and self.run_options[name] is not None:
            return self.run_options[name]
        return self.options.get(name)

    def input_entry(self, *roles: str) -> Mapping[str, Any] | None:
        """First input entry whose ``role`` is in ``roles`` (or ``None``)."""
        wanted = {str(role) for role in roles}
        for entry in self.inputs:
            if str(entry.get("role")) in wanted:
                return entry
        return None

    def input_dtype(self, *roles: str) -> Any:
        """``io.dtype`` of an input entry (``None`` when it is not described)."""
        entry = self.input_entry(*roles)
        io = entry.get("io") if isinstance(entry, Mapping) else None
        return io.get("dtype") if isinstance(io, Mapping) else None

    def input_sha256(self, *roles: str) -> Any:
        """``sha256`` of an input entry (only ``prewarm`` computes it)."""
        entry = self.input_entry(*roles)
        return None if entry is None else entry.get("sha256")


def phase_of(mode: Any) -> Any:
    """Reference ``phase`` of one full-mode invocation (``None`` for a plan).

    ``solve``/``run``/``merge`` produce solve rows (the modes differ in *how*
    the rows were produced, not in what a row is); ``prepare`` and ``prewarm``
    produce input rows; ``validate`` reports on rows it did not execute;
    ``list`` is a plan, and a plan has no phase in the reference vocabulary.
    """
    text = "" if mode is None else str(mode)
    if text in ("", "solve", "run", "merge"):
        return "solve"
    if text in ("prepare", "prewarm", "validate"):
        return text
    return None


# ---------------------------------------------------------------------------
# cell serialisation
# ---------------------------------------------------------------------------
def csv_cell(value: Any) -> str:
    """Render one cell exactly like the reference writer.

    ``csv.DictWriter`` stringifies whatever ``row.get(column)`` returned, so the
    reference table contains ``""`` for ``None``, ``True``/``False`` for
    booleans, Python ``repr`` for lists (``[24, 24, 24, 64]``) and ``repr`` for
    floats.  Keeping the same rule is what makes a column-by-column diff of two
    tables meaningful.
    """
    if value is None:
        return ""
    return str(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


# ---------------------------------------------------------------------------
# per-column values
# ---------------------------------------------------------------------------
def _variant(tag: Any) -> str:
    """Third segment of a ``backend/area/variant`` tag (``""`` if absent)."""
    parts = str(tag or "").split("/")
    return parts[2].strip() if len(parts) > 2 else ""


def _solver_of(row: Mapping[str, Any]) -> Any:
    """Reference ``solver`` value of one row (``None`` when not derivable).

    Only ``*/solve/*``, ``*/mg/*`` and the native ``<backend>/native/*`` rows
    name an algorithm; a dslash, smear, gauge, lattice or ABI case has no
    solver, and writing one of the reference's three solver words there would
    invent a solve that never happened.

    The native rows are decided by their *kind*, not by a tag variant: the
    ``native/solve`` runner calls ``applyCloverBistabCgQcu`` and the
    ``native/mg`` runner calls ``applyCloverMultigridQcu`` (Legacy MG), so
    ``bistabcg`` and ``mg`` are facts about the runner, not a guess.
    """
    parts = str(row.get("tag") or "").split("/")
    if len(parts) < 2:
        return None
    area = parts[1].strip()
    variant = parts[2].strip() if len(parts) > 2 else ""
    if area == "native":
        if variant == "solve":
            return "bistabcg"
        if variant == "mg":
            return SOLVER_COLUMN_BY_VARIANT.get(variant, "mg")
        return None
    if area == "solve":
        mapped = SOLVER_COLUMN_BY_VARIANT.get(variant, variant)
        return mapped or None
    if area in ("mg", "multigrid"):
        return SOLVER_COLUMN_BY_VARIANT.get(variant, "mg")
    return None


def _is_multigrid_row(row: Mapping[str, Any]) -> bool:
    """True when the row's case is a multigrid one (``*/mg/*``, native MG)."""
    parts = str(row.get("tag") or "").split("/")
    area = parts[1].strip() if len(parts) > 1 else ""
    variant = parts[2].strip() if len(parts) > 2 else ""
    if area in ("mg", "multigrid"):
        return True
    if area == "native" and variant == "mg":
        return True
    return area == "solve" and variant in ("mg1", "mg2")


def _metric(row: Mapping[str, Any], *names: str) -> Any:
    """First present metric among ``names`` (flat or under ``metrics``)."""
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _lattice_extent(row: Mapping[str, Any], index: int) -> Any:
    lattice = row.get("lattice")
    if isinstance(lattice, (list, tuple)) and len(lattice) > index:
        return int(lattice[index])
    return None


def _grid(row: Mapping[str, Any], context: LedgerContext) -> Any:
    """The lattice grid the row ran with (per row first, run options second).

    A native row records ``metrics['lattice_grid']`` - the grid its runner was
    configured with - and a tier-0 row records it in the run configuration.  A
    ``--mode merge`` document carries no run options at all, so the per-row
    value is what keeps a merged table from losing the grid column.
    """
    grid = _metric(row, "lattice_grid")
    if grid is None:
        grid = context.option("lattice_grid")
    return list(grid) if _is_sequence(grid) else grid


def _grid_mode(row: Mapping[str, Any], context: LedgerContext) -> Any:
    grid = _grid(row, context)
    ranks = row.get("ranks")
    if not _is_sequence(grid) or ranks is None:
        return None
    return "t-finest" if tuple(int(v) for v in grid) == (1, 1, 1, int(ranks)) \
        else "balanced"


def _rank_x_thread(row: Mapping[str, Any]) -> Any:
    ranks = row.get("ranks")
    threads = _metric(row, "threads")
    if ranks is None or threads is None:
        return None
    return f"{int(ranks)}x{int(threads)}"


def _shm(row: Mapping[str, Any]) -> Any:
    """The shared-halo decision as the reference's ``0``/``1`` (or ``None``)."""
    value = _metric(row, "shm")
    return None if value is None else int(bool(value))


def _first(*values: Any) -> Any:
    """First non-``None`` candidate (``None`` when every one is unset)."""
    for value in values:
        if value is not None:
            return value
    return None


def _mg_dof(row: Mapping[str, Any], levels: Any) -> Any:
    """The MG dof vector of one row, cut to the levels actually in use.

    ``native._common_observations`` records the whole ``option.dof`` vector while
    the runner (and its cache identity) only consumes ``dof[:mg_levels]``; the
    reference column is ``args.dof[:levels]``.  Reporting the prefix is the same
    fact the runner used, not a truncation of a measurement.
    """
    dof = _metric(row, "dof")
    if not _is_sequence(dof):
        return dof
    values = [int(value) for value in dof]
    if isinstance(levels, int) and not isinstance(levels, bool) \
            and 0 < levels <= len(values):
        return values[:levels]
    return values


def _notes(row: Mapping[str, Any]) -> list[str]:
    """Ordered provenance notes of one row (deterministic, no timestamps)."""
    notes: list[str] = []
    if row.get("backend"):
        notes.append(f"backend={row['backend']}")
    if row.get("tag"):
        notes.append(f"tag={row['tag']}")
    if row.get("tier") is not None:
        notes.append(f"tier={row['tier']}")
    if row.get("device"):
        notes.append(f"device={row['device']}")
    if str(row.get("solver") or "") not in ("", "all"):
        notes.append(f"solver_filter={row['solver']}")
    if str(row.get("dslash") or "") not in ("", "all"):
        notes.append(f"dslash_filter={row['dslash']}")
    if row.get("run_label"):
        notes.append(f"run_label={row['run_label']}")
    if row.get("resumed"):
        notes.append("resumed=true")
    if row.get("mpi_size") is not None:
        notes.append(f"mpi_size={row['mpi_size']}")
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    if metrics.get("cache_reused") is not None:
        notes.append("cache_reused=true" if metrics["cache_reused"]
                     else "cache_reused=false")
    if metrics.get("asset_source"):
        notes.append(f"asset_source={metrics['asset_source']}")
    for gate in row.get("failed") or ():
        notes.append(f"failed_gate={gate}")
    if row.get("skip_token"):
        notes.append(f"skip_token={row['skip_token']}")
    reason = str(row.get("reason") or "").strip()
    if reason and row.get("status") != "PASS":
        notes.append(f"reason={reason}")
    return notes


def _opt(context: LedgerContext, name: str) -> Any:
    value = context.option(name)
    return None if value is None else value


def _manifest(context: LedgerContext, key: str) -> Any:
    manifest = context.cache_manifest
    if not isinstance(manifest, Mapping):
        return None
    return manifest.get(key)


def _cache_path(context: LedgerContext) -> Any:
    cache = context.option("cache_dir")
    if cache is None:
        return None
    path = Path(str(cache))
    return str(path) if path.is_file() else None


def _cache_manifest_path(context: LedgerContext) -> Any:
    """Manifest of a real cache *artifact* (``--cache-dir`` naming a file).

    A cache *directory* is not an artifact: ``<dir>/index.json`` is the port's
    own probe index, and reporting it as a cache manifest would claim that an
    artifact exists.  The prepare input rows carry it under their own path
    instead.
    """
    cache = context.option("cache_dir")
    if cache is None:
        return None
    path = Path(str(cache))
    if not path.is_file():
        return None
    return str(path.with_suffix(".json"))


def _file_digest(context: LedgerContext, value: Any) -> Any:
    """SHA256 of ``value`` when opt-in hashing is on and the file exists."""
    if not context.hash_files or not value:
        return None
    path = Path(str(value))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_record(row: Mapping[str, Any],
                     context: LedgerContext) -> dict[str, Any]:
    """One :mod:`pyqcu.test` ledger row in the reference's 100-column shape.

    Every key of :data:`REFERENCE_CSV_COLUMNS` is present; a column the formal
    port cannot observe is ``None`` (which the CSV writer renders as an empty
    field, exactly like an unobserved driver value).  Values are the *raw*
    Python objects of the row (list for lattice/grid, float for seconds) so the
    caller can serialise them like the reference does or feed them to the
    workbook writer without a second parse.
    """
    status = row.get("status")
    residual = _metric(row, "relative_residual", "rel_res", "true_residual")
    context_notes = list(_notes(row))
    mg_levels = _first(_metric(row, "mg_levels"), _opt(context, "mg_levels"))
    mg_restart = _first(_opt(context, "restart"),
                        _metric(row, "restart")
                        if _is_multigrid_row(row) else None)
    mg_grid = _first(_metric(row, "mg_grid"), context.option("mg_grid"))
    mg_dof = _mg_dof(row, mg_levels)
    return {
        "run_id": row.get("run_id"),
        "phase": phase_of(context.mode),
        "solver": _solver_of(row),
        "dtype": row.get("dtype"),
        "atol": _metric(row, "atol", "solver_tolerance"),
        "lattice": list(row["lattice"]) if _is_sequence(row.get("lattice"))
                   else row.get("lattice"),
        "lat_x": _lattice_extent(row, 0),
        "lat_y": _lattice_extent(row, 1),
        "lat_z": _lattice_extent(row, 2),
        "lat_t": _lattice_extent(row, 3),
        "mass": _metric(row, "mass"),
        "csw": row.get("csw"),
        "kappa": _metric(row, "kappa"),
        "ranks": _first(row.get("ranks"), _metric(row, "mpi_size")),
        "threads": _metric(row, "threads"),
        "rank_x_thread": _rank_x_thread(row),
        "grid": _grid(row, context),
        "grid_mode": _grid_mode(row, context),
        "shm": _shm(row),
        "source_kind": _first(row.get("source"), _metric(row, "source_kind")),
        "source_path": _first(row.get("source_file"),
                              _metric(row, "source_path")),
        "source_file_dtype": _first(_metric(row, "source_file_dtype"),
                                    context.input_dtype("source")),
        "source_sha256": _first(_metric(row, "source_sha256"),
                                context.input_sha256("source")),
        "seed": row.get("seed"),
        "gauge_kind": _first(row.get("gauge"), _metric(row, "gauge_kind")),
        "gauge_path": _first(row.get("gauge_file"),
                             _metric(row, "gauge_path")),
        "gauge_file_dtype": _first(_metric(row, "gauge_file_dtype"),
                                   context.input_dtype("gauge")),
        "gauge_sha256": _first(_metric(row, "gauge_sha256"),
                               context.input_sha256("gauge")),
        "cache_kind": None,
        "cache_path": _first(_metric(row, "cache_path"), _cache_path(context)),
        "cache_sha256": _first(_metric(row, "cache_file_sha256"),
                               _manifest(context, "file_sha256")),
        "cache_identity_sha256": _first(
            _metric(row, "cache_identity_sha256"),
            _manifest(context, "identity_sha256")),
        "cache_manifest_path": _first(_metric(row, "cache_manifest_path"),
                                      _cache_manifest_path(context)),
        "cache_manifest_sha256": _file_digest(
            context, _first(_metric(row, "cache_manifest_path"),
                            _cache_manifest_path(context))),
        "mg_levels": mg_levels,
        "mg_restart": mg_restart,
        "mg_grid": (list(mg_grid) if _is_sequence(mg_grid) else mg_grid),
        "mg_dof": mg_dof,
        "nv_iters": _metric(row, "nv_iters"),
        "coarse_max_iter": _metric(row, "coarse_max_iter"),
        "coarse_tol_factor": _metric(row, "coarse_tol_factor"),
        "deflate": _metric(row, "deflate"),
        "max_iter": _first(_metric(row, "max_iter"), _opt(context, "max_iter")),
        "verify_status": _metric(row, "verify_status"),
        "iterations": _metric(row, "iterations", "iters", "iteration"),
        "wall_seconds": row.get("seconds"),
        "builder_seconds": _metric(row, "cache_seconds"),
        "build_mg_cache_seconds": _first(_metric(row, "cache_setup_seconds"),
                                         _manifest(context, "setup_seconds")),
        "final_true_residual": residual,
        "relative_full_residual": residual,
        "distributed_relative_residual": None,
        "residual_source": _metric(row, "residual_source"),
        "converged": _metric(row, "converged"),
        "status": status,
        "node": None,
        "freq_khz": None,
        "freq_max_khz": None,
        "governor": None,
        "profile_rank": None,
        "dslash_seconds": None,
        "dslash_calls": None,
        "dslash_bytes_mb": None,
        "dslash_gbps_rank": None,
        "dslash_gbps_total": None,
        "clover_seconds": None,
        "clover_calls": None,
        "clover_bytes_mb": None,
        "clover_gbps_rank": None,
        "clover_gbps_total": None,
        "reduce_seconds": None,
        "reduce_calls": None,
        "halo_pack_seconds": None,
        "halo_sync_seconds": None,
        "halo_copy_seconds": None,
        "phase_prepare_seconds": None,
        "phase_solve_seconds": None,
        "phase_reconstruct_seconds": None,
        "input_prebind_seconds": None,
        "gauge_read_seconds": None,
        "source_read_seconds": None,
        "gauge_generate_seconds": None,
        "source_generate_seconds": None,
        "gauge_prewarm_seconds": _input_seconds(context, "gauge"),
        "source_prewarm_seconds": _input_seconds(context, "source"),
        "gauge_read_mode": None,
        "source_read_mode": None,
        "hdf5_mpio_read": None,
        "residual_seconds": _metric(row, "reference_seconds"),
        "report_path": row.get("report_path"),
        "report_sha256": _file_digest(context, row.get("report_path")),
        "solution_path": row.get("solution_path"),
        "solution_sha256": _file_digest(context, row.get("solution_path")),
        "out_path": None,
        "err_path": None,
        "returncode": None,
        "elapsed_seconds": _metric(row, "elapsed_seconds", "case_seconds"),
        "review": None,
        "timeout": (True if status == "TIMEOUT" else
                    (False if status in policy.STATUSES else None)),
        "error": (str(row.get("reason") or "") or None
                  if status == "ERROR" else None),
        "notes": context_notes,
    }


def _input_seconds(context: LedgerContext, role: str) -> Any:
    """``seconds`` of the prepare/prewarm entry for ``role`` (or ``None``)."""
    entry = context.input_entry(role)
    return None if entry is None else entry.get("seconds")


# ---------------------------------------------------------------------------
# prepare/prewarm input rows
# ---------------------------------------------------------------------------
def _input_run_id(role: str, entry: Mapping[str, Any], *,
                  mode: str | None = None) -> str:
    """Deterministic id of an input row.

    The reference uses ``cache-<cache_key>`` for prepare rows and
    ``prewarm-<sha1(role/lattice/dtype/kind/seed)>`` for prewarm rows.  The
    same *shape* is used here with the fields the formal port actually has: the
    cache manifest's ``identity_sha256`` when available, otherwise the hashed
    (role, path), so the id never depends on wall-clock time.  The prefix is
    the row's :func:`_input_phase`, so a ``cache-`` id never labels a gauge or
    source row.
    """
    kind = "cache" if _input_phase(role, mode) == "prepare" else "prewarm"
    if kind == "cache":
        identity = None
        if isinstance(entry.get("identity_sha256"), str):
            identity = entry["identity_sha256"]
        if identity:
            return f"cache-{identity[:16]}"
    payload = {key: json_safe(entry.get(key))
               for key in ("role", "path", "lattice", "dtype", "seed")}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return f"{kind}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def _input_phase(role: str, mode: str | None = None) -> str:
    """Reference ``phase`` of one input entry: ``prepare`` or ``prewarm``.

    The reference driver files a *cache* artifact under ``phase=prepare`` /
    ``solver=cache_build`` and a gauge/source read under ``phase=prewarm`` /
    ``solver=input_prewarm``, and its own ledger contains both kinds in one run
    (``{"solve": 130, "prewarm": 7, "prepare": 5}``).  The formal port's
    ``--mode prepare`` validates the gauge/source files *and* probes the cache,
    so the document phase would label a gauge validation row ``cache_build`` -
    a claim about an artifact that row never touched.  The role decides, and
    the document phase only decides for a role outside the three known ones.
    """
    text = str(role or "")
    if text == "cache":
        return "prepare"
    if text in ("gauge", "source"):
        return "prewarm"
    return "prepare" if phase_of(mode) == "prepare" else "prewarm"


def input_records(inputs: Iterable[Mapping[str, Any]], *,
                  context: LedgerContext, mode: str | None = None
                  ) -> list[dict[str, Any]]:
    """Prepare/prewarm input entries in the reference's row shape.

    The reference writes ``phase=prepare`` / ``solver=cache_build`` for cache
    artifacts and ``phase=prewarm`` / ``solver=input_prewarm`` for gauge/source
    inputs, with a small row that carries the artifact path and timings.  The
    formal port's ``prepare``/``prewarm`` documents carry the same facts
    (``role``/``path``/``status``/``bytes``/``sha256``/``seconds``), so they are
    rendered as rows instead of being dropped - otherwise the exported table
    would silently lose the input half of the ledger.  ``phase`` follows the
    entry's role (:func:`_input_phase`), not the document: one ``--mode
    prepare`` invocation produces both a cache row and input rows.
    """
    records: list[dict[str, Any]] = []
    for entry in inputs:
        if not isinstance(entry, Mapping):
            continue
        role = str(entry.get("role") or "")
        status = entry.get("status")
        phase = _input_phase(role, mode if mode is not None else context.mode)
        record = {name: None for name in REFERENCE_CSV_COLUMNS}
        record.update({
            "run_id": _input_run_id(role, entry, mode=phase),
            "phase": phase,
            "solver": PREPARE_SOLVER if phase == "prepare" else PREWARM_SOLVER,
            "dtype": context.option("dtype"),
            "lattice": (list(context.option("lattice"))
                        if _is_sequence(context.option("lattice")) else None),
            "lat_x": _extent(context, 0),
            "lat_y": _extent(context, 1),
            "lat_z": _extent(context, 2),
            "lat_t": _extent(context, 3),
            "seed": context.option("seed"),
            "status": status,
            "notes": _input_notes(entry),
        })
        if role == "gauge":
            record.update({
                "gauge_path": entry.get("path"),
                "gauge_sha256": entry.get("sha256"),
                "gauge_prewarm_seconds": entry.get("seconds"),
                "gauge_file_dtype": _entry_dtype(entry),
            })
        elif role == "source":
            record.update({
                "source_path": entry.get("path"),
                "source_sha256": entry.get("sha256"),
                "source_prewarm_seconds": entry.get("seconds"),
                "source_file_dtype": _entry_dtype(entry),
            })
        elif role == "cache":
            path = entry.get("path") or None
            artifact, manifest = _cache_entry_paths(path)
            record.update({
                "cache_path": artifact,
                "cache_manifest_path": manifest,
                "cache_identity_sha256": entry.get("identity_sha256"),
                "cache_sha256": entry.get("file_sha256"),
                "build_mg_cache_seconds": entry.get("setup_seconds"),
                # The manifest is the row's evidence file: reporting it is the
                # input-row counterpart of a solve row's report_path.
                "report_path": path,
            })
        records.append(record)
    return records


def _cache_entry_paths(path: Any) -> tuple[Any, Any]:
    """``(artifact, manifest)`` of a prepare-mode cache entry.

    ``--cache-dir`` may name the artifact (then the manifest is its ``.json``
    sibling) or the run's cache *directory* (then the entry path is the probe
    index ``<dir>/index.json`` and there is no artifact to name).  Writing the
    index as ``cache_path`` would claim an artifact that does not exist.
    """
    if path in (None, ""):
        return None, None
    text = str(path)
    if text.endswith(".json"):
        return None, text
    return text, str(Path(text).with_suffix(".json"))


def _extent(context: LedgerContext, index: int) -> Any:
    lattice = context.option("lattice")
    if _is_sequence(lattice) and len(lattice) > index:
        return int(lattice[index])
    return None


def _entry_dtype(entry: Mapping[str, Any]) -> Any:
    io = entry.get("io")
    return io.get("dtype") if isinstance(io, Mapping) else None


def _input_notes(entry: Mapping[str, Any]) -> list[str]:
    """Ordered notes of a prepare/prewarm input row."""
    notes = [f"role={entry.get('role')}"]
    if entry.get("bytes") is not None:
        notes.append(f"bytes={entry['bytes']}")
    if entry.get("path"):
        notes.append(f"path={entry['path']}")
    reason = str(entry.get("reason") or "").strip()
    if reason:
        notes.append(f"reason={reason}")
    io = entry.get("io")
    if isinstance(io, Mapping):
        for key in ("dataset", "dtype", "shape", "layout", "lattice",
                    "h5py_version"):
            if io.get(key) is not None:
                notes.append(f"io.{key}={io[key]}")
    return notes


# ---------------------------------------------------------------------------
# record sets
# ---------------------------------------------------------------------------
def reference_records(rows: Iterable[Mapping[str, Any]], *,
                      context: LedgerContext) -> list[dict[str, Any]]:
    """Reference-shaped records of ledger rows, ordered by ``run_id``.

    Sorting by ``run_id`` (not by execution order) is what ``merge_rows`` does,
    so a re-export of the same ledger is byte-identical regardless of the order
    the rows were solved in.
    """
    records = [reference_record(row, context)
               for row in rows if isinstance(row, Mapping)]
    records.sort(key=lambda record: str(record.get("run_id")))
    return records


def _status_rank(value: Any) -> int:
    """Rank of a status when two observations of one ``run_id`` compete.

    ``PASS`` (and the "no status" of a plan) is a success; every other status
    is not.  The collapse in :func:`_one_row_per_run_id` keeps the higher rank,
    so a later success can never hide an earlier failure.
    """
    text = "" if value is None else str(value).strip().upper()
    return 0 if text in ("", "PASS") else 1


def _observed_cells(record: Mapping[str, Any]) -> int:
    """Number of cells a record actually fills (``None``/``""`` are empty)."""
    return sum(1 for value in record.values()
               if value is not None and value != "")


def _one_row_per_run_id(
        batches: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """One row per ``run_id`` - the reference table's own invariant.

    ``--mode run`` reports the same gauge/source entry twice: once under
    ``steps.prepare`` (the file is validated and ``stat()``-ed) and once under
    ``steps.prewarm`` (it is read once and SHA256-hashed).  Both records carry
    the *same* ``run_id``, because :func:`_input_run_id` derives it from the
    role and :func:`_input_phase` files a gauge/source read under
    ``phase=prewarm`` in both steps.  Writing both would put two rows with the
    same id in the CSV - a downstream join on ``run_id`` double-counts - and
    the duplication would reach ``04_Performance_All`` (one row per record) and
    ``14_Input_Integrity`` (one row per record and role).  The reference has
    one row per executed row (``append_csv`` appends once, ``merge_rows``
    iterates the row files).

    ``batches`` are ``(step, records)`` pairs in pipeline order (the document's
    own ``inputs``, then ``steps.prepare``, then ``steps.prewarm``, then the
    solve rows).  Of several observations of one ``run_id`` the best one wins:

    1. a non-``PASS`` status outranks ``PASS`` - a failure is never hidden;
    2. more observed cells outrank fewer - the full prewarm read (SHA256 plus
       duration) wins over the prepare ``stat()``;
    3. a later pipeline step outranks an earlier one - the observation that
       supersedes.

    A record is taken *whole*: columns are never mixed between two
    observations, so nothing is invented and nothing unobserved is filled in.
    A record without a ``run_id`` cannot be identified and is never collapsed.
    """
    unique: list[dict[str, Any]] = []
    ranks: list[tuple[int, int, int]] = []
    slots: dict[str, int] = {}
    for step, batch in batches:
        for record in batch:
            row = dict(record)
            rank = (_status_rank(row.get("status")), _observed_cells(row), step)
            key = "" if row.get("run_id") is None else str(row["run_id"])
            index = None if not key else slots.get(key)
            if index is None:
                if key:
                    slots[key] = len(unique)
                unique.append(row)
                ranks.append(rank)
                continue
            if rank > ranks[index]:
                unique[index] = row
                ranks[index] = rank
    return unique


def ledger_records(payload: Mapping[str, Any], *,
                   rows: Sequence[Mapping[str, Any]] | None = None,
                   run_options: Mapping[str, Any] | None = None,
                   cache_manifest: Mapping[str, Any] | None = None,
                   hash_files: bool = False) -> list[dict[str, Any]]:
    """Reference-shaped records of a full-mode document.

    ``payload`` is what ``run_mode`` returns (and ``--json`` prints): for
    ``solve``/``run`` it carries ``rows.items`` together with the ``options``
    and ``inputs`` blocks the per-column derivations need; for ``prepare``/
    ``prewarm`` it carries ``inputs`` and produces input rows; for ``list`` it
    carries planned rows whose ``status`` is ``None`` (rendered with an empty
    status - the reference has no "planned" phase, so a plan is not exported as
    a measurement).  ``merge``/``validate`` documents carry ``rows.items`` too,
    so the same function rebuilds the table from the row files alone.

    ``--mode run`` nests the prepare/prewarm documents under ``steps``; their
    ``inputs`` are exported with the phase they were produced under, so the
    reference's ``phase`` column keeps all three values.  A run document lists
    the same gauge/source entry under ``prepare`` *and* ``prewarm`` (both steps
    looked at it), and both entries derive the same ``run_id``; the export
    collapses them to **one row per ``run_id``** (:func:`_one_row_per_run_id`),
    because the reference has one row per executed row and a duplicated id
    makes a downstream join double-count.  The surviving row is a real
    observation taken whole: a failure is never hidden by a later success, the
    fuller observation wins (the prewarm read with its SHA256 and duration over
    the prepare ``stat()``), and a late step supersedes an early one.

    ``run_options`` is the run configuration (``Test.config()``).  Pass it
    whenever you have it: the columns ``grid``, ``grid_mode``, ``mg_grid``,
    ``mg_levels``, ``mg_restart`` and ``max_iter`` describe the *run*, and a
    full-mode payload only carries the *mode* options.
    """
    context = LedgerContext.from_payload(payload, run_options=run_options,
                                        cache_manifest=cache_manifest,
                                        hash_files=hash_files)
    document_rows = rows
    if document_rows is None:
        block = payload.get("rows")
        items = block.get("items") if isinstance(block, Mapping) else None
        document_rows = [item for item in (items or ())
                         if isinstance(item, Mapping)]
    batches: list[tuple[int, list[dict[str, Any]]]] = []
    mode = phase_of(context.mode)
    if mode in ("prepare", "prewarm"):
        batches.append((0, input_records(context.inputs, context=context)))
    steps = payload.get("steps")
    if isinstance(steps, Mapping):
        for step, (step_name, step_mode) in enumerate(
                (("prepare", "prepare"), ("prewarm", "prewarm")), start=1):
            step_document = steps.get(step_name)
            entries = (step_document.get("inputs")
                       if isinstance(step_document, Mapping) else None)
            batches.append((step, input_records(
                [entry for entry in (entries or ())
                 if isinstance(entry, Mapping)],
                context=context, mode=step_mode)))
    batches.append((len(batches) + 1,
                    reference_records(document_rows, context=context)))
    # One row per ``run_id``: a run document reports the same input under both
    # steps (see :func:`_one_row_per_run_id`), and a duplicated id would make
    # every join on it double-count.
    records = _one_row_per_run_id(batches)
    records.sort(key=lambda record: str(record.get("run_id")))
    return records


# ---------------------------------------------------------------------------
# text and atomic writes
# ---------------------------------------------------------------------------
def reference_csv_text(records: Iterable[Mapping[str, Any]], *,
                       columns: Sequence[str] = REFERENCE_CSV_COLUMNS) -> str:
    """The CSV text of ``records`` (reference header, CRLF, Python repr cells).

    A :class:`csv.writer` is used - not manual string joins - so quoting and
    escaping are the same as the driver's ``csv.DictWriter``.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=LINE_TERMINATOR)
    writer.writerow(list(columns))
    for record in records:
        writer.writerow([record.get(name) for name in columns])
    return buffer.getvalue()


def _fsync_directory(directory: Path) -> None:
    """Best-effort ``fsync`` of a directory entry (ignored when unsupported)."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_text(target: Path, text: str) -> None:
    """Write ``text`` atomically: sibling temp file, fsync, ``os.replace``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target.parent)


def _read_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read an existing CSV into ``(header, records)``; empty when unusable."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = [name for name in (reader.fieldnames or [])]
            return header, [dict(record) for record in reader]
    except (OSError, csv.Error, UnicodeDecodeError):
        return [], []


def upgrade_columns(path: Any, *,
                    columns: Sequence[str] = REFERENCE_CSV_COLUMNS) -> list[str]:
    """Header to write for ``path``, upgrading an older table in place.

    The counterpart of ``upgrade_csv_schema``: a CSV written by an earlier
    version keeps its old header, and appending reference rows to it would shift
    every value.  The file is rewritten with the reference header plus any
    column the old header had and the reference does not (so no data is
    dropped), every existing record is preserved by name, and a one-shot
    ``<name>.schema.bak`` backup is left next to it.
    """
    target = Path(str(path))
    wanted = list(columns)
    if not target.is_file() or target.stat().st_size == 0:
        return wanted
    header, records = _read_records(target)
    if not header or header == wanted:
        return wanted
    merged = wanted + [name for name in header if name not in wanted]
    backup = target.with_name(f"{target.name}.schema.bak")
    if not backup.exists():
        try:
            backup.write_bytes(target.read_bytes())
        except OSError:
            pass
    _atomic_write_text(target, reference_csv_text(records, columns=merged))
    return merged


def write_ledger_csv(path: Any, records: Iterable[Mapping[str, Any]], *,
                     columns: Sequence[str] | None = None,
                     upgrade: bool = True) -> Path:
    """Atomically write ``records`` to ``path`` in the reference CSV format.

    ``upgrade`` keeps legacy-only columns of an existing table (the reference's
    ``upgrade_csv_schema``); pass ``False`` to write exactly
    :data:`REFERENCE_CSV_COLUMNS`.
    """
    target = Path(str(path))
    chosen = list(columns) if columns is not None else (
        upgrade_columns(target) if upgrade else list(REFERENCE_CSV_COLUMNS))
    _atomic_write_text(target, reference_csv_text(records, columns=chosen))
    return target


# ---------------------------------------------------------------------------
# workbook column selection (shared with ledger_excel)
# ---------------------------------------------------------------------------
#: The 44 curated columns of the reference workbook's ``04_Performance_All``
#: sheet, in order.  The workbook is *not* a dump of the CSV: ``status`` is
#: moved to position C, only a subset of the 100 columns is carried, and
#: ``error``/``timeout`` follow ``notes`` instead of preceding it.
PERFORMANCE_COLUMNS: tuple[str, ...] = (
    "run_id", "phase", "status", "dtype", "lattice", "solver", "grid_mode",
    "grid", "rank_x_thread", "ranks", "threads", "shm", "source_kind",
    "gauge_kind", "seed", "mg_restart", "atol", "mass", "csw", "wall_seconds",
    "builder_seconds", "build_mg_cache_seconds", "iterations",
    "final_true_residual", "relative_full_residual",
    "distributed_relative_residual", "verify_status", "node", "freq_khz",
    "freq_max_khz", "governor", "gauge_read_seconds", "elapsed_seconds",
    "hdf5_mpio_read", "cache_kind", "cache_sha256", "gauge_sha256",
    "source_sha256", "solution_sha256", "report_path", "notes", "error",
    "returncode", "timeout",
)


def performance_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one reference record onto :data:`PERFORMANCE_COLUMNS`."""
    return {name: record.get(name) for name in PERFORMANCE_COLUMNS}
