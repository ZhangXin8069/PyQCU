"""Observation-only row driver for the full modes of :mod:`pyqcu.test`.

Why this module exists
---------------------
:data:`pyqcu.test.common.MODE_HOOKS` has always declared ``driver:("run_row",)``
and :func:`pyqcu.test.common._solve_row` has always looked it up, but the module
did not exist, so every full-mode row silently fell back to the tier-0 executor
and ``--gauge-file``/``--source-file`` were only *evidence*: the numeric field
still came from ``--gauge``/``--source``.  This module is that missing hook for
the slice the L48 design (``docs/pyqcu_test_l48_remote.md`` §6) asks for:

* the row's inputs are the row's inputs - an HDF5 ``--gauge-file`` /
  ``--source-file`` is read, validated and hashed here, and the very same file
  is then handed to the native runner;
* threads, ranks, shared halo and input origin are *recorded* per row, so a
  reader never has to infer them from a command line;
* the driver returns **observations only**.  It never returns ``PASS``/``FAIL``:
  the harness evaluates the case's ``expect`` gates
  (:func:`pyqcu.test.common._apply_driver`), exactly like a plain case run.

What a row is
-------------
A row this driver owns has the tag ``"<backend>/native/<kind>"`` with
``backend`` in ``pyqcu.test.native.BACKENDS`` and ``kind`` in
:data:`NATIVE_KINDS`.  Those rows are produced by
:func:`pyqcu.test.native.case` (``python -m pyqcu.test --mode solve --native
solve``).  Every other tag - the tier-0 ``torch``/``kupl``/``cuda`` cases -
returns ``None``, which the harness reads as "not mine" and runs through the
built-in executor exactly as before; that is a routing decision, not a
substitution of a requested input.

Statuses
--------
A mapping without ``status`` is a completed native call: the harness judges it.
A driver may *declare* only an execution outcome:

``SKIP``   the launch cannot produce the row (an environmental absence, e.g.
           ``kupl_library`` missing, or a rank count that does not match
           ``MPI_COMM_WORLD``); always with a reason
``ERROR``  an input/option contradiction (unreadable file, wrong lattice or
           dtype, a solver the native runner cannot drive); always with a reason
``TIMEOUT`` reserved for a wall-clock overrun

``PASS``/``FAIL`` from a driver are turned into ``ERROR`` by the harness.

The backend is the row's
-----------------------
The tag names the backend (``cuda/native/solve`` is a CUDA row) and the ledger
row carries the same name.  That name - not a run-level default - is passed to
:func:`pyqcu.test.native.run_case`, because ``run_case`` resolves an omitted
backend to ``native.DEFAULT_BACKEND`` (``kupl``): a CUDA-tagged row would
otherwise compute on KUPL and be recorded as a CUDA result (audit N7).  A row
whose ``backend`` field contradicts its tag is an ``ERROR`` rather than a guess.

What is still missing (explicit TODO, not a silent fallback)
------------------------------------------------------------
The driver runs the native call **in this process** (one Python process = one
MPI rank, which is what :mod:`pyqcu.test.native` implements).  It does not start
``mpirun``: there is no rankfile/``CCS_ALLOC_FILE`` handling yet
(``docs/pyqcu_test_l48_remote.md`` §5.3), so a row whose ``ranks`` exceed the
current world size is reported ``SKIP`` with the launch it would need instead of
being run at the wrong size.  Every result carries ``launcher`` evidence saying
so.

Multi-rank: one logical row
---------------------------
A ``mpirun -np 32``/``drun`` launch runs this function on all 32 ranks of the
same row, so the row must behave as one logical row rather than 32 independent
ones.  Two collectives - both from :mod:`pyqcu.test.common`, both entered by
every rank exactly once per row - do that without a second runner:

``pre-flight`` (before the native call) allgathers "can this rank run?"; if any
rank must skip (``mpi_grid``, an unreadable input, a library missing on that
node), **every** rank skips.  That keeps ``SKIP(mpi_grid)`` ahead of execution
and stops a rank from entering a native collective the others will never join
(the hang of audit P0-5).

``outcome`` (after the native call) allgathers the compact record
``status``/``skip_token``/``reason``/``iterations``/``relative_residual``/
``converged``; the **worst** rank wins (``ERROR`` > ``FAIL`` > ``TIMEOUT`` >
``SKIP`` > ``PASS``), the worst numbers are kept as the row's evidence, and
``rank_disagreement`` records whether the ranks agreed.  A single rank's
``ERROR`` can therefore never be hidden by 31 ``PASS`` rows, and the row's
``iterations``/``relative_residual`` are not rank 0's alone.

Both collectives degrade to a no-op with an explicit
``rank_gather``/``rank_gather_reason`` observation when there is no usable
communicator (one process, or mpi4py absent); they never block and never claim
the ranks agreed when they could not be compared.

Import rules
------------
Standard library plus :mod:`pyqcu.test.native` (itself stdlib-only at import
time).  ``h5py``/``numpy``/``torch`` are imported inside the functions that need
them, so ``--list``/``--mode list`` never pay for a backend.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import common, l48, native, policy
from .common import Skip

__all__ = [
    "NATIVE_KINDS",
    "balanced_grid",
    "input_identity",
    "owns_row",
    "parse_tag",
    "run_row",
]

#: Native case kinds :func:`pyqcu.test.native.run_case` can drive.
NATIVE_KINDS: tuple[str, ...] = ("dslash", "solve", "mg")

#: Tag segment that marks a native full-mode row (``<backend>/native/<kind>``).
NATIVE_SEGMENT = "native"

#: Observation keys with harness meaning; the rest are observations/metrics.
_RESERVED = ("status", "reason", "skip_token", "failed", "seconds")


# ---------------------------------------------------------------------------
# row classification
# ---------------------------------------------------------------------------
def parse_tag(tag: Any) -> tuple[str, str] | None:
    """``(backend, kind)`` of a native or L48-matrix row tag, else ``None``.

    ``"kupl/native/solve"`` is a native full-mode row; ``"kupl/solve/bistabcg"``
    is the tier-0 case of the same backend and belongs to the built-in
    executor.  ``"kupl/l48/bicgstab/c64"`` is a row of the tier-1 L48 two-node
    matrix (:mod:`pyqcu.test.l48`), whose fixed parameters this driver applies.
    """
    matrix = l48.parse_tag(tag)
    if matrix is not None:
        return matrix
    text = str(tag if tag is not None else "").strip()
    parts = text.split("/")
    if len(parts) != 3:
        return None
    backend, segment, kind = (part.strip() for part in parts)
    if segment != NATIVE_SEGMENT or kind not in NATIVE_KINDS:
        return None
    if backend not in native.BACKENDS:
        return None
    return backend, kind


def owns_row(row: Mapping[str, Any] | None) -> bool:
    """True when :func:`run_row` drives ``row`` (see :func:`parse_tag`)."""
    if not isinstance(row, Mapping):
        return False
    return parse_tag(row.get("tag")) is not None


# ---------------------------------------------------------------------------
# rank grid
# ---------------------------------------------------------------------------
def balanced_grid(lattice: Sequence[int], ranks: int) -> tuple[int, int, int, int] | None:
    """Deterministic rank grid for ``ranks`` on ``lattice`` (``None``: none).

    The native backends split a parity-compact field, so every local extent must
    stay even and divide the global one (see
    ``docs/pyqcu_test_l48_remote.md`` §5.1).  Factors are handed out
    round-robin from ``t`` to ``x`` while the product still divides ``ranks``,
    which reproduces the documented production shapes::

        balanced_grid((48, 48, 48, 96), 16)  # (2, 2, 2, 2) local 24^3 x 24
        balanced_grid((48, 48, 48, 96), 32)  # (2, 2, 2, 4) local 24^3 x 24
        balanced_grid((48, 48, 48, 96), 64)  # (2, 2, 4, 4) local 24x24x12x24
        balanced_grid((8, 8, 8, 16), 1)      # (1, 1, 1, 1)

    ``None`` means "no grid with even local extents divides this lattice for
    that many ranks"; the caller must report that instead of running with a
    different size.
    """
    shape = tuple(int(value) for value in lattice)
    if len(shape) != 4:
        raise ValueError(f"lattice must have 4 extents, got {lattice!r}")
    world = int(ranks)
    if world < 1:
        raise ValueError(f"ranks must be >= 1, got {ranks!r}")
    if any(extent % 2 for extent in shape):
        raise ValueError(f"lattice extents must be even, got {shape}")
    factors = [1, 1, 1, 1]
    # Largest factor of ``world`` each axis can take while keeping the local
    # extent even and non-empty.
    limits = [math.gcd(world, shape[axis]) for axis in range(4)]
    product = 1
    while product < world:
        progressed = False
        for axis in (3, 2, 1, 0):
            candidate = factors[axis] * 2
            if product * 2 > world or world % (product * 2):
                continue
            if limits[axis] % candidate:
                continue
            if (shape[axis] // candidate) < 2:
                continue
            factors[axis] = candidate
            product *= 2
            progressed = True
            if product == world:
                break
        if not progressed:
            return None
    if any((shape[axis] // factors[axis]) % 2 for axis in range(4)):
        return None
    return tuple(factors)  # type: ignore[return-value]


def _row_grid(row: Mapping[str, Any], lattice: Sequence[int], ranks: int
              ) -> tuple[tuple[int, int, int, int] | None, str]:
    """Rank grid of one row: an explicit ``lattice_grid`` wins, else derived."""
    declared = row.get("lattice_grid")
    if declared not in (None, "", []):
        try:
            grid = tuple(int(value) for value in declared)
        except (TypeError, ValueError):
            return None, "invalid"
        if len(grid) != 4 or any(value < 1 for value in grid):
            return None, "invalid"
        return grid, "row"  # type: ignore[return-value]
    return balanced_grid(lattice, ranks), "derived"


# ---------------------------------------------------------------------------
# input identity
# ---------------------------------------------------------------------------
def input_identity(role: str, path: Any, *, lattice: Sequence[int],
                   dtype: str) -> dict[str, Any]:
    """Validate one HDF5 ``role`` artifact and return its identity.

    Uses :mod:`pyqcu.test.io` (the only reader: it opens the file, checks the
    dataset/shape/dtype/attributes and touches real elements) plus a streamed
    SHA256.  ``role`` is ``"gauge"`` or ``"source"``.

    Raises whatever ``io.describe`` raises (``ArtifactMissing``,
    ``ArtifactCorrupt``, ``ArtifactSchemaError``) for an unusable file, and
    ``ValueError`` for a file that is valid but describes another lattice or
    another complex precision: a run must never consume a file that is not the
    one its row declares.
    """
    from . import io

    target = Path(str(path)).expanduser()
    record = io.describe(target, role=role)
    declared = tuple(int(value) for value in record["lattice"])
    wanted_lattice = tuple(int(value) for value in lattice)
    if declared != wanted_lattice:
        raise ValueError(
            f"{role} file {target} holds lattice {list(declared)}, the row asks "
            f"{list(wanted_lattice)}")
    wanted_dtype = "complex128" if str(dtype) == "c128" else "complex64"
    if str(record["dtype"]) != wanted_dtype:
        raise ValueError(
            f"{role} file {target} holds {record['dtype']}, the row asks "
            f"{dtype} ({wanted_dtype}); convert the file explicitly")
    return {
        "role": role,
        "origin": "hdf5",
        "path": str(target),
        "sha256": io.sha256_file(target),
        "layout": str(record["layout"]),
        "file_dtype": str(record["dtype"]),
        "lattice": [int(value) for value in declared],
        "bytes": int(record["bytes"]),
        "file_bytes": int(record["file_bytes"]),
    }


def _requested_input(role: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """The input a row *declares*, before any I/O (always recorded).

    Recording the request first means a row whose file is rejected still shows
    ``gauge_origin="hdf5"`` with the path and ``sha256=None`` next to the error:
    the ledger then says "this file was requested and not consumed", which is
    the opposite of the 2026-11 failure mode where the row looked like a clean
    generated-field run.
    """
    path = row.get(f"{role}_file")
    requested = path not in (None, "")
    return {
        f"{role}_origin": "hdf5" if requested else row.get(role),
        f"{role}_path": str(path) if requested else None,
        f"{role}_sha256": None,
        f"{role}_layout": None,
        f"{role}_lattice": None,
        f"{role}_file_dtype": None,
        f"{role}_bytes": None,
    }


def _validated_input(role: str, path: Any, lattice: Sequence[int],
                     dtype: str) -> dict[str, Any]:
    """The origin/path/hash observations of one validated HDF5 input."""
    identity = input_identity(role, path, lattice=lattice, dtype=dtype)
    return {
        f"{role}_origin": identity["origin"],
        f"{role}_path": identity["path"],
        f"{role}_sha256": identity["sha256"],
        f"{role}_layout": identity["layout"],
        f"{role}_lattice": identity["lattice"],
        f"{role}_file_dtype": identity["file_dtype"],
        f"{role}_bytes": identity["bytes"],
        f"{role}_file_bytes": identity["file_bytes"],
    }


# ---------------------------------------------------------------------------
# small coercions
# ---------------------------------------------------------------------------
def _int_or(value: Any, default: int | None) -> int | None:
    """``int(value)`` when it is a usable integer, else ``default``."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or(value: Any) -> float | None:
    """Finite ``float(value)``, else ``None`` (an unset physics field)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _split_skip(reason: Any, token: str | None) -> tuple[str, str | None]:
    """Split ``"token: reason"`` into ``(reason, token)`` when possible."""
    text = str(reason or "").strip() or "native runner skipped without a reason"
    if token is not None and not policy.is_skip_token(token):
        token = None
    if token is None and ":" in text:
        head, _, tail = text.partition(":")
        if policy.is_skip_token(head.strip()) and tail.strip():
            return tail.strip(), head.strip()
    return text, token


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run_row(*, row: Mapping[str, Any] | None = None,
            options: Mapping[str, Any] | None = None,
            paths: Mapping[str, Any] | None = None,
            tag: Any = None) -> dict[str, Any] | None:
    """Execute one native row and return observations (never a verdict).

    Parameters are the documented hook contract
    (``docs/pyqcu_test_l48_remote.md`` §6): ``row`` is the ledger row,
    ``options`` is :meth:`pyqcu.test.common.FullModeOptions.as_dict`, ``paths``
    is :meth:`pyqcu.test.common.ModePaths.as_dict` and ``tag`` is the case tag.

    Returns ``None`` when the row is not a native full-mode row (the harness
    then runs it through the built-in executor), otherwise a mapping of
    observations - including ``gauge_origin``/``gauge_sha256``,
    ``source_origin``/``source_sha256``, ``ranks``/``mpi_size``,
    ``lattice_grid``/``local_lattice``, ``threads_requested``/
    ``threads_applied`` and the ``launcher`` evidence - or a mapping carrying an
    execution ``status`` (``SKIP``/``ERROR``/``TIMEOUT``) with a non-empty
    ``reason``.  ``PASS``/``FAIL`` are never returned.

    Multi-rank rows are **one logical row** (see the module docstring): the
    launch decision is agreed by all ranks before the native call, and the
    outcome is an allgather of ``status``/``skip_token``/``reason``/
    ``iterations``/``relative_residual``/``converged`` whose worst rank wins.
    The returned observations therefore carry ``rank_disagreement``,
    ``mpi_rank``/``mpi_size``, ``rank_gather``, ``rank_outcomes`` and the worst
    of the three numbers, and a rank that saw ``ERROR`` fails the row even if
    the other 31 ranks passed.
    """
    data = dict(row or {})
    resolved_tag = tag if tag is not None else data.get("tag")
    parsed = parse_tag(resolved_tag)
    if parsed is None:
        return None
    backend, kind = parsed
    settings = dict(options or {})
    layout = dict(paths or {})

    started = time.perf_counter()
    # An L48 two-node row carries its declaration as the source of truth: the
    # lattice, dtype, rank count, grid and solver parameters come from
    # ``pyqcu.test.l48``, not from a run-level flag that may default to one
    # rank.  A native row keeps the previous row-driven behaviour.
    declaration = l48.parameters(resolved_tag)
    lattice = (l48.LATTICE if declaration
               else policy.validate_lattice(
                   data.get("lattice") or policy.DEFAULT_LATTICE))
    dtype = policy.validate_dtype(
        declaration["dtype"] if declaration
        else (data.get("dtype") or policy.DEFAULT_DTYPE))
    world, rank = native.mpi_size(), native.mpi_rank()
    ranks = (int(declaration["ranks"]) if declaration
             else (_int_or(data.get("ranks"), world) or world))
    threads = _int_or(data.get("threads"), None)
    shm = data.get("shm")
    observations: dict[str, Any] = {
        "driver": "pyqcu.test.driver",
        "driver_ok": True,
        "run_id": data.get("run_id"),
        "row_tag": str(data.get("tag") or ""),
        "backend": backend,
        "lattice": list(lattice),
        "dtype": dtype,
        "mass": _float_or(data.get("mass")),
        "kappa": _float_or(data.get("kappa")),
        "csw": _float_or(data.get("csw")),
        "seed": _int_or(data.get("seed"), None),
        # Threads/ranks/shm are *recorded* here and *applied* by
        # ``native.apply_environment`` (OMP_NUM_THREADS/KUPL_OMP_THREADS/
        # KUPL_SHM_HALO); the applied values come back as ``threads_applied``/
        # ``shm_applied`` after the call.
        "threads_requested": threads,
        "shm_requested": shm,
        "shared": bool(data.get("shared")),
        "ranks": ranks,
        "mpi_size": world,
        "mpi_rank": rank,
        "ranks_match": int(ranks) == int(world),
        "timeout_seconds": _float_or(settings.get("timeout")),
        "repeat": _int_or(settings.get("repeat"), 1),
        "warmup": _int_or(settings.get("warmup"), 1),
        "solver": declaration["solver"] if declaration else data.get("solver"),
        "dslash": declaration["dslash"] if declaration else data.get("dslash"),
        "gauge_field": data.get("gauge"),
        "source_field": data.get("source"),
        # The MPI launcher is the remaining TODO (see the module docstring);
        # recording the gap is part of the evidence contract.
        "launcher": {
            "kind": "in-process",
            "mpi_launch": False,
            "mpirun": None,
            "rankfile": None,
            "nodes": None,
            "ranks_per_node": None,
            "note": ("one Python process per MPI rank; the cross-node launcher "
                     "(mpirun + rankfile from CCS_ALLOC_FILE) is not "
                     "implemented yet - docs/pyqcu_test_l48_remote.md 5.3"),
        },
    }
    if declaration:
        launcher = l48.launch_plan(resolved_tag)
        observations.update({
            "l48_group": declaration["group"],
            "l48_evidence": declaration["evidence"],
            "l48_blocked_by": declaration["blocked_by"],
            "l48_notes": declaration["notes"],
            "nodes": declaration["nodes"],
            "ranks_per_node": declaration["ranks_per_node"],
            "mg_levels": declaration["mg_levels"],
            "restart": declaration["restart"],
        })
        observations["launcher"].update({
            "kind": "in-process-mpi-rank",
            "mpirun": launcher["command"],
            "nodes": launcher["nodes"],
            "ranks_per_node": launcher["ranks_per_node"],
            "grid": launcher["grid"],
            "scheduler_required": launcher["scheduler_required"],
            "note": ("the L48 matrix declares the mpirun/rankfile plan; the "
                     "cross-node launcher itself is not implemented "
                     "(docs/pyqcu_test_l48_remote.md 5.3)"),
        })

    # 1) Input identity first: an unusable input is reported before any compute,
    #    and ``--gauge-file``/``--source-file`` are inputs here, not evidence.
    #    Every check below only *records* a failure: the row's verdict is the
    #    result of the collective funnel at the end, because one rank returning
    #    early while the others keep going is exactly what deadlocks a 32-rank
    #    job (docs/pyqcu_test_l48_remote.md 0.1.6, audit P0-5).
    failure = _input_failure(observations, data, lattice, dtype, declaration)

    # 2) The backend is the row's *own*, and it is the one that executes.  The
    #    tag names it (``cuda/native/solve`` is a CUDA row) and ``build_rows``
    #    copies the same name into the row; the resolved name is handed to
    #    ``native.run_case`` below.  Without that argument every row falls
    #    through to ``native.DEFAULT_BACKEND`` ("kupl"), so a CUDA-tagged row
    #    would silently execute KUPL and be recorded as a CUDA pass (audit N7).
    #    A row whose field contradicts its tag is an ERROR, never a coin flip.
    backend, backend_failure = _row_backend(data, backend, resolved_tag)
    if failure is None:
        failure = backend_failure

    # 3) Geometry: the row's lattice and rank count decide the grid.  A grid
    #    that cannot be tiled, or a rank count the launch does not provide, is
    #    reported - never replaced by a smaller lattice or fewer ranks.  An L48
    #    matrix row uses its declared grid (the matrix is the source of truth).
    if failure is None:
        failure = _geometry_failure(observations, data, lattice, ranks, world,
                                    declaration)

    # 4) The launch decision is unanimous: a rank that has to skip stops the
    #    launch on every rank, so nobody enters a native collective the others
    #    will never join.  This happens *before* the native call, so a rank
    #    mismatch still wins over execution.
    observations, failure = _agree_on_launch(
        observations, failure, world=world, rank=rank)
    if failure is not None:
        return _failed(observations, started, failure[0], failure[1],
                       token=failure[2])

    # 5) The native call: one case kind, the row's backend, the row's inputs,
    #    this process's rank.
    failure = _native_failure(
        observations, started, kind=kind, backend=backend, lattice=lattice,
        dtype=dtype, grid=observations["lattice_grid"],
        declaration=declaration, data=data, layout=layout, settings=settings)
    # 6) The outcome is one logical row: the compact allgather of
    #    status/skip_token/reason/iterations/relative_residual/converged picks
    #    the worst rank, so a single rank's ERROR cannot hide behind 31 PASSes.
    return _finish(observations, started, failure, world=world, rank=rank)


# ---------------------------------------------------------------------------
# the row funnel: record a failure, agree on the launch, gather the outcome
# ---------------------------------------------------------------------------
#: A row failure: ``(status, reason, skip token)``.  ``status`` is one of the
#: execution statuses a driver may declare (never ``PASS``/``FAIL``).
_Failure = tuple[str, str, str | None]

#: Observations of the gathered rank evidence (see
#: :func:`pyqcu.test.common.gather_rank_outcome`).  They are merged into
#: ``row["metrics"]`` by the harness and are what makes a 32-process run
#: reviewable as one row.
_GATHERED_KEYS: tuple[str, ...] = (
    "rank_phase", "rank_gather", "rank_gather_reason", "rank_disagreement",
    "rank_status", "rank_worst", "mpi_rank", "mpi_size", "rank_outcomes")


#: Backend spellings :func:`pyqcu.test.native._resolve_backend` reads as "the
#: default backend".  A row carrying one of them claims no backend of its own, so
#: it does not contradict its tag; any other name must match the tag exactly.
_BACKEND_UNCLAIMED: tuple[str, ...] = ("", "auto", "all")


def _row_backend(data: Mapping[str, Any], backend: str, tag: Any
                 ) -> tuple[str, _Failure | None]:
    """The backend ``native.run_case`` must execute for this row.

    The tag names the backend (:func:`parse_tag`) and a ledger row carries the
    same name in its ``backend`` field (``build_rows`` copies
    ``Case.backend``).  Returning it is what keeps a CUDA-tagged row from
    silently executing the KUPL default: :func:`pyqcu.test.native.run_case`
    resolves ``None``/``"auto"`` to ``native.DEFAULT_BACKEND``, so omitting the
    argument turns every row into a KUPL run (audit N7).  A row whose field
    disagrees with its tag is an ``ERROR``: executing one backend while the
    ledger records another is exactly the substitution a row must never
    tolerate, and there is no basis for preferring either name.
    """
    claim = str(data.get("backend") or "").strip().lower()
    if claim in _BACKEND_UNCLAIMED or claim == str(backend):
        return str(backend), None
    return str(backend), (
        "ERROR",
        f"row backend {claim!r} contradicts its tag {str(tag)!r} (backend "
        f"{backend!r}); refusing to execute one backend under another's name",
        None)


def _input_failure(observations: dict[str, Any], data: Mapping[str, Any],
                   lattice: Sequence[int], dtype: str,
                   declaration: Mapping[str, Any] | None) -> _Failure | None:
    """Validate the row's HDF5 inputs; return the failure instead of raising.

    The requested identity is recorded *before* the file is opened, so a row
    whose input is rejected still shows ``gauge_origin="hdf5"`` with
    ``gauge_sha256=None`` next to the error (see :func:`_requested_input`).
    """
    if declaration and declaration["requires_input"]:
        missing = [role for role in ("gauge", "source")
                   if data.get(f"{role}_file") in (None, "")]
        if missing:
            return ("ERROR",
                    "the L48 input/IO group needs a real L48 HDF5 "
                    f"--gauge-file/--source-file; missing: "
                    f"{', '.join('--' + role + '-file' for role in missing)}",
                    None)
    for role in ("gauge", "source"):
        observations.update(_requested_input(role, data))
        path = data.get(f"{role}_file")
        if path in (None, ""):
            continue
        try:
            observations.update(_validated_input(role, path, lattice, dtype))
        except Exception as error:  # noqa: BLE001 - reported as an observation
            return ("ERROR",
                    f"{role} input rejected: {type(error).__name__}: {error}",
                    None)
    return None


def _geometry_failure(observations: dict[str, Any], data: Mapping[str, Any],
                      lattice: Sequence[int], ranks: int, world: int,
                      declaration: Mapping[str, Any] | None) -> _Failure | None:
    """Resolve the rank grid and check it against ``MPI_COMM_WORLD``."""
    if declaration:
        grid, grid_source = tuple(declaration["grid"]), "l48-matrix"
    else:
        grid, grid_source = _row_grid(data, lattice, ranks)
    if grid is None:
        return ("SKIP",
                f"no rank grid with even local extents tiles lattice "
                f"{list(lattice)} for {ranks} rank(s); the row was not run",
                "mpi_grid")
    observations["lattice_grid"] = list(grid)
    observations["grid_source"] = grid_source
    observations["local_lattice"] = list(native.local_lattice(lattice, grid))
    if int(ranks) != int(world):
        return ("SKIP",
                f"the row declares {ranks} rank(s) but MPI_COMM_WORLD has "
                f"{world}; run it under 'mpirun -np {ranks}' (the launcher is "
                "still a TODO for this port) instead of accepting a "
                "wrong-size result",
                "mpi_grid")
    return None


def _agree_on_launch(observations: dict[str, Any], failure: _Failure | None,
                     *, world: int, rank: int
                     ) -> tuple[dict[str, Any], _Failure | None]:
    """Collective pre-flight: the native call runs only if **every** rank can run.

    A rank that has to skip (``mpi_grid``, an unreadable input, a library that
    is not loadable there) must stop the launch on all ranks: if the other 31
    ranks entered ``native.run_case`` while this one returned, they would wait
    in a collective the failed rank never joins - the hang the audit records as
    P0-5.  The pre-flight allgather makes "run" and "do not run" unanimous and
    happens before any native call, so ``SKIP(mpi_grid)`` still wins over
    execution.

    Without a usable communicator (one process, or no mpi4py) the local decision
    stands and the ``rank_gather`` observation says so - never a silent claim
    that the ranks agreed.

    The returned failure is the **worse** of this rank's own decision and the
    gathered one, so every rank of the world reports the same execution status
    even when the local failure is the milder of several (a rank that must
    ``SKIP`` while another rank ``ERROR``-ed reports the ``ERROR``, exactly as
    the row will be written).
    """
    merged = common.gather_rank_outcome(
        observations, size=world, rank=rank,
        status="PASS" if failure is None else failure[0],
        skip_token=None if failure is None else failure[2],
        reason=None if failure is None else failure[1],
        phase="preflight")
    observations.update({key: merged[key] for key in _GATHERED_KEYS
                         if key in merged})
    local = "PASS" if failure is None else str(failure[0])
    worst = str(merged.get("rank_status") or "PASS")
    if common.rank_status_key(worst) < common.rank_status_key(local):
        return (observations,
                (worst, merged.get("rank_reason") or f"rank verdict {worst}",
                 merged.get("rank_skip_token")))
    return observations, failure


def _native_failure(observations: dict[str, Any], started: float, *,
                    kind: str, backend: str, lattice: Sequence[int], dtype: str,
                    grid: Sequence[int], declaration: Mapping[str, Any] | None,
                    data: Mapping[str, Any], layout: Mapping[str, Any],
                    settings: Mapping[str, Any]) -> _Failure | None:
    """Run the native case in this process and record its observations.

    The call consumes the row's own configuration (backend/lattice/dtype/grid,
    inputs, solver knobs), not a run-level default: a row must never compute
    something other than what its ledger entry says.  ``backend`` is the tag's
    backend (:func:`_row_backend`), so a ``cuda/native/solve`` row probes and
    runs CUDA - without it every row would fall through to
    ``native.DEFAULT_BACKEND`` and a CUDA-tagged row would silently execute the
    KUPL backend (audit N7).
    """
    kwargs: dict[str, Any] = {
        "backend": backend,
        "lattice": lattice,
        "dtype": dtype,
        "lattice_grid": grid,
        "seed": observations["seed"],
        "repeat": observations["repeat"],
        "warmup": observations["warmup"],
        "threads": observations.get("threads_requested"),
        "shm": observations.get("shm_requested"),
        "device": data.get("device"),
        "solver": observations.get("solver"),
        "dslash": observations.get("dslash"),
        "gauge": data.get("gauge"),
        "source": data.get("source"),
    }
    if kind == "mg":
        # A run-level (CLI) value wins; the L48 matrix declaration supplies the
        # reproducible default when the row has one.  The declaration must NOT
        # gate this: a plain `--native mg` row has `declaration is None`, and
        # gating it meant a sweep recorded the runner default 5 for every point
        # while coarse_tol_factor (not gated) arrived - the exact split seen on
        # 2026-11-24 in rows r10_c1e4..r20_c1e2.
        for name in ("mg_levels", "restart"):
            value = settings.get(name)
            if value is None and declaration is not None:
                value = declaration.get(name)
            if value is not None:
                kwargs[name] = value
        # The matrix declares neither of these, so settings is the only source;
        # an unset value stays unset so the runner keeps its own default.
        for name in ("mg_grid", "coarse_tol_factor"):
            if settings.get(name) is not None:
                kwargs[name] = settings[name]
    for name in ("mass", "kappa", "csw"):
        if observations.get(name) is not None:
            kwargs[name] = observations[name]
    if data.get("gauge_file") not in (None, ""):
        kwargs["gauge_file"] = str(data["gauge_file"])
    if data.get("source_file") not in (None, ""):
        kwargs["source_file"] = str(data["source_file"])
    cache_dir = layout.get("cache") or settings.get("cache_dir")
    if cache_dir and kind == "mg":
        # A ranked Legacy-MG artifact is a single HDF5 root with a sidecar
        # manifest and a `<root>.d/` rank directory; pass it as ``cache``.
        # A directory remains a cache *directory* for a future publish.
        candidate = Path(str(cache_dir))
        if candidate.is_file():
            kwargs["cache"] = str(candidate)
        else:
            kwargs["cache_directory"] = str(candidate)
    try:
        result = native.run_case(kind, **kwargs)
    except Skip as error:
        reason, token = _split_skip(error.reason, error.token)
        return ("SKIP", reason, token)
    except Exception as error:  # noqa: BLE001 - a row never escapes the ledger
        return ("ERROR",
                f"native {kind} failed: {type(error).__name__}: {error}", None)
    observations.update(result)
    observations["threads_requested"] = kwargs["threads"]
    observations["threads_applied"] = result.get("threads")
    observations["shm_applied"] = result.get("shm")
    observations["driver_seconds"] = round(time.perf_counter() - started, 6)
    if result.get("seconds") is not None:
        observations["seconds"] = result["seconds"]
    return None


def _finish(observations: dict[str, Any], started: float,
            failure: _Failure | None, *, world: int, rank: int
            ) -> dict[str, Any]:
    """Gather the row's outcome across ranks and return the observations.

    Exactly one collective per executed row, reached by every rank whether its
    own native call returned, skipped or raised.  The aggregated worst outcome
    becomes the declared execution status (only ``SKIP``/``ERROR``/``TIMEOUT``
    are declarable - a ``PASS`` is the harness's judgement), and the worst
    ``iterations``/``relative_residual``/``converged`` are merged so the gates
    judge the worst rank's evidence, not rank 0's.

    Returning only after the collective is what makes rank 0 safe to write
    ``rows/<run_id>.json``: it can only do so once every rank has reported.
    """
    merged = common.gather_rank_outcome(
        observations, size=world, rank=rank,
        status="PASS" if failure is None else failure[0],
        skip_token=None if failure is None else failure[2],
        reason=None if failure is None else failure[1],
        phase="outcome")
    for key in _GATHERED_KEYS:
        if key in merged:
            observations[key] = merged[key]
    for key in ("iterations", "relative_residual", "converged"):
        if merged.get(key) is not None:
            observations[key] = merged[key]
    worst = str(merged.get("rank_status") or "PASS")
    if worst != "PASS":
        return _failed(observations, started, worst,
                       merged.get("rank_reason") or f"rank verdict {worst}",
                       token=merged.get("rank_skip_token"))
    if failure is not None:
        # Only reachable without a communicator: the local failure is all the
        # evidence there is, and it must not be dropped.
        return _failed(observations, started, failure[0], failure[1],
                       token=failure[2])
    return observations


def _failed(observations: Mapping[str, Any], started: float, status: str,
            reason: str, *, token: str | None = None) -> dict[str, Any]:
    """One non-PASS execution observation (never a numeric verdict)."""
    payload = dict(observations)
    payload.update({
        "status": status,
        "reason": str(reason).strip() or f"driver reported {status} without a "
                                         "reason",
        "driver_seconds": round(time.perf_counter() - started, 6),
    })
    if token is not None:
        payload["skip_token"] = token
    return payload
