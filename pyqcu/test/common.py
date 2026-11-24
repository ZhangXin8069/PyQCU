"""Execution core of the formal :mod:`pyqcu.test` port (stdlib only).

Data model
----------
``Case``      declarative case: tag, runner string, lattice/dtype, ``expect``.
``Options``   the run configuration handed to a runner.
``Result``    one case's verdict, metrics and timing.
``Test``      a selected sequence of cases plus the run configuration.
``TestReport`` the aggregate, JSON-safe report.

The one rule that shapes everything else: **a runner reports observations and
never a verdict.**  ``Case.expect`` is evaluated after the call, by this
module, so a missing, mistyped or over-claimed observation cannot pass.  A
runner that raises is an ``ERROR`` (never a traceback out of the harness), a
``TimeoutError`` is a ``TIMEOUT`` and a skip must carry a non-empty reason.

Runner resolution is lazy
-------------------------
``Case.runner`` is ``"<module>:<function>"``.  A module part without a dot is
resolved inside :mod:`pyqcu.test` (``torch_qcu`` -> ``pyqcu.test.torch_qcu``)
with :func:`importlib.import_module`, at the moment the case runs.  Importing
the harness therefore never imports torch, numpy, mpi4py, pytest or the
informal ``pyqcu.testing`` tree.

Timing
------
``timed()`` discards ``warmup`` calls (>= 1: the first call carries plan
caches, native library binding and JIT) and returns the median of ``repeat``
samples.  A single sample is not a measurement, and a warmup-free measurement
is not a steady state.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import signal
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .. import __version__ as _PACKAGE_VERSION
from . import l48, policy

__all__ = [
    "CASE_MODE",
    "Case",
    "CONFIG_FIELDS",
    "DEFAULT_RUN_ROOT",
    "FULL_MODE_MODULES",
    "FULL_MODE_TIER",
    "FullModeOptions",
    "LATTICE_PRESETS",
    "MODES",
    "ModePaths",
    "Options",
    "RANK_RECORD_FIELDS",
    "RANK_RESIDUAL_RTOL",
    "RANK_STATUS_PRECEDENCE",
    "ROW_FIELDS",
    "Result",
    "Skip",
    "Test",
    "TestReport",
    "UsageError",
    "agreed_resume_row",
    "allgather_rank_records",
    "as_dict",
    "build_rows",
    "call_hook",
    "evaluate_expect",
    "GATE_OPERATORS",
    "gather_rank_outcome",
    "grid_product",
    "json_safe",
    "ledger_owner",
    "load_rows",
    "load_mode_module",
    "merge_ledger",
    "mode_hook",
    "module_availability",
    "mpi_info",
    "probe_env",
    "rank_outcome",
    "rank_record",
    "rank_status_key",
    "relative_error",
    "relative_residual",
    "resolve_runner",
    "row_id",
    "row_is_resumable",
    "run_case",
    "run_mode",
    "summarize",
    "time_samples",
    "timed",
    "validate_ledger",
    "validate_rows",
    "write_json_atomic",
    "write_ledger_json",
    "write_row",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class UsageError(ValueError):
    """Usage or environment error; the CLI maps it to exit code 2.

    A ``ValueError`` subclass so library callers can catch it with plain
    ``except ValueError``; the CLI catches it before the generic handler and
    exits ``policy.EXIT_CODES["usage"]``, keeping "the command is wrong"
    distinguishable from "the numbers are wrong" (exit 1).
    """

    exit_code = policy.EXIT_CODES["usage"]


class Skip(Exception):
    """Raise from a runner to report ``SKIP`` with a mandatory reason.

    ``token`` should be one of :data:`policy.SKIP_TOKENS`.  When the token is
    omitted it is recovered from a ``"token: reason"`` message prefix, so the
    common ``raise Skip("cuda_bridge: pyqcu.cuda.qcu is not built")`` form is
    enough.
    """

    def __init__(self, reason: str, token: str | None = None) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)
        self.token = token

    def __str__(self) -> str:
        return self.reason


# ---------------------------------------------------------------------------
# gate vocabulary
# ---------------------------------------------------------------------------
#: Binary comparison operators: ``actual <operator> threshold``.
_COMPARISONS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=",
                "gt": ">", "ge": ">=", "in": "in", "not_in": "not in"}

#: Operators whose right-hand side is another observation of the same runner.
_FIELD_COMPARISONS = {"eq_field": "eq", "ne_field": "ne", "lt_field": "lt",
                      "le_field": "le", "gt_field": "gt", "ge_field": "ge"}

#: Operators with no right-hand side at all.
_DIRECT_OPERATORS = ("truthy", "falsy", "empty", "nonempty")

#: Every operator ``Case.expect`` accepts; anything else is a definition bug.
GATE_OPERATORS: tuple[str, ...] = (
    tuple(_COMPARISONS) + tuple(_FIELD_COMPARISONS) + _DIRECT_OPERATORS
    + ("between",)
)

#: Observation keys with harness meaning; they never become ``Result.metrics``.
_RESERVED_OBSERVATIONS = frozenset({"skip", "reason", "status", "seconds", "token"})


def _format_value(value: Any) -> str:
    """Short, stable rendering of a gate value (floats in scientific form)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.3e}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
        return _truncate(repr(value), 80)
    return _truncate(repr(value), 80)


def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut ``text`` to ``limit`` characters."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 3, 0)] + "..."


def _threshold(raw: Any, dtype: str, atol: float | None) -> Any:
    """Resolve a gate threshold; a string names a :mod:`policy` tolerance."""
    if isinstance(raw, str):
        try:
            return policy.tolerance(dtype, raw, atol=atol)
        except KeyError as error:  # unknown tolerance name in a case definition
            raise ValueError(str(error)) from error
    return raw


def _resolve_observation(key: str,
                         observations: Mapping[str, Any]) -> tuple[Any, bool]:
    """Return ``(value, present)`` honouring :data:`policy.OBSERVATION_ALIASES`."""
    if key in observations:
        return observations[key], True
    for alias in policy.OBSERVATION_ALIASES.get(key, ()):
        if alias in observations:
            return observations[alias], True
    return None, False


def _evaluate_operator(key: str, operator: str, actual: Any, target: Any,
                       observations: Mapping[str, Any], dtype: str,
                       atol: float | None) -> tuple[bool, str]:
    """Evaluate one ``operator`` for ``key``; return ``(ok, failure-message)``."""
    if operator in _FIELD_COMPARISONS:
        if target not in observations and not any(
                alias in observations
                for alias in policy.OBSERVATION_ALIASES.get(str(target), ())):
            return False, f"{key}: missing comparison field {target!r}"
        field_value, _ = _resolve_observation(str(target), observations)
        return _evaluate_operator(key, _FIELD_COMPARISONS[operator], actual,
                                  field_value, observations, dtype, atol)

    if operator == "between":
        try:
            lower, upper = target
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"expect[{key!r}]['between'] must be [lower, upper], "
                f"got {target!r}") from error
        lower = _threshold(lower, dtype, atol)
        upper = _threshold(upper, dtype, atol)
        try:
            ok = bool(lower <= actual <= upper)
        except TypeError:
            ok = False
        return ok, (f"{key}={_format_value(actual)} outside "
                    f"[{_format_value(lower)}, {_format_value(upper)}]")

    if operator in ("truthy", "falsy"):
        ok = bool(actual) if operator == "truthy" else not bool(actual)
        want = "truthy" if operator == "truthy" else "falsy"
        return ok, f"{key}={_format_value(actual)} is not {want}"

    if operator in ("empty", "nonempty"):
        try:
            size = len(actual)
        except TypeError:
            return False, f"{key}: value of type {type(actual).__name__} has no length"
        ok = size == 0 if operator == "empty" else size > 0
        want = "empty" if operator == "empty" else "non-empty"
        return ok, f"{key} has {size} item(s), expected {want}"

    if operator in ("in", "not_in"):
        try:
            present = actual in target
        except TypeError:
            present = False
        ok = present if operator == "in" else not present
        return ok, (f"{key}={_format_value(actual)} "
                    f"{'not in' if operator == 'in' else 'in'} "
                    f"{_format_value(target)}")

    # remaining operators are binary comparisons
    symbol = _COMPARISONS[operator]
    value = target
    if operator not in ("eq", "ne"):
        value = _threshold(target, dtype, atol)
    if operator == "eq" and isinstance(value, bool):
        # ``True == 1`` in Python; a boolean gate must be a real boolean.
        ok = isinstance(actual, bool) and actual is value
        if not ok and isinstance(actual, bool):
            return False, (f"{key}={_format_value(actual)} {symbol} "
                           f"{_format_value(value)}")
        if not ok:
            return False, (f"{key}={_format_value(actual)} is not a boolean "
                           f"(expected {_format_value(value)})")
        return True, ""
    try:
        ok = bool(actual == value) if operator in ("eq", "ne") \
            else bool(_compare(operator, actual, value))
    except TypeError:
        return False, (f"{key}={_format_value(actual)} {symbol} "
                       f"{_format_value(value)} (incomparable types)")
    return ok, f"{key}={_format_value(actual)} {symbol} {_format_value(value)}"


def _compare(operator: str, actual: Any, value: Any) -> bool:
    """Apply an ordering comparison, returning ``False`` on type mismatch."""
    if operator == "lt":
        return actual < value
    if operator == "le":
        return actual <= value
    if operator == "gt":
        return actual > value
    if operator == "ge":
        return actual >= value
    raise ValueError(f"unknown comparison operator {operator!r}")


def evaluate_expect(expect: Mapping[str, Mapping[str, Any]],
                    observations: Mapping[str, Any], *, dtype: str,
                    atol: float | None = None) -> tuple[str, ...]:
    """Evaluate ``expect`` against runner ``observations``.

    Returns a tuple of human-readable failure strings; empty means PASS.  A
    missing observation is a failure (never a silent pass), and a malformed
    gate definition raises ``ValueError`` because that is a bug in
    ``units.py``, not a numerical result.
    """
    failed: list[str] = []
    for key, spec in expect.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"expect[{key!r}] must be a mapping, got {spec!r}")
        for operator in spec:
            if operator not in GATE_OPERATORS:
                raise ValueError(
                    f"expect[{key!r}] uses unknown operator {operator!r}; "
                    f"choose from {', '.join(GATE_OPERATORS)}")
        actual, present = _resolve_observation(key, observations)
        if not present:
            aliases = policy.OBSERVATION_ALIASES.get(key, ())
            hint = f" (aliases tried: {', '.join(aliases)})" if aliases else ""
            failed.append(f"{key}: missing observation{hint}")
            continue
        for operator, target in spec.items():
            ok, message = _evaluate_operator(
                key, operator, actual, target, observations, dtype, atol)
            if not ok:
                failed.append(message)
    return tuple(failed)


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------
def json_safe(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Recursively convert ``value`` into a JSON-serializable structure.

    Handles the mapping proxies and tuples used by the frozen dataclasses,
    dataclass instances, ``pathlib.Path``, ``complex`` (as ``[re, im]``) and
    duck-typed arrays (``.tolist()``, so numpy/torch values can be reported
    without importing either).  Non-finite floats become ``"NaN"``,
    ``"Infinity"`` and ``"-Infinity"`` strings: the JSON standard has no NaN
    and a silent ``null`` would erase the fact that the value was not finite.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")

    if _seen is None:
        _seen = set()
    identity = id(value)

    if isinstance(value, Mapping):
        if identity in _seen:
            return "<cycle>"
        _seen.add(identity)
        try:
            return {str(key): json_safe(item, _seen=_seen)
                    for key, item in value.items()}
        finally:
            _seen.discard(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        if identity in _seen:
            return "<cycle>"
        _seen.add(identity)
        try:
            return [json_safe(item, _seen=_seen) for item in value]
        finally:
            _seen.discard(identity)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Field-wise instead of ``dataclasses.asdict``: the frozen dataclasses
        # hold ``MappingProxyType`` values that ``deepcopy`` refuses.
        return {item.name: json_safe(getattr(value, item.name), _seen=_seen)
                for item in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist(), _seen=_seen)
        except (TypeError, ValueError, RuntimeError):
            pass
    return _truncate(repr(value), 200)


def as_dict(value: Any) -> Any:
    """Return a JSON-safe ``dict`` for any report object.

    Uses the object's own ``as_dict()`` when it has one (``Case``, ``Result``,
    ``TestReport``) and falls back to :func:`json_safe`.
    """
    method = getattr(value, "as_dict", None)
    if callable(method):
        return method()
    return json_safe(value)


def write_json_atomic(path: str | os.PathLike[str], payload: Any,
                      *, indent: int = 2) -> Path:
    """Write ``payload`` as JSON to ``path`` atomically.

    The temporary file lives in the destination directory (so the final
    ``os.replace`` is a rename on one filesystem) and is flushed before the
    replace: a reader either sees the previous report or the complete new one,
    never a half-written file.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(payload), indent=indent, sort_keys=True,
                      ensure_ascii=False, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


# ---------------------------------------------------------------------------
# timing and vector helpers
# ---------------------------------------------------------------------------
def time_samples(call: Callable[[], Any], *,
                 repeat: int = policy.DEFAULT_REPEAT,
                 warmup: int = policy.DEFAULT_WARMUP) -> tuple[float, ...]:
    """Time ``call``: discard ``warmup`` calls, return ``repeat`` samples (s).

    ``warmup`` must be >= 1.  The first call of an operator carries plan
    caches, native library binding and JIT; measuring it as the steady state
    is the classic 80x-overestimate mistake this port exists to avoid.
    """
    if int(repeat) < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat!r}")
    if int(warmup) < 1:
        raise ValueError(
            f"warmup must be >= 1, got {warmup!r} "
            "(the first call is not a steady state)")
    for _ in range(int(warmup)):
        call()
    samples: list[float] = []
    for _ in range(int(repeat)):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return tuple(samples)


def timed(call: Callable[[], Any], *,
          repeat: int = policy.DEFAULT_REPEAT,
          warmup: int = policy.DEFAULT_WARMUP) -> float:
    """Median seconds of ``repeat`` samples after ``warmup`` discarded calls."""
    return float(statistics.median(
        time_samples(call, repeat=repeat, warmup=warmup)))


def _flat_values(value: Any) -> tuple[Any, ...]:
    """Flatten nested sequences (and duck-typed arrays) into a tuple."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (list, tuple)):
        try:
            value = tolist()
        except (TypeError, ValueError, RuntimeError):
            pass
    values: list[Any] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            values.append(item)
    return tuple(values)


def relative_error(actual: Any, expected: Any) -> float:
    """``max|actual - expected| / max|expected|`` (``inf`` on shape mismatch).

    Normalizing by the largest reference magnitude is the repository-wide
    convention (``pyqcd.einsum.test``, the benchmark scripts).  A per-element
    absolute tolerance is not equivalent: it misjudges near-zero elements.
    """
    left = _flat_values(actual)
    right = _flat_values(expected)
    if len(left) != len(right):
        return math.inf
    if not right:
        return 0.0
    scale = max(abs(item) for item in right)
    if scale == 0.0:
        scale = 1.0
    difference = max(abs(a - b) for a, b in zip(left, right))
    return float(difference / scale)


def _norm(vector: Any) -> float:
    """Euclidean norm of a duck-typed vector without importing numpy/torch."""
    norm = getattr(vector, "norm", None)
    if callable(norm):
        try:
            return float(norm())
        except (TypeError, ValueError, RuntimeError, AttributeError):
            pass
    absolute = getattr(vector, "abs", None)
    total = getattr(vector, "sum", None)
    if callable(absolute) and callable(total):
        try:
            return float((absolute() ** 2).sum() ** 0.5)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            pass
    values = _flat_values(vector)
    return math.sqrt(math.fsum(abs(item) ** 2 for item in values))


def relative_residual(matvec: Callable[[Any], Any], x: Any, b: Any) -> float:
    """Independent ``||matvec(x) - b|| / ||b||``.

    The harness never trusts a solver's own residual recurrence: a solver can
    converge in its internal (preconditioned) norm and still be wrong in the
    full Wilson norm.  Returns ``inf`` when ``b`` is zero (nothing to
    normalize) so a degenerate case fails loudly instead of passing at 0/0.
    """
    denominator = _norm(b)
    numerator = _norm(matvec(x) - b)
    if denominator == 0.0:
        return math.inf if numerator != 0.0 else 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# process/environment probes (no heavy imports)
# ---------------------------------------------------------------------------
_RANK_VARIABLES = ("PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
                   "CCS_TASK_INDEX", "I_MPI_RANK", "MV2_COMM_WORLD_RANK",
                   "SLURM_PROCID")
_SIZE_VARIABLES = ("CCS_TASK_REPLICA", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE",
                   "PMI_SIZE", "I_MPI_SIZE",
                   "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")


def _first_int_environment(names: Iterable[str]) -> int | None:
    """Return the first integer launcher variable that is set, else None."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def mpi_info() -> tuple[int, int]:
    """Return ``(size, rank)`` from launcher variables, without mpi4py.

    Reading the launcher environment keeps the harness importable in an
    interpreter that has no MPI stack and avoids initializing MPI as a side
    effect of ``--list``.
    """
    rank = _first_int_environment(_RANK_VARIABLES)
    size = _first_int_environment(_SIZE_VARIABLES)
    if rank is None:
        rank = 0
    if size is None:
        size = 1 if rank == 0 else rank + 1
    return size, rank


def _spec_available(name: str) -> bool:
    """True when a top-level module can be imported (no import performed)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def probe_env() -> dict[str, Any]:
    """Environment summary for the report header; never imports a backend.

    Only top-level specs are probed: ``find_spec`` on a submodule would import
    its parent package, which is exactly the side effect the port avoids.
    """
    size, rank = mpi_info()
    names = ("torch", "numpy", "mpi4py", "h5py")
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "mpi_size": size,
        "mpi_rank": rank,
        "modules": {name: _spec_available(name) for name in names},
        "loaded": sorted(name for name in names if name in sys.modules),
    }


# ---------------------------------------------------------------------------
# multi-rank ledger protocol: N processes, one logical row
# ---------------------------------------------------------------------------
#: Worst-first status order of one logical row.  A row executed by several MPI
#: ranks carries the **worst** status any rank reported, never rank 0's alone:
#: one rank's ``ERROR`` must not disappear behind another rank's ``PASS``.
RANK_STATUS_PRECEDENCE: tuple[str, ...] = ("ERROR", "FAIL", "TIMEOUT", "SKIP",
                                           "PASS")

#: Fields of the compact record one row's allgather exchanges (7 values/rank).
#: The full observation mapping is deliberately *not* shipped between ranks: it
#: carries input identity and byte counts that would make the collective the
#: slowest part of a 32-rank L48 row.
RANK_RECORD_FIELDS: tuple[str, ...] = (
    "rank", "status", "skip_token", "reason", "iterations",
    "relative_residual", "converged")

#: Relative spread under which two ranks' residuals count as the same number.
#: A distributed solve reduces its inner products globally, so ranks differ by
#: rounding only; a wider spread is a real disagreement and is recorded.
RANK_RESIDUAL_RTOL = 1e-9

#: Cache of ``size -> (communicator, error)`` for :func:`_rank_communicator`.
#: ``mpi4py`` is imported inside that function (the port stays stdlib-only at
#: import time) and only once per world size, so a single-rank run never pays
#: for - or initializes - MPI.
_RANK_COMMUNICATORS: dict[int, tuple[Any, str | None]] = {}


def _rank_int(value: Any) -> int | None:
    """``int(value)`` for a real integer, ``None`` for anything else."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rank_float(value: Any) -> float | None:
    """Finite-or-infinite ``float(value)``; ``None`` when unusable or NaN.

    ``inf`` is kept on purpose: an infinite residual is the *worst* possible
    evidence and must survive into the aggregate, while a ``NaN`` cannot be
    ordered and is dropped (its rank's gate would fail on its own anyway).
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _rank_bool(value: Any) -> bool | None:
    """``value`` when it is a real boolean, ``None`` otherwise."""
    return value if isinstance(value, bool) else None


def rank_status_key(status: Any) -> int:
    """Sort key of a status: lower is worse (``ERROR`` 0 ... ``PASS`` 4).

    An unknown status sorts *before* ``ERROR``: a record that cannot be parsed
    must never win over a real verdict, and :func:`rank_outcome` turns it into
    an ``ERROR`` instead of trusting it.
    """
    text = str(status if status is not None else "").strip().upper()
    try:
        return RANK_STATUS_PRECEDENCE.index(text)
    except ValueError:
        return -1


def ledger_owner(*, rank: int | None = None) -> bool:
    """True on the one process allowed to write the ledger and the report.

    ``rows/<run_id>.json``, ``reports/<run_id>.json``, the append-only CSV, the
    solution copies and the ``--json``/``--outdir`` payload have exactly one
    writer: rank 0 of the MPI world (:func:`mpi_info`, which reads the launcher
    environment without importing mpi4py).  Every other rank computes, reports
    through :func:`gather_rank_outcome` and writes nothing, so no two processes
    can interleave a JSON document or append the same CSV concurrently.

    With one process - the default, and every non-MPI invocation - this is
    always true, so single-rank behaviour is unchanged.
    """
    world_rank = mpi_info()[1] if rank is None else int(rank)
    return int(world_rank) == 0


def write_ledger_json(path: Any, payload: Any,
                      *, owner: bool | None = None) -> Path | None:
    """Write one ledger document from the owning rank only.

    A replica rank returns ``None`` instead of writing: the row/report/CSV
    layout is single-writer (see :func:`ledger_owner`), so ``--mode prepare``,
    ``prewarm``, ``merge`` and the per-row reports cannot be written by 32
    processes at once.  ``owner`` overrides the probe for tests and library
    callers.
    """
    if not (ledger_owner() if owner is None else bool(owner)):
        return None
    return write_json_atomic(path, payload)


def rank_record(rank: int, observations: Mapping[str, Any] | None = None, *,
                status: Any = None, skip_token: Any = None,
                reason: Any = None) -> dict[str, Any]:
    """One rank's compact record of a row (see :data:`RANK_RECORD_FIELDS`).

    ``status``/``skip_token``/``reason`` override what ``observations`` says,
    which is what the driver's pre-flight needs ("this rank *can* run" is a
    decision, not an observation).  An unparsable status becomes ``ERROR``: a
    record the aggregator cannot read must fail the row, never pass it.
    """
    observation = observations if isinstance(observations, Mapping) else {}
    declared = status if status is not None else observation.get("status")
    text = str(declared).strip().upper() if declared is not None else "PASS"
    detail = reason if reason is not None else observation.get("reason")
    if text not in policy.STATUSES:
        detail = f"rank {int(rank)} reported the unknown status {declared!r}"
        text = "ERROR"
    token = skip_token if skip_token is not None \
        else observation.get("skip_token")
    iterations = _rank_int(_resolve_observation("iterations", observation)[0])
    residual = _rank_float(
        _resolve_observation("relative_residual", observation)[0])
    converged = _rank_bool(
        _resolve_observation("converged", observation)[0])
    return {
        "rank": int(rank),
        "status": text,
        "skip_token": str(token) if token else None,
        "reason": None if text == "PASS" else (str(detail or "").strip() or None),
        "iterations": iterations,
        "relative_residual": residual,
        "converged": converged,
    }


def rank_outcome(records: Sequence[Mapping[str, Any]],
                 *, size: int | None = None) -> dict[str, Any]:
    """Aggregate the per-rank records of one row into its single outcome.

    The status is the worst one any rank reported
    (:data:`RANK_STATUS_PRECEDENCE`), so a single rank's ``ERROR`` fails the row
    even when every other rank passed; the numbers are the worst *evidence*
    (largest iteration count, largest independent residual, ``converged`` only
    when every rank converged), so a rank whose solve did not converge cannot be
    outvoted either.  ``disagreement`` records whether the ranks agreed at all,
    which is what lets a reader tell "all 32 ranks passed" from "the worst rank
    passed while another one reported something else".

    An empty record list is an ``ERROR``: "no rank reported" is not a pass.
    """
    current = [dict(item) for item in (records or ())
               if isinstance(item, Mapping)]
    total = len(current) if size is None else int(size)
    if not current:
        return {
            "status": "ERROR",
            "skip_token": None,
            "reason": "no rank reported an outcome for this row",
            "iterations": None,
            "relative_residual": None,
            "converged": None,
            "worst_rank": None,
            "disagreement": False,
            "records": [],
        }
    ordered = sorted(
        current, key=lambda item: (rank_status_key(item.get("status")),
                                   _rank_int(item.get("rank")) or 0))
    worst = ordered[0]
    status = str(worst.get("status") or "").strip().upper()
    if status not in policy.STATUSES:
        status = "ERROR"
    worst_rank = _rank_int(worst.get("rank"))
    reason = str(worst.get("reason") or "").strip() or None
    if status == "SKIP" and reason is None:
        # A SKIP without a reason is the documented ERROR, and it must not look
        # like a clean skip just because it lost the aggregation.
        status = "ERROR"
    if status == "PASS" and reason is None:
        reason = None
    elif reason is None:
        reason = f"rank {worst_rank} reported {status} without a reason"
    if reason is not None and len(current) > 1:
        # Only a genuine comparison is labelled with the rank it came from: a
        # rank-local fallback (no communicator) must not look like "rank 0/32".
        reason = f"rank {worst_rank}/{total}: {reason}"

    iterations = [value for value in
                  (_rank_int(item.get("iterations")) for item in current)
                  if value is not None]
    residuals = [value for value in
                 (_rank_float(item.get("relative_residual"))
                  for item in current) if value is not None]
    converged = [item["converged"] for item in current
                 if isinstance(item.get("converged"), bool)]
    statuses = {str(item.get("status") or "").strip().upper()
                for item in current}
    disagreement = len(statuses) > 1
    if not disagreement and len(set(iterations)) > 1:
        disagreement = True
    if not disagreement and len(set(converged)) > 1:
        disagreement = True
    if not disagreement and len(residuals) > 1:
        disagreement = not math.isclose(
            min(residuals), max(residuals),
            rel_tol=RANK_RESIDUAL_RTOL, abs_tol=0.0)
    return {
        "status": status,
        "skip_token": (str(worst.get("skip_token"))
                       if status == "SKIP" and worst.get("skip_token") else None),
        "reason": None if status == "PASS" else reason,
        "iterations": max(iterations) if iterations else None,
        "relative_residual": max(residuals) if residuals else None,
        "converged": all(converged) if converged else None,
        "worst_rank": worst_rank,
        "disagreement": bool(disagreement),
        "records": current,
    }


def _rank_communicator(size: int, rank: int) -> tuple[Any, str | None]:
    """``MPI.COMM_WORLD`` when this process really is one of ``size`` ranks.

    Returns ``(communicator, None)`` or ``(None, reason)``.  mpi4py is resolved
    through :func:`importlib.import_module` **inside** this function and only
    when the launcher environment claims more than one rank, so a single-rank
    run never pays for, nor initializes, MPI; the harness module graph stays
    stdlib-only (``selfcheck.source_imports`` forbids an ``import mpi4py``
    statement here, and a missing mpi4py is a reported limitation, never a
    hard dependency).  A world whose size contradicts the launcher probe is
    refused instead of gathered on: two different worlds must not silently
    look like agreement.
    """
    if int(size) <= 1:
        return None, None
    cached = _RANK_COMMUNICATORS.get(int(size))
    if cached is not None:
        return cached
    try:
        MPI = importlib.import_module("mpi4py").MPI
    except Exception as error:  # noqa: BLE001 - reported, never propagated
        result: tuple[Any, str | None] = (
            None, f"mpi4py unavailable: {type(error).__name__}: {error}")
        _RANK_COMMUNICATORS[int(size)] = result
        return result
    try:
        communicator = MPI.COMM_WORLD
        found = int(communicator.Get_size())
    except Exception as error:  # noqa: BLE001 - reported
        result = (None, "MPI.COMM_WORLD unavailable: "
                        f"{type(error).__name__}: {error}")
        _RANK_COMMUNICATORS[int(size)] = result
        return result
    if found != int(size):
        result = (None,
                  f"MPI_COMM_WORLD has {found} rank(s) but the launcher "
                  f"environment reports {int(size)}; refusing to treat them "
                  "as one world")
        _RANK_COMMUNICATORS[int(size)] = result
        return result
    result = (communicator, None)
    _RANK_COMMUNICATORS[int(size)] = result
    return result


def allgather_rank_records(record: Mapping[str, Any], *,
                           size: int | None = None, rank: int | None = None,
                           comm: Any = None
                           ) -> tuple[list[dict[str, Any]], str, str | None]:
    """Exchange one compact rank record; return ``(records, mode, reason)``.

    Exactly one collective per call and no retry: the caller enters it at the
    same point of every rank's execution of the same row, which is what keeps
    the protocol deadlock-free.  ``mode`` is

    ``allgather``    a real collective; ``records`` holds one entry per rank
    ``single_rank``  one process; there is nothing to exchange
    ``unavailable``  several ranks but no usable communicator - ``reason`` says
                     why, and the caller must record it instead of claiming the
                     ranks agreed (a silent single-rank fallback would turn 32
                     processes into 32 independent, unreviewable verdicts)
    """
    world, world_rank = mpi_info()
    total = int(world if size is None else size)
    here = int(world_rank if rank is None else rank)
    if total <= 1:
        return [dict(record)], "single_rank", None
    communicator = comm
    if communicator is None:
        communicator, failure = _rank_communicator(total, here)
        if communicator is None:
            return [dict(record)], "unavailable", failure
    try:
        gathered = list(communicator.allgather(dict(record)))
    except Exception as error:  # noqa: BLE001 - reported, never propagated
        return ([dict(record)], "unavailable",
                f"allgather failed: {type(error).__name__}: {error}")
    records = [dict(item) for item in gathered if isinstance(item, Mapping)]
    if len(records) != total:
        return ([dict(record)], "unavailable",
                f"allgather returned {len(records)} record(s) for {total} "
                "rank(s); the ranks cannot be compared")
    records.sort(key=lambda item: _rank_int(item.get("rank")) or 0)
    return records, "allgather", None


def gather_rank_outcome(observations: Mapping[str, Any] | None = None, *,
                        size: int | None = None, rank: int | None = None,
                        status: Any = None, skip_token: Any = None,
                        reason: Any = None, comm: Any = None,
                        phase: str = "row") -> dict[str, Any]:
    """Allgather this rank's outcome of one row and return the row evidence.

    This is the one place the port collects across ranks, and every rank calls
    it exactly once per row: :func:`pyqcu.test.driver.run_row` after its native
    call and the full-mode ``solve`` loop after each row.  Returning only after
    the collective is what makes rank 0 safe to write ``rows/<run_id>.json``,
    ``reports/<run_id>.json`` and the CSV - it can only do so once every rank
    has reported - and what keeps ``--fail-fast`` from deadlocking: the gathered
    outcome is identical on every rank, so every rank stops at the same row.

    The returned mapping is JSON-safe and carries ``rank_disagreement``,
    ``mpi_rank``/``mpi_size``, ``rank_gather``, the worst status/reason/token
    and the worst ``iterations``/``relative_residual``/``converged``.
    ``status``/``reason``/``skip_token`` are *not* renamed onto the observation
    keys here: the driver and the ledger translate them explicitly, so a
    gathered ``PASS`` can never be mistaken for a declared verdict.
    """
    world, world_rank = mpi_info()
    total = int(world if size is None else size)
    here = int(world_rank if rank is None else rank)
    record = rank_record(here, observations, status=status,
                         skip_token=skip_token, reason=reason)
    records, mode, failure = allgather_rank_records(
        record, size=total, rank=here, comm=comm)
    outcome = rank_outcome(records, size=total)
    evidence: dict[str, Any] = {
        "rank_phase": str(phase),
        "rank_gather": mode,
        "rank_disagreement": bool(outcome["disagreement"]),
        "rank_status": str(outcome["status"]),
        "rank_worst": outcome["worst_rank"],
        "mpi_rank": here,
        "mpi_size": total,
        "rank_outcomes": records,
    }
    if failure:
        evidence["rank_gather_reason"] = str(failure)
    if outcome["status"] != "PASS":
        evidence["rank_reason"] = outcome["reason"]
        evidence["rank_skip_token"] = outcome["skip_token"]
    for key in ("iterations", "relative_residual", "converged"):
        if outcome[key] is not None:
            evidence[key] = outcome[key]
    return evidence


# ---------------------------------------------------------------------------
# runner resolution and invocation
# ---------------------------------------------------------------------------
_RUNNERS: dict[str, Callable[..., Any]] = {}
_OPTIONS_NAMES = ("options", "ctx", "context", "config")


def resolve_runner(runner: str) -> Callable[..., Any]:
    """Resolve ``"module:function"`` lazily, caching the imported callable.

    A module part without a dot is relative to :mod:`pyqcu.test`; a dotted
    path is imported as-is.  Resolution errors carry the runner string so an
    ``ERROR`` line names the missing module or attribute.
    """
    if not isinstance(runner, str) or ":" not in runner:
        raise ValueError(
            f"runner must be a 'module:function' string, got {runner!r}")
    cached = _RUNNERS.get(runner)
    if cached is not None:
        return cached
    module_name, _, attribute = runner.partition(":")
    module_name = module_name.strip()
    attribute = attribute.strip()
    if not module_name or not attribute:
        raise ValueError(
            f"runner must be a 'module:function' string, got {runner!r}")
    if "." not in module_name:
        module_name = f"{policy.RUNNER_PREFIX}.{module_name}"
    module = importlib.import_module(module_name)
    try:
        function = getattr(module, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"{module_name!r} has no runner {attribute!r}") from error
    if not callable(function):
        raise TypeError(f"{runner!r} resolves to a non-callable "
                        f"{type(function).__name__}")
    _RUNNERS[runner] = function
    return function


def _invoke_runner(function: Callable[..., Any], options: "Options") -> Any:
    """Call a runner with only the arguments it declares.

    This keeps the runner interface small and honest: a runner takes the run
    configuration it needs (``lattice``, ``lattice_grid``, ``mg_grid``,
    ``dtype``, ``seed``, ``repeat``, ``warmup``, ``atol``, the source/gauge
    selection and the solver knobs), optionally the frozen ``options`` object,
    and nothing else.  Passing exactly the declared keyword arguments means a
    simple ``def run_lattice(lattice, dtype, seed, repeat, warmup, atol)``
    works unchanged next to ``def run_...(options)``.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(options)
    parameters = signature.parameters
    kwargs: dict[str, Any] = dict(options.as_kwargs())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD
           for parameter in parameters.values()):
        kwargs["options"] = options
        return function(**kwargs)
    for name in _OPTIONS_NAMES:
        if name in parameters:
            kwargs[name] = options
            break
    selected = {
        name: value for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**selected)


class _Timer:
    """Bookkeeping for an armed ``SIGALRM`` case timeout."""

    __slots__ = ("previous",)

    def __init__(self, previous: Any) -> None:
        self.previous = previous


_SIGALRM = getattr(signal, "SIGALRM", None)


def _on_timeout(signum: int, frame: Any) -> None:  # pragma: no cover - signal
    raise TimeoutError("case exceeded its wall-clock budget")


def _arm_timeout(timeout: float | None) -> _Timer | None:
    """Arm a per-case wall-clock timer when the platform supports one.

    ``SIGALRM`` only exists on POSIX and only fires on the main thread; when
    it is unavailable the caller falls back to a post-hoc elapsed check.
    """
    if timeout is None or float(timeout) <= 0.0:
        return None
    if _SIGALRM is None or not hasattr(signal, "setitimer"):
        return None
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        previous = signal.getsignal(_SIGALRM)
        signal.signal(_SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    except (ValueError, OSError, RuntimeError):
        return None
    return _Timer(previous)


def _disarm_timeout(timer: _Timer | None) -> None:
    """Cancel a timer armed by :func:`_arm_timeout` and restore the handler."""
    if timer is None:
        return
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    except (ValueError, OSError):
        pass
    try:
        signal.signal(_SIGALRM, timer.previous)
    except (ValueError, OSError, TypeError):
        pass


# ---------------------------------------------------------------------------
# run-configuration normalisation (shared by Options and Test)
# ---------------------------------------------------------------------------
#: Configuration fields :class:`Options` and :class:`Test` have in common, in
#: ``--list --json`` / report order.  ``Test.config()`` emits them; ``run()``
#: copies them into every :class:`Options`.
CONFIG_FIELDS: tuple[str, ...] = (
    "lattice", "lattice_grid", "mg_grid", "dtype", "seed", "source", "gauge",
    "gauge_sigma", "mass", "kappa", "csw", "solver", "dslash", "device",
    "max_iter", "restart", "mg_levels", "mg_nu_pre", "mg_nu_post", "threads",
    "device_id", "atol", "coarse_tol_factor", "reference_budget",
)


def _finite_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite float (booleans are not numbers here)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite number, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _optional_float(name: str, value: Any, *, minimum: float | None = None,
                    exclusive: bool = False) -> float | None:
    """Validate an optional float, honouring a lower bound."""
    if value is None:
        return None
    number = _finite_float(name, value)
    if minimum is not None:
        too_small = number <= minimum if exclusive else number < minimum
        if too_small:
            comparison = ">" if exclusive else ">="
            raise ValueError(
                f"{name} must be {comparison} {minimum:g}, got {value!r}")
    return number


def _optional_dof(value: Any) -> tuple[int, ...] | None:
    """Normalise the Legacy-MG per-level component counts (``--mg-dof``).

    ``dof[0]`` is the 12-component fine odd-Schur field and ``dof[level]`` the
    coarse basis of that level, so ``dof[1]`` is ``E``: the coarse basis
    dimension that reaches the solver as ``params[kMgLevel1E]`` (19) and that
    sets the coarse space's capacity, ``capture_bound = sqrt(E / (e * block))``
    (cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:1019-1023).  ``None`` means
    "not given" and stays ``None`` so the runner keeps its own default and, for
    a ranked Legacy-MG cache, adopts the ``E`` the artifact was built with.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"mg_dof must be a list of per-level component counts, got "
            f"{value!r}")
    values = tuple(_optional_int("mg_dof", item) for item in value)
    if len(values) < 2 or values[0] != 12:
        raise ValueError(
            "mg_dof must start with the 12-component fine field and list at "
            f"least two levels, got {list(values)}")
    return values


def _optional_int(name: str, value: Any, *, minimum: int = 1) -> int | None:
    """Validate an optional integer (``int`` or an integer string only)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        number = int(value.strip())
    else:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def _validate_device(value: Any) -> str:
    """Validate a ``--device``/``Test(device=...)`` spelling.

    The vocabulary is ``auto``/``cpu``/``gpu``/``dcu``/``npu``, but a concrete
    target such as ``cuda:3`` is documented (``policy.device_for_runner``
    passes it through).  The ``device`` field keeps the user spelling and
    :meth:`Test.config` reports the runner-facing string separately, so both
    forms must validate here instead of only the five words.
    """
    text = str(value).strip().lower()
    if not text:
        raise ValueError("device must not be empty")
    if text in policy.DEVICE_CHOICES:
        return text
    policy.device_for_runner(text)  # raises ValueError for an unusable string
    return text


def _normalize_run_fields(instance: Any) -> None:
    """Normalise the shared configuration fields of ``Options``/``Test``.

    ``instance`` is a frozen dataclass instance; the values are written with
    ``object.__setattr__``.  Keeping one normaliser is what makes the CLI, the
    ``Test`` API, the forwarded kwargs and the JSON report agree on every
    spelling (``complex128`` -> ``c128``, ``"2,2,2,2"`` -> ``(2, 2, 2, 2)``).
    """
    object.__setattr__(instance, "lattice",
                       policy.validate_lattice(instance.lattice))
    object.__setattr__(instance, "lattice_grid",
                       policy.validate_lattice_grid(instance.lattice_grid))
    object.__setattr__(instance, "mg_grid",
                       policy.validate_mg_grid(instance.mg_grid))
    object.__setattr__(instance, "dtype", policy.validate_dtype(instance.dtype))
    object.__setattr__(instance, "source",
                       policy.validate_source(instance.source))
    object.__setattr__(instance, "gauge", policy.validate_gauge(instance.gauge))
    object.__setattr__(instance, "gauge_sigma",
                       _optional_float("gauge_sigma", instance.gauge_sigma,
                                       minimum=0.0))
    object.__setattr__(instance, "csw",
                       _optional_float("csw", instance.csw, minimum=0.0))
    object.__setattr__(instance, "mass",
                       _optional_float("mass", instance.mass))
    object.__setattr__(instance, "kappa",
                       _optional_float("kappa", instance.kappa, minimum=0.0,
                                       exclusive=True))
    object.__setattr__(instance, "solver",
                       policy.validate_solver(instance.solver))
    object.__setattr__(instance, "dslash",
                       policy.validate_dslash(instance.dslash))
    for name in ("max_iter", "restart", "mg_levels", "mg_nu_pre",
                 "mg_nu_post", "threads"):
        object.__setattr__(instance, name,
                           _optional_int(name, getattr(instance, name)))
    object.__setattr__(instance, "device_id",
                       _optional_int("device_id", instance.device_id,
                                     minimum=0))
    if hasattr(instance, "mg_dof"):
        object.__setattr__(instance, "mg_dof",
                           _optional_dof(getattr(instance, "mg_dof")))


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
def _freeze(value: Any) -> Any:
    """Deep-freeze mappings/sequences so a ``Case`` is truly immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class Options:
    """Run configuration handed to a runner (immutable, normalised).

    Every field is normalised once, here: grids become four-tuples of positive
    integers, ``dtype`` a canonical ``c64``/``c128``, ``solver``/``dslash``/
    ``source``/``gauge`` a declared choice, and ``device`` the *runner-facing*
    string (``None`` for ``--device auto``, so a backend keeps its own default
    instead of receiving a word like ``"auto"`` it cannot parse).

    ``as_kwargs()`` forwards only fields that have a value: an unset ``mass``,
    ``kappa``, ``max_iter``... must not reach a runner as ``None``, where it
    would override the runner's own default with a value it cannot use.
    """

    lattice: tuple[int, int, int, int] = policy.DEFAULT_LATTICE
    lattice_grid: tuple[int, int, int, int] = policy.DEFAULT_LATTICE_GRID
    mg_grid: tuple[int, int, int, int] = policy.DEFAULT_MG_GRID
    dtype: str = policy.DEFAULT_DTYPE
    seed: int = policy.DEFAULT_SEED
    repeat: int = policy.DEFAULT_REPEAT
    warmup: int = policy.DEFAULT_WARMUP
    atol: float | None = None
    timeout: float | None = None
    source: str = policy.DEFAULT_SOURCE
    gauge: str = policy.DEFAULT_GAUGE
    gauge_sigma: float = policy.DEFAULT_GAUGE_SIGMA
    mass: float | None = None
    kappa: float | None = None
    csw: float = policy.DEFAULT_CSW
    solver: str = policy.DEFAULT_SOLVER
    dslash: str = policy.DEFAULT_DSLASH
    device: str | None = None
    max_iter: int | None = None
    restart: int | None = None
    mg_levels: int | None = None
    mg_nu_pre: int | None = None
    mg_nu_post: int | None = None
    # Per-level component counts of the Legacy-MG hierarchy, ``(12, E)``: the
    # 12-component fine odd-Schur field and the coarse basis dimension ``E`` of
    # level 1.  ``E`` is the capacity of the coarse space
    # (``capture_bound = sqrt(E / (e * block))``,
    # cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:1019-1023) and it reaches the
    # solver through ``params[kMgLevel1E]`` (=19), which this port copies out of
    # the coarse ``lonv`` tensor's first axis
    # (``native._configure_legacy_mg``).  ``None`` means "not given": the runner
    # then keeps its own default and, for a ranked Legacy-MG cache, adopts the
    # ``E`` the artifact was built with.
    mg_dof: tuple[int, ...] | None = None
    threads: int | None = None
    device_id: int | None = None
    # Legacy-MG coarse-level tolerance multiplier.  A runner option, not a
    # cross-runner field: without this the value only lived on the Case and
    # the harness silently ran the default (2026-11-24, six sweep points
    # all recorded 1e4 while the CLI asked for 1e2 and 1e1).
    coarse_tol_factor: float | None = None
    # Per-rank workspace budget of the independent residual reference, in
    # bytes.  ``None`` keeps the native default
    # (``native.REFERENCE_BUDGET_BYTES`` = 4 GiB); a value is what the
    # reference preflight plans against, so a field that needs more than the
    # default must be able to declare it or the row can only SKIP with
    # ``reference_workspace`` (2026-11-24, L64 16-rank cells: 7.50 GiB).
    reference_budget: int | None = None

    def __post_init__(self) -> None:
        _normalize_run_fields(self)
        object.__setattr__(self, "device",
                           policy.device_for_runner(self.device))
        if int(self.repeat) < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat!r}")
        if int(self.warmup) < 1:
            raise ValueError(f"warmup must be >= 1, got {self.warmup!r}")
        if self.atol is not None and float(self.atol) <= 0.0:
            raise ValueError(f"atol must be > 0, got {self.atol!r}")
        if self.reference_budget is not None \
                and int(self.reference_budget) <= 0:
            raise ValueError(
                "reference_budget must be a positive byte count, got "
                f"{self.reference_budget!r}")
        if self.timeout is not None and float(self.timeout) <= 0.0:
            raise ValueError(f"timeout must be > 0, got {self.timeout!r}")

    def as_kwargs(self) -> dict[str, Any]:
        """Keyword arguments advertised to runners (never ``timeout``).

        The always-present keys are the ones with a policy default (lattice,
        grids, dtype, seed, timing, source/gauge, gauge width, ``csw`` and the
        solver/dslash selection).  Everything the user did *not* set is
        omitted, so a runner's own default applies instead of receiving
        ``None`` and failing inside a native call.
        """
        kwargs: dict[str, Any] = {
            "lattice": self.lattice,
            "lattice_grid": self.lattice_grid,
            "mg_grid": self.mg_grid,
            "dtype": self.dtype,
            "seed": int(self.seed),
            "repeat": int(self.repeat),
            "warmup": int(self.warmup),
            "source": self.source,
            "gauge": self.gauge,
            "gauge_sigma": float(self.gauge_sigma),
            "csw": float(self.csw),
            "solver": self.solver,
            "dslash": self.dslash,
        }
        optional: dict[str, Any] = {
            "atol": self.atol,
            "mass": self.mass,
            "kappa": self.kappa,
            "device": self.device,
            "max_iter": self.max_iter,
            "restart": self.restart,
            "mg_levels": self.mg_levels,
            "mg_nu_pre": self.mg_nu_pre,
            "mg_nu_post": self.mg_nu_post,
            # ``dof`` is the runner-facing name of this field: a native row
            # forwards it straight into ``native.run_mg(dof=...)``, where it
            # becomes ``NativeCase.dof`` and then the coarse basis of the
            # hierarchy (``E = dof[1]``).
            "dof": self.mg_dof,
            "threads": self.threads,
            "device_id": self.device_id,
            "coarse_tol_factor": self.coarse_tol_factor,
            "reference_budget": self.reference_budget,
        }
        for key, value in optional.items():
            if value is not None:
                kwargs[key] = value
        return kwargs


@dataclass(frozen=True)
class Case:
    """One declarative test case.

    ``runner`` is a ``"module:function"`` string resolved at run time;
    ``expect`` maps observation keys to gates that this module evaluates after
    the runner returns.  ``requires`` lists skip tokens the runner is expected
    to honour (it is metadata for ``--list`` and the report: probing a native
    backend is the runner module's job, not the harness's).

    ``pinned`` marks a case whose lattice and dtype are part of its identity
    (the L48 two-node matrix): :meth:`Test.select` will not rescale it to the
    run-level ``--lattice-size``/``--dtype``.  Every tier-0 case is unpinned,
    so the ``8x8x8x16`` matrix keeps its historical retarget behaviour.
    """

    tag: str
    backend: str
    kind: str
    runner: str
    lattice: tuple[int, int, int, int] = policy.DEFAULT_LATTICE
    dtype: str = policy.DEFAULT_DTYPE
    tier: int = 0
    requires: tuple[str, ...] = ()
    expect: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    timeout: float | None = None
    pinned: bool = False

    def __post_init__(self) -> None:
        if not self.tag or "/" not in self.tag:
            raise ValueError(
                f"case tag must look like 'backend/function', got {self.tag!r}")
        if self.backend not in policy.BACKENDS:
            raise ValueError(
                f"case {self.tag!r}: unknown backend {self.backend!r}; "
                f"choose from {', '.join(policy.BACKENDS)}")
        if not self.tag.startswith(f"{self.backend}/"):
            raise ValueError(
                f"case {self.tag!r}: tag must start with {self.backend + '/'!r}")
        if not isinstance(self.runner, str) or ":" not in self.runner:
            raise ValueError(
                f"case {self.tag!r}: runner must be 'module:function', "
                f"got {self.runner!r}")
        if int(self.tier) < 0:
            raise ValueError(f"case {self.tag!r}: tier must be >= 0")
        if self.timeout is not None and float(self.timeout) <= 0.0:
            raise ValueError(f"case {self.tag!r}: timeout must be > 0")
        unknown = tuple(token for token in self.requires
                        if not policy.is_skip_token(token))
        if unknown:
            raise ValueError(
                f"case {self.tag!r}: unknown skip token(s) {unknown!r}; "
                f"choose from {', '.join(policy.SKIP_TOKENS)}")
        expect = _freeze(self.expect or {})
        if not isinstance(expect, Mapping) or not expect:
            raise ValueError(f"case {self.tag!r}: expect must not be empty")
        for key, spec in expect.items():
            if not isinstance(spec, Mapping) or not spec:
                raise ValueError(
                    f"case {self.tag!r}: expect[{key!r}] must be a non-empty "
                    "mapping of gates")
            for operator in spec:
                if operator not in GATE_OPERATORS:
                    raise ValueError(
                        f"case {self.tag!r}: expect[{key!r}] uses unknown "
                        f"operator {operator!r}")
        object.__setattr__(self, "lattice",
                           policy.validate_lattice(self.lattice))
        object.__setattr__(self, "dtype", policy.validate_dtype(self.dtype))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "expect", expect)
        object.__setattr__(self, "tier", int(self.tier))
        object.__setattr__(self, "pinned", bool(self.pinned))
        object.__setattr__(self, "timeout",
                           None if self.timeout is None else float(self.timeout))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description (used by ``--list``)."""
        return {
            "tag": self.tag,
            "backend": self.backend,
            "kind": self.kind,
            "runner": self.runner,
            "tier": self.tier,
            "lattice": list(self.lattice),
            "dtype": self.dtype,
            "requires": list(self.requires),
            "expect": json_safe(self.expect),
            "timeout": self.timeout,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class Result:
    """One case's verdict, metrics and timing."""

    tag: str
    backend: str
    kind: str
    status: str
    seconds: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failed: tuple[str, ...] = ()
    reason: str | None = None
    lattice: tuple[int, int, int, int] = policy.DEFAULT_LATTICE
    dtype: str = policy.DEFAULT_DTYPE
    seed: int = policy.DEFAULT_SEED
    mpi_size: int = 1
    tier: int = 0
    runner: str = ""
    skip_token: str | None = None

    def __post_init__(self) -> None:
        if self.status not in policy.STATUSES:
            raise ValueError(
                f"unknown result status {self.status!r}; choose from "
                f"{', '.join(policy.STATUSES)}")
        object.__setattr__(self, "metrics", _freeze(self.metrics or {}))
        object.__setattr__(self, "failed", tuple(self.failed))
        if self.seconds is not None:
            object.__setattr__(self, "seconds", float(self.seconds))
        if self.status == "SKIP" and not (self.reason or "").strip():
            raise ValueError("a SKIP result must carry a non-empty reason")

    @property
    def ok(self) -> bool:
        """True only for a PASS verdict."""
        return self.status == "PASS"

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe flat record for the report."""
        return {
            "tag": self.tag,
            "backend": self.backend,
            "kind": self.kind,
            "status": self.status,
            "seconds": self.seconds,
            "metrics": json_safe(self.metrics),
            "failed": list(self.failed),
            "reason": self.reason,
            "lattice": list(self.lattice),
            "dtype": self.dtype,
            "seed": self.seed,
            "mpi_size": self.mpi_size,
            "tier": self.tier,
            "runner": self.runner,
            "skip_token": self.skip_token,
        }


def summarize(cases: Sequence[Result]) -> tuple[dict[str, int], str]:
    """Return ``(counts, aggregate status)`` for a sequence of results.

    Precedence is ``ERROR`` > ``FAIL`` > ``TIMEOUT`` > ``SKIP`` (all skipped)
    > ``PASS``, matching the exit-code policy: any of the first three is a
    failure, and a suite where every case skipped is not a pass.
    """
    return _status_summary([result.status for result in cases])


def _status_summary(statuses: Iterable[str]) -> tuple[dict[str, int], str]:
    """One verdict per status string; shared by results and the row ledger.

    A status outside :data:`policy.STATUSES` is not counted (a pending row has
    no verdict yet), so a ledger of untouched rows summarises as ``SKIP`` -
    "nothing ran" - rather than as a clean pass.
    """
    counts = {status: 0 for status in policy.STATUSES}
    for status in statuses:
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    if counts["ERROR"]:
        status = "ERROR"
    elif counts["FAIL"]:
        status = "FAIL"
    elif counts["TIMEOUT"]:
        status = "TIMEOUT"
    elif total == 0 or counts["SKIP"] == total:
        status = "SKIP"
    else:
        status = "PASS"
    return counts, status


@dataclass(frozen=True)
class TestReport:
    """Aggregate report; ``as_dict()`` is the ``--json`` payload."""

    #: Tell pytest the class is a report type, not a test case (its name would
    #: otherwise match ``python_classes = Test*`` and emit a collection warning).
    __test__ = False

    status: str
    lattice: tuple[int, int, int, int]
    dtype: str
    seed: int
    cases: tuple[Result, ...]
    counts: Mapping[str, int]
    duration_seconds: float
    mpi_size: int = 1
    mpi_rank: int = 0
    package: str = policy.PACKAGE
    version: str = _PACKAGE_VERSION
    env: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    allow_all_skipped: bool = False

    def __post_init__(self) -> None:
        if self.status not in policy.STATUSES:
            raise ValueError(f"unknown report status {self.status!r}")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "counts", _freeze(self.counts))
        object.__setattr__(self, "env", _freeze(self.env or {}))
        object.__setattr__(self, "options", _freeze(self.options or {}))
        object.__setattr__(self, "lattice",
                           policy.validate_lattice(self.lattice))

    @property
    def all_skipped(self) -> bool:
        """True when every executed case was skipped."""
        return bool(self.cases) and all(
            result.status == "SKIP" for result in self.cases)

    @property
    def failed(self) -> bool:
        """True when the run must not exit 0."""
        return self.exit_code() != policy.EXIT_CODES["pass"]

    def exit_code(self) -> int:
        """Process exit code for this report (0/1; 2 is CLI usage only)."""
        return policy.exit_code(self.status, all_skipped=self.all_skipped,
                                allow_all_skipped=self.allow_all_skipped)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe report payload."""
        return {
            "package": self.package,
            "version": self.version,
            "status": self.status,
            "exit_code": self.exit_code(),
            "lattice": list(self.lattice),
            "dtype": self.dtype,
            "seed": self.seed,
            "mpi_size": self.mpi_size,
            "mpi_rank": self.mpi_rank,
            "selected": len(self.cases),
            "counts": json_safe(self.counts),
            "all_skipped": self.all_skipped,
            "duration_seconds": round(float(self.duration_seconds), 6),
            "options": json_safe(self.options),
            "env": json_safe(self.env),
            "cases": [case.as_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class Test:
    """A selected case list plus the run configuration it will use.

    The configuration fields mirror :class:`Options` field for field (same
    names, same normalisation), except ``device``: here it is the *choice*
    (``auto``/``cpu``/``gpu``/``dcu``/``npu``) the user asked for, and
    :meth:`config` reports both the choice and the runner-facing string.
    That is what lets ``--list --json`` and ``TestReport.options`` show the
    effective configuration of a run without translating spellings.
    """

    #: Not a pytest test class despite the ``Test*`` name.
    __test__ = False

    cases: tuple[Case, ...] = ()
    lattice: tuple[int, int, int, int] = policy.DEFAULT_LATTICE
    lattice_grid: tuple[int, int, int, int] = policy.DEFAULT_LATTICE_GRID
    mg_grid: tuple[int, int, int, int] = policy.DEFAULT_MG_GRID
    dtype: str = policy.DEFAULT_DTYPE
    seed: int = policy.DEFAULT_SEED
    atol: float | None = None
    source: str = policy.DEFAULT_SOURCE
    gauge: str = policy.DEFAULT_GAUGE
    gauge_sigma: float = policy.DEFAULT_GAUGE_SIGMA
    mass: float | None = None
    kappa: float | None = None
    csw: float = policy.DEFAULT_CSW
    solver: str = policy.DEFAULT_SOLVER
    dslash: str = policy.DEFAULT_DSLASH
    device: str = policy.DEFAULT_DEVICE
    max_iter: int | None = None
    restart: int | None = None
    mg_levels: int | None = None
    mg_nu_pre: int | None = None
    mg_nu_post: int | None = None
    threads: int | None = None
    device_id: int | None = None
    coarse_tol_factor: float | None = None
    #: ``--reference-budget`` in bytes (``None`` = the native default).  It is
    #: part of :data:`CONFIG_FIELDS` so ``--list --json`` and the report show
    #: the budget a row was planned against, and ``run()`` forwards it into
    #: every :class:`Options` the way it does for the other shared fields.
    reference_budget: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        _normalize_run_fields(self)
        object.__setattr__(self, "device", _validate_device(self.device))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) \
                or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        if self.atol is not None and float(self.atol) <= 0.0:
            raise ValueError(f"atol must be > 0, got {self.atol!r}")
        if self.reference_budget is not None \
                and int(self.reference_budget) <= 0:
            raise ValueError(
                "reference_budget must be a positive byte count, got "
                f"{self.reference_budget!r}")
        # The L48 CG group is retired: an L48 plan never carries a CG case.
        # The tier-0 CG cases stay untouched on the 8x8x8x16 quick lattice.
        object.__setattr__(self, "cases",
                           l48.drop_retired(self.cases, self.lattice))

    @classmethod
    def default(cls) -> "Test":
        """The tier-0 registry declared in :mod:`pyqcu.test.units`."""
        from . import units

        return cls(cases=tuple(units.CASES))

    def config(self) -> dict[str, Any]:
        """The effective run configuration as a JSON-safe mapping.

        The keys are exactly :data:`CONFIG_FIELDS` (the fields ``Options`` and
        ``Test`` have in common) plus ``runner_device``.  ``device`` is the
        user-level choice and ``runner_device`` the string a runner receives
        (``None`` for ``auto``); both are reported so a reader never has to
        guess which mapping was applied.
        """
        values: dict[str, Any] = {}
        for name in CONFIG_FIELDS:
            value = getattr(self, name)
            values[name] = list(value) if isinstance(value, tuple) else value
        values["runner_device"] = policy.device_for_runner(self.device)
        return values

    def select(self, *, backends: Any = None, tags: Any = None,
               tier: int | str | None = None, quick: bool = False,
               lattice: Any = None, dtype: str | None = None,
               seed: int | None = None, solver: Any = None,
               dslash: Any = None, lattice_grid: Any = None,
               mg_grid: Any = None, device: Any = None, source: Any = None,
               gauge: Any = None, mass: float | None = None,
               gauge_sigma: float | None = None, csw: float | None = None,
               kappa: float | None = None, max_iter: int | None = None,
               restart: int | None = None, mg_levels: int | None = None,
               mg_nu_pre: int | None = None, mg_nu_post: int | None = None,
               threads: int | None = None,
               device_id: int | None = None,
               coarse_tol_factor: float | None = None,
               reference_budget: int | None = None) -> "Test":
        """Return a filtered, re-configured ``Test`` (never mutates ``self``).

        ``backends`` accepts a name or a sequence (``"all"`` clears the
        filter; ``cpu`` expands to torch+kupl, ``dcu`` to cuda, ``npu`` to
        torch); ``tags`` accepts exact tags, ``backend``/``kind`` segments or
        tag prefixes (``"torch/dslash"``); ``tier`` keeps cases with
        ``tier <= N``.  ``solver``/``dslash`` filter the matrix the same way
        the CLI flags do (:func:`policy.solver_selects` /
        :func:`policy.dslash_selects`); every other argument re-configures the
        run and is propagated to :class:`Options` by :meth:`run`.

        An unmatched tag or an unknown choice raises :class:`UsageError`
        instead of silently running nothing - a typo in ``--only`` must not
        look like a clean run.
        """
        backend_filter = self._normalize_backends(backends)
        patterns = self._normalize_tags(tags)

        if quick:
            defaults = policy.quick_defaults()
            tier = defaults["tier"]
            lattice = defaults["lattice"]
            dtype = defaults["dtype"]
            seed = defaults["seed"]

        solver_choice = (self.solver if solver is None
                         else _checked(policy.validate_solver, solver,
                                       "--solver"))
        dslash_choice = (self.dslash if dslash is None
                         else _checked(policy.validate_dslash, dslash,
                                       "--dslash"))

        new_lattice = (policy.validate_lattice(lattice) if lattice is not None
                       else self.lattice)
        new_dtype = policy.validate_dtype(dtype) if dtype is not None else self.dtype
        new_seed = int(seed) if seed is not None else self.seed

        # The L48 CG group is retired.  Asking for it explicitly on an L48
        # plan is a usage error (a silent empty selection would look clean);
        # an implicit selection simply loses the CG cases, while the tier-0
        # CG cases remain selectable on the 8x8x8x16 quick lattice.
        retired_cg = l48.is_l48_lattice(new_lattice)
        if retired_cg and solver is not None and solver_choice == "cg":
            raise UsageError(
                "--solver cg requests the retired L48 CG group; the L48 "
                "two-node matrix is BiCGStab + Legacy-MG only, and the tier-0 "
                "CG cases stay available on 8 8 8 16 (drop --lattice-preset "
                "L48 or use --solver bicgstab)")
        if retired_cg:
            cg_patterns = [pattern for pattern in patterns
                           if l48.is_cg_tag(pattern)]
            if cg_patterns:
                raise UsageError(
                    f"--only {', '.join(cg_patterns)} requests the retired L48 "
                    "CG group; the tier-0 CG cases stay available on 8 8 8 16")

        selected = tuple(
            case for case in self.cases
            if (backend_filter is None or case.backend in backend_filter)
            and (tier is None or case.tier <= int(tier))
            and (not patterns or any(_tag_matches(case, pattern)
                                     for pattern in patterns))
            and not (retired_cg and l48.is_cg_tag(case.tag))
            and policy.solver_selects(case.kind, case.tag, solver_choice)
            and policy.dslash_selects(case.kind, case.tag, dslash_choice))

        if patterns:
            unmatched = [
                pattern for pattern in patterns
                if not any(_tag_matches(case, pattern) for case in self.cases)
            ]
            if unmatched:
                raise UsageError(
                    f"--only matched no case: {', '.join(unmatched)}; "
                    f"known tags: {', '.join(case.tag for case in self.cases)}")

        cases = tuple(
            case if case.pinned
            else replace(case, lattice=new_lattice, dtype=new_dtype)
            for case in selected)
        new_lattice_grid = (
            _checked(policy.validate_lattice_grid, lattice_grid,
                     "--lattice-grid")
            if lattice_grid is not None else self.lattice_grid)
        new_mg_grid = (
            _checked(policy.validate_mg_grid, mg_grid, "--mg-grid")
            if mg_grid is not None else self.mg_grid)
        return Test(
            cases=cases,
            lattice=new_lattice,
            lattice_grid=new_lattice_grid,
            mg_grid=new_mg_grid,
            dtype=new_dtype,
            seed=new_seed,
            atol=self.atol,
            source=(_checked(policy.validate_source, source, "--source")
                    if source is not None else self.source),
            gauge=(_checked(policy.validate_gauge, gauge, "--gauge")
                   if gauge is not None else self.gauge),
            gauge_sigma=(self.gauge_sigma if gauge_sigma is None
                         else gauge_sigma),
            mass=self.mass if mass is None else mass,
            kappa=self.kappa if kappa is None else kappa,
            csw=self.csw if csw is None else csw,
            solver=solver_choice,
            dslash=dslash_choice,
            device=(_checked(_validate_device, device, "--device")
                    if device is not None else self.device),
            max_iter=self.max_iter if max_iter is None else max_iter,
            restart=self.restart if restart is None else restart,
            mg_levels=self.mg_levels if mg_levels is None else mg_levels,
            mg_nu_pre=self.mg_nu_pre if mg_nu_pre is None else mg_nu_pre,
            mg_nu_post=self.mg_nu_post if mg_nu_post is None else mg_nu_post,
            threads=self.threads if threads is None else threads,
            device_id=self.device_id if device_id is None else device_id,
            coarse_tol_factor=(self.coarse_tol_factor
                               if coarse_tol_factor is None
                               else coarse_tol_factor),
            reference_budget=(self.reference_budget
                              if reference_budget is None
                              else reference_budget),
        )

    @staticmethod
    def _normalize_backends(backends: Any) -> frozenset[str] | None:
        """Normalize ``--backend``; ``None`` (or ``all``) means "no filter".

        Aliases expand here too (``cpu`` -> torch+kupl), so a case filter is
        always expressed in canonical backend names.
        """
        if backends is None:
            return None
        try:
            names = policy.expand_backends(backends)
        except ValueError as error:
            raise UsageError(str(error)) from error
        if names == ("all",):
            return None
        return frozenset(names)

    @staticmethod
    def _normalize_tags(tags: Any) -> tuple[str, ...]:
        """Normalize ``--only`` values; empty (or ``all``) means "no filter"."""
        if tags is None:
            return ()
        if isinstance(tags, str):
            items = [part.strip() for part in tags.split(",")]
        else:
            items = [str(part).strip() for part in tags]
        patterns = tuple(item.strip("/") for item in items if item)
        if not patterns or "all" in patterns:
            return ()
        return patterns

    def run(self, *, repeat: int | None = None, warmup: int | None = None,
            seed: int | None = None, timeout: float | None = None,
            fail_fast: bool = False, allow_all_skipped: bool = False,
            on_result: Callable[[Result], None] | None = None) -> TestReport:
        """Execute the selected cases and return a :class:`TestReport`.

        ``repeat``/``warmup``/``seed`` default to the policy values; ``warmup
        < 1`` and ``repeat < 1`` are usage errors.  ``timeout`` overrides the
        per-case budget when given.  ``fail_fast`` stops after the first
        ``FAIL``/``ERROR``/``TIMEOUT`` (a ``SKIP`` is not a failure and never
        stops the run).  Every configuration field of this ``Test`` is copied
        into the per-case :class:`Options` and into ``TestReport.options``.
        """
        repeat = policy.DEFAULT_REPEAT if repeat is None else int(repeat)
        warmup = policy.DEFAULT_WARMUP if warmup is None else int(warmup)
        seed = self.seed if seed is None else int(seed)
        if repeat < 1:
            raise UsageError(f"--repeat must be >= 1, got {repeat}")
        if warmup < 1:
            raise UsageError(
                f"--warmup must be >= 1, got {warmup}: the first call carries "
                "plan caches, native binding and JIT and is not a steady state")
        if timeout is not None and float(timeout) <= 0.0:
            raise UsageError(f"--timeout must be > 0, got {timeout!r}")

        mpi_size, mpi_rank = mpi_info()
        started = time.perf_counter()
        results: list[Result] = []
        for case in self.cases:
            options = Options(
                lattice=case.lattice,
                lattice_grid=self.lattice_grid,
                mg_grid=self.mg_grid,
                dtype=case.dtype,
                seed=seed,
                repeat=repeat,
                warmup=warmup,
                atol=self.atol,
                timeout=case.timeout if timeout is None else float(timeout),
                source=self.source,
                gauge=self.gauge,
                gauge_sigma=self.gauge_sigma,
                mass=self.mass,
                kappa=self.kappa,
                csw=self.csw,
                solver=self.solver,
                dslash=self.dslash,
                device=self.device,
                max_iter=self.max_iter,
                restart=self.restart,
                mg_levels=self.mg_levels,
                mg_nu_pre=self.mg_nu_pre,
                mg_nu_post=self.mg_nu_post,
                threads=self.threads,
                device_id=self.device_id,
                coarse_tol_factor=self.coarse_tol_factor,
                reference_budget=self.reference_budget,
            )
            result = run_case(case, options, mpi_size=mpi_size)
            results.append(result)
            if on_result is not None:
                on_result(result)
            if fail_fast and result.status in ("FAIL", "ERROR", "TIMEOUT"):
                break
        duration = time.perf_counter() - started
        counts, status = summarize(results)
        options_summary = dict(self.config())
        options_summary.update({
            "repeat": repeat,
            "warmup": warmup,
            "timeout": timeout,
            "fail_fast": bool(fail_fast),
            "allow_all_skipped": bool(allow_all_skipped),
            "mpi_grid_matches": grid_product(self.lattice_grid) == mpi_size,
            "backends": sorted({case.backend for case in self.cases}),
            "tags": [case.tag for case in self.cases],
        })
        return TestReport(
            status=status,
            lattice=self.lattice,
            dtype=self.dtype,
            seed=seed,
            cases=tuple(results),
            counts=counts,
            duration_seconds=duration,
            mpi_size=mpi_size,
            mpi_rank=mpi_rank,
            env=probe_env(),
            options=options_summary,
            allow_all_skipped=allow_all_skipped,
        )


# ---------------------------------------------------------------------------
# full-mode pipeline: list / prepare / prewarm / solve / merge / run / validate
# ---------------------------------------------------------------------------
#: The implicit mode of a plain ``python -m pyqcu.test`` invocation: run the
#: selected cases once.  It is deliberately *not* one of :data:`MODES` - the
#: seven full modes are explicit work-ledger operations, never a silent default.
CASE_MODE = "cases"

#: The seven full-mode operations, in the order the informal
#: ``pyqcu/testing/kupl/test_kunpeng_all.py`` driver established them.
#: ``run`` is the whole pipeline (prepare -> prewarm -> solve -> merge).
MODES: tuple[str, ...] = (
    "list", "prepare", "prewarm", "solve", "merge", "run", "validate")

#: Tier a full mode selects when the user does not pin ``--tier``/``--quick``.
#: Tier 1 is the "everything declared" selection; it is only reached through an
#: explicit ``--mode``, so no invocation turns into a heavy run by accident.
FULL_MODE_TIER = 1

#: Named latices of the release-acceptance matrix.  ``--lattice-preset`` is the
#: only way to spell them: a bare ``--lattice-size 24 24 24 64`` keeps working
#: and keeps meaning the same thing.
#:
#: ``L48`` is the production configuration
#: (``beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime``, ``48x48x48x96``); the
#: preset only *names* the geometry - it selects no case by itself, so the tier-0
#: ``8x8x8x16`` matrix and ``--quick`` are unaffected, and a run still needs
#: ``--native`` (or an explicit ``Test``) to plan rows at this size.
LATTICE_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "L24": (24, 24, 24, 64),
    "L32": (32, 32, 32, 64),
    "L48": (48, 48, 48, 96),
}

#: Default work root when ``--outdir``/``--rows``/... are not given.
DEFAULT_RUN_ROOT = "pyqcu-test-runs"

#: Modules the full modes wire in **lazily** (``pyqcu.test.<name>``).  They are
#: optional: when one is absent the built-in stdlib path is used, and when one
#: is present but raises, the failure is recorded with a non-empty reason -
#: never swallowed and never turned into a PASS.  ``--list`` only checks
#: :func:`importlib.util.find_spec`; it never imports any of them.
FULL_MODE_MODULES: tuple[str, ...] = (
    "io", "cache", "native", "torch_matrix", "driver")

#: Hook names looked up, in order, per module.  A hook is called with
#: signature-filtered keyword arguments (see :func:`call_hook`) and must return
#: a mapping of observations or ``None``; its keys are merged into the ledger
#: entry (``status``/``reason``/``seconds``/``sha256``/``path``).
MODE_HOOKS: dict[str, tuple[str, ...]] = {
    "io": ("ensure_gauge", "ensure_source", "prewarm"),
    "torch_matrix": ("ensure_gauge", "ensure_source"),
    "cache": ("prepare", "validate"),
    "native": ("probe",),
    "driver": ("run_row",),
}

#: Fields that decide *what one row computes*.  ``run_id`` is their SHA-1, so
#: two invocations that would compute the same thing share one ledger row and
#: a changed lattice/dtype/rank count is a different row.  ``--run-label`` is
#: appended as a suffix (not hashed) to keep cold/warm or before/after runs of
#: the same configuration apart, exactly like the informal driver.
ROW_DIGEST_FIELDS: tuple[str, ...] = (
    "tag", "lattice", "dtype", "seed", "solver", "dslash", "mass", "kappa",
    "csw", "ranks", "threads", "shm", "shared", "gauge", "source",
    "gauge_file", "source_file", "combination")

#: Column order of the row ledger CSV.  A closed vocabulary: a row key outside
#: it stays in ``rows/<run_id>.json`` only, so the CSV never grows silently.
ROW_FIELDS: tuple[str, ...] = (
    "run_id", "status", "tag", "backend", "kind", "tier", "lattice", "dtype",
    "seed", "solver", "dslash", "device", "ranks", "threads", "shm", "shared",
    "gauge", "source", "gauge_file", "source_file", "mass", "kappa", "csw",
    "seconds", "reason", "skip_token", "resumed", "run_label", "report_path",
    "solution_path")

#: Chunk size of the built-in prewarm read (8 MiB: one page-cache pass, no
#: per-element work, bounded memory even for a multi-GB gauge file).
PREWARM_CHUNK = 8 * 1024 * 1024

_MODE_MODULES: dict[str, tuple[Any, str | None]] = {}


def absolute_path(value: Any) -> Path:
    """Absolute ``Path`` for a CLI value, without resolving symlinks.

    ``Path.resolve()`` would touch the filesystem and rewrite symlinked work
    roots; ``--list --json`` must be able to report a path that does not exist
    yet, so this is ``abspath(expanduser(...))`` and nothing else.
    """
    return Path(os.path.abspath(os.path.expanduser(str(value))))


@dataclass(frozen=True)
class ModePaths:
    """The directories and the ledger file one full-mode run reads and writes.

    Every path is resolved once, here, and then reported by ``--list --json``
    and by every mode payload: a run can be reproduced from its own output, and
    a typo in ``--rows`` cannot silently write into a second location.
    """

    root: Path
    rows: Path
    reports: Path
    solutions: Path
    cache: Path
    csv: Path

    @classmethod
    def resolve(cls, *, root: Any = None, rows: Any = None,
                reports: Any = None, solutions: Any = None,
                cache: Any = None, csv: Any = None) -> "ModePaths":
        """Resolve the work layout; ``--outdir`` is the default root."""
        base = (absolute_path(root) if root is not None
                else absolute_path(DEFAULT_RUN_ROOT))
        rows_dir = absolute_path(rows) if rows is not None else base / "rows"
        reports_dir = (absolute_path(reports) if reports is not None
                       else base / "reports")
        solutions_dir = (absolute_path(solutions) if solutions is not None
                         else base / "solutions")
        cache_dir = (absolute_path(cache) if cache is not None
                     else base / "caches")
        csv_path = (absolute_path(csv) if csv is not None
                    else reports_dir / "rows.csv")
        return cls(root=base, rows=rows_dir, reports=reports_dir,
                   solutions=solutions_dir, cache=cache_dir, csv=csv_path)

    def as_dict(self) -> dict[str, str]:
        """JSON-safe path map (``--list --json`` and mode payloads)."""
        return {
            "root": str(self.root),
            "rows": str(self.rows),
            "reports": str(self.reports),
            "solutions": str(self.solutions),
            "cache": str(self.cache),
            "csv": str(self.csv),
        }

    def ensure(self) -> None:
        """Create every directory the modes write into (``parents=True``).

        ``--cache-dir`` may name an existing cache *artifact* (one HDF5 file),
        so a file there is accepted and left alone; a file where a row/report
        directory belongs is a usage error, not a traceback.
        """
        for label, directory in (("rows", self.rows),
                                 ("reports", self.reports),
                                 ("solutions", self.solutions)):
            if directory.exists() and not directory.is_dir():
                raise UsageError(
                    f"--{label} {directory} exists and is not a directory")
            directory.mkdir(parents=True, exist_ok=True)
        if not self.cache.is_file():
            self.cache.mkdir(parents=True, exist_ok=True)
        if self.csv.exists() and self.csv.is_dir():
            raise UsageError(f"--csv {self.csv} is a directory")
        self.csv.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FullModeOptions:
    """Full-mode work-ledger configuration (the ``--mode`` half of the CLI).

    ``mode=None`` means "no full mode requested": the invocation runs the
    selected cases once, exactly as before.  ``tier`` is recorded separately
    from :class:`Test` because a full mode defaults to
    :data:`FULL_MODE_TIER` while a plain run keeps "all declared tiers".
    """

    mode: str | None = None
    paths: ModePaths = field(default_factory=ModePaths.resolve)
    run_label: str = ""
    resume: bool = False
    skip_failed: bool = False
    shared: bool = False
    ranks: int = 1
    threads: int | None = None
    shm: bool | None = None
    gauge_file: str | None = None
    source_file: str | None = None
    validate: bool = False
    lattice_preset: str | None = None
    #: The grid the invocation *asked for* (``--lattice-grid``), or ``None``
    #: when it asked for nothing.  ``None`` is not the same as the default
    #: ``(1, 1, 1, 1)``: a native row derives its grid from ``ranks`` in that
    #: case, while an explicit grid is the decomposition the row must use (or
    #: report a ``SKIP(mpi_grid)``) - a requested grid is never replaced by a
    #: different one.  ``build_rows`` records it as the row's ``lattice_grid``,
    #: which is what :func:`pyqcu.test.driver.run_row` consumes.
    lattice_grid: tuple[int, int, int, int] | None = None
    tier: int = FULL_MODE_TIER
    repeat: int = policy.DEFAULT_REPEAT
    warmup: int = policy.DEFAULT_WARMUP
    timeout: float | None = None
    # Solver knobs a full-mode row must be able to vary; absent means the
    # matrix declaration still supplies the reproducible default.
    restart: int | None = None
    mg_levels: int | None = None
    mg_grid: tuple[int, int, int, int] | None = None
    #: ``--mg-dof``: per-level component counts, ``(12, E)``.  ``E`` is the
    #: coarse basis dimension of level 1 and therefore the capacity of the
    #: coarse space; it reaches the solver as ``params[kMgLevel1E]`` through the
    #: coarse ``lonv`` tensor, so a value here must match the cache artifact
    #: that supplies those tensors (``native._load_ranked_mg_assets`` refuses a
    #: mismatch).  ``None`` means the flag was not given.
    mg_dof: tuple[int, ...] | None = None
    coarse_tol_factor: float | None = None
    #: Solver tolerance (relative residual).  ``None`` means the flag was not
    #: given, so the native runner keeps its own default
    #: (``native.DEFAULT_ATOL_C64`` / ``native.DEFAULT_MG_ATOL_C64``); a value
    #: is passed straight through.  Without this field the whole full-mode path
    #: dropped ``--atol`` and two runs that looked matched solved to a decade
    #: apart (2026-11-24, rows 2245167 vs 2245127).
    atol: float | None = None
    #: ``--reference-budget`` in bytes (``None`` = the native default).  A
    #: *run* option: ``driver.run_row`` receives ``as_dict()`` and hands it to
    #: ``native.run_case``, which plans the independent residual against it.
    reference_budget: int | None = None
    allow_all_skipped: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in MODES:
            raise ValueError(
                f"unknown mode {self.mode!r}; choose from {', '.join(MODES)}")
        label = str(self.run_label or "").strip()
        if "/" in label or "\\" in label or os.sep in label:
            raise ValueError(
                f"run_label must not contain a path separator, got {label!r}")
        if int(self.ranks) < 1:
            raise ValueError(f"ranks must be >= 1, got {self.ranks!r}")
        if self.threads is not None and int(self.threads) < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads!r}")
        if self.lattice_preset is not None \
                and self.lattice_preset not in LATTICE_PRESETS:
            raise ValueError(
                f"unknown lattice preset {self.lattice_preset!r}; choose from "
                f"{', '.join(LATTICE_PRESETS)}")
        if int(self.tier) not in policy.TIERS:
            raise ValueError(
                f"mode tier must be one of {policy.TIERS}, got {self.tier!r}")
        if self.atol is not None and float(self.atol) <= 0.0:
            raise ValueError(f"atol must be > 0, got {self.atol!r}")
        if self.reference_budget is not None \
                and int(self.reference_budget) <= 0:
            raise ValueError(
                "reference_budget must be a positive byte count, got "
                f"{self.reference_budget!r}")
        object.__setattr__(self, "run_label", label)
        object.__setattr__(self, "ranks", int(self.ranks))
        object.__setattr__(self, "tier", int(self.tier))
        object.__setattr__(self, "threads",
                           None if self.threads is None else int(self.threads))
        object.__setattr__(self, "shm",
                           None if self.shm is None else bool(self.shm))
        object.__setattr__(self, "shared", bool(self.shared))
        object.__setattr__(self, "gauge_file",
                           None if self.gauge_file is None
                           else str(absolute_path(self.gauge_file)))
        object.__setattr__(self, "source_file",
                           None if self.source_file is None
                           else str(absolute_path(self.source_file)))
        object.__setattr__(self, "lattice_grid",
                           None if self.lattice_grid is None
                           else policy.validate_lattice_grid(self.lattice_grid))
        object.__setattr__(self, "mg_dof", _optional_dof(self.mg_dof))

    def as_dict(self) -> dict[str, Any]:
        """The effective full-mode configuration (``--list --json``/report)."""
        return {
            "mode": self.mode,
            "tier": self.tier,
            "paths": self.paths.as_dict(),
            "run_label": self.run_label,
            "resume": self.resume,
            "skip_failed": self.skip_failed,
            "shared": self.shared,
            "ranks": self.ranks,
            "threads": self.threads,
            "shm": self.shm,
            "gauge_file": self.gauge_file,
            "source_file": self.source_file,
            "cache_dir": str(self.paths.cache),
            "validate": self.validate,
            "lattice_preset": self.lattice_preset,
            "lattice_grid": (None if self.lattice_grid is None
                             else list(self.lattice_grid)),
            "repeat": self.repeat,
            "warmup": self.warmup,
            "timeout": self.timeout,
            "restart": self.restart,
            "mg_levels": self.mg_levels,
            "mg_grid": (None if self.mg_grid is None
                        else [int(v) for v in self.mg_grid]),
            "mg_dof": (None if self.mg_dof is None
                       else [int(v) for v in self.mg_dof]),
            "coarse_tol_factor": self.coarse_tol_factor,
            "atol": self.atol,
            "reference_budget": self.reference_budget,
            "allow_all_skipped": self.allow_all_skipped,
            "fail_fast": self.fail_fast,
        }


# ---------------------------------------------------------------------------
# row identity and the append-only ledger
# ---------------------------------------------------------------------------
def row_id(config: Mapping[str, Any], *, label: str = "") -> str:
    """Deterministic ``run_id`` of one row: configuration digest + label.

    The digest covers :data:`ROW_DIGEST_FIELDS` only, so the ledger identity is
    "what would be computed", not "when it was computed".  ``--run-label`` is a
    suffix on purpose: two labelled runs of one configuration must not share a
    ``rows/<run_id>.json``.
    """
    payload = {name: json_safe(config.get(name))
               for name in ROW_DIGEST_FIELDS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    label = str(label or "").strip()
    return f"{digest}-{label}" if label else digest


def build_rows(test: "Test", options: FullModeOptions, *,
               combination: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One ledger row per selected case (and sweep combination, if any).

    A row records the *effective* configuration: the case's lattice/dtype after
    ``Test.select``, the run's seed/solver/dslash/gauge/source, and the
    full-mode fields (``ranks``/``threads``/``shm``/``shared``/input files).
    ``status`` stays ``None`` until a mode writes a verdict, so a listed plan
    can never be mistaken for an executed one.
    """
    rows: list[dict[str, Any]] = []
    for case in test.cases:
        row: dict[str, Any] = {
            "run_id": "",
            "status": None,
            "tag": case.tag,
            "backend": case.backend,
            "kind": case.kind,
            "tier": case.tier,
            "lattice": list(case.lattice),
            "dtype": case.dtype,
            "seed": int(test.seed),
            "solver": test.solver,
            "dslash": test.dslash,
            "device": test.device,
            "ranks": int(options.ranks),
            # ``None`` = "derive one from ranks" (the native driver's job);
            # an explicit ``--lattice-grid`` is recorded so the row consumes the
            # grid it was asked for instead of silently deriving another one.
            "lattice_grid": (None if options.lattice_grid is None
                             else list(options.lattice_grid)),
            "threads": (options.threads if options.threads is not None
                        else test.threads),
            "shm": options.shm,
            "shared": bool(options.shared),
            "gauge": test.gauge,
            "source": test.source,
            "gauge_file": options.gauge_file,
            "source_file": options.source_file,
            "mass": test.mass,
            "kappa": test.kappa,
            "csw": float(test.csw),
            "seconds": None,
            "metrics": {},
            "failed": [],
            "reason": None,
            "skip_token": None,
            "resumed": False,
            "run_label": options.run_label,
            "report_path": None,
            "solution_path": None,
        }
        if combination is not None:
            row["combination"] = dict(combination)
        # The L48 two-node matrix declares its own geometry and solver
        # parameters; a row must reproduce the declaration instead of the
        # run-level defaults of an unrelated flag (which may be 1 rank and
        # ``--solver all``).  Non-matrix tags get no extra fields.
        overrides = l48.row_overrides(case.tag)
        if overrides:
            row.update({
                "dtype": overrides["dtype"],
                "ranks": overrides["ranks"],
                "lattice_grid": list(overrides["lattice_grid"]),
                "nodes": overrides["nodes"],
                "solver": overrides["solver"],
                "mg_levels": overrides["mg_levels"],
                "restart": overrides["restart"],
                "dslash": overrides["dslash"],
                "l48_group": overrides["group"],
                "l48_evidence": overrides["evidence"],
                "l48_blocked_by": overrides["blocked_by"],
            })
        row["run_id"] = row_id(row, label=options.run_label)
        rows.append(row)
    return rows


def row_path(paths: ModePaths, run_id: str) -> Path:
    """Ledger path of one row (``<rows>/<run_id>.json``)."""
    return paths.rows / f"{run_id}.json"


def read_row(paths: ModePaths, run_id: str) -> dict[str, Any] | None:
    """Read one row file; ``None`` when it is absent or unparsable."""
    path = row_path(paths, run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def row_is_resumable(existing: Any, *, skip_failed: bool = False) -> bool:
    """True when ``--resume`` may reuse ``existing`` instead of re-running.

    ``PASS`` and ``SKIP`` are terminal; ``FAIL`` is terminal only under
    ``--skip-failed`` (the informal driver's documented trap: it then never
    retries a real failure - so it is an explicit opt-in here too).  ``ERROR``
    and ``TIMEOUT`` are always retried: they may be transient.
    """
    if not isinstance(existing, Mapping):
        return False
    status = existing.get("status")
    if status in ("PASS", "SKIP"):
        return True
    return bool(skip_failed) and status == "FAIL"


def agreed_resume_row(paths: ModePaths, row: Mapping[str, Any], *,
                      skip_failed: bool = False, size: int | None = None,
                      rank: int | None = None, comm: Any = None
                      ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The row every rank may resume, or ``None``: one logical decision.

    ``--resume`` reads ``rows/<run_id>.json`` and ``--resume``/``--skip-failed``
    are per-row decisions, so a replica rank that resumed a row while another
    rank re-ran it would skip the driver's collectives and leave the others
    waiting inside one - the 32-rank hang the audit records as P0-5.  The
    decision is therefore taken once for the whole world: every rank reports
    whether *it* could resume the row and the row is reused only when **all** of
    them can.

    Returns ``(existing_row, evidence)``.  ``existing_row`` is the row to reuse
    (``None``: execute it), and ``evidence`` is JSON-safe so the caller can
    record why:

    ``resume_gather``        ``allgather``/``single_rank``/``unavailable``
    ``resume_disagreement``  the ranks did not agree, so the row was re-run
    ``resume_reason``        why a resumable row was not reused
    ``resume_ranks``         how many ranks were asked

    Without a usable communicator a multi-rank world never resumes anything (the
    same answer on every rank, so the collectives stay aligned) and the local
    ledger is still read for the single-process case, where ``--resume`` behaves
    exactly as before.
    """
    run_id = str(row.get("run_id") or "")
    existing = read_row(paths, run_id)
    local = row_is_resumable(existing, skip_failed=skip_failed)
    world, world_rank = mpi_info()
    total = int(world if size is None else size)
    here = int(world_rank if rank is None else rank)
    evidence: dict[str, Any] = {"resume_ranks": total}
    if total <= 1:
        evidence["resume_gather"] = "single_rank"
        evidence["resume_disagreement"] = False
        return (dict(existing) if local and existing is not None else None), evidence
    records, mode, failure = allgather_rank_records(
        {"rank": here, "resumable": bool(local)}, size=total, rank=here,
        comm=comm)
    evidence["resume_gather"] = mode
    evidence["resume_disagreement"] = False
    if failure:
        evidence["resume_gather_reason"] = str(failure)
    claimed = [item.get("resumable") is True for item in records]
    if mode != "allgather" or len(claimed) != total:
        # No comparison was possible: re-running is the only answer that every
        # rank reaches on its own, so no rank can drop out of a collective.
        evidence["resume_reason"] = (
            "the multi-rank resume decision is unavailable "
            f"({failure or 'no communicator'}); the row was re-run so that "
            "no rank skips a collective the others enter")
        return None, evidence
    evidence["resume_disagreement"] = len(set(claimed)) > 1
    if not all(claimed):
        evidence["resume_reason"] = (
            "not every rank could resume this row "
            f"({sum(1 for value in claimed if not value)}/{total} could not); "
            "the row was re-run so that no rank skips a collective the others "
            "enter")
        return None, evidence
    if existing is None:  # pragma: no cover - all ranks said "resumable"
        return None, evidence
    return dict(existing), evidence


def _csv_value(value: Any) -> str:
    """Render one CSV cell: ``""`` for unset, vectors as ``8x8x8x16``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)) and value \
            and all(isinstance(item, int) and not isinstance(item, bool)
                    for item in value):
        return "x".join(str(item) for item in value)
    if isinstance(value, (list, tuple, Mapping)):
        return json.dumps(json_safe(value), sort_keys=True)
    return str(value)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a ledger CSV into ``(header, records)``; empty when absent."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            return header, [dict(record) for record in reader]
    except (OSError, csv.Error, UnicodeDecodeError):
        return [], []


def _write_csv_records(path: Path, columns: Sequence[str],
                       records: Sequence[Mapping[str, Any]]) -> Path:
    """Atomically rewrite a CSV with ``columns`` (temporary sibling + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _csv_columns(path: Path, columns: Sequence[str]) -> list[str]:
    """Header to append with, upgrading an older CSV instead of misaligning it.

    Appending a new row set to a CSV written by an earlier version would shift
    every value; the header is therefore reconciled first, keeping any
    legacy-only column and every existing record.
    """
    wanted = list(columns)
    if not path.is_file() or path.stat().st_size == 0:
        return wanted
    header, records = _read_csv(path)
    if not header or header == wanted:
        return wanted
    merged = wanted + [name for name in header if name not in wanted]
    _write_csv_records(path, merged, records)
    return merged


def append_csv(path: Any, rows: Sequence[Mapping[str, Any]], *,
               columns: Sequence[str] = ROW_FIELDS) -> Path:
    """Append rows to the ledger CSV (header added/upgraded as needed)."""
    target = absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return target
    header = _csv_columns(target, columns)
    fresh = not target.is_file() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header,
                                extrasaction="ignore")
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name))
                             for name in header})
    return target


def write_row(paths: ModePaths, row: dict[str, Any], *,
              columns: Sequence[str] = ROW_FIELDS,
              owner: bool | None = None) -> Path:
    """Persist one row: ``rows/<run_id>.json``, its report copy and the CSV.

    The ledger has exactly one writer: rank 0 of the MPI world
    (:func:`ledger_owner`).  A non-owning rank of a 32-process row validates the
    row exactly like rank 0 and then performs **no I/O** - it has already
    reported through :func:`gather_rank_outcome` - so ``rows/``, ``reports/``
    and the append-only CSV can never be written concurrently.  ``owner``
    overrides the probe for tests and library callers.

    Returns the row path in every case, so a caller can report what the owning
    rank wrote (or will write) without a second path derivation.
    """
    run_id = str(row.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("cannot write a row without a run_id")
    if row.get("status") == "SKIP" and not str(row.get("reason") or "").strip():
        raise ValueError(f"row {run_id}: a SKIP needs a non-empty reason")
    row["report_path"] = str(paths.reports / f"{run_id}.json")
    path = row_path(paths, run_id)
    if not (ledger_owner() if owner is None else bool(owner)):
        return path
    paths.ensure()
    write_json_atomic(path, row)
    write_json_atomic(paths.reports / f"{run_id}.json", row)
    append_csv(paths.csv, [row], columns=columns)
    return path


def load_rows(paths: ModePaths, *,
              limit: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every ``rows/*.json`` (sorted by ``run_id``) plus parse problems.

    A row that cannot be parsed is reported, never skipped: a ledger that
    silently drops an unreadable row would merge into a clean-looking CSV.
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    directory = paths.rows
    if not directory.is_dir():
        return rows, [f"row directory does not exist: {directory}"]
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            problems.append(
                f"{path.name}: unreadable row ({type(error).__name__}: {error})")
            continue
        if not isinstance(data, dict):
            problems.append(
                f"{path.name}: row is {type(data).__name__}, expected an object")
            continue
        data.setdefault("run_id", path.stem)
        rows.append(data)
    rows.sort(key=lambda row: str(row.get("run_id")))
    if limit is not None:
        rows = rows[:max(int(limit), 0)]
    return rows, problems


# ---------------------------------------------------------------------------
# lazy wiring of the optional full-mode modules
# ---------------------------------------------------------------------------
def module_availability() -> dict[str, bool]:
    """``{module: present}`` for :data:`FULL_MODE_MODULES` without importing.

    ``importlib.util.find_spec`` locates a module without executing it, so
    ``--list`` and ``--mode list`` can report what would be wired while staying
    free of torch/h5py/mpi4py (``pyqcu.test.io`` would import ``h5py``).
    """
    availability: dict[str, bool] = {}
    for name in FULL_MODE_MODULES:
        try:
            availability[name] = importlib.util.find_spec(
                f"{policy.TEST_PACKAGE}.{name}") is not None
        except (ImportError, ValueError, AttributeError):
            availability[name] = False
    return availability


def load_mode_module(name: str) -> tuple[Any, str | None]:
    """Import ``pyqcu.test.<name>`` lazily; return ``(module, error)``.

    Import failures are cached and returned as a message instead of raising:
    a missing optional module is a normal state, a broken one must be visible
    in the payload (both are non-empty reasons, never a silent fallback).
    """
    if name not in MODE_HOOKS:
        raise ValueError(
            f"unknown full-mode module {name!r}; choose from "
            f"{', '.join(FULL_MODE_MODULES)}")
    cached = _MODE_MODULES.get(name)
    if cached is not None:
        return cached
    try:
        module = importlib.import_module(f"{policy.TEST_PACKAGE}.{name}")
    except Exception as error:  # noqa: BLE001 - reported, never propagated
        cached = (None, f"{type(error).__name__}: {error}")
    else:
        cached = (module, None)
    _MODE_MODULES[name] = cached
    return cached


def mode_hook(module_name: str, hook_name: str) -> Callable[..., Any] | None:
    """The hook callable when the module is present and declares it."""
    module, _ = load_mode_module(module_name)
    if module is None:
        return None
    function = getattr(module, hook_name, None)
    return function if callable(function) else None


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    """True when ``function`` accepts keyword ``name`` (or ``**kwargs``)."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD
               for parameter in parameters.values())


def call_hook(function: Callable[..., Any], **kwargs: Any) -> tuple[bool, Any, str | None]:
    """Call a hook with the keyword arguments its signature accepts.

    The optional modules are not part of this tree yet, so the contract is
    *keyword-only and filtered*: an unknown keyword is dropped, a raised
    exception is returned as ``(False, None, "Type: message")``.  A hook can
    therefore be written with any subset of the documented keywords without
    breaking the CLI, and a broken one never becomes a silent PASS.
    """
    accepted = {key: value for key, value in kwargs.items()
                if _accepts_keyword(function, key)}
    try:
        return True, function(**accepted), None
    except Exception as error:  # noqa: BLE001 - reported, never propagated
        return False, None, f"{type(error).__name__}: {error}"


def _hooks(module_names: Sequence[str],
           hook_names: Sequence[str]) -> list[tuple[str, str, Callable[..., Any]]]:
    """Every available ``(module, hook, callable)`` in the requested order."""
    found: list[tuple[str, str, Callable[..., Any]]] = []
    for module_name in module_names:
        for hook_name in hook_names:
            function = mode_hook(module_name, hook_name)
            if function is not None:
                found.append((module_name, hook_name, function))
    return found


# ---------------------------------------------------------------------------
# mode documents and entries
# ---------------------------------------------------------------------------
def _mode_entry(role: str, path: Any = None, *, status: str, reason: str | None,
                **extra: Any) -> dict[str, Any]:
    """One input/cache entry of ``prepare``/``prewarm`` (status vocabulary)."""
    if status not in policy.STATUSES:
        raise ValueError(f"unknown entry status {status!r}")
    text = "" if reason is None else str(reason).strip()
    if status == "SKIP" and not text:
        raise ValueError(f"{role}: a SKIP needs a non-empty reason")
    entry: dict[str, Any] = {
        "role": role,
        "path": None if path is None else str(path),
        "status": status,
        "reason": text or None,
    }
    entry.update({key: json_safe(value) for key, value in extra.items()})
    return entry


def _mode_document(mode: str, options: FullModeOptions, *, status: str,
                   counts: Mapping[str, int] | None = None,
                   rows: Sequence[Mapping[str, Any]] | None = None,
                   inputs: Sequence[Mapping[str, Any]] | None = None,
                   notes: Sequence[str] = (),
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The JSON document a full mode returns (``--json``/``--mode ...``)."""
    document: dict[str, Any] = {
        "package": policy.PACKAGE,
        "version": _PACKAGE_VERSION,
        "mode": mode,
        "status": status,
        "counts": dict(counts or {name: 0 for name in policy.STATUSES}),
        "paths": options.paths.as_dict(),
        "options": options.as_dict(),
        "modules": module_availability(),
        "notes": [str(note) for note in notes],
        "duration_seconds": 0.0,
    }
    if rows is not None:
        document["rows"] = {
            "dir": str(options.paths.rows),
            "count": len(rows),
            "csv": str(options.paths.csv),
            "items": [json_safe(row) for row in rows],
        }
    if inputs is not None:
        document["inputs"] = [json_safe(entry) for entry in inputs]
    if extra:
        document.update(json_safe(dict(extra)))
    return document


def _mode_exit(status: str) -> int:
    """Exit code of a non-numeric mode: 1 only for a real row failure.

    ``prepare``/``prewarm`` skip an input that was not requested; that is a
    no-op, not a failing test, so ``SKIP`` alone stays ``0``.  The numeric
    modes (``solve``/``run``) use :func:`policy.exit_code` instead, where an
    all-skipped run is a failure.
    """
    if status in ("FAIL", "ERROR", "TIMEOUT"):
        return policy.EXIT_CODES["fail"]
    return policy.EXIT_CODES["pass"]


# ---------------------------------------------------------------------------
# inputs: prepare and prewarm
# ---------------------------------------------------------------------------
def _io_probe(role: str, target: Path) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate one HDF5 artifact through :mod:`pyqcu.test.io` when present.

    ``io`` is imported lazily only here: it is the module that knows the HDF5
    layout, so "the file exists" is *not* accepted as "the file is valid"
    (that mistake cost the 2026-11 matrix an entire L32 block).  ``is_readable``
    is the non-raising gate; ``describe`` returns the self-description used to
    cross-check the run's lattice.  A module that raises is a failure with its
    message - never a silent fallback to ``stat()``.
    """
    module, error = load_mode_module("io")
    if module is None:
        if error is not None:
            return False, f"io module failed to import: {error}", None
        return True, "", None
    readable = getattr(module, "is_readable", None)
    describe = getattr(module, "describe", None)
    if callable(readable):
        try:
            outcome = readable(target, role=role)
        except TypeError:
            try:
                outcome = readable(target)
            except Exception as nested:  # noqa: BLE001 - reported
                return False, ("io.is_readable failed: "
                               f"{type(nested).__name__}: {nested}"), None
        except Exception as failure:  # noqa: BLE001 - reported
            return False, (f"io.is_readable failed: "
                           f"{type(failure).__name__}: {failure}"), None
        ok, reason = (outcome if isinstance(outcome, tuple) else (outcome, ""))
        if not ok:
            return False, f"io.is_readable: {reason or 'unreadable'}", None
    if callable(describe):
        try:
            info = describe(target, role=role)
        except Exception as failure:  # noqa: BLE001 - reported
            return False, (f"io.describe failed: "
                           f"{type(failure).__name__}: {failure}"), None
        if isinstance(info, Mapping):
            return True, "", json_safe(dict(info))
    return True, "", None


def _input_entry(role: str, path: Any, *, lattice: Sequence[int], dtype: str,
                 seed: int, prewarm: bool) -> dict[str, Any]:
    """One input file's ``prepare``/``prewarm`` entry.

    Validation goes through :func:`_io_probe` (``io``), the read through the
    built-in 8 MiB chunk loop (page-cache warm, bounded memory).  The declared
    lattice of an HDF5 artifact must match the run: a mistyped ``--gauge-file``
    from another lattice is a ``FAIL`` with the two lattices in the reason,
    instead of a numerically meaningless solve.
    """
    flag = f"--{role}-file"
    entry = _mode_entry(
        role, None, status="SKIP",
        reason=f"no {role} file requested ({flag} unset)")
    if path is None:
        return entry
    target = absolute_path(path)
    entry["path"] = str(target)
    if not target.exists():
        entry.update(status="FAIL",
                     reason=f"{role} file not found: {target}")
        return entry
    if target.is_dir():
        entry.update(status="FAIL",
                     reason=f"{role} path is a directory: {target}")
        return entry

    ok, reason, info = _io_probe(role, target)
    if not ok:
        entry.update(status="FAIL",
                     reason=f"{role} file rejected: {reason}")
        return entry
    if info:
        entry["io"] = {key: info[key] for key in
                       ("role", "dataset", "shape", "dtype", "layout",
                        "lattice", "bytes", "file_bytes",
                        "h5py_version")
                       if key in info}
        declared = info.get("lattice")
        if isinstance(declared, (list, tuple)) and declared \
                and tuple(int(value) for value in declared) \
                != tuple(int(value) for value in lattice):
            entry.update(
                status="FAIL",
                reason=(f"{role} file lattice {list(declared)} != run lattice "
                        f"{list(lattice)}"))
            return entry

    started = time.perf_counter()
    try:
        if prewarm:
            digest = hashlib.sha256()
            total = 0
            with target.open("rb") as handle:
                while True:
                    chunk = handle.read(PREWARM_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
            entry["bytes"] = total
            entry["sha256"] = digest.hexdigest()
        else:
            entry["bytes"] = target.stat().st_size
    except OSError as error:
        entry.update(status="FAIL",
                     reason=f"cannot read {role} file {target}: "
                            f"{type(error).__name__}: {error}")
        return entry
    entry["seconds"] = round(time.perf_counter() - started, 6)
    entry["status"] = "PASS"
    entry["reason"] = None
    return entry


def _native_probe(options: FullModeOptions, notes: list[str]) -> dict[str, Any] | None:
    """Optional ``native.probe`` call: ranks/threads/shm/shared evidence."""
    function = mode_hook("native", "probe")
    if function is None:
        return None
    ok, value, error = call_hook(
        function, ranks=int(options.ranks), threads=options.threads,
        shm=options.shm, shared=bool(options.shared),
        lattice_grid=None)
    if not ok:
        notes.append(f"native.probe failed: {error}")
        return {"error": error}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        notes.append(f"native.probe returned {type(value).__name__}, "
                     "expected a mapping")
        return {"error": f"returned {type(value).__name__}"}
    return json_safe(dict(value))


def _cache_row_binding(rows: Sequence[Mapping[str, Any]],
                       notes: list[str]) -> dict[str, Any] | None:
    """Partial Legacy-MG cache identity the selected rows agree on.

    A probe that does not say which configuration it is probing for accepts a
    cache built for something else (see the A30 audit), so the row fields whose
    meaning is the same for every backend are bound into
    :func:`cache.cache_status`: ``lattice``, ``dtype``, ``seed``, an explicitly
    requested ``mass`` and the process count.  The cache-only fields
    (``dof``/``mg_grid``/``nv_iters``) are not row properties, and ``csw`` is
    not bound either: a row defaults it to ``policy.DEFAULT_CSW`` while the
    native MG builder defaults to ``native.DEFAULT_CSW``, so an unset ``--csw``
    would reject a cache that is in fact correct.  A field the rows disagree on
    is left unbound (and noted) instead of guessed.
    """
    extractors: dict[str, Callable[[Mapping[str, Any]], Any]] = {
        "lattice": lambda row: row.get("lattice"),
        "dtype": lambda row: row.get("dtype"),
        "seed": lambda row: row.get("seed"),
        "mass": lambda row: row.get("mass"),
        "mpi_size": lambda row: row.get("ranks"),
    }
    bound: dict[str, Any] = {}
    disagreed: list[str] = []
    for name, extract in extractors.items():
        values: list[Any] = []
        for row in rows:
            value = extract(row)
            if value is None:
                continue
            values.append(tuple(value) if name == "lattice" else value)
        distinct = set(values)
        if not distinct:
            continue
        if len(distinct) > 1:
            disagreed.append(name)
            continue
        only = distinct.pop()
        bound[name] = list(only) if name == "lattice" else only
    if disagreed:
        notes.append("cache probe: rows disagree on "
                     + ", ".join(sorted(disagreed))
                     + "; those fields are not bound to the cache identity")
    return bound or None


def _cache_prepare(options: FullModeOptions, rows: Sequence[Mapping[str, Any]],
                   notes: list[str]) -> list[dict[str, Any]]:
    """Prepare the cache location and probe an existing artifact with ``cache``.

    The cache module owns the Legacy-MG cache identity and manifest
    (``pyqcu.test.native-mg-cache.v1``); this mode does **not** fabricate a
    cache - building one needs a native builder.  What it does is create the
    directory, record the layout the rows expect, and, when ``--cache-dir``
    names an existing artifact, run :func:`cache.cache_status` (a non-raising
    probe) so a stale, damaged or wrong-configuration cache is visible *before*
    a solve.  The probe verifies the file SHA256 and is bound to every identity
    field the rows agree on (:func:`_cache_row_binding`).
    """
    entries: list[dict[str, Any]] = []
    owner = ledger_owner()
    cache = options.paths.cache
    if not cache.is_file() and owner:
        cache.mkdir(parents=True, exist_ok=True)
    manifest = cache if cache.is_file() else cache / "index.json"
    payload: dict[str, Any] = {
        "cache": str(cache),
        "rows": int(len(rows)),
        "dtype": sorted({str(row.get("dtype")) for row in rows} - {"None"}),
        "lattices": sorted({str(row.get("lattice")) for row in rows}
                           - {"None"}),
    }
    # What the rows ask the cache for; recorded even when no artifact is probed
    # so a later lattice/dtype mismatch can be seen in the index.
    bound = _cache_row_binding(rows, notes)
    payload["cache_require"] = bound
    module, error = load_mode_module("cache")
    if module is None:
        if error is not None:
            entries.append(_mode_entry(
                "cache", manifest, status="FAIL",
                reason=f"cache module failed to import: {error}"))
            return entries
        notes.append("cache module absent: only the cache directory was created")
        payload["hook"] = None
        try:
            write_ledger_json(manifest, payload, owner=owner)
        except (OSError, ValueError) as failure:
            entries.append(_mode_entry(
                "cache", manifest, status="FAIL",
                reason=(f"cannot write cache manifest: "
                        f"{type(failure).__name__}: {failure}")))
            return entries
        entries.append(_mode_entry(
            "cache", manifest, status="PASS", reason=None,
            bytes=(manifest.stat().st_size if manifest.is_file() else None)))
        return entries

    payload["hook"] = "cache"
    payload["schema"] = getattr(module, "MANIFEST_SCHEMA", None)
    payload["identity_fields"] = list(
        getattr(module, "REQUIRED_IDENTITY_FIELDS", ()))
    default_dir = getattr(module, "default_cache_dir", None)
    if callable(default_dir):
        try:
            payload["default_cache_dir"] = str(default_dir())
        except Exception as failure:  # noqa: BLE001 - informational only
            notes.append(f"cache.default_cache_dir failed: "
                         f"{type(failure).__name__}: {failure}")
    status = "PASS"
    reason: str | None = None
    status_fn = getattr(module, "cache_status", None)
    if cache.is_file() and callable(status_fn):
        try:
            usable, message = status_fn(cache, require=bound)
        except Exception as failure:  # noqa: BLE001 - reported
            usable, message = False, f"{type(failure).__name__}: {failure}"
        payload["ready"] = bool(usable)
        payload["cache_status"] = message or None
        if not usable:
            status = "FAIL"
            reason = f"cache artifact rejected: {message or 'unusable'}"
    elif cache.is_file():
        payload["ready"] = None
        notes.append("cache module has no cache_status(): artifact not probed")
    else:
        payload["ready"] = None
    try:
        if not cache.is_file():
            write_ledger_json(manifest, payload, owner=owner)
    except (OSError, ValueError) as failure:
        entries.append(_mode_entry(
            "cache", manifest, status="FAIL",
            reason=(f"cannot write cache manifest: "
                    f"{type(failure).__name__}: {failure}")))
        return entries
    entries.append(_mode_entry("cache", manifest, status=status, reason=reason,
                               bytes=(manifest.stat().st_size
                                      if manifest.is_file() else None)))
    return entries


def _io_evidence(notes: list[str]) -> dict[str, Any] | None:
    """``{"module","h5py","h5py_version"}`` for a mode that already imports io."""
    module, error = load_mode_module("io")
    if module is None:
        if error is not None:
            notes.append(f"io module failed to import: {error}")
            return {"module": False, "error": error}
        return {"module": False}
    evidence: dict[str, Any] = {"module": True}
    available = getattr(module, "h5py_available", None)
    if callable(available):
        try:
            evidence["h5py"] = bool(available())
        except Exception as failure:  # noqa: BLE001 - informational
            notes.append(f"io.h5py_available failed: "
                         f"{type(failure).__name__}: {failure}")
    version = getattr(module, "h5py_version", None)
    if callable(version) and evidence.get("h5py"):
        try:
            evidence["h5py_version"] = version()
        except Exception:  # noqa: BLE001 - informational
            pass
    return evidence


def _prepare_inputs(options: FullModeOptions, test: "Test", *,
                    prewarm: bool,
                    notes: list[str]) -> list[dict[str, Any]]:
    """Input entries for ``prepare``/``prewarm`` (gauge, source, cache)."""
    entries = [
        _input_entry("gauge", options.gauge_file, lattice=test.lattice,
                     dtype=test.dtype, seed=test.seed, prewarm=prewarm),
        _input_entry("source", options.source_file, lattice=test.lattice,
                     dtype=test.dtype, seed=test.seed, prewarm=prewarm),
    ]
    if not prewarm:
        # The selected rows carry the lattice/dtype/seed the cache probe binds
        # to the artifact; without them the probe would accept a cache built
        # for another configuration.
        entries.extend(_cache_prepare(options, build_rows(test, options), notes))
    return entries


def _mode_prepare(options: FullModeOptions, test: "Test",
                  on_entry: Callable[[dict[str, Any]], None] | None,
                  notes: list[str]) -> tuple[dict[str, Any], int]:
    """``--mode prepare``: create the layout, verify inputs, write the cache.

    Every rank validates the inputs (a per-node page-cache warm is exactly what
    a two-node run wants), but only rank 0 writes the layout, the cache
    manifest and ``reports/prepare.json`` - see :func:`ledger_owner`.
    """
    owner = ledger_owner()
    if owner:
        options.paths.ensure()
    entries = _prepare_inputs(options, test, prewarm=False, notes=notes)
    for entry in entries:
        if on_entry is not None:
            on_entry(entry)
    counts, aggregate = _status_summary(
        entry.get("status") for entry in entries)
    document = _mode_document(
        "prepare", options, status=aggregate, counts=counts, inputs=entries,
        notes=notes, extra={"io": _io_evidence(notes),
                            "native": _native_probe(options, notes)})
    document["duration_seconds"] = 0.0
    try:
        write_ledger_json(options.paths.reports / "prepare.json", document,
                          owner=owner)
    except (OSError, ValueError) as error:
        notes.append(f"cannot write reports/prepare.json: {error}")
    return document, _mode_exit(aggregate)


def _mode_prewarm(options: FullModeOptions, test: "Test",
                  on_entry: Callable[[dict[str, Any]], None] | None,
                  notes: list[str]) -> tuple[dict[str, Any], int]:
    """``--mode prewarm``: read each requested input once (page-cache warm).

    As in ``prepare``, the read happens on every rank (each node warms its own
    page cache) while only rank 0 writes ``reports/prewarm.json``.
    """
    owner = ledger_owner()
    if owner:
        options.paths.ensure()
    entries = _prepare_inputs(options, test, prewarm=True, notes=notes)
    for entry in entries:
        if on_entry is not None:
            on_entry(entry)
    counts, aggregate = _status_summary(
        entry.get("status") for entry in entries)
    document = _mode_document(
        "prewarm", options, status=aggregate, counts=counts, inputs=entries,
        notes=notes, extra={"io": _io_evidence(notes),
                            "native": _native_probe(options, notes)})
    try:
        write_ledger_json(options.paths.reports / "prewarm.json", document,
                          owner=owner)
    except (OSError, ValueError) as error:
        notes.append(f"cannot write reports/prewarm.json: {error}")
    return document, _mode_exit(aggregate)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def _default_executor(test: "Test",
                      options: FullModeOptions) -> Callable[[Case], Result]:
    """The built-in executor: one case through the normal ``Test.run`` path.

    A full mode is an orchestration layer, not a second runner: repeat/warmup/
    timeout/seed and the PASS/FAIL judgement stay the ones of a plain run, so a
    row's verdict means exactly what the equivalent single-case run means.
    """
    def execute(case: Case) -> Result:
        single = replace(test, cases=(case,))
        report = single.run(
            repeat=int(options.repeat),
            warmup=int(options.warmup),
            seed=None,
            timeout=options.timeout,
            fail_fast=False,
            allow_all_skipped=True,
        )
        if not report.cases:
            raise RuntimeError(f"case {case.tag!r} produced no result")
        return report.cases[0]

    return execute


def _copy_solution(row: dict[str, Any], paths: ModePaths) -> str | None:
    """Copy a runner-produced ``solution_path`` into ``--solutions``.

    A runner that reports a solution file is honoured; the ledger never
    fabricates one.  When the copy fails, or when this process is a replica
    rank (the solution directory is owned by rank 0, see :func:`ledger_owner`),
    the original path is recorded - which keeps the evidence traceable instead
    of dropping it, and keeps 32 ranks from copying the same file at once.
    """
    metrics = row.get("metrics") or {}
    candidate = metrics.get("solution_path") if isinstance(metrics, Mapping) else None
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    source = absolute_path(candidate)
    if not source.is_file() or not ledger_owner():
        return str(source)
    try:
        target = paths.solutions / f"{row['run_id']}{source.suffix or '.bin'}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError:
        return str(source)
    return str(target)


def _apply_result(row: dict[str, Any], result: Result,
                  paths: ModePaths) -> dict[str, Any]:
    """Copy one :class:`Result` into its ledger row (verdict + evidence)."""
    row.update({
        "status": result.status,
        "seconds": result.seconds,
        "metrics": json_safe(result.metrics),
        "failed": list(result.failed),
        "reason": result.reason,
        "skip_token": result.skip_token,
        "mpi_size": result.mpi_size,
    })
    row["solution_path"] = _copy_solution(row, paths)
    return row


#: Keys a driver may return that carry harness meaning; everything else is an
#: observation and lands in ``row["metrics"]``.
_DRIVER_ROW_KEYS = ("status", "reason", "skip_token", "failed", "seconds")

#: Statuses an observation-only driver may *declare*.  ``PASS``/``FAIL`` are a
#: verdict, and a verdict belongs to ``Case.expect`` (see :func:`_judge`), so a
#: driver that returns one is an ``ERROR`` instead of a trusted result.
_DRIVER_STATUSES = ("SKIP", "ERROR", "TIMEOUT")


def _apply_driver(row: dict[str, Any], observations: Mapping[str, Any],
                  case: Case | None) -> dict[str, Any]:
    """Merge one driver result into its row and let the harness judge it.

    ``pyqcu.test.driver.run_row`` is a *runner* in the full-mode pipeline: it
    reports what it observed and never decides ``PASS``/``FAIL``.  What it may
    declare is an execution status (``SKIP``/``ERROR``/``TIMEOUT``, each with a
    non-empty reason); a mapping without a status is judged here with the case's
    own ``expect`` gates, exactly like a plain case run, so a native row cannot
    pass by returning numbers nobody checked.

    Observations are merged into ``row["metrics"]`` (the row fields stay at the
    top level) because ``--mode validate`` re-checks a ``PASS`` row against the
    gate keys its case declares - a driver result dumped at the top level would
    make every native ``PASS`` unverifiable.
    """
    declared = observations.get("status")
    text = None if declared is None else str(declared).strip().upper()
    if text is not None and text not in _DRIVER_STATUSES:
        row.update(
            status="ERROR",
            reason=(f"driver.run_row returned a verdict (status={text!r}); a "
                    "driver reports observations and Case.expect decides "
                    "PASS/FAIL") if text in ("PASS", "FAIL") else
                   (f"driver.run_row returned status {declared!r}; a driver may "
                    f"declare only {', '.join(_DRIVER_STATUSES)}"))
        return row
    for key in _DRIVER_ROW_KEYS[1:]:
        if key in observations:
            row[key] = json_safe(observations[key])
    metrics = dict(row.get("metrics") or {})
    metrics.update({str(key): json_safe(value)
                    for key, value in observations.items()
                    if key not in _DRIVER_ROW_KEYS})
    row["metrics"] = metrics
    if text is not None:
        row["status"] = text
        if not str(row.get("reason") or "").strip():
            row.update(status="ERROR",
                       reason=f"driver.run_row reported {text} without a reason")
        return row
    if case is None:
        row.update(status="ERROR",
                   reason=f"tag {row.get('tag')!r} is not in the selected matrix")
        return row
    try:
        failed = evaluate_expect(case.expect, observations, dtype=case.dtype)
    except ValueError as error:
        row.update(status="ERROR", reason=f"invalid expect gate: {error}")
        return row
    if failed:
        row.update(status="FAIL", failed=list(failed), reason="; ".join(failed))
    else:
        row.update(status="PASS", failed=[], reason=None)
    return row


def _solve_row(row: dict[str, Any], case: Case | None, *, test: "Test",
               options: FullModeOptions, execute: Callable[[Case], Result] | None,
               paths: ModePaths,
               reuse: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute one row and return it with a real verdict.

    ``reuse`` is the already-written row to resume, decided for the **whole**
    world by :func:`agreed_resume_row`; a row that is reused is not executed at
    all.  The decision is deliberately not taken here: a rank that resumed a row
    on its own would skip the collectives every other rank enters (the driver's
    pre-flight and outcome gather), which is exactly the 32-rank hang the audit
    records as P0-5.
    """
    if reuse is not None:
        resumed = dict(reuse)
        resumed["resumed"] = True
        return resumed
    row["resumed"] = False
    if case is None:
        row.update(status="ERROR",
                   reason=f"tag {row.get('tag')!r} is not in the selected matrix")
        return row

    driver = mode_hook("driver", "run_row")
    if driver is not None and _accepts_keyword(driver, "row"):
        ok, value, error = call_hook(
            driver, row=dict(row), options=options.as_dict(),
            paths=paths.as_dict(), tag=row["tag"])
        if not ok:
            row.update(status="ERROR",
                       reason=f"driver.run_row failed: {error}")
            return row
        if value is None:
            # ``None`` is the driver saying "this row is not mine" (a tier-0
            # case of a backend it does not drive): the built-in executor below
            # runs it exactly as it did before the driver existed.  It is a
            # routing decision, never a substitute for a requested input.
            pass
        elif not isinstance(value, Mapping):
            row.update(
                status="ERROR",
                reason=(f"driver.run_row returned {type(value).__name__}, "
                        "expected a mapping of observations or None"))
            return row
        else:
            return _apply_driver(row, value, case)

    executor = execute if execute is not None else _default_executor(test, options)
    try:
        result = executor(case)
    except Exception as error:  # noqa: BLE001 - a row never escapes the ledger
        row.update(status="ERROR",
                   reason=f"{type(error).__name__}: {error}")
        return row
    if isinstance(result, Result):
        return _apply_result(row, result, paths)
    if isinstance(result, Mapping):
        status = result.get("status")
        if status in policy.STATUSES:
            row.update(json_safe(dict(result)))
            if row["status"] == "SKIP" \
                    and not str(row.get("reason") or "").strip():
                row.update(status="ERROR",
                           reason="executor skipped without a reason")
            return row
    row.update(status="ERROR",
               reason=(f"executor returned {type(result).__name__}, expected a "
                       "Result or a mapping carrying status"))
    return row


def _gather_row_verdict(entry: dict[str, Any], *,
                        notes: list[str] | None = None) -> dict[str, Any]:
    """Reduce one solved row to the worst verdict across the MPI ranks.

    Every rank executes the row and calls :func:`gather_rank_outcome` once; the
    row keeps the worst status and the worst evidence (largest iteration count,
    largest independent residual, ``converged`` only if every rank converged).
    This is the harness-side counterweight to the driver's own gather: a tier-0
    row is executed by the built-in executor instead of the driver and can
    diverge across ranks too, and a rank that computed ``FAIL`` must not be
    overwritten by rank 0's ``PASS``.

    The collective also acts as the row's barrier: rank 0 writes
    ``rows/<run_id>.json`` only after it returns, i.e. after every rank has
    reported, and ``--fail-fast`` sees the same verdict on every rank (so every
    rank stops at the same row instead of leaving the others in a collective).
    """
    metrics = dict(entry.get("metrics") or {})
    merged = gather_rank_outcome(
        metrics, status=entry.get("status"),
        skip_token=entry.get("skip_token"), reason=entry.get("reason"),
        phase="verdict")
    for key in ("rank_phase", "rank_gather", "rank_gather_reason",
                "rank_disagreement", "rank_status", "rank_worst", "mpi_rank",
                "mpi_size", "rank_outcomes"):
        if key in merged:
            metrics[key] = merged[key]
    for key in ("iterations", "relative_residual", "converged"):
        if merged.get(key) is not None:
            metrics[key] = merged[key]
    entry["metrics"] = metrics
    if merged.get("rank_gather") == "unavailable" and notes is not None:
        notes.append(
            f"row {entry.get('run_id')}: multi-rank reduction unavailable "
            f"({merged.get('rank_gather_reason') or 'no communicator'}); the "
            "verdict of this row is rank-local")
    worst = str(merged.get("rank_status") or "")
    if worst and worst != entry.get("status"):
        # The disagreement is written into the row itself, not only into its
        # metrics: a reader of rows/<run_id>.json must see the verdict the run
        # actually earned.
        entry["status"] = worst
        entry["reason"] = (merged.get("rank_reason")
                           or f"rank verdict {worst}")
        entry["skip_token"] = (merged.get("rank_skip_token")
                               if worst == "SKIP" else None)
        if worst != "FAIL":
            entry["failed"] = []
    return entry


def _mode_solve(options: FullModeOptions, test: "Test",
                rows: list[dict[str, Any]], *,
                execute: Callable[[Case], Result] | None,
                on_entry: Callable[[dict[str, Any]], None] | None,
                notes: list[str]) -> tuple[dict[str, Any], int]:
    """``--mode solve``: execute every row, persist it, append the ledger.

    One logical row over N processes: every rank executes the row and reports a
    compact record (:func:`_gather_row_verdict`), the row keeps the worst
    verdict, and only rank 0 (:func:`ledger_owner`) writes it
    (:func:`write_row`) - so a 32-process run produces exactly one row file,
    one report copy and one CSV line per row.

    ``--resume`` is part of the same protocol: the row is reused only when
    every rank can resume it (:func:`agreed_resume_row`), so no rank can skip
    the collectives the others enter by trusting a different copy of the ledger.
    """
    owner = ledger_owner()
    if owner:
        options.paths.ensure()
    cases = {case.tag: case for case in test.cases}
    executed: list[dict[str, Any]] = []
    stopped = False
    for index, row in enumerate(rows, 1):
        # ``--resume`` is one decision for the whole world: a replica that
        # resumed a row on its own would skip the driver's collectives and hang
        # every other rank (P0-5), so the decision is gathered before the row.
        reuse: dict[str, Any] | None = None
        resume_evidence: dict[str, Any] = {}
        if options.resume:
            reuse, resume_evidence = agreed_resume_row(
                options.paths, row, skip_failed=options.skip_failed)
        entry = _solve_row(row, cases.get(str(row.get("tag"))), test=test,
                           options=options, execute=execute,
                           paths=options.paths, reuse=reuse)
        if resume_evidence:
            metrics = dict(entry.get("metrics") or {})
            metrics.update(resume_evidence)
            entry["metrics"] = metrics
        entry.setdefault("run_label", options.run_label)
        entry = _gather_row_verdict(entry, notes=notes)
        write_row(options.paths, entry, owner=owner)
        executed.append(entry)
        if on_entry is not None:
            on_entry(entry)
        if options.fail_fast and entry.get("status") in ("FAIL", "ERROR",
                                                         "TIMEOUT"):
            stopped = True
            notes.append(
                f"--fail-fast stopped after {index}/{len(rows)} row(s)")
            break
    counts, aggregate = _status_summary(
        entry.get("status") for entry in executed)
    if not executed:
        aggregate = "SKIP"
        notes.append("no row was executed")
    document = _mode_document(
        "solve", options, status=aggregate, counts=counts, rows=executed,
        notes=notes,
        extra={"planned": len(rows), "executed": len(executed),
               "fail_fast_stopped": stopped,
               "ledger_writer": owner,
               "native": _native_probe(options, notes)})
    if options.validate and owner:
        validation, validation_exit = validate_ledger(
            options.paths, test.cases, notes=notes)
        document["validation"] = validation
        if validation_exit != policy.EXIT_CODES["pass"]:
            document["status"] = "FAIL"
    all_skipped = bool(executed) and all(
        entry.get("status") == "SKIP" for entry in executed)
    code = policy.exit_code(
        str(document["status"]), all_skipped=all_skipped,
        allow_all_skipped=options.allow_all_skipped)
    return document, code


# ---------------------------------------------------------------------------
# merge and validate
# ---------------------------------------------------------------------------
def merge_ledger(paths: ModePaths, *, notes: list[str] | None = None,
                 limit: int | None = None) -> tuple[dict[str, Any], int]:
    """``--mode merge``: rebuild the CSV and the merged document from rows.

    The row files are the source of truth; the CSV is a derived, rewritable
    view.  A merge of an empty ledger is a ``SKIP`` (exit 1): "nothing to
    merge" must not look like a green merge.
    """
    notes = [] if notes is None else notes
    rows, problems = load_rows(paths, limit=limit)
    counts, aggregate = _status_summary(row.get("status") for row in rows)
    document = {
        "package": policy.PACKAGE,
        "version": _PACKAGE_VERSION,
        "mode": "merge",
        "status": aggregate if rows else "SKIP",
        "counts": counts,
        "rows": {
            "dir": str(paths.rows),
            "count": len(rows),
            "csv": str(paths.csv),
            "items": [json_safe(row) for row in rows],
        },
        "problems": list(problems),
        "notes": [str(note) for note in notes],
        "paths": paths.as_dict(),
    }
    if not rows:
        notes.append(
            f"no row file under {paths.rows}: nothing to merge")
        document["status"] = "SKIP"
        return document, policy.EXIT_CODES["fail"]
    owner = ledger_owner()
    if not owner:
        # A replica rank of a multi-process ``--mode run`` must not rewrite the
        # CSV 31 more times: the merged view is rank 0's artefact.
        document["duration_seconds"] = 0.0
        return document, (policy.EXIT_CODES["fail"] if problems
                          else policy.EXIT_CODES["pass"])
    try:
        write_csv(paths.csv, rows)
        write_json_atomic(paths.reports / "merged.json", document)
    except (OSError, ValueError) as error:
        notes.append(f"cannot write the merged ledger: "
                     f"{type(error).__name__}: {error}")
        document["status"] = "FAIL"
        return document, policy.EXIT_CODES["fail"]
    document["duration_seconds"] = 0.0
    return document, (policy.EXIT_CODES["fail"] if problems
                      else policy.EXIT_CODES["pass"])


def write_csv(path: Any, rows: Sequence[Mapping[str, Any]], *,
              columns: Sequence[str] = ROW_FIELDS) -> Path:
    """Rewrite a ledger CSV from ``rows`` (atomic; the merge target)."""
    target = absolute_path(path)
    return _write_csv_records(
        target, list(columns),
        [{name: _csv_value(row.get(name)) for name in columns}
         for row in rows])


def validate_rows(rows: Sequence[Mapping[str, Any]], *,
                  cases: Sequence[Case] | None = None,
                  paths: ModePaths | None = None) -> list[str]:
    """Every way a ledger row can lie; empty list means "consistent".

    The checks are the contract of the ledger: a known status, a non-empty
    reason for every non-PASS verdict, the observation a PASS needs for its
    case's ``expect`` gates, a ``run_id`` that still matches the configuration
    digest, and (with ``paths``) a per-row report that agrees with the row.
    """
    problems: list[str] = []
    known = {case.tag: case for case in cases} if cases else {}
    for row in rows:
        run_id = str(row.get("run_id") or "<missing>")

        def bad(message: str) -> None:
            problems.append(f"{run_id}: {message}")

        status = row.get("status")
        if status not in policy.STATUSES:
            bad(f"status {status!r} is not one of "
                f"{', '.join(policy.STATUSES)}")
            continue
        reason = str(row.get("reason") or "").strip()
        if status == "SKIP" and not reason:
            bad("SKIP without a non-empty reason")
        if status in ("FAIL", "ERROR", "TIMEOUT") and not reason \
                and not row.get("failed"):
            bad(f"{status} without a reason or failed gate")
        if status == "PASS":
            case = known.get(str(row.get("tag")))
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                bad("metrics is not an object")
            elif case is not None:
                missing = [key for key in case.expect if key not in metrics]
                if missing:
                    bad(f"PASS without the observation(s) "
                        f"{', '.join(missing)} required by {case.tag}")
        expected = row_id(row, label=str(row.get("run_label") or ""))
        if str(row.get("run_id")) != expected:
            bad(f"run_id does not match its configuration digest ({expected})")
        if paths is not None:
            report = paths.reports / f"{run_id}.json"
            if not report.is_file():
                bad(f"missing per-row report {report}")
            else:
                try:
                    data = json.loads(report.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError) as error:
                    bad(f"unreadable per-row report ({type(error).__name__}: "
                        f"{error})")
                else:
                    if isinstance(data, Mapping) and data.get("status") != status:
                        bad(f"report status {data.get('status')!r} != row "
                            f"status {status!r}")
    return problems


def validate_ledger(paths: ModePaths, cases: Sequence[Case], *,
                    notes: list[str] | None = None) -> tuple[dict[str, Any], int]:
    """``--mode validate``: check rows, reports and the CSV against each other."""
    notes = [] if notes is None else notes
    rows, read_problems = load_rows(paths)
    problems = list(read_problems) + validate_rows(rows, cases=cases, paths=paths)
    row_ids = {str(row.get("run_id")) for row in rows}
    header, records = _read_csv(paths.csv) if paths.csv.is_file() else ([], [])
    if header:
        csv_ids = {str(record.get("run_id")) for record in records}
        missing = sorted(run_id for run_id in row_ids if run_id not in csv_ids)
        if missing:
            problems.append(
                f"{len(missing)} row(s) missing from {paths.csv}: "
                f"{', '.join(missing[:5])}")
        extra = sorted(run_id for run_id in csv_ids
                       if run_id and run_id not in row_ids)
        if extra:
            notes.append(
                f"{len(extra)} CSV row(s) have no row file (a merge will drop "
                f"them): {', '.join(extra[:5])}")
    else:
        notes.append(f"no ledger CSV at {paths.csv} (run --mode merge)")
    counts, aggregate = _status_summary(row.get("status") for row in rows)
    status = ("FAIL" if problems else
              ("PASS" if rows else "SKIP"))
    if not rows:
        notes.append(f"no row file under {paths.rows}: nothing to validate")
    document = {
        "package": policy.PACKAGE,
        "version": _PACKAGE_VERSION,
        "mode": "validate",
        "status": status,
        "counts": counts,
        "aggregate": aggregate,
        "problems": list(problems),
        "notes": [str(note) for note in notes],
        "paths": paths.as_dict(),
        "rows": {
            "dir": str(paths.rows),
            "count": len(rows),
            "csv": str(paths.csv),
            "items": [json_safe(row) for row in rows],
        },
    }
    return document, _mode_exit(status)


# ---------------------------------------------------------------------------
# mode dispatch
# ---------------------------------------------------------------------------
def run_mode(mode: str, test: "Test", options: FullModeOptions, *,
             rows: Sequence[Mapping[str, Any]] | None = None,
             execute: Callable[[Case], Result] | None = None,
             on_entry: Callable[[dict[str, Any]], None] | None = None,
             argv: Sequence[str] | None = None) -> tuple[dict[str, Any], int]:
    """Run one full mode and return ``(payload, exit_code)``.

    ``list``/``prepare``/``prewarm``/``merge``/``validate`` perform no numeric
    case run; ``solve`` executes rows through ``execute`` (or the built-in
    :func:`_default_executor`); ``run`` chains prepare -> prewarm -> solve ->
    merge and returns the solve verdict.  Exit codes stay ``0``/``1``; a usage
    error is raised as :class:`UsageError` and mapped to ``2`` by the CLI.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    ledger_rows = [dict(row) for row in
                   (rows if rows is not None else build_rows(test, options))]
    notes: list[str] = []
    started = time.perf_counter()

    if mode == "list":
        # A listing executes nothing: ``status`` says "the mode completed",
        # while every row keeps ``status=None`` so a plan can never be mistaken
        # for an executed row.  No native/io probe here - ``--mode list`` must
        # stay as light as ``--list``.
        counts = {name: 0 for name in policy.STATUSES}
        if not ledger_rows:
            notes.append("no row selected")
        document = _mode_document(
            "list", options,
            status="PASS" if ledger_rows else "SKIP",
            counts=counts, rows=ledger_rows, notes=notes)
        code = (policy.EXIT_CODES["pass"] if ledger_rows
                else policy.EXIT_CODES["fail"])
    elif mode == "prepare":
        document, code = _mode_prepare(options, test, on_entry, notes)
    elif mode == "prewarm":
        document, code = _mode_prewarm(options, test, on_entry, notes)
    elif mode == "solve":
        document, code = _mode_solve(options, test, ledger_rows,
                                     execute=execute, on_entry=on_entry,
                                     notes=notes)
    elif mode == "merge":
        document, code = merge_ledger(options.paths, notes=notes)
    elif mode == "validate":
        document, code = validate_ledger(options.paths, test.cases, notes=notes)
    else:  # mode == "run"
        document, code = _mode_run(options, test, ledger_rows,
                                   execute=execute, on_entry=on_entry,
                                   notes=notes)

    document["duration_seconds"] = round(time.perf_counter() - started, 6)
    document["argv"] = None if argv is None else list(argv)
    document["modules"] = module_availability()
    document["exit_code"] = code
    return document, code


def _mode_run(options: FullModeOptions, test: "Test",
              rows: list[dict[str, Any]], *,
              execute: Callable[[Case], Result] | None,
              on_entry: Callable[[dict[str, Any]], None] | None,
              notes: list[str]) -> tuple[dict[str, Any], int]:
    """``--mode run``: the whole pipeline, stopping before solve on bad input."""
    prepare_document, _ = _mode_prepare(options, test, on_entry, notes)
    prewarm_document, _ = _mode_prewarm(options, test, on_entry, notes)
    blocked = [
        entry for entry in prewarm_document.get("inputs", [])
        if entry.get("status") in ("FAIL", "ERROR", "TIMEOUT")
    ]
    if blocked:
        reasons = "; ".join(
            f"{entry.get('role')}: {entry.get('reason')}" for entry in blocked)
        notes.append(f"solve skipped: input preparation failed ({reasons})")
        document = _mode_document(
            "run", options, status="FAIL", rows=[],
            counts={name: 0 for name in policy.STATUSES}, notes=notes,
            extra={"steps": {"prepare": prepare_document,
                             "prewarm": prewarm_document, "solve": None,
                             "merge": None}})
        return document, policy.EXIT_CODES["fail"]
    solve_document, code = _mode_solve(
        options, test, rows, execute=execute, on_entry=on_entry, notes=notes)
    merge_document, _ = merge_ledger(options.paths, notes=notes)
    document = _mode_document(
        "run", options, status=solve_document["status"],
        counts=solve_document["counts"],
        rows=solve_document.get("rows", {}).get("items", []), notes=notes,
        extra={
            "planned": solve_document.get("planned"),
            "executed": solve_document.get("executed"),
            "fail_fast_stopped": solve_document.get("fail_fast_stopped"),
            "steps": {"prepare": prepare_document,
                      "prewarm": prewarm_document,
                      "solve": {key: solve_document.get(key) for key in
                                ("status", "counts", "planned", "executed",
                                 "fail_fast_stopped")},
                      "merge": {key: merge_document.get(key) for key in
                                ("status", "counts", "problems")}},
        })
    if options.validate:
        validation, validation_exit = validate_ledger(
            options.paths, test.cases, notes=notes)
        document["validation"] = validation
        if validation_exit != policy.EXIT_CODES["pass"]:
            document["status"] = "FAIL"
            code = policy.EXIT_CODES["fail"]
    return document, code


def grid_product(values: Iterable[int]) -> int:
    """Product of a grid (``(1, 1, 1, 1) -> 1``); empty product is 1."""
    total = 1
    for value in values:
        total *= int(value)
    return total


def _checked(validate: Callable[[Any], Any], value: Any, flag: str) -> Any:
    """Run a policy validator, re-raising ``ValueError`` as :class:`UsageError`."""
    try:
        return validate(value)
    except ValueError as error:
        raise UsageError(f"{flag}: {error}") from error


def _tag_matches(case: Case, pattern: str) -> bool:
    """True when ``pattern`` selects ``case`` (tag, segment, backend or prefix)."""
    if not pattern:
        return True
    if case.tag == pattern or case.backend == pattern:
        return True
    if case.tag.startswith(f"{pattern}/"):
        return True
    return pattern in case.tag.split("/")


def _split_skip(reason: Any, token: str | None) -> tuple[str, str | None]:
    """Split ``"token: reason"`` into ``(reason, token)`` when possible."""
    text = str(reason or "").strip()
    if token is None and ":" in text:
        head, _, tail = text.partition(":")
        if policy.is_skip_token(head.strip()) and tail.strip():
            return tail.strip(), head.strip()
    return text, (token if policy.is_skip_token(token) else None)


def _observation_skip(observation: Mapping[str, Any]) -> tuple[bool, str, str | None]:
    """Detect a skip request in an observation mapping."""
    skip = observation.get("skip")
    if isinstance(skip, str):
        reason, token = _split_skip(skip, observation.get("token"))
        return True, reason, token
    if skip is True:
        reason, token = _split_skip(observation.get("reason"),
                                    observation.get("token"))
        return True, reason, token
    declared = observation.get("status")
    if isinstance(declared, str) and declared.strip().upper() == "SKIP":
        reason, token = _split_skip(observation.get("reason"),
                                    observation.get("token"))
        return True, reason, token
    return False, "", None


def _judge(case: Case, options: Options,
           observation: Any) -> tuple[str, str | None, tuple[str, ...], str | None]:
    """Turn a runner's observation into ``(status, reason, failed, token)``."""
    if not isinstance(observation, Mapping):
        return (
            "ERROR",
            f"runner returned {type(observation).__name__}, expected a mapping "
            "of observations",
            (),
            None,
        )
    requested, reason, token = _observation_skip(observation)
    if requested:
        if not reason:
            return (
                "ERROR",
                "skip without a reason: every SKIP needs a non-empty reason",
                (),
                token,
            )
        return "SKIP", reason, (), token
    declared = observation.get("status")
    if isinstance(declared, str) and declared.strip().upper() in ("PASS", "FAIL"):
        return (
            "ERROR",
            f"runner returned a verdict (status={declared!r}); runners report "
            "observations, Case.expect decides PASS/FAIL",
            (),
            None,
        )
    try:
        failed = evaluate_expect(case.expect, observation, dtype=options.dtype,
                                 atol=options.atol)
    except ValueError as error:
        return "ERROR", f"invalid expect gate: {error}", (), None
    if failed:
        return "FAIL", "; ".join(failed), failed, None
    return "PASS", None, (), None


def _metrics(observation: Any) -> dict[str, Any]:
    """Observation payload without the harness-reserved keys."""
    if not isinstance(observation, Mapping):
        return {}
    return {key: value for key, value in observation.items()
            if key not in _RESERVED_OBSERVATIONS}


def run_case(case: Case, options: Options, *, mpi_size: int = 1) -> Result:
    """Execute one case: resolve, time, judge, and never propagate.

    Any exception from resolution or execution becomes an ``ERROR`` result;
    a ``TimeoutError`` becomes ``TIMEOUT``.  The only way out of this function
    is a :class:`Result`.
    """
    started = time.perf_counter()
    observation: Any = None
    status = "PASS"
    reason: str | None = None
    failed: tuple[str, ...] = ()
    skip_token: str | None = None
    timer: _Timer | None = None
    # ``Options.timeout`` wins (``Test.run`` copies the per-case budget into
    # it); ``Case.timeout`` is the fallback so a direct ``run_case`` call
    # honours the case declaration too.
    timeout = options.timeout if options.timeout is not None else case.timeout
    try:
        runner = resolve_runner(case.runner)
        timer = _arm_timeout(timeout)
        try:
            observation = _invoke_runner(runner, options)
        finally:
            _disarm_timeout(timer)
    except Skip as error:
        reason, skip_token = _split_skip(error.reason, error.token)
        if not reason:
            status, reason = "ERROR", "skip without a reason"
        else:
            status = "SKIP"
    except TimeoutError:
        status = "TIMEOUT"
        reason = (f"exceeded {float(timeout):g}s wall-clock budget"
                  if timeout is not None else "wall-clock timeout")
    except Exception as error:  # noqa: BLE001 - a case never escapes the harness
        status = "ERROR"
        reason = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started

    if status == "PASS":
        status, reason, failed, skip_token = _judge(case, options, observation)
    if (status in ("PASS", "FAIL") and timeout is not None
            and timer is None and elapsed > float(timeout)):
        # No signal timer was available (non-POSIX or worker thread): the
        # elapsed wall clock is the only timeout evidence we have.
        status = "TIMEOUT"
        reason = f"exceeded {float(timeout):g}s wall-clock budget"
        failed = ()

    metrics = _metrics(observation)
    metrics["elapsed_seconds"] = round(elapsed, 6)
    reported = None
    if isinstance(observation, Mapping):
        raw = observation.get("seconds")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            reported = float(raw)
    seconds = reported if reported is not None else elapsed
    return Result(
        tag=case.tag,
        backend=case.backend,
        kind=case.kind,
        status=status,
        seconds=seconds,
        metrics=metrics,
        failed=failed,
        reason=reason,
        lattice=case.lattice,
        dtype=case.dtype,
        seed=options.seed,
        mpi_size=mpi_size,
        tier=case.tier,
        runner=case.runner,
        skip_token=skip_token,
    )
