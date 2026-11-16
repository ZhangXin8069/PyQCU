"""KUPL CPU backend runners for the formal :mod:`pyqcu.test` port.

The module is import-light: only the standard library is imported at module
import time.  ``torch``, ``mpi4py`` (reached through ``pyqcu.kupl.define`` and
``pyqcu.cuda.define``) and the ctypes KUPL binding are imported inside the
runner functions, so ``python -m pyqcu.test --list`` never pays for a backend
import.

Contract
--------
Each ``run_*`` function returns a *plain observation dict*: JSON-safe values
and timing only.  It never decides PASS/FAIL; ``pyqcu.test.units`` declares the
``expect`` mapping for each case and ``pyqcu.test.common`` evaluates it.
Runners accept either no argument, one positional ``case`` (an object, mapping
or lattice sequence carrying ``lattice``/``dtype``/``seed``), or explicit
keyword overrides, so the harness may call them in any of those shapes.

Run options
-----------
Every runner accepts the expanded option set and applies it to the private
native ABI controls (``params``/``argv``) before the first transaction:

======================  ====================================================
option                  effect for this backend
======================  ====================================================
``lattice``             ``params[_LAT_*_]``; four positive even extents
``dtype``               ``params[_DATA_TYPE_]``; ``c64``/``c128`` (also the
                        ``complex64``/``complex128`` spellings)
``mass`` / ``kappa``    ``argv[_MASS_]``; ``kappa = 1/(2m+8)``, the two are
                        mutually validated so ``kappa`` may be given alone
``csw``                 ``argv[_CSW_]``; the clover coefficient.  The Wilson
                        dslash/solver entry points ignore it, the clover
                        dslash entry point consumes it
``gauge_sigma``         ``argv[_SIGMA_]`` and the disorder of the generated
                        gauge field
``gauge``               ``"random"`` (fixed seeded SU(3) field), ``"unit"``
                        (identity links) or a full/compact gauge tensor
``source``              ``"random"``, ``"point"``, ``"unit"`` or a full/
                        compact/single-parity fermion tensor
``seed``                ``params[_SEED_]``; derives the gauge/source seeds
``lattice_grid``        ``params[_GRID_*_]``; ``prod(lattice_grid)`` must equal
                        the MPI world size
``mg_grid``             validated and recorded; these runners are not
                        multigrid drivers, so it never reaches the ABI
``max_iter``            ``params[_MAX_ITER_]``
``atol``                ``argv[_ATOL_]``; per-dtype default when omitted (the
                        harness passes ``--atol`` through, so the flag drives
                        the native tolerance as well as the compare gate)
``device``              ``cpu`` only (``cpu``/``cpu:N``/``host``); anything
                        else is an environmental skip
``dslash`` / ``solver`` ``auto`` (the runner's own operator), or the matching
                        name; a contradictory request is a usage error
======================  ====================================================

Every accepted option is reported back.  The ``*_applied`` observations prove
that the value really reached the ABI control array (``argv_mass``,
``argv_csw``, ``params_grid`` ...); that is how "the runner honors ``mass``" is
demonstrated for the Wilson dslash, whose native entry point is a bare hopping
sum that does not depend on the mass on its own.

An option that is not passed explicitly is read from the ``options``/``ctx``
container when that object exposes it (so an expanded harness
``pyqcu.test.common.Options`` needs no runner edit), then falls back to the
runner default.  Unknown trailing keyword arguments are recorded under
``extra_options`` instead of being silently dropped.

Lifecycle
---------
Every native call is one ordinary KUPL transaction::

    select a free slot in set_ptrs[0:30)
        -> applyInitQcu
        -> operation (and any same-slot getters)
        -> applyEndQcu
        -> _SET_INDEX_ += 1 (wrapped inside the ordinary slot window)

``_SET_INDEX_`` is never advanced before ``applyEndQcu``.  All ABI control
arrays (``params``/``argv``/``set_ptrs``) belong to a private runtime object;
the module-level arrays in ``pyqcu.cuda.define`` are never passed to the
library, and the runners verify that they are unchanged afterwards.

Skips vs errors
---------------
Only environmental absence is skipped, always with a non-empty reason:

``kupl_library``
    ``libqcu.so`` is missing or cannot be loaded;
``mpi_grid``
    ``prod(lattice_grid)`` does not equal the MPI world size;
``mpi_single_rank``
    the job has more than one rank, so a full-lattice reference cannot be
    validated without halo exchange;
``device_unavailable``
    the requested ``device`` is not a KUPL CPU device.

:class:`Skip` carries a non-empty ``reason``; when the harness exposes its own
skip exception in ``pyqcu.test.common`` (or ``pyqcu.test``) that type is raised
instead, so the harness always maps the case to ``SKIP``.  A token the policy
does not know yet (``mpi_grid``, ``device_unavailable``) is kept in the reason
as a ``"<token>: <reason>"`` prefix so the skip stays self-describing.  Any
other import failure (for example a broken ``mpi4py``) is deliberately *not*
skipped: it propagates so the harness reports ``ERROR`` with the real
exception.

Native failure policy
---------------------
The KUPL C ABI turns a C++ exception into ``abort_with_message`` (``MPI_Abort``
plus ``std::abort``), so a native error terminates the process instead of
raising.  The runners therefore validate every operand shape/dtype/contiguity
and every control value *before* the first native call, and they keep the
solver tolerance inside a range the native solver can accept.
"""

from __future__ import annotations

import ctypes
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn

__all__ = [
    "BACKEND",
    "DEFAULT_MASS",
    "EXPECTED_ABI_SYMBOLS",
    "GAUGE_KINDS",
    "QUICK_LATTICE",
    "SOLVER_CHOICES",
    "SOURCE_KINDS",
    "SUPPORTED_DEVICES",
    "Skip",
    "library_status",
    "load_qcu",
    "mpi_rank",
    "mpi_size",
    "run_abi_symbols",
    "run_dslash_clover",
    "run_dslash_wilson",
    "run_solve_bistabcg",
    "run_solve_cg",
]

BACKEND = "kupl"
#: Correctness lattice of the project (AGENTS.md, `pyqcu/testing/_config.py`).
QUICK_LATTICE = (8, 8, 8, 16)
#: Number of CUDA QCU ABI entry points the KUPL library must mirror.
EXPECTED_ABI_SYMBOLS = 40

DEFAULT_MASS = 0.17
#: Wilson dslash / solver default: no clover term, so the operator is Wilson.
DEFAULT_CSW = 0.0
#: ``run_dslash_clover`` default: a non-zero term, otherwise the case
#: degenerates into the Wilson dslash it is meant to extend.
DEFAULT_CLOVER_CSW = 1.0
DEFAULT_GAUGE_SIGMA = 0.15
DEFAULT_SOLVE_ATOL = 1.0e-6
#: c128 residual requests are tighter because ``policy``'s c128
#: ``native_solve`` gate is 1e-8, which a 1e-6 solve tolerance cannot reach.
DEFAULT_SOLVE_ATOL_C128 = 1.0e-10
#: Smallest solver tolerance a double-precision native solve is asked for.
MIN_SOLVE_ATOL = 1.0e-13
DEFAULT_MAX_ITER = 1000
DEFAULT_SEED = 42
DEFAULT_GAUGE_SEED = 42
DEFAULT_SOURCE_SEED = 7
DEFAULT_REPEAT = 1
DEFAULT_WARMUP = 1
#: One process, one full lattice: the tier-0 KUPL cases are single-rank.
DEFAULT_LATTICE_GRID = (1, 1, 1, 1)
DEFAULT_SOURCE_SITE = (0, 0, 0, 0)
DEFAULT_SOURCE_SPIN = 0
DEFAULT_SOURCE_COLOR = 0

#: ``gauge`` selectors this backend can build without a data file.
GAUGE_KINDS = ("random", "unit")
#: ``source`` selectors this backend can build without a data file.
SOURCE_KINDS = ("random", "point", "unit")
#: ``dslash`` selector values (``auto`` means "the runner's own operator").
DSLASH_CHOICES = ("auto", "wilson", "clover")
#: ``solver`` selector values.
SOLVER_CHOICES = ("auto", "cg", "bistabcg")
#: KUPL is a CPU backend; these device spellings are accepted.
SUPPORTED_DEVICES = ("cpu", "host")

_SLOT_LIMIT = 30  # define._SET_PTRS_COARSE_BASE_
_LIBRARY_TOKEN = "kupl_library"
_RANK_TOKEN = "mpi_single_rank"
_GRID_TOKEN = "mpi_grid"
_DEVICE_TOKEN = "device_unavailable"

_COMPACT_LAYOUT = "compact"
_FULL_LAYOUT = "full"

_SOURCE_ALIASES = {
    "rand": "random",
    "random": "random",
    "gaussian": "random",
    "complex": "random",
    "point": "point",
    "delta": "point",
    "point_source": "point",
    "unit": "unit",
    "ones": "unit",
    "one": "unit",
}
_GAUGE_ALIASES = {
    "rand": "random",
    "random": "random",
    "gaussian": "random",
    "unit": "unit",
    "identity": "unit",
    "ones": "unit",
    "one": "unit",
}
_SOLVER_ALIASES = {
    "auto": "auto",
    # ``all``/``multigrid``/``mg`` are selection choices, not a request for a
    # different solver: ``all`` keeps every solve case and the multigrid
    # spelling never selects a solve case, so both mean "this runner's own
    # solver" instead of a contradiction.
    "all": "auto",
    "multigrid": "auto",
    "mg": "auto",
    "cg": "cg",
    "conjugate_gradient": "cg",
    "conjugategradient": "cg",
    "normal_equations": "cg",
    "bistabcg": "bistabcg",
    "bicgstab": "bistabcg",
    "bistab": "bistabcg",
}
_DSLASH_ALIASES = {
    "auto": "auto",
    # ``--dslash all`` and ``--dslash both`` are "no filter" spellings.
    "all": "auto",
    "both": "auto",
    "wilson": "wilson",
    "wilson_dslash": "wilson",
    "clover": "clover",
    "clover_dslash": "clover",
}
_DTYPE_ALIASES = {
    "c64": "c64",
    "complex64": "c64",
    "csingle": "c64",
    "float32": "c64",
    "single": "c64",
    "c128": "c128",
    "complex128": "c128",
    "cdouble": "c128",
    "float64": "c128",
    "double": "c128",
}


# ---------------------------------------------------------------------------
# Skip handling
# ---------------------------------------------------------------------------


class Skip(Exception):
    """A case cannot run for an environmental reason (mapped to ``SKIP``).

    ``reason`` is always a non-empty human readable string and ``token`` names
    a :data:`pyqcu.test.policy.SKIP_TOKENS` entry.  The class is never used for
    numerical or programming errors.
    """

    def __init__(self, reason: str, token: str | None = None) -> None:
        message = str(reason).strip() or "KUPL backend unavailable"
        super().__init__(message)
        self.reason = message
        self.token = token


_SKIP_NAMES = (
    "Skip",
    "SkipCase",
    "CaseSkipped",
    "CaseSkip",
    "Skipped",
    "SkipTest",
    "SkipError",
    "BackendSkip",
)


def _harness_skip_type() -> type[BaseException] | None:
    """Return the harness skip exception type when it can be discovered.

    ``pyqcu.test.common.Skip`` is preferred (it carries a skip token); the
    lookup is lazy so this module stays importable on its own and importing it
    can never create a cycle with the harness.  When the class is not
    importable, an already-imported harness module is scanned for a skip
    exception, first by well-known name and then by any name containing
    "skip".
    """
    try:
        from .common import Skip as harness_skip  # noqa: F401 - lazy probe
    except Exception:
        harness_skip = None
    if (isinstance(harness_skip, type)
            and issubclass(harness_skip, BaseException)):
        return harness_skip
    for module_name in ("pyqcu.test.common", "pyqcu.test.policy",
                        "pyqcu.test"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute in _SKIP_NAMES:
            candidate = getattr(module, attribute, None)
            if (isinstance(candidate, type)
                    and issubclass(candidate, BaseException)):
                return candidate
        for attribute, candidate in vars(module).items():
            if attribute.startswith("_") or "skip" not in attribute.lower():
                continue
            if (isinstance(candidate, type)
                    and issubclass(candidate, BaseException)):
                return candidate
    return None


def _token_is_known(token: str) -> bool:
    """True when ``token`` is part of the harness skip vocabulary."""
    try:
        from . import policy
    except Exception:
        return False
    try:
        return bool(policy.is_skip_token(token))
    except Exception:
        return False


def _skip_message(token: str, reason: str) -> str:
    """Build a skip reason that always names its token.

    A token the harness already knows stays out of the message (the harness
    reports it in ``Result.skip_token``); an unknown token - for example
    ``mpi_grid`` before the policy grows it - is prefixed so the reason is
    self-describing and never empty.
    """
    text = str(reason).strip() or "KUPL backend unavailable"
    if token and not _token_is_known(token):
        return f"{token}: {text}"
    return text


def _skip(reason: str, token: str | None = None) -> NoReturn:
    """Raise the harness skip exception, or :class:`Skip` when unavailable."""
    message = _skip_message(token or "", reason)
    skip_type = _harness_skip_type()
    if skip_type is not None and skip_type is not Skip:
        instance: BaseException | None = None
        if token is not None:
            try:
                instance = skip_type(message, token=token)
            except TypeError:
                instance = None
        if instance is None:
            try:
                instance = skip_type(message)
            except TypeError:
                try:
                    instance = skip_type()
                except TypeError:
                    instance = None
        if instance is not None:
            if not str(getattr(instance, "reason", "") or "").strip():
                try:
                    instance.reason = message  # type: ignore[attr-defined]
                except Exception:
                    pass
            if token is not None and not getattr(instance, "token", None):
                try:
                    instance.token = token  # type: ignore[attr-defined]
                except Exception:
                    pass
            raise instance
    raise Skip(message, token)


# ---------------------------------------------------------------------------
# Backend discovery and environment probes
# ---------------------------------------------------------------------------


def load_qcu() -> Any:
    """Import and return the KUPL ctypes binding.

    Raises the harness skip exception (see :func:`_skip`) with the
    ``kupl_library`` token when ``libqcu.so`` is missing or cannot be loaded.
    Every other import failure propagates as a real error.
    """
    try:
        from pyqcu.kupl import qcu
    except (FileNotFoundError, OSError) as error:
        _skip(f"KUPL qcu library is unavailable: "
              f"{type(error).__name__}: {error}", token=_LIBRARY_TOKEN)
    return qcu


def library_status() -> tuple[bool, str]:
    """Return ``(available, reason)`` for the KUPL library without raising."""
    try:
        qcu = load_qcu()
    except BaseException as error:  # noqa: BLE001 - probe must not raise
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        reason = str(getattr(error, "reason", "") or error).strip()
        return False, reason or f"{type(error).__name__}: {error}"
    try:
        return True, str(qcu.library_path())
    except BaseException:  # noqa: BLE001 - path is informational only
        return True, "pyqcu.kupl.qcu"


def mpi_size() -> int:
    """Return the MPI world size without failing when mpi4py is missing."""
    try:
        from pyqcu.kupl import define
    except Exception:
        return 1
    return int(getattr(define, "size", 1))


def mpi_rank() -> int:
    """Return this process' MPI rank (0 when ``mpi4py`` is unavailable)."""
    try:
        from pyqcu.kupl import define
    except Exception:
        return 0
    return int(getattr(define, "rank", 0))


# ---------------------------------------------------------------------------
# Option vocabulary: validation helpers
# ---------------------------------------------------------------------------


def _normalize_dtype(value: Any) -> str:
    """Map a dtype spelling (``c64``/``complex128``/torch dtype) to policy form."""
    if value is None:
        return "c64"
    name = getattr(value, "__name__", None) or str(value)
    name = name.replace("torch.", "").strip().lower()
    try:
        return _DTYPE_ALIASES[name]
    except KeyError:
        raise ValueError(
            f"KUPL supports c64/c128 (complex64/complex128), got {value!r}"
        ) from None


def _normalize_choice(value: Any, aliases: Mapping[str, str], name: str) -> str:
    """Normalize a selector option, rejecting an unknown value."""
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return aliases[text]
    except KeyError:
        raise ValueError(
            f"{name} must be one of "
            f"{', '.join(sorted(set(aliases.values())))}, got {value!r}"
        ) from None


def _normalize_role(value: Any, aliases: Mapping[str, str], name: str,
                    allowed: tuple[str, ...]) -> str:
    """Normalize ``gauge``/``source``: a selector name or a tensor."""
    if isinstance(value, str):
        selected = _normalize_choice(value, aliases, name)
        if selected not in allowed:
            raise ValueError(
                f"{name} must be one of {', '.join(allowed)} or a tensor, "
                f"got {value!r}")
        return selected
    return "tensor"


def _validate_shape(values: Any,
                    name: str = "lattice") -> tuple[int, int, int, int]:
    """Normalize four positive even extents (the parity split needs them)."""
    try:
        shape = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be four integers, got {values!r}") from error
    if len(shape) != 4:
        raise ValueError(f"{name} must have four extents, got {shape}")
    if any(value <= 0 or value % 2 for value in shape):
        raise ValueError(
            f"{name} extents must be positive even integers, got {shape}")
    return shape  # type: ignore[return-value]


def _validate_grid(values: Any, name: str) -> tuple[int, int, int, int]:
    """Normalize a four-dimensional process/coarse grid."""
    if isinstance(values, (str, bytes)):
        text = str(values).strip().strip("()[]")
        parts = [part for part in text.replace(",", " ").split() if part]
    else:
        try:
            parts = list(values)
        except TypeError as error:
            raise ValueError(
                f"{name} must be four integers, got {values!r}") from error
    try:
        grid = tuple(int(value) for value in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be four integers, got {values!r}") from error
    if len(grid) != 4:
        raise ValueError(f"{name} must contain four extents, got {grid}")
    if any(value <= 0 for value in grid):
        raise ValueError(f"{name} extents must be positive, got {grid}")
    return grid  # type: ignore[return-value]


def _resolve_site(values: Any,
                  shape: tuple[int, int, int, int]
                  ) -> tuple[int, int, int, int]:
    """Validate the source site against the lattice extents."""
    site = tuple(int(value) for value in values)
    if len(site) != 4:
        raise ValueError(
            f"source site must contain four coordinates, got {site}")
    for coordinate, extent in zip(site, shape):
        if coordinate < 0 or coordinate >= extent:
            raise ValueError(f"source site {site} is outside lattice {shape}")
    return site  # type: ignore[return-value]


def _resolve_mass_kappa(mass: Any, kappa: Any) -> tuple[float, float]:
    """Resolve ``(mass, kappa)`` with ``kappa = 1/(2*mass+8)``.

    ``kappa`` alone is enough (``mass = (1/kappa - 8)/2``); giving both is
    accepted only when they describe the same operator, because a silent
    mismatch would mean the reference and the native controls disagree.
    """
    if kappa is not None:
        value = float(kappa)
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"kappa must be finite and non-zero, got {value}")
        derived = 0.5 / value - 4.0
        if mass is not None and not math.isclose(
                float(mass), derived, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"mass={float(mass)!r} and kappa={value!r} are inconsistent: "
                f"kappa={value!r} implies mass={derived!r}")
        return derived, value
    value = DEFAULT_MASS if mass is None else float(mass)
    if not math.isfinite(value):
        raise ValueError(f"mass must be finite, got {value}")
    denominator = 2.0 * value + 8.0
    if denominator == 0.0:
        raise ValueError(
            f"mass={value!r} is degenerate: kappa = 1/(2m+8) is infinite")
    return value, 1.0 / denominator


def _resolve_atol(atol: Any, dtype_name: str) -> float:
    """Resolve the solver tolerance; c128 needs a tighter default than c64."""
    if atol is not None:
        value = float(atol)
    elif dtype_name == "c128":
        value = DEFAULT_SOLVE_ATOL_C128
    else:
        value = DEFAULT_SOLVE_ATOL
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"atol must be finite and positive, got {value}")
    if value < MIN_SOLVE_ATOL:
        raise ValueError(
            f"atol={value!r} is below the smallest native tolerance "
            f"({MIN_SOLVE_ATOL:g}); the KUPL solver aborts on a native error")
    if value > 1.0:
        raise ValueError(
            f"atol={value!r} is not a relative residual tolerance (<= 1)")
    return value


def _resolve_max_iter(max_iter: Any) -> int:
    value = DEFAULT_MAX_ITER if max_iter is None else int(max_iter)
    if value <= 0:
        raise ValueError(f"max_iter must be positive, got {value}")
    return value


def _resolve_device(device: Any) -> str:
    """Return a normalized KUPL device name or skip ``device_unavailable``."""
    requested = "cpu" if device is None else str(device).strip().lower()
    if not requested:
        requested = "cpu"
    base, _, index = requested.partition(":")
    if base not in SUPPORTED_DEVICES:
        _skip(
            f"KUPL is a CPU backend; device {device!r} is not available "
            f"(supported: {', '.join(SUPPORTED_DEVICES)})",
            token=_DEVICE_TOKEN)
    if index and not index.isdigit():
        raise ValueError(f"invalid device index in {device!r}")
    return f"cpu:{index}" if index else "cpu"


# ---------------------------------------------------------------------------
# Normalized run options
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Options:
    """Validated runner configuration (one object per runner call)."""

    shape: tuple[int, int, int, int]
    dtype: str
    seed: int
    repeat: int
    warmup: int
    mass: float
    kappa: float
    csw: float
    gauge_sigma: float
    gauge: Any
    gauge_kind: str
    source: Any
    source_kind: str
    source_site: tuple[int, int, int, int]
    source_spin: int
    source_color: int
    lattice_grid: tuple[int, int, int, int]
    mg_grid: tuple[int, int, int, int] | None
    max_iter: int
    atol: float
    device: str
    dslash: str
    solver: str
    gauge_seed: int
    source_seed: int
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def compact(self) -> tuple[int, int, int, int]:
        """Parity-compact ``[X, Y, Z, T/2]`` extents of the ABI tensors."""
        return (*self.shape[:3], self.shape[3] // 2)

    @property
    def mg_grid_valid(self) -> bool | None:
        """True when a requested ``mg_grid`` block-divides the lattice."""
        if self.mg_grid is None:
            return None
        return all(extent % block == 0
                   for extent, block in zip(self.shape, self.mg_grid))

    def source_spec(self, parity: int | None) -> dict[str, Any]:
        """JSON-safe description of the requested source."""
        return {
            "kind": self.source_kind,
            "parity": None if parity is None else int(parity),
            "spin": int(self.source_spin),
            "color": int(self.source_color),
            "site": list(self.source_site),
            "seed": (int(self.source_seed) if self.source_kind == "random"
                     else None),
        }


def _container_value(container: Any, name: str) -> Any:
    """Read ``name`` from an options container (mapping or object)."""
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _pick(explicit: Any, container: Any, name: str, default: Any) -> Any:
    """Explicit keyword wins, then the options container, then the default."""
    if explicit is not None:
        return explicit
    value = _container_value(container, name)
    return default if value is None else value


def _options(case: Any = None, *, lattice: Any = None, dtype: Any = None,
             seed: Any = None, repeat: Any = None, warmup: Any = None,
             mass: Any = None, kappa: Any = None, csw: Any = None,
             gauge_sigma: Any = None, source: Any = None, gauge: Any = None,
             lattice_grid: Any = None, mg_grid: Any = None,
             max_iter: Any = None, atol: Any = None, device: Any = None,
             dslash: Any = None, solver: Any = None, gauge_seed: Any = None,
             source_seed: Any = None, source_site: Any = None,
             source_spin: Any = None, source_color: Any = None,
             options: Any = None, default_csw: float = DEFAULT_CSW,
             default_source: str = "random",
             extras: Mapping[str, Any] | None = None) -> _Options:
    """Merge ``case``, explicit keywords, the options container and defaults.

    ``case`` may be a mapping, an object carrying the option names or a plain
    lattice sequence; an explicitly passed keyword always wins, so a caller can
    override a ``Case``/``Options`` field without rebuilding the case.
    """
    container = options
    if case is not None:
        if isinstance(case, Mapping):
            merged = dict(case)
            if lattice is None:
                lattice = merged.pop("lattice", None)
            if dtype is None:
                dtype = merged.pop("dtype", None)
            if seed is None:
                seed = merged.pop("seed", None)
            rest = {key: value for key, value in merged.items()
                    if value is not None}
            if rest and container is None:
                container = rest
        elif isinstance(case, (tuple, list)):
            if lattice is None:
                lattice = case
        else:
            if lattice is None:
                lattice = getattr(case, "lattice", None)
            if dtype is None:
                dtype = getattr(case, "dtype", None)
            if seed is None:
                seed = getattr(case, "seed", None)
            if container is None:
                container = case

    def _get(name: str, explicit: Any, default: Any) -> Any:
        return _pick(explicit, container, name, default)

    shape = _validate_shape(_get("lattice", lattice, QUICK_LATTICE))
    dtype_name = _normalize_dtype(_get("dtype", dtype, "c64"))
    seed_value = int(_get("seed", seed, DEFAULT_SEED))
    if seed_value < 0:
        raise ValueError(f"seed must be non-negative, got {seed_value}")
    repeat_value = int(_get("repeat", repeat, DEFAULT_REPEAT))
    warmup_value = int(_get("warmup", warmup, DEFAULT_WARMUP))
    if repeat_value < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat_value}")
    if warmup_value < 1:
        raise ValueError(f"warmup must be >= 1, got {warmup_value}")

    mass_value, kappa_value = _resolve_mass_kappa(
        _get("mass", mass, None), _get("kappa", kappa, None))
    csw_value = float(_get("csw", csw, default_csw))
    if not math.isfinite(csw_value):
        raise ValueError(f"csw must be finite, got {csw_value}")
    sigma_value = float(_get("gauge_sigma", gauge_sigma, DEFAULT_GAUGE_SIGMA))
    if not math.isfinite(sigma_value) or sigma_value < 0.0:
        raise ValueError(
            f"gauge_sigma must be finite and non-negative, got {sigma_value}")

    gauge_value = _get("gauge", gauge, "random")
    gauge_kind = _normalize_role(gauge_value, _GAUGE_ALIASES, "gauge",
                                 GAUGE_KINDS)
    if gauge_kind != "tensor":
        gauge_value = gauge_kind

    source_value = _get("source", source, default_source)
    source_kind = _normalize_role(source_value, _SOURCE_ALIASES, "source",
                                  SOURCE_KINDS)
    if source_kind != "tensor":
        source_value = source_kind

    grid_value = _validate_grid(
        _get("lattice_grid", lattice_grid, DEFAULT_LATTICE_GRID),
        "lattice_grid")
    mg_value = _get("mg_grid", mg_grid, None)
    mg_tuple = None if mg_value is None else _validate_grid(mg_value, "mg_grid")

    site = _resolve_site(
        _get("source_site", source_site, DEFAULT_SOURCE_SITE), shape)
    spin = int(_get("source_spin", source_spin, DEFAULT_SOURCE_SPIN))
    color = int(_get("source_color", source_color, DEFAULT_SOURCE_COLOR))
    if not 0 <= spin < 4:
        raise ValueError(f"source_spin must be in [0, 4), got {spin}")
    if not 0 <= color < 3:
        raise ValueError(f"source_color must be in [0, 3), got {color}")

    dslash_name = _normalize_choice(
        _get("dslash", dslash, None), _DSLASH_ALIASES, "dslash")
    solver_name = _normalize_choice(
        _get("solver", solver, None), _SOLVER_ALIASES, "solver")

    return _Options(
        shape=shape,
        dtype=dtype_name,
        seed=seed_value,
        repeat=repeat_value,
        warmup=warmup_value,
        mass=mass_value,
        kappa=kappa_value,
        csw=csw_value,
        gauge_sigma=sigma_value,
        gauge=gauge_value,
        gauge_kind=gauge_kind,
        source=source_value,
        source_kind=source_kind,
        source_site=site,
        source_spin=spin,
        source_color=color,
        lattice_grid=grid_value,
        mg_grid=mg_tuple,
        max_iter=_resolve_max_iter(_get("max_iter", max_iter, None)),
        atol=_resolve_atol(_get("atol", atol, None), dtype_name),
        device=_resolve_device(_get("device", device, None)),
        dslash=dslash_name,
        solver=solver_name,
        gauge_seed=int(_get("gauge_seed", gauge_seed, DEFAULT_GAUGE_SEED)),
        source_seed=int(_get("source_seed", source_seed,
                             DEFAULT_SOURCE_SEED)),
        extras=dict(extras or {}),
    )


def _require_operator(option: _Options, expected_dslash: str | None = None,
                      expected_solver: str | None = None) -> None:
    """Reject a ``dslash``/``solver`` request that contradicts the runner.

    ``auto`` selects the runner's own entry point; naming the matching operator
    is accepted so one option set can drive the whole matrix; naming a
    different one is a usage error, because silently running the wrong operator
    would make the observation meaningless.
    """
    if (expected_dslash is not None and option.dslash != "auto"
            and option.dslash != expected_dslash):
        raise ValueError(
            f"this runner applies the {expected_dslash} operator, but "
            f"dslash={option.dslash!r} was requested")
    if (expected_solver is not None and option.solver != "auto"
            and option.solver != expected_solver):
        raise ValueError(
            f"this runner uses the {expected_solver} solver, but "
            f"solver={option.solver!r} was requested")


# ---------------------------------------------------------------------------
# MPI grid and rank gates
# ---------------------------------------------------------------------------


def _require_mpi_grid(option: _Options) -> tuple[int, int]:
    """Return ``(size, rank)``, skipping ``mpi_grid`` on a product mismatch."""
    size = mpi_size()
    rank = mpi_rank()
    requested = int(math.prod(option.lattice_grid))
    if requested != size:
        _skip(
            f"lattice_grid {option.lattice_grid} describes {requested} rank(s) "
            f"but MPI_COMM_WORLD has {size}; the native ABI needs one process "
            f"grid matching the world size", token=_GRID_TOKEN)
    return size, rank


def _require_single_rank(size: int | None = None) -> int:
    """Skip ``mpi_single_rank`` when the job has more than one rank."""
    value = mpi_size() if size is None else int(size)
    if value != 1:
        _skip(f"KUPL CPU cases run on one MPI rank; "
              f"MPI_COMM_WORLD size is {value}", token=_RANK_TOKEN)
    return value


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _library_text(qcu: Any) -> str:
    try:
        return str(qcu.library_path())
    except BaseException:  # noqa: BLE001 - diagnostics only
        return str(getattr(getattr(qcu, "_LIBRARY", None), "_name", ""))


@contextmanager
def _preserve_torch_rng():
    """Keep ``generate_gauge_field`` from leaking its ``manual_seed`` state."""
    import torch

    state = torch.random.get_rng_state()
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    return torch.complex128 if dtype_name == "c128" else torch.complex64


def _real_dtype(torch_dtype: Any) -> Any:
    import torch

    return (torch.float64 if torch_dtype is torch.complex128
            else torch.float32)


def _relative_error(actual: Any, expected: Any) -> float:
    """``max|actual-expected| / max|expected|`` (``inf`` on shape mismatch).

    The same normalization as :func:`pyqcu.test.common.relative_error`, so a
    runner observation and the harness gate agree on what "relative" means.
    """
    actual_shape = tuple(getattr(actual, "shape", ()))
    expected_shape = tuple(getattr(expected, "shape", ()))
    if actual_shape != expected_shape:
        return math.inf
    if int(expected.numel()) == 0:
        return 0.0
    scale = float(expected.abs().max())
    difference = float((actual - expected).abs().max())
    if scale == 0.0:
        return difference
    return difference / scale


def _time_samples(call: Any, repeat: Any, warmup: Any) -> tuple[float, ...]:
    """Time ``call`` with ``warmup`` discarded calls and ``repeat`` samples.

    ``pyqcu.test.common.time_samples`` is used when it is importable, so the
    harness and the runner share one timing rule; the local fallback applies
    the same rule when this module is used on its own.  ``warmup`` and
    ``repeat`` are clamped to at least one: a warmup-free measurement is not a
    steady state.
    """
    samples = max(1, int(DEFAULT_REPEAT if repeat is None else repeat))
    discarded = max(1, int(DEFAULT_WARMUP if warmup is None else warmup))
    try:
        from .common import time_samples as harness_time_samples
    except Exception:
        harness_time_samples = None
    if harness_time_samples is not None:
        return tuple(harness_time_samples(
            call, repeat=samples, warmup=discarded))
    for _ in range(discarded):
        call()
    measured: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        call()
        measured.append(time.perf_counter() - started)
    return tuple(measured)


def _global_abi_snapshot() -> dict[str, Any]:
    """Clone the module-level ABI control arrays of the CUDA/KUPL packages."""
    snapshot: dict[str, Any] = {}
    for module_name in ("pyqcu.cuda.define", "pyqcu.kupl.define"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute in ("params", "argv", "set_ptrs"):
            value = getattr(module, attribute, None)
            clone = getattr(value, "clone", None)
            if callable(clone):
                snapshot[f"{module_name}.{attribute}"] = clone()
    return snapshot


def _changed_global_arrays(snapshot: Mapping[str, Any]) -> list[str]:
    import torch

    changed = []
    for name, before in snapshot.items():
        module_name, _, attribute = name.rpartition(".")
        module = sys.modules.get(module_name)
        current = getattr(module, attribute, None) if module else None
        if current is None or not torch.equal(before, current):
            changed.append(name)
    return changed


# ---------------------------------------------------------------------------
# Gauge field and source construction
# ---------------------------------------------------------------------------


def _full_gauge_shape(shape: tuple[int, int, int, int]) -> tuple[int, ...]:
    return (3, 3, 4, *shape)


def _compact_gauge_shape(shape: tuple[int, int, int, int]) -> tuple[int, ...]:
    return (2, 3, 3, 4, *(*shape[:3], shape[3] // 2))


def _compact_fermion_shape(shape: tuple[int, int, int, int]) -> tuple[int, ...]:
    return (2, 4, 3, *(*shape[:3], shape[3] // 2))


def _build_gauge(option: _Options, torch: Any, torch_dtype: Any
                 ) -> dict[str, Any]:
    """Build the gauge field in both layouts (full ``oooxyzt`` and compact).

    ``gauge="random"`` uses the project generator with an explicit seed and a
    disorder of ``gauge_sigma``; ``gauge="unit"`` builds identity links (a
    deterministic, exactly SU(3) field); a tensor is accepted in either layout
    and converted.  Generation restores the process-wide torch RNG state.
    """
    from pyqcu import lattice as lattice_module
    from pyqcu import tools

    shape = option.shape
    full_shape = _full_gauge_shape(shape)
    compact_shape = _compact_gauge_shape(shape)
    value = option.gauge
    if option.gauge_kind == "tensor":
        if not hasattr(value, "shape"):
            raise TypeError(
                f"gauge must be a tensor in a {full_shape} or {compact_shape} "
                f"layout, got {type(value).__name__}")
        candidate = value.to(dtype=torch_dtype)
        if tuple(candidate.shape) == full_shape:
            full = candidate.contiguous()
            gauge = tools.oooxyzt2poooxyzt(full).contiguous()
            layout = _FULL_LAYOUT
        elif tuple(candidate.shape) == compact_shape:
            gauge = candidate.contiguous()
            full = tools.poooxyzt2oooxyzt(gauge).contiguous()
            layout = _COMPACT_LAYOUT
        else:
            raise ValueError(
                f"gauge tensor must have shape {full_shape} or "
                f"{compact_shape}, got {tuple(candidate.shape)}")
    elif option.gauge_kind == "unit":
        index = torch.arange(3)
        full = torch.zeros(full_shape, dtype=torch_dtype)
        full[index, index, :] = 1.0 + 0.0j
        full = full.contiguous()
        gauge = tools.oooxyzt2poooxyzt(full).contiguous()
        layout = _FULL_LAYOUT
    else:
        full = torch.empty(full_shape, dtype=torch_dtype)
        with _preserve_torch_rng():
            lattice_module.generate_gauge_field(
                full, sigma=float(option.gauge_sigma),
                seed=int(option.gauge_seed), verbose=False)
        full = full.contiguous()
        gauge = tools.oooxyzt2poooxyzt(full).contiguous()
        layout = _FULL_LAYOUT

    try:
        su3 = bool(lattice_module.check_su3(full, tol=1e-3, verbose=False))
    except BaseException:  # noqa: BLE001 - diagnostics only
        su3 = None
    return {
        "full": full,
        "compact": gauge,
        "su3": su3,
        "kind": option.gauge_kind,
        "layout": layout,
        "seed": (int(option.gauge_seed)
                 if option.gauge_kind == "random" else None),
        "sigma": (float(option.gauge_sigma)
                  if option.gauge_kind == "random" else None),
    }


def _build_source(option: _Options, torch: Any, torch_dtype: Any
                  ) -> dict[str, Any]:
    """Build the two-parity compact source ``[2, 4, 3, X, Y, Z, T/2]``.

    ``random`` uses a private generator (reproducible, and it does not consume
    the process-wide stream); ``point`` places a unit charge at
    ``(spin, color, site)``; ``unit`` fills every component with one.  A tensor
    source is accepted in the full layout, the compact layout or as a single
    parity block (which is then embedded in parity 0).
    """
    from pyqcu import tools

    shape = option.shape
    compact = option.compact
    full_shape = _compact_fermion_shape(shape)
    single_shape = (4, 3, *compact)
    real_dtype = _real_dtype(torch_dtype)
    value = option.source
    layout = _COMPACT_LAYOUT

    if option.source_kind == "tensor":
        if not hasattr(value, "shape"):
            raise TypeError(
                f"source must be a tensor in a {full_shape}, {single_shape} "
                f"or full-lattice layout, got {type(value).__name__}")
        candidate = value.to(dtype=torch_dtype)
        if tuple(candidate.shape) == full_shape:
            source = candidate.contiguous()
        elif tuple(candidate.shape) == single_shape:
            source = torch.zeros(full_shape, dtype=torch_dtype)
            source[0] = candidate
            source = source.contiguous()
        elif tuple(candidate.shape) == (2, 4, 3, *shape):
            source = tools.oooxyzt2poooxyzt(
                candidate.contiguous()).contiguous()
            layout = _FULL_LAYOUT
        else:
            raise ValueError(
                f"source tensor must have shape {full_shape}, {single_shape} "
                f"or {(2, 4, 3, *shape)}, got {tuple(candidate.shape)}")
    elif option.source_kind == "point":
        source = torch.zeros(full_shape, dtype=torch_dtype)
        source[0, option.source_spin, option.source_color,
               *option.source_site] = 1.0 + 0.0j
        source = source.contiguous()
    elif option.source_kind == "unit":
        source = torch.ones(full_shape, dtype=torch_dtype).contiguous()
    else:
        generator = torch.Generator().manual_seed(int(option.source_seed))
        single = torch.complex(
            torch.randn(single_shape, generator=generator, dtype=real_dtype),
            torch.randn(single_shape, generator=generator, dtype=real_dtype),
        ).to(torch_dtype)
        source = torch.stack((single, single)).contiguous()

    return {
        "compact": source,
        "kind": option.source_kind,
        "layout": layout,
        "seed": (int(option.source_seed)
                 if option.source_kind == "random" else None),
    }


# ---------------------------------------------------------------------------
# Private ABI runtime: control arrays plus the strict transaction protocol
# ---------------------------------------------------------------------------


class _PrivateRuntime:
    """Own one private set of ABI controls and run KUPL transactions.

    ``QcuRuntime.from_lattice`` builds the ``params``/``argv``/``set_ptrs``
    tensors, so nothing here can reach the module-level arrays.  The lifecycle
    wrapper is local because the formal port must advance ``_SET_INDEX_``
    exactly once *after* ``applyEndQcu``.
    """

    def __init__(self, qcu: Any, option: _Options, *, plan: int,
                 parity: int) -> None:
        from pyqcu.cuda import QcuRuntime, define

        self.qcu = qcu
        self.define = define
        self.option = option
        self.dtype = _torch_dtype(option.dtype)
        self.runtime = QcuRuntime.from_lattice(
            option.shape, mass=option.mass, csw=option.csw,
            atol=option.atol, sigma=option.gauge_sigma,
            max_iter=option.max_iter, dtype=self.dtype, device="cpu",
            plan=plan, parity=parity, seed=option.seed, verbose=0,
            grid=option.lattice_grid, qcu_module=qcu)
        self.params = self.runtime.params
        self.argv = self.runtime.argv
        self.set_ptrs = self.runtime.set_ptrs
        self.slot_limit = min(int(self.runtime.slot_limit), _SLOT_LIMIT)
        self._cursor = 0
        self._index: int | None = None
        self.init_calls = 0
        self.end_calls = 0

    # -- controls ----------------------------------------------------------

    def configure(self, **values: Any) -> None:
        """Update the private ABI controls (delegates to ``QcuRuntime``)."""
        self.runtime.configure(**values)

    def empty(self, shape: tuple[int, ...]) -> Any:
        """Allocate a private tensor on the runtime dtype (CPU)."""
        import torch

        return torch.empty(shape, dtype=self.dtype)

    @property
    def control_values(self) -> dict[str, Any]:
        """Effective physics controls of the private arrays."""
        define = self.define
        return {
            "argv_mass": float(self.argv[define._MASS_]),
            "argv_atol": float(self.argv[define._ATOL_]),
            "argv_sigma": float(self.argv[define._SIGMA_]),
            "argv_csw": float(self.argv[define._CSW_]),
            "params_max_iter": int(self.params[define._MAX_ITER_]),
            "params_plan": int(self.params[define._SET_PLAN_]),
            "params_parity": int(self.params[define._PARITY_]),
            "params_data_type": int(self.params[define._DATA_TYPE_]),
            "params_seed": int(self.params[define._SEED_]),
            "params_grid": [
                int(self.params[define._GRID_X_]),
                int(self.params[define._GRID_Y_]),
                int(self.params[define._GRID_Z_]),
                int(self.params[define._GRID_T_]),
            ],
        }

    def controls_applied(self) -> dict[str, bool]:
        """Whether each requested option really reached the ABI controls.

        ``argv`` is real32 for c64 and real64 for c128, so a real-valued option
        is compared with the resolution of that array instead of exact float
        equality (``0.17`` is not representable in binary32).
        """
        values = self.control_values
        option = self.option
        define = self.define
        tolerance = 1e-6 if option.dtype == "c64" else 1e-12

        def _same(value: float, expected: float) -> bool:
            return abs(float(value) - float(expected)) <= tolerance * max(
                1.0, abs(float(expected)))

        return {
            "mass_applied": _same(values["argv_mass"], option.mass),
            "csw_applied": _same(values["argv_csw"], option.csw),
            "sigma_applied": _same(values["argv_sigma"], option.gauge_sigma),
            "atol_applied": _same(values["argv_atol"], option.atol),
            "max_iter_applied": values["params_max_iter"] == option.max_iter,
            "grid_applied": tuple(values["params_grid"]) == option.lattice_grid,
            "lattice_applied": all(
                int(self.params[index]) == extent for index, extent in zip(
                    (define._LAT_X_, define._LAT_Y_, define._LAT_Z_,
                     define._LAT_T_), option.shape)),
            "dtype_applied": values["params_data_type"] == (
                define._LAT_C128_ if option.dtype == "c128"
                else define._LAT_C64_),
        }

    # -- lifecycle ---------------------------------------------------------

    def _allocate_slot(self) -> int:
        for offset in range(self.slot_limit):
            index = (self._cursor + offset) % self.slot_limit
            if int(self.set_ptrs[index]) == 0:
                self._cursor = (index + 1) % self.slot_limit
                return index
        raise RuntimeError("all KUPL set_ptrs slots are busy")

    @contextmanager
    def transaction(self, name: str):
        """Run one ``init -> operation -> end -> index += 1`` transaction."""
        function = getattr(self.qcu, name)
        index = self._allocate_slot()
        self.params[self.define._SET_INDEX_] = index
        self._index = index
        self.init_calls += 1
        try:
            self.qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
        except BaseException:
            # No slot was initialized; release the pointer defensively but do
            # not touch _SET_INDEX_ (it must only advance after a real end).
            try:
                self.qcu.applyEndQcu(self.set_ptrs, self.params)
            except BaseException:
                pass
            self.set_ptrs[index] = 0
            raise
        try:
            yield _Operation(self, function, name)
        finally:
            try:
                self.qcu.applyEndQcu(self.set_ptrs, self.params)
                self.end_calls += 1
            finally:
                self.set_ptrs[index] = 0
                # Advance only now, wrapping inside the ordinary slot window.
                self.params[self.define._SET_INDEX_] = (
                    (index + 1) % self.slot_limit)

    @property
    def set_index(self) -> int | None:
        return self._index

    @property
    def kappa(self) -> float:
        return float(self.runtime.kappa)

    @property
    def next_index(self) -> int:
        return int(self.params[self.define._SET_INDEX_])

    @property
    def slot_released(self) -> bool:
        index = self._index
        if index is None:
            return True
        return int(self.set_ptrs[index]) == 0

    def lifecycle(self) -> dict[str, Any]:
        return {
            "set_index": self.set_index,
            "set_index_after": self.next_index,
            "slot_released": self.slot_released,
            "init_calls": self.init_calls,
            "end_calls": self.end_calls,
        }

    # -- operations --------------------------------------------------------

    def validate(self, operands: tuple[Any, ...],
                 shapes: tuple[Any, ...] = ()) -> None:
        """Reject an operand the native ABI would abort on.

        The KUPL ABI converts a native exception into ``std::abort``, so a
        wrong dtype, device, layout or shape must be caught here.  ``shapes``
        entries are either a tuple to compare or ``None`` to skip the check.
        """
        for position, value in enumerate(operands):
            name = f"operand {position}"
            if not hasattr(value, "data_ptr"):
                raise TypeError(f"KUPL {name} must be an array with data_ptr")
            if getattr(value, "dtype", None) != self.dtype:
                raise TypeError(f"KUPL {name} must have dtype {self.dtype}")
            device = getattr(value, "device", None)
            if device is not None and getattr(device, "type", "cpu") != "cpu":
                raise ValueError(f"KUPL {name} must be a CPU tensor")
            if not value.is_contiguous():
                raise ValueError(f"KUPL {name} must be contiguous")
        for position, expected in enumerate(shapes):
            if expected is None:
                continue
            actual = tuple(operands[position].shape)
            if actual != tuple(expected):
                raise ValueError(
                    f"KUPL operand {position} must have shape "
                    f"{tuple(expected)}, got {actual}")


class _Operation:
    """Operation handle; valid only inside `_PrivateRuntime.transaction`."""

    def __init__(self, runtime: _PrivateRuntime, function: Any,
                 name: str) -> None:
        self.runtime = runtime
        self.function = function
        self.name = name

    def check(self, operands: tuple[Any, ...],
              shapes: tuple[Any, ...] = ()) -> None:
        """Validate operands (shape, dtype, device, contiguity) up front."""
        self.runtime.validate(operands, shapes)

    def __call__(self, *operands: Any) -> Any:
        self.runtime.validate(operands)
        return self.function(*operands, self.runtime.set_ptrs,
                             self.runtime.params)

    def read(self, name: str, *args: Any) -> Any:
        """Read a same-slot getter before the transaction releases the slot."""
        getter = getattr(self.runtime.qcu, name)
        return getter(*args, self.runtime.set_ptrs, self.runtime.params)


# ---------------------------------------------------------------------------
# ABI header / export parsing (mirrors pyqcu/testing/kupl/check_abi.py)
# ---------------------------------------------------------------------------

_DECLARATION = re.compile(
    r"\b(?P<return_type>void|int)\s+"
    r"(?P<name>(?:apply|test|get|verify)[A-Za-z0-9]+Qcu)\s*"
    r"\((?P<arguments>.*?)\)\s*;",
    re.DOTALL,
)
_EXPORTED_SYMBOL = re.compile(r"(?:apply|test|get|verify)[A-Za-z0-9]+Qcu")
_HEADER_CTYPES = {
    "double": ctypes.c_double,
    "double*": ctypes.POINTER(ctypes.c_double),
    "int": ctypes.c_int,
    "int*": ctypes.POINTER(ctypes.c_int),
    "long long": ctypes.c_longlong,
    "unsigned long long": ctypes.c_ulonglong,
    "unsigned long long*": ctypes.POINTER(ctypes.c_ulonglong),
}
_CTYPES_NAMES = {value: key for key, value in _HEADER_CTYPES.items()}


def _declarations(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Parse the ``apply|test|get|verify*Qcu`` declarations of one header."""
    declarations: dict[str, tuple[str, tuple[str, ...]]] = {}
    for match in _DECLARATION.finditer(path.read_text(encoding="utf-8")):
        arguments = []
        for argument in match.group("arguments").split(","):
            normalized = " ".join(argument.split())
            normalized = re.sub(r"\s+[A-Za-z_]\w*$", "", normalized)
            normalized = re.sub(r"\*[A-Za-z_]\w*$", "*", normalized)
            normalized = re.sub(r"\s*\*\s*", "*", normalized)
            arguments.append(normalized)
        declarations[match.group("name")] = (
            match.group("return_type"), tuple(arguments))
    return declarations


def _ctypes_name(value: Any) -> str:
    name = _CTYPES_NAMES.get(value)
    if name is not None:
        return name
    return getattr(value, "__name__", str(value))


def _signature_mismatches(
        declarations: dict[str, tuple[str, tuple[str, ...]]],
        signatures: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Compare parsed C declarations with the ctypes ``_SIGNATURES`` table."""
    mismatches = []
    for name, (_, arguments) in sorted(declarations.items()):
        expected = signatures.get(name)
        if expected is None:
            mismatches.append({
                "name": name,
                "declared": list(arguments),
                "ctypes": None,
                "reason": "missing from _SIGNATURES",
            })
            continue
        declared = tuple(_HEADER_CTYPES.get(argument, argument)
                         for argument in arguments)
        if declared != tuple(expected):
            mismatches.append({
                "name": name,
                "declared": [_ctypes_name(value) for value in declared],
                "ctypes": [_ctypes_name(value) for value in expected],
                "reason": "argument mismatch",
            })
    for name in sorted(set(signatures) - set(declarations)):
        mismatches.append({
            "name": name,
            "declared": None,
            "ctypes": [_ctypes_name(value) for value in signatures[name]],
            "reason": "not declared in the header",
        })
    return mismatches


def _exported_symbols(library: str) -> tuple[list[str] | None, str]:
    """Return ``(symbols, note)`` for the dynamic exports of ``library``."""
    nm = shutil.which("nm")
    if nm is None:
        return None, "nm executable is not available"
    try:
        completed = subprocess.run(
            [nm, "-D", "--defined-only", library],
            check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"nm could not run: {type(error).__name__}: {error}"
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines()
        detail = lines[0] if lines else "no stderr"
        return None, f"nm exited {completed.returncode}: {detail}"
    exported = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        symbol = fields[-1].split("@", 1)[0]
        if _EXPORTED_SYMBOL.fullmatch(symbol):
            exported.add(symbol)
    return sorted(exported), ""


# ---------------------------------------------------------------------------
# Runner: 40-symbol C ABI parity
# ---------------------------------------------------------------------------


def run_abi_symbols(case: Any = None, **options: Any) -> dict[str, Any]:
    """Check the KUPL library against the 40-symbol CUDA QCU C ABI.

    Required observations (the ``kupl/abi/symbols`` gate):

    ``symbol_count``
        Number of ABI names bound on the loaded library (must be 40).
    ``count``
        Alias of ``symbol_count``, for readers that prefer the shorter name.
    ``set_match``
        Composite boolean over every check that could run.
    ``missing``
        Sorted union of names missing from the library, from the Python
        wrappers or from the ``nm`` export table (empty on success).

    Additional observations (JSON-safe):

    ``symbols``
        Sorted ABI names.
    ``missing_symbols`` / ``missing_wrappers`` / ``missing_exports``
        The three components of ``missing``.
    ``signature_mismatches`` / ``restype_mismatches``
        Per-name ``declared`` vs ``ctypes`` differences against the
        authoritative header.
    ``name_set_matches_cuda`` / ``signatures_match_cuda``
        KUPL header vs CUDA header declaration parity (``None`` when a header
        is not part of the distribution).
    ``ctypes_signatures_match``
        Header declarations vs the bound ctypes signatures/return types
        (``None`` when no header is available).
    ``unexpected_exports`` / ``export_set_matches``
        ``nm -D --defined-only`` symbol set (``None`` when ``nm`` cannot run).
    ``names_match`` / ``signatures_match`` / ``signature_match``
        Aliases of ``set_match``.
    ``notes``
        Human readable reasons for optional checks that were skipped.

    The runner skips (``SKIP``) with token ``kupl_library`` only when
    ``libqcu.so`` cannot be imported.
    """
    del options  # forward compatibility for harness keywords
    started = time.perf_counter()
    qcu = load_qcu()
    signatures = dict(getattr(qcu, "_SIGNATURES", {}) or {})
    names = sorted(signatures)
    library = _library_text(qcu)
    notes: list[str] = []

    root = _repo_root()
    header_paths = {
        "kupl": root / "cpp" / "kupl" / "qcu" / "include" / "kupl" / "qcu.h",
        "cuda": root / "cpp" / "cuda" / "qcu" / "python" / "pyqcu.h",
    }
    declarations: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {}
    for key, path in header_paths.items():
        if path.is_file():
            declarations[key] = _declarations(path)
        else:
            notes.append(f"{key} ABI header is not available: {path}")

    kupl_declared = declarations.get("kupl")
    cuda_declared = declarations.get("cuda")

    missing_symbols = [name for name in names
                       if not _has_attribute(qcu._LIBRARY, name)]
    missing_wrappers = [name for name in names
                        if not callable(getattr(qcu, name, None))]

    # One header is authoritative for the ctypes comparison; header-to-header
    # parity below already proves the two declarations are identical.
    authority = "kupl" if kupl_declared is not None else "cuda"
    authoritative = declarations.get(authority)
    signature_mismatches: list[dict[str, Any]] = []
    restype_mismatches: list[dict[str, Any]] = []
    if authoritative is not None:
        for item in _signature_mismatches(authoritative, signatures):
            item["header"] = authority
            signature_mismatches.append(item)
        for name, (return_type, _) in sorted(authoritative.items()):
            function = getattr(qcu._LIBRARY, name, None)
            if function is None:
                continue
            wanted = ctypes.c_int if return_type == "int" else None
            if getattr(function, "restype", None) is not wanted:
                restype_mismatches.append({
                    "header": authority,
                    "name": name,
                    "declared": return_type,
                    "ctypes": _ctypes_name(function.restype),
                })

    name_set_matches_cuda = None
    signatures_match_cuda = None
    if kupl_declared is not None and cuda_declared is not None:
        name_set_matches_cuda = set(kupl_declared) == set(cuda_declared)
        signatures_match_cuda = kupl_declared == cuda_declared

    kupl_header_count_ok = (None if kupl_declared is None
                            else len(kupl_declared) == EXPECTED_ABI_SYMBOLS)
    cuda_header_count_ok = (None if cuda_declared is None
                            else len(cuda_declared) == EXPECTED_ABI_SYMBOLS)
    ctypes_signatures_match = (
        None if authoritative is None
        else not signature_mismatches and not restype_mismatches)

    exported: list[str] | None = None
    export_note = ""
    if library.lower().endswith(".so"):
        exported, export_note = _exported_symbols(library)
    else:
        export_note = f"not an ELF shared object: {library or '<unknown>'}"
    if export_note:
        notes.append(export_note)
    missing_exports: list[str] = []
    unexpected_exports: list[str] = []
    export_set_matches: bool | None = None
    if exported is not None:
        declared_names = set()
        for declared in declarations.values():
            declared_names |= set(declared)
        wanted = declared_names or set(names)
        missing_exports = sorted(wanted - set(exported))
        unexpected_exports = sorted(set(exported) - wanted)
        export_set_matches = not missing_exports and not unexpected_exports

    count_ok = len(names) == EXPECTED_ABI_SYMBOLS
    symbols_ok = not missing_symbols and not missing_wrappers
    headers_ok = all(check in (None, True) for check in (
        kupl_header_count_ok, cuda_header_count_ok,
        name_set_matches_cuda, signatures_match_cuda))
    composite = (count_ok and symbols_ok and headers_ok
                 and ctypes_signatures_match in (None, True)
                 and export_set_matches in (None, True))

    return {
        "backend": BACKEND,
        "check": "abi_symbols",
        "device": "cpu",
        "library": library,
        "headers": {key: (str(path) if path.is_file() else None)
                    for key, path in header_paths.items()},
        "headers_available": bool(declarations),
        "count": len(names),
        "expected_count": EXPECTED_ABI_SYMBOLS,
        "kupl_declared": (None if kupl_declared is None
                          else len(kupl_declared)),
        "cuda_declared": (None if cuda_declared is None
                          else len(cuda_declared)),
        "symbols": names,
        "symbol_count": len(names),
        "missing": sorted(set(missing_symbols) | set(missing_wrappers)
                          | set(missing_exports)),
        "missing_symbols": missing_symbols,
        "missing_wrappers": missing_wrappers,
        "missing_exports": missing_exports,
        "unexpected_exports": unexpected_exports,
        "signature_mismatches": signature_mismatches,
        "restype_mismatches": restype_mismatches,
        "name_set_matches_cuda": name_set_matches_cuda,
        "signatures_match_cuda": signatures_match_cuda,
        "ctypes_header": authority if authoritative is not None else None,
        "ctypes_signatures_match": ctypes_signatures_match,
        "kupl_header_count_ok": kupl_header_count_ok,
        "cuda_header_count_ok": cuda_header_count_ok,
        "export_check": "nm" if exported is not None else "unavailable",
        "export_set_matches": export_set_matches,
        "set_match": composite,
        "names_match": composite,
        "signatures_match": composite,
        "signature_match": composite,
        "seconds": time.perf_counter() - started,
        "notes": notes,
    }


def _has_attribute(target: Any, name: str) -> bool:
    try:
        getattr(target, name)
    except AttributeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Shared runner scaffolding
# ---------------------------------------------------------------------------


def _prepare(option: _Options, *, plan: int, parity: int
             ) -> tuple[Any, _PrivateRuntime, dict[str, Any],
                        dict[str, Any], float]:
    """Probe, build the inputs and one private runtime for a numeric runner."""
    qcu = load_qcu()
    size, _ = _require_mpi_grid(option)
    _require_single_rank(size)
    import torch  # noqa: F401 - the builders need a live torch

    started = time.perf_counter()
    torch_dtype = _torch_dtype(option.dtype)
    gauge = _build_gauge(option, torch, torch_dtype)
    source = _build_source(option, torch, torch_dtype)
    runtime = _PrivateRuntime(qcu, option, plan=plan, parity=parity)
    return qcu, runtime, gauge, source, started


def _common_observations(option: _Options, runtime: _PrivateRuntime,
                         gauge: Mapping[str, Any], source_parity: int | None,
                         library: str, started: float) -> dict[str, Any]:
    """Observation block shared by every KUPL numerical runner."""
    return {
        "backend": BACKEND,
        "device": option.device,
        "lattice": list(option.shape),
        "dtype": option.dtype,
        "seed": int(option.seed),
        "mass": float(option.mass),
        "kappa": runtime.kappa,
        "csw": float(option.csw),
        "gauge_sigma": float(option.gauge_sigma),
        "gauge_seed": gauge["seed"],
        "gauge_kind": gauge["kind"],
        "gauge_layout": gauge["layout"],
        "gauge_su3": gauge["su3"],
        "source_seed": (int(option.source_seed)
                        if option.source_kind == "random" else None),
        "source_kind": option.source_kind,
        "source": option.source_spec(source_parity),
        "lattice_grid": list(option.lattice_grid),
        "mpi_size": mpi_size(),
        "mpi_rank": mpi_rank(),
        "mg_grid": None if option.mg_grid is None else list(option.mg_grid),
        "mg_grid_valid": option.mg_grid_valid,
        "mg_grid_applied": False,
        "max_iter": int(option.max_iter),
        "atol": float(option.atol),
        "dslash_option": option.dslash,
        "solver_option": option.solver,
        "library": library,
        "repeat": int(option.repeat),
        "warmup": int(option.warmup),
        "extra_options": dict(option.extras),
        "case_seconds": time.perf_counter() - started,
        **runtime.control_values,
        **runtime.controls_applied(),
    }


def _finish(observations: dict[str, Any], runtime: _PrivateRuntime,
            global_before: Mapping[str, Any], samples: tuple[float, ...],
            immutability: Mapping[str, bool]) -> dict[str, Any]:
    """Add timing, lifecycle and global-state evidence, then return."""
    changed = _changed_global_arrays(global_before)
    observations.update(
        seconds=float(statistics.median(samples)),
        samples_seconds=list(samples),
        **runtime.lifecycle(),
        global_abi_unchanged=not changed,
        changed_global_arrays=changed,
    )
    observations.update(immutability)
    if immutability:
        observations["inputs_unchanged"] = all(
            bool(value) for value in immutability.values())
    return observations


def _bare_dslash_reference(torch: Any, source: Any, gauge: Any,
                           target: int) -> Any:
    """Bare hopping sum from the opposite parity, computed by torch.

    ``give_wilson_eo/oe`` return ``-kappa * hopping``; at ``kappa = 1`` they are
    exactly the negated hopping sum the native dslash entry points apply.
    ``target`` names the target parity (0 = even, 1 = odd).
    """
    from pyqcu.torch import dslash

    kappa_one = torch.tensor([1.0])
    if target == 0:
        return -dslash.give_wilson_eo(src_o=source, U_eo=gauge,
                                      kappa=kappa_one)
    return -dslash.give_wilson_oe(src_e=source, U_eo=gauge, kappa=kappa_one)


def _full_wilson_reference(torch: Any, source: Any, gauge: Any, target: int,
                           kappa: float,
                           shape: tuple[int, int, int, int]) -> Any:
    """Bare hopping implied by the full-lattice Wilson operator.

    The native entry point applies ``hopping(x_other)``.  Composing it with the
    identity term gives the full operator ``D_W x = x - kappa * hopping``; on
    the target parity ``x`` is zero, so
    ``hopping(x_other) = -D_W x|_target / kappa``.  Building that quotient from
    the full-lattice torch operator with the run's ``kappa`` exercises
    ``mass``/``kappa`` instead of hiding it behind a hard-coded reference.
    ``shape`` is the full lattice; ``gauge`` is the parity-compact field.
    """
    from pyqcu import tools
    from pyqcu.torch import dslash

    shape = tuple(int(value) for value in shape)
    compact = (*shape[:3], shape[3] // 2)
    full_source = torch.zeros((2, 4, 3, *compact), dtype=source.dtype)
    full_source[1 - target] = source
    full_field = tools.poooxyzt2oooxyzt(full_source)
    real_dtype = (torch.float64 if source.dtype is torch.complex128
                  else torch.float32)
    kappa_tensor = torch.tensor([float(kappa)], dtype=real_dtype)
    applied = dslash.give_wilson(
        full_field, tools.poooxyzt2oooxyzt(gauge), kappa_tensor, with_I=True)
    target_block = tools.oooxyzt2poooxyzt(applied)[target]
    return (-target_block / float(kappa)).contiguous()


# ---------------------------------------------------------------------------
# Runner: single-rank Wilson dslash against a torch reference
# ---------------------------------------------------------------------------


def run_dslash_wilson(case: Any = None, *, lattice: Any = None,
                      dtype: Any = None, seed: Any = None,
                      repeat: Any = None, warmup: Any = None,
                      mass: Any = None, kappa: Any = None, csw: Any = None,
                      gauge_seed: Any = None, source_seed: Any = None,
                      gauge_sigma: Any = None, source: Any = None,
                      gauge: Any = None, lattice_grid: Any = None,
                      mg_grid: Any = None, max_iter: Any = None,
                      atol: Any = None, device: Any = None,
                      dslash: Any = None, solver: Any = None,
                      source_site: Any = None, source_spin: Any = None,
                      source_color: Any = None, options: Any = None,
                      **extras: Any) -> dict[str, Any]:
    """Apply ``applyWilsonDslashQcu`` once per parity and compare to torch.

    Options honored (see the module docstring): ``mass``/``kappa``, ``csw``,
    ``gauge_sigma``, ``gauge``, ``source``, ``lattice_grid``, ``max_iter`` and
    ``atol`` are written into the private ABI controls before the first
    transaction and cross-checked afterwards (``*_applied`` observations).  The
    native entry point applies the *bare* hopping sum, so ``csw`` reaches
    ``argv[_CSW_]`` but does not enter the operator
    (``csw_used_by_operator=False``).

    Required observation (the ``kupl/dslash/wilson`` gate):

    ``relative``
        ``max|out - reference| / max|reference|`` over both target parities
        (even target from an odd source, odd target from an even source).

    Additional observations: ``relative_error`` (gate alias),
    ``relative_error_even_target``, ``relative_error_odd_target``,
    ``relative_error_full_wilson`` (full-lattice Wilson operator with the run's
    ``kappa``, so ``mass`` is numerically exercised), ``true_residual`` (alias),
    norms, ``seconds`` (median native transaction time honoring
    ``repeat``/``warmup``), ``samples_seconds``, lifecycle counters,
    input-immutability flags and ``global_abi_unchanged``.
    """
    option = _options(
        case, lattice=lattice, dtype=dtype, seed=seed, repeat=repeat,
        warmup=warmup, mass=mass, kappa=kappa, csw=csw,
        gauge_sigma=gauge_sigma, source=source, gauge=gauge,
        lattice_grid=lattice_grid, mg_grid=mg_grid, max_iter=max_iter,
        atol=atol, device=device, dslash=dslash, solver=solver,
        gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, options=options,
        default_csw=DEFAULT_CSW, default_source="random", extras=extras)
    _require_operator(option, expected_dslash="wilson")

    qcu, runtime, gauge_data, source_data, started = _prepare(
        option, plan=0, parity=0)
    import torch

    gauge = gauge_data["compact"]
    source_full = source_data["compact"]
    # The ABI operand is one parity block: target 0 is fed by the odd source
    # and target 1 by the even source.
    hop_source = {0: source_full[1].contiguous(),
                  1: source_full[0].contiguous()}
    expected = (4, 3, *option.compact)
    outputs = {0: torch.empty(expected, dtype=runtime.dtype),
               1: torch.empty(expected, dtype=runtime.dtype)}
    global_before = _global_abi_snapshot()
    gauge_before = gauge.clone()
    source_before = source_full.clone()

    def apply_parities() -> None:
        for target in (0, 1):
            # The parity is a control read by applyInitQcu, so it must be set
            # before the transaction opens and stay fixed inside it.
            runtime.params[runtime.define._PARITY_] = target
            with runtime.transaction("applyWilsonDslashQcu") as operation:
                operation.check(
                    (outputs[target], hop_source[target], gauge),
                    (expected, expected, None))
                operation(outputs[target], hop_source[target], gauge)

    samples = _time_samples(apply_parities, option.repeat, option.warmup)

    errors: dict[int, float] = {}
    full_errors: dict[int, float] = {}
    reference_norm = 0.0
    output_norm = 0.0
    for target in (0, 1):
        reference = _bare_dslash_reference(
            torch, hop_source[target], gauge, target)
        full_reference = _full_wilson_reference(
            torch, hop_source[target], gauge, target, runtime.kappa,
            option.shape)
        errors[target] = _relative_error(outputs[target], reference)
        full_errors[target] = _relative_error(outputs[target], full_reference)
        reference_norm = max(reference_norm,
                             float(torch.linalg.vector_norm(reference)))
        output_norm = max(output_norm,
                          float(torch.linalg.vector_norm(outputs[target])))

    relative = max(errors.values())
    observations = _common_observations(
        option, runtime, gauge_data, 0, _library_text(qcu), started)
    observations.update(
        check="dslash_wilson",
        equation="wilson_hopping",
        abi_symbol="applyWilsonDslashQcu",
        csw_used_by_operator=False,
        mass_used_by_operator=False,
        source_parity={"even_target": 1, "odd_target": 0},
        relative=relative,
        relative_error=relative,
        relative_error_even_target=errors[0],
        relative_error_odd_target=errors[1],
        relative_error_full_wilson=max(full_errors.values()),
        relative_error_full_wilson_even=full_errors[0],
        relative_error_full_wilson_odd=full_errors[1],
        true_residual=relative,
        output_norm=output_norm,
        reference_norm=reference_norm,
        output_finite=bool(torch.isfinite(outputs[0]).all()
                           and torch.isfinite(outputs[1]).all()),
        notes=[
            "applyWilsonDslashQcu applies the bare hopping sum: mass/kappa "
            "reach argv[_MASS_] and the full-Wilson reference, csw only "
            "reaches argv[_CSW_]",
        ],
    )
    return _finish(observations, runtime, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge, gauge_before)),
        "source_unchanged": bool(torch.equal(source_full, source_before)),
    })


# ---------------------------------------------------------------------------
# Runner: clover dslash (applyCloverDslashQcu) against a torch reference
# ---------------------------------------------------------------------------


def _native_clover_terms(torch: Any, runtime: _PrivateRuntime, gauge: Any
                         ) -> tuple[list[Any], list[Any]]:
    """Build the native clover term and its inverse for both parities.

    One ``applyCloversQcu`` transaction per parity; the parity is a control read
    by ``applyInitQcu``, so it is written before the transaction opens and left
    untouched inside it.  The runtime's previous plan is restored afterwards.
    """
    shape = (4, 3, 4, 3, *runtime.option.compact)
    previous_plan = int(runtime.params[runtime.define._SET_PLAN_])
    runtime.configure(plan=runtime.define._SET_PLAN2_)
    terms: list[Any] = []
    inverses: list[Any] = []
    try:
        for parity in (0, 1):
            term = runtime.empty(shape)
            inverse = runtime.empty(shape)
            runtime.params[runtime.define._PARITY_] = parity
            with runtime.transaction("applyCloversQcu") as operation:
                operation.check((term, inverse, gauge), (shape, shape, None))
                operation(term, inverse, gauge)
            terms.append(term)
            inverses.append(inverse)
    finally:
        runtime.configure(plan=previous_plan)
    return terms, inverses


def _torch_clover_terms(torch: Any, shape: tuple[int, int, int, int],
                        full_gauge: Any, kappa: float, csw: float
                        ) -> dict[str, Any]:
    """Torch-side clover term ``I + T`` and its inverse, both parities.

    Returned in the ABI layout ``[4, 3, 4, 3, X, Y, Z, T/2]`` per parity, so a
    caller can compare the native ``applyCloversQcu`` output directly.  This is
    the independent convention: it comes from ``pyqcu.torch.dslash`` only.
    """
    from pyqcu import tools
    from pyqcu.torch import dslash

    compact = (*shape[:3], shape[3] // 2)
    term = dslash.add_I(
        dslash.make_clover(full_gauge, kappa=torch.tensor([float(kappa)]),
                           csw=float(csw), verbose=False),
        verbose=False)
    inverse = dslash.inverse(term, verbose=False)
    split_term = tools.oooxyzt2poooxyzt(term.reshape(12, 12, *shape))
    split_inverse = tools.oooxyzt2poooxyzt(inverse.reshape(12, 12, *shape))
    terms = [split_term[parity].reshape(4, 3, 4, 3, *compact).contiguous()
             for parity in (0, 1)]
    inverses = [split_inverse[parity].reshape(4, 3, 4, 3, *compact)
                .contiguous() for parity in (0, 1)]
    return {
        "term": terms,
        "inverse": inverses,
        "term_abs_max": max(float(value.abs().max()) for value in terms),
        "inverse_abs_max": max(float(value.abs().max()) for value in inverses),
    }


def run_dslash_clover(case: Any = None, *, lattice: Any = None,
                      dtype: Any = None, seed: Any = None,
                      repeat: Any = None, warmup: Any = None,
                      mass: Any = None, kappa: Any = None, csw: Any = None,
                      gauge_seed: Any = None, source_seed: Any = None,
                      gauge_sigma: Any = None, source: Any = None,
                      gauge: Any = None, lattice_grid: Any = None,
                      mg_grid: Any = None, max_iter: Any = None,
                      atol: Any = None, device: Any = None,
                      dslash: Any = None, solver: Any = None,
                      source_site: Any = None, source_spin: Any = None,
                      source_color: Any = None, options: Any = None,
                      **extras: Any) -> dict[str, Any]:
    """Apply ``applyCloverDslashQcu`` once per parity and compare to torch.

    The native entry point applies ``A^-1 * D_W`` where ``A = I + c_sw * T`` is
    the clover term and ``D_W`` the bare hopping sum, i.e. the operator the
    clover solver inverts.  The runner therefore

    1. builds the native clover term and its inverse with ``applyCloversQcu``
       (one ordinary transaction per parity);
    2. applies ``applyCloverDslashQcu`` per parity;
    3. rebuilds the reference from the **native** inverse with the torch
       matrix-vector product on top of the torch bare dslash.

    Required observation (the ``kupl/dslash/clover`` gate):

    ``relative_error``
        ``max|out - give_clover(bare_dslash, A^-1)| / max|reference|`` over
        both target parities (``relative`` is the same value).

    Gate rationale: the reference consumes the native ``A^-1``, so the gated
    number isolates every native contribution *except* the clover-term
    construction, which is compared separately and reported ungated as
    ``clover_term_relative`` / ``clover_inverse_relative`` (native vs
    ``pyqcu.torch.dslash.make_clover``).  In c128 the two constructions agree to
    ~1e-9, above ``policy``'s c128 ``compare`` gate of 1e-10, so gating that
    comparison would report a false numerical failure; it is evidence, not a
    verdict.

    Additional observations: ``clover_identity_relative`` (``max|A A^-1 - I|``),
    ``clover_abs_max``, ``clover_inverse_abs_max``,
    ``relative_error_even_target`` / ``relative_error_odd_target``, timings,
    lifecycle and immutability flags.
    """
    option = _options(
        case, lattice=lattice, dtype=dtype, seed=seed, repeat=repeat,
        warmup=warmup, mass=mass, kappa=kappa, csw=csw,
        gauge_sigma=gauge_sigma, source=source, gauge=gauge,
        lattice_grid=lattice_grid, mg_grid=mg_grid, max_iter=max_iter,
        atol=atol, device=device, dslash=dslash, solver=solver,
        gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, options=options,
        default_csw=DEFAULT_CLOVER_CSW, default_source="random",
        extras=extras)
    _require_operator(option, expected_dslash="clover")

    qcu, runtime, gauge_data, source_data, started = _prepare(
        option, plan=0, parity=0)
    import torch

    from pyqcu.torch import dslash as dslash_module

    gauge = gauge_data["compact"]
    source_full = source_data["compact"]
    # The ABI operand is one parity block: target 0 is fed by the odd source
    # and target 1 by the even source.
    hop_source = {0: source_full[1].contiguous(),
                  1: source_full[0].contiguous()}
    expected = (4, 3, *option.compact)
    global_before = _global_abi_snapshot()
    gauge_before = gauge.clone()
    source_before = source_full.clone()

    native_terms, native_inverses = _native_clover_terms(torch, runtime, gauge)
    torch_clover = _torch_clover_terms(
        torch, option.shape, gauge_data["full"], runtime.kappa, option.csw)
    term_relative = max(
        _relative_error(native_terms[parity], torch_clover["term"][parity])
        for parity in (0, 1))
    inverse_relative = max(
        _relative_error(native_inverses[parity],
                        torch_clover["inverse"][parity])
        for parity in (0, 1))
    identity_relative = 0.0
    identity = torch.zeros(12, 12)
    index = torch.arange(12)
    identity[index, index] = 1.0
    for parity in (0, 1):
        product = torch.einsum(
            "ijx,jkx->ikx", native_terms[parity].reshape(12, 12, -1),
            native_inverses[parity].reshape(12, 12, -1))
        reference = identity.reshape(12, 12, 1).expand_as(product)
        identity_relative = max(identity_relative,
                                float((product - reference).abs().max()))

    outputs = {0: torch.empty(expected, dtype=runtime.dtype),
               1: torch.empty(expected, dtype=runtime.dtype)}

    def apply_parities() -> None:
        for target in (0, 1):
            runtime.params[runtime.define._PARITY_] = target
            with runtime.transaction("applyCloverDslashQcu") as operation:
                operation.check(
                    (outputs[target], hop_source[target], gauge),
                    (expected, expected, None))
                operation(outputs[target], hop_source[target], gauge)

    samples = _time_samples(apply_parities, option.repeat, option.warmup)

    errors: dict[int, float] = {}
    reference_norm = 0.0
    output_norm = 0.0
    for target in (0, 1):
        bare = _bare_dslash_reference(
            torch, hop_source[target], gauge, target)
        reference = dslash_module.give_clover(
            src=bare, clover_term=native_inverses[target], verbose=False)
        errors[target] = _relative_error(outputs[target], reference)
        reference_norm = max(reference_norm,
                             float(torch.linalg.vector_norm(reference)))
        output_norm = max(output_norm,
                          float(torch.linalg.vector_norm(outputs[target])))

    relative = max(errors.values())
    observations = _common_observations(
        option, runtime, gauge_data, 0, _library_text(qcu), started)
    observations.update(
        check="dslash_clover",
        equation="clover_inverse_times_wilson_hopping",
        abi_symbol="applyCloverDslashQcu",
        clover_abi_symbol="applyCloversQcu",
        csw_used_by_operator=True,
        mass_used_by_operator=True,
        reference="give_clover(bare_dslash, native_clover_inverse)",
        source_parity={"even_target": 1, "odd_target": 0},
        relative=relative,
        relative_error=relative,
        relative_error_even_target=errors[0],
        relative_error_odd_target=errors[1],
        true_residual=relative,
        clover_term_relative=term_relative,
        clover_inverse_relative=inverse_relative,
        clover_identity_relative=identity_relative,
        clover_abs_max=torch_clover["term_abs_max"],
        clover_inverse_abs_max=torch_clover["inverse_abs_max"],
        native_clover_abs_max=max(float(value.abs().max())
                                  for value in native_terms),
        native_clover_inverse_abs_max=max(float(value.abs().max())
                                          for value in native_inverses),
        output_norm=output_norm,
        reference_norm=reference_norm,
        output_finite=bool(torch.isfinite(outputs[0]).all()
                           and torch.isfinite(outputs[1]).all()),
        notes=[
            "clover_term_relative compares the native applyCloversQcu term "
            "with pyqcu.torch.dslash.make_clover+add_I; it is reported, not "
            "gated (c128 agreement is ~1e-9, above the 1e-10 compare "
            "tolerance)",
        ],
    )
    return _finish(observations, runtime, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge, gauge_before)),
        "source_unchanged": bool(torch.equal(source_full, source_before)),
    })


# ---------------------------------------------------------------------------
# Solve helpers
# ---------------------------------------------------------------------------


def _resolve_parity(value: Any) -> int:
    parity = int(value)
    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, got {parity}")
    return parity


def _full_wilson_residual(torch: Any, solution: Any, source: Any,
                          full_gauge: Any, kappa: float) -> dict[str, Any]:
    """Independent ``||D_W x - b|| / ||b||`` on the full lattice.

    ``pyqcu.test.common.relative_residual`` is used when the harness provides
    it, so the harness and the runner share one definition of the gate; the
    operator application itself is always the torch full-lattice Wilson
    operator, never the solver recurrence.
    """
    from pyqcu import tools
    from pyqcu.torch import dslash

    started = time.perf_counter()
    full_solution = tools.poooxyzt2oooxyzt(solution)
    full_source = tools.poooxyzt2oooxyzt(source)
    kappa_tensor = torch.tensor([float(kappa)])
    cache: dict[str, Any] = {}

    def _matvec(vector: Any) -> Any:
        applied = dslash.give_wilson(vector, full_gauge, kappa_tensor,
                                     with_I=True)
        cache["applied"] = applied
        return applied

    try:
        from .common import relative_residual as harness_relative_residual
    except Exception:
        harness_relative_residual = None
    if harness_relative_residual is not None:
        value = float(harness_relative_residual(
            _matvec, full_solution, full_source))
        applied = cache.get("applied")
        if applied is None:
            applied = _matvec(full_solution)
    else:
        applied = _matvec(full_solution)
        denominator = float(torch.linalg.vector_norm(full_source))
        numerator = float(torch.linalg.vector_norm(applied - full_source))
        value = math.inf if denominator == 0.0 else numerator / denominator
    residual = applied - full_source
    return {
        "relative_residual": value,
        "source_norm": float(torch.linalg.vector_norm(full_source)),
        "residual_norm": float(torch.linalg.vector_norm(residual)),
        "solution_norm": float(torch.linalg.vector_norm(full_solution)),
        "reference_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Runner: full single-rank BiCGStab solve
# ---------------------------------------------------------------------------


def run_solve_bistabcg(case: Any = None, *, lattice: Any = None,
                       dtype: Any = None, seed: Any = None,
                       repeat: Any = None, warmup: Any = None,
                       mass: Any = None, kappa: Any = None, csw: Any = None,
                       gauge_seed: Any = None, source_seed: Any = None,
                       gauge_sigma: Any = None, source: Any = None,
                       gauge: Any = None, lattice_grid: Any = None,
                       mg_grid: Any = None, max_iter: Any = None,
                       atol: Any = None, device: Any = None,
                       dslash: Any = None, solver: Any = None,
                       parity: int = 1, source_site: Any = None,
                       source_spin: Any = None, source_color: Any = None,
                       options: Any = None,
                       **extras: Any) -> dict[str, Any]:
    """Solve a Wilson system with the native KUPL BiCGStab.

    One ``applyWilsonBistabCgQcu`` transaction performs the complete solve; the
    iteration count is read from the same slot with the ABI's generic
    ``getCloverBistabCgIterationsQcu`` getter.  The reported residual is an
    *independent* true residual on the full lattice::

        || D_W psi - b || / || b ||,  D_W = give_wilson(with_I=True)

    computed with the torch operator from the raw solution, never from the
    solver recurrence.  The default solver tolerance is ``1e-6`` for c64 (the
    tier-0 case) and ``1e-10`` for c128, whose ``native_solve`` gate is 1e-8.

    Required observations (the ``kupl/solve/bistabcg`` gate):

    ``relative_residual``
        Independent full-Wilson ``||D psi - b|| / ||b||``.
    ``iterations``
        Native BiCGStab iteration count read from the same slot.
    ``max_iter``
        Iteration budget, so the gate can compare ``iterations < max_iter``.

    Additional observations: ``relative_error``/``true_residual`` (aliases of
    ``relative_residual``), ``hit_max_iter``, norms, ``seconds`` (median solve
    transaction time honoring ``repeat``/``warmup``), ``samples_seconds``,
    lifecycle counters, input-immutability flags and ``global_abi_unchanged``.
    """
    option = _options(
        case, lattice=lattice, dtype=dtype, seed=seed, repeat=repeat,
        warmup=warmup, mass=mass, kappa=kappa, csw=csw,
        gauge_sigma=gauge_sigma, source=source, gauge=gauge,
        lattice_grid=lattice_grid, mg_grid=mg_grid, max_iter=max_iter,
        atol=atol, device=device, dslash=dslash, solver=solver,
        gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, options=options,
        default_csw=DEFAULT_CSW, default_source="point", extras=extras)
    _require_operator(option, expected_solver="bistabcg")
    parity = _resolve_parity(parity)

    qcu, runtime, gauge_data, source_data, started = _prepare(
        option, plan=1, parity=parity)
    import torch

    gauge = gauge_data["compact"]
    source = source_data["compact"]
    solution = torch.zeros_like(source)
    global_before = _global_abi_snapshot()
    gauge_before = gauge.clone()
    source_before = source.clone()
    state = {"iterations": 0}

    def solve_once() -> None:
        with runtime.transaction("applyWilsonBistabCgQcu") as operation:
            operation.check((solution, source, gauge),
                            (tuple(source.shape), tuple(source.shape), None))
            operation(solution, source, gauge)
            state["iterations"] = int(
                operation.read("getCloverBistabCgIterationsQcu"))

    samples = _time_samples(solve_once, option.repeat, option.warmup)
    iterations = int(state["iterations"])
    residual = _full_wilson_residual(
        torch, solution, source, gauge_data["full"], runtime.kappa)

    observations = _common_observations(
        option, runtime, gauge_data, parity, _library_text(qcu), started)
    observations.update(
        check="solve_bistabcg",
        equation="wilson",
        abi_symbol="applyWilsonBistabCgQcu",
        iteration_getter="getCloverBistabCgIterationsQcu",
        csw_used_by_operator=False,
        mass_used_by_operator=True,
        parity=parity,
        iterations=iterations,
        iterations_available=iterations > 0,
        hit_max_iter=iterations >= option.max_iter,
        max_iter=int(option.max_iter),
        relative_residual=residual["relative_residual"],
        relative_error=residual["relative_residual"],
        true_residual=residual["relative_residual"],
        source_norm=residual["source_norm"],
        residual_norm=residual["residual_norm"],
        solution_norm=residual["solution_norm"],
        reference_seconds=residual["reference_seconds"],
    )
    return _finish(observations, runtime, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge, gauge_before)),
        "source_unchanged": bool(torch.equal(source, source_before)),
    })


# ---------------------------------------------------------------------------
# Runner: full single-rank conjugate-gradient solve
# ---------------------------------------------------------------------------


def run_solve_cg(case: Any = None, *, lattice: Any = None,
                 dtype: Any = None, seed: Any = None,
                 repeat: Any = None, warmup: Any = None,
                 mass: Any = None, kappa: Any = None, csw: Any = None,
                 gauge_seed: Any = None, source_seed: Any = None,
                 gauge_sigma: Any = None, source: Any = None,
                 gauge: Any = None, lattice_grid: Any = None,
                 mg_grid: Any = None, max_iter: Any = None,
                 atol: Any = None, device: Any = None,
                 dslash: Any = None, solver: Any = None,
                 parity: int = 1, source_site: Any = None,
                 source_spin: Any = None, source_color: Any = None,
                 options: Any = None,
                 **extras: Any) -> dict[str, Any]:
    """Solve a Wilson system with the native KUPL conjugate-gradient solver.

    One ``applyWilsonCgQcu`` transaction performs the complete solve.  The
    native CG works on the Schur complement of the target parity, so its
    iteration counter is read with the ABI's generic
    ``getCloverBistabCgIterationsQcu`` getter and reported only when it *moved*
    during the solve (``iterations_available``), never as a hard-coded or
    guessed number.  With the KUPL library it moves: ``applyWilsonCgQcu`` ->
    ``solve_compact_cg`` -> ``last_iterations_``.

    The reported residual is the independent full-Wilson true residual::

        || D_W psi - b || / || b ||,  D_W = give_wilson(with_I=True)

    computed from the raw solution with the torch operator, never from the
    solver recurrence, so ``relative_residual`` is directly comparable with the
    BiCGStab runner's number.  The native CG floors at its own acceptance
    level: a c128 solve lands near 1e-8 rather than at ``argv[_ATOL_]``.

    Required observations for a ``solver`` gate: ``relative_residual``,
    ``iterations``.  A case that gates ``iterations > 0`` must also accept a
    backend without an iteration getter; ``iterations_available`` says whether
    the number is real.

    Additional observations: ``relative_error``/``true_residual`` (aliases),
    ``iterations_source``, ``iterations_before``/``iterations_after``,
    ``hit_max_iter``, ``native_reported_converged``, norms, timings, lifecycle
    and immutability flags.
    """
    option = _options(
        case, lattice=lattice, dtype=dtype, seed=seed, repeat=repeat,
        warmup=warmup, mass=mass, kappa=kappa, csw=csw,
        gauge_sigma=gauge_sigma, source=source, gauge=gauge,
        lattice_grid=lattice_grid, mg_grid=mg_grid, max_iter=max_iter,
        atol=atol, device=device, dslash=dslash, solver=solver,
        gauge_seed=gauge_seed, source_seed=source_seed,
        source_site=source_site, source_spin=source_spin,
        source_color=source_color, options=options,
        default_csw=DEFAULT_CSW, default_source="point", extras=extras)
    _require_operator(option, expected_solver="cg", expected_dslash="wilson")
    parity = _resolve_parity(parity)

    qcu, runtime, gauge_data, source_data, started = _prepare(
        option, plan=1, parity=parity)
    import torch

    gauge = gauge_data["compact"]
    source = source_data["compact"]
    solution = torch.zeros_like(source)
    global_before = _global_abi_snapshot()
    gauge_before = gauge.clone()
    source_before = source.clone()
    state: dict[str, Any] = {"before": 0, "after": 0, "captured": False}

    def solve_once() -> None:
        with runtime.transaction("applyWilsonCgQcu") as operation:
            operation.check((solution, source, gauge),
                            (tuple(source.shape), tuple(source.shape), None))
            before = int(operation.read("getCloverBistabCgIterationsQcu"))
            operation(solution, source, gauge)
            after = int(operation.read("getCloverBistabCgIterationsQcu"))
            if not state["captured"]:
                state.update(before=before, after=after, captured=True)

    samples = _time_samples(solve_once, option.repeat, option.warmup)
    moved = bool(state["after"] > state["before"])
    iterations = int(state["after"]) if moved else None
    residual = _full_wilson_residual(
        torch, solution, source, gauge_data["full"], runtime.kappa)

    observations = _common_observations(
        option, runtime, gauge_data, parity, _library_text(qcu), started)
    observations.update(
        check="solve_cg",
        equation="wilson",
        abi_symbol="applyWilsonCgQcu",
        iteration_getter="getCloverBistabCgIterationsQcu",
        iterations_source=("getCloverBistabCgIterationsQcu"
                           if moved else None),
        iterations_before=int(state["before"]),
        iterations_after=int(state["after"]),
        csw_used_by_operator=False,
        mass_used_by_operator=True,
        parity=parity,
        iterations=iterations,
        iterations_available=moved,
        hit_max_iter=bool(iterations is not None
                          and iterations >= option.max_iter),
        max_iter=int(option.max_iter),
        native_reported_converged=bool(
            moved and iterations is not None
            and iterations < option.max_iter),
        relative_residual=residual["relative_residual"],
        relative_error=residual["relative_residual"],
        true_residual=residual["relative_residual"],
        source_norm=residual["source_norm"],
        residual_norm=residual["residual_norm"],
        solution_norm=residual["solution_norm"],
        reference_seconds=residual["reference_seconds"],
        notes=[
            "the KUPL CG floors at its internal true-residual acceptance "
            "level; a c128 solve lands near 1e-8 rather than at argv[_ATOL_]",
            "a native CG failure aborts the process (KUPL ABI contract), so "
            "atol/max_iter are validated before the transaction",
        ],
    )
    return _finish(observations, runtime, global_before, samples, {
        "gauge_unchanged": bool(torch.equal(gauge, gauge_before)),
        "source_unchanged": bool(torch.equal(source, source_before)),
    })
