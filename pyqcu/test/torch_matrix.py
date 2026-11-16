"""Torch parameter-matrix runners for the formal :mod:`pyqcu.test` port.

:mod:`pyqcu.test.torch_qcu` owns the tier-0 torch cases: one lattice
(``8x8x8x16``), one dtype (``c64``), one configuration per case.  This module
adds the *matrix* around that point: the same torch physics at every declared
lattice preset, precision, operator, gauge field, source, solver and multigrid
variant, with explicit file inputs and a standalone harness that turns the
observations into real ``PASS``/``FAIL`` verdicts through
:func:`pyqcu.test.common.evaluate_expect`.

Why a second module
-------------------
The tier-0 matrix is a gate, not a sweep: 21 cases, seconds each, one number
per case.  The questions a release or a port needs next are matrix questions -
"does the Wilson identity still hold at ``24^3x64`` in ``c128``?", "does
strict multigrid converge on the production gauge?", "does the clover inverse
survive ``csw = 1.16``?" - and they must be answered with the same gates, the
same skip vocabulary, the same report shape and the same runner contract as
tier 0.  That is what this module provides.

It is deliberately **additive**: it imports no core module at import time, it
edits nothing, it registers nothing into :mod:`pyqcu.test.units`, and
``python -m pyqcu.test --list`` never imports it.  A caller opts in with
``from pyqcu.test import torch_matrix`` (or
``python -m pyqcu.test.torch_matrix``).

Lattice presets
---------------
============================  =========================================
preset                        shape
============================  =========================================
``quick`` (default)           ``[8, 8, 8, 16]`` - the project gate
``L24`` (heavy)               ``[24, 24, 24, 64]`` - production
``L32`` (heavy)               ``[32, 32, 32, 64]`` - production
============================  =========================================

``L24`` and ``L32`` are **exposed but never run by default**: they are
declared (``LATTICE_PRESETS``, ``HEAVY_PRESETS``, ``HEAVY_MATRIX``), they can
be selected explicitly (``--preset L24``, ``matrix_cases(presets="L24")``),
and every runner accepts them, but :data:`MATRIX` (the default selection) and
``python -m pyqcu.test.torch_matrix`` without ``--preset`` stay on
``quick``.  Any other four-extent shape is accepted too
(``resolve_lattice([16, 16, 16, 16])``), subject to the policy rules: four
positive **even** extents of at least 4.

Heavy presets also carry a memory pre-flight: a field-sized lower bound is
compared against the memory the target device reports, and a run that cannot
fit raises ``Skip("device_unavailable: ...")`` with the estimate, the budget
and the available bytes in the reason - never a fake pass and never a
``MemoryError`` traceback.  An allocation that still runs out of memory is
converted to the same skip.

Public surface
--------------
``resolve_lattice`` / ``resolve_dtype``  preset and alias resolution
``lattice_preset`` / ``is_heavy_lattice`` / ``lattice_tag``
``MATRIX`` / ``HEAVY_MATRIX`` / ``matrix_cases`` / ``select_cases``
``MatrixCase``  one declared matrix row (tag, runner, options, ``expect``)
``run_case`` / ``run_matrix`` / ``matrix_payload``  execution + gating
``main``  the standalone ``python -m pyqcu.test.torch_matrix`` harness
``RUNNERS`` / ``TAG_RUNNERS``  runner registry

Runners (all report observations, never a verdict)
--------------------------------------------------
=============================  ================================================
runner                         checks
=============================  ================================================
``run_gauge``                  ``unit``/``random``/``ones`` links, ``sigma``,
                               seed reproducibility, distinct-seed sensitivity
``run_dslash``                 Wilson and/or clover ``matvec`` against an
                               independent ``give_wilson``(+``give_clover``)
                               reference and the compact even/odd path
``run_clover``                 clover term build, ``(I + T)`` inverse round trip,
                               ``csw = 0`` limit
``run_smear``                  stout (serial vs ``support_parallel``) and
                               Wuppertal (constant/time-leak/one-step) smearing
``run_bicgstab``               full-system BiCGStab + independent true residual
``run_cg``                     even-odd Schur CA-CG + independent full residual
``run_mg_legacy``              explicit torch ``LegacyMultigrid`` variant
``run_mg_strict``              explicit torch ``StrictMultigrid`` variant
``run_solve``                  convenience dispatch onto ``run_cg``/``run_bicgstab``
=============================  ================================================

Independent true residuals
--------------------------
Every solve reports ``relative_residual = ||D x - b|| / ||b||`` where ``D x``
is rebuilt from the public ``give_wilson(with_I=True)`` (+ ``give_clover``)
helpers - never from the solver's own recurrence and never from the operator
whose convergence is being claimed.  ``converged`` is that number against the
requested ``tolerance``; the solver's internal stop flag is only an
observation (``native_final_residual``).

Run options (all optional keyword arguments)
-------------------------------------------
=========================  =====================================================
option                     meaning
=========================  =====================================================
``lattice``                ``"quick"``/``"L24"``/``"L32"``/four extents
``preset``                 preset name used when ``lattice`` is absent
``dtype``                  ``c64``/``c128`` plus ``complex64``/``complex128``/
                           ``float32``/``float64``/``single``/``double``/
                           ``fp32``/``fp64`` aliases
``seed``                   deterministic gauge/source seed (default 42)
``device``/``device_id``   ``auto``/``cpu`` -> CPU, ``gpu``/``cuda[:N]``,
                           ``dcu[:N]`` (ROCm) and ``npu[:N]``; an unusable
                           device is ``Skip(device_unavailable)``
``mass``/``kappa``         one physical knob: ``kappa = 1/(2*mass + 8)``
                           (runner default ``kappa = 0.125``, ``mass = 0``)
``csw``                    clover coefficient (default 1.0, ``0`` = Wilson)
``gauge``                  ``random`` (default), ``unit`` or ``ones``
``gauge_sigma``            Gaussian width of the generated SU(3) field
``gauge_file``             HDF5 gauge field ``[3,3,4,x,y,z,t]`` (see below)
``source``                 ``point``/``ones``/``random``
``source_file``            HDF5 fermion field ``[4,3,x,y,z,t]``
``dslash``                 ``wilson`` (default), ``clover``, or ``all``/``both``
``solver``                 ``cg``/``cacg`` (Schur CA-CG), ``bicgstab``/
                           ``bistabcg``, ``mr``, ``fgmres``; ``all``/``mg``/
                           ``multigrid`` select the CG default
``smear``                  ``stout`` (default) or ``wuppertal``
``mg_grid``/``mg_levels``  multigrid block grid / hierarchy depth
``mg_nu_pre``/``mg_nu_post``  strict-smoother counts
``max_iter``/``restart``   iteration budget / Krylov restart length
``tolerance``              gate tolerance for ``converged`` (default 1e-6)
``solver_tolerance``       tolerance handed to the solver (default derived)
``n_krylov``/``nvec``/``omega``/``support_parity``/``min_size``/
``materialize_coarse``/``nstep``/``rho``/``su3_tol``
``threads``                ``torch.set_num_threads`` for the run
``repeat``/``warmup``      timing samples / discarded samples (``warmup >= 1``)
``memory_budget``          fraction of the reported device memory a run may
                           plan for (default 0.8)
``capture_stdout``         buffer the core's prints into ``log_tail``
=========================  =====================================================

File inputs
-----------
``gauge_file`` / ``source_file`` name an HDF5 file with a ``data`` dataset in
the documented PyQCU layout (``[3,3,4,x,y,z,t]`` gauge, ``[4,3,x,y,z,t]``
source).  When :mod:`pyqcu.test.io` is available it is consulted first
(``read_gauge``/``load_gauge``/``ensure_gauge`` and the ``*_source``
counterparts, called with only the keyword arguments they declare); otherwise
the dataset is read directly with ``h5py``.  A missing file is a ``ValueError``
(a usage error), a missing ``h5py`` or ``numpy`` is ``Skip(h5py_mpi)``, and a
field that does not carry the declared shape is a ``ValueError`` naming both
shapes - a wrong file never silently produces a number.

Import rules
------------
Importing this module is stdlib-only: ``torch``, ``numpy``, ``h5py``,
``pyqcu.lattice``, ``pyqcu.tools`` and ``pyqcu.torch`` are imported lazily
inside the runners.  ``--list`` therefore works in an interpreter without any
backend, exactly like ``python -m pyqcu.test --list``.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import math
import os
import signal
import statistics
import sys
import time
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import __version__ as _PACKAGE_VERSION
from . import policy
from .common import (
    Case,
    Result,
    Skip,
    evaluate_expect,
    json_safe,
    mpi_info,
    summarize,
    time_samples,
    write_json_atomic,
)

__all__ = [
    "DEFAULT_CSW",
    "DEFAULT_GAUGE_SIGMA",
    "DEFAULT_KAPPA",
    "DEFAULT_MEMORY_BUDGET",
    "DEFAULT_PRESET",
    "DEFAULT_SEED",
    "DEFAULT_TOLERANCE",
    "DTYPES",
    "DTYPE_ALIASES",
    "HEAVY_LATTICES",
    "HEAVY_MATRIX",
    "HEAVY_PRESETS",
    "LATTICE_PRESETS",
    "MATRIX",
    "MATRIX_PRESETS",
    "QUICK_LATTICE",
    "RUNNERS",
    "SMOKE_MATRIX",
    "TAG_RUNNERS",
    "MatrixCase",
    "available_memory_bytes",
    "estimate_memory_bytes",
    "is_heavy_lattice",
    "lattice_preset",
    "lattice_tag",
    "main",
    "matrix_cases",
    "matrix_payload",
    "resolve_dtype",
    "resolve_lattice",
    "run_bicgstab",
    "run_case",
    "run_cg",
    "run_clover",
    "run_dslash",
    "run_gauge",
    "run_matrix",
    "run_mg_legacy",
    "run_mg_strict",
    "run_smear",
    "run_solve",
    "select_cases",
    "torch_dtype",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: Project correctness lattice (``policy.QUICK_LATTICE``), the default selection.
QUICK_LATTICE: Tuple[int, int, int, int] = tuple(policy.QUICK_LATTICE)  # type: ignore[assignment]
#: Named lattice presets.  ``L24``/``L32`` are the production shapes; they are
#: declared here and excluded from the default selection on purpose.
LATTICE_PRESETS: Mapping[str, Tuple[int, int, int, int]] = MappingProxyType({
    "quick": QUICK_LATTICE,
    "L24": (24, 24, 24, 64),
    "L32": (32, 32, 32, 64),
})
#: Presets that must never run unless they are asked for explicitly.
HEAVY_PRESETS = frozenset({"L24", "L32"})
#: The heavy preset shapes, for callers that only need the geometries.
HEAVY_LATTICES: Mapping[str, Tuple[int, int, int, int]] = MappingProxyType(
    {name: LATTICE_PRESETS[name] for name in sorted(HEAVY_PRESETS)})
#: Preset used when the caller does not name one.
DEFAULT_PRESET = "quick"
#: Declared preset names in report order.
MATRIX_PRESETS: Tuple[str, ...] = ("quick", "L24", "L32")

#: Accepted preset aliases (normalised: lower case, no ``-``/``_``/spaces).
_PRESET_ALIASES: Mapping[str, str] = MappingProxyType({
    "q": "quick",
    "quick": "quick",
    "small": "quick",
    "default": "quick",
    "l8": "quick",
    "l24": "L24",
    "l32": "L32",
})

#: Canonical dtype tags (the report vocabulary is :mod:`pyqcu.test.policy`'s).
DTYPES: Tuple[str, str] = tuple(policy.DTYPE_CHOICES)  # type: ignore[assignment]
#: Every accepted dtype spelling -> canonical tag.  ``policy`` handles the six
#: documented aliases; the numeric-width spellings are added here so a matrix
#: row can be written ``single``/``double`` without a second parser.
DTYPE_ALIASES: Mapping[str, str] = MappingProxyType({
    "c64": "c64",
    "complex64": "c64",
    "float32": "c64",
    "f32": "c64",
    "fp32": "c64",
    "single": "c64",
    "csingle": "c64",
    "complexsingle": "c64",
    "c128": "c128",
    "complex128": "c128",
    "float64": "c128",
    "f64": "c128",
    "fp64": "c128",
    "double": "c128",
    "cdouble": "c128",
    "complexdouble": "c128",
})

#: ``kappa = 1/(2*mass + 8)`` at ``mass = 0`` - the tier-0 torch default.
DEFAULT_KAPPA = 0.125
#: Clover coefficient when the caller does not ask for another one.
DEFAULT_CSW = policy.DEFAULT_CSW
#: Gaussian width of the generated SU(3) background field.
DEFAULT_GAUGE_SIGMA = policy.DEFAULT_GAUGE_SIGMA
#: Reproducible gauge/source seed.
DEFAULT_SEED = policy.DEFAULT_SEED
#: Default ``converged`` threshold (the *request*; the case gate is
#: ``policy.RESIDUAL_GATES`` and is never relaxed by this value).
DEFAULT_TOLERANCE = 1e-6
#: Fraction of the reported device memory a run is allowed to plan for.
DEFAULT_MEMORY_BUDGET = 0.8
#: MG block grid used by the matrix rows (``policy.DEFAULT_MG_GRID``).
DEFAULT_MG_GRID: Tuple[int, int, int, int] = tuple(policy.DEFAULT_MG_GRID)  # type: ignore[assignment]

#: Complex numbers per site of each working tensor, used by the pre-flight.
#: ``stencil`` is one ``[12, 12, x, y, z, t]`` hopping block: the core stores
#: eight of them per operator (four wards x two directions), so a dslash is
#: dominated by ``12 * 12``-per-site tensors rather than by the spinor.
_COMPLEX_PER_SITE: Mapping[str, int] = MappingProxyType({
    "gauge": 3 * 3 * 4,        # [color, color, direction]
    "spinor": 4 * 3,           # [spin, color]
    "stencil": 12 * 12,        # [spin*color, spin*color] hopping block
    "clover": 4 * 3 * 4 * 3,   # [spin, spin, color, color]
})
_BYTES_PER_COMPLEX = {"c64": 8, "c128": 16}
#: Gauge-field-sized temporaries of the three ways a field is obtained.
#: ``random`` runs the whole-lattice SU(3) exponential map (~15 fields of
#: scratch), ``unit``/``ones`` are a direct construction (~3) and a file read
#: holds the file array plus the converted copy (~6).
_GAUGE_GENERATE_BUFFERS = 17
_GAUGE_CONSTANT_BUFFERS = 3
_GAUGE_FILE_BUFFERS = 6
#: Hopping blocks alive at once: eight ``[12, 12]`` stencils plus the two
#: parity-split pairs the ``support_parity=True`` build materialises.
_STENCIL_BUFFERS = 10
#: Extra gauge-sized scratch of ``make_clover`` (F, its adjoint and the twelve
#: rolled plaquette operands).
_CLOVER_SCRATCH_BUFFERS = 12

#: Per-case wall-clock budgets: the quick matrix is second-scale, a heavy row
#: is a production run and gets an hour before it is a TIMEOUT.
_QUICK_TIMEOUT = 300.0
_HEAVY_TIMEOUT = 7200.0

#: Option names reported in ``option_sources`` (argument vs default).
_OPTION_NAMES: Tuple[str, ...] = (
    "csw", "device", "device_id", "dslash", "gauge", "gauge_file",
    "gauge_sigma", "kappa", "lattice", "lattice_grid", "mass", "max_iter",
    "memory_budget", "mg_grid", "mg_levels", "mg_nu_post", "mg_nu_pre",
    "nstep", "preset", "repeat", "restart", "rho", "seed", "smear", "solver",
    "source", "source_file", "threads", "tolerance", "warmup",
)

_LATTICE_KEYS = ("lat", "lat_size", "lattice_size", "lattice_shape", "shape",
                 "lat_size_xyzt")
_DTYPE_KEYS = ("precision",)
_SEED_KEYS = ("rng_seed",)
_DEVICE_KEYS = ("dev",)
_CAPTURE_KEYS = ("capture",)

#: Device aliases: option spelling -> (torch device type, reported kind).
_DEVICE_ALIASES: Mapping[str, Tuple[str, str]] = MappingProxyType({
    "auto": ("cpu", "cpu"),
    "cpu": ("cpu", "cpu"),
    "cuda": ("cuda", "gpu"),
    "dcu": ("cuda", "dcu"),
    "gpu": ("cuda", "gpu"),
    "hip": ("cuda", "dcu"),
    "npu": ("npu", "npu"),
    "rocm": ("cuda", "dcu"),
})

#: ``pyqcu.torch.solver`` entry points addressable through ``solver``.
_KRYLOV_ENTRY_POINTS: Mapping[str, str] = MappingProxyType({
    "bistabcg": "bistabcg",
    "cacg": "cacg",
    "cg": "cacg",
    "fgmres": "fgmres",
    "mr": "mr",
})
#: ``solver`` choice -> implementation family.
_SOLVER_SELECTION: Mapping[str, str] = MappingProxyType({
    "all": "cg",
    "bicgstab": "bicgstab",
    "bistabcg": "bicgstab",
    "cacg": "cg",
    "cg": "cg",
    "fgmres": "fgmres",
    "mg": "cg",
    "mr": "mr",
    "multigrid": "cg",
})
#: Solver vocabulary accepted by the runners.
SOLVER_KINDS: Tuple[str, ...] = tuple(policy.SOLVER_CHOICES) + (
    "cacg", "bistabcg", "mr", "fgmres")
#: ``dslash`` choice -> the operators to exercise.
_DSLASH_SELECTION: Mapping[Optional[str], Tuple[str, ...]] = MappingProxyType({
    None: ("wilson",),
    "all": ("wilson", "clover"),
    "both": ("wilson", "clover"),
    "clover": ("clover",),
    "wilson": ("wilson",),
})
SMEAR_KINDS: Tuple[str, str] = ("stout", "wuppertal")

#: Observation keys with harness meaning (never reported as metrics).
_RESERVED_OBSERVATIONS = frozenset({"skip", "reason", "status", "seconds",
                                    "token"})

#: Modules cached by :func:`_load`.
_MODULES: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# lattice / dtype resolution
# ---------------------------------------------------------------------------
def _normalise_preset(text: Any) -> str:
    """Lower-case a preset spelling and strip ``-``/``_``/whitespace."""
    return (str(text).strip().lower().replace("-", "").replace("_", "")
            .replace(" ", ""))


def _preset_name(spec: Any) -> Optional[str]:
    """Return the canonical preset name ``spec`` spells, else ``None``."""
    return _PRESET_ALIASES.get(_normalise_preset(spec))


def resolve_lattice(spec: Any = None) -> Tuple[int, int, int, int]:
    """Resolve a lattice preset name or an explicit four-extent shape.

    ``None`` (and the empty string) mean the project gate ``[8, 8, 8, 16]``.
    A name is looked up in :data:`LATTICE_PRESETS` through
    :data:`_PRESET_ALIASES` (case/``-``/``_`` insensitive: ``l24`` == ``L24``).
    Anything else is handed to :func:`pyqcu.test.policy.validate_lattice`, so
    ``[8,8,8,16]``, ``"8,8,8,16"``, ``"8 8 8 16"`` and a four-int sequence are
    the same lattice and get the same validation (four positive even extents,
    each >= 4).
    """
    if spec is None:
        return QUICK_LATTICE
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return QUICK_LATTICE
        name = _preset_name(text)
        if name is not None:
            return LATTICE_PRESETS[name]
        try:
            return tuple(policy.validate_lattice(policy.parse_int_list(text)))  # type: ignore[return-value]
        except ValueError as error:
            raise ValueError(
                f"lattice {spec!r} is neither a preset "
                f"({', '.join(sorted(_PRESET_ALIASES))}) nor four positive even "
                f"extents >= 4: {error}") from error
    try:
        return tuple(policy.validate_lattice(policy.parse_int_list(spec)))  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(f"lattice {spec!r}: {error}") from error


def lattice_preset(spec: Any) -> Optional[str]:
    """Name the preset a lattice spec resolves to (``None`` when it is custom)."""
    shape = resolve_lattice(spec)
    for name, preset in LATTICE_PRESETS.items():
        if tuple(preset) == shape:
            return name
    return None


def is_heavy_lattice(spec: Any) -> bool:
    """True when ``spec`` resolves onto an ``L24``/``L32`` production shape."""
    return lattice_preset(spec) in HEAVY_PRESETS


def lattice_tag(spec: Any) -> str:
    """``x``-joined lattice label (``24x24x24x64``), used in reports."""
    return "x".join(str(int(extent)) for extent in resolve_lattice(spec))


def _normalise_dtype_token(text: Any) -> str:
    """Lower-case a dtype spelling and drop ``-``/``_``/spaces."""
    return str(text).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_dtype(spec: Any = None) -> str:
    """Resolve a dtype spelling onto the canonical ``c64``/``c128`` tag.

    Accepts the policy aliases (``complex64``/``complex128``/``float32``/
    ``float64``) plus ``single``/``double``, ``fp32``/``fp64`` and the
    ``c``-prefixed single/double spellings, and the ``str()`` form of a
    ``torch.dtype``/``numpy.dtype`` (``"torch.complex128"``,
    ``"complex64"``).  ``None`` is the policy default (``c64``).
    """
    if spec is None:
        return policy.DEFAULT_DTYPE
    candidates: List[str] = []
    if isinstance(spec, str):
        candidates.append(spec)
    else:
        candidates.extend((str(spec), str(spec).split(".")[-1]))
    for candidate in candidates:
        tag = DTYPE_ALIASES.get(_normalise_dtype_token(candidate))
        if tag is not None:
            return tag
    raise ValueError(
        f"dtype must be one of {', '.join(sorted(DTYPE_ALIASES))}, got {spec!r}")


def torch_dtype(tag: Any) -> Any:
    """Return the ``torch.dtype`` for a dtype tag (imports torch on demand)."""
    torch = _load().torch
    return torch.complex128 if resolve_dtype(tag) == "c128" else torch.complex64


# ---------------------------------------------------------------------------
# memory pre-flight
# ---------------------------------------------------------------------------
def estimate_memory_bytes(lattice: Any, dtype: Any, *, clover: bool = False,
                          spinor_buffers: int = 8, gauge_buffers: int = 2,
                          stencil_buffers: int = 0,
                          mg_levels: int = 1,
                          mg_grid: Any = DEFAULT_MG_GRID) -> int:
    """Lower-bound the working set of one torch run, in bytes.

    The count is in complex numbers per site, with one term per tensor the
    run keeps alive:

    ``gauge_buffers``    gauge-field-sized tensors (36/site).  Generating a
                         random SU(3) field costs about 17 of them because the
                         exponential map works on whole-lattice temporaries;
                         ``unit``/``ones`` need 3 and a file read needs 6.
    ``stencil_buffers``  ``[12, 12, x, y, z, t]`` hopping blocks (144/site).
                         Building an operator keeps eight (four wards times
                         two directions) and materialises two more per ward
                         while splitting them by parity.
    ``spinor_buffers``   spinor-sized work vectors (12/site): source,
                         destination, reference, Krylov basis, halo copies.
    ``clover``           the ``[4, 3, 4, 3, x, y, z, t]`` clover term
                         (144/site) and its builder's scratch.

    Each coarse multigrid level adds three spinor-sized vectors on a volume
    reduced by ``mg_grid``.

    The factors follow what the current core implementation actually
    allocates; they were calibrated against ``L24`` ``c64`` peak RSS (gauge
    generation ~4 GiB, operator + matvec ~15 GiB, clover ~15 GiB) so the number
    is a floor rather than a hopeful guess.  It exists so a heavy preset
    reports ``Skip(device_unavailable)`` with numbers instead of exhausting the
    machine; the budget fraction (:data:`DEFAULT_MEMORY_BUDGET`) leaves
    head-room for the core's own temporaries and the interpreter.
    """
    shape = resolve_lattice(lattice)
    sites = math.prod(int(extent) for extent in shape)
    item = _BYTES_PER_COMPLEX[resolve_dtype(dtype)]
    buffers = max(int(spinor_buffers), 0)
    units = (
        max(int(gauge_buffers), 0) * _COMPLEX_PER_SITE["gauge"]
        + max(int(stencil_buffers), 0) * _COMPLEX_PER_SITE["stencil"]
        + buffers * _COMPLEX_PER_SITE["spinor"]
        + (_COMPLEX_PER_SITE["clover"] if clover else 0)
    )
    total = sites * max(units, 1) * item
    levels = max(int(mg_levels), 1)
    if levels > 1:
        divisor = max(math.prod(int(value) for value in mg_grid), 1)
        for level in range(1, levels):
            total += (3 * sites * _COMPLEX_PER_SITE["spinor"] * item
                      // (divisor ** level))
    return int(total)


def _meminfo_available_bytes() -> Optional[int]:
    """``MemAvailable`` from ``/proc/meminfo`` (``None`` when unavailable)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(size, int) and pages > 0:
            return int(pages) * int(size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def available_memory_bytes(device: Any = None) -> Optional[int]:
    """Bytes a run may plan for on ``device`` (``None`` when unknowable).

    An accelerator reports its own free memory (``torch.cuda.mem_get_info``,
    ``torch.npu.mem_get_info``); the CPU path reads ``/proc/meminfo``.  A
    missing probe is reported as ``None`` and never becomes a skip: the
    pre-flight can only refuse a run it *knows* cannot fit.
    """
    kind = str(getattr(device, "type", device or "cpu"))
    index = int(getattr(device, "index", 0) or 0)
    if kind == "cuda":
        try:
            torch = _load().torch
            free, _total = torch.cuda.mem_get_info(index)
            return int(free)
        except Exception:  # noqa: BLE001 - a probe failure is "unknown"
            return None
    if kind == "npu":
        try:
            torch = _load().torch
            free, _total = torch.npu.mem_get_info(index)
            return int(free)
        except Exception:  # noqa: BLE001 - a probe failure is "unknown"
            return None
    return _meminfo_available_bytes()


def _is_out_of_memory(error: BaseException) -> bool:
    """True when ``error`` is an allocator failure (torch, numpy, host)."""
    name = type(error).__name__.lower()
    if "outofmemory" in name or "memoryerror" in name:
        return True
    text = str(error).lower()
    return ("out of memory" in text or "insufficient memory" in text
            or "cannot allocate memory" in text)


@contextlib.contextmanager
def _skip_on_oom(config: "_Config"):
    """Turn an allocator failure into ``Skip(device_unavailable)``.

    Every other exception is re-raised untouched: the harness owns the
    ``ERROR``/``TIMEOUT`` mapping and a runner must not hide a real bug.
    """
    try:
        yield
    except Skip:
        raise
    except Exception as error:  # noqa: BLE001 - only OOM is re-labelled
        if not _is_out_of_memory(error):
            raise
        estimate = config.memory.get("estimate_bytes")
        available = config.memory.get("available_bytes")
        detail = ""
        if estimate is not None:
            detail = f" (planned ~{float(estimate) / 2 ** 30:.2f} GiB"
            if available is not None:
                detail += f", {float(available) / 2 ** 30:.2f} GiB reported free"
            detail += ")"
        raise Skip(
            f"device_unavailable: {config.lattice_tag()} {config.dtype_tag} "
            f"ran out of memory on {config.device}{detail}: "
            f"{type(error).__name__}: {error}") from error


# ---------------------------------------------------------------------------
# lazy imports
# ---------------------------------------------------------------------------
def _load() -> types.SimpleNamespace:
    """Import the torch stack on first use and cache it.

    Kept out of module scope so ``python -m pyqcu.test.torch_matrix --list``
    and ``import pyqcu.test`` stay stdlib-only (see the module docstring).
    """
    if "torch" not in _MODULES:
        import torch
        from pyqcu import lattice, tools
        from pyqcu.torch import dslash, smear, solver

        _MODULES.update(torch=torch, lattice=lattice, tools=tools,
                        dslash=dslash, smear=smear, solver=solver)
    return types.SimpleNamespace(**_MODULES)


def probe_env() -> Dict[str, Any]:
    """Environment observations for a report header (never imports torch)."""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": None,
        "torch_threads": None,
        "cuda_available": None,
    }
    torch = _MODULES.get("torch")
    if torch is not None:
        env["torch"] = str(torch.__version__)
        env["torch_threads"] = int(torch.get_num_threads())
        try:
            env["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - report only
            env["cuda_available"] = None
    return env


# ---------------------------------------------------------------------------
# small numeric / option helpers
# ---------------------------------------------------------------------------
def _get(options: Mapping[str, Any], name: str, default: Any,
         aliases: Sequence[str] = ()) -> Any:
    """Fetch ``name`` (or one of ``aliases``) from ``options``."""
    for key in (name, *aliases):
        if key in options and options[key] is not None:
            return options[key]
    return default


def _choice(value: Any, name: str, allowed: Sequence[str]) -> Optional[str]:
    """Normalise an enumerated option; ``None`` stays ``None``."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return text


def _float_option(options: Mapping[str, Any], name: str,
                  default: Optional[float],
                  aliases: Sequence[str] = ()) -> Optional[float]:
    value = _get(options, name, default, aliases)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _int_option(options: Mapping[str, Any], name: str,
                default: Optional[int],
                aliases: Sequence[str] = ()) -> Optional[int]:
    value = _get(options, name, default, aliases)
    return None if value is None else int(value)


def _lattice_shape(spec: Any) -> Tuple[int, int, int, int]:
    """Alias of :func:`resolve_lattice` kept for the ``_Config`` readers."""
    return resolve_lattice(spec)


def _four_extents(spec: Any, name: str, minimum: int = 1) -> Tuple[int, ...]:
    """Parse an ``x y z t`` option given as a sequence or a string."""
    values = policy.parse_int_list(spec)
    if len(values) != 4:
        raise ValueError(f"{name} must have four extents (x y z t), got {values}")
    if any(value < minimum for value in values):
        raise ValueError(f"{name} extents must be >= {minimum}, got {values}")
    return tuple(int(value) for value in values)


def _lattice_grid(spec: Any,
                  lattice: Sequence[int]) -> Tuple[int, int, int, int]:
    """Validate the recorded MPI process grid against the local extents."""
    values = policy.validate_lattice_grid(spec)
    for extent, divisor in zip(lattice, values):
        if divisor > 1 and int(extent) % int(divisor):
            raise ValueError(
                f"lattice_grid {values} does not divide lattice "
                f"{tuple(int(value) for value in lattice)}")
    return values


def _kappa_from_mass(mass: float) -> float:
    """``kappa = 1 / (2 * mass + 8)``, rejecting the degenerate denominator."""
    denominator = 2.0 * float(mass) + 8.0
    if denominator == 0.0:
        raise ValueError(
            f"degenerate mass {mass!r}: kappa = 1/(2*mass+8) has a zero "
            "denominator")
    if not math.isfinite(denominator):
        raise ValueError(f"mass {mass!r} is not finite")
    return 1.0 / denominator


def _mass_from_kappa(kappa: float) -> float:
    """Inverse of :func:`_kappa_from_mass` (reporting only)."""
    return (1.0 / float(kappa) - 8.0) / 2.0


def _norm(torch: Any, value: Any) -> float:
    return float(torch.linalg.vector_norm(value))


def _relative(torch: Any, value: Any, reference: Any) -> float:
    """``||value - reference|| / ||reference||`` (exact, no clipping)."""
    denominator = _norm(torch, reference)
    numerator = _norm(torch, value) if denominator == 0.0 else _norm(
        torch, value - reference)
    if denominator == 0.0:
        # A zero reference has no meaningful scale: keep the failure visible.
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _device(spec: Any, torch: Any,
            device_id: Optional[int] = None) -> Tuple[Any, str, str]:
    """Resolve ``device`` to ``(torch.device, kind, requested)`` and probe it."""
    requested = "auto" if spec is None else str(spec).strip()
    lowered = requested.lower() or "auto"
    base, _, index = lowered.partition(":")
    if base not in _DEVICE_ALIASES:
        raise ValueError(
            f"device must be one of {', '.join(sorted(_DEVICE_ALIASES))} "
            f"or <kind>:<index>, got {spec!r}")
    device_type, kind = _DEVICE_ALIASES[base]
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        number = int(index) if index else (0 if device_id is None
                                           else int(device_id))
        if number < 0:
            raise ValueError(f"device index must be >= 0, got {spec!r}")
        try:
            device = torch.device(f"{device_type}:{number}")
        except Exception as error:  # noqa: BLE001 - unknown backend is a skip
            raise Skip(
                f"device_unavailable: torch {torch.__version__} does not know "
                f"device type {device_type!r} (requested {requested!r}): "
                f"{type(error).__name__}: {error}") from error
    _probe_device(torch, device, requested)
    return device, kind, requested


def _probe_device(torch: Any, device: Any, requested: str) -> None:
    """Allocate on ``device``; an unusable device is a skip, never a fallback."""
    kind = str(device.type)
    index = 0 if device.index is None else int(device.index)
    try:
        if kind == "cuda":
            if not torch.cuda.is_available():
                raise Skip(
                    f"device_unavailable: requested {requested!r} but "
                    "torch.cuda.is_available() is False")
            count = int(torch.cuda.device_count())
            if index >= count:
                raise Skip(
                    f"device_unavailable: requested {requested!r} but only "
                    f"{count} CUDA device(s) are visible")
        elif kind == "npu":
            npu = getattr(torch, "npu", None)
            if npu is None or not bool(npu.is_available()):
                raise Skip(
                    f"device_unavailable: requested {requested!r} but this "
                    "torch build has no usable torch.npu backend")
        elif kind != "cpu":
            raise Skip(
                f"device_unavailable: torch has no backend for {kind!r} "
                f"(requested {requested!r})")
        probe = torch.zeros((3, 3), dtype=torch.complex64, device=device)
        probe = probe + probe.conj().transpose(0, 1)
        if not bool(torch.isfinite(probe.real).all()):
            raise Skip(
                f"device_unavailable: {device} produced a non-finite probe")
    except Skip:
        raise
    except Exception as error:  # noqa: BLE001 - any allocation failure is a skip
        raise Skip(
            f"device_unavailable: {device} could not be used: "
            f"{type(error).__name__}: {error}") from error


def _device_name(torch: Any, device: Any) -> Optional[str]:
    """Human device name for the report (``None`` when unavailable)."""
    try:
        if device.type == "cuda":
            return str(torch.cuda.get_device_name(device.index or 0))
        if device.type == "npu":
            return str(torch.npu.device_name(device.index or 0))
    except Exception:  # pragma: no cover - report only
        return None
    return None


@contextlib.contextmanager
def _preserve_rng(torch: Any, device: Any):
    """Restore torch's global RNG around ``lattice.generate_gauge_field``.

    The core generator seeds the *global* generator (it has no generator
    argument), so generating a background field would otherwise perturb every
    later random draw in the process.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if str(getattr(device, "type", "cpu")) == "cuda":
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:  # pragma: no cover - CUDA may be present but idle
            cuda_states = None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:  # pragma: no cover
                pass


class _StdoutCapture:
    """Buffer ``print`` output so one case stays one line on stdout.

    The core layers (``LegacyMultigrid`` in particular) print unconditionally.
    Their text is kept as ``log_tail`` observations; if the runner raises, the
    buffered text is re-emitted on stderr so nothing is lost and the exception
    still propagates unchanged.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.buffer = io.StringIO()
        self._redirect: Optional[Any] = None

    def __enter__(self) -> "_StdoutCapture":
        if self.enabled:
            self._redirect = contextlib.redirect_stdout(self.buffer)
            self._redirect.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.enabled and self._redirect is not None:
            self._redirect.__exit__(exc_type, exc, traceback)
            if exc_type is not None:
                sys.stderr.write(self.buffer.getvalue())
        return False

    @property
    def text(self) -> str:
        return self.buffer.getvalue()

    def observations(self, tail: int = 400) -> Dict[str, Any]:
        text = self.text
        lines = [line for line in text.splitlines() if line.strip()]
        return {
            "log_lines": len(lines),
            "log_tail": text.strip()[-tail:] if lines else "",
        }


# ---------------------------------------------------------------------------
# run configuration
# ---------------------------------------------------------------------------
class _Config:
    """Resolved call configuration shared by every runner."""

    __slots__ = (
        "lattice", "lattice_tag", "preset", "heavy", "volume",
        "dtype_tag", "dtype", "seed", "device", "device_kind",
        "device_name", "device_requested", "device_id", "capture_stdout",
        "options", "mass", "kappa", "kappa_source", "csw",
        "gauge", "gauge_sigma", "gauge_file", "gauge_origin",
        "source", "source_file", "source_origin",
        "mg_grid", "mg_levels", "mg_nu_pre", "mg_nu_post", "max_iter",
        "restart", "tolerance", "dslash", "dslash_kinds", "solver",
        "solver_selected", "smear", "threads", "threads_previous",
        "threads_applied", "threads_error", "lattice_grid", "process_grid",
        "process_grid_error", "lattice_grid_applied", "repeat", "warmup",
        "memory", "option_sources",
    )

    def __init__(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
        options: Dict[str, Any] = dict(kwargs)
        explicit = set(options)
        torch = _load().torch
        lattice_spec: Any = None
        dtype_spec: Any = None
        seed: Any = None
        device_spec: Any = None
        case_items: List[Any] = []
        for item in args:
            if item is None:
                continue
            if isinstance(item, Mapping):
                merged = dict(item)
                explicit.update(merged)
                merged.update(options)
                options = merged
                continue
            if (isinstance(item, (tuple, list)) and len(item) == 4
                    and all(isinstance(value, int)
                            and not isinstance(value, bool) for value in item)):
                lattice_spec = item
                continue
            if isinstance(item, int) and not isinstance(item, bool):
                seed = item
                continue
            if isinstance(item, str):
                try:
                    resolve_dtype(item)
                    dtype_spec = item
                    continue
                except ValueError:
                    pass
                lattice_spec = resolve_lattice(item)
                continue
            # Case-like object (``MatrixCase``/``common.Case``/Namespace).
            case_items.append(item)
            lattice_spec = getattr(item, "lattice", lattice_spec)
            dtype_spec = getattr(item, "dtype", dtype_spec)
            seed = getattr(item, "seed", seed)
            device_spec = getattr(item, "device", device_spec)

        # Options a case-like object carries as attributes.
        for item in case_items:
            for name in _OPTION_NAMES:
                if name not in options:
                    value = getattr(item, name, None)
                    if value is not None:
                        options[name] = value
                        explicit.add(name)
        self.option_sources = {
            name: ("argument" if name in explicit else "default")
            for name in _OPTION_NAMES
        }
        self.options = options

        lattice_option = _get(options, "lattice", lattice_spec,
                              _LATTICE_KEYS[1:])
        preset_option = _get(options, "preset", None, ("lattice_preset",))
        if lattice_option is None and preset_option is not None:
            lattice_option = preset_option
        self.lattice = _lattice_shape(lattice_option)
        self.lattice_tag = lattice_tag(self.lattice)
        self.volume = math.prod(self.lattice)
        self.preset = (lattice_preset(self.lattice)
                       or (str(preset_option).strip() if preset_option else "custom"))
        self.heavy = self.preset in HEAVY_PRESETS
        self.dtype_tag = resolve_dtype(_get(options, "dtype", dtype_spec,
                                            _DTYPE_KEYS[1:]))
        self.dtype = torch_dtype(self.dtype_tag)
        seed = _get(options, "seed", seed, _SEED_KEYS[1:])
        self.seed = DEFAULT_SEED if seed is None else int(seed)
        device_spec = _get(options, "device", device_spec, _DEVICE_KEYS[1:])
        self.device_id = _int_option(options, "device_id", None,
                                     ("device_index",))
        if self.device_id is not None and self.device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {self.device_id!r}")
        self.device, self.device_kind, self.device_requested = _device(
            device_spec, torch, self.device_id)
        self.device_name = _device_name(torch, self.device)
        self.capture_stdout = bool(_get(options, "capture_stdout", True,
                                        _CAPTURE_KEYS[1:]))

        # mass <-> kappa: an explicit kappa wins, otherwise mass derives it.
        kappa_value = _float_option(options, "kappa", None)
        mass_value = _float_option(options, "mass", None)
        if kappa_value is not None:
            if kappa_value <= 0.0:
                raise ValueError(f"kappa must be > 0, got {kappa_value!r}")
            self.kappa = kappa_value
            self.mass = _mass_from_kappa(kappa_value)
            self.kappa_source = "explicit"
        elif mass_value is not None:
            self.kappa = _kappa_from_mass(mass_value)
            self.mass = mass_value
            self.kappa_source = "mass"
        else:
            self.kappa = DEFAULT_KAPPA
            self.mass = _mass_from_kappa(DEFAULT_KAPPA)
            self.kappa_source = "default"

        self.csw = _float_option(options, "csw", DEFAULT_CSW)
        if self.csw is None or self.csw < 0.0:
            raise ValueError(f"csw must be >= 0, got {self.csw!r}")
        self.gauge = _choice(_get(options, "gauge", policy.DEFAULT_GAUGE),
                             "gauge", policy.GAUGE_CHOICES)
        sigma = _float_option(options, "gauge_sigma", DEFAULT_GAUGE_SIGMA)
        if sigma is None or sigma < 0.0:
            raise ValueError(f"gauge_sigma must be >= 0, got {sigma!r}")
        self.gauge_sigma = sigma
        self.gauge_file = _path_option(options, "gauge_file", None,
                                       ("gauge_path",))
        self.gauge_origin = "file" if self.gauge_file else "generated"
        self.source = _choice(_get(options, "source", None), "source",
                              policy.SOURCE_CHOICES)
        self.source_file = _path_option(options, "source_file", None,
                                        ("source_path",))
        self.source_origin = "file" if self.source_file else "generated"

        self.dslash = _choice(_get(options, "dslash", None), "dslash",
                              policy.DSLASH_CHOICES)
        self.dslash_kinds = _DSLASH_SELECTION[self.dslash]
        self.solver = _choice(_get(options, "solver", None), "solver",
                              SOLVER_KINDS)
        self.solver_selected = _SOLVER_SELECTION[self.solver or "all"]
        self.smear = _choice(_get(options, "smear", "stout"), "smear",
                             SMEAR_KINDS) or "stout"

        self.mg_grid = policy.validate_mg_grid(
            _get(options, "mg_grid", DEFAULT_MG_GRID))
        self.mg_levels = _int_option(options, "mg_levels", None,
                                     ("max_level", "levels"))
        if self.mg_levels is not None and self.mg_levels < 1:
            raise ValueError(f"mg_levels must be >= 1, got {self.mg_levels!r}")
        self.mg_nu_pre = _int_option(options, "mg_nu_pre", None,
                                     ("nu_pre", "npre"))
        self.mg_nu_post = _int_option(options, "mg_nu_post", None,
                                      ("nu_post", "npost"))
        for name in ("mg_nu_pre", "mg_nu_post"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        self.max_iter = _int_option(options, "max_iter", None, ("maxiter",))
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter!r}")
        self.restart = _int_option(options, "restart", None, ("num_restart",))
        if self.restart is not None and self.restart < 1:
            raise ValueError(f"restart must be >= 1, got {self.restart!r}")
        self.tolerance = _float_option(options, "tolerance", DEFAULT_TOLERANCE,
                                       ("tol",))
        if self.tolerance is None or self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance!r}")

        # ``lattice_grid`` is recorded, never applied: the torch runners do not
        # domain-decompose, so claiming a grid would be a fake speedup.
        self.lattice_grid = _lattice_grid(
            _get(options, "lattice_grid", policy.DEFAULT_LATTICE_GRID),
            self.lattice)
        self.process_grid = None
        self.process_grid_error = None
        try:
            from pyqcu import tools

            self.process_grid = tuple(int(value) for value in
                                      tools.give_grid_size())
        except Exception as error:  # noqa: BLE001 - report, never gate
            self.process_grid_error = f"{type(error).__name__}: {error}"
        self.lattice_grid_applied = bool(
            self.process_grid is not None
            and tuple(self.process_grid) == tuple(self.lattice_grid))

        self.threads = _int_option(options, "threads", None)
        if self.threads is not None and self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads!r}")
        self.threads_previous = int(torch.get_num_threads())
        self.threads_applied = False
        self.threads_error = None
        if self.threads is not None:
            if self.threads_previous != self.threads:
                try:
                    torch.set_num_threads(int(self.threads))
                    self.threads_applied = True
                except Exception as error:  # noqa: BLE001 - report, never gate
                    self.threads_error = f"{type(error).__name__}: {error}"
            else:
                self.threads_applied = True

        repeat = _get(options, "repeat", None, ("samples",))
        self.repeat = 1 if repeat is None else int(repeat)
        warmup = _get(options, "warmup", None, ("warmups",))
        self.warmup = 1 if warmup is None else int(warmup)
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat!r}")
        if self.warmup < 1:
            raise ValueError(
                f"warmup must be >= 1, got {self.warmup!r} "
                "(the first call is not a steady state)")
        self.memory = {
            "estimate_bytes": None,
            "available_bytes": None,
            "budget_bytes": None,
            "budget_fraction": None,
        }

    # -- option readers -----------------------------------------------------
    def option(self, name: str, default: Any = None,
               aliases: Sequence[str] = ()) -> Any:
        """Read one raw option (argument value wins over the runner default)."""
        return _get(self.options, name, default, aliases)

    def float_option(self, name: str, default: Optional[float] = None,
                     aliases: Sequence[str] = ()) -> Optional[float]:
        return _float_option(self.options, name, default, aliases)

    def int_option(self, name: str, default: Optional[int] = None,
                   aliases: Sequence[str] = ()) -> Optional[int]:
        return _int_option(self.options, name, default, aliases)

    def choice_option(self, name: str, default: Optional[str],
                      allowed: Sequence[str]) -> Optional[str]:
        return _choice(_get(self.options, name, default), name, allowed)

    def kappa_tensor(self) -> Any:
        """The hopping parameter as the 1-element tensor the core expects."""
        return _MODULES["torch"].tensor([float(self.kappa)])

    def solver_tolerance(self, default: Optional[float] = None) -> float:
        """Tolerance handed to the solver (``solver_tolerance``/``solver_tol``)."""
        if default is None:
            default = (self.tolerance if self.dtype_tag == "c64"
                       else self.tolerance * 1e-2)
        value = self.float_option("solver_tolerance", float(default),
                                  ("solver_tol",))
        if value is None or value <= 0.0:
            raise ValueError(f"solver_tolerance must be > 0, got {value!r}")
        return float(value)

    # -- memory pre-flight --------------------------------------------------
    def gauge_memory_buffers(self) -> int:
        """Gauge-field-sized temporaries the configured field construction needs."""
        if self.gauge_origin == "file":
            return _GAUGE_FILE_BUFFERS
        if self.gauge in ("ones", "unit"):
            return _GAUGE_CONSTANT_BUFFERS
        return _GAUGE_GENERATE_BUFFERS

    def check_memory(self, *, clover: bool = False, spinor_buffers: int = 8,
                     gauge_buffers: Optional[int] = None,
                     stencil_buffers: int = 0, mg_levels: int = 1,
                     note: str = "run") -> Dict[str, Any]:
        """Estimate this run's working set and refuse what cannot fit.

        ``Skip(device_unavailable)`` carries the estimate, the budget and the
        bytes the device reports, so a heavy preset that is too large for this
        machine is an explained skip instead of an allocator crash.
        """
        if gauge_buffers is None:
            gauge_buffers = self.gauge_memory_buffers()
        estimate = estimate_memory_bytes(
            self.lattice, self.dtype_tag, clover=clover,
            spinor_buffers=spinor_buffers, gauge_buffers=gauge_buffers,
            stencil_buffers=stencil_buffers, mg_levels=mg_levels,
            mg_grid=self.mg_grid)
        available = available_memory_bytes(self.device)
        fraction = self.float_option("memory_budget", DEFAULT_MEMORY_BUDGET,
                                     ("memory_fraction",))
        if fraction is None or not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"memory_budget must be in (0, 1], got {fraction!r}")
        budget = None if available is None else int(available * fraction)
        self.memory = {
            "estimate_bytes": int(estimate),
            "available_bytes": None if available is None else int(available),
            "budget_bytes": budget,
            "budget_fraction": float(fraction),
            "note": str(note),
        }
        if budget is not None and estimate > budget:
            raise Skip(
                f"device_unavailable: {note} on {self.lattice_tag} "
                f"{self.dtype_tag} plans ~{estimate / 2 ** 30:.2f} GiB, above "
                f"the {fraction:.0%} budget ({budget / 2 ** 30:.2f} GiB of "
                f"{available / 2 ** 30:.2f} GiB) reported for {self.device}")
        return self.memory


def _path_option(options: Mapping[str, Any], name: str,
                 default: Optional[str],
                 aliases: Sequence[str] = ()) -> Optional[str]:
    """Read an optional filesystem path option (stored as ``str``)."""
    value = _get(options, name, default, aliases)
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# field builders (all option-aware)
# ---------------------------------------------------------------------------
def _gauge_field(config: _Config, torch: Any, lattice: Any, *,
                 kind: Optional[str] = None, seed: Optional[int] = None,
                 sigma: Optional[float] = None) -> Any:
    """Build the background gauge field ``[3, 3, 4, x, y, z, t]``.

    ``random`` is the seeded :func:`pyqcu.lattice.generate_gauge_field`
    exponential map with ``gauge_sigma``; ``unit`` is the exact identity link
    on every site; ``ones`` is a deliberately non-unitary all-ones tensor (a
    code-path probe, never an SU(3) field)::

        _gauge_field(config, torch, lattice)                  # configured field
        _gauge_field(config, torch, lattice, kind="unit")     # probe field
        _gauge_field(config, torch, lattice, sigma=0.0)       # identity probe

    ``gauge_file`` is honoured only for the configured field: passing
    ``kind``/``seed``/``sigma`` asks for a generated probe, so a file input
    cannot silently replace the field a probe is supposed to generate.
    """
    gauge_kind = (config.gauge if kind is None
                  else _choice(kind, "gauge", policy.GAUGE_CHOICES))
    if kind is None and seed is None and sigma is None and config.gauge_file:
        field, _origin = _load_field_file(
            config, torch, role="gauge", path=config.gauge_file,
            shape=(3, 3, 4, *config.lattice))
        return field
    gauge_seed = config.seed if seed is None else int(seed)
    gauge_sigma = config.gauge_sigma if sigma is None else float(sigma)
    if gauge_kind == "random":
        field = torch.zeros((3, 3, 4, *config.lattice), dtype=config.dtype,
                            device=config.device)
        with _preserve_rng(torch, config.device):
            lattice.generate_gauge_field(field, seed=gauge_seed,
                                         sigma=gauge_sigma, verbose=False)
        return field
    if gauge_kind == "unit":
        eye = torch.eye(3, dtype=config.dtype, device=config.device)
        return eye.reshape(3, 3, 1, 1, 1, 1, 1).expand(
            3, 3, 4, *config.lattice).clone()
    if gauge_kind == "ones":
        return torch.ones((3, 3, 4, *config.lattice), dtype=config.dtype,
                          device=config.device)
    raise ValueError(
        f"unsupported gauge {gauge_kind!r}; choose from "
        f"{', '.join(policy.GAUGE_CHOICES)}")


def _expected_gauge(config: _Config, torch: Any,
                    kind: str) -> Optional[Any]:
    """The exact field ``kind`` is supposed to produce (``None`` for random)."""
    if kind == "unit":
        eye = torch.eye(3, dtype=config.dtype, device=config.device)
        return eye.reshape(3, 3, 1, 1, 1, 1, 1).expand(
            3, 3, 4, *config.lattice).clone()
    if kind == "ones":
        return torch.ones((3, 3, 4, *config.lattice), dtype=config.dtype,
                          device=config.device)
    return None


def _dense_source(config: "_Config", torch: Any, seed: int) -> Any:
    """Seeded dense Gaussian field, independent of the ``source`` option.

    ``torch.Generator`` is used explicitly so a source never depends on, nor
    perturbs, the process-global RNG.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    real = torch.randn((4, 3, *config.lattice), generator=generator,
                       dtype=torch.float32)
    imag = torch.randn((4, 3, *config.lattice), generator=generator,
                       dtype=torch.float32)
    return torch.complex(real, imag).to(dtype=config.dtype,
                                        device=config.device)


def _fermion_field(config: "_Config", torch: Any, seed: int,
                   default: str = "random") -> Any:
    """Build the source ``[4, 3, x, y, z, t]`` honouring ``source``.

    ``random`` uses its own ``torch.Generator``; ``point`` is the
    spin-0/colour-0 delta at the origin (``||b|| == 1``); ``ones`` is the
    constant spinor/colour field.  ``source_file`` wins over ``source`` for the
    configured field; passing ``default`` does not disable it, ``source`` does
    (a case that names a source kind asks for that kind).
    """
    if config.source is None and config.source_file:
        field, _origin = _load_field_file(
            config, torch, role="source", path=config.source_file,
            shape=(4, 3, *config.lattice))
        return field
    kind = config.source or default
    lattice = _load().lattice
    if kind == "random":
        return _dense_source(config, torch, seed)
    if kind == "point":
        return lattice.point_source(
            latt_size=list(config.lattice), t_srce=[0, 0, 0, 0],
            spin=0, color=0, dtype=config.dtype, device=config.device)
    if kind == "ones":
        return torch.ones((4, 3, *config.lattice), dtype=config.dtype,
                          device=config.device)
    raise ValueError(
        f"unsupported source {kind!r}; choose from "
        f"{', '.join(policy.SOURCE_CHOICES)}")


def _clover_term(config: "_Config", dslash: Any, gauge: Any, kappa: Any,
                 csw: Optional[float] = None) -> Any:
    """Clover term ``T`` at ``csw`` (the option value unless one is passed)."""
    coefficient = config.csw if csw is None else float(csw)
    return dslash.make_clover(U=gauge, kappa=kappa, csw=coefficient,
                              verbose=False)


def _wilson_plus_clover_reference(dslash: Any, gauge: Any, clover: Any,
                                  kappa: Any, vector: Any) -> Any:
    """Independent ``D x = (1 + T + hop) x`` from the public helpers."""
    reference = dslash.give_wilson(src=vector, U=gauge, kappa=kappa,
                                   with_I=True, verbose=False)
    if clover is not None:
        reference = reference + dslash.give_clover(
            src=vector, clover_term=clover, verbose=False)
    return reference


def _su3_deviations(torch: Any, gauge: Any,
                    chunk: int = 1 << 16) -> Tuple[float, float]:
    """Max unitarity and determinant deviation of a gauge field.

    Evaluated one direction at a time in site chunks: the full
    ``permute(...).reshape(-1, 3, 3)`` copy of an ``L32`` ``c128`` gauge field
    is over a gigabyte, and the pre-flight budget is not spent on a diagnostic.
    """
    unitarity = 0.0
    determinant = 0.0
    eye3 = torch.eye(3, dtype=gauge.dtype, device=gauge.device)
    for direction in range(gauge.shape[2]):
        link = gauge[:, :, direction]
        matrices = link.permute(*range(2, link.ndim), 0, 1).reshape(-1, 3, 3)
        for start in range(0, matrices.shape[0], chunk):
            block = matrices[start:start + chunk]
            product = torch.matmul(block.conj().transpose(-1, -2), block)
            unitarity = max(unitarity, float(
                (product - eye3.expand_as(product)).abs().max()))
            determinant = max(determinant, float(
                (torch.linalg.det(block) - 1.0).abs().max()))
    return unitarity, determinant


# ---------------------------------------------------------------------------
# file inputs (``pyqcu.test.io`` when available, HDF5 otherwise)
# ---------------------------------------------------------------------------
#: Reader entry points probed on :mod:`pyqcu.test.io`, most specific first.
_FIELD_READERS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "gauge": ("read_gauge", "load_gauge", "ensure_gauge", "read_field",
              "load_field", "read_array", "read_dataset"),
    "source": ("read_source", "load_source", "ensure_source", "read_field",
               "load_field", "read_array", "read_dataset"),
})
#: Dataset layout each role must carry (``data`` in a PyQCU artifact).
_FIELD_LAYOUT: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "gauge": ("color", "color", "direction", "x", "y", "z", "t"),
    "source": ("spin", "color", "x", "y", "z", "t"),
})


def _input_module() -> Optional[Any]:
    """Return :mod:`pyqcu.test.io` when this checkout ships it, else ``None``.

    The module is optional: a checkout without it must still be able to read a
    production gauge/source file, so the caller falls back to a direct HDF5
    read instead of failing.
    """
    try:
        from . import io as module  # noqa: PLC0415 - optional sibling module
    except Exception:  # noqa: BLE001 - absent or unimportable is "unavailable"
        return None
    return module


def _call_with_supported_kwargs(function: Callable[..., Any],
                                kwargs: Mapping[str, Any]) -> Any:
    """Call ``function`` with the keyword arguments it actually declares."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD
           for parameter in signature.parameters.values()):
        return function(**kwargs)
    accepted = {name: value for name, value in kwargs.items()
                if name in signature.parameters}
    return function(**accepted)


def _read_hdf5_field(config: "_Config", path: Path, shape: Tuple[int, ...],
                     role: str) -> Any:
    """Read a PyQCU HDF5 field (``data`` dataset) directly with ``h5py``."""
    try:
        import numpy as np
        import h5py
        import torch
    except ImportError as error:
        raise Skip(
            f"h5py_mpi: cannot read the {role} file {path}: "
            f"{type(error).__name__}: {error}") from error
    numpy_dtype = np.complex128 if config.dtype_tag == "c128" else np.complex64
    with h5py.File(str(path), "r") as handle:
        key = "data" if "data" in handle else None
        if key is None:
            datasets = [name for name, item in handle.items()
                        if isinstance(item, h5py.Dataset)]
            if len(datasets) != 1:
                raise ValueError(
                    f"{role} file {path} has no 'data' dataset and "
                    f"{len(datasets)} top-level dataset(s); a PyQCU artifact "
                    "carries exactly one 'data' dataset in layout "
                    f"[{', '.join(_FIELD_LAYOUT[role])}]")
            key = datasets[0]
        dataset = handle[key]
        actual = tuple(int(value) for value in dataset.shape)
        if actual != tuple(shape):
            raise ValueError(
                f"{role} dataset {key!r} in {path} has shape {actual}, "
                f"expected {tuple(shape)} in layout "
                f"[{', '.join(_FIELD_LAYOUT[role])}]")
        array = np.empty(actual, dtype=numpy_dtype)
        dataset.read_direct(array)
    # ``from_numpy`` shares the buffer: no second copy of a production field.
    return torch.from_numpy(array)


def _as_field(value: Any, shape: Tuple[int, ...], torch: Any, dtype: Any,
              device: Any, *, role: str, path: Path) -> Any:
    """Validate a loaded field's shape and move it onto the run dtype/device."""
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception as error:  # noqa: BLE001 - a bad object is a usage bug
            raise ValueError(
                f"{role} reader returned {type(value).__name__} for {path}, "
                f"which is not a tensor: {error}") from error
    actual = tuple(int(extent) for extent in value.shape)
    if actual != tuple(shape):
        raise ValueError(
            f"{role} field from {path} has shape {actual}, expected "
            f"{tuple(shape)} in layout [{', '.join(_FIELD_LAYOUT[role])}]")
    return value.to(dtype=dtype, device=device)


def _load_field_file(config: "_Config", torch: Any, *, role: str, path: Any,
                     shape: Tuple[int, ...]) -> Tuple[Any, str]:
    """Load one gauge/source file, preferring :mod:`pyqcu.test.io`.

    Returns ``(field, origin)`` where ``origin`` names the reader
    (``pyqcu.test.io:<name>`` or ``hdf5``).  A path that does not exist is a
    ``ValueError`` (a usage error): a silently generated replacement field
    would turn "read the production configuration" into "run a random one".
    """
    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise ValueError(f"{role} file {target} does not exist")
    module = _input_module()
    origin = "hdf5"
    field = None
    if module is not None:
        for name in _FIELD_READERS[role]:
            reader = getattr(module, name, None)
            if not callable(reader):
                continue
            result = _call_with_supported_kwargs(reader, {
                "path": str(target),
                "lattice": tuple(config.lattice),
                "lattice_size": list(config.lattice),
                "dtype": config.dtype_tag,
                "device": str(config.device),
                "shape": tuple(shape),
                "role": role,
                "layout": _FIELD_LAYOUT[role],
            })
            origin = f"pyqcu.test.io:{name}"
            if isinstance(result, (str, Path)):
                # The helper ensured/produced a file: read that file.
                target = Path(str(result)).expanduser()
                if not target.is_file():
                    raise ValueError(
                        f"pyqcu.test.io.{name} returned {target}, which is not "
                        f"a readable {role} file")
                continue
            field = result
            break
    if field is None:
        field = _read_hdf5_field(config, target, shape, role)
    return _as_field(field, shape, torch, config.dtype, config.device,
                     role=role, path=target), origin


# ---------------------------------------------------------------------------
# timing and observation helpers
# ---------------------------------------------------------------------------
def _measure(config: "_Config",
             call: Callable[[], Any]) -> Tuple[Tuple[float, ...], Any]:
    """Time ``call`` with the case's ``warmup``/``repeat`` policy.

    :func:`pyqcu.test.common.time_samples` owns the policy (discard
    ``warmup >= 1`` calls, then collect ``repeat`` samples) and the closure
    records the last return value, so the reported observation comes from the
    same deterministic execution as the timing sample.
    """
    captured: Dict[str, Any] = {}

    def _observed() -> Any:
        value = call()
        captured["value"] = value
        return value

    samples = time_samples(_observed, repeat=config.repeat,
                           warmup=config.warmup)
    return tuple(samples), captured.get("value")


def _timing(samples: Sequence[float]) -> Dict[str, Any]:
    """Timing observations: median (``seconds``), min, max and raw samples."""
    values = [float(value) for value in samples]
    return {
        "seconds": float(statistics.median(values)) if values else 0.0,
        "seconds_min": float(min(values)) if values else 0.0,
        "seconds_max": float(max(values)) if values else 0.0,
        "samples": len(values),
        "timing_samples": [round(value, 6) for value in values],
    }


def _base(config: _Config, torch: Any, case: str) -> Dict[str, Any]:
    """The observation block every runner reports."""
    return {
        "case": case,
        "backend": "torch",
        "module": "torch_matrix",
        "preset": config.preset,
        "heavy": bool(config.heavy),
        "lattice": list(config.lattice),
        "lattice_tag": config.lattice_tag,
        "volume": int(config.volume),
        "lattice_grid": list(config.lattice_grid),
        "process_grid": (list(config.process_grid)
                         if config.process_grid is not None else None),
        "process_grid_error": config.process_grid_error,
        "lattice_grid_applied": config.lattice_grid_applied,
        "dtype": config.dtype_tag,
        "seed": int(config.seed),
        "device": str(config.device),
        "device_kind": config.device_kind,
        "device_type": str(config.device.type),
        "device_requested": config.device_requested,
        "device_id": config.device_id,
        "device_name": config.device_name,
        "torch_version": str(torch.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "threads_requested": config.threads,
        "threads_previous": config.threads_previous,
        "threads_applied": config.threads_applied,
        "threads_error": config.threads_error,
        "cuda_available": bool(torch.cuda.is_available()),
        "mass": config.mass,
        "kappa": config.kappa,
        "kappa_source": config.kappa_source,
        "csw": config.csw,
        "gauge": config.gauge,
        "gauge_sigma": config.gauge_sigma,
        "gauge_origin": config.gauge_origin,
        "gauge_file": config.gauge_file,
        "source": config.source,
        "source_origin": config.source_origin,
        "source_file": config.source_file,
        "dslash": config.dslash,
        "dslash_kinds": list(config.dslash_kinds),
        "solver": config.solver_selected,
        "solver_requested": config.solver,
        "smear": config.smear,
        "mg_grid": list(config.mg_grid),
        "mg_levels": config.mg_levels,
        "mg_nu_pre": config.mg_nu_pre,
        "mg_nu_post": config.mg_nu_post,
        "max_iter": config.max_iter,
        "restart": config.restart,
        "tolerance": config.tolerance,
        "repeat": config.repeat,
        "warmup": config.warmup,
        "memory_estimate_bytes": config.memory.get("estimate_bytes"),
        "memory_available_bytes": config.memory.get("available_bytes"),
        "memory_budget_bytes": config.memory.get("budget_bytes"),
        "memory_budget_fraction": config.memory.get("budget_fraction"),
        "option_sources": dict(config.option_sources),
    }


def _finish(observations: Dict[str, Any], started: float) -> Dict[str, Any]:
    observations["wall_seconds"] = time.perf_counter() - started
    return observations


# ---------------------------------------------------------------------------
# runner: gauge fields
# ---------------------------------------------------------------------------
def run_gauge(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/gauge``: build the requested background field.

    ``unit`` and ``ones`` are compared against the exact tensor they are
    defined to be (an independent construction, ``drift_relative``), ``random``
    against the invariants of a generated SU(3) field: same seed regenerates
    bit-identically (``repeat_relative``), a different seed gives a different
    field (``distinct_relative``, gated ``> 0`` so a case proves the seed
    option acts), ``sigma = 0`` is the exact identity (``sigma_zero_dev``) and
    doubling ``sigma`` changes the field (``sigma_relative``).

    ``relative_error`` is the largest deviation that must vanish, so one gate
    covers the whole construction.  ``su3`` is the core's
    :func:`pyqcu.lattice.check_su3`; it is reported ``False`` for ``ones``
    (which is not an SU(3) field by design) and computed from the file when
    ``gauge_file`` is used.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, lattice = modules.torch, modules.lattice
    su3_tol = config.float_option("su3_tol", 1e-3, ("su3_tolerance",))
    from_file = config.gauge_origin == "file"
    config.check_memory(
        spinor_buffers=6, note="gauge",
        gauge_buffers=config.gauge_memory_buffers()
        + (0 if from_file else 4))
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            samples, gauge = _measure(
                config, lambda: _gauge_field(config, torch, lattice))
            finite = bool(torch.isfinite(gauge.real).all()
                          and torch.isfinite(gauge.imag).all())
            if from_file or config.gauge != "ones":
                su3 = bool(lattice.check_su3(gauge, tol=su3_tol, verbose=False))
            else:
                su3 = False
            unitarity_dev, det_dev = _su3_deviations(torch, gauge)
            expected = (None if from_file
                        else _expected_gauge(config, torch, config.gauge))
            drift = (_relative(torch, gauge, expected)
                     if expected is not None else None)
            repeat_relative = distinct_relative = sigma_relative = None
            sigma_zero_dev = None
            if from_file:
                # The configured field came from a file: the generated-field
                # probes below describe a different field and are reported as
                # ``None`` instead of as a meaningless deviation.
                relative = max(unitarity_dev, det_dev)
            else:
                replay = _gauge_field(config, torch, lattice)
                repeat_relative = _relative(torch, replay, gauge)
                distinct = _gauge_field(config, torch, lattice,
                                       seed=config.seed + 1)
                distinct_relative = _relative(torch, distinct, gauge)
                doubled = _gauge_field(config, torch, lattice,
                                       sigma=2.0 * config.gauge_sigma)
                sigma_relative = _relative(torch, doubled, gauge)
                identity = _gauge_field(config, torch, lattice, kind="random",
                                        sigma=0.0)
                eye = _expected_gauge(config, torch, "unit")
                sigma_zero_dev = float((identity - eye).abs().max())
                if config.gauge == "ones":
                    # All-ones links are deliberately not SU(3): the only
                    # invariant is that the field is exactly the requested
                    # tensor, so the unitarity/determinant deviations stay
                    # observations instead of entering the gate.
                    relative = max(repeat_relative,
                                   drift if drift is not None else 0.0)
                else:
                    relative = max(repeat_relative, sigma_zero_dev,
                                   unitarity_dev, det_dev)
                    if drift is not None:
                        relative = max(relative, drift)

    observations = _base(config, torch, "torch/matrix/gauge")
    observations.update(_timing(samples))
    observations.update(
        su3=su3,
        su3_tolerance=su3_tol,
        unitarity_dev=unitarity_dev,
        det_dev=det_dev,
        finite=finite,
        drift_relative=drift,
        repeat_relative=repeat_relative,
        distinct_relative=distinct_relative,
        sigma_relative=sigma_relative,
        sigma_zero_dev=sigma_zero_dev,
        relative=relative,
        rel=relative,
        relative_error=relative,
        gauge_norm=_norm(torch, gauge),
        gauge_shape=list(gauge.shape),
        gauge_bytes=int(gauge.numel() * gauge.element_size()),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


# ---------------------------------------------------------------------------
# runner: Wilson / clover dslash
# ---------------------------------------------------------------------------
def run_dslash(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/dslash``: Wilson and/or clover ``matvec`` identities.

    Three independent paths must agree on the same source:

    * ``reference_relative`` -- the operator against
      ``give_wilson(with_I=True)`` + ``give_clover``, i.e. the public helpers
      rather than the operator under test;
    * ``parity_relative`` -- the compact even/odd reconstruction
      (``matvec_all``) against the full ``matvec``;
    * ``relative_e``/``relative_o`` -- the ``matvec_eeo``/``matvec_oeo``
      blocks against the parity split of the full result (reported, not gated:
      they are the diagnostic of *where* a parity bug would sit).

    ``relative_error`` is the maximum of the gated numbers.  ``dslash`` selects
    the operator (``wilson`` default, ``clover`` adds ``csw``, ``all``/``both``
    exercise both); the clover term is built from ``csw`` and ``kappa`` exactly
    once and shared by the operator and the reference.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, lattice, tools, dslash = (modules.torch, modules.lattice,
                                     modules.tools, modules.dslash)
    clover_needed = "clover" in config.dslash_kinds
    config.check_memory(clover=clover_needed, spinor_buffers=12,
                        stencil_buffers=_STENCIL_BUFFERS, note="dslash")
    kappa = config.kappa_tensor()
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, lattice)
            source = _fermion_field(config, torch, config.seed + 1,
                                    default="random")
            clover = (_clover_term(config, dslash, gauge, kappa)
                      if clover_needed else None)
            operator = dslash.operator(
                U=gauge, clover_term=clover, kappa=kappa,
                support_parity=True, verbose=False)
            flat = source.reshape([12, *config.lattice])
            full = operator.matvec(source)
            samples, applied = _measure(config, lambda: operator.matvec(source))
            reference = _wilson_plus_clover_reference(
                dslash, gauge, clover, kappa, source)
            reference_relative = _relative(torch, full, reference)
            compact = operator.matvec_all(flat)
            parity_relative = _relative(torch, compact,
                                        full.reshape(flat.shape))
            src_e, src_o = tools.oooxyzt2poooxyzt(input_array=flat,
                                                  verbose=False)
            dest_e = operator.matvec_eeo(src_e=src_e, src_o=src_o)
            dest_o = operator.matvec_oeo(src_e=src_e, src_o=src_o)
            full_e, full_o = tools.oooxyzt2poooxyzt(
                input_array=full.reshape(flat.shape), verbose=False)
            relative_e = _relative(torch, dest_e, full_e)
            relative_o = _relative(torch, dest_o, full_o)
            relative = max(reference_relative, parity_relative)
            finite = bool(torch.isfinite(full.real).all()
                          and torch.isfinite(full.imag).all())

    observations = _base(config, torch, "torch/matrix/dslash")
    observations.update(_timing(samples))
    observations.update(
        dslash="+".join(config.dslash_kinds),
        operators=list(config.dslash_kinds),
        with_clover=bool(clover_needed),
        relative=relative,
        rel=relative,
        relative_error=relative,
        reference_relative=reference_relative,
        parity_relative=parity_relative,
        relative_e=relative_e,
        relative_o=relative_o,
        max_abs_error=float((full - reference).abs().max()),
        full_norm=_norm(torch, full),
        reference_norm=_norm(torch, reference),
        parity_norm=_norm(torch, compact),
        operator_norm=_norm(torch, applied),
        finite=finite,
        clover_norm=(None if clover is None else _norm(torch, clover)),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


# ---------------------------------------------------------------------------
# runner: clover term and inverse
# ---------------------------------------------------------------------------
def run_clover(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/clover``: clover term, ``(I + T)`` inverse round trip.

    ``inverse_relative`` applies ``(I + T)``, then ``inverse(I + T)``, and
    compares with the input -- a self-contained algebraic identity that does
    not consult the operator.  ``operator_relative`` additionally applies the
    full clover operator and compares it with the independent
    ``give_wilson(with_I=True) + give_clover`` reference.  ``csw_zero_dev``
    confirms ``csw = 0`` switches the term off exactly, and
    ``relative_error`` is the maximum of the three.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, dslash = modules.torch, modules.dslash
    config.check_memory(
        clover=True, spinor_buffers=8, stencil_buffers=_STENCIL_BUFFERS,
        gauge_buffers=config.gauge_memory_buffers()
        + _CLOVER_SCRATCH_BUFFERS, note="clover")
    kappa = config.kappa_tensor()
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, modules.lattice)
            source = _fermion_field(config, torch, config.seed + 3,
                                    default="random")
            clover = _clover_term(config, dslash, gauge, kappa)
            clover_plus_identity = dslash.add_I(clover_term=clover,
                                                verbose=False)
            clover_inverse = dslash.inverse(clover_term=clover_plus_identity,
                                            verbose=False)
            samples, applied = _measure(
                config,
                lambda: dslash.give_clover(src=source, clover_term=clover,
                                           verbose=False))
            round_trip = dslash.give_clover(
                src=dslash.give_clover(src=source,
                                       clover_term=clover_inverse,
                                       verbose=False),
                clover_term=clover_plus_identity, verbose=False)
            inverse_relative = _relative(torch, round_trip, source)
            zero_csw = _clover_term(config, dslash, gauge, kappa, csw=0.0)
            csw_zero_dev = (float(zero_csw.abs().max())
                            if zero_csw.numel() else 0.0)
            operator = dslash.operator(U=gauge, clover_term=clover,
                                       kappa=kappa, verbose=False)
            operator_relative = _relative(
                torch, operator.matvec(source),
                _wilson_plus_clover_reference(dslash, gauge, clover, kappa,
                                              source))
            relative = max(inverse_relative, csw_zero_dev, operator_relative)

    observations = _base(config, torch, "torch/matrix/clover")
    observations.update(_timing(samples))
    observations.update(
        clover="csw",
        csw=config.csw,
        relative=relative,
        rel=relative,
        relative_error=relative,
        inverse_relative=inverse_relative,
        operator_relative=operator_relative,
        csw_zero_dev=csw_zero_dev,
        clover_norm=_norm(torch, clover),
        clover_shape=list(clover.shape),
        clover_bytes=int(clover.numel() * clover.element_size()),
        applied_norm=_norm(torch, applied),
        round_trip_norm=_norm(torch, round_trip),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


# ---------------------------------------------------------------------------
# runner: smearing
# ---------------------------------------------------------------------------
def _wuppertal_step(torch: Any, source: Any, gauge: Any,
                    sigma: float) -> Any:
    """One independent Gaussian (Wuppertal) step, written from the formula.

    ``x' = (1 - 6 sigma) x + sigma sum_{mu=x,y,z} [U_mu x(x+mu) +
    U_mu^dag(x-mu) x(x-mu)]`` on a periodic lattice.  This is the reference the
    core ``smear.wuppertal_smear`` result is compared against; it is written
    here so the check does not call the code under test.
    """
    acc = torch.zeros_like(source)
    for mu in range(3):
        ward = mu - 4
        link = gauge[:, :, mu]
        forward = torch.roll(source, shifts=-1, dims=ward)
        dagger = torch.roll(
            link.conj().permute(1, 0, *range(2, link.ndim)),
            shifts=1, dims=ward)
        backward = torch.roll(source, shifts=1, dims=ward)
        acc = acc + torch.einsum("abxyzt,mbxyzt->maxyzt", link, forward)
        acc = acc + torch.einsum("abxyzt,mbxyzt->maxyzt", dagger, backward)
    return (1.0 - 6.0 * sigma) * source + sigma * acc


def run_smear(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/smear``: stout and Wuppertal smearing invariants.

    ``stout`` compares the serial path with ``support_parallel=True`` on an
    unpartitioned grid (with no halo exchange they must describe the same
    smearing), checks SU(3) after one and two steps, and records how far the
    field moved (``delta``).

    ``wuppertal`` is a fermion smearing, so ``su3`` reports the (untouched)
    gauge field it consumed.  ``relative_error`` is the largest of three
    invariants that must vanish: a constant field with unit links is an exact
    eigenvector (``constant_relative``), the kernel only sums spatial
    neighbours so a single-time-slice source stays there
    (``time_leak_relative``), and one step reproduces an independent
    transcription of the formula (``one_step_relative``).

    ``nstep``/``rho`` come from the options (defaults ``1``/``0.12`` for stout,
    ``10``/``1.0`` for Wuppertal, matching the tier-0 torch cases).
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, lattice, smear = modules.torch, modules.lattice, modules.smear
    su3_tol = config.float_option("su3_tol", 1e-3, ("su3_tolerance",))
    stout = config.smear == "stout"
    nstep = config.int_option("nstep", 1 if stout else 10, ("steps",))
    if nstep is None or nstep < 1:
        raise ValueError(f"nstep must be >= 1, got {nstep!r}")
    rho = config.float_option("rho", 0.12 if stout else 1.0)
    config.check_memory(
        spinor_buffers=8, note="smear",
        gauge_buffers=config.gauge_memory_buffers() + 12)
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, lattice)
            su3 = bool(lattice.check_su3(gauge, tol=su3_tol, verbose=False))
            if stout:
                samples, serial = _measure(
                    config,
                    lambda: smear.stout_smear(U=gauge, nstep=nstep, rho=rho,
                                              support_parallel=False))
                parallel = smear.stout_smear(U=gauge, nstep=nstep, rho=rho,
                                             support_parallel=True)
                relative = _relative(torch, parallel, serial)
                delta = _relative(torch, serial, gauge)
                su3_smeared = bool(lattice.check_su3(
                    serial, tol=su3_tol, verbose=False))
                su3_parallel = bool(lattice.check_su3(
                    parallel, tol=su3_tol, verbose=False))
                two_steps = smear.stout_smear(U=gauge, nstep=nstep + 1,
                                             rho=rho, support_parallel=False)
                su3_two_steps = bool(lattice.check_su3(
                    two_steps, tol=su3_tol, verbose=False))
                extra: Dict[str, Any] = {
                    "su3_smeared": su3_smeared,
                    "su3_parallel": su3_parallel,
                    "su3_two_steps": su3_two_steps,
                    "two_step_delta": _relative(torch, two_steps, serial),
                    "delta": delta,
                    "smeared_norm": _norm(torch, serial),
                    "parallel_norm": _norm(torch, parallel),
                    "sigma": None,
                    "norm_ratio": None,
                    "norm_bound": None,
                    "norm_ok": None,
                }
            else:
                sigma = rho * rho / (4.0 * nstep)
                source = _fermion_field(config, torch, config.seed + 4,
                                        default="random")
                samples, smeared = _measure(
                    config,
                    lambda: smear.wuppertal_smear(
                        src=source, U=gauge, rho=rho, nstep=nstep,
                        support_parallel=False))
                source_norm = _norm(torch, source)
                smeared_norm = _norm(torch, smeared)
                norm_ratio = (smeared_norm / source_norm
                              if source_norm else 0.0)
                norm_bound = abs(1.0 - 6.0 * sigma) + 6.0 * sigma
                norm_ok = bool(norm_ratio <= norm_bound + 1e-6)

                unit_gauge = _gauge_field(config, torch, lattice, kind="unit")
                constant = torch.ones((4, 3, *config.lattice),
                                      dtype=config.dtype, device=config.device)
                constant_smeared = smear.wuppertal_smear(
                    src=constant, U=unit_gauge, rho=rho, nstep=nstep,
                    support_parallel=False)
                constant_relative = _relative(torch, constant_smeared,
                                              constant)

                local = torch.zeros_like(source)
                local[..., 0] = source[..., 0]
                local_smeared = smear.wuppertal_smear(
                    src=local, U=gauge, rho=rho, nstep=nstep,
                    support_parallel=False)
                total = _norm(torch, local_smeared)
                time_leak_relative = (_norm(torch, local_smeared[..., 1:]) / total
                                      if total else 0.0)

                one_step = smear.wuppertal_smear(
                    src=source, U=gauge, rho=rho, nstep=1,
                    support_parallel=False)
                expected_step = _wuppertal_step(torch, source, gauge,
                                                rho * rho / 4.0)
                one_step_relative = _relative(torch, one_step, expected_step)
                relative = max(constant_relative, time_leak_relative,
                               one_step_relative)
                extra = {
                    "su3_smeared": su3,
                    "su3_parallel": None,
                    "su3_two_steps": None,
                    "delta": _relative(torch, smeared, source),
                    "smeared_norm": smeared_norm,
                    "parallel_norm": None,
                    "sigma": sigma,
                    "norm_ratio": norm_ratio,
                    "norm_bound": norm_bound,
                    "norm_ok": norm_ok,
                    "constant_relative": constant_relative,
                    "time_leak_relative": time_leak_relative,
                    "one_step_relative": one_step_relative,
                    "source_norm": source_norm,
                }

    observations = _base(config, torch, "torch/matrix/smear")
    observations.update(_timing(samples))
    observations.update(
        smear=config.smear,
        nstep=nstep,
        rho=rho,
        su3=su3,
        su3_tolerance=su3_tol,
        relative=relative,
        rel=relative,
        relative_error=relative,
    )
    observations.update(extra)
    observations.update(capture.observations())
    return _finish(observations, started)


# ---------------------------------------------------------------------------
# runners: solvers
# ---------------------------------------------------------------------------
def run_bicgstab(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/solve/bicgstab``: full-system BiCGStab solve.

    Solves ``D x = b`` on the full (non-parity) Wilson or clover operator with
    ``pyqcu.torch.solver.bistabcg`` from a zero initial guess, then rebuilds
    ``||D x - b|| / ||b||`` from the public ``give_wilson``(+``give_clover``)
    reference -- never from the solver's own recurrence.  ``converged`` is that
    number against ``tolerance``; the recurrence's last residual is reported
    separately as ``native_final_residual``.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, dslash, solver = (modules.torch, modules.dslash, modules.solver)
    kappa = config.kappa_tensor()
    clover_needed = "clover" in config.dslash_kinds
    solver_tolerance = config.solver_tolerance()
    max_iter = config.max_iter if config.max_iter is not None else 1000
    if_rtol = bool(config.option("if_rtol", True))
    config.check_memory(clover=clover_needed, spinor_buffers=14,
                        stencil_buffers=_STENCIL_BUFFERS,
                        note="bicgstab")
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, modules.lattice)
            clover = (_clover_term(config, dslash, gauge, kappa)
                      if clover_needed else None)
            b = _fermion_field(config, torch, config.seed, default="point")
            operator = dslash.operator(U=gauge, clover_term=clover,
                                       kappa=kappa, verbose=False)

            def _solve_once() -> Tuple[Any, List[float]]:
                # A fresh history per call: the solver appends to it, so reusing
                # one list across samples would corrupt the iteration count.
                call_history: List[float] = []
                vector = solver.bistabcg(
                    b=b, matvec=operator.matvec, tol=solver_tolerance,
                    max_iter=max_iter, x0=torch.zeros_like(b), if_rtol=if_rtol,
                    verbose=False, history=call_history)
                return vector, call_history

            samples, (solution, history) = _measure(config, _solve_once)
            reference = _wilson_plus_clover_reference(dslash, gauge, clover,
                                                      kappa, solution)
            residual = reference - b
            rel_res = _relative(torch, reference, b)
            iterations = max(len(history) - 1, 0)
            history_tail = [float(value) for value in history[-3:]]

    observations = _base(config, torch, "torch/matrix/solve")
    observations.update(_timing(samples))
    observations.update(
        operator="wilson+clover" if clover is not None else "wilson",
        system="full",
        dslash="+".join(config.dslash_kinds),
        solver="bistabcg",
        solver_impl="bistabcg",
        source=config.source or ("file" if config.source_file else "point"),
        tolerance=config.tolerance,
        solver_tolerance=solver_tolerance,
        if_rtol=if_rtol,
        max_iter=max_iter,
        converged=bool(rel_res <= config.tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=iterations,
        history_points=len(history),
        history_tail=history_tail,
        native_final_residual=history_tail[-1] if history_tail else None,
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def _krylov_entry(solver_module: Any, name: str) -> Tuple[Callable[..., Any], str]:
    """Resolve the ``solver`` option to a ``pyqcu.torch.solver`` callable."""
    attribute = _KRYLOV_ENTRY_POINTS[name]
    function = getattr(solver_module, attribute, None)
    if function is None:  # pragma: no cover - guarded by SOLVER_KINDS
        raise ValueError(
            f"pyqcu.torch.solver has no {attribute!r} entry point")
    return function, attribute


def run_cg(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/solve/cg``: even-odd Schur CA-CG + full residual.

    Plain CG needs a Hermitian positive-definite operator and the Wilson
    operator is only gamma5-Hermitian, so this case solves the **even-odd
    Schur complement** (QUDA ``MATPC``):

    ``S_o x_o = b_o - D_oe D_ee^-1 b_e``, ``x_e = D_ee^-1 (b_e - D_eo x_o)``.

    The gate is the physical ``||D x - b|| / ||b||`` of the reconstructed full
    field, rebuilt from the public ``give_wilson``(+``give_clover``) reference.
    ``solver`` selects the implementation (``cg``/``cacg`` -> ``cacg``,
    ``bicgstab``/``bistabcg``, ``mr``, ``fgmres``) and ``matvecs`` counts the
    operator applications of the reported solve.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, tools, dslash, solver = (modules.torch, modules.tools,
                                    modules.dslash, modules.solver)
    kappa = config.kappa_tensor()
    clover_needed = "clover" in config.dslash_kinds
    solver_tolerance = config.solver_tolerance()
    max_iter = config.max_iter if config.max_iter is not None else 400
    restart = config.restart if config.restart is not None else 30
    n_krylov = config.int_option("n_krylov", 8, ("krylov", "s"))
    omega = config.float_option("omega", 1.0)
    solver_name = config.solver_selected or "cg"
    implementation, implementation_name = _krylov_entry(solver, solver_name)
    config.check_memory(clover=clover_needed, spinor_buffers=16,
                        stencil_buffers=_STENCIL_BUFFERS, note="cg")
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, modules.lattice)
            clover = (_clover_term(config, dslash, gauge, kappa)
                      if clover_needed else None)
            # ``support_parity`` builds the compact ``D_ee^-1``/``D_oo^-1``
            # blocks the Schur complement needs; ``parity_only`` stays False so
            # the full operator is available for the independent residual.
            operator = dslash.operator(U=gauge, clover_term=clover,
                                       kappa=kappa, support_parity=True,
                                       verbose=False)
            b = _fermion_field(config, torch, config.seed + 5,
                               default="point")
            b_flat = b.reshape([12, *config.lattice])
            b_e, b_o = tools.oooxyzt2poooxyzt(input_array=b_flat, verbose=False)
            rhs = operator.give_b_parity(b_e=b_e, b_o=b_o)
            counter = {"matvec": 0}

            def _solve_once() -> Tuple[Any, List[float], int]:
                counter["matvec"] = 0

                def _counted(vector: Any) -> Any:
                    counter["matvec"] += 1
                    return operator.matvec_parity(vector)

                history: List[float] = []
                call: Dict[str, Any] = dict(
                    b=rhs, matvec=_counted, tol=solver_tolerance,
                    max_iter=max_iter, x0=torch.zeros_like(rhs), if_rtol=True,
                    verbose=False, history=history)
                if solver_name == "fgmres":
                    call["restart"] = restart
                elif solver_name == "mr":
                    call["omega"] = omega
                else:
                    call["n_krylov"] = n_krylov
                vector = implementation(**call)
                return vector, history, counter["matvec"]

            samples, (x_o, history, matvecs) = _measure(config, _solve_once)
            x_e = operator.give_x_e(b_e=b_e, x_o=x_o)
            solution = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([x_e, x_o], dim=0),
                verbose=False).reshape(b.shape)
            reference = _wilson_plus_clover_reference(dslash, gauge, clover,
                                                      kappa, solution)
            residual = reference - b
            rel_res = _relative(torch, reference, b)
            iterations = max(len(history) - 1, 0)
            history_tail = [float(value) for value in history[-3:]]

    observations = _base(config, torch, "torch/matrix/solve")
    observations.update(_timing(samples))
    observations.update(
        operator="wilson+clover" if clover is not None else "wilson",
        system="schur_eo",
        dslash="+".join(config.dslash_kinds),
        solver=solver_name,
        solver_impl=implementation_name,
        source=config.source or ("file" if config.source_file else "point"),
        tolerance=config.tolerance,
        solver_tolerance=solver_tolerance,
        if_rtol=True,
        max_iter=max_iter,
        n_krylov=n_krylov,
        restart=restart,
        omega=omega,
        converged=bool(rel_res <= config.tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=iterations,
        matvecs=int(matvecs),
        history_points=len(history),
        history_tail=history_tail,
        native_final_residual=history_tail[-1] if history_tail else None,
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_solve(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Dispatch a solve on the ``solver`` option.

    ``cg``/``cacg``/``all``/``mg``/``multigrid`` -> :func:`run_cg` (the
    even-odd Schur CA-CG), ``bicgstab``/``bistabcg`` -> :func:`run_bicgstab`
    (the full-system BiCGStab), ``mr``/``fgmres`` -> :func:`run_cg` with that
    Krylov method on the Schur operator.  The matrix rows name the concrete
    runner; this entry point exists for a caller that wants one solver knob.
    """
    config = _Config(args, kwargs)
    if config.solver_selected == "bicgstab":
        return run_bicgstab(*args, **kwargs)
    return run_cg(*args, **kwargs)


# ---------------------------------------------------------------------------
# runners: multigrid
# ---------------------------------------------------------------------------
def run_mg_legacy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/mg/legacy``: explicit torch ``LegacyMultigrid``.

    Runs the pure torch legacy hierarchy (``with_cuda_qcu`` stays ``False``,
    therefore ``clover_ee_inv``/``clover_oo_inv`` are never handed in).
    ``support_parity=False`` is used deliberately: the compact odd-field path of
    this pure torch variant feeds a ``t // 2`` residual into ``tools.restrict``
    with a null basis built for the full time extent (DimensionError in
    ``_legacy_multigrid.coarse_correct``), so the full-field variant is the one
    that runs end to end and the requested configuration is reported instead of
    being hidden.

    ``converged`` is the independently recomputed ``||D x - b|| / ||b||``
    against ``tolerance``; the hierarchy's own stop reason and residual are
    reported as ``native_*`` observations.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, dslash, solver = (modules.torch, modules.dslash, modules.solver)
    kappa = config.kappa_tensor()
    clover_needed = "clover" in config.dslash_kinds
    # The hierarchy stops on a transformed (Schur/MATPC) norm, so ask for one
    # extra order below the independent gate: c128 has the headroom, c64 does
    # not (fp32 stalls around 1e-7).
    solver_tolerance = config.solver_tolerance(
        1e-7 if config.dtype_tag == "c64" else 1e-9)
    max_iter = config.max_iter if config.max_iter is not None else 1000
    max_level = config.mg_levels if config.mg_levels is not None else 2
    nvec = config.int_option("nvec", 12, ("num_vec",))
    num_restart = config.restart if config.restart is not None else 3
    support_parity = bool(config.option("support_parity", False))
    min_size = config.int_option("min_size", 4)
    config.check_memory(clover=clover_needed, spinor_buffers=20,
                        stencil_buffers=_STENCIL_BUFFERS,
                        mg_levels=max_level, note="mg/legacy")
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, modules.lattice)
            clover = (_clover_term(config, dslash, gauge, kappa)
                      if clover_needed else None)
            b = _fermion_field(config, torch, config.seed, default="point")
            dof_list = [12] + [nvec] * 9
            build_started = time.perf_counter()
            multigrid = solver.LegacyMultigrid(
                dtype_list=[config.dtype] * 10,
                device_list=[config.device] * 10,
                U=gauge, clover_term=clover, kappa=kappa,
                tol=solver_tolerance, max_iter=max_iter, max_level=max_level,
                mg_grid_size=list(config.mg_grid),
                num_restart=num_restart, support_parity=support_parity,
                min_size=min_size,
                dof_list=dof_list, verbose=False, seed=config.seed)
            construct_seconds = time.perf_counter() - build_started
            try:
                setup_started = time.perf_counter()
                multigrid.init()
                setup_seconds = time.perf_counter() - setup_started
                levels = int(getattr(multigrid, "num_level", 0))
                samples, solution = _measure(config,
                                             lambda: multigrid.solve(b=b))
                report = dict(getattr(multigrid, "last_solve_report", {}) or {})
                history = [float(value) for value in
                           (getattr(multigrid, "convergence_history", [])
                            or [])]
            finally:
                multigrid.end()

            reference = _wilson_plus_clover_reference(dslash, gauge, clover,
                                                      kappa, solution)
            residual = reference - b
            rel_res = _relative(torch, reference, b)

    observations = _base(config, torch, "torch/matrix/mg")
    observations.update(_timing(samples))
    observations.update(
        variant="legacy",
        mg_backend="torch",
        mg_variant="legacy",
        dslash="+".join(config.dslash_kinds),
        with_cuda_qcu=bool(getattr(multigrid, "with_cuda_qcu", False)),
        source=config.source or ("file" if config.source_file else "point"),
        tolerance=config.tolerance,
        solver_tolerance=solver_tolerance,
        support_parity=support_parity,
        max_level=max_level,
        mg_levels=max_level,
        smoother_counts_supported=False,
        min_size=min_size,
        nvec=nvec,
        dof_list=dof_list[:max(levels, 1)],
        num_restart=num_restart,
        max_iter=max_iter,
        levels=levels,
        converged=bool(rel_res <= config.tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=int(report.get("iterations", -1)),
        vcycles=int(report.get("vcycles", -1)),
        residual_scope=report.get("residual_scope"),
        outer_solver=report.get("outer_solver"),
        native_stop_reason=report.get("stop_reason"),
        native_converged=bool(
            str(report.get("stop_reason", "")).endswith("tolerance")),
        native_history_tail=history[-3:],
        construct_seconds=construct_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=_timing(samples)["seconds"],
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_mg_strict(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/matrix/*/mg/strict``: explicit torch ``StrictMultigrid``.

    Strict is opt-in only: this runner always asks for ``hierarchy_mode``
    "strict" through the public ``StrictMultigrid`` class and never falls back
    to legacy.  ``mg_grid``/``mg_levels``/``mg_nu_pre``/``mg_nu_post``/
    ``restart``/``max_iter`` map to ``block_size``/``max_level``/``nu_pre``/
    ``nu_post``/``restart``/``max_iter``; the smoother counts are only passed
    when they were requested, so the core defaults stay in force otherwise.
    ``converged`` is again the independent ``relative_residual`` gate.
    """
    config = _Config(args, kwargs)
    started = time.perf_counter()
    modules = _load()
    torch, dslash, solver = (modules.torch, modules.dslash, modules.solver)
    kappa = config.kappa_tensor()
    clover_needed = "clover" in config.dslash_kinds
    solver_tolerance = config.solver_tolerance(
        1e-7 if config.dtype_tag == "c64" else 1e-9)
    max_iter = config.max_iter if config.max_iter is not None else 200
    max_level = config.mg_levels if config.mg_levels is not None else 2
    nvec = config.int_option("nvec", 6, ("num_vec",))
    restart = config.restart if config.restart is not None else 20
    materialize = bool(config.option("materialize_coarse", True))
    smoother: Dict[str, int] = {}
    if config.mg_nu_pre is not None:
        smoother["nu_pre"] = int(config.mg_nu_pre)
    if config.mg_nu_post is not None:
        smoother["nu_post"] = int(config.mg_nu_post)
    config.check_memory(clover=clover_needed, spinor_buffers=20,
                        stencil_buffers=_STENCIL_BUFFERS,
                        mg_levels=max_level, note="mg/strict")
    with _skip_on_oom(config):
        with _StdoutCapture(config.capture_stdout) as capture:
            gauge = _gauge_field(config, torch, modules.lattice)
            clover = (_clover_term(config, dslash, gauge, kappa)
                      if clover_needed else None)
            b = _fermion_field(config, torch, config.seed, default="point")
            build_started = time.perf_counter()
            multigrid = solver.StrictMultigrid(
                U=gauge, clover_term=clover, kappa=kappa,
                lat_size=list(config.lattice), nvec_list=[nvec],
                max_level=max_level, block_size=tuple(config.mg_grid),
                tol=solver_tolerance, max_iter=max_iter,
                restart=restart, use_parity=True,
                materialize_coarse=materialize, verbose=False,
                seed=config.seed, **smoother)
            construct_seconds = time.perf_counter() - build_started
            try:
                setup_started = time.perf_counter()
                multigrid.setup()
                setup_seconds = time.perf_counter() - setup_started
                samples, solution = _measure(config,
                                             lambda: multigrid.solve(b))
                report = dict(getattr(multigrid, "last_solve_report", {}) or {})
                history = [float(value) for value in
                           (getattr(multigrid, "convergence_history", [])
                            or [])]
            finally:
                multigrid.close()

            reference = _wilson_plus_clover_reference(dslash, gauge, clover,
                                                      kappa, solution)
            residual = reference - b
            rel_res = _relative(torch, reference, b)

    observations = _base(config, torch, "torch/matrix/mg")
    observations.update(_timing(samples))
    observations.update(
        variant="strict",
        mg_backend="torch",
        mg_variant="strict",
        hierarchy_mode="strict",
        dslash="+".join(config.dslash_kinds),
        source=config.source or ("file" if config.source_file else "point"),
        tolerance=config.tolerance,
        solver_tolerance=solver_tolerance,
        max_level=max_level,
        mg_levels=max_level,
        smoother_counts_supported=True,
        smoother=dict(smoother),
        nvec=nvec,
        restart=restart,
        max_iter=max_iter,
        use_parity=True,
        materialize_coarse=materialize,
        converged=bool(rel_res <= config.tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=int(report.get("iterations", -1)),
        vcycles=(int(report["vcycles"])
                 if report.get("vcycles") is not None else None),
        residual_scope=report.get("residual_scope"),
        outer_solver=report.get("outer_solver"),
        native_stop_criterion=report.get("stop_criterion"),
        native_relative_residual=report.get("native_relative_residual"),
        native_converged=bool(
            report.get("native_relative_residual") is not None
            and float(report["native_relative_residual"])
            <= config.tolerance),
        native_history_tail=history[-3:],
        construct_seconds=construct_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=_timing(samples)["seconds"],
    )
    observations.update(capture.observations())
    return _finish(observations, started)


#: Runner registry keyed by the ``MatrixCase.runner`` short name.
RUNNERS: Mapping[str, Callable[..., Dict[str, Any]]] = MappingProxyType({
    "run_bicgstab": run_bicgstab,
    "run_cg": run_cg,
    "run_clover": run_clover,
    "run_dslash": run_dslash,
    "run_gauge": run_gauge,
    "run_mg_legacy": run_mg_legacy,
    "run_mg_strict": run_mg_strict,
    "run_smear": run_smear,
    "run_solve": run_solve,
})


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixCase:
    """One declared matrix row: what to run and which gates decide it.

    A ``MatrixCase`` is the :class:`pyqcu.test.common.Case` of the matrix: the
    tag, the runner, the lattice/dtype of the row, the runner keyword arguments
    (``options``) and the ``expect`` gates.  ``heavy`` marks a row that must
    never run unless it was asked for explicitly, and ``tier`` mirrors the
    policy tiers (0 = quick gate, 1 = production shape).
    """

    tag: str
    runner: str
    kind: str
    lattice: Tuple[int, int, int, int]
    dtype: str
    expect: Mapping[str, Mapping[str, Any]]
    options: Mapping[str, Any] = field(default_factory=dict)
    tier: int = 0
    heavy: bool = False
    timeout: float = _QUICK_TIMEOUT
    preset: str = DEFAULT_PRESET

    def __post_init__(self) -> None:
        object.__setattr__(self, "lattice", resolve_lattice(self.lattice))
        object.__setattr__(self, "dtype", resolve_dtype(self.dtype))
        object.__setattr__(
            self, "expect",
            MappingProxyType({str(key): MappingProxyType(dict(spec))
                              for key, spec in self.expect.items()}))
        object.__setattr__(self, "options",
                           MappingProxyType({str(key): value for key, value
                                             in self.options.items()}))
        object.__setattr__(self, "tier", int(self.tier))
        object.__setattr__(self, "heavy", bool(self.heavy))
        object.__setattr__(self, "timeout", float(self.timeout))
        if not self.tag.startswith("torch/"):
            raise ValueError(
                f"matrix tag {self.tag!r} must start with 'torch/'")
        if self.runner not in RUNNERS:
            raise ValueError(
                f"matrix row {self.tag!r}: unknown runner {self.runner!r}; "
                f"choose from {', '.join(sorted(RUNNERS))}")

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe description (used by ``--list``)."""
        return {
            "tag": self.tag,
            "runner": f"torch_matrix:{self.runner}",
            "kind": self.kind,
            "preset": self.preset,
            "heavy": self.heavy,
            "tier": self.tier,
            "lattice": list(self.lattice),
            "lattice_tag": lattice_tag(self.lattice),
            "dtype": self.dtype,
            "timeout": self.timeout,
            "options": dict(self.options),
            "expect": {key: dict(spec) for key, spec in self.expect.items()},
        }

    def case(self) -> Case:
        """The equivalent :class:`pyqcu.test.common.Case` (registry form)."""
        return Case(
            tag=self.tag,
            backend="torch",
            kind=self.kind,
            runner=f"torch_matrix:{self.runner}",
            lattice=self.lattice,
            dtype=self.dtype,
            tier=self.tier,
            expect=self.expect,
            timeout=self.timeout,
        )


def _row(preset: str, lattice: Tuple[int, int, int, int], dtype: str,
         area: str, runner: str, kind: str,
         expect: Mapping[str, Mapping[str, Any]],
         options: Mapping[str, Any], *, heavy: bool) -> MatrixCase:
    """Build one matrix row (tag = ``torch/matrix/<preset>/<area>``)."""
    return MatrixCase(
        tag=f"torch/matrix/{preset}/{area}",
        runner=runner,
        kind=kind,
        lattice=lattice,
        dtype=dtype,
        expect=expect,
        options=options,
        tier=1 if heavy else 0,
        heavy=heavy,
        timeout=_HEAVY_TIMEOUT if heavy else _QUICK_TIMEOUT,
        preset=preset,
    )


def _rows(preset: str, lattice: Tuple[int, int, int, int], dtype: str, *,
          gauge_file: Optional[str] = None,
          source_file: Optional[str] = None) -> List[MatrixCase]:
    """Every runner row of one preset at one dtype."""
    heavy = (preset in HEAVY_PRESETS
             or (tuple(lattice) != QUICK_LATTICE
                 and math.prod(lattice) > math.prod(QUICK_LATTICE)))
    operator_gate = policy.RESIDUAL_GATES["operator"]
    gauge_gate = policy.RESIDUAL_GATES["gauge"]
    rows = [
        _row(preset, lattice, dtype, "gauge/random", "run_gauge", "gauge",
             {**gauge_gate, "relative_error": {"le": "compare"},
              "distinct_relative": {"gt": 0.0}},
             {"gauge": "random", "gauge_sigma": DEFAULT_GAUGE_SIGMA},
             heavy=heavy),
        _row(preset, lattice, dtype, "gauge/unit", "run_gauge", "gauge",
             {**gauge_gate, "relative_error": {"le": "compare"}},
             {"gauge": "unit"}, heavy=heavy),
        _row(preset, lattice, dtype, "gauge/ones", "run_gauge", "gauge",
             {"relative_error": {"le": "compare"}, "finite": {"eq": True},
              "su3": {"eq": False}},
             {"gauge": "ones"}, heavy=heavy),
        _row(preset, lattice, dtype, "dslash/wilson", "run_dslash", "dslash",
             operator_gate,
             {"dslash": "wilson", "gauge": "random", "source": "random"},
             heavy=heavy),
        _row(preset, lattice, dtype, "dslash/clover", "run_dslash", "dslash",
             operator_gate,
             {"dslash": "clover", "csw": DEFAULT_CSW, "gauge": "random",
              "source": "random"}, heavy=heavy),
        _row(preset, lattice, dtype, "clover/term", "run_clover", "clover",
             operator_gate,
             {"csw": DEFAULT_CSW, "gauge": "random", "source": "random"},
             heavy=heavy),
        _row(preset, lattice, dtype, "smear/stout", "run_smear", "smear",
             policy.RESIDUAL_GATES["smear"],
             {"smear": "stout", "gauge": "random", "nstep": 1, "rho": 0.12},
             heavy=heavy),
        _row(preset, lattice, dtype, "smear/wuppertal", "run_smear", "smear",
             policy.RESIDUAL_GATES["smear"],
             {"smear": "wuppertal", "gauge": "random", "source": "random",
              "nstep": 10, "rho": 1.0}, heavy=heavy),
        _row(preset, lattice, dtype, "solve/bicgstab", "run_bicgstab", "solve",
             policy.RESIDUAL_GATES["solver"],
             {"solver": "bicgstab", "dslash": "wilson", "gauge": "random",
              "source": "point", "max_iter": 1000}, heavy=heavy),
        _row(preset, lattice, dtype, "solve/cg", "run_cg", "solve",
             policy.RESIDUAL_GATES["solver"],
             {"solver": "cg", "dslash": "wilson", "gauge": "random",
              "source": "point", "max_iter": 400}, heavy=heavy),
        _row(preset, lattice, dtype, "mg/legacy", "run_mg_legacy", "mg",
             {**policy.RESIDUAL_GATES["mg"], "variant": {"eq": "legacy"}},
             {"gauge": "random", "source": "point", "dslash": "clover",
              "mg_levels": 2, "max_iter": 1000, "restart": 3}, heavy=heavy),
        _row(preset, lattice, dtype, "mg/strict", "run_mg_strict", "mg",
             {**policy.RESIDUAL_GATES["mg"], "variant": {"eq": "strict"}},
             {"gauge": "random", "source": "point", "dslash": "clover",
              "mg_levels": 2, "max_iter": 200, "restart": 20}, heavy=heavy),
    ]
    if gauge_file is not None:
        rows.append(_row(preset, lattice, dtype, "gauge/file", "run_gauge",
                         "gauge", gauge_gate,
                         {"gauge_file": str(gauge_file)}, heavy=heavy))
    if gauge_file is not None or source_file is not None:
        options: Dict[str, Any] = {"dslash": "wilson"}
        if gauge_file is not None:
            options["gauge_file"] = str(gauge_file)
        if source_file is not None:
            options["source_file"] = str(source_file)
        rows.append(_row(preset, lattice, dtype, "dslash/file", "run_dslash",
                         "dslash", operator_gate, options, heavy=heavy))
    return rows


#: The default selection: the quick gate at ``c64``.  It never contains an
#: ``L24``/``L32`` row - those live in :data:`HEAVY_MATRIX` and must be asked
#: for explicitly.
MATRIX: Tuple[MatrixCase, ...] = tuple(_rows(
    DEFAULT_PRESET, QUICK_LATTICE, policy.DEFAULT_DTYPE))
#: Alias of :data:`MATRIX` for callers that name the quick selection.
SMOKE_MATRIX: Tuple[MatrixCase, ...] = MATRIX
#: The production presets, declared but **not** part of the default selection.
HEAVY_MATRIX: Tuple[MatrixCase, ...] = tuple(
    row for preset in sorted(HEAVY_PRESETS)
    for row in _rows(preset, LATTICE_PRESETS[preset], policy.DEFAULT_DTYPE))
#: Tag -> runner, for the declared (quick) matrix.
TAG_RUNNERS: Mapping[str, Callable[..., Dict[str, Any]]] = MappingProxyType(
    {entry.tag: RUNNERS[entry.runner] for entry in MATRIX})


def _as_list(values: Any) -> List[Any]:
    """Flatten a scalar / comma-and-space separated string / iterable."""
    if values is None:
        return []
    if isinstance(values, str):
        items: List[Any] = []
        for part in values.replace(",", " ").split():
            items.append(part)
        return items
    if isinstance(values, Iterable):
        return list(values)
    return [values]


def _lattice_specs(specs: Any) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Expand preset names, ``all``/``heavy`` and explicit shapes."""
    items = _as_list(specs) or [DEFAULT_PRESET]
    resolved: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            lowered = _normalise_preset(text)
            if lowered in ("all", "*"):
                for name in MATRIX_PRESETS:
                    resolved.append((name, LATTICE_PRESETS[name]))
                continue
            if lowered in ("heavy", "production"):
                for name in sorted(HEAVY_PRESETS):
                    resolved.append((name, LATTICE_PRESETS[name]))
                continue
            name = _preset_name(text)
            if name is not None:
                resolved.append((name, LATTICE_PRESETS[name]))
                continue
        shape = resolve_lattice(item)
        resolved.append((lattice_preset(shape) or lattice_tag(shape), shape))
    unique: List[Tuple[str, Tuple[int, int, int, int]]] = []
    seen = set()
    for name, shape in resolved:
        key = (name, tuple(shape))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, tuple(shape)))
    return unique


def matrix_cases(*, presets: Any = None, dtypes: Any = None,
                 gauge_file: Optional[str] = None,
                 source_file: Optional[str] = None) -> Tuple[MatrixCase, ...]:
    """Build the matrix rows for ``presets`` x ``dtypes``.

    ``presets`` accepts preset names (case-insensitive), ``all`` (every
    declared preset, heavy ones included), ``heavy`` (``L24`` + ``L32``) and
    explicit shapes; the default is ``quick`` only, so a caller that does not
    ask for ``L24``/``L32`` never gets them.  ``dtypes`` accepts any spelling
    :func:`resolve_dtype` understands; the default is the policy dtype (c64).

    ``gauge_file``/``source_file`` add the two file-input rows
    (``gauge/file``, ``dslash/file``); without them the matrix only builds its
    fields from options, so an accidental production file can never change
    what the generated-field rows mean.
    """
    specs = _lattice_specs(presets)
    dtype_items = _as_list(dtypes) or [policy.DEFAULT_DTYPE]
    dtype_list: List[str] = []
    for item in dtype_items:
        tag = resolve_dtype(item)
        if tag not in dtype_list:
            dtype_list.append(tag)
    cases: List[MatrixCase] = []
    seen = set()
    for name, shape in specs:
        for dtype in dtype_list:
            for entry in _rows(name, shape, dtype, gauge_file=gauge_file,
                               source_file=source_file):
                if (entry.tag, entry.dtype) in seen:
                    continue
                seen.add((entry.tag, entry.dtype))
                cases.append(entry)
    return tuple(cases)


def _tag_matches(entry: MatrixCase, pattern: str) -> bool:
    """True when ``pattern`` selects ``entry`` (tag, suffix, segment or prefix).

    ``torch/matrix/quick/solve/cg`` is selected by its full tag, by the
    ``solve/cg`` suffix, by the ``solve``/``cg`` segment, by the
    ``torch/matrix/quick`` prefix and by the ``solve`` kind.
    """
    if not pattern:
        return True
    if entry.tag == pattern or entry.kind == pattern:
        return True
    if entry.tag.startswith(f"{pattern}/") or entry.tag.endswith(f"/{pattern}"):
        return True
    return pattern in entry.tag.split("/")


def select_cases(*, presets: Any = None, dtypes: Any = None,
                 tags: Any = None, kinds: Any = None,
                 runners: Any = None, include_heavy: Optional[bool] = None,
                 gauge_file: Optional[str] = None,
                 source_file: Optional[str] = None) -> Tuple[MatrixCase, ...]:
    """Select matrix rows by preset/dtype/tag/kind/runner.

    ``include_heavy=False`` (the default behaviour of :func:`matrix_cases`) is
    enforced again here, so no filter combination can smuggle an ``L24``/
    ``L32`` row into a run that did not ask for it.  ``include_heavy=True``
    keeps them; ``None`` keeps whatever the preset selection produced.
    """
    cases = matrix_cases(presets=presets, dtypes=dtypes,
                         gauge_file=gauge_file, source_file=source_file)
    tag_patterns = _as_list(tags)
    kind_patterns = _as_list(kinds)
    runner_patterns = _as_list(runners)
    selected: List[MatrixCase] = []
    for entry in cases:
        if include_heavy is False and entry.heavy:
            continue
        if tag_patterns and not any(_tag_matches(entry, str(pattern))
                                    for pattern in tag_patterns):
            continue
        if kind_patterns and entry.kind not in {str(kind) for kind in
                                                kind_patterns}:
            continue
        if runner_patterns and entry.runner not in {str(runner) for runner in
                                                    runner_patterns}:
            continue
        selected.append(entry)
    return tuple(selected)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _split_skip_reason(error: BaseException) -> Tuple[str, Optional[str]]:
    """Split a skip into ``(reason, token)`` exactly like ``common`` does."""
    text = str(error).strip()
    token = error.token if isinstance(error, Skip) else None
    if token is None and ":" in text:
        head, _, tail = text.partition(":")
        if policy.is_skip_token(head.strip()) and tail.strip():
            return tail.strip(), head.strip()
    return text, (token if policy.is_skip_token(token) else None)


def _skip_observation(observation: Mapping[str, Any]) -> Tuple[bool, str,
                                                               Optional[str]]:
    """Detect a skip request in an observation mapping."""
    skip = observation.get("skip")
    if isinstance(skip, str):
        reason, token = _split_skip_reason(Skip(skip))
        return True, reason, token
    if skip is True:
        reason, token = _split_skip_reason(Skip(str(observation.get("reason")
                                                    or "")))
        return True, reason, token
    declared = observation.get("status")
    if isinstance(declared, str) and declared.strip().upper() == "SKIP":
        reason, token = _split_skip_reason(Skip(str(observation.get("reason")
                                                    or "")))
        return True, reason, token
    return False, "", None


def _metrics(observation: Any) -> Dict[str, Any]:
    """Observation payload without the harness-reserved keys."""
    if not isinstance(observation, Mapping):
        return {}
    return {key: value for key, value in observation.items()
            if key not in _RESERVED_OBSERVATIONS}


_SIGALRM = getattr(signal, "SIGALRM", None)


def _on_timeout(signum: int, frame: Any) -> None:  # pragma: no cover - signal
    raise TimeoutError("matrix case exceeded its wall-clock budget")


def _arm_timeout(timeout: Optional[float]) -> Optional[Any]:
    """Arm a per-case wall clock when the platform supports one."""
    if timeout is None or float(timeout) <= 0.0:
        return None
    if _SIGALRM is None or not hasattr(signal, "setitimer"):
        return None
    try:
        previous = signal.getsignal(_SIGALRM)
        signal.signal(_SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    except (ValueError, OSError, RuntimeError):
        return None
    return previous


def _disarm_timeout(previous: Optional[Any]) -> None:
    """Cancel a timer armed by :func:`_arm_timeout` and restore the handler."""
    if previous is None:
        return
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    except (ValueError, OSError):
        pass
    try:
        signal.signal(_SIGALRM, previous)
    except (ValueError, OSError, TypeError):
        pass


def run_case(entry: MatrixCase, *, timeout: Optional[float] = None,
             atol: Optional[float] = None,
             **overrides: Any) -> Result:
    """Execute one matrix row and judge it with its own ``expect`` gates.

    The control flow is :func:`pyqcu.test.common.run_case`'s: a :class:`Skip`
    becomes ``SKIP`` (with its token), ``TimeoutError`` becomes ``TIMEOUT``,
    any other exception becomes ``ERROR``, and the verdict comes from
    :func:`pyqcu.test.common.evaluate_expect` - never from the runner.  A
    missing observation fails, and a runner that returns a verdict is an
    ``ERROR``, exactly as in the CLI harness.
    """
    started = time.perf_counter()
    kwargs: Dict[str, Any] = dict(entry.options)
    kwargs.setdefault("lattice", entry.lattice)
    kwargs.setdefault("dtype", entry.dtype)
    kwargs.update(overrides)
    budget = entry.timeout if timeout is None else float(timeout)
    status = "PASS"
    reason: Optional[str] = None
    failed: Tuple[str, ...] = ()
    skip_token: Optional[str] = None
    observations: Any = None
    timer: Optional[Any] = None
    try:
        timer = _arm_timeout(budget)
        try:
            observations = RUNNERS[entry.runner](**kwargs)
        finally:
            _disarm_timeout(timer)
    except Skip as error:
        reason, skip_token = _split_skip_reason(error)
        status = "SKIP" if reason else "ERROR"
        reason = reason or "skip without a reason"
    except TimeoutError:
        status = "TIMEOUT"
        reason = f"exceeded {budget:g}s wall-clock budget"
    except Exception as error:  # noqa: BLE001 - a case never escapes the harness
        status = "ERROR"
        reason = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started

    if status == "PASS":
        if not isinstance(observations, Mapping):
            status = "ERROR"
            reason = (f"runner returned {type(observations).__name__}, "
                      "expected a mapping of observations")
        else:
            requested, skip_reason, observed_token = _skip_observation(
                observations)
            if requested:
                status = "SKIP" if skip_reason else "ERROR"
                reason = skip_reason or "skip without a reason"
                skip_token = observed_token
            else:
                declared = observations.get("status")
                if isinstance(declared, str) and declared.strip().upper() in (
                        "PASS", "FAIL"):
                    status = "ERROR"
                    reason = (f"runner returned a verdict (status={declared!r});"
                              " runners report observations, expect decides")
                else:
                    try:
                        failed = evaluate_expect(entry.expect, observations,
                                                 dtype=entry.dtype, atol=atol)
                    except ValueError as error:
                        status = "ERROR"
                        reason = f"invalid expect gate: {error}"
                    else:
                        if failed:
                            status = "FAIL"
                            reason = "; ".join(failed)
    if (status in ("PASS", "FAIL") and timer is None and budget
            and elapsed > float(budget)):
        status = "TIMEOUT"
        reason = f"exceeded {budget:g}s wall-clock budget"
        failed = ()

    metrics = _metrics(observations)
    metrics["elapsed_seconds"] = round(elapsed, 6)
    seconds = elapsed
    if isinstance(observations, Mapping):
        raw = observations.get("seconds")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            seconds = float(raw)
    seed = kwargs.get("seed")
    mpi_size, _mpi_rank = mpi_info()
    return Result(
        tag=entry.tag,
        backend="torch",
        kind=entry.kind,
        status=status,
        seconds=seconds,
        metrics=metrics,
        failed=failed,
        reason=reason,
        lattice=entry.lattice,
        dtype=entry.dtype,
        seed=DEFAULT_SEED if seed is None else int(seed),
        mpi_size=int(mpi_size),
        tier=entry.tier,
        runner=f"torch_matrix:{entry.runner}",
        skip_token=skip_token,
    )


def run_matrix(cases: Sequence[MatrixCase], *, timeout: Optional[float] = None,
               atol: Optional[float] = None, fail_fast: bool = False,
               on_result: Optional[Callable[[Result], None]] = None,
               overrides: Optional[Mapping[str, Any]] = None
               ) -> List[Result]:
    """Run every case in order; return the results (never raises per case).

    ``fail_fast`` stops after the first ``FAIL``/``ERROR``/``TIMEOUT`` (a
    ``SKIP`` is not a failure and never stops a run).  ``overrides`` are
    forwarded to every runner (``threads``, ``seed``, ``mass``, ...).
    """
    overrides = dict(overrides or {})
    results: List[Result] = []
    for entry in cases:
        result = run_case(entry, timeout=timeout, atol=atol, **overrides)
        results.append(result)
        if on_result is not None:
            on_result(result)
        if fail_fast and result.status in ("FAIL", "ERROR", "TIMEOUT"):
            break
    return results


def _listing_payload(cases: Sequence[MatrixCase]) -> Dict[str, Any]:
    """JSON-safe ``--list`` document for the selected matrix rows."""
    return {
        "package": policy.TEST_PACKAGE,
        "module": "torch_matrix",
        "quick_lattice": list(QUICK_LATTICE),
        "presets": list(MATRIX_PRESETS),
        "heavy_presets": sorted(HEAVY_PRESETS),
        "count": len(cases),
        "heavy_count": sum(1 for entry in cases if entry.heavy),
        "cases": [entry.as_dict() for entry in cases],
    }


def matrix_payload(cases: Sequence[MatrixCase], results: Sequence[Result], *,
                   presets: Any = None, dtypes: Any = None,
                   duration_seconds: float = 0.0,
                   allow_all_skipped: bool = False) -> Dict[str, Any]:
    """JSON-safe report of one matrix run (``--json`` payload).

    ``status``/``exit_code`` follow :func:`pyqcu.test.policy.exit_code`: any
    ``FAIL``/``ERROR``/``TIMEOUT`` is a failure, a run where every case skipped
    is a failure unless ``allow_all_skipped`` is set, and an empty selection is
    never a pass.
    """
    results = list(results)
    counts, status = summarize(tuple(results))
    effective_presets = [str(item) for item in _as_list(presets)] or sorted(
        {entry.preset for entry in cases})
    exit_code = policy.exit_code(status, all_skipped=bool(results) and all(
        result.status == "SKIP" for result in results),
        allow_all_skipped=allow_all_skipped)
    return {
        "package": policy.TEST_PACKAGE,
        "module": "torch_matrix",
        "status": status,
        "exit_code": exit_code,
        "counts": dict(counts),
        "selected": len(cases),
        "executed": len(results),
        "all_skipped": bool(results) and all(result.status == "SKIP"
                                             for result in results),
        "allow_all_skipped": bool(allow_all_skipped),
        "duration_seconds": round(float(duration_seconds), 6),
        "quick_lattice": list(QUICK_LATTICE),
        "presets": list(MATRIX_PRESETS),
        "heavy_presets": sorted(HEAVY_PRESETS),
        "requested_presets": effective_presets,
        "lattices": sorted({lattice_tag(result.lattice)
                            for result in results}),
        "dtypes": sorted({result.dtype for result in results}),
        "tags": [result.tag for result in results],
        "cases": [result.as_dict() for result in results],
        "env": probe_env(),
    }


# ---------------------------------------------------------------------------
# standalone harness
# ---------------------------------------------------------------------------
def _format_result(result: Result) -> str:
    """One stable, parseable stdout line per case."""
    seconds = "-" if result.seconds is None else f"{result.seconds:.3f}s"
    return (f"{result.status:<7} {result.tag:<54} "
            f"{lattice_tag(result.lattice):<14} {result.dtype:<5} "
            f"{seconds:>12}")


def _emit(payload: Any, target: Optional[str]) -> Optional[str]:
    """Write ``payload`` as JSON to ``target`` (``-`` = stdout).

    Returns an error message instead of raising, so the harness can map a
    failed write to the usage exit code.
    """
    if target is None:
        return None
    try:
        if target == "-":
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True,
                             ensure_ascii=False, allow_nan=False))
        else:
            write_json_atomic(target, payload, indent=2)
    except (OSError, TypeError, ValueError) as error:
        return f"cannot write {target!r}: {type(error).__name__}: {error}"
    return None


def build_parser() -> argparse.ArgumentParser:
    """Parser for ``python -m pyqcu.test.torch_matrix``."""
    parser = argparse.ArgumentParser(
        prog="python -m pyqcu.test.torch_matrix",
        description="Torch parameter-matrix runners of the formal pyqcu.test "
                    "port: dslash, clover, gauge, smear, CG/BiCGStab and "
                    "multigrid at every declared lattice preset, judged by the "
                    "policy gates.  L24/L32 are heavy and only run when "
                    "--preset names them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="print the selected matrix rows and exit "
                             "(no torch import, no numerics)")
    parser.add_argument("--json", nargs="?", const="-", default=None,
                        metavar="PATH",
                        help="emit the report as JSON: PATH, or '-' for stdout")
    parser.add_argument("--preset", action="append", default=None,
                        metavar="NAME",
                        help="lattice preset(s): quick (default), L24, L32, "
                             "all or heavy; repeatable/comma separated")
    parser.add_argument("--dtype", action="append", default=None,
                        metavar="NAME",
                        help="dtype(s): c64 (default), c128 and the aliases "
                             "of resolve_dtype; repeatable/comma separated")
    parser.add_argument("--only", action="append", default=None,
                        metavar="TAG",
                        help="select rows by tag, tag prefix, kind or tag "
                             "segment; repeatable")
    parser.add_argument("--gauge-file", default=None, metavar="PATH",
                        help="HDF5 gauge field [3,3,4,x,y,z,t] to add as the "
                             "gauge/file row and to hand to the file rows")
    parser.add_argument("--source-file", default=None, metavar="PATH",
                        help="HDF5 fermion field [4,3,x,y,z,t] for the "
                             "dslash/file row")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads for the run")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"gauge/source seed (default {DEFAULT_SEED})")
    parser.add_argument("--mass", type=float, default=None,
                        help="Dirac mass; kappa = 1/(2*mass + 8)")
    parser.add_argument("--kappa", type=float, default=None,
                        help=f"hopping parameter (default {DEFAULT_KAPPA})")
    parser.add_argument("--csw", type=float, default=None,
                        help=f"clover coefficient (default {DEFAULT_CSW})")
    parser.add_argument("--gauge", default=None,
                        choices=policy.GAUGE_CHOICES,
                        help="background field for the generated rows")
    parser.add_argument("--gauge-sigma", "--guage-sigma", dest="gauge_sigma",
                        type=float, default=None,
                        help="Gaussian width of the generated SU(3) field")
    parser.add_argument("--source", default=None, choices=policy.SOURCE_CHOICES,
                        help="fermion source for the solver rows")
    parser.add_argument("--dslash", default=None, choices=policy.DSLASH_CHOICES,
                        help="operator override handed to every runner")
    parser.add_argument("--solver", default=None,
                        choices=tuple(SOLVER_KINDS),
                        help="solver override handed to every runner")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="iteration budget override")
    parser.add_argument("--restart", type=int, default=None,
                        help="Krylov/MG restart override")
    parser.add_argument("--tolerance", "--tol", dest="tolerance", type=float,
                        default=None,
                        help=f"converged threshold (default "
                             f"{DEFAULT_TOLERANCE:g})")
    parser.add_argument("--atol", type=float, default=None,
                        help="override the 'compare' gate only (never the "
                             "solve/gamma gates)")
    parser.add_argument("--repeat", type=int, default=None,
                        help="timing samples per case (default 1)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="discarded timing samples (default 1)")
    parser.add_argument("--device", default=None,
                        help="auto/cpu/gpu/dcu/npu or a concrete device string")
    parser.add_argument("--device-id", type=int, default=None,
                        help="accelerator index when --device names none")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-case wall-clock budget in seconds")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop after the first FAIL/ERROR/TIMEOUT")
    parser.add_argument("--allow-all-skipped", action="store_true",
                        help="return 0 when every selected case skipped")
    parser.add_argument("--verbose", action="store_true",
                        help="write metrics and failed gates to stderr")
    return parser


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Runner overrides from the parsed flags (``None`` means "not set")."""
    values = {
        "threads": args.threads,
        "seed": args.seed,
        "mass": args.mass,
        "kappa": args.kappa,
        "csw": args.csw,
        "gauge": args.gauge,
        "gauge_sigma": args.gauge_sigma,
        "source": args.source,
        "dslash": args.dslash,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "restart": args.restart,
        "tolerance": args.tolerance,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "device": args.device,
        "device_id": args.device_id,
    }
    return {key: value for key, value in values.items() if value is not None}


def _device_spelling_error(spec: Any) -> Optional[str]:
    """Validate a ``--device`` spelling without importing torch."""
    text = str(spec).strip().lower()
    if not text:
        return None
    base, _, index = text.partition(":")
    if base not in _DEVICE_ALIASES:
        return (f"--device must be one of {', '.join(sorted(_DEVICE_ALIASES))} "
                f"or <kind>:<index>, got {spec!r}")
    if index and (not index.isdigit() or int(index) < 0):
        return f"--device index must be a non-negative integer, got {spec!r}"
    return None


def _validate_flags(args: argparse.Namespace) -> Optional[str]:
    """Validate the flag values the runners would otherwise reject.

    A bad flag is a *usage* error (exit code 2, like ``python -m pyqcu.test``),
    not an ``ERROR`` case: the run never had a chance to mean anything.
    """
    checks = (
        ("--repeat", args.repeat, lambda value: value >= 1),
        ("--warmup", args.warmup, lambda value: value >= 1),
        ("--max-iter", args.max_iter, lambda value: value >= 1),
        ("--restart", args.restart, lambda value: value >= 1),
        ("--threads", args.threads, lambda value: value >= 1),
        ("--device-id", args.device_id, lambda value: value >= 0),
        ("--timeout", args.timeout, lambda value: value > 0.0),
        ("--atol", args.atol, lambda value: value > 0.0),
        ("--tolerance", args.tolerance, lambda value: value > 0.0),
        ("--gauge-sigma", args.gauge_sigma, lambda value: value >= 0.0),
        ("--csw", args.csw, lambda value: value >= 0.0),
        ("--kappa", args.kappa, lambda value: value > 0.0),
    )
    for name, value, predicate in checks:
        if value is None:
            continue
        try:
            ok = bool(predicate(value))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"{name} is out of range: {value!r}"
    if args.mass is not None:
        try:
            _kappa_from_mass(float(args.mass))
        except ValueError as error:
            return f"--mass: {error}"
    if args.device is not None:
        error = _device_spelling_error(args.device)
        if error is not None:
            return error
    for name, path in (("--gauge-file", args.gauge_file),
                       ("--source-file", args.source_file)):
        if path is not None and not Path(path).expanduser().is_file():
            return f"{name} {path!r} does not exist"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the standalone matrix harness; return a policy exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    usage = policy.EXIT_CODES["usage"]
    invalid = _validate_flags(args)
    if invalid is not None:
        print(f"pyqcu.test.torch_matrix: error: {invalid}", file=sys.stderr)
        return usage
    try:
        cases = select_cases(presets=args.preset, dtypes=args.dtype,
                            tags=args.only, gauge_file=args.gauge_file,
                            source_file=args.source_file)
    except ValueError as error:
        print(f"pyqcu.test.torch_matrix: error: {error}", file=sys.stderr)
        return usage
    if not cases:
        print("pyqcu.test.torch_matrix: error: the selection is empty; "
              "check --preset/--dtype/--only", file=sys.stderr)
        return usage

    if args.list:
        if args.json is not None:
            error = _emit(_listing_payload(cases), args.json)
            if error is not None:
                print(f"pyqcu.test.torch_matrix: error: {error}",
                      file=sys.stderr)
                return usage
            return policy.EXIT_CODES["pass"]
        for entry in cases:
            print(json.dumps(entry.as_dict(), sort_keys=True,
                             ensure_ascii=False))
        return policy.EXIT_CODES["pass"]

    overrides = _overrides(args)
    started = time.perf_counter()
    results = run_matrix(cases, timeout=args.timeout, atol=args.atol,
                         fail_fast=args.fail_fast, overrides=overrides)
    duration = time.perf_counter() - started
    for result in results:
        print(_format_result(result))
        if args.verbose:
            if result.failed:
                print(f"  failed: {result.reason}", file=sys.stderr)
            elif result.reason:
                print(f"  {result.status.lower()}: {result.reason}",
                      file=sys.stderr)
            if result.metrics:
                print("  metrics: " + json.dumps(
                    json_safe(result.metrics), sort_keys=True,
                    ensure_ascii=False)[:2000], file=sys.stderr)
    payload = matrix_payload(cases, results, presets=args.preset,
                             dtypes=args.dtype, duration_seconds=duration,
                             allow_all_skipped=args.allow_all_skipped)
    counts = payload["counts"]
    print(f"matrix: {payload['executed']}/{payload['selected']} cases, "
          f"{counts['PASS']} pass, {counts['FAIL']} fail, "
          f"{counts['SKIP']} skip, {counts['TIMEOUT']} timeout, "
          f"{counts['ERROR']} error in {duration:.1f}s -> "
          f"{payload['status']}")
    error = _emit(payload, args.json)
    if error is not None:
        print(f"pyqcu.test.torch_matrix: error: {error}", file=sys.stderr)
        return usage
    return int(payload["exit_code"])


if __name__ == "__main__":  # pragma: no cover - manual harness
    raise SystemExit(main())
