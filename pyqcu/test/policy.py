"""Frozen policy of the formal :mod:`pyqcu.test` port.

This module is the only place in the port that *decides* something: the
canonical lattice, dtype tolerances, residual gates, exit codes, the skip
vocabulary and the normalisation rules for the run configuration (grids,
solver/dslash selection, device and backend spellings).
:mod:`pyqcu.test.common` (execution), :mod:`pyqcu.test.units` (case registry),
``__main__`` (CLI) and ``selfcheck`` (contract checks) only *read* it, so
changing a gate or a spelling never requires touching a runner module.

Import rules
------------
Standard library only, and no ``pyqcu`` submodule import beyond the package
itself.  ``python -m pyqcu.test --list`` must work in an interpreter without
``torch``, ``numpy``, ``mpi4py`` or ``pytest``; every constant here is plain
data so that stays true.

Configuration spellings
-----------------------
The CLI accepts several spellings for the same physical value and normalises
them here, once:

* ``--lattice-size``/``--lattice``, ``--lattice-grid`` and ``--mg-grid`` accept
  ``8 8 8 16``, ``8,8,8,16`` and ``[8,8,8,16]`` (:func:`parse_int_list`);
* ``--dtype`` accepts ``c64``/``c128`` and the ``complex64``/``complex128``/
  ``float32``/``float64`` aliases (:func:`normalize_dtype`);
* ``--backend`` accepts the canonical backends plus ``cpu`` (torch+kupl),
  ``dcu`` (cuda) and ``npu`` (torch) (:func:`expand_backends`);
* ``--device`` accepts ``auto``/``cpu``/``gpu``/``dcu``/``npu`` and maps to the
  runner-facing device string (:func:`device_for_runner`).

Exit codes differ from ``pyqcu --selftest`` on purpose
-------------------------------------------------------
``pyqcu --selftest`` returns **2** for a numerical failure (it is a one-shot
install check).  This port distinguishes "the numbers are wrong" from "the
command is wrong", the same way ``python -m pyqcd.einsum.test`` does:

===========  ==========================================================
``0``        no ``FAIL`` / ``ERROR`` / ``TIMEOUT``
``1``        any ``FAIL`` / ``ERROR`` / ``TIMEOUT``, or every case SKIPped
``2``        usage or environment error (bad flag, unreadable ``--json``)
===========  ==========================================================
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "PACKAGE",
    "TEST_PACKAGE",
    "RUNNER_PREFIX",
    "QUICK_LATTICE",
    "DEFAULT_LATTICE",
    "DEFAULT_LATTICE_GRID",
    "DEFAULT_MG_GRID",
    "DEFAULT_GAUGE_SIGMA",
    "DEFAULT_CSW",
    "DEFAULT_MASS",
    "DEFAULT_KAPPA",
    "DEFAULT_SOURCE",
    "DEFAULT_GAUGE",
    "DEFAULT_SOLVER",
    "DEFAULT_DSLASH",
    "DEFAULT_DEVICE",
    "QUICK_DTYPE",
    "QUICK_SEED",
    "DEFAULT_DTYPE",
    "DEFAULT_SEED",
    "DEFAULT_REPEAT",
    "QUICK_REPEAT",
    "DEFAULT_WARMUP",
    "DTYPE_CHOICES",
    "DTYPE_ALIASES",
    "DTYPE_INPUT_CHOICES",
    "DTYPES",
    "BACKENDS",
    "BACKEND_CHOICES",
    "BACKEND_ALIASES",
    "SOLVER_CHOICES",
    "SOLVER_TAG_FILTERS",
    "DSLASH_CHOICES",
    "DSLASH_TAG_FILTERS",
    "DEVICE_CHOICES",
    "DEVICE_RUNNER_DEVICES",
    "SOURCE_CHOICES",
    "GAUGE_CHOICES",
    "TIERS",
    "STATUSES",
    "EXIT_CODES",
    "DTYPE_TOLERANCES",
    "TOLERANCE_NAMES",
    "RESIDUAL_GATES",
    "GATE_GROUPS",
    "OBSERVATION_ALIASES",
    "SKIP_TOKENS",
    "SKIP_TOKEN_DESCRIPTIONS",
    "FORBIDDEN_IMPORT_ROOTS",
    "FORBIDDEN_IMPORT_PREFIXES",
    "TIER0_RUNNERS",
    "TIER0_TAGS",
    "quick_defaults",
    "validate_lattice",
    "validate_lattice_grid",
    "validate_mg_grid",
    "validate_dtype",
    "normalize_dtype",
    "validate_solver",
    "validate_dslash",
    "validate_device",
    "validate_source",
    "validate_gauge",
    "device_for_runner",
    "expand_backends",
    "solver_selects",
    "dslash_selects",
    "parse_int_list",
    "tolerance",
    "is_skip_token",
    "describe_skip",
    "exit_code",
]

#: Installed distribution and package names (``pyqcu.test`` ships in both the
#: sdist and the wheel; see ``setup.py`` ``find_packages``).
PACKAGE = "pyqcu"
TEST_PACKAGE = "pyqcu.test"

#: Relative module prefix used by ``Case.runner`` strings.  A runner string
#: without a dot in the module part (``torch_qcu:run_lattice``) is resolved
#: against this prefix; a dotted module part is treated as absolute.
RUNNER_PREFIX = TEST_PACKAGE

#: Project correctness lattice.  ``8x8x8x16`` is the end-to-end gate used by
#: ``pyqcu/testing`` and the release checks; it is small enough to run in
#: seconds and large enough that parity/EO and multigrid paths are non-trivial.
QUICK_LATTICE: tuple[int, int, int, int] = (8, 8, 8, 16)
DEFAULT_LATTICE: tuple[int, int, int, int] = QUICK_LATTICE

#: MPI rank grid (ranks per space-time dimension) and multigrid block grid.
#: A single rank is the default; the native runners skip with ``mpi_grid`` when
#: ``prod(lattice_grid)`` does not match the MPI size, so a torch run must not
#: fail on that mismatch.
DEFAULT_LATTICE_GRID: tuple[int, int, int, int] = (1, 1, 1, 1)
DEFAULT_MG_GRID: tuple[int, int, int, int] = (2, 2, 2, 2)

#: Gaussian width of the generated SU(3) background field (the historical
#: release setting) and the default clover coefficient.
DEFAULT_GAUGE_SIGMA = 0.1
DEFAULT_CSW = 1.0

#: ``mass``/``kappa`` have no default on purpose: the runners own the physical
#: default (``kappa = 1/(2m+8)``), and an unset flag must not be forwarded as
#: ``None`` (see ``Options.as_kwargs``).
DEFAULT_MASS: float | None = None
DEFAULT_KAPPA: float | None = None

#: ``--source`` and ``--gauge`` vocabularies.  ``point`` is the single
#: spin/colour source at the origin (``||b|| == 1``); ``ones`` is a constant
#: vector; ``random`` is a seeded Gaussian field.
SOURCE_CHOICES: tuple[str, str, str] = ("point", "ones", "random")
GAUGE_CHOICES: tuple[str, str, str] = ("random", "unit", "ones")
DEFAULT_SOURCE = "point"
DEFAULT_GAUGE = "random"

#: ``--quick`` locks the dtype and seed so two runs are byte-comparable.
QUICK_DTYPE = "c64"
QUICK_SEED = 42
DEFAULT_DTYPE = QUICK_DTYPE
DEFAULT_SEED = QUICK_SEED

#: Timing policy.  Warmup is never zero: the first call carries plan caches,
#: native library binding and JIT, which is exactly the cost this port must
#: not report as the steady state.
DEFAULT_REPEAT = 5
QUICK_REPEAT = 1
DEFAULT_WARMUP = 1

#: Canonical dtype names (the report/listing vocabulary).
DTYPE_CHOICES: tuple[str, str] = ("c64", "c128")
#: Accepted ``--dtype`` spellings -> canonical name.  ``float32``/``float64``
#: are the *element* precision of the same lattice fields, not a different
#: gate: both normalise onto the complex name, so ``--dtype float32`` is
#: ``c64`` and a ``--quick`` conflict still compares like with like.
DTYPE_ALIASES: dict[str, str] = {
    "c64": "c64",
    "complex64": "c64",
    "float32": "c64",
    "c128": "c128",
    "complex128": "c128",
    "float64": "c128",
}
#: Every spelling the CLI accepts (in help/completion order).
DTYPE_INPUT_CHOICES: tuple[str, ...] = tuple(DTYPE_ALIASES)
#: Backwards-compatible alias (the plan and older scripts say ``DTYPES``).
DTYPES = DTYPE_CHOICES

#: Backend order is part of the ``--list``/report output contract.  Only these
#: canonical names ever reach a filter or a report.
BACKENDS: tuple[str, str, str] = ("torch", "cuda", "kupl")
#: Backend spellings accepted by ``--backend``; the aliases expand to canonical
#: backends (:data:`BACKEND_ALIASES`) so a report never says ``cpu``.
BACKEND_CHOICES: tuple[str, ...] = BACKENDS + ("cpu", "dcu", "npu", "all")
BACKEND_ALIASES: dict[str, tuple[str, ...]] = {
    "cpu": ("torch", "kupl"),
    "dcu": ("cuda",),
    "npu": ("torch",),
    "all": BACKENDS,
}

#: ``--solver`` vocabulary.  The selection filters the *tag* matrix: a solver
#: choice never removes a case that is not a solver or multigrid check.
SOLVER_CHOICES: tuple[str, ...] = ("all", "bicgstab", "cg", "multigrid", "mg")
DEFAULT_SOLVER = "all"
#: Solver choice -> tag patterns.  ``solve/bistabcg`` matches
#: ``<backend>/solve/bistabcg`` (the historical ``bistabcg`` spelling of
#: BiCGStab); a trailing ``/`` marks a tag *segment* match (``mg``).  The
#: ``l48/*`` entries keep the tier-1 L48 two-node matrix selectable by the same
#: flag: its tags are ``<backend>/l48/bicgstab/<dtype>`` and
#: ``<backend>/l48/legacy-mg/<dtype>``.  There is deliberately no ``l48/cg``
#: pattern - that group is retired (see :mod:`pyqcu.test.l48`).
SOLVER_TAG_FILTERS: dict[str, tuple[str, ...]] = {
    "all": (),
    "bicgstab": ("solve/bistabcg", "l48/bicgstab"),
    "cg": ("solve/cg",),
    "multigrid": ("mg/", "l48/legacy-mg"),
    "mg": ("mg/", "l48/legacy-mg"),
}

#: ``--dslash`` vocabulary.  Only ``kind == "dslash"`` cases are filtered;
#: ``all`` and ``both`` are the "no filter" spellings (``both`` = Wilson and
#: clover, which is what ``all`` already means for the dslash cases).
DSLASH_CHOICES: tuple[str, str, str, str] = ("all", "wilson", "clover", "both")
DEFAULT_DSLASH = "all"
DSLASH_TAG_FILTERS: dict[str, tuple[str, ...]] = {
    "all": (),
    "both": (),
    "wilson": ("dslash/wilson",),
    "clover": ("dslash/clover",),
}

#: ``--device`` vocabulary and the runner-facing device each choice selects.
#: ``auto`` forwards nothing so the backend keeps its own default (CPU for
#: torch), which is what a machine without an accelerator must do.
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "gpu", "dcu", "npu")
DEFAULT_DEVICE = "auto"
DEVICE_RUNNER_DEVICES: dict[str, str | None] = {
    "auto": None,
    "cpu": "cpu",
    "gpu": "cuda:0",
    "dcu": "dcu:0",
    "npu": "npu:0",
}

#: Declared case tiers.  ``--tier N`` selects every case with ``tier <= N``.
TIERS: tuple[int, int] = (0, 1)

#: Result vocabulary.  ``SKIP`` is not a failure; ``ERROR`` is a harness or
#: runner defect, ``TIMEOUT`` a wall-clock overrun.
STATUSES: tuple[str, ...] = ("PASS", "FAIL", "SKIP", "TIMEOUT", "ERROR")

#: Exit-code policy; see the module docstring for the ``--selftest`` contrast.
EXIT_CODES: dict[str, int] = {"pass": 0, "fail": 1, "usage": 2}

#: Relative tolerances per dtype.  Every gate below is a *relative* number
#: (normalized by ``max|expected|`` or ``||b||``), never a per-element
#: absolute tolerance: absolute thresholds misjudge near-zero elements.
#:
#: ``compare``       operator/smearing equality against an independent path
#: ``solve``         solver true residual (full Wilson operator, ``||Dx-b||/||b||``)
#: ``native_solve``  solver true residual for a native (KUPL/CUDA) backend
#: ``gamma``         Dirac-algebra identity deviation (gamma_mu^2 vs 1)
DTYPE_TOLERANCES: dict[str, dict[str, float]] = {
    "c64": {
        "compare": 1e-5,
        "solve": 1e-3,
        "native_solve": 2e-4,
        "gamma": 1e-6,
    },
    "c128": {
        "compare": 1e-10,
        "solve": 1e-6,
        "native_solve": 1e-8,
        "gamma": 1e-12,
    },
}

#: Tolerance names an ``expect`` gate may reference as a string.
TOLERANCE_NAMES: tuple[str, str, str, str] = (
    "compare", "solve", "native_solve", "gamma")

#: Canonical gate groups.  ``units.py`` references these by name instead of
#: restating thresholds, so a tolerance change is a one-line policy edit.
#:
#: Every group is a mapping ``observation-key -> {operator: threshold}``:
#:
#: * numeric operators (``lt``/``le``/``gt``/``ge``, ``between``) resolve a
#:   string threshold through :func:`tolerance` for the run dtype;
#: * ``lt_field``/``le_field``/``gt_field``/``ge_field``/``eq_field`` compare
#:   against another observation of the same runner (``iterations`` against
#:   ``max_iter``);
#: * ``eq``/``ne``/``in``/``not_in``/``truthy``/``falsy``/``empty``/
#:   ``nonempty`` compare directly.
RESIDUAL_GATES: dict[str, dict[str, dict[str, Any]]] = {
    "lattice": {
        "su3": {"eq": True},
        "gamma_deviation": {"le": "gamma"},
    },
    "gauge": {
        # A generated background field must be a valid SU(3) configuration;
        # unitarity/determinant deviations are reported as observations, not
        # gates, because they are a property of the same check.
        "su3": {"eq": True},
    },
    "operator": {
        "relative_error": {"le": "compare"},
    },
    "smear": {
        "su3": {"eq": True},
        "relative_error": {"le": "compare"},
    },
    "solver": {
        "relative_residual": {"le": "solve"},
        "iterations": {"gt": 0},
    },
    "mg": {
        "converged": {"eq": True},
        "relative_residual": {"le": "solve"},
    },
    "abi": {
        # count == 40 declared C API entry points, and the whole set matches
        # (missing symbols/wrappers, header parity, ctypes signatures).
        "count": {"eq": 40},
        "set_match": {"eq": True},
    },
    "smoke": {
        "init_ok": {"eq": True},
    },
    "native_solver": {
        "relative_residual": {"le": "native_solve"},
        "iterations": {"gt": 0, "lt_field": "max_iter"},
    },
}

#: Alias kept for readers of the plan, which calls these "residual gates".
GATE_GROUPS = RESIDUAL_GATES

#: Accepted alternative spellings for an observation key.  The legacy
#: ``pyqcu/testing`` reports and the backend runners spell the same quantities
#: differently (``relative``, ``rel_res``, ``gamma_dev``); accepting those names
#: lets a runner keep the established vocabulary instead of duplicating values,
#: and does not weaken a gate (the aliases are exact synonyms, never a
#: different measurement).  The canonical names are the ones ``units.py``
#: documents.
OBSERVATION_ALIASES: dict[str, tuple[str, ...]] = {
    "relative_error": ("relative",),
    "relative_residual": ("rel_res", "true_residual", "relative_error"),
    "gamma_deviation": ("gamma_dev",),
}

#: Skip vocabulary.  A skip always carries one of these tokens plus a non-empty
#: human reason; an empty reason is an ``ERROR``, never a silent skip.
#:
#: ``mpi_grid``         ``prod(--lattice-grid)`` != MPI size (native runners)
#: ``device_unavailable`` the requested ``--device`` is not usable here
SKIP_TOKENS: tuple[str, ...] = (
    "kupl_library",
    "cuda_bridge",
    "cuda_device",
    "mpi_single_rank",
    "h5py_mpi",
    "mpi_grid",
    "device_unavailable",
)

SKIP_TOKEN_DESCRIPTIONS: dict[str, str] = {
    "kupl_library": "KUPL ctypes binding or libqcu.so is absent",
    "cuda_bridge": "pyqcu.cuda.qcu (Cython bridge) is not built",
    "cuda_device": "torch.cuda.is_available() is False",
    "mpi_single_rank": "the case requires exactly one MPI rank",
    "h5py_mpi": "h5py was built without MPI support",
    "mpi_grid": "prod(lattice_grid) does not match the MPI size",
    "device_unavailable": "the requested --device is not available here",
}

#: Import isolation enforced by ``selfcheck``: importing the port must not pull
#: any of these top-level modules, nor the informal ``pyqcu.testing`` tree.
FORBIDDEN_IMPORT_ROOTS: tuple[str, str, str, str] = (
    "torch", "numpy", "mpi4py", "pytest")
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = ("pyqcu.testing",)

#: Canonical tier-0 case list: tag -> ``"module:function"`` runner string.
#: ``units.py`` restates the same pairs literally (it is the case registry);
#: ``selfcheck`` asserts the two stay identical.  The duplication is a real
#: check, not an oversight: a renamed runner must fail the self-check.
TIER0_RUNNERS: dict[str, str] = {
    "torch/lattice": "torch_qcu:run_lattice",
    "torch/dslash/wilson": "torch_qcu:run_dslash_wilson",
    "torch/dslash/parity": "torch_qcu:run_dslash_parity",
    "torch/dslash/clover": "torch_qcu:run_dslash_clover",
    "torch/smear/stout": "torch_qcu:run_smear_stout",
    "torch/solve/bistabcg": "torch_qcu:run_solve_bistabcg",
    "torch/mg/legacy": "torch_qcu:run_mg_legacy",
    "torch/mg/strict": "torch_qcu:run_mg_strict",
    "kupl/abi/symbols": "kupl_qcu:run_abi_symbols",
    "kupl/dslash/wilson": "kupl_qcu:run_dslash_wilson",
    "kupl/solve/bistabcg": "kupl_qcu:run_solve_bistabcg",
    "cuda/smoke/init_end": "cuda_qcu:run_smoke_init_end",
    "cuda/dslash/wilson": "cuda_qcu:run_dslash_wilson",
    "cuda/solve/bistabcg": "cuda_qcu:run_solve_bistabcg",
    "torch/gauge/random": "torch_qcu:run_gauge_random",
    "torch/smear/wuppertal": "torch_qcu:run_smear_wuppertal",
    "torch/solve/cg": "torch_qcu:run_solve_cg",
    "kupl/dslash/clover": "kupl_qcu:run_dslash_clover",
    "kupl/solve/cg": "kupl_qcu:run_solve_cg",
}

#: Tier-0 tags in declaration (and therefore ``--list``) order.
TIER0_TAGS: tuple[str, ...] = tuple(TIER0_RUNNERS)

#: A concrete device string (``cuda:3``, ``dcu:0``, ``npu``...) passes through
#: :func:`device_for_runner` unchanged; only the five vocabulary words are
#: rewritten.
_CONCRETE_DEVICE = re.compile(r"^(?:cpu|cuda|dcu|npu)(?::\d+)?$")


def quick_defaults() -> dict[str, Any]:
    """Return the locked ``--quick`` defaults.

    ``--quick`` is a *profile*, not a shorthand: it pins the tier, lattice,
    dtype and seed so a quick run is comparable across machines and commits.
    ``repeat`` stays 1 because ``--quick`` makes no performance claim.  The
    grid/device/solver flags are orthogonal to the profile and stay usable.
    """
    return {
        "tier": 0,
        "lattice": QUICK_LATTICE,
        "dtype": QUICK_DTYPE,
        "seed": QUICK_SEED,
        "repeat": QUICK_REPEAT,
        "warmup": DEFAULT_WARMUP,
    }


# ---------------------------------------------------------------------------
# parsing and validation
# ---------------------------------------------------------------------------
def _split_tokens(values: Any) -> list[Any]:
    """Flatten ``"[8,8,8,16]"``, ``"8,8,8,16"``, ``"8 8 8 16"`` and sequences.

    One tokenisation for every list-valued flag: the CLI may hand in a string
    (one argv word), a list of argv words, or a Python sequence from the API.
    Brackets, parentheses, commas and whitespace are separators; a value that
    is not a string (an ``int`` from the API) is kept as is.
    """
    if isinstance(values, str):
        raw: list[Any] = [values]
    elif isinstance(values, Iterable):
        raw = list(values)
    else:
        raw = [values]
    tokens: list[Any] = []
    for item in raw:
        if not isinstance(item, str):
            tokens.append(item)
            continue
        for word in item.split():
            cleaned = word.strip().strip("[]()")
            if not cleaned:
                continue
            tokens.extend(part.strip() for part in cleaned.split(",")
                          if part.strip())
    return tokens


def parse_int_list(values: Any) -> tuple[int, ...]:
    """Parse an integer list from any accepted spelling.

    Accepts ``[8,8,8,16]``, ``8,8,8,16``, ``8 8 8 16``, the equivalent argv
    list and a Python sequence of ints/strings.  Raises ``ValueError`` with the
    offending token on anything that is not an integer.
    """
    tokens = _split_tokens(values)
    if not tokens:
        raise ValueError(f"expected a list of integers, got {values!r}")
    parsed: list[int] = []
    for token in tokens:
        if isinstance(token, bool):
            raise ValueError(f"expected integers, got {token!r}")
        if isinstance(token, str):
            text = token.strip()
            if not text.lstrip("+-").isdigit():
                raise ValueError(f"expected integers, got {token!r}")
            parsed.append(int(text))
            continue
        if isinstance(token, float) and not float(token).is_integer():
            raise ValueError(f"expected integers, got {token!r}")
        try:
            parsed.append(int(token))
        except (TypeError, ValueError) as error:
            raise ValueError(f"expected integers, got {token!r}") from error
    return tuple(parsed)


def _validate_grid(values: Any, name: str) -> tuple[int, int, int, int]:
    """Validate a four-dimensional rank/block grid (positive integers only)."""
    try:
        grid = parse_int_list(values)
    except ValueError as error:
        raise ValueError(f"{name} must be four positive integers: "
                         f"{error}") from error
    if len(grid) != 4:
        raise ValueError(
            f"{name} must be four positive integers (x y z t), got "
            f"{len(grid)}: {grid!r}")
    if any(extent <= 0 for extent in grid):
        raise ValueError(f"{name} extents must be positive: {grid!r}")
    return grid  # type: ignore[return-value]


def validate_lattice_grid(values: Any) -> tuple[int, int, int, int]:
    """Validate ``--lattice-grid`` (MPI ranks per dimension).

    Only "four positive integers" is enforced.  ``prod(grid) != mpi_size`` is
    deliberately **not** an error: the torch path ignores the grid, and the
    native runners report ``SKIP(mpi_grid)`` with a reason instead of failing a
    run that simply is not MPI-shaped.
    """
    return _validate_grid(values, "lattice_grid")


def validate_mg_grid(values: Any) -> tuple[int, int, int, int]:
    """Validate ``--mg-grid`` (multigrid block size per dimension)."""
    return _validate_grid(values, "mg_grid")


def validate_lattice(values: Iterable[Any]) -> tuple[int, int, int, int]:
    """Normalize and validate a lattice shape.

    Accepts the same spellings as ``--lattice-size`` (``[8,8,8,16]``,
    ``8,8,8,16``, four separate values or a Python sequence), so the API and
    the CLI cannot disagree about what a lattice is.  Requires four positive
    **even** extents of at least 4: the parity/EO decomposition and the
    checkerboard clover term need an even volume, and a smaller extent
    degenerates the Dirac operator on every backend.
    """
    try:
        shape = parse_int_list(values)
    except ValueError as error:
        raise ValueError(
            f"lattice must be four integers, got {values!r}") from error
    if len(shape) != 4:
        raise ValueError(
            f"lattice must be four integers (x y z t), got {len(shape)}: "
            f"{shape!r}")
    if any(extent <= 0 for extent in shape):
        raise ValueError(f"lattice extents must be positive: {shape!r}")
    if any(extent % 2 for extent in shape):
        raise ValueError(
            f"lattice extents must be even (parity decomposition): {shape!r}")
    if any(extent < 4 for extent in shape):
        raise ValueError(f"lattice extents must be >= 4: {shape!r}")
    return shape  # type: ignore[return-value]


def normalize_dtype(value: Any) -> str:
    """Map any accepted ``--dtype`` spelling onto ``c64``/``c128``."""
    text = str(value).strip().lower()
    if text not in DTYPE_ALIASES:
        raise ValueError(
            f"dtype must be one of {', '.join(DTYPE_INPUT_CHOICES)}, "
            f"got {value!r}")
    return DTYPE_ALIASES[text]


def validate_dtype(dtype: str) -> str:
    """Return the canonical dtype name for ``dtype`` (aliases accepted)."""
    return normalize_dtype(dtype)


def _validate_choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    """Return ``value`` lower-cased when it is a declared choice."""
    text = str(value).strip().lower()
    if text not in choices:
        raise ValueError(
            f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return text


def validate_solver(value: Any) -> str:
    """Return a declared ``--solver`` choice."""
    return _validate_choice("solver", value, SOLVER_CHOICES)


def validate_dslash(value: Any) -> str:
    """Return a declared ``--dslash`` choice."""
    return _validate_choice("dslash", value, DSLASH_CHOICES)


def validate_device(value: Any) -> str:
    """Return a declared ``--device`` choice (``auto``/``cpu``/``gpu``/...)."""
    return _validate_choice("device", value, DEVICE_CHOICES)


def validate_source(value: Any) -> str:
    """Return a declared ``--source`` choice."""
    return _validate_choice("source", value, SOURCE_CHOICES)


def validate_gauge(value: Any) -> str:
    """Return a declared ``--gauge`` choice.

    The flag selects how the background field is built (``random`` seeded
    Gaussian, ``unit`` identity links, ``ones`` all-ones links).
    """
    return _validate_choice("gauge", value, GAUGE_CHOICES)


def device_for_runner(device: Any) -> str | None:
    """Map ``--device`` to the backend device string handed to a runner.

    ``auto`` (and ``None``) return ``None`` so nothing is forwarded and the
    backend keeps its own default - forwarding the literal ``"auto"`` would be
    parsed by ``torch.device()`` and fail.  A concrete string such as
    ``cuda:1`` passes through unchanged, so the API never has to spell the
    vocabulary to target a device.
    """
    if device is None:
        return None
    text = str(device).strip().lower()
    if not text:
        return None
    if text in DEVICE_RUNNER_DEVICES:
        return DEVICE_RUNNER_DEVICES[text]
    if _CONCRETE_DEVICE.match(text):
        return text
    raise ValueError(
        f"device must be one of {', '.join(DEVICE_CHOICES)} or a concrete "
        f"device string, got {device!r}")


def expand_backends(values: Any) -> tuple[str, ...]:
    """Normalize ``--backend`` onto canonical backend names.

    ``cpu`` expands to ``torch``+``kupl``, ``dcu`` to ``cuda`` and ``npu`` to
    ``torch``; ``all`` (like an empty list) means "no filter" and returns
    ``("all",)``.  The result never contains an alias, so a report or a
    ``--list --json`` payload always names a real backend.
    """
    tokens = _split_tokens(values)
    names = [str(token).strip().lower() for token in tokens]
    names = [name for name in names if name]
    if not names:
        raise ValueError("backend must not be empty")
    if "all" in names:
        return ("all",)
    expanded: set[str] = set()
    unknown: set[str] = set()
    for name in names:
        if name in BACKENDS:
            expanded.add(name)
        elif name in BACKEND_ALIASES:
            expanded.update(BACKEND_ALIASES[name])
        else:
            unknown.add(name)
    if unknown:
        raise ValueError(
            f"unknown backend(s): {', '.join(sorted(unknown))}; choose from "
            f"{', '.join(BACKEND_CHOICES)}")
    return tuple(name for name in BACKENDS if name in expanded)


def _tag_matches_patterns(tag: str, patterns: tuple[str, ...]) -> bool:
    """True when ``tag`` matches one filter pattern.

    A pattern without a trailing ``/`` is a tag suffix (``solve/cg`` matches
    ``torch/solve/cg``); a pattern with one is a tag segment (``mg/`` matches
    every ``*/mg/*`` tag); a pattern with an inner ``/`` also matches a
    contiguous run of tag segments (``l48/bicgstab`` matches
    ``kupl/l48/bicgstab/c64``).
    """
    segments = tag.split("/")
    for pattern in patterns:
        if pattern.endswith("/"):
            if pattern.rstrip("/") in segments:
                return True
        elif tag == pattern or tag.endswith("/" + pattern):
            return True
        elif f"/{pattern}/" in f"/{tag}/":
            return True
    return False


def _is_solve_like(kind: str, tag: str) -> bool:
    """True for solver and multigrid cases (the ones ``--solver`` filters)."""
    segments = tag.split("/")
    return kind in ("solve", "mg") or "solve" in segments or "mg" in segments


def solver_selects(kind: str, tag: str, solver: Any) -> bool:
    """True when the ``--solver`` choice keeps a case.

    Only solver and multigrid cases are filtered; every other case stays, so
    ``--solver cg`` still runs the dslash/gauge/ABI checks.  ``bicgstab`` maps
    onto the historical ``solve/bistabcg`` tag, ``multigrid``/``mg`` onto the
    ``mg`` segment.
    """
    choice = validate_solver(solver)
    patterns = SOLVER_TAG_FILTERS[choice]
    if not patterns or not _is_solve_like(kind, tag):
        return True
    return _tag_matches_patterns(tag, patterns)


def dslash_selects(kind: str, tag: str, dslash: Any) -> bool:
    """True when the ``--dslash`` choice keeps a case.

    Only ``kind == "dslash"`` cases are filtered; ``all`` and ``both`` keep
    everything (the dslash matrix already consists of the Wilson and clover
    checks plus the parity decomposition).
    """
    choice = validate_dslash(dslash)
    patterns = DSLASH_TAG_FILTERS[choice]
    if not patterns or kind != "dslash":
        return True
    return _tag_matches_patterns(tag, patterns)


def tolerance(dtype: str, name: str, atol: float | None = None) -> float:
    """Resolve a named relative tolerance for ``dtype``.

    ``atol`` overrides the generic ``compare`` tolerance only.  Residual gates
    (``solve``/``native_solve``) and the gamma identity keep their dtype
    values: a user flag must not be able to silently weaken "the solver
    actually converged", which is the whole point of the port.
    """
    canonical = validate_dtype(dtype)
    table = DTYPE_TOLERANCES[canonical]
    if name not in table:
        raise KeyError(
            f"unknown tolerance {name!r} for dtype {canonical!r}; "
            f"choose from {', '.join(sorted(table))}")
    if atol is not None and name == "compare":
        return float(atol)
    return float(table[name])


def is_skip_token(token: Any) -> bool:
    """True when ``token`` is part of the skip vocabulary."""
    return token in SKIP_TOKENS


def describe_skip(token: Any) -> str:
    """Human description of a skip token (empty string when unknown)."""
    return SKIP_TOKEN_DESCRIPTIONS.get(str(token), "")


def exit_code(status: str, *, all_skipped: bool = False,
              allow_all_skipped: bool = False) -> int:
    """Map an aggregate status to the process exit code.

    ``SKIP`` as the aggregate status means *nothing ran* (every case skipped,
    or no case selected).  That is a failure by default: a suite that tested
    nothing must not look green.  ``--allow-all-skipped`` is the explicit
    opt-out.  ``all_skipped`` is accepted for callers that already know the
    count and is treated the same as ``status == "SKIP"``.
    """
    if status in ("FAIL", "ERROR", "TIMEOUT"):
        return EXIT_CODES["fail"]
    if status == "SKIP" or all_skipped:
        return (EXIT_CODES["pass"] if allow_all_skipped
                else EXIT_CODES["fail"])
    return EXIT_CODES["pass"]
