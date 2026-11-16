"""Torch backend runners for the formal :mod:`pyqcu.test` port.

This module is the *observation layer* for the torch cases.  It never decides
``PASS``/``FAIL``: every ``run_*`` function returns a plain ``dict`` of
measurements (JSON-safe Python scalars) and the ``expect`` mapping owned by
``pyqcu/test/units.py`` (A1) performs all gating.  Runner exceptions are never
caught here on purpose -- A1 maps them to ``ERROR``/``TIMEOUT``.

Contract
--------
* Importing this module is **stdlib-only**.  ``torch``, ``pyqcu.lattice``,
  ``pyqcu.torch.dslash``, ``pyqcu.torch.smear`` and ``pyqcu.torch.solver`` are
  imported lazily inside the runners, so ``python -m pyqcu.test --list`` stays
  light even when this module is imported.
* Every runner accepts ``*args, **kwargs`` and is forgiving about how the case
  description is handed in.  Any of these work::

      run_lattice()
      run_lattice(lattice=(8, 8, 8, 16), dtype="c64", seed=42, mass=0.1)
      run_lattice(case)                      # object with .lattice/.dtype/.seed
      run_lattice((8, 8, 8, 16), "c128", 7)  # positional lattice/dtype/seed

* Defaults: lattice ``(8, 8, 8, 16)``, ``complex64`` (``"c64"``), seed ``42``,
  device ``cpu``.  The device is always requested explicitly and probed, so a
  runner never silently falls back to CPU and never fakes a device.
* Every returned dict carries ``case``, ``backend="torch"``, ``device``,
  ``device_kind``, ``lattice``, ``lattice_grid``, ``process_grid``, ``dtype``,
  ``seed``, ``mass``, ``kappa``, ``csw``, ``gauge``, ``gauge_sigma``,
  ``source``, ``dslash``, ``solver``, ``mg_grid``, ``mg_levels``,
  ``mg_nu_pre``, ``mg_nu_post``, ``max_iter``, ``restart``, ``threads``,
  ``torch_version``, ``cuda_available``, ``option_sources`` and
  ``wall_seconds``.
* Solvers report an **independently recomputed** ``||D x - b|| / ||b||`` as
  ``rel_res``; the solver's own recurrence is never trusted for that gate.
  ``converged`` is defined by that independent residual against the documented
  ``tolerance`` (not by the solver's internal stop flag, which is reported
  separately as ``native_stop_reason``/``native_converged`` where available).

Primary metric keys consumed by ``units.CASES[*].expect``
---------------------------------------------------------
============================  ==================================================
runner                        keys
============================  ==================================================
``run_lattice``               ``su3``, ``gamma_deviation``
``run_gauge_random``          ``su3``, ``relative_error`` (+ ``distinct_relative``,
                              ``sigma_relative``: must be non-zero, they prove the
                              seed/sigma options act)
``run_dslash_wilson``         ``relative_error`` (+ ``relative_dense``: the
                              same parity identity on a dense seeded source)
``run_dslash_parity``         ``relative_error`` (+ ``relative_e``/``relative_o``)
``run_dslash_clover``         ``relative_error`` (+ ``inverse_relative``)
``run_smear_stout``           ``su3``, ``relative_error``
``run_smear_wuppertal``       ``su3``, ``relative_error`` (+ ``norm_ratio``,
                              ``norm_bound``, ``norm_ok``)
``run_solve_bistabcg``        ``relative_residual``, ``iterations``, ``max_iter``
``run_solve_cg``              ``relative_residual``, ``iterations``, ``max_iter``
``run_mg_legacy``             ``variant="legacy"``, ``converged``,
                              ``relative_residual``
``run_mg_strict``             ``variant="strict"``, ``converged``,
                              ``relative_residual``
============================  ==================================================

``wall_seconds`` is always present.  Extra keys are observations, not gates.
The canonical :mod:`pyqcu.test.policy` spellings are emitted **directly**
(``gamma_deviation``, ``relative_error``, ``relative_residual``), next to the
short aliases the legacy ``pyqcu/testing`` reports use (``gamma_dev``,
``relative``, ``rel``, ``rel_res``), so an ``expect`` gate works with either
vocabulary and never depends on ``policy.OBSERVATION_ALIASES``.

Run options (all optional keyword arguments)
--------------------------------------------
========================  =====================================================
option                    meaning
========================  =====================================================
``lattice``/``lat_size``  four even extents, default ``(8, 8, 8, 16)``
``dtype``/``precision``   ``"c64"`` (default) or ``"c128"``
``seed``                  gauge/source seed, default ``42``
``device``                ``auto``/``cpu`` -> ``cpu``; ``gpu``/``cuda[:N]`` ->
                          ``cuda:N``; ``dcu[:N]`` -> ROCm ``cuda:N``;
                          ``npu[:N]`` -> ``npu:N``.  A device that cannot be
                          allocated is a ``Skip("device_unavailable: ...")``
                          (a declared policy skip token), never a CPU fallback
``device_id``             accelerator index when ``device`` carries none
``mass``                  Dirac mass; used only when ``kappa`` is absent
``kappa``                 hopping parameter; overrides ``mass``
``csw``                   clover coefficient, default ``1.0``
``gauge``                 ``random`` (default), ``unit`` or ``ones``
``gauge_sigma``           ``generate_gauge_field`` sigma, default ``0.1``
``source``                ``point``/``ones``/``random``; each runner has its
                          own default (``random`` for dslash/smear,
                          ``point`` for solves)
``dslash``                policy vocabulary: ``all``/``both`` (no filter: the
                          runner keeps its own operator, ``clover`` for the MG
                          runners and ``wilson`` for the solves), ``wilson`` or
                          ``clover`` to force one
``solver``                policy vocabulary ``all``/``bicgstab``/``cg``/
                          ``multigrid``/``mg`` for ``run_solve_cg``: ``cg`` and
                          ``all`` use PyQCU's CA-CG, ``bicgstab`` the BiCGStab
                          implementation, ``multigrid``/``mg`` keep the CG
                          default (the CLI filters those cases out anyway).
                          The concrete ``cacg``/``bistabcg``/``mr``/``fgmres``
                          entry points are accepted too.  ``run_solve_bistabcg``
                          always runs BiCGStab (the tag names the algorithm) and
                          reports the request as ``solver_requested``
``mg_grid``               MG block size, default ``(2, 2, 2, 2)``
``mg_levels``/``max_level``  MG hierarchy depth
``mg_nu_pre``/``mg_nu_post``  smoother counts (``StrictMultigrid`` only)
``max_iter``              iteration budget (per runner default)
``restart``/``num_restart``  restart length (MG strict / FGMRES / legacy)
``threads``               ``torch.set_num_threads`` for the run
``lattice_grid``          MPI process grid; **recorded, not applied**: torch
                          runners never domain-decompose, so the effective grid
                          stays the launcher's (usually ``1 1 1 1``) and
                          ``lattice_grid_applied`` says whether they agree
``capture_stdout``        buffer native prints into ``log_tail`` (default on)
``kappa``/``csw``/...     additionally ``tolerance``/``tol``, ``solver_tolerance``,
                          ``if_rtol``, ``repeat``/``warmup``, ``nstep``, ``rho``,
                          ``n_krylov``, ``omega``, ``su3_tol``, ``gamma_tol``,
                          ``with_clover``, ``nvec``, ``support_parity``,
                          ``materialize_coarse``, ``min_size``
========================  =====================================================

Precedence: explicit keyword argument > positional mapping / case attribute >
documented default.  ``common.Options.as_kwargs`` forwards the fields that have
a policy default and omits the rest, so an unset ``mass``/``max_iter``...
reaches a runner as "absent" and the runner's own default applies.  Every value
that ends up in effect is reported in the observations
(``kappa``/``mass``/``csw``/``gauge``/``source``/``dslash``/``solver``/...), and
``option_sources`` says whether it came from an argument or from a default, so
nothing is hidden.

Physical conventions
--------------------
``mass`` and ``kappa`` are the same knob: ``kappa = 1 / (2 * mass + 8)``.
``mass=0`` gives the historical ``kappa=0.125``.  A degenerate ``2m+8 == 0`` is
rejected instead of producing an infinite hopping parameter.

``run_solve_cg`` solves the **even-odd Schur complement** (QUDA ``MATPC``)
system ``S_o x_o = b_o - D_oe D_ee^-1 b_e`` with the CG family from
``pyqcu.torch.solver``.  Plain CG needs a Hermitian positive-definite operator
and the Wilson operator is only gamma5-Hermitian, so the full system is not a
valid CG target; the Schur complement is.  The reported ``rel_res`` is still
the physical ``||D x - b|| / ||b||`` of the reconstructed full field.
"""

from __future__ import annotations

import contextlib
import io
import math
import statistics
import sys
import time
import types
from collections.abc import Mapping as _Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import policy
from .common import Skip, time_samples

__all__ = [
    "DEFAULT_CSW",
    "DEFAULT_DTYPE",
    "DEFAULT_GAUGE",
    "DEFAULT_GAUGE_SIGMA",
    "DEFAULT_KAPPA",
    "DEFAULT_MG_GRID",
    "DEFAULT_SEED",
    "DSLASH_KINDS",
    "GAUGE_KINDS",
    "QUICK_LATTICE",
    "RUNNERS",
    "SOLVER_KINDS",
    "SOURCE_KINDS",
    "TAG_RUNNERS",
    "probe_env",
    "run_dslash_clover",
    "run_dslash_parity",
    "run_dslash_wilson",
    "run_gauge_random",
    "run_lattice",
    "run_mg_legacy",
    "run_mg_strict",
    "run_smear_stout",
    "run_smear_wuppertal",
    "run_solve_bistabcg",
    "run_solve_cg",
]

#: Project correctness lattice (``pyqcu/testing/_config.py``, ``AGENTS.md``).
QUICK_LATTICE: Tuple[int, int, int, int] = (8, 8, 8, 16)
#: Default precision tag used by the ``--quick`` port.
DEFAULT_DTYPE = "c64"
#: Fixed seed so every observation is reproducible.
DEFAULT_SEED = 42
#: ``kappa = 1 / (2 * mass + 8)`` at ``mass = 0`` -- the value used by
#: ``pyqcu/_selftest.py`` and the informal torch suite.
DEFAULT_KAPPA = 0.125
#: Clover coefficient used when the caller does not ask for another one.
DEFAULT_CSW = 1.0
#: ``generate_gauge_field`` sigma used by the informal suites.
DEFAULT_GAUGE_SIGMA = 0.1
#: Background gauge field kind: random SU(3) (default), exact unit links, or an
#: all-ones (deliberately non-unitary) tensor used to exercise code paths.
DEFAULT_GAUGE = policy.DEFAULT_GAUGE
#: MG block size (``mg_grid_size``/``block_size``).
DEFAULT_MG_GRID: Tuple[int, int, int, int] = tuple(policy.DEFAULT_MG_GRID)

#: The vocabularies are :mod:`pyqcu.test.policy`'s: the CLI validates a flag
#: against policy before a runner ever sees it, and a runner that accepted a
#: different spelling would fail on the value the port forwards by default.
GAUGE_KINDS: Tuple[str, ...] = tuple(policy.GAUGE_CHOICES)
SOURCE_KINDS: Tuple[str, ...] = tuple(policy.SOURCE_CHOICES)
#: ``all``/``both`` are the "no filter" spellings: the runner keeps its own
#: operator.  Only ``wilson``/``clover`` express a preference.
DSLASH_KINDS: Tuple[str, ...] = tuple(policy.DSLASH_CHOICES)
#: ``policy.SOLVER_CHOICES`` plus the concrete ``pyqcu.torch.solver`` entry
#: points a direct caller may name: ``cg``/``cacg`` both select
#: ``solver.cacg`` (PyQCU has no plain CG entry point; its CG-family solver is
#: the s-step CA-CG), and ``bistabcg`` is the solver's own historical spelling
#: of ``bicgstab``.
SOLVER_KINDS: Tuple[str, ...] = tuple(policy.SOLVER_CHOICES) + (
    "cacg", "bistabcg", "mr", "fgmres")

_DTYPES = {
    "c64": ("c64", "complex64"),
    "c128": ("c128", "complex128"),
    "complex64": ("c64", "complex64"),
    "complex128": ("c128", "complex128"),
    "single": ("c64", "complex64"),
    "double": ("c128", "complex128"),
}

_LATTICE_KEYS = ("lattice", "lat_size", "lattice_shape", "shape", "lat",
                 "lat_size_xyzt")
_DTYPE_KEYS = ("dtype", "precision")
_SEED_KEYS = ("seed", "rng_seed")
_DEVICE_KEYS = ("device", "dev")
_CAPTURE_KEYS = ("capture_stdout", "capture")

#: Every option ``_Config`` resolves; used for the ``option_sources`` report.
_OPTION_NAMES: Tuple[str, ...] = (
    "csw", "device", "device_id", "dslash", "gauge", "gauge_sigma", "kappa",
    "lattice_grid", "mass", "max_iter", "mg_grid", "mg_levels",
    "mg_nu_post", "mg_nu_pre", "restart", "solver", "source", "threads",
)

#: Device aliases: option spelling -> (torch device type, reported kind).
_DEVICE_ALIASES = {
    "auto": ("cpu", "cpu"),
    "cpu": ("cpu", "cpu"),
    "cuda": ("cuda", "gpu"),
    "dcu": ("cuda", "dcu"),
    "gpu": ("cuda", "gpu"),
    "hip": ("cuda", "dcu"),
    "npu": ("npu", "npu"),
    "rocm": ("cuda", "dcu"),
}

#: ``pyqcu.torch.solver`` entry points addressable through ``solver``.
_KRYLOV_ENTRY_POINTS = {
    "bistabcg": "bistabcg",
    "cacg": "cacg",
    "cg": "cacg",
    "fgmres": "fgmres",
    "mr": "mr",
}

#: ``policy.SOLVER_CHOICES`` value -> the concrete implementation it selects.
_SOLVER_SELECTION = {
    "all": "cg",
    "cacg": "cg",
    "cg": "cg",
    "bicgstab": "bistabcg",
    "bistabcg": "bistabcg",
    "fgmres": "fgmres",
    "mg": "cg",
    "mr": "mr",
    "multigrid": "cg",
}

#: ``policy.DSLASH_CHOICES`` value -> the operator the runner should build.
#: ``None`` means "the runner's own default".
_DSLASH_SELECTION = {
    None: None,
    "all": None,
    "both": None,
    "clover": "clover",
    "wilson": "wilson",
}

_MODULES: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# lazy imports
# ---------------------------------------------------------------------------
def _load() -> types.SimpleNamespace:
    """Import the torch stack on first use and cache it.

    Kept out of module scope so ``pyqcu.test`` remains stdlib-only at import
    time (``--list`` must not pull torch in).
    """
    if "torch" not in _MODULES:
        import torch
        from pyqcu import lattice, tools
        from pyqcu.torch import dslash, smear, solver

        _MODULES.update(
            torch=torch,
            lattice=lattice,
            tools=tools,
            dslash=dslash,
            smear=smear,
            solver=solver,
        )
    return types.SimpleNamespace(**_MODULES)


def probe_env() -> Dict[str, Any]:
    """Return the torch/MPI environment as observations (no gating).

    Used for the report ``env`` block and for A1's human-readable summary.  The
    torch cases themselves never require CUDA; ``cuda_available`` is reported
    only so a reader knows which machine produced the numbers.
    """
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": None,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_name": None,
        "torch_threads": None,
        "mpi_size": None,
        "mpi_rank": None,
    }
    try:
        torch = _load().torch
    except Exception:  # pragma: no cover - report-only helper, never gates
        return env
    env["torch"] = str(torch.__version__)
    env["torch_threads"] = int(torch.get_num_threads())
    env["torch_cuda_available"] = bool(torch.cuda.is_available())
    env["torch_cuda_device_count"] = int(torch.cuda.device_count())
    if env["torch_cuda_available"]:
        try:
            env["torch_cuda_device_name"] = str(torch.cuda.get_device_name(0))
        except Exception:  # pragma: no cover
            env["torch_cuda_device_name"] = None
    try:
        from mpi4py import MPI

        env["mpi_size"] = int(MPI.COMM_WORLD.Get_size())
        env["mpi_rank"] = int(MPI.COMM_WORLD.Get_rank())
    except Exception:  # pragma: no cover - mpi4py is optional here
        pass
    return env


# ---------------------------------------------------------------------------
# small numeric / config helpers
# ---------------------------------------------------------------------------
def _get(options: Mapping, name: str, default: Any,
         aliases: Sequence[str] = ()) -> Any:
    """Fetch ``name`` (or one of ``aliases``) from ``options``."""
    for key in (name, *aliases):
        if key in options and options[key] is not None:
            return options[key]
    return default


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


def _complex_dtype(spec: Any) -> Tuple[Any, str]:
    """Resolve ``"c64"``/``"c128"``/``torch.dtype`` to ``(dtype, tag)``."""
    torch = _load().torch
    if spec is None:
        return torch.complex64, "c64"
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "")
        if key in _DTYPES:
            tag, name = _DTYPES[key]
            return getattr(torch, name), tag
        raise ValueError(
            f"unsupported torch dtype tag {spec!r}; use 'c64' or 'c128'")
    if isinstance(spec, torch.dtype):
        for tag, name in _DTYPES.values():
            if spec is getattr(torch, name):
                return spec, tag
        raise ValueError(
            f"unsupported torch dtype {spec!r}; torch cases need complex64/complex128")
    raise TypeError(f"unsupported dtype specification {spec!r}")


def _lattice_shape(spec: Any) -> Tuple[int, int, int, int]:
    if spec is None:
        return QUICK_LATTICE
    values = tuple(int(value) for value in spec)
    if len(values) != 4:
        raise ValueError(f"lattice must have four extents, got {values}")
    if any(value < 4 or value % 2 for value in values):
        raise ValueError(
            "lattice extents must be even and >= 4 (parity splitting needs it), "
            f"got {values}")
    return values  # type: ignore[return-value]


def _four_extents(spec: Any, name: str, minimum: int = 1) -> Tuple[int, ...]:
    """Parse an ``x y z t`` option given as a sequence or a string.

    ``PYQCU_TEST_MG_GRID="2 2 2 2"``, ``"2,2,2,2"`` and ``"(2,2,2,2)"`` all mean
    the same four extents; the environment fallback can only deliver text, so
    the string form has to be accepted here.
    """
    if isinstance(spec, str):
        text = spec.strip().strip("()[]")
        parts = [part for part in text.replace(",", " ").split() if part]
        try:
            values = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                f"{name} must contain four integers, got {spec!r}") from error
    elif isinstance(spec, (tuple, list)):
        values = tuple(int(value) for value in spec)
    else:
        raise TypeError(
            f"{name} must be a sequence or a string of four integers, "
            f"got {spec!r}")
    if len(values) != 4:
        raise ValueError(f"{name} must have four extents (x y z t), got {values}")
    if any(value < minimum for value in values):
        raise ValueError(f"{name} extents must be >= {minimum}, got {values}")
    return values


def _lattice_grid(spec: Any,
                  lattice: Sequence[int]) -> Tuple[int, int, int, int]:
    """Validate the requested MPI process grid against the local extents."""
    values = _four_extents(spec, "lattice_grid", minimum=1)
    for extent, divisor in zip(lattice, values):
        if divisor > 1 and int(extent) % int(divisor):
            raise ValueError(
                f"lattice_grid {values} does not divide lattice "
                f"{tuple(int(value) for value in lattice)}")
    return values  # type: ignore[return-value]


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
    """Inverse of :func:`_kappa_from_mass`; used for reporting only."""
    return (1.0 / float(kappa) - 8.0) / 2.0


def _float_option(options: Mapping, name: str, default: Optional[float],
                  aliases: Sequence[str] = ()) -> Optional[float]:
    value = _get(options, name, default, aliases)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _int_option(options: Mapping, name: str, default: Optional[int],
                aliases: Sequence[str] = ()) -> Optional[int]:
    value = _get(options, name, default, aliases)
    return None if value is None else int(value)


def _choice(value: Any, name: str, allowed: Sequence[str]) -> Optional[str]:
    """Normalize an enumerated option; ``None`` stays ``None``."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return text


def _device(spec: Any, torch: Any,
            device_id: Optional[int] = None) -> Tuple[Any, str, str]:
    """Resolve ``device`` to ``(torch.device, kind, requested)``.

    ``auto``/``cpu`` -> ``cpu``; ``gpu``/``cuda[:N]`` -> ``cuda:N``; ``dcu`` is
    the ROCm backend of torch (``cuda:N``); ``npu[:N]`` is ``torch_npu``.
    ``device_id`` supplies the accelerator index when the device string carries
    none (``policy`` forwards ``--device dcu`` and ``--device-id 1``
    separately).  The device is allocated once here so an unusable request
    becomes a ``Skip("device_unavailable: ...")`` instead of a silent CPU
    fallback.
    """
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
        number = int(index) if index else (
            0 if device_id is None else int(device_id))
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
                    f"torch.cuda.is_available() is False")
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
    except Exception:  # pragma: no cover - report-only
        return None
    return None


@contextlib.contextmanager
def _preserve_rng(torch: Any, device: Any):
    """Restore torch's global RNG around a global-seed helper.

    ``lattice.generate_gauge_field`` seeds the *global* generator (it has no
    generator argument), so generating a background field would otherwise
    perturb every later random draw in the process.
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


class _Config:
    """Resolved call configuration shared by every runner."""

    __slots__ = (
        "lattice", "dtype", "dtype_tag", "seed", "device", "device_kind",
        "device_name", "device_requested", "device_id", "capture_stdout",
        "options", "mass", "kappa", "csw", "kappa_source",
        "gauge", "gauge_sigma", "source",
        "mg_grid", "mg_levels", "mg_nu_pre", "mg_nu_post", "max_iter",
        "restart", "dslash", "dslash_selected", "solver", "solver_selected",
        "threads", "threads_previous", "threads_applied", "threads_error",
        "lattice_grid", "process_grid", "process_grid_error",
        "lattice_grid_applied", "option_sources",
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
            if isinstance(item, _Mapping):
                merged = dict(item)
                explicit.update(merged)
                merged.update(options)
                options = merged
                continue
            if (isinstance(item, (tuple, list)) and len(item) == 4
                    and all(isinstance(value, int) and not isinstance(value, bool)
                            for value in item)):
                lattice_spec = item
                continue
            if isinstance(item, int) and not isinstance(item, bool):
                seed = item
                continue
            if isinstance(item, str):
                lowered = item.strip().lower()
                if lowered in _DTYPES:
                    dtype_spec = item
                else:
                    raise TypeError(
                        f"cannot interpret positional argument {item!r}")
                continue
            # Case-like object (``units.Case`` or an argparse Namespace).
            case_items.append(item)
            lattice_spec = getattr(item, "lattice", lattice_spec)
            dtype_spec = getattr(item, "dtype", dtype_spec)
            seed = getattr(item, "seed", seed)
            device_spec = getattr(item, "device", device_spec)

        # Options a *case object* (or a future ``common.Options`` extension)
        # carries as attributes; the frozen dataclass has no other channel.
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

        lattice_spec = _get(options, "lattice", lattice_spec, _LATTICE_KEYS[1:])
        dtype_spec = _get(options, "dtype", dtype_spec, _DTYPE_KEYS[1:])
        seed = _get(options, "seed", seed, _SEED_KEYS[1:])
        device_spec = _get(options, "device", device_spec, _DEVICE_KEYS[1:])
        capture = _get(options, "capture_stdout", True, _CAPTURE_KEYS[1:])

        self.lattice = _lattice_shape(lattice_spec)
        self.dtype, self.dtype_tag = _complex_dtype(dtype_spec)
        self.seed = DEFAULT_SEED if seed is None else int(seed)
        tools = _load().tools
        self.device_id = _int_option(options, "device_id", None, ("device_index",))
        if self.device_id is not None and self.device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {self.device_id!r}")
        self.device, self.device_kind, self.device_requested = _device(
            device_spec, torch, self.device_id)
        self.device_name = _device_name(torch, self.device)
        self.capture_stdout = bool(capture)
        self.options = options

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
        self.gauge = _choice(_get(options, "gauge", DEFAULT_GAUGE), "gauge",
                             GAUGE_KINDS) or DEFAULT_GAUGE
        sigma = _float_option(options, "gauge_sigma", DEFAULT_GAUGE_SIGMA)
        if sigma < 0.0:
            raise ValueError(f"gauge_sigma must be >= 0, got {sigma!r}")
        self.gauge_sigma = sigma
        self.source = _choice(_get(options, "source", None), "source",
                              SOURCE_KINDS)
        self.dslash = _choice(_get(options, "dslash", None), "dslash",
                              DSLASH_KINDS)
        self.dslash_selected = _DSLASH_SELECTION[self.dslash]
        self.solver = _choice(_get(options, "solver", None), "solver",
                              SOLVER_KINDS)
        self.solver_selected = _SOLVER_SELECTION[self.solver or "all"]

        mg_grid_spec = _get(options, "mg_grid", DEFAULT_MG_GRID)
        self.mg_grid = _four_extents(mg_grid_spec, "mg_grid", minimum=1)
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

        # ``lattice_grid`` is recorded, never applied: the torch runners do not
        # domain-decompose, so claiming a 2x2x2x1 grid would be a fake speedup.
        self.lattice_grid = _lattice_grid(
            _get(options, "lattice_grid", (1, 1, 1, 1)), self.lattice)
        self.process_grid = None
        self.process_grid_error = None
        try:
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
        if self.threads is not None and self.threads_previous != self.threads:
            try:
                torch.set_num_threads(int(self.threads))
                self.threads_applied = True
            except Exception as error:  # noqa: BLE001 - report, never gate
                self.threads_error = f"{type(error).__name__}: {error}"
        elif self.threads is not None:
            self.threads_applied = True

class _StdoutCapture:
    """Buffer ``print`` output so one case stays one line on stdout.

    The help/native layers (``LegacyMultigrid`` in particular) print
    unconditionally.  Their text is kept as ``log_tail`` observations; if the
    runner raises, the buffered text is re-emitted on stderr so nothing is lost
    and the exception still propagates unchanged.
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


def _base(config: _Config, torch: Any, case: str) -> Dict[str, Any]:
    return {
        "case": case,
        "backend": "torch",
        "device": str(config.device),
        "device_kind": config.device_kind,
        "device_type": str(config.device.type),
        "device_requested": config.device_requested,
        "device_id": config.device_id,
        "device_name": config.device_name,
        "lattice": list(config.lattice),
        "lattice_grid": list(config.lattice_grid),
        "process_grid": (list(config.process_grid)
                         if config.process_grid is not None else None),
        "process_grid_error": config.process_grid_error,
        "lattice_grid_applied": config.lattice_grid_applied,
        "decomposed": bool(config.process_grid is not None
                           and tuple(config.process_grid) != (1, 1, 1, 1)),
        "dtype": config.dtype_tag,
        "seed": config.seed,
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
        "source": config.source,
        "dslash": config.dslash_selected,
        "dslash_requested": config.dslash,
        "solver": config.solver_selected,
        "solver_requested": config.solver,
        "mg_grid": list(config.mg_grid),
        "mg_levels": config.mg_levels,
        "mg_nu_pre": config.mg_nu_pre,
        "mg_nu_post": config.mg_nu_post,
        "max_iter": config.max_iter,
        "restart": config.restart,
        "option_sources": dict(config.option_sources),
    }


def _finish(observations: Dict[str, Any], started: float) -> Dict[str, Any]:
    observations["wall_seconds"] = time.perf_counter() - started
    return observations


def _measure(config: "_Config",
             call: Callable[[], Any]) -> Tuple[Tuple[float, ...], Any]:
    """Time ``call`` with the case's ``warmup``/``repeat`` policy.

    :func:`pyqcu.test.common.time_samples` owns the policy (discard
    ``warmup >= 1`` calls, then collect ``repeat`` samples), so a runner never
    re-implements it.  The closure records the last return value: the reported
    observation then comes from the same deterministic execution as the timing
    sample instead of an extra, untimed run.
    """
    captured: Dict[str, Any] = {}

    def _observed() -> Any:
        value = call()
        captured["value"] = value
        return value

    samples = time_samples(
        _observed,
        repeat=int(_get(config.options, "repeat", 1, ("samples",)) or 1),
        warmup=int(_get(config.options, "warmup", 1, ("warmups",)) or 1),
    )
    return tuple(samples), captured.get("value")


def _timing(samples: Sequence[float]) -> Dict[str, Any]:
    """Timing observations: median (``seconds``), min, max and raw samples.

    ``Result.seconds`` prefers the runner-reported ``seconds``, so the port
    reports the median of the timed core operation rather than the wall time of
    setup plus one untimed call.  ``--quick`` makes no performance claim; the
    fields are diagnostics, and a median only means something for
    ``--repeat >= 5``.
    """
    values = [float(value) for value in samples]
    return {
        "seconds": float(statistics.median(values)),
        "seconds_min": float(min(values)),
        "seconds_max": float(max(values)),
        "samples": len(values),
        "timing_samples": [round(value, 6) for value in values],
    }


# ---------------------------------------------------------------------------
# field builders (all option-aware)
# ---------------------------------------------------------------------------
def _gauge_field(config: _Config, torch: Any, lattice: Any, *,
                 kind: Optional[str] = None, seed: Optional[int] = None,
                 sigma: Optional[float] = None) -> Any:
    """Generate the background gauge field requested by ``gauge``.

    ``random`` is the seeded ``lattice.generate_gauge_field`` exponential map
    with ``gauge_sigma``; ``unit`` is the exact identity link on every site;
    ``ones`` is a deliberately non-unitary all-ones tensor (a code path probe,
    never an SU(3) field).  ``seed``/``sigma``/``kind`` override the configured
    values for the reproducibility observations.
    """
    gauge_kind = config.gauge if kind is None else _choice(kind, "gauge",
                                                           GAUGE_KINDS)
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
            3, 3, 4, *config.lattice).contiguous()
    if gauge_kind == "ones":
        return torch.ones((3, 3, 4, *config.lattice), dtype=config.dtype,
                          device=config.device)
    raise ValueError(
        f"unsupported gauge {gauge_kind!r}; choose from {', '.join(GAUGE_KINDS)}")


def _dense_source(config: "_Config", torch: Any, seed: int) -> Any:
    """Seeded dense Gaussian field, independent of the ``source`` option.

    Used for the extra probe of a case whose configured source is sparse: a
    point/constant source would leave most sites untouched, and a stencil bug
    outside that small support would go unnoticed.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    real = torch.randn((4, 3, *config.lattice), generator=generator,
                       dtype=torch.float32)
    imag = torch.randn((4, 3, *config.lattice), generator=generator,
                       dtype=torch.float32)
    return torch.complex(real, imag).to(dtype=config.dtype,
                                        device=config.device)


def _fermion_field(config: _Config, torch: Any, seed: int,
                   default: str = "random") -> Any:
    """Build the case source ``[4, 3, x, y, z, t]`` honouring ``source``.

    ``random`` uses its own ``torch.Generator`` (never the global RNG);
    ``point`` is the spin-0/colour-0 delta at the origin (``||b|| == 1``);
    ``ones`` is the constant spinor/colour field.  ``default`` is the runner's
    own choice, so ``source`` only has to be given when it should differ.
    """
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
        f"unsupported source {kind!r}; choose from {', '.join(SOURCE_KINDS)}")


def _clover_term(config: _Config, dslash: Any, torch: Any, gauge: Any,
                 kappa: Any, csw: Optional[float] = None) -> Any:
    """Clover term at ``csw`` (the option value unless one is passed)."""
    del torch  # the core helper owns the tensor construction
    coefficient = config.csw if csw is None else float(csw)
    return dslash.make_clover(U=gauge, kappa=kappa, csw=coefficient,
                              verbose=False)


def _solve_dslash_kind(config: _Config, default: str = "wilson") -> str:
    """Operator kind for a solve: the ``dslash`` choice or the runner default.

    ``all``/``both`` (policy's "no filter" spellings) and an absent option mean
    "keep the runner's own operator"; only ``wilson``/``clover`` override it.
    """
    return config.dslash_selected or default


def _wilson_plus_clover_reference(config: _Config, modules: types.SimpleNamespace,
                                  gauge: Any, clover: Any, kappa: Any,
                                  vector: Any) -> Any:
    """Independent ``D x = (1 + T + hop) x`` reference from public helpers."""
    del config  # the reference is built from the public helpers only
    dslash = modules.dslash
    reference = dslash.give_wilson(src=vector, U=gauge, kappa=kappa,
                                   with_I=True, verbose=False)
    if clover is not None:
        reference = reference + dslash.give_clover(
            src=vector, clover_term=clover, verbose=False)
    return reference


def _su3_observations(torch: Any, lattice: Any, gauge: Any,
                      su3_tol: float) -> Dict[str, Any]:
    """SU(3) observations of a gauge field (unitarity, determinant, check)."""
    matrices = gauge.permute(*range(2, gauge.ndim), 0, 1).reshape(-1, 3, 3)
    eye3 = torch.eye(3, dtype=matrices.dtype,
                     device=matrices.device).expand_as(matrices)
    unitarity_dev = float((torch.matmul(
        matrices.conj().transpose(-1, -2), matrices) - eye3).abs().max())
    det_dev = float((torch.linalg.det(matrices) - 1.0).abs().max())
    return {
        "su3": bool(lattice.check_su3(gauge, tol=su3_tol, verbose=False)),
        "su3_tolerance": su3_tol,
        "unitarity_dev": unitarity_dev,
        "det_dev": det_dev,
    }


def _gamma_identity(torch: Any, lattice: Any,
                    config: "_Config") -> Tuple[float, float]:
    """Max ``|gamma_mu^2 - 1|`` and ``|gamma5^2 - 1|`` for the run dtype."""
    gamma = lattice.gamma.to(device=config.device).type(config.dtype)
    gamma5 = lattice.gamma_5.to(device=config.device).type(config.dtype)
    eye4 = torch.eye(4, dtype=config.dtype, device=config.device)
    deviation = max(
        float((torch.matmul(gamma[mu], gamma[mu]) - eye4).abs().max())
        for mu in range(4))
    fifth = float((torch.matmul(gamma5, gamma5) - eye4).abs().max())
    return deviation, fifth


# ---------------------------------------------------------------------------
# tier-0 runners
# ---------------------------------------------------------------------------
def run_lattice(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/lattice``: seeded SU(3) generation/check and gamma algebra.

    Observations: ``su3`` (``lattice.check_su3``), ``unitarity_dev``,
    ``det_dev``, ``gamma_dev`` (max ``|gamma_mu^2 - 1|``), ``gamma5_dev``
    (max ``|gamma5^2 - 1|``), ``gauge_norm``, and the ``gauge``/``gauge_sigma``
    that actually produced the field.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice = modules.torch, modules.lattice
    su3_tol = float(_get(config.options, "su3_tol", 1e-3, ("su3_tolerance",)))
    gamma_tol = float(_get(config.options, "gamma_tol", 1e-6, ()))

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        su3_observations = _su3_observations(torch, lattice, gauge, su3_tol)
        samples, (gamma_dev, gamma5_dev) = _measure(
            config, lambda: _gamma_identity(torch, lattice, config))

    observations = _base(config, torch, "torch/lattice")
    observations.update(_timing(samples))
    observations.update(su3_observations)
    observations.update(
        gamma_dev=gamma_dev,
        gamma_deviation=gamma_dev,
        gamma5_dev=gamma5_dev,
        gamma_tolerance=gamma_tol,
        gauge_norm=_norm(torch, gauge),
        gauge_shape=list(gauge.shape),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_gauge_random(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/gauge/random``: seeded sigma-generated SU(3) background field.

    The field is *always* generated by ``lattice.generate_gauge_field`` (the
    tag names the case): ``gauge`` is reported as ``random`` next to
    ``gauge_requested`` so a ``--gauge unit`` request cannot silently turn this
    into an identity-link check.

    Invariants measured on top of the SU(3) check:

    * ``relative_error`` -- max of the deviations that must vanish: same-seed
      regeneration is bit-identical (``repeat_relative``), ``sigma=0`` gives the
      exact identity (``sigma_zero_dev``), unitarity and determinant deviations.
    * ``distinct_relative`` -- a different seed must give a different field
      (a gate of ``{"gt": 0.0}`` proves the seed option acts).
    * ``sigma_relative`` -- doubling ``gauge_sigma`` must change the field (same
      for the sigma option).
    * ``su3_doubled_sigma`` -- the exponential map stays SU(3) at ``2*sigma``.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice = modules.torch, modules.lattice
    su3_tol = float(_get(config.options, "su3_tol", 1e-3, ("su3_tolerance",)))

    with _StdoutCapture(config.capture_stdout) as capture:
        samples, gauge = _measure(
            config,
            lambda: _gauge_field(config, torch, lattice, kind="random"))
        replay = _gauge_field(config, torch, lattice, kind="random")
        repeat_relative = _relative(torch, replay, gauge)
        distinct = _gauge_field(config, torch, lattice, kind="random",
                                seed=config.seed + 1)
        distinct_relative = _relative(torch, distinct, gauge)
        doubled = _gauge_field(config, torch, lattice, kind="random",
                               sigma=2.0 * config.gauge_sigma)
        sigma_relative = _relative(torch, doubled, gauge)
        identity = _gauge_field(config, torch, lattice, kind="random",
                                sigma=0.0)
        eye = torch.eye(3, dtype=config.dtype, device=config.device).reshape(
            3, 3, 1, 1, 1, 1, 1).expand(3, 3, 4, *config.lattice)
        sigma_zero_dev = float((identity - eye).abs().max())

        su3_observations = _su3_observations(torch, lattice, gauge, su3_tol)
        su3_doubled = bool(lattice.check_su3(doubled, tol=su3_tol,
                                             verbose=False))
        gamma_dev, gamma5_dev = _gamma_identity(torch, lattice, config)
        relative = max(repeat_relative, sigma_zero_dev,
                       su3_observations["unitarity_dev"],
                       su3_observations["det_dev"])

    observations = _base(config, torch, "torch/gauge/random")
    observations.update(_timing(samples))
    observations.update(su3_observations)
    observations.update(
        gauge="random",
        gauge_requested=config.gauge,
        gauge_sigma=config.gauge_sigma,
        gauge_seed=config.seed,
        su3_doubled_sigma=su3_doubled,
        gamma_deviation=gamma_dev,
        gamma_dev=gamma_dev,
        gamma5_dev=gamma5_dev,
        relative=relative,
        rel=relative,
        relative_error=relative,
        repeat_relative=repeat_relative,
        distinct_relative=distinct_relative,
        sigma_relative=sigma_relative,
        sigma_zero_dev=sigma_zero_dev,
        gauge_norm=_norm(torch, gauge),
        gauge_shape=list(gauge.shape),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_dslash_wilson(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/dslash/wilson``: full Wilson matvec vs parity decomposition.

    The parity (``give_wilson_eo``/``give_wilson_oe``) helpers return the
    hopping contribution only, so the even/odd source is added back before the
    comparison, exactly as the informal suite does.  A second, independent
    ``operator.matvec`` path is reported as ``relative_operator``.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, tools, dslash = (
        modules.torch, modules.lattice, modules.tools, modules.dslash)
    kappa_value = config.kappa
    kappa = torch.tensor([kappa_value])

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        source = _fermion_field(config, torch, config.seed + 1, default="random")
        full = dslash.give_wilson(src=source, U=gauge, kappa=kappa,
                                  with_I=True, verbose=False)
        gauge_eo = tools.oooxyzt2poooxyzt(input_array=gauge, verbose=False)
        source_eo = tools.oooxyzt2poooxyzt(input_array=source, verbose=False)
        dest_e = dslash.give_wilson_eo(src_o=source_eo[1], U_eo=gauge_eo,
                                      kappa=kappa, verbose=False)
        dest_o = dslash.give_wilson_oe(src_e=source_eo[0], U_eo=gauge_eo,
                                      kappa=kappa, verbose=False)
        reconstructed = tools.poooxyzt2oooxyzt(
            input_array=torch.stack([source_eo[0] + dest_e,
                                     source_eo[1] + dest_o], dim=0),
            verbose=False)
        relative = _relative(torch, reconstructed, full)
        operator = dslash.operator(U=gauge, kappa=kappa, verbose=False)
        samples, applied = _measure(config, lambda: operator.matvec(source))
        relative_operator = _relative(torch, applied, full)
        max_abs = float((reconstructed - full).abs().max())

        # A point or constant source leaves only a handful of non-zero sites,
        # so the same identity is additionally checked on a dense seeded vector.
        # The gate stays ``relative_error <= compare``; the dense number is
        # reported next to it instead of silently replacing the configured
        # source.
        dense_relative = None
        if (config.source or "random") != "random":
            dense = _dense_source(config, torch, config.seed + 1)
            dense_full = dslash.give_wilson(src=dense, U=gauge, kappa=kappa,
                                            with_I=True, verbose=False)
            dense_eo = tools.oooxyzt2poooxyzt(input_array=dense, verbose=False)
            dense_e = dslash.give_wilson_eo(src_o=dense_eo[1], U_eo=gauge_eo,
                                            kappa=kappa, verbose=False)
            dense_o = dslash.give_wilson_oe(src_e=dense_eo[0], U_eo=gauge_eo,
                                            kappa=kappa, verbose=False)
            dense_reconstructed = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([dense_eo[0] + dense_e,
                                         dense_eo[1] + dense_o], dim=0),
                verbose=False)
            dense_relative = _relative(torch, dense_reconstructed, dense_full)

    observations = _base(config, torch, "torch/dslash/wilson")
    observations.update(_timing(samples))
    observations.update(
        dslash="wilson",
        kappa=kappa_value,
        mass=config.mass,
        source=config.source or "random",
        relative=relative,
        rel=relative,
        relative_error=relative,
        relative_operator=relative_operator,
        relative_dense=dense_relative,
        max_abs_error=max_abs,
        full_norm=_norm(torch, full),
        parity_norm=_norm(torch, reconstructed),
        operator_norm=_norm(torch, applied),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_dslash_parity(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/dslash/parity``: ``matvec_all`` vs explicit ``eeo``/``oeo`` blocks.

    Compares the compact even/odd path (``operator.matvec_all``) against the
    full stencil path (``operator.matvec``), against the block-by-block
    ``matvec_eeo``/``matvec_oeo`` reconstruction, and against the direct
    ``give_wilson`` reference.  ``with_clover=True`` (or ``dslash=clover``)
    exercises the even-even/odd-odd sitting blocks as well.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, tools, dslash = (
        modules.torch, modules.lattice, modules.tools, modules.dslash)
    kappa_value = config.kappa
    with_clover = bool(_get(config.options, "with_clover",
                            config.dslash == "clover", ("clover",)))
    kappa = torch.tensor([kappa_value])

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        source = _fermion_field(config, torch, config.seed + 2, default="random")
        clover = (_clover_term(config, dslash, torch, gauge, kappa)
                  if with_clover else None)
        operator = dslash.operator(U=gauge, clover_term=clover, kappa=kappa,
                                  support_parity=True, verbose=False)
        flat = source.reshape([12, *config.lattice])
        full = operator.matvec(source).reshape([12, *config.lattice])
        samples, compact = _measure(config, lambda: operator.matvec_all(flat))
        relative = _relative(torch, compact, full)

        source_eo = tools.oooxyzt2poooxyzt(input_array=flat, verbose=False)
        dest_e = operator.matvec_eeo(src_e=source_eo[0], src_o=source_eo[1])
        dest_o = operator.matvec_oeo(src_e=source_eo[0], src_o=source_eo[1])
        full_eo = tools.oooxyzt2poooxyzt(input_array=full, verbose=False)
        relative_e = _relative(torch, dest_e, full_eo[0])
        relative_o = _relative(torch, dest_o, full_eo[1])

        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, source)
        relative_reference = _relative(torch, compact.reshape(source.shape),
                                       reference.to(dtype=config.dtype,
                                                    device=config.device))

    observations = _base(config, torch, "torch/dslash/parity")
    observations.update(_timing(samples))
    observations.update(
        dslash="clover" if with_clover else "wilson",
        kappa=kappa_value,
        mass=config.mass,
        csw=config.csw,
        with_clover=with_clover,
        source=config.source or "random",
        relative=relative,
        rel=relative,
        relative_error=relative,
        relative_e=relative_e,
        relative_o=relative_o,
        relative_reference=relative_reference,
        max_abs_error=float((compact - full).abs().max()),
        full_norm=_norm(torch, full),
        compact_norm=_norm(torch, compact),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_dslash_clover(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/dslash/clover``: clover term, ``I + T`` inverse round trip.

    ``relative`` compares ``operator.matvec`` against an independently
    assembled ``give_wilson(with_I=True) + give_clover`` reference.
    ``inverse_relative`` applies ``(I + T)``, then ``inverse(I + T)``, and
    compares with the input.  ``csw_zero_dev`` confirms ``csw=0`` switches the
    clover term off.  ``kappa``/``csw`` come from the run options.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, dslash = (
        modules.torch, modules.lattice, modules.dslash)
    kappa_value = config.kappa
    csw = config.csw
    kappa = torch.tensor([kappa_value])

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        source = _fermion_field(config, torch, config.seed + 3, default="random")
        clover = _clover_term(config, dslash, torch, gauge, kappa, csw=csw)
        clover_plus_identity = dslash.add_I(clover_term=clover, verbose=False)
        clover_inverse = dslash.inverse(clover_term=clover_plus_identity,
                                        verbose=False)
        operator = dslash.operator(U=gauge, clover_term=clover, kappa=kappa,
                                  verbose=False)
        samples, applied = _measure(config, lambda: operator.matvec(source))
        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, source)
        relative = _relative(torch, applied, reference)

        round_trip = dslash.give_clover(
            src=dslash.give_clover(src=source, clover_term=clover_inverse,
                                   verbose=False),
            clover_term=clover_plus_identity, verbose=False)
        inverse_relative = _relative(torch, round_trip, source)

        zero_csw = _clover_term(config, dslash, torch, gauge, kappa, csw=0.0)

    observations = _base(config, torch, "torch/dslash/clover")
    observations.update(_timing(samples))
    observations.update(
        dslash="clover",
        kappa=kappa_value,
        mass=config.mass,
        csw=csw,
        source=config.source or "random",
        relative=relative,
        rel=relative,
        relative_error=relative,
        inverse_relative=inverse_relative,
        csw_zero_dev=float(zero_csw.abs().max()) if zero_csw.numel() else 0.0,
        clover_norm=_norm(torch, clover),
        clover_shape=list(clover.shape),
        operator_norm=_norm(torch, applied),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_smear_stout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/smear/stout``: SU(3) preservation and local/global consistency.

    ``relative`` compares the serial stout path with the
    ``support_parallel=True`` path on an unpartitioned (1x1x1x1) grid: with no
    halo exchange they must describe the same smearing.  ``delta`` records how
    far the field actually moved.  ``su3``/``su3_parallel`` are the SU(3)
    checks of both results, ``su3_two_steps`` a two-step check.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, smear = modules.torch, modules.lattice, modules.smear
    nstep = int(_get(config.options, "nstep", 1, ("steps",)))
    rho = float(_get(config.options, "rho", 0.12, ()))
    su3_tol = float(_get(config.options, "su3_tol", 1e-3, ("su3_tolerance",)))

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        samples, serial = _measure(
            config,
            lambda: smear.stout_smear(U=gauge, nstep=nstep, rho=rho,
                                      support_parallel=False))
        parallel = smear.stout_smear(U=gauge, nstep=nstep, rho=rho,
                                     support_parallel=True)
        relative = _relative(torch, parallel, serial)
        delta = _relative(torch, serial, gauge)
        su3_serial = bool(lattice.check_su3(serial, tol=su3_tol, verbose=False))
        su3_parallel = bool(lattice.check_su3(parallel, tol=su3_tol,
                                              verbose=False))
        two_steps = smear.stout_smear(U=gauge, nstep=nstep + 1, rho=rho,
                                     support_parallel=False)
        su3_two_steps = bool(lattice.check_su3(two_steps, tol=su3_tol,
                                               verbose=False))
        two_step_delta = _relative(torch, two_steps, serial)

    observations = _base(config, torch, "torch/smear/stout")
    observations.update(_timing(samples))
    observations.update(
        nstep=nstep,
        rho=rho,
        su3=su3_serial,
        su3_parallel=su3_parallel,
        su3_two_steps=su3_two_steps,
        su3_tolerance=su3_tol,
        relative=relative,
        rel=relative,
        relative_error=relative,
        delta=delta,
        two_step_delta=two_step_delta,
        max_abs_error=float((parallel - serial).abs().max()),
        smeared_norm=_norm(torch, serial),
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def _wuppertal_step(torch: Any, source: Any, gauge: Any,
                    sigma: float) -> Any:
    """One independent Gaussian (Wuppertal) step, written from the formula.

    ``x' = (1 - 6 sigma) x + sigma sum_{mu=x,y,z} [U_mu x(x+mu) +
    U_mu^dag(x-mu) x(x-mu)]`` on a periodic lattice.  This is the reference the
    core ``smear.wuppertal_smear`` result is compared against; it is computed
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


def run_smear_wuppertal(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/smear/wuppertal``: Gaussian fermion smearing invariants.

    The fermion field is smeared with ``smear.wuppertal_smear`` at the
    configured ``rho``/``nstep``; ``su3`` reports the (untouched) gauge field the
    smearing consumed, so the standard ``smear`` gate group applies.

    ``relative_error`` is the largest of three invariants that must be zero:

    * ``constant_relative`` -- with unit links a constant field is an exact
      eigenvector (``(1 - 6 sigma) + 6 sigma = 1``);
    * ``time_leak_relative`` -- the kernel only sums spatial neighbours, so a
      source that lives on one time slice must stay on that time slice;
    * ``one_step_relative`` -- one step must reproduce
      :func:`_wuppertal_step`, an independent transcription of the formula.

    ``norm_ratio``/``norm_bound``/``norm_ok`` record the contraction bound
    ``||x'|| <= (|1 - 6 sigma| + 6 sigma) ||x||`` (the bound only equals 1 for
    ``sigma <= 1/6``, so it is reported instead of assumed).
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, smear = modules.torch, modules.lattice, modules.smear
    nstep = int(_get(config.options, "nstep", 10, ("steps",)))
    rho = float(_get(config.options, "rho", 1.0, ()))
    su3_tol = float(_get(config.options, "su3_tol", 1e-3, ("su3_tolerance",)))
    sigma = rho * rho / (4.0 * nstep)

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        su3 = bool(lattice.check_su3(gauge, tol=su3_tol, verbose=False))
        source = _fermion_field(config, torch, config.seed + 4,
                                default="random")
        samples, smeared = _measure(
            config,
            lambda: smear.wuppertal_smear(src=source, U=gauge, rho=rho,
                                          nstep=nstep, support_parallel=False))
        source_norm = _norm(torch, source)
        smeared_norm = _norm(torch, smeared)
        norm_ratio = (smeared_norm / source_norm) if source_norm else 0.0
        norm_bound = abs(1.0 - 6.0 * sigma) + 6.0 * sigma
        norm_ok = bool(norm_ratio <= norm_bound + 1e-6)

        unit_gauge = _gauge_field(config, torch, lattice, kind="unit")
        constant = torch.ones((4, 3, *config.lattice), dtype=config.dtype,
                              device=config.device)
        constant_smeared = smear.wuppertal_smear(
            src=constant, U=unit_gauge, rho=rho, nstep=nstep,
            support_parallel=False)
        constant_relative = _relative(torch, constant_smeared, constant)

        local = torch.zeros_like(source)
        local[..., 0] = source[..., 0]
        local_smeared = smear.wuppertal_smear(
            src=local, U=gauge, rho=rho, nstep=nstep, support_parallel=False)
        total = _norm(torch, local_smeared)
        time_leak_relative = (
            _norm(torch, local_smeared[..., 1:]) / total if total else 0.0)

        one_step = smear.wuppertal_smear(src=source, U=gauge, rho=rho, nstep=1,
                                         support_parallel=False)
        expected_step = _wuppertal_step(torch, source, gauge, rho * rho / 4.0)
        one_step_relative = _relative(torch, one_step, expected_step)

        delta = _relative(torch, smeared, source)
        relative = max(constant_relative, time_leak_relative, one_step_relative)

    observations = _base(config, torch, "torch/smear/wuppertal")
    observations.update(_timing(samples))
    observations.update(
        smear="wuppertal",
        rho=rho,
        nstep=nstep,
        sigma=sigma,
        su3=su3,
        su3_tolerance=su3_tol,
        relative=relative,
        rel=relative,
        relative_error=relative,
        constant_relative=constant_relative,
        time_leak_relative=time_leak_relative,
        one_step_relative=one_step_relative,
        norm_ratio=norm_ratio,
        norm_bound=norm_bound,
        norm_ok=norm_ok,
        delta=delta,
        source=config.source or "random",
        source_norm=source_norm,
        smeared_norm=smeared_norm,
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_solve_bistabcg(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/solve/bistabcg``: point-source Wilson solve + true residual.

    The source is a single spin/colour delta at the origin (``||b|| == 1``)
    unless ``source`` says otherwise, the solve is ``solver.bistabcg`` at
    ``kappa`` with a zero initial guess, and the reported ``rel_res`` is rebuilt
    from the public ``give_wilson`` (+ optional clover) reference -- never from
    the solver's own recurrence.  ``dslash=clover`` adds the ``csw`` clover term
    to both the solve and the independent reference.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, dslash, solver = (
        modules.torch, modules.lattice, modules.dslash, modules.solver)
    kappa_value = config.kappa
    tolerance = float(_get(config.options, "tolerance", 1e-6, ("tol",)))
    # Ask for one extra digit in complex128: the requested tolerance is a
    # recurrence norm, so stopping exactly on it would leave the independently
    # recomputed residual sitting on the 1e-6 policy boundary.  ``tolerance``
    # stays the gate; ``solver_tolerance`` is what the solver is asked for.
    solver_tolerance = float(_get(
        config.options, "solver_tolerance",
        tolerance if config.dtype is not torch.complex128 else tolerance * 1e-2,
        ("solver_tol",)))
    max_iter = config.max_iter if config.max_iter is not None else 1000
    if_rtol = bool(_get(config.options, "if_rtol", True, ()))
    kappa = torch.tensor([kappa_value])
    dslash_kind = _solve_dslash_kind(config, "wilson")

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        clover = (_clover_term(config, dslash, torch, gauge, kappa)
                  if dslash_kind == "clover" else None)
        b = _fermion_field(config, torch, config.seed, default="point")
        operator = dslash.operator(U=gauge, clover_term=clover, kappa=kappa,
                                  verbose=False)

        def _solve_once() -> Tuple[Any, List[float]]:
            # A fresh history per call: the solver appends to it, so reusing one
            # list across samples would corrupt the iteration count.
            call_history: List[float] = []
            vector = solver.bistabcg(
                b=b, matvec=operator.matvec, tol=solver_tolerance,
                max_iter=max_iter, x0=torch.zeros_like(b), if_rtol=if_rtol,
                verbose=False, history=call_history)
            return vector, call_history

        samples, (solution, history) = _measure(config, _solve_once)
        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, solution)
        residual = reference - b
        rel_res = _relative(torch, reference, b)
        iterations = max(len(history) - 1, 0)
        history_tail = [float(value) for value in history[-3:]]

    observations = _base(config, torch, "torch/solve/bistabcg")
    observations.update(_timing(samples))
    observations.update(
        operator="wilson+clover" if clover is not None else "wilson",
        system="full",
        dslash=dslash_kind,
        solver="bistabcg",
        solver_impl="bistabcg",
        kappa=kappa_value,
        mass=config.mass,
        csw=config.csw,
        source=config.source or "point",
        source_site=[0, 0, 0, 0],
        source_spin=0,
        source_color=0,
        tolerance=tolerance,
        solver_tolerance=solver_tolerance,
        if_rtol=if_rtol,
        max_iter=max_iter,
        converged=bool(rel_res <= tolerance),
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


def run_solve_cg(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/solve/cg``: even-odd Schur CG solve + independent full residual.

    Plain CG needs a Hermitian positive-definite operator; the Wilson operator
    is only gamma5-Hermitian, so this case solves the **even-odd Schur
    complement** (QUDA ``MATPC``) with ``pyqcu.torch.solver``'s CG family:

    ``S_o x_o = b_o - D_oe D_ee^-1 b_e``, ``x_e = D_ee^-1 (b_e - D_eo x_o)``.

    The gate is still the physical ``||D x - b|| / ||b||`` of the reconstructed
    full field, rebuilt from the public ``give_wilson`` (+ optional clover)
    reference.  ``solver`` selects the implementation (``cg``/``cacg`` ->
    ``cacg``, or ``bistabcg``/``mr``/``fgmres``), ``max_iter``/``restart``/
    ``n_krylov`` tune it, and ``matvecs`` counts the operator applications of
    the reported solve, so ``iterations`` can be compared with the budget.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, tools, dslash, solver = (
        modules.torch, modules.lattice, modules.tools, modules.dslash,
        modules.solver)
    kappa_value = config.kappa
    tolerance = float(_get(config.options, "tolerance", 1e-6, ("tol",)))
    solver_tolerance = float(_get(
        config.options, "solver_tolerance",
        tolerance if config.dtype is not torch.complex128 else tolerance * 1e-2,
        ("solver_tol",)))
    max_iter = config.max_iter if config.max_iter is not None else 400
    restart = config.restart if config.restart is not None else 30
    n_krylov = int(_get(config.options, "n_krylov", 8, ("krylov", "s")))
    omega = float(_get(config.options, "omega", 1.0, ()))
    # ``solver`` is the *policy* vocabulary (``all``/``bicgstab``/``cg``/
    # ``multigrid``/``mg``) plus the concrete entry points; only
    # ``config.solver_selected`` is normalised to a key of
    # ``_KRYLOV_ENTRY_POINTS`` (``all``/``cg``/``cacg``/``mg``/``multigrid``
    # -> ``cg``, PyQCU's CA-CG; ``bicgstab`` -> ``bistabcg``; ``mr`` and
    # ``fgmres`` as named).  Indexing ``_KRYLOV_ENTRY_POINTS`` with the raw
    # option raised ``KeyError: 'all'`` (``policy.DEFAULT_SOLVER``);
    # ``solver_requested`` keeps the user spelling in the observations.
    solver_name = config.solver_selected or "cg"
    implementation, implementation_name = _krylov_entry(solver, solver_name)
    dslash_kind = _solve_dslash_kind(config, "wilson")
    kappa = torch.tensor([kappa_value])

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        clover = (_clover_term(config, dslash, torch, gauge, kappa)
                  if dslash_kind == "clover" else None)
        # ``support_parity`` builds the compact ``D_ee^-1``/``D_oo^-1`` blocks
        # the Schur complement needs; ``parity_only`` stays False so the full
        # operator is available for the independent residual.
        operator = dslash.operator(U=gauge, clover_term=clover, kappa=kappa,
                                  support_parity=True, verbose=False)
        b = _fermion_field(config, torch, config.seed + 5, default="point")
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
            vector = implementation(**call)
            return vector, history, counter["matvec"]

        samples, (x_o, history, matvecs) = _measure(config, _solve_once)
        x_e = operator.give_x_e(b_e=b_e, x_o=x_o)
        solution = tools.poooxyzt2oooxyzt(
            input_array=torch.stack([x_e, x_o], dim=0),
            verbose=False).reshape(b.shape)
        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, solution)
        residual = reference - b
        rel_res = _relative(torch, reference, b)
        iterations = max(len(history) - 1, 0)
        history_tail = [float(value) for value in history[-3:]]

    observations = _base(config, torch, "torch/solve/cg")
    observations.update(_timing(samples))
    observations.update(
        operator="wilson+clover" if clover is not None else "wilson",
        system="schur_eo",
        dslash=dslash_kind,
        solver=solver_name,
        solver_impl=implementation_name,
        kappa=kappa_value,
        mass=config.mass,
        csw=config.csw,
        source=config.source or "point",
        source_site=[0, 0, 0, 0],
        source_spin=0,
        source_color=0,
        tolerance=tolerance,
        solver_tolerance=solver_tolerance,
        if_rtol=True,
        max_iter=max_iter,
        n_krylov=n_krylov,
        restart=restart,
        omega=omega,
        converged=bool(rel_res <= tolerance),
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


def run_mg_legacy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/mg/legacy``: explicit Torch ``LegacyMultigrid`` variant.

    Runs the *pure torch* legacy hierarchy (``with_cuda_qcu`` must stay
    ``False``, therefore ``clover_ee_inv``/``clover_oo_inv`` are never handed
    in) on ``(8, 8, 8, 16)`` with two levels.  ``support_parity=False`` is used
    deliberately: the compact odd-field path of this pure torch variant feeds
    a ``t // 2`` residual into ``tools.restrict`` with a null basis built for
    the full time extent (DimensionError in ``_legacy_multigrid.coarse_correct``
    for this checkout), so the full-field variant is the only one that runs
    end to end here.  The requested configuration is reported in the metrics
    instead of being hidden.

    ``dslash=clover`` (default) hands the hierarchy a clover term at ``csw``;
    ``dslash=wilson`` passes ``clover_term=None`` (the hierarchy's own
    identity-block path).  ``mg_grid``/``mg_levels``/``restart``/``max_iter``
    map to ``mg_grid_size``/``max_level``/``num_restart``/``max_iter``;
    ``mg_nu_pre``/``mg_nu_post`` have no legacy counterpart and are reported as
    ``None``.

    ``converged`` is decided by the independently recomputed ``rel_res``
    against ``tolerance``; the hierarchy's own stop reason and the residual it
    stopped at are reported as ``native_*`` observations.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, dslash, solver = (
        modules.torch, modules.lattice, modules.dslash, modules.solver)
    kappa_value = config.kappa
    tolerance = float(_get(config.options, "tolerance", 1e-6, ("tol",)))
    # The hierarchy stops on a transformed (Schur/MATPC) norm, so ask for one
    # extra order below the independent gate: c128 has the headroom, c64 does
    # not (fp32 stalls around 1e-7).
    solver_tolerance = float(_get(
        config.options, "solver_tolerance",
        1e-7 if config.dtype is not torch.complex128 else 1e-9,
        ("solver_tol",)))
    max_iter = config.max_iter if config.max_iter is not None else 1000
    max_level = config.mg_levels if config.mg_levels is not None else 2
    nvec = int(_get(config.options, "nvec", 12, ("num_vec",)))
    num_restart = config.restart if config.restart is not None else 3
    support_parity = bool(_get(config.options, "support_parity", False, ()))
    min_size = int(_get(config.options, "min_size", 4, ()))
    dslash_kind = _solve_dslash_kind(config, "clover")
    kappa = torch.tensor([kappa_value])

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        clover = (_clover_term(config, dslash, torch, gauge, kappa)
                  if dslash_kind == "clover" else None)
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
            samples, solution = _measure(config, lambda: multigrid.solve(b=b))
            solve_seconds = float(statistics.median(samples))
            report = dict(getattr(multigrid, "last_solve_report", {}) or {})
            history = [float(value) for value in
                       (getattr(multigrid, "convergence_history", []) or [])]
        finally:
            multigrid.end()

        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, solution)
        residual = reference - b
        rel_res = _relative(torch, reference, b)

    observations = _base(config, torch, "torch/mg/legacy")
    observations.update(_timing(samples))
    observations.update(
        variant="legacy",
        mg_backend="torch",
        dslash=dslash_kind,
        with_cuda_qcu=bool(getattr(multigrid, "with_cuda_qcu", False)),
        kappa=kappa_value,
        mass=config.mass,
        csw=config.csw,
        source=config.source or "point",
        tolerance=tolerance,
        solver_tolerance=solver_tolerance,
        support_parity=support_parity,
        max_level=max_level,
        mg_levels=max_level,
        mg_grid=list(config.mg_grid),
        mg_nu_pre=None,
        mg_nu_post=None,
        smoother_counts_supported=False,
        min_size=min_size,
        nvec=nvec,
        dof_list=dof_list[:max(levels, 1)],
        num_restart=num_restart,
        max_iter=max_iter,
        levels=levels,
        converged=bool(rel_res <= tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=int(report.get("iterations", -1)),
        vcycles=int(report.get("vcycles", -1)),
        native_stop_reason=report.get("stop_reason"),
        native_converged=bool(
            str(report.get("stop_reason", "")).endswith("tolerance")),
        native_history_tail=history[-3:],
        residual_scope=report.get("residual_scope"),
        outer_solver=report.get("outer_solver"),
        construct_seconds=construct_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
    )
    observations.update(capture.observations())
    return _finish(observations, started)


def run_mg_strict(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``torch/mg/strict``: explicit Torch ``StrictMultigrid`` variant.

    Strict is opt-in only: this runner always asks for ``hierarchy_mode``
    "strict" through the public ``StrictMultigrid`` class and never falls back
    to legacy.  ``converged`` is again the independent ``rel_res`` gate; the
    hierarchy reports its own MATPC-scope criterion in ``native_*``.

    ``mg_grid``/``mg_levels``/``mg_nu_pre``/``mg_nu_post``/``restart``/
    ``max_iter`` map to ``block_size``/``max_level``/``nu_pre``/``nu_post``/
    ``restart``/``max_iter``; the smoother counts are only passed when they were
    requested, so the core defaults stay in force otherwise.
    """
    config = _Config(args, kwargs)  # warms the lazy torch/pyqcu imports
    started = time.perf_counter()
    modules = _load()
    torch, lattice, dslash, solver = (
        modules.torch, modules.lattice, modules.dslash, modules.solver)
    kappa_value = config.kappa
    tolerance = float(_get(config.options, "tolerance", 1e-6, ("tol",)))
    # The hierarchy stops on a transformed (Schur/MATPC) norm, so ask for one
    # extra order below the independent gate: c128 has the headroom, c64 does
    # not (fp32 stalls around 1e-7).
    solver_tolerance = float(_get(
        config.options, "solver_tolerance",
        1e-7 if config.dtype is not torch.complex128 else 1e-9,
        ("solver_tol",)))
    max_iter = config.max_iter if config.max_iter is not None else 200
    max_level = config.mg_levels if config.mg_levels is not None else 2
    nvec = int(_get(config.options, "nvec", 6, ("num_vec",)))
    restart = config.restart if config.restart is not None else 20
    materialize = bool(_get(config.options, "materialize_coarse", True, ()))
    dslash_kind = _solve_dslash_kind(config, "clover")
    kappa = torch.tensor([kappa_value])
    smoother: Dict[str, int] = {}
    if config.mg_nu_pre is not None:
        smoother["nu_pre"] = int(config.mg_nu_pre)
    if config.mg_nu_post is not None:
        smoother["nu_post"] = int(config.mg_nu_post)

    with _StdoutCapture(config.capture_stdout) as capture:
        gauge = _gauge_field(config, torch, lattice)
        clover = (_clover_term(config, dslash, torch, gauge, kappa)
                  if dslash_kind == "clover" else None)
        b = _fermion_field(config, torch, config.seed, default="point")
        build_started = time.perf_counter()
        multigrid = solver.StrictMultigrid(
            U=gauge, clover_term=clover, kappa=kappa,
            lat_size=list(config.lattice), nvec_list=[nvec],
            max_level=max_level, block_size=tuple(config.mg_grid),
            tol=solver_tolerance, max_iter=max_iter,
            restart=restart, use_parity=True,
            materialize_coarse=materialize, verbose=False, seed=config.seed,
            **smoother)
        construct_seconds = time.perf_counter() - build_started
        try:
            setup_started = time.perf_counter()
            multigrid.setup()
            setup_seconds = time.perf_counter() - setup_started
            samples, solution = _measure(config, lambda: multigrid.solve(b))
            solve_seconds = float(statistics.median(samples))
            report = dict(getattr(multigrid, "last_solve_report", {}) or {})
            history = [float(value) for value in
                       (getattr(multigrid, "convergence_history", []) or [])]
        finally:
            multigrid.close()

        reference = _wilson_plus_clover_reference(
            config, modules, gauge, clover, kappa, solution)
        residual = reference - b
        rel_res = _relative(torch, reference, b)

    observations = _base(config, torch, "torch/mg/strict")
    observations.update(_timing(samples))
    observations.update(
        variant="strict",
        mg_backend="torch",
        hierarchy_mode="strict",
        dslash=dslash_kind,
        kappa=kappa_value,
        mass=config.mass,
        csw=config.csw,
        source=config.source or "point",
        tolerance=tolerance,
        solver_tolerance=solver_tolerance,
        max_level=max_level,
        mg_levels=max_level,
        mg_grid=list(config.mg_grid),
        mg_nu_pre=config.mg_nu_pre,
        mg_nu_post=config.mg_nu_post,
        smoother_counts_supported=True,
        smoother=dict(smoother),
        nvec=nvec,
        restart=restart,
        max_iter=max_iter,
        use_parity=True,
        materialize_coarse=materialize,
        converged=bool(rel_res <= tolerance),
        rel_res=rel_res,
        rel=rel_res,
        relative_residual=rel_res,
        residual_norm=_norm(torch, residual),
        b_norm=_norm(torch, b),
        x_norm=_norm(torch, solution),
        iterations=int(report.get("iterations", -1)),
        vcycles=int(report.get("vcycles", -1))
        if report.get("vcycles") is not None else None,
        residual_scope=report.get("residual_scope"),
        outer_solver=report.get("outer_solver"),
        native_stop_criterion=report.get("stop_criterion"),
        native_relative_residual=report.get("native_relative_residual"),
        native_converged=bool(
            report.get("native_relative_residual") is not None
            and float(report["native_relative_residual"]) <= tolerance),
        native_history_tail=history[-3:],
        construct_seconds=construct_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
    )
    observations.update(capture.observations())
    return _finish(observations, started)


#: Runner registry keyed by the tag used in ``units.CASES``.
RUNNERS = {
    "lattice": run_lattice,
    "gauge_random": run_gauge_random,
    "dslash_wilson": run_dslash_wilson,
    "dslash_parity": run_dslash_parity,
    "dslash_clover": run_dslash_clover,
    "smear_stout": run_smear_stout,
    "smear_wuppertal": run_smear_wuppertal,
    "solve_bistabcg": run_solve_bistabcg,
    "solve_cg": run_solve_cg,
    "mg_legacy": run_mg_legacy,
    "mg_strict": run_mg_strict,
}

#: ``units.CASES`` tag -> runner, for a direct ``TAG_RUNNERS[case.tag]`` lookup.
TAG_RUNNERS = {
    "torch/lattice": run_lattice,
    "torch/gauge/random": run_gauge_random,
    "torch/dslash/wilson": run_dslash_wilson,
    "torch/dslash/parity": run_dslash_parity,
    "torch/dslash/clover": run_dslash_clover,
    "torch/smear/stout": run_smear_stout,
    "torch/smear/wuppertal": run_smear_wuppertal,
    "torch/solve/bistabcg": run_solve_bistabcg,
    "torch/solve/cg": run_solve_cg,
    "torch/mg/legacy": run_mg_legacy,
    "torch/mg/strict": run_mg_strict,
}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny local harness: ``python -m pyqcu.test.torch_qcu [tag ...]``.

    Prints one compact observation line per runner.  This is not the ``pyqcu.test``
    CLI (A1 owns that); it exists so the torch layer can be exercised alone.
    ``NAME=VALUE`` arguments before the tags set a run option, e.g.
    ``python -m pyqcu.test.torch_qcu mass=0.1 dtype=c128 torch/solve/cg``.
    """
    import json

    arguments = list(argv if argv is not None else sys.argv[1:])
    overrides: Dict[str, Any] = {}
    tags: List[str] = []
    for item in arguments:
        name, separator, value = item.partition("=")
        if separator and name and name.replace("_", "").isalnum() \
                and not name.startswith("torch"):
            overrides[name] = value
        else:
            tags.append(item)
    if not tags or tags == ["all"]:
        tags = list(TAG_RUNNERS)
    env = probe_env()
    print(json.dumps({"event": "env", **env}, sort_keys=True))
    exit_code = 0
    for tag in tags:
        runner = TAG_RUNNERS.get(tag) or RUNNERS.get(tag)
        if runner is None:
            print(f"unknown runner {tag!r}; choose from {sorted(TAG_RUNNERS)}",
                  file=sys.stderr)
            return 2
        try:
            observations = runner(**overrides)
        except Skip as error:
            print(f"SKIP    {tag}: {error}", file=sys.stderr)
            continue
        except Exception as error:  # harness only: the runners must not catch
            print(f"ERROR {tag}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            exit_code = 1
            continue
        print(json.dumps({"tag": tag, **observations}, sort_keys=True,
                         default=str))
    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual harness
    raise SystemExit(_main())
