"""Tier-0 case registry of the formal :mod:`pyqcu.test` port.

This module is **declarative**: it names the 21 tier-0 cases, the runner that
produces each one's observations and the ``expect`` gates that decide the
verdict.  It imports no backend (no torch, numpy, mpi4py or pytest), so
``python -m pyqcu.test --list`` can print the whole matrix in an interpreter
that has none of them installed.

Runner contract
---------------
``Case.runner`` is ``"<module>:<function>"``.  A module part without a dot is
resolved inside :mod:`pyqcu.test` (``torch_qcu`` -> ``pyqcu.test.torch_qcu``)
with :func:`importlib.import_module`, lazily, at the moment the case runs.

The harness calls the function with the keyword arguments it actually
declares, chosen from::

    lattice        tuple[int, int, int, int]  e.g. (8, 8, 8, 16)
    lattice_grid   tuple[int, int, int, int]  MPI rank grid, e.g. (1, 1, 1, 1)
    mg_grid        tuple[int, int, int, int]  multigrid block grid
    dtype          "c64" | "c128"
    seed           int                        deterministic source/gauge seed
    repeat         int >= 1                   timing samples
    warmup         int >= 1                   discarded timings
    atol           float | None               relative-comparison override
    source         "point" | "ones" | "random"
    gauge          "random" | "unit" | "ones"
    gauge_sigma    float                      Gaussian width of the background
    mass/kappa     float | None               fermion mass / hopping parameter
    csw            float                      clover coefficient (0 = Wilson)
    solver         "all" | "bicgstab" | "cg" | "multigrid" | "mg"
    dslash         "all" | "wilson" | "clover" | "both"
    device         str | None                 runner-facing device (None: auto)
    max_iter       int | None                 solver iteration budget
    restart        int | None                 Krylov restart length
    mg_levels      int | None                 multigrid level count
    mg_nu_pre      int | None                 pre-smoothing steps
    mg_nu_post     int | None                 post-smoothing steps
    threads        int | None                 native thread count
    device_id      int | None                 accelerator index

Fields the user did not set are **not** forwarded (``None`` means "runner
default"), so a runner's own default always applies.  A runner that declares
``options``/``ctx``/``context``/``config`` (or ``**kwargs``) also receives the
frozen :class:`pyqcu.test.common.Options` object.  Nothing else is passed: a
runner owns its backend import, its device probe, its skip decision and its
own timing (``pyqcu.test.common.timed``).

Return value: a **mapping of observations** - numbers, booleans, strings and
lists of those.  A runner must never return ``PASS``/``FAIL``; ``expect``
decides the verdict after the call, so a runner cannot claim success.
Optional keys with harness meaning:

``seconds``   float   median of the timed core operation (else harness wall time)
``skip``      str     skip the case; the string is the mandatory reason,
                      preferably ``"<token>: <reason>"`` with a token from
                      :data:`pyqcu.test.policy.SKIP_TOKENS`
``token``     str     explicit skip token
``reason``    str     reason accompanying ``skip: True`` or ``status: SKIP``

Raising :class:`pyqcu.test.common.Skip` is the preferred way to skip.  Any
other exception is reported as ``ERROR`` (never a traceback out of the CLI),
and ``TimeoutError`` as ``TIMEOUT``.  A skip without a non-empty reason is an
``ERROR``: an unexplained skip is indistinguishable from a broken probe.

Expected observations per case
------------------------------
==========================  ==================================================
tag                         required observation keys
==========================  ==================================================
``torch/lattice``           ``su3`` (bool), ``gamma_deviation`` (float)
``torch/dslash/wilson``     ``relative_error`` (float)
``torch/dslash/parity``     ``relative_error`` (float)
``torch/dslash/clover``     ``relative_error`` (float)
``torch/smear/stout``       ``su3`` (bool), ``relative_error`` (float)
``torch/smear/wuppertal``   ``su3`` (bool), ``relative_error`` (float)
``torch/gauge/random``      ``su3`` (bool)
``torch/solve/bistabcg``    ``relative_residual``, ``iterations``
``torch/solve/cg``          ``relative_residual``, ``iterations``
``torch/mg/legacy``         ``variant="legacy"``, ``converged``,
                            ``relative_residual``
``torch/mg/strict``         ``variant="strict"``, ``converged``,
                            ``relative_residual``
``kupl/abi/symbols``        ``count`` (int, 40), ``set_match`` (bool)
``kupl/dslash/wilson``      ``relative_error`` (float)
``kupl/dslash/clover``      ``relative_error`` (float)
``kupl/solve/bistabcg``     ``relative_residual``, ``iterations``,
                            ``max_iter``
``kupl/solve/cg``           ``relative_residual``, ``iterations``,
                            ``max_iter``
``cuda/smoke/init_end``     ``init_ok`` (bool)
``cuda/dslash/wilson``      ``relative_error`` (float)
``cuda/dslash/clover``      ``relative_error`` (float)
``cuda/solve/bistabcg``     ``relative_residual``, ``iterations``,
                            ``max_iter``
``cuda/solve/cg``           ``relative_residual``, ``iterations``,
                            ``max_iter``
==========================  ==================================================

``relative_error`` is the relative error against an independent reference path
(:func:`pyqcu.test.common.relative_error`); ``relative_residual`` is the
independently reconstructed ``||D x - b|| / ||b||`` of the **full** Wilson
operator (:func:`pyqcu.test.common.relative_residual`), never the solver's own
recurrence.  The spellings used by the backend runners and the legacy
``pyqcu/testing`` reports are accepted as aliases
(``policy.OBSERVATION_ALIASES``): ``relative`` for ``relative_error``,
``rel_res``/``true_residual`` for ``relative_residual`` and ``gamma_dev`` for
``gamma_deviation``.

Skip tokens declared by each case (``Case.requires``) are metadata for
``--list`` and the report.  Enforcing them - probing the KUPL library, the
CUDA bridge, the device or the MPI size - is the runner module's job, because
only it knows how to probe its backend without disturbing it.  Two tokens are
*conditional* and therefore not listed as a requirement: ``mpi_grid`` (the
native runners skip when ``prod(--lattice-grid)`` does not match the MPI size;
the torch path ignores the grid and must not fail) and ``device_unavailable``
(the requested ``--device`` is not usable on this machine).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from . import policy
from .common import Case

__all__ = ["CASES", "BY_TAG", "TAGS", "cases", "get", "tags"]

#: Default per-case wall-clock budget (seconds).  Tier-0 cases are second-scale
#: at ``8x8x8x16``; the budget is a runaway guard, not a performance gate.
_QUICK = 120.0
_SOLVE = 300.0
_MG = 600.0
_ABI = 60.0


def _case(tag: str, runner: str, kind: str, *,
          requires: tuple[str, ...] = (),
          expect: Mapping[str, Mapping[str, object]],
          timeout: float) -> Case:
    """Build one tier-0 case, deriving ``backend`` from the tag prefix."""
    return Case(
        tag=tag,
        backend=tag.split("/", 1)[0],
        kind=kind,
        runner=runner,
        lattice=policy.QUICK_LATTICE,
        dtype=policy.QUICK_DTYPE,
        tier=0,
        requires=requires,
        expect=expect,
        timeout=timeout,
    )


#: The 21 tier-0 cases, in plan order (the first 14 are the original matrix,
#: the last 7 are the gauge/Wuppertal/CG/clover additions).  Runner strings are
#: part of the public contract (``policy.TIER0_RUNNERS`` restates them;
#: ``selfcheck`` compares).
CASES: tuple[Case, ...] = (
    _case(
        "torch/lattice", "torch_qcu:run_lattice", "lattice",
        expect=policy.RESIDUAL_GATES["lattice"],
        timeout=_QUICK,
    ),
    _case(
        "torch/dslash/wilson", "torch_qcu:run_dslash_wilson", "dslash",
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "torch/dslash/parity", "torch_qcu:run_dslash_parity", "dslash",
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "torch/dslash/clover", "torch_qcu:run_dslash_clover", "dslash",
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "torch/smear/stout", "torch_qcu:run_smear_stout", "smear",
        expect=policy.RESIDUAL_GATES["smear"],
        timeout=_QUICK,
    ),
    _case(
        "torch/solve/bistabcg", "torch_qcu:run_solve_bistabcg", "solve",
        expect=policy.RESIDUAL_GATES["solver"],
        timeout=_SOLVE,
    ),
    _case(
        "torch/mg/legacy", "torch_qcu:run_mg_legacy", "mg",
        # The variant is explicit in the gate: a runner that silently falls
        # back to the default variant must not pass as a legacy check.
        expect={**policy.RESIDUAL_GATES["mg"], "variant": {"eq": "legacy"}},
        timeout=_MG,
    ),
    _case(
        "torch/mg/strict", "torch_qcu:run_mg_strict", "mg",
        expect={**policy.RESIDUAL_GATES["mg"], "variant": {"eq": "strict"}},
        timeout=_MG,
    ),
    _case(
        "kupl/abi/symbols", "kupl_qcu:run_abi_symbols", "abi",
        requires=("kupl_library",),
        expect=policy.RESIDUAL_GATES["abi"],
        timeout=_ABI,
    ),
    _case(
        "kupl/dslash/wilson", "kupl_qcu:run_dslash_wilson", "dslash",
        requires=("kupl_library", "mpi_single_rank"),
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "kupl/solve/bistabcg", "kupl_qcu:run_solve_bistabcg", "solve",
        requires=("kupl_library", "mpi_single_rank"),
        expect=policy.RESIDUAL_GATES["native_solver"],
        timeout=_SOLVE,
    ),
    _case(
        "cuda/smoke/init_end", "cuda_qcu:run_smoke_init_end", "smoke",
        requires=("cuda_bridge", "cuda_device"),
        expect=policy.RESIDUAL_GATES["smoke"],
        timeout=_QUICK,
    ),
    _case(
        "cuda/dslash/wilson", "cuda_qcu:run_dslash_wilson", "dslash",
        requires=("cuda_bridge", "cuda_device"),
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "cuda/solve/bistabcg", "cuda_qcu:run_solve_bistabcg", "solve",
        requires=("cuda_bridge", "cuda_device"),
        expect=policy.RESIDUAL_GATES["native_solver"],
        timeout=_SOLVE,
    ),
    _case(
        "torch/gauge/random", "torch_qcu:run_gauge_random", "gauge",
        expect=policy.RESIDUAL_GATES["gauge"],
        timeout=_QUICK,
    ),
    _case(
        "torch/smear/wuppertal", "torch_qcu:run_smear_wuppertal", "smear",
        expect=policy.RESIDUAL_GATES["smear"],
        timeout=_QUICK,
    ),
    _case(
        "torch/solve/cg", "torch_qcu:run_solve_cg", "solve",
        expect=policy.RESIDUAL_GATES["solver"],
        timeout=_SOLVE,
    ),
    _case(
        "kupl/dslash/clover", "kupl_qcu:run_dslash_clover", "dslash",
        requires=("kupl_library", "mpi_single_rank"),
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "kupl/solve/cg", "kupl_qcu:run_solve_cg", "solve",
        requires=("kupl_library", "mpi_single_rank"),
        expect=policy.RESIDUAL_GATES["native_solver"],
        timeout=_SOLVE,
    ),
    _case(
        "cuda/dslash/clover", "cuda_qcu:run_dslash_clover", "dslash",
        requires=("cuda_bridge", "cuda_device"),
        expect=policy.RESIDUAL_GATES["operator"],
        timeout=_QUICK,
    ),
    _case(
        "cuda/solve/cg", "cuda_qcu:run_solve_cg", "solve",
        requires=("cuda_bridge", "cuda_device"),
        expect=policy.RESIDUAL_GATES["native_solver"],
        timeout=_SOLVE,
    ),
)

#: Tag -> case, for lookups from the CLI and the self-check.
BY_TAG: Mapping[str, Case] = MappingProxyType(
    {case.tag: case for case in CASES})

#: Declared tags in order.
TAGS: tuple[str, ...] = tuple(case.tag for case in CASES)



def cases() -> tuple[Case, ...]:
    """Return the full tier-0 case tuple (fresh tuple, same ``Case`` objects)."""
    return CASES


def tags() -> tuple[str, ...]:
    """Return every declared tag in declaration order."""
    return TAGS


def get(tag: str) -> Case:
    """Return the case registered under ``tag`` (``KeyError`` when unknown)."""
    return BY_TAG[tag]
