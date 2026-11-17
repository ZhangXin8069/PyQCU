"""L48 two-node formal test matrix (tier 1, declarative).

What this module is
-------------------
The tier-0 registry (:mod:`pyqcu.test.units`) answers "is the installed library
correct?" on ``8x8x8x16``.  The production question is different: does the same
gate still hold on the real ``48x48x48x96`` configuration, split over **two**
compute nodes with 32 MPI ranks?  This module is the *declaration* of that
matrix - the groups, the per-case parameters, the rank grids and the evidence
status.  It is deliberately not an executor: the full-mode work ledger
(:func:`pyqcu.test.common.build_rows`) and
:func:`pyqcu.test.driver.run_row` consume it, so ``--mode list`` can describe
the matrix on a machine without torch, KUPL, MPI or a scheduler.

Groups
------
``bicgstab``       BiCGStab, c64 and c128 (parity-Schur: levels=1, restart=0)
``legacy_mg``      Legacy-MultiGrid, c64 and c128 (levels=2, restart=5),
                   **explicitly UNVERIFIED** at 32 ranks (see below)
``dslash``         Wilson/clover dslash at L48 through the *existing* KUPL
                   harnesses - no new solver and no new kernel
``input_io``       real L48 HDF5 input, rank-local slabs and an independent
                   full-lattice residual
``grid_variants``  rank grids whose rank-local extents stay even (``2x2x2x4``
                   is the production shape; ``1x2x2x8`` etc. are documented)

Hard exclusions
---------------
* **No Strict-MG.**  The L48 two-node matrix is Legacy-MG only: the Strict
  hierarchy is not part of this acceptance path.
* **No CG.**  The L48 CG group is **retired** (:data:`RETIREMENT`).  The
  shipped tier-0 CG cases (``torch/solve/cg``, ``kupl/solve/cg``) stay exactly
  as they are at ``8x8x8x16``; what no longer exists is a CG group *at L48*.
  :func:`drop_retired` enforces that for every L48 plan.

Evidence, not aspiration
------------------------
Every numeric case carries ``evidence="UNVERIFIED"`` until a real L48 run
produced the independent residual.  A definition is not a PASS: the status
table in ``docs/pyqcu_test_l48_two_node_matrix.md`` and the design note
``docs/pyqcu_test_l48_remote.md`` must say the same thing.

The launcher hook
-----------------
:func:`launch_plan` returns the launch a scheduler script would run (nodes,
ranks per node, grid, rankfile, command) as **data**.  Nothing here starts a
process, and no remote scheduler is required to define, list or review the
matrix.

Called with no launcher arguments it is a *declaration*: ``kind="mpirun"``,
``mpi_launch=False``, no rankfile and no hosts, because the port owns no
allocation.  Pass ``allocation`` (a ``CCS_ALLOC_FILE``) or ``rankfile`` and it
becomes an executable plan built by :mod:`pyqcu.test.launch`:
``kind="drun"`` (the verified ``drun -mpi pmix`` route of job ``2155536``,
``docs/pyqcu_test_l48_remote.md`` §0.1.6) or ``kind="mpirun"`` with an explicit
``-rf`` rankfile.  Asking for a launcher without either is **refused**
(``launch.LaunchError``, "refusing a cross-node launch"): the matrix must never
guess hosts (design note §5.3).

Import rules
------------
Standard library plus :mod:`pyqcu.test.policy`; :mod:`pyqcu.test.launch` (the
launcher planner: allocation parsing, rankfile rows and the ``drun`` /
``mpirun`` commands) is imported **inside** :func:`launch_plan` and
:func:`formal_command`, and :class:`pyqcu.test.common.Case` inside
:func:`cases`.  Nothing here pulls a backend or the execution core, and
``python -m pyqcu.test.launch`` never finds its own module already in
``sys.modules`` (which would give the CLI a second ``LaunchError`` class).
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from . import policy

__all__ = [
    "BICGSTAB",
    "BY_TAG",
    "CASES",
    "CG_TAGS",
    "DECLARATIONS",
    "FORBIDDEN_GROUPS",
    "GRID",
    "GRID_VARIANTS",
    "GridVariant",
    "Group",
    "Harness",
    "LATTICE",
    "LEGACY_MG",
    "LOCAL_LATTICE",
    "MatrixCase",
    "NODES",
    "RANKS",
    "RANKS_PER_NODE",
    "REJECTED_GRID_VARIANTS",
    "RETIREMENT",
    "RETIRED_GROUPS",
    "TIER",
    "cases",
    "drop_retired",
    "formal_command",
    "get",
    "is_cg_tag",
    "is_l48_lattice",
    "launch_plan",
    "parameters",
    "parse_tag",
    "plan_payload",
    "retired_cg_tags",
    "row_overrides",
    "validate",
]

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
#: Production configuration lattice (``beta6.20_mu-0.2770_ms-0.2400_L48x96``).
LATTICE: tuple[int, int, int, int] = (48, 48, 48, 96)
#: Two compute nodes, 16 ranks each: the shape the L48 design targets.
RANKS = 32
NODES = 2
RANKS_PER_NODE = RANKS // NODES
#: Production rank grid: 48/2 x 48/2 x 48/2 x 96/4 = 24^3 x 24 per rank.
GRID: tuple[int, int, int, int] = (2, 2, 2, 4)
LOCAL_LATTICE: tuple[int, int, int, int] = tuple(
    extent // factor for extent, factor in zip(LATTICE, GRID))  # type: ignore[assignment]
#: These cases are tier 1: ``--quick`` (tier 0) never selects them.
TIER = 1

#: Parity-Schur BiCGStab: one level and no periodic coarse correction.
BICGSTAB: Mapping[str, Any] = MappingProxyType(
    {"solver": "bistabcg", "mg_levels": 1, "restart": 0})
#: Legacy-MultiGrid: two levels with a coarse correction every 5 iterations.
LEGACY_MG: Mapping[str, Any] = MappingProxyType(
    {"solver": "mg", "mg_levels": 2, "restart": 5})

#: Group names in declaration order.  :func:`validate` pins these names.
GROUP_NAMES: tuple[str, ...] = (
    "bicgstab", "legacy_mg", "dslash", "input_io", "grid_variants")

#: Groups this matrix must never contain (Strict-MG is excluded by design; the
#: L48 CG group is retired, not merely unselected).
FORBIDDEN_GROUPS: tuple[str, ...] = ("strict-mg", "strict_mg", "cg")
RETIRED_GROUPS: tuple[str, ...] = ("cg",)

#: The retirement record.  This is the single machine-readable statement that
#: L48 CG is gone; docs and tests quote it instead of re-deriving the decision.
RETIREMENT: Mapping[str, Any] = MappingProxyType({
    "group": "l48/cg",
    "status": "RETIRED",
    "date": "2026-11-17",
    "scope": "L48 only",
    "reason": (
        "CG is not part of the L48 two-node acceptance path: the production "
        "L48 solver menu is parity-Schur BiCGStab and Legacy-MG, and a CG "
        "run at L48 has no independent-residual evidence."),
    "kept": (
        "the shipped tier-0 CG cases (torch/solve/cg, kupl/solve/cg) at "
        "8x8x8x16 - retiring the L48 group must not delete tier-0 coverage"),
    "enforced_by": (
        "pyqcu.test.l48.drop_retired (Test/Test.select) and the CLI "
        "usage error for an explicit CG request on an L48 lattice"),
})

#: Tier-0 tags that are CG.  Kept from :mod:`pyqcu.test.policy`, never copied.
CG_TAGS: tuple[str, ...] = tuple(
    tag for tag in policy.TIER0_TAGS if str(tag).split("/")[-1] == "cg")


# ---------------------------------------------------------------------------
# rank grids
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridVariant:
    """One documented rank grid and the rank-local geometry it produces.

    ``status`` is ``"VALID"`` or ``"REJECTED"``.  A valid variant must divide
    the lattice on every axis and keep **every** rank-local extent even - the
    native parity decomposition has no meaning on an odd local extent.  A
    rejected variant documents *why* it cannot be used (``1x1x1x32`` leaves a
    local ``t`` of 3), which is exactly the trap the L24/L32 ``t-finest``
    habit walks into.
    """

    name: str
    ranks: int
    grid: tuple[int, int, int, int]
    status: str = "VALID"
    reason: str = ""
    #: Derived: rank-local extents, whether they are all even, and nodes used
    #: at :data:`RANKS_PER_NODE` ranks per node.
    local: tuple[int, int, int, int] = ()
    even: bool = False
    nodes: int = 0

    def __post_init__(self) -> None:
        grid = tuple(int(value) for value in self.grid)
        if len(grid) != 4 or any(value < 1 for value in grid):
            raise ValueError(
                f"grid variant {self.name!r}: grid must be four positive "
                f"integers, got {self.grid!r}")
        ranks = int(self.ranks)
        if ranks < 1:
            raise ValueError(
                f"grid variant {self.name!r}: ranks must be >= 1, got {ranks}")
        if math.prod(grid) != ranks:
            raise ValueError(
                f"grid variant {self.name!r}: prod({list(grid)}) != ranks "
                f"{ranks}")
        if self.status not in ("VALID", "REJECTED"):
            raise ValueError(
                f"grid variant {self.name!r}: status must be VALID or "
                f"REJECTED, got {self.status!r}")
        divides = all(LATTICE[axis] % grid[axis] == 0 for axis in range(4))
        local = tuple(LATTICE[axis] // grid[axis] for axis in range(4))
        even = divides and all(extent % 2 == 0 for extent in local)
        usable = divides and even
        if self.status == "VALID" and not usable:
            raise ValueError(
                f"grid variant {self.name!r}: local extents {local} are not "
                "all even - a VALID variant must keep the parity "
                "decomposition defined")
        if self.status == "REJECTED" and not str(self.reason).strip():
            raise ValueError(
                f"grid variant {self.name!r}: a REJECTED variant needs a "
                "reason")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "even", even)
        object.__setattr__(self, "nodes",
                           ranks // RANKS_PER_NODE if ranks % RANKS_PER_NODE == 0
                           else 0)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description (``--list --json``)."""
        return {
            "name": self.name,
            "ranks": self.ranks,
            "nodes": self.nodes,
            "ranks_per_node": (self.ranks // self.nodes if self.nodes else None),
            "grid": list(self.grid),
            "local_lattice": list(self.local),
            "local_even": self.even,
            "status": self.status,
            "reason": self.reason,
        }


#: Usable grids.  ``2x2x2x4`` is the production two-node shape; the rest are
#: documented alternatives that keep every rank-local extent even.
GRID_VARIANTS: tuple[GridVariant, ...] = (
    GridVariant("2x2x2x4", 32, (2, 2, 2, 4)),
    GridVariant("1x2x2x8", 32, (1, 2, 2, 8)),
    GridVariant("1x1x2x16", 32, (1, 1, 2, 16)),
    GridVariant("4x2x2x2", 32, (4, 2, 2, 2)),
    GridVariant("2x2x4x2", 32, (2, 2, 4, 2)),
    GridVariant("2x2x2x2", 16, (2, 2, 2, 2)),
    GridVariant("1x1x1x16", 16, (1, 1, 1, 16)),
    GridVariant("2x2x2x8", 64, (2, 2, 2, 8)),
)

#: Grids that must not be used, with the reason.  ``1x1x1x32`` is the
#: ``t-finest`` spelling inherited from L24/L32: 96/32 = 3 is odd.
REJECTED_GRID_VARIANTS: tuple[GridVariant, ...] = (
    GridVariant("1x1x1x32", 32, (1, 1, 1, 32), status="REJECTED",
                reason="96/32 = 3 is an odd rank-local t extent"),
    GridVariant("1x1x1x64", 64, (1, 1, 1, 64), status="REJECTED",
                reason="96/64 is not an integer rank-local t extent"),
)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixCase:
    """One executable L48 row: case identity plus its fixed parameters.

    ``kind`` is the *case* kind (``solve``/``mg``/``io``); ``native_kind`` is
    what :func:`pyqcu.test.native.run_case` dispatches to.  The parameters are
    part of the declaration, not of a run flag: a matrix row must mean the same
    thing on every invocation, so ``row_overrides`` hands them to the ledger
    and the driver.
    """

    name: str
    group: str
    kind: str
    dtype: str
    tag_group: str
    solver: str
    mg_levels: int
    restart: int
    dslash: str = "auto"
    backend: str = "kupl"
    ranks: int = RANKS
    grid: tuple[int, int, int, int] = GRID
    nodes: int = NODES
    requires_input: bool = False
    evidence: str = "UNVERIFIED"
    blocked_by: str = ""
    notes: str = ""
    timeout: float = 3600.0

    def __post_init__(self) -> None:
        if self.kind not in ("solve", "mg", "io"):
            raise ValueError(
                f"L48 case {self.name!r}: kind must be solve/mg/io, got "
                f"{self.kind!r}")
        if self.backend not in policy.BACKENDS:
            raise ValueError(
                f"L48 case {self.name!r}: unknown backend {self.backend!r}")
        if self.dtype not in policy.DTYPE_CHOICES:
            raise ValueError(
                f"L48 case {self.name!r}: dtype must be one of "
                f"{', '.join(policy.DTYPE_CHOICES)}, got {self.dtype!r}")
        if self.solver not in ("bistabcg", "mg"):
            raise ValueError(
                f"L48 case {self.name!r}: solver must be bistabcg or mg, got "
                f"{self.solver!r}")
        if self.group == "bicgstab":
            if self.solver != BICGSTAB["solver"] \
                    or int(self.mg_levels) != BICGSTAB["mg_levels"] \
                    or int(self.restart) != BICGSTAB["restart"]:
                raise ValueError(
                    f"L48 case {self.name!r}: BiCGStab must be "
                    "levels=1/restart=0")
        elif self.group == "legacy_mg":
            if self.solver != LEGACY_MG["solver"] \
                    or int(self.mg_levels) != LEGACY_MG["mg_levels"] \
                    or int(self.restart) != LEGACY_MG["restart"]:
                raise ValueError(
                    f"L48 case {self.name!r}: Legacy-MG must be "
                    "levels=2/restart=5")
        elif self.group == "input_io":
            if not self.requires_input:
                raise ValueError(
                    f"L48 case {self.name!r}: the input/IO group needs real "
                    "HDF5 input (requires_input=True)")
        else:
            raise ValueError(
                f"L48 case {self.name!r}: unknown group {self.group!r}")
        if "cg" in self.name.lower().split("-") \
                or self.name.lower().endswith("-cg"):
            raise ValueError(
                f"L48 case {self.name!r}: the L48 CG group is retired")
        if "strict" in self.name.lower():
            raise ValueError(
                f"L48 case {self.name!r}: Strict-MG is excluded from L48")
        if int(self.ranks) != math.prod(self.grid):
            raise ValueError(
                f"L48 case {self.name!r}: prod({list(self.grid)}) != ranks "
                f"{self.ranks}")
        if self.evidence not in ("UNVERIFIED", "VERIFIED"):
            raise ValueError(
                f"L48 case {self.name!r}: evidence must be UNVERIFIED or "
                f"VERIFIED, got {self.evidence!r}")
        if float(self.timeout) <= 0.0:
            raise ValueError(f"L48 case {self.name!r}: timeout must be > 0")
        object.__setattr__(self, "grid",
                           tuple(int(value) for value in self.grid))
        object.__setattr__(self, "ranks", int(self.ranks))
        object.__setattr__(self, "nodes", int(self.nodes))
        object.__setattr__(self, "timeout", float(self.timeout))

    @property
    def tag(self) -> str:
        """Ledger/case tag: ``<backend>/l48/<group>/<dtype>``."""
        return f"{self.backend}/l48/{self.tag_group}/{self.dtype}"

    @property
    def native_kind(self) -> str:
        """``run_case`` kind (the input/IO group is a native solve)."""
        return "mg" if self.kind == "mg" else "solve"

    @property
    def sampler(self) -> str:
        """Runner contract string of the case (:mod:`pyqcu.test.native`)."""
        return f"native:run_{self.native_kind}"

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe case description."""
        return {
            "tag": self.tag,
            "group": self.group,
            "kind": self.kind,
            "native_kind": self.native_kind,
            "backend": self.backend,
            "dtype": self.dtype,
            "lattice": list(LATTICE),
            "grid": list(self.grid),
            "ranks": self.ranks,
            "nodes": self.nodes,
            "ranks_per_node": (self.ranks // self.nodes if self.nodes else None),
            "solver": self.solver,
            "mg_levels": int(self.mg_levels),
            "restart": int(self.restart),
            "dslash": self.dslash,
            "requires_input": bool(self.requires_input),
            "tier": TIER,
            "timeout": self.timeout,
            "evidence": self.evidence,
            "blocked_by": self.blocked_by,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Harness:
    """An existing repository harness an L48 group is executed with."""

    name: str
    path: str
    purpose: str
    command: str
    evidence: str = "UNVERIFIED"

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe harness description."""
        return {
            "name": self.name,
            "path": self.path,
            "purpose": self.purpose,
            "command": self.command,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Group:
    """One named group of the L48 two-node matrix."""

    name: str
    title: str
    summary: str
    kind: str = "cases"
    cases: tuple[MatrixCase, ...] = ()
    harnesses: tuple[Harness, ...] = ()
    variants: tuple[GridVariant, ...] = ()
    evidence: str = "UNVERIFIED"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.name not in GROUP_NAMES:
            raise ValueError(
                f"L48 group {self.name!r}: unknown group; choose from "
                f"{', '.join(GROUP_NAMES)}")
        if self.name in FORBIDDEN_GROUPS:
            raise ValueError(
                f"L48 group {self.name!r}: forbidden by the matrix "
                f"({', '.join(FORBIDDEN_GROUPS)})")
        if self.kind not in ("cases", "harness", "geometry"):
            raise ValueError(
                f"L48 group {self.name!r}: kind must be cases/harness/"
                f"geometry, got {self.kind!r}")
        if self.kind == "cases" and not self.cases:
            raise ValueError(f"L48 group {self.name!r}: a cases group is empty")
        if self.kind == "harness" and not self.harnesses:
            raise ValueError(
                f"L48 group {self.name!r}: a harness group is empty")
        if self.kind == "geometry" and not self.variants:
            raise ValueError(
                f"L48 group {self.name!r}: a geometry group is empty")
        for case in self.cases:
            if case.group != self.name:
                raise ValueError(
                    f"L48 group {self.name!r}: case {case.name!r} belongs to "
                    f"group {case.group!r}")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "harnesses", tuple(self.harnesses))
        object.__setattr__(self, "variants", tuple(self.variants))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe group description."""
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "kind": self.kind,
            "evidence": self.evidence,
            "notes": self.notes,
            "cases": [case.as_dict() for case in self.cases],
            "harnesses": [harness.as_dict() for harness in self.harnesses],
            "variants": [variant.as_dict() for variant in self.variants],
        }


#: Case declarations of the numerical groups.  Every parameter is explicit so
#: the ledger row, the docs table and the driver agree by construction.
_BICGSTAB_CASES = tuple(
    MatrixCase(
        name=f"bicgstab-{dtype}", group="bicgstab", kind="solve", dtype=dtype,
        tag_group="bicgstab", solver="bistabcg",
        mg_levels=int(BICGSTAB["mg_levels"]),
        restart=int(BICGSTAB["restart"]),
        evidence="UNVERIFIED",
        notes=("parity-Schur BiCGStab (levels=1/restart=0); no L48 run has "
               "produced an independent residual yet"),
    )
    for dtype in policy.DTYPE_CHOICES
)

_LEGACY_MG_CASES = tuple(
    MatrixCase(
        name=f"legacy-mg-{dtype}", group="legacy_mg", kind="mg", dtype=dtype,
        tag_group="legacy-mg", solver="mg",
        mg_levels=int(LEGACY_MG["mg_levels"]),
        restart=int(LEGACY_MG["restart"]),
        evidence="UNVERIFIED",
        blocked_by=(
            "pyqcu.test.native refuses multi-rank Legacy-MG: its coarse "
            "assets come from the serial local 33-stencil builder, which "
            "cannot represent coarse couplings across a rank boundary.  The "
            "32-rank/2-node path is the existing kupl_multigrid.py harness "
            "(distributed cache); until it runs, this group is UNVERIFIED."),
        notes=("Legacy-MultiGrid levels=2/restart=5; 32 ranks or explicitly "
               "UNVERIFIED - this row is the latter"),
        timeout=7200.0,
    )
    for dtype in policy.DTYPE_CHOICES
)

_IO_CASE = MatrixCase(
    name="io-residual-c64", group="input_io", kind="io", dtype="c64",
    tag_group="io-residual", solver="bistabcg",
    mg_levels=int(BICGSTAB["mg_levels"]), restart=int(BICGSTAB["restart"]),
    requires_input=True, evidence="UNVERIFIED",
    notes=("consumes a real L48 HDF5 gauge/source pair through "
           "--gauge-file/--source-file, reads one rank-local slab per rank "
           "and reconstructs the independent residual ||D x - b|| / ||b|| "
           "on the full lattice"),
)

_Dslash_HARNESSES = (
    Harness(
        "kupl-dslash-block-mpi",
        "pyqcu/testing/kupl/test_dslash_block_mpi.py",
        ("Wilson dslash against the block-MPI reference at 32 ranks; "
         "c64/c128 through --precision"),
        ("mpirun -np 32 --map-by ppr:16:node --bind-to none python "
         "pyqcu/testing/kupl/test_dslash_block_mpi.py --library "
         "<libqcu.so> --lattice 48 48 48 96 --grid 2 2 2 4 --threads 18 "
         "--precision c64"),
    ),
    Harness(
        "kupl-clover-check",
        "pyqcu/testing/simple/mpi/check_clover.py",
        ("existing QCU/QUDA clover layout check on the prepared L48 input; "
         "correctness only, no new clover kernel"),
        ("mpirun -np 32 --map-by ppr:16:node --bind-to none python -m "
         "pyqcu.testing.simple.mpi.check_clover --input "
         "<gauge_l48_c64.h5> --backend qcu"),
    ),
)

_DSLASH_GROUP = Group(
    "dslash", "dslash / clover at L48",
    ("correctness and timing of the Wilson and clover operators at L48, "
     "executed with the existing KUPL harnesses; no new solver is added"),
    kind="harness", harnesses=_Dslash_HARNESSES, evidence="UNVERIFIED",
    notes=("The formal tier-0 kupl/dslash/* cases are single-rank by design; "
           "the 32-rank L48 path is the existing block-MPI harness.  A timing "
           "number is only a measurement, never an acceptance gate."),
)

#: The five groups of the matrix, in declaration order.
GROUPS: tuple[Group, ...] = (
    Group(
        "bicgstab", "BiCGStab (c64, c128)",
        "parity-Schur BiCGStab, levels=1/restart=0, grid 2x2x2x4",
        cases=_BICGSTAB_CASES, evidence="UNVERIFIED",
        notes="production solver path; 32 ranks over 2 nodes"),
    Group(
        "legacy_mg", "Legacy-MG (c64, c128)",
        "Legacy-MultiGrid, levels=2/restart=5, 32 ranks/2 nodes "
        "(UNVERIFIED: native path is single-rank today)",
        cases=_LEGACY_MG_CASES, evidence="UNVERIFIED",
        notes=("do not turn this into Strict-MG; the L48 acceptance path is "
               "Legacy-MG only")),
    _DSLASH_GROUP,
    Group(
        "input_io", "input / IO + independent residual",
        "real L48 HDF5 gauge/source, rank-local slabs, independent residual",
        cases=(_IO_CASE,), evidence="UNVERIFIED",
        notes=("a missing --gauge-file/--source-file is an ERROR, never a "
               "silent generated-field run")),
    Group(
        "grid_variants", "rank-grid variants",
        "grids that keep every rank-local extent even",
        kind="geometry", variants=GRID_VARIANTS, evidence="DEFINED",
        notes="1x1x1x32 is rejected: 96/32 = 3 is odd"),
)

#: Executable declarations, in group order.
DECLARATIONS: tuple[MatrixCase, ...] = tuple(
    case for group in GROUPS for case in group.cases)
#: Declarations by tag.
BY_TAG: Mapping[str, MatrixCase] = MappingProxyType(
    {declaration.tag: declaration for declaration in DECLARATIONS})


def cases() -> tuple[Any, ...]:
    """Build the tier-1 :class:`pyqcu.test.common.Case` objects of the matrix.

    ``pyqcu.test.common`` is imported here, not at module import, so
    ``--list``/``--mode list`` stay free of the execution core.
    """
    from .common import Case

    built = []
    for declaration in DECLARATIONS:
        if declaration.native_kind == "mg":
            expect = {**policy.RESIDUAL_GATES["mg"],
                      "variant": {"eq": "legacy"}}
        else:
            expect = dict(policy.RESIDUAL_GATES["native_solver"])
        built.append(Case(
            tag=declaration.tag,
            backend=declaration.backend,
            kind=declaration.kind,
            runner=declaration.sampler,
            lattice=LATTICE,
            dtype=declaration.dtype,
            tier=TIER,
            requires=("kupl_library",),
            expect=expect,
            timeout=declaration.timeout,
            pinned=True,
        ))
    return tuple(built)


def get(tag: Any) -> MatrixCase:
    """Return the declaration for ``tag`` (``KeyError`` when not in the matrix)."""
    return BY_TAG[str(tag)]


def parse_tag(tag: Any) -> tuple[str, str] | None:
    """``(backend, native_kind)`` of an L48 matrix tag, else ``None``.

    This is the routing hook for :func:`pyqcu.test.driver.run_row`: the driver
    drives these tags and leaves the tier-0 tags to the built-in executor.
    """
    declaration = BY_TAG.get(str(tag if tag is not None else ""))
    if declaration is None:
        return None
    return declaration.backend, declaration.native_kind


def parameters(tag: Any) -> dict[str, Any]:
    """Fixed execution parameters of ``tag`` (empty for a non-matrix tag)."""
    declaration = BY_TAG.get(str(tag if tag is not None else ""))
    if declaration is None:
        return {}
    return {
        "group": declaration.group,
        "kind": declaration.kind,
        "native_kind": declaration.native_kind,
        "backend": declaration.backend,
        "dtype": declaration.dtype,
        "solver": declaration.solver,
        "mg_levels": int(declaration.mg_levels),
        "restart": int(declaration.restart),
        "dslash": declaration.dslash,
        "ranks": int(declaration.ranks),
        "grid": list(declaration.grid),
        "nodes": int(declaration.nodes),
        "ranks_per_node": (int(declaration.ranks // declaration.nodes)
                           if declaration.nodes else None),
        "requires_input": bool(declaration.requires_input),
        "evidence": declaration.evidence,
        "blocked_by": declaration.blocked_by,
        "notes": declaration.notes,
    }


def row_overrides(tag: Any) -> dict[str, Any]:
    """Ledger overrides for one L48 row (empty for a non-matrix tag).

    ``pyqcu.test.common.build_rows`` applies these so a row says what the
    matrix declares - ranks, grid, dtype, solver, levels, restart - instead of
    the run-level defaults of an unrelated flag.
    """
    payload = parameters(tag)
    if not payload:
        return {}
    return {
        "group": payload["group"],
        "evidence": payload["evidence"],
        "blocked_by": payload["blocked_by"],
        "dtype": payload["dtype"],
        "ranks": payload["ranks"],
        "lattice_grid": payload["grid"],
        "nodes": payload["nodes"],
        "solver": payload["solver"],
        "mg_levels": payload["mg_levels"],
        "restart": payload["restart"],
        "dslash": payload["dslash"],
    }


# ---------------------------------------------------------------------------
# retirement of the L48 CG group
# ---------------------------------------------------------------------------
def is_l48_lattice(lattice: Any) -> bool:
    """True when ``lattice`` is the production L48 geometry."""
    try:
        values = tuple(int(value) for value in lattice)
    except (TypeError, ValueError):
        return False
    return values == LATTICE


def is_cg_tag(tag: Any) -> bool:
    """True for a ``.../solve/cg`` tag (``bistabcg`` is *not* CG)."""
    return str(tag if tag is not None else "").split("/")[-1].strip().lower() \
        == "cg"


def retired_cg_tags() -> tuple[str, ...]:
    """Tier-0 CG tags kept outside the matrix (documented, not deleted)."""
    return CG_TAGS


def drop_retired(cases: Iterable[Any], lattice: Any) -> tuple[Any, ...]:
    """Drop the retired CG group from an L48 plan; other lattices are untouched.

    The tier-0 CG cases stay in :mod:`pyqcu.test.units` and remain selectable
    on the ``8x8x8x16`` quick lattice.  What this function removes is their
    *escalation to L48*, which is the retired group.
    """
    items = tuple(cases)
    if not is_l48_lattice(lattice):
        return items
    return tuple(case for case in items if not is_cg_tag(getattr(case, "tag", "")))


# ---------------------------------------------------------------------------
# launcher hook (data only: no process is started here)
# ---------------------------------------------------------------------------
def formal_command(declaration: MatrixCase, *, python: str = "python3",
                   threads: int | None = None, gauge_file: Any = None,
                   source_file: Any = None, outdir: Any = None) -> list[str]:
    """The ``pyqcu.test`` payload one MPI rank of ``declaration`` executes."""
    from . import launch

    return launch.formal_payload(
        only=declaration.tag, ranks=declaration.ranks,
        grid=declaration.grid, python=python, threads=threads,
        gauge_file=gauge_file, source_file=source_file, outdir=outdir)


def launch_plan(tag: Any, *, mpirun: str = "mpirun", python: str = "python3",
                threads: int | None = None,
                gauge_file: Any = None, source_file: Any = None,
                outdir: Any = None,
                launcher: str | None = None,
                allocation: Any = None,
                rankfile: Any = None,
                job_id: Any = None,
                cores_per_rank: int | None = None,
                drun: str | None = None) -> dict[str, Any]:
    """The launch plan of ``tag``: a declaration, or a real ``drun``/``mpirun``.

    Returns a mapping, not a process.  Three shapes:

    * no ``launcher``/``allocation``/``rankfile`` - the **declaration** used by
      ``--mode list`` and by ``driver.run_row``.  ``kind="mpirun"``,
      ``mpi_launch=False``, ``scheduler_required=False``, ``rankfile=None``,
      ``exec_hosts=None``: the plan names the numbers (2 nodes, 32 ranks, grid
      ``2x2x2x4``) without pretending to own an allocation.
    * ``allocation=...`` (a ``CCS_ALLOC_FILE``) - a **real plan** built by
      :mod:`pyqcu.test.launch`: the nodes come from the file, the rankfile rows
      (``rank N=<host> slot=<first>-<last>``, 18 cores per rank for the
      verified 288-core/16-rank node) are validated for
      ``prod(grid)==ranks``, ``ranks == nodes*ranks_per_node``, even rank-local
      extents and non-overlapping slots, and ``mpi_launch`` is ``True``.
      ``kind`` is ``launcher`` (``"auto"``/``None`` -> the verified ``drun``
      route of job ``2155536``) or ``"mpirun"`` for the ``-rf`` rankfile form.
    * ``rankfile=...`` only - a real plan whose placement cannot be verified
      (no hosts, no rows): ``validated=False``.  The verified ``drun`` route
      records ``-R core=<width>``, so pass ``cores_per_rank`` as well or ask
      for ``launcher="mpirun"`` and its ``-rf`` form.

    A cross-node tag with an explicit ``launcher`` but no allocation and no
    rankfile raises :class:`pyqcu.test.launch.LaunchError` ("refusing a
    cross-node launch"); the matrix never guesses hosts
    (``docs/pyqcu_test_l48_remote.md`` §5.3).

    A verified two-node **KUPL harness** run exists (job ``2155536`` via
    ``drun -mpi pmix``), but it is not a ``pyqcu.test`` launch; this hook never
    starts one and the formal matrix rows remain ``UNVERIFIED``.
    """
    from . import launch

    declaration = get(tag)
    ranks_per_node = (declaration.ranks // declaration.nodes
                      if declaration.nodes else RANKS_PER_NODE)
    payload = formal_command(declaration, python=python, threads=threads,
                             gauge_file=gauge_file, source_file=source_file,
                             outdir=outdir)
    command = [
        str(mpirun), "-np", str(declaration.ranks),
        "--map-by", f"ppr:{ranks_per_node}:node",
        "--bind-to", "none",
        *payload,
    ]
    plan: dict[str, Any] = {
        "kind": "mpirun",
        "launcher": "mpirun",
        "mpi_launch": False,
        "command": command,
        "command_line": shlex.join(command),
        "nodes": declaration.nodes,
        "ranks": declaration.ranks,
        "ranks_per_node": ranks_per_node,
        "grid": list(declaration.grid),
        "lattice": list(LATTICE),
        "rankfile": None,
        "rankfile_rows": [],
        "rankfile_text": None,
        "hostfile": None,
        "exec_hosts": None,
        "cores_per_rank": None,
        "allocation_file": None,
        "job_id": (str(job_id) if job_id not in (None, "") else None),
        "task_replica": None,
        "environment": {},
        "scheduler_required": False,
        "validated": False,
        "requires": (
            "declaration only: pass allocation=... ($CCS_ALLOC_FILE) or "
            "rankfile=... ($CCS_DRUN_RANK_FILE) for an executable plan; the "
            "verified cross-node route is 'drun -mpi pmix' with the rankfile, "
            "not ssh (docs/pyqcu_test_l48_remote.md 0.1.6)"),
        "group": declaration.group,
        "evidence": declaration.evidence,
    }
    if allocation is None and rankfile is None and launcher in (None, ""):
        return plan
    built = launch.build_plan(
        payload,
        nodes=declaration.nodes or 1,
        ranks=declaration.ranks,
        ranks_per_node=ranks_per_node,
        grid=declaration.grid,
        lattice=LATTICE,
        cores_per_rank=cores_per_rank,
        allocation=allocation,
        rankfile=rankfile,
        job_id=job_id,
        launcher=(launcher or "auto"),
        drun=(drun or launch.DEFAULT_DRUN),
        mpirun=mpirun,
    )
    plan.update(built)
    # The matrix, not the run flags, owns the case's identity and evidence.
    plan["group"] = declaration.group
    plan["evidence"] = declaration.evidence
    return plan


# ---------------------------------------------------------------------------
# payload and validation
# ---------------------------------------------------------------------------
def plan_payload() -> dict[str, Any]:
    """JSON-safe document of the whole matrix (``--list --json``)."""
    return {
        "status": "DEFINED" if all(
            group.evidence == "DEFINED" for group in GROUPS) else "UNVERIFIED",
        "tier": TIER,
        "lattice": list(LATTICE),
        "local_lattice": list(LOCAL_LATTICE),
        "grid": list(GRID),
        "ranks": RANKS,
        "nodes": NODES,
        "ranks_per_node": RANKS_PER_NODE,
        "groups": [group.as_dict() for group in GROUPS],
        "cases": [declaration.as_dict() for declaration in DECLARATIONS],
        "case_tags": [declaration.tag for declaration in DECLARATIONS],
        "grid_variants": [variant.as_dict() for variant in GRID_VARIANTS],
        "rejected_grid_variants": [variant.as_dict()
                                   for variant in REJECTED_GRID_VARIANTS],
        "forbidden_groups": list(FORBIDDEN_GROUPS),
        "retired": dict(RETIREMENT),
        "cg_tier0_tags": list(CG_TAGS),
    }


def validate() -> None:
    """Assert the matrix invariants; raise ``ValueError`` on any violation.

    Called by ``--check`` and by the focused tests.  It is deliberately strict:
    a manifest that grows a Strict-MG or CG entry, an odd rank grid, or a
    mismatched level/restart pair must fail loudly instead of being listed.
    """
    problems: list[str] = []
    names = tuple(group.name for group in GROUPS)
    if names != GROUP_NAMES:
        problems.append(f"group order {names} != {GROUP_NAMES}")
    for group in GROUPS:
        if group.name in FORBIDDEN_GROUPS:
            problems.append(f"group {group.name!r} is forbidden")
    for declaration in DECLARATIONS:
        if declaration.group in FORBIDDEN_GROUPS:
            problems.append(
                f"case {declaration.tag!r} belongs to forbidden group "
                f"{declaration.group!r}")
        if "strict" in declaration.tag.lower():
            problems.append(f"case {declaration.tag!r} is Strict-MG")
        if is_cg_tag(declaration.tag):
            problems.append(f"case {declaration.tag!r} is CG")
    bicgstab = tuple(case for case in DECLARATIONS
                     if case.group == "bicgstab")
    if {case.dtype for case in bicgstab} != set(policy.DTYPE_CHOICES):
        problems.append(
            f"BiCGStab dtypes {sorted(case.dtype for case in bicgstab)} != "
            f"{list(policy.DTYPE_CHOICES)}")
    legacy = tuple(case for case in DECLARATIONS if case.group == "legacy_mg")
    if {case.dtype for case in legacy} != set(policy.DTYPE_CHOICES):
        problems.append(
            f"Legacy-MG dtypes {sorted(case.dtype for case in legacy)} != "
            f"{list(policy.DTYPE_CHOICES)}")
    for case in DECLARATIONS:
        if case.group == "bicgstab" and (int(case.mg_levels) != 1
                                         or int(case.restart) != 0):
            problems.append(f"{case.tag}: BiCGStab is not levels=1/restart=0")
        if case.group == "legacy_mg" and (int(case.mg_levels) != 2
                                          or int(case.restart) != 5):
            problems.append(f"{case.tag}: Legacy-MG is not levels=2/restart=5")
        if math.prod(case.grid) != case.ranks:
            problems.append(f"{case.tag}: grid does not match ranks")
        if case.ranks % RANKS_PER_NODE:
            problems.append(f"{case.tag}: ranks are not a multiple of "
                            f"{RANKS_PER_NODE}")
    if not any(case.requires_input for case in DECLARATIONS):
        problems.append("the input/IO group declares no real-input case")
    if not any(variant.name == "2x2x2x4" and variant.status == "VALID"
               for variant in GRID_VARIANTS):
        problems.append("the 2x2x2x4 production grid is missing")
    for variant in GRID_VARIANTS:
        if variant.status != "VALID" or not variant.even:
            problems.append(f"grid variant {variant.name} is not valid")
    for variant in REJECTED_GRID_VARIANTS:
        if variant.status != "REJECTED" or not str(variant.reason).strip():
            problems.append(f"rejected grid {variant.name} lacks a reason")
    if "cg" not in RETIRED_GROUPS:
        problems.append("the L48 CG retirement record is missing")
    if RETIREMENT.get("status") != "RETIRED":
        problems.append("the L48 CG retirement is not marked RETIRED")
    built = cases()
    if len(built) != len(DECLARATIONS):
        problems.append(
            f"cases() returned {len(built)} cases for "
            f"{len(DECLARATIONS)} declarations")
    for case in built:
        if int(case.tier) != TIER:
            problems.append(f"{case.tag}: tier {case.tier} != {TIER}")
        if not case.pinned:
            problems.append(f"{case.tag}: the L48 geometry is not pinned")
        if not is_l48_lattice(case.lattice):
            problems.append(f"{case.tag}: lattice is not L48")
    if problems:
        raise ValueError("; ".join(problems))
