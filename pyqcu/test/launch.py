"""Cross-node launch planning for the formal L48 two-node matrix.

Why this module exists
----------------------
:func:`pyqcu.test.driver.run_row` runs the native call **in this process** (one
Python process = one MPI rank), so the port has never been able to start the
32 ranks of ``docs/pyqcu_test_l48_remote.md`` §5.1.  §5.3 names the missing
piece: the scheduler allocation has to become a *rankfile*, and that rankfile
has to reach the launcher.  §0.1.6 of the same note and §7.5 of
``docs/kupl_remote_l48_experience_20261117.md`` record the launcher that
actually worked on 2026-11-17 (job ``2155536``)::

    export CCS_DRUN_RANK_FILE=$OUT/logs/$JOBTAG-rankfile
    /opt/batch/agent/tools/drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v -- \
        python3 -B -m pyqcu.test --mode solve --l48-two-node ...

``drun`` warns that ``-N/-nn/-rpn/-nl/-R`` are ignored when
``CCS_DRUN_RANK_FILE`` is set: the *rankfile* is the real placement.  It must
therefore exist, hold exactly ``nodes * ranks_per_node`` rows, and give every
rank a disjoint physical slot range inside its node's allocation - 18 cores per
rank for the verified 2x16 shape (``288 / 16``), **not** 36.

What this module produces
-------------------------
Pure data plus one file writer; no process is ever started here:

* :func:`parse_allocation` / :func:`read_allocation` read the documented
  ``CCS_ALLOC_FILE`` format (one ``host total_cores rank_count`` per line);
* :func:`rankfile_rows` turns that into ``rank N=<host> slot=<first>-<last>``
  rows with non-overlapping physical slots, after checking
  ``prod(grid) == ranks``, ``ranks == nodes * ranks_per_node``, even rank-local
  extents and the per-rank core count;
* :func:`write_rankfile` writes those rows (temporary file + ``os.replace``);
* :func:`drun_command` and :func:`mpirun_command` render the two launch forms -
  ``drun -mpi pmix -nn ... -rpn ... -R core=... -v -- <payload>`` (the verified
  route) and ``mpirun ... -rf <rankfile> --mca rmaps_rank_file_physical true``
  (the vendor template of §7.4);
* :func:`build_plan` refuses a cross-node plan that has neither an allocation
  nor a rankfile: *"no scheduler allocation file ... refusing a cross-node
  launch"* (§5.3).  Guessing hosts, falling back to one node or to a smaller
  rank count is exactly the failure mode the design forbids.

The command line
----------------
``python -m pyqcu.test.launch {rankfile,argv,plan}`` exposes the same three
steps to the submission script
(``pyqcu/testing/kupl/submit_pyqcu_test.sh``), so the job script never
re-derives ranks, hosts or slots from prose:

``rankfile`` writes the rankfile, ``argv`` prints the launch arguments one per
line (``mapfile``-friendly, no ``eval``), ``plan`` prints the plan as JSON.
A refusal exits ``2``.

Import rules
------------
Standard library only, at import and in the CLI.  :mod:`pyqcu.test.l48` is
imported lazily inside :func:`main` (its geometry defaults come from the
matrix, so they are not duplicated here), and ``l48`` imports this module
lazily as well - which is why ``python -m pyqcu.test.launch`` never finds its
own module already in ``sys.modules`` (that would give the CLI a second
``LaunchError`` class).  ``--list``/``--mode list`` never import this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ALLOC_ENV",
    "Allocation",
    "DEFAULT_DRUN",
    "DEFAULT_LAUNCHER",
    "DEFAULT_MPIRUN",
    "JOB_ID_ENV",
    "LAUNCHERS",
    "LaunchError",
    "RANKFILE_ENV",
    "RankfileRow",
    "TASK_REPLICA_ENV",
    "allocation_from_env",
    "build_plan",
    "check_slot_overlap",
    "drun_command",
    "formal_payload",
    "main",
    "mpirun_command",
    "parse_allocation",
    "rankfile_rows",
    "read_allocation",
    "render_rankfile",
    "validate_geometry",
    "write_rankfile",
]

#: Scheduler variable: a file of ``host total_cores rank_count`` lines.
ALLOC_ENV = "CCS_ALLOC_FILE"
#: Variable ``drun`` reads the rankfile from; setting it also selects the CCS
#: launcher, which is why the verified path works without ssh.
RANKFILE_ENV = "CCS_DRUN_RANK_FILE"
#: Job id, used to name the rankfile (``<job_id>-rankfile``).
JOB_ID_ENV = "CCS_JOB_ID"
#: Number of tasks of the job (the manual's ``mpirun -np`` source).  Recorded,
#: never required: the rank count comes from ``nodes * ranks_per_node``.
TASK_REPLICA_ENV = "CCS_TASK_REPLICA"

#: The verified launcher.  ``drun`` only exists on the compute nodes.
DEFAULT_DRUN = "/opt/batch/agent/tools/drun"
DEFAULT_MPIRUN = "mpirun"
#: Default launcher: the ``drun`` route of job ``2155536``, not ``plm_rsh``.
DEFAULT_LAUNCHER = "drun"
LAUNCHERS = ("drun", "mpirun")

#: Executable part of a rankfile row, kept in one place for the reader.
_RANKFILE_TEMPLATE = "rank {rank}={host} slot={first}-{last}"


class LaunchError(ValueError):
    """A launch plan that cannot be stated truthfully.

    Raised for a malformed allocation, a geometry mismatch, a rankfile that
    would overlap, and for the explicit refusal of a cross-node plan without an
    allocation or rankfile.  It is a ``ValueError`` so callers that already
    treat bad options as usage errors keep working.
    """


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _positive_int(value: Any, label: str) -> int:
    """``value`` as an integer >= 1, else :class:`LaunchError`."""
    if isinstance(value, bool) or value is None:
        raise LaunchError(f"{label} must be a positive integer, got {value!r}")
    if isinstance(value, float) and not float(value).is_integer():
        raise LaunchError(f"{label} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise LaunchError(
            f"{label} must be an integer, got {value!r}") from None
    if number < 1:
        raise LaunchError(f"{label} must be >= 1, got {number}")
    return number


def _shape(value: Any, label: str) -> tuple[int, int, int, int]:
    """A four-positive-integer shape (grid or lattice)."""
    try:
        items = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise LaunchError(
            f"{label} must be four integers, got {value!r}") from None
    if len(items) != 4 or any(item < 1 for item in items):
        raise LaunchError(
            f"{label} must be four positive integers, got {value!r}")
    return items  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CCS_ALLOC_FILE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Allocation:
    """One ``CCS_ALLOC_FILE`` line: a host, its cores and its rank slots.

    The documented format is ``host total_cores rank_count``; the per-rank core
    width is ``total_cores // rank_count`` (288 / 16 = 18 for the verified
    L48 allocation).  ``total_cores`` is what the job was *given*, not the
    node's physical CPU count, so the division is meaningful.
    """

    host: str
    total_cores: int
    rank_count: int

    def __post_init__(self) -> None:
        host = str(self.host).strip()
        if not host:
            raise LaunchError("allocation line has an empty host name")
        cores = _positive_int(self.total_cores, f"allocation {host}: cores")
        ranks = _positive_int(self.rank_count, f"allocation {host}: rank_count")
        if cores % ranks:
            raise LaunchError(
                f"allocation {host}: total_cores {cores} is not divisible by "
                f"rank_count {ranks}, so there is no integer per-rank core "
                f"width")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "total_cores", cores)
        object.__setattr__(self, "rank_count", ranks)

    @property
    def cores_per_rank(self) -> int:
        """Width of one rank's physical slot range on this host."""
        return self.total_cores // self.rank_count

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description."""
        return {
            "host": self.host,
            "total_cores": self.total_cores,
            "rank_count": self.rank_count,
            "cores_per_rank": self.cores_per_rank,
        }


def parse_allocation(text: str) -> tuple[Allocation, ...]:
    """Parse ``CCS_ALLOC_FILE`` content (``host total_cores rank_count``).

    Blank lines and ``#`` comments are skipped; a repeated host is refused
    (two lines for one node would describe overlapping slot ranges), and so is
    an empty document - a plan built on "no allocation" is not a plan.
    """
    allocations: list[Allocation] = []
    seen: dict[str, int] = {}
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise LaunchError(
                f"allocation line {number}: expected 'host total_cores "
                f"rank_count', got {raw.strip()!r}")
        host = fields[0]
        if host in seen:
            raise LaunchError(
                f"allocation line {number}: host {host!r} already appeared on "
                f"line {seen[host]}")
        seen[host] = number
        allocations.append(Allocation(host, fields[1], fields[2]))
    if not allocations:
        raise LaunchError(
            "the allocation file is empty: no 'host total_cores rank_count' "
            "line was found")
    return tuple(allocations)


def read_allocation(path: Any) -> tuple[Allocation, ...]:
    """Read and parse an allocation file (``CCS_ALLOC_FILE``)."""
    target = Path(str(path)).expanduser()
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise LaunchError(
            f"cannot read the allocation file {target}: {error}") from error
    try:
        return parse_allocation(text)
    except LaunchError as error:
        raise LaunchError(f"{target}: {error}") from None


def allocation_from_env(
        environ: Mapping[str, str] | None = None) -> str | None:
    """``$CCS_ALLOC_FILE`` when it is set and non-empty, else ``None``."""
    environment = os.environ if environ is None else environ
    value = str(environment.get(ALLOC_ENV) or "").strip()
    return value or None


def _as_allocations(value: Any) -> tuple[tuple[Allocation, ...] | None, str | None]:
    """Coerce ``allocation`` to ``(nodes, allocation_file)``.

    Accepts a path (read now), a multi-line document, a sequence of
    :class:`Allocation`/mappings/3-tuples, or ``None``.
    """
    if value is None:
        return None, None
    if isinstance(value, Allocation):
        return (value,), None
    if isinstance(value, (str, Path)):
        text = str(value)
        if "\n" in text:
            return parse_allocation(text), None
        return read_allocation(text), text
    try:
        items = tuple(value)
    except TypeError:
        raise LaunchError(
            f"allocation must be a path, a document or a sequence of nodes, "
            f"got {value!r}") from None
    nodes: list[Allocation] = []
    for item in items:
        if isinstance(item, Allocation):
            nodes.append(item)
        elif isinstance(item, Mapping) and "host" in item:
            nodes.append(Allocation(item["host"], item.get("total_cores", 0),
                                    item.get("rank_count", 0)))
        else:
            try:
                host, cores, ranks = item
            except (TypeError, ValueError):
                raise LaunchError(
                    f"allocation entry {item!r} is not an Allocation, a "
                    f"mapping or a (host, cores, ranks) triple") from None
            nodes.append(Allocation(host, cores, ranks))
    if not nodes:
        raise LaunchError("the allocation lists no node")
    return tuple(nodes), None


# ---------------------------------------------------------------------------
# geometry and rankfile rows
# ---------------------------------------------------------------------------
def validate_geometry(*, ranks: Any, nodes: Any, ranks_per_node: Any,
                      grid: Any = None, lattice: Any = None) -> None:
    """Check the L48 geometry contract shared by the plan and its rankfile.

    * ``ranks == nodes * ranks_per_node`` - the two-node shape is exactly
      ``16 ranks x 2 nodes``; one of the three numbers is not a free variable.
    * ``prod(grid) == ranks`` - the design's
      ``docs/pyqcu_test_l48_remote.md`` §5.1 rule.
    * every rank-local extent is even **and** divides the lattice: the parity
      decomposition of the native runners has no meaning otherwise.
    """
    world = _positive_int(ranks, "ranks")
    node_count = _positive_int(nodes, "nodes")
    per_node = _positive_int(ranks_per_node, "ranks_per_node")
    if world != node_count * per_node:
        raise LaunchError(
            f"ranks {world} != nodes {node_count} * ranks_per_node {per_node}")
    if grid is None:
        return
    shape = _shape(grid, "grid")
    if math.prod(shape) != world:
        raise LaunchError(
            f"prod(grid) {list(shape)} = {math.prod(shape)} != ranks {world}")
    if lattice is None:
        return
    extents = _shape(lattice, "lattice")
    for axis in range(4):
        if extents[axis] % shape[axis]:
            raise LaunchError(
                f"grid {list(shape)} does not divide lattice {list(extents)} "
                f"on axis {axis}")
        local = extents[axis] // shape[axis]
        if local % 2:
            raise LaunchError(
                f"grid {list(shape)} leaves an odd rank-local extent "
                f"{local} on axis {axis} (lattice {list(extents)}); the parity "
                f"decomposition is undefined")


@dataclass(frozen=True)
class RankfileRow:
    """One ``rank N=<host> slot=<first>-<last>`` line."""

    rank: int
    host: str
    first_slot: int
    last_slot: int

    def __post_init__(self) -> None:
        rank = _positive_int(int(self.rank) + 1, "rankfile rank") - 1
        first = int(self.first_slot)
        last = int(self.last_slot)
        if not str(self.host).strip():
            raise LaunchError(f"rankfile rank {rank}: empty host")
        if first < 0 or last < first:
            raise LaunchError(
                f"rankfile rank {rank}: slot range {first}-{last} is not a "
                f"valid physical range")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "host", str(self.host).strip())
        object.__setattr__(self, "first_slot", first)
        object.__setattr__(self, "last_slot", last)

    @property
    def width(self) -> int:
        """Cores this rank owns (``last_slot - first_slot + 1``)."""
        return self.last_slot - self.first_slot + 1

    def as_line(self) -> str:
        """One rankfile line, in the vendor template's spelling."""
        return _RANKFILE_TEMPLATE.format(
            rank=self.rank, host=self.host, first=self.first_slot,
            last=self.last_slot)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description."""
        return {
            "rank": self.rank,
            "host": self.host,
            "first_slot": self.first_slot,
            "last_slot": self.last_slot,
            "width": self.width,
        }


def _resolve_cores_per_rank(allocations: Sequence[Allocation],
                            cores_per_rank: Any) -> int:
    """Per-rank core width, checked against every node's allocation."""
    if cores_per_rank is None:
        widths = {node.cores_per_rank for node in allocations}
        if len(widths) != 1:
            raise LaunchError(
                f"the allocation gives different per-rank widths "
                f"{sorted(widths)}; pass cores_per_rank explicitly")
        return widths.pop()
    width = _positive_int(cores_per_rank, "cores_per_rank")
    for node in allocations:
        needed = node.rank_count * width
        if needed > node.total_cores:
            raise LaunchError(
                f"{node.host}: {node.rank_count} ranks x {width} cores = "
                f"{needed} exceeds the allocated {node.total_cores} cores")
    return width


def rankfile_rows(
    allocations: Sequence[Allocation],
    *,
    ranks: Any = None,
    ranks_per_node: Any = None,
    cores_per_rank: Any = None,
    grid: Any = None,
    lattice: Any = None,
) -> tuple[RankfileRow, ...]:
    """The rankfile rows of one allocation, with disjoint physical slots.

    Rank ``node_index * ranks_per_node + local`` runs on ``host`` and owns the
    physical slots ``local * width`` .. ``(local + 1) * width - 1``.  That is
    the documented ``total_cores / rank_count`` width (18 for 288/16), so the
    16 ranks of a node cover ``0-287`` and never overlap; the leftover cores of
    a larger allocation stay unused instead of being handed to a rank twice.
    """
    nodes = tuple(allocations)
    if not nodes:
        raise LaunchError("no allocation node was given")
    node_count = len(nodes)
    per_node = (nodes[0].rank_count if ranks_per_node is None
                else _positive_int(ranks_per_node, "ranks_per_node"))
    for node in nodes:
        if node.rank_count != per_node:
            raise LaunchError(
                f"{node.host}: the allocation grants {node.rank_count} ranks, "
                f"the plan asks for {per_node} per node")
    world = node_count * per_node if ranks is None else _positive_int(
        ranks, "ranks")
    if world != node_count * per_node:
        raise LaunchError(
            f"ranks {world} != nodes {node_count} * ranks_per_node {per_node}")
    validate_geometry(ranks=world, nodes=node_count, ranks_per_node=per_node,
                      grid=grid, lattice=lattice)
    width = _resolve_cores_per_rank(nodes, cores_per_rank)

    rows: list[RankfileRow] = []
    slots: dict[str, list[tuple[int, int, int]]] = {}
    for index, node in enumerate(nodes):
        for local in range(per_node):
            first = local * width
            last = first + width - 1
            if last >= node.total_cores:
                raise LaunchError(
                    f"{node.host}: rank {local} needs slots {first}-{last} but "
                    f"only {node.total_cores} cores were allocated")
            rank = index * per_node + local
            rows.append(RankfileRow(rank, node.host, first, last))
            slots.setdefault(node.host, []).append((first, last, rank))
    check_slot_overlap(slots)
    return tuple(rows)


def check_slot_overlap(
        slots: Mapping[str, Sequence[tuple[int, int, int]]]) -> None:
    """Refuse two ranks of one host that claim the same physical core.

    ``slots`` maps a host to its ``(first, last, rank)`` triples; the
    generator never produces an overlap, but a hand-written or foreign
    rankfile must not be accepted either.
    """
    for host, ranges in slots.items():
        ordered = sorted(ranges)
        previous: tuple[int, int, int] | None = None
        for first, last, rank in ordered:
            if previous is not None and first <= previous[1]:
                raise LaunchError(
                    f"{host}: rank {rank} slots {first}-{last} overlap rank "
                    f"{previous[2]} slots {previous[0]}-{previous[1]}")
            previous = (first, last, rank)


def render_rankfile(rows: Iterable[RankfileRow]) -> str:
    """The rankfile text of ``rows`` (trailing newline, ``\\n`` separators)."""
    items = tuple(rows)
    if not items:
        raise LaunchError("a rankfile needs at least one row")
    ranks = [row.rank for row in items]
    if ranks != list(range(len(items))):
        raise LaunchError(
            f"rankfile ranks must be 0..{len(items) - 1} in order, got {ranks}")
    return "".join(f"{row.as_line()}\n" for row in items)


def write_rankfile(path: Any, rows: Iterable[RankfileRow]) -> Path:
    """Write the rankfile atomically and return its path.

    ``drun`` reads ``CCS_DRUN_RANK_FILE``; a partially written file would
    launch a partial rank set, so the text is written to ``<name>.tmp`` and
    renamed into place.
    """
    target = Path(str(path)).expanduser()
    text = render_rankfile(rows)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    except OSError as error:
        raise LaunchError(f"cannot write the rankfile {target}: {error}") from None
    return target


# ---------------------------------------------------------------------------
# launch commands
# ---------------------------------------------------------------------------
def formal_payload(*, only: Any, ranks: Any, grid: Any, python: str = "python3",
                   threads: Any = None, gauge_file: Any = None,
                   source_file: Any = None, outdir: Any = None) -> list[str]:
    """The ``pyqcu.test`` argument list one MPI rank executes.

    Shared by :func:`pyqcu.test.l48.launch_plan` and the ``argv`` subcommand so
    the declared plan and the submitted job cannot drift apart.
    """
    shape = _shape(grid, "grid")
    payload = [str(python), "-m", "pyqcu.test", "--mode", "solve",
               "--l48-two-node", "--only", str(only),
               "--ranks", str(_positive_int(ranks, "ranks")),
               "--lattice-grid", *(str(value) for value in shape)]
    if threads is not None:
        payload.extend(("--threads", str(_positive_int(threads, "threads"))))
    if outdir is not None:
        # Keep rows/reports/solutions/caches of the submitted job in one root
        # instead of the default ``pyqcu-test-runs/`` relative to the cwd.
        payload.extend(("--outdir", str(outdir)))
    if gauge_file is not None:
        payload.extend(("--gauge-file", str(gauge_file)))
    if source_file is not None:
        payload.extend(("--source-file", str(source_file)))
    return payload


def drun_command(argv: Sequence[str], *, nodes: Any, ranks_per_node: Any,
                 cores_per_rank: Any, drun: str = DEFAULT_DRUN,
                 mpi: str = "pmix", verbose: bool = True) -> list[str]:
    """``drun -mpi pmix -nn N -rpn M -R core=C -v -- <payload>``.

    This is the verified shape of job ``2155536``.  ``drun`` ignores
    ``-N/-nn/-rpn/-nl/-R`` once ``CCS_DRUN_RANK_FILE`` is set, so the numbers
    here are *recorded* evidence: the rankfile decides the placement.
    """
    command = [str(drun), "-mpi", str(mpi),
               "-nn", str(_positive_int(nodes, "nodes")),
               "-rpn", str(_positive_int(ranks_per_node, "ranks_per_node")),
               "-R", f"core={_positive_int(cores_per_rank, 'cores_per_rank')}"]
    if verbose:
        command.append("-v")
    command.append("--")
    command.extend(str(item) for item in argv)
    return command


def mpirun_command(argv: Sequence[str], *, rankfile: Any = None, ranks: Any,
                   mpirun: str = DEFAULT_MPIRUN, physical: bool = True,
                   bind: str = "none",
                   forward_env: Sequence[str] = ("PATH", "LD_LIBRARY_PATH"),
                   allow_run_as_root: bool = False) -> list[str]:
    """``mpirun -np N -rf <rankfile> --mca rmaps_rank_file_physical true ...``.

    The vendor template of ``docs/kupl_remote_l48_experience_20261117.md`` §7.4:
    an explicit rankfile, physical slot binding and the HPCKit environment
    forwarded.  It still needs a working PLM; on this account ``plm_rsh`` is
    *not* usable (compute nodes accept the key, then close the session), which
    is why :data:`DEFAULT_LAUNCHER` is ``drun``.  Without a ``rankfile`` the
    command is the plain single-node ``mpirun -np N`` (no ``-rf``, no physical
    mapping): there is nothing to bind to.
    """
    command = [str(mpirun), "-np", str(_positive_int(ranks, "ranks"))]
    if rankfile not in (None, ""):
        command.extend(("-rf", str(rankfile)))
        if physical:
            command.extend(("--mca", "rmaps_rank_file_physical", "true"))
    for name in forward_env:
        command.extend(("-x", str(name)))
    command.extend(("--bind-to", str(bind)))
    if allow_run_as_root:
        command.append("--allow-run-as-root")
    command.extend(str(item) for item in argv)
    return command


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _default_rankfile_name(job_id: Any) -> str:
    """``<job_id>-rankfile`` (the remote scripts' spelling)."""
    text = str(job_id).strip() if job_id is not None else ""
    return f"{text}-rankfile" if text else "pyqcu-test-rankfile"


def build_plan(
    argv: Sequence[str],
    *,
    nodes: Any,
    ranks_per_node: Any,
    ranks: Any = None,
    grid: Any = None,
    lattice: Any = None,
    cores_per_rank: Any = None,
    allocation: Any = None,
    rankfile: Any = None,
    job_id: Any = None,
    launcher: str = "auto",
    drun: str = DEFAULT_DRUN,
    mpirun: str = DEFAULT_MPIRUN,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The executable cross-node plan of one L48 launch.

    ``argv`` is the per-rank payload (:func:`formal_payload`).  ``allocation``
    is a path to ``CCS_ALLOC_FILE``, a multi-line document, or an already
    parsed sequence of :class:`Allocation`; ``rankfile`` is where the rows are
    (or will be) written.  ``environ`` is **opt-in**: when it is given it
    supplies ``CCS_ALLOC_FILE``/``CCS_DRUN_RANK_FILE``/``CCS_JOB_ID``
    defaults and cross-checks ``CCS_TASK_REPLICA`` against the rank count -
    which is what a job container wants and what a unit test must not depend
    on.

    A cross-node plan with neither an allocation nor a rankfile is refused
    (``docs/pyqcu_test_l48_remote.md`` §5.3): no host is guessed and no
    one-node fallback is invented.  A single-node plan may omit both.
    """
    environment = dict(environ or {})
    node_count = _positive_int(nodes, "nodes")
    per_node = _positive_int(ranks_per_node, "ranks_per_node")
    world = (node_count * per_node if ranks is None
             else _positive_int(ranks, "ranks"))
    validate_geometry(ranks=world, nodes=node_count, ranks_per_node=per_node,
                      grid=grid, lattice=lattice)
    payload = [str(item) for item in argv]
    if not payload:
        raise LaunchError("the launch payload is empty")
    if launcher not in ("auto",) + LAUNCHERS:
        raise LaunchError(
            f"unknown launcher {launcher!r}; choose from auto, "
            f"{', '.join(LAUNCHERS)}")

    if allocation is None and environ:
        allocation = allocation_from_env(environment)
    if rankfile in (None, "") and environ:
        rankfile = str(environment.get(RANKFILE_ENV) or "").strip() or None
    if job_id in (None, "") and environ:
        job_id = str(environment.get(JOB_ID_ENV) or "").strip() or None
    # The manual's MPI template sizes ``mpirun -np`` from CCS_TASK_REPLICA; a
    # value that disagrees with ``nodes * ranks_per_node`` is a broken request,
    # not something to average over.
    replica: int | None = None
    raw_replica = str(environment.get(TASK_REPLICA_ENV) or "").strip()
    if raw_replica:
        replica = _positive_int(raw_replica, TASK_REPLICA_ENV)
        if replica != world:
            raise LaunchError(
                f"{TASK_REPLICA_ENV}={replica} disagrees with the plan's "
                f"{world} ranks ({node_count} nodes x {per_node})")

    allocations, allocation_file = _as_allocations(allocation)
    if allocations is None and rankfile in (None, "") and node_count > 1:
        raise LaunchError(
            f"no scheduler allocation file and no rankfile: refusing a "
            f"cross-node launch of {node_count} nodes / {world} ranks.  Pass "
            f"allocation=... (${ALLOC_ENV}) or rankfile=... "
            f"(${RANKFILE_ENV}); the plan must not guess hosts "
            f"(docs/pyqcu_test_l48_remote.md 5.3)")
    kind = DEFAULT_LAUNCHER if launcher == "auto" else str(launcher)

    rows: tuple[RankfileRow, ...] | None = None
    exec_hosts: list[str] | None = None
    if allocations is not None:
        if len(allocations) != node_count:
            raise LaunchError(
                f"the allocation names {len(allocations)} node(s), the plan "
                f"asks for {node_count}")
        rows = rankfile_rows(allocations, ranks=world,
                             ranks_per_node=per_node,
                             cores_per_rank=cores_per_rank, grid=grid,
                             lattice=lattice)
        exec_hosts = [node.host for node in allocations]
        width = _resolve_cores_per_rank(allocations, cores_per_rank)
    else:
        width = (None if cores_per_rank is None
                 else _positive_int(cores_per_rank, "cores_per_rank"))

    if rankfile not in (None, ""):
        resolved_rankfile: str | None = str(rankfile)
    elif rows is not None:
        # A validated allocation always names its rankfile (drun reads it from
        # ``CCS_DRUN_RANK_FILE``); the job scripts use ``<job_id>-rankfile``.
        resolved_rankfile = _default_rankfile_name(job_id)
    else:
        resolved_rankfile = None
    if kind == "drun":
        if width is None:
            raise LaunchError(
                "drun needs the per-rank core width: pass cores_per_rank or "
                f"an allocation file (${ALLOC_ENV})")
        if resolved_rankfile is None:
            raise LaunchError(
                f"drun needs a rankfile: pass rankfile=... (${RANKFILE_ENV}) "
                f"or job_id=... so the name can be derived")
        command = drun_command(payload, nodes=node_count,
                               ranks_per_node=per_node, cores_per_rank=width,
                               drun=drun)
        environment_variables = {RANKFILE_ENV: resolved_rankfile}
    else:
        command = mpirun_command(payload, rankfile=resolved_rankfile,
                                 ranks=world, mpirun=mpirun)
        environment_variables = {}
    if job_id not in (None, ""):
        # ``drun``/the job scripts name the rankfile after the job; recording
        # it keeps the launcher evidence reproducible.
        environment_variables.setdefault(JOB_ID_ENV, str(job_id))

    plan: dict[str, Any] = {
        "kind": kind,
        "launcher": kind,
        "mpi_launch": True,
        "command": command,
        "command_line": shlex.join(command),
        "nodes": node_count,
        "ranks": world,
        "ranks_per_node": per_node,
        "grid": (None if grid is None else list(_shape(grid, "grid"))),
        "lattice": (None if lattice is None
                    else list(_shape(lattice, "lattice"))),
        "cores_per_rank": width,
        "rankfile": resolved_rankfile,
        "rankfile_rows": ([row.as_dict() for row in rows]
                          if rows is not None else []),
        "rankfile_text": render_rankfile(rows) if rows is not None else None,
        "hostfile": None,
        "exec_hosts": exec_hosts,
        "allocation_file": allocation_file,
        "job_id": (str(job_id) if job_id not in (None, "") else None),
        "task_replica": replica,
        "environment": environment_variables,
        "scheduler_required": True,
        "validated": rows is not None,
        "requires": _plan_requirement(
            kind, has_rankfile=resolved_rankfile is not None),
    }
    return plan


def _plan_requirement(kind: str, *, has_rankfile: bool = True) -> str:
    """The one-line truth about what an executable plan still needs."""
    if kind == "drun":
        return (
            "run inside the CCS allocation; write the rankfile "
            f"({RANKFILE_ENV}) and export it before drun - drun ignores "
            "-nn/-rpn/-R when the rankfile is set, so the rankfile is the "
            "placement (docs/pyqcu_test_l48_remote.md 0.1.6)")
    if not has_rankfile:
        return (
            "single-node launch: mpirun needs no rankfile, but the ranks are "
            "only correct if MPI_COMM_WORLD is the declared 32 - check the "
            "report's mpi_size/lattice_grid")
    return (
        "run inside the CCS allocation with the rankfile passed as -rf; the "
        "PLM must be able to start remote ranks (plm_rsh is not usable on "
        "this account, use the drun route - docs/kupl_remote_l48_experience_"
        "20261117.md 7.4)")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by every subcommand (defaults: the L48 matrix)."""
    parser.add_argument(
        "--alloc-file", default=None, metavar="PATH",
        help=f"CCS allocation file (default: ${ALLOC_ENV})")
    parser.add_argument(
        "--rankfile", default=None, metavar="PATH",
        help=f"rankfile to write/use (default: ${RANKFILE_ENV})")
    parser.add_argument(
        "--job-id", default=None, metavar="ID",
        help=f"job id, used to name the rankfile (default: ${JOB_ID_ENV})")
    parser.add_argument("--nodes", type=int, default=None, metavar="N",
                        help="compute nodes (default: the L48 matrix)")
    parser.add_argument("--ranks", type=int, default=None, metavar="N",
                        help="total ranks (default: nodes * ranks-per-node)")
    parser.add_argument("--ranks-per-node", type=int, default=None, metavar="N",
                        help="ranks per node (default: the L48 matrix)")
    parser.add_argument("--cores-per-rank", type=int, default=None, metavar="N",
                        help="physical cores per rank (default: from the "
                             "allocation, 288/16 = 18)")
    parser.add_argument("--grid", nargs=4, type=int, default=None,
                        metavar="N", help="rank grid (default: 2 2 2 4)")
    parser.add_argument("--lattice", nargs=4, type=int, default=None,
                        metavar="N", help="lattice (default: 48 48 48 96)")


def _add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that only make sense when a command is rendered."""
    parser.add_argument("--launcher", choices=("auto",) + LAUNCHERS,
                        default="auto", help="launch route (default: drun)")
    parser.add_argument("--drun", default=DEFAULT_DRUN, metavar="PATH",
                        help=f"drun executable (default: {DEFAULT_DRUN})")
    parser.add_argument("--mpirun", default=DEFAULT_MPIRUN, metavar="PATH",
                        help=f"mpirun executable (default: {DEFAULT_MPIRUN})")
    parser.add_argument("--only", default=None, metavar="TAG",
                        help="--only value of the formal command "
                             "(default: every L48 matrix row)")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="OpenMP threads per rank (default: 18)")
    parser.add_argument("--python", default="python3", metavar="PATH",
                        help="Python of the formal command (default: python3)")
    parser.add_argument("--outdir", default=None, metavar="DIR",
                        help="--outdir of the formal command (ledger root)")
    parser.add_argument("--gauge-file", default=None, metavar="PATH",
                        help="--gauge-file of the formal command")
    parser.add_argument("--source-file", default=None, metavar="PATH",
                        help="--source-file of the formal command")


def _l48_defaults() -> dict[str, Any]:
    """Geometry/thread defaults from :mod:`pyqcu.test.l48` (lazy import)."""
    from . import l48

    return {
        "nodes": l48.NODES,
        "ranks": l48.RANKS,
        "ranks_per_node": l48.RANKS_PER_NODE,
        "grid": list(l48.GRID),
        "lattice": list(l48.LATTICE),
        "only": "kupl/l48",
        "threads": 18,
    }


def _resolve_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Fill a parsed namespace from l48 defaults and the job environment."""
    defaults = _l48_defaults()
    geometry = {
        "nodes": int(args.nodes if args.nodes is not None
                     else defaults["nodes"]),
        "ranks": int(args.ranks if args.ranks is not None
                     else defaults["ranks"]),
        "ranks_per_node": int(args.ranks_per_node
                              if args.ranks_per_node is not None
                              else defaults["ranks_per_node"]),
        "grid": list(args.grid if args.grid is not None else defaults["grid"]),
        "lattice": list(args.lattice if args.lattice is not None
                        else defaults["lattice"]),
    }
    environ = {name: os.environ[name] for name in
               (ALLOC_ENV, RANKFILE_ENV, JOB_ID_ENV, TASK_REPLICA_ENV)
               if os.environ.get(name)}
    allocation = args.alloc_file or allocation_from_env(environ)
    rankfile = args.rankfile or environ.get(RANKFILE_ENV) or None
    job_id = args.job_id or environ.get(JOB_ID_ENV) or None
    return {
        "geometry": geometry,
        "allocation": allocation,
        "rankfile": rankfile,
        "job_id": job_id,
        "environ": environ,
    }


def _run_rankfile(args: argparse.Namespace) -> int:
    """Validate the allocation and write the rankfile."""
    resolved = _resolve_arguments(args)
    geometry = resolved["geometry"]
    if resolved["allocation"] is None:
        raise LaunchError(
            f"no allocation file: pass --alloc-file or export ${ALLOC_ENV}; "
            f"refusing to invent slots for {geometry['nodes']} nodes")
    allocations, _ = _as_allocations(resolved["allocation"])
    rows = rankfile_rows(allocations, ranks=geometry["ranks"],
                         ranks_per_node=geometry["ranks_per_node"],
                         cores_per_rank=args.cores_per_rank,
                         grid=geometry["grid"], lattice=geometry["lattice"])
    target = (resolved["rankfile"]
              or _default_rankfile_name(resolved["job_id"]))
    path = write_rankfile(target, rows)
    hosts = [node.host for node in allocations]
    print(f"rankfile={path} rows={len(rows)} hosts={','.join(hosts)} "
          f"ranks={geometry['ranks']} "
          f"cores_per_rank={rows[0].width if rows else 0}")
    return 0


def _launch_payload(args: argparse.Namespace, geometry: Mapping[str, Any]
                    ) -> list[str]:
    """The formal ``pyqcu.test`` payload of one rank."""
    only = args.only or _l48_defaults()["only"]
    threads = (args.threads if args.threads is not None
               else _l48_defaults()["threads"])
    return formal_payload(only=only, ranks=geometry["ranks"],
                          grid=geometry["grid"], python=args.python,
                          threads=threads, gauge_file=args.gauge_file,
                          source_file=args.source_file, outdir=args.outdir)


def _build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The plan of a parsed namespace (shared by ``argv`` and ``plan``)."""
    resolved = _resolve_arguments(args)
    geometry = resolved["geometry"]
    return build_plan(
        _launch_payload(args, geometry),
        nodes=geometry["nodes"], ranks=geometry["ranks"],
        ranks_per_node=geometry["ranks_per_node"], grid=geometry["grid"],
        lattice=geometry["lattice"], cores_per_rank=args.cores_per_rank,
        allocation=resolved["allocation"], rankfile=resolved["rankfile"],
        job_id=resolved["job_id"], launcher=args.launcher, drun=args.drun,
        mpirun=args.mpirun, environ=resolved["environ"])


def _run_argv(args: argparse.Namespace) -> int:
    """Print the launch arguments, one per line (``mapfile``-friendly)."""
    plan = _build_from_args(args)
    for item in plan["command"]:
        print(item)
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    """Print the plan as JSON."""
    print(json.dumps(_build_from_args(args), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m pyqcu.test.launch`` entry point (exit ``2`` on refusal)."""
    parser = argparse.ArgumentParser(
        prog="python -m pyqcu.test.launch",
        description=("Plan the cross-node L48 launch: validate a CCS "
                     "allocation, write its rankfile and render the drun / "
                     "mpirun command.  Starts nothing."))
    subparsers = parser.add_subparsers(dest="action", required=True)

    writer = subparsers.add_parser(
        "rankfile", help="validate the allocation and write the rankfile")
    _add_geometry_arguments(writer)
    writer.set_defaults(handler=_run_rankfile)

    for name, handler, text in (
            ("argv", _run_argv,
             "print the launch arguments, one per line"),
            ("plan", _run_plan, "print the launch plan as JSON")):
        sub = subparsers.add_parser(name, help=text)
        _add_geometry_arguments(sub)
        _add_launch_arguments(sub)
        sub.set_defaults(handler=handler)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except LaunchError as error:
        print(f"pyqcu.test.launch: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via -m
    raise SystemExit(main())
