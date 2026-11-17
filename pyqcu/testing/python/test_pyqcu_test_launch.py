"""Focused tests for :mod:`pyqcu.test.launch` - the cross-node launch planner.

What is pinned here
-------------------
``docs/pyqcu_test_l48_remote.md`` §5.3 left the port without a launcher; §0.1.6
and ``docs/kupl_remote_l48_experience_20261117.md`` §7.5 then recorded the path
that worked on 2026-11-17 (job ``2155536``): a rankfile derived from
``CCS_ALLOC_FILE``, exported as ``CCS_DRUN_RANK_FILE``, launched with
``drun -mpi pmix``.  These tests keep that translation honest:

* **allocation** - the documented ``host total_cores rank_count`` lines parse,
  and malformed/duplicated/divisibility-breaking lines are refused;
* **rankfile** - exactly ``nodes * ranks_per_node`` rows (32 for the L48
  matrix), every rank owning ``total_cores / rank_count = 18`` disjoint
  physical slots, for the production geometry and for other node counts;
* **geometry** - ``prod(grid) == ranks``, ``ranks == nodes * ranks_per_node``,
  even rank-local extents and a per-rank width that fits the allocation are
  *checked*, not assumed;
* **commands** - the ``drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v --`` shape of
  the verified run and the ``mpirun -rf ... --mca rmaps_rank_file_physical``
  vendor template, both carrying the formal ``pyqcu.test`` payload;
* **refusal** - a cross-node plan with neither an allocation nor a rankfile is
  refused instead of guessing hosts or shrinking to one node;
* **stdlib-only** - importing the module (and its CLI) pulls in no torch,
  numpy, mpi4py, pytest or ``pyqcu.testing``;
* **submission script** - ``pyqcu/testing/kupl/submit_pyqcu_test.sh`` writes a
  serial single job that exports ``CCS_DRUN_RANK_FILE``, refuses a missing
  allocation, and launches through the rendered argv (exercised with a fake
  ``drun``, so no scheduler and no MPI are needed).

Nothing here starts MPI, a scheduler or a remote job.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLI_TIMEOUT_SECONDS = 300

#: Two-node L48 allocation exactly as the scheduler reports it
#: (288 cores per node = 16 ranks x 18 cores).
ALLOCATION = "cn19544 288 16\ncn19546 288 16\n"

#: The formal payload every launcher test carries (one MPI rank executes it).
PAYLOAD = ["python3", "-m", "pyqcu.test", "--mode", "solve", "--l48-two-node",
           "--only", "kupl/l48", "--ranks", "32", "--lattice-grid",
           "2", "2", "2", "4"]


def _skip_if_port_missing() -> None:
    try:
        import pyqcu.test  # noqa: F401
        import pyqcu.test.launch  # noqa: F401
    except Exception:  # noqa: BLE001 - a partial tree is "not runnable"
        pytest.skip("formal pyqcu.test port is not runnable in this checkout",
                    allow_module_level=True)


_skip_if_port_missing()

from pyqcu.test import l48, launch  # noqa: E402


def _environment() -> dict[str, str]:
    """Environment that imports the working tree, not an installed copy."""
    environment = os.environ.copy()
    pythonpath = str(ROOT)
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test.launch", *arguments],
        cwd=ROOT, env=_environment(), text=True, capture_output=True,
        check=False, timeout=CLI_TIMEOUT_SECONDS)


def _write_allocation(tmp_path: Path, text: str = ALLOCATION) -> Path:
    path = tmp_path / "ccs_alloc"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CCS_ALLOC_FILE
# ---------------------------------------------------------------------------
def test_allocation_lines_parse_with_the_documented_format() -> None:
    allocations = launch.parse_allocation(ALLOCATION)
    assert [node.host for node in allocations] == ["cn19544", "cn19546"]
    assert [node.total_cores for node in allocations] == [288, 288]
    assert [node.rank_count for node in allocations] == [16, 16]
    assert [node.cores_per_rank for node in allocations] == [18, 18]
    assert allocations[0].as_dict() == {
        "host": "cn19544", "total_cores": 288, "rank_count": 16,
        "cores_per_rank": 18}


def test_allocation_parser_ignores_comments_and_blank_lines() -> None:
    text = ("# host total_cores rank_count\n\n"
            "cn19544 288 16   # first node\n\n"
            "cn19546 288 16\n")
    assert launch.parse_allocation(text) == launch.parse_allocation(ALLOCATION)


@pytest.mark.parametrize("text, fragment", [
    ("cn19544 288\n", "expected 'host total_cores rank_count'"),
    ("cn19544 many 16\n", "must be an integer"),
    ("cn19544 288 0\n", "rank_count must be >= 1"),
    ("cn19544 0 16\n", "cores must be >= 1"),
    ("cn19544 289 16\n", "not divisible by rank_count"),
    ("cn19544 288 16\ncn19544 288 16\n", "already appeared on line 1"),
    ("# only a comment\n", "allocation file is empty"),
])
def test_malformed_allocations_are_refused(text: str, fragment: str) -> None:
    with pytest.raises(launch.LaunchError) as error:
        launch.parse_allocation(text)
    assert fragment in str(error.value)


def test_allocation_file_round_trip_and_missing_file(tmp_path: Path) -> None:
    path = _write_allocation(tmp_path)
    assert launch.read_allocation(path) == launch.parse_allocation(ALLOCATION)
    with pytest.raises(launch.LaunchError, match="cannot read the allocation"):
        launch.read_allocation(tmp_path / "absent")


def test_allocation_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(launch.ALLOC_ENV, raising=False)
    assert launch.allocation_from_env() is None
    monkeypatch.setenv(launch.ALLOC_ENV, "/tmp/alloc")
    assert launch.allocation_from_env() == "/tmp/alloc"
    assert launch.allocation_from_env({launch.ALLOC_ENV: "  "}) is None


# ---------------------------------------------------------------------------
# rankfile rows
# ---------------------------------------------------------------------------
def test_rankfile_has_32_rows_with_disjoint_physical_slots() -> None:
    allocations = launch.parse_allocation(ALLOCATION)
    rows = launch.rankfile_rows(
        allocations, ranks=l48.RANKS, ranks_per_node=l48.RANKS_PER_NODE,
        grid=l48.GRID, lattice=l48.LATTICE)

    assert len(rows) == 32, "one row per rank, never fewer"
    assert [row.rank for row in rows] == list(range(32))
    assert all(row.width == 18 for row in rows), "288 / 16 = 18, not 36"
    # Ranks 0-15 on the first node, 16-31 on the second, 0-287 on each node.
    assert {row.host for row in rows[:16]} == {"cn19544"}
    assert {row.host for row in rows[16:]} == {"cn19546"}
    assert (rows[0].first_slot, rows[0].last_slot) == (0, 17)
    assert (rows[15].first_slot, rows[15].last_slot) == (270, 287)
    assert (rows[16].first_slot, rows[16].last_slot) == (0, 17)
    assert (rows[31].first_slot, rows[31].last_slot) == (270, 287)

    for host in ("cn19544", "cn19546"):
        ranges = sorted((row.first_slot, row.last_slot) for row in rows
                        if row.host == host)
        assert len(ranges) == 16
        assert ranges[0][0] == 0 and ranges[-1][1] == 287
        for previous, current in zip(ranges, ranges[1:]):
            assert current[0] == previous[1] + 1, "slots must be contiguous"
            assert current[0] > previous[1], "slots must not overlap"


def test_rankfile_generalises_to_another_node_count() -> None:
    allocations = launch.parse_allocation(
        "cn1 152 8\ncn2 152 8\ncn3 152 8\ncn4 152 8\n")
    rows = launch.rankfile_rows(allocations, grid=(2, 2, 2, 4),
                                lattice=l48.LATTICE)
    assert len(rows) == 32
    assert [node.host for node in allocations] == sorted(
        {row.host for row in rows})
    assert all(row.width == 19 for row in rows)  # 152 / 8
    assert rows[-1].rank == 31 and rows[-1].last_slot == 151


#: The documented L48 rank counts: 16 ranks on one node (``2x2x2x2``), the
#: production 32 ranks on two nodes (``2x2x2x4``) and 64 ranks on four nodes
#: (``2x2x2x8``).  All three keep 16 ranks and 288 cores per node, so the width
#: stays 288 / 16 = 18.
RANK_SHAPES = (
    (1, 16, (2, 2, 2, 2)),
    (2, 32, (2, 2, 2, 4)),
    (4, 64, (2, 2, 2, 8)),
)


def _shape_allocation(nodes: int) -> str:
    """``nodes`` lines of the verified 288-core / 16-rank allocation."""
    return "".join(f"cn{index} 288 16\n" for index in range(nodes))


@pytest.mark.parametrize("nodes, ranks, grid", RANK_SHAPES)
def test_rankfile_scales_over_16_32_and_64_ranks(
        nodes: int, ranks: int, grid: tuple[int, int, int, int]) -> None:
    """One row per rank, 16 rows per node, 18 disjoint cores per row."""
    allocations = launch.parse_allocation(_shape_allocation(nodes))
    rows = launch.rankfile_rows(allocations, ranks=ranks, ranks_per_node=16,
                                grid=grid, lattice=l48.LATTICE)

    assert len(rows) == ranks == nodes * 16
    assert [row.rank for row in rows] == list(range(ranks))
    assert all(row.width == 18 for row in rows), "288 / 16 = 18"
    for index, node in enumerate(allocations):
        block = rows[index * 16:(index + 1) * 16]
        assert {row.host for row in block} == {node.host}
        assert [row.first_slot for row in block] == [
            18 * step for step in range(16)]
        assert (block[0].first_slot, block[0].last_slot) == (0, 17)
        assert (block[-1].first_slot, block[-1].last_slot) == (270, 287)
        # Every rank's slots are disjoint and together they cover the node.
        covered = sorted(slot for row in block
                         for slot in range(row.first_slot, row.last_slot + 1))
        assert covered == list(range(node.total_cores))
    text = launch.render_rankfile(rows)
    assert text.count("\n") == ranks
    assert text.splitlines()[0].startswith("rank 0=cn0 ")
    assert text.splitlines()[-1].startswith(f"rank {ranks - 1}=cn{nodes - 1} ")


@pytest.mark.parametrize("nodes, ranks, grid", RANK_SHAPES)
def test_plan_scales_over_16_32_and_64_ranks(
        nodes: int, ranks: int, grid: tuple[int, int, int, int]) -> None:
    """The executable plan keeps the verified ``drun`` shape at every size."""
    allocation = _shape_allocation(nodes)
    payload = launch.formal_payload(only="kupl/l48", ranks=ranks, grid=grid)
    plan = launch.build_plan(
        payload, nodes=nodes, ranks_per_node=16, ranks=ranks, grid=grid,
        lattice=l48.LATTICE, allocation=allocation, job_id=f"job{ranks}")

    assert plan["kind"] == "drun" and plan["mpi_launch"] is True
    assert plan["validated"] is True and plan["scheduler_required"] is True
    assert plan["nodes"] == nodes and plan["ranks"] == ranks
    assert plan["ranks_per_node"] == 16 and plan["cores_per_rank"] == 18
    assert plan["exec_hosts"] == [f"cn{index}" for index in range(nodes)]
    assert plan["rankfile"] == f"job{ranks}-rankfile"
    assert len(plan["rankfile_rows"]) == ranks
    assert plan["command"][:11] == [
        launch.DEFAULT_DRUN, "-mpi", "pmix", "-nn", str(nodes), "-rpn", "16",
        "-R", "core=18", "-v", "--"]
    assert plan["command"][-len(payload):] == payload
    assert f"--ranks {ranks}" in plan["command_line"]
    json.dumps(plan)
    if nodes > 1:
        # A short allocation must be refused, never silently shrunk to fit.
        with pytest.raises(launch.LaunchError, match="names 1 node"):
            launch.build_plan(payload, nodes=nodes, ranks_per_node=16,
                              ranks=ranks, grid=grid, lattice=l48.LATTICE,
                              allocation=_shape_allocation(1))


#: The two-node variants of ``docs/kupl_remote_l48_experience_20261117.md``
#: §7.6: the same two 288-core nodes at 16/32/64 ranks, so the width is
#: ``288 / ranks_per_node`` = 36, 18 and 9 (jobs 2155830/2155536/2155868).
TWO_NODE_VARIANTS = (
    (16, 8, 36, (2, 2, 2, 2)),
    (32, 16, 18, (2, 2, 2, 4)),
    (64, 32, 9, (2, 2, 4, 4)),
)


@pytest.mark.parametrize("ranks, ranks_per_node, width, grid",
                         TWO_NODE_VARIANTS)
def test_two_node_variants_scale_ranks_and_width(
        ranks: int, ranks_per_node: int, width: int,
        grid: tuple[int, int, int, int]) -> None:
    """16/32/64 ranks on two nodes: the allocation sets ranks *and* width."""
    allocation = "".join(f"cn{host} 288 {ranks_per_node}\n"
                         for host in ("a", "b"))
    allocations = launch.parse_allocation(allocation)
    rows = launch.rankfile_rows(allocations, ranks=ranks,
                                ranks_per_node=ranks_per_node, grid=grid,
                                lattice=l48.LATTICE)
    assert len(rows) == ranks == 2 * ranks_per_node
    assert rows[0].rank == 0 and rows[-1].rank == ranks - 1
    assert all(row.width == width for row in rows)
    assert (rows[0].first_slot, rows[0].last_slot) == (0, width - 1)
    assert (rows[ranks_per_node - 1].first_slot,
            rows[ranks_per_node - 1].last_slot) == (288 - width, 287)

    plan = launch.build_plan(
        launch.formal_payload(only="kupl/l48", ranks=ranks, grid=grid),
        nodes=2, ranks_per_node=ranks_per_node, ranks=ranks, grid=grid,
        lattice=l48.LATTICE, allocation=allocation, job_id=f"job{ranks}")
    assert plan["kind"] == "drun" and plan["mpi_launch"] is True
    assert plan["validated"] is True
    assert plan["exec_hosts"] == ["cna", "cnb"]
    assert plan["cores_per_rank"] == width
    assert len(plan["rankfile_rows"]) == ranks
    assert plan["command"][4] == "2" and plan["command"][6] == str(
        ranks_per_node)
    assert plan["command"][8] == f"core={width}"


def test_rankfile_width_follows_the_allocation_not_the_l48_default() -> None:
    """``total_cores / rank_count`` is the width: 608 / 16 = 38, not 18."""
    allocations = launch.parse_allocation("cn1 608 16\ncn2 608 16\n")
    rows = launch.rankfile_rows(allocations, ranks=32, ranks_per_node=16,
                                grid=l48.GRID, lattice=l48.LATTICE)
    assert all(row.width == 38 for row in rows)
    assert (rows[0].first_slot, rows[0].last_slot) == (0, 37)
    assert (rows[15].first_slot, rows[15].last_slot) == (570, 607)
    assert (rows[31].first_slot, rows[31].last_slot) == (570, 607)
    # An explicit width the node cannot hold is refused, not truncated.
    with pytest.raises(launch.LaunchError, match="exceeds the allocated 608"):
        launch.rankfile_rows(allocations, ranks=32, ranks_per_node=16,
                             cores_per_rank=39, grid=l48.GRID,
                             lattice=l48.LATTICE)
    # Per-node widths that disagree cannot be averaged into one rankfile.
    with pytest.raises(launch.LaunchError, match="different per-rank widths"):
        launch.rankfile_rows(
            launch.parse_allocation("cn1 288 16\ncn2 304 16\n"), ranks=32,
            ranks_per_node=16, grid=l48.GRID, lattice=l48.LATTICE)


def test_rankfile_row_text_matches_the_vendor_template() -> None:
    rows = launch.rankfile_rows(
        launch.parse_allocation(ALLOCATION), grid=l48.GRID,
        lattice=l48.LATTICE)
    assert rows[0].as_line() == "rank 0=cn19544 slot=0-17"
    assert rows[-1].as_line() == "rank 31=cn19546 slot=270-287"
    text = launch.render_rankfile(rows)
    assert text.count("\n") == 32 and text.endswith("\n")
    assert text.splitlines()[1] == "rank 1=cn19544 slot=18-35"


def test_overlapping_slots_are_refused() -> None:
    # The generator cannot produce this; the guard must still reject a
    # hand-written rankfile that hands the same core to two ranks.
    with pytest.raises(launch.LaunchError, match="overlap"):
        launch.check_slot_overlap({"cn1": [(0, 17, 0), (10, 27, 1)]})


def test_rankfile_refuses_geometry_and_capacity_violations() -> None:
    allocations = launch.parse_allocation(ALLOCATION)
    with pytest.raises(launch.LaunchError, match="!= ranks"):
        launch.rankfile_rows(allocations, grid=(2, 2, 2, 2),
                             lattice=l48.LATTICE)
    with pytest.raises(launch.LaunchError, match="odd rank-local extent"):
        launch.rankfile_rows(allocations, grid=(2, 2, 2, 4),
                             lattice=(48, 48, 48, 92))
    with pytest.raises(launch.LaunchError, match="does not divide"):
        launch.rankfile_rows(allocations, grid=(2, 2, 2, 4),
                             lattice=(48, 48, 48, 95))
    with pytest.raises(launch.LaunchError, match="exceeds the allocated"):
        launch.rankfile_rows(launch.parse_allocation("cn1 160 16\ncn2 160 16\n"),
                             cores_per_rank=18, grid=l48.GRID,
                             lattice=l48.LATTICE)
    with pytest.raises(launch.LaunchError, match="!= nodes 2"):
        launch.rankfile_rows(allocations, ranks=31, grid=l48.GRID,
                             lattice=l48.LATTICE)


def test_geometry_requires_ranks_equal_nodes_times_ranks_per_node() -> None:
    launch.validate_geometry(ranks=32, nodes=2, ranks_per_node=16,
                             grid=l48.GRID, lattice=l48.LATTICE)
    with pytest.raises(launch.LaunchError,
                       match=r"ranks 32 != nodes 2 \* ranks_per_node 15"):
        launch.validate_geometry(ranks=32, nodes=2, ranks_per_node=15)


def test_write_rankfile_is_atomic_and_leaves_no_temporary(
        tmp_path: Path) -> None:
    rows = launch.rankfile_rows(
        launch.parse_allocation(ALLOCATION), grid=l48.GRID,
        lattice=l48.LATTICE)
    target = tmp_path / "logs" / "2155536-rankfile"
    written = launch.write_rankfile(target, rows)
    assert written == target
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert lines[0] == "rank 0=cn19544 slot=0-17"
    assert list(target.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# launch commands
# ---------------------------------------------------------------------------
def test_drun_command_matches_the_verified_job_shape() -> None:
    command = launch.drun_command(PAYLOAD, nodes=2, ranks_per_node=16,
                                  cores_per_rank=18)
    assert command[:11] == [
        launch.DEFAULT_DRUN, "-mpi", "pmix", "-nn", "2", "-rpn", "16",
        "-R", "core=18", "-v", "--"]
    assert command[11:] == PAYLOAD
    assert launch.DEFAULT_DRUN == "/opt/batch/agent/tools/drun"


def test_mpirun_command_passes_the_rankfile_and_physical_mapping() -> None:
    command = launch.mpirun_command(PAYLOAD, rankfile="/out/j-rankfile",
                                    ranks=32)
    assert command[:7] == ["mpirun", "-np", "32", "-rf", "/out/j-rankfile",
                           "--mca", "rmaps_rank_file_physical"]
    assert command[7] == "true"
    assert "-x" in command and "PATH" in command
    assert command[-len(PAYLOAD):] == PAYLOAD
    # Without a rankfile there is nothing to bind: no -rf, no physical mapping.
    plain = launch.mpirun_command(PAYLOAD, rankfile=None, ranks=32)
    assert "-rf" not in plain and "rmaps_rank_file_physical" not in plain


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_cross_node_plan_without_allocation_or_rankfile_is_refused() -> None:
    with pytest.raises(launch.LaunchError) as error:
        launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                          grid=l48.GRID, lattice=l48.LATTICE)
    message = str(error.value)
    assert "refusing a cross-node launch" in message
    assert launch.ALLOC_ENV in message and launch.RANKFILE_ENV in message


def test_plan_from_an_allocation_is_executable_and_validated(
        tmp_path: Path) -> None:
    alloc = _write_allocation(tmp_path)
    plan = launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                             grid=l48.GRID, lattice=l48.LATTICE,
                             allocation=alloc, job_id="2155536")
    assert plan["kind"] == "drun" and plan["mpi_launch"] is True
    assert plan["scheduler_required"] is True
    assert plan["validated"] is True
    assert plan["exec_hosts"] == ["cn19544", "cn19546"]
    assert plan["allocation_file"] == str(alloc)
    assert plan["cores_per_rank"] == 18
    assert plan["rankfile"] == "2155536-rankfile"
    assert len(plan["rankfile_rows"]) == 32
    assert plan["rankfile_text"].count("\n") == 32
    assert plan["environment"][launch.RANKFILE_ENV] == plan["rankfile"]
    assert plan["command"][0] == launch.DEFAULT_DRUN
    assert "-mpi" in plan["command"] and "pmix" in plan["command"]
    assert plan["command"][-len(PAYLOAD):] == PAYLOAD
    assert plan["command_line"].startswith(launch.DEFAULT_DRUN)
    json.dumps(plan)  # the whole plan must be JSON-safe (--json)


def test_plan_with_only_a_rankfile_is_real_but_unvalidated() -> None:
    plan = launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                             rankfile="/out/rankfile", launcher="mpirun")
    assert plan["kind"] == "mpirun" and plan["mpi_launch"] is True
    assert plan["validated"] is False
    assert plan["exec_hosts"] is None and plan["rankfile_rows"] == []
    assert "-rf" in plan["command"] and "/out/rankfile" in plan["command"]


def test_drun_plan_with_only_a_rankfile_refuses_the_missing_width() -> None:
    """``drun`` records ``-R core=<width>``; without one it must not guess."""
    with pytest.raises(launch.LaunchError, match="drun needs the per-rank"):
        launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                          rankfile="/out/rankfile")
    # The same request is executable through the ``-rf`` form, or through
    # ``drun`` once the width is stated.
    mpirun_plan = launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16,
                                    ranks=32, rankfile="/out/rankfile",
                                    launcher="mpirun")
    assert mpirun_plan["kind"] == "mpirun"
    assert mpirun_plan["mpi_launch"] is True
    assert mpirun_plan["validated"] is False
    drun_plan = launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16,
                                  ranks=32, rankfile="/out/rankfile",
                                  cores_per_rank=18)
    assert drun_plan["kind"] == "drun"
    assert drun_plan["command"][8] == "core=18"
    assert drun_plan["validated"] is False


def test_single_node_plan_needs_no_rankfile() -> None:
    plan = launch.build_plan(PAYLOAD, nodes=1, ranks_per_node=32, ranks=32,
                             launcher="mpirun")
    assert plan["kind"] == "mpirun" and plan["mpi_launch"] is True
    assert plan["rankfile"] is None and plan["exec_hosts"] is None
    assert "-rf" not in plan["command"]


def test_plan_reads_the_job_environment_only_when_asked(
        tmp_path: Path) -> None:
    alloc = _write_allocation(tmp_path)
    environ = {launch.ALLOC_ENV: str(alloc), launch.RANKFILE_ENV:
               str(tmp_path / "env-rankfile"), launch.JOB_ID_ENV: "2155536",
               launch.TASK_REPLICA_ENV: "32"}
    plan = launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                             grid=l48.GRID, lattice=l48.LATTICE,
                             environ=environ)
    assert plan["allocation_file"] == str(alloc)
    assert plan["rankfile"] == str(tmp_path / "env-rankfile")
    assert plan["job_id"] == "2155536"
    assert plan["task_replica"] == 32, "the manual's -np source agrees"
    # A replica count that contradicts the rank count is a broken request.
    with pytest.raises(launch.LaunchError, match="CCS_TASK_REPLICA=16"):
        launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                          grid=l48.GRID, lattice=l48.LATTICE,
                          environ={**environ, launch.TASK_REPLICA_ENV: "16"})
    # Same arguments, no ``environ``: still the explicit refusal.
    with pytest.raises(launch.LaunchError, match="refusing a cross-node"):
        launch.build_plan(PAYLOAD, nodes=2, ranks_per_node=16, ranks=32,
                          grid=l48.GRID, lattice=l48.LATTICE)


# ---------------------------------------------------------------------------
# import rules and command line
# ---------------------------------------------------------------------------
_IMPORT_PROBE = """
import json, sys
import pyqcu.test.launch
import pyqcu.test.l48
plan = pyqcu.test.l48.launch_plan("kupl/l48/bicgstab/c64")
forbidden = sorted(
    name for name in sys.modules
    if name.split(".")[0] in ("torch", "numpy", "mpi4py", "pytest")
    or name == "pyqcu.testing" or name.startswith("pyqcu.testing."))
print(json.dumps({"forbidden": forbidden,
                  "has_main": hasattr(pyqcu.test.launch, "main"),
                  "kind": plan["kind"],
                  "mpi_launch": plan["mpi_launch"]}))
"""


def test_launch_imports_only_the_standard_library() -> None:
    """Neither the planner nor the L48 hook pulls a backend or ``testing``."""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_PROBE], cwd=ROOT,
        env=_environment(), text=True, capture_output=True, check=False,
        timeout=CLI_TIMEOUT_SECONDS)
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["forbidden"] == [], payload
    assert payload["has_main"] is True
    # The no-arg hook is a declaration, not a launch, and it imports no
    # launcher process module either.
    assert payload["kind"] == "mpirun"
    assert payload["mpi_launch"] is False


def test_cli_help_and_unknown_subcommand() -> None:
    completed = _run_cli("--help")
    assert completed.returncode == 0
    assert "rankfile" in completed.stdout and "argv" in completed.stdout
    assert _run_cli("nonsense").returncode == 2


def test_cli_rankfile_then_argv_and_plan(tmp_path: Path) -> None:
    alloc = _write_allocation(tmp_path)
    rankfile = tmp_path / "logs" / "job-rankfile"
    written = _run_cli("rankfile", "--alloc-file", str(alloc), "--rankfile",
                       str(rankfile), "--cores-per-rank", "18")
    assert written.returncode == 0, written.stderr[-2000:]
    assert "rows=32" in written.stdout
    assert len(rankfile.read_text(encoding="utf-8").splitlines()) == 32

    argv = _run_cli("argv", "--alloc-file", str(alloc), "--rankfile",
                    str(rankfile), "--cores-per-rank", "18", "--threads", "18")
    assert argv.returncode == 0, argv.stderr[-2000:]
    arguments = argv.stdout.splitlines()
    assert arguments[0] == launch.DEFAULT_DRUN
    assert arguments[1:11] == ["-mpi", "pmix", "-nn", "2", "-rpn", "16",
                              "-R", "core=18", "-v", "--"]
    assert arguments[-2:] == ["--threads", "18"]
    assert "kupl/l48" in arguments

    plan = _run_cli("plan", "--alloc-file", str(alloc), "--rankfile",
                    str(rankfile), "--cores-per-rank", "18", "--json")
    assert plan.returncode == 2  # ``plan`` takes no --json
    plain = _run_cli("plan", "--alloc-file", str(alloc), "--rankfile",
                     str(rankfile), "--cores-per-rank", "18")
    assert plain.returncode == 0, plain.stderr[-2000:]
    payload = json.loads(plain.stdout)
    assert payload["kind"] == "drun" and payload["mpi_launch"] is True
    assert payload["exec_hosts"] == ["cn19544", "cn19546"]


def test_cli_plans_64_ranks_from_a_four_node_allocation(
        tmp_path: Path) -> None:
    """The CLI geometry flags, not the L48 32-rank default, size the plan."""
    alloc = tmp_path / "ccs_alloc64"
    alloc.write_text(_shape_allocation(4), encoding="utf-8")
    completed = _run_cli(
        "plan", "--alloc-file", str(alloc), "--job-id", "job64",
        "--nodes", "4", "--ranks", "64", "--ranks-per-node", "16",
        "--grid", "2", "2", "2", "8", "--cores-per-rank", "18")
    assert completed.returncode == 0, completed.stderr[-2000:]
    plan = json.loads(completed.stdout)
    assert plan["kind"] == "drun" and plan["mpi_launch"] is True
    assert plan["nodes"] == 4 and plan["ranks"] == 64
    assert plan["ranks_per_node"] == 16 and plan["cores_per_rank"] == 18
    assert plan["exec_hosts"] == ["cn0", "cn1", "cn2", "cn3"]
    assert len(plan["rankfile_rows"]) == 64
    assert plan["command"][4] == "4" and plan["command"][6] == "16"
    assert plan["rankfile"] == "job64-rankfile"
    # A 64-rank request over a two-node allocation is refused.
    short = tmp_path / "ccs_alloc32"
    short.write_text(_shape_allocation(2), encoding="utf-8")
    refused = _run_cli(
        "plan", "--alloc-file", str(short), "--nodes", "4", "--ranks", "64",
        "--ranks-per-node", "16", "--grid", "2", "2", "2", "8")
    assert refused.returncode == 2
    assert "names 2 node" in refused.stderr


def test_cli_refuses_without_an_allocation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(launch.ALLOC_ENV, raising=False)
    monkeypatch.delenv(launch.RANKFILE_ENV, raising=False)
    # 1) Writing a rankfile needs the allocation: nothing to derive it from.
    written = _run_cli("rankfile")
    assert written.returncode == 2, written.stdout
    assert "no allocation file" in written.stderr
    # 2) A cross-node launch command with neither file is refused outright.
    command = _run_cli("argv")
    assert command.returncode == 2, command.stdout
    assert "refusing a cross-node launch" in command.stderr


# ---------------------------------------------------------------------------
# submission script
# ---------------------------------------------------------------------------
SCRIPT = ROOT / "pyqcu" / "testing" / "kupl" / "submit_pyqcu_test.sh"
FAKE_DRUN = """#!/bin/bash
printf 'FAKE-DRUN %s\\n' "$*"
exit 0
"""


def _skip_without_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is not available here")


def test_submit_script_exists_and_is_shell_clean() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash")
    # Dependency direction: the job calls pyqcu.test, never the reverse.
    # Dependency direction: nothing under pyqcu/test imports it.
    assert "import pyqcu.testing" not in text
    assert "from pyqcu.testing" not in text
    assert launch.ALLOC_ENV in text and launch.RANKFILE_ENV in text
    assert "pyqcu.test.launch" in text
    _skip_without_bash()
    completed = subprocess.run(["bash", "-n", str(SCRIPT)], text=True,
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_submit_script_dry_run_prints_the_job_without_submitting(
        tmp_path: Path) -> None:
    _skip_without_bash()
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--tag", "l48_dry",
         "--outdir", str(tmp_path), "--only", "kupl/l48"],
        text=True, capture_output=True, check=False, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "dry-run: would run 'dsub -s" in completed.stdout
    job = tmp_path / "jobs" / "l48_dry.sh"
    assert job.is_file()
    body = job.read_text(encoding="utf-8")
    for expected in ("#DSUB -nn 2", "#DSUB -N 32", "#DSUB -rpn 16",
                     "#DSUB -x job", launch.RANKFILE_ENV,
                     "pyqcu.test.launch rankfile", "pyqcu.test.launch argv",
                     "mapfile", '"${LAUNCH_ARGV[@]}"'):
        assert expected in body, f"{expected!r} missing from the job script"
    assert "\neval " not in body


def test_submit_script_job_body_launches_through_the_rankfile(
        tmp_path: Path) -> None:
    _skip_without_bash()
    alloc = _write_allocation(tmp_path)
    drun = tmp_path / "fake_drun"
    drun.write_text(FAKE_DRUN, encoding="utf-8")
    drun.chmod(0o755)
    tag = "l48_body"
    generated = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--tag", tag, "--outdir",
         str(tmp_path), "--drun", str(drun)],
        text=True, capture_output=True, check=False, timeout=120)
    assert generated.returncode == 0, generated.stderr[-2000:]

    environment = _environment()
    environment[launch.ALLOC_ENV] = str(alloc)
    completed = subprocess.run(
        ["bash", str(tmp_path / "jobs" / f"{tag}.sh")], text=True,
        capture_output=True, check=False, timeout=300, env=environment,
        cwd=str(ROOT))
    assert completed.returncode == 0, (
        f"job body failed:\n{completed.stdout[-2000:]}\n"
        f"{completed.stderr[-2000:]}")
    rankfile = tmp_path / "logs" / f"{tag}-rankfile"
    assert len(rankfile.read_text(encoding="utf-8").splitlines()) == 32
    assert "rank 31=cn19546 slot=270-287" in completed.stdout or \
        "slot=270-287" in rankfile.read_text(encoding="utf-8")
    assert "FAKE-DRUN -mpi pmix -nn 2 -rpn 16 -R core=18 -v --" in \
        completed.stdout
    assert "pyqcu.test --mode solve --l48-two-node --only kupl/l48" in \
        completed.stdout
    assert f"--outdir {tmp_path}" in completed.stdout, (
        "the ledger root must be the job's outdir, not pyqcu-test-runs/")
    assert "rc=0" in completed.stdout


def test_submit_script_job_body_refuses_a_missing_allocation(
        tmp_path: Path) -> None:
    _skip_without_bash()
    tag = "l48_noalloc"
    generated = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--tag", tag, "--outdir",
         str(tmp_path)],
        text=True, capture_output=True, check=False, timeout=120)
    assert generated.returncode == 0, generated.stderr[-2000:]
    environment = _environment()
    environment.pop(launch.ALLOC_ENV, None)
    completed = subprocess.run(
        ["bash", str(tmp_path / "jobs" / f"{tag}.sh")], text=True,
        capture_output=True, check=False, timeout=120, env=environment,
        cwd=str(ROOT))
    assert completed.returncode == 2
    assert "refusing the cross-node launch" in completed.stderr
    assert not (tmp_path / "logs" / f"{tag}-rankfile").exists()
