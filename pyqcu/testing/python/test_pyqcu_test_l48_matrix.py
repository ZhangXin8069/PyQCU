"""Focused tests for the L48 two-node matrix and the L48 CG retirement.

The formal port ships ``pyqcu.test.l48`` as a declarative, stdlib-only manifest.
These tests are the informal adapter that keeps three promises honest:

* the matrix is **present** - five groups, the required per-case parameters,
  even rank grids and a data-only launcher hook;
* the L48 **CG group is absent** - no CG case in the matrix or in any L48 plan,
  an explicit retirement record, and a usage error when CG is requested on an
  L48 lattice;
* the tier-0 matrix is **unchanged** - ``--quick`` still selects exactly the 19
  shipped cases (CG included) and ``--list`` still reports 19 by default.

No solver, backend or scheduler is exercised here: every check is declarative
or a ``--list``/``--mode list`` invocation, so the tests run on a machine
without torch, KUPL, MPI or a remote scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLI_TIMEOUT_SECONDS = 300


def _skip_if_port_missing() -> None:
    try:
        import pyqcu.test  # noqa: F401
        import pyqcu.test.__main__  # noqa: F401
    except Exception:  # noqa: BLE001 - a partial tree is "not runnable"
        pytest.skip("formal pyqcu.test port is not runnable in this checkout",
                    allow_module_level=True)


_skip_if_port_missing()

from pyqcu.test import l48, launch, policy  # noqa: E402
from pyqcu.test.common import Test  # noqa: E402


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    pythonpath = str(ROOT)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        check=False, timeout=CLI_TIMEOUT_SECONDS)


def _json(*arguments: str) -> dict:
    completed = _run_cli(*arguments, "--json", "-")
    assert completed.returncode == 0, (
        f"pyqcu.test {' '.join(arguments)} failed: rc={completed.returncode}\n"
        f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def test_manifest_has_the_five_required_groups() -> None:
    names = tuple(group.name for group in l48.GROUPS)
    assert names == ("bicgstab", "legacy_mg", "dslash", "input_io",
                     "grid_variants")
    assert tuple(group.name for group in l48.GROUPS) == l48.GROUP_NAMES
    for group in l48.GROUPS:
        if group.kind == "cases":
            assert group.cases, f"{group.name}: no cases"
        elif group.kind == "harness":
            assert group.harnesses, f"{group.name}: no harness"
        elif group.kind == "geometry":
            assert group.variants, f"{group.name}: no variants"
    l48.validate()  # raises ValueError on a broken manifest


def test_no_strict_mg_anywhere() -> None:
    assert "strict-mg" in l48.FORBIDDEN_GROUPS
    texts = [group.name for group in l48.GROUPS]
    texts += [group.title for group in l48.GROUPS]
    texts += [group.summary for group in l48.GROUPS]
    texts += [case.tag for case in l48.cases()]
    texts += [declaration.name for declaration in l48.DECLARATIONS]
    lowered = " ".join(texts).lower()
    assert "strict" not in lowered
    payload = l48.plan_payload()
    assert all("strict" not in str(group.get("name", "")).lower()
               for group in payload["groups"])
    assert all("strict" not in str(tag).lower()
               for tag in payload["case_tags"])


def test_l48_cg_group_is_retired_and_absent() -> None:
    assert l48.RETIREMENT["group"] == "l48/cg"
    assert l48.RETIREMENT["status"] == "RETIRED"
    assert l48.RETIREMENT["scope"] == "L48 only"
    assert "tier-0" in l48.RETIREMENT["kept"]
    assert l48.RETIRED_GROUPS == ("cg",)
    assert "cg" in l48.FORBIDDEN_GROUPS
    # No CG case in the matrix and no CG tag reachable through cases().
    assert not [case for case in l48.cases() if l48.is_cg_tag(case.tag)]
    assert not [declaration for declaration in l48.DECLARATIONS
                if l48.is_cg_tag(declaration.tag)]
    # The tier-0 CG cases are *kept* - the retirement is L48-only.
    assert l48.CG_TAGS == ("torch/solve/cg", "kupl/solve/cg")
    assert set(l48.CG_TAGS) <= set(policy.TIER0_TAGS)


def test_bicgstab_group_parameters() -> None:
    cases = [case for case in l48.DECLARATIONS if case.group == "bicgstab"]
    assert {case.dtype for case in cases} == {"c64", "c128"}
    for case in cases:
        assert case.solver == "bistabcg"
        assert case.mg_levels == 1, "BiCGStab is levels=1"
        assert case.restart == 0, "BiCGStab is restart=0"
        assert case.grid == (2, 2, 2, 4)
        assert case.ranks == 32
        assert case.nodes == 2
        assert case.evidence == "UNVERIFIED"


def test_legacy_mg_group_parameters_and_unverified_status() -> None:
    cases = [case for case in l48.DECLARATIONS if case.group == "legacy_mg"]
    assert {case.dtype for case in cases} == {"c64", "c128"}
    for case in cases:
        assert case.solver == "mg"
        assert case.mg_levels == 2, "Legacy-MG is levels=2"
        assert case.restart == 5, "Legacy-MG is restart=5"
        assert case.ranks == 32
        assert case.grid == (2, 2, 2, 4)
        assert case.nodes == 2
        # 32 ranks/2 nodes is the definition; the native path refuses
        # multi-rank Legacy-MG, so the evidence must say UNVERIFIED and carry
        # the blocker instead of implying a PASS.
        assert case.evidence in ("UNVERIFIED", "VERIFIED")
        if case.evidence == "UNVERIFIED":
            assert case.blocked_by.strip(), "UNVERIFIED needs a blocker"
            assert "multi-rank" in case.blocked_by.lower()


def test_dslash_group_uses_the_existing_kupl_harness() -> None:
    group = next(group for group in l48.GROUPS if group.name == "dslash")
    assert group.kind == "harness"
    assert not group.cases, "the dslash group must not add a formal case"
    assert {harness.name for harness in group.harnesses} == {
        "kupl-dslash-block-mpi", "kupl-clover-check"}
    for harness in group.harnesses:
        assert harness.path.startswith("pyqcu/testing/")
        assert (ROOT / harness.path).is_file(), f"missing {harness.path}"
        assert "--lattice 48 48 48 96" in harness.command \
            or "--input" in harness.command


def test_input_io_group_needs_real_l48_hdf5() -> None:
    cases = [case for case in l48.DECLARATIONS if case.group == "input_io"]
    assert len(cases) == 1
    case = cases[0]
    assert case.requires_input is True
    assert case.kind == "io"
    notes = case.notes.lower()
    assert "hdf5" in notes
    assert "rank-local" in notes
    assert "residual" in notes


def test_grid_variants_keep_local_extents_even() -> None:
    by_name = {variant.name: variant for variant in l48.GRID_VARIANTS}
    assert "2x2x2x4" in by_name, "the production grid must be documented"
    production = by_name["2x2x2x4"]
    assert production.ranks == 32
    assert production.grid == (2, 2, 2, 4)
    assert production.local == (24, 24, 24, 24)
    assert production.even
    assert "1x2x2x8" in by_name, "1x2x2x8 must be documented if valid"
    assert by_name["1x2x2x8"].even
    for variant in l48.GRID_VARIANTS:
        assert variant.status == "VALID"
        assert variant.even, f"{variant.name}: odd local extent"
        assert all(extent % 2 == 0 for extent in variant.local)
        assert variant.nodes * l48.RANKS_PER_NODE == variant.ranks
    rejected = {variant.name: variant for variant
                in l48.REJECTED_GRID_VARIANTS}
    assert "1x1x1x32" in rejected
    assert "odd" in rejected["1x1x1x32"].reason.lower()


def test_l48_cases_are_tier_one_and_pinned() -> None:
    cases = l48.cases()
    assert len(cases) == len(l48.DECLARATIONS) == 5
    for case in cases:
        assert case.tier == 1
        assert case.pinned is True
        assert case.lattice == (48, 48, 48, 96)
        declaration = l48.get(case.tag)
        assert case.dtype == declaration.dtype
        assert case.runner == declaration.sampler
        assert case.timeout and case.timeout > 0


def test_policy_solver_filters_select_the_matrix() -> None:
    assert policy.solver_selects("solve", "kupl/l48/bicgstab/c64", "bicgstab")
    assert policy.solver_selects("mg", "kupl/l48/legacy-mg/c128", "mg")
    assert policy.solver_selects("mg", "kupl/l48/legacy-mg/c128", "multigrid")
    assert not policy.solver_selects("solve", "kupl/l48/bicgstab/c64", "cg")
    assert not policy.solver_selects("mg", "kupl/l48/legacy-mg/c64", "cg")


# ---------------------------------------------------------------------------
# launcher hook
# ---------------------------------------------------------------------------
#: Two-node L48 allocation as the scheduler reports it (288 = 16 x 18 cores).
ALLOCATION = "cn19544 288 16\ncn19546 288 16\n"


def test_launcher_plan_is_data_only_and_needs_no_scheduler() -> None:
    tag = "kupl/l48/bicgstab/c64"
    plan = l48.launch_plan(tag)
    assert plan["kind"] == "mpirun"
    assert plan["launcher"] == "mpirun"
    assert plan["mpi_launch"] is False
    assert plan["scheduler_required"] is False
    assert plan["nodes"] == 2
    assert plan["ranks"] == 32
    assert plan["ranks_per_node"] == 16
    assert plan["grid"] == [2, 2, 2, 4]
    # A declaration owns no allocation: no hosts, no rankfile, no rows and no
    # claim that anything was validated.
    assert plan["rankfile"] is None
    assert plan["exec_hosts"] is None
    assert plan["rankfile_rows"] == []
    assert plan["allocation_file"] is None
    assert plan["validated"] is False
    assert "declaration only" in plan["requires"]
    command = plan["command"]
    assert command[0] == "mpirun"
    assert "-np" in command and command[command.index("-np") + 1] == "32"
    assert "--l48-two-node" in command
    assert tag in command


def test_launcher_plan_becomes_a_real_drun_plan_with_an_allocation(
        tmp_path: Path) -> None:
    allocation = tmp_path / "ccs_alloc"
    allocation.write_text(ALLOCATION, encoding="utf-8")
    tag = "kupl/l48/bicgstab/c64"
    plan = l48.launch_plan(tag, allocation=str(allocation), job_id="2155536",
                           threads=18)

    assert plan["kind"] == "drun", "the verified route, not plm_rsh"
    assert plan["mpi_launch"] is True, "a real plan may claim a launch"
    assert plan["scheduler_required"] is True
    assert plan["validated"] is True
    assert plan["exec_hosts"] == ["cn19544", "cn19546"]
    assert plan["nodes"] == 2 and plan["ranks"] == 32
    assert plan["ranks_per_node"] == 16
    assert plan["cores_per_rank"] == 18
    assert plan["rankfile"].endswith("2155536-rankfile")
    assert len(plan["rankfile_rows"]) == 32
    assert plan["rankfile_text"].splitlines()[0] == "rank 0=cn19544 slot=0-17"
    assert plan["rankfile_text"].splitlines()[-1] == "rank 31=cn19546 slot=284-301"
    assert plan["environment"][launch.RANKFILE_ENV] == plan["rankfile"]

    command = plan["command"]
    assert command[0] == launch.DEFAULT_DRUN
    assert command[1:11] == ["-mpi", "pmix", "-nn", "2", "-rpn", "16",
                             "-R", "core=18", "-v", "--"]
    assert command[command.index("--only") + 1] == tag
    assert "--l48-two-node" in command and "--threads" in command
    # The matrix still owns the verdict: a launchable plan is not a PASS.
    assert plan["group"] == "bicgstab"
    assert plan["evidence"] == "UNVERIFIED"


def test_launcher_plan_renders_the_mpirun_rankfile_form(tmp_path: Path) -> None:
    allocation = tmp_path / "ccs_alloc"
    allocation.write_text(ALLOCATION, encoding="utf-8")
    plan = l48.launch_plan("kupl/l48/legacy-mg/c64", launcher="mpirun",
                           allocation=str(allocation),
                           rankfile=str(tmp_path / "job-rankfile"),
                           cores_per_rank=18)
    assert plan["kind"] == "mpirun" and plan["mpi_launch"] is True
    command = plan["command"]
    assert command[:5] == ["mpirun", "-np", "32", "-rf",
                           str(tmp_path / "job-rankfile")]
    assert "--mca" in command and "rmaps_rank_file_physical" in command
    assert plan["rankfile"] == str(tmp_path / "job-rankfile")


def test_launcher_plan_refuses_a_cross_node_launch_without_an_allocation(
        ) -> None:
    for launcher in ("drun", "mpirun", "auto"):
        with pytest.raises(launch.LaunchError) as error:
            l48.launch_plan("kupl/l48/bicgstab/c64", launcher=launcher)
        assert "refusing a cross-node launch" in str(error.value)


def test_grid_variants_cover_16_32_and_64_ranks() -> None:
    """The documented shapes are 16/32/64 ranks at 16 ranks per node."""
    by_ranks: dict[int, list[str]] = {}
    for variant in l48.GRID_VARIANTS:
        by_ranks.setdefault(variant.ranks, []).append(variant.name)
    assert set(by_ranks) == {16, 32, 64}
    assert set(by_ranks[16]) == {"2x2x2x2", "1x1x1x16"}
    assert set(by_ranks[32]) >= {"2x2x2x4"}
    assert set(by_ranks[64]) == {"2x2x2x8"}
    for variant in l48.GRID_VARIANTS:
        assert variant.nodes == variant.ranks // l48.RANKS_PER_NODE
        assert variant.even and variant.status == "VALID"


@pytest.mark.parametrize("name", ["2x2x2x2", "2x2x2x4", "2x2x2x8"])
def test_grid_variants_generate_their_rankfile(name: str) -> None:
    """Each documented grid turns its allocation into 18-core rankfile rows."""
    variant = next(item for item in l48.GRID_VARIANTS if item.name == name)
    allocation = "".join(
        f"cn{index} 288 16\n" for index in range(variant.nodes))
    rows = launch.rankfile_rows(launch.parse_allocation(allocation),
                                ranks=variant.ranks,
                                ranks_per_node=l48.RANKS_PER_NODE,
                                grid=variant.grid, lattice=l48.LATTICE)
    assert len(rows) == variant.ranks
    assert {row.host for row in rows} == {
        f"cn{index}" for index in range(variant.nodes)}
    assert all(row.width == 18 for row in rows)
    assert launch.render_rankfile(rows).count("\n") == variant.ranks

    payload = launch.formal_payload(only="kupl/l48", ranks=variant.ranks,
                                    grid=variant.grid)
    plan = launch.build_plan(
        payload, nodes=variant.nodes, ranks_per_node=l48.RANKS_PER_NODE,
        ranks=variant.ranks, grid=variant.grid, lattice=l48.LATTICE,
        allocation=allocation, job_id=f"l48_{variant.ranks}")
    assert plan["kind"] == "drun" and plan["mpi_launch"] is True
    assert plan["nodes"] == variant.nodes and plan["ranks"] == variant.ranks
    assert len(plan["exec_hosts"]) == variant.nodes
    assert plan["command"][4] == str(variant.nodes)
    assert plan["command"][6] == str(l48.RANKS_PER_NODE)


def test_a_64_rank_plan_refuses_a_two_node_allocation() -> None:
    """Four nodes are required: a two-node allocation is refused."""
    allocation = "cn1 288 16\ncn2 288 16\n"
    with pytest.raises(launch.LaunchError, match="names 2 node"):
        launch.build_plan(
            launch.formal_payload(only="kupl/l48", ranks=64,
                                  grid=(2, 2, 2, 8)),
            nodes=4, ranks_per_node=16, ranks=64, grid=(2, 2, 2, 8),
            lattice=l48.LATTICE, allocation=allocation)
    # The 32-rank plan is the one that two nodes can hold.
    plan = launch.build_plan(
        launch.formal_payload(only="kupl/l48", ranks=32, grid=l48.GRID),
        nodes=2, ranks_per_node=16, ranks=32, grid=l48.GRID,
        lattice=l48.LATTICE, allocation=allocation, job_id="2155536")
    assert plan["exec_hosts"] == ["cn1", "cn2"]
    assert len(plan["rankfile_rows"]) == 32


# ---------------------------------------------------------------------------
# CLI: tier-0 unchanged, L48 plan CG-free
# ---------------------------------------------------------------------------
def test_default_list_and_quick_stay_nineteen_with_cg() -> None:
    default = _json("--list")
    assert default["count"] == 19
    tags = {case["tag"] for case in default["cases"]}
    assert tags == set(policy.TIER0_TAGS)
    assert {"torch/solve/cg", "kupl/solve/cg"} <= tags

    quick = _json("--quick", "--list", "--l48-two-node")
    assert quick["count"] == 19
    assert not [case for case in quick["cases"]
                if "l48" in str(case.get("tag", ""))]
    assert {"torch/solve/cg", "kupl/solve/cg"} <= {
        case["tag"] for case in quick["cases"]}


def test_l48_plan_retires_cg_but_keeps_the_rest() -> None:
    payload = _json("--mode", "list", "--lattice-preset", "L48")
    rows = payload["rows"]["items"]
    tags = {row["tag"] for row in rows}
    assert len(rows) == 19 - len(l48.CG_TAGS) == 17
    assert not [tag for tag in tags if tag.endswith("/cg")]
    assert set(policy.TIER0_TAGS) - set(l48.CG_TAGS) <= tags
    assert {tuple(row["lattice"]) for row in rows} == {(48, 48, 48, 96)}
    assert payload["retired_groups"][0]["status"] == "RETIRED"


def test_explicit_cg_on_l48_is_a_usage_error() -> None:
    for arguments in (
            ("--mode", "list", "--lattice-preset", "L48", "--solver", "cg"),
            ("--mode", "list", "--lattice-preset", "L48",
             "--only", "torch/solve/cg"),
    ):
        completed = _run_cli(*arguments)
        assert completed.returncode == 2, (
            f"{arguments} should be a usage error, got "
            f"{completed.returncode}: {completed.stdout[-500:]}")
        assert "retired L48 CG" in completed.stderr


def test_mode_list_emits_the_l48_rows_with_declared_parameters() -> None:
    payload = _json("--mode", "list", "--l48-two-node", "--only", "kupl/l48")
    rows = payload["rows"]["items"]
    assert len(rows) == len(l48.DECLARATIONS)
    assert [row["tag"] for row in rows] == [case.tag for case in l48.cases()]
    for row in rows:
        declaration = l48.get(row["tag"])
        assert row["lattice"] == [48, 48, 48, 96]
        assert row["dtype"] == declaration.dtype
        assert row["ranks"] == 32
        assert row["lattice_grid"] == [2, 2, 2, 4]
        assert row["nodes"] == 2
        assert row["mg_levels"] == declaration.mg_levels
        assert row["restart"] == declaration.restart
        assert row["solver"] == declaration.solver
        assert row["l48_group"] == declaration.group
        assert row["l48_evidence"] == declaration.evidence
    manifest = payload["l48_two_node"]
    assert manifest["lattice"] == [48, 48, 48, 96]
    assert manifest["grid"] == [2, 2, 2, 4]
    assert manifest["ranks"] == 32 and manifest["nodes"] == 2
    assert manifest["retired"]["status"] == "RETIRED"
    assert "cg" in manifest["forbidden_groups"]
    assert "strict-mg" in manifest["forbidden_groups"]
    assert tuple(manifest["case_tags"]) == tuple(
        case.tag for case in l48.cases())


def test_l48_flag_needs_a_full_mode_or_list() -> None:
    completed = _run_cli("--quick", "--l48-two-node")
    assert completed.returncode == 2
    assert "full mode" in completed.stderr


def test_l48_matrix_has_no_strict_or_cg_case_in_the_cli() -> None:
    payload = _json("--list", "--l48-two-node")
    tags = [str(case["tag"]) for case in payload["cases"]]
    assert len(tags) == 19 + len(l48.DECLARATIONS)
    assert not [tag for tag in tags if tag.endswith("/cg") and "l48" in tag]
    assert not [tag for tag in tags if "strict" in tag and "l48" in tag]
    assert payload["retired_groups"][0]["group"] == "l48/cg"


# ---------------------------------------------------------------------------
# driver routing (no MPI needed)
# ---------------------------------------------------------------------------
def test_driver_owns_l48_rows_and_reports_the_launch_gap() -> None:
    from pyqcu.test import driver

    assert driver.parse_tag("kupl/l48/bicgstab/c64") == ("kupl", "solve")
    assert driver.parse_tag("kupl/l48/legacy-mg/c64") == ("kupl", "mg")
    assert driver.parse_tag("kupl/solve/cg") is None

    result = driver.run_row(
        row={"tag": "kupl/l48/bicgstab/c64", "lattice": [8, 8, 8, 16],
             "dtype": "c64", "solver": "all"},
        options={"repeat": 1, "warmup": 1},
        tag="kupl/l48/bicgstab/c64")
    assert result is not None
    assert result["status"] == "SKIP"
    assert "32 rank" in result["reason"]
    assert result["lattice"] == [48, 48, 48, 96]
    assert result["dtype"] == "c64"
    assert result["ranks"] == 32
    assert result["lattice_grid"] == [2, 2, 2, 4]
    assert result["nodes"] == 2
    assert result["l48_group"] == "bicgstab"
    assert result["l48_evidence"] == "UNVERIFIED"
    assert result["launcher"]["mpi_launch"] is False
    assert result["launcher"]["nodes"] == 2
    assert result["launcher"]["mpirun"][0] == "mpirun"


def test_driver_rejects_the_io_group_without_real_input() -> None:
    from pyqcu.test import driver

    result = driver.run_row(
        row={"tag": "kupl/l48/io-residual/c64"}, options={},
        tag="kupl/l48/io-residual/c64")
    assert result is not None
    assert result["status"] == "ERROR"
    assert "--gauge-file" in result["reason"]
    assert "--source-file" in result["reason"]


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------
def test_docs_page_documents_the_matrix_and_the_retirement() -> None:
    page = ROOT / "docs" / "pyqcu_test_l48_two_node_matrix.md"
    assert page.is_file(), "the L48 matrix docs page is missing"
    text = page.read_text(encoding="utf-8")
    for group in l48.GROUP_NAMES:
        assert group in text, f"{group} is not documented"
    assert "2x2x2x4" in text
    assert "1x2x2x8" in text
    assert "L48 CG" in text or "l48/cg" in text
    assert "Strict-MG" in text
    assert "retired" in text.lower()
    design = (ROOT / "docs" / "pyqcu_test_l48_remote.md").read_text(
        encoding="utf-8")
    assert "pyqcu_test_l48_two_node_matrix.md" in design
    assert "L48 CG" in design or "l48/cg" in design
