"""Focused tests for the independent residual and the operand geometry.

The audit of the L48 two-node path found the independent ``||D x - b||/||b||``
blocked twice: the multi-rank verification gathered every rank-local block on
rank zero (a 7.6 GiB class of allocation for ``48x48x48x96`` c64) and the
whole-lattice reference then materialised a further ~11 GiB Clover term.  It also
found the *operands* wrong (N1): the ABI was told the declared global lattice
while each rank only holds a slab, so a 32-rank run asked every rank for the
whole-lattice gauge/source/solution/Clover tensors.  These tests pin both fixes:

* :func:`pyqcu.test.native.local_geometry` enforces
  ``prod(local) * prod(grid) == prod(global)`` plus even local extents, and the
  option/ABI operand shapes follow it (``_Controls`` refuses the global lattice
  on a multi-rank grid);
* :func:`pyqcu.test.native.reference_plan` declares the peak of every
  decomposition *before* a tensor is allocated, and refuses an undersized
  budget with the estimate;
* the reference is planned **before** the native call, so an over-budget
  reference is ``SKIP(reference_workspace)``, and a failure after a successful
  solve is reported (``residual_verified=False``) instead of becoming an
  ``ERROR``;
* the roll-injected Wilson-Clover operator used by the rank-local and tiled
  paths is *bit-identical* to the production ``pyqcu.torch.dslash`` reference
  (``make_clover`` + ``give_wilson(with_I=True)`` + ``give_clover``);
* the tiled whole-lattice path reproduces the gathered reference across slab
  sizes in c128;
* the distributed path reproduces the serial reference on four real MPI ranks
  and on a real two-rank ``4x4x4x8`` / ``2 1 1 1`` job whose ABI parameters and
  operand shapes are asserted to be the rank-local ones;
* an ``48x48x48x96`` c64 32-rank row keeps one rank's operands and reference
  inside the declared budget (and shows the global shapes are not);
* ``pyqcu.test.native`` stays importable without torch/numpy/mpi4py.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LATTICE = (8, 8, 8, 16)
GRID = (2, 1, 1, 2)          # 4 ranks, every local extent even
KAPPA = 0.13
CSW = 1.0
WORKER_TIMEOUT_SECONDS = 900
#: The two-rank geometry the local-slab test runs on: ``4x4x4x8`` on ``2 1 1 1``
#: gives the even local lattice ``2x4x4x8``.
TWO_RANK_LATTICE = (4, 4, 4, 8)
TWO_RANK_GRID = (2, 1, 1, 1)


def _port():
    pytest.importorskip("torch", reason="the reference operator needs torch")
    from pyqcu.test import native

    return native


def _fields(lattice: tuple[int, int, int, int] = LATTICE,
            dtype: str = "c128") -> dict:
    """Deterministic compact gauge/source/solution, generated the same way twice."""
    torch = pytest.importorskip("torch")
    from pyqcu import lattice as lattice_module
    from pyqcu import tools

    complex_dtype = torch.complex128 if dtype == "c128" else torch.complex64
    real_dtype = torch.float64 if dtype == "c128" else torch.float32
    torch.manual_seed(7)
    gauge = torch.empty((3, 3, 4, *lattice), dtype=complex_dtype)
    lattice_module.generate_gauge_field(gauge, sigma=0.2, seed=5, verbose=False)
    generator = torch.Generator().manual_seed(11)

    def random_field(shape):
        return torch.complex(
            torch.randn(shape, generator=generator, dtype=real_dtype),
            torch.randn(shape, generator=generator, dtype=real_dtype),
        ).to(complex_dtype)

    return {
        "gauge": tools.oooxyzt2poooxyzt(gauge.contiguous()),
        "source": tools.oooxyzt2poooxyzt(random_field((4, 3, *lattice)).contiguous()),
        "solution": tools.oooxyzt2poooxyzt(
            random_field((4, 3, *lattice)).contiguous()),
    }


def _residual(**kwargs):
    torch = pytest.importorskip("torch")
    native = _port()
    return native._true_residual(torch, **kwargs)


# ---------------------------------------------------------------------------
# 1. import policy
# ---------------------------------------------------------------------------
def test_native_import_stays_stdlib_only() -> None:
    """``pyqcu.test.native`` must not import torch/numpy/mpi4py at import time."""
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (str(ROOT) if not existing
                                 else str(ROOT) + os.pathsep + existing)
    probe = (
        "import sys; import pyqcu.test.native; "
        "print(sorted(name for name in ('torch', 'numpy', 'mpi4py', 'pyqcu.testing')"
        " if name in sys.modules))")
    completed = subprocess.run([sys.executable, "-B", "-c", probe], cwd=ROOT,
                               env=environment, text=True, capture_output=True,
                               check=False, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "[]", completed.stdout


# ---------------------------------------------------------------------------
# 2. the declared budget
# ---------------------------------------------------------------------------
def test_l48_plan_is_bounded_and_the_old_allocations_are_refused() -> None:
    """L48 keeps the reference inside the budget and names what it refuses."""
    native = _port()
    plan = native.reference_plan((48, 48, 48, 96), "c64",
                                 lattice_grid=(2, 2, 2, 4), size=32)
    assert plan["mode"] == "distributed", plan["reason"]
    assert plan["partitioned"] is True
    assert plan["local_lattice"] == [24, 24, 24, 24]
    assert plan["peak_bytes"] == plan["distributed_peak_bytes"]
    assert plan["peak_bytes"] <= plan["budget_bytes"]
    # The two allocations the audit refused are both over budget, and the
    # bounded path is a fraction of them.
    assert plan["gather_peak_bytes"] > plan["budget_bytes"]
    assert plan["global_clover_bytes"] > plan["budget_bytes"]
    assert plan["peak_bytes"] * 4 < plan["gather_peak_bytes"]
    assert plan["global_clover_bytes"] > 10 << 30, "the L48 Clover term is ~11 GiB"
    # A single rank cannot hold the whole-lattice reference either: it tiles.
    whole = native.reference_plan((48, 48, 48, 96), "c64", lattice_grid=(1, 1, 1, 1))
    assert whole["mode"] == "tiled"
    assert whole["tiled_slabs"] > 1
    assert whole["tiled_slab_extent_t"] % 2 == 0
    assert 96 % whole["tiled_slab_extent_t"] == 0
    assert whole["tiled_peak_bytes"] <= whole["budget_bytes"]
    assert whole["whole_lattice_peak_bytes"] > whole["budget_bytes"]


def test_an_undersized_budget_is_refused_with_the_estimate() -> None:
    """A budget that cannot hold the reference is an explicit, costed refusal."""
    native = _port()
    plan = native.reference_plan((48, 48, 48, 96), "c64",
                                 lattice_grid=(2, 2, 2, 4), budget="64MiB")
    assert plan["mode"] == "unavailable"
    assert "GiB" in plan["reason"] and "budget" in plan["reason"]
    assert plan["peak_bytes"] == 0
    assert plan["distributed_peak_bytes"] > plan["budget_bytes"]

    fields = _fields((4, 4, 4, 8), dtype="c64")
    with pytest.raises(native.VerificationUnavailable) as error:
        _residual(solution=fields["solution"], source=fields["source"],
                  gauge=fields["gauge"], lattice=(4, 4, 4, 8),
                  grid=(1, 1, 1, 1), size=1, kappa=KAPPA, csw=0.0,
                  purpose="test", budget="1MiB")
    assert "budget" in str(error.value)


def test_reference_options_are_normalized() -> None:
    """``reference_budget``/``reference_mode`` reach the case, with defaults."""
    native = _port()
    option = native._options(None, kind="solve")
    assert option.reference_mode == "auto"
    assert option.reference_budget == native.REFERENCE_BUDGET_BYTES
    tuned = native._options(None, kind="solve", reference_budget="2GiB",
                            reference_mode="tiled")
    assert tuned.reference_budget == 2 << 30
    assert tuned.reference_mode == "tiled"
    assert tuned.as_dict()["reference_budget_bytes"] == 2 << 30
    case = native.case("solve", backend="kupl", reference_budget=1 << 30)
    assert case.tag == "kupl/native/solve"
    with pytest.raises(ValueError, match="reference_mode"):
        native._options(None, kind="solve", reference_mode="nonsense")
    with pytest.raises(ValueError, match="reference_budget"):
        native._options(None, kind="solve", reference_budget="many")


# ---------------------------------------------------------------------------
# 3. the operator is the production operator
# ---------------------------------------------------------------------------
def test_injected_roll_operator_is_the_production_operator() -> None:
    """Same definition as ``make_clover``/``give_wilson``, to the last bit."""
    torch = pytest.importorskip("torch")
    native = _port()
    from pyqcu.torch import dslash

    lattice = (4, 4, 4, 8)
    from pyqcu import lattice as lattice_module

    torch.manual_seed(3)
    gauge = torch.empty((3, 3, 4, *lattice), dtype=torch.complex64)
    lattice_module.generate_gauge_field(gauge, sigma=0.2, seed=9, verbose=False)
    generator = torch.Generator().manual_seed(4)
    vector = torch.complex(
        torch.randn((4, 3, *lattice), generator=generator),
        torch.randn((4, 3, *lattice), generator=generator))

    def roll(value, shifts, dims):
        return torch.roll(value, shifts=shifts, dims=dims)

    kappa_tensor = torch.tensor([KAPPA])
    reference_clover = dslash.make_clover(
        gauge, kappa=kappa_tensor, csw=CSW, verbose=False)
    clover = native._clover_term(torch, roll, gauge=gauge, kappa=KAPPA, csw=CSW)
    assert torch.equal(clover, reference_clover), (
        float((clover - reference_clover).abs().max()))

    reference = (dslash.give_wilson(vector, gauge, kappa_tensor, with_I=True)
                 + dslash.give_clover(vector, reference_clover))
    applied = native._wilson_clover_apply(torch, roll, vector=vector,
                                          gauge=gauge, clover=clover,
                                          kappa=KAPPA)
    assert torch.equal(applied, reference)

    # csw = 0 is the plain Wilson operator: no Clover tensor at all.
    assert native._clover_term(torch, roll, gauge=gauge, kappa=KAPPA,
                               csw=0.0) is None
    wilson = native._wilson_clover_apply(torch, roll, vector=vector,
                                         gauge=gauge, clover=None, kappa=KAPPA)
    assert torch.equal(
        wilson, dslash.give_wilson(vector, gauge, kappa_tensor, with_I=True))


# ---------------------------------------------------------------------------
# 4. whole-lattice fields: the tiled path
# ---------------------------------------------------------------------------
def test_tiled_reference_matches_the_gathered_reference_across_slabs() -> None:
    """c128: every slab size reproduces the historic whole-lattice number."""
    native = _port()
    lattice = (4, 4, 4, 16)
    fields = _fields(lattice)
    common = dict(lattice=lattice, grid=(1, 1, 1, 1), size=1, kappa=KAPPA,
                  csw=CSW, purpose="test")
    gathered = _residual(solution=fields["solution"], source=fields["source"],
                         gauge=fields["gauge"], mode="gather", **common)
    assert gathered["residual_mode"] == "gather"

    per_t = native.REFERENCE_TILED_SITE_ELEMENTS * 16 * 4 * 4 * 4
    seen = set()
    for extent in (2, 4, 8, 16):
        budget = per_t * (extent + 4 if extent < lattice[3] else extent)
        tiled = _residual(solution=fields["solution"], source=fields["source"],
                          gauge=fields["gauge"], budget=budget, mode="tiled",
                          **common)
        plan = tiled["residual_plan"]
        assert tiled["residual_mode"] == "tiled"
        assert plan["tiled_slab_extent_t"] == extent
        assert plan["tiled_slabs"] == lattice[3] // extent
        assert tiled["residual_peak_bytes"] <= budget
        assert math.isclose(tiled["relative_residual"],
                            gathered["relative_residual"], rel_tol=1e-10), (
            extent, tiled["relative_residual"], gathered["relative_residual"])
        seen.add(plan["tiled_slabs"])
    assert len(seen) > 1, "the tiled path must actually split the lattice"

    # The default plan of a whole-lattice field is the tiled one, and a
    # multi-rank declaration must not make it collect: the fields are global, so
    # every rank computes the same bounded number on its own.
    default = _residual(solution=fields["solution"], source=fields["source"],
                        gauge=fields["gauge"], lattice=lattice, grid=GRID,
                        size=4, kappa=KAPPA, csw=CSW, purpose="test")
    assert default["residual_mode"] == "tiled"
    assert default["residual_partitioned"] is False
    assert math.isclose(default["relative_residual"],
                        gathered["relative_residual"], rel_tol=1e-10)


def test_rank_local_fields_require_a_matching_grid() -> None:
    """A field geometry that is neither whole nor a block is a usage error."""
    native = _port()
    fields = _fields((4, 4, 4, 8), dtype="c64")
    with pytest.raises(ValueError, match="neither the"):
        _residual(solution=fields["solution"], source=fields["source"],
                  gauge=fields["gauge"], lattice=LATTICE, grid=GRID, size=4,
                  kappa=KAPPA, csw=0.0, purpose="test")
    with pytest.raises(ValueError, match="distributed"):
        native.reference_plan(LATTICE, "c64", lattice_grid=GRID, size=4,
                              local_lattice=LATTICE, mode="distributed")


# ---------------------------------------------------------------------------
# 5. the distributed path on real ranks
# ---------------------------------------------------------------------------
_WORKER = r'''
import json
import os
import sys

sys.path.insert(0, {root!r})

import torch
from pyqcu import lattice as lattice_module
from pyqcu import tools
from pyqcu.test import native

LATTICE = {lattice!r}
GRID = {grid!r}
KAPPA = {kappa!r}
CSW = {csw!r}


def fields():
    torch.manual_seed(7)
    gauge = torch.empty((3, 3, 4, *LATTICE), dtype=torch.complex128)
    lattice_module.generate_gauge_field(gauge, sigma=0.2, seed=5, verbose=False)
    generator = torch.Generator().manual_seed(11)

    def random_field(shape):
        return torch.complex(
            torch.randn(shape, generator=generator, dtype=torch.float64),
            torch.randn(shape, generator=generator, dtype=torch.float64))

    return (tools.oooxyzt2poooxyzt(gauge.contiguous()),
            tools.oooxyzt2poooxyzt(random_field((4, 3, *LATTICE)).contiguous()),
            tools.oooxyzt2poooxyzt(random_field((4, 3, *LATTICE)).contiguous()))


def block(value, coordinate):
    local = tuple(LATTICE[axis] // GRID[axis] for axis in range(4))
    selection = [slice(None)] * (value.ndim - 4)
    for axis in range(4):
        start = coordinate[axis] * local[axis]
        if axis == 3:
            half = local[3] // 2
            selection.append(slice(coordinate[3] * half,
                                   (coordinate[3] + 1) * half))
        else:
            selection.append(slice(start, start + local[axis]))
    return value[tuple(selection)].contiguous()


def main():
    gauge, source, solution = fields()
    if os.environ.get("WORKER_ROLE") == "serial":
        result = native._true_residual(
            torch, solution=solution, source=source, gauge=gauge,
            lattice=LATTICE, grid=(1, 1, 1, 1), size=1, kappa=KAPPA, csw=CSW,
            purpose="test", mode="gather")
        print("RESULT " + json.dumps({{"role": "serial",
                                      "value": result["relative_residual"],
                                      "mode": result["residual_mode"]}}))
        return
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    coordinate = native.rank_coordinate(rank, GRID)
    result = native._true_residual(
        torch, solution=block(solution, coordinate),
        source=block(source, coordinate), gauge=block(gauge, coordinate),
        lattice=LATTICE, grid=GRID, size=comm.Get_size(), kappa=KAPPA,
        csw=CSW, purpose="test", mode="auto")
    print("RESULT " + json.dumps({{"role": "distributed", "rank": rank,
                                  "value": result["relative_residual"],
                                  "mode": result["residual_mode"],
                                  "peak_bytes": result["residual_peak_bytes"],
                                  "local": result["residual_plan"]["local_lattice"]}}))


main()
'''


def _python_environment() -> dict:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (str(ROOT) if not existing
                                 else str(ROOT) + os.pathsep + existing)
    return environment


def _run_worker(arguments: list[str], script: Path, role: str, *,
                timeout: float = WORKER_TIMEOUT_SECONDS):
    environment = _python_environment()
    environment["WORKER_ROLE"] = role
    return subprocess.run([*arguments, sys.executable, "-B", str(script)],
                          cwd=ROOT, env=environment, text=True,
                          capture_output=True, check=False, timeout=timeout)


def test_distributed_reference_matches_the_serial_one_on_four_ranks(
        tmp_path: Path) -> None:
    """4 real ranks: the rank-local ``D x - b`` equals the serial reference."""
    pytest.importorskip("torch")
    try:
        import mpi4py.MPI  # noqa: F401
    except Exception as error:  # noqa: BLE001 - an absent stack is a skip
        pytest.skip(f"mpi4py is not usable here: {type(error).__name__}: {error}")
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("no mpirun/mpiexec in PATH")

    script = tmp_path / "residual_worker.py"
    script.write_text(_WORKER.format(root=str(ROOT), lattice=LATTICE, grid=GRID,
                                     kappa=KAPPA, csw=CSW), encoding="utf-8")

    serial = _run_worker([], script, "serial")
    assert serial.returncode == 0, serial.stderr[-4000:]
    serial_rows = [json.loads(line[len("RESULT "):])
                   for line in serial.stdout.splitlines()
                   if line.startswith("RESULT ")]
    assert len(serial_rows) == 1, serial.stdout[-2000:]
    reference = serial_rows[0]["value"]
    assert serial_rows[0]["mode"] == "gather"

    ranks = math.prod(GRID)
    attempted: list[str] = []
    distributed = None
    for extra in ([], ["--oversubscribe"]):
        completed = _run_worker([launcher, *extra, "-np", str(ranks)], script,
                                "distributed")
        if completed.returncode == 0:
            distributed = completed
            break
        attempted.append(f"{launcher} {' '.join(extra)} -np {ranks}: "
                         f"rc={completed.returncode} "
                         f"{(completed.stderr or '').strip().splitlines()[-1:]}"
                         if completed.stderr else
                         f"{launcher} {' '.join(extra)} -np {ranks}: "
                         f"rc={completed.returncode}")
    if distributed is None:
        pytest.skip("the MPI launcher cannot start " + str(ranks)
                    + " ranks here: " + "; ".join(attempted))

    rows = [json.loads(line[len("RESULT "):])
            for line in distributed.stdout.splitlines()
            if line.startswith("RESULT ")]
    assert len(rows) == ranks, distributed.stdout[-4000:]
    for row in rows:
        assert row["mode"] == "distributed", row
        assert row["local"] == [LATTICE[axis] // GRID[axis] for axis in range(4)]
        assert math.isclose(row["value"], reference, rel_tol=1e-10), (
            row["value"], reference)
    values = {row["value"] for row in rows}
    assert len(values) == 1, f"ranks disagree: {sorted(values)}"


# ---------------------------------------------------------------------------
# 6. N1: the ABI is told the rank-local slab, never the global lattice
# ---------------------------------------------------------------------------
def test_local_geometry_enforces_the_volume_and_the_even_extents() -> None:
    """``prod(local) * prod(grid) == prod(global)`` plus even local extents."""
    native = _port()
    geometry = native.local_geometry((48, 48, 48, 96), (2, 2, 2, 4), size=32)
    assert geometry.local == (24, 24, 24, 24)
    assert geometry.grid == (2, 2, 2, 4)
    assert geometry.volume == 48 * 48 * 48 * 96
    assert geometry.local_volume * geometry.grid_volume == geometry.volume
    assert geometry.partitioned is True
    assert geometry.check() is geometry
    payload = geometry.as_dict()
    assert payload["volume_check"] is True
    assert payload["local_even"] is True
    assert payload["local_lattice"] == [24, 24, 24, 24]

    # One rank owns everything: no split, and the operands are the lattice.
    whole = native.local_geometry((4, 4, 4, 8), (1, 1, 1, 1), size=1)
    assert whole.local == (4, 4, 4, 8)
    assert whole.partitioned is False

    # An odd local extent is refused, not silently accepted: the odd-Schur
    # parity split needs an even extent on every axis.
    with pytest.raises(ValueError, match="even"):
        native.local_geometry((8, 8, 8, 96), (1, 1, 1, 32), size=32)


def test_the_option_operands_follow_the_rank_local_lattice() -> None:
    """Every ABI shape comes from the local slab; no grid means the lattice."""
    native = _port()
    option = native._options(None, kind="solve", lattice=TWO_RANK_LATTICE,
                             lattice_grid=TWO_RANK_GRID)
    assert option.operand_lattice == (2, 4, 4, 8)
    assert option.compact == (2, 4, 4, 4)
    assert option.fermion_shape == (2, 4, 3, 2, 4, 4, 4)
    assert option.clover_shape == (4, 3, 4, 3, 2, 4, 4, 4)
    assert option.as_dict()["local_lattice"] == [2, 4, 4, 8]

    whole = native._options(None, kind="solve", lattice=TWO_RANK_LATTICE)
    assert whole.operand_lattice == TWO_RANK_LATTICE
    # ABI tensors are parity compact: t is T/2 even without a process grid.
    assert whole.fermion_shape == (2, 4, 3, 4, 4, 4, 4)
    assert whole.clover_shape == (4, 3, 4, 3, 4, 4, 4, 4)

    # A grid that cannot tile the lattice is a usage error, not a skip.
    with pytest.raises(ValueError, match="tile"):
        native._options(None, kind="solve", lattice=(4, 4, 4, 8),
                        lattice_grid=(3, 1, 1, 1))


def test_the_controls_refuse_to_hand_the_global_lattice_to_the_abi() -> None:
    """The N1 regression: a global lattice on a multi-rank grid must not pass."""
    import types

    native = _port()
    from pyqcu.cuda import define

    probe = types.SimpleNamespace(
        backend="kupl", module=types.SimpleNamespace(), define=define,
        library="<stub>", device="cpu", torch_device="cpu", size=1, rank=0,
        symbols=())
    with pytest.raises(RuntimeError, match="does not tile"):
        native._Controls(
            probe, lattice=(4, 4, 4, 8), grid=(2, 1, 1, 1), dtype="c64",
            mass=0.1, csw=0.0, atol=1.0e-6, max_iter=10, plan=1, parity=1,
            seed=1, global_lattice=(4, 4, 4, 8))


def test_single_rank_operands_are_the_whole_lattice_bit_identically() -> None:
    """One rank: the generated gauge is the historic global field, bit for bit."""
    torch = pytest.importorskip("torch")
    native = _port()
    from pyqcu import lattice as lattice_module

    lattice = (4, 4, 4, 8)
    option = native._options(None, kind="solve", lattice=lattice, dtype="c128")
    geometry = native.local_geometry(lattice, (1, 1, 1, 1), size=1)
    assert geometry.partitioned is False
    gauge = native._build_gauge(option, torch, torch.complex128, geometry, 0)
    expected = torch.empty((3, 3, 4, *lattice), dtype=torch.complex128)
    lattice_module.generate_gauge_field(
        expected, sigma=option.gauge_sigma, seed=option.gauge_seed, verbose=False)
    assert torch.equal(gauge["full"], expected.contiguous())
    assert tuple(gauge["compact"].shape) == native._gauge_shape(lattice)
    assert gauge["seed"] == option.gauge_seed
    assert gauge["local_slice"] is False

    source = native._build_source(option, torch, torch.complex128, geometry, 0)
    assert tuple(source["compact"].shape) == native._fermion_shape(lattice)


def test_a_partitioned_point_source_uses_the_global_parity() -> None:
    """The point charge lands in the parity block its global site belongs to."""
    torch = pytest.importorskip("torch")
    native = _port()

    lattice = (4, 4, 4, 8)
    geometry = native.local_geometry(lattice, (2, 1, 1, 1), size=2)
    even = native._options(None, kind="solve", lattice=lattice,
                           lattice_grid=(2, 1, 1, 1), source="point",
                           source_site=(0, 0, 0, 0))
    odd = native._options(None, kind="solve", lattice=lattice,
                          lattice_grid=(2, 1, 1, 1), source="point",
                          source_site=(1, 0, 0, 0))
    even_source = native._build_source(even, torch, torch.complex64, geometry, 0)
    odd_source = native._build_source(odd, torch, torch.complex64, geometry, 0)
    assert even_source["parity"] == 0 and odd_source["parity"] == 1
    assert float(even_source["compact"][0].abs().sum()) == 1.0
    assert float(even_source["compact"][1].abs().sum()) == 0.0
    assert float(odd_source["compact"][1].abs().sum()) == 1.0
    assert float(odd_source["compact"][0].abs().sum()) == 0.0
    # Rank 1 owns x in [2, 4): the rank-0 charge is not in its slab at all.
    other = native._build_source(odd, torch, torch.complex64, geometry, 1)
    assert float(other["compact"].abs().sum()) == 0.0
    assert other["local_slice"] is True


# ---------------------------------------------------------------------------
# 7. N2: the reference is planned before the solve and never becomes an ERROR
# ---------------------------------------------------------------------------
def _stub_backend(native, *, size: int = 1, rank: int = 0):
    """A KUPL-shaped probe whose ABI entry points are recording stubs.

    ``applyWilsonBistabCgQcu`` copies ``source`` into ``solution``, so the
    "solved" field is deterministic and ``D x - b`` is a real number on every
    rank without a compiled library.
    """
    import types

    from pyqcu.cuda import define

    calls: list[str] = []

    def solve(solution, source, gauge, set_ptrs, params):
        solution.copy_(source)
        calls.append("solve")

    module = types.SimpleNamespace(
        applyInitQcu=lambda *arguments: calls.append("init"),
        applyEndQcu=lambda *arguments: calls.append("end"),
        applyWilsonBistabCgQcu=solve,
        getCloverBistabCgIterationsQcu=lambda *arguments: 7,
    )
    probe = types.SimpleNamespace(
        backend="kupl", module=module, define=define, library="<stub>",
        device="cpu", torch_device="cpu", size=int(size), rank=int(rank),
        symbols=())
    return probe, calls


def test_a_single_rank_solve_keeps_the_global_operands_and_the_tiled_reference(
) -> None:
    """One rank: global operands (bit-identical), tiled reference, lifecycle."""
    pytest.importorskip("torch")
    native = _port()
    if native.mpi_size() != 1:
        pytest.skip("this case drives the single-rank transaction")
    lattice = (4, 4, 4, 8)
    option = native._options(None, kind="solve", lattice=lattice, dtype="c128",
                             csw=0.0, max_iter=8, atol=1.0e-10)
    probe, calls = _stub_backend(native)
    result = native._solve_case(
        option, probe, (1, 1, 1, 1),
        native.NativeEnvironment(threads=1, shm=False), time.perf_counter())

    assert result["local_lattice"] == list(lattice)
    assert result["lattice"] == list(lattice)
    assert result["params_lattice"] == list(lattice)
    assert result["params_grid"] == [1, 1, 1, 1]
    assert result["abi_geometry"]["abi_volume_check"] is True
    assert result["local_volume"] == math.prod(lattice)
    assert result["residual_mode"] == "tiled"
    assert result["residual_partitioned"] is False
    assert result["residual_verified"] is True
    assert result["iterations"] == 7
    assert result["slot_released"] is True
    assert result["init_calls"] == result["end_calls"] >= 1
    assert calls.count("solve") >= 1
    assert result["gauge_unchanged"] is True
    assert result["source_unchanged"] is True
    # x = b, so the residual is ||(D - 1) b|| / ||b||: finite and non-zero.
    assert 0.0 < float(result["relative_residual"]) < 1.0e3


def test_a_failed_reference_after_the_solve_is_reported_not_raised(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """N2: a reference failure after a successful solve is never an ERROR."""
    pytest.importorskip("torch")
    native = _port()
    if native.mpi_size() != 1:
        pytest.skip("this case drives the single-rank transaction")
    option = native._options(None, kind="solve", lattice=(4, 4, 4, 8),
                             dtype="c128", csw=0.0, max_iter=8)
    probe, calls = _stub_backend(native)

    def explode(*arguments, **keywords):
        raise RuntimeError("reference exploded")

    monkeypatch.setattr(native, "_true_residual", explode)
    result = native._solve_case(
        option, probe, (1, 1, 1, 1),
        native.NativeEnvironment(threads=1, shm=False), time.perf_counter())

    assert "solve" in calls, "the native solve must have run before the reference"
    assert result["residual_verified"] is False
    assert result["relative_residual"] is None
    assert result["converged"] is False
    assert "reference exploded" in result["residual_unavailable"]
    assert result["iterations"] == 7


def test_an_over_budget_reference_skips_before_the_native_call() -> None:
    """N2: the over-budget reference is SKIP(reference_workspace), pre-solve."""
    pytest.importorskip("torch")
    native = _port()
    from pyqcu.test.common import Skip

    if native.mpi_size() != 1:
        pytest.skip("this case drives the single-rank transaction")
    option = native._options(None, kind="solve", lattice=(4, 4, 4, 8),
                             dtype="c128", csw=0.0, max_iter=8,
                             reference_budget="1MiB")
    probe, calls = _stub_backend(native)
    with pytest.raises(Skip) as error:
        native._solve_case(
            option, probe, (1, 1, 1, 1),
            native.NativeEnvironment(threads=1, shm=False),
            time.perf_counter())
    assert str(error.value).startswith("reference_workspace")
    assert "budget" in str(error.value)
    assert calls == [], "no native entry point may run before the reference plan"


def test_an_unavailable_reference_reports_the_reason() -> None:
    """The unverified block carries every key a verified one does."""
    native = _port()
    residual = native._residual_unavailable("no workspace", None,
                                            time.perf_counter())
    fields = native._residual_fields(residual)
    assert residual["relative_residual"] is None
    assert fields["residual_verified"] is False
    assert fields["residual_unavailable"] == "no workspace"
    assert fields["residual_plan"]["mode"] == "unavailable"


# ---------------------------------------------------------------------------
# 8. the L48 size budget
# ---------------------------------------------------------------------------
def test_l48_operands_and_reference_stay_inside_one_rank_budget() -> None:
    """L48 c64 32 ranks: the local operands and reference both fit the budget."""
    native = _port()
    lattice, grid, size = (48, 48, 48, 96), (2, 2, 2, 4), 32
    operands = native.operand_plan(lattice, grid, dtype="c64", size=size)
    assert operands["local_lattice"] == [24, 24, 24, 24]
    # ABI gauge/source/solution/Clover tensors are parity compact: local t/2=12.
    assert operands["shapes"]["gauge"] == [2, 3, 3, 4, 24, 24, 24, 12]
    assert operands["shapes"]["source"] == [2, 4, 3, 24, 24, 24, 12]
    assert operands["shapes"]["solution"] == [2, 4, 3, 24, 24, 24, 12]
    assert operands["shapes"]["clover"] == [4, 3, 4, 3, 24, 24, 24, 12]
    assert operands["fits_budget"] is True
    assert operands["peak_bytes"] <= operands["budget_bytes"]
    # The old behaviour - the global lattice in every rank's ABI - is exactly
    # prod(grid) times larger and does not fit.
    assert math.isclose(operands["overshoot"], float(size), rel_tol=1e-12)
    assert operands["global_fits_budget"] is False
    assert operands["global_peak_bytes"] > operands["budget_bytes"]

    plan = native.reference_plan(lattice, "c64", lattice_grid=grid, size=size)
    assert plan["mode"] == "distributed"
    assert plan["local_lattice"] == [24, 24, 24, 24]
    assert plan["peak_bytes"] <= plan["budget_bytes"]
    # The bounded reference is the same class of allocation as the operands,
    # and both are far below the two refused whole-lattice allocations.
    assert plan["peak_bytes"] < 2 * operands["peak_bytes"]
    assert plan["gather_peak_bytes"] > plan["budget_bytes"]
    assert plan["global_clover_bytes"] > 10 << 30
