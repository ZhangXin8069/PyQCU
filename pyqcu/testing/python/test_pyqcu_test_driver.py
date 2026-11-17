"""Tests for the L48 / real-input slice of the formal ``pyqcu.test`` port.

The formal port ships as ``python -m pyqcu.test`` (no pytest inside
``pyqcu/test``); these tests are the informal adapter for the three pieces the
L48 design (`docs/pyqcu_test_l48_remote.md`) asks for and that used to be
missing:

* the ``L48`` lattice preset (``48x48x48x96``) is registered and reaches the
  row planner without touching the tier-0 ``8x8x8x16`` gate;
* ``pyqcu.test.driver.run_row`` - the hook :data:`pyqcu.test.common.MODE_HOOKS`
  has always declared - consumes ``--gauge-file``/``--source-file`` as *real*
  native inputs and reports their origin/SHA256, and it never decides
  ``PASS``/``FAIL``;
* ``pyqcu.test.native`` builds the ABI gauge/source operands **from the file**,
  so there is no silent fallback to a generated field, to a smaller lattice or
  to another dtype.

The HDF5 fixtures are written with :mod:`pyqcu.test.io` (the port's own writer,
which validates its output before publishing), so the tests also prove the
reader accepts what the writer produces.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy", reason="HDF5 fixtures need numpy")
pytest.importorskip("h5py", reason="pyqcu.test.io reads HDF5 through h5py")

ROOT = Path(__file__).resolve().parents[3]
LATTICE = (8, 8, 8, 16)
THREADS = 2
CLI_TIMEOUT_SECONDS = 1800


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hdf5_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One ``8x8x8x16`` c64 gauge/source pair written by ``pyqcu.test.io``.

    The gauge is a genuine SU(3) configuration from the project generator (not
    the identity, so "the file was read" is distinguishable from "a unit gauge
    was built"), the source is a seeded complex field.
    """
    from pyqcu.test import io

    directory = tmp_path_factory.mktemp("pyqcu-test-l48-inputs")
    torch = pytest.importorskip(
        "torch", reason="the SU(3) fixture uses pyqcu.lattice's generator")

    from pyqcu import lattice as lattice_module

    gauge = torch.empty((3, 3, 4, *LATTICE), dtype=torch.complex64)
    lattice_module.generate_gauge_field(gauge, sigma=0.1, seed=11, verbose=False)
    gauge_array = gauge.numpy()

    rng = numpy.random.default_rng(20261117)
    source_array = (
        rng.normal(size=(4, 3, *LATTICE))
        + 1j * rng.normal(size=(4, 3, *LATTICE))
    ).astype(numpy.complex64)

    gauge_path = io.save_gauge(
        directory / "gauge_8x8x8x16_c64.h5", gauge_array,
        attrs={"generator": "pyqcu.lattice.generate_gauge_field", "seed": 11})
    source_path = io.save_source(
        directory / "source_8x8x8x16_c64.h5", source_array,
        attrs={"generator": "numpy PCG64", "seed": 20261117})
    return {
        "gauge_path": gauge_path,
        "source_path": source_path,
        "gauge": gauge_array,
        "source": source_array,
    }


def _native_case(kind: str = "solve"):
    """The native full-mode case a ``--mode solve --native <kind>`` run plans."""
    from pyqcu.test import native

    return native.case(
        kind, backend="kupl", tier=0, lattice=LATTICE, dtype="c64",
        mass=0.1, csw=0.0, threads=THREADS, shm=False)


def _row_and_options(tmp_path: Path, inputs: dict, *, ranks: int = 1,
                     gauge_file: Path | None = None,
                     source_file: Path | None = None):
    """A ledger row + full-mode options exactly as the CLI would build them."""
    from pyqcu.test import common

    cases = (_native_case(),)
    test = common.Test(cases=cases, lattice=LATTICE, dtype="c64",
                       threads=THREADS)
    options = common.FullModeOptions(
        mode="solve",
        paths=common.ModePaths.resolve(root=tmp_path / "run"),
        ranks=ranks, threads=THREADS, shm=False,
        gauge_file=str(gauge_file if gauge_file is not None
                       else inputs["gauge_path"]),
        source_file=str(source_file if source_file is not None
                        else inputs["source_path"]),
    )
    rows = common.build_rows(test, options)
    return test, options, rows[0]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pyqcu.test`` with the repository importable."""
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (str(ROOT) if not existing
                                 else str(ROOT) + os.pathsep + existing)
    return subprocess.run(
        [sys.executable, "-B", "-m", "pyqcu.test", *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        check=False, timeout=CLI_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# 1. the L48 preset
# ---------------------------------------------------------------------------
def test_l48_preset_is_registered_next_to_l24_and_l32() -> None:
    from pyqcu.test.common import LATTICE_PRESETS

    assert LATTICE_PRESETS["L48"] == (48, 48, 48, 96)
    # The older presets keep their exact geometry.
    assert LATTICE_PRESETS["L24"] == (24, 24, 24, 64)
    assert LATTICE_PRESETS["L32"] == (32, 32, 32, 64)


def test_l48_preset_reaches_the_row_planner_but_not_quick() -> None:
    """``--lattice-preset L48`` plans L48 rows; ``--quick`` stays tier-0."""
    completed = _run_cli("--mode", "list", "--lattice-preset", "L48", "--json", "-")
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    assert payload["lattice"] == [48, 48, 48, 96]
    assert payload["rows"]["count"] >= 1
    lattices = {tuple(row["lattice"]) for row in payload["rows"]["items"]}
    assert lattices == {(48, 48, 48, 96)}, lattices
    # A plan, not a run: every row is still pending.
    assert {row["status"] for row in payload["rows"]["items"]} == {None}

    # ``--quick`` still locks the tier-0 8x8x8x16 gate and rejects the preset.
    quick = _run_cli("--quick", "--lattice-preset", "L48", "--list")
    assert quick.returncode == 2, quick.stdout[-2000:]
    plain = _run_cli("--list", "--json", "-")
    assert plain.returncode == 0, plain.stderr[-2000:]
    listed = json.loads(plain.stdout)
    assert listed["count"] == 19
    assert listed["lattice"] == [8, 8, 8, 16]


# ---------------------------------------------------------------------------
# 2. the native ABI operands come from the file
# ---------------------------------------------------------------------------
def _one_rank_geometry(lattice=LATTICE):
    """The rank geometry the native builders take (one rank owns everything).

    ``pyqcu.test.native`` splits a parity-compact field per rank, so
    ``_build_gauge``/``_build_source`` are handed a
    :class:`pyqcu.test.native.LocalGeometry`; with grid ``1x1x1x1`` the local
    extents are the global ones, which is the single-rank case these tests
    describe.
    """
    from pyqcu.test import native

    return native.LocalGeometry(lattice=tuple(lattice), local=tuple(lattice),
                                grid=(1, 1, 1, 1))


def test_native_builds_the_abi_fields_from_the_hdf5_inputs(hdf5_inputs: dict) -> None:
    """``_build_gauge``/``_build_source`` read the file, not the generator."""
    torch = pytest.importorskip("torch")
    from pyqcu import tools
    from pyqcu.test import native

    option = native._options(
        None, kind="solve", lattice=LATTICE, dtype="c64",
        gauge="random", gauge_seed=999, source="point", source_seed=1234,
        gauge_file=str(hdf5_inputs["gauge_path"]),
        source_file=str(hdf5_inputs["source_path"]))
    geometry = _one_rank_geometry()

    gauge = native._build_gauge(option, torch, torch.complex64, geometry)
    source = native._build_source(option, torch, torch.complex64, geometry)

    assert gauge["origin"] == "hdf5"
    assert gauge["path"] == str(hdf5_inputs["gauge_path"])
    assert gauge["layout"] == "[color,color,direction,x,y,z,t]"
    assert gauge["sha256"], "an hdf5 input must report its SHA256"
    expected_compact = tools.oooxyzt2poooxyzt(
        torch.from_numpy(hdf5_inputs["gauge"]))
    assert torch.equal(gauge["compact"], expected_compact), (
        "the ABI gauge operand is not the file's field")
    assert torch.equal(gauge["full"], torch.from_numpy(hdf5_inputs["gauge"]))

    assert source["origin"] == "hdf5"
    assert source["sha256"], "an hdf5 input must report its SHA256"
    expected_source = tools.oooxyzt2poooxyzt(
        torch.from_numpy(hdf5_inputs["source"]))
    assert torch.equal(source["compact"], expected_source), (
        "the ABI source operand is not the file's field")


def test_a_mismatching_file_is_rejected_not_cast(hdf5_inputs: dict) -> None:
    """A wrong lattice/dtype is an error, never a silent conversion or fallback."""
    torch = pytest.importorskip("torch")
    from pyqcu.test import native

    wrong_lattice = native._options(
        None, kind="solve", lattice=(8, 8, 8, 8), dtype="c64",
        gauge_file=str(hdf5_inputs["gauge_path"]))
    with pytest.raises(ValueError, match="holds lattice"):
        native._build_gauge(wrong_lattice, torch, torch.complex64,
                            _one_rank_geometry((8, 8, 8, 8)))

    wrong_dtype = native._options(
        None, kind="solve", lattice=LATTICE, dtype="c128",
        gauge_file=str(hdf5_inputs["gauge_path"]))
    with pytest.raises(ValueError, match="complex64"):
        native._build_gauge(wrong_dtype, torch, torch.complex128,
                            _one_rank_geometry())


# ---------------------------------------------------------------------------
# 3. the driver: observations only, file origin recorded
# ---------------------------------------------------------------------------
def test_run_row_reports_the_file_origin_and_never_a_verdict(
        tmp_path: Path, hdf5_inputs: dict) -> None:
    """``run_row`` records origin/hash/threads/ranks and returns no PASS/FAIL."""
    from pyqcu.test import driver, io

    test, options, row = _row_and_options(tmp_path, hdf5_inputs)
    assert row["gauge_file"] == str(hdf5_inputs["gauge_path"])
    assert row["source_file"] == str(hdf5_inputs["source_path"])

    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert isinstance(result, dict), result

    # The numeric verdict is the harness's job: a driver returns observations.
    status = result.get("status")
    assert status is None or status in ("SKIP", "ERROR", "TIMEOUT"), (
        f"the driver decided a verdict: status={status!r}")
    if status is not None:
        # An environment skip (e.g. no KUPL library) still has to explain
        # itself and still has to carry the input identity.
        assert str(result.get("reason") or "").strip(), result

    assert result["gauge_origin"] == "hdf5"
    assert result["gauge_path"] == str(hdf5_inputs["gauge_path"])
    assert result["gauge_sha256"] == io.sha256_file(hdf5_inputs["gauge_path"])
    assert result["gauge_layout"] == "[color,color,direction,x,y,z,t]"
    assert result["gauge_lattice"] == list(LATTICE)
    assert result["gauge_file_dtype"] == "complex64"
    assert result["source_origin"] == "hdf5"
    assert result["source_sha256"] == io.sha256_file(hdf5_inputs["source_path"])

    assert result["ranks"] == 1 and result["mpi_size"] == 1
    assert result["ranks_match"] is True
    assert result["lattice_grid"] == [1, 1, 1, 1]
    assert result["local_lattice"] == list(LATTICE)
    assert result["threads_requested"] == THREADS
    assert result["launcher"]["mpi_launch"] is False
    assert "TODO" in result["launcher"]["note"] or "not implemented" in result["launcher"]["note"]

    if status is None:
        # A completed native call: the observations the case gates need are
        # there and the environment really was applied.
        assert result["threads_applied"] == THREADS
        assert result["environment"]["OMP_NUM_THREADS"] == str(THREADS)
        assert result["relative_residual"] is not None
        assert result["iterations"] > 0


def test_run_row_ignores_rows_it_does_not_own(tmp_path: Path,
                                             hdf5_inputs: dict) -> None:
    """Tier-0 tags are not native rows: ``None`` keeps the old executor path."""
    from pyqcu.test import driver

    _, options, _ = _row_and_options(tmp_path, hdf5_inputs)
    for tag in ("torch/lattice", "kupl/solve/bistabcg", "kupl/native",
                "not-a-tag", ""):
        assert driver.parse_tag(tag) is None, tag
        assert driver.run_row(row={"tag": tag}, options=options.as_dict()) is None
    assert driver.parse_tag("kupl/native/solve") == ("kupl", "solve")


def test_missing_input_is_reported_not_replaced(tmp_path: Path,
                                                hdf5_inputs: dict) -> None:
    """A requested file that does not exist is an ERROR, not a random gauge."""
    from pyqcu.test import driver

    missing = tmp_path / "does_not_exist.h5"
    _, options, row = _row_and_options(tmp_path, hdf5_inputs,
                                       gauge_file=missing)
    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert result["status"] == "ERROR"
    assert str(missing) in result["reason"]
    assert result["gauge_origin"] == "hdf5"  # requested, and not honoured
    assert result["gauge_sha256"] is None


def test_rank_mismatch_skips_instead_of_running_at_the_wrong_size(
        tmp_path: Path, hdf5_inputs: dict) -> None:
    """A row that declares more ranks than MPI provides is a SKIP, never a run."""
    from pyqcu.test import driver

    _, options, row = _row_and_options(tmp_path, hdf5_inputs, ranks=4)
    result = driver.run_row(row=row, options=options.as_dict(),
                            paths=options.paths.as_dict(), tag=row["tag"])
    assert result["status"] == "SKIP"
    assert result["skip_token"] == "mpi_grid"
    assert "mpirun" in result["reason"]
    assert result["ranks_match"] is False
    assert result["lattice_grid"] == [1, 1, 2, 2]  # derived, and reported


# ---------------------------------------------------------------------------
# 3b. the row's backend executes the row (audit N7)
# ---------------------------------------------------------------------------
def _native_row(tag: str, **fields) -> dict:
    """A minimal native ledger row, as ``build_rows`` would write it.

    ``backend`` is copied from the case (``Case.backend``), so it always equals
    the tag's first segment - exactly the pair the driver must not mix up.
    """
    backend, _, kind = tag.partition("/")
    row = {"run_id": f"unit-{kind}", "tag": tag, "backend": backend,
           "kind": kind, "lattice": list(LATTICE), "dtype": "c64", "seed": 42,
           "ranks": 1, "threads": 1, "shm": False, "solver": "all",
           "dslash": "all"}
    row.update(fields)
    return row


@pytest.mark.parametrize("tag", ["kupl/native/solve", "cuda/native/solve"])
def test_the_row_backend_reaches_native_run_case(
        tag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_case`` is handed the row's backend, not the KUPL default.

    ``pyqcu.test.native.run_case`` resolves an omitted backend to
    ``native.DEFAULT_BACKEND`` ("kupl"), so before audit N7 a
    ``cuda/native/solve`` row silently executed KUPL and was recorded as a CUDA
    result.  The call must therefore carry ``backend=<tag backend>``.
    """
    from pyqcu.test import driver

    seen: dict = {}

    def recording(kind, **kwargs):
        seen["kind"] = kind
        seen.update(kwargs)
        return {"relative_residual": 1e-12, "iterations": 4,
                "converged": True, "threads": 1, "shm": False}

    monkeypatch.setattr(driver.native, "run_case", recording)
    row = _native_row(tag)
    result = driver.run_row(row=row, options={"repeat": 1, "warmup": 1},
                            tag=tag)
    assert seen["kind"] == "solve"
    assert seen["backend"] == row["backend"] == tag.partition("/")[0], (
        f"{tag} executed backend {seen.get('backend')!r}")
    assert result["backend"] == row["backend"]
    assert "status" not in result, "a completed row must not declare a verdict"


def test_a_cuda_row_resolves_a_cuda_native_case(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``run_case`` consumes the backend: ``NativeCase.backend`` is CUDA."""
    from pyqcu.test import driver, native

    seen: dict = {}

    def fake_run_native(option):
        seen["backend"] = option.backend
        return {"relative_residual": 1e-12, "iterations": 4,
                "converged": True, "threads": 1, "shm": False}

    monkeypatch.setattr(native, "_run_native", fake_run_native)
    row = _native_row("cuda/native/solve")
    result = driver.run_row(row=row, options={"repeat": 1, "warmup": 1},
                            tag=row["tag"])
    assert seen["backend"] == "cuda", (
        "native.run_case resolved the CUDA row to the default backend")
    assert native.DEFAULT_BACKEND == "kupl", (
        "this test is only meaningful while the implicit default is KUPL")
    assert "status" not in result, result


def test_a_row_backend_that_contradicts_its_tag_is_an_error() -> None:
    """One backend must never run under another's name; no silent preference."""
    from pyqcu.test import driver

    row = _native_row("cuda/native/solve", backend="kupl")
    result = driver.run_row(row=row, options={"repeat": 1, "warmup": 1},
                            tag=row["tag"])
    assert result["status"] == "ERROR", result
    assert "contradicts" in result["reason"], result["reason"]
    assert "cuda" in result["reason"] and "kupl" in result["reason"]


# ---------------------------------------------------------------------------
# 4. the harness judges the driver's observations
# ---------------------------------------------------------------------------
def test_the_harness_rejects_a_driver_verdict_and_judges_observations() -> None:
    """A driver may not decide ``PASS``/``FAIL``; observations are judged here."""
    from pyqcu.test import common

    case = _native_case()
    row = {"run_id": "unit", "tag": case.tag, "metrics": {}}

    for verdict in ("PASS", "FAIL"):
        merged = common._apply_driver(dict(row), {"status": verdict}, case)
        assert merged["status"] == "ERROR", merged
        assert "verdict" in merged["reason"], merged

    # An execution status is allowed, but only with a reason.
    skipped = common._apply_driver(dict(row), {"status": "SKIP"}, case)
    assert skipped["status"] == "ERROR" and "reason" in skipped["reason"]

    # Observation-only results are judged by the case's own expect gates.
    good = common._apply_driver(
        dict(row), {"relative_residual": 1e-12, "iterations": 3,
                    "max_iter": 100}, case)
    assert good["status"] == "PASS", good
    assert good["metrics"]["iterations"] == 3
    bad = common._apply_driver(
        dict(row), {"relative_residual": 1.0, "iterations": 3,
                    "max_iter": 100}, case)
    assert bad["status"] == "FAIL", bad
    assert bad["failed"], bad


def test_mode_solve_judges_the_driver_result_and_writes_file_evidence(
        tmp_path: Path, hdf5_inputs: dict) -> None:
    """``--mode solve`` writes one row whose verdict the harness decided."""
    from pyqcu.test import common, io

    test, options, row = _row_and_options(tmp_path, hdf5_inputs)
    payload, code = common.run_mode("solve", test, options, rows=[row])
    assert payload["rows"]["count"] == 1
    entry = payload["rows"]["items"][0]
    assert entry["status"] in ("PASS", "FAIL", "SKIP", "ERROR"), entry
    metrics = entry["metrics"]
    assert metrics["gauge_origin"] == "hdf5"
    assert metrics["gauge_sha256"] == io.sha256_file(hdf5_inputs["gauge_path"])
    assert metrics["source_origin"] == "hdf5"
    assert metrics["lattice_grid"] == [1, 1, 1, 1]
    if entry["status"] == "PASS":
        # The harness judged it with Case.expect, so the gate keys are present
        # and the ledger validates; a driver-claimed PASS could do neither.
        assert metrics["relative_residual"] is not None
        assert metrics["iterations"] > 0
        assert code == 0
    else:
        assert str(entry.get("reason") or "").strip(), entry
    validation, _ = common.validate_ledger(options.paths, test.cases, notes=[])
    assert not validation["problems"], validation["problems"]
