"""``--reference-budget`` must reach the runner kwargs that plan the reference.

The independent residual is planned against a per-rank workspace budget before
the native call, and ``pyqcu.test.native.REFERENCE_BUDGET_BYTES`` (4 GiB) is the
only value the port could express until 2026-11-24: the option existed on
``NativeCase`` but no CLI flag carried it, and ``FullModeOptions.as_dict()`` -
the mapping ``driver.run_row`` actually reads in a full mode - had no key for
it.  Both L64 16-rank cells therefore returned ``SKIP(reference_workspace)``
("the distributed independent reference needs about 7.50 GiB per rank, above the
declared 4.00 GiB budget") with a correct launch and no solve at all.

These tests pin every hop the value has to survive: ``build_parser`` ->
``_resolve_config`` -> ``_build_test`` -> ``FullModeOptions.as_dict()`` ->
``driver.run_row`` -> ``driver._native_failure`` -> ``native.run_case(**kwargs)``.
``run_row`` is the entry point :func:`pyqcu.test.common._solve_row` calls with
``options=options.as_dict()``, so the whole chain is driven from the CLI down to
the kwargs the native runner receives - a value that stops at any one hop is the
failure mode this module exists to catch, and the ``as_dict()`` hop is the one
that was missing for ``restart``/``coarse_tol_factor`` (commit 6328e9cb).  The
gate itself is untouched: a budget only decides whether the reference may
allocate, never whether a residual counts as verified.
"""

from __future__ import annotations

import pytest

from pyqcu.test import common, driver, native

LATTICE = (64, 64, 64, 128)
GRID = (2, 2, 2, 2)
#: ``native.reference_plan`` for L64 c64 at 16 ranks: 7.50 GiB per rank, which
#: is 1.875x the 4 GiB default.  8 GiB is the smallest round budget that clears
#: it, and it is the one the two 16-rank cells were run with.
CELL_PEAK_BYTES = 8_053_063_680
CELL_BUDGET = "8GiB"

ARGV = ["--mode", "solve", "--native", "solve", "--only", "kupl/native/solve",
        "--dtype", "c64", "--lattice-size", "64", "64", "64", "128",
        "--lattice-grid", "2", "2", "2", "2", "--ranks", "16",
        "--threads", "18", "--no-shm", "--repeat", "3", "--warmup", "1",
        "--atol", "1e-5"]


def _config(argv):
    from pyqcu.test.__main__ import _resolve_config, build_parser

    return _resolve_config(build_parser().parse_args(argv))


def _run_row(options, tag, *, ranks=1):
    """``driver.run_row`` of an ``as_dict()`` document, native call recorded.

    ``run_row`` is the documented hook contract ``common._solve_row`` calls
    (``row``/``options``/``paths``/``tag``); only the module boundary
    ``native.run_case`` is replaced, so what is recorded here is exactly the
    kwargs the native runner would have received.  The row declares a single
    rank because the test process is one rank; the 16-rank arithmetic is
    ``test_the_budget_is_the_one_the_cell_needs``' job, and the budget itself
    does not depend on the rank count.
    """
    captured: dict = {}

    def fake_run_case(kind, **kwargs):
        captured.update({"kind": kind, **kwargs})
        return {"threads": kwargs.get("threads"), "seconds": 0.5}

    original = native.run_case
    native.run_case = fake_run_case
    try:
        row = {"tag": tag, "backend": "kupl", "lattice": list(LATTICE),
               "dtype": "c64", "ranks": ranks, "solver": "bistabcg",
               "dslash": "clover", "gauge": "ones", "source": "point",
               "seed": 42}
        result = driver.run_row(row=row, options=options,
                                paths=options.get("paths") or {}, tag=tag)
    finally:
        native.run_case = original
    assert "status" not in result, result
    return captured


@pytest.mark.parametrize("text, expected", [
    ("8GiB", 8 << 30),                 # binary unit, as the row estimates it
    (str(8 << 30), 8 << 30),           # a plain byte count
    ("8e9", 8_000_000_000),            # decimal exponent
    ("7500MiB", 7500 << 20),           # the bare minimum the L64 row asks for
])
def test_the_flag_accepts_every_spelling_the_runner_parses(text, expected):
    """The CLI must accept exactly what ``_resolve_reference_budget`` parses."""
    config = _config(ARGV + ["--reference-budget", text])
    assert config.reference_budget == expected
    assert config.full.reference_budget == expected
    assert native._resolve_reference_budget(text) == expected


def test_a_bad_value_is_a_usage_error_not_a_late_runner_error():
    from pyqcu.test import common as common_module

    with pytest.raises(common_module.UsageError, match="reference-budget"):
        _config(ARGV + ["--reference-budget", "many"])


def test_the_flag_survives_the_whole_full_mode_chain():
    """build_parser -> _resolve_config -> _build_test -> as_dict() -> driver.

    ``as_dict()`` is the mapping ``common._solve_row`` hands to
    ``driver.run_row`` as ``options``, and ``run_row`` is called here - not the
    private ``_native_failure`` - so the whole chain the harness walks is
    exercised and a value that reaches the ledger but not the runner is caught.
    """
    from pyqcu.test.__main__ import _build_test

    config = _config(ARGV + ["--reference-budget", CELL_BUDGET])
    test = _build_test(config)
    assert test.reference_budget == 8 << 30
    assert test.config()["reference_budget"] == 8 << 30

    document = config.full.as_dict()
    assert document["reference_budget"] == 8 << 30

    captured = _run_row(document, "kupl/native/solve")
    assert captured["kind"] == "solve"
    assert captured["reference_budget"] == 8 << 30
    # The other run-level knobs of the same hop, so a regression that drops the
    # whole settings block cannot hide behind this test.
    assert captured["atol"] == pytest.approx(1.0e-5)
    assert captured["repeat"] == 3
    assert captured["warmup"] == 1


def test_an_unset_budget_is_not_forwarded():
    """``None`` must stay unset: the runner keeps ``REFERENCE_BUDGET_BYTES``."""
    config = _config(ARGV)
    assert config.reference_budget is None
    assert config.full.as_dict()["reference_budget"] is None
    assert "reference_budget" not in common.Options().as_kwargs()

    captured = _run_row(config.full.as_dict(), "kupl/native/solve")
    assert "reference_budget" not in captured


def test_test_run_forwards_the_budget_into_the_per_case_options(monkeypatch):
    """The plain (non-full-mode) path: ``Test.run`` -> ``Options`` -> kwargs.

    ``_build_test`` pins the budget on the ``Test``; a plain invocation runs
    each case through ``Test.run``, which builds the per-case ``Options``.
    Without that hop the budget would be reported and never applied.
    """
    seen: list[dict] = []

    def fake_run_case(case, options, *, mpi_size=1):
        seen.append(dict(options.as_kwargs()))
        return common.Result(tag=case.tag, backend=case.backend,
                            kind=case.kind, status="PASS")

    monkeypatch.setattr(common, "run_case", fake_run_case)
    test = common.Test.default().select(backends="torch", tier=0,
                                        reference_budget=8 << 30)
    assert test.cases, "the selection must plan at least one case"
    test.run(repeat=1, warmup=1)
    assert seen, "no case reached the runner"
    assert all(kwargs["reference_budget"] == 8 << 30 for kwargs in seen), seen


def test_the_budget_is_the_one_the_cell_needs():
    """The flag is only useful if 8 GiB turns the L64 16-rank plan into a run.

    The estimate is the port's own arithmetic (``native.reference_plan``), not a
    hand-written constant: 7.50 GiB per rank exceeds the 4 GiB default and fits
    8 GiB, and a dslash row is not offered the budget because it has no
    reference workspace.
    """
    default = native.reference_plan(LATTICE, "c64", lattice_grid=GRID, size=16)
    assert default["mode"] == "unavailable"
    assert default["distributed_peak_bytes"] == CELL_PEAK_BYTES
    assert "7.50 GiB per rank" in default["reason"]
    assert "4.00 GiB budget" in default["reason"]

    raised = native.reference_plan(LATTICE, "c64", lattice_grid=GRID, size=16,
                                   budget=8 << 30)
    assert raised["mode"] == "distributed"
    assert raised["reason"] == ""
    assert raised["peak_bytes"] == CELL_PEAK_BYTES
    assert raised["budget_bytes"] == 8 << 30

    captured = _run_row({"reference_budget": 8 << 30}, "kupl/native/dslash")
    # A dslash row compares an operator on the whole lattice: no reference
    # workspace, so the flag is inert there rather than silently ignored.
    assert "reference_budget" not in captured


def test_the_budget_is_part_of_the_run_configuration():
    """``Test``/``Options`` agree, so the budget is reported, not just used."""
    test = common.Test.default().select(reference_budget=8 << 30)
    assert test.reference_budget == 8 << 30
    assert test.config()["reference_budget"] == 8 << 30
    # ``select()`` is not a reset: re-selecting keeps the budget unless the
    # caller overrides it (a --sweep row must not lose it).
    assert test.select(dtype="c64").reference_budget == 8 << 30
    options = common.Options(reference_budget=8 << 30)
    assert options.as_kwargs()["reference_budget"] == 8 << 30
    with pytest.raises(ValueError, match="reference_budget must be a positive"):
        common.Options(reference_budget=0)
    with pytest.raises(ValueError, match="reference_budget must be a positive"):
        common.FullModeOptions(reference_budget=-1)
