"""``--atol`` must survive the full-mode path into the native runner.

``atol`` has been declared in :data:`pyqcu.test.common.CONFIG_FIELDS` and
carried by :meth:`pyqcu.test.common.Options.as_kwargs` all along, but a
``--mode solve``/``--mode run`` invocation does **not** go through ``Options``:
:func:`pyqcu.test.common._mode_solve` hands
:func:`pyqcu.test.driver.run_row` ``options=FullModeOptions.as_dict()``
(common.py, ``driver, row=..., options=options.as_dict()``), and that mapping
had no ``atol`` entry - so ``driver._native_failure`` could not forward it and
each native runner fell back to its own default.

The evidence that found it (2026-11-24): jobs 2245167 and 2245127 were both
launched with ``--atol 1e-5``, yet their rows recorded ``atol 1e-06`` (solve)
and ``atol 1e-05`` (mg) - i.e. ``native.DEFAULT_ATOL_C64`` and
``native.DEFAULT_MG_ATOL_C64``.  A "matched" MG-vs-BiCGStab comparison was
therefore solving two different problems, a decade apart in residual.

Each test below drives the real chain - ``build_parser`` -> ``_resolve_config``
-> ``_build_test`` -> ``FullModeOptions.as_dict()`` -> ``driver.run_row`` - so it
fails when any one of the four drop points is restored:

``common.FullModeOptions``   the field and its ``as_dict`` entry
``__main__.__resolve_config`` the ``--atol`` population (``args.atol``, because
                             ``config`` is resolved after the construction)
``driver._native_failure``   the kwargs entry (ungated on ``kind``)
"""

from __future__ import annotations

import pytest

from pyqcu.test.__main__ import UsageError, _build_test, _resolve_config, build_parser

LATTICE = ("8", "8", "8", "16")
GRID = ("1", "1", "1", "1")


def _argv(kind: str, *extra: str) -> list[str]:
    """The smallest ``--mode solve --native <kind>`` invocation that plans a row."""
    return ["--mode", "solve", "--native", kind, "--only", f"kupl/native/{kind}",
            "--dtype", "c64",
            "--lattice-size", *LATTICE, "--lattice-grid", *GRID,
            "--ranks", "1", *extra]


def _settings(kind: str, *extra: str) -> dict:
    """``FullModeOptions.as_dict()`` of a real CLI resolution.

    ``_build_test`` is part of the chain on purpose: a knob that reaches
    ``FullModeOptions`` but breaks the case planning is not a working knob.
    """
    config = _resolve_config(build_parser().parse_args(_argv(kind, *extra)))
    test = _build_test(config)
    assert test.cases, "the invocation must still plan at least one case"
    return config.full.as_dict()


# ---------------------------------------------------------------------------
# 1. the value arrives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["solve", "mg"])
def test_atol_reaches_full_mode_options(kind: str) -> None:
    """``--atol 1e-5`` is in ``FullModeOptions.as_dict()`` for both kinds.

    The key itself is what was missing: ``as_dict()`` is the mapping the driver
    receives, so an absent key is indistinguishable from "not requested" and the
    runner silently substituted its default.
    """
    settings = _settings(kind, "--atol", "1e-5")
    assert "atol" in settings, (
        "FullModeOptions.as_dict() has no 'atol' key, so driver.run_row can "
        f"never forward --atol; keys are {sorted(settings)}")
    assert settings["atol"] == pytest.approx(1e-5)


def test_atol_reaches_the_native_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drop point that made the 2026-11-24 rows unmatched: the kwargs.

    Uses the real ``as_dict()`` payload and intercepts only the module
    boundary (``driver.native.run_case``), so this fails if
    ``driver._native_failure`` stops forwarding what the CLI resolved - the
    layer that was broken while every other layer tested green.
    """
    from pyqcu.test import driver

    seen: dict = {}

    def recording(kind, **kwargs):
        seen["kind"] = kind
        seen.update(kwargs)
        return {"relative_residual": 1e-12, "iterations": 4,
                "converged": True, "threads": None, "shm": None}

    monkeypatch.setattr(driver.native, "run_case", recording)
    options = _settings("solve", "--atol", "1e-5")
    row = {"tag": "kupl/native/solve", "backend": "kupl",
           "lattice": [8, 8, 8, 16], "dtype": "c64", "ranks": 1,
           "solver": "bistabcg", "dslash": "clover", "gauge": "one",
           "source": "point", "seed": 1}
    result = driver.run_row(row=row, options=options,
                            paths=options["paths"], tag=row["tag"])
    assert "status" not in result, result
    assert seen["kind"] == "solve"
    assert seen.get("atol") == pytest.approx(1e-5), (
        "the native runner got no atol and fell back to its own default; "
        f"kwargs were {sorted(seen)}")


# ---------------------------------------------------------------------------
# 2. an unset flag stays unset
# ---------------------------------------------------------------------------
def test_unset_atol_keeps_the_runner_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag the override must not be sent at all.

    ``None`` and "not given" must be the same thing downstream: an explicit
    ``atol=None`` would still be *present* in the mapping, and the native
    resolver distinguishes an omitted value (per-kind default) from a supplied
    one.  Asserting ``is None`` alone would pass for a broken forwarder, so the
    kwargs are checked too.
    """
    from pyqcu.test import driver

    settings = _settings("solve")
    assert settings.get("atol") is None

    seen: dict = {}

    def recording(kind, **kwargs):
        seen.update(kwargs)
        return {"relative_residual": 1e-12, "iterations": 4,
                "converged": True, "threads": None, "shm": None}

    monkeypatch.setattr(driver.native, "run_case", recording)
    row = {"tag": "kupl/native/solve", "backend": "kupl",
           "lattice": [8, 8, 8, 16], "dtype": "c64", "ranks": 1,
           "solver": "bistabcg", "dslash": "clover", "gauge": "one",
           "source": "point", "seed": 1}
    result = driver.run_row(row=row, options=settings,
                            paths=settings["paths"], tag=row["tag"])
    assert "status" not in result, result
    assert "atol" not in seen, (
        "an unset --atol must not become an explicit override; the runner has "
        f"to keep its own default, kwargs were {sorted(seen)}")


# ---------------------------------------------------------------------------
# 3. a tolerance that cannot be a tolerance is refused before compute
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["0", "-1e-5"])
def test_a_non_positive_atol_is_a_usage_error(value: str) -> None:
    """``--atol 0`` is rejected at the CLI, exactly like ``Options``/``Test``.

    A negative value has to be spelled ``--atol=-1e-5``: argparse reads a bare
    ``-1e-5`` as another option, which is its own (earlier) usage error and not
    the one under test here.
    """
    argument = f"--atol={value}" if value.startswith("-") else "--atol"
    extra = (argument,) if argument.startswith("--atol=") else ("--atol", value)
    with pytest.raises(UsageError, match=r"--atol must be > 0"):
        _resolve_config(build_parser().parse_args(_argv("solve", *extra)))


def test_full_mode_options_validates_atol_directly() -> None:
    """A direct ``FullModeOptions`` constructor keeps the same guarantee."""
    from pyqcu.test import common

    with pytest.raises(ValueError, match=r"atol must be > 0"):
        common.FullModeOptions(mode="solve", atol=0.0)
    assert common.FullModeOptions(mode="solve", atol=1e-5).as_dict()["atol"] == 1e-5
