"""Pin the guards this session added, so a later change cannot undo them silently.

Review round 7 (`logs/v20261123.txt` section 163) recorded that five mechanisms
were correct only as far as hand verification went, and that none of them would
fail a test if broken.  This file pins the ones that are reachable without a
cluster:

  - the Kunpeng binding guard really refuses a partial binding (not just a
    missing ``KUPL_NUMA_BIND``);
  - the ranked-cache identity really carries the null-vector builder revision,
    so an artefact from the old ``v -= S^-1(S v)`` loop cannot satisfy a case
    that asks for the current builder;
  - ``dof_declared`` really distinguishes "asked for the default E" from "asked
    for nothing", which is what makes the declared-E guard meaningful.

Everything here runs on the torch path and needs no MPI, no native library and
no cluster allocation.
"""

from __future__ import annotations

import os

import pytest

from pyqcu.test import cache, native

BINDING_VARIABLES = (
    "KUPL_NUMA_BIND",
    "KUPL_MEMORY_NODE",
    "KUPL_HUGEPAGE",
    native.ALLOW_UNBOUND_VARIABLE,
)


@pytest.fixture()
def clean_binding_environment(monkeypatch):
    """Start every case from a process with none of the four variables set."""
    for name in BINDING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _binding_error():
    with pytest.raises(RuntimeError) as caught:
        native._require_kunpeng_binding("kupl", 4)
    return str(caught.value)


def test_complete_binding_is_accepted(clean_binding_environment):
    env = clean_binding_environment
    env.setenv("KUPL_NUMA_BIND", "1")
    env.setenv("KUPL_MEMORY_NODE", "local+16")
    env.setenv("KUPL_HUGEPAGE", "1")
    assert native._require_kunpeng_binding("kupl", 4) is None


@pytest.mark.parametrize("missing", ("KUPL_MEMORY_NODE", "KUPL_HUGEPAGE"))
def test_a_partial_binding_is_refused(clean_binding_environment, missing):
    """The old guard returned as soon as KUPL_NUMA_BIND was truthy.

    Each of these three runs keeps ``KUPL_NUMA_BIND=1`` and drops one of the
    other two, which is exactly the configuration the guard used to admit.
    """
    env = clean_binding_environment
    for name in BINDING_VARIABLES[1:-1]:
        env.delenv(name, raising=False)
    env.setenv("KUPL_NUMA_BIND", "1")
    for name in BINDING_VARIABLES[1:-1]:
        if name != missing:
            env.setenv(name, "local+16" if name == "KUPL_MEMORY_NODE" else "1")

    message = _binding_error()
    assert missing in message, message


def test_a_false_numa_bind_is_refused(clean_binding_environment):
    """``KUPL_NUMA_BIND=0`` is present but not truthy, so it is still missing."""
    env = clean_binding_environment
    env.setenv("KUPL_NUMA_BIND", "0")
    env.setenv("KUPL_MEMORY_NODE", "local+16")
    env.setenv("KUPL_HUGEPAGE", "1")
    assert "KUPL_NUMA_BIND" in _binding_error()


def test_nothing_set_names_all_three(clean_binding_environment):
    message = _binding_error()
    for name in BINDING_VARIABLES[:-1]:
        assert name in message, message


def test_the_escape_hatch_still_works(clean_binding_environment):
    clean_binding_environment.setenv(native.ALLOW_UNBOUND_VARIABLE, "1")
    assert native._require_kunpeng_binding("kupl", 4) is None


@pytest.mark.parametrize("world", [1, 0, -1])
def test_single_rank_and_non_kupl_are_untouched(clean_binding_environment, world):
    assert native._require_kunpeng_binding("kupl", world) is None
    assert native._require_kunpeng_binding("torch", 4) is None


def _identity(**overrides):
    values = {
        "lattice": (8, 8, 8, 16),
        "dtype": "c64",
        "dof": (12, 4),
        "mg_grid": (2, 2, 2, 2),
        "nv_iters": 2,
        "seed": 7,
        "mass": -0.277,
        "csw": 1.160920226,
        "lattice_grid": (1, 1, 1, 1),
    }
    values.update(overrides)
    return cache.mg_identity(**values)


def test_the_cache_identity_carries_the_builder_revision():
    """An artefact built by the old null-vector loop must not match.

    The old builder stored ``v -= S^-1(S v)``, i.e. solver noise, under exactly
    the same inputs as the current inverse iteration.  Without this field the two
    would share an identity and ``cache_policy=auto`` would silently measure the
    old basis.
    """
    identity = _identity()
    assert identity["nv_builder"] == cache.NULL_VECTOR_BUILDER
    assert "nv_builder" in cache.IDENTITY_FIELDS


def test_an_artefact_without_the_revision_does_not_match():
    """An old manifest has no ``nv_builder`` key at all.

    That absence is what makes a cache built by the ``v -= S^-1(S v)`` loop
    fail to match instead of silently satisfying the current case.
    """
    current = _identity()
    legacy = {k: v for k, v in current.items() if k != "nv_builder"}
    assert "nv_builder" not in legacy
    assert legacy != current


def test_the_identity_still_separates_the_parameters_that_matter():
    """Guard the guard: the fields the identity was already keyed on still work."""
    base = _identity()
    for field, value in (
        ("dof", (12, 2)),
        ("mg_grid", (2, 2, 2, 4)),
        ("dtype", "c128"),
        ("nv_iters", 3),
        ("lattice", (8, 8, 8, 32)),
    ):
        assert _identity(**{field: value}) != base, field


def _case(**explicit):
    return native._options(
        None, kind="mg", lattice=(8, 8, 8, 16), lattice_grid=(1, 1, 1, 1),
        dtype="c64", **explicit)


def test_dof_declared_separates_the_default_from_saying_nothing():
    """``dof`` alone cannot: both resolve to DEFAULT_DOF.

    This is the distinction the ranked-cache guard needs; without it
    ``--mg-dof 12 12`` (the default, asked for explicitly) skipped the check and
    the run silently used the artefact's E while the row recorded 12.
    """
    silent = _case()
    declared_default = _case(dof=(12, 12))
    declared_other = _case(dof=(12, 4))

    assert tuple(silent.dof) == tuple(declared_default.dof) == tuple(native.DEFAULT_DOF)
    assert silent.dof_declared is False
    assert declared_default.dof_declared is True
    assert declared_other.dof_declared is True
    assert declared_other.dof != declared_default.dof


def test_min_nu_pre_defaults_to_zero_and_forwards(clean_binding_environment):
    """``0`` means "keep the solver default"; the value must reach the case."""
    assert _case().mg_nu_pre == 0
    assert _case(mg_nu_pre=7).mg_nu_pre == 7


def test_a_negative_nu_pre_is_refused():
    with pytest.raises(ValueError):
        _case(mg_nu_pre=-1)
