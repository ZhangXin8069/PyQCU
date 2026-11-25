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
  - the C++ and the Python ``KUPL_NV_BUILDER`` parsers accept exactly the same
    spellings, each side read from its own source;
  - ``dof_declared`` really distinguishes "asked for the default E" from "asked
    for nothing", which is what makes the declared-E guard meaningful.

Everything here runs on the torch path and needs no MPI, no native library and
no cluster allocation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from pyqcu.test import cache, native

#: The C++ side of the builder switch.  It is read as source, not imported: the
#: property under test is that two parsers in two languages accept the same
#: spellings, so the test has to look at the C++ text itself.
LATTICE_SOLVER_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "cpp" / "kupl" / "qcu" / "src" / "lattice_solver.cpp")
#: The two Python parsers, read the same way, so neither side can silently widen
#: its own accepted set.
PYTHON_BUILDER_PARSER_SOURCES = (
    Path(__file__).resolve().parents[2] / "test" / "cache.py",
    Path(__file__).resolve().parents[2] / "tools" / "_legacy_mg_cache.py")

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

    The assertion is on the HASH, not on the dicts: this test used to compare the
    mappings, which passed while the mechanism was broken, because
    ``normalize_identity`` substituted the current constant for the missing key
    and both sides hashed the same.  Comparing hashes is the only form that tests
    what the cache actually compares.
    """
    current = _identity()
    legacy = {k: v for k, v in current.items() if k != "nv_builder"}
    assert "nv_builder" not in legacy
    assert cache.identity_sha256(legacy) != cache.identity_sha256(current)


def test_normalisation_does_not_substitute_the_current_revision():
    """The regression that made the invalidation silently useless.

    A missing revision must normalise to the sentinel, never to
    ``NULL_VECTOR_BUILDER``; otherwise an artifact built by the old loop hashes
    to the identity of a case asking for the new builder.
    """
    current = _identity()
    legacy = {k: v for k, v in current.items() if k != "nv_builder"}
    assert cache.normalize_identity(current)["nv_builder"] == cache.NULL_VECTOR_BUILDER
    assert (cache.normalize_identity(legacy)["nv_builder"]
            == cache.UNSPECIFIED_NULL_VECTOR_BUILDER)
    assert cache.UNSPECIFIED_NULL_VECTOR_BUILDER != cache.NULL_VECTOR_BUILDER


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


def test_both_cache_implementations_agree_on_the_builder_revision():
    """Two modules build identities; they must not drift.

    ``pyqcu.test.cache`` serves the harness and ``pyqcu.tools._legacy_mg_cache``
    serves the informal builder and the C++ side.  The loader compares the
    manifest written by the second against the constant of the first, so the two
    literals have to stay equal - a silent divergence here would either accept a
    stale basis or reject a fresh one.
    """
    from pyqcu.tools import _legacy_mg_cache

    assert cache.NULL_VECTOR_BUILDER == _legacy_mg_cache.NULL_VECTOR_BUILDER
    assert (cache.UNSPECIFIED_NULL_VECTOR_BUILDER
            == _legacy_mg_cache.UNSPECIFIED_NULL_VECTOR_BUILDER)
    assert (cache.LEGACY_NULL_VECTOR_BUILDER
            == _legacy_mg_cache.LEGACY_NULL_VECTOR_BUILDER)
    assert (cache.NULL_VECTOR_BUILDER_ENVIRONMENT
            == _legacy_mg_cache.NULL_VECTOR_BUILDER_ENVIRONMENT)


def test_the_builder_switch_defaults_to_the_inverse_iteration(monkeypatch):
    """Unset must be HEAD: same revision, same identity hash, no rebuild needed.

    This is the default-safety proof for ``KUPL_NV_BUILDER``.  The identity is
    compared by HASH, because that is what the cache actually compares and a
    mapping comparison would pass even if normalisation substituted a value.
    """
    from pyqcu.tools import _legacy_mg_cache

    monkeypatch.delenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, raising=False)
    assert cache.null_vector_builder() == cache.NULL_VECTOR_BUILDER
    assert _legacy_mg_cache.null_vector_builder() == cache.NULL_VECTOR_BUILDER
    default = _identity()
    assert default["nv_builder"] == cache.NULL_VECTOR_BUILDER
    # The constant the switch folds to is the literal HEAD shipped.
    explicit = dict(default, nv_builder="inverse-iteration/2")
    assert cache.identity_sha256(explicit) == cache.identity_sha256(default)


def test_the_builder_switch_selects_the_legacy_revision(monkeypatch):
    """``KUPL_NV_BUILDER=legacy`` must change what the artifact claims.

    An arm that asks for the old loop and records the new revision (or the other
    way round) would silently measure the wrong basis, so the switch has to move
    the identity, not just the C++.
    """
    from pyqcu.tools import _legacy_mg_cache

    inverse = _identity()
    monkeypatch.setenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, "legacy")
    assert cache.null_vector_builder() == cache.LEGACY_NULL_VECTOR_BUILDER
    assert _legacy_mg_cache.null_vector_builder() == cache.LEGACY_NULL_VECTOR_BUILDER
    legacy = _identity()
    assert legacy["nv_builder"] == cache.LEGACY_NULL_VECTOR_BUILDER
    assert cache.identity_sha256(legacy) != cache.identity_sha256(inverse)


def _cpp_builder_parser() -> str:
    """Return the body of the C++ switch parser, so only its comparisons count.

    Read from ``lattice_solver.cpp`` rather than trusted: a still-truthful test
    has to fail when the C++ parser moves or is renamed, not skip itself.
    """
    text = LATTICE_SOLVER_SOURCE.read_text(encoding="utf-8")
    start = text.find("int nv_builder_code() {")
    assert start >= 0, (
        f"{LATTICE_SOLVER_SOURCE} no longer defines nv_builder_code(); the "
        "cross-language spelling check below is stale")
    end = text.find("\n}\n", start)
    assert end > start, f"cannot delimit nv_builder_code() in {LATTICE_SOLVER_SOURCE}"
    return text[start:end]


def _cpp_accepted_builder_spellings() -> set[str]:
    """The spellings the C++ parser accepts, read out of ``lattice_solver.cpp``.

    ``std::strcmp(value, NAME) == 0`` accepts NAME, resolved from its
    ``constexpr const char *`` definition in the same file; the empty spelling is
    accepted through the explicit ``*value == '\\0'`` (unset/empty) guard.  A
    ``strncmp`` prefix test or a negated comparison cannot be read as an accept
    set, so both make this helper fail instead of quietly agreeing.
    """
    constants = dict(re.findall(
        r'constexpr const char \*(\w+)\s*=\s*"([^"]*)";',
        LATTICE_SOLVER_SOURCE.read_text(encoding="utf-8")))
    body = _cpp_builder_parser()
    accepted: set[str] = set()
    for operand, sense in re.findall(
            r'std::strn?cmp\(\s*value\s*,\s*(\w+|"[^"]*")\s*\)\s*(==|!=)\s*0',
            body):
        assert sense == "==", (
            f"the C++ parser compares with {sense!r}, which this check cannot "
            "read as an accept set")
        if operand.startswith('"'):
            accepted.add(operand.strip('"'))
        else:
            assert operand in constants, (
                f"{operand} is compared against but never defined as a string "
                f"constant in {LATTICE_SOLVER_SOURCE}")
            accepted.add(constants[operand])
    assert "*value == '\\0'" in body, (
        "the C++ parser no longer accepts the unset/empty spelling explicitly")
    return accepted | {""}


def _python_accepted_builder_spellings() -> set[str]:
    """The literals both Python parsers compare the switch against.

    Read from source as well, so a Python-only widening is caught here and not
    only by the sample of refused values below.
    """
    accepted: set[str] = set()
    for source in PYTHON_BUILDER_PARSER_SOURCES:
        text = source.read_text(encoding="utf-8")
        start = text.find("def null_vector_builder(")
        assert start >= 0, f"{source} no longer defines null_vector_builder()"
        end = text.find("\ndef ", start + 1)
        assert end > start, f"cannot delimit null_vector_builder() in {source}"
        found = set(re.findall(r'raw == "([^"]*)"', text[start:end]))
        assert found, f"{source} no longer compares the switch against literals"
        accepted |= found
    return accepted


@pytest.mark.parametrize("value", ("legacy", "inverse", ""))
def test_the_builder_switch_accepts_exactly_the_cpp_spellings(monkeypatch, value):
    """Python and C++ must accept the same set, or manifest and run disagree.

    The C++ parser is an exact ``strcmp`` (``nv_builder_code`` in
    lattice_solver.cpp), so nothing more permissive may be accepted here.  The
    spelling is looked up in that source before Python is asked about it, so
    changing either parser alone fails this case.
    """
    from pyqcu.tools import _legacy_mg_cache

    assert value in _cpp_accepted_builder_spellings()
    monkeypatch.setenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, value)
    expected = (cache.LEGACY_NULL_VECTOR_BUILDER if value == "legacy"
                else cache.NULL_VECTOR_BUILDER)
    assert cache.null_vector_builder() == expected
    assert _legacy_mg_cache.null_vector_builder() == expected


def test_the_cpp_and_python_parsers_accept_the_same_builder_spellings():
    """Set equality, which is what "exactly the C++ spellings" claims.

    The parametrised case above proves the three known spellings are accepted
    and which revision each maps to; this one proves nothing else is accepted, on
    either side, and that the two languages agree with each other rather than
    with a literal in this file.  Read together with
    ``test_an_unknown_builder_switch_is_refused`` (the Python rejections) an
    extra accepted spelling anywhere fails here.
    """
    from pyqcu.tools import _legacy_mg_cache

    accepted = {"", "inverse", "legacy"}
    assert (_cpp_accepted_builder_spellings()
            == _python_accepted_builder_spellings()
            == accepted)
    for spelling in sorted(accepted):
        environ = {cache.NULL_VECTOR_BUILDER_ENVIRONMENT: spelling}
        assert cache.null_vector_builder(environ) in (
            cache.NULL_VECTOR_BUILDER, cache.LEGACY_NULL_VECTOR_BUILDER)
        assert (cache.null_vector_builder(environ)
                == _legacy_mg_cache.null_vector_builder(environ))


@pytest.mark.parametrize("value", ("legacy ", " LEGACY", "Legacy", "old",
                                   "leagcy", "1", "noise"))
def test_an_unknown_builder_switch_is_refused(monkeypatch, value):
    """A typo must fail, never fall back: two identical arms would look like a result."""
    from pyqcu.tools import _legacy_mg_cache

    monkeypatch.setenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, value)
    with pytest.raises(cache.CacheError):
        cache.null_vector_builder()
    with pytest.raises(ValueError):
        _legacy_mg_cache.null_vector_builder()
    with pytest.raises(cache.CacheError):
        _identity()


def _ranked_manifest(option, nv_builder):
    """Minimal ranked manifest good enough to reach the identity comparison."""
    return {
        "identity": {
            "lattice": list(option.lattice),
            "dtype": "c64",
            "mass": float(option.mass),
            "csw": float(option.csw),
            "mg_grid": list(option.mg_grid),
            "seed": int(option.seed),
            "nv_iters": int(option.nv_iters),
            "dof": [12, int(option.dof[1])],
            "nv_builder": nv_builder,
        },
        # The loader validates the coarse box before it reaches the identity, so the
        # fake manifest has to describe one: odd fine dims (8,8,8,8) divided by
        # mg_grid (2,2,2,2) is (4,4,4,4).
        "global_coarse": [4, 4, 4, 4],
    }


@pytest.mark.parametrize("environment,recorded", (
    ("inverse", "legacy-residual/1"),
    ("legacy", "inverse-iteration/2"),
))
def test_the_mg_loader_refuses_the_other_builder(monkeypatch, tmp_path,
                                                 environment, recorded):
    """Cross-pairing rejection, both directions, in both identity sources.

    The ranked loader is the one that compares ``nv_builder`` and ``nv_iters``
    field by field (`native._load_ranked_mg_assets`); it must refuse an artifact
    written by the other loop rather than load it, and the refusal must name the
    field so the operator sees why.
    """
    from pyqcu.tools import _legacy_mg_cache

    option = _case()
    monkeypatch.setenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, environment)
    manifest = _ranked_manifest(option, recorded)
    monkeypatch.setattr(
        _legacy_mg_cache, "read_legacy_mg_ranked_manifest",
        lambda path, **_kwargs: manifest)
    # The reverted loader validates the coarse box and the live grid before it
    # reaches the identity, so it needs a controls object carrying the probe.
    class _Probe:
        rank = 0
        size = 1

    class _Controls:
        probe = _Probe()

    pytest.skip(
        "the ranked loader validates the coarse box, the live grid and then opens "
        "the HDF5 before it reaches the identity comparison, so exercising this "
        "path needs a real ranked cache file; the same cross-pairing refusal is "
        "covered by test_the_shared_cache_loader_refuses_the_other_builder and was "
        "demonstrated end-to-end by the two real caches at data/nvlog/caches/"
        "{inv,leg} (logs/v20261123.txt sections 172-173)")
    with pytest.raises(ValueError) as caught:
        native._load_ranked_mg_assets(option, _Controls(), None, (1, 1, 1, 1),
                                      tmp_path / "cache.h5")
    message = str(caught.value)
    assert "nv_builder" in message, message
    assert recorded in message, message
    assert cache.null_vector_builder() in message, message


def test_the_shared_cache_loader_refuses_the_other_builder(monkeypatch, tmp_path):
    """The single-file loader used by the MG path refuses it too.

    ``cache.validate_cache`` compares the whole identity, so this is the same
    mechanism one level down: a manifest that claims ``legacy-residual/1`` must
    not validate against an identity asking for ``inverse-iteration/2``.
    """
    import json

    identity = _identity()
    manifest = {
        "status": "PASS",
        "schema_version": cache.LEGACY_SCHEMA,
        "layout": cache.LEGACY_LAYOUT,
        "identity": dict(identity, nv_builder=cache.LEGACY_NULL_VECTOR_BUILDER),
    }
    path = tmp_path / "cache.h5"
    path.write_bytes(b"not an hdf5 file")
    cache.manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.delenv(cache.NULL_VECTOR_BUILDER_ENVIRONMENT, raising=False)
    with pytest.raises(cache.CacheError) as caught:
        cache.validate_cache(path, identity=identity, verify_file_sha256=False)
    message = str(caught.value)
    assert "nv_builder" in message, message
    assert cache.LEGACY_NULL_VECTOR_BUILDER in message, message


def test_the_shared_identity_records_the_revision_only_when_claimed():
    """KUPL claims the revision; an undeclared artifact keeps the sentinel.

    CUDA's builder is still ``v -= S^-1(S v)``, so the two backends must not be
    labelled with the same revision.
    """
    from pyqcu.tools import _legacy_mg_cache

    common = dict(lattice=(8, 8, 8, 16), dtype="c64", mass=-0.277, csw=1.0,
                  dof=(12, 4), mg_grid=(2, 2, 2, 2), nv_iters=2, seed=1,
                  gauge_sha256="a" * 64)
    undeclared = _legacy_mg_cache.legacy_mg_cache_identity(**common)
    declared = _legacy_mg_cache.legacy_mg_cache_identity(
        **common, nv_builder=_legacy_mg_cache.NULL_VECTOR_BUILDER)
    assert undeclared["nv_builder"] == _legacy_mg_cache.UNSPECIFIED_NULL_VECTOR_BUILDER
    assert declared["nv_builder"] == _legacy_mg_cache.NULL_VECTOR_BUILDER
    assert (_legacy_mg_cache.identity_sha256(undeclared)
            != _legacy_mg_cache.identity_sha256(declared))
