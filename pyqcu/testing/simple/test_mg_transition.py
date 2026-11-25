"""Tests for the Python Legacy-MG transition builder (``mg_transition``).

The builder exists because a 3-level Legacy-MG chain has no coarse gauge: level
L's operator is the Galerkin projection of level L-1's 33-tensor stencil, so the
C++ gauge + Clover builders cannot express it.  These tests pin the geometry the
cache format requires and run the whole build once against a synthetic parent
operator, whose Galerkin identity is checkable in exact arithmetic.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    is_legacy_mg_ranked_cache,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    save_legacy_mg_cache,
    validate_legacy_mg_cache,
)
from pyqcu.testing.simple.mg_transition import (
    GALERKIN_TOLERANCE,
    GRAM_TOLERANCE,
    _parse_args,
    build_transition_cache,
    stencil_support_offsets,
    transition_geometry,
)


def test_transition_2_geometry_matches_the_canonical_shape_helper() -> None:
    """The one shape helper must describe transition 2 exactly.

    ``legacy_mg_cache_shapes`` hardcodes the 12-component fine dof, so it only
    describes a higher transition because ``dof[0]`` (the parent basis size) *is*
    that fine dof and ``dof[1]`` is the child level's E.
    """
    geometry = transition_geometry((16, 16, 16, 32), (2, 2, 2, 2), (12, 12))
    helper = legacy_mg_cache_shapes((8, 8, 8, 16), (2, 2, 2, 2), (12, 12))

    assert geometry["parent_fine_odd"] == (16, 16, 16, 16)
    assert geometry["parent_coarse"] == (8, 8, 8, 8)
    # The child's parent view: the parent's coarse box at twice the odd time.
    assert geometry["child_lattice"] == (8, 8, 8, 16)
    assert geometry["child_fine_odd"] == (8, 8, 8, 8)
    assert geometry["child_coarse"] == (4, 4, 4, 4)
    assert geometry["shapes"] == helper
    assert geometry["shapes"] == {
        "lonv": (12, 12, 4, 2, 4, 2, 4, 2, 4, 2),
        "hnn": (2, 4, 12, 12, 4, 4, 4, 4),
        "hdg": (2, 2, 6, 12, 12, 4, 4, 4, 4),
        "sit": (12, 12, 4, 4, 4, 4),
    }
    # The same helper on the *parent's own* lattice describes transition 1, so
    # the two transitions are the same call with a different lattice: this is
    # why one cache root per transition is enough to describe the whole chain.
    assert legacy_mg_cache_shapes((16, 16, 16, 32), (2, 2, 2, 2), (12, 12)) == {
        "lonv": (12, 12, 8, 2, 8, 2, 8, 2, 8, 2),
        "hnn": (2, 4, 12, 12, 8, 8, 8, 8),
        "hdg": (2, 2, 6, 12, 12, 8, 8, 8, 8),
        "sit": (12, 12, 8, 8, 8, 8),
    }


def test_transition_geometry_rejects_a_non_canonical_parent_dof() -> None:
    try:
        transition_geometry((16, 16, 16, 32), (2, 2, 2, 2), (4, 12))
    except ValueError as error:
        assert "E_parent == 12" in str(error)
    else:  # pragma: no cover - the raise above is the assertion
        raise AssertionError("E_parent != 12 must be refused")


def test_stencil_support_offsets_counts_the_33_point_star() -> None:
    shape = (12, 12, 3, 3, 3, 3)
    sit = np.ones(shape, dtype=np.complex64)
    hop_nn = np.ones((2, 4, *shape), dtype=np.complex64)
    hop_diag = np.ones((2, 2, 6, *shape), dtype=np.complex64)

    offsets = stencil_support_offsets(sit, hop_nn, hop_diag)
    assert len(offsets) == 33
    assert len(set(offsets)) == 33
    assert (0, 0, 0, 0) in offsets
    # 8 nearest neighbours + 24 diagonal couplings.
    assert sum(1 for offset in offsets
               if sum(abs(value) for value in offset) == 1) == 8
    assert sum(1 for offset in offsets
               if sum(abs(value) for value in offset) == 2) == 24

    # A stencil that only kept the onsite term must not report 33.
    empty = np.zeros((2, 4, *shape), dtype=np.complex64)
    empty_diag = np.zeros((2, 2, 6, *shape), dtype=np.complex64)
    assert stencil_support_offsets(sit, empty, empty_diag) == [(0, 0, 0, 0)]


def test_report_path_may_not_be_the_cache_manifest(tmp_path: Path) -> None:
    output = tmp_path / "lv2.h5"
    try:
        _parse_args([
            "--parent", str(tmp_path / "lv1.h5"),
            "--output", str(output),
            "--parent-lattice", "16", "16", "16", "32",
            "--report", str(output.with_suffix(".json")),
        ])
    except SystemExit as error:  # argparse error()
        assert int(error.code) == 2
    else:  # pragma: no cover
        raise AssertionError("a report must never overwrite a cache manifest")


def _synthetic_parent(root: Path, dtype: np.dtype) -> dict:
    """Write a well-conditioned synthetic 33-tensor parent transition.

    The parent *is* transition 1 of a chain: its identity carries
    ``dof=(12, E_parent)`` with ``E_parent=12``, so its own 33-tensor operator
    acts on a 12-component coarse field.  The transition built from it then
    moves those 12 parent components to ``E_child=2``.
    """
    lattice = (8, 8, 8, 16)
    block = (2, 2, 2, 2)
    dof = (12, 12)
    shapes = legacy_mg_cache_shapes(lattice, block, dof)
    coarse = shapes["sit"][2:]
    rng = np.random.default_rng(11)
    eye = np.eye(12, dtype=dtype)
    sit = (eye[(slice(None), slice(None)) + (None,) * 4]
           + 0.10 * (rng.standard_normal((12, 12, *coarse))
                     + 1j * rng.standard_normal((12, 12, *coarse)))
           ).astype(dtype)
    hop_nn = (0.05 * (rng.standard_normal((2, 4, 12, 12, *coarse))
                      + 1j * rng.standard_normal((2, 4, 12, 12, *coarse)))
              ).astype(dtype)
    hop_diag = (0.02 * (rng.standard_normal((2, 2, 6, 12, 12, *coarse))
                        + 1j * rng.standard_normal((2, 2, 6, 12, 12, *coarse)))
                ).astype(dtype)
    identity = legacy_mg_cache_identity(
        gauge_canonical_sha256=hashlib.sha256(b"synthetic-parent").hexdigest(),
        lattice=lattice, dtype=dtype, mass=0.25, csw=1.0, dof=dof,
        mg_grid=block, nv_iters=2, seed=42,
        nv_builder="inverse-iteration/2")
    save_legacy_mg_cache(
        root, {"lonv": np.zeros(shapes["lonv"], dtype=dtype),
               "hnn": hop_nn, "hdg": hop_diag, "sit": sit},
        identity=identity, setup_method="test-synthetic",
        setup_backend="python-test")
    return {"identity": identity, "lattice": lattice, "block": block,
            "dof": dof}


def test_build_transition_end_to_end_on_a_synthetic_parent(tmp_path: Path) -> None:
    dtype = np.dtype(np.complex64)
    parent_root = tmp_path / "lv1_synthetic.h5"
    parent = _synthetic_parent(parent_root, dtype)
    output = tmp_path / "lv2.h5"

    report = build_transition_cache(
        parent_root, output, parent["lattice"], parent["block"],
        (12, 2), dtype="c64", device="cpu", seed=42, samples=1,
        verbose=False)

    assert report["galerkin_relative_error"] <= GALERKIN_TOLERANCE
    assert report["lonv_gram_error"] <= GRAM_TOLERANCE
    # The probe construction is exact up to the working precision: at c64 the
    # achieved error is ~4e-7, far below the 2e-5 gate (the same level as the
    # C++ builder's own patch Galerkin error, 5.8e-7).
    assert report["galerkin_relative_error"] < 1.0e-5
    assert report["inner_bicgstab"]["converged"] is True
    assert report["stencil_offset_count"] == 33
    # The basis is the parent field: E_parent components per coarse site.
    assert report["lonv_shape"] == [2, 12, 2, 2, 2, 2, 2, 2, 2, 2]
    assert report["identity"]["lattice"] == [4, 4, 4, 8]
    assert report["identity"]["dof"] == [12, 2]
    assert report["identity"]["nv_builder"] == "inverse-iteration/2"

    assert is_legacy_mg_ranked_cache(output)
    manifest = json.loads(
        output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["global_shapes"]["lonv"] == [2, 12, 2, 2, 2, 2, 2, 2, 2, 2]
    assert manifest["global_shapes"]["hnn"] == [2, 4, 2, 2, 2, 2, 2, 2]
    assert manifest["setup_validation"]["stencil_offset_count"] == 33
    validate_legacy_mg_cache(
        output, report["identity"], verify_file_sha256=True,
        verify_dataset_sha256=True, ranks=(0,))
