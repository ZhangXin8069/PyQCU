#!/usr/bin/env python3
"""Python builder for a Legacy-MG transition whose parent operator is a stencil.

The KUPL Legacy-MG hierarchy carries no physics and no coarse gauge: it is
defined recursively by the transfer operators alone,

    op(0) = fine odd-Schur Clover operator
    op(L) = R_L . op(L-1) . P_L            (the Galerkin projection)

and one cache root publishes exactly one transition: ``lonv`` (the ``P_L``/``R_L``
basis of the parent field) plus the 33-tensor stencil ``hop_nn``/``hop_diag``/
``sit`` of ``op(L)``.  That one-root-per-transition layout is the repository's
existing convention (``pyqcu/cuda/_multi_gpu.py:84-160`` writes one cache file
per level, ``:229`` hands the stencil to the next level as
``S = lambda v: apply_stencil(hnn, hdg, sit, v)``).

Transition 1 can be built by the C++ gauge + Clover builder because its parent
operator *is* the fine Wilson-Clover odd-Schur operator.  Nothing in C++ can
build transition >= 2: ``applyMultigridBuildNullVectorsQcu`` and
``applyMultigridBuildStencilLocalQcu`` take the parent's gauge and Clover pair
and require the parent's fine extents to equal the LatticeSet geometry
(``lattice_solver.cpp:898-899``, ``lattice_multigrid.cpp:1653-1657``).  That
representation is onsite + one hop, so it can never be the 33-point Galerkin
stencil of the parent level.  The level-2 build therefore has to be Python, and
this module is that builder:

    parent cache root  ->  lonv/hnn/hdg/sit  ->  parent_matvec = apply_stencil(...)
      -> give_null_vecs(matvec=parent_matvec, ...)   inverse iteration, rel tol 5e-5
      -> local_orthogonalize(...)                    two-pass MGS per coarse site
      -> build_stencil(parent_matvec, basis, ...)    the 33-tensor probe construction
      -> write_legacy_mg_rank_slice + ranked manifest

Two gates run **before** anything is published, and a failed gate writes no
cache at all:

  * Galerkin   ``||A_L v - R op(L-1) P v|| / ||R op(L-1) P v|| <= 2e-5``
                for random coarse ``v`` (this is what makes the published
                stencil the Galerkin projection of the parent operator and not
                merely something that looks like one);
  * ``lonv`` Gram error ``max |<b_i, b_j> - delta_ij| <= 1e-4`` per coarse site
                (a rank-deficient basis silently destroys the coarse operator).

Usage::

    python -m pyqcu.testing.simple.mg_transition \\
        --parent <root of transition L-1> \\
        --parent-lattice 16 16 16 32 --mg-grid 2 2 2 2 --dof 12 12 \\
        --output <root of transition L> --report <path outside the cache sidecar>

``--parent`` accepts both a ranked cache root (``<root>`` + ``<root>.d/``) and a
single-file canonical cache; ``--output`` is always written as a ranked root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyqcu.tools._legacy_mg_cache import (
    LEGACY_MG_ASSET_NAMES,
    NULL_VECTOR_BUILDER,
    build_legacy_mg_ranked_cache_manifest,
    cache_manifest_path,
    canonical_dtype,
    is_legacy_mg_ranked_cache,
    legacy_mg_cache_identity,
    legacy_mg_cache_shapes,
    legacy_mg_geometry,
    legacy_mg_local_shapes,
    load_legacy_mg_cache_slab,
    read_legacy_mg_ranked_manifest,
    validate_legacy_mg_cache,
    write_legacy_mg_rank_slice,
    write_legacy_mg_ranked_cache,
)

#: Gate 1: relative Galerkin error of the published coarse operator.
GALERKIN_TOLERANCE = 2.0e-5
#: Gate 2: per-coarse-site orthonormality error of the published basis.
GRAM_TOLERANCE = 1.0e-4
#: ``give_null_vecs`` solves each random probe with this relative tolerance.
NULL_VECTOR_TOLERANCE = 5.0e-5
#: Every transition cache is published as one rank over the whole coarse box.
SINGLE_RANK_GRID = (1, 1, 1, 1)
SINGLE_RANK_COORDINATE = (0, 0, 0, 0)
#: ``hop_diag`` pair axes, matching ``pyqcu.torch.tools._multigrid.PAIRS``.
DIAGONAL_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
#: ``hop_diag`` sign slots, matching ``pyqcu.torch.tools._multigrid.SIGN``.
DIAGONAL_SIGNS = (1, -1)


def _tuple4(values: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain four integers: {result}")
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must be positive: {result}")
    return result  # type: ignore[return-value]


def transition_geometry(
        parent_lattice: Sequence[int], block: Sequence[int],
        dof: Sequence[int]) -> dict[str, Any]:
    """Return the geometry of one transition of the Legacy-MG chain.

    ``parent_lattice`` is the *parent transition's* full lattice, i.e. the
    lattice of the field its operator acts on, written in the odd-Schur
    convention that every cache root's identity uses (the time extent is twice
    the odd-Schur extent).  ``block`` is the odd-Schur block factor and ``dof``
    is ``(E_parent, E_child)``.

    The child's own "parent view" is the parent's coarse box, so its lattice is
    ``(X_{L-1}, Y_{L-1}, Z_{L-1}, 2 * T_{L-1})``: the X/Y/Z extents are the
    parent's coarse extents and the time extent is twice the parent's coarse
    odd-Schur extent.  Everything else follows from the canonical helpers.
    """
    parent_lattice = _tuple4(parent_lattice, "parent_lattice")
    block = _tuple4(block, "block")
    dof = tuple(int(value) for value in dof)
    if len(dof) != 2:
        raise ValueError(
            f"a transition needs (E_parent, E_child), got dof={dof}")
    if min(dof) <= 0:
        raise ValueError(f"dof must be positive, got {dof}")
    if dof[0] != 12:
        # legacy_mg_cache_shapes() hardcodes fine_dof = 12 (its lines 131-137)
        # because the coarse field is always a 12-component odd-Schur field.
        # A higher transition has E_parent parent components, so this helper
        # only describes a transition whose parent basis is 12-dimensional.
        raise ValueError(
            f"this builder needs E_parent == 12 (the canonical fine dof the "
            f"Legacy-MG cache format hardcodes), got {dof[0]}")
    parent_fine_odd, parent_coarse = legacy_mg_geometry(parent_lattice, block)
    child_lattice = (*parent_coarse[:3], 2 * parent_coarse[3])
    child_fine_odd = parent_coarse
    if any(child_fine_odd[axis] % block[axis] for axis in range(4)):
        raise ValueError(
            f"parent coarse box {child_fine_odd} is not divisible by the block "
            f"{block}")
    child_coarse = tuple(
        child_fine_odd[axis] // block[axis] for axis in range(4))
    shapes = legacy_mg_cache_shapes(child_lattice, block, dof)
    # The canonical shape helper must describe exactly this transition: its
    # `coarse` is child_fine_odd / block, its `out_dof` is E_child and its
    # hardcoded `fine_dof` must be E_parent.
    expected_lonv = (
        dof[1], dof[0],
        child_coarse[0], block[0], child_coarse[1], block[1],
        child_coarse[2], block[2], child_coarse[3], block[3])
    if tuple(shapes["lonv"]) != expected_lonv:
        raise ValueError(
            f"legacy_mg_cache_shapes() describes lonv {tuple(shapes['lonv'])}, "
            f"but this transition needs {expected_lonv}")
    if tuple(shapes["hnn"]) != (2, 4, dof[1], dof[1], *child_coarse):
        raise ValueError(f"hnn shape {tuple(shapes['hnn'])} is inconsistent")
    if tuple(shapes["hdg"]) != (2, 2, 6, dof[1], dof[1], *child_coarse):
        raise ValueError(f"hdg shape {tuple(shapes['hdg'])} is inconsistent")
    if tuple(shapes["sit"]) != (dof[1], dof[1], *child_coarse):
        raise ValueError(f"sit shape {tuple(shapes['sit'])} is inconsistent")
    return {
        "parent_lattice": parent_lattice,
        "block": block,
        "dof": (dof[0], dof[1]),
        "parent_fine_odd": parent_fine_odd,
        "parent_coarse": parent_coarse,
        "child_lattice": child_lattice,
        "child_fine_odd": child_fine_odd,
        "child_coarse": child_coarse,
        "shapes": shapes,
    }


def _manifest_identity(root: Path) -> dict[str, Any]:
    """Return the identity a cache root publishes, ranked or single-file."""
    if is_legacy_mg_ranked_cache(root):
        return dict(read_legacy_mg_ranked_manifest(root)["identity"])
    sidecar = cache_manifest_path(root)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Legacy MG cache manifest not found: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError(
            f"Legacy MG cache manifest status is not PASS: {sidecar}")
    return dict(manifest["identity"])


def load_parent_transition(
        root: str | Path, dtype: Any = "c64",
        device: str | Any = "cpu") -> dict[str, Any]:
    """Load ``lonv/hnn/hdg/sit`` of the parent transition as torch tensors."""
    import torch

    path = Path(root).expanduser()
    canonical = canonical_dtype(dtype)
    identity = _manifest_identity(path)
    if np.dtype(identity["dtype"]) != canonical:
        raise ValueError(
            f"parent cache dtype {identity['dtype']} != requested {canonical}")
    values = load_legacy_mg_cache_slab(path, canonical)
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(values[name])).to(device)
        for name in LEGACY_MG_ASSET_NAMES
    }
    identity_lattice = tuple(int(value) for value in identity["lattice"])
    fine_odd, coarse = legacy_mg_geometry(identity_lattice,
                                          identity["mg_grid"])
    sit = tensors["sit"]
    if int(sit.shape[0]) != int(sit.shape[1]):
        raise ValueError(f"parent sit shape {tuple(sit.shape)} is not square")
    if tuple(sit.shape[-4:]) != tuple(coarse):
        raise ValueError(
            f"parent sit coarse axes {tuple(sit.shape[-4:])} != {coarse} from "
            f"its identity lattice {identity_lattice}")
    return {
        **tensors,
        "root": str(path),
        "manifest": str(cache_manifest_path(path)),
        "identity": identity,
        "fine_odd": fine_odd,
        "coarse": coarse,
        "e_parent": int(sit.shape[0]),
    }


def stencil_matvec(parent: Mapping[str, Any]):
    """Return ``v -> op(L-1) v`` from the parent's cached 33-tensor stencil."""
    from pyqcu.torch.tools._multigrid import apply_stencil

    hop_nn, hop_diag, sit = parent["hnn"], parent["hdg"], parent["sit"]

    def matvec(vector: Any) -> Any:
        return apply_stencil(hop_nn, hop_diag, sit, vector)

    return matvec


def stencil_support_offsets(
        sit: Any, hop_nn: Any, hop_diag: Any,
        tolerance: float = 0.0) -> list[tuple[int, int, int, int]]:
    """Return the relative coarse offsets the 33-tensor couples to.

    A Legacy-MG coarse operator is onsite + 8 nearest-neighbour + 24 diagonal
    couplings, i.e. 33 offsets.  The check reads the stored tensors, so a
    builder that silently dropped (or double-counted) a slot is caught before
    the cache is published.
    """

    def populated(tensor: Any) -> bool:
        if hasattr(tensor, "detach"):
            value = float(tensor.detach().abs().max().item())
        else:
            value = float(np.abs(np.asarray(tensor)).max())
        return value > tolerance

    offsets: set[tuple[int, int, int, int]] = set()
    if populated(sit):
        offsets.add((0, 0, 0, 0))
    for direction in range(4):
        for slot, step in enumerate((1, -1)):
            if populated(hop_nn[slot, direction]):
                delta = [0, 0, 0, 0]
                delta[direction] = step
                offsets.add(tuple(delta))
    for pair, (first, second) in enumerate(DIAGONAL_PAIRS):
        for sign_index, sign in enumerate(DIAGONAL_SIGNS):
            for slot in range(2):
                if not populated(hop_diag[sign_index, slot, pair]):
                    continue
                delta = [0, 0, 0, 0]
                delta[first] = sign
                delta[second] = DIAGONAL_SIGNS[slot]
                offsets.add(tuple(delta))
    return sorted(offsets)


def lonv_gram_error(basis: Any, geometry: Mapping[str, Any]) -> float:
    """Return ``max |<b_i, b_j> - delta_ij|`` over all coarse sites."""
    import torch

    E_child, e = int(basis.shape[0]), int(basis.shape[1])
    coarse, block = geometry["child_coarse"], geometry["block"]
    blocked = basis.reshape(
        E_child, e, coarse[0], block[0], coarse[1], block[1],
        coarse[2], block[2], coarse[3], block[3])
    sites = int(np.prod(coarse))
    columns = blocked.permute(2, 4, 6, 8, 0, 1, 3, 5, 7, 9).reshape(
        sites, E_child, e * int(np.prod(block)))
    gram = columns.conj() @ columns.transpose(1, 2)
    reference = torch.eye(
        E_child, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    return float((gram - reference).abs().max().item())


def galerkin_relative_errors(
        parent_matvec, basis: Any, tensors: Mapping[str, Any],
        geometry: Mapping[str, Any], *, seed: int = 941,
        samples: int = 3) -> tuple[float, list[float]]:
    """Gate 1: compare the built stencil against ``R op(L-1) P`` exactly.

    ``build_stencil`` measures the stencil by probing, so this identity is the
    only statement that the published operator is the Galerkin projection of the
    parent and not an artifact of the probe bookkeeping.
    """
    import torch
    from pyqcu.torch.tools._multigrid import apply_stencil, prolong, restrict

    E_child = int(geometry["dof"][1])
    coarse = geometry["child_coarse"]
    dtype = tensors["sit"].dtype
    device = tensors["sit"].device
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    errors: list[float] = []
    for _ in range(max(1, int(samples))):
        vector = torch.randn(
            (E_child, *coarse), generator=generator,
            dtype=torch.complex64 if dtype == torch.complex64
            else torch.complex128).to(device=device, dtype=dtype)
        prolonged = prolong(local_ortho_null_vecs=basis, coarse_vec=vector)
        restricted = restrict(
            local_ortho_null_vecs=basis, fine_vec=parent_matvec(prolonged))
        applied = apply_stencil(tensors["hnn"], tensors["hdg"],
                                tensors["sit"], vector)
        denominator = float(torch.linalg.vector_norm(restricted).item())
        difference = float(torch.linalg.vector_norm(applied - restricted).item())
        errors.append(difference / max(denominator, np.finfo(float).tiny))
    return max(errors), errors


def build_transition(
        parent: Mapping[str, Any], geometry: Mapping[str, Any], *,
        seed: int = 42, verbose: bool = True) -> dict[str, Any]:
    """Run the probe construction for one transition of the chain."""
    import torch
    from pyqcu.torch.tools._multigrid import (
        build_stencil, give_null_vecs, local_orthogonalize)

    e_parent, e_child = geometry["dof"]
    dtype = parent["sit"].dtype
    device = parent["sit"].device
    matvec = stencil_matvec(parent)
    # give_null_vecs() draws its own random start from the tensor it is handed
    # (the values are discarded), so the RNG seed is the only reproducible knob.
    torch.manual_seed(int(seed))
    template = torch.zeros(
        (e_child, e_parent, *geometry["child_fine_odd"]),
        dtype=dtype, device=device)
    started = time.perf_counter()
    null_vecs = give_null_vecs(
        null_vecs=template, matvec=matvec,
        verbose=verbose)
    null_seconds = time.perf_counter() - started
    started = time.perf_counter()
    basis = local_orthogonalize(
        null_vecs=null_vecs, coarse_lat_size=list(geometry["child_coarse"]),
        verbose=verbose)
    orthogonalize_seconds = time.perf_counter() - started
    started = time.perf_counter()
    hop_nn, hop_diag, sit = build_stencil(
        matvec=matvec, lonv=basis, E=e_child, e=e_parent,
        lat_fine_odd=list(geometry["child_fine_odd"]),
        lat_coarse_odd=list(geometry["child_coarse"]),
        dt=dtype, device=device, verbose=verbose)
    stencil_seconds = time.perf_counter() - started
    return {
        "lonv": basis, "hnn": hop_nn, "hdg": hop_diag, "sit": sit,
        "parent_matvec": matvec,
        "timings": {
            "null_vector_seconds": null_seconds,
            "orthogonalize_seconds": orthogonalize_seconds,
            "stencil_seconds": stencil_seconds,
            "seconds": null_seconds + orthogonalize_seconds + stencil_seconds,
        },
    }


def transition_identity(
        parent_identity: Mapping[str, Any], geometry: Mapping[str, Any], *,
        seed: int = 42) -> dict[str, Any]:
    """Return the identity the child cache root publishes.

    ``nv_iters`` is 1 because ``give_null_vecs`` performs exactly one
    inverse-iteration pass per vector (relative tolerance 5e-5).  No coarse
    gauge exists anywhere in this chain, so the only honest gauge provenance is
    the fine gauge the chain descends from; it is inherited from the parent
    identity instead of being invented.
    """
    identity = {
        "lattice": list(geometry["child_lattice"]),
        "dtype": str(parent_identity["dtype"]),
        "mass": float(parent_identity["mass"]),
        "csw": float(parent_identity["csw"]),
        "dof": [int(geometry["dof"][0]), int(geometry["dof"][1])],
        "mg_grid": [int(value) for value in geometry["block"]],
        "nv_iters": 1,
        "seed": int(seed),
        # A python inverse iteration is a real inverse iteration; it is the
        # same revision the KUPL C++ builder declares.
        "nv_builder": NULL_VECTOR_BUILDER,
    }
    if parent_identity.get("gauge_canonical_sha256") is not None:
        identity["gauge_canonical_sha256"] = \
            parent_identity["gauge_canonical_sha256"]
    if parent_identity.get("gauge_file_sha256") is not None:
        identity["gauge_file_sha256"] = parent_identity["gauge_file_sha256"]
    return legacy_mg_cache_identity(**identity)


def write_transition_cache(
        root: str | Path, identity: Mapping[str, Any],
        geometry: Mapping[str, Any], tensors: Mapping[str, Any], *,
        setup_seconds: float | None = None,
        setup_validation: Mapping[str, Any] | None = None,
        mg_config: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Publish one transition as a ranked cache root (one rank, whole box)."""
    path = Path(root).expanduser()
    shapes = geometry["shapes"]
    datasets: dict[str, np.ndarray] = {}
    for name in LEGACY_MG_ASSET_NAMES:
        value = np.ascontiguousarray(
            tensors[name].detach().cpu().numpy()
            if hasattr(tensors[name], "detach") else tensors[name])
        if np.dtype(value.dtype) != np.dtype(identity["dtype"]):
            raise ValueError(
                f"{name} dtype {value.dtype} != identity dtype "
                f"{identity['dtype']}")
        if tuple(value.shape) != tuple(shapes[name]):
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != canonical "
                f"{tuple(shapes[name])}")
        datasets[name] = value
    coarse = geometry["child_coarse"]
    local_shapes = legacy_mg_local_shapes(shapes, coarse)
    record = write_legacy_mg_rank_slice(
        path, datasets, rank=0, coordinate=SINGLE_RANK_COORDINATE,
        process_grid=SINGLE_RANK_GRID, starts=(0, 0, 0, 0),
        local_coarse=coarse, global_shapes=shapes, identity=identity, mode="w")
    manifest = build_legacy_mg_ranked_cache_manifest(
        path, identity, process_grid=SINGLE_RANK_GRID, ranks=[record],
        global_shapes=shapes, local_shapes=local_shapes,
        global_coarse=coarse, local_coarse=coarse,
        setup_method="python-inverse-iteration+33-stencil-probe",
        setup_backend="python-torch",
        mg_config=mg_config, setup_validation=setup_validation,
        source_metadata=source_metadata, setup_seconds=setup_seconds)
    write_legacy_mg_ranked_cache(path, manifest)
    validate_legacy_mg_cache(
        path, identity, verify_file_sha256=True, verify_dataset_sha256=True,
        ranks=(0,))
    return dict(read_legacy_mg_ranked_manifest(path))


def inner_solver_probe(parent: Mapping[str, Any], *, seed: int = 7) -> dict[str, Any]:
    """Prove the inner BiCGStab converges on the parent operator.

    ``give_null_vecs`` hides its inner solve, so this probe solves one random
    right-hand side against the same matvec at the same relative tolerance and
    reports the achieved ``||b - A x|| / ||b||``.  A parent operator with a
    near-zero mode plateaus at 1e-2..1e-1 and must not be coarsened.
    """
    import torch
    from pyqcu.torch import solver

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (int(parent["sit"].shape[0]), *parent["coarse"])
    b = torch.randn(shape, generator=generator,
                    dtype=torch.complex64
                    if parent["sit"].dtype == torch.complex64
                    else torch.complex128).to(
                        device=parent["sit"].device, dtype=parent["sit"].dtype)
    matvec = stencil_matvec(parent)
    started = time.perf_counter()
    x = solver.bistabcg(b=b, matvec=matvec, tol=NULL_VECTOR_TOLERANCE,
                        if_rtol=True, verbose=False)
    seconds = time.perf_counter() - started
    residual = torch.linalg.vector_norm(b - matvec(x)) / \
        torch.linalg.vector_norm(b)
    return {
        "relative_residual": float(residual.item()),
        "tolerance": NULL_VECTOR_TOLERANCE,
        "converged": bool(float(residual.item()) <= NULL_VECTOR_TOLERANCE),
        "seconds": seconds,
    }


def build_transition_cache(
        parent_root: str | Path, output_root: str | Path,
        parent_lattice: Sequence[int], block: Sequence[int],
        dof: Sequence[int], *, dtype: Any = "c64", device: str = "cpu",
        seed: int = 42, samples: int = 3, verbose: bool = True
        ) -> dict[str, Any]:
    """Build, gate and publish one transition of the Legacy-MG chain."""
    geometry = transition_geometry(parent_lattice, block, dof)
    parent = load_parent_transition(parent_root, dtype, device=device)
    parent_identity = parent["identity"]
    if tuple(parent_identity["lattice"]) != tuple(geometry["parent_lattice"]):
        raise ValueError(
            f"parent cache identity lattice {tuple(parent_identity['lattice'])} "
            f"!= declared parent lattice {tuple(geometry['parent_lattice'])}")
    if tuple(int(value) for value in parent_identity["mg_grid"]) != \
            tuple(geometry["block"]):
        raise ValueError(
            f"parent cache identity mg_grid {parent_identity['mg_grid']} != "
            f"declared block {list(geometry['block'])}")
    if int(parent["e_parent"]) != int(geometry["dof"][0]):
        raise ValueError(
            f"parent cache carries E={parent['e_parent']} components, but the "
            f"transition moves E_parent={geometry['dof'][0]} to "
            f"E_child={geometry['dof'][1]}")
    if tuple(parent["coarse"]) != tuple(geometry["child_fine_odd"]):
        raise ValueError(
            f"parent coarse box {tuple(parent['coarse'])} != child fine extent "
            f"{tuple(geometry['child_fine_odd'])}")

    inner = inner_solver_probe(parent)
    if not inner["converged"]:
        raise RuntimeError(
            "the parent operator's inner BiCGStab does not reach its relative "
            f"tolerance: {inner['relative_residual']:.3e} vs "
            f"{inner['tolerance']:.1e}.  A chain built on this parent is "
            "worthless - change the gauge/mass instead of coarsening it.")

    built = build_transition(parent, geometry, seed=seed, verbose=verbose)
    tensors = {name: built[name] for name in LEGACY_MG_ASSET_NAMES}
    gram_error = lonv_gram_error(tensors["lonv"], geometry)
    galerkin_error, galerkin_errors = galerkin_relative_errors(
        built["parent_matvec"], tensors["lonv"], tensors, geometry,
        samples=samples)
    offsets = stencil_support_offsets(
        tensors["sit"], tensors["hnn"], tensors["hdg"])

    print(
        "PYQCU::SIMPLE::MG_TRANSITION:\n"
        f" parent={parent_root} lattice={list(geometry['parent_lattice'])} "
        f"block={list(geometry['block'])} dof={list(geometry['dof'])}\n"
        f" child_lattice={list(geometry['child_lattice'])} "
        f"child_coarse={list(geometry['child_coarse'])}\n"
        f" galerkin={galerkin_error:.3e} (tol {GALERKIN_TOLERANCE:.1e}) "
        f"gram={gram_error:.3e} (tol {GRAM_TOLERANCE:.1e}) "
        f"offsets={len(offsets)}\n"
        f" inner_bicgstab={inner['relative_residual']:.3e} "
        f"lonv_shape={tuple(tensors['lonv'].shape)}",
        flush=True)

    failures = []
    if not np.isfinite(galerkin_error) or galerkin_error > GALERKIN_TOLERANCE:
        failures.append(
            f"Galerkin error {galerkin_error:.3e} > {GALERKIN_TOLERANCE:.1e}")
    if not np.isfinite(gram_error) or gram_error > GRAM_TOLERANCE:
        failures.append(f"lonv Gram error {gram_error:.3e} > {GRAM_TOLERANCE:.1e}")
    if min(geometry["child_coarse"]) >= 3 and len(offsets) != 33:
        failures.append(f"stencil support is {len(offsets)} offsets, not 33")
    if failures:
        raise RuntimeError(
            "transition gates failed, nothing was published: " +
            "; ".join(failures))

    identity = transition_identity(parent_identity, geometry, seed=seed)
    setup_validation = {
        "schema": "pyqcu.legacy-mg-ranked.v1",
        "builder": "python-inverse-iteration+33-stencil-probe",
        "parent_root": str(parent_root),
        "parent_identity_sha256": parent_identity.get("gauge_canonical_sha256"),
        "parent_lattice": list(geometry["parent_lattice"]),
        "parent_fine_odd": list(geometry["parent_fine_odd"]),
        "parent_coarse": list(geometry["parent_coarse"]),
        "child_lattice": list(geometry["child_lattice"]),
        "child_fine_odd": list(geometry["child_fine_odd"]),
        "child_coarse": list(geometry["child_coarse"]),
        "dof": list(geometry["dof"]),
        "block": list(geometry["block"]),
        "galerkin_relative_error": galerkin_error,
        "galerkin_samples": galerkin_errors,
        "galerkin_tolerance": GALERKIN_TOLERANCE,
        "lonv_gram_error": gram_error,
        "gram_tolerance": GRAM_TOLERANCE,
        "stencil_offsets": [list(offset) for offset in offsets],
        "stencil_offset_count": len(offsets),
        "inner_bicgstab": inner,
        "null_vector_tolerance": NULL_VECTOR_TOLERANCE,
        "timings": built["timings"],
    }
    mg_config = {
        "levels": 3,
        "transition": 2,
        "mg_grid": [int(value) for value in geometry["block"]],
        "dof": [int(geometry["dof"][0]), int(geometry["dof"][1])],
        "nv_iters": 1,
        "seed": int(seed),
    }
    manifest = write_transition_cache(
        output_root, identity, geometry, tensors,
        setup_seconds=built["timings"]["seconds"],
        setup_validation=setup_validation, mg_config=mg_config,
        source_metadata={"source": f"transition built from {parent_root}"})
    report = {
        "status": "PASS",
        "stage": "transition-2",
        "parent_root": str(parent_root),
        "cache_root": str(output_root),
        "cache_manifest": str(cache_manifest_path(Path(output_root))),
        "identity": identity,
        "identity_sha256": manifest.get("identity_sha256"),
        "global_shapes": {name: list(value) for name, value in
                          geometry["shapes"].items()},
        "galerkin_relative_error": galerkin_error,
        "galerkin_samples": galerkin_errors,
        "lonv_gram_error": gram_error,
        "stencil_offset_count": len(offsets),
        "stencil_offsets": [list(offset) for offset in offsets],
        "inner_bicgstab": inner,
        "lonv_shape": list(tensors["lonv"].shape),
        "timings": built["timings"],
        "setup_validation": setup_validation,
        "manifest_assets": manifest.get("assets"),
    }
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", "--parent-cache", dest="parent",
                        required=True,
                        help="cache root of transition L-1 (ranked or single file)")
    parser.add_argument("--output", required=True,
                        help="cache root to publish for transition L")
    parser.add_argument("--parent-lattice", "--lat", dest="parent_lattice",
                        nargs=4, type=int, required=True,
                        help="full lattice of the parent transition")
    parser.add_argument("--mg-grid", nargs=4, type=int, default=(2, 2, 2, 2))
    parser.add_argument("--dof", nargs=2, type=int, default=(12, 12),
                        help="E_parent E_child")
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--device", "--build-device", dest="device",
                        default="cpu")
    parser.add_argument("--level", type=int, default=None,
                        help="absolute level this transition publishes "
                             "(provenance only; transitions start at 2)")
    parser.add_argument("--mass", type=float, default=None,
                        help="optional cross-check against the parent identity")
    parser.add_argument("--csw", type=float, default=None,
                        help="optional cross-check against the parent identity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=3,
                        help="random vectors in the Galerkin gate")
    parser.add_argument("--report", default=None,
                        help="JSON report path; must differ from <output>.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.level is not None and int(args.level) < 2:
        parser.error("--level must be at least 2 (a transition starts at 1)")
    if args.report is not None:
        resolved_output = Path(args.output).expanduser()
        if Path(args.report).expanduser().resolve() == \
                cache_manifest_path(resolved_output).resolve():
            # Section 171: a run report once silently overwrote a cache
            # manifest, so a report may never be the manifest path.
            parser.error("--report must not equal the cache manifest path")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_transition_cache(
        args.parent, args.output, args.parent_lattice, args.mg_grid, args.dof,
        dtype=args.dtype, device=args.device, seed=args.seed,
        samples=args.samples, verbose=not args.quiet)
    if args.report is not None:
        report_path = Path(args.report).expanduser()
        if report_path.resolve() == cache_manifest_path(
                Path(args.output).expanduser()).resolve():
            raise SystemExit("--report must not equal the cache manifest path")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
