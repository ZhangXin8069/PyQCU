"""Legacy-MG coarse-asset cache for the formal :mod:`pyqcu.test` port.

Why a cache exists
------------------
A Legacy multigrid solve needs four coarse tensors (``lonv`` blocked null
vectors, ``hnn``/``hdg`` blocked 33-stencils, ``sit`` on-site blocks).  Building
them costs far more than the solve itself, so they are published once as an
HDF5 file plus a JSON manifest and then reused.  Reuse is only sound when the
bytes on disk were produced for *exactly* the same physics and geometry, so the
cache is identified by a hash of the identity below and every read re-validates
it instead of trusting the file name.

Identity
--------
``dof``, ``mg_grid``, ``nv_iters``, ``seed``, ``dtype``, ``lattice`` are the
required fields (:data:`REQUIRED_IDENTITY_FIELDS`); ``mass``, ``csw``,
``gauge_sha256``, ``lattice_grid``, ``mpi_size`` and ``platform`` are bound as
well because a different mass, gauge, process grid or CPU architecture produces
different coarse tensors.  ``platform`` (:func:`platform_identity`) keeps a
cache built on one machine/byte order from being read on another.

Manifest
--------
The manifest is JSON next to the HDF5 file (``<cache>.json``) and mirrors the
cross-backend ``pyqcu.legacy-mg-cache.v1`` sidecar so the informal builders and
this port can read each other's caches:

======================  ====================================================
``status``              ``PASS`` only after both files are complete
``schema_version``      ``pyqcu.legacy-mg-cache.v1`` (shared layout)
``manifest_schema``     ``pyqcu.test.native-mg-cache.v1`` (this module)
``layout``              ``odd-Schur``
``identity``            canonical identity mapping
``identity_sha256``     SHA256 of the canonical identity JSON
``assets``              ``name -> {shape, dtype, sha256}`` per tensor
``file_sha256``         SHA256 of the HDF5 file at publication time
``file_size_bytes``     size of the HDF5 file at publication time
``platform``            :func:`platform_identity` of the writer
======================  ====================================================

Atomicity
---------
:func:`create_cache` publishes both files or neither: both are written to
sibling ``.<name>.<pid>.<n>.partial`` files, flushed and ``fsync``ed, the
manifest is built from the *written* data file, and only then are the two
published with ``os.replace`` (data first, manifest last, then a directory
``fsync``).  The data file published by an earlier call is kept as a hard link
(or a byte copy where hard links are unavailable) for the duration of the
publication, so a failure at any point - a failing builder, a manifest error,
or a failed post-write validation - removes the partials and restores the
previous artifact byte for byte.  A failed rebuild never destroys a good cache.
A crash leaves either a previous cache or a partial file that
:func:`validate_cache` rejects (status, size or hash mismatch), never a cache
that looks valid but is truncated.  Nothing here edits core code; the HDF5
layout and identity are the shared ``pyqcu.tools._legacy_mg_cache``
definitions, imported lazily.

Integrity
---------
Probing (:func:`cache_status`) and reading (:func:`read_cache`) verify the file
SHA256 recorded in the manifest by default, so a byte-corrupted payload is
reported instead of consumed; only an explicit ``verify_file_sha256=False``
skips the hash (for a caller that already verified those bytes, e.g. a reopen
after :func:`validate_cache`).  Both also accept ``require=...`` - a *partial*
identity (``lattice``, ``dtype``, ``seed``, ``mass``, ``csw``, ``mpi_size``,
...) the artifact must match - so a driver can bind the row it is about to
solve without inventing the cache-only fields (``dof``, ``mg_grid``,
``nv_iters``).  An HDF5 file that cannot be opened at all (truncated, empty,
garbage) is reported as :class:`CacheError`, never as a raw backend
``OSError``, so :func:`prepare_cache` can rebuild it.

Import policy
-------------
Only the standard library is imported at module import time; ``numpy``,
``h5py`` and the core cache helpers are imported inside the functions that need
them, so ``python -m pyqcu.test --list`` never pays for a backend import.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "ASSET_NAMES",
    "CACHE_ENVIRONMENT",
    "LEGACY_LAYOUT",
    "LEGACY_NULL_VECTOR_BUILDER",
    "LEGACY_SCHEMA",
    "MANIFEST_SCHEMA",
    "NULL_VECTOR_BUILDER",
    "NULL_VECTOR_BUILDER_ENVIRONMENT",
    "REQUIRED_IDENTITY_FIELDS",
    "UNSPECIFIED_NULL_VECTOR_BUILDER",
    "CacheEntry",
    "CacheError",
    "CacheMiss",
    "array_sha256",
    "asset_shapes",
    "cache_path",
    "cache_status",
    "canonical_identity_json",
    "check_datasets",
    "create_cache",
    "default_cache_dir",
    "file_sha256",
    "identity_sha256",
    "manifest_path",
    "mg_identity",
    "normalize_expected_identity",
    "normalize_identity",
    "null_vector_builder",
    "platform_identity",
    "prepare_cache",
    "read_cache",
    "read_manifest",
    "validate_cache",
    "write_manifest",
]

#: This module's manifest revision (the on-disk tensor layout is the legacy one).
MANIFEST_SCHEMA = "pyqcu.test.native-mg-cache.v1"
#: Shared cross-backend cache schema; false compatibility would break KUPL/CUDA.
LEGACY_SCHEMA = "pyqcu.legacy-mg-cache.v1"
#: Legacy caches are always the blocked odd-Schur operator.
LEGACY_LAYOUT = "odd-schur"
#: Dataset names of the four coarse tensors, in publication order.
ASSET_NAMES: tuple[str, ...] = ("lonv", "hnn", "hdg", "sit")
#: Identity fields a caller must provide.
REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = (
    "lattice", "dtype", "dof", "mg_grid", "nv_iters", "seed")
#: Every field bound into the identity hash, in report order.
IDENTITY_FIELDS: tuple[str, ...] = (
    "lattice", "dtype", "dof", "mg_grid", "nv_iters", "seed", "mass", "csw",
    "lattice_grid", "mpi_size", "gauge_sha256", "platform", "nv_builder")
#: Identity fields whose value is a four-extent integer tuple.
_EXTENT_IDENTITY_FIELDS = frozenset({"lattice", "mg_grid", "lattice_grid"})
#: Identity fields whose value is a single integer.
_SCALAR_INT_IDENTITY_FIELDS = frozenset({"nv_iters", "seed", "mpi_size"})
#: Revision of the null-vector builder that produced an artifact.  Part of the
#: identity because the builder's algorithm, not its inputs, decides what the
#: coarse basis is.
NULL_VECTOR_BUILDER = "inverse-iteration/2"
#: Revision recorded when the *old* loop ran: ``v -= S^-1(S v)``, i.e.
#: ``v - v``, which stores the inner solver's residual (white noise) instead of a
#: near-null vector.  Kept as a first-class revision so the old basis stays
#: measurable: ``KUPL_NV_BUILDER=legacy`` makes the C++ builder run that loop
#: again (cpp/kupl/qcu/src/lattice_solver.cpp), and this is what the artifact it
#: writes must record.
LEGACY_NULL_VECTOR_BUILDER = "legacy-residual/1"
#: Environment switch that selects the loop.  It is *not* a cache setting: it is
#: read by the C++ builder and by the Python identity code, so the revision an
#: artifact records and the revision a loader expects always come from the same
#: place.  Unset/empty means the current inverse iteration.
NULL_VECTOR_BUILDER_ENVIRONMENT = "KUPL_NV_BUILDER"
#: What an identity with no recorded builder normalizes to.  It must differ from
#: every real revision, otherwise normalisation would silently upgrade an old
#: artifact to the current builder and defeat the invalidation.
UNSPECIFIED_NULL_VECTOR_BUILDER = "unspecified"


def _builder_revision(value: Any) -> str:
    """Return the recorded builder revision, or the "unspecified" sentinel.

    Normalisation must not substitute the current constant for a missing field:
    that is exactly how an artifact written by the old ``v -= S^-1(S v)`` loop
    would end up hashing to the identity of a case that asks for the current
    builder.  An absent revision stays absent-in-effect so the hash differs.
    """
    if value is None:
        return UNSPECIFIED_NULL_VECTOR_BUILDER
    text = str(value).strip()
    if not text:
        return UNSPECIFIED_NULL_VECTOR_BUILDER
    return text


def null_vector_builder(environ: Mapping[str, str] | None = None) -> str:
    """Return the revision the C++ builder records for this environment.

    ``KUPL_NV_BUILDER=legacy`` selects the historical ``v -= S^-1(S v)`` loop,
    anything else (including unset and empty) selects the current inverse
    iteration, and an unrecognised value raises: a typo must never silently pick
    one of the two, because that is how an A/B ends up comparing the same basis
    twice.  The C++ side parses the same variable the same way
    (``cpp/kupl/qcu/src/lattice_solver.cpp``), and it prints the revision it
    actually used under ``KUPL_MG_DEBUG=1`` so the manifest and the run can be
    checked against each other.
    """
    source = os.environ if environ is None else environ
    raw = source.get(NULL_VECTOR_BUILDER_ENVIRONMENT) or ""
    # Exact spellings only, because the C++ parser is exact too
    # (lattice_solver.cpp nv_builder_legacy): a value Python accepted but C++
    # rejected (or vice versa) would make the manifest and the run disagree.
    if raw == "" or raw == "inverse":
        return NULL_VECTOR_BUILDER
    if raw == "legacy":
        return LEGACY_NULL_VECTOR_BUILDER
    raise CacheError(
        f"{NULL_VECTOR_BUILDER_ENVIRONMENT} must be 'legacy' or 'inverse' "
        f"(unset means 'inverse'), got {raw!r}")


#: Environment variable that overrides :func:`default_cache_dir`.
CACHE_ENVIRONMENT = "PYQCU_TEST_MG_CACHE"
#: dtype spellings mapped to the two cache dtypes.
_DTYPE_ALIASES = {
    "c64": "c64",
    "complex64": "c64",
    "float32": "c64",
    "single": "c64",
    "c128": "c128",
    "complex128": "c128",
    "float64": "c128",
    "double": "c128",
}
_NUMPY_DTYPES = {"c64": "complex64", "c128": "complex128"}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class CacheError(RuntimeError):
    """A cache artifact is invalid, inconsistent or was written for something else."""


class CacheMiss(CacheError):
    """No usable cache exists (absent, rejected by validation, or not buildable)."""


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def _int4(values: Any, name: str, *, positive: bool = True) -> tuple[int, ...]:
    """Normalize a four-extent value, rejecting the wrong length or sign."""
    if isinstance(values, (str, bytes)):
        text = str(values).strip().strip("()[]")
        parts: list[Any] = [part for part in text.replace(",", " ").split() if part]
    else:
        try:
            parts = list(values)
        except TypeError as error:
            raise CacheError(f"{name} must contain four integers, got {values!r}") from error
    try:
        extents = tuple(int(value) for value in parts)
    except (TypeError, ValueError) as error:
        raise CacheError(f"{name} must contain four integers, got {values!r}") from error
    if len(extents) != 4:
        raise CacheError(f"{name} must contain four extents, got {extents}")
    if any(value <= 0 for value in extents):
        raise CacheError(f"{name} extents must be positive, got {extents}")
    if not positive:
        return extents
    return extents


def _canonical_dtype(value: Any) -> str:
    """Map a dtype spelling to ``c64``/``c128`` (the only cache dtypes)."""
    name = getattr(value, "__name__", None) or str(value)
    name = name.replace("torch.", "").replace("numpy.", "").strip().lower()
    try:
        return _DTYPE_ALIASES[name]
    except KeyError:
        raise CacheError(
            f"cache dtype must be c64/c128 (complex64/complex128), got {value!r}"
        ) from None


def _sha256_text(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise CacheError(f"{name} must be a lowercase SHA256 digest, got {value!r}")
    return text


def platform_identity() -> dict[str, Any]:
    """Describe the platform a cache may be reused on.

    A coarse operator depends on the floating-point dtype and on the byte order
    of the machine that built it, so both are part of the identity.  The host
    name and the full kernel release are deliberately *not* included: they
    change without changing the bytes, and a cache identity that moves for an
    unrelated reason is useless.
    """
    return {
        "system": sys.platform,
        "machine": _platform.machine() or "unknown",
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
    }


def _normalize_platform(value: Any) -> dict[str, str]:
    if value is None:
        return {key: str(item) for key, item in platform_identity().items()}
    if not isinstance(value, Mapping):
        raise CacheError(f"platform identity must be a mapping, got {value!r}")
    return {str(key): str(item) for key, item in value.items()}


def normalize_identity(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a cache identity.

    The returned mapping is JSON-safe (lists, not tuples) and stable: the same
    physical request always produces the same mapping, which is what
    :func:`identity_sha256` hashes.
    """
    if not isinstance(values, Mapping):
        raise CacheError(f"cache identity must be a mapping, got {type(values).__name__}")
    missing = [name for name in REQUIRED_IDENTITY_FIELDS if name not in values]
    if missing:
        raise CacheError(f"cache identity is missing {missing}")

    lattice = _int4(values["lattice"], "lattice")
    if lattice[3] % 2:
        raise CacheError(f"odd-Schur cache requires an even time extent, got {lattice}")
    mg_grid = _int4(values["mg_grid"], "mg_grid")
    lattice_grid = _int4(values.get("lattice_grid", (1, 1, 1, 1)), "lattice_grid")
    for axis in range(4):
        fine_odd = lattice[axis] if axis < 3 else lattice[3] // 2
        if fine_odd % mg_grid[axis]:
            raise CacheError(
                f"odd-Schur fine extent {fine_odd} on axis {axis} is not "
                f"divisible by mg_grid {mg_grid}")
    dof = tuple(int(value) for value in values["dof"])
    if len(dof) < 2 or any(value <= 0 for value in dof):
        raise CacheError(f"dof must list at least two positive levels, got {dof}")
    if dof[0] != 12:
        raise CacheError(
            f"a transition cache starts from the 12-component fine field, "
            f"got dof[0]={dof[0]}")
    nv_iters = int(values["nv_iters"])
    if nv_iters <= 0:
        raise CacheError(f"nv_iters must be positive, got {nv_iters}")
    seed = int(values["seed"])
    if seed < 0:
        raise CacheError(f"seed must be non-negative, got {seed}")
    mpi_size = int(values.get("mpi_size", 1))
    if mpi_size < 1:
        raise CacheError(f"mpi_size must be positive, got {mpi_size}")
    if int(_product(lattice_grid)) != mpi_size:
        raise CacheError(
            f"lattice_grid {lattice_grid} describes {_product(lattice_grid)} "
            f"rank(s), not mpi_size {mpi_size}")
    mass = _finite(values.get("mass", 0.0), "mass")
    csw = _finite(values.get("csw", 0.0), "csw")

    identity: dict[str, Any] = {
        "lattice": list(lattice),
        "dtype": _canonical_dtype(values["dtype"]),
        "dof": [int(value) for value in dof],
        "mg_grid": list(mg_grid),
        "nv_iters": nv_iters,
        "seed": seed,
        "mass": mass,
        "csw": csw,
        "lattice_grid": list(lattice_grid),
        "mpi_size": mpi_size,
        "platform": _normalize_platform(values.get("platform")),
        # The builder revision, not the parameters.  KUPL's null-vector loop was
        # `v -= S^-1 (S v)`, which is v - v and therefore stored the inner
        # solver's residual instead of a near-null vector; it is now a real
        # inverse iteration, so artifacts from the two builders describe
        # different bases under identical inputs.  Without this field a cache
        # built by the old loop would satisfy the identity of a case that asks
        # for the new one, and `cache_policy=auto` would silently measure the
        # old basis.  Bump the string whenever the algorithm changes again.
        "nv_builder": _builder_revision(values.get("nv_builder")),
    }
    gauge = values.get("gauge_sha256")
    if gauge is not None:
        identity["gauge_sha256"] = _sha256_text(gauge, "gauge_sha256")
    return identity


def _int_value(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise CacheError(f"{name} must be an integer, got {value!r}") from error


def normalize_expected_identity(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a *partial* identity binding for a probe or a read.

    A caller that only knows the row it is about to solve (``lattice``,
    ``dtype``, ``seed``, ``mass``, ``csw``, ``mpi_size``, ...) must not invent
    the cache-only fields (``dof``, ``mg_grid``, ``nv_iters``), but the values
    it does know have to be normalized exactly like a full identity or the
    comparison would be meaningless.  Unknown field names are rejected: a typo
    in a bound field must fail the probe, never silently widen it.  ``None``
    values mean "not bound" and are dropped.
    """
    if not isinstance(values, Mapping):
        raise CacheError(
            f"expected cache identity fields must be a mapping, got "
            f"{type(values).__name__}")
    unknown = sorted(set(values) - set(IDENTITY_FIELDS))
    if unknown:
        raise CacheError(
            f"unknown cache identity field(s) {unknown}; bindable fields are "
            f"{list(IDENTITY_FIELDS)}")
    normalized: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            continue
        if name in _EXTENT_IDENTITY_FIELDS:
            normalized[name] = list(_int4(value, name))
        elif name == "dof":
            try:
                normalized[name] = [int(item) for item in value]
            except (TypeError, ValueError) as error:
                raise CacheError(
                    f"dof must list integers, got {value!r}") from error
        elif name == "dtype":
            normalized[name] = _canonical_dtype(value)
        elif name in ("mass", "csw"):
            normalized[name] = _finite(value, name)
        elif name in _SCALAR_INT_IDENTITY_FIELDS:
            normalized[name] = _int_value(value, name)
        elif name == "gauge_sha256":
            normalized[name] = _sha256_text(value, name)
        elif name == "platform":
            normalized[name] = _normalize_platform(value)
    return normalized


def _compare_required(actual: Mapping[str, Any], required: Mapping[str, Any],
                      path: Path) -> None:
    """Raise unless every bound (partial) identity field matches the artifact."""
    expected = normalize_expected_identity(required)
    differences = [
        f"{name}: cache={actual.get(name)!r} expected={expected[name]!r}"
        for name in sorted(expected)
        if actual.get(name) != expected[name]
    ]
    if differences:
        raise CacheError(
            f"cache {path} was not built for this request: "
            + "; ".join(differences))


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CacheError(f"{name} must be a finite number, got {value!r}") from error
    if number != number or number in (float("inf"), float("-inf")):
        raise CacheError(f"{name} must be finite, got {value!r}")
    return number


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def mg_identity(*, lattice: Sequence[int], mg_grid: Sequence[int],
                dtype: Any = "c64", dof: Sequence[int] = (12, 12),
                nv_iters: int = 2, seed: int = 42, mass: float = 0.0,
                csw: float = 0.0, gauge_sha256: str | None = None,
                lattice_grid: Sequence[int] = (1, 1, 1, 1),
                mpi_size: int = 1,
                platform: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a normalized Legacy-MG cache identity.

    ``dof`` is the per-level component count (``dof[0]`` is the 12-component
    fine odd-Schur field, ``dof[1]`` the coarse basis dimension), ``mg_grid``
    the blocking factor, ``nv_iters`` the null-vector inverse-iteration count
    and ``seed`` its random start.  ``gauge_sha256`` binds the gauge
    configuration when one is available; a cache built from a generated gauge
    simply omits it and is then only reusable for that same generated input.
    """
    return normalize_identity({
        "lattice": lattice,
        "dtype": dtype,
        "dof": dof,
        "mg_grid": mg_grid,
        "nv_iters": nv_iters,
        "seed": seed,
        "mass": mass,
        "csw": csw,
        "gauge_sha256": gauge_sha256,
        "lattice_grid": lattice_grid,
        "mpi_size": mpi_size,
        "platform": platform_identity() if platform is None else platform,
        # The builder revision this code writes.  Passed explicitly because
        # normalize_identity deliberately does NOT substitute it for a missing
        # value: an artifact that records no revision must hash differently from
        # one written by the current builder.  ``null_vector_builder()`` reads
        # KUPL_NV_BUILDER, so an identity built while the legacy loop is selected
        # declares the legacy revision instead of claiming the inverse one.
        "nv_builder": null_vector_builder(),
    })


def canonical_identity_json(identity: Mapping[str, Any]) -> str:
    """Canonical JSON of a normalized identity (the hashed payload)."""
    return json.dumps(normalize_identity(identity), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """SHA256 of :func:`canonical_identity_json`."""
    return hashlib.sha256(canonical_identity_json(identity).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# layout / paths
# ---------------------------------------------------------------------------
def asset_shapes(identity: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the exact dataset shape of every asset for ``identity``.

    The formula is the shared ``pyqcu.tools._legacy_mg_cache`` one (imported
    lazily), so a cache written here stays readable by the informal KUPL/CUDA
    builders and vice versa.
    """
    normalized = normalize_identity(identity)
    core = _core_module()
    return {
        name: tuple(int(value) for value in shape)
        for name, shape in core.legacy_mg_cache_shapes(
            normalized["lattice"], normalized["mg_grid"],
            normalized["dof"]).items()
    }


def _core_module() -> Any:
    """Import the shared cache-format helpers (read-only use of core)."""
    try:
        from pyqcu.tools import _legacy_mg_cache as core
    except Exception as error:  # noqa: BLE001 - reported as a cache failure
        raise CacheError(
            f"the shared Legacy-MG cache helpers are unavailable: "
            f"{type(error).__name__}: {error}") from error
    return core


def _numpy_dtype(identity: Mapping[str, Any]) -> Any:
    import numpy as np

    return np.dtype(_NUMPY_DTYPES[normalize_identity(identity)["dtype"]])


def default_cache_dir() -> Path:
    """Cache directory: ``$PYQCU_TEST_MG_CACHE``, else the user cache dir.

    The packaged port must not write into the repository, so the fallback is
    ``$XDG_CACHE_HOME/pyqcu/mg`` (``~/.cache/pyqcu/mg``).
    """
    override = os.environ.get(CACHE_ENVIRONMENT, "").strip()
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "pyqcu" / "mg"


def manifest_path(cache: str | os.PathLike[str]) -> Path:
    """Sidecar manifest path (``<cache>.json``) of an HDF5 cache file."""
    return Path(cache).with_suffix(".json")


def cache_path(identity: Mapping[str, Any],
               directory: str | os.PathLike[str] | None = None, *,
               prefix: str = "legacy_mg") -> Path:
    """Content-addressed cache path for ``identity``."""
    root = default_cache_dir() if directory is None else Path(directory)
    digest = identity_sha256(identity)[:24]
    return root / f"{prefix}_{digest}.h5"


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def array_sha256(value: Any) -> str:
    """Logical SHA256 of an array, independent of HDF5 storage details.

    Same recipe as ``pyqcu.tools._legacy_mg_cache.logical_array_sha256``: hash
    a canonical ``{shape, dtype}`` header and then the C-order bytes, so a
    tensor read from HDF5, from a gathered MPI slab or built in memory compares
    equal.
    """
    shape = tuple(int(extent) for extent in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", ""))
    tobytes = getattr(value, "tobytes", None)
    if not callable(tobytes):
        raise CacheError(
            f"cannot hash {type(value).__name__}: no array-like .tobytes()")
    digest = hashlib.sha256()
    header = {"shape": shape, "dtype": dtype}
    digest.update(json.dumps(header, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))
    digest.update(tobytes())
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Streaming SHA256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# manifest I/O
# ---------------------------------------------------------------------------
def read_manifest(cache: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the sidecar manifest of ``cache``; raise :class:`CacheMiss` if absent."""
    path = manifest_path(cache)
    if not path.is_file():
        raise CacheMiss(f"cache manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CacheError(f"cache manifest is unreadable: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CacheError(f"cache manifest is not a JSON object: {path}")
    return payload


def _scratch_path(target: Path, kind: str) -> Path:
    """Sibling scratch name (same directory, so ``os.replace`` works)."""
    return target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.{kind}")


def _partial_path(target: Path) -> Path:
    """Sibling name a not-yet-published artifact is written to."""
    return _scratch_path(target, "partial")


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _fsync_path(path: Path) -> None:
    """Best-effort durability for one file or directory path."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for the directory entries a publish created."""
    _fsync_path(path)


def _write_bytes_synced(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` and flush it to disk (no rename)."""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial_path(path)
    try:
        _write_bytes_synced(temporary, payload)
        os.replace(temporary, path)
    finally:
        _unlink_quietly(temporary)
        _fsync_directory(path.parent)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _manifest_text(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest exactly as it is published."""
    return json.dumps(dict(manifest), indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False) + "\n"


def write_manifest(cache: str | os.PathLike[str],
                   manifest: Mapping[str, Any]) -> Path:
    """Atomically publish the manifest of ``cache``."""
    path = manifest_path(cache)
    _atomic_write_text(path, _manifest_text(manifest))
    return path


def check_datasets(datasets: Mapping[str, Any],
                   identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate in-memory assets against the layout; return contiguous arrays.

    Every asset must be present, carry the canonical shape and the identity's
    dtype.  The returned mapping is C-contiguous, so the same arrays can be
    hashed, written and handed to the native ABI without a second copy.
    """
    if not isinstance(datasets, Mapping):
        raise CacheError(
            f"cache assets must be a mapping, got {type(datasets).__name__}")
    unknown = sorted(set(datasets) - set(ASSET_NAMES))
    if unknown:
        raise CacheError(
            f"unknown cache asset(s) {unknown}; expected {list(ASSET_NAMES)}")
    missing = [name for name in ASSET_NAMES if name not in datasets]
    if missing:
        raise CacheError(f"cache assets are incomplete: missing {missing}")
    expected = asset_shapes(identity)
    dtype = _numpy_dtype(identity)
    checked: dict[str, Any] = {}
    for name in ASSET_NAMES:
        value = _as_array(datasets[name])
        if tuple(value.shape) != expected[name]:
            raise CacheError(
                f"{name} shape {tuple(value.shape)} != {expected[name]}")
        if value.dtype != dtype:
            raise CacheError(f"{name} dtype {value.dtype} != {dtype}")
        checked[name] = value
    return checked


def _asset_records(identity: Mapping[str, Any],
                   datasets: Mapping[str, Any]) -> dict[str, Any]:
    """Per-asset ``{shape, dtype, sha256}`` records for in-memory arrays."""
    checked = check_datasets(datasets, identity)
    return {
        name: {
            "shape": [int(extent) for extent in value.shape],
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
        for name, value in checked.items()
    }


def _open_hdf5_read(cache: Path) -> Any:
    """Open ``cache`` for reading, reporting an unopenable file as a cache error.

    ``h5py`` raises a bare :class:`OSError` for a truncated, empty or otherwise
    corrupt container.  Left untyped, that exception escapes the
    :class:`CacheError` contract and :func:`prepare_cache` cannot rebuild the
    artifact; converting it here keeps every cache read failure in one
    hierarchy while preserving the backend message.
    """
    h5py = _import_h5py()
    try:
        return h5py.File(cache, "r")
    except OSError as error:
        raise CacheError(
            f"cache {cache} cannot be opened as HDF5 "
            f"({type(error).__name__}: {error})") from error


def _file_asset_records(cache: Path,
                        identity: Mapping[str, Any]) -> dict[str, Any]:
    """Per-asset records hashed from the already-written HDF5 file."""
    expected = asset_shapes(identity)
    dtype = _numpy_dtype(identity)
    records: dict[str, Any] = {}

    try:
        with _open_hdf5_read(cache) as handle:
            missing = [name for name in ASSET_NAMES if name not in handle]
            if missing:
                raise CacheError(f"cache {cache} is missing datasets {missing}")
            for name in ASSET_NAMES:
                dataset = handle[name]
                if tuple(dataset.shape) != expected[name]:
                    raise CacheError(
                        f"{name} shape {tuple(dataset.shape)} != "
                        f"{expected[name]}")
                if _numpy_dtype_of(dataset) != dtype:
                    raise CacheError(f"{name} dtype {dataset.dtype} != {dtype}")
                records[name] = {
                    "shape": [int(extent) for extent in dataset.shape],
                    "dtype": str(dataset.dtype),
                    "sha256": _dataset_sha256(dataset),
                }
    except OSError as error:
        raise CacheError(
            f"cache {cache} cannot be read as HDF5 "
            f"({type(error).__name__}: {error})") from error
    return records


def _dataset_sha256(dataset: Any) -> str:
    """Hash an HDF5 dataset in bounded blocks (no full-tensor allocation)."""
    try:
        core = _core_module()
        return str(core.logical_dataset_sha256(dataset))
    except Exception:  # noqa: BLE001 - fall back to a whole-array read
        return array_sha256(dataset[...])


def _numpy_dtype_of(dataset: Any) -> Any:
    import numpy as np

    return np.dtype(dataset.dtype)


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:
        raise CacheError(
            f"h5py is required for Legacy-MG cache I/O: "
            f"{type(error).__name__}: {error}") from error
    return h5py


def _as_array(value: Any) -> Any:
    """Return a C-contiguous numpy view of a numpy/torch array."""
    import numpy as np

    numpy_value = getattr(value, "numpy", None)
    if callable(numpy_value):
        value = numpy_value()
    return np.ascontiguousarray(value)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_manifest(cache: str | os.PathLike[str], identity: Mapping[str, Any], *,
                   datasets: Mapping[str, Any] | None = None,
                   setup_method: str = "native",
                   setup_backend: str = "unknown",
                   setup_seconds: float | None = None,
                   validation: Mapping[str, Any] | None = None,
                   notes: Sequence[str] = ()) -> dict[str, Any]:
    """Build the manifest of an existing (or about-to-be-written) cache file.

    ``datasets=None`` hashes the datasets already present in the HDF5 file;
    passing the arrays hashes those instead (the write path uses this to avoid
    a second read).
    """
    path = Path(cache)
    normalized = normalize_identity(identity)
    records = (_file_asset_records(path, normalized) if datasets is None
               else _asset_records(normalized, datasets))
    report: dict[str, Any] = {
        "status": "PASS",
        "schema_version": LEGACY_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "layout": LEGACY_LAYOUT,
        "generator": "pyqcu.test.cache",
        "created_at": _utc_timestamp(),
        "identity": normalized,
        "identity_sha256": identity_sha256(normalized),
        "platform": dict(normalized["platform"]),
        "assets": records,
        "setup_method": str(setup_method),
        "setup_backend": str(setup_backend),
        "notes": [str(note) for note in notes],
    }
    # Flat aliases keep the informal readers (``pyqcu.tools._legacy_mg_cache``)
    # working; ``identity`` above stays the single source of truth.
    for name in REQUIRED_IDENTITY_FIELDS:
        report[name] = normalized[name]
    report["mass"] = normalized["mass"]
    report["csw"] = normalized["csw"]
    if "gauge_sha256" in normalized:
        report["gauge_sha256"] = normalized["gauge_sha256"]
    if setup_seconds is not None:
        report["setup_seconds"] = float(setup_seconds)
    if validation is not None:
        report["setup_validation"] = dict(validation)
    if path.is_file():
        report["file_size_bytes"] = path.stat().st_size
        report["file_sha256"] = file_sha256(path)
        # Aliases used by older hitting drivers.
        report["output_size_bytes"] = report["file_size_bytes"]
        report["output_sha256"] = report["file_sha256"]
    return report


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheEntry:
    """One resolved cache artifact and the evidence that it is the right one."""

    path: Path
    manifest_path: Path
    identity: Mapping[str, Any]
    identity_sha256: str
    status: str = "PASS"
    reused: bool = False
    file_sha256: str | None = None
    file_size_bytes: int | None = None
    assets: Mapping[str, Any] = field(default_factory=dict)
    setup_method: str | None = None
    setup_backend: str | None = None
    setup_seconds: float | None = None
    platform: Mapping[str, Any] = field(default_factory=dict)
    file_hash_verified: bool = False
    dataset_hashes_verified: bool = False
    seconds: float | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description for a report or an observation dict."""
        return {
            "cache_path": str(self.path),
            "cache_manifest_path": str(self.manifest_path),
            "cache_status": self.status,
            "cache_reused": bool(self.reused),
            "cache_identity": dict(self.identity),
            "cache_identity_sha256": self.identity_sha256,
            "cache_file_sha256": self.file_sha256,
            "cache_file_size_bytes": self.file_size_bytes,
            "cache_assets": {name: dict(record)
                             for name, record in self.assets.items()},
            "cache_setup_method": self.setup_method,
            "cache_setup_backend": self.setup_backend,
            "cache_setup_seconds": self.setup_seconds,
            "cache_platform": dict(self.platform),
            "cache_file_hash_verified": bool(self.file_hash_verified),
            "cache_dataset_hashes_verified": bool(self.dataset_hashes_verified),
            "cache_seconds": self.seconds,
            "cache_notes": list(self.notes),
        }


def _entry_from_manifest(cache: Path, manifest: Mapping[str, Any], *,
                         identity: Mapping[str, Any] | None,
                         reused: bool, status: str,
                         file_hash_verified: bool = False,
                         dataset_hashes_verified: bool = False,
                         seconds: float | None = None,
                         notes: Sequence[str] = ()) -> CacheEntry:
    stored = manifest.get("identity")
    if not isinstance(stored, Mapping):
        stored = {name: manifest[name] for name in IDENTITY_FIELDS
                  if name in manifest}
    resolved = normalize_identity(identity if identity is not None else stored)
    return CacheEntry(
        path=cache,
        manifest_path=manifest_path(cache),
        identity=resolved,
        identity_sha256=identity_sha256(resolved),
        status=status,
        reused=reused,
        file_sha256=manifest.get("file_sha256"),
        file_size_bytes=manifest.get("file_size_bytes"),
        assets={name: dict(record)
                for name, record in (manifest.get("assets") or {}).items()},
        setup_method=manifest.get("setup_method"),
        setup_backend=manifest.get("setup_backend"),
        setup_seconds=manifest.get("setup_seconds"),
        platform=manifest.get("platform") or resolved.get("platform") or {},
        file_hash_verified=file_hash_verified,
        dataset_hashes_verified=dataset_hashes_verified,
        seconds=seconds,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# atomic create
# ---------------------------------------------------------------------------
def _write_hdf5(target: Path, datasets: Mapping[str, Any],
                identity: Mapping[str, Any]) -> None:
    """Write the four datasets to ``target`` (a caller-owned partial file)."""
    h5py = _import_h5py()
    target.parent.mkdir(parents=True, exist_ok=True)
    attrs = {
        "pyqcu_cache_schema": LEGACY_SCHEMA,
        "pyqcu_cache_layout": LEGACY_LAYOUT,
        "pyqcu_cache_identity_sha256": identity_sha256(identity),
        "pyqcu_test_manifest_schema": MANIFEST_SCHEMA,
    }
    try:
        with h5py.File(target, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            for name in ASSET_NAMES:
                handle.create_dataset(name, data=_as_array(datasets[name]))
    except BaseException:
        _unlink_quietly(target)
        raise


def _preserve_previous(path: Path, backup: Path) -> None:
    """Keep the published artifact restorable while a replacement is published.

    A hard link is free and leaves ``path`` itself valid for the whole
    publication; a filesystem without hard links (some network and exFAT
    mounts) gets a byte copy instead, which costs one copy only on a replace.
    If neither works the caller must not publish: the previous artifact could
    not be restored afterwards, so failing here - before anything is replaced -
    is the only way to keep the "a failed rebuild never destroys a cache"
    promise on every filesystem.
    """
    try:
        os.link(path, backup)
        return
    except OSError:
        pass
    try:
        shutil.copyfile(path, backup)
    except OSError as error:
        _unlink_quietly(backup)
        raise CacheError(
            f"cannot preserve the previous cache {path} before replacing it "
            f"({type(error).__name__}: {error}); the previous artifact was "
            f"left untouched") from error
    _fsync_path(backup)


def create_cache(cache: str | os.PathLike[str],
                 datasets: Mapping[str, Any], *, identity: Mapping[str, Any],
                 setup_method: str = "native",
                 setup_backend: str = "unknown",
                 setup_seconds: float | None = None,
                 validation: Mapping[str, Any] | None = None,
                 notes: Sequence[str] = (), replace_existing: bool = False,
                 verify: bool = True) -> CacheEntry:
    """Publish a cache atomically and return the validated entry.

    Both files are written to sibling partial files first: the HDF5 data, then
    the manifest hashed *from that written file*.  Publication is two
    ``os.replace`` calls (data first, manifest last) followed by a directory
    ``fsync``, so a reader never sees a manifest without the tensors it
    describes - and if it does catch the old manifest against the new data, the
    hash check rejects it.  The data file published by an earlier call is kept
    alive as a hard link (or a byte copy when the filesystem has no hard links)
    until the new artifact has been validated, so on any failure (a raising
    builder, a manifest error, or a failed post-write validation) the partial
    files are removed and the previous cache - data and manifest - is restored
    byte for byte.  If the previous artifact cannot be preserved or restored at
    all, the publication is refused before anything is replaced and the backup
    is kept on disk, with its path in the error.  A failed rebuild never
    destroys a good artifact and never leaves a manifest without its file.
    """
    started = time.perf_counter()
    path = Path(cache)
    normalized = normalize_identity(identity)
    if path.exists() and not replace_existing:
        raise CacheError(
            f"cache already exists and replace_existing=False: {path}")
    if path.exists() and not path.is_file():
        raise CacheError(f"cache path is not a file: {path}")
    # Shapes/dtypes are checked before anything touches the filesystem, so an
    # inconsistent request can never leave a partially written file behind.
    checked = check_datasets(datasets, normalized)
    manifest_file = manifest_path(path)
    data_partial = _partial_path(path)
    manifest_partial = _partial_path(manifest_file)
    backup = _scratch_path(path, "backup")
    previous_manifest = (manifest_file.read_bytes()
                         if manifest_file.is_file() else None)
    had_previous = path.is_file()
    backup_linked = False
    backup_consumed = False
    restore_failed = False
    published_data = False
    published_manifest = False

    def rollback() -> None:
        """Put the artifact published before this call back into place."""
        nonlocal backup_consumed
        if backup_linked:
            os.replace(backup, path)
            backup_consumed = True
        elif published_data:
            _unlink_quietly(path)
        if published_manifest:
            if previous_manifest is None:
                _unlink_quietly(manifest_file)
            else:
                _atomic_write_bytes(manifest_file, previous_manifest)

    entry: CacheEntry | None = None
    try:
        _write_hdf5(data_partial, checked, normalized)
        manifest = build_manifest(
            data_partial, normalized, datasets=None,
            setup_method=setup_method, setup_backend=setup_backend,
            setup_seconds=setup_seconds, validation=validation, notes=notes)
        _write_bytes_synced(manifest_partial,
                            _manifest_text(manifest).encode("utf-8"))
        if had_previous:
            # Refuses (before anything is replaced) when the previous bytes
            # cannot be preserved, so the restore below is always possible.
            _preserve_previous(path, backup)
            backup_linked = True
        os.replace(data_partial, path)
        published_data = True
        os.replace(manifest_partial, manifest_file)
        published_manifest = True
        _fsync_directory(path.parent)
        if verify:
            entry = validate_cache(path, identity=normalized)
        else:
            entry = _entry_from_manifest(
                path, read_manifest(path), identity=normalized, reused=False,
                status="CREATED")
    except BaseException as error:
        try:
            rollback()
        except OSError as restore_error:
            restore_failed = True
            kept = backup_linked and not backup_consumed and backup.is_file()
            raise CacheError(
                f"cache publication failed ({type(error).__name__}: "
                f"{error}) and the previous artifact at {path} could not be "
                f"restored ({type(restore_error).__name__}: {restore_error})"
                + (f"; the previous data file is preserved at {backup}" if kept
                   else "")) from restore_error
        raise
    finally:
        _unlink_quietly(data_partial)
        _unlink_quietly(manifest_partial)
        # A backup that could not be restored is the only copy of the previous
        # data: keep it and let the CacheError above name it.
        if backup_linked and not restore_failed:
            _unlink_quietly(backup)
    return replace(entry, status="CREATED", reused=False,
                   seconds=time.perf_counter() - started)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_cache(cache: str | os.PathLike[str], *,
                   identity: Mapping[str, Any] | None = None,
                   require: Mapping[str, Any] | None = None,
                   verify_file_sha256: bool = True,
                   verify_dataset_sha256: bool = False) -> CacheEntry:
    """Fail closed unless ``cache`` is the artifact described by ``identity``.

    Checks, in order: the file exists and is non-empty; the manifest exists,
    reports ``PASS`` and uses a known schema/layout; the manifest identity is
    self-consistent and matches ``identity`` field by field; every field of
    ``require`` (a *partial* identity, see :func:`normalize_expected_identity`)
    matches too; every asset shape/dtype matches the canonical layout; the file
    size (and SHA256 when requested) matches the manifest.  Dataset content
    hashes are checked on request because they read the whole file.

    An HDF5 container that cannot even be opened raises :class:`CacheError`
    (never a raw ``OSError``), so a caller such as :func:`prepare_cache` can
    treat it like any other unusable artifact.
    """
    path = Path(cache)
    if not path.is_file():
        raise CacheMiss(f"cache is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise CacheError(f"cache is empty: {path}")
    manifest = read_manifest(path)
    status = str(manifest.get("status", "")).upper()
    if status != "PASS":
        raise CacheError(f"cache manifest status is {status or '<unset>'}: "
                         f"{manifest_path(path)}")
    schema = manifest.get("schema_version")
    if schema not in (None, LEGACY_SCHEMA):
        raise CacheError(f"unsupported cache schema {schema!r}: {path}")
    layout = manifest.get("layout")
    if layout not in (None, LEGACY_LAYOUT):
        raise CacheError(f"unsupported cache layout {layout!r}: {path}")

    stored = manifest.get("identity")
    if not isinstance(stored, Mapping):
        stored = {name: manifest[name] for name in IDENTITY_FIELDS
                  if name in manifest}
    if not stored:
        raise CacheError(f"cache manifest carries no identity: {path}")
    actual = normalize_identity(stored)
    if manifest.get("manifest_schema") == MANIFEST_SCHEMA:
        # Our own manifests declare the digest of the identity they carry; a
        # mismatch means the mapping was edited after publication.  A foreign
        # (cross-backend) manifest is not checked here: the informal writers
        # canonicalize a different field set.
        declared_identity = manifest.get("identity_sha256")
        if declared_identity is not None:
            digest = identity_sha256(actual)
            if str(declared_identity) != digest:
                raise CacheError(
                    f"cache manifest identity hash {declared_identity} != "
                    f"{digest}: {path}")
    if identity is not None:
        expected = normalize_identity(identity)
        _compare_identity(actual, expected, path)
        resolved = expected
    else:
        resolved = actual
    if require is not None:
        _compare_required(actual, require, path)

    expected_shapes = asset_shapes(resolved)
    records = manifest.get("assets")
    if not isinstance(records, Mapping) or not records:
        raise CacheError(f"cache manifest lists no assets: {path}")
    dtype = _numpy_dtype(resolved)
    dataset_verified = False
    try:
        with _open_hdf5_read(path) as handle:
            missing = [name for name in ASSET_NAMES if name not in handle]
            if missing:
                raise CacheError(f"cache {path} is missing datasets {missing}")
            for name in ASSET_NAMES:
                dataset = handle[name]
                if tuple(dataset.shape) != expected_shapes[name]:
                    raise CacheError(
                        f"{name} shape {tuple(dataset.shape)} != "
                        f"{expected_shapes[name]}: {path}")
                if _numpy_dtype_of(dataset) != dtype:
                    raise CacheError(
                        f"{name} dtype {dataset.dtype} != {dtype}: {path}")
                record = records.get(name)
                if not isinstance(record, Mapping):
                    raise CacheError(
                        f"cache manifest has no record for {name}: {path}")
                if record.get("sha256") and verify_dataset_sha256:
                    digest = _dataset_sha256(dataset)
                    if str(record["sha256"]) != digest:
                        raise CacheError(
                            f"{name} content hash {digest} != manifest "
                            f"{record['sha256']}: {path}")
                    dataset_verified = True
    except OSError as error:
        raise CacheError(
            f"cache {path} cannot be read as HDF5 "
            f"({type(error).__name__}: {error})") from error

    declared_size = manifest.get("file_size_bytes")
    if declared_size is not None and int(declared_size) != int(size):
        raise CacheError(
            f"cache size {size} != manifest {declared_size}: {path}")
    file_verified = False
    declared_digest = manifest.get("file_sha256")
    if verify_file_sha256:
        if not declared_digest:
            raise CacheError(f"cache manifest has no file_sha256: {path}")
        digest = file_sha256(path)
        if str(declared_digest) != digest:
            raise CacheError(
                f"cache file hash {digest} != manifest {declared_digest}: {path}")
        file_verified = True
    return _entry_from_manifest(
        path, manifest, identity=resolved, reused=True, status="PASS",
        file_hash_verified=file_verified,
        dataset_hashes_verified=dataset_verified)


def _compare_identity(actual: Mapping[str, Any], expected: Mapping[str, Any],
                      path: Path) -> None:
    """Raise unless every bound identity field matches."""
    differences = [
        f"{name}: cache={actual.get(name)!r} expected={expected.get(name)!r}"
        for name in IDENTITY_FIELDS
        if name != "gauge_sha256" and actual.get(name) != expected.get(name)
    ]
    gauge_expected = expected.get("gauge_sha256")
    gauge_actual = actual.get("gauge_sha256")
    if gauge_expected is not None and gauge_actual != gauge_expected:
        differences.append(
            f"gauge_sha256: cache={gauge_actual!r} expected={gauge_expected!r}")
    if differences:
        raise CacheError(
            f"cache identity does not match the request for {path}: "
            + "; ".join(differences))


# ---------------------------------------------------------------------------
# prepare / read
# ---------------------------------------------------------------------------
def prepare_cache(identity: Mapping[str, Any], *,
                  cache: str | os.PathLike[str] | None = None,
                  directory: str | os.PathLike[str] | None = None,
                  builder: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
                  setup_method: str = "native",
                  setup_backend: str = "unknown",
                  replace_invalid: bool = True,
                  verify_file_sha256: bool = True,
                  verify_dataset_sha256: bool = False,
                  notes: Sequence[str] = ()) -> CacheEntry:
    """Resolve a cache for ``identity``: reuse it, or build it with ``builder``.

    ``builder`` receives the normalized identity and returns a mapping
    ``name -> array`` for all four assets (:data:`ASSET_NAMES`).  It is only
    called when no valid cache exists, and only for a single-rank identity: the
    coarse tensors are rank-local slabs, so a multi-rank request must build
    them per rank and cannot share one single-file cache.

    An unusable artifact is *not* deleted before the rebuild: it stays on disk
    until :func:`create_cache` publishes the replacement atomically, so a
    builder that fails - or any error in the publish - leaves the previous file
    exactly as it was.  "Unusable" includes an HDF5 file that cannot be opened
    and a data file whose manifest is missing (the state an interrupted
    publication leaves behind); both are rebuilt.  An artifact that is
    internally valid but was built for a different identity is a good cache for
    something else, so it is never overwritten: that case raises
    :class:`CacheError` and asks for another path.  ``replace_invalid`` only
    governs genuinely unusable artifacts.
    """
    normalized = normalize_identity(identity)
    path = Path(cache) if cache is not None else cache_path(normalized, directory)
    if path.is_file():
        try:
            entry = validate_cache(
                path, identity=normalized,
                verify_file_sha256=verify_file_sha256,
                verify_dataset_sha256=verify_dataset_sha256)
        except CacheError as error:
            if not replace_invalid or builder is None:
                raise
            other = _valid_for_other_identity(path)
            if other is not None:
                raise CacheError(
                    f"refusing to replace {path}: it holds a valid cache "
                    f"built for another identity (identity_sha256={other}); "
                    f"remove it or pass another cache/directory to rebuild"
                ) from error
            # The unusable file is replaced by create_cache's atomic publish,
            # which keeps it intact until the new artifact is complete.
            notes = tuple(notes) + (f"rebuilt stale cache: {error}",)
        else:
            return replace(entry, status="REUSE", reused=True)
    elif path.exists():
        raise CacheError(f"cache path is not a file: {path}")
    if builder is None:
        raise CacheMiss(
            f"no valid Legacy-MG cache at {path} and no builder was supplied")
    if int(normalized["mpi_size"]) != 1:
        raise CacheError(
            f"Legacy-MG coarse assets are rank-local: refusing to publish a "
            f"single-file cache for mpi_size={normalized['mpi_size']} "
            f"(lattice_grid={normalized['lattice_grid']})")
    datasets = builder(dict(normalized))
    if not isinstance(datasets, Mapping):
        raise CacheError(
            f"cache builder returned {type(datasets).__name__}, expected a mapping")
    return create_cache(
        path, datasets, identity=normalized, setup_method=setup_method,
        setup_backend=setup_backend, replace_existing=True,
        verify=True, notes=notes)


def _valid_for_other_identity(path: Path) -> str | None:
    """Digest of the artifact at ``path`` if it is a valid cache for anything.

    A file that :func:`validate_cache` accepts without an expected identity is
    a good artifact for *some* request; replacing it because the current
    request differs would destroy usable data.  An unusable file returns
    ``None`` and may be rebuilt.
    """
    try:
        entry = validate_cache(path, verify_file_sha256=True)
    except CacheError:
        return None
    return entry.identity_sha256


def read_cache(cache: str | os.PathLike[str], *,
               identity: Mapping[str, Any] | None = None,
               require: Mapping[str, Any] | None = None,
               names: Sequence[str] = ASSET_NAMES,
               validate: bool = True,
               verify_file_sha256: bool = True,
               verify_dataset_sha256: bool = False,
               ) -> dict[str, Any]:
    """Validate and read the cached coarse tensors as contiguous numpy arrays.

    The manifest file SHA256 is verified by default: a cache whose payload was
    damaged on disk must not be consumed silently.  A caller that has already
    verified the same bytes (for example right after :func:`validate_cache`, or
    a hot read loop that accepts the risk) must say so with
    ``verify_file_sha256=False``; ``validate=False`` skips validation entirely
    and is the documented "I already checked this file" path.
    """
    path = Path(cache)
    if validate:
        validate_cache(path, identity=identity, require=require,
                       verify_file_sha256=verify_file_sha256,
                       verify_dataset_sha256=verify_dataset_sha256)
    elif not path.is_file():
        raise CacheMiss(f"cache is missing: {path}")
    requested = tuple(str(name) for name in names)
    unknown = [name for name in requested if name not in ASSET_NAMES]
    if unknown:
        raise CacheError(f"unknown cache asset(s) {unknown}; expected {list(ASSET_NAMES)}")
    assets: dict[str, Any] = {}
    try:
        with _open_hdf5_read(path) as handle:
            for name in requested:
                if name not in handle:
                    raise CacheError(f"cache {path} is missing dataset {name}")
                value = handle[name][...]
                assets[name] = _as_array(value)
    except OSError as error:
        raise CacheError(
            f"cache {path} cannot be read as HDF5 "
            f"({type(error).__name__}: {error})") from error
    return assets


def cache_status(cache: str | os.PathLike[str],
                 identity: Mapping[str, Any] | None = None, *,
                 require: Mapping[str, Any] | None = None,
                 verify_file_sha256: bool = True,
                 verify_dataset_sha256: bool = False) -> tuple[bool, str]:
    """Non-raising probe: ``(usable, reason)`` for ``cache``.

    The file SHA256 is verified by default: a probe exists to keep a broken
    artifact away from a solve, and probing a cache is not a hot loop.
    ``require`` binds the fields the caller does know (the selected rows'
    ``lattice``/``dtype``/``seed``/...) so a cache built for another
    configuration reads as unusable instead of ready.  Pass
    ``verify_file_sha256=False`` only for an explicit cheap structural check.
    """
    try:
        validate_cache(cache, identity=identity, require=require,
                       verify_file_sha256=verify_file_sha256,
                       verify_dataset_sha256=verify_dataset_sha256)
    except BaseException as error:  # noqa: BLE001 - a probe must not raise
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return False, f"{type(error).__name__}: {error}"
    return True, ""
