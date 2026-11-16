"""HDF5 gauge/source artifacts for the formal :mod:`pyqcu.test` port.

Why this module exists
----------------------
The 2026-11 KUPL production matrix lost a whole L32 block to two input files
that had the *full* byte length but an uninterpretable HDF5 object header: every
rank died inside ``h5py.File(..., "r")`` with ``bad object header version
number``.  The post-mortem - ``docs/kupl_hdf5_source_gauge_rootcause_20261115.md``
and its reference driver ``pyqcu/testing/kupl/test_kunpeng_all.py`` - found
three defects that this module closes:

1. **"exists" was treated as "valid".**  The reuse path only checked
   ``Path.is_file()``/``stat().st_size``, which by construction cannot tell a
   complete file from an interrupted write (see the root-cause document, §2.2:
   a broken file has the payload size *plus* normal metadata slack).  Every
   ``load_*``/``ensure_*`` here opens the file, classifies its shape, checks its
   dtype and attributes and performs a real element read before the file is
   accepted.
2. **The writer was not durable.**  ``temp + os.replace`` without ``fsync``
   lets the rename reach disk before the data does.  Writing here is
   ``temp -> flush -> verify -> fsync(file) -> rename -> fsync(directory)``, so
   a reader either sees the previous complete file or the new complete file.
3. **The writer never read its own output.**  ``_publish`` re-opens the
   temporary file, re-validates shape/dtype/attributes and compares the corners
   of the payload against the array it was asked to store *before* publishing
   it.  A mismatch never reaches the final path.

A broken file is never silently reused: :func:`ensure_gauge` /
:func:`ensure_source` raise :class:`ArtifactCorrupt` (or
:class:`ArtifactSchemaError`) unless the caller passes ``regenerate=True``, and
only then is the bad file - and nothing else - removed and re-created.

Storage contract
----------------
The in-memory layout is the canonical PyQCU layout, space-time last and in
``x, y, z, t`` order (``AGENTS.md``, "张量布局"):

``gauge``   ``[3, 3, 4, x, y, z, t]`` (full) or
            ``[2, 3, 3, 4, x, y, z, t/2]`` (parity-compressed)
``source``  ``[4, 3, x, y, z, t]`` (full) or
            ``[2, 4, 3, x, y, z, t/2]`` (parity-compressed)

The array is stored verbatim as the HDF5 dataset ``data``; the layout is
recorded in a ``layout`` attribute (plus ``role``/``dtype``/``shape``/
``lattice``/``compact``) so a reader never has to infer the meaning from the
file name.  A parity-compressed file **must** carry that attribute - its
``t/2`` axis is otherwise ambiguous - and a full file may omit it for
compatibility with the legacy writers.

No ``zyxt`` permutation happens here, on purpose.  The repository's ``zyxt``
and parity helpers (``pyqcu/tools/_define.py::ccdxyzt2ccdptzyx`` /
``scxyzt2psctzyx``, ``pyqcu/tools/_io.py::gridoooxyzt2hdf5oooxyzt``) are core,
torch-based code that this port must not import; converting "in the style of"
those helpers without using them would create a second, subtly different
on-disk convention.  Files written here are plain canonical ``xyzt`` with an
explicit ``layout`` attribute, which is exactly what the reference driver's
readers (``h5_readable``/``prewarm_h5``) and ``pyqcu/tools/_io.py``'s
``save_tensor_h5`` consume.

File format
-----------
``data``        the payload (complex64 or complex128 only).
``format``      ``"pyqcu.simple.v1"`` - the same tag the reference driver writes.
``layout``      canonical layout string (see above).
``role``        ``"gauge"`` or ``"source"``.
``dtype``       dataset dtype as text; a file whose attribute disagrees with its
                dataset is rejected (the 2026-11 matrix recorded c64/c128 from
                the *file name*, which was wrong for every production gauge).
``shape``       the dataset shape as a list (self-describing, no chunk guessing).
``lattice``     ``[x, y, z, t]``; for a compressed file ``t`` is the full extent.
``compact``     ``True`` for the parity-compressed layouts.
``created_at``  UTC ISO-8601 timestamp of the write.
``writer``      ``"pyqcu.test.io"``.

Import rules
------------
Standard library only at module level.  ``h5py`` and ``numpy`` are imported on
first use, so ``python -m pyqcu.test --list`` never pays for them and this
module can be imported on a machine without HDF5.  ``torch``, ``mpi4py`` and
``pytest`` are never imported at all; a torch tensor passed to ``save_*`` is
converted through its own ``detach().cpu().numpy()`` without importing torch.

Public surface
--------------
write   :func:`save_gauge`, :func:`save_source`
read    :func:`load_gauge`, :func:`load_source`, :func:`describe`,
        :func:`validate_gauge`, :func:`validate_source`, :func:`is_readable`
reuse   :func:`ensure_gauge`, :func:`ensure_source`
shapes  :func:`gauge_shape`, :func:`source_shape`,
        :func:`compact_gauge_shape`, :func:`compact_source_shape`,
        :func:`gauge_layout`, :func:`source_layout`
files   :func:`sha256_file`, :func:`file_identity`, :func:`atomic_write_text`,
        :func:`atomic_write_json`
errors  :class:`ArtifactError` and its subclasses
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, NamedTuple, Sequence

__all__ = [
    "ArtifactCorrupt",
    "ArtifactError",
    "ArtifactMissing",
    "ArtifactSchemaError",
    "COMPACT_GAUGE_LAYOUT",
    "COMPACT_GAUGE_PREFIX",
    "COMPACT_SOURCE_LAYOUT",
    "COMPACT_SOURCE_PREFIX",
    "COMPLEX_DTYPES",
    "DATASET",
    "FORMAT",
    "GAUGE_LAYOUT",
    "GAUGE_PREFIX",
    "H5pyUnavailable",
    "HASH_CHUNK_BYTES",
    "SOURCE_LAYOUT",
    "SOURCE_PREFIX",
    "atomic_write_json",
    "atomic_write_text",
    "compact_gauge_shape",
    "compact_source_shape",
    "describe",
    "ensure_gauge",
    "ensure_source",
    "file_identity",
    "gauge_layout",
    "gauge_shape",
    "h5py_available",
    "h5py_version",
    "is_readable",
    "load_gauge",
    "load_source",
    "save_gauge",
    "save_source",
    "sha256_file",
    "source_layout",
    "source_shape",
    "validate_gauge",
    "validate_source",
]


# ---------------------------------------------------------------------------
# constants: dataset name, format tag and the two canonical layouts per role
# ---------------------------------------------------------------------------
#: Name of the payload dataset, identical to the reference driver and to
#: ``pyqcu/tools/_io.py``.
DATASET = "data"

#: Format tag written by ``pyqcu/testing/kupl/test_kunpeng_all.py``; a file that
#: carries a *different* tag is not a gauge/source file this module understands.
FORMAT = "pyqcu.simple.v1"

#: Full gauge layout: ``[color, color, direction, x, y, z, t]``.
GAUGE_LAYOUT = "[color,color,direction,x,y,z,t]"
GAUGE_PREFIX = (3, 3, 4)

#: Parity-compressed gauge layout used by the KUPL C++ backend.
COMPACT_GAUGE_LAYOUT = "[parity,color,color,direction,x,y,z,t/2]"
COMPACT_GAUGE_PREFIX = (2, 3, 3, 4)

#: Full source layout: ``[spin, color, x, y, z, t]``.
SOURCE_LAYOUT = "[spin,color,x,y,z,t]"
SOURCE_PREFIX = (4, 3)

#: Parity-compressed source layout used by the KUPL C++ backend.
COMPACT_SOURCE_LAYOUT = "[parity,spin,color,x,y,z,t/2]"
COMPACT_SOURCE_PREFIX = (2, 4, 3)

#: The only dtypes a gauge/source artifact may hold.  ``complex256``
#: (``clongdouble``) is excluded: it has no portable HDF5 representation and no
#: backend consumes it.
COMPLEX_DTYPES = ("complex64", "complex128")

#: ``dtype=`` accepted by :func:`load_gauge`/:func:`load_source`.
_DTYPE_ALIASES = {
    "c64": "complex64",
    "c128": "complex128",
    "complex64": "complex64",
    "complex128": "complex128",
    "csingle": "complex64",
    "cdouble": "complex128",
    "single_complex": "complex64",
    "double_complex": "complex128",
}

#: Hash block size used by :func:`sha256_file` (same 8 MiB as the reference
#: driver's ``file_sha256``).
HASH_CHUNK_BYTES = 8 * 1024 * 1024

#: Per-process counter that keeps concurrent writers to one destination from
#: sharing an intermediate file name.
_TEMP_IDS = count()


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ArtifactError(Exception):
    """Base class for every failure this module reports.

    Callers that only care "this artifact is unusable" catch this; callers that
    need the distinction catch a subclass.  The messages always name the path
    and the failed check, because the 2026-11 incident was made much harder to
    diagnose by ledgers that only recorded ``no report.json``.
    """


class H5pyUnavailable(ArtifactError, RuntimeError):
    """``h5py`` cannot be imported, so no HDF5 artifact can be read or written.

    A capability problem, not a data problem: it says nothing about the file the
    caller asked for, so a runner should report it as a skip/environment error
    rather than as a failed artifact.
    """


class ArtifactMissing(ArtifactError, FileNotFoundError):
    """The artifact does not exist (or is not a regular file)."""


class ArtifactCorrupt(ArtifactError, OSError):
    """The artifact exists but cannot be read as HDF5.

    Raised for a zero-length file, an unopenable file (``bad object header
    version number``, truncated metadata, wrong container) and for a data
    region that cannot be read.  The file is *left alone*: deleting evidence is
    the caller's decision, taken by passing ``regenerate=True`` to
    :func:`ensure_gauge`/:func:`ensure_source`.
    """


class ArtifactSchemaError(ArtifactError, ValueError):
    """The file opens but does not honour the gauge/source contract.

    Wrong ndim/prefix, a non-positive space-time extent, a dtype outside
    :data:`COMPLEX_DTYPES`, an unknown ``format`` tag, a ``layout`` attribute
    that contradicts the shape, a parity-compressed shape without the explicit
    ``layout`` attribute, a ``dtype``/``role`` attribute that disagrees with the
    payload, or a read-back mismatch after a write.
    """


# ---------------------------------------------------------------------------
# layout table
# ---------------------------------------------------------------------------
class _Layout(NamedTuple):
    """One accepted shape/layout for a role."""

    text: str
    prefix: tuple[int, ...]
    compact: bool

    @property
    def ndim(self) -> int:
        """Number of axes: the role prefix plus the four space-time axes."""
        return len(self.prefix) + 4


_LAYOUTS: dict[str, tuple[_Layout, _Layout]] = {
    "gauge": (
        _Layout(GAUGE_LAYOUT, GAUGE_PREFIX, compact=False),
        _Layout(COMPACT_GAUGE_LAYOUT, COMPACT_GAUGE_PREFIX, compact=True),
    ),
    "source": (
        _Layout(SOURCE_LAYOUT, SOURCE_PREFIX, compact=False),
        _Layout(COMPACT_SOURCE_LAYOUT, COMPACT_SOURCE_PREFIX, compact=True),
    ),
}

_ROLES = ("gauge", "source")


# ---------------------------------------------------------------------------
# lazy imports
# ---------------------------------------------------------------------------
_NUMPY: Any = None
_H5PY: Any = None


def _numpy() -> Any:
    """Return the ``numpy`` module, importing it on first use."""
    global _NUMPY
    if _NUMPY is None:
        import numpy

        _NUMPY = numpy
    return _NUMPY


def _h5py() -> Any:
    """Return the ``h5py`` module, importing it on first use.

    Nothing in this module touches ``h5py`` at import time, so the harness'
    stdlib-only rule holds and ``--list`` works where HDF5 is absent.
    """
    global _H5PY
    if _H5PY is None:
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - environment dependent
            raise H5pyUnavailable(
                "h5py is required to read or write HDF5 gauge/source "
                "artifacts; install it with 'pip install h5py'"
            ) from error
        _H5PY = h5py
    return _H5PY


def h5py_available() -> bool:
    """True when ``h5py`` can be imported (does not raise)."""
    try:
        _h5py()
    except H5pyUnavailable:
        return False
    return True


def h5py_version() -> str | None:
    """``h5py`` version string, or ``None`` when unavailable (never raises)."""
    try:
        return str(_h5py().__version__)
    except H5pyUnavailable:
        return None


# ---------------------------------------------------------------------------
# filesystem helpers: paths, fsync, atomic publication
# ---------------------------------------------------------------------------
def _as_path(value: str | os.PathLike[str]) -> Path:
    """Normalise ``value`` to an expanded :class:`~pathlib.Path`."""
    return Path(os.fspath(value)).expanduser()


def _existing_file(value: str | os.PathLike[str]) -> Path:
    """Return ``value`` as a path to an existing regular file.

    Raises :class:`ArtifactMissing` for "no such file", "is a directory" and for
    a dangling symlink, so every ``load_*``/``ensure_*`` failure is one of this
    module's exceptions.
    """
    path = _as_path(value)
    if not path.is_file():
        reason = "is a directory" if path.is_dir() else "does not exist"
        raise ArtifactMissing(f"{path} {reason} (not a regular file)")
    return path


def _unlink(path: Path) -> bool:
    """Remove ``path`` if it is a file or symlink; never a directory.

    Returns ``True`` when something was removed.  Missing files and races are
    not errors: the caller asked for "this path must not exist any more".
    """
    try:
        if path.is_dir() and not path.is_symlink():
            return False
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _temp_path(target: Path) -> Path:
    """A hidden, per-process, per-call temporary name next to ``target``.

    Same directory as the destination, so the final ``os.replace`` is a rename
    on one filesystem and cannot expose a half-written file under the real name.
    """
    return target.with_name(f".{target.name}.{os.getpid()}-{next(_TEMP_IDS)}.tmp")


def _fsync_file(path: Path) -> None:
    """Flush ``path``'s data and metadata to stable storage.

    Called after the HDF5 handle is closed: at that point every write lives in
    the page cache, and fsync on any descriptor of the inode pushes it out.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry so a rename survives a crash.

    Best effort: some filesystems (and Windows) refuse to fsync a directory;
    the rename is still atomic there, it is only the durability barrier that is
    weaker.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)


def _json_default(value: Any) -> Any:
    """Encode the few non-JSON values metadata actually contains."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def atomic_write_text(path: str | os.PathLike[str], text: str, *,
                      encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically and durably.

    ``temp -> write -> flush -> fsync -> os.replace -> fsync(dir)``.  A reader
    sees the previous content or the complete new content, never a truncation,
    and the rename is on disk before the call returns.
    """
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temp_path(target)
    try:
        with temporary.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        _unlink(temporary)
        raise
    _fsync_directory(target.parent)
    return target


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *,
                      indent: int = 2) -> Path:
    """Write ``payload`` as JSON with :func:`atomic_write_text`.

    ``NaN``/``Infinity`` are rejected (``allow_nan=False``) because they are not
    JSON: a manifest that cannot be parsed by a strict reader is worse than a
    loud failure.  Non-finite diagnostics belong in a float column of the
    ledger, not in JSON metadata.
    """
    text = json.dumps(payload, indent=indent, sort_keys=True,
                      ensure_ascii=False, allow_nan=False,
                      default=_json_default) + "\n"
    return atomic_write_text(path, text)


def sha256_file(path: str | os.PathLike[str], *,
                chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """Stream ``path`` through SHA-256 and return the hex digest.

    Streamed in ``chunk_size`` blocks (8 MiB by default) so hashing a 1.2 GB
    production gauge does not allocate 1.2 GB.
    """
    target = _existing_file(path)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | os.PathLike[str], *,
                  with_sha256: bool = True) -> dict[str, Any]:
    """Return a JSON-safe identity record for one file.

    The record answers "which bytes did this run consume?", the question the
    2026-11 ledger could not answer because it inferred ``c64``/``c128`` from
    file names (root-cause document §4.4, lesson 8).  Fields:

    ``path``/``name``   absolute path and file name
    ``size``            ``st_size`` in bytes
    ``mtime_ns``        ``st_mtime_ns`` (sub-second, so a rewrite is visible)
    ``inode``/``device`` ``st_ino``/``st_dev`` - catches "same name, new file"
                        and a bind-mount/copy masquerading as the same path
    ``stat_key``        ``"<path>:<size>:<mtime_ns>"``, the cache key the
                        reference driver's ``HashCache`` uses
    ``sha256``          content digest, or ``None`` with ``with_sha256=False``
                        (a 1.2 GB hash costs a second or two; skip it when only
                        the stat identity is needed)
    """
    target = _existing_file(path)
    stat = target.stat()
    resolved = target.resolve()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "name": target.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
        "stat_key": f"{resolved}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}",
        "sha256": None,
    }
    if with_sha256:
        identity["sha256"] = sha256_file(target)
    return identity


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------
def _lattice4(lattice: Sequence[int]) -> tuple[int, int, int, int]:
    """Validate a ``(x, y, z, t)`` lattice and return it as four ints."""
    values = tuple(int(value) for value in lattice)
    if len(values) != 4:
        raise ValueError(f"lattice must have 4 extents, got {lattice!r}")
    if any(value <= 0 for value in values):
        raise ValueError(f"lattice extents must be positive, got {values}")
    return values  # type: ignore[return-value]


def gauge_shape(lattice: Sequence[int]) -> tuple[int, int, int, int, int, int, int]:
    """Full gauge shape ``[3, 3, 4, x, y, z, t]`` for ``lattice=(x, y, z, t)``."""
    return (*GAUGE_PREFIX, *_lattice4(lattice))


def source_shape(lattice: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    """Full source shape ``[4, 3, x, y, z, t]`` for ``lattice=(x, y, z, t)``."""
    return (*SOURCE_PREFIX, *_lattice4(lattice))


def compact_gauge_shape(
        lattice: Sequence[int]) -> tuple[int, int, int, int, int, int, int, int]:
    """Parity-compressed gauge shape ``[2, 3, 3, 4, x, y, z, t/2]``."""
    x, y, z, t = _lattice4(lattice)
    if t % 2:
        raise ValueError(f"a parity-compressed gauge needs an even t, got {t}")
    return (*COMPACT_GAUGE_PREFIX, x, y, z, t // 2)


def compact_source_shape(
        lattice: Sequence[int]) -> tuple[int, int, int, int, int, int, int]:
    """Parity-compressed source shape ``[2, 4, 3, x, y, z, t/2]``."""
    x, y, z, t = _lattice4(lattice)
    if t % 2:
        raise ValueError(f"a parity-compressed source needs an even t, got {t}")
    return (*COMPACT_SOURCE_PREFIX, x, y, z, t // 2)


def _classify(role: str, shape: Sequence[int]) -> tuple[_Layout, tuple[int, ...]]:
    """Match ``shape`` against the accepted layouts of ``role``.

    Returns ``(layout, spacetime_extents)``.  The four space-time extents are
    always returned as the *full* ``(x, y, z, t)`` - a compressed file's last
    axis is doubled, because ``t/2`` is a storage detail, not a lattice.
    """
    table = _LAYOUTS.get(role)
    if table is None:
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
    observed = tuple(int(value) for value in shape)
    for layout in table:
        if len(observed) != layout.ndim:
            continue
        if observed[:len(layout.prefix)] != layout.prefix:
            continue
        tail = observed[len(layout.prefix):]
        if any(value <= 0 for value in tail):
            raise ArtifactSchemaError(
                f"{role} shape {observed} ({layout.text}) has a non-positive "
                f"space-time extent")
        extents = list(tail)
        if layout.compact:
            extents[3] *= 2
        return layout, tuple(extents)
    expected = "; ".join(
        f"{layout.ndim}D {layout.text}" for layout in table)
    raise ArtifactSchemaError(
        f"{role} shape {observed} is not one of the accepted layouts "
        f"({expected})")


def gauge_layout(value: Any) -> str:
    """Layout string of a gauge-shaped array (or shape) - ``full``/``compact``."""
    return _classify("gauge", _shape_of(value))[0].text


def source_layout(value: Any) -> str:
    """Layout string of a source-shaped array (or shape) - ``full``/``compact``."""
    return _classify("source", _shape_of(value))[0].text


def _shape_of(value: Any) -> tuple[int, ...]:
    """Shape of an array-like object or of a plain shape sequence."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(axis) for axis in shape)
    if isinstance(value, (tuple, list)):
        return tuple(int(axis) for axis in value)
    raise ArtifactSchemaError(
        f"expected an array or a shape tuple, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# attribute encoding
# ---------------------------------------------------------------------------
_SCALAR_ATTR_TYPES = (str, bytes, bool, int, float)
_RESERVED_ATTRS = ("format", "layout", "role", "dtype", "shape", "lattice",
                   "compact", "created_at", "writer")


def _encode_attr(key: str, value: Any, *, nested: bool = False) -> Any:
    """Convert one caller-supplied attribute into an HDF5-acceptable value.

    HDF5 attributes are 1-D scalars or arrays of scalars; there is no complex
    attribute type, no nested list and no ``None``.  Anything outside that is
    rejected with the offending key, instead of being silently stringified -
    a reader must never have to guess whether ``"1.0"`` means the number or the
    string.
    """
    if isinstance(value, _SCALAR_ATTR_TYPES):
        return value
    if isinstance(value, complex):
        raise ArtifactError(
            f"attribute {key!r} is complex and HDF5 attributes have no complex "
            f"type; store the real and imaginary parts under two keys")
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if converted is not value:
            return _encode_attr(key, converted, nested=nested)
    if isinstance(value, (list, tuple)):
        if nested:
            raise ArtifactError(
                f"attribute {key!r} is nested; HDF5 attributes must be 1-D")
        return [_encode_attr(key, item, nested=True) for item in value]
    if hasattr(value, "item"):
        return _encode_attr(key, value.item(), nested=nested)
    if value is None:
        raise ArtifactError(
            f"attribute {key!r} is None; omit the key instead of storing a "
            f"non-value")
    raise ArtifactError(
        f"attribute {key!r} has unsupported type {type(value).__name__}")


def _decode_attr(value: Any) -> Any:
    """Convert one h5py attribute value into a JSON-safe Python value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, bytes):
            return converted.decode("utf-8", "replace")
        if isinstance(converted, list):
            return [item.decode("utf-8", "replace")
                    if isinstance(item, bytes) else item for item in converted]
        return converted
    if hasattr(value, "item"):
        return value.item()
    return value


def _decode_attrs(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a whole h5py attribute mapping."""
    return {str(key): _decode_attr(value) for key, value in attributes.items()}


def _normalize_layout_text(value: Any) -> str:
    """Canonical form of a layout string: no whitespace, lower case."""
    return "".join(str(value).split()).lower()


# ---------------------------------------------------------------------------
# validation: open, classify, check attributes, touch real data
# ---------------------------------------------------------------------------
def _open_validated(path: str | os.PathLike[str], role: str,
                    dataset: str = DATASET) -> tuple[Any, Any, dict[str, Any]]:
    """Open ``path`` and prove it is a usable ``role`` artifact.

    Returns ``(handle, node, record)``; the caller owns the handle and must
    close it.  Every check the root-cause document asks for happens here:

    * the file exists, is regular and is not zero-length;
    * ``h5py`` can open it (the ``bad object header`` case fails here);
    * the ``data`` dataset exists and is a dataset;
    * the shape is one of the role's two canonical layouts;
    * the dtype is ``complex64``/``complex128``;
    * ``format``/``layout``/``dtype``/``role`` attributes, when present, agree
      with the payload, and a parity-compressed file carries ``layout``;
    * a real element at the first and last corner can be read (metadata-only
      checks cannot see a truncated data region).
    """
    target = _existing_file(path)
    stat = target.stat()
    if stat.st_size <= 0:
        raise ArtifactCorrupt(
            f"{target} is 0 bytes: an interrupted write, not a valid HDF5 file")
    h5py = _h5py()
    try:
        handle = h5py.File(str(target), "r")
    except Exception as error:  # h5py raises OSError/ValueError/RuntimeError
        raise ArtifactCorrupt(
            f"{target} cannot be opened as HDF5 "
            f"({type(error).__name__}: {error})") from error
    try:
        if dataset not in handle:
            raise ArtifactSchemaError(
                f"{target} has no {dataset!r} dataset "
                f"(root keys: {sorted(handle.keys())})")
        node = handle[dataset]
        if not isinstance(node, h5py.Dataset):
            raise ArtifactSchemaError(
                f"{target}: {dataset!r} is a {type(node).__name__}, not a dataset")
        shape = tuple(int(axis) for axis in node.shape)
        dtype_name = str(node.dtype)
        layout, extents = _classify(role, shape)
        if dtype_name not in COMPLEX_DTYPES:
            raise ArtifactSchemaError(
                f"{target}: {role} dtype is {dtype_name}, expected one of "
                f"{COMPLEX_DTYPES}")
        attributes = _decode_attrs(handle.attrs)
        _check_attrs(target, role, layout, shape, dtype_name, attributes)
        try:
            _read_corners(node)
        except Exception as error:
            raise ArtifactCorrupt(
                f"{target}: the {role} payload cannot be read "
                f"({type(error).__name__}: {error})") from error
        record: dict[str, Any] = {
            "path": str(target.resolve()),
            "name": target.name,
            "role": role,
            "dataset": dataset,
            "shape": [int(axis) for axis in shape],
            "dtype": dtype_name,
            "layout": layout.text,
            "compact": bool(layout.compact),
            "lattice": [int(value) for value in extents],
            "bytes": int(node.nbytes),
            "file_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "attrs": attributes,
        }
        return handle, node, record
    except BaseException:
        handle.close()
        raise


def _check_attrs(path: Path, role: str, layout: _Layout, shape: Sequence[int],
                 dtype_name: str, attributes: Mapping[str, Any]) -> None:
    """Cross-check the descriptive attributes against the payload."""
    declared = attributes.get("format")
    if declared is not None and str(declared) != FORMAT:
        raise ArtifactSchemaError(
            f"{path}: format is {declared!r}, this module only reads {FORMAT!r}")
    declared_role = attributes.get("role")
    if declared_role is not None and str(declared_role) != role:
        raise ArtifactSchemaError(
            f"{path}: role attribute is {declared_role!r} but it is being read "
            f"as {role!r}")
    declared_dtype = attributes.get("dtype")
    if declared_dtype is not None and str(declared_dtype) != dtype_name:
        raise ArtifactSchemaError(
            f"{path}: dtype attribute is {declared_dtype!r} but the dataset is "
            f"{dtype_name!r} - the file name is not evidence, the dataset is")
    declared_layout = attributes.get("layout")
    if declared_layout is None:
        if layout.compact:
            raise ArtifactSchemaError(
                f"{path}: a parity-compressed {role} {layout.text} must carry an "
                f"explicit 'layout' attribute; its t/2 axis is ambiguous "
                f"without it")
        return
    normalized = _normalize_layout_text(declared_layout)
    if normalized != _normalize_layout_text(layout.text):
        raise ArtifactSchemaError(
            f"{path}: layout attribute {declared_layout!r} contradicts the "
            f"dataset shape {tuple(shape)} ({layout.text})")
    declared_shape = attributes.get("shape")
    if declared_shape is not None:
        try:
            recorded = tuple(int(value) for value in declared_shape)
        except (TypeError, ValueError) as error:
            raise ArtifactSchemaError(
                f"{path}: shape attribute {declared_shape!r} is not a list of "
                f"integers") from error
        if recorded != tuple(int(axis) for axis in shape):
            raise ArtifactSchemaError(
                f"{path}: shape attribute {recorded} contradicts the dataset "
                f"shape {tuple(shape)}")


def _corners(ndim: int) -> tuple[tuple[int, ...], ...]:
    """Index tuples of the first and the last element of an ``ndim`` dataset."""
    if ndim <= 0:
        return ((),)
    first = tuple(0 for _ in range(ndim))
    last = tuple(-1 for _ in range(ndim))
    return (first,) if first == last else (first, last)


def _read_corners(node: Any) -> None:
    """Touch the first and last element of ``node`` (a real read, not metadata)."""
    for index in _corners(int(node.ndim)):
        node[index]


def describe(path: str | os.PathLike[str], *, role: str | None = None,
             dataset: str = DATASET) -> dict[str, Any]:
    """Validate ``path`` and return its self-description.

    ``role=None`` tries ``gauge`` then ``source``; the two prefixes are
    distinct, so there is no ambiguity.  The returned mapping is JSON-safe and
    contains ``path``, ``name``, ``role``, ``dataset``, ``shape``, ``dtype``,
    ``layout``, ``compact``, ``lattice`` (full ``x, y, z, t``), ``bytes``
    (payload size), ``file_bytes`` (``st_size``), ``mtime_ns`` and the decoded
    ``attrs``.

    Raises :class:`ArtifactMissing`, :class:`ArtifactCorrupt` or
    :class:`ArtifactSchemaError`; a returned record means the file passed every
    gate.
    """
    if role is not None:
        return _describe_as(path, _check_role(role), dataset)
    failures: list[str] = []
    for candidate in _ROLES:
        try:
            return _describe_as(path, candidate, dataset)
        except ArtifactSchemaError as error:
            failures.append(f"{candidate}: {error}")
    raise ArtifactSchemaError(
        f"{path} is neither a gauge nor a source artifact "
        f"({'; '.join(failures)})")


def _describe_as(path: str | os.PathLike[str], role: str,
                 dataset: str) -> dict[str, Any]:
    """Open, validate and close ``path`` as ``role``; return its record."""
    handle, _, record = _open_validated(path, role, dataset)
    handle.close()
    return record


def validate_gauge(path: str | os.PathLike[str], *,
                   dataset: str = DATASET) -> dict[str, Any]:
    """Return :func:`describe` of ``path`` as a gauge artifact (raises on failure)."""
    return describe(path, role="gauge", dataset=dataset)


def validate_source(path: str | os.PathLike[str], *,
                    dataset: str = DATASET) -> dict[str, Any]:
    """Return :func:`describe` of ``path`` as a source artifact (raises on failure)."""
    return describe(path, role="source", dataset=dataset)


def _check_role(role: str) -> str:
    """Reject a role outside ``("gauge", "source")`` before touching the disk."""
    if role not in _ROLES:
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
    return role


def is_readable(path: str | os.PathLike[str], *, role: str | None = None,
                dataset: str = DATASET) -> tuple[bool, str]:
    """Cheap gate: ``(True, "")`` or ``(False, reason)`` - never raises.

    The reference driver's ``h5_readable`` shape, made strict: the reason is the
    exception message, so a ledger can record *why* a candidate was skipped
    (``bad object header``, ``no 'data' dataset``, ``dtype is float64``) instead
    of the useless ``no report.json`` of the 2026-11 rows.
    """
    try:
        describe(path, role=role, dataset=dataset)
    except ArtifactError as error:
        return False, str(error) or type(error).__name__
    except OSError as error:
        return False, f"{type(error).__name__}: {error}"
    return True, ""


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _read(path: str | os.PathLike[str], role: str, dtype: str | None,
          dataset: str) -> Any:
    """Read a validated artifact and return an in-memory ``numpy`` array."""
    numpy = _numpy()
    handle, node, _ = _open_validated(path, role, dataset)
    try:
        array = node[...]
    except Exception as error:
        raise ArtifactCorrupt(
            f"{path}: the {role} payload cannot be read "
            f"({type(error).__name__}: {error})") from error
    finally:
        handle.close()
    if dtype is not None:
        wanted = _normalize_dtype(dtype)
        if str(array.dtype) != wanted:
            array = array.astype(numpy.dtype(wanted), copy=False)
    return numpy.ascontiguousarray(array)


def _normalize_dtype(value: Any) -> str:
    """Map ``c64``/``c128``/``complex64``/``complex128`` to a numpy dtype name."""
    text = str(value).strip().lower()
    normalized = _DTYPE_ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            f"unsupported complex dtype {value!r}; expected one of "
            f"{sorted(_DTYPE_ALIASES)} (gauge/source artifacts are complex)")
    return normalized


def load_gauge(path: str | os.PathLike[str], *, dtype: str | None = None,
               dataset: str = DATASET) -> Any:
    """Read a gauge artifact as ``[3, 3, 4, x, y, z, t]`` (or compressed).

    The file is validated first (:func:`describe` semantics); ``dtype`` may
    request ``c64``/``c128`` and is applied as a cast after the read.  The
    returned array is a fresh, C-contiguous ``numpy`` array, so the caller may
    mutate it freely.
    """
    return _read(path, "gauge", dtype, dataset)


def load_source(path: str | os.PathLike[str], *, dtype: str | None = None,
                dataset: str = DATASET) -> Any:
    """Read a source artifact as ``[4, 3, x, y, z, t]`` (or compressed)."""
    return _read(path, "source", dtype, dataset)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _as_array(value: Any, role: str) -> tuple[Any, _Layout, tuple[int, ...]]:
    """Validate ``value`` as a ``role`` array and return ``(array, layout, t)``.

    Accepts numpy arrays, shape-compatible sequences and torch tensors: a torch
    tensor is unwrapped through its own methods, so torch is never imported
    here.
    """
    numpy = _numpy()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    array = numpy.asarray(value)
    dtype_name = str(array.dtype)
    if dtype_name not in COMPLEX_DTYPES:
        raise ArtifactSchemaError(
            f"{role} payload dtype is {dtype_name}, expected one of "
            f"{COMPLEX_DTYPES}; cast explicitly so the file records the "
            f"precision it really holds")
    layout, extents = _classify(role, array.shape)
    if not array.flags["C_CONTIGUOUS"]:
        array = numpy.ascontiguousarray(array)
    return array, layout, extents


def _build_attrs(role: str, layout: _Layout, array: Any,
                 extents: Sequence[int],
                 extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compose the attribute mapping written next to the payload."""
    attributes: dict[str, Any] = {
        "format": FORMAT,
        "layout": layout.text,
        "role": role,
        "dtype": str(array.dtype),
        "shape": [int(axis) for axis in array.shape],
        "lattice": [int(value) for value in extents],
        "compact": bool(layout.compact),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "writer": "pyqcu.test.io",
    }
    for key, value in (extra or {}).items():
        name = str(key)
        if name in _RESERVED_ATTRS:
            raise ArtifactError(
                f"attrs may not override the reserved attribute {name!r}; it is "
                f"derived from the payload")
        attributes[name] = _encode_attr(name, value)
    return attributes


def save_gauge(path: str | os.PathLike[str], array: Any, *,
               attrs: Mapping[str, Any] | None = None,
               dataset: str = DATASET) -> Path:
    """Atomically write a gauge artifact.

    ``array`` must be ``[3, 3, 4, x, y, z, t]`` (full) or
    ``[2, 3, 3, 4, x, y, z, t/2]`` (parity-compressed, recorded in ``layout``),
    ``complex64`` or ``complex128``.  ``attrs`` adds descriptive attributes
    (``generator``, ``seed``, ``mass``, ...); reserved keys such as ``layout``
    are derived from the payload and cannot be overridden.

    The write is ``temp -> flush -> verify -> fsync -> rename -> fsync(dir)``,
    so the destination never contains a partial file and the returned path is a
    file that has already been read back successfully.  An existing file at
    ``path`` is replaced atomically by that rename; its old content is never
    truncated in place, so a reader racing the write sees one of the two
    complete versions.
    """
    return _publish(_as_path(path), array, "gauge", attrs, dataset)


def save_source(path: str | os.PathLike[str], array: Any, *,
                attrs: Mapping[str, Any] | None = None,
                dataset: str = DATASET) -> Path:
    """Atomically write a source artifact (``[4, 3, x, y, z, t]`` or compressed)."""
    return _publish(_as_path(path), array, "source", attrs, dataset)


def _publish(target: Path, array: Any, role: str,
             attrs: Mapping[str, Any] | None, dataset: str) -> Path:
    """Publish ``array`` at ``target`` with no window in which it is half-written."""
    if target.is_dir() and not target.is_symlink():
        raise ArtifactError(f"{target} is a directory, not an artifact path")
    h5py = _h5py()
    numpy = _numpy()
    payload, layout, extents = _as_array(array, role)
    attributes = _build_attrs(role, layout, payload, extents, attrs)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temp_path(target)
    try:
        with h5py.File(str(temporary), "w") as handle:
            handle.create_dataset(dataset, data=payload)
            for key, value in attributes.items():
                handle.attrs[key] = value
            handle.flush()
        _verify_written(temporary, role, payload, dataset, numpy)
        _fsync_file(temporary)
        os.replace(temporary, target)
    except BaseException:
        _unlink(temporary)
        raise
    _fsync_directory(target.parent)
    return target


def _verify_written(path: Path, role: str, payload: Any, dataset: str,
                    numpy: Any) -> None:
    """Re-open a freshly written file and prove it holds ``payload``.

    The root-cause document (§4.1.2) asks for shape/dtype/size checks before the
    rename; the corner comparison is the cheap addition that catches a data
    region which was never written, not just a bad header.
    """
    handle, node, record = _open_validated(path, role, dataset)
    try:
        expected_shape = [int(axis) for axis in payload.shape]
        if record["shape"] != expected_shape:
            raise ArtifactSchemaError(
                f"{path}: wrote shape {expected_shape} but read back "
                f"{record['shape']}")
        if record["dtype"] != str(payload.dtype):
            raise ArtifactSchemaError(
                f"{path}: wrote dtype {payload.dtype} but read back "
                f"{record['dtype']}")
        for index in _corners(int(node.ndim)):
            if not bool(numpy.array_equal(node[index], payload[index])):
                raise ArtifactSchemaError(
                    f"{path}: read-back mismatch at index {index}")
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# reuse / repair
# ---------------------------------------------------------------------------
def ensure_gauge(path: str | os.PathLike[str],
                 factory: Callable[..., Any] | None = None, *,
                 regenerate: bool = False,
                 attrs: Mapping[str, Any] | None = None,
                 dataset: str = DATASET,
                 report: MutableMapping[str, Any] | None = None) -> Path:
    """Return a validated gauge file, creating it only when necessary.

    ``factory`` produces the array (no argument: ``factory() -> array``) or
    writes the file itself (one required argument: ``factory(path) -> path |
    None``); bind extra parameters with :func:`functools.partial`.  The
    factory's output is validated again, so a broken producer cannot publish.
    ``attrs`` describes the file written from a returned array; a writer factory
    describes its own output.

    A file that fails validation is **never** reused: the exception propagates
    with a hint, and only an explicit ``regenerate=True`` removes that one file
    (nothing else) and re-creates it.  A file that validates is always reused,
    even with ``regenerate=True`` - deleting good data is never automatic.

    ``report``, when given, is filled with ``role``, ``path``, ``origin``
    (``"reused"``/``"created"``/``"repaired"``), ``regenerated``, ``removed``,
    ``seconds`` and the validated metadata, so a runner can record where its
    input came from without a second stat.
    """
    return _ensure(_as_path(path), "gauge", factory, regenerate=regenerate,
                   attrs=attrs, dataset=dataset, report=report)


def ensure_source(path: str | os.PathLike[str],
                  factory: Callable[..., Any] | None = None, *,
                  regenerate: bool = False,
                  attrs: Mapping[str, Any] | None = None,
                  dataset: str = DATASET,
                  report: MutableMapping[str, Any] | None = None) -> Path:
    """Source counterpart of :func:`ensure_gauge`."""
    return _ensure(_as_path(path), "source", factory, regenerate=regenerate,
                   attrs=attrs, dataset=dataset, report=report)


def _ensure(target: Path, role: str, factory: Callable[..., Any] | None, *,
            regenerate: bool, attrs: Mapping[str, Any] | None,
            dataset: str,
            report: MutableMapping[str, Any] | None) -> Path:
    """Implementation shared by :func:`ensure_gauge`/:func:`ensure_source`."""
    started = time.perf_counter()
    repaired = False
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            raise ArtifactError(
                f"{target} is a directory, not a {role} artifact")
        try:
            handle, _, record = _open_validated(target, role, dataset)
        except (ArtifactCorrupt, ArtifactSchemaError, ArtifactMissing) as failure:
            if not regenerate:
                raise type(failure)(
                    f"{failure} - refusing to reuse an unusable {role}: pass "
                    f"regenerate=True to replace {target}") from failure
            # The handle is already closed by _open_validated's finally block;
            # remove exactly this one file, never a directory or a sibling.
            # A file that vanished in between is not an error.
            _unlink(target)
            repaired = True
        else:
            handle.close()
            _fill_report(report, role, target, record, origin="reused",
                         regenerated=False, removed=False, started=started)
            return target
    if factory is None:
        raise ArtifactMissing(
            f"{target} does not exist and no factory was given to build it")
    target.parent.mkdir(parents=True, exist_ok=True)
    _materialize(factory, target, role, attrs, dataset)
    try:
        handle, _, record = _open_validated(target, role, dataset)
    except ArtifactError as failure:
        _unlink(target)
        raise type(failure)(
            f"{failure} (the {role} file this factory produced was removed)"
        ) from failure
    handle.close()
    _fill_report(report, role, target, record,
                 origin="repaired" if repaired else "created",
                 regenerated=repaired, removed=repaired, started=started)
    return target


def _materialize(factory: Callable[..., Any], target: Path, role: str,
                 attrs: Mapping[str, Any] | None, dataset: str) -> None:
    """Call ``factory`` and make sure ``target`` ends up holding its payload."""
    result = factory(target) if _accepts_path(factory) else factory()
    if result is None or isinstance(result, (str, os.PathLike)):
        produced = target if result is None else _as_path(result)
        if produced.resolve() != target.resolve():
            raise ArtifactError(
                f"factory wrote {produced} but {target} was requested; a "
                f"factory must write exactly the path it is given")
        if not target.is_file():
            raise ArtifactMissing(
                f"factory returned without creating {target}")
        return
    _publish(target, result, role, attrs, dataset)


def _accepts_path(factory: Callable[..., Any]) -> bool:
    """True when ``factory`` wants the destination path as its only argument.

    ``factory()`` (no required positional parameter) is an *array* producer;
    ``factory(path)`` (exactly one) is a *writer*.  A factory with more than one
    required parameter is a usage error: bind the rest with
    :func:`functools.partial` rather than guessing an argument order here.
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # builtins and C callables
        return False
    required = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    if len(required) > 1:
        names = ", ".join(parameter.name for parameter in required)
        raise ArtifactError(
            f"factory {getattr(factory, '__name__', factory)!r} needs "
            f"{len(required)} arguments ({names}); it must take no argument "
            f"(return the array) or exactly one (the destination path) - bind "
            f"the rest with functools.partial")
    return len(required) == 1


def _fill_report(report: MutableMapping[str, Any] | None, role: str,
                 target: Path, record: Mapping[str, Any], *, origin: str,
                 regenerated: bool, removed: bool, started: float) -> None:
    """Record what :func:`_ensure` did, when the caller asked for it."""
    if report is None:
        return
    report.update({
        "role": role,
        "path": str(target),
        "origin": origin,
        "regenerated": bool(regenerated),
        "removed": bool(removed),
        "seconds": float(time.perf_counter() - started),
        "shape": list(record["shape"]),
        "dtype": record["dtype"],
        "layout": record["layout"],
        "compact": bool(record["compact"]),
        "lattice": list(record["lattice"]),
        "bytes": int(record["bytes"]),
    })
