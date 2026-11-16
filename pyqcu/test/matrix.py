"""Declarative parameter-sweep planner for the formal :mod:`pyqcu.test` port.

A run of ``python -m pyqcu.test`` exercises one configuration: one lattice, one
dtype, one backend filter, one device.  A *sweep* asks the more useful question
- "does the same gate still hold at ``16x16x16x16``, in ``c128``, with the other
solver?" - and it must answer it with the same parser, the same validators and
the same report shape as a single run.

This module is the planner, not the executor: it turns a compact spec into an
explicit, bounded, deterministic list of configurations, and renders the
resulting case matrix as JSON-safe rows.  It imports the standard library plus
:mod:`pyqcu.test.policy` and nothing else, so ``python -m pyqcu.test --list``
can describe a sweep on a machine without torch, KUPL or the CUDA bridge.

Public surface
--------------
:data:`AXES`           the fourteen sweepable names, each with an :class:`Axis`
:data:`AXIS_NAMES`     their canonical order (also the expansion order)
:func:`parse_axis`     one axis text -> validated values
:func:`parse_spec`     a whole spec -> ``{axis: values}``, ordered
:func:`expand_matrix`  that mapping -> one dict per combination
:func:`render_matrix`  selected cases + configurations -> JSON-safe rows
:func:`matrix_payload` the ``--sweep --list --json`` document

Spec syntax
-----------
::

    --sweep dtype=c64,c128;lattice=[8,8,8,16];[16,16,16,16]
    --sweep dtype=c64 --sweep backend=torch,kupl
    --sweep 'backend=cpu;device=gpu,cpu;seed=42'

* ``key=value`` groups are separated by ``;`` or by repeating the flag;
* a ``;`` group without ``key=`` continues the previous axis, which is what
  makes ``lattice=[8,8,8,16];[16,16,16,16]`` two lattices instead of two keys;
* scalar axes split their value on ``,`` and whitespace (``device=gpu,cpu``);
* list-valued axes (``lattice``, ``lattice_grid``, ``mg_grid``) take one
  four-dimensional vector per ``;`` group, in every spelling
  :func:`pyqcu.test.policy.parse_int_list` accepts for the CLI flags;
* axis names ignore case and ``-``/``_``, and ``lattice_size``/``backends``
  are accepted as aliases of ``lattice``/``backend``, so a sweep may reuse the
  flag spelling.

Bounds and determinism
----------------------
The expansion is the Cartesian product in :data:`AXIS_NAMES` order, so one spec
always yields the same rows.  Duplicates are removed per axis and per row
(``dtype=c64,complex64`` is one case, not two), and a product above
:data:`MAX_CASES` is refused *before* it is built: a typo must fail with a
count, not allocate a matrix or launch a day of compute.

Rows
----
:func:`render_matrix` reports one row per planned run: ``tag`` plus ``lattice``,
``dtype``, ``backend``, ``device``, ``solver`` and ``dslash`` - the case and the
six values that change what it exercises.  The rest of the configuration
(``seed``, ``mass``, ``csw``, the gauge fields, the grids) is run-level and is
reported once by :meth:`pyqcu.test.common.Test.config`, not repeated per row.

Errors
------
Every rejection is a ``ValueError`` that names the axis and the offending token.
Nothing here raises :class:`pyqcu.test.common.UsageError`, so the module stays
free of the execution core; the CLI can still turn a bad ``--sweep`` into its
usage exit code, because ``UsageError`` is itself a ``ValueError``.

Import rules
------------
``--sweep`` is wired by the CLI, but ``--list`` must stay backend-free.  Nothing
here imports a runner module, torch, numpy, mpi4py, pytest or ``pyqcu.testing``;
the only project import is :mod:`pyqcu.test.policy`, which is plain data and
validators.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import policy

__all__ = [
    "AXES",
    "AXIS_ALIASES",
    "AXIS_NAMES",
    "Axis",
    "MAX_CASES",
    "ROW_FIELDS",
    "defaults",
    "expand_matrix",
    "format_spec",
    "matrix_payload",
    "parse_axis",
    "parse_spec",
    "render_matrix",
    "select_kwargs",
]

#: Hard cap on the number of axis combinations one sweep may expand to.  The
#: expansion is refused *before* it is materialised: a mistyped axis must fail
#: with a count, not allocate a matrix or launch a day of compute.  Sixteen
#: combinations of four axes is already a long run; 512 is 4x the tier-0 matrix
#: squared, which is more than a quick gate should ever ask for.
MAX_CASES = 512

#: The identity of one planned run, in ``--sweep --list --json`` row order.
#: ``tag`` names the case; the other six values are what a sweep varies about
#: it.  Extra fields would make a row a second, divergent copy of the run
#: configuration, so the vocabulary stays closed.
ROW_FIELDS: tuple[str, ...] = (
    "tag", "lattice", "dtype", "backend", "device", "solver", "dslash")

#: Placeholder for "this source has no value for that field" (``None`` is a
#: value in some configurations - an unset ``mass`` - so it cannot be the
#: sentinel).
_MISSING = object()

#: ``Axis.kind`` values.  The structured kinds have a dedicated validator in
#: :mod:`pyqcu.test.policy`; ``float``/``int`` are the two scalar kinds and
#: ``backend`` is the only kind whose single token can expand into several
#: values (``cpu`` -> ``torch``, ``kupl``).
_KINDS: tuple[str, ...] = (
    "lattice", "lattice_grid", "mg_grid", "dtype", "backend", "device",
    "solver", "dslash", "source", "gauge", "float", "int",
)

#: Kinds whose value is one four-dimensional vector, not a list of scalars.
_VECTOR_KINDS = frozenset({"lattice", "lattice_grid", "mg_grid"})

#: Choice kinds -> the :mod:`pyqcu.test.policy` validator for that choice.  A
#: sweep must accept exactly the vocabulary the corresponding flag accepts.
_CHOICES: Mapping[str, Callable[[Any], str]] = MappingProxyType({
    "device": policy.validate_device,
    "solver": policy.validate_solver,
    "dslash": policy.validate_dslash,
    "source": policy.validate_source,
    "gauge": policy.validate_gauge,
})

#: A plain decimal integer; ``float``/``int`` must not silently accept
#: ``"1.5"`` or ``"1e3"`` where the flag takes a count or a seed.
_INTEGER = re.compile(r"^[+-]?\d+$")


# ---------------------------------------------------------------------------
# axis table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Axis:
    """Metadata and validation rules for one sweepable name.

    ``kind`` drives validation (:func:`_values_from_token`) and rendering
    (:func:`format_spec`); ``default`` is the value a run uses when the axis is
    not swept, and it is always the same value the CLI flag defaults to, so
    "not swept" and "swept with the default" mean the same run.
    """

    name: str
    kind: str
    default: Any
    help: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    exclusive_minimum: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(
                f"axis {self.name!r}: unknown kind {self.kind!r}; choose from "
                f"{', '.join(_KINDS)}")
        if self.kind in _VECTOR_KINDS and not isinstance(self.default, tuple):
            raise ValueError(
                f"axis {self.name!r}: a {self.kind} default must be a tuple, "
                f"got {self.default!r}")
        if self.kind not in ("float", "int") and self.minimum is not None:
            raise ValueError(
                f"axis {self.name!r}: only numeric axes take a minimum")
        if self.exclusive_minimum and self.minimum is None:
            raise ValueError(
                f"axis {self.name!r}: an exclusive minimum needs a minimum")

    @property
    def vector(self) -> bool:
        """True when one value is a four-dimensional vector, not a scalar."""
        return self.kind in _VECTOR_KINDS

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description of the axis (for ``--help``/``--list``)."""
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "vector": self.vector,
            "default": _jsonable(self.default),
            "help": self.help,
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.minimum is not None:
            payload["minimum"] = self.minimum
            payload["exclusive_minimum"] = self.exclusive_minimum
        return payload


#: The fourteen sweepable axes, in canonical order.  The order is the product
#: order of :func:`expand_matrix` and the key order of every payload, so it is
#: part of the reproducible output contract and not an implementation detail.
AXES: Mapping[str, Axis] = MappingProxyType({
    axis.name: axis for axis in (
        Axis(
            "lattice", "lattice", policy.DEFAULT_LATTICE,
            "space-time extents x y z t (four positive even integers >= 4), "
            "e.g. lattice=8,8,8,16;16,16,16,16"),
        Axis(
            "lattice_grid", "lattice_grid", policy.DEFAULT_LATTICE_GRID,
            "MPI rank grid x y z t (four positive integers); a product that "
            "does not match the MPI size is a runner SKIP (mpi_grid)"),
        Axis(
            "dtype", "dtype", policy.DEFAULT_DTYPE,
            "complex precision: c64/c128, plus the complex64/complex128/"
            "float32/float64 aliases (normalised onto them)",
            choices=policy.DTYPE_INPUT_CHOICES),
        Axis(
            "backend", "backend", "all",
            "backend names; cpu/dcu/npu/all expand exactly like --backend",
            choices=policy.BACKEND_CHOICES),
        Axis(
            "device", "device", policy.DEFAULT_DEVICE,
            "device selection; auto forwards nothing and lets the backend "
            "choose, an unavailable device is SKIP (device_unavailable)",
            choices=policy.DEVICE_CHOICES),
        Axis(
            "solver", "solver", policy.DEFAULT_SOLVER,
            "solver tag filter (all keeps every solver case; cases that are "
            "neither a solver nor a multigrid check stay selected)",
            choices=policy.SOLVER_CHOICES),
        Axis(
            "dslash", "dslash", policy.DEFAULT_DSLASH,
            "dslash tag filter (all/both keep every dslash case; a non-dslash "
            "case is never filtered)",
            choices=policy.DSLASH_CHOICES),
        Axis(
            "mg_grid", "mg_grid", policy.DEFAULT_MG_GRID,
            "multigrid block grid x y z t (four positive integers)"),
        Axis(
            "mass", "float", policy.DEFAULT_MASS,
            "fermion mass forwarded to the runners; it has no default because "
            "the runner owns kappa = 1/(2m+8)"),
        Axis(
            "gauge_sigma", "float", policy.DEFAULT_GAUGE_SIGMA,
            "Gaussian width of the generated SU(3) background field",
            minimum=0.0),
        Axis(
            "csw", "float", policy.DEFAULT_CSW,
            "clover coefficient; 0 is the plain Wilson limit", minimum=0.0),
        Axis(
            "source", "source", policy.DEFAULT_SOURCE,
            "fermion source (point, ones or a seeded random field)",
            choices=policy.SOURCE_CHOICES),
        Axis(
            "gauge", "gauge", policy.DEFAULT_GAUGE,
            "background gauge field (random, unit or ones)",
            choices=policy.GAUGE_CHOICES),
        Axis(
            "seed", "int", policy.DEFAULT_SEED,
            "deterministic gauge/source seed (>= 0)", minimum=0),
    )
})

#: Axis names in canonical (declaration, product, payload) order.
AXIS_NAMES: tuple[str, ...] = tuple(AXES)

#: Accepted aliases for axis names, so a sweep can reuse the CLI flag spelling
#: (``--lattice-size``, ``--backend``).  Keys are already lower-cased and
#: ``-``/``_`` normalised by :func:`_normalize_name`.
AXIS_ALIASES: Mapping[str, str] = MappingProxyType({
    "lattice_size": "lattice",
    "backends": "backend",
})


def defaults() -> dict[str, Any]:
    """Return the value each axis uses when it is not swept.

    These are the :mod:`pyqcu.test.policy` defaults, i.e. what the same flag
    would do when omitted, so ``expand_matrix({})`` and a full run agree.
    """
    return {name: axis.default for name, axis in AXES.items()}


# ---------------------------------------------------------------------------
# names and tokenisation
# ---------------------------------------------------------------------------
def _normalize_name(name: Any) -> str:
    """Normalise an axis name (case, ``-``/``_``) and resolve its alias."""
    text = str(name).strip().lower().replace("-", "_")
    return AXIS_ALIASES.get(text, text)


def _axis(name: Any) -> Axis:
    """Look up an :class:`Axis` by name; ``ValueError`` when it is unknown."""
    try:
        return AXES[_normalize_name(name)]
    except KeyError:
        raise ValueError(
            f"unknown sweep axis {str(name)!r}; choose from "
            f"{', '.join(AXIS_NAMES)}") from None


def _is_axis(name: Any) -> bool:
    """True when ``name`` spells a known axis (used to spot ``a=1;b=2``)."""
    return _normalize_name(name) in AXES


def _split_outside(text: str, separators: str) -> list[str]:
    """Split ``text`` on any character in ``separators`` outside ``[]``/``()``.

    Brackets are value syntax (``lattice=[8,8,8,16]``): a separator inside one
    belongs to the value, not to the grammar.  Nesting is tracked with a single
    depth counter, and an unbalanced bracket simply keeps the rest of the text
    in one part, which the value parser then rejects with a message about the
    value instead of one about the spec.
    """
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for character in text:
        if character in "[(":
            depth += 1
        elif character in "])" and depth:
            depth -= 1
        if character in separators and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(character)
    parts.append("".join(buffer))
    return parts


def _words(text: str) -> list[str]:
    """Whitespace-separated words, leaving a bracketed group whole."""
    return [word for word in _split_outside(text, " \t\r\n") if word]


def _scalar_tokens(text: str) -> list[str]:
    """Tokenise a scalar value group: commas and whitespace both separate.

    Brackets around the whole list are tolerated (``dtype=[c64,c128]``), the
    same tolerance :func:`pyqcu.test.policy.parse_int_list` gives the CLI's
    list flags.
    """
    tokens: list[str] = []
    for word in _words(text):
        # Strip the brackets before splitting: inside ``[c64,c128]`` the comma
        # separates two values, not two extents of one vector.
        for piece in _split_outside(word.strip().strip("[]()"), ","):
            token = piece.strip()
            if token:
                tokens.append(token)
    return tokens


def _dedupe(values: Iterable[Any]) -> list[Any]:
    """Order-preserving duplicate removal (values are hashable by design)."""
    unique: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------
def _check_minimum(axis: Axis, number: float) -> None:
    """Reject a numeric value below the axis bound (if it declares one)."""
    if axis.minimum is None:
        return
    too_small = (number <= axis.minimum if axis.exclusive_minimum
                 else number < axis.minimum)
    if too_small:
        comparison = ">" if axis.exclusive_minimum else ">="
        raise ValueError(f"must be {comparison} {axis.minimum:g}, got {number!r}")


def _float_value(axis: Axis, token: Any) -> float:
    """Parse one finite float for a ``float`` axis."""
    if isinstance(token, bool):
        raise ValueError(f"expected a number, got {token!r}")
    try:
        number = float(token)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a number, got {token!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {token!r}")
    _check_minimum(axis, number)
    return number


def _int_value(axis: Axis, token: Any) -> int:
    """Parse one integer for an ``int`` axis (``"1.5"`` and ``"1e3"`` fail)."""
    if isinstance(token, bool):
        raise ValueError(f"expected an integer, got {token!r}")
    if isinstance(token, int):
        number = token
    elif isinstance(token, str) and _INTEGER.match(token.strip()):
        number = int(token.strip())
    else:
        raise ValueError(f"expected an integer, got {token!r}")
    _check_minimum(axis, number)
    return int(number)


def _values_from_token(axis: Axis, token: Any) -> tuple[Any, ...]:
    """Validate one value token for ``axis``; returns one or more values.

    Most kinds return exactly one value.  ``backend`` is the exception: an
    alias such as ``cpu`` names two canonical backends (``torch`` and
    ``kupl``) and ``all`` names three, so one token can select several points
    of the matrix - exactly what ``--backend`` already does.
    """
    try:
        if axis.kind == "backend":
            names = policy.expand_backends(token)
            if names == ("all",):
                names = policy.BACKENDS
            selected = set(names)
            return tuple(name for name in policy.BACKENDS if name in selected)
        if axis.kind == "lattice":
            return (policy.validate_lattice(policy.parse_int_list(token)),)
        if axis.kind == "lattice_grid":
            return (policy.validate_lattice_grid(policy.parse_int_list(token)),)
        if axis.kind == "mg_grid":
            return (policy.validate_mg_grid(policy.parse_int_list(token)),)
        if axis.kind == "dtype":
            return (policy.validate_dtype(token),)
        if axis.kind in _CHOICES:
            return (_CHOICES[axis.kind](token),)
        if axis.kind == "float":
            return (_float_value(axis, token),)
        if axis.kind == "int":
            return (_int_value(axis, token),)
    except ValueError as error:
        raise ValueError(f"axis {axis.name}={token!r}: {error}") from error
    raise ValueError(  # pragma: no cover - _KINDS keeps this unreachable
        f"axis {axis.name}: unknown kind {axis.kind!r}")


def _coerce_values(axis: Axis, value: Any) -> tuple[Any, ...]:
    """Validate a value (or a sequence of values) supplied by a caller.

    A string is a value list in spec syntax, an already-parsed scalar is one
    value, and a sequence is one value per element.  A list-valued axis also
    accepts a list of vectors, so ``{"lattice": [(8, 8, 8, 16), ...]}`` from an
    API caller is not mistaken for a single eight-integer vector.
    """
    if isinstance(value, str):
        return parse_axis(axis.name, value)
    if axis.vector:
        if (isinstance(value, (list, tuple)) and value
                and all(isinstance(item, (list, tuple)) for item in value)):
            return tuple(_values_from_token(axis, item)[0] for item in value)
        return _values_from_token(axis, value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items: Iterable[Any] = (
            sorted(value, key=repr) if isinstance(value, (set, frozenset))
            else value)
        parsed: list[Any] = []
        for item in items:
            parsed.extend(_values_from_token(axis, item))
        return tuple(_dedupe(parsed))
    return _values_from_token(axis, value)


# ---------------------------------------------------------------------------
# axis and spec parsing
# ---------------------------------------------------------------------------
def _strip_axis_name(axis: Axis, text: Any) -> str:
    """Drop a leading ``name=`` so ``parse_axis(name, "name=8,8,8,16")`` works."""
    body = "" if text is None else str(text).strip()
    head, separator, tail = body.partition("=")
    if not separator:
        return body
    candidate = _normalize_name(head)
    if candidate == axis.name:
        return tail.strip()
    if head.strip() and candidate in AXES:
        raise ValueError(
            f"axis {axis.name!r}: got a value for axis {candidate!r} in "
            f"{body!r}; pass one axis at a time or use parse_spec() for "
            "key=value pairs")
    return body


def _reject_foreign_assignment(axis: Axis, group: str, text: Any) -> None:
    """Refuse a second ``key=`` inside one axis value (a spec passed by mistake)."""
    head, separator, _ = group.partition("=")
    if separator and head.strip() and _is_axis(head):
        raise ValueError(
            f"axis {axis.name!r}: unexpected {_normalize_name(head)!r}= in "
            f"{str(text)!r}; pass one axis at a time or use parse_spec() for "
            "key=value pairs")


def _group_values(axis: Axis, group: str) -> tuple[Any, ...]:
    """Parse one ``;``-separated value group of an axis."""
    if axis.vector:
        return _values_from_token(axis, group)
    tokens = _scalar_tokens(group)
    if not tokens:
        raise ValueError(f"axis {axis.name!r}: no value in {group!r}")
    values: list[Any] = []
    for token in tokens:
        values.extend(_values_from_token(axis, token))
    return tuple(values)


def parse_axis(name: str, text: Any) -> tuple[Any, ...]:
    """Parse one axis value list into validated values.

    ``text`` is the value part of ``name=text`` (a leading ``name=`` is
    tolerated).  Values are separated by ``;``; a scalar axis additionally
    splits on ``,`` and whitespace, while a list-valued axis reads one
    four-dimensional vector per ``;`` group::

        parse_axis("dtype", "c64,c128")                     # ('c64', 'c128')
        parse_axis("dtype", "c64;c128")                     # ('c64', 'c128')
        parse_axis("dtype", "complex64")                    # ('c64',)
        parse_axis("lattice", "8,8,8,16")                   # ((8,8,8,16),)
        parse_axis("lattice", "[8,8,8,16];[16,16,16,16]")    # two lattices
        parse_axis("backend", "cpu")                         # ('torch', 'kupl')
        parse_axis("seed", "42")                             # (42,)

    Duplicates are removed in place (``dtype=c64,complex64`` is one value) so
    the product in :func:`expand_matrix` never repeats a point, and the values
    keep the order they were written in (an alias such as ``cpu`` expands in
    canonical backend order).  Anything the matching flag would reject raises
    ``ValueError`` here with the axis name and the offending token.
    """
    axis = _axis(name)
    body = _strip_axis_name(axis, text)
    groups = [group.strip() for group in _split_outside(body, ";")
              if group.strip()]
    if not groups:
        raise ValueError(
            f"axis {axis.name!r}: no value in {text!r}; write "
            f"{axis.name}=<value>[;<value>...] ({axis.help})")
    values: list[Any] = []
    for group in groups:
        _reject_foreign_assignment(axis, group, text)
        values.extend(_group_values(axis, group))
    unique = _dedupe(values)
    if not unique:
        raise ValueError(f"axis {axis.name!r}: no value in {text!r}")
    return tuple(unique)


def _join(text: str, words: Iterable[str]) -> str:
    """Join a value text and its continuation words with single spaces."""
    parts = [text.strip(), *[str(word).strip() for word in words]]
    return " ".join(part for part in parts if part)


def _split_spec(chunk: str) -> list[tuple[str, str]]:
    """Split ``key=value;key=value`` into ``(axis, value-text)`` groups.

    The value syntax needs three separators and they mean three things:

    ``;``     a new value group (``lattice=[8,8,8,16];[16,16,16,16]``);
    ``,``     a new scalar value of the same axis (``dtype=c64,c128``);
    space     a continuation of the current value group (``lattice=8 8 8 16``
              is one vector of four extents), or a new ``key=value``.

    So a ``;`` group without ``key=`` continues the previous axis, and
    whitespace inside a group belongs to that group's value.
    """
    groups: list[list[Any]] = []  # [axis name, [value text per ';' group]]
    for segment in _split_outside(chunk, ";"):
        trailing: list[str] = []
        started = False  # did a key= start in this segment?
        for word in _words(segment):
            head, separator, tail = word.partition("=")
            if separator and head.strip():
                name = _normalize_name(head)
                if name not in AXES:
                    raise ValueError(
                        f"unknown sweep axis {head.strip()!r} in {chunk!r}; "
                        f"choose from {', '.join(AXIS_NAMES)}")
                if trailing:
                    groups[-1][1][-1] = _join(groups[-1][1][-1], trailing)
                    trailing = []
                groups.append([name, [tail.strip()]])
                started = True
            elif groups:
                trailing.append(word)
            else:
                raise ValueError(
                    f"sweep spec {chunk!r} must start with key=value, got "
                    f"{word!r}")
        if trailing:
            if started:
                groups[-1][1][-1] = _join(groups[-1][1][-1], trailing)
            else:
                groups[-1][1].append(" ".join(trailing))
    return [(name, ";".join(texts)) for name, texts in groups]


def _normalize_axes(axes: Mapping[Any, Any]) -> dict[str, tuple[Any, ...]]:
    """Validate a ``{axis: value(s)}`` mapping into canonical order."""
    normalized: dict[str, tuple[Any, ...]] = {}
    for raw_name, raw_value in axes.items():
        axis = _axis(raw_name)
        values = _coerce_values(axis, raw_value)
        if not values:
            raise ValueError(
                f"axis {axis.name!r} has no values: {raw_value!r}")
        if axis.name in normalized:
            values = tuple(_dedupe([*normalized[axis.name], *values]))
        normalized[axis.name] = values
    return {name: normalized[name] for name in AXIS_NAMES if name in normalized}


def parse_spec(spec: Any) -> dict[str, tuple[Any, ...]]:
    """Parse a whole sweep spec into ``{axis: values}`` in canonical order.

    ``spec`` is the ``--sweep`` string (``key=value;key=value``), a sequence of
    such strings (a repeatable flag), or a mapping of axis names to values
    (the API form).  A repeated axis accumulates its values::

        parse_spec("dtype=c64,c128;seed=1,2")
        parse_spec(["dtype=c64", "dtype=c128"])     # same as dtype=c64,c128
        parse_spec({"lattice": [(8, 8, 8, 16), (16, 16, 16, 16)]})

    The result is ordered by :data:`AXIS_NAMES`, which is what makes the
    expansion, the payload and the rendered rows reproducible.
    """
    if isinstance(spec, Mapping):
        return _normalize_axes(spec)
    if isinstance(spec, str):
        chunks: Sequence[str] = (spec,)
    elif isinstance(spec, Iterable):
        chunks = [str(item) for item in spec]
    else:
        raise ValueError(
            "sweep spec must be a string, a sequence of key=value strings or "
            f"a mapping, got {type(spec).__name__}")
    collected: dict[str, list[str]] = {}
    for chunk in chunks:
        for name, value_text in _split_spec(chunk):
            collected.setdefault(name, []).append(value_text)
    if not collected:
        raise ValueError(
            "empty sweep spec: expected key=value pairs such as "
            "dtype=c64,c128;lattice=[8,8,8,16];[16,16,16,16]")
    return {name: parse_axis(name, ";".join(collected[name]))
            for name in AXIS_NAMES if name in collected}


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------
def expand_matrix(axes: Any) -> list[dict[str, Any]]:
    """Expand a sweep into the explicit list of configurations.

    The result is the Cartesian product of the axis values in
    :data:`AXIS_NAMES` order (``itertools.product``, last axis varies fastest),
    one dict per combination, so the same spec always yields the same list.
    Duplicate rows are dropped, which matters when two spellings normalise onto
    the same value (``dtype=c64,complex64``, ``backend=cpu,torch,kupl``).

    An empty spec is the empty product: one configuration (``{}``), which is
    exactly "run the current selection once".  More than :data:`MAX_CASES`
    combinations raise ``ValueError`` *before* the list is built.
    """
    normalized = parse_spec(axes)
    if not normalized:
        return [{}]
    per_axis = ", ".join(f"{name}={len(values)}"
                         for name, values in normalized.items())
    total = math.prod(len(values) for values in normalized.values())
    if total > MAX_CASES:
        raise ValueError(
            f"sweep expands to {total} combinations ({per_axis}), above the "
            f"hard cap of {MAX_CASES}; narrow one axis (seed=42 alone is one "
            "run)")
    combinations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for combination in itertools.product(*normalized.values()):
        if combination in seen:
            continue
        seen.add(combination)
        combinations.append(dict(zip(normalized, combination)))
    return combinations


def _format_value(value: Any) -> str:
    """Render one plan value in the spec syntax (vectors as ``[a,b,c,d]``)."""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def format_spec(spec: Any) -> str:
    """Render a sweep spec back to its canonical text form.

    The text re-parses to the same mapping (``parse_spec(format_spec(x)) == x``
    for any valid spec), which is what lets a report quote the sweep it ran in
    a form the user can paste back into ``--sweep``.
    """
    axes = parse_spec(spec)
    parts: list[str] = []
    for name, values in axes.items():
        # A list-valued axis needs ``;`` between values: inside one value the
        # comma already separates the four extents.  A scalar axis reads both,
        # and the comma is the shorter, more familiar spelling.
        separator = ";" if AXES[name].vector else ","
        rendered = separator.join(_format_value(value) for value in values)
        parts.append(f"{name}={rendered}")
    return ";".join(parts)


def select_kwargs(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one expanded configuration into ``Test.select`` keywords.

    The axis names are the :meth:`pyqcu.test.common.Test.select` keywords, with
    two deliberate differences: ``backend`` is spelled ``backends`` there (it
    accepts a sequence, which is what an expanded ``cpu`` produces), and every
    other axis must resolve to exactly one value, because one configuration is
    one run.

    Values may be text (``{"dtype": "c64,c128"}``) or already parsed; unknown
    axes and multi-value axes raise ``ValueError`` instead of silently dropping
    part of the sweep.
    """
    if not isinstance(configuration, Mapping):
        raise ValueError(
            "select_kwargs needs one expanded configuration mapping, got "
            f"{type(configuration).__name__}")
    normalized = _normalize_axes(configuration)
    kwargs: dict[str, Any] = {}
    for name, values in normalized.items():
        if name == "backend":
            kwargs["backends"] = values[0] if len(values) == 1 else values
            continue
        if len(values) != 1:
            raise ValueError(
                f"axis {name!r} expands to {len(values)} values; a single "
                "configuration needs exactly one (see expand_matrix)")
        kwargs[name] = values[0]
    return kwargs


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """JSON-safe form of a plan value (tuples become lists, scalars stay)."""
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _value_of(source: Any, name: str) -> Any:
    """Read one field from a mapping or an object; ``None`` counts as absent."""
    if source is None:
        return _MISSING
    if isinstance(source, Mapping):
        value = source.get(name, _MISSING)
    else:
        value = getattr(source, name, _MISSING)
    return _MISSING if value is None else value


def _first_value(case: Any, configuration: Any, name: str, default: Any) -> Any:
    """Resolve one row field from the case, the configuration and the defaults.

    A *mapping* case is a row the caller already composed, so its own value
    wins.  A :class:`~pyqcu.test.common.Case` is a template: the configuration
    (the sweep point) wins, because changing the lattice, dtype, device, solver
    or dslash is the whole purpose of a sweep, and for a non-swept run the two
    agree - ``Test.select`` rewrites ``Case.lattice``/``Case.dtype`` whenever
    it changes them.  The axis default is the last resort for a row that
    declared nothing (``device`` on a bare ``Case``, for instance).
    """
    sources = ((case, configuration) if isinstance(case, Mapping)
               else (configuration, case))
    for source in sources:
        value = _value_of(source, name)
        if value is not _MISSING:
            return value
    return default


def _lattice(value: Any) -> tuple[int, int, int, int]:
    """Normalise a row lattice (``"8,8,8,16"`` or a sequence) for the report."""
    if isinstance(value, str):
        value = policy.parse_int_list(value)
    return policy.validate_lattice(value)


def _scalar(value: Any) -> Any:
    """Keep JSON scalars as they are; stringify anything else."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _row(case: Any, configuration: Any) -> dict[str, Any]:
    """Build one JSON-safe matrix row for ``case`` under ``configuration``."""
    tag = _value_of(case, "tag")
    if tag is _MISSING:
        raise ValueError(
            "a matrix row needs a case tag of the form "
            f"'backend/area/variant', but {type(case).__name__} has none; pass "
            "Case objects (Test.cases) or mappings that carry a tag")
    if not isinstance(tag, str) or "/" not in tag:
        raise ValueError(
            "a matrix row needs a case tag of the form "
            f"'backend/area/variant', got {tag!r}")
    backend = _first_value(case, configuration, "backend", tag.split("/", 1)[0])
    lattice = _first_value(case, configuration, "lattice",
                           AXES["lattice"].default)
    dtype = _first_value(case, configuration, "dtype", AXES["dtype"].default)
    device = _first_value(case, configuration, "device", AXES["device"].default)
    solver = _first_value(case, configuration, "solver", AXES["solver"].default)
    dslash = _first_value(case, configuration, "dslash",
                          AXES["dslash"].default)
    try:
        values = {
            "tag": tag,
            "backend": str(backend),
            "lattice": [int(extent) for extent in _lattice(lattice)],
            "dtype": policy.validate_dtype(dtype),
            "device": _scalar(device),
            "solver": str(solver),
            "dslash": str(dslash),
        }
    except ValueError as error:
        raise ValueError(f"matrix row {tag!r}: {error}") from error
    return {name: values[name] for name in ROW_FIELDS}


def _configurations(config: Any) -> tuple[Any, ...]:
    """Normalise ``config`` into a tuple of per-run configurations."""
    if config is None:
        return (None,)
    if isinstance(config, Mapping) or not isinstance(config, (list, tuple)):
        return (config,)
    configurations = tuple(config)
    if not configurations:
        raise ValueError("a matrix needs at least one configuration")
    return configurations


def _items(cases: Any) -> tuple[Any, ...]:
    """Normalise ``cases`` into a tuple of case-like objects or row mappings."""
    if cases is None:
        raise ValueError(
            "a matrix needs cases: pass Case objects (Test.cases) or mappings "
            "that carry a tag")
    if isinstance(cases, str):
        raise ValueError(f"a matrix needs cases, got the string {cases!r}")
    if isinstance(cases, Mapping):
        return (cases,)
    if not isinstance(cases, Iterable):
        # ``Test`` is not iterable but holds the selected cases; a single
        # ``Case`` has no ``cases`` attribute and is its own single item.
        nested = getattr(cases, "cases", None)
        if isinstance(nested, (list, tuple)):
            return tuple(nested)
        return (cases,)
    return tuple(cases)


def render_matrix(cases: Any, config: Any = None) -> list[dict[str, Any]]:
    """Render the planned matrix as JSON-safe rows.

    ``cases`` is a sequence of :class:`pyqcu.test.common.Case` objects (usually
    ``Test.cases``), a ``Test`` itself, or mappings that already carry the row
    fields.  ``config`` is one run configuration or a sequence of them; a
    sequence is the sweep, and every case is rendered once per configuration.
    A configuration is a mapping of run values (the dicts
    :func:`expand_matrix` returns) or a ``Test``/``Options`` object.

    Order is configuration-major: all rows of the first configuration, then all
    rows of the second.  That is the same order as the product in
    :func:`expand_matrix`, so
    ``render_matrix(cases, expand_matrix(spec))`` lines up with the
    combinations without a second sort.

    Precedence per field: a mapping case (an already-composed row) wins over
    everything, otherwise the configuration wins over the ``Case`` template -
    the sweep point is what the row is *about*, and for a non-swept run the two
    agree.  Values that neither declares fall back to the axis default.

    Rows carry exactly :data:`ROW_FIELDS`: the tag plus the six configuration
    values that change what the case exercises.  Values are JSON scalars and
    lists of ints by construction - there is nothing here that needs a
    serializer (no tensor, no numpy scalar, no complex number).
    """
    configurations = _configurations(config)
    items = _items(cases)
    return [_row(case, configuration)
            for configuration in configurations
            for case in items]


def matrix_payload(spec: Any, cases: Any,
                   config: Any = None) -> dict[str, Any]:
    """Build the ``--sweep --list --json`` document for ``spec`` and ``cases``.

    Without ``config`` the spec is expanded (:func:`expand_matrix`) and every
    case is rendered once per combination; with ``config`` the caller has
    already expanded the sweep and that sequence is rendered instead.  The
    payload is JSON-safe end to end (``json.dumps(..., allow_nan=False)``
    accepts it) and is ordered by :data:`AXIS_NAMES`.

    ``cases`` is grouped by ``combinations``/``per_combination``: each block of
    ``per_combination`` rows belongs to the next configuration, in the product
    order of :func:`expand_matrix`.
    """
    axes = parse_spec(spec)
    configurations = (expand_matrix(axes) if config is None
                      else _configurations(config))
    rows = render_matrix(cases, config=configurations)
    return {
        "spec": format_spec(axes),
        "order": list(axes),
        "axes": {name: [_jsonable(value) for value in values]
                 for name, values in axes.items()},
        "limit": MAX_CASES,
        "combinations": len(configurations),
        "per_combination": len(rows) // len(configurations),
        "count": len(rows),
        "cases": rows,
    }
