"""Named run profiles of the formal :mod:`pyqcu.test` port (pure data).

A profile is a named, *complete* set of run options: it says "run the CPU
suite", "run everything" or "run the clover checks in double precision" with
one word instead of a dozen flags, and the same table serves scripts, CI and
the future ``--profile`` flag.  This module is **data plus validation only**:
it adds no CLI flag, runs nothing and imports no backend.

Import rules
------------
Standard library and :mod:`pyqcu.test.policy` only - no ``torch``, ``numpy``,
``mpi4py``, ``pytest`` and no ``pyqcu.testing`` - so a ``--list`` path stays
importable in a bare interpreter.  Importing this module reads no file, no
environment variable and no device; it only builds the tables below.

Declared values vs resolved values
----------------------------------
:data:`PROFILES` stores the *declared* option values, one entry per option in
:data:`PROFILE_OPTIONS`.  :func:`resolve_profile` and :func:`validate_profile`
return a **new, normalised** dict whose values are already in the spelling
:meth:`pyqcu.test.common.Test.select` and
:class:`pyqcu.test.common.Options` accept:

* ``tier``  ``None`` (every declared tier) or an int from ``policy.TIERS``;
  the CLI spelling ``"all"`` is accepted and normalises to ``None``;
* ``lattice``/``lattice_grid``/``mg_grid``  four-tuples of ints (lattice
  extents even and >= 4, grid extents positive);
* ``dtype``  canonical ``c64``/``c128`` (``complex64``... aliases accepted);
* ``backend``  a tuple of *canonical* backend names in ``policy.BACKENDS``
  order; the ``"all"`` spelling and the ``cpu``/``dcu``/``npu`` aliases are
  expanded here, so a profile never leaks an alias into a report;
* ``solver``/``dslash``/``source``/``gauge``  a declared choice; ``device`` a
  declared choice or a concrete ``cuda:3``-style string;
* ``seed``/``repeat``/``warmup``/``gauge_sigma``/``mass``/``csw``  numbers,
  bounds-checked the way :class:`Options` checks them (``repeat``/``warmup``
  >= 1, ``seed``/``gauge_sigma``/``csw`` >= 0, ``mass`` free).

Values are declared **literally** instead of being read from :mod:`policy`,
so a profile is a stable pin: changing a policy default must not silently
change what ``full`` runs.  :func:`validate_profile` re-checks every literal
against the policy vocabulary, which is what turns that duplication into a
real check - the same pattern as ``policy.TIER0_RUNNERS`` vs
``units.CASES``.  :data:`PROFILES` is a plain dict and may be extended at
runtime (``PROFILES["mine"] = ...``); the accessors always return copies, so
mutating a resolved dict never changes :data:`PROFILES`, and mutating
:data:`PROFILES` never changes a dict already handed out.

The file is named ``profiles.py`` on purpose: a ``test_*.py`` inside
``pyqcu/test`` would be reported by the repository form audit as a test in
the wrong place.  The module defines no ``test_*`` function and runs nothing
when imported.

Profiles
--------
==========  ============================================================
name        pins
==========  ============================================================
``quick``   tier 0, ``8 8 8 16``, ``c64``, seed 42, repeat 1, warmup 1,
            every backend/solver/dslash, device ``auto``
``cpu``     ``quick`` on torch + kupl, device ``cpu``
``gpu``     ``quick`` on torch + cuda, device ``gpu``
``c128``    ``quick`` in double precision (dtype ``c128``)
``clover``  ``quick`` with the clover dslash (``dslash clover``,
            ``csw 1.0``)
``mg``      ``quick`` with multigrid (``solver multigrid``,
            ``mg_grid 2 2 2 2``)
``full``    ``quick`` with every declared tier (``tier 'all'``, resolved
            to ``None``) and ``repeat 5``
==========  ============================================================

Example
-------
>>> from pyqcu.test import profiles
>>> profiles.profile_names()
('quick', 'cpu', 'gpu', 'c128', 'clover', 'mg', 'full')
>>> quick = profiles.resolve_profile("quick")
>>> quick["lattice"], quick["dtype"], quick["repeat"]
((8, 8, 8, 16), 'c64', 1)
>>> profiles.resolve_profile("full")["tier"] is None      # every tier
True
>>> base = {"only": ["torch/dslash"], "tier": 1, "dtype": None}
>>> merged = profiles.merge_profile(base, "c128")
>>> merged["only"], merged["tier"], merged["dtype"]
(['torch/dslash'], 1, 'c128')

:func:`merge_profile` fills **only** values the base leaves unset: ``None``
is this port's "not given" (the ``argparse`` default, and what
``Options.as_kwargs`` drops), while ``0``/``0.0``/``False`` are real values
(``tier 0``, ``csw 0.0`` is the Wilson limit, ``gauge_sigma 0.0``).  Unknown
profile names raise ``ValueError`` and list what is available.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from . import policy

__all__ = [
    "PROFILES",
    "PROFILE_OPTIONS",
    "merge_profile",
    "profile_names",
    "resolve_profile",
    "validate_profile",
]


# ---------------------------------------------------------------------------
# option validators (one per entry of PROFILE_OPTIONS)
# ---------------------------------------------------------------------------
def _delegate(option: str,
              validate: Callable[[Any], Any]) -> Callable[[str, Any], Any]:
    """Bind ``option`` to a :mod:`policy` validator.

    :mod:`pyqcu.test.policy` owns every spelling and bound (dtype aliases,
    backend aliases, grid arity, solver/dslash/device vocabulary).  Reusing
    those validators instead of restating them is the repository rule; this
    wrapper only adds the profile and option name to the error message.
    """
    def check(profile: str, value: Any) -> Any:
        try:
            return validate(value)
        except ValueError as error:
            raise ValueError(
                f"profile {profile!r}: {option}: {error}") from error
    return check


def _integer(option: str, *, minimum: int) -> Callable[[str, Any], Any]:
    """Build a validator for an integer option with a lower bound.

    ``bool`` is rejected on purpose: ``True`` is an ``int`` in Python, but
    ``warmup=True`` is a bug, not a value.
    """
    def check(profile: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"profile {profile!r}: {option} must be an integer >= "
                f"{minimum}, got {value!r}")
        if value < minimum:
            raise ValueError(
                f"profile {profile!r}: {option} must be >= {minimum}, "
                f"got {value!r}")
        return value
    return check


def _number(option: str, *, minimum: float | None = None,
            optional: bool = False) -> Callable[[str, Any], Any]:
    """Build a validator for a finite float option.

    Mirrors ``Options``: a numeric string is accepted (``"0.1"``), ``bool`` is
    not, and ``NaN``/``inf`` are rejected.  ``optional=True`` lets ``None``
    through (``mass`` has no default: the runners own the physical default).
    """
    def check(profile: str, value: Any) -> float | None:
        if value is None:
            if optional:
                return None
            raise ValueError(
                f"profile {profile!r}: {option} must be a finite number, "
                "got None")
        if isinstance(value, bool):
            raise ValueError(
                f"profile {profile!r}: {option} must be a finite number, "
                f"got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"profile {profile!r}: {option} must be a finite number, "
                f"got {value!r}") from error
        if not math.isfinite(number):
            raise ValueError(
                f"profile {profile!r}: {option} must be finite, got "
                f"{value!r}")
        if minimum is not None and number < minimum:
            raise ValueError(
                f"profile {profile!r}: {option} must be >= {minimum:g}, "
                f"got {value!r}")
        return number
    return check


def _tier(profile: str, value: Any) -> int | None:
    """Validate ``tier``: ``None``/``"all"`` (every tier) or ``policy.TIERS``.

    ``None`` is what ``pyqcu.test.common.Test.select`` reads as "no tier
    filter", and it is also what the CLI's ``--tier all`` resolves to; both
    spellings are accepted so a profile can be written either way.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    choices = ", ".join(str(item) for item in policy.TIERS)
    if isinstance(value, bool) or not isinstance(value, int) \
            or value not in policy.TIERS:
        raise ValueError(
            f"profile {profile!r}: tier must be one of {choices}, 'all' or "
            f"None, got {value!r}")
    return value


def _backends(profile: str, value: Any) -> tuple[str, ...]:
    """Validate ``backend`` and expand it onto concrete canonical names.

    ``policy.expand_backends`` owns the vocabulary, so ``"all"``, the
    ``cpu``/``dcu``/``npu`` aliases and comma/space separated spellings are
    accepted.  ``"all"`` is expanded to every declared backend rather than
    returned as the ``("all",)`` marker, so a resolved profile always names
    the backends it means.
    """
    try:
        names = policy.expand_backends(value)
    except ValueError as error:
        raise ValueError(
            f"profile {profile!r}: backend: {error}") from error
    if names == ("all",):
        return tuple(policy.BACKENDS)
    return tuple(names)


def _device(profile: str, value: Any) -> str:
    """Validate ``device``: a declared choice or a concrete device string."""
    try:
        return policy.validate_device(value)
    except ValueError:
        pass
    try:
        resolved = policy.device_for_runner(value)
    except ValueError as error:
        raise ValueError(
            f"profile {profile!r}: device: {error}") from error
    if resolved is None:
        raise ValueError(
            f"profile {profile!r}: device must be one of "
            f"{', '.join(policy.DEVICE_CHOICES)}, got {value!r}")
    return resolved


#: One validator per profile option: ``(profile, value) -> normalised``.
#: The declaration order is the ``--list``/report order and it is also the
#: order of :data:`PROFILE_OPTIONS`, so an option can never be listed without
#: a validator (or validated without being listed).
_VALIDATORS: dict[str, Callable[[str, Any], Any]] = {
    "tier": _tier,
    "lattice": _delegate("lattice", policy.validate_lattice),
    "dtype": _delegate("dtype", policy.validate_dtype),
    "seed": _integer("seed", minimum=0),
    "repeat": _integer("repeat", minimum=1),
    "warmup": _integer("warmup", minimum=1),
    "backend": _backends,
    "device": _device,
    "solver": _delegate("solver", policy.validate_solver),
    "dslash": _delegate("dslash", policy.validate_dslash),
    "mg_grid": _delegate("mg_grid", policy.validate_mg_grid),
    "lattice_grid": _delegate("lattice_grid", policy.validate_lattice_grid),
    "gauge_sigma": _number("gauge_sigma", minimum=0.0),
    "mass": _number("mass", optional=True),
    "csw": _number("csw", minimum=0.0),
    "source": _delegate("source", policy.validate_source),
    "gauge": _delegate("gauge", policy.validate_gauge),
}

#: The option names every profile declares, in ``--list``/report order.
#:
#: A profile is *complete*: :func:`validate_profile` rejects a missing option
#: (a profile that silently falls back to a runtime default is not
#: reproducible) and an unknown one (a typo such as ``"masss"`` must not be
#: ignored).  The tuple is taken from the validator table, so an option cannot
#: be listed without a validator (or validated without being listed); adding
#: an option means adding its validator above.
PROFILE_OPTIONS: tuple[str, ...] = tuple(_VALIDATORS)


# ---------------------------------------------------------------------------
# the profiles
# ---------------------------------------------------------------------------
#: The ``quick`` pins - tier 0, the project correctness lattice, ``c64``,
#: seed 42, repeat/warmup 1, every backend, solver and dslash, device
#: ``auto``.  Every other profile is this baseline plus the deltas its name
#: promises, which is what keeps "``cpu`` is ``quick`` on CPU" true by
#: construction instead of by review.
_QUICK: dict[str, Any] = {
    "tier": 0,
    "lattice": (8, 8, 8, 16),
    "dtype": "c64",
    "seed": 42,
    "repeat": 1,
    "warmup": 1,
    "backend": "all",
    "device": "auto",
    "solver": "all",
    "dslash": "all",
    "mg_grid": (2, 2, 2, 2),
    "lattice_grid": (1, 1, 1, 1),
    "gauge_sigma": 0.1,
    "mass": None,
    "csw": 1.0,
    "source": "point",
    "gauge": "random",
}


def _profile(**overrides: Any) -> dict[str, Any]:
    """Return a complete profile: the ``quick`` pins plus ``overrides``.

    A misspelled option raises here, at import time, instead of becoming an
    extra key that only a later :func:`validate_profile` call would report.
    """
    unknown = sorted(set(overrides) - set(PROFILE_OPTIONS))
    if unknown:
        raise ValueError(
            "profile override(s) outside PROFILE_OPTIONS: "
            + ", ".join(unknown))
    profile = dict(_QUICK)
    profile.update(overrides)
    return profile


#: The named profiles.  A plain dict on purpose: a site may add its own entry
#: at runtime (``PROFILES["mine"] = {...}``) and the accessors accept it.
PROFILES: dict[str, dict[str, Any]] = {
    "quick": _profile(),
    "cpu": _profile(backend=("torch", "kupl"), device="cpu"),
    "gpu": _profile(backend=("torch", "cuda"), device="gpu"),
    "c128": _profile(dtype="c128"),
    # ``csw 1.0`` is spelled out even though it is also the ``quick`` value:
    # the Wilson limit is ``csw 0``, so the clover profile must state it.
    "clover": _profile(dslash="clover", csw=1.0),
    # ``mg_grid 2 2 2 2`` is the multigrid coarse block grid, not the lattice.
    "mg": _profile(solver="multigrid", mg_grid=(2, 2, 2, 2)),
    # ``tier 'all'`` is every declared tier and resolves to ``None`` (the
    # value ``Test.select`` reads as "no tier filter"); ``repeat 5`` is the
    # only repeat at which a median is meaningful.
    "full": _profile(tier="all", repeat=5),
}


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------
def profile_names() -> tuple[str, ...]:
    """Return every available profile name, in declaration order."""
    return tuple(PROFILES)


def _available() -> str:
    """Comma-separated profile names, for error messages."""
    names = profile_names()
    return ", ".join(names) if names else "<none>"


def _lookup(name: Any) -> tuple[str, Mapping[str, Any]]:
    """Return ``(key, profile)`` for ``name``, or raise ``ValueError``.

    Names are matched case-insensitively after stripping surrounding
    whitespace.  An unknown name lists what is available: a typo in a script
    must say which profiles exist instead of failing somewhere downstream.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"profile name must be a non-empty string, got {name!r}; "
            f"available profiles: {_available()}")
    key = name.strip().lower()
    try:
        profile = PROFILES[key]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; available profiles: {_available()}"
        ) from None
    if not isinstance(profile, Mapping):
        raise ValueError(
            f"profile {key!r} must be a mapping of option names to values, "
            f"got {type(profile).__name__}")
    return key, profile


def validate_profile(name: str) -> dict[str, Any]:
    """Validate a named profile and return a new, normalised option dict.

    Every option in :data:`PROFILE_OPTIONS` must be present and every value
    must be accepted by :mod:`pyqcu.test.policy`; otherwise ``ValueError``
    names the profile, the option and the offending value.  An unknown
    profile name lists every available profile.  The return value is a fresh
    dict: mutating it (or :data:`PROFILES`) afterwards never affects the
    other side.
    """
    key, profile = _lookup(name)
    missing = [option for option in PROFILE_OPTIONS if option not in profile]
    if missing:
        raise ValueError(
            f"profile {key!r} is incomplete: missing option(s) "
            f"{', '.join(missing)}")
    unknown = sorted(set(profile) - set(PROFILE_OPTIONS))
    if unknown:
        raise ValueError(
            f"profile {key!r} has unknown option(s) {', '.join(unknown)}; "
            f"known options: {', '.join(PROFILE_OPTIONS)}")
    return {
        option: _VALIDATORS[option](key, profile[option])
        for option in PROFILE_OPTIONS
    }


def resolve_profile(name: str) -> dict[str, Any]:
    """Return a new, normalised option dict for a named profile.

    This is the accessor form of :func:`validate_profile` - same lookup,
    same validation, same errors - and the function a CLI or ``--list``
    should call.  The result is a fresh dict whose values match
    :class:`~pyqcu.test.common.Options` field for field apart from ``tier``
    and ``backend``, which are selection rather than execution options
    (``Test.select`` takes them as ``tier=`` and ``backends=``).
    Normalisation is described in the module docstring (``tier "all"`` ->
    ``None``, ``backend "all"`` -> every canonical backend, and so on).
    """
    return validate_profile(name)


def merge_profile(base: Mapping[str, Any] | None,
                  name: str) -> dict[str, Any]:
    """Overlay a named profile onto ``base``, filling only unset values.

    ``base`` is any mapping of option names to values - typically
    ``vars(cli_args)`` or another profile's :func:`resolve_profile` result -
    and may carry keys a profile does not know (``only``, ``atol``,
    ``timeout``); those are copied through untouched, and ``None`` is treated
    as an empty mapping.  A value counts as unset when its key is absent
    **or** its value is ``None``: ``None`` is this port's "not given" (the
    ``argparse`` default, and what ``Options.as_kwargs`` drops).
    ``0``/``0.0``/``False`` are *values*, not "unset": ``tier 0``, ``csw
    0.0`` (the Wilson limit) and ``gauge_sigma 0.0`` must never be
    overwritten by a profile default.

    Neither ``base`` nor :data:`PROFILES` is modified.  The result is a new
    dict, and its profile half is immutable (tuples, numbers, strings), so
    the caller owns the result outright.  An unknown or invalid profile
    raises ``ValueError``; a non-mapping ``base`` raises ``TypeError``.
    """
    if base is None:
        base = {}
    if not isinstance(base, Mapping):
        raise TypeError(
            "base must be a mapping of option names to values, got "
            f"{type(base).__name__}")
    merged = dict(base)
    for option, value in resolve_profile(name).items():
        if merged.get(option) is None:
            merged[option] = value
    return merged
