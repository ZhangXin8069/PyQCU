"""Contract self-check for the formal :mod:`pyqcu.test` port.

The port ships in the sdist/wheel, so its invariants are checked here rather
than in a repository-only test module.  The file is deliberately **not** named
``test_*.py``: a ``pyqcu/test/test_*.py`` path trips the repository form audit
(``WARNING TEST_LOCATION``), which reserves that layout for ``pyqcu/testing``.

What is checked
---------------
1. ``import pyqcu.test`` (and every sibling module) pulls in no ``torch``,
   ``numpy``, ``mpi4py``, ``pytest`` and no ``pyqcu.testing`` - verified in a
   fresh interpreter, because this process may already have torch loaded.
2. ``python -m pyqcu.test --list --json`` returns the 19 tier-0 cases and
   still leaves those modules unloaded.
3. ``--list --json`` carries the *effective* configuration: every new run flag
   (grids, solver/dslash selection, device, dtype aliases, source/gauge,
   mass/kappa/csw, iteration and multigrid knobs, threads/device id) reaches
   the payload normalised, and ``--quick`` still locks tier 0, ``8 8 8 16``,
   ``c64``, seed 42 and repeat 1 while leaving the other new flags usable.
4. Usage errors return exit code 2 (``--warmup 0``, unknown ``--only``,
   unknown ``--backend``, ``--quick`` conflict, malformed grid) and ``--help``
   advertises every documented flag.  The comma-separated (``a,b``),
   space-separated (``a b``) and repeated (``--only a --only b``) ``--only``
   spellings all select the same cases.
5. The six harness modules contain no forbidden import statement, and
   ``pyqcu/test`` contains no ``test_*.py``/``*_test.py`` file.
6. ``setup.py`` ``find_packages(include=..., exclude=...)`` includes
   ``pyqcu.test`` and excludes ``pyqcu.testing``.
7. ``MANIFEST.in`` does not prune ``pyqcu/test`` and still ships ``*.py``.
8. Policy constants, the 19 ``(tag, runner)`` pairs, tolerances, skip tokens,
   exit codes, the grid/device/solver normalisers and the gate vocabulary
   behave as documented.
9. The gate evaluator, the runner resolver and the aggregate exit codes behave
   as documented (a wrong value must FAIL, a missing key must FAIL).
10. The expanded features are wired, not just declared: ``--profile`` applies
    a named option set with explicit flags winning, ``--sweep``/``--matrix``
    expands deterministically within ``matrix.MAX_CASES`` and ``--list --json``
    emits those rows, and ``report.write_report`` writes JSON/JUnit/CSV/
    Markdown atomically with no ``.tmp`` left behind.

``main()`` returns ``0`` when every check passes (skips are not failures) and
``1`` otherwise.  ``python -m pyqcu.test --check`` delegates here.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

from .. import __version__
from . import matrix, policy, profiles, report, units
from .common import (
    Case,
    Result,
    TestReport,
    evaluate_expect,
    json_safe,
    relative_error,
    resolve_runner,
    summarize,
    timed,
    time_samples,
    write_json_atomic,
)

__all__ = ["Check", "main", "run_checks", "test_policy_constants",
           "test_selfcheck"]

#: ``pyqcu/test`` directory (works from a checkout and from site-packages).
PACKAGE_DIR = Path(__file__).resolve().parent
#: Repository root when running from a checkout; ``site-packages`` otherwise.
REPO_ROOT = PACKAGE_DIR.parents[1]

#: Harness modules owned by this agent; the import scans only look at these.
HARNESS_MODULES = (
    "__init__.py",
    "__main__.py",
    "common.py",
    "matrix.py",
    "policy.py",
    "profiles.py",
    "report.py",
    "units.py",
    "selfcheck.py",
)

#: Command-line flags the help text must advertise.  The two aliases are part
#: of the contract: ``--lattice`` (for ``--lattice-size``) and the historical
#: ``--guage-sigma`` typo must keep parsing, so ``--help`` has to name them.
REQUIRED_FLAGS = (
    "--backend", "--profile", "--lattice-size", "--lattice", "--lattice-grid",
    "--mg-grid", "--quick", "--tier", "--only", "--sweep", "--matrix",
    "--solver", "--dslash", "--device", "--source", "--gauge",
    "--gauge-sigma", "--guage-sigma", "--mass", "--kappa", "--csw",
    "--max-iter", "--restart", "--mg-levels", "--mg-nu-pre", "--mg-nu-post",
    "--threads", "--device-id", "--list", "--json", "--junit", "--csv",
    "--markdown", "--repeat", "--warmup", "--seed", "--dtype", "--atol",
    "--timeout", "--fail-fast", "--allow-all-skipped", "--outdir",
    "--verbose", "--check",
)

#: An import statement of a forbidden module (top-level or ``pyqcu.testing``).
_FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:pyqcu\.testing|torch|numpy|mpi4py|pytest)\b",
    re.MULTILINE,
)


class Check(NamedTuple):
    """One self-check outcome."""

    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _subprocess_environment() -> dict[str, str]:
    """Environment that imports the working tree, not an installed copy."""
    environment = dict(os.environ)
    root = str(REPO_ROOT)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_python(source: str, *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run ``source`` in a fresh interpreter rooted at the checkout."""
    cwd = str(REPO_ROOT) if REPO_ROOT.is_dir() else None
    return subprocess.run(
        [sys.executable, "-B", "-c", source],
        cwd=cwd,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _offender_expression() -> str:
    """Python source computing the forbidden modules currently loaded."""
    return (
        "offenders = sorted(\n"
        "    name for name in sys.modules\n"
        f"    if name.split('.')[0] in {policy.FORBIDDEN_IMPORT_ROOTS!r}\n"
        "    or any(name == prefix or name.startswith(prefix + '.')\n"
        f"           for prefix in {policy.FORBIDDEN_IMPORT_PREFIXES!r}))\n"
    )


def _check_import_is_stdlib_only() -> Check:
    """Import every harness module in a fresh interpreter and inspect modules."""
    source = (
        "import json, sys\n"
        "import pyqcu.test\n"
        "import pyqcu.test.common\n"
        "import pyqcu.test.matrix\n"
        "import pyqcu.test.policy\n"
        "import pyqcu.test.profiles\n"
        "import pyqcu.test.report\n"
        "import pyqcu.test.units\n"
        "import pyqcu.test.selfcheck\n"
        "import pyqcu.test.__main__\n"
        "import pyqcu.test.torch_qcu\n"
        "import pyqcu.test.kupl_qcu\n"
        "import pyqcu.test.cuda_qcu\n"
        + _offender_expression()
        + "print(json.dumps({'offenders': offenders,\n"
        "                   'version': pyqcu.test.__version__}))\n"
    )
    completed = _run_python(source)
    if completed.returncode != 0:
        return Check(
            "import_is_stdlib_only", "FAIL",
            f"fresh interpreter import failed (rc={completed.returncode}): "
            f"{_tail(completed.stderr)}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return Check("import_is_stdlib_only", "FAIL",
                     f"unparsable probe output: {_tail(completed.stdout)}")
    if payload["offenders"]:
        return Check(
            "import_is_stdlib_only", "FAIL",
            f"importing pyqcu.test loaded forbidden module(s): "
            f"{', '.join(payload['offenders'])}")
    return Check(
        "import_is_stdlib_only", "PASS",
        "import pyqcu.test loads no torch/numpy/mpi4py/pytest/pyqcu.testing")


def _check_cli_list_light() -> Check:
    """``--list --json`` must emit the 19 cases without loading a backend."""
    source = (
        "import contextlib, io, json, sys\n"
        "from pyqcu.test.__main__ import main\n"
        "buffer = io.StringIO()\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    code = main(['--list', '--json'])\n"
        "payload = json.loads(buffer.getvalue())\n"
        + _offender_expression()
        + "print(json.dumps({'code': code, 'count': payload['count'],\n"
        "                   'tags': [case['tag'] for case in payload['cases']],\n"
        "                   'config': {key: payload.get(key) for key in\n"
        "                              ('lattice', 'lattice_grid',\n"
        "                               'mg_grid', 'dtype', 'solver',\n"
        "                               'dslash', 'device',\n"
        "                               'runner_device', 'source',\n"
        "                               'gauge', 'gauge_sigma', 'csw',\n"
        "                               'atol')},\n"
        "                   'offenders': offenders}))\n"
    )
    completed = _run_python(source)
    if completed.returncode != 0:
        return Check(
            "cli_list_light", "FAIL",
            f"--list --json failed (rc={completed.returncode}): "
            f"{_tail(completed.stderr)}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return Check("cli_list_light", "FAIL",
                     f"unparsable --list output: {_tail(completed.stdout)}")
    problems = []
    if payload["code"] != 0:
        problems.append(f"exit code {payload['code']} != 0")
    if payload["count"] != len(units.CASES):
        problems.append(f"count {payload['count']} != {len(units.CASES)}")
    if tuple(payload["tags"]) != policy.TIER0_TAGS:
        problems.append("declared tags differ from policy.TIER0_TAGS")
    if payload["offenders"]:
        problems.append("loaded " + ", ".join(payload["offenders"]))
    expected_config = {
        "lattice": list(policy.DEFAULT_LATTICE),
        "lattice_grid": list(policy.DEFAULT_LATTICE_GRID),
        "mg_grid": list(policy.DEFAULT_MG_GRID),
        "dtype": policy.DEFAULT_DTYPE,
        "solver": policy.DEFAULT_SOLVER,
        "dslash": policy.DEFAULT_DSLASH,
        "device": policy.DEFAULT_DEVICE,
        "runner_device": policy.device_for_runner(policy.DEFAULT_DEVICE),
        "source": policy.DEFAULT_SOURCE,
        "gauge": policy.DEFAULT_GAUGE,
        "gauge_sigma": policy.DEFAULT_GAUGE_SIGMA,
        "csw": policy.DEFAULT_CSW,
        "atol": None,
    }
    for key, want in expected_config.items():
        got = payload["config"][key]
        if got != want:
            problems.append(f"{key} default {got!r} != {want!r}")
    if problems:
        return Check("cli_list_light", "FAIL", "; ".join(problems))
    return Check("cli_list_light", "PASS",
                 f"{len(units.CASES)} cases listed without a backend import, "
                 "effective defaults reported")


def _check_cli_effective_config() -> Check:
    """New run flags must reach ``--list --json`` as the effective config."""
    source = (
        "import contextlib, io, json\n"
        "from pyqcu.test.__main__ import main\n"
        "def listing(argv):\n"
        "    buffer = io.StringIO()\n"
        "    with contextlib.redirect_stdout(buffer):\n"
        "        code = main(['--list', '--json'] + argv)\n"
        "    return code, json.loads(buffer.getvalue())\n"
        "custom = ('--lattice-size', '8 8 8 16', '--lattice-grid',\n"
        "          '[2,1,1,1]', '--mg-grid', '2,2,1,2', '--gauge-sigma',\n"
        "          '0.25', '--mass', '0.17', '--kappa', '0.12', '--csw',\n"
        "          '0.9', '--max-iter', '50', '--restart', '30',\n"
        "          '--mg-levels', '3', '--mg-nu-pre', '1', '--mg-nu-post',\n"
        "          '2', '--threads', '4', '--device-id', '2', '--solver',\n"
        "          'cg', '--dslash', 'clover', '--device', 'dcu', '--dtype',\n"
        "          'complex128', '--source', 'ones', '--gauge', 'unit',\n"
        "          '--backend', 'cpu')\n"
        "custom_code, custom = listing(list(custom))\n"
        "quick_code, quick = listing(['--quick', '--device', 'cpu',\n"
        "                             '--threads', '2'])\n"
        "print(json.dumps({'custom_code': custom_code, 'custom': custom,\n"
        "                  'quick_code': quick_code, 'quick': quick}))\n"
    )
    completed = _run_python(source)
    if completed.returncode != 0:
        return Check("cli_effective_config", "FAIL",
                     f"probe failed: {_tail(completed.stderr)}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return Check("cli_effective_config", "FAIL",
                     f"unparsable probe output: {_tail(completed.stdout)}")
    problems: list[str] = []
    codes = {"custom": payload["custom_code"], "quick": payload["quick_code"]}
    for name, code in codes.items():
        if code != 0:
            problems.append(f"{name} --list --json exited {code}")
    custom = payload["custom"]
    expected = {
        "lattice_grid": [2, 1, 1, 1],
        "mg_grid": [2, 2, 1, 2],
        "dtype": "c128",
        "gauge_sigma": 0.25,
        "mass": 0.17,
        "kappa": 0.12,
        "csw": 0.9,
        "max_iter": 50,
        "restart": 30,
        "mg_levels": 3,
        "mg_nu_pre": 1,
        "mg_nu_post": 2,
        "threads": 4,
        "device_id": 2,
        "solver": "cg",
        "dslash": "clover",
        "device": "dcu",
        "runner_device": "dcu:0",
        "source": "ones",
        "gauge": "unit",
        "backends": ["torch", "kupl"],
    }
    if list(custom["lattice"]) != list(policy.DEFAULT_LATTICE):
        problems.append("'8 8 8 16' did not parse to the default lattice")
    if custom["count"] != len(custom["cases"]):
        problems.append(
            f"custom count {custom['count']} != {len(custom['cases'])} cases")
    if not custom["cases"]:
        problems.append("the custom selection is empty")
    if custom["quick"]:
        problems.append("custom run reported quick=True")
    for key, want in expected.items():
        got = custom[key]
        if got != want:
            problems.append(f"{key} {got!r} != {want!r}")
    if any(case["tag"].endswith("/bistabcg") for case in custom["cases"]):
        problems.append("--solver cg kept a BiCGStab case")
    if any(case["tag"].startswith("cuda/") for case in custom["cases"]):
        problems.append("--backend cpu kept a cuda case")
    quick = payload["quick"]
    locked = {
        "quick": True,
        "tier": 0,
        "lattice": list(policy.QUICK_LATTICE),
        "dtype": policy.QUICK_DTYPE,
        "seed": policy.QUICK_SEED,
        "repeat": policy.QUICK_REPEAT,
        "device": "cpu",
        "threads": 2,
    }
    for key, want in locked.items():
        got = quick[key]
        if got != want:
            problems.append(f"--quick {key} {got!r} != {want!r}")
    if problems:
        return Check("cli_effective_config", "FAIL", "; ".join(problems))
    return Check(
        "cli_effective_config", "PASS",
        f"{len(expected)} configuration keys propagate and --quick stays "
        "locked with other new flags usable")


def _check_cli_usage_codes() -> Check:
    """Usage/environment errors must return exit code 2, not 1."""
    source = (
        "import json\n"
        "from pyqcu.test.__main__ import main\n"
        "codes = {\n"
        "    'warmup_zero': main(['--warmup', '0']),\n"
        "    'only_unknown': main(['--only', 'no/such/case']),\n"
        "    'backend_unknown': main(['--backend', 'nope']),\n"
        "    'quick_dtype': main(['--quick', '--dtype', 'c128']),\n"
        "    'quick_seed': main(['--quick', '--seed', '7']),\n"
        "    'quick_tier': main(['--quick', '--tier', '1']),\n"
        "    'quick_lattice': main(['--quick', '--lattice-size',\n"
        "                                 '4,4,4,4']),\n"
        "    'repeat_zero': main(['--repeat', '0']),\n"
        "    'lattice_short': main(['--lattice-size', '8,8,8']),\n"
        "    'lattice_odd': main(['--lattice-size', '8,8,8,15']),\n"
        "    'grid_zero': main(['--lattice-grid', '0,1,1,1']),\n"
        "    'grid_short': main(['--mg-grid', '2,2,2']),\n"
        "    'kappa_zero': main(['--kappa', '0']),\n"
        "    'profile_unknown': main(['--profile', 'nope', '--list']),\n"
        "    'profile_quick_conflict': main(['--quick', '--profile', 'c128',\n"
        "                                    '--list']),\n"
        "    'sweep_unknown': main(['--sweep', 'dtype=nope', '--list']),\n"
        "    'sweep_quick_conflict': main(['--quick', '--sweep', 'dtype=c128',\n"
        "                                  '--list']),\n"
        "    'sweep_cap': main(['--sweep',\n"
        "                       'seed=' + ','.join(str(i) for i in range(600)),\n"
        "                       '--list']),\n"
        "    'list_export': main(['--list', '--csv', 'unused.csv']),\n"
        "    'stdout_clash': main(['--json', '-', '--csv', '-', '--only',\n"
        "                          'torch/lattice']),\n"
        "    'check_sweep': main(['--check', '--sweep', 'dtype=c64']),\n"
        "}\n"
        "print(json.dumps(codes))\n"
    )
    completed = _run_python(source)
    if completed.returncode != 0:
        return Check("cli_usage_codes", "FAIL",
                     f"probe failed: {_tail(completed.stderr)}")
    try:
        codes = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return Check("cli_usage_codes", "FAIL",
                     f"unparsable probe output: {_tail(completed.stdout)}")
    usage = policy.EXIT_CODES["usage"]
    wrong = {name: code for name, code in codes.items() if code != usage}
    if wrong:
        return Check(
            "cli_usage_codes", "FAIL",
            "expected exit code 2 for " + ", ".join(
                f"{name} (got {code})" for name, code in sorted(wrong.items())))
    return Check("cli_usage_codes", "PASS",
                 f"{len(codes)} usage errors return exit code {usage}")


def _check_cli_flags() -> Check:
    """Every documented flag must appear in ``--help`` as its own option.

    A plain substring test would accept ``--lattice`` because
    ``--lattice-size`` contains it; matching on option boundaries keeps the
    aliases honest.
    """
    from .__main__ import build_parser

    text = build_parser().format_help()
    missing = [
        flag for flag in REQUIRED_FLAGS
        if not re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", text)
    ]
    if missing:
        return Check("cli_flags", "FAIL",
                     f"--help is missing: {', '.join(missing)}")
    return Check("cli_flags", "PASS",
                 f"{len(REQUIRED_FLAGS)} documented flags present in --help")


def _check_source_imports() -> Check:
    """The harness sources must not import a backend or ``pyqcu.testing``."""
    offenders: list[str] = []
    for name in HARNESS_MODULES:
        path = PACKAGE_DIR / name
        if not path.is_file():
            offenders.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN_IMPORT.finditer(text):
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{name}:{line}: {match.group(0).strip()}")
    if offenders:
        return Check("source_imports", "FAIL", "; ".join(offenders))
    return Check("source_imports", "PASS",
                 f"{len(HARNESS_MODULES)} modules import only the standard library")


def _check_no_test_modules() -> Check:
    """No ``test_*.py`` / ``*_test.py`` inside the shipped ``pyqcu/test``."""
    offenders = sorted(
        path.name for path in PACKAGE_DIR.glob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py"))
    if offenders:
        return Check(
            "no_test_modules", "FAIL",
            f"pyqcu/test must not contain pytest modules: {', '.join(offenders)}")
    return Check("no_test_modules", "PASS",
                 "pyqcu/test holds no test_*.py (see selfcheck.py instead)")


def _setup_find_packages_arguments() -> dict[str, list[str]]:
    """Read the literal ``find_packages(include=..., exclude=...)`` arguments."""
    setup = REPO_ROOT / "setup.py"
    tree = ast.parse(setup.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = getattr(function, "id", None) or getattr(function, "attr", None)
        if name != "find_packages":
            continue
        arguments: dict[str, list[str]] = {}
        for keyword in node.keywords:
            if keyword.arg in ("include", "exclude"):
                try:
                    arguments[keyword.arg] = list(ast.literal_eval(keyword.value))
                except (ValueError, TypeError):
                    continue
        return arguments
    return {}


def _check_package_inclusion() -> Check:
    """``pyqcu.test`` must be packaged; ``pyqcu.testing`` must not."""
    setup = REPO_ROOT / "setup.py"
    if not setup.is_file():
        return Check("package_inclusion", "SKIP",
                     f"not a source checkout ({REPO_ROOT})")
    arguments = _setup_find_packages_arguments()
    if not arguments:
        return Check("package_inclusion", "FAIL",
                     "setup.py has no literal find_packages(include=..., "
                     "exclude=...) call")
    include = arguments.get("include", [])
    exclude = arguments.get("exclude", [])
    if "pyqcu.*" not in include and "pyqcu" not in include:
        return Check("package_inclusion", "FAIL",
                     f"find_packages include={include!r} does not cover pyqcu.*")
    if not any(item == "pyqcu.testing" or item.startswith("pyqcu.testing.")
               for item in exclude):
        return Check("package_inclusion", "FAIL",
                     f"find_packages exclude={exclude!r} does not prune "
                     "pyqcu.testing")
    if any(item == "pyqcu.test" or item.startswith("pyqcu.test.")
           for item in exclude):
        return Check("package_inclusion", "FAIL",
                     f"find_packages exclude={exclude!r} prunes pyqcu.test")

    try:
        from setuptools import find_packages
    except ImportError:
        return Check(
            "package_inclusion", "PASS",
            "static: setup.py excludes pyqcu.testing and not pyqcu.test "
            "(setuptools unavailable)")
    packages = find_packages(where=str(REPO_ROOT), include=include,
                             exclude=exclude)
    problems = []
    if "pyqcu.test" not in packages:
        problems.append("pyqcu.test missing from find_packages()")
    if any(item == "pyqcu.testing" or item.startswith("pyqcu.testing.")
           for item in packages):
        problems.append("pyqcu.testing present in find_packages()")
    if problems:
        return Check("package_inclusion", "FAIL", "; ".join(problems))
    return Check("package_inclusion", "PASS",
                 f"find_packages() has pyqcu.test, not pyqcu.testing "
                 f"({len(packages)} packages)")


def _check_manifest() -> Check:
    """``MANIFEST.in`` must not prune ``pyqcu/test``."""
    manifest = REPO_ROOT / "MANIFEST.in"
    if not manifest.is_file():
        return Check("manifest", "SKIP", f"no MANIFEST.in at {REPO_ROOT}")
    lines = [
        line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    problems = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        directive, target = parts[0], parts[1].rstrip("/")
        if directive in ("prune", "exclude", "recursive-exclude") \
                and (target == "pyqcu/test" or target.startswith("pyqcu/test/")):
            problems.append(f"{directive} {target}")
    if not any(line.split()[0] == "recursive-include"
               and len(line.split()) > 2
               and line.split()[1].rstrip("/") == "pyqcu"
               and "*.py" in line.split()[2:]
               for line in lines):
        problems.append("no 'recursive-include pyqcu ... *.py' rule")
    if problems:
        return Check("manifest", "FAIL", "; ".join(problems))
    testing_pruned = any(line.split()[:2] == ["prune", "pyqcu/testing"]
                         for line in lines)
    return Check(
        "manifest", "PASS",
        "MANIFEST.in ships pyqcu/**/*.py and does not prune pyqcu/test"
        + (" (pyqcu/testing pruned)" if testing_pruned else ""))


def _check_policy_constants() -> Check:
    """The frozen policy must match the documented contract exactly."""
    problems: list[str] = []

    def expect(name: str, condition: bool, detail: str) -> None:
        if not condition:
            problems.append(f"{name}: {detail}")

    expect("QUICK_LATTICE", policy.QUICK_LATTICE == (8, 8, 8, 16),
           f"expected (8, 8, 8, 16), got {policy.QUICK_LATTICE!r}")
    expect("DEFAULT_LATTICE", policy.DEFAULT_LATTICE == (8, 8, 8, 16),
           f"expected (8, 8, 8, 16), got {policy.DEFAULT_LATTICE!r}")
    expect("quick dtype", policy.QUICK_DTYPE == "c64",
           f"expected c64, got {policy.QUICK_DTYPE!r}")
    expect("quick seed", policy.QUICK_SEED == 42,
           f"expected 42, got {policy.QUICK_SEED!r}")
    expect("warmup", policy.DEFAULT_WARMUP >= 1,
           f"warmup must be >= 1, got {policy.DEFAULT_WARMUP!r}")
    expect("repeat", policy.DEFAULT_REPEAT >= 1 and policy.QUICK_REPEAT >= 1,
           "repeat must be >= 1")
    expect("DTYPE_CHOICES", tuple(policy.DTYPE_CHOICES) == ("c64", "c128"),
           f"got {policy.DTYPE_CHOICES!r}")
    expect("BACKENDS", tuple(policy.BACKENDS) == ("torch", "cuda", "kupl"),
           f"got {policy.BACKENDS!r}")
    expect("STATUSES",
           tuple(policy.STATUSES) == ("PASS", "FAIL", "SKIP", "TIMEOUT", "ERROR"),
           f"got {policy.STATUSES!r}")
    expect("EXIT_CODES",
           dict(policy.EXIT_CODES) == {"pass": 0, "fail": 1, "usage": 2},
           f"got {policy.EXIT_CODES!r}")
    expect("SKIP_TOKENS",
           tuple(policy.SKIP_TOKENS) == (
               "kupl_library", "cuda_bridge", "cuda_device",
               "mpi_single_rank", "h5py_mpi", "mpi_grid",
               "device_unavailable"),
           f"got {policy.SKIP_TOKENS!r}")
    expect("skip descriptions",
           all(policy.describe_skip(token) for token in policy.SKIP_TOKENS),
           "a skip token has no description")
    expect("grid/device skip tokens",
           policy.is_skip_token("mpi_grid")
           and policy.is_skip_token("device_unavailable"),
           "mpi_grid/device_unavailable are not in the skip vocabulary")
    expect("FORBIDDEN_IMPORT_ROOTS",
           tuple(policy.FORBIDDEN_IMPORT_ROOTS) == (
               "torch", "numpy", "mpi4py", "pytest"),
           f"got {policy.FORBIDDEN_IMPORT_ROOTS!r}")
    expect("FORBIDDEN_IMPORT_PREFIXES",
           tuple(policy.FORBIDDEN_IMPORT_PREFIXES) == ("pyqcu.testing",),
           f"got {policy.FORBIDDEN_IMPORT_PREFIXES!r}")
    expect("TIER0_RUNNERS", len(policy.TIER0_RUNNERS) == 19,
           f"expected 19 entries, got {len(policy.TIER0_RUNNERS)}")
    expect("TIER0_TAGS", tuple(policy.TIER0_TAGS) == tuple(policy.TIER0_RUNNERS),
           "TIER0_TAGS disagrees with TIER0_RUNNERS")

    # Grids, dtypes, devices and backend aliases: the normalisers the CLI and
    # the API share.  A changed default here changes every run silently.
    expect("DEFAULT_LATTICE_GRID",
           tuple(policy.DEFAULT_LATTICE_GRID) == (1, 1, 1, 1),
           f"got {policy.DEFAULT_LATTICE_GRID!r}")
    expect("DEFAULT_MG_GRID", tuple(policy.DEFAULT_MG_GRID) == (2, 2, 2, 2),
           f"got {policy.DEFAULT_MG_GRID!r}")
    expect("DEFAULT_GAUGE_SIGMA", policy.DEFAULT_GAUGE_SIGMA == 0.1,
           f"got {policy.DEFAULT_GAUGE_SIGMA!r}")
    expect("DEFAULT_CSW", policy.DEFAULT_CSW == 1.0,
           f"got {policy.DEFAULT_CSW!r}")
    expect("DEFAULT_MASS/KAPPA",
           policy.DEFAULT_MASS is None and policy.DEFAULT_KAPPA is None,
           "mass/kappa must default to None so the runner default applies")
    expect("SOLVER_CHOICES",
           tuple(policy.SOLVER_CHOICES)
           == ("all", "bicgstab", "cg", "multigrid", "mg"),
           f"got {policy.SOLVER_CHOICES!r}")
    expect("DSLASH_CHOICES",
           tuple(policy.DSLASH_CHOICES) == ("all", "wilson", "clover", "both"),
           f"got {policy.DSLASH_CHOICES!r}")
    expect("DEVICE_CHOICES",
           tuple(policy.DEVICE_CHOICES) == ("auto", "cpu", "gpu", "dcu", "npu"),
           f"got {policy.DEVICE_CHOICES!r}")
    expect("SOURCE_CHOICES",
           tuple(policy.SOURCE_CHOICES) == ("point", "ones", "random"),
           f"got {policy.SOURCE_CHOICES!r}")
    expect("GAUGE_CHOICES",
           tuple(policy.GAUGE_CHOICES) == ("random", "unit", "ones"),
           f"got {policy.GAUGE_CHOICES!r}")
    expect("dtype aliases",
           (policy.validate_dtype("c64"), policy.validate_dtype("complex64"),
            policy.validate_dtype("float32"), policy.validate_dtype("c128"),
            policy.validate_dtype("complex128"),
            policy.validate_dtype("float64"))
           == ("c64", "c64", "c64", "c128", "c128", "c128"),
           "a dtype alias does not normalise onto c64/c128")
    try:
        policy.normalize_dtype("float16")
        expect("dtype unknown", False, "float16 was accepted")
    except ValueError:
        pass
    expect("backend aliases",
           (policy.expand_backends("cpu"), policy.expand_backends("dcu"),
            policy.expand_backends("npu"),
            policy.expand_backends("torch,cuda"),
            policy.expand_backends("[kupl]"), policy.expand_backends("all"))
           == (("torch", "kupl"), ("cuda",), ("torch",),
               ("torch", "cuda"), ("kupl",), ("all",)),
           f"got {policy.expand_backends('cpu')!r} ...")
    try:
        policy.expand_backends("quantum")
        expect("backend unknown", False, "quantum was accepted")
    except ValueError:
        pass
    expect("device mapping",
           (policy.device_for_runner("auto"), policy.device_for_runner("cpu"),
            policy.device_for_runner("gpu"), policy.device_for_runner("dcu"),
            policy.device_for_runner("npu"), policy.device_for_runner("cuda:3"))
           == (None, "cpu", "cuda:0", "dcu:0", "npu:0", "cuda:3"),
           "device_for_runner does not follow the documented mapping")
    expect("lattice_grid ok",
           policy.validate_lattice_grid("[1,1,2,4]") == (1, 1, 2, 4),
           "rejected a legal lattice grid")
    expect("mg_grid ok",
           policy.validate_mg_grid("2 2 2 2") == (2, 2, 2, 2),
           "rejected a legal mg grid")
    for bad in (("1,1,1",), ("0,1,1,1",), ("1,-1,1,1",), ("a,b,c,d",)):
        try:
            policy.validate_lattice_grid(*bad)
            expect(f"validate_lattice_grid{bad}", False,
                   "accepted an invalid grid")
        except ValueError:
            pass
        try:
            policy.validate_mg_grid(*bad)
            expect(f"validate_mg_grid{bad}", False, "accepted an invalid grid")
        except ValueError:
            pass
    expect("parse_int_list",
           (policy.parse_int_list("[8,8,8,16]"),
            policy.parse_int_list("8,8,8,16"),
            policy.parse_int_list(["8", "8", "8", "16"]),
            policy.parse_int_list((8, 8, 8, 16)))
           == ((8, 8, 8, 16),) * 4,
           "a lattice list spelling did not parse")
    expect("solver_selects",
           policy.solver_selects("solve", "torch/solve/cg", "cg")
           and not policy.solver_selects("solve", "torch/solve/bistabcg", "cg")
           and policy.solver_selects("solve", "kupl/solve/bistabcg", "bicgstab")
           and not policy.solver_selects("mg", "torch/mg/legacy", "cg")
           and policy.solver_selects("mg", "torch/mg/legacy", "mg")
           and policy.solver_selects("dslash", "torch/dslash/wilson", "cg"),
           "the solver selection does not follow solve/bistabcg|solve/cg|mg/")
    expect("dslash_selects",
           policy.dslash_selects("dslash", "torch/dslash/wilson", "wilson")
           and not policy.dslash_selects("dslash", "torch/dslash/clover",
                                         "wilson")
           and policy.dslash_selects("dslash", "cuda/dslash/clover", "clover")
           and not policy.dslash_selects("dslash", "torch/dslash/parity",
                                         "wilson")
           and policy.dslash_selects("dslash", "torch/dslash/parity", "both")
           and policy.dslash_selects("solve", "torch/solve/cg", "clover"),
           "the dslash selection does not filter kind=dslash tags only")

    for dtype in policy.DTYPE_CHOICES:
        table = policy.DTYPE_TOLERANCES.get(dtype, {})
        expect(f"{dtype} tolerances", bool(table), "no tolerance table")
        for name in policy.TOLERANCE_NAMES:
            value = table.get(name)
            expect(f"{dtype}/{name}", isinstance(value, float) and value > 0.0,
                   f"expected a positive float, got {value!r}")
        compare = table.get("compare", 0.0)
        solve = table.get("solve", 0.0)
        native = table.get("native_solve", 0.0)
        expect(f"{dtype} ordering", compare <= native <= solve,
               f"expected compare <= native_solve <= solve, got "
               f"{compare}, {native}, {solve}")
    for name in policy.TOLERANCE_NAMES:
        expect(f"c128 tighter/{name}",
               policy.tolerance("c128", name)
               < policy.tolerance("c64", name),
               "c128 must be tighter than c64")
    expect("atol override",
           policy.tolerance("c64", "compare", atol=1e-7) == 1e-7
           and policy.tolerance("c64", "solve", atol=1e-7) == 1e-3,
           "atol must override 'compare' only")
    try:
        policy.tolerance("c64", "nope")
        expect("unknown tolerance", False, "no KeyError raised")
    except KeyError:
        pass

    for bad in ((8, 8, 8), (8, 8, 8, 15), (2, 8, 8, 16), (8, 8, 8, -16)):
        try:
            policy.validate_lattice(bad)
            expect(f"validate_lattice{bad}", False, "accepted an invalid lattice")
        except ValueError:
            pass
    expect("validate_lattice ok",
           policy.validate_lattice([8, 8, 8, 16]) == (8, 8, 8, 16),
           "rejected the project lattice")

    defaults = policy.quick_defaults()
    expect("quick_defaults",
           defaults["tier"] == 0
           and tuple(defaults["lattice"]) == (8, 8, 8, 16)
           and defaults["dtype"] == "c64"
           and defaults["seed"] == 42,
           f"got {defaults!r}")
    expect("skip tokens known",
           all(policy.is_skip_token(token) for token in policy.SKIP_TOKENS),
           "a declared skip token is not known")
    expect("describe skip",
           policy.describe_skip("cuda_bridge") != "", "empty description")

    if problems:
        return Check("policy_constants", "FAIL", "; ".join(problems))
    return Check("policy_constants", "PASS",
                 "lattice, tolerances, skip tokens, statuses and exit codes")


def _good_observation(case: Case) -> dict[str, Any]:
    """A synthetic observation that satisfies every gate of ``case``."""
    compare = policy.tolerance(case.dtype, "compare") / 10.0
    solve = policy.tolerance(case.dtype, "solve") / 10.0
    native = policy.tolerance(case.dtype, "native_solve") / 10.0
    where = {
        "lattice": {"su3": True, "gamma_deviation": compare / 10.0},
        "gauge": {"su3": True},
        "dslash": {"relative_error": compare},
        "smear": {"su3": True, "relative_error": compare},
        "solve": {"relative_residual": solve, "iterations": 12,
                  "max_iter": 100},
        "mg": {"converged": True, "relative_residual": solve,
               "variant": case.tag.rsplit("/", 1)[-1]},
        "abi": {"count": 40, "set_match": True},
        "smoke": {"init_ok": True},
    }
    return dict(where[case.kind])


def _check_units_contract() -> Check:
    """Tags, runner strings, skip tokens and gates must match the contract."""
    problems: list[str] = []
    tags = tuple(case.tag for case in units.CASES)
    if tags != policy.TIER0_TAGS:
        problems.append(f"units tags {tags!r} != policy.TIER0_TAGS "
                        f"{policy.TIER0_TAGS!r}")
    if tuple(units.TAGS) != tags:
        problems.append("units.TAGS disagrees with units.CASES")
    if dict(units.BY_TAG) != {case.tag: case for case in units.CASES}:
        problems.append("units.BY_TAG disagrees with units.CASES")

    for case in units.CASES:
        expected = policy.TIER0_RUNNERS.get(case.tag)
        if expected != case.runner:
            problems.append(
                f"{case.tag}: runner {case.runner!r} != contract {expected!r}")
        if case.tier != 0:
            problems.append(f"{case.tag}: tier {case.tier} != 0")
        if tuple(case.lattice) != policy.QUICK_LATTICE:
            problems.append(f"{case.tag}: lattice {case.lattice} != "
                            f"{policy.QUICK_LATTICE}")
        if case.dtype != "c64":
            problems.append(f"{case.tag}: dtype {case.dtype} != c64")
        for token in case.requires:
            if not policy.is_skip_token(token):
                problems.append(f"{case.tag}: unknown skip token {token!r}")
        for key, spec in case.expect.items():
            for operator, threshold in spec.items():
                if isinstance(threshold, str) and operator in (
                        "lt", "le", "gt", "ge"):
                    try:
                        policy.tolerance(case.dtype, threshold)
                    except KeyError:
                        problems.append(
                            f"{case.tag}: expect[{key!r}] uses unknown "
                            f"tolerance {threshold!r}")
        try:
            failed = evaluate_expect(case.expect, _good_observation(case),
                                     dtype=case.dtype)
        except ValueError as error:
            problems.append(f"{case.tag}: gate definition error: {error}")
            continue
        if failed:
            problems.append(
                f"{case.tag}: satisfies-nothing gate: {'; '.join(failed)}")

    by_backend = {backend: sum(1 for case in units.CASES
                               if case.backend == backend)
                  for backend in policy.BACKENDS}
    expected_backends = {"torch": 11, "cuda": 3, "kupl": 5}
    if by_backend != expected_backends:
        problems.append(f"backend coverage {by_backend} != {expected_backends}")

    if problems:
        return Check("units_contract", "FAIL", "; ".join(problems))
    return Check(
        "units_contract", "PASS",
        f"{len(units.CASES)} cases: 11 torch, 5 kupl, 5 cuda; runners and "
        "gates match policy")


def _check_run_config() -> Check:
    """``Options``/``Test``/``TestReport.options`` must agree field by field.

    This is the contract of the run flags: one normaliser, one key set, one
    forward rule (``as_kwargs`` never sends ``None``), and the same values in
    the API, in ``--list --json`` and in a report.
    """
    from .common import CONFIG_FIELDS, Options, Test  # noqa: F401 - contract

    problems: list[str] = []
    expected_keys = set(CONFIG_FIELDS) | {"runner_device"}
    config = Test.default().config()
    if set(config) != expected_keys:
        problems.append(
            f"Test.config keys {sorted(set(config) - expected_keys)} extra / "
            f"{sorted(expected_keys - set(config))} missing")
    defaults = {
        "lattice": list(policy.DEFAULT_LATTICE),
        "lattice_grid": list(policy.DEFAULT_LATTICE_GRID),
        "mg_grid": list(policy.DEFAULT_MG_GRID),
        "dtype": policy.DEFAULT_DTYPE,
        "seed": policy.DEFAULT_SEED,
        "source": policy.DEFAULT_SOURCE,
        "gauge": policy.DEFAULT_GAUGE,
        "gauge_sigma": policy.DEFAULT_GAUGE_SIGMA,
        "csw": policy.DEFAULT_CSW,
        "solver": policy.DEFAULT_SOLVER,
        "dslash": policy.DEFAULT_DSLASH,
        "device": policy.DEFAULT_DEVICE,
        "runner_device": None,
        "mass": None,
        "kappa": None,
        "max_iter": None,
        "restart": None,
        "mg_levels": None,
        "mg_nu_pre": None,
        "mg_nu_post": None,
        "threads": None,
        "device_id": None,
        "atol": None,
    }
    for key, want in defaults.items():
        if config.get(key) != want:
            problems.append(f"default {key} {config.get(key)!r} != {want!r}")

    base = Options()
    kwargs = base.as_kwargs()
    if any(value is None for value in kwargs.values()):
        problems.append("Options().as_kwargs() forwards a None value")
    if "timeout" in kwargs:
        problems.append("Options().as_kwargs() forwarded timeout")
    for key in ("mass", "kappa", "max_iter", "device"):
        if key in kwargs:
            problems.append(f"unset {key} was forwarded")

    tuned = Options(lattice=[8, 8, 8, 16], lattice_grid="2,1,1,1",
                    mg_grid="[2,2,2,2]", dtype="complex128", device="dcu",
                    mass=0.17, kappa=0.1, max_iter=100, restart=30,
                    mg_levels=2, mg_nu_pre=1, mg_nu_post=1, threads=4,
                    device_id=1, source="ones", gauge="unit", csw=0.9)
    tuned_kwargs = tuned.as_kwargs()
    if tuned.dtype != "c128" or tuned.device != "dcu:0":
        problems.append(
            f"Options normalisation gave dtype={tuned.dtype!r}, "
            f"device={tuned.device!r}")
    for key, want in (("lattice_grid", (2, 1, 1, 1)), ("mg_grid", (2, 2, 2, 2)),
                      ("dtype", "c128"), ("device", "dcu:0"), ("mass", 0.17),
                      ("kappa", 0.1), ("max_iter", 100), ("restart", 30),
                      ("mg_levels", 2), ("mg_nu_pre", 1), ("mg_nu_post", 1),
                      ("threads", 4), ("device_id", 1), ("source", "ones"),
                      ("gauge", "unit"), ("csw", 0.9)):
        if tuned_kwargs.get(key) != want:
            problems.append(
                f"as_kwargs[{key!r}] {tuned_kwargs.get(key)!r} != {want!r}")

    selected = Test.default().select(
        lattice_grid="1,1,1,1", mg_grid=[2, 2, 2, 2], solver="cg",
        dslash="clover", device="npu", source="random", gauge="ones",
        gauge_sigma=0.2, mass=0.1, csw=0.5, max_iter=40, threads=2,
        device_id=0, dtype="float32", lattice="8,8,8,16")
    if selected.dtype != "c64":
        problems.append(f"select(dtype='float32') gave {selected.dtype!r}")
    if selected.device != "npu" or selected.config()["runner_device"] != "npu:0":
        problems.append(
            f"select(device='npu') gave {selected.device!r} / "
            f"{selected.config()['runner_device']!r}")
    for key, want in (("lattice_grid", (1, 1, 1, 1)), ("mg_grid", (2, 2, 2, 2)),
                      ("solver", "cg"), ("dslash", "clover"),
                      ("source", "random"), ("gauge", "ones"),
                      ("gauge_sigma", 0.2), ("mass", 0.1), ("csw", 0.5),
                      ("max_iter", 40), ("threads", 2), ("device_id", 0)):
        got = getattr(selected, key)
        if got != want:
            problems.append(f"select({key}={want!r}) gave {got!r}")
    tags = [case.tag for case in selected.cases]
    if any(tag.endswith("/bistabcg") or "/mg/" in tag for tag in tags):
        problems.append(f"--solver cg kept a BiCGStab/MG case: {tags}")
    if any(case.kind == "dslash" and not tag.endswith("/clover")
           for tag, case in zip(tags, selected.cases)):
        problems.append(f"--dslash clover kept a non-clover dslash: {tags}")

    # ``--check`` runs no numerics: an empty case list still builds the report
    # (and its options) through the same code path a real run uses.
    empty = replace(selected, cases=())
    report = empty.run(repeat=1, warmup=1, allow_all_skipped=True)
    payload = report.as_dict()
    options = payload["options"]
    missing = expected_keys - set(options)
    if missing:
        problems.append(
            f"TestReport.options is missing {sorted(missing)}")
    for key in ("lattice_grid", "mg_grid", "solver", "dslash", "device",
                "runner_device", "source", "gauge", "gauge_sigma", "csw",
                "mass", "max_iter", "threads", "device_id"):
        if options.get(key) != selected.config().get(key):
            problems.append(
                f"TestReport.options[{key!r}] {options.get(key)!r} != "
                f"{selected.config().get(key)!r}")
    if problems:
        return Check("run_config", "FAIL", "; ".join(problems))
    return Check(
        "run_config", "PASS",
        f"{len(CONFIG_FIELDS)} shared fields normalise, forward and report "
        "identically")


def _check_gate_evaluation() -> Check:
    """A wrong value or a missing key must fail; every operator must work."""
    scenarios: list[tuple[str, dict[str, Any], dict[str, Any], int]] = [
        ("boolean pass", {"su3": {"eq": True}}, {"su3": True}, 0),
        ("boolean type", {"su3": {"eq": True}}, {"su3": 1}, 1),
        ("missing", {"su3": {"eq": True}}, {}, 1),
        ("alias", {"relative_error": {"le": "compare"}},
         {"relative": 1e-9}, 0),
        ("too large", {"relative_error": {"le": "compare"}},
         {"relative_error": 1.0}, 1),
        ("field lt", {"iterations": {"gt": 0, "lt_field": "max_iter"}},
         {"iterations": 3, "max_iter": 10}, 0),
        ("field limit", {"iterations": {"gt": 0, "lt_field": "max_iter"}},
         {"iterations": 10, "max_iter": 10}, 1),
        ("field missing", {"iterations": {"gt": 0, "lt_field": "max_iter"}},
         {"iterations": 3}, 1),
        ("empty list", {"missing": {"empty": True}}, {"missing": []}, 0),
        ("non-empty list", {"missing": {"empty": True}}, {"missing": [1]}, 1),
        ("between", {"x": {"between": [0.0, 1.0]}}, {"x": 0.5}, 0),
        ("between out", {"x": {"between": [0.0, 1.0]}}, {"x": 2.0}, 1),
        ("membership", {"x": {"in": ("a", "b")}}, {"x": "a"}, 0),
        ("truthy", {"x": {"truthy": None}}, {"x": 0}, 1),
        ("not equal", {"variant": {"eq": "legacy"}}, {"variant": "strict"}, 1),
    ]
    problems = []
    for name, expect, observations, want in scenarios:
        failed = evaluate_expect(expect, observations, dtype="c64")
        if len(failed) != want:
            problems.append(f"{name}: expected {want} failure(s), got "
                            f"{len(failed)} ({'; '.join(failed) or 'none'})")
    # Unknown operators must be rejected when the case is defined.
    try:
        Case(tag="torch/x", backend="torch", kind="dslash",
             runner="torch_qcu:run", expect={"x": {"nope": 1}})
        problems.append("Case accepted an unknown gate operator")
    except ValueError:
        pass
    try:
        Case(tag="torch/x", backend="torch", kind="dslash",
             runner="torch_qcu:run", requires=("not_a_token",))
        problems.append("Case accepted an unknown skip token")
    except ValueError:
        pass
    if problems:
        return Check("gate_evaluation", "FAIL", "; ".join(problems))
    return Check("gate_evaluation", "PASS",
                 f"{len(scenarios)} gate scenarios behave as documented")


def _check_exit_codes() -> Check:
    """Aggregate status and exit codes: 0 clean, 1 failure, all-SKIP is 1."""
    scenarios: list[tuple[tuple[str, ...], bool, int]] = [
        ((), False, 1),
        (("PASS",), False, 0),
        (("PASS", "SKIP"), False, 0),
        (("SKIP",), False, 1),
        (("SKIP",), True, 0),
        (("FAIL",), False, 1),
        (("ERROR",), False, 1),
        (("TIMEOUT",), False, 1),
        (("PASS", "FAIL"), False, 1),
        (("PASS", "SKIP"), True, 0),
    ]
    problems = []
    for statuses, allow_skipped, want in scenarios:
        results = tuple(
            Result(tag=f"torch/{status.lower()}", backend="torch",
                   kind="dslash", status=status,
                   reason=("skipped" if status == "SKIP" else None))
            for status in statuses)
        counts, status = summarize(results)
        report = TestReport(
            status=status,
            lattice=policy.QUICK_LATTICE,
            dtype="c64",
            seed=policy.DEFAULT_SEED,
            cases=results,
            counts=counts,
            duration_seconds=0.0,
            allow_all_skipped=allow_skipped,
        )
        if report.exit_code() != want:
            problems.append(
                f"{statuses or 'empty'} (allow_all_skipped={allow_skipped}): "
                f"exit {report.exit_code()} != {want}")
    # An empty selection must never be reported as PASS.
    counts, status = summarize(())
    if status != "SKIP" or counts["PASS"] != 0:
        problems.append(f"summarize(()) = {status!r}/{counts!r}")
    if problems:
        return Check("exit_codes", "FAIL", "; ".join(problems))
    return Check("exit_codes", "PASS",
                 f"{len(scenarios)} status combinations map to 0/1 correctly")


def _check_runner_resolution() -> Check:
    """Runner strings must resolve lazily and relative to ``pyqcu.test``."""
    problems = []
    function = resolve_runner("policy:validate_lattice")
    if function is not policy.validate_lattice:
        problems.append("relative runner 'policy:validate_lattice' did not "
                        "resolve to pyqcu.test.policy.validate_lattice")
    try:
        resolve_runner("definitely_missing_runner:run")
        problems.append("a missing runner module did not raise")
    except ModuleNotFoundError as error:
        if "pyqcu.test.definitely_missing_runner" not in str(error):
            problems.append(f"missing module error is not qualified: {error}")
    except Exception as error:  # noqa: BLE001 - any other type is a defect
        problems.append(f"unexpected error for a missing runner: {error!r}")
    try:
        resolve_runner("no-colon")
        problems.append("a malformed runner string did not raise")
    except ValueError:
        pass
    if problems:
        return Check("runner_resolution", "FAIL", "; ".join(problems))
    return Check("runner_resolution", "PASS",
                 "module:function resolves lazily, relative to pyqcu.test")


def _check_timing_helpers() -> Check:
    """``timed``/``time_samples``/``relative_error`` must follow the contract."""
    problems = []
    calls = {"count": 0}

    def work() -> None:
        calls["count"] += 1

    samples = time_samples(work, repeat=5, warmup=2)
    if len(samples) != 5:
        problems.append(f"time_samples returned {len(samples)} samples")
    if calls["count"] != 7:
        problems.append(f"warmup+repeat called {calls['count']} times, expected 7")
    if timed(work, repeat=3, warmup=1) < 0.0:
        problems.append("timed returned a negative duration")
    for name, kwargs in (("warmup", {"warmup": 0}), ("repeat", {"repeat": 0})):
        try:
            time_samples(work, **kwargs)
            problems.append(f"time_samples accepted {name}=0")
        except ValueError:
            pass
    if relative_error([1.0, 2.0], [1.0, 2.0]) != 0.0:
        problems.append("relative_error of identical vectors is not 0")
    if abs(relative_error([1.1], [1.0]) - 0.1) > 1e-12:
        problems.append("relative_error scale is not max|expected|")
    if relative_error([1.0], [1.0, 2.0]) != float("inf"):
        problems.append("relative_error did not report a shape mismatch")
    if json_safe(float("nan")) != "NaN":
        problems.append("json_safe did not encode NaN as a string")
    if json_safe((1, 2.0)) != [1, 2.0]:
        problems.append("json_safe did not convert tuples to lists")
    result = Result(tag="torch/x", backend="torch", kind="dslash",
                    status="PASS", metrics={"relative_error": 1e-9})
    payload = json_safe(result)
    if not isinstance(payload, dict) or payload.get("tag") != "torch/x":
        problems.append("json_safe(Result) is not a plain dict")
    else:
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as error:
            problems.append(f"json_safe(Result) is not JSON serializable: {error}")
    if problems:
        return Check("timing_helpers", "FAIL", "; ".join(problems))
    return Check("timing_helpers", "PASS",
                 "warmup>=1, median timing, relative error and JSON safety")


def _check_extensions() -> Check:
    """Profiles, sweep planner and report exporters must match their contracts.

    These are the three expanded features the CLI wires: ``--profile`` (named
    option sets, explicit flags win), ``--sweep``/``--matrix`` (deterministic
    bounded expansion through ``Test.select``) and ``--junit``/``--csv``/
    ``--markdown`` (atomic exporters over the same report payload).  The check
    runs them in-process; the ``--list`` forms import no backend and the
    exporter sample is an empty report, so no numeric case runs here.
    """
    import contextlib
    import io
    import tempfile

    problems: list[str] = []

    # -- profiles -----------------------------------------------------------
    names = profiles.profile_names()
    missing = [name for name in
               ("quick", "cpu", "gpu", "c128", "clover", "mg", "full")
               if name not in names]
    if missing:
        problems.append(f"profiles missing {missing}; have {list(names)}")
    cpu = profiles.resolve_profile("cpu")
    if tuple(sorted(cpu)) != tuple(sorted(profiles.PROFILE_OPTIONS)):
        problems.append("resolve_profile('cpu') does not declare every "
                        "PROFILE_OPTIONS key")
    if cpu.get("backend") != ("torch", "kupl") or cpu.get("device") != "cpu":
        problems.append(f"profile cpu resolved to {cpu.get('backend')!r}/"
                        f"{cpu.get('device')!r}, expected torch+kupl/cpu")
    merged = profiles.merge_profile(
        {"dtype": "c64", "tier": 0, "only": ["torch/lattice"]}, "c128")
    if merged.get("dtype") != "c64":
        problems.append("merge_profile overwrote an explicit dtype (c128)")
    if merged.get("tier") != 0 or merged.get("only") != ["torch/lattice"]:
        problems.append("merge_profile changed a real 0 or a foreign key")
    zeros = profiles.merge_profile(
        {"csw": 0.0, "gauge_sigma": 0.0, "tier": 0}, "clover")
    if (zeros.get("csw"), zeros.get("gauge_sigma"), zeros.get("tier")) != (
            0.0, 0.0, 0):
        problems.append("merge_profile treated 0/0.0 as unset")
    try:
        profiles.resolve_profile("nope")
        problems.append("resolve_profile accepted an unknown name")
    except ValueError:
        pass

    # -- sweep planner ------------------------------------------------------
    spec = "dtype=c64;backend=torch;dslash=wilson"
    axes = matrix.parse_spec(spec)
    if axes != {"dtype": ("c64",), "backend": ("torch",),
                "dslash": ("wilson",)}:
        problems.append(f"parse_spec({spec!r}) -> {axes!r}")
    combinations = matrix.expand_matrix(axes)
    if combinations != [{"dtype": "c64", "backend": "torch",
                         "dslash": "wilson"}]:
        problems.append(f"expand_matrix({spec!r}) -> {combinations!r}")
    if matrix.expand_matrix("dtype=c64,complex64") != [{"dtype": "c64"}]:
        problems.append("expand_matrix kept a duplicate (c64,complex64)")
    if matrix.format_spec(axes) != spec:
        problems.append(f"format_spec round-trip -> {matrix.format_spec(axes)!r}")
    if matrix.select_kwargs({"backend": "cpu"}) != {
            "backends": ("torch", "kupl")}:
        problems.append("select_kwargs did not map backend -> backends")
    try:
        matrix.expand_matrix(
            "seed=" + ",".join(str(index)
                               for index in range(matrix.MAX_CASES + 1)))
        problems.append("expand_matrix accepted more than MAX_CASES rows")
    except ValueError:
        pass

    # -- report exporters ---------------------------------------------------
    sample = TestReport(
        status="SKIP",
        lattice=policy.QUICK_LATTICE,
        dtype=policy.QUICK_DTYPE,
        seed=policy.QUICK_SEED,
        cases=(),
        counts={status: 0 for status in policy.STATUSES},
        duration_seconds=0.0,
        allow_all_skipped=True,
    )
    for fmt in report.FORMATS:
        try:
            text = report.render(sample, fmt)
        except Exception as error:  # noqa: BLE001 - report the formatter
            problems.append(f"render({fmt}) raised {type(error).__name__}: "
                            f"{error}")
            continue
        if not text.strip():
            problems.append(f"render({fmt}) is empty")
        if not text.endswith("\n") or text.endswith("\n\n"):
            problems.append(f"render({fmt}) does not end with exactly one \\n")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        for fmt, suffix in (("json", ".json"), ("junit", ".xml"),
                            ("csv", ".csv"), ("markdown", ".md")):
            target = base / f"report{suffix}"
            try:
                written = report.write_report(target, sample, fmt)
            except Exception as error:  # noqa: BLE001 - report the writer
                problems.append(f"write_report({fmt}) raised "
                                f"{type(error).__name__}: {error}")
                continue
            if not written.is_file():
                problems.append(f"write_report({fmt}) did not create {target}")
                continue
            if written.read_text(encoding="utf-8") != report.render(sample, fmt):
                problems.append(f"write_report({fmt}) content differs from "
                                "render()")
        leftovers = sorted(path.name for path in base.iterdir()
                           if path.name.endswith(".tmp"))
        if leftovers:
            problems.append(f"write_report left temporary file(s) {leftovers}")
        try:
            report.write_report(base / "inferred.csv", sample)
        except Exception as error:  # noqa: BLE001 - suffix inference
            problems.append(f"write_report(.csv suffix) raised "
                            f"{type(error).__name__}: {error}")
        try:
            report.write_report(base / "unknown.out", sample)
            problems.append("write_report guessed a format for '.out'")
        except ValueError:
            pass

    # -- CLI wiring ---------------------------------------------------------
    from .__main__ import main

    def listing(argv: list[str]) -> tuple[int, dict[str, Any]]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(list(argv) + ["--list", "--json"])
        return code, json.loads(buffer.getvalue())

    code, payload = listing(["--profile", "cpu", "--only", "torch/lattice"])
    if code != 0:
        problems.append(f"--profile cpu --list exited {code}")
    if payload.get("profile") != ["cpu"]:
        problems.append("--list did not report the requested profile")
    if payload.get("backends") != ["torch", "kupl"] \
            or payload.get("device") != "cpu":
        problems.append("--profile cpu did not pin torch+kupl/cpu "
                        f"({payload.get('backends')!r}/{payload.get('device')!r})")
    if payload.get("profile_resolved", {}).get("dtype") != "c64":
        problems.append("--list profile_resolved is missing or wrong")
    if payload.get("count") != 1:
        problems.append(f"--profile cpu --only torch/lattice listed "
                        f"{payload.get('count')} cases")

    # ``--only`` accepts the comma-separated form (``a,b``), the
    # space-separated form (``a b``) and the repeated flag (``--only a
    # --only b``).  The comma form is the one that regressed (A43-F6): the CLI
    # handed one literal "a,b" pattern to Test.select, which rejected it as an
    # unknown tag.  All three spellings must select the same two cases and
    # report the same normalised filter in ``--list --json``.
    want_only = ["torch/lattice", "torch/gauge/random"]
    for label, argv in (
            ("comma", ["--only", "torch/lattice,torch/gauge/random"]),
            ("space", ["--only", "torch/lattice", "torch/gauge/random"]),
            ("repeated", ["--only", "torch/lattice",
                          "--only", "torch/gauge/random"])):
        only_code, only_payload = listing(argv)
        if only_code != 0:
            problems.append(f"--only {label} form --list exited {only_code}")
            continue
        if only_payload.get("tags_filter") != want_only:
            problems.append(
                f"--only {label} form listed tags "
                f"{only_payload.get('tags_filter')!r}, expected {want_only!r}")
        if only_payload.get("count") != len(want_only):
            problems.append(
                f"--only {label} form listed {only_payload.get('count')} "
                f"cases, expected {len(want_only)}")

    code, payload = listing(
        ["--sweep", spec, "--only", "torch/dslash/wilson"])
    if code != 0:
        problems.append(f"--sweep --list exited {code}")
    if payload.get("count") != 1 or payload.get("combinations") != 1:
        problems.append("--sweep --list did not emit one expanded row "
                        f"(count={payload.get('count')!r}, "
                        f"combinations={payload.get('combinations')!r})")
    rows = payload.get("cases") or []
    if not rows or rows[0].get("dslash") != "wilson":
        problems.append("--sweep --list row does not carry dslash=wilson")
    code, payload = listing(
        ["--profile", "cpu", "--sweep", "dtype=c64",
         "--only", "torch/lattice"])
    if code != 0 or not payload.get("cases") \
            or payload["cases"][0].get("device") != "cpu":
        problems.append("a sweep row did not inherit the effective profile "
                        "device")

    code, quick = listing(["--quick"])
    if code != 0:
        problems.append(f"--quick --list exited {code}")
    for key, want in (("tier", 0), ("dtype", "c64"), ("seed", 42),
                      ("repeat", 1), ("warmup", 1)):
        if quick.get(key) != want:
            problems.append(f"--quick {key} {quick.get(key)!r} != {want!r}")

    usage = policy.EXIT_CODES["usage"]
    for argv in (
            ["--profile", "nope", "--list"],
            ["--sweep", "dtype=nope", "--list"],
            ["--quick", "--dtype", "c128", "--list"],
            ["--quick", "--profile", "c128", "--list"],
            ["--list", "--csv", "unused.csv"],
            ["--json", "-", "--csv", "-", "--only", "torch/lattice"],
            ["--check", "--sweep", "dtype=c64"],
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(list(argv))
        if code != usage:
            problems.append(f"{' '.join(argv)} did not exit {usage} "
                            f"(got {code})")

    if problems:
        return Check("extensions", "FAIL", "; ".join(problems))
    return Check(
        "extensions", "PASS",
        f"{len(names)} profiles, {len(matrix.AXIS_NAMES)} sweep axes and "
        f"{len(report.FORMATS)} report formats wired")


def _check_runner_availability() -> Check:
    """Every declared ``Case.runner`` must resolve to a callable now.

    ``units_contract`` compares the runner *strings* against
    ``policy.TIER0_RUNNERS``; it cannot see a runner that the module never
    defined.  Resolving each string is the check that catches a
    declared-but-missing runner (A15 D2/D7), so ``--check`` fails instead of
    reporting green over an unrunnable case.  All three runner modules are
    import-light by contract; resolving them here loads no backend.
    """
    from .common import resolve_runner

    missing: list[str] = []
    modules: set[str] = set()
    for case in units.CASES:
        module_name = case.runner.split(":", 1)[0]
        modules.add(module_name)
        try:
            function = resolve_runner(case.runner)
        except Exception as error:  # noqa: BLE001 - report the defect
            missing.append(f"{case.tag} -> {case.runner} "
                           f"({type(error).__name__}: {error})")
            continue
        if not callable(function):
            missing.append(f"{case.tag} -> {case.runner} is not callable")
    if missing:
        return Check("runner_availability", "FAIL", "; ".join(missing))
    return Check(
        "runner_availability", "PASS",
        f"{len(units.CASES)} declared runners resolve to callables in "
        f"{', '.join(sorted(modules))}")


#: Ordered check registry.
_CHECKS: tuple[tuple[str, Callable[[], Check]], ...] = (
    ("import_is_stdlib_only", _check_import_is_stdlib_only),
    ("cli_list_light", _check_cli_list_light),
    ("cli_effective_config", _check_cli_effective_config),
    ("cli_usage_codes", _check_cli_usage_codes),
    ("cli_flags", _check_cli_flags),
    ("run_config", _check_run_config),
    ("source_imports", _check_source_imports),
    ("no_test_modules", _check_no_test_modules),
    ("package_inclusion", _check_package_inclusion),
    ("manifest", _check_manifest),
    ("extensions", _check_extensions),
    ("policy_constants", _check_policy_constants),
    ("units_contract", _check_units_contract),
    ("gate_evaluation", _check_gate_evaluation),
    ("exit_codes", _check_exit_codes),
    ("runner_resolution", _check_runner_resolution),
    ("runner_availability", _check_runner_availability),
    ("timing_helpers", _check_timing_helpers),
)


def run_checks() -> list[Check]:
    """Run every check; an unexpected exception is a FAIL, never a traceback."""
    checks: list[Check] = []
    for name, function in _CHECKS:
        try:
            checks.append(function())
        except Exception as error:  # noqa: BLE001 - a check never escapes
            checks.append(Check(name, "FAIL",
                                f"{type(error).__name__}: {error}"))
    return checks


def _tail(text: str, limit: int = 400) -> str:
    """Last ``limit`` characters of ``text``, whitespace collapsed."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return "..." + collapsed[-limit:]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyqcu.test.selfcheck",
        description="Contract self-check for the formal pyqcu.test port.",
    )
    parser.add_argument(
        "--json", nargs="?", const="-", default=None, metavar="PATH",
        help="emit the checks as one JSON object: PATH, or '-' for stdout "
             "(also the value when the flag is given without a path)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the self-check; return 0 when nothing failed, 1 otherwise."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    checks = run_checks()
    failed = [check for check in checks if not check.ok]
    status = "FAIL" if failed else "PASS"
    if args.json is not None:
        payload = {
            "package": policy.TEST_PACKAGE,
            "version": __version__,
            "status": status,
            "exit_code": 1 if failed else 0,
            "checks": [
                {"name": check.name, "status": check.status,
                 "detail": check.detail}
                for check in checks
            ],
            "failed": len(failed),
        }
        if args.json == "-":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            try:
                write_json_atomic(args.json, payload)
            except (OSError, TypeError, ValueError) as error:
                print(f"pyqcu.test.selfcheck: error: cannot write "
                      f"{args.json!r}: {type(error).__name__}: {error}",
                      file=sys.stderr)
                return 1
        return 1 if failed else 0
    print(f"pyqcu.test {__version__} self-check ({PACKAGE_DIR})")
    for check in checks:
        print(f"  {check.status:<4} {check.name:<22} {check.detail}")
    skipped = sum(1 for check in checks if check.status == "SKIP")
    print(f"selfcheck: {len(checks)} checks, {len(checks) - len(failed) - skipped} "
          f"passed, {skipped} skipped, {len(failed)} failed")
    print(f"overall: {status}")
    return 1 if failed else 0


def test_selfcheck() -> None:
    """Pytest entry point (``pytest pyqcu/test/selfcheck.py``).

    Kept here rather than in a ``test_*.py`` file so the shipped package stays
    free of repository-only test layout; no ``pytest`` import is needed.
    """
    assert main([]) == 0


def test_policy_constants() -> None:
    """Pytest entry point for the two pure-Python contract checks."""
    checks = (_check_policy_constants(), _check_units_contract())
    failures = [f"{check.name}: {check.detail}" for check in checks if not check.ok]
    assert not failures, failures


if __name__ == "__main__":
    raise SystemExit(main())
