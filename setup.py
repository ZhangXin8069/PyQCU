"""Build entry for the PyQCU native layers.

Project metadata (name, version, dependencies, ...) lives in
``pyproject.toml``. This file supplies the build mechanics that PEP 621
cannot express: the KUPL CPU library, the optional CUDA/DCU library, and the
optional Cython bridge.

Backend policy
--------------
* ``PYQCU_BUILD_BACKEND=kupl`` -> KUPL only.
* ``PYQCU_BUILD_BACKEND=cuda`` (default) -> KUPL, then CUDA when the toolchain
  is available.
* ``PYQCU_BUILD_BACKEND=dcu`` -> KUPL, then the DTK/ROCm-compatible library.

The KUPL build is attempted on every machine with CMake, a C++ compiler and
MPI. A failure degrades the install to the pure-Torch paths and only becomes
fatal with ``PYQCU_REQUIRE_NATIVE=1``. CUDA/DCU builds are always
opportunistic: a missing ``nvcc`` or a failed compile leaves the KUPL backend
usable and never aborts the install.

Packaging integrity
-------------------
A wheel may only carry the library this invocation compiled.

* ``package_data`` lists no ``*.so``.  :class:`NativeBuildPy` copies the build
  products into ``build_lib``, and ``install_lib`` feeds a wheel from there,
  so a library that merely sits in ``pyqcu/<backend>/`` is unreachable from
  the packaging code.
* Every ``pyqcu/*/libqcu.so`` that this invocation did not produce is deleted,
  on the skip path and on the failure path alike, and packaging fails if such
  a file survives.  ``pyqcu/kupl/qcu.py`` dlopen()s that path through ctypes,
  so a leftover would be executed, not merely shipped.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import NamedTuple

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

HOME = Path(__file__).resolve().parent
os.chdir(HOME)
if str(HOME) not in sys.path:
    sys.path.insert(0, str(HOME))

from pyqcu._build_backend import GPU_BUILD_BACKENDS, normalize_build_backend

REQUESTED_BACKEND = os.environ.get("PYQCU_BUILD_BACKEND")
BUILD_BACKEND = normalize_build_backend(REQUESTED_BACKEND)
SKIP_NATIVE = os.environ.get("PYQCU_SKIP_NATIVE_BUILD") == "1"
REQUIRE_NATIVE = os.environ.get("PYQCU_REQUIRE_NATIVE") == "1"
print(f"PYQCU build backend: {BUILD_BACKEND}")

_NATIVE = {"kupl": False, "gpu": False, "done": False}
_WARNED: set[str] = set()
# Native library this invocation compiled -> its sha256.  Keyed by backend
# label ("kupl", "cuda"/"dcu"); empty means "nothing was built here".
_BUILT: dict[str, tuple[Path, str]] = {}
# Every ``libqcu.so`` this invocation wrote -> its sha256.  Anything found at
# one of those paths with one of those digests is ours; anything else is a
# leftover and must not be packaged.
_STAGED: dict[Path, str] = {}


def _notify_progress(message: str) -> None:
    """Write build progress where pip cannot swallow it."""
    line = f"[pyqcu build] {message}\n"
    try:
        with open("/dev/tty", "w") as terminal:
            terminal.write(line)
            terminal.flush()
    except OSError:
        sys.stderr.write(line)
        sys.stderr.flush()


def _warn(message: str) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    warnings.warn(f"[PyQCU build] {message}", RuntimeWarning, stacklevel=2)
    # pip captures stderr on a successful build; write through the same
    # /dev/tty channel as the progress lines so the user actually sees a
    # missing CUDA/DCU backend.
    _notify_progress(message)


def _relative(path: Path) -> str:
    """Path as the user sees it: relative to the checkout when it is inside."""
    try:
        return str(path.relative_to(HOME))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _toolchain_problem(backend: str) -> str | None:
    if shutil.which("bash") is None:
        return "bash is not available on PATH"
    if shutil.which("cmake") is None:
        return "cmake is not available on PATH"
    host = os.environ.get("CXX", "c++")
    if shutil.which(host) is None:
        return f"host C++ compiler {host!r} is not available on PATH"
    if backend == "cuda":
        nvcc = os.environ.get("NVCC", "nvcc")
        if shutil.which(nvcc) is None:
            return f"CUDA compiler {nvcc!r} is not available on PATH"
        if not _host_compiler_works():
            return f"host C++ compiler {host!r} cannot compile a trivial program"
    elif backend == "dcu":
        compiler = os.environ.get("QCU_DCU_CUDA_COMPILER")
        if compiler and not Path(compiler).is_file() and shutil.which(compiler) is None:
            return f"DCU compiler {compiler!r} is not available"
    return None


def _host_compiler_works() -> bool:
    compiler = os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        return False
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.cpp"
        source.write_text("int main() { return 0; }\n")
        try:
            subprocess.run(
                [compiler, str(source), "-o", str(Path(tmp) / "probe")],
                check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            return False
    return True


def _run_build(command: list[str], backend: str, environment: dict[str, str]) -> bool:
    started = time.time()
    _notify_progress(
        f"compiling the native {backend} library (pip stays silent until it finishes)")
    result = subprocess.run(
        command, cwd=str(HOME), env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-12:])
        _warn(
            f"native {backend} build failed ({' '.join(command)} exited "
            f"with {result.returncode}).\n{tail}")
        return False
    _notify_progress(
        f"native {backend} library built in {time.time() - started:.1f}s")
    return True


def _kupl_build_directory() -> Path:
    configured = os.environ.get("PYQCU_KUPL_BUILD_DIR")
    return (Path(configured).expanduser() if configured
            else HOME / "cpp" / "kupl" / "qcu" / "build").resolve()


def _kupl_built_library() -> Path:
    return _kupl_build_directory() / "libqcu.so"


def _gpu_label() -> str:
    return "dcu" if BUILD_BACKEND == "dcu" else "cuda"


def _gpu_built_library() -> Path:
    return HOME / "cpp" / "cuda" / "qcu" / "libqcu.so"


class _StagedLibrary(NamedTuple):
    """One native library: what the build writes and where it is staged.

    ``source`` is produced by this invocation's build and never by packaging.
    ``destination`` is the copy next to the Python package, which is what
    ``ctypes`` (KUPL) and the ``$ORIGIN`` rpath (CUDA) load at run time; it is
    written for in-place/editable use and is the path that must never hold a
    library this invocation did not produce.
    """

    label: str
    source: Path
    destination: Path


def _staged_libraries() -> tuple[_StagedLibrary, ...]:
    """The two (label, source, destination) triples, in ``_NATIVE`` order."""
    return (
        _StagedLibrary("kupl", _kupl_built_library(),
                       HOME / "pyqcu" / "kupl" / "libqcu.so"),
        _StagedLibrary(_gpu_label(), _gpu_built_library(),
                       HOME / "pyqcu" / "cuda" / "libqcu.so"),
    )


def _produced_libraries() -> list[_StagedLibrary]:
    """The libraries this invocation compiled."""
    return [item for item, ready
            in zip(_staged_libraries(), (_NATIVE["kupl"], _NATIVE["gpu"]))
            if ready]


def _build_kupl() -> bool:
    problem = _toolchain_problem("kupl")
    if problem is not None:
        _warn(f"KUPL backend not built: {problem}")
        return False
    environment = os.environ.copy()
    environment["PYQCU_KUPL_BUILD_DIR"] = str(_kupl_build_directory())
    # The sdist prunes cpp/kupl/qcu/tests; an inherited ON would make CMake
    # fail on a missing test source. Python builds never need the test targets.
    environment["KUPL_BUILD_TESTS"] = "OFF"
    environment.setdefault("KUPL_ENABLE_NATIVE", "OFF")
    environment.setdefault("KUPL_ENABLE_LTO", "OFF")
    if not _run_build(["bash", "cpp/kupl/qcu/make.sh"], "KUPL", environment):
        return False
    library = _kupl_built_library()
    if not library.is_file():
        _warn(f"KUPL library was not produced: {library}")
        return False
    _BUILT["kupl"] = (library, _sha256(library))
    return True


def _build_gpu() -> bool:
    kind = _gpu_label()
    problem = _toolchain_problem(kind)
    if problem is not None:
        _warn(f"{kind.upper()} backend not built: {problem}")
        return False
    script = "build-dcu.sh" if kind == "dcu" else "build-nv.sh"
    if not (HOME / script).is_file():
        _warn(f"{kind.upper()} backend not built: {script} is missing")
        return False
    environment = os.environ.copy()
    if not _run_build(["bash", script], kind.upper(), environment):
        return False
    library = _gpu_built_library()
    if not library.is_file():
        _warn(f"{kind.upper()} library was not produced: {library}")
        return False
    _BUILT[kind] = (library, _sha256(library))
    return True


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    """Copy ``source`` to ``destination``, refusing to write other bytes."""
    if not source.is_file():
        raise RuntimeError(f"native library vanished before staging: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, pending)
        digest = _sha256(pending)
        if digest != expected:
            raise RuntimeError(
                f"staging {source} -> {_relative(destination)} changed the "
                f"content (sha256 {digest} != {expected}); refusing to write "
                f"an unverified native library")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
    _STAGED[destination] = expected


def _stage_library(item: _StagedLibrary, destination: Path) -> str:
    """Copy the library this invocation built to ``destination``."""
    built = _BUILT.get(item.label)
    if built is None:
        raise RuntimeError(
            f"refusing to write {_relative(destination)}: this build did not "
            f"produce the {item.label} library")
    source, digest = built
    _copy_verified(source, destination, digest)
    return digest


def _install_native_libraries() -> None:
    """Stage this run's libraries next to the Python packages.

    This is the in-place/editable layout only: packaging reads the build tree
    (see :class:`NativeBuildPy`), never these copies.
    """
    for item in _produced_libraries():
        digest = _stage_library(item, item.destination)
        _notify_progress(f"staged {_relative(item.destination)} (sha256 {digest})")


def _libqcu_copies(*roots: Path) -> list[Path]:
    """Every ``pyqcu/<backend>/libqcu.so`` under ``roots``."""
    copies: list[Path] = []
    for root in roots:
        copies.extend(entry.resolve()
                      for entry in sorted(Path(root).glob("pyqcu/*/libqcu.so")))
    return copies


def _remove_foreign_libraries(*roots: Path) -> list[Path]:
    """Delete every ``libqcu.so`` this invocation did not produce.

    Runs on the skip path, on the failure path and again before a wheel is
    written.  ``build_lib`` is included because it survives between
    invocations: a stale copy there would be installed into an otherwise clean
    wheel even when the source tree is clean.
    """
    removed: list[Path] = []
    for copy in _libqcu_copies(*roots):
        digest = _sha256(copy)
        if _STAGED.get(copy) == digest:
            continue
        copy.unlink()
        removed.append(copy)
        _warn(f"removed {_relative(copy)} (sha256 {digest}): this build did "
              f"not produce it, so it must not be packaged or loaded")
    return removed


def _verify_packaged_libraries(*roots: Path) -> None:
    """Fail the build if a library this invocation did not produce is present.

    :func:`_remove_foreign_libraries` normally leaves nothing to find; this
    pass is the fail-closed guard for a copy that could not be deleted
    (read-only checkout, bind mount) or for any future path that stages a
    library without recording it in ``_STAGED``.
    """
    for copy in _libqcu_copies(*roots):
        digest = _sha256(copy)
        if _STAGED.get(copy) == digest:
            continue
        produced = ", ".join(
            f"{_relative(path)} sha256 {expected}"
            for path, expected in _STAGED.items()) or "nothing"
        raise RuntimeError(
            f"{_relative(copy)} has sha256 {digest}, which this build did not "
            f"produce (produced: {produced}). Refusing to package a native "
            f"library that came from somewhere else")


def _build_native_libraries() -> None:
    if SKIP_NATIVE:
        _warn("PYQCU_SKIP_NATIVE_BUILD=1: no native backend will be built")
        if REQUIRE_NATIVE:
            raise RuntimeError(
                "PYQCU_REQUIRE_NATIVE=1 conflicts with "
                "PYQCU_SKIP_NATIVE_BUILD=1")
        return

    _NATIVE["kupl"] = _build_kupl()
    if BUILD_BACKEND in GPU_BUILD_BACKENDS:
        _NATIVE["gpu"] = _build_gpu()

    if REQUIRE_NATIVE:
        if BUILD_BACKEND == "kupl" and not _NATIVE["kupl"]:
            raise RuntimeError(
                "PYQCU_REQUIRE_NATIVE=1 but the KUPL backend was not built")
        if BUILD_BACKEND in GPU_BUILD_BACKENDS and not _NATIVE["gpu"]:
            raise RuntimeError(
                f"PYQCU_REQUIRE_NATIVE=1 but the {BUILD_BACKEND} backend was "
                "not built")

    _install_native_libraries()
    if not _NATIVE["kupl"] and not _NATIVE["gpu"]:
        _warn(
            "installing PyQCU without a native backend; only the pure-Torch "
            "paths will be usable")


def _build_native_once() -> None:
    if _NATIVE["done"]:
        return
    _NATIVE["done"] = True
    try:
        _build_native_libraries()
    finally:
        # Unconditional: this covers the PYQCU_SKIP_NATIVE_BUILD=1 return, a
        # failed compile and a refused build.  The old guard lived inside the
        # copy step, which only ran after a successful build, so both failure
        # paths kept a previous session's library exactly where the package
        # loader and the wheel builder look for it.
        _remove_foreign_libraries(HOME)


def _cuda_extension() -> Extension:
    import numpy

    return Extension(
        "pyqcu.cuda.qcu",
        ["pyqcu/cuda/qcu/qcu.pyx"],
        depends=[
            "pyqcu/cuda/qcu/qcu.pxd",
            "pyqcu/cuda/qcu/qcu_api.pxd",
            "cpp/cuda/qcu/python/pyqcu.h",
        ],
        include_dirs=[
            "pyqcu/cuda/qcu",
            "cpp/cuda/qcu/python",
            numpy.get_include(),
        ],
        library_dirs=["cpp/cuda/qcu"],
        runtime_library_dirs=["$ORIGIN"],
        libraries=["qcu"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-Wl,-rpath,$ORIGIN"],
    )


def _cythonized_extensions() -> list[Extension]:
    from Cython.Build import cythonize

    return cythonize(
        [_cuda_extension()],
        build_dir="build/cython",
        compiler_directives={"language_level": 3},
        force=True,
    )


class NativeBuildPy(build_py):
    """Stage this run's native libraries into the build tree.

    ``package_data`` deliberately lists no ``*.so``: the wheel content is
    copied here from the build products, so nothing that happens to sit in
    ``pyqcu/<backend>/`` can reach a wheel at all.
    """

    def run(self):
        _build_native_once()
        super().run()
        build_lib = Path(self.build_lib).resolve()
        for item in _produced_libraries():
            destination = build_lib / item.destination.relative_to(HOME)
            digest = _stage_library(item, destination)
            _notify_progress(f"packaged {_relative(destination)} "
                             f"(sha256 {digest})")
        _remove_foreign_libraries(HOME, build_lib)
        _verify_packaged_libraries(HOME, build_lib)


class NativeBuildExt(build_ext):
    """Build the optional CUDA/DCU bridge only when its library exists."""

    def run(self):
        _build_native_once()
        if not _NATIVE["gpu"]:
            self.extensions = []
        else:
            extensions = _cythonized_extensions()
            for extension in extensions:
                # finalize_options() ran before we replaced the list; setuptools
                # expects this attribute on every extension at build time.
                extension._needs_stub = False
            self.extensions = extensions
        super().run()


try:
    # setuptools >= 70.1 ships the command; older trees import it from wheel.
    from setuptools.command.bdist_wheel import bdist_wheel as _BdistWheel
except ImportError:  # pragma: no cover - depends on the installed setuptools
    try:
        from wheel.bdist_wheel import bdist_wheel as _BdistWheel
    except ImportError:
        _BdistWheel = None


if _BdistWheel is not None:

    class NativeBdistWheel(_BdistWheel):
        """Tag a wheel that carries a native library as platform-specific.

        ``ext_modules`` is empty for a KUPL-only build, so
        ``Distribution.has_ext_modules()`` is False and bdist_wheel would
        write ``py3-none-any`` for a wheel that holds an ELF ``libqcu.so``.
        Purity is frozen in ``finalize_options`` before the build runs, so the
        decision is taken in ``get_tag()``: that method runs after
        ``build_py`` (which fills ``_BUILT``) and again when the ``WHEEL``
        metadata is written, so the filename and the ``Tag:`` lines agree.
        """

        def get_tag(self):
            if self.root_is_pure and _BUILT:
                self.root_is_pure = False
            return super().get_tag()

        def run(self):
            if self.skip_build:
                # --skip-build never runs build_py, so anything in the build
                # tree was compiled by an earlier invocation: drop it instead
                # of shipping code this run did not build.
                build_lib = Path(
                    self.get_finalized_command("build").build_lib).resolve()
                _remove_foreign_libraries(build_lib)
                _verify_packaged_libraries(build_lib)
            super().run()


_CMDCLASS = {"build_py": NativeBuildPy, "build_ext": NativeBuildExt}
if _BdistWheel is not None:
    _CMDCLASS["bdist_wheel"] = NativeBdistWheel

setup(
    packages=find_packages(
        include=["pyqcu", "pyqcu.*"],
        exclude=["pyqcu.testing", "pyqcu.testing.*"],
    ),
    package_data={
        # No "*.so" here on purpose: NativeBuildPy copies the libraries this
        # build produced from the build tree into build_lib.  Listing them
        # would let any file that happens to sit in pyqcu/<backend>/ be
        # packaged, which is exactly the defect this rules out.
        "pyqcu.cuda": ["qcu.pyi", "qcu/*.pyi", "qcu/*.pxd"],
    },
    ext_modules=(
        [Extension("pyqcu.cuda.qcu", ["pyqcu/cuda/qcu/qcu.pyx"])]
        if BUILD_BACKEND in GPU_BUILD_BACKENDS
        else []
    ),
    cmdclass=_CMDCLASS,
    zip_safe=False,
)
