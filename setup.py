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
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

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
    return True


def _build_gpu() -> bool:
    kind = "dcu" if BUILD_BACKEND == "dcu" else "cuda"
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
    library = HOME / "cpp" / "cuda" / "qcu" / "libqcu.so"
    if not library.is_file():
        _warn(f"{kind.upper()} library was not produced: {library}")
        return False
    return True


def _copy_library(source: Path, destination: Path) -> bool:
    source = source.resolve()
    if not source.is_file():
        _warn(f"native library is missing: {source}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, staged)
    os.replace(staged, destination)
    return True


def _install_native_libraries() -> None:
    if _NATIVE["kupl"]:
        _copy_library(_kupl_built_library(),
                      HOME / "pyqcu" / "kupl" / "libqcu.so")
    if _NATIVE["gpu"]:
        _copy_library(
            HOME / "cpp" / "cuda" / "qcu" / "libqcu.so",
            HOME / "pyqcu" / "cuda" / "libqcu.so")


def _build_native_once() -> None:
    if _NATIVE["done"]:
        return
    _NATIVE["done"] = True

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
    """Build the native libraries before package data is collected."""

    def run(self):
        _build_native_once()
        super().run()


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


setup(
    packages=find_packages(
        include=["pyqcu", "pyqcu.*"],
        exclude=["pyqcu.testing", "pyqcu.testing.*"],
    ),
    package_data={
        "pyqcu.kupl": ["libqcu.so"],
        "pyqcu.cuda": ["libqcu.so", "qcu.pyi", "qcu/*.pyi", "qcu/*.pxd"],
    },
    ext_modules=(
        [Extension("pyqcu.cuda.qcu", ["pyqcu/cuda/qcu/qcu.pyx"])]
        if BUILD_BACKEND in GPU_BUILD_BACKENDS
        else []
    ),
    cmdclass={"build_py": NativeBuildPy, "build_ext": NativeBuildExt},
    zip_safe=False,
)
