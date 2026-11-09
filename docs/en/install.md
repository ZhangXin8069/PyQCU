# Install

## Requirements

* Python 3.10 or newer (tested with 3.12).
* Linux on x86_64 or aarch64.
* `numpy`, `torch`, `mpi4py`, and `h5py` are installed automatically as
  runtime dependencies.
* A C++17 compiler, CMake, and an MPI implementation are needed to build the
  KUPL CPU library. CUDA additionally needs `nvcc`; DCU needs a DTK
  `nvcc`-compatible compiler.

```bash
pip install pyqcu
```

The package is published as a source archive only. The first install compiles
the native KUPL library when a toolchain is available; later installs reuse
pip's wheel cache.

## Optional extras

| Extra | Adds | Use |
|---|---|---|
| `pyqcu[plot]` | `matplotlib` | Benchmark and convergence plots |
| `pyqcu[monitor]` | `psutil` | Memory/process monitoring helpers |
| `pyqcu[all]` | both | Convenience for interactive work |

`torch` is a hard dependency because the Torch backend and the public lattice
helpers import it. Choose the CPU or CUDA build of Torch that matches the
machine before installing PyQCU.

## What the installer does

1. Reads PEP 621 metadata from `pyproject.toml`.
2. Tries to build the KUPL CPU library (`cpp/kupl/qcu`) with CMake, MPI, and
   OpenMP.
3. Tries the CUDA library when `nvcc` is available and the backend is
   `cuda`, or the DCU library when the backend is `dcu`.
4. Builds the Cython bridge `pyqcu.cuda.qcu` only when the CUDA/DCU library
   was produced.
5. Copies the produced libraries into `pyqcu/kupl/libqcu.so` and
   `pyqcu/cuda/libqcu.so`, so the installed package does not depend on the
   source tree or `LD_LIBRARY_PATH`.

A missing toolchain is not fatal: the install still succeeds and the pure
Torch paths remain usable.

## Native build variables

| Variable | Default | Meaning |
|---|---|---|
| `PYQCU_BUILD_BACKEND` | `cuda` | `kupl`, `cuda`, or `dcu`; controls the GPU attempt |
| `PYQCU_SKIP_NATIVE_BUILD` | unset | `1` skips every native build |
| `PYQCU_REQUIRE_NATIVE` | unset | `1` turns a missing requested backend into a hard error |
| `PYQCU_KUPL_BUILD_DIR` | `cpp/kupl/qcu/build` | KUPL CMake build directory |
| `PYQCU_KUPL_LIB` | package library | Explicit KUPL runtime library |
| `KUPL_BUILD_TESTS` | `OFF` for pip builds | Build the KUPL CTest executables |
| `KUPL_ENABLE_NATIVE` | `OFF` | Compile for the local CPU architecture |
| `KUPL_ENABLE_LTO` | `OFF` | Link-time optimization |

Examples:

```bash
# CPU-only: build KUPL and skip the GPU attempt
PYQCU_BUILD_BACKEND=kupl pip install pyqcu

# Require the KUPL build to succeed
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu

# Build both KUPL and CUDA on a machine with nvcc
PYQCU_BUILD_BACKEND=cuda pip install pyqcu
```

## Verify the installation

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

`pyqcu` is the installed console script; `python -m pyqcu` is equivalent.
`pyqcu --selftest` runs the Torch/KUPL/CUDA layers and reports a missing
native backend as `SKIP`.

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--json"],
    check=True, capture_output=True, text=True).stdout)
assert report["package"] == "pyqcu"
assert report["version"] == "0.1.2"
assert "path" in report["backends"]["kupl"]
print("pyqcu", report["version"], "at", report["path"])
```

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--selftest", "--json",
     "--lattice", "4", "4", "4", "4"],
    check=True, capture_output=True, text=True).stdout)
assert report["status"] == "PASS"
assert report["layers"]["torch"]["status"] == "PASS"
print("self-test:", {
    name: layer["status"] for name, layer in report["layers"].items()})
```

```python
import importlib.util
import pyqcu

assert pyqcu.__version__ == "0.1.2"
missing = [
    name for name in
    ("api", "lattice", "tools", "torch", "solver", "cuda", "kupl")
    if importlib.util.find_spec(f"pyqcu.{name}") is None
]
assert not missing, missing
print("all public subpackages importable")
```

If the KUPL report says `FileNotFoundError`, the native library was not built.
Either install a toolchain and reinstall, or set `PYQCU_KUPL_LIB` to an
existing `libqcu.so`.

## From a source checkout

```bash
git clone git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git
cd PyQCU
source ./env.sh

# Pure KUPL CPU backend
bash ./install.sh kupl

# CUDA backend plus the Cython bridge
bash ./install.sh cuda

# DCU/DTK backend
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu
```

The source checkout exposes additional test and benchmark drivers under
`pyqcu/testing/`; those are intentionally excluded from the PyPI package.

## Tests from a source checkout

```bash
cd pyqcu/testing
pytest .
```

MPI-specific tests are started explicitly, for example:

```bash
mpirun -np 4 python pyqcu/testing/python/conftest.py
```

## Backends and array types

| Input / build | Execution path |
|---|---|
| Torch tensor, CPU | `pyqcu.torch` or KUPL when explicitly called |
| Torch tensor, CUDA | `pyqcu.torch` CUDA kernels or the CUDA bridge |
| NumPy host array | KUPL `ctypes` ABI |
| No native library | Torch backend only |

## Troubleshooting

* **`cmake` or `mpi4py` missing**: install CMake and an MPI implementation, or
  set `PYQCU_SKIP_NATIVE_BUILD=1` for a pure Torch installation.
* **CUDA import error**: check `python -m pyqcu --json`; the CUDA bridge is
  built only when `nvcc` and the CUDA library are available.
* **Wrong version imported**: run from outside the source tree and assert
  `pyqcu.__file__` contains `site-packages`.
* **`/tmp` full**: pip builds in a temporary directory; use a larger `TMPDIR`
  or clean old virtual environments.
