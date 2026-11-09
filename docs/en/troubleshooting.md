# Troubleshooting

## First response: run the self-test

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--json"],
    check=True, capture_output=True, text=True).stdout)
assert report["package"] == "pyqcu"
assert "backends" in report
print("diagnostics OK", report["version"])
```

`pyqcu --selftest` runs the Torch, KUPL, and CUDA layers. `SKIP` means the
backend was not built or no device is visible; `FAIL` means the layer ran and
its independent residual check failed. Use `--json` for a machine-readable
report and `--lattice X Y Z T` to choose another even lattice. Extents below
4 are rejected; the CUDA native path is not validated at `2^4`, and release
verification uses `8x8x16x16`.

## Install and build

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'Cython'` | `--no-build-isolation` without Cython/numpy in the environment | Remove `--no-build-isolation`, or install `Cython` and `numpy` first |
| `cmake` not found | KUPL needs CMake | Install CMake or set `PYQCU_SKIP_NATIVE_BUILD=1` for a Torch-only install |
| MPI headers/library not found | KUPL needs an MPI implementation | Install MPI, or use `PYQCU_SKIP_NATIVE_BUILD=1` |
| CUDA build skipped | `nvcc` is missing or the host compiler cannot compile | Install the CUDA toolkit or use `PYQCU_BUILD_BACKEND=kupl` |
| `pip install` looks frozen | Native compilation is silent under pip | Wait; progress is written to `/dev/tty` when available |
| `No space left on device` | `/tmp` is a small tmpfs | Set `TMPDIR` to a larger disk before install |

To make a missing native backend a hard error:

```bash
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu
```

## Native backend

| Symptom | Meaning | Fix |
|---|---|---|
| `pyqcu` reports KUPL `FileNotFoundError` | The package has no `libqcu.so` | Build with a toolchain, or set `PYQCU_KUPL_LIB=/abs/path/libqcu.so` |
| CUDA bridge is not built | `nvcc`/CUDA library was unavailable at install time | Reinstall with `PYQCU_BUILD_BACKEND=cuda` on a CUDA machine |
| CUDA import fails after a successful build | The library was built for a different architecture | Rebuild on the target GPU or embed a compatible PTX architecture |
| `PYQCU_KUPL_LIB` points to an old library | A stale environment value wins | Unset it and let the package-local library resolve |

The KUPL lookup order is `PYQCU_KUPL_LIB`, `pyqcu/kupl/libqcu.so`,
`pyqcu/kupl/build/libqcu.so`, then the source-checkout path. The CUDA bridge
uses an `$ORIGIN` rpath and expects `pyqcu/cuda/libqcu.so`.

## Solver

| Symptom | Likely cause | Check |
|---|---|---|
| `solve_result` fields are `None` | Torch Legacy does not expose them | Use `mg.implementation.last_solve_report`, `convergence_history`, or an independent residual |
| CUDA `iterations=None` | CUDA Wilson has no iteration getter | Use `converged` and the residual; Clover has a getter |
| Residual is above the c64 acceptance | Wrong scope or single-precision floor | Check `residual_scope`; the c64 floor is `max(atol, 2048*eps32) ≈ 2.4414e-4` |
| `no Legacy levels` | `min_size` is larger than the local extent | Lower `min_size` or use a larger local lattice |
| `coarse_assets ... does not match` | Cache identity, shape, or precision mismatch | Rebuild the cache with the same lattice/block/precision |
| `process_grid ... does not match MPI size` | Wrong rank grid | Make `prod(process_grid) == MPI.COMM_WORLD.Get_size()` |

For multigrid details, see {doc}`solvers`. For the low-level native ABI, see
`docs/api_contract.md` in the source repository.

## MPI and layout

* Run `pyqcu --selftest` as a single process. Under `mpirun`, the native layers
  are reported as `SKIP` with a one-rank requirement, and only rank 0 prints
  the report; `pyqcu --json` is also rank-0 only.
* Each local extent must be even for parity layouts.
* Parity-split fields compress the **time** axis to `T/2`; do not reshape a
  full field into that layout.
* Production CUDA is one Python process per MPI rank per device. Do not create
  multiple GPU workers inside one process.
* KUPL uses MPI across ranks and OpenMP inside each rank. Pin
  `OMP_NUM_THREADS` and `KUPL_OMP_THREADS`; avoid oversubscription.
* If `mpirun` exits 135 with `Bus error`, check `df -h /tmp`. OpenMPI's
  shared-memory pool can fail when `/tmp` is full; point
  `OMPI_MCA_orte_tmpdir_base` and `TMPDIR` at a writable path, remove stale
  MPI temp files, or set `OMPI_MCA_btl=^smcuda` for CPU-only diagnostics.

## Warnings and known limitations

* `TileLang is unavailable` is an optional-dependency notice. It does not
  affect Torch, KUPL, or CUDA paths.
* `pyqcu.tools.norm()` already returns a Python `float`; do not call `.item()`.
* Torch Legacy `Multigrid.solve_result()` returns `UNKNOWN`/`None` for fields
  the backend does not expose. This is deliberate; do not infer them.
* A native library built on another machine may import but fail at the first
  call. Record the library path and SHA256 in benchmarks.

## Reporting a bug

Include the output of:

```bash
pyqcu --json
pyqcu --selftest --json
python -m pip check
```

and the failing command, lattice, precision, backend, and whether the issue
reproduces from a clean virtual environment.
