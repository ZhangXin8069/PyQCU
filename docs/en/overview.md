# Overview

PyQCU is a lattice-QCD operator and solver toolkit. It provides Wilson and
Clover Dirac operators, stout and Wuppertal smearing, BiCGStab/CG/FGMRES
solvers, Legacy and Strict multigrid hierarchies, and an MPI rank-grid layer.
The same public Python API drives three execution paths:

* the Torch backend (`pyqcu.torch`), running on CPU, CUDA, or Ascend NPU
  through the `pyqcu.cann` compatibility layer;
* the KUPL CPU backend (`pyqcu.kupl`), an independent C++17 MPI+OpenMP
  implementation loaded through `ctypes`;
* the CUDA/DCU backend (`pyqcu.cuda`), a C++/CUDA implementation exposed
  through a Cython bridge.

This release is `0.1.2`. The package is distributed as a source archive; the
KUPL library is built on the installing machine when the toolchain is
available, and the CUDA/DCU library is built opportunistically.

## Data layout

Tensors use a channel-first layout with the four spacetime axes last:

| Object | Shape |
|---|---|
| Gauge field | `[3, 3, 4, X, Y, Z, T]` |
| Fermion field | `[4, 3, X, Y, Z, T]` |
| Clover term | `[4, 3, 4, 3, X, Y, Z, T]` |
| Parity-split field | `[2, 4, 3, X, Y, Z, T/2]` |

Ward indices use negative indexing (`wards["x"] == -4`, `wards["t"] == -1`),
so they remain correct when a tensor carries extra prefix axes.

## Quick check

```python
import torch
from pyqcu import lattice

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=42, sigma=0.1)
assert U.shape == (3, 3, 4, *shape)
assert lattice.check_su3(U, tol=1e-3)
print("gauge field OK", tuple(U.shape))
```

The Torch backend is always importable once the package and its dependencies
are installed. Native KUPL/CUDA availability is reported by:

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

`pyqcu` is the installed console script; `python -m pyqcu` is equivalent.
`--selftest` runs the Torch/KUPL/CUDA layers and reports missing native
backends as `SKIP`.

## Package layers

| Module | Purpose |
|---|---|
| `pyqcu.api` | Backend-neutral configuration, result, capability, and Protocol objects |
| `pyqcu.lattice` | Gamma matrices, Gell-Mann matrices, SU(3) checks, gauge/source generation |
| `pyqcu.torch` | Torch/CANN dslash, smearing, solvers, and numerical helpers |
| `pyqcu.kupl` | `ctypes` binding and multigrid wrappers for the KUPL CPU library |
| `pyqcu.cuda` | Cython bridge and multigrid wrappers for the CUDA/DCU library |
| `pyqcu.solver` | Unified `Krylov` and `Multigrid` selectors across backends |
| `pyqcu.tools` | MPI/layout, HDF5, environment, budget, and plotting helpers |

## See also

* {doc}`usage` for operators, smearing, solvers, and HDF5 round-trips.
* {doc}`solvers` for choosing iterative, multigrid, and native solver paths.
* {doc}`install` for installation, native build variables, and verification.
* {doc}`performance` for historical cross-platform measurements and their
  reproducibility status.
* {doc}`troubleshooting` for installation, native-backend, and solver failures.
