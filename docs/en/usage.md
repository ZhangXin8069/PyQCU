# Usage

The examples below use a small `8^4` lattice so they run quickly on a laptop.
Every block is self-checking: it asserts its own result instead of printing a
number that could be wrong.

## Gauge field and lattice constants

```python
import torch
from pyqcu import lattice

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1)
kappa = torch.tensor([0.125])
assert lattice.check_su3(U, tol=1e-3)
```

`kappa` is the hopping parameter, `kappa = 1 / (2 * mass + 8)`. The Clover
coefficient `csw` is stored separately (in the native ABI it is `argv[7]`).

## Wilson and Clover operators

```python
from pyqcu.torch import dslash

src = torch.randn((4, 3, *shape), dtype=torch.complex64)
out = dslash.give_wilson(src=src, U=U, kappa=kappa)
assert out.shape == src.shape
assert torch.isfinite(out).all()
```

```python
clover = dslash.make_clover(U=U, kappa=kappa)
assert clover.shape == (4, 3, 4, 3, *shape)
assert torch.isfinite(clover).all()
```

## A small BiCGStab solve

```python
from pyqcu.torch import solver

b = torch.randn_like(src)
operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)
x = solver.bistabcg(
    b=b, matvec=operator.matvec, tol=1e-5, max_iter=200, verbose=False)
residual = (
    torch.linalg.vector_norm(operator.matvec(src=x) - b)
    / torch.linalg.vector_norm(b))
assert residual < 1e-3, residual
```

For a pure Wilson operator, pass a zero Clover term or use the Wilson-only
path. `solver.bistabcg` is the Torch implementation; the native CUDA/KUPL
whole-solve entry points are reached through `pyqcu.solver.Krylov`.
See {doc}`solvers` for multigrid, native Krylov, cache, and MPI solver usage.

## Stout smearing

```python
from pyqcu.torch import smear

U_smeared = smear.stout_smear(U=U, nstep=1, rho=0.12)
assert U_smeared.shape == U.shape
assert lattice.check_su3(U_smeared, tol=1e-2)
```

## HDF5 round-trip

```python
import os
import tempfile
from pyqcu.tools import load_tensor_h5, save_tensor_h5

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "fermion.h5")
    save_tensor_h5(src, path)
    restored = load_tensor_h5(path)
assert torch.equal(restored, src)
```

The tensor layout is preserved as-is; the caller owns the layout convention.
Production helpers such as `pyqcu.tools.save_tensor_h5` create a fresh HDF5
handle per call, so independent threads can write different files safely.

## Native KUPL backend

The KUPL binding is lazy: importing `pyqcu.kupl` does not load the library
until `pyqcu.kupl.qcu` is imported. It searches, in order:

1. `PYQCU_KUPL_LIB`, when set;
2. `pyqcu/kupl/libqcu.so` inside the installed package;
3. `pyqcu/kupl/build/libqcu.so`;
4. `cpp/kupl/qcu/libqcu.so` in a source checkout.

```python
# docs-skip: requires a native KUPL library built for this machine
from pyqcu.kupl import qcu
print(qcu.library_path())
```

The unified Krylov facade accepts a `KuplLinearOperator`; see
`pyqcu.solver.Krylov` and `pyqcu.api.BiCGStabControl`.

## Command-line diagnostics

```bash
python -m pyqcu
python -m pyqcu --json
```

The report prints the package version and location, dependency versions, and
whether the KUPL and CUDA native backends are available.

## Common pitfalls

* `pyqcu.tools.norm()` already returns a Python float; calling `.item()` on it
  raises `AttributeError`.
* Parity-split tensors compress the **time** axis in PyQCU. Do not reshape a
  QUDA/QDP fermion directly into PyQCU parity layout.
* CUDA production uses one Python process per MPI rank and per device.
  Multi-GPU work inside one process is a test-only driver.
* A native library built for a different CPU/GPU architecture can import but
  fail at the first call; record the library path and SHA256 in benchmarks.
