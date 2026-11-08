# PyQCU

PyQCU is a lattice-QCD operator and solver toolkit with three execution
paths:

* a Torch/CANN backend for CPU, CUDA, and Ascend NPU;
* an independent C++17 MPI+OpenMP CPU backend (KUPL);
* a CUDA/DCU C++ backend exposed through a Cython bridge.

It provides Wilson and Clover Dirac operators, stout and Wuppertal smearing,
BiCGStab/CG/FGMRES solvers, Legacy and Strict multigrid hierarchies, HDF5
helpers, and a four-dimensional MPI rank grid.

## Install

```bash
pip install pyqcu
```

Requirements: Python 3.10+, Linux, `numpy`, `torch`, `mpi4py`, and `h5py`
(installed automatically). A C++17 compiler, CMake, and MPI are needed to
build the KUPL CPU library; `nvcc` is needed for the optional CUDA library.

The package is distributed as a source archive. The first install builds KUPL
when the toolchain is available and tries CUDA/DCU opportunistically; a
missing toolchain leaves the pure-Torch paths usable and does not fail the
install.

```bash
# CPU-only
PYQCU_BUILD_BACKEND=kupl pip install pyqcu

# Require the KUPL build
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu

# Build KUPL and CUDA on a machine with nvcc
PYQCU_BUILD_BACKEND=cuda pip install pyqcu
```

Verify the installation:

```bash
python -m pyqcu
python -m pyqcu --json
```

## Quick start

```python
import torch
from pyqcu import lattice
from pyqcu.torch import dslash

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1)
assert lattice.check_su3(U, tol=1e-3)

src = torch.randn((4, 3, *shape), dtype=torch.complex64)
kappa = torch.tensor([0.125])
out = dslash.give_wilson(src=src, U=U, kappa=kappa)
assert out.shape == src.shape
assert torch.isfinite(out).all()
```

The public layers are `pyqcu.api` (backend-neutral contracts), `pyqcu.lattice`
(gamma/SU(3) and gauge/source generation), `pyqcu.torch` (Torch operators and
solvers), `pyqcu.kupl` (CPU ctypes binding), `pyqcu.cuda` (CUDA/DCU Cython
bridge), `pyqcu.solver` (unified Krylov/Multigrid selectors), and `pyqcu.tools`
(MPI/layout, HDF5, environment, and plotting helpers).

## Native backends

`python -m pyqcu` reports the package location and whether the KUPL and CUDA
libraries are available. The KUPL binding searches `PYQCU_KUPL_LIB`, then the
installed `pyqcu/kupl/libqcu.so`, then the source checkout. The CUDA bridge is
built only when `nvcc` and the CUDA library are available; the extension uses
an `$ORIGIN` rpath so it finds the library inside the installed package.

Typical release verification uses a `8x8x16x16` complex64 lattice with a
Gaussian SU(3) gauge field (`sigma=0.1`, `seed=42`), `mass=0.17`, `csw=1.0`,
and `atol=1e-6`. On the release machine (WSL2, Intel i7-13620H, RTX 4060
Laptop sm_89), both the installed KUPL and CUDA paths converged in 26
iterations with full-lattice true residuals below `3e-7`; the acceptance
threshold for complex64 is `max(atol, 2048*eps32) ≈ 2.4414e-4`.

## Historical performance

The table records measurements on hardware that is not part of this checkout.
Absolute seconds are not comparable across machines, and CPU/GPU rows are
never combined into one speed-up. Raw JSON files for these runs are not
retained in the current tree; the linked report and git tag are the
provenance.

| Case | Platform | Lattice / precision | Result | Comparison | Reproducibility |
|---|---|---|---|---|---|
| CUDA BiCGStab vs QUDA | Tesla V100-SXM2-32GB, sm_70 | 24³×64, c64 | 0.147801 s / 18 iterations | QUDA 0.156680 s / 19 iterations → 1.060× | Historical; needs V100 and the `dev103`-era checkout |
| KUPL parity-Schur BiCGStab | Intel Xeon E5-2696 v3, 18 cores | 24³×64, c64 | 8 ranks×2 threads: 7.532 s / 19 iterations | 1 rank×4 threads 19.229 s → 2.553× | Historical; same CPU class required |
| CUDA Legacy-MG vs QUDA GCR-MG | A100-PCIE-40GB | 32³×64, c64 | 1/2/4 ranks: 0.2145 / 0.2195 / 0.2183 s | QUDA 0.4824 / 0.3144 / 0.2875 s → 2.2495× / 1.4323× / 1.3166× | Historical; needs A100, L32 LIME, E12 cache |
| CUDA native BiCGStab | A100-PCIE-40GB | 32³×64, c64 | 1 rank 0.1224 s; 2 ranks 0.1154 s | QUDA 0.1304 / 0.0961 s → 1.065× / **0.833×** | Historical; shows the native solver is not universally faster |
| CUDA Strict-MG vs QUDA aligned MG | Tesla V100-SXM2-32GB | 8×8×8×16, c64 | 0.021693 s / 10 iterations | QUDA 0.144755 s / 8 outer GCR → 6.673× | Time-point evidence; small-lattice fixed overhead dominates |
| CUDA(HIP) Legacy-MG | DCU Pre-Wukong D160A1, gfx906 | 24³×64, c64 | 1/2/4 ranks: 0.3017 / 0.4619 / 1.0247 s | QUDA blocked; no ratio | Historical; needs the same DCU and DTK |

The full table, including the historical tag-only rows and the local
`8x8x16x16` measurement, is in the performance documentation.

## Documentation

* English: https://pyqcu.readthedocs.io/en/latest/
* 中文: https://pyqcu.readthedocs.io/zh-cn/latest/

The documentation sources live in `docs/en/` and `docs/zh/`. Both sites are
built with Sphinx and MyST; the release check builds them with `-W`.

## From a source checkout

```bash
git clone git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git
cd PyQCU
source ./env.sh

bash ./install.sh kupl    # CPU KUPL backend
bash ./install.sh cuda    # CUDA backend + Cython bridge
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu
```

Tests and benchmark drivers live under `pyqcu/testing/` and are intentionally
excluded from the PyPI package:

```bash
cd pyqcu/testing
pytest .
```

## License

MIT. See `LICENSE`.
