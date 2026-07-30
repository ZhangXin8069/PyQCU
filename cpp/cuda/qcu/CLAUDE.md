# CLAUDE.md — cpp/cuda/qcu

Primary C++ CUDA backend for PyQCU. Hand-tuned CUDA kernels with MPI halo exchange for Wilson/Clover Dirac operators, BiStabCG/CG solvers, multigrid, and gauge field generation.

## Build

```bash
source ./env.sh       # CUDA toolkit paths, MPI, etc.
bash ./make.sh        # symlinks CMakeLists-nv.txt → CMakeLists.txt, then cmake + make
```

Output: `libqcu.so` — dynamically linked library loaded by the Cython bridge.

## Source Organization

```
include/          — 26 header files (templated C++ with CUDA kernels inline)
├── define.h      — Parameter index constants (must mirror pyqcu/cuda/define.py)
├── lattice_complex.h   — Complex number arithmetic (operator*= was fixed for overwrite bug)
├── lattice_set.h       — Lattice geometry, grid layout, site indexing
├── lattice_cuda.h      — CUDA utility functions (stream management, etc.)
├── lattice_mpi.h       — MPI halo exchange helpers (blocking Sendrecv)
├── qcu.h               — Top-level include aggregator
├── dslash.h            — Dslash dispatch
├── wilson_dslash.h     — Wilson dslash kernel
├── clover_dslash.h     — Clover dslash entry
├── lattice_wilson_dslash.h   — Wilson dslash implementation
├── lattice_clover_dslash.h   — Clover dslash implementation
├── bistabcg.h          — BiCGStab algorithm (GPU kernels)
├── cg.h                — Conjugate gradient algorithm
├── lattice_wilson_bistabcg.h — Wilson BiStabCG solver
├── lattice_wilson_cg.h        — Wilson CG solver
├── lattice_clover_bistabcg.h  — Clover BiStabCG solver
├── multigrid.h                — MG restrict/prolong/coarse-dslash
├── lattice_multigrid.h        — MG implementation
├── lattice_clover_multigrid.h — Clover multigrid solver (~1100 lines)
├── laplacian.h / lattice_laplacian.h — Laplacian operator
└── gauss_gauge.h              — Gaussian gauge field generation

src/              — .cu files that #include the headers and instantiate kernels
python/
└── pyqcu.h       — C API declarations (extern "C"); must match pyqcu/cuda/qcu/qcu.pxd
```

## Parameter Protocol

Parameters are passed from Python as flat arrays. Index constants in `include/define.h` must stay in sync with `pyqcu/cuda/define.py`:

- **`params`** (int32[54]): lattice dims, grid sizes, data types, iteration counts, plan selection, MG level configs
- **`argv`** (float[7]): mass, atol, sigma, MG tolerances
- **`set_ptrs`** (int64[100]): scratch buffer pointers

`_SET_PLAN_` (params[16]) selects the kernel plan:
- `-2` = Laplacian, `-1` = Gauss gauge, `0` = Wilson dslash, `1` = BiStabCG/CG, `2` = Clover dslash

## Clover Multigrid Stream Architecture (5 streams)

```
main (strm):   dslash operations (fine_dslash_op / coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), convergence-check dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## Critical Invariants (from bug fixes)

1. **Scalars live only in `device_vals`** — no host→device scalar memcpy inside iteration loops
2. **Full stream sync at bottom of each iteration** — sync ALL 5 streams before next iteration
3. **`_send_tmp_` scratch for dot products** — cublasDot → scratch slot 7 → MPI_Allreduce → copy to target (never write cublasDot directly to target)
4. **`mpi_real_type<T>()` template** — dispatches `MPI_FLOAT`/`MPI_DOUBLE` per template type
5. **`run_mpi` uses blocking `MPI_Sendrecv`** — no `MPI_Wait` needed (only `run_mpi_non_block` requires it)

## Block Size

`_BLOCK_SIZE_` in `define.h`: use 8/16 for testing small lattices, 128 for NVIDIA production, 256 for AMD DCU production.
