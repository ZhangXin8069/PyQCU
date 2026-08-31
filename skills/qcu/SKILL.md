---
name: qcu
description: cpp/cuda/qcu 目录的完整生成 skill：PyQCU 主 C++ CUDA 后端（hand-tuned CUDA 内核 + MPI halo 交换），含 strict QUDA-style MultiGrid、参数协议与显存生命周期不变量。
---
# cpp/cuda/qcu

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
└── pyqcu.h       — C API declarations (extern "C"); must match pyqcu/cuda/qcu/qcu_api.pxd
```

## Parameter Protocol

Parameters are passed from Python as flat arrays. Index constants in `include/define.h` must stay in sync with `pyqcu/cuda/define.py`:

- **`params`** (CPU int32[58]): lattice dims, grid sizes, data types, iteration counts, plan selection, MG level configs and controls. Slots 54–57 are `_MG_USE_GCR_`, `_MG_USE_DEFLATE_`, `_MG_MU_PRE_`, and `_MG_USE_INIT_GUESS_`; slot 57 is the 0/1 warm-start flag and the ABI remains length 58.
- **`argv`** (real[7]): mass, atol, sigma, MG tolerances；c64 为 float32，c128 为 float64
- **`set_ptrs`** (CPU int64[100]): LatticeSet/scratch pointers. Per strict transition `t`, slots `60+4*t+0..3` are blocked `V`, optional raw `Y`, `Yhat`, and onsite pair `(X,X^-1)`; slot `80` is the persistent hierarchy handle.

`_SET_PLAN_` (params[16]) selects the kernel plan:
- `-2` = Laplacian, `-1` = Gauss gauge, `0` = Wilson dslash, `1` = BiStabCG/CG, `2` = Clover dslash

## Strict QUDA-style MultiGrid

The strict path is parallel to the retained legacy MG implementation. Select it with `hierarchy_mode="strict"` or `QudaStrictMultigrid`; `setup_operator="schur"` belongs to the legacy compact odd-Schur path and is not a synonym for Strict. The currently supported all-level mode is `coarse_grid_solution_type=matpc` with `smoother_solve_type=direct_pc`. At level `l`, split the full operator as

\[
D_l=X_l+H_l,\qquad \widehat D_l=X_l^{-1}D_l,\qquad
D_{l+1}=R_l\widehat D_lP_l.
\]

For Wilson/Clover aggregation, every coarse level has `coarse_spin=2` (`E=2*nvec`) and stores full-lattice `X`, forward/backward `Y`, and preconditioned `Yhat=X^{-1}Y`. The coarse operator is `D=X+H`; Strict Galerkin setup applies `R(X^{-1}D)P`, while runtime `Yhat` links represent `X^{-1}H`. `P` uses the blocked full-lattice null vectors to map a full coarse field to the selected fine parity, while `R=P†` maps that compact fine field back to a full coarse field; their coarse side is never checkerboarded. MATPC alone acts on a compact target parity as `I-Hhat_pq Hhat_qp`. Do not halve coarse geometry or replace the coarse operator with hopping-only dslash.

Formal QUDA null-vector interoperability fixes the spin convention to `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`; treat the gamma basis as part of the input identity, not an implicit adapter assumption.

`src/apply_multigrid_strict.cu` provides raw/full coarse application, MATPC, prepare/reconstruct, parity R/P, a recursive V-cycle, and fused fine-grid right-FGMRES. The current fused path fixes `target_parity=1` and `start_level=1`: fine full fields use `[2,4,3,X,Y,Z,T/2]`, compact Schur fields use `[12,X,Y,Z,T/2]`, and every coarse field remains `[E,Xc,Yc,Zc,Tc]` full. Fine inputs are Gauge `[2,3,3,4,X,Y,Z,T/2]` plus Clover even/odd and inverses `[4,3,4,3,X,Y,Z,T/2]`; coarse levels use Galerkin `X/Y/Yhat` instead of a second physical Gauge/Clover pair. A canonical null vector `[nvec,4,3,X,Y,Z,T]` must first be converted to the C-order blocked C++ ABI `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`; passing the canonical 7-D tensor directly is invalid. `applyMultigridStrictInitQcu` creates the persistent recursive hierarchy and aligned V-cycle arena. The first `applyMultigridStrictFgmresQcu` call lazily attaches a C++ outer workspace to the same slot-80 hierarchy; subsequent calls with the same geometry/restart reuse it. Its exact device footprint is `(2*m+5)*B_f + 2*B_c` (`B_f`: one compact fine-parity vector, `B_c`: one full first-coarse vector); Hessenberg/Givens arrays stay as small host vectors and the iteration loop performs no device allocation. `applyMultigridStrictEndQcu` releases both workspaces. The runtime asset set defaults to `Yhat`, onsite `(X,X^-1)` and inter-level blocked null vectors; packed raw `Y` is optional diagnostic data and should not be resident during ordinary solves.

MPI stage 1 provides c64/c128 global complex dot/norm reduction (`global_reduction=True`) and rank-symmetric preflight. It does not make the strict solver distributed: `setup_halo=False`, `full_halo=False`, `compact_halo=False`, and distributed `fused_fgmres=False`. Keep production multi-rank entry points fail-closed until all halo and fused-solve capabilities are independently validated; never infer strict distributed correctness from the legacy halo path or from a passing reduction probe.

For memory work, distinguish requested live bytes from allocator reservation. Account separately for the persistent recursive hierarchy/V-cycle arena, packed assets, blocked transfers, and the exact fused C++ workspace above; Python outer-Krylov and coarse-I/O resident bytes are zero. Galerkin setup batching uses a conservative four-full-field peak model. The library default setup cap is `512 MiB`; the formal `16×32×32×48` profile is independent of that default and selects colored `C=12` under `4 GiB` for c64, or `C=1` under `1 GiB` for c128. The outer fused cap is a separate `max_krylov_bytes`: solver API default `512 MiB`, formal c64 `512 MiB`, formal c128 `1 GiB`. With `strict_galerkin_mode="auto"`, select site-batch only when its modeled workspace fits and its operator-call count is no greater than colored; record requested/effective mode, `C`, projection batch `K`, cap and setup stats, and let formal validation reject any silent batch shrink. Bind packed assets before calling `seal_cuda_runtime(runtime_assets_bound=True)` to detach native setup duplicates, then add `LatticeSet` scratch/device halo, pinned host halo, Gauge/Clover inputs and CUDA/cuBLAS high-water state. Formal records require memory schema version 2 and the device-wide field `device_used_max_observed_bytes`; the independent `cudaMemGetInfo` probe is untimed, while `setup_seconds` excludes sampler stop. Query `nvidia-smi` by the target GPU UUID and report only `max_observed`, not a guaranteed peak. Verify no live growth over repeated warm solves and that close returns owned live storage to baseline; `empty_cache()` is not evidence of leak freedom.

Strict lifecycle is fixed: `hierarchy.setup()` → construct `CudaSchurOp` (its constructor calls `applyInitQcu`) → bind runtime assets → `applyMultigridStrictInitQcu` → repeated strict V-cycle/FGMRES → `applyMultigridStrictEndQcu` → `CudaSchurOp.release()` (its `applyEndQcu`). Keep the per-instance `_SET_INDEX_` unchanged across all Strict calls; the legacy increment rule applies only to the ordinary `applyInitQcu`/`applyEndQcu` operation sequence.

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

---

## Related skill documents

The subdirectories below maintain their own current skill documents; consult
those documents for details instead of treating this overview as a copied
snapshot.

### `include/` — `skills/include/SKILL.md`

# cpp/cuda/qcu/include

C++ header files for the CUDA backend. 26 templated headers containing CUDA kernel implementations (kernels are inline in headers).

## Key Headers

| Header | Purpose |
|--------|---------|
| `define.h` | Parameter index constants, block size — must mirror `pyqcu/cuda/define.py` |
| `lattice_complex.h` | Complex number arithmetic on GPU |
| `lattice_set.h` | Lattice geometry, grid layout, site indexing (use ceiling division for grid dims) |
| `lattice_cuda.h` | CUDA stream management, device utilities |
| `lattice_mpi.h` | MPI halo exchange (blocking `MPI_Sendrecv`) |
| `qcu.h` | Top-level include aggregator |
| `dslash.h` | Dslash dispatch (Wilson vs Clover) |
| `wilson_dslash.h` | Wilson dslash kernel |
| `clover_dslash.h` | Clover dslash dispatch |
| `lattice_wilson_dslash.h` | Wilson dslash implementation |
| `lattice_clover_dslash.h` | Clover dslash implementation |
| `bistabcg.h` / `cg.h` | BiCGStab and CG algorithm kernels |
| `lattice_wilson_bistabcg.h` / `lattice_wilson_cg.h` | Wilson solver wrappers |
| `lattice_clover_bistabcg.h` | Clover BiStabCG solver |
| `lattice_clover_multigrid.h` | Clover multigrid V-cycle (~1100 lines, 5-stream architecture) |
| `lattice_multigrid.h` / `multigrid.h` | Multigrid restrict/prolong/coarse-dslash |
| `laplacian.h` / `lattice_laplacian.h` | Laplacian operator |
| `gauss_gauge.h` | Gaussian gauge field generation |

Headers correspond to `.cu` source files in `../src/` that `#include` them and instantiate the templates.

### `src/` — `skills/src/SKILL.md`

# cpp/cuda/qcu/src

CUDA kernel source files. Each `.cu` file `#include`s the corresponding header from `../include/` and provides template instantiations and kernel launch wrappers.

## Files

| File | Purpose |
|------|---------|
| `apply_init.cu` / `apply_end.cu` | Memory allocation/free lifecycle |
| `apply_dslash.cu` | Dslash dispatch (Wilson or Clover based on plan) |
| `wilson_dslash.cu` | Wilson dslash kernel |
| `clover_dslash_single.cu` / `clover_dslash_multi.cu` / `clover_dslash_comm.cu` | Clover dslash: single-GPU, multi-GPU, halo exchange |
| `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` | Wilson BiStabCG solver + its dslash |
| `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` | Wilson CG solver + its dslash |
| `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` | Clover BiStabCG solver + its dslash |
| `apply_multigrid.cu` | MG restrict/prolong/coarse-dslash |
| `apply_clover_multigrid.cu` | Clover multigrid solver entry (C API bridge) |
| `apply_multigrid_strict.cu` | Strict full-coarse/MATPC primitives, persistent recursive V-cycle and fused right-FGMRES |
| `lattice_mpi.cu` | MPI halo exchange helpers |
| `lattice_cuda.cu` | CUDA utility functions |

### `python/` — `skills/python/SKILL.md`

# cpp/cuda/qcu/python

Python-facing C API declarations. This is the interface boundary between the C++ CUDA backend and the Python Cython bridge.

## Files

| File | Purpose |
|------|---------|
| `pyqcu.h` | C API header — `extern "C"` functions taking raw pointers as `long long`; must include strict MG primitives, lifecycle and fused FGMRES entry points |

This header must stay in exact sync with `pyqcu/cuda/qcu/qcu_api.pxd` (the aliased Cython declarations). Any mismatch causes silent memory corruption.

All functions take three parameter arrays:
- `set_ptrs` (int64[100]): scratch buffer pointers managed by C++ runtime
- `params` (int32[58]): lattice dims, grid sizes, data types, iteration counts, plan selection and MG controls
- `argv` (real[7]): mass, atol, sigma, MG tolerances；c64 为 float32，c128 为 float64

C++→Python data pointers are cast to `long long` from `tensor.contiguous().data_ptr()`.

### `logs/` runtime output

# cpp/cuda/qcu/logs runtime output

Runtime output directory for the C++ CUDA backend (`cpp/cuda/qcu`). Holds generated log files produced by building, testing, and benchmarking the C++ backend.

## Contents

Currently empty. Logs written here may include:

- Build output from `bash ./make.sh` (compiler messages, linker output)
- Test output from running the C++ backend tests (e.g., `examples/qcu/conftest.clover.multigrid.py`)
- Performance / convergence reports

## Notes

- This is a local runtime directory. `cpp/cuda/qcu/logs/` is not tracked in git.
- The canonical location for development reports and test outputs is the repository `logs/` directory. Only backend-local runtime artifacts belong here.
