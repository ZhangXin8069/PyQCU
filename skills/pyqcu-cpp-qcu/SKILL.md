---
name: pyqcu-cpp-qcu
description: 当任务涉及 cpp/cuda/qcu 的 CUDA 内核、MPI halo、MultiGrid 或 C++ 参数与显存生命周期时使用。
---
# cpp/cuda/qcu

Primary C++ CUDA backend for PyQCU. Hand-tuned CUDA kernels with MPI halo exchange for Wilson/Clover Dirac operators, BiStabCG/CG solvers, multigrid, and gauge field generation.

Production execution topology is fixed:

```text
one Python process = one MPI rank = one GPU/DCU/GPGPU
```

Cross-device solves use MPI domain decomposition and C++ halo/global-reduction
paths. The Python-level `pyqcu.cuda._multi_gpu.MultiGpuMultigrid` driver is a
test-only, explicitly constructed exception and must not be used as the default
template for new CUDA programs.

Normative project rules:
[`docs/cuda_execution_topology.md`](../../docs/cuda_execution_topology.md).

## Build

```bash
source ./env.sh       # CUDA toolkit paths, MPI, etc.
bash ./make-nv.sh     # symlinks CMakeLists-nv.txt → CMakeLists.txt, then cmake + make
```

Output: `libqcu.so` — dynamically linked library loaded by the Cython bridge.

CMake 默认自动使用 `env.sh` 探测到的 GPU compute capability（cubin + PTX），
不需要为单一 GPU 型号锁死架构；V100 生产基准仍可显式设置：

```bash
QCU_CUDA_ARCHITECTURES='70-real;70-virtual' bash ./build-nv.sh
cuobjdump --list-elf cpp/cuda/qcu/libqcu.so | head
```

`cuobjdump` 必须显示 `sm_70.cubin`，而不是仅显示 `sm_60.cubin`。不要把
sm_60 JIT 与 sm_70 native 的 solve 数字混在同一速度比中；至少记录
`QCU_CUDA_ARCHITECTURES`、libqcu SHA256 和实际 `cuobjdump` 架构。当前
V100 side-only 复测中 sm70 大 c64 为 `0.657074 s`、大 c128 为
`2.062569 s`；随后 full formal 独立复测大 c64 为 `0.655570 s`。
sm70 相对默认 sm60 JIT 改善有限，但必须保留架构 provenance。
benchmark worker 已自动写入 `provenance.pyqcu_library`；不要依赖口头
记录或只看 `LD_LIBRARY_PATH`。

## Source Organization

```
include/          — 24 header files (templated C++ with CUDA kernels inline)
├── define.h      — Parameter index constants (must mirror pyqcu/cuda/define.py)
├── include.h     — 系统/第三方头聚合（cuda/cublas/curand/mpi/chrono/...）
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
├── lattice_sap.h              — SAP（Schwarz 交替过程）mask/update 内核，5-stream MG 的平滑器
├── multigrid.h                — MG restrict/prolong/coarse-dslash
├── lattice_multigrid.h        — MG implementation
├── lattice_clover_multigrid.h — Clover multigrid solver（5745 行，5-stream 架构）
├── laplacian.h / lattice_laplacian.h — Laplacian operator
└── gauss_gauge.h              — Gaussian gauge field generation

src/              — 32 个 .cu 文件：#include 对应头并做模板实例化与启动封装
python/
└── pyqcu.h       — C API declarations (extern "C"，37 个入口); must match pyqcu/cuda/qcu/qcu_api.pxd
```

## Parameter Protocol

Parameters are passed from Python as flat arrays. Index constants in `include/define.h` must stay in sync with `pyqcu/cuda/define.py`:

- **`params`** (CPU int32[58]): lattice dims, grid sizes, data types, iteration counts, plan selection, MG level configs and controls. Slots 54–57 are `_MG_USE_GCR_`, `_MG_USE_DEFLATE_`, `_MG_MU_PRE_`, and `_MG_USE_INIT_GUESS_`; slot 57 is the 0/1 warm-start flag and the ABI remains length 58.
- **`argv`** (real[8]): mass, atol, sigma, MG tolerances, csw；c64 为 float32，
  c128 为 float64。csw 位于槽 7，Clover 系数为 `-kappa*csw/8`。
- **`set_ptrs`** (CPU int64[100]): LatticeSet/scratch pointers. Per strict transition `t`, slots `60+4*t+0..3` are blocked `V`, optional raw `Y`, `Yhat`, and onsite pair `(X,X^-1)`; slot `80` is the persistent hierarchy handle.

`_SET_PLAN_` (params[16]) selects the kernel plan:
- `-2` = Laplacian, `-1` = Gauss gauge, `0` = Wilson dslash, `1` = BiStabCG/CG, `2` = Clover dslash

## Strict QUDA-style MultiGrid

The strict path is parallel to the retained legacy MG implementation. Select it with `hierarchy_mode="strict"` or `StrictMultigrid`; `setup_operator="schur"` belongs to the legacy compact odd-Schur path and is not a synonym for Strict. The currently supported all-level mode is `coarse_grid_solution_type=matpc` with `smoother_solve_type=direct_pc`. At level `l`, split the full operator as

\[
D_l=X_l+H_l,\qquad \widehat D_l=X_l^{-1}D_l,\qquad
D_{l+1}=R_l\widehat D_lP_l.
\]

The full/raw coarse onsite block and directional links are dense
\(E\times E\) matrices, not \(SU(3)\) links:
\[
X_C(X)=\sum_{x\in\mathcal A_X}V_X^\dagger C V_X,
\]
\[
Y^{\rm f}_{\mu}(X)=-\kappa\sum_{x\in\mathcal A_X}
(AV_X)^\dagger U_\mu V_{X+\hat\mu},
\]
\[
Y^{\rm b}_{\mu}(X-\hat\mu)=-\kappa\sum_{x\in\mathcal A_X}
V_X^\dagger U^\dagger_\mu AV_{X-\hat\mu},
\]
where \(AV=C^{-1}V\) for Clover-PC and \(AV=V\) for the raw path.
For a Wilson-like operator the onsite contribution also includes the
identity \(I_E\); a coarse-to-coarse construction projects the current
\(X_\ell\), not the fine Clover tensor. In the current Strict left-
preconditioned path, \(AV\) is constructed first and the coarse local
operator is based on \(X_f^{-1}D_f\); therefore \(V^\dagger CV\) describes
the full/raw Clover construction and must not be stated as an unconditional
formula for the Strict PC path.
The raw coarse action is
\[
(D_cz)(X)=X(X)z(X)+\sum_\mu\left[
Y_\mu^{\rm f}(X)z(X+\hat\mu)
+Y_\mu^{\rm b}(X-\hat\mu)^\dagger z(X-\hat\mu)\right],
\]
and the preconditioned links are
\[
\widehat Y_\mu^{\rm f}(X)=X^{-1}(X)Y_\mu^{\rm f}(X),\qquad
\widehat Y_\mu^{\rm b}(X-\hat\mu)
=Y_\mu^{\rm b}(X-\hat\mu)X^{-\dagger}(X).
\]
The backward site and left/right multiplication order are part of the ABI:
PC dslash applies only \(\widehat Y\) hopping, while MATPC is
\(M_p^c=I-\widehat H_{pq}\widehat H_{qp}\). See
`docs/report_multigrid_quda_pyqcu_20260913.tex:116-168,171-204`.
The `-kappa` coefficient may either be absorbed into stored \(Y\) or
applied later by the dslash; record the convention before comparing
buffers. Current PyQCU CUDA uses links with the coefficient already applied.

For Wilson/Clover aggregation, every coarse level has `coarse_spin=2` (`E=2*nvec`) and stores full-lattice `X`, forward/backward `Y`, and preconditioned `Yhat=X^{-1}Y`. The coarse operator is `D=X+H`; Strict Galerkin setup applies `R(X^{-1}D)P`, while runtime `Yhat` links represent `X^{-1}H`. `P` uses the blocked full-lattice null vectors to map a full coarse field to the selected fine parity, while `R=P†` maps that compact fine field back to a full coarse field; their coarse side is never checkerboarded. MATPC alone acts on a compact target parity as `I-Hhat_pq Hhat_qp`. Do not halve coarse geometry or replace the coarse operator with hopping-only dslash.

Formal QUDA null-vector interoperability fixes the spin convention to `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`; treat the gamma basis as part of the input identity, not an implicit adapter assumption.

`src/apply_multigrid_strict.cu` provides raw/full coarse application, MATPC, prepare/reconstruct, parity R/P, a recursive V-cycle, and fused fine-grid right-FGMRES. The current fused path fixes `target_parity=1` and `start_level=1`: fine full fields use `[2,4,3,X,Y,Z,T/2]`, compact Schur fields use `[12,X,Y,Z,T/2]`, and every coarse field remains `[E,Xc,Yc,Zc,Tc]` full. Fine inputs are Gauge `[2,3,3,4,X,Y,Z,T/2]` plus Clover even/odd and inverses `[4,3,4,3,X,Y,Z,T/2]`; coarse levels use Galerkin `X/Y/Yhat` instead of a second physical Gauge/Clover pair. A canonical null vector `[nvec,4,3,X,Y,Z,T]` must first be converted to the C-order blocked C++ ABI `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`; passing the canonical 7-D tensor directly is invalid. `applyMultigridStrictInitQcu` creates the persistent recursive hierarchy and aligned V-cycle arena. The first `applyMultigridStrictFgmresQcu` call lazily attaches a C++ outer workspace to the same slot-80 hierarchy; subsequent calls with the same geometry/restart reuse it. Its exact device footprint is `(2*m+5)*B_f + 2*B_c` (`B_f`: one compact fine-parity vector, `B_c`: one full first-coarse vector); Hessenberg/Givens arrays stay as small host vectors and the iteration loop performs no device allocation. `applyMultigridStrictEndQcu` releases both workspaces. The runtime asset set defaults to `Yhat`, onsite `(X,X^-1)` and inter-level blocked null vectors; packed raw `Y` is optional diagnostic data and should not be resident during ordinary solves.

Current strict MPI provides rank-symmetric preflight plus c64/c128 global reduction, setup/full/compact/R-P halos and distributed fused FGMRES. Its capability record is all-true for `setup_halo`, `fine_halo`, `full_halo`, `compact_halo`, `rp_coarse_halo`, `global_reduction`, and `fused_fgmres`. Multi-rank production still requires one process/rank/device, a successful capability preflight and an independently recomputed true residual; do not infer correctness from a passing reduction probe alone.

For memory work, distinguish requested live bytes from allocator reservation. Account separately for the persistent recursive hierarchy/V-cycle arena, packed assets, blocked transfers, and the exact fused C++ workspace above; Python outer-Krylov and coarse-I/O resident bytes are zero. Galerkin setup batching uses a conservative four-full-field peak model. The library default setup cap is `512 MiB`; the formal `16×32×32×48` profile is independent of that default and selects colored `C=12` under `4 GiB` for c64, or `C=1` under `1 GiB` for c128. The outer fused cap is a separate `max_krylov_bytes`: solver API default `512 MiB`, formal c64 `512 MiB`, formal c128 `1 GiB`. With `strict_galerkin_mode="auto"`, select site-batch only when its modeled workspace fits and its operator-call count is no greater than colored; record requested/effective mode, `C`, projection batch `K`, cap and setup stats, and let formal validation reject any silent batch shrink. Bind packed assets before calling `seal_cuda_runtime(runtime_assets_bound=True)` to detach native setup duplicates, then add `LatticeSet` scratch/device halo, pinned host halo, Gauge/Clover inputs and CUDA/cuBLAS high-water state. Formal records require memory schema version 2 and the device-wide field `device_used_max_observed_bytes`; the independent `cudaMemGetInfo` probe is untimed, while `setup_seconds` excludes sampler stop. Query `nvidia-smi` by the target GPU UUID and report only `max_observed`, not a guaranteed peak. Verify no live growth over repeated warm solves and that close returns owned live storage to baseline; `empty_cache()` is not evidence of leak freedom.

Strict lifecycle is fixed: `hierarchy.setup()` → construct `CudaSchurOp` (its constructor calls `applyInitQcu`) → bind runtime assets → `applyMultigridStrictInitQcu` → repeated strict V-cycle/FGMRES → `applyMultigridStrictEndQcu` → `CudaSchurOp.release()` (its `applyEndQcu`). Keep the per-instance `_SET_INDEX_` unchanged across all Strict calls. Ordinary transactions use `applyInitQcu → operation → applyEndQcu → _SET_INDEX_ += 1`.

## Clover Multigrid Stream Architecture (5 streams)

```
main (strm):   dslash operations (fine_dslash_op / coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), convergence-check dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## Critical Invariants (from bug fixes)

0. **Error expressions execute once** — `checkCudaErrors(expr)` and
   `checkMpiErrors(expr)` must store `(expr)` in a local temporary before
   testing/printing it.  The former two-evaluation form retried
   `cudaMallocAsync` inside the error path and could replace the real
   allocation failure with a misleading `CUDA error 0000 "no error"`.
1. **Scalars live only in `device_vals`** — no host→device scalar memcpy inside iteration loops
2. **Full stream sync at bottom of each iteration** — sync ALL 5 streams before next iteration
3. **`_send_tmp_` scratch for dot products** — cublasDot → scratch slot 7 → MPI_Allreduce → copy to target (never write cublasDot directly to target)
4. **`mpi_real_type<T>()` template** — dispatches `MPI_FLOAT`/`MPI_DOUBLE` per template type
5. **`run_mpi` uses blocking `MPI_Sendrecv`** — no `MPI_Wait` needed (only `run_mpi_non_block` requires it)

## Block Size

`_BLOCK_SIZE_` in `define.h`: use 8/16 for testing small lattices, 128 for NVIDIA production, 256 for AMD DCU production.

## Related skill documents

子目录各自维护现行技能文档；细节以那些文档为准，本文**不复制**其正文（此前内嵌的
include/src/python/logs 全文副本已于 2026-10-15 删除，避免与源文档多处漂移）：

| 子目录 | 技能 |
|---|---|
| `include/` | `skills/pyqcu-cpp-qcu-include/SKILL.md` |
| `src/` | `skills/pyqcu-cpp-qcu-src/SKILL.md` |
| `python/` | `skills/pyqcu-cpp-qcu-python/SKILL.md` |
| ~~`logs/`~~（2026-10-15 已删除） | `skills/pyqcu-cpp-qcu-logs/SKILL.md`（归档说明） |

上级容器（`cpp/cuda`、`cpp/{cann,dtk,maca}`）见 `skills/pyqcu-cpp-containers/SKILL.md`。
