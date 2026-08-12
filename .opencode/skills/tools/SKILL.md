---
name: tools
description: pyqcu.tools 目录的完整生成 skill：MPI 网格/奇偶分割/维度重排/HDF5 I/O/线性代数/多重网格转移/TileLang JIT 工具集。
---
# CLAUDE.md — pyqcu.tools

Utility modules for MPI grid management, HDF5 I/O, linear algebra, tensor operations, multigrid transfers, and TileLang JIT kernels.

## Files

| File | Purpose |
|------|---------|
| `_define.py` | MPI grid size factorization, rank neighbors, parity splitting (`oooxyzt2poooxyzt`/`poooxyzt2oooxyzt`), dimension reordering (ccdxyzt↔ccdptzyx, scxyzt↔psctzyx), dtype conversion tables, device setup, slice helpers, prime factorization |
| `_io.py` | HDF5 I/O with MPI parallel I/O (`driver='mpio'`, `h5py`) and serial gather/scatter fallback (`comm.gather` + `comm.scatter`) |
| `_linalg.py` | Vector dot product (`vdot`) and norm (`norm`) via `_torch` |
| `_einsum.py` | TileLang JIT-compiled einsum kernels — currently `Eexyzt_exyzt2Exyzt` (optional, try/except import) |
| `_matul.py` | TileLang-based matrix multiply kernels: `matmul_gpu(M,N,K,...)` and `matmul_cpu(M,N,K,...)` (optional) |
| `_multigrid.py` | Null vector generation (`give_null_vecs`), local orthogonalization (`local_orthogonalize`), restrict/prolong operators — all with NPU-compatible fallback paths |
| `_roll.py` | Tensor rolling utilities |

## Exported API

### MPI Grid (`_define.py`)

| Function | Purpose |
|----------|---------|
| `give_grid_size()` | Auto-factor MPI communicator size into 4D grid `[gx, gy, gz, gt]` via prime factorization (sorted ascending) |
| `give_grid_index(rank)` | Convert flat rank to 4D grid index `[ix, iy, iz, it]` |
| `give_rank_plus(ward, rank)` | Neighbor rank in +direction |
| `give_rank_minus(ward, rank)` | Neighbor rank in −direction |
| `give_rank_plus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, +b) |
| `give_rank_plus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (+a, −b) |
| `give_rank_minus_minus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, −b) |
| `give_rank_minus_plus(ward_a, ward_b, rank)` | Diagonal neighbor (−a, +b) |
| `set_device(device, verbose)` | Set CUDA/NPU device based on MPI rank (round-robin assignment) |

### Parity Splitting (`_define.py`)

- **`oooxyzt2poooxyzt(input_array, verbose) → [2, ..., t, z, y, x//2]`** — Standard layout → parity-split. Separates even/odd sites based on (x+y+z+t) % 2. Splits along the fastest-varying (x) dimension.
- **`poooxyzt2oooxyzt(input_array, verbose) → [..., t, z, y, x]`** — Reverse: parity-split → standard layout. Recombines even/odd halves.

Both support NPU via explicit real/imaginary handling.

### Even-Odd Mask (`_define.py`)

- **`give_eo_mask(oootzy_t_p, eo, verbose)`** — Returns boolean mask for even (`eo=0`) or odd (`eo=1`) sites. Uses `(x+y+z) % 2` checkerboard. Results cached by shape+device+eo key.

### Dimension Reordering (`_define.py`)

HDF5 I/O uses dimension order `zyxt` (fastest to slowest: t, z, y, x):

- **`ccdxyzt2ccdptzyx(ccdxyzt) → [c,c,d,p,t,z,y,x]`** — Gauge field to file layout
- **`ccdptzyx2ccdxyzt(ccdptzyx) → [c,c,d,x,y,z,t]`** — File layout to gauge field
- **`scxyzt2psctzyx(scxyzt) → [p,s,c,t,z,y,x]`** — Fermion field to file layout
- **`psctzyx2scxyzt(psctzyx) → [s,c,x,y,z,t]`** — File layout to fermion field

### MPI Gather/Scatter (`_define.py`)

- **`local_xyzt2whole_xyzt(local_array, root) → Tensor | None`** — Gather distributed tensor chunks into a full global tensor on root rank. Uses `comm.Gather`.
- **`whole_xyzt2local_xyzt(dtype, device, whole_shape, whole_array, root) → Tensor`** — Scatter a global tensor (or shape template) to all ranks. Uses `comm.Scatter`. Each rank gets its grid block.

### Slice Helpers (`_define.py`)

- **`slice_dim(dims_num, ward, start, stop, step, point)`** — Build Python slice tuple for indexing along a specific ward dimension (using negative indexing). For `point`, returns integer index.
- **`slice_dim_dim(dims_num, ward_a, ..., ward_b, ...)`** — Two-dimension slice
- **`slice_dim_none_dim(dims_num, ward, ..., ward_none)`** — Slice with one skipped dimension

### Memory Helpers (`_define.py`)

- **`to_contiguous_real(tensor, channel, *shape)`** — Extract real/imag channel from complex tensor and return a truly stride-1 contiguous real tensor. Uses `empty + copy_` pattern instead of `.contiguous()` for correctness on single-element tensors.

### HDF5 I/O (`_io.py`)

- **`gridoooxyzt2hdf5oooxyzt(input_tensor, file_name, lat_size, verbose)`** — Write distributed tensor to HDF5. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses `comm.gather` to root.
- **`hdf5oooxyzt2gridoooxyzt(file_name, lat_size, device, verbose)`** — Read HDF5 into distributed tensor. MPI path uses `h5py.File(..., driver='mpio')`; serial path uses root-read + `comm.scatter`.

**MPI support detection:** `HAS_MPI_SUPPORT = check_mpi_support()` at module import time. Tests h5py config and tries creating a test file with `driver='mpio'`. Can be manually overridden.

**Serial fallback note:** `comm.scatter` uses pickle serialization; may hit 2GB limit for very large lattices (>64⁴ float32). MPI I/O path preferred for production.

### Linear Algebra (`_linalg.py`)

- **`norm(input, p='fro', dim=None, keepdim=False)`** — Frobenius/vector norm via `_torch.norm`
- **`vdot(input, other)`** — Complex inner product `Σ conj(a_i) * b_i` via `_torch.vdot`

### Multigrid Utilities (`_multigrid.py`)

- **`give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)`** — Generate near-null-space vectors via inverse iteration: v_i = v_i − A^{-1} A v_i. Optionally orthogonalizes against previous vectors. `null_vecs` parameter is used as shape/dtype/device template only; values are overwritten with random init.
- **`local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)`** — Block-local Gram-Schmidt orthogonalization via batched QR decomposition. Splits null vectors into coarse-grid blocks, applies QR per block. NPU path avoids >8-dim tensors.
- **`restrict(local_ortho_null_vecs, fine_vec)`** — P^T v_fine = Σ v_fine · null_vec^†. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.
- **`prolong(local_ortho_null_vecs, coarse_vec)`** — P v_coarse = Σ null_vec · v_coarse. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.

**NPU compatibility:** NPU limits tensors to ≤8 dimensions, so restrict/prolong/orthogonalize all have `_npu` variants that use reshape/permute chains to stay within this limit. Cross-validated against standard path (max diff ~1e-7 for float32).

### TileLang Integration (`_einsum.py`, `_matul.py`)

Optional — try/except import at package level; silently degrades if TileLang unavailable.

- **`Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)`** — JIT-compiled TileLang kernel for specific einsum pattern used in Wilson dslash (disabled by default; `tools_Eexyzt_exyzt2Exyzt = False`)
- **`matmul_gpu(M, N, K, block_M, block_N, block_K)`** / **`matmul_cpu(M, N, K, ...)`** — TileLang kernel definitions for matrix multiply benchmarking

Kernels use `warp_size = 128` from `_define`.

### Dtype Conversion Tables (`_define.py`)

- **`np2torch_dtype`**, **`torch2np_dtype`** — bidirectional NumPy ↔ PyTorch dtype maps
- **`torch2tl_dtype`** — PyTorch → TileLang dtype map (float16/32/64 only)

## Logging Convention

`PYQCU::TOOLS::<SUBMODULE>::\n message`
