---
name: pyqcu-tools
description: 当任务涉及 pyqcu.tools 的 MPI/布局/HDF5/env/budget 或 pyqcu.torch.tools 数值工具时使用。
---
# pyqcu.tools + pyqcu.torch.tools

Common utilities remain in `pyqcu.tools`; Torch tensor operations, multigrid transfers and TileLang JIT kernels live in `pyqcu.torch.tools`.

Canonical tensor-axis names, storage order, and environment entry points are extracted
in [`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#1-环境构建与布局).

## 多线程构建工具（2026-08-14）

- `give_null_vecs_mt(matvec_ops, ...)` — 多线程 null 向量（每线程独立 CUDA generator、
  nan/breakdown 重试、worker 内 set_device、nthreads<=1 直接执行防嵌套线程池）
- `build_stencil_mt(matvec_ops, lonv, ...)` — 多线程 33-tensor stencil（probe 写集不相交、
  src_c 预分配复用）；与单线程构建逐元素一致（0 误差）
- `apply_stencil(hop_nn, hop_diag, sit, v)` — Python 参考实现（A_c = P^T S P）
- `BatchedLocalSchur(...)`（dev80）— 批量 Schur 分解 stencil 构建：W=10 使大格子
  stencil 构建 24min→2min

### Stencil 验证与反模式（2026-08-24）

- **重构可静默丢一行赋值致 stencil 全零**：e46a4cf 重构遗失 `f_local[...]=_blk`，
  无任何报错（bug35）；定位靠资产 mtime × 提交时间交叉验证 + worktree 时间线二分。
- **等价性铁证口径**：C9 双实现对比（build_stencil_local vs 全格参考）rel≤1e-06 即等价；
  Galerkin 投影闭环 5.6e-07（见 pyqcu/testing/regression/session-2026-08-24）。

## Files

| File | Purpose |
|------|---------|
| `_define.py` | MPI grid size factorization, rank neighbors, parity splitting (`oooxyzt2poooxyzt`/`poooxyzt2oooxyzt`), dimension reordering (ccdxyzt↔ccdptzyx, scxyzt↔psctzyx), dtype conversion tables, device setup, slice helpers, prime factorization |
| `_io.py` | HDF5 I/O with MPI parallel I/O (`driver='mpio'`, `h5py`) and serial gather/scatter fallback (`comm.gather` + `comm.scatter`)；另含 tensor/dict 级本地持久化（`save_tensor_h5`/`load_tensor_h5` 就地写、`save_dict_h5`/`load_dict_h5` 走 `tmp + os.replace`），两者都不含 `fsync`，见下节 |
| `_env.py` | 环境与资源快照：`git_snapshot`/`gpu_snapshot`/`gpu_used_mb`/`rss_kb`/`nullvec_cache_dir`/`cache_disk_mb`/`dump_env_h5` |
| `_budget.py` | 显存/RSS/磁盘预算模型：`vram_model`(cold)/`vram_model_warm`/`rss_model`/`disk_cache_bytes`/`fit_from_bench`/`budget_table`；常量 CONST_PER_V=24192 B/V、α=2.6 KB/V、β=512 MB |
| `../torch/tools/_hierarchical.py` | 分层显存管理（Torch）：`HierarchicalTensor`、`HierarchicalCache` 与 RAM/DISK offload |
| `../torch/tools/_linalg.py` | Torch `vdot` / `norm` |
| `../torch/tools/_einsum.py` | TileLang JIT einsum |
| `../torch/tools/_matul.py` | TileLang matrix multiply |
| `../torch/tools/_multigrid.py` | Torch null vectors、restrict/prolong 与 33-tensor stencil |
| `../torch/tools/_roll.py` | Torch tensor rolling |
| `../torch/tools/_mpi_roll.py` | Torch distributed periodic roll |
| `../torch/tools/_distributed_setup.py` | Torch distributed fine operator wrapper |
| `../torch/tools/_strict_galerkin.py` | Torch strict full-coarse Galerkin construction |
| `_bench_plots.py` | 基准图表共用层：调色板常量 `BENCH_PALETTE`/`BENCH_SEQ_BLUE`/`BENCH_INK*`/`BENCH_GRID`/`BENCH_SURF`，`mpl_cjk_setup`/`style_axes`/`save_figure`/`load_json`/`short_label`/`label_of`，以及 `plot_convergence`/`plot_hotspot`/`plot_speedup`/`plot_time`/`plot_sweep_curves`/`plot_bench_report`。**顶层不 import matplotlib**（惰性在函数内），故 `pyqcu.tools` 在无 matplotlib 环境仍可导入 |

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

For GPGPU production, `set_device` selects the one device owned by the current
MPI rank. Do not use round-robin or multiple workers to put one production
solve on multiple devices; that route uses MPI. `MultiGpuMultigrid` is a
test-only exception. See
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology).

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

### Linear Algebra (`pyqcu.torch.tools/_linalg.py`)

- **`norm(input, p='fro', dim=None, keepdim=False)`** — Frobenius/vector norm via `_torch.norm`
- **`vdot(input, other)`** — Complex inner product `Σ conj(a_i) * b_i` via `_torch.vdot`

### Multigrid Utilities (`pyqcu.torch.tools/_multigrid.py`)

- **`give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)`** — Generate near-null-space vectors via inverse iteration: v_i = v_i − A^{-1} A v_i. Optionally orthogonalizes against previous vectors. `null_vecs` parameter is used as shape/dtype/device template only; values are overwritten with random init.
- **`local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)`** — Block-local Gram-Schmidt orthogonalization via batched QR decomposition. Splits null vectors into coarse-grid blocks, applies QR per block. NPU path avoids >8-dim tensors.
- **`restrict(local_ortho_null_vecs, fine_vec)`** — P^T v_fine = Σ v_fine · null_vec^†. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.
- **`prolong(local_ortho_null_vecs, coarse_vec)`** — P v_coarse = Σ null_vec · v_coarse. Standard path uses 10-dim einsum; NPU path reshapes to ≤8 dims.

**NPU compatibility:** NPU limits tensors to ≤8 dimensions, so restrict/prolong/orthogonalize all have `_npu` variants that use reshape/permute chains to stay within this limit. Cross-validated against standard path (max diff ~1e-7 for float32).

### TileLang Integration (`pyqcu.torch.tools/_einsum.py`, `_matul.py`)

Optional — try/except import at package level; silently degrades if TileLang unavailable.

- **`Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)`** — JIT-compiled TileLang kernel for specific einsum pattern used in Wilson dslash (disabled by default; `tools_Eexyzt_exyzt2Exyzt = False`)
- **`matmul_gpu(M, N, K, block_M, block_N, block_K)`** / **`matmul_cpu(M, N, K, ...)`** — TileLang kernel definitions for matrix multiply benchmarking

Kernels use `warp_size = 128` from `_define`.

### Dtype Conversion Tables (`_define.py`)

- **`np2torch_dtype`**, **`torch2np_dtype`** — bidirectional NumPy ↔ PyTorch dtype maps
- **`torch2tl_dtype`** — PyTorch → TileLang dtype map (float16/32/64 only)

## HDF5 原子写与并发（2026-11-15）

仓库有多套写入路径，契约不同，混用会产生"文件存在但读不了"的资产：

| API | 原子性 | 适用 |
|---|---|---|
| `_io.py::save_tensor_h5(tensor, path)` | **无**：就地 `h5py.File(path,'w')` 截断重写，无 tmp、无 `os.replace`、无 `fsync` | 本进程私有、读不到也无所谓的中间产物 |
| `_io.py::save_dict_h5(path, d)` | `tmp + os.replace` 改名原子，但**无 `fsync`**（数据仍在页缓存） | 单机结果字典、可重算的 report |
| `_io.py::gridoooxyzt2hdf5oooxyzt(...)` | `driver='mpio'` 集体写，每 rank 写自己的 owned slab | 多 rank 分布式张量 |
| `pyqcu/testing/simple/_common.py::write_h5`（:191） | **无**：直接 `h5py.File(path,'w')`，无 tmp/`fsync` | 驱动落 solution/gauge/source；**写生产输入即缺陷来源** |
| `pyqcu/testing/simple/_common.py::write_json`（:262） | **无**：`Path.write_text(...)` 先截断再写，进程被杀留空/半截 JSON | `report.json`、manifest、`last_solve_*.json` |
| `pyqcu/testing/simple/mpi/cuda_pair_reduce_samples.py::atomic_write_json`（:122） | `.partial` + `os.replace`，失败时删 partial；**无 `fsync`** | 结果 JSON 的推荐写法 |
| `pyqcu/tools/_legacy_mg_cache.py::write_legacy_mg_ranked_cache` | `.partial` + `os.replace`，先写容器再写 sidecar；**无 `fsync`** | Legacy-MG cache 根与 manifest |
| `pyqcu/testing/kupl/test_kunpeng_all.py::write_{point,wall}_source` | `.{name}.{pid}.tmp` + `os.replace`；**无 `flush()`/`fsync()`** | 测试源生成 |

- **输入只读**：`data/production_*/**` 与任何 `--gauge/--source` 入参都只读；驱动不得
  原地修补坏文件（`bad object header` 只能换名重生成 + `.corrupt-*` 备份）。
- **solution / report / cache manifest 是原地写风险点**：`write_h5` 写
  `--output` solution、`write_json` 写 `--report`，两者都是就地截断；同一路径被复跑
  或被杀时先毁旧文件再写新文件。需要"旧结果一直可读"时必须改成
  `atomic_write_json` 式 `.partial`+`os.replace`，或在文件名里带 `run_id`。
- **原子写建议**：同目录唯一临时名 → 写完 `flush()` → `os.fsync(fileobj.fileno())`
  → `os.replace` → 父目录 `fsync`。只做 `os.replace` 挡得住"读到半个文件"，挡不住
  掉电/杀进程后的半写数据；`.tmp`/`.partial` 永远不是资产（不要当输入、不要统计）。
  就地 `File(path,'w')` 连"读到半个文件"都挡不住（2026-11-15 的
  `bad object header version number` 见 `pyqcu-testing-data`）。
- **MPI**：`save_tensor_h5`/`save_dict_h5` 不收 communicator，也不做 rank 过滤 →
  只能在单个 rank 调用，或每 rank 用不同路径。所有 rank 写同一路径是"最后写者胜"
  + 交错覆盖，不是并行 I/O。分布式写一律走 `gridoooxyzt2hdf5oooxyzt`（mpio）。
- **多线程**：每次调用新建独立 `h5py.File` 句柄、不持有全局句柄，因此不同线程写
  *不同文件* 是安全的；同一文件并发写不安全。HDF5 的线程安全还取决于库构建
  （threadsafe 与否），不要据此假设可以在线程池里并行写同一文件。
- `HAS_MPI_SUPPORT = check_mpi_support()` 在 `pyqcu.tools` 导入时确定：为真时
  `gridoooxyzt2hdf5oooxyzt` 用 mpio，为假时退回 `comm.gather` 到 root 再单点写
  （root 汇总多 GiB 张量会 OOM，见 `pyqcu-testing-qcu` 的 L48 记录）。鲲鹏 conda
  环境实测 h5py `3.7.0` / HDF5 `1.10.6`、`get_config().mpi=False`、`mpio` driver 不存在
  → `KUPL_HDF5_MPIO_READ` 被驱动主动关闭（`h5py_has_mpi()` 为假时连自动分支都不进），
  读取退回 per-rank 独立 open + hyperslab，ranked cache 只能 `SKIP`。
- **MPI h5py 只做接口检查，不重新编译/安装**（2026-11-15 裁定）：每批开跑前记录
  `h5py.__version__`、`hdf5_version`、`get_config().mpi` 与 `mpio` 是否存在即可；
  后续只做 capability-aware fallback（无 mpio → 不设该变量、不排 ranked cache、I/O 预算
  按"文件字节 × rank 数"估、`prewarm` 当唯一低成本重复读）。不要把"编译/安装 MPI h5py"
  写成下一步，也不引用未验证的离线构建或 ranked cache 尝试产物。
- 校验只做两件事：`h5py.File(p,"r")` 能否打开 + `data` 数据集 `.size` 是否合理
  （矩阵驱动的 `h5_readable()`）。它证明"此刻可读"，不证明写完整；并发写窗口内
  也会通过。

## Logging Convention

`PYQCU::TOOLS::<SUBMODULE>::\n message`
