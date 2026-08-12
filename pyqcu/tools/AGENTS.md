# AGENTS.md — pyqcu.tools

工具集：MPI 网格管理、HDF5 I/O、线性代数、张量运算、多重网格转移、TileLang JIT 内核。

## 文件

| 文件 | 用途 |
|---|---|
| `_define.py` | 网格分解/邻居 rank、奇偶分割、维度重排、dtype 表、设备设置、切片助手、素因子分解 |
| `_io.py` | HDF5 MPI 并行 I/O（`driver='mpio'`）+ 串行 gather/scatter 回退 |
| `_linalg.py` | `vdot`/`norm`（经 `_torch`） |
| `_einsum.py` | TileLang einsum 内核（`Eexyzt_exyzt2Exyzt`，可选） |
| `_matul.py` | TileLang 矩阵乘内核（可选） |
| `_multigrid.py` | null 向量生成、局部正交化、restrict/prolong（NPU 兼容） |
| `_roll.py` | 张量滚动 |

## 关键 API

- **MPI 网格**：`give_grid_size()`（素因子分解为 4D 网格）、`give_grid_index(rank)`、`give_rank_plus/minus(ward, rank)` 及四个对角邻居变体、`set_device(device, verbose)`（按 rank 轮询分配）
- **奇偶分割**：`oooxyzt2poooxyzt`（→`[2,...,t,z,y,x//2]`，沿最快维 x 分裂）、`poooxyzt2oooxyzt` 反向；`give_eo_mask`（`(x+y+z)%2` 棋盘，按 shape+device+eo 缓存）
- **维度重排**（HDF5 用 zyxt 序）：`ccdxyzt2ccdptzyx`/`ccdptzyx2ccdxyzt`、`scxyzt2psctzyx`/`psctzyx2scxyzt`
- **Gather/Scatter**：`local_xyzt2whole_xyzt`（`comm.Gather`）、`whole_xyzt2local_xyzt`（`comm.Scatter`）
- **切片**：`slice_dim`/`slice_dim_dim`/`slice_dim_none_dim`（负索引 ward 维）
- **内存**：`to_contiguous_real` — 用 `empty + copy_` 而非 `.contiguous()`（单元素张量正确性）
- **HDF5**：`gridoooxyzt2hdf5oooxyzt`/`hdf5oooxyzt2gridoooxyzt`；`HAS_MPI_SUPPORT` import 时检测。注意 `comm.scatter` 走 pickle，>64⁴ float32 可能撞 2GB 上限，生产用 mpio
- **多重网格**：`give_null_vecs`（逆迭代，`null_vecs` 仅作 shape/dtype 模板）、`local_orthogonalize`（分块 QR）、`restrict`（P^T v，10 维 einsum）、`prolong`（P v）。**NPU 限 ≤8 维**，三函数均有 `_npu` 变体（reshape/permute 链，与标准路径交叉验证 ~1e-7）
- **dtype 表**：`np2torch_dtype`/`torch2np_dtype`/`torch2tl_dtype`

## TileLang（可选）

包级 try/except import，缺失时静默降级。`matmul_gpu`/`matmul_cpu` 用 `warp_size = 128`；`tools_Eexyzt_exyzt2Exyzt = False` 默认禁用。

## 日志约定

`PYQCU::TOOLS::<SUBMODULE>::\n message`
