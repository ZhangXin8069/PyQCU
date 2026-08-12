# AGENTS.md — pyqcu.tools

MPI 网格管理、HDF5 I/O、线性代数、张量操作、多重网格转移与 TileLang JIT 内核的工具模块。

## 文件

| 文件 | 用途 |
|---|---|
| `_define.py` | MPI 网格尺寸分解、rank 邻居、奇偶拆分（oooxyzt2poooxyzt/poooxyzt2oooxyzt）、维度重排（ccdxyzt↔ccdptzyx、scxyzt↔psctzyx）、dtype 转换表、设备设置、slice 工具、质因数分解 |
| `_io.py` | HDF5 I/O（`driver='mpio'` MPI 并行 I/O + 串行 gather/scatter 回退） |
| `_linalg.py` | 向量点积（`vdot`）与范数（`norm`） |
| `_einsum.py` | TileLang JIT einsum 内核 — `Eexyzt_exyzt2Exyzt`（可选，try/except 导入） |
| `_matul.py` | TileLang 矩阵乘内核 `matmul_gpu`/`matmul_cpu`（可选） |
| `_multigrid.py` | Null 向量生成（`give_null_vecs`）、局部正交化（`local_orthogonalize`）、restrict/prolong — 全部带 NPU 兼容回退 |
| `_roll.py` | 张量滚动工具 |

## 导出 API 要点

### MPI 网格（`_define.py`）
`give_grid_size()`（质因数分解自动将通信子大小分解为 4D 网格）、`give_grid_index(rank)`、`give_rank_plus/minus/plus_plus/plus_minus/minus_minus/minus_plus(ward, rank)`、`set_device(device, verbose)`（按 rank 轮询分配 CUDA/NPU 设备）

### 奇偶拆分
- `oooxyzt2poooxyzt(input, verbose) → [2, ..., t, z, y, x//2]` — 标准布局 → 奇偶拆分，按 (x+y+z+t)%2 分离，沿最快变化（x）维拆分
- `poooxyzt2oooxyzt(...) → [..., t, z, y, x]` — 逆操作

两者都支持 NPU（显式实虚处理）。

### 偶奇掩码
`give_eo_mask(oootzy_t_p, eo, verbose)` — `(x+y+z) % 2` 棋盘格，结果按 shape+device+eo 键缓存。

### 维度重排（HDF5 文件序 zyxt，最快→最慢：t,z,y,x）
- `ccdxyzt2ccdptzyx` → `[c,c,d,p,t,z,y,x]`；`ccdptzyx2ccdxyzt` 逆
- `scxyzt2psctzyx` → `[p,s,c,t,z,y,x]`；`psctzyx2scxyzt` 逆

### MPI Gather/Scatter
`local_xyzt2whole_xyzt(local, root)`（root 汇总全张量）、`whole_xyzt2local_xyzt(dtype, device, whole_shape, whole_array, root)`（分发到各 rank）

### 内存工具
`to_contiguous_real(tensor, channel, *shape)` — 提取实/虚通道并返回真正 stride-1 连续实张量。用 `empty + copy_` 而非 `.contiguous()`（单元素张量正确性）。

### HDF5 I/O（`_io.py`）
- `gridoooxyzt2hdf5oooxyzt(input, file_name, lat_size, verbose)` — 写分布式张量；MPI 路径 `driver='mpio'`，串行路径 `comm.gather` 到 root
- `hdf5oooxyzt2gridoooxyzt(file_name, lat_size, device, verbose)` — 读；MPI 路径 mpio，串行路径 root 读 + `comm.scatter`

**MPI 支持检测**：import 时 `HAS_MPI_SUPPORT = check_mpi_support()`（测试 h5py 配置与 mpio 建文件）。可手动覆盖。

**串行回退注意**：`comm.scatter` 用 pickle，超大格点（>64⁴ float32）可能触及 2GB 限制。生产优先 MPI I/O。

### 线性代数（`_linalg.py`）
`norm(input, p='fro', dim=None, keepdim=False)`、`vdot(input, other)`（Σ conj(a_i)·b_i）

### 多重网格工具（`_multigrid.py`）
- `give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)` — 逆迭代生成近零空间向量 v_i = v_i − A⁻¹ A v_i；`null_vecs` 仅作 shape/dtype/device 模板
- `local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)` — 块局部 Gram-Schmidt（批量 QR）；NPU 路径避免 >8 维张量
- `restrict(local_ortho_null_vecs, fine_vec)` — P^T v_fine = Σ v_fine·null_vec^†（标准 10 维 einsum；NPU 重排 ≤8 维）
- `prolong(local_ortho_null_vecs, coarse_vec)` — P v_coarse = Σ null_vec·v_coarse

**NPU 兼容**：NPU 张量 ≤8 维，restrict/prolong/orthogonalize 都有 `_npu` 变体（reshape/permute 链）。与标准路径交叉验证（float32 最大差 ~1e-7）。

### TileLang（`_einsum.py`、`_matul.py`）
可选 — 包级 try/except 导入，不可用时静默降级。`Eexyzt_exyzt2Exyzt`（Wilson dslash 用，默认禁用：`tools_Eexyzt_exyzt2Exyzt = False`）、`matmul_gpu`/`matmul_cpu`。内核用 `warp_size = 128`。

### dtype 表（`_define.py`）
`np2torch_dtype`、`torch2np_dtype` 双向映射；`torch2tl_dtype`（仅 float16/32/64）

## 日志约定

`PYQCU::TOOLS::<SUBMODULE>::\n message`
