# AGENTS.md — pyqcu.tools

MPI 网格、数据布局、HDF5 I/O、环境/预算和绘图公共工具。Torch 专用数值工具已迁移到
`pyqcu/torch/tools/`；本目录顶层 API 对旧导入保留惰性兼容导出。

## 文件

| 文件 | 用途 |
|---|---|
| `_define.py` | MPI 网格尺寸分解、rank 邻居、奇偶拆分（oooxyzt2poooxyzt/poooxyzt2oooxyzt）、维度重排（ccdxyzt↔ccdptzyx、scxyzt↔psctzyx）、dtype 转换表、设备设置、slice 工具、质因数分解 |
| `_io.py` | HDF5 I/O（`driver='mpio'` MPI 并行 I/O + 串行 gather/scatter 回退）+ dict 级序列化（`save_dict_h5`/`load_dict_h5`） |
| `_env.py` | 环境与资源快照（2026-08-22 整合自 logs/dev78_2 与 pyqcu/testing/qcu/multigrid/scaling，后者 2026-10-15 删除）：`git_snapshot`/`gpu_snapshot`/`gpu_used_mb`/`rss_kb`/`nullvec_cache_dir`/`cache_disk_mb`/`dump_env_h5`（写 env.h5 经 save_dict_h5） |
| `_budget.py` | 显存/RSS/磁盘预算模型（2026-08-22 整合，dev74 主本，源套件 scaling 2026-10-15 删除）：`vram_model`(cold)/`vram_model_warm`/`rss_model`/`disk_cache_bytes`(E 参数化)/`fit_from_bench`(json 实测拟合 α)/`budget_table(mode, vram_gb)`；常量 CONST_PER_V=24192 B/V、α=2.6 KB/V、β=512 MB |
| `_bench_plots.py` | 基准图表共用层（2026-10-15 整合自 dev73/dev74/volume24 的 5 处绘图副本，源套件 legacy/scaling 已删除）：调色板常量 `BENCH_PALETTE`/`BENCH_SEQ_BLUE`/`BENCH_INK*`/`BENCH_GRID`/`BENCH_SURF`，`mpl_cjk_setup`/`style_axes`/`save_figure`/`load_json`/`short_label`/`label_of`，以及 `plot_convergence`/`plot_hotspot`/`plot_speedup`/`plot_time`/`plot_sweep_curves`/`plot_bench_report`。**顶层不 import matplotlib**（惰性在函数内），故 `pyqcu.tools` 在无 matplotlib 环境仍可导入 |
| `../torch/tools/` | Torch 专用线性代数、multigrid、strict Galerkin、TileLang 与分层存储实现 |

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

### 线性代数（`pyqcu/torch/tools/_linalg.py`）
`norm(input, p='fro', dim=None, keepdim=False)`、`vdot(input, other)`（Σ conj(a_i)·b_i）

### 多重网格工具（`pyqcu/torch/tools/_multigrid.py`）
- `give_null_vecs(null_vecs, matvec, bistabcg, normalize, ortho_r, ortho_null_vecs, verbose)` — 逆迭代生成近零空间向量 v_i = v_i − A⁻¹ A v_i；`null_vecs` 仅作 shape/dtype/device 模板
- `give_null_vecs_mt(..., batch_matvec=...)` — 多右端 batch BiCGStab 默认目标 6 个场/块，并按 `cudaMemGetInfo` 实时缩块；生产 24³×64 c64 的 12 向量构建由 chunk=4 的 13–15 s 降至 chunk=6 的约 10 s，chunk=12 会耗尽 32 GiB V100。
- `local_orthogonalize(null_vecs, coarse_lat_size, normalize, verbose)` — 块局部 Gram-Schmidt（批量 QR）；NPU 路径避免 >8 维张量
- `restrict(local_ortho_null_vecs, fine_vec)` — P^T v_fine = Σ v_fine·null_vec^†（标准 10 维 einsum；NPU 重排 ≤8 维）
- `prolong(local_ortho_null_vecs, coarse_vec)` — P v_coarse = Σ null_vec·v_coarse

**NPU 兼容**：NPU 张量 ≤8 维，restrict/prolong/orthogonalize 都有 `_npu` 变体（reshape/permute 链）。与标准路径交叉验证（float32 最大差 ~1e-7）。

### TileLang（`pyqcu/torch/tools/_einsum.py`、`_matul.py`）
可选 — 统一通过 `pyqcu._tilelang.import_tilelang()` 探测；不可用时每进程只 warning
一次并缓存。`Eexyzt_exyzt2Exyzt` CPU/无 TileLang 路径回退 `torch.einsum`
（Wilson dslash 用，默认禁用：`tools_Eexyzt_exyzt2Exyzt = False`）；
`matmul_gpu`/`matmul_cpu` 缺依赖时抛 `TileLangUnavailable`。内核用
`warp_size = 128`。

### dtype 表（`_define.py`）
`np2torch_dtype`、`torch2np_dtype` 双向映射；`torch2tl_dtype`（仅 float16/32/64）

## 日志约定

`PYQCU::TOOLS::<SUBMODULE>::\n message`

### 33-tensor Schur stencil build（`pyqcu/torch/tools/_multigrid.py`）

- `PAIRS`/`SIGN` 常量；`build_stencil(matvec, lonv, E, e, lat_fine_odd, lat_coarse_odd, dt, device)` 单线程；
  `build_stencil_mt(matvec_ops, ..., nthreads)` 多线程（每线程一个 matvec 算子，探测点写集不相交，线程安全）；
  `apply_stencil(hop_nn, hop_diag, sit, v_c)` Python 参考实现（A_c = P^T S P）。
- `BatchedLocalSchur._flat_window_index` 将周期窗口 `[K,4,W]` 一次展平，`_slicem` 用 expanded base + `torch.gather`
  一次复制全部 K 个窗口；每次算子调用只构造一次索引。V100、24³×64 c64 微基准 K=4 约
  `4.027 ms -> 0.272 ms`（14.81x），wrap/non-wrap 逐位一致；生产 cold stencil
  `598.107 s -> 323.745 s`（45.87%）。`W=8` 已通过 full-grid probe 与周期边界回归，
  同形状 local-op 中位 `27.337 ms -> 13.725 ms`。
- 由 dev73/mg_stencil_build.py 与 test12/main.py::build_stencil_mt 合并迁移。

### HDF5 本地持久化（`_io.py`）

- `save_tensor_h5(tensor, file_name, dataset)` / `load_tensor_h5(file_name, dataset, device)` —
  每次调用独立 File 句柄（with 语句），多线程安全；null-vector/粗网格算子缓存 `.h5` 用。
- 多 dataset 缓存须**单句柄一次写完**（`with h5py.File(f,'w') as f: create_dataset(...)` 多个），
  逐 dataset 用 'w' 模式会覆盖重建文件导致前序 dataset 丢失。

### dict 级 HDF5 序列化（`_io.py`，2026-08-22 整合自 logs/dev78_2）

- `save_dict_h5(path, d, verbose)` / `load_dict_h5(path, verbose) -> dict|list` —
  嵌套 dict（标量/bool/complex/str/None/list/tuple/ndarray/Tensor/嵌套 dict/dict 列表）
  ↔ .h5；单句柄 tmp + `os.replace` 原子替换，多线程安全。
- 编码：标量/字符串 → attrs（np.generic 先 `.item()`）；数组/数值列表 → dataset `d_<key>`；
  变长数值列表 → vlen f8；**字符串列表 → vlen utf-8**（h5py 无定长 unicode 转换路径）；
  dict 列表 → 数字 key 子组，读取对称还原为 list。
- **往返对称**：读取自动剥 `d_` 前缀（dev78_2 原版需消费方自行剥，库级已修复）。
