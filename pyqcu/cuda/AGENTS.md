# AGENTS.md — pyqcu.cuda

C++ CUDA 后端（`libqcu.so`）的 Cython 桥包。

## 文件

| 文件 | 用途 |
|---|---|
| `__init__.py` | 使 `pyqcu.cuda` 成为合法 Python 包（2026-07-28 R3 增加） |
| `qcu/qcu.pyx` | Cython 扩展源码 — 包装 `pyqcu.h` 的 C 函数（`applyInitQcu`、`applyWilsonDslashQcu` 等） |
| `qcu/qcu.pxd` | Cython 声明 — `cdef extern` 块匹配 `pyqcu.h`（所有函数声明带 `nogil`） |
| `qcu/qcu_api.pxd` | pxd 别名声明（`from qcu_api cimport X as _c_X`）— pxd 声明名不得与 pyx 内 def 同名，否则被覆盖为 Python 函数 |
| `qcu/qcu.pyi` | 类型 stub（155 行）— 完整类型注解、docstring、默认值 |
| `define.py` | 参数常量（`_LAT_X_`、`_SET_PLAN_` 等）与 dtype 转换工具（`dtype()`、`epytd()`） |
| `_schur_op.py` | `CudaSchurOp` — C++ Schur 奇偶算子封装（每实例独立 params/set_ptrs 副本 + 槽位分配器，多线程安全） |
| `_multi_gpu.py` | `MultiGpuMultigrid` — 多线程多卡 C++ Clover MG 驱动（一线程一卡，单 MPI rank），`build_schur_levels`（33-tensor 粗算子 + h5py 缓存） |
| `_logs.py` | C++ 后端收敛日志解析（2026-08-22 整合）：`parse_mg_log(path)` → (残差列表, PROF_SECTIONS ms 字典, 总迭代数)；`parse_convergence_histories(path, offset)` → (histories, 新偏移) 增量收集逐次求解的 CONVERGENCE_HISTORY；默认路径 `<repo>/logs/clover_multigrid.log` |

## 公共 API

```python
from pyqcu.cuda import qcu      # Cython 桥到 libqcu.so
from pyqcu.cuda import define   # 参数常量、dtype 工具、预构建 params/argv/set_ptrs
```

## Cython 扩展暴露的 C 函数

| 函数 | 用途 | Plan |
|---|---|---|
| `applyInitQcu` / `applyEndQcu` | 分配/释放 scratch 缓冲 | — |
| `applyWilsonDslashQcu` | Wilson dslash | 0 |
| `applyCloverDslashQcu` | Clover dslash | 2 |
| `applyWilsonBistabCgQcu` / `applyWilsonBistabCgDslashQcu` | Wilson BiStabCG 求解器 + 其 dslash | 1 |
| `applyWilsonCgQcu` / `applyWilsonCgDslashQcu` | Wilson CG 求解器 + 其 dslash | 1 |
| `applyCloverBistabCgQcu` / `applyCloverBistabCgDslashQcu` | Clover BiStabCG（需 clover_ee/oo + 逆） | 1 |
| `applyCloverQcu` / `applyCloversQcu` | 构建 Clover 项（及其逆） | 2 |
| `applyDslashQcu` | 组合 Wilson+Clover dslash | 0+2 |
| `applyLaplacianQcu` | Laplacian 算子 | -2 |
| `applyGaussGaugeQcu` | 高斯规范场生成 | -1 |
| `applyMultigridRestrictQcu` / `applyMultigridProLongQcu` | MG restrict/prolong（null 向量） | MG |
| `applyMultigridCoarseDslashQcu` | 粗网格 dslash（hopping + sitting） | MG |
| `applyCloverMultigridQcu` | 全 Clover multigrid V-cycle 求解器 | MG |

所有函数接收 `tensor.contiguous().data_ptr()` 转为 `long long` 的裸指针。

## 参数协议

- **`params`**（int32, 54）— 格点维度（`_LAT_X_`…`_LAT_XYZT_`）、网格大小（`_GRID_X_`…）、数据类型（`_DATA_TYPE_`）、迭代次数（`_MAX_ITER_`）、计划选择（`_SET_PLAN_`）、verbosity（`_VERBOSE_`）、奇偶（`_PARITY_`）、MG 层级配置（`_MG_LEVEL1_X_`…、`_MG_NUM_LEVEL_`）
- **`argv`**（float, 7）— `_MASS_`(0)、`_ATOL_`(1)、`_SIGMA_`(2)、各层 MG 容差(3–6)
- **`set_ptrs`**（int64, 100）— C++ 运行时管理的 scratch 指针

`define.py` 索引常量**必须**与 `cpp/cuda/qcu/include/define.h` 同步。

`define.py` 还提供预构建张量 `params`、`argv`、`set_ptrs`，由求解器代码就地修改。

## 关键：`_SET_INDEX_` 递增

同一 `applyInitQcu`/`applyEndQcu` 生命周期内，连续 C++ 调用之间**必须** `params[define._SET_INDEX_] += 1`。否则 scratch 缓冲复用冲突导致结果错误。

例外：粗网格 dslash 将 `_SET_INDEX_` 重置为 0（不同 MG 层，与细层操作无重叠）。

## dtype 映射

- `define.dtype(data_type)` — QCU 内部常量（`_LAT_C64_`、`_LAT_R32_` 等）→ PyTorch dtype
- `define.epytd(torch_dtype)` — PyTorch dtype → QCU 内部常量
- `define.lat_shape(params)` — 从 params 提取 `[Lt, Lz, Ly, Lx]`

## 计划选择

`_SET_PLAN_N_2_`=-2 Laplacian、`_SET_PLAN_N_1_`=-1 Gauss gauge、`_SET_PLAN0_`=0 Wilson dslash、`_SET_PLAN1_`=1 BiStabCG/CG（及其 dslash）、`_SET_PLAN2_`=2 Clover dslash。

## 调用生命周期

```python
qcu.applyInitQcu(set_ptrs, params, argv)          # 分配
# ... 操作，调用间 _SET_INDEX_ += 1 ...
qcu.applyEndQcu(set_ptrs, params)                  # 释放
```

## 多线程（一线程一卡）约定

- qcu.pyx 全部桥函数：GIL 段取指针（`.contiguous().data_ptr()`）→ `with nogil` 调 C++；指针用**函数内局部 cdef 变量**（线程安全，不再用模块级共享 cdef）。
- pxd 的 cdef extern 声明必须带 `nogil` 关键字；pxd 声明名不得与 pyx 内 def 同名（用 `qcu_api.pxd` 别名 `cimport X as _c_X`，同名会被覆盖为 Python 函数导致 nogil 调用失败）。
- 每线程独立 params/argv/set_ptrs 副本 + `torch.cuda.set_device(dev_id)`（CUDA current device 线程局部）。
- 多线程多卡驱动见 `_multi_gpu.py`（`MultiGpuMultigrid`，单 MPI rank）；单算子见 `_schur_op.py`（`CudaSchurOp`）。

## dev84 平台与算法结论（2026-08-22，详见 examples/qcu/dev84/dev84_report.md）

- **WSL2 内核执行税 ~300µs/内核**（GPU 侧派发，与工作量无关）：细粒度多内核路径
  （逐迭代点积/标量更新）在本箱不可扩展。落地模式：CUDA Graph 段回放
  （8 迭代/段，lattice_clover_multigrid.h `coarse_graph_ensure/run`）+
  零拷贝映射内存读标量（cudaHostAllocMapped，绕开 D2H memcpyAsync）+
  守卫型标量内核（mg_give_1beta_rp 等，防 ρ→0 分裂 NaN）+ SYNC DIET
  （同流序不中途同步）。V-cycle 156→60ms。
- **nullvec 容差语义**：`solver.bistabcg` 默认绝对容差 —— give_null_vecs* 传小 tol
  在大格子上退化为精确逆→归一化噪声向量（‖Sv‖/‖v‖≈谱 RMS，粗空间无效）。
  已修为 if_rtol 相对容差（dev84_1）；诊断方法=隔离测单次校正收缩因子
  ρ_V=||r−S·P·A_c⁻¹·R·r||/||r||（Galerkin 精确粗解），ρ_V≈1 即粗空间无效。
- **16×32×32×48 m=0.05 Schur 谱为连续中等谱**（无孤立低模簇；收敛 ~0.77/iter
  几何式）——聚合粗空间/谱收缩/块 Jacobi 均无法在该格子给出 >2 真实加速比；
  历史小格子高加速比系测量口径问题（指令 9 复证）。

## 已知问题与修复（2026-08-14）

- **conftest.clover.multigrid.py 旧协议修复**（已解决）：该测试原用旧协议
  （set_ptrs base=10 3 槽/层、无 hop_diag、`level1_T = Lt//MG_GRID[3]`、E=12 粗空间）
  与 C++ 新协议（base=30 4 槽 33-tensor、粗层 T = Lt/(2·MG_GRID[3])）不符，
  2L 配置触发 illegal memory access（lattice_clover_multigrid.h:1523）或 nan。
  已重构为 `build_schur_levels`（33-tensor）+ E=48 粗空间；8x8x8x16 2L/2L_r3、
  12³×16 2L 全 PASS（残差 ~1.5e-7）。

- **P100(sm_60) 混合多卡修复**（2026-08-14，2×P100-16GB + V100-32GB 三卡实测）：
  - **C++ LatticeSet::init 不再强制 `cudaSetDevice`**（lattice_set.h）：单 MPI rank
    （多线程多卡）时跳过设备切换——线程已 `torch.cuda.set_device` 绑定本卡，
    强制切到 device 0 会让 P100 线程的内核在 V100 上写 P100 内存 → illegal
    memory access；多进程分布（comm_size>1）仍按 local rank 绑定。
  - **P100 上 torch 无 kernel image**（torch 2.10 仅 sm_70+）：`_multi_gpu.py`
    所有设备张量改用 `torch.empty`（纯 cudaMalloc）+ CPU 生成后 H2D 拷贝；
    禁用 `zeros/randn` 填充内核。C++ 后端 libqcu.so 含 sm_60 SASS，不受影响。
  - **独立问题模式粗算子构建移到 V100 主线程**：worker 内构建含 torch 运算
    （`give_null_vecs_mt`/`build_stencil_mt`）在 P100 不可行；且 worker 不得
    重新 `applyGaussGaugeQcu`（会用模块默认 _SEED_ 覆盖主线程 seed=42+tid
    预生成的规范场 → 粗算子不匹配发散）。
  - **粗算子拷贝保留引用**：`.to(dev)` 跨设备产生新张量，须持有引用再取
    `data_ptr`，否则 GC 回收后悬垂指针 → 4x4x4x8 等小格子 MG 结果 nan。
  - **cooperative launch 按当前设备查询 occupancy**（lattice_clover_multigrid.h）：
    硬编码 device 0（V100 80 SM）会让 P100 线程（56 SM）grid 超限；occupancy/
    sm_count 缓存须按设备分槽（static 数组逐设备初始化）。
  - **性能参数**：8x8x8x16 3 线程×3 卡，`num_restart=5, coarse_max_iter=15,
    coarse_tol_factor=1e5` → MG 加速比 median 2.36、min 2.18、最优 3.06
    （对比多线程 BiStabCG）；单线程 V100 基准 2.19。大格子（16³×16、
    8x16x16x32）MG 本身慢（coarse solve 开销，历史特性），用 r10。
