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
