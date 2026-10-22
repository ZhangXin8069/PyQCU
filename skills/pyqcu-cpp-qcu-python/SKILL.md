---
name: pyqcu-cpp-qcu-python
description: 当任务涉及 cpp/cuda/qcu/python、C API、qcu_api.pxd、qcu.pyx 或 qcu.pyi 同步时使用。
---
# cpp/cuda/qcu/python

`cpp/cuda/qcu/python` 是 C++ CUDA 后端与 Python/Cython 之间的 ABI 边界。

CUDA 生产程序保持一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU。ABI 与
`nogil` 支持不表示应在单进程内创建多设备生产 worker；`MultiGpuMultigrid`
仅限显式调用的测试路径。完整规则见
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology)。

## Files

**边界两侧分居两处**——本目录只有 C 侧声明，Cython/Python 侧在 `pyqcu/cuda/qcu/`：

| File | 位置 | Purpose |
|------|------|---------|
| `pyqcu.h` | `cpp/cuda/qcu/python/` | C 侧 `extern "C"` 声明——37 个入口，字段指针统一以 `long long` 传递，包含 legacy 与 Strict 全部入口；由 `qcu.pyx` 用 `cdef extern from "pyqcu.h"` 引入 |
| `qcu_api.pxd` | `pyqcu/cuda/qcu/` | **权威** Cython 声明（38 行）；每个 C 调用必须带 `nogil`，`applyInitQcu` 另用 `except +` 转译 C++ 初始化异常，与 `pyqcu.h` 逐项同步 |
| `qcu.pxd` | `pyqcu/cuda/qcu/` | `qcu_api.pxd` 的逐字节副本（同 md5），供按模块名 `qcu` 引用时使用 |
| `qcu.pyx` | `pyqcu/cuda/qcu/` | Python 包装：`from qcu_api cimport ...` 全部 37 个入口；扩展以 C++ 编译，负责控制数组长度/dtype/device/contiguous 门禁与结果转换 |
| `qcu.pyi` | `pyqcu/cuda/qcu/` | 390 行类型存根（IDE 支持）；`pyqcu/cuda/qcu.pyi` 是指向它的符号链接 |

`pyqcu.h` 必须与 `pyqcu/cuda/qcu/qcu_api.pxd` 完全同步；声明、参数顺序或返回类型
不一致可能导致静默内存破坏。**核对方式**：数两侧的声明条数——
`grep -cE '^(void|int|double) [A-Za-z_]+Qcu' cpp/cuda/qcu/python/pyqcu.h` 与
`qcu_api.pxd` 中 `cdef extern` 块内的函数行数，两者应相等（当前均为 37）；
再抽查每个签名的参数个数。`apply_multigrid_strict.cu` 另导出 `testMultigridStrictGlobalReductionQcu`，
它**不在** `pyqcu.h` 中——加新边界函数时不要把「.cu 里有定义」当作「ABI 已声明」。

## Shared flat-array ABI

所有需要运行时控制的入口共享以下 CPU 控制数组：

- `set_ptrs`: `int64[100]`，低槽位为 `LatticeSet`/scratch 句柄；Strict transition
  `t` 使用 `60+4*t+0..3` 存放 `V`、可选 raw `Y`、`Yhat`、`(X,X^-1)`，槽位 `80`
  存放持久 Strict hierarchy 句柄。
- `params`: `int32[58]`，索引常量由 `pyqcu/cuda/define.py` 与
  `cpp/cuda/qcu/include/define.h` 镜像；54、55、56、57 分别是
  `_MG_USE_GCR_`、`_MG_USE_DEFLATE_`、`_MG_MU_PRE_`、`_MG_USE_INIT_GUESS_`。
  `params[57]` 只能为 0/1：0 表示冷启动，1 表示从调用方预填的 `fermion_out`
  奇半场读取 x0，不改变数组长度。
- `argv`: 8 元实数张量：mass、atol、sigma、四个 MG 容差、csw；c64 使用
  `float32`，c128 使用 `float64`。csw 位于槽 7，Clover 系数为
  `-kappa*csw/8`，且 0 是合法值。

数据张量必须在取 `data_ptr()` 前满足所需 dtype、device、shape 和 contiguous
约束；Cython 在 `with nogil` 段调用 C API。

## Strict C API surface

Strict 入口分为：

- full-coarse `X/Yhat` 应用与 compact MATPC；
- full↔target-parity 的 prepare/reconstruct，以及 compact fine ↔ full coarse 的 R/P；
- 持久层级的 init、recursive V-cycle、end；
- 带 MPI 全局归约的 fused restarted right-FGMRES。

Strict primitive 返回 `int` 状态；init/V-cycle/fused FGMRES 另通过
`unsigned long long*` 返回其自有 arena 的精确字节数，fused 入口还返回迭代次数、
收敛标志和最终 true residual。Cython 声明必须保持 `nogil`，且 Strict 生命周期
内复用同一 `params[_SET_INDEX_]` 槽位。普通事务顺序为
`applyInitQcu → operation → applyEndQcu → _SET_INDEX_ += 1`；不要在 end
前递增，否则会释放空槽并泄漏当前 `LatticeSet`。
