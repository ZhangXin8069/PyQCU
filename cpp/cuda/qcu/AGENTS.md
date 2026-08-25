# AGENTS.md — cpp/cuda/qcu

PyQCU 主 C++ CUDA 后端。手写 CUDA 内核 + MPI halo 交换：Wilson/Clover Dirac 算子、BiStabCG/CG 求解器、multigrid、规范场生成。

## 构建

```bash
source ./env.sh       # CUDA toolkit 路径、MPI 等
bash ./make.sh        # symlink CMakeLists-nv.txt → CMakeLists.txt, 然后 cmake + make
```

产物：`libqcu.so` — 由 Cython 桥加载的动态库。

## 源码组织

- `include/` — 26 个模板头（CUDA 内核内联）；`define.h` 必须镜像 `pyqcu/cuda/define.py`
- `src/` — .cu 文件：#include 头 + 模板实例化 + 内核启动封装
- `python/pyqcu.h` — C API 声明（extern "C"），必须与 `pyqcu/cuda/qcu/qcu.pxd` 精确同步

## 参数协议

- `params` (int32[58])：格点维度、网格大小、数据类型、迭代次数、计划选择、MG 层级配置、`_MG_USE_INIT_GUESS_`(57, x0 热启动)
- `argv` (float[7])：mass、atol、sigma、MG 容差
- `set_ptrs` (int64[100])：scratch 缓冲指针

`_SET_PLAN_` (params[16])：-2 Laplacian、-1 Gauss gauge、0 Wilson dslash、1 BiStabCG/CG、2 Clover dslash。

## Clover Multigrid 5-stream 架构

```
main (strm):   dslash 操作 (fine/coarse_dslash_op)
_a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_:           give_1rho_prev → give_x_o
_c_:           dot(t,s), 收敛检查 dot(r,r)
_d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## 关键不变量（来自 bug 修复）

1. 标量只存在于 `device_vals` — 迭代循环内禁止 host→device 标量 memcpy
2. 每轮迭代底部全流同步 — 同步全部 5 个流
3. 点积用 `_send_tmp_` scratch — cublasDot → scratch slot 7 → MPI_Allreduce → 复制到目标（绝不直接写目标）
4. `mpi_real_type<T>()` 模板 — 按模板类型分发 `MPI_FLOAT`/`MPI_DOUBLE`
5. `run_mpi` 用阻塞 `MPI_Sendrecv` — 无需 `MPI_Wait`（仅 `run_mpi_non_block` 需要）

## Block Size

`_BLOCK_SIZE_`（define.h）：小格点测试用 8/16，NVIDIA 生产 128，AMD DCU 生产 256。
