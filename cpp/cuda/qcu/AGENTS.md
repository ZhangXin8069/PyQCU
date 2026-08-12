# AGENTS.md — cpp.cuda.qcu

PyQCU 的主 C++ CUDA 后端。手工调优 CUDA 内核 + MPI halo 交换，实现 Wilson/Clover 狄拉克算子、BiStabCG/CG 求解器、多重网格与规范场生成。

## 构建

```bash
source ./env.sh   # CUDA toolkit 路径、MPI 等
bash ./make.sh    # 软链 CMakeLists-nv.txt → CMakeLists.txt，然后 cmake + make
```

产物：`libqcu.so`（Cython 桥动态加载）。

## 源码组织

```
include/   — 26 个模板化头文件（CUDA 内核内联；define.h 必须镜像 pyqcu/cuda/define.py）
src/       — .cu 文件：#include 对应头 + 实例化模板 + 内核启动封装
python/    — pyqcu.h：C API 声明（extern "C"），必须与 pyqcu/cuda/qcu/qcu.pxd 完全一致
logs/      — 本地运行日志（gitignored）
```

## 参数协议（来自 Python 的扁平数组）

- `params` (int32[54])：格点维度、网格、数据类型、迭代数、plan、MG 层配置；`_SET_PLAN_`(idx 16) 选内核 plan（-2 Laplacian / -1 Gauss 规范场 / 0 Wilson / 1 BiStabCG·CG / 2 Clover）
- `argv` (float[7])：mass、atol、sigma、MG 容差
- `set_ptrs` (int64[100])：scratch 指针

`include/define.h` 的索引常量必须与 `pyqcu/cuda/define.py` 同步。

## Clover 多重网格 5-stream 架构

```
main(strm): dslash（fine/coarse_dslash_op）
_a_: dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
_b_: give_1rho_prev → give_x_o
_c_: dot(t,s)、收敛检查 dot(r,r)
_d_: dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
```

## 关键不变量（bug 修复沉淀，勿破坏）

1. 标量只存于 `device_vals` — 迭代循环内禁止 host→device 标量 memcpy
2. 每迭代底部**全 5 stream 同步**后再进下一迭代
3. dot 积用 `_send_tmp_` scratch：cublasDot → scratch slot 7 → MPI_Allreduce → 拷贝到目标（禁止直接把 cublasDot 写目标）
4. `mpi_real_type<T>()` 模板按模板类型分发 `MPI_FLOAT`/`MPI_DOUBLE`
5. `run_mpi` 用阻塞 `MPI_Sendrecv`（无需 `MPI_Wait`；仅 `run_mpi_non_block` 需要）

## 块大小

`define.h` 的 `_BLOCK_SIZE_`：小格点测试用 8/16，NVIDIA 生产用 128，AMD DCU 生产用 256。
