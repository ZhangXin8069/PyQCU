# AGENTS.md — pyqcu

PyQCU 顶层 Python 包：CUDA 加速格点 QCD 库。实现 Wilson/Clover Dirac 算子、BiStabCG 与多重网格求解器、stout smearing、规范场生成 — MPI 分布于 4D 进程网格。

## 两层架构

1. **纯 Python**（`dslash/`、`solver/`、`smear/`）— PyTorch 实现，跑 CPU、CUDA GPU 或昇腾 NPU（经 `pyqcu.cann`）。
2. **C++ CUDA 后端**（`cuda/` → `cpp/cuda/qcu/`）— 手写 CUDA 内核 + MPI halo 交换，经 Cython 桥（`pyqcu.cuda.qcu`）访问。

multigrid 可混合两层：最细层平滑用 C++ 后端（`with_cuda_qcu=True`），更粗层用纯 Python。

## 子包

| 包 | 用途 |
|---|---|
| `lattice/` | gamma 矩阵、Gell-Mann 矩阵、SU(3) 检查、规范场生成 |
| `dslash/` | Wilson & Clover Dirac 算子、hopping/sitting 分解、奇偶预处理、粗网格 Galerkin 投影 |
| `solver/` | BiCGStab(l) 求解器、自适应多重网格 (AMG) V-cycle 求解器、GMRES 占位 |
| `smear/` | Stout 规范场 smearing（迭代、支持 MPI） |
| `tools/` | MPI 网格工具、HDF5 I/O（并行+串行）、einsum（TileLang JIT）、线性代数、multigrid restrict/prolong/null-vectors |
| `testing/` | 全组件集成测试 |
| `cuda/` | 到 `libqcu.so` 的 Cython 桥 + 参数常量（`define.py`） |
| `cann/` | 昇腾 NPU 的 Torch 兼容层（复数运算分解） |
| `dtk/` | DCU/ROCm 后端占位（暂无实现） |
| `maca/` | Maca 后端占位（暂无实现） |

## 关键约定

所有代码 `import pyqcu.cann as _torch` 而非直接 `import torch`。CPU/CUDA 上委托 torch；NPU 上把复数运算分解为实虚部（昇腾 NPU 不原生支持复数张量）。

## 数据布局约定

| 张量 | 形状 | 说明 |
|---|---|---|
| 规范场 (U) | `[3, 3, 4, Lx, Ly, Lz, Lt]` | `[color, color, direction, x, y, z, t]` |
| 费米子场 | `[4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, x, y, z, t]` |
| Clover 项 | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | `[spin, color, spin, color, x, y, z, t]` |
| 奇偶拆分（前缀 p） | `[2, ...原...]` | `p=0` 偶格点、`p=1` 奇格点（前置维度） |
| 展平 spin×color | `[12, ...]` 或 `[E, ...]` | E = 每格点自由度 |

时空维永远是最后 4 轴（`...xyzt` 布局）。Ward 索引用负整数（`wards['x'] = -4`、`wards['t'] = -1`），对任意前缀维度鲁棒。

## 构建与运行

```bash
source ./env.sh                # LD_LIBRARY_PATH、PYTHONPATH、MPI 标志
bash ./build.sh                # 构建 libqcu.so（C++ CUDA 后端）
bash ./install.sh              # 就地构建 Cython 扩展

# 测试
cd examples && pytest .
mpirun -np 4 python examples/pyqcu/conftest.py
```

## 日志约定

所有模块：`PYQCU::MODULE::SUBMODULE:\n message`
