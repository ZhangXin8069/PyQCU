# AGENTS.md — pyqcu

PyQCU 顶层 Python 包：CUDA 加速格点 QCD 库。实现 Wilson/Clover Dirac 算子、BiStabCG 与多重网格求解器、stout smearing、规范场生成 — MPI 分布于 4D 进程网格。

## 后端分层

1. **Torch/CANN 后端**（`torch/dslash/`、`torch/solver/`、`torch/smear/`、`torch/tools/`）— PyTorch 实现，跑 CPU、CUDA GPU 或昇腾 NPU。
2. **C++ CUDA 后端**（`cuda/` → `cpp/cuda/qcu/`）— 手写 CUDA 内核 + MPI halo 交换，经 Cython 桥（`pyqcu.cuda.qcu`）访问。
3. **KUPL CPU 后端**（`kupl/` → `cpp/kupl/qcu/`）— MPI + OpenMP 独立实现。

公共 `lattice/`、MPI/HDF5/env/budget `tools/`、`solver/multigrid/` 选择器以及
`cann/` 兼容层保留在顶层。顶层 `dslash/`、`smear/`、`solver/`、`tools/` 是旧导入兼容门面。

## 子包

| 包 | 用途 |
|---|---|
| `api/` | 后端无关的几何、配置、solve result、能力、资产 manifest 和 Protocol 契约 |
| `lattice/` | gamma 矩阵、Gell-Mann 矩阵、SU(3) 检查、规范场生成 |
| `torch/dslash/` | Torch Wilson/Clover 算子、hopping/sitting、奇偶预处理、Galerkin 投影 |
| `torch/solver/` | Torch BiCGStab/FGMRES/MR/CA-CG/Lanczos 与 strict/legacy multigrid |
| `torch/smear/` | Torch Stout 与 Wuppertal smearing |
| `torch/tools/` | Torch 线性代数、null vectors、restrict/prolong、strict Galerkin、TileLang |
| `tools/` | 公共 MPI 网格、布局、HDF5 I/O、env/budget、绘图 |
| `solver/multigrid/` | Torch/CUDA/KUPL 统一 `Multigrid` 选择器 |
| `solver/krylov/` | Torch 迭代与 CUDA/KUPL 原生整次 solve 的统一 `Krylov` 选择器 |
| `testing/` | 全组件集成测试 |
| `cuda/` | 到 `libqcu.so` 的 Cython 桥 + 参数常量（`define.py`） |
| `kupl/` | 独立 MPI+OpenMP CPU 后端绑定 |
| `cann/` | Torch 兼容层（NPU 复数运算分解） |
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
cd pyqcu/testing && pytest .
mpirun -np 4 python pyqcu/testing/python/conftest.py
```

## 日志约定

所有模块：`PYQCU::MODULE::SUBMODULE:\n message`
