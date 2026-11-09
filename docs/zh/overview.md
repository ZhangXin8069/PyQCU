# 概览

PyQCU 是一个格点 QCD 算子与求解器工具库，提供 Wilson/Clover Dirac 算子、
stout 与 Wuppertal smearing、BiCGStab/CG/FGMRES 求解器、Legacy/Strict 多重网格
层级，以及 MPI rank 网格层。同一套 Python 公共接口驱动三条执行路径：

* Torch 后端（`pyqcu.torch`），通过 `pyqcu.cann` 兼容层运行在 CPU、CUDA 或
  昇腾 NPU 上；
* KUPL CPU 后端（`pyqcu.kupl`），独立的 C++17 MPI+OpenMP 实现，经 `ctypes`
  加载；
* CUDA/DCU 后端（`pyqcu.cuda`），C++/CUDA 实现，经 Cython 桥暴露。

本版本为 `0.1.6`。发行包是源码包：安装机具备工具链时现场构建 KUPL 库，
CUDA/DCU 库则按机会构建。

## 数据布局

张量采用通道在前、四个时空轴在最后的布局：

| 对象 | 形状 |
|---|---|
| 规范场 | `[3, 3, 4, X, Y, Z, T]` |
| 费米子场 | `[4, 3, X, Y, Z, T]` |
| Clover 项 | `[4, 3, 4, 3, X, Y, Z, T]` |
| 奇偶分裂场 | `[2, 4, 3, X, Y, Z, T/2]` |

ward 索引用负整数（`wards["x"] == -4`、`wards["t"] == -1`），因此张量带额外
前缀轴时仍然正确。

## 快速自检

```python
import torch
from pyqcu import lattice

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=42, sigma=0.1)
assert U.shape == (3, 3, 4, *shape)
assert lattice.check_su3(U, tol=1e-3)
print("gauge field OK", tuple(U.shape))
```

安装好包与依赖后，Torch 后端总是可导入。原生 KUPL/CUDA 是否可用由下面的
命令报告：

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

`pyqcu` 是安装后的 console script；`python -m pyqcu` 等价。
`--selftest` 会跑 Torch/KUPL/CUDA 三层，缺少原生后端时报告 `SKIP`。

## 包结构

| 模块 | 用途 |
|---|---|
| `pyqcu.api` | 后端无关的配置、结果、能力与 Protocol 对象 |
| `pyqcu.lattice` | Gamma/Gell-Mann 矩阵、SU(3) 检查、规范场与源场生成 |
| `pyqcu.torch` | Torch/CANN 的 dslash、smearing、求解器与数值工具 |
| `pyqcu.kupl` | KUPL CPU 库的 `ctypes` 绑定与多重网格封装 |
| `pyqcu.cuda` | CUDA/DCU 库的 Cython 桥与多重网格封装 |
| `pyqcu.solver` | 跨后端的统一 `Krylov` 与 `Multigrid` 选择器 |
| `pyqcu.tools` | MPI/布局、HDF5、环境、预算与绘图工具 |

## 参见

* {doc}`usage`：算子、smearing、求解器与 HDF5 往返。
* {doc}`solvers`：迭代、多重网格与原生求解路径的选择与完整调用。
* {doc}`install`：安装、原生构建变量与验证。
* {doc}`performance`：历史跨平台测量及其可复现性状态。
* {doc}`troubleshooting`：安装、原生后端与求解器故障排查。
