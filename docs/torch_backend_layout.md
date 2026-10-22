# Torch 后端目录与公共层边界

## 目标

PyQCU 的 Python 实现按执行后端分层。Torch/CANN 实现统一位于
`pyqcu/torch/`，CUDA 与 KUPL 分别位于 `pyqcu/cuda/`、`pyqcu/kupl/`。
后端无关的物理定义、共享协议和选择器保留在 `pyqcu/` 顶层。

## 目录

| 路径 | 性质 | 内容 |
|---|---|---|
| `pyqcu/torch/dslash/` | Torch 实现 | Wilson/Clover dslash、hopping/sitting、Galerkin 投影 |
| `pyqcu/torch/solver/` | Torch 实现 | BiCGStab、FGMRES、MR、CA-CG、Lanczos、strict/legacy multigrid |
| `pyqcu/torch/smear/` | Torch 实现 | stout 与 Wuppertal smearing |
| `pyqcu/torch/tools/` | Torch 实现 | 线性代数、null vectors、restrict/prolong、strict Galerkin、TileLang |
| `pyqcu/lattice/` | 公共 | gamma/Gell-Mann、ward、SU(3)、源场和规范场定义 |
| `pyqcu/tools/` | 公共 | MPI 网格、布局、HDF5、env、budget、绘图 |
| `pyqcu/solver/multigrid/` | 公共选择器 | `Multigrid(variant=..., backend=...)` |
| `pyqcu/cann/` | 公共兼容层 | CPU/CUDA/NPU 的 Torch 复数运算适配 |
| `pyqcu/cuda/` | CUDA 后端 | Cython 桥、runtime、strict cache/MPI、CUDA multigrid |
| `pyqcu/kupl/` | KUPL 后端 | ctypes ABI 与 CPU multigrid |

## Canonical 导入

```python
from pyqcu import lattice, tools
from pyqcu.torch import dslash, smear, solver
from pyqcu.torch.tools import norm, vdot, local_orthogonalize
from pyqcu.solver.multigrid import Multigrid

operator = dslash.operator(U=U, clover_term=clover, kappa=kappa)
smoothed = smear.stout_smear(U, nstep=2)
mgs = Multigrid(variant="strict", backend="torch", **hierarchy_options)
```

顶层 `pyqcu.dslash`、`pyqcu.smear`、`pyqcu.solver` 和 `pyqcu.tools` 保留旧导入兼容。
它们不承载 Torch 实现文件；`pyqcu.solver.multigrid` 是三后端共同选择的例外。

## 约束

1. 新 Torch 数值实现只能放入 `pyqcu/torch/`，不得重新写入顶层兼容门面。
2. 顶层 `pyqcu.tools` 只直接实现公共协议；Torch 专用函数通过惰性兼容导出访问。
3. `Multigrid` 选择器不包含算法实现；`StrictMultigrid`/`LegacyMultigrid` 仍由各后端拥有。
4. CUDA/KUPL 复用 Torch 参考算法时必须显式导入 `pyqcu.torch.*`。
5. 历史日志与外部 QUDA 对照路径不属于本次目录规则。

## 兼容性

- 旧 `from pyqcu import dslash, solver, smear, tools` 仍可用。
- 旧 `from pyqcu.solver import StrictMultigrid` 等公开类仍可用。
- 私有模块导入 `pyqcu.solver._...` 或 `pyqcu.tools._...` 不再作为规范路径；仓内代码已迁移到 `pyqcu.torch.*`。

