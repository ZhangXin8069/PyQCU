# 使用

下面的示例使用 `8^4` 小格点，便于在笔记本上快速运行。每个代码块都自校验：
断言自己的结果，而不是打印一个可能错误的数字。

## 规范场与格点常数

```python
import torch
from pyqcu import lattice

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1)
kappa = torch.tensor([0.125])
assert lattice.check_su3(U, tol=1e-3)
```

`kappa` 是 hopping 参数，`kappa = 1 / (2 * mass + 8)`。Clover 系数 `csw`
单独存放（原生 ABI 中为 `argv[7]`）。

## Wilson 与 Clover 算子

```python
from pyqcu.torch import dslash

src = torch.randn((4, 3, *shape), dtype=torch.complex64)
out = dslash.give_wilson(src=src, U=U, kappa=kappa)
assert out.shape == src.shape
assert torch.isfinite(out).all()
```

```python
clover = dslash.make_clover(U=U, kappa=kappa)
assert clover.shape == (4, 3, 4, 3, *shape)
assert torch.isfinite(clover).all()
```

## 一个小型 BiCGStab 求解

```python
from pyqcu.torch import solver

b = torch.randn_like(src)
operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)
x = solver.bistabcg(
    b=b, matvec=operator.matvec, tol=1e-5, max_iter=200, verbose=False)
residual = (
    torch.linalg.vector_norm(operator.matvec(src=x) - b)
    / torch.linalg.vector_norm(b))
assert residual < 1e-3, residual
```

纯 Wilson 算子可以传零 Clover 项，或走 Wilson-only 路径。
`solver.bistabcg` 是 Torch 实现；CUDA/KUPL 的原生整次求解入口通过
`pyqcu.solver.Krylov` 访问。多重网格、原生 Krylov、cache 与 MPI 求解用法见
{doc}`solvers`。

## Stout smearing

```python
from pyqcu.torch import smear

U_smeared = smear.stout_smear(U=U, nstep=1, rho=0.12)
assert U_smeared.shape == U.shape
assert lattice.check_su3(U_smeared, tol=1e-2)
```

## HDF5 往返

```python
import os
import tempfile
from pyqcu.tools import load_tensor_h5, save_tensor_h5

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "fermion.h5")
    save_tensor_h5(src, path)
    restored = load_tensor_h5(path)
assert torch.equal(restored, src)
```

张量布局原样保存，布局约定由调用方负责。`pyqcu.tools.save_tensor_h5` 等
生产工具每次调用新建独立 HDF5 句柄，因此不同线程可以安全地写不同文件。

## 原生 KUPL 后端

KUPL 绑定是惰性的：导入 `pyqcu.kupl` 不会加载库，直到导入
`pyqcu.kupl.qcu`。它按以下顺序查找：

1. 设置了的 `PYQCU_KUPL_LIB`；
2. 安装包内的 `pyqcu/kupl/libqcu.so`；
3. `pyqcu/kupl/build/libqcu.so`；
4. 源码树中的 `cpp/kupl/qcu/libqcu.so`。

```python
# docs-skip: requires a native KUPL library built for this machine
from pyqcu.kupl import qcu
print(qcu.library_path())
```

统一 Krylov 门面接受 `KuplLinearOperator`；参见 `pyqcu.solver.Krylov` 与
`pyqcu.api.BiCGStabControl`。

## 命令行诊断

```bash
python -m pyqcu
python -m pyqcu --json
```

报告会打印包版本与位置、依赖版本，以及 KUPL/CUDA 原生后端是否可用。

## 常见陷阱

* `pyqcu.tools.norm()` 已经返回 Python float；再调用 `.item()` 会抛
  `AttributeError`。
* PyQCU 的奇偶分裂张量压缩的是 **时间** 轴。不要把 QUDA/QDP 费米子直接
  reshape 成 PyQCU parity 布局。
* CUDA 生产约定是一个 Python 进程对应一个 MPI rank 与一张设备；单进程内
  多卡只是测试驱动。
* 为不同 CPU/GPU 架构构建的原生库可能可以导入，但第一次调用就失败；基准
  测试应记录库路径与 SHA256。
