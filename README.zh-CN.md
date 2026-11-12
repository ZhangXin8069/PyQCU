# PyQCU（中文）

[English](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCU/-/blob/main/README.md) | **中文**

PyQCU 是一个格点 QCD 算子与求解器工具库，提供三条执行路径：

* Torch/CANN 后端，运行在 CPU、CUDA 或昇腾 NPU 上；
* 独立的 C++17 MPI+OpenMP CPU 后端（KUPL）；
* 经 Cython 桥暴露的 CUDA/DCU C++ 后端。

它提供 Wilson/Clover Dirac 算子、stout 与 Wuppertal smearing、
BiCGStab/CG/FGMRES 求解器、Legacy/Strict 多重网格层级、HDF5 工具，以及
四维 MPI rank 网格。

## 安装

```bash
pip install pyqcu
```

要求：Python 3.9+、Linux，以及 `numpy`、`torch`、`mpi4py`、`h5py`
（会自动安装）。构建 KUPL CPU 库需要 C++17 编译器、CMake 与 MPI；可选的 CUDA
库需要 `nvcc`。

发行包是源码包。首次安装时若工具链可用会构建 KUPL，并按机会尝试 CUDA/DCU；
缺少工具链不会让安装失败，纯 Torch 路径仍然可用。

```bash
# CPU-only
PYQCU_BUILD_BACKEND=kupl pip install pyqcu

# Require the KUPL build
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu

# Build KUPL and CUDA on a machine with nvcc
PYQCU_BUILD_BACKEND=cuda pip install pyqcu
```

验证安装：

```bash
pyqcu                 # 安装的 console script；等价于 python -m pyqcu
pyqcu --selftest      # Torch/KUPL/CUDA 三层自检；缺原生后端显示 SKIP
pyqcu --json
```

`pyqcu --selftest` 会在每个可用层上跑一次小规模独立残差求解。缺少 KUPL 或 CUDA
后端会报告 `SKIP`，不会被当成安装失败。

## 文档

* English — https://pyqcu.readthedocs.io/en/latest/
* 中文 — https://pyqcu.readthedocs.io/zh-cn/latest/

站点源文位于 `docs/en/` 与 `docs/zh/`，使用 Sphinx 与 MyST 构建；发布检查会用
`-W` 构建两份站点。

## 代码托管

| 平台 | 用途 | 地址 |
|---|---|---|
| GitLab（gitlab.itp.ac.cn） | 国内开发——主开发仓库 | https://gitlab.itp.ac.cn/ZhangXin8069/PyQCU |
| Gitee | 国内发布——国内镜像 | https://gitee.com/zhangxin8069/PyQCU |
| GitHub | 国外发布——海外镜像 | https://github.com/zhangxin8069/PyQCU |

## 快速开始

```python
import torch
from pyqcu import lattice
from pyqcu.torch import dslash

shape = (8, 8, 8, 8)
U = torch.zeros((3, 3, 4, *shape), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1)
assert lattice.check_su3(U, tol=1e-3)

src = torch.randn((4, 3, *shape), dtype=torch.complex64)
kappa = torch.tensor([0.125])
out = dslash.give_wilson(src=src, U=U, kappa=kappa)
assert out.shape == src.shape
assert torch.isfinite(out).all()
```

公共层包括 `pyqcu.api`（后端无关契约）、`pyqcu.lattice`（gamma/SU(3) 与
规范场/源场生成）、`pyqcu.torch`（Torch 算子与求解器）、`pyqcu.kupl`（CPU
ctypes 绑定）、`pyqcu.cuda`（CUDA/DCU Cython 桥）、`pyqcu.solver`（统一
Krylov/Multigrid 选择器）与 `pyqcu.tools`（MPI/布局、HDF5、环境与绘图工具）。

## 原生后端

`python -m pyqcu` 会报告包位置，以及 KUPL/CUDA 库是否可用。KUPL 绑定依次查找
`PYQCU_KUPL_LIB`、安装包内的 `pyqcu/kupl/libqcu.so`、源码仓库。只有 `nvcc` 与
CUDA 库都可用时才会构建 CUDA 桥；扩展使用 `$ORIGIN` rpath，从安装包内部加载
`libqcu.so`。

发布验证使用 `8x8x16x16` complex64 格点、Gaussian SU(3) 规范场
（`sigma=0.1`、`seed=42`）、`mass=0.17`、`csw=1.0`、`atol=1e-6`。在发布机器
（WSL2、Intel i7-13620H、RTX 4060 Laptop sm_89）上，安装后的 KUPL 与 CUDA
路径都在 26 次迭代内收敛，全格点真残差低于 `3e-7`；complex64 的验收线为
`max(atol, 2048*eps32) ≈ 2.4414e-4`。

## 历史性能

下表记录的是当前 checkout 之外硬件上的测量。跨机器绝对秒数不可比，CPU/GPU
行从不合并成单一加速比。这些运行的原始 JSON 不在当前工作树中；链接的报告与
git 标签是 provenance。

| 用例 | 平台 | 格点/精度 | 结果 | 对照 | 可复现性 |
|---|---|---|---|---|---|
| CUDA BiCGStab vs QUDA | Tesla V100-SXM2-32GB, sm_70 | 24³×64, c64 | 0.147801 s / 18 次迭代 | QUDA 0.156680 s / 19 次 → 1.060× | 历史数据；需 V100 与 `dev103` 时期 checkout |
| KUPL parity-Schur BiCGStab | Intel Xeon E5-2696 v3, 18 核 | 24³×64, c64 | 8 ranks×2 threads：7.532 s / 19 次迭代 | 1 rank×4 threads 19.229 s → 2.553× | 历史数据；需同级别 CPU |
| CUDA Legacy-MG vs QUDA GCR-MG | A100-PCIE-40GB | 32³×64, c64 | 1/2/4 ranks：0.2145 / 0.2195 / 0.2183 s | QUDA 0.4824 / 0.3144 / 0.2875 s → 2.2495× / 1.4323× / 1.3166× | 历史数据；需 A100、L32 LIME、E12 cache |
| CUDA native BiCGStab | A100-PCIE-40GB | 32³×64, c64 | 1 rank 0.1224 s；2 ranks 0.1154 s | QUDA 0.1304 / 0.0961 s → 1.065× / **0.833×** | 历史数据；说明 native 求解器并非普遍更快 |
| CUDA Strict-MG vs QUDA aligned MG | Tesla V100-SXM2-32GB | 8×8×8×16, c64 | 0.021693 s / 10 次迭代 | QUDA 0.144755 s / 8 outer GCR → 6.673× | 时点证据；小格固定开销占主导 |
| CUDA(HIP) Legacy-MG | DCU Pre-Wukong D160A1, gfx906 | 24³×64, c64 | 1/2/4 ranks：0.3017 / 0.4619 / 1.0247 s | QUDA 阻塞；无加速比 | 历史数据；需同型号 DCU 与 DTK |

完整表格（含仅存在于历史标签的行，以及本机 `8x8x16x16` 测量）见性能文档。

## 从源码安装

```bash
git clone git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git
cd PyQCU
source ./env.sh

bash ./install.sh kupl    # CPU KUPL backend
bash ./install.sh cuda    # CUDA backend + Cython bridge
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu
```

测试与基准驱动位于 `pyqcu/testing/`，并有意不进入 PyPI 包：

```bash
cd pyqcu/testing
pytest .
```

## 许可证

MIT。见 `LICENSE`。
