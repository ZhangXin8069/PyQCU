# PyQCU Python 环境

PyQCU 要求 Python ≥ 3.10。核心依赖为 PyTorch、mpi4py、h5py 和 NumPy；测试另需
pytest。Cython 只用于 CUDA Cython bridge，纯 KUPL 源码安装不导入 Cython；
CUDA C++ 后端还需要与目标 GPU 匹配的 CUDA toolkit；DCU 构建需要匹配的
ROCm/HIP PyTorch 以及 DTK nvcc-compatible compiler。

TileLang 是可选依赖，仅用于 JIT 内核实验。缺少时 `pyqcu._tilelang` 只发出一次
`RuntimeWarning`；TileLang 专项测试必须明确 skip，CPU 通用路径回退 PyTorch。
Warning 和 skip 都不等于 TileLang 测试通过。

纯 KUPL 构建使用 `PYQCU_BUILD_BACKEND=kupl`，只编译 C++17 MPI+OpenMP 后端，
不要求 CUDA toolkit、`nvcc` 或 Cython CUDA 扩展。`torch.dtype.to_real()` 不是所有
PyTorch 版本都提供；PyQCU 使用 `pyqcu.cann.real_dtype()` 做复杂/实数 dtype 映射。

DCU 构建使用 `PYQCU_BUILD_BACKEND=dcu` 和 `bash install.sh dcu`。该模式复用
`cpp/cuda/qcu` 与 `pyqcu.cuda`，但 native compiler 由
`QCU_DCU_CUDA_COMPILER` 或 `ROCM_PATH/cuda/bin/nvcc` 选择，不运行 NVIDIA
`build-nv.sh`。

从仓库根加载统一环境：

```bash
source ./env.sh
```

`env.sh` 会：

- 把仓库根加入 `PYTHONPATH`；
- 把 `cpp/cuda/qcu` 和可用的 QUDA 库目录加入 `LD_LIBRARY_PATH`；
- 仅在仓库内 QUDA 库存在时设置 `QUDA_PATH`；
- 允许显式 `QUDA_INSTALL` 覆盖默认安装树。

不要在文档或脚本中硬编码个人仓库绝对路径；需要外部安装树时通过 `QUDA_INSTALL`、
`PYQCU_KUPL_LIB` 或构建参数显式传入。

KUPL 的 `KUPL_OMP_THREADS`、`KUPL_NUMA_BIND`、HMPI/OpenMPI 绑定和内存策略见
[`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)。生产运行应先由调度器/MPI
完成 NUMA-local cpuset，再导入 Python 和分配大型输入张量。

纯 KUPL 的端到端安装、测试和生产流程见
[`kupl_only_deployment.md`](kupl_only_deployment.md)。
