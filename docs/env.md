# PyQCU Python 环境

PyQCU 要求 Python ≥ 3.10。核心依赖为 PyTorch、mpi4py、h5py 和 NumPy；测试另需
pytest。Cython 只用于 CUDA Cython bridge，纯 KUPL 源码安装不导入 Cython；
CUDA C++ 后端还需要与目标 GPU 匹配的 CUDA toolkit。

TileLang 是可选依赖，仅用于 JIT 内核实验；缺少时相关测试应明确 skip，不能把静默降级写成通过。

纯 KUPL 构建使用 `PYQCU_BUILD_BACKEND=kupl`，只编译 C++17 MPI+OpenMP 后端，
不要求 CUDA toolkit、`nvcc` 或 Cython CUDA 扩展。`torch.dtype.to_real()` 不是所有
PyTorch 版本都提供；PyQCU 使用 `pyqcu.cann.real_dtype()` 做复杂/实数 dtype 映射。

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
