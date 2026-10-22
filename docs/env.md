# PyQCU Python 环境

PyQCU 要求 Python ≥ 3.10。核心依赖为 PyTorch、Cython、mpi4py、h5py 和 NumPy；
测试另需 pytest。CUDA C++ 后端需要与目标 GPU 匹配的 CUDA toolkit。

TileLang 是可选依赖，仅用于 JIT 内核实验；缺少时相关测试应明确 skip，不能把静默降级写成通过。

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
