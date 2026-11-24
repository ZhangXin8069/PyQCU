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

该脚本可由 Bash 或 zsh `source`；仓库路径和库目录会被去重，重复执行不会不断
拉长 `PYTHONPATH` 或 `LD_LIBRARY_PATH`。

`env.sh` 会：

- 导出 `PYQCU_ROOT`，并把当前仓库根放在 `PYTHONPATH` 最前，避免旧 editable
  install 或 `PyQCU-test*` 目录抢先导入；
- 把 `cpp/cuda/qcu` 和可用的 QUDA 库目录加入 `LD_LIBRARY_PATH`；
- 仅在仓库内 QUDA 库存在时设置 `QUDA_PATH`；
- 允许显式 `QUDA_INSTALL` 覆盖默认安装树；
- 导出当前仓库的 CUDA、DCU、KUPL 默认库路径：

```text
PYQCU_CUDA_LIB=<repo>/cpp/cuda/qcu/libqcu.so
PYQCU_DCU_LIB=<QCU_DCU_BUILD_DIR>/libqcu.so
PYQCU_KUPL_LIB=<repo>/cpp/kupl/qcu/libqcu.so
```

`cpp/kupl/qcu/libqcu.so` 是构建后指向 `${PYQCU_KUPL_BUILD_DIR}/libqcu.so` 的稳定
软链接，与 `cpp/cuda/qcu/libqcu.so` 的路径形态对齐。

- 固定安全的 KUPL 基础值：`OMP_DYNAMIC=FALSE`、`KUPL_NUMA_BIND=0`、
  `KUPL_FUSED_BICG_DOTS=1`、`KUPL_DISABLE_CLOVER_CACHE=0`、
  `KUPL_ALLOW_OVERSUBSCRIBE=0`、`KUPL_HUGEPAGE=0`。

以上变量只在调用者未显式设置时赋默认值。`KUPL_OMP_THREADS` 与
`OMP_NUM_THREADS` 若只设置其中一个，`env.sh` 会同步另一个；两者都已设置时保持原值。
`KUPL_NUMA_BIND=1` 时还会清除冲突的 `OMP_PROC_BIND`、`OMP_PLACES`、
`GOMP_CPU_AFFINITY` 和 `KMP_AFFINITY`。

`KUPL_NUMA_BIND=0`/`KUPL_HUGEPAGE=0` 只是不绑定的起点，不是多 rank KUPL 的
生产配置：鲲鹏节点上 `pyqcu.test` 的多 rank KUPL 运行要求 `KUPL_NUMA_BIND=1`、
`KUPL_MEMORY_NODE=local+16`、`KUPL_HUGEPAGE=1`，并配 `mpirun --bind-to none`
（放置由库负责）。同一 L48 c64 16 rank 求解未绑定 196.00 s、绑定 2.06 s
（作业 2236451 / 2236485，95x）；缺这三个变量时 `pyqcu.test` 直接报 ERROR，
而不是记录一个不可比较的数字。唯一例外是刻意的绑定/不绑定 A/B：
`PYQCU_ALLOW_UNBOUND_KUPL=1`。

数量级差来自访存布局而非算法：Clover/gauge 是顺序 streaming 读，每 rank 的 fine
apply 要流过 GB 级 gauge slab 与四个 Clover 项；未绑定时页按 4K 散落在节点内多个
NUMA 域、大页未生效、OpenMP 线程也不固定在 NUMA-local CPU 上，两个效应相乘即
30-60x。32 rank 同一形状由 102.47 s 降到 3.04 s（作业 2236599），迭代数与真残差
逐位相同（`logs/v20261123.txt` §46/§49/§54）。

不要在文档或脚本中硬编码个人仓库绝对路径；需要外部安装树时通过 `QUDA_INSTALL`、
`PYQCU_CUDA_LIB`、`PYQCU_DCU_LIB`、`PYQCU_KUPL_LIB` 或构建参数显式传入。

KUPL 的 `KUPL_OMP_THREADS`、`KUPL_NUMA_BIND`、HMPI/OpenMPI 绑定和内存策略见
[`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)。生产运行应先由调度器/MPI
完成 NUMA-local cpuset，再导入 Python 和分配大型输入张量。

纯 KUPL 的端到端安装、测试和生产流程见
[`kupl_only_deployment.md`](kupl_only_deployment.md)。
