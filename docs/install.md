# PyQCU 安装与构建

从仓库根初始化环境并构建。CUDA、DCU 和纯 KUPL 使用不同安装入口：

```bash
source ./env.sh

# CUDA QCU，需要匹配的 NVIDIA CUDA toolkit
bash ./install.sh cuda

# AMD DCU/DTK，需要匹配的 ROCm/HIP 和 nvcc-compatible compiler
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu

# 纯 KUPL CPU QCU，不编译 CUDA/Cython 扩展
bash ./install.sh kupl
```

- `install.sh cuda` 调用 CUDA native build 并原地构建 Cython 扩展。
- `install.sh dcu` 使用 `cpp/cuda/qcu/make-dcu.sh` 和
  `CMakeLists-dcu.txt` 构建 DCU 兼容库，再构建同一个 Cython bridge。
- `install.sh kupl` 只构建 `cpp/kupl/qcu`，不查找 `nvcc`，也不创建
  `pyqcu.cuda.qcu`。
- CUDA 架构必须在 `libqcu.so` 中实际存在；正式 benchmark 应记录架构请求值、
  `libqcu.so` 路径与 SHA256。

`source ./env.sh` 会把当前仓库置于 `PYTHONPATH` 首位，并导出可覆盖的
`PYQCU_CUDA_LIB`、`PYQCU_DCU_LIB` 和 `PYQCU_KUPL_LIB`。批处理作业应在计算进程内
重新 source，避免继承旧 `PyQCU-test*` 安装路径。

纯 KUPL 的安装、旧版 PyTorch 兼容、测试、生产运行、Multigrid、升级和故障排查见
[`kupl_only_deployment.md`](kupl_only_deployment.md)。

DCU 的编译器预检、构建、ROCm/HIP 验证、测试和生产限制见
[`dcu_build_backend.md`](dcu_build_backend.md)。

## KUPL CPU 后端

The independent MPI+OpenMP backend does not require CUDA:

```bash
KUPL_ENABLE_NATIVE=OFF KUPL_ENABLE_LTO=OFF bash ./install.sh kupl
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
```

只构建 native 库和测试程序时可运行 `bash cpp/kupl/qcu/make.sh`；
`make.sh --check` 只检查构建变量，不写文件。

Python uses the same flat ABI through `pyqcu.kupl.qcu`:

```bash
python -m pytest -q pyqcu/testing/kupl
```

`source ./env.sh` defaults `PYQCU_KUPL_LIB` to the stable
`cpp/kupl/qcu/libqcu.so` symlink. Set an absolute override only when deliberately
selecting a library outside the current checkout/build selection.

`PYQCU_BUILD_BACKEND=kupl` 会让 `setup.py` 完全跳过 CUDA Cython 扩展。无 CUDA
toolkit 的节点不得使用默认 `cuda` 后端，也不应把 `PYQCU_SKIP_NATIVE_BUILD=1`
当作纯 KUPL 模式。

鲲鹏/HMPI、NUMA、片上内存和 OpenMP 放置请继续阅读
[`kupl_kunpeng_guide.md`](kupl_kunpeng_guide.md)。可部署的通用构建使用
`-DKUPL_ENABLE_NATIVE=OFF`；目标节点性能构建可显式启用
`-DKUPL_ENABLE_NATIVE=ON -DKUPL_ENABLE_LTO=ON`。

## QUDA/PyQUDA 运行时

The simple PyQUDA examples preload the repository-local
`data/quda-sm70-install/lib/libquda.so` when present. `env.sh` also places that
installation before stale global QUDA libraries. Set `QUDA_INSTALL` explicitly
to select another architecture or precision build:

```bash
export QUDA_INSTALL=/absolute/path/to/quda-install
source ./env.sh
python pyqcu/testing/simple/quda_bistabcg.py --help
```

The selected library must contain a CUDA architecture compatible with the
active GPU, and its QUDA ABI must match the installed PyQUDA extension.
