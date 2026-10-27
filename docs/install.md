# PyQCU 安装与构建

从仓库根初始化环境并构建：

```bash
source ./env.sh
bash ./build.sh
bash ./install.sh
```

- `build.sh` 构建 `cpp/cuda/qcu` 原生 CUDA 后端，产出 `libqcu.so`。
- `install.sh` 原地构建 Python/Cython 扩展。
- CUDA 架构必须在 `libqcu.so` 中实际存在；正式 benchmark 应记录架构请求值、
  `libqcu.so` 路径与 SHA256。

## KUPL CPU 后端

The independent MPI+OpenMP backend does not require CUDA:

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/kupl/qcu/build -j
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
```

Python uses the same flat ABI through `pyqcu.kupl.qcu`:

```bash
python -m pytest -q pyqcu/testing/kupl
```

Set `PYQCU_KUPL_LIB=/absolute/path/libqcu.so` when the library is installed
outside the default `cpp/kupl/qcu/build/` directory.

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
