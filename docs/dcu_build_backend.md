# DCU/DTK 构建与运行手册

本文说明 PyQCU 的 AMD DCU/ROCm(HIP) 构建选路。DCU 复用
`cpp/cuda/qcu` 的 CUDA 兼容源码和 `pyqcu.cuda.qcu` Cython ABI，但使用 DTK
提供的 nvcc-compatible wrapper 编译，不经过 NVIDIA 专用的
`build-nv.sh`/`cmake-nv` 架构探测。

`cpp/dtk/qcu` 与 `pyqcu/dtk` 当前仍是占位目录，不是 DCU 构建实现。真实入口是：

```text
PYQCU_BUILD_BACKEND=dcu
bash install.sh dcu
cpp/cuda/qcu/make-dcu.sh
cpp/cuda/qcu/CMakeLists-dcu.txt
```

## 1. 兼容边界

DCU 构建需要：

- Linux、Python 3.10 或更高版本；
- C++11 或更高版本编译器、CMake；
- MPI 3.1 或更高版本；
- 与 PyTorch 匹配的 ROCm/HIP 运行时；
- DTK 提供的 nvcc-compatible compiler；
- Cython，用于 `pyqcu.cuda.qcu` bridge。

当前 CMake 文件保留 DTK 23.10 的兼容处理，关闭 CMake CUDA response file，
因为该版本的 wrapper 不接受 CMake 的 `--options-file` 参数。

DCU 构建不需要 NVIDIA GPU，但必须保证 DTK、ROCm、HIP、MPI 和 PyTorch 来自兼容的
软件栈。不得把系统 CUDA、系统 ROCm 和 DTK 自带运行时混在一起。

## 2. 编译选择

`setup.py` 支持三个构建后端：

| `PYQCU_BUILD_BACKEND` | 行为 |
|---|---|
| `kupl` | 纯 MPI+OpenMP CPU 后端，不构建 Cython GPU bridge |
| `cuda` | NVIDIA CUDA native build + Cython bridge |
| `dcu` | DTK/ROCm-compatible native build + Cython bridge |

推荐从仓库根安装 DCU：

```bash
source ./env.sh

export ROCM_PATH=/path/to/dtk-or-rocm
QCU_DCU_CUDA_COMPILER="$ROCM_PATH/cuda/bin/nvcc" \
QCU_DCU_BUILD_DIR="$PWD/data/build/cuda-qcu-dcu" \
PYQCU_BUILD_JOBS=32 \
bash ./install.sh dcu
```

如果站点把兼容编译器放在其他位置，使用 `QCU_DCU_CUDA_COMPILER` 显式指定。
脚本不会为了“构建成功”而静默切换成另一个 NVIDIA toolkit。

### 2.1 编译器预检

只检查 DCU 选路和编译器，不启动 CMake：

```bash
QCU_DCU_CUDA_COMPILER=/absolute/path/to/dtk/nvcc \
  bash cpp/cuda/qcu/make-dcu.sh --check
```

成功输出应包含：

```text
PYQCU DCU backend: ready
  compiler: /absolute/path/to/dtk/nvcc
```

不带 `QCU_DCU_CUDA_COMPILER` 时，脚本只尝试
`$ROCM_PATH/cuda/bin/nvcc`。如果站点使用其他路径，必须显式设置
`QCU_DCU_CUDA_COMPILER`；脚本不会静默回退到 `PATH` 中可能属于 NVIDIA 的 `nvcc`。

显式选择 DCU 时，生产报告必须记录最终 compiler 路径和版本，不能只写
`nvcc` 或 `hipcc`。

### 2.2 手工构建

```bash
source ./env.sh

QCU_DCU_CUDA_COMPILER=/absolute/path/to/dtk/nvcc \
QCU_DCU_BUILD_DIR="$PWD/data/build/cuda-qcu-dcu" \
  bash cpp/cuda/qcu/make-dcu.sh

PYQCU_BUILD_BACKEND=dcu \
PYQCU_SKIP_NATIVE_BUILD=1 \
  python setup.py build_ext --inplace
```

`make-dcu.sh` 把最终动态库写到 `${QCU_DCU_BUILD_DIR}/libqcu.so`，并复制到
`cpp/cuda/qcu/libqcu.so` 供 Cython bridge 链接。`source ./env.sh` 会把前者导出为
`PYQCU_DCU_LIB`，把 NVIDIA CUDA 路径导出为 `PYQCU_CUDA_LIB`。
`PYQCU_SKIP_NATIVE_BUILD=1` 只用于已经完成 native build、仅重建 bridge 的场景。

## 3. 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PYQCU_BUILD_BACKEND` | `cuda` | 设置 `dcu` 选择 DCU 构建路径 |
| `QCU_DCU_CUDA_COMPILER` | 自动探测 | DTK nvcc-compatible compiler |
| `ROCM_PATH` | 未设置 | 自动查找 `$ROCM_PATH/cuda/bin/nvcc` |
| `QCU_DCU_BUILD_DIR` | `data/build/cuda-qcu-dcu` | native CMake build tree |
| `PYQCU_DCU_LIB` | `${QCU_DCU_BUILD_DIR}/libqcu.so` | `env.sh` 导出的 DCU native 库路径 |
| `PYQCU_CUDA_LIB` | `cpp/cuda/qcu/libqcu.so` | `env.sh` 导出的 NVIDIA CUDA native 库路径 |
| `QCU_DCU_BUILD_TYPE` | `Release` | CMake build type |
| `PYQCU_BUILD_JOBS` | 在线 CPU 数 | native 并行编译任务数 |
| `PYQCU_SKIP_NATIVE_BUILD` | 未设置 | `1` 时只构建 Cython bridge |

## 4. 安装后验证

确认 PyTorch 是 ROCm/HIP 构建：

```bash
python -B - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("devices", torch.cuda.device_count())
PY
```

确认 DCU 可见：

```bash
rocm-smi
```

确认 Cython bridge 可导入：

```bash
source ./env.sh
python -B -c 'from pyqcu.cuda import qcu; print(qcu.__file__)'
```

如果只看到 NVIDIA CUDA 而没有 `torch.version.hip`，当前 Python 环境不是 ROCm
PyTorch，不能作为 DCU 运行证据。

## 5. 测试

无需 DCU 硬件的构建选路、帮助和编译器预检：

```bash
python -B pyqcu/testing/dcu/test_install_selection.py
```

DCU 硬件上的最小检查：

```bash
python pyqcu/testing/dcu/conftest_1.py
PYQCU_ENABLE_DCU_TESTS=1 python pyqcu/testing/dcu/conftest_1.py
```

`conftest_1.py` 在非 HIP PyTorch 环境会明确 SKIP；只有设置
`PYQCU_ENABLE_DCU_TESTS=1` 才运行 TileLang ROCm 测试。

若 TileLang 未安装，DCU TileLang 入口发出一次 warning 后以退出码 0 SKIP，不产生
traceback。CUDA-compatible 的普通 PyTorch fallback 仍可继续使用。

PyQCU QCU 的单算子/求解器测试使用与 CUDA 相同的 Python ABI。执行前先确认
`torch.version.hip` 非空、设备数正确，并只在这套 DCU 软件栈内选择测试入口。不要把
NVIDIA GPU 上通过的测试结果写成 DCU 数值验证。

## 6. 生产运行

DCU 使用 ROCm PyTorch 提供的 CUDA-compatible API，因此 Python 运行时后端名仍为
`cuda`，设备仍写作 `cuda:0`。`PYQCU_BUILD_BACKEND=dcu` 描述 native 编译配置，
不表示存在独立的 `pyqcu.dcu` Python 包。

生产拓扑不变：

```text
一个 Python 进程 = 一个 MPI rank = 一张 DCU
```

跨 DCU 求解必须使用 MPI 四维格点分域。设备隔离变量必须在 PyTorch、bridge 和 native
库初始化之前设置。调度器通常仍使用 `CUDA_VISIBLE_DEVICES` 暴露 DCU 编号。

生产入口沿用 CUDA QCU 命令，例如：

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend cuda \
  --device cuda:0 \
  --lat 24 24 24 64 \
  --out-dir data/production_dcu_L24
```

报告必须额外记录：

- `PYQCU_BUILD_BACKEND=dcu`；
- `QCU_DCU_CUDA_COMPILER` 实际路径和版本；
- `torch.version.hip`、ROCm/DTK 版本；
- `rocm-smi` 设备 UUID、可见设备列表和每 rank 设备映射；
- MPI rank/grid、每进程一张 DCU 的拓扑证据；
- native `libqcu.so` SHA256、独立真残差和实际求解迭代数。

## 7. 与 KUPL 的区别

| 项目 | DCU | KUPL |
|---|---|---|
| native 源码 | `cpp/cuda/qcu` | `cpp/kupl/qcu` |
| Python bridge | `pyqcu.cuda.qcu` | `pyqcu.kupl.qcu` ctypes |
| 编译后端 | DTK/ROCm-compatible | MPI+OpenMP C++17 |
| 运行时后端名 | `cuda` | `kupl` |
| 设备 | DCU | CPU/NUMA |
| 生产并行 | MPI rank/DCU | MPI rank/OpenMP |

## 8. 故障排查

### 8.1 找不到 DCU compiler

```bash
QCU_DCU_CUDA_COMPILER=/absolute/path/to/dtk/nvcc \
  bash cpp/cuda/qcu/make-dcu.sh --check
```

不要通过修改 `PATH` 让脚本误用 NVIDIA `nvcc`。显式指定并记录路径。

### 8.2 CMake 报 CUDA 架构或 nvcc 文件错误

DCU 选路应使用 `CMakeLists-dcu.txt`，由 `make-dcu.sh` 自动选择。不要在 DCU 上运行
NVIDIA 的 `build-nv.sh` 或 `CMakeLists-nv.txt`。

### 8.3 PyTorch 没有 HIP

```bash
python -c 'import torch; print(torch.version.hip)'
```

输出为 `None` 表示当前是 CUDA/CPU PyTorch，不能运行 DCU 数值测试。重新加载站点
提供的 ROCm PyTorch 环境。

### 8.4 设备序号和调度器不一致

在导入 Python 前检查 `CUDA_VISIBLE_DEVICES` 与 `rocm-smi` 的 UUID 映射；不要在
rank 内部临时切换设备来掩盖调度错误。

## 9. 当前限制

- `cpp/dtk/qcu` 和 `pyqcu/dtk` 是占位，不包含独立 DCU 实现。
- 当前没有公共 `backend="dcu"` selector；DCU 通过 `backend="cuda"` 使用同一 ABI。
- `CMakeLists-dcu.txt` 的数值 smoke 需要实际 DTK/DCU 环境，不能由本机静态测试替代。
- DCU 性能、NUMA、UCX、MPI 集合算法和 cache identity 必须在目标机器重新取证。
