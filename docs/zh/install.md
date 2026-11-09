# 安装

## 依赖

* Python 3.10 或更新版本（实测 3.12）。
* Linux x86_64 或 aarch64。
* `numpy`、`torch`、`mpi4py`、`h5py` 会作为运行时依赖自动安装。
* 构建 KUPL CPU 库需要 C++17 编译器、CMake 与 MPI 实现。CUDA 另外需要
  `nvcc`；DCU 需要 DTK 的 `nvcc` 兼容编译器。

```bash
pip install pyqcu
```

发行包只提供源码包。首次安装时若有工具链会现场编译 KUPL 原生库；后续安装
复用 pip 的 wheel 缓存。

## 可选 extras

| extra | 增加 | 用途 |
|---|---|---|
| `pyqcu[plot]` | `matplotlib` | 基准与收敛曲线绘图 |
| `pyqcu[monitor]` | `psutil` | 内存/进程监控工具 |
| `pyqcu[all]` | 两者 | 交互式工作的便利组合 |

`torch` 是硬依赖，因为 Torch 后端与公共格点工具直接导入它。安装 PyQCU 前，
请先按机器选择 CPU 或 CUDA 版 Torch。

## 安装器做了什么

1. 从 `pyproject.toml` 读取 PEP 621 元数据。
2. 用 CMake、MPI 与 OpenMP 尝试构建 KUPL CPU 库（`cpp/kupl/qcu`）。
3. 当 `nvcc` 可用且后端为 `cuda` 时尝试 CUDA 库；后端为 `dcu` 时尝试
   DCU 库。
4. 只有 CUDA/DCU 库构建成功时才编译 Cython 桥 `pyqcu.cuda.qcu`。
5. 把生成的库复制到 `pyqcu/kupl/libqcu.so` 与 `pyqcu/cuda/libqcu.so`，
   使安装后的包不依赖源码树或 `LD_LIBRARY_PATH`。

缺少工具链不会导致安装失败：安装仍然成功，纯 Torch 路径可用。

## 原生构建变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `PYQCU_BUILD_BACKEND` | `cuda` | `kupl`、`cuda` 或 `dcu`；控制 GPU 尝试 |
| `PYQCU_SKIP_NATIVE_BUILD` | 未设置 | `1` 跳过所有原生构建 |
| `PYQCU_REQUIRE_NATIVE` | 未设置 | `1` 把请求的后端缺失变成硬错误 |
| `PYQCU_KUPL_BUILD_DIR` | `cpp/kupl/qcu/build` | KUPL CMake 构建目录 |
| `PYQCU_KUPL_LIB` | 包内库 | 显式指定 KUPL 运行库 |
| `KUPL_BUILD_TESTS` | pip 构建时 `OFF` | 构建 KUPL CTest 可执行文件 |
| `KUPL_ENABLE_NATIVE` | `OFF` | 针对本机 CPU 架构编译 |
| `KUPL_ENABLE_LTO` | `OFF` | 链接时优化 |

示例：

```bash
# CPU-only: build KUPL and skip the GPU attempt
PYQCU_BUILD_BACKEND=kupl pip install pyqcu

# Require the KUPL build to succeed
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu

# Build both KUPL and CUDA on a machine with nvcc
PYQCU_BUILD_BACKEND=cuda pip install pyqcu
```

## 验证安装

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

`pyqcu` 是安装后的 console script；`python -m pyqcu` 等价。
`pyqcu --selftest` 会跑 Torch/KUPL/CUDA 三层，缺少原生后端时报告 `SKIP`。
默认格点是 `4^4`，小于 4 的维度会被拒绝。

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--json"],
    check=True, capture_output=True, text=True).stdout)
assert report["package"] == "pyqcu"
assert report["version"] == "0.1.7"
assert "path" in report["backends"]["kupl"]
print("pyqcu", report["version"], "at", report["path"])
```

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--selftest", "--json",
     "--lattice", "4", "4", "4", "4"],
    check=True, capture_output=True, text=True).stdout)
assert report["status"] == "PASS"
assert report["layers"]["torch"]["status"] == "PASS"
print("self-test:", {
    name: layer["status"] for name, layer in report["layers"].items()})
```

```python
import importlib.util
import pyqcu

assert pyqcu.__version__ == "0.1.7"
missing = [
    name for name in
    ("api", "lattice", "tools", "torch", "solver", "cuda", "kupl")
    if importlib.util.find_spec(f"pyqcu.{name}") is None
]
assert not missing, missing
print("all public subpackages importable")
```

如果 KUPL 报告 `FileNotFoundError`，说明原生库没有构建。安装工具链后重装，
或把 `PYQCU_KUPL_LIB` 指向已有的 `libqcu.so`。

## 从源码仓库安装

```bash
git clone git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git
cd PyQCU
source ./env.sh

# Pure KUPL CPU backend
bash ./install.sh kupl

# CUDA backend plus the Cython bridge
bash ./install.sh cuda

# DCU/DTK backend
QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu
```

源码仓库在 `pyqcu/testing/` 下提供额外的测试与基准驱动；这些内容有意不进入
PyPI 包。

## 源码仓库内测试

```bash
cd pyqcu/testing
pytest .
```

MPI 专用测试需要显式启动，例如：

```bash
mpirun -np 4 python pyqcu/testing/python/conftest.py
```

## 后端与数组类型

| 输入 / 构建 | 执行路径 |
|---|---|
| Torch 张量，CPU | `pyqcu.torch`，或显式调用 KUPL |
| Torch 张量，CUDA | `pyqcu.torch` CUDA 内核或 CUDA 桥 |
| NumPy 主机数组 | KUPL `ctypes` ABI |
| 无原生库 | 仅 Torch 后端 |

## 排错

* **缺 `cmake` 或 `mpi4py`**：安装 CMake 与 MPI，或设置
  `PYQCU_SKIP_NATIVE_BUILD=1` 走纯 Torch 安装。
* **CUDA 导入错误**：查看 `python -m pyqcu --json`；只有 `nvcc` 与 CUDA
  库都可用时才会构建 CUDA 桥。
* **导入到错误版本**：在源码树外运行，并断言 `pyqcu.__file__` 含
  `site-packages`。
* **`/tmp` 满**：pip 在临时目录构建；设置更大的 `TMPDIR` 或清理旧 venv。

按症状排查见 {doc}`troubleshooting`。
