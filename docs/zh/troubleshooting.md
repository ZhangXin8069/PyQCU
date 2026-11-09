# 排错

## 第一步：跑自检

```bash
pyqcu
pyqcu --selftest
pyqcu --json
```

```python
import json
import subprocess
import sys

report = json.loads(subprocess.run(
    [sys.executable, "-m", "pyqcu", "--json"],
    check=True, capture_output=True, text=True).stdout)
assert report["package"] == "pyqcu"
assert "backends" in report
print("diagnostics OK", report["version"])
```

`pyqcu --selftest` 会跑 Torch、KUPL、CUDA 三层。`SKIP` 表示后端未构建或没有
可见设备；`FAIL` 表示该层实际运行了且独立残差检查失败。`--json` 输出机器可读
报告，`--lattice X Y Z T` 可换另一个偶数格点；小于 4 的维度会被拒绝，CUDA
原生路径未在 `2^4` 上验证，发布验证使用 `8x8x16x16`。

## 安装与构建

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'Cython'` | 用了 `--no-build-isolation`，环境里没有 Cython/numpy | 去掉 `--no-build-isolation`，或先装 `Cython` 与 `numpy` |
| 找不到 `cmake` | KUPL 需要 CMake | 安装 CMake，或设 `PYQCU_SKIP_NATIVE_BUILD=1` 走纯 Torch |
| 找不到 MPI 头/库 | KUPL 需要 MPI 实现 | 安装 MPI，或设 `PYQCU_SKIP_NATIVE_BUILD=1` |
| CUDA 构建被跳过 | 缺 `nvcc` 或主机编译器无法编译 | 安装 CUDA toolkit，或设 `PYQCU_BUILD_BACKEND=kupl` |
| `pip install` 看起来卡住 | 原生编译在 pip 下没有输出 | 等待；有终端时进度写到 `/dev/tty` |
| `No space left on device` | `/tmp` 是小 tmpfs | 安装前把 `TMPDIR` 设到大磁盘 |

把缺失的原生后端变成硬错误：

```bash
PYQCU_BUILD_BACKEND=kupl PYQCU_REQUIRE_NATIVE=1 pip install pyqcu
```

## 原生后端

| 现象 | 含义 | 处理 |
|---|---|---|
| `pyqcu` 报 KUPL `FileNotFoundError` | 包内没有 `libqcu.so` | 用工具链重装，或设 `PYQCU_KUPL_LIB=/abs/path/libqcu.so` |
| CUDA bridge is not built | 安装时没有 `nvcc`/CUDA 库 | 在有 CUDA 的机器上以 `PYQCU_BUILD_BACKEND=cuda` 重装 |
| 构建成功后 CUDA 导入失败 | 库是为其他架构构建的 | 在目标 GPU 上重建，或嵌入兼容的 PTX 架构 |
| `PYQCU_KUPL_LIB` 指向旧库 | 旧环境变量优先 | unset 它，让包内库解析 |

KUPL 查找顺序是 `PYQCU_KUPL_LIB`、`pyqcu/kupl/libqcu.so`、
`pyqcu/kupl/build/libqcu.so`，最后是源码树路径。CUDA 桥使用 `$ORIGIN`
rpath，期望 `pyqcu/cuda/libqcu.so` 同目录。

## 求解器

| 现象 | 可能原因 | 检查 |
|---|---|---|
| `solve_result` 字段为 `None` | Torch Legacy 不暴露这些字段 | 用 `mg.implementation.last_solve_report`、`convergence_history` 或独立残差 |
| CUDA `iterations=None` | CUDA Wilson 没有 iteration getter | 看 `converged` 与残差；Clover 有 getter |
| 残差高于 c64 验收线 | 口径错误或单精度地板 | 检查 `residual_scope`；c64 地板为 `max(atol, 2048*eps32) ≈ 2.4414e-4` |
| `no Legacy levels` | `min_size` 大于局部尺寸 | 降低 `min_size` 或使用更大局部格点 |
| `coarse_assets ... does not match` | cache identity/形状/精度不匹配 | 用相同格点/块/精度重建 cache |
| `process_grid ... does not match MPI size` | rank 网格错误 | 让 `prod(process_grid) == MPI.COMM_WORLD.Get_size()` |

多重网格细节见 {doc}`solvers`；底层原生 ABI 见源码仓库的
`docs/api_contract.md`。

## MPI 与布局

* `pyqcu --selftest` 应以单进程运行。在 `mpirun` 下，原生层会以单 rank 要求
  报告 `SKIP`，且只有 rank 0 输出报告；`pyqcu --json` 也只在 rank 0 输出。
* 奇偶布局要求每个局部维度为偶数。
* 奇偶分裂场压缩的是 **时间** 轴到 `T/2`；不要把完整场直接 reshape 成该布局。
* CUDA 生产是一个 Python 进程对应一个 MPI rank 与一张设备。不要在一个进程内
  创建多个 GPU worker。
* KUPL 跨 rank 用 MPI、rank 内用 OpenMP。固定 `OMP_NUM_THREADS` 与
  `KUPL_OMP_THREADS`，避免超订。
* `mpirun` 以 `Bus error`、退出码 135 结束时先看 `df -h /tmp`。`/tmp` 满会让
  OpenMPI 共享内存池失败；把 `OMPI_MCA_orte_tmpdir_base` 与 `TMPDIR` 指向可写
  路径、清理遗留 MPI 临时文件，或对纯 CPU 诊断设置
  `OMPI_MCA_btl=^smcuda`。

## 警告与已知限制

* `TileLang is unavailable` 是可选依赖提示，不影响 Torch/KUPL/CUDA 路径。
* `pyqcu.tools.norm()` 已经返回 Python `float`，不要再调用 `.item()`。
* Torch Legacy 的 `Multigrid.solve_result()` 对未暴露字段返回
  `UNKNOWN`/`None`；这是设计行为，不要根据配置推断。
* 在其他机器构建的原生库可能可以导入，但第一次调用就失败；基准测试应记录库
  路径与 SHA256。

## 报告问题

请附上：

```bash
pyqcu --json
pyqcu --selftest --json
python -m pip check
```

以及失败命令、格点、精度、后端，并说明能否在干净 venv 中复现。
