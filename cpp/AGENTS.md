# AGENTS.md — cpp

PyQCU 的 C++ 后端。每个子目录对应一种 GPU 架构。

## 后端

| 目录 | 架构 | 状态 |
|---|---|---|
| `cuda/qcu/` | NVIDIA CUDA | **活跃** — 主生产后端 |
| `cann/qcu/` | 华为昇腾 CANN | 占位 stub |
| `dtk/qcu/` | AMD DCU / ROCm (HIP) | 占位 stub |
| `maca/qcu/` | Maca | 占位 stub |

## 活跃后端 cpp/cuda/qcu

手写 CUDA 内核（Wilson/Clover dslash、BiStabCG/CG 求解器、multigrid V-cycle、规范场生成），MPI halo 交换分布于 4D 进程网格。经 Cython 桥 `pyqcu/cuda/qcu/` 从 Python 访问。

## 构建

每个后端自带 `env.sh`（编译器/链接路径）与 `make.sh` 或 CMake 构建脚本。活跃 CUDA 后端用 `CMakeLists-nv.txt`（symlink 为 `CMakeLists.txt`）cmake + make 链。

## 关键约定

- `include/define.h` 参数索引常量必须与 `pyqcu/cuda/define.py` 同步
- 占位目录仅含空 `PASS` 文件
