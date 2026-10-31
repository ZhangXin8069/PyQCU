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

从仓库根执行 `bash ./build-nv.sh` 构建活跃 CUDA 后端。DCU/DTK 使用
`QCU_DCU_CUDA_COMPILER=/path/to/dtk/nvcc bash ./install.sh dcu`。独立 KUPL CPU
后端在 `cpp/kupl/qcu` 使用 `bash make.sh`；不要假定所有占位后端都有统一构建脚本。

## 关键约定

- `include/define.h` 参数索引常量必须与 `pyqcu/cuda/define.py` 同步
- 占位目录仅含空 `PASS` 文件
