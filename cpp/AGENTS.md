# AGENTS.md — cpp

PyQCU 的 C++ 后端实现。每个子目录面向一种 GPU 架构。

## 后端

| 目录 | 架构 | 状态 |
|---|---|---|
| `cuda/qcu/` | NVIDIA CUDA | **活跃** — 主生产后端 |
| `cann/qcu/` | 华为昇腾 CANN | 占位 stub |
| `dtk/qcu/` | AMD DCU / ROCm (HIP) | 占位 stub |
| `maca/qcu/` | Maca | 占位 stub |

## 构建

每后端自带 `env.sh`（编译器/链接器路径）与 `make.sh` 或 CMake 构建脚本。活跃的 CUDA 后端用 `CMakeLists-nv.txt`（软链为 `CMakeLists.txt`）+ cmake + make 串联；产物为 `libqcu.so`，由 Cython 桥 `pyqcu/cuda/qcu/` 加载。

## Skills

`skills/` 下子目录的完整生成 skill（原 CLAUDE.md 内容）：`cuda`、`cann`、`dtk`、`maca`。
