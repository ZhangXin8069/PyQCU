# AGENTS.md — cpp/cuda/qcu/logs

C++ CUDA 后端（`cpp/cuda/qcu`）的本地运行输出目录。存放构建、测试与基准的生成日志。

## 内容

- `bash ./make.sh` 构建输出（编译器/链接信息）
- C++ 后端测试输出（如 `examples/qcu/conftest.clover.multigrid.py`）
- 性能/收敛报告

## 注意

- 本地运行目录，`cpp/cuda/qcu/logs/` 不入 git
- 开发报告与测试输出的规范位置是仓库根 `logs/`（见 `logs/CLAUDE.md`）；本目录只放后端本地产物
