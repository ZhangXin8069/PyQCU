---
name: logs
description: cpp/cuda/qcu/logs 目录的完整生成 skill：CUDA 后端本地运行日志目录（gitignored），正式报告存放于仓库根 logs/。
---
# CLAUDE.md — cpp/cuda/qcu/logs

Runtime output directory for the C++ CUDA backend (`cpp/cuda/qcu`). Holds generated log files produced by building, testing, and benchmarking the C++ backend.

## Contents

Currently empty. Logs written here may include:

- Build output from `bash ./make.sh` (compiler messages, linker output)
- Test output from running the C++ backend tests (e.g., `examples/qcu/conftest.clover.multigrid.py`)
- Performance / convergence reports

## Notes

- This is a local runtime directory. `cpp/cuda/qcu/logs/` is not tracked in git.
- The canonical location for development reports and test outputs is the repo-root `logs/` directory (see `logs/CLAUDE.md` for its file patterns). Only backend-local artifacts belong here.
