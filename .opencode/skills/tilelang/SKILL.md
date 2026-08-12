---
name: tilelang
description: examples/tilelang 目录的完整生成 skill：TileLang JIT 内核测试（CUDA）。
---
# CLAUDE.md — examples/tilelang

TileLang JIT-compiled kernel tests for CUDA. Exercises the TileLang integration in `pyqcu/tools/_einsum.py` and `pyqcu/tools/_matul.py`.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | TileLang test entry |

## Usage

```bash
python examples/tilelang/conftest.py
```

Requires TileLang to be installed (optional dependency, silently degrades if unavailable).
