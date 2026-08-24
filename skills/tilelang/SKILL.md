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

## Known upstream defects & hardware limits (2026-08-24, bug36)

- v0.1.7.post3 ships four same-named emitter classes (one per architecture fallback path)
  missing `_legalize_to_buffer_region` — the installed copy monkey-patches all four.
  Locate the real throw site via full-stack traceback; patching one copy is not enough.
  Installed version vs upstream tag diff is IDENTICAL → native upstream defect.
- SM70 (V100) TensorCore has no fp32 input instruction (`mma_sm70.h` static_assert):
  slow fp32-gemm is a hardware limit, not fixable by hot-patch; long-term fix = upgrade TileLang.
- Measured anchor after hot-fix: 4096³ fp16 matmul 38.7 TFLOPS ≈ 94% of cuBLAS.
