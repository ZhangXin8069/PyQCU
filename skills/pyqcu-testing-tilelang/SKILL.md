---
name: pyqcu-testing-tilelang
description: 当任务涉及 pyqcu/testing/tilelang、TileLang JIT 集成或 no-op 测试入口边界时使用。
---
# pyqcu/testing/tilelang

TileLang JIT integration for CUDA lives behind `pyqcu/torch/tools/_einsum.py` and
`pyqcu/torch/tools/_matul.py`. The historical `pyqcu/testing/tilelang/` no-op
directory has been removed; validate through `pyqcu.testing.test_matmul()` or a new
real pytest node.

## Files

当前没有独立测试目录或有效测试入口。历史 no-op `conftest.py` 与空目录已删除。

## Usage

不要再引用 `python pyqcu/testing/tilelang/conftest.py`。历史文件不导入
`pyqcu.testing`，也没有调用 `test_matmul()`，进程会无输出退出，现已删除。需要恢复该入口时，
先启用真实测试调用，再要求 pytest 收集到至少一个测试；
只有 `exit=0` 或只有 `TESTS=...` 而没有断言结果都不构成验证。来源：
`docs/multigrid_deprecation_audit_20261022.md:41-61`。

TileLang 是可选依赖。统一探测入口是 `pyqcu._tilelang.import_tilelang()`：不可用时
每个进程只发一次 `RuntimeWarning`，结果缓存；专项测试必须以明确 skip 原因呈现，
不能把空收集、warning 或 CPU fallback 报告成 TileLang 测试通过。

无依赖契约回归：

```bash
python -m pytest -q pyqcu/testing/python/test_tilelang_optional.py
```

`Eexyzt_exyzt2Exyzt` 的 CPU 和非 TileLang 路径必须回退 `torch.einsum`。直接调用
`matmul_gpu()`/`matmul_cpu()` 这类只返回 TileLang program 的 API 时，缺依赖应抛出
`TileLangUnavailable`，不得返回伪对象。

## Known upstream defects & hardware limits (2026-08-24, bug36)

- v0.1.7.post3 ships four same-named emitter classes (one per architecture fallback path)
  missing `_legalize_to_buffer_region` — the installed copy monkey-patches all four.
  Locate the real throw site via full-stack traceback; patching one copy is not enough.
  Installed version vs upstream tag diff is IDENTICAL → native upstream defect.
- SM70 (V100) TensorCore has no fp32 input instruction (`mma_sm70.h` static_assert):
  slow fp32-gemm is a hardware limit, not fixable by hot-patch; long-term fix = upgrade TileLang.
- Measured anchor after hot-fix: 4096³ fp16 matmul 38.7 TFLOPS ≈ 94% of cuBLAS.
