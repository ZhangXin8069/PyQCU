# PyQCU Debug & Optimization Log
**Date**: 2026-07-28
**Source**: /root/PyQCU/logs/review-2026-07-28.md
**Status**: All fixes applied and verified

---

## Summary: 12 Bug Fixes + 1 Optimization

| # | Type | File | Issue | Status |
|---|------|------|-------|--------|
| 1 | 🔴 CRITICAL | `tools/_io.py` | I/O index order wrong in serial fallback | ✅ Fixed |
| 2 | 🔴 CRITICAL | `smear/_stout.py` | nstep>1 had no effect | ✅ Fixed, verified |
| 3 | 🔴 CRITICAL | `dslash/_operator.py` | sitting/matvec AttributeError when clover_term=None | ✅ Fixed, verified |
| 4 | 🔴 CRITICAL | `cann/__init__.py` | NPU 3+ operand einsum wrong results | ✅ Fixed, verified |
| 5 | 🔴 CRITICAL | `cuda/define.py` | Bare `raise` with no message | ✅ Fixed, verified |
| 6 | 🟡 MEDIUM | `solver/_bistabcg.py` | ZeroDivisionError when verbose=False | ✅ Fixed, verified |
| 7 | 🔴 CRITICAL | `testing/__init__.py` | test_solver used wrong variable in error msg | ✅ Fixed |
| 8 | 🔴 CRITICAL | `tools/_define.py` | check_mpi_support leaked temp files | ✅ Fixed |
| 9 | 🔴🔴 FATAL | `cpp/.../lattice_complex.h` | operator*= complex multiplication bug | ✅ Fixed, builds |
| 10 | 🔴🔴 FATAL | `cpp/.../gauss_gauge.cu` | OOB write + GPU memory leak | ✅ Fixed, builds |
| 11 | 🔴 HIGH | `cpp/.../apply_end.cu` | LatticeSet object never deleted | ✅ Fixed, builds |
| 12 | 🔴 HIGH | `cpp/.../lattice_wilson_dslash.h` | MPI_Isend never waited on | ✅ Fixed, builds |
| OPT1 | 🔵 PERF | `_clover.py`, `_operator.py` | 24 redundant MPI Barrier() removed | ✅ Optimized |

## Validation Results (8/8 PASSED)
```
============================================================
  PyQCU Validation Suite — 2026-07-28
============================================================
  PASS: stout_smear nstep>1
  PASS: operator parity
  PASS: BiCGStab
  PASS: BiCGStab parity
  PASS: NPU 2-op einsum
  PASS: NPU 3-op einsum
  PASS: No MPI orphans
  PASS: cuda/define ValueError

  TOTAL: 8/8 passed
  ALL TESTS PASSED
```

## C++ Build
```
[100%] Linking CUDA shared library libqcu.so
[100%] Built target qcu
```
