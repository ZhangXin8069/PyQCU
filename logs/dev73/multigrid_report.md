# PyQCU C++ CUDA Clover Multigrid Solver — Debug & Optimization Report

**Date:** 2026-07-28
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9)
**Repository:** PyQCU

---

## Summary

| Metric | Value |
|--------|-------|
| Correctness (‖x_mg−x_ref‖/‖x_ref‖) | **6.23×10⁻⁷** ✅ |
| NaN in log | **0** ✅ |
| Speedup vs BiStabCG | **0.87×−1.28×** (varies with run) |
| Convergence rate | ~104 iterations to 5.7×10⁻⁷ |
| Log format | Matches `conftest.clover.multigrid-v20260506.log` ✅ |

---

## Bugs Fixed

### 1. Segfault from Uninitialized `MPI_Wait` (Critical)
**File:** `cpp/cuda/qcu/include/lattice_wilson_dslash.h`
**Symptom:** Immediate segfault on any BiStabCG call
**Root cause:** `run_mpi()` uses blocking `MPI_Sendrecv`, but `MPI_Wait` calls were added on uninitialized `send_request` handles after the blocking path
**Fix:** Removed `MPI_Wait` calls from the `run_mpi` path (line 547-562). Only `run_mpi_non_block` (which uses `MPI_Isend`) needs `MPI_Wait`.

### 2. `device_vals` Race Condition → NaN (Critical)
**File:** `cpp/cuda/qcu/include/lattice_clover_multigrid.h`
**Symptom:** Residuals become NaN after ~4-6 iterations (8399 NaN entries in log)
**Root cause:** Host→device scalar `cudaMemcpyAsync` uploads inside `bistabcg_iter` loop raced with GPU kernels reading `device_vals` on other streams. The host-side writes to `vals[_rho_]`, `vals[_rho_prev_]`, `vals[_alpha_]`, `vals[_omega_]` could be interleaved with `give_1beta`/`give_1alpha`/`give_1omega` kernel execution on streams `_a_`/`_d_`.
**Fix:** Removed ALL host→device scalar memcpy from inside the iteration loop. BiStabCG scalars now live ONLY in `device_vals` and are modified exclusively by GPU kernels, exactly matching `LatticeCloverBistabCg::_run()`.

### 3. Missing Bottom-Of-Iteration Stream Sync (Major)
**File:** `cpp/cuda/qcu/include/lattice_clover_multigrid.h`
**Symptom:** Residual oscillating with 3-12x jumps during convergence
**Root cause:** `bistabcg_iter` only synced stream `_a_` at the bottom for the residual norm computation. Streams `_b_` (x update), `_c_`/`_d_` (dot products), and `strm` (dslash) could still have pending operations when the next iteration started, causing stale reads of `device_vals` scalar slots.
**Fix:** Added `cudaStreamSynchronize` for ALL 5 streams (strm, _a_, _b_, _c_, _d_) at the bottom of `bistabcg_iter`, matching `LatticeCloverBistabCg::_run()` exactly.

### 4. cublasDot Target Slot Corruption (Minor)
**File:** `cpp/cuda/qcu/include/lattice_clover_multigrid.h`
**Symptom:** Occasional residual jumps due to stale scalar reads
**Root cause:** `dot_mpi_to_device` wrote cublasDot results directly to the target slot (`_rho_`, `_tmp0_`, etc.), then copied to host, did MPI_Allreduce, and copied back. The target slot was temporarily overwritten by the raw cublasDot result before the MPI-reduced value was written back. If another stream read the slot during this window, it got the wrong value.
**Fix:** Changed to always write cublasDot results to `_send_tmp_` (scratch slot, index 7) first, then copy to the target slot only AFTER MPI_Allreduce, matching `LatticeCloverBistabCg::_dot_mpi` exactly.

### 5. `MPI_FLOAT` Hardcoded → Double Broken (Major)
**File:** `cpp/cuda/qcu/include/lattice_clover_multigrid.h`
**Symptom:** `double`-precision mode would produce incorrect results
**Root cause:** `MPI_Allreduce(..., MPI_FLOAT, ...)` was hardcoded instead of using the template-appropriate type
**Fix:** Added `mpi_real_type<T>()` template function returning `MPI_FLOAT` for `float` and `MPI_DOUBLE` for `double`.

### 6. Pre-existing: `wilson_dslash.cu` Undefined `idx`
**File:** `cpp/cuda/qcu/src/wilson_dslash.cu`
**Symptom:** Compilation failure (8 errors)
**Root cause:** Pre-existing bugfix used `idx` variable name but the kernel's thread index variable was named `parity`
**Fix:** Changed `if (idx >= lat_xyzt)` → `if (parity >= lat_xyzt)` (8 occurrences).

---

## Optimization Summary

| Optimization | Impact |
|-------------|--------|
| Zero host→device scalar memcpy in iteration loop | Eliminated NaN, improved stability |
| Batched `device_vals` initialization (single H→D for all 4 scalars) | Reduced kernel launch overhead |
| Bottom-of-iteration full stream sync | Correct bitwise residual, no stale reads |
| `_send_tmp_` scratch pattern for all dot products | Prevents target slot corruption window |
| Single `cublasH` on main stream for convergence check | Avoids interfering with iteration streams |

---

## Architecture

```
Python (PyTorch)                    C++ CUDA Backend
┌──────────────────┐              ┌──────────────────────────┐
│  conftest.py     │──Cython──▶  │  applyCloverMultigridQcu │
│  (test driver)   │              │         ↓                │
│       ↓          │              │  LatticeCloverMultigrid  │
│  qcu.pyx         │              │    ├─ fine_dslash_op()   │
│  (Cython bridge) │              │    ├─ coarse_dslash_op() │
│       ↓          │              │    ├─ restrict_op()      │
│  libqcu.so       │              │    ├─ prolong_op()       │
│  (CUDA library)  │              │    ├─ bistabcg_iter()    │
└──────────────────┘              │    ├─ v_cycle()          │
                                  │    └─ run()              │
                                  └──────────────────────────┘

Stream usage in bistabcg_iter:
  main (strm):   dslash operations (fine_dslash_op / coarse_dslash_op)
  _a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
  _b_:           give_1rho_prev → give_x_o
  _c_:           dot(t,s), convergence-check dot(r,r)
  _d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega

Synchronization (matching LatticeCloverBistabCg::_run exactly):
  TOP:    sync(strm, _a_, _b_, _c_, _d_)
  Step 1: dot(r_tilde,r) → _a_          (writes to _send_tmp_→host→_rho_)
  Step 2: sync(_b_) → give_1beta(_a_) → sync(_a_) → give_1rho_prev(_b_)
  Step 3: give_p(_a_) → sync(_a_)
  Step 4: sync(strm) → dslash(v,p)(strm)
  Step 5: dot(r_tilde,v)→_d_           (writes to _send_tmp_→host→_tmp0_)
          give_1alpha(_d_) → sync(_d_)
  Step 6: give_s(_a_) → sync(_a_)
  Step 7: sync(strm) → dslash(t,s)(strm)
  Step 8: dot(t,s)→_c_, dot(t,t)→_d_   (both write to _send_tmp_→host→target)
  Step 9: sync(_c_) → give_1omega(_d_) → sync(_d_)
  Step 10: give_r(_a_), give_x_o(_b_)
  BOTTOM: sync(strm, _a_, _b_, _c_, _d_)
```

---

## Files Modified

| File | Status | Change |
|------|--------|--------|
| `cpp/cuda/qcu/include/lattice_clover_multigrid.h` | **New** | Main solver class (~1400 lines) with detailed comments |
| `cpp/cuda/qcu/src/apply_clover_multigrid.cu` | **New** | C API bridge function |
| `cpp/cuda/qcu/python/pyqcu.h` | Modified | Added `applyCloverMultigridQcu` declaration |
| `cpp/cuda/qcu/include/qcu.h` | Modified | Added `lattice_clover_multigrid.h` include |
| `pyqcu/cuda/qcu/qcu.pyx` | Modified | Cython wrapper |
| `pyqcu/cuda/qcu/qcu.pxd` | Modified | Cython declaration |
| `cpp/cuda/qcu/include/lattice_wilson_dslash.h` | Fixed | Removed uninitialized `MPI_Wait` in `run_mpi` |
| `cpp/cuda/qcu/src/wilson_dslash.cu` | Fixed | Fixed `idx`→`parity` variable name (8 occurrences) |
| `cpp/cuda/qcu/include/multigrid.h` | Fixed | Renamed `T`→`Lt` to avoid template parameter shadowing |
| `cpp/cuda/qcu/src/multigrid.cu` | Fixed | Same |
| `cpp/cuda/qcu/include/lattice_multigrid.h` | Fixed | Same |
| `cpp/cuda/qcu/src/apply_multigrid.cu` | Fixed | Same |
| `examples/qcu/conftest.clover.multigrid.py` | **New** | Test script with charts + JSON report |

---

## Output Files (`/root/PyQCU/logs/`)

| File | Content |
|------|---------|
| `clover_multigrid.log` | Full C++ convergence log (PYQCU::SOLVER::MULTIGRID:: format) |
| `multigrid_report.json` | Machine-readable JSON with convergence data |
| `multigrid_result.png` | Performance comparison + convergence history chart |
| `test_multigrid.log` | Python test execution log |
| `multigrid_report.md` | This report |

