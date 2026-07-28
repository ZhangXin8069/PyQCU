# PyQCU Debug & Optimization — Final Results
**Date**: 2026-07-28
**Working Directory**: /root/PyQCU

## Completed Work

### 1. Code Review
- Full codebase reviewed: 23 Python files, 20+ C++ CUDA files, 3 Cython files, build system, docs
- 74 findings documented in `/root/PyQCU/logs/review-2026-07-28.md`
- 14 items confirmed as PASS (design decisions) with code comments and CLAUDE.md updates

### 2. Bug Fixes (12 implemented)
- **Python**: 8 fixes (I/O, stout_smear, operator parity, NPU einsum, bare raise, BiCGStab, test_solver, temp file leak)
- **C++ CUDA**: 4 fixes (complex operator*=, gauss_gauge OOB+leak, LatticeSet leak, MPI_Isend wait)
- All verified by 8/8 test suite passing
- C++ build successful (`libqcu.so` linked)

### 3. Performance Optimization (1 implemented)
- Removed 24 redundant `MPI.Barrier()` calls around blocking `MPI.Sendrecv` in 3 files

### 4. Documentation Updates
- CLAUDE.md: gamma matrix description, negative ward index convention, per-module force_use_npu note, GMRES stub status
- Source code: detailed comments on 12+ design decisions (ward indices, parity reuse, null_vecs template, BLOCK_SIZE, CUDA templates, shape detection heuristic)

### 5. Code Quality
- Removed duplicate `from typing import Optional` import
- Removed debug `print()` calls in `matvec_all`
- Fixed broken indentation from linter collision
- Fixed duplicated error messages from linter collision

## Files Modified
```
pyqcu/lattice/__init__.py        — ward index comments, duplicate import removed
pyqcu/dslash/_operator.py        — sitting None fix, matvec_eo/oe condition fix, debug prints removed
pyqcu/smear/_stout.py            — nstep>1 fix
pyqcu/solver/_bistabcg.py        — ZeroDivisionError fix, verbose stats guard
pyqcu/solver/_gmres.py           — placeholder comment
pyqcu/tools/_io.py               — I/O index order fix (shape + rank decomposition)
pyqcu/tools/_define.py           — temp file cleanup fix
pyqcu/tools/_multigrid.py        — null_vecs template comment
pyqcu/cann/__init__.py           — NPU 3+ operand einsum fix
pyqcu/cuda/define.py             — bare raise → ValueError
pyqcu/testing/__init__.py        — test_solver error message fix

cpp/cuda/qcu/include/lattice_complex.h      — operator*= fix
cpp/cuda/qcu/include/lattice_wilson_dslash.h — MPI_Isend wait fix
cpp/cuda/qcu/include/define.h               — BLOCK_SIZE comment improvement
cpp/cuda/qcu/src/gauss_gauge.cu             — OOB write + GPU leak fix
cpp/cuda/qcu/src/apply_end.cu               — LatticeSet delete fix
cpp/cuda/qcu/src/wilson_dslash.cu           — parity reuse + template comments

/root/PyQCU/CLAUDE.md             — 4 documentation updates
/root/PyQCU/logs/review-2026-07-28.md  — full review report (892 lines)
/root/PyQCU/logs/debug/fix-log.md       — fix tracking log
/root/PyQCU/logs/results/final-report.md — this file
