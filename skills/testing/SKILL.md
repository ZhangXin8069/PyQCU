---
name: testing
description: pyqcu.testing 目录的完整生成 skill：全组件集成测试与 strict MultiGrid 三级快速闸门，含参考数据、CUDA/MPI、显存稳定性和日志约定。
---
# pyqcu.testing

Integration tests for all PyQCU components. Tests are Python functions imported by `examples/*/conftest.py` entry points.

## Architecture

All test functions live in `pyqcu/testing/__init__.py`. They import from all PyQCU subpackages (`lattice`, `solver`, `dslash`, `tools`, `smear`). Each `examples/*/conftest.py` acts as a pytest entry point that imports specific test functions and calls them. The conftest files are manually edited to uncomment the test(s) to run.

The module imports `tilelang` at module level (with try/except fallback) for `test_matmul`.

## Test Functions

### `test_lattice(lat_size, dtype, device)`
Tests SU(3) gauge generation + gamma matrix algebra.
- Generates random gauge field, runs `check_su3`
- Verifies γ_μ² = I for all 4 gamma matrices
- **Assertion:** `check_su3` must return True

### `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)`
Tests Wilson Dirac operator.
- `with_data=False`: Generates random gauge field + source, applies full Wilson operator and eo/oe preconditioned variants
- `with_data=True`: Loads reference HDF5 data (`refer.wilson.*.L32K0_125.*.h5`), validates operator.matvec against known result
- **Assertion:** Relative difference < 1e-4

### `test_dslash_parity(lat_size, kappa, dtype, device)`
Tests parity-preconditioned Wilson+Clover operator with MPI.
- Distributes gauge field across MPI grid
- Root rank computes full operator result as reference
- All ranks compare local parity-preconditioned operator against reference
- Tests both `matvec_all` and `matvec_eeo`/`matvec_oeo` paths

### `test_dslash_clover(device, with_data, dtype)`
Tests Clover term construction.
- `with_data=True`: Loads reference data, validates clover term and inverse against known results
- `with_data=False`: Tests parallel vs serial clover construction across MPI grid

### `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)`
Tests BiStabCG and multigrid solvers.
- `method='bistabcg'`: Standard or parity-preconditioned BiCGStab
- `method='multigrid'`: Full multigrid V-cycle with `init()` + `solve()` + `plot()`
- `with_data=True`: Validates against reference Wilson data
- **Assertion:** Relative error < 1e-3

### `test_matmul()`
Benchmarks TileLang JIT-compiled matrix multiply vs PyTorch (cuBLAS/MKL).
- GPU: 4096×4096 matmul, TileLang vs cuBLAS
- CPU: 1024×1024 matmul, TileLang (LLVM or C backend) vs MKL/OneDNN
- Prints TFLOPS comparison table

### `test_smear_stout(lat_size, device, dtype)`
Tests stout smearing across MPI grid.
- Distributes gauge field, root computes whole-grid reference
- All ranks compare local parallel smear against reference
- Verifies SU(3) before and after smearing

### `test_smear_wuppertal()` (test16, 2026-08-24)
Wuppertal Gaussian smearing with triple invariants (cpu+cuda both PASS):
- `nstep>=1` guard; U=I fixed point (<1e-4); white-noise contraction ratio < 1.0.
- Golden criterion: np=2/4 constant-source rel≈5e-08.

### `verify_nullvecs()` — block structure required
Null-vector quality diagnostic requires the explicit 10-dim block structure argument
(documented in dev85); non-block layouts fail fast instead of being auto-corrected.

## Running Tests

```bash
cd examples && pytest .                              # all conftest.py files
mpirun -np 4 python examples/pyqcu/conftest.py       # single file with MPI
```

## Strict MultiGrid Fast Gates

Use `python examples/qcu/dev87/run_strict_fast.py`; tiers are cumulative and the default is tier 0. For the shortest edit loop, first run `--list` (no environment sourcing or setup), then run the default tier with `--fail-fast --json <path>` when a machine-readable result is useful:

```bash
python examples/qcu/dev87/run_strict_fast.py --list
python examples/qcu/dev87/run_strict_fast.py --fail-fast --json strict-fast.json
```

- **Tier 0 — CPU algebra smoke:** the focused synthetic suite covers 19 checks for exports, FGMRES edge cases (including complex-Givens phase cancellation), strict mode/geometry guards, `R=P†`, full-coarse parity transfer, MATPC, `X/Y/Yhat` assets/layouts, matrix-free guards and colored Galerkin batching/memory models. This is the edit-loop default.
- **Tier 1 — CUDA small lattice:** cumulatively adds strict primitive/V-cycle/complete-solve and fused-C++ FGMRES checks covering lazy persistent workspace reuse, warm x0, budget/descriptor guards and complex128 dispatch. Runtime depends on GPU, driver and build; do not encode a fixed seconds claim. Use it before handing off a CUDA change, and ensure Python does not regain a duplicate Krylov arena.
- **Tier 2 — real gauge + QUDA formal gate:** only selected explicitly with `--tier 2`; runs the formal `bench_strict_vs_quda.py` collector with the canonical real-gauge/null-vector bundle, cache-hit and QIO contracts. It records correctness, true residual, setup/solve timing and schema-v2 memory evidence; a fair speedup is emitted only when both sides pass. It may write its documented dev87 artifacts.

Before any formal QUDA comparison, run and persist two fast, single-rank gates: the `4^4` reduction smoke (`examples/qcu/dev87/smoke_quda_reduction.py`) and an `8^4` Nc24 setup-only probe using `n_vec=12`, `coarse_spin=2`, and no timed solve. Both must be green before formal collection. Judge smoke success from resolved/read-back parameters after setup, never from requested arguments alone; missing resolved evidence is a failure. The formal path must build `QUDA_MULTIGRID_NVEC_LIST=12,24` (comma-separated): `12` serves `BlockOrthogonalize`'s `B.size`, and `24 = n_vec × coarse_spin` serves the coarse color/operator. A build containing only `12` or only `24` ends in `MPI_ABORT`.

Strict CUDA tests must use `hierarchy_mode="strict"`/`QudaStrictMultigrid`, fixed fine `target_parity=1` and coarse `start_level=1`; `setup_operator="schur"` is not a substitute for Strict. Keep the per-instance `_SET_INDEX_` fixed from `CudaSchurOp` construction through Strict init, V-cycle/FGMRES and Strict end; the legacy increment rule is tested separately. For ABI edits, also assert CPU `int32[58]`/`int64[100]` controls and the `params[57]` cold/warm behavior; do not use a fast gate that only checks requested CLI metadata.

When configuring PyQUDA, `QudaMultigridParam` array getters return copies. Copy each complete array column, edit it, assign the complete column with `setattr`, and immediately read it back; indexed mutation such as `param.n_vec[0] = 12` or `param.vec_load[0] = ...` silently changes nothing. Keep the QDP host gauge contiguous `complex128` even for c64 device precision; device `setPrecision(single)` is not a host-gauge dtype conversion.

The runner supports `--list`, `--only <gate>` (repeatable), per-command `--timeout`, `--fail-fast`, and `--json`. `--only` is the short edit-loop path for a named gate and bypasses cumulative tier selection. Keep tier 0 data-free and single-startup where possible; never move real-gauge setup or external QUDA imports into the default gate. The protocol/cache/QIO gate is a seconds-scale suite kept separate from the tiered runner. Run the four focused files together when changing the collector, cache or conversion contract:

```bash
python -B -m pytest -q -p no:cacheprovider \
  examples/qcu/dev87/test_prepare_fair_nullvec.py \
  examples/qcu/dev87/test_convert_full_nullvec_to_quda_qio.py \
  examples/qcu/dev87/test_bench_strict_protocol.py \
  examples/qcu/dev87/test_strict_runtime_cache.py
```

The default tier 0 embeds the three pure-CPU Galerkin fast checks; `--only cpu-smoke` runs them in the same pytest startup, while `--only <other-gate>` isolates a single edit target. Benchmark protocol tests require repository-contained cache directories, persist cache `directory/expect` in the execution record, and prove that a hit/miss mismatch fails before heavy imports or device allocation. They also cover QMP FUNNELED initialization and atexit lifetime without importing PyQUDA. WSL2 guard fixtures must fail closed when forced synchronization is disabled or the selected `libquda.so` is missing, not first in `LD_LIBRARY_PATH`, or lacks the patch marker; synthetic fixtures must assert `report["library_sha256"] == sha256(fixture_binary)`. Qualify the selected production library dynamically in the real reduction smoke rather than hard-coding its digest.

MPI coverage is deliberately separate from the tiered runner:

```bash
python -m pytest -q -p no:cacheprovider examples/qcu/dev87/test_strict_mpi_preflight.py
mpirun -np 2 python -m pytest -q -p no:cacheprovider examples/qcu/dev87/test_strict_mpi_preflight.py
```

These MPI tests cover rank-symmetric preflight plus c64/c128 global dot/norm reduction. The expected capabilities are `global_reduction=True` but `setup_halo=False`, `full_halo=False`, `compact_halo=False`, and distributed `fused_fgmres=False`; production multi-rank solves must still be rejected, and passing these tests must not be reported as a distributed strict solve.

Runtime-cache tests must enforce schema v2 per-tensor streaming SHA256, reject any tensor/metadata tamper before device transfer, bound host chunks to about 8 MiB, and account for two logical reads on a hit. A same-identity concurrent-publication test must fully validate the winning target's manifest, dataset attrs, and tensor SHA256 values before reuse. Fair-QIO protocol tests must fingerprint canonical full `[12,4,3,X,Y,Z,T]` data against `canonical_dataset_sha256`, require `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`, and verify round-trip content with a two-file 8 MiB streaming scan; `source_sha256` is checked only as E12 provenance.

Strict memory tests must distinguish live allocation from allocator reservation: before the first solve the C++ fused workspace is planned but not resident; after it, resident bytes must equal `(2*m+5)*B_f+2*B_c`. Galerkin tests use a separate four-full-field-arena budget: c64 production selects colored `C=12` under a `4 GiB` setup cap, while c128 stays at `C=1` under `1 GiB`; the c64 `512 MiB` value belongs only to outer Krylov. The formal benchmark's memory schema version 2 is a success-record hard gate: sampler start must not call `mem_get_info` on the main thread, stop must not add a final sample, and join timeout must retain the thread handle and fail closed. Require `device_used_max_observed_bytes`, keep the independent device-wide probe and sampler stop outside formal timing/`setup_seconds`, and filter `nvidia-smi` by target GPU UUID with fields named only `max_observed`. QUDA setup and warmup exception tests must release the sampler, multigrid, and Gauge while preserving the primary failure. Warm up, repeat solves, assert no new Torch allocation and stable owned/live bytes, then call `close()` while retaining the solver object and verify hierarchy slots/assets are released. Do not call `empty_cache()` before the leak assertion.

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`

## Important Notes

- Tests use `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` for MPI reference comparison
- Reference HDF5 data lives in `examples/data/`
- The `path` variable in tests is computed from `pyqcu.__file__` to locate data files
- **R3 fix:** Tests now include `assert` statements so pytest can detect failures
