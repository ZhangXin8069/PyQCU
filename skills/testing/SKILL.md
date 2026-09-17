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

- **Tier 0 — CPU algebra smoke:** the focused synthetic suite covers 20 checks for exports, FGMRES edge cases (including complex-Givens phase cancellation), strict mode/geometry guards, `R=P†`, full-coarse parity transfer, MATPC, `X/Y/Yhat` assets/layouts, recursive null-vector propagation, matrix-free guards and colored Galerkin batching/memory models. This is the edit-loop default.
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

2026-09-17 实测：单 rank 为 18 passed/9 skipped；`mpirun -np 2` 每个 rank
为 14 passed/13 skipped。异常 capability 路径按预期 fail closed。该结果只
证明阶段 1 的全局标量归约和 preflight，不代表 halo 或分布式 fused
FGMRES 已启用。

2026-09-17 方向三补充：tier 0 为 20 passed，tier 1 `cuda-strict` 为
11 passed（新增 c128 blocked basis + c64 CPU-staged Galerkin 与 CUDA
runtime assets 的数值等价检查），`cuda-fused-fgmres` 为 3 passed。CPU
block staging 的正式契约是 `retain_blocks=False`；若测试或调用方把 CPU
canonical blocks 留到安装阶段，必须 fail closed，而不是静默混入 CUDA
asset。

2026-09-17 全量闸门：`source ./env.sh && source
examples/qcu/dev87/quda_env.sh && python examples/qcu/dev87/run_all.py
--with-quda` 在 V100 上 `PASS 5/5`，总耗时 `757.4 s`；其中
`quda_solve_scaled_agreement` 与 `quda_mg_scaled_agreement` 均通过，
后者的 PyQCU wall 为 `1.22 s`、QUDA setup/solve 为 `613.78/70.05 s`。
该结果只证明既有 dev87 回归集合未被本轮 cache/staging 改动破坏，不替代
单独的速度比报告。

parity 分组改动后复跑同一闸门，2026-09-17 再次 `PASS 5/5`，总耗时
`756.3 s`；PyQCU wall 为 `1.18 s`、QUDA setup/solve 为
`615.33/69.60 s`。该复跑确认新的 16-bucket source 顺序没有破坏
dev87 的算子/求解器/QUDA 对照回归。

Runtime-cache tests must enforce schema v2 per-tensor streaming SHA256, reject any tensor/metadata tamper before device transfer, bound host chunks to about 8 MiB, and account for two logical reads on a hit. A same-identity concurrent-publication test must fully validate the winning target's manifest, dataset attrs, and tensor SHA256 values before reuse. Fair-QIO protocol tests must fingerprint canonical full `[12,4,3,X,Y,Z,T]` data against `canonical_dataset_sha256`, require `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`, and verify round-trip content with a two-file 8 MiB streaming scan; `source_sha256` is checked only as E12 provenance.

Strict memory tests must distinguish live allocation from allocator reservation: before the first solve the C++ fused workspace is planned but not resident; after it, resident bytes must equal `(2*m+5)*B_f+2*B_c`. Galerkin tests use a separate four-full-field-arena budget: c64 production selects colored `C=12` under a `4 GiB` setup cap, while c128 stays at `C=1` under `1 GiB`; the c64 `512 MiB` value belongs only to outer Krylov. The formal benchmark's memory schema version 2 is a success-record hard gate: sampler start must not call `mem_get_info` on the main thread, stop must not add a final sample, and join timeout must retain the thread handle and fail closed. Require `device_used_max_observed_bytes`, keep the independent device-wide probe and sampler stop outside formal timing/`setup_seconds`, and filter `nvidia-smi` by target GPU UUID with fields named only `max_observed`. QUDA setup and warmup exception tests must release the sampler, multigrid, and Gauge while preserving the primary failure. Warm up, repeat solves, assert no new Torch allocation and stable owned/live bytes, then call `close()` while retaining the solver object and verify hierarchy slots/assets are released. Do not call `empty_cache()` before the leak assertion.

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`

## Important Notes

- Tests use `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` for MPI reference comparison
- Reference HDF5 data lives in `examples/data/`
- The `path` variable in tests is computed from `pyqcu.__file__` to locate data files
- **R3 fix:** Tests now include `assert` statements so pytest can detect failures

## 分层 trace / matrix tests（2026-09-16）

`test_bench_strict_protocol.py` covers variable `--lattice/--levels`,
recursive cache manifests, default-off PyQCU/QUDA trace parsing, and QUDA
outer-iteration scope.  `test_bench_mg_matrix.py` verifies two/three levels,
c64/c128, and three lattice volumes.  `summarize_mg_trace.py` aggregates
diagnostic stage/residual rows; never use its trace wall times as formal
performance evidence.

### 2026-09-17 double MG / BiCGStab 回归

新增回归必须覆盖：

- `--quda-strategy aligned|library` 的预期参数：aligned 为 MR/一次递归；
  library 为 PyQUDA 原生 CA-GCR、`nu_post=8`、coarse maxiter 16。
- `QUDA_MULTIGRID_DOUBLE=ON` 与 `QUDA_PRECISION=12` 的 CMake provenance；
  c128 正式记录必须含 double bit，且 `precision_null` 全为
  `QUDA_DOUBLE_PRECISION`。
- BiCGStab 参考解通过独立真残差门；QUDA 参考路径必须清除
  `invert_param.preconditioner`，防止 MG 指针残留导致 solver factory
  拒绝 BiCGStab。
- 正式 profile 在 trace 环境变量存在时拒绝运行；只有
  `--allow-trace` 才能产生诊断 JSON。
- c128 `16·32·32·48` 三层当前有两条路径：formal 冷 `C=1` 仍未完成；
  extended `C=4`/`K=16` 全 double setup 已在 32 GiB V100 完成，并可
  通过 asset-identity cache 进入 formal `C=1` solve。测试必须区分
  cold-C1 能力缺口与 extended 全 double 成功，不得把 cache-hit solve
  冒充冷启动总时间。
- `--quda-coarse-precision single` 的 mixed-coarse smoke 必须检查
  `gauge_param`/`invert_param` precision readback；只修改
  `invert_param` 不足以让 coarse single 与 fine double 一致。
- `--pyqcu-strict-block-precision single` 与
  `--strict-galerkin-projection-batch 1` 是 setup 低内存实验选项；
  它们仍不能替代当前已完成的 extended 全 double `C=4` setup；不得
  据单精度路径宣称 PyQCU mixed-coarse 已支持。

快速命令：

```bash
python -B -m pytest -q -p no:cacheprovider \
  examples/qcu/dev87/test_bench_strict_protocol.py \
  examples/qcu/dev87/test_bench_mg_matrix.py
python examples/qcu/dev87/run_strict_fast.py --tier 1
python examples/qcu/dev87/run_all.py --with-quda
```

正式 JSON 来源：
`data/mg_matrix_20260916_round2/formal-*-l*/benchmark.json`；
trace 开销证据：`trace-small-c64-l3.json`。trace-on 数字只用于说明
trace 开销，不能进入 `speedup_pyqcu_over_quda`。

`run_quda_py.py` 必须在首次导入 PyQUDA 前按 strict benchmark 的方式初始化
QMP；先导入 PyQUDA 会直接触发 `QMP_comm_get_default` abort，进程内无法恢复。
legacy solve/MG case 在 precision-12 组合构建中显式选择 single precision，
以保持与 c64 PyQCU 参考一致；`case_opcmp` 仍保留其有意使用的 double
precision 路径。当前 `run_all.py --with-quda` 的 5 项断言全绿。
