# Strict MultiGrid 快速闸门（pyqcu.testing）

本文件是 [`../SKILL.md`](../SKILL.md) 的按需参考：`run_strict_fast.py` 的分层闸门、
QUDA/PyQUDA 对照契约、Strict runtime cache / 内存判据，以及 2026-09 的回归记录。
日常摘要与最短命令入口见 SKILL.md「Strict MultiGrid Fast Gates」；这里保留完整细节，
文中的时点数字（passed 数、耗时）只用于对照，不构成永久承诺。

## Strict MultiGrid Fast Gates

Use `python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py`; tiers are cumulative and the default is tier 0. For the shortest edit loop, first run `--list` (no environment sourcing or setup), then run the default tier with `--fail-fast --json <path>` when a machine-readable result is useful:

```bash
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --list
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --fail-fast --json strict-fast.json
```

- **Tier 0 — CPU algebra smoke:** the focused synthetic suite covers 20 checks for exports, FGMRES edge cases (including complex-Givens phase cancellation), strict mode/geometry guards, `R=P†`, full-coarse parity transfer, MATPC, `X/Y/Yhat` assets/layouts, recursive null-vector propagation, matrix-free guards and colored Galerkin batching/memory models. This is the edit-loop default.
- **Tier 1 — CUDA small lattice:** cumulatively adds strict primitive/V-cycle/complete-solve and fused-C++ FGMRES checks covering lazy persistent workspace reuse, warm x0, budget/descriptor guards and complex128 dispatch. Runtime depends on GPU, driver and build; do not encode a fixed seconds claim. Use it before handing off a CUDA change, and ensure Python does not regain a duplicate Krylov arena.
- **Tier 2 — real gauge + QUDA formal gate:** only selected explicitly with `--tier 2`; runs the formal `bench_strict_vs_quda.py` collector with the canonical real-gauge/null-vector bundle, cache-hit and QIO contracts. It records correctness, true residual, setup/solve timing and schema-v2 memory evidence; a fair speedup is emitted only when both sides pass. It may write its documented dev87 artifacts.

Before any formal QUDA comparison, run and persist two fast, single-rank gates: the `4^4` reduction smoke (`pyqcu/testing/qcu/strict/quda_comparison/smoke_quda_reduction.py`) and an `8^4` Nc24 setup-only probe using `n_vec=12`, `coarse_spin=2`, and no timed solve. Both must be green before formal collection. Judge smoke success from resolved/read-back parameters after setup, never from requested arguments alone; missing resolved evidence is a failure. The formal path must build `QUDA_MULTIGRID_NVEC_LIST=12,24` (comma-separated): `12` serves `BlockOrthogonalize`'s `B.size`, and `24 = n_vec × coarse_spin` serves the coarse color/operator. A build containing only `12` or only `24` ends in `MPI_ABORT`.

Strict CUDA tests must use `hierarchy_mode="strict"`/`StrictMultigrid`, fixed fine `target_parity=1` and coarse `start_level=1`; `setup_operator="schur"` is not a substitute for Strict. Keep the per-instance `_SET_INDEX_` fixed from `CudaSchurOp` construction through Strict init, V-cycle/FGMRES and Strict end; the legacy increment rule is tested separately. For ABI edits, also assert CPU `int32[58]`/`int64[100]` controls and the `params[57]` cold/warm behavior; do not use a fast gate that only checks requested CLI metadata.

When configuring PyQUDA, `QudaMultigridParam` array getters return copies. Copy each complete array column, edit it, assign the complete column with `setattr`, and immediately read it back; indexed mutation such as `param.n_vec[0] = 12` or `param.vec_load[0] = ...` silently changes nothing. Keep the QDP host gauge contiguous `complex128` even for c64 device precision; device `setPrecision(single)` is not a host-gauge dtype conversion.

The runner supports `--list`, `--only <gate>` (repeatable), per-command `--timeout`, `--fail-fast`, and `--json`. `--only` is the short edit-loop path for a named gate and bypasses cumulative tier selection. Keep tier 0 data-free and single-startup where possible; never move real-gauge setup or external QUDA imports into the default gate. The protocol/cache/QIO gate is a seconds-scale suite kept separate from the tiered runner. Run the four focused files together when changing the collector, cache or conversion contract:

```bash
python -B -m pytest -q -p no:cacheprovider \
  pyqcu/testing/qcu/strict/quda_comparison/test_prepare_fair_nullvec.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_convert_full_nullvec_to_quda_qio.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_bench_strict_protocol.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_strict_runtime_cache.py
```

The default tier 0 embeds the three pure-CPU Galerkin fast checks; `--only cpu-smoke` runs them in the same pytest startup, while `--only <other-gate>` isolates a single edit target. Benchmark protocol tests require repository-contained cache directories, persist cache `directory/expect` in the execution record, and prove that a hit/miss mismatch fails before heavy imports or device allocation. They also cover QMP FUNNELED initialization and atexit lifetime without importing PyQUDA. WSL2 guard fixtures must fail closed when forced synchronization is disabled or the selected `libquda.so` is missing, not first in `LD_LIBRARY_PATH`, or lacks the patch marker; synthetic fixtures must assert `report["library_sha256"] == sha256(fixture_binary)`. Qualify the selected production library dynamically in the real reduction smoke rather than hard-coding its digest.

MPI coverage is deliberately separate from the tiered runner:

```bash
python -m pytest -q -p no:cacheprovider pyqcu/testing/qcu/strict/quda_comparison/test_strict_mpi_preflight.py
mpirun -np 2 python -m pytest -q -p no:cacheprovider pyqcu/testing/qcu/strict/quda_comparison/test_strict_mpi_preflight.py
```

These MPI tests cover rank-symmetric preflight plus c64/c128 global dot/norm reduction. Current capabilities declare setup/full/compact/R-P halos, global reduction and distributed fused FGMRES all true. The repository additionally has a two-rank P100 end-to-end strict solve with an independent true residual below `1e-6`. Keep the production gates: one process/rank/device, symmetric capability preflight and independent residual validation.

2026-09-17 方向三补充：tier 0 为 20 passed，tier 1 `cuda-strict` 为
11 passed（新增 c128 blocked basis + c64 CPU-staged Galerkin 与 CUDA
runtime assets 的数值等价检查），`cuda-fused-fgmres` 为 3 passed。CPU
block staging 的正式契约是 `retain_blocks=False`；若测试或调用方把 CPU
canonical blocks 留到安装阶段，必须 fail closed，而不是静默混入 CUDA
asset。

2026-09-17 全量闸门：`source ./env.sh && source
pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh && python pyqcu/testing/qcu/strict/quda_comparison/run_all.py
--with-quda` 在 V100 上 `PASS 5/5`，总耗时 `757.4 s`；其中
`quda_solve_scaled_agreement` 与 `quda_mg_scaled_agreement` 均通过，
后者的 PyQCU wall 为 `1.22 s`、QUDA setup/solve 为 `613.78/70.05 s`。
该结果只证明既有 dev87 回归集合未被本轮 cache/staging 改动破坏，不替代
单独的速度比报告。

parity 分组改动后复跑同一闸门，2026-09-17 再次 `PASS 5/5`，总耗时
`756.3 s`；PyQCU wall 为 `1.18 s`、QUDA setup/solve 为
`615.33/69.60 s`。该复跑确认新的 16-bucket source 顺序没有破坏
dev87 的算子/求解器/QUDA 对照回归。

随后使用 `QCU_CUDA_ARCHITECTURES='70-real;70-virtual'` 重建 native V100
`libqcu.so`（`cuobjdump` 含 `sm_70.cubin`）并第三次运行同一闸门：
`PASS 5/5`，总耗时 `751.2 s`；PyQCU wall 为 `1.17 s`、QUDA
setup/solve 为 `609.79/69.73 s`。因此 CUDA 源码、Python strict
路径和 QUDA 对照在 sm70 cubin 下均已回归。

修正 `checkCudaErrors`/`checkMpiErrors` 双重求值后，重新构建 sm70
并第四次运行全量闸门：`PASS 5/5`，总耗时 `755.5 s`；PyQCU wall
为 `1.17 s`、QUDA setup/solve 为 `613.31/70.26 s`。该次结果同时
覆盖了错误宏修复后的 C++ 二进制。

Runtime-cache tests must enforce schema v2 per-tensor streaming SHA256, reject any tensor/metadata tamper before device transfer, bound host chunks to about 8 MiB, and account for two logical reads on a hit. A same-identity concurrent-publication test must fully validate the winning target's manifest, dataset attrs, and tensor SHA256 values before reuse. Fair-QIO protocol tests must fingerprint canonical full `[12,4,3,X,Y,Z,T]` data against `canonical_dataset_sha256`, require `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`, and verify round-trip content with a two-file 8 MiB streaming scan; `source_sha256` is checked only as E12 provenance.

Strict memory tests must distinguish live allocation from allocator reservation: before the first solve the C++ fused workspace is planned but not resident; after it, resident bytes must equal `(2*m+5)*B_f+2*B_c`. Galerkin tests use a separate four-full-field-arena budget: c64 production selects colored `C=12` under a `4 GiB` setup cap, while c128 stays at `C=1` under `1 GiB`; the c64 `512 MiB` value belongs only to outer Krylov. The formal benchmark's memory schema version 2 is a success-record hard gate: sampler start must not call `mem_get_info` on the main thread, stop must not add a final sample, and join timeout must retain the thread handle and fail closed. Require `device_used_max_observed_bytes`, keep the independent device-wide probe and sampler stop outside formal timing/`setup_seconds`, and filter `nvidia-smi` by target GPU UUID with fields named only `max_observed`. QUDA setup and warmup exception tests must release the sampler, multigrid, and Gauge while preserving the primary failure. Warm up, repeat solves, assert no new Torch allocation and stable owned/live bytes, then call `close()` while retaining the solver object and verify hierarchy slots/assets are released. Do not call `empty_cache()` before the leak assertion.
