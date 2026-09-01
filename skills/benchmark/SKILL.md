---
name: benchmark
description: PyQCU 性能基准 skill：覆盖 examples/benchmark 的通用基准，以及 dev87 strict MultiGrid 对 QUDA 的可复现公平计时与显存口径。
---
# examples/benchmark

Performance benchmarks comparing PyTorch, TileLang, and C++ CUDA backend implementations.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Benchmark entry point |
| `env.py` | Benchmark environment configuration |

## Usage

```bash
python examples/benchmark/conftest.py
```

2026-08-24：conftest.py 收集期笔误已修（bug37，pytest 正常收集 exit=0，见
`logs/fix-report-2026-08-24.md`）。基准锚点：4096³ fp16 matmul TileLang 热修后
38.7 TFLOPS ≈ cuBLAS 94%（V100，见 tilelang 技能）。

Strict benchmark records use the current bridge ABI: `params` is CPU
`int32[58]`, `argv` is a seven-element real tensor (`float32` for c64 and
`float64` for c128), and `set_ptrs` is CPU `int64[100]`. `params[57]` is
`_MG_USE_INIT_GUESS_`: formal cold solves set it to `0`; only an explicitly
prefilled `fermion_out` warm start may set it to `1`. Do not report a speedup
from a run whose side records use different ABI, initial-guess, or asset
contracts.

For fast iteration, use `run_strict_fast.py --list` to inspect the command
plan, then `--only <gate>` for one named gate or the default tier 0 for data-free CPU algebra. Tier 1 is the small
CUDA handoff gate. Tier 2 is explicit and intentionally expensive: it invokes
the formal real-gauge/QUDA collector and may emit a fair speedup, so it is not
part of the default edit loop. Keep the protocol/cache/QIO tests separate from
tier 0/1 so a documentation or collector edit does not import external QUDA or
allocate a large gauge before the cheap gate passes.

## Strict MultiGrid vs QUDA

Use `examples/qcu/dev87/bench_strict_vs_quda.py` directly, or through
`run_strict_fast.py --tier 2`, for formal timing. The direct collector is the
authoritative protocol implementation; tier 2 is the same formal gate with
runner-level timeout/JSON handling, not a cheap substitute. The PyQCU side must
use `hierarchy_mode="strict"`/`QudaStrictMultigrid`, fixed fine odd
`target_parity=1` and coarse `start_level=1`; `setup_operator="schur"` is a
legacy compact path, not a substitute. The collector fixes `16×32×32×48`, the
Gauge/RHS/canonical-full-null-vector bundle and SHA256 identities, zero initial
guesses, precision, two warmups, five measured solves, and median/MAD. The
formal input null vectors are canonical complex64 `[12,4,3,X,Y,Z,T]`; PyQCU
converts them to the C++ blocked 10-D ABI
`[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`, while QUDA consumes the separately converted
QIO artifact. Do not confuse canonical input layout with the legacy blocked
runtime layout. Each side must converge and pass the independently recomputed
full Wilson/Clover relative true-residual gate before a speedup is emitted.

The QUDA side requires QIO null vectors converted from that same canonical full dataset plus a manifest whose QIO artifact hashes match. The formal gamma basis is `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`. The fairness gate hashes the canonical full dataset and compares it with `canonical_dataset_sha256`; `source_sha256` records only the original E12 odd-Schur provenance and must never be compared with the full dataset digest. QIO round-trip verification compares the staging and read-back files with a bounded two-file 8 MiB streaming scan, rather than mapping both complete assets. Missing QIO evidence must produce an explicit skip; native random QUDA null vectors are not a fair substitute. Keep Gauge/Clover normalization, mass/κ, RHS, precision, topology, level/block geometry, `coarse_spin=2`, smoother/coarse budgets and stopping criterion aligned, and retain config/input hashes in every side record.

Before formal timing, persist two cheap single-rank gates: the `4^4` reduction smoke (`examples/qcu/dev87/smoke_quda_reduction.py`) and an `8^4` Nc24 setup-only probe with `n_vec=12` and `coarse_spin=2`. Formal collection is blocked until both pass. A smoke is `PASS` only when post-setup resolved/read-back parameters and observed capabilities satisfy the gate; requested CLI values are metadata, not evidence. Record each gate's status and requested-versus-resolved values, without embedding transient build digests in this skill.

`QUDA_MULTIGRID_NVEC_LIST` is a comma-separated compile-time instantiation set. For the formal `n_vec=12`, `coarse_spin=2` path, configure `12,24`: `12` is needed by `BlockOrthogonalize` (`B.size`), while `24 = n_vec × coarse_spin` is the coarse color/coarse-operator instance. A list containing only `12` or only `24` ends in `MPI_ABORT`. After setup, read back and record the effective `n_vec`, `coarse_spin`, coarse color, and build-instance configuration; fail closed on missing or mismatched values.

PyQUDA `QudaMultigridParam` array getters return copies. For every array-valued field (`n_vec`, `nu_pre`, `nu_post`, solver arrays, MMA flags, `vec_load`, `vec_infile`, and similar), copy the whole column, modify the copy, assign the whole column with `setattr`, then immediately read it back and compare. `param.n_vec[0] = 12` (and the analogous `vec_load[0] = ...`) is a silent no-op anti-pattern. For c64 device precision, the QDP host gauge remains contiguous `complex128`; `setPrecision(single)` selects device precision and does not justify downcasting the host QDP gauge.

On the PyQCU side, bind packed runtime assets, copy `strict_setup_stats`, then call `seal_cuda_runtime(runtime_assets_bound=True)` before steady solves; report the detached-storage estimate and observed allocator delta. The strict worker lifecycle is setup → `CudaSchurOp`/`applyInitQcu` → asset bind → Strict init → warmup/steady solves → Strict end → `applyEndQcu`; retain the same per-instance `_SET_INDEX_` throughout this sequence. The outer workspace is one C++ fused allocation, lazily created on the first solve and reused, with exact size `(2*m+5)*B_f+2*B_c`; Python outer-Krylov and coarse-I/O resident bytes are zero. Report this separately from the persistent recursive hierarchy/V-cycle arena, packed assets, Gauge/Clover, backend scratch/halos, caller fields, and CUDA allocated/reserved. Omitting raw `Y` is a real steady-state saving only after native setup duplicates have been sealed away.

Formal memory evidence uses schema version 2 as a hard gate for every successful side record. The `first_solve` object is the first zero-initial-guess warmup, not a measured repeat: it enables the device-wide sampler so lazy native/CUDA solver workspace creation (including PyQCU's lazy fused workspace) is observable. Its `excluded_from_formal_timing` field must be `true`; its duration and memory are diagnostic first-allocation evidence and do not enter formal steady timing or speedup statistics. The separate `steady.untimed_device_memory_probe` is also marked `excluded_from_formal_timing=true` and exists only to sample native device-wide allocations after the measured repeats.

Interpret the schema-v2 memory scopes separately. `steady.baseline.allocated_bytes` and `steady.baseline.reserved_bytes` are PyTorch allocator live and reserved bytes immediately after all warmups and before measured repeats. Each `steady.samples[*].cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` is collected after resetting PyTorch peak statistics immediately before that measured solve; the aggregate `steady.cuda_peak_*` fields are the maxima across measured repeats. `allocated` is the allocator live-allocation high-water value, while `reserved` is the caching-allocator reservation high-water value; neither is a complete device-wide/native-allocation total. The exact device-wide sampler field is `device_used_max_observed_bytes`, the maximum sampled `cudaMemGetInfo` used-bytes value, and may include other processes. Sampler `start()` only launches the worker—the first `cudaMemGetInfo` call is not executed on the main thread. `stop()` signals and joins without taking a final sample; if join times out, it retains the thread handle and the record fails closed. Formal solve timing uses the separate untimed probe, and `setup_seconds` is captured before sampler stop. Filter `nvidia-smi` observations by the target GPU UUID; `nvidia-smi_*_max_observed_bytes` are post-solve snapshots, not peaks or high-water marks. On QUDA setup or warmup failure, stop any active sampler and best-effort destroy multigrid plus free Gauge without masking the primary exception.

Runtime-cache records use schema v2 with a streaming SHA256 for every logical tensor. A cache hit verifies every tensor before any device transfer, uses host chunks of at most about 8 MiB, and performs two logical reads (digest verification, then device transfer); report cache verification/load time rather than treating a hit as zero-cost. If no-clobber publication encounters a competing target with the same identity, reuse is allowed only after fully validating its manifest, every dataset attribute, and every tensor SHA256; identity equality or file existence alone is insufficient. Formal Galerkin setup uses a four-full-field-arena budget independent of outer Krylov: c64 defaults to colored `C=12` with a `4 GiB` setup cap, while c128 defaults to `C=1` with `1 GiB`. The `512 MiB` c64 value is the outer-Krylov default (scaled to `1 GiB` for c128), not a Galerkin cap. Record requested/effective column batch, projection batch, setup cap and observed setup statistics separately; a formal run must fail rather than silently shrink the batch.

PyQCU cache experiments must set both `--strict-cache-dir` and `--cache-expect {miss,hit}` explicitly, with the cache directory kept inside the repository. Use the smoke profile plus `miss` for cold-cache generation and the formal profile plus `hit` for a measured cache-hit run; a mismatch must fail before heavy imports or device allocation. Reserve `any`, reduced repeats and relaxed tolerances for smoke runs. A smoke document must never emit a formal speedup.

On WSL2, patch only an independent QUDA source shadow with `examples/qcu/dev87/quda_wsl2_reduce_sync.patch`; do not modify `refer/git-rep/quda`. Set `DEV87_REDUCE_SYNC=1` and put the patched install's `lib` first in `LD_LIBRARY_PATH`. Qualify the selected install dynamically on every production smoke: record the actual `libquda.so` and `libqmp.so` paths and SHA256 values, verify the WSL2 marker/path precedence, and require `BUILD_QDP_INTERFACE`, `HAVE_QIO`, `QMP_COMMS`, `QUDA_RECONSTRUCT=7`, plus a `QUDA_PRECISION` bitmask containing the requested precision. For 12 fine near-null vectors with `coarse_spin=2`, `QUDA_MULTIGRID_NVEC_LIST` must contain both `12` and `24`; compiling only either value aborts in the multigrid setup path. Do not hard-code a production digest in this skill or a synthetic fixture; every changed build needs a fresh reduction smoke. Patched WSL2 timing is environment-scoped evidence, not a portable upstream-QUDA claim.

Run and merge sides without rerunning a successful compatible record:

```bash
python examples/qcu/dev87/bench_strict_vs_quda.py --profile formal --side pyqcu --strict-cache-dir /root/PyQCU/data/strict_runtime_cache --cache-expect hit --output pyqcu.json
python examples/qcu/dev87/bench_strict_vs_quda.py --profile formal --side quda --quda-nullvec-prefix PREFIX --quda-nullvec-manifest MANIFEST --output quda.json
python examples/qcu/dev87/bench_strict_vs_quda.py --merge pyqcu.json quda.json --output combined.json
```

Only a merged document with matching config/input hashes, both side statuses `ok`, passing true residuals, and `comparison.fair=true` may report `speedup_pyqcu_over_quda = median(QUDA)/median(PyQCU)`. Do not claim PyQCU is faster until repeated fair runs show a stable value above one; distinguish patched or unhealthy QUDA environments from portable results.
