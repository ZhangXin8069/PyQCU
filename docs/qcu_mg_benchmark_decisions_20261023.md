# Large-lattice MultiGrid benchmark decisions (2026-10-23)

## Strict-MultiGrid setup is not accepted

For the requested production lattices, Strict-MultiGrid initialization is
marked **SKIPPED** with reason `strict_initialization_time_unacceptable`.

The single-rank CUDA Strict two-level attempt on `24x24x24x64`, c64,
`nvec=12`, block `2x2x2x2` ran from `03:07:22` to `03:41:12` without producing
stdout or a side JSON. The observed initialization lower bound is therefore
`2030 s` (about 33m50s), with the V100 at roughly 32 GiB resident and 100% SM
utilization. Because setup never completed, this is not a completed timing
measurement.

The corresponding CUDA Legacy cold asset build is `323.906 s`; the observed
Strict lower bound is already `6.27x` that cost. Strict is consequently omitted
from the `24x24x24x64` and `32x32x32x64` benchmark rows. This is a project
acceptance decision, not a numerical PASS or FAIL and not a performance result.

## Scope

The skip applies to Torch Strict, CUDA Strict, KUPL Strict and QUDA Strict
benchmark rows for these lattices. It does not disable explicit Strict
correctness debugging or ABI tests when separately requested.

## Evidence

- Machine-readable decision:
  [`strict-skip.json`](../logs/qcu_l24x64_mpi_20261023/strict-skip.json)
- Incomplete stdout: `logs/qcu_l24x64_mpi_20261023/single/cuda-strict-2l-smoke.stdout`
- Legacy cold setup:
  `data/benchmark_legacy_vs_strict_20261022/reports/legacy_build_24x24x24x64.json`
