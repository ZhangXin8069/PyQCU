# Performance

## Measurement protocol

Every number below is tied to a reported platform, lattice, precision, backend,
solver, and iteration count. Cross-machine absolute seconds are **not**
comparable; CPU and GPU rows are never combined into a single speed-up.
Historical rows are retained because they document the implementation on
hardware that is not available in this checkout. Their raw JSON files are not
part of the current working tree; the source report and the git tag are the
provenance.

The local `8x8x16x16` case uses a Gaussian SU(3) gauge field (`sigma=0.1`,
`seed=42`), a deterministic complex64 source (`seed=42`), `mass=0.17`,
`csw=1.0`, and `atol=1e-6`. The Clover BiStabCG driver reports an independent
full-lattice true residual.

## Historical cross-platform results

| Case | Platform | Lattice / precision | Backend and solver | Result | Comparison | Source | Reproducibility |
|---|---|---|---|---|---|---|---|
| A | 1× Tesla V100-SXM2-32GB, sm_70 | 24³×64, complex64 | CUDA `applyCloverBistabCgQcu` | 0.147801 s, 18 iterations, 8.211 ms/iteration, true residual 6.709e-7 | QUDA Clover BiCGStab 0.156680 s / 19 iterations → 1.060× | `docs/three_way_benchmark_20261022.md:82-84` | Historical; requires V100 and the `dev103`-era checkout |
| B | Intel Xeon E5-2696 v3, 18 cores, 1 NUMA | 24³×64, complex64 | KUPL parity-Schur BiStabCG (`--restart 0`) | 8 ranks×2 threads, grid 1×4×2×1: 7.532 s / 19 iterations | 1 rank×4 threads 19.229 s → 2.553×; 4 ranks×4 threads 7.842 s | `docs/three_way_benchmark_20261022.md:85-86,180-185` | Historical; same Xeon class required for absolute seconds |
| C | P100-PCIE-16GB (sm_60) vs V100-SXM2-32GB (sm_70) | 8×8×16×16, complex64 | CUDA `applyCloverBistabCgQcu`, solve wall including load | P100 0.0425 s; V100 0.0149 s; residuals 2.846712e-7 / 2.846721e-7 | Architecture A/B, not a speed-up claim | `docs/three_way_benchmark_20261022.md:321-323` | Historical; load-inclusive timing, raw JSON absent |
| D | 1× Tesla V100-SXM2-32GB | 8×8×8×16, complex64, real LIME prefix | CUDA Strict-MG `applyMultigridStrictFgmresQcu` | 0.021693 s steady median, 10 iterations, true residual 3.348e-7 | QUDA aligned MG 0.144755 s / 8 outer GCR → 6.673× | `docs/qcu_six_way_benchmark_20261023.md:128-131` | Time-point evidence; working tree not a clean snapshot |
| E | 1× A100-PCIE-40GB | 32³×64, complex64 | CUDA Legacy-MG `applyCloverMultigridQcu`, restart 20, E12, dof [12,12] | 1/2/4 ranks: 0.214456 / 0.219506 / 0.218343 s, 20 iterations | QUDA GCR-MG 0.482426 / 0.314393 / 0.287465 s → 2.2495× / 1.4323× / 1.3166× | `docs/qcu_mg_performance_ledger_20261029.md:92-110` | Historical; needs A100, L32 LIME, and the E12 cache |
| F | 1× A100-PCIE-40GB | 32³×64, complex64 | CUDA native BiStabCG | 1 rank 0.122422 s vs QUDA 0.130358 s; 2 ranks 0.115411 s vs QUDA 0.096089 s | 1.065× at 1 rank; **0.833× at 2 ranks** | `docs/qcu_mg_performance_ledger_20261029.md:48-59` | Historical; shows the native solver is not universally faster |
| G | 1× DCU Pre-Wukong D160A1, gfx906, 16 GB, DTK 23.10 | 24³×64, complex64 | CUDA(HIP)-compatible QCU Legacy-MG | 1/2/4 ranks: 0.301690 / 0.461908 / 1.024669 s, 20 iterations, residual ≈2.58e-7 | QUDA `BLOCKED-RUNTIME`; no ratio | `docs/qcu_quda_solver_dcu_l24_20261029.md:99-103` | Historical; requires the same DCU model and DTK |
| H | 1× Tesla V100, tag `test27`/`stab55` | 16×32×32×48, complex64, 3 levels | PyQCU Strict-MG, trace-off | 0.808893 s / 14 iterations | QUDA aligned MG 1.640769 s / 39 iterations → 2.0284× | `docs/profiler_solver_performance_history.md:261,643` | Historical tag; current refactors may change absolute times |

Two honest restrictions follow from the table:

* Row F is included specifically to avoid a one-sided claim: the native
  BiStabCG is faster at one rank but slower than QUDA at two ranks on that
  window.
* Rows C, D, and G were measured on shared or small lattices where fixed
  overheads dominate. They are not evidence of kernel throughput on large
  volumes.

### Code-change audit for historical rows

`git diff --stat dev103..HEAD -- cpp/kupl/qcu/src cpp/cuda/qcu/src` reports
18 files changed, 4,130 insertions and 318 deletions, including the KUPL
periodic-geometry repair at HEAD (`8847f84`). Rows A and B were measured in
the `dev103` era, so they are reproducible only from that tag, not from
`pyqcu==0.1.1`. For rows E and F, the ledger records `dev106=2c8fccc8`, while
the local tag `dev106` resolves to `0a632f4`; the exact commit is therefore
unverified. Row H's `test27` report exists at tag `test27`; the corresponding
`stab55` documentation path is absent, so `test27` is the primary provenance.

## Local verification: 8×8×16×16, complex64

Local build: `pyqcu==0.1.1`, WSL2, Intel Core i7-13620H (8 logical CPUs, 1
NUMA node), NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, sm_89, driver 617.42),
CUDA 12.4, Torch 2.15.0+cu132. Installed library SHA256:
KUPL `793970c6c9903cf4fdb649666604437ef5dd19541dfe9335df486f6041963b4a`,
CUDA `bfd1f8b91be9137c621c4ebb2f30556f07ca9e5b19ad775f0231f9cfe811229e`,
Cython bridge `b00a84f0a3b73e907e93f6983122694bd3d545f19e876ad6720dd99913e119df`.

0.1.1 was metadata-only, 0.1.2 added the CLI self-test, and 0.1.3 hardens it and 0.1.4 makes the CLI MPI-aware and 0.1.5 preserves non-zero-rank failure exit codes. The solver code is
identical to 0.1.0, so the 8x8x16x16 numbers below remain valid; they were
measured on 0.1.1 from a clean PyPI venv. The repository driver with its own
NumPy full-lattice residual reported 2.21e-7 at 0.134 s.

| Backend | Device | Iterations | True residual | Acceptance | Solve wall | Notes |
|---|---|---|---|---|---|---|
| KUPL | i7-13620H, 1 rank × 4 OpenMP threads | 26 | 2.28e-7 | 2.44e-4 | 0.127 s | Clean PyPI venv; reinstall 0.143 s |
| CUDA | RTX 4060 Laptop, sm_89 | 26 | 2.96e-7 | 2.44e-4 | 0.039 s | Clean PyPI venv; one process, one rank, one device |

The installed package was verified from outside the source tree: the KUPL
report pointed at `site-packages/pyqcu/kupl/libqcu.so`, the CUDA bridge
imported from `site-packages`, and both solves passed the same acceptance
threshold. The CUDA whole-solve wall was 2.6–3.0× shorter than KUPL on this
machine; this compares different hardware paths and is not a kernel-only
ratio.

The command and acceptance criteria are:

```bash
python pyqcu/testing/simple/kupl_bistabcg.py \
  --lat 8 8 16 16 --dtype c64 \
  --mass 0.17 --csw 1.0 --atol 1e-6 --max-iter 1000 --seed 42 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --output data/release_8x8x16x16/solution_kupl_8x8x16x16_c64.h5 \
  --report data/release_8x8x16x16/report_kupl.json
```

Acceptance: `status == "PASS"`, `final_true_residual <= max(atol, 2048*eps32)`
≈ `2.4414e-4`, and `0 < iterations <= max_iter`. The report's `wall_seconds`
is the solve wall time; the independent residual is computed afterwards in
NumPy.

## What runs where

| Input | Execution path | Notes |
|---|---|---|
| Torch CPU tensor | `pyqcu.torch` operators | Always available |
| Torch CUDA tensor | `pyqcu.torch` CUDA paths, or `pyqcu.cuda` when the bridge is built | The bridge needs a matching `libqcu.so` |
| NumPy host array | `pyqcu.kupl.qcu` | Requires the KUPL library and exact ABI layout |
| No native library | Torch backend only | `python -m pyqcu` reports the missing backend |

## Memory

For `8×8×16×16` complex64 (`16384` sites):

| Object | Elements | Bytes |
|---|---|---|
| Gauge field `[3,3,4,X,Y,Z,T]` | 589,824 | 4.72 MB |
| Fermion field `[4,3,X,Y,Z,T]` | 196,608 | 1.57 MB |
| One parity Clover term `[4,3,4,3,X,Y,Z,T/2]` | 1,179,648 | 9.44 MB |

The KUPL single-rank driver keeps four Clover tensors, so its persistent
working set is about 52 MB plus a transient complex128 copy for the independent
residual (about 33 MB). The CUDA device working set is the same order; the
CUDA context and Torch allocator dominate on such a small lattice.

## Correctness

The native KUPL path is covered by the source-checkout test
`pyqcu/testing/kupl/test_kupl_python.py`:

* 40 exported symbols with C ABI signatures;
* Wilson dslash against a Torch reference at `rtol=2e-4`, `atol=2e-5`;
* Clover dslash at `rtol=5e-4`, `atol=5e-5`;
* Clover BiStabCG with an independent full residual below `2.0e-4`;
* `set_ptrs` slot lifecycle after `applyEndQcu`.

The `8×8×16×16` release case adds a full-lattice NumPy reconstruction of the
Wilson-Clover operator, so the reported residual does not reuse the native
solver's internal recurrence.

## Reproduce

```bash
# 1) build the native KUPL library
bash ./install.sh kupl

# 2) generate the release gauge field
python pyqcu/testing/simple/give_gauss_gauge.py \
  --lat 8 8 16 16 --seed 42 --sigma 0.1 --backend torch \
  --output data/release_8x8x16x16/gauge_8x8x16x16_c64.h5

# 3) KUPL single-rank solve with independent residual
python pyqcu/testing/simple/kupl_bistabcg.py \
  --lat 8 8 16 16 --dtype c64 \
  --mass 0.17 --csw 1.0 --atol 1e-6 --max-iter 1000 --seed 42 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --report data/release_8x8x16x16/report_kupl.json

# 4) CUDA solve on a machine with the CUDA bridge
python pyqcu/testing/simple/qcu_bistabcg.py \
  --lat 8 8 16 16 --mass 0.17 --csw 1.0 --warmup 0 --repeats 1 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --device cuda:0 \
  --report data/release_8x8x16x16/report_cuda.json
```

## Limitations

* The historical reports were produced on hardware and checkouts that are not
  available here. Their absolute seconds are not a specification.
* The P100/V100 row includes library loading; it is not a kernel-only timing.
* The CPU and GPU rows answer different questions and are not combined.
* A native library built for a different architecture may import but fail at
  the first call; record the library path and SHA256 in any new measurement.
