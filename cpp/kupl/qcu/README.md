# KUPL QCU

Independent C++17 CPU backend for the PyQCU/QCU flat ABI. The implementation
uses a four-dimensional MPI process grid, rank-local OpenMP parallelism,
non-blocking fermion halos, cached gauge halos, and fused 12x12 onsite
operations with a 6x6 spin-block fast path for cached Clover inverses.

## Source organization

The backend follows the CUDA QCU layering while retaining CPU/OpenMP
specializations:

```text
include/kupl/       public ABI, geometry, CPU runtime, and operator declarations
src/apply_*.cpp     thin C ABI adapters, one file per CUDA-compatible entry family
src/lattice_*.cpp   geometry, halo, Wilson/Clover, solver, Laplacian, Gauss, MG
src/backend.cpp     LatticeSet facade, validation, and scratch ownership
qcu.map             ELF export allow-list for the 37 QCU C symbols
```

`src/backend.cpp` no longer contains the numerical implementation. Internal
identity is `LatticeSet<Real>`, and each implementation translation unit
instantiates both `float` and `double`.

## ABI

- `params`: `int32[58]`
- `argv`: real `[8]`; slots `0..6` remain
  `mass, atol, sigma, MG-level-1..4 tol`
- `argv[7]`: `csw`
- `set_ptrs`: `int64[100]`

The implemented C symbols are declared in `include/kupl/qcu.h`. The shared
library is built as `libqcu.so`, so a future Cython backend can bind the same
names without changing call sites.

## Data layout

All tensors use canonical C-order with the spacetime dimensions last and time
represented by checkerboard-compressed `T/2`:

```text
gauge:  [2, 3, 3, 4, X, Y, Z, T/2]
fermion:[2, 4, 3, X, Y, Z, T/2]
clover: [4, 3, 4, 3, X, Y, Z, T/2]
```

The first gauge and fermion axis is parity, not an additional physical
dimension. A Clover tensor contains one selected parity per call; `params[9]`
selects it. All local `X,Y,Z,T` extents must be even because the checkerboard
representation stores `T/2` slices.

Memory indices are component-major, with `site` the C-order flattened
`(x,y,z,t2)` coordinate:

```text
fermion component c: (parity * 12 + c) * half_sites + site
clover row r, column c: (r * 12 + c) * half_sites + site
```

The MPI fermion halo is an internal face-site-major buffer and never leaks
through the public ABI.

The bare hopping entry points return

```text
D_bare psi = sum_mu [(1-gamma_mu) U_mu psi(x+mu)
                   + (1+gamma_mu) U_mu(x-mu)^dagger psi(x-mu)]
```

The full Wilson-Clover matrix is

```text
M = A - kappa D_bare,  A = I + T,
T = -(kappa csw / 8) sum_{mu<nu} gamma_mu gamma_nu F_{mu nu},
kappa = 1 / (2 mass + 8).
```

`applyCloverQcu` and `applyCloversQcu` write the complete onsite matrix `A`,
not the traceless Clover contribution `T`. `applyCloversQcu` additionally
writes `A^-1`. Consequently, `csw=0` produces exactly `A=I` and `A^-1=I`.

For a target parity `p`, define `q=1-p` and the compact Wilson Schur operator

```text
S_W = I - kappa^2 D_bare,pq D_bare,qp.
```

`applyWilsonBistabCgDslashQcu` applies `S_W`.
`applyWilsonBistabCgQcu` solves the full Wilson system by solving the target
parity Schur system with BiStabCG, then reconstructs

```text
x_q = b_q + kappa D_bare,qp x_p.
```

`applyWilsonCgDslashQcu` applies `S_W^dagger S_W`.
`applyWilsonCgQcu` is a CG solver on that Schur normal equation, not a
BiStabCG rename. Its iteration trajectory therefore differs from both
`applyWilsonBistabCgQcu` and QCU's Schur BiStabCG path, while it solves the
same full Wilson matrix.

The Clover Schur helpers use `A_p x_p - kappa D_bare,pq x_q = b_p`:

```text
prepare:     compact_rhs_p = b_p + kappa D_bare,pq A_q^-1 b_q
reconstruct: x_q = A_q^-1 (b_q + kappa D_bare,qp x_p)
```

Both return `0` on success and a nonzero status on a validation or numerical
failure.

The standalone Laplacian entry uses the 3D color layout
`gauge[3,3,3,X,Y,Z]` and `field[3,X,Y,Z]`, with the color axis first and the
gauge axes `[row,column,direction]`. It applies

```text
L psi(x) = 6 psi(x)
           - sum_mu U_mu(x) psi(x+mu)
           - sum_mu U_mu(x-mu)^dagger psi(x-mu).
```

The three space dimensions use periodic MPI halos. `params[_LAT_T_]` and
`params[_GRID_T_]` are ignored for this operation, and the process grid is the
product of `grid_x`, `grid_y`, and `grid_z`.

Nested compiler options do not use `-ffast-math`, because that option is unsafe
for the complex phase structure used by the Clover matrix.

## Implemented operations

The header exports all 37 CUDA QCU ABI entry names, and all 37 now have real
CPU implementations.

- `applyInitQcu`, `applyEndQcu`
- `applyWilsonDslashQcu` and `testWilsonDslashQcu`
- Wilson Schur BiStabCG solve/dslash and Schur-normal CG solve/dslash
- `applyCloverQcu` (`A=I+T`), `applyCloversQcu` (`A`, `A^-1`)
- `applyCloverDslashQcu`, `testCloverDslashQcu`
- `applyDslashQcu` (applies a supplied target-parity onsite matrix to bare
  hopping)
- even-odd Schur Clover operator, preconditioned BiStabCG, and iteration query
- Clover Schur RHS prepare and full-field reconstruct
- 3D color Laplacian with MPI and OpenMP
- Gaussian SU(3) gauge generation with deterministic Box-Muller coefficients
- Multigrid restrict/prolong and narrow/wide coarse 33-stencil dslash with MPI
- Recursive full-site Clover multigrid with V-cycle, deflation and true
  residual checks
- Strict coarse raw/Yhat, MATPC, prepare/reconstruct, parity R/P, persistent
  hierarchy, recursive V-cycle and right-preconditioned FGMRES
- `verifyCloverMultigridQcu` R/P projection and Galerkin diagnostics

The Clover BiStabCG solver iterates on the selected-parity Schur complement
`A_p - kappa^2 D_pq A_q^-1 D_qp`, then reconstructs the opposite parity.

New Wilson solve entry points compute a true residual after reconstruction.
Convergence uses a relative criterion with the numerical floor
`max(atol, 2048*epsilon)` for c64 and `max(atol, 128*epsilon)` for c128;
this prevents an impossible fp32 target from being reported as converged.

The CPU MG paths currently require homogeneous c64 or c128 assets. Legacy
Clover-MG supports the V-cycle and optional one-time deflation; CUDA-only
mixed-precision, GCR, CA-GCR and BiCGStabL mode bits are rejected rather than
silently downgraded. LIME/HDF5 I/O and GPU execution remain outside this
library.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For a target-node performance build, enable IPO and local ISA tuning explicitly:

```bash
cmake -S . -B build-lto -DCMAKE_BUILD_TYPE=Release \
  -DKUPL_ENABLE_NATIVE=ON -DKUPL_ENABLE_LTO=ON
cmake --build build-lto -j
ctest --test-dir build-lto --output-on-failure
```

Use `-DKUPL_ENABLE_NATIVE=OFF` for a binary intended to run across different ARM64
or x86 hosts.

The 34 C++/Python differential tests check c64/c128, one/two/four MPI ranks,
Wilson/Clover hopping, Clover construction/inversion, Wilson and Clover Schur
operators, Wilson BiStabCG/CG true residuals, Clover prepare/reconstruct,
Gaussian SU(3) generation, multigrid transfer/coarse stencil MPI halos,
strict primitives/V-cycle/FGMRES, recursive legacy Clover-MG and verify,
slot-isolated lifecycles, non-contiguous binding rejection, the 3D Laplacian,
the exact 37-symbol dynamic ABI surface, synthetic dual-NUMA rank binding,
strict Kunpeng profile/rank-consistency rejection, runtime initialization RSS
timing with post-binding input allocation, and explicit
non-convergence/fail-closed behavior.

## Runtime controls

- `KUPL_OMP_THREADS`: requested rank-local thread count. A positive explicit
  request is honored exactly, including non-power-of-two values such as the
  24 cores in one Kunpeng NUMA node. Without it, the OpenMP maximum is clipped
  to the initial process affinity set.
- Leave `OMP_PROC_BIND`/`OMP_PLACES` unset by default. They are optional A/B
  variables for a stable cpuset, not a portable default for short OpenMP regions.
- `KUPL_NUMA_BIND=1`: discover Linux NUMA cpulists, assign each local MPI rank a
  real NUMA-local CPU subset, bind the actual OpenMP team, and apply
  `KUPL_MEMORY_POLICY` to future allocations. It rejects `OMP_PROC_BIND`,
  `OMP_PLACES`, `GOMP_CPU_AFFINITY` and `KMP_AFFINITY`; launch MPI with
  `--bind-to none` in this mode.
- `KUPL_CPU_NODE`: force a specific CPU NUMA node when `KUPL_NUMA_BIND=1`.
- `KUPL_MEMORY_NODE=local|N|local+N`: select the memory-policy node. `local+N`
  supports separately numbered on-chip-memory NUMA nodes without hard-coding a
  machine-specific offset.
- `KUPL_MEMORY_POLICY=preferred|bind`: default `preferred` permits fallback when a
  node is full; `bind` is strict. `KUPL_REQUIRE_MEMORY_BIND=1` makes policy failure
  fatal instead of a logged warning.
- `KUPL_TOPOLOGY_REPORT=1`: print each rank's resolved NUMA node, cpuset, memory
  node/policy and OpenMP thread count.
- `PYQCU_KUPL_PROFILE=generic|kunpeng|kunpeng-2numa`: select strict runtime
  expectations. Kunpeng profiles reject non-Kunpeng hosts, and profile/topology
  controls must be identical on every MPI rank.
- `KUPL_HUGEPAGE=1`: request `MADV_HUGEPAGE` for large internal scratch buffers.
  It is only an advice and does not modify host THP settings.
- `KUPL_DISABLE_CLOVER_CACHE=1`: refresh the gauge halo and target-parity Clover
  inverse on every call. Use this when mutating a gauge buffer in place; the
  default cache assumes gauge content is immutable for the set lifetime.
- `KUPL_FUSED_BICG_DOTS=0`: disable paired `(t,t)`/`(t,s)` reductions for A/B.
- `params[5..8]` must contain a grid whose product equals `MPI_COMM_WORLD`
  size (three dimensions for Laplacian mode). Invalid grids fail during
  initialization; no silent repartition is attempted.

For a two-rank local test:

```bash
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
KUPL_OMP_THREADS=4 KUPL_NUMA_BIND=1 \
mpirun --bind-to none -np 2 ./build/kupl_qcu_tests \
  --lattice 2 2 2 4 --grid 2 1 1 1
```

`KUPL_NUMA_BIND=1` owns rank/thread placement and therefore requires MPI
process binding to be disabled (`--bind-to none` for OpenMPI) and
`OMP_PROC_BIND`/`OMP_PLACES`/`GOMP_CPU_AFFINITY`/`KMP_AFFINITY` to be unset. If the scheduler
already binds each MPI process to its NUMA-local CPU set, leave
`KUPL_NUMA_BIND=0` and set `KUPL_OMP_THREADS`/`OMP_NUM_THREADS` to that set's
size. CPU subsets are assigned by `MPI_COMM_TYPE_SHARED` local
rank; halo peers continue to use the global four-dimensional Cartesian rank,
so the local NUMA numbering does not leak into cross-node topology.

The Kunpeng-specific HMPI, on-chip-memory, page-policy, topology-planner and
evidence workflow is documented in
[`docs/kupl_kunpeng_guide.md`](../../../docs/kupl_kunpeng_guide.md).

For same-node multi-rank runs, prefer scheduler or MPI binding before Python,
NumPy, or application tensors are allocated. `KUPL_NUMA_BIND=1` runs during
`applyInitQcu`, so it can place library scratch and later allocations but does
not migrate pages that callers already faulted in on another NUMA node. The
repository's LIME benchmark and multigrid driver perform a private prebind
lifecycle before loading inputs when library binding is requested.

Solver staging fields and Clover inverse caches are lazy. A Wilson/Clover-only
lifecycle does not allocate full solver scratch, and the Clover inverse replaces
its temporary term buffer in place. Reports must distinguish init allocation,
first-solve allocation and steady scratch.

For a Dslash throughput measurement:

```bash
KUPL_OMP_THREADS=16 \
  ./build/kupl_qcu_bench --lattice 16 16 16 32 --iterations 10

OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
KUPL_BENCH_SHOW_AFFINITY=1 KUPL_OMP_THREADS=8 KUPL_NUMA_BIND=1 \
mpirun --bind-to none -np 2 ./build/kupl_qcu_bench \
  --lattice 8 8 8 16 --grid 2 1 1 1 --iterations 20
```

The benchmark also measures both parity Clover setup calls and the parity Schur
operator. FLOP rates use full-lattice-equivalent models: `1152` FLOP/site for
one-parity Wilson hopping, `1728` for Clover dslash, and `3456` for the Schur
operator. These are documented models, not hardware counters.

Large local volumes use a cross-site SIMD kernel for the periodic interior and
spatial faces; local `half_sites < 65536` automatically retain the scalar site
kernel. For A/B diagnostics, set `KUPL_DISABLE_BLOCK_DSLASH=1`; rectangular
threshold regression can override `KUPL_BLOCK_DSLASH_MIN_HALF_SITES`.

For repeatable median/MAD evidence and the same operator suite:

```bash
PYTHONPATH=. python pyqcu/testing/simple/benchmark_kupl.py \
  --lat 16 16 16 32 --iterations 10 --repeats 5 --threads 16
```
