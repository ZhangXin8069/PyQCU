# Solvers

PyQCU has several solver layers. They share the same lattice conventions but
differ in what they own: the Torch solvers run in Python, while the CUDA/KUPL
Krylov drivers call one native whole-solve routine and never iterate through
Python. Multigrid adds a hierarchy, coarse operators, and optional HDF5
assets. Pick the smallest layer that matches the job.

## Choosing a solver

| Need | Recommended entry point | What it owns |
|---|---|---|
| Small solve, prototyping | `pyqcu.torch.solver.bistabcg` | Explicit `matvec`, Python iteration |
| Production Clover, Torch | `pyqcu.solver.Multigrid(backend="torch", variant="legacy", ...)` | Null vectors, 33-stencil coarse operators, V-cycle |
| CUDA/KUPL whole-solve | `pyqcu.solver.Krylov` + `CudaLinearOperator`/`KuplLinearOperator` | One native BiCGStab/CG call |
| Native multigrid | `pyqcu.solver.Multigrid(backend="cuda"/"kupl", ...)` | Lifecycle around a pre-built coarse hierarchy |
| Strict reference | `Multigrid(variant="strict", ...)` or `StrictMultigrid` | Explicit Strict hierarchy, not the default |
| Lowest eigenvalues | `pyqcu.torch.solver.tr_lanczos` | Hermitian Lanczos |

Legacy-MultiGrid is the default and the main development target.
Strict-MultiGrid is selected only by name (`variant="strict"` or
`StrictMultigrid`); do not treat it as an implicit default.

## Common conventions

* `kappa = 1 / (2 * mass + 8)`. The Clover coefficient `csw` is separate; in
  the native ABI it is `argv[7]`.
* Full fields use `[4, 3, X, Y, Z, T]` for fermions and
  `[3, 3, 4, X, Y, Z, T]` for gauge links. Parity-split fields compress the
  **time** axis: `[2, 4, 3, X, Y, Z, T/2]`.
* `ResidualScope` says what a residual measures: `FULL`, `SCHUR`, `NORMAL`,
  or `MATPC`. A `SCHUR` residual is not a full-lattice residual.
* For complex64, the source-checkout acceptance used by the release drivers is
  `max(atol, 2048 * eps32) = 2.4414e-4`. A smaller `atol` does not make the
  single-precision floor smaller.

## Torch iterative solvers

| Function | Signature highlights | Use |
|---|---|---|
| `bistabcg` | `(b, matvec, tol=1e-6, max_iter=1000, x0=None, if_rtol=False, verbose=True, history=None)` | General non-Hermitian Wilson/Clover |
| `bistabcg_history` | `(b, matvec, tol=1e-6, max_iter=2000, if_rtol=False)` | Residual history only |
| `cacg` | `(b, matvec, tol=1e-6, max_iter=1000, n_krylov=8, ...)` | s-step CG for Hermitian positive definite operators |
| `fgmres` | `(b, matvec, tol=1e-6, max_iter=1000, restart=30, precond=None, ...)` | Flexible right-preconditioned GMRES |
| `mr` | `(b, matvec, tol=1e-6, max_iter=1000, matvec_dag=None, omega=1.0, ...)` | Minimal residual smoothing |
| `multishift_cg` | `(b, matvec, shifts, tol=1e-6, max_iter=1000, ...)` | Several masses at once, HPD operator |
| `tr_lanczos` | `(matvec, v0, ncv=32, k=6, tol=1e-8, ...)` | Lowest Hermitian eigenpairs |

A minimal non-Hermitian solve:

```python
import torch
from pyqcu import lattice
from pyqcu.torch import dslash, solver

lat = (4, 4, 4, 4)
U = torch.zeros((3, 3, 4, *lat), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=7, sigma=0.1, verbose=False)
kappa = torch.tensor([0.125])
clover = dslash.make_clover(U=U, kappa=kappa)
src = torch.randn((4, 3, *lat), dtype=torch.complex64)
operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)

x = solver.bistabcg(
    b=src, matvec=operator.matvec, tol=1e-5, max_iter=200, verbose=False)
relative = float(
    torch.linalg.vector_norm(operator.matvec(x) - src)
    / torch.linalg.vector_norm(src))
assert relative < 1e-3, relative
```

## Torch Legacy-MultiGrid

The backend-neutral selector is the recommended entry point. It calls
`setup()` on entry, `close()` on exit, and exposes the backend object through
`.implementation`.

```python
import torch
from pyqcu import lattice
from pyqcu.solver import Multigrid
from pyqcu.torch import dslash

lat = (4, 4, 4, 4)
U = torch.zeros((3, 3, 4, *lat), dtype=torch.complex64)
lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
kappa = torch.tensor([0.125])
clover = dslash.make_clover(U=U, kappa=kappa)
src = torch.randn((4, 3, *lat), dtype=torch.complex64)

with Multigrid(
    variant="legacy", backend="torch",
    dtype_list=[torch.complex64] * 2,
    device_list=[torch.device("cpu")] * 2,
    U=U, clover_term=clover, kappa=kappa,
    max_level=1, min_size=4, mg_grid_size=[2, 2, 2, 2],
    dof_list=[12, 24], tol=1e-5, max_iter=100, num_restart=2,
    verbose=False,
) as mg:
    x = mg.solve(src)

operator = dslash.operator(U=U, clover_term=clover, kappa=kappa, verbose=False)
relative = float(
    torch.linalg.vector_norm(operator.matvec(x) - src)
    / torch.linalg.vector_norm(src))
assert relative < 1e-3, relative
```

The important constructor groups:

| Group | Arguments | Meaning |
|---|---|---|
| Geometry | `max_level`, `min_size`, `mg_grid_size`, `dof_list` | Coarsening depth, smallest allowed local extent, block size, and per-level null-vector counts |
| Solver | `tol`, `max_iter`, `num_restart`, `coarse_tol`, `coarse_max_iter` | Outer stopping tolerance and coarse-solve budget |
| Placement | `dtype_list`, `device_list`, `process_grid` | Per-level dtype/device and the MPI rank grid |
| Assets | `coarse_assets`, `clover_ee_inv`, `clover_oo_inv` | Reuse a 33-stencil cache or the CUDA/KUPL Clover inverses |
| Behaviour | `support_parity`, `adaptive_correction`, `deflate`, `seed`, `verbose` | Parity layout, adaptive correction, deterministic setup, logging |

A `4^4` lattice with `max_level=1` has no coarse transition; it is a useful
smoke test of the lifecycle, not a multigrid speed-up demonstration. Real
coarsening needs a larger lattice and several levels.

### `solve_result` on Torch Legacy

`Multigrid.solve_result()` returns a `SolveResult`, but Torch Legacy does not
expose every field to the compatibility envelope. In the current code,
`status` is `UNKNOWN` and `converged`, `final_residual`, `residual_scope`,
`krylov_iterations`, and `cycle_count` stay `None`. Do not infer them from
configuration. Use the backend report or an independent residual:

```python
# after a Legacy solve
report = mg.implementation.last_solve_report
history = mg.implementation.convergence_history
assert "solve_seconds" in report
assert len(history) > 0
```

## Strict-MultiGrid

Strict is an explicit alternative for comparison, algorithm reference, or a
special fallback. Select it by name:

```python
# docs-skip: requires a complete Strict hierarchy and backend assets
from pyqcu.solver import Multigrid

strict = Multigrid(
    variant="strict", backend="torch",
    # Strict constructor shares the Torch hierarchy arguments; see
    # StrictMultigrid and pyqcu.api.StrictMultigridConfig.
    **strict_options,
)
```

`StrictMultigridConfig` groups the Strict controls:

| Object | Fields | Meaning |
|---|---|---|
| `CoarsenConfig` | `nvec`, `block_size`, `levels` | Block coarsening; `levels >= 2` |
| `MRConfig` | `steps` | Fixed-step MR smoother |
| `CycleConfig` | `kind` (`V`/`W`/`F`/`K`), `pre`, `post` | Cycle shape and smoothing counts |
| `FGMRESControl` | `restart`, `rtol`, `atol`, `max_iter` | Outer right-FGMRES |

Strict setup is more expensive than Legacy. The repository's production
drivers use Legacy by default and reserve Strict for explicit comparisons.

## Native CUDA/KUPL Krylov

The native path is one call into the C++ whole-solve routine. The Python side
only builds the operator and checks the returned evidence. A KUPL example:

```python
import torch
from pyqcu import lattice, tools
from pyqcu.api import BiCGStabControl, ResidualScope
from pyqcu.cuda import QcuRuntime
from pyqcu.solver import Krylov
from pyqcu.torch import dslash

try:
    from pyqcu.kupl import qcu
    from pyqcu.kupl.linear_operator import KuplLinearOperator
except Exception as error:
    qcu = None
    print("KUPL backend unavailable; skipping native example:", error)

assert qcu is None or hasattr(qcu, "applyCloverBistabCgQcu")
if qcu is not None:
    lat = (4, 4, 4, 4)
    compact = (*lat[:3], lat[3] // 2)
    full_gauge = torch.empty((3, 3, 4, *lat), dtype=torch.complex64)
    lattice.generate_gauge_field(
        full_gauge, sigma=0.1, seed=59, verbose=False)
    gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
    generator = torch.Generator().manual_seed(61)
    source = torch.complex(
        torch.randn((2, 4, 3, *compact), generator=generator),
        torch.randn((2, 4, 3, *compact), generator=generator))
    runtime = QcuRuntime.from_lattice(
        lat, mass=0.17, atol=1e-7, max_iter=200,
        device="cpu", plan=1, grid=(1, 1, 1, 1), qcu_module=qcu)
    operator = KuplLinearOperator(
        runtime=runtime, gauge=gauge, parity=1)
    result = Krylov(
        backend="kupl", method="bicgstab", operator=operator,
        control=BiCGStabControl(rtol=1e-6, max_iter=200),
    ).solve_result(source)
    full_source = tools.poooxyzt2oooxyzt(source)
    full_solution = tools.poooxyzt2oooxyzt(result.x)
    residual = dslash.give_wilson(
        full_solution, full_gauge,
        torch.tensor([runtime.kappa]), with_I=True) - full_source
    relative = float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(full_source))
    assert result.converged is True
    assert result.residual_scope is ResidualScope.SCHUR
    assert 0 < result.krylov_iterations < 200
    assert relative < 2e-4, relative
```

For CUDA, replace `pyqcu.kupl.qcu` with `pyqcu.cuda.qcu`,
`KuplLinearOperator` with `CudaLinearOperator`, and `backend="kupl"` with
`backend="cuda"`. The runtime device must be `cuda:N`; the gauge, source, and
Clover assets must be on that device.

Native Krylov limits:

* the native ABI does not expose every `SolveResult` field; unexposed fields
  stay `None`;
* CUDA/KUPL drivers require the operator kind to match the requested Krylov
  method (`CG` needs the normal/HPD Schur operator);
* do not wrap the native solve in a Python iteration loop — that defeats the
  whole-solve path.

## Native multigrid wrappers

`Multigrid(backend="cuda"|"kupl", variant="legacy", ...)` wraps a hierarchy
whose coarse `params/argv/set_ptrs` assets already exist. The wrapper owns
`setup()` and `close()`; it does not build null vectors or Galerkin operators.

```python
# docs-skip: requires pre-built coarse 33-stencil assets in params/argv/set_ptrs
from pyqcu.solver import Multigrid

mg = Multigrid(
    variant="legacy", backend="kupl",  # or backend="cuda"
    params=params, argv=argv, set_ptrs=set_ptrs,
    gauge=gauge,
    clover_even=clover_even, clover_odd=clover_odd,
    clover_even_inv=clover_even_inv, clover_odd_inv=clover_odd_inv,
)
mg.setup()
x = mg.solve(rhs)
mg.close()
```

For a production multi-level run, use the source-checkout reference driver
(not shipped in the PyPI package):

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl --lat 8 8 16 16 --ranks 4 --grid 1 2 2 1 \
  --levels 2 --restart 5 --omp-threads 4 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5
```

## Caches

`LegacyMultigridCache` and `StrictMultigridCache` validate and load HDF5
assets. The identity must match the gauge, precision, lattice, block size, and
variant exactly; a mismatch is a cache miss, not a silent fallback.

```python
# docs-skip: identity and path must match an existing cache
from pyqcu.solver.multigrid import LegacyMultigridCache

cache = LegacyMultigridCache(backend="cuda")
loaded = cache.load(
    path, identity=identity, local_shape=(X, Y, Z, T))
assert loaded.variant == "legacy"
```

## MPI and geometry

* Production CUDA topology is one Python process = one MPI rank = one device.
  Multi-device work inside one process is a test-only driver.
* `process_grid` must multiply to the MPI size; each local extent must be even
  for parity layouts.
* KUPL uses the same rank grid with OpenMP inside each rank. Pin
  `OMP_NUM_THREADS` and `KUPL_OMP_THREADS`, and avoid oversubscription.
* The local time extent in a parity-split field is `T/2`; do not reshape a full
  field directly into that layout.

## Acceptance and troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| `no Legacy levels` | `min_size` larger than the local extent | Lower `min_size` or use a larger local lattice |
| `process_grid ... does not match MPI size` | Wrong rank grid | Match `prod(process_grid)` to `MPI.COMM_WORLD.Get_size()` |
| `coarse_assets ... does not match` | Cache identity/shape mismatch | Rebuild the cache with the same lattice/block/precision |
| `solve_result` fields are `None` | Torch Legacy does not expose them | Use `last_solve_report`, `convergence_history`, or an independent residual |
| Native library missing | KUPL/CUDA was not built | `python -m pyqcu` and the package README |
| Residual above acceptance | Wrong scope or single-precision floor | Check `residual_scope`; c64 floor is `2.4414e-4` |

## Production checklist

Use this sequence when a notebook solve becomes a production run:

1. **Freeze the geometry and precision.** Record `(X, Y, Z, T)`, dtype,
   `mass`/`csw`/`kappa`, and the MPI process grid. `prod(process_grid)` must
   equal the MPI size; every local extent must be even for parity layouts.
2. **Prepare canonical tensors.** Gauge `[3, 3, 4, ...]`, full fermion
   `[4, 3, ...]`, parity-split fermion `[2, 4, 3, ..., T/2]`. Run
   `lattice.check_su3` before the solve.
3. **Pick the smallest layer.** Torch `bistabcg` for prototyping, `Krylov`
   for one native whole solve, and `Multigrid(variant="legacy")` for
   production Clover. Select Strict only by name.
4. **Build or reuse assets.** Native Legacy-MG needs a pre-built coarse
   33-stencil hierarchy. Cache identity covers gauge, precision, lattice,
   block size, and variant; a mismatch is a miss, not a fallback.
5. **Solve once.** Native drivers already own the iteration; do not put a
   Python loop around them.
6. **Verify independently.** Check `SolveResult.residual_scope`; a `SCHUR`
   residual is not a full-lattice residual. Compute a full residual for the
   acceptance decision. The c64 floor is `2.4414e-4`.
7. **Record provenance.** Backend, native library path and SHA256, MPI grid,
   thread counts, iterations, residual, and wall time.

## See also

* {doc}`usage` for operators, smearing, and HDF5.
* {doc}`install` for native build variables.
* {doc}`performance` for measured solves and provenance.
* {doc}`troubleshooting` for solver and backend failures.
