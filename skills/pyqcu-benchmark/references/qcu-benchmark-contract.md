# QCU benchmark 契约、性能图和实时账本

本文件承接 `SKILL.md` 的跨后端基准公共契约。正式跑批前完整读取；领域算法细节回到
`qcu-multigrid-20261023.md` 和对应后端技能。

## Resource and fairness contract

Identity-safe reuse is mandatory. When a compatible cache, prepared input,
incremental build tree, or completed compatible benchmark record exists, reuse
it instead of rebuilding. A cache hit still validates the manifest, physical
identity, exact geometry/dtype, rank shard, and every tensor digest before
binding. Rebuild only on miss, identity mismatch, corruption, or an explicit
cold-cache/miss request. Report verification/load and solve timing separately.

For a coarse-cache miss, launch the native MPI builder with the benchmark's
requested rank count and process grid. It must construct local null vectors,
compute disjoint coarse slabs with periodic overlap halos, gather and verify
`lonv/hnn/hdg/sit`, and atomically publish the HDF5 plus manifest. A rank-0-only
builder is not valid multi-rank evidence. Preserve per-rank patch geometry,
global tensor SHA256, and the 1-vs-N-rank tensor relative-error comparison.

For distributed strict solves, run the post-solve independent full residual in
a fresh process after MPI teardown. Re-entering distributed dslash while only
one rank remains inside the solve communicator can deadlock in `COMM_WORLD`
busy-wait. A residual collected that way is not a valid sample. Strict 4-rank
geometry also requires even rank-local fine and coarse extents; for an
`8x8x8x16` quick case use an eligible grid such as `2x2x1x1`, not y-only
`1x4x1x1`.

Shared-login benchmarks must record load average, user count, per-rank thread
count and cpuset. A run at `load average≈91` on 48 CPUs is useful for
correctness/capacity but cannot support a strong-scaling or Kunpeng production
claim. Report absolute median, MAD, sample count and parallel efficiency
together; a rank increase that raises wall time is a valid negative result and
must not be replaced by an older favorable window.

QUDA is a benchmark dependency, not the optimization target. If a source
rebuild fails, use the explicitly recorded pre-existing QUDA installation under
`$HOME/lib` and verify its path, SHA256, target architecture, precision,
multigrid instantiation, and QDP capability before running the QUDA side. If
that fallback has no usable MG capability, retain the passing QUDA BiCGStab
reference and mark MG-vs-MG as capability-blocked; continue QCU-MG optimization
and do not substitute a narrow benchmark for the missing comparison.

The formal matrix includes larger lattices and single-node 1/2/4-rank instances.
`24^3x64` is a tuning/smoke volume and cannot support the final performance
claim. The minimum formal large-volume case is `32^3x64`; add `48^3x96` when
capacity and the 10-minute per-job gate allow. Each input keeps its own LIME
mass/CSW metadata unless the physical comparison explicitly uses one target
mass. A 1-rank or small-volume result cannot replace the larger multi-rank
acceptance; every case preserves backend, gauge/RHS, precision, stopping,
residual scope, cache identity, and independent true-residual gates.

## Performance figure and speedup contract

Every formal performance batch must emit a figure set from the same CSV/JSON
records used by the report. Keep the raw data, generation command, and SHA256
next to the figures. Prefer SVG for the report and also write a PNG preview;
figures are derived artifacts and never replace machine-readable evidence.

Minimum figure set:

1. total solve wall time with sample count and spread/error bars;
2. PyQCU/QUDA speedup or QUDA/PyQCU ratio with the direction in the axis label;
3. parallel efficiency for 1/2/4 MPI ranks;
4. Krylov/outer iteration count;
5. independent true residual with tolerance line;
6. stage or cache/setup breakdown, kept separate from steady solve.

Any speedup must be shown together with the numerator and denominator absolute
times, their units, aggregation statistic, and sample count. For example show
`R = QUDA/PyQCU = 0.130358 s / 0.122422 s = 1.065x (median, 5 samples)`, not a
bare `1.065x`. The same rule applies to tables, figure labels, captions, briefs,
and final reports. A ratio without absolute times is not acceptance evidence.

Separate panels by backend, Legacy/Strict variant, lattice, precision, and
rank count when those variables change. Do not combine different timing
scopes, traces, or cache-hit/cold-cache records in one series. Use one unit per
axis, log scales only where the data span requires them, show `1.0x`/tolerance
reference lines, and include hardware, gauge/RHS identity, cache identity,
sample count, error definition, and `R = baseline / candidate` in the title or
caption. Sort categories by the primary metric, preserve stable series colors,
and use labels large enough for projection.

The extension plots must include larger lattices and 1/2/4-rank single-node
instances. A smoke point may appear only in a separately labeled smoke figure;
it cannot be connected to or averaged with formal points.

The formal solver matrix contains both MultiGrid and native BiCGStab. CUDA
BiCGStab must call `applyCloverBistabCgQcu`; QUDA must use its BiCGStab inverter
with the same Clover, mass/CSW, precision, RHS, tolerance, stopping rule and
1/2/4-rank geometry. A `restart=0` MG wrapper may be reported as a sensitivity
point but does not replace the native BiCGStab entry. Report MG-vs-MG and
BiCGStab-vs-BiCGStab as separate comparisons, then show the MG/BiCGStab ratio
within each backend as contextual evidence.

## Required run recipe

Every performance report row and every brief must include the complete run
recipe rather than only a timing number. Required fields are: lattice XYZT,
MPI ranks and process grid, one-process-per-rank/device mapping, backend,
Legacy/Strict variant, precision, mass, kappa, csw, the exact public entry
function and native operation, levels, block/`mg_grid`, dof/nvec, restart,
coarse solver budget/tolerance, outer/inner tolerances, `atol`, `max_iter`,
warmups, repeats, actual iterations/cycles/corrections, residual scope,
independent true residual, cache path/identity/hit result, gauge and RHS
SHA256, device model/UUID, and PyQCU/QUDA library SHA256.

Missing optional diagnostics may be `None`; missing parameters that can change
the algorithm, convergence, or timing make the result non-comparable. Do not
replace the full recipe with phrases such as "same settings" or "default".

## Real-time document ledger

When performance runs are requested to be recorded in real time, update
[`docs/qcu_mg_performance_ledger_20261029.md`](../../../docs/qcu_mg_performance_ledger_20261029.md)
or a dated successor after every completed, failed, timed-out, or cancelled
batch. Do not defer the record until the final report.

The ledger records job/time/status/duration, complete solver recipe, setup/cache/
solve breakdown, samples/MAD, iterations/cycles/corrections, absolute times and
speedup ratio pairs, independent true residual, identities, devices, and raw
evidence paths. Failures and cancellations remain visible and cannot disappear
from the formal sample set silently.

For distributed Strict runtime binding, the coarse `level_specs` shape is local:
`global_lattice/(process_grid*mg_block)`. Keep the cold hierarchy and cached C++ path
on that same shape. The 2-rank `1x2x1x1` point therefore binds `[4,2,4,8]`, not the
global `[4,4,4,8]`. Always use a job-unique trace directory because QUDA/QCU trace
writers may append and are not protected by a path-existence precheck.
