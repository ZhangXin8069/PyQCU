# L48 two-node test matrix (`pyqcu.test`, tier 1)

> **Status: definitions only - UNVERIFIED numerically.** The matrix below is
> declared, listed and validated by `python -m pyqcu.test`, but **no formal L48
> two-node matrix case has produced a PASS**. Every numerical case carries
> `evidence="UNVERIFIED"` in the manifest. Do not quote a definition as a
> result, and do not turn an `UNVERIFIED` row into a performance claim.
>
> **2026-11-17 22:15 update:** a **KUPL harness** two-node c64 run now has a
> PASS (job `2155536`, `drun -mpi pmix`, 32 ranks, true residual
> `7.12609e-05`). That is not a `pyqcu.test` matrix execution; all formal
> matrix rows below remain `UNVERIFIED`. See
> [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) §0.1.6.
>
> **2026-11-17 23:30 update:** four more two-node **KUPL harness** PASSes are
> recorded - c128 32r (`2155785`), c64 16r (`2155830`), c64 64r (`2155868`)
> and c64 Legacy-MG 16r cache-load (`2155888`). They were launched with
> `CCS_DRUN_RANK_FILE` + `drun -mpi pmix`, not by `pyqcu.test`; the Legacy-MG
> one is cache-load only (coarse correction numerically inactive). The remote
> overlay has **no `--l48-two-node` flag** (`unrecognized arguments`, RC=2),
> so **every formal row below remains `UNVERIFIED`**. See
> [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) §0.1.7.
>
> Scope: the formal port `pyqcu.test` (shipped, stdlib-only at import).
> Declaration: [`pyqcu/test/l48.py`](../pyqcu/test/l48.py).
> Design background: [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md).
> Date: 2026-11-17.

## 1. What this matrix is

The tier-0 matrix answers "is the installed library correct?" on
`8x8x8x16`. This matrix answers the production question: does the same gate
hold on `48x48x48x96`, split over **two compute nodes with 32 MPI ranks**?
`2x2x2x4` is the production rank grid: every rank owns a local
`24x24x24x24` block, which is even on all four axes.

The matrix is **declarative**: `pyqcu.test.l48` imports only the standard
library plus `pyqcu.test.policy`. `--list`, `--mode list` and `--check` can
describe and validate it without torch, KUPL, MPI, h5py or a remote
scheduler.

## 2. Groups

| Group | Contents | Parameters | Evidence |
|---|---|---|---|
| `bicgstab` | `kupl/l48/bicgstab/c64`, `kupl/l48/bicgstab/c128` | BiCGStab (`bistabcg`), levels=1, restart=0; grid `2x2x2x4`, 32 ranks / 2 nodes | **UNVERIFIED** |
| `legacy_mg` | `kupl/l48/legacy-mg/c64`, `kupl/l48/legacy-mg/c128` | Legacy-MG (`mg`), levels=2, restart=5; grid `2x2x2x4`, 32 ranks / 2 nodes | **UNVERIFIED** (multi-rank legacy asset builder is not wired; see §5) |
| `dslash` | existing KUPL harnesses, no formal case | Wilson block-MPI dslash (`c64`/`c128`) and the clover layout check at L48 | **UNVERIFIED** for 32-rank timing; harness exists |
| `input_io` | `kupl/l48/io-residual/c64` | real L48 HDF5 gauge + source, rank-local slabs, independent full-lattice residual | **UNVERIFIED** |
| `grid_variants` | geometry table (§4) | valid and rejected rank grids | DEFINED |

No group contains **Strict-MG**. Strict-MG is not part of the L48 acceptance
path and `pyqcu.test.l48.validate()` rejects a manifest that grows one.

No group contains **CG**. The L48 CG group is retired; see §3.

## 3. The L48 CG group is retired

The L48 CG group no longer exists in the matrix, in L48 planning, in the docs
or in the tests:

* `pyqcu.test.l48.RETIREMENT` records `group="l48/cg"`, `status="RETIRED"`,
  `scope="L48 only"` and the reason (the L48 production solver menu is
  parity-Schur BiCGStab + Legacy-MG; L48 CG has no independent-residual
  evidence).
* `l48.drop_retired(...)` removes CG cases from **every** L48 plan, and
  `Test`/`Test.select` apply it centrally.
* `--solver cg` or `--only .../solve/cg` combined with an L48 lattice is a
  usage error (exit code `2`), not a silent empty selection.
* `--mode list --lattice-preset L48` plans **17** tier-0 cases (19 minus the
  two retired CG cases), all at `48x48x48x96`.

What is **kept**: the shipped tier-0 CG cases
`torch/solve/cg` and `kupl/solve/cg` remain in
[`pyqcu/test/units.py`](../pyqcu/test/units.py), run on the `8x8x8x16` quick
lattice and are still part of the 19-case tier-0 gate. Retiring the L48 group
must not delete tier-0 coverage; `--quick` still selects all 19 cases.

## 4. Rank-grid variants

`48^3 x 96` is even on every axis, but a rank grid also has to divide each
axis and keep the **rank-local** extent even (the parity/EO decomposition is
undefined on an odd local extent).

| Grid | Ranks | Nodes | Local extents | Status |
|---|---:|---:|---|---|
| `2x2x2x4` | 32 | 2 | `24x24x24x24` | **VALID** (production shape) |
| `1x2x2x8` | 32 | 2 | `48x24x24x12` | VALID |
| `1x1x2x16` | 32 | 2 | `48x48x24x6` | VALID |
| `4x2x2x2` | 32 | 2 | `12x24x24x48` | VALID |
| `2x2x4x2` | 32 | 2 | `24x24x12x48` | VALID |
| `2x2x2x2` | 16 | 1 | `24x24x24x24` | VALID (single-node reference) |
| `1x1x1x16` | 16 | 1 | `48x48x48x6` | VALID |
| `2x2x2x8` | 64 | 4 | `24x24x24x12` | VALID |
| `1x1x1x32` | 32 | 2 | `48x48x48x3` | **REJECTED** - local `t` is odd |
| `1x1x1x64` | 64 | 4 | `48x48x48x1` (not integral) | **REJECTED** - `96/64` is not an integer |

The `t-finest` spelling inherited from L24/L32 (`1x1x1xR`) is therefore **not**
safe to copy: `1x1x1x32` fails the even-local-extent rule. Nodes assume 16
ranks per node, the `2x2x2x4` production shape.

## 5. Execution paths and evidence boundaries

| Group | Executable through | Boundary |
|---|---|---|
| `bicgstab` | `pyqcu.test --mode solve --l48-two-node --only kupl/l48/bicgstab/<dtype>` (process-per-rank, 32 ranks) | needs a real 32-rank MPI launch and the L48 input; currently **UNVERIFIED** |
| `input_io` | the same full-mode row, plus `--gauge-file`/`--source-file` pointing at a real L48 HDF5 pair | a missing file is an `ERROR`, never a generated-field fallback |
| `legacy_mg` | definition is the `kupl_multigrid.py` harness (`--lat 48 48 48 96 --levels 2 --restart 5 --grid 2 2 2 4 --ranks 32`) | `pyqcu.test.native` refuses multi-rank Legacy-MG because its coarse assets come from the per-rank serial 33-stencil builder; the formal row is **UNVERIFIED** until the distributed-cache path is wired |
| `dslash` | existing `pyqcu/testing/kupl/test_dslash_block_mpi.py` (Wilson, 32 ranks) and `pyqcu/testing/simple/mpi/check_clover.py` (clover layout) | correctness harnesses; timing is a measurement, never an acceptance gate, and no new solver/kernel is added |

The c64 `bicgstab` production shape has one **harness** PASS via `drun`
(job `2155536`), but it was not launched by `pyqcu.test`; the formal row above
remains `UNVERIFIED` until the same rank geometry and independent residual are
produced through the port.

`--l48-two-node` requires a full mode (`--mode list` is enough) or `--list`, so
a plain run can never execute L48 at the wrong rank count. It cannot be
combined with `--sweep`, because each case declares its own dtype and geometry.

## 6. CLI

```bash
# Describe the matrix (no backend, no scheduler)
python -m pyqcu.test --list --l48-two-node
python -m pyqcu.test --mode list --l48-two-node --only kupl/l48 --json -

# Tier-0 is unchanged: 19 cases, CG included
python -m pyqcu.test --quick --list --l48-two-node

# L48 planning retires CG: 17 tier-0 cases
python -m pyqcu.test --mode list --lattice-preset L48

# Explicit CG on an L48 lattice is a usage error (exit 2)
python -m pyqcu.test --mode list --lattice-preset L48 --solver cg
```

The ledger row of a matrix case carries the declared geometry instead of the
run-level defaults: `lattice=[48,48,48,96]`, `dtype`, `ranks=32`,
`lattice_grid=[2,2,2,4]`, `nodes=2`, `mg_levels`, `restart`, `solver`, plus
`l48_group`/`l48_evidence`/`l48_blocked_by`.

## 7. Launcher hook

`pyqcu.test.l48.launch_plan(tag)` returns the `mpirun` command a scheduler
script would run, as **data**:

```python
from pyqcu.test import l48
plan = l48.launch_plan("kupl/l48/bicgstab/c64",
                       gauge_file="data/gauge_l48_c64.h5")
# {'kind': 'mpirun', 'mpi_launch': False, 'nodes': 2, 'ranks': 32,
#  'ranks_per_node': 16, 'grid': [2, 2, 2, 4], 'command': [...], ...}
```

`mpi_launch` is `False` and `scheduler_required` is `False`: the cross-node
launcher (`CCS_ALLOC_FILE` → rankfile, passwordless ssh to compute nodes) is
**not** implemented by this port. The interface exists so a remote batch script
can consume ranks/nodes/grid without re-deriving them from prose; it never
starts a process by itself.

## 8. Validation

```bash
python -m pyqcu.test --check        # includes the l48_matrix self-check
python -m pytest -q pyqcu/testing/python/test_pyqcu_test_l48_matrix.py
```

`l48.validate()` asserts: the five group names and order, no Strict-MG/CG,
BiCGStab levels=1/restart=0, Legacy-MG levels=2/restart=5, `prod(grid) == ranks`,
every valid grid's local extents even, a real-input case, tier 1 and pinned
geometry.

## 9. Remaining gaps

* No formal L48 matrix result exists: all five formal cases are `UNVERIFIED`.
  The only real two-node numerical results are KUPL harness runs (`2155536`
  plus the four variants in §0.1.7 of
  [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md)), not
  `pyqcu.test --l48-two-node` rows; the remote overlay has no such flag.
* The cross-node launcher is a declaration only; `CCS_ALLOC_FILE`/rankfile and
  compute-node ssh are not wired. The verified harness route uses
  `CCS_DRUN_RANK_FILE` + `drun -mpi pmix`; direct compute-node SSH is not a
  usable shell path (`Server accepts key` then `Connection closed`).
* Multi-rank Legacy-MG is refused by the formal native path (serial per-rank
  coarse builder); the distributed harness is the intended execution path.
* L48 c64 HDF5 conversion/`io.describe` read-back is not confirmed for the
  remote files (see [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) §2.4.4).
* The dslash/clover harness commands are documented, not executed; no timing
  claim is made.
