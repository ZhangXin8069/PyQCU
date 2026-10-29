# Simple MPI entry-point mapping

This directory is the flat MPI specialization of `pyqcu/testing/simple/`.
Every public parent entry has a same-named counterpart here; shared
implementation lives in `_entrypoint.py`, `_compare.py`, `_common.py`, and
`worker.py`.

The task checklist contains 26 unique names but describes it as 27 items.  The
parent directory also has two executable benchmark entries.  Both are included
below, so this map covers all 28 public parent entry points without a silent
gap.

## Mapping table

| Parent entry | MPI counterpart | Mode | Backends | Output |
|---|---|---|---|---|
| `benchmark_kupl.py` | `benchmark_kupl.py` | delegated launcher | KUPL | root-atomic JSON report |
| `benchmark_kupl_lime_mpi.py` | `benchmark_kupl_lime_mpi.py` | native MPI | KUPL | root-atomic JSON report |
| `build_full_mg_cache.py` | `build_full_mg_cache.py` | native MPI | CPU/CUDA builder | distributed HDF5 cache and optional JSON report |
| `check_clover.py` | `check_clover.py` | rank0-root | QCU/QUDA layout | rank-0 JSON stdout |
| `check_fermion.py` | `check_fermion.py` | rank0-root | QCU/QUDA layout | rank-0 JSON stdout |
| `check_gauge.py` | `check_gauge.py` | rank0-root | QCU/QUDA layout | rank-0 JSON stdout |
| `compare_bistabcg.py` | `compare_bistabcg.py` | isolated orchestrator | CUDA-QCU, QUDA | root-atomic orchestration JSON |
| `compare_kupl_bistabcg.py` | `compare_kupl_bistabcg.py` | isolated orchestrator | KUPL, QUDA | root-atomic orchestration JSON |
| `compare_kupl_multigrid.py` | `compare_kupl_multigrid.py` | isolated orchestrator | KUPL MG, QUDA BiCGStab | root-atomic orchestration JSON |
| `compare_three_way.py` | `compare_three_way.py` | isolated orchestrator | CUDA-QCU, KUPL, QUDA | root-atomic orchestration JSON |
| `give_gauss_gauge.py` | `give_gauss_gauge.py` | rank0-root | QCU/Torch | root-atomic HDF5 gauge |
| `kupl_bistabcg.py` | `kupl_bistabcg.py` | delegated launcher | KUPL | root-atomic JSON report |
| `kupl_multigrid.py` | `kupl_multigrid.py` | delegated launcher | KUPL MG | root-atomic solution and JSON report |
| `kupl_residual.py` | `kupl_residual.py` | native MPI | KUPL distributed residual | root-atomic synthetic smoke JSON |
| `prepare_case.py` | `prepare_case.py` | rank0-root | backend-neutral | gauge, source, manifest from rank 0 |
| `qcu_bistabcg.py` | `qcu_bistabcg.py` | delegated launcher | CUDA-QCU | root-atomic JSON report |
| `qcu_clover2quda_clover.py` | `qcu_clover2quda_clover.py` | rank0-root | layout conversion | root-atomic `.npz` |
| `qcu_fermion2quda_fermion.py` | `qcu_fermion2quda_fermion.py` | rank0-root | layout conversion | root-atomic `.npy` |
| `qcu_gauge2quda_gauge.py` | `qcu_gauge2quda_gauge.py` | rank0-root | layout conversion | root-atomic `.npy` |
| `qcu_multigrid.py` | `qcu_multigrid.py` | delegated launcher | CUDA-QCU MG | root-atomic solution and JSON report |
| `quda_bistabcg.py` | `quda_bistabcg.py` | delegated launcher | QUDA/PyQUDA | root-atomic JSON report |
| `quda_clover2qcu_clover.py` | `quda_clover2qcu_clover.py` | rank0-root | layout conversion | root-atomic HDF5 |
| `quda_fermion2qcu_fermion.py` | `quda_fermion2qcu_fermion.py` | rank0-root | layout conversion | root-atomic HDF5 |
| `quda_gauge2qcu_gauge.py` | `quda_gauge2qcu_gauge.py` | rank0-root | layout conversion | root-atomic HDF5 |
| `quda_multigrid.py` | `quda_multigrid.py` | delegated launcher | QUDA MG | root-atomic JSON report |
| `slurm_lime_backend_matrix.sh` | `slurm_lime_backend_matrix.sh` | delegated launcher | QCU, QUDA | per-case rank-0 reports under `data/` |
| `solve_case.py` | `solve_case.py` | delegated launcher | CUDA-QCU, KUPL | root-atomic solution and JSON report |
| `test_coarse_cache.py` | `test_coarse_cache.py` | rank-slab | HDF5/NumPy | root-atomic PASS/FAIL JSON |

No entry is mapped to `N/A`: each parent file has a meaningful MPI execution or
orchestration contract.  A rank0-root entry still uses a collective broadcast
of the root exit state, so a root failure makes every rank exit nonzero.

## Common contract

- Rank counts are validated by `mpi_ranks_power_of_two.py`; only positive
  powers of two are accepted.
- A delegated benchmark runs one backend per MPI job.  CUDA uses one Python
  process, one MPI rank, and one visible GPU; incompatible backend libraries
  are never loaded into the same MPI job.
- `--dry-run` and shell `DRY_RUN=1` print the exact mapped command without
  reading lattice data or starting a solve.
- Benchmark environments remove `PYQCU_STRICT_TRACE_FILE`,
  `PYQCU_QUDA_TRACE_FILE`, `QUDA_MG_TRACE`, and `QUDA_MG_TRACE_FILE`; QUDA
  tuning is forced off with `QUDA_ENABLE_TUNING=0`.
- Rank-zero reports and outputs are staged as hidden sibling files and renamed
  atomically after the producer exits.  A failure may publish an explicit
  `FAIL` report, but never publishes a solution as successful.
