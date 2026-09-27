# 2026-09-27 MultiGrid optimization evidence

This directory contains the compact evidence used by
`docs/report_multigrid_optimized_20260927.tex`.

- `overlap_ab.{json,pdf,svg}`: two counter-ordered P100 large-c64 MG-2
  comparison runs (five steady samples each) with
  `PYQCU_MPI_OVERLAP=0/1`.
- `final_protocol/overlap_matrix.{json,csv,pdf,svg}`: final exact-configuration
  comparison for all 72 PyQCU/QUDA units, with zero config/input mismatches.
- `final_protocol/audit.json` and `combined/`: 72 combined documents, all
  `fair=true`; `units.csv`, `stages.csv`, and `references.csv` retain total
  solve, six-category MG stage, and 1 cold/2 warmup/5 steady reference data.
- `final_protocol/report/`: automated level-time, iteration, residual, and
  speedup tables/figures generated from the 72 combined documents.
- `final_protocol/trace_evidence.json`: fail-closed trace/reference-phase
  validation for all 144 side documents.
- `strict_mpi_overlap_off.final.log` and
  `strict_mpi_overlap_on.final.log`: independent two-rank true-residual
  probes on the final library.
- The direct large-c64 QUDA tuning reruns are recorded at
  `logs/data/dev87_quda_large_c64_g2_c03_sm60.json` and
  `logs/data/dev87_quda_large_c64_g3_c03_sm60.json`.
- `direct_large_comparison.json` summarizes the final large-c64 MG-2/MG-3
  ratios against those direct QUDA runs.

Every final-protocol unit completed under the 600-second per-unit budget.
The final shared protocol is c64 coarse-tol `0.003`, c128 coarse-tol
`3e-5`, and reference phases `1 cold + 2 warmup + 5 steady`; P100 and
V100 c64 use `nu=1`, V100 c128 uses `nu=2`. Historical c03 records remain
as A/B diagnostics only and are not merged into the final matrix.
