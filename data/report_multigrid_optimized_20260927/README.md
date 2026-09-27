# 2026-09-27 MultiGrid optimization evidence

This directory contains the compact evidence used by
`docs/report_multigrid_optimized_20260927.tex`.

- `overlap_ab.{json,pdf,svg}`: two counter-ordered P100 large-c64 MG-2
  comparison runs (five steady samples each) with
  `PYQCU_MPI_OVERLAP=0/1`.
- `overlap_matrix.{json,csv,pdf,svg}`: exact-configuration comparison for
  48 completed PyQCU/QUDA comparison units.
- `pyqcu_trace_evidence.json` and `quda_trace_evidence.json`: fail-closed
  trace/reference-phase validation.
- `strict_mpi_overlap_off.final.log` and
  `strict_mpi_overlap_on.final.log`: independent two-rank true-residual
  probes on the final library.
- The direct large-c64 QUDA tuning reruns are recorded at
  `data/dev87_quda_large_c64_g2_c03_sm60.json` and
  `data/dev87_quda_large_c64_g3_c03_sm60.json`.
- `direct_large_comparison.json` summarizes the final large-c64 MG-2/MG-3
  ratios against those direct QUDA runs.

Large-lattice QUDA units that exceeded the 600-second collection budget are
not represented as fresh evidence. The report labels historical c03 records
separately and does not merge them into the exact-configuration table.
