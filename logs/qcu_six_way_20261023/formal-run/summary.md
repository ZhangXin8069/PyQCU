# QCU six-way raw summary

| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-legacy-fine-only | FAIL | False | fine_only | 0.004281 | 0.018613 | 0.041934 | 0.013664 | 32.0 | 5.962e-01 | 645922816 |
| torch-legacy | PASS | True | legacy_periodic_correction | 2.852016 | 0.000000 | 0.642516 | 0.617253 | 22.0 | 1.649e-07 | 402653184 |
| torch-strict | PASS | True | strict_recursive_vcycle | 4.898475 | 0.021533 | 1.801785 | 1.910403 | 11.0 | 2.759e-07 | 308281344 |
| cuda-legacy | FAIL | True | periodic_coarse_correction | 0.004013 | 0.019919 | 0.038597 | 0.009607 | 12.0 | 5.763e-01 | 618659840 |
| cuda-strict | PASS | True | strict_recursive_vcycle | 5.003079 | 0.014459 | 0.036116 | 0.021484 | 10.0 | 3.348e-07 | 773849088 |
| kupl-legacy | PASS | True | fine_only_after_rejected_correction | 0.013400 | 0.017847 | 0.333073 | 0.263692 | 48.0 | 6.179e-07 | None |
| kupl-strict | PASS | True | strict_recursive_vcycle | 0.014215 | 0.022415 | 0.268976 | 0.269136 | 10.0 | 3.080e-07 | None |
| quda-strict | PASS | True | quda_aligned_multigrid | 0.534179 | - | 0.190070 | 0.130690 | 8.0 | 5.067e-07 | 138412032 |
