# QCU six-way raw summary

| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-legacy-fine-only | PASS | False | fine_only | 0.004124 | 0.018273 | 0.040452 | 0.008545 | 16.0 | 2.220e-07 | 645922816 |
| torch-legacy | PASS | True | legacy_periodic_correction | 2.840991 | 0.000000 | 0.659930 | 0.707687 | 22.0 | 1.649e-07 | 402653184 |
| torch-strict | PASS | True | strict_recursive_vcycle | 5.316478 | 0.025055 | 1.940403 | 1.804293 | 11.0 | 2.759e-07 | 308281344 |
| cuda-legacy | PASS | True | initial_deflation_only | 0.004542 | 0.018125 | 0.039670 | 0.014634 | 16.0 | 2.208e-07 | 648019968 |
| cuda-strict | PASS | True | strict_recursive_vcycle | 4.857335 | 0.020819 | 0.036077 | 0.024312 | 10.0 | 3.348e-07 | 773849088 |
| kupl-legacy | PASS | True | fine_only_after_rejected_correction | 0.012522 | 0.016418 | 0.299933 | 0.262590 | 48.0 | 6.187e-07 | None |
| kupl-strict | PASS | True | strict_recursive_vcycle | 0.012570 | 0.021445 | 0.286777 | 0.294218 | 10.0 | 2.854e-07 | None |
| quda-strict | PASS | True | quda_aligned_multigrid | 0.548087 | - | 0.199223 | 0.151357 | 8.0 | 5.067e-07 | 138412032 |
