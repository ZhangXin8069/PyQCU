# QCU six-way raw summary

| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-legacy-fine-only | PASS | False | fine_only | 0.004845 | 0.013542 | 0.043101 | 0.012146 | 28.0 | 5.764e-01 | 658505728 |
| torch-legacy | PASS | True | legacy_periodic_correction | 2.868248 | 0.000000 | 0.677576 | 0.641921 | 22.0 | 1.649e-07 | 402653184 |
| torch-strict | PASS | True | strict_recursive_vcycle | 5.183893 | 0.023412 | 1.906401 | 1.748673 | 11.0 | 2.759e-07 | 308281344 |
| cuda-legacy | PASS | True | periodic_coarse_correction | 0.004067 | 0.012014 | 0.040734 | 0.010018 | 12.0 | 5.764e-01 | 648019968 |
| cuda-strict | PASS | True | strict_recursive_vcycle | 5.070966 | 0.014765 | 0.038373 | 0.023369 | 10.0 | - | 807403520 |
| kupl-legacy | PASS | True | fine_only_after_rejected_correction | 0.012640 | 0.022816 | 0.293252 | 0.297478 | 48.0 | 6.203e-07 | None |
| kupl-strict | PASS | True | strict_recursive_vcycle | 0.015300 | 0.040244 | 0.493085 | 0.384420 | 10.0 | 3.383e-07 | None |
| quda-strict | PASS | True | quda_aligned_multigrid | 0.571297 | - | 0.239006 | 0.158439 | 8.0 | 5.067e-07 | 138412032 |
