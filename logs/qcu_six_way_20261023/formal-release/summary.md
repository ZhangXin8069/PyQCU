# QCU six-way raw summary

| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-legacy-fine-only | PASS | False | fine_only | 0.004511 | 0.012697 | 0.035618 | 0.008500 | 16.0 | 2.220e-07 | 622854144 |
| torch-legacy | PASS | True | legacy_periodic_correction | 2.644805 | 0.000000 | 0.613778 | 0.607062 | 22.0 | 1.649e-07 | 402653184 |
| torch-strict | PASS | True | strict_recursive_vcycle | 4.843557 | 0.022790 | 1.803473 | 1.670798 | 11.0 | 2.759e-07 | 308281344 |
| cuda-legacy | PASS | True | initial_deflation_only | 0.003782 | 0.013491 | 0.041019 | 0.009302 | 16.0 | 2.208e-07 | 648019968 |
| cuda-strict | PASS | True | strict_recursive_vcycle | 4.967338 | 0.013578 | 0.038042 | 0.021693 | 10.0 | 3.348e-07 | 782237696 |
| kupl-legacy | PASS | True | fine_only_after_rejected_correction | 0.013240 | 0.015308 | 0.260122 | 0.256530 | 48.0 | 6.199e-07 | None |
| kupl-strict | PASS | True | strict_recursive_vcycle | 0.013145 | 0.022495 | 0.309197 | 0.268440 | 10.0 | 2.930e-07 | None |
| quda-strict | PASS | True | quda_aligned_multigrid | 0.535602 | - | 0.197704 | 0.144755 | 8.0 | 5.067e-07 | 138412032 |
