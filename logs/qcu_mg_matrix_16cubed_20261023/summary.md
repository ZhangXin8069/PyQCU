# QCU six-way raw summary

| variant | status | coarse executed | mode | setup s | cache s | first s | steady median s | iterations | true res | peak device B |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-legacy-fine-only | PASS | True | periodic_coarse_correction | 0.004610 | 0.097398 | 0.078324 | 0.051113 | 82.0 | 5.341e-07 | 2399141888 |
| torch-legacy | PASS | True | legacy_periodic_correction | 1.039635 | 0.000000 | 2.585570 | 2.539016 | 79.0 | 7.406e-07 | 2151677952 |
| torch-strict | PASS | True | strict_recursive_vcycle | 37.620282 | 0.123567 | 12.955793 | 12.495584 | 23.0 | 6.577e-07 | 1688207360 |
| cuda-legacy | PASS | True | initial_deflation_plus_periodic_correction | 0.003692 | 0.088381 | 0.083868 | 0.056414 | 87.0 | 3.051e-07 | 2376073216 |
| cuda-strict | PASS | True | strict_recursive_vcycle | 38.827962 | 0.109975 | 0.211919 | 0.194964 | 23.0 | 6.543e-07 | 3644850176 |
| kupl-legacy | PASS | True | periodic_coarse_correction | 0.111246 | 0.056443 | 3.146277 | 4.000842 | 73.0 | 3.362e-07 | None |
| kupl-strict | PASS | True | strict_recursive_vcycle | - | 0.162710 | 13.856721 | 13.910091 | 23.0 | 6.535e-07 | None |
| quda-strict | PASS | True | quda_aligned_multigrid | 0.765986 | - | 0.476498 | 0.404103 | 12.0 | 4.744e-07 | 641728512 |
