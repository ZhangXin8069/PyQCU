# Three-way Clover BiCGStab benchmark

- lattice: `16x16x16x16`
- mass: `0.05`, csw: `1.0`, atol: `1e-06`, seed: `42`
- warmup: 1, timed repeats: 3
- gauge: `/root/PyQCU/data/qcu_16cubed_quick_20261023/case/gauge_16x16x16x16_c64.h5`

| backend | solve (s) | wall (s) | iterations | true residual | vs QUDA |
|---|---:|---:|---:|---:|---:|
| CUDA QCU BiCGStab | 0.044095 | 0.044095 | 68 | 4.175e-07 | 2.402x |
| CUDA QCU MG (fine-only) | 0.045592 | 0.045592 | 82 | 5.333e-07 | 2.323x |
| QUDA Clover BiCGStab | 0.105915 | 0.106621 | 70 | 4.726e-07 | 1.000x |
| KUPL QCU parity-Schur BiCGStab | 1.730579 | 1.730579 | 68 | 3.276e-07 | 0.061x |
