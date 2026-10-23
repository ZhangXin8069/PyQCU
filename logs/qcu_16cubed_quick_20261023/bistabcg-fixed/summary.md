# Three-way Clover BiCGStab benchmark

- lattice: `16x16x16x16`
- mass: `0.05`, csw: `1.0`, atol: `1e-06`, seed: `42`
- warmup: 1, timed repeats: 3
- gauge: `/root/PyQCU/data/qcu_16cubed_quick_20261023/case/gauge_16x16x16x16_c64.h5`

| backend | solve (s) | wall (s) | iterations | true residual | vs QUDA |
|---|---:|---:|---:|---:|---:|
| CUDA QCU BiCGStab | 0.039621 | 0.039621 | 68 | 4.175e-07 | 2.484x |
| CUDA QCU MG (fine-only) | 0.047139 | 0.047139 | 84 | 3.555e-07 | 2.088x |
| QUDA Clover BiCGStab | 0.098406 | 0.099098 | 70 | 4.726e-07 | 1.000x |
