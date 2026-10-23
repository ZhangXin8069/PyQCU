# Three-way Clover BiCGStab benchmark

- lattice: `16x16x16x16`
- mass: `0.05`, csw: `1.0`, atol: `1e-06`, seed: `42`
- warmup: 0, timed repeats: 1
- gauge: `/root/PyQCU/data/qcu_16cubed_quick_20261023/case/gauge_16x16x16x16_c64.h5`

| backend | solve (s) | wall (s) | iterations | true residual | vs QUDA |
|---|---:|---:|---:|---:|---:|
| KUPL QCU parity-Schur BiCGStab | 2.460888 | 2.460888 | 68 | - | - |
