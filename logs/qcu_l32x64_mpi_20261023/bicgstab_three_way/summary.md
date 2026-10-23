# Three-way Clover BiCGStab benchmark

- lattice: `32x32x32x64`
- mass: `-0.277`, csw: `1.160920226`, atol: `1e-06`, seed: `42`
- warmup: 1, timed repeats: 3
- gauge: `/root/PyQCU/data/qcu_l32x64_mpi_20261023/case/gauge_32x32x32x64_c64.h5`

| backend | solve (s) | wall (s) | iterations | true residual | vs QUDA |
|---|---:|---:|---:|---:|---:|
| CUDA QCU BiCGStab | 0.276695 | 0.276695 | 18 | 6.183e-07 | 0.975x |
| CUDA QCU MG (fine-only) | 0.334627 | 0.334627 | 20 | 8.257e-07 | 0.806x |
| QUDA Clover BiCGStab | 0.269647 | 0.270275 | 17 | 8.644e-07 | 1.000x |
| KUPL QCU parity-Schur BiCGStab | 17.737555 | 17.737555 | 19 | 3.424e-07 | 0.015x |
