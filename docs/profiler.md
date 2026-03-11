cd `PyQCU/examples/profiler`
run with `mpirun -np 1 python -u conftest.py`
get `trace_***.json`, then put it into `https://ui.perfetto.dev` for `profiler`
> such as optimizating `/root/PyQCU/pyqcu/tools/_einsum.py:Eexyzt_exyzt2Exyzt` 