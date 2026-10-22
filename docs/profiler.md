# PyQCU Profiler

从仓库根运行单 rank profiler：

```bash
source ./env.sh
cd pyqcu/testing/profiler
mpirun -np 1 python -u conftest.py
```

命令生成 `trace_*.json`，可载入 [Perfetto UI](https://ui.perfetto.dev/) 查看 CPU/CUDA
时间线。典型分析对象包括 `pyqcu/torch/tools/_einsum.py:Eexyzt_exyzt2Exyzt`。

Profiler 仍遵守“一 Python 进程 = 一 MPI rank = 一设备”。多设备 profile 只允许使用
显式测试专用驱动，不作为生产拓扑。
