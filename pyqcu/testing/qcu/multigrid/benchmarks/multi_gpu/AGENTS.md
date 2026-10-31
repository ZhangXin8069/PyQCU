# AGENTS.md — .../benchmarks/multi_gpu

CUDA `MultiGpuMultigrid` 的测试专用基准目录。

- 该驱动在一个 Python 进程内用 N 线程绑定 N 张设备，每线程执行完整 Legacy MG；
  它是吞吐/一致性测试，不是生产跨卡分域。
- 生产拓扑固定为一个 Python 进程 = 一个 MPI rank = 一张 GPU/DCU/GPGPU；
  不得把本入口接入生产求解器。
- 每线程必须持有独立的 `params/argv/set_ptrs` 和 `_SET_INDEX_`，并保证设备绑定、
  `applyEndQcu` 与异常清理正确。
- 运行前显式核对可见设备、显存和 P100/V100 代码能力；失败时保留原始证据，
  不修改协议或伪造成功。默认不作为回归门。
