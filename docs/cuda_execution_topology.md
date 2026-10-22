# CUDA QCU 生产执行拓扑

本文规定 PyQCU CUDA QCU 后续程序的默认进程、MPI rank 与设备映射。约定适用于
NVIDIA GPU、DCU 以及其他 GPGPU 后端；除非用户明确要求测试多设备例外，新程序
必须按本文执行。

## 1. 强制约定

生产 CUDA QCU 程序保持：

```text
一个 Python 进程
    = 一个 MPI rank
    = 一张 GPU/DCU/GPGPU
```

多张设备之间的并行通过 MPI 多进程和四维格点分域完成。每个进程只负责一个
rank-local 子格点，跨 rank 边界通过 halo 交换，标量内积和停机判据通过全局归约
同步。

推荐结构：

```text
mpirun -np N python application.py ...

Python process 0 / MPI rank 0 / device 0 -> local subdomain 0
Python process 1 / MPI rank 1 / device 1 -> local subdomain 1
...
Python process N-1 / MPI rank N-1 / device N-1 -> local subdomain N-1
```

生产程序必须满足：

- `MPI.COMM_WORLD.Get_size()` 等于参与设备数。
- 每个 rank 在导入 CUDA native 库之前完成设备选择。
- 每个 Python 进程只构造一个当前 rank 的 QCU `LatticeSet`/solver 上下文。
- 不在一个进程内为多张设备启动多个生产求解 worker。
- rank 数超过可见设备数只允许作为正确性冒烟，不得报告为多卡加速。
- CUDA、DCU 或 GPGPU 的 `CUDA_VISIBLE_DEVICES` 或等价设备隔离机制必须在
  Python/PyTorch 和 native 库导入前生效。

## 2. 明确禁止的默认路径

以下做法不得作为新 CUDA 程序的默认实现：

- 一个 Python 进程创建 N 个线程，每个线程驱动一张设备并执行完整生产 solve。
- 在一个进程内通过多个 `CudaSchurOp` 或 `StrictMultigrid` 实例拼接
  一次跨设备分布式 solve。
- 依赖 CUDA current device 是线程局部来隐式实现多设备分解。
- 将设备间 halo、全局归约或 FGMRES 点积留在 Python 线程间自行实现，而不走
  MPI rank 协议。

一个进程内可以继续使用 GPU 内核线程、CUDA streams、CPU OpenMP 或库内部并行，
但这些线程不能各自代表一张参与同一次分布式 solve 的生产设备。

## 3. 测试专用例外：MultiGpuMultigrid

[`MultiGpuMultigrid`](../pyqcu/cuda/_multi_gpu.py) 是显式 opt-in 的测试和基准
驱动，不属于生产执行拓扑。

```text
一个 MPI rank
├─ thread 0 -> device 0 -> 完整 Legacy MG solve
├─ thread 1 -> device 1 -> 完整 Legacy MG solve
└─ thread 2 -> device 2 -> 完整 Legacy MG solve
```

它的用途是：

- 同一问题在多个设备上的结果一致性测试。
- 不同问题的多设备吞吐测试。
- 构建、生命周期、设备隔离和回归诊断。

它不是：

- 一次 solve 的跨设备格点分域。
- 单次求解延迟优化。
- `solve_case.py` 或其他生产入口的默认后端。
- 后续 CUDA 程序可复制的进程/rank/设备模板。

只有调用方显式导入并构造 `MultiGpuMultigrid` 时，该路径才启用。导入
`pyqcu`、`pyqcu.cuda` 或 `pyqcu.solver` 不会自动导入、构造或启动它。

## 4. 与 MPI 多卡的区别

| 维度 | 生产 MPI 多卡 | 测试专用 MultiGpuMultigrid |
|---|---|---|
| 进程/rank/设备 | 一对一 | 单进程单 rank，多线程多设备 |
| 同一次 solve 是否跨设备 | 是 | 否 |
| 是否交换 halo | MPI rank 间交换 | 每个 worker 独立整题，无 halo |
| 主要目标 | 降低单次 solve 时间和扩展格点 | 一致性、吞吐和诊断 |
| 是否用于生产 | 是 | 否，除非明确测试要求 |

## 5. 新 CUDA 程序检查清单

新增或修改 CUDA 驱动时必须检查：

- [ ] Python 进程数与 MPI rank 数一致。
- [ ] 每个 rank 只绑定一张 GPU/DCU/GPGPU。
- [ ] 设备选择发生在 PyTorch CUDA 和 native QCU 库初始化之前。
- [ ] 同一 solve 的格点分域、halo 和全局归约只走 MPI。
- [ ] 默认入口不导入或构造 `MultiGpuMultigrid`。
- [ ] 文档明确标注测试专用路径，不把它描述成生产多卡方案。
- [ ] 若使用 `MultiGpuMultigrid`，任务描述、报告和日志必须写明显式测试目的。

## 6. 相关文档

- [`pyqcu.cuda` 约定](../pyqcu/cuda/AGENTS.md)
- [C++ CUDA 后端约定](../cpp/cuda/qcu/AGENTS.md)
- [Simple MPI 用户指南](../pyqcu/testing/simple/mpi/USER_GUIDE.md)
- [QCU 与 QUDA MultiGrid 算法组成](qcu_quda_multigrid_algorithms.md)
