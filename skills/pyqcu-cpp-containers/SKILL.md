---
name: pyqcu-cpp-containers
description: 当任务涉及 cpp/ 下 CUDA、KUPL、CANN、DTK、MACA 后端容器结构或构建入口时使用。
---
# cpp/ — 多后端容器目录

`cpp/` 下的目录镜像同一结构：容器层只做后端分组，真实现（或 stub）在各自的
`qcu/` 子目录。新增后端时按同一形状建 `cpp/<backend>/{AGENTS.md,qcu/}`。

CUDA/DCU/GPGPU 生产运行遵循一 Python 进程 = 一 MPI rank = 一设备；容器层
不得把单进程多设备 worker 作为默认后端拓扑。`MultiGpuMultigrid` 是显式调用的
测试专用例外，规则见
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology)。

| 容器 | 状态 | 内容 |
|---|---|---|
| `cpp/cuda/` | **真实现** | `qcu/` 是 PyQCU 主 C++ CUDA 后端，构建产物 `libqcu.so` |
| `cpp/kupl/` | **真实现** | `qcu/` 是独立 MPI+OpenMP CPU 后端；Python 经 `pyqcu.kupl` 使用 |
| `cpp/cann/` | 占位 | `qcu/PASS` 空文件，无活跃昇腾 CANN C++ 后端 |
| `cpp/dtk/` | 占位 | `qcu/PASS` 空文件，无活跃 DCU/ROCm(HIP) C++ 后端 |
| `cpp/maca/` | 占位 | `qcu/PASS` 空文件，无活跃 Maca C++ 后端 |

`cpp/cuda/` 目前不放共享工具或顶层 `CMakeLists.txt`——构建入口仍由
`cpp/cuda/qcu/{env.sh,make.sh,CMakeLists-nv.txt}` 自持；容器目录存在的意义是保持
后端结构对称，便于后续 CANN/DTK/Maca 后端接入时路径不变。

## 子目录技能

各子目录维护自己的现行技能文档；细节以那些文档为准，本文**不复制**其内容（此前
每个容器 skill 内嵌全部子目录正文，2026-10-15 整理时改为链接，避免多处副本漂移）：

| 子目录 | 技能 |
|---|---|
| `cpp/cuda/qcu/` | `skills/pyqcu-cpp-qcu/SKILL.md` |
| `cpp/cuda/qcu/include/` | `skills/pyqcu-cpp-qcu-include/SKILL.md` |
| `cpp/cuda/qcu/src/` | `skills/pyqcu-cpp-qcu-src/SKILL.md` |
| `cpp/cuda/qcu/python/` | `skills/pyqcu-cpp-qcu-python/SKILL.md` |
| `cpp/kupl/qcu/` | `skills/pyqcu-kupl/SKILL.md` |
| ~~`cpp/cuda/qcu/logs/`~~（2026-10-15 已删除） | `skills/pyqcu-cpp-qcu-logs/SKILL.md`（归档说明） |

## cpp/cuda/qcu 速查（详规见 `pyqcu-cpp-qcu`）

- **构建**：`source ./env.sh && bash ./make.sh`（`make.sh` 把 `CMakeLists-nv.txt`
  软链为 `CMakeLists.txt` 后 cmake+make）；仓库根 `bash ./build.sh` 是同一流程的包装。
  `env.sh` 自动探测当前 NVIDIA GPU 的 compute capability，CMake 生成对应
  cubin + PTX；需要复现或跨机型构建时用 `QCU_CUDA_ARCHITECTURES` 显式覆盖，
  并把架构写进 provenance。
- **三数组 ABI**：`params`(int32[58]) / `argv`(real[8]，c64 为 float32、c128 为
  float64) / `set_ptrs`(int64[100])；`_SET_PLAN_`(params[16]) 选内核路径。
- **`_SET_INDEX_` 生命周期**：普通事务为
  `applyInitQcu → operation → applyEndQcu → _SET_INDEX_ += 1`；
  Strict 生命周期内复用同一槽位、不递增。
- **Clover MultiGrid 5 stream 架构**与 **Critical Invariants**（标量只住
  `device_vals`、每轮全流同步、点积走 `_send_tmp_` 槽 7、`mpi_real_type<T>()` 分派、
  `run_mpi` 用阻塞 `MPI_Sendrecv`）见 `pyqcu-cpp-qcu/SKILL.md`。
- **`_BLOCK_SIZE_`**（`include/define.h`）：小格点测试 8/16，NVIDIA 生产 128，
  AMD DCU 生产 256。
