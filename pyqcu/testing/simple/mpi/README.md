# Simple MPI 小格点入口

`pyqcu/testing/simple/mpi/` 是 `simple/` 的 MPI 快速验收入口，用于在小格点上验证：

- PyQUDA/QIO LIME 读取和小体积切片。
- 四维 MPI process grid、rank 坐标和本地格点 geometry。
- KUPL 的 MPI halo、OpenMP 内核和分布式独立真残差。
- CUDA C++ 后端的一 rank 一 GPU 路径。
- 两级 33-tensor 粗算子缓存和 rank-local 超切片。
- `PASS/FAIL` 报告、每 rank CPU affinity 和 GPU 选择。

该目录不用于大格点正式性能发布。默认格点为 `8x8x8x16`，目标是几分钟内完成输入准备、
粗算子构建和小规模求解验证。

## 文档导航

| 文档 | 内容 |
|---|---|
| `README.md` | 快速上手、后端语义、最短命令和产物 |
| [`USER_GUIDE.md`](USER_GUIDE.md) | 完整 MPI 架构、全部参数、OpenMPI 绑定、LIME、缓存、报告和排错 |
| [`../USER_GUIDE.md`](../USER_GUIDE.md) | 父目录 simple 工具的完整说明 |
| [`../COMMAND_REFERENCE.md`](../COMMAND_REFERENCE.md) | 父目录全部命令参考 |

## 两种后端语义

### KUPL：真实 MPI 分域 + OpenMP

- 四维格点按 `--grid` 切分。
- MPI rank 间交换 halo 和做全局归约。
- 每个 rank 内由 OpenMP 执行本地 dslash/solver。
- 分布式 NumPy 真残差使用边界面 `sendrecv` 和 `Allreduce`。

### CUDA：一 MPI rank 一 GPU

- 每个 MPI rank 选择一个 visible GPU。
- C++ CUDA 后端执行四维 MPI halo。
- CUDA worker 通过 `MPI.COMM_TYPE_SHARED` 的 local rank 选择 GPU。
- 这不是 `MultiGpuMultigrid` 的单进程整题复制模式。

`MultiGpuMultigrid` 会在每张 GPU 上求解一个完整问题，适合吞吐/一致性验证；本目录的
CUDA 入口是把同一个格点拆分到多个 rank。

## 最短 KUPL 示例

使用给定的 PyQUDA LIME，取 `8x8x8x16` 前缀子体积：

```bash
source ./env.sh

python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --lat 8 8 8 16 \
  --grid 1 4 1 1 \
  --ranks 4 \
  --omp-threads 8 \
  --levels 2 \
  --dof 12 12 \
  --mg-grid 2 2 2 2
```

启动器的核心行为：

1. 读取 LIME metadata、checksum 和 gauge。
2. 切出 `[0:8,0:8,0:8,0:16]`。
3. 生成同一 seed 的 PCG64 source。
4. 构建全局两级 33-tensor 粗算子缓存。
5. 设置 `OMP_NUM_THREADS=8`、`KUPL_OMP_THREADS=8`。
6. 设置 `KUPL_NUMA_BIND=0`，因为 MPI 启动器负责 CPU 绑定。
7. 启动 `mpirun`，rank 0 汇总解和 JSON 报告。

在多 NUMA 主机上，预期绑定为：

```bash
export OMP_NUM_THREADS=8
mpirun \
  --bind-to numa \
  --map-by ppr:1:numa \
  -np 4 \
  python -m pyqcu.testing.simple.mpi.worker ...
```

## 最短 CUDA 示例

两张 GPU、两个 MPI rank：

```bash
source ./env.sh

CUDA_VISIBLE_DEVICES=0,1 \
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend cuda \
  --libqcu sm60 \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 \
  --ranks 2 \
  --omp-threads 1 \
  --levels 2 --dof 12 12 --mg-grid 2 2 2 2
```

`--libqcu auto` 按 logical `cuda:0` 所属物理卡的 compute capability 选择架构库。
混合架构节点若不存在兼容所有卡的库，应显式使用包含兼容 PTX 的 `default` 库，或把
运行限制在同架构 GPU。

## OpenMPI 进程绑定

这是 KUPL 混合并行最容易出错的部分。

### 推荐多 NUMA 模式

```bash
export OMP_NUM_THREADS=8
export KUPL_OMP_THREADS=8
export KUPL_NUMA_BIND=0
unset OMP_PROC_BIND OMP_PLACES

mpirun \
  --bind-to numa \
  --map-by ppr:1:numa \
  -np 4 \
  python -m pyqcu.testing.simple.mpi.worker ...
```

原则：

- 一个 NUMA 域放置一个 MPI rank。
- 每 rank 线程数由 `OMP_NUM_THREADS` 控制。
- MPI 已绑定时，KUPL 库必须 `KUPL_NUMA_BIND=0`。
- 不要在短 OpenMP 并行区频繁重绑，避免设置 `OMP_PROC_BIND/PLACES`。

### OpenMPI 4.1 的 `pe=N` 冲突

下面命令在 OpenMPI 4.1 会报绑定策略冲突：

```text
--bind-to numa --map-by ppr:1:numa:pe=8
```

应改为：

```bash
# 推荐：绑整个 NUMA，线程数由 OMP_NUM_THREADS 控制
mpirun --bind-to numa --map-by ppr:1:numa -np 4 ...

# 或让 OpenMPI 分配 8 个 PE
mpirun --bind-to core --map-by ppr:1:numa:pe=8 -np 4 ...
```

### 单 NUMA 开发机

若本机只有 1 个 NUMA 域却要求 4 个 rank，`ppr:1:numa` 无法映射。启动器会自动回退：

```bash
mpirun --oversubscribe --bind-to none -np 4 ...
```

报告中的：

```json
{"bind_to": "none", "binding_fallback": true}
```

表示这是 oversubscribe 快速验证，不是正确的每 NUMA 一 rank 性能测量。

## LIME 输入

公共入口直接接受 PyQUDA/QIO LIME：

```bash
--lime /absolute/path/reference.lime
```

也支持用 `--gauge` 覆盖为 LIME 或 HDF5：

```bash
--gauge data/gauge_8x8x8x16_c64.h5
```

兼容点：

- 按 LIME magic 识别，不依赖扩展名大小写。
- 支持 `scidac-private-file-xml`/`scidac-file-xml` 以及
  `scidac-private-record-xml`/`scidac-record-xml`。
- 读取 `clovCoeff` 和 `Mass`。
- 支持单/双精度 payload 并转换到请求 dtype。
- 校验 LIME payload 大小和 SciDAC checksum。
- 大于目标格点的输入会自动切前缀块。

子体积的切口不保持原周期边界，只用于代码和 MPI 正确性验证。

## 默认产物

默认工作目录：

```text
data/mpi_simple/<case-tag>/
```

文件：

| 文件 | 内容 |
|---|---|
| `gauge_global_c64.h5` | 切片后的 canonical gauge |
| `source_global_c64.h5` | 共享 source |
| `case_manifest.json` | 输入、metadata、SHA256 和子体积说明 |
| `coarse_*.h5` | 全局两级 33-tensor cache |
| `solution_mpi_c64.h5` | rank 0 汇总的 `[spin,color,x,y,z,t]` 解 |
| `report_mpi.json` | MPI 几何、计时、残差、rank affinity 和 PASS/FAIL |

## 只检查启动命令

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --dry-run \
  --backend kupl \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime
```

`--dry-run` 不写 case、cache、solution 或 report，只打印解析后的 `mpirun` 命令和绑定
策略。

完整参数、报告字段和故障排查见 [`USER_GUIDE.md`](USER_GUIDE.md)。
