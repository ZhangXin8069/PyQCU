# Simple MPI 完整用户指南

本文档详述 `pyqcu/testing/simple/mpi/` 的架构、输入、参数、OpenMPI 绑定、CUDA
设备选择、粗算子缓存、输出、报告、正确性门限和故障排查。

## 目录

1. 入口与设计目标
2. 文件结构
3. 端到端执行流程
4. 运行前提
5. LIME、HDF5、source 和物理参数
6. 公共启动器参数
7. 4D process grid
8. Multigrid 参数与缓存
9. KUPL 后端
10. CUDA 后端
11. OpenMPI 进程绑定
12. 工作目录与文件发布
13. JSON 报告字段
14. 状态、退出码和真残差
15. 完整命令示例
16. 故障排查
17. 已知限制
18. 验证清单

## 1. 入口与设计目标

生产求解公共入口：

```text
pyqcu/testing/simple/mpi/solve_case.py
```

内部 worker：

```text
pyqcu/testing/simple/mpi/worker.py
```

`worker.py` 的 `--verbose` 只用于内部调试：KUPL 会转发 verbose，CUDA 会把 verbose
写入 QCU params。日常用户应通过 `solve_case.py` 启动，不需要直接传该选项。

`simple/` 的每个公开脚本在本目录都有同名 MPI 特化入口。完整的 source→MPI
模式、后端和输出映射见 [`MAPPING.md`](MAPPING.md)。其中 `check_*`、`prepare_case`
和纯格式转换是 rank0-root；`*_bistabcg`、`quda_multigrid` 委托现有 native MPI
benchmark；`build_full_mg_cache.py` 使用 native MPI 分布式 slab 构建；
`qcu_multigrid`、`kupl_multigrid` 复用 `solve_case.py`/`worker.py`；
`compare_*` 仅作为隔离 MPI job 的串行编排器；`test_coarse_cache.py` 使用
rank-slab。所有入口统一拒绝非 2 次幂 rank 数。

设计目标：

- 从 PyQUDA 常用 LIME 直接启动。
- 默认只跑小格点 `8x8x8x16`。
- 由同一启动器管理输入准备、source、coarse cache、mpirun 和 JSON 汇总。
- KUPL 与 CUDA 都使用真实四维 MPI process grid，而不是每 rank 复制整题。
- CUDA 生产拓扑固定为一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU。
- 求解后用独立 NumPy Wilson-Clover 重构计算相对真残差。
- 让用户可通过 `--dry-run` 检查实际 `mpirun` 命令和绑定。

非目标：

- 不替代大格点生产性能测量。
- 不自动跨节点调度。
- 不实现单进程多 GPU 的整题复制；`MultiGpuMultigrid` 仅是显式调用的测试专用
  例外，不属于生产拓扑。
- 不把 LIME 子体积解释为物理周期组态。

## 2. 文件结构

| 文件 | 职责 |
|---|---|
| `solve_case.py` | 用户入口；LIME 读取、切片、source、coarse cache、mpirun、报告合并 |
| `worker.py` | MPI 子进程入口；KUPL delegate 或 CUDA 分布式 MG |
| `_common.py` | LIME/HDF5、grid、local geometry、coarse hyperslab、layout 转换 |
| `_entrypoint.py` | rank-0 集体退出、原子发布、rank 校验、trace 清理和 delegated launcher |
| `_compare.py` | 后端隔离编排和聚合报告 |
| `check_*.py`、`prepare_case.py`、`*2*` | rank0-root 检查、准备与格式转换 |
| `qcu_*`、`quda_*`、`kupl_*` | 对应后端的 delegated/native MPI 入口 |
| `build_full_mg_cache.py` | native MPI null-vector、overlap stencil 和全局 cache 原子发布 |
| `test_coarse_cache.py` | 多 rank HDF5 slab、shape/dtype/hash 和 fail-closed 校验 |
| `MAPPING.md` | 全部公开父入口的逐项映射 |
| `test_mpi_contract.py` | grid、LIME magic、切片、库选择和 mpirun 命令契约测试 |
| `test_counterparts.py` | 入口覆盖、rank 拒绝、DRY_RUN、trace 和原子发布契约测试 |
| `README.md` | 快速入口和绑定摘要 |
| `USER_GUIDE.md` | 本详细指南 |

## 3. 端到端执行流程

```text
LIME/HDF5
   |
   v
solve_case.py
   |-- 读取 metadata，解析 mass/csw
   |-- 切出 --lat 前缀块
   |-- 生成/读取 source
   |-- 计算 case tag 和 work dir
   |-- levels>1 且 restart!=0: 以相同 ranks/grid 分布式构建 coarse cache
   |-- 构造 mpirun 和 env
   v
mpirun -np N
   |
   v
worker.py
   |-- 校验 grid == N
   |-- 计算 rank coordinate/local lattice/local starts
   |-- KUPL: 调用 kupl_multigrid.py 路径
   |-- CUDA: rank-local gauge/source/clover/MG solve
   |-- 分布式独立 NumPy true residual
   |-- rank 0 gather solution
   |-- rank 0 写 report
   v
solve_case.py
   |-- 检查 worker exit code 和 report
   |-- 合并原始 LIME 与启动命令信息
   v
solution_mpi_*.h5 + report_mpi.json
```

## 4. 运行前提

### 4.1 Python 与 MPI

需要：

- Python >= 3.10。
- `mpi4py`。
- 可执行 `mpirun`。
- NumPy、h5py。
- 对应后端依赖。

从仓库根目录执行：

```bash
source ./env.sh
python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.size)'
mpirun --version
```

### 4.2 KUPL

需要：

- `cpp/kupl/qcu/build/libqcu.so`，或 `PYQCU_KUPL_LIB` 指向其他库。
- OpenMP runtime。
- MPI 库与 `mpi4py` 匹配。

检查：

```bash
test -f cpp/kupl/qcu/build/libqcu.so
```

### 4.3 CUDA

需要：

- `pyqcu.cuda.qcu` Cython 扩展。
- `libqcu.so`。
- 至少一个 visible CUDA GPU。
- 与 GPU 架构匹配的 native library。

检查：

```bash
python - <<'PY'
import torch
from pyqcu.cuda import qcu
print('cuda', torch.cuda.is_available(), torch.cuda.device_count())
print('applyCloverMultigridQcu', hasattr(qcu, 'applyCloverMultigridQcu'))
PY
```

## 5. LIME、HDF5、source 和物理参数

### 5.1 gauge 输入优先级

- `--gauge` 非空：使用该 LIME/HDF5。
- 否则使用 `--lime`。
- 默认 `--lime` 为 `~/data/beta6.20...L24x64_cfg_9950.lime`。

### 5.2 LIME 兼容

`_common.is_lime_file()` 读取文件前 4 字节 magic：

```text
45 67 89 ab
```

因此：

- 扩展名大小写不影响识别。
- 即使文件没有 `.lime` 后缀，只要 magic 正确也会按 LIME 读取。

读取支持：

- `scidac-private-file-xml`
- `scidac-file-xml`
- `scidac-record-xml`
- `scidac-private-record-xml`
- `ildg-binary-data`
- `scidac-checksum`
- `clovCoeff`
- `Mass`

payload 必须：

- `datacount=4`
- `colors=3`
- precision `F` 或 `D`
- 长度严格等于 `X*Y*Z*T*4*3*3*itemsize`

### 5.3 子体积

若 LIME 为 `[24,24,24,64]`，而 `--lat 8 8 8 16`：

```text
gauge = full[:, :, :, 0:8, 0:8, 0:8, 0:16]
```

该前缀块保留原始 link。边界缝不是原格点的周期 seam，因此：

- 真残差可检验求解器和 MPI halo 实现。
- 结果不应作为物理论文数据。
- 不同子体积之间没有周期一致性要求。

### 5.4 source

未提供 `--source` 时，启动器用：

```python
numpy.random.default_rng(seed)
real = rng.normal(...)
imag = rng.normal(...)
source = (real + 1j*imag).astype(complex64/128)
```

shape：

```text
[spin,color,X,Y,Z,T]
```

提供 `--source` 时：

- HDF5 dataset `data` 或 `.npy`。
- 可以大于目标格点，启动器取同一前缀子体积。
- dtype 会转换到 `--dtype`。

### 5.5 mass 和 csw

优先级：

1. CLI `--mass` / `--csw`。
2. LIME/HDF5 metadata 中的 `reference_mass`/`mass` 和 `csw`。
3. common default mass `0.05`、csw `1.160920226`。

内部：

```text
kappa = 1 / (2*mass + 8)
```

Clover onsite：

```text
A = I + T, T_onsite coefficient = -kappa*csw/8
```

### 5.6 dtype

- `--dtype c64`：NumPy/Torch complex64，QCU code 2。
- `--dtype c128`：complex128，QCU code 3。

启动器的 gauge/source case 文件分别带 `_c64` 或 `_c128` 后缀。

## 6. 公共启动器参数

### 6.1 后端与输入

| 参数 | 默认 | 说明 |
|---|---|---|
| `--backend kupl|cuda` | `kupl` | 选择后端 |
| `--lime` | home reference LIME | gauge 输入 |
| `--gauge` | 无 | LIME/HDF5，覆盖 `--lime` |
| `--source` | 自动生成 | source HDF5/NPY |
| `--lat X Y Z T` | `8 8 8 16` | 小验证格点，必须为正且 T 偶数 |
| `--mass` | metadata/default | 质量 |
| `--csw` | metadata/default | Clover 系数 |
| `--dtype c64|c128` | `c64` | 精度 |
| `--seed` | `42` | source seed |

### 6.2 solver/MG

| 参数 | 默认 | 说明 |
|---|---|---|
| `--atol` | `1e-6` | 后端目标容差 |
| `--max-iter` | `1000` | 外层迭代上限 |
| `--levels 1|2` | `2` | hierarchy 层数 |
| `--dof` | `12 12` | 每层 E，数量至少等于 levels |
| `--mg-grid` | `2 2 2 2` | 粗化 block |
| `--restart` | `5` | coarse correction interval；0 关闭 |
| `--coarse-max-iter` | `200` | 粗层迭代上限 |
| `--coarse-tol-factor` | `1e4` | 粗层 atol 倍率 |
| `--nv-iters` | `1` | null-space inverse iteration |

### 6.3 MPI 与绑定

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ranks` | `4` | MPI world size |
| `--grid` | 自动 | process grid，乘积必须等于 ranks |
| `--omp-threads` | `8` | OMP/KUPL 线程数 |
| `--bind-to numa|none` | 自动 | OpenMPI `--bind-to` |
| `--map-by` | 自动 | OpenMPI `--map-by` |
| `--kupl-numa-bind scheduler|library` | `scheduler` | MPI 先绑定或 KUPL 库内绑定 |
| `--kupl-memory-policy preferred|bind` | 无 | library 模式的 NUMA 内存策略 |
| `--kupl-memory-node local|N|local+N` | 无 | library 模式显式/片上内存节点 |
| `--kupl-profile generic|kunpeng|kunpeng-2numa` | `generic` | 严格拓扑预检 |
| `--kupl-hugepage` | 关闭 | 内部 scratch 请求 THP advice |
| `--kupl-topology-report` | 关闭 | 每 rank 输出实际 cpuset/内存策略 |
| `--oversubscribe/--no-oversubscribe` | 开启 | 允许超过 slot |
| `--allow-run-as-root/--no-allow-run-as-root` | root 自动 | OpenMPI root |
| `--mpirun` | `mpirun` | 可替换可执行文件 |

自动策略：

- KUPL，且 `ranks <= local online NUMA count`：`--bind-to numa --map-by ppr:1:numa`。
- 其他 KUPL：`--bind-to none`，`binding_fallback=true`。
- `--kupl-numa-bind library`：强制 `--bind-to none --map-by ppr:1:numa`，库在
  `applyInitQcu` 内选择 NUMA-local CPU 和内存策略。
- CUDA：`--bind-to none`。

鲲鹏/HMPI、片上内存、LTO 和详细证据模板见
[`docs/kupl_kunpeng_guide.md`](../../../../docs/kupl_kunpeng_guide.md)。

### 6.4 CUDA library

| 参数 | 默认 | 说明 |
|---|---|---|
| `--libqcu auto|sm60|sm70|default` | `auto` | native library 目录 |

- `auto`：按 logical `cuda:0` 对应物理卡 compute capability 选择。
- `sm60/sm70`：强制目录。
- `default`：保留调用者环境。

混合 GPU 架构下，`auto` 只依据首张 visible GPU。跨架构作业应确保库同时支持所有卡，
或限制 ranks 到同架构 GPU。

### 6.5 缓存和路径

| 参数 | 默认 | 说明 |
|---|---|---|
| `--build-device cpu|cuda` | `cpu` | coarse cache 构建设备 |
| `--work-dir` | `data/mpi_simple/<case-tag>` | case 输出目录 |
| `--coarse-cache` | work dir 内自动名 | 自定义全局 cache |
| `--rebuild-coarse-cache` | 关闭 | 强制重建 cache |
| `--output` | work dir/solution_mpi_<dtype>.h5 | 解路径 |
| `--report` | work dir/report_mpi.json | 报告路径 |
| `--skip-independent-residual` | 关闭 | 跳过 NumPy 真残差 |
| `--dry-run` | 关闭 | 只打印命令，不写文件 |

## 7. 4D process grid

### 7.1 rank 坐标

使用 C order：

```python
coordinate = np.unravel_index(rank, grid, order="C")
```

例如：

```text
grid = [1,4,1,1]
rank 0 -> [0,0,0,0]
rank 1 -> [0,1,0,0]
rank 2 -> [0,2,0,0]
rank 3 -> [0,3,0,0]
```

### 7.2 local geometry

```text
local[axis] = global[axis] / grid[axis]
start[axis] = coordinate[axis] * local[axis]
```

硬性要求：

- 每个 global extent 必须能被 grid 整除。
- 每个 local extent 必须为正偶数。
- 若使用二层级 MG，每个 local extent 还必须能被 `--mg-grid` 整除。

### 7.3 自动 grid

未给 `--grid` 时，按 y→z→t→x 顺序寻找因子，优先避免切最慢/打包最贵的 x。

`8x8x8x16, ranks=4` 的默认结果：

```text
[1,4,1,1]
```

若无法形成偶数 local extent，启动器报错，要求显式 `--grid`。

## 8. Multigrid 参数与缓存

### 8.1 levels

- `--levels 1`：fine-level solve，无粗网格 correction。
- `--levels 2`：一个 coarse level。
- `--dof` 必须有至少 `levels` 项。

### 8.2 restart

- `--restart 5`：每 5 次外层迭代执行 coarse correction。
- `--restart 0`：关闭 coarse hierarchy，KUPL 使用 optimized parity-Schur BiCGStab；
  CUDA worker 把 `levels` 报告为 1。

### 8.3 cache 构建条件

启动器在以下条件构建 cache：

```text
levels > 1 and restart != 0
```

cache 路径默认包含：

```text
coarse_<dtype>_lv<levels>_E<dof>_mg<grid>_nvi<nv_iters>_s<seed>.h5
```

case tag 已包含原始输入 stat、mass、csw、seed 和 dtype；cache 文件名包含 MG 参数。

### 8.4 cache 数据集

| dataset | 含义 |
|---|---|
| `lonv` | null-vector basis |
| `hnn` | nearest-neighbor hopping tensors |
| `hdg` | diagonal hopping tensors |
| `sit` | sitting/onsite term |

### 8.5 rank-local 切片

worker 不读取完整全局 cache 到每个 rank。其 coarse axes 使用：

| dataset | HDF5 lattice axes |
|---|---|
| `lonv` | `(2,4,6,8)` |
| `hnn` | `(4,5,6,7)` |
| `hdg` | `(5,6,7,8)` |
| `sit` | `(2,3,4,5)` |

coarse starts：

```text
coarse_start[axis] = fine_start[axis] / mg_grid[axis]
coarse_local[axis] = local_lattice[axis] / mg_grid[axis]
```

### 8.6 `--build-device`

- `cpu`：默认，适合 KUPL 和跨架构 GPU 节点。
- `cuda`：用 `cuda:0` 的逻辑 PyTorch 设备构建；需要 PyTorch kernel 支持该 GPU。

如果目标卡是 P100 而系统 PyTorch 不支持 sm_60，优先用 `--build-device cpu`。

## 9. KUPL 后端

### 9.1 执行方式

`worker.py --backend kupl` 将参数转发给成熟的
`pyqcu/testing/simple/kupl_multigrid.py`。

该路径负责：

- 四维 MPI rank halo。
- parity-compressed canonical 转换。
- Clover prepare/inverse。
- legacy recursive MG 或 parity-Schur BiCGStab。
- 分布式 NumPy 真残差。
- rank 0 gather solution。

### 9.2 环境

启动器设置：

```text
OMP_NUM_THREADS=<omp-threads>
KUPL_OMP_THREADS=<omp-threads>
KUPL_NUMA_BIND=0   # --kupl-numa-bind scheduler（默认）
OMP_DYNAMIC=FALSE
```

并清除：

```text
OMP_PROC_BIND
OMP_PLACES
GOMP_CPU_AFFINITY
KMP_AFFINITY
```

原因：

- 默认由 MPI/OpenMPI 负责进程和 CPU 放置，KUPL 不得二次绑定。
- `--kupl-numa-bind library` 时改为 `KUPL_NUMA_BIND=1`；此时 MPI 必须
  `--bind-to none`，并可用 `KUPL_MEMORY_POLICY/NODE` 设置库内分配策略。
- library 模式会在读取 gauge/source 和构建 coarse 资产前执行一次私有
  `applyInitQcu/applyEndQcu` prebind，使输入页在 affinity/memory policy 建立后
  first-touch；耗时单列为 `input_prebind_seconds`。
- KUPL 由大量短并行区组成，OpenMP runtime 重复线程绑定会造成明显性能回退。

### 9.3 library

加载顺序由 `pyqcu.kupl.qcu` 决定：

1. `PYQCU_KUPL_LIB`。
2. `cpp/kupl/qcu/build/libqcu.so`。

MPI 启动器目前没有单独 `--kupl-lib` 参数；需要 LTO 或其他库时设置：

```bash
export PYQCU_KUPL_LIB=/absolute/path/libqcu.so
```

### 9.4 多 rank MG

要求：

- `grid` 乘积等于 rank 数。
- local extent 偶数且能被 `mg-grid` 整除。
- `levels>1, restart!=0` 时存在全局 cache。
- 所有 rank 使用同一 mass、csw、dtype、seed 和 cache。

## 10. CUDA 后端

### 10.1 GPU 选择

生产约束是一个 Python 进程对应一个 MPI rank、对应一张 GPU/DCU/GPGPU。
跨设备协同必须使用 MPI 格点分域；不得在单进程内创建多设备求解 worker。
完整规则见
[`docs/cuda_execution_topology.md`](../../../../docs/cuda_execution_topology.md)。

worker 调用：

```text
MPI.COMM_TYPE_SHARED split -> local rank
PYQCU_MPI_DEVICE_ID = local_rank
device_index = local_rank % torch.cuda.device_count()
```

`pyqcu.cuda.qcu` 的 native 层读取 `PYQCU_MPI_DEVICE_ID` 选择设备。

推荐：

- 设置 `CUDA_VISIBLE_DEVICES` 明确可见卡。
- rank 数不超过 GPU 数。
- 若 rank 数更多，多个 rank 会共享 GPU；这可以用于正确性冒烟，但不是吞吐加速。

### 10.2 distributed solve

CUDA code 0/1/2/3 分别设置本地 lattice、grid、rank、size 和 dtype：

- `params[0:4]`：local physical `[X,Y,Z,T]`。
- `params[5:9]`：4D process grid。
- `params[10:12]`：node rank/size。
- `params[14]`：complex precision code。
- `params[17:51]`：MG hierarchy controls。
- `set_ptrs[30:...]`：coarse 33-tensor pointers。

它调用 `applyCloverMultigridQcu`，由 C++ 后端完成 rank 间 halo。

### 10.3 native library

`--libqcu auto` 使用父目录 `_runner.libqcu_environment()`：

- P100 节点：`data/libqcu-sm60`。
- V100 节点：`data/libqcu-sm70`。
- 显式 variant 强制 A/B。
- `default` 不改变 `LD_LIBRARY_PATH`。

若 library 只含单架构 SASS，混合 P100/V100 作业会失败。使用兼容 PTX 的 default build，
或拆分为同架构作业。

## 11. OpenMPI 进程绑定

### 11.1 推荐生产式多 NUMA 命令

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

每个 NUMA 域一个 rank，每 rank 最多 8 个 OpenMP 线程。实际可用线程仍受 cpuset 和
NUMA 核数限制。

### 11.2 为什么不用 `--map-by ppr:1:numa:pe=8`

OpenMPI 4.1 将 `pe=8` 解释为每进程 8 个 processing elements，并要求相应 binding
策略。与 `--bind-to numa` 同时使用时可能报：

```text
A request for multiple cpus-per-proc was given, but a conflicting binding
policy was specified
```

替代：

```bash
# 推荐：线程数由 OMP_NUM_THREADS 控制
mpirun --bind-to numa --map-by ppr:1:numa -np 4 ...

# 若需要 OpenMPI 分 8 个 PE
mpirun --bind-to core --map-by ppr:1:numa:pe=8 -np 4 ...
```

### 11.3 单 NUMA 或容器

如果：

```text
local online NUMA = 1
ranks = 4
```

`ppr:1:numa` 只能放 1 个 rank。启动器自动回退：

```bash
mpirun --allow-run-as-root --oversubscribe --bind-to none -np 4 ...
```

这适合快速正确性测试，但不是有效 NUMA 性能测量。

### 11.4 调度器已提供 cpuset

如果 Slurm/MPI 已给每 rank 独立 cpuset：

```bash
mpirun --bind-to none -np 4 ...
```

保持：

```bash
KUPL_NUMA_BIND=0
```

不要让 KUPL 再次迁移线程或内存。

### 11.5 验证 affinity

查看每 rank CPU 集：

```bash
mpirun --bind-to numa --map-by ppr:1:numa -np 4 \
  python -c 'import os; print(sorted(os.sched_getaffinity(0)))'
```

期望每 rank 打印对应 NUMA 的不同 CPU 集。报告中的 `ranks[].affinity_cpus` 也记录该值。

## 12. 工作目录与文件发布

### 12.1 默认 case tag

case tag 包含：

- 输入文件 stem。
- lattice。
- mass。
- source seed。
- 输入路径、size、mtime、mass、csw、dtype 的 SHA256 前 12 位。

因此：

- 同一输入和参数复用同一目录。
- 输入文件变化会产生新 tag。
- 不同 mass/csw/seed/dtype 不共用 case。

### 12.2 文件覆盖

启动器会：

- 覆盖 case gauge/source 文件，内容由当前参数确定。
- 删除旧 report 后启动 worker，避免 stale report 冒充成功。
- 复用有效 coarse cache，除非 `--rebuild-coarse-cache`。
- CUDA 解先写 `.partial`，完成 H5 后 `os.replace()` 发布。
- KUPL 解由既有 `kupl_multigrid.py` 在 rank 0 写正式 H5。

### 12.3 自定义路径

```bash
--work-dir data/my_case
--coarse-cache data/my_case/cache.h5
--output data/my_case/solution.h5
--report data/my_case/report.json
```

当 `--coarse-cache` 指向外部 cache 时，启动器只验证 `lonv.shape[0] == dof[1]`；
用户仍负责保证 lattice、mass、csw、mg-grid 和 dtype 与该 cache 一致。

## 13. JSON 报告字段

### 13.1 两种后端共有

| 字段 | 含义 |
|---|---|
| `status` | `PASS` 或 `FAIL` |
| `backend` | `kupl` 或 `cuda` |
| `solver` | `applyCloverMultigridQcu` |
| `lattice` | 全局 `[X,Y,Z,T]` |
| `grid` | process grid |
| `dtype` | c64/c128 |
| `mass`、`kappa`、`csw`、`atol` | 物理/容差 |
| `levels`、`dof`、`mg_grid`、`restart` | MG 配置 |
| `wall_seconds` | 所有 rank 最大 solve wall |
| `residual_seconds` | 所有 rank 最大残差 wall |
| `final_true_residual` | 全局相对真残差，或 `null` |
| `residual_acceptance` | Python 验收门限 |
| `residual_source` | 独立残差或 skip 说明 |
| `converged` | 是否满足验收 |
| `ranks` | 每 rank geometry、affinity 和后端环境 |
| `launcher` | 启动器追加的命令、LIME、manifest 和子体积说明 |

### 13.2 KUPL 额外字段

| 字段 | 含义 |
|---|---|
| `local_lattice`、`local_starts`、`grid_coordinate` | rank 0 geometry |
| `mpi_rank`、`mpi_size` | 当前/全局 rank |
| `builder_seconds` | 各 rank 最大 cache load/build time |
| `coarse_source` | cache 路径或 `unused: restart=0` |
| `deflate` | initial coarse correction |
| `iterations` | restart=0 时后端 native iteration，否则常为 `null` |
| `verify_status` | KUPL MG verify 返回码，0 为通过 |
| `memory` | rank 0 RSS 和 OMP 环境 |

`ranks[]` 还包含：

- `rank`
- `coordinate`
- `local_lattice`
- `local_starts`
- `affinity_cpus`
- `omp_num_threads`
- `kupl_omp_threads`
- `kupl_numa_bind`
- `kupl_memory_policy`：库内 memory policy；未指定时为 `null`
- `kupl_memory_node`：显式内存节点/`local+N`；未指定时为 `null`
- `kupl_require_memory_bind`：memory policy 失败是否必须中止
- `kupl_disable_clover_cache`：是否禁用跨调用 Clover/gauge cache
- `kupl_fused_bicg_dots`：是否启用成对 BiCGStab 点积
- `kupl_hugepage`：THP advice 请求值；未指定时为 `null`
- `pyqcu_kupl_profile`：严格拓扑 profile；generic/未指定时为 `null`

### 13.3 CUDA 额外字段

| 字段 | 含义 |
|---|---|
| `mpi_size` | rank 数 |
| `coarse_cache` | cache 路径 |
| `ranks[].gpu` | logical GPU index |
| `ranks[].gpu_name` | GPU 型号 |
| `ranks[].local_lattice` / `local_starts` | rank geometry |
| `ranks[].affinity_cpus` | MPI绑定后的 CPU 集 |

### 13.4 `launcher` 字段

启动完成后，父进程把以下信息加入报告：

```text
launcher.exit_code
launcher.command
launcher.lime
launcher.input_format
launcher.case_manifest
launcher.small_lattice
launcher.sub_volume_note
```

因此报告可以直接用于复现实验，不需要推断实际 mpirun 参数。

## 14. 状态、退出码和真残差

### 14.1 验收公式

```text
acceptance = max(atol, precision_floor)
```

- c64：`max(atol, 2048*eps32)`，默认约 `2.4414e-4`。
- c128：`max(atol, 128*eps64)`。

### 14.2 状态

- `PASS`：求解完成，true residual 有限且不超过 acceptance。
- `FAIL`：未收敛、非有限、MPI/后端异常或没有 report。

MPI 启动器只把 `PASS` 视为成功。

### 14.3 退出码

| 情况 | 退出码 |
|---|---:|
| `--dry-run` 成功 | 0 |
| worker 与 report 都 PASS | 0 |
| worker 非零 | 非零 |
| worker 无 report | 非零 |
| report `FAIL` | 1 |
| report 被启动器改成 `FAIL` | 1 |

### 14.4 `--skip-independent-residual`

开启后：

- 不执行分布式 NumPy 真残差。
- 报告 `final_true_residual=null`。
- 状态只代表 native solver 完成，不代表独立正确性验证。

该选项只应用于计时或临时调试，不建议用于正确性验收。

## 15. 完整命令示例

### 15.1 KUPL 4 rank × 8 threads

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 4 1 1 \
  --ranks 4 \
  --omp-threads 8 \
  --levels 2 \
  --dof 12 12 \
  --mg-grid 2 2 2 2 \
  --restart 5 \
  --work-dir data/mpi_kupl_smoke
```

### 15.2 单 rank KUPL

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 1 1 1 \
  --ranks 1 \
  --omp-threads 4 \
  --levels 2 --dof 12 2 --mg-grid 2 2 2 2
```

### 15.3 无粗网格 correction

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 --ranks 2 --omp-threads 4 \
  --levels 2 --dof 12 12 \
  --restart 0
```

该路径复用 KUPL optimized parity-Schur BiCGStab，不构建 coarse cache。

### 15.4 c128

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 4 4 4 8 \
  --grid 1 2 1 1 --ranks 2 --omp-threads 2 \
  --dtype c128 \
  --levels 2 --dof 12 2 --mg-grid 2 2 2 2
```

c128 需要后端和 cache 都使用 complex128。

### 15.5 CUDA 2 ranks

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend cuda \
  --libqcu auto \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 \
  --ranks 2 --omp-threads 1 \
  --levels 2 --dof 12 12
```

### 15.6 复用 cache

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend cuda \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 --ranks 2 \
  --coarse-cache data/mpi_kupl_smoke/coarse_c64_lv2_E12_12_mg2x2x2x2_nvi1_s42.h5
```

### 15.7 强制重建 cache

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --rebuild-coarse-cache \
  --work-dir data/mpi_rebuild
```

### 15.8 Dry run

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --dry-run --backend kupl --ranks 4 --omp-threads 8 \
  --lime ~/data/reference.lime
```

## 16. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `--grid product != MPI size` | grid 与 `-np` 不一致 | 让 grid 乘积等于 `--ranks` |
| local extent 奇数 | checkerboard 压缩要求 | 改 grid 或 global lattice |
| local lattice 不能被 mg-grid 整除 | coarse geometry 不合法 | 改 `--mg-grid` 或 grid |
| `MPI_COMM_WORLD size` 错 | 手动直接运行 worker 但未用 mpirun | 通过 `solve_case.py` 启动 |
| LIME checked mismatch | 文件损坏或不是目标格式 | 先用 `../check_gauge.py` |
| requested lattice 超过 LIME | `--lat` 太大 | 调整为不大于 LIME 的偶数格点 |
| OpenMPI `pe=N` binding conflict | numa 和 pe 策略冲突 | `ppr:1:numa` 或 `--bind-to core` |
| `ppr:1:numa` 进程过多 | rank > NUMA 域 | `--bind-to none --oversubscribe` |
| 每 rank affinity 都是全部 CPU | 回退到 bind none 或未绑定 | 检查多 NUMA 拓扑和 `--map-by` |
| KUPL 极慢 | 多重 CPU 绑定 | `KUPL_NUMA_BIND=0`，清 OMP bind vars |
| cache E mismatch | `--dof[1]` 与 cache 不同 | 使用匹配 cache 或 rebuild |
| CUDA `no kernel image` | libqcu 架构不匹配 | `--libqcu sm60/sm70/default` |
| CUDA rank 选错卡 | `CUDA_VISIBLE_DEVICES` 映射不一致 | 统一使用 logical device 和可见列表 |
| 多 rank 共享 GPU | ranks > visible GPUs | 明确拒绝；生产路径要求一 rank 一 device |
| report 不存在 | worker 崩溃或旧 report 被删除 | 查看 mpirun stderr 和退出码 |
| `status=FAIL` 但 native residual 很小 | Python 验收 floor/配置不可比 | 检查 acceptance 和 residual source |
| `final_true_residual=null` | 使用了 skip 选项 | 去掉 `--skip-independent-residual` |
| LIME 子体积边界不满足原周期条件 | 原 link 在切口不闭合 | 只用于代码/MPI 正确性验证 |

## 17. 已知限制

1. 只支持两级 MG。
2. 不支持层间改变 process grid 或 rank agglomeration。
3. LIME 子体积是固定原点前缀块，不支持任意 offset。
4. 启动器会自动构建 cache，但大格点构建可能耗时很长。
5. 外部 `--coarse-cache` 只验证 E 维度，完整身份由用户负责。
6. CUDA `auto` 只依据首张 visible GPU，不保证混合架构。
7. KUPL 绑定策略依赖本机 `/sys/devices/system/node/online`；容器看不到拓扑时会回退。
8. 当前不提供跨节点专用 launcher；多节点需要站点 MPI/Slurm 参数。
9. 独立残差用 NumPy 重构，适合小格点；大格点应使用父目录分布式 production 路径。
10. 报告不承诺给性能加速结论；它首先是一个正确性门。

## 18. 验证清单

### 18.1 文档/接口改动

```bash
source ./env.sh
python -m compileall -q pyqcu/testing/simple/mpi
python -m pytest -q -p no:cacheprovider \
  pyqcu/testing/simple/mpi/test_mpi_contract.py \
  pyqcu/testing/simple/mpi/test_counterparts.py
python pyqcu/testing/simple/mpi/solve_case.py --help
python -m pyqcu.testing.simple.mpi.worker --help
git diff --check
bash skills/pyqcu-form/form-audit.sh
```

### 18.2 KUPL smoke

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 --ranks 2 --omp-threads 2 \
  --levels 2 --dof 12 12
```

验收：

- 进程返回 0。
- `report_mpi.json` 的 `status=PASS`。
- `final_true_residual <= residual_acceptance`。
- `ranks` 数量和 rank coordinate 正确。

### 18.3 CUDA smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend cuda --libqcu auto \
  --lime ~/data/reference.lime \
  --lat 8 8 8 16 \
  --grid 1 2 1 1 --ranks 2 --omp-threads 1 \
  --levels 2 --dof 12 12
```

验收：

- 两 rank 都进入不同 logical GPU。
- 每 rank local starts 不同。
- 两项 independent residual 路径均报告 `PASS`。

设备不满足时明确记录“未验证”，不能用 KUPL 结果替代 CUDA smoke。
