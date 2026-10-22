# PyQCU Simple 用户入口

`pyqcu/testing/simple/` 提供可直接运行的小型与生产求解示例，覆盖：

- CUDA QCU 后端。
- MPI + OpenMP 的 KUPL CPU 后端。
- QUDA / PyQUDA 参考后端。
- QCU、QDP、QIO/LIME、NumPy/HDF5 之间的格式转换与检查。
- BiStabCG、Clover BiStabCG、legacy Clover Multigrid。
- 单后端求解、双后端数值对照、三方性能测量和缓存构建。

本目录不是库的公共 C++ ABI，也不是完整生产工作流框架。它的目标是让用户能用
最少的前置知识验证输入、运行求解器、检查真残差，并得到可复现的 JSON 报告。

## 文档导航

| 文档 | 内容 |
|---|---|
| `README.md` | 最短上手、常用命令、目录职责和结果位置 |
| [`USER_GUIDE.md`](USER_GUIDE.md) | 完整用户指南：格式、归一化、后端、缓存、报告、限制和排错 |
| [`COMMAND_REFERENCE.md`](COMMAND_REFERENCE.md) | 全部主要脚本的逐项命令参考 |
| [`mpi/README.md`](mpi/README.md) | MPI 小格点快速入口 |
| [`mpi/USER_GUIDE.md`](mpi/USER_GUIDE.md) | MPI 架构、绑定、LIME、缓存、报告和故障排查 |

## 最短上手

所有命令默认从仓库根目录执行：

```bash
source ./env.sh
```

不依赖 CUDA、QUDA 或 KUPL 的自检：

```bash
python pyqcu/testing/simple/check_gauge.py
python pyqcu/testing/simple/check_fermion.py
python pyqcu/testing/simple/check_clover.py
```

检查一个 PyQUDA/QIO LIME 组态：

```bash
python pyqcu/testing/simple/check_gauge.py \
  --input ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime
```

将 LIME 转为 canonical PyQCU HDF5：

```bash
python pyqcu/testing/simple/quda_gauge2qcu_gauge.py \
  --input ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --output data/gauge_24x24x24x64_c64.h5
```

## 推荐生产流程

### 1. 准备共享输入

```bash
python pyqcu/testing/simple/prepare_case.py \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --lat 24 24 24 64 \
  --out-dir data/production_L24
```

该命令生成：

```text
data/production_L24/
├── gauge_24x24x24x64_c64.h5
├── source_24x24x24x64_c64.h5
└── case_manifest.json
```

`case_manifest.json` 记录输入 LIME、mass、csw、seed、shape、文件大小和 SHA256。
如果 `--lat` 小于 LIME 格点，会取从原点开始的前缀子体积。子体积的切口缝不满足
原始周期边界，只能用于尺寸研究、并行扩展和代码验证，不应作为物理论文结果。

### 2. 选择后端执行一次冷求解

CUDA QCU：

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend cuda \
  --lat 24 24 24 64 \
  --device cuda:2 \
  --libqcu auto \
  --out-dir data/production_L24
```

KUPL QCU，8 MPI rank × 2 OpenMP 线程：

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend kupl \
  --lat 24 24 24 64 \
  --kupl-ranks 8 \
  --kupl-threads 2 \
  --kupl-lib default \
  --out-dir data/production_L24
```

QUDA / PyQUDA：

```bash
python pyqcu/testing/simple/solve_case.py \
  --backend quda \
  --lat 24 24 24 64 \
  --device cuda:2 \
  --quda-variant auto \
  --out-dir data/production_L24
```

生产驱动只做一次冷求解。CUDA/QUDA 不会重复 warmup；KUPL 默认计算一次分布式独立
真残差。三个后端使用同一份 HDF5 gauge/source。

### 3. 检查产物

```text
data/production_L24/
├── solution_24x24x24x64_<backend>_c64.h5
├── report_<backend>.json
└── last_solve_<backend>.json
```

- `solution_*.h5` 的 dataset 为 `data`，布局 `[spin,color,x,y,z,t]`。
- `report_<backend>.json` 是后端原始报告，字段随后端不同。
- `last_solve_<backend>.json` 是统一摘要，包含状态、迭代、耗时、真残差、MPI 几何和错误。
- 解文件先写 `.partial`，仅当后端报告 `PASS` 时原子替换为正式文件。

## 快速数值对照

QCU CUDA 与 QUDA BiStabCG 同 RHS 对照：

```bash
python pyqcu/testing/simple/compare_bistabcg.py \
  --gauge ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --lat 8 8 16 16 \
  --qcu-device cuda:0
```

KUPL CPU 与 QUDA BiStabCG 对照：

```bash
python pyqcu/testing/simple/compare_kupl_bistabcg.py \
  --lat 4 4 4 4 \
  --mass 0.17 --csw 0.8 \
  --kupl-threads 2
```

KUPL MG 与 QUDA BiStabCG 对照：

```bash
python pyqcu/testing/simple/compare_kupl_multigrid.py \
  --lat 4 4 4 4 \
  --mass 0.17 --csw 0.8 \
  --levels 2 --dof 12 2 --mg-grid 2 2 2 2 \
  --kupl-threads 2
```

这些对照脚本会自动生成同一 gauge/source、隔离后端进程、比较 mass/csw/归一化、
检查两边 `PASS` 和真实解差，并输出固定结构的 JSON。

## 三方性能对照

```bash
python pyqcu/testing/simple/compare_three_way.py \
  --lat 24 24 24 64 \
  --device cuda:2 \
  --kupl-ranks 8 \
  --kupl-threads 2 \
  --kupl-lib default \
  --warmup 1 \
  --repeats 3
```

默认运行：

- CUDA QCU standalone BiStabCG。
- CUDA QCU fine-only legacy MG。
- KUPL parity-Schur BiStabCG。
- QUDA Clover BiStabCG。

这里的三方表是 Clover BiStabCG 性能对照，不是“完整多层级 MG 等价性证明”。
结果默认写到 `data/benchmark_three_way/`：

```text
summary.json
summary.md
cuda_bicg.json
cuda_mg_fine_only.json
kupl_bicg_r*.json
quda_bicg.json
```

## MPI 小格点快速入口

需要验证四维 MPI 分域、halo、LIME 切体积、粗算子缓存和真残差时，使用：

```bash
python pyqcu/testing/simple/mpi/solve_case.py \
  --backend kupl \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime \
  --lat 8 8 8 16 \
  --grid 1 4 1 1 \
  --ranks 4 \
  --omp-threads 8
```

详见 [`mpi/README.md`](mpi/README.md) 和 [`mpi/USER_GUIDE.md`](mpi/USER_GUIDE.md)。

OpenMPI 4.1 下的推荐混合绑定是每 NUMA 一个 rank：

```bash
export OMP_NUM_THREADS=8
export KUPL_NUMA_BIND=0
mpirun --bind-to numa --map-by ppr:1:numa -np 4 ...
```

`KUPL_NUMA_BIND=0` 表示 CPU 放置由 MPI 负责。`--bind-to numa` 与
`--map-by ...:pe=8` 在 OpenMPI 4.1 中会冲突；完整解释见
[`USER_GUIDE.md`](USER_GUIDE.md) 的 OpenMPI 绑定章节。

## 目录职责

| 路径 | 职责 |
|---|---|
| `prepare_case.py` | LIME/HDF5 输入准备、子体积切片、共享 source、manifest |
| `solve_case.py` | 生产单次冷求解入口 |
| `qcu_bistabcg.py` / `qcu_multigrid.py` | CUDA QCU 直接示例 |
| `kupl_bistabcg.py` / `kupl_multigrid.py` | KUPL CPU 直接示例 |
| `quda_bistabcg.py` / `quda_multigrid.py` | QUDA/PyQUDA 直接示例 |
| `compare_*` | 双后端正确性或三方性能对照 |
| `benchmark_kupl*.py` | KUPL C++ 算子或 LIME MPI setup/solve 测量 |
| `build_full_mg_cache.py` | 构建两级 33-tensor MG 缓存 |
| `qcu_*2quda_*` / `quda_*2qcu_*` | 显式格式转换 |
| `check_*.py` | 输入自检与往返布局验证 |
| `give_gauss_gauge.py` | 生成 Gaussian SU(3) gauge |
| `_common.py` | 路径、LIME、HDF5、布局和残差共享工具 |
| `_runner.py` | 子进程、GPU 架构、KUPL 启动和报告共享工具 |
| `mpi/` | MPI 小格点快速验收入口 |
| `slurm_lime_backend_matrix.sh` | 集群环境下的 QCU/QUDA LIME 矩阵测量 |

## 状态与退出码

后端报告状态通常为：

| 状态 | 含义 |
|---|---|
| `PASS` | 后端完成求解，真残差满足验收，配置可比 |
| `WARN` | 数值完成但存在非致命可比性警告，例如显式 csw 与 gauge 元数据不同 |
| `FAIL` | 非有限结果、未收敛、配置错误、布局错误或后端运行失败 |
| `SKIP` | 直接示例检测到依赖缺失；聚合脚本会把 `SKIP` 视为不可通过 |

聚合入口 `solve_case.py`、`compare_*`、`mpi/solve_case.py` 的退出码：

- `0`：最终状态为 `PASS`。
- 非 `0`：失败、错误、超时、缺失依赖或没有发布报告。

## 必须先读的限制

1. QCU 内部使用 `kappa=1/(2*mass+8)`；`csw` 单独存放在 `argv[7]`。不要把
   QUDA asymmetric normalization 的解直接与 QCU 比较。
2. QUDA/QDP fermion 沿 x 做奇偶压缩；QCU parity 张量沿 t 压缩。禁止直接
   `reshape` 替代转换函数。
3. `solve_case.py` 是单次生产求解；`compare_three_way.py` 才是重复计时。
4. CUDA 与 QUDA 库不应在同一 Python 进程中同时加载。
5. LIME 子体积不是周期格点；它的真残差用于实现正确性，不是物理测量。

完整的输入、报告字段、环境变量、缓存和故障排查见
[`USER_GUIDE.md`](USER_GUIDE.md)，全部脚本参数见
[`COMMAND_REFERENCE.md`](COMMAND_REFERENCE.md)。
