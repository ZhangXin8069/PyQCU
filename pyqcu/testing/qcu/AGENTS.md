# AGENTS.md — pyqcu/testing/qcu

C++ CUDA 后端测试（经 Cython 桥）。从 Python 驱动 `libqcu.so`。

CUDA 生产拓扑固定为“一个 Python 进程 = 一个 MPI rank = 一张
GPU/DCU/GPGPU”。`bench_multi_gpu.py`、`conftest.multi_gpu.py` 和
`slurm_multi_gpu_smoke.sh` 使用的 `MultiGpuMultigrid` 是显式测试例外，只验证
多卡一致性或吞吐，不是生产部署模板。

## 入口分类

| 入口 | 选择规则 |
|---|---|
| 顶层 `test_*.py` | 定向 pytest；按文件选择，不用整个 `qcu/` 作为默认门禁 |
| `conftest.*.py` | 手工生命周期/API 脚本，编辑参数后直接执行；不是 pytest 测试模块 |
| `single_qcu_*.py` | 单接口诊断入口；无 CUDA 时运行纯 PyTorch 参考并报告 `SKIP` |
| `strict/quda_comparison/test_*.py` | 按改动选择具体模块或 node；GPU/QUDA 依赖、矩阵编排和报告测试不得盲收 |
| `multigrid/`、`bench_multi_gpu.py`、Slurm 脚本 | 大格、多卡和报告专用入口，不进入日常单元门禁 |

## 定向 pytest

| 文件 | 测试内容 |
|---|---|
| `test_qcu_api_smoke.py` | `pyqcu.h` 符号与 Cython bridge 集合；`QCU_API_REQUIRE_CUDA=1` 时追加最小 init/end 生命周期 |
| `test_csw_reference.py` | CSW metadata/ABI/cache identity、`QcuRuntime` 生命周期、槽位和异常清理 |
| `test_csw_cuda.py` | CUDA Clover CSW 数值与 ABI 控制；CUDA 扩展不可用时 skip |
| `test_nullvec_batch_sizing.py` | null-vector 批大小与显存预算边界 |
| `test_benchmark_qcu_lime_mpi.py` | QCU LIME benchmark 的 parity、RHS、真残差、rank slab 和生命周期契约 |
| `test_simple_lime_io.py` | simple/LIME 路径、状态、布局与 QUDA Dirac 选择契约 |

常用命令：

```bash
source ./env.sh
python -m pytest -q \
  pyqcu/testing/qcu/test_qcu_api_smoke.py \
  pyqcu/testing/qcu/test_csw_reference.py
QCU_API_REQUIRE_CUDA=1 \
  python -m pytest -q pyqcu/testing/qcu/test_qcu_api_smoke.py
python -m pytest --collect-only -q pyqcu/testing/qcu/test_qcu_api_smoke.py
```

整目录还包含可选 Strict/QUDA 依赖和仍在迁移的诊断模块，不能把
`pytest pyqcu/testing/qcu` 当作稳定门禁；需要成组运行时显式列出目标文件。Strict 测试按
`strict/quda_comparison/AGENTS.md` 的模块说明逐文件选择。

## 显式生命周期脚本

| 文件 | 测试内容 |
|---|---|
| `conftest.cuda.py` | CUDA 可用性与上下文 |
| `conftest.mpi.py` | MPI 网格设置、halo 与分布式求解 |
| `conftest.wilson.bistabcg.py` | Wilson BiStabCG 求解器 |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash 内核 |
| `conftest.wilson.cg.py` | Wilson CG 求解器 |
| `conftest.clover.py` | Clover 项构造 |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover legacy multigrid V-cycle |
| `conftest.multigrid.mt.py` | 单进程多线程 legacy MG |
| `conftest.multi_gpu.py` | 测试专用多线程多卡 MG 一致性，不是生产并行路径 |

这些文件通常使用模块级 `params/argv/set_ptrs` 并提供 `main()`，直接执行才会运行。
旧的 `conftest.bistabcg.dslash.py`、`conftest.schur.multigrid.py` 与
`bench_cpp_mg.py` 已被 `single_qcu_*`、现行 MG benchmark 和 `QcuRuntime` 生命周期
测试取代，不再保留重复入口。
单 rank CUDA 示例：

```bash
mpirun -np 1 python pyqcu/testing/qcu/conftest.clover.multigrid.py
python pyqcu/testing/qcu/conftest.multi_gpu.py
```

双卡 Slurm 冒烟（从仓库根目录提交）：

```bash
sbatch --gpus=2 --cpus-per-task=64 --time=01:00:00 \
  --output=logs/slurm-%j-multi-gpu.out \
  --error=logs/slurm-%j-multi-gpu.err \
  pyqcu/testing/qcu/slurm_multi_gpu_smoke.sh
```

纯 MG 吞吐基准应传 `--no-reference`；一致性测试保留默认
`run_reference=True`。稳态总耗时使用 `total_solve_median`，冷启动和逐线程
`mg_time` 分开报告。

## 单功能接口测试（当前维护入口）

`single_qcu_*.py` 每个文件只调用一个功能族，并在 CUDA 可用时执行一次完整的
`applyInitQcu → operation → applyEndQcu → params[_SET_INDEX_]+=1` 生命周期；CUDA 不可用时仍运行
纯 PyTorch 参考和布局契约，明确输出 `SKIP`。`single_qcu_api.py` 对照 `pyqcu.h`、`qcu_api.pxd`、
`qcu.pyx` 的符号集合，Strict/MG 文件额外校验参数形状。设置 `QCU_STRICT_NUMERIC=1` 可把数值偏差
升级为失败，默认模式只报告偏差以便在不同 CUDA 架构上做诊断。

| 文件 | 目标接口 |
|---|---|
| `single_qcu_gauss_gauge.py` | `applyGaussGaugeQcu` |
| `single_qcu_wilson_dslash.py` | `applyWilsonDslashQcu` |
| `single_qcu_clover_dslash.py` | `applyCloverDslashQcu` |
| `single_qcu_wilson_bistabcg.py` / `single_qcu_wilson_cg.py` | Wilson BiStabCG / CG |
| `single_qcu_clover.py` | `applyCloverQcu` / `applyCloversQcu` |
| `single_qcu_clover_bistabcg.py` | Clover BiStabCG、Schur prepare/reconstruct |
| `single_qcu_laplacian.py` | `applyLaplacianQcu` |
| `single_qcu_multigrid_transfer.py` / `single_qcu_multigrid_coarse.py` | legacy MG R/P、coarse dslash（含 wide） |
| `single_qcu_multigrid_strict.py` | Strict coarse/MATPC/R/P/prepare/reconstruct/FGMRES 生命周期 |
| `single_qcu_clover_multigrid.py` | Clover MG 与 `verifyCloverMultigridQcu` |
| `single_qcu_api.py` | 全部 `pyqcu.h` 导出符号 ABI 集合 |

常用入口：

```bash
python pyqcu/testing/qcu/single_qcu_api.py
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py --lat 8 8 8 8 --mass 0.05
python pyqcu/testing/qcu/single_qcu_wilson_dslash.py --pure-only
```

`single_qcu_*.py` 的正确性门统一默认使用 `8 8 8 16`；支持 `--lat X Y Z T`、`--mass M`、
`--pure-only`、`--no-doc` 和 `--no-progress`。设置 `QCU_STRICT_NUMERIC=1`
会把数值偏差升级为失败，默认只报告偏差。

`16 16 16 16` 及更大格点只能用于性能、容量或历史数据验证，不得替代
`8 8 8 16` 正确性门。

### QCU 单功能数值协议

- `applyWilsonDslashQcu` 返回目标 parity 的 compact 场，输入必须取相反 parity；C++ hopping 未乘
  `kappa`，符号与 `dslash.give_wilson_{eo,oe}` 的返回值相反。测试统一使用
  `single_function_common.qcu_wilson_dslash_reference`。
- `applyCloverDslashQcu` 在同一个 compact parity 上计算裸 Wilson hopping 后左乘
  `(I + T_clover)^{-1}`；参考实现见
  `single_function_common.qcu_clover_dslash_reference`，不能直接用 full-site Clover matvec 代替。
- `applyLaplacianQcu` 是三维、内部 `T=1` 的颜色场入口，gauge 布局为
  `[colour_out, colour_in, direction, X, Y, Z]`；小体积时 kernel 必须保留
  `idx < volume/face_volume` 边界守卫，避免 `_BLOCK_SIZE_=128` 的尾部线程越界。

## MultiGrid 与 Strict 现役入口

| 目录 | 用途 |
|---|---|
| `strict/quda_comparison/` | 当前 Strict-MG/QUDA 正确性探针、性能矩阵、报告和作图；先读该目录 `AGENTS.md` |
| `multigrid/volume24/final/` | 24³×72 Clover MG 规范基准套件 |
| `multigrid/benchmarks/large_volume/` | 16×32×32×48 大体积加速比研究 |
| `multigrid/benchmarks/multi_gpu/` | 多线程多卡基准 |

现役基准按目录文档直接运行，不进入 pytest 全树门禁。性能报告必须区分
`total_solve_median`、冷启动和逐线程 `mg_time`；纯吞吐运行可传
`--no-reference`，一致性运行保留 `run_reference=True`。

`conftest.clover.multigrid.py` 默认使用设备 0，可用 `QCU_DEVICE_ID` 覆盖；需要
重定向 C++ 日志时设置 `QCU_LOG_DIR`，否则日志按 cwd 相对路径写入
`logs/clover_multigrid.log`。当前 PyTorch 不包含 sm_60 kernel 的环境不能用
P100 执行该脚本。

## 迁移边界

2026-10-15 已删除旧的 `multigrid/{legacy,scaling}/` 和
`benchmarks/{volume32,volume1632}`，不要恢复这些工作区入口或继续引用其日志路径。
可复用算法已进入 `pyqcu/torch/tools/`、`pyqcu/torch/solver/`、`pyqcu/cuda/`、
`pyqcu/tools/` 和 `pyqcu/testing/__init__.py` 的现行实现；历史报告使用
`git show <tag>:<path>` 读取。新测试只加入上表的现役目录或当前 backend 测试文件。
