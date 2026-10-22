# AGENTS.md — pyqcu/testing/qcu

C++ CUDA 后端测试（经 Cython 桥）。从 Python 驱动 `libqcu.so`。

CUDA 生产拓扑固定为“一个 Python 进程 = 一个 MPI rank = 一张
GPU/DCU/GPGPU”。`bench_multi_gpu.py`、`conftest.multi_gpu.py` 和
`slurm_multi_gpu_smoke.sh` 使用的 `MultiGpuMultigrid` 是显式调用的测试专用
例外，只验证多卡一致性或吞吐，不是生产部署模板。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.cuda.py` | CUDA 可用性与上下文 |
| `conftest.mpi.py` | MPI 网格设置与 halo 交换 |
| `conftest.wilson.bistabcg.py` | C++ 后端 Wilson BiStabCG |
| `conftest.wilson.bistabcg.dslash.py` | Wilson BiStabCG dslash 内核 |
| `conftest.wilson.cg.py` | Wilson CG 求解器 |
| `conftest.clover.py` | Clover 项构造 |
| `conftest.clover.bistabcg.py` | Clover BiStabCG 求解器 |
| `conftest.clover.bistabcg.dslash.py` | Clover BiStabCG dslash |
| `conftest.clover.multigrid.py` | Clover multigrid V-cycle 求解器 |
| `bench_multi_gpu.py` | 测试专用双/多卡 Clover-MG 稳态基准；显式调用 `MultiGpuMultigrid`，不是生产拓扑 |
| `slurm_multi_gpu_smoke.sh` | 测试专用 Slurm 2-GPU 构建、数值门、双卡一致性与真实 LIME 回归 |

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

## 用法

```bash
mpirun -np 1 python pyqcu/testing/qcu/conftest.clover.multigrid.py
```

双卡 Slurm 冒烟（从仓库根目录提交）：

```bash
sbatch --gpus=2 --cpus-per-task=64 --time=01:00:00 \
  --output=logs/slurm-%j-multi-gpu.out \
  --error=logs/slurm-%j-multi-gpu.err \
  pyqcu/testing/qcu/slurm_multi_gpu_smoke.sh
```

纯 MG 吞吐基准应传 `--no-reference`；一致性测试则必须保留默认的
`run_reference=True`。缓存命中后的稳态总耗时应使用 `total_solve_median`，冷启动
总耗时和逐线程 `mg_time` 必须分开报告。

`single_qcu_*.py` 独立入口默认使用 `16 16 16 16` 格点。启动时打印脚本文档和输入参数，运行中显示分步进度及耗时，结束时输出总耗时与分项耗时；可用 `--lat X Y Z T`、`--mass M` 调整输入，`--pure-only` 只运行纯 PyTorch 参考，`--no-doc`/`--no-progress` 分别隐藏文档/进度输出。

输出：收敛日志 → `logs/clover_multigrid.log`，性能报告 → `logs/clover_multigrid_report.log`

### conftest.clover.multigrid.py 运行约定（2026-08-16 修复后）

- **设备选择**：脚本默认 `QCU_DEVICE_ID=0`（可用环境变量覆盖）。**CUDA 运行时枚举与 nvidia-smi 顺序不同**——本机实测 `cuda:0=V100-32G`（性能最佳）、`cuda:1/2=P100-16G`（nvidia-smi 为 `0/1=P100, 2=V100`）；C++ 端单 rank 不调用 `cudaSetDevice`，跟随 torch 当前设备，脚本内 `torch.cuda.set_device` 同时约束两端。P100（sm_60）在当前 torch（CUDA 12.6，sm_70+）下无 kernel image，无法跑本脚本。
- **日志路径**：脚本必须设置 `os.environ["QCU_LOG_DIR"] = LOG_DIR`——C++ `log_write` 默认写 cwd 相对路径 `logs/clover_multigrid.log`，不重定向则 Python 端（读 `~/PyQCU/logs/tmp/clover_multigrid.log`）解析不到 `CONVERGENCE_HISTORY`，`Conv pts=0`、出图失败。
- **MG 参数**（2026-08-16 optim 后）：`COARSE_MAX_ITER` 必须 ≥200（=50 时粗 solve 每轮截断在 target 之前，粗解精度不足 → V-cycle 校正无效 → 500 次跑满）；`nv_iters` 用 20（=1 时粗算子质量差，V-cycle 同样失效）；`COARSE_TOL_FACTOR` 用 3000（粗 solve 相对 tol=3e-3，扫描 10/100/300/1000/3000/10000/30000，3000 为速度-稳定最优：8x8x8x16 MG 0.503→0.255s，speedup 1.0→1.9）；`_MG_LEVEL1_NUM_RESTART_`=5（V-cycle 频率 3→5，n_vcycles 6→4）；`use_cache=True`（粗算子缓存 `~/PyQCU/data/logs/nullvec_cache/`，key 含格子/dof/nv_iters/nv_tol，参数变化自动 miss 重建；3 配置全缓存命中时总运行 9min→18s）。
- **3L 配置**（大格子优化，2026-08-16）：`12x12x12x16`/`16x16x16x16` 用 3 层 `[12,48,48]`（较 2L 提速 25-40%，coarsest 变小、level1 普通路径 ~13-28 次迭代即达 tol）；8x8x8x16 保持 2L（3L 时 level1 变普通路径反劣化 4.6 倍）。配套：`_MG_LEVEL2_ATOL_=ATOL×CF×3`（level2 可比 level1 松，迭代减半）；`_MG_LEVEL2_NUM_RESTART_`=5（level1→level2 校正频率）；`_MG_LEVEL2_T_ = level1_T//MG_GRID[3]`（SCHUR 半 T 链，原 `Lt//(MG_GRID[3]^2)` 与实际粗算子 3x3x3x8 不符 → C++ 越界读）。
- C++ 端（`lattice_clover_multigrid.h` / `multigrid.cu`）配套修复：
  - 粗 solve（fused + 普通路径）`r0 < 1e-4` 时跳过（fp32 下 target 不可达 → BiStabCG 0/0 → nan 毒化 fine 残差）；
  - `run()` V-cycle 在 fine Schur 残差 `rn ≤ 100·atol` 时停止（空转校正 + state reset 使残差反弹 ~1e-5）；
  - `run_test` 全算子残差须在掩码棋盘布局 `[12,X,Y,Z,T]`（通道 `lat_4dim_SC`）上直算（`b=b_e+b_o`，`D·x` 用掩码算子组件）；`parity_to_full`/`full_to_parity` 假设 `[..,T/2]` 压缩布局，与细层不匹配，误用会报 `|D*x-b|/|b|~1.16`（解实际正确）；
  - fused 粗 solve 阈值 65536→262144（大粗层 82944/196608 也走 fused——普通路径每迭代 ~5ms host 同步主导；fused 大粗层 ~13ms/iter 带宽受限，仍占优 ~10%）。
  - **fused grid 下限实验（回退）**：grid<SM 数时补 block 会引入 nan（cooperative 部分空转 block 的 block_dot 竞争），勿再启用。
  - **fused 数值非确定性（WSL2）**：`coarse_solve_cg`（cooperative + grid.sync）在同一输入下解有 ~1e-7 级双模波动（NT=128、`__threadfence` 均无效；普通路径完全确定）——WSL2 驱动层 cooperative 同步问题，解始终正确收敛（PASS）；mg_time 波动（如 16x16x16x16 1.1-1.9s）主因为环境 GPU 频率/调度（`nvidia-smi -lgc` 锁频在 WSL2 报 Unknown Error 不可用），非代码问题。

## MultiGrid 历史套件

按里程碑先提交报告和产物并打 tag，随后在后续清理提交中删除工作区副本；历史读取使用
`git show <tag>:<path>`，不保留 `logs/<tag>/` 报告目录。2026-10-15 整理：
`multigrid/legacy/`（dev73）与 `multigrid/scaling/`（dev74）两代套件已整体删除
（代码可由 git 历史恢复），其可复用算法已全部收入库内；现存套件为
`volume24/final/`（24³×72 规范套件）与 `benchmarks/{large_volume,multi_gpu}`。

已删除套件中可复用部分的库内去向：

| 已删除来源 | 库内规范位置 |
|---|---|
| `legacy/integrated_single_thread/main.py`、`legacy/verify_results.py` | `verify_nullvecs` → `pyqcu/testing/__init__.py` |
| `legacy/mg_stencil_build.py`、`test12/main.py::build_stencil_mt` | `pyqcu/tools/_multigrid.py`：`build_stencil` / `build_stencil_mt` / `apply_stencil` |
| `legacy/quick_test.py`、`legacy/mg_nullvec_cache.py` | 缓存路径与 h5 存取 → `pyqcu/cuda/_multi_gpu.py` + `pyqcu/tools/_io.py` |
| `legacy/mg_schur_build.py` | `pyqcu/tools/_strict_galerkin.py`：`build_strict_galerkin` |
| `legacy/mg_pyref_expt.py` | `pyqcu/solver/_quda_multigrid.py` |
| `legacy/reference_benchmark.py`（`parse_mg_log` / `bistabcg_history`） | `pyqcu/cuda/_logs.py` / `pyqcu/solver/_bistabcg.py` |
| `scaling/dslash_adapter.py` | `CudaSchurOp` → `pyqcu/cuda/_schur_op.py` |
| `scaling/budget.py` | 预算模型 → `pyqcu/tools/_budget.py` |
| `scaling/benchmark.py`（环境/资源统计） | `pyqcu/tools/_env.py` |
| 5 处绘图副本（`legacy/make_plots.py`、`scaling/{make_plots,validation_plots}.py`、`volume24/*/generate_assets.py`） | 图表共用层 → `pyqcu/tools/_bench_plots.py` |

`data/logs/nullvec_cache` 为共享缓存，勿改动。

### 已归档结论（dev73/dev74，代码已删除，结论保留）

- 本地小卡 MG 恒慢（speedup < 1，硬件特性）；服务器 V100-32G `8x8x8x16` 实测 2.43x 达标。
- 参数相对行为（3L>2L、r20>r10）在两块 GPU 上一致，可迁移。
- 多线程 stencil 构建在单卡无收益（GPU 瓶颈），面向多卡/多节点集群。
- `CudaSchurOp` 依赖 C++ 端 `applyCloverBistabCgDslashQcu`（已移除首尾全局
  `cudaDeviceSynchronize`，见 `cpp/cuda/qcu/src/apply_clover_bistabcg_dslash.cu`）。

产出与运行指南保留为当时记录：`pyqcu/testing/logs/multigrid/{legacy,scaling}/`
（`server_validation_guide.md`/`.tex`/`.pdf`、报告、LaTeX 表、PNG 图）。

| `conftest.multi_gpu.py` | 测试专用多线程多卡 C++ Clover MG 一致性验证；不是生产并行路径 |
