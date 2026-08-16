# AGENTS.md — examples/qcu

C++ CUDA 后端测试（经 Cython 桥）。从 Python 驱动 `libqcu.so`。

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

## 用法

```bash
mpirun -np 1 python examples/qcu/conftest.clover.multigrid.py
```

输出：收敛日志 → `logs/clover_multigrid.log`，性能报告 → `logs/clover_multigrid_report.log`

### conftest.clover.multigrid.py 运行约定（2026-08-16 修复后）

- **设备选择**：脚本默认 `QCU_DEVICE_ID=0`（可用环境变量覆盖）。**CUDA 运行时枚举与 nvidia-smi 顺序不同**——本机实测 `cuda:0=V100-32G`（性能最佳）、`cuda:1/2=P100-16G`（nvidia-smi 为 `0/1=P100, 2=V100`）；C++ 端单 rank 不调用 `cudaSetDevice`，跟随 torch 当前设备，脚本内 `torch.cuda.set_device` 同时约束两端。P100（sm_60）在当前 torch（CUDA 12.6，sm_70+）下无 kernel image，无法跑本脚本。
- **日志路径**：脚本必须设置 `os.environ["QCU_LOG_DIR"] = LOG_DIR`——C++ `log_write` 默认写 cwd 相对路径 `logs/clover_multigrid.log`，不重定向则 Python 端（读 `~/PyQCU/logs/tmp/clover_multigrid.log`）解析不到 `CONVERGENCE_HISTORY`，`Conv pts=0`、出图失败。
- **MG 参数**（2026-08-16 optim 后）：`COARSE_MAX_ITER` 必须 ≥200（=50 时粗 solve 每轮截断在 target 之前，粗解精度不足 → V-cycle 校正无效 → 500 次跑满）；`nv_iters` 用 20（=1 时粗算子质量差，V-cycle 同样失效）；`COARSE_TOL_FACTOR` 用 3000（粗 solve 相对 tol=3e-3，扫描 10/100/300/1000/3000/10000/30000，3000 为速度-稳定最优：8x8x8x16 MG 0.503→0.255s，speedup 1.0→1.9）；`_MG_LEVEL1_NUM_RESTART_`=5（V-cycle 频率 3→5，n_vcycles 6→4）；`use_cache=True`（粗算子缓存 `~/PyQCU/logs/nullvec_cache/`，key 含格子/dof/nv_iters/nv_tol，参数变化自动 miss 重建；3 配置全缓存命中时总运行 9min→18s）。
- **3L 配置**（大格子优化，2026-08-16）：`12x12x12x16`/`16x16x16x16` 用 3 层 `[12,48,48]`（较 2L 提速 25-40%，coarsest 变小、level1 普通路径 ~13-28 次迭代即达 tol）；8x8x8x16 保持 2L（3L 时 level1 变普通路径反劣化 4.6 倍）。配套：`_MG_LEVEL2_ATOL_=ATOL×CF×3`（level2 可比 level1 松，迭代减半）；`_MG_LEVEL2_NUM_RESTART_`=5（level1→level2 校正频率）；`_MG_LEVEL2_T_ = level1_T//MG_GRID[3]`（SCHUR 半 T 链，原 `Lt//(MG_GRID[3]^2)` 与实际粗算子 3x3x3x8 不符 → C++ 越界读）。
- C++ 端（`lattice_clover_multigrid.h` / `multigrid.cu`）配套修复：
  - 粗 solve（fused + 普通路径）`r0 < 1e-4` 时跳过（fp32 下 target 不可达 → BiStabCG 0/0 → nan 毒化 fine 残差）；
  - `run()` V-cycle 在 fine Schur 残差 `rn ≤ 100·atol` 时停止（空转校正 + state reset 使残差反弹 ~1e-5）；
  - `run_test` 全算子残差须在掩码棋盘布局 `[12,X,Y,Z,T]`（通道 `lat_4dim_SC`）上直算（`b=b_e+b_o`，`D·x` 用掩码算子组件）；`parity_to_full`/`full_to_parity` 假设 `[..,T/2]` 压缩布局，与细层不匹配，误用会报 `|D*x-b|/|b|~1.16`（解实际正确）；
  - fused 粗 solve 阈值 65536→262144（大粗层 82944/196608 也走 fused——普通路径每迭代 ~5ms host 同步主导；fused 大粗层 ~13ms/iter 带宽受限，仍占优 ~10%）。
  - **fused grid 下限实验（回退）**：grid<SM 数时补 block 会引入 nan（cooperative 部分空转 block 的 block_dot 竞争），勿再启用。
  - **fused 数值非确定性（WSL2）**：`coarse_solve_cg`（cooperative + grid.sync）在同一输入下解有 ~1e-7 级双模波动（NT=128、`__threadfence` 均无效；普通路径完全确定）——WSL2 驱动层 cooperative 同步问题，解始终正确收敛（PASS）；mg_time 波动（如 16x16x16x16 1.1-1.9s）主因为环境 GPU 频率/调度（`nvidia-smi -lgc` 锁频在 WSL2 报 Unknown Error 不可用），非代码问题。

## Dev 套件（dev73_5 / dev74 / dev74_1）

按里程碑归档于子目录，产物统一落到 `logs/<tag>/` 对应子目录：

| 子目录 | 内容 | 产物 |
|---|---|---|
| `dev73/` | `mg_dev73_5_*.py` 系列：对 `applyCloverMultigridQcu` vs Clover BiStabCG 参考（`applyCloverBistabCgQcu`）做精度/格点/求解器参数扫描，及早期 mg_v4/mg_dev_* 历史脚本 | `logs/dev73/`（报告、LaTeX 表、PNG 图） |
| `dev74/` | `mg_dev74_*.py` 进阶套件（大格子 + 资源统计 + 多线程构建）与 `mg_dev74_1_*.py` 服务器加速比套件 | `logs/dev74/` |

所有脚本 `LOG_DIR` 硬编码为 `~/PyQCU/logs/<tag>`（如 dev74 脚本 → `logs/dev74/`）；`logs/nullvec_cache` 为共享缓存，勿改动。

### Dev74 进阶套件（大格子 + 资源统计 + 多线程构建）

`mg_dev74_*.py` 系列在 dev73_5 协议上扩展：

| 脚本 | 功能 |
|---|---|
| `mg_dev74_dslash.py` | `CudaSchurOp`：封装 `applyCloverBistabCgDslashQcu`（C++ Schur 奇偶算子，输入/输出 `[12,X,Y,Z,T/2]`），每实例独立 params 副本 + set_index 槽位，多线程安全 |
| `mg_dev74_layout_test.py` | C++ dslash 输入布局对照实验（vs Python `matvec_parity`） |
| `mg_dev74_stencil_mt.py` | 多线程 stencil build（探测点写集不相交，线程安全）+ 对照验证 |
| `mg_dev74_budget.py` | 显存/内存/磁盘预算模型（cold 53KB/V、warm 27KB/V 实测校准；`--fit`） |
| `mg_dev74_bench.py` | 本地（默认）/集群（`--cluster`）bench + 资源统计（cold/warm 显存、RSS、磁盘） |
| `mg_dev74_clean.py` | 干净测量（独立进程交叉计时）+ 资源统计 |
| `mg_dev74_verify.py` | 正确性验证（gauge/解/null_vecs + CudaSchurOp 对照） |
| `mg_dev74_collect.py` | 汇总 → `logs/dev74/dev74_results.json` |
| `mg_dev74_mktable.py` / `mg_dev74_plots.py` | LaTeX 表 / PNG 图 |
| `mg_dev74_cluster.sh` | 集群大格子运行（dry-run 默认，`RUN=1` 执行；16x32x32x32 单卡可行，16x32x32x64 需分阶段构建，24x32x32x64 需多卡） |

注意：`CudaSchurOp` 依赖 C++ 端 `applyCloverBistabCgDslashQcu`（已移除首尾全局 `cudaDeviceSynchronize`，见 `cpp/cuda/qcu/src/apply_clover_bistabcg_dslash.cu`）；多线程构建在单卡无收益（GPU 瓶颈），面向多卡/多节点集群。

### Dev74_1 服务器套件（加速比 > 1.5 验证）

| 脚本 | 功能 |
|---|---|
| `mg_dev74_1_sweep.py` | 本地/服务器参数扫描（r/ct/cmi/levels，独立进程干净测量）→ `logs/dev74/dev74_1_sweep.json` |
| `mg_dev74_1_check.py` | 加速比断言（默认 gate=1.5；`--file` 显式指定 json，exit 0/1/2） |
| `mg_dev74_1_plots.py` | 作图（范围与 dev73_5 一致：收敛历史/热点/加速比/耗时/参数扫描）→ `logs/dev74/dev74_1_*.png` |
| `mg_dev74_1_server.sh` | 服务器一键流程（Step 0 自检 → Step 1 强制闸门 8x8x8x16 → Step 2 扫描 → Step 3/4 大格子 → Step 5 断言；`RUN=1` 执行） |

运行指南：`logs/dev74/dev74_1_guide.md` / `.tex` / `.pdf`。关键结论：本地小卡 MG 恒慢
（speedup<1，硬件特性），服务器 V100-32G 8x8x8x16 实测 2.43x 达标；参数相对行为
（3L>2L、r20>r10）两 GPU 一致可迁移；16x32x32x32 单卡 cold 可行，16x32x32x64 需
分阶段构建，24x32x32x64 需多卡。

| `conftest.multi_gpu.py` | 多线程多卡 C++ Clover MG 一致性验证（`test_multi_gpu_multigrid`；单卡环境 N 线程共享一卡验证线程隔离） |
