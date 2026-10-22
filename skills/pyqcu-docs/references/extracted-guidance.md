# 从 `docs/**` 提取的工程参考

本文件只保存跨目录、会改变 agent 行为的稳定知识。完整公式、长表和历史记录回到
[`project-docs-map.md`](project-docs-map.md) 指向的原文。

## 1. 环境、构建与布局

```bash
source ./env.sh
bash ./build.sh
bash ./install.sh
```

- 基础运行要求 Python ≥ 3.10；`PYTHONPATH` 必须包含仓库根。
- CUDA 后端由 `build.sh` 构建 `cpp/cuda/qcu`，Cython 扩展由 `install.sh` 原地构建。
- 独立 KUPL 后端不依赖 CUDA：

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/kupl/qcu/build -j
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
python -m pytest -q pyqcu/testing/kupl
```

KUPL 动态库不在默认构建目录时设置 `PYQCU_KUPL_LIB=/absolute/path/libqcu.so`。
`env.sh` 始终设置 `LD_LIBRARY_PATH` 和 `PYTHONPATH`，仅当仓库内 QUDA 库存在时
设置 `QUDA_PATH`；`QUDA_INSTALL` 可显式覆盖安装树。库架构和 ABI 必须与当前设备
及 PyQUDA 扩展一致。来源：`docs/env.md:1-22`、`docs/install.md:1-50`。

规范数据轴为 `s=spin`、`c=color`、`d=direction`、`p=parity`，时空维固定为最后
四轴 `x,y,z,t`。核心布局：

```text
splited_rank = [x,y,z,t]
gauge       = [c,c,d,x,y,z,t]
gauge_eo    = [p,c,c,d,x,y,z,t]
fermion     = [s,c,x,y,z,t]
fermion_eo  = [p,s,c,x,y,z,t]
wilson_term = [s,c,s,c,x,y,z,t]
wilson_term_eo = [p,s,c,s,c,x,y,z,t]
clover_term = [s,c,s,c,x,y,z,t]
clover_term_eo = [p,s,c,s,c,x,y,z,t]
```

来源：`docs/dims.md:1-24`。

## 2. 测试与示例入口

旧顶层 `examples/` 已删除，现行路径统一为 `pyqcu/testing/**`；历史文档中的
`examples/...` 只按迁移规则理解。来源：`docs/examples.md:1-30`。

```bash
source ./env.sh
(cd pyqcu/testing && pytest .)
mpirun -np 4 python pyqcu/testing/python/conftest.py
python pyqcu/testing/qcu/single_qcu_api.py
pytest pyqcu/testing/quda
```

各子目录 `conftest.py` 是手动选择性入口；运行前检查被取消注释的测试集合，避免把
“pytest 启动成功”误认为目标测试已执行。

## 3. CUDA 生产执行拓扑

默认不变量：

```text
one Python process = one MPI rank = one GPU/DCU/GPGPU
```

跨设备并行通过 MPI rank 与四维格点分域；halo 和全局归约也走 MPI。设备选择必须在
PyTorch CUDA 与 native QCU 库初始化前完成。rank 数超过设备数只能作正确性冒烟，
不能报告为多卡加速。来源：`docs/cuda_execution_topology.md:7-53`。

每个进程只构造一个 rank-local `LatticeSet`/solver 上下文；`CUDA_VISIBLE_DEVICES`
等设备隔离变量必须在 PyTorch、Cython 桥或 native QCU 初始化前生效。MPI rank 数
默认应与参与设备数一致。

`MultiGpuMultigrid` 是单进程多线程、每线程跑完整 Legacy MG 的显式测试例外，
不是一次 solve 的跨设备分域，也不是生产模板。只有调用方显式导入并构造才启用。
来源：`docs/cuda_execution_topology.md:56-103`。

## 4. QCU、Strict 与 QUDA MultiGrid

### 4.1 先区分路径

- **QCU legacy** 指 `applyCloverMultigridQcu`，fine 系统为 odd-site Schur，粗层直接
  保存 33-tensor `A_c`，默认外层是 BiCGStab + 周期 additive coarse correction。
- **QCU Strict** 指 `applyMultigridStrictFgmresQcu`，使用 full-coarse、MATPC、
  `X/Y/Yhat` 持久层级和 fused right-FGMRES。
- **QUDA/PyQUDA** simple 默认使用 even-even MATPC、外层 GCR(8)、MG 右预条件、
  CA-GCR smoother 和 recursive coarse cycle。

来源：`docs/qcu_quda_multigrid_algorithms.md:18-43,272-418,437-467`。

### 4.2 Legacy 默认不是 QUDA-style MG

`simple/qcu_multigrid.py` 的默认路径是：

```text
Schur BiCGStab
+ 周期性 coarse correction
+ guarded coarse BiCGStab
+ adaptive correction gate
```

默认不是外层 FGMRES 或 CA-GCR。只有显式选择相应模式并配置 `_MG_MU_PRE_`、smoother、
cycle 位时才进入 MG 右预条件路径。`_MG_MU_PRE_=0` 在预条件路径回退为 4 步。
来源：`docs/qcu_quda_multigrid_algorithms.md:123-270,437-451`。

自适应门先跑 30 次纯 BiCGStab 标定；之后每 12 次比较窗口斜率，连续两个窗口改善
不足 15% 就永久关闭该 solve 的粗校正。单 rank 默认每 48 次用真 Schur 残差刷新；
旧注释中的“50 次”与实现不一致。来源：
`docs/qcu_quda_multigrid_algorithms.md:152-201`、
`docs/qcu_quda_multigrid_subalgorithms.md:822-868`。

### 4.3 粗算子的数据表示

| 路径 | 粗格几何 | 粗算子表示 | 邻接 |
|---|---|---|---|
| QCU legacy | compact odd coarse | 1 onsite + 8 nearest + 24 diagonal 的 33-tensor | 两跳支撑 |
| QCU Strict / QUDA | full coarse | onsite `X` + link `Y`；运行期常存 `X^-1`、`Yhat` | 一跳 |

33-tensor 不是“33 个邻居”；每个 block 是粗自由度之间的矩阵。Strict/QUDA 的
`X/Y` 是 Galerkin 有效量，不是原始物理 gauge/clover 的简单分块平均。两种 cache
的 geometry、槽序和 kernel 不兼容。来源：
`docs/qcu_quda_multigrid_coarse_operator.md:11-180`。

### 4.4 报告迭代数前的检查

只有以下条件一致时才能比较 MG 迭代数或倍率：

- fine system（Schur vs MATPC）一致；
- 外层 Krylov 与 MG 是 additive correction 还是 right preconditioner 一致；
- smoother、cycle、terminal solver 一致；
- coarse space、block、null-vector 数和容差一致；
- 计时边界和冷/热启动一致。

不同路径的“iteration”不是同一工作量单位。

### 4.5 当前 QUDA/PyQUDA 默认快照

以下默认值只绑定仓库当前 `QUDA 1.1.0-sm70` 与 `PyQUDA 0.10.54`；升级安装树或
修改 `QudaMultigridParam` 构造后必须重审，不能当作跨版本常量：

- setup 使用 CGNR，`setup_tol=1e-6`、`setup_maxiter=1000`、refresh 100、
  `n_vec_batch=1`、`num_setup_iter=1`、aggregate transfer；
- 默认 `nvec=24`，coarse spin 2，coarse color 48，transition block `2^4`，
  terminal block `4^4`；
- 外层 GCR(8)；每层 CA-GCR smoother，`nu_pre=0`、`nu_post=8`、
  `smoother_tol=0.25`；
- 非终端 coarse solver 为 GCR wrapper，terminal 为 CA-GCR，
  `coarse_solver_maxiter=16`、`coarse_solver_tol=0.25`；
- `reliable_delta=1e-5`，`tol_restart=5000*tol`，simple 驱动强制
  `QUDA_KAPPA_NORMALIZATION` 和 `use_init_guess=NO`。

来源：`docs/qcu_quda_multigrid_algorithms.md`、`docs/qcu_quda_multigrid_subalgorithms.md`。

2026-10-22 弃用审计固定于 `9e678901` 且未包含当时并发重构。当前树中的
TileLang no-op 仍可直接复核；旧 QUDA 风格别名、历史 harness 和占位目录则在每次
删除或改名任务中重新检查当前导出。若兼容别名仍然公开，应先给 deprecation
notice 和迁移周期，不能把历史审计表当成永久删除指令。

### 4.6 六实现统一架构方向

Torch/CUDA/KUPL 各有 Legacy/Strict，共六条路径。当前架构建议不是删掉后端差异，
而是把公共控制面、物理语义和结果协议放在 `pyqcu.api`，三套后端继续拥有数据面、
kernel、halo、stream/OpenMP/MPI 和 workspace 特化。Strict 作为通用语义骨架，
Legacy 作为显式 compact/cache 特化；公共 `SolveResult` 必须记录实际 Krylov、
correction、cycle 次数和 `residual_scope`，禁止把 MATPC/Schur/full residual
互相冒充。P0 的数据模型、配置、能力和 Protocol 已落地到 `pyqcu.api`；
`create_backend()`、完整后端工厂和逐后端原生结果对象仍未实现。当前
`pyqcu.solver.multigrid.Multigrid` 是兼容选择器。来源：
[`docs/qcu_unified_backend_strategy_20261023.md`](../../../docs/qcu_unified_backend_strategy_20261023.md)。

### 4.7 统一 API P0 契约

`pyqcu.api` 是后端无关控制面，不包含 kernel 或第四套算法：

- `LatticeGeometry` 区分 full/parity view，parity view 要求局部 T 为偶数；
- `SolveResult` 统一状态、计数和 `ResidualScope`，未知字段必须保持 `None`；
- `BackendCapabilities.require()` fail-closed，不根据名称或配置推断未声明能力；
- `AssetManifest.identity` 对 canonical JSON 做 SHA256，metadata 顺序不影响摘要；
- Protocol 约束语义，不允许 CUDA/KUPL 生产热循环逐迭代穿过 Python；
- `Multigrid.solve()` 保持原返回，`solve_result()` 只映射后端已暴露字段；
- `create_backend()` 与完整后端工厂尚未实现，不得写成现行入口。

来源：`docs/api_contract.md`、`pyqcu/api/AGENTS.md`。

## 5. 性能与证据解释

`docs/three_way_benchmark_20261022.md` 是时点证据，不是永久性能承诺。摘要中的
`24³×64`、c64、V100 同窗主结果：

| 路径 | solve 中位 | 迭代 | 真残差 |
|---|---:|---:|---:|
| CUDA QCU standalone BiCGStab | 0.147801 s | 18 | 6.709e-7 |
| QUDA Clover BiCGStab | 0.156680 s | 19 | 4.207e-7 |
| KUPL `restart=0`，8 ranks × 2 threads | 7.532 s | 19 | 4.011e-7 |

来源：`docs/three_way_benchmark_20261022.md:5-120`。

必须保留的口径：

- CUDA/QUDA 是单 V100 solver-only；KUPL 是 18 核 Xeon CPU、不同硬件类别。
- `24³×64` 适合同窗比较；小体积受 QUDA 26–43 ms 固定开销影响，不能外推 kernel
  吞吐。
- 共享机窗口漂移可使同一 CUDA solve 从约 `0.12 s` 变到 `0.15 s`；不混合窗口。
- 子体积是真实 LIME 的非周期裁剪，只能作尺寸/并行研究。
- 当前证据没有生产多卡 CUDA 扩展；`MultiGpuMultigrid` 数据不得混入。
- 同一 V100/L24 窗口内，`sm_60+PTX` 为 `0.146966 s`、原生 `sm_70` 为
  `0.147531 s`，差异 `0.38%`，小于样本散布；不要宣称原生 SASS 有可测计算收益。
  架构选择的价值是消除 PTX JIT、固定 SASS 和保证设备兼容。
- KUPL LTO 的单点 A/B 为 `7.556 -> 7.196 s`（`4.8%`），解逐位一致、CTest
  `29/29`；该结果只覆盖 `24^3x64`、8 ranks × 2 threads，且 LTO 不是默认库。
- Legacy/Strict status 报告是对指定 JSON/stdout 目录的静态审计；其中“KUPL 无
  记录”只限该目录，跨工作区性能判断以后续同窗三方记录为准。
- KUPL `restart=5/10/20` 在当前 E12 组态上为负收益，但“2.8–3.0x”混用了不同
  窗口；同窗 `restart=0` 为 `7.842 s` 对 coarse 约 `15.6 s`，只能保守写成约
  `2.0x`，且主要是 47 对 19 次迭代的差异。
- P100 16 GiB 的大格 c64 六单元曾因 `15.9/16.0 GiB` 门限和 `570 s` staging
  超时被替换；单 P100 `16^3x32` 峰值 `16045/16384 MiB`，双卡 `16^3x48`
  `1800 s` 超时。这些只是容量证据，不是正式性能结论。
- Legacy CUDA MG 历史收益随尺寸非单调：小格点可超过 `2x`，大格大量路径低于
  `1x`；当前 L24 E12 实际 fine-only。旧小格倍率不得外推。
- 当前 33-tensor 精确 shape 下 E24 cache 约 `5.22 GB`、E48 约 `18.86 GB`，
  E12 local-stencil setup 峰值约 `29.75 GB`；32 GiB V100 扩 E 前必须先降低
  coarse/setup 内存。
- 工作区 benchmark 数字不足以单独承担长期复现证据。归档前必须固定并保存输入、
  RHS、库、配置和 cache identity；`data/` 中 gitignored 的原始 JSON 需随 tag
  归档，否则后续只剩摘要而无法复核。

来源：`docs/three_way_benchmark_20261022.md:41-75,103-120,281-385,386-412,536-555`。

当前工作区还表明：CUDA E12 单 null-vector 粗空间未在真实 LIME 上产生 V-cycle 净收益，
fine-only `n_vcycles=0` 是现点时点选择；KUPL 同一组态上 `restart=0` 优于 coarse
correction。该结论状态为 `userinterlude`，不能脱离当前 cache/算法条件外推。来源：
`docs/progress_until_20261022_worktree.md:20-33,90-164,192-210`。

### 5.1 Legacy/Strict 运行时显存

同一输入、c64、两级、`nvec=12` 的设备级 A/B 中，Legacy-MG 的实际
fine-only solve 峰值低于 Strict-MG：

| 格点 | Legacy | Strict | Strict / Legacy |
|---|---:|---:|---:|
| `8x8x8x16` | 0.596 GiB | 0.789 GiB | 1.3246 |
| `12x12x12x24` | 1.533 GiB | 2.326 GiB | 1.5172 |
| `16x16x16x32` | 4.113 GiB | 6.715 GiB | 1.6325 |

必须保留两个边界：Legacy 记录均为 `n_vcycles=0`，并未执行 coarse V-cycle；
`24x24x24x64` 没有 Strict solve JSON。Strict 峰值还包含 hierarchy setup 与首次
fused FGMRES workspace，不能把该表解释成单一常驻集大小。来源：
[`docs/legacy_vs_strict_mg_memory_20261023.md`](../../../docs/legacy_vs_strict_mg_memory_20261023.md)。

复现入口：

```bash
python pyqcu/testing/simple/prepare_case.py --lat 24 24 24 64 \
  --out-dir data/production_L24
python pyqcu/testing/simple/solve_case.py --backend cuda \
  --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24
python pyqcu/testing/simple/compare_three_way.py \
  --lat 24 24 24 64 --device cuda:2 --kupl-ranks 4 --kupl-threads 4
```

来源：`docs/three_way_benchmark_20261022.md:413-491`。

## 6. Profiler

```bash
cd pyqcu/testing/profiler
mpirun -np 1 python -u conftest.py
```

将生成的 `trace_*.json` 载入 Perfetto。profiler 仍遵守一进程一 rank 一设备；
多设备 profile 只有显式测试目的才使用测试专用驱动。来源：`docs/profiler.md:1-15`、
`docs/cuda_execution_topology.md:42-103`。

历史性能汇总 `docs/profiler_solver_performance_history.md` 适合查记录 ID、失败/撤回项和证据路径，
不适合把不同标签、计时边界、算法和硬件的绝对值重新汇总成加速比。其明确要求先看
计时边界、算法、组态/精度、硬件/窗口。来源：
`docs/profiler_solver_performance_history.md:5-79,697-732`。

## 7. 文档生命周期

- 正式独立文档写 `docs/**`；当前任务日志和未归档阶段报告写 `logs/**`。
- 阶段报告先提交并打 tag；后续清理提交再删除工作区副本，历史通过 tag 读取。
- 数据/构建实体写 `data/**`；测试和可维护脚本写 `pyqcu/testing/**`。
- 技能新增内容前先判断它是否只是 `docs/**` 的完整副本；若是，改为来源映射。

来源：`docs/ORGANIZATION.md:7-53`、`docs/AGENTS.md:8-80`。

## 8. 全部正式文档的结论覆盖索引

本表是 `docs/**` 覆盖检查的可执行摘要。详细公式、长表和历史数字回到对应文档；
同一技术规则若已在域技能展开，本表只保留选路与验收要点。带日期的耗时和倍率是
时点证据，不得因出现在索引中而升级为永久承诺。

| 文档 | 有参考价值的稳定结论 | 落点 |
|---|---|---|
| `docs/AGENTS.md` | `docs/**` 是正式文档规范源；新增/迁移文档必须同步索引，tag 前报告只在工作区暂存。 | `pyqcu-docs`, `pyqcu-form` |
| `docs/ORGANIZATION.md` | 文档、日志、数据、测试各只有一个规范目录；tag 报告先提交打标，再在后续提交清理工作区副本。 | `pyqcu-form` |
| `docs/dims.md` | `s/c/d/p/x/y/z/t` 轴语义固定，时空维永远是最后四轴；gauge/fermion/Clover 使用 canonical 顺序。 | `pyqcu-tools`, `pyqcu-lattice`, `pyqcu-dslash` |
| `docs/env.md` | Python ≥ 3.10，`PYTHONPATH` 包含仓库根；环境入口为 `env.sh`。 | `pyqcu-tools`, `pyqcu-testing` |
| `docs/install.md` | CUDA 走 `build.sh`/`install.sh`；KUPL 独立 CMake/CTest；QUDA/PyQUDA 必须匹配当前架构与 ABI。 | `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-testing-pyquda` |
| `docs/examples.md` | 顶层 `examples/` 已删除，现行测试/示例入口统一为 `pyqcu/testing/**`；conftest 是手动选择入口。 | `pyqcu-testing` |
| `docs/profiler.md` | 使用 `torch.profiler` 生成 Chrome trace，再用 Perfetto 分析；继续遵守一进程一 rank 一设备。 | `pyqcu-testing-profiler` |
| `docs/cuda_execution_topology.md` | 生产拓扑固定为一 Python 进程 = 一 MPI rank = 一设备；`MultiGpuMultigrid` 仅测试专用。 | `pyqcu-cuda`, `pyqcu-benchmark` |
| `docs/torch_backend_layout.md` | Torch/CANN 实现统一位于 `pyqcu/torch/**`；顶层旧导入保留兼容门面，`pyqcu.solver.multigrid` 是三后端公共选择器的例外。 | `pyqcu-solver`, `pyqcu-dslash`, `pyqcu-smear`, `pyqcu-tools` |
| `docs/api_contract.md` | P0 公共数据模型、配置、能力、资产 manifest 和 Protocol 已实现；`create_backend()` 与完整后端工厂未实现。 | `pyqcu-api`, `pyqcu-solver` |
| `docs/qcu_unified_backend_strategy_20261023.md` | 六实现共享控制面和数学契约，但保留后端数据面；P0 API 已落地，完整 backend factory 仍属设计提案。 | `pyqcu-api`, `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` |
| `docs/qcu_quda_multigrid_algorithms.md` | QCU legacy 默认是 Schur BiCGStab + 周期 additive correction，不是 QUDA GCR+CA-GCR；自适应门为 30/12/15%，单 rank 真 Schur refresh 条件为 48 步。 | `pyqcu-solver`, `pyqcu-cpp-qcu` |
| `docs/qcu_quda_multigrid_subalgorithms.md` | QCU 使用 inverse iteration、local QR、33-tensor Galerkin、guarded coarse BiCGStab；QUDA 当前使用 CGNR setup、aggregate transfer、CA-GCR smoother、GCR(8) 与 recursive cycle。 | `pyqcu-solver` |
| `docs/qcu_quda_multigrid_coarse_operator.md` | Legacy 保存 compact odd `P†SP` 33-tensor；Strict/QUDA 保存 full-coarse `X/Y/Yhat`。粗层量是有效 Galerkin 算符，不是物理 gauge/clover。 | `pyqcu-solver`, `pyqcu-tools` |
| `docs/progress_kupl_cuda_mg_20261021.md` | `dev99..1640349d` 时点证据：KUPL 6×6 Clover 与 512-site coarse tile、CUDA `run_reference`、完整 33-tensor cache 已落地；仍有双 rank/CUDA-aware MPI 与 QUDA MG 缺口。 | `pyqcu-kupl`, `pyqcu-cuda` |
| `docs/progress_until_20261022_worktree.md` | `userinterlude` 证据：CUDA E12 粗空间不足，生产选择 fine-only；W=8 经边界验证；若干双流/Graph/Allreduce/SIMD 候选已撤销。 | `pyqcu-cuda`, `pyqcu-kupl`, `pyqcu-benchmark` |
| `docs/three_way_benchmark_20261022.md` | V100 L24 c64 同窗主表中 CUDA BiCGStab `0.147801 s`、QUDA `0.156680 s`；CUDA fine-only MG 不应替代 BiCGStab；KUPL 最优为 `restart=0`。跨窗口和 CPU/GPU 绝对值不可混算。 | `pyqcu-benchmark`, `pyqcu-kupl`, `pyqcu-cuda` |
| `docs/qcu_six_way_benchmark_20261023.md` | 真实 `8^3×16` LIME 前缀上六版本均完成；CUDA Strict 与 QUDA aligned MG 为 `0.021693 s` 对 `0.144755 s`，Torch Strict 为 `1.670798 s`。CUDA Legacy 初始 deflation 实际执行但比 fine-only 慢；KUPL Legacy correction 7/7 被拒绝。QUDA 未加载同一 QIO null basis，CPU/GPU 不得混算。 | `pyqcu-benchmark`, `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` |
| `docs/legacy_vs_strict_mg_status_20261022.md` | Legacy build/solve 4/4 PASS，但 solve 均 `n_vcycles=0`；Strict 3/4 PASS，24³×64 缺失；KUPL 无对应记录，不能计算跨后端 MG 倍率。 | `pyqcu-benchmark`, `pyqcu-solver` |
| `docs/legacy_vs_strict_mg_memory_20261023.md` | 同输入 c64 两级 `nvec=12` 中 Legacy 峰值更低；Strict 高 32.46%–63.25%。Legacy 未执行 coarse V-cycle，24³×64 Strict 缺失。 | `pyqcu-benchmark`, `pyqcu-solver`, `pyqcu-cuda` |
| `docs/multigrid_legacy_strict_comparison_20261022.md` | 统一公开名称为 `Multigrid` / `LegacyMultigrid` / `StrictMultigrid`；Torch/CUDA/KUPL 共用 setup/solve/close，但内部算法、槽序和生命周期不可互换。 | `pyqcu-solver`, `pyqcu-cuda`, `pyqcu-kupl` |
| `docs/multigrid_deprecation_audit_20261022.md` | 稳定基线审计中 TileLang `conftest.py` 是 no-op 假阳性；历史 CUDA harness 是删除候选；占位后端目录不是死代码；源码重构后必须重审。 | `pyqcu-testing-tilelang`, `pyqcu-solver`, `pyqcu-form` |
| `docs/profiler_solver_performance_history.md` | 409 条历史记录只能按求解器、输入、精度、硬件、窗口和计时边界解释；`4.3817x` 已撤回，不得恢复为正式结果。 | `pyqcu-benchmark`, `pyqcu-testing-profiler` |
| `docs/form_pyqcu_20261020.md` | 记录 form 托管块、历史例外和标签交付门的演进；当前操作规则以根 `AGENTS.md` 和 `pyqcu-form` 为准。 | `pyqcu-form` |
| `docs/form_pyqcu_20261022.md` | 记录 tag 名路径整改、索引补齐与恢复方法；历史路径只用于追溯，不作为现行命名模板。 | `pyqcu-form` |
