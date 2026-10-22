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
QUDA/PyQUDA 可通过 `QUDA_INSTALL` 选择安装树；库架构和 ABI 必须与当前设备及
PyQUDA 扩展一致。来源：`docs/env.md:1-9`、`docs/install.md:5-39`。

规范数据轴为 `s=spin`、`c=color`、`d=direction`、`p=parity`，时空维固定为最后
四轴 `x,y,z,t`。核心布局：

```text
gauge       = [c,c,d,x,y,z,t]
gauge_eo    = [p,c,c,d,x,y,z,t]
fermion     = [s,c,x,y,z,t]
fermion_eo  = [p,s,c,x,y,z,t]
wilson_term = [s,c,s,c,x,y,z,t]
clover_term = [s,c,s,c,x,y,z,t]
```

来源：`docs/dims.md:1-11`。

## 2. 测试与示例入口

旧顶层 `examples/` 已删除，现行路径统一为 `pyqcu/testing/**`；历史文档中的
`examples/...` 只按迁移规则理解。来源：`docs/examples.md:1-30`。

```bash
source ./env.sh
cd pyqcu/testing && pytest .
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

来源：`docs/three_way_benchmark_20261022.md:41-75,103-120,281-385,386-412,536-555`。

当前工作区还表明：CUDA E12 单 null-vector 粗空间未在真实 LIME 上产生 V-cycle 净收益，
fine-only `n_vcycles=0` 是现点时点选择；KUPL 同一组态上 `restart=0` 优于 coarse
correction。该结论状态为 `userinterlude`，不能脱离当前 cache/算法条件外推。来源：
`docs/progress_until_20261022_worktree.md:20-33,90-164,192-210`。

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
多设备 profile 只有显式测试目的才使用测试专用驱动。来源：`docs/profiler.md:1-4`、
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

来源：`docs/ORGANIZATION.md:7-53`、`docs/AGENTS.md:8-43`。
