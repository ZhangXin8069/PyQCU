# MultiGrid 分层性能与残差对照实现计划

**目标:** 为 PyQCU Strict MultiGrid 与 QUDA MultiGrid 增加默认关闭的分层计时/残差 trace，完成至少三层、不同格子与精度的公平对照，并优化 PyQCU 直至在正式协议上显著领先。

**架构:** 两侧 trace 均由环境变量显式开启；未设置环境变量时不分配诊断缓冲、不增加同步、不改变求解路径。公共收集器保存原始 trace 与结构化 JSON，报告只引用实测运行生成的数据。

**技术栈:** CUDA C++、Cython、PyTorch、PyQUDA、QUDA、HDF5、pytest、LaTeX。

**规格:** 用户 2026-09-16 任务定义；报告版式参考 `docs/report_multigrid_quda_pyqcu_20260909.tex`。

**全局约束:**
- 所有新增性能功能默认关闭。
- PyQCU 与 QUDA 使用相同 gauge、RHS、null-vector 来源、精度、层级、block、MATPC parity、停止容差和计时边界。
- 最细层迭代计数必须来自与 QUDA 对齐的最外层 solver iteration；同时记录每层 preconditioner 调用与 coarse Krylov iteration，禁止混写。
- 正式计时排除 trace 日志 I/O、真残差复算、内存采样和 QUDA setup 日志。
- 所有输出数据写入 `/root/PyQCU/data`，报告写入 `/root/PyQCU/docs`。
- 未经完整回归不得提交或打标签。

## Task 1: QUDA 分层 trace

**文件:**
- 修改: `refer/git-rep/quda/lib/multigrid.cpp`
- 修改: `refer/git-rep/quda/lib/inv_gcr_quda.cpp`
- 同步构建源: `data/quda-qio-build/source-shadow/lib/multigrid.cpp`
- 同步构建源: `data/quda-qio-build/source-shadow/lib/inv_gcr_quda.cpp`
- 测试: `examples/qcu/dev87/test_bench_strict_protocol.py`

**接口:**
- 消费: `QUDA_MG_TRACE_FILE` 环境变量；未设置时全部埋点关闭。
- 产出: TSV 事件 `solve_begin/solve_end`、`mg_begin/mg_end`、`stage`、`residual` 与 `outer_iteration`。

- [ ] **Step 1: 写协议失败测试**
      在 `test_bench_strict_protocol.py` 增加 QUDA MG trace 解析器夹具，覆盖 0/1/2 层号、stage、residual 与外层 iteration 重排。
- [ ] **Step 2: 运行确认失败**
      运行: `python -B -m pytest -q -p no:cacheprovider examples/qcu/dev87/test_bench_strict_protocol.py`
      预期: 新增测试因解析器不存在而 FAIL。
- [ ] **Step 3: 最小实现**
      在 `multigrid.cpp` 的 `MG::operator()` 内加入仅 trace 开启时执行的真实残差和阶段计时；在 `inv_gcr_quda.cpp` 记录 outer iteration、iterated residual、true residual。
- [ ] **Step 4: 同步并重建**
      将相同补丁应用到 `source-shadow`，执行 `cmake --build data/quda-qio-build/attempt2 --parallel N` 和 `cmake --install`。
- [ ] **Step 5: 验证**
      运行协议测试、4^4 reduction smoke、8^4 setup smoke；检查关闭 trace 时事件文件不创建。

## Task 2: PyQCU 分层残差 trace

**文件:**
- 修改: `cpp/cuda/qcu/src/apply_multigrid_strict.cu`
- 修改: `examples/qcu/dev87/trace_strict_vs_quda.py`
- 修改: `examples/qcu/dev87/analyze_strict_vs_quda_detailed.py`

**接口:**
- 消费: `PYQCU_STRICT_TRACE_FILE`。
- 产出: 现有 `stage` 事件加 `residual` 事件，包含 `outer_iteration/level/phase/rnorm/relative`。

- [ ] **Step 1: 扩展解析器测试**
      构造含残差事件的临时 trace，断言解析后每层每次外层调用的阶段顺序与残差数量。
- [ ] **Step 2: 运行确认失败**
      运行: `python -B -m pytest -q -p no:cacheprovider examples/qcu/dev87/test_bench_strict_protocol.py`
      预期: 旧解析器不识别 `residual`，测试 FAIL。
- [ ] **Step 3: C++ 最小实现**
      在 `StrictFgmresTrace` 增加 `residual`；仅在 `trace.enabled()` 时用现有 scratch 计算 compact/fine residual。
- [ ] **Step 4: 回归**
      运行 `run_strict_fast.py --tier 1 --fail-fast`，并分别在 trace 开/关下比较解和迭代数。

## Task 3: 公平收集器扩展

**文件:**
- 修改: `examples/qcu/dev87/bench_strict_vs_quda.py`
- 修改: `examples/qcu/dev87/trace_strict_vs_quda.py`
- 修改: `pyqcu/solver/_quda_multigrid.py`
- 修改: `examples/qcu/dev87/test_bench_strict_protocol.py`

**接口:**
- 消费: `--lattice XYZT`、`--levels N`、`--precision c64|c128`。
- 产出: 每条 case 的 side JSON、trace TSV/QIO hash、真实残差、外层/逐层 iteration 与 setup/solve 中位数。

- [ ] **Step 1: 写失败测试**
      测试 2/3 层配置、`generate_all_levels=false`、coarse V 逐层 restriction 和 trace marker 组合。
- [ ] **Step 2: 运行确认失败**
      运行: `python -B -m pytest -q -p no:cacheprovider examples/qcu/dev87/test_bench_strict_protocol.py`
      预期: 当前固定 `LEVELS=2` 与 cache manifest FAIL。
- [ ] **Step 3: 最小实现**
      泛化 config/cache manifest/worker geometry；PyQCU 以 `R` 传播上层 V，QUDA 关闭逐层独立 setup；加入 QUDA trace marker。
- [ ] **Step 4: 验证**
      先跑 2 层同配置回归，确认与 `stab54` 基线的残差和迭代语义可解释；再跑 3 层 smoke。

## Task 4: 多层与多精度矩阵

**文件:**
- 新建: `examples/qcu/dev87/bench_mg_matrix.py`
- 新建: `examples/qcu/dev87/test_bench_mg_matrix.py`
- 数据: `data/` 下的 canonical null-vector、QIO、cache 和矩阵结果

**接口:**
- 消费: 收集器的单 case CLI。
- 产出: `data/mg_matrix_20260916/` 下的原始 JSON、trace、CSV 与 SHA256 manifest。

- [ ] **Step 1: 准备小格 canonical/null-vector/QIO**
      使用现有 `prepare_fair_nullvec.py` 与 `convert_full_nullvec_to_quda_qio.py` 生成 8×8×8×16、16×16×16×16 资产。
- [ ] **Step 2: 写矩阵编排测试**
      断言 case 覆盖 `c64/c128`、`2/3 levels`、至少三种 volume，并拒绝输入/配置 hash 不一致。
- [ ] **Step 3: 执行 smoke 矩阵**
      每 case 先 1 warmup + 2 repeats，记录失败/跳过原因，不把 smoke 当正式加速比。
- [ ] **Step 4: 执行正式矩阵**
      对可容纳的 c64/c128、2/3 层 case 执行正式协议；结果超过设备资源时明确 skip 并保留证据。
- [ ] **Step 5: 多卡辅助测试**
      运行现有 `bench_multigpu_repeat.py`；Strict production 仍按 fail-closed 能力门禁记录为未支持，不冒充分布式 Strict 结果。

## Task 5: 优化与复测

**文件:**
- 修改: `cpp/cuda/qcu/src/apply_multigrid_strict.cu`
- 修改: `pyqcu/solver/_quda_multigrid.py`（仅在证据指向 Python setup/preconditioner 时）

**接口:**
- 消费: Task 4 的 stage/residual trace。
- 产出: 优化前后相同 case 的中位时间、迭代数、逐层耗时与残差曲线。

- [ ] **Step 1: 基线归因**
      按 `coarse solve / smoother / transfer / outer Arnoldi` 聚合 trace，列出每次外层迭代的主耗时层。
- [ ] **Step 2: 一轮一动作**
      优先验证 3 层粗化；若仍不足，再对最粗层 solver/kernel 做最小优化。
- [ ] **Step 3: 回归**
      每次改动后跑 tier 0+1，并通过同一真实解/真实残差门禁。
- [ ] **Step 4: 收敛判定**
      正式协议至少重复 5 次，PyQCU 中位时间必须显著低于 QUDA；否则继续下一轮，不虚报。

## Task 6: 报告与交付

**文件:**
- 新建: `docs/report_multigrid_quda_pyqcu_20260916.tex`
- 新建: `docs/report_multigrid_quda_pyqcu_20260916.pdf`
- 更新: `skills/benchmark/SKILL.md`
- 更新: `skills/cuda/SKILL.md`
- 更新: `skills/solver/SKILL.md`

**接口:**
- 消费: Task 4/5 的结构化结果与原始 trace。
- 产出: 可追溯的算法解析、性能表、残差图、结论与复现命令。

- [ ] **Step 1: 生成报告**
      按参考 TeX 的章节风格写入真实命令、文件行号、配置、误差和未验证项。
- [ ] **Step 2: 编译与检查**
      两遍 XeLaTeX，确保 PDF 非空、页数一致、无 Overfull/Float too large，并逐页渲染检查。
- [ ] **Step 3: diff/init/tag 前检查**
      运行全量测试、`git diff --check`、skill 索引一致性检查与标签链检查。
- [ ] **Step 4: 提交、推送与 stab 标签**
      commit message 说明 trace、三层对照、优化和验证；推送 `main` 后创建注释 `stab55` 并推送。
