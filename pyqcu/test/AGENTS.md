# AGENTS.md — pyqcu.test

`pyqcu/test/` 是**随发行包发布的正式快速测试端口**（`python -m pyqcu.test`）。
目录级规则如下；用户视角的契约见 [`README.md`](README.md)，非正式套件在
`pyqcu/testing/**`（仅 checkout，安装用户没有该包）。

## 边界（硬性）

- **不修改 core 代码**。`pyqcu/torch`、`pyqcu/cuda`、`pyqcu/kupl`、`pyqcu/lattice`、
  `pyqcu/tools`、`cpp/**`、`setup.py`、`MANIFEST.in` 不因某个用例不通过而改动；这里
  只写测试与策略。需要新行为时先证明是实现缺陷，走独立任务。
- **不 import `pyqcu.testing`**（任何方向、任何函数内延迟导入都不行）；`pyqcu.test`
  与生产代码同样互不 import：`pyqcu` 的生产模块不得 import `pyqcu.test`。
- **stdlib-only 导入**：模块顶层不得 import `torch`、`numpy`、`mpi4py`、`h5py`、
  `pytest`、`pyqcu.cann`、`pyqcu.testing`。重量级依赖只在同一次 `run()` 内部按需导入，
  `--list` 必须在这些 import 之前短路。
- **不 import pytest**。`pytest -q pyqcu/test/selfcheck.py` 是包外的显式调用，pytest
  不是运行期依赖；自检文件叫 `selfcheck.py`。
- **不得出现 `test_*.py`**。form 审计只对 `<repo>/testing/**` 免除 `TEST_LOCATION`，
  `pyqcu/test/test_*.py` 会被判为“测试在 testing/ 之外”。新文件按职责命名
  （`common.py`、`policy.py`、`units.py`、`<backend>_qcu.py`、`selfcheck.py`）。

## CLI 契约（已发布，改名=破坏用户脚本）

- 选项拼写、别名与归一化**只准**在 `policy.py` 里定义：`--lattice-size`/`--lattice`、
  `--lattice-grid`、`--mg-grid` 共用 `parse_int_list`（`8 8 8 16`、`8,8,8,16`、
  `[8,8,8,16]`）；`--dtype` 经 `DTYPE_ALIASES` 归一化（`complex64`/`complex128`/
  `float32`/`float64`）；`--backend` 经 `expand_backends`（`cpu`→`torch,kupl`、
  `dcu`→`cuda`、`npu`→`torch`、`all`→三后端）；`--device` 经 `device_for_runner`
  （`auto`→不转发，`gpu`→`cuda:0`，`dcu`→`dcu:0`，`npu`→`npu:0`，具体串透传）。
  CLI、`Options`、`Test` 和报告不得各自发明第二套解析。
- `--gauge-sigma` 与历史拼写 `--guage-sigma` 是同一选项（argparse alias），不能只保留
  一个；报告只出现 `gauge_sigma`。
- `--only` 的值在 CLI 层统一按逗号/空格拆分（`--only a,b` == `--only a b` ==
  `--only a --only b`）；拆分必须发生在 `Test.select` 之前，不能把 `a,b` 当成一个
  字面 tag。
- `--mode {list,prepare,prewarm,solve,merge,run,validate}` 是已发布的 full-mode 入口：
  模式名、`rows/reports/solutions/cache` 布局、`reports/rows.csv` 列和 `run_id` 身份
  都属于契约；改名或改布局按破坏性变更处理。
- `--csv` 在普通运行是 per-case 报告导出，在 full mode 是台账 CSV；两种语义不能合并。
  `--junit`/`--markdown` 始终是报告导出，`--list`/`--mode list` 下拒绝。
- `--ledger-export {none,csv,excel,both}` 是 full-mode 唯一的**参考导出**入口：只写
  `<outdir>/ledger/kupl_kunpeng_all.csv`（参考 100 列）与
  `<outdir>/ledger/kupl_kunpeng_performance.xlsx`（参考 21 sheet），绝不覆盖
  `reports/rows.csv`（30 列 `ROW_FIELDS` 语义不变）。默认 `none`；无 `--mode` 与
  `--mode list` 下是用法错误（退出 `2`），不得改成静默忽略。正式端口观测不到的列必须是
  空单元格，禁止用参考快照补默认值；新增/改列要同步 `ledger_csv.COLUMN_RULES`
  （workbook 的 `17_Data_Dictionary` 由它生成）。
- `--resume` 只复用 `PASS`/`SKIP`；`--skip-failed` 额外复用 `FAIL`（此后不再重试）；
  `ERROR`/`TIMEOUT` 永远重试。`--run-label` 只能作为 `run_id` 后缀，不得含路径分隔符。
- `--gauge-file`/`--source-file` 是 full-mode 的显式 HDF5 输入（dataset `data`，
  canonical `xyzt` 布局，见 README「HDF5 输入」）。`--native` 规划的原生行由
  `pyqcu.test.driver.run_row` **真的消费**这两个文件，并必须在行里写
  `gauge_origin`/`gauge_sha256`/`source_origin`/`source_sha256`；内置 tier-0 runner
  不得用文件替换 `--gauge`/`--source`（那会变成“证据与数值不一致”）。
- `--mass` 与 `--kappa` 是同一物理量（`kappa = 1/(2*mass + 8)`）。两者都**没有** CLI
  默认值：未给出就不出现在 `Options.as_kwargs()` 里，由 runner 自己的默认决定；不要
  在 CLI 里补一个“统一默认”去覆盖 runner（tier 0 torch 默认 `kappa = 0.125`）。
- `--solver`/`--dslash` 既随 `Options` 转发给 runner，又**筛选用例矩阵**：只筛 solve/mg
  与 `kind=="dslash"` 的用例，规则在 `policy.SOLVER_TAG_FILTERS`/`DSLASH_TAG_FILTERS`。
  要新增算法覆盖，新增 tag 并登记 `units.CASES`，不要复用旧 tag 改语义。
- `--quick` **只锁 tier 0、格点 `8 8 8 16`、`c64`、`seed 42`、`repeat 1`**；不得把
  `--device`/`--solver`/`--dslash`/`--mass`/`--csw`/`--gauge`/`--source`/网格/线程等
  加进冲突列表。`--lattice-grid` 乘积与 MPI size 不一致**不是**用法错误。
- `--atol` 只覆盖 `compare` 门；不得让它放宽 `solve`/`native_solve`/`gamma`。
- 退出码固定 `0`/`1`/`2`：数值失败是 `1`，用法/环境错误是 `2`，全 SKIP 默认 `1`。
  与 `pyqcu --selftest`（FAIL 退 2）不同，改动前先读 `policy.EXIT_CODES`。

## 用例与报告

- 用例 tag 固定 `<backend>/<area>/<variant>` 三段式，且必须以 `"<backend>/"` 开头
  （`Case.__post_init__` 校验）；当前 19 个 tier-0 用例（含 `torch/gauge/random`、
   `torch/smear/wuppertal`、`torch/solve/cg`、`kupl/dslash/clover`、`kupl/solve/cg`）。
   `--only`/`--solver`/`--dslash`/`--list` 都按它匹配，改名等于破坏已发布的 CLI。
- runner 字符串是 `<module>:<function>`（如 `torch_qcu:run_lattice`）；模块名不含点时
  按 `policy.RUNNER_PREFIX`（`pyqcu.test`）解析。`policy.TIER0_RUNNERS` 与 `units.CASES`
  是同两份声明，`selfcheck` 要求它们逐项一致——改 runner 名必须两处同改。
- 状态只有 `PASS`/`SKIP`/`FAIL`/`ERROR`/`TIMEOUT`：runner 抛异常 → `ERROR`，
  `TimeoutError` → `TIMEOUT`；不要发明 `XFAIL`/`WARN`/`OK`。
- runner 只返回**观测值**；PASS/FAIL 一律由 `Case.expect` 判定，禁止 runner 内提前断言
  数字或把阈值写进 runner。`expect` 只能引用 `policy.RESIDUAL_GATES` 的具名门
  （`compare`/`solve`/`native_solve`/`gamma`），不得在 `units.py` 里写裸阈值。
- 每个 SKIP 必须带非空 `reason` 和词表 token（`kupl_library`、`cuda_bridge`、
  `cuda_device`、`mpi_single_rank`、`h5py_mpi`、`mpi_grid`、`shm_unavailable`、
  `device_unavailable`）；空 reason 按缺陷处理。`mpi_grid`/`shm_unavailable`/
  `device_unavailable` 是**条件** token，不写进 `Case.requires`。
- 报告 `options` 记录本次生效的全部配置（含 `runner_device`、`mass`/`kappa` 的 `null`
  与 `argv`），`--list --json` 与 `--json` 必须能复现同一次运行；不要在别处另存一份。
- **`--threads` 的生效边界**：tier-0 torch runner 调 `torch.set_num_threads`；
  `--native` 行由 `driver.run_row` → `native.apply_environment` 设置
  `OMP_NUM_THREADS`/`KUPL_OMP_THREADS`/`OMP_DYNAMIC`（行里给
  `threads_requested`/`threads_applied`/`environment`）。tier-0 的
  `kupl_qcu.py`/`cuda_qcu.py` 仍不读取它，那些行只能说“记录”。
- 输出约定：stdout 每用例一行、列宽稳定可解析；metrics 与失败 gate 只在 `--verbose`
  下写 stderr。runner 的调试打印不得混进 stdout 的报告行（`--json` 复用同一份报告对象）。

## Full mode（工作台账）

- `--mode` 是编排层，不是第二套 runner：`solve` 的每一行必须走与单用例运行相同的
  `Test.run`/`Case.expect` 判据；不要为 full mode 复制阈值或另写 PASS/FAIL 判定。
- `rows/<run_id>.json` 是唯一事实源；`reports/rows.csv`/`merged.json` 是派生视图，
  必须能从 rows 重建（`merge`）。`validate` 必须同时核对 rows、每行报告与 CSV。
- `--mode list` 不写台账（行 `status` 保持 `null`）；`--mode prepare` 只创建布局、校验
  `--gauge-file`/`--source-file`、探测已有 cache，**不重建 cache**；`prewarm` 只读输入
  并哈希，不跑数值。
- `--ranks`、`--threads`、`--shm`/`--no-shm`、`--shared`、`--gauge-file`/`--source-file`
  记录进 rows/`--list --json`；只有确实被 runner 消费的字段才能描述为“已应用”
  （tier-0 只有 torch 行应用 `--threads`；`--native` 行应用 threads/shm/输入文件）。
- `--native {dslash,solve,mg}` 追加 tag 为 `<backend>/native/<kind>` 的 tier-0 行，由
  `pyqcu.test.driver.run_row` 执行 `pyqcu.test.native`。该钩子**只返回观测值**，不得返回
  `PASS`/`FAIL`（harness 会把 driver 的 `PASS`/`FAIL` 记成 `ERROR`）；非原生 tag 一律
  返回 `None`，让内置执行器照旧执行。跨节点 `mpirun`/rankfile 仍未实现，只能报
  `SKIP(mpi_grid)` 并写 `launcher` 证据，不得在小规模上“代跑”出结果。
- full mode 未显式给 `--tier` 时默认 tier 1；`--quick` 仍锁 tier 0。`--validate`
  单独出现等价于 `--mode validate`，与 `solve`/`run` 同时出现是后置校验。
- 默认工作根 `pyqcu-test-runs/`（相对当前目录）；新增/改名路径必须同步 README、
  `--list --json` 与 `docs/ORGANIZATION.md` 的发行边界。
- `--ledger-export` 的 `<outdir>/ledger/` 是**派生的第二份表**，不是台账事实源：它必须
  能由 rows/`--json` 文档重建（`export_ledger`/`export_from_payload`），`merge` 之后重导
  必须得到同样的表；写失败按环境错误退出 `2`，不得把半份文件留在原地（temp + rename）。

## 数值与证据

- solve 用例的真残差必须独立重构 `||D x - b|| / ||b||`；复用求解器内部递推即无效。
  CG 用例按 γ5-hermitian 约定求解，但残差仍在全 Wilson 算子上重构。
- 默认格点 `[8,8,8,16]`、`c64`、`seed 42`、tier 0、单 rank、不读数据文件；`8x8x16x16`
  是发布验收格点，不是本端口的默认门禁。
- **`--quick` 不产生性能结论**：不得据此写加速比、回退或机器对比；`seconds` 只是诊断。
  中位数要求 `--repeat >= 5` 且 `warmup >= 1`，否则只报单次观测。
- **CUDA 在本 checkout 预期 SKIP**：`pyqcu/cuda/qcu/` 没有构建出的 `qcu*.so` 时，
  用例必须报 `SKIP(cuda_bridge)`（带 reason），不是 FAIL；不得伪造 PASS、不得静默
  回退到只跑其他后端、也不得为变绿去构建桥。
- KUPL 缺库/缺符号同样是 `SKIP(kupl_library)`；ABI 用例只读符号集合，不改全局 ABI 状态。

## 改动检查

- 至少运行 `python -m pyqcu.test --list`、`python -m pyqcu.test --list --json`、
  `python -m pyqcu.test --quick`、`python -m pyqcu.test --quick --json <scratch 路径>`，
  并确认 `--list` 不拉起重型 import；改了选择/映射逻辑再加跑
  `--backend cpu --list`、`--solver cg --list`、`--dslash clover --list` 与一条拒绝路径。
  改了 full mode 再加跑（用 scratch 目录，不要写仓库）：
  `--mode list --json <scratch>`、`--quick --only torch/lattice --mode prepare|solve|merge|validate
  --outdir <scratch>`，以及 `--resume --skip-failed --run-label cold` 的第二次 solve。
  改了导出再加跑：`--mode merge --outdir <scratch> --ledger-export both`，确认
  `<scratch>/ledger/` 里是 100 列 CSV + 21 sheet workbook，且 `reports/rows.csv` 仍是
  30 列 `ROW_FIELDS`。
- 文档/策略改动另跑 `bash skills/pyqcu-form/form-audit.sh`、
  `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`、`git diff --check`。
- **Python 3.10 语法门**：`--check` 的 `py310_syntax` 用
  `ast.parse(feature_version=(3, 10))` 解析每个 `pyqcu/test/*.py`，并结构化拦截
  PEP 646 星号下标/注解（`x[a, *b]`、`def f(*args: *Ts)`；3.12 的
  `feature_version` 不拦它们，`x[(a, *b)]` 是合法元组显示，不算违规）。新增 3.11+
  语法会 FAIL 并报 `file:line`；`feature_version` 不可用时只报 SKIP（不得伪造 PASS）。
  改判据请在 `pyqcu/testing/python/test_pyqcu_test_syntax_floor.py` 同步回归。
