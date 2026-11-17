---
name: pyqcu-testing
description: 当任务涉及 pyqcu.testing 非正式集成测试、正式 pyqcu.test 快速端口、发布验证（release gate）、Strict MG 快速闸门、MPI/缓存/QIO 或显存验证时使用。
---
# pyqcu.testing

Integration tests for all PyQCU components. Tests are Python functions imported by `pyqcu/testing/*/conftest.py` entry points.

CUDA/DCU/GPGPU test launchers follow one Python process = one MPI rank = one
device. Tests that explicitly construct `MultiGpuMultigrid` must be labeled as
test-only consistency/throughput evidence and must not become production
templates. See
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology).
The migrated example paths and standard test commands are summarized in
[`pyqcu-docs/references/extracted-guidance.md`](../pyqcu-docs/references/extracted-guidance.md#2-测试与示例入口).
L48 双节点真实 LIME/HDF5 的操作清单见
[`references/l48-remote.md`](references/l48-remote.md)：远端 `login02` 基线已验证，
单节点 c64 已有结果（32r `1.6646 s`），但 c128/两节点仍 PENDING、CG 组退役。
本轮经验速查（0.1.17 覆盖树 18/18 vs 本机 19/19、3.10 PEP 646 语法门、dtype/读取放大、
183 s 病态不作基线、16r “MG” 实为 fallback、cache identity 与 launcher 修复）见
[`references/remote-l48-experience.md`](references/remote-l48-experience.md)，完整证据与
链接见 [`docs/kupl_remote_l48_experience_20261117.md`](../../docs/kupl_remote_l48_experience_20261117.md)。

## Architecture

All test functions live in `pyqcu/testing/__init__.py`. They import from all PyQCU subpackages (`lattice`, `solver`, `dslash`, `tools`, `smear`). Each `pyqcu/testing/*/conftest.py` acts as a pytest entry point that imports specific test functions and calls them. The conftest files are manually edited to uncomment the test(s) to run.

The module imports `tilelang` at module level (with try/except fallback) for `test_matmul`.
The historical `pyqcu/testing/tilelang/conftest.py` no-op file has been removed; it
previously exited `0` without executing `test_matmul()`. Require real pytest collection
and assertion output before reporting a TileLang test pass.

## Test Functions

### `test_lattice(lat_size, dtype, device)`
Tests SU(3) gauge generation + gamma matrix algebra.
- Generates random gauge field, runs `check_su3`
- Verifies γ_μ² = I for all 4 gamma matrices
- **Assertion:** `check_su3` must return True

### `test_dslash_wilson(kappa, lat_size, dtype, device, with_data, support_parallel)`
Tests Wilson Dirac operator.
- `with_data=False`: Generates random gauge field + source, applies full Wilson operator and eo/oe preconditioned variants
- `with_data=True`: Explicitly selects the historical `16x16x16x16` fixture (`refer.wilson.*.L16K0_125.*.h5`); it must not silently override the caller lattice
- **Assertion:** Relative difference < 1e-4

The standard cross-backend correctness lattice is `8x8x8x16`; `16^4` and larger
volumes are performance/capacity or explicit historical-reference inputs only.

### `test_dslash_parity(lat_size, kappa, dtype, device)`
Tests parity-preconditioned Wilson+Clover operator with MPI.
- Distributes gauge field across MPI grid
- Root rank computes full operator result as reference
- All ranks compare local parity-preconditioned operator against reference
- Tests both `matvec_all` and `matvec_eeo`/`matvec_oeo` paths

### `test_dslash_clover(device, with_data, dtype)`
Tests Clover term construction.
- `with_data=True`: Loads reference data, validates clover term and inverse against known results
- `with_data=False`: Tests parallel vs serial clover construction across MPI grid

### `test_solver(kind, method, kappa, lat_size, dtype, device, with_data, max_level, num_restart, support_parity)`
Tests BiStabCG and multigrid solvers.
- `method='bistabcg'`: Standard or parity-preconditioned BiCGStab
- `method='multigrid'`: Full multigrid V-cycle with `init()` + `solve()` + `plot()`
- `with_data=True`: Validates against reference Wilson data
- **Assertion:** Relative error < 1e-3

### `test_matmul()`
Benchmarks TileLang JIT-compiled matrix multiply vs PyTorch (cuBLAS/MKL).
- GPU: 4096×4096 matmul, TileLang vs cuBLAS
- CPU: 1024×1024 matmul, TileLang (LLVM or C backend) vs MKL/OneDNN
- Prints TFLOPS comparison table

### `test_smear_stout(lat_size, device, dtype)`
Tests stout smearing across MPI grid.
- Distributes gauge field, root computes whole-grid reference
- All ranks compare local parallel smear against reference
- Verifies SU(3) before and after smearing

### `test_smear_wuppertal()` (test16, 2026-08-24)
Wuppertal Gaussian smearing with triple invariants (cpu+cuda both PASS):
- `nstep>=1` guard; U=I fixed point (<1e-4); white-noise contraction ratio < 1.0.
- Golden criterion: np=2/4 constant-source rel≈5e-08.

### `verify_nullvecs()` — block structure required
Null-vector quality diagnostic requires the explicit 10-dim block structure argument
(documented in dev85); non-block layouts fail fast instead of being auto-corrected.

## Running Tests（非正式套件）

```bash
(cd pyqcu/testing && pytest .)                             # all conftest.py files
mpirun -np 4 python pyqcu/testing/python/conftest.py       # single file with MPI
```

## 正式端口 `pyqcu.test` 与非正式套件 `pyqcu.testing`

测试代码只有两个规范位置，按“是否随包发行”划分，不按文件新旧划分：

| 端口 | 路径 | 发行 | 依赖与范围 |
|---|---|---|---|
| 正式 | `pyqcu/test/**` | 随 sdist/wheel（`recursive-include pyqcu *.py`） | stdlib-only 导入，安装态可跑；19 个 tier 0 跨后端正确性门，默认格点 `[8,8,8,16]`、c64、单 rank、无数据文件；full-mode 工作台账 `--mode list/prepare/prewarm/solve/merge/run/validate` |
| 非正式（本技能） | `pyqcu/testing/**` | 不进 sdist（`prune pyqcu/testing` + `find_packages(exclude=["pyqcu.testing", "pyqcu.testing.*"])`） | 模块级 import `torch`/`mpi4py`/`pytest`；集成、回归、性能、QUDA/HPCKit 矩阵与历史资产 |

```bash
python -m pyqcu.test --list                      # 只列用例，不 import torch/native
python -m pyqcu.test --quick                     # tier 0 + 8 8 8 16 + c64 + seed 42
python -m pyqcu.test --quick --json -            # JSON 报告写 stdout
python -m pyqcu.test --only torch/mg/legacy --repeat 1 --warmup 1
python -m pyqcu.test --backend cpu --dtype complex128 --solver cg --list
python -m pyqcu.test --mode list --json -        # full-mode 台账行，不执行
```

正式端口的展开选项（默认值与判据见 README）：`--lattice-size`/`--lattice`、
`--lattice-grid`（MPI 网格）、`--mg-grid`、`--lattice-preset L24|L32|L48`、
`--gauge random|unit|ones`、`--gauge-sigma`/`--guage-sigma`、`--source point|ones|random`、
`--mass`/`--kappa`（`kappa = 1/(2m+8)`）、`--csw`、`--max-iter`、`--restart`、
`--mg-levels`、`--mg-nu-pre`/`--mg-nu-post`、`--device auto|cpu|gpu|dcu|npu`、
`--device-id`、`--threads`、`--solver all|bicgstab|cg|multigrid|mg`（只筛 solve/mg）、
`--dslash all|wilson|clover|both`（只筛 dslash）、`--backend` 别名
`cpu`=torch+kupl、`dcu`=cuda、`npu`=torch、`--dtype complex64|complex128`、`--atol`、
`--profile quick|cpu|gpu|c128|clover|mg|full`、`--sweep`、`--native dslash|solve|mg`、
`--junit/--csv/--markdown`；
`--quick` 只锁 tier/格点/dtype/seed/repeat。

- `--threads` **只在 torch runner 真正应用**（`torch.set_num_threads`，metrics 里
  `threads_applied`/`torch_threads`）；tier-0 的 `kupl_qcu.py`/`cuda_qcu.py` 不读取它，
  只把它记录进 `rows`/`options`。不要把 `--threads` 写成 KUPL/OpenMP 的已转发参数；
  `native.py` 的 `apply_environment` 由 `--native` 行经 `driver.run_row` 接线（设置
  `OMP_NUM_THREADS`/`KUPL_OMP_THREADS`，行里给 `threads_requested`/`threads_applied`）。
- full mode（`--mode`）是编排层，不是第二套 runner：row 仍走同一 `Test.run`/`Case.expect`。
  `rows/<run_id>.json` 是事实源，`reports/rows.csv`/`merged.json` 是派生视图；
  `--resume` 复用 `PASS`/`SKIP`，`--skip-failed` 额外复用 `FAIL`（不再重试），
  `ERROR`/`TIMEOUT` 永远重试；`--run-label` 是 `run_id` 后缀。默认工作根
  `pyqcu-test-runs/`（可用 `--outdir`/`--rows`/`--reports`/`--solutions`/`--cache-dir` 覆盖）。
  `--mode list` 不写台账；`prepare` 只校验输入/探测 cache（不重建 cache）；`prewarm`
  只读并哈希输入；`merge`/`validate` 对空台账返回 1。`--csv` 在 full mode 是台账 CSV，
  不是报告导出。
- HDF5 输入（full mode `--gauge-file`/`--source-file`）：dataset `data`，仅
  `complex64`/`complex128`，canonical `xyzt` 布局（gauge `[3,3,4,x,y,z,t]`，source
  `[4,3,x,y,z,t]`；parity-compressed 加前导 `2` 且 `t/2`，必须带 `layout` 属性）。
  `prepare` 校验、`prewarm` 读取并哈希。消费方有两条：`--native` 行由
  `pyqcu.test.driver.run_row` 真的读文件（校验 + SHA256 后交给 `native`，行里写
  `gauge_origin=hdf5`/`gauge_sha256`），内置 tier-0 runner 不会用文件替换
  `--gauge`/`--source`。文件缺失/形状不符是输入 `FAIL`（退出 1），不是用法错误；
  未请求的输入是 `SKIP`。跨节点 `mpirun`（rankfile/`CCS_ALLOC_FILE`）仍是 TODO：
  `--native` 行在 `--ranks` 与 `MPI_COMM_WORLD` 不符时报 `SKIP(mpi_grid)`。

- 退出码：`0` 无 FAIL/ERROR/TIMEOUT；`1` 任一失败**或全部 SKIP**（除非
  `--allow-all-skipped`）；`2` 用法/环境错误。`pyqcu --selftest` 的 FAIL 退 2，两者不可混读。
- **新增正式用例只写 `pyqcu/test/units.py`**；本目录不再新增“正式”用例，只保留探索性、
  研究性、性能性和历史回归资产。两个方向都不得互相 import，生产模块也不得 import `pyqcu.test`。
- SKIP 必须带非空 reason（`kupl_library`/`cuda_bridge`/`cuda_device`/`mpi_single_rank`/
  `h5py_mpi`/`mpi_grid`/`device_unavailable`）；当前 checkout 没有 `pyqcu/cuda/qcu/*.so` 时
  CUDA 用例报 `SKIP(cuda_bridge)` 是预期结果，不是 FAIL；`prod(--lattice-grid)` 与 MPI size
  不一致时原生用例报 `SKIP(mpi_grid)`（torch 用例不读网格），加速器族不可用报 `device_unavailable`。
- `pyqcu/test` 内不得出现 `test_*.py`（form 审计只在 `<repo>/testing/**` 内免除 `TEST_LOCATION`），
  也不 import `pytest`；自检文件是 `pyqcu/test/selfcheck.py`。
- `--quick` 只做正确性：不得据此报性能、加速比或回退。
- 安装态没有 `pyqcu.testing`，用户侧验证只用 `pyqcu.test` 或公开 API；入包由发布门正向条目
  `sdist ships pyqcu.test` 复核，判据见 [`pyqcu/test/README.md`](../../pyqcu/test/README.md)。

## 发布验证（release gate）

发布（PyPI 上传、RTD 文档、发布日志）前的固定闸门，全部在仓库根执行；任一项失败即停发布，
不得用上一轮结果替代。本次发布的完整记录见 `logs/v20261109.txt`。

```bash
python -m pyqcu.testing.release_check --offline      # 本次 13/13
python -m pyqcu.testing.docs_examples --keep-going   # 30 ok / 8 skipped / 0 failed
python -m pyqcu --selftest --json                    # Torch/KUPL/CUDA 三层
python -B -m pytest -q -p no:cacheprovider \
  pyqcu/testing/dcu/test_install_selection.py \
  pyqcu/testing/kupl/test_pure_kupl_install.py       # 打包契约 9 passed
python -B -m pytest -q -p no:cacheprovider \
  pyqcu/testing/kupl/test_kupl_python.py \
  pyqcu/testing/kupl/test_krylov_api.py \
  pyqcu/testing/python/test_api_contract.py \
  pyqcu/testing/python/test_krylov_api.py            # KUPL/API 29 passed
```

`release_check --offline` 的 13 项为 version、sdist exists、sdist no testing、sdist ships
pyqcu.test、sdist no binaries、sdist native sources（KUPL+CUDA 源码）、sdist no home paths、
sdist Chinese README、twine check、bilingual blocks、docs examples、Sphinx en `-W`、Sphinx zh `-W`。
检查项随打包契约增长（0.1.0 为 11/11，0.1.1 增加 `sdist Chinese README` 后为 12/12，端口随包后为 13/13），
汇报时引用真实输出，不照抄历史分母。前提是先构建 `dist/pyqcu-<ver>.tar.gz` 并装好
`twine`，否则 sdist/twine 项必然失败；`release_check.py` 以 `parents[2]` 定位仓库根，
只能在源码 checkout 里跑。`--online` 追加 PyPI JSON/simple 索引与 RTD en/zh-cn 探测；
网络不可用时只跑 `--offline`，并把 PyPI 传播与 RTD 构建列为未验证项。`--skip-sphinx`
只用于确认文档未改动的轮次。

`docs_examples` 要求每个 Python 代码块自带 `assert`，否则必须显式标注 `# docs-skip` 或
`# pseudo-code`；同页代码块共享命名空间并按页内顺序执行。Sphinx 构建通过不等于文档示例
可执行，两条闸门都要跑。

### 8x8x16x16 complex64 验收

发布基准格点固定为 `8x8x16x16`、`complex64`、`mass=0.17`、`csw=1.0`、`atol=1e-6`、
Gaussian gauge `sigma=0.1 seed=42`、source `seed=42`：

```bash
python pyqcu/testing/simple/kupl_bistabcg.py \
  --lat 8 8 16 16 --dtype c64 \
  --mass 0.17 --csw 1.0 --atol 1e-6 --max-iter 1000 --seed 42 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --report data/release_8x8x16x16/report_kupl.json
python pyqcu/testing/simple/qcu_bistabcg.py \
  --lat 8 8 16 16 --mass 0.17 --csw 1.0 --warmup 0 --repeats 1 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --device cuda:0 --report data/release_8x8x16x16/report_cuda.json
```

本次实测（WSL2，i7-13620H，RTX 4060 Laptop 8 GB `sm_89`，CUDA 12.4）：

| 后端 | 迭代 | 独立真残差 | solve wall | 状态 |
| --- | --- | --- | --- | --- |
| KUPL（1 rank × 4 OpenMP） | 26 | 2.28e-7 | 0.13 s | PASS |
| CUDA（1 进程 = 1 rank = 1 设备） | 26 | 2.96e-7 | 0.05 s | PASS |

KUPL 行取安装包公开 API 的结果；仓库驱动的独立 NumPy 全残差在同机为 `2.21e-7` / `0.13 s`。
验收线是 `final_true_residual <= max(atol, 2048*eps32) ≈ 2.4414e-4`，不是 `atol` 本身；
同时要求 `status == "PASS"` 且 `0 < iterations <= max_iter`。真残差必须由独立的全格点
NumPy/Torch 重构给出，不能复用求解器内部递推。CPU 与 GPU 不合并成单一加速比；
墙钟跨机器比较必须先在报告中固定硬件与环境身份（见下节），身份缺失只能标「不可比」。

### 复用既有入口（强制，先于一切测试实现）

**先盘点，再动手。** `pyqcu/testing/simple/` 已经是全部测量场景的规范实现，禁止另起
临时脚本重拼算子调用：

| 场景 | 既有入口 |
|---|---|
| 单次生产求解（含独立真残差、原子写解） | `simple/solve_case.py --backend cuda\|quda\|kupl` |
| 三后端对照测量 | `simple/compare_three_way.py` |
| 单后端 BiCGStab（含 `--gauge/--source/--mass/--csw/--warmup/--repeats/--report`） | `simple/{kupl,qcu,quda}_bistabcg.py` |
| 单后端 MultiGrid（含 `--levels/--dof/--mg-grid/--coarse-cache/--restart`） | `simple/{kupl,qcu,quda}_multigrid.py` |
| 输入准备与布局转换 | `simple/prepare_case.py`、`simple/_common.py`（`read_qdp_gauge`/`qdp_gauge_to_qcu`/`resolve_path`/`estimated_clover_gflops`） |
| MPI 编排 | `simple/mpi/compare_*.py`、`simple/mpi/_{common,compare}.py` |

这些入口已经处理好 layout 转换、parity Clover 资产（`ce/co/cei/coi`）、
`_SET_PLAN_` 生命周期、`give_wilson`/`give_clover` 的 kappa/`u_0` 设备一致性、
独立全格点真残差与报告字段（`iterations`/`wall_samples`/`final_true_residual`）。

只有在下列条件之一成立时才允许新写脚本，并且必须在结论里写明具体缺口与复现证据：

1. 现成入口确实不支持所需配置（先读它的 `--help` 与 argparse 定义，不要凭印象假设）；
2. 需要新增测量维度，且已以向后兼容方式扩展既有入口（首选）而非另建平行实现。

绕开既有入口手工拼装 `QcuRuntime` + `apply*Qcu` 调用属于**缺陷**，不是捷径。本库已因
这种绕路得到过三份错误结论：Clover 参数未传入导致算子与组态不一致、单 parity Schur
在偶点源上退化为零 RHS、`make_clover` 之后 `_SET_PLAN_` 停在 2 触发原生 `MPI_ABORT`。

### 测试源约定（强制）

- **性能测试统一点源**：`pyqcu.lattice.point_source(lat, [0,0,0,0])`（单时空点、spin/color 全填），
  再经 `pyqcu.tools.oooxyzt2poooxyzt` 转 parity-split 布局喂给原生算子。点源激发全部本征模、
  迭代次数更多、更接近真实生产，且源条件完全固定。
- **正确性测试用「点源 + 随机源」双源**，且固定在小格点（当前门禁格点 `8x8x8x16`）：
  点源覆盖全谱收敛路径，随机源暴露与源结构无关的实现缺陷。两种源都要有独立真残差。
- 报告必须记录：源类型、源坐标/seed、gauge 身份（文件或 seed+sigma）、`mass`/`csw`。
  **点源与随机源的迭代数不可互相比较**，混用即视为配置不一致。
- 墙钟比较需要完整硬件身份：GPU 型号与 compute_cap、CPU 型号与核数、NUMA、`OMP_NUM_THREADS`、
  编译架构（`sm_XX` 与 cubin/PTX）、库 SHA256、驱动与 CUDA 版本、warmup/重复次数与 median/min/max。
  写明身份后墙钟可以跨机器比较；缺少任一项时只能标「不可比」，不得据此宣称加速或回退。

### 干净 venv 复验（用户视角）

发布后在**源码树之外**新建 venv，只从 PyPI 安装：

```bash
python -m venv /tmp/venv-pypi && . /tmp/venv-pypi/bin/activate
pip install --no-cache-dir --index-url https://pypi.org/simple pyqcu==<ver>
cd /tmp    # 必须在源码树之外
python -c 'import pyqcu; assert "site-packages" in pyqcu.__file__, pyqcu.__file__'
python -m pyqcu
pip install --no-cache-dir --force-reinstall --index-url https://pypi.org/simple pyqcu==<ver>
pip check
```

- `python -m pyqcu` 必须报出 `site-packages/pyqcu/...` 与包内 KUPL/Cython 库；回落到源码
  树路径即视为安装验证失败。
- `pyqcu --selftest` 是安装态第一命令：Torch 必跑，KUPL/CUDA 缺库为 SKIP，
  整体 PASS 才退出 0；`--json` 可机器读取。CUDA Wilson 的
  `iterations=None` 是合法结果，不是失败。
- `mpirun -np N pyqcu --selftest --json` 只由 rank 0 输出一个 JSON；native
  层在 MPI size>1 时 SKIP 并注明需要单 rank。`pyqcu --json` 同样 rank-0
  only，避免多份 JSON 拼接。
- 非零 rank 虽不输出，但必须返回真实状态码（PASS 0 / FAIL 2）；无条件
  `return 0` 会让 `mpirun` 掩盖该 rank 的 Torch 失败。0.1.5 有对应单测。
- `mpirun -np N pyqcu --version/--help` 只由 rank 0 输出。rank 在 argparse
  之前从 `OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`I_MPI_RANK`/`MV2_COMM_WORLD_RANK`/
  `SLURM_PROCID` 预扫描，不 import mpi4py；0.1.8 起 15 项 CLI 测试覆盖。
- 自检格点下限为 4；0.1.6 在 `2 2 2 2` 上 CUDA 残差 42.7，0.1.7 起接口
  直接拒绝。发布验证用默认 `4 4 4 4` 或 `8 8 16 16`。
- 自动续跑注入失败时先看 `tui-watch.log`：`idle-nudge-failed detail=FileNotFoundError
  command=--` 表示命令 JSON 首项 `--` 被当成可执行文件。修复生成端
  （`~/configure/bin/agent.sh` 的 `python3 -c ... --` 去掉 `--`）并在
  `agent-tui-watch.py::render_recovery_command` 防御性剥离前导 `--`；回归
  `agent-tui-watch-idle.test.py` 8 项、`agent-tui-watch.test.py` 9 项。
- 10s 监督需同时设 `--idle-seconds 10` 和 `--idle-min-turn-seconds 10`；
  后者默认 30s，会把首次注入推迟到回合完成后 30s。`--idle-poll 2` 足够响应。
- command-code TUI 不会刷新外部 `-p --session` 写入的回合：`idle-nudge-ok`
  只证明 headless 回合执行成功，TUI 里可能看不到该分支。排查看
  `tui-watch.log`、`idle-nudge-last.txt` 和 `~/.commandcode/projects/...jsonl`；
  要在 TUI 显示该回合需 `/reload`。不要把“TUI 无新消息”误判为注入失败。
- 并发 nudge 事故：`agent-session-state.py` 曾把含 `tool_use` 的中间 assistant
  当作回合完成，watcher 在主回合工具调用间隙每 10s 注入；`last_nudge_at` 又在
  命令启动前写入，长 nudge 返回后立即再次触发。修复为只有“无 tool_use 且有
  最终 text”的 assistant 才算完成；nudge 返回后再重置空闲时钟；
  `last_user_input` 忽略 `【自动续跑】` 消息。回归 session-state 16、idle 8、
  watcher 9；重启后主回合只记录 `idle-skip reason=turn_running`。
- 停 watcher 必须杀整个进程组（它用 `setsid` 启动时 PGID=watcher PID）：
  `kill -TERM -- -<watcher_pid>`。只杀 watcher PID 会留下 `command-code`
  nudge 子进程继续改文件/上传包；0.1.9 的半成品发布就是这样产生的。
- `mpirun` 返回 135 / `Bus error` 且无 stdout 时先查 `df -h /tmp`；`/tmp` 满会
  让 OpenMPI 共享内存池失败。设置 `TMPDIR`/`OMPI_MCA_orte_tmpdir_base` 到可写
  目录、清理 `/tmp/ompi.*` 遗留，或纯 CPU 诊断加 `OMPI_MCA_btl=^smcuda`。
  0.1.8 MPI 契约：自检 8×8×16×16 单份 JSON/Torch PASS/rc0，2⁴ 单份 FAIL/rc2，
  `--json` version/size/rank 正确。
- CUDA bridge 测试不能只按 `torch.cuda.is_available()` 跳过；源码树里
  `pyqcu.cuda.qcu` 可能只是空 namespace。必须检查 `hasattr(qcu,
  "applyInitQcu")`，缺失时 `pytest.skip("CUDA QCU bridge is not built")`。
  全量 `pyqcu/testing/python` 当前 143 passed / 4 skipped。
- `--force-reinstall` 复验安装幂等；`pip check` 无输出才算依赖闭合。
- 在仓库根直接 `python -c 'import pyqcu'` 只会命中源码树（`pyqcu.__file__` 不含
  `site-packages`），不得当作安装证据。
- `<ver>` 取 `pyproject.toml` 与 `pyqcu.__version__` 的一致值；venv 也可建在
  `data/pypi/venv-<name>`（`data/.gitignore` 已忽略），不污染仓库状态。

### 测试入口的边界

- `pyqcu/testing/**` 不进 sdist（`release_check` 的 `sdist no testing` 硬门），安装态用户
  没有 `pyqcu.testing`；随包交付的“用户可跑”检查是 `pyqcu.test`（`python -m pyqcu.test --quick`，
  入包由 `sdist ships pyqcu.test` 正向复核），其余发布后验证必须用公开 API 另写。
- `data/pypi/verify_installed.py` 与 `data/pypi/verify_cuda.py` 只是本地发布操作者的验证
  脚本（`data/.gitignore` 忽略、不进包、不是包内 API）：不在文档或技能中承诺其存在，也
  不从包内 import。
- 打包契约与 KUPL/API 回归必须在源码树运行（`pyqcu/testing/**` 只存在于 checkout）；
  安装态验证另走上面的 venv 路径，两者不能互相替代。

## Strict MultiGrid Fast Gates

Strict-MG 快速闸门分三层（tier 0 CPU 代数 smoke、tier 1 CUDA 小格、tier 2 真实
gauge + QUDA 正式门），由
`pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py` 执行；tier 累积，
默认 tier 0。最短编辑回路先 `--list`，再用 `--fail-fast --json <path>` 固化结果：

```bash
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --list
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --fail-fast --json strict-fast.json
```

完整的 tier 契约、QUDA/PyQUDA 构建与参数回读要求、Strict runtime-cache/内存判据、
MPI preflight 与 2026-09 回归记录见
[`references/strict-mg-fast-gates.md`](references/strict-mg-fast-gates.md)。
涉及 ABI、cache/schema v2、QIO 或显存断言时先读该参考，不要把 tier 0 的
“20 checks”当成当前分母。

## Logging Convention

All test output uses: `PYQCU::TESTING::<MODULE>::\n message`

## Important Notes

- Tests use `tools.local_xyzt2whole_xyzt` / `tools.whole_xyzt2local_xyzt` for MPI reference comparison
- Reference HDF5 data lives in `pyqcu/testing/data/`
- The `path` variable in tests is computed from `pyqcu.__file__` to locate data files
- **R3 fix:** Tests now include `assert` statements so pytest can detect failures

## 分层 trace / matrix tests（2026-09-16）

`test_bench_strict_protocol.py` covers variable `--lattice/--levels`,
recursive cache manifests, default-off PyQCU/QUDA trace parsing, and QUDA
outer-iteration scope.  `test_bench_mg_matrix_full.py`（CPU-only）测试
`bench_mg_matrix_full.py` 编排器：units 展开覆盖 levels 1/3、c64/c128 与
`matrix.LATTICES` 全部格子，并校验 asset 根解析与 unit tag 拼装。
（早先的 `test_bench_mg_matrix.py` 及其配套 `bench_mg_matrix.py` 已被
`*_full` 版本取代并于 2026-10-15 删除。）`summarize_mg_trace.py` aggregates
diagnostic stage/residual rows; never use its trace wall times as formal
performance evidence.

### 2026-09-17 double MG / BiCGStab 回归

新增回归必须覆盖：

- `--quda-strategy aligned|library` 的预期参数：aligned 为 MR/一次递归；
  library 为 PyQUDA 原生 CA-GCR、`nu_post=8`、coarse maxiter 16。
- `QUDA_MULTIGRID_DOUBLE=ON` 与 `QUDA_PRECISION=12` 的 CMake provenance；
  c128 正式记录必须含 double bit，且 `precision_null` 全为
  `QUDA_DOUBLE_PRECISION`。
- BiCGStab 参考解通过独立真残差门；QUDA 参考路径必须清除
  `invert_param.preconditioner`，防止 MG 指针残留导致 solver factory
  拒绝 BiCGStab。
- 正式 profile 在 trace 环境变量存在时拒绝运行；只有
  `--allow-trace` 才能产生诊断 JSON。
- c128 `16·32·32·48` 三层当前有两条路径：formal 冷 `C=1` 通过
  `release_python_setup_assets()` 完成，setup `2709.45 s`；extended
  `C=4`/`K=16` 全 double setup 为 `585.28 s`，并可通过
  asset-identity cache 进入 formal `C=1` solve。测试必须把冷 setup
  时间与 cache-hit solve 比值分开报告。
- `--quda-coarse-precision single` 的 mixed-coarse smoke 必须检查
  `gauge_param`/`invert_param` precision readback；只修改
  `invert_param` 不足以让 coarse single 与 fine double 一致。
- `--pyqcu-strict-block-precision single` 与
  `--strict-galerkin-projection-batch 1` 是 setup 低内存实验选项；
  它们仍不能替代当前已完成的 extended 全 double `C=4` setup；不得
  据单精度路径宣称 PyQCU mixed-coarse 已支持。

快速命令：

```bash
python -B -m pytest -q -p no:cacheprovider \
  pyqcu/testing/qcu/strict/quda_comparison/test_bench_strict_protocol.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_bench_mg_matrix_full.py
python pyqcu/testing/qcu/strict/quda_comparison/run_strict_fast.py --tier 1
python pyqcu/testing/qcu/strict/quda_comparison/run_all.py --with-quda
```

正式 JSON 来源：
`data/mg_matrix_20260916_round2/formal-*-l*/benchmark.json`；
trace 开销证据：`trace-small-c64-l3.json`。trace-on 数字只用于说明
trace 开销，不能进入 `speedup_pyqcu_over_quda`。

`run_quda_py.py` 必须在首次导入 PyQUDA 前按 strict benchmark 的方式初始化
QMP；先导入 PyQUDA 会直接触发 `QMP_comm_get_default` abort，进程内无法恢复。
legacy solve/MG case 在 precision-12 组合构建中显式选择 single precision，
以保持与 c64 PyQCU 参考一致；`case_opcmp` 仍保留其有意使用的 double
precision 路径。当前 `run_all.py --with-quda` 的 5 项断言全绿。

## 2026-09-27 MPI overlap 回归

`strict_mpi_solve_probe.py` 现在按 `MPI.COMM_TYPE_SHARED` 的 local rank
设置 Torch 逻辑卡和 `PYQCU_MPI_DEVICE_ID`，双 rank P100 不再误把 rank 1
张量放到 GPU 0。复杂场路径按相同参数分别跑
`PYQCU_MPI_OVERLAP=0/1`，两侧必须收敛到独立 Python 真残差门；当前
`8^3x16`、c64、`grid=2x1x1x1` 的 off/on 分别为 27/26 个外层迭代，
真残差 `6.43e-7/7.52e-7`。

矩阵产物必须用 `validate_matrix_trace_evidence.py` 做 fail-closed
检查：trace-on 只校验当前 side 真正产生的 trace 文件；BiCGStab reference
必须含 1 cold、2 warmup、5 steady，并带 `unit_levels=1`。不要再用
`levels=2` 的 MG phase 冒充 BiCGStab reference。

```bash
python -B -m pytest -q -p no:cacheprovider \
  pyqcu/testing/qcu/strict/quda_comparison/test_bench_strict_protocol.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_bench_mg_matrix_full.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_build_mg_report.py \
  pyqcu/testing/qcu/strict/quda_comparison/test_validate_matrix_trace_evidence.py
```

## 2026-10-23 全量收尾回归

最终 tag 前必须在最后一批微调之后重新运行完整套件，不能用前一轮定向结果替代：

```bash
cd pyqcu/testing
timeout --signal=TERM --kill-after=30s 1000s pytest -q .
```

当前基线为 `449 passed, 10 skipped`。另外保留 `29/29` KUPL CTest、CUDA API ABI
和 KUPL 37 项导出 ABI 门。

- CUDA Strict 构造器在 `hierarchy.setup()` 之前只能依赖已配置的
  `_transition_count`/`_block_sizes`，不得读取 setup 后才填充的
  `transfers/operators`。预检测试探针必须提供现代 `_strict_mode` 与 CPU/GPU
  张量接口，但不能削弱 fail-fast 资产检查。
- 测试需要读取已按文档规范清理的历史报告时，先查工作树；不存在则用
  `git log --all --diff-filter=AM -- <path>` 找到最后增改提交，再通过
  `git show <commit>:<path>` 读取。不得把历史 TeX 恢复到 `docs/`，也不得静默跳过。
- `mg_cross_impl_parity.py` 的退出码旁路文件统一为 `*.exit_code.txt`；裸
  `*.exit_code` 不符合仓库 logs 扩展名规则。
