# PyQCU 正式快速测试端口（`pyqcu.test`）

`pyqcu.test` 是**随发行包发布的正式快速测试端口**：`pip install pyqcu` 之后仍然存在，
用一条命令在秒级到分钟级验证 Torch / KUPL / CUDA 三条后端路径的核心正确性，并支持
指定格点、MPI 网格、规范场、源、质量、clover 系数、求解器与设备。它只回答
“装好的库算得对不对”，不做性能测量，也不替代源码 checkout 里的集成测试。

| 端口 | 路径 | 是否随包发行 | 定位 |
|---|---|---|---|
| 正式 | `pyqcu/test/**` | 是（sdist/wheel） | 跨后端正确性门，安装态可运行 |
| 非正式 | `pyqcu/testing/**` | 否（`MANIFEST.in` prune、`setup.py` 排除） | 集成/回归/性能/QUDA/HPCKit 矩阵，仅源码 checkout |

`pip install pyqcu` 的用户**没有** `pyqcu.testing`：不要在新的用户文档、发布说明或
验证脚本里把 `pyqcu.testing` 写成安装态入口。

## 最短上手

```bash
python -m pyqcu.test --list          # 只列用例矩阵，不 import torch/native 后端
python -m pyqcu.test --check         # 导入/打包/策略自检（不跑数值）
python -m pyqcu.test --quick         # tier 0 + 格点 8 8 8 16 + c64 + seed 42
python -m pyqcu.test --quick --json  # 机器可读报告（默认写 stdout）
python -m pyqcu.test --mode list     # full-mode 台账行：只计划，不执行、不写台账
```

源码 checkout 中先 `source ./env.sh`；安装态直接从任意目录执行 `python -m pyqcu.test`。
`--list` / `--list --json` 是当前版本的权威用例矩阵与生效选项记录：任何 flag 的最终
语义以 `python -m pyqcu.test --help`、`--list --json` 的实际字段和 `pyqcu/test/policy.py`
为准，本文档描述契约，不重复实现细节。full mode 的台账契约见
[「Full mode（工作台账）」](#full-mode工作台账)，其 `--list --json` 是 `--mode list --json`。

## CLI

```text
python -m pyqcu.test [--backend LIST [LIST ...]] [--profile NAME]
  [--lattice-size N [N ...] | --lattice N [N ...]] [--lattice-grid N [N ...]]
  [--mg-grid N [N ...]] [--lattice-preset {L24,L32}]
  [--mode {list,prepare,prewarm,solve,merge,run,validate}]
  [--quick] [--tier 0|1|all] [--only TAG [TAG ...]] [--sweep SPEC]
  [--solver CHOICE] [--dslash CHOICE] [--device DEVICE]
  [--gauge random|unit|ones] [--gauge-sigma F | --guage-sigma F]
  [--source point|ones|random] [--mass F | --kappa F] [--csw F]
  [--dtype c64|c128|complex64|complex128|float32|float64]
  [--device-id N] [--threads N] [--max-iter N] [--restart N] [--mg-levels N]
  [--mg-nu-pre N] [--mg-nu-post N] [--atol F] [--timeout S]
  [--list] [--json [PATH|-]] [--junit PATH] [--csv PATH] [--markdown PATH]
  [--repeat N] [--warmup N] [--seed N] [--fail-fast] [--allow-all-skipped]
  [--outdir DIR] [--rows DIR] [--reports DIR] [--solutions DIR] [--cache-dir DIR]
  [--run-label LABEL] [--resume] [--skip-failed] [--ranks N] [--shm | --no-shm]
  [--shared] [--gauge-file PATH] [--source-file PATH] [--validate]
  [--verbose] [--check] [--version]
```

四个整数的参数（格点、进程网格、MG 块网格）接受三种写法：`8 8 8 16`、`8,8,8,16`、
`[8,8,8,16]`；`--backend` 与 `--only` 接受逗号分隔、空格分隔或重复给出的多个值
（`--only a,b`、`--only a b`、`--only a --only b` 三者等价）。`--mode` 的
`rows/reports/solutions/cache` 路径、`--resume`/`--skip-failed`/`--run-label` 与
`--gauge-file`/`--source-file` 见 [「Full mode（工作台账）」](#full-mode工作台账)
和 [「HDF5 输入」](#hdf5-输入)。

### 用例选择

| 选项 | 含义 |
|---|---|
| `--backend` | 逗号或空格分隔：`torch,cuda,kupl` 加别名 `cpu`（=`torch,kupl`）、`dcu`（=`cuda`）、`npu`（=`torch`），或 `all`（默认 = 全部规范后端）。`all` 不要求三条后端都可用，不可用的按 SKIP 处理；报告里只出现规范名，不出现别名 |
| `--profile` | 命名配置：`quick`/`cpu`/`gpu`/`c128`/`clover`/`mg`/`full`；可重复，后给出的名字覆盖先前名字，显式 flag 始终优先于 profile。`full` 不锁 tier，`quick` 等其余 profile 锁 tier 0 与 `8 8 8 16`/`c64`/seed 42 |
| `--tier` | `0`/`1`/`all`：跑 tier ≤ N 的用例（`--quick` 强制 `0`；full mode 未显式给 tier 时默认 `1`） |
| `--only TAG ...` | 可重复，值可用逗号或空格分隔；接受完整 tag（`torch/dslash/wilson`）、backend、kind 段（`dslash`）或 tag 前缀（`torch/dslash`） |
| `--sweep SPEC` / `--matrix SPEC` | 参数扫描：`key=value` 组用 `;` 分隔（`dtype=c64,c128;lattice=[8,8,8,16];[16,16,16,16]`），可重复；按固定轴序展开，上限 512 个组合。`--list` 只列展开行，执行时按序跑完 |
| `--solver` | `all`（默认，不筛）/`bicgstab`/`cg`/`multigrid`/`mg`。只筛 solve/mg 用例：`bicgstab` 保留 `*/solve/bistabcg`，`cg` 保留 `*/solve/cg`，`multigrid`/`mg` 保留 `*/mg/*`；dslash、smear、gauge、lattice、ABI 用例不受影响 |
| `--dslash` | `all`/`both`（默认，不筛）/`wilson`/`clover`。只筛 `kind == dslash` 的用例；其余用例不受影响 |
| `--quick` | 只锁 **tier 0、格点 `8 8 8 16`、`c64`、`seed 42`、`repeat 1`**（`warmup` 仍为 1）。设备、求解器、算子、质量、clover、源、规范场、网格、线程等选项继续生效，可与 `--quick` 组合；只有与 `--lattice-size`/`--dtype`/`--seed`/`--tier` 冲突才按用法错误退出 `2` |
| `--list` | 打印选中用例矩阵后退出；重型 import 之前短路，不 import torch / native 桥。`--list --json` 给出 `cases`/`count` 与本次生效配置；`--list --verbose` 把完整配置写到 stderr |
| `--check` | 跑 `selfcheck`（导入隔离、打包包含性、策略常量）后退出，不跑数值 |
| `--version` | 打印端口版本后退出 |

### 几何、MPI 与多重网格网格

| 选项 | 含义 |
|---|---|
| `--lattice-size` / `--lattice` | 同一选项的两个拼写，取 `8 8 8 16`、`8,8,8,16` 或 `[8,8,8,16]`。四个正整数且**偶数、≥ 4**（奇偶分解与 clover 需要偶数体积），默认 `8 8 8 16`；非法值按用法错误退出 `2` |
| `--lattice-grid` | MPI 4D 进程网格（`x y z t`），默认 `1 1 1 1`。**乘积与 `mpirun -np` 的 rank 数不一致不是用法错误**：原生后端（KUPL/CUDA）报 `SKIP(mpi_grid)`，torch 用例不看该网格。真正执行且选中原生用例时，不一致会在 stderr 打一行提示 |
| `--mg-grid` | 多重网格粗化块网格（每维因子），默认 `2 2 2 2`；只被 `*/mg/*` 用例使用 |
| `--lattice-preset` | 发布验收格点 `L24`（`24 24 24 64`）/ `L32`（`32 32 32 64`）；与 `--lattice-size`/`--lattice` 或 `--quick` 同时给出是用法错误（退出 `2`）。它只改格点，不改变 tier 或判据 |

### 规范场、源与物理参数

| 选项 | 含义 |
|---|---|
| `--gauge` | 背景规范场来源：`random`（默认，按 `--seed` 生成 Gaussian 场并投影到 SU(3)）、`unit`（单位链接）、`ones`（全 1 链接）。**不是文件路径**；文件组态用 full-mode 的 `--gauge-file`（见 [「HDF5 输入」](#hdf5-输入)） |
| `--gauge-sigma` / `--guage-sigma` | `--gauge random` 的 Gaussian 宽度，默认 `0.1`。`--guage-sigma` 是历史拼写，与正确拼写指向同一个选项（argparse 别名），效果完全等价；报告只记录 `gauge_sigma` |
| `--source` | 费米子源：`point`（默认，原点单点源，`\|\|b\|\| = 1`）、`ones`（常数源）、`random`（按 `--seed` 的高斯源） |
| `--mass` / `--kappa` | 同一物理量的两种写法：`kappa = 1/(2*mass + 8)`，反之 `mass = 1/(2*kappa) - 4`。两者都**没有** CLI 默认值，未给出时不向前转发、由各 runner 的默认决定（tier 0 的 torch 路径默认 `kappa = 0.125`，即 `mass = 0`）；`--kappa` 必须 `> 0`。只给一个；两个都给时都会记入报告 `options`，换算责任在 runner，跨后端对齐时优先统一用 `--kappa` |
| `--csw` | clover 系数，默认 `1.0`；`0` 即纯 Wilson 极限（clover 用例在 `csw = 0` 时退化为 Wilson 一致性检查） |
| `--dtype` | `c64`/`c128`，以及 `complex64`/`complex128`、`float32`/`float64` 四个别名（分别归一化到 `c64`/`c128`）；报告与 `--list` 只出现规范名 |

### 求解器与多重网格

| 选项 | 含义 |
|---|---|
| `--max-iter` | 求解迭代上限；未给出时用 runner 默认（KUPL 1000、CUDA 200，torch 见 runner） |
| `--restart` | Krylov 重启长度；未给出时用 runner 默认 |
| `--mg-levels` | 多重网格层数；未给出时用 runner 默认（tier 0 为 2） |
| `--mg-nu-pre` / `--mg-nu-post` | 前/后平滑步数；未给出时用 runner 默认 |
| `--atol` | 只覆盖 `compare`（算子/涂抹一致性）相对容差；`solve`/`native_solve`/`gamma` 仍按 dtype 固定，用户参数不能放宽“真的收敛了” |

### 设备与线程

| 选项 | 含义 |
|---|---|
| `--device` | `auto`（默认，不向前转发，由后端自选；torch 路径即 CPU）、`cpu`、`gpu`（→ `cuda:0`）、`dcu`（→ `dcu:0`）、`npu`（→ `npu:0`）。也可直接给具体设备串（如 `cuda:3`）。请求的加速器族在本机不可用时报 `SKIP(device_unavailable)`，不是 FAIL |
| `--device-id` | 加速器序号（`>= 0`），转发给原生 runner |
| `--threads` | 线程数；**记录到 `rows/*.json`、`--list --json` 与报告 `options.threads`**。当前 CLI 真正应用它的只有 tier-0 torch 用例：调用 `torch.set_num_threads`，并在 metrics 里给 `torch_threads`/`threads_requested`/`threads_applied`/`threads_error`。tier-0 的 `kupl_qcu.py`/`cuda_qcu.py` **不读取** `--threads`；`native.py` 的原生 runner 扩展点会在首次原生调用前用 `apply_environment` 设置 `OMP_NUM_THREADS`/`KUPL_OMP_THREADS`/`OMP_DYNAMIC`，但该路径需要 `driver:run_row` 模块，当前 checkout 未提供。因此看到 KUPL/CUDA 行里的 `threads` 只表示“记录值”，不表示 KUPL/OpenMP 线程数被改变。未给出时记录 `null`，由环境/后端默认决定 |

### 执行与输出

| 选项 | 含义 |
|---|---|
| `--json [PATH\|-]` | 省略值或 `-` 写 stdout；给路径时原子写（临时名 + rename），失败不留下半份文件 |
| `--junit PATH` / `--markdown PATH` | 用同一份报告对象额外原子写 JUnit XML / Markdown（`-` 写 stdout）；`--list` 下拒绝（不执行就没有报告） |
| `--csv PATH` | 普通运行：额外写 per-case CSV 报告。**full mode 下 `--csv` 是台账 CSV**（默认 `<reports>/rows.csv`），不是报告导出；`--csv -` 在 full mode 下是用法错误 |
| `--repeat` / `--warmup` | 默认 `5` / `1`；`--quick` 固定 `1` / `1`。warmup 传 `0` 会被拒绝 |
| `--seed` | 规范场/源的确定性 seed，默认 `42` |
| `--timeout` | 每个用例的超时秒数（默认按用例给出）；超时记 `TIMEOUT`，不是 `FAIL` |
| `--fail-fast` | 首个 `FAIL`/`ERROR`/`TIMEOUT` 立即停止（`SKIP` 不触发停止）；full-mode `solve`/`run` 同样在首个非 SKIP 失败行停止 |
| `--allow-all-skipped` | 全部 SKIP 时仍返回 0（默认返回 1）；只影响数值运行与 full-mode `solve`/`run`，不改变 `prepare`/`prewarm` 的输入 SKIP |
| `--outdir DIR` | 普通运行：额外把 JSON 报告原子写到 `DIR/report.json`。full mode：默认工作根（默认 `pyqcu-test-runs`），其下派生 `rows/`、`reports/`、`solutions/`、`caches/` |
| `--verbose` | 每个用例的 metrics 与失败 gate 写 stderr；stdout 保持每用例一行、稳定可解析 |

`--json` 报告是 `TestReport.as_dict()`：顶层 `package`、`version`、`status`、
`exit_code`、`lattice`、`dtype`、`seed`、`mpi_size`、`mpi_rank`、`selected`、`counts`、
`all_skipped`、`duration_seconds`、`options`、`env`、`cases`；每个 case 带 `tag`、
`backend`、`kind`、`status`、`seconds`、`metrics`、`failed`、`reason`、`lattice`、
`dtype`、`seed`、`mpi_size`、`tier`、`runner`、`skip_token`。`options` 记录本次生效的
全部配置（`lattice`、`lattice_grid`、`mg_grid`、`dtype`、`seed`、`source`、`gauge`、
`gauge_sigma`、`mass`、`kappa`、`csw`、`solver`、`dslash`、`device`、`runner_device`、
`max_iter`、`restart`、`mg_levels`、`mg_nu_pre`、`mg_nu_post`、`threads`、`device_id`、
`atol`、`repeat`、`warmup`、`quick`、`backend_filter`、`tier`、`argv`，以及
`fail_fast`/`allow_all_skipped`/`mpi_grid_matches`/`backends`/`tags`），因此一次运行可由
它自己的输出复现。观测值在 `metrics` 里（`iterations`、`max_iter`、
`relative_residual` 及别名 `relative_error`/`true_residual`）。
状态词汇与 `pyqcu --selftest --json` 一致：`PASS`/`SKIP`/`FAIL`/`ERROR`/`TIMEOUT`。

## Full mode（工作台账）

`--mode` 把一次运行从“跑一遍选中的 case”切换成**工作台账**：先把选中矩阵展开成行
（row），每行是“一个 case × 一组生效配置”，再按模式对行做计划、准备、预热、求解、
合并或校验。它沿用非正式树 `pyqcu/testing/kupl/test_kunpeng_all.py` 的
`list/prepare/prewarm/solve/merge/run/validate` 词汇，但**不是**它的 HPCKit/mpirun/
CSV 分片实现：full mode 是编排层，每个 row 仍走与单用例运行完全相同的 runner 与
PASS/FAIL 判据，不另造第二套执行器。

| `--mode` | 做什么 | 写什么 |
|---|---|---|
| `list` | 只展开并打印选中行（每行 `status` 保持 `null`），不执行、不 import 后端 | 不写台账；`--json`/`--outdir` 只落这一份清单 |
| `prepare` | 创建 `rows/reports/solutions/caches`，校验 `--gauge-file`/`--source-file`，写 cache 索引并探测已有 cache | `reports/prepare.json`、`caches/index.json`、`report.json` |
| `prewarm` | 把每个已请求输入完整读一次（page-cache 预热）并算 SHA256；不跑数值 | `reports/prewarm.json`、`report.json` |
| `solve` | 执行每一行，判定与单用例运行一致 | `rows/<run_id>.json`、`reports/<run_id>.json`、`reports/rows.csv`、`report.json` |
| `merge` | 只从 `rows/*.json` 重建 CSV 与汇总文档（row 文件是唯一事实源） | `reports/rows.csv`、`reports/merged.json` |
| `validate` | 交叉核对 rows、每行报告与 CSV 一致（缺行/多行/字段不符都算 problem） | 校验文档（`report.json`） |
| `run` | `prepare` → `prewarm` → `solve` → `merge`；输入准备失败时跳过 solve 并返回 `1` | 上述全部 |

**默认 tier**：full mode 未显式给 `--tier` 时默认 tier 1（当前 tier 0 覆盖全部已声明
用例，因此 tier 1 选中同一批行）；`--quick` 仍锁 tier 0、`8 8 8 16`、c64、seed 42、
repeat 1。`--quick --mode solve` 是合法组合，但仍是 tier-0 小格运行。`--validate`
单独出现等价于 `--mode validate`；与 `--mode solve`/`run` 同时出现时作为该模式结束后的
额外校验步骤。

### 路径与产物

默认工作根是当前目录下的 `pyqcu-test-runs/`；用 `--outdir` 改根，或用
`--rows`/`--reports`/`--solutions`/`--cache-dir` 单独覆盖（相对路径按当前目录解析）。

| 路径 | 内容 |
|---|---|
| `<outdir>/rows/<run_id>.json` | 一行一个 JSON：生效配置 + `status`/`metrics`/`failed`/`reason`/`skip_token` + `report_path`/`solution_path`。**row 文件是唯一事实源** |
| `<outdir>/reports/<run_id>.json` | 该行的完整报告对象（与单用例 `--json` 同构） |
| `<outdir>/reports/rows.csv` | 由 rows 派生的台账 CSV（含 run_id/status/tag/backend/kind/tier/lattice/dtype/seed/ranks/threads/shm/shared/输入路径/resumed/run_label 等列）；可随时用 `merge` 重建 |
| `<outdir>/reports/prepare.json`、`prewarm.json`、`merged.json` | 对应模式的输入/合并文档 |
| `<outdir>/solutions/` | runner 报告 `solution_path` 时复制到这里的解文件；没有就保持为空，台账绝不伪造解文件 |
| `<outdir>/caches/index.json` | Legacy-MG cache 目录索引：行数、dtype/lattice、`cache_require` 绑定、`identity_fields`、`default_cache_dir`，以及已有 artifact 的 `ready`/`cache_status` |

`--cache-dir` 可以指向目录，也可以指向一个已有的 Legacy-MG cache **artifact 文件**；
`prepare` 只创建目录/索引并调用 `pyqcu.test.cache.cache_status` 探测，**不重建 cache**
（重建需要 native builder，属于非正式树）。探测默认校验 manifest 的 `file_sha256`，
并绑定 rows 一致同意的 `lattice`/`dtype`/`seed`/`mpi_size`；被拒绝时该输入记 `FAIL`。

### run_id、resume 与 run-label

- `run_id` 是行配置摘要（16 位 SHA-1）加可选 `--run-label` 后缀
  （`<digest>-<label>`）；同一配置的未标注运行共享同一个 row 路径，不同 label 的
  冷/热或 before/after 运行不会互相覆盖。label 不能含路径分隔符。
- `--resume`：已有 row 的 `status` 是 `PASS` 或 `SKIP` 时直接复用，并标记 `resumed=true`。
- `--skip-failed`：只在 `--resume` 下生效，额外复用 `FAIL` 行——这些行**以后也不会再重试**。
  `ERROR`/`TIMEOUT` 永远重试（可能是暂时性故障）。
- `--resume` 只读同一路径下的 row；换 `--outdir`/`--rows` 就是另一份台账。

### full-mode 字段：记录 vs 生效

| 字段 | 记录位置 | 是否真正生效 |
|---|---|---|
| `--ranks` | `rows/*.json`、`--list --json` | 台账/证据字段；默认 `prod(--lattice-grid)`。真正的 MPI 规模仍由 `mpirun -np` 与 `--lattice-grid` 决定 |
| `--threads` | 同上 | 只在 torch 行生效（`torch.set_num_threads`）；KUPL/CUDA 行只记录，见 [设备与线程](#设备与线程) |
| `--shm` / `--no-shm` | 同上 | 内置执行器只记录；`native.py` 的 `apply_environment` 会写 `KUPL_SHM_HALO` 并在不可用时报 `SKIP(shm_unavailable)`，但该原生路径需要 `driver:run_row` 模块，当前 checkout 未提供 |
| `--shared` | 同上 | 只作为“请求单节点共享执行路径”的证据字段；当前内置 runner 不据此改变执行路径 |
| `--gauge-file` / `--source-file` | 同上 | `prepare` 校验、`prewarm` 读取并哈希；内置 tier-0 case 执行仍用 case 自己的 `--gauge`/`--source` 生成场（见 [「HDF5 输入」](#hdf5-输入)） |

### 退出码

full mode 与普通运行共用 `0`/`1`/`2` 语义，但有两个模式特例：

- `prepare`/`prewarm` 的输入 SKIP 是“没请求这个输入”，整体 SKIP 仍退出 `0`；输入
  `FAIL`/`ERROR`/`TIMEOUT` 退出 `1`。
- `merge`/`validate` 面对空台账返回 `1`（`SKIP`），不会把“什么都没合并/校验”当成绿色
  通过；`validate` 发现 rows/reports/CSV 不一致返回 `1`。
- `--mode list` 选中 0 行时返回 `1`。

## HDF5 输入

full mode 的 `--gauge-file`/`--source-file` 是**显式输入的 HDF5 文件**（普通 case 运行的
`--gauge`/`--source` 仍是生成器，不是路径）。文件由 `pyqcu.test.io` 校验/读写，契约如下：

| 项 | 规则 |
|---|---|
| dataset | 固定 `data` |
| dtype | 仅 `complex64`/`complex128`；文件名不是证据，dataset dtype 才是 |
| gauge 布局 | full `[3,3,4,x,y,z,t]`；parity-compressed `[2,3,3,4,x,y,z,t/2]` |
| source 布局 | full `[4,3,x,y,z,t]`；parity-compressed `[2,4,3,x,y,z,t/2]` |
| 属性 | `format="pyqcu.simple.v1"`、`layout`、`role`（`gauge`/`source`）、`dtype`、`shape`、`lattice`、`compact`；parity-compressed 文件必须带 `layout` |
| 校验 | 打开文件、检查 dataset/形状/dtype/属性并真实读取首尾元素后才接受；损坏、截断、形状不符或属性矛盾的输入报错，不会被静默复用 |
| 写入 | `pyqcu.test.io.save_gauge`/`save_source` 以 temp → flush → 回读校验 → `fsync` → rename → `fsync(dir)` 原子发布，校验通过才落到目标路径 |

`prepare` 对给定文件做上述校验并把结果写进 `reports/prepare.json`；`prewarm` 再完整读
一次并记录 `sha256`/字节数。两个模式都只处理**已请求**的输入：未给 `--gauge-file` 时
该条目记 `SKIP`（不是 FAIL）。文件缺失、不可读、形状不符记 `FAIL` 并使该模式退出 `1`。

内置 tier-0 case 执行器**不会**把 `--gauge-file`/`--source-file` 偷偷替换进 case：
case 的数字仍由它自己的 `--gauge`/`--source`/`--seed` 决定，台账里的文件字段是本次
运行声明的输入证据。需要“按文件组态求解”的驱动属于非正式树的职责；若后续把文件
接入 runner，必须同时让 `options`/`rows` 记录 `gauge_origin`/`source_origin`，不得让
报告声称的参数与实际读到的不一致。

## 参数映射（先读这一节）

**`--mass` vs `--kappa`**：同一个物理量。`kappa = 1/(2*mass + 8)`，默认 tier 0 值是
`kappa = 0.125`（`mass = 0`）。两者都未给出时 runner 的默认生效；给出时只给一个，
`--mass -3.4` 这类接近退化区的值合法（`kappa ≈ 0.8333`），但会显著改变算子条件数；
残差门不随质量放宽，接近退化区的质量只会更容易 FAIL。

**规范场 sigma**：`--gauge-sigma` 只在 `--gauge random` 时有物理意义；`--gauge unit` /
`--gauge ones` 是确定性组态，sigma 仍会记录进报告但不参与构造。

**MPI 格点网格 vs torch**：`--lattice-grid` 是**原生后端**的 MPI 分域网格
（`prod(grid) == MPI size` 时正常执行）；乘积不匹配时原生用例报 `SKIP(mpi_grid)` 并带
reason，**不是**用法错误，也不会被静默改成单 rank 结果。torch tier-0 用例是单进程计算，
不读该网格；因此 `--backend torch --lattice-grid [1,2,2,1]` 只影响解析与报告，不会让
torch 变成分布式运行。分域 MPI 覆盖属于非正式套件（`pyqcu/testing/**`）。

**`--backend` 别名展开**：`cpu` → `torch,kupl`；`dcu` → `cuda`（DCU/ROCm 复用同一 C++
桥与 `cuda/*` 用例，设备族由 `--device dcu` 指定）；`npu` → `torch`（经 `pyqcu.cann` 把
复数运算分解为实/虚部）；`all` → 三个规范后端。展开发生在参数解析阶段，报告与
`--list --json` 只出现规范名。

**`--device` 映射**：`auto` 不转发任何设备串（后端保留自己的默认，torch 即 CPU）；
`cpu`→`cpu`、`gpu`→`cuda:0`、`dcu`→`dcu:0`、`npu`→`npu:0`；具体设备串（`cuda:3`）原样
透传。设备族不可用时相关用例报 `SKIP(device_unavailable)`（带 reason），不是 FAIL。

**CUDA 桥缺失**：`pyqcu.cuda.qcu` 没有构建出的扩展时，`cuda/*` 用例报
`SKIP(cuda_bridge)`；这是本 checkout 的预期结果，不得为变绿去构建桥或伪造 PASS。

**`--quick` 只锁五件事**：tier、格点、dtype、seed、repeat。它**不**锁定设备、求解器、
算子、质量、clover、源、规范场、网格、线程和 `--atol`，这些参数在 `--quick` 下照常生效
（因此 `--quick --device gpu --solver cg` 是合法组合，`--quick --dtype complex128` 才是
冲突用法错误）。

**`--atol` 的边界**：只覆盖 `compare` 门（算子/涂抹的独立路径一致性），不覆盖
`solve`/`native_solve`（真残差）与 `gamma`（Dirac 代数恒等式）。它也可能作为求解器请求
阈值转给 runner，但**判定门不变**：把 `--atol` 放宽只会让 solve 用例因为真残差达不到
dtype 门而 FAIL，不会被“放宽成通过”。

## 退出码

| 码 | 含义 |
|---|---|
| `0` | 无 `FAIL`/`ERROR`/`TIMEOUT`（允许 `SKIP`） |
| `1` | 任一 `FAIL`/`ERROR`/`TIMEOUT`，或所选用例**全部 SKIP**（除非显式 `--allow-all-skipped`） |
| `2` | 用法/环境错误：未知 backend/solver/dslash/device、非法格点或网格、非法 dtype、无法写 `--json` 路径等 |

与安装自检的差异必须记住：`python -m pyqcu --selftest` 把数值 `FAIL` 记为 **2**；
本端口把数值失败记为 **1**、用法错误记为 **2**。批量脚本不得把两者当同一约定。

## Tier 0 用例（默认门禁）

默认单 rank、`c64`、不读数据文件；`[8,8,8,16]` 是项目正确性门格点
（`pyqcu/testing/_config.py`、根 `AGENTS.md`），**不是**发布验收用的 `8x8x16x16`；
不要互换。共 **19** 个用例：前 14 个是原始矩阵，后 5 个是 gauge/Wuppertal/CG/clover
补充。

| # | tag | 判据 | 可能 SKIP |
|---|---|---|---|
| 1 | `torch/lattice` | `check_su3` 为真；`gamma_mu^2 = I` 偏差 ≤ `gamma` | `device_unavailable` |
| 2 | `torch/dslash/wilson` | 全算子 vs 奇偶分解相对差 ≤ `compare` | `device_unavailable` |
| 3 | `torch/dslash/parity` | `matvec_all` vs `matvec_eeo`/`matvec_oeo` 相对差 ≤ `compare` | `device_unavailable` |
| 4 | `torch/dslash/clover` | clover 项 + 逆往返相对差 ≤ `compare`（`csw=0` 退化为 Wilson） | `device_unavailable` |
| 5 | `torch/smear/stout` | smearing 前后 SU(3) + 局部/全局一致相对差 ≤ `compare` | `device_unavailable` |
| 6 | `torch/solve/bistabcg` | 独立真残差 ≤ `solve` 且 `iterations > 0` | `device_unavailable` |
| 7 | `torch/mg/legacy` | 显式 legacy 变体收敛，相对残差 ≤ `solve` | `device_unavailable` |
| 8 | `torch/mg/strict` | 显式 `strict` 变体收敛，相对残差 ≤ `solve` | `device_unavailable` |
| 9 | `kupl/abi/symbols` | C API 40 个符号，名称集合完全一致 | `kupl_library` |
| 10 | `kupl/dslash/wilson` | vs torch 参考相对差 ≤ `compare` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 11 | `kupl/solve/bistabcg` | 全解 + 独立残差 ≤ `native_solve`，`0 < iterations < max_iter` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 12 | `cuda/smoke/init_end` | 桥 `applyInitQcu`/`applyEndQcu` 生命周期成功 | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 13 | `cuda/dslash/wilson` | vs torch 参考相对差 ≤ `compare` | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 14 | `cuda/solve/bistabcg` | 全解 + 独立残差 ≤ `native_solve` | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 15 | `torch/gauge/random` | 生成并投影后的规范场 `check_su3` 为真（σ/seed 只作为观测记录在 `metrics`，不是额外门） | `device_unavailable` |
| 16 | `torch/smear/wuppertal` | smearing 前后 SU(3) + 局部/全局一致相对差 ≤ `compare` | `device_unavailable` |
| 17 | `torch/solve/cg` | CG（γ5-hermitian 算子）独立真残差 ≤ `solve` 且 `iterations > 0` | `device_unavailable` |
| 18 | `kupl/dslash/clover` | clover dslash vs torch 参考相对差 ≤ `compare` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 19 | `kupl/solve/cg` | 全解 + 独立残差 ≤ `native_solve`，`0 < iterations < max_iter` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |

相对容差按 dtype 固定：`c64` 为 `compare` `1e-5`、`solve` `1e-3`、`native_solve`
`2e-4`、`gamma` `1e-6`；`c128` 为 `1e-10`/`1e-6`/`1e-8`/`1e-12`。`--atol` 只能覆盖
`compare`，不能放宽残差门。

> CUDA 的正式 tier-0 只保留 smoke、Wilson dslash、BiCGStab 三个已实现路径；
> CUDA clover/CG 目前不在正式矩阵中，不能把 SKIP 当成已验证。

`case` 的 `expect` 是唯一的 PASS/FAIL 判据；runner 只回观测值。所有 solve 用例的
真残差必须由 `||D x - b|| / ||b||` 独立重构，不得复用求解器内部递推；CG 用例额外要求
算子按 γ5-hermitian 约定求解，但残差仍在全 Wilson 算子上重构。torch 与 KUPL 数值路径
不依赖 GPU；CUDA 用例只有在桥与设备同时存在时才真正执行（上述两个未实现 runner 例外）。

`--solver`/`--dslash` 只筛矩阵，不新增用例：新增算法覆盖必须新增 tag（并进
`pyqcu/test/units.py`），不得复用旧 tag 改语义。

## SKIP 策略

| reason | 触发条件 |
|---|---|
| `kupl_library` | `pyqcu.kupl` 的 ctypes 加载抛 `FileNotFoundError`/`OSError`，或全部候选库路径不存在 |
| `cuda_bridge` | `pyqcu.cuda.qcu` 缺少 `applyInitQcu`，或 Cython 扩展未构建 |
| `cuda_device` | `torch.cuda.is_available()` 为 False |
| `mpi_single_rank` | MPI size > 1（用例要求单 rank） |
| `h5py_mpi` | `h5py.get_config().mpi` 为 False（需要 mpio 的 HDF5 读取） |
| `mpi_grid` | `prod(--lattice-grid)` 与 MPI size 不一致（原生 runner 报此 token） |
| `shm_unavailable` | 原生 full-mode 请求共享 halo（`--shm`/`KUPL_SHM_HALO=1`）但当前环境不可用：非 KUPL 后端、单 rank、非 Linux 或跨节点 |
| `device_unavailable` | `--device` 请求的加速器族在本机不可用 |

- 每个 SKIP 必须带**非空** `reason`；空 reason 视为缺陷而不是“跳过了”。
- 本仓库 checkout 中 **CUDA SKIP 是预期结果**：`pyqcu/cuda/qcu/` 只有
  `.pyx/.pxd/.pyi`，没有构建出的 `qcu*.so`，用例报 `SKIP(cuda_bridge)`。它不是 FAIL，
  也不得为了变绿去构建桥、伪造 PASS 或跳过断言。
- `--lattice-grid` 与 MPI size 不一致时，torch 用例仍照常执行（它们不看网格），
  原生用例报 `SKIP(mpi_grid)`；一条都没验证成功时整体退出 `1`。
- 全部选中用例 SKIP 默认退出 `1`：一条都没验证不能算通过。

## 示例

```bash
# 1) 全量参数解析演示：--list 不 import 后端，也不跑数值（本次输出 7 行 torch 用例）
python -m pyqcu.test --lattice-size [8,8,8,16] --gauge-sigma 0.1 --mass -3.4 \
  --lattice-grid [1,2,2,1] --solver multigrid --backend torch --device cpu \
  --dslash clover --mg-grid [2,2,2,2] --atol 1e-6 --dtype complex128 --list
```

```bash
# 2) 单用例可复现跑（MG legacy，8x8x8x16、c64、seed 42；本机 PASS，约 47 s 含 setup）
python -m pyqcu.test --only torch/mg/legacy --repeat 1 --warmup 1
```

```bash
# 3) CPU-only 子集 + complex128 + 机器可读报告
python -m pyqcu.test --backend cpu --dtype complex128 --repeat 1 --json cpu.json

# 4) 只保留 CG solve 与 clover dslash（其他非 solve/mg、非 dslash 用例照常保留）
python -m pyqcu.test --solver cg --dslash clover --list

# 5) 指定规范场与源（不读文件：random/unit/ones 三选一）
python -m pyqcu.test --backend kupl --only kupl/solve/bistabcg \
  --gauge random --gauge-sigma 0.1 --source random --kappa 0.1199 \
  --csw 1.0 --atol 1e-6 --max-iter 1000 --repeat 1 --json - > report.json

# 6) NPU / DCU 设备族（不可用时该后端用例 SKIP，不伪造 PASS）
python -m pyqcu.test --backend npu --device npu --repeat 1
python -m pyqcu.test --backend dcu --device dcu --allow-all-skipped
```

```bash
# 7) full mode：准备 → 求解 → 合并 → 校验（同一台账目录；solve 可续跑）
python -m pyqcu.test --mode prepare --outdir runs/l24 --lattice-preset L24
python -m pyqcu.test --mode solve --outdir runs/l24 --ranks 16 --shm \
  --resume --skip-failed --run-label cold
python -m pyqcu.test --mode merge --outdir runs/l24
python -m pyqcu.test --mode validate --outdir runs/l24

# 8) HDF5 输入：只校验/预热，不跑数值（路径换成你自己的、与 8x8x8x16 匹配的文件）
python -m pyqcu.test --quick --only torch/lattice --mode prewarm \
  --gauge-file runs/inputs/gauge_8x8x8x16_c64.h5 \
  --source-file runs/inputs/source_8x8x8x16_c64.h5
```

`--lattice-size`/`--lattice-grid`/`--mg-grid` 的方括号写法在 bash 下可直接用；在
`zsh`/`csh` 或含特殊字符的 shell 里建议加引号（`--lattice-size '[8,8,8,16]'`）。

## 正式 vs 非正式（边界）

| 维度 | 正式 `pyqcu.test` | 非正式 `pyqcu.testing` |
|---|---|---|
| 发行 | 随 sdist/wheel（`recursive-include pyqcu *.py` 自动带上） | 不进 sdist（`prune pyqcu/testing` + `find_packages(... exclude=["pyqcu.testing", ...])`） |
| 安装用户 | 有 | 没有 |
| 导入成本 | stdlib-only：不 import `torch`/`numpy`/`mpi4py`/`h5py`/`pytest` | 模块级 import `torch`、`mpi4py`、`pytest` |
| 运行目标 | 秒级到分钟级跨后端正确性门；full-mode 工作台账（`--mode list/prepare/prewarm/solve/merge/run/validate`） | 集成、回归、性能、QUDA/HPCKit 矩阵、历史资产 |
| 新增用例落点 | `pyqcu/test/units.py` | `pyqcu/testing/**` |

- 两个方向都不许互相 import：`pyqcu.test` 不 import `pyqcu.testing`；生产代码
  （`pyqcu.torch|cuda|kupl|lattice|tools`）不 import `pyqcu.test`。
- **不要在 `pyqcu.testing` 新增“正式”用例**：发布门要跑的用例属于 `pyqcu/test/units.py`，
  `pyqcu/testing` 只留探索性、研究性、性能性和历史回归资产。
- 可选的 pytest 适配器只放在非正式树（`pyqcu/testing/python/test_pyqcu_test_port.py`），
  SKIP 时 `pytest.skip(reason)`；它不属于发行内容。
- 本目录的 `README.md` 与 `AGENTS.md` 不进 sdist：`MANIFEST.in` 只收 `pyqcu` 下的
  `*.py/*.pyi/*.pyx/*.pxd`，且 `global-exclude AGENTS.md`。

## 目录约定

- 本目录**不得**出现 `test_*.py`：form 审计对 `<repo>/testing/` 之外的 `test_*.py`
  报 `WARNING TEST_LOCATION`，而 `pyqcu/test/` 不是 `testing/`。自检文件固定为
  `selfcheck.py`。
- 本包任何模块**不得 import pytest**：`pytest -q pyqcu/test/selfcheck.py` 只是在包外
  显式调用 `selfcheck.py` 中的 `test_*` 函数，pytest 不是运行期依赖。
- 不读 HDF5/仓库数据、不写仓库路径。**普通运行**的 `--json` 目标由调用者给，
  `--outdir` 缺省不落盘；**full mode** 只写 `--outdir`/`--rows`/`--reports`/
  `--solutions`/`--cache-dir` 解析出的目录（默认工作根是当前目录的 `pyqcu-test-runs/`），
  且只读调用者用 `--gauge-file`/`--source-file` 显式给出的输入文件。
- `--quick` 只做正确性：**不得**用它报性能、加速比或回退；计时字段是诊断信息，
  中位数只在 `--repeat >= 5` 且 `warmup >= 1` 时才有一点意义。示例里的耗时数字是
  单机诊断值，不是性能结论。

## 与其他入口的关系

| 入口 | 定位 |
|---|---|
| `python -m pyqcu.test --quick` | 安装态可跑的正式正确性门（本目录） |
| `python -m pyqcu.test --mode <mode>` | 正式端口的 full-mode 工作台账（本目录；模式名沿用非正式驱动的词汇） |
| `python -m pyqcu --selftest` | 安装自检：Torch/KUPL/CUDA 三层独立残差，`FAIL` 退出 2 |
| `cd pyqcu/testing && pytest .` | 非正式全树盘点，仅源码 checkout |
| `python pyqcu/testing/kupl/test_kunpeng_all.py` | 非正式 HPCKit/mpirun 矩阵驱动；full-mode 的词汇来源，但不是本端口的执行实现 |
| `python -m pyqcu.testing.release_check --offline` | 发布门（仅 checkout；`sdist ships pyqcu.test` 正向核对本端口入包） |
