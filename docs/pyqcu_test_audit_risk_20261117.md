# pyqcu.test / L48 远端审计风险登记（2026-11-17）

> 证据等级：**C 级时点证据**（代码走查 + 2026-11-17 远端实测记录）。本文是**风险
> 登记**，不是修复报告：**P0/P1 全部为“未修复（NOT FIXED）”**。没有新证据前，任何
> 一条都不得写成已解决。本文不声称 L48 solve 或两节点 PASS，也不含私网地址、密钥
> 或口令。
>
> 设计、接口契约与远端基线见 [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md)；
> 本文只登记“哪里会算错/会崩/会误报/不可复核”，不复制设计正文。

## 1. 范围与判定

### 1.1 严重度

| 级别 | 含义 | 使用边界 |
|---|---|---|
| **P0** | 会算错、会崩、会误报 PASS，或让 L48/两节点结果不可信 | 修复并有复核证据前，不得发布该路径的结果 |
| **P1** | 显著功能缺口、性能/计时口径偏差、台账或证据不可用 | 不得作为生产结论；只能标 TODO/UNVERIFIED |

### 1.2 “未修复”的定义

当前发布树 = **`0.1.17` / HEAD `1f65469e`**。判定以 `git cat-file -e HEAD:<path>` 与
`git show HEAD:<path>` 为准；工作区里未提交的并发实现只算“候选”，**不算修复**。
本文所有“未修复”都指发布树没有可复核的修复，且本轮没有在远端重跑验证。

### 1.3 既有审计与来源

| 来源 | 覆盖范围 | 本文如何引用 |
|---|---|---|
| [`logs/pyqcu_audit_20261112.txt`](../logs/pyqcu_audit_20261112.txt) | `pyqcu/testing/simple/kupl_multigrid.py` 及辅助层的 L24 KUPL 求解路径；P0/P1/P2 分级、计时边界、真残差与 cache 口径 | **不是** `pyqcu.test` 端口的审计；本文只沿用其 P0/P1 判定方法，不把它的 L24 结论直接搬到 L48 |
| [`docs/kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) | HDF5 输入不可读根因、`h5_readable` 读侧闸门、写侧缺 `fsync`/回读、同名 `*_c64.h5` 实为 complex128 的既往缺陷 | L48 dtype/文件名误标属同一类；是 P1-5 的直接依据 |
| [`docs/multigrid_deprecation_audit_20261022.md`](multigrid_deprecation_audit_20261022.md) | MultiGrid 弃用、历史残留、占位目录与“静态审计不替代运行验证”的方法学 | 审计方法参照；不把它对 MG 的结论当作 `pyqcu.test` 的结论 |
| [`docs/kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md) | KUPL 多 rank/多线程扩展性、t 方向几何上限、跨节点缺 hostfile 与历史 `RC=255` | P0-3 与 P1-6 的跨节点背景 |
| [`docs/kupl_kunpeng_hpc_config_20261112.md`](kupl_kunpeng_hpc_config_20261112.md) | 鲲鹏 HBM 节点、HMPI 映射、`dsub` 调度与 `execNodes` 核对、超时协议 | 远端环境与两节点作业形态的背景，不是 L48 实测 |
| [`docs/qcu_quda_solver_dcu_l48_20261029.md`](qcu_quda_solver_dcu_l48_20261029.md) | L48 容量门与 Legacy-MG cache 逻辑 payload（`18,728,091,648 B`） | 仅作容量预算；其 `BLOCKED-BY-USER` 与本轮 LX2 远端状态无关 |
| [`docs/pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) | `pyqcu.test` L48 双节点设计、状态表（§0）、远端基线（§2.4）与开放问题（§9） | 设计/状态主文；本文只登记风险 |
| [`logs/v202611171402.txt`](../logs/v202611171402.txt) | 上一批次的远端基线、L48 转换、两节点探测、CSV/Excel 需求 | 远端事实的原始任务记录 |
| [`logs/v202611171416.txt`](../logs/v202611171416.txt) | 本批次归档、命令结果与阻塞 | 本轮任务记录 |

## 2. P0 风险（全部未修复）

| ID | 风险 | 证据锚点 | 影响 | 状态 |
|---|---|---|---|---|
| **P0-1** | 发布树的 `--gauge-file`/`--source-file` 只是**台账证据**，不驱动原生 solve；`--mode solve` 仍可能回退到 tier-0 executor | `pyqcu/test/common.py:1571`（`MODE_HOOKS` 声明 `driver:("run_row",)`）、`:2683`（查找 `driver`）、`pyqcu/test/README.md:206`、`:237`；`git cat-file -e HEAD:pyqcu/test/driver.py` 非 0（本机为 128，发布树无该文件） | 传入 L48 HDF5 也可能得到 `8x8x8x16` 的 `PASS`；把该 PASS 当 L48 结果是**误报** | **未修复（NOT FIXED）**；工作区 `pyqcu/test/driver.py` 为未跟踪并发实现，未提交、未审计 |
| **P0-2** | 发布树没有 `driver.run_row`，full-mode 的 `solve/mg` 行没有原生全格点执行器 | `pyqcu/test/common.py:1571` 的 `MODE_HOOKS`、`:2683` 的 `mode_hook("driver","run_row")`；HEAD 无 `pyqcu/test/driver.py`；`pyqcu/test/README.md:202`（`--ranks` 只记录） | `--ranks`/`--threads`/`--shm` 只进台账；没有 L48 原生 solve，任何 L48 `PASS` 都缺少执行路径 | **未修复（NOT FIXED）**；工作区 `driver.py` 是 observation-only 候选，仍未做 `mpirun` 启动 |
| **P0-3** | 没有跨节点 launcher：`pyqcu.test` 与 `pyqcu/testing/simple/mpi` 都不解析 `CCS_ALLOC_FILE`、不生成 hostfile/rankfile | `pyqcu/test/driver.py:50`（明确 no rankfile/`CCS_ALLOC_FILE`）、`:380-388`（`launcher` 标 `mpi_launch=false`）；`pyqcu/test/AGENTS.md:105` 与 `pyqcu/test/README.md:226` 也写“仍未实现”；`pyqcu/testing/simple/mpi/solve_case.py:327`、`_entrypoint.py:407` 的 `build_mpirun_command` 只拼 `--bind-to/--map-by/-np`；`docs/kupl_mpi_scaling_20261114.md:32-33`（历史两节点 `RC=255`） | `dsub -nn 2` + 一条 `mpirun` 无法组成；两节点 row 只会 `SKIP(mpi_grid)` 或启动失败，**不可能产生两节点证据** | **未修复（NOT FIXED）** |
| **P0-4** | CUDA tier-0 的 rank-grid 校验是 **SKIP 而非覆盖**：MPI size≠1 时 `_probe` 直接跳过，多 rank 的 `_require_grid` 路径不可达 | `pyqcu/test/cuda_qcu.py:56-58`（多 rank 跳 `mpi_single_rank`）、`:621-623`（`mpi_size != 1` → `_skip`）、`:639-646`（`_require_grid` 只比较 `prod(grid)` 与 `mpi_size`） | `--check`（远端 18/18；本机工作树 20/20）是**单 rank** 正确性；CUDA 多 rank 网格/设备绑定缺陷不会被它发现，不能据此证明跨 rank 可用 | **未修复（NOT FIXED）** |
| **P0-5** | MPI 失败语义是 **abort/deadlock 风险**，不是分级 FAIL：KUPL C ABI 把 C++ 异常转成 `abort_with_message`（`MPI_Abort` + `std::abort`），`pyqcu.test` 没有针对 full-mode MPI 的 abort/超时协调 | `pyqcu/test/kupl_qcu.py:107-108`；`pyqcu/test/native.py:134`（C++ MG 失败直接 abort）、`:2575`（Legacy-MG 同）；`grep -rn "Abort" pyqcu/test/*.py` 未发现 MPI abort 守卫 | 一个 rank 的原生错误会终止进程；其余 rank 若停在 collective，作业可能**挂起**而不是写出 `ERROR/TIMEOUT` 行；操作者会把挂起误判为“还在算” | **未修复（NOT FIXED）**；风险来自代码走查，**本轮未在远端复现** |
| **P0-6** | `_SET_INDEX_` 游标不在对照/恢复范围：控制快照刻意排除它，MG 包装又使用 30+ 槽位 | `pyqcu/test/cuda_qcu.py:208-220`（`_CONTROL_PARAM_NAMES` 不含 `_SET_INDEX_`，注释称其“expected to move”）、`:1397-1398`（smoke 手工设 0/1）、`:827`（`_set_index`）、`:78`（30+ 槽位保留给 MG） | 某个 runner/MG 调用把游标留在推进后的位置时，下一个 case 可能选到**错误槽位**；`globals_unchanged` 只检查模块级数组，不检查 runtime 游标 → 错误 buffer 或 abort，且不会被现有 PASS/FAIL 捕获 | **未修复（NOT FIXED）**；风险来自代码走查，**未复现** |

## 3. P1 风险（全部未修复）

| ID | 风险 | 证据锚点 | 影响 | 状态 |
|---|---|---|---|---|
| **P1-1** | 发布树没有 `L48` preset | `git show HEAD:pyqcu/test/common.py` 的 `LATTICE_PRESETS` 只有 `L24`/`L32`（HEAD `:1539-1542`）；工作区 `:1545-1549` 才加入 `L48=(48,48,48,96)`（未提交） | 0.1.17 上 `--lattice-preset L48` 被 argparse 拒绝；L48 只能靠 `--lattice-size` 表达，且仍不产生生产行 | **未修复（NOT FIXED）**；工作区有候选 |
| **P1-2** | 即使接受 `L48` preset，`--mode list` 仍只展开 17 个 tier-0 case（CG 退役后 19−2），没有 L48 生产 `solve/mg` 行 | `python -B -m pyqcu.test --mode list --lattice-preset L48 --json -` → 17 行，首行 `torch/lattice`、`[48,48,48,96]`、c64；`--lattice-size 48 48 48 96` 同样 17 行 | 没有可执行的 L48 row；100 列 CSV/21 sheet 无法在 L48 上产生证据 | **未修复（NOT FIXED）** |
| **P1-3** | 旧 KUPL 全矩阵驱动没有 L48，也没有 LIME 分支 | `pyqcu/testing/kupl/test_kunpeng_all.py:87`（`LATTICES` 只有 L24/L32）；全文输入只经 `h5py` 的 `h5_readable`/`prewarm_h5`（`:310`） | 旧驱动无法直接消费 L48 LIME；它也不能产出 L48 的 100 列/21 sheet 证据，替换关系无法在 L48 上闭环 | **未修复（NOT FIXED）** |
| **P1-4** | `submit_kunpeng_all.sh` 写死单节点与 8 GB：`-nn 1`、`-N $RANKS`、`-rpn $RANKS`、`mem=8G` | `pyqcu/testing/kupl/submit_kunpeng_all.sh:129`（`-nn 1`）、`:137-140`（`-N $RANKS`/`-rpn $RANKS`/`mem=8G`）、`:94-97`（`ranks*threads>608` 直接退出 2） | 不能申请两节点；L48 的 gauge（约 5.7 GB LIME / 6.1 GB c128 HDF5）、source 与 17.4 GiB cache 预算也远超每任务 8 GB | **未修复（NOT FIXED）** |
| **P1-5** | L48 c64 dtype 未确认：`gauge_..._c64.h5` 实为 complex128，c64 cast 与 `io.describe` 回读未完成 | `gauge_48x48x48x96_c64.h5` = `6,115,301,376 B`、`dataset.dtype == complex128`；source 代码默认写 complex64（`prepare_case.py:85-101`）；`pyqcu.test.io.describe(..., role="gauge")` 尚未在远端回读 | 把 c128 文件当 c64 用会同时误导 dtype 门与容量预算；在回读前任何 c64 行都只能是 **UNVERIFIED** | **TODO / 待验证**；不是已修复 |
| **P1-6** | 两节点探测 `2152277` 仍 PENDING：`No available node can fit ... execution nodes preference` | `djob -l 2152277`；无 `execNodes`、无两节点成功记录；同时刻队列 127 running/211 pending，仅 7 个 `OK` 节点 | 没有跨节点证据；任何“两节点可跑/已通过”都是把 PENDING 写成 PASS | **PENDING / 未通过** |
| **P1-7** | CSV/Excel parity 是硬性要求，但尚未接线、尚未跑完 | `pyqcu/test/ledger_csv.py`、`ledger_excel.py`、`pyqcu/testing/python/test_ledger_parity.py` 均未跟踪；`__main__.py`/`common.py` 未引用；`--csv` 仍写 30 列 `ROW_FIELDS`（`common.py:1586`） | 下游 100 列 `CSV_COLUMNS` 与 21 sheet 工作簿契约不成立；不能声称已替换旧驱动的台账 | **未修复（NOT FIXED）** |
| **P1-8** | 远端原始输出未归档：`--check`、版本探针、`prepare_case.py` 41 s 计时、`djob -l 2152277` 都不在仓库 | 本任务只记录命令形状与摘要；原始 stdout 未被保存 | 证据只能靠日志叙述，无法独立复核；复跑前不能把摘要当原始证据 | **TODO** |
| **P1-9** | 文档计数漂移：根 `AGENTS.md:29,64,88` 与 `skills/AGENTS.md:93` 曾写 21 个 tier 0，实际 `--list` 与 `pyqcu/test/README.md` 为 19 | `python -B -m pyqcu.test --list --json -` → 19；`--check` 报告 19 个 case（远端 0.1.17 树为 18/18，本机工作树为 20/20；11 torch, 5 kupl, 3 cuda）；2026-11-17 docs 门批次已把 8 处旧 tier-0 计数 21 改为 19、6 处旧 CUDA 计数 5 改为 3 | 报告与索引对不上；不影响运行时，但会影响审计结论 | **FIXED（文档计数，2026-11-17 docs 门批次；未提交）** |
| **P1-10** | `--mode validate` 名称冲突：旧驱动是数值冒烟，正式端口是台账一致性校验 | `pyqcu/test/common.py` 的 validate 分支；`pyqcu/testing/kupl/test_kunpeng_all.py:1749`（强制 `8 8 8 16`、2 rank、`shm=0`） | 操作者按旧习惯跑 `--mode validate` 会得到“空台账退出 1”，或被误读成数值失败；数值冒烟与台账校验必须分开命名 | **未修复（NOT FIXED）** |

## 4. 与 L48/两节点结论的边界

- **不得**声称 L48 solve、L48 MG 或两节点 PASS：本文所有 P0/P1 都未修复，且没有
  L48 数值结果或 `execNodes` 证据。
- `pyqcu.test --check` 的 **远端 18/18**（本机当前工作树 20/20）是**单 rank 的正确性
  自检**，不是 L48 能力、不是跨节点能力，也不是 CSV/Excel parity 的证明。
- 工作区未提交的 `driver.py`、`LATTICE_PRESETS` 的 `L48`、`native.py` 的 HDF5 输入
  身份只是**候选实现**；在提交、审计、远端复跑之前，不能把 P0-1/P0-2/P1-1 改写成
  “已修复”。
- L48 容量数字（`18,728,091,648 B` 逻辑 payload）来自
  [`qcu_quda_solver_dcu_l48_20261029.md`](qcu_quda_solver_dcu_l48_20261029.md) 的
  预算，不是 LX2 远端实测；不得把预算写成已分配/已通过。

## 5. 只读复核命令

以下命令只用于复核“发布树有没有修复”，不提交作业、不改代码：

```bash
cd /home/zhangxin/PyQCU

# P0-1/P0-2：发布树是否有 driver.py / ledger 模块（非 0 = 缺失；本机为 128）
git cat-file -e HEAD:pyqcu/test/driver.py; echo "driver=$?"
git cat-file -e HEAD:pyqcu/test/ledger_csv.py; echo "ledger_csv=$?"

# P1-1：发布树的 preset 只有 L24/L32
git show HEAD:pyqcu/test/common.py | grep -A3 '^LATTICE_PRESETS'

# P0-3：跨节点 launcher 关键字
grep -rn "hostfile\|rankfile\|CCS_ALLOC_FILE" pyqcu/test pyqcu/testing/simple/mpi

# P1-2：L48 仍只展开 17 个 tier-0 case
python -B -m pyqcu.test --mode list --lattice-preset L48 --json - | \
  python -c 'import json,sys; d=json.load(sys.stdin); print(len(d["rows"]["items"]))'
```

## 6. 变更记录

- **2026-11-17 14:16**：首次建立 P0-1…P0-6、P1-1…P1-10 登记；本轮**没有修改任何
  核心代码**，没有提交、没有 push、没有打 tag；全部条目状态为 **未修复 / TODO /
  PENDING**。后续若修复，必须在同一条目追加“修复提交 + 远端复核命令 + 结果”，
  不能只把状态改成“已修复”。
