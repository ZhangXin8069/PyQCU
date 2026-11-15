# KUPL 鲲鹏矩阵本轮范围裁定：MPI h5py、放弃项、L32 source/cache 与队列（2026-11-15）

> 证据等级：**B/C 混合，纯裁定文档**。本文只写本轮**已确认**的"做什么 / 不做什么"及其
> 判据，不产生新的性能数字，不重跑任何用例、不新增任何用例。
> - 矩阵枚举、94 列 CSV 口径、42 行快照与失败清单归
>   [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md)；
> - 生产 HDF5 输入不可读的根因与写/读侧修复归
>   [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md)；
> - **本文只负责范围裁定**：MPI h5py 的处置、放弃轴清单、L32 的 source/cache/重测判定、
>   以及被 kill 的作业与队列残留。
>
> 状态：时点证据（2026-11-15 17:0x 本地），未打 tag；不得升级为永久性能结论。
> 本轮**没有**改代码、没有改驱动、没有 commit/push、没有删除任何数据。

## 0. 四条结论（先读这个）

| # | 结论 | 一句话依据 |
|---|---|---|
| 1 | **MPI h5py 只做接口检查，不重新编译/安装** | 计算节点 conda `2023.3` 的 h5py `3.7.0` / HDF5 `1.10.6` 实测 `get_config().mpi=False`、`mpio` driver 不存在、`h5py_has_mpi()` 为 `False`；没有任何一行记录 `hdf5_mpio_read=true` |
| 2 | **6 个轴/组合放弃**（thread 9、seed24、random gauge、random 源、point 源、mg-restart 50） | 耗时代价过大；已有部分数据保留，不重测、不新增 |
| 3 | **L32 的 6 条 FAIL 是"缺一次有效运行"，不是"重测"**；cache 侧只需重签 L32 c64 与 L24 c64 的 identity | 6 条都用了坏 ones `a9c2acd8…`；数值等价的 cache 不重建 |
| 4 | **`bg-26/27/28/32` 已 kill，但其 pending 作业未能在 CLI 取消** | `djob -c` 与 `dsub -c` 均报 `invalid option`；保留作业不得再 kill |

## 1. MPI h5py：只做接口检查，不重新编译/安装

### 1.1 实测能力（计算节点，本轮）

| 项 | 实测值 | 判定 |
|---|---|---|
| h5py | `3.7.0` | 无 MPI 构建 |
| HDF5 | `1.10.6` | 同上 |
| `h5py.get_config().mpi` | `False` | 驱动侧唯一判据 |
| `mpio` driver | 不存在 | 集体读不可用 |
| `h5py_has_mpi()` | `False` | 见 `pyqcu/testing/kupl/test_kunpeng_all.py:1090-1096` |

`h5py_has_mpi()` 只读 `h5py.get_config().mpi`，不尝试创建 `driver='mpio'` 文件；因此它
给出的"无 MPI"结论是**能力声明级**的，不是探测失败。

> 这条结论**只对计算节点的那个 Python 环境成立**：本地开发机（非计算节点）实测是
> h5py `3.16.0`/HDF5 `2.2.0` 且 `get_config().mpi=True`。因此不能把任意一台机器上的
> 探测结果当作本批能力，必须按批次在**计算节点环境**内记录（这也正是 §1.3 保留
> "接口检查"的原因）。

### 1.2 这条能力如何影响本轮行为

1. **包装器自动开启分支不会触发**：`test_kunpeng_all.py:718-721` 是
   `if h5py_has_mpi(): env["KUPL_HDF5_MPIO_READ"]="1" else: env.pop(...)`。本轮走 `else`
   分支，`KUPL_HDF5_MPIO_READ` 被主动移除，因此底层
   `kupl_multigrid._hdf5_mpio_read_enabled()`（`pyqcu/testing/simple/kupl_multigrid.py:95-112`）
   看到的请求值为 0，集体 opt-in 与"rank 间不一致立即抛错"的校验都不会进入。
2. **没有任何一行记录 `hdf5_mpio_read=true`**：42 行里 28 行显式写 `false`（并带 note
   `h5py lacks MPI; HDF5 parallel read disabled`，`test_kunpeng_all.py:1442-1443`），
   另有 14 行（5 条 prepare + 9 条 solve，由更早的脚本版本写出）缺该键（`null`）。
   这是**记录下来的能力事实**，不是失败信号。
3. **读路径回退为每 rank 全量读**：`_read_local_slab`（`kupl_multigrid.py:1210`）在无 mpio
   时由每个 rank 独立 `h5py.File(path,"r")` 打开整文件再取本地 slab。32 rank 读 L32 那份
   1,207,965,696 B 的 gauge ≈ **38.6 GB** 聚合流量，这是 `gauge_read_seconds` 里
   `2.4–5.2 s`（warm）与 `29.9–35.6 s`（cold）差距的主要来源。
4. **ranked cache 被 blocked**：`prepare_cache_rows` 在 `kind == "ranked" and not
   h5py_has_mpi()` 时直接产出 `status=SKIP`（`test_kunpeng_all.py:1123-1131`），本轮落在
   `cache-042c63a815c8d0b3` 这一行；12 条 `mg2` 行全部退回 global cache，`balanced` 行另带
   `ranked cache unavailable; using global cache`。

### 1.3 明确不做的事

- **不重新编译、不重新安装带 MPI 的 h5py**；不换 HDF5 构建。
- **`bg-3` 的离线构建、`bg-23` 的 ranked cache 尝试均不作为本轮交付**；它们的产物未经本轮
  验证，**不得引用**（包括版本号、so 路径、构建耗时、任何"已可 mpio"的说法）。
- **ranked cache 不列入后续计划**：它需要 mpio，而 mpio 不在本轮范围。后续批次不排 ranked
  cache 行，也不为它预留作业。
- 保留的只有两件事：
  1. **接口检查**：每批开跑前在同一 Python 环境记录
     `h5py.__version__`、`h5py.version.hdf5_version`、`h5py.get_config().mpi`，以及
     `h5py.h5f.registered_drivers()` 里是否有 `mpio`（该 API 不存在时只记前两项），并把
     结果写进批次日志；
  2. **capability-aware fallback**：`h5py_has_mpi()` 为假时——不设
     `KUPL_HDF5_MPIO_READ`、不排 ranked cache、I/O 预算按 per-rank 全量读估算
     （聚合流量 = 文件字节 × rank 数），并把 `prewarm` 视为唯一的低成本重复读手段。

> 反模式：把"编译/安装 MPI h5py"写成下一步。本轮已把这条从计划里删除；需要再评估时必须
> 作为独立任务重新立项，不能在矩阵结论里顺手带出。

## 2. 放弃项（耗时代价过大）

以下 6 个轴/组合**不再投入机时**。统一标注：
**"耗时代价过大，已有部分数据保留，不重测/不新增"**。

| 轴 / 组合 | 已保留数据（`rows/*.json`，n=1） | 处置 |
|---|---|---|
| thread 9（16×9、32×9、32×9 shm1 复核） | 3 行：`b614ad9349f5d4e6`（16×9 shm1 `0.442830 s`）、`baf92e2b99937b2b`（32×9 shm0 `0.359793 s`）、`861ac912ac3c7c42`（32×9 shm1 `11.437897 s`，离群） | 不扩展、不重测；`861ac912` 的 31.8× 偏差**不再复核**，引用时必须剔除 |
| seed24 | 2 行：`0041182f2d8e1c50`（real1）、`9cc71c5d52dbbf22`（random 源） | 不扩展；`0041182f` 在带宽崩塌窗口（`0.19 GB/s·rank`、`20.86 s`），`9cc71c5d` 正常（`12.76 GB/s·rank`、`0.341 s`、19 it）但属已放弃的 random 源 |
| random gauge | 1 行：`4c5cd8a65e492c1e`（`6.358685 s`、7 it） | 不扩展、不重跑 |
| random 源 | 2 行：`9cc71c5d52dbbf22`、`e1574f172b525c02` | 不新增 |
| point 源 | 1 行：`c3bd73d2999000f0`（FAIL） | 源文件已修复，但 **point 性能不跑**；该 FAIL 行保留不删 |
| mg-restart 50 | **0 行** | 不新增 |

补充说明：

- **point 源虽已修复，point 性能不跑**：`source_24x24x24x64_c64_point.h5` 已在 16:28 换成
  可读文件，但 point 源的迭代数、残差口径与 `ones` 不同，补齐它需要另一整轴机时，本轮不做。
- **seed24 / random / thread 9 的部分行不扩展**：它们已经作为"覆盖缺口"留在台账里，
  缺口的标注本身就是结论，不再用更多单样本去填。
- **r50 为 0 行**：`mg_restart=50` 在 37 条 solve 中一条都没有，且不属于本轮要补的项。
- `mg1` 在 42 行快照里也是 0 行，但它**不在放弃清单**（见下）。

### 2.1 保留（继续做）

| 项 | 现状 | 后续 |
|---|---|---|
| `16×36`（thread 36） | 已有 **3 行**：`b63ba9c835d4f81c`、`095ce6dc4a35ece1`（L24 16×36 t-finest shm1/shm0）、`220f00939aeda0ab`（L24 16×36 mg2 r5） | 保留，继续（32×36=1152 > 608 仍按设计不测） |
| MG1 | 0 行 | 继续；但要先接受 `mg1` 与 `bistabcg` 在驱动里参数完全等价（都是 `levels=1`/`restart=0`），补它等于补重复标签 |
| r10 | 0 行（`mg_restart` 只有 5→8 行、20→4 行） | 继续 |
| L32 / c128 | L32 6 条 FAIL、c128 6 行（L24 4 PASS + L32 2 FAIL） | 继续（见 §3） |
| MG + shm | 12 条 `mg2` 行，含 1 条 `shm=1` 握手超时 | 继续；`b3267714` 的握手超时经同 config 2/2 复测判定为**不稳定失败**（1 FAIL / 2 PASS），不是 known skip；不得整体禁 `balanced` 或 `shm1`，也不得静默回退 `shm=0`，见 [`kupl_l24_c64_mg2_shm1_boundary_20261115.md`](kupl_l24_c64_mg2_shm1_boundary_20261115.md) |

## 3. L32 source / cache / 重测判定

### 3.1 source 语义：`source_..._c64.h5` 不是 ones

`data/production_L32_v1/` 下同时存在三个"看起来像 ones"的文件，语义完全不同：

| 文件 | 实际内容 | 能否当 ones 用 |
|---|---|---|
| `source_32x32x32x64_c64.h5` | **random，seed 42** | **不能**；它不是 ones，替换坏 ones 会改变源的物理语义 |
| `source_32x32x32x64_c64_ones.h5` | 全 `1+0i`（wall 源） | 能；16:28 起已换成可读文件 |
| `source_32x32x32x64_c64_ones_v2.h5` | 全 `1+0i`（wall 源） | 能；16:28 前是唯一可读版 |

**`ones.h5` 与 `ones_v2.h5` 的关系已闭合**：

- `file_sha256` 同为 `9cdcfc821437816f342f74859ce7c5a44eae8f02b4fe6df7eba2c34a9b2ef55a`
  （`9cdcfc82…f55a`）；
- dataset 逐元素为 `1+0i`；
- 两者**逐字节相同**（同 sha256 + 同 201,375,008 B）。

因此 **不需要重新生成 L32 ones**。凡是要 L32 wall 源的地方，直接用
`source_32x32x32x64_c64_ones.h5`（= `ones_v2` 的内容）；不要用
`source_32x32x32x64_c64.h5`，也不要为了"文件名里没有 `_ones`"去新建文件。

同目录还有第三个可读源、语义同样不能混：`source_32x32x32x64_c64_point.h5`
（`dc427e47…`，point 源）。快照后的 `82ba09ec3a8e0d2d` 用的就是它，所以那条 L32 PASS
是 **point 源 + c128 + `mg2`** 的组合，与 6 条 FAIL 的 ones 配置不是同一口径。

本轮全部输入身份（供引用时对照，行数统计自 42 行快照）：

| 角色 | `sha256` 前缀 | 行数 | 语义 |
|---|---|---:|---|
| L24 wall 源（`source_24x24x24x64_c64_ones.h5`） | `486d358e` | 29 `ones` + 1 `real1` | 全 `1+0i`，`complex64` |
| L32 wall 源（修复后 = `ones_v2`） | `9cdcfc82` | 0（待首跑） | 全 `1+0i`，`complex64`，201,375,008 B |
| L32 wall 源（坏，`.corrupt-20261115`） | `a9c2acd8` | 6 FAIL | 行里记录的就是它 |
| L32 point 源 | `dc427e47` | 0（快照后 1 条 PASS） | point；dtype 未单独复核（**待确认**），文件名同族 |
| L24 point 源（坏 → 16:28 重生成） | `945782f3` → `b437e68f` | 1 FAIL | point |
| L24 random 源 | `af13dcc3` / `af8a3188` | 各 1 | random（已放弃轴） |
| L24 real gauge（`gauge_24x24x24x64_c64.h5`） | `4777e795` | 32 | disk 上是 `complex128`，509,614,080 B |
| L32 real gauge（`production_L32_v1/gauge_32x32x32x64_c64.h5`） | `b2a57594` | 4（全 FAIL） | disk 上是 `complex128`，1,207,965,696 B |
| L24 随机 gauge（`gauge_random_..._seed42.h5`） | `25603783` | 1 | 真 `complex64` |

### 3.2 修复落点

| 项 | 状态 |
|---|---|
| L32 `ones` | 2026-11-15 **16:28** 已修复（可读、`complex64`、`(4,3,32,32,32,64)`） |
| L24 `point` | 同日 **16:28** 已修复（可读、`complex64`、`(4,3,24,24,24,64)`，新 sha `b437e68f…`） |
| 坏字节 | 保留为 `*.corrupt-20261115`（L32 ones `a9c2acd8…`、L24 point `945782f3…`），**未删除** |
| 旧 FAIL 行 | `rows/*.json` 与 CSV **未改、未删**；42 行快照仍是 16:06 的那一份 |

### 3.3 6 条 L32 FAIL：缺一次有效运行，不是重测

| run_id | dtype | grid_mode / grid | shm | 说明 |
|---|---|---|---:|---|
| `07c80fc8` | — | — | — | 快照后新增（本地无 row），同一坏 ones |
| `23daf98e9c477bdd` | c128 | t-finest `1 1 1 32` | 0 | 42 行快照内 |
| `327f9aa531167d55` | c64 | t-finest `1 1 1 32` | 0 | 42 行快照内 |
| `35548a2b06ff0989` | c128 | t-finest `1 1 1 32` | 1 | 42 行快照内（失败与 shm 无关） |
| `44faded2` | — | — | — | 快照后新增（本地无 row），同一坏 ones |
| `c9bfd6bfb733b052` | c64 | balanced `2 2 2 4` | 0 | 42 行快照内（失败与 grid 无关） |

判定：

- 6 条 **全部**用坏 ones（`source_sha256 = a9c2acd8b36e6cab…`），在 source 读取阶段就退出，
  solver 未启动 → **没有有效性能数据**。
- 因此它们不是"重测"，而是 **L32 的首次有效运行**；重跑时必须用修复后的 ones 文件，
  并以 `--run-label` 开新 `run_id`（旧 FAIL 行保留）。
- **L24 point `c3bd73d2` 已放弃**（见 §2）；它不再进入任何补跑清单。
- **快照后的 `82ba09ec3a8e0d2d` 不能抵这 6 条**：它是 L32 c128 `mg2` t-finest shm0 32×18
  restart5、用 `source_32x32x32x64_c64_point.h5`（`dc427e47…`）跑出的 PASS（残差 `4.03e-10`），
  口径与 6 条 FAIL（ones 源、bistabcg、t-finest/balanced × shm0/1、c64/c128）不同，且只有
  1 个样本。它的作用是证明"source 可读后 L32 gauge + c128 cache 能算"，**不是**补齐。

### 3.4 cache 判定：能复用 ≠ identity 正确

`cache_index.json` 的 4 条 global cache 与它们的 identity 处置：

| cache | 大小 | `identity.gauge_file_sha256` | `identity.gauge_canonical_sha256` | 处置 |
|---|---:|---|---|---|
| L32 c64（`2139960e885a0627`） | — | `b2a57594…`（**1.2 GB complex128 文件**） | `9ac7c15c…` | **重签/重建 identity**：改用 `production_L32_c64/gauge_32x32x32x64_c64.h5`（604 MB，file `70bbe621…`，canonical `9ac7c15c…`）；数值不变 |
| L32 c128（`3b6e1e37b3af0b9c`） | 679 MB | `b2a57594…`（同上，c128 文件） | `52f3c07a…` | **不重建**：它本来就是 c128 canonical |
| L24 c64（`b708cefe4583941a`） | 143,335,424 B | `4777e795…`（509 MB complex128 文件） | `1748b007…` | **建议重签**：正确 c64 文件为 `production_L24/gauge_global_c64.h5`（file `8a482d6a…`） |
| L24 c128（`9b7a383e23a633be`） | — | `4777e795…` | `b2e00420…` | 不重建（c128 canonical 本来就来自 c128 文件） |

读法：

1. **"数值等价"与"identity 正确"是两件事**。c128 文件 cast 到 c64 之后与原 c64 文件的值
   等价（L32 c64 的 canonical `9ac7c15c…` 与 604 MB c64 文件的 canonical 相同），所以
   **31 条 PASS / L24 数据不因此作废**；但 identity 里的 `gauge_file_sha256` 指到了另一份
   （更大的 c128）文件，这会让"按 identity 复用"在换回 c64 文件时误判 miss/冲突。
2. **L32 c64 cache 要重签或重建**：换成 `production_L32_c64/gauge_32x32x32x64_c64.h5`
   （604 MB）后，canonical 仍是 `9ac7c15c…`，因此**不需要重跑数值**；只需把 identity 的
   `gauge_file_sha256` 重签为 `70bbe621…` 并复核 `cache_identity_sha256`，或按新 identity
   重建一次 cache。
3. **L32 c128 cache 不重建**（679 MB，`b2a57594…`/`52f3c07a…`）。
4. **MG cache 不含 source 哈希**：identity 只有
   `lattice/dtype/mass/csw/mg_grid/nv_iters/dof/seed/gauge_kind/ranks/grid`（见
   [`pyqcu-testing-data`](../skills/pyqcu-testing-data/SKILL.md)）。source 修复**不使任何
   MG cache 失效**。
5. **L24 c64 cache 同类 identity 问题**：正确文件为 `production_L24/gauge_global_c64.h5`
   （file `8a482d6a…`），建议重签；同样不重跑 L24 数值。

> 口径提醒：`gauge_file_sha256` 与 `gauge_canonical_sha256` 不是同一个量。前者是磁盘文件
> 字节哈希（随 c64/c128 副本而变），后者是转成 canonical 布局/dtype 后的内容哈希（同值
> 等价时不变）。换文件前先算 canonical，只有 canonical 变才必须重建。

本地可复核的 cache identity 全字段（`data/kunpeng_all_20261115/cache_index.json`，
不需要远端即可核对）：

| cache | `cache_sha256` | `cache_identity_sha256` | `build_mg_cache_seconds` | notes |
|---|---|---|---:|---|
| L32 c64 `2139960e885a0627` | `96238dca…` | `e49fb38a61d079a0` | 100.576 | `global build rc=124 elapsed=602.2s` |
| L32 c128 `3b6e1e37b3af0b9c` | `7c6abf0f…` | `717438eecee27861` | 130.371 | `global build rc=124 elapsed=602.1s` |
| L24 c64 `b708cefe4583941a` | `2e5a463c…` | —（缺键） | 493.638 | `setup_method=cxx-null-vector+local-33-stencil` |
| L24 c128 `9b7a383e23a633be` | `dcf6473b…` | —（缺键） | **48.191** | 同上；构建并行度未逐条记录，与 c64 的 493.638 s 不可直接相减（**待确认**） |

另：快照后的 `82ba09ec3a8e0d2d` 用的就是上表 L32 c128 的 `7c6abf0f…` cache，并在
`verify_status=0` 下收敛到 `4.03e-10` —— 这是"c128 cache 不重建"的直接旁证。

### 3.5 重测判定规则（本轮统一口径）

**不重测**当且仅当以下各项全程同口径：

1. source 语义与 hash（ones / point / random 不可互换；`a9c2acd8…` 与 `9cdcfc82…` 不可混）；
2. gauge dtype 与 `gauge_canonical_sha256`；
3. cache identity（含 `gauge_file_sha256`、`cache_identity_sha256`、`verify_status=0`）；
4. `atol`（c64 `1e-6` / c128 `1e-9`，且带 `--strict-acceptance`）；
5. `grid`、`ranks`、`threads`、`shm`、`solver`、`mg_restart`。

任一不同 → 必须重跑，并在 `rows/` 里以新 `run_id` 落账。按这条规则：

- **31 条 PASS / L24 数据不重跑**：L24 c64 gauge 的 c128 文件 cast 后与 c64 值等价；L32 c64
  cache 数值等价（只差 identity 的文件哈希）；L32 c128 cache 不重建；MG cache 不含 source
  哈希，source 修复不使其失效。
- **L32 的 6 条 FAIL 不属于"重测"**，是首次有效运行（见 §3.3）。
- **L32 c64/c128 与 L24 的同口径对比在修复后有效**：修复只换 source 文件字节（语义同为
  wall 源），不改变几何、算法、精度门槛。

## 4. 状态与队列

### 4.1 已终止的后台任务

`bg-26`、`bg-27`、`bg-28`、`bg-32` **已 kill**。它们对应的轴与残留队列状态：

| 后台任务 | 对应轴 | 本轮结论 |
|---|---|---|
| `bg-26` | thread 9（16×9、32×9、32×9 shm1 复核） | 放弃（§2） |
| `bg-27` | seed24 / random（源与组态） | 放弃（§2） |
| `bg-28` | point 源 | 修复但性能不跑（§2） |
| `bg-32` | mg-restart 50 | 放弃（0 行，§2） |

### 4.2 pending 作业未能在 CLI 取消

- `djob -c` 与 `dsub -c` **都报 `invalid option`**（当前 LX2 客户端没有取消子命令），因此
  上述作业**没有在 CLI 侧取消成功**。
- 后果：队列里这些作业**仍可能处于 `PENDING`**。用户可通过**调度器 UI** 取消；本轮不再
  尝试其它 CLI 手段（`dkill`/`dsub` 其它开关不在本轮验证范围）。
- 这些作业即使后来被调度执行，其产出也属**放弃轴**，不得据此新增结论；若真跑出结果，
  按 §2 的标注原则归档为"代价过大、保留但不扩展"。

### 4.3 保留作业（不得 kill）

以下作业族**保留**，不做任何终止操作：

- `kall_l32_*`（L32 修复后的有效运行依赖它）
- `kall_l24_c1*`（L24 c128）
- `kall_mg_shm*`（MG + shm）
- `kall_mg1_r5*`（MG1 / restart 5）
- `kall_mgr10*`（MG restart 10）
- `k9_16x36*`（16×36 / thread 36）

同时：**不删除任何数据**（`rows/`、`reports/`、`logs/`、`caches/`、`*.corrupt-20261115`
全部保留）。机器可读版本见
`data/kunpeng_all_20261115/known_skips.json` 与同目录 `quarantine.json`。

## 5. 本轮不做的动作（红线）

1. 不写代码、不改驱动（`test_kunpeng_all.py`、`submit_kunpeng_all.sh`、`kupl_multigrid.py`
   均未触碰）。
2. 不 commit / 不 push / 不打 tag。
3. 不改、不删、不重排 `rows/*.json` 与任何 CSV；旧 FAIL 行原样保留。
4. 不把"编译/安装 MPI h5py"写成下一步；不引用 `bg-3`/`bg-23` 的任何未验证产物。
5. 不 kill 保留作业；不因 identity 问题重跑数值（只重签/重建 cache 元数据）。
6. 不引用已放弃轴的单样本数字（尤其 `861ac912ac3c7c42` 的 `11.437897 s`）。

## 6. 证据与复核入口

| 证据 | 位置 | 说明 |
|---|---|---|
| `h5py_has_mpi()` 与自动 env 分支 | `pyqcu/testing/kupl/test_kunpeng_all.py:1090-1096`、`718-721` | `get_config().mpi` 单判据 |
| mpio 集体 opt-in 校验 | `pyqcu/testing/simple/kupl_multigrid.py:95-112` | rank 间不一致立即抛错 |
| per-rank 全量读回退 | `pyqcu/testing/simple/kupl_multigrid.py:1210` | `_read_local_slab` |
| ranked cache SKIP | `pyqcu/testing/kupl/test_kunpeng_all.py:1123-1131` | 产出 `status=SKIP` |
| `hdf5_mpio_read` 与 note | `pyqcu/testing/kupl/test_kunpeng_all.py:1442-1443`；`rows/*.json`（28 行 `false`、14 行缺键、0 行 `true`） | 无一行记录 `true` |
| 6 条 L32 FAIL 的坏 ones | `rows/{23daf98e,327f9aa5,35548a2b,c9bfd6bf}*.json` 的 `source_sha256`；`logs/*.err` | `a9c2acd8…` |
| cache identity | `data/kunpeng_all_20261115/cache_index.json` | 4 条 global cache |
| 损坏文件与修复 | `data/kunpeng_all_20261115/quarantine.json` 的 `input_remediation` | `*.corrupt-20261115` |
| 机器可读 skips | `data/kunpeng_all_20261115/known_skips.json` | 放弃轴 + known skip；其中 `b3267714` 条目已被 2/2 复测推翻，修订候选在 `known_skips.candidate.json`（默认 disabled，数据侧待合并） |

> 本机没有 `data/production_L24`、`data/production_L32_v1`、`data/production_L32_c64`，
> 也没有 cache `.h5`；604 MB / 679 MB / `70bbe621…` / `8a482d6a…` 这些文件级事实来自本轮
> 会话的远端只读核验（C 级），本地只能复核 `cache_index.json` 里已记录的哈希与
> `rows/*.json` 的 `source_sha256`。
