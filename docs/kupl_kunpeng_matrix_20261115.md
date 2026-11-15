# KUPL 鲲鹏全矩阵测试驱动与结果矩阵（2026-11-15）

> 证据等级：**B/C 混合**。驱动 `pyqcu/testing/kupl/test_kunpeng_all.py` 与提交脚本
> `pyqcu/testing/kupl/submit_kunpeng_all.sh` 在本机存在但**未被 Git 跟踪**；42 行台账是
> 远端 `data/kunpeng_all_20261115/**` 的本地副本（`data/.gitignore` 为 `*`，整个
> `data/**` 不跟踪）。本文数值全部由 `rows/*.json`、`logs/*.err|out`、`cache_index.json`
> 字段级读取得到，不是从别的报告转抄；性能解释与跨版本对照归
> [`kupl_performance_20261115.md`](kupl_performance_20261115.md)，
> 单节点最优点归 [`kupl_l24_perf_20261113.md`](kupl_l24_perf_20261113.md)，
> 扩展性与断崖判据归 [`kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md)，
> HBM/绑定/调度历史配方归
> [`kupl_kunpeng_hpc_config_20261112.md`](kupl_kunpeng_hpc_config_20261112.md)。
>
> 范围：KUPL CPU 后端；L24 `24³×64`、L32 `32³×64`；c64（相对 norm 门 `1e-6`）与 c128
> （`1e-9`）；节点 `cn19699`/`cn19650`、LX2 `q_hpcapp`。**37 条 solve 行是 37 个不同
> 单元，每个只有 1 个样本（n=1）**；本文不把任何单样本差 <5% 判成快或慢。
>
> 状态：时点证据（2026-11-15 16:06 本地快照），未打 tag；不得升级为永久性能结论。

## 0. 与既有报告的分工

| 主题 | 权威文档 | 本文的做法 |
|---|---|---|
| L24 单节点最优、分项剖析、OMP/HALO A/B | `kupl_l24_perf_20261113.md` | 只引用结论，不重复表格 |
| halo/计算比、roofline、`TH_max`、断崖 | `kupl_mpi_scaling_20261114.md` | 只引用判据 |
| dev143 对 stab64 的跨版本对照、可比性红线 | `kupl_performance_20261115.md` | 只引用"不可直接比"的结论 |
| HBM 绑定、MPI 映射、超时协议 | `kupl_kunpeng_hpc_config_20261112.md` | 只引用配方 |
| **矩阵枚举、驱动模式、CSV 口径、调度边界、本轮 42 行快照** | **本文** | — |
| **输入不可变、超时语义、写原子性与启动/loop 原型成本** | **本文 §12–§13** | 只落可复核规则、实测分段与复核命令；逐文件逐行号的审计明细以 2026-11-15 只读审计报告为准（该报告在会话工作目录，未入仓库） |
| **生产 HDF5 source/gauge 不可读的根因与修复** | [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) | 只给指针 |
| **本轮范围裁定：MPI h5py 接口检查、放弃项、L32 source/cache、队列状态** | [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) | 只给指针；本文不再重复"做什么/不做什么"的裁定 |

## 1. 可复用结论

1. **矩阵可以枚举，但远未跑完**：默认展开 **7,680** 行（`--mode list --expand` 实测行数），
   落盘只有 **37 条 solve**（0.48%），且没有任何重复样本。驱动当前是"口径机器"，
   不是性能数据源。
2. **L24 c64 的 2×2 全因子格是唯一可直接比较的部分**（同节点 cn19699、同 18 it、同
   gauge/source/seed，只变 grid 与 shm）：t-finest+shm `0.2405 s`、t-finest+MPI
   `0.2476 s`、balanced+shm `0.3133 s`、balanced+MPI `0.3497 s`。
3. **共享 halo 的收益依赖 grid**：balanced 上 `0.3497 → 0.3133 s`（−10.4%），
   t-finest 上 `0.2476 → 0.2405 s`（−2.9%）。与
   `kupl_l24_perf_20261113.md` 的 14.77% 不是同一批样本，不能混引。
4. **balanced 慢的原因是带宽而不是通信量**：同 18 it；balanced 的 dslash 字节反而
   更少（`857.309 MB/76`），但 rank-0 只有 `4.00 GB/s`（聚合 `128.10 GB/s`）；
   t-finest 是 `1210.319 MB/76`、`9.18 GB/s`（`293.76 GB/s`）。4D 邻居打散了局部性。
5. **c128 L24 32×18 t-finest+shm 为 `0.4756 s` / 31 it / 真残差 `5.33e-11`**；同格
   c64 为 `0.2405 s` / 18 it。两者差 1.978×，其中迭代数就占 `31/18=1.72×`，不能用
   这个比值当 c128 单次算子成本。
6. **L32 在 42 行快照内 0 PASS**：快照内 4 条 solve、快照后 2 条（`07c80fc8`、`44faded2`）共 **6 条**
   全部因生产 source 文件 `bad object header version number` 失败；2 条 L32 cache prepare 行虽
   `PASS`、`verify_status=0`，但 note 是 `global build rc=124 elapsed=602.2s`，不能算干净构建。
   6 条是"缺一次有效运行"，不是"重测"，裁定见
   [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3.3。
   **快照后已有 1 条 L32 PASS 旁证**：`82ba09ec3a8e0d2d`（L32 c128 `mg2` t-finest `1 1 1 32` shm0
   32×18 restart5、cn19699、16:32）`PASS rc=0`、真残差 `4.03e-10`，用的 source 是
   `source_32x32x32x64_c64_point.h5`（`dc427e47…`）、gauge 是同一份 1.2 GB `b2a57594…` 文件、
   cache 是 L32 c128 那份（`7c6abf0f…`）。它证明的是"source 可读后 L32 gauge + c128 cache 能算"，
   **不能**代替 6 条 FAIL 配置的首次有效运行，也不能把 42 行快照的 L32 覆盖率写成非 0。
   出处：`data/kunpeng_all_20261115/quarantine.json` 的 `post_snapshot_remote_evidence`。
7. **solve 墙钟不是主要成本**：MG 每 run 还有 `4.400–6.071 s` 的 cache 载入，L24 全局
   cache 一次性构建 `493.638 s`；整行 `elapsed_seconds` 为 `95–271 s`，主要是 Python
   导入、MPI 启动与 HDF5 读入。任何"端到端"表述必须把这三段分开。
8. **旧 `0.17 s` 与 `0.3497 s` 的 2.04× 已经可以分解**（本地审计 `data/kunpeng_all_20261115_audit/report.md` §3）：`1.6364`（18/11 it）× `1.4126`（grid）× `1.0295`（shm）× `0.8577`（内核）= `2.0412`。但第 4 个因子被固定开销摊销污染，合理区间是 `0.86–0.95`，**不能引用 `0.858` 这个点值当内核加速**。
   旧指的准确身份（归 [`kupl_performance_20261115.md`](kupl_performance_20261115.md) §4.1）：
   `0.171331–0.182659 s`（3 样本、中位 `0.173362 s`）、**`atol=1e-4` 且非 strict**（实际门槛被
   `max(atol, eps(float32)×2048)=2.44140625e-4` 抬高）、**11 it**、全 1 源、32×18 shm、dev143 批次。
   与本文矩阵的 `atol=1e-6`（31 条）/`1e-9`（6 条）、18 it 相差 **244.1×** 的有效容差，
   所以"旧 0.17 对 balanced 0.3497"只能做分解式的口径核算，**不能当成代码回退或加速**。

## 2. 矩阵枚举

### 2.1 默认维度（`build_parser`，行 1869–1935）

| 维度 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 格点 | `--lattices` | `L24,L32` | `L24=(24,24,24,64)`、`L32=(32,32,32,64)`（行 87） |
| 精度 | `--dtypes` | `c64,c128` | `--atol` 未给时按 dtype 自动取 `1e-6`/`1e-9`（行 175–176） |
| 求解器 | `--solvers` | `bistabcg,mg1,mg2` | `mg2` 为 `levels=2` |
| MG restart | `--mg-restarts` | `5,10,20,50` | 只对 `mg2` 展开；`mg1` 固定 0、`bistabcg` 固定 0（行 916–921、832–835） |
| 分格 | `--grid-modes` | `balanced,t-finest` | t-finest 恒为 `(1,1,1,ranks)`；balanced 由 `choose_grid` 取最多非单轴、最均衡的偶分解（行 249–263） |
| rank | `--ranks-list` | `16,32` | |
| 线程 | `--threads-list` | `9,18,36` | 与 rank 相乘后受 `--max-total-threads` 过滤 |
| 共享 halo | `--shm-list` | `0,1` | `KUPL_SHM_HALO`，行 717 |
| 源 | `--sources` | `ones,real1,point,random` | |
| 组态 | `--gauge-kinds` | `real,random` | |
| 种子 | `--seeds` | `42,24` | |

### 2.2 组合数与排除规则

**几何单元 8 个**（lattice × ranks × grid_mode）：

| 格点 | ranks | balanced | t-finest |
|---|---:|---|---|
| L24 | 16 | `2 2 2 2` | `1 1 1 16` |
| L24 | 32 | `2 2 2 4` | `1 1 1 32` |
| L32 | 16 | `2 2 2 2` | `1 1 1 16` |
| L32 | 32 | `2 2 2 4` | `1 1 1 32` |

**线程数过滤**：`valid_rank_thread` 要求 `ranks*threads <= 608`（行 93、266–267），
提交脚本同样在 `ranks*threads > 608` 时直接退出（`submit_kunpeng_all.sh` 行 94–97）。
因此 **32×36 = 1,152 > 608 被排除**，16×36 = 576 保留。合计
`4×3 + 4×2 = 20` 个 (lattice, ranks, grid_mode, threads) 单元。

**每个单元 384 行**：2 dtype × 6 个 solver/restart 变体（bistabcg 1 + mg1 1 + mg2 4）
× 2 shm × 4 source × 2 gauge × 2 seed。

`20 × 384 = 7,680`，与下面这条命令的输出行数一致：

```bash
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode list --expand | wc -l
```

**去重后约 5,760 个不同输入组合**：`ones` 与 `real1` 都映射到
`source_<lat>_c64_ones.h5`（`production_source_path`，行 290–303），所以 source 只有
3 个不同产物，`7,680 × 3/4 = 5,760`。引用组合数时必须说明用的是哪一个口径。

### 2.3 本轮实际覆盖

37 条 solve 行覆盖 37 个不同单元，全部 n=1：

| 维度 | solve 覆盖（37 条） | 含 5 条 prepare 时 |
|---|---|---|
| 格点 | L24 33、L32 4 | L24 36、L32 6 |
| dtype | c64 31、c128 6 | c64 34、c128 8 |
| solver | bistabcg 25、mg2 12 | 另 5 条 `cache_build` |
| 分格 | t-finest 28、balanced 9 | t-finest 32、balanced 10 |
| rank×thread | 32×18 15、16×18 16、16×36 3、32×9 2、16×9 1 | 32×18 20 |
| shm | 1 19、0 18 | 1 24、0 18 |
| 源 | ones 33、random 2、real1 1、point 1 | ones 36、random 4 |
| gauge/seed | real 36 / random 1；seed 42 35、seed 24 2 | real 41 / random 1；42 40、24 2 |
| restart | 0 25、5 8、20 4 | 0 25、5 13、20 4 |

`mg1`、`restart=10/50`、L32 的 c128 balanced 组合都没有任何行——这是覆盖缺口，
不是"已测未通过"。另外 **`mg1` 与 `bistabcg` 在本驱动里参数完全等价**：两者都是
`levels=1`、`restart=0`（行 832–835、916–921），只差 `solver` 标签。要么把 `mg1`
从 `--solvers` 默认值里去掉，要么在统计时把它们视为同一算法的重复标签。

## 3. 运行模式与命令行入口

### 3.1 七个模式

| 模式 | 是否要分配节点 | 行为 | 证据 |
|---|---|---|---|
| `list` | 否 | 展开矩阵并按 JSONL 打印；配 `--expand` 才逐行展开 | 本文 §2.2 的 7,680 行即此模式实测 |
| `prepare` | 是（cache 构建）；gauge/source 解析可在登录节点 | 解析/生成 gauge、source；构建或复用 Legacy-MG cache；写 `cache_index.json` | `rows/cache-*.json` 5 条 |
| `prewarm` | 否 | 显式预读 gauge/source，把页缓存加热（`read_mode=warm`） | 行 1622 起；本轮未产生独立账本行 |
| `solve` | 是 | 跑一个 shard，每条 solve 追加 `rows/*.json` 与 CSV | 37 条 solve |
| `merge` | 否 | 把 `rows/*.json` 合并成一个 CSV（94 列，可保留旧列） | 行 1529–1554 |
| `run` | 是（单机 `mpirun`） | 跑单条显式配置（`--lat/--mass/--csw/--dtype/--solver/...`） | 行 1727 起 |
| `validate` | 否 | 小格点冒烟：`8 8 8 16`、2 rank×2 thread、c64、atol `1e-5`、random gauge、`mg-grid 2 2 2 2` | 行 1749–1777 |

> `validate` 的本地快照（`data/kunpeng_all_20261115/validate/`，12:57）是 **FAIL**：
> `gauge missing` + `no report.json`，且该行 `gauge_kind=real`，与当前
> `mode_validate` 硬编码的 `random` 不一致——它是旧版本产物，**不能当作冒烟通过**。
> 冒烟必须在有生产 gauge 的节点跑，或显式 `--allow-missing-inputs`。

### 3.2 统一入口

所有模式共用同一组物理参数，便于把任意子矩阵原样提交到别的节点：

```bash
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode solve \
  --lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --dtype c64 \
  --atol 1e-6 --max-iter 50 --solver bistabcg --levels 1 --restart 0 \
  --grid 1 1 1 32 --ranks 32 --threads 18 --shm 1 \
  --source ones --gauge real --seed 42 \
  --mg-grid 2 2 2 1 --nv-iters 2 --dof 12 2 2 2 2 \
  --coarse-max-iter 200 --coarse-tol-factor 1e4 \
  --outdir data/kunpeng_all_20261115 --run-label <tag>
```

- `--lat/--mass/--csw/--dtype` 是四个主入口；`kappa` 由 `1/(2*mass+8)` 派生并落账
  （行 847），不要手填。
- `--atol` 缺省时按 dtype 取 `1e-6`（c64）/`1e-9`（c128）；驱动固定向底层追加
  `--strict-acceptance`（行 1318），所以台账里的 `atol` **就是**判据，不再被
  dtype 地板抬高。
- `--run-label` 只改 `run_id` 后缀，用来把同一配置的 cold/warm 或重跑分开
  （行 878–887）；不加标签时同配置会覆盖同一个 `rows/<id>.json`。

### 3.3 续跑、分片与输入策略

| 选项 | 语义 | 陷阱 |
|---|---|---|
| `--resume` | 跳过 `rows/` 里已存在的 `run_id`（判重只看 `outdir/rows/<run_id>.json`，**与 `--csv` 无关**） | 与 `--run-label` 一起用，否则"重跑"会被当成已存在；**换一个新 CSV 重跑同一 config 不会真的重跑**（实测作业 `2133280` 返回 `{"done": 2}`、0 条新行），要重测必须换新 `outdir` 或先让旧行失效 |
| `--skip-failed` | 在 `--resume` 之上再跳过 `status=FAIL` 的行 | **本轮 3 个 MG 空洞因此从未重跑**（审计 P3-3）；补跑前必须先让旧 FAIL 行失效（改名或删行） |
| `--shard-index/--shard-count` | 按 `rows[idx::count]` 取模分片 | 分片按展开顺序，不按单元分组；同一单元仍只出现一次 |
| `--limit` | 本作业最多跑多少行 | 与分片叠加，先分片后截断 |
| `--allow-missing-inputs` | 缺 gauge/source 时用 identity gauge / seeded random source 继续 | 结果不再对应生产 gauge，不能与生产行混比 |
| `--no-solution` | 不写 `solutions/*.h5` | 失去"用 gauge+src+dest 复算真残差"的独立校验入口 |
| `--timeout/--hard-timeout` | 审查阈值 / 硬终止阈值（默认 300/600 s） | 超时记为 `TIMEOUT`，`RC=124` 不等于死锁 |
| `--prewarm-inputs` | 每个输入先整体读一遍再进 `mpirun` | 只影响 `read_mode=warm` 与 I/O 计时，不改求解 |

输入解析顺序（`resolve_source` 行 425–477、`resolve_gauge` 行 550–599）：
**可读的生产文件 → 可读的 `outdir` 文件 → 现场生成**。每一步都过
`h5_readable`（打开文件、要求存在 `data` 数据集、能取 `size`，行 310–337），
失败原因写进 `notes`。三个时间戳分别落账：`*_generate_seconds`（本次生成）、
`*_prewarm_seconds`（本次预热）、`*_read_seconds`（求解进程内的读），
`*_read_mode` 取 `cold`/`warm`/`pre-generated`。

## 4. CSV 字段与判读

### 4.1 94 列的分组

`CSV_COLUMNS`（行 95–127）共 94 列，按用途分 9 组：

| 组 | 代表字段 | 用途 |
|---|---|---|
| 身份 | `run_id`、`phase`、`solver`、`dtype`、`atol` | `run_id` 由配置哈希前 16 位（行 868–875），配置相同即同一 id |
| 几何 | `lattice`、`lat_*`、`grid`、`grid_mode`、`ranks`、`threads`、`rank_x_thread`、`shm` | `grid_mode` 由 grid 是否等于 `(1,1,1,ranks)` 判定（行 836） |
| 物理 | `mass`、`csw`、`kappa` | `kappa` 为派生值 |
| 输入 | `source_kind/path/file_dtype/sha256`、`seed`、`gauge_kind/path/file_dtype/sha256` | 只按**文件名**推断 `*_file_dtype`（行 1352–1355）。实测：L24、L32 两份 `real` gauge 文件都是 **dataset `complex128`**（`509,614,080 B` / `1,207,965,696 B`），所以 31 条 PASS 里 **30 条**的 `gauge_file_dtype=c64` 是误报；唯一正确标 `c64` 的是 `4c5cd8a6…` 用的真 `complex64` 随机 gauge。该列不可用于判精度 |
| cache | `cache_kind/path/sha256/identity_sha256/manifest_path/manifest_sha256` | `cache_kind` 为 `global`/`ranked` |
| MG 参数 | `mg_levels`、`mg_restart`、`mg_grid`、`mg_dof`、`nv_iters`、`coarse_max_iter`、`coarse_tol_factor`、`deflate`、`max_iter` | cache 命中时 `mg_grid/nv_iters/dof` 会被 cache identity 覆盖（行 1395–1401） |
| 结果 | `verify_status`、`iterations`、`wall_seconds`、`builder_seconds`、`build_mg_cache_seconds`、`final_true_residual`、`relative_full_residual`、`distributed_relative_residual`、`residual_source`、`converged`、`status` | 判读见 §4.2 |
| 环境/剖析 | `node`、`freq_khz`、`freq_max_khz`、`governor`、`profile_rank`、`dslash_*`、`clover_*`、`reduce_*`、`halo_*`、`phase_*` | 只有开 `KUPL_PROFILE_OPT=1` 的 bistabcg 行才有带宽字段 |
| I/O 与产物 | `input_prebind_seconds`、`gauge_read_seconds`、`source_read_seconds`、`*_generate/prewarm_seconds`、`*_read_mode`、`hdf5_mpio_read`、`residual_seconds`、`report_path/sha256`、`solution_path/sha256`、`out_path`、`err_path`、`returncode`、`elapsed_seconds`、`review`、`timeout`、`error`、`notes` | `elapsed_seconds` 是整行墙钟，含启动与 I/O |

### 4.2 判读规则

1. **`wall_seconds` 只包住一次底层求解调用**（`applyCloverMultigridQcu`；`levels=1`
   时该入口内部就是 parity-Schur BiCGStab），不含 Python 导入、输入读取、`mpirun`
   启动、独立真残差，也**不含** MG cache 构建/载入。整行成本看 `elapsed_seconds`。
2. **`builder_seconds` 与 `build_mg_cache_seconds` 不是同一个量**：前者是本次 run
   载入/构建 cache 的时间（`4.400–6.071 s`），后者来自 `cache_index.json` 的
   `setup_seconds`（L24 `493.638 s`、L32 c64 `100.576 s`、L32 c128 `130.371 s`），
   跨 run 摊销。两者都不在 `wall_seconds` 内，所以 **MG 与 BiCGStab 的 wall 不能横比**。
3. **MG 行的 `iterations` 键不存在**（不是 null）：`kupl_multigrid.py` 在
   coarse-asset 路径上写 `None`，`json_safe(omit_none=True)` 直接删键。带宽字段同理，
   6/6 mg2 行全缺。不要把它当成失败，也不要按迭代数比较 MG 与 BiCGStab。
4. **`atol` 是判据，不是 `residual_acceptance`**：当前驱动固定向底层追加
   `--strict-acceptance`（行 1318），所以台账里的 `atol` 就是验收门槛；本轮 37 条
   solve 的 `atol` 只有 c64 `1e-6`（31 条）与 c128 `1e-9`（6 条）两种。旧参考批次
   （`atol=1e-4`）的实际门槛是 dtype 地板 `2.44140625e-4`，有效容差差 `244.1×`。
   这是 11 it → 18 it 的真正原因，也是"旧 0.17 s 不可直接比"的第一条。
5. **两个残差列互斥**：按 `residual_source` 是否含子串 `distributed` 分流
   （行 1434–1436）。本轮 31 条 PASS 全部落在 `distributed_relative_residual`，
   `relative_full_residual` 恒为 null。
6. **`status` 只看"有没有 report"**：`TIMEOUT` 优先，其次 `no report.json` 记
   `FAIL`（行 1444–1449）；`returncode != 0` 但报告已落盘时会沿用报告里的 `PASS`
   （审计 P3-4 的潜在漏洞，本轮无违规行）。
7. **`node` 必须与数值一起引用**：本轮 4 条 `16×18` 行全部落在 `cn19650`，4 条
   `32×18` bistabcg 行全部落在 `cn19699`；两节点同配置实测差约 `1.45×`
   （`a466a820` `0.247571` 对 `baf92e2b` `0.359793`，各 1 样本）。例外是 3 条
   `16×36`（在 `cn19699`）与 1 条 `32×9`（在 `cn19650`）。因此"16 秩 vs 32 秩"在
   本轮**不是扩展性曲线**。
8. **`gauge_file_dtype`/`source_file_dtype` 只能当文件名标签，不能当 dtype 判据**：
   L24 与 L32 两份 `real` gauge 在磁盘上都是 `complex128`（§4.1），c64 求解靠
   `kupl_bistabcg.load_gauge()` 读真实 dtype 后 `astype(complex64)`。因此矩阵行的
   `dtype=c64` 说明"求解精度 c64"，**不说明**输入文件是 c64；查文件 dtype 必须打开
   dataset，不能看列、也不能看文件名。
9. **本轮输入身份（由 `rows/*.json` 字段复核，引用时必须带 sha 前缀）**：

   | 角色 | 文件 | `sha256` 前缀 | 行数 | 备注 |
   |---|---|---|---:|---|
   | L24 wall 源 | `production_L24/source_24x24x24x64_c64_ones.h5` | `486d358e` | 29 `ones` + 1 `real1` | `real1` 与 `ones` 是同一文件（驱动映射），不是两个源 |
   | L32 wall 源（坏） | `production_L32_v1/source_32x32x32x64_c64_ones.h5.corrupt-20261115` | `a9c2acd8` | 6 FAIL | 16:28 前的坏字节；行里记的就是它 |
   | L32 wall 源（修复） | 同名 `.h5`（= `_ones_v2.h5` 内容） | `9cdcfc82` | 0（待首跑） | 与 `ones_v2` 逐字节相同、201,375,008 B |
   | L24 point 源（坏） | `production_L24/source_24x24x24x64_c64_point.h5.corrupt-20261115` | `945782f3` | 1 FAIL | 16:28 重生成 |
   | L24 point 源（修复） | 同名 `.h5` | `b437e68f` | 0（不跑） | point 轴已放弃 |
   | L24 random 源 | `source_24x24x24x64_c64_random*.h5` | `af13dcc3` / `af8a3188` | 各 1 | 已放弃轴 |
   | L24 随机 gauge | `gauge_random_24x24x24x64_c64_seed42.h5` | `25603783` | 1 | 真 `complex64`，唯一标注正确的 gauge 行 |
   | L24 real gauge | `production_L24/gauge_24x24x24x64_c64.h5` | `4777e795` | 32 | dataset `complex128`，509,614,080 B |
   | L32 real gauge | `production_L32_v1/gauge_32x32x32x64_c64.h5` | `b2a57594` | 4（全 FAIL） | dataset `complex128`，1,207,965,696 B；真正 604 MB c64 副本在 `production_L32_c64/`（`70bbe621`） |

### 4.3 CSV 写入与 schema 升级

`append_csv`（行 1507–1520）在追加前先调用 `upgrade_csv_schema`（行 1471–1504）：
旧表头会被重写为当前 94 列、保留旧有列、留一份 `<name>.schema.bak`，然后
`flock` 追加。这修掉了审计 P2-1 的"93 列旧表头 + 94 列新行会整体错位"问题；
`kupl_kunpeng_l24_c64.csv`、`kupl_kunpeng_prepare.csv`、`kupl_kunpeng_single1.csv`
三个历史文件是 93 列，合并/重跑一次即归一。`merge_rows`（行 1529–1554）以
`rows/*.json` 为准整体重写，行数、列数和 `status=PASS` 会打印出来。

已知遗留：`notes` 列写的是 Python repr（`['a', 'b']`）而不是 JSON，`json.loads`
读不了，下游要用 `ast.literal_eval`（审计 P4-1）。

## 5. 调度与作业模型

### 5.1 `dsub` 的三条硬规则

1. **`#DSUB` 头只有在 `dsub -s <script>` 下才会被解析**。提交脚本固定用
   `bash -lc "dsub -s '$JOB'"`（`submit_kunpeng_all.sh` 行 167–169）；作者注释写明
   "current dsub parses `#DSUB` resource blocks only with `-s/--script`"，直接
   `dsub <script>` 会丢资源块。这与
   `kupl_kunpeng_hpc_config_20261112.md` §4 记录的"-N/-rpn/-R/-nl 可能被忽略"
   现象一致；该历史现象是否**全部**由这一条造成，本文不逐一回溯（其余提交路径未审计）。
2. **`-T` 是墙钟秒数**，本轮 40 个作业脚本里 22 个 `-T 1200`、18 个 `-T 900`。
   它不是单条 solve 的超时；单条由驱动的 `--timeout/--hard-timeout` 管。
   `RC=124` 是作业级超时，不等于卡死（`kupl_kunpeng_hpc_config_20261112.md` §5）。
3. **一个 dsub 作业 = 一个 task = 整个 Python 驱动**，`mpirun` 由 Python 在内部
   拉起（驱动 docstring 行 46–48、提交脚本头注释）。外层再包一层 `mpirun` 会形成
   递归 MPI 并直接失败。

### 5.2 两种 task model

| 模型 | `#DSUB` 资源块 | 适用 | 证据 |
|---|---|---|---|
| `mpi`（默认） | `--mpi hmpi`、`-N <ranks> -rpn <ranks> -R "cpu=<threads>;mem=8G"` | 每 rank 一个 task，cpuset 由调度器按 rank 切 | 历史可复现配方 |
| `single` | `-N 1 -rpn 1 -R "cpu=<ranks*threads>;mem=400G"` | 整节点给一个 task，rank 内绑定自己做 | 本轮 `kall_l24_c64_full.sh` 用 `cpu=576`、`kall_16r_cn19650b.sh` 用 `cpu=288`、`kall_l24_c64_t9_16r_shm1.sh` 用 `cpu=144` |

`single` 模型下 `mpirun` 必须在作业内使用 `--bind-to none --map-by ppr:2:numa`
（驱动行 1302–1303 硬编码），CPU 绑定交给 `KUPL_NUMA_BIND`。

### 5.3 独占与共享

- 默认 `-x job`（整节点独占）。**节点被别人占着一部分时，`-x job` 的作业只能
  PENDING 等待**，不会降级——本轮 40 个作业脚本里 21 个写了 `-x job`，19 个没写
  （由 `--shared` 生成或手写，允许半节点）。
- 判据：需要稳定独占 cpuset 做 A/B 时用 `-x job`；只是补一条单样本、或节点紧张时
  用 `--shared`，并在台账 `notes` 里记下共享状态，否则 `1.45×` 的节点/邻居噪声会被
  误读成代码差异。
- `-nl <node>` 只是偏好；目标节点无资源时作业会挂起并报
  `No node is available to meet node preferences`（`kupl_kunpeng_hpc_config_20261112.md` §4）。

### 5.4 资源与超时的经验值

| 参数 | 取值 | 依据 |
|---|---|---|
| `ranks*threads` 上限 | 608 | 驱动常量 `MAX_TOTAL_THREADS`（行 93）；节点 592 CPU，留在同一节点上的超订阅余量；32×36 被脚本与驱动双重拒绝 |
| 单作业 task 上限（账号 cap） | 64 task 会被拒 | 历史 `mainStateReasonMsg: The limit <testusers_rebuild> has been exceeded`（`kupl_kunpeng_hpc_config_20261112.md` §4）。本轮最大 32 rank，未触发；要上 64 rank 必须先确认该 cap，不能假设"节点有 592 CPU 就能排 64 task" |
| 每 rank cpuset | 18 CPU（`ppr:2:numa`、32 rank） | `TH=36` 会触发库内守卫 `KUPL_OMP_THREADS exceeds the NUMA-local CPU set` |
| 作业墙钟 | 900 s 或 1200 s（本轮 18/22 个脚本） | 本轮 40 个脚本；cache 批次与 16 秩批次多用 1200 s |
| 单条硬超时 | 600 s 默认 | `--hard-timeout` |
| L24 全局 cache 构建 | `493.638 s`（1 rank×18 thread） | 占掉大半个 900 s 作业；L32 构建则触发 600 s 硬超时（`rc=124`，见 §6.6） |

## 6. 结果快照（42 行，2026-11-15 16:06 本地副本）

### 6.1 计数

| 项 | 数量 | 说明 |
|---|---:|---|
| `rows/*.json` | 42 | 37 solve + 5 prepare |
| solve 状态 | 31 PASS / 6 FAIL | 无 TIMEOUT、无 SKIP |
| prepare 状态 | 2 PASS / 2 REUSE / 1 SKIP | SKIP 是 ranked cache（h5py 无 MPI） |
| `reports/*.json` | 31 | PASS 行各一份；FAIL 行无报告 |
| `logs/` | 154 个文件 | 每行 `.out` + `.err`，另有作业级日志 |
| `kupl_kunpeng_all.csv` | 94 列 × 42 行 | 合并后的唯一宽表 |

### 6.2 L24 c64 的 2×2 受控格（cn19699，32×18，atol 1e-6，ones 源，seed 42）

| grid | shm | run_id | wall (s) | 迭代 | 真残差 | rank-0 dslash | 聚合 |
|---|---:|---|---:|---:|---|---:|---:|
| t-finest `1 1 1 32` | 1 | `232d6b859b93d000` | **0.240473** | 18 | `7.1247e-07` | 9.18 GB/s | 293.76 GB/s |
| t-finest `1 1 1 32` | 0 | `a466a8207fe379e1` | 0.247571 | 18 | `7.1247e-07` | 8.00 GB/s | 256.02 GB/s |
| balanced `2 2 2 4` | 1 | `72e04aa67eb8241c` | 0.313275 | 18 | `7.1216e-07` | 6.24 GB/s | 199.64 GB/s |
| balanced `2 2 2 4` | 0 | `5c6cdbe48901c409` | **0.349715** | 18 | `7.1216e-07` | 4.00 GB/s | 128.10 GB/s |

读法：

- grid 效应：同 shm=0 时 balanced/t-finest = `1.4126×`；同 shm=1 时 `1.3027×`。
  两条都是同节点、同 18 it、同 gauge/source/seed 的单样本比较，方向可信，绝对值
  仍受单样本限制。
- shm 效应：balanced 上 `−10.42%`，t-finest 上 `−2.87%`。与 11-13 的 14.77%
  （8+8 样本、点源、atol 1e-4）不是同一批样本。
- 带宽不是"通信量"差异：balanced 的 dslash 字节更少（`857.309 MB/76` 对
  `1210.319 MB/76`），但每 rank 只有 `4.00 GB/s`；t-finest 保留了时间方向的
  高局部性，`9.18 GB/s`。
- 固定开销：`phases prepare+solve+reconstruct` 为 `0.2045 s`（t-finest shm1）与
  `0.3066 s`（balanced shm0），分别占 `wall` 的 15% 与 12%；把 `wall/iterations`
  当"每迭代内核时间"会同时高估两边、低估差异（审计 §3.4a）。

### 6.3 其它 c64 单元（全部 n=1）

| 用例 | 单元 | run_id | wall (s) | 迭代 | 真残差 |
|---|---|---|---:|---:|---|
| L24 16×18 t-finest shm1 | cn19650 | `b2493fd05d6fc276` | 0.326054 | 18 | `7.1218e-07` |
| L24 16×18 t-finest shm0 | cn19650 | `dfdc9bf54c935e5d` | 0.350344 | 18 | `7.1218e-07` |
| L24 16×18 balanced shm1 | cn19650 | `10b4206e57ac132c` | 0.451844 | 18 | `7.1223e-07` |
| L24 16×18 balanced shm0 | cn19650 | `8a6e3367f25f9f59` | 0.486863 | 18 | `7.1223e-07` |
| L24 16×36 t-finest shm1 | cn19699 | `b63ba9c835d4f81c` | 0.212354 | 18 | `7.1221e-07` |
| L24 16×36 t-finest shm0 | cn19699 | `095ce6dc4a35ece1` | 0.220305 | 18 | `7.1221e-07` |
| L24 32×9 t-finest shm0 | cn19650 | `baf92e2b99937b2b` | 0.359793 | 18 | `7.1256e-07` |
| L24 32×9 t-finest shm1 | cn19650 | `861ac912ac3c7c42` | **11.437897** | 18 | `7.1256e-07` |
| L24 16×9 t-finest shm1 | cn19650 | `b614ad9349f5d4e6` | 0.442830 | 18 | `7.1211e-07` |
| L24 16×18 t-finest shm1、random 源 seed 24 | cn19650 | `9cc71c5d52dbbf22` | 0.340956 | 19 | `4.3384e-07` |

`16×36` 的两条在 cn19699，`16×18` 的四条在 cn19650，所以**不能把
`0.2124` 与 `0.3261` 读成"线程 36 比 18 快"**；这是节点与线程同时变。

### 6.4 c128

| 用例 | 单元 | run_id | wall (s) | 迭代 | 真残差 |
|---|---|---|---:|---:|---|
| L24 32×18 t-finest shm1 | cn19699 | `d2ce9be32b406b0f` | **0.475555** | 31 | `5.3257e-11` |
| L24 32×18 t-finest shm0 | cn19699 | `f19ba1fef0ccff32` | 0.567444 | 31 | `5.3257e-11` |
| L24 16×18 t-finest shm1 | cn19650 | `f212adf893cb23e0` | 0.822888 | 31 | `5.3257e-11` |
| L24 16×18 t-finest shm1 MG2 r5 | cn19650 | `749b3b028b3248c9` | 1.445706 | 无 | `6.8738e-10` |

c128 与 c64 的 `0.475555/0.240473 = 1.978×` 中，迭代数比已是 `1.72×`；
剩余约 15% 才是每迭代成本差。不要把它写成"c128 单次算子慢 2×"。
c128 的残差 `5.3e-11` 远低于 `1e-9` 判据，说明停机由 `atol` 触发而非地板。

### 6.5 MG2（L24 c64，全部走 global cache）

| grid | 单元 | restart | shm | run_id | wall (s) | 真残差 | verify |
|---|---|---:|---:|---|---:|---|---:|
| t-finest | 32×18 | 20 | 0 | `4d187eadc00a7feb` | **0.252589** | `6.6326e-07` | 0 |
| t-finest | 32×18 | 5 | 1 | `c98060824b55ebf7` | 0.394854 | `7.9362e-07` | 0 |
| balanced | 32×18 | 20 | 0 | `cb136a9593aa2021` | 0.300639 | `6.6286e-07` | 0 |
| balanced | 32×18 | 5 | 0 | `b2715635518ec65b` | 0.725579 | `7.9232e-07` | 0 |
| balanced | 16×18 | 5 | 0 | `716b1f6a3fd49353` | 0.888075 | `7.9246e-07` | 0 |
| t-finest | 16×18 | 5 | 0 | `7668f066600ef451` | 0.625900 | `7.9219e-07` | 0 |
| t-finest | 16×18 | 5 | 1 | `cf5659638b00b61e` | 0.566035 | `7.9219e-07` | 0 |
| t-finest | 16×18 | 20 | 0 | `06ad0f1f0accf025` | 0.342719 | `6.6290e-07` | 0 |
| t-finest | 16×18 | 20 | 1 | `9c122f681fa3ffe1` | 0.312759 | `6.6290e-07` | 0 |
| t-finest | 16×36 | 5 | 0 | `220f00939aeda0ab` | 0.452098 | `7.9225e-07` | 0 |

三条红线：

- 表里不含 `b3267714a641ea60`（balanced 32×18 shm=1 r5，FAIL，shared halo handshake
  超时，见 §7 第 3 条——该条已被 2/2 复测推翻，属**不稳定失败**而非 known skip），
  也不含任何 `restart=10/50` 的行（本轮未跑）。
- `wall` 不含 `builder_seconds`（4.400–6.071 s）与一次性
  `build_mg_cache_seconds=493.638 s`。按端到端算，`4d187ead` 是
  `0.2526+4.40 = 4.65 s`，比同格 BiCGStab 的 `0.2405 s` **慢约 19×**，
  不是"MG 快 1.38×"。
- `restart` 不同的行不可互比（5 与 20 是不同算法参数）；`iterations` 不存在，
  MG 与 BiCGStab 不能按迭代数比较。

### 6.6 L32（0 PASS）

| 单元 | dtype | shm | run_id | 结果 | 证据 |
|---|---|---|---:|---|---|
| t-finest 32×18 | c64 | 0 | `327f9aa531167d55` | FAIL rc=1 | h5py `bad object header version number` (43,465 B .err) |
| balanced 32×18 | c64 | 0 | `c9bfd6bfb733b052` | FAIL rc=1 | 同上 |
| t-finest 32×18 | c128 | 0 | `23daf98e9c477bdd` | FAIL rc=1 | 同上 |
| t-finest 32×18 | c128 | 1 | `35548a2b06ff0989` | FAIL rc=1 | 同上 |
| prepare global cache | c64 | — | `cache-2139960e885a0627` | PASS/verify 0 | `global build rc=124 elapsed=602.2s`、`build_mg_cache_seconds=100.576` |
| prepare global cache | c128 | — | `cache-3b6e1e37b3af0b9c` | PASS/verify 0 | `global build rc=124 elapsed=602.1s`、`build_mg_cache_seconds=130.371` |
| **快照后旁证** t-finest 32×18 | c128 | 0 | `82ba09ec3a8e0d2d` | **PASS** rc=0 | 16:32、`mg2` restart5；source 换成可读的 `source_32x32x32x64_c64_point.h5`（`dc427e47`），gauge 同一份 `b2a57594`，L32 c128 cache `7c6abf0f`，真残差 `4.03e-10` |

快照内 4 条 FAIL 的 `source_path` 都是
`production_L32_v1/source_32x32x32x64_c64_ones.h5`（`source_sha256=a9c2acd8…`）；快照后
新增的 `07c80fc8`、`44faded2` 两条同样是坏 ones（本地无 row，只有远端证据）。这 4 行的
`gauge_path` 是 `production_L32_v1/gauge_32x32x32x64_c64.h5`：文件名写 `c64`，实测
**1,207,965,696 B、dataset `complex128`**（`sha256=b2a57594…`），该文件在 prepare 阶段被成功
用来建 cache；真正的 604 MB c64 副本在 `production_L32_c64/gauge_32x32x32x64_c64.h5`
（file `70bbe621…`、canonical `9ac7c15c…`）。c64 cache 的 identity 因此指错文件，见
[`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3.4。
根因、尺寸核算与修复见
[`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md)。

> L32 的 FAIL 是**输入坏**，不是 L32 几何不可算。两条 L32 cache 行
> `verify_status=0` 已经证明 32 rank 的 patch 构建路径能跑通；把 L32 写成
> "算法不支持"是错误归因。
>
> 快照后的 `82ba09ec3a8e0d2d` 是同一结论的**正面**证据：source 换成可读文件后，
> 32 rank L32 c128 `mg2` 直接 PASS（残差 `4.03e-10`）。但它只覆盖 1 个单元、1 个样本，
> 且 source 是 point 不是 ones，**不能**用来宣布"L32 已补齐"：6 条 FAIL 配置
> （c64/c128 × t-finest/balanced × shm0/1）仍欠首次有效运行。

### 6.7 离群与不可用行

| run_id | 单元 | wall (s) | 迭代 | 判定 |
|---|---|---:|---:|---|
| `0041182f2d8e1c50` | L24 16×18 t-finest shm1、`real1`、seed 24 | 20.859674 | 18 | 离群（同配置 `ones` 为 `0.326054`），不可用 |
| `e1574f172b525c02` | L24 16×18 t-finest shm1、random gauge、seed 42 | 10.927863 | 18 | 离群，不可用 |
| `4c5cd8a65e492c1e` | L24 16×18 t-finest shm0、random gauge、seed 42 | 6.358685 | 7 | 离群；7 it 却慢 20×，与迭代数无关 |
| `861ac912ac3c7c42` | L24 32×9 t-finest shm1 | 11.437897 | 18 | 同配置 shm0 为 `0.359793`，31.8× 偏差；原因待确认 |
| `c3bd73d2999000f0` | L24 16×18 t-finest shm1、point 源 | — | — | FAIL：point 源坏 |

这 5 行都保留在 `rows/` 里、没有删除；引用任何中位数前必须先剔除。

## 7. 失败与边界

| # | 现象 | 已确认的原因 | 状态 |
|---|---|---|---|
| 1 | L32 6 条 solve 全 FAIL（rc=1，无 report；快照内 4 + 快照后 2） | 生产 source 文件 object header 未完成，h5py 打不开 | **已定位并已修复**（2026-11-15 16:28 换成 `ones`/`ones_v2`，`9cdcfc82…`）；6 条待**首次有效运行**，见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3.3 |
| 2 | L24 point 源 FAIL（rc=1） | 同类坏文件：`source_24x24x24x64_c64_point.h5` | **已定位并已修复**（16:28 重生成，`b437e68f…`）；point 性能按范围裁定**不跑** |
| 3 | `b3267714a641ea60`：L24 c64 MG2 balanced 32×18 **shm=1** r5，rc=2，无 report，`elapsed=271.437 s` | `.err` 反复出现 `KUPL shared halo handshake timeout: publish`，随后 `MPI_ABORT ... errorcode 2` | **不是稳定失败，`known skip` 判定已被推翻**：同 config 在两个独立 outdir 复测 **2/2 PASS**（cn12104，`verify=0`、真残差 `7.923186435088777e-07`，与 shm0 控制行逐位相同）→ 1 FAIL / 2 PASS。且 FAIL 行由 **14:20 版**驱动（64,642 B）产生、复测用 **16:30:50 版**（85,625 B），**版本不同**也是变量。`data/kunpeng_all_20261115/known_skips.json` 仍留旧条目（数据侧未合并），候选条目在 `known_skips.candidate.json`（默认 disabled）。权威分析与行表见 [`kupl_l24_c64_mg2_shm1_boundary_20261115.md`](kupl_l24_c64_mg2_shm1_boundary_20261115.md) |
| 4 | L32 c64 曾指向 c128 gauge | 驱动只按文件名推断 `gauge_file_dtype`；同目录 c64 副本（604 MB）与 c128 副本（1.207 GB）并存 | **已裁定**：本轮行 `gauge_path` 全是 `_c64.h5`，但该文件 dataset 是 `complex128`；c64 cache 的 `identity.gauge_file_sha256` 因而指错文件，需按 `production_L32_c64/gauge_32x32x32x64_c64.h5`（`70bbe621…`）重签，见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3.4 |
| 5 | `32×9 + shm1` 的 `11.4379 s` | 同节点同配置 shm0 为 `0.3598 s`，偏差 31.8× | **不再复核**：thread 9 整轴按范围裁定放弃（耗时代价过大），该行保留但引用时必须剔除 |
| 6 | ranked cache 一条 `SKIP` | 该 Python 环境的 h5py 无 MPI（`mpio` 不可用），ranked builder 跑不了 | **blocked，不列入后续计划**：MPI h5py 只做接口检查，不重新编译/安装；12 条 mg2 行最终都用 global cache，`balanced` 的行另带 `ranked cache unavailable; using global cache` note，见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §1 |
| 7 | 3 个 MG 空洞（32 秩 t-finest shm1、16 秩 shm1 两种 grid） | `--skip-failed` 让已 FAIL 的单元永不重跑 | **流程缺陷**（审计 P3-3） |
| 8 | L32 cache `rc=124` 但 `PASS` | 构建触发驱动的 600 s 硬超时；超时前 cache 已落盘，后续作业按 manifest 校验通过 | 可用但要注明"非干净构建" |

## 8. 数据路径与复现

### 8.1 目录职责（本地副本，全部被 `data/.gitignore` 忽略）

| 路径 | 内容 |
|---|---|
| `data/kunpeng_all_20261115/rows/` | 42 个 `run_id`.json，一行一文件；权威台账 |
| `data/kunpeng_all_20261115/reports/` | 31 个底层 `report.json`（真残差、`residual_source`、`hdf5_mpio_read`） |
| `data/kunpeng_all_20261115/caches/` | 3 个 cache **manifest**（`.json`）；`.h5` 本体在远端 |
| `data/kunpeng_all_20261115/logs/` | 154 个 `.out`/`.err`；带宽与 handshake 签名只能从这里取 |
| `data/kunpeng_all_20261115/jobs/` | 40 个作业脚本，`#DSUB` 资源块与 `--extra` 的原始记录 |
| `data/kunpeng_all_20261115/kupl_kunpeng_all.csv` | 94 列 × 42 行合并宽表 |
| `data/kunpeng_all_20261115/cache_index.json` | 4 条 global cache 的身份、`setup_seconds`、`verify_status`、notes（L24 c128 构建 `48.191 s` 也在这里） |
| `data/kunpeng_all_20261115/file_hashes.json` | `path:size:mtime` → sha256 缓存 |
| `data/kunpeng_all_20261115/FAILURES.md` | 6 条失败/不可用行的隔离摘要（含 CSV 行号、`.err` 大小与 mtime、替代命令）；`quarantine.json` 的人类可读版 |
| `data/kunpeng_all_20261115/quarantine.json` | 机器可读隔离与裁定：`quarantined`（6 行）、`input_remediation`（`.corrupt-*` + 替换文件）、`scope_decisions`、`post_snapshot_remote_evidence`（`82ba09ec`/`8a9404d3`） |
| `data/kunpeng_all_20261115/known_skips.json` | 放弃轴、保留项、2 条 known skip、待首跑 6 条、不重测判据、cache 重签、队列、MPI h5py。**注意**：其中 `b3267714` 的 known skip 已被 2/2 复测推翻，候选修订在 `known_skips.candidate.json`（默认 disabled），数据侧尚未合并 |
| `data/kunpeng_all_20261115/crossnode_cn12237/` | 跨节点单行（cn12237 L24 c128 balanced 32×18 shm0 `0.510834 s`，PASS）+ 独立 CSV/报告 |
| `data/kunpeng_all_20261115/validate/` | 12:57 的冒烟失败记录（见 §3.1） |
| `data/kunpeng_all_20261115_audit/` | 只读审计：`ledger.md`（4 张分表）、`ledger.csv`、`report.md`（口径、P 级缺陷清单与 2.04× 分解） |
| `data/kupl_loop_proto_20261115/` | loop 原型：`loop_proto.py`（一次 `mpirun` 内跑多 case、可注入故障）、`make_plan.py`、`run_node.sh`、`run_login06.sh`、`make_job.sh`、`kupl_env.sh`、`login06/`、`node/`｜实测分段见 §13 |

本地体量（2026-11-15 快照）：`data/kunpeng_all_20261115` 约 1.8 MB、
`data/kupl_loop_proto_20261115` 约 340 KB、`data/kunpeng_all_20261115_audit` 约 144 KB；
三者内 `*.h5` 计数均为 **0**——solution/cache/gauge/source 全部留在远端。

本地摘要与时点记录：`logs/v20261115.txt`（输入记录）、`logs/v20261115-2.txt`（本批运行记录）、
`logs/v20261115-3.txt`（范围裁定）、`logs/v20261115-4.txt`（补全/纠错）逐份保留，
**不删、不改名、不合并**；`data/**` 因 `data/.gitignore` 为 `*` 全部不跟踪，
所以"本地副本"只能靠 `file_hashes.json` 的 sha256 与 rows/reports 的字段复核，
不能当远端字节级一致性的证明。

远端权威根为 `/home/share/nsls_yangyb/PyQCU-main/data/kunpeng_all_20261115/`。
审计快照（2026-11-15T06:40:26Z，当时只有 16 行）的 `rows`/`reports`/`*.csv` 曾与远端
逐 md5 比对一致；此后远端继续追加到 42 行，本地副本随之后同步，**没有再逐文件复核
md5**（只有 `file_hashes.json` 的 sha256 缓存）。
本机**没有** `data/production_L24`、`data/production_L32_v1`，也没有 cache `.h5`，
所以本地只能复核 sha256 与字段，不能复核文件大小与内容。

### 8.2 最小复现链

```bash
# 1) 在登录节点确认矩阵展开与计数
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode list --expand | wc -l   # 7680

# 2) 只跑一个 2×2 格（32 秩、c64、t-finest/balanced × shm 0/1）
for grid in "1 1 1 32" "2 2 2 4"; do
  for shm in 0 1; do
    python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode solve \
      --lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --dtype c64 --atol 1e-6 \
      --solver bistabcg --grid $grid --ranks 32 --threads 18 --shm $shm \
      --source ones --gauge real --seed 42 --run-label g${grid// /}-s${shm}
  done
done

# 3) 合并与核对
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode merge
```

作业化时只把上面第 2 步交给 `submit_kunpeng_all.sh`：

```bash
./pyqcu/testing/kupl/submit_kunpeng_all.sh --mode solve --tag kall_2x2 \
  --ranks 32 --threads 18 --walltime 1200 \
  --extra "--lattices L24 --dtypes c64 --solvers bistabcg --grid-modes balanced,t-finest --shm-list 0,1 --sources ones --gauge-kinds real --seeds 42"
```

## 9. 运行手册（合并自可选 runbook）

原计划单列 `kupl_kunpeng_runbook_20261115.md`；其内容与本节、§5 高度重复，按任务
约定**合并到本文**，不再新增第三份文档。

### 9.1 环境

```bash
cd /path/to/PyQCU
source ./env.sh                     # 导出当前树优先的 PYTHONPATH / KUPL 库路径
source /work_ssd/software/HPCKit/25.2.1/setvars.sh
export PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH
export OMPI_MCA_rmaps_base_oversubscribe=1
```

- 计算节点默认环境**没有** `mpirun`；不 source `setvars.sh` 会得到 `RC=127`。
- `dsub --mpi hmpi` 只是调度器声明，**不会**把 MPI 加进 `PATH`。
- conda 必须显式进 `PATH`：作业默认 `python3` 没有 numpy/h5py。
- 2026-11-14 曾出现 HMPI `mpirun` 对所有选项报 `unknown option`（缺 `OPAL_PREFIX`），
  属环境问题，不是 KUPL 代码问题；遇到时先设 `OPAL_PREFIX` 再重试。
- 计算节点 conda `2023.3` 的 HDF5 栈实测：h5py `3.7.0` / HDF5 `1.10.6`、
  `get_config().mpi=False`、无 `mpio` driver。**逐批开跑前都要在这台机器上重测这三个值**
  再决定 I/O 预算与 ranked cache 是否可排（口径见
  [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §1.3）：

  ```bash
  python3 -c "import h5py;print(h5py.__version__, h5py.version.hdf5_version, h5py.get_config().mpi)"
  ```

### 9.2 构建

```bash
bash cpp/kupl/qcu/make.sh --check                     # 只显示 build dir/native/LTO/tests/jobs
KUPL_ENABLE_NATIVE=ON KUPL_ENABLE_LTO=ON bash ./install.sh kupl   # 目标节点
KUPL_ENABLE_NATIVE=OFF bash ./install.sh kupl                    # 跨型号部署
```

矩阵作业本身不改库；换构建必须在台账外单独记 `libqcu.so` 身份，否则不同批次不可比。

### 9.3 提交

```bash
./pyqcu/testing/kupl/submit_kunpeng_all.sh --mode solve --tag <tag> \
  --ranks 32 --threads 18 --walltime 1200 --node cn19699 \
  --extra "--lattices L24 --dtypes c64 --solvers bistabcg"
# 提交后核对实际节点
djob -ll <job_id>
```

`--shared` 去掉 `-x job`；`--task-model single` 改成整节点单 task。

### 9.4 常见错误对照表

| 现象 | 根因 | 处理 |
|---|---|---|
| `RC=127`，12 s 结束 | 没 source HPCKit，找不到 `mpirun` | §9.1 |
| `mpirun: Error: unknown option "-n"` | `OPAL_PREFIX` 缺失 / 环境的 HMPI 包装层 | 显式设 `OPAL_PREFIX`，或换 HPCKit 版本 |
| 作业一直 PENDING | `-x job` 要求整节点，节点被部分占用 | `--shared` 重提，或换节点 |
| `KUPL_OMP_THREADS exceeds the NUMA-local CPU set` | `TH > 18`（32 秩、`ppr:2:numa`） | 降到 18；36 线程只在 16 秩且 576≤608 时合法 |
| `No node is available to meet node preferences` | `-nl` 偏好无法满足 | 换节点或去掉 `-nl` |
| solve 600 s 无 `report.json`，`RC=124` | 作业墙钟截断 | 提高 `-T`/拆小批次；不要把 124 写成死锁 |
| `OSError: Unable to open file (bad object header version number)` | 生产 HDF5 输入损坏 | 见根因文档；坏字节已改名 `*.corrupt-20261115`，L32 ones / L24 point 于 16:28 换成可读文件 |
| `KUPL shared halo handshake timeout: publish` + `MPI_ABORT` | MG2 + `shm=1` 的握手抖动（**不是稳定失败**） | 换新 `outdir` 复测：同 config 已 2/2 PASS；不要整体禁 `balanced`/`shm1`，也不要静默回退 `shm=0`（见 [`kupl_l24_c64_mg2_shm1_boundary_20261115.md`](kupl_l24_c64_mg2_shm1_boundary_20261115.md)） |
| `ranked cache unavailable; using global cache` | h5py 无 MPI，ranked builder 不可用 | 接受 global 回退；**不把编译/安装 MPI h5py 列为下一步**（见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §1） |
| 提交被拒：`The limit <testusers_rebuild> has been exceeded` | 账号 task 上限（64 task 量级），不是资源不足 | 降 `-N`/`-rpn`，或先确认该 cap；不要反复重提 |
| `djob -c` / `dsub -c` 报 `invalid option` | 当前 LX2 客户端没有这两个取消子命令 | 用调度器 UI 取消；不要把"CLI 取消成功"写进记录（[`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §4.2） |
| 需要终止已排队/在跑作业 | `dkill <job_id>` **可用**（会话记录：job `2133396` 于 17:01:36 被 `dkill` 终止并记 FAILED） | 该条是 `[会话]` 证据、工作区无工件可复核，标**待确认**；不要与 `djob -c`/`dsub -c` 混为一谈 |
| 16 秩与 32 秩差 1.4× | 两组落在不同节点 | 固定同一 `node` 再比 |
| 同配置单次 11 s / 6 s / 20 s | 单样本 + 页缓存/邻居噪声 | 每单元 ≥3 样本，记 min/median |

## 10. 证据、缺口与下一步

### 10.1 证据等级

| 内容 | 等级 | 位置 |
|---|---|---|
| 42 行台账、cache index、作业脚本 | B | `data/kunpeng_all_20261115/**`（本地副本；16 行快照曾与远端 md5 一致，后续追加未复核） |
| 驱动行为、字段、模式、参数 | B | `pyqcu/testing/kupl/test_kunpeng_all.py`（本地未跟踪，行号以本机版本为准） |
| 调度规则、`#DSUB` 资源块 | B | 40 个 `jobs/*.sh` + `submit_kunpeng_all.sh` |
| 与远端文件的字节级一致性 | C | 本机未挂载远程根，只有 sha256 |
| 2.04× 分解 | B | `data/kunpeng_all_20261115_audit/report.md` §3 |

> 本地脚本 md5 为 `7ee523b5998152f23987a55d33e8bab5`，审计快照记录的远端脚本 md5 为
> `3127528d84d88b6b7af1de5ff9ef4ff7`。两者不同，说明快照之后脚本又被改过；
> 复现时以实际文件为准，不要按本文行号盲改。

### 10.2 必须补齐才能给出结论的项

1. 每个单元 ≥3 样本（当前 37 个单元各 1 样本），并记录 min/median。
2. 16 秩与 32 秩固定在同一节点；两节点因子约 `1.45×`，比任何待测差异都大。
3. L32 用修复后的 ones（`ones`/`ones_v2`，`9cdcfc82…`）做**一次首次有效运行**，覆盖 6 条
   FAIL 的配置；这不是"重测"。L32 至今仍无有效 PASS。
4. MG 的 `builder_seconds` 单独列出，并把 `total = wall + builder` 落成字段。
5. `iterations` 的"设计性缺失"与"未记录"分成两个枚举值，避免下游误判。
6. `residual_acceptance`/`strict_acceptance` 提升为一等字段（现在只在 report 里）。
7. 把 `n_repeats`、`stat`（min/median）与 `node` 写进 schema，让"单样本"在表里可见。
8. 补 `24³×64` 的 c128 MG 真实回读（当前只有 16 秩一条 `1.4457 s`）。

> 本节只覆盖**保留轴**（L24/L32 主格、c128、MG1/r10、MG+shm、16×36）。thread 9、seed24、
> random gauge、random 源、point 源、mg-restart 50 已按范围裁定放弃，不在补齐清单里；
> 裁定见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §2。

### 10.3 不建议做的事

- 不要用 37 个单样本拼"性能矩阵"或热力图；当前只够做口径校验与失败边界。
- 不要把 `0.2405` 与 `0.154883`（点源、atol 1e-4、12 it）或 `0.171331`（全 1 源、
  11 it）直接相除；源、容差、迭代、grid、shm 至少差三项。
- 不要把 MG 的 `0.2526` 与 BiCGStab 的 `0.2405` 当成 "MG 快/慢"；前者不含
  4.4 s cache 载入。
- 不要把 `16×36` 的 `0.2124` 当"36 线程更快"；它在 cn19699，16×18 在 cn19650。
- 不要把 L32 的 FAIL 写成 L32 不可算：失败在输入文件，不在几何或算法。
- 不要把已放弃轴（thread 9 / seed24 / random gauge / random 源 / point 源 / r50）的单样本
  数字再拼进任何对照表；它们的处置是"耗时代价过大，部分数据保留，不重测/不新增"。
- 不要把"编译/安装 MPI h5py"或"启用 ranked cache"写成下一步；`bg-3` 的离线构建与
  `bg-23` 的 ranked cache 尝试都不作为本轮交付，其产物未经验证、不得引用。

## 11. 本轮范围裁定（追加，不改上面的快照数字）

2026-11-15 收口时对"做什么/不做什么"下的裁定，权威版本在
[`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md)；这里只列与本文表格直接
相关的四条：

1. **MPI h5py 只做接口检查**：h5py `3.7.0` / HDF5 `1.10.6`、`get_config().mpi=False`、无
   `mpio` driver、`h5py_has_mpi()` 为 `False`；包装器的自动 `KUPL_HDF5_MPIO_READ=1` 分支
   不触发，没有任何一行 `hdf5_mpio_read=true`（28 行显式 `false`、14 行旧版本缺键），走
   per-rank 全量读回退。**不重新编译/安装**，
   ranked cache 不列入后续计划。
2. **6 个轴放弃**：thread 9、seed24、random gauge、random 源、point 源、mg-restart 50。
   保留 16×36（3 行）、MG1/r10、L32/c128、MG+shm。
3. **L32 6 条 FAIL 是首次有效运行缺口**：都用了坏 ones `a9c2acd8…`；修复后的
   `ones`/`ones_v2` 同为 `9cdcfc82…f55a`、全 `1+0i`、逐字节相同，无需重新生成。
   `source_32x32x32x64_c64.h5` 是 **random seed42**，不能当 ones 用。cache 侧只需重签
   L32 c64（改用 `production_L32_c64/gauge_32x32x32x64_c64.h5`，`70bbe621…`）与 L24 c64
   （`production_L24/gauge_global_c64.h5`，`8a482d6a…`）；L32 c128 cache 不重建；MG cache
   不含 source 哈希，不受 source 修复影响。31 条 PASS 不重跑。
4. **队列**：`bg-26/27/28/32` 已 kill；它们对应的 pending 作业（thread9、seed24/random、
   point、r50）`djob -c` 与 `dsub -c` 均报 `invalid option`，未能在 CLI 取消，队列中仍可能
   `PENDING`（可由用户经调度器 UI 取消）。保留 `kall_l32_*`、`kall_l24_c1*`、`kall_mg_shm*`、
   `kall_mg1_r5*`、`kall_mgr10*`、`k9_16x36*`，不 kill、不删数据。

## 12. 输入不可变、超时语义与写原子性（追加，2026-11-15 只读审计）

> 证据等级：**B**（源码静态阅读 + 只读 `ls/find/grep`，未执行任何写路径、未提交作业）。
> 该审计的逐文件行号明细（写路径 W1–W17、驱动 D1–D7、构建器 B1–B7、
> `_common` C1–C5、prepare P1–P5、违规 V1–V15、修复建议 F1–F9、规则 R1–R10）
> 以**审计报告**为准；报告写在会话工作目录、**未入仓库**，不能当稳定链接引用。
> 本文只落已复核的结论、规则和可重跑的复核命令，行号以审计时的脚本版本为准
> （本地 `test_kunpeng_all.py` 未被 Git 跟踪，md5 会随版本变，见 §10.1）。

### 12.1 原输入在源码层是只读的

| 输入 | 读取点 | 打开模式 |
|---|---|---|
| LIME 参考组态 | `_common._default_lime()`、`build_full_mg_cache.py`、`prepare_case.py` → `_lime.read_lime_gauge` | `open("rb")` |
| production gauge/source | `resolve_gauge/resolve_source` → `--gauge/--source` → `kupl_multigrid.py`、`kupl_bistabcg.load_gauge/load_source`、`_read_local_slab` | `h5py.File(..., "r")` / `read_h5` = `"r"` |
| slab 目录（`--gauge-slab-dir/--source-slab-dir`） | `kupl_multigrid._read_local_slab` | `"r"` |
| MG cache（`--coarse-cache`） | `kupl_multigrid` → `validate_legacy_mg_cache(..., verify_file_sha256=True, verify_dataset_sha256=True)` | `"r"` + identity **fail-closed**（不符直接 `ValueError`） |

结论：**超时/被杀不会改写 gauge、source、既有 cache**。驱动唯一写死的非 outdir 输出是
`kupl_multigrid.py` 里 `--report` 未显式给出时的 `DATA_DIR` 缺省目录（属仓库 `data/`，
不是输入文件）。

### 12.2 真正会被超时破坏的是"上一次的产物"

- solution / report 在**最终路径**上直接 `h5py.File(...,"w")` / `Path.write_text`，
  同 `run_id` 重跑即原地截断，超时会留下半写文件被下游当成产物；`logs/<run_id>.{out,err}`
  同样是 `open("wb")` 截断。
- 本轮驱动自己的小文件（`rows/<id>.json`、`cache_index.json`、`file_hashes.json`、
  merge CSV）走 `temp + os.replace`；`append_csv` 用 `flock` 追加；`upgrade_csv_schema`
  留 `.schema.bak`。**但 8 个被审文件里 `os.fsync` 计数全为 0**，rename 可能先于数据落盘。
- ranked builder 经 `mpi/_entrypoint` staging，最终 `.h5`/`.json` 只在子进程成功后
  `os.replace`；单 rank `--write-coarse-cache`（`kupl_multigrid.save_legacy_mg_cache`）
  **不经过** entrypoint，直接落最终路径。
- 因此"保留历史"只能靠 `--run-label` 开新 `run_id`；不要靠"重跑同一配置"。

### 12.3 本轮登记、未修的 4 个合规缺口

| # | 触发 | 后果 | 规则 |
|---|---|---|---|
| 1 | `gauge_kind=real` 但 production gauge 缺失/不可读 → 静默改用 identity gauge（`--allow-missing-inputs`），CSV 仍记 `real` | 物理输入被替换而标签不变，属溯源缺陷 | real 缺失应 `FAIL`/`SKIP`；identity gauge 必须显式 opt-in 并落 `gauge_origin` |
| 2 | outdir 内 fallback source/gauge 不可读 → 直接生成新内容覆盖，**无 `.corrupt-*` 逻辑** | 损坏证据永久丢失 | 覆盖前先改名 `<name>.corrupt-<UTC>-<sha8>` |
| 3 | solution/report 原地写；`timeout -k 5s` 只对直接子进程 `mpirun` 发信号、未建进程组 | 超时残活 rank 可能迟写；半写文件被当产物 | 先写 `.<name>.<run_id>.partial` 再 replace；`start_new_session=True` 并 kill 整个进程组；`TIMEOUT` 行禁止 PASS、`--resume` 不得当完成 |
| 4 | 手动把 `--write-coarse-cache` 指向**既有** cache | 重建一开始就替换 rank 文件并把 manifest 改成 `status=BUILDING`，超时后原 cache 不可恢复 | 禁止该用法；重建前 `mv cache cache.bak-<utc>`，默认 no-clobber |

另记：`--mode prepare` 每次都会重建 `outdir/caches` 下的 global cache（只查外部
`DEFAULT_CACHE_DIRS`，不查 outdir 已有 cache）；`cache_index.json`/`file_hashes.json`
无 `flock`（多 shard 并发 last-writer-wins）；`--resume` 只跳过 PASS/SKIP，`TIMEOUT`
会被重跑并再次覆盖同一 `run_id` 的产物。

> **`.corrupt-20261115` 的来源**：16:28 对 L32 `ones` 与 L24 `point` 的改名是**会话内的
> 手工修复动作**，不是脚本行为——`grep -c corrupt pyqcu/testing/kupl/test_kunpeng_all.py`
> 当前为 `0`。任何"脚本会自动备份坏输入"的说法都不成立。

### 12.4 后续批次照此执行的规则（输入不可变）

1. **写域白名单**：只写 `--outdir`；`data/production_*`、`data/kupl_*`、slab 目录、LIME 一律只读。
2. **读入即登记**：读任何输入先算 `sha256` 落 row，行末复核 `mtime+size+sha`，变了即 `FAIL`。
3. **输出三件套**：`tmp → fsync(file) → rename → fsync(dir)`；禁止对最终路径
   `h5py.File(final,"w")` / `write_text(final)` / `open(final,"wb")`。
4. **超时语义**：本轮 `run_id` 的产物一律不可信，必须列 `*.partial` 并隔离；不碰输入。
5. **坏输入**：只有 outdir 内坏产物允许 `.corrupt-*` 备份后重建；**原输入损坏只 FAIL 并报告，
   禁止自动重建/自动替换**（与"real gauge 不重生成"同一条）。
6. **cache 生命周期**：默认 no-clobber，重建前备份，interim manifest 不得出现在最终路径。
7. **只读审计留痕**：每轮落 `input_immutability.json`（每个输入 `path/sha256/size/
   mtime_before/mtime_after/verdict`）到 `--outdir/logs/`。

### 12.5 复核命令（只读）

```bash
cd /home/zhangxin/PyQCU
# 1) 全域 fsync 计数（期望 0）
for f in pyqcu/testing/kupl/test_kunpeng_all.py pyqcu/testing/simple/kupl_multigrid.py \
         pyqcu/testing/simple/build_full_mg_cache.py pyqcu/testing/simple/_common.py \
         pyqcu/tools/_io.py pyqcu/tools/_legacy_mg_cache.py \
         pyqcu/testing/simple/prepare_case.py pyqcu/testing/simple/prepare_mpi_slab.py; do
  printf '%-55s %s\n' "$f" "$(grep -c fsync "$f")"; done
# 2) 脚本内无 .corrupt-* 逻辑（期望 none）
grep -rn corrupt pyqcu/testing/kupl/test_kunpeng_all.py || echo none
# 3) 输入只读：驱动里所有 h5py 打开点（应只见 "r"）
grep -n 'h5py.File' pyqcu/testing/simple/kupl_multigrid.py
# 4) 无 --force/no-clobber 语义（期望 none）
grep -n force pyqcu/testing/kupl/test_kunpeng_all.py pyqcu/testing/simple/build_full_mg_cache.py || echo none
```

## 13. 启动成本与 loop 原型（追加）

> 证据等级：**B**。数值来自 `data/kupl_loop_proto_20261115/**` 的落盘 `timings*.txt`
> 与 JSON；loop 原型**没有**并入 `pyqcu/testing/kupl/test_kunpeng_all.py`
> （`grep -c loop` = `0`），所以它只是可复用的测量，不是现行模式。

### 13.1 计算节点 cn12090：A/B/C/D 分段

| 阶段 | 内容 | wall (s) | 剩余预算 |
|---|---|---:|---:|
| A `A_mpi_boot` | 仅 MPI 启动（16 rank，各报 `ready_seconds` 32.2–33.5） | **38.993** | 230 |
| B `B_import_only` | 仅 Python 侧 import | **91.716** | 139 |
| C `C_loop_3cases` | 同进程连续跑 3 个 case | **113.939** | 24 |
| D | **跳过**（只剩 24 s） | — | — |

- **A+B = 130.709 s 是单次启动成本**；C（3 例）只要 113.939 s。若逐例起进程，
  3 例启动成本约 `3×130.709 = 392.1 s` → 本批可摊销量级约 **261.4 s**。
- **口径必须写清 rank 数**：`node/logs/phases.log` 第一行是
  `host=cn12090 NPROC=16 GRID=2 2 2 2`，落盘证据只支持 **16 rank**；
  任务记录里"32r"的说法**没有**独立产物支撑（待确认）。
- 作业侧：该作业 `elapsed 246 s`、分配 270 s（`start 1794726700` / `deadline …970`），
  说明"一个作业只跑 3 例"已经接近 900 s 作业的安全区边缘。

### 13.2 登录节点 login06（`mpi_size=2`、OMP 18）：**没有**摊销

| 组 | 例数 | 外层 wall (s) | process / session (s) | 读法 |
|---|---:|---:|---|---|
| `baseline_b3` / `baseline_b4` | 1 / 1 | 22.024 / 28.015 | — | 单例基线 |
| `loop_same3` | 3 | **74.772** | 74.22 / 72.92 | 24.9 s/例，与单例 22.0–28.0 s/例 在噪声内 |
| `loop_fault`（中途注入故障） | 3 | 66.888 | 66.18 / 64.03 | 故障后仍 `rc=0` |
| `loop_syserr`（系统错误注入） | 3 | 67.802 | 66.02 / 64.48 | 同上 |
| `loop_mixed` | 4 | — | 176.12 / 142.09 | `import_total=34.01 s`（其中 numpy `32.88 s`，离群） |

- 登录节点 `import_total` 只有 `1.3 s`（`loop_same3`），单例主导项是**独立真残差重建**
  （`call_seconds` 20.8–31.5 s 量级）与 `builder 2.47–2.54 s`，每例都要付 → loop 摊销不可见。
- 计算节点相反：`import 91.7 s + MPI boot 39.0 s` 占绝对主导，loop 只付一次 → 摊销成立。
- **结论**："一次启动多 case"是降低**每单元成本**的唯一有效手段（整行 `elapsed_seconds`
  为 `95.4–271.4 s`、中位 `178.1`，与 A+B+C 同量级），但在登录节点无收益；
  要用就得按计算节点的 130.7 s 启动成本设计批大小，并受 900/1200 s 作业墙钟约束。

### 13.3 与"启动 100/200 s"口径的关系

- 可复核的量级：**import ≈ 92 s**（计算节点 B 阶段）、**单次启动 A+B ≈ 131 s**、
  **典型整行 ≈ 178 s（中位，范围 95–271 s）**。
- "启动 100/200 s"若指"一个 case 从提交到出结果通常 100–200 s"，与本表一致；
  若指"MPI 启动本身 100–200 s"，则**不成立**（A 阶段实测 39.0 s）。
  引用时必须写清是哪一段（**待确认**外部口径的具体含义）。
