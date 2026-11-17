# KUPL 多进程/多线程扩展性与性能断崖巡查（2026-11-14）

> 证据等级：B 级（源码机制 + 既有生产剖析 + 本轮节点/环境核验 + C++ 算子强扩展实测；Python solve 测量未完成）。本文只讨论
> **KUPL CPU 后端**在单节点、固定全局格点 `24³×64`（L24）、c64 下的扩展性。
> 跨节点结论单独标注，未验证项不写成结论。
> **例外**：§10 是 2026-11-17 追加的 L48 / 两节点现状（C 级），不在上述 L24 范围内。

## 1. 结论摘要

1. **断崖不是单点，而是三层机制叠加**：
   - **带宽 roofline**：32 rank 时 dslash 已接近节点 STREAM 带宽上限，
     继续加 rank 不会增加带宽，只会降低每 rank 计算量；
   - **内核路径门限**：固定全局格点增加 rank 会让 `half_sites` 跌破
     block-SIMD 阈值（65536），并把 `local[3]` 从 2 变成 1 以下（非法）；
   - **固定同步成本**：每迭代 2 次全局归约，归约延迟随 rank 数上升，
     而计算时间随 rank 数下降，同步占比按 `1/N` 放大。
2. **算子级强扩展已实测**：全局 L24 固定、每 rank 18 线程时，4/8/16/32 rank
   的 Wilson dslash 为 `0.626/0.445/0.283/0.239 s`（50 it），32 rank 相对
   4 rank 仅 `2.63×`（线程数 8×），16→32 边际只有 `1.19×`；每线程吞吐从
   `1.13` 降到 `0.37 Gflop/s`（§7）。
3. **线程断崖是“失败关闭”而非静默降级**：调度器 `ppr:2:numa` + 32 rank
   给每 rank 的实测 cpuset 是 **18 CPU**；`KUPL_OMP_THREADS=18` 是当前上限，
   `TH=36` 实测直接抛 `KUPL_OMP_THREADS exceeds the NUMA-local CPU set`。
   只有脱离调度器 cpuset、手工整节点 `mpirun` 时才可能出现 38-CPU 集合，
   此时 `TH=20..38` 才会别名绑定，必须用 `KUPL_TOPOLOGY_REPORT` 核对。
4. **rank 上限是几何硬约束**：KUPL 要求 local 各维为偶数
   （`make_geometry`），L24 的 t 方向最多 32 rank。超过 32 必须切空间维
   （2D/4D 分解），此时 `grid[0..2]==1` 不再成立：
   `t2 lane`、`split-boundary` 两条生产路径同时失效，并引入 gauge corner
   halo。这是“多 rank 后突然变慢”的首要算法原因。
5. **多节点**：共享内存 halo 要求 `shared_size == world`，跨节点自动回退
   MPI；单次 wait 从约 `0.16 ms` 回到约 `0.78 ms`，52 次 dslash 约多
   `32 ms`（相对 0.155 s 约 +21%）。跨节点生产的驱动还缺 hostfile
   （历史两节点作业在 mpirun 阶段 RC=255）。

## 2. 几何与流量模型（L24 c64）

### 2.1 每 rank 的局部格点与 halo

L24：`24³ = 13,824` 个空间点/时间片，全局 `884,736` site。
KUPL 的 parity 压缩把 t 折半，t 方向 face 始终是**一个 t2 切片**：

```text
face_complex = 13,824 site × 12 comp = 165,888 complex = 1.327 MB (c64)
```

| ranks (t) | local t | half_sites | full sites/rank | 每 dslash 收+发 | 每 site halo B | block SIMD | t2 lane | split-boundary |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 4 | 16 | 110,592 | 221,184 | 5.31 MB | 24 | **开** | 关 | 关 |
| 8 | 8 | 55,296 | 110,592 | 5.31 MB | 48 | 关（<65536） | 关 | 关 |
| 16 | 4 | 27,648 | 55,296 | 5.31 MB | 96 | 关 | 关 | 关（除非 `KUPL_ENABLE_HT2_TBOUNDARY_SPLIT`） |
| 32 | 2 | 13,824 | 27,648 | 2.65 MB（dedup） | 96 | 关 | **开** | **开** |

读法：

- halo 字节数与 rank 数无关（除 32 rank 的 dedup 减半），但**每 rank 计算量
  按 1/N 下降**，所以 halo/计算比在 16 rank 之后不再恶化，真正恶化的是
  “固定同步”和“内核路径”。
- 4 rank 是唯一命中 block-SIMD 的点；8/16 rank 落到标量 + boundary-lane
  路径；32 rank 命中 t2 lane + split-boundary。**跨 rank 比较不是同一内核**。
- 32 rank 时 `local[3]/2 == 1` ⇒ 所有 site 都是 boundary site
  （`backend.cpp:452-467`），part1/part2 各覆盖全格，这是 t2 路径的代价，
  也是它仍最快的原因：dedup 打包 + 共享 halo。

### 2.2 带宽 roofline

按每 site 的解析访存（c64）：

```text
gauge: 8 dir × 9 complex × 8 B = 576 B
src  : 12 complex × 8 B        =  96 B
dst  : 12 complex × 8 B        =  96 B
合计                            = 768 B/site
```

32 rank：每 dslash 每 rank `27,648 × 768 B = 21.2 MB`，52 次 dslash →
`1.10 GB`。这与 profiler 的 `828 MB` 计数口径不同（profiler 模型偏低），
两者都列出，不混用。

- 节点 STREAM triad 探针：`18.3–20.1 GB/s·rank × 32 ≈ 585–643 GB/s`。
- 实测 dslash 中位 `0.067483 s`：
  - 解析流量：`1.10 GB / 0.0675 s = 16.3 GB/s·rank`，聚合 `522 GB/s`，
    约为探针峰值的 **87%**；
  - profiler 口径：`828 MB / 0.0675 s = 12.3 GB/s·rank`，聚合 `393 GB/s`，
    约 **65%**。
- 结论：dslash 已经进入**带宽受限区**。`0.1 s` 总目标对应的 dslash 预算
  约 `59 ms`（按解析流量与 600 GB/s），而当前 dslash 已经是 `67.5 ms`；
  不减少字节数或复用 gauge，单靠加 rank/调线程无法达到 0.1 s。这与
  2026-11-05 的结论一致（`logs/kupl_qcu_mg_instructions_20261105.txt`：
  “0.1 s 在当前硬件与几何下不可达，只能寄望于规模或算法级变化”）。

### 2.3 固定同步成本

BiCGStab 每迭代的全局归约（KUPL 融合后）：

```text
1) global_norm_and_dot(rhat, r)  → 1 次 Allreduce（3 real）
2) global_norm_dot_pair(t, s)    → 1 次 Allreduce（3 real）
```

12 次外层迭代 + 初始化/收尾 ≈ 51 次归约（与 `KUPL_PROFILE_OPT` 的
`reduce calls=51` 一致）。32 rank 实测 `0.45 ms/次` → `23 ms`，占
`0.155 s` 的 **15%**。

强扩展下：计算时间 ∝ 1/N，归约延迟随 N 缓慢上升（log P + 延迟项），
所以同步占比近似 ∝ N。这是 Amdahl 型串行段，也是“很多 rank 后收益消失”
的第二层原因。

## 3. 线程断崖的定量判据

当前节点（18/18 节点一致，见
[`kupl_kunpeng_node_survey_20261114.md`](kupl_kunpeng_node_survey_20261114.md)）：

```text
nproc = 592        CPU NUMA = 16      每 NUMA cpulist = 38 CPU
dnode CPU = 608    16 CPU 不在作业 cgroup 内
```

`mpirun --map-by ppr:2:numa` + `KUPL_NUMA_BIND=1`：

- 调度器 `ppr:2:numa` + 32 rank 给每个 rank 的实际 cpuset 是 **18 CPU**
  （`KUPL_TOPOLOGY_REPORT` 实测 `cpuset=94-111`、`18-35` 等，宽 18）；
- 因此**当前生产上限是 `KUPL_OMP_THREADS=18`**；`TH=19..36` 会被库内
  守卫直接拒绝（失败关闭），不会静默超订阅。

库内守卫（`lattice_cpu.cpp:668-673`）在 `threads > cpuset.size()` 时抛错。
本轮 `TH=36` 实测报
`KUPL_OMP_THREADS exceeds the NUMA-local CPU set`，没有产生任何计时；
这不是断崖，而是**明确的参数错误**。只有在脱离调度器 cpuset、手工在整节点
上 `mpirun` 时，NUMA-local 集合才可能是 38（每 NUMA 38 CPU、两个 rank
各 19），此时 `TH=20..38` 才会按 `cpus[thread % 38]` 重叠绑定，必须用
`KUPL_TOPOLOGY_REPORT` 核对。

判据（单节点、ppr:P、每 NUMA 宽 W、调度器每 rank cpuset 宽 C）：

```text
TH_max = min( floor(W / P), C )          # 本节点 W=38, P=2, C=18 → 18
```

**生产配置应满足 `threads ≤ 18`（32 rank）**，并用
`KUPL_TOPOLOGY_REPORT=1` 核对每个 rank 的 `cpuset` 互不重叠、
`threads == bound_threads`。

## 4. rank 断崖的几何判据

1. **t 方向上限**：`local[3]` 必须为偶数（`lattice_geometry.cpp:37-46`），
   L24 的 `T=64` ⇒ 最多 32 rank（`local[3]=2`）。64 rank 沿 t 会得到
   `local[3]=1`，直接非法。
2. **超过 32 rank** 必须 `grid = (2,1,1,16)` 一类 2D 分解：
   - `grid[0..2]==1` 不成立 → `t2 lane` 与 `split-boundary` 失效
     （`lattice_wilson_dslash.cpp:64-73,1414-1416`）；
   - 需要 x/y/z 面 halo；gauge 需要 corner halo
     （`lattice_clover_dslash.cpp:144-192`）；
   - `boundary_sites` 的结构改变，part1/part2 不再覆盖全格。
3. **block SIMD**：`half_sites ≥ 65536` 才启用
   （`lattice_wilson_dslash.cpp:36-51`）。L24 上只有 ≤4 rank 满足；
   从 4 → 8 rank 会**关闭**跨 site 向量化，单 site 成本上升，这是
   “4→8 rank 不成比例”的候选原因（历史 rank 扫描中 8 rank 中位
   `0.4378 s` 对 32 rank `0.1832 s`，同时每 rank 计算量 ×4）。
4. **共享 halo**：要求 `shared_size == world`，跨节点自动回退
   （`lattice_wilson_dslash.cpp:1088-1096`）；回退后 wait 从 `0.16` 回到
   `0.78 ms/次`，52 次 dslash 约 +32 ms。

## 5. 本轮实测

> 本轮尝试在 2026-11-14 09:00–09:20 补齐 `TH` 与 rank 扫描，但**未取得新的
> solve 数字**；按 10 min 门限终止。以下记录失败链与可复现配方，避免下一位
> 接手者重复踩坑。扩展性结论仍以第 2–4 节的源码机制与
> [`kupl_l24_perf_20261113.md`](kupl_l24_perf_20261113.md) 的既有实测为依据。

### 5.1 既有实测（2026-11-13，L24 c64，cn19699）

| 量 | 值 | 来源 |
|---|---:|---|
| solve 墙钟（32r×18t，shm halo） | 0.154883 s（8 样本中位） | `kupl_l24_perf_20261113.md` §14 |
| solve 墙钟（同配置，MPI halo） | 0.181733 s（8 样本中位） | 同上 |
| dslash 合计 | 0.067483 s / 52 次 | `KUPL_PROFILE_OPT=1` |
| clover 合计 | 0.013720 s / 52 次 | 同上 |
| reduce | 0.45 ms/次 × 51 | 同上 |
| rank 扫描 | 8: 0.437795/0.449878/1.731601；16: 3.054642/0.479433/3.533757；32: 0.183248/0.176644/0.185366 | 同上 §12 |

### 5.2 本轮失败链（按发现顺序）

| 作业 | 节点 | 现象 | 根因 |
|---|---|---|---|
| 2122996 / 2122998 | cn12072 / cn12108 | 12 s 结束，`RC=127` | 该批 cn12xxx 节点按默认环境无 `mpirun`；需先 `source /work_ssd/software/HPCKit/25.2.1/setvars.sh` |
| 2123038 / 2123054 | cn19699 | 12 s 结束，无 MPI | 作业脚本未 source HPCKit；CLI `--mpi hmpi` 不改变 `PATH` |
| 2123119 / 2123134 | cn19699 | solve 10 min 无输出 | `-N 1 -R cpu=8` 只给 8 CPU，32 rank×18 thread 在 8 CPU 上超订阅；改用 `-R cpu=576` 后仍未见输出 |
| 2123143 / 2123146 | cn12070 / cn19699 | preflight 通过（`mpirun -np 2 hostname`、numpy/h5py 正常），solve 阶段 600 s 无 `report.json` | 未定位；按 10 min 门限终止，不把超时写成通过 |
| 2123210（bench 线程扫描） | cn19699 | `TH=36` 立即报 `KUPL_OMP_THREADS exceeds the NUMA-local CPU set` | 库内守卫失败关闭；调度器 cpuset=18/rank，TH>18 非法。同作业 `TH=1/4/9/18` 均正常 |
| 2123827 / 2123789 等（重测） | cn19699 | `mpirun: Error: unknown option "-n"`（`-np`、`--version` 同样） | 2026-11-14 10:40 起，HPCKit 25.1.0/25.2.0/25.2.1/26.0/26.1 的 HMPI `mpirun` 对所有选项均报 unknown option；01:10 同驱动仍可运行。原因未定位（疑环境/license 变更），不是 KUPL 代码问题 |

### 5.3 已验证可用的作业配方

历史成功作业（`t18df1..3`，01:03–01:14）的特征：

```bash
# 提交端（登录节点，已 source ~/env.sh 以启用 conda）
dsub -q q_hpcapp -T 900 -n <tag> \
     -N 32 -rpn 32 -R "cpu=18,mem=8G" \
     env OMP_WAIT_POLICY=ACTIVE ... bash ~/sh/x_case_l24_v7.sh <tag> 32 18 2 600

# 作业内
source /work_ssd/software/HPCKit/25.2.1/setvars.sh
export PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH
export OMPI_MCA_rmaps_base_oversubscribe=1
```

要点：

- **`-N <ranks> -rpn <ranks> -R "cpu=<threads>"`** 是历史可复现的资源配置；
  `-N 1 -R cpu=576` 的 cgroup 行为与前者不同，本轮未调通。
- `-N 16 -rpn 1` **只执行 1 个 task**（`BATCH_TASK_INDEX=0`），因此不能用来
  一次采集多节点；逐节点核验必须一个节点一个作业（见节点核验文档）。
- `--mpi hmpi` 命令行开关不等于 MPI 进入 `PATH`；必须在作业脚本内 source
  HPCKit 的 `setvars.sh`。
- conda 环境不要依赖作业默认 `python3`（无 numpy）；直接
  `export PATH=/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH`，
  或 source `~/env.sh`（内含 `module purge/load`，在容器内可能更慢）。

### 5.4 本轮新增但可用的证据

- 节点谱：18/18 节点 1.55 GHz / 上限 2.0 GHz、592 CPU、16+16 NUMA
  （见节点核验文档）。这修正了“存在 2.0 GHz 正常节点”的前提：
  **当前采样窗口内没有 2.0 GHz 运行态节点**。
- 线程上限判据 `TH_max = min(floor(W/P), cpuset_size)` 由节点谱与调度器
  cpuset 直接导出；32 rank、ppr=2、cpuset=18 时为 18，`TH>18` 失败关闭。
- `-N`/`-rpn`/`-R` 语义与 MPI 环境入口通过失败链确认。

### 5.4.1 点源（全 1）文件（2026-11-14）

按“所有点实部=1、虚部=0”生成，`(4,3,X,Y,Z,T)` complex64，已抽样验证
`imag_max=0.0`：

- L24：`data/production_L24/source_24x24x24x64_c64_ones.h5`（84.9 MB）
- L32：`data/production_L32_v1/source_32x32x32x64_c64_ones_v2.h5`（201.4 MB）

驱动已加 `SOURCE_OVERRIDE`，两个 `run_l*_prod.sh` 已指向上述文件；
`mpirun` 启动问题解决后可直接重投。

### 5.4.2 远端 LIME 文件校验（2026-11-14）

`rsync --append-verify` 完成后：

- 本地 `/home/zhangxin/data/beta6.20_mu-0.2825_ms-0.2310_L64x128_cfg_1020.lime`
  与远端 `~/data/...` 均为 **19,327,493,568 B**；
- 远端前 4 字节为 `45 67 89 ab`（LIME 魔数）；
- 传输日志 `100%`、`xfr#1, to-chk=0/1`，耗时约 55 分钟（平均 5.55 MB/s）。

### 5.4.3 求解产物协议（2026-11-14）

- 每次 solve 必须保存 `dest`：驱动传 `--output "$W/solution_${DTYPE}.h5"`
  （可用 `OUTPUT_OVERRIDE` 覆盖），并在 `meta.txt` 写入 `SRC=`/`DEST=`。
- `src` 若原本就是被读取的文件，则跳过重复保存。
- 校验：用 gauge + src + dest 复算 `A x = b`，与 `report.json` 的
  `final_true_residual` 对照；独立实现可以是 Torch-QCU 或 NumPy 参考。

### 5.4.4 收敛判据（2026-11-14 新规）

- 生产：c64 `1e-6`、c128 `1e-9`，均为相对 **norm**（`||r||/||b||`）。
- 驱动按 dtype 设 `ATOL` 并传 `--strict-acceptance`；Python 报告的
  PASS/FAIL 只按 `atol`，不再叠加 dtype 地板。
- 测试保留地板（c64 `2048ε`、c128 `128ε`）以探测浮点极限；生产不得用
  地板放行。
- 注意：c64 在 L24/L32 上能否真正达到 `1e-6` 需实测；若到不了，严格门
  会报 FAIL，不能改回宽松地板。

### 5.5 下一步（按优先级）

1. 用 §5.3 的配方在**同一节点、同一时段**重跑 `sc32t18` 作为基线；
   成功后再加 `TH=36`（验证失败关闭边界）与 `TH=9`。
2. 若 32r 基线复现，再做 `grid 2 1 1 16` 的 2D 分解（>32 rank）对照，
   确认 t2/split-boundary 失效的代价。
3. 多节点回退：先给驱动补 hostfile，再做 16+16 rank 的 MPI halo 对照。
4. 每次作业记录 `scaling_cur_freq`、`half_sites`、`local[3]`、命中内核与
   `KUPL_TOPOLOGY_REPORT` 的 cpuset。

## 6. 当前最佳配置速查（2026-11-14 子代理汇总）

### 6.1 L24（`24³×64`，c64）— 有可引用最佳

| 项 | 值 |
|---|---|
| 墙钟 | **中位 `0.154883 s`**（n=8，均值 `0.154132`、sd `0.002503`、范围 `0.150240–0.157304`） |
| 算法 | **parity-Schur BiCGStab**：`--levels 1 --restart 0`（不是 MG） |
| 并行 | 32 rank × 18 thread，`grid 1 1 1 32`，`ppr:2:numa`，`--bind-to none` |
| halo | 单节点共享内存 halo（默认）；`KUPL_SHM_HALO=0` 对照中位 `0.181733 s` |
| 绑定/内存 | `KUPL_NUMA_BIND=1`、`KUPL_MEMORY_POLICY=preferred`、`KUPL_MEMORY_NODE=local+16`、`KUPL_HUGEPAGE=1` |
| 内核开关 | `OMP_WAIT_POLICY=ACTIVE`、`KUPL_ENABLE_T2_LANES=1`、`HALO_PACK6=1`、`HALO_DEDUP_PACK=1`、`T_LOCAL_IN_PART1=1`、`ENABLE_BOUNDARY_LANES=1` |
| 物理参数 | `mass=-0.277`、`csw=1.160920226`、`atol=1e-4`、12 it、真残差 `5.384866e-05` PASS |
| 节点 | `cn19699`（1.55 GHz 现态 / 2.0 GHz 上限） |
| 口径 | 求解器内最终残差复查被 `KUPL_SKIP_FINAL_RESIDUAL=1` 跳过；外层独立 FULL 真残差门仍在跑 |
| 证据 | `docs/kupl_l24_perf_20261113.md` §13–§14；`skills/pyqcu-kupl/SKILL.md` L24 表；dev141 |

引用时必须带限定：单节点、BiCGStab（非 MG）、不含 init/IO、n=8、1.55 GHz 现态。
文档内仍有三处旧“最优”表述（`0.186230`、`0.188526`、`0.4346`）需要后续清理，
不要在对外材料中混用。

MG 路径另有 L24 `0.135579 s`（`restart=1`、32 rank、全 1 点源、单样本，cache 构建
`493.6 s` 未计入），见 §6.4；源与算法都与本节不同，**不替换**本节 `0.154883 s`
的“最佳”表述，两者不可直接比较。

### 6.2 L32（`32³×64`，c64）— 无当前最佳

**没有可按当前代码/当前配置引用的 L32 最佳耗时。** 唯一真实 L32 KUPL 实测是
2026-10-23 的首测（早于 2026-11 全部优化）：

| 用例 | 结果 | 说明 |
|---|---|---|
| KUPL Legacy-MG 2 levels | 1 rank `36.2029 s`、2 rank `36.8231 s`，真残差 `3.424e-7/3.422e-7` PASS | grid/threads/restart/样本数/内存峰值均未记录 |
| KUPL BiCGStab | 2 rank `19 it / 17.7376 s`，真残差 `3.424e-7` PASS | 同上；同批 MG 比 BiCGStab 慢 2.08× |
| 2026-11-06 鲲鹏复跑 | 1-rank cache 构建与 32-rank BiCG 均超 10 min 门禁，无 report | `BLOCKED`，不是算法结论 |

易误引：`0.214–0.219 s` 的 L32 数字是 **CUDA Legacy-MG on A100**；`8.3e-8`
是 Strict wrapper vs raw ABI 的**数值等价性**，不是计时。若必须给替代口径，
只能引用 L24 的 `0.154883 s` 并注明与 L32 不可换算。

## 7. C++ 算子强扩展实测（2026-11-14，全局 L24 固定）

为绕开 Python/HDF5 流水线，改用 `cpp/kupl/qcu/build-native-lto/kupl_qcu_bench`
（`applyWilsonDslashQcu`，单 parity，50 次迭代，c64，全局 `24³×64` 固定）。
作业 `2123210`，cn19699，`-R cpu=576`，默认环境（**未开** t2 lane / PACK6 /
dedup / T_LOCAL；split-boundary 由几何触发）。

| ranks | 总线程 | 50 it (s) | 每迭代 (ms) | 相对 4 rank | 每线程 Gflop/s |
|---:|---:|---:|---:|---:|---:|
| 4 | 72 | 0.626466 | 12.529 | 1.00 | 1.13 |
| 8 | 144 | 0.445490 | 8.910 | 1.41 | 0.79 |
| 16 | 288 | 0.283023 | 5.660 | 2.21 | 0.63 |
| 32 | 576 | 0.238596 | 4.772 | **2.63** | **0.37** |

读法：

- 并行效率（相对 4 rank）：8 rank `70.5%`、16 rank `55.3%`、32 rank `32.9%`；
  边际收益 `1.41× → 1.57× → 1.19×`，**16→32 只有 +19%**，这就是“多 rank
  后收益断崖”的直接量化。
- 每线程吞吐从 `1.13` 降到 `0.37 Gflop/s`（−67%），说明瓶颈不是单核算力，
  而是共享内存带宽/延迟与固定同步；继续加 rank 只是把同一批带宽切得更碎。
- 该曲线是**默认内核路径**；同几何的生产配置（t2 lane + PACK6 + dedup +
  T_LOCAL + boundary lanes）在 NP=32 的 dslash 为 1.30 ms/次（52 次
  0.067483 s），比基准的 4.77 ms/次快 **3.68×**。因此不能把这条曲线当作
  生产路径的扩展性；生产路径需单独测。

同作业的线程扫描（NP=32，local t=2，同一默认路径）：

| TH | 50 it (s) | 每迭代 (ms) | 相对 TH=1 |
|---:|---:|---:|---:|
| 1 | 1.543505 | 30.870 | 1.00 |
| 4 | 0.499424 | 9.988 | 3.09 |
| 9 | 0.311865 | 6.237 | 4.95 |
| 18 | 0.238596 | 4.772 | **6.47** |
| 36 | — | — | 失败关闭（`KUPL_OMP_THREADS exceeds the NUMA-local CPU set`） |

18× 线程只换来 6.47×（每线程效率 36%）；`TH=36` 被库内守卫拒绝，不是静默降级。

## 8. 调优与验证建议

1. **固定格点强扩展**：先确认 `half_sites`、`local[3]`、命中的内核路径；
   不要把 4/8/16/32 rank 当作同一算法比较。
2. **线程设置**：按 §3 的 `TH_max` 公式；`KUPL_TOPOLOGY_REPORT=1`
   验证 cpuset 不重叠；不要在 `KUPL_NUMA_BIND=1` 时同时设
   `OMP_PROC_BIND`。
3. **多 rank 优先 t 方向**：L24 的 32 rank 是 t 方向上限，也是当前最优；
   超过 32 必须切空间维并重新做基线，不能沿用 t2 路径的数字。
4. **多节点**：先给驱动补 hostfile/`-host` 支持，再用两节点 16+16
   rank 验证回退；跨节点数字必须单独标注“MPI halo”，不得与单节点
   共享 halo 混算。
5. **频率**：跨节点比较前读 `scaling_cur_freq`；1.55 vs 2.0 GHz 对应
   最大 1.29× 差异（见节点核验文档）。
6. **不要**用单次样本判定 <5% 的差异；同节点抖动约 ±2%，跨节点更大。

### 6.3 2026-11-14 点源（全 1）重测

源定义按用户要求：**全格所有点实部=1、虚部=0**（`(4,3,X,Y,Z,T)` complex64，
抽样 `imag_max=0.0`）。配置与 §6.1 相同：32r×18t、`grid 1 1 1 32`、
`ppr:2:numa`、共享 halo、`OMP_WAIT_POLICY=ACTIVE`、t2/PACK6/DEDUP/T_LOCAL/
BOUNDARY 全开、`levels=1/restart=0`（parity-Schur BiCGStab）、`atol=1e-4`、
`mass=-0.277`、`csw=1.160920226`、cn19699。

| 格点 | 中位 (s) | 样本 (s) | 迭代 | 真残差 | 状态 |
|---|---:|---|---:|---:|---|
| L24 `24³×64` | **0.173362** | 0.171331 / 0.173362 / 0.182659 | 11 | `6.756443e-05` | PASS |
| L32 `32³×64` | **0.348549** | 0.348549 / 0.369371 / 0.340658 | 12 | `6.831831e-05` | PASS |

样本说明：L24 与 L32 各 3 次（`rem14_*` 1 次 + `rem15_*_2/3` 2 次，作业
`2123971/2123978`、`2124128`，均 `RC=0`）。首次样本使用 P1 之前的构建，
后两次使用 P1 构建（零拷贝默认关闭，仅多出 profile 计时，量级 ~10 µs）；
三次迭代数与真残差完全一致。L24 极差 `0.171331–0.182659`（约 ±3.3%），
L32 极差 `0.340658–0.369371`（约 ±4.1%），均属同节点抖动范围。

TH 扫描（全 1 点源，32 rank，其余配置同上，单样本）：

| TH | L24 墙钟 (s) | L24 相对 TH18 | L32 墙钟 (s) | L32 相对 TH18 |
|---:|---:|---:|---:|---:|
| 9 | 0.215964 | +24.6% | 0.460573 | +32.1% |
| 16 | 0.182221 | +5.1% | 0.352243 | +1.06% |
| 18 | 0.173362（中位） | — | 0.348549（中位） | — |
| 36 | — | 失败关闭 | — | 失败关闭 |

L32 的 TH16 与 TH18 基本持平（+1.06%，在抖动内），TH9 明显变差；两尺寸均以 TH18 为当前最优。

证据：`data/kupl_opt_20261113/remote/rem14_l24_ones/report.json`、
`.../rem14_l32_ones/report.json`（作业 `2123971`、`2123978`，`RC=0`）。

注意：

- 这是**单样本**，且源与 §6.1 的旧点源/随机源不同，**不能与 `0.154883 s`
  直接比较**；L24 的 11 it 与旧源的 12 it 也说明收敛路径不同。
- L32 的 `0.348549 s` 是**当前代码 + 当前配置**下的首个可引用 L32 数字，
  但仍只有 1 个样本，且 `levels=1`（BiCGStab），不是 MG。
- 运行环境：`OPAL_PREFIX` 必须在作业/驱动内显式设置，否则 HMPI 的
  `mpirun` 会把所有选项报成 `unknown option`（见 §5.2）。

### 6.4 L24 Legacy-MG cache 构建与参数扫描（2026-11-14）

> 证据等级：C 级时点证据（远端作业产物；MG 扫描为单样本）。所有数字取自远端
> `data/kupl_scaling_20261114/`；该目录尚未同步到本地工作区，引用时以远端文件为准。

#### 6.4.1 cache 身份（1 rank 构建）

| 项 | 值 |
|---|---|
| cache 文件 | `data/kupl_scaling_20261114/cache_l24_parallel.h5` |
| 伴随清单 | 同名 `.json` |
| `file_size_bytes` | `143,335,424`（约 `136.7 MiB`） |
| `setup_seconds` | `493.63817879895214` |
| `status` | `PASS` |
| `setup_method` | `cxx-null-vector+local-33-stencil` |
| `setup_backend` | `kupl-qcu-cxx` |
| `mg_grid` | `[2,2,2,1]` |
| `nv_iters` | `2` |
| lattice | `[24,24,24,64]` |
| 构建并行 | 1 rank × 18 线程，并行 stencil 默认开启 |

#### 6.4.2 构建期的四项修复

| # | 修复 | 说明 |
|---|---|---|
| 1 | 删除 per-probe 全量 fill | 每个探针不再重填整份 fine 向量 |
| 2 | 第一步 dslash 稀疏到 `block + 1-hop` | 只对约 48 个 site 应用，不再全量 dslash |
| 3 | 时间窗口改为 `min(fine, 3*block+3)` | 窗口计算含 t 维 |
| 4 | 并行 stencil 默认开 | 1 rank 构建即走并行 stencil |

修复前 1-rank **串行与并行均 `rc=124`**（600 s 门限内未完成）；修复后可完成，
但 `setup_seconds=493.63817879895214` 仍远超 `30 s` 目标（约 `16.5×`）。

#### 6.4.3 32 rank MG 参数扫描

配置：32 rank、`grid 1 1 1 32`、`--coarse-cache` 复用 §6.4.1 的 cache、
`--mg-grid 2 2 2 1 --nv-iters 2`、`atol=1e-4`、全 1 点源、c64、
`OMP_WAIT_POLICY=ACTIVE`，t2 / PACK6 / DEDUP / T_LOCAL / BOUNDARY 全开。

| 用例 | restart | coarse-max-iter | tol | wall (s) | 真残差 | 状态 |
|---|---:|---:|---|---:|---:|---|
| r3_m200_t4 | 3 | 200 | 1e4 | 0.136258 | 6.756443214052956e-05 | PASS |
| r1_m200_t4 | 1 | 200 | 1e4 | 0.135579 | 6.756443214052956e-05 | PASS |
| r6_m200_t4 | 6 | 200 | 1e4 | 0.136228 | 6.756443214052956e-05 | PASS |

读法：

- 三个用例收敛到**同一真残差**（`6.756443214052956e-05`，逐位相同）。
- restart 1/3/6 的墙钟极差 `0.135579–0.136258 s`（约 `0.5%`），**单样本、在
  噪声内**，不足以对 restart 排序；`r1` 略低不等于 `restart=1` 更优。
- MG 的 `iterations` 字段为 `None`：native MG 不暴露可靠的外迭代计数，
  **不得按迭代数比较 MG 与 BiCGStab**。

#### 6.4.4 与 BiCGStab 的对照

同 gauge/source、`atol=1e-4`、32r×18t 的 BiCGStab **3 样本中位** `0.173362 s`
（§6.3）。MG `restart=1` 的 `0.135579 s` 比它快约 **21.8%**，真残差与对照
**逐位一致**。

口径不对称，引用时必须写明：

- MG 是**单样本**，BiCGStab 是**3 样本中位**；
- MG 墙钟**不含 cache 构建**（`493.6 s`）。cache 只有在同一 gauge 上跨作业复用
  时才能摊薄；若把构建计入单次求解，MG 的总代价是 `493.6 s` 量级，远慢于
  BiCGStab，21.8% 只在长期复用场景下成立；
- 两边都不是生产 `atol`（本轮 `1e-4`，生产 c64 目标 `1e-6`）。

#### 6.4.5 口径与限制

- cache 由 **1 rank** 构建、**32 rank 直接读取复用**，跨 rank 数复用已验证；
  这是本轮打通 L24 真实 MG 多 rank 的关键。
- MG 数字不含 init/IO，也不含 cache 构建；单样本。
- 32 rank 仍是 L24 的 t 方向几何上限（§4）；64 rank 需 2D 分解，未测。
- **L32 仍无可用 cache**：1-rank 构建时间随体积增长（L24 已 `493.6 s`），多 rank
  构建尚未完成，L32 MG 仍不可用；不能把 L24 的 21.8% 外推到 L32。

#### 6.4.6 第二轮扫描（作业 `2125551`，running）

第二轮参数：`cmax=50`、`tol=1e3/1e5`、`--no-deflate`、`atol=1e-6`；结果应落在
`data/kupl_scaling_20261114/mg_l24/timing.txt` 与同目录各 `*.json`。写入本节时
该目录在本地工作区不存在（远端产物未同步），作业状态为 **running**，本节不预判
其结果；拿到 `timing.txt` 后按 §6.4.3 的表格式补入。

#### 6.4.7 结论

1. **1-rank 构建的全局 cache 可被 32 rank 读取复用**：L24 Legacy-MG 多 rank 首次
   跑通。SKILL/历史文档中“多 rank MG 无法运行”的旧表述，只对未预置 cache 或
   无可复用 cache 的尺寸成立。
2. **L24 MG 已可跑**：`restart=1` `0.135579 s`、`PASS`、真残差
   `6.756443214052956e-05`。
3. **相对 BiCGStab 快约 21.8%**，但样本数与摊销口径不对称，只能作为“值得继续
   扫”的信号，不能作为生产选路定论。
4. **cache 构建仍是瓶颈**：`493.63817879895214 s`，比 `30 s` 目标慢约 `16.5×`；
   L32 因此仍不可用。
5. 多 rank 构建（bg-22）与第二步稀疏化（bg-12）在进行中；两者决定 cache 构建
   能否进入 30 s 量级。

### 7.1 P1 shm halo 零拷贝的本地 A/B（2026-11-14）

实现：新增 `KUPL_SHM_HALO_ZEROCOPY`（默认关），consume 握手前移后直接
`pack_fermion_face` 进本 rank 的 shm 槽；dedup 面从 minus 槽 memcpy 到 plus
槽；MPI 路径不变。`KUPL_PROFILE_OPT` 新增 `halo_pack=` / `halo_sync=` /
`halo_copy=` 三项。

正确性：本地 `ctest -R "kupl_qcu_(1rank|2rank|t2|shm|4rank)"` 在默认与零拷贝下
均 **11/11 PASS**。

性能（本地 x86，4 rank，`24³×16`，8 线程/rank，20 it）：

| 算子 | 默认 (s) | 零拷贝 (s) | 变化 |
|---|---:|---:|---:|
| wilson | 3.2608 | 3.4520 | **+5.9%** |
| clover | 3.7494 | 3.9603 | **+5.6%** |
| clover_schur | 7.4659 | 7.5320 | +0.9% |

**本地 x86 结论**：原顺序“先打包、再等 consume 后 memcpy”把打包时间藏在
等待里；零拷贝省掉一次 memcpy，但把打包暴露到关键路径，净亏约 6%。

**鲲鹏实测（2026-11-14，cn19699，32r×18t，L24 点源全 1，`KUPL_PROFILE_OPT=1`）**：

| 用例 | 墙钟 (s) | 迭代 | 真残差 | halo_pack | halo_sync | halo_copy |
|---|---:|---:|---:|---:|---:|---:|
| zc=0（默认） | 0.180983 | 11 | `6.756443e-05` | 未计时 | 0.000985 | 0.044387 |
| zc=1（零拷贝） | **0.172885** | 11 | `6.756443e-05` | 0.040603 | 0.000203 | 0.006638 |

- 零拷贝 **快 4.47%**（`0.180983 → 0.172885 s`），迭代数与真残差**逐位一致**。
- 机制：默认路径的 copy 段 `44.4 ms`（打包后 memcpy + 等待后 memcpy）；
  零拷贝把打包 `40.6 ms` 直接写进 shm 槽，copy 降到 `6.6 ms`。
- 与本地 x86 的符号相反：x86 上 memcpy 便宜、打包重叠更重要；鲲鹏上
  memcpy/写 shm 更贵，但**3 样本中位并不支持稳定收益**。
- 补充 3 样本（非 profile，`zc1_l24_2/3`）：zc1 为
  `0.172885（profile）/0.171816/0.185719`，**中位 0.172885**；zc0 为
  `0.171331/0.173362/0.182659`，**中位 0.173362**。两者仅差 **0.28%**，
  落在同节点 ±4% 抖动内；单次 profiled 配对的 -4.47% 不能外推。
- 结论：**保持默认关闭**；若要改默认，必须先做交替配对（zc0/zc1 交替）
  的 ≥5 组样本，确认中位差 >2% 且区间不重叠。

## 9. 多 rank Legacy-MG cache：patch 构建（2026-11-14 突破）

### 9.1 根因：gauge patch 取自 rank 局部 slab

旧导出路径在 `pyqcu/testing/simple/kupl_multigrid.py` 用
`_periodic_window(gauge_local, patch_starts, patch_extent, ...)` 构造每个 rank
的 overlap patch。`gauge_local` 只是本 rank 的局部 slab；当 patch 在
x/y 上比局部盒大（4-rank `8³×16`：patch `8×8×8×16` vs local `4×4×8×16`），
该函数把局部盒**周期平铺**，halo 区域拿到的是本 rank 自己的 gauge，而不是
邻居的。实测每个 rank 的 patch 有 `6144/8192 = 75.0%` 站点与真实全局 gauge
不同。

两次逐位复现闭合了因果：

| 实验 | gauge 窗口 | lonv 来源 | 结果 |
|---|---|---|---|
| A（生产窗口） | 局部 slab 周期平铺 | ranked cache | 与失败 cache 组装**逐位相同**（四项 `max_abs=0.0`） |
| B（正确窗口） | 真实全局 gauge | 参考 lonv | 与 1-rank 参考**逐位相同**（四项 `max_abs=0.0`） |

因此 `hnn=0.493 / hdg=0.787 / sit=0.039` 全部由 gauge 窗口错误产生，
与 C++ slot/halo 语义无关。

### 9.2 修复后的算法：照抄 CUDA-QCU 的 patch 语义

每个 rank 构造一个**周期 overlap patch**，在 patch 上用已验证的单 rank
`1×1×1×1` builder 建 33-stencil，再截取中心 block：

```
halo    = 2 * mg[axis]                 # 单位：fine site
desired = local[axis] + 2 * halo       # = local + 4*mg
if desired <= global:
    start, extent, center = (coord*local - halo) % global, desired, halo
elif global % mg == 0:
    start, extent, center = 0, global, coord*local      # 退化为整轴
else:
    raise ValueError
patch_coarse = extent / step,  step = mg*(2 if axis==3 else 1)
# 截取 [center/step, center/step + local_coarse)，与全局 coarse 坐标平移一致
```

实现要点：
- patch gauge 必须来自**全局**文件（`_hdf5_gauge_patch`）、已加载的全局数组，
  或 slab 目录的全局 gather；**不得**从 rank 局部 slab 平铺。
- LatticeSet 用 `grid=1×1×1×1`、`node_rank=0`、`node_size=1`（serial-local），
  不复用旧的分布式 `StencilFrame`。
- patch 内一律周期回绕；中心截取为非周期直接切片。
- 保持 `patch_galerkin=True` 作为“halo 是否足够”的运行时门。

### 9.3 32-rank 构建结果（全部 `verify_status=0`）

| 格点 | grid | `builder_seconds` | `coarse_export` | stencil | gauge patch | lonv patch | 真残差 |
|---|---|---:|---:|---:|---:|---:|---:|
| `24³×64` | `2×2×2×4` | **24.476 s** | 22.751 s | 14.118 s | 3.455 s | 2.346 s | 4.535e-07 |
| `32³×64` | `2×2×2×4` | **53.023 s** | 48.943 s | 29.879 s | 10.275 s | 2.874 s | 4.543e-07 |
| `32³×12` | `4×2×2×2` | **11.368 s** | 10.650 s | 6.394 s | 1.164 s | 2.054 s | 4.460e-07 |

4-rank `8³×16`（grid `2×2×1×1`）对拍 1-rank 参考：
`lonv 2.697e-05 / hnn 1.710e-05 / hdg 3.098e-05 / sit 1.367e-06`，
`verify_status=0`、`converged=true`。这是 null-vector 集合构建的噪声量级，
不是错位；若需逐位一致，应复用同一份全局 lonv。

### 9.4 t-only（`grid 1 1 1 32`）在 patch 算法下不可用

patch 在 x/y 轴因 `grid=1` 必然退化为**整轴**（24），t 轴也退化为整轴（64），
于是每个 rank 都构建**整格**的 coarse 算子（约 493 s 的 1-rank 工作量），
32 rank 并行仍超时。实测所有 t-only MG 作业 `rc=124`（≈355 s），
其中**纯 BiCGStab 对照也超时**，证明是 patch 退化/启动问题，不是 MG 数值问题。

结论：多 rank MG 一律使用**均衡 4D 分解**（`2×2×2×4`、`4×2×2×2` 等），
不要再用 `1×1×1×N` 作为默认或基准。

### 9.5 MG 容差语义：1e-4 的“MG 0.128 s”不是 MG

`coarse_atol = atol × coarse_tol_factor × 3^(levels−1)`，是**乘**不是除。

- `atol=1e-4`、默认 `factor=1e4` → `coarse_atol=1.0`；粗层 BiCGStab 首行
  残差判据立即成立，解保持 0 → 修正恒为 0 → damping 搜索全部回滚 →
  `fallback_to_fine=true`，最终走纯 BiCGStab。
- 因此历史 `0.128289 s`（`restart=1, cmax=200, tol=1e5`）是 **BiCGStab 回退**，
  不是 MG；`MG 比 BiCGStab 快 26.0%` 的结论作废。
- `atol=1e-6`、默认 `factor=1e4` → `coarse_atol=1e-2`，MG 真正运行。
- cache identity 不含 `atol/coarse_tol_factor/cmax/restart/deflate`，这是正确设计：
  cache 内容只依赖 gauge/mass/csw/dof/mg_grid/nv_iters/seed/dtype。

1e-6 同网格对照（4-rank `8³×16`，grid `2×2×1×1`，cache 回读）：

| 求解器 | wall | 真残差 | 状态 |
|---|---:|---:|---|
| BiCGStab | 0.031577 s | 5.421e-07 | PASS |
| MG | 0.037935 s | 3.409e-07 | PASS |

MG 仅慢 **20%**，不是 2.93×。32-rank 均衡 grid `2×2×2×4` 的 MG 1e-6 回读：
L24 `0.519831 s`（残差 `4.536e-07`）、L32 `1.175139 s`（残差 `4.543e-07`）、
`32³×12` `0.266999 s`（残差 `4.461e-07`），三者均 `verify_status=0`、
`converged=true`。

### 9.6 旧分布式 `StencilFrame` 的潜伏缺陷（patch 路径不经过）

只读对照确认 CUDA 与 KUPL 的 slot/方向/pair/sign 表完全一致；旧分布式
push/pull 路径另有缺陷，当前导出路径（`grid=1×1×1×1` serial-local）不会触发，
但一旦强制分布式构建就会致命：

1. **push 通道 slot 混叠**：`lattice_multigrid.cpp:2316/2388` 直接用
   `spec.reverse_slot`/`spec.slot`（nn 0..7 与 diag 0..23）做 33 槽缓冲的通道号，
   nn k 与 diag k 互相覆盖；应改为 `1+slot`（nn）/`9+slot`（diag）。
2. **pull cell 索引**：`:2385-2386` 用 `point=row+shift` 覆盖所有 4 维，
   非跨界维仍应取 `row`；否则读错 cell。
3. **压缩时间 hop**：`:2079-2081` 用固定 `±1`，正确值是随站点奇偶变化的
   `Δt2 = ((x+y+z)&1)^parity`（+t）/ `1−该值`（−t）。
4. **3 维 corner 读取缺失**：`halo_direction` 对 `count>=3` 返回 -1，
   静默回绕；33 点 stencil 当前不可达，但无断言保护。

在这些缺陷修复前，不要把 `KUPL_MG_STENCIL_FORCE_DISTRIBUTED=1` 用于生产。

## 10. 2026-11-17 追加：L48 与两节点的扩展性现状

> 本节为归档批次追加，**不改 §1–§9 的 L24 结论**。L48 的权威记录见
> [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) §0.1；证据等级 **C**
> （远端作业，原始日志未归档）。

| 观测 | 数值 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| L48 c64 BiCGStab，32 rank / 1 独占节点（`2152757`） | `1.6646 s`、11 it、`7.126e-5`、RSS `3.14 GiB` | 单节点 32 rank 能跑通并达 strict `1e-4` | 不能外推两节点 |
| L48 c64 BiCGStab，16 rank / 共享节点（`2152325`） | `183.0 s`、11 it、同残差、RSS `4.42 GiB` | 共享节点下退化到不可用 | **不是 16 rank 基线**；不证明“16 rank 慢 110×” |
| L48 c64 Legacy-MG，16 rank | wall `110.35 s` + builder `1741.88 s`（总 `2656 s`） | 16 rank MG 能收敛（`7.126e-5`） | 无 32 rank / 两节点 MG 证据 |
| 两节点（`2153008`） | PENDING | 请求语法被受理 | 无 `execNodes`，两节点未验证 |

- 本文件的 32/16 rank 断崖机制（§4、§7）是在**固定 L24 全局格点**下得到的。L48 的
  `half_sites` 是 L24 的 12×（`10,616,832` 对 `884,736` site），block-SIMD（`65536`）
  与 t2-lane/split-boundary 的门限命中点完全不同，**不能把 L24 的 rank 断崖曲线
  平移到 L48**；L48 的 grid 仍受 §4 的“局部各维必须偶数”约束（16 rank 用 `2 2 2 2`，
  32 rank 用 `2 2 2 4`，`1 1 1 32` 的 `t_finest` 在 32 rank 下为奇数、不可用）。
- 两节点仍缺 hostfile/rankfile（§1 第 5 条）：`2153008` 的 launcher **没有**传
  `--rankfile`，即使分到两个节点也不能跨节点起 rank。
- **共享节点会同时压缩 cpuset 与 HBM 带宽**，本文件 `18 CPU/rank`（`ppr:2:numa`）
  的前提在共享节点上不成立。任何共享节点样本必须标注资源独占性，否则不得进入
  扩展性表；16 rank 的可用基线仍缺。
- L48 cache：只认同 identity 复用，当前只有 **16 rank** cache（E=2、`1.276 GiB`）；
  32 rank / 两节点**没有匹配 cache → 暂缓，不重建**。CG 组已退役，不再排 CG 用例。
