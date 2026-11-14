# KUPL 多进程/多线程扩展性与性能断崖巡查（2026-11-14）

> 证据等级：B 级（源码机制 + 既有生产剖析 + 本轮节点/环境核验；solve 测量未完成）。本文只讨论
> **KUPL CPU 后端**在单节点、固定全局格点 `24³×64`（L24）、c64 下的扩展性。
> 跨节点结论单独标注，未验证项不写成结论。

## 1. 结论摘要

1. **断崖不是单点，而是三层机制叠加**：
   - **带宽 roofline**：32 rank 时 dslash 已接近节点 STREAM 带宽上限，
     继续加 rank 不会增加带宽，只会降低每 rank 计算量；
   - **内核路径门限**：固定全局格点增加 rank 会让 `half_sites` 跌破
     block-SIMD 阈值（65536），并把 `local[3]` 从 2 变成 1 以下（非法）；
   - **固定同步成本**：每迭代 2 次全局归约，归约延迟随 rank 数上升，
     而计算时间随 rank 数下降，同步占比按 `1/N` 放大。
2. **线程断崖有明确阈值**：当前节点 16 个 CPU NUMA、每 NUMA 38 CPU、
   可用 592 CPU；`ppr:2:numa` 下每 rank 配额 19 CPU，因此
   **`KUPL_OMP_THREADS ≤ 19` 是物理上限，18 是当前最优**。超过 19 时
   库内守卫只比较“线程数 vs 本 rank 的 NUMA cpuset（38）”，不会抛错，
   而是把两个 rank 的线程绑到同一批 CPU 上（`cpus[thread % size]`），
   形成静默超订阅。
3. **rank 上限是几何硬约束**：KUPL 要求 local 各维为偶数
   （`make_geometry`），L24 的 t 方向最多 32 rank。超过 32 必须切空间维
   （2D/4D 分解），此时 `grid[0..2]==1` 不再成立：
   `t2 lane`、`split-boundary` 两条生产路径同时失效，并引入 gauge corner
   halo。这是“多 rank 后突然变慢”的首要算法原因。
4. **多节点**：共享内存 halo 要求 `shared_size == world`，跨节点自动回退
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

- 每 rank 分到 **一个完整 NUMA 的 38 CPU**（`bind_logical_numa` 按
  `local_rank % nodes` 选节点，再按 `threads` 截断）；
- 每个 NUMA 放 2 个 rank，所以**每 rank 的物理配额是 19 CPU**；
- `KUPL_OMP_THREADS=18` → 36 线程/38 CPU，留 2 CPU 给系统；
  `=19` → 38/38，刚好；`=20` → 40 线程/38 CPU，**静默超订阅**。

库内守卫（`lattice_cpu.cpp:668-673`）只在 `threads > cpuset.size()` 时
抛错；这里 `cpuset.size()==38`，所以 `TH=20..38` 都不会抛错，而是让
两个 rank 的线程按 `cpus[thread % 38]` 重叠绑定。**这是线程断崖的真正
入口**：用户以为“加线程=加速”，实际是把两个 rank 压到同一批核上。

判据（单节点、ppr:P、NUMA 宽 W）：

```text
TH_max(physical) = floor(W / P)          # 本节点 W=38, P=2 → 19
TH_max(cgroup)   = floor(592 / ranks)    # 32 rank → 18.5 → 18
```

取两者较小值。**生产配置应满足 `threads ≤ 18`（32 rank）**，并用
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
- 线程上限判据 `TH_max = min(floor(W/P), floor(usable_cpus/ranks))`
  由节点谱直接导出；32 rank、ppr=2 时为 18。
- `-N`/`-rpn`/`-R` 语义与 MPI 环境入口通过失败链确认。

### 5.5 下一步（按优先级）

1. 用 §5.3 的配方在**同一节点、同一时段**重跑 `sc32t18` 作为基线；
   成功后再加 `TH=36`（验证静默超订阅）与 `TH=9`。
2. 若 32r 基线复现，再做 `grid 2 1 1 16` 的 2D 分解（>32 rank）对照，
   确认 t2/split-boundary 失效的代价。
3. 多节点回退：先给驱动补 hostfile，再做 16+16 rank 的 MPI halo 对照。
4. 每次作业记录 `scaling_cur_freq`、`half_sites`、`local[3]`、命中内核与
   `KUPL_TOPOLOGY_REPORT` 的 cpuset。

## 6. 调优与验证建议

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
