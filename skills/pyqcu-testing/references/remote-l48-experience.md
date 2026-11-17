# L48 远端经验速查（pyqcu.test / KUPL）

本文件是 [`../SKILL.md`](../SKILL.md) 的按需参考，只列 **2026-11-17 L48 远端会话**
中可复用的判定规则与最短命令。设计、状态表与完整证据见
[`docs/kupl_remote_l48_experience_20261117.md`](../../../docs/kupl_remote_l48_experience_20261117.md)
与 [`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md)；
操作者逐步检查单见 [`l48-remote.md`](l48-remote.md)。本文不复制设计正文。

## 1. 先记住 12 条

1. **版本字符串不是制品身份**：远端是 0.1.15 非 git 快照 + 0.1.17 Python 手工覆盖，
   不是 PyPI 0.1.17；用 md5/SHA 和文件计数绑定具体树。
2. **三个分母不能混**：远端覆盖树 `--check` 18/18；本机当前工作树 **20/20**
   （17:55；17:47 曾为 19/19，并发批次又加了一项检查）；
   `--quick`/`--list` 19 个 tier-0 case（11 torch / 5 kupl / 3 cuda）。check 数会漂移，
   引用必须带时间。
3. **Python 3.10 语法门**：`source[*tuple]` 是 PEP 646，3.10 直接 `SyntaxError`；
   先拼元组再索引。`ast.parse(feature_version=(3,10))` **不够**，必须加结构性
   PEP 646 扫描；解析器不遵守 flag 时返回 `SKIP`，不得 PASS。
4. **文件名不是 dtype**：`*_c64.h5` 实际可能是 complex128。只认
   `dataset.dtype` + `pyqcu.test.io.describe(..., role="gauge")` 回读。
5. **HDF5 rank-local hyperslab 很贵**：contiguous、`t` 最快 → 384 B 基本读；
   16 rank 1/16 slab 是 10,380 次 `pread`、3.56× 放大；32 rank 1/32 是 7.1×。
   先试 `--gauge-slab-dir`/`--source-slab-dir` + `prepare_mpi_slab.py`。
6. **`dtfs` 会间歇性停 33–36 s**：syscall 数/CPU 时间不变、wall 突变；固定惩罚
   像 storage-client timeout/retry。不要把它归因于算法或带宽。
7. **`wall_seconds` 不是端到端**：32r 是 1.66 s solver / 248 s mpirun / 251 s job；
   16r 是 183 s / 703 s / 705 s；16r MG 是 110 s / 2656 s。引用必须带上下文。
8. **183 s 不是 16-rank 基线**：共享节点 `cn12038`（历史排除名单）、grid 不同、
   每 site-iteration 25.07 µs vs 32r 0.456 µs（55×）；只能写“病态记录”，不能
   写 scaling，也不能反推 16 rank 太慢。
9. **标 MG 的结果可能回退到 fine BiCGStab**：`atol=1e-4` × `coarse_tol_factor=1e4`
   ⇒ `coarse_atol=1.0`，粗修正为 0，`fallback_to_fine=true`。用
   `KUPL_REQUIRE_MG_CORRECTION=1` 验证；残差逐位相同就是回退信号。
10. **cache 只认同 identity**：`lattice`、`dtype`、`mass`、`csw`、`mg_grid`、
    `nv_iters`、`dof`、`seed`、`gauge` SHA、`ranks`、`grid`、`verify_status=0`。
    本轮只有 16r cache（1.276 GiB）；32r/两节点没有匹配 cache → **DEFERRED，不重建**。
11. **2 节点分配 ≠ 跨节点 MPI；本账号的 SSH 不是可用 launcher**：`2152277` 是
    placement probe（12 s，只打印分配，无 `mpirun`）；`2152326/2152334` 的
    `mpirun` 都是 `RC=255`（host key/permission）。2026-11-17 22:11 生成
    `~/.ssh/id_ed25519_pyqcu` 后，计算节点 `sshd` 已 `Server accepts key`，但随后
    `Connection closed`——这是计算节点不允许直接 shell，不是缺 key/host key。**不要
    继续走 `plm_rsh`**；已验证的两节点路径是 `CCS_DRUN_RANK_FILE` + `drun -mpi pmix`
    （job `2155536`, 32 rank, c64 PASS）。`drun` 会忽略 `-nn/-rpn/-R`，实际用 rankfile；
    rankfile 每 rank 18 core，不是 36。正式 `pyqcu.test` 矩阵行仍 **UNVERIFIED**。
    harness 两节点已覆盖 c128 32r / c64 16r / c64 64r / c64 MG 16r（§8.1），全部
    PASS；MG 只是 cache-load，粗修正未生效。
12. **PENDING 不携带数值信息**：`2153008` / `2153005` / `2152970` 都没有
    `execNodes`；不能填单节点控制值，不能写两节点成功。

## 2. 环境最短配方

`<HK>` = `/work_ssd/software/HPCKit/26.1.RC1`；`source ./env.sh` 不够，必须叠加：

```bash
export PATH=<HK>/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:/opt/batch/cli/bin:$PATH
export LD_LIBRARY_PATH=<HK>/hmpi/bisheng/release/hmpi/lib:<HK>/compiler/gcc/lib64:<HK>/compiler/bisheng/lib:/work_ssd/software/HPCKit/alllib:$LD_LIBRARY_PATH
export OPAL_PREFIX=<HK>/hmpi/bisheng/release/hmpi
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=18 KUPL_OMP_THREADS=18
export KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16 KUPL_SHM_HALO=0 KUPL_HUGEPAGE=1
export OMPI_MCA_rmaps_base_oversubscribe=1
```

| 缺失项 | 典型症状 |
|---|---|
| `compiler/gcc/lib64` | `GLIBCXX_3.4.29 not found` |
| `compiler/bisheng/lib` | `libomp` 加载失败 |
| `OPAL_PREFIX` | `mpirun` 选项被忽略/报错 |
| `hmpi/lib` | HMPI 运行库找不到 |

单 rank 冒烟：

```bash
mpirun -np 1 --bind-to none python3 -m pyqcu.test --quick --backend kupl \
  --only kupl/abi/symbols,kupl/dslash/wilson --json -
```

> 多 rank 正式 KUPL 用例会 `SKIP(mpi_single_rank)`；多 rank 走
> `pyqcu/testing/simple/kupl_multigrid.py` 或 `kupl_qcu_tests`。

UCG 26.0/26.1 警告每 rank 一行，非致命；25–47 次 reduction × 0.45–1.02 ms
≈ 0.011–0.048 s（183 s 的 0.006–0.026%）。**不能解释 183 s**；但它是 HPCKit
安装不一致，引用性能前要记录并优先修。`--mca coll ^ucg` A/B 本轮未做（TODO）。

## 3. Python 3.10 快速自检

```bash
python -B -m pyqcu.test --check            # 本机 20/20（17:55；17:47 为 19/19）；远端覆盖树 18/18
python -B -m pyqcu.test --list --json -    # count=19, [8,8,8,16], c64
python -m py_compile pyqcu/test/*.py       # exit 0
python -B -m pytest -q pyqcu/testing/python/test_pyqcu_test_syntax_floor.py
```

反例（必须被拒）：

```python
source[0, spin, color, *site]        # PEP 646, Python 3.11+ only
source[(0, spin, color, *site)]      # 3.10 OK（括号元组），不要误报
```

闸门要两层：`ast.parse(..., feature_version=(3,10))` + 结构性 PEP 646 扫描。
`feature_version` 在 CPython 3.12/3.13 会接受上面的非法写法；若发现 flag 未被
执行，检查返回 `SKIP`（带原因），不得 PASS。

## 4. 输入与 dtype

```bash
# 1) 真实 dtype/shape（文件名不算）
python3 -c 'import h5py,sys; d=h5py.File(sys.argv[1],"r")["data"]; print(d.dtype, d.shape)' <file.h5>

# 2) 正式端口 role-aware 回读
python3 -c 'from pyqcu.test import io; print(io.describe("<file.h5>", role="gauge"))'

# 3) 全文件身份
sha256sum <file.h5>
```

L48 canonical 参考（远端）：

| 对象 | bytes | dtype | SHA256 前缀 |
|---|---:|---|---|
| `gauge_..._c128.h5` | 6,115,301,376 | complex128 | `b931b0cf…` |
| `gauge_..._c64.h5` | 3,057,653,856 | complex64 | `c8e8df3b…` |
| `source_..._c64.h5` | 1,019,222,016 | complex64 | `e4dc2a4c…` |
| `source_..._c128.h5` | 2,038,461,656 | complex128 | `3e07d910…` |

- c64 cast 必须按 t-slab 分块，不要一次读 6 GB；`.partial` 有 HDF5 锁，**不要**
  在旧 writer 未退出时启动第二个。
- c64 cast 必须 bit-exact 校验（elements/mismatches/max_abs_diff/zero_slabs/
  nonfinite），再拿 `io.describe` 回读。
- source 是 `PCG64 normal → astype(complex64)`、`seed=42` 的自造源；c128 source
  是精确 widening，不是新物理。
- `case_manifest.json` 可能把 c64 文件名和 c128 size/SHA 混记；以 `dataset.dtype`
  和实测 SHA 为准。

## 5. HDF5 读取与 `dtfs`

| 事实 | 数值 |
|---|---:|
| 16r 1/16 slab 逻辑 | 191,102,976 B |
| HDF5 384 B elementary runs | 3,981,312 |
| 合并后 `pread` | 10,380 |
| `rchar` / 放大 | 679,480,530 B / **3.56×** |
| 32r 1/32 slab 放大 | **7.1×** |
| 全文件顺序读 3.06 GB | **0.506 s**（21 次 `pread`） |
| 16r 全文件各读一遍 | **4.603 s**（比 16 个 hyperslab 的 36.2 s 快 ~9×） |
| `dtfs` 停顿 | **+33…+36 s**，3/12 次；与 syscall 数无关 |

规避：

```bash
python3 pyqcu/testing/simple/prepare_mpi_slab.py \
  --input <gauge.h5> --lat 48 48 48 96 --grid 2 2 2 4 \
  --output-dir <slab-dir> --kind gauge --dtype c64 --hash-input

python3 pyqcu/testing/simple/kupl_multigrid.py ... \
  --gauge-slab-dir <slab-dir> --source-slab-dir <slab-dir>
```

实测 per-rank slab 文件 16r cold `3.855 s` / warm `0.254 s`；预期 183 s 作业 →
~129 s（**预测，未在 L48 32r 复跑**）。`--write-coarse-cache` 与 slab-dir 是否
兼容**未实测**；MG cache 构建要单独选路径。

## 6. 结果判读表

| 说法 | 判定 |
|---|---|
| 32r c64 BiCGStab `1.6646 s` / 11 it / `7.126e-5` / 3.14 GiB | **VERIFIED**（`2152757`，`cn12072`，独占，单样本） |
| 16r `183.0 s` / 11 it | **VERIFIED 但病态**；不是基线，不作分母 |
| `183 / 1.6646 ≈ 110×` | **不是 scaling**；节点/独占性/grid 全不同 |
| 16r MG `110.35 s` + builder `1741.88 s` | **VERIFIED 数字，但不是有效 MG**；实际 fallback fine BiCGStab |
| 16r MG cache `1.276 GiB` | **VERIFIED**；有效、可复用（`patch_galerkin_error 3.59e-07`） |
| 17.44 GiB L48 cache | **错误**；那是 E=12 历史预算，不是本轮 E=2 |
| 两节点已跑通 | **必须限定**：KUPL harness 已有 `2155536`（c64 32r）与 `2155785`/`2155830`/`2155868`/`2155888`（c128 32r、c64 16r、c64 64r、c64 MG 16r cache-load）PASS；正式 `pyqcu.test` 矩阵行仍 `UNVERIFIED`（远端无 `--l48-two-node`），MG 只是 cache-load |
| `2152277` 两节点成功 | 只证明 placement；12 s 脚本无 `mpirun` |
| 32r dslash 可用 | **UNVERIFIED**；计算节点挂起，登录 smoke ~2 s 不算 |
| L48 CG | **BLOCKED/已退役**；无 L48 CG harness |
| Strict-MG | **排除**；`--strict-acceptance` 只是容差门，不是 Strict-MG |

### 6.1 时钟

| 作业 | solver wall | mpirun | job wall |
|---|---:|---:|---:|
| `2152757` 32r | 1.6646 s | 248 s | 251 s |
| `2152325` 16r | 183.0004 s | 703 s | 705 s |
| `2152391` 16r MG | 110.3546 s | — | 2656 s |

`wall_seconds` 只覆盖 solver 调用；`elapsed` 还含 MPI 启动、HDF5、独立残差。
不能相加，不能互换。

## 7. Cache 与 break-even

```text
identity = lattice, dtype, mass, csw, mg_grid, nv_iters, dof, seed,
           gauge_kind/gauge_canonical_sha256, ranks, grid, verify_status=0
```

- 只允许同 identity 复用；不要按文件名/路径/大小复用。
- 16r cache：`caches/l48_16r_mg_l2r5`，1.276 GiB，E=2；只给同 identity 的 16r 行。
- 32r/两节点：无匹配 cache → **DEFERRED**；不要拿 16r cache 改名。
- 不要每次 run 重建；`--write-coarse-cache` 只在需要新 identity 时用。
- break-even：`1741.88 / (183.00 − 110.35) ≈ 24` 次 solve（算 233 s 额外 gauge
  read 约 27 次）。**前提**是那 110.35 s 真的是 MG；本轮不是，所以这个数只能作
  “cache 构建摊销”的量级，不是 MG 加速结论。

## 8. 调度与 launcher

请求 `-nn 2 -N 32 -rpn 16` 是**正确**的：`djob -ll 2153008` 解析为
`nnodes=2, replica=32, ntasksPerNode=16, reqCPU=18, reqMem=8192, exclusive=true`；
每节点 16×18=288 CPU（608 的 47%）、128 GiB。

PENDING 三因素：

1. 队列/节点：7–8 个 `OK` 节点，没有 608 空闲 CPU；
2. `-x job` 独占：需要两个完全空节点；
3. 账号配额：`testusers_rebuild` used 608 / limit 1216 = 2×608；2 节点独占吃满
   整个配额（**推断，`dlimit` permission denied，规则未直接读到 → UNVERIFIED**）。

两节点启动（2026-11-17 22:15 已验证，KUPL harness）：

```bash
# rankfile 由 CCS_ALLOC_FILE 生成；每 rank 18 core
export CCS_DRUN_RANK_FILE=$OUT/logs/$JOBTAG-rankfile
/opt/batch/agent/tools/drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v -- \
  python3 -B pyqcu/testing/simple/kupl_multigrid.py --lat 48 48 48 96 \
  --grid 2 2 2 4 --levels 1 --restart 0 --max-iter 50 --atol 1e-4 ...
```

- job `2155536`（`cn[19544,19546]`）PASS：32 rank、11 it、真残差
  `7.12609211762054e-05`、solve wall `1.1778153 s`。
- 设置 `CCS_DRUN_RANK_FILE` 后，`drun` 会警告并忽略
  `-N/-nn/-rpn/-nl/-R/--cpu-bind/--distribution`；实际用 rankfile。必须用报告的
  `mpi_size`/`grid`/`local_lattice` 复核，不能根据告警推断 rank 数。
- **SSH 不是启动路径**：22:11 的新 key 已被计算节点接受
  （`Server accepts key`），随后 `Connection closed`；计算节点不允许直接 shell。
  `plm_donau` 在该环境仍 `query()` 拒绝，`--mca plm donau` 不得恢复。
- `drun` 只在计算节点存在；登录节点没有。必须在作业容器内调用。
- 以上都是 **KUPL harness** 结果；不是 `pyqcu.test --l48-two-node` 正式矩阵行
  （远端无该 flag），不能用于正式矩阵或 scaling 结论。

### 8.1 两节点变体（KUPL harness，2026-11-17 23:30）

| 变体 | 作业 | dtype | ranks / 节点（core） | grid / local | it | 真残差 | solver wall | RSS | 报告（`data/l48_variants_20261118/`） |
|---|---|---|---:|---|---:|---|---:|---|---|
| c128 32r | `2155785` | c128 | 32 / 2（18） | `2 2 2 4` / `24×24×24×24` | 16 | `7.466169937510113e-06` | `3.9853145 s` | 3.26 GiB | `reports/v1_c128_2n32r_l1r0.json` |
| c64 16r | `2155830` | c64 | 16 / 2（36） | `2 2 2 2` / `24×24×24×48` | 11 | `7.125979321648942e-05` | `3.0647245 s` | 4.12 GiB | `reports/v2_c64_2n16r_l1r0.json` |
| c64 64r | `2155868` | c64 | 64 / 2（9） | `2 2 4 4` / `24×24×12×24` | 11 | `7.125993463624718e-05` | `2.7971336 s` | 1.32 GiB | `reports/v3_c64_2n64r_l1r0.json` |
| c64 MG 16r cache-load | `2155888` | c64 | 16 / 2（36） | `2 2 2 2` / `24×24×24×48` | `null` | `7.125979321648942e-05` | `2.1298542 s` | 8.10 GiB | `reports/v4_c64_2n16r_legacy_mg_l2r5.json` |

启动/判读教训：

- **rankfile 胜过 `-nn/-rpn/-R`**：四个变体每次都告警并忽略这些 flag；实际
  ranks/node 由 rankfile 行数决定（8、16、32 都跑过），core/rank 由宽度决定
  （9、18、36 都被接受）。只看报告 `mpi_size`/`grid`/`local_lattice`。
- 16/32/64 rank × 2 nodes 都能起，但每变体 1 样本、节点对与 cores 不同 → **不算
  scaling**。
- MG `2155888` 的残差与 c64 16r levels=1 **逐位相同**，且
  `atol=1e-4 × coarse_tol_factor=1e4 ⇒ coarse_atol=1.0` → fine fallback；只证明
  cache-load + solve 路径（`verify_status=0`、`builder_seconds=21.42 s`），不是 MG
  加速。32r Legacy-MG 无同 identity cache → **DEFERRED**，不重建。
- 上报三件套：`djob -l` 的 `jobExitCode=0` + report `status=PASS` + 独立真残差
  `< 1e-4`。v4b 诊断 `2155899` 归档时仍 PENDING，不作结果。

## 9. 发布与证据纪律

- 已发布 `0.1.17` / `test37`，**不得覆盖**；下一版 `0.1.18` / `test38` 只在十轮
  复审全过后发布。
- `bug45` 的 tag-chain `[BLOCK]` 是历史例外：不 force-push、不移动/改写
  `bug45`/`dev106`、不说“已修复”。
- 100 列 CSV / 21 sheet Excel parity 未通过前，不得声称已替换旧驱动。
- 每份报告至少给：`job_id`、`exec_nodes`、`ranks/rpn/grid/threads/shm`、输入
  SHA/dtype/shape、`wall`/`elapsed`、`iterations`、独立真残差、`status`、报告路径。
- 缺 `exec_nodes`、SHA 或独立残差的数字 → `UNVERIFIED`。
- 状态词：`VERIFIED` / `VERIFIED-C` / `PENDING` / `BLOCKED` / `DEFERRED` /
  `UNVERIFIED` / `TODO`；`PENDING` 不是 PASS。
- 只读审计优先；不写远端 repo；Python 用 `-B`；监视器有硬超时；不 cancel 其他
  作业；快照带时间戳（`2152277` 会从 PENDING 变 SUCCEEDED）。

## 10. 相关入口

- 经验全文：[`docs/kupl_remote_l48_experience_20261117.md`](../../../docs/kupl_remote_l48_experience_20261117.md)
- 设计与权威状态：[`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md)
- 十轮发布终审：[`docs/pyqcu_test_l48_review.md`](../../../docs/pyqcu_test_l48_review.md)
- P0/P1 风险登记：[`docs/pyqcu_test_audit_risk_20261117.md`](../../../docs/pyqcu_test_audit_risk_20261117.md)
- 操作者检查单：[`l48-remote.md`](l48-remote.md)
- 本轮任务日志：[`logs/v202611172230.txt`](../../../logs/v202611172230.txt)
- 两节点变体日志：[`logs/v202611172330.txt`](../../../logs/v202611172330.txt)
- 上一轮任务日志：[`logs/v202611171745.txt`](../../../logs/v202611171745.txt)
- 上一轮任务日志：[`logs/v202611171416.txt`](../../../logs/v202611171416.txt)
