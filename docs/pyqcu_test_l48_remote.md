# pyqcu.test 的 L48 双节点正式端口设计（2026-11-17）

> 状态：**设计提案（P）+ 已验证远端基线 + L48 单节点 c64 + KUPL harness 两节点
> （c64 + 四变体）PASS**。状态以 §0 与 §0.1 为准：远端访问、0.1.17 覆盖、
> `--check` 18/18、HMPI 单 rank、L48 LIME→HDF5 转换已验证；**c64 BiCGStab（32 rank）、
> c64 Legacy-MG（16 rank）与 KUPL harness 两节点 c64（32 rank，job `2155536`）已有
> 实测结果**（§0.1、§0.1.6）；
> 23:30 又归档 **KUPL harness 两节点四个 PASS**（c128 32r、c64 16r、c64 64r、c64
> Legacy-MG 16r cache-load，§0.1.7）。**正式端口 `pyqcu.test --l48-two-node` 的两节点
> 矩阵行仍未验证**（远端覆盖树无该 flag）；c128 只有 harness 结果、两节点 MG 只走
> cache-load 路径、CG、32 rank dslash/clover 均无有效结果，CSV/Excel parity 仍在
> 验证。工作区存在另一批次的并发实现（未提交、未审计、未在远端跑过），见 §2.4.7；
> 不得据此把任何 TODO 当成已修复。设计正文（§1、§3–§8）保持原样，§2.4 收录
> 2026-11-17 在用户内部登录环境（`login02`）实测的通路、运行时、L48 转换与调度
> 探测结论，§0.1 收录同日的首批数值结果与 cache/CG 策略。**两节点 harness PASS；
> 正式端口两节点未 PASS。**
>
> 目标环境：灵晟（Lingsheng / LX2）两个计算节点，真实 L48 组态
> `beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`（`48x48x48x96`）。
> 目标是用正式端口 `pyqcu.test` 取代
> [`pyqcu/testing/kupl/test_kunpeng_all.py`](../pyqcu/testing/kupl/test_kunpeng_all.py)
> 与 [`submit_kunpeng_all.sh`](../pyqcu/testing/kupl/submit_kunpeng_all.sh) 的生产矩阵，
> 同时保留后者的 `list/prepare/prewarm/solve/merge/run/validate` 台账语义，以及它的
> 100 列 CSV 与 21-sheet Excel 汇总契约（见 §2.4.6）。

## 0. 状态：已验证 vs TODO（2026-11-17 14:16；结果批次更新 17:45）

本节是本文的**权威状态表**。`已验证` 只指远端实测或本机可复跑命令；`TODO`/`PENDING`
在任何时候都不得写成 PASS。设计正文与契约见 §1、§3–§8，本节不复制其内容；同日
首批数值结果、cache/CG 策略与阻塞见 §0.1。

| # | 项目 | 状态 | 证据 / 边界 |
|---:|---|---|---|
| 1 | 远端登录 `login02` / `nsls_yangyb` | **已验证** | §2.4.1；经用户提供的内部路由，私网地址与密钥不入库 |
| 2 | 远端仓库 `/home/share/nsls_yangyb/PyQCU-main` | **已验证** | 非 git 的 **0.1.15 快照**；0.1.17 的 Python 代码覆盖前先备份为 `pyqcu.bak.20261117`（§2.4.2） |
| 3 | `python -m pyqcu.test --check` | **已验证：18/18** | 远端 `kupl_qcu.py`/`cuda_qcu.py` 先修掉仅 Python 3.11 合法的 `source[*tuple]` 索引（3.10 会 `SyntaxError`）；本仓库同一改动**未提交**（§2.4.2） |
| 4 | Python / 依赖版本 | **已验证（部分原始输出 TODO）** | Python 3.10.9（anaconda 2023.3，aarch64）、numpy 1.23.5、h5py 3.7.0（`mpi=False`）、mpi4py 4.1.2、torch 1.12.1；原始 stdout 未归档（§2.4.2） |
| 5 | HPCKit 26.1.RC1 环境 | **已验证** | HMPI 库、gcc `libstdc++`（`GLIBCXX_3.4.29`）、bisheng `libomp`、`alllib` 均按 §2.4.3 叠加；`OPAL_PREFIX` 必填 |
| 6 | HMPI 单 rank | **已验证** | `mpirun -np 1 --bind-to none` 可启动；KUPL `abi/symbols` 与 `dslash/wilson` PASS（§2.4.3） |
| 7 | 远端 L48 LIME | **已验证** | `~/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`，5.7 GB，complex128；与本地同大小、同前 4 KiB SHA256（§2.4.4） |
| 8 | L48 LIME → canonical HDF5 | **已验证（单次）** | `prepare_case.py` 41 s；原始 stdout 未归档 → 计时证据仍是 TODO（§2.4.4） |
| 9 | `gauge_..._c64.h5` 的真实 dtype | **已验证：实际 complex128** | `6,115,301,376 B`，**文件名误标**；source 侧为 complex64（§2.4.4） |
| 10 | L48 c64 cast 与 `io.describe` 回读 | **已验证（按实际求解使用）** | c64 行已真实求解（§0.1 第 1–3 行）；`io.describe` 回读的原始 stdout 仍未归档 → 该条证据只能标 C 级（§2.4.4） |
| 11 | `dsub -nn 2` 两节点（`2152277`、`2153008`） | **PENDING / 未通过** | `2152277` 为 `No available node can fit ... execution nodes preference`；`2153008` 请求语法正确但卡在队列/限额；两次都没有 `execNodes`，**不证明两节点可跑**（§2.4.5、§0.1.4） |
| 12 | L48 solve / MG | **部分已验证（c64 单节点 + harness 两节点变体）** | c64 BiCGStab 32 rank 独占节点 `1.6646 s`/11 it/`7.126e-5`（`2152757`）；c64 Legacy-MG 16 rank wall `110.35 s` + builder `1741.88 s`；c64 harness 两节点 `1.1778 s`/11 it/`7.126e-5`（`2155536`，§0.1.6）；harness 变体 c128 32r / c64 16r / c64 64r / c64 Legacy-MG 16r 见 §0.1.7。**正式端口 c128 行仍 PENDING**（`2153005`），CG 无 harness、32r dslash/clover 挂起（§0.1） |
| 13 | 跨节点 launcher / rankfile | **TODO / 未实现** | 正式端口与 `pyqcu/testing/simple/mpi` 都没有 `--hostfile`/`--rankfile`/`CCS_ALLOC_FILE` 解析；历史两节点作业在 `mpirun` 阶段 `RC=255`（§5.3） |
| 14 | L48 生产 `solve/mg` 行 | **TODO** | 工作区并发实现已加入 `L48` preset（未提交），但 `--mode list --lattice-preset L48` 仍只展开 17 个 tier-0 case（CG 退役后 19−2；§2.4.7、§9.5） |
| 15 | CSV 100 列 / Excel 21 sheet parity | **验证中（未通过）** | `ledger_csv.py`/`ledger_excel.py` 仍未接线，`--csv` 仍写 30 列 `ROW_FIELDS`；驱动侧自报 `--check` 19/19、`--quick` 19 个 case、87 个测试通过（**转述，本轮未独立复跑**）。**parity 是硬性要求**，未通过前不得声称已替换旧驱动（§2.4.6） |
| 16 | 审计 P0/P1 | **未修复（NOT FIXED）** | 见 [`pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md)；本文不把任何 P0/P1 写成已修复 |
| 17 | 版本 / 标签 | **0.1.18 发布准备** | 历史发布 `0.1.17` / `test37` 保持不可覆盖；本轮按用户指令提升到 `0.1.18` / `test38`（`follow test37`）。十轮复查通过后才 upload/push/tag；`bug45` 历史 tag-chain 例外不重写（§9.6）。 |
| 18 | KUPL harness 两节点 c64 | **已验证：PASS** | job `2155536`（`cn[19544,19546]`，32 rank，`drun -mpi pmix`，rankfile）；报告 `status=PASS`、11 it、真残差 `7.12609e-05`、solve wall `1.1778 s`。**这是 harness，不是 `pyqcu.test --l48-two-node` 矩阵行**；正式矩阵仍为 `UNVERIFIED`，SSH 直接 shell 也仍不可用（§0.1.6） |
| 19 | KUPL harness 两节点变体（c128 / c64） | **已验证：4 PASS（harness）** | job `2155785`（c128 32r）、`2155830`（c64 16r）、`2155868`（c64 64r）、`2155888`（c64 Legacy-MG 16r cache-load）；全部 `status=PASS` 且独立真残差 `< 1e-4`（§0.1.7）。**MG 粗修正未生效**（cache-load 路径）；远端覆盖树无 `--l48-two-node`，正式矩阵仍 `UNVERIFIED` |

**结论**：远端基线可复跑；单节点 **c64** 的 L48 BiCGStab 与 Legacy-MG、**KUPL
harness 两节点 c64（job `2155536`）**，以及 23:30 归档的四个 harness 变体（**c128
两节点**、c64 16r/64r、c64 Legacy-MG 16r cache-load，§0.1.7）已有实测结果（§0.1、
§0.1.6）。但 **正式端口 `pyqcu.test --l48-two-node` 的两节点矩阵行、正式 c128 行、
两节点 MG 加速、CG、32 rank dslash/clover 仍无有效结果**，CSV/Excel parity 仍在
验证中。任何“L48 已全部求解”“正式端口两节点已跑通”的表述都与本表冲突。

### 0.1 L48 首批结果、cache/CG 策略与阻塞（2026-11-17 归档）

本节是 §0 的结果附录。数值来自远端作业（可用 `djob -l <id>` 复核），**只记录已被
用户确认的结果**；未给出作业号或原始 stdout 未归档的字段一律标 `TODO`/`UNVERIFIED`。
`c64` 指本轮实际求解所用的输入；`io.describe` 回读的原始 stdout 仍未归档。

#### 0.1.1 结果表

| # | 用例 | 规模 / 资源 | 结果 | 迭代 / 残差 | 峰值 RSS | 状态 |
|---:|---|---|---|---|---:|---|
| 1 | c64 BiCGStab | 32 rank，1 个**独占**节点 | `1.6646 s` | 11 it，`‖r‖/‖b‖ = 7.126e-5`（strict `1e-4`） | `3.14 GiB` | **已验证**（作业 `2152757`） |
| 2 | c64 BiCGStab | 16 rank，**共享**节点 | `183.0 s` | 11 it，同残差 `7.126e-5` | `4.42 GiB` | **已验证但病态**（作业 `2152325`）；**不作基线、不作分母** |
| 3 | c64 Legacy-MG | 16 rank | wall `110.35 s` + builder `1741.88 s`，总计 `2656 s` | 残差 `7.126e-5` | `11.93 GiB` | **已验证**；cache（E=2）`1.276 GiB`；作业号未记录 `TODO` |
| 4 | c128 BiCGStab | — | 无结果（**正式端口**单节点） | — | — | **PENDING**（作业 `2153005`）；c128 输入已验证；c128 **harness** 两节点结果见 §0.1.7 |
| 5 | 两节点（**正式端口**） | `-nn 2` | 无结果 | — | — | **PENDING**（作业 `2153008`）；请求语法正确，卡在队列/限额；harness 两节点见 §0.1.6、§0.1.7 |
| 6 | CG（任何规模） | — | 无结果 | — | — | **无 harness**；CG 组已由用户退役（§0.1.3） |
| 7 | dslash / clover 32r | 32 rank | 无结果 | — | — | **UNVERIFIED**：计算节点挂起；仅登录节点 smoke 通过 |
| 8 | Strict-MG | — | 不评估 | — | — | **排除**（本轮范围外，不作 L48 备选） |
| 9 | c64 BiCGStab **KUPL harness 两节点** | 32 rank，2 nodes `cn[19544,19546]` | `1.1778 s` solver；输入 prebind `17.433 s` | 11 it，`7.126092e-5`（strict `1e-4`） | `2.27 GiB` | **已验证：PASS（作业 `2155536`）**；`drun -mpi pmix`，**不是**正式端口矩阵行（§0.1.6） |

读法（硬约束）：

1. 第 1 与第 2 行**不是**扩展性对照：rank 数、节点独占性与排队窗口都不同，
   `183.0 / 1.6646 ≈ 110×` 只说明**共享节点下的病态**，不得写成 scaling 结论，
   也不得反推“16 rank 太慢”。16 rank 的可用基线仍缺。
2. 第 3 行 `110.35 + 1741.88 = 1852.23 s` 与总计 `2656 s` 差约 `803.8 s`
   （I/O、启动、残差等未分解项）→ **分项未拆，标 `TODO`**，不要把 `2656 s`
   全部归给 builder，也不要把 builder 当作 `wall`。
3. 第 3 行的 `E=2` 以 cache manifest 为准（本轮未展开字段定义）；MG 的 `wall`
   不含 cache 构建，引用时必须同时给 `builder_seconds`。
4. 第 5 行只证明**申请被受理**；没有 `execNodes` 就没有两节点证据。
5. 第 7 行的登录节点 smoke 只证明单 rank 能启动，**不能**替代 32 rank 结论。
6. 全部数字缺原始 `report.json`/每 rank 日志 → 引用时标 **证据等级 C**。
7. 第 9 行是 **KUPL harness**（`kupl_multigrid.py`），不是 `pyqcu.test --l48-two-node`；
   它的 PASS 不能把正式矩阵行改成 PASS，也不能外推到 c128、两节点 Legacy-MG 或
   dslash/clover。
8. 第 4 行的 c128 是**正式端口单节点**作业；c128 的 **KUPL harness 两节点**结果见
   §0.1.7（`2155785`、32 rank、16 it、`7.466e-6`）。harness 结果不能关闭 §0.1.4 第 1
   条，也不能写成正式 c128 行。

#### 0.1.2 cache 复用规则（只认同 identity；当前只有 16r cache）

- **只允许同 identity 复用**：`lattice`、`dtype`、`mass`、`csw`、`mg_grid`、
  `nv_iters`、`dof`、`seed`、`gauge_kind`/`gauge_canonical_sha256`、`ranks`、`grid`，
  且 `verify_status=0`。identity **不含** `atol`/`coarse_tol_factor`/
  `coarse_max_iter`/`restart`/`deflate`（这些不改变 cache 内容）。
- **禁止**按文件名、路径或文件大小复用：2026-11-15 的 `*_c64.h5` 实为 complex128
  已经证明文件名不可信。
- L48 当前可用 cache 只有 **16 rank 的 Legacy-MG cache**（`E=2`、`1.276 GiB`），
  它只能被**同 identity 的 16 rank 行**复用。
- **32 rank / 两节点 MG 没有 identity 匹配的 cache → 暂缓（DEFERRED），不重建**。
  重建 32 rank cache 不在本轮范围；需要时另立批次并重新记录 identity、`builder_seconds`
  与 `verify_status`。

#### 0.1.3 CG 组退役

> 两节点矩阵的声明式清单见
> [`pyqcu_test_l48_two_node_matrix.md`](pyqcu_test_l48_two_node_matrix.md)。
> **L48 CG** 组已退役；这里保留的是退役记录，不是待跑项。

- L48 **没有 CG harness**：正式端口与旧驱动都没有 L48 CG 行，也没有 L48 CG 的数值门。
- 用户已决定**退役 CG 组**：后续矩阵、脚本与文档**不再排 CG 用例**，也不新增只为
  L48 CG 服务的代码路径。
- 因此本文与相关 skills 不再保留“CG 待跑 / CG TODO”的表述。若将来重启 CG，必须
  重新立项并重跑输入门与残差门，不得沿用本轮任何结论。

#### 0.1.4 开放阻塞（本轮结束时仍未通过）

| # | 阻塞 | 影响 | 解除条件 |
|---:|---|---|---|
| 1 | c128 BiCGStab `2153005` PENDING | 无**正式端口** c128 性能/残差结果（harness 两节点见 §0.1.7） | 正式作业真正运行并产出可读报告 |
| 2 | 两节点 `2153008` PENDING | 无 `execNodes`，**正式端口**两节点未验证（harness 见 §0.1.6、§0.1.7） | 分配到两节点且 `djob -ll` 恰好两个节点 |
| 3 | 跨节点 launcher 未传 rankfile | 即使分到两节点也无法跨节点起 rank | `CCS_ALLOC_FILE` → rankfile 接线并由 `driver.run_row` 传 `--rankfile` |
| 4 | 32r/两节点 MG 无匹配 cache | MG 无法在 32r/两节点复现 | 另立批次重建 cache（本轮**不做**） |
| 5 | dslash/clover 32r 计算节点挂起 | 无 32r 算子级证据 | 定位挂起原因后重跑；登录 smoke 不算 |
| 6 | Legacy-MG 分项与作业号缺失 | `2656 s` 无法归因 | 归档 `report.json`/每 rank 日志与 `djob -l <id>` |
| 7 | CSV/Excel parity 仍在验证 | 不得声称已替换旧驱动 | 100 列逐列同序 + 21 sheet 逐项一致 |
| 8 | 远端原始 stdout 未归档 | 计时/dtype 证据只能标 C 级 | 归档 `--check`、`io.describe`、`prepare_case.py`、`djob -l` |
| 9 | Strict-MG 排除 | 不构成 L48 备选 | 需要时单独立项 |

> **22:15 更新**：KUPL harness 两节点 c64 已 **PASS**（§0.1.6，job `2155536`）。上表第 2
> 条指的是**正式端口**的旧作业 `2153008`，仍然成立；不要用一个 harness PASS 改写正式
> 矩阵，也不要把 `drun` 的两节点成功外推成 SSH/`plm_rsh` 可用。
>
> **23:30 更新**：§0.1.7 的四个 harness 变体 PASS 同样不改变上表：第 1 条的 c128 指
> **正式端口行**，仍无结果（harness 两节点 c128 另记 §0.1.7）；第 2 条正式端口
> `2153008` 仍 PENDING；第 4 条 32r Legacy-MG 仍 **CACHE-MISSING**。

**未发布**：已发布仍是 `0.1.17` / tag `test37`；下一次发布只会在十轮复审全过后作为
`0.1.18` / `test38`。本轮不改版本、不 commit、不 push、不打 tag。

### 0.1.5 收尾状态（2026-11-17 20:00 快照）

- `2153005`（c128 单节点 32r）已按用户收尾指令停止；最终 `FAILED`，从未启动，无 c128 样本。
- 20:00 快照时账号无活跃作业、两节点未提交；22:15 已用 `drun` 提交两节点 harness 作业
  并 PASS（§0.1.6）。
- 本机最终检查：`--check` 20/20、`--quick` rc 0、适配器测试 127 passed、`git diff --check` clean。
- 仍未发布：正式端口两节点行、c128、32r dslash、CSV/Excel 未闭环；`0.1.17` / `test37` 保持。

### 0.1.6 两节点 c64 harness PASS 与 drun 经验（2026-11-17 22:15）

这是本轮第一条**两节点数值 PASS**（后续四个变体见 §0.1.7），但它是 **KUPL harness**
（`pyqcu/testing/simple/kupl_multigrid.py`）的结果，不是 `pyqcu.test --l48-two-node`
的正式矩阵行；正式矩阵仍为 `UNVERIFIED`，不得把两者混写。

| 项目 | 值 | 状态 |
|---|---|---|
| 作业 | `2155536`，`l48_2n32r_l1r0_drun`，`cn[19544,19546]`，`SUCCEEDED` | **VERIFIED**（22:15:21 → 22:16:53） |
| 资源 | 2 节点 / 32 rank / 16 rank/node / 18 core/rank | **VERIFIED** |
| 几何 | `48×48×48×96`，`2×2×2×4`，local `24×24×24×24` | **VERIFIED** |
| 求解器 | c64，Clover parity-Schur BiCGStab，levels=1，restart=0，max_iter=50，atol=1e-4 | **VERIFIED** |
| 数值 | 11 it；`final_true_residual=7.12609211762054e-05 < 1e-4`；`converged=true`；`status=PASS` | **VERIFIED** |
| 时钟 | solver `wall_seconds=1.1778152959886938 s`；`input_prebind=17.433025 s`；gauge `9.392124 s`；source `4.551332 s`；residual `2.590815 s` | **VERIFIED** |
| 内存 | `max_rss_kib=2382824`（约 `2.27 GiB`），`OMP_NUM_THREADS=18` | **VERIFIED** |
| 报告 | `data/l48_2node_20261117/drun_run/reports/l48_2n32r_l1r0.json` | **VERIFIED** |

启动路径（这是本轮最有价值的经验）：

```bash
# 作业内已经由 CCS_ALLOC_FILE 生成 32 行 rankfile（每 rank 18 core）
export CCS_DRUN_RANK_FILE=$OUT/logs/$JOBTAG-rankfile
/opt/batch/agent/tools/drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v -- \
  python3 -B pyqcu/testing/simple/kupl_multigrid.py --lat 48 48 48 96 \
  --grid 2 2 2 4 ...
```

- `drun` 警告 `-N/-nn/-rpn/-nl/-R/--cpu-bind/--distribution` 在设置了
  `CCS_DRUN_RANK_FILE` 时被忽略；**实际用的是 rankfile**。不要因为告警就以为只起了
  1 个 rank，必须用报告里的 `mpi_size=32` / `grid` / `local_lattice` 复核。
- 不要再用 `plm_rsh` 作为本账号的两节点启动器。2026-11-17 22:11 生成的
  `~/.ssh/id_ed25519_pyqcu` 已被计算节点 `sshd` **接受**（`Server accepts key`），
  但会话随后 `Connection closed`；这是计算节点不允许直接 shell，不是 key 或
  known_hosts 问题。主机键可通过 `accept-new` 正常写入。
- `plm_donau` 仍在该环境 `query()` 拒绝；`--mca plm donau` 不得恢复。`drun` 是
  已验证的 CCS-native 路径，`drun` 只存在于计算节点（登录节点没有）。
- 这是**单样本** harness 结果；c128 / 两节点 Legacy-MG 的 harness 结果另见 §0.1.7，
  但不能据此宣称正式端口矩阵或 scaling 已通过。CI/发布门仍以正式 `pyqcu.test`
  矩阵和独立残差证据为准。

### 0.1.7 两节点 KUPL harness 变体矩阵（2026-11-17 23:30 归档）

本节只登记 **KUPL harness**（`pyqcu/testing/simple/kupl_multigrid.py`）的两节点实测，
**不是** `pyqcu.test --l48-two-node` 矩阵行。远端覆盖树实测
`python3 -m pyqcu.test --list --l48-two-node` →
`unrecognized arguments: --l48-two-node`、`RC=2`；`pyqcu/test/l48.py` 不存在，
`pyqcu/test/__main__.py` 与 `common.py` 的 `grep -c l48` 均为 `0`。因此**正式矩阵
五行仍是 `UNVERIFIED`**，本节任何数字都不得写进正式矩阵，也不得当成正式端口证据。

PASS 规则（未放宽）：报告 `status == "PASS"` **且** `final_true_residual < 1e-4`，
残差由独立的分布式 NumPy Wilson-Clover 全格点重构给出。四个作业**串行提交**
（每次 `dsub -s` 前 `djob -u nsls_yangyb` 为空），提交形状
`dsub -s data/l48_variants_20261118/<wrapper>.sh`，作业内走
`CCS_DRUN_RANK_FILE` + `drun -mpi pmix`（与 §0.1.6 同一条已验证路径）。报告路径
相对 `/home/share/nsls_yangyb/PyQCU-main/data/l48_variants_20261118/`。

| # | 变体（harness） | 作业 | execNodes | dtype | ranks / 节点（core/rank） | grid / local | 报告 |
|---:|---|---|---|---|---|---|---|
| 1 | c128 BiCGStab，32r | `2155785` | `cn[12044,12054]` | `complex128` | 32 / 2（18） | `2 2 2 4` / `24x24x24x24` | `reports/v1_c128_2n32r_l1r0.json` |
| 2 | c64 BiCGStab，16r | `2155830` | `cn[12041,12067]` | `complex64` | 16 / 2（36） | `2 2 2 2` / `24x24x24x48` | `reports/v2_c64_2n16r_l1r0.json` |
| 3 | c64 BiCGStab，64r | `2155868` | `cn[12041,12067]` | `complex64` | 64 / 2（9） | `2 2 4 4` / `24x24x12x24` | `reports/v3_c64_2n64r_l1r0.json` |
| 4 | c64 Legacy-MG，16r cache-load | `2155888` | `cn[12054,12067]` | `complex64` | 16 / 2（36） | `2 2 2 2` / `24x24x24x48` | `reports/v4_c64_2n16r_legacy_mg_l2r5.json` |

| 作业 | it | `final_true_residual` | converged | solver wall (s) | prebind (s) | mpirun / job wall (s) | max RSS (KiB → GiB) |
|---:|---|---|---|---:|---:|---|---|
| `2155785` | 16 | `7.466169937510113e-06` | true | `3.9853145200759172` | `17.786567784147337` | 90 / 94 | `3415116` → 3.26 |
| `2155830` | 11 | `7.125979321648942e-05` | true | `3.0647244779393077` | `8.75744577893056` | 56 / 60 | `4315972` → 4.12 |
| `2155868` | 11 | `7.125993463624718e-05` | true | `2.797133596148342` | `34.120542434975505` | 81 / 84 | `1386252` → 1.32 |
| `2155888` | `null`（MG） | `7.125979321648942e-05` | true | `2.129854183178395` | `9.21071798610501` | 79 / 82 | `8487320` → 8.10 |

- **求解器与开关**：四个作业都是 `applyCloverMultigridQcu` + `--strict-acceptance`；
  变体 1–3 为 levels=1 / restart=0 / max_iter=50；变体 4 为 levels=2 / restart=5 /
  max_iter=30 / `dof 12 2` / `mg-grid 2 2 2 2`，复用既有 16r cache
  （`data/l48_validation_20261117/caches/l48_16r_mg_l2r5`，`verify_status=0`，
  `builder_seconds=21.420243138913065 s`，未重建）。全部 `jobExitCode=0`、
  `execNodeCnt=2`，`OMP_NUM_THREADS` 等于上表 core/rank。
- **输入**：变体 1 用 c128 `gauge_..._c128.h5` / `source_..._c128.h5`；变体 2–4 用
  c64 对。dtype 取自 `dataset.dtype`，不取文件名；实测 SHA256 见远端
  `data/l48_variants_20261118_shas.txt`（`gauge` c128 `b931b0cf…`、c64 `c8e8df3b…`；
  `source` c128 `3e07d910…`、c64 `e4dc2a4c…`）。`case_manifest.json` 仍混记 c64
  文件名与 c128 size/SHA，**不得**作为 dtype 证据。
- **MG 解释边界（必读）**：变体 4 的 `final_true_residual` 与同几何变体 2
  （levels=1）**逐位相同**（`7.125979321648942e-05`），且
  `atol=1e-4 × coarse_tol_factor=1e4 ⇒ coarse_atol=1.0`，是“fine BiCGStab 回退”的
  已知签名。因此 `2155888` 只证明**分布式 Legacy-MG cache 加载 + solve 路径**
  （identity 校验、hyperslab 读、`verify_status=0`、独立残差），**不证明粗修正生效、
  不证明 MG 加速**。32 rank Legacy-MG（grid `2 2 2 4`）无同 identity cache →
  **DEFERRED / CACHE-MISSING**，未重建、未改名。
- **不是 scaling**：四个变体各 1 样本，节点对、ranks/node、core/rank 均不同；不计算
  加速比，也不把 c64 的 16/32/64 rank 点（32r 见 §0.1.6）当扩展性曲线。
- **诊断作业**：`2155899`（变体 4 + `KUPL_REQUIRE_MG_CORRECTION=1`）归档时仍
  `PENDING`、无 `execNodes`，不携带数值信息，不作为结果。
- **时钟口径**：`job wall` 取调度器 `jobWallclockDuration`；`mpirun` 段为作业内计时
  块；`solver wall` 只覆盖一次 solver 调用，不能与 `mpirun`/job wall 互换相加。

### 0.1.8 明确跳过、留给下一次任务的部分（2026-11-18）

以下项目在 0.1.18 发布中**不声称完成**；它们保留为下一次任务，不阻塞本次十轮发布
复查，但在 release notes / 日志中必须原样保留：

| 项目 | 当前状态 | 原因 / 下次入口 |
|---|---|---|
| 正式 `pyqcu.test --l48-two-node` 两节点执行 | **DEFERRED / UNVERIFIED** | 远端 overlay 缺 `--l48-two-node`；本地 launcher/ledger/native 代码已有，但没有在真实 32-rank 作业中用正式端口跑出 PASS。需同步 overlay 后重跑。 |
| 正式 L48 矩阵的 c128 / 16 / 64 rank 变体 | **DEFERRED** | §0.1.7 的四个 PASS 全部是 **KUPL harness**，不是正式矩阵行；不得把 harness 结果填进矩阵 `evidence`。 |
| 32 rank / 两节点 Legacy-MG | **DEFERRED / CACHE-MISSING** | 没有 identity 匹配的 32r cache；本轮不重建。16r cache-load PASS 的粗修正未激活，不能当 MG 加速证据。 |
| CSV F1/F2 | **DEFERRED** | 100 列 CSV / 21 sheet 导出已端到端验证过，但 `--mode run` 的重复 `run_id`（F1）与 `PYQCU_TEST_LEDGER_EXPORT` typo 的 usage error（F2）尚未修复；本次不把它们写成 parity 完成。 |
| `2155899` MG fallback 诊断 | **PENDING** | 无 `execNodes`/报告，不作为结果；下次批次再判。 |
| 正式端口多 rank `/tmp` 实跑与独立残差 | **DEFERRED** | 只完成了单元/小格点测试，未完成真实两节点正式端口作业。 |

本次发布只声明：基于已提交的代码与 sdist 的门禁、打包、安装和已验证的 harness
结果；不声明上表项目通过。

## 1. 目标与非目标

### 1.1 目标

1. 在正式端口 `pyqcu.test` 的 full mode 中支持真实 L48 LIME/HDF5 输入，而不是只把
   `--gauge-file`/`--source-file` 记录为证据。
2. 让一个 row 真正执行 KUPL 原生 `dslash`/`solve`/`mg`，由 harness 决定 PASS/FAIL，
   并记录独立真残差、迭代数、输入哈希和 MPI 几何。
3. 用 `dsub -nn 2` 申请两个计算节点，在一个作业内用一条 `mpirun` 跨两节点运行，
   不再把“单节点整节点分配 + 内部 mpirun”的假设写死。
4. 复用 `pyqcu.test` 的 `rows/<run_id>.json`、`reports/rows.csv`、`--resume`、
   `--skip-failed`、`--run-label` 与 `--mode validate` 台账契约。
5. 保留可回退边界：单节点先跑通，再验证两节点；任何未验证的组合必须明确标
   `SKIP`/`TODO`，不得静默回退到小格点或随机 gauge。

### 1.2 非目标

- 不把 `pyqcu.test --quick` 的 `8x8x8x16` 正确性门改造成性能门。
- 不在本设计里修改 `pyqcu/testing/kupl/test_kunpeng_all.py` 或 C++/Python 核心代码。
- 不承诺 L48 性能、强扩展或相对加速比；没有同窗口、同节点的数据就不计算加速比。
- 不假设远端 SSH、数据盘、HDF5 MPI 或调度器行为已经可用；这些全部是下面的开放问题。

## 2. 已验证基线

### 2.1 本地 L48 输入

以下均为本机（WSL2）实测，不包含远端结论：

| 项目 | 事实 | 证据 |
|---|---|---|
| 路径 | `/home/zhangxin/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime` | `~/data -> /mnt/f/Data` 符号链接；`stat` 可读 |
| 文件大小 | `6,115,435,952 B`（约 5.70 GiB） | `stat` |
| LIME 魔数 | `45 67 89 ab`，version `0x00018000` | 文件头 |
| LIME 维度 | `dims = 48 48 48 96` | `pyqcu/testing/simple/_lime.py::lime_metadata` |
| LIME 精度 | `precision = D`（complex128），`datacount = 4`，`colors = 3` | `scidac-private-record-xml` |
| LIME `clovCoeff` | `1.160920225783` | `scidac-record-xml` |
| LIME `Mass` | `-0.2770` | `scidac-record-xml` |
| LIME payload | `6,115,295,232 B` | `ildg-binary-data` record length |
| 未做（本机） | 未做全量 `read_lime_gauge`、未校验 SciDAC checksum、未转 HDF5、未求解 | 本任务未执行；**远端**对应进展见 §2.4.4 |

L48 `c64` 的 canonical 尺寸按 `48^3 x 96 = 10,616,832` 个 site 计算：

| 对象 | 布局 | 逻辑字节 |
|---|---|---:|
| gauge c64 | `[3,3,4,48,48,48,96]` | `3,057,647,616 B`（约 2.85 GiB） |
| source c64 | `[4,3,48,48,48,96]` | `1,019,215,872 B`（约 0.95 GiB） |
| Legacy-MG cache（逻辑） | 按 `dof=12 12`、`mg-grid=2 2 2 2` 的历史预算，见 `docs/qcu_quda_solver_dcu_l48_20261029.md` | `18,728,091,648 B`（约 17.44 GiB） |

### 2.2 现有两条入口的分工

| 入口 | 当前定位 | 关键限制 |
|---|---|---|
| `pyqcu/testing/kupl/test_kunpeng_all.py` | 非正式 KUPL 鲲鹏全矩阵驱动；`list/prepare/prewarm/solve/merge/run/validate` 七模 | 只认 `L24/L32`；只读 HDF5；无 LIME；`--mode validate` 是 2 rank、`8x8x8x16` 小格冒烟；自己启动 `mpirun` |
| `pyqcu.test --mode ...` | 正式端口的 full-mode 工作台账；随包发行；stdlib-only 导入 | 19 个 tier 0 case 的编排层；`--gauge-file`/`--source-file` 原先只做 HDF5 证据；`driver:run_row` 与 `L48` preset 由并发批次补上（§2.4.7，未提交） |

具体静态依据（**设计时**的仓库状态；并发实现见 §2.4.7）：

- `test_kunpeng_all.py` 的 `LATTICES = {"L24": ..., "L32": ...}` 在
  [`pyqcu/testing/kupl/test_kunpeng_all.py:87`](../pyqcu/testing/kupl/test_kunpeng_all.py)；
  全文没有 `lime` 分支，输入只经 `h5py` 的 `h5_readable`/`prewarm_h5`
  （[`:310`](../pyqcu/testing/kupl/test_kunpeng_all.py)）。
- 该驱动的 `mode_validate` 在
  [`pyqcu/testing/kupl/test_kunpeng_all.py:1749`](../pyqcu/testing/kupl/test_kunpeng_all.py)
  把 `--lat` 强制为 `8 8 8 16`、`--ranks 2`、`--shm 0`，不是正式端口的台账校验。
- `submit_kunpeng_all.sh` 把 `#DSUB -nn 1` 写死，并把 `-N $RANKS -rpn $RANKS`
  作为单节点整节点分配
  （[`submit_kunpeng_all.sh:137`](../pyqcu/testing/kupl/submit_kunpeng_all.sh)）；
  `ranks*threads > 608` 直接拒绝（[`:94-97`](../pyqcu/testing/kupl/submit_kunpeng_all.sh)）。
- 设计时正式端口的 preset 只有 `L24=(24,24,24,64)` 与 `L32=(32,32,32,64)`；
  2026-11-17 14:0x 的并发实现把 `L48=(48,48,48,96)` 加进了 `LATTICE_PRESETS`
  （[`pyqcu/test/common.py:1545-1549`](../pyqcu/test/common.py)），见 §2.4.7。
- 设计时 full mode 的模块钩子只声明 `driver:("run_row",)` 而仓库没有
  `pyqcu/test/driver.py`，`_solve_row` 查不到 `driver` 时回退到 tier-0 executor
  （[`pyqcu/test/common.py:1571`](../pyqcu/test/common.py)、
  [`pyqcu/test/common.py:2683`](../pyqcu/test/common.py)）；并发实现补了
  `pyqcu/test/driver.py` 与 `common._apply_driver`，见 §2.4.7。
- `pyqcu/test/native.py` 已有 `run_dslash`/`run_solve`/`run_mg` 与
  `run_case`（[`:2539`](../pyqcu/test/native.py)），但设计时 `_build_gauge`/`_build_source`
  只从 `seed`/`source` 生成场（[`:1081`](../pyqcu/test/native.py)、
  [`:1132`](../pyqcu/test/native.py)），没有 HDF5/LIME 输入参数；并发实现新增
  `_hdf5_identity` 等输入身份路径（§2.4.7）。
- 正式端口 README 明确写明 `--ranks`/`--threads`/`--shm` 的“记录 vs 生效”边界，
  以及 `--gauge-file`/`--source-file` 只做证据：
  [`pyqcu/test/README.md:198-206`](../pyqcu/test/README.md)、
  [`:218-241`](../pyqcu/test/README.md)。
- 本机实测 `python -B -m pyqcu.test --list --json -` 给出 **19** 个 tier 0 case，
  格点 `[8,8,8,16]`、c64；`--mode list --lattice-preset L24` 也给出 19 行，
  但标签仍是 `torch/*`、`kupl/*`、`cuda/*` 的 tier 0 case，不是 L24 生产
  `solve/mg` 行。`--mode list --lattice-size 48 48 48 96` 同样只展开这 19 个
  tier 0 case。设计时 `--lattice-preset L48` 被 argparse 拒绝；并发实现在 14:0x 后
  接受 `L48`，但**仍只展开同一批 19 个 tier-0 case**（实测
  `--mode list --lattice-preset L48 --json -` → 19 行，首行 `torch/lattice`、
  `[48,48,48,96]`、c64），没有 L48 生产 `solve/mg` 行。
- 文档计数已统一（2026-11-17 docs 门批次）：根 `AGENTS.md`、`pyqcu/AGENTS.md`、
  `pyqcu/testing/AGENTS.md`、`skills/AGENTS.md` 与
  `skills/pyqcu-docs/references/extracted-guidance.md` 的 8 处旧 tier-0 计数 21 已改为 19，
  与 `--list`（11 torch / 5 kupl / 3 cuda）和 `pyqcu/test/README.md` 一致。

### 2.3 调度器事实（来自用户手册）

[`refer/kunpeng/深圳超算二期用户手册.pdf`](../refer/kunpeng/深圳超算二期用户手册.pdf)
“四、作业提交”与“调度脚本常用参数示例”给出：

| 选项 | 手册语义 |
|---|---|
| `-nn` / `--nnodes` | 作业分布的**节点数**；支持 `-nn "min,max"`，`min=max` 时可写 `-nn x` |
| `-N` / `--replica` | 资源副本数；默认一份资源启动一个任务 |
| `-rpn` / `--replica-per-node` | 每节点最大任务数 |
| 组合 | 未给 `-N` 但同时给 `-nn` 和 `-rpn` 时，`N = nn * rpn`；仅给 `-nn` 时也按手册规则推导 |
| 多节点示例 | `#DSUB -nn 2` + `#DSUB -rpn 32` 表示 2 个节点、每节点 32 份资源、合计 64 份 |
| 作业脚本 | `dsub -s 脚本`；`dattach jobid` 默认连第一个节点，`dattach -en "node" jobid` 连指定节点 |
| 作业查询 | `djob -l jobid`、`djob -ll jobid`；`dkill jobid` |
| MPI 脚本模板 | 手册示例从 `CCS_ALLOC_FILE` 读取节点列表构造 rankfile，并用 `CCS_TASK_REPLICA` 作为 `mpirun -np` 的规模 |

这些只是**手册语义**，不是本账号/本集群的实跑结果。当前仓库文档另有相反的现场
记录：`#DSUB` 头里的 `-N/-rpn/-R/-nl` 可能被忽略，调度参数应写在 `dsub` 命令行；
两节点作业曾在 `mpirun` 阶段因缺少 hostfile 以 `RC=255` 失败
（[`docs/kupl_l24_perf_20261113.md:496-499`](kupl_l24_perf_20261113.md)、
[`docs/kupl_mpi_scaling_20261114.md:32-33`](kupl_mpi_scaling_20261114.md)）。
因此“`#DSUB` 头是否生效”与“`CCS_ALLOC_FILE` 是否存在”必须先在远端验证，见
§2.4.5 与 §9。

### 2.4 远端基线（2026-11-17 已验证）

本节是**实测记录**，不是设计愿望。只写结论与可复跑的命令形状；不写私网地址、密钥或
口令。作业号与节点名保留，可用 `djob -l <id>` 复核。

#### 2.4.1 访问

| 项目 | 事实 | 证据 |
|---|---|---|
| 登录 | 用户内部登录 `login02`，账号 `nsls_yangyb`，经用户提供的内部路由进入 | `hostname; whoami` → `login02` / `nsls_yangyb` |
| `ssh lingsheng` | **DNS 不解析**；本机 `~/.ssh/config`、`/etc/hosts`、`ssh-keygen -F lingsheng` 都没有 `lingsheng` | 本机命令 |
| 前提 | 必须先连上用户的 VPN/内部路由；否则表现为 `No route to host` / 超时 | 首次探针失败、路由就绪后成功 |
| 结论 | “`lingsheng` 不可用”只说明**别名不存在**，不等于集群不可达；改用 `login02` + 用户提供的内部路由，具体地址不入库 | — |

#### 2.4.2 远端代码与运行时

| 项目 | 事实 | 证据 |
|---|---|---|
| 仓库 | `/home/share/nsls_yangyb/PyQCU-main`，**非 git 的 0.1.15 快照**（`git -C` 报 not a git repository） | `git -C` |
| 代码来源 | 先备份为 `pyqcu.bak.20261117`，再把 0.1.17 的 Python 代码覆盖上去；当前是 0.1.15 底座 + 0.1.17 Python 代码的混合树 | 远端目录 |
| 自检 | `python -m pyqcu.test --check` = **18/18 PASS** | 远端命令 |
| 为此修的缺陷 | 0.1.17 的 `pyqcu/test/kupl_qcu.py`、`pyqcu/test/cuda_qcu.py` 用 `source[0, spin, color, *site]` 形式索引（Python 3.11 起才合法），远端 Python 3.10 直接 `SyntaxError`；改为先拼 `site = (0, spin, color, *site)` 再 `source[site]` 后 18/18 | 本仓库同一改动：`git diff pyqcu/test/kupl_qcu.py pyqcu/test/cuda_qcu.py`（未提交）；**远端 0.1.17 覆盖树 18/18**；本机当前工作树因并发未提交改动为 **19/19**（多 1 项检查，不能与远端 18/18 混读） |
| Python | 3.10.9（anaconda 2023.3），aarch64 | `python -V`、`uname -m` |
| numpy | 1.23.5 | `import numpy` |
| h5py | 3.7.0，HDF5 1.10.6，`h5py.get_config().mpi == False` | `h5py.version` / `get_config().mpi` |
| mpi4py | 4.1.2 | `import mpi4py` |
| torch | 1.12.1 | `import torch` |

> `mpi4py`/`torch` 版本与远端 `--check` 的原始 stdout **未归档进仓库**
> （`TODO(原始输出归档)`）。重跑前不要把这三行当成可复核证据；其余行有本仓库同名
> 文件/命令可对照。

#### 2.4.3 HPCKit 26.1.RC1 环境（HMPI + KUPL）

`/work_ssd/software/HPCKit` 下并存 25.1.0/25.2.1/26.1.RC1 等多套；本轮使用
**26.1.RC1**。`source ./env.sh` 之后再叠加下表（`<HK>` = `/work_ssd/software/HPCKit/26.1.RC1`）：

| 变量 | 值 |
|---|---|
| `PATH` | `<HK>/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:/opt/batch/cli/bin:$PATH` |
| `LD_LIBRARY_PATH` | `<HK>/hmpi/bisheng/release/hmpi/lib:<HK>/compiler/gcc/lib64:<HK>/compiler/bisheng/lib:/work_ssd/software/HPCKit/alllib:$LD_LIBRARY_PATH` |
| `OPAL_PREFIX` | `<HK>/hmpi/bisheng/release/hmpi` |
| `PYTHONPATH` | 仓库根（`$PWD`） |

四段库路径各有分工：`hmpi/lib` 是 HMPI 运行库；`compiler/gcc/lib64` 提供
`GLIBCXX_3.4.29`（系统 `libstdc++` 版本不足）；`compiler/bisheng/lib` 提供 `libomp`
（KUPL 的 OpenMP 运行库）；`alllib` 补齐其余第三方库。缺 `OPAL_PREFIX` 时 `mpirun`
的选项会被忽略或报错。

| 项目 | 结果 |
|---|---|
| `mpirun --version` | HMPI（Open MPI 4.1.6rc4） |
| `mpirun -np 1 --bind-to none python3 -m pyqcu.test --quick --backend kupl --only ...` | 可启动；`kupl/abi/symbols` 与 `kupl/dslash/wilson` 均 PASS |
| 未验证 | 2 rank 及以上：formal KUPL dslash/solve 用例声明 `mpi_single_rank`，2 rank 会 `SKIP`；多 rank KUPL 需单独入口 |

HMPI 的 `-np 1 --bind-to none` 是本轮唯一**已实跑**的 MPI 形态；它只证明“单 rank 能
在 HMPI 下启动 KUPL 原生路径”，不证明跨节点。

#### 2.4.4 L48 输入转换（c128 HDF5 已验证；c64 已用于求解）

| 项目 | 事实 | 证据 |
|---|---|---|
| 远端 L48 | `~/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`，`6,115,435,952 B`；与本地同大小、同前 4 KiB SHA256 | `stat`、文件头 |
| 输出目录 | `~/PyQCU-main/data/l48_canonical_20261117` | `ls` |
| 转换入口 | `pyqcu/testing/simple/prepare_case.py --lime <L48> --lat 48 48 48 96 --mass -0.277 --csw 1.160920226 --seed 42 --out-dir <out>` | 远端命令 |
| 耗时 | **41 s**（单次，读 LIME + 写 HDF5） | 本批次远端执行记录；原始 stdout 未归档 → `TODO` |
| **产物 dtype** | `prepare_case.py` 写出的 `gauge_48x48x48x96_c64.h5` **实际是 complex128**，`6,115,301,376 B`；源 LIME 本身是 complex128（`precision D`） | `h5py` 读 `dataset.dtype` |
| 结论 | 文件名**误标**，与 [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) 记录的 L24/L32 同类缺陷一致；dtype 只能以 `dataset.dtype` 为准 | 同上 |
| source 侧 dtype | `prepare_case.py` 的 source 生成器按 `numpy PCG64 normal float64 → astype(complex64)` 写出，**source 是 complex64**（[`prepare_case.py:85-101`](../pyqcu/testing/simple/prepare_case.py)） | 代码与产物路径；远端未单独回读归档 → 原始 stdout 仍 TODO |
| c64 转换（TODO，待验证） | 先把误标文件改名为 `..._c128.h5`，再按 t-slab 转 `complex64`，写 `source_dtype`/`cast_to` 属性，`os.replace` 原子落盘 | 远端转换脚本形状；**尚未完成回读** |
| c64 门 | 只有 `pyqcu.test.io.describe(..., role="gauge")` 回读 `dtype == complex64`、shape `[3,3,4,48,48,48,96]`、`lattice == [48,48,48,96]` 才可写 `c64`；**在此之前不得声称已有 c64 gauge** | [`pyqcu/test/io.py`](../pyqcu/test/io.py) |

> 截至 2026-11-17 14:16：c64 cast **尚未完成**、`io.describe` **尚未回读**，因此 L48
> gauge 的 c64 状态仍是 **TODO/待验证**；当前唯一可引用的 dtype 事实是
> “`*_c64.h5` 实为 complex128”。source 侧代码默认写 complex64，但远端未单独归档
> 回读结果，同样只能标“待复核”。
>
> **2026-11-17 结果批次补充**：c64 输入已被真实求解使用（§0.1 第 1–3 行），因此
> “c64 未确认”不再成立；但 c64 cast 与 `io.describe` 回读的**原始 stdout 仍未归档**，
> 引用时只能标 **C 级**，不得把“能跑 c64”当成“已归档 dtype 证据”。

#### 2.4.5 两节点调度探测（PENDING，未通过）

| 项目 | 事实 |
|---|---|
| 探测作业 | `2152277`（两节点探测）、`2153008`（两节点求解） |
| 状态 | 两者均 **PENDING**：`2152277` 的 `mainStateReasonMsg` 为 `No available node can fit ... execution nodes preference`；`2153008` 的请求语法正确，卡在队列/限额。`2153008` 的 launcher **未传 rankfile**（§5.3） |
| 能证明什么 | 只证明“申请被受理、但在节点偏好下排不到节点”；**不证明两节点可跑**，不得写成两节点成功 |
| 同时刻队列 | `q_hpcapp` OPEN/ACTIVE，127 running / 211 pending；`dinfo` 仅 7 个 `OK` 节点，其余 `CLOSE_FULL`/`CLOSE_ADMIN`/`UNAVAILABLE` |
| 手册语义（复核） | `-nn` 节点数；`-N` **总**资源副本数；`-rpn` 每节点最大副本数；不给 `-N` 时 `N = nn(min) × rpn` |
| rankfile | 手册与 HMPI 作业脚本从 `CCS_ALLOC_FILE`（每行 `host total_cores rank_count`）生成 rankfile，`mpirun -np ${CCS_TASK_REPLICA} --rankfile ...` |
| 提交方式 | `#DSUB` 头只在 `dsub -s <script>` 时被解析；`-nl` 是偏好，节点忙时会 pend |
| 仍缺 | 计算节点免密（`login02` → `cn*` 仍 `Permission denied (publickey)`）；`dsub -nn 2` 的实际 `execNodes`；`djob -l 2152277` 原始输出归档 |

#### 2.4.6 新增需求：CSV 与 Excel 台账（parity 未通过）

正式端口 full mode 现在写 `<reports>/rows.csv`，列集合是自身的 30 列
`ROW_FIELDS`（[`pyqcu/test/common.py:1586`](../pyqcu/test/common.py)）。新要求是**改用**
旧驱动的公开契约，让下游表格与 `awk` 管道不必改：

| 契约 | 位置 |
|---|---|
| 100 列 `CSV_COLUMNS` | [`pyqcu/testing/kupl/test_kunpeng_all.py:95`](../pyqcu/testing/kupl/test_kunpeng_all.py) |
| `upgrade_csv_schema` / `append_csv` / `merge_rows` | 同文件 `:1471` / `:1507` / `:1529` |
| 汇总工作簿（21 sheet，openpyxl 3.1.5 生成） | `data/kunpeng_all_20261115/kupl_kunpeng_performance_20261115.xlsx` |

实现（**未跟踪、未接线、未审计**，见 §9.6）：

| 文件 | 作用 |
|---|---|
| `pyqcu/test/ledger_csv.py` | 冻结的 100 列名单 + 与参考一致的单元格序列化/CRLF/表头升级/行序；formal 端口观测不到的列写空，并在 `COLUMN_RULES` 注明原因，不从旧快照编造取值 |
| `pyqcu/test/ledger_excel.py` | 不依赖第三方库的 SpreadsheetML 写入器，复现 21 sheet、表对象、状态条件着色与两张汇总图 |
| `pyqcu/testing/python/test_ledger_parity.py` | 与 `CSV_COLUMNS`/工作簿逐项对照的 parity 测试 |

状态：这三个文件尚未被 `pyqcu/test/__main__.py` 或 `common.py` 引用，`--csv` 仍写
`ROW_FIELDS`；parity 测试尚未在最终树上跑完 → **不发 PASS**。

**parity 是硬性要求**：100 列必须逐列同序、21 sheet 必须逐 sheet 与参考工作簿一致；
在此之前只能说“实现中”，不得声称已经替换旧驱动的 CSV/Excel 契约。该缺口登记为
[`pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md) 的 P1，
状态 **未修复（NOT FIXED）**。

#### 2.4.7 并发实现状态（截至 2026-11-17 14:16，非本任务产出）

写这份文档时有另一个批次在改 `pyqcu/test/**`。**这些改动不是本任务的产出，未提交、
未审计、也没有在远端跑过**；这里只记录它们改变了哪些“原缺口”的表述，避免读者把
设计时的静态结论当成现状。

| 文件/改动 | 状态 | 对本文的影响 |
|---|---|---|
| `pyqcu/test/driver.py`（新文件，observation-only `run_row`） | 未跟踪 | §2.2/§6 的“没有 driver”改为“设计时缺失，现已有并发实现” |
| `pyqcu/test/common.py`：`LATTICE_PRESETS` 加 `L48=(48,48,48,96)`、`_apply_driver` | 已修改 | `--lattice-preset L48` 不再被 argparse 拒绝；但 `--mode list` 仍只展开 17 个 tier-0 case（CG 退役） |
| `pyqcu/test/native.py`：`_hdf5_identity` 等 HDF5 输入身份 | 已修改 | `--gauge-file`/`--source-file` 的读取/校验/哈希下沉到 native 输入层（未审计） |
| `pyqcu/test/__main__.py` | 已修改 | full-mode 选项与帮助文本变化 |
| `pyqcu/test/selfcheck.py` | 已修改 | 本机 `--check` 从 18 项变 **20 项**（并发批次新增检查）；远端 0.1.17 覆盖树仍是 18/18，两个分母不能混读 |
| `pyqcu/testing/python/test_pyqcu_test_driver.py`、`test_pyqcu_test_syntax_floor.py` | 未跟踪 | 并发批次为 driver 与 Python 3.10 语法下限新增的测试；未审计、未在远端跑过 |
| `pyqcu/test/ledger_csv.py`、`ledger_excel.py`、`pyqcu/testing/python/test_ledger_parity.py` | 未跟踪 | CSV/Excel 仍未接线到 CLI |

结论更新（2026-11-17 结果批次 + 23:30 变体批次）：单节点 c64 已有两条实测结果
（§0.1：BiCGStab 32 rank `1.6646 s`、Legacy-MG 16 rank）；两节点已有 **KUPL harness**
证据（§0.1.6 的 c64 32r，§0.1.7 的 c128 32r / c64 16r / c64 64r / c64 Legacy-MG 16r
cache-load），但**正式端口两节点仍无任何数值证据**，正式 c128/CG/32r dslash-clover
同样没有；§8 的验收门（尤其第 2 条启动门的两节点 `execNodes` 与正式 flag）仍未
通过。并发实现本身也**不能**降低任何 P0/P1：它未提交、未审计、未在远端运行，且
`driver.run_row` 仍明确不做 `mpirun`/rankfile 启动（§5.3）。P0/P1 的逐条状态见
[`pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md)（全部
**未修复**）。

### 2.5 既有 L48 容量与 LIME 转换材料（只引用，不复制）

| 文档 | 与本任务的关系 | 使用边界 |
|---|---|---|
| [`qcu_quda_solver_dcu_l48_20261029.md`](qcu_quda_solver_dcu_l48_20261029.md) | 唯一的 L48 容量预算（`dof=12 12`、`mg-grid=2 2 2 2` → `18,728,091,648 B` 逻辑 payload）与 DCU L48 暂停记录 | 是**容量预算**，不是 LX2 远端证据；其 `BLOCKED-BY-USER` 与本轮远端状态无关 |
| [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) | 同名 `*_c64.h5` 实为 complex128 的既往根因、`h5_readable` 读取闸门、写侧缺 `fsync`/回读的缺口 | 本轮 L48 误标属同一类；dtype 一律以 `dataset.dtype` 为准 |
| [`kupl_kunpeng_progress_20261110.md`](kupl_kunpeng_progress_20261110.md) | `prepare_case.py` 的 source 生成式（PCG64 normal → c64）、`seed=42`、layout 元数据口径 | 说明“gauge 真实、source 自造”，两者无物理关联 |
| [`kupl_only_deployment.md`](kupl_only_deployment.md) | `prepare_case.py --lime ... --out-dir ...` 的标准调用形状 | CLI 形状可复用；L48 参数（`--lat 48 48 48 96`、`--csw 1.160920226`）是本轮新增 |
| [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) | LIME 在源码层的只读读取点（`_common._default_lime()`、`prepare_case.py` → `_lime.read_lime_gauge`，`open("rb")`） | 证明转换不会改写原 LIME |

上述表格与数字以原文档为准，这里不复述。

## 3. 目标架构

### 3.1 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ CLI: python -m pyqcu.test --mode ...                         │
│   --lattice-preset L48 | --lattice-size 48 48 48 96          │
│   --gauge-lime <file> | --gauge-file <h5>                    │
│   --source-file <h5> --ranks 32 --threads 18 --no-shm        │
└──────────────────────────────┬───────────────────────────────┘
                               │  full-mode rows
┌──────────────────────────────▼───────────────────────────────┐
│ pyqcu.test.common.run_mode / _solve_row                     │
│   rows/<run_id>.json, reports/rows.csv, resume, validate     │
└──────────────────────────────┬───────────────────────────────┘
                               │  mode_hook("driver", "run_row")
┌──────────────────────────────▼───────────────────────────────┐
│ pyqcu.test.driver.run_row  (设计时缺失；并发实现已补，见 §2.4.7)  │
│   - 解析 row 的输入/几何/线程/shm                             │
│   - 调 native.run_case 或 MPI 子进程                          │
│   - 只返回 observations，不自判 PASS/FAIL                     │
└──────────────────────────────┬───────────────────────────────┘
                               │  one Python process = one MPI rank
┌──────────────────────────────▼───────────────────────────────┐
│ pyqcu.test.native.run_dslash/run_solve/run_mg                 │
│   - 读取 rank-local gauge/source slab                         │
│   - 调 KUPL ABI / CUDA bridge                                 │
│   - 独立全格点真残差                                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ 输入层: LIME -> canonical HDF5 -> pyqcu.test.io               │
│   pyqcu/testing/simple/_lime.py::read_lime_gauge              │
│   pyqcu/testing/simple/prepare_case.py::prepare_gauge/source  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 各层职责

| 层 | 负责 | 不负责 |
|---|---|---|
| 输入层 | LIME 魔数/元数据/校验和、canonical `xyzt` HDF5、输入 SHA256、shape/dtype/lattice 回读 | 不生成“看起来像生产”的随机/单位组态来顶替缺失输入 |
| row 规划 | L48 preset、`--sweep`/矩阵展开、`run_id` 摘要、CSV 字段 | 不在 row 里写死某台节点的 hostname |
| `driver.run_row` | 把 row 变成一次 `mpirun` 或一次本地原生调用；返回观测值 | 不决定 PASS/FAIL；不伪造 `solutions/*.h5` |
| `native.run_*` | 一进程一 rank；rank-local slab；ABI 调用；独立残差 | 不读完整的全局 LIME/HDF5 到每个 rank（必须按 rank 切片或 MPIO） |
| 调度层 | `dsub` 两节点分配、`CCS_ALLOC_FILE` → rankfile、作业日志 | 不在每个 task 里重复启动整个驱动 |
| 台账层 | `rows/` 事实源、`reports/rows.csv` 派生视图、`--resume`、`--mode validate` | 不把“空台账”当 PASS |

### 3.3 一次 L48 两节点运行的数据流

1. `--mode prepare`：校验 LIME 魔数与元数据，生成/复用 canonical HDF5 gauge/source，
   写入 `reports/prepare.json` 与 `caches/index.json`；此时不做 solve。
2. `--mode prewarm`：完整读一遍已请求输入并记录 `sha256`/字节数；L48 全量读取约
   6.1 GB LIME 或 3.1 GB c64 HDF5，耗时必须在报告中单独记录。
3. `dsub`：申请两个节点（目标形态 `-nn 2`），由作业脚本从 `CCS_ALLOC_FILE` 生成
   rankfile，并在一个 task 内启动一条 `mpirun`。
4. `--mode solve`：每个 row 由 `driver.run_row` 在 `mpirun` 的 rank 上执行；
   每个 rank 只加载自己的 gauge/source slab。
5. `--mode merge` / `--mode validate`：从 `rows/*.json` 重建 CSV，并交叉校验
   row/report/CSV；这一步与数值求解分离。

## 4. L48 输入契约

### 4.1 LIME → canonical HDF5

优先复用现成入口，不另写第二套 LIME reader：

- LIME 读取：`pyqcu/testing/simple/_lime.py::read_lime_gauge`
  （[`_lime.py:90`](../pyqcu/testing/simple/_lime.py)）。它按 SCIDAC record
  读取，支持 `scidac-private-file-xml`/`scidac-file-xml` 与
  `scidac-private-record-xml`/`scidac-record-xml`，并在有 `scidac-checksum` 时校验。
- LIME → QCU HDF5：`pyqcu/testing/simple/prepare_case.py::prepare_gauge`
  （[`prepare_case.py:51`](../pyqcu/testing/simple/prepare_case.py)）与
  `prepare_source`（[`:85`](../pyqcu/testing/simple/prepare_case.py)）。
  它写 `data` dataset、`format="pyqcu.simple.v1"`、`lattice`/`mass`/`csw`/`seed`
  等属性；full gauge 是 `[3,3,4,x,y,z,t]`，full source 是 `[4,3,x,y,z,t]`
  （见 `pyqcu/testing/simple/_common.py::_attrs` 与 `write_h5`）。
- 正式端口读取：`pyqcu.test.io.load_gauge`/`load_source`/`describe` 接受
  `complex64`/`complex128`、full 或 parity-compressed canonical 布局；full 文件
  可省略 `layout` 属性，parity-compressed 文件必须带 `layout`。因此
  `prepare_case.py` 的输出原则上可被正式端口读取，但必须先在远端用
  `pyqcu.test.io.describe` 做一次属性/形状验证；这是 `TODO`，不是已证事实。

**硬约束**：

- 不允许用文件名判断 dtype（2026-11-15 的 `gauge_file_dtype` 文件名误标已经证明
  这一点，见 `docs/kupl_hdf5_source_gauge_rootcause_20261115.md`）。
- 不允许“文件存在就复用”；必须校验 `lattice`、`dtype`、`mass`、`csw`、`role`、
  `layout` 与 payload 回读。
- 不允许 LIME 读取失败后静默降级为 identity gauge 或随机源；这必须记 FAIL。

### 4.2 输入身份字段

每个 L48 row 至少记录：

```text
gauge_origin = "lime" | "hdf5"
gauge_path, gauge_bytes, gauge_sha256
gauge_lattice = [48,48,48,96]
gauge_dtype   = "c64" | "c128"        # dataset dtype，不是文件名
gauge_format  = "LIME/SciDAC" | "QCU HDF5" | "QDP HDF5"
source_origin = "ones" | "point" | "random" | "hdf5"
source_path, source_bytes, source_sha256
source_seed, source_site, source_spin, source_color
mass, csw, kappa
```

## 5. 双节点执行模型

### 5.1 L48 几何约束

`pyqcu.test.native` 与 `pyqcu/testing/simple/mpi/_common.py::local_geometry`
要求每个 rank 的局部格点在四个方向都是**偶数**。L48 的 `48^3 x 96` 下：

| ranks | balanced 示例 | 局部格点 | 备注 |
|---:|---|---|---|
| 16 | `2x2x2x2` | `24x24x24x24` | 可用；`t-finest` 为 `1x1x1x16`，`96/16=6` 偶数 |
| 32 | `2x2x2x4` | `24x24x24x24` | 可用；`t-finest` 为 `1x1x1x32`，`96/32=3` 为奇数，**不可用** |
| 64 | `2x2x2x8` | `24x24x24x12` | 可用；`t-finest` 的 `96/64` 不整除，**不可用** |

因此 L48 的两节点首选 `ranks=32`、`grid=2 2 2 4`（16 rank/节点），或
`ranks=64`、`grid=2 2 2 8`（32 rank/节点）。任何 `1x1x1xR` 的 t-finest 写法
必须先做整除与偶数校验，不能照搬 L24/L32 的默认 `t-finest` 习惯。

### 5.2 调度分配

目标作业形态（**待远端验证**）：

```bash
# 形态 A：一个作业脚本 + 一个 mpirun，rankfile 来自 CCS_ALLOC_FILE
# 具体 flag 先按手册写，提交前必须用 dsub --help 与本账号核对
dsub -q q_hpcapp -nn 2 -N 32 -rpn 16 -R "cpu=18;mem=8G" -x job -s job.sh

# job.sh 内部的 MPI 启动（示意，不是已跑命令）
mpirun -np 32 --rankfile "$PWD/${CCS_JOB_ID}-rankfile" \
  --map-by ppr:16:node --bind-to none --allow-run-as-root ...
```

要点：

- `-nn 2` 是节点数，`-rpn 16` 是每节点最大任务数，`-N 32` 是总资源副本数；
  三者必须满足 `N = nn * rpn`（或按手册的 min/max 语义）。
- 当前 `submit_kunpeng_all.sh` 的 `-N $RANKS -rpn $RANKS` 只在单节点成立；
  两节点必须显式引入 `--nodes` 与 `--ranks-per-node`，并把 `-nn 1` 改成 `-nn 2`。
- 不能用“每个 task 都跑一遍驱动”的方式凑两节点；驱动的语义是**一个 task 内**
  启动一条 `mpirun`，外层必须是单任务（见 `test_kunpeng_all.py` docstring 与
  `submit_kunpeng_all.sh` 注释）。
- 提交后用 `djob -ll <id>` 核对 `execNodes` 恰好是两个目标节点；不足两节点时
  只允许把该作业标为 `BLOCKED/UNVERIFIED`，不得把单节点结果写成两节点。
- 2026-11-17 的两节点探测 `2152277` 仍是 **PENDING**
  （`No available node can fit ... execution nodes preference`，见 §2.4.5）；
  同日提交的两节点求解 `2153008` **同样 PENDING**（请求语法正确，卡队列/限额）。
  上面的 `-nn 2 -N 32 -rpn 16` 形态已与 `dsub -h` 对齐，但**没有**任何两节点
  `execNodes` 证据，不能据此写“两节点可跑”。

### 5.3 hostfile / rankfile 是当前最大缺口

现有驱动与 `pyqcu/testing/simple/mpi/solve_case.py` 的 `build_mpirun_command`
只拼 `mpirun --bind-to ... --map-by ... -np N`，没有 `--hostfile`/`--host`/
`--rankfile` 参数
（[`mpi/solve_case.py:327`](../pyqcu/testing/simple/mpi/solve_case.py)、
[`mpi/_entrypoint.py:407`](../pyqcu/testing/simple/mpi/_entrypoint.py)）。
仓库文档记录的跨节点作业正是因此在 `mpirun` 阶段 `RC=255`
（[`docs/kupl_l24_perf_20261113.md:496-499`](kupl_l24_perf_20261113.md)）。
目标 `driver.run_row` 必须：

1. 先查 `CCS_ALLOC_FILE`；若存在，解析节点列表生成 rankfile（格式按手册示例）。
2. 若不存在，显式报错 `no scheduler allocation file; refusing cross-node launch`，
   不尝试“猜”节点或退回单节点。
3. 把 `--rankfile`、`-np`、`--map-by`、`--bind-to` 和节点数写进 row 的
   `launcher` 证据字段，便于审计。

远端已确认 `CCS_ALLOC_FILE` 的真实格式是每行 `host total_cores rank_count`
（Donau 侧另有 `CCS_NTASKS_PER_NODE=16`），手册与 HMPI 作业脚本据此生成 rankfile，
所以第 1 条有确定的解析目标。但 `login02` → `cn*` 仍报
`Permission denied (publickey)`：`mpirun` 的 `--mca plm_rsh_agent /usr/bin/ssh`
起不到远端 rank。**免密未补齐前，rankfile 齐备也不能跨节点**（见 §2.4.5、§9.2）。

### 5.4 共享 halo（`--shm`）

`pyqcu.test.native.resolve_shm` 的当前语义是：请求共享 halo 但 rank 跨节点时
`SKIP(shm_unavailable)`（[`native.py:418-466`](../pyqcu/test/native.py)）。
目标规则：

- 两节点 L48 默认 `--no-shm`，使用 MPI halo。
- 若用户显式 `--shm` 且 rank 跨节点：必须是可解释的 `SKIP(shm_unavailable)`，
  不得静默回退到 `shm=0` 后仍报告 `shm=true`。
- 单节点先做 `--shm` 对照；跨节点结果只能标“MPI halo”，不能和单节点共享 halo
  混算加速比。

## 6. `driver.run_row` 契约（建议）

设计时 `MODE_HOOKS` 只声明 `driver:("run_row",)` 而仓库没有 `pyqcu/test/driver.py`；
2026-11-17 14:0x 的并发批次补上了 `pyqcu/test/driver.py`（observation-only，未提交、
未审计，见 §2.4.7）。本节保留当时的契约要求，作为审阅该实现的清单：

```python
def run_row(*, row, options, paths, tag):
    """Return observations only; never decide PASS/FAIL."""
```

输入：

- `row`：`run_id`、`tag`、`lattice`、`dtype`、`solver`、`dslash`、`mass`、`kappa`、
  `csw`、`ranks`、`threads`、`shm`、`shared`、`gauge_file`、`source_file`、
  `gauge`/`source` 生成器字段、`mg_*`、`max_iter`、`atol`。
- `options`：`FullModeOptions.as_dict()`，含 `paths` 与 `quick`。
- `paths`：`rows`/`reports`/`solutions`/`caches` 的绝对路径。
- `tag`：`Case` tag（用于选择 `native.run_dslash`/`run_solve`/`run_mg`）。

输出（至少）：

```text
status: PASS/SKIP/FAIL/ERROR/TIMEOUT   # 只允许由 runner 观测；harness 仍会复核
reason: 非空（SKIP/FAIL/ERROR 时）
seconds, wall_seconds, elapsed_seconds
iterations, max_iter, final_true_residual, residual_source
gauge_origin, gauge_path, gauge_sha256, source_origin, source_path, source_sha256
lattice, grid, ranks, threads, shm, shared
launcher: {kind, mpirun, rankfile, nodes, ranks_per_node}
```

硬约束：

- `solver`/`dslash`/`mass`/`csw` 必须真正传到 ABI；不能只写进报告。
- `--threads` 必须在启动原生调用前设置 `OMP_NUM_THREADS`/`KUPL_OMP_THREADS`，
  并在 `metrics` 里记录 `threads_applied`；不能沿用“只记录”的 tier-0 行为。
- `--shm` 的处理见 §5.4。
- row 里不得出现 `gauge_origin=random` 而实际读的是 LIME 文件这类矛盾。

## 7. 与旧驱动的替换映射

| 旧入口 | 目标入口 | 说明 |
|---|---|---|
| `test_kunpeng_all.py --mode list --expand` | `pyqcu.test --mode list` + `--sweep`/矩阵 | 需要补 L48 preset 与矩阵轴；`--limit`/`--shard-index` 语义可映射到 `--run-label` + 外部分片 |
| `--mode prepare` | `--mode prepare` | 复用 `reports/prepare.json`、`caches/index.json` |
| `--mode prewarm` | `--mode prewarm` | 复用 SHA256/字节数记录 |
| `--mode solve` | `--mode solve` | 需要 `driver.run_row` 真正执行原生求解 |
| `--mode merge` | `--mode merge` | 复用 `rows/*.json` → CSV |
| `--mode validate`（小格冒烟） | **不能直接映射** | 正式端口 `--mode validate` 是台账一致性校验；数值冒烟应新增 `--mode smoke` 或 `--validate` 的数值子命令，见 §9 |
| `submit_kunpeng_all.sh` | 新的 `submit_pyqcu_test.sh`（建议新文件） | 增加 `--nodes`、`--ranks-per-node`、rankfile；不再写死 `-nn 1` |

`run_id` 与 `--resume` 语义必须保持：`run_id` 由配置摘要决定，`--run-label` 只追加
后缀；`--resume` 只复用 PASS/SKIP，`--skip-failed` 额外复用 FAIL；`ERROR/TIMEOUT`
永远重试。CSV 列名尽量保持与旧驱动兼容，避免下游分析脚本失效。

> 2026-11-17 14:0x 的并发实现已补上 `L48` preset 与 `driver.run_row`（§2.4.7），但
> 矩阵轴扩展、L48 生产行与 100 列 CSV 接线仍未完成。

## 8. 验收门

任何一条 L48 row 只有同时满足以下条件才能写 PASS：

1. **输入门**：LIME 魔数/元数据正确；HDF5 `describe` 的 `role`/`shape`/`dtype`/
   `lattice` 与 row 一致；`sha256` 已记录；没有 `.partial` 或 0 字节文件。
2. **启动门**：`mpirun` 的 rank 数等于 `prod(grid)`；两节点作业的 `djob -ll`
   `execNodes` 恰好两个；rankfile 中每个 rank 的节点/槽位可审计。
3. **数值门**：`status=PASS`；`0 < iterations <= max_iter`；独立真残差
   `||D x - b|| / ||b||` 不超过 dtype 门（c64 至少 `1e-6` 量级，具体按
   `pyqcu.test.policy`）；残差必须由独立重构给出，不能复用求解器递推。
4. **产物门**：`reports/<run_id>.json` 存在且可读；`solutions/<run_id>.h5` 若声明
   必须可读；`rows/<run_id>.json` 的 `status`/`metrics` 与 report 一致。
5. **台账门**：`--mode merge` 后 `reports/rows.csv` 行数正确；`--mode validate`
   无 problem；空台账返回 1，不得当绿色。
6. **证据门**：作业号、提交命令、节点数、每 rank 日志、`wall_seconds`/`elapsed_seconds`
   分开记录；没有这些字段的数字只能标 `UNVERIFIED`。

## 9. 开放问题（未解决项为 TODO/待验证）

### 9.1 远端访问

- [x] 远端已可登录：`login02` / `nsls_yangyb`，经用户提供的内部路由（§2.4.1）。
      `ssh lingsheng` 仍不解析，那是别名缺失，不是访问方式；地址与密钥不入库。
- [x] 远端仓库路径已知：`/home/share/nsls_yangyb/PyQCU-main`（非 git 快照，
      `pyqcu.bak.20261117` 为 0.1.17 覆盖前的备份）。
- [x] 远端 L48 已存在：`~/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`，
      大小与前 4 KiB 哈希与本地一致。**不需要上传**。
- [ ] 全文件 SHA256 未比对（只比了大小与前 4 KiB）；远端 `/home/share` 配额
      （实测 366 T 可用）之外的 per-user 限额未确认。

### 9.2 调度器与两节点

> **22:15 更新**：KUPL harness 两节点 c64 已 PASS（`2155536`，`drun`，32 rank，
> 真残差 `7.12609e-05`，§0.1.6）；**23:30 更新**：harness 两节点已扩到四个变体
> （c128 32r、c64 16r、c64 64r、c64 Legacy-MG 16r cache-load，§0.1.7）。下面的
> `2152277`/`2153008` 记录是正式端口旧快照；正式端口两节点矩阵仍未验证，且远端
> 覆盖树没有 `--l48-two-node`。SSH key 已存在并被接受，但计算节点关闭 shell，因此
> `plm_rsh` 仍不可用。

- [x] `#DSUB` 头的语义已按手册与 `dsub -h` 复核：`-nn` 节点数、`-N` 总副本数、
      `-rpn` 每节点副本数；`#DSUB` 只在 `dsub -s <script>` 时被解析（§2.4.5）。
- [ ] 两节点探测作业 `2152277` 仍 **PENDING**（`No available node can fit ...
      execution nodes preference`）：没有 `execNodes` 证据，两节点未验证。
- [ ] `CCS_ALLOC_FILE` 格式已在 Donau 侧样例中确认（`host total_cores rank_count`，
      `CCS_NTASKS_PER_NODE=16`）；`CCS_TASK_REPLICA` 只在手册模板中出现，本作业未观测。
- [ ] 计算节点免密仍是硬缺口：`login02` → `cn*` 报 `Permission denied (publickey)`；
      不补齐则手册模板的 `--mca plm_rsh_agent /usr/bin/ssh` 无法跨节点起 rank。
- [ ] `-nn "min,max"` 的弹性申请、两节点 walltime 上限、L48 cache（历史 4-rank
      `499 s`）加 solve/I/O 是否超过默认 900 s，均未实测。

### 9.3 输入与格式

- [x] 远端 `h5py.get_config().mpi` = **False**（h5py 3.7.0 / HDF5 1.10.6）：没有并行
      HDF5，多 rank cache 构建/MPIO slab 读取不可用当前实现。
- [x] `prepare_case.py` 能在远端写出 L48 HDF5（41 s），但**文件名 dtype 误标**
      （`*_c64.h5` 实为 complex128，`6,115,301,376 B`），见 §2.4.4。
- [ ] `prepare_case.py` 的 HDF5 是否被 `pyqcu.test.io.describe(role="gauge"/"source")`
      全部接受（`role`/`layout` 属性缺失时 full 布局允许）——c64 转换完成后才可判。
- [ ] L48 c64 canonical gauge/source 磁盘占用约 4.08 GB（3.80 GiB），cache 逻辑约
      17.4 GiB；远端配额是否足够、是否要先清理旧 cache。
- [ ] L48 `c128` 是否需要；LIME 本身是 `precision D`，直接读入可省一次转换，
      但内存/带宽代价未测。

### 9.4 数值与性能

> **22:15 更新**：`2155536` 是 KUPL harness 两节点 c64 PASS（`1.1778 s`、11 it、
> 真残差 `7.12609e-05`）；**23:30 更新**：§0.1.7 又归档 c128 32r、c64 16r、c64 64r
> 与 c64 Legacy-MG 16r cache-load 四个 harness PASS。它们都不改变正式端口 L48 矩阵
> 的 `UNVERIFIED` 状态：正式 c128 行仍 `UNVERIFIED`，两节点 MG 只走 cache-load 路径
> （粗修正未生效），也没有 c128 正式行或两节点 MG 加速证据。

- [x] 单节点 c64 已收敛：BiCGStab 32 rank 独占节点 `1.6646 s`/11 it/`7.126e-5`、
      Legacy-MG 16 rank 残差 `7.126e-5`（§0.1）。**两节点**只有 KUPL harness 证据
      （c64 32r §0.1.6；c128 32r / c64 16r / c64 64r / c64 Legacy-MG 16r §0.1.7），
      正式端口 `solve`/`mg` 行仍未验证。
- [ ] L48 **c128** BiCGStab（`2153005` PENDING）与**两节点** `solve`/`mg` 是否收敛。
      **KUPL harness** 侧已有 c128 两节点 32r PASS（`2155785`，§0.1.7）和 c64
      Legacy-MG 16r cache-load PASS（`2155888`，粗修正未生效）；但**正式端口**的
      c128 行与两节点 `solve`/`mg` 行仍无有效结果。
- [ ] L48 Legacy-MG cache 在两节点/32 rank 下能否构建和复用；历史 L48 cache 是
      4-rank DCU 证据，不是 LX2 两节点证据。
- [ ] 独立真残差在 L48 上需要多大的 host 内存；`native.py` 的全格点 torch 参考
      是否会在 565 GiB 节点内完成，还是需要分块/流式残差。
- [ ] `--shm` 跨节点 `SKIP(shm_unavailable)` 的实际触发路径；共享 halo 在单节点
      的对照值不能外推到两节点。
- [ ] `--threads` 在 KUPL 原生路径是否真正设置到 `OMP_NUM_THREADS`：设计时
      `apply_environment` 存在但 `driver:run_row` 未接线；并发实现补了 `driver.py`
      （§2.4.7），是否真正下传线程数**未审计**。
- [ ] 是否需要 `--max-iter`/`--atol` 的 L48 专用默认值；不能沿用 L24 的 50 iter
      或旧 `1e-4` 口径。

### 9.5 正式端口语义

- [x] `LATTICE_PRESETS` 已含 `L48=(48,48,48,96)`（并发实现，未提交）。但
      `--mode list --lattice-preset L48` 仍只展开 17 个 tier-0 case（CG 退役后 19−2），
      **L48 生产 `solve/mg` 行仍缺**。
- [ ] `--mode validate` 名称冲突：旧驱动是数值冒烟，正式端口是台账校验。需要
      新增 `--mode smoke` 或在 `--validate` 上加明确的数值子选项。
- [ ] `--mode list --lattice-size 48 48 48 96` 同样只列 17 个 tier 0 case，
      不会生成 L48 生产 `solve/mg` 行；需要 row 规划层扩展。
- [ ] `--gauge-file`/`--source-file` 原先只接受 HDF5、不接受 LIME；并发实现把
      HDF5 读取下沉到 native 输入身份，但仍无 `--gauge-lime`，LIME 必须先转 HDF5。
- [x] **文档计数漂移（已修复）**：根 `AGENTS.md:29`、`:64`、`:88`、`pyqcu/AGENTS.md`、
      `pyqcu/testing/AGENTS.md` 与 `skills/AGENTS.md:93` 的 8 处旧 tier-0 计数 21 已改为 19；
      `docs`/`skills` 里 6 处旧 CUDA 计数 5 已改为 3。实际 `python -m pyqcu.test --list`
      与本机 `--check` 为 **19 个 case / 20 个 check（11 torch, 5 kupl, 3 cuda）**；
      `pyqcu/test/README.md` 也是 19。

### 9.6 本轮新增待办（2026-11-17 远端）

- [x] L48 c64 已用于真实求解（§0.1 第 1–3 行）；c64 cast 与 `io.describe` 回读的
      原始 stdout 仍需归档（证据等级 C）。
- [ ] 归档远端原始输出：`--check`、版本探针、`prepare_case.py` 计时、`djob -l 2152277`。
- [ ] CSV/Excel 台账接线：`pyqcu/test/ledger_csv.py`、`ledger_excel.py` 尚未被
      `__main__.py`/`common.py` 引用，`--csv` 仍写 30 列 `ROW_FIELDS`；parity
      测试 `pyqcu/testing/python/test_ledger_parity.py` 未在最终树跑完 → 不发 PASS。
- [ ] 版本冻结：当前发布是 **0.1.17 / tag `test37`**（HEAD `1f65469e`）；本轮不改
      版本、不改 tag。下一次发布只会在十轮复审全过后作为 `0.1.18` / `test38`，
      不 PyPI、不 push、不打 tag。

> 本节结果批次产生的 9 条阻塞（c128 `2153005`、两节点 `2153008`、32r MG 无匹配
> cache、dslash/clover 32r 挂起、Legacy-MG 分项缺失、CSV/Excel parity 等）逐条见
> §0.1.4，不在此重复登记。

## 10. 参考入口

| 用途 | 路径 |
|---|---|
| 旧 KUPL 全矩阵驱动 | [`pyqcu/testing/kupl/test_kunpeng_all.py`](../pyqcu/testing/kupl/test_kunpeng_all.py) |
| 旧 dsub 提交脚本 | [`pyqcu/testing/kupl/submit_kunpeng_all.sh`](../pyqcu/testing/kupl/submit_kunpeng_all.sh) |
| 正式端口契约 | [`pyqcu/test/README.md`](../pyqcu/test/README.md) |
| full-mode 编排与 row 台账 | [`pyqcu/test/common.py`](../pyqcu/test/common.py) |
| 原生 runner（dslash/solve/mg） | [`pyqcu/test/native.py`](../pyqcu/test/native.py) |
| HDF5 读写/校验 | [`pyqcu/test/io.py`](../pyqcu/test/io.py) |
| CSV 台账（实现中，未接线） | [`pyqcu/test/ledger_csv.py`](../pyqcu/test/ledger_csv.py) |
| Excel 汇总（实现中，未接线） | [`pyqcu/test/ledger_excel.py`](../pyqcu/test/ledger_excel.py) |
| CSV/Excel parity 测试（未跑完） | [`pyqcu/testing/python/test_ledger_parity.py`](../pyqcu/testing/python/test_ledger_parity.py) |
| 审计 P0/P1 风险登记（全部未修复） | [`docs/pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md) |
| 既有 Python L24 KUPL 审计 | [`logs/pyqcu_audit_20261112.txt`](../logs/pyqcu_audit_20261112.txt) |
| 本轮结果归档任务日志 | [`logs/v202611171745.txt`](../logs/v202611171745.txt) |
| 两节点变体结果日志（harness） | [`logs/v202611172330.txt`](../logs/v202611172330.txt)、§0.1.7 |
| L48 结果的 KUPL 侧记录 | [`docs/kupl_performance_20261115.md`](kupl_performance_20261115.md) §13、[`docs/kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md) §10、[`docs/kupl_kunpeng_hpc_config_20261112.md`](kupl_kunpeng_hpc_config_20261112.md) §11 |
| LIME 读取 | [`pyqcu/testing/simple/_lime.py`](../pyqcu/testing/simple/_lime.py) |
| LIME → HDF5 准备 | [`pyqcu/testing/simple/prepare_case.py`](../pyqcu/testing/simple/prepare_case.py) |
| MPI 小格点入口（无 hostfile） | [`pyqcu/testing/simple/mpi/solve_case.py`](../pyqcu/testing/simple/mpi/solve_case.py) |
| 两节点失败记录 | [`docs/kupl_l24_perf_20261113.md`](kupl_l24_perf_20261113.md)、[`docs/kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md) |
| L48 容量/缓存边界 | [`docs/qcu_quda_solver_dcu_l48_20261029.md`](qcu_quda_solver_dcu_l48_20261029.md) |
| LIME 转换（调用形状） | [`docs/kupl_only_deployment.md`](kupl_only_deployment.md) |
| LIME 转换（source/layout 口径） | [`docs/kupl_kunpeng_progress_20261110.md`](kupl_kunpeng_progress_20261110.md) |
| dtype/文件名误标根因 | [`docs/kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) |
| LIME 只读读取点 | [`docs/kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) |
| 用户手册 | [`refer/kunpeng/深圳超算二期用户手册.pdf`](../refer/kunpeng/深圳超算二期用户手册.pdf) |
