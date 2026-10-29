# PyQCU test34 L48 容量与性能门

> 状态：`BLOCKED-BY-USER`，2026-10-29 23:35 CST 起暂停。  
> 后续状态：不再提交 canonical/cache/性能作业；恢复需用户明确解除暂停。  
> 范围：真实 LIME、`48x48x48x96`、c64、QCU Legacy-MG 与新 QUDA MG。  
> 原则：历史 L48 cache/A100 性能不作为本轮结果；无通过认证的新 cache 不运行性能。

## 1. 输入与执行约束

| 项目 | 值 |
|---|---|
| LIME | `/public/home/sunpeng_1/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime` |
| lattice | `48 48 48 96` |
| mass | `-0.277` |
| csw | `1.160920226` |
| dtype | c64 |
| cache builder | 4 ranks，4 DCU，32 CPU，`2x2x1x1` |
| 性能 | 2/4 ranks，2/4 DCU，16/32 CPU，一 rank 一 DCU |
| 作业上限 | 每作业 600 s；300 s 强制审计；不得并发 L48 作业 |

canonical gauge/source 只由 LIME 在本轮生成，放在
`data/qcu_solver_test34_20261029/L48/case/`。cache builder 只读取 canonical
gauge HDF5，不直接读取 LIME。

## 2. 容量预算

按本轮 `dof=12 12`、`mg-grid=2 2 2 2` 的 canonical odd-Schur schema，
粗格为 `12x12x12x24 = 20736` 点。四个 complex64 tensor 的未压缩 payload
预算如下：

| tensor | shape | payload |
|---|---:|---:|
| `lonv` | `12x12x12x2x12x2x12x2x24x2` | 23.9 MB |
| `hnn` | `2x4x12x12x12x12x12x24` | 191.1 MB |
| `hdg` | `2x2x6x12x12x12x12x12x24` | 573.3 MB |
| `sit` | `12x12x12x12x12x24` | 23.9 MB |
| 合计 |  | 812.2 MB（十进制） |

转换前公共文件系统可用约 14 GB；该预算不作为 cache 已发布的证据。

## 3. 当前状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| canonical gauge/source | BLOCKED-BY-USER | 已按用户 50GB 磁盘约束暂停；无效 `.partial` 已清理 |
| 4-rank GPU cache build | BLOCKED-BY-USER | 未提交作业 |
| cache 认证 | BLOCKED-BY-USER | 无 PASS |
| QCU Legacy-MG 2/4 rank | BLOCKED ON CACHE | 未运行 |
| QUDA MG 2/4 rank | BLOCKED ON CACHE | 未运行 |
| 1-rank 容量尝试 | BLOCKED-BY-USER | 暂停期间不提交 |

清理记录：转换进程被终止后留下的
`data/qcu_solver_test34_20261029/L48/case/.gauge_48x48x48x96_c64.h5.7637.partial`
（`254,831,528 B`）不是可发布 canonical 文件，已于 2026-10-29 23:28 CST 删除。

## 4. 结果使用边界

- 只有 `status=PASS`、4/4 `rank_slab_reconstruction`、identity/shape、逐 tensor
  SHA256 和 global output SHA256 均通过，才可将 cache 标记为认证通过。
- setup（cache build）与 solve 分开报告；cache 构建成功不代表 QCU/QUDA solve
  完成。
- 若 10 分钟内未发布 cache，结论只能是 `CAPACITY/TIMEOUT BLOCKED`，不得引用
  历史 cache 时间或历史 A100 solve 时间。
- 未完成的 rank、level 或 `mg_grid` 扩展均标记 `BLOCKED/UNVERIFIED`。
