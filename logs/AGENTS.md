# AGENTS.md — logs

`logs/` 只保存当前任务尚未形成 tag 归档的日志、图表和阶段报告。阶段结果先进入
待 tag 提交，tag 创建后再用独立清理提交删除工作区报告副本。

## 生命周期

1. 当前任务把 `.json/.log/.tsv/.txt/.stdout`、阶段报告和图表写入 `logs/**`；
2. 验证完成后提交报告并创建对应 tag；
3. 在后续清理提交中删除报告、图表及其它可由 tag 恢复的阶段产物；
4. 后续读取使用 `git show <tag>:<path>`，不恢复永久副本。

`logs/<tag>/**` 不作为报告的长期存放位置。若报告已经进入 tag，工作区仍存在同名副本
即属于格式冲突；若尚未进入 tag，则必须先提交并打 tag，再执行删除。

## 目录职责

- `logs/data/`：从 `data/**` 提升出的当前运行摘要；
- `logs/clover_multigrid.log`：现役 C++ 求解器收敛输出；
- `data/logs/<tag>/`：tag 关联的数据、张量或 cache 实体；
- `data/logs/nullvec_cache/`：共享 null-vector/粗算子缓存实体；
- 测试与脚本实体位于 `pyqcu/testing/**`，不放入 `logs/`。

## 日志索引

### 2026-10-20-big-file-purge - final - 2026-10-20 00:55:24 +0800
- 主任务: Git 历史大文件永久删除与仓库瘦身
- 日志文件: logs/v2026102000.txt
- 文件大小: 5512 bytes
- 触发原因: 任务完成
- 关键内容: 删除 8 个远程历史大文件路径、重写并强制同步 2 个分支和 275 个标签，完成本地与远端可达历史验证。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### 2026-10-20-tag-dev96 - final - 2026-10-20 01:01:47 +0800
- 主任务: 创建本地 dev96 附注标签
- 日志文件: logs/v2026102001.txt
- 文件大小: 3442 bytes
- 触发原因: 任务完成
- 关键内容: 基于 dev95 创建本地 dev96，记录 Git 历史瘦身；peeled commit 与 HEAD 一致，标签链自检通过，未推送远程。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### 2026-10-20-form-agents-final - final - 2026-10-20 01:16:53 +0800
- 主任务: PyQCU 2026-10-20 form 根规则托管块同步与本地技能补强
- 日志文件: logs/v2026102001-2.txt
- 文件大小: 7359 bytes
- 触发原因: 任务完成
- 关键内容: 完成 AGENTS form 唯一标记块同步、本地技能例外边界说明、任务文档与全部指定验证，终态未提交。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### tag-dev97-20261020 - final - 2026-10-20 01:32:20 +0800
- 主任务: 创建 PyQCU dev97 本地附注标签
- 日志文件: logs/v2026102001-3.txt
- 文件大小: 3261 bytes
- 触发原因: 任务完成
- 关键内容: PyQCU dev97 已创建并与 HEAD 对齐；历史链旧问题保留，标签未推送
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
历史读取示例：

```bash
git show dev73:logs/dev73/review-2026-07-28.md
git show dev94:logs/stab33/report_multigrid_python_20260828.tex
```
## dev98-csw-kupl-final - final - 2026-10-20 05:22:40 +0800
- 主任务: PyQCU Csw、QcuRuntime/simple 与独立 KUPL MPI+OpenMP 后端
- 日志文件: logs/v2026-10-20-05.txt
- 文件大小: 8205 bytes
- 触发原因: 任务完成
- 关键内容: Csw/接口/简单示例完成；KUPL 24/37 实现，高级入口 fail closed；验证与外部 QUDA 阻塞如实记录
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## quda-sm70-runtime-align - final - 2026-10-20 09:04:09 +0800
- 主任务: 修复 PyQUDA 加载 sm60 QUDA 导致 V100 no kernel image，并对齐 QCU/QUDA mass normalization
- 日志文件: logs/v2026-10-20-09.txt
- 文件大小: 5477 bytes
- 触发原因: 任务完成
- 关键内容: sm70 预载、env 选择、kappa normalization 与同源数值比较均修复并验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## kupl-qcu-structure - final - 2026-10-20 09:18:07 +0800
- 主任务: KUPL QCU 与 CUDA QCU 结构对齐重构及回归
- 日志文件: logs/v2026102009.txt
- 文件大小: 6743 bytes
- 触发原因: 任务完成
- 关键内容: KUPL QCU 分层重构、LatticeSet 门面、37 项 ABI 门禁与 Release/UBSAN 回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## kupl-cuda-alignment - final - 2026-10-20 11:51:08 +0800
- 主任务: KUPL QCU 与 CUDA QCU 对齐重构
- 日志文件: logs/v2026102011.txt
- 文件大小: 6565 bytes
- 触发原因: 任务完成
- 关键内容: 37 项 ABI、strict/legacy MG 语义、workspace 账本与回归完成；Release/UBSAN/CTest/pytest/form 全绿。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## 01a1574f-context - final - 2026-10-20 14:12:06 +0800
- 主任务: KUPL independent simple tests and performance optimization
- 日志文件: logs/v2026102014.txt
- 文件大小: 6288 bytes
- 触发原因: 任务完成
- 关键内容: KUPL simple/QUDA correctness plus adaptive SIMD and solver/MG optimization
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## 20261021-kupl-cuda-multigpu - final - 2026-10-21 00:54:57 +0800
- 主任务: KUPL Kunpeng NUMA 与 CUDA A100 多卡 debug/优化
- 日志文件: logs/v2026102100.txt
- 文件大小: 10499 bytes
- 触发原因: 任务完成
- 关键内容: 修复 KUPL 24 核线程契约、验证双 NUMA，优化 CUDA 纯 MG 路径并完成双 A100 与真实 LIME 回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## refer-git-rep - final - 2026-10-21 05:10:19 +0800
- 主任务: 整理 refer/git-rep 参考库结构
- 日志文件: logs/v2026102105.txt
- 文件大小: 5523 bytes
- 触发原因: 任务完成
- 关键内容: 参考库改为上游 submodule，34 份自产报告集中归档，form 审计通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## docs-progress-dev99-1640349d - final - 2026-10-21 06:31:38 +0800
- 主任务: archive-dev99-to-1640349d-summary
- 日志文件: logs/v2026102106.txt
- 文件大小: 2509 bytes
- 触发原因: 任务完成
- 关键内容: 完整进度总结已保存到 docs，并完成文档格式审计
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tag-dev100 - final - 2026-10-21 06:36:52 +0800
- 主任务: create-dev100-local-tag
- 日志文件: logs/v2026102106-2.txt
- 文件大小: 3494 bytes
- 触发原因: 任务完成
- 关键内容: 创建并验证本地附注标签 dev100，未推送远程
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
