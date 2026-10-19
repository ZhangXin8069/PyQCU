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
