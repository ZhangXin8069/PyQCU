# AGENTS.md — skills/pyqcu-docs/scripts

`pyqcu-docs` 的结构与覆盖审计脚本目录。

- `audit-docs-skills.py` 是 docs/skills 变更的规范审计入口；从仓库根运行
  `python -B skills/pyqcu-docs/scripts/audit-docs-skills.py`。
- 修改脚本时同步父目录 `SKILL.md`/`AGENTS.md` 中的命令说明，并保留 fail-closed 校验。
- 不在此目录写入审计输出、缓存或会话日志。
