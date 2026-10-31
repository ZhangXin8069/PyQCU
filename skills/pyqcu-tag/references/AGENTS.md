# AGENTS.md — skills/pyqcu-tag/references

`pyqcu-tag` 的详细操作参考；标签类型、消息格式与安全闸门以父目录 `SKILL.md` 为准。

- `tag-operations.md` 提供创建、查看、删除、改写与 follow 链检查命令。
- 已推送标签的删除或改写必须获得明确授权；默认先运行历史链 `--check`，不可安全推导时停止。
- 修改参考命令时同步父技能，并运行 `bash skills/pyqcu-tag/tag-chain.test.sh` 与
  `git diff --check`。
