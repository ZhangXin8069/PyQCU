# AGENTS.md — skills/pyqcu-tag/references

`pyqcu-tag` 的详细操作参考；标签类型、消息格式与安全闸门以父目录 `SKILL.md` 为准。

- `tag-operations.md` 提供创建、查看、删除、改写与 follow 链检查命令。
- `dual-remote-release.md` 提供发布任务的 GitLab/Gitee 双远端推送顺序、四端同 hash 核验、
  提交 trailer 与发布标签的双远端推送约束。
- 已推送标签的删除或改写必须获得明确授权；默认先运行历史链 `--check`，不可安全推导时停止。
- 修改参考命令时同步父技能，并运行 `bash skills/pyqcu-tag/tag-chain.test.sh` 与
  `git diff --check`。
