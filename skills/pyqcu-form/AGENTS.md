# AGENTS.md — pyqcu-form

PyQCU 格式治理技能。入口为 `bash skills/pyqcu-form/form-audit.sh`；
本地规则见根 `AGENTS.md` 的“form 格式约定”和 `docs/ORGANIZATION.md`。

通用命名矩阵、参考快照和完整治理流程位于
`${CONFIGURE_ROOT:-$HOME/configure}/skills/form/SKILL.md`。`form-audit.sh`
会校验技能目录、技能表与 frontmatter，并过滤本库已登记的 `data/**`、`logs/**`、
解包 Office 资源和技能自带测试例外；任何其他 finding 仍返回失败。

tag 报告先提交并随 tag 归档，再在后续清理提交中从工作区删除。不得保留
`logs/<tag>/**` 报告副本；历史读取使用 `git show <tag>:<path>`。
