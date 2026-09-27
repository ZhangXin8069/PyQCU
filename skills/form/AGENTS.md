# AGENTS.md — form

PyQCU 格式治理技能。入口为 `bash skills/form/form-audit.sh`；
本地规则见根 `AGENTS.md` 的“form 格式约定”和 `docs/ORGANIZATION.md`。

通用命名矩阵、参考快照和完整治理流程位于
`/root/configure/skills/form/SKILL.md`。`form-audit.sh` 过滤本库已登记的
`data/**`、`logs/**`、解包 Office 资源和技能自带测试例外，任何其他 finding 仍返回失败。
