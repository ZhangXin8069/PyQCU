# AGENTS.md — pyqcu-form

PyQCU 格式治理技能。入口 `bash skills/pyqcu-form/form-audit.sh`；本地规则见根 `AGENTS.md` 的
“form 格式约定”与 `docs/ORGANIZATION.md`，通用命名矩阵见
`${CONFIGURE_ROOT:-$HOME/configure}/skills/form/SKILL.md`。

- 发布文件、sdist/RTD、Git 交付读 `references/packaging-delivery.md`（索引 `references/AGENTS.md`），
  仅在改 `pyproject.toml`/`MANIFEST.in`、`.gitignore` 放行、sdist 内容或提交/推送时读取。
- `form-audit.sh` 校验技能目录、技能表与 frontmatter，精确豁免 `data/**`、`logs/**`、
  `docs/en|zh` 站点源、Office 资源与技能自带测试例外；其他 finding 仍失败，新增例外须
  精确匹配并附证据，不得改写历史项。
- tag 报告先提交并随 tag 归档，再在后续清理提交中删除；历史读取用 `git show <tag>:<path>`。
