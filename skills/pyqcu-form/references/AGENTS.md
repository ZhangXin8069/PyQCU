# AGENTS.md — skills/pyqcu-form/references

`pyqcu-form` 的按需参考；命名矩阵与目录/交付硬规则以父目录 `SKILL.md` 为准。

- `packaging-delivery.md`：根入口白名单（`pyproject.toml`/`MANIFEST.in`）、根 `.gitignore`
  与 `docs/.gitignore` 放行清单、`MANIFEST.in` 的 sdist 形态、双远端 Git 交付、发布台账与
  技能索引计数。仅在涉及发布文件、sdist/RTD 或提交推送时读取。
- 修改参考内容必须同步父 `SKILL.md`，并运行 `bash skills/pyqcu-form/form-audit.sh` 与
  `git diff --check`。
