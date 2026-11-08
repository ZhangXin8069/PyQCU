# AGENTS.md — pyqcu-pypi-release

PyQCU 的 PyPI 发布与复验技能：版本同步、sdist 打包、原生库落位、发布门、上传、干净环境复验、RTD/镜像交接。

- 规范全文与硬边界：`SKILL.md`
- 逐项清单、sdist 基线、故障对照表、台账模板：`references/release-checklist.md`
- 硬边界：只发 sdist 不发 wheel；不覆盖、不重传已发布版本；token 只读环境变量名，不落盘不回显；发布内容必须来自已提交的树。
- 发布即记账：`logs/v<YYYYMMDD>.txt`（0.1.0 台账为 `logs/v20261109.txt`）；不自动提交或推送。
