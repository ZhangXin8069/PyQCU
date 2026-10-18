# pyqcu-cpp-qcu-logs

cpp/cuda/qcu/logs 的归档 skill：**该目录已于 2026-10-15 删除**（删除前只有 `AGENTS.md`
与 `schur_mg_test_stdout.txt` 两个已入 git 的文件），此处只保留内容清单、删除依据与
后端运行产物的现行落点约定。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 现役约定：后端测试输出走 `pyqcu/testing/qcu/`，收敛日志落在仓库根
  `logs/clover_multigrid.log`；阶段报告先提交并打 tag，随后删除工作区副本。`cpp/**`
  只保留源码与构建脚本。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
