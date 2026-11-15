# AGENTS.md — pyqcu-testing-data

pyqcu/testing/data：`with_data` 参考 HDF5，以及生产 gauge/source 输入资产的写/读契约。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 输入只读：损坏文件只改名重生成 + `*.corrupt-*` 备份，不回写、不修补；写路径见
  `SKILL.md` §输入资产写入契约 与 `../pyqcu-tools/SKILL.md`。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
