# pyqcu-testing-dcu

pyqcu/testing/dcu 目录的完整生成 skill：AMD DCU / ROCm (HIP) 测试。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- DCU 拓扑：一进程一 rank 一设备；单进程多设备只限显式测试。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
