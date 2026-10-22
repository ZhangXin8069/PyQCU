# AGENTS.md — pyqcu-dslash

pyqcu.torch.dslash 的完整生成 skill：Wilson/Clover 狄拉克算子，含 MPI halo、奇偶预处理、Galerkin 投影与反模式清单；顶层 pyqcu.dslash 仅兼容。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- GPGPU dslash 拓扑：一进程一 rank 一设备；跨设备 halo 走 MPI。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
