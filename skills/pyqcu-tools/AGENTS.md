# pyqcu-tools

pyqcu.tools 目录的完整生成 skill：MPI 网格/奇偶分割/维度重排/HDF5 I/O/线性代数/多重网格转移/TileLang JIT 工具集。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 设备工具：生产 GPGPU 一进程一 rank 一设备，不按线程创建多设备 worker。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
