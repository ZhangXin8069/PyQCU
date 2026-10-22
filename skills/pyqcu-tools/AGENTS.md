# AGENTS.md — pyqcu-tools

pyqcu.tools 公共 MPI/布局/I/O/env/budget 与 pyqcu.torch.tools 专用数值工具。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 设备工具：生产 GPGPU 一进程一 rank 一设备，不按线程创建多设备 worker。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
