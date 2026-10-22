# AGENTS.md — pyqcu-cpp-qcu-include

cpp/cuda/qcu/include 的 24 个模板化 CUDA 头文件 skill：`define.h` 镜像
`params[58]`、`params[57]` 语义和 Strict `set_ptrs` 槽位。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- CUDA 拓扑：头文件与运行时状态保持 rank-local；多设备只通过 MPI 协同。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
