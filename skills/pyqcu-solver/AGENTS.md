# AGENTS.md — pyqcu-solver

pyqcu.torch.solver 的 BiCGStab(l)/FGMRES 与 legacy/Strict MultiGrid，
以及 pyqcu.solver.multigrid 三后端选择器
skill：含逐层奇偶、Galerkin、C++ fused FGMRES、持久显存语义，以及
33-tensor legacy cache 与 Strict runtime cache 的选择边界。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 缓存对比细则：`references/mg-cache-comparison.md`
- CUDA 拓扑：一进程一 rank 一设备；MultiGpu 仅限显式测试调用。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
