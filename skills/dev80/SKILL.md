---
name: dev80
description: logs/dev80、dev80_2、dev80_3 系列的完整生成 skill：大格子 MultiGrid 攻坚三代——sm_60+sm_70 fat-binary、HierarchicalCache VRAM→RAM→DISK offload（22.97GB 大格子可跑）、BatchedLocalSchur W=10 stencil 提速 12×。
---
# CLAUDE.md — logs/dev80, dev80_2, dev80_3

Three generations of large-lattice multigrid enablement work (see `logs/dev80*/`).

## Key Mechanisms

- **CMakeLists-nv.txt fat-binary** — compile for sm_60 + sm_70 in one binary (P100 + V100 mixed nodes).
- **HierarchicalCache** — VRAM → RAM → DISK offload; makes 22.97 GB lattices runnable.
- **BatchedLocalSchur (W=10)** — batched Schur-based local stencil build; large-lattice
  stencil construction 24 min → 2 min (~12×). Exposed via `pyqcu/tools` (see `tools` skill).
- **lattice_sap.h** — SAP solver header present but NOT yet wired into the build (spare part).

Reports: `logs/dev80*/dev80*_diff.md`, `logs/dev80_3_analy.md`.
