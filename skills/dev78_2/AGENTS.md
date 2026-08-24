# dev78_2

logs/dev78_2 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化 + 全求解器迭代残差图（conv/conv_plots 子命令：MG 实测 CONVERGENCE_HISTORY + 参考 BiStabCG Python 复现，逐配置/逐格子/汇总全采集）。测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
