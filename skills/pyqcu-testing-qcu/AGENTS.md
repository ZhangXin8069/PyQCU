# AGENTS.md — pyqcu-testing-qcu

pyqcu/testing/qcu：经 Cython 桥测 C++ CUDA 后端、MultiGrid 基准套件与集群作业模型。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- CUDA 拓扑：一进程一 rank 一设备；本目录多卡基准均为测试专用。
- 调度硬边界：一作业一节点 + 内层 `mpirun`；`dsub -s`、`#DSUB -T`（秒）、`-x job`、
  `testusers_rebuild` 上限与 `dkill` 取消边界见 `SKILL.md` §集群作业模型与节点身份。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
