# pyqcu-cpp-qcu

cpp/cuda/qcu 主 C++ CUDA 后端 skill：legacy/Strict MultiGrid、逐层
P/R/X/Y/Yhat、MATPC/DIRECT_PC、ABI、构建流程与显存不变量。

- 规范全文：`SKILL.md`（frontmatter description 为触发依据）
- 执行拓扑：一进程一 rank 一设备；单进程多设备只允许显式测试调用。
- 维护约定：更新内容时同步本文件与库级 `../AGENTS.md` 技能表（先读后写、最小改动、不代提交）
