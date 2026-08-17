# AGENTS.md — refer

开发历史参考文档。

## 内容

| 文件 | 说明 |
|---|---|
| `dev71.md` | 开发里程碑 71 markdown 报告 |
| `dev71.pdf` | 开发里程碑 71 PDF 报告 |
| `dev71.tex` | 开发里程碑 71 LaTeX 源码 |

历史参考文档，记录开发进展与设计决策。

## git-rep/ 源码快照

| 目录 | 说明 |
|---|---|
| `git-rep/DDalphaAMG` | Wilson-Clover AMG 求解器（C/CUDA）源码快照，非独立 git 仓库 |
| `git-rep/DDalphaAMG-SM` | 2D Schwinger 模型版 AMG（C++/MPI）源码快照 |
| `git-rep/PyQUDA` | QUDA 的 Python/Cython 包装库源码快照 |
| `git-rep/quda` | QUDA 1.1.0（NVIDIA GPU 格点 QCD 库）源码快照 |

四库均为参考用源码快照（共享仓库 `/root/PyQCU/.git`，无独立 `.git`）；各库 `docs/` 下存放
分析报告（`analy_*`/`pure_*`，YYYYMMDD 日期命名），由 opencode analy/pure 技能生成。
