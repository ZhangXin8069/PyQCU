# AGENTS.md — refer

开发历史参考文档。

## 内容

| 文件 | 说明 |
|---|---|
| `dev71.md` | 开发里程碑 71 markdown 报告 |
| `dev71.pdf` | 开发里程碑 71 PDF 报告 |
| `dev71.tex` | 开发里程碑 71 LaTeX 源码 |

历史参考文档，记录开发进展与设计决策。

## git-rep/ 参考 Git 库

| 目录 | 说明 |
|---|---|
| `git-rep/DDalphaAMG` | Wilson-Clover AMG 求解器，Git submodule |
| `git-rep/DDalphaAMG-SM` | 2D Schwinger 模型版 AMG，Git submodule |
| `git-rep/PyQUDA` | QUDA 的 Python/Cython 包装库，Git submodule |
| `git-rep/quda` | NVIDIA GPU 格点 QCD 库，Git submodule |
| `git-rep/kuqcd` | 鲲鹏 CPU 格点 QCD 参考实现，Git submodule |
| `git-rep/docs/` | PyQCU 对上述参考库的分析、纯化剖析和技术报告 |

参考源码以上游仓库为准，submodule 工作树只读。来源 URL、固定提交、用途和致谢见
`git-rep/README.md`。PyQCU 自产报告不得写回上游工作树，统一放入 `git-rep/docs/`；
实验补丁和裁剪构建应使用 `data/**` 下的独立工作树。
