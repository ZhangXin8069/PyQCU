# AGENTS.md — PyQCU skills 项目技能目录

PyQCU 项目专用技能库：每个技能一个子目录，内含 `SKILL.md`（frontmatter + 正文）与本目录
简短说明。与通用工作流技能库 `${CONFIGURE_ROOT:-$HOME/configure}/skills`
（all/debug/optim/tag/up/auto 等）互补——
本库存放 PyQCU 代码库各目录的领域知识文档（2026-08-25 自 `.opencode/skills` 迁出并更新至
当前仓库状态，源目录已按计划删除；如需 opencode 加载，从本库同步到 `.opencode/skills`）。

## 全局约定（继承 configure/skills/AGENTS.md）

- **日志预读策略**：会话第一步只读最新 1 份 `.X.<时间戳>.log` 尾部汇总区（tail -20 上限），
  只取上次任务/结论/遗留项，秒级速览；无历史日志正常跳过。
- **前置上下文预读**：说明文件按 AGENTS.md > README.md > AGENT.md > CODEX.md
  取第一个存在者 head -40；agent 文件夹按 .opencode > .claude > .codex 取第一个存在者 ls 限行。
- **AGENTS.md 同步**：任意技能会话发现工作目录 AGENTS.md 缺失确凿有益内容时最小改动补充
  （先读后写、不重复条目、不代提交）。
- **TODO 管理**：会话执行第一步用 todowrite 生成详细 TODO 列表，逐步实时更新，收尾核对全完成。
- **跨技能调用**：必要时调用其他技能；2+ 独立子任务（无共享状态/无顺序依赖/互不改同文件）
  优先并行派发。
- **技能表同步约定（硬性）**：新增/更改技能必须同步下方技能表行与计数（当前 32/32），
  避免表格与实际脱节。
- **命名约定（硬性）**：自用技能目录一律 `pyqcu-` 前缀（2026-10-15 全量改名完成），
  frontmatter `name` 必须与目录名逐字符一致。

## 文档范式

frontmatter：name 与目录名一致 + description 写触发场景（何时使用）；正文为目录技能文档
（Files / Exported API / Key Anti-Patterns / Lessons）。项目知识条目须带实测数字与
出处路径（logs/<tag>/...），不写未验证内容。

## 技能表（32/32）

按域分组；每行用途即该技能 frontmatter description 的摘要。2026-10-15 全量加 `pyqcu-`
前缀，`cann.v2`/`cuda.v2`/`dtk`/`maca` 四个占位技能合并为 `pyqcu-cpp-containers`。

### 库内 Python 包

| 技能 | 用途 |
|---|---|
| `pyqcu-lattice` | pyqcu.lattice：gamma/Gell-Mann 矩阵、SU(3) 检查、规范场生成与 Ward 负索引约定。 |
| `pyqcu-dslash` | pyqcu.dslash：Wilson/Clover 狄拉克算子（hoping/sitting/operator 三类），含 MPI halo 交换、奇偶预处理、Galerkin 粗网格投影与反模式清单。 |
| `pyqcu-solver` | pyqcu.solver：BiCGStab(l)/FGMRES 与 legacy/Strict QUDA-style MultiGrid，含奇偶、Galerkin 与持久显存语义。 |
| `pyqcu-smear` | pyqcu.smear：stout smearing（Morningstar-Peardon SU(3) 投影）与 Wuppertal 高斯模糊，含数值稳定性处理与 MPI 支持。 |
| `pyqcu-cann` | pyqcu.cann：Ascend NPU 的 torch 兼容层，将复数算子分解为实/虚部；含 force_use_npu 全局标志与 einsum 组合分解算法。 |
| `pyqcu-cuda` | pyqcu.cuda：Cython/libqcu.so 桥接、Strict MultiGrid、params[58]/argv[7]/set_ptrs[100] 与显存生命周期。 |
| `pyqcu-tools` | pyqcu.tools：MPI 网格/奇偶分割/维度重排/HDF5 I/O/线性代数/多重网格转移/TileLang JIT/基准图表工具集。 |

### C++ 后端（cpp/）

| 技能 | 用途 |
|---|---|
| `pyqcu-cpp-containers` | cpp/ 四个后端容器目录（cuda 真实现 + cann/dtk/maca 占位）的镜像结构与构建入口索引。 |
| `pyqcu-cpp-qcu` | cpp/cuda/qcu 主 C++ CUDA 后端：legacy/Strict MultiGrid、逐层 P/R/X/Y/Yhat、MATPC/DIRECT_PC、构建与显存不变量。 |
| `pyqcu-cpp-qcu-include` | cpp/cuda/qcu/include 的 24 个模板化 CUDA 头文件；define.h 镜像 params[58]、Strict 槽位和 pyqcu/cuda/define.py。 |
| `pyqcu-cpp-qcu-src` | cpp/cuda/qcu/src 的 32 个 .cu 启动/模板源，含 apply_multigrid_strict.cu 的 full-coarse/MATPC/R/P/fused FGMRES。 |
| `pyqcu-cpp-qcu-python` | cpp/cuda/qcu/python 的 C API 边界：pyqcu.h（37 入口）与 pyqcu/cuda/qcu 下 qcu_api.pxd、qcu.pyx/qcu.pyi 的同步。 |
| `pyqcu-cpp-qcu-logs` | cpp/cuda/qcu/logs **已于 2026-10-15 删除**（删除前仅 2 个已跟踪文件）；保留内容清单、删除依据与后端产物落点约定。 |

### 测试（pyqcu/testing/）

| 技能 | 用途 |
|---|---|
| `pyqcu-testing` | pyqcu.testing 集成测试与 Strict MultiGrid tier 0/1/2 快速闸门、MPI/缓存/QIO/显存验证协议。 |
| `pyqcu-testing-python` | pyqcu/testing/python：纯 Python 算子/求解器主测试套件（conftest 入口 + 各 conftest.*.py 变体）。 |
| `pyqcu-testing-cpu` | pyqcu/testing/cpu：纯 Python 后端的 CPU 专用测试。 |
| `pyqcu-testing-gpu` | pyqcu/testing/gpu：通用 GPU 测试占位。 |
| `pyqcu-testing-npu` | pyqcu/testing/npu：昇腾 NPU 测试（可用 force_use_npu 在 CPU 上测 NPU 路径）。 |
| `pyqcu-testing-dcu` | pyqcu/testing/dcu：AMD DCU / ROCm (HIP) 测试。 |
| `pyqcu-testing-profiler` | pyqcu/testing/profiler：torch.profiler 性能剖析，导出 Chrome trace 供 Perfetto 可视化。 |
| `pyqcu-testing-tilelang` | pyqcu/testing/tilelang：TileLang JIT 内核测试（CUDA）。 |
| `pyqcu-testing-data` | pyqcu/testing/data：测试验证参考 HDF5 文件（with_data=True 时使用）。 |
| `pyqcu-testing-qcu` | pyqcu/testing/qcu：经 Cython 桥测 C++ CUDA 后端；含按职责分类的多重网格性能基准套件与 Strict 对照。 |
| `pyqcu-qcu-single-tests` | pyqcu/testing/qcu 与 pyqcu/testing/quda 的 QCU/QUDA 单功能测试、PyTorch 参考校验和布局转换约定。 |
| `pyqcu-testing-pyquda` | pyqcu/testing/pyquda：PyQCU 与 PyQuda 双进程隔离对比套件——Wilson/Clover dslash、BiCGStab/CG 的结果与性能对比，含维度排布转换与归一化锚定。 |

### 历史攻坚与日志

| 技能 | 用途 |
|---|---|
| `pyqcu-benchmark` | PyQCU 性能基准与 dev87 Strict MultiGrid/QUDA 公平计时、显存证据协议。 |
| `pyqcu-logs-debug` | logs/debug：逐轮调试/修复日志（fix-log*.md），完成后升级为 logs/fix-report-*.md。 |
| `pyqcu-logs-results` | logs/results：最终/剩余修复报告（权威记录）。 |

### 交付与治理

| 技能 | 用途 |
|---|---|
| `pyqcu-form` | PyQCU 命名、目录、文档/日志/数据/测试布局与 Git 交付格式治理；含本地例外审计。 |
| `pyqcu-tag` | Git stab/dev/bug/test 标签管理、历史 follow 链校验与合法并行分支识别。 |
| `pyqcu-md-tex-sync` | Markdown 唯一源、LaTeX 派生源与 XeLaTeX PDF 的同源生成；全文源快照、全附件 SHA256 清册、幂等哈希同步与全页渲染闸门。 |
| `pyqcu-report` | PyQCU 中文科研汇报：横版 16:9 Beamer、竖版完整讲稿、PPTX、报告原图提取、公式/算法/图表编号与全页版式验收。 |
