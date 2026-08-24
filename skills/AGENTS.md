# AGENTS.md — PyQCU skills 项目技能目录

PyQCU 项目专用技能库：每个技能一个子目录，内含 `SKILL.md`（frontmatter + 正文）与本目录
简短说明。与通用工作流技能库 `/root/configure/skills`（all/debug/optim/tag/up/auto 等）互补——
本库存放 PyQCU 代码库各目录的领域知识文档（2026-08-25 自 `.opencode/skills` 迁出并更新至
当前仓库状态，源目录已按计划删除；如需 opencode 加载，从本库同步到 `.opencode/skills`）。

## 全局约定（继承 configure/skills/AGENTS.md）

- **日志预读策略**：会话第一步只读最新 1 份 `.X.<时间戳>.log` 尾部汇总区（tail -20 上限），
  只取上次任务/结论/遗留项，秒级速览；无历史日志正常跳过。
- **前置上下文预读**：说明文件按 AGENTS.md > README.md > CLAUDE.md > AGENT.md > CODEX.md
  取第一个存在者 head -40；agent 文件夹按 .opencode > .claude > .codex 取第一个存在者 ls 限行。
- **AGENTS.md 同步**：任意技能会话发现工作目录 AGENTS.md 缺失确凿有益内容时最小改动补充
  （先读后写、不重复条目、不代提交）。
- **TODO 管理**：会话执行第一步用 todowrite 生成详细 TODO 列表，逐步实时更新，收尾核对全完成。
- **跨技能调用**：必要时调用其他技能；2+ 独立子任务（无共享状态/无顺序依赖/互不改同文件）
  优先并行派发。
- **技能表同步约定（硬性）**：新增/更改技能必须同步下方技能表行与计数（当前 39/39），
  避免表格与实际脱节。

## 文档范式

frontmatter：name 与目录名一致 + description 写触发场景（何时使用）；正文为 CLAUDE.md 式
目录文档（Files / Exported API / Key Anti-Patterns / Lessons）。项目知识条目须带实测数字与
出处路径（logs/<tag>/...），不写未验证内容。

## 技能表（39/39）

| 技能 | 用途 |
|---|---|
| `benchmark` | examples/benchmark 目录的完整生成 skill：PyTorch / TileLang / C++ CUDA 后端性能基准。 |
| `cann.v2` | cpp/cann 目录的完整生成 skill：昇腾 CANN C++ 后端容器目录（占位）。 |
| `cann` | pyqcu.cann 目录的完整生成 skill：Ascend NPU 的 torch 兼容层，将复数算子分解为实/虚部；含 force_use_npu 全局标志与 einsum 组合分解算法。 |
| `cpu` | examples/cpu 目录的完整生成 skill：纯 Python 后端的 CPU 专用测试。 |
| `cuda.v2` | cpp/cuda 目录的完整生成 skill：CUDA 后端容器目录，真实现位于 qcu/。 |
| `cuda` | pyqcu.cuda 目录的完整生成 skill：C++ CUDA 后端（libqcu.so）的 Cython 桥接包；含 22 个 C 函数封装、params/argv/set_ptrs 参数协议与 _SET_INDEX_ 递增关键约束。 |
| `data` | examples/data 目录的完整生成 skill：测试验证参考 HDF5 文件（with_data=True 时使用）。 |
| `dcu` | examples/dcu 目录的完整生成 skill：AMD DCU / ROCm (HIP) 测试。 |
| `debug` | logs/debug 目录的完整生成 skill：逐轮调试/修复日志（fix-log*.md），完成后升级为 logs/fix-report-*.md。 |
| `dev76` | logs/test76 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `dev78` | logs/test78 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `dev78_1` | logs/dev78_1 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `dev78_2` | logs/dev78_2 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化 + 全求解器迭代残差图（conv/conv_plots 子命令：MG 实测 CONVERGENCE_HISTORY + 参考 BiStabCG Python 复现，逐配置/逐格子/汇总全采集）。测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `dev80` | logs/dev80、dev80_2、dev80_3 系列的完整生成 skill：大格子 MultiGrid 攻坚三代——sm_60+sm_70 fat-binary、HierarchicalCache VRAM→RAM→DISK offload（22.97GB 大格子可跑）、BatchedLocalSchur W=10 stencil 提速 12×。 |
| `dev84` | examples/qcu/dev84 目录的完整生成 skill：16×32×32×48 MultiGrid 真实加速比攻坚套件（CUDA Graph 段回放/零拷贝标量/守卫标量内核/粗空间诊断 ρ_V），报告 dev84_report.md。 |
| `dslash` | pyqcu.dslash 目录的完整生成 skill：Wilson/Clover 狄拉克算子（hoping/sitting/operator 三类），含 MPI halo 交换、奇偶预处理、Galerkin 粗网格投影与反模式清单。 |
| `dtk` | cpp/dtk 目录的完整生成 skill：DCU/ROCm (HIP) C++ 后端容器目录（占位）。 |
| `gpu` | examples/gpu 目录的完整生成 skill：通用 GPU 测试占位（当前为空）。 |
| `include` | cpp/cuda/qcu/include 目录的完整生成 skill：26 个模板化 CUDA 头文件（内核内联），define.h 须镜像 pyqcu/cuda/define.py。 |
| `lattice` | pyqcu.lattice 目录的完整生成 skill：gamma/Gell-Mann 矩阵、SU(3) 检查、规范场生成与 Ward 负索引约定。 |
| `logs` | cpp/cuda/qcu/logs 目录的完整生成 skill：CUDA 后端本地运行日志目录（gitignored），正式报告存放于仓库根 logs/。 |
| `maca` | cpp/maca 目录的完整生成 skill：Maca C++ 后端容器目录（占位）。 |
| `npu` | examples/npu 目录的完整生成 skill：昇腾 NPU 测试（可用 force_use_npu 在 CPU 上测 NPU 路径）。 |
| `profiler` | examples/profiler 目录的完整生成 skill：torch.profiler 性能剖析，导出 Chrome trace 供 Perfetto 可视化。 |
| `pyqcu` | examples/pyqcu 目录的完整生成 skill：纯 Python 算子/求解器主测试套件（conftest 入口 + 各 conftest.*.py 变体）。 |
| `python` | cpp/cuda/qcu/python 目录的完整生成 skill：pyqcu.h C API 声明（22 个 extern C 函数），必须与 qcu.pxd 完全同步。 |
| `qcu.v2` | examples/qcu 目录的完整生成 skill：经 Cython 桥测 C++ CUDA 后端；含 dev73_5 多重网格性能基准套件（clean/bench/verify/collect/mktable/plots）。 |
| `qcu` | cpp/cuda/qcu 目录的完整生成 skill：PyQCU 主 C++ CUDA 后端（hand-tuned CUDA 内核 + MPI halo 交换），含构建流程、参数协议、5-stream 架构与关键不变量。 |
| `results` | logs/results 目录的完整生成 skill：最终/剩余修复报告（权威记录）。 |
| `session-2026-08-24` | logs/session-2026-08-24 的完整生成 skill：bug31–37 无人值守会话验证资产（8 脚本 + README，15/15 PASS；覆盖基线/求解器族/MPI/Wuppertal/stencil/Galerkin/等价性）与确定性参考数据再生器。 |
| `smear` | pyqcu.smear 目录的完整生成 skill：stout smearing（Morningstar-Peardon SU(3) 投影）与 Wuppertal 高斯模糊，含数值稳定性处理与 MPI 支持。 |
| `solver` | pyqcu.solver 目录的完整生成 skill：BiCGStab(l) 与自适应多重网格 V-cycle 求解器，含 CUDA 后端集成与粗网格校正后状态重置（R3 fix）。 |
| `src` | cpp/cuda/qcu/src 目录的完整生成 skill：.cu 内核源文件（include 对应头 + 模板实例化 + 启动封装）。 |
| `test12` | logs/test12 目录的完整生成 skill：dev74*（dev74 + dev74_1）整合测试套件 test11 的优化版 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/，测试 CUDA C++ MultiGrid 求解器性能（正确性/干净测量/参数扫描/大格子预算/加速比图表）。 |
| `test13` | logs/test13 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `test14` | logs/test14 目录的完整生成 skill：多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件 — 单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化，测试 MultiGpuMultigrid（pyqcu/cuda/_multi_gpu.py）相对多线程 BiStabCG 的正确性与加速比（P100×2 多线程 + V100 单线程大格子）。 |
| `testing` | pyqcu.testing 目录的完整生成 skill：全组件集成测试（函数名 test_*，由 examples/*/conftest.py 引用），含参考 HDF5 数据验证与日志约定。 |
| `tilelang` | examples/tilelang 目录的完整生成 skill：TileLang JIT 内核测试（CUDA）。 |
| `tools` | pyqcu.tools 目录的完整生成 skill：MPI 网格/奇偶分割/维度重排/HDF5 I/O/线性代数/多重网格转移/TileLang JIT 工具集。 |
