# AGENTS.md — PyQCU

PyQCU：Lattice QCD 的 Python 库（DeepSeek-based、后处理），支持纯 Python 回退与多后端加速（CUDA / DCU / 昇腾 / MACA）。代码由 agent 深度参与生成；本文件与 `skills/` 目录共同构成工作指引。

## 概要命令

- 构建与安装：`./build.sh`（调用 `setup.py`，经 `CC=gcc` 环境变量走 `pybind11` 编译 C++ 后端）；依赖仅 NumPy。
- 运行：`source env.sh` 后 `import pyqcu`；纯 Python 路径默认可用。
- 测试：`pytest pyqcu/testing/`（根目录无打包级测试入口；子目录测试见各目录 AGENTS.md）。
- Git 标签：`stab<N>`/`dev<N>`/`bug<N>` 独立编号 + 子版本（如 `stab15_1`），见 tag 技能；归档说明见 `.CLAUDE.md.*.bak`。

## 架构：两个层级

1. `pyqcu/` — 纯 Python 数值核心（每模块带同目录测试文件，函数名 `test_*` 对应 pyqcu 符号）。
2. `cpp/` — 高性能 C++/Cython 后端，编译为 `pyqcu.<backend>.qcu`（CUDA/DTK/MACA/昇腾），经 `pybind11` 暴露 `qcu_abi`；封装函数带 `_SET_INDEX_` 调用（向 C 侧传 batch/index 向量）。

核心惯例（贯穿 `pyqcu` 与 `cpp`）：
- **参数协议 v1.0**：所有算子统一 `{func_name: {param: value}}` 字典，CUDA/CPU 实现共用同一协议头；C++ 侧由 `qcu_abi`/`c_qcu_abi`/`ptr` 参数三元组传递。
- **张量约定**：批外维度下标 0,1 分别为 batch、index（M/M' = batch, P/P' = index），`_SET_INDEX_` 向量按此展开。
- 支持 `index: 'left'|'right'` 配置的算子，直接支持 spinor 因子分解。
- 算子入参一律 NumPy 数组或内建标量；禁止字节串入参（Cython 签名含 `const char*` 时务必声明，避免 argtypes 冲突）。
- Cython 接口文件由 `macro_gen.py` 生成；修改算子签名后需重新生成并重编后端。
- 测试构造失败（无法 import）时以显式错误代替静默失败。

## 目录结构

| 路径 | 内容 |
|---|---|
| `pyqcu/` | Python 实现与测试：`dslash/`、`lattice/`（Dslash 符号约定官方文档）、`solver/`、`smear/`、`tools/`（原型：multi_grid/schwinger/continuum）、`cuda/dtk/maca/cann`（Cython 桥接） |
| `cpp/` | C++ 后端：`cuda/`（真实现，`src` 源码、`include` 头、`python` Cython 桥、`logs` 构建日志）、`dtk/maca/cann`（占位 PASS） |
| `examples/` | 示例：`qcu/`（共享 protocol）、`pyqcu/`（本地 python 算法、`data` 从 parquet 采读）、`benchmark/`、`cpu/`、`gpu/`、`dcu/`、`npu/`、`profiler/`、`tilelang/` |
| `logs/` | 实验日志：`results/`（README + `dev73_5` 报告）、`debug/`；README 须在生成报告时同步更新 |
| `docs/` | 文档 |
| `refer/` | 参考资料（dev71 系列报告等） |

## Skills（生成各目录内容的完整 skill 均归档于此）

- `skills/<name>/SKILL.md` — 与顶层目录同名，即该目录的完整生成 skill（原 `CLAUDE.md` 内容）。
- `skills/past-work/SKILL.md` — 项目构建/优化历史与 TODO（原 `.claude/skills/past-work.md`）。
- `.opencode/skills/<name>/SKILL.md` — 全库 skill 的**集中归集处**（opencode 标准约定，由 init 技能维护）：散落于各 `skills/`、`cpp/*/skills/`、`examples/skills/`、`logs/skills/`、`pyqcu/skills/` 的 SKILL.md 均合并于此；同名不同内容并列保留（`<name>.v2`，如 `cann/cann.v2`、`cuda/cuda.v2`、`qcu/qcu.v2`、`pyqcu/pyqcu.v2`、`logs/logs.v2`）。原位置保留不删除。

## 历史说明

原仓库根 `CLAUDE.md` 及其 42 个目录级副本已归档为 `.CLAUDE.md.<时间戳>.bak`（供格式参考）；`.claude/`（含其 2 个 `CLAUDE.md` 与 `skills/past-work.md`）归档为 `.claude.md.<时间戳>.bak`。
