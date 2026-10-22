# PyQCU MultiGrid 弃用与历史残留审计（2026-10-22）

状态：工作区结果，尚未提交或打 tag。审计基线为稳定提交
`9e67890138f8f8eab0ba4a21428c4c20be5f48f1`；当前工作树中的并发
MultiGrid 重构未纳入本审计结论。本次只做静态调用图、测试入口和文档核对，
未运行 GPU、CUDA、MPI 或新的数值 benchmark。

相关性能与完成状态见
[Legacy/Strict/KUPL 状态报告](legacy_vs_strict_mg_status_20261022.md)。

2026-10-23 后续处理：已删除 `pyqcu/testing/qcu/conftest.schur.multigrid.py`、
`conftest.bistabcg.dslash.py`、`bench_cpp_mg.py`，以及 TileLang no-op
`conftest.py` 和 simple `bench_three_way.py` 弃用转发壳；现行替代入口见
`pyqcu/testing/AGENTS.md` 与 `pyqcu/testing/qcu/AGENTS.md`。

## 1. 结论摘要

| 严重度 | 结论 | 建议 |
|---|---|---|
| Important | `pyqcu/testing/tilelang/conftest.py` 只有两行注释，执行退出 0 但不运行测试，却被文档登记为测试入口 | 实现真实入口，或降级为示例并从测试入口索引移除 |
| Minor | 三个 CUDA 旧 standalone harness/benchmark 没有当前文档或测试注册 | 确认无外部使用后归档删除；不要直接静默删除 |
| Minor | 四个 `QUDA` 风格 MultiGrid 公开别名没有仓内调用方 | 作为 API 兼容面保留；若弃用，先补 deprecation notice 和迁移周期 |
| Minor | `pyqcu/cuda/_logs.py` 只被文档引用，当前代码和测试没有导入 | 补定向回归，或明确为外部工具接口并降级其“规范入口”表述 |
| Info | 六个 `PASS` 后端占位目录和 `pyqcu/testing/gpu` 空目录是有意占位 | 保持占位分类，不作为普通死代码删除 |

没有发现可确证的 Critical 问题，也没有发现稳定 C API 中存在“头文件声明但
实现/Cython 包装缺失”的死导出。Legacy MultiGrid、KUPL 的 fail-closed
模式拒绝和 CUDA 测试专用多卡路径均有明确测试或 include 入口。

## 2. 审计范围与方法

审计范围：

- `pyqcu/solver`、`pyqcu/kupl`、`pyqcu/testing/kupl`、`cpp/kupl/qcu`；
- `pyqcu/cuda`、`cpp/cuda/qcu`、`pyqcu/testing/qcu`；
- `cpp/{cann,dtk,maca}/qcu`、`pyqcu/{cann,dtk,maca}`；
- `pyqcu/testing/{cpu,npu,dcu,gpu,tilelang,qcu}` 中的占位和历史入口。

方法：

1. 使用 `git show`/`git grep` 固定稳定 HEAD，核对定义、导出、调用方和测试入口。
2. 区分 active production、test/reference、compatibility、placeholder 和
   dead/unreachable，不因名称含 `legacy` 或 `quda` 就直接判定删除。
3. 对每个删除候选记录“是否有文档入口、是否有测试、是否有导入/构建引用”三类证据。

## 3. Findings

### 3.1 Important：TileLang 测试入口是空操作

`pyqcu/testing/tilelang/conftest.py` 仅有：

```python
# from pyqcu.testing import *

# test_matmul()
```

但 `pyqcu/testing/tilelang/AGENTS.md` 将其描述为 TileLang 测试入口，并给出
`python pyqcu/testing/tilelang/conftest.py`。实测该命令退出码为 `0` 且无输出，
说明“入口存在”不等于“测试执行”。这是当前最明确的事实性历史残留。

建议二选一：

- 恢复一个真实、可用 `PYQCU_ENABLE_*` 或硬件门控的测试入口；
- 将文件降级为示例，移除 AGENTS/技能中的测试入口描述，并补明确的 `SKIP` 原因。

### 3.2 Minor：旧 CUDA standalone harness/benchmark

以下稳定 HEAD 文件没有出现在 `pyqcu/testing/qcu/AGENTS.md` 的当前测试表，
也没有其他代码、文档或测试注册引用：

| 文件 | 事实 | 建议 |
|---|---|---|
| `pyqcu/testing/qcu/conftest.schur.multigrid.py` | 214 行旧 Schur MG 手工 harness；仅自身 docstring 给出运行命令，且部分 `applyInitQcu` 路径没有对应 `applyEndQcu` | 归档删除候选；其能力已由 `single_qcu_*` 和现行 MG 套件覆盖 |
| `pyqcu/testing/qcu/conftest.bistabcg.dslash.py` | 旧 manual Wilson/Clover dslash 自测，未登记 | 归档删除候选；保留现行 `single_qcu_wilson_dslash.py`/`single_qcu_clover_dslash.py` |
| `pyqcu/testing/qcu/bench_cpp_mg.py` | 旧 standalone benchmark，直接写 `logs/bench_report.json`，未登记 | 移入规范 benchmark 目录或归档删除 |

`conftest.schur.multigrid.py:119-128` 和 `:153-158` 直接修改全局
`params`/`set_ptrs` 并调用 `applyInitQcu`，与当前 `QcuRuntime` 事务生命周期
规范不一致。该文件可以作为“历史手工脚本”保留，但不应继续被描述为当前
测试入口。

`pyqcu/testing/qcu/conftest.multigrid.mt.py` 不列入删除候选：它只是对
`pyqcu.testing.test_multigrid_multithread` 的显式手工包装，后者仍有测试和
回归入口。

### 3.3 Minor：QUDA 风格公开别名没有仓内调用方

`pyqcu/solver/_quda_multigrid.py` 在稳定 HEAD 导出：

- `quda_multigrid = QudaMultigrid`
- `QUDAMultigrid = QudaMultigrid`
- `quda_strict_multigrid = QudaStrictMultigrid`
- `QUDAStrictMultigrid = QudaStrictMultigrid`

这些名称被 `pyqcu/solver/__init__.py` 再导出，但仓内没有实际调用方。
它们不是立即删除项：属于公开 Python API 兼容面，且 `QudaMultigrid`
默认 `hierarchy_mode="legacy"`，不能与 `QudaStrictMultigrid` 混为一谈。
若统一命名，建议新名称为 `LegacyMultigrid`、`StrictMultigrid` 和
`Multigrid`，旧别名至少保留一个 deprecation cycle。

### 3.4 Minor：日志解析模块缺少仓内调用方

`pyqcu/cuda/_logs.py` 提供 `parse_mg_log` 和
`parse_convergence_histories`，文档称其为规范日志解析层；稳定 HEAD 中没有
Python 代码或测试导入该模块。它不应仅因“私有模块且无调用方”立即删除，
因为文档可能把它定位为外部诊断工具。建议补一个最小纯文本日志解析测试，
并将文档表述改为“诊断工具 API”；若确认没有外部用户，再进入删除流程。

### 3.5 Info：占位目录是明确的预留

以下目录稳定 HEAD 中只含 `AGENTS.md` 和零字节 `PASS`，没有 Python 导入、
构建脚本引用或测试收集入口：

- `cpp/cann/qcu/`
- `cpp/dtk/qcu/`
- `cpp/maca/qcu/`
- `pyqcu/cann/qcu/`
- `pyqcu/dtk/`
- `pyqcu/maca/`

这些目录的目录级 AGENTS 明确写明“占位/暂无实现”。因此它们是
placeholder，不是可证明的 production dead code。只有在项目决定取消对应
后端预留时，才应连同 AGENTS、索引和构建说明一起删除。

`pyqcu/testing/gpu/` 也只有 `AGENTS.md` 和 `__init__.py`，文档明确说明
“当前为空，通用 GPU 测试预留”。`pytest` 收集不到用例是预期状态。

## 4. 不应误判为弃用的路径

| 路径 | 保留理由 |
|---|---|
| `solver.multigrid` / `pyqcu/solver/_multigrid.py` | Python Legacy/参考路径；测试、回归和多 GPU 测试仍调用 |
| `QudaMultigrid` | 默认 Legacy 层级，仍有多组测试；不能因名称含 QUDA 删除 |
| `QudaStrictMultigrid` / `hierarchy_mode="strict"` | 当前 Strict full-coarse 路径的显式入口 |
| `setup_operator="schur"` | Legacy compact odd-Schur 选项，仍有测试和诊断用途；不是 Strict 选择器 |
| KUPL CUDA-only 模式拒绝 | 对 GCR/cycle 和混合精度的 fail-closed 行为，已有 failure-mode 回归 |
| `lattice_sap.h` 与 legacy MG helper kernels | 被 `qcu.h`/`lattice_clover_multigrid.h` 实际 include 并由 Legacy MG 使用 |
| `MultiGpuMultigrid` | 显式 opt-in 的测试专用多卡例外；不是生产拓扑，但仍有独立 benchmark/测试 |

## 5. Benchmark 状态摘要

完整证据和数据边界见
[Legacy/Strict/KUPL 状态报告](legacy_vs_strict_mg_status_20261022.md)。
以下是审计用摘要，均来自现有 JSON/stdout，不来自本次重跑：

| 格点 | Legacy build | Legacy solve | Strict null-prep | Strict solve |
|---|---|---|---|---|
| 8³×16 | PASS | PASS | created | PASS |
| 12³×24 | PASS | PASS | created | PASS |
| 16³×32 | PASS | PASS | created | PASS |
| 24³×64 | PASS | PASS | created | 缺失：空 stdout，无 JSON |

已成功的 Strict 记录与 Legacy 记录的稳态中位耗时如下。该表只描述已记录路径的
wall 差异，不是算法效率或“MG 加速比”：

| 格点 | Legacy median (s) | Strict median (s) | Strict/Legacy |
|---|---:|---:|---:|
| 8³×16 | 0.0115697170 | 0.0240973420 | 2.083 |
| 12³×24 | 0.0115448180 | 0.0461964710 | 4.001 |
| 16³×32 | 0.0233167160 | 0.1042182120 | 4.470 |
| 24³×64 | 0.1111170500 | 缺失 | 不可计算 |

关键边界：

- Legacy 四组 solve 的 stdout 均记录 `n_vcycles=0`、`coarse_iters=0`，
  不能宣称执行了 coarse V-cycle；
- Strict JSON 没有 V-cycle 计数器，`restart=20` 不是 20 次 V-cycle；
- KUPL 在目标报告目录没有 build、null-prep 或 solve 记录，不能计算跨后端倍数；
- `restart=0` 关闭 coarse hierarchy，是 fine-level parity-Schur BiCGStab，
  不是 coarse V-cycle。

## 6. 验证

本次文档任务执行：

- `git diff --check`：通过；
- `bash skills/pyqcu-form/form-audit.sh`：通过；
- 新文档 Markdown 结构、相对链接和代码围栏检查：通过；
- benchmark 数据断言：29 个 JSON/stdout 记录与状态报告一致；
- GPU/CUDA/MPI 运行：未执行。

## 7. 限制与后续动作

1. 当前工作树存在并发 MultiGrid 重构，审计结论仅对应稳定 HEAD
   `9e678901`；重构稳定后必须重跑调用图和测试入口审计。
2. 删除候选只记录，不在本次任务中删除；删除前需确认外部用户、历史 tag
   归档和文档引用。
3. `TileLang` 空入口应优先处理，因为它会让测试命令产生假阳性。
4. 若统一 `QUDA` 命名，应先设计公开 API 迁移，而不是直接删除别名。

## 8. 证据索引

- `9e678901:pyqcu/solver/_quda_multigrid.py:2275`：Legacy 默认层级；
- `9e678901:pyqcu/solver/_quda_multigrid.py:2362-2365`：Schur 与 Strict 分离；
- `9e678901:pyqcu/solver/_quda_multigrid.py:3838-3859`：Strict 类与兼容别名；
- `9e678901:pyqcu/solver/__init__.py:5-19`：Legacy 与 QUDA 风格导出；
- `9e678901:pyqcu/testing/__init__.py:655-680`：多线程 Legacy MG 测试入口；
- `9e678901:pyqcu/testing/qcu/conftest.schur.multigrid.py:119-128,153-158`：
  旧手工生命周期；
- `9e678901:pyqcu/testing/qcu/conftest.multigrid.mt.py:1-5`：现行多线程测试包装；
- `9e678901:cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:151-153,793-796`：
  KUPL GCR/cycle fail-closed；
- `9e678901:cpp/kupl/qcu/src/lattice_clover_multigrid.cpp:207-209`：
  KUPL 混合精度拒绝；
- `9e678901:pyqcu/cuda/_logs.py:21-56`：日志解析函数；
- `9e678901:pyqcu/cuda/qcu/qcu.pyx:66-84,195,733`：CUDA 测试/迭代/verify 包装；
- `9e678901:pyqcu/testing/qcu/single_qcu_api.py:8-37`：C API 到现行单功能入口映射；
- `9e678901:pyqcu/testing/tilelang/conftest.py:1-2`：TileLang 空入口；
- `9e678901:pyqcu/testing/gpu/AGENTS.md:3`：GPU 测试占位声明；
- `9e678901:cpp/cann/qcu/PASS` 等六个零字节占位文件。
