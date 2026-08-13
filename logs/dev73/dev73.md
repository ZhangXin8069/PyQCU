# PyQCU Development Report — stab22 → dev73

> 对照 tag `stab22`（2026-07-05），截至 2026-07-28 的全库变更总结  
> 涵盖：性能优化 · 新功能 · Bug 修复 · 代码审查 · 文档 · 测试

---

## 目录

1. [时间线概览](#1-时间线概览)
2. [性能优化（07-08）](#2-性能优化07-08)
3. [CLQCD 报告与参考资料（07-17）](#3-clqcd-报告与参考资料07-17)
4. [C++ CUDA 多网格后端（07-17→07-28）](#4-c-cuda-多网格后端07-1707-28)
5. [基础设施与文档（07-14→07-28）](#5-基础设施与文档07-1407-28)
6. [全面代码审查（07-28）](#6-全面代码审查07-28)
7. [Bug 修复总表](#7-bug-修复总表)
8. [文件变更统计](#8-文件变更统计)
9. [测试验证结果](#9-测试验证结果)

---

## 1. 时间线概览

| 日期 | 里程碑 | 关键变更 |
|------|--------|---------|
| 07-05 | **stab22** | 基线标签 |
| 07-08 | 性能优化 | 12 项优化，6 文件，+515/−110 行 |
| 07-08 | CLAUDE.md 初始化 | 90 行项目文档 |
| 07-14 | CLAUDE.md 扩展 | +36 行，补充架构细节 |
| 07-17 | CLQCD 报告 | `refer/dev71.md`(861行) + `.pdf` + `.tex`(853行) |
| 07-17 | C++ CUDA MG 骨架 | `lattice_multigrid.h`(59行), `multigrid.cu`(232行), Python MG CUDA 桥接 |
| 07-19 | MG 后端修复 | `multigrid.cu` 重构，Python 层 CUDA 加速修复 |
| 07-20 | 基础设置 | PASS 标记, skills/past-work.md(115行), CLAUDE.md 大修(183行) |
| **07-28** | **大规模审查+修复** | Clover MG 求解器, 3 轮审查 135 项发现, 33 个 Bug 修复 |

---

## 2. 性能优化（07-08）

> 来源: `OPTIMIZATION_REPORT.md` → `log/stab23.log`（commit `a3c3e96`）

**修改文件**: 6 个 | **代码变更**: +120 / -110 行

### 2.1 批量矩阵求逆 (`pyqcu/dslash/_clover.py`)

**影响：高** — O(N) Python 循环 → 单次 GPU 批量调用。

`inverse()` 原使用 Python for 循环逐一求 12×12 复矩阵的逆。优化后调用 `torch.linalg.inv` 批量处理（`permute(2,0,1)` 将格点维度提升为 batch），预计 L=32 格点上 **10–50x** 加速。

```python
# 优化前：逐格点循环
for i in range(_clover_term.shape[-1]):
    _clover_term[:, :, i] = torch.linalg.inv(_clover_term[:, :, i])

# 优化后：批量求逆
_clover_term = torch.linalg.inv(_clover_term.permute(2, 0, 1)).permute(1, 2, 0)
```

### 2.2 张量设备/类型缓存 (`pyqcu/dslash/_wilson.py`)

**影响：高** — 消除每方向重复的 `.to()`/`.type()` 分配。

在 Wilson dslash 入口处预计算 `I ± γ` 矩阵字典，消除 4 方向 × 多次转换的 GPU 显存分配。同样应用于 `give_wilson_eo`, `give_wilson_oe`, `give_hopping_plus`, `give_hopping_minus`。

### 2.3 预计算 sigma 矩阵与 clover 系数 (`pyqcu/dslash/_clover.py`)

- **sigma 矩阵缓存**: 6 个 γγ 矩阵预计算到设备，消除 6 次重复 `.to()` 调用
- **clover 系数**: `float(kappa/u_0)` 从循环内提到循环外

### 2.4 内存优化 (`pyqcu/dslash/_clover.py`)

移除 `add_I()`, `cut_I()`, `inverse()` 中不必要的 `.clone()` 调用。每次调用节省约 72 MB（12×12×32⁴×8 bytes）内存分配。

### 2.5 缓存 `give_eo_mask` (`pyqcu/tools/_define.py`)

引入 `_eo_mask_cache` 字典，按 `(Lx, Ly, Lz, Lt, eo, device)` 键缓存 checkerboard mask，避免重复 `meshgrid` 创建。

### 2.6 求解器优化 (`pyqcu/solver/_bistabcg.py`)

- 存储 `norm(b)` 避免重复 MPI Allreduce
- `verbose=False` 时跳过 `perf_counter()` 调用
- 移除 `_multigrid.py` 重复 import
- 移除 `_linalg.py` 中 `norm()` 的冗余 `.flatten()`

### 2.7 日志修复 (`pyqcu/dslash/_clover.py`)

`cut_I()` 日志从错误的 "Clover is adding I" 修正为 "Clover is cutting I"。

### 预期性能提升总览

| 场景 | 预期加速 |
|------|----------|
| Clover 求逆 (L=32) | **10–50x** |
| Wilson Dslash 单次 | **1.5–2x** |
| Clover 项构建 | **1.2–1.5x** |
| BiStabCG silent 模式 | 每次迭代 -2% |
| Parity Dslash | mask 计算 -5% |
| Clover add_I/cut_I | 每次调用 -70 MB 峰值 |

---

## 3. CLQCD 报告与参考资料（07-17）

> 来源: `refer/dev71.md`, `refer/dev71.tex`, `refer/dev71.pdf`（commits `fcf6e9d`, `34c71c6`）

创建 CLQCD 会议/论文参考资料，标题 **"CUDA C++ 版 MultiGrid 编写与优化解析文档"**（861 行 Markdown + 853 行 LaTeX + 编译 PDF 345 KB）。

内容涵盖：
- 总体架构概览（Python/C++ 双层设计）
- Python 层 MultiGrid 完整实现分析（336 行源码步行）
- C++ CUDA 后端现有基础设施（`lattice_set.h`, `lattice_wilson_bistabcg.h`, `lattice_clover_bistabcg.h` 等）
- CUDA C++ MultiGrid 设计方案（V-cycle 流程、流并行策略、通信模式）
- 优化策略（内存合并、共享内存、warp shuffle、寄存器优化）
- 实现路线图（Phase 1–4）

该文档作为后续 C++ Clover Multigrid 实现的技术规格。

---

## 4. C++ CUDA 多网格后端（07-17 → 07-28）

### 4.1 初始骨架（07-17, commit `a221237`）

新增文件：
- `cpp/cuda/qcu/include/lattice_multigrid.h` — 59 行，多网格类声明
- `cpp/cuda/qcu/include/multigrid.h` — 20 行，multigrid 参数结构体
- `cpp/cuda/qcu/src/multigrid.cu` — 232 行，V-cycle + 粗网格算子实现
- `cpp/cuda/qcu/src/apply_multigrid.cu` — 125 行，C API 桥接函数

修改文件：
- `cpp/cuda/qcu/python/pyqcu.h` — 新增 9 个 C 函数声明
- `pyqcu/cuda/qcu/qcu.pyx` / `qcu.pxd` — Cython 桥接新增函数
- `pyqcu/solver/_multigrid.py` — 111 行新增 CUDA 加速路径

核心能力：
- `applyMultigridRestrictQcu` — CUDA 限制算子
- `applyMultigridProLongQcu` — CUDA 延拓算子
- `applyMultigridCoarseDslashQcu` — CUDA 粗网格 Dirac 算子
- Python 层 `with_cuda_qcu=True` 自动启用 CUDA 加速

### 4.2 MG 后端修复（07-19, commit `b122fda`）

- `cpp/cuda/qcu/src/multigrid.cu` — 重构 MPI 通信和内存管理（+43/-31 行）
- `pyqcu/solver/_multigrid.py` — CUDA 路径错误处理和流同步修复（+54/-12 行）

### 4.3 Clover Multigrid 求解器（07-28, commit `4a41814`）

**重大新增**：完整的 C++ CUDA Clover 多网格求解器。

新增核心文件：
- `cpp/cuda/qcu/include/lattice_clover_multigrid.h` — 953 行，主求解器类
- `cpp/cuda/qcu/src/apply_clover_multigrid.cu` — 72 行，C API 桥接
- `examples/qcu/conftest.clover.multigrid.py` — 340 行测试驱动脚本

关键设计决策：

| 特性 | 说明 |
|------|------|
| 流并行 | 5 个 CUDA 流（`strm` + `_a_`/`_b_`/`_c_`/`_d_`），匹配 `LatticeCloverBistabCg` 模式 |
| 标量管理 | `device_vals` 仅被 GPU kernel 修改，无 host→device memcpy |
| 同步 | 每轮迭代底部 5 流全同步，顶部 5 流全同步 |
| dot 协议 | cublasDot → `_send_tmp_` → MPI_Allreduce → 目标槽位 |
| 精度 | `mpi_real_type<T>()` 模板根据 float/double 选择 `MPI_FLOAT`/`MPI_DOUBLE` |

**测试结果**（NVIDIA GeForce RTX 4060 Laptop, SM 8.9）：

![multigrid_result.png](multigrid_result.png)

| 指标 | 数值 |
|------|------|
| 正确性 (‖x_mg−x_ref‖/‖x_ref‖) | **6.23×10⁻⁷** ✅ |
| NaN | **0** ✅ |
| 收敛率 | ~104 次迭代到 5.7×10⁻⁷ |
| vs BiStabCG | 0.87×–1.28× (随运行变化) |
| 求解时间 | 1742–3636 ms (早冷启动→稳定) |

收敛日志示例（`clover_multigrid.log`）：
```
2026-07-28 12:18:09 | PYQCU::SOLVER::MULTIGRID::
 0:Norm of b:247.299469
 B-0-bistabcg-Iteration 0: Residual = 2.472995e+02
 F-0-bistabcg-Iteration 0: Residual = 2.472995e+02, Time = 0.016208 s
 ...
```

### 4.4 Clover MG Bug 修复（07-28, commit `514d135`）

修复了 Clover Multigrid 求解器中发现的 5 个关键 Bug + 1 个预存 Bug：

| # | Bug | 严重度 | 症状 | 修复 |
|---|-----|--------|------|------|
| 1 | `run_mpi` 未初始化 `MPI_Wait` | 🔴🔴 | 立即 segfault | 从 `run_mpi` 路径移除 `MPI_Wait`（阻塞式不需要） |
| 2 | `device_vals` 竞争条件 → NaN | 🔴🔴 | 残差 4-6 次迭代后变 NaN | 移除迭代循环内所有 host→device 标量 memcpy |
| 3 | 底部流同步缺失 | 🔴 | 残差 3-12x 跳跃 | 底部 5 流全同步 |
| 4 | cublasDot 目标槽位损坏 | 🟡 | 偶发残差跳跃 | 写入 `_send_tmp_` → MPI 后复制到目标 |
| 5 | `MPI_FLOAT` 硬编码 | 🔴 | double 精度结果错误 | `mpi_real_type<T>()` 模板 |
| 6 | `wilson_dslash.cu` 变量名错误 | 🔴🔴 | 8 处编译失败 | `idx` → `parity` |

详细信息见 [`multigrid_report.md`](multigrid_report.md)。

---

## 5. 基础设施与文档（07-14 → 07-28）

### 5.1 CLAUDE.md 演进

| 日期 | 行数 | 主要变化 |
|------|------|---------|
| 07-08 | 90 | 初始化：项目结构、构建命令、架构概述 |
| 07-14 | +36 | GPU 后端抽象、测试运行方式、模块级代码说明 |
| 07-20 | +183 (大修) | 全面重写：两层架构图、数据布局约定、C++ 后端计划系统、Cython 桥接、TileLang 集成、`cann` 层文档 |
| 07-28 | +66 | 方法列表细化、`pyqcu.cann` 详细文档、GMRES stub 状态、ward 索引惯例注释 |

### 5.2 其他基础设施

- **`.gitignore`** (07-17): 新增 9 条忽略规则
- **`cpp/*/PASS`** (07-20): CANN/CUDA/DTK/MACA 四个后端放置 PASS 标记文件
- **`.claude/skills/past-work.md`** (07-20): 115 行历史工作总结，供 AI 助手上下文
- **`build.sh` / `make.sh`** (07-28): 添加 `set -e` 错误检测，`&&` 链式调用，`rm -f` 安全删除
- **`pyqcu/cuda/__init__.py`** (07-28, 新建): 修复 pip install 后 CUDA 包不可用问题
- **`pyqcu/cuda/qcu/qcu.pyi`** (07-28, 新建): 155 行类型桩文件，包含所有 22 个 Cython 桥接函数签名

---

## 6. 全面代码审查（07-28）

### 6.1 审查流程

```
Round 1 (74 findings)     →   12 bug fixes + 1 optimization
    ↓
Round 2 (28 findings)     →   8 bug fixes + 1 optimization + 5 refinements + 6 docs
    ↓
Round 3 (33 findings)     →   13 fixes + 2 docs
    ↓
Total: 135 findings       →   33 bugs fixed + 2 optimizations + 12 docs + 13 false positives
```

审查报告文件：
- [`review-2026-07-28.md`](review-2026-07-28.md) — Round 1 审查报告（892 行）
- [`review-2026-07-28-r2.md`](review-2026-07-28-r2.md) — Round 2 审查报告（460 行）
- [`review-2026-07-28-r3.md`](review-2026-07-28-r3.md) — Round 3 审查报告（460 行）

修复日志：
- [`debug/fix-log.md`](debug/fix-log.md) — Round 1 修复（48 行）
- [`debug/fix-log-r2.md`](debug/fix-log-r2.md) — Round 2 修复（82 行）
- [`debug/fix-log-r3.md`](debug/fix-log-r3.md) — Round 3 修复（123 行）
- [`fix-report-2026-07-28.md`](fix-report-2026-07-28.md) — 完整修复清单（210 行）

### 6.2 审查方法

- **Round 1**: 逐文件阅读，交叉引用 CLAUDE.md 文档
- **Round 2**: 三轮并行 agent 审查 + 人工深度追踪关键算法路径（Python core + C++ backend + Cython/tests/docs + deep-dive）
- **Round 3**: 四路并行 agent（跨模块一致性 + 物理/数值正确性 + 构建/测试/边界 + 人工 multigrid cycle 追踪）

### 6.3 发现的假阳性（误报）

13 项经深度分析确认为设计决策而非 Bug：

| 项目 | 审查发现 | 分析结论 |
|------|---------|---------|
| Gamma 矩阵代数 | 怀疑平方/反对易关系错误 | ✅ 所有 5 个 gamma 矩阵恒等式正确 |
| Wilson dslash 符号 | 怀疑方向索引排列错误 | ✅ 所有 22 个 einsum 方程符号正确 |
| BiCGStab 算法 | 怀疑与 van der Vorst 1992 不符 | ✅ 精确匹配标准算法 |
| MG V-cycle | 怀疑 Galerkin 修正和宇称分解错误 | ✅ 修正和分解逻辑完全正确 |
| C++ host_vals 同步 | 怀疑竞争条件 | ✅ `_dot_mpi` 正确使用 D2H→MPI→H2D 顺序 |
| C++ BiCGStab 缓冲区泄漏 | 怀疑 GPU buffer 泄漏 | ✅ 在 `end()` 中正确释放 |
| MPI_FLOAT 硬编码 | 怀疑精度错误 | ✅ 已在早期修复（`mpitype<T>()` 模板） |
| 导入图 | 怀疑循环导入 | ✅ 无环（通过部分模块加载实现） |
| .pxd vs .pyx | 怀疑签名不匹配 | ✅ 所有 22 个函数签名一致 |
| 错误传播 | 怀疑 vdot 错误丢失 | ✅ 错误正确传播 |
| 格点模块级状态 | 怀疑设备放置问题 | ✅ 正确使用 init-on-CPU, move-on-demand 模式 |
| HDF5 I/O | 怀疑资源泄漏 | ✅ 正确使用 `with` 语句 |
| ward 负索引 | 怀疑索引错误 | ✅ 有意的设计模式（适应任意前缀维度） |

---

## 7. Bug 修复总表

### 7.1 🔴🔴 致命 Bug（5 个，全部修复）

| # | 文件 | 问题 | 症状 | 轮次 |
|---|------|------|------|------|
| F1 | `lattice_complex.h:79-83` | `operator*=` 复数乘法：`_data.x` 被覆盖后影响 `_data.y` | 所有复数乘法结果错误 | R1 |
| F2 | `gauss_gauge.cu:183-186` | 非 verbose 路径每 site 仅分配 4 元素，核函数写入 32 元素 | 越界写入 (OOB Write) | R1 |
| F3 | `lattice_wilson_cg.h:41-51` | CG `_init()` 重复分配 `device_vec0/1/2` + `device_vals` | GPU 内存泄漏 | R2 |
| F4 | `lattice_set.h:140-163` | Grid dim 整数除法截断 | 格点遗漏 | R2 |
| F5 | `pyqcu/cuda/` 无 `__init__.py` | `pip install` 后 CUDA 包不可用 | `ImportError` | R3 |

### 7.2 🔴 严重 Bug（14 个，全部修复）

| # | 文件 | 问题 | 轮次 |
|---|------|------|------|
| S1 | `_io.py:62,69` | gather 元组 `(t,z,y,x)` → unpack `(x,y,z,t)` 索引错位 | R1 |
| S2 | `_stout.py:64-172` | `nstep>1` 无效：每次迭代使用原始 U | R1 |
| S3 | `_stout.py:22-63` | MPI 边界数据在多次 smearing 中不更新 | R1 |
| S4 | `cann/__init__.py:81-83` | NPU 3+ operand 复数 einsum 丢弃虚部 | R1 |
| S5 | `_operator.py:228,259` | `self.sitting` 对象始终 truthy，条件判断无效 | R1 |
| S6 | `_define.py:97-101` | `check_mpi_support` 非 root 进程临时文件泄漏 | R1 |
| S7 | `testing/__init__.py:363` | 错误消息使用模块对象而非参数 | R1 |
| S8 | `setup.py:1,3` | distutils `setup` 覆盖 setuptools `setup` | R1 |
| S9 | `qcu.pyx:2-13` | 模块级 `cdef` 变量 GIL 依赖风险 | R1 |
| S10 | `_operator.py:333-334` | 遗留调试 `print()` | R1 |
| S11 | `_stout.py:131-136` | `c1=0 → c0_max=0 → arccos(0/0)` → NaN | R2 |
| S12 | `_stout.py:149` | `f_denom` 除以零（`9u²=w²` 时） | R2 |
| S13 | `_stout.py:155-162` | NPU f1 宇称缺失实部取反 | R2 |
| S14 | `smear/_stout.py:160-172` | R2 f1/f2 宇称修复的虚部符号反转回归 | R3 |

### 7.3 🟡 中等 Bug（13 个，全部修复）

| # | 文件 | 问题 | 轮次 |
|---|------|------|------|
| M1 | `_bistabcg.py:66` | `verbose=False` 时 ZeroDivisionError | R1 |
| M2 | `_bistabcg.py:66-72` | 性能统计无视 verbose 标志 | R1 |
| M3 | `cuda/define.py:94-119` | `dtype()` 裸 `raise` 无异常消息 | R1 |
| M4 | `setup.py:38` | `find_packages(exclude=["test.*"])` 不匹配 `pyqcu.testing` | R1 |
| M5 | `apply_end.cu` | `LatticeSet` 对象未 delete | R1 |
| M6 | `lattice_wilson_dslash.h` | `MPI_Isend` 从未 `MPI_Wait` | R1 |
| M7 | `lattice_complex.h:89-95` | `operator/=` 与 `operator*=` 同类 bug | R1 |
| M8 | `gauss_gauge.cu` | GPU 内存泄漏（`cudaMallocAsync` 无对应 free） | R1 |
| M9 | `_linalg.py:21,26` | 冗余 `MPI.Barrier` 包围 `Allreduce` | R2 |
| M10 | `_bistabcg.py:38-47` | BiCGStab 无除以零检测（`rho`/`rtv`/`tts`≈0） | R2 |
| M11 | `_multigrid.py:351-359` | MG BiCGStab 同上除以零问题 | R2 |
| M12 | `_multigrid.py:407-420` | MG cycle 粗网格修正后 BiCGStab 状态未重置 | R3 |
| M13 | `setup.py:46` | `python_requires ">=3.6"` 与 PyTorch 2.x 不兼容 | R3 |

### 7.4 🟢 代码质量（6 项修复）

| # | 问题 | 轮次 |
|---|------|------|
| Q1 | `lattice/__init__.py` 重复 `from typing import Optional` | R1 |
| Q2 | 多处使用可变默认参数 `torch.Tensor([0.1])` | R1 |
| Q3 | Gamma 矩阵 ward 负索引惯例添加注释 | R1 |
| Q4 | `check_su3` float32 默认 `atol=1e-8` 过严 | R2 |
| Q5 | `build.sh` / `make.sh` 无 `set -e` 错误检测 | R3 |
| Q6 | 多处 `bare except:` → `except Exception:` | R3 |

### 7.5 🔵 性能优化（2 项）

| # | 优化 | 轮次 |
|---|------|------|
| P1 | 移除 3 个文件中 24 个冗余 `MPI.Barrier()`（阻塞 `Sendrecv` 不需要 Barrier） | R1 |
| P2 | `tools/_multigrid.py` ortho_r/ortho_null_vecs 当 normalize=True 跳过冗余 vdot 分母 | R2 |

### 7.6 测试与类型桩修复（3 项）

| # | 修复 | 轮次 |
|---|------|------|
| T1 | `pyqcu/cuda/qcu/qcu.pyi` — 155 行类型桩，覆盖 22 个函数 | R3 |
| T2 | `pyqcu/testing/__init__.py` — 添加 pytest assert（原只有 print） | R3 |
| T3 | `examples/profiler/conftest.py` — `import comm` → `from mpi4py import MPI as comm` | R3 |

---

## 8. 文件变更统计

> `git diff stab22..HEAD --stat` 结果

```
68 files changed, 9282 insertions(+), 333 deletions(-)
```

### 新增文件（20+ 个）

| 类别 | 文件 | 行数 |
|------|------|------|
| **C++ 后端** | `lattice_clover_multigrid.h` | 953 |
| | `lattice_multigrid.h` | 59 |
| | `multigrid.h` | 20 |
| | `apply_clover_multigrid.cu` | 72 |
| | `apply_multigrid.cu` | 125 |
| | `multigrid.cu` | 232 |
| | `wilson_dslash.cu.bak` | 1462 |
| **文档** | `log/bug30.md` (审查报告) | 892 |
| | `log/review-2026-07-28.md` | 838 |
| | `log/review-2026-07-28-r2.md` | 460 |
| | `log/review-2026-07-28-r3.md` | 460 |
| | `log/fix-report-2026-07-28.md` | 210 |
| | `log/multigrid_report.md` | 144 |
| | `log/debug/fix-log.md` | 48 |
| | `log/debug/fix-log-r2.md` | 82 |
| | `log/debug/fix-log-r3.md` | 123 |
| | `log/results/final-report.md` | 55 |
| | `log/results/remaining-fixes-report.md` | 45 |
| | `refer/dev71.md` | 861 |
| | `refer/dev71.tex` | 853 |
| | `CLAUDE.md` (扩充) | 271 |
| **Cython/类型** | `pyqcu/cuda/qcu/qcu.pyi` | 155 |
| | `pyqcu/cuda/__init__.py` | 10 |
| **测试** | `examples/qcu/conftest.clover.multigrid.py` | 138 |
| **配置** | `.claude/skills/past-work.md` | 115 |
| | `.gitignore` | 9 |
| **图片** | `log/multigrid_performance.png` | 59 KB |
| | `log/multigrid_performance_0.png` | 218 KB |
| | `log/multigrid_result.png` | 73 KB |
| **PDF** | `refer/dev71.pdf` | 345 KB |

### 修改核心文件（20+ 个）

```
pyqcu/dslash/_wilson.py         +170 行  性能优化 + 缓存
pyqcu/smear/_stout.py           +144 行  nstep 修复 + NaN 防护 + MPI 边界
pyqcu/solver/_multigrid.py      +187 行  CUDA 加速 + 除以零检测 + 状态重置
pyqcu/dslash/_operator.py       +63 行  条件修复 + 调试 print 删除
pyqcu/dslash/_clover.py         +58 行  批量求逆 + 预计算
pyqcu/cann/__init__.py          +55 行  3+ operand einsum
pyqcu/lattice/__init__.py       +53 行  check_su3 修复 + ward 注释
pyqcu/cuda/qcu/qcu.pyx          +51 行  MG 桥接函数 + cdef 声明
pyqcu/solver/_bistabcg.py       +45 行  norm(b) 缓存 + 除以零检测
pyqcu/tools/_multigrid.py       +42 行  正交化优化
pyqcu/tools/_define.py          +31 行  eo_mask 缓存 + 异常处理
pyqcu/tools/_io.py              +24 行  索引修复
pyqcu/testing/__init__.py       +20 行  assert + 异常处理
pyqcu/cuda/define.py            +17 行  dtype ValueError
setup.py                        +13 行  python_requires + exclude
```

---

## 9. 测试验证结果

### Round 1 验证（8/8 PASSED）

```
============================================================
  PyQCU Validation Suite — 2026-07-28
============================================================
  PASS: stout_smear nstep>1
  PASS: operator parity
  PASS: BiCGStab
  PASS: BiCGStab parity
  PASS: NPU 2-op einsum
  PASS: NPU 3-op einsum
  PASS: No MPI orphans
  PASS: cuda/define ValueError

  TOTAL: 8/8 passed
  ALL TESTS PASSED
```

C++ 构建: `[100%] Linking CUDA shared library libqcu.so` ✅

### Round 2 验证（8/8 + 10/10 PASSED）

```
PASS: check_su3 float32 (atol fix)
PASS: stout NaN guard (trivial gauge)
PASS: stout_smear nstep>1
PASS: operator parity
PASS: BiCGStab + breakdown guard
PASS: NPU stout parity fix
PASS: NPU einsum 3-op
PASS: vdot after Barrier removal
```

额外验证：
```
PASS: set_device verbose=False
PASS: NPU restrict validation
PASS: ortho_r vdot cache (safe matvec)
```

### Round 3 验证（13/13 PASSED）

```
PASS: R3 NPU stout f1/f2 fix
PASS: pyqcu.cuda __init__.py
PASS: check_su3 atol
PASS: stout NaN guard
PASS: stout nstep>1
PASS: operator parity
PASS: BiCGStab
PASS: MG solve
PASS: test_lattice with assert
PASS: dtype() raises ValueError
PASS: No bare except in _define.py
PASS: vdot Barrier removal
PASS: set_device verbose=False

13/13 passed — ALL R3 FIXES VERIFIED
```

C++ 构建: `libqcu.so: 22.8 MB` ✅

### Clover Multigrid 正确性验证

| 指标 | 结果 |
|------|------|
| 相对残差 (‖x_mg − x_ref‖/‖x_ref‖) | 6.23×10⁻⁷ ✅ |
| NaN 出现次数 | 0 ✅ |
| 收敛到 5.7×10⁻⁷ 的迭代次数 | ~104 |
| 日志格式匹配 | 与 `conftest.clover.multigrid-v20260506.log` 一致 ✅ |

---

## 参考图与参考源

### 图表

| 文件 | 描述 | 大小 |
|------|------|------|
| [`multigrid_performance.png`](multigrid_performance.png) | Clover MG 性能对比 + 收敛历史 | 59 KB |
| [`multigrid_performance_0.png`](multigrid_performance_0.png) | 初始性能测试（高分辨率） | 218 KB |
| [`multigrid_result.png`](multigrid_result.png) | 最终性能对比 + 收敛历史 | 73 KB |

### 报告文档

| 文件 | 行数 | 描述 |
|------|------|------|
| [`bug30.md`](bug30.md) | 892 | Round 1 完整审查报告 |
| [`review-2026-07-28.md`](review-2026-07-28.md) | 838 | Round 1 审查（另一版本） |
| [`review-2026-07-28-r2.md`](review-2026-07-28-r2.md) | 460 | Round 2 审查报告 |
| [`review-2026-07-28-r3.md`](review-2026-07-28-r3.md) | 460 | Round 3 审查报告 |
| [`multigrid_report.md`](multigrid_report.md) | 144 | C++ Clover MG Debug & Optimization |
| [`fix-report-2026-07-28.md`](fix-report-2026-07-28.md) | 210 | 完整修复清单 |
| [`debug/fix-log.md`](debug/fix-log.md) | 48 | R1 修复日志 |
| [`debug/fix-log-r2.md`](debug/fix-log-r2.md) | 82 | R2 修复日志 |
| [`debug/fix-log-r3.md`](debug/fix-log-r3.md) | 123 | R3 修复日志 |
| [`results/final-report.md`](results/final-report.md) | 55 | 最终结果摘要 |
| [`results/remaining-fixes-report.md`](results/remaining-fixes-report.md) | 45 | R2 剩余项修复报告 |

### 参考源

| 文件 | 描述 |
|------|------|
| [`refer/dev71.md`](../refer/dev71.md) | CUDA C++ MultiGrid 编写与优化解析（Markdown, 861 行） |
| [`refer/dev71.tex`](../refer/dev71.tex) | LaTeX 源文件（853 行） |
| [`refer/dev71.pdf`](../refer/dev71.pdf) | 编译 PDF（345 KB） |
| `cpp/cuda/qcu/include/lattice_clover_multigrid.h` | Clover MG 求解器源码（最终版本 1128 行） |
| `examples/qcu/conftest.clover.multigrid.py` | MG 测试驱动脚本（最终版本） |

### 原始日志

| 文件 | 描述 |
|------|------|
| [`clover_multigrid.log`](clover_multigrid.log) | C++ 收敛日志（`PYQCU::SOLVER::MULTIGRID::` 格式） |
| [`clover_multigrid_report.log`](clover_multigrid_report.log) | 性能报告日志（13 次测试运行） |
| [`multigrid_report.json`](multigrid_report.json) | JSON 格式收敛数据 |
| [`test_multigrid.log`](test_multigrid.log) | Python 测试执行日志 |

---

## 三轮修复总结

| 轮次 | 审查发现 | Bug 修复 | 优化 | 文档 | 假阳性 | 累计修复 |
|------|---------|---------|------|------|--------|---------|
| R1 | 74 | 12 | 1 | 4 | 0 | 13 |
| R2 | 28 | 8 | 1 | 4 | 3 | 22 |
| R2 剩余 | — | 0 | 5 | 6 | — | 27 |
| R3 | 33 | 13 | 0 | 2 | 10 | 40 |
| **合计** | **135** | **33** | **7** | **16** | **13** | **40** |

---

*报告生成: 2026-07-28 | 从 stab22 (2026-07-05) 到 dev73 (2026-07-28)*  
*总代码变更: 68 files, +9,282 / −333 lines*
