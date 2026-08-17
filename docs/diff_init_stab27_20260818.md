# PyQCU 改动概览（init → stab27）

> 本文件由 `diff` 技能生成，对 PyQCU 仓库从**初始提交**到**最新标签 `stab27`** 的全部改动做只读结构化概览。
> 查看范围：`0204fa56d6d63d80938468bd1d5006e44a419e0d` (init) → `stab27`（2026-08-18）。
> 纯只读操作，未修改/暂存/提交任何文件。

## 改动概览

```
✓ 改动概览
    范围:   初始提交 0204fa5 (init) → 标签 stab27（全历史）
    提交:   区间内 1120 个提交
    统计:   2138 个文件, +1,163,982 / -2 行
    分组:   新增(全量) / 外部镜像 refer/ 1559 文件 / 测试与产物 examples+logs 384 文件
    主导:   refer/git-rep/(外部库镜像) 与 examples/pyqcu/*.txt(运行日志) 占增行绝大部分
    自有库: pyqcu/ 47 文件 +6847 行；cpp/ 78 文件 +16234 行（真·库代码演进）
    关注:   冲突标记 0；二进制 170；未检测重命名
```

## 规模与分布（实测）

| 顶层目录 | 文件数 | 增行 | 说明 |
|---|---:|---:|---|
| `refer` | 1559 | 385,023 | 外部库镜像（DDalphaAMG / PyQUDA / quda 等），非 PyQCU 自有代码 |
| `examples` | 136 | 723,046 | 测试入口与运行日志（`*.txt` 占主导，见边界检查） |
| `logs` | 261 | 29,394 | 测试/调试产物归档（`.gitignore` 豁免入库，但历史已提交） |
| `cpp` | 78 | 16,234 | C++ CUDA 后端（手写内核 + MPI halo 交换） |
| `pyqcu` | 47 | 6,847 | 纯 Python 实现 + Cython 桥 + cann 兼容层 |
| `.opencode` | 36 | 2,493 | opencode 运行环境（node_modules 等），与库无关 |
| `docs` | 12 | 730 | 文档与既有分析报告 |
| 根文件 | 8 | 215 | `setup.py`/`env.sh`/`build.sh`/`AGENTS.md` 等 |
| 合计 | 2138 | 1,163,982 | — |

## PyQCU 自有库代码演进（剔除 refer/ .opencode/ 噪声）

真正的库代码集中在两个目录：

- **`pyqcu/`（47 文件, +6847）** 子目录分布：

  | 子目录 | 文件数 | 增行 |
  |---|---:|---:|
  | `tools` | 9 | 1846 |
  | `cuda` | 11 | 1604 |
  | `dslash` | 5 | 1099 |
  | `solver` | 5 | 749 |
  | `testing` | 2 | 697 |
  | `smear` | 3 | 248 |
  | `cann` | 4 | 286 |
  | `lattice` | 2 | 250 |
  | 其他 (`dtk`/`maca`/根) | 7 | 68 |

- **`cpp/`（78 文件, +16234）**：CUDA 后端头文件与内核。

### 主导文件（自有库，按增行）

| 文件 | 增行 | 角色 |
|---|---:|---|
| `cpp/cuda/qcu/include/lattice_clover_multigrid.h` | 1833 | Clover 多重网格核心头 |
| `cpp/cuda/qcu/src/clover_dslash_multi.cu` | 1610 | 多重网格 Clover dslash 内核 |
| `cpp/cuda/qcu/src/wilson_dslash.cu` | 1442 | Wilson dslash 内核 |
| `cpp/cuda/qcu/src/multigrid.cu` | 981 | 多重网格 V-cycle 内核 |
| `cpp/cuda/qcu/src/clover_dslash_single.cu` | 918 | 单层 Clover dslash 内核 |
| `cpp/cuda/qcu/include/lattice_set.h` | 764 | `LatticeSet` 运行时集合（scratch 管理） |
| `cpp/cuda/qcu/src/laplacian.cu` | 744 | Laplacian/Gauss gauge 内核 |
| `pyqcu/tools/_multigrid.py` | 819 | 纯 Python 多重网格工具 |
| `pyqcu/testing/__init__.py` | 658 | 集成测试套件 |
| `pyqcu/cuda/_multi_gpu.py` | 568 | 一线程一卡多线程多卡 MG 驱动 |
| `pyqcu/solver/_multigrid.py` | 558 | 纯 Python 多重网格求解器 |
| `pyqcu/tools/_define.py` | 501 | 参数协议常量定义 |
| `pyqcu/dslash/_operator.py` | 362 | 算子基类 |
| `pyqcu/dslash/_clover.py` | 345 | Clover 算子 |
| `pyqcu/dslash/_wilson.py` | 324 | Wilson 算子 |
| `pyqcu/cuda/qcu/qcu.pyx` | 215 | Cython 桥（生产路径） |

## 边界检查（易漏项）

- **冲突标记**：`git diff | grep '^<<<<<<<|^=======|^>>>>>>'` → **0**（无合并冲突残留）。
- **调试残留**：主导增量文件为 `examples/pyqcu/log-v*.txt`（运行日志，非源码），无 `print/echo` 调试行混入库代码的确证。
- **未跟踪文件**：本视图为版本间 diff，不含未跟踪内容；运行日志类属已跟踪历史产物。
- **二进制/大文件**：`git diff --numstat` 首字段为 `-` 的共 **170** 个，主要为 `refer/git-rep/` 内外部库二进制与 `examples/` 数据/`.ipynb`，非 PyQCU 可读源码。
- **权限位变化**：`--summary` 未报告 mode 变化。
- **意外删除**：全量仅 **2 行删除**（`-2`），无结构性删除，属历史清理。
- **重命名**：`git diff -M` 未将任何"删+增"合并为重命名，结构演进以新增为主。

## 结论与后续

- 从 init 到 `stab27`，PyQCU 完成了从空仓库到完整 Lattice QCD 库的建设：**核心增量在 `cpp/`（CUDA 后端）与 `pyqcu/`（Python 实现 + Cython 桥 + 多线程多卡 MG）**，主导文件直接对应 Wilson/Clover dslash、多重网格与测试体系。
- `refer/`（外部库镜像，1559 文件）与 `examples/*.txt`、`logs/` 属于参考/产物，不应计入"库自有代码"规模评估。
- 后续建议：
  1. 将 `logs/`、`examples/pyqcu/log-*.txt` 等运行产物移出版本跟踪（已在 `.gitignore` 豁免但历史已提交，可考虑 `git rm --cached` 清理历史包袱）；
  2. `refer/git-rep/` 体积庞大，建议以 submodule 或外部引用替代整库镜像；
  3. 版本对比如需聚焦"库代码"，可用 `git diff 0204fa5 stab27 -- pyqcu/ cpp/` 排除噪声。

> 注：涉及 git 跟踪内容如需提交/清理，请用户自行操作（本技能不代提交）。
