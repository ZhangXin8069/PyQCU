# AGENTS.md — PyQCU

PyQCU：Lattice QCD 的 Python/Cython 库 —— CUDA 与鲲鹏/NUMA CPU 加速的
Wilson/Clover Dirac 算子、BiStabCG 与多重网格求解器、stout smearing、
规范场生成，MPI 分布于 4D 进程网格。

> 2026-09-27 测试迁移：顶层 `examples/` 已整体删除，原内容按 backend 分类迁入
> `pyqcu/testing/`（`pyqcu`、`qcu`、`quda`、`pyquda` 等）。新代码、文档和命令
> 必须使用新路径；历史 `logs/` 中的旧路径仅作为当时记录保留。
>
> 2026-09-27 文件分类：独立文档的规范位置是 `docs/**`，当前任务尚未归档的
> 日志与阶段报告位于 `logs/**`。阶段结果先提交并打 tag 归档，随后在后续清理
> 提交中删除工作区报告副本；标签是历史报告的唯一规范归档，通过
> `git show <tag>:<path>` 读取。HDF5/SO/DAT/CSV/SVG 及构建安装树留在
> `data/**`，测试及其他代码统一进入 `pyqcu/testing/**`。
> 完整规则见 `docs/ORGANIZATION.md`。

## 概要命令

- 环境：`source ./env.sh`（LD_LIBRARY_PATH、PYTHONPATH、MPI root 运行权限）
- 构建 C++ CUDA 后端：`bash ./build.sh`（→ `libqcu.so`，cd cpp/cuda/qcu && make.sh）
- 构建独立 KUPL CPU 后端：`cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/kupl/qcu/build -j`
- 构建 Cython 扩展：`bash ./install.sh`（setup.py build_ext --inplace）
- 生产求解：`python pyqcu/testing/simple/solve_case.py --backend cuda|quda|kupl`（原子写解，
  失败不发布）；性能测量用 `compare_three_way.py`，说明见 `pyqcu/testing/simple/README.md`
- 测试：`cd pyqcu/testing && pytest .`；MPI 单文件：`mpirun -np 4 python pyqcu/testing/python/conftest.py`
- Git 标签：`stab<N>`/`dev<N>`/`bug<N>` 独立编号 + 子版本（如 `stab15_1`），见 tag 技能
- Python ≥ 3.10，依赖 PyTorch、Cython、mpi4py、h5py、numpy、CUDA toolkit；TileLang 可选

## 架构：三个执行层级

1. **Torch 后端**（`pyqcu/torch/dslash/`、`pyqcu/torch/solver/`、`pyqcu/torch/smear/`、`pyqcu/torch/tools/`）— PyTorch/CANN 实现，跑 CPU/CUDA/昇腾 NPU。公共物理/布局定义保留在 `pyqcu/lattice/`，公共 MPI/HDF5/env/budget 工具保留在 `pyqcu/tools/`，三后端 `Multigrid` 选择器保留在 `pyqcu/solver/multigrid/`。顶层 `pyqcu/dslash|solver|smear|tools` 只作兼容门面。所有 Python 代码 `import pyqcu.cann as _torch`，不得直接 `import torch`（NPU 不支持复数张量，cann 层自动分解实虚部）。
2. **C++ CUDA 后端**（`cpp/cuda/qcu/`）— 手写 CUDA 内核 + MPI halo 交换，经 Cython 桥 `pyqcu/cuda/qcu/qcu.pyx` 暴露，生产路径。
3. **C++ KUPL CPU 后端**（`cpp/kupl/qcu/`）— 独立 C++17 MPI+OpenMP 实现，
   不引用 CUDA 源码；经 `pyqcu/kupl/qcu.py` 的 `ctypes` 绑定使用相同
   `params/argv/set_ptrs` 与控制生命周期。

## 关键惯例

- **参数协议**：`params`（int32[58]，dev84 起 `_MG_USE_DEFLATE_=55`/`_MG_MU_PRE_=56`，dev87 起 `_MG_USE_INIT_GUESS_=57`=x0 热启动开关，warm 时以 fermion_out 预填解为初值）、`argv`（float[8]，槽 7 为 `_CSW_`）、`set_ptrs`（int64[100]）三个扁平张量桥接 Python↔C++；`pyqcu/cuda/define.py` 与 `cpp/cuda/qcu/include/define.h` 必须同步。
- **`_SET_PLAN_` 计划选择**：-2 Laplacian、-1 Gauss gauge、0 Wilson dslash、1 BiStabCG/CG、2 Clover dslash。
- **调用生命周期**：普通单操作事务使用 `applyInitQcu` → 操作 → `applyEndQcu` →
  `params[define._SET_INDEX_] += 1`。`applyEndQcu` 按**当前**索引释放对象，
  不能先递增再 end。普通 scratch 只使用 `set_ptrs[0:30]`；
  `_SET_PTRS_COARSE_BASE_=30` 之后保留给 coarse/strict 指针。新代码优先使用
  `QcuRuntime.call/operation()`；Strict 多调用生命周期在同一私有槽位内保持索引不变。
- **张量布局**：规范场 `[3,3,4,Lx,Ly,Lz,Lt]`、费米子场 `[4,3,Lx,Ly,Lz,Lt]`、Clover 项 `[4,3,4,3,Lx,Ly,Lz,Lt]`；时空维永远是最后 4 轴（`...xyzt`），ward 索引用负整数（`wards['x']=-4`）。HDF5 内部用 `zyxt` 序，经 `ccdxyzt2ccdptzyx`/`scxyzt2psctzyx` 转换。
- **日志约定**：`PYQCU::MODULE::SUBMODULE:\n message`，由 verbose 标志控制。
- **测试**：测试函数在 `pyqcu/testing/__init__.py`，`pyqcu/testing/*/conftest.py` 手动取消注释要运行的测试。
- **CUDA 生产执行拓扑**：一个 Python 进程对应一个 MPI rank、对应一张 GPU/DCU/GPGPU。跨设备并行必须走 MPI 多进程格点分域；生产驱动不得在一个 Python 进程内创建多 GPU worker，也不得让 rank 数超过设备数后把共享设备当作加速路径。DCU/GPGPU 程序遵循同一进程-rank-设备映射。
- **多 GPU 测试专用例外**：`pyqcu/cuda/_multi_gpu.py`（`MultiGpuMultigrid`）是显式 opt-in 的测试/基准驱动，不会在导入 `pyqcu` 时自动构造或启动。它在一个单 MPI rank 进程内用 N 线程分别绑定 N 张设备，每线程执行完整 Legacy MG，不是一次 solve 的跨卡分域；除非任务明确要求多卡一致性或吞吐测试，新 CUDA 程序不得使用。该测试驱动仍须保证每线程独立 `params/argv/set_ptrs`（`_SET_INDEX_` 各自从 0 计数）；Cython 桥的 `nogil` 调用与 pxd 声明规则保持不变。
- **求解器停机语义（dev87 起）**：`applyCloverMultigridQcu` 主循环停机为相对判据 rn²<atol²·‖b__o‖²，且单 rank 每 50 迭代做周期真残差刷新（reliable-update，防 fp32 递推漂移）；多 rank 刷新未启用。
- **Strict Hopper overlap（2026-09-27）**：分布式 complex64 Strict-MG 默认用非阻塞 vector halo 与 local/remote 分区计算；`PYQCU_MPI_OVERLAP=0` 回退原路径。该路径改变 remote-forward 浮点累加顺序，complex128 默认关闭；仅显式 `PYQCU_STRICT_OVERLAP_C128=1` 才启用，且必须以大格真残差/迭代回归为门槛。
- **HDF5 持久化（h5py）**：所有保存/读取走 h5py；`pyqcu/tools/_io.py` 的 `save_tensor_h5`/`load_tensor_h5`（每调用独立 File 句柄，多线程安全）+ MPI mpio 路径（`gridoooxyzt2hdf5oooxyzt`）。null-vector/粗网格算子缓存 `.h5`（单句柄一次写全部 dataset，勿逐 dataset 覆盖重建）。
- **分布式 benchmark 输入**：`pyqcu/testing/qcu/strict/quda_comparison/bench_strict_vs_quda.py::_load_h5_local_array` 在多 rank 下用 h5py mpio 读取 rank slab（gauge/source 为 checkerboard-compressed 尾部轴，null 为 full \(x,y,z,t\)）；不要退回每 rank 全量读取。trace-on 与 trace-off 的 Strict runtime cache identity 相同，formal 编排优先让 trace-on 复用 trace-off cache；显式 `--cache-expect hit` 时 cold 与 warm/steady phase 也应允许同一 cache。

## 目录结构

| 路径 | 内容 |
|---|---|
| `pyqcu/` | 公共门面与后端：`api/`（后端无关的配置、结果、能力、资产清单与 Protocol）、`torch/`（Torch/CANN 的 dslash、solver、smear、张量专用 tools）、`lattice/`（公共 gamma/Gell-Mann、SU(3)、源场定义）、`tools/`（公共 MPI/HDF5/env/budget/绘图）、`solver/multigrid/`（三后端统一 MG 选择器）、`solver/krylov/`（Torch 迭代与 CUDA/KUPL 原生整次 solve 的统一 Krylov 选择器）、`cuda/`（Cython CUDA 后端与多卡测试驱动）、`kupl/`（独立 CPU 后端）、`testing/`（集成测试）、`cann/dtk/maca`（Tensor 兼容层与占位）。顶层 `dslash/`、`smear/`、`solver/`、`tools/` 保留旧导入兼容。 |
| `cpp/cuda/qcu/` | C++ CUDA 后端：`src/`（32 个 .cu 内核）、`include/`（24 个模板头）、`python/pyqcu.h`（C API 37 个入口，须与 `qcu_api.pxd` 同步） |
| `cpp/kupl/qcu/` | 独立 C++17 MPI+OpenMP CPU 后端：4D rank halo、NUMA/OpenMP 绑定、canonical c64/c128 Wilson/Clover dslash、Wilson/Clover 求解器、3D Laplacian、正确性与吞吐测试 |
| `cpp/{cann,dtk,maca}/qcu/` | 占位 PASS，无实现 |
| `pyqcu/testing/` | 统一测试入口：`pyqcu/`（纯 Python 主套件）、`qcu/`（C++ CUDA 后端）、`kupl/`（CPU 后端 Python ABI/参考差分）、`quda/`、`pyquda/`、`cpu/npu/dcu/gpu/tilelang/profiler/benchmark/`、`regression/session-2026-08-24/`、`data/`（参考 HDF5） |
| `skills/` | 项目技能库（35 个技能目录：SKILL.md + 简短 AGENTS.md，目录级领域知识文档；目录名一律 `pyqcu-` 前缀；索引与技能表见 `skills/AGENTS.md`） |
| `docs/` | 正式独立文档与指南（`.pdf`/`.tex`/`.md`）：`dims.md`、`env.md`、`install.md`、`examples.md`、`profiler.md`、`ORGANIZATION.md`；从 `data/**` 提升的文档位于 `docs/data/` |
| `refer/` | 参考实现脚本（`huzc/`）、开发历史报告（`multigrid_analysis_20260717.*`）与 `git-rep/` 参考 Git 子库 |
| `logs/` | 当前任务尚未归档的运行记录（JSON/LOG/TSV/TXT/STDOUT 和阶段报告/图表）；`logs/data/` 汇总从 `data/**` 提升出的运行摘要。阶段报告先进入待 tag 提交，打 tag 后在后续清理提交中删除；张量/cache 实体在 `data/logs/`。 |
| `data/` | 数据、安装与构建产物（`.h5`、`.so`、`.dat`、`.csv`、`.svg` 等）；`data/docs/` 保存非文档 Office 文件，`data/logs/` 保存从 `logs/**` 提升的张量/cache；`quda-*`、`venv-*`、`p100-*` 作为自包含构建/安装树整体保留 |

## 已知反模式（勿重复）

- `torch.Tensor(<标量>)` 已被新版 PyTorch 移除（TypeError: data must be a sequence）——一律用列表包装 `torch.Tensor([x])`（bug31）
- `tools.norm()` 返回 **float**（内部已 .item()）——二次 `.item()` 会 AttributeError（本会话三踩）
- 复数 `operator*=` 覆盖 `_data.x` 前先使用（lattice_complex.h）
- `cudaMallocAsync` 缓冲大小与内核写大小不匹配
- 裸 `except:` 吞掉 KeyboardInterrupt
- `torch.linalg.inv` 逐点 for 循环（须批量）
- 用对象（如 `self.sitting`）做 truthy 判断
- `nstep>1` 循环内不更新 U（stout）
- `python_requires` 低于 3.8
- 未经 large-lattice（尤其双 P100 c64 \(16\times32\times32\times48\)）收敛回归就重排 coarsest BiCGStab 的全局归约顺序；triple-dot/pipelined 实验曾导致外层 FGMRES 停滞，已回退。
- 在多个长 CUDA 任务并发时忽略 QUDA `cudaHostRegister` mapped ghost-buffer 的瞬时失败；该失败应以同配置有界重试处理，不得修改求解协议或伪造成功。

<!-- form-skill:start -->
## form 格式约定

- 库类型：复杂库。主导语言：C++/CUDA（`cpp/**`）、Python/Cython（`pyqcu/**`、`setup.py`）、Bash（根构建入口和 `skills/**/*.sh`）。
- 文件命名：C++ 头文件、`.cu` 源文件、Python 模块和 Bash 脚本使用小写下划线或既有点划分；语言生态固定名称和上游 vendored 文件保留原名。
- **Agent 规则文件名称（硬性）**：所有目录的 agent 指令文件统一使用 `AGENTS.md`；`AGENTS.md` 是唯一名称，`AGENT.md` 是历史输入错误，不是兼容别名。只存在单数文件时直接重命名为 `AGENTS.md`；同目录双文件并存时，以 `AGENTS.md` 为主文件，完整合并并去重 `AGENT.md` 的有效内容后删除 `AGENT.md`，并同步现行脚本、配置、文档和本地规则中的全部引用。历史日志正文可保留旧路径作为既往证据。
- 函数命名：C++ 对外 C API 使用小驼峰（如 `applyInitQcu`），内部 kernel/helper 使用小写下划线；Python 公共函数和私有函数分别使用 `snake_case`、`_snake_case`。
- 变量命名：宏和编译期常量使用全大写下划线，C++ 宏保留 `_NAME_` 锚点；普通变量使用小写下划线，Python 模块级常量使用全大写下划线。
- 对象命名：C++/Python 类型使用大驼峰，领域缩写作为完整词保留；绑定的 C API 名称必须与 `qcu.pyx`、`qcu.pxd` 和 `pyqcu.h` 同步。
- 顶层目录白名单：`cpp/`、`data/`、`docs/`、`logs/`、`pyqcu/`、`refer/`、`skills/`。根入口文件仅允许 `AGENTS.md`、`.gitignore`、`LICENSE`、`README.md`、`build.sh`、`env.sh`、`install.sh`、`setup.py`。
- 测试入口：主套件为 `cd pyqcu/testing && pytest .`；MPI 入口为 `mpirun -np 4 python pyqcu/testing/python/conftest.py`；技能自带 shell 回归保留在对应技能目录，如 `bash skills/pyqcu-tag/tag-chain.test.sh`。
- 功能复现：构建使用 `bash ./build.sh`，Cython 扩展使用 `bash ./install.sh`；纯格式/文档改动至少执行 `bash -n`、相关回归、`git diff --check` 和 `bash skills/pyqcu-form/form-audit.sh`。
- 文档/日志/数据：正式文档在 `docs/**`；当前任务未归档的运行摘要及 `log/json/tsv/csv/txt/stdout/jsonl` 在 `logs/**`。阶段结果先提交并打 tag，随后删除工作区报告副本；历史读取使用 `git show <tag>:<path>`。数据、SVG、构建/安装树和缓存实体在 `data/**`。
- 日志文件名：一律使用紧凑时间戳 `vYYYYMMDD[HH[MM[SS]]].txt`，同名冲突追加 `-2`、`-3`；不得使用 `vYYYY-MM-DD-HH-MM.txt` 一类连字符时间戳格式。
- Git 交付：form 批次按逻辑分组提交，只暂存本任务文件，禁止 `git add -A`/`git add .` 和 force push；推送与开发标签按 tag 技能的 follow 链执行。
- 大文件闸门：任何提交前运行 `${CONFIGURE_ROOT:-$HOME/configure}/skills/form/scripts/form-large-files-check.sh --root "$(git rev-parse --show-toplevel)"`；默认阻止单个新增、修改、复制或重命名文件超过 `10,000,000` 字节，命中时不得提交或推送。
- 本地例外：`data/**` 可跟踪正式证据、构建树和自包含安装树；`logs/**` 可在打 tag 前暂存第一方报告 `.md/.tex/.pdf` 与图表，打 tag 后不得保留 `logs/<tag>/**` 报告副本；`docs/张鑫 508应用测试报告-PyQCU/` 是文档资产保留的解包 Office 资源；`skills/pyqcu-tag/tag-chain.test.sh` 与该技能实现同目录；`refer/git-rep/**`、`refer/huzc/**` 与已存在的上游 vendored/generated 树保留来源名称。
- 历史例外边界：当前本地包装器登记并豁免 `1069` 项既有资产/合法布局 finding（以 `bash skills/pyqcu-form/form-audit.sh` 输出为准）；不得删除、迁移、整体禁用检查或改写排除项来绕过这些历史资产。新增例外必须精确匹配、保留证据并执行审计回归。
- 规则维护：本节是根规则文件中唯一的 `form-skill` 起止标记托管块，由 `/root/configure/skills/form/scripts/form-agents-sync.sh` 更新并执行 `--check`；不得手工制造重复、未闭合标记或绕过同步检查。目标库原有明确约定优先于 form 通用规则。
<!-- form-skill:end -->
