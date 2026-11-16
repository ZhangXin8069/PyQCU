# PyQCU 正式快速测试端口（`pyqcu.test`）

`pyqcu.test` 是**随发行包发布的正式快速测试端口**：`pip install pyqcu` 之后仍然存在，
用一条命令在秒级到分钟级验证 Torch / KUPL / CUDA 三条后端路径的核心正确性，并支持
指定格点、MPI 网格、规范场、源、质量、clover 系数、求解器与设备。它只回答
“装好的库算得对不对”，不做性能测量，也不替代源码 checkout 里的集成测试。

| 端口 | 路径 | 是否随包发行 | 定位 |
|---|---|---|---|
| 正式 | `pyqcu/test/**` | 是（sdist/wheel） | 跨后端正确性门，安装态可运行 |
| 非正式 | `pyqcu/testing/**` | 否（`MANIFEST.in` prune、`setup.py` 排除） | 集成/回归/性能/QUDA/HPCKit 矩阵，仅源码 checkout |

`pip install pyqcu` 的用户**没有** `pyqcu.testing`：不要在新的用户文档、发布说明或
验证脚本里把 `pyqcu.testing` 写成安装态入口。

## 最短上手

```bash
python -m pyqcu.test --list          # 只列用例矩阵，不 import torch/native 后端
python -m pyqcu.test --check         # 导入/打包/策略自检（不跑数值）
python -m pyqcu.test --quick         # tier 0 + 格点 8 8 8 16 + c64 + seed 42
python -m pyqcu.test --quick --json  # 机器可读报告（默认写 stdout）
```

源码 checkout 中先 `source ./env.sh`；安装态直接从任意目录执行 `python -m pyqcu.test`。
`--list` / `--list --json` 是当前版本的权威用例矩阵与生效选项记录：任何 flag 的最终
语义以 `python -m pyqcu.test --help`、`--list --json` 的实际字段和 `pyqcu/test/policy.py`
为准，本文档描述契约，不重复实现细节。

## CLI

```text
python -m pyqcu.test [--backend LIST] [--lattice-size N N N N | --lattice N N N N]
  [--lattice-grid N N N N] [--mg-grid N N N N]
  [--quick] [--tier 0|1|all] [--only TAG ...] [--solver CHOICE] [--dslash CHOICE]
  [--gauge random|unit|ones] [--gauge-sigma F | --guage-sigma F]
  [--source point|ones|random] [--mass F | --kappa F] [--csw F]
  [--dtype c64|c128|complex64|complex128|float32|float64]
  [--device auto|cpu|gpu|dcu|npu] [--device-id N] [--threads N]
  [--max-iter N] [--restart N] [--mg-levels N] [--mg-nu-pre N] [--mg-nu-post N]
  [--atol F] [--list] [--json [PATH|-]] [--repeat N] [--warmup N] [--seed N]
  [--timeout S] [--fail-fast] [--allow-all-skipped] [--outdir DIR] [--verbose]
  [--check] [--version]
```

四个整数的参数（格点、进程网格、MG 块网格）接受三种写法：`8 8 8 16`、`8,8,8,16`、
`[8,8,8,16]`；`--backend`/`--only` 接受逗号分隔或空格分隔的多个值。

### 用例选择

| 选项 | 含义 |
|---|---|
| `--backend` | 逗号或空格分隔：`torch,cuda,kupl` 加别名 `cpu`（=`torch,kupl`）、`dcu`（=`cuda`）、`npu`（=`torch`），或 `all`（默认 = 全部规范后端）。`all` 不要求三条后端都可用，不可用的按 SKIP 处理；报告里只出现规范名，不出现别名 |
| `--tier` | `0`/`1`/`all`：跑 tier ≤ N 的用例（`--quick` 强制 `0`） |
| `--only TAG ...` | 可重复；接受完整 tag（`torch/dslash/wilson`）、backend、kind 段（`dslash`）或 tag 前缀（`torch/dslash`） |
| `--solver` | `all`（默认，不筛）/`bicgstab`/`cg`/`multigrid`/`mg`。只筛 solve/mg 用例：`bicgstab` 保留 `*/solve/bistabcg`，`cg` 保留 `*/solve/cg`，`multigrid`/`mg` 保留 `*/mg/*`；dslash、smear、gauge、lattice、ABI 用例不受影响 |
| `--dslash` | `all`/`both`（默认，不筛）/`wilson`/`clover`。只筛 `kind == dslash` 的用例；其余用例不受影响 |
| `--quick` | 只锁 **tier 0、格点 `8 8 8 16`、`c64`、`seed 42`、`repeat 1`**（`warmup` 仍为 1）。设备、求解器、算子、质量、clover、源、规范场、网格、线程等选项继续生效，可与 `--quick` 组合；只有与 `--lattice-size`/`--dtype`/`--seed`/`--tier` 冲突才按用法错误退出 `2` |
| `--list` | 打印选中用例矩阵后退出；重型 import 之前短路，不 import torch / native 桥。`--list --json` 给出 `cases`/`count` 与本次生效配置；`--list --verbose` 把完整配置写到 stderr |
| `--check` | 跑 `selfcheck`（导入隔离、打包包含性、策略常量）后退出，不跑数值 |
| `--version` | 打印端口版本后退出 |

### 几何、MPI 与多重网格网格

| 选项 | 含义 |
|---|---|
| `--lattice-size` / `--lattice` | 同一选项的两个拼写，取 `8 8 8 16`、`8,8,8,16` 或 `[8,8,8,16]`。四个正整数且**偶数、≥ 4**（奇偶分解与 clover 需要偶数体积），默认 `8 8 8 16`；非法值按用法错误退出 `2` |
| `--lattice-grid` | MPI 4D 进程网格（`x y z t`），默认 `1 1 1 1`。**乘积与 `mpirun -np` 的 rank 数不一致不是用法错误**：原生后端（KUPL/CUDA）报 `SKIP(mpi_grid)`，torch 用例不看该网格。真正执行且选中原生用例时，不一致会在 stderr 打一行提示 |
| `--mg-grid` | 多重网格粗化块网格（每维因子），默认 `2 2 2 2`；只被 `*/mg/*` 用例使用 |

### 规范场、源与物理参数

| 选项 | 含义 |
|---|---|
| `--gauge` | 背景规范场来源：`random`（默认，按 `--seed` 生成 Gaussian 场并投影到 SU(3)）、`unit`（单位链接）、`ones`（全 1 链接）。**不是文件路径**；读 HDF5 组态属于非正式套件 |
| `--gauge-sigma` / `--guage-sigma` | `--gauge random` 的 Gaussian 宽度，默认 `0.1`。`--guage-sigma` 是历史拼写，与正确拼写指向同一个选项（argparse 别名），效果完全等价；报告只记录 `gauge_sigma` |
| `--source` | 费米子源：`point`（默认，原点单点源，`\|\|b\|\| = 1`）、`ones`（常数源）、`random`（按 `--seed` 的高斯源） |
| `--mass` / `--kappa` | 同一物理量的两种写法：`kappa = 1/(2*mass + 8)`，反之 `mass = 1/(2*kappa) - 4`。两者都**没有** CLI 默认值，未给出时不向前转发、由各 runner 的默认决定（tier 0 的 torch 路径默认 `kappa = 0.125`，即 `mass = 0`）；`--kappa` 必须 `> 0`。只给一个；两个都给时都会记入报告 `options`，换算责任在 runner，跨后端对齐时优先统一用 `--kappa` |
| `--csw` | clover 系数，默认 `1.0`；`0` 即纯 Wilson 极限（clover 用例在 `csw = 0` 时退化为 Wilson 一致性检查） |
| `--dtype` | `c64`/`c128`，以及 `complex64`/`complex128`、`float32`/`float64` 四个别名（分别归一化到 `c64`/`c128`）；报告与 `--list` 只出现规范名 |

### 求解器与多重网格

| 选项 | 含义 |
|---|---|
| `--max-iter` | 求解迭代上限；未给出时用 runner 默认（KUPL 1000、CUDA 200，torch 见 runner） |
| `--restart` | Krylov 重启长度；未给出时用 runner 默认 |
| `--mg-levels` | 多重网格层数；未给出时用 runner 默认（tier 0 为 2） |
| `--mg-nu-pre` / `--mg-nu-post` | 前/后平滑步数；未给出时用 runner 默认 |
| `--atol` | 只覆盖 `compare`（算子/涂抹一致性）相对容差；`solve`/`native_solve`/`gamma` 仍按 dtype 固定，用户参数不能放宽“真的收敛了” |

### 设备与线程

| 选项 | 含义 |
|---|---|
| `--device` | `auto`（默认，不向前转发，由后端自选；torch 路径即 CPU）、`cpu`、`gpu`（→ `cuda:0`）、`dcu`（→ `dcu:0`）、`npu`（→ `npu:0`）。也可直接给具体设备串（如 `cuda:3`）。请求的加速器族在本机不可用时报 `SKIP(device_unavailable)`，不是 FAIL |
| `--device-id` | 加速器序号（`>= 0`），转发给原生 runner |
| `--threads` | 原生 CPU 线程数（KUPL/OpenMP 路径），转发给 runner；未给出时用环境默认 |

### 执行与输出

| 选项 | 含义 |
|---|---|
| `--json [PATH\|-]` | 省略值或 `-` 写 stdout；给路径时原子写（临时名 + rename），失败不留下半份文件 |
| `--repeat` / `--warmup` | 默认 `5` / `1`；`--quick` 固定 `1` / `1`。warmup 传 `0` 会被拒绝 |
| `--seed` | 规范场/源的确定性 seed，默认 `42` |
| `--timeout` | 每个用例的超时秒数（默认按用例给出）；超时记 `TIMEOUT`，不是 `FAIL` |
| `--fail-fast` | 首个 `FAIL`/`ERROR`/`TIMEOUT` 立即停止（`SKIP` 不触发停止） |
| `--allow-all-skipped` | 全部 SKIP 时仍返回 0（默认返回 1） |
| `--outdir DIR` | 额外把 JSON 报告原子写到 `DIR/report.json` |
| `--verbose` | 每个用例的 metrics 与失败 gate 写 stderr；stdout 保持每用例一行、稳定可解析 |

`--json` 报告是 `TestReport.as_dict()`：顶层 `package`、`version`、`status`、
`exit_code`、`lattice`、`dtype`、`seed`、`mpi_size`、`mpi_rank`、`selected`、`counts`、
`all_skipped`、`duration_seconds`、`options`、`env`、`cases`；每个 case 带 `tag`、
`backend`、`kind`、`status`、`seconds`、`metrics`、`failed`、`reason`、`lattice`、
`dtype`、`seed`、`mpi_size`、`tier`、`runner`、`skip_token`。`options` 记录本次生效的
全部配置（`lattice`、`lattice_grid`、`mg_grid`、`dtype`、`seed`、`source`、`gauge`、
`gauge_sigma`、`mass`、`kappa`、`csw`、`solver`、`dslash`、`device`、`runner_device`、
`max_iter`、`restart`、`mg_levels`、`mg_nu_pre`、`mg_nu_post`、`threads`、`device_id`、
`atol`、`repeat`、`warmup`、`quick`、`backend_filter`、`tier`、`argv`，以及
`fail_fast`/`allow_all_skipped`/`mpi_grid_matches`/`backends`/`tags`），因此一次运行可由
它自己的输出复现。观测值在 `metrics` 里（`iterations`、`max_iter`、
`relative_residual` 及别名 `relative_error`/`true_residual`）。
状态词汇与 `pyqcu --selftest --json` 一致：`PASS`/`SKIP`/`FAIL`/`ERROR`/`TIMEOUT`。

## 参数映射（先读这一节）

**`--mass` vs `--kappa`**：同一个物理量。`kappa = 1/(2*mass + 8)`，默认 tier 0 值是
`kappa = 0.125`（`mass = 0`）。两者都未给出时 runner 的默认生效；给出时只给一个，
`--mass -3.4` 这类接近退化区的值合法（`kappa ≈ 0.8333`），但会显著改变算子条件数；
残差门不随质量放宽，接近退化区的质量只会更容易 FAIL。

**规范场 sigma**：`--gauge-sigma` 只在 `--gauge random` 时有物理意义；`--gauge unit` /
`--gauge ones` 是确定性组态，sigma 仍会记录进报告但不参与构造。

**MPI 格点网格 vs torch**：`--lattice-grid` 是**原生后端**的 MPI 分域网格
（`prod(grid) == MPI size` 时正常执行）；乘积不匹配时原生用例报 `SKIP(mpi_grid)` 并带
reason，**不是**用法错误，也不会被静默改成单 rank 结果。torch tier-0 用例是单进程计算，
不读该网格；因此 `--backend torch --lattice-grid [1,2,2,1]` 只影响解析与报告，不会让
torch 变成分布式运行。分域 MPI 覆盖属于非正式套件（`pyqcu/testing/**`）。

**`--backend` 别名展开**：`cpu` → `torch,kupl`；`dcu` → `cuda`（DCU/ROCm 复用同一 C++
桥与 `cuda/*` 用例，设备族由 `--device dcu` 指定）；`npu` → `torch`（经 `pyqcu.cann` 把
复数运算分解为实/虚部）；`all` → 三个规范后端。展开发生在参数解析阶段，报告与
`--list --json` 只出现规范名。

**`--device` 映射**：`auto` 不转发任何设备串（后端保留自己的默认，torch 即 CPU）；
`cpu`→`cpu`、`gpu`→`cuda:0`、`dcu`→`dcu:0`、`npu`→`npu:0`；具体设备串（`cuda:3`）原样
透传。设备族不可用时相关用例报 `SKIP(device_unavailable)`（带 reason），不是 FAIL。

**CUDA 桥缺失**：`pyqcu.cuda.qcu` 没有构建出的扩展时，`cuda/*` 用例报
`SKIP(cuda_bridge)`；这是本 checkout 的预期结果，不得为变绿去构建桥或伪造 PASS。

**`--quick` 只锁五件事**：tier、格点、dtype、seed、repeat。它**不**锁定设备、求解器、
算子、质量、clover、源、规范场、网格、线程和 `--atol`，这些参数在 `--quick` 下照常生效
（因此 `--quick --device gpu --solver cg` 是合法组合，`--quick --dtype complex128` 才是
冲突用法错误）。

**`--atol` 的边界**：只覆盖 `compare` 门（算子/涂抹的独立路径一致性），不覆盖
`solve`/`native_solve`（真残差）与 `gamma`（Dirac 代数恒等式）。它也可能作为求解器请求
阈值转给 runner，但**判定门不变**：把 `--atol` 放宽只会让 solve 用例因为真残差达不到
dtype 门而 FAIL，不会被“放宽成通过”。

## 退出码

| 码 | 含义 |
|---|---|
| `0` | 无 `FAIL`/`ERROR`/`TIMEOUT`（允许 `SKIP`） |
| `1` | 任一 `FAIL`/`ERROR`/`TIMEOUT`，或所选用例**全部 SKIP**（除非显式 `--allow-all-skipped`） |
| `2` | 用法/环境错误：未知 backend/solver/dslash/device、非法格点或网格、非法 dtype、无法写 `--json` 路径等 |

与安装自检的差异必须记住：`python -m pyqcu --selftest` 把数值 `FAIL` 记为 **2**；
本端口把数值失败记为 **1**、用法错误记为 **2**。批量脚本不得把两者当同一约定。

## Tier 0 用例（默认门禁）

默认单 rank、`c64`、不读数据文件；`[8,8,8,16]` 是项目正确性门格点
（`pyqcu/testing/_config.py`、根 `AGENTS.md`），**不是**发布验收用的 `8x8x16x16`；
不要互换。共 **21** 个用例：前 14 个是原始矩阵，后 7 个是 gauge/Wuppertal/CG/clover
补充。

| # | tag | 判据 | 可能 SKIP |
|---|---|---|---|
| 1 | `torch/lattice` | `check_su3` 为真；`gamma_mu^2 = I` 偏差 ≤ `gamma` | `device_unavailable` |
| 2 | `torch/dslash/wilson` | 全算子 vs 奇偶分解相对差 ≤ `compare` | `device_unavailable` |
| 3 | `torch/dslash/parity` | `matvec_all` vs `matvec_eeo`/`matvec_oeo` 相对差 ≤ `compare` | `device_unavailable` |
| 4 | `torch/dslash/clover` | clover 项 + 逆往返相对差 ≤ `compare`（`csw=0` 退化为 Wilson） | `device_unavailable` |
| 5 | `torch/smear/stout` | smearing 前后 SU(3) + 局部/全局一致相对差 ≤ `compare` | `device_unavailable` |
| 6 | `torch/solve/bistabcg` | 独立真残差 ≤ `solve` 且 `iterations > 0` | `device_unavailable` |
| 7 | `torch/mg/legacy` | 显式 legacy 变体收敛，相对残差 ≤ `solve` | `device_unavailable` |
| 8 | `torch/mg/strict` | 显式 `strict` 变体收敛，相对残差 ≤ `solve` | `device_unavailable` |
| 9 | `kupl/abi/symbols` | C API 40 个符号，名称集合完全一致 | `kupl_library` |
| 10 | `kupl/dslash/wilson` | vs torch 参考相对差 ≤ `compare` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 11 | `kupl/solve/bistabcg` | 全解 + 独立残差 ≤ `native_solve`，`0 < iterations < max_iter` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 12 | `cuda/smoke/init_end` | 桥 `applyInitQcu`/`applyEndQcu` 生命周期成功 | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 13 | `cuda/dslash/wilson` | vs torch 参考相对差 ≤ `compare` | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 14 | `cuda/solve/bistabcg` | 全解 + 独立残差 ≤ `native_solve` | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 15 | `torch/gauge/random` | 生成并投影后的规范场 `check_su3` 为真（σ/seed 只作为观测记录在 `metrics`，不是额外门） | `device_unavailable` |
| 16 | `torch/smear/wuppertal` | smearing 前后 SU(3) + 局部/全局一致相对差 ≤ `compare` | `device_unavailable` |
| 17 | `torch/solve/cg` | CG（γ5-hermitian 算子）独立真残差 ≤ `solve` 且 `iterations > 0` | `device_unavailable` |
| 18 | `kupl/dslash/clover` | clover dslash vs torch 参考相对差 ≤ `compare` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 19 | `kupl/solve/cg` | 全解 + 独立残差 ≤ `native_solve`，`0 < iterations < max_iter` | `kupl_library`、`mpi_single_rank`、`mpi_grid` |
| 20 | `cuda/dslash/clover` | clover dslash vs torch 参考相对差 ≤ `compare` | `cuda_bridge`、`cuda_device`、`device_unavailable` |
| 21 | `cuda/solve/cg` | 全解 + 独立残差 ≤ `native_solve`，`0 < iterations < max_iter` | `cuda_bridge`、`cuda_device`、`device_unavailable` |

相对容差按 dtype 固定：`c64` 为 `compare` `1e-5`、`solve` `1e-3`、`native_solve`
`2e-4`、`gamma` `1e-6`；`c128` 为 `1e-10`/`1e-6`/`1e-8`/`1e-12`。`--atol` 只能覆盖
`compare`，不能放宽残差门。

`case` 的 `expect` 是唯一的 PASS/FAIL 判据；runner 只回观测值。所有 solve 用例的
真残差必须由 `||D x - b|| / ||b||` 独立重构，不得复用求解器内部递推；CG 用例额外要求
算子按 γ5-hermitian 约定求解，但残差仍在全 Wilson 算子上重构。torch 与 KUPL 数值路径
不依赖 GPU；CUDA 用例只有在桥与设备同时存在时才真正执行。

`--solver`/`--dslash` 只筛矩阵，不新增用例：新增算法覆盖必须新增 tag（并进
`pyqcu/test/units.py`），不得复用旧 tag 改语义。

## SKIP 策略

| reason | 触发条件 |
|---|---|
| `kupl_library` | `pyqcu.kupl` 的 ctypes 加载抛 `FileNotFoundError`/`OSError`，或全部候选库路径不存在 |
| `cuda_bridge` | `pyqcu.cuda.qcu` 缺少 `applyInitQcu`，或 Cython 扩展未构建 |
| `cuda_device` | `torch.cuda.is_available()` 为 False |
| `mpi_single_rank` | MPI size > 1（用例要求单 rank） |
| `h5py_mpi` | `h5py.get_config().mpi` 为 False（需要 mpio 的 HDF5 读取） |
| `mpi_grid` | `prod(--lattice-grid)` 与 MPI size 不一致（原生 runner 报此 token） |
| `device_unavailable` | `--device` 请求的加速器族在本机不可用 |

- 每个 SKIP 必须带**非空** `reason`；空 reason 视为缺陷而不是“跳过了”。
- 本仓库 checkout 中 **CUDA SKIP 是预期结果**：`pyqcu/cuda/qcu/` 只有
  `.pyx/.pxd/.pyi`，没有构建出的 `qcu*.so`，用例报 `SKIP(cuda_bridge)`。它不是 FAIL，
  也不得为了变绿去构建桥、伪造 PASS 或跳过断言。
- `--lattice-grid` 与 MPI size 不一致时，torch 用例仍照常执行（它们不看网格），
  原生用例报 `SKIP(mpi_grid)`；一条都没验证成功时整体退出 `1`。
- 全部选中用例 SKIP 默认退出 `1`：一条都没验证不能算通过。

## 示例

```bash
# 1) 全量参数解析演示：--list 不 import 后端，也不跑数值（本次输出 7 行 torch 用例）
python -m pyqcu.test --lattice-size [8,8,8,16] --gauge-sigma 0.1 --mass -3.4 \
  --lattice-grid [1,2,2,1] --solver multigrid --backend torch --device cpu \
  --dslash clover --mg-grid [2,2,2,2] --atol 1e-6 --dtype complex128 --list
```

```bash
# 2) 单用例可复现跑（MG legacy，8x8x8x16、c64、seed 42；本机 PASS，约 47 s 含 setup）
python -m pyqcu.test --only torch/mg/legacy --repeat 1 --warmup 1
```

```bash
# 3) CPU-only 子集 + complex128 + 机器可读报告
python -m pyqcu.test --backend cpu --dtype complex128 --repeat 1 --json cpu.json

# 4) 只保留 CG solve 与 clover dslash（其他非 solve/mg、非 dslash 用例照常保留）
python -m pyqcu.test --solver cg --dslash clover --list

# 5) 指定规范场与源（不读文件：random/unit/ones 三选一）
python -m pyqcu.test --backend kupl --only kupl/solve/bistabcg \
  --gauge random --gauge-sigma 0.1 --source random --kappa 0.1199 \
  --csw 1.0 --atol 1e-6 --max-iter 1000 --repeat 1 --json - > report.json

# 6) NPU / DCU 设备族（不可用时该后端用例 SKIP，不伪造 PASS）
python -m pyqcu.test --backend npu --device npu --repeat 1
python -m pyqcu.test --backend dcu --device dcu --allow-all-skipped
```

`--lattice-size`/`--lattice-grid`/`--mg-grid` 的方括号写法在 bash 下可直接用；在
`zsh`/`csh` 或含特殊字符的 shell 里建议加引号（`--lattice-size '[8,8,8,16]'`）。

## 正式 vs 非正式（边界）

| 维度 | 正式 `pyqcu.test` | 非正式 `pyqcu.testing` |
|---|---|---|
| 发行 | 随 sdist/wheel（`recursive-include pyqcu *.py` 自动带上） | 不进 sdist（`prune pyqcu/testing` + `find_packages(... exclude=["pyqcu.testing", ...])`） |
| 安装用户 | 有 | 没有 |
| 导入成本 | stdlib-only：不 import `torch`/`numpy`/`mpi4py`/`h5py`/`pytest` | 模块级 import `torch`、`mpi4py`、`pytest` |
| 运行目标 | 秒级到分钟级跨后端正确性门 | 集成、回归、性能、QUDA/HPCKit 矩阵、历史资产 |
| 新增用例落点 | `pyqcu/test/units.py` | `pyqcu/testing/**` |

- 两个方向都不许互相 import：`pyqcu.test` 不 import `pyqcu.testing`；生产代码
  （`pyqcu.torch|cuda|kupl|lattice|tools`）不 import `pyqcu.test`。
- **不要在 `pyqcu.testing` 新增“正式”用例**：发布门要跑的用例属于 `pyqcu/test/units.py`，
  `pyqcu/testing` 只留探索性、研究性、性能性和历史回归资产。
- 可选的 pytest 适配器只放在非正式树（`pyqcu/testing/python/test_pyqcu_test_port.py`），
  SKIP 时 `pytest.skip(reason)`；它不属于发行内容。
- 本目录的 `README.md` 与 `AGENTS.md` 不进 sdist：`MANIFEST.in` 只收 `pyqcu` 下的
  `*.py/*.pyi/*.pyx/*.pxd`，且 `global-exclude AGENTS.md`。

## 目录约定

- 本目录**不得**出现 `test_*.py`：form 审计对 `<repo>/testing/` 之外的 `test_*.py`
  报 `WARNING TEST_LOCATION`，而 `pyqcu/test/` 不是 `testing/`。自检文件固定为
  `selfcheck.py`。
- 本包任何模块**不得 import pytest**：`pytest -q pyqcu/test/selfcheck.py` 只是在包外
  显式调用 `selfcheck.py` 中的 `test_*` 函数，pytest 不是运行期依赖。
- 不读 HDF5/仓库数据、不写仓库路径；`--json` 目标由调用者给，`--outdir` 缺省不落盘。
- `--quick` 只做正确性：**不得**用它报性能、加速比或回退；计时字段是诊断信息，
  中位数只在 `--repeat >= 5` 且 `warmup >= 1` 时才有一点意义。示例里的耗时数字是
  单机诊断值，不是性能结论。

## 与其他入口的关系

| 入口 | 定位 |
|---|---|
| `python -m pyqcu.test --quick` | 安装态可跑的正式正确性门（本目录） |
| `python -m pyqcu --selftest` | 安装自检：Torch/KUPL/CUDA 三层独立残差，`FAIL` 退出 2 |
| `cd pyqcu/testing && pytest .` | 非正式全树盘点，仅源码 checkout |
| `python -m pyqcu.testing.release_check --offline` | 发布门（仅 checkout；`sdist ships pyqcu.test` 正向核对本端口入包） |
