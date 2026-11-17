# L48 双节点操作参考（pyqcu.test / KUPL）

本文件是 [`../SKILL.md`](../SKILL.md) 的按需参考，只面向 **L48 + 两节点 + 真实
LIME/HDF5** 的操作者。这里只列已核实的事实和可安全执行的命令；远端结论按
2026-11-17 的实测基线记录（§2.2），**两节点与 L48 solve 仍未通过**，标 `TODO` 的
一律不得当作已完成。

设计背景、接口契约、完整证据与开放问题见
[`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md)（远端基线
§2.4、既有材料 §2.5）。本文件只保留操作者需要的最小集合，详细表格不在此复制。

## 1. 先记住四个结论

1. `pyqcu.test` 是随包发行的正式快速端口；当前 `--quick` 是 `8x8x8x16`、c64、
   单 rank 的正确性门，不是 L48 生产入口。
2. `pyqcu.test` full mode 的 `--gauge-file`/`--source-file` 原先是 HDF5 证据：
   `prepare` 校验、`prewarm` 读取并哈希，内置 tier-0 case 不会用文件替换
   `--gauge`/`--source` 生成场。直接把 `.lime` 传给 `--gauge-file` 不是已支持路径
   （并发实现把 HDF5 读取下沉到 native 输入身份，未审计，见 §2.2）。
3. 旧驱动 `pyqcu/testing/kupl/test_kunpeng_all.py` 只支持 `L24/L32`、只读 HDF5，
   没有 LIME；它的 `--mode validate` 是 `8x8x8x16` 小格冒烟，不是正式端口的台账
   校验。
4. 远端 `login02` 已可登录、L48 LIME 已在远端、`--check` 18/18（§2.2）。2026-11-17
   首批结果（§2.4）：单节点 **c64 BiCGStab 32 rank `1.6646 s`/11 it/`7.126e-5`**、
   **c64 Legacy-MG 16 rank wall `110.35 s`** 已跑通；**c128、两节点、CG 与 32 rank
   dslash/clover 仍无结果**——不得写成“L48 已跑完”或“两节点可跑”。

## 2. 已核实事实

### 2.1 本机（WSL2）

| 项目 | 事实 | 证据 |
|---|---|---|
| 本机 | WSL2；`~/data -> /mnt/f/Data` | `readlink -f ~/data` |
| L48 文件 | `/home/zhangxin/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime` | `stat` |
| 文件大小 | `6,115,435,952 B`（约 5.70 GiB） | `stat` |
| LIME 元数据 | `dims=48 48 48 96`、`precision=D`、`datacount=4`、`colors=3`、`clovCoeff=1.160920225783`、`Mass=-0.2770` | `_lime.py::lime_metadata` + record XML |
| LIME payload | `6,115,295,232 B` | `ildg-binary-data` record length |
| 未做（本机） | 未全量读取 LIME、未校验 checksum、未转 HDF5、未求解 | 本机未执行；远端进展见 §2.2 |
| 正式端口 tier 0 | `--list --json -` 实测 `19` 个 case，`[8,8,8,16]`、c64 | 本机命令 |
| 正式端口 preset | 设计时只有 `L24`/`L32`；并发实现在 14:0x 加入 `L48`（未提交），实测 `--mode list --lattice-preset L48` 被接受 | `python -m pyqcu.test --help` |
| 正式端口 L48 列表 | `--mode list --lattice-size 48 48 48 96` 与 `--lattice-preset L48` 都只列 19 个 tier 0 case，不是 L48 生产行 | 本机命令 |
| 旧驱动 L48 | 全文无 LIME；`LATTICES` 只有 `L24/L32` | `test_kunpeng_all.py:87` |
| 旧驱动 validate | 强制 `8 8 8 16`、2 rank、`shm=0` 的小格冒烟 | `test_kunpeng_all.py:1749` |
| 提交脚本 | `#DSUB -nn 1` 写死；`-N $RANKS -rpn $RANKS` 是单节点形态；`ranks*threads<=608` | `submit_kunpeng_all.sh:94,137` |
| SSH | 本机 `~/.ssh/config`、`/etc/hosts` 均无 `lingsheng`；`ssh lingsheng` 报 `Could not resolve hostname`（别名缺失，远端实际入口见 §2.2） | 本机命令 |
| 版本 | `pyproject.toml` 为 `0.1.17`；`git describe` = `test37-1-g1f65469e`（HEAD `1f65469e`）。本轮不改版本、不打 tag | `git describe` |
| 自检 | 远端 0.1.17 覆盖树 `python -m pyqcu.test --check` = **18/18**；本机当前工作树因并发未提交改动为 **20/20**（17:55；17:47 为 19/19，多 2 项检查，不能混读）；`--list` = **19** 个 tier 0 case（11 torch, 5 kupl, 3 cuda） | 本机命令 |

L48 c64 逻辑尺寸（仅算术，不是已写出的文件）：

```text
sites          = 48*48*48*96 = 10,616,832
gauge c64      = 3,057,647,616 B（约 2.85 GiB）  [3,3,4,48,48,48,96]
source c64     = 1,019,215,872 B（约 0.95 GiB）  [4,3,48,48,48,96]
MG cache 逻辑  = 18,728,091,648 B（约 17.44 GiB；`dof=12 12`、`mg-grid=2 2 2 2` 历史预算）
```

### 2.2 远端基线（2026-11-17 实测，`login02`）

只写操作者需要的事实；逐条证据与命令形状见
[`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md) §2.4。
私网地址、密钥与口令不入库。

| 项目 | 事实 |
|---|---|
| 登录 | `login02`，账号 `nsls_yangyb`，经用户提供的内部路由；VPN/路由未连时报 `No route to host`。`ssh lingsheng` 只是别名缺失，不代表不可达 |
| 仓库 | `/home/share/nsls_yangyb/PyQCU-main`，**非 git 的 0.1.15 快照**；0.1.17 Python 代码覆盖前备份为 `pyqcu.bak.20261117`（当前是 0.1.15 底座 + 0.1.17 Python 的混合树） |
| 自检 | `python -m pyqcu.test --check` = **18/18**；此前 Python 3.10 会在 `kupl_qcu.py`/`cuda_qcu.py` 的 `source[*tuple]` 索引上 `SyntaxError` |
| 运行时 | Python 3.10.9 aarch64（anaconda 2023.3）、numpy 1.23.5、h5py 3.7.0（`mpi=False`）、mpi4py 4.1.2、torch 1.12.1 |
| HPCKit | 26.1.RC1：`PATH` 加 `.../hmpi/bisheng/release/hmpi/bin`；`LD_LIBRARY_PATH` 需 `hmpi/lib` + `compiler/gcc/lib64`（`GLIBCXX_3.4.29`）+ `compiler/bisheng/lib`（`libomp`）+ `/work_ssd/software/HPCKit/alllib`；`OPAL_PREFIX` 指向 `.../hmpi/bisheng/release/hmpi` |
| MPI | `mpirun -np 1 --bind-to none` 可跑；KUPL `abi/symbols`、`dslash/wilson` PASS；**多 rank 未验证** |
| L48 输入 | 远端 `~/data/..._L48x96_cfg_7000.lime` 已存在（5.7 GB，complex128）；`prepare_case.py` 41 s 写出 `gauge_..._c64.h5`，**实为 complex128、6,115,301,376 B**；source 代码默认 complex64 |
| c64 | 已用于真实求解（§2.4）；`io.describe` 回读的原始 stdout 仍未归档 → 证据只到 C 级 |
| 两节点 | 探测 `2152277` 与两节点求解 `2153008` 均 **PENDING**；无 `execNodes`，不得写两节点成功（§2.4） |
| 并发实现 | 2026-11-17 14:0x 工作区出现**未提交**的 `pyqcu/test/driver.py`、`LATTICE_PRESETS` 的 `L48`、`native.py` HDF5 输入身份；未审计、未在远端跑过，详见设计说明 §2.4.7 |

### 2.3 操作者检查单（按顺序）

只列本任务已验证的入口；标 **未验证** 的命令不得当 PASS，也不得在没核对
`dsub --help` 时提交。远端命令都在 `login02` 的仓库根执行；地址、密钥、口令不入库。

| # | 检查 | 命令（形状） | 期望 / 判据 | 状态 |
|---:|---|---|---|---|
| 1 | 环境 | `source ./env.sh`；再按 [`docs/kupl_kunpeng_hpc_config_20261112.md`](../../../docs/kupl_kunpeng_hpc_config_20261112.md) 叠加 `PATH`/`LD_LIBRARY_PATH`/`OPAL_PREFIX`/`PYTHONPATH`；`python -V`；`python -c 'import numpy,h5py,mpi4py,torch; print(numpy.__version__, h5py.__version__, h5py.get_config().mpi, mpi4py.__version__, torch.__version__)'`；`mpirun --version` | Python 3.10.9（anaconda 2023.3）、numpy 1.23.5、h5py 3.7.0（`mpi=False`）、mpi4py 4.1.2、torch 1.12.1；HPCKit 26.1.RC1；HMPI 4.1.6rc4；`GLIBCXX_3.4.29`/bisheng `libomp`/`alllib` 齐全 | **已验证**（原始 stdout 未归档） |
| 2 | 自检 | `python -m pyqcu.test --check` | **18/18 PASS**；Python 3.10 的 `source[*tuple]` `SyntaxError` 已修 | **已验证** |
| 3 | HMPI 1 rank | `${OPAL_PREFIX}/bin/mpirun -np 1 --bind-to none python3 -m pyqcu.test --quick --backend kupl --only kupl/abi/symbols,kupl/dslash/wilson --json -` | `kupl/abi/symbols`、`kupl/dslash/wilson` 均 PASS；只证明单 rank 能启动 | **已验证**；2 rank 未验证（formal KUPL 用例声明 `mpi_single_rank`，2 rank 会 `SKIP`） |
| 4 | L48 c64 dtype | `python3 -c 'import h5py,sys; d=h5py.File(sys.argv[1],"r")["data"]; print(d.dtype, d.shape)' <gauge.h5>`；再 `python3 -c 'from pyqcu.test import io; print(io.describe("<gauge.h5>", role="gauge"))'` | 文件名不是 dtype；只有 `io.describe` 回读 `complex64`/`[3,3,4,48,48,48,96]`/`[48,48,48,96]` 才可写 c64。当前 `*_c64.h5` 实为 **complex128**（`6,115,301,376 B`），source 代码默认 complex64 | 第一条**已验证**（只证明现状）；`io.describe` 回读与 c64 cast **未验证** |
| 5 | dsub 两节点 | `dsub -s <job.sh>`，头部 `-nn 2 -N 32 -rpn 16`；提交后 `djob -ll <id>` | `execNodes` 恰好两个节点；没有 `execNodes` 只能标 `PENDING/BLOCKED`。探测 `2152277` 当前 **PENDING**（`No available node can fit ... execution nodes preference`） | **PENDING 行为已验证**；成功分配**未验证**；不得把 PENDING 写成两节点可跑 |
| 6 | 两节点前单节点冒烟 | `mpirun -np 16 ... --no-shm ...`（先 `8x8x8x16` 子体积，再 L48 局部块） | KUPL 原生 solve + 独立真残差 PASS；单节点 `--shm` 只作单节点对照，不外推到两节点 | **未验证** |

- 上述命令的远端记录见设计说明 §2.4；本轮没有重跑的命令一律按 **未验证** 处理。
- 提交两节点作业前必须先补跨节点 launcher/rankfile（设计说明 §5.3）；否则只会得到
  `SKIP(mpi_grid)` 或历史 `RC=255`，不是两节点结果。
- P0/P1 逐条状态（全部未修复）见
  [`docs/pyqcu_test_audit_risk_20261117.md`](../../../docs/pyqcu_test_audit_risk_20261117.md)；
  本清单不把任何 TODO 写成 PASS。

### 2.4 首批 L48 结果、cache 与 CG 策略（2026-11-17 归档）

权威表与逐条边界见
[`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md) §0.1；这里只留
操作者需要的判据。证据等级 **C**（远端作业，原始 `report.json` 未归档）。

| 用例 | 资源 | 结果 | 迭代 / 残差 | RSS | 状态 |
|---|---|---|---:|---:|---|
| c64 BiCGStab | 32 rank / 1 **独占**节点 | `1.6646 s` | 11 it / `7.126e-5`（strict `1e-4`） | `3.14 GiB` | 已验证（`2152757`） |
| c64 BiCGStab | 16 rank / **共享**节点 | `183.0 s` | 11 it / `7.126e-5` | `4.42 GiB` | **病态**（`2152325`），不作基线/分母 |
| c64 Legacy-MG | 16 rank | wall `110.35 s` + builder `1741.88 s`（总 `2656 s`） | `7.126e-5` | `11.93 GiB` | 已验证；cache（E=2）`1.276 GiB`；作业号 TODO |
| c128 BiCGStab | — | 无结果 | — | — | **PENDING**（`2153005`） |
| 两节点 | `-nn 2` | 无结果 | — | — | **PENDING**（`2153008`），launcher 未传 rankfile |
| CG | — | 无结果 | — | — | **组已退役** |
| dslash / clover 32r | 32 rank | 无结果 | — | — | **UNVERIFIED**（计算节点挂起；登录 smoke 不算） |
| Strict-MG | — | 不评估 | — | — | **排除** |

- **cache 只认同 identity**：`lattice`/`dtype`/`mass`/`csw`/`mg_grid`/`nv_iters`/`dof`/
  `seed`/`gauge_kind`+canonical sha/`ranks`/`grid`，且 `verify_status=0`；**不**按
  文件名/路径/大小复用。当前只有 **16 rank** cache（E=2、`1.276 GiB`）；
  32 rank / 两节点**没有匹配 cache → 暂缓，不重建**。
- **CG 已退役**：L48 没有 CG harness，后续不排 CG 用例、不新增 CG 路径。
- 未通过项：c128（`2153005`）、两节点（`2153008`）、32r dslash/clover、
  Legacy-MG 的 `2656 − 1852.23 = 803.8 s` 未分解；CSV/Excel parity 仍在验证
  （驱动自报 `--check` 19/19、`--quick` 19、87 测试，属**转述**）。

## 3. 本地可安全执行的命令（已验证形状）

以下命令不启动远端作业；`--mode list` 不写台账。

```bash
cd /home/zhangxin/PyQCU

# 1) 正式端口矩阵：19 个 tier 0 case
python -B -m pyqcu.test --list --json - | python -c \
  'import json,sys; d=json.load(sys.stdin); print(d["count"], d["lattice"], d["dtype"])'

# 2) full-mode L24 计划：19 行，但仍是 tier-0 case（不是生产 L24 求解）
python -B -m pyqcu.test --mode list --lattice-preset L24 --json - | python -c \
  'import json,sys; d=json.load(sys.stdin); print(len(d["rows"]["items"]), d["rows"]["items"][0]["tag"])'

# 3) 显式 L48 格点的 full-mode 计划：能列出 19 行，但仍是 tier-0 case
python -B -m pyqcu.test --mode list --lattice-size 48 48 48 96 --json - | python -c \
  'import json,sys; d=json.load(sys.stdin); print(len(d["rows"]["items"]), d["rows"]["items"][0]["tag"])'

# 4) 旧驱动矩阵：只列前 5 行，不执行
python -B pyqcu/testing/kupl/test_kunpeng_all.py --mode list --expand --limit 5

# 5) 只查看帮助，不跑数值
python -B pyqcu/testing/kupl/test_kunpeng_all.py --help
python -B pyqcu/testing/simple/prepare_case.py --help
```

注意：`--mode list --lattice-preset L24` 的 row 里 `ranks`/`shm` 即使通过
`--ranks 16 --shm` 传入也只是记录字段；内置 tier-0 case 仍是单进程执行，不能当作
L24/L48 生产 MPI 行。

## 4. 明确不要执行（会失败或语义错误）

| 命令/写法 | 原因 |
|---|---|
| `python -m pyqcu.test --mode list --lattice-preset L48` 期望 L48 生产行 | 并发实现后 argparse 已接受 `L48`，但仍只展开 19 个 tier-0 case（`torch/*`、`kupl/*`、`cuda/*`），不是 L48 生产 `solve/mg` 行 |
| `python -m pyqcu.test --mode prewarm --gauge-file foo.lime` | full mode 的 `--gauge-file` 是 HDF5 reader；LIME 在 `io.is_readable` 阶段被拒绝。普通运行的 `--gauge-file` 也不替换 `--gauge` 生成场 |
| `python -m pyqcu.test --mode validate` 当作数值冒烟 | 正式端口的 `validate` 是台账一致性校验，空台账返回 1 |
| `test_kunpeng_all.py --mode list --expand --lattices L48` | `LATTICES` 字典没有 L48，`KeyError`；不加 `--expand` 时该模式只用 `--lat` 的单行，不触发字典 |
| 两节点作业使用 `--shm` 并期望共享 halo | `native.resolve_shm` 在 rank 跨节点时 `SKIP(shm_unavailable)`；默认用 `--no-shm` |
| 旧脚本 `submit_kunpeng_all.sh` 直接申请两节点 | 它写死 `#DSUB -nn 1`，`-N/-rpn` 是单节点形态；必须先加 `--nodes`/`--ranks-per-node` |
| 把作业 `2152277` 的 PENDING 当作两节点成功 | 它从未分配到节点（`No available node can fit ... execution nodes preference`），`execNodes` 为空 |
| 把 `2153008` 的 PENDING 当作两节点失败结论 | 请求语法正确、只是没排到节点；它既不证明能跑，也不证明不能跑 |
| 用 16 rank/共享节点的 `183.0 s` 当 L48 基线或与 32r `1.6646 s` 算比值 | 节点独占性、rank 数与排队窗口都不同；共享节点是病态配置（§2.4） |
| 把 `prepare_case.py` 直接写出的 `gauge_..._c64.h5` 当 c64 用 | 该**特定产物**实为 complex128（`6,115,301,376 B`）；c64 行必须用另行 cast 并回读过的文件，文件名不是 dtype 证据 |
| 本轮改版本号、打 tag 或推 PyPI | 当前发布固定为 `0.1.17` / tag `test37`；发布门（含十轮复审）未过 |

## 5. 远端检查清单（已完成项保留为记录，未完成项仍是 TODO）

### 5.1 访问与数据

- [x] 入口已确认：`login02` / `nsls_yangyb` + 用户提供的内部路由；`lingsheng` 只是
      别名缺失，不是访问方式。地址/密钥不入库。
- [x] `hostname; pwd; df -h ~/data` 成功；`/home/share` 实测 366 T 可用。
- [x] 远端已有 `beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime`，大小
      `6115435952` 与前 4 KiB 哈希与本地一致；**不需要上传**。
- [ ] 全文件 `sha256sum` 未比对。

### 5.2 远端环境探测

- [x] `h5py` 3.7.0 / HDF5 1.10.6 / `get_config().mpi == False`：没有并行 HDF5，
      多 rank cache 构建与 MPIO slab 读取不可用当前实现。
- [x] `mpirun` = HMPI（Open MPI 4.1.6rc4）；`CCS_ALLOC_FILE` 每行
      `host total_cores rank_count`（另有 `CCS_NTASKS_PER_NODE=16`）。
      `CCS_TASK_REPLICA` 只在手册模板出现，本作业未观测。
- [x] `dsub -h` / `djob` / `dinfo` / `dnode` 可用；`q_hpcapp` OPEN/ACTIVE，
      同时刻只有 7 个 `OK` 节点（其余 `CLOSE_FULL`/`CLOSE_ADMIN`/`UNAVAILABLE`）。
- [ ] `login02` → 计算节点免密（当前 `Permission denied (publickey)`）；不补齐则
      手册模板的 `--mca plm_rsh_agent /usr/bin/ssh` 无法跨节点起 rank。

### 5.3 输入转换与正式端口校验

- [x] 远端 `prepare_case.py` 已跑通 L48：`--lime <L48> --lat 48 48 48 96
      --mass -0.277 --csw 1.160920226 --seed 42 --out-dir <out>`，41 s。
      但产物 `gauge_48x48x48x96_c64.h5` **实为 complex128**（`6,115,301,376 B`）。
- [ ] c64 cast 完成并用 `pyqcu.test.io.describe(..., role="gauge")` 回读
      `complex64` / shape `[3,3,4,48,48,48,96]` / `lattice [48,48,48,96]`；
      **在此之前不得声称已有 c64 gauge**。
- [ ] 记录 gauge/source 的 `sha256`、字节数、生成耗时；不要用文件名判断 dtype
      （本机与远端各踩过一次）。
- [ ] `prepare_case.py` 缺少/未验证的 `role`/`layout` 属性按 full 布局兼容规则确认，
      必要时用 `pyqcu.test.io.save_gauge/save_source` 重写。

> c64 转换的远端形状（已验证可行、尚未完成）：先把误标文件改名为 `..._c128.h5`，
> 再按 t-slab（4 slice 一块）写成 `complex64`，复制原属性并追加
> `source_dtype`/`cast_to`，最后 `os.replace` 原子落盘；不要一次性把 6 GB 读进内存
> 再 cast。

### 5.4 单节点冒烟（先于两节点）

- [ ] 单节点 2 rank、`8x8x8x16` 子体积或 L48 局部块，`--no-shm`，确认 KUPL
      原生 solve 和独立真残差路径通过。
- [x] 单节点 32 rank、L48、c64：BiCGStab `1.6646 s`/11 it/`7.126e-5`（`2152757`，
      独占节点）；Legacy-MG 16 rank wall `110.35 s`（§2.4）。**证据等级 C**。
- [ ] 单节点 16 rank、L48、`grid 2 2 2 2`、`--no-shm`，确认输入读取和
      rank-local slab 正确——现有的 16 rank 样本是**共享节点**（`183.0 s`），
      只能当病态对照，**不能**算作本项通过。
- [ ] 单节点 `--shm` 对照只作单节点证据，不得外推到两节点。
- [ ] 单节点 32 rank 的 dslash/clover：计算节点**挂起**，UNVERIFIED；登录节点
      smoke 不能替代。

### 5.5 两节点作业（模板未验证；`2152277` 与 `2153008` 均 PENDING）

```bash
# 以下仅为手册语义模板；提交前必须核对 dsub --help 与本账号实际行为。
# <job.sh> 必须是“一个 task 内启动一条 mpirun”的脚本，不能让每个 task 都跑驱动。
dsub -q q_hpcapp -nn 2 -N 32 -rpn 16 -R "cpu=18;mem=8G" -x job -s <job.sh>

# job.sh 内部示意（需要先由 CCS_ALLOC_FILE 生成 rankfile）：
mpirun -np 32 --rankfile "$PWD/${CCS_JOB_ID}-rankfile" \
  --map-by ppr:16:node --bind-to none --allow-run-as-root \
  python -B -m pyqcu.test --mode solve --outdir <outdir> --lattice-preset L48 ...
```

> `--lattice-preset L48` 已由并发批次加入（未提交，见 §2.2），但 `--mode list/run`
> 仍只走 19 个 tier-0 case，不会产生 L48 生产 `solve/mg` 行。当前只能用
> `--lattice-size 48 48 48 96` 表达几何，语义同上。

- [x] 语义已核对：`-nn` 节点数、`-N` 总副本数、`-rpn` 每节点副本数（`dsub -h` +
      手册 `p.24`）；`#DSUB` 只在 `dsub -s` 下生效。
- [x] 已提交两节点探测作业 `2152277`（`dsub -s`）：结果 **PENDING**，
      `No available node can fit ... execution nodes preference`。
- [ ] 两节点求解作业 `2153008` 仍 **PENDING**（请求语法正确，卡队列/限额）；
      launcher **未传 rankfile**，即使分到两节点也不能起跨节点 rank。
- [ ] `djob -ll <job_id>` 的 `execNodes` 恰好两个节点；记录节点名、作业号、起止时间。
- [ ] `dattach <job_id>` 或 `dattach -en "<node>" <job_id>` 检查每 rank 日志。
- [ ] 第一次两节点只跑 **1 个 row**（`--limit 1` 或在 `--sweep` 中只选一项），
      通过后再扩大矩阵。
- [ ] 失败时保留 `rows/`、`reports/`、`logs/`、`.err`，不要删除现场。

## 6. 建议的 L48 row 参数

以下只是候选，不是已通过配置；L48 的 `t-finest` 在 32 rank 下局部 `t=3` 为奇数，
不可用：

| 项目 | 候选 | 备注 |
|---|---|---|
| lattice | `48 48 48 96` | LIME 元数据确认 |
| mass / csw | `-0.277` / `1.160920225783` | LIME `Mass`/`clovCoeff` |
| dtype | `c64`（已实际用于 2026-11-17 的求解） | 但 `io.describe` 回读的原始 stdout 未归档；没有该证据时只能标 C 级，不能外推到 c128 |
| ranks / grid | `32` / `2 2 2 4` | 16 rank/节点；局部 `24^3x24` |
| threads | `18` | `32*18=576 <= 608`；需实测 cpuset |
| shm | `0`（两节点） | 跨节点共享 halo 不可用 |
| solver | `bistabcg` 或 `mg2` | MG 需要 L48 cache；先 BiCGStab |

## 7. 证据记录模板

每个 L48 row 至少记录：

```text
job_id, submit_command, exec_nodes, ranks, ranks_per_node, grid, threads, shm
gauge_path, gauge_bytes, gauge_sha256, gauge_dtype, gauge_format
source_path, source_bytes, source_sha256, source_seed
mass, csw, kappa, atol, max_iter, solver, mg_levels, mg_restart
wall_seconds, elapsed_seconds, iterations, final_true_residual, residual_source
status, reason, report_path, solution_path
```

缺少 `exec_nodes`、`sha256` 或独立残差的数字，只能标 `UNVERIFIED`，不能写成性能结论。

## 8. 相关入口

- 设计说明（含远端基线 §2.4、既有材料 §2.5）：[`docs/pyqcu_test_l48_remote.md`](../../../docs/pyqcu_test_l48_remote.md)
- 审计 P0/P1 风险登记（全部未修复）：[`docs/pyqcu_test_audit_risk_20261117.md`](../../../docs/pyqcu_test_audit_risk_20261117.md)
- 上一轮远端任务日志：[`logs/v202611171402.txt`](../../../logs/v202611171402.txt)；上一轮归档日志：[`logs/v202611171416.txt`](../../../logs/v202611171416.txt)；本轮结果归档日志：[`logs/v202611171745.txt`](../../../logs/v202611171745.txt)
- 旧驱动：[`pyqcu/testing/kupl/test_kunpeng_all.py`](../../../pyqcu/testing/kupl/test_kunpeng_all.py)
- 旧提交脚本：[`pyqcu/testing/kupl/submit_kunpeng_all.sh`](../../../pyqcu/testing/kupl/submit_kunpeng_all.sh)
- 正式端口 README：[`pyqcu/test/README.md`](../../../pyqcu/test/README.md)
- LIME 读取：[`pyqcu/testing/simple/_lime.py`](../../../pyqcu/testing/simple/_lime.py)
- LIME → HDF5：[`pyqcu/testing/simple/prepare_case.py`](../../../pyqcu/testing/simple/prepare_case.py)
- L48 容量/缓存预算（只引用）：[`docs/qcu_quda_solver_dcu_l48_20261029.md`](../../../docs/qcu_quda_solver_dcu_l48_20261029.md)
- LIME 转换调用形状：[`docs/kupl_only_deployment.md`](../../../docs/kupl_only_deployment.md)
- LIME source/layout 口径：[`docs/kupl_kunpeng_progress_20261110.md`](../../../docs/kupl_kunpeng_progress_20261110.md)
- dtype/文件名误标根因：[`docs/kupl_hdf5_source_gauge_rootcause_20261115.md`](../../../docs/kupl_hdf5_source_gauge_rootcause_20261115.md)
- CSV/Excel 台账（实现中，未接线）：[`pyqcu/test/ledger_csv.py`](../../../pyqcu/test/ledger_csv.py)、
  [`pyqcu/test/ledger_excel.py`](../../../pyqcu/test/ledger_excel.py)
- 手册：[`refer/kunpeng/深圳超算二期用户手册.pdf`](../../../refer/kunpeng/深圳超算二期用户手册.pdf)
