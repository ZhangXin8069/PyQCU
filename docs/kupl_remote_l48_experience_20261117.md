# L48 / LX2 远端实测经验（2026-11-17）

> **性质**：2026-11-17 `L48 + remote + pyqcu.test` 会话的**经验与判定规则**汇总。
> 本文不复制设计正文、不重复逐条状态表：设计、接口契约与权威状态见
> [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md)（§0、§0.1、§2.4）；
> 发布前终审见 [`pyqcu_test_l48_review.md`](pyqcu_test_l48_review.md)；
> P0/P1 风险登记见 [`pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md)；
> 操作者最短检查单见
> [`skills/pyqcu-testing/references/l48-remote.md`](../skills/pyqcu-testing/references/l48-remote.md)
> 与本次新增的
> [`skills/pyqcu-testing/references/remote-l48-experience.md`](../skills/pyqcu-testing/references/remote-l48-experience.md)。
>
> **边界**：只记录可复核事实、失败路径与复现命令。不写私网地址、密钥、口令或
> 账号凭据；远端路径保留到“目录/作业号”粒度。所有未验证项显式标 `UNVERIFIED` /
> `PENDING` / `BLOCKED` / `DEFERRED`；`PENDING` 不是 PASS，`登录节点 smoke` 不是
> 计算节点结果，`文件名` 不是 dtype，`version 字符串` 不是制品身份。
>
> **2026-11-17 23:30 补充**：KUPL harness 两节点已从单条 c64 32r 扩到**四个 PASS**
> （c128 32r `2155785`、c64 16r `2155830`、c64 64r `2155868`、c64 Legacy-MG 16r
> cache-load `2155888`，矩阵见 §7.6）。正式 `pyqcu.test --l48-two-node` 在远端覆盖树
> **仍不存在**（`unrecognized arguments: --l48-two-node`，RC=2），因此**正式矩阵五行
> 仍是 `UNVERIFIED`**；MG 只证明 cache-load + solve 路径，粗修正未生效。

## 0. 状态标记与读法

| 标记 | 含义 | 允许的写法 |
|---|---|---|
| **VERIFIED** | 远端作业/命令或本机可复跑命令直接产出，附 job id、报告或命令 | 可写成事实，但必须带证据路径 |
| **VERIFIED-C** | 事实有远端记录，但原始 `stdout`/`report.json` 未入库，只能按 C 级时点证据引用 | 必须写“原始输出未归档” |
| **PENDING** | 作业已提交、未被分配（无 `execNodes`） | 只能写“已受理、未运行”，不得填数值 |
| **BLOCKED** | 缺 harness、launcher、凭据或输入，无法执行 | 写阻塞原因与解除条件 |
| **DEFERRED** | 本轮明确不做，不是失败 | 写“暂缓”与重启条件 |
| **UNVERIFIED** | 有间接证据或推测，但无直接测量 | 不得当结论、不得参与比值 |
| **TODO** | 尚未执行或尚未归档 | 保留为待办 |

**三个分母不能混读**：远端 0.1.17 覆盖树的 `--check` 是 **18/18**；本机当前工作树
（含未提交并发改动）在 17:55 是 **20/20**（17:47 曾为 19/19，并发批次又加了一项检查）；
`--quick`/`--list` 是 **19 个 tier-0 case**。引用时必须写清环境与时间戳。详见 §3。

## 1. 远端访问与运行时（对应经验 1）

### 1.1 访问

| 项目 | 事实 | 状态 | 证据 / 边界 |
|---|---|---|---|
| 登录入口 | `login02`，账号 `nsls_yangyb`，经用户提供的内部路由 | **VERIFIED-C** | `hostname; whoami`；私网地址、密钥不入库 |
| `ssh lingsheng` | 本机 DNS 不解析；`~/.ssh/config`、`/etc/hosts`、`ssh-keygen -F` 均无此别名 | **VERIFIED** | 别名缺失 ≠ 集群不可达；实际入口是 `login02` |
| 前提 | 必须先连用户 VPN/内部路由；未连通表现为 `No route to host`/超时 | **VERIFIED** | 首次探针失败、路由就绪后成功 |
| 远端仓库 | `/home/share/nsls_yangyb/PyQCU-main`，**非 git 仓库** | **VERIFIED** | `git -C` 报 not a git repository |
| 代码底座 | **0.1.15 快照 + 0.1.17 Python 代码手工覆盖**；覆盖前备份 `pyqcu.bak.20261117` | **VERIFIED** | 远端目录；不是 PyPI `0.1.17` 制品 |
| 代码身份 | 会话开始时 `pyqcu/testing/simple/kupl_multigrid.py` md5 `c38aefb89962be8b40e8798d86581233`，与当时本机工作树一致 | **VERIFIED（时点）** | `md5sum`；**17:46 后本机该文件被并发批次修改，17:55 实测 md5 `87a43f447ea5d518076e06a7f10c31bc`**，远端仍是 `c38aef…`；复跑前必须重新比对，不能把“当时一致”当“现在一致” |
| 数据盘 | `/home/share` 为 `dtfs`（自定义内核客户端，不是 FUSE/NFS/Lustre） | **VERIFIED** | `/proc/mounts`、`stat -f` |
| 登录节点 | 128 CPU、约 502 GB 内存 | **VERIFIED** | `nproc`、`free -g` |

> **教训**：远端“0.1.17”只是版本字符串。它是 0.1.15 底座上的手工覆盖树，既不是
> git 快照，也不是 PyPI `0.1.17` sdist（后者只含 16 个 `pyqcu/test/*.py`，工作树是
> 19 个）。任何“0.1.17 已验证”的表述都必须追加“远端覆盖树”，并用 md5/SHA 绑定到
> 具体文件。

### 1.2 运行时版本

| 组件 | 版本 | 状态 | 备注 |
|---|---|---|---|
| Python | 3.10.9，aarch64，anaconda 2023.3 | **VERIFIED-C** | `python -V`、`uname -m` |
| numpy | 1.23.5 | **VERIFIED-C** | `import numpy` |
| h5py | 3.7.0，HDF5 1.10.6，`h5py.get_config().mpi == False` | **VERIFIED-C** | 无并行 HDF5；`mpio` 路径不可用 |
| mpi4py | 4.1.2 | **VERIFIED-C** | 原始 stdout 未归档 → 只能 C 级 |
| torch | 1.12.1 | **VERIFIED-C** | 同上 |
| HPCKit | 26.1.RC1（25.1.0/25.2.1 并存） | **VERIFIED** | `/work_ssd/software/HPCKit/26.1.RC1` |
| MPI | HMPI（Open MPI 4.1.6rc4） | **VERIFIED** | `mpirun --version` |
| 调度器 | CCS/Donau：`dsub`/`djob`/`dnode`/`dqueue`/`duser`，队列 `q_hpcapp` | **VERIFIED** | `/opt/batch/cli/bin` |

> **教训**：`mpi4py`/`torch` 版本只有口述记录，没有原始探针输出；报告里不能把这两行
> 和 `h5py` 的 `VERIFIED` 混成同一等级。

## 2. HPCKit / HMPI 环境（对应经验 2）

### 2.1 必须叠加的四段库路径

`source ./env.sh` 只提供仓库默认值，**不足以**跑 HPCKit 26.1.RC1 的 HMPI + KUPL。
远端作业实际使用（`<HK>` = `/work_ssd/software/HPCKit/26.1.RC1`）：

| 变量 | 值 | 为什么需要 | 状态 |
|---|---|---|---|
| `PATH` | `<HK>/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:/opt/batch/cli/bin:$PATH` | 选对 `mpirun`、`python3`、调度器 CLI | **VERIFIED** |
| `LD_LIBRARY_PATH` | `<HK>/hmpi/bisheng/release/hmpi/lib` | HMPI 运行库 | **VERIFIED** |
| 同上 | `<HK>/compiler/gcc/lib64` | 提供 `GLIBCXX_3.4.29`；系统 `libstdc++` 版本不足 | **VERIFIED** |
| 同上 | `<HK>/compiler/bisheng/lib` | 提供 `libomp`（KUPL 的 OpenMP 运行时） | **VERIFIED** |
| 同上 | `/work_ssd/software/HPCKit/alllib` | 补齐其余第三方库 | **VERIFIED** |
| `OPAL_PREFIX` | `<HK>/hmpi/bisheng/release/hmpi` | 缺它时 `mpirun` 的选项会被忽略或报错 | **VERIFIED** |
| `PYTHONPATH` | 仓库根（`$PWD`） | 远端非安装态，直接跑源码树 | **VERIFIED** |

典型故障：缺 `compiler/gcc/lib64` → `GLIBCXX_3.4.29 not found`；缺
`compiler/bisheng/lib` → `libomp` 加载失败；缺 `OPAL_PREFIX` → `mpirun` 行为异常。
这三类都不是代码缺陷，先查环境再查代码。

### 2.2 MPI 形态

| 形态 | 命令形状 | 状态 | 边界 |
|---|---|---|---|
| 单 rank | `mpirun -np 1 --bind-to none python3 -m pyqcu.test --quick --backend kupl --only ...` | **VERIFIED** | `kupl/abi/symbols`、`kupl/dslash/wilson` PASS；只证明单 rank 能启动 |
| 多 rank 正式 KUPL 用例 | `--quick --backend kupl` 在 `-np 2` | **BLOCKED（设计如此）** | 正式 KUPL dslash/solve 用例声明 `mpi_single_rank`，2 rank 会 `SKIP`；多 rank 必须走 `pyqcu/testing/simple/*` 或 `kupl_qcu_tests` |
| 跨节点 16/32/64 rank | `CCS_DRUN_RANK_FILE` rankfile + `drun -mpi pmix` | **VERIFIED（KUPL harness）** | `2155536`（c64 32r）、`2155785`（c128 32r）、`2155830`（c64 16r）、`2155868`（c64 64r）、`2155888`（c64 MG 16r）；全部 `status=PASS`、独立真残差 `< 1e-4`（§7.5、§7.6）。正式 `pyqcu.test` 矩阵行仍 **UNVERIFIED**；SSH 直接 shell 不可用 |

### 2.3 UCG 26.0/26.1 警告：非致命，但必须当性能证据的“环境缺陷”记账

每个 rank 在 MPI init 时打印：

```text
ucg_context.c:123  UCG  ERROR UCG version is incompatible, required: 26.1, actual: 26.0.RC1
[coll_ucg_module.c:264] ERROR UCG context init failed: Version incompatible
```

| 项目 | 事实 | 状态 |
|---|---|---|
| 计数 | 16 rank 作业 16+16 行；32 rank 作业 32+32 行；**每 rank 一行** | **VERIFIED** |
| 后果 | HMPI 的 UCG collective 组件初始化失败后回退到其他 collective；作业继续，数值 PASS | **VERIFIED** |
| 时间预算 | 25–47 次 reduction × 0.45–1.02 ms ≈ 0.011–0.048 s，占 183 s 的 **0.006–0.026%** | **VERIFIED（按文档参数估算）** |
| 能否解释 183 s | **不能**；差 3–4 个数量级 | **VERIFIED** |
| 性能相关性 | 它标记 HPCKit 安装不一致（HMPI 26.1 链接 UCG 26.0.RC1）；**引用任何性能数字前应记录并优先修复** | **VERIFIED-C** |
| A/B 关闭 UCG（`--mca coll ^ucg`） | 本轮未做 | **TODO / UNVERIFIED** |

> **教训**：不要把 UCG 警告当“无害噪音”直接略过，也不要把 183 s 甩锅给 UCG。
> 正确写法是：“非致命、已按 rank 计数；时间预算 ≤0.03%，不能解释异常；作为安装
> 不一致记入性能证据的 caveat”。

## 3. Python 3.10 可移植性（对应经验 3）

### 3.1 实际故障

远端 Python 3.10.9 在导入 `pyqcu/test/kupl_qcu.py`、`cuda_qcu.py` 时直接
`SyntaxError`，位置是：

```python
source[0, spin, color, *site]
```

这是 PEP 646 的 **starred subscript**，Python 3.11 起才合法；3.10 解析器不接受。
它让 `python -m pyqcu.test --check` 在远端完全跑不起来。

### 3.2 修复

先拼元组再索引，语义不变：

```python
site = (0, spin, color, *site)
source[site] = ...
```

修复后远端覆盖树 `--check` 从 0 变 **18/18 PASS**。注意：本仓库同一改动**未提交**；
远端是手工覆盖的，不能据此说已发布的 0.1.17 支持 3.10。

### 3.3 语法闸门：`ast.parse(feature_version=(3, 10))` 不够

| 事实 | 状态 | 证据 |
|---|---|---|
| `ast.parse('x[1:2, *ys]', feature_version=(3, 10))` 在 CPython 3.12/3.13 **仍然接受** | **VERIFIED** | 3.12 实测；真实 3.10.22 拒绝 |
| 仅靠 `feature_version` 会漏掉本次故障 | **VERIFIED** | 同上 |
| 需要第二层结构性 PEP 646 扫描（starred subscript item / starred `*args` 注解） | **VERIFIED** | `pyqcu/test/selfcheck.py` 的 `check_py310_syntax()` |
| 若解析器不遵守 `feature_version`，检查必须返回 **SKIP**（带原因），不得返回 PASS | **VERIFIED** | 回归测试 patch `ast.parse` 去掉 flag |
| 括号元组 `source[(0, spin, color, *site)]` 是 3.10 合法的，不能误报 | **VERIFIED** | `_is_parenthesized()` 用 `ast.get_source_segment()` 区分 |
| 回归测试 | `pyqcu/testing/python/test_pyqcu_test_syntax_floor.py`，6 cases | **VERIFIED**（本地 6 passed，2.36 s） |

### 3.4 精确计数（2026-11-17 17:47–17:55 CST）

| 环境 | 命令 | 结果 | 状态 |
|---|---|---|---|
| 远端 0.1.17 覆盖树 | `python3 -m pyqcu.test --check` | **18 checks / 18 passed / 0 skipped / 0 failed** | **VERIFIED-C**（原始 stdout 未归档） |
| 本机当前工作树（17:47） | `python -B -m pyqcu.test --check` | **19 checks / 19 passed** | **VERIFIED**（时点；之后并发批次又改） |
| 本机当前工作树（17:55，最新） | `python -B -m pyqcu.test --check` | **20 checks / 20 passed / 0 skipped / 0 failed** | **VERIFIED**（17:55 实跑；并发批次仍在改 `pyqcu/test/**`，引用须带时间） |
| 本机 tier-0 矩阵 | `python -B -m pyqcu.test --list --json -` | **19 cases**，`[8,8,8,16]`，c64（11 torch / 5 kupl / 3 cuda） | **VERIFIED** |
| `--quick` | 同上矩阵 | **19 个 tier-0 case** | **VERIFIED** |
| 驱动侧“87 tests passed” | 转述自并发批次，本轮未独立复跑 | **UNVERIFIED** |

> **教训**：远端 18 与本机 19/20 的差值是并发批次新增的检查（先是 `py310_syntax`，
> 后又多 1 项），不是
> “远端少跑了一个用例”。写报告时必须写“远端覆盖树 18/18；本机 20/20（17:55）；
> 两者不可混读”。`--quick`/`--list` 的 19 是 **case 数**，不是 check 数。check 数会
> 随并发批次新增检查而漂移（本日 17:47 为 19、17:55 为 20），引用必须带时间。

## 4. L48 输入：LIME、dtype 与 HDF5 读取（对应经验 4）

### 4.1 LIME 事实

| 项目 | 值 | 状态 |
|---|---|---|
| 远端路径 | `~/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime` | **VERIFIED** |
| 大小 | `6,115,435,952 B`（约 5.70 GiB） | **VERIFIED** |
| 维度 | `48 × 48 × 48 × 96` | **VERIFIED** |
| 精度 | `precision = D`（complex128） | **VERIFIED** |
| `datacount` / `colors` | `4` / `3` | **VERIFIED** |
| `Mass` / `clovCoeff` | `-0.2770` / `1.160920225783` | **VERIFIED** |
| 与本地一致性 | 同大小、同前 4 KiB SHA256；**全文件 SHA256 未比对** | **部分 VERIFIED / UNVERIFIED（全文件）** |

### 4.2 canonical 转换与 dtype 陷阱

转换命令形状（`prepare_case.py`）：

```bash
python3 pyqcu/testing/simple/prepare_case.py \
  --lime ~/data/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_7000.lime \
  --lat 48 48 48 96 --mass -0.277 --csw 1.160920226 --seed 42 \
  --out-dir ~/PyQCU-main/data/l48_canonical_20261117
```

| 事实 | 状态 | 证据 / 边界 |
|---|---|---|
| 转换耗时 `41 s`（单次，读 LIME + 写 HDF5） | **VERIFIED-C** | 本批次口述记录；原始 stdout 未归档 → 计时只能 C 级 |
| 产物文件名 `gauge_48x48x48x96_c64.h5` | **VERIFIED** | 但文件名**不表示 dtype** |
| 该文件实际 dtype | **complex128**，`6,115,301,376 B` | **VERIFIED**（`h5py` 读 `dataset.dtype`） |
| 结论 | 文件名误标；与 [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) 的 L24/L32 同类缺陷一致 | **VERIFIED** |
| 真正 c64 cast | 先把误标文件改名为 `..._c128.h5`，再按 t-slab（4 slice 一块）写 `complex64`；复制属性并追加 `source_dtype`/`cast_to`；`os.replace` 原子落盘 | **VERIFIED**（执行方式） |
| cast 校验 | 382,205,952 elements，`mismatches 0`，`max_abs_diff_vs_cast 0.0`，`zero_slabs 0`，`nonfinite 0`，`status BITEXACT`，52.6 s | **VERIFIED**（`logs/verify_c64.log`） |
| `io.describe` 回读 | `dtype=complex64`，`shape=[3,3,4,48,48,48,96]`，`lattice=[48,48,48,96]`，dataset bytes `3,057,647,616`，attrs `source_dtype=complex128`、`cast_to=complex64`、`mass=-0.277`、`csw=1.160920226` | **VERIFIED-C**（原始 stdout 未归档） |

**并发写陷阱**：第一次 c64 cast 以 orphan 进程继续运行（约 13:52 启动、13:59 完成），
期间 `.partial` 被 HDF5 锁住；另一个 writer 会得到
`BlockingIOError: unable to lock file, errno = 11`。**不要**在已有 writer 未退出时
再启动第二个 cast；先 `ps`/锁检查，再决定等待或重跑。

### 4.3 canonical 文件身份

| 对象 | 文件字节 | dtype | shape | SHA256 | 状态 |
|---|---:|---|---|---|---|
| `gauge_..._c128.h5` | 6,115,301,376 | complex128 | `[3,3,4,48,48,48,96]` | `b931b0cf0b00e6f3e323535b71ba0e04a8a7c329fdbc43a8299012e2bae8e9b3` | **VERIFIED** |
| `gauge_..._c64.h5` | 3,057,653,856 | complex64 | 同上 | `c8e8df3b18276d40b397f1f25ab19177a6bc89ddb292a813aef500d8d3f64b54` | **VERIFIED** |
| `source_..._c64.h5` | 1,019,222,016 | complex64 | `[4,3,48,48,48,96]` | `e4dc2a4c275780014d190ee675eed65adeb6dbb2c6c72062ccefc6b4cc27bcbf` | **VERIFIED** |
| `source_..._c128.h5` | 2,038,461,656 | complex128 | `[4,3,48,48,48,96]`，chunks `[1,1,48,48,48,8]` | `3e07d9109ee13ddc37f5a2ce10a54cca829f6eade24cc9eb0ad5dd7e3bb6e3f9` | **VERIFIED** |

> **source 语义**：c64 source 是 `numpy PCG64 normal float64 → astype(complex64)`、
> `seed=42` 的**自造源**，不是物理源；c128 source 是 c64 的**精确 widening**
> （每个 complex64 值都能精确表示为 complex128），16.299 s，`slice_equal true`。
> 它用于隔离 dtype 变量，不是新物理。
>
> **manifest 漂移**：`case_manifest.json` 仍把 `gauge` 指向 `*_c64.h5` 名字，却记录
> c128 的 size/SHA。引用 manifest 前先核对 `dataset.dtype` 与 SHA；不要把 manifest
> 当 dtype 证据。

### 4.4 HDF5 布局与 rank-local 读取放大

`gauge`/`source` 的 HDF5 dataset 都是 **contiguous、无 chunks、无压缩**；最后四轴
`(x, y, z, t)` 中 **`t` 最快**。这直接导致 rank-local hyperslab 的读取放大：

| 事实 | 数值 | 状态 |
|---|---:|---|
| 16 rank、grid `2 2 2 2`、1/16 slab 逻辑字节 | `191,102,976 B` | **VERIFIED** |
| 同 slab 的 HDF5 分解 | `3,981,312` 个 **384 B** elementary run（每 96 个 complex64 只用 48 个，384 B 用 + 384 B 空） | **VERIFIED** |
| HDF5 合并后的 `pread` 数 | `10,380` 次（平均约 65 KB） | **VERIFIED** |
| `rchar`（实际拉取） | `679,480,530 B` | **VERIFIED** |
| 读放大 | **3.56×**（679 MB 拉取交付 191 MB） | **VERIFIED** |
| 1/32 slab（grid `2 2 2 4`） | 95,551,488 B，`rchar 679,480,534 B` → **7.1×** | **VERIFIED** |
| 全文件顺序读（同 3.06 GB） | `0.506 s`（21 次 `pread`，5762 MiB/s） | **VERIFIED** |
| 16 rank 全文件各读一遍 | `4.603 s`（46,656 MiB，10,136 MiB/s）——比 16 个 1/16 hyperslab 的 `36.2 s` 快约 9× | **VERIFIED** |

**结论**：问题不是带宽，也不是“每个 rank 读了全文件”；是 **HDF5 小请求序列化 +
`dtfs` 每次 `pread` 的固定成本**。`h5py` 没有 MPI，`mpio` 也只会改变“谁发请求”，
不会自动合并 384 B 请求。

### 4.5 `dtfs` 间歇性停顿

| 事实 | 数值 | 状态 |
|---|---|---|
| 相同工作、相同 syscall 数、相同 CPU 时间，wall 差 | **+33…+36 s** | **VERIFIED** |
| 观测到的停顿幅度 | `32.68, 33.32, 33.44, 35.55, 35.62, 36.06 s` | **VERIFIED** |
| 频率 | 首批 0/10；仪器化批次 3/12；下一批 0/22；并发时 1/16、2/16、32/32 ranks | **VERIFIED** |
| 机制判断 | 固定 ~33 s 惩罚、与请求数/字节数无关 → **storage-client timeout/retry**，不是调度或带宽 | **UNVERIFIED（机制推断）** |
| 影响 | 183 s 的 16r 作业里 `gauge_read 43.764 s`，干净窗口同 slab 只需 2.4–2.9 s；差约 33–41 s | **VERIFIED**（数字）/ **UNVERIFIED**（归因到单一机制） |

### 4.6 可落地的规避：`--gauge-slab-dir` / `--source-slab-dir`

`kupl_multigrid.py` 已实现 rank-local 连续 slab 文件分支（`_read_local_slab` 的
`slab_dir` 分支）；生成器是仓库内已有的
[`pyqcu/testing/simple/prepare_mpi_slab.py`](../pyqcu/testing/simple/prepare_mpi_slab.py)：

```bash
python3 pyqcu/testing/simple/prepare_mpi_slab.py \
  --input data/l48_canonical_20261117/gauge_48x48x48x96_c64.h5 \
  --lat 48 48 48 96 --grid 2 2 2 4 --output-dir <slab-dir> \
  --kind gauge --dtype c64 --hash-input

python3 pyqcu/testing/simple/kupl_multigrid.py ... \
  --gauge-slab-dir <gauge-slab-dir> --source-slab-dir <source-slab-dir>
```

| 事实 | 数值 | 状态 |
|---|---:|---|
| 16 rank per-rank slab 文件，cold，本地 ext4 | `3.855 s`（757 MiB/s） | **VERIFIED**（I/O probe） |
| 同上，warm | **`0.254 s`** | **VERIFIED** |
| 用户既有 slab-dir 作业 warm | `0.034–0.048 s` | **VERIFIED-C**（用户记录） |
| 预期收益 | 54.3 s → 0.05–0.3 s warm；183 s 作业 → 约 129 s | **UNVERIFIED（预测，未在 L48 32r 复跑）** |
| 与 `--write-coarse-cache` 的组合 | 文档注明 `--write-coarse-cache` 不支持与 slab-dir 同用 | **UNVERIFIED（未实测；需按 `COMMAND_REFERENCE.md` 复核）** |

> **选择规则**：fine BiCGStab/性能行优先用 slab-dir 去掉小请求放大；MG cache 构建
> 需要 `--write-coarse-cache` 时，二者不能同时选，必须单独记录用了哪条路径。

## 5. 单节点性能结果与“不能做的比较”（对应经验 5）

### 5.1 结果表

| 用例 | 作业 / 节点 | 资源 | grid / local | solver wall | it | true residual | 峰值 RSS | 状态 / caveat |
|---|---|---|---|---|---|---|---|---|
| c64 BiCGStab | `2152757` / `cn12072` | 32 rank，**1 节点独占**，18 t | `2 2 2 4` / `24³×24` | **1.6645688 s** | 11 | `7.12609211762054e-05`（strict `1e-4`） | `3,288,864 KiB = 3.137 GiB` | **VERIFIED**；单样本 |
| c64 BiCGStab | `2152325` / `cn12038` | 16 rank，**1 节点共享**，18 t | `2 2 2 2` / `24³×48` | **183.0004 s** | 11 | `7.125789e-05`（非 strict gate `2.44140625e-4`） | `4,426,012 KiB = 4.221 GiB` | **VERIFIED 但病态**；不作基线、不作分母 |
| c64 “Legacy-MG” | `2152391` / `cn12038` | 16 rank，1 节点共享，18 t | `2 2 2 2` / `24³×48` | **110.3546 s** + builder **1741.882 s** | 未记录 | `7.125789e-05` | `12,507,028 KiB = 11.93 GiB` | **VERIFIED 但实际回退到 fine BiCGStab**；不是有效 MG 点；作业号来自会话记录，设计 §0.1.1 仍标 `TODO`（归档 `djob -l` 前按 C 级） |
| c128 BiCGStab | `2153005` / 未分配 | 32 rank，1 节点独占请求 | `2 2 2 4` | — | — | — | — | **PENDING**；**正式端口单节点**行，无任何数值；c128 harness 两节点见下一行 |
| c64 BiCGStab **两节点** | `2155536` / `cn[19544,19546]` | 32 rank，2 节点独占，18 t，`drun -mpi pmix` | `2 2 2 4` / `24³×24` | **1.1778153 s** | 11 | `7.12609211762054e-05`（strict `1e-4`） | `2,382,824 KiB = 2.27 GiB` | **VERIFIED（KUPL harness）**；单样本，不是正式端口矩阵行（§7.5） |
| c128 BiCGStab **两节点** | `2155785` / `cn[12044,12054]` | 32 rank（16/node，18 t），2 节点独占，`drun` | `2 2 2 4` / `24³×24` | **3.9853145 s** | 16 | `7.466169937510113e-06`（strict `1e-4`） | `3,415,116 KiB = 3.26 GiB` | **VERIFIED（KUPL harness）**；单样本（§7.6） |
| c64 BiCGStab **两节点** | `2155830` / `cn[12041,12067]` | 16 rank（8/node，36 t），2 节点独占，`drun` | `2 2 2 2` / `24³×48` | **3.0647245 s** | 11 | `7.125979321648942e-05`（strict `1e-4`） | `4,315,972 KiB = 4.12 GiB` | **VERIFIED（KUPL harness）**；单样本（§7.6） |
| c64 BiCGStab **两节点** | `2155868` / `cn[12041,12067]` | 64 rank（32/node，9 t），2 节点独占，`drun` | `2 2 4 4` / `24×24×12×24` | **2.7971336 s** | 11 | `7.125993463624718e-05`（strict `1e-4`） | `1,386,252 KiB = 1.32 GiB` | **VERIFIED（KUPL harness）**；单样本（§7.6） |
| c64 Legacy-MG **两节点** | `2155888` / `cn[12054,12067]` | 16 rank（8/node，36 t），levels 2 / restart 5，复用 16r cache | `2 2 2 2` / `24³×48` | **2.1298542 s** | `null` | `7.125979321648942e-05`（与 `2155830` 逐位相同 → fine fallback） | `8,487,320 KiB = 8.10 GiB` | **VERIFIED（cache-load 路径）**；粗修正未生效，不是 MG 加速证据（§7.6） |
| 两节点 32 rank | `2153008` / 未分配 | `-nn 2 -N 32 -rpn 16` | `2 2 2 4` | — | — | — | — | **PENDING**；无 `execNodes` |
| 32 rank MG | `2152970` / 未分配 | 32 rank，1 节点独占 | `2 2 2 4` | — | — | — | — | **PENDING（17:44 仍 PENDING）** |
| L48 CG | — | — | — | — | — | — | — | **BLOCKED / 已退役**；无 L48 CG harness |
| dslash/clover 32r | `2152996` 等 | 32 rank，1 节点 | `2 2 2 4` | — | — | — | — | **BLOCKED/UNVERIFIED**；计算节点 MPI init 后挂起 |
| Strict-MG | — | — | — | — | — | — | — | **排除**（本轮范围外） |

### 5.2 三个时钟不能混

| 作业 | solver `wall_seconds` | `mpirun` 段 | 作业 wallclock / elapsed | 差值来源 |
|---|---:|---:|---:|---|
| `2152757`（32r） | **1.6646 s** | 248 s | **251 s** | import/init、prebind 58.9 s、gauge 19.2 s、source 8.4 s、独立残差 6.4 s、solution 写 |
| `2152325`（16r） | **183.0004 s** | 703 s | **705 s** | prebind 70.6 s、gauge 43.8 s、source 10.5 s、独立残差 139.8 s、solution 写 |
| `2152391`（16r MG） | **110.3546 s** | — | **2656 s** | builder 1741.9 s、gauge 276.8 s、残差 161.7 s、prebind 46.8 s、source 28.1 s、未归因约 290 s |

> 规则来自 [`kupl_l24_c64_mg2_shm1_boundary_20261115.md`](kupl_l24_c64_mg2_shm1_boundary_20261115.md)：
> `wall_seconds` 只覆盖一次 solver 调用，`elapsed` 覆盖 MPI 启动 + HDF5 + 独立残差。
> 不能把 `wall` 当端到端，也不能把 builder 加进 `wall`。

### 5.3 为什么 183 s 不是 16-rank 基线

| 证据 | 16r `2152325` | 32r `2152757` | 说明 |
|---|---:|---:|---|
| 节点独占性 | **共享**（`exclusive false`） | **独占**（`exclusive true`） | 不是同条件 |
| 节点 | `cn12038`（在仓库既有排除名单中） | `cn12072` | 节点不同 |
| grid / local | `2 2 2 2` / `24³×48` | `2 2 2 4` / `24³×24` | 每 rank 体积不同 |
| 每 site-iteration | **25.07 µs** | **0.456 µs** | 差 **55×**，rank 数/分解无法解释 |
| 独立残差阶段 | 139.837 s | 6.430 s | 差 21.7×，同一节点内多个阶段都慢 |
| gauge 读 | 43.764 s | 19.214 s | I/O 反而随 rank 数改善 |
| `wall` 比 | — | — | `183.0 / 1.6646 ≈ 110×` 只说明**共享节点下的病态** |

**结论**：`183 s` 只能作为“共享节点 + 排除节点 + 单样本”的**病态记录**，不能写成
16-rank 基线，不能反推“16 rank 太慢”，更不能与 32r 相除得出 scaling。真正的 16r
基线仍缺（需同节点、同 grid、独占、`numastat -p` 证据）。

### 5.4 16r “MG” 不是 MG：必须记录的 P0

`2152391` 的 `final_true_residual` 与 `2152325`（`levels=1`）**逐位相同**
（`7.125789483932747e-05`）。原因：

- `atol=1e-4` × `coarse_tol_factor=1e4` ⇒ `coarse_atol = 1.0`；
- 粗层第一次残差检验就通过（`iterations=0`），粗层解恒 0、粗空间不贡献；
- 当时 `fallback_to_fine=true`、实际走 fine parity-Schur BiCGStab——该回退路径已于
  2026-11-25 随修正器删除（见 `logs/v20261126.txt`）。

诊断作业 `2152601` 用 `KUPL_REQUIRE_MG_CORRECTION=1` 复现并硬失败：
`rejected fine BiCGStab fallback`，`MPI_ABORT`。该门现在只拒绝没有粗空间的配置
（`levels<=1` / `_MG_USE_DEFLATE_==0`），因为 `accepted_corrections` 改计预条件应用
次数，它不再能区分"粗层是否真的贡献"。所以：

| 说法 | 判定 |
|---|---|
| “16r MG solve = 110.35 s” | **错误**；它是 fine BiCGStab 的调用时间，不是 MG 粗修正 |
| “builder 1741.88 s 买到了 MG 加速” | **错误**；粗修正没有生效 |
| “L48 已有有效 MG 性能点” | **UNVERIFIED / 目前不存在** |
| “Legacy-MG 的 cache/export 路径可跑” | **VERIFIED**；cache 有效（`patch_galerkin_error 3.59e-07`、`verify_status=0`） |

> 有效 MG 的复跑条件（未执行）：`atol=1e-6`（`coarse_atol=1e-2`）+ 复用现有 cache，
> 并与 `levels=1` 的独立真残差/迭代数对照。2026-11-25 起外层是 FGMRES(m=10)，
> `KUPL_REQUIRE_MG_CORRECTION=1` 不再能证明粗层真的贡献。**删除前那套按
> `restart`/修正间隔复跑的做法已不存在**：历史数字在 `logs/v20261123.txt` §153/§182，
> 无法从当前源码复现。coarse tolerance 机制见
> [`kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md) §9.5。

### 5.5 32r 数值的可信度与不可声称项

| 项目 | 结论 | 状态 |
|---|---|---|
| 32r 单节点 c64 正确性 | `grid 2 2 2 4`、`local 24³×24`、11 it、`7.1261e-5`、strict `1e-4` PASS、rc 0、job exit 0 | **VERIFIED**（单样本） |
| HBM 驻留 | **未证明**；没有 `KUPL_TOPOLOGY_REPORT`/`numastat` 页证据 | **UNVERIFIED** |
| 32r vs 16r 的 110× | **不可作为 scaling**；节点、独占性、grid 全不同 | **UNVERIFIED（因果）** |
| 32r 计时是否合理 | 与 L24 32r×18t `1.5979 s`/16 it（`0.0999 s/it`）相比，L48 `0.1513 s/it` 在合理范围 | **VERIFIED（量级交叉核对）** |
| 内存口径 | 用 in-process `max_rss_kib`；Donau `totalMaxMem`（16r 50,002 MB / 32r 4,249 MB）与进程 RSS 矛盾，**不可用** | **VERIFIED（矛盾）** |
| 重复性 | 只有 1 个样本；无 repeat、无 16r 在 `cn12072` 的对照 | **UNVERIFIED** |

## 6. Cache 策略（对应经验 6）

### 6.1 只认同 identity

复用条件（与 [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) §0.1.2 一致）：

```text
lattice, dtype, mass, csw, mg_grid, nv_iters, dof, seed,
gauge_kind / gauge_canonical_sha256, ranks, grid, verify_status=0
```

`atol`、`coarse_tol_factor`、`coarse_max_iter`、`restart`、`deflate` **不在** identity
内（它们不改变 cache 内容）。**禁止**按文件名、路径或文件大小复用；`*_c64.h5` 实为
complex128 已经证明文件名不可信。

### 6.2 当前唯一可用 cache

| 项目 | 值 | 状态 |
|---|---|---|
| 路径 | `PyQCU-main/data/l48_validation_20261117/caches/l48_16r_mg_l2r5` | **VERIFIED**（远端） |
| 规模 | `1,369,800,704 B = 1.276 GiB`（16 × 85,612,544 B），E=2 | **VERIFIED** |
| 校验 | `verify_status=0`，`patch_galerkin_error 3.59e-07`，`status=PASS` | **VERIFIED** |
| 复用 | 诊断作业 `2152601` 复用了该 cache；作业因 `KUPL_REQUIRE_MG_CORRECTION=1` 硬失败（设计如此） | **VERIFIED**（机制） |
| 适用行 | 只能给**同 identity 的 16 rank** 行复用 | **VERIFIED** |

> **注意**：17.44 GiB 是 `docs/qcu_quda_solver_dcu_l48_20261029.md` 的 E=12 历史预算，
> 不是本轮 cache。本轮 E=2 cache 是 **1.276 GiB**。不要把两者混写。

### 6.3 32r / 两节点：DEFERRED，不重建

| 事实 | 状态 |
|---|---|
| 32r 的 grid 是 `2 2 2 4`，16r cache 的 identity 是 `2 2 2 2` → **不匹配** | **VERIFIED** |
| 没有 32r / 两节点 L48 cache 实体 | **VERIFIED**（无产物） |
| 32r MG 作业 `2152970` 带 `--write-coarse-cache`，但 17:44 仍 PENDING | **PENDING** |
| 因此 32r/两节点 MG 暂缓，**不重建**；重建需另立批次并记录 identity、`builder_seconds`、`verify_status` | **DEFERRED** |
| 不要把 16r cache 改名/改路径给 32r 用 | **禁止** |

### 6.4 break-even ≈ 24 次 solve（有前提）

`1741.88 / (183.00 − 110.35) = 1741.88 / 72.65 ≈ 23.97`，约 **24 次 solve**；
若把该 MG 作业多付的 233 s gauge read 也算进去，约 **27 次**。

| 说法 | 判定 |
|---|---|
| “缓存构建在 ~24 次 solve 后回本” | **VERIFIED（仅对 16r 这对数字）** |
| “MG 本身比 fine solver 快 72.65 s” | **错误/UNVERIFIED**；110.35 s 是 fallback 的 fine BiCGStab，不是 MG |
| “32r 也可以按 24 次算” | **UNVERIFIED**；32r 无 cache、无 builder 数据 |
| “每次 run 重建 cache” | **禁止**；只同 identity 复用 |

## 7. 调度器与两节点 launcher（对应经验 7）

### 7.1 `-nn 2 -N 32 -rpn 16` 是正确的

`djob -ll 2153008` 解析结果与 header 一致：

| 字段 | 值 | 判定 |
|---|---|---|
| `nnodes` | 2 | 正确 |
| `replica` / `minReplica` | 32 / 32 | 正确 |
| `ntasksPerNode` | 16 | 正确（16 × 2 = 32） |
| `reqCPU` / `reqMem` | 18 / 8192 MB | 正确（每 rank 18 CPU、8 GiB） |
| `exclusive` | true | 正确 |
| `timeout` | 600 | 正确 |

每节点 16 × 18 = **288 CPU**（608 的 47%）、128 GiB；不是“32 × 18 = 576 per node”。
`-rpn` 是**每节点最大副本数**，16 正好强制 16/16 拆分；`-rpn 32` 会全落到一个节点。

### 7.2 为什么 PENDING

| 因素 | 证据 | 状态 |
|---|---|---|
| 队列/节点不足 | `q_hpcapp` OPEN/ACTIVE，130–146 running / 183–191 pending；只有 7–8 个 `OK` 节点，且没有 608 空闲 CPU 的节点 | **VERIFIED**（时点快照） |
| 独占 `-x job` | 需要两个**完全空**的 608-CPU 节点；`CPU_FREE` 最高 496/608 | **VERIFIED** |
| 账号配额 | `testusers_rebuild` 显示 used `608` / limit `1216` = 2 × 608；2 节点独占作业会吃掉整个配额，必须等本账号其他作业结束 | **UNVERIFIED（推断；`dlimit` permission denied，规则未直接读到）** |
| 结果 | `2153008` 在 16:11:54 → 17:06:47 的 55 分钟观察窗内一直 PENDING；最后记录（17:06:47）仍无 `execNodes`；17:44 只单独复查了 `2152970`（仍 PENDING），未再复查 `2153008` | **PENDING（截至最后记录）** |

**判读**：请求语法正确、被受理；PENDING 是资源/配额/队列问题，不是 malformed
request。PENDING 不携带任何数值信息。

### 7.3 `2152277` 的状态更正

| 事实 | 值 | 状态 |
|---|---|---|
| `2152277` 最终状态 | **SUCCEEDED**，15:48:25 → 15:48:37（12 s），`cn12254`/`cn19475` | **VERIFIED** |
| 请求 | `-nn 2 -N 2 -rpn 1 -R "cpu=1,mem=1G" -x job` | **VERIFIED** |
| 它证明了什么 | 2 节点独占**可以被调度**（等待约 1 h 57 m）；脚本只打印 `CCS_ALLOC_FILE`，**没有 `mpirun`** | **VERIFIED** |
| 它不能证明什么 | 不是两节点 MPI 结果，不是 L48 结果 | **VERIFIED** |

> `docs/pyqcu_test_l48_remote.md` §2.4.5/§0 仍写“`2152277` PENDING”，那是 13:51–
> 15:48 之前的快照；**以 15:48 的 `djob -ll` 为准**。写文档时要带时间戳，状态会变。

### 7.4 两节点 launcher：为什么必失败、怎么修

> 2026-11-17 22:15 更新：SSH 私钥已生成且被计算节点接受，但计算节点在认证后关闭
> 会话，因此 `plm_rsh` 仍不可用；已验证的替代路径是 `drun`，见 §7.5。本节保留
> 失败诊断，不再代表当前唯一结论。

已观测失败：

| 作业 | 形态 | 结果 |
|---|---|---|
| `2152326` | 2 节点共享，`mpirun -np 2 hostname` | **RC=255**：`Host key verification failed` + `ORTE was unable to reliably start one or more daemons` |
| `2152334` | 同上 + tolerant rsh agent + `--host` | **RC=255**；`ssh cn12045` → `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)` |
| `2153008` | 提交脚本 | 未运行；脚本写的 rankfile **从未传给 `mpirun`** |

根因：

1. 默认 `plm_rsh_agent = "ssh : rsh"`，本账号**没有 SSH 私钥**，`known_hosts` 只有
   5 条；跨节点 daemon 起不来。
2. 脚本只从 `CCS_ALLOC_FILE` 取两个 hostname，写了一个 16 slot/node 的 rankfile，
   但 `mpirun` 行没有 `--rankfile`/`-rf`、没有
   `--mca rmaps_rank_file_physical true`、没有 `$CCS_MPI_OPTIONS`、没有
   `CCS_DRUN_RANK_FILE`、没有 `-x` 转发。
3. `--host N1,N2 --map-by ppr:16:node` 只表达 mapping，不保证 launcher；它会继续走
   broken 的 ssh 路径。`--host` 不带 slot 数（`N1:16,N2:16`）更脆弱。

修复配方（厂商模板形状，**未在 L48 两节点上实跑**）：

```bash
# rankfile: 16 rows/node, width = total_cores / rank_count = 288 / 16 = 18 (NOT 36)
# rank N=<host> slot=<start>-<end>
export CCS_DRUN_RANK_FILE="$PWD/${CCS_JOB_ID}-rankfile"   # selects plm_donau
mpirun $CCS_MPI_OPTIONS -np $((CCS_TASK_REPLICA)) \
  -x PATH -x LD_LIBRARY_PATH \
  -rf "$CCS_DRUN_RANK_FILE" \
  --mca rmaps_rank_file_physical true \
  --mca plm_rsh_agent /usr/bin/ssh \
  --bind-to none --allow-run-as-root \
  python3 -B pyqcu/testing/simple/kupl_multigrid.py ...
```

| 修复点 | 作用 | 状态 |
|---|---|---|
| `-rf/--rankfile` | 用分配文件显式给出 rank→host/slot | **VERIFIED（厂商模板）** |
| `--mca rmaps_rank_file_physical true` | 让 binding 使用 physical slot，不越出 cgroup cpuset | **VERIFIED（手册/厂商模板）** |
| `CCS_DRUN_RANK_FILE` | 选 `plm_donau`（drun）而不是 ssh | **VERIFIED（厂商模板）** |
| `$CCS_MPI_OPTIONS` | 站点注入的 launcher 选项；厂商脚本会前置 | **UNVERIFIED（内容不可读）** |
| `-x PATH -x LD_LIBRARY_PATH` | 把 HPCKit 环境带到远端 rank | **VERIFIED（厂商模板）** |
| `--mca plm_rsh_agent /usr/bin/ssh` | 私钥已生成、host key 可接受，但计算节点认证后 `Connection closed`；不是可用 shell 路径 | **BLOCKED（compute shell）** |
| `cores_per_rank = total_cores / rank_count = 18` | 16 rank × 18 = 288，不等于 32×18=576 | **VERIFIED（公式）** |
| `plm_donau` 是否可加载 | 组件存在但 `query()` 拒绝，不能强制选中；`--mca plm donau` 不得恢复 | **VERIFIED（不可用）** |

> **结论**：即使 `2153008` 分到两个节点，按当前脚本也会在 `mpirun` 阶段因 ssh 失败。
> 先修 launcher，再谈两节点性能；不要把“分到节点”当“跨节点 MPI 成功”。

### 7.5 已验证的两节点 `drun` 路径（2026-11-17 22:15）

> 本节取代 §7.4 中“未在 L48 两节点上实跑”的旧状态；§7.4 保留为失败诊断记录。

**结果**：job `2155536`（`l48_2n32r_l1r0_drun`）在 `cn[19544,19546]` 上
`SUCCEEDED`，报告 `status=PASS`：

| 字段 | 值 |
|---|---|
| `mpi_size` / `grid` | `32` / `2×2×2×4` |
| `local_lattice` | `24×24×24×24` |
| dtype / solver | `complex64` / `applyCloverMultigridQcu`（levels=1, restart=0） |
| `iterations` | `11` |
| `final_true_residual` | `7.12609211762054e-05`（strict `1e-4`） |
| `wall_seconds` | `1.1778152959886938 s` |
| `input_prebind_seconds` | `17.433025036007166 s` |
| `max_rss_kib` | `2382824`（约 `2.27 GiB`） |

**启动形状**（远端 `sh/l48_job_2n32r_l1r0_drun.sh`；本地不执行）：

```bash
export CCS_DRUN_RANK_FILE=$OUT/logs/$JOBTAG-rankfile
/opt/batch/agent/tools/drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v -- \
  python3 -B pyqcu/testing/simple/kupl_multigrid.py --lat 48 48 48 96 \
  --grid 2 2 2 4 --mass -0.277 --csw 1.160920226 --dtype c64 \
  --levels 1 --restart 0 --max-iter 50 --atol 1e-4 --strict-acceptance ...
```

关键经验：

1. **rankfile 胜过 `-nn/-rpn/-R`（本轮最硬的一条）**：只要设置了
   `CCS_DRUN_RANK_FILE`，`drun` 就警告 `-N, -nn, -rpn, -nl, -R, --cpu-bind,
   --distribution are ignored when rankfile environment variable
   CCS_DRUN_RANK_FILE is specified`，并以 rankfile 为准。四个变体每次都有该告警；
   实际 ranks/node 由 rankfile 行数决定（8、16、32 都跑过），core/rank 由 rankfile
   宽度决定（9、18、36 均被接受，生成器 `cpr ≤ 37`）。**不要**用 `-nn/-rpn/-R`
   推断 rank 数或绑定；必须用报告 `mpi_size`/`grid`/`local_lattice` 复核，并保存
   rankfile 与 `RANKFILE_OK rows=… nodes=2 ranks_per_node=… width=…` 校验行。
2. **SSH key 已不再缺失，但 SSH 仍不是可用路径**：22:11 生成
   `~/.ssh/id_ed25519_pyqcu` 并追加公钥后，`ssh -vvv` 显示
   `Server accepts key`，随后服务端 `Connection closed`；这是计算节点不允许直接
   shell 会话，不是 `known_hosts`/私钥问题。不要恢复 `plm_rsh` 或
   `--mca plm donau`。
3. **`drun` 只在计算节点存在**（登录节点 `command -v drun` 为空）；必须在作业
   容器内启动。`plm_donau` 在该环境仍 `query()` 拒绝。
4. **边界**：这些是 KUPL harness 的单样本两节点结果；不是
   `pyqcu.test --l48-two-node` 正式矩阵行（该 flag 在远端覆盖树不存在），不覆盖
   正式 c128、两节点 MG 加速、dslash/clover 或 scaling。正式端口仍为 `UNVERIFIED`。

### 7.6 两节点变体矩阵（2026-11-17 23:30，KUPL harness）

PASS 规则未放宽：报告 `status == "PASS"` **且** `final_true_residual < 1e-4`
（独立分布式 NumPy Wilson-Clover 重构）。作业串行提交，每次 `dsub -s` 前
`djob -u nsls_yangyb` 为空；提交形状
`dsub -s data/l48_variants_20261118/<wrapper>.sh`。报告根
`/home/share/nsls_yangyb/PyQCU-main/data/l48_variants_20261118/`。

| # | 变体 | 作业 | execNodes | dtype | ranks / 节点（core/rank） | grid / local | it | `final_true_residual` | solver wall (s) | max RSS (KiB → GiB) | 报告 |
|---:|---|---|---|---|---|---|---:|---|---:|---|---|
| 1 | c128 BiCGStab 32r | `2155785` | `cn[12044,12054]` | `complex128` | 32 / 2（18） | `2 2 2 4` / `24×24×24×24` | 16 | `7.466169937510113e-06` | `3.9853145` | `3415116` → 3.26 | `reports/v1_c128_2n32r_l1r0.json` |
| 2 | c64 BiCGStab 16r | `2155830` | `cn[12041,12067]` | `complex64` | 16 / 2（36） | `2 2 2 2` / `24×24×24×48` | 11 | `7.125979321648942e-05` | `3.0647245` | `4315972` → 4.12 | `reports/v2_c64_2n16r_l1r0.json` |
| 3 | c64 BiCGStab 64r | `2155868` | `cn[12041,12067]` | `complex64` | 64 / 2（9） | `2 2 4 4` / `24×24×12×24` | 11 | `7.125993463624718e-05` | `2.7971336` | `1386252` → 1.32 | `reports/v3_c64_2n64r_l1r0.json` |
| 4 | c64 Legacy-MG 16r cache-load | `2155888` | `cn[12054,12067]` | `complex64` | 16 / 2（36） | `2 2 2 2` / `24×24×24×48` | `null` | `7.125979321648942e-05` | `2.1298542` | `8487320` → 8.10 | `reports/v4_c64_2n16r_legacy_mg_l2r5.json` |

- **MG 读法**：变体 4 的残差与变体 2（levels=1）**逐位相同**，且
  `atol=1e-4 × coarse_tol_factor=1e4 ⇒ coarse_atol=1.0`：这是 fine BiCGStab 回退
  签名。它只证明**分布式 cache 加载 + solve 路径**（identity 校验、
  `verify_status=0`、`builder_seconds=21.420243138913065 s` 为加载/校验而非
  1741.88 s 重建），**不证明 MG 加速**，也不是有效 MG 性能点。
- **32r Legacy-MG**（grid `2 2 2 4`）无同 identity cache → **DEFERRED /
  CACHE-MISSING**；未重建、未改名。
- **formal flag 缺失**：远端覆盖树 `python3 -m pyqcu.test --list --l48-two-node`
  → `unrecognized arguments: --l48-two-node`、RC=2；`pyqcu/test/l48.py` 不存在，
  `__main__.py`/`common.py` 的 `l48` 计数为 0。正式矩阵五行仍 **UNVERIFIED**。
- **不是 scaling**：每个变体 1 样本，节点对、ranks/node、core/rank 全不同；不计算
  加速比，也不把 16/32/64 rank 当扩展性曲线。
- **诊断作业** `2155899`（变体 4 + `KUPL_REQUIRE_MG_CORRECTION=1`）归档时仍
  `PENDING`，无 `execNodes`、无数值，不作为结果。

### 7.7 正式 `pyqcu.test` native MG2 16r/2n PASS（2026-11-19）

job `2175522`（`cn[12118,12139]`）通过 formal `pyqcu.test` native row
（`kupl/native/mg`），不是 harness、也不是 L48 矩阵 `legacy-mg`：

- 16 ranks / 2 nodes / 8 per node / 36 cores；grid `2×2×2×2`，local
  `24×24×24×48`，c64，mass `-0.277`，csw `1.160920226`，levels=2/restart=5；
- ranked cache `l48_16r_mg_l2r5`，`cache_reused=true`，`asset_source=cache`；
  `mpi_size=16`、`mpi_rank=0`、`cache_rank=0`；
- `relative_residual=7.941582482721049e-06`、`residual_verified=true`、
  `converged=true`、`status=PASS`，solver `2.9444 s`，job `181.59 s`。

调试链：

1. 连续 slot rankfile 跨过 NUMA 保留核 37 → `drun` 拒绝；改用站点
   `0-35,38-73,...,266-301`。
2. `drun -mpi pmix` 的 rank0 环境与其它 rank 不同；只把 CCS 当 fallback，
   rank 优先 PMIX/OMPI，size 可用 `CCS_TASK_REPLICA`。
3. formal native 只认单文件 cache；新增 ranked cache manifest/rank slab 读取，
   并把 `c64` 与 `complex64` 归一化。
4. cache observations 的 `asset_source` 重复；保留显式 `asset_source=cache`。
5. 修正后 rank0 单独写 report，`mpi_rank=0/cache_rank=0`；早先 16 份输出是
   `CCS_TASK_INDEX=0` 全 rank 同值造成的 owner 误判。

边界：16r/2n、单样本、cache-load；不声明 32r/两节点 MG、L48 矩阵行或 scaling
通过。`iterations=0` 是 Legacy-MG 的正常观测，不是失败。

### 7.8 正式 native MG2 32r/64r PASS：同 mg_grid cache 复用（2026-11-19）

两条正式 `pyqcu.test` native row 都复用 **16r cache**，没有为 64r 重建 cache：

- **32r** job `2176880`（`cn[12068,12071]`），grid `2 2 2 4`，local
  `24×24×24×24`，cache local coarse `[12,12,12,6]`；
  `relative_residual=7.942248513387823e-06`，PASS，case `342.07 s`。
- **64r** job `2177086`（`cn[12071,12072]`），grid `2 2 4 4`，local
  `24×24×12×24`，cache local coarse `[12,12,6,6]`；
  `relative_residual=7.941747295216055e-06`，PASS，case `396.17 s`。
- 两条都 `cache_reused=true`、`asset_source=cache`，`cache_notes` 记录
  `ranked Legacy-MG cache repartitioned to live process grid`；levels=2/restart=5。
- 关键：同 identity + 同 `mg_grid` 时复用的是**全局 coarse 张量**；ranked
  cache 的 `process_grid/local_coarse` 可以不同，但必须按 live grid 重新计算
  coarse 窗口并读取覆盖 slab。直接拿 16r 的 rank_00000 当 32r rank_00000 会错。
- 32r cache `l48_32r_mg_l2r5` 本身也 PASS（job `2176278`），但本批次没有用于
  最终 32r/64r 数值；两者都复用 16r cache。

边界：这是单样本 `repeat=1, warmup=1` 的正式 native 行，不是稳定性能曲线；
不是 L48 矩阵 `legacy-mg` 行；也不证明 scaling（32r 和 64r 的启动/读输入成本
不同）。`iterations=0` 仍是 Legacy-MG 正常观测。

## 8. 发布与过程纪律（对应经验 8）

### 8.1 不允许假 PASS

| 场景 | 错误写法 | 正确写法 |
|---|---|---|
| `2153008` PENDING | “两节点已跑通” | “请求已受理，未分配；无 `execNodes`，两节点未验证” |
| `2152277` 12 s 脚本成功 | “两节点 MPI 成功” | “2 节点独占可调度；脚本只打印分配，无 `mpirun`” |
| 16r `2152391` 标 MG | “MG 110.35 s” | “实际回退 fine BiCGStab；不是有效 MG 点” |
| harness 两节点四个 PASS | “正式端口两节点已跑通” | “KUPL harness 四变体 PASS（§7.6）；正式矩阵仍 `UNVERIFIED`，MG 只是 cache-load 路径” |
| 登录节点 32r smoke ~2 s | “32r dslash 可用” | “计算节点 32r 挂起；登录 smoke 不算” |
| `--check` 18/18 或 19/19/20/20 | 混成一个分母 | 写清环境与时间：远端 18/18，本机 19/19（17:47）→ 20/20（17:55） |
| c64 文件名 | “c64 输入” | `dataset.dtype` + `io.describe` 回读才算 |

### 8.2 Strict-MG 排除

- 本轮**没有跑 Strict-MG**；它不是 L48 的备选，也不在范围。
- `--strict-acceptance` 只是**容差门**（把 PASS 门从 dtype floor 改成 `atol`），
  与 Strict-MG 算法无关；不要把两者混写。
- Legacy-MG 也只跑到 cache/export 路径；粗修正因 tolerance 回退，见 §5.4。

### 8.3 版本与标签

| 项目 | 值 | 状态 |
|---|---|---|
| 已发布版本 | `0.1.17` | **VERIFIED** |
| 已发布标签 | `test37`（peeled `e0b25a75626c8bbf94a7d219606264dd0ce66f76`） | **VERIFIED** |
| PyPI 0.1.17 sdist | `912,755 B`，sha256 `82d1af5f8f0ac8b2c5f38b4a3c9d5cff32fd114cb6ebaf196b9e631b29b19dca` | **VERIFIED** |
| 下一版本 | `0.1.18` | **计划** |
| 下一标签 | `test38`（`follow test37`） | **计划** |
| 发布门 | **十轮全部 PASS 之前不得发布**（R1–R10） | **VERIFIED**（评审清单） |
| 本轮状态 | R1/R2/R6/R7/R8/R9 FAIL，R3/R4/R5 部分 PASS，R10 UNVERIFIED | **VERIFIED-C**（评审快照） |
| 本轮动作 | 不改版本、不 commit、不 push、不打 tag、不上 PyPI | **VERIFIED** |

### 8.4 `bug45` 是历史例外，不重写

`bash skills/pyqcu-tag/tag-chain.sh --check` 的唯一历史 `[BLOCK]`：

```text
[BLOCK] bug45 的 peeled commit 不是 dev106 的祖先，且没有可验证的共同 follow 祖先
FAIL: 发现 1 个问题；可安全修正候选 0 个。
```

- 该缺口自 2026-10-31 起存在，不是本轮引入。
- **禁止**：force-push、移动/改写 `bug45`/`dev106`、删标签、把“0 个可修正候选”
  说成“已修复”。
- 正确写法：“tag gate 有 1 条已知历史阻塞（`bug45`），由发布负责人书面接受”。

### 8.5 不得覆盖

- `0.1.17` / `test37` 已发布，PyPI 拒绝同版本重传；只能发 `0.1.18` / `test38`。
- 旧驱动的 **100 列 CSV / 21 sheet Excel** 契约是硬性 parity 目标；未逐列/逐 sheet
  通过前，不得声称已替换旧驱动。
- 既有 `docs/**`、`logs/**`、`skills/**` 由其他 agent 维护的文件不覆盖；新增文件只
  链接，不改写。

## 9. Agent 工作流与验证协议（对应经验 9）

### 9.1 只读审计优先

```bash
# 调度器只读
djob -l <job>; djob -ll <job>; djob -u <user>; dnode -q q_hpcapp -s OK

# 远端身份/环境只读
hostname; whoami; python -V; uname -m
python3 -c 'import h5py; print(h5py.__version__, h5py.version.hdf5_version, h5py.get_config().mpi)'
md5sum pyqcu/testing/simple/kupl_multigrid.py
sha256sum <input.h5>

# 证明 repo 未被写
find "$REPO" -maxdepth 2 -newermt "<start>" -type f
```

原则：

- 不在远端 repo 写文件；作业脚本/报告/解放在 `data/<batch>/**` 或 repo 外的
  `/home/share/<user>/<batch>/**`。
- Python 用 `-B`，避免 `__pycache__` 写入。
- 不 `dkill`/`drelease` 别的作业；自己的作业也尽量保留 PENDING，除非明确要求。
- 只读审计也要记录快照时间：调度器状态会变（`2152277` 从 PENDING 变 SUCCEEDED）。

### 9.2 监视器要有界

| 模式 | 做法 | 证据 |
|---|---|---|
| 轮询 | 20 s 一次，硬超时（如 840 s、14 min、55 min）后停止 | `poll_2152757.log`、`poll_2152970.log` |
| 超时输出 | `WATCH_TIMEOUT after 844s state=PENDING ...` | 明确写“未在窗口内完成”，不推断 |
| 终态 | 只认 `SUCCEEDED`/`FAILED`/`CANCELLED` + `jobExitCode` + `execNodes` | `djob -l/-ll` |
| 禁止 | `sleep` 无限循环、反复重提交、把 PENDING 当失败/成功 | — |

### 9.3 状态词表

- `VERIFIED`：有 job id + 报告/日志/命令输出。
- `VERIFIED-C`：有远端记录但原始 stdout 未入库。
- `PENDING`：已提交、未分配；没有数值。
- `BLOCKED`：缺 harness/launcher/凭据；写解除条件。
- `DEFERRED`：本轮不做；写重启条件。
- `UNVERIFIED`：有推测/间接证据；不得写成结论。
- `TODO`：尚未执行或未归档。

### 9.4 证据模板

每个 L48 row 至少记录：

```text
job_id, submit_command, exec_nodes, ranks, ranks_per_node, grid, threads, shm
gauge_path, gauge_bytes, gauge_sha256, gauge_dtype, gauge_format
source_path, source_bytes, source_sha256, source_seed
mass, csw, kappa, atol, max_iter, solver, mg_levels, mg_restart
wall_seconds, elapsed_seconds, iterations, final_true_residual, residual_source
status, reason, report_path, solution_path
```

缺少 `exec_nodes`、`sha256` 或独立残差的数字，只能标 `UNVERIFIED`。

### 9.5 文件与批次边界

- 只写自己拥有的新文件；不改其他 agent 正在编辑的文件。
- 新文档不复制设计正文，只链接设计/评审/风险登记/操作者参考。
- 发现旧文档状态过期时，在自己的新文件里写“更正 + 时间戳 + 证据”，不直接改
  对方文件；由对方批次合并。
- 发布/提交/打 tag 一律不在“记录经验”批次执行。

## 10. 证据索引

### 10.1 仓库内文档与代码

| 用途 | 路径 |
|---|---|
| L48 设计、权威状态表、远端基线 | [`pyqcu_test_l48_remote.md`](pyqcu_test_l48_remote.md) |
| 十轮发布终审清单 | [`pyqcu_test_l48_review.md`](pyqcu_test_l48_review.md) |
| P0/P1 风险登记（全部未修复） | [`pyqcu_test_audit_risk_20261117.md`](pyqcu_test_audit_risk_20261117.md) |
| 操作者最短检查单 | [`skills/pyqcu-testing/references/l48-remote.md`](../skills/pyqcu-testing/references/l48-remote.md) |
| 本次经验速查（技能侧） | [`skills/pyqcu-testing/references/remote-l48-experience.md`](../skills/pyqcu-testing/references/remote-l48-experience.md) |
| dtype/文件名误标根因 | [`kupl_hdf5_source_gauge_rootcause_20261115.md`](kupl_hdf5_source_gauge_rootcause_20261115.md) |
| L48 容量/cache 历史预算 | [`qcu_quda_solver_dcu_l48_20261029.md`](qcu_quda_solver_dcu_l48_20261029.md) |
| `wall` vs `elapsed` 规则 | [`kupl_l24_c64_mg2_shm1_boundary_20261115.md`](kupl_l24_c64_mg2_shm1_boundary_20261115.md) |
| 鲲鹏 HPC 环境/调度 | [`kupl_kunpeng_hpc_config_20261112.md`](kupl_kunpeng_hpc_config_20261112.md) |
| 性能可比性红线 | [`kupl_performance_20261115.md`](kupl_performance_20261115.md) |
| coarse tolerance 回退机制 | [`kupl_mpi_scaling_20261114.md`](kupl_mpi_scaling_20261114.md) |
| 旧 KUPL 全矩阵驱动 | [`pyqcu/testing/kupl/test_kunpeng_all.py`](../pyqcu/testing/kupl/test_kunpeng_all.py) |
| 旧 dsub 提交脚本 | [`pyqcu/testing/kupl/submit_kunpeng_all.sh`](../pyqcu/testing/kupl/submit_kunpeng_all.sh) |
| L48 驱动 | [`pyqcu/testing/simple/kupl_multigrid.py`](../pyqcu/testing/simple/kupl_multigrid.py) |
| per-rank slab 生成器 | [`pyqcu/testing/simple/prepare_mpi_slab.py`](../pyqcu/testing/simple/prepare_mpi_slab.py) |
| LIME 读取 | [`pyqcu/testing/simple/_lime.py`](../pyqcu/testing/simple/_lime.py) |
| LIME → HDF5 | [`pyqcu/testing/simple/prepare_case.py`](../pyqcu/testing/simple/prepare_case.py) |
| 正式端口 IO/dtype 闸门 | [`pyqcu/test/io.py`](../pyqcu/test/io.py) |
| 3.10 语法闸门 | [`pyqcu/test/selfcheck.py`](../pyqcu/test/selfcheck.py) |
| 3.10 回归测试 | [`pyqcu/testing/python/test_pyqcu_test_syntax_floor.py`](../pyqcu/testing/python/test_pyqcu_test_syntax_floor.py) |
| 本轮任务日志 | [`logs/v202611171745.txt`](../logs/v202611171745.txt) |
| 两节点变体任务日志 | [`logs/v202611172330.txt`](../logs/v202611172330.txt) |
| 上一轮任务日志 | [`logs/v202611171416.txt`](../logs/v202611171416.txt) |

### 10.2 远端证据（路径为远端 repo/共享目录；原始 stdout 未入库时标 C）

| 用例 | 作业 | 报告/日志路径 | 等级 |
|---|---:|---|---|
| 32r c64 BiCGStab | `2152757` | `/home/share/nsls_yangyb/l48_rank32_20261117/reports/l48_32r_l1r0.json` | C（report 在远端，未入库） |
| 32r c64 脚本 | `2152757` | `/home/share/nsls_yangyb/l48_rank32_20261117/l48_job_32r_l1r0.sh` | C |
| 16r c64 BiCGStab | `2152325` | `PyQCU-main/data/l48_validation_20261117/reports/l48_16r_l1r0.json` | C |
| 16r c64 “MG”/fallback | `2152391` | `PyQCU-main/data/l48_validation_20261117/reports/l48_16r_l2r5.json` | C |
| 16r MG cache | `2152391` | `PyQCU-main/data/l48_validation_20261117/caches/l48_16r_mg_l2r5.json` | C |
| c64 cast 校验 | — | `PyQCU-main/data/l48_validation_20261117/logs/verify_c64.log` | C |
| c128 source widening | — | `/home/share/nsls_yangyb/l48_c128_32r_20261117/logs/cast_source_c128.json` | C |
| 两节点作业（未运行） | `2153008` | `PyQCU-main/data/l48_2node_20261117/logs/djob_2153008_snapshot.txt`；无 `.out`/`.err` | PENDING |
| 两节点 c128 32r（harness） | `2155785` | `PyQCU-main/data/l48_variants_20261118/reports/v1_c128_2n32r_l1r0.json` | C |
| 两节点 c64 16r（harness） | `2155830` | `PyQCU-main/data/l48_variants_20261118/reports/v2_c64_2n16r_l1r0.json` | C |
| 两节点 c64 64r（harness） | `2155868` | `PyQCU-main/data/l48_variants_20261118/reports/v3_c64_2n64r_l1r0.json` | C |
| 两节点 c64 MG 16r cache-load（harness） | `2155888` | `PyQCU-main/data/l48_variants_20261118/reports/v4_c64_2n16r_legacy_mg_l2r5.json` | C（cache-load） |
| 正式 flag 探针 | — | `PyQCU-main/data/l48_variants_20261118/logs/v5_port_probe.txt`（RC=2，flag 不存在） | VERIFIED-C |
| MG 回退诊断（未运行完） | `2155899` | 无 report；归档时 PENDING | PENDING |
| 两节点探测（已运行） | `2152277` | `djob -ll 2152277`；`sh/probe_two_node.out` | VERIFIED-C |
| 两节点 ssh 失败 | `2152326` / `2152334` | `PyQCU-main/data/l48_validation_20261117/logs/probe2_shared.err`、`probe2_rsh.out` | VERIFIED-C |
| 32r dslash 挂起 | `2152996` 等 | `PyQCU-main/data/l48_dslash_32r/logs/diag2_np32.err` | UNVERIFIED（草案，诊断未完成） |
| 32r MG | `2152970` | 无 report；17:44 仍 PENDING | PENDING |
| c128 BiCGStab | `2153005` | 无 report；16:11–16:26 一直 PENDING | PENDING |

> **本地会话草稿**（`perf32.md`、`perf_mg.md`、`perf_hdf5.md`、`perf_input.md`、
> `perf_solver.md`、`l48_2node.md`、`two_node_launcher.md` 等）是本轮工作副本，
> 不是长期证据；本文只把其中可复核的数值/路径固化到仓库，原文件不随仓库保存。

## 11. 仍未验证 / 阻塞清单（截至 2026-11-17 23:30）

| # | 项目 | 状态 | 解除条件 |
|---:|---|---|---|
| 1 | L48 c128 **正式端口** BiCGStab | **PENDING**（`2153005`）；harness 两节点 32r 已 PASS（`2155785`，§7.6） | 正式行分配到节点并产出 report |
| 2 | L48 **正式端口**两节点 | **PENDING**（`2153008`）；远端覆盖树无 `--l48-two-node`；harness 已有 16/32/64 rank + c128 + MG cache-load PASS（§7.5、§7.6） | flag 落地 + `djob -ll` 恰好两个 `execNodes` + 实际 `mpirun` 结果 |
| 3 | 正式端口跨节点 launcher/rankfile | **BLOCKED** | `CCS_ALLOC_FILE` → rankfile 接线，传 `--rankfile`/`$CCS_MPI_OPTIONS`；**harness** 侧的 `CCS_DRUN_RANK_FILE` + `drun` 已验证（§7.5），但未接入正式端口 |
| 4 | 32r/两节点 MG cache | **DEFERRED** | 另立批次重建并记录 identity/builder/verify |
| 5 | 32r dslash/clover | **UNVERIFIED** | 定位计算节点 MPI init 挂起 |
| 6 | Legacy-MG 分项/作业号归档 | **TODO** | 归档 `report.json`、每 rank 日志、`djob -l 2152391` |
| 7 | CSV 100 列 / Excel 21 sheet parity | **UNVERIFIED/未通过** | 逐列同序 + 逐 sheet 一致；CLI 接线 |
| 8 | 远端原始 stdout | **TODO** | 归档 `--check`、版本探针、`io.describe`、`prepare_case.py` 41 s、`djob -l` |
| 9 | Strict-MG | **DEFERRED** | 单独立项 |
| 10 | UCG A/B（`--mca coll ^ucg`） | **TODO** | 同节点/几何/seed 3 样本 |
| 11 | 16r 干净基线（同节点、独占、`numastat`） | **TODO** | 需要独占节点窗口 |
| 12 | HBM 页驻留证据 | **UNVERIFIED** | `KUPL_TOPOLOGY_REPORT`/`numastat -p` |
| 13 | `--write-coarse-cache` + slab-dir 兼容性 | **UNVERIFIED** | 按 `COMMAND_REFERENCE.md` 复核并实测 |
| 14 | `$CCS_MPI_OPTIONS` 内容 / `plm_donau` 可加载性 | **UNVERIFIED** | 进入作业读环境；`ldd` + 实际 drun 启动 |

---

**一句话总结**：这一天真正可复用的经验不是“L48 跑通了”，而是**一套判定纪律**：
版本字符串不是制品身份，文件名不是 dtype，PENDING 不是 PASS，登录 smoke 不是计算
节点结果，`wall` 不是端到端，fallback 不是 MG，2 节点分配不是跨节点 MPI。
把这些边界写进文档和技能，比多跑一个未验证的矩阵更有价值。
