# KUPL 生产 HDF5 输入（source/gauge）不可读：根因、修复与验收（2026-11-15）

> 证据等级：**B/C 混合**。`rows/**` 侧的失败是可复核的 B 级（sha256、`returncode`、
> `.err` 栈）；文件**大小与 mtime** 来自 2026-11-15 11:07–11:20 远端共享目录的
> `ls -l` 快照（C 级，本机未挂载 `/home/share/nsls_yangyb`，无法重新 `stat`）。
> 本文只描述已验证事实；没有日志、作业号或调用栈支撑的触发者一律标"待确认"。
>
> 影响面：L32（`32³×64`）4 条 solve 全部失败、L24 点源 1 条失败；L32 因此在本轮
> 矩阵里 **0 PASS**。这不是 L32 几何或 KUPL 算法的问题（同批 L32 cache 的
> `verify_status=0`）。

## 1. 结论

1. **直接原因**：源文件在写入过程中被中断，HDF5 的 object header 没有完成，
   `h5py.File(..., "r")` 在 open 阶段就抛
   `OSError: Unable to open file (bad object header version number)`。文件长度
   已经到全尺寸（含正常元数据余量），所以文件**看起来是"全尺寸"的**，但 header
   不可解释。
2. **不是 HDF5 版本不兼容，也不是磁盘整体损坏**：同一目录、同一 `h5py`、同一时段
   的兄弟文件（L32 两个 gauge、`ones_v2`、L24 `ones`）都能打开；用这些 gauge 构建的
   L32/L24 cache 也都能 `verify_status=0`。
3. **放大器是复用逻辑**：旧解析路径只按"文件存在"复用，一个坏的但存在的文件会被
   反复选中，于是每一个用到它的 run 都稳定失败，看起来像"L32 不可算"。
4. **当前驱动已有读取侧闸门**：`h5_readable` 会先 open + 检查 `data` 数据集 +
   取 `size`，不可读就跳过并回落生成（`pyqcu/testing/kupl/test_kunpeng_all.py`
   行 310–337、425–477、550–599）。失败的 6 行由更早的版本写出，没有过这道闸门。
5. **写侧还不完整**：驱动自己的生成器是 `temp + os.replace`（行 497–503、517–523、
   539–546），但**没有 `fsync`**，也没有"写完重新打开校验 shape/dtype"的步骤。
   这两条是本轮之后仍需补的。

## 2. 证据表

### 2.1 文件清单

| 文件 | 大小 | mtime | dtype/角色 | h5py 结果 | 用途证据 |
|---|---:|---|---|---|---|
| `production_L32_v1/gauge_32x32x32x64_c128.h5` | 1.207 GB | — | complex128 gauge | **可开** | 作为可读对照（本轮行未使用） |
| `production_L32_v1/gauge_32x32x32x64_c64.h5` | **1,207,965,696 B** | — | **dataset `complex128`**（名字写 c64） | **可开** | L32 c64/c128 cache 的身份 `gauge_file_sha256=b2a57594…` 实际指这个 1.2 GB 文件 |
| `production_L32_c64/gauge_32x32x32x64_c64.h5` | 604 MB（`603,979,776 B`） | — | complex64 gauge | **可开**（2026-11-15 远端核验） | **真正的 c64 副本**：file `70bbe621…`、canonical `9ac7c15c…`（与上表 c128 文件 cast 后同值） |
| `production_L24/gauge_24x24x24x64_c64.h5` | **509,614,080 B** | — | **dataset `complex128`**（名字写 c64） | **可开** | 32 条 L24 solve 行都用它；L24 c64 cache 的 `gauge_file_sha256=4777e795…` 指的就是它；正确 c64 文件是 `gauge_global_c64.h5`（`8a482d6a…`） |
| `production_L24/gauge_global_c64.h5` | — | — | complex64 gauge | 可开（远端核验） | L24 c64 cache **应当**指向的文件（file `8a482d6a…`） |
| `production_L32_v1/source_32x32x32x64_c64_ones.h5` | **201,375,008 B** | **11:20:03** | wall 源（全 1） | **FAIL**（16:28 前）→ **OK**（16:28 后 `9cdcfc82…`） | 4 条 L32 solve 的 `source_path`，行里记的 `source_sha256=a9c2acd8…` 现在只对应 `.corrupt-20261115` 备份 |
| `production_L32_v1/source_32x32x32x64_c64_point.h5` | — | — | point 源 | **OK** | `dc427e47…`；快照后 `82ba09ec3a8e0d2d` 用它跑出 L32 c128 `mg2` PASS（`4.03e-10`） |
| `production_L32_v1/source_32x32x32x64_c64_ones_v2.h5` | 约 201.4 MB | **11:20:29** | wall 源（全 1） | **OK** | 2026-11-14 的点源重测用它跑出 `0.348549 s` |
| `production_L24/source_24x24x24x64_c64_point.h5` | **84,959,592 B** | **11:07:12** | point 源 | **FAIL** | `c3bd73d2999000f0` 的 `source_path`，`source_sha256=945782f3…` |
| `production_L24/source_24x24x24x64_c64_ones.h5` | 84.9 MB | **11:10:09** | wall 源（全 1） | **OK** | L24 全部 ones 行；`source_sha256=486d358e…` |

### 2.2 尺寸自洽核算（本文按布局公式复算）

KUPL canonical 布局是 `[spin=4, color=3, x, y, z, t]` 的复数数组，c64 每复数 8 B。

```text
L32 c64 source : 4*3*32^3*64 * 8 = 201,326,592 B
                 实测 201,375,008 B  → 多 48,416 B（约 0.024%，HDF5 元数据）
L24 c64 source : 4*3*24^3*64 * 8 =  84,934,656 B
                 实测  84,959,592 B  → 多 24,936 B（约 0.029%）
L32 c64 gauge  : 8*9*32^3*(64/2) * 8 =   603,979,776 B  （约 604 MB）
L32 c128 gauge : 2 × 603,979,776      = 1,207,959,552 B （约 1.207 GB）
L24 c64 gauge  : 8*9*24^3*(64/2) * 8 =   254,803,968 B  （约 255 MB）
```

结论：两个坏文件的长度**正好是"完整 payload + 正常元数据余量"**，所以
"按文件大小判断是否写完"必然误判。判断只能用 `h5py` 打开。

### 2.3 同一批次的对照与时间线

| 时间 | 事件 | 结论 |
|---|---|---|
| 11:07:12 | L24 `..._point.h5` 落盘 | 之后每次使用都 `bad object header` |
| 11:10:09 | L24 `..._ones.h5` 落盘 | 可开，L24 全部 ones 行正常 |
| 11:20:03 | L32 `..._ones.h5` 落盘 | 4 条 L32 solve 全部失败 |
| 11:20:29 | L32 `..._ones_v2.h5` 落盘 | 可开；**距坏文件 26 s** |
| 之后 | L32 c64/c128 cache 用 604 MB gauge 构建 | `verify_status=0`，证明 gauge 与磁盘都没坏 |

26 s 后同一目录、同一进程环境写出的同名变体完全可读，是"**写中断**"而不是"环境
不具备写 HDF5 能力"的强证据。两个坏文件都出现在"第一次生成"的时间点，而后续重写
的文件都正常——与"第一次写被打断、重试成功"的形态一致。

## 3. 根因

### 3.1 直接原因

`h5py` 的 open 阶段只读 superblock 与 root group 的 object header。栈是

```text
File "h5py/h5f.pyx", line 106, in h5py.h5f.open
OSError: Unable to open file (bad object header version number)
```

这条错误表示 header 的版本字节不是 HDF5 认识的取值（不是"文件不存在"、不是
"truncated"、也不是"dataset 缺失"）。典型成因是写入进程在 header 落盘前后被
中断/覆盖，使 header 区域停留在未初始化数据上。文件长度已到全尺寸——可能是
写入端预分配了数据区，也可能是数据写完但 header 未刷新；两种都与观测相容，
本文不区分。

### 3.2 为什么"全尺寸"不能当"写完"

HDF5 的写入顺序是：创建文件时写 superblock 与 root header，创建 dataset 时写
dataspace/datatype，数据按 chunk 写，最后 close 时刷新元数据。若进程在 header
写完之后、数据写完之前被杀，得到的是"长度短"的文件；若在 header 区被预分配但
版本字节尚未写定的窗口内被杀（或被另一进程原地重写同一路径），得到的就是本例：
**长度完整、header 不可解释**。两种形态都不能靠 `stat` 区分。

### 3.3 复用逻辑把一次性损坏放大成"稳定失败"

旧解析路径的复用判据是"文件存在且非空"，没有 open 校验。任务记录把这个入口叫
`ensure_source`；本机当前树没有同名函数，对应实现是
`production_source_path` + `resolve_source` 的候选循环（行 290–303、425–477）。
只要坏文件在候选列表里排第一，每个 run 都会选中它，于是：

- 4 条 L32 solve 全部 rc=1、无 `report.json`；
- 台账的 `notes` 只写 `no report.json`，**看不到**真正的 h5py 错误（栈只在
  `logs/<run_id>.err` 里）；
- 下游读者容易把"L32 全 FAIL"误读成几何/算法限制。

### 3.4 已排除的假设

| 假设 | 排除证据 |
|---|---|
| HDF5 版本不兼容 | 同目录同批次的 `ones_v2`、两个 gauge、L24 `ones` 都能被同一 `h5py` 打开 |
| 磁盘/文件系统整体损坏 | 坏文件与好文件在同一目录、同一时段；用同目录 gauge 构建的 cache `verify_status=0` |
| 文件被截断 | 长度等于完整 payload + 元数据余量（§2.2） |
| dtype 不匹配（c64 文件被 c128 读） | 3 条 L24 c128 PASS 行（`f19ba1fe`、`f212adf8`、`d2ce9be3`）用的就是 `_c64` 的 source 与 gauge，说明 c64→c128 的转换在 open 之后完成；4 条 L32 行里 c64 与 c128 也都失败在同一 open 调用 |
| KUPL 库/MPI 问题 | 失败栈完全在 `h5py`，`mpirun` 正常拉起并在 open 阶段退出 |

### 3.5 待确认与本轮已闭合项

仍然**未确认**：

1. **谁打断了写入**：没有这两个文件的生成作业号、写进程 PID 或磁盘告警记录。
   可能是生成脚本被 `dkill`/超时终止、两个进程同时写同一路径、或写完后被并发
   rsync 覆盖——三种都与证据相容，**待确认**。
2. **是否用了原地覆盖**：若生成器先 `h5py.File(path, "w")` 再写 dataset，第二次
   尝试会在同一 inode 上重写 header；这与"11:20:03 坏、11:20:29 好"相容，但没有
   生成器源码/日志可证，**待确认**。

本轮**已闭合**（见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3）：

3. **`ones.h5` 与 `ones_v2.h5` 语义一致**：`file_sha256` 同为
   `9cdcfc821437816f342f74859ce7c5a44eae8f02b4fe6df7eba2c34a9b2ef55a`
   （`9cdcfc82…f55a`），dataset 逐元素 `1+0i`，二者**逐字节相同**；**无需重新生成**。
   16:28 的修复就是把这份内容落到 `..._c64_ones.h5`，坏字节保留为
   `*.corrupt-20261115`（`a9c2acd8…`）。
4. **同目录的 `source_32x32x32x64_c64.h5` 是 random seed42，不是 ones**：文件名不带
   `_ones` 的那个文件不能顶替坏 ones，即使它可读、尺寸相同。凡是要 wall 源，只能用
   `..._c64_ones.h5`（= `ones_v2` 内容）。
5. **"L32 c64 运行曾指向 c128 gauge"** 已解释清楚：本轮 42 行的 `gauge_path` 确实全是
   `production_L32_v1/gauge_32x32x32x64_c64.h5`，但**该文件本身是 1,207,965,696 B 的
   `complex128` 数据集**（`b2a57594…`）；真正的 604 MB c64 副本在 `production_L32_c64/`
   （`70bbe621…`）。所以"指向 c128"不是拿错路径，而是 c64 cache 的
   `identity.gauge_file_sha256` 写进了这份 c128 副本 —— 见 §4.4 与验收门 4。

## 4. 修复

### 4.1 写入侧（必须）

1. **`temp + fsync + rename`**：写 `.tmp` → `handle.flush()` → `os.fsync(fd)` →
   `os.replace()` → 对父目录 `fsync`。当前驱动生成器只做了 `temp + replace`
   （行 497–503、517–523、539–546），**没有 fsync**；掉电/被杀时 rename 可能先于
   数据落盘。
2. **写完立即回读校验**：重新 `h5py.File(tmp, "r")`，检查
   `data` 存在、`shape == (4,3,X,Y,Z,T)`、`dtype == complex64/complex128`、
   `size` 与期望一致，任一不满足就不 rename、直接报错。当前 `h5_readable` 只检查
   open + `data` + `size`，**没有 shape/dtype 校验**。
3. **禁止原地覆盖生产输入**：同一路径只允许"生成到新名字再 rename"，旧文件改名
   保留（如 `*.broken-<YYYYMMDDHHMMSS>`），否则无法事后区分"当时就坏"与"后来被改坏"。
4. **写侧留日志**：把生成作业号、PID、目标路径、期望 shape/dtype/size、耗时写入
   `meta`，这是本次唯一缺失的因果链。

**2026-11-15 只读审计的补强结论**（逐文件行号明细以该审计报告为准，报告在会话工作目录、
未入仓库）：

- 原输入（LIME、`data/production_L*/gauge|source`、slab 目录、`--coarse-cache`）在源码层
  只有 `"r"` 打开点，**超时/被杀不会改写它们**；被超时破坏的是 solution/report/log 这些
  "上一次的产物"（最终路径直写、原地截断）。
- 脚本内**没有** `.corrupt-*` 逻辑（`grep -c corrupt` 为 `0`）；16:28 的改名是会话内的
  手工动作。想让它成为可复现行为，必须按上面第 3 条写进脚本。
- 8 个被审文件里 `os.fsync` 计数**全为 0**：`temp + os.replace` 只保证"不出现半文件被
  当作最终文件"，不保证掉电后 rename 早于数据落盘。R3 三件套（tmp → fsync → rename →
  fsync dir）仍是待补项。
- `real` gauge 缺失/不可读时会静默回退到 identity gauge（`--allow-missing-inputs`）而
  CSV 仍记 `gauge_kind=real`；这与"real gauge 不重生成"的规则直接冲突，必须改为 FAIL/SKIP
  或显式 opt-in + `gauge_origin`。

### 4.2 读取侧闸门（已实现，建议加强）

| 检查 | 现状 | 位置 |
|---|---|---|
| 文件存在、非空 | 已实现 | `h5_readable` 行 318–324 |
| `h5py` 可打开 | 已实现 | 行 329–335 |
| 存在 `data` 数据集 | 已实现 | 行 331–332 |
| 能取 `dataset.size` | 已实现 | 行 333–335 |
| shape == `(4,3,X,Y,Z,T)` | **未实现** | 建议补 |
| dtype ∈ {complex64, complex128} | **未实现** | 建议补 |
| 抽样读一个元素（真实 I/O 路径） | **未实现** | 建议补（`d[0,0,0,0,0,0]`） |

闸门失败时的行为已正确：把原因写进 `notes`（`<role> source unreadable (...)`），
继续试下一个候选，最后回落到 `outdir` 生成，而不是让整个作业死在 rank 里
（行 443–477）。

### 4.3 失败恢复

1. 坏文件**改名保留**，不要删除；删除会丢掉"当时坏在哪"的证据。
2. 重新生成后把新 `sha256` 写进 row/report；**旧 FAIL 行不要原地改**，用
   `--run-label` 开新 `run_id`，让"坏输入"与"好输入"两段历史都留在台账里。
3. 若确认 `ones_v2` 与 `ones` 语义等价，可以把 `ones_v2` 作为新的生产文件并在
   `cache_index`/`file_hashes` 里登记新 hash；此时旧行的 `source_sha256` 会与新
   文件不一致——这是预期行为，不要为了"看起来一致"去改旧 hash。
4. 重跑前先处理 `--skip-failed`：它会跳过已 FAIL 的 `run_id`，导致补跑根本不发生
   （本轮 3 个 MG 空洞就是这么来的）。

### 4.4 L32 的 gauge 与 cache 选择

| 项 | 本轮实测 | 建议 |
|---|---|---|
| `production_L32_v1/gauge_32x32x32x64_c64.h5` | **1,207,965,696 B**，dataset `complex128`（名字写 c64），file `b2a57594…` | 可读、数值有效；**但它是 c128 副本**，不能当作"c64 文件身份" |
| `production_L32_c64/gauge_32x32x32x64_c64.h5` | 604 MB（`8*9*32³*32*8 = 603,979,776 B`），complex64，file `70bbe621…`，canonical `9ac7c15c…` | **真正的 c64 副本**；c64 cache 的 identity 应指向它 |
| L32 c64 cache | `cache_global_32x32x32x64_c64_...`，`verify_status=0`，`build_mg_cache_seconds=100.576`，note `global build rc=124 elapsed=602.2s` | 数值可用（canonical `9ac7c15c…` 与 604 MB 文件一致），但 identity 的 `gauge_file_sha256` 写的是 `b2a57594…` → **重签/重建**；注明"非干净构建"（600 s 超时前已落盘） |
| L32 c128 cache | `cache_global_32x32x32x64_c128_...`（679 MB），`verify_status=0`，`build_mg_cache_seconds=130.371`，file `b2a57594…` / canonical `52f3c07a…` | **不重建**：它读的就是这份 c128 文件，identity 正确 |

要点：

1. **c128 求解不一定需要 c128 gauge 文件**，本轮 c128 行与 c128 cache 都是读同一份
   `complex128` 数据后在库内取精度。但 identity 的 `gauge_file_sha256` 必须是**实际被
   identity 覆盖的那份字节**：c64 cache 现在写的是 1.2 GB 的 c128 文件，与实际应消费的
   604 MB c64 文件不是同一份。
2. **c64 ↔ c128 的 canonical 值在这里相同**（`9ac7c15c…`），说明 c128 文件 cast 到 c64 后
   数值等价。因此换回 604 MB 文件**不需要重跑数值**，只需把 `gauge_file_sha256` 重签为
   `70bbe621…` 并复核 `cache_identity_sha256`，或按新 identity 重建一次 cache。
3. L24 侧是同类问题：`production_L24/gauge_24x24x24x64_c64.h5`（509,614,080 B，dataset
   `complex128`，file `4777e795…`）被 L24 c64 cache 记进 `gauge_file_sha256`，正确 c64 文件
   是 `production_L24/gauge_global_c64.h5`（file `8a482d6a…`）→ **建议重签**，同样不重跑
   数值。
4. **MG cache 不含 source 哈希**（identity 只有
   lattice/dtype/mass/csw/mg_grid/nv_iters/dof/seed/gauge_kind/ranks/grid），source 修复不使
   任何 cache 失效。

另外，驱动只按**文件名**推断 `gauge_file_dtype`/`source_file_dtype`
（`test_kunpeng_all.py` 行 1352–1355：文件名含 `c64` 即记 c64，否则记 c128）。
文件名与内容不一致时这两列会误报，所以"L32 c64 用了 c128 gauge"这类问题在台账里
**看不出来**；要么改成读 dataset dtype，要么在 `notes` 里显式写文件设备号/inode。

## 5. 验收门槛

修完必须依次过四道门，任何一道不过都不许把 L32 写回矩阵：

| # | 门槛 | 命令/判据 | 本轮状态 |
|---|---|---|---|
| 1 | `h5py` 能打开 | `python3 -c "import h5py;f=h5py.File('FILE','r');print(f['data'].shape,f['data'].dtype)"` | 坏文件 FAIL，好文件 PASS |
| 2 | 单 rank 读一次真实数据 | 单 rank 跑 `--source <file>`，`report.json` 有 `source_read_seconds` 且无 `bad object header` | 未系统执行 |
| 3 | 16×18 与 32×18 各一条 solve | `rows/` 里出现 `status=PASS` 且 `final_true_residual` ≤ `atol` | **部分达成**：快照内 L32 无 PASS；快照后 `82ba09ec3a8e0d2d`（32×18、c128、`mg2`、point 源）PASS、残差 `4.03e-10`，满足 32×18 这一半。16×18 仍缺，且 6 条 FAIL（`07c80fc8`、`23daf98e`、`327f9aa5`、`35548a2b`、`44faded2`、`c9bfd6bf`）的 ones 配置仍需**一次首次有效运行** |
| 4 | L32 c64/c128 cache identity | `cache_index.json` 的 `identity.gauge_file_sha256` 与实际消费文件 sha256 一致；`verify_status=0` | **c128 过**（`b2a57594…` 就是那份 1.2 GB c128 文件）；**c64 未过**（应指 604 MB 的 `production_L32_c64/gauge_32x32x32x64_c64.h5`，`70bbe621…`）→ 重签/重建，不重跑数值 |

第 2 道门是这次最容易漏的：`h5_readable` 在 Python 侧打开的是同一路径，但它只取
`dataset.size`（元数据），不做真实读。要证明"rank 里能读"，必须让 `mpirun` 内的
进程真正读一次。

## 6. 复现与数据路径

```bash
# 1) 判可读（不要用 stat 大小判断）
python3 - <<'PY'
import h5py
for path in (
    "SOURCE_OR_GAUGE_1.h5",
    "SOURCE_OR_GAUGE_2.h5",
):
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        print(path, data.shape, data.dtype, data.size)
PY

# 2) 看失败行的真实原因（台账 notes 里没有）
grep -n "bad object header\|MPI_ABORT\|handshake" \
  data/kunpeng_all_20261115/logs/<run_id>.err

# 3) 重新生成并重跑（新 run_id，不改旧行）
python3 -B pyqcu/testing/kupl/test_kunpeng_all.py --mode solve \
  --lat 32 32 32 64 --mass -0.277 --csw 1.160920226 --dtype c64 --atol 1e-6 \
  --solver bistabcg --grid 1 1 1 32 --ranks 32 --threads 18 --shm 0 \
  --source ones --gauge real --seed 42 --run-label after-source-fix
```

| 证据 | 位置 | 说明 |
|---|---|---|
| 4 条 L32 FAIL 行 | `data/kunpeng_all_20261115/rows/{327f9aa5,c9bfd6bf,23daf98e,35548a2b}*.json` | `source_sha256=a9c2acd8…`，`returncode=1` |
| L24 point FAIL 行 | `data/kunpeng_all_20261115/rows/c3bd73d2999000f0.json` | `source_sha256=945782f3…` |
| 栈 | `data/kunpeng_all_20261115/logs/<run_id>.err` | `h5py/h5f.pyx` + `bad object header version number` |
| cache 身份 | `data/kunpeng_all_20261115/cache_index.json`、`caches/*.json` | `gauge_file_sha256`、`gauge_canonical_sha256`、`verify_status` |
| 可读对照 | 2026-11-14 的 `source_..._ones_v2.h5` 批次（`docs/kupl_mpi_scaling_20261114.md` §5.4.1） | 201.4 MB，点源重测跑出 `0.348549 s` |
| 失败行在矩阵中的位置 | [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §6.6、§7 | L32 0 PASS |

> 本机没有 `data/production_L24`、`data/production_L32_v1`，也没有 cache `.h5`；
> 上表只能复核 sha256、`returncode` 与栈，不能复核大小与 mtime。

## 7. 教训清单

1. **不要用 `stat` 判断 HDF5 是否写完**：全尺寸也可以是坏文件。
2. **不要用文件名推断 dtype/角色**：`_c64`/`_c128`/`_ones`/`_v2` 只是约定。
3. **不要按"存在"复用生产输入**：复用前必须 open 校验；校验失败要改名保留 + 重生成。
4. **坏输入不是算法结论**：L32 的 6 条 FAIL 只说明"这一批输入坏了"，不说明 L32
   不可算；同批 cache `verify_status=0` 已经反证。
5. **台账要写失败签名**：`notes` 只写 `no report.json` 会让根因埋进 22–43 KB 的
   `.err`；应把 `bad object header`、`handshake timeout`、`MPI_ABORT` 落成结构化
   `failure_reason`。
6. **重试成功不是"没问题"**：`ones_v2` 26 s 后成功，说明写侧没有原子性；不修写侧，
   下一次仍会在别的文件上复发。
7. **源语义要按内容判，不按文件名判**：同目录的 `source_..._c64.h5` 是 **random
   seed42**，`..._ones.h5` / `..._ones_v2.h5` 才是全 `1+0i`。三者尺寸同级、都能打开，
   只看文件名会把随机源当 wall 源用。
8. **`gauge_file_dtype`/`source_file_dtype` 两列只能当文件名标签**：L24 与 L32 两份
   `real` gauge 在磁盘上都是 `complex128`（509,614,080 B / 1,207,965,696 B），
   31 条 PASS 里 30 条的 `c64` 标注是误报；判 dtype 只能打开 dataset。
9. **输入只读是一条要写进代码的规则，不是现状总结**：源码层 gauge/source/cache 确实
   只读（超时改不了它们），但 solution/report/log 是最终路径直写、无 fsync、无
   `.corrupt-*` 备份，且 `real` gauge 缺失会静默换成 identity gauge。这四条要按
   [`kupl_kunpeng_matrix_20261115.md`](kupl_kunpeng_matrix_20261115.md) §12 的 R1–R7 修，
   修好之前"输入没被动过"只是**因为没人写它**，不是因为代码守着。
10. **"重跑"必须换 `run_id`**：`--resume` 只跳过 PASS/SKIP，`TIMEOUT` 会被重跑并原地
    覆盖同 `run_id` 的 solution/report/log；要保留两段历史只能用 `--run-label`。

## 8. 本轮裁定（追加）

修复完成后，本轮对"哪些数据要重跑、哪些 cache 要重建"的裁定如下（权威版本见
[`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §3）：

| 对象 | 判定 | 依据 |
|---|---|---|
| 31 条 PASS / L24 数据 | **不重跑** | L24 c64 gauge 的 c128 文件 cast 后与 c64 值等价；MG cache 不含 source 哈希；cache 数值等价 |
| L32 6 条 FAIL | **一次首次有效运行**（不是重测） | 6 条都用了坏 ones `a9c2acd8…`，solver 未启动，无有效性能数据 |
| L24 point `c3bd73d2` | **放弃** | point 轴整体放弃（耗时代价过大），源虽已修复但不跑 |
| L32 c64 cache | **重签/重建 identity** | `gauge_file_sha256` 指到 1.2 GB c128 文件；应改指 604 MB c64 文件（`70bbe621…`），canonical `9ac7c15c…` 不变 |
| L32 c128 cache（679 MB） | **不重建** | `b2a57594…`/`52f3c07a…` 与实际消费文件一致 |
| L24 c64 cache | **建议重签** | 应改指 `production_L24/gauge_global_c64.h5`（`8a482d6a…`） |
| 所有 MG cache | **不失效** | identity 不含 source 哈希，只含 lattice/dtype/mass/csw/mg_grid/nv_iters/dof/seed/gauge_kind/ranks/grid |

判定规则（同口径才不重测）：source 语义与 hash、gauge dtype 与 canonical、cache
identity、`atol`、`grid`/`ranks`/`threads`/`shm`/`solver`/`mg_restart` 全部一致；任一不同
必须重跑。L32 的 6 条 FAIL 不适用"重测"一词——它们是首次有效运行；L32 c64/c128 与 L24
的同口径对比在 source 修复后有效。

另：**MPI h5py 不在本文范围**。接口检查结论（h5py `3.7.0`/HDF5 `1.10.6`、
`get_config().mpi=False`、无 `mpio`、无一行 `hdf5_mpio_read=true`、per-rank 全量读回退）与"不重新
编译/安装"的裁定见 [`kupl_matrix_scope_20261115.md`](kupl_matrix_scope_20261115.md) §1。
