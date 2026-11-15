---
name: pyqcu-logs-debug
description: 当任务需要写入或维护 logs/debug 的逐轮调试与修复日志时使用。
---
# logs/debug

Per-round debug and fix logs generated during development and bug-fixing sessions.

## File Pattern

`fix-log*.md` — per-round fix logs documenting individual bug fixes, root cause analysis, and verification results.

These are temporary/working files — final summaries are promoted to `logs/fix-report-*.md`
（在本目录的上层，见 `skills/pyqcu-logs-results/SKILL.md`）。

当前内容正是三个文件（2026-10-15 核对）：

| 文件 | 对应审查轮次 |
|---|---|
| `fix-log.md` | `git show dev73:logs/dev73/review-2026-07-28.md`（首轮） |
| `fix-log-r2.md` | `git show dev73:logs/dev73/review-2026-07-28-r2.md` |
| `fix-log-r3.md` | `git show dev73:logs/dev73/review-2026-07-28-r3.md` |

**审查报告不在工作区保留**——三份 `review-2026-07-28*.md` 已随 dev73 tag 归档并从
当前工作树清理；本目录只记录对应轮次，不要恢复 `logs/dev73/` 永久副本。

Methodology notes: worktree timeline bisection and asset-mtime × commit-time
cross-checking (key to locating silent regressions such as bug35).

## 失败定位法（2026-11-15 复盘，可复用）

**1. 先用 traceback 把故障钉到"哪一侧"再动手。** 同一函数被多处调用时，调用行
就是分类器：`kupl_multigrid.py::_read_local_slab` 的 `source_local = ...` 与
`gauge_local = ...` 分别对应源与组态。32 个 rank 同时抛
`OSError: Unable to open file (bad object header version number)` 且行号一致 →
文件侧（资产损坏），不是 MPI/halo/几何/数值问题；改并行参数或重跑同一命令无意义。
判据是"**所有 rank 同一调用点同一异常**"，只有一两个 rank 异常才是并行/绑定问题。

**2. 因果闭合用逐位对照实验，不用相关性。** 把"生产输入"与"参考输入"分别喂给同一
builder，比较产物的 `max_abs`：生产窗口与失败 cache **逐位相同**（`0.0`）、真实全局
窗口与 1-rank 参考**逐位相同**（`0.0`）才叫闭合。只给"改了之后 PASS"是弱证据。
同一手法给出比例证据（4-rank `8³×16` 有 `6144/8192 = 75.0%` 站点 gauge 错误）比
"差很多"有用得多。

**3. 记录脚本指纹，否则台账无法归因。** 一批数据跨多个脚本版本时，只能靠作业
`.err` 里的 traceback 行号反推版本（`<module>@1530` / `mode_solve@1326` 等），
字段也会在批次中途新增（本批 `hdf5_mpio_read` 就是 14:06–14:24 之间加的）。
新写驱动应在每行落 `script_sha256`；复盘旧批时必须显式写出"该行由哪个版本产生"。

**4. 性能类失败先排除输入与噪声，再怀疑算法。** 判读顺序
`status/returncode` → `*_read_mode` 与 `*_read_seconds` → `atol` →
`node/grid/shm` → 才看 `wall_seconds`。cold 读行（`gauge_read_seconds` 从 `2.4–5.2 s`
涨到 `29.9–35.6 s`）的 `wall` 会被抬高 `33–60×`，不是代码回退；同一 `run_id`
复测可差 `36×`（`builder` `72.1/78.5/5.64 s`），单样本不能定论。

**5. 台账级复盘用脚本，不靠人眼。** 一组只读检查器按维度分工：
表头 vs 列定义、每行字段数、`run_id` 唯一/跨文件重复、CSV vs 行 JSON、
残差分流与缺失标注、`report_sha256` 完整性、以及乘法分解（迭代 × grid × halo ×
内核）。先跑脚本再写结论，能把"哪一列从什么时候开始缺"这种问题变成可复现证据
（示例脚本组见 `pyqcu-kupl` 的矩阵参考 §复现与审计入口）。

**6. 资产类失败先做"身份核对"，再改代码。** 同一目录常并存语义不同的同名族文件：
`source_<lat>_c64.h5` 是 random seed42（不是 ones）、`..._c64_ones.h5` 是墙源、
`..._c64_ones_v2.h5` 与它逐字节相同（同 `file_sha256 9cdcfc82…f55a`、同
`201,375,008 B`）；`gauge_<lat>_c64.h5` 里的 dataset 实际可能是 `complex128`。
判据是 `file_sha256` + dataset dtype/shape，不是文件名，也不是 `st_size`（本次坏
ones 与好 `_v2` 同尺寸）。修复只做"换名重生成 + 保留 `*.corrupt-<date>` 备份"，
不回写、不修补；`_v2` 这类后缀是**资产版本**，不能被"按命名规则推断"吃掉——矩阵
驱动只认 `_ones/_point/_c64` 三种后缀，L32 的 ones 行因此 6 行全 FAIL。
