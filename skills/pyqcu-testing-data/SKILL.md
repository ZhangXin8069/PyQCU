---
name: pyqcu-testing-data
description: 当任务涉及 pyqcu/testing/data 的 HDF5 参考数据、with_data 测试、生产 gauge/source 输入文件损坏排查或 HDF5 资产写入契约时使用。
---
# pyqcu/testing/data

Reference HDF5 files for test validation. Used when tests are run with `with_data=True` to validate against precomputed expected results.

Contains gauge fields, fermion sources, and expected operator/solver outputs for various lattice sizes.

Current end-to-end correctness defaults to `8x8x8x16`. The L16/L8 Wilson and
Clover files here are historical fixtures. `with_data=True` must receive an explicit
historical lattice and must not silently replace the current correctness lattice.

Reference HDF5 files are gitignored (not committed)。**注意：这些 HDF5 缺失时
`with_data=True` 的用例会在 OSError 层失败，属于已知且预期的情况**（例如 `logs/session-2026-08-24/README.md`
里 `r1_baseline.py` 记的 8/10 PASS、2 FAIL 就是这一层），不要当成代码回归。

Regenerate deterministically via `pyqcu/testing/regression/session-2026-08-24/gen_clover_ref.py`
and `gen_wilson_ref.py` — bit-level reproducible across architectures；solver dual-backend
cross-check rel=8.6e-07。生成器的逐项说明与 README 位置在
`pyqcu/testing/regression/session-2026-08-24/AGENTS.md`（本技能不再另建副本；
旧的 `skills/pyqcu-testing-regression-session-2026-08-24/` 路径已随技能改名删除）。

## 输入资产写入契约（2026-11-15）

生产 gauge/source（`data/production_L24/`、`data/production_L32_v1/`）是长作业的
唯一输入，写坏一次会让整批 rank 在求解前全部退出。规则：

- **原子写**：同目录唯一临时名（`.{name}.{pid}.tmp`）→ `h5py.File(tmp, "w")`
  写完 → 句柄内 `flush()` → `os.fsync(fileobj.fileno())` → `os.replace(tmp, path)`
  → 对父目录再 `fsync`。直接 `h5py.File(path, "w")` 是就地截断：进程被杀或并发读
  只要落在写窗口内，文件就永久损坏。`os.replace` 前不 `fsync` 只保证"改名原子"，
  断电/杀进程仍可能留下半写数据。
- **读取端只接受 rename 后的最终名**，并在提交作业前用廉价门校验：
  `h5py.File(p,"r")` + 存在 `data` + 能取到 `int(ds.size)`（矩阵驱动的
  `h5_readable()`，只判"非空且可打开"，**不核对期望 shape/dtype**）。
  该门只能证明"此刻可读"，不能证明写入完整——并发写窗口内它也会通过。
- **原地写是真实缺陷来源**：`pyqcu/testing/simple/_common.py::write_h5` 与
  `pyqcu/tools/_io.py::save_tensor_h5` 都是就地 `h5py.File(path, "w")`（无 tmp、无
  `fsync`），任何用它写生产输入的脚本一旦被杀就留下半写 header。驱动侧的源生成
  （`test_kunpeng_all.py` 的 `write_point_source`/`write_wall_source`/
  `generate_random_source`）已用 `.{name}.{pid}.tmp` + `os.replace`，但仍缺
  `flush()`+`fsync()`；超时/被杀只留 `.tmp` 或半写最终文件，**只能备份成
  `.corrupt-<date>` 后重生成，不能修补**。详见 `pyqcu-tools` 的写路径表。
- **输入资产不可变（只读）**：`data/production_*/**` 只被读取，任何任务都不得回写、
  不得"就地修好"。需要修就换名重生成并新开 run_id，旧坏文件保留为
  `*.corrupt-<date>`（L32 ones `a9c2acd8…`、L24 point `945782f3…`）。
- **文件大小不等于完整性**：坏文件的 `st_size` 可以正常。文件字节还含 HDF5 元数据
  开销，与 dataset 理论字节不同：L32 source `201,375,008 B` 文件 vs `201,326,592 B`
  dataset、L32 gauge c128 文件 `1,207,965,696 B` vs `1,207,959,552 B` dataset、
  L24 gauge c128 文件 `509,614,080 B` vs `509,607,936 B`。**用 `st_size` 反推
  dtype/格点或判完整性都会误判**。

## `bad object header` 根因与排查

症状（2026-11-15，L32 与 L24 点源）：

```text
File ".../h5py/_hl/files.py", line 226, in make_fid -> h5f.open
OSError: Unable to open file (bad object header version number)
```

- 32 个 rank **同时**在 `kupl_multigrid.py::_read_local_slab` 的 `h5py.File(...)`
  处抛同一错误 → 不是 MPI/halo/几何问题，也不是数值失败；先怀疑**输入文件**。
- 根因是 superblock/object header 半写或被截成 0（非原子写 + 中断/并发读），
  读取端因此拒绝；**与 h5py/HDF5 版本无关，文件不可修复，只能重新生成**。
  换 SDK、加 `driver='mpio'`、重跑同一条命令都不会好。
- 定位顺序：① traceback 里是 `gauge_local = _read_local_slab(...)` 还是
  `source_local = _read_local_slab(...)`——这决定是 gauge 还是 source；
  ② 同格点的 gauge 与 source 分别 `h5_readable()`；③ 查同目录是否有 `_v2`
  等重新生成版本。本次 L32 的 gauge 全部可读、只有 source 不可读。

## `ones_v2` / `source...c64` 文件名语义

| 名字 | 内容 |
|---|---|
| `source_<lat>_c64.h5` | **随机源（seed 42）**：不是 ones，**不能**用它顶替坏 ones（即使可读、尺寸相同） |
| `source_<lat>_c64_ones.h5` | 全 `1+0i`（墙）源；2026-11-15 16:28 起为可读版（`9cdcfc82…f55a`） |
| `source_<lat>_c64_ones_v2.h5` | 重新生成的全 `1+0i` 墙源；与 `ones.h5` **逐字节相同**（同 sha256、同 201,375,008 B） |
| `source_<lat>_c64_point.h5` | 点源；L24 这份于 16:28 重生成（`b437e68f…`） |
| `gauge_<lat>_c64.h5` | 真实 SU(3) 组态；**L24/L32 生产目录里的这两份 dataset 实际是 `complex128`**（名字写 c64） |
| `gauge_global_c64.h5` / `production_L32_c64/gauge_..._c64.h5` | 真正的 c64 副本（L24 `8a482d6a…`、L32 `70bbe621…`），c64 cache identity 应指它 |

要点（2026-11-15 裁定）：

- `ones.h5` 与 `ones_v2.h5` 的 `file_sha256` 同为 `9cdcfc82…f55a`、dataset 全 `1+0i`、
  **逐字节相同 → 无需重新生成**。修复就是把这份内容落到 `..._c64_ones.h5`，坏字节保留为
  `*.corrupt-20261115`（L32 `a9c2acd8…`、L24 `945782f3…`）。
- `_v2` 是**资产版本**，不是装饰：只按 `_ones.h5` 推断会命中坏文件。矩阵驱动的
  `production_source_path()` 只认 `_ones/_point/_c64` 三种后缀，因此 L32 的 ones 行曾解析
  到坏文件，**6 行**全 FAIL（`07c80fc8`/`23daf98e…`/`327f9aa5…`/`35548a2b…`/`44faded2`/
  `c9bfd6bf…`）。这 6 行**没有有效性能数据**，需要一次**首次有效运行**（不是"重测"）。
- **换资产或换格点时，先枚举目录并逐个 `h5_readable()`，不要按命名规则假定。**

## 尺寸与读取预算

`V = Lx*Ly*Lz*Lt`；gauge `3*3*4*V` 个复数量，source `4*3*V` 个：

| 格点 | gauge c64 | gauge c128 | source c64 |
|---|---:|---:|---:|
| L24 `24³×64` | 254.8 MB | 509.6 MB | 84.9 MB |
| L32 `32³×64` | **603,979,776 B**（≈604 MB） | **1,207,959,552 B**（≈1.2 GB） | 201.4 MB |

- 32 rank 各自整文件读 = 32× 流量；只有 h5py 带 MPI 时才能走 `driver='mpio'`
  集体读（见 `pyqcu-tools`）。当前鲲鹏 conda h5py 无 MPI，`--mode prewarm`
  预热页缓存是唯一低成本的重复读方案。
- c128 gauge 约 2× c64（`1.2 GB`），L32 c128 的 cache 与 scratch 未做过容量验证。

## cache identity（Legacy-MG）

cache 复用只看 manifest `identity`，不看文件名或文件大小：

- identity 字段：`lattice`、`dtype`、`mass`、`csw`、`mg_grid`、`nv_iters`、`dof`、
  `seed`、`gauge_kind`、`ranks`、`grid`（global cache 固定 1 rank / `1 1 1 1`）。
- **不含** `atol`/`coarse_tol_factor`/`coarse_max_iter`/`restart`/`deflate`：
  这些不改变 cache 内容，改它们不需要重建。
- manifest 是 cache 同名的 `.json`（`setup_seconds`、`identity`、`file_sha256`、
  `setup_method`）；`cache_identity_sha256` 是 identity JSON 的 SHA1 前 16 位。
- 命中后仍要校验 manifest 与 `.h5` 的 `file_sha256`；identity 相同但哈希不同说明
  文件被改写过，必须重建。
- **`gauge_file_sha256` ≠ `gauge_canonical_sha256`**：前者是磁盘文件字节哈希（c64/c128
  副本不同值不同），后者是转成 canonical 布局/dtype 后的内容哈希（数值等价时不变）。
  换文件前先算 canonical；只有 canonical 变才必须重建，否则只需重签 identity。
- **identity 不含 source 哈希**（只有 lattice/dtype/mass/csw/mg_grid/nv_iters/dof/seed/
  gauge_kind/ranks/grid），所以 source 修复/换文件**不会**使任何 MG cache 失效。
- 已确认的两个重签项（2026-11-15，数值不变、不重跑）：L32 c64 cache 的
  `gauge_file_sha256=b2a57594…` 指到了 1.2 GB 的 c128 文件，应改指
  `production_L32_c64/gauge_32x32x32x64_c64.h5`（`70bbe621…`，canonical 仍 `9ac7c15c…`）；
  L24 c64 cache 的 `4777e795…` 同理，应改指 `production_L24/gauge_global_c64.h5`
  （`8a482d6a…`）。L32 c128 cache（679 MB，`b2a57594…`/`52f3c07a…`）identity 正确，
  **不重建**。
