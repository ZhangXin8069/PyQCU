# PyQCU L48 / 远端 十轮终审清单（2026-11-17）

> **2026-11-18 0.1.18 复查注记**：本文件前半部分是 **0.1.17 历史快照**，其中 R1/R2/R6/R7/R9
> 的 FAIL 是当时状态。0.1.18 的十轮执行结果、未完成 DEFERRED 项和最终 go/no-go 记录见
> [`logs/v202611180010.txt`](../logs/v202611180010.txt)；不得用历史 FAIL 直接否决
> 0.1.18，也不得把历史 0.1.17 结论改写成已通过。

> **性质**：发布前终审的执行清单，**不是发布记录**。本文只定义“每轮审什么、怎么跑、
> 什么算通过”，并登记**当前快照**下每轮的 PASS / FAIL / UNVERIFIED 与证据路径。
> 本文不执行发布：不 commit、不 push、不打 tag、不 build、不 upload、不改版本。
>
> **快照**：`git rev-parse HEAD` = `1f65469e6928bd5dc7788c0d017dbbe1c0c776e3`，
> `git describe --tags --always` = `test37-1-g1f65469e`，版本 `0.1.17`
> （`pyproject.toml:11`、`pyqcu/__init__.py:3`、`docs/en/conf.py:13`、
> `docs/zh/conf.py:13`）。本地 `git status --short` 撰写时 **27** 条、新增本文后 **28** 条；
> 2026-11-17 预检时 **47** 条（24 M + 23 ??，含并发批次未提交改动），因此**任何一轮的
> 结论都必须绑定到一个已提交的评审快照**，不能引用“此刻的
> 工作区”（见 R1）。
>
> **总门禁**：**十轮全部 PASS 之前不得发布**。任一轮 FAIL 或 UNVERIFIED，发布动作
> （`build` / `twine upload` / `git tag` / 双远端 push）一律禁止。
>
> **边界**：本清单只新增本文一个文件（`docs/**`），未改代码、版本、标签、
> `pyqcu/testing/**`、`pyqcu/**` 实现，也未改任何 `AGENTS.md`。
>
> **2026-11-17 docs 门批次注记**：本文的计数事实已刷新到预检实测值（`--check`
> 20/20、工作树 21 个 `pyqcu/test/*.py`、47 条未提交项、L48 计划 17/19 行），
> 并补齐 docs/skills 索引；R1–R10 的裁决与证据不变，未发布、未改版本/标签。

---

## 0. 结论摘要（本快照）

| 轮次 | 主题 | 本快照状态 | 一句话结论 |
|---|---|---|---|
| R1 | 范围 / 核心零未授权改动 | **FAIL**（工作区未冻结）/ 内容本身 PASS | 核心代码零改动成立，但工作区有 47 条未提交项，无法作为评审基线 |
| R2 | 与 `test_kunpeng_all.py` 的端口对等 | **FAIL** | 七模名称与 resume 语义对齐；100 列 CSV / 21 sheet 未接线，`validate` 同名异义 |
| R3 | 数值正确性 / 独立真残差 | **PASS（小格点）** / **UNVERIFIED（L48）** | 19/19 `--quick` PASS，native KUPL solve 经 driver 独立残差 PASS；无任何 L48 数值 |
| R4 | CLI / API / 退出码 | **PASS（本机工作区）** / **UNVERIFIED（3.10 远端最终树）** | `--check` 20/20（19 cases）；用法错 2、全 SKIP 1、空台账 1 均实测 |
| R5 | HDF5 / LIME 原子性与损坏恢复 | **PASS（HDF5 写/读）** / **FAIL（LIME 入口、quarantine）** | `io.py` 五段式原子写 + 损坏拒绝已实测；端口无 `--gauge-lime`，生成器无 fsync |
| R6 | MPI / 两节点 / shm / 线程 / cache | **FAIL（两节点）** / PASS（线程、shm 策略、cache 探测） | 无 rankfile/`CCS_ALLOC_FILE` launcher；两节点作业 `2152277` 仍 PENDING |
| R7 | 打包 / sdist | **FAIL（待重建）** | 0.1.17 sdist 只有 16 个 `pyqcu/test/*.py`，工作树 21 个（新增 5 个）；`dist/` 仍是旧产物 |
| R8 | docs / skills / AGENTS / form | **PASS（本批修复后）** | form PASS；docs 审计 11 条（预检实测）已全部修复为 `PASS docs=58 skills=36`；“21 vs 19 tier 0” 漂移 8 处已改 19、6 处旧 CUDA 计数 5 已改 3；本文与矩阵页已登记 |
| R9 | 独立只读复审 + 版本/标签链 | **FAIL（未冻结）** | `tag-chain --check` 有唯一历史 `[BLOCK] bug45`（接受为历史例外）；版本未升、工作区未提交 |
| R10 | 发布 dry-run | **UNVERIFIED（未执行，按约束）** | 需在新版本 + 干净提交树上重建 sdist 后跑 13/13 与 `twine check` |

**R1 + R9 未冻结时，R2–R8 的结论只能算“候选树上的预检”，不能作为发布证据。**

---

## 0.1 四条硬性红线（必须写在评审结论最前面）

### A. `0.1.17` / `test37` 已经发布，**不得覆盖**

| 证据 | 值 |
|---|---|
| PyPI 最新版本 | `0.1.17`（`curl -s https://pypi.org/pypi/pyqcu/json` → `info.version`） |
| PyPI 0.1.17 文件 | `pyqcu-0.1.17.tar.gz`，`912,755 B`，`upload_time 2026-11-16T13:24:22Z` |
| PyPI 0.1.17 sha256 | `82d1af5f8f0ac8b2c5f38b4a3c9d5cff32fd114cb6ebaf196b9e631b29b19dca` |
| 本地 `dist/pyqcu-0.1.17.tar.gz` sha256 | **同值** `82d1af5f…b19dca`（`sha256sum`） |
| `test37` peeled commit | `e0b25a75626c8bbf94a7d219606264dd0ce66f76`（2026-11-16） |

**裁定**：PyPI 拒绝同版本重传，yank 也不释放版本号（
`skills/pyqcu-pypi-release/references/release-checklist.md` §7「400 File already exists」、
`SKILL.md`「回滚」）。下一版必须是**新版本**，建议 `0.1.18`，对应新标签 `test38`
（`follow test37`）。**任何“修补 0.1.17”的路径都是发布事故**。

### B. `bug45` 标签链阻塞是**历史例外**，不重写

`bash skills/pyqcu-tag/tag-chain.sh --check` 本快照输出：

```text
Managed tags: 343
[BLOCK] bug45 的 peeled commit 不是 dev106 的祖先，且没有可验证的共同 follow 祖先
FAIL: 发现 1 个问题；可安全修正候选 0 个。
```

- 该缺口自 2026-10-31 起存在，不是本轮引入（`skills/pyqcu-tag/SKILL.md`
  「本仓库已知既有状态」、`skills/pyqcu-pypi-release/SKILL.md`
  「已知的 tag-chain 既有阻塞（`bug45`）」）。
- **禁止**：force-push、移动/改写 `bug45`/`dev106`、删标签、把“0 个可修正候选”
  说成“已修复”。
- **要求**：把该条作为**显式未决项**交发布负责人签字确认（R9 签署表），
  且**不得**宣称 “tag gate 全绿”。

### C. 四个显式缺口：发布树缺失，工作区只是**未提交候选**

判定以 `git cat-file -e HEAD:<path>` 为准（`128` = 发布树没有该文件）。

| 缺口 | 发布树 `0.1.17` / `HEAD 1f65469e` | 当前工作区 | 裁定 |
|---|---|---|---|
| **driver**（`driver.run_row`） | **缺失**：`git cat-file -e HEAD:pyqcu/test/driver.py` → `128`；`MODE_HOOKS` 声明 `driver:("run_row",)`（`pyqcu/test/common.py:1566-1572`），`_solve_row` 查不到就回退 tier-0 executor（`pyqcu/test/common.py:2683`） | `pyqcu/test/driver.py` 未跟踪（478 行），observation-only，`launcher.mpi_launch=false`（`pyqcu/test/driver.py:378-390`） | **未修复（P0-1/P0-2）**；未提交、未审计、未在远端跑过 |
| **L48 preset** | **缺失**：`LATTICE_PRESETS` 只有 `L24`/`L32`（`git show HEAD:pyqcu/test/common.py`） | `L48=(48,48,48,96)` 已加（`pyqcu/test/common.py:1545-1549`），`--mode list --lattice-preset L48` 接受但**只展开 17 个 tier-0 case（CG 退役后 19−2）**；`--native solve` 额外规划 `kupl/native/solve`、`cuda/native/solve` 共 19 行 | **未修复（P1-1/P1-2）**；L48 生产 solve/mg 行未在 L48 执行过 |
| **两节点 launcher** | **缺失** | **仍缺失**：`pyqcu/test/driver.py:50`、`:380-388` 明确写 no rankfile/`CCS_ALLOC_FILE`；`pyqcu/testing/simple/mpi/solve_case.py`、`_entrypoint.py` 的 `build_mpirun_command` 只拼 `--bind-to/--map-by/-np` | **未修复（P0-3）**；两节点探测 `2152277` 仍 **PENDING** |
| **CSV / Excel parity** | **缺失**：`--csv` 写 30 列 `ROW_FIELDS`（`pyqcu/test/common.py:1586`） | `pyqcu/test/ledger_csv.py`（100 列）、`ledger_excel.py`（21 sheet）、`pyqcu/testing/python/test_ledger_parity.py` 未跟踪；`grep` 确认 `__main__.py`/`common.py` **未引用**；实测 `--mode solve` 的 `rows.csv` 表头仍 30 列 | **未修复（P1-7）**；parity 测试 79 项 PASS 只证明“模块能与参考对齐”，**不等于 CLI 已切换** |

### D. 十轮全过之前不发布

发布动作清单（**本轮全部禁止**）：`python -m build`、`twine upload`、
`git tag -a`、`git push`（origin / gitee）、RTD 触发、PyPI 元数据修改。
R10 只允许 “dry-run + 证据归档”，且必须在 R1–R9 全部 PASS 之后。

---

## R1 — 范围与核心零未授权改动

**目标**：证明本轮改动只落在允许的路径（`pyqcu/test/**` 测试端口、`docs/**`、
`skills/**`、`logs/**`、`pyqcu/testing/**` 非正式测试），核心实现、打包入口与版本
全部零改动；并冻结一个可复现的评审快照。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 已提交范围（`test37..HEAD`） | **PASS** | `git diff --stat test37..HEAD` → 仅 `skills/pyqcu-tag/SKILL.md`（+6） |
| 核心实现零改动 | **PASS** | `git status --short` 与本快照 `git diff --stat` 中无 `pyqcu/torch/**`、`pyqcu/cuda/**`（`.pyx/.pxd`）、`pyqcu/kupl/**`、`pyqcu/lattice/**`、`pyqcu/tools/**`、`cpp/**`、`setup.py`、`MANIFEST.in`、`pyproject.toml` |
| 版本冻结 | **PASS** | 4 处均 `0.1.17`，未改（见文首快照） |
| 测试端口改动范围 | **PASS** | 工作区已改：`pyqcu/test/{__main__,common,native,cuda_qcu,kupl_qcu,selfcheck}.py`、`README.md`、`AGENTS.md`；新增 `driver.py`、`l48.py`、`ledger_csv.py`、`ledger_excel.py`、`ledger_export.py` |
| 工作区可作评审基线 | **FAIL** | `git status --short` 撰写时 **27** 条（15 M + 12 ??），新增本文后 **28** 条（15 M + 13 ??）；2026-11-17 预检时 **47** 条（24 M + 23 ??）；并发批次仍在写（本清单撰写期间 `docs/AGENTS.md`、`docs/ORGANIZATION.md`、`pyqcu/test/AGENTS.md`、`pyqcu/test/README.md`、`pyqcu/test/selfcheck.py`、`skills/pyqcu-docs/**` 又被改） |
| 无 `pyqcu/testing` 正式用例新增 | **UNVERIFIED** | 新增 `pyqcu/testing/python/test_{pyqcu_test_driver,pyqcu_test_syntax_floor,ledger_parity}.py` 是**非正式** pytest 适配器（不进 sdist），需确认不违反 `pyqcu/test/AGENTS.md`「不得在 `pyqcu/testing` 新增正式用例」的语义边界 |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 1) 已提交范围（相对上一发布标签）
git diff --stat test37..HEAD
git log --oneline test37..HEAD

# 2) 核心路径必须零改动（无输出即 PASS）
git status --porcelain -- pyqcu/torch pyqcu/cuda pyqcu/kupl pyqcu/lattice \
  pyqcu/tools cpp setup.py MANIFEST.in pyproject.toml
git diff --stat test37 -- pyqcu/torch pyqcu/cuda pyqcu/kupl pyqcu/lattice \
  pyqcu/tools cpp setup.py MANIFEST.in pyproject.toml

# 3) 版本冻结
grep -n 'version = ' pyproject.toml | head -2
grep -n '__version__' pyqcu/__init__.py
grep -n 'release' docs/en/conf.py docs/zh/conf.py

# 4) 改动清单（评审输入；必须逐条登记到评审记录）
git status --short

# 5) 冻结评审快照（由评审负责人在允许提交的批次执行；本轮不执行）
#    git add <本任务文件>   # 禁止 git add -A / git add .
#    git commit -m "test(pyqcu.test): freeze L48 review snapshot"
```

### 验收标准

- [ ] `test37..HEAD` 与工作区的改动**全部**属于：`pyqcu/test/**`、`pyqcu/testing/**`、
      `docs/**`、`skills/**`、`logs/**`，且逐条在评审记录中有 owner 与理由。
- [ ] `pyqcu/torch`、`pyqcu/cuda`、`pyqcu/kupl`、`pyqcu/lattice`、`pyqcu/tools`、
      `cpp/**`、`setup.py`、`MANIFEST.in`、`pyproject.toml` 的 diff 为空。
- [ ] 版本 4 处仍是 `0.1.17`（本轮冻结；升版本是 R9/R10 之后的独立动作）。
- [ ] 工作区冻结为一个 commit（`git status --porcelain` 为空），并把该 commit
      写入本清单头部，后续轮次全部基于该 commit 复跑。
- [ ] 提交前跑大文件闸门：
      `${CONFIGURE_ROOT:-$HOME/configure}/skills/form/scripts/form-large-files-check.sh --root "$(git rev-parse --show-toplevel)"`

**阻塞**：工作区 47 条改动 + 并发写入。未冻结前 R2–R8 只能记“候选树预检”。

---

## R2 — 与 `pyqcu/testing/kupl/test_kunpeng_all.py` 的测试端口对等

**目标**：证明正式端口 `pyqcu.test` 能**替代**旧 KUPL 全矩阵驱动，而不是只共享词汇：
七模语义、`run_id`/`resume`/`skip-failed`/`run-label`、100 列 CSV、21 sheet Excel、
输入/L48/LIME 覆盖、分片语义都要逐项对齐。

### 现状（逐项）

| 对等项 | 旧驱动 | 正式端口（本快照） | 状态 |
|---|---|---|---|
| 模式名 | `list/prepare/prewarm/solve/merge/run/validate`（`test_kunpeng_all.py:1875-1877`） | 同名七模（`pyqcu/test/common.py` `MODES`） | **PASS** |
| `--mode validate` 语义 | 数值冒烟：强制 `8 8 8 16`、2 rank、`shm=0`（`test_kunpeng_all.py:1749-1778`） | 台账一致性校验（实测空台账 → `problems: 1` / exit 1） | **FAIL**（同名异义；缺 `--mode smoke`，P1-10） |
| `--resume` / `--skip-failed` | 已实现（`:1706-1710`、`:1918-1920`） | 已实现；实测第二次 solve → `resumed=True`、0.01 s | **PASS** |
| `--run-label` | `test_kunpeng_all.py:885` | `run_id` 摘要 + label 后缀 | **PASS** |
| 台账 CSV | 100 列 `CSV_COLUMNS`（`:95`）、`upgrade_csv_schema`/`append_csv`/`merge_rows` | `--csv` 仍写 30 列 `ROW_FIELDS`（`pyqcu/test/common.py:1586`）；实测 `rows.csv` 表头 30 列 | **FAIL**（P1-7） |
| 汇总工作簿 | 21 sheet（`data/kunpeng_all_20261115/kupl_kunpeng_performance_20261115.xlsx`，141,575 B） | `ledger_excel.py` 有 `SHEET_NAMES` 21 项，但未被 CLI 调用 | **FAIL**（P1-7） |
| L48 | `LATTICES = {"L24","L32"}`（`:87`），无 L48 | `--lattice-preset L48` 接受（工作区）；`--native solve` 规划 2 个 native 行 | **UNVERIFIED**（P1-1/P1-2；无 L48 行执行） |
| LIME 输入 | 无 LIME 分支，只走 `h5py`（`:310`） | 端口无 `--gauge-lime`，只能先 `prepare_case.py` 转 HDF5 | **FAIL**（P1-3，见 R5） |
| 分片 | `--limit`/`--shard-index`/`--shard-count`（`:1921-1924`） | 无对应项；只有 `--run-label` + 外部分片 | **UNVERIFIED**（需在 R2 记录“外部分片=可接受”或补实现） |
| 原生行执行 | 驱动自己 `mpirun -np <ranks>`（`:1243`、`:1302`） | `pyqcu.test.driver` 只在**当前进程**调 native（无 mpirun） | **FAIL**（两节点，见 R6） |
| 排他/证据字段 | `--ranks`/`--threads`/`--shm` 真生效 | tier-0 只记录；`--native` 行经 `native.apply_environment` 真生效（实测 `threads_applied=8`） | **PASS（native 行）** / 记录（tier-0） |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 旧驱动的公开契约（只读）
python -B - <<'PY'
import ast
src = open('pyqcu/testing/kupl/test_kunpeng_all.py').read()
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t, 'id', '') in ('CSV_COLUMNS', 'LATTICES'):
                print(t.id, ast.literal_eval(node.value) if t.id == 'LATTICES'
                      else len(ast.literal_eval(node.value)))
PY

# 正式端口台账：CSV 列数必须是 100（当前 30 → FAIL）
S=$(mktemp -d)
python -B -m pyqcu.test --quick --only torch/lattice --mode solve --outdir "$S/run" >/dev/null
head -1 "$S/run/reports/rows.csv" | tr ',' '\n' | wc -l        # 期望 100
python -B -m pyqcu.test --mode merge --outdir "$S/run" >/dev/null
python -B -m pyqcu.test --mode validate --outdir "$S/run"    # 期望 exit 0

# 21 sheet 汇总：只在 parity 测试层验证，CLI 无入口
python -m pytest -q pyqcu/testing/python/test_ledger_parity.py -p no:cacheprovider

# 模式的“同名异义”证据：空台账 validate 退出 1
python -B -m pyqcu.test --mode validate --outdir "$S/empty"   # 期望 exit 1（文档化）

# resume / skip-failed / run-label
python -B -m pyqcu.test --quick --only torch/lattice --mode solve --outdir "$S/run" --resume
python -B -m pyqcu.test --quick --only torch/lattice --mode solve --outdir "$S/run" \
  --resume --skip-failed --run-label cold

# L48 行规划（必须显式区分 tier-0 行与 native 行）
python -B -m pyqcu.test --mode list --lattice-preset L48 --json - | \
  python -c 'import json,sys; d=json.load(sys.stdin); print("tier0", d["count"])'
python -B -m pyqcu.test --mode list --native solve --lattice-preset L48 --ranks 32 --json - | \
  python -c 'import json,sys; d=json.load(sys.stdin); print("all", d["count"],
    [c["tag"] for c in d["cases"] if "native" in c["tag"]])'
```

### 验收标准

- [ ] `--csv`（full mode）表头 = 旧驱动 `CSV_COLUMNS` 的 **100 列同序同名**；
      单元格序列化、CRLF、表头升级、行序与参考一致（以 `test_ledger_parity.py` 全绿为准）。
- [ ] `--mode merge` 可从 `rows/*.json` 重建同一份 100 列 CSV；`--mode validate`
      同时核对 rows/reports/CSV，空台账返回 1。
- [ ] 21 sheet 工作簿可从台账生成（sheet 名与顺序、表头、状态着色、两张汇总图与参考一致）。
- [ ] `--mode validate` 的“同名异义”被消除：新增 `--mode smoke`（数值冒烟）或在
      README/AGENTS/CLI 中显式声明 `validate` 只做台账校验，并给出旧驱动迁移映射。
- [ ] L48 生产 `solve/mg` 行可从 `--mode list` 得到，且与旧驱动同配置行可逐列对照
      （不是性能加速比，`docs/pyqcu_test_l48_remote.md` §7）。
- [ ] `--resume`/`--skip-failed`/`--run-label` 语义与旧驱动一致（ERROR/TIMEOUT 永远重试）。

**阻塞**：P1-7（未接线）、P1-10（validate 语义）、P1-1/P1-2（L48 行）、P1-3（LIME）。

---

## R3 — 数值正确性与独立真残差

**目标**：每个 PASS 都必须由**独立重构**的 `||D x − b|| / ||b||` 支撑，禁止复用求解器
内部递推；并覆盖 c64/c128 门与 L48 生产规模。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 端口内独立残差实现 | **PASS** | `pyqcu/test/native.py:1464-1509` `_true_residual`：`operator(fields["solution"]) - fields["source"]`，`residual_source="independent torch full Wilson-Clover reconstruction"`；MPI 下 `bcast` 到所有 rank |
| 小格点跨后端数值门 | **PASS** | `python -B -m pyqcu.test --quick` → `19 cases, 19 pass, 0 skip, 0 fail`（129.69 s，本机含 CUDA 桥）；gates：`compare 1e-5`、`solve 1e-3`、`native_solve 2e-4`、`gamma 1e-6`（`pyqcu/test/policy.py:271-284`） |
| native KUPL solve 经 driver | **PASS** | `--quick --mode solve --native solve --only kupl/native/solve` → 实测 `iterations=25`、`relative_residual=3.2798664051369997e-07`、`true_residual` 同值、`max_iter=1000` |
| HDF5 驱动的 native solve | **PASS** | 同一命令加 `--gauge-file/--source-file` → `iterations=25`、`relative_residual=4.1788692328736943e-07`、`gauge_origin=hdf5`、`gauge_sha256=5185537473a1a278…`、`source_origin=hdf5` |
| pytest 层回归 | **PASS** | `pytest -q pyqcu/testing/python/test_pyqcu_test_port.py test_pyqcu_test_driver.py test_ledger_parity.py` → **79 passed**（123.23 s） |
| L48 数值 | **UNVERIFIED** | 无任何 L48 `solve`/`mg` 行执行；`docs/pyqcu_test_l48_remote.md` §9.4、`docs/pyqcu_test_audit_risk_20261117.md` P1-2 |
| L48 独立残差可行性（全格点 torch 参考内存） | **UNVERIFIED** | `_true_residual` 会把 global field 聚到 rank 0（`_full_fields`）；L48 c64 gauge ≈ 3.06 GB + source ≈ 1.02 GB + 解/残差，565 GiB 节点是否足够未测（`docs/pyqcu_test_l48_remote.md` §9.4） |
| L48 MG cache 数值等价 | **UNVERIFIED** | 无 L48 cache 产物；历史 L48 cache 是 4-rank DCU 证据（`docs/qcu_quda_solver_dcu_l48_20261029.md`） |
| CUDA 多 rank 数值 | **UNVERIFIED** | `cuda_qcu.py` 多 rank 直接 `SKIP(mpi_single_rank)`（P0-4），`_require_grid` 路径不可达 |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 1) 小格点全矩阵（本机含 CUDA 桥时 19/19；无桥时 cuda/* 必须 SKIP 且整体 exit 1）
python -B -m pyqcu.test --quick --verbose

# 2) c128 门（compare 1e-10 / solve 1e-6 / native_solve 1e-8 / gamma 1e-12）
python -B -m pyqcu.test --quick --dtype complex128 --backend torch,kupl --repeat 1

# 3) 单用例可复现（必须能从报告自证配置）
python -B -m pyqcu.test --only kupl/native/solve --repeat 1 --max-iter 300 \
  --json - | python -c 'import json,sys; d=json.load(sys.stdin);
print(d["counts"], d["options"]["dtype"], d["lattice"])'

# 4) 独立残差必须来自重建，不是求解器递推（代码级复核）
grep -n "residual_source\|_true_residual" pyqcu/test/native.py | head

# 5) L48（当前没有行可跑；先证明“无证据”）
python -B -m pyqcu.test --mode list --native solve --lattice-preset L48 --ranks 32 --json - | \
  python -c 'import json,sys; d=json.load(sys.stdin);
print("rows", d["count"], "native", [c["tag"] for c in d["cases"] if "native" in c["tag"]])'
```

### 验收标准

- [ ] 每个 `PASS` 行都带 `relative_residual`（= 独立重构值）、`iterations`、
      `0 < iterations < max_iter`、输入 `sha256`、`lattice_grid`、`local_lattice`。
- [ ] c64 门：`solve ≤ 1e-3`、`native_solve ≤ 2e-4`；c128 门：`1e-6` / `1e-8`；
      `compare` 门只在 `--atol` 下放宽，残差门不得放宽（`policy.tolerance`）。
- [ ] L48 单节点：至少 1 条 `kupl/native/solve` PASS（c64，`--no-shm`），真残差 ≤ `2e-4`，
      并记录输入 SHA256、迭代数、`wall/elapsed` 分离。
- [ ] L48 两节点：32 rank、`grid 2 2 2 4`、`--no-shm`，至少 1 条 PASS + `djob -ll` `execNodes`
      证据；未通过时只能写 BLOCKED/UNVERIFIED。
- [ ] L48 MG：cache 构建/复用有 `verify_status=0` 证据，且 identity 字段与实际输入一致
      （`gauge_file_sha256` 指向真实消费文件，见 `docs/kupl_hdf5_source_gauge_rootcause_20261115.md` §4.4）。
- [ ] 任何“性能”数字不得由 `--quick` 产生（`pyqcu/test/README.md`「目录约定」）。

---

## R4 — CLI / API / 退出码契约

**目标**：选项拼写、归一化、报告字段、退出码语义可被已发布脚本依赖；Python 3.10
语法底线通过；CLI 契约变更（如 `--native`、L48）有文档。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 端口自检 | **PASS（工作区）** | `python -B -m pyqcu.test --check` → **20 checks, 20 passed**；含 `cli_flags 46`、`cli_usage_codes 21`、`py310_syntax 21 modules`、`package_inclusion 21 packages`、`manifest` |
| 退出码 0/1/2 | **PASS** | 实测：`--warmup 0`=2、`--backend nope`=2、`--quick --dtype complex128`=2、`--only nope/tag`=2、`--lattice-size 8 8 8`=2；`--list`=0；全 SKIP（`--lattice-grid 2 2 2 4` + kupl）=1；空 `--mode validate`=1、空 `--mode merge`=1 |
| `--list` 轻导入 | **PASS** | `selfcheck import_is_stdlib_only` / `cli_list_light`；`--list` 不拉起 torch/native |
| 报告可复现 | **PASS** | `--list --json` 与 `--json` 共用同一 `TestReport.as_dict()`；`options` 含 21 个生效配置键（selfcheck `cli_effective_config`） |
| API（`Options`/`Test`/`Case`/`Result`） | **PASS** | `test_pyqcu_test_port.py` + `test_pyqcu_test_driver.py` + `test_ledger_parity.py` 共 **79 passed** 覆盖；`policy.py` 是唯一决策点 |
| Python 3.10 语法底线 | **PASS（本机 AST）/ UNVERIFIED（远端最终树）** | `py310_syntax` 21 模块 `ast.parse(feature_version=(3,10))` PASS；远端 0.1.17 覆盖树 `--check` 18/18（`logs/v202611171402.txt` §3.2）——**分母不同，不能混读** |
| 新增 CLI 变更文档化 | **PASS（工作区 README）** | `pyqcu/test/README.md:37,59,85,114,205-208,210+` 已写 `--native`、`L48`、HDF5 真输入、driver 行 |
| `--mode validate` 语义冲突 | **FAIL** | 见 R2（P1-10） |
| 远端 `--check` 原始输出归档 | **UNVERIFIED** | P1-8：`--check`、`mpi4py`/`torch` 版本探针原始 stdout 未入库 |

### 命令

```bash
cd /home/zhangxin/PyQCU
python -B -m pyqcu.test --check
python -B -m pyqcu.test --list --json - | python -c 'import json,sys;d=json.load(sys.stdin);print(d["count"],d["lattice"],d["dtype"],sorted(d["options"]))'
python -B -m pyqcu.test --list --verbose >/dev/null

# 退出码矩阵
set +e
for c in "--warmup 0" "--backend nope" "--quick --dtype complex128" \
         "--only nope/tag" "--lattice-size 8 8 8" "--mode validate"; do
  python -B -m pyqcu.test $c >/dev/null 2>&1; echo "[$c] exit=$?"
done

# 3.10 语法底线（必须与远端同语法）
python -B - <<'PY'
import ast, pathlib
for p in sorted(pathlib.Path('pyqcu/test').glob('*.py')):
    ast.parse(p.read_text(), feature_version=(3, 10))
print('py310 ok')
PY

# 远端（需用户内部路由；地址不入库）
# ssh <user-provided> 'cd ~/PyQCU-main && python -m pyqcu.test --check'
# 归档 --check / 版本探针 / prepare_case 计时的原始 stdout 到 logs/v<YYYYMMDD>.txt
```

### 验收标准

- [ ] `--check` 在**最终树**上全绿（记录检查条数与 case 数：当前 20 checks / 19 cases）。
- [ ] 三条退出码路径实测：数值失败 = 1、用法/环境 = 2、全 SKIP = 1（`--allow-all-skipped` 例外）。
- [ ] `--list --json` 与对应 `--json` 的 `options` 完全一致，可复现同一次运行。
- [ ] 新增 `--native`、`--lattice-preset L48`、`--mode` 行为在 README/AGENTS/`--help` 三处一致。
- [ ] Python 3.10 语法/运行在远端最终树复跑（含 `--check` 20/20 或明确记录分母差异），
      原始输出归档。
- [ ] `--mode validate` 语义冲突有明确结论（重命名或文档声明 + 迁移说明）。

---

## R5 — HDF5 / LIME 原子性与损坏恢复

**目标**：任何输入/产物都不可能“半写被当真”；损坏输入必须被拒绝并留下证据；
LIME → canonical HDF5 的转换同样满足原子性与 dtype/形状回读。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 写侧原子性（端口） | **PASS** | `pyqcu/test/io.py:433-449` `_publish`：`temp → write → flush → verify → fsync(file) → os.replace → fsync(dir)`；`save_gauge`/`save_source` 共用（`:1006-1057`） |
| 写后回读校验 | **PASS** | `_verify_written` 在 rename 前重新打开并校验 payload（`io.py:1052`） |
| 读侧闸门 | **PASS** | `describe`（`io.py:828`）校验 dataset/形状/dtype/属性并**真实读首尾元素**；实测 `describe("gauge_ok.h5", role="gauge")` 返回 shape `[3,3,4,8,8,8,16]`、dtype `complex64`、layout、bytes |
| 损坏拒绝 | **PASS** | 手写 header 损坏（前 8 字节清零）→ `ArtifactCorrupt: cannot be opened as HDF5 (file signature not found)`；截断到 50% → `ArtifactCorrupt: truncated file` |
| `prepare` 失败语义 | **PASS** | `--mode prepare --gauge-file <bad>.h5` → exit **1**，`reports/prepare.json` 记 `gauge file rejected: io.is_readable: …`；缺失文件 exit 1；好文件 exit 0 |
| `prewarm` 哈希 | **PASS** | `--mode prewarm --gauge-file gauge_ok.h5` → exit 0，`prewarm.json` 有 `sha256` |
| 端口 LIME 入口 | **FAIL** | 无 `--gauge-lime`；LIME 必须先经 `pyqcu/testing/simple/prepare_case.py` 转 HDF5（`docs/pyqcu_test_l48_remote.md` §9.5） |
| 生成器写侧原子性（非正式树） | **FAIL（未修）** | `prepare_case.py` 生成器只有 `temp + os.replace`，**无 fsync、无 shape/dtype 回读**（`docs/kupl_hdf5_source_gauge_rootcause_20261115.md` §4.1；该文档明写“仍是待补项”） |
| 坏文件隔离（`.corrupt-*`） | **FAIL** | 脚本内无 `.corrupt-*` 逻辑（同文档 §4.1“脚本内 `grep -c corrupt` = 0”）；2026-11-15 的改名是手工动作 |
| 同名 dtype 误标 | **FAIL（未修）** | L48 `gauge_48x48x48x96_c64.h5` 实为 `complex128`（6,115,301,376 B）；c64 cast + `io.describe` 回读未完成（P1-5） |
| `real` gauge 缺失静默回退 identity | **FAIL** | `docs/kupl_hdf5_source_gauge_rootcause_20261115.md` §4.1 末条：会静默回退但 CSV 仍记 `gauge_kind=real` |

### 命令

```bash
cd /home/zhangxin/PyQCU
S=$(mktemp -d)
python -B - "$S" <<'PY'
import sys, numpy as np
from pyqcu.test import io
lat = (8, 8, 8, 16)
g = np.zeros((3, 3, 4, *lat), np.complex64)
for d in range(4):
    g[0, 0, d] = g[1, 1, d] = g[2, 2, d] = 1
io.save_gauge(f"{sys.argv[1]}/gauge_ok.h5", g)
raw = open(f"{sys.argv[1]}/gauge_ok.h5", "rb").read()
bad = bytearray(raw); bad[:8] = b"\0" * 8
open(f"{sys.argv[1]}/gauge_bad.h5", "wb").write(bytes(bad))
open(f"{sys.argv[1]}/gauge_trunc.h5", "wb").write(raw[: len(raw) // 2])
for name in ("gauge_ok.h5", "gauge_bad.h5", "gauge_trunc.h5"):
    try:
        print(name, "OK", io.describe(f"{sys.argv[1]}/{name}", role="gauge")["dtype"])
    except Exception as e:
        print(name, "REJECT", type(e).__name__, str(e)[:70])
PY

python -B -m pyqcu.test --quick --only torch/lattice --mode prepare --outdir "$S/p_ok"  --gauge-file "$S/gauge_ok.h5";    echo "good=$?"
python -B -m pyqcu.test --quick --only torch/lattice --mode prepare --outdir "$S/p_bad" --gauge-file "$S/gauge_bad.h5";  echo "bad=$?"
python -B -m pyqcu.test --quick --only torch/lattice --mode prewarm --outdir "$S/p_ok"  --gauge-file "$S/gauge_ok.h5";    echo "prewarm=$?"

# 原子写：不得留下 .tmp
find "$S" -name '*.tmp' -o -name '.*.tmp' | head

# L48 dtype 回读（远端，需 c64 cast 完成）
# python3 -c 'import h5py;h=h5py.File("<out>/gauge_48x48x48x96_c64.h5","r");print(h["data"].dtype,h["data"].shape)'
# python3 -c 'from pyqcu.test import io;print(io.describe("<out>/gauge_48x48x48x96_c64.h5", role="gauge"))'

# 源码层“只读输入”复核（不改文件）
grep -rn "open(.*\"r\"\|h5py.File(.*\"r\"" pyqcu/testing/simple/_lime.py pyqcu/testing/simple/prepare_case.py
grep -rn "fsync" pyqcu/testing/simple/prepare_case.py pyqcu/testing/simple/_common.py
grep -rn "corrupt" pyqcu/testing/simple/*.py | head
```

### 验收标准

- [ ] `io.save_*` 之后目录中无 `.tmp` 残留；目标文件可被 `describe` 全字段回读。
- [ ] 三种坏文件（header 损坏、截断、0 字节）全部被拒绝，错误类型稳定
      （`ArtifactCorrupt`/`ArtifactMissing`），不得 traceback 或静默复用。
- [ ] `--mode prepare` 对坏输入 exit 1 且 `prepare.json` 有结构化原因；`prewarm` 记录 sha256。
- [ ] 生成器（`prepare_case.py`）补齐 `temp → flush → verify → fsync(file) → rename →
      fsync(dir)`，并对 shape/dtype/lattice/role 回读；坏文件改名保留（`.corrupt-<ts>`）
      而不是删除（`docs/kupl_hdf5_source_gauge_rootcause_20261115.md` §4.1/§4.3）。
- [ ] L48 c64 gauge 有 `dataset.dtype == complex64`、shape `[3,3,4,48,48,48,96]`、
      `lattice == [48,48,48,96]` 的 `io.describe` 回读证据；在此之前只能写 UNVERIFIED。
- [ ] `real` gauge 缺失/不可读必须 FAIL 或显式 opt-in + `gauge_origin`，不得静默 identity。

---

## R6 — MPI / 两节点 / shm / 线程 / cache

**目标**：证明“一个 Python 进程 = 一个 MPI rank = 一张设备”的生产拓扑在 KUPL 上可跨节点
成立；shm、线程、cache 的“记录 vs 生效”边界无歧义。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 线程真正下传（native 行） | **PASS** | `pyqcu/test/native.py:495-530` `apply_environment` 设 `OMP_NUM_THREADS`/`KUPL_OMP_THREADS`/`OMP_DYNAMIC`/`KUPL_SHM_HALO`；实测行 metrics `threads_applied=8`、`shm_applied=false` |
| tier-0 threads 边界 | **PASS（文档化）** | 仅 torch runner `torch.set_num_threads`；`kupl_qcu.py`/`cuda_qcu.py` 不读（`pyqcu/test/README.md:114`） |
| shm 可用性判定 | **PASS（代码）** | `native.py:418-444`：非 kupl / 单 rank / 非 Linux / `OMPI_COMM_WORLD_LOCAL_SIZE < world` → 不可用；`resolve_shm` 报 `SKIP(shm_unavailable)`（`:447-467`） |
| shm 跨节点实测 | **UNVERIFIED** | 没有 2 rank 单节点 shm 对照，也没有跨节点 `SKIP(shm_unavailable)` 实测记录 |
| `mpi_grid` 不一致 SKIP | **PASS** | 实测 `--lattice-grid 2 2 2 4` 单 rank → `SKIP mpi_grid`，`exit 1`、`all_skipped True` |
| 跨节点 launcher / rankfile | **FAIL** | `pyqcu/test/driver.py:50`、`:380-388` 明写未实现；`pyqcu/testing/simple/mpi/*` 的 `build_mpirun_command` 无 `--hostfile/--rankfile`（P0-3） |
| 两节点作业 | **UNVERIFIED / PENDING** | `djob -l 2152277` → PENDING，`No available node can fit ... execution nodes preference`；无 `execNodes`（P1-6） |
| 计算节点免密 | **FAIL** | `login02 → cn*` 报 `Permission denied (publickey)`（`docs/pyqcu_test_l48_remote.md` §2.4.5） |
| MPI 失败语义（abort/挂起） | **UNVERIFIED（风险）** | KUPL C ABI 把异常转 `abort_with_message`（`MPI_Abort` + `std::abort`）；`pyqcu/test` 无 abort/超时协调（P0-5） |
| CUDA 多 rank | **FAIL** | 多 rank 直接 `SKIP(mpi_single_rank)`（P0-4），`_require_grid` 不可达 |
| `_SET_INDEX_` 游标 | **UNVERIFIED（风险）** | 控制快照排除 `_SET_INDEX_`（P0-6） |
| cache 探测 | **PASS** | `--mode prepare --cache-dir` 写 `caches/index.json`：`rows=1`、`cache_require={dtype,lattice,mpi_size,seed}`、`identity_fields=[lattice,dtype,dof,mg_grid,nv_iters,seed]`、`default_cache_dir=~/.cache/pyqcu/mg`；`cache.cache_status` 默认校验 `file_sha256` |
| L48 cache 构建/复用 | **UNVERIFIED** | 端口只探测不重建；native builder 在非正式树；L48 cache 逻辑 ≈ 17.44 GiB 预算（`docs/qcu_quda_solver_dcu_l48_20261029.md`） |
| L48 几何可分域 | **PASS（推导）** | `docs/pyqcu_test_l48_remote.md` §5.1：32 rank `2 2 2 4` → local `24×24×24×24`；64 rank `2 2 2 8` → `24×24×24×12`；`1×1×1×R` 的 t-finest 在 32/64 rank 下不可用 |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 1) 线程/shm 记录与生效（单 rank native 行）
S=$(mktemp -d)
python -B -m pyqcu.test --quick --mode solve --native solve --only kupl/native/solve \
  --threads 18 --no-shm --max-iter 200 --outdir "$S/run" >/dev/null
python - "$S" <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/run/rows/*.json"):
    m = json.load(open(p))["metrics"]
    print({k: m.get(k) for k in ("threads_requested", "threads_applied",
                                 "shm_requested", "shm_applied", "launcher")})
PY

# 2) 单节点 2 rank：mpi_grid / shm 行为（需要 mpiexec；本机若无则记 UNVERIFIED）
mpirun -np 2 --bind-to none python -B -m pyqcu.test --quick --backend kupl \
  --only kupl/dslash/wilson --lattice-grid 1 1 1 2 --shm --json - || true

# 3) launcher 缺口复核（无输出=缺失）
grep -rn "hostfile\|rankfile\|CCS_ALLOC_FILE" pyqcu/test pyqcu/testing/simple/mpi

# 4) cache 探测
python -B -m pyqcu.test --quick --only torch/lattice --mode prepare \
  --outdir "$S/run2" --cache-dir "$S/cache" >/dev/null
python -c "import json;d=json.load(open('$S/cache/index.json'));print(d['identity_fields'], d['cache_require'])"

# 5) 远端两节点（需用户内部路由 + 计算节点免密）
# dsub -q q_hpcapp -nn 2 -N 32 -rpn 16 -s job.sh
# djob -ll <jobid>          # 必须 execNodes 恰好 2 个节点才可写两节点证据
# mpirun -np 32 --rankfile "$PWD/$CCS_JOB_ID-rankfile" --map-by ppr:16:node \
#   --bind-to none python3 -m pyqcu.test --mode solve --native solve \
#   --lattice-preset L48 --ranks 32 --no-shm --outdir runs/l48_2n
```

### 验收标准

- [ ] rank 数 = `prod(--lattice-grid)`，不一致时原生行 `SKIP(mpi_grid)` 且整体不得假绿。
- [ ] `--threads` 在 native 行 `threads_applied` 等于请求值，`OMP_NUM_THREADS`/
      `KUPL_OMP_THREADS` 在首次原生调用前生效；tier-0 行只记录。
- [ ] `--shm` 单节点 2 rank 有对照值；跨节点必须 `SKIP(shm_unavailable)`，不得静默
      回退 `shm=0` 仍报 `shm=true`。
- [ ] 跨节点 launcher：解析 `CCS_ALLOC_FILE` → rankfile，`-np`/`--map-by`/`--bind-to`
      与节点数写进 `launcher` 证据字段；无分配文件时显式报错，不猜节点、不退单节点。
- [ ] 两节点作业 `djob -ll` 的 `execNodes` 恰好两个；否则只能 BLOCKED/UNVERIFIED。
- [ ] MPI 失败可分级：至少一条“rank 内原生失败 → ERROR/TIMEOUT 行 + 有界退出”的实测，
      不允许整作业挂起（P0-5）。
- [ ] CUDA 多 rank 网格/设备绑定有结论（修复或明确“仅单 rank 支持”，P0-4）。
- [ ] L48 MG cache 可构建/复用，`verify_status=0`，identity 与实际输入一致。
- [ ] `_SET_INDEX_` 游标在连续 case 之间被断言（修复或记录为已知限制，P0-6）。

---

## R7 — 打包与 sdist

**目标**：sdist 只带该带的：`pyqcu.test` 全量、`pyqcu.testing` 为零、无 `.so/.o/.dylib`、
KUPL+CUDA 源码齐全、无构建机路径、中文 README 在列；且**与发布 commit 一一对应**。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| `MANIFEST.in` 契约 | **PASS** | `recursive-include pyqcu *.py *.pyi *.pyx *.pxd`；`prune pyqcu/testing`；`global-exclude *.so *.so.* *.o *.dylib`；`include README.zh-CN.md` |
| 自检包含性 | **PASS** | `--check`：`package_inclusion`（21 packages，含 `pyqcu.test`，不含 `pyqcu.testing`）、`manifest`（不 prune `pyqcu/test`） |
| 当前已发布 sdist 内容 | **PASS（0.1.17 基线）** | `dist/pyqcu-0.1.17.tar.gz`：273 成员、104 `.py`、`pyqcu/test/*.py` **16**、`pyqcu/testing/` **0**、`.so` **0**、`README.zh-CN.md` 存在 |
| 工作树模块数 vs sdist | **FAIL** | 工作树 `pyqcu/test/*.py` = **21**（新增 `driver.py`、`l48.py`、`ledger_csv.py`、`ledger_excel.py`、`ledger_export.py`，均未跟踪）→ 0.1.17 sdist 落后 5 个模块；`release_check` 的 “sdist ships pyqcu.test” 一一对应判据在当前 `dist/` 上会 FAIL |
| `dist/` 清理 | **FAIL** | `dist/` 仍有 `pyqcu-0.1.17.tar.gz`（2026-11-16 21:22）；发布流程要求 `rm -rf dist` 后重建，否则 `release_check` 读到旧文件 |
| 下一版 sdist | **UNVERIFIED** | 未 build（本清单禁止执行发布）；且新模块必须先进 commit 再构建 |
| 无 `.dylib`/macOS 残留 | **PASS** | `MANIFEST.in` `global-exclude *.dylib`；根 `AGENTS.md` 明确 macOS 不支持、不得让 `*.dylib` 入包 |
| `SOURCES.txt` 无 `/home/` | **PASS（0.1.17 基线）** | 0.1.17 sdist 由相对路径 Extension 源构建；重建后需复验 |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 1) 当前 dist 基线（只读）
tar tzf dist/pyqcu-0.1.17.tar.gz > /tmp/sd017.txt
wc -l < /tmp/sd017.txt
grep -c '\.py$' /tmp/sd017.txt
grep -cE '/pyqcu/test/.*\.py$' /tmp/sd017.txt        # 0.1.17 = 16
grep -c 'pyqcu/testing/' /tmp/sd017.txt             # 必须 0
grep -c '\.so' /tmp/sd017.txt                       # 必须 0
grep -c 'README.zh-CN.md' /tmp/sd017.txt            # 必须 >= 1
tar xzOf dist/pyqcu-0.1.17.tar.gz pyqcu-0.1.17/pyqcu.egg-info/SOURCES.txt | grep -c '/home/'

# 2) 工作树模块数（必须与 sdist 一一对应）
ls pyqcu/test/*.py | wc -l                          # 当前 21

# 3) 下一版重建（R1–R9 PASS 后才可执行；必须在仓库外调用 build）
#    rm -rf dist
#    cd <非仓库目录> && python -m build --sdist --outdir /home/zhangxin/PyQCU/dist /home/zhangxin/PyQCU
#
# 4) 重建后逐项复验（同 §1，把 0.1.17 换成 0.1.18）
#    tar tzf dist/pyqcu-0.1.18.tar.gz | grep -cE '/pyqcu/test/.*\.py$'   # 必须 = 21
#    PYQCU_SKIP_NATIVE_BUILD=1 快速验“打包可装”
```

### 验收标准

- [ ] sdist 的 `/pyqcu/test/*.py` 与发布 commit 的工作树**同名文件一一对应**（当前应为 21）。
- [ ] `pyqcu/testing/` = 0、`.so/.o/.dylib` = 0、`SOURCES.txt` 无 `/home/`；
      KUPL `src`、CUDA `src` 计数非 0；`README.zh-CN.md` 在包内。
- [ ] `pyqcu/test/README.md`、`AGENTS.md` 不进 sdist（`MANIFEST.in` 只收 `*.py`，
      `global-exclude AGENTS.md`）。
- [ ] `dist/` 在构建前清空；构建命令在**仓库外**执行（避免本地 `build/` 遮蔽 `import build`）。
- [ ] 构建产物 sha256 记录到发布台账；不得复用 0.1.17 的产物。

---

## R8 — docs / skills / AGENTS / form

**目标**：文档与技能索引自洽；AGENTS 的硬性约定不被本轮改动破坏；form 审计与
docs/skills 审计的结论被如实记录（含既有告警）。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| form 审计 | **PASS** | `bash skills/pyqcu-form/form-audit.sh` → `PASS: 36 个技能目录与索引一致，豁免 18 个既有资产 finding` |
| docs/skills 审计 | **PASS（本批修复后）** | `python3 -B skills/pyqcu-docs/scripts/audit-docs-skills.py` → `PASS docs=58 skills=36`；修复前 `FAIL docs=58 skills=36 errors=11`：3×DOCS-COUNT（57→58）、`pyqcu_test_l48_two_node_matrix.md` 未登记（DOCS-INDEX/MAP/GUIDANCE + 3×DOCS-UNIQUE）、`[SKILL-SIZE]` SKILL.md 530 行、`[AGENTS-SIZE]` AGENTS.md 15 行（已拆出 `references/strict-mg-fast-gates.md`，AGENTS.md 压缩为 12 行） |
| 新增文档登记 | **PASS（本批）** | `docs/pyqcu_test_l48_remote.md`、`docs/pyqcu_test_audit_risk_20261117.md`、`docs/pyqcu_test_l48_review.md`、`docs/pyqcu_test_l48_two_node_matrix.md` 均已登记（docs 58 = 实际 58） |
| 本文追加后的登记 | **PASS（本批修复）** | 本文与 `pyqcu_test_l48_two_node_matrix.md` 均已登记进 `docs/AGENTS.md`、`project-docs-map.md`、`extracted-guidance.md`；`DOCS-COUNT`/`DOCS-UNIQUE` 无残留（预检时曾 8 条） |
| tier-0 计数漂移 | **FIXED（本批）** | 根 `AGENTS.md:29`、`:64`、`:88`、`pyqcu/AGENTS.md`、`pyqcu/testing/AGENTS.md`、`skills/AGENTS.md:93` 与 `extracted-guidance.md` 的 8 处旧 tier-0 计数 21 已改为 19；6 处旧 CUDA 计数 5 已改为 3；实测 `--list` = **19**（11 torch / 5 kupl / 3 cuda） |
| 技能参考 | **PASS（工作区）** | `skills/pyqcu-testing/references/l48-remote.md` 已加操作者检查单（`logs/v202611171416.txt` §4） |
| `git diff --check` | **PASS** | 无输出、退出 0 |
| 脚本语法 | **UNVERIFIED** | 新增 `submit_pyqcu_test.sh` 尚未存在；未来新增脚本需 `bash -n` |

### 命令

```bash
cd /home/zhangxin/PyQCU
bash skills/pyqcu-form/form-audit.sh
python3 -B skills/pyqcu-docs/scripts/audit-docs-skills.py
git diff --check
bash -n build-nv.sh build-dcu.sh install.sh env.sh
grep -rn "21 个 tier 0" AGENTS.md pyqcu/AGENTS.md skills/AGENTS.md   # 现在无输出（本批已改 19）
python -B -m pyqcu.test --list --json - | python -c 'import json,sys;print(json.load(sys.stdin)["count"])'

# 相对链接检查（本文与相关 docs）
python3 - <<'PY'
import pathlib, re
for p in pathlib.Path('docs').glob('*.md'):
    for m in re.finditer(r'\]\((?!https?://)([^)#]+)', p.read_text()):
        target = (p.parent / m.group(1)).resolve()
        if not target.exists():
            print('MISSING', p, m.group(1))
PY
```

### 验收标准

- [x] `form-audit.sh` PASS；`audit-docs-skills.py` 的每条 finding 都在评审记录里
      分类为“既有告警”或“本轮新增并修复”；不得整体禁用检查。（本批：`PASS docs=58 skills=36`）
- [x] 本文 `docs/pyqcu_test_l48_review.md` 登记进 `docs/AGENTS.md`、
      `skills/pyqcu-docs/references/project-docs-map.md`、
      `skills/pyqcu-docs/references/extracted-guidance.md`，docs 计数与实际一致。
- [x] 根 `AGENTS.md`、`pyqcu/AGENTS.md`、`skills/AGENTS.md` 的旧 tier-0 计数 21 改为 19
      （或明确标注“端口扩容后计数”），与 `--list` 一致。（本批共 8 处；另 6 处旧 CUDA 计数 5 改为 3）
- [ ] `pyqcu/test/AGENTS.md` 的硬性约定复核：无 `test_*.py`、无 `pytest` 导入、
      stdlib-only 顶层、CLI 契约只在 `policy.py` 定义、`--quick` 只锁五件事。
- [ ] 文档不得出现“L48 已 PASS”“两节点可跑”“CSV/Excel 已切换”等未证实结论
      （`docs/pyqcu_test_audit_risk_20261117.md` §4）。
- [ ] `git diff --check`、`bash -n` 全部通过；新增脚本有 `--dry-run`。

---

## R9 — 独立只读复审 + 版本 / 标签链

**目标**：由**非本轮实现者**在只读快照上逐条复核 R1–R8 的结论与证据；确认版本四处
一致、下一版号未被占用、标签链的已知例外被显式接受；产出可签字的评审结论。

### 现状

| 检查项 | 状态 | 证据 |
|---|---|---|
| 评审快照 | **FAIL** | 工作区 47 条未提交项；无冻结 commit |
| 版本四处一致 | **PASS（0.1.17 冻结）** | `pyproject.toml:11`、`pyqcu/__init__.py:3`、`docs/en/conf.py:13`、`docs/zh/conf.py:13` 均为 `0.1.17` |
| 下一版未发布 | **PASS（需复查）** | PyPI 最新 = `0.1.17`；`0.1.18` 尚不存在（发布前用 `curl -o /dev/null -w '%{http_code}' https://pypi.org/pypi/pyqcu/0.1.18/json` 复验，期望 404） |
| 同版本不可覆盖 | **PASS（红线已记录）** | 见 §0.1 A；PyPI 0.1.17 sha256 与本地一致 |
| 标签链自检 | **FAIL（唯一历史 `[BLOCK] bug45`，接受为例外）** | `bash skills/pyqcu-tag/tag-chain.sh --check` → 1 个 `[BLOCK]`、0 个可安全修正候选、exit 1；不得重写 |
| 标签命名/消息规范 | **PASS（模板）** | 下一标签建议 `test38`：`follow test37, 1. …; 2. …; [<agent-name>].`（`skills/pyqcu-tag/SKILL.md`） |
| 双远端同步 | **UNVERIFIED（不执行）** | `origin` = gitlab.itp.ac.cn、`gitee` = gitee.com；R10 之外不得 push |
| 审计 P0/P1 关闭证据 | **FAIL（全部未修复）** | `docs/pyqcu_test_audit_risk_20261117.md` P0-1…P0-6、P1-1…P1-10 状态均为 NOT FIXED / TODO / PENDING |

### 命令

```bash
cd /home/zhangxin/PyQCU
# 1) 快照与差异
git rev-parse HEAD; git describe --tags --always; git status --porcelain
git diff --stat test37..HEAD

# 2) 版本与旧版本残留
grep -n 'version = ' pyproject.toml | head -2
grep -n '__version__' pyqcu/__init__.py
grep -n 'release' docs/en/conf.py docs/zh/conf.py
grep -rn "0\.1\.17" --include=*.py --include=*.toml --include=*.md --include=*.txt \
  --include=*.in . | grep -v '^./logs/' | grep -v '^./data/' | grep -v '^./build/' | head -20

# 3) 标签链（记录 exit 1 与唯一 BLOCK；不得 --repair --apply）
bash skills/pyqcu-tag/tag-chain.sh --check || true
bash skills/pyqcu-tag/tag-chain.sh --repair --dry-run --remote origin || true

# 4) 下一版号未被占用
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/pyqcu/0.1.18/json   # 期望 404
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/simple/pyqcu/

# 5) 审计条目逐条复核（只读；每条要给“仍失败”的命令证据）
git cat-file -e HEAD:pyqcu/test/driver.py; echo "driver=$?"
git cat-file -e HEAD:pyqcu/test/ledger_csv.py; echo "ledger_csv=$?"
git show HEAD:pyqcu/test/common.py | grep -A3 '^LATTICE_PRESETS'
grep -rn "hostfile\|rankfile\|CCS_ALLOC_FILE" pyqcu/test pyqcu/testing/simple/mpi
```

### 验收标准

- [ ] 评审在一个**已提交快照**上进行；评审人 ≠ 实现者；只读（无 `git commit`、无文件修改）。
- [ ] R1–R8 每条结论有“命令 + 输出摘要 + 证据路径”，FAIL/UNVERIFIED 不得改写成 PASS。
- [ ] 版本四处一致；旧版本 grep 只剩 `cpp/kupl/qcu/CMakeLists.txt` 的 soname `0.1.0`
      （`skills/pyqcu-pypi-release/SKILL.md`「soname 不是发行版本」）。
- [ ] 下一版号为 `0.1.18`，PyPI 上不存在；`0.1.17` 不可覆盖（红线 A 已签字）。
- [ ] 标签链：唯一 `[BLOCK] bug45` 由发布负责人**书面接受**为历史例外；
      `bug45`/`dev106` 未被改写；结论写作“tag gate 有 1 条已知历史阻塞”，
      不得写“全绿”。
- [ ] 下一标签为 `test38`，`follow test37`，消息按 tag 技能模板（每项以 `;` 结尾）。
- [ ] P0/P1 的每一条要么有“修复 commit + 复核命令 + 结果”，要么保留 NOT FIXED 并
      在发布决定中显式豁免（默认不豁免）。

---

## R10 — 发布 dry-run（不发布）

**目标**：在**新版本 + 干净提交树**上完整走一遍发布命令，证明除“上传”外的每一步
可复现；所有产物与门禁输出归档到 `logs/v<YYYYMMDD>.txt`。

> 前置：R1–R9 全部 PASS，且版本已升到 `0.1.18`（4 处 + 两语 `install.md` 断言 +
> 两语 `overview.md`/`performance.md` 散文）。**本轮仍不上传、不打 tag、不 push。**

### 现状

**UNVERIFIED（按约束未执行）**。可复用的既有基线：0.1.17 sdist 273 成员 / 104 `.py` /
16 test 模块 / 0 testing / 0 `.so`；0.1.17 在 PyPI 的 sha256 与本地一致。

### 命令（dry-run 序列）

```bash
cd /home/zhangxin/PyQCU
# 0) 前置检查
git status --porcelain            # 必须为空
git describe --tags --always      # 记录基线
grep -n 'version = ' pyproject.toml | head -2

# 1) 清空旧产物并从“仓库外”构建（避免本地 build/ 遮蔽 import build）
rm -rf dist
cd /tmp && python -m build --sdist --outdir /home/zhangxin/PyQCU/dist /home/zhangxin/PyQCU

# 2) sdist 逐项核对（成员数、.py、pyqcu/test 一一对应 19、testing=0、.so=0、
#    README.zh-CN.md、SOURCES.txt 无 /home/）
tar tzf /home/zhangxin/PyQCU/dist/pyqcu-0.1.18.tar.gz > /tmp/sd018.txt
wc -l < /tmp/sd018.txt
grep -c '\.py$' /tmp/sd018.txt
grep -cE '/pyqcu/test/.*\.py$' /tmp/sd018.txt
grep -c 'pyqcu/testing/' /tmp/sd018.txt
grep -c '\.so' /tmp/sd018.txt
grep -c 'README.zh-CN.md' /tmp/sd018.txt
tar xzOf /home/zhangxin/PyQCU/dist/pyqcu-0.1.18.tar.gz pyqcu-0.1.18/pyqcu.egg-info/SOURCES.txt | grep -c '/home/'
sha256sum /home/zhangxin/PyQCU/dist/pyqcu-0.1.18.tar.gz

# 3) 发布门（13/13）与 twine
cd /home/zhangxin/PyQCU
python -m pyqcu.testing.release_check --offline
python -m twine check dist/pyqcu-0.1.18.tar.gz

# 4) 干净环境（venv 放 data/pypi/，不污染系统）
python -m venv data/pypi/venv-018-dryrun
data/pypi/venv-018-dryrun/bin/pip install --no-cache-dir --no-index \
  --find-links dist "pyqcu==0.1.18"    # 或 PYQCU_SKIP_NATIVE_BUILD=1 快速验打包
cd /tmp && /home/zhangxin/PyQCU/data/pypi/venv-018-dryrun/bin/python -m pyqcu
data/pypi/venv-018-dryrun/bin/python -m pip check
data/pypi/venv-018-dryrun/bin/python -m pyqcu --selftest

# 5) 只读复核：PyPI 上仍无 0.1.18（dry-run 不得上传）
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/pyqcu/0.1.18/json   # 期望 404

# 6) 台账（logs/v<YYYYMMDD>.txt）：产物大小/sha256、门禁 13/13、sdist 计数、
#    数值（L48 单/两节点）、外部阻塞（bug45）、dry-run 明确“未上传”
```

### 验收标准

- [ ] `release_check --offline` **13/13**（version、sdist exists、sdist no testing、
      sdist ships pyqcu.test、sdist no binaries、sdist native sources、sdist no home paths、
      sdist Chinese README、twine check、bilingual blocks、docs examples、sphinx en、sphinx zh）。
- [ ] `twine check dist/pyqcu-0.1.18.tar.gz` → `PASSED`。
- [ ] sdist 的 `pyqcu/test/*.py` 与工作树一一对应（当前 19），`pyqcu/testing/` = 0。
- [ ] 干净 venv：`python -m pyqcu` 的 `location` 落在 `site-packages`、版本为 `0.1.18`、
      `pip check` 干净、`--selftest` 结论符合后端可用性（缺原生为 SKIP 不算失败）。
- [ ] `docs_examples`、`bilingual blocks`、Sphinx en/zh 全绿（两语同步改，不得只改一边）。
- [ ] 台账写入 `logs/v<YYYYMMDD>.txt`，含“未上传/未打 tag/未 push”的显式声明。
- [ ] 上传动作**不执行**；执行需另行授权，且必须满足：R1–R9 PASS + 十轮签署 +
      `PYPI_ZHANGXIN8069_API_KEY` 从环境读取（不回显、不入库）。

---

## 附录 A — 证据索引（本快照实际跑过的命令）

| # | 命令 | 结果 |
|---:|---|---|
| 1 | `git rev-parse HEAD` / `git describe --tags --always` | `1f65469e…` / `test37-1-g1f65469e` |
| 2 | `git diff --stat test37..HEAD` | 仅 `skills/pyqcu-tag/SKILL.md`（+6） |
| 3 | `git status --short` | 撰写时 27 条、新增本文后 28 条（15 M + 13 ??）；2026-11-17 预检时 47 条（24 M + 23 ??），含并发批次改动 |
| 4 | `python -B -m pyqcu.test --check` | **20 checks, 20 passed**，overall PASS（19 cases） |
| 5 | `python -B -m pyqcu.test --list --json -` | `count 19`、`[8,8,8,16]`、`c64` |
| 6 | `python -B -m pyqcu.test --quick` | **19 pass / 0 skip / 0 fail**（129.69 s） |
| 7 | `python -B -m pyqcu.test --mode list --lattice-preset L48 --json -` | 17 个 tier-0 行（CG 退役；无 L48 生产行） |
| 8 | `python -B -m pyqcu.test --mode list --native solve --lattice-preset L48 --ranks 32 --json -` | 19 行（+ `kupl/native/solve`、`cuda/native/solve`） |
| 9 | native KUPL solve（driver，无文件） | `PASS`，`iterations=25`，`relative_residual=3.2798664051369997e-07` |
| 10 | native KUPL solve（driver + HDF5 输入） | `PASS`，`iterations=25`，`4.1788692328736943e-07`，`gauge_origin=hdf5` |
| 11 | `pytest -q test_pyqcu_test_port.py test_pyqcu_test_driver.py test_ledger_parity.py` | **79 passed**（123.23 s） |
| 12 | full-mode solve 的 `rows.csv` 表头 | **30** 列（目标 100） |
| 13 | `--mode prepare --gauge-file`（好/坏/缺失） | exit `0` / `1` / `1`；坏文件 `gauge file rejected: io.is_readable …` |
| 14 | `io.describe` + 手写损坏/截断 | 正常回读；`ArtifactCorrupt`（file signature / truncated） |
| 15 | `--mode validate` / `--mode merge`（空台账） | 均 exit `1` |
| 16 | `--resume` 二次 solve | `resumed=True`，0.01 s |
| 17 | `--lattice-grid 2 2 2 4`（单 rank，kupl） | `SKIP mpi_grid`，exit 1，`all_skipped True` |
| 18 | `--mode prepare --cache-dir` | `caches/index.json`：`rows=1`、`identity_fields=[lattice,dtype,dof,mg_grid,nv_iters,seed]` |
| 19 | `bash skills/pyqcu-tag/tag-chain.sh --check` | 1 × `[BLOCK] bug45`，0 可修正候选，exit 1 |
| 20 | `bash skills/pyqcu-form/form-audit.sh` | PASS（36 技能目录一致，18 条豁免） |
| 21 | `python3 -B skills/pyqcu-docs/scripts/audit-docs-skills.py` | 撰写时 FAIL `docs=55 skills=36 errors=2`；预检时 FAIL `docs=58 skills=36 errors=11`；本批修复后 **PASS `docs=58 skills=36`** |
| 22 | `curl https://pypi.org/pypi/pyqcu/json` / `…/0.1.17/json` | latest `0.1.17`，912,755 B，sha256 `82d1af5f…b19dca`（= 本地 dist） |
| 23 | `sha256sum dist/pyqcu-0.1.17.tar.gz` | `82d1af5f8f0ac8b2c5f38b4a3c9d5cff32fd114cb6ebaf196b9e631b29b19dca` |
| 24 | `tar tzf dist/pyqcu-0.1.17.tar.gz` | 273 成员 / 104 `.py` / 16 test 模块 / 0 testing / 0 `.so` |
| 25 | `git cat-file -e HEAD:pyqcu/test/{driver,ledger_csv}.py` | `128`（发布树缺失） |

## 附录 B — 评审签署表

| 轮次 | 评审人 | 结论（PASS/FAIL/UNVERIFIED） | 证据链接 | 日期 |
|---|---|---|---|---|
| R1 范围/核心 |  |  |  |  |
| R2 端口对等 |  |  |  |  |
| R3 数值/独立残差 |  |  |  |  |
| R4 CLI/API/退出码 |  |  |  |  |
| R5 HDF5/LIME |  |  |  |  |
| R6 MPI/两节点/shm/线程/cache |  |  |  |  |
| R7 打包/sdist |  |  |  |  |
| R8 docs/skills/AGENTS/form |  |  |  |  |
| R9 只读复审 + 版本/标签链 |  |  |  |  |
| R10 发布 dry-run |  |  |  |  |

发布负责人对以下两条**历史例外**的书面确认（缺一不可）：

- [ ] 接受 `bug45` 的 `[BLOCK]` 为历史缺口，不重写、不移动 `bug45`/`dev106`。
- [ ] 确认 `0.1.17` / `test37` 已发布且不可覆盖，本轮只发 `0.1.18` / `test38`。

最终发布决定：`□ 允许上传（十轮全 PASS + 上述两条已确认）`　`□ 不允许（存在 FAIL/UNVERIFIED）`

## 附录 C — 本清单独有的禁止项

1. 不得为了让某一轮变绿而修改测试阈值、跳过断言、伪造 PASS
   （`pyqcu/test/AGENTS.md`「数值与证据」）。
2. 不得为“让 sdist 计数对上”而删除或改名 `pyqcu/test/*.py`；计数以工作树为准。
3. 不得用 `--quick` 产生任何性能/加速比结论（`pyqcu/test/README.md`「目录约定」）。
4. 不得把单节点结果写成两节点结果；不得把 PENDING 写成 PASS。
5. 不得把工作区未提交的 `driver.py`/`ledger_*.py`/`L48` preset 当作“已修复”；判定只认
   `git cat-file -e HEAD:<path>` 与远端复跑证据（`docs/pyqcu_test_audit_risk_20261117.md` §1.2）。
6. 不得重写/删除任何已推送标签；`tag-chain.sh --repair --apply` 只在安全候选非 0 且有
   明确授权时使用。
7. 不得在本清单之外顺手改 `AGENTS.md`、版本、标签、`pyqcu/testing/**` 或
   `pyqcu/**` 实现。
