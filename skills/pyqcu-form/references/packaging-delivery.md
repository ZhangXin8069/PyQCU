# packaging-delivery — 发布文件、sdist 与 Git 交付

`pyqcu-form` 的按需参考：只在任务涉及发布文件（`pyproject.toml`、`MANIFEST.in`、RTD 站点源）、
sdist 内容或提交/推送时读取。命名矩阵、顶层目录白名单和本地例外仍以父 `SKILL.md`、
根 `AGENTS.md` 的 form 托管块与 `docs/ORGANIZATION.md` 为准；本文件不重复命名规则。

## 1. 根入口白名单的两个新成员

根 `AGENTS.md` 托管块与 `SKILL.md` 的根文件白名单已加入 `pyproject.toml`、`MANIFEST.in`
（提交 `b63e779` 的 `AGENTS.md` diff 只有这一处）：

| 根文件 | 为什么必须是根入口例外 |
|---|---|
| `pyproject.toml` | PEP 517/518 构建入口与 PEP 621 元数据，setuptools 只认同仓库根；移到子目录即无法 `python -m build`。 |
| `MANIFEST.in` | sdist 清单，`setuptools` 在仓库根读取；与 `setup.py` 同级才有意义。 |

同时明确**不新增仓库根 `.readthedocs.yaml`**：RTD 沿用 PyQCD 的 per-language 约定，配置放在
`docs/en/.readthedocs.yaml`、`docs/zh/.readthedocs.yaml`，在 RTD 网页分别指定。仓库根一旦
出现 `.readthedocs.yaml`，RTD 会改读根配置，与 per-language 约定分叉，按格式冲突处理。

## 2. 根 `.gitignore`：`*` + 扩展白名单 + `.*`

根文件由 `give_git_ignore.sh` 生成，顺序即语义：

1. `*` 全忽略 + `!*/` 保留目录遍历；
2. 无扩展名文件逐条放行（`!/LICENSE` 等）；
3. 扩展白名单（`!*.py`、`!*.cpp`、`!*.h`、`!*.md`、`!*.sh`、`!*.txt` …）；
4. `.*` 忽略隐藏文件/目录，并追加 `pyqcu/*.cpp`、`*.so` 两条忽略规则；
5. 常见工程文件名放行（`!Makefile`、`!README.\*` 等）；
6. 发布文件放行块；
7. `*.egg-info/`、`build/`、`dist/`。

第 1 条是 `*`，所以**每个发布文件都要自己写放行**。`.gitignore` 第 6 节（发布文件块）现行内容：

| 行 | 作用 | 缺了会怎样 |
|---|---|---|
| `!pyproject.toml` | 放行构建入口 | 被 `*` 忽略，`git add` 前 `git status` 看不到，发布缺元数据 |
| `!MANIFEST.in` | 放行 sdist 清单 | 同上；sdist 内容漂移且无法复现 |
| `!docs/en/.readthedocs.yaml` | 显式放行隐藏配置文件 | 被第 4 节 `.*` 吃掉，RTD 找不到配置 |
| `!docs/zh/.readthedocs.yaml` | 同上（中文站点） | 同上 |
| `!docs/en/**` | 放行整棵英文站点源 | `index.rst`（`.rst` 不在扩展白名单）整棵丢失 |
| `!docs/zh/**` | 放行整棵中文站点源 | 同上 |
| `*.egg-info/` | 构建元数据 | 每次构建都产生未跟踪噪声 |
| `build/`、`dist/` | 构建/发布产物 | sdist、wheel、临时树被误提交 |

顺序要点：

- 后出现的规则覆盖先出现的。第 3 节的 `!*.so` 被第 4 节 `*.so` 覆盖，`.so` 始终忽略；
  发布块里**不要**重新放行二进制。
- `.*` 在第 4 节，发布块必须排在其后，否则 `.readthedocs.yaml` 仍被隐藏规则忽略。
- 根 `!docs/en/**` 会连带放行 `docs/en/_build/**`（实测见下），构建产物只能在更深的
  `docs/.gitignore` 兜住——更深目录的 `.gitignore` 规则优先。

实测（scratch 仓库只放根 `.gitignore`，`git check-ignore -v --no-index <path>`）：

```
docs/en/index.rst            .gitignore:92:!docs/en/**
docs/en/conf.py              .gitignore:92:!docs/en/**
docs/en/requirements.txt     .gitignore:92:!docs/en/**
docs/en/.readthedocs.yaml    .gitignore:92:!docs/en/**
docs/en/_build/x.html        .gitignore:92:!docs/en/**   # 根规则放行了，需 docs/.gitignore 兜底
docs/ja/index.rst            .gitignore:8:*             # 新语言目录必须自己登记
pyproject.toml               .gitignore:88:!pyproject.toml
MANIFEST.in                  .gitignore:89:!MANIFEST.in
README.zh-CN.md              .gitignore:30:!*.md
pyqcu.egg-info/PKG-INFO      .gitignore:94:*.egg-info/
build/x.py                   .gitignore:95:build/
dist/x.tar.gz                .gitignore:96:dist/
```

**不要用 `!*.toml` / `!*.yaml` 广域放行**：那会把本机配置、venv/缓存元数据和未来任何
YAML 一起拉进跟踪范围，与“每条发布路径都有登记理由”的白名单模型相反，也失去“新发布文件
必须显式登记”的检查点。本次真正需要的 YAML 只有两份 per-language RTD 配置，逐文件写即可。

## 3. `docs/.gitignore`：只忽略 `_build/`

规范形态是“默认跟踪、只挡构建输出”：

```
# Keep the Read the Docs sources (docs/en, docs/zh); ignore local build output only.
_build/
```

不得把 `**/` 当作常规起点：`**/` 会把 `docs/**` 的目录连同内容一起忽略，必须为每个语言目录
补 `!en/` + `!en/**`（以及 `!zh/` + `!zh/**`）成对放行，漏一对就整棵静默丢失。现行文件
第 2 行仍是历史 `**/`，第 3–6 行是成对放行，第 7 行 `_build/` 必须在放行之后：

```
$ git check-ignore -v --no-index docs/ja/index.rst
docs/.gitignore:2:**/          # 未登记的新语言目录被整棵忽略
$ git check-ignore -v --no-index docs/en/index.md
docs/.gitignore:4:!en/**       # en 正常跟踪
$ git check-ignore -v --no-index docs/en/_build/index.html
docs/.gitignore:7:_build/      # 构建产物被兜住
```

规则：新增站点语言（`docs/<lang>/`）先补成对放行再 `git add`，并用 `git check-ignore -v`
证明目标文件未被忽略；清理 `**/` 属于独立批次，必须复测 `docs/en|zh` 全量仍被跟踪。

## 4. sdist 形态（`MANIFEST.in`）

sdist 要在没有预编译原生库、甚至没有仓库的机器上装起来，所以源码与构建脚本必须自带：

| 段 | 内容 | 理由 |
|---|---|---|
| 元文件 | 逐行 `include LICENSE`、`README.md`、`README.zh-CN.md`、`pyproject.toml`、`setup.py`、`MANIFEST.in` | 占位符与自重建；`MANIFEST.in` 自包含保证从 sdist 再打包结果一致 |
| 根入口 | `include env.sh build-nv.sh build-dcu.sh install.sh` | 安装机需要同一套构建/安装入口 |
| Python 源码 | `recursive-include pyqcu *.py *.pyi *.pyx *.pxd` | 必须带 `*.pyx`/`*.pxd`，否则 `install.sh` 无法 cythonize |
| KUPL 树 | `recursive-include cpp/kupl/qcu *.txt *.sh *.map *.md *.h *.hpp *.cpp` + `prune cpp/kupl/qcu/tests`、`prune cpp/kupl/qcu/build`、`prune cpp/kupl/qcu/build-lto` | 自包含 C++17 构建树，不带测试与本地构建目录 |
| CUDA 树 | `LICENSE`、`CMakeLists-{nv,dcu}.txt`、`make-{nv,dcu}.sh`、`env.sh`、`python/*.h`、`include/*.h(pp)`、`src/*.cu/*.cpp/*.h` | C API 与内核源码，安装时机遇性构建 |
| 排除目录 | `prune pyqcu/testing`、`build`、`dist`、`data`、`docs`、`logs`、`refer`、`skills` | 测试、证据、文档站点、历史参考与技能都不进发布物 |
| 排除产物 | `global-exclude *.so *.so.* *.o *.pyc *.pyo *.pyd` | sdist 不带二进制 |
| 排除缓存 | `global-exclude __pycache__ *.egg-info` | |
| 排除构建残留 | `global-exclude CMakeCache.txt cmake_install.cmake CTestTestfile.cmake Makefile` | |
| 排除规则文件 | `global-exclude AGENTS.md` | agent 规则文件不属于发布物 |

检查点：

- `README.zh-CN.md` 被 `include`，就必须同时被 git 跟踪（`!*.md` 已放行）；clean clone 里缺文件
  会让 `python -m build` 直接失败，`release_check` 的 "sdist Chinese README" 也会红。
  它还是根入口文件，提交前必须登记进根 `AGENTS.md` 的根文件白名单（托管块由
  `form-agents-sync.sh` 维护，未登记即白名单与仓库事实不一致；本次任务不改根 `AGENTS.md`）。
- `SOURCES.txt` 出现 `/home/` 是绝对路径泄漏（`recursive-include` 写错或 `setup.py` 生成路径），
  必须修 manifest，不得改检查或忽略该断言。
- `prune pyqcu/testing` 与 `global-exclude *.so` 是发布闸门两条硬线，`release_check` 会独立复核。

```bash
python -m build --sdist
tar tzf dist/pyqcu-<ver>.tar.gz | grep -E '/pyqcu/testing/|\.so$|\.o$'      # 期望无输出
tar xzOf dist/pyqcu-<ver>.tar.gz pyqcu-<ver>/pyqcu.egg-info/SOURCES.txt | grep '/home/'  # 期望无输出
python -m twine check dist/pyqcu-<ver>.tar.gz
python -m pyqcu.testing.release_check --offline   # 逐条记录条目数与通过数，不写死历史数字
```

`release_check --offline` 覆盖：version、sdist exists、no testing、no binaries、native sources、
no home paths、Chinese README、twine check、bilingual blocks、docs examples、sphinx en、sphinx zh。
0.1.0 台账（`logs/v20261109.txt`）：585 KiB、sha256 `bed98ee7e6d774c69efa8f8752d40fe0d24b79c02df91701d720be9627927ba4`、
86 个 Python 文件、0 个 `pyqcu/testing`、0 个 `.so`、KUPL 34 条 src、CUDA 33 条 src、无 `/home/`。

## 5. Git 交付

暂存与提交：

- 只暂存本任务文件：`git add -- <显式路径>`；**禁止** `git add -A`、`git add .`、`git commit -a`。
- 提交前用 `git diff --cached --name-only` 与任务清单逐条对齐；出现未预期路径先
  `git restore --staged <path>`，再按显式路径重加。
- 标题用 `<area>: <summary>` + 要点正文；最后一段固定 trailer：
  `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`。
- 提交前依次跑大文件闸门（根 `AGENTS.md` 的 `form-large-files-check.sh`）、`git diff --check`、
  `bash skills/pyqcu-form/form-audit.sh`。

远端拓扑（推送必须双远端）：

| 名称 | URL | 角色 |
|---|---|---|
| `origin` | `git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git` | 主开发远端（GitLab） |
| `gitee` | `git@gitee.com:zhangxin8069/PyQCU.git` | 国内发布远端；GitHub 由它自动镜像 |
| github | `https://github.com/zhangxin8069/PyQCU.git` | 自动镜像，**不是**推送目标 |

```bash
git remote -v                                   # 缺 gitee 时先加，仅首次
git remote add gitee git@gitee.com:zhangxin8069/PyQCU.git
git push origin main
git push gitee main
```

- 不要直接推 GitHub：Gitee 推送成功后由镜像同步；先确认 Gitee 已到目标 commit，再等镜像重查。
- 推送失败不 force push：先 `git fetch <remote>` 判断领先/分叉，无法安全推进时保留本地并报告。
- 不改写、不删除已推送标签；标签链检查用 `skills/pyqcu-tag/tag-chain.sh --check`。

四端核验（local / GitLab / Gitee / GitHub 必须同一 commit）：

```bash
local=$(git rev-parse HEAD)
gitlab=$(git ls-remote origin refs/heads/main | cut -f1)
gitee=$(git ls-remote gitee refs/heads/main | cut -f1)
github=$(git ls-remote https://github.com/zhangxin8069/PyQCU.git refs/heads/main | cut -f1)
printf '%s\n%s\n%s\n%s\n' "$local" "$gitlab" "$gitee" "$github" | sort -u | wc -l   # 期望 1
```

镜像同步有延迟：首次查到的 GitHub commit 落后属于传播延迟，间隔重查后再判定；不得因此改推
GitHub 或重写历史。

## 6. 发布台账与文档归属

- 发布台账是 `logs/vYYYYMMDD.txt`（本次 `logs/v20261109.txt`，随待 tag 提交进入仓库）。
  必要段落：Scope / Files changed / Artifact（大小、sha256、twine check、sdist 内容统计）/
  Build & install evidence / 验证格点与结果 / Release gate（`release_check` 条目数与通过数）/
  PyPI/RTD 传播状态 / Next steps（外部阻塞项）。
- 凭证只写名称（如 `PYPI_ZHANGXIN8069_API_KEY`），不写值、token 或 `~/.pypirc` 内容。
- `docs/en/**`、`docs/zh/**` 是 Read the Docs **站点源文**（`conf.py`、`index.rst`、
  `requirements.txt`、各页 `.md`、`.readthedocs.yaml`），是站点构建输入，不是 `docs/` 顶层
  研究报告；归档与命名都不套用顶层报告规则。
- 台账仍属 `logs/**`：报告已进入 tag 后按 `logs/AGENTS.md` 生命周期在清理提交中删除工作区副本。

## 7. 技能索引与计数

新增技能必须一次落齐四件事，缺一即审计失败：

1. `skills/pyqcu-<name>/` 目录（`pyqcu-` 前缀 + 小写连字符）；
2. `SKILL.md`：frontmatter `name` 与目录名逐字符一致、description 写触发条件、< 500 行、唯一 H1；
3. `AGENTS.md`：首行 `# AGENTS.md — pyqcu-<name>`、≤ 12 行、必须出现 `SKILL.md`；
4. `skills/AGENTS.md` 的技能表行（恰好一行）与两处计数 `当前 N/N`、`## 技能表（N/N）`；
   同时更新根 `AGENTS.md` 目录表的 `N 个技能目录`。

审计口径：

- `form-audit.sh` 用实际目录数比对索引行数，并要求 `当前 N/N`、`## 技能表（N/N）` 同时出现
  （最近一次新增是 35→36：索引行、两处计数和根 `AGENTS.md` 的 `N 个技能目录` 一起改）；
- 每个 `pyqcu-*` 目录必须有 `SKILL.md` 与 `AGENTS.md`，frontmatter `name` 必须等于目录名；
- 索引项不能多也不能少（`SKILL_INDEX_MISSING` / `SKILL_INDEX_EXTRA`）；
- `references/`、`scripts/`、`assets/`、`evals/` 下的每个文件都必须在 `SKILL.md` 或 `AGENTS.md`
  里声明读取分支，否则 `POINTER` 失败（`audit-docs-skills.py`）。

索引行与计数只在**确实增删技能**时改；只加参考文件时保持原计数（本次 35/35 不变）。

## 8. 本地豁免登记（`form-audit.sh`）

- 先判断是资产本身合法还是布局真违规；只有前者才登记例外。
- 例外写在 `case` 分支，精确到 `severity:rule:path`；用前缀通配时限定在同一目录树，
  并在分支内注释记录证据（哪些文件、为什么不能改名）。
- 不改历史豁免项、不整体 `--exclude`、不关闭审计；新增后必须重跑并记录
  `PASS: … 豁免 N 个既有资产 finding` 的输出。
- 根 `AGENTS.md` 托管块里的豁免计数由 `form-agents-sync.sh` 维护，不在手工编辑范围。
- 已登记：`WARNING:DOC_EXTENSION:docs/en/*` 与 `WARNING:DOC_EXTENSION:docs/zh/*`。证据是 8 个已跟踪站点源
  （`conf.py`、`index.rst`、`requirements.txt`、`.readthedocs.yaml` × en/zh）：Sphinx 与 RTD
  要求这些固定文件名，无法改成 `.md`；它们属于站点构建输入，不是 `docs/` 顶层研究报告。

## 9. 交付检查清单

```bash
git status --short
git diff --check
bash skills/pyqcu-form/form-audit.sh
python -B skills/pyqcu-docs/scripts/audit-docs-skills.py
python -m build --sdist && python -m pyqcu.testing.release_check --offline
git check-ignore -v --no-index pyproject.toml MANIFEST.in docs/en/.readthedocs.yaml
```

出现以下信号说明改错了方向：`git status` 里成片未跟踪的 `docs/en|zh` 文件（放行缺失）、
`sdist` 里出现 `pyqcu/testing` 或 `.so`（manifest 漏 prune）、`SOURCES.txt` 含 `/home/`
（绝对路径泄漏）、`form-audit.sh` 报 `SKILL_INDEX_COUNT`（计数未同步）、
四端 commit 不一致（推送未完成或镜像未同步）。
