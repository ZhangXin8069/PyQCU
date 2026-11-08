# Read the Docs 双语站点发布检查单

本文件是 PyQCU 两个 Read the Docs（RTD）站点的唯一操作规范：项目命名、配置布局、
站点结构、本地构建门、`release_check` 联动、版本同步点和远端排障。

来源：`docs/en/.readthedocs.yaml`、`docs/zh/.readthedocs.yaml`、`docs/{en,zh}/conf.py`、
`docs/{en,zh}/index.rst`、`docs/{en,zh}/{overview,usage,performance,install}.md`、
`docs/.gitignore`、`pyqcu/testing/docs_examples.py`、`pyqcu/testing/release_check.py`、
`logs/v20261109.txt:145-167`。

状态：0.1.1 发布窗口的时点证据。本地构建门已实测通过；远端 RTD 在 `b63e779`
推送到 Gitee、GitHub 镜像自动更新后**构建成功**，`en/latest` 与 `zh-cn/latest`
均 HTTP 200（标题为 0.1.0；0.1.1 提交推送后应变为 0.1.1）。因此
“推 Gitee → GitHub 镜像 → RTD 自动构建”是已验证路径；早期 8847f84 的
`duration=0` 失败是 tracked revision 缺少配置的对照，不是项目配置或 token 阻塞。

## 1. RTD 项目事实与命名

| 事实 | 值 | 说明 |
|---|---|---|
| 项目 | `pyqcu`（已存在） | 沿用既有公开项目，不新建 |
| 站点名 | PyQCU | RTD 项目 name |
| slug / URL 前缀 | `pyqcu` | `https://pyqcu.readthedocs.io/` |
| 主语言 | `en` | 站点根路径 `/en/latest/` |
| 默认版本 | `latest` | 不要把 `stable` 当默认版本 |
| 跟踪仓库 | `https://github.com/ZhangXin8069/PyQCU.git` | GitHub 是 Gitee 的自动镜像 |

- slug 是 **`pyqcu`**，不是 `py-qcu`，也不是 `pyqcd`。RTD 上“PyQCD 用 `py-qcd`”
  只因为 `pyqcd` 被别人占用，是那一次的历史替代，**不是可推广的命名规则**；
  PyQCU 不要复制。
- 追踪仓库是 GitHub 镜像：推 Gitee 后 GitHub 自动更新，RTD 再拉 GitHub。因此
  “已推 Gitee”不等于“RTD 能构建”；发布时确认镜像的 `main` 已经包含
  `docs/en/.readthedocs.yaml` 与 `docs/zh/.readthedocs.yaml`（可用
  `git ls-remote https://github.com/ZhangXin8069/PyQCU.git refs/heads/main` 对比本地
  HEAD）。
- 观测到的构建耗时：成功构建 `duration=35s` 量级；这是单次观测，不是 SLA。
- **`duration=0` + `success=false` 通常表示 tracked revision 上没有可用的
  `.readthedocs.yaml`**（RTD 在 Sphinx 启动前就退出），不是 Sphinx 报错。
  2026-11 窗口内 35737742/35737652/35737585/35727377/35726644/35726643 全部是这个
  形态。查 Sphinx 错误时先确认该构建的 checkout 里配置文件存在。
- 无 RTD API token 时**不能从 CLI 手动触发**构建或读取构建日志；但推 Gitee 会经
  GitHub 镜像自动触发，不需要 token（本次已实测）。不要把“本地构建通过”写成
  “RTD 已发布”，要按页探针确认 HTTP 200 与标题版本。

## 2. 配置文件布局

- 只有两份 per-language 配置：`docs/en/.readthedocs.yaml` 与
  `docs/zh/.readthedocs.yaml`。
- **仓库根不设 `.readthedocs.yaml`**（沿用 PyQCD 约定）：两个站点的配置文件在 RTD
  网页按语言分别指定。代价是这一步无法用仓库内容自证，交付时必须写明“需要在 RTD
  UI 里为 en 和 zh 各选一次配置文件”。
- 两份配置体量一致，只换目录名：

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.12"
sphinx:
  configuration: docs/en/conf.py      # zh 侧为 docs/zh/conf.py
python:
  install:
    - requirements: docs/en/requirements.txt
submodules:
  exclude: all
```

- `docs/.gitignore` 必须保留对 `en/`、`zh/` 的否定规则（`!en/`、`!en/**`、`!zh/`、
  `!zh/**`），只忽略 `_build/`。**不能用 `**/` 或 `*` 把 `en/`、`zh/` 整个忽略**：
  RTD 只看 Git 内容，被忽略的源目录等于仓库里没有文档，症状同样是
  `duration=0`/找不到配置。
- 自检（不依赖 RTD）：

```bash
git ls-files docs/en docs/zh | wc -l          # 应等于 8×2：.readthedocs.yaml、conf.py、
                                              # index.rst、requirements.txt + 4 个页面
git check-ignore -v docs/en/conf.py docs/zh/conf.py   # 期望无输出（exit 1）
```

## 3. 站点结构

- 每侧一套独立的 Sphinx 源：`index.rst` + `overview.md`、`usage.md`、`performance.md`、
  `install.md` + `conf.py` + `requirements.txt`。
- `index.rst` 只放站点标题、一句定位、`pip install pyqcu` 代码块和
  `toctree`（`overview → usage → performance → install`，中英顺序一致）。
- 正文用 MyST Markdown；`conf.py` 用 `extensions = ["myst_parser"]`、
  `source_suffix = {".rst": "restructuredtext", ".md": "markdown"}`、
  `master_doc = "index"`、`html_theme = "sphinx_rtd_theme"`、
  `exclude_patterns = ["_build"]`。
- **`conf.py` 不得 import `pyqcu`**：包可能要求本机 CUDA/KUPL 原生库，RTD 构建机
  没有 GPU 也不该装包。`project`/`version`/`release` 都是字面量。
- `version = "0.1"`、`release = "0.1.1"`：`release` 带补丁号（页面显示），`version`
  是次版本；发布时只改 `release` 属于漏改检查项之一。
- zh 侧额外 `language = "zh_CN"`（Sphinx locale）与 `html_search_language = "zh"`，
  `project = "PyQCU（中文文档）"`。
- **`language = "zh_CN"` 与 URL 里的 `zh-cn` 是两个命名空间**：前者是 Sphinx
  语言码，后者是 RTD 翻译语言 slug。中文站 URL 是
  `https://pyqcu.readthedocs.io/zh-cn/latest/`；不要为了“对齐 URL”把 `conf.py` 改成
  `zh-cn`，也不要以为改了 `conf.py` 就会自动出现 `zh-cn` 翻译——翻译条目要在 RTD
  UI 添加。
- `requirements.txt` 两侧保持一致且锁定下限：`sphinx>=7`、`sphinx-rtd-theme>=2`、
  `myst-parser>=2`。单侧改版本会让两个站点解析到不同 Sphinx，出现只有一侧失败的
  `-W` 警告。

## 4. 本地构建门（每次改 docs 必跑）

```bash
python -m sphinx -b html -d data/release_check/doctrees-en -W docs/en data/release_check/sphinx-en
python -m sphinx -b html -d data/release_check/doctrees-zh -W docs/zh data/release_check/sphinx-zh
```

- 两个命令都必须 exit 0，日志含 `build succeeded`。**`-W` 把警告变错误**：未登记
  toctree、未知指令、未解析引用都会让站点门失败。
- `-d` 把 doctrees 放到 `data/release_check/`，不要写进 `docs/**`；否则会被当成源
  文件并触发重复文档警告。
- 中英围栏代码块必须**逐字节一致**：`release_check.check_bilingual()` 抓取每页所有
  带小写语言标签的围栏（`python` 与 `bash` 都算），再逐块比较。只翻译围栏**外**的
  说明文字；围栏内的注释（如 `# 2) generate the release gauge field`）也是被比较
  的字节，两侧必须相同。
- 文档示例门：

```bash
python -m pyqcu.testing.docs_examples --keep-going
```

  规则：每个 python 围栏块要么含 `assert `，要么标 `# docs-skip` / `# pseudo-code`。
  同一页的块共享命名空间并按页内顺序执行（前面的 `U`/`src`/`kappa` 供后面的块使用，
  重命名会连带打断后续块）。0.1.1 基线：**18 ok / 2 skipped / 0 failed**（en+zh 共
  20 个 python 块）。
- 两个容易踩的匹配细节：块必须写成行首的 python 围栏（带 info-string 后缀或缩进就
  不会被收集）；未闭合围栏会让正则吞掉后续内容。`performance.md` 当前只有 bash 块，
  **bash 块从不被执行**，其中的命令路径只能靠人工复核。
- `# docs-skip` 只用于环境依赖块（例如需要本机原生 KUPL 库的 import）。用它掩盖
  真实失败会让门失效；跳过计数仍会打印，交付时要把 `ok/skipped/failed` 三个数字
  一起报。

## 5. `release_check` 与发布联动

```bash
python -m pyqcu.testing.release_check --offline   # 版本 + sdist + 双语 + 示例 + 双站 -W
python -m pyqcu.testing.release_check --online    # 追加 PyPI 与 RTD 探针
```

- `--offline` 已包含双站 `-W`（`check_sphinx()` 复用上面的命令），所以本地发布门
  通过就意味着两站 exit 0；不必再手写一遍。0.1.1 工作树实测 **12/12 passed**
  （0.1.0 记录是 11/11，多出的一条是 `sdist Chinese README`）。
- `--offline` 仍然是完整发布门：它先查 `dist/pyqcu-<ver>.tar.gz` 存在性并跑
  `twine check`。没先 `python -m build --sdist`、或环境没有 `twine`，会得到
  `[FAIL] sdist exists` / `[FAIL] twine check` —— 这和文档无关，别把它读成 Sphinx
  失败，也别为了让门变绿去删检查。
- `check_version()` 要求 `pyproject.toml` 的 `version` == `pyqcu.__version__`；
  文档里的版本号不在这条检查里，必须另外 grep（见 §6）。
- `--online` 的 RTD 探针只用 `pyproject.toml` 的版本串去 grep
  `https://pyqcu.readthedocs.io/en/latest/` 与 `.../zh-cn/latest/` 两个根页，
  即 `HTTP 200 + 版本号出现在正文`，对“页面还是旧构建”不敏感。人工探针要**按页**
  检查：

```bash
for lang in en/latest zh-cn/latest; do
  for page in overview usage performance install; do
    curl -fsS "https://pyqcu.readthedocs.io/$lang/$page.html" \
      | grep -o "<title>[^<]*</title>"            # 页面身份
  done
done
```

  再对每页抓一个该页独有的章节名（如 `Historical cross-platform results` /
  `历史跨平台结果`）。只看根页 200 或只在整页文本里 grep 版本号，会漏掉“404 页被
  CDN 缓存”和“部分页已是新版本”两种情况。
- 重建有**分钟级延迟**，且构建期间 CDN 与新旧产物并存：不同页可能停在不同版本。
  判定失败前先重试并检查最近一次构建 id/duration；单页旧内容不是回滚证据。

## 6. 文档版本同步点

每次发布版本变更，以下位置必须一起改（顺序即检查顺序）：

| 位置 | 0.1.1 的值 | 漏改后果 |
|---|---|---|
| `pyproject.toml` / `pyqcu/__init__.py` | `0.1.1` | `release_check.check_version` 失败 |
| `docs/{en,zh}/conf.py` | `version = "0.1"`、`release = "0.1.1"` | 站点页脚显示旧补丁号 |
| `docs/{en,zh}/overview.md` | “This release is `0.1.1`” / “本版本为 `0.1.1`” | 概览与 PyPI 不一致 |
| `docs/{en,zh}/install.md` | `assert report["version"] == "0.1.1"`、`assert pyqcu.__version__ == "0.1.1"` | **docs_examples 直接失败**，不是外观问题 |
| `docs/{en,zh}/performance.md` | 本机验证段的 `pyqcu==0.1.1` | 本机构建基线失真 |

```bash
rg -n '0\.1\.[0-9]+' docs/en docs/zh pyproject.toml pyqcu/__init__.py
```

历史性能行里的 tag/版本（如 `dev103`、`0.1.0` 时代）不跟着改：它们记录的是当时
的复现条件，改了就破坏了“历史数据需 tag”的语义。

## 7. 内容边界

- 站点视角是**已发布包**：用户能做的是 `pip install pyqcu`、`python -m pyqcu`、
  公共 API import。首页和安装页必须能只靠 PyPI 走通“安装 → 自检”。
- **不把仓库内 `pyqcu/testing/**` 写成用户入口**；这些驱动不在 sdist 里。0.1.1 的
  现状偏差：`docs/{en,zh}/performance.md` 各有 5 行、`install.md` 各有 3 行仍带
  `pyqcu/testing/...`（复现与正确性小节）。这是待清理项，不是可模仿的模板；
  新增内容一律用公共 API 或 `python -m` 入口。
- 性能页每一行必须同时给出：平台（设备/CPU/GPU 型号与架构）、格点与精度、后端与
  求解器、线程/MPI 布局、结果（时间 + 迭代数 + 真残差）、比较值、来源文档与
  复现条件。缺任一项该行不能进表。
- 历史行必须显式标 “Historical / 历史数据” 并写明所需 tag 或时代 checkout、原始
  JSON 是否还在仓库里；不同硬件、不同窗口的绝对值不得合并成加速比。
- 本机 `8×8×16×16` 验证**单独成段**（`Local verification` / `本机验证`），写清
  `pyqcu==<ver>`、OS/CPU/GPU/驱动/CUDA/Torch、三个库的 SHA256、验收阈值
  `max(atol, 2048*eps32) ≈ 2.4414e-4`，并注明“CPU 与 GPU 比较的是不同硬件路径，
  不是 kernel-only 倍率”。
- 保留不利结果（例如 native BiStabCG 在 2 ranks 的 `0.833×`）：删掉不利行等于把
  对比变成单边宣传，也会让历史行失去可信度锚点。

## 8. 发布检查单（按顺序执行）

1. `rg -n '0\.1\.[0-9]+' docs/en docs/zh pyproject.toml pyqcu/__init__.py` 完成 §6 的
   五个同步点，中英两侧都要改。
2. `python -m pyqcu.testing.docs_examples --keep-going` → `failed=0`，记录 ok/skipped。
3. 双站 `python -m sphinx ... -W` → 两条 exit 0。
4. `python -m pyqcu.testing.release_check --offline` → 全 PASS（含上面两站）。
5. `git ls-files docs/en docs/zh | wc -l` 应为 16（每侧 8 个文件，含 `.readthedocs.yaml`）。
6. 推送后确认 GitHub 镜像 `main` 已含配置，再在 RTD UI 为 en/zh 各选一次配置文件、
   添加 `zh-cn` 翻译并触发构建。
7. 构建记录：`duration>0` 才说明进了 Sphinx；失败时先看 `duration`，再看 checkout
   里是否有配置。
8. 按页探针（§5），每页 `<title>` + 独有章节名 + 版本号三项都通过才算发布完成。

## 9. 失败判读

| 症状 | 常见原因 | 处置 |
|---|---|---|
| `success=false`、`duration=0` | tracked revision 没有 `.readthedocs.yaml`，或 `docs/.gitignore` 把 `en/`/`zh/` 忽略了 | 查镜像 revision 与 `git ls-files`，不要先怀疑 Sphinx |
| 线上 URL 404 | 还没有成功构建 | 先修构建，不要把 404 当缓存问题 |
| 只有一侧 `-W` 失败 | 两侧 `requirements.txt` 或 toctree 不一致 | 对齐依赖与 `index.rst` |
| `docs_examples` 报 `no assertion` | 新块缺 `assert` 且没标 `# docs-skip` | 补断言或显式跳过并说明原因 |
| `docs_examples` 版本断言失败 | §6 漏改 `install.md` | 两侧同时改 |
| `bilingual blocks` 失败 | 围栏内注释/命令被翻译或改动 | 恢复逐字节一致，只改围栏外文字 |
| 页面版本号混杂 | 构建进行中或 CDN 缓存 | 等分钟级重建后按页重探 |
| 无法触发构建 | 环境无 RTD token | 交给有网页权限的人；不要伪造构建结果 |

## 10. 未验证与边界

- 远端 RTD 构建成功、`zh-cn` 翻译页面、线上 `latest` 版本三者在本窗口都**未观测
  到成功样本**；本地门通过不等于站点已发布。
- `duration=35s` 是成功构建的单次观测值，不是耗时承诺。
- RTD 的 per-language 配置文件选择、翻译添加、token 权限都在网页侧，仓库内无法
  自证，也无法用 CLI 复核。
- `docs/{en,zh}/performance.md` 与 `install.md` 的 `pyqcu/testing/**` 路径属于 0.1.1
  现状，是否在下一版移出站点需随发布决定，本文只登记偏差。
