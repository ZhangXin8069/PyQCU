# PyPI 发布逐项清单（pyqcu）

`SKILL.md` 的执行附件。只有在真正执行发布、核对 sdist 成员或排障时读本文件。
命令都在仓库根 `PyQCU/` 下执行；`<ver>` 替换为当前版本，如 `0.1.1`。

## 1. 预发布清单

- [ ] 4 个版本落点一致：`pyproject.toml`、`pyqcu/__init__.py`、`docs/en/conf.py`、`docs/zh/conf.py`
- [ ] 两语 `docs/<lang>/install.md` 的两个版本断言改成新版本
- [ ] 两语 `overview.md`（`This release is`/`本版本为`）与 `performance.md`
      的版本引用同步（无自动检查，必须 grep）
- [ ] 旧版本 grep 只剩 `cpp/kupl/qcu/CMakeLists.txt:98` 的 soname `VERSION 0.1.0`
- [ ] README 未引入版本字面量（`README.md`、`README.zh-CN.md`）
- [ ] `MANIFEST.in` 含 `README.zh-CN.md`，未把 `.so/.o` 或 `pyqcu/testing` 带进 sdist
- [ ] 工作树干净：`git status --porcelain` 为空，`git describe --tags` 指向本次提交
- [ ] `dist/` 已清空（旧版本产物会让 release_check 读到错文件）
- [ ] 发布门 12/12、`twine check` PASSED
- [ ] 台账 `logs/v<YYYYMMDD>.txt` 已写（产物大小、sha256、门禁输出、数值、外部阻塞）

## 2. 版本同步核对

```bash
grep -n 'version = ' pyproject.toml | head -3
grep -n '__version__' pyqcu/__init__.py
grep -n 'version\|release' docs/en/conf.py docs/zh/conf.py
grep -n 'version' docs/en/install.md docs/zh/install.md
```

旧版本残留扫描（排除历史日志与数据）：

```bash
grep -rn "0\.1\.0" --include=*.py --include=*.toml --include=*.md --include=*.txt \
  --include=*.in . | grep -v '^./logs/' | grep -v '^./data/' | grep -v '^./build/'
```

唯一合法残留：`cpp/kupl/qcu/CMakeLists.txt:98  VERSION 0.1.0`（KUPL 原生库
soname，`SOVERSION 0`，与 Python 发行版本解耦；两个发行版本期间都保持 0.1.0）。

## 3. sdist 基线核对

构建：`rm -rf dist && python -m build --sdist`（本次用的 build 1.6.1）。

```bash
tar tzf dist/pyqcu-<ver>.tar.gz > /tmp/sd.txt
wc -l < /tmp/sd.txt                                   # 成员数
grep -c '\.py$' /tmp/sd.txt                           # Python 文件数
grep -c 'pyqcu/testing/' /tmp/sd.txt                  # 必须 0
grep -c '\.so' /tmp/sd.txt                            # 必须 0
grep -c 'cpp/kupl/qcu/src/' /tmp/sd.txt               # KUPL 源
grep -c 'cpp/cuda/qcu/src/' /tmp/sd.txt               # CUDA 源
grep -c 'README.zh-CN.md' /tmp/sd.txt                 # 必须 >= 1
tar xzOf dist/pyqcu-<ver>.tar.gz pyqcu-<ver>/pyqcu.egg-info/SOURCES.txt | grep -c '/home/'
```

0.1.1 实测基线：成员 254、`.py` 87、`pyqcu/testing/` 0、`.so` 0、KUPL src 34、
CUDA src 33、`README.zh-CN.md` 1、`SOURCES.txt` 无 `/home/`。0.1.0 成员 253、
598,903 B；0.1.1 601,420 B。两个 sdist 的逐成员对比只有增量没有改动：新增
`README.zh-CN.md`（+5,519 B 未压缩）、`README.md` 的 Repositories 段（+457 B）、
`pyproject.toml` 的 3 个链接（+202 B），以及派生的 `PKG-INFO`/`SOURCES.txt`。

判读：成员数骤降或 `cpp/kupl/qcu/src` 为 0 说明 `MANIFEST.in` 的
`recursive-include` 被改坏；`.so` 计数非 0 说明构建产物被误打包。

## 4. 发布门

```bash
python -m pyqcu.testing.release_check --offline     # 要求 12/12
python -m pyqcu.testing.release_check --online      # 追加 PyPI/RTD 探针
```

`--offline` 的 12 项（顺序即输出顺序）：

1. `version` — `pyproject` 与 `pyqcu.__version__` 一致
2. `sdist exists`
3. `sdist no testing`
4. `sdist no binaries`
5. `sdist native sources`（KUPL 与 CUDA 源同时存在）
6. `sdist no home paths`
7. `sdist Chinese README`
8. `twine check`
9. `bilingual blocks`（overview/usage/performance/install 的代码块 en==zh）
10. `docs examples`（`pyqcu.testing.docs_examples --keep-going`）
11. `sphinx en`
12. `sphinx zh`

`--online` 追加：`PyPI JSON`、`PyPI project links`（6 个键齐全）、
`PyPI repositories section`（描述含 `## Repositories`）、`PyPI simple index`、
`RTD en/latest`、`RTD zh-cn/latest`。

`docs_examples` 的规则：每个 ```python 块必须包含 `assert `，否则要标
`# docs-skip` 或 `# pseudo-code`；同页代码块共享命名空间、按页序执行。
0.1.1 基线：`total: ok=18 skipped=2 failed=0`。

## 5. 上传与复验

```bash
python -m twine check dist/pyqcu-<ver>.tar.gz        # PASSED
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_ZHANGXIN8069_API_KEY" \
  python -m twine upload --non-interactive dist/pyqcu-<ver>.tar.gz

# 传播
curl -s https://pypi.org/simple/pyqcu/ | grep -o "pyqcu-<ver>.tar.gz"
```

干净安装（venv 放在 `data/pypi/` 下，不污染系统环境）：

```bash
python -m venv data/pypi/venv-<ver>-clean
data/pypi/venv-<ver>-clean/bin/pip install --no-cache-dir \
  --index-url https://pypi.org/simple pyqcu==<ver>

cd /tmp && /abs/path/data/pypi/venv-<ver>-clean/bin/python -m pyqcu
```

断言要点：`location` 必须落在 `site-packages/pyqcu/__init__.py`（不能是源码
树），`kupl` 行给出包内 `libqcu.so` 路径，`cuda bridge` 行给出
`qcu.cpython-*.so` 路径。

其余复验（都在源码树之外的 cwd 执行）：

```bash
<venv>/bin/python -m pip check                        # No broken requirements found.
<venv>/bin/python -m pip install --force-reinstall --no-deps \
  --no-build-isolation /abs/path/PyQCU/dist/pyqcu-<ver>.tar.gz   # 幂等重装
<venv>/bin/python -m pip uninstall -y pyqcu && <venv>/bin/pip install /abs/path/PyQCU/dist/pyqcu-<ver>.tar.gz
<venv>/bin/python -m pip install --dry-run "pyqcu[plot]==<ver>"
<venv>/bin/python -m pip install --dry-run "pyqcu[monitor]==<ver>"
<venv>/bin/python -m pip install --dry-run "pyqcu[all]==<ver>"
```

注意：

- 已装旧版时装新版必须写 `pyqcu==<ver>` 或加 `--upgrade`；裸
  `pip install pyqcu` 只会输出 `Requirement already satisfied ... (0.1.0)`
  （见 `data/pypi/pypi-upgrade-011.log`）。
- 刚上传后的第一次安装可能仍拿到旧版本：0.1.1 首次 clean venv 装到 0.1.0
  （CDN 边缘未刷新），重试才升级并卸载 0.1.0。装完必须用
  `python -m pyqcu` 核实版本，不要只看 pip 退出码；必要时 `--no-cache-dir`
  后重试。
- 想快速验“打包能装”而不重建原生库时，加 `PYQCU_SKIP_NATIVE_BUILD=1`：
  0.1.1 的免原生 wheel 约 311 KB，带 KUPL+CUDA 的 wheel 约 17.1 MB。
- extras 解析核对（不依赖网络）：

  ```bash
  <venv>/bin/python - <<'PY'
  import importlib.metadata as md
  d = md.distribution("pyqcu")
  print(d.metadata.get_all("Provides-Extra"))
  print(d.metadata.get_all("Requires-Dist"))
  PY
  ```

  → `['plot', 'monitor', 'all']` 与 `matplotlib>=3.6; extra == "plot"` 等 4 条
  extras 记录。0.1.1 实测与 PyPI JSON 的 `provides_extra`/`requires_dist` 一致。

## 6. 项目链接与 RTD / 镜像核对

```bash
python - <<'PY'
import json, urllib.request
info = json.load(urllib.request.urlopen("https://pypi.org/pypi/pyqcu/<ver>/json"))["info"]
print(sorted(info["project_urls"]))
print("## Repositories" in (info["description"] or ""))
PY
```

必须出现且只认这 6 个键（与 pyqcd 同形式）：

`Documentation`、`Homepage`、`国内开发 (GitLab)`、`国内发布 (Gitee)`、
`国外发布 (GitHub)`、`Issues`。

RTD：

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pyqcu.readthedocs.io/en/latest/
curl -s -o /dev/null -w '%{http_code}\n' https://pyqcu.readthedocs.io/zh-cn/latest/
git ls-remote https://github.com/ZhangXin8069/PyQCU.git main
```

- 配置只有 `docs/en/.readthedocs.yaml`、`docs/zh/.readthedocs.yaml`，仓库根
  不设；两个文件在 RTD 网页按语言分别指定。
- RTD 跟踪 GitHub，GitHub 跟随 Gitee（推 Gitee 后自动更新），所以“四端是否
  同一 commit”决定文档何时可见。
- 构建 `success=false` + `duration=0` = tracked revision 里没有
  `.readthedocs.yaml`；先确认 `git ls-remote` 指向的提交里存在该文件。

## 7. 故障对照表

| 症状 | 判据 | 处理 |
|---|---|---|
| 隔离构建 `ImportError: pyqcu._build_backend` | `pip install .` 失败、`--no-build-isolation` 成功 | setup.py 顶部 `os.chdir` + `sys.path.insert` 兜底 |
| `AttributeError: 'Extension' object has no attribute '_needs_stub'` | 只在 GPU 后端建成功时出现 | 逐个扩展补 `_needs_stub = False` |
| sdist 缺 KUPL/CUDA 源 | `tar tzf` 里 `cpp/*/qcu/src` 计数为 0 | 修 `MANIFEST.in` 的 `recursive-include` |
| sdist 混进 `.so` | `grep -c '\.so'` 非 0 | `global-exclude *.so *.so.*` 与 `prune build` |
| `sdist no home paths` 失败 | `SOURCES.txt` 含 `/home/` | Extension 源改相对路径，清 `build/` 重构建 |
| `docs examples` 失败 | 输出指出 `en/install#N: no assertion` 或断言值不符 | 改 `install.md` 版本断言或补 `# docs-skip` |
| `bilingual blocks` 失败 | 指出哪一页 en/zh 块不同 | 两语同步改，别只改一边 |
| 装完 `python -m pyqcu` 报 KUPL `FileNotFoundError` | 无原生库的合法降级 | 装工具链重装，或设 `PYQCU_KUPL_LIB` |
| pip 长时间无输出 | 原生构建进度写 `/dev/tty` | 不要中断；`ls cpp/kupl/qcu/build` 看进展 |
| 上传报 `400 File already exists` | 该版本号已存在 | 发新版本（不能覆盖，也不能 yank 后重传） |

## 8. 台账模板

每次发布追加一节到当天的 `logs/v<YYYYMMDD>.txt`（0.1.0 与 0.1.1 两节现在都在
`logs/v20261109.txt`），字段如下：

```
PyQCU <ver> PyPI release log
Date / Operator / Package / PyPI project / Upload credential (name only)

Artifact        : dist/pyqcu-<ver>.tar.gz, <size>, sha256 <hash>, twine PASSED
sdist contents  : <members> / <py files> / 0 testing / 0 .so / KUPL src <n> / CUDA src <n>
Release gate    : release_check --offline <n>/<n>; pytest <a> passed + <b> passed
Verification    : 8x8x16x16 c64 KUPL <it> it / <residual> / <wall>; CUDA <...>; acceptance 2.4414e-4
Install check   : clean venv <path>, pip install from PyPI OK, python -m pyqcu -> site-packages, pip check OK
Git             : commit <sha>, local/gitlab/gitee/github 同一 commit, tag <tag>
RTD             : <en/zh page title + HTTP status; push-triggered build>
Next steps      : <逐条>
```
