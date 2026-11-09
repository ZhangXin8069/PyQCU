---
name: pyqcu-pypi-release
description: |
  当用户要求发布、补发或复验 PyQCU 到 PyPI，调试 sdist 打包与原生库落位，
  同步版本号与文档站点版本，补齐 PyPI 项目链接，或排查 Read the Docs 构建与
  GitLab/Gitee/GitHub 三端同步时使用。
metadata:
  openclaw:
    emoji: 📦
---

# pyqcu-pypi-release — PyPI 发布与复验

## 目的与边界

覆盖 `pyqcu` 发行包从改版本到 PyPI 上架、再到干净环境复验的完整链路，并把
2026-11-09 的 0.1.0/0.1.1 两次真实发布作为基线证据。所有数字均为本机实测
（WSL2、i7-13620H、RTX 4060 Laptop sm_89、Torch 2.15.0+cu132）。

读取分支：

- 默认只读本文件即可完成版本同步、构建、发布门和上传。
- 逐项打勾发布、核对 sdist 成员、复验命令原文、故障对照表时读
  [`references/release-checklist.md`](references/release-checklist.md)。
- 版本/打包之外的领域问题交给相邻技能：C++/Cython 桥见
  [`pyqcu-cuda`](../pyqcu-cuda/SKILL.md)，KUPL 构建与 ABI 见
  [`pyqcu-kupl`](../pyqcu-kupl/SKILL.md)，标签与 follow 链见
  [`pyqcu-tag`](../pyqcu-tag/SKILL.md)，文档权威级别见
  [`pyqcu-docs`](../pyqcu-docs/SKILL.md)。

硬边界：只发布 sdist，不发布 wheel；不覆盖、不重传已发布版本；token 只从环境
变量读，不写入仓库、不回显；发布内容必须来自已提交的树。

## 发布事实

- 分发名 = 导入名 = `pyqcu`（`pyproject.toml` `name = "pyqcu"`，包目录
  `pyqcu/`）。没有下划线、大小写或别名变体。
- 只发 sdist：`python -m build --sdist` 产出 `dist/pyqcu-<ver>.tar.gz`。
  PyPI 侧只有 sdist——`https://pypi.org/simple/pyqcu/` 仅列出
  `pyqcu-0.1.0.tar.gz`、`pyqcu-0.1.1.tar.gz`，JSON 的 `urls[].packagetype`
  全部是 `sdist`。原生 `libqcu.so` 与 Cython 桥绑定构建机的 glibc/CPU/GPU
  架构，wheel 不可移植，因此不发 wheel。
- 上传凭据：token 只从环境变量 `PYPI_ZHANGXIN8069_API_KEY` 读取（这里只写
  变量名；值不进仓库、不进日志、不打印）。
- 项目页：`https://pypi.org/project/pyqcu/`；JSON 元数据
  `https://pypi.org/pypi/pyqcu/<ver>/json`。
- 已发布版本不可覆盖：0.1.0 的元数据只有 `Documentation`/`Homepage`/`Issues`
  三个 `project_urls`，修正只能发 0.1.1（实测 6 个链接）。同版本重传会被 PyPI
  拒绝，yank 也不能腾出版本号。
- 版本台账（PyPI JSON 与本地 `dist/` 双向核对一致）：

  | 版本 | 上传时间 (UTC) | sdist 大小 | sha256 |
  |---|---|---|---|
  | 0.1.0 | 2026-11-08T23:03:15Z | 598,903 B | `bed98ee7e6d774c69efa8f8752d40fe0d24b79c02df91701d720be9627927ba4` |
  | 0.1.1 | 2026-11-08T23:26:10Z | 601,420 B | `0499bedb4e68eaefed90626ff0bab62c703636587b95aec6f84d1381881f8995` |

- 0.1.1 相对 0.1.0 的全部增量（元数据修正版，无算法改动）：`pyproject.toml`
  增加 3 个中文项目链接；`README.md` 增加 `## Repositories` 表与中英互链；
  新增 `README.zh-CN.md` 并写入 `MANIFEST.in`；`docs/en|zh/conf.py`、两语
  `install.md`、两语 `performance.md` 的版本引用改为 0.1.1；
  `pyqcu/testing/release_check.py` 增加 “sdist Chinese README”“PyPI project
  links”“PyPI repositories section” 三项检查。
- `logs/v20261109.txt` 是本次两次发布的合并台账（0.1.0 节 + 0.1.1 节，含构建、
  校验、数值、重复安装矩阵和 RTD 自动构建记录）。0.1.1 的元数据修正必须与
  代码/文档在同一提交中推送；发布前确认 sdist 与提交内容一致。标签只在显式
  要求时创建，不把“已发 PyPI”自动等同于“已打 tag”。

## 版本与依赖

版本号有 4 个权威落点，必须同一次改动内一起改：

| 落点 | 发版时取值 | 作用 |
|---|---|---|
| `pyproject.toml` `[project].version` | `"<ver>"` | 唯一权威源，release_check 从这里读 |
| `pyqcu/__init__.py::__version__` | `"<ver>"` | `import pyqcu` 与 `python -m pyqcu` 用它 |
| `docs/en/conf.py` `release` | `"<ver>"` | 英文站版本 |
| `docs/zh/conf.py` `release` | `"<ver>"` | 中文站版本 |

- `conf.py` 里的 `version = "0.1"` 是主次版本，不跟补丁号，不要一起改。
- `docs/en/install.md`、`docs/zh/install.md` 各有两个**会被执行**的字面断言
  （`report["version"] == "<ver>"`、`pyqcu.__version__ == "<ver>"`）。
  `docs_examples` 真跑这些代码块，漏改即 FAIL。
- README 不写死版本：`README.md` 与 `README.zh-CN.md` 全文没有 `0.1.x`
  字面量，只写 “source archive”。新增文案不要引入版本号，否则会多一个漏改点。
- 两语 `overview.md`（英文 “This release is …” / 中文 “本版本为 …”）和
  `performance.md`（`pyqcu==<ver>` 与本地实测段落）还有散文版本引用，没有
  自动检查兜底，只能靠下一条的 grep 找全。
- 改完必须 grep 旧版本：
  `grep -rn "<oldver>" --include=*.py --include=*.toml --include=*.md --include=*.txt --include=*.in . | grep -v '^./logs/' | grep -v '^./data/' | grep -v '^./build/'`。
  当前唯一合法残留是 `cpp/kupl/qcu/CMakeLists.txt:98` 的 `VERSION 0.1.0`：
  那是 KUPL 原生库的 soname 版本（`SOVERSION 0`、产出
  `libqcu.so.0.1.0`），与 Python 发行版本解耦，两个版本都保持 0.1.0。

依赖分层（`pyproject.toml`，PyPI JSON `requires_dist` 实测一致）：

| 层 | 内容 | 说明 |
|---|---|---|
| 硬依赖 | `numpy>=1.23`、`torch>=2.0`、`mpi4py>=3.1`、`h5py>=3.0` | `pyqcu.torch`、lattice 工具、HDF5、MPI 布局在 import 时就要用 |
| extras `plot` | `matplotlib>=3.6` | 基准/收敛曲线 |
| extras `monitor` | `psutil>=5.9` | 内存/进程监控 |
| extras `all` | 上述两者 | 交互式工作便利 |

- `scipy`、`opt_einsum` 已从依赖和代码中删除：在 `pyqcu/`、`pyproject.toml`、
  `setup.py`、`README*`、`docs/en|zh` 上 grep 均 0 命中。不要再把它们加回
  extras。
- 可选依赖的导入方式（实测）：matplotlib 在函数内惰性导入
  （`pyqcu/tools/_bench_plots.py:61` 的 `_mpl()`、`pyqcu/torch/solver/_legacy_multigrid.py:1101`）；
  psutil 是模块级 `try/except ImportError` → `psutil = None`
  （`pyqcu/torch/tools/_hierarchical.py:24-27`）。两者缺装都不影响
  `import pyqcu`，判断“extras 是否必需”要看真实调用点而不是 import。

## 构建与落位

`setup.py` 的原生构建策略（对应代码位置见括号）：

- KUPL 始终尝试（`_build_kupl`, setup.py:136）：先探测 `bash`/`cmake`/`CXX`，
  再跑 `bash cpp/kupl/qcu/make.sh`；pip 安装时默认
  `KUPL_BUILD_TESTS=OFF`、`KUPL_ENABLE_NATIVE=OFF`、`KUPL_ENABLE_LTO=OFF`，
  要求产出 `cpp/kupl/qcu/build/libqcu.so`。
- CUDA/DCU 机会性（`_build_gpu`, setup.py:155）：`cuda` 需 `nvcc` 且宿主
  编译器能编译一个空程序，`dcu` 走 DTK 的 nvcc 兼容编译器；失败只
  `_warn`，不中断安装。
- `PYQCU_BUILD_BACKEND=kupl|cuda|dcu`（`pyqcu/_build_backend.py` 白名单，
  非法值 `ValueError`）。默认 `cuda`，但语义始终是**尝试**：默认值不等于
  要求 CUDA 成功。
- `PYQCU_SKIP_NATIVE_BUILD=1` 跳过所有原生构建；`PYQCU_REQUIRE_NATIVE=1`
  才会在后端没建成时硬失败；两者同时设置直接 `RuntimeError`。
- 构建进度写 `/dev/tty`（`_notify_progress`, setup.py:53），因为 pip 会吞掉
  构建 stdout；`/dev/tty` 不可用时回退 stderr。pip 长时间无输出不等于卡死。
- 其它可调变量：`PYQCU_KUPL_BUILD_DIR`、`PYQCU_KUPL_LIB`、
  `KUPL_BUILD_TESTS`、`KUPL_ENABLE_NATIVE`、`KUPL_ENABLE_LTO`
  （见 `docs/en/install.md` 变量表）。

安装后落位（0.1.1 原生 venv 实测，`data/pypi/venv-011-native/`）：

```
site-packages/pyqcu/kupl/libqcu.so
site-packages/pyqcu/cuda/libqcu.so
site-packages/pyqcu/cuda/qcu.cpython-312-x86_64-linux-gnu.so   # Cython 桥
```

- `_install_native_libraries()`（setup.py:187）把构建产物复制进包内，
  `package_data` 声明 `pyqcu.kupl: *.so`、`pyqcu.cuda: libqcu.so, qcu.pyi,
  qcu/*.pyi, qcu/*.pxd`。因此安装后不依赖源码树或 `LD_LIBRARY_PATH`。
- Cython 扩展用 `$ORIGIN` rpath（`runtime_library_dirs=["$ORIGIN"]` +
  `-Wl,-rpath,$ORIGIN`，setup.py:248），桥与 `pyqcu/cuda/libqcu.so` 同目录
  即可解析。
- KUPL 运行期查找顺序（`pyqcu/kupl/qcu.py:38-52`）：`PYQCU_KUPL_LIB`
  环境变量 → 包内 `pyqcu/kupl/libqcu.so` → 包内 `pyqcu/kupl/build/libqcu.so`
  → 源码树 `cpp/kupl/qcu/libqcu.so`；全落空时 `FileNotFoundError` 列出全部
  候选。
- 诊断入口 `pyqcu` / `python -m pyqcu` / `--json`
  （`pyqcu/__main__.py`）：打印 version、`pyqcu.__file__`、
  numpy/torch/h5py/mpi4py 版本、KUPL 库路径与导出符号数、CUDA 桥路径。
  `pyqcu --selftest` 跑 Torch/KUPL/CUDA 三层独立残差；缺原生后端为 SKIP，
  整体 PASS 才退出 0，`--json` 机器可读。
- 无原生库是合法安装状态（`PYQCU_SKIP_NATIVE_BUILD=1` 的
  `data/pypi/venv-011-nolib/`）：`import pyqcu` 成功、KUPL 报候选路径。
  不要用“import 成功”判断原生后端可用，要看 `python -m pyqcu`。

## 最小发布流程

1. 同步版本 4 处 + 两语 `install.md` 断言，grep 旧版本，确认只剩
   `cpp/kupl/qcu/CMakeLists.txt` 的 soname 版本。
2. 提交（只暂存本次文件，禁 `git add -A`/`git add .`），按
   [`pyqcu-tag`](../pyqcu-tag/SKILL.md) 打 tag；发布内容必须能由该 tag 复现。
3. `rm -rf dist && python -m build --sdist`。
4. 发布门：`python -m pyqcu.testing.release_check --offline`，要求 12/12。
5. `python -m twine check dist/pyqcu-<ver>.tar.gz`，要求 `PASSED`。
6. 只上传那一个文件（不要 `dist/*`，避免误带 wheel）：
   `TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_ZHANGXIN8069_API_KEY" python -m twine upload --non-interactive dist/pyqcu-<ver>.tar.gz`。
7. 轮询 `https://pypi.org/simple/pyqcu/` 直到出现 `pyqcu-<ver>.tar.gz`，再跑
   `release_check --online` 核验 PyPI/RTD 分支。
8. 干净 venv 复验（见下节矩阵），记录实际安装的版本、路径和本机数值。
9. 提交、推 GitLab `origin` 与 Gitee、核验四端 commit 一致（见“交接”）。
10. 写台账 `logs/v<YYYYMMDD>.txt`：产物大小/sha256、门禁输出、数值、外部阻塞项。

## 复验矩阵

| 检查 | 命令 / 判据 | 本次证据 |
|---|---|---|
| 版本一致 | `release_check` 的 `version` 项：pyproject 与 `pyqcu.__version__` 相同 | 0.1.1 通过 |
| sdist 内容 | 无 `pyqcu/testing/`、无 `.so/.o`、KUPL+CUDA 源都在、`SOURCES.txt` 无 `/home/`、含 `README.zh-CN.md` | 0.1.1：254 成员 / 87 个 `.py` / 0 testing / 0 `.so` / KUPL src 34 / CUDA src 33 |
| twine | `python -m twine check dist/pyqcu-<ver>.tar.gz` → `PASSED` | 0.1.0、0.1.1 均 PASSED |
| 双语代码块 | `release_check` 的 `bilingual blocks`：6 页 × en/zh 逐块相同 | 0.1.2 通过 |
| 文档示例 | `python -m pyqcu.testing.docs_examples --keep-going`：每块必须有 `assert` 或 `# docs-skip` | `ok=30 skipped=8 failed=0` |
| 文档站点 | Sphinx `-W` 构建 en 与 zh | 两站 exit=0 |
| 打包契约 | `pytest pyqcu/testing/dcu/test_install_selection.py pyqcu/testing/kupl/test_pure_kupl_install.py` | 9 passed |
| KUPL/API 回归 | `pytest pyqcu/testing/kupl/test_kupl_python.py pyqcu/testing/kupl/test_krylov_api.py pyqcu/testing/python/test_api_contract.py pyqcu/testing/python/test_krylov_api.py` | 29 passed |
| KUPL 数值 | 8x8x16x16 c64，26 it，真残差 ≈2.28e-7，≈0.13 s | 0.1.0：2.2785e-7/0.131 s；0.1.1 原生 venv：2.2791e-7/0.1237 s |
| CUDA 数值 | RTX 4060 sm_89，26 it，≈2.96e-7，≈0.04–0.05 s | 0.1.0：2.9641e-7/0.0497 s；0.1.1：2.9641e-7/0.0389 s |
| 验收线 | `max(atol, 2048*eps32) = 2.4414e-4` | 两条路径均远低于 |
| PyPI JSON | HTTP 200；`info.project_urls` 含 6 个键；`description` 含 `## Repositories` | 0.1.1 实测通过 |
| simple 索引 | 页面含 `pyqcu-<ver>.tar.gz` | 0.1.1 在列 |
| 干净安装 | `pip install --no-cache-dir --index-url https://pypi.org/simple pyqcu==<ver>` | 0.1.0 clean venv 装成功；0.1.1 在 clean venv 由 0.1.0 升级后两条后端都 PASS |
| 源码树外断言 | 在 `/tmp` 等目录跑 `python -m pyqcu`，`location` 必须含 `site-packages` | 0.1.1 指向 `venv-011-native/.../site-packages/pyqcu/__init__.py` |
| 依赖自洽 | `python -m pip check` → `No broken requirements found.` | 0.1.1 通过 |
| 重装 | `pip install --force-reinstall --no-deps --no-build-isolation ./dist/pyqcu-<ver>.tar.gz`（可加 `PYQCU_SKIP_NATIVE_BUILD=1` 快速验打包） | 0.1.1：no-native 与 native 两个 venv 各做 install #1/#2（force-reinstall）均成功 |
| 卸载重装 | `pip uninstall -y pyqcu` 后从 PyPI 重装 | 0.1.1 从 PyPI sdist 重建成功，诊断仍 0.1.1，`pip check` 干净 |
| 升级/降级 | 已装旧版时装新版必须写 `pyqcu==<ver>` 或 `--upgrade` | 裸 `pip install pyqcu` 只报 `already satisfied (0.1.0)`（`data/pypi/pypi-upgrade-011.log`） |
| 离线安装 | `PIP_NO_INDEX=1 PYQCU_SKIP_NATIVE_BUILD=1 pip install --no-index --find-links wheelhouse pyqcu==<ver>`（wheelhouse 含运行时依赖 + setuptools/wheel） | 0.1.8：233 MB CPU-only wheelhouse（torch 2.15.0+cpu）在无系统 site-packages 的全新 venv 中完整安装，build isolation 也走本地 wheel；Torch PASS，KUPL/CUDA SKIP，`pip check` 干净。CUDA wheelhouse 未测试 |
| extras | `pip install "pyqcu[<extra>]==<ver>"`（或 `--dry-run` 先看解析） | 0.1.1：`pyqcu[all]` 装出 matplotlib 3.11.2 + psutil 7.2.2，`pyqcu.tools.mpl_cjk_setup` 可调用 |
| CLI 自检 | `pyqcu --selftest` / `python -m pyqcu --selftest --json` | 0.1.2：no-native `torch=PASS, kupl=SKIP, cuda=SKIP`；native 三层 PASS |

### 0.1.2 安装态自检经验

- `pyqcu` console script 由 `[project.scripts] pyqcu = "pyqcu.__main__:main"`
  提供；`python -m pyqcu` 等价。
- `pyqcu --selftest`：Torch 4^4 Clover BiCGStab；KUPL/CUDA 4^4 Wilson
  BiCGStab；每层用独立全格点残差；缺原生库为 SKIP，整体 PASS exit 0，
  FAIL exit 2。
- CUDA Wilson 当前没有 iteration getter，`iterations=None` 是合法结果，
  不是失败；不要 `int(None)`。
- 原生 CUDA C++ 用 `printf` 写 stdout，且缓冲可能在 fd 恢复后才 flush。
  CLI 在 fd 1 层捕获，恢复前调用 `ctypes.CDLL(None).fflush(None)`；PASS 时
  丢弃，FAIL 时把截断日志放进 `native_output`，保证 `--json` 可解析。
- 0.1.2 实测：no-native `torch 78 it/1.246e-6`；native `torch 78 it/1.246e-6`、
  `kupl 17 it/2.48e-7`、`cuda residual 2.199e-7 (iterations None)`；
  `pytest test_cli_selftest.py` 3 passed。

### 0.1.3 bugfix 经验

- `--lattice` 非偶数必须给 argparse 错误或 JSON FAIL，不能 traceback；
  `run_all(include=())` 必须拒绝，`include="torch"` 要按单层处理。
- Torch 自检要同时固定 gauge/source 与 x0，否则残差/迭代不可复现。
- 原生库损坏（dlopen OSError / ImportError）是 **FAIL**，只有
  `FileNotFoundError`（KUPL 缺失）或 `ModuleNotFoundError`/无 bridge（CUDA）
  才是 SKIP；否则会把安装缺陷伪装成“未安装”。
- 原生 C `printf` 的缓冲可能在 fd 恢复后才 flush；捕获 stdout 后要在
  restore 前 `ctypes.CDLL(None).fflush(None)`。
- `KUPL_BUILD_TESTS` 在 Python 构建中必须强制 `OFF`：sdist 已 prune
  `cpp/kupl/qcu/tests`，继承 `ON` 会让 CMake 找不到测试源。
- LaTeX docstring 的反斜杠用 raw string，避免 fresh/-OO 编译的
  `SyntaxWarning: invalid escape sequence`。
- 0.1.3 实测：无 CXX/NVCC 安装 exit 0；`KUPL_BUILD_TESTS=ON` 安装成功；
  只读 site-packages 全 PASS；损坏 KUPL/CUDA 均 FAIL exit 2；
  `pytest test_cli_selftest.py` 7 passed。

### 0.1.4 MPI-aware CLI 经验

- 自检和诊断在 `mpirun` 下不能每个 rank 各打一份 JSON。做法：`_selftest`
  在 MPI size>1 时把 KUPL/CUDA 标为 SKIP（reason 写明单 rank 要求），报告带
  `mpi_size`/`mpi_rank`；`__main__` 只让 rank 0 输出，其他 rank 静默退出。
- `mpirun -np 2 pyqcu --selftest --json` 实测：一个 JSON 对象、Torch PASS、
  KUPL/CUDA SKIP、exit 0；`mpirun -np 2 pyqcu --json` 也是一个对象。
- 无效格点的 JSON 错误也要 rank-0 gating，并补齐 `version`/`lattice`/`layers`
  字段，避免每 rank 各打一份导致不可解析。
- `pyqcu --version`/`--help` 仍由 argparse 在每个 rank 输出，属于命令行本身
  行为；文档明确“自检单进程运行”。
- **非零 rank 不能无条件 `return 0`**：静默只代表不打印；返回值仍必须是
  `0/2` 的真实状态，否则 `mpirun` 会把失败的 Torch 层掩盖成成功。0.1.5
  专门修了这一点，并加了 `_selftest_exit_code` 单测。
- RTD 对默认 `Python-urllib` User-Agent 返回 HTTP 403（curl 正常）。
  `release_check` 的 `_fetch` 已改用浏览器 UA；修正后 0.1.5 的 `--online`
  为 18/18。

### 0.1.6 安装警告可见性

- pip 在成功时**吞掉 stderr**；`_warn` 只写 stderr 会让“KUPL/CUDA 未构建”
  在成功安装里完全不可见。修法：与进度条一样写 `/dev/tty`，无 tty 才回退
  stderr。
- 验证要伪终端：`script -qec '<pip install ...>' /dev/null`，然后 grep
  `[pyqcu build]`；普通重定向到日志只能验证 fallback，不能验证用户可见性。
- 非零 MPI rank 的 FAIL 原因写 stderr，stdout 仍只有 rank 0 的一份 JSON；
  退出码仍必须 0/2。
- `https://pypi.org/simple/pyqcu/` 的不同 CDN 边缘会短暂缺新版本；不要用
  一次 `curl` 判发版失败。`release_check` 现在轮询 12 次 × 5 s，0.1.6
  实测首次 attempt=1 即命中，但保留轮询以防边缘滞后。

### 0.1.7 自检格点下限

- 0.1.7 起自检拒绝任何小于 4 的维度（`extents must be at least 4`）。
  0.1.6 在 `2 2 2 2` 上实测 CUDA 原生解残差 42.7，而 Torch/KUPL 正常；
  CUDA 2^4 未验证，接口层面直接拒绝比给出误导性 FAIL 更清晰。
- 发布验证仍用 `4 4 4 4` 默认值或 `8 8 16 16`；后者 0.1.6 实测
  Torch 115 it/5.48e-7、KUPL 26 it/1.71e-7、CUDA 2.10e-7。
- 项目级 `https://pypi.org/pypi/pyqcu/json` 可能比版本级 JSON 和 simple
  index 慢；`release_check` 现在同时轮询 latest metadata。0.1.7 `--online`
  为 19/19（新增 `PyPI latest metadata`）。

### 0.1.8 MPI 信息输出去重

- `mpirun -np N pyqcu --version/--help` 在 0.1.7 会打印 N 份。0.1.8 在
  argparse 前预扫描 `OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`SLURM_PROCID` 等，
  非零 rank 静默返回 0，不 import mpi4py。
- 干净安装复验要用 `env -u PYTHONPATH -u PYQCU_KUPL_LIB` 再 `source env.sh`，
  确保命中 venv 的 console script；PyPI simple index 仍可能需轮询 1-2 次。
- Python 包层矩阵：3.10.22/3.11.18/3.13.18 的 sdist `--no-deps` 安装、
  `import pyqcu`、`--version`、`compileall` 全过；3.12 由干净 PyPI venv 覆盖；
  3.14 系统缺 `ensurepip`，用 `setup.py build` + `PYTHONPATH=build/lib` 验证。
  cp314 的 torch/mpi4py/h5py/numpy 轮子存在，但完整依赖只在 3.12 端到端跑过。

### 0.1.10 诊断与 JSON 错误

- 0.1.10 的诊断路径不再 import torch/mpi4py：依赖版本走
  `importlib.metadata`，MPI size/rank 走 launcher 环境变量，后端可用性用
  `find_spec`/包内库路径；`pyqcu --json` 更快且不触发 OpenMPI 初始化。
- 非整数 `--lattice` 在 `--json` 下返回 `status=FAIL`/exit 2，人类模式走
  `parser.error`；CLI 测试 19 项。
- 事故：0.1.9 曾被未被杀死的 nudge 子进程从半成品工作树抢先上传；0.1.10
  是审查后的超集。停 watcher 必须杀进程组，不能只杀 watcher PID。
- 0.1.11 诊断准确性：bridge 的 `find_spec` 成功不代表可加载；必须检查
  companion `libqcu.so`（`PYQCU_CUDA_LIB` 或 bridge 同目录/上级目录）。缺库时
  `pyqcu --json` 返回 `cuda.available=False`，`--selftest` 仍会 FAIL。
- 0.1.11 sdist 606,633 B，sha256 109517de…498c75；CLI 20、全量 148/4。

发布门 12 项（`release_check --offline` 顺序）：version、sdist exists、sdist no
testing、sdist no binaries、sdist native sources、sdist no home paths、sdist
Chinese README、twine check、bilingual blocks、docs examples、sphinx en、
sphinx zh。`--online` 追加 PyPI JSON、PyPI project links、PyPI repositories
section、PyPI simple index、RTD en/zh-cn。

## 坑与判据

1. **PEP 517 隔离构建不把源码根放进 `sys.path`。** setup.py 里的
   `from pyqcu._build_backend import ...` 在隔离环境会 `ImportError`。
   处理：setup.py:37-39 先 `os.chdir(HOME)` 再 `sys.path.insert(0, str(HOME))`。
   判据：`--no-build-isolation` 能装、默认隔离装失败时先看这里。
2. **`build_ext.finalize_options()` 之后替换 `self.extensions` 要补
   `_needs_stub = False`。** `NativeBuildExt.run()` 用 cythonize 结果覆盖
   扩展列表（setup.py:283-288），setuptools 之后会访问 `_needs_stub`，
   缺了报 `AttributeError`。
3. **cythonize 必须留在 build 阶段。** `_cythonized_extensions()`
   （setup.py:256）只在 `build_ext.run()` 里调用；放到 `setup()` 调用点会让
   没装 Cython、或没有 GPU 的机器在读取元数据阶段就失败。
4. **Extension 源用相对路径。** 用 `pyqcu/cuda/qcu/qcu.pyx` +
   `build_dir="build/cython"`；绝对路径会把构建机路径写进 egg-info，触发
   `sdist no home paths` 失败。
5. **sdist 必须自带源码与构建脚本。** `MANIFEST.in` 必须包含
   `cpp/kupl/qcu` 源与 `make.sh`、`cpp/cuda/qcu` 的 `src/include/python`、
   `*.pyx/*.pxd/*.pyi`、根 `env.sh`/`build-nv.sh`/`build-dcu.sh`/`install.sh`、
   `README.zh-CN.md`；同时 `prune` 掉 `pyqcu/testing`、`build`、`dist`、`data`、
   `docs`、`logs`、`refer`、`skills`，并 `global-exclude *.so *.so.* *.o`。
6. **`include-package-data = false`**（pyproject.toml:63）：package_data
   只认 setup.py，新增数据文件必须同时改 `package_data`，否则不进包。
7. **`README.zh-CN.md` 只随包分发，不做 PyPI 描述。** `readme = "README.md"`
   是唯一渲染源；中文 README 由 `MANIFEST.in` 的 `include` 和
   release_check 的 `sdist Chinese README` 保障，别把它设成 `readme`。
8. **不要在一个已发布版本上追改。** 0.1.0 缺 3 个项目链接就只能发 0.1.1；
   0.1.1 里的 RTD 文案顺带一起发，避免再发 0.1.2。
9. **裸 `pip install pyqcu` 不会升级。** 已装 0.1.0 的环境只会输出
   `Requirement already satisfied ... (0.1.0)`；复验必须
   `pyqcu==<ver>` 或 `--upgrade`。
10. **无原生库不算安装失败。** KUPL 构建失败只降级为纯 Torch 路径；只有
    `PYQCU_REQUIRE_NATIVE=1` 才硬失败。判据看 `python -m pyqcu` 的
    `kupl` 行，不要看 import 是否成功。
11. **双语代码块与可执行断言是强耦合。** 只改 `docs/en` 不改 `docs/zh`
    会被 `bilingual blocks` 抓；只改 `conf.py` 不改 `install.md` 会被
    `docs examples` 抓。
12. **soname 不是发行版本。** `cpp/kupl/qcu/CMakeLists.txt` 的
    `VERSION 0.1.0` 决定 `libqcu.so.0.1.0` 目标名和 soname，跟着 Python
    版本一起改会牵动 `make.sh`/`setup.py` 的产物路径判断。
13. **刚上传后首次安装可能拿到旧版本。** 0.1.1 第一次在干净 venv 里
    `pip install pyqcu==0.1.1` 仍装到 0.1.0（CDN 边缘还没刷新），重试才升级
    成功并卸载 0.1.0。判据：装完立刻 `python -m pyqcu` 看版本号，不要只看
    pip 退出码；必要时 `--no-cache-dir` 后重试。

## 回滚

- 同版本不可覆盖：PyPI 拒绝重复文件名，也没有“替换文件”的接口。任何修正
  都是新版本（0.1.0 → 0.1.1 就是这样产生的）。
- yank 只能在网页上做（项目页 → Manage → Yank），效果是隐藏该版本、阻止
  新解析，但**不删除文件、不释放版本号**；yank 后仍不能重传同版本。
- 不发 wheel，所以没有“撤 wheel”这条退路；原生 `.so` 不可移植，临时拼一个
  wheel 只会把构建机架构固化成平台标签。
- 上传了错误文件且尚未有人安装：yank 该版本并立刻发修正版本，不要试图复用
  版本号。
- PyPI 描述是版本快照：README/项目链接的文案改动需要发新版本；RTD 站点独立
  部署，文档修正不需要发版。

## 交接

本次发布（0.1.1 元数据修正）的交接事实：

- 0.1.0 与 0.1.1 都已上架，PyPI 侧只有 sdist；0.1.1 已含 6 个项目链接。
- 0.1.1 的改动必须与代码/文档在同一提交中提交，并推送到 GitLab `origin` +
  Gitee；推送后核验本地/GitLab/Gitee/GitHub 四端同一 commit。台账并入
  `logs/v20261109.txt`。
- RTD：项目 slug `pyqcu`（`py-qcd` 只是 PyQCD 因 `pyqcd` 被占用的历史替代，
  不是本项目命名约定）；`en/latest` 与 `zh-cn/latest` 在 0.1.0 提交推送后
  均 HTTP 200，标题为 0.1.0；0.1.1 提交推送后应显示 0.1.1。配置只有
  `docs/en/.readthedocs.yaml` 与 `docs/zh/.readthedocs.yaml`（`version: 2`、
  ubuntu-24.04、python 3.12、各自 `sphinx.configuration` 与 requirements），
  仓库根**不设** `.readthedocs.yaml`；两个文件在 RTD 网页上分别指定。
- RTD 跟踪的是 GitHub 仓库；GitHub 是 Gitee 的自动镜像，推 Gitee 后 GitHub
  自动更新，RTD 再跟随 GitHub。判据：`git ls-remote` 四端 commit 一致才算
  同步完成；GitHub API 的 `mirror_url` 未登记该镜像关系，不要用它判断。
- RTD 构建 `success=false` 且 `duration=0` 的判据是：tracked revision 里没有
  `.readthedocs.yaml`（0.1.0 时期 GitHub main 停在 `8847f84`，配置推上去
  之前每次都秒失败）。先 `git ls-remote` 确认 GitHub 上有
  `docs/<lang>/.readthedocs.yaml`，再触发重建。
- 已验证路径：`b63e779` 推 Gitee 后 GitHub 镜像约 10 s 更新；0.1.4 的日志提交实测
  约 5 分钟才同步。按分钟级轮询，单次超时不判失败。RTD 自动构建成功，
  `en/latest` 与 `zh-cn/latest` 均 HTTP 200（当时标题 0.1.0）；0.1.1 推送后要
  按页确认标题变为 0.1.1。zh-cn 翻译条目已存在，push-triggered build 不需要
  RTD API token。
- 本次未创建标签；发布交付要求 0.1.1 提交与双远端推送完成，tag 仅在显式要求时
  创建。

Git 交付规则（与根 `AGENTS.md` 一致）：只暂存本任务文件，禁用 `git add -A`/
`git add .`；提交消息末尾加 `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`；
推送必须同时到 GitLab `origin` 与 `git@gitee.com:zhangxin8069/PyQCU.git`，
并核验 local/gitlab/gitee/github 四端同一 commit；不改写、不删除已推送标签。
