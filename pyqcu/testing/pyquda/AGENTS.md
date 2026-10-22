# AGENTS.md — pyqcu/testing/pyquda

PyQCU vs PyQUDA（QUDA 1.1.0）双进程对比套件（结果 + 性能，含作图）。
版本口径以实际 `pip show PyQUDA`/`pyquda.__version__` 为准；2026-10-23 当前环境为
0.10.54，`refer/git-rep/PyQUDA/` 固定在 `v0.10.54-1-g138d8ad`。历史 0.3.2 API 和
基线只能作为时点证据，升级后必须重新验证。

- 一键运行：`bash run_all.sh 8x8x8x16 [tol] [csw]`（三阶段：
  run_pyqcu → run_pyquda → compare）。两侧都显式使用同一 `csw`，
  Clover onsite 系数为 `-kappa*csw/8`。
  脚本会先加载 `qcu/strict/quda_comparison/quda_env.sh`，优先选择仓库
  `data/quda-sm70-install`，确保 QUDA 的 CUDA 架构与 GPU 匹配。
- 双进程隔离（dev87 F2）：pyqcu 与 pyquda 不得同进程；数据经 h5 交换，
  存放于 `pyqcu/testing/data/pyquda_cmp/<lat>/`（`.gitignore *.h5` 不入库，可再生）。
- 维度排布以实际代码为准：pyqcu `[c,c,d,xyzt]`/`[s,c,xyzt]`（奇偶切 **t**）；
  pyquda `[d,q,tzyx,c,c]`/`[q,tzyx,s,c]`（奇偶切 **x**）；棋盘格约定一致。
- 归一化锚定：κ=1/(2m+8)；当前 PyQUDA 侧设置 `QUDA_MASS_NORMALIZATION`，通过
  `WilsonDirac`/`CloverWilsonDirac.invert` 求解。历史 0.3.2 的直接 `invertQuda` 与
  `(m+4)` 缩放记录保留为时点证据，不作为 0.10.54 的默认结论。
- 详细文档见 skills/pyqcu-testing-pyquda/SKILL.md（维度表 / API / 反模式 / 实测基线）。
