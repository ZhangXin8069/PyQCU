# AGENTS.md — examples/pyquda

PyQCU vs PyQuda-0.3.2（QUDA 1.1.0）双进程对比套件（结果 + 性能，含作图）。

- 一键运行：`bash run_all.sh 8x8x8x16 [tol]`（三阶段：run_pyqcu → run_pyquda → compare）。
- 双进程隔离（dev87 F2）：pyqcu 与 pyquda 不得同进程；数据经 h5 交换，
  存放于 `examples/data/pyquda_cmp/<lat>/`（`.gitignore *.h5` 不入库，可再生）。
- 维度排布以实际代码为准：pyqcu `[c,c,d,xyzt]`/`[s,c,xyzt]`（奇偶切 **t**）；
  pyquda `[d,q,tzyx,c,c]`/`[q,tzyx,s,c]`（奇偶切 **x**）；棋盘格约定一致。
- 归一化锚定：κ=1/(2m+8)；quda 侧 mass 归一化 + 直调 invertQuda（绕 ×2κ'）；
  解满足 x_pyqcu = (m+4)·x_quda。
- 详细文档见 skills/pyquda/SKILL.md（维度表 / API / 反模式 / 实测基线）。