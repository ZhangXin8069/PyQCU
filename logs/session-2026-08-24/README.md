# session-2026-08-24 — 无人值守会话验证资产（bug31–37）

8 个回归验证脚本，对应 `logs/fix-report-2026-08-24.md` 的方法论闭环。
运行方式：`source ./env.sh && python logs/session-2026-08-24/<脚本>`（工作目录=仓库根；
MPI 脚本加 `mpirun -np N --oversubscribe`）。

| 脚本 | 验证内容 | 预期结果 |
|------|---------|---------|
| `r1_baseline.py` | 10 组件基线（lattice/dslash/solver/smear/h5py/多线程 MG，CUDA） | 8/10 PASS；2 FAIL 仅为 with_data 参考 HDF5 缺失（OSError 层，见 examples/data/AGENTS.md）；MG 项 ~6.4s |
| `c2_solver_smoke.py` | 求解器族七电池：bistabcg/fgmres/mr(γ5†)/cacg(正规方程)/multishift_cg/tr_lanczos(A†A, ncv=48 需 ~3000 matvec)/verify_nullvecs | 7/7 PASS；tr_lanczos 约 60s |
| `c4_mpi_smoke.py` | MPI 主套件 parity/stout/bistabcg/lattice × CUDA | mpirun -np 2 与 -np 4 均 4/4 PASS |
| `c5_final.py` | Wuppertal 三重不变量：nstep≥1 防护、自由场(U=I)常数场不动点、默认参数白噪声收缩 | 4/4 PASS（不动点 <1e-4，ratio<1.0） |
| `c6_gold.py` | Wuppertal MPI 黄金判据：root 全格串行 vs 子格并行 | np=2 rel≈5.1e-08 / np=4 ≈3.9e-08 PASS（bug34 守护） |
| `c7_schur_smoke.py` | stencil 构建非零性+确定性+apply_stencil（中格子 [16,16,16,8]，W=10） | ALL PASS，hop_nn≈2.2e-02（bug35 守护；勿用 W>维度或 [8,8,8,8] 以下格子） |
| `c8_galerkin.py` | Galerkin 恒等式 A_c ≈ PᵀSP（verify_nullvecs + 块结构 lonv + op.matvec_parity 口径） | rel_diff ≈5.6e-07 PASS（S 勿显式乘 κ²） |
| `c9_equiv.py` | 双实现逐元素等价：build_stencil_local(34s) vs build_stencil 全格参考(396s) | 三张 stencil 张量 rel ≤1.1e-06 EQUIVALENCE PASS |

## 关联

- 权威记录：`logs/fix-report-2026-08-24.md`
- 会话日志：`.auto.2026-08-24-02-37-52.log`（不入库）
- 标签区间：`56e6ee6 (dev84_2_2)` → `1025a90`（bug31–36、test16、dev85、bug37）
