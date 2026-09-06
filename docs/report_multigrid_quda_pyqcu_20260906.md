# QUDA 与 PyQCU Strict MultiGrid：从 Clover/Gauge、P/R 到 MATPC 与外层求解的可复现对照

## 任务与受众

本报告面向需要从零复现 MultiGrid setup、算子应用和正式 benchmark 的开发者与研究人员。目标有三项：

1. 用同一套记号解释 QUDA 参考实现和 PyQCU Strict 实现中的细层 Clover/Gauge、null vector、aggregate、$P/R$、Galerkin 粗算子、$X/Y/Yhat$、奇偶 Schur 和外层求解器。
2. 把 full、asymmetric/symmetric Schur、coarse-PC、legacy MultiGrid、Strict MultiGrid、V/W/F/K cycle 以及 MR/CG/Chebyshev/Schwarz/FGMRES 等不同路径分开，避免把不同算子误当成同一个实现。
3. 在 V100、c64、$16\times32\times32\times48$、odd--odd MATPC 协议下复现正式结果，并保存每次外层迭代的残差与平均时间。

报告中的“确证”来自本次源码读取或本次测试产物；“推断”只表示由公式和代码组合得到、尚未逐元素对照的结论；“未验证”表示本次没有把它升级为事实。

## 结论摘要

QUDA 与 PyQCU Strict 的共同核心不是“把 Gauge 场缩小后继续计算”，而是对每一层的有效 Dirac 算子做局部 Galerkin 投影。对层级 $\ell$，最重要的关系是

$$
D_{\ell+1}=R_\ell A_\ell P_\ell,
\qquad R_\ell=P_\ell^\dagger,
$$

其中 $A_\ell$ 可以是原始 full-field 算子，也可以是先用 onsite block 左预条件后的 $\widehat D_\ell=X_\ell^{-1}D_\ell$。严格路径在粗层保存完整 coarse lattice；奇偶裁剪只发生在 MATPC、fine Schur prepare/reconstruct 和调用 $P/R$ 的边界。

细层 Clover/Gauge 只存在于 level 0。进入 level 1 后，$X$ 是 Galerkin onsite block，$Y$ 是四个方向的有效 hopping；若启用 coarse-PC，则还保存

$$
\widehat Y_\mu^{\rm f}=X^{-1}Y_\mu^{\rm f},
\qquad
\widehat Y_\mu^{\rm b}=Y_\mu^{\rm b}X^{-\dagger},
$$

以及 $X^{-1}$。后续层继续对这些有效 block 做 Galerkin 投影，不能重新读取或“下采样”原始 SU(3) Gauge。

本次正式复现为 fair pass：

| 侧 | 5 次 steady 外层迭代 | median solve-only | median 每次外层迭代 | full-op 真残差 |
|---|---:|---:|---:|---:|
| PyQCU Strict | `11,11,11,11,11` | `2.041653 s` | `185.605 ms` | `3.6013e-7` |
| QUDA | `37,37,37,37,37` | `2.094841 s` | `56.617 ms` | `7.3030e-7` |

因此，在这个固定的 V100/WSL2/编译配置和输入 bundle 上，PyQCU 的总 solve-only 时间为 QUDA 的约 `0.9746` 倍，报告的 `speedup_pyqcu_over_quda=1.02605`。这不是跨设备或 upstream QUDA 的普遍性能承诺：QUDA 使用了启用 `DEV87_REDUCE_SYNC` 的 WSL2 构建，且 setup 阶段加载了共享 null-vector 文件。

## 主问题与验收标准

### 记号、布局与奇偶

物理坐标按 $(x,y,z,t)$ 排列，格点奇偶为

$$
p(x)=\left(x+y+z+t\right)\bmod 2,
\qquad e=p=0,
\qquad o=p=1.
$$

细层 Wilson/Clover 自由度为 $4\times3=12$；本次每个 aggregate 使用 $n_v=12$ 个 null vectors，Clover/Wilson coarse spin 固定为 $2$，所以 coarse dof 为

$$
E=n_v\times N_s^{\rm coarse}=12\times2=24.
$$

这三个概念必须分开：物理 parity、Transfer 的 site subset、数组的 compact index。odd compact 数组的第 $j$ 个元素不是物理坐标奇偶的第 $j$ 个点；它是经过固定 checkerboard 映射后的索引。

### 通过标准

| 验收项 | 标准 | 本次证据 |
|---|---|---|
| 正式协议 | 格点、mass、$n_v$、block、coarse spin、target parity、层数与 formal profile 完全一致 | `data/strict_vs_quda_formal_20260906.json` 的 `config_hash=654b...2478` |
| 输入一致 | trace 与无 trace benchmark 的 `bundle_hash` 相同 | `bundle_hash=8c866bd0...a6ca3d31` |
| 迭代详情 | 每侧 2 warmup + 5 steady；逐外层迭代有残差记录 | `data/strict_trace_detailed_20260906.csv`，250 行 |
| 正确性 | 5 次 steady 的 full-op true residual 均小于 $5\times10^{-6}$ | PyQCU `3.6013e-7`，QUDA `7.3030e-7` |
| CPU smoke | Strict/Transfer/Galerkin/MATPC/FGMRES 基础断言全部通过 | `19 passed`，约 `22.5 s`，结果见 `data/strict_fast_cpu_20260906.json` |
| 性能 | 只用无 trace benchmark 的 steady 样本 | `data/strict_vs_quda_formal_20260906.json` |
| 图表 | 生成 residual 对数图和每次外层迭代时间柱状图 | `data/strict_trace_detailed_20260906.svg` |

## 方法、设置与证据

### 1. 细层 Clover/Gauge 算子

把细层算子写成

$$
D_0=A_0-\kappa H_0,
$$

其中 $A_0$ 是每个站点上的 $12\times12$ Clover onsite block，$H_0$ 是只连接相反 parity 的 Wilson hopping。以未归一化的 Wilson hopping 表示，

$$
(H_0\psi)(x)=\sum_{\mu=0}^{3}\left[
(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
 +(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)
\right].
$$

这里 forward link 位于源点 $x$；backward gather 使用 $x-\hat\mu$ 处的 link dagger。Clover 只作用于站点内 spin/color，不改变 parity，因此 block matrix 为

$$
D_0=
\begin{pmatrix}
A_e&-\kappa H_{eo}\\
-\kappa H_{oe}&A_o
\end{pmatrix}.
$$

QUDA 的 `DiracClover::M` 通过 `ApplyWilsonClover(...,-kappa,...)` 同时应用 hopping 与 Clover；`DiracClover::Clover` 单独应用 Clover，证据为 `refer/git-rep/quda/lib/dirac_clover.cpp:36`、`:51`、`:58`。PyQCU 的正式 full-op true residual 由 `give_wilson + give_clover - rhs` 计算，benchmark 入口为 `examples/qcu/dev87/bench_strict_vs_quda.py:1546`。

### 2. Fine odd/even Schur 与 prepare/reconstruct

令 $p$ 为目标 parity，$q=1-p$。非对称 Schur 补为

$$
S_p^{\rm asym}=A_p-\kappa^2H_{pq}A_q^{-1}H_{qp}.
$$

若将目标 parity 的 onsite inverse 也吸收进去，则得到

$$
S_p^{\rm sym}=I-\kappa^2A_p^{-1}H_{pq}A_q^{-1}H_{qp}.
$$

对 full RHS $b=(b_p,b_q)$，非对称消元和恢复为

$$
\begin{aligned}
b_p^{S}&=b_p+\kappa H_{pq}A_q^{-1}b_q,\\
S_p^{\rm asym}x_p&=b_p^{S},\\
x_q&=A_q^{-1}\left(b_q+\kappa H_{qp}x_p\right).
\end{aligned}
$$

对称版本先再施加 $A_p^{-1}$ 到 Schur RHS。QUDA `DiracCloverPC::prepare` 和 `reconstruct` 的实现对应 `refer/git-rep/quda/lib/dirac_clover.cpp:223` 与 `:251`；其中 `M` 的两种 asymmetric/symmetric 组合对应 `:174` 与 `:187`。PyQCU 的 compact Schur 参考路径在 `pyqcu/solver/_quda_multigrid.py:1762`，通用 coarse parity Schur 在 `:1893`。

奇偶关系的工程含义是：fine MATPC 只保存一个 compact parity，但被消去的 parity 仍需要参与一次 hopping 和一次 Clover inverse；粗层 full geometry 不能因此被删掉。

### 3. Null vector、aggregate 与 $P/R$

对 block size $b=(b_x,b_y,b_z,b_t)$，fine 坐标映射为

$$
X_\mu=\left\lfloor x_\mu/b_\mu\right\rfloor,
\qquad
\Lambda_c=(L_x/b_x,L_y/b_y,L_z/b_z,L_t/b_t).
$$

每个 coarse site $X$ 与每个 coarse spin block 独立进行 block Gram--Schmidt。Wilson/Clover 的 $4\to2$ spin 映射为 $s_c=\lfloor s_f/2\rfloor$；PyQCU 在 `QudaTransfer.spin_map` 中对应 `pyqcu/solver/_quda_multigrid.py:441`，block 正交化入口为 `:446`。重复 CGS 的数学形式为

$$
v_j\leftarrow b_j-\sum_{i<j}v_i\langle v_i,b_j\rangle_{\mathcal A_X},
\qquad
v_j\leftarrow v_j/\sqrt{\langle v_j,v_j\rangle_{\mathcal A_X}}.
$$

正交后基张量可记为

$$
V_f(s,c,s_c,j,x),
$$

其中 $j=0,\ldots,n_v-1$。QUDA kernel 先通过 `geo_map` 找到 coarse site，再执行 spin map 与 fine-color rotation，证据为 `refer/git-rep/quda/include/kernels/prolongator.cuh:65` 与 `:85`；限制 kernel 逐项使用 $V^\dagger$，证据为 `refer/git-rep/quda/include/kernels/restrictor.cuh:87` 与 `:116`。

对 coarse 场 $\phi$，延拓和限制为

$$
\begin{aligned}
(P\phi)_{s,c}(x)&=\sum_jV_{s,c,s_c,j}(x)\phi_{s_c,j}(X(x)),\\
(R\psi)_{s_c,j}(X)&=\sum_{x\in\mathcal A_X}\sum_{s,c}V_{s,c,s_c,j}(x)^*\psi_{s,c}(x),\\
R&=P^\dagger.
\end{aligned}
$$

PyQCU 的 full-field `P`/`R` 分别在 `pyqcu/solver/_quda_multigrid.py:538` 和 `:610`；`to_qcu_blocked` 在 `:573` 把逻辑布局转换为 C++ 所需的 `[E,e,Xc,bx,Yc,by,Zc,bz,Tc,bt]`。奇偶视图只在 `prolong_parity`/`restrict_parity`，即 `:660`/`:673`，输出 compact fine parity 或读取一个 compact fine parity，但 coarse 输出仍为完整 coarse lattice。

### 4. Galerkin 粗算子：先定义算子，再取矩阵 block

通用 full-field Galerkin 是

$$
D_{\ell+1}=R_\ell D_\ell P_\ell.
$$

Strict Clover-PC 路径采用左预条件输入

$$
A_\ell=\widehat D_\ell=X_\ell^{-1}D_\ell,
\qquad
D_{\ell+1}=R_\ell\widehat D_\ell P_\ell.
$$

这一区别决定了 Clover 和粗层 onsite 的位置：fine $A_\ell^{-1}$ 先进入被投影的算子，而不是在 Galerkin 投影后再随意乘到最终结果上。PyQCU 在 `QudaMultigrid.setup` 选择 Strict + `direct_pc` 时构造 `_LeftPreconditionedOperator`，证据为 `pyqcu/solver/_quda_multigrid.py:2715`；`build_strict_galerkin` 对 full nearest-neighbour support 做 batched projection，证据为 `pyqcu/tools/_strict_galerkin.py:594` 与 `:678`。

固定一个 coarse source site $X$，只需探测它自身和八个相邻 coarse sites 的 block。令 $\mathcal T(X)$ 是这些 displacement，粗 block 为

$$
B_{\Delta}(X)=\left(P^\dagger A_\ell P\right)_{X,X+\Delta},
\qquad \Delta\in\{0,\pm\hat x,\pm\hat y,\pm\hat z,\pm\hat t\}.
$$

其中 $B_0=X_{\ell+1}$，正/负方向 block 组成 $Y^{\rm f}$ 和 $Y^{\rm b}$。PyQCU 的 `build_strict_galerkin` 逐 site batch 置入 blocked basis、调用 batch matvec，再用 `einsum` 计算 $V^\dagger A V$，对应 `pyqcu/tools/_strict_galerkin.py:657`、`:700`、`:713`。

QUDA 的 coarse-op kernel 将同一收缩拆成可并行的步骤：$UV$、$AV$、$V^\dagger UV$、coarse Clover、reverse-Y、diagonal/rescale 和 storage conversion。该计算图的源码级说明见 `refer/git-rep/quda/docs/analy_multigrid_impl_20260824.tex:168`，粗算子入口为 `refer/git-rep/quda/lib/coarse_op.in.cpp:8`。

### 5. $X/Y/Yhat$ 的 storage 和矩阵乘法顺序

粗层未预条件算子可以写成

$$
D_c=X+H_c,
$$

其中符号正负取决于该层把 $-\kappa$ 吸收到 $Y$ 还是保留在 kernel 的 `kappa`；比较实现时必须同时看 storage 和 kernel 的符号。

Strict runtime 需要

$$
X,\quad X^{-1},\quad Y^{\rm f}_\mu,\quad Y^{\rm b}_\mu,
\quad \widehat Y^{\rm f}_\mu,\quad \widehat Y^{\rm b}_\mu.
$$

对 forward storage，

$$
\widehat Y^{\rm f}_\mu(X)=X^{-1}(X)Y^{\rm f}_\mu(X).
$$

对 backward storage，物理 link 存在邻点 $X-\hat\mu$，因此

$$
\widehat Y^{\rm b}_\mu(X-\hat\mu)
=Y^{\rm b}_\mu(X-\hat\mu)X^{-\dagger}(X-\hat\mu).
$$

QUDA `computeYhat` 明确分成 backward 与 forward 两个分支：backward 使用 `Y * X^{-\dagger}`，forward 使用 `X^{-1} * Y`，证据为 `refer/git-rep/quda/include/kernels/coarse_op_preconditioned.cuh:60` 和 `:104`。QUDA `DiracCoarsePC::createCoarseOp` 的注释直接说明递归粗化使用 `Yhat` 而不是 `Y`，证据为 `refer/git-rep/quda/lib/dirac_coarse.cpp:628`。

PyQCU 在 `_finish_strict_galerkin` 中先形成 `X_inv`，再把 backward link roll 到 `q-\mu` storage，最后分别计算 `X_inv @ forward` 和 `backward_storage @ X_inv(q-\mu)^\dagger`，证据为 `pyqcu/tools/_strict_galerkin.py:489`、`:515`。`QudaCoarseOperator.to_qcu_strict_assets` 把 `preconditioned_links` 和 `(X,X^{-1})` 打包，证据为 `pyqcu/solver/_quda_multigrid.py:1269`。

这里的 `Yhat` 不是“对已经算出的粗 dslash 结果再乘一次逆”。如果把 backward 也误写为 $X^{-1}Y^{\rm b}$，在平凡 Gauge 上可能仍能收敛，但非平凡 Clover 情形下矩阵乘法顺序已经改变。

### 6. 粗 dslash、Gauge/Clover 的层间消失

粗层的 dslash 读取的是有效 link-like block，而不是 SU(3) Gauge：

$$
(D_c z)(X)=X(X)z(X)
 +\sum_{\mu}\left[
Y^{\rm f}_\mu(X)z(X+\hat\mu)
 +Y^{\rm b\dagger}_\mu(X-\hat\mu)z(X-\hat\mu)
\right].
$$

QUDA `dslash_coarse.cuh` 的注释和 `applyDslash` 对应 `refer/git-rep/quda/include/kernels/dslash_coarse.cuh:126`、`:141`；onsite block 应用 `applyClover` 对应 `:267`。在 coarse-PC 路径中同一 dslash kernel 的 link 输入替换为 `Yhat`，`DiracCoarsePC::Dslash` 证据为 `refer/git-rep/quda/lib/dirac_coarse.cpp:506`。

所以层间数据流是

$$
(U_0,C_0,B_0)
\longrightarrow V_0
\longrightarrow (X_1,Y_1,X_1^{-1},Yhat_1)
\longrightarrow V_1
\longrightarrow (X_2,Y_2,X_2^{-1},Yhat_2).
$$

原始 Gauge/Clover 只负责 fine operator 及第一层投影；后续层只作用于上一层的有效矩阵 block。把 $X/Y$ 当作 SU(3) Gauge 会丢失 coarse spin、null-vector 颜色空间和 Clover 消元顺序。

### 7. MATPC：奇偶只在应用边界裁剪

令 $\widehat D=I+\widehat H$，其中 $\widehat H$ 的每一次作用都会改变 coarse parity。目标 parity $p$ 的 MATPC 算子为

$$
M_p=I-\widehat H_{pq}\widehat H_{qp}.
$$

应用过程必须是：

| 单列伪代码：coarse MATPC |
|---|
| 输入：完整 coarse link/onsite 资产、目标 compact parity 场 $x_p$、$p\in\{0,1\}$。 |
| 把 $x_p$ embed 到完整 coarse geometry 的 parity-$p$ 位置，另一 parity 置零。 |
| 用 `Yhat` 做第一次 hopping，得到 parity-$q$ 场 $u_q=\widehat H_{qp}x_p$。 |
| 把 $u_q$ embed 回完整 coarse geometry 的 parity-$q$ 位置。 |
| 用 `Yhat` 做第二次 hopping，得到 parity-$p$ 场 $v_p=\widehat H_{pq}u_q$。 |
| 输出 $M_px_p=x_p-v_p$。 |
| 保持 $X/Y/Yhat` 的 full coarse geometry；只对输入和输出视图做 parity crop。 |

PyQCU 的 `QudaMatPCOperator.apply` 逐步实现上述过程，证据为 `pyqcu/solver/_quda_multigrid.py:1539`、`:1589`；其严格 assets 的 `apply_matpc` 也明确写出 `I-Hhat_pq Hhat_qp`，证据为 `pyqcu/tools/_strict_galerkin.py:378`。QUDA coarse PC 的 even-even/odd-odd 分支在 `refer/git-rep/quda/lib/dirac_coarse.cpp:530`，odd--odd 的两次 dslash 为 `:545`--`:557`。

### 8. QUDA 的完整 setup 与 solve 伪代码

以下单列长表把 QUDA 当前实现的逻辑串起来。`V` 是每一层的 aggregate basis；`D` 是 residual operator；`Dhat` 是 coarse-PC operator；`S` 是配置的 smoother；`G` 是外层/粗层 Krylov solver。

\begin{table}[htbp]
\centering
\caption*{QUDA 单列伪代码：从输入到 full solution}
\small
\begin{tabular}{@{}l@{}}
输入：$U_\mu$、fine Clover $A_e,A_o$、$\kappa$、null-vector 数 $n_v$、block size、层数 $L$、MATPC type、smoother、coarse solver。\\
定义 $D_0=A-\kappa H(U)$；hopping 连接相反 parity，Clover 保持 parity。\\
若配置 `vec_load`/`vec_infile`，读取已转换的 $B^{(0)}$；否则按 `setup_type` 生成随机、test-vector、inverse、CG、CA-CG 或 MG setup 向量。\\
按 `geo_bs` 建立 `geo_map`，使每个 fine site 指向一个 coarse aggregate；同时准备 coarse-to-fine 的连续访问映射。\\
对每个 aggregate、每个 coarse spin block 做重复 block Gram--Schmidt，得到 $V_\ell$；Wilson/Clover 使用 fine spin $4\to$ coarse spin $2$。\\
令 $P_\ell$ 用 $V_\ell$ 把 coarse color/spin 展开到 fine，令 $R_\ell=P_\ell^\dagger$ 把 fine field 投影回 coarse。\\
根据 `smoother_solve_type` 和 `coarse_grid_solution_type` 选择 full 或 MATPC site subset；此时设置 parity 视图，不能修改 coarse 资产的 full geometry。\\
若当前层是 direct Clover coarsening，计算 $D_{\ell+1}=R_\ell D_\ell P_\ell$ 的 $X/Y$ block；若当前层已经是 Clover-PC，则把 $Yhat_\ell$ 作为下一层 coarsening 的 link 输入。\\
形成 onsite $X_{\ell+1}$、batch inverse $X_{\ell+1}^{-1}$，并按 `need_bidirectional` 决定独立 forward/backward links 还是使用适用条件受限的 reverse。\\
若下一层使用 coarse-PC，计算 forward $\widehat Y^f=X^{-1}Y^f$；计算 backward $\widehat Y^b=Y^bX^{-\dagger}$，并交换需要的 halo。\\
创建 residual Dirac、smoother Dirac、presmoother、postsmoother、粗 solver；`MG::createSmoother` 对应 `refer/git-rep/quda/lib/multigrid.cpp:273`，`createCoarseDirac` 对应 `:342`。\\
生成下一级 null vectors：可以从独立 setup 生成，也可以 `R_\ell B_\ell` 后重新正交；`MG::generateNullVectors` 对应 `refer/git-rep/quda/lib/multigrid.cpp:1275`。\\
对一次 level-$\ell$ correction，先执行 $x\leftarrow S_\ell^{\nu_{pre}}(x,b)$；formal 配置中 $\nu_{pre}=1$。\\
若 smoother 返回的 residual 与 residual operator/solution type 一致，直接复用；否则显式计算 $r=b-D_\ell^{res}x$。\\
限制 $b_{\ell+1}=R_\ell r_\ell$，把 coarse initial solution 清零或按允许的 warm policy 设置。\\
若 $\ell<L-1$，递归执行 `coarse_solver`；若 $\ell=L-1$，执行 bottom GCR/CA-GCR/其他配置的粗求解器。\\
延拓 $e_\ell=P_\ell e_{\ell+1}$，更新 $x_\ell\leftarrow x_\ell+e_\ell$。\\
执行 $\nu_{post}$ 次 post-smoothing；formal 配置中 $\nu_{post}=1$。\\
若外层要求 full solution，调用 fine `reconstruct` 恢复被消去 parity；若是 MATPC solution，则保留 compact parity。\\
外层 GCR/FGMRES 取 $r_0=b-Dx_0$，每轮以 $z_j=M_{MG,j}^{-1}v_j$ 调用一次 MG preconditioner，再计算 $w=D z_j$。\\
对 $w$ 与已有 Krylov basis 正交化，形成 Arnoldi/Hessenberg 或 GCR 小系统；应用 Givens/最小残差更新。\\
若外层估计残差满足容差，更新 full solution 并独立计算 $\lVert b-Dx\rVert/\lVert b\rVert$；否则到 restart 后重算 true residual，继续。\\
\end{tabular}
\end{table}

QUDA 高层执行关系由 `MG::operator()` 的 `refer/git-rep/quda/lib/multigrid.cpp:1131`、`createCoarseSolver` 的 `:564` 和 null-vector setup 的 `:1275` 共同决定。当前快照的 `createCoarseSolver` 实际主路径是 `VCYCLE` 或 `RECURSIVE`；枚举中存在 F/W 不代表这条 factory 分支已经实现了 F/W 的完整运行语义。

### 9. PyQCU Strict 的完整 setup 与 solve 伪代码

PyQCU 同时保留 legacy `_multigrid.py` 和 Strict `_quda_multigrid.py`。本节只描述 formal 使用的 Strict 路径。

\begin{table}[htbp]
\centering
\caption*{PyQCU Strict 单列伪代码：full coarse setup + fine odd Schur solve}
\small
\begin{tabular}{@{}l@{}}
输入：full fine Gauge、Clover even/odd block 及其 inverse、canonical full null vectors、$p=o$、block size、coarse spin $2$。\\
先把 canonical null vectors 变为 block layout `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`，保持 C-order、dtype 和共轭约定。\\
用 `QudaTransfer` 对每个 aggregate/chiral block 做重复 CGS，得到 $V$；生成 `fine_to_coarse` 与 `coarse_to_fine`。\\
在 Strict + `direct_pc` 下把当前 full operator 包装为 $\widehat D_\ell=X_\ell^{-1}D_\ell$；Strict setup 不是把第一层 coarse field 直接 checkerboard crop。\\
用 batched site probes 或 colored probes 构造 $R\widehat D_\ell P$；检查 fine support 只落在 source aggregate 与八个相邻 aggregate。\\
将 displacement-zero block 记为 $X$，把六/八个 nearest-neighbour block 按 forward/backward storage 放入 $Y$；当前 formal Strict runtime 只接受可表示的 nearest-neighbour support。\\
对每个 coarse site batch 做 batched inverse 得到 $X^{-1}$；构造 forward $Yhat=X^{-1}Y$。\\
把 backward link roll 到 $q-\hat\mu$ storage，再构造 $Yhat^b=Y^bX^{-\dagger}(q-\hat\mu)$。\\
将 fine blocked $V$、各层 `preconditioned_links` 和 $(X,X^{-1})$ 绑定到 C++ runtime；raw $Y$ 只在 setup/诊断需要时保留。\\
构造 `CudaSchurOp`，在 fine odd parity 上应用 $S_o=A_o-\kappa^2H_{oe}A_e^{-1}H_{eo}$；`applyInitQcu` 与该实例的 `_SET_INDEX_` 绑定。\\
外层使用右预条件 FGMRES：$r=b-S_ox$，$v_0=r/\|r\|$。\\
每轮计算 $z_j=M_{MG,j}^{-1}v_j$，其中 $M_{MG}^{-1}$ 依次执行 fine MR、$R$、递归 coarse correction、$P$、fine MR。\\
用 fine Schur operator 计算 $w=S_oz_j$，对 $w$ 做 Arnoldi 正交化；由于 MG 可能随层级、精度或 guard 改变，使用 flexible basis 保存 $z_j$。\\
每 `restart_effective=4` 轮做 true residual 检查；正式外层最多 1000 轮，容差为 $10^{-6}$。\\
收敛后用 fine parity reconstruct 恢复 full solution，并独立用 full Wilson/Clover operator 计算 true residual。\\
solve 结束后保留 runtime slot 生命周期：Strict sequence 为 setup $\to$ `CudaSchurOp` $\to$ bind assets $\to$ strict init $\to$ V-cycle/FGMRES $\to$ strict end $\to$ release。\\
\end{tabular}
\end{table}

PyQCU 的 Strict 层级模式与 legacy setup operator 分离：`setup_operator="schur"` 是旧 compact odd-Schur setup 选项，`hierarchy_mode="strict"` 才选择 Strict full-coarse hierarchy，证据为 `pyqcu/solver/_quda_multigrid.py:1971`、`:2089`。Strict setup 选择左预条件 coarse operator 的代码为 `:2715`；运行时资产绑定和显存账本在 `pyqcu/cuda/_strict_multigrid.py:117`、`:336`。

### 10. 不同 cycle、smoother 和外层 solver 的区别

#### 10.1 V/W/F/recursive/K

设 $C_\ell$ 为一次 coarse correction，$S_{pre/post}$ 为平滑器：

$$
\begin{aligned}
C_\ell^{V}&=S_{post}P_\ell C_{\ell+1}^{V}R_\ell S_{pre},\\
C_\ell^{W}&=S_{post}P_\ell C_{\ell+1}^{W}R_\ell P_\ell C_{\ell+1}^{W}R_\ell S_{pre},\\
C_\ell^{F}&=S_{post}P_\ell C_{\ell+1}^{F}R_\ell P_\ell C_{\ell+1}^{V}R_\ell S_{pre}.
\end{aligned}
$$

| 路径 | 子问题调用 | 适用含义 |
|---|---|---|
| V-cycle | 每层一次 child solve | 正式 QUDA `VCYCLE` 与 PyQCU 默认 baseline |
| W-cycle | 同一层两次 child solve | 粗空间较弱时更强，但粗层成本近似增加 |
| F-cycle | setup/早期层更积极 | 需要明确 factory 是否真的进入该分支 |
| recursive | coarse solver 本身再包一层 MG | QUDA 当前 formal 可运行的递归粗 solver 语义 |
| K-cycle | child 内部做短 Krylov/变预条件 | PyQCU 有 FGMRES/K-cycle 风格路径；不能把它当作 QUDA 当前 VCYCLE 的同义词 |

#### 10.2 MR、CG、Chebyshev、Schwarz

MR 平滑一步为

$$
v=Ar,
\qquad
\alpha=\frac{\langle v,r\rangle}{\langle v,v\rangle},
\qquad
x\leftarrow x+\alpha r,
\qquad
r\leftarrow r-\alpha v.
$$

MR 不要求 $A$ Hermitian，适合 coarse Schur 或方向相关 `Yhat`；代价是只沿当前 residual，不积累长期 Krylov 方向。

CG 递推为

$$
\rho_k=\langle r_k,r_k\rangle,
\quad
\alpha_k=\rho_k/\langle p_k,Ap_k\rangle,
\quad
\beta_k=\rho_{k+1}/\rho_k,
\quad
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

它要求实际 operator 在当前内积下 Hermitian positive definite；“symmetric Schur”这个名字本身不能证明这一点。PyQCU 当前把 guarded BiStabCG/FGMRES 用作更一般的 coarse solver。

Chebyshev 用谱区间构造多项式，优点是减少每步全局判断，风险是非 Hermitian Schur 上的谱界只是启发式。Schwarz 则是局部块求解器：

$$
M_{AS}^{-1}=\sum_I P_IA_I^{-1}R_I,
$$

乘性 Schwarz 还依赖 block 顺序和通信同步；它是平滑/预处理外壳，不是与 MR/CG 相同的 scalar recurrence。

#### 10.3 BiCGStab、GCR、FGMRES 与 CA-GCR

BiCGStab 适合非 Hermitian，单轮状态小，但需要 shadow residual 和 $\rho,\alpha,\omega$ breakdown guard。GCR/FGMRES 保存 image basis；右预条件 FGMRES 的关键是

$$
z_j=M_j^{-1}v_j,
\qquad
w_j=Az_j,
$$

而非把固定的 $M^{-1}$ 假定成线性常算子。正因为 MultiGrid 可能含递归 cycle、混合精度、warm state 和 guard，PyQCU Strict 采用 flexible outer path。CA-GCR 把多个 residual power 组成 block，减少 global reduction，但增加 Gram conditioning 和 workspace。

## 关键结果与误差

### 正式运行协议

| 参数 | 值 |
|---|---|
| GPU | Tesla V100-SXM2-32GB，UUID `be23...f0a8`，compute capability 7.0 |
| 软件 | PyTorch `2.10.0+cu128`，CUDA `12.8`，PyQUDA `0.10.54` |
| 格点 | $(L_x,L_y,L_z,L_t)=(16,32,32,48)$ |
| 物理 | $m=0.05$，$\kappa=1/(2m+8)=0.1234567901234568$，seed 42，周期边界 |
| 精度 | c64，complex bytes=8，real=float32 |
| hierarchy | 2 levels，block $(2,2,2,2)$，$n_v=12$，coarse spin=2，coarse dof=24 |
| parity | target parity=1，QUDA `QUDA_MATPC_ODD_ODD` |
| outer | restarted right FGMRES/GCR，requested restart=16，effective restart=4，tol=$10^{-6}$ |
| smoothing | $\nu_{pre}=\nu_{post}=1$；coarse maxiter=200，coarse tol=$3\times10^{-3}$ |
| 统计 | 2 次 warmup，5 次 steady；median + MAD；warmup 不计入 steady |

### 每次 steady solve 的外层迭代与时间

| side | solve 0 | solve 1 | solve 2 | solve 3 | solve 4 | median (s) | MAD (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyQCU | 11 / 2.021366 | 11 / 2.110346 | 11 / 2.015191 | 11 / 2.041653 | 11 / 2.068720 | 2.041653 | 0.026462 |
| QUDA | 37 / 2.094076 | 37 / 2.094841 | 37 / 2.082558 | 37 / 2.123275 | 37 / 2.108543 | 2.094841 | 0.012283 |

表内每格为“迭代数 / solve 秒数”。PyQCU 的总迭代数为 QUDA 的 $11/37=0.2973$，但单次外层迭代约为 QUDA 的 $185.605/56.617=3.278$ 倍；总时间因此只相差约 `2.54%`。

### 每次外层迭代残差

以下是 5 次 steady trace 中一致的首个 steady 曲线；完整 5 次记录保存在 CSV/JSON。PyQCU 列是 Arnoldi least-squares estimate，QUDA 列是 GCR iterated residual。二者的标量定义不同，最终收敛仍看 full-op true residual。

| $k$ | PyQCU reported residual | QUDA reported residual |
|---:|---:|---:|
| 0 | 1.0000000e+00 | 1.0000000e+00 |
| 1 | 9.3650706e-02 | 2.6259840e-01 |
| 2 | 1.6313400e-02 | 1.0671660e-01 |
| 3 | 2.8122286e-03 | 5.6699110e-02 |
| 4 | 6.4573408e-04 | 3.2547150e-02 |
| 5 | 2.1028987e-04 | 2.1907580e-02 |
| 6 | 6.2526138e-05 | 1.2670970e-02 |
| 7 | 2.1975775e-05 | 8.4522200e-03 |
| 8 | 7.4830368e-06 | 5.3343160e-03 |
| 9 | 3.2432135e-06 | 3.8731680e-03 |
| 10 | 1.2322430e-06 | 2.4618190e-03 |
| 11 | 4.0650434e-07 | 1.7779130e-03 |
| 12 | — | 1.1777290e-03 |
| 13 | — | 8.7806670e-04 |
| 14 | — | 6.0349930e-04 |
| 15 | — | 4.3366300e-04 |
| 16 | — | 3.1065080e-04 |
| 17 | — | 2.3763540e-04 |
| 18 | — | 1.6512580e-04 |
| 19 | — | 1.2454290e-04 |
| 20 | — | 9.0326790e-05 |
| 21 | — | 6.8630000e-05 |
| 22 | — | 5.0507290e-05 |
| 23 | — | 3.7877360e-05 |
| 24 | — | 2.8157010e-05 |
| 25 | — | 2.2275490e-05 |
| 26 | — | 1.6195150e-05 |
| 27 | — | 1.2465930e-05 |
| 28 | — | 9.3341650e-06 |
| 29 | — | 7.4462140e-06 |
| 30 | — | 5.4595650e-06 |
| 31 | — | 4.2656100e-06 |
| 32 | — | 3.2030910e-06 |
| 33 | — | 2.5857410e-06 |
| 34 | — | 1.8816930e-06 |
| 35 | — | 1.5172770e-06 |
| 36 | — | 1.1199800e-06 |
| 37 | — | 9.1188520e-07 |

PyQCU 在第 11 轮已经达到内部 estimate $4.07\times10^{-7}$；QUDA 继续到第 37 轮达到 $9.12\times10^{-7}$。这些数只用于解释外层行为；两侧 full-op 真残差分别是 `3.6013e-7` 和 `7.3030e-7`，均低于 `5e-6` gate。

### Setup、cache 与显存

| 指标 | PyQCU Strict | QUDA | 解释 |
|---|---:|---:|---|
| setup 秒数 | 14.777423 | 364.369869 | PyQCU 为 schema-v2 cache hit；QUDA 读取共享 null-vector 并建立 MG runtime |
| PyQCU cache | hit，schema v2 | 不适用 | manifest、metadata、stats、3 个 tensor digest 全部验证通过 |
| PyQCU owned assets | 4,076,863,488 B ≈ 3.797 GiB | 未由 PyQUDA 暴露 | fine blocked $V$、onsite pair、Yhat 等 runtime assets |
| fused workspace | 509,607,936 B ≈ 0.475 GiB | native workspace 未暴露 | PyQCU 使用 $(2m+5)B_f+2B_c$，预算 512 MiB 内 |
| setup device-wide peak | 11,219,046,400 B ≈ 10.917 GiB | 21,014,843,392 B ≈ 19.564 GiB | 设备级峰值可能含 allocator/native allocation |
| first-solve device-wide peak | 11,722,362,880 B ≈ 10.917 GiB | 24,529,670,144 B ≈ 22.845 GiB | 首次 lazy workspace 单独统计，不计入 steady timing |

cache 的逻辑资产大小约 3.797 GiB，PyQCU 的 `strict_setup_stats` 显示 coarse sites=49,152、coarse dof=24、colored column batch=12、projection site batch=4、44 次 operator calls。该 setup 统计写入 `strict_vs_quda_formal_20260906.json`，不能用“cache hit setup”解释为在线构造成本。

## 风险与未验证项

1. **粗层非平凡 storage 的逐元素证明仍未完成。** 当前已有 synthetic `Y/Yhat`、Galerkin、MATPC、33-point 与 CUDA primitive 断言，formal solve 也通过；但还没有把非平凡 Gauge/Clover 下每个方向、每个 coarse site、每个 parity 的 `X^{-1}`、forward `Yhat`、backward `Yhat` 与 QUDA storage 逐元素导出对照。尤其要检查 backward storage 在 $q-\hat\mu$、gather 时 dagger 和周期边界 wrap 的一致性。
2. **trace 残差的标量不等价。** PyQCU 的曲线是 FGMRES Arnoldi estimate，QUDA 的曲线是 GCR iterated residual；只能比较下降趋势和最终 full-op true residual，不能逐点相减后宣称算法误差。
3. **性能结论受平台限制。** 本次 QUDA 使用 WSL2 `DEV87_REDUCE_SYNC=1` 的构建，库归约行为与健康 Linux/CUDA 平台不同；PyQCU 使用当前 Torch/CUDA runtime。结论限定在本机固定协议。
4. **Strict 当前边界。** Strict runtime 对多 rank、逐层异 dtype、标准 staggered/KD、奇数 coarse extent 和未支持的非 nearest-neighbour strict stencil fail-closed；不能把 legacy MPI/mixed-precision 结果转述为 Strict 已支持。
5. **setup 成本不可直接比较。** PyQCU 本次使用已经验证的 HDF5 runtime cache hit，QUDA 使用已经转换的 canonical null-vector bundle。两侧 setup 策略不同，报告只把 solve-only steady timing 作为性能比较。
6. **full residual 才是物理停机证据。** Schur/MATPC 内部 residual、Arnoldi/GCR estimate 和 full Wilson/Clover residual 是三个不同对象；代码修改后必须至少保留最后一项。

## 下一步与请求

优先级最高的是新增一个非平凡 Gauge/Clover coarse asset dump，对 `X`、`X^{-1}`、四个 forward/backward `Yhat` 的 storage site、parity、dagger 和周期边界做 QUDA/PyQCU 对照。测试应使用同一 canonical null vectors，并固定 DeGrand--Rossi gamma basis、coarse dof=24 和 odd target parity。

随后可对 `strict_hopping_parity_kernel` 做 block size 128/256 的小范围 benchmark，记录 kernel 次数、寄存器/occupancy、外层迭代、full true residual 与 steady time；任何优化都必须保持本报告的 formal protocol 和 `config_hash`。

## 来源与附录

### 本次产物

- [正式无 trace benchmark](../data/strict_vs_quda_formal_20260906.json)
- [trace benchmark 原始结果](../data/strict_trace_benchmark_20260906.json)
- [逐迭代详细 JSON](../data/strict_trace_detailed_20260906.json)
- [逐迭代 CSV](../data/strict_trace_detailed_20260906.csv)
- [残差与时间对比 SVG](../data/strict_trace_detailed_20260906.svg)
- [自动生成摘要](../data/strict_trace_detailed_20260906.md)
- [CPU smoke 结果](../data/strict_fast_cpu_20260906.json)
- [详细分析器](../examples/qcu/dev87/analyze_strict_vs_quda_detailed.py)

### 复现实验命令

```bash
source ./env.sh
source examples/qcu/dev87/quda_env.sh

# 诊断 trace：开启日志，仅用于逐迭代残差和事件顺序
python3 -B examples/qcu/dev87/trace_strict_vs_quda.py \
  --output data/strict_trace_20260906.json \
  --plot data/strict_trace_20260906.svg \
  --benchmark-output data/strict_trace_benchmark_20260906.json \
  --reference data/strict_vs_quda_formal_20260902.json \
  --repeats 5 --timeout 1800

# 正式 timing：无 trace 日志，作为性能真值
python3 -B examples/qcu/dev87/bench_strict_vs_quda.py \
  --profile formal --side both --cache-expect hit \
  --quda-nullvec-prefix data/L16x32x32x48_nvec12_quda \
  --quda-nullvec-manifest data/L16x32x32x48_nvec12_quda.conversion.json \
  --output data/strict_vs_quda_formal_20260906.json

# 校验哈希、逐轮迭代数、残差曲线并生成 CSV/JSON/SVG/Markdown
python3 -B examples/qcu/dev87/analyze_strict_vs_quda_detailed.py
```

### 关键源码索引

| 对象 | PyQCU | QUDA |
|---|---|---|
| aggregate、spin map、CGS、P/R | `pyqcu/solver/_quda_multigrid.py:349`、`:446`、`:538`、`:610` | `refer/git-rep/quda/include/kernels/prolongator.cuh:65`、`:85`；`restrictor.cuh:87` |
| strict Galerkin site-batch | `pyqcu/tools/_strict_galerkin.py:594`、`:657`、`:713` | `refer/git-rep/quda/lib/coarse_op.in.cpp:8`、`coarse_op_kernel.cuh` |
| X inverse 与 Yhat | `pyqcu/tools/_strict_galerkin.py:489`、`:515` | `refer/git-rep/quda/include/kernels/coarse_op_preconditioned.cuh:48` |
| coarse dslash/Clover | `pyqcu/solver/_quda_multigrid.py:1400` | `refer/git-rep/quda/include/kernels/dslash_coarse.cuh:126`、`:267` |
| MATPC | `pyqcu/solver/_quda_multigrid.py:1539`；`pyqcu/tools/_strict_galerkin.py:378` | `refer/git-rep/quda/lib/dirac_coarse.cpp:530` |
| fine Clover Schur | `pyqcu/solver/_quda_multigrid.py:1762` | `refer/git-rep/quda/lib/dirac_clover.cpp:174`、`:223`、`:251` |
| MG recursion/smoother/solver | `pyqcu/solver/_quda_multigrid.py:1971`、`:2663` | `refer/git-rep/quda/lib/multigrid.cpp:273`、`:342`、`:564`、`:1131` |
| strict CUDA runtime | `pyqcu/cuda/_strict_multigrid.py:117`、`:336` | QUDA runtime assets in `DiracCoarse/DiracCoarsePC` |

### 物理与工程校验

量纲上 $m$、$\kappa$、Clover coefficient、粗层 matrix block 都是无量纲格点单位；$P/R$ 不应改变字段的物理量纲，只改变表示空间。极限检查包括：

- $U_\mu=I$ 时 forward/backward link 的周期邻居关系应退化为平移加 spin projector；
- $P^\dagger P=I$ 只在每个 aggregate/chiral block 的局部正交意义下成立，而非粗格全局单位矩阵；
- $R=P^\dagger$ 时，$\langle P\phi,\psi\rangle=\langle\phi,R\psi\rangle$ 是 transfer 的首要不变量；
- backward `Yhat` 若漏掉 dagger、错用 $X^{-1}$ 左乘或错放到 $X$ 而非 $X-\hat\mu$，在非平凡 Clover/周期边界上会破坏 Galerkin 与 MATPC 等价性；
- full solution 的正确性必须回到 $\lVert b-D_0x\rVert/\lVert b\rVert$，不能用 compact Schur 的内部估计替代。
