# PyQCU 数值线性代数、格点费米子作用量与多重网格粗算子总报告

**身份：物理+代码**　　**日期：2026-09-11**　　**状态：理解与文档化完成候选稿**

本报告以公式、可执行伪代码和源码证据为主，统一说明下列三组对象：

1. 数值线性代数：`native/legacy/strict-MultiGrid`、Galerkin、full/asymmetric/symmetric-Schur、coarse-PC、MR、CG、Chebyshev、Schwarz、FGMRES、BiCG、BiCGStab、GCR、flexible-GCR、CA-GCR、Arnoldi、SAP、CGS、QR；
2. 格点 QCD 费米子作用量：Wilson、Clover-improved-Wilson、Twisted-mass（含 non-degenerate pairs）、Twisted-mass with a clover term、Staggered、Improved-staggered（asqtad/HISQ）、Domain-wall（4-d/5-d preconditioned）、Möbius、Overlap；
3. 粗层算子：QUDA `coarse_op`、PyQCU 旧/legacy/compact/33-tensor Schur 粗算子，以及 strict-QCU 的 full `X/Y/Yhat` Galerkin 粗算子。

“实现”表示在本库或随库保存的 QUDA 快照中找到源码入口；“参考”表示有理论或 QUDA 源码可对照，但 PyQCU 当前入口没有该功能；“未验证”表示本报告没有把它升级为运行事实。行号以本次工作区快照为准。

## 跳转目录（按首字母）

- [0. 记号、证据与接口边界](#0-记号证据与接口边界)
- [A. Arnoldi](#a-arnoldi)
- [A. Asymmetric Schur](#a-asymmetric-schur)
- [A. Asqtad/HISQ](#a-asqtadhisq)
- [B. BiCG](#b-bicg)
- [B. BiCGStab](#b-bicgstab)
- [C. CA-GCR](#c-ca-gcr)
- [C. CG](#c-cg)
- [C. CGS](#c-cgs)
- [C. Chebyshev](#c-chebyshev)
- [C. Clover-improved-Wilson](#c-clover-improved-wilson)
- [D. Domain-wall](#d-domain-wall)
- [F. FGMRES](#f-fgmres)
- [F. Full Schur 与 full coarse](#f-full-schur-与-full-coarse)
- [G. Galerkin](#g-galerkin)
- [G. GCR 与 flexible-GCR](#g-gcr-与-flexible-gcr)
- [L. Lanczos 与多移位 CG](#l-lanczos-与多移位-cg)
- [M. Möbius](#m-möbius)
- [M. MR](#m-mr)
- [M. MultiGrid：native、legacy/compact、strict](#m-multigridnativelegacycompactstrict)
- [O. Overlap](#o-overlap)
- [Q. QR、CGS 与局部基](#q-qrcgs-与局部基)
- [Q. QUDA coarse-op](#q-quda-coarse-op)
- [S. SAP](#s-sap)
- [S. Schwarz](#s-schwarz)
- [S. Staggered fermions](#s-staggered-fermions)
- [S. Strict-QCU coarse-op](#s-strict-qcu-coarse-op)
- [S. Symmetric Schur](#s-symmetric-schur)
- [T. Twisted-mass（含 non-degenerate pairs）](#t-twisted-mass含-non-degenerate-pairs)
- [T. Twisted-mass with a clover term](#t-twisted-mass-with-a-clover-term)
- [W. Wilson](#w-wilson)
- [X. 统一比较、验收与来源](#x-统一比较验收与来源)

## 0. 记号、证据与接口边界

### 0.1 线性代数与格点记号

细格点坐标写作 (x=(x,y,z,t))，(\hat\mu) 是四个欧氏方向的单位向量，
(p(x)=(x+y+z+t)\bmod2) 是偶奇性。内积统一为

$$
\langle u,v\rangle=\sum_x u(x)^\dagger v(x),
\qquad \|u\|_2=\sqrt{\langle u,u\rangle}.
$$

PyQCU 的规范场为 `[3,3,4,Lx,Ly,Lz,Lt]`，费米子场为 `[4,3,Lx,Ly,Lz,Lt]`，展平自由度
(N_sN_c=12)；时空轴永远是最后四轴。QUDA 快照通常以 parity-ordered、tzyx 或 QDP
布局保存数据，比较时必须先做维度和奇偶转换，不能直接比较裸指针。

细层 Wilson/Clover 块矩阵写为

$$
D=\begin{pmatrix}A_e&B_{eo}\\B_{oe}&A_o\end{pmatrix},
\qquad A_p=I+C_p,
$$

其中 (B_{pq}) 是含规范链接和 (1\pm\gamma_\mu) 投影的 hopping。粗层不再使用 SU(3)
链接，而使用自由度为 (E=n_vN_s^{\rm coarse}) 的矩阵块 (X,Y^{\rm f},Y^{\rm b})。

### 0.2 证据等级

| 标记 | 含义 | 本报告的处理 |
|---|---|---|
| **实现** | 本库源码或 `refer/git-rep/quda` 有直接入口 | 给出文件和函数/行号 |
| **参考** | 理论或 QUDA 有实现，本库未暴露等价入口 | 明确写“参考”，不冒充 PyQCU 功能 |
| **未验证** | 只完成源码/公式核对，没有本轮设备运行 | 不写成性能或正确性结论 |

### 0.3 运行期 ABI 不变量

`params` 是 `int32[58]`，`argv` 是 `float[7]` 或相应实数精度，`set_ptrs` 是
`int64[100]`。`applyInitQcu` 建立 `LatticeSet` 和 scratch，操作之间必须递增
`params[define._SET_INDEX_]`，最后调用 `applyEndQcu`。Strict 资产按 transition 保存
`Yhat`、`(X,X^{-1})`、blocked null vectors；raw (Y) 只在 setup/诊断需要时保留。

### 0.4 覆盖矩阵：本次任务列出的每个名字

| 名称 | 报告章节 | PyQCU/QCU 现状 |
|---|---|---|
| native-MG | [M.MultiGrid](#m-multigridnativelegacycompactstrict) | `pyqcu/solver/_multigrid.py` 的 Python V-cycle |
| legacy/compact-MG | 同上 | `_quda_multigrid.py` 的 `hierarchy_mode="legacy"`，可用 compact odd Schur |
| strict-MG | 同上 | `QudaStrictMultigrid` 与 `applyMultigridStrict*Qcu` |
| Galerkin | [G.Galerkin](#g-galerkin) | `RDP`、strict batched Galerkin |
| full/asymmetric/symmetric Schur | [F/S](#f-full-schur-与-full-coarse)、[A/S](#a-asymmetric-schur) | fine Clover 与 coarse-PC 路径分开实现 |
| coarse-PC | [F.Full Schur](#f-full-schur-与-full-coarse) | (I-\widehat H_{pq}\widehat H_{qp}) |
| MR | [M.MR](#m-mr) | Python `_mr.py`、QCU coarse smoother |
| CG | [C.CG](#c-cg) | QCU Wilson CG、QUDA/纯 Python 参考 |
| Chebyshev | [C.Chebyshev](#c-chebyshev) | QUDA CA basis/平滑器参考；QCU 非默认外层 |
| Schwarz/SAP | [S.Schwarz](#s-schwarz)、[S.SAP](#s-sap) | QUDA 预条件概念；PyQCU 以 MR/FGMRES 组合为主 |
| FGMRES | [F.FGMRES](#f-fgmres) | `pyqcu/solver/_gmres.py`、strict fused right-FGMRES |
| BiCG/BiCGStab | [B](#b-bicg)、[B](#b-bicgstab) | Python BiCGStab、QCU/QUDA BiCGStab(l) |
| GCR/flexible-GCR | [G.GCR](#g-gcr-与-flexible-gcr) | QUDA GCR，PyQCU FGMRES 等价右预条件框架 |
| CA-GCR | [C.CA-GCR](#c-ca-gcr) | QUDA `inv_ca_gcr.cpp` 参考；PyQCU 有 CA-CG |
| Arnoldi | [A.Arnoldi](#a-arnoldi) | FGMRES 与 strict FGMRES 的正交核心 |
| CGS | [C.CGS](#c-cgs)、[Q.QR](#q-qrcgs-与局部基) | CGS solver 为参考；局部重复 CGS 用于 null basis |
| QR | [Q.QR](#q-qrcgs-与局部基) | 局部正交化的批量 QR/CGS 等价实现 |
| Wilson/Clover | [W](#w-wilson)/[C](#c-clover-improved-wilson) | PyQCU 原生实现，QCU/QUDA 对照 |
| twisted mass/twisted clover | [T](#t-twisted-mass含-non-degenerate-pairs) | QUDA 参考；PyQCU 无完整生产入口 |
| staggered/asqtad/HISQ | [S](#s-staggered-fermions)/[A](#a-asqtadhisq) | QUDA 参考；PyQCU 主路径为 Wilson/Clover |
| domain-wall/Möbius | [D](#d-domain-wall)/[M](#m-möbius) | QUDA 参考；QCU 当前 ABI 未实现 5-d 动力学 |
| overlap | [O.Overlap](#o-overlap) | 理论参考；当前快照未发现 `DiracOverlap` 生产实现 |
| QUDA/QCU 粗算子 | [Q](#q-quda-coarse-op)/[S](#s-strict-qcu-coarse-op) | 分别覆盖 `coarse_op`、legacy 33-tensor、strict X/Y/Yhat |

### 0.5 全链路超长伪代码：从作用量到一次严格 MG 外迭代

下表把作用量、奇偶消元、局部基、Galerkin、平滑、粗解和外层 Krylov 放在同一条链上；
所有实现变体都可以在对应行替换 (D_l)、(P_l)、(S_l) 或 coarse solver。

\begin{table}[htbp]
\centering
\caption*{Table 0.1. 从规范场到 full solution 的统一伪代码}
\small
\begin{tabular}{@{}l@{}}
输入：规范场 (U_\mu)，质量/\(\kappa\)，可选 Clover (C)，格点 (L_xL_yL_zL_t)，\(n_v\)，block，层数 (L\)。\\
检查边界条件、gamma basis、dtype、奇偶定义和 `...xyzt`/QUDA 布局；建立 \(p(x)=(x+y+z+t)\bmod2\)。\\
构造细层 (D_0=D_W(U,m))；若有 Clover，令 (A_p=I+C_p)，并把 (D_0=A-\kappa H\) 分成 even/odd block。\\
若目标是 compact MATPC，选定目标 parity (p)，令 (q=1-p)，准备 (b_p^S=b_p+\kappa H_{pq}A_q^{-1}b_q)。\\
若目标是 full coarse，保留两个 parity 的完整场；只在 solver 边界执行 `prepare/reconstruct`，不裁剪粗层资产。\\
生成随机/test/inverse/CG/GCR/CA-CG setup 向量 (B_l)，使其覆盖 (D_l) 的低模和拓扑近零空间。\\
按 aggregate 把 fine site 映射到 coarse site；对每个 coarse spin block 重复 CGS 或块 QR，得到局部正交 (V_l)。\\
定义 (P_l\phi) 为按 (V_l) 的局部展开，定义 (R_l=P_l^\dagger)；先验证 (R_lP_l\approx I)。\\
选择被投影算子：full Galerkin 用 (A_l=D_l)，strict direct-PC 用 (A_l=X_l^{-1}D_l)，legacy Schur 用 (A_l=S_l)。\\
对每个 coarse source block 置单位向量，计算 (w=A_l(P_le_j))，再限制 (R_lw)，得到一列 (D_{l+1}=R_lA_lP_l)。\\
按位移把 (D_{l+1}) 分成 onsite (X_{l+1})、forward (Y^f_{l+1})、backward (Y^b_{l+1})；若出现非最近邻项，strict 模式 fail-closed。\\
批量求 (X_{l+1}^{-1})，构造 \(\widehat Y^f=X^{-1}Y^f\) 与 \(\widehat Y^b=Y^bX^{-\dagger}\)（backward 的左右次序不能交换）。\\
缓存本层 (V_l)、(X_l)、(X_l^{-1})、(Y_l)、(Yhat_l)，然后把 (D_{l+1}) 当作下一层算子，递归直到粗层。\\
一次 V-cycle 以 (x_l=0) 或热启动开始，执行 \(\nu_{pre}\) 次 MR/CG/SAP/Schwarz 平滑。\\
形成真残差 (r_l=b_l-D_lx_l)；若当前是 fine MATPC，先用完整算子重建被消去 parity 再计算 (r_l)。\\
限制粗右端 (b_{l+1}=R_lr_l)，并保持 coarse field 的 full geometry；只有 `restrict_parity` 视图携带 target parity。\\
若 (l<L-1)，递归调用 child V/W/F/K-cycle；若 (l=L-1)，调用 coarse CG/BiCGStab/GCR/CA-GCR 或直接解。\\
得到粗误差 (e_{l+1})，延拓 (e_l=P_le_{l+1})，执行 (x_l\leftarrow x_l+e_l)。\\
执行 \(\nu_{post}\) 次平滑；如 solver 状态含 (p,v,s,t,\rho,\alpha,\omega)，粗校正后必须全部重置。\\
外层若用右预条件 FGMRES/GCR，令 (r_0=b-Dx_0\)，(v_1=r_0/\|r_0\|)。\\
第 (j) 轮取 (z_j=M_j^{-1}v_j)（MG 可随层、精度和 warm state 改变），再算 (w_j=Dz_j)。\\
用 Arnoldi MGS/CGS/QR 得 (H_{ij}=\langle v_i,w_j\rangle)，对 Hessenberg 列做复 Givens，更新小三角系统。\\
重启或达到估计容差后回代 (y)，更新 (x=x_0+\sum_j y_jz_j)，独立计算 full true residual。\\
若可靠残差满足 (\|b-Dx\|\le\texttt{atol}\|b\|)，结束；否则回到平滑/限制/粗解或下一次外层 Krylov。\\
释放 Strict hierarchy、C++ `LatticeSet` 和 scratch；保持每个实例独立的 `set_ptrs/params/argv`，不得跨线程复用。\\
\end{tabular}
\end{table}

**源码锚点：** `pyqcu/solver/_multigrid.py:cycle/solve`；`pyqcu/solver/_quda_multigrid.py:2715-3060`；
`pyqcu/tools/_strict_galerkin.py:310-538,594-800`；`cpp/cuda/qcu/src/apply_multigrid_strict.cu`；
`refer/git-rep/quda/lib/multigrid.cpp:1131-1224`、`dirac_coarse.cpp:461-649`。

## A. Arnoldi

### 数学结构

Arnoldi 在非 Hermitian (A) 上构造正交 Krylov 基
(mathcal K_m(A,r_0)=\operatorname{span}\{r_0,Ar_0,\ldots,A^{m-1}r_0\})。令
(\beta=\|r_0\|)、(v_1=r_0/\beta)，每一步为

$$
 w_j=Av_j,
\quad h_{ij}=\langle v_i,w_j\rangle,
\quad w_j\leftarrow w_j-\sum_{i=1}^{j}h_{ij}v_i,
\quad h_{j+1,j}=\|w_j\|,
\quad v_{j+1}=w_j/h_{j+1,j}.
$$

于是 (AV_m=V_{m+1}\bar H_m)。(h_{j+1,j}=0) 是不变子空间或 breakdown；数值实现
通常再做一次 MGS 或选择 CGS/QR 以抑制失正交。

### 长表伪代码

\begin{table}[htbp]
\centering
\caption*{Table A.1. Arnoldi 正交化与小系统更新}
\small
\begin{tabular}{@{}l@{}}
输入 (r_0,A,m)；若 (|r_0|=0) 直接返回 (x_0)。\\
设 (\beta=|r_0|)，(v_1=r_0/\beta)，令 (H=0)，(g=(\beta,0,\ldots)^T)。\\
for (j=1,\ldots,m)：计算 (w=Av_j)。\\
for (i=1,\ldots,j)：(h_{ij}=\langle v_i,w\rangle)，(w\leftarrow w-h_{ij}v_i)。\\
可再做一次 (h_{ij}\leftarrow h_{ij}+\langle v_i,w\rangle)，(w\leftarrow w-\langle v_i,w\rangle v_i)。\\
令 (h_{j+1,j}=|w|)；若不为零，(v_{j+1}=w/h_{j+1,j})，否则标记 happy breakdown。\\
把已有 Givens 旋转作用到第 (j) 列，选择 (c_j,s_j) 使 (H_{j+1,j}=0)。\\
更新 (g_{j+1}=-s_jg_j)、(g_j=\overline c_jg_j)，残差估计为 (|g_{j+1}|)。\\
达到容差或 (j=m) 时回代 (H_{1:j,1:j}y=g_{1:j})，返回 (x=x_0+V_jy)。\\
若为 FGMRES，不保存 (v_j) 作为修正，而保存 (z_j=M_j^{-1}v_j)，返回 (x=x_0+Z_jy)。\\
\end{tabular}
\end{table}

**优点：** 适用于非 Hermitian 算子，残差最小化有清楚的投影解释；Givens 只维护小 Hessenberg。
**缺点：** 需要 (O(m)) 向量和 (O(m^2)) 内积，重启过短会丢失谱信息；长基在 GPU 上受全局归约和内存带宽限制。
**适用：** Wilson/Clover Schur、coarse-PC、MG 右预条件；本库 `_gmres.py` 用复 Givens，
严格路径的 C++ `applyMultigridStrictFgmresQcu` 固定右预条件语义。

**实现来源：** `pyqcu/solver/_gmres.py:_givens_rotation,fgmres`；`refer/git-rep/DDalphaAMG-SM` 的
`fgmres.cpp`；`cpp/cuda/qcu/src/apply_multigrid_strict.cu`。Arnoldi 不是独立的物理作用量，
而是外层 Krylov 的骨架。

## A. Asymmetric Schur

### 从 block elimination 出发

对 (D=\begin{psmallmatrix}A_p&B_{pq}\\B_{qp}&A_q\end{psmallmatrix})，消去 (q) parity 得到

$$
S_p^{\rm asym}=A_p-B_{pq}A_q^{-1}B_{qp}.
$$

对 Wilson/Clover，(B_{pq}=-\kappa H_{pq})，因此

$$
S_p^{\rm asym}=A_p-\kappa^2H_{pq}A_q^{-1}H_{qp}.
$$

右端和恢复公式为

$$
 b_p^S=b_p-B_{pq}A_q^{-1}b_q,
\qquad
 x_q=A_q^{-1}(b_q-B_{qp}x_p).
$$

由于 (A_p) 与 (A_q) 一般不同，(S_p^{\rm asym}) 不必 Hermitian；BiCGStab、GCR 或 FGMRES 比 CG 更稳妥。

### 单列伪代码

\begin{table}[htbp]
\centering
\caption*{Table A.2. Asymmetric Schur prepare/solve/reconstruct}
\small
\begin{tabular}{@{}l@{}}
输入 full (b=(b_p,b_q))、局部块 (A_p,A_q)、hopping (B_{pq},B_{qp})。\\
解局部系统 (u_q=A_q^{-1}b_q)，形成 (b_p^S=b_p-B_{pq}u_q)。\\
以 (S_p^{\rm asym}=A_p-B_{pq}A_q^{-1}B_{qp}) 作为 Krylov matvec。\\
用 BiCGStab/GCR/FGMRES 得 (S_p^{\rm asym}x_p=b_p^S)。\\
计算 (v_q=b_q-B_{qp}x_p)，局部求解 (x_q=A_q^{-1}v_q)。\\
将 (x_p,x_q) 按全场 parity map 合并，独立算 (|b-Dx|)。\\
\end{tabular}
\end{table}

**优点：** 只对目标 parity 求解，Clover onsite inverse 可以局部批量求；是 QUDA `DiracCloverPC` 和 PyQCU
`applyCloverBistabCgPrepareQcu/ReconstructQcu` 的直接数学来源。
**缺点：** 算子非 Hermitian，不能无条件使用 CG；两次 hopping 加一次局部 inverse 使每次 matvec 成本高；
局部 inverse 的条件数直接影响 Schur 谱。
**适用：** Clover fine MATPC、QCU 的 `applyCloverBistabCgDslashQcu`，以及 legacy odd-Schur MG。

**实现来源：** `pyqcu/dslash/_operator.py:matvec_parity,give_b_parity,give_x_e`；
`cpp/cuda/qcu/src/apply_clover_bistabcg_dslash.cu`；`refer/git-rep/quda/lib/dirac_clover.cpp` 的
`prepare/reconstruct`。

## A. Asqtad/HISQ

### 作用量与改进链

一分量 staggered 核为

$$
(D_{\rm stag}\chi)(x)=m\chi(x)+\frac12\sum_\mu\eta_\mu(x)
\left[V_\mu(x)\chi(x+\hat\mu)-V_\mu^\dagger(x-\hat\mu)\chi(x-\hat\mu)\right],
$$

其中 (eta_\mu(x)=(-1)^{\sum_{\nu<\mu}x_\nu})。Asqtad 以 Fat7、Lepage 和 Naik 三类路径改进
(V_\mu)，HISQ 再进行一次 reunitarization/smearing，降低 taste breaking 和 (O(a^2)) 误差。
Naik 长链连接 (x\to x+3\hat\mu)，因此 setup halo 需要更深的 `nFace`；QUDA coarse Wilson dslash 的
一跳 `nFace=1` 不能混用为 HISQ setup 的通信深度。

### 伪代码

\begin{table}[htbp]
\centering
\caption*{Table A.3. Asqtad/HISQ 链接与算子}
\small
\begin{tabular}{@{}l@{}}
输入 thin links (U_\mu)、质量 (m)、Fat7/Lepage/Naik 系数。\\
构造所有允许的 3-, 5-, 7-link staples，按路径系数线性组合为 (V^{\rm Fat7})。\\
减去 Lepage 修正，加入 (c_N[U_\mu(x)U_\mu(x+\hat\mu)U_\mu(x+2\hat\mu)])。\\
HISQ 路径先对 (V^{\rm Fat7}) 做 reunitarize，再进行第二次 smearing，得到 (V^{\rm HISQ})。\\
交换至少三层长链 ghost，应用 (eta_\mu) 相位和 forward/backward gather。\\
若做 HMC，再按每条路径反向传播链式法则，累加 gauge force。\\
\end{tabular}
\end{table}

**优点：** taste breaking 小、连续极限的 (a^2) 系统误差低；staggered 自由度少，适合大体积。
**缺点：** 长路径通信和力项复杂；根号行列式/ROOT 分支需要 RHMC；粗化 stencil 可能超出 strict 的最近邻 X/Y ABI。
**适用：** QUDA `dirac_improved_staggered_kd.cpp`、`coarse_op.cuh` 的 staggered 分支；当前 PyQCU Strict
显式拒绝标准 staggered/KD 模式（`_quda_multigrid.py:2197-2202`），因此这里只能作为 QUDA 参考。

**来源：** `refer/git-rep/quda/README.md:14-16`；`lib/dslash_improved_staggered.hpp`、
`lib/dirac_improved_staggered_kd.cpp`、`include/kernels/hisq_paths_force.cuh`；QUDA `NEWS` 中
HISQ long-link、force 与多移位 solver 条目。

## B. BiCG

### 算法

BiCG 同时在 (A) 和 (A^\dagger) 的对偶 Krylov 空间推进。给定 shadow residual
(\tilde r_0)，

$$
\rho_k=\langle\tilde r_0,r_k\rangle,
\quad
\alpha_k=\frac{\rho_k}{\langle\tilde p_k,Ap_k\rangle},
\quad
r_{k+1}=r_k-\alpha_kAp_k,
$$

并在对偶空间用 (A^\dagger) 更新 (\tilde p)。理论上每步只需一次 (A) 与一次 (A^\dagger)，但
(\rho_k) 或 pivot 过小会 breakdown。实际格点代码通常用 BiCGStab 的 residual smoothing 避免显式 transpose solve。

### 伪代码、优缺点与适用范围

\begin{table}[htbp]
\centering
\caption*{Table B.1. BiCG 双 Krylov 伪代码}
\small
\begin{tabular}{@{}l@{}}
(r_0=b-Ax_0,;\tilde r_0=r_0,;p_0=\tilde p_0=0,;\rho_{-1}=1)。\\
for (k=0,1,\ldots)：(\rho_k=\langle\tilde r_0,r_k\rangle)，若 (\rho_k=0) 则 breakdown。\\
(\beta_k=(\rho_k/\rho_{k-1})(\alpha_{k-1}/\cdots))，更新 (p_k,\tilde p_k)。\\
(q_k=Ap_k,;\alpha_k=\rho_k/\langle\tilde r_0,q_k\rangle)。\\
(x_{k+1}=x_k+\alpha_kp_k,;r_{k+1}=r_k-\alpha_kq_k)。\\
同时以 (A^\dagger) 更新 shadow 侧；检查 true residual 和所有分母。\\
\end{tabular}
\end{table}

**优点：** 存储量小、适合非 Hermitian；**缺点：** breakdown 频繁、残差可能剧烈振荡，且需要 shadow 方向。
**适用：** 理论基线和 BiCGStab 推导；本库没有单独导出的 `bicg` API，QCU/QUDA 生产路径以 BiCGStab、GCR 或 FGMRES 替代。

**来源：** `refer/git-rep/quda/README.md:19-20`；QUDA solver factory；本库
`pyqcu/solver/_bistabcg.py` 的 shadow residual 初始化是 BiCGStab 继承来的稳定化形式。

## B. BiCGStab

### 标准递推

BiCGStab 在每个 BiCG 步后用一维 MR 方向平滑 residual。设 (\tilde r) 固定：

$$
\rho_{i-1}=\langle\tilde r,r_{i-1}\rangle,
\quad
\beta_{i-1}=\frac{\rho_{i-1}}{\rho_{i-2}}\frac{\alpha_{i-1}}{\omega_{i-1}},
$$

$$
 p_i=r_{i-1}+\beta_{i-1}(p_{i-1}-\omega_{i-1}v_{i-1}),
\quad v_i=Ap_i,
\quad \alpha_i=\frac{\rho_{i-1}}{\langle\tilde r,v_i\rangle},
$$

$$
 s_i=r_{i-1}-\alpha_iv_i,
\quad t_i=As_i,
\quad \omega_i=\frac{\langle t_i,s_i\rangle}{\langle t_i,t_i\rangle},
$$

$$
 x_i=x_{i-1}+\alpha_ip_i+\omega_is_i,
\qquad r_i=s_i-\omega_it_i.
$$

右预条件时 (p_i,s_i) 先解 (M\hat p=p_i,M\hat s=s_i)，再以 (A\hat p,A\hat s) 计算。

### 超长单列伪代码

\begin{table}[htbp]
\centering
\caption*{Table B.2. Bi-CGStab with preconditioning matrix (M)}
\small
\begin{tabular}{@{}l@{}}
(r^{(0)}=b-Ax^{(0)})，选择 (\tilde r=r^{(0)})，置 (p=v=s=t=0)，(\rho_{-1}=\alpha_0=\omega_0=1)。\\
for (i=1,2,\ldots)：计算 (\rho_{i-1}=\tilde r^\dagger r^{(i-1)})，若相对尺度下接近零则执行 breakdown guard/重启。\\
若 (i=1)，(p^{(1)}=r^{(0)})；否则计算 (\beta_{i-1}=\alpha_{i-1}\rho_{i-1}/(\omega_{i-1}\rho_{i-2}))。\\
更新 (p^{(i)}=r^{(i-1)}+\beta_{i-1}(p^{(i-1)}-\omega_{i-1}v^{(i-1)}))。\\
解 (M\hat p=p^{(i)})，计算 (v^{(i)}=A\hat p)。\\
计算 (\alpha_i=\rho_{i-1}/(\tilde r^\dagger v^{(i)}))；令 (s=r^{(i-1)}-\alpha_iv^{(i)})。\\
若 (|s|) 已满足可靠容差，令 (x^{(i)}=x^{(i-1)}+\alpha_i\hat p) 并停止。\\
解 (M\hat s=s)，计算 (t=A\hat s)，再取 (\omega_i=(t^\dagger s)/(t^\dagger t))。\\
更新 (r^{(i)}=s-\omega_it)，(x^{(i)}=x^{(i-1)}+\alpha_i\hat p+\omega_i\hat s)。\\
若 (|r^{(i)}|) 达标则停止；否则保存 (\rho_i,\alpha_i,\omega_i) 进入下一轮。\\
每次 MG 粗校正改变 (r) 后，必须把 (\tilde r,p,v,s,t,\rho,\alpha,\omega) 全部重置。\\
\end{tabular}
\end{table}

**优点：** 非 Hermitian 友好、每轮只需少量向量、比 BiCG 平滑；适合 Clover Schur、粗层非对称矩阵和 QCU GPU。
**缺点：** (\rho)、(omega)、(t^\dagger t) breakdown；递推 residual 可能与 full true residual 脱钩，c64 尤其明显；单次迭代有两次 operator apply。
**适用：** `pyqcu/solver/_bistabcg.py`、`cpp/cuda/qcu/include/bistabcg.h`、QUDA `inv_bicgstabl_quda.cpp`；
QUDA 的 BiCGStab(l) 用 (l)-维 GCR-like residual minimization，Wilson/Clover 上通常比 (l=1) 稳定。

**来源：** `pyqcu/solver/_bistabcg.py:23-89`（rho/pivot/tts guard）；
`cpp/cuda/qcu/include/lattice_clover_multigrid.h:302-510`（GPU scalar guard 与五 stream）；
`refer/git-rep/quda/README.md:19-20`、`NEWS:202-205`。

## C. CA-GCR

### 通信规避思想

CA-GCR 每个 block 用 (s) 次 (A) 作用生成
(V_j=[r,Ar,\ldots,A^{s-1}r])，集中完成 Gram 矩阵和小规模最小残差问题，从而减少全局归约次数。
若 (Z_j=M_j^{-1}V_j)，则求解

$$
G_j=(AZ_j)^\dagger(AZ_j),
\qquad g_j=(AZ_j)^\dagger r,
\qquad G_j\gamma_j=g_j,
\qquad x\leftarrow x+Z_j\gamma_j.
$$

### 伪代码与分析

\begin{table}[htbp]
\centering
\caption*{Table C.1. CA-GCR/s-step block}
\small
\begin{tabular}{@{}l@{}}
输入 (r_j,x_j)、block 长度 (s)、右预条件 (M_j^{-1})。\\
生成 (K=[r_j,Ar_j,\ldots,A^{s-1}r_j])，必要时对每列应用 (M_j^{-1})。\\
得到 (W=AK)，使用 MGS/QR 对 (K,W) 同步正交化，避免 power basis 条件数爆炸。\\
批量归约 (G=W^\dagger W,;g=W^\dagger r_j)，解 (G\gamma=g)。\\
更新 (x_{j+1}=x_j+K\gamma)，可靠地重算 (r_{j+1}=b-Ax_{j+1})。\\
若收敛则停；否则将 block image 与已有 GCR basis 正交化并继续。\\
\end{tabular}
\end{table}

**优点：** 全局同步由 (s) 次 matvec 共用，适合 GPU/MPI；可与 coarse GCR 和 mixed precision 组合。
**缺点：** (G) 条件数可能快速恶化，workspace (O(s))，block 末小系统求解开销不可忽略；谱估计或 Chebyshev basis 错误会造成假收敛。
**适用：** QUDA `lib/inv_ca_gcr.cpp`；QUDA MG coarse solver 可以设置 `QUDA_CA_GCR_INVERTER`。
本库目前有 `pyqcu/solver/_cacg.py` 的 CA-CG（不是 CA-GCR），因此不能把两者名称互换。

## C. CG

### 递推与适用条件

对 Hermitian positive definite (A)，

$$
\rho_k=\langle r_k,r_k\rangle,
\quad
\alpha_k=\rho_k/\langle p_k,Ap_k\rangle,
\quad
x_{k+1}=x_k+\alpha_kp_k,
$$

$$
 r_{k+1}=r_k-\alpha_kAp_k,
\quad
\beta_k=\rho_{k+1}/\rho_k,
\quad
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

Wilson (D) 通常只满足 (gamma_5)-Hermiticity，安全做法是解 (D^\dagger D)（CGNE/CGNR）或使用
经过 Hermitian 变换的 Schur；Clover symmetric Schur 的名字本身不等于正定。

### 伪代码与工程映射

\begin{table}[htbp]
\centering
\caption*{Table C.2. CG}
\small
\begin{tabular}{@{}l@{}}
(x_0) 给定，(r_0=b-Ax_0)，(p_0=r_0)。\\
for (k=0,1,\ldots)：(Ap_k=Ap\)，(\alpha_k=\rho_k/(p_k^\dagger Ap_k))。\\
(x_{k+1}=x_k+\alpha_kp_k)，(r_{k+1}=r_k-\alpha_kAp_k)。\\
(\rho_{k+1}=r_{k+1}^\dagger r_{k+1})，(\beta_k=\rho_{k+1}/\rho_k)。\\
(p_{k+1}=r_{k+1}+\beta_kp_k)，检查 true residual 和 denominator。\\
若使用 reliable update，周期性以 (b-Ax) 替换递推 (r)，并重新设置 (p=r)。\\
\end{tabular}
\end{table}

**优点：** 每轮一个 matvec，一个或两个全局归约；内存小，SPD 情形理论收敛界清楚。
**缺点：** 对非 Hermitian/不定矩阵失效；有限精度破坏共轭，低精度需要 reliable update。
**实现来源：** `cpp/cuda/qcu/include/cg.h`、`lattice_wilson_cg.h`；QUDA `inv_cg_quda.cpp`；
`examples/qcu/conftest.wilson.cg.py`。PyQCU 纯 Python 主求解器默认以 BiCGStab/FGMRES 覆盖非 Hermitian。

## C. CGS

“CGS”有两种必须区分的语义：

1. **Conjugate Gradient Squared solver**：把 BiCG 的二项式残差多项式平方，消除显式 shadow solve；
2. **Classical Gram–Schmidt**：在局部 aggregate 内做基正交，本报告 [Q.QR](#q-qrcgs-与局部基) 另述。

### CGS solver

设 BiCG 的 residual polynomial 为 (P_k(A))，CGS 用 (P_k(A)^2r_0) 更新，典型形式为

$$
q_k=u_k-\alpha_kAu_k,
\quad
u_{k+1}=u_k+\beta_kq_k,
\quad
r_{k+1}=u_{k+1}-\alpha_kA(u_{k+1}+\beta_kq_k).
$$

**优点：** 不需要显式 (A^\dagger)，理论上 matvec 次数与 BiCG 相当；**缺点：** residual 多项式平方造成极强振荡和数值不稳定，格点 QCD 生产中通常优先 BiCGStab(l)/GCR。当前 PyQCU/QCU API 未提供 CGS solver；仅将其作为算法对照。

### Classical Gram–Schmidt 语义

对局部向量 (b_j) 和已正交 (q_i)，

$$
q_j=b_j-\sum_{i<j}q_i(q_i^\dagger b_j),
\qquad q_j\leftarrow q_j/\|q_j\|.
$$

单次 CGS 易有失正交；本库 `n_block_ortho=2` 和 QUDA block orthogonalize 都采用重复投影或 QR 等价增强。

**来源：** `pyqcu/solver/_quda_multigrid.py` 的 `QudaTransfer`、`refer/git-rep/quda/lib/block_orthogonalize.in.cpp`；
CGS solver 仅作通用线性代数参考。

## C. Chebyshev

### 多项式平滑

对 Hermitian 谱区间 ([\lambda_{\min},\lambda_{\max}])，把
(A=cI+d\tilde A)，(c=(\lambda_{\max}+\lambda_{\min})/2)，
(d=(\lambda_{\max}-\lambda_{\min})/2)，以 Chebyshev (T_k) 构造

$$
T_{k+1}(\tilde A)=2\tilde A T_k(\tilde A)-T_{k-1}(\tilde A).
$$

平滑器常取抑制低频/高频指定谱段的多项式 (p_k(A)r)，避免每步全局内积。

\begin{table}[htbp]
\centering
\caption*{Table C.3. Chebyshev 平滑器}
\small
\begin{tabular}{@{}l@{}}
输入 (r)、阶数 (k)、谱估计 ([\lambda_l,\lambda_h])，设置 (p_0=r)、(p_{-1}=0)。\\
按三项递推系数计算 (p_{j+1}=a_jAp_j+b_jp_j+c_jp_{j-1})。\\
累计 (x\leftarrow x+\omega_jp_j)，每步只需 operator apply 和 axpy。\\
若谱估计越界，降低阶数或重新估计；若算子非 Hermitian，必须说明使用的是包络/正规算子谱。\\
\end{tabular}
\end{table}

**优点：** 无 dot 或少 dot，适合 GPU smoother 和 CA basis；**缺点：** 对谱界敏感，非 Hermitian Schur 上不保证最小残差。
QUDA CA-CG 使用 Chebyshev/power basis 选项；本库当前 QCU coarse 默认 MR/CG/BiCGStab，未将 Chebyshev 作为公开独立 API。

## C. Clover-improved-Wilson

### 作用量

Wilson 核加入 Sheikholeslami–Wohlert 局部项：

$$
D_{\rm SW}=D_W-\frac{\kappa c_{\rm SW}}{2}
\sum_{\mu<\nu}\sigma_{\mu\nu}F_{\mu\nu}^{\rm clover},
\qquad
\sigma_{\mu\nu}=\frac12[\gamma_\mu,\gamma_\nu].
$$

(F_{\mu\nu}^{\rm clover}) 是四个 plaquette 叶片的反厄米无迹组合。不同库把 (1/2)、
(i)、(kappa) 或 (u_0) 吸入 (C) 的方式不同；PyQCU `_clover.py` 使用
`_clover_factor=-0.125*kappa/u_0`，因此比较时必须同时看 `make_clover` 和 `give_clover`，不能只比裸矩阵。

### 伪代码

\begin{table}[htbp]
\centering
\caption*{Table C.4. Clover 构造与应用}
\small
\begin{tabular}{@{}l@{}}
对每个 ((\mu,\nu)) 收集四个方向的 plaquette，计算 (Q_{\mu\nu}=P_{\mu\nu}-P_{\mu\nu}^\dagger)。\\
按 (F_{\mu\nu}=(Q_{\mu\nu}-\operatorname{Tr}Q_{\mu\nu}/3)/8i)（系数随实现约定）嵌入 color block。\\
组装 (C(x)=\sum_{\mu<\nu}\sigma_{\mu\nu}\otimes F_{\mu\nu}(x))，返回 (A=I+C)。\\
按 parity 抽取 (A_e,A_o)，批量求 (A_e^{-1},A_o^{-1})，禁止逐点 Python inverse 循环。\\
应用 full operator 时执行 (D_W\psi+C\psi)；应用 Schur 时用对应 parity inverse。\\
\end{tabular}
\end{table}

**优点：** 消除 on-shell (O(a)) 离散误差，Clover inverse 是局部 (12\times12) 批量矩阵；
**缺点：** 保留 γ₅-Hermiticity 但 onsite block 不再是单位阵，存储、局部 inverse 和 setup 成本上升；系数/边界条件不一致会造成百分比级偏差。
**实现来源：** `pyqcu/dslash/_clover.py`；`cpp/cuda/qcu/include/lattice_clover_dslash.h`；
`refer/git-rep/quda/lib/dirac_clover.cpp`、`clover_quda.cu`；`examples/qcu/dev87/comparison_matrix.md`。

## D. Domain-wall

### 5-d 作用量与 4-d reduction

Domain-wall fermion 在第五维 (s=0,\ldots,L_s-1) 放置四维 Wilson kernel：

$$
D_{\rm DW}(x,s;y,s')=D_W^{4d}(x,y)\delta_{ss'}+D_5(s,s'),
$$

$$
D_5(s,s')=P_+\delta_{s+1,s'}+P_-\delta_{s-1,s'}
-m_f\left(P_-\delta_{s,0}\delta_{s',L_s-1}+P_+\delta_{s,L_s-1}\delta_{s',0}\right),
$$

其中 (P_\pm=(1\pm\gamma_5)/2)。有限 (L_s) 的残余质量来自两墙混合；(L_s\to\infty) 才逼近精确手征。

QUDA 的 4-d preconditioned 形式把 (M_5) 的局部逆和四维 hopping 分开。以
(kappa_5) 表示第五维系数，非对称形式为

$$
M_{\rm PC}=I-\kappa_5D_5-\kappa_5^2D_4M_5^{-1}D_4,
$$

对称形式可写成

$$
M_{\rm PC}^{\rm sym}=I-\kappa_5^2M_5^{-1}D_4M_5^{-1}D_4.
$$

### 伪代码、优缺点与范围

\begin{table}[htbp]
\centering
\caption*{Table D.1. Domain-wall 5-d/4-d preconditioned solve}
\small
\begin{tabular}{@{}l@{}}
输入 (U,m_f,m_5,L_s)，构造每个 (s) 层的 (D_W^{4d}) 和相邻墙投影。\\
建立 (M_5) 的 block-tridiagonal 结构；若为 Shamir，系数由 (m_5) 和 (m_f) 固定。\\
5-d 直接路径：应用 (D_4) 后沿 (s) 方向做 (P_\pm) gather，再用 CG/BiCGStab/GCR 求解。\\
4-d 预条件路径：计算 (u=M_5^{-1}b_q)，形成 Schur RHS (b_p+\kappa_5D_4u)。\\
在目标 parity 上求 (M_{\rm PC}x_p=b_p^{\rm PC})，再以 (M_5^{-1}(b_q+\kappa_5D_4x_p)) 恢复另一 parity。\\
独立应用完整 5-d operator 检查残差，并报告 residual mass/墙间混合。\\
\end{tabular}
\end{table}

**优点：** 手征对称性可系统恢复；4-d preconditioning 显著减少求解自由度；**缺点：** 内存和通信随 (L_s) 线性增长，墙边界和 (M_5^{-1}) 系数易错。
**适用：** QUDA `dirac_domain_wall_5d.cpp`、`dirac_domain_wall_4d.cpp`、`dslash_domain_wall_5d.hpp`；当前 PyQCU/QCU 的
参数协议只描述四维 Wilson/Clover，不能把 `applyCloverMultigridQcu` 当成 DWF 实现。

## F. FGMRES

### Flexible 右预条件

FGMRES 允许每个 Krylov 向量使用不同预条件器 (M_j^{-1})，这是 MG、SAP、混合精度和 warm-start
组合的关键。对 (v_j) 先算

$$
 z_j=M_j^{-1}v_j,
\qquad w_j=Az_j,
$$

再对 (w_j) 做 Arnoldi；更新解必须保存 (Z=[z_1,\ldots,z_m])，不能误用 (V)。

### 伪代码

\begin{table}[htbp]
\centering
\caption*{Table F.1. FGMRES(m) 右预条件}
\small
\begin{tabular}{@{}l@{}}
(x_0) 给定，(r=b-Ax_0)，(\beta=\|r\|)，(v_1=r/\beta)。\\
for restart cycle：清空 (V,Z,H)，令 (g_1=\beta)。\\
for (j=1,\ldots,m)：(z_j=M_j^{-1}v_j)，保存 (Z_j=z_j)。\\
计算 (w=Az_j)，以 MGS/CGS/QR 得 (h_{ij}=\langle v_i,w\rangle)，(v_{j+1}=w/\|w\|)。\\
用复 Givens 旋转消去 (H_{j+1,j})，更新 (g)，记录 (|g_{j+1}|)。\\
内层结束或估计残差达标时回代 (Hy=g)，更新 (x\leftarrow x+\sum_jy_jZ_j)。\\
重算 (r=b-Ax)；若 full true residual 达标则结束，否则重启。\\
\end{tabular}
\end{table}

**优点：** 允许变化预条件器，适合 MG V-cycle、SAP 和自适应层级；**缺点：** 保存 (Z) 的内存大于 GMRES，
重启会损失谱信息；估计残差必须以 true residual 复核。
**实现来源：** `pyqcu/solver/_gmres.py:fgmres`（默认零初值、任意布局 reshape、内层估计+周期真实残差）；
`cpp/cuda/qcu/src/apply_multigrid_strict.cu` 的 fused right-FGMRES；`refer/git-rep/DDalphaAMG-SM/include/fgmres.h`。

## F. Full Schur 与 full coarse

### 三种对象不能混名

1. **Full operator：** 在所有 parity 上应用 (D)，场形状保持 full lattice；
2. **Fine Schur/MATPC：** 只在目标 parity 解 (S_p) 或 (I-\widehat H_{pq}\widehat H_{qp})，
   prepare/reconstruct 负责与 full RHS/solution 互换；
3. **Full coarse：** 粗层保存完整 coarse geometry 的 (X,Y,Yhat)，parity 只在 MATPC、R/P 的边界视图使用。

Strict 路径定义

$$
D_l=X_l+H_l,
\qquad \widehat D_l=X_l^{-1}D_l,
\qquad D_{l+1}=R_l\widehat D_lP_l,
$$

所以不能把“fine odd Schur 的 compact 场”直接当成“coarse full field”，也不能把 coarse `Y` 当 SU(3) Gauge。

### full→MATPC 伪代码

\begin{table}[htbp]
\centering
\caption*{Table F.2. Fine full/Schur 与 coarse full 的接口}
\small
\begin{tabular}{@{}l@{}}
输入 full (b)、目标 parity (p)、fine (A_p,A_q,B_{pq},B_{qp})。\\
`prepare`：按 asymmetric 或 symmetric 公式形成 compact (b_p^S)，不丢失被消去 parity 的信息。\\
在 compact 空间应用 (S_p) 或 (M_p=I-\widehat H_{pq}\widehat H_{qp})。\\
`restrict_parity` 只读取 target parity 的 fine residual，输出下一层 full coarse field。\\
coarse V-cycle 使用完整 (X/Y/Yhat)，`prolong_parity` 只把修正放回 target parity。\\
`reconstruct` 用被消去 parity 的 RHS 与 (x_p) 恢复 (x_q)，再用 full (D) 计算验收残差。\\
\end{tabular}
\end{table}

**实现来源：** `pyqcu/solver/_quda_multigrid.py:1539-1610,2961-3060`；
`cpp/cuda/qcu/python/pyqcu.h:55-63,78-118`；`refer/git-rep/quda/lib/dirac_coarse.cpp:530-626`。

## G. Galerkin

### 定义与局部化

Galerkin 粗算子是

$$
D_c=RDP,
\qquad R=P^\dagger \text{（正交基时）}.
$$

对严格 X/Y 表示，只需探测 coarse source aggregate 与其 (pm\hat\mu) 相邻 aggregate，
把位移零项放入 (X)，六个轴向项放入 (Y^{\rm f/b})。若局部 (P) 正交且 fine (D) 是最近邻，
粗算子也保持有限邻接；若 fine 是 HISQ/长链，strict 最近邻假设可能不成立。

### 逐列、site-batch、colored 三种实现

\begin{table}[htbp]
\centering
\caption*{Table G.1. Galerkin 组装的三种实现}
\small
\begin{tabular}{@{}l@{}}
逐列：对每个 coarse basis (e_j) 计算 (R(D(Pe_j)))，内存小但 operator call 数为粗自由度倍数。\\
site-batch：同一组 coarse source site 叠加成 batch，调用 `matvec_batch`，再以 einsum 做 (V^\dagger D V)。\\
colored：按不相交 aggregate/color 分组并行探测，减少峰值工作区；必须验证不同 source 的支撑不重叠。\\
完成后检查 (\|D_c^{\rm explicit}v-RDPv\|/\|RDPv\|)，并按 displacement 检查非零支撑。\\
\end{tabular}
\end{table}

**优点：** 保持物理算子低模等价性，误差可由 (RDP) 直接验收；**缺点：** setup 需要大量 operator apply 和显存，
局部 basis 质量决定粗层条件数；长程作用量会产生更多 stencil 项。

**实现来源：** `pyqcu/dslash/_operator.py:operator.__init__` 的显式 Galerkin；
`pyqcu/tools/_strict_galerkin.py:594-800` 的 batched/colored；`refer/git-rep/quda/lib/coarse_op.cuh:1043-1457` 的
UV/VUV/coarse block kernel；`docs/report_multigrid_quda_pyqcu_20260909.md:203-268`。

## G. GCR 与 flexible-GCR

### GCR

GCR 不要求 (A) Hermitian，通过构造 (A)-共轭 image basis 使残差最小化：

$$
z_j=M^{-1}r_j,
\quad w_j=Az_j,
\quad
\alpha_j=\frac{\langle w_j,r_j\rangle}{\langle w_j,w_j\rangle},
\quad x_{j+1}=x_j+\alpha_jz_j,
\quad r_{j+1}=r_j-\alpha_jw_j.
$$

后续 (w_j) 对已有 image basis 正交化。GCR 与 GMRES 等价地维护 image 空间，但更新形式不同。

### flexible-GCR

若 (M_j) 改变，保存 (z_j=M_j^{-1}v_j) 而非固定 (M^{-1}v_j)，小系统由
(W_j=Az_j) 形成。这与 FGMRES 的右预条件语义一致；比较 QUDA GCR 和 PyQCU FGMRES 时应比较 operator/preconditioner 顺序，不能只比较迭代名称。

\begin{table}[htbp]
\centering
\caption*{Table G.1. GCR/flexible-GCR}
\small
\begin{tabular}{@{}l@{}}
(r_0=b-Ax_0)。每轮计算 (z_j=M_j^{-1}r_j) 或 (M_j^{-1}v_j)，(w_j=Az_j)。\\
对 (w_j) 与旧 image basis 做 MGS，修正 (z_j) 同步做同样线性组合。\\
取 (\alpha_j=\langle w_j,r_j\rangle/\langle w_j,w_j\rangle)，更新 (x,r)。\\
若使用 restart，保留最近 (m) 个 image；若使用 flexible 版本，禁止把不同 (M_j) 合并成单一固定算子。\\
\end{tabular}
\end{table}

**优点：** 非 Hermitian、预条件器灵活；**缺点：** basis 存储和 (O(m^2)) 正交开销；GCR 的长期 basis 需要 restart/deflation。
**来源：** `refer/git-rep/quda/lib/inv_gcr_quda.cpp`、`lib/inv_ca_gcr.cpp`、`lib/multigrid.cpp:620-666`；
PyQCU 的等价外层为 `_gmres.py:fgmres` 和 strict fused FGMRES。

## L. Lanczos 与多移位 CG

### 厚重启 Lanczos

对 Hermitian 算子 A，Lanczos 生成三对角投影矩阵 T：

$$
A v_j = β_{j-1}v_{j-1}+α_jv_j+β_jv_{j+1},
\qquad V_m^†V_m≈I.
$$

Rayleigh–Ritz 对 T 求本征对，厚重启保留低端 k 个 Ritz 向量和残差方向，再继续扩展。它把
拓扑近零模变成 deflation basis，可供 eigCG/GMRES-DR/MG setup 使用。

| 步骤 | 公式或动作 | PyQCU/QUDA 对照 |
|---|---|---|
| 扩展 | `w = A v_j`，对已有 V 全重正交 | `pyqcu/solver/_lanczos.py` |
| 投影 | `T[i,j]=<v_i,w>`，对称双写 | 避免 `torch.linalg.eigh` 读到未写三角 |
| Ritz | `T s_i=θ_i s_i`，`y_i=V s_i` | 低端 θ 与残差估计 |
| 重启 | 保留 k 个 Ritz + β·s[last,i] 残差列 | thick restart / arrowhead |
| 验收 | `||A y_i − θ_i y_i||` 再算一次 | c64 不接受只看估计残差 |

**优点：** 对 Hermitian 低模和 deflation 直接；**缺点：** 全重正交成本 O(m²)、基存储 O(m)，非 Hermitian 场景需 Arnoldi。
实现来源：`pyqcu/solver/_lanczos.py:tr_lanczos`；QUDA `lib/eig_trlm.cpp`、`lib/eig_block_trlm.cpp`。

### Multi-shift CG

当所有系统共享 Hermitian positive definite A 且只差标量 shift 时，

$$
(A+σ_i I)x_i=b
$$

可以共用一次 `A p`，用 ζ_i、α_i、β_i 递推全部移位。最小 shift 作为主链，必须把 σ₀ p
折入主链 `Ap`，否则递推 residual 会漂移。适用 overlap rational sign、staggered RHMC 和多质量传播子；
不适用于不同 gauge/hopping 的 Wilson 质量核。

| 优点 | 缺点 | 本库状态 |
|---|---|---|
| 一次 matvec 解多个质量，内存 O(Nshift) | ζ 递推在 c64 可能爆炸；不同初值不能直接共用 | `pyqcu/solver/_multishift_cg.py`；QUDA `inv_multi_cg_quda.cpp` |

## M. Möbius

### 5-d kernel

Möbius domain-wall 把第五维系数推广为 (b_s,c_s)，常用关系为

$$
D_{\rm Mobius}=\left[b_sD_W+1\right]\delta_{ss'}
+\left[c_sD_W-1\right]D_5(s,s'),
$$

或等价地以

$$
\kappa_b=\frac{1/2}{b_s(m_5+4)+1},
\quad
\kappa_c=\frac{1/2}{c_s(m_5+4)-1},
\quad
\kappa=\frac{\kappa_b}{\kappa_c}
$$

组织 (M_5^{-1}) 的闭式系数。具体负号和 (m_5) 归一化依 QUDA `dirac_mobius.cpp` 约定，必须与
`dslash_domain_wall_m5.cuh` 一起看。

**优点：** 在相同 (L_s) 下改善近似 sign function 的质量；**缺点：** 多一组系数和第五维通信，参数调优依赖谱。
**适用：** QUDA `dirac_mobius.cpp`、`dslash5_mobius_eofa.cu`、`NEWS` 的 MSPCG/EOFA 条目；当前 PyQCU 四维 QCU ABI 未实现。

### 伪代码

\begin{table}[htbp]
\centering
\caption*{Table M.1. Möbius 5-d/4-d precondition}
\small
\begin{tabular}{@{}l@{}}
输入 (b_s,c_s,m_5,m_f,L_s)，构造 (M_5(s,s')) 与 (D_4(U))。\\
按闭式系数应用 (M_5^{-1})，避免逐 (s) 的通用矩阵求逆。\\
以对称/非对称 4-d Schur 组合 (D_4M_5^{-1}D_4)，在 parity 子空间求解。\\
恢复第五维字段，并用完整 Möbius operator 做 residual check。\\
\end{tabular}
\end{table}

## M. MR

### 最小残差平滑器

PyQCU 的 MR 采用 QUDA `inv_mr` 风格。令 (p=A^\dagger r)，(Ap=A p)，则

$$
\alpha=\omega\frac{\langle p,p\rangle}{\langle Ap,Ap\rangle},
\qquad
x\leftarrow x+\alpha p,
\qquad
r\leftarrow r-\alpha Ap.
$$

若 (A) Hermitian 可令 `matvec_dag=None`；Wilson full operator 只有 (gamma_5)-Hermiticity 时应显式传
(A^\dagger=\gamma_5A\gamma_5)。Strict MATPC 当前直接以当前 residual 做方向，相当于
(p=r) 的非对称 MR 版本。

\begin{table}[htbp]
\centering
\caption*{Table M.2. MR smoother}
\small
\begin{tabular}{@{}l@{}}
(r=b-Ax)，选择 (p=A^\dagger r)（或非对称 coarse 路径的 (p=r)）。\\
计算 (Ap)、(\alpha=\omega(p^\dagger p)/(Ap^\dagger Ap))，分母过小则停止并报告 breakdown。\\
(x\leftarrow x+\alpha p,;r\leftarrow r-\alpha Ap)，重复固定步数或直到容差。\\
作为 MG pre/post smoother 时通常不追求独立收敛，只需压低高频误差。\\
\end{tabular}
\end{table}

**优点：** O(1) 向量状态，适合非 Hermitian coarse；**缺点：** 一步需要 (A) 与 (A^\dagger)，单方向收敛慢，
(omega) 过大可能过冲；**实现来源：** `pyqcu/solver/_mr.py`、`_quda_multigrid.py:2991-3043`、
`cpp/cuda/qcu/include/lattice_clover_multigrid.h:202-240`。

## M. MultiGrid：native、legacy/compact、strict

### 三条实现的精确定义

| 路径 | 细层对象 | 粗层对象 | parity 位置 | 入口 |
|---|---|---|---|---|
| native/Python | Wilson/Clover full 或 parity operator | matrix-free 或显式 33-tensor | 可选，旧 `multigrid` 层级由 `use_parity` 控制 | `pyqcu/solver/_multigrid.py` |
| legacy/compact | 首层可为 (S_o)，后续 (R S_oP) | `[2,4,E,E,...]` hopping/diag/sitting，旧 33-tensor | 粗层常 compact，不能误当 full QUDA coarse | `QudaMultigrid(hierarchy_mode="legacy", setup_operator="schur")` |
| strict-QCU | full (D_l=X_l+H_l)，fine 边界用 MATPC | full (X,Y,Yhat)，coarse spin=2 | 仅 MATPC/R/P 视图裁剪 | `QudaStrictMultigrid`、`applyMultigridStrict*Qcu` |

native 的 `local_orthogonalize/restrict/prolong` 是 Python 参考；legacy 的 `build_stencil` 将
odd-Schur 的 nearest/diagonal 项打包为 33 tensors；strict 则遵循 QUDA full coarse 语义，
不减半 coarse geometry、不把粗算子替换成 hopping-only dslash。

### Native/legacy 的 V-cycle

设 (A_l) 为第 (l) 层算子，(S_l) 为平滑器，(C_{l+1}) 为粗求解：

$$
M_l^{V}=S_l^{post}\left[I+P_lC_{l+1}R_l\left(I-A_lS_l^{pre}\right)\right]S_l^{pre}.
$$

legacy 33-tensor 的 odd-Schur 常见结构是

$$
S_o=D_{oo}-\kappa^2H_{oe}D_{ee}^{-1}H_{eo},
\qquad
A_c=R S_o P,
$$

并把 (pm\\hat\mu) nearest block 与“同 coarse site、不同 fine parity”的六类对角 block 分开存储。

\begin{table}[htbp]
\centering
\caption*{Table M.3. Native/legacy 33-tensor V-cycle}
\small
\begin{tabular}{@{}l@{}}
若 (l=0)，以 C++ Clover BiStabCG 或 Python operator 计算 fine residual；否则应用当前粗 dslash。\\
执行 (
u_{pre}) 次 MR/CG/BiCGStab 平滑，得到 (r=b-Ax)。\\
`restrict`：(r_c=Rr)，若 legacy Schur 则只取 odd compact 视图。\\
coarse solve：粗层应用 (A_c v)，可用 BiStabCG；最粗层按 `coarse_max_iter` 或 direct solve 结束。\\
`prolong`：(e=P e_c)，更新 (x\leftarrow x+e)，再执行 (
u_{post}) 次平滑。\\
粗校正后若继续 BiCGStab，重置全部递推状态；记录校正前后 residual。\\
\end{tabular}
\end{table}

### Strict full/X-Y-Yhat V-cycle

Strict 层级的关键递归为

$$
D_l=X_l+H_l,
\quad \widehat H_l=X_l^{-1}H_l,
\quad M_{l,p}=I-\widehat H_{l,pq}\widehat H_{l,qp},
\quad D_{l+1}=R_lX_l^{-1}D_lP_l.
$$

粗层 (Yhat) 的 forward/backward 次序分别为

$$
\widehat Y^f=X^{-1}Y^f,
\qquad
\widehat Y^b=Y^bX^{-\dagger}_{\rm neighbor}.
$$

Strict 运行期生命周期固定为

$$
\texttt{hierarchy.setup()}\to\texttt{CudaSchurOp}\to\text{bind assets}
\to\texttt{applyMultigridStrictInitQcu}\to\text{V-cycle/FGMRES}
\to\texttt{applyMultigridStrictEndQcu}\to\texttt{release()}.
$$

\begin{table}[htbp]
\centering
\caption*{Table M.4. Strict full-coarse V-cycle}
\small
\begin{tabular}{@{}l@{}}
输入 fine full RHS，`prepare` 在 target parity 上形成 (b_p^S)，但 transfer coarse field 保持 full。\\
pre-smooth：在 (M_{l,p}=I-\widehat H_{pq}\widehat H_{qp}) 上做固定步 MR。\\
计算 (r_p=b_p^S-M_{l,p}x_p)，以 `restrict_parity` 形成完整 coarse rhs。\\
递归 child level；coarsest 用 direct-PC/FGMRES/BiCGStab，热启动由 `params[57]` 控制。\\
`prolong_parity` 将 coarse correction 写回 target fine parity，post-smooth。\\
若外层是 fused right-FGMRES，保存每轮 (z_j=M_j^{-1}v_j)，并在重启点用 full true residual 验收。\\
释放 runtime assets 后再 `CudaSchurOp.release()`；同一实例的 `_SET_INDEX_` 不跨 strict 调用递增。\\
\end{tabular}
\end{table}

### Cycle、smoother 与 solver 选择

$$
\begin{aligned}
C_l^V&=S_{post}P_lC_{l+1}^VR_lS_{pre},\\
C_l^W&=S_{post}P_lC_{l+1}^WP_lC_{l+1}^WR_lS_{pre},\\
C_l^F&=S_{post}P_lC_{l+1}^FP_lC_{l+1}^VR_lS_{pre}.
\end{aligned}
$$

V-cycle 成本最低；W-cycle 对粗空间弱时更稳但粗解次数增加；F-cycle 常用于 setup/早期层；K-cycle
在 child 内做短 Krylov，必须明确其预条件器是否变化。QUDA `MG::createSmoother/createCoarseSolver`
在 `refer/git-rep/quda/lib/multigrid.cpp:273-666`，主递归在 `:1131-1224`；当前 PyQCU strict
主路径是 MR + V-cycle + FGMRES。

### 优缺点与使用边界

- **native：** 代码最简单、可在 CPU/CUDA/NPU 验证；但粗层自由度布局和 operator 选择较旧，性能与 QUDA 不可直接类比。
- **legacy/compact：** odd-Schur 和 33-tensor 成熟、缓存小；但粗层不是完整 full dslash，不能用于验证 QUDA strict 的 full coarse 语义。
- **strict：** (X/Y/Yhat)、full coarse、MATPC 和生命周期与 QUDA 对齐；当前 strict Galerkin 原型只证明单 MPI rank、Wilson/Clover nearest-neighbour，分布式 halo/fused solve 必须 fail-closed。

**来源：** `pyqcu/solver/_multigrid.py`；`pyqcu/solver/_quda_multigrid.py:1985-2350,2687-3060`；
`pyqcu/tools/_multigrid.py:738-900,1279-1340`；`cpp/cuda/qcu/src/apply_multigrid.cu`、
`apply_multigrid_strict.cu`；`docs/report_multigrid_quda_pyqcu_20260909.md`。

## O. Overlap

### Neuberger 算子

Overlap 费米子利用 Wilson kernel 的 sign function 实现精确有限格距手征：

$$
D_{ov}(m)=\left(1-\frac{am}{2}\right)D_{ov}(0)+am,
\qquad
D_{ov}(0)=1+\gamma_5\,\operatorname{sign}\!\left(H_W(-m_0)\right),
$$

其中 (H_W=\gamma_5D_W(-m_0)) 为 Hermitian kernel。数值上用 rational/Zolotarev 或 Chebyshev
近似 sign，并以内层多移位 CG 求 ((H_W^2+\sigma_i)^{-1})。

### 伪代码与边界

\begin{table}[htbp]
\centering
\caption*{Table O.1. Overlap sign-function solve}
\small
\begin{tabular}{@{}l@{}}
输入 (U,m,m_0)，构造 (H_W=\gamma_5D_W(-m_0))。\\
估计 (H_W^2) 的谱区间，选择 Zolotarev/Chebyshev/rational coefficients。\\
对每个 shift (sigma_i) 用 multi-shift CG 解 ((H_W^2+\sigma_i)x_i=b)。\\
组合 (\operatorname{sign}(H_W)b\approx H_W\sum_i\omega_ix_i)。\\
外层以 FGMRES/BiCGStab 解 (D_{ov}(m)x=b)，每次 matvec 调 sign approximation。\\
以 Ginsparg–Wilson defect (|\gamma_5D+D\gamma_5-aD\gamma_5D\|) 和 true residual 验收。\\
\end{tabular}
\end{table}

**优点：** 手征对称精确、拓扑零模物理清楚；**缺点：** 每次外层 matvec 包含内层多移位 solve，计算和内存昂贵，
谱下界误估会破坏 sign 精度。
**仓库结论：** 当前 `refer/git-rep/quda` 快照 README 列出的主作用量没有 Overlap 入口，
`rg` 未找到 `DiracOverlap`/`sign(H_W)` 生产实现；因此本节是理论参考，不把 Overlap 写入 PyQCU 已实现功能。

## Q. QR、CGS 与局部基

### 关系

局部 aggregate 的 null vectors 需要 (V_X^\dagger V_X\approx I)。可用 Householder QR、MGS、重复 CGS 或
批量 `torch.linalg.qr`。QR 的稳定性最好但需要更多 workspace；重复 CGS 更适合固定小 block 和 GPU einsum。

\begin{table}[htbp]
\centering
\caption*{Table Q.1. 局部 QR/重复 CGS}
\small
\begin{tabular}{@{}l@{}}
对 aggregate (X) 取候选列 (B_X=[b_1,\ldots,b_k])。\\
QR 路径：(B_X=Q_XR_X)，保留 (Q_X) 作为 (V_X)，检查 (Q_X^\dagger Q_X-I)。\\
MGS 路径：(q_j=b_j-\sum_{i<j}q_i(q_i^\dagger b_j))，归一化后再重复一次。\\
若 (|q_j|) 低于阈值，注入随机/test vector 或减少 coarse dof，避免零列污染 Galerkin。\\
按 fine spin 到 coarse spin 的 map（Wilson/Clover 为 (4\to2)）写入 blocked ABI。\\
\end{tabular}
\end{table}

**实现来源：** `pyqcu/tools/_multigrid.py:69-110`；`pyqcu/solver/_quda_multigrid.py:441-500,538-610`；
`refer/git-rep/quda/lib/block_orthogonalize.in.cpp`、`lib/transfer.cpp`。

## Q. QUDA coarse-op

### `RDP` 到 `X/Y`

QUDA coarse-op 的核心不是把 SU(3) Gauge 直接降采样，而是对 transfer basis 做

$$
D_c=R D_f P,
\qquad
(UV)_\mu^{s,c'}(x)=\sum_c U_\mu^c(x)V_\mu^{s,c'}(x+\hat\mu),
$$

再完成 (V^\dagger UV)、Clover/identity/mass/twist local terms 和 storage conversion。对 Wilson/Clover，
coarse color 是 `nvec`，coarse spin 通常 (N_s^{coarse}=2)；粗 link 的 `Y` 是矩阵而不是 SU(3)。

### QUDA preconditioned coarse operator

`DiracCoarse::createCoarseOp` 生成 (Y,X)，`createYhat` 分配
(Yhat,X^{-1})。`DiracCoarsePC::Dslash` 读取 Yhat，`M` 形成

$$
M_{even}=I-\widehat H_{eo}\widehat H_{oe},
\qquad
M_{odd}=I-\widehat H_{oe}\widehat H_{eo}.
$$

backward `Yhat` 必须右乘邻点 (X^{-\dagger})，forward 必须左乘本点 (X^{-1})。

\begin{table}[htbp]
\centering
\caption*{Table Q.2. QUDA coarse-op setup/apply}
\small
\begin{tabular}{@{}l@{}}
读取 fine Gauge/Clover、Transfer (V/R)、coarse geometry 和 action flags。\\
交换 (V) ghost，计算每个 direction 的 UV 与 VUV；按 coarse displacement 累加 onsite (X) 和 links (Y^f,Y^b)。\\
把 identity、mass、twisted-mass、Clover 局部项各加一次，避免 internal hopping 重复计入 (X)。\\
生成 (X^{-1})，构造 forward (Yhat=X^{-1}Y^f)、backward (Yhat=Y^bX^{-\dagger}_{neighbor})。\\
应用 full coarse dslash 时读取 (X,Y)；应用 coarse-PC/MATPC 时读取 (X^{-1},Yhat)。\\
按 parity 做两次 hopping 得 (I-\widehat H_{pq}\widehat H_{qp})，prepare/reconstruct 采用同一 block elimination。\\
\end{tabular}
\end{table}

**优点：** action-specific local terms 和多层递归完整，多 GPU ghost/tuning 成熟；**缺点：** setup kernel 与 field order 复杂，
不同 action 的 coarse support/自旋约定不可混用。
**源码来源：** `refer/git-rep/quda/lib/coarse_op.cuh:1043-1457`、`include/kernels/coarse_op_kernel.cuh`、
`lib/dirac_coarse.cpp:121-336,387-649`、`lib/multigrid.cpp:1131-1466`。

## S. SAP

### Schwarz alternating procedure

SAP 把格点划分为重叠块 (Omega_i)，在每个块上近似解局部 Dirac 方程，再按固定顺序更新：

$$
M_{\rm SAP}^{-1}r
=\sum_{i=1}^{N_b}P_iA_i^{-1}R_ir
\quad\text{（加性近似）},
$$

乘性/alternating 版本则把第 (i) 块的修正立即作用到下一块的 residual。块内可以用 MR、CG、BiCGStab 或短 GCR；边界重叠宽度决定局部解对低频误差的覆盖。

\begin{table}[htbp]
\centering
\caption*{Table S.1. SAP smoother}
\small
\begin{tabular}{@{}l@{}}
选定块集合 (Omega_i)、重叠宽度和 local solver tolerance。\\
计算 (r_i=R_i(b-Ax))，解 (A_idelta_i=r_i)（通常只做固定少量迭代）。\\
乘性 SAP：(x\leftarrow x+P_idelta_i)，立即更新全局 residual；加性 SAP：并行求所有 (delta_i) 后累加。\\
在 MPI 中交换 overlap halo，完成一轮后测量高频 residual 衰减。\\
将 SAP 作为 FGMRES/MG 的可变右预条件器，不能在 flexible 外层假定每轮相同。\\
\end{tabular}
\end{table}

**优点：** 局部性强、通信可与计算重叠、适合并行平滑；**缺点：** 块边界误差和顺序依赖，重叠增加内存；
**仓库状态：** QUDA `NEWS`/`invert_param.overlap` 支持 domain-overlap 预条件，PyQCU 当前公开 solver 以 MR/MG 为主，未提供独立 `sap()` API。

## S. Schwarz

### Additive/Multiplicative Schwarz

统一写成

$$
M_{AS}^{-1}=\sum_iP_iA_i^{-1}R_i,
\qquad
M_{RAS}^{-1}=\sum_iP_iA_i^{-1}\widetilde R_i,
$$

其中 (widetilde R_i) 只保留 non-overlap 归属。Multiplicative Schwarz 依赖块顺序，通常收敛强但并行度低；additive/RAS 可并行但需要外层 Krylov 消化块间耦合。

**与 SAP 的边界：** SAP 是一种按序实现的 Schwarz smoother；Schwarz 是预条件器家族，SAP 是其中的调度策略。
在 QUDA coarse solver 中 `preconditioner` 可以是递归 MG；在本库 strict 外层，MR V-cycle 作为变化的 Schwarz-like preconditioner，
由 FGMRES 而非固定 CG 包装。

**优缺点：** Schwarz 对局部强耦合和多 GPU 友好，但块太小无法消除低频误差，块太大又接近原问题。

## S. Staggered fermions

### 作用量

Kogut–Susskind 一分量作用量为

$$
S_{\rm stag}=\sum_x\bar\chi(x)\left[m\chi(x)+\frac12\sum_\mu\eta_\mu(x)
\big(U_\mu(x)\chi(x+\hat\mu)-U_\mu^\dagger(x-\hat\mu)\chi(x-\hat\mu)\big)\right].
$$

四味 taste 在连续极限恢复；有限 (a) 有 taste breaking。它没有 Wilson 的显式 spin 维，粗化时
coarse spin block 规则、Kähler-Dirac preconditioning 和 (X\) 的质量项都不同，不能直接套用 Wilson/Clover
`coarse_spin=2` 的 strict ABI。

### 伪代码与比较

\begin{table}[htbp]
\centering
\caption*{Table S.2. Staggered dslash}
\small
\begin{tabular}{@{}l@{}}
输入一分量 color field (chi)、(U)、质量 (m)，预计算 (eta_\mu(x))。\\
按 forward/backward gather 应用 (U_\mu) 和 (U_\mu^\dagger)，累加反对称 hopping。\\
若使用 even-odd，质量项在 diagonal，hopping 连接相反 parity。\\
求解可用 CG on (D^\dagger D)、multi-shift CG 或 staggered MG；恢复 taste observables 时保持相位约定。\\
\end{tabular}
\end{table}

**优点：** 自由度少、CG 结构简洁；**缺点：** taste breaking、rooting/RHMC 解释复杂，改进长链通信昂贵。
**来源：** `refer/git-rep/quda/lib/dirac_staggered_kd.cpp`、`dirac_improved_staggered_kd.cpp`、
`staggered_coarse_op.in.cpp`；`refer/git-rep/quda/README.md:14-16`。PyQCU 主 `dslash` 包只提供 Wilson/Clover。

## S. Strict-QCU coarse-op

### 语义与资产

Strict-QCU 对应 QUDA full-coarse 而不是旧 33-tensor odd-Schur：

$$
D_c=X+H_c,
\quad
H_c\phi(X)=\sum_\mu\left[Y_\mu^f(X)\phi(X+\hat\mu)+Y_\mu^b(X)\phi(X-\hat\mu)\right].
$$

运行期保存

$$
\{Yhat^f,Yhat^b,X,X^{-1},V_{l\to l+1}\},
$$

raw (Y) 可丢弃。Strict `set_ptrs` 每个 transition 有四槽：preconditioned links、onsite pair、可选 raw links、blocked null。

### 构造与应用伪代码

\begin{table}[htbp]
\centering
\caption*{Table S.3. Strict-QCU coarse-op}
\small
\begin{tabular}{@{}l@{}}
校验 fine field 为 full lattice、coarse spin=2、gamma basis 与 QUDA DeGrand–Rossi 一致；若 support 非 nearest-neighbour，立即 fail-closed。\\
把 canonical (V[n_v,4,3,X,Y,Z,T]) 转为 C-order blocked `[E,12,Xc,bx,Yc,by,Zc,bz,Tc,bt]`。\\
按 site-batch/colored probes 计算 (R(X_l^{-1}D_l)P)，提取 (X,Y^f,Y^b)。\\
批量 inverse (X)，按方向构造 (Yhat^f=X^{-1}Y^f)、(Yhat^b=Y^bX^{-\dagger}_{neighbor})。\\
将 (Yhat,(X,X^{-1}),V) bind 到 `set_ptrs`；seal 后不再保留 native setup duplicate。\\
`applyMultigridStrictCoarseQcu`：full (D_c)；`StrictMatPC`：两次 (Yhat) hopping 后输出 (I-Hhat_{pq}Hhat_{qp})。\\
`StrictPrepare/Reconstruct`：只在 fine target parity 做 Schur 消元和恢复；coarse R/P 仍读 full geometry。\\
`StrictVCycle/Fgmres`：递归 correction + MR/FGMRES；结束时 `StrictEnd` 释放 workspace，再释放 Schur op。\\
\end{tabular}
\end{table}

### 与 legacy/compact 的差别

| 项目 | legacy/compact | strict-QCU |
|---|---|---|
| 投影算子 | 常见 (R S_oP) | (R X^{-1}DP) |
| 粗几何 | odd compact 或旧 33 tensor | full coarse |
| 粗数据 | `hop_nn/hop_diag/sit` | (X,Y,Yhat,X^{-1}) |
| parity | 粗层常已裁剪 | 只在 MATPC/R/P 边界裁剪 |
| 适用 action | PyQCU Wilson/Clover Schur | 当前只证明 Wilson/Clover nearest-neighbour |
| MPI | legacy halo 路径 | strict setup/fused solve 当前单 rank fail-closed |

**实现来源：** `pyqcu/tools/_strict_galerkin.py:310-538,553-800`；
`pyqcu/solver/_quda_multigrid.py:1265-1320,2715-2905,3193-3286`；
`cpp/cuda/qcu/src/apply_multigrid_strict.cu`；`skills/qcu/SKILL.md`。

## S. Symmetric Schur

### 定义

在 asymmetric Schur 基础上左右吸收局部块逆，可得到

$$
S_p^{\rm sym}=I-A_p^{-1}B_{pq}A_q^{-1}B_{qp}.
$$

对 Wilson/Clover 的 (-\kappa H) 约定，

$$
S_p^{\rm sym}=I-\kappa^2A_p^{-1}H_{pq}A_q^{-1}H_{qp}.
$$

它的优点是 diagonal 变为 identity，适合与 Hermitian transform 或 CG 家族组合；但只有当
(A_p,A_q) 与 hopping 满足相应共轭关系时才可当作 Hermitian positive definite。

### 伪代码

\begin{table}[htbp]
\centering
\caption*{Table S.4. Symmetric Schur}
\small
\begin{tabular}{@{}l@{}}
对 full RHS 先算 (u_q=A_q^{-1}b_q)，(b_p^S=b_p-B_{pq}u_q)。\\
再左乘 (A_p^{-1})：(\widehat b_p=A_p^{-1}b_p^S)。\\
应用 (I-A_p^{-1}B_{pq}A_q^{-1}B_{qp})；每次 matvec 依次做 (B_{qp})、(A_q^{-1})、(B_{pq})、(A_p^{-1})。\\
用 CG 仅在 Hermitian/SPD 证据充分时；否则使用 BiCGStab/GCR/FGMRES。\\
解出 (x_p) 后按非对称恢复式恢复 (x_q)，以 full operator 检查。\\
\end{tabular}
\end{table}

**来源：** `refer/git-rep/quda/lib/dirac_clover.cpp:174-251`；`dirac_coarse.cpp:530-626`；
`pyqcu/solver/_quda_multigrid.py` 的 `QudaMatPCOperator`。对照 QUDA 时必须同时记录 `solve_type`、`solution_type` 和 `matpc_type`。

## T. Twisted-mass（含 non-degenerate pairs）

### 简并双重态

twisted-mass 在 Wilson 核加入 flavor 非简并的手征旋转质量：

$$
D_{tm}=D_W(m_0)+i\mu\gamma_5\tau_3.
$$

在 maximal twist，物理质量由 (mu) 控制，自动消除部分 on-shell (O(a)) 误差；(\tau_3) 使两个 flavor 的 twist 符号相反。

### non-degenerate doublet

常用非简并双重态写作

$$
D_{nd}=D_W(m_0)+i\mu_\sigma\gamma_5\tau_1+\mu_\delta\tau_3,
$$

其中 (mu_\sigma) 控制平均质量，(mu_\delta) 控制质量分裂。不同文献可交换 flavor Pauli 矩阵，比较代码时以 QUDA flavor block 和 dagger 约定为准。

\begin{table}[htbp]
\centering
\caption*{Table T.1. Twisted-mass solver}
\small
\begin{tabular}{@{}l@{}}
构造 Wilson hopping 和 Clover/identity diagonal block。\\
在 flavor block 加 (+i\mu\gamma_5) 与 (-i\mu\gamma_5)，或加入 (i\mu_\sigma\gamma_5\tau_1+\mu_\delta\tau_3)。\\
选择 direct twisted solve、Hermitian (gamma_5\tau) 变换或 normal equation；确认 even-odd block 的 flavor inverse。\\
以 BiCGStab/GCR/CGNR 求解，粗化时把 flavor/twist local terms 一次性加入 coarse (X)。\\
\end{tabular}
\end{table}

**优点：** maximal twist 的 (O(a)) 改进，non-degenerate pair 可同时描述两种质量；**缺点：** flavor block 使局部 inverse 和 coarse (X) 更大，符号/dagger 错误会破坏物理质量。
**实现状态：** QUDA `lib/dirac_twisted_mass.cpp`、`dslash_ndeg_twisted_mass.cpp`、README:12；PyQCU 当前没有对应的公开 dslash/params 入口，属于参考实现。

## T. Twisted-mass with a clover term

### 作用量

把 Clover 改进和 twist 合并：

$$
D_{tmC}=D_W(m_0)+C_{SW}+i\mu\gamma_5\tau_3,
$$

non-degenerate 时替换为 (i\mu_\sigma\gamma_5\tau_1+\mu_\delta\tau_3)。Clover 项仍为 spin-color local block，twist 是 flavor-spin local block，二者在 onsite (X) 中相加后再求逆。

### 算法要点

\begin{table}[htbp]
\centering
\caption*{Table T.2. Twisted-clover}
\small
\begin{tabular}{@{}l@{}}
构造 (A_p=I+C_p+i\mu\gamma_5\tau_3)（或 non-degenerate flavor block）。\\
对 (A_e,A_o) 做批量 inverse，形成 asymmetric/symmetric Schur。\\
Galerkin 时计算 (V^\dagger(A+H)V)，Clover 与 twist 只能进入 onsite (X)，不得重复加入 hopping。\\
粗层 coarse (X) 保留 spin/flavor 结构；coarse-PC 的 (Yhat) 左右次序与 Clover-PC 相同。\\
用 direct BiCGStab/GCR 或 FGMRES，full true residual 必须用同一 twist/clover convention 计算。\\
\end{tabular}
\end{table}

**优点：** 同时获得 twist 的质量/手征性质与 Clover 的 (O(a)) 改进；**缺点：** 局部矩阵更大且非 Hermitian，
粗化/预条件复杂。**来源：** `refer/git-rep/quda/lib/dirac_twisted_clover.cpp`、`dslash_twisted_clover_preconditioned.hpp`、
`README.md:12-13,28-29`；PyQCU 目前未实现生产入口。

## W. Wilson

### 作用量与 hopping

Wilson 费米子消除 doublers 的离散作用量可写为

$$
(D_W\psi)(x)=(m_0+4r)\psi(x)
-\frac12\sum_\mu\left[(r-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
+(r+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right].
$$

以 (r=1)、(kappa=1/(2m_0+8)) 归一化后常写成

$$
D_W=I-\kappa H,
$$

$$
(H\psi)(x)=\sum_\mu\left[(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right].
$$

(P_\pm^\mu=(1\pm\gamma_\mu)/2) 把四分量旋量投影到两分量，QUDA CUDA kernel 直接在投影后做 color matvec；
PyQCU `_wilson.py` 以 `einsum` 和 `torch.roll`/MPI halo 实现同一结构。

### 伪代码与实现

\begin{table}[htbp]
\centering
\caption*{Table W.1. Wilson dslash 与求解}
\small
\begin{tabular}{@{}l@{}}
输入 (U_\mu,psi,\kappa)，按最后四轴建立 periodic/anti-periodic neighbor。\\
对每个 (mu)：取 (\psi(x+\hat\mu))，应用 ((1-\gamma_\mu)) 和 (U_\mu(x))；取 (\psi(x-\hat\mu))，应用 ((1+\gamma_\mu)) 和 (U_\mu^\dagger(x-\hat\mu))。\\
累加 (H\psi)，输出 (D\psi=\psi-\kappa H\psi)（或仅输出 hopping，需显式注明符号）。\\
偶奇路径只让 (D_{eo}) 或 (D_{oe}) 作用；完整 solve 选择 CG on (D^\dagger D) 或 BiCGStab/FGMRES。\\
C++ QCU 生命周期：`applyInitQcu` → dslash/solver → 递增 `_SET_INDEX_` → `applyEndQcu`。\\
\end{tabular}
\end{table}

**优点：** stencil 最近邻、spin projection 高效、实现和验证简单；**缺点：** 显式破坏有限 (a) 手征，临界质量有加性重整化，
condition number 在轻质量时恶化。**来源：** `pyqcu/dslash/_wilson.py`、`_operator.py`；
`cpp/cuda/qcu/include/wilson_dslash.h`、`lattice_wilson_dslash.h`；`refer/git-rep/quda/include/kernels/dslash_wilson.cuh`；
`examples/qcu/single_qcu_wilson_dslash.py`、`examples/pyquda/test_wilson_dslash.py`。

## X. 统一比较、验收与来源

### X.1 算法选择速查

| 问题性质 | 首选 | 不能直接替代的算法 |
|---|---|---|
| SPD (D^\dagger D) | CG/CA-CG/multi-shift CG | 非 Hermitian 原始 (D) 上裸 CG |
| 一般非 Hermitian | BiCGStab/GCR/FGMRES | 把递推 residual 当 true residual |
| 变化 MG/SAP 预条件 | FGMRES 或 flexible-GCR | 固定预条件 CG |
| 粗层高频平滑 | MR/Chebyshev/SAP/Schwarz | 只用粗 solver 代替 smoother |
| fine Clover parity | asymmetric/symmetric Schur + prepare/reconstruct | 只在一个 parity 上算 full residual |
| QUDA 对齐 coarse | full (X/Y/Yhat)+MATPC | legacy odd compact 当 strict full |
| 长链 staggered/HISQ | QUDA action-specific coarse op | strict 最近邻 X/Y ABI |
| 拓扑近零模 | MG/deflation/TR-Lanczos | 只增加 Krylov restart 而不改低模 |

### X.2 本库当前可复现实证

- `pyqcu/solver/_bistabcg.py`：breakdown guard、history、absolute/relative tolerance；c64 递推残差必须用 full true residual 复核。
- `pyqcu/solver/_gmres.py`：FGMRES(m) 的 Arnoldi + 复 Givens + restart；零 RHS/zero residual 有显式短路。
- `pyqcu/solver/_mr.py`：MR 需要 `matvec_dag` 语义，适合 smoother 而非无条件全局 solver。
- `pyqcu/solver/_cacg.py`：MGS 基 + block least-residual 的 CA-CG，和 QUDA power-basis CA-CG 有意不同。
- `pyqcu/solver/_lanczos.py`：thick-restart Lanczos，显式对称 (H)、Ritz true-residual 双闸门。
- `pyqcu/tools/_multigrid.py`：null vector、局部正交、R/P、33-tensor stencil；`_strict_galerkin.py`：strict full (X/Y/Yhat)。
- QCU/Cython：`cpp/cuda/qcu/python/pyqcu.h` 与 `pyqcu/cuda/qcu/qcu.pyx` 覆盖 Wilson/Clover dslash、CG/BiStabCG、legacy transfer/coarse dslash、strict coarse/MATPC/FGMRES。

### X.3 物理与工程验收清单

\begin{table}[htbp]
\centering
\caption*{Table X.1. 任何新增 solver/action/coarse-op 的验收闸门}
\small
\begin{tabular}{@{}l@{}}
先做量纲和极限检查：(U=I)、(C=0)、(mu=0)、(L_s\to1)、(P^\dagger P\to I)、平凡 parity。\\
检查对称性：Wilson/Clover 的 \(\gamma_5\)-Hermiticity、Schur 的 dagger 顺序、twisted flavor 共轭、coarse (R=P^\dagger)。\\
检查 support：最近邻只产生 (X/Y)，长链必须显式进入 stencil 或 fail-closed。\\
检查 solver：估计 residual 与 (\|b-Ax\|) 分离记录；breakdown 不能吞掉 NaN；mixed precision 必须可靠刷新。\\
检查布局：PyQCU `xyzt`、HDF5 `tzyx`、QUDA parity/order、blocked C++ ABI 完成 round-trip。\\
检查资源：每线程独立 `params/argv/set_ptrs`；Strict close 后 live bytes 回到 baseline；不以 `empty_cache()` 代替泄漏证明。\\
检查接口生命周期：每次普通 QCU 操作递增 `_SET_INDEX_`；Strict 同一 hierarchy 生命周期内保持实例索引语义。\\
\end{tabular}
\end{table}

### X.4 参考源

**仓库源码与本库文档**

1. `pyqcu/solver/_bistabcg.py`、`_gmres.py`、`_mr.py`、`_cacg.py`、`_lanczos.py`、`_multishift_cg.py`；
2. `pyqcu/dslash/_wilson.py`、`_clover.py`、`_operator.py`；
3. `pyqcu/tools/_multigrid.py`、`_strict_galerkin.py`；
4. `pyqcu/solver/_multigrid.py`、`_quda_multigrid.py`；
5. `cpp/cuda/qcu/include/{bistabcg.h,cg.h,lattice_clover_multigrid.h}`、`src/apply_multigrid_strict.cu`、`python/pyqcu.h`；
6. `refer/git-rep/quda/README.md`、`NEWS`、`lib/multigrid.cpp`、`lib/dirac_coarse.cpp`、`lib/coarse_op.cuh`、
   `lib/dirac_clover.cpp`、`lib/dirac_twisted_mass.cpp`、`lib/dirac_twisted_clover.cpp`、
   `lib/dirac_staggered_kd.cpp`、`lib/dirac_improved_staggered_kd.cpp`、`lib/dirac_domain_wall_4d.cpp`、
   `lib/dirac_mobius.cpp`、`lib/inv_gcr_quda.cpp`、`lib/inv_ca_gcr.cpp`、`lib/inv_bicgstabl_quda.cpp`；
7. `docs/report_multigrid_quda_pyqcu_20260909.md`、`docs/report_pyqcu_mg_operator_construction_20260830.tex`、
   `examples/qcu/dev87/comparison_matrix.md`、`skills/qcu/SKILL.md`、`skills/pyquda/SKILL.md`；
8. 补充日志：`logs/v20260825.txt`、`v20260826.txt`、`v20260827.txt`、`v20260829.txt`、`v20260902.txt`、
   `v20260906.txt`、`v20260907.txt`、`v20260909.txt`、`v20260911.txt`、`v20260912.txt`。日志只作为任务/实测上下文，
   公式和实现结论以源码为准。

**理论参考**

- K. G. Wilson, *Confinement of quarks*, Phys. Rev. D 10 (1974) 2445（Wilson action）；
- B. Sheikholeslami and R. Wohlert, Nucl. Phys. B259 (1985) 572（Clover improvement）；
- L. H. Karsten and J. Smit, Nucl. Phys. B183 (1981) 103；Kogut–Susskind staggered fermions；
- G. P. Lepage, Phys. Rev. D59 (1999) 074502；Follana et al., Phys. Rev. D75 (2007) 054502（Asqtad/HISQ）；
- D. B. Kaplan, Phys. Lett. B288 (1992) 342；Y. Shamir, Nucl. Phys. B406 (1993) 90（Domain-wall）；
- R. C. Brower, H. Neff and K. Orginos, Nucl. Phys. B (Proc. Suppl.) 153 (2006) 3（Möbius）；
- H. Neuberger, Phys. Lett. B417 (1998) 141（Overlap）；
- Y. Saad and M. H. Schultz, SIAM J. Sci. Stat. Comput. 7 (1986) 856（GMRES/Arnoldi）；
- H. A. van der Vorst, SIAM J. Sci. Stat. Comput. 13 (1992) 631（BiCGStab）；
- 格点 QCD 理论与实现映射：`/Users/zhangxin/MyQCD/docs/report_pyqcd_lattice_qcd_theory_20260906.tex`。

### X.5 结论边界

本报告已经把任务列出的算法、作用量和粗算子变体逐项分开，给出公式、优缺点、适用范围和源码锚点。
其中 Wilson/Clover、Python/native MG、legacy/compact MG、strict full coarse、Galerkin、MATPC、MR、CG、
BiCGStab、FGMRES、局部 CGS/QR 和 QCU 生命周期属于本仓库可直接追溯对象；Twisted-mass、staggered/HISQ、
Domain-wall、Möbius、Overlap 主要由 QUDA 快照和理论文档提供参考，当前 PyQCU 没有对应的完整生产 API。
性能结论仍应以 `examples/qcu/dev87` 的具体设备、精度、边界条件和 true residual 数据为准，不由公式推断加速比。
