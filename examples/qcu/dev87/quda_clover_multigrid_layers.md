# 以 Clover Dslash 为实例理解 QUDA MultiGrid 各层算子

本文是 `dev87_report.md` 第 22 节的独立 Markdown 版本，目标是把 QUDA
MultiGrid 中容易被“黑盒算子”掩盖的细节写成一条连续的数学与实现链：细层
Clover/Gauge 算子、奇偶 Schur、null vector、$P/R$ 转移、粗层 $X/Y/Yhat$、递归粗化以及 V-cycle 中的各种预处理；同时把这些对象与 PyQCU 当前的
33-stencil、FGMRES、MR/CG/Chebyshev 及 V/W/F/K 路径逐项对比。

本文采用三种证据标记：`[确证]` 表示可由当前源码的调用或 kernel 逐行读出，
`[推断]` 表示把 storage/index 翻译为矩阵记号后的代数解释，`[未验证]` 表示
尚未进行逐 block、逐方向、逐 parity 的数值锚定。文中的“优势/劣势”是针对
算子性质、同步、存储和适用条件的工程分析，不等同于任何单一 GPU 上的普遍
性能结论。

## 1. 统一记号与 QUDA 的对象层次

对第 $\ell$ 层定义

$$
D_\ell = X_\ell - \kappa\left(\bar Y^f_\ell+\bar Y^b_\ell\right),
\qquad
D_{\ell+1}=R_\ell D_\ell P_\ell .
$$

当第 $\ell$ 层是 Clover-PC 或 coarse-PC 时，上式表示其 raw
Galerkin 分解；实际应用的 PC operator 是 $X_\ell^{-1}D_\ell$（或对称的
等价组合），方向 storage 改用 $\widehat Y_\ell$。因此不能把所有层都机械地
解释成同一个“$X-\kappa Y$ 加一次 dslash”的代码路径。

细层的特殊之处是

$$
D_0\equiv D_f=C_f-\kappa H_f,
$$

其中：

| 符号 | 含义 | QUDA 中的实际角色 |
|---|---|---|
| $C_f$ | 每个细格点上的 12×12 Clover onsite 矩阵 | `CloverField` 及其偶/奇子格逆 |
| $H_f$ | Wilson hopping，只连接相反奇偶格点 | `dslash_wilson`/Clover dslash kernel |
| $P_\ell$ | 粗格向量提升到细格 | prolongator，由 null vectors 构成 |
| $R_\ell$ | 细格残差限制到粗格 | restrictor，数学上取 $P_\ell^\dagger$ |
| $X_\ell$ | onsite 与 aggregate 内部 hopping 的粗矩阵 | coarse diagonal/onsite block |
| $\bar Y^f_\ell,\bar Y^b_\ell$ | aggregate 之间的有向 hopping，尚未乘 $-\kappa$ | coarse forward/backward link-like matrices |
| $Y^{f/b}_\ell=-\kappa\bar Y^{f/b}_\ell$ | 实际进入粗算子的 hopping | coarse dslash 的方向矩阵 |
| `Yhat` | Clover-PC 后经过 $X^{-1}$ 处理的粗 hopping | 下一层递归粗化的输入 |

因此，QUDA 的粗层并不是把原始 Gauge 场简单下采样；它保存的是细层算子在
聚合子空间上的 Galerkin 投影。

## 2. 细层 Clover Dslash：先固定算子本身

忽略不同版本中的整体归一化约定，Wilson-Clover 算子可写成

$$
\begin{aligned}
(D_f\psi)(x)= {}& C_f(x)\psi(x) \\
&-\kappa\sum_{\mu=0}^{3}\left[
(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)
\right].
\end{aligned}
$$

这里 $\psi$ 同时带 spin/color，自由度为 $4\times3=12$。forward hopping
使用 $U_\mu(x)$ 与 $1-\gamma_\mu$，backward hopping 使用相邻点的反向
链 $U_\mu^\dagger(x-\hat\mu)$ 与 $1+\gamma_\mu$。投影矩阵只作用于
spin，Gauge 矩阵只作用于 color，Clover 矩阵则在一个格点内混合 spin/color。

定义格点奇偶

$$
p(x)=\left(\sum_{\mu}x_\mu\right)\bmod 2,
\qquad p(x\pm\hat\mu)=1-p(x).
$$

于是 $H_f$ 只有 off-diagonal parity block，而 Clover 项是 parity-diagonal：

$$
D_f=
\begin{pmatrix}
D_{ee} & D_{eo}\\
D_{oe} & D_{oo}
\end{pmatrix},
\qquad
D_{ee}=C_{ee},\quad D_{oo}=C_{oo}.
$$

对 odd Schur，QUDA 实际求解的对象是

$$
M_o=D_{oo}-D_{oe}D_{ee}^{-1}D_{eo},
$$

其中 $D_{ee}^{-1}$ 是逐格点 Clover inverse；even Schur 则交换 $e/o$。这也
解释了为什么 `prepare` 与 `reconstruct` 不能省略：它们负责在全格点 RHS、
Schur RHS 和全格点解之间来回消去/恢复被消掉的 parity。

### 2.1 奇偶 Schur 的伪代码

$$
\begin{array}{l}
\text{给定 }D_f=C_f-\kappa H_f,\ b=(b_e,b_o),\text{选择被保留 parity }p=o;\\
\quad b_o^{\rm Schur}=b_o-D_{oe}D_{ee}^{-1}b_e;\\
\quad \text{solve }M_o x_o=b_o^{\rm Schur};\\
\quad x_e=D_{ee}^{-1}(b_e-D_{eo}x_o);\\
\quad x=(x_e,x_o);\\
\text{若保留 }p=e,\text{则交换 }e\leftrightarrow o.\\
\end{array}
$$

`prepare` 实现前两行，Schur MultiGrid 作用于 $x_p$，`reconstruct` 实现后
两行。每次 hopping 读取的邻居仍然是 $q=1-p$，不能把 compact parity
storage 中的数组下标误当成物理 parity。

## 3. Null vectors、aggregate 与 $P/R$

把细格划分为不重叠 aggregate $B_X$，其中 $X$ 是粗格点坐标。每个
aggregate 保留 $N_v$ 个 null vectors $v_a(x)$，再压缩 spin/color 为粗
自由度 $\alpha$。更一般地写成

$$
V_{x\,s c,\,X\,\alpha},
\qquad x\in B_X,
\quad s=0,\ldots,3,
\quad c=0,\ldots,2.
$$

aggregate 内 block Gram-Schmidt 要求

$$
\sum_{x\in B_X}
V_{x,\alpha}^\dagger V_{x,\beta}
=\delta_{\alpha\beta}.
$$

若 $I_X(x)$ 是 aggregate 指示函数，则 prolongator 的作用为

$$
(P\,\phi)(x,s,c)=
\sum_{\alpha}I_X(x)V_{x,s c,\alpha}\phi(X,\alpha),
\qquad x\in B_X.
$$

对应的 restrictor 为

$$
(R\,\psi)(X,\alpha)=
\sum_{x\in B_X,s,c}V_{x,s c,\alpha}^\dagger\psi(x,s,c),
\qquad R=P^\dagger.
$$

因此，$P$ 不是复制或平均，$R$ 也不是普通 pool；二者是由 null-vector
子空间定义的带 spin/color 结构的局部基变换。奇偶 transfer 只对目标 parity
的细格点执行，但粗格完整几何仍保留，便于粗层 dslash 访问 $X\pm\hat\mu$。

## 4. 第一层粗算子：$RDP$ 每一项如何落位

Galerkin 粗化为

$$
D_1=R D_f P=R(C_f-\kappa H_f)P.
$$

按照 aggregate 之间的关系拆分：

$$
D_1=X_1+Y_1^f+Y_1^b,
$$

其中 $X_1$ 不仅是 $RC_fP$，还包含同一 aggregate 内部的 hopping：

$$
X_1(X)=R_X C_f P_X
-\kappa\,R_X H_{\rm internal}P_X.
$$

跨 aggregate 的 hopping 才存入有向矩阵 $Y^f,Y^b$：

$$
Y^f_\mu(X)= -\kappa\bar Y^f_\mu(X),
\qquad
Y^b_\mu(X)= -\kappa\bar Y^b_\mu(X).
$$

### 4.1 直接 Clover coarsening

若算子未作 Clover-PC，Clover onsite 直接出现在 Galerkin block：

$$
X_1\supset R C_f P.
$$

forward 方向的未缩放矩阵可按 QUDA kernel 的 storage 约定概括为

$$
\bar Y^f_\mu(X)=
\sum_{\substack{x\in B_X\\x+\hat\mu\in B_Y}}
V(x)^\dagger(1-\gamma_\mu)U_\mu(x)V(x+\hat\mu),
$$

其中 $Y=X+\hat\mu$。backward 项在 storage 中不是把物理反向 link
再共轭一次，而是保存从 $X-\hat\mu$ 指向 $X$ 的正向 block：

$$
\bar Y^b_\mu(X-\hat\mu)=
\sum_{\substack{x\in B_{X-\hat\mu}\\x+\hat\mu\in B_X}}
V(x)^\dagger(1+\gamma_\mu)U_\mu(x)V(x+\hat\mu).
$$

coarse dslash 在输出 $X$ 处读取该 storage block 的 dagger：

$$
\bar Y^b_\mu(X-\hat\mu)^\dagger\phi(X-\hat\mu).
$$

实际代码还要处理 parity、周期边界、halo、矩阵布局和 dagger；公式只描述
矩阵元素的来源。这样写可以避免在 setup 公式和 dslash 应用公式中重复
计算 $U^\dagger$。[确证]

### 4.2 Clover-PC coarsening：为什么需要 $AV=C^{-1}V$

在 Clover-preconditioned 路径中，细层 onsite inverse 已经进入算子定义。设

$$
A=C_f^{-1},
\qquad
AV=C_f^{-1}V.
$$

QUDA 的 direction-dependent construction 需要两侧不同的 $AV$ 放置：

$$
\bar Y^f_\mu
\sim (AV)^\dagger(1-\gamma_\mu)UV,
\qquad
\bar Y^b_\mu
\sim V^\dagger(1+\gamma_\mu)U(AV).
$$

这不是记号上的装饰：forward 与 backward 的左/右侧分别对应 Schur 消元后
的矩阵乘法顺序，所以 PC 情形必须独立构造两个方向，不能用非 PC 的
`reverse` 假设由一个方向生成另一个方向。

对粗 onsite block 批量求逆，记为 $X^{-1}$ 或 `Xinv`。PC 粗 hopping 的存储
对象是

$$
\widehat Y^f=X^{-1}Y^f,
\qquad
\widehat Y^b=Y^b X^{-\dagger}.
$$

可见 `Yhat` 不是对所有方向统一左乘 $X^{-1}$：backward 方向是右乘
$X^{-\dagger}$。这正是 Clover-PC 粗算子与普通 Wilson-like 粗算子的关键
差别。

## 5. 粗层 dslash、方向读取与奇偶

对粗格点 $X$ 和 storage parity $p$，相邻粗格点 parity 为 $q=1-p$。
粗层算子可写为

$$
\begin{aligned}
(D_c z)(X,p)={}&X_c(X,p)z(X,p)\\
&+\sum_\mu\left[
Y^f_\mu(X,p)z(X+\hat\mu,q)
+Y^b_\mu(X,p)z(X-\hat\mu,q)
\right].
\end{aligned}
$$

在 QUDA coarse kernel 的等价读取语义中：

$$
\begin{array}{ll}
\text{forward:}&\text{读取当前位置 }(X,p)\text{ 的 }Y(d+4,p,X)，\\
\text{backward:}&\text{读取 }(X-\hat\mu,q)\text{ 的 }Y(d,q,X-\hat\mu)^\dagger，\\
\text{neighbor parity:}&q=1-p\text{，对每个方向都成立。}
\end{array}
$$

实现层面还必须同时保证：

1. `Y`/`Yhat` 的 direction-major 或 site-major storage 与 kernel 一致；
2. backward 的 dagger 在正确的 source/destination site 上执行；
3. 周期边界穿越时 halo 已交换且坐标 wrap 正确；
4. coarse odd/even compact layout 与完整粗几何之间的 index map 一致；
5. 下一层 coarsening 使用本层的 $X,Yhat$，而不是再次读取原始细层
   Gauge/Clover。

## 6. 从第一层到后续层：原始 Gauge/Clover 如何消失

第一层构造需要细层的 $C_f,U$，但第二层开始，上一层的有效算子就是新的
“细层”：

$$
D_1=(X_1,Y_1^f,Y_1^b),
\qquad
D_2=R_1D_1P_1.
$$

因此后续递归只对 $X_1$、$Y_1$、必要时的 $Yhat_1$ 做 Galerkin projection。
粗层的 link-like 矩阵不再是 SU(3) Gauge 场，通常是更大的复矩阵；它只是
保留方向与邻居关系的有效 hopping。把粗矩阵误解释成原始 Gauge 会丢失：

$$
\text{aggregate 内部 hopping}\subset X,
\qquad
\text{Clover-PC 左右消元顺序}\subset Yhat.
$$

## 7. V-cycle 与外层求解器的完整逻辑

下面先给出标准 V-cycle 的数学骨架，再给出一段与用户要求的格式一致的长表格。
关键是把“算子构造”“一次粗校正”和“外层 Krylov”分成三层：外层看到的是
$M_{\rm MG}^{-1}$，它内部才会触发 $R$、粗解和 $P$；粗解内部还可能递归触发
下一层 cycle。`A_\ell^{\rm res}` 与 `A_\ell^{\rm sm}` 可以不同，因此 residual
必须由 residual operator 重新计算，不能从不一致的 smoother 递推量偷取。

$$
\begin{aligned}
x_\ell^{(1)}&=S_\ell^{\nu_{pre}}(x_\ell^{(0)},b_\ell),\\
r_\ell&=b_\ell-A_\ell^{\rm res}x_\ell^{(1)},\\
b_{\ell+1}&=R_\ell r_\ell,\\
A_{\ell+1}^{\rm res}e_{\ell+1}&=b_{\ell+1},\\
x_\ell^{(2)}&=x_\ell^{(1)}+P_\ell e_{\ell+1},\\
x_\ell^{out}&=S_\ell^{\nu_{post}}(x_\ell^{(2)},b_\ell).
\end{aligned}
$$

```latex
\begin{table}[htbp]
\centering
\caption*{以 Clover Dslash 为实例的 QUDA MultiGrid：从 setup 到右预条件外层求解的全链路伪代码}
\small
\begin{tabular}{@{}l@{}}
\text{输入：fine lattice }\Lambda_0,\ U_\mu(x),\ C_f(x),\ \kappa,\ N_v,\ b_\mu,\ L,\ \text{目标 parity }p;\quad q=1-p\\
\text{定义 }p(x)=\sum_\mu x_\mu\bmod 2,\quad D_0=C_f-\kappa H_f,\quad D_{0,pp}=C_p,\quad D_{0,pq}=-\kappa H_{pq}\\
\text{fine hopping: }(H_f\psi)_p(x)=\sum_\mu[(1-\gamma_\mu)U_\mu(x)\psi_q(x+\hat\mu)+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi_q(x-\hat\mu)]\\
\text{若求 full solution，则先选 }S_p^{asym}=C_p-\kappa^2H_{pq}C_q^{-1}H_{qp}\text{ 或 }S_p^{sym}=I-\kappa^2C_p^{-1}H_{pq}C_q^{-1}H_{qp}\\
\text{若输入是 MATPC source： }src\leftarrow b_p;\quad\text{否则 }t_q\leftarrow C_q^{-1}b_q,\quad src\leftarrow b_p+\kappa H_{pq}t_q\\
\text{若为 symmetric MATPC： }src\leftarrow C_p^{-1}src;\quad\text{inner unknown is }x_p\\
\text{for }\ell=0,1,\ldots,L-2\text{ setup:}\\
\quad\text{load/generate }B_i^{(\ell)}(x)\text{ and divide }\Lambda_\ell\text{ into aggregates }\mathcal A_X\\
\quad X_\mu=\lfloor x_\mu/b_\mu^{(\ell)}\rfloor;\quad\text{build }fine\_to\_coarse\text{ and }coarse\_to\_fine\\
\quad\text{build spin map }s_c=\lfloor s_f/2\rfloor;\quad\text{parity subset is independent: }q=1-p\\
\quad\text{for every }X,\ \chi\text{ and vector }j:\quad v_j\leftarrow B_j-\sum_{i<j}v_i\langle v_i,B_j\rangle_{\mathcal A_X,\chi};\quad v_j\leftarrow v_j/\|v_j\|_{\mathcal A_X,\chi}\\
\quad V_\ell(x)\leftarrow[v_0(x),\ldots,v_{N_v-1}(x)];\quad P_\ell\phi(x)=V_\ell(x)\phi(X(x),s_c(s_f))\\
\quad R_\ell\psi(X)\leftarrow\sum_{x\in\mathcal A_X}V_\ell(x)^\dagger\psi(x);\quad R_\ell=P_\ell^\dagger\\
\quad\text{若继续生成 coarse null vectors： }B_i^{(\ell+1)}\leftarrow R_\ell B_i^{(\ell)}\text{（或使用独立 eig/null setup）}\\
\quad\text{选择 residual/smoother operator；若是 Clover-PC coarsening，令 }A_\ell=C_\ell^{-1}\text{，否则 }A_\ell=I\\
\quad\text{Clover-PC only: }AV_\ell(x)\leftarrow C_\ell(x)^{-1}V_\ell(x);\quad\text{exchange }V_\ell\text{ 与 }AV_\ell\text{ 的 halo}\\
\quad\text{交换需要的 Gauge/link、spinor 和边界 basis；周期边界执行 coordinate wrap}\\
\quad\text{for }\mu=0,1,2,3\text{ and each fine output }x:\\
\qquad\text{forward: }UV\leftarrow U_\mu(x)V_\ell(x+\hat\mu);\quad vuv\leftarrow(AV_\ell(x))^\dagger(1-\gamma_\mu)UV\\
\qquad\text{backward storage: }UV\leftarrow U_\mu(x)A_\ell V_\ell(x+\hat\mu);\quad vuv\leftarrow V_\ell(x)^\dagger(1+\gamma_\mu)UV\\
\qquad\text{若 }x,x+\hat\mu\in\mathcal A_X:\quad X_{\ell+1}\mathrel{+}= -\kappa\,vuv\\
\qquad\text{否则：}\quad\bar Y^f_{\ell+1,\mu}\text{ 或 }\bar Y^b_{\ell+1,\mu}\mathrel{+}=vuv;\quad Y^{f/b}_{\ell+1}=-\kappa\bar Y^{f/b}_{\ell+1}\\
\quad\text{direct Clover only: }X_{\ell+1}\mathrel{+}=R_\ell C_\ell P_\ell;\quad\text{ordinary Wilson only: }X_{\ell+1}\mathrel{+}=I\\
\quad\text{Clover-PC only: }X_{\ell+1}\text{ 是单位项加块内 PC hopping，fine }C_\ell\text{ 已进入 }AV_\ell\\
\quad\text{若允许 reverse：由一个方向生成另一方向，但只适用于没有方向相关 }AV\text{ 的路径}\\
\quad\text{若本层或之前已 PC：独立保存 }Y^f\text{ 与 }Y^b\text{，并设置 bidirectional\_links=true}\\
\quad D_{\ell+1}\leftarrow X_{\ell+1}-\kappa(\bar Y^f_{\ell+1}+\bar Y^b_{\ell+1});\quad N_s^{(c)}=N_s^{(f)}/2,\quad N_c^{(c)}=N_v\\
\quad\text{若下一层 smoother 为 PC： }X_{\ell+1}^{-1}\leftarrow\operatorname{batch\_invert}(X_{\ell+1})\\
\qquad\widehat Y^f_{\ell+1}(p,X)\leftarrow X_p^{-1}(X)Y^f_{\ell+1}(p,X)\\
\qquad\widehat Y^b_{\ell+1}(q,X-\hat\mu)\leftarrow Y^b_{\ell+1}(q,X-\hat\mu)X_p^{-\dagger}(X)\\
\quad\text{创建 level }\ell+1\text{ 的 residual Dirac、smoother Dirac、sloppy field、粗 solver 与可选 deflation}\\
\text{coarse dslash at output parity }p:\\
\quad(\bar H_\ell z)_p(X)\leftarrow\sum_\mu[Y^f_\mu(p,X)z_q(X+\hat\mu)+Y^b_\mu(q,X-\hat\mu)^\dagger z_q(X-\hat\mu)]\\
\quad(D_\ell z)_p(X)\leftarrow X_p(X)z_p(X)-\kappa(\bar H_\ell z)_p(X);\quad\text{PC smoother 用 }\widehat Y\text{ 替换 }Y\\
\text{定义 }\operatorname{MG}_\ell(b_\ell):\\
\quad\text{if }\ell=L-1:\quad x_\ell\leftarrow\text{coarse direct/Krylov solve with selected solution type};\quad\text{return }x_\ell\\
\quad\text{otherwise initialize }x_\ell\text{ (zero or permitted warm start), and select }p\text{ from MATPC type}\\
\quad\text{set Transfer site subset to full-site or parity-site }p;\quad\text{do not replace physical parity by compact array index}\\
\quad\text{prepare }(out,in)\text{ according to full/asymmetric/symmetric MATPC semantics}\\
\quad\text{pre-smooth }\nu_{pre}\text{ fixed steps with MR, CG, Chebyshev, Schwarz or configured solver}\\
\quad\text{if smoother solution type differs from residual/coarse type: }r_\ell\leftarrow b_\ell-A_\ell^{res}x_\ell\\
\quad\text{else use the smoother-returned residual only when its operator and solution type match}\\
\quad b_{\ell+1}\leftarrow R_\ell r_\ell;\quad x_{\ell+1}\leftarrow0\text{（或按 cycle 策略清除 warm state）}\\
\quad e_{\ell+1}\leftarrow\operatorname{MG}_{\ell+1}(b_{\ell+1});\quad x_\ell\leftarrow x_\ell+P_\ell e_{\ell+1}\\
\quad\text{post-smooth }\nu_{post}\text{ fixed steps};\quad\text{reconstruct eliminated parity if outer solution is full}\\
\quad\text{return }M_{\ell}^{-1}b_\ell\equiv x_\ell\\
\text{right-preconditioned outer FGMRES/GCR}:\\
\quad r_0\leftarrow b-Ax_0;\quad \beta\leftarrow\|r_0\|;\quad v_0\leftarrow r_0/\beta\\
\quad\text{for restart block }j=0,1,\ldots,m-1:\\
\qquad z_j\leftarrow M_{MG,j}^{-1}v_j;\quad w\leftarrow A z_j\\
\qquad h_{ij}\leftarrow\langle v_i,w\rangle;\quad w\leftarrow w-h_{ij}v_i\quad(i=0,\ldots,j)\\
\qquad h_{j+1,j}\leftarrow\|w\|;\quad v_{j+1}\leftarrow w/h_{j+1,j};\quad\text{apply Givens and test residual estimate}\\
\qquad\text{若收敛：解小型上三角系统 }Hy=g;\quad x\leftarrow x+\sum_{j}z_jy_j;\quad\text{break}\\
\qquad\text{若到 restart：同样更新 }x\text{，重新计算 }r=b-Ax\text{，再开下一块}\\
\quad\text{若使用 CA-GCR：以 residual powers 组成 block，并对 }Z\text{ 与 }AZ\text{ 做两次 MGS}\\
\quad\text{若使用 BiCGStab/BiCGStabL：改用 shadow residual 与 }\rho,\alpha,\omega\text{ 递推，并处理 breakdown guard}\\
\quad\text{最终独立应用 full Clover }D_f\text{，报告 }\|b-D_fx\|/\|b\|\text{ 与 parity reconstruction 误差}\\
\end{tabular}
\end{table}
```

该表明确了三个容易混淆的“输入/输出”：`P` 的输入是 coarse field、输出是
fine field；`R` 的输入是 fine residual、输出是 coarse RHS；`Yhat` 是 PC
coarse dslash 的方向相关 storage，不是另一个独立的物理 Gauge。外层 Krylov
迭代次数、一次外层迭代中调用多少次 MG、每个粗层的迭代次数和每次 host/global
reduction 都应分别记录；只报总时间无法判断收益来自算法还是计时口径。

## 8. 本次正式大格对照

协议为 $16\times32\times32\times48$、c64、mass $=0.05$、seed $=42$、
$N_v=12$、aggregate block $=(2,2,2,2)$、coarse spin $=2$、odd--odd
Schur；2 次 warmup 后取 5 次 steady solve。逐次残差原始数据位于：

- [正式 benchmark JSON](../../../data/strict_vs_quda_formal_20260902.json)
- [逐迭代 trace JSON](../../../data/strict_trace_20260902_final.json)
- [逐迭代对比图](../../../data/strict_trace_20260902_final.svg)

| 实现 | 5 次 steady 外层迭代 | median solve-only | 平均每外层迭代 | full-op 真残差 |
|---|---:|---:|---:|---:|
| PyQCU Strict | 11, 11, 11, 11, 11 | 2.383584 s | 216.689 ms | $3.6013\times10^{-7}$ |
| QUDA | 37, 37, 37, 37, 37 | 2.411687 s | 65.181 ms | $7.3030\times10^{-7}$ |

PyQCU 的总 solve-only 时间约低 1.18%，主要来自更少的外层迭代；单次外层
迭代成本高于 QUDA。trace 中 PyQCU 使用 Arnoldi estimate，QUDA 使用 GCR
iterated residual，二者是不同内部标量；最终正确性以 full Clover operator
重新计算的真残差为准。

## 9. 源码证据与边界

主要对照源码包括：

```text
refer/git-rep/quda/lib/dirac_clover.cpp
refer/git-rep/quda/lib/transfer.cpp
refer/git-rep/quda/lib/coarse_op.cuh
refer/git-rep/quda/include/kernels/coarse_op_kernel.cuh
refer/git-rep/quda/lib/coarse_op_preconditioned.in.cu
refer/git-rep/quda/include/kernels/coarse_op_preconditioned.cuh
refer/git-rep/quda/include/kernels/dslash_wilson*.cuh
refer/git-rep/quda/include/kernels/dslash_wilson_clover.cuh
refer/git-rep/quda/include/kernels/prolongator.cuh
refer/git-rep/quda/include/kernels/restrictor.cuh
refer/git-rep/quda/include/kernels/dslash_coarse.cuh
refer/git-rep/quda/lib/dirac_coarse.cpp
refer/git-rep/quda/lib/multigrid.cpp
refer/git-rep/quda/lib/solver.cpp
cpp/cuda/qcu/include/lattice_clover_multigrid.h
cpp/cuda/qcu/src/multigrid.cu
```

本说明严格区分数学结构与具体 storage。公式确认的是算子来源、矩阵乘法顺序
和奇偶关系；若要声称逐元素 bitwise 等价，还必须在非平凡 Gauge/Clover 上
逐方向、逐 storage site、逐 parity 比较 $X,Y,Yhat$ 及 halo 边界。当前正式
大格 solve 与组件级回归已通过，但上述逐元素 storage 证明仍是独立验证目标。

## 10. 算法间比较：不能把所有“粗化”看成同一个算法

前九节回答“每个对象是什么、如何从源码得到”。本节回答“为什么存在多条
实现路径，以及如何选择”。比较时必须同时看四个量：

$$
\text{总成本}=
\text{外层迭代次数}\times
(\text{fine operator}+\text{预条件器}+\text{全局规约}).
$$

单独减少外层迭代，或者单独缩短一次 matvec，都不能推出总求解更快。特别是
Clover-PC 会以额外的局部矩阵逆和方向相关 link storage 换取更好的谱性质；
Schur 会以减半的未知量换取一次应用中的两次 hopping 与 Clover inverse。

### 10.1 full、asymmetric Schur、symmetric Schur 与 coarse PC

写出 block matrix

$$
D=
\begin{pmatrix}
C_p & -\kappa H_{pq}\\
-\kappa H_{qp} & C_q
\end{pmatrix},
\qquad
D_{pq}:\mathcal V_q\rightarrow\mathcal V_p .
$$

消去 $q$ 后，四种常见对象的代数关系如下：

| 路径 | 实际求解对象 | 主要优点 | 主要代价/风险 | 适合的验证 |
|---|---|---|---|---|
| full Clover | $D$，未知量为 $e+o$ | 语义最直接；full residual 可直接计算；不需要 parity reconstruct | 未知量和存储约为 Schur 的 2 倍；Krylov matvec 承担完整的局部/邻居工作 | 直接检查 $\|b-Dx\|/\|b\|$ |
| asymmetric Schur | $S_p=C_p-\kappa^2H_{pq}C_q^{-1}H_{qp}$ | 只保存一个 parity；保留原始 Clover block，和 full operator 的消元关系清楚 | 每次 $S_p$ 至少包含两次 hopping 与一次 $C_q^{-1}$；一般非 Hermitian；需要 prepare/reconstruct | 同时检查 Schur residual 与 reconstruct 后 full residual |
| symmetric Schur | $S_p^{sym}=I-\kappa^2C_p^{-1}H_{pq}C_q^{-1}H_{qp}$ | onsite 归一化；形式上更适合 $\gamma_5$-Hermitian 处理和某些预条件器 | 额外一次目标 parity 的 $C_p^{-1}$；名称中的 symmetric 不等于普通 Hermitian；错误的左右顺序会改变算子 | 检查 $S^\dagger$ 与 $\gamma_5S\gamma_5$，不能只检查迭代下降 |
| coarse PC | $X_p^{-1}D_{pq}$ 组成的粗 Schur | 粗层把大块 onsite 消去；可显著改善粗层尺度差异 | 需要 batch inverse；$\widehat Y^f$ 左乘、$\widehat Y^b$ 右乘 $X^{-\dagger}$；方向 storage 更难核对 | 逐方向比较 $X^{-1}$、$Yhat^f$、$Yhat^b$ 及 dagger |

fine asymmetric 的右端和恢复公式为

$$
\begin{aligned}
b_p^{S}&=b_p+\kappa H_{pq}C_q^{-1}b_q,\\
x_q&=C_q^{-1}\left(b_q+\kappa H_{qp}x_p\right).
\end{aligned}
$$

若将两侧都归一化，则

$$
b_p^{S,sym}=C_p^{-1}b_p+
\kappa C_p^{-1}H_{pq}C_q^{-1}b_q,
\qquad
x_q=C_q^{-1}\left(b_q+\kappa H_{qp}x_p\right).
$$

这里的符号来自 block elimination，而不是某个 kernel 名称的猜测。QUDA
DiracCloverPC 的 prepare/reconstruct 调用顺序在
refer/git-rep/quda/lib/dirac_clover.cpp:223-261；其 asymmetric/symmetric
Dslash 组合在 :173-209。[确证]

粗层完全复用这个代数结构，但把 $C$ 换成 block matrix $X$：

$$
\begin{aligned}
D_c&=X-\kappa H_c,\\
S_{p,c}^{asym}&=X_p-\kappa^2H_{pq}X_q^{-1}H_{qp},\\
S_{p,c}^{sym}&=I-\kappa^2X_p^{-1}H_{pq}X_q^{-1}H_{qp}.
\end{aligned}
$$

因此，Yhat 不是“对粗 dslash 最终结果再乘一个逆”：

$$
\widehat Y^f(p,X)=X_p^{-1}Y^f(p,X),
\qquad
\widehat Y^b(q,X-\hat\mu)=Y^b(q,X-\hat\mu)X_p^{-\dagger}.
$$

对 backward storage 做 dagger 后才得到

$$
\widehat Y^b(q,X-\hat\mu)^\dagger
=X_p^{-1}Y^b(q,X-\hat\mu)^\dagger.
$$

这解释了一个很实用的诊断：若把 backward 也写成
$X_p^{-1}Y^b$，粗层在某些平凡输入上可能仍然收敛，但非平凡 Clover
上得到的并不是同一个 Galerkin/Schur 算子。[推断]

### 10.2 transfer 与 coarse representation 的比较

#### (a) aggregate basis、继承 null vector 与独立生成

aggregate transfer 的核心是局部投影：

$$
\begin{aligned}
P_\ell&=\bigoplus_X V_{\ell,X},\\
R_\ell&=P_\ell^\dagger,\\
D_{\ell+1}&=R_\ell D_\ell P_\ell.
\end{aligned}
$$

下一层的 null basis 有两种常见来源：

| basis 来源 | 公式/流程 | 优势 | 劣势 |
|---|---|---|---|
| 独立生成/迭代平滑 | $B_{\ell+1}$ 由本层粗算子反复作用得到 | 可针对新层谱重新优化；通常 coarse quality 更可控 | setup 需要更多 matvec、规约和显存；参数更敏感 |
| 继承限制 | $B_{\ell+1}=R_\ell B_\ell$，再做 block orthogonalization | 快；可复用已有 null space；容易缓存 | 限制操作可能丢掉新层低模；层数深时基质量可能退化 |
| 未正交或全局正交 | 不满足或改变 aggregate-local Gram 结构 | 代码可能简单，或者全局条件数更易分析 | 未正交会使 $R=P^\dagger$ 的投影质量变差；全局正交通常破坏局部性、增加通信 |

QUDA transfer.cpp:117-162 的 aggregate 路径进行 block
orthogonalization；这保证的是每个 aggregate/chiral block 内的正交，不是
粗格所有 site 之间的全局正交。[确证]

#### (b) full-site、parity-site 与 compact index

这三个量必须分开：

$$
\text{physical parity }p(x)
\ne
\text{Transfer 的 site subset}
\ne
\text{compact array 的线性 index}.
$$

| 场景 | Transfer subset | 算子中的邻居 | 常见错误 |
|---|---|---|---|
| full solve | full-site | full field 的 $e\leftrightarrow o$ | 将 full correction 错放到单 parity |
| odd--odd MATPC | parity-site，$p=o$ | Schur 内部仍是 $o\to e\to o$ | 把 compact odd index 当成物理 even/odd |
| even--even MATPC | parity-site，$p=e$ | Schur 内部仍是 $e\to o\to e$ | 只交换输出数组，未交换 prepare/reconstruct |
| coarse full field | 通常保留完整 coarse geometry | coarse neighbor 仍由 $q=1-p$ 读取 | 因为输入只存一个 parity 就删除 coarse geometry 的另一侧坐标 |

QUDA 在 multigrid.cpp:1135-1144 根据 matpc_type 设置 parity，并单独设置
site_subset；transfer.cpp:259-328 再按 subset 应用 $P/R$。[确证]

#### (c) 单向 reverse、双向 link 与 33-stencil

直接 Wilson/Clover 粗化可以在某些条件下只构造一个方向，再利用

$$
\gamma_\mu(1+\gamma_\mu)\gamma_\mu=1+\gamma_\mu,
\qquad
\gamma_\mu(1-\gamma_\mu)\gamma_\mu=1-\gamma_\mu
$$

的 spin 结构生成 reverse link；实现上 reverse 会保持 coarse-spin
diagonal block，而给 off-diagonal block 改变符号。它不是一般矩阵的
conjugate-transpose。

| storage 策略 | 数学对象 | 优势 | 劣势/边界 |
|---|---|---|---|
| 单向 + reverse | 一次计算，另一方向由 projector spin 结构恢复 | setup 约少一半方向计算和临时 storage | 只适用于没有方向相关 $AV$、PC 左右乘的路径；错误使用会破坏 Galerkin |
| bidirectional | 独立生成 $Y^f$、$Y^b$ | 正确覆盖 Clover-PC、COARSE-PC 和历史层已 PC 的情况 | setup、link memory、halo 通信增加 |
| QUDA coarse 8-slot | onsite $X$ 加 4 backward/4 forward slot | 邻居 stencil 简洁；dslash kernel 易复用 | fine Schur 的两跳结构由算子组合体现，单次 coarse dslash 不等于 wide Schur |
| PyQCU wide 33-tensor | onsite、nearest 和 diagonal path 的 $P^\dagger S P$ | 将 fine Schur 的两跳耦合显式存为单次 coarse matvec；可减少递归 kernel 组合 | 矩阵 block 更多、显存和 build 成本更高；索引/周期边界更易出错 |

QUDA 的 need_bidirectional 判定见
refer/git-rep/quda/lib/coarse_op.cuh:977-1059,1375-1433；其
dslash_coarse.cuh:127-322 读取 8-slot coarse field。PyQCU 的普通
nearest-neighbor 和 Schur-consistent 33-stencil 分别位于
cpp/cuda/qcu/src/multigrid.cu:768-856 与 :858-985。[确证]

### 10.3 V、W、F、K cycle 的代价与适用条件

令 $C_\ell$ 表示“在 $\ell$ 层执行一次粗校正并返回父层”的算子。理想递归
cycle 的差异在子问题调用次数：

$$
\begin{aligned}
C_\ell^{V}&=S_{post}\,P_\ell C_{\ell+1}^{V}R_\ell S_{pre},\\
C_\ell^{W}&=S_{post}\,P_\ell C_{\ell+1}^{W}R_\ell
                 P_\ell C_{\ell+1}^{W}R_\ell S_{pre},\\
C_\ell^{F}&=S_{post}\,P_\ell C_{\ell+1}^{F}R_\ell
                 P_\ell C_{\ell+1}^{V}R_\ell S_{pre}.
\end{aligned}
$$

上式只表示递归拓扑；实现仍可以在两次 correction 之间重新计算父层 residual、
清理 coarse Krylov 状态或使用 warm start。

| cycle | 一层的粗校正模式 | 收敛/性能优势 | 代价与风险 | 本次源码状态 |
|---|---|---|---|---|
| V | 每个非粗层一次 child solve | 成本最低；最适合稳态生产求解和先做 baseline | 对难处理低模可能收敛慢；coarse quality 不足时外层迭代多 | QUDA VCYCLE；PyQCU 默认 V |
| W | 同一层两次 child solve | 更强地消除低频误差；对弱 coarse space 更有韧性 | 粗层 work 近似翻倍；通信、规约和显存访问也增加 | PyQCU 实际两次 coarse_correction；本 QUDA 快照的 solver-wrapper 分支不接收 W |
| F | 一次 F child 加一次 V child | setup/早期 cycle 中比 V 更积极，通常比 W 便宜 | work 随深度和调用上下文变化；实现语义容易和 recursive 混淆 | PyQCU 实际 F 后接 V；本 QUDA 快照枚举有 F，但 createCoarseSolver 对非 V/recursive 进入 unsupported |
| K | child solve 内再做短 Krylov，K-cycle 作为其变预条件器 | 对非正规 coarse operator 更灵活；能把 coarse Krylov 信息纳入 cycle | 小型 Arnoldi/Gram 系统、额外 basis、dot 和同步；breakdown 需 fallback | PyQCU m=2 右预条件 FGMRES；QUDA 当前 MultiGrid 路径不能据此宣称已支持 K |
| recursive | 父层 solver 显式包住 child MG | 允许 coarse solver/deflation/inner MG 递归组合 | 配置、生命周期和精度边界复杂 | QUDA 当前快照真正支持的递归粗 solver 路径 |

QUDA 的 enum 仍列出 V/F/W/recursive，但当前快照
lib/multigrid.cpp:538-558,564-697 创建 coarse solver 时只处理
QUDA_MG_CYCLE_VCYCLE 和 QUDA_MG_CYCLE_RECURSIVE（或最靠近底层的特殊
分支）；不能把 enum 存在误写成 F/W 已在该路径可运行。[确证]

PyQCU 的 cycle bits 在 lattice_clover_multigrid.h:2575-2613 解析且互斥；
coarse_correction、k_cycle_correction 和 v_cycle 在
:3345-3800 实现。其当前具体语义还有两个重要细节：

1. 粗层非最粗层先固定步数预平滑，再做所选 correction，最后固定步数后平滑；
   warm start 会复用 coarse solution，但 W/F/K 会清除 child correction 状态。
2. level 0 的 apply_mg_prec 版本执行 fine pre-smooth、$R$、coarse solve、
   $P$，v1 省略 fine post-smooth，让外层 FGMRES 处理剩余高频误差。这和教科书
   “对称 pre/post”是不同的工程取舍，不能只看函数名判断。[确证]

### 10.4 MR、CG、Chebyshev、Schwarz 与 CA-GCR

设当前层残差为 $r=b-Ax$。平滑器不是为了精确求解，而是用少量固定步骤压制
高频误差；因此应比较“每步 matvec/规约/同步”和“对算子谱性质的假设”。

#### MR：最少状态、适配非 Hermitian

一次 MR 更新是

$$
\begin{aligned}
v&=Ar,\\
\alpha&=\frac{\langle v,r\rangle}{\langle v,v\rangle},\\
x&\leftarrow x+\alpha r,\qquad r\leftarrow r-\alpha v.
\end{aligned}
$$

$\langle v,v\rangle$ 为实非负量，$\alpha$ 可以是复数。MR 不要求 $A$ Hermitian，
故适合粗层仅 $\gamma_5$-Hermitian 或明显非正规的情形；代价是方向不积累，
每一步只沿当前 residual，低模消除能力通常弱于 Krylov solver。PyQCU 的
mr_iter_coarse 在 lattice_clover_multigrid.h:2137-2158 exactly 使用这
一条 recurrence，并对零/NaN denominator 做 no-op guard。[确证]

#### CG：若假设成立则高效，假设错误则会发散

$$
\begin{aligned}
\rho_k&=\langle r_k,r_k\rangle,\\
\alpha_k&=\rho_k/\langle p_k,Ap_k\rangle,\\
x_{k+1}&=x_k+\alpha_kp_k,\qquad
r_{k+1}=r_k-\alpha_kAp_k,\\
\beta_k&=\rho_{k+1}/\rho_k,\qquad
p_{k+1}=r_{k+1}+\beta_kp_k.
\end{aligned}
$$

CG 需要 $A$ 为 Hermitian positive definite（至少在实际内积和精度下满足）。
它每步一个 matvec、两个 dot，方向历史只有一个 $p$，通常比 MR 有更强的
谱过滤。但 Clover coarse Schur 若仅满足

$$
A^\dagger=\gamma_5A\gamma_5
$$

而不满足 $A^\dagger=A$，不能由“对称 Schur”这个名字直接推出可用 CG。
PyQCU 源码明确记录 coarse plain CG 曾发散，因此 coarse solver 主路径使用
guarded BiStabCG；fine Schur 的特定 smoother 配置仍可使用固定步 CG。这个
“fine smoother 可 CG、coarse solve 不一定 CG”必须分开记录。[确证]

#### Chebyshev：低同步，但谱界错误会放大误差

用一次采样估计 $\rho=\|Ar\|/\|r\|$，PyQCU 当前 coarse 版本取

$$
\lambda_{\max}=8\rho,\qquad
\lambda_{\min}=0.05\lambda_{\max},\qquad
\theta=\frac{\lambda_{\max}+\lambda_{\min}}2,\quad
\delta=\frac{\lambda_{\max}-\lambda_{\min}}2.
$$

令 $p_0=0$，每步执行

$$
\begin{aligned}
p_{k+1}&=\alpha_k r_k+\beta_k p_k,\\
x_{k+1}&=x_k+p_{k+1},\\
r_{k+1}&=r_k-Ap_{k+1},\\
\beta_{k+1}&=\left(\frac{\delta\alpha_k}{2}\right)^2,\qquad
\alpha_{k+1}=\frac1{\theta-\beta_{k+1}}.
\end{aligned}
$$

PyQCU 的 chebyshev_smooth_coarse 在
lattice_clover_multigrid.h:1814-1894 采用该固定步递推；谱界是保守启发式，
不是对非 Hermitian Schur 谱的证明，并且 kernel 对 non-finite update 直接拒绝。
优势是可把标量递推留在 host-free/device path、没有每步 convergence branch；
劣势是若谱在复平面、$\lambda_{\min}$ 估计不合适，可能变成低效甚至不稳定
的多项式。QUDA 的 CA solver 还可使用 Chebyshev basis，但那是 Krylov basis
构造方式，不应与 PyQCU 这个 smoother 混名。

#### Schwarz：局部求解换取通信边界

以重叠 block $I$ 的限制/延拓算子 $R_I,P_I$ 表示，additive Schwarz 可写成

$$
M_{\rm AS}^{-1}=\sum_I P_I A_I^{-1}R_I,
$$

而 multiplicative Schwarz 按 block 顺序更新 residual：

$$
x^{(j+1)}=x^{(j)}+
P_{I_j}A_{I_j}^{-1}R_{I_j}
\left(b-Ax^{(j)}\right).
$$

它对局部高频误差通常很强，并可用 local Clover/block solve；但 overlap、
边界 halo、block 顺序和同步决定实际成本。QUDA 的 smoother 参数把 Schwarz
类型、cycle 和 Nsteps 作为独立配置；inv_mr_quda.cpp:19,108 还对
multiplicative Schwarz 的步数奇偶有约束。[确证] 因此 Schwarz 是“平滑/预处理
外壳”，不应与 MR/CG 这种 scalar recurrence 直接当成同一层级比较。

#### CA-GCR：减少 global reduction，增加 block memory

通信规避 GCR 将 residual powers 组成 block：

$$
\mathcal R_k=[r_k,Ar_k,A^2r_k,\ldots],
\qquad
Z_k=M^{-1}\mathcal R_k,
\qquad
W_k=AZ_k,
$$

随后在 residual space 与 image space 做正交化，再解小 Gram 系统：

$$
(W_k^\dagger W_k)\alpha=W_k^\dagger r,\qquad
x\leftarrow x+Z_k\alpha.
$$

优势是把多步 dot/reduction 合并为 block 级操作，适合 MPI latency 较高的环境；
劣势是需要多个 basis/image/correction 向量，Gram 矩阵可能病态，非正规算子或
有限精度下需要降阶、重启和 fallback。PyQCU 当前 CA-GCR block size 为 4，
对 $Z$ 和 $AZ$ 做两次 MGS，block 失败后回退 validated FGMRES，见
lattice_clover_multigrid.h:5115-5360。[确证]

综合选择表：

| 平滑器 | 算子要求 | 每步主要工作 | 数值优点 | 典型缺点 |
|---|---|---|---|---|
| MR | 非 Hermitian 也可 | 1 matvec + 2 dot | 简单、状态小、不会假设正定 | 谱过滤弱，固定步数可能不够 |
| CG | Hermitian positive definite | 1 matvec + 2 dot | 方向积累，通常比 MR 强 | 对非 Hermitian/非正定会失效或发散 |
| Chebyshev | 可用的正实谱区间/近似区间 | 多项式 matvec，少分支 | 适合固定步和低同步 | 谱界敏感；非正规 coarse 需谨慎 |
| Schwarz | 局部子问题可解 | block solve + halo/overlap | 高频抑制强，局部性好 | block 边界、通信、步数约束 |
| CA-GCR | 任意但需稳定 block basis | block matvec + Gram solve | 降低 reduction 次数 | memory/conditioning/fallback 成本高 |

#### QUDA 的 pre-smoother、post-smoother 与 residual operator

QUDA 的 smoother 不是一次匿名的 A 调用，而是有生命周期和返回值语义的
对象。对每个层至少要区分

$$
A_\ell^{res}
\quad\text{与}\quad
A_\ell^{sm},
\qquad
r_\ell^{res}=b_\ell-A_\ell^{res}x_\ell .
$$

MG::createSmoother 为 pre-smoother 设置 return_residual=true，post-smoother
不需要返回 residual；而 MG::operator() 只有在 solution type 和 smoother
operator 一致时才直接复用 smoother 的 residual，否则重新调用 residual
operator 计算 $b-Ax$。对应源码为
refer/git-rep/quda/lib/multigrid.cpp:273-337,1146-1193。[确证]

| 条件 | coarse setup 使用的对象 | V-cycle 中的处理 | 优势 | 风险 |
|---|---|---|---|---|
| MAT solution + DIRECT smoother | residual 与 smoother 可保持同一 solution type | 可复用 smoother residual，少一次 matvec | 简单、快 | 仍需确认 sloppy precision/Schwarz 不改变语义 |
| MATPC solution + DIRECT-PC smoother | coarsen preconditioned smoother，构造 DiracCoarsePC | parity prepare/reconstruct 与 coarse correction 对齐 | 粗层可直接在 PC 空间工作 | 不能混入普通 DiracCoarse；要求 PC smoother |
| solution/operator 不一致 | residual Dirac 与 smoother Dirac 分开 | 先 reconstruct，再显式算 $r=b-A^{res}x$ | 语义正确，适合不同精度或不同 PC | 多一次 operator apply；性能不能和复用 residual 的路径直接比 |
| post-smoother | 只负责校正后的局部高频误差 | 不返回 coarse RHS residual | 可独立调 $\nu_{post}$ | 过强 post-smooth 会把 cycle 成本推高 |

preconditioned_coarsen 的判定、DiracCoarse residual 与
DiracCoarsePC smoother 的创建见
multigrid.cpp:342-429。这也是算法比较中必须单独列出 residual/smoother
轴的原因：即使两者都叫 coarse dslash，实际被投影的输入、是否含 $X^{-1}$、
以及是否对目标 parity 做 reconstruct 都可能不同。[确证]

PyQCU 当前的 apply_mg_prec 也保留这一原则：先做 fine pre-smooth，再以
selected fine operator 形成 residual，之后 $R$、coarse solve、$P$；level 0
v1 省略 post-smooth。它在
cpp/cuda/qcu/include/lattice_clover_multigrid.h:4924-5020 的实现路径与
QUDA 对应，但不是 QUDA 两个 Dirac object 的逐字段复制。[推断]

### 10.5 外层 CG、BiCGStab、BiCGStabL、GCR/FGMRES 与 CA-GCR

外层 solver 的选择由“真正被求解的 operator”决定，而不是由 fine field 是否
叫 Clover 决定。右预条件 FGMRES/GCR 的基本关系为

$$
\begin{aligned}
z_j&=M_j^{-1}v_j,\qquad w_j=A z_j,\\
h_{ij}&=\langle v_i,w_j\rangle,\qquad
w_j\leftarrow w_j-\sum_i h_{ij}v_i,\\
h_{j+1,j}&=\|w_j\|,\qquad v_{j+1}=w_j/h_{j+1,j}.
\end{aligned}
$$

预条件器 $M_j^{-1}$ 可以随 $j$ 变化；这正是固定右预条件 CG 不能直接替代
FGMRES 的原因。

| 外层 solver | 适用算子 | 主要优势 | 主要劣势 | QUDA/PyQCU 对照 |
|---|---|---|---|---|
| CG/PCG | Hermitian positive definite | 单向量历史、每步一个 matvec；PCG 理论清楚 | 对 $\gamma_5$-Hermitian 但非 Hermitian 的 Schur 不适用；PC 需保持对称性 | QUDA solver factory 允许 PCG 显式 MG，但由 mat.hermitian() 检查；PyQCU 不把当前通用 Schur 外层宣称为 CG |
| BiCGStab | 非 Hermitian | 低存储；通常每轮约两次 operator apply | shadow residual、$\rho/\omega$ breakdown；残差可能非单调 | QUDA 提供；PyQCU fine/coarse 有 guarded BiStabCG recurrence |
| BiCGStabL | 非 Hermitian、希望更高阶稳定多项式 | 每个 outer cycle 可处理更长 residual polynomial | 单轮 matvec/状态更多；breakdown 与参数更复杂 | QUDA factory 提供；PyQCU 与 FGMRES/CA-GCR mode 互斥 |
| GCR | 非 Hermitian、可变预条件 | 正交化 image，适配 MG 变预条件 | basis storage 与 dot 随 restart 增长 | QUDA 的显式 MG preconditioner 主要接 GCR/PCG；源码 solver.cpp:47-113 |
| FGMRES | 非 Hermitian、右预条件可变 | 对 variable MG 最稳健；可容纳 cycle、精度变化和 guard | 每步保存 $V/Z$，Arnoldi dot/host work 大 | PyQCU MG_USE_GCR 路径实际为 restarted FGMRES，默认 $m=10$ |
| CA-GCR | 非 Hermitian、通信受限 | block 化 global reduction | Gram conditioning、额外 workspace、失败回退 | PyQCU block=4；QUDA solver factory 提供 CA-GCR |
| GMRES-DR/deflation | 有稳定低模/可转移 deflation | 反复 RHS 时低模加速明显 | setup/eigenvector memory 和维护成本高 | QUDA coarse solver 可选 deflation；不是本次 PyQCU formal 主路径 |

外层每步都必须明确：

$$
\text{outer step}
=
\underbrace{\text{one }A_0\text{ apply}}_{\text{fine/full or Schur}}
+
\underbrace{\text{one }M_{\rm MG}^{-1}\text{ apply}}_{\text{pre-smooth}+R+\text{coarse}+P}
+
\underbrace{\text{Krylov orthogonalization/reduction}}_{\text{solver dependent}}.
$$

因此“BiCGStab 迭代数更少”不自动优于“FGMRES 迭代数更多”；前者可能每步
更便宜，也可能因 residual non-monotonic、breakdown guard 或重算真残差付出
额外代价。formal 对照中 PyQCU Strict 为 11 个外层步骤、QUDA GCR 为 37 个，
但单步平均时间分别是 216.689 ms 与 65.181 ms，正好展示这个区别。

### 10.6 面向 Clover MultiGrid 的选择矩阵

| 目标/约束 | 首选路径 | 为什么 | 需要同时监控 |
|---|---|---|---|
| 先证明数学正确性 | full Clover + independent full residual | 不把 Schur prepare/reconstruct 和 coarse parity 混在一起 | full-op residual、Clover basis、Gauge dagger |
| 显存紧张、奇偶结构稳定 | asymmetric Schur | 未知量减半，代数消元关系直接 | 两次 hopping、被消 parity 的 Clover inverse、reconstruct |
| 想让 coarse onsite 归一化 | symmetric Schur 或 coarse PC | 粗层 block scale 更均匀 | 是否真的满足要求的 Hermitian/相似 Hermitian 条件 |
| coarse 仅 $\gamma_5$-Hermitian | MR / guarded BiCGStab / FGMRES | 不错误假设 HPD | $\rho,\omega$ breakdown、NaN guard、真 residual |
| global reduction 是瓶颈 | fixed-step Chebyshev、CA-GCR、融合 dot | 减少每步 host/MPI round trip | 谱界、Gram condition、fallback、数值漂移 |
| coarse space 质量不足 | W/F 或重新生成 null vectors | 增强低频 correction 或改善 basis | coarse work 是否超过 outer iteration 节省 |
| MG 每步成本很高 | V + 适量 fixed smoother + warm cache | 控制 work，保留主要低频校正 | setup/solve 分开计时，$N_{outer}$ 与每步时间 |

最终选择应服从以下优先级：

$$
\text{算子等价与 residual 正确}
\;\succ\;
\text{稳定收敛}
\;\succ\;
\text{端到端总成本}
\;\succ\;
\text{solve-only 成本}
\;\succ\;
\text{单步微观成本}
\quad\text{（在前者已通过 gate 后）}.
$$

这里最后一项不是说单步性能不重要，而是提醒不能用放宽 tolerance、跳过
reconstruct、缩短 coarse solve 或混淆 trace scalar 的方式制造“加速”。

### 10.7 setup、steady solve 与同步的代价模型

把 setup 与反复 RHS 的 steady solve 分开：

$$
\begin{aligned}
T_{\rm end\text{-}to\text{-}end}
&=T_{\rm input}+T_{\rm setup}+N_{\rm rhs}T_{\rm solve},\\
T_{\rm setup}
&=T_{\rm null}+T_{\rm orth}
 +T_{\rm transfer}+T_{RDP}+T_{X^{-1}}+T_{\rm halo},\\
T_{\rm solve}
&=N_{\rm outer}\left(T_A+T_{\rm MG}+T_{\rm Krylov}\right).
\end{aligned}
$$

在一个 MG apply 内可以进一步写成

$$
T_{\rm MG}\simeq
\nu_{\rm pre}T_s+T_R+
N_{\rm child}T_{\rm coarse}+
T_P+\nu_{\rm post}T_s,
$$

$$
T_{\rm coarse}\simeq
N_{\rm coarse}\left(T_{A_c}+T_{\rm dot}+T_{\rm halo}\right)
 +T_{\rm restart} .
$$

其中 $N_{\rm child}=1$（V）、约为 2（W/F 的当前实现）或由 K-cycle 的短
FGMRES 及递归 child 共同决定。GPU 计算量还要乘以每层的 block dimension：

$$
\operatorname{bytes}(Y_\ell)
\sim
8\,V_\ell^{site}\,(N_s^{(\ell)}N_c^{(\ell)})^2
\,\operatorname{sizeof}(\text{complex}),
$$

直接粗层 full-site 的 8 个 direction slot 因而与 coarse dof 的平方成正比；
33-stencil 还要加入 diagonal path。另一方面，CG/BiCGStab 的 dot 是小标量，
但在 MPI 或 WSL2 中一次 host/device/global synchronization 可能比一个 coarse
dslash 更贵，所以应报告

$$
N_{\rm matvec},\quad N_{\rm dot},\quad N_{\rm host\ sync},\quad
N_{\rm halo},\quad N_{\rm outer},\quad
T_{\rm setup},\quad T_{\rm solve}.
$$

本次 formal 数据采用 steady solve-only 口径；trace 只为得到残差轨迹，不把
trace 的日志开销混入性能。该口径下 PyQCU 总时间约快 1.18%，但单 outer
step 约为 QUDA 的 3.32 倍。这个结果只能说明“迭代次数和单步成本的乘积在该
固定协议下接近抵消”，不能推导出所有格点、GPU、cache 或 setup 口径下的优势。

## 11. QUDA 与 PyQCU 当前实现的逐轴映射

下表把“数学上相同”与“kernel/storage 相同”分开。相同的 Galerkin 符号并不
意味着可以直接复用一侧的数组布局。

| 算法轴 | QUDA 当前快照 | PyQCU 当前实现 | 结论 |
|---|---|---|---|
| fine operator | DiracClover::M：$C-\kappa H$；Clover 在 chiral basis 的两个 $6\times6$ block | C++ fine Clover/Gauge 与 odd--odd Schur | 数学对象对应；归一化需按 $\kappa$/mass 协议对齐 |
| parity | Transfer 由 MATPC 设置 full/parity subset；coarse dslash 邻居为 $1-p$ | fine Schur 用 odd/even packed layout；outer 结束 recover $e$ | 都不能把 compact index 当 physical parity |
| transfer | aggregate $V$、$P$、$R=P^\dagger$，coarse spin map | packed transfer asset 与 restrict/prolong | 局部基结构对应；逐元素 layout 仍需锚定 |
| direct coarse | $X$ 加 8-direction coarse links | 普通 coarse nearest-neighbor dslash | 方向拓扑对应 |
| PC coarse | Xinv、方向相关 Yhat，向下粗化 Yhat | strict packed $X/Y/Yhat$ 语义；不默认驻留 raw $Y$ | 公式可对应，storage block 尚未逐项证明 |
| fine Schur coarse | coarse dslash 组合表达两跳 | multigrid_coarse_dslash_wide 显式 33-tensor $P^\dagger S P$ | 算子表示不同；不能只比较某一次 dslash 的输出 |
| cycle | 当前 QUDA 路径确证 V/recursive；enum 的 F/W 不代表 wrapper 可用 | V/W/F/K bit 互斥并有实际分支 | PyQCU 支持面更宽，但要分别验证 work 与 residual |
| smoother | solver factory 可用 CG、MR、CA-CG、CA-GCR 等；Schwarz 作为 preconditioner | fixed CG/MR/Chebyshev；coarse guarded BiStabCG；outer FGMRES/CA-GCR/BiCGStabL | 名称相同不一定是同一层级的 recurrence |
| outer MG | 显式 MG preconditioner 受 GCR/PCG 约束 | right-preconditioned FGMRES，支持 variable MG | 非 Hermitian/variable PC 时 FGMRES 更自然 |
| precision | residual/smoother/sloppy field 可独立配置 | mixed path 以 level_data_type dispatch | 需要分别核对 dtype 与 halo，不可用 c64/c128 void pointer 互换 |
| communication | fine/coarse field halo、link ghost、MPI global reduction | Strict 对非单 rank fail-closed；单 rank 多卡是独立完整问题并发 | 多卡吞吐测试不是域分解 strong scaling |

对本表最后一列的边界要特别强调：当前 formal solve 的 full residual 通过，只
能确认端到端解在给定协议下正确；不能替代以下组件级证明：

$$
\forall(\ell,p,X,\mu):
\quad
X_\ell^{PyQCU}=X_\ell^{QUDA},\quad
Y_{\ell,\mu}^{PyQCU}=Y_{\ell,\mu}^{QUDA},\quad
\widehat Y_{\ell,\mu}^{PyQCU}=\widehat Y_{\ell,\mu}^{QUDA}.
$$

右侧等式还要指定 dtype、矩阵布局、storage site、direction slot、周期 wrap
和 dagger 顺序；否则“数值相同”没有足够精度的语义。

## 12. 双 P100 重复测试：正确性通过，但没有稳定双卡加速

本节记录同一工作区中新增的可复现实验，脚本为
[bench_multigpu_repeat.py](./bench_multigpu_repeat.py)，原始数据和图分别为
[multigpu_formal_20260902.json](../../../data/multigpu_formal_20260902.json)、
[multigpu_formal_20260902.svg](../../../data/multigpu_formal_20260902.svg)。

### 12.1 测试语义与计时边界

| 项目 | 设置 |
|---|---|
| lattice | $16\times32\times32\times48$ |
| physics | mass $=0.05$，atol $=10^{-6}$，seed $=42$ |
| hierarchy | 2 levels，dof $=[12,12]$，block $=(2,2,2,2)$，CG smoother，V-cycle |
| GPU | 单卡基线 P100 device 1；双卡 P100 devices 1、2 |
| 并行模型 | 一个 Python thread 绑定一张 GPU；每个 thread 解一个完整复制的问题 |
| 不是的模型 | 不是把一个 lattice 切成两块的 MPI/domain-decomposition strong scaling |
| 重复 | 3 repeats；每次 _run_one 完整建立/运行该问题，无额外计时 warmup |
| 主指标 | mg_parallel_wall_s=max(thread.mg_time)；mg_time 是 C++ MG solve-only 计时 |
| 正确性 | verify_consistency(tol=10^{-5})，并比较 MG 解与参考 BiStabCG |

因双卡 wall time 由较慢 thread 决定，吞吐比和双卡效率定义为

$$
\operatorname{ratio}=\frac{T_{\rm single,median}}{T_{\rm dual,median}},
\qquad
\eta_{2GPU}=\frac{\operatorname{ratio}}2.
$$

### 12.2 三次结果

| 配置 | repeat 1 (s) | repeat 2 (s) | repeat 3 (s) | median (s) | MAD (s) |
|---|---:|---:|---:|---:|---:|
| 单 P100（device 1） | 10.249874 | 10.460159 | 10.143614 | 10.249874 | 0.106260 |
| 双 P100（device 1+2） | 10.739143 | 8.300990 | 10.254227 | 10.254227 | 0.484917 |

于是

$$
\operatorname{ratio}
=\frac{10.249873876997299}{10.254226506000123}
=0.9995755283,
\qquad
\eta_{2GPU}=0.4997877641.
$$

双卡三次的两个 thread 都通过 consistency gate，MG 解相对参考解的最大相对
误差为 $6.7986866\times10^{-6}$，小于 $10^{-5}$。因此可以下的结论是：

1. 双卡线程隔离和数值一致性通过，3/3 repeats 均 PASS。
2. 双卡 wall time 的中位数与单卡几乎相同，ratio 约 $0.9996\times$，不是
   2 倍吞吐；两卡效率约 $49.98\%$。
3. 当前结果不能称为“双卡加速成功”。device 1 是双卡 wall time 的瓶颈，
   device 2 的短时间不能抵消 device 1 的长时间。

### 12.3 每个 thread 的不对称

| repeat | P100 device 1 MG(s) | P100 device 2 MG(s) | 双卡 wall(s) |
|---:|---:|---:|---:|
| 1 | 10.739143 | 1.515111 | 10.739143 |
| 2 | 8.300990 | 1.517761 | 8.300990 |
| 3 | 10.254227 | 1.529833 | 10.254227 |

device 2 的 1.515--1.530 s 与 device 1 的 8.301--10.739 s 存在数量级差异，
而三次 consistency 都为 PASS。这说明本次首先暴露的是线程/设备计时不对称或
运行环境抖动，而不是通过错误结果制造出的加速。仅凭这三次样本无法把原因
唯一归因于 PCIe、CPU 调度、GPU 时钟、上下文或缓存；下一步应采集每线程 CUDA
event、GPU utilization/clock、CPU affinity 和 process placement。

### 12.4 复现实验命令与证据限制

~~~text
source ./env.sh
python -B examples/qcu/dev87/bench_multigpu_repeat.py --repeats 3 --devices 1 2
~~~

脚本写出的 JSON 同时保留每个 thread 的 device、mg_s、ref_s、解差异和
consistency checks；SVG 是无依赖的柱状图，柱高取每个 repeat 的 max-thread
MG time。该实验没有修改 MultiGrid 核心算法，也没有把 P100 结果与 V100 单卡
直接混作算法 speedup。要证明真正的多 GPU 域分解加速，还需要 MPI 分区、相同
单问题和通信 halo 的另一套 protocol。[确证]

## 13. 验证清单、结论与未验证边界

### 13.1 推荐的最小验证顺序

~~~text
1. 用 identity / unit Gauge 和可解析 Clover 检查 fine full D 与两个 parity block。
2. 固定 p=o，独立检查 prepare -> Schur solve -> reconstruct -> full residual。
3. 在一个 aggregate 上手算 block Gram matrix，检查 V†V 与 R=P†。
4. 在一个跨边界 aggregate 上逐方向检查 UV、VUV、X/Y 分流。
5. 对 PC 路径分别检查 AV、Xinv、Yhat^f、Yhat^b 及 backward dagger。
6. 以相同 coarse RHS 比较 nearest coarse dslash 与 wide 33-stencil 的定义域。
7. 只在 component gate 通过后比较 V/W/F/K、smoother 和 outer solver 的时间。
8. 性能报告同时给 setup、steady solve、outer count、每步时间、halo、reduction。
~~~

### 13.2 本文能确认什么

$$
\begin{array}{ll}
\text{能确认：}&
\text{Clover/Gauge 的 fine 算子、奇偶 Schur、prepare/reconstruct 的代数顺序};\\
&
\text{aggregate }P/R\text{、coarse }X/Y\text{、PC }AV/Yhat\text{ 的源码语义};\\
&
\text{QUDA 当前快照的 V/recursive 支持边界，以及 PyQCU V/W/F/K 的调用拓扑};\\
&
\text{formal solve 的 residual gate 与双 P100 线程一致性/计时结果}.\\[2pt]
\text{不能确认：}&
\text{两个库每个 coarse storage block 的 bitwise 相等};\\
&
\text{所有 GPU、格点、precision、setup/cache 口径下的性能优劣};\\
&
\text{双 P100 设备不对称的唯一根因};\\
&
\text{把 PyQCU 33-stencil 与 QUDA 8-slot dslash 直接视为同一 storage}.
\end{array}
$$

最终应把“正确性”“收敛效率”“单步成本”“端到端成本”和“多卡吞吐”分开
报告。对于本次数据，最诚实的摘要是：

$$
\boxed{
\text{PyQCU Strict：外层迭代更少但单步更重；}
\quad
\text{双 P100：正确性通过但当前无稳定加速。}
}
$$
