# 以 Clover Dslash 为实例理解 QUDA MultiGrid 各层算子

本文是 `dev87_report.md` 第 22 节的独立 Markdown 版本，目标是把 QUDA
MultiGrid 中容易被“黑盒算子”掩盖的细节写成一条连续的数学与实现链：细层
Clover/Gauge 算子、奇偶 Schur、null vector、$P/R$ 转移、粗层
$X/Y/Yhat$、
递归粗化以及 V-cycle 中的各种预处理。

## 1. 统一记号与 QUDA 的对象层次

对第 ℓ 层定义

\[
D_\ell = X_\ell - \kappa\,\bar Y_\ell,
\qquad
D_{\ell+1}=R_\ell D_\ell P_\ell .
\]

细层的特殊之处是

\[
D_0\equiv D_f=C_f-\kappa H_f,
\]

其中：

| 符号 | 含义 | QUDA 中的实际角色 |
|---|---|---|
| $C_f$ | 每个细格点上的 12×12 Clover onsite 矩阵 | `CloverField` 及其偶/奇子格逆 |
| $H_f$ | Wilson hopping，只连接相反奇偶格点 | `dslash_wilson`/Clover dslash kernel |
| $P_\ell$ | 粗格向量提升到细格 | prolongator，由 null vectors 构成 |
| $R_\ell$ | 细格残差限制到粗格 | restrictor，数学上取 $P_\ell^\dagger$ |
| $X_\ell$ | onsite 与 aggregate 内部 hopping 的粗矩阵 | coarse diagonal/onsite block |
| $\bar Y_\ell$ | aggregate 之间的有向 hopping，尚未乘 $-\kappa$ | coarse forward/backward link-like matrices |
| $Y_\ell=-\kappa\bar Y_\ell$ | 实际进入粗算子的 hopping | coarse dslash 的方向矩阵 |
| `Yhat` | Clover-PC 后经过 $X^{-1}$ 处理的粗 hopping | 下一层递归粗化的输入 |

因此，QUDA 的粗层并不是把原始 Gauge 场简单下采样；它保存的是细层算子在
聚合子空间上的 Galerkin 投影。

## 2. 细层 Clover Dslash：先固定算子本身

忽略不同版本中的整体归一化约定，Wilson-Clover 算子可写成

\[
\begin{aligned}
(D_f\psi)(x)= {}& C_f(x)\psi(x) \\
&-\kappa\sum_{\mu=0}^{3}\left[
(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)
\right].
\end{aligned}
\]

这里 $\psi$ 同时带 spin/color，自由度为 $4\times3=12$。forward hopping
使用 $U_\mu(x)$ 与 $1-\gamma_\mu$，backward hopping 使用相邻点的反向
链 $U_\mu^\dagger(x-\hat\mu)$ 与 $1+\gamma_\mu$。投影矩阵只作用于
spin，Gauge 矩阵只作用于 color，Clover 矩阵则在一个格点内混合 spin/color。

定义格点奇偶

\[
p(x)=\left(\sum_{\mu}x_\mu\right)\bmod 2,
\qquad p(x\pm\hat\mu)=1-p(x).
\]

于是 $H_f$ 只有 off-diagonal parity block，而 Clover 项是 parity-diagonal：

\[
D_f=
\begin{pmatrix}
D_{ee} & D_{eo}\\
D_{oe} & D_{oo}
\end{pmatrix},
\qquad
D_{ee}=C_{ee},\quad D_{oo}=C_{oo}.
\]

对 odd Schur，QUDA 实际求解的对象是

\[
M_o=D_{oo}-D_{oe}D_{ee}^{-1}D_{eo},
\]

其中 $D_{ee}^{-1}$ 是逐格点 Clover inverse；even Schur 则交换 $e/o$。这也
解释了为什么 `prepare` 与 `reconstruct` 不能省略：它们负责在全格点 RHS、
Schur RHS 和全格点解之间来回消去/恢复被消掉的 parity。

### 2.1 奇偶 Schur 的伪代码

\[
\begin{array}{l}
\text{给定 }D_f=C_f-\kappa H_f,\ b=(b_e,b_o),\text{选择被保留 parity }p=o;\\
\quad b_o^{\rm Schur}=b_o-D_{oe}D_{ee}^{-1}b_e;\\
\quad \text{solve }M_o x_o=b_o^{\rm Schur};\\
\quad x_e=D_{ee}^{-1}(b_e-D_{eo}x_o);\\
\quad x=(x_e,x_o);\\
\text{若保留 }p=e,\text{则交换 }e\leftrightarrow o.\\
\end{array}
\]

`prepare` 实现前两行，Schur MultiGrid 作用于 $x_p$，`reconstruct` 实现后
两行。每次 hopping 读取的邻居仍然是 $q=1-p$，不能把 compact parity
storage 中的数组下标误当成物理 parity。

## 3. Null vectors、aggregate 与 $P/R$

把细格划分为不重叠 aggregate $B_X$，其中 $X$ 是粗格点坐标。每个
aggregate 保留 $N_v$ 个 null vectors $v_a(x)$，再压缩 spin/color 为粗
自由度 $\alpha$。更一般地写成

\[
V_{x\,s c,\,X\,\alpha},
\qquad x\in B_X,
\quad s=0,\ldots,3,
\quad c=0,\ldots,2.
\]

aggregate 内 block Gram-Schmidt 要求

\[
\sum_{x\in B_X}
V_{x,\alpha}^\dagger V_{x,\beta}
=\delta_{\alpha\beta}.
\]

若 $I_X(x)$ 是 aggregate 指示函数，则 prolongator 的作用为

\[
(P\,\phi)(x,s,c)=
\sum_{\alpha}I_X(x)V_{x,s c,\alpha}\phi(X,\alpha),
\qquad x\in B_X.
\]

对应的 restrictor 为

\[
(R\,\psi)(X,\alpha)=
\sum_{x\in B_X,s,c}V_{x,s c,\alpha}^\dagger\psi(x,s,c),
\qquad R=P^\dagger.
\]

因此，$P$ 不是复制或平均，$R$ 也不是普通 pool；二者是由 null-vector
子空间定义的带 spin/color 结构的局部基变换。奇偶 transfer 只对目标 parity
的细格点执行，但粗格完整几何仍保留，便于粗层 dslash 访问 $X\pm\hat\mu$。

## 4. 第一层粗算子：$RDP$ 每一项如何落位

Galerkin 粗化为

\[
D_1=R D_f P=R(C_f-\kappa H_f)P.
\]

按照 aggregate 之间的关系拆分：

\[
D_1=X_1+Y_1^f+Y_1^b,
\]

其中 $X_1$ 不仅是 $RC_fP$，还包含同一 aggregate 内部的 hopping：

\[
X_1(X)=R_X C_f P_X
-\kappa\,R_X H_{\rm internal}P_X.
\]

跨 aggregate 的 hopping 才存入有向矩阵 $Y^f,Y^b$：

\[
Y^f_\mu(X)= -\kappa\bar Y^f_\mu(X),
\qquad
Y^b_\mu(X)= -\kappa\bar Y^b_\mu(X).
\]

### 4.1 直接 Clover coarsening

若算子未作 Clover-PC，Clover onsite 直接出现在 Galerkin block：

\[
X_1\supset R C_f P.
\]

forward 方向的未缩放矩阵可按 QUDA kernel 的 storage 约定概括为

\[
\bar Y^f_\mu(X,Y)=
\sum_{\substack{x\in B_X\\x+\hat\mu\in B_Y}}
V(x)^\dagger(1-\gamma_\mu)U_\mu(x)V(x+\hat\mu),
\]

backward 方向为

\[
\bar Y^b_\mu(X,Y)=
\sum_{\substack{x\in B_X\\x-\hat\mu\in B_Y}}
V(x)^\dagger(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)V(x-\hat\mu).
\]

实际代码还要处理 parity、周期边界、halo、矩阵布局和 dagger；公式只描述
矩阵元素的来源。

### 4.2 Clover-PC coarsening：为什么需要 $AV=C^{-1}V$

在 Clover-preconditioned 路径中，细层 onsite inverse 已经进入算子定义。设

\[
A=C_f^{-1},
\qquad
AV=C_f^{-1}V.
\]

QUDA 的 direction-dependent construction 需要两侧不同的 $AV$ 放置：

\[
\bar Y^f_\mu
\sim (AV)^\dagger(1-\gamma_\mu)UV,
\qquad
\bar Y^b_\mu
\sim V^\dagger(1+\gamma_\mu)U(AV).
\]

这不是记号上的装饰：forward 与 backward 的左/右侧分别对应 Schur 消元后
的矩阵乘法顺序，所以 PC 情形必须独立构造两个方向，不能用非 PC 的
`reverse` 假设由一个方向生成另一个方向。

对粗 onsite block 批量求逆，记为 $X^{-1}$ 或 `Xinv`。PC 粗 hopping 的存储
对象是

\[
\widehat Y^f=X^{-1}Y^f,
\qquad
\widehat Y^b=Y^b X^{-\dagger}.
\]

可见 `Yhat` 不是对所有方向统一左乘 $X^{-1}$：backward 方向是右乘
$X^{-\dagger}$。这正是 Clover-PC 粗算子与普通 Wilson-like 粗算子的关键
差别。

## 5. 粗层 dslash、方向读取与奇偶

对粗格点 $X$ 和 storage parity $p$，相邻粗格点 parity 为 $q=1-p$。
粗层算子可写为

\[
\begin{aligned}
(D_c z)(X,p)={}&X_c(X,p)z(X,p)\\
&+\sum_\mu\left[
Y^f_\mu(X,p)z(X+\hat\mu,q)
+Y^b_\mu(X,p)z(X-\hat\mu,q)
\right].
\end{aligned}
\]

在 QUDA coarse kernel 的等价读取语义中：

\[
\begin{array}{ll}
\text{forward:}&\text{读取当前位置 }(X,p)\text{ 的 }Y(d+4,p,X)，\\
\text{backward:}&\text{读取 }(X-\hat\mu,q)\text{ 的 }Y(d,q,X-\hat\mu)^\dagger，\\
\text{neighbor parity:}&q=1-p\text{，对每个方向都成立。}
\end{array}
\]

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

\[
D_1=(X_1,Y_1^f,Y_1^b),
\qquad
D_2=R_1D_1P_1.
\]

因此后续递归只对 $X_1$、$Y_1$、必要时的 $Yhat_1$ 做 Galerkin projection。
粗层的 link-like 矩阵不再是 SU(3) Gauge 场，通常是更大的复矩阵；它只是
保留方向与邻居关系的有效 hopping。把粗矩阵误解释成原始 Gauge 会丢失：

\[
\text{aggregate 内部 hopping}\subset X,
\qquad
\text{Clover-PC 左右消元顺序}\subset Yhat.
\]

## 7. V-cycle 与外层求解器的完整逻辑

下面的伪代码把算子构造、奇偶处理、平滑、限制、粗解、提升和校正串成一个
可执行的逻辑。`A_l` 可以是普通粗算子或经过 $X_l^{-1}$ 的 PC 算子。

\[
\begin{array}{l}
\textbf{setup}(l=0):\\
\quad \text{load Gauge }U\text{ and Clover }C_f;\\
\quad \text{build }C_{ee}^{-1},C_{oo}^{-1}\text{ and choose Schur parity }p;\\
\quad \text{generate/restore null vectors }V_0;\\
\quad \text{block-orthonormalize }V_0\text{ in every aggregate};\\
\quad P_0\leftarrow V_0,\quad R_0\leftarrow P_0^\dagger;\\
\quad \text{if direct Clover: build }X_1\supset R_0C_fP_0;\\
\quad \text{if Clover-PC: compute }AV_0=C_f^{-1}V_0;\\
\quad \text{build independent }\bar Y_0^f,\bar Y_0^b;\\
\quad Y_0^{f/b}\leftarrow-\kappa\bar Y_0^{f/b};\\
\quad \text{if PC: }Yhat_0^f\leftarrow X_1^{-1}Y_0^f,\quad
Yhat_0^b\leftarrow Y_0^bX_1^{-\dagger};\\
\textbf{setup}(l>0):\\
\quad \text{treat }(X_l,Y_l^f,Y_l^b)\text{ or }(X_l,Yhat_l^f,Yhat_l^b)\text{ as input};\\
\quad \text{construct }P_l,R_l\text{ from level-}l\text{ null vectors};\\
\quad D_{l+1}\leftarrow R_lD_lP_l;\\
\quad \text{split }D_{l+1}\text{ into aggregate-internal }X_{l+1}\text{ and
cross-aggregate }Y_{l+1}^{f/b};\\
\quad \text{form }Yhat_{l+1}^{f/b}\text{ when the next level is PC};\\
\textbf{V-cycle}(l,r_l):\\
\quad \text{if }l=L:\\
\quad\quad z_l\leftarrow\text{coarse direct solve or coarse Krylov solve};\\
\quad\quad \text{return }z_l;\\
\quad \text{pre-smooth }\nu_{pre}\text{ steps with MR/Chebyshev/Schwarz/GCR};\\
\quad r_l\leftarrow b_l-D_lz_l;\\
\quad r_{l+1}\leftarrow R_lr_l;\\
\quad e_{l+1}\leftarrow\textbf{V-cycle}(l+1,r_{l+1});\\
\quad z_l\leftarrow z_l+P_le_{l+1};\\
\quad \text{post-smooth }\nu_{post}\text{ steps};\\
\quad \text{return }z_l;\\
\textbf{outer solve}:\\
\quad r^{(0)}\leftarrow b-Ax^{(0)};\\
\quad \text{for }k=1,2,\ldots:\\
\quad\quad z^{(k)}\leftarrow\text{V-cycle}(0,r^{(k-1)})\quad(\text{right preconditioner});\\
\quad\quad w^{(k)}\leftarrow Az^{(k)};\\
\quad\quad \text{update GCR/FGMRES/BiCGStab/CG basis};\\
\quad\quad x^{(k)}\leftarrow x^{(k-1)}+\text{Krylov correction};\\
\quad\quad r^{(k)}\leftarrow b-Ax^{(k)};\\
\quad\quad \text{stop when the specified relative/full-op residual is small enough}.\\
\end{array}
\]

外层 Krylov 迭代次数与单次迭代时间必须分开报告：MultiGrid 预条件器可能使
迭代次数显著下降，但 coarse operator、transfer 或同步代价仍会影响每次迭代
的墙钟时间。

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
refer/git-rep/quda/lib/coarse_op_preconditioned.in.cu
refer/git-rep/quda/include/kernels/dslash_wilson*.cuh
refer/git-rep/quda/include/kernels/prolongator.cuh
refer/git-rep/quda/include/kernels/restrictor.cuh
refer/git-rep/quda/include/kernels/dslash_coarse.cuh
refer/git-rep/quda/lib/dirac_coarse.cpp
refer/git-rep/quda/lib/multigrid.cpp
```

本说明严格区分数学结构与具体 storage。公式确认的是算子来源、矩阵乘法顺序
和奇偶关系；若要声称逐元素 bitwise 等价，还必须在非平凡 Gauge/Clover 上
逐方向、逐 storage site、逐 parity 比较 $X,Y,Yhat$ 及 halo 边界。当前正式
大格 solve 与组件级回归已通过，但上述逐元素 storage 证明仍是独立验证目标。
