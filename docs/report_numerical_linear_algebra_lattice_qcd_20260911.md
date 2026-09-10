# PyQCU 数值线性代数、格点费米子作用量与多重网格粗算子报告

> **身份**：物理 + 代码　**基线**：`stab52`　**日期**：2026-09-11　**文档版本**：重排与扩充版
>
> 本文采用 GitHub/MathJax 兼容的 LaTeX 数学语法：行内公式写作 `$...$`，独立公式写作 `$$...$$`。正文不再混用普通括号数学、完整 LaTeX 表格环境或依赖 LaTeX 编译器的表格；这样在 Markdown 预览、静态站点和本地 MathJax 中都能保持一致显示。

## 摘要

本报告把 PyQCU 当前涉及的三层对象放在同一套记号下讨论：

1. **数值线性代数**：CG、CA-CG、多移位 CG、BiCG、CGS、BiCGStab、GMRES/FGMRES、GCR、CA-GCR、MR、Chebyshev、Schwarz、SAP、Arnoldi、Lanczos 和 QR/CGS 局部正交化；
2. **格点费米子算子**：Wilson、Clover-improved Wilson，以及在 QUDA 快照中可对照的 twisted-mass、twisted-clover、staggered、asqtad/HISQ、domain-wall、Möbius 和 overlap；
3. **多重网格与粗算子**：局部零模、限制/延拓、Galerkin 投影、奇偶 Schur、QUDA `coarse_op`、PyQCU 的 legacy/compact 33-tensor 路径，以及 strict full `X/Y/Yhat` 路径。

本文的核心目标不是把所有算法都声称为 PyQCU 的生产功能，而是给出可追溯的数学定义、物理约束、实现状态、验证门槛和选择依据。凡是只有理论或 QUDA 参考实现而没有 PyQCU 入口的内容，统一标记为“参考”；凡是只做了源码核对而没有本轮设备运行的内容，统一标记为“未验证”。

## 目录

- [1. 范围、证据与显示约定](#1-范围证据与显示约定)
- [2. 格点 QCD 与 Wilson/Clover 基础](#2-格点-qcd-与-wilsonclover-基础)
- [3. 奇偶分块、Schur 补与残差语义](#3-奇偶分块schur-补与残差语义)
- [4. Krylov、投影与正交化算法](#4-krylov投影与正交化算法)
- [5. 平滑器、局部预条件与多重网格](#5-平滑器局部预条件与多重网格)
- [6. 粗空间、Galerkin 与 coarse operator](#6-粗空间galerkin-与-coarse-operator)
- [7. 其他费米子作用量的对照](#7-其他费米子作用量的对照)
- [8. PyQCU/QUDA 实现映射与 ABI 约束](#8-pyqcuquda-实现映射与-abi-约束)
- [9. 误差、性能与精度预算](#9-误差性能与精度预算)
- [10. 可复现验证矩阵](#10-可复现验证矩阵)
- [11. 算法选择速查](#11-算法选择速查)
- [12. 资料与源码索引](#12-资料与源码索引)
- [附录 A：完整流程](#附录-a从规范场到一次可审计求解的完整流程)
- [附录 B：审计公式](#附录-b残差误差与等价性的审计公式)
- [附录 C：源码矩阵](#附录-c算法物理对象与源码矩阵)
- [附录 D：基准报告模板](#附录-d基准报告模板)
- [附录 E：谱分析与验收阈值](#附录-e谱分析通信模型与验收阈值)

---

## 1. 范围、证据与显示约定

### 1.1 三种证据等级

| 标记 | 判定标准 | 文中的写法 |
|---|---|---|
| **实现** | 在 PyQCU、QCU C++ 后端或仓库内 QUDA 快照中找到可定位的入口 | 给出文件、类、函数或接口名 |
| **参考** | 有明确的理论定义或 QUDA 代码可对照，但 PyQCU 没有等价生产入口 | 明确说明不能直接调用 |
| **未验证** | 只完成源码或公式检查，没有本轮 GPU/MPI 数值运行 | 不写成性能、收敛或正确性结论 |

`refer/git-rep/quda` 是本库内的参考快照；它不能证明 PyQCU 已经支持对应 action 或 solver。源码位置优先于历史日志，性能结论必须以相同格点、精度、边界、设备和残差定义下的实测为准。

### 1.2 记号和维度

欧氏格点坐标记为

$$
 x=(x_0,x_1,x_2,x_3)=(x,y,z,t),
 \qquad \hat\mu \in \{\hat x,\hat y,\hat z,\hat t\}.
$$

偶奇性为

$$
 p(x)=(x+y+z+t)\bmod 2,
 \qquad \Lambda=\Lambda_e\cup\Lambda_o.
$$

PyQCU 的主要张量约定如下：

| 对象 | 形状 | 说明 |
|---|---|---|
| 规范场 | `[3, 3, 4, Lx, Ly, Lz, Lt]` | 色彩矩阵、四个方向、时空轴在最后 |
| 费米子场 | `[4, 3, Lx, Ly, Lz, Lt]` | 旋量 × 色彩 × `xyzt` |
| Clover 局部块 | `[4, 3, 4, 3, Lx, Ly, Lz, Lt]` | 行旋量、行色彩、列旋量、列色彩 |
| 粗场 | 依实现而定 | full 粗场保留两种 parity；compact 路径可只保留目标 parity |

内积和范数统一取复 Euclidean 内积：

$$
 \langle u,v\rangle = \sum_{x,\,s,\,c} u(x,s,c)^\dagger v(x,s,c),
 \qquad
 \|u\|_2 = \sqrt{\langle u,u\rangle}.
$$

若使用分布式格点，求和还包含 MPI rank 的局部贡献；因此局部 `dot` 与全局 `dot` 不能混用。HDF5 文件通常使用 `zyxt` 相关内部顺序，进入 PyQCU 计算前必须经过仓库已有的布局转换函数，不能把磁盘顺序当作计算顺序。

### 1.3 Markdown 中的数学规范

- 行内公式使用 `$D=I-\kappa H$`，不要写成普通文本 `(D=I-\kappa H)`；
- 多行推导使用独立 `$$...$$`，每个公式块前后留空行；
- 伪代码使用 fenced code block，算法变量在代码中用 ASCII 名称，解释性公式放在代码块外；
- 表格使用 Markdown 表格，不把 `tabular`、`caption` 等 LaTeX 环境直接放入 `.md`；
- 文件路径、函数名、参数名使用反引号；LaTeX 的反斜杠只在数学块或代码块中出现。

---

## 2. 格点 QCD 与 Wilson/Clover 基础

### 2.1 规范链接和协变差分

规范场的基本变量是链接矩阵 $U_\mu(x)\in SU(3)$，它把 $x$ 与 $x+\hat\mu$ 连接起来。正向、反向平行移动分别为

$$
 T_{+\mu}\psi(x)=U_\mu(x)\psi(x+\hat\mu),
 \qquad
 T_{-\mu}\psi(x)=U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu).
$$

在 $U_\mu(x)=I$ 的自由场极限，Fourier 模式 $\psi(x)=e^{ip\cdot x}u$ 给出熟悉的动量因子。这个极限同时是最便宜、最重要的回归测试：链接恒等、周期边界和反周期时间边界都应能独立检查。

### 2.2 Wilson 算子

取 Wilson 参数 $r=1$，未归一化的 Wilson 算子可写成

$$
 (D_W\psi)(x)
 = (m_0+4)\psi(x)
 -\frac12\sum_{\mu=0}^3
 \left[(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
 +(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right].
$$

在代码中常用 hopping 归一化

$$
 D_W = I-\kappa H,
$$

其中

$$
 (H\psi)(x)=\sum_\mu
 \left[(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
 +(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right],
 \qquad
 \kappa=\frac{1}{2m_0+8}
$$

（具体质量归一化仍需以调用入口为准）。由于 $1\pm\gamma_\mu$ 是秩为二的投影算子乘以常数，CUDA dslash 通常先做 spin projection，再做颜色矩阵乘法，以减少带宽和浮点操作。

Wilson 算子的两个基本性质是：

1. 它是最近邻 stencil，单次应用的通信只涉及各方向一层 halo；
2. 在合适的 Euclidean gamma 约定下满足 $\gamma_5$-Hermiticity：

$$
 D_W^\dagger = \gamma_5 D_W \gamma_5.
$$

第二条不是说 $D_W$ 本身 Hermitian，而是说明可构造 Hermitian 算子 $H_W=\gamma_5D_W$，也解释了为什么不同 solver 对 dagger、左右预条件和真残差的要求不同。

**仓库映射（实现）**：`pyqcu/dslash/_wilson.py`、`pyqcu/dslash/_operator.py`，以及 `cpp/cuda/qcu/include/wilson_dslash.h`、`lattice_wilson_dslash.h`。调用行为可由 `examples/qcu` 和 `examples/pyquda` 中的 Wilson dslash 测试对照。

### 2.3 Clover 改进

Clover 项用局部的色彩-旋量矩阵近似离散场强张量：

$$
 D_C = D_W + c_{\mathrm{SW}}\,\frac{i}{4}
 \sum_{\mu<\nu}\sigma_{\mu\nu}F_{\mu\nu},
 \qquad
 \sigma_{\mu\nu}=\frac12[\gamma_\mu,\gamma_\nu].
$$

把局部块记为 $A_p=I+C_p$ 后，完整算子具有

$$
 D_C=
 \begin{pmatrix}
 A_e & B_{eo}\\
 B_{oe} & A_o
 \end{pmatrix}.
$$

$C_p$ 不改变最近邻 hopping 的支持，但会把每个站点的局部逆从标量/颜色块提升为旋量-颜色块。工程上应批量求解或分解所有站点的小矩阵，避免在 Python 循环里逐点调用通用矩阵逆。

**实现状态**：PyQCU 的 `_clover.py` 和 QCU 的 Clover 头文件覆盖主路径；twisted-clover 只在 QUDA 快照中作为参考。Clover 的验证必须同时覆盖局部块 Hermiticity、$\gamma_5$-Hermiticity、Schur 重建和完整 true residual。

### 2.4 规范、边界与量纲检查

所有 action/solver 组合都应先做以下三个极限：

- **自由场极限**：$U_\mu=I$，结果与动量空间或直接 stencil 计算一致；
- **无 Clover 极限**：$c_{\mathrm{SW}}=0$，Clover 路径退化到 Wilson 路径；
- **质量极限**：增大质量时谱隙扩大、迭代通常减少；趋近临界质量时条件数恶化，不能把迭代增加误判为实现错误。

这些检查只使用无量纲格点量。若文档或日志同时出现物理质量和格点质量，必须明确 $am$、$a\mu$ 或 $\kappa$ 的归一化，不能在没有尺度因子的情况下直接比较。


### 2.5 对称性、谱与条件数

局部规范变换 $G(x)in SU(3)$ 作用为

$$
 U_mu(x)\mapsto G(x)U_mu(x)G(x+\hat\mu)^\dagger,
 \qquad
 \psi(x)\mapsto G(x)\psi(x).
$$

因此，正确的 dslash 必须满足协变性：先变换规范场和输入场再应用算子，与先应用算子再变换输出场相同。这个检查比单个随机输入的数值相等更强，因为它同时覆盖颜色矩阵左右乘法和邻居索引。

在自由场中，Wilson 算子的动量空间形式可写为

$$
 D_W(p)=m_0+\sum_{\mu}(1-\cos p_\mu)
 +i\sum_{\mu}\gamma_\mu\sin p_\mu.
$$

当 $p_\mu$ 接近零时，物理模的虚部近似线性；当多个 $p_\mu$ 接近 $\pi$ 时，Wilson 项抬高 doubler 模。这个表达式提供三个量纲和极限检查：$p=0$ 的质量项、$p\to-p$ 的 dagger 关系，以及大质量时谱隙增大。

对任意可逆算子 $A$，条件数

$$
 \kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

控制许多 Krylov 迭代的难度。对 $D^\dagger D$，其条件数是 $\kappa_2(D)^2$，所以 normal equation 虽然能使用 CG，却可能把条件数平方；这也是 MG、Schur 或 Hermitian 变换常比直接 CGNR 更有吸引力的原因。接近临界质量时 $\sigma_{\min}$ 变小，必须把物理的 critical slowing down 与代码错误分开诊断。

### 2.6 源场与物理观测量的边界

点源、墙源、体源、Z2 源和动量源只改变右端 $b$，不改变算子 $D$。比较 solver 时应固定源的归一化，否则迭代历史的绝对范数不可直接比较。对 propagator 或双线性观测量，还要记录源位置、flavor、边界条件和是否做了 gauge fixing；solver 收敛并不自动保证观测量已经完成统计或重整化。

---

## 3. 奇偶分块、Schur 补与残差语义

### 3.1 Block elimination

将 full 场按 parity 排列，写成

$$
 D=
 \begin{pmatrix}
 A_e & B_{eo}\\
 B_{oe} & A_o
 \end{pmatrix},
 \qquad
 b=\begin{pmatrix}b_e\\b_o\end{pmatrix},
 \qquad
 x=\begin{pmatrix}x_e\\x_o\end{pmatrix}.
$$

消去 odd 分量得到 even Schur：

$$
 S_e=A_e-B_{eo}A_o^{-1}B_{oe},
 \qquad
 b_e^{\,S}=b_e-B_{eo}A_o^{-1}b_o.
$$

求得 $x_e$ 后恢复

$$
 x_o=A_o^{-1}(b_o-B_{oe}x_e).
$$

若采用 $D=I-\kappa H$，则 $A_p$ 通常是 $I+C_p$，而 $B_{pq}=-\kappa H_{pq}$；因此 Schur 中的 hopping 符号和 $\kappa^2$ 因子必须由具体归一化推导，不能凭经验复制。

### 3.2 三种 Schur 语义

| 名称 | 定义 | 主要用途 | 需要注意 |
|---|---|---|---|
| full | 直接作用于 $(x_e,x_o)$ 的完整 $D$ | full Galerkin、full residual、strict hierarchy | 粗层保留两种 parity |
| asymmetric Schur | $A_e-B_{eo}A_o^{-1}B_{oe}$ | 直接 MATPC、BiCGStab、GCR、FGMRES | 一般非 Hermitian，左右乘法次序固定 |
| symmetric Schur | 先用合适的 $A_p^{-1/2}$ 或等价对称缩放 | 希望保留 Hermitian 结构的场景 | 缩放、右端和重建必须成对出现 |

“coarse-PC”是对粗层算子做 onsite 预条件后的对象，不能与 fine-level Schur 补混为同一个矩阵。对粗层 full operator $D_c$，常见表示为

$$
 \widehat D_c = X_c^{-1}D_c,
$$

其中 $X_c$ 是 coarse onsite block。若把 backward link 写为右乘形式，还可能出现

$$
 \widehat Y^f=X^{-1}Y^f,
 \qquad
 \widehat Y^b=Y^bX^{-1}
$$

或带 dagger 的变体；最终形式必须与内核读取顺序一致，不能仅凭符号相似性替换。

### 3.3 full residual 与 Schur residual

MATPC 解出的向量只是目标 parity 的压缩解。可靠停止条件应在 full 向量上计算

$$
 r_{\mathrm{true}}=b-Dx,
 \qquad
 \rho_{\mathrm{rel}}=\frac{\|r_{\mathrm{true}}\|_2}{\|b\|_2},
$$

而不是只看 Schur 递推中的 $r_S$。若右端为零，应直接返回零解并避免计算 $\|b\|^{-1}$。

对单 rank strict C++ solver，主循环采用相对于右端的判据

$$
 \|r_n\|_2^2 < \mathrm{atol}^2\,\|b\|_2^2,
$$

并在可用路径周期性刷新 true residual；多 rank 路径是否刷新取决于具体实现，不能把单 rank 行为外推到 MPI 运行。

### 3.4 Schur 伪代码

```text
输入 full b=(b_e,b_o)、局部块 A_e,A_o、hopping B_eo,B_oe
1. 计算 y_o = A_o^{-1} b_o
2. 构造 b_e^S = b_e - B_eo y_o
3. 用目标 solver 解 S_e x_e = b_e^S
4. 恢复 x_o = A_o^{-1}(b_o - B_oe x_e)
5. 用 full D 重新计算 r_true=b-Dx
6. 只有 r_true 同时满足绝对/相对停止条件才报告收敛
```

---

## 4. Krylov、投影与正交化算法

### 4.1 CG：只在 SPD 条件成立时使用

对 Hermitian positive-definite（SPD）矩阵 $A$，CG 递推为

$$
\begin{aligned}
 r_0&=b-Ax_0, & p_0&=r_0,\\
 \alpha_k&=\frac{\langle r_k,r_k\rangle}{\langle p_k,Ap_k\rangle},\\
 x_{k+1}&=x_k+\alpha_kp_k, & r_{k+1}&=r_k-\alpha_kAp_k,\\
 \beta_k&=\frac{\langle r_{k+1},r_{k+1}\rangle}{\langle r_k,r_k\rangle}, &
 p_{k+1}&=r_{k+1}+\beta_kp_k.
\end{aligned}
$$

原始 Wilson/Clover $D$ 通常不是 SPD；可对 $D^\dagger D$ 或适当 Hermitian 变换使用 CG。把裸 CG 应用于非 Hermitian $D$ 会破坏共轭方向证明，出现负或复分母时应立即报告 breakdown，而不是继续迭代。

**实现**：QCU 的 CG 入口、`pyqcu/solver/_cacg.py` 的块变体和 QUDA CG 可作为不同层次的参考。每次比较都要注明求解的是 $D$、$D^\dagger D$ 还是 Schur operator。

### 4.2 多移位 CG

若需要同时求解

$$
 (A+\sigma_j I)x^{(j)}=b,
$$

多移位 CG 利用所有移位共享的 Krylov 子空间，仅保存与移位相关的标量递推。它要求基准矩阵保持 Hermitian positive-definite，且移位是标量单位阵；Clover 的非平凡局部块或一般 Schur 不能不加证明地套用该公式。

优点是多个质量/谱移位的矩阵应用近似共享；限制是所有系统共享同一个预条件器，且最小残差与可靠停止必须分别检查。PyQCU 的 `_multishift_cg.py` 是 Python 参考实现。

### 4.3 BiCG、CGS 与 BiCGStab

BiCG 同时构造 $A$ 和 $A^\dagger$ 的左右 Krylov 序列，基本标量为

$$
 \alpha_k=\frac{\langle \tilde r_k,r_k\rangle}
 {\langle \tilde p_k,Ap_k\rangle}.
$$

它的内存占用低，但需要 dagger matvec，并且容易因影子残差正交性丢失而 breakdown。CGS 通过平方 residual polynomial 去掉显式 shadow 系列，减少算子调用却放大非正规矩阵上的不稳定性。

BiCGStab 用一阶最小残差平滑替代 CGS 的剧烈振荡。常用递推写成

$$
\begin{aligned}
 \rho_k&=\langle \tilde r_0,r_k\rangle,\\
 \beta_k&=\frac{\rho_k}{\rho_{k-1}}
 \frac{\alpha_{k-1}}{\omega_{k-1}},\\
 p_k&=r_k+\beta_k(p_{k-1}-\omega_{k-1}v_{k-1}),\\
 v_k&=Ap_k,\qquad
 \alpha_k=\rho_k/\langle\tilde r_0,v_k\rangle,\\
 s_k&=r_k-\alpha_kv_k,\qquad t_k=As_k,\\
 \omega_k&=\langle t_k,s_k\rangle/\langle t_k,t_k\rangle,\\
 x_{k+1}&=x_k+\alpha_kp_k+\omega_ks_k,\\
 r_{k+1}&=s_k-\omega_kt_k.
\end{aligned}
$$

需要保护 $\rho_k$、$\omega_k$、各分母和 NaN/Inf；递推残差只是一种估计，必须周期性以 full operator 刷新。`pyqcu/solver/_bistabcg.py` 暴露 `bistabcg` 与历史记录接口，strict C++ 内核也实现了融合的 BiCGStab 更新。

### 4.4 Arnoldi、GMRES 与 FGMRES

Arnoldi 在一般非 Hermitian $A$ 上构造

$$
 \mathcal K_m(A,r_0)=\operatorname{span}\{r_0,Ar_0,\ldots,A^{m-1}r_0\},
 \qquad
 AV_m=V_{m+1}\bar H_m.
$$

Modified Gram–Schmidt（MGS）的一步为

$$
 w=Av_j,
 \quad h_{ij}=\langle v_i,w\rangle,
 \quad w\leftarrow w-h_{ij}v_i,
 \quad h_{j+1,j}=\|w\|_2,
 \quad v_{j+1}=w/h_{j+1,j}.
$$

GMRES 在小型 Hessenberg 系统上最小化残差；FGMRES 允许每步预条件器变化：

$$
 z_j=M_j^{-1}v_j,
 \qquad w_j=Az_j,
 \qquad x_m=x_0+Z_my.
$$

当 MG hierarchy、warm start、SAP 或内层迭代次数在外层循环中变化时，$M_j$ 不再固定，应使用 FGMRES 或 flexible-GCR，不能假设普通 GMRES 的固定预条件语义仍然成立。

**实现**：`pyqcu/solver/_gmres.py` 包含复数 Givens 与 restart 逻辑；strict C++ 入口 `applyMultigridStrictFgmresQcu` 对应右预条件语义。应分别记录 Arnoldi 估计残差和 full true residual。

### 4.5 GCR、CA-GCR 与通信规避

GCR 直接对预条件后的方向做残差正交化，适合非 Hermitian 和变化的预条件器。CA-GCR/CA-CG 通过块 Krylov 或 power basis 减少全局归约次数，但会增加局部正交化和失正交风险。通信规避只减少同步，不能改变算子谱；若块长度过大，局部基条件数会成为主导误差源。

QUDA 的 `inv_gcr_quda.cpp`、`inv_ca_gcr.cpp` 是参考入口；PyQCU 当前生产路径更接近 FGMRES 和 CA-CG 的组合。比较时要报告 restart/block 长度、正交化方法和归约精度。

### 4.6 Lanczos、QR 与局部基

Hermitian $H$ 上的 Lanczos 递推为

$$
 Hq_j=\beta_{j-1}q_{j-1}+\alpha_jq_j+\beta_jq_{j+1}.
$$

有限精度会产生 ghost eigenvalues，因此 thick-restart 版本必须用 Ritz true residual 复核。`pyqcu/solver/_lanczos.py` 的定位是特征底座，而不是自动替代 MG setup。

粗空间的局部基在每个 aggregate 内构造。对局部矩阵 $V$，理想条件是

$$
 V^\dagger V\approx I.
$$

CGS 单次正交便宜但容易失正交；MGS 更稳定；块 QR 最稳健但需要更多局部工作区。实现可根据 block 大小选择方法，但必须以 $\|V^\dagger V-I\|$ 和 coarse true residual 验收，而非只看 setup 是否完成。

### 4.7 算法特性对比

| 算法 | 算子条件 | 主要成本 | 常见失败模式 | PyQCU 状态 |
|---|---|---|---|---|
| CG | SPD | 1 次 $A$、少量归约 | 非 SPD、负分母 | 实现/参考 |
| 多移位 CG | SPD + 标量移位 | 共享 Krylov，多个标量更新 | 非标量局部项、停止不一致 | 实现（Python） |
| BiCG | 一般非 Hermitian + $A^\dagger$ | 双 matvec/归约 | shadow breakdown | 参考 |
| CGS | 一般非 Hermitian | 少 shadow 存储 | residual polynomial 放大误差 | 局部正交参考 |
| BiCGStab | 一般非 Hermitian | 2 次 matvec/迭代 | $\rho$/$\omega$ breakdown | 实现 |
| GMRES | 一般非 Hermitian | 长基和全局归约 | 内存增长、失正交 | 参考/部分实现 |
| FGMRES | 变化预条件 | 保存 $Z$ 与 $V$ | 预条件器返回异常 | 实现 |
| GCR | 一般非 Hermitian | 历史方向内积 | 历史向量多、失正交 | QUDA 参考 |
| CA-CG/CA-GCR | 适合块 Krylov | 少同步、多局部算子 | 块条件数恶化 | CA-CG 实现 |
| MR/Chebyshev | 平滑或谱区间已知 | 固定短递推 | 谱估计错误 | MR 实现，Chebyshev 参考 |


### 4.8 两个可执行递推模板

下面的模板刻意把“递推残差”和“可靠残差”分开。`true_residual()` 必须重新调用完整目标算子，而不是复用上一轮的局部变量。

```text
BiCGStab(b, A, x0):
    x = x0
    r = b - A(x)
    r_shadow = copy(r)
    p = 0; v = 0; rho_old = alpha = omega = 1
    for k = 0, 1, ...:
        rho = dot(r_shadow, r)
        if nonfinite(rho) or abs(rho) <= breakdown_floor: return BREAKDOWN
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = A(p)
        denom = dot(r_shadow, v)
        if nonfinite(denom) or abs(denom) <= breakdown_floor: return BREAKDOWN
        alpha = rho / denom
        s = r - alpha * v
        if norm(s) <= inner_tol: x = x + alpha * p; break
        t = A(s)
        tt = dot(t, t)
        if nonfinite(tt) or real(tt) <= breakdown_floor: return BREAKDOWN
        omega = dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        if k % reliable_period == 0: r = b - A(x)
        if reliable_norm(r) <= atol * norm(b): return x
        rho_old = rho
```

```text
FGMRES(m, b, A, preconditioner_sequence):
    r0 = b - A(x0); beta = norm(r0)
    if beta == 0: return x0
    v[0] = r0 / beta; g = (beta, 0, ...)
    for j in 0, ..., m-1:
        z[j] = preconditioner_sequence[j](v[j])
        w = A(z[j])
        for i in 0, ..., j:
            H[i,j] = dot(v[i], w); w -= H[i,j] * v[i]
        H[j+1,j] = norm(w)
        if H[j+1,j] != 0: v[j+1] = w / H[j+1,j]
        apply_previous_givens(H[:,j]); choose_new_givens(H[j,j], H[j+1,j])
        update_small_rhs(g)
        if abs(g[j+1]) <= restart_tol: break
    y = back_substitute(H[0:j,0:j], g[0:j])
    x = x0 + sum(y[i] * z[i] for i in range(j))
    return x, true_residual(b, A, x)
```

`breakdown_floor` 应与数据类型、局部范数和全局归约精度相关，不能固定成与所有格点尺度无关的常数。对分布式计算，`dot` 的归约顺序会影响最后若干位；验收时应比较残差数量级和物理结果，而不是要求不同 MPI 排布逐 bit 相同。

---

## 5. 平滑器、局部预条件与多重网格

### 5.1 Richardson、MR 与 Chebyshev

最简单的 Richardson 更新为

$$
 x_{k+1}=x_k+\omega(b-Ax_k).
$$

MR 选择复标量使新残差在当前方向上最小：

$$
 \alpha_k=\frac{\langle Ar_k,r_k\rangle}{\langle Ar_k,Ar_k\rangle},
 \qquad
 x_{k+1}=x_k+\alpha_kr_k.
$$

实际代码可能使用 $A^\dagger r_k$ 或不同符号约定，必须检查 `matvec_dag` 的语义。`pyqcu/solver/_mr.py` 适合作为短迭代 smoother，不能仅凭函数名把它当作对任意非 Hermitian 算子的全局求解器。

Chebyshev smoother 用谱区间 $[\lambda_{\min},\lambda_{\max}]$ 上的多项式压制高频误差。谱上下界若估计错误，过度放大会造成不稳定；因此实际流程应先用短 Lanczos/幂迭代估计区间，并在小规模上验证误差传播因子。

### 5.2 Schwarz、SAP 与局部性

把格点划分为重叠或不重叠子域，Schwarz 预条件器可写为

$$
 M^{-1}=\sum_i R_i^T A_i^{-1}R_i
$$

（additive）或按子域顺序复合（multiplicative）。SAP 在偶数子域与奇数子域之间交替更新，适合最近邻算子和 MPI halo。子域越小，局部求解便宜但跨子域耦合更强；重叠越大，迭代改善通常越好但通信和存储成本上升。

PyQCU 当前主路径以 MR、FGMRES 和 MG 组合为主；SAP/Schwarz 在 QUDA 侧有成熟参考实现，属于对照而非 PyQCU 的统一生产 API。

### 5.3 Galerkin 粗化

令 $P$ 为延拓、$R$ 为限制，标准 Galerkin 粗算子为

$$
 A_c=R A_f P.
$$

若 $R=P^\dagger$，且细层算子有相应 Hermitian 结构，粗层可保留部分对称性。对非 Hermitian 或 asymmetric Schur 路径，$R$ 与 $P^\dagger$ 是否相同必须显式说明。

一次 V-cycle 可以抽象为

```text
pre-smooth:   x <- S_pre(A_f, b, x)
r_f = b - A_f x
r_c = R r_f
e_c = coarse_solve(A_c, r_c)
x <- x + P e_c
post-smooth:  x <- S_post(A_f, b, x)
```

平滑器压制局部高频误差，粗空间表示跨 aggregate 的低频误差。若 null vector 不包含 near-null space，coarse correction 可能对高频有效而对物理低模无效，外层迭代不会真正改善。

### 5.4 Native、legacy/compact 与 strict

| 路径 | 粗化对象 | parity 语义 | 典型用途 | 限制 |
|---|---|---|---|---|
| native Python MG | PyTorch 算子与 R/P | 由 Python hierarchy 决定 | 原型、CPU/CUDA 验证 | 性能和 ABI 不等同 QCU |
| legacy/compact | odd/even Schur 或 33-tensor stencil | 可只保留目标 parity | 与旧 QCU/QUDA compact 接口对齐 | full operator 信息被压缩 |
| strict full | full $D_f$ 的 Galerkin $R D_f P$ | coarse field 保留 full geometry | strict `X/Y/Yhat`、full MATPC | 当前原型有单 rank/最近邻等门槛 |

`pyqcu/solver/_multigrid.py` 是 native 入口；`_quda_multigrid.py` 覆盖 legacy/compact 兼容逻辑；`pyqcu/tools/_strict_galerkin.py` 和 `cpp/cuda/qcu/src/apply_multigrid_strict.cu` 对应 strict 构造与应用。三者的 transfer、residual、资产布局不能互换。

### 5.5 预条件外层的选择

若 coarse solve 或 smoother 的精度、迭代数、warm state 会变化，记预条件器为 $M_k^{-1}$，外层选择 FGMRES/GCR：

$$
 z_k=M_k^{-1}v_k,
 \qquad
 w_k=A z_k.
$$

若预条件器固定且目标算子 SPD，CG 更节省存储；若是 Schur、Clover 或 strict full 非 Hermitian，则使用 BiCGStab、FGMRES 或 GCR，并独立检查 full residual。


### 5.6 完整 V-cycle 的层间语义

设第 $l$ 层算子为 $A_l$、延拓为 $P_l$、限制为 $R_l$。一次递归 V-cycle 可写成

$$
\begin{aligned}
 x_l^{(1)}&=S_l^{\nu_{\mathrm{pre}}}(A_l,b_l,x_l^{(0)}),\\
 r_l&=b_l-A_lx_l^{(1)},\\
 b_{l+1}&=R_lr_l,\\
 e_{l+1}&=\operatorname{Vcycle}(l+1,b_{l+1},0),\\
 x_l^{(2)}&=x_l^{(1)}+P_le_{l+1},\\
 x_l^{\mathrm{out}}&=S_l^{\nu_{\mathrm{post}}}(A_l,b_l,x_l^{(2)}).
\end{aligned}
$$

最粗层不再递归，而调用指定的 coarse solver。若 outer solver 使用右预条件，V-cycle 返回的是 $e_l=M_l^{-1}r_l$；它不是直接把 $b_l$ 当作独立物理右端求解。这个区别决定 FGMRES 中保存 $z_j$ 而不是把 $v_j$ 直接累加。

```text
Vcycle(level, b, x):
    if level == coarsest:
        return coarse_solver(A[level], b, x)
    x = smoother(A[level], b, x, nu_pre)
    r = b - A[level](x)
    b_c = restrict(level, r)       # full 或 compact 由 hierarchy 定义
    e_c = Vcycle(level + 1, b_c, 0)
    x = x + prolong(level, e_c)
    x = smoother(A[level], b, x, nu_post)
    return x
```

`restrict` 返回的 coarse geometry 必须和 hierarchy 的 parity 约定一致。对于 fine MATPC，只有在 prepare/reconstruct 边界才使用 compact parity 视图；将 compact rhs 直接传给 full coarse operator 会造成维度看似匹配但物理语义错误。

### 5.7 Setup、solve 和 warm state 的生命周期

MG setup 至少包含三类持久资产：

1. **transfer**：每层的 null vectors、局部正交基、aggregate 映射；
2. **operator**：$X$、raw $Y$、preconditioned $\widehat Y$ 或 33-tensor stencil；
3. **solver state**：粗层分解、预分配 workspace、可选的 warm initial guess。

setup 改变任一资产后，旧的 solver state 都不能静默复用。热启动只表示用上一次 `fermion_out` 作为 $x_0$；它不改变 $A$、$b$ 或 stopping criterion。若用户切换 gauge、质量、边界、precision 或 hierarchy，却保留旧 state，必须显式失效并重建。

---

## 6. 粗空间、Galerkin 与 coarse operator

### 6.1 局部 null vector 与 transfer

设每个 fine aggregate 内有 $n_v$ 个局部向量，拼成

$$
 V_l(x)=\big[v_1(x),v_2(x),\ldots,v_{n_v}(x)\big].
$$

对 coarse 向量 $\phi_c$，延拓为

$$
 (P\phi_c)(x)=V_l(x)\phi_c(\mathcal A(x)),
$$

若 $R=P^\dagger$，限制为

$$
 (R\psi_f)(\mathcal A)=\sum_{x\in\mathcal A}V_l(x)^\dagger\psi_f(x).
$$

在每个 aggregate 内进行 CGS、MGS 或块 QR，并检查

$$
 \epsilon_{RP}=\|R P-I\|,
 \qquad
 \epsilon_{\mathrm{orth}}=\max_{\mathcal A}\|V_l(\mathcal A)^\dagger V_l(\mathcal A)-I\|.
$$

这两个量比“生成了多少个向量”更能反映粗空间质量。

### 6.2 Strict full `X/Y/Yhat`

对 full fine operator $D_f$ 做 strict Galerkin：

$$
 D_c=R D_f P.
$$

如果 coarse stencil 只允许 onsite 与一个格点位移，则按位移提取

$$
 D_c = X_c + \sum_\mu
 \left(Y_c^{+\mu}T_{+\mu}+Y_c^{-\mu}T_{-\mu}\right).
$$

其中 $X_c$ 是 onsite block，$Y_c^{\pm\mu}$ 是最近邻 block。若出现超出一跳的 support，strict 路径应 fail-closed，而不是静默丢弃项。

对 coarse-PC，常用资产为

$$
 \widehat Y^{+\mu}=X^{-1}Y^{+\mu},
 \qquad
 \widehat Y^{-\mu}=Y^{-\mu}X^{-1}
$$

或与实现约定等价的 dagger 版本。文档和代码必须共同说明：$X^{-1}$ 在左边还是右边、是否需要 $X^{-\dagger}$、以及 kernel 读取哪个方向的 link。

`build_strict_galerkin` 使用 batched source probes；colored 版本按不重叠颜色分批以降低峰值内存。当前输入检查明确限制 blocked-V、spin/parity 和设备范围；这些约束属于实现事实，不应在报告中泛化为所有 QUDA 粗算子。

### 6.3 Legacy 33-tensor 与 full operator 的区别

legacy 33-tensor stencil 可把一个 coarse site 的局部矩阵和方向耦合压缩为固定张量集合，适合旧 ABI 和 compact Schur。strict full 路径则首先保留 full geometry，再由位移提取 $X/Y$。因此：

- compact odd operator 不能直接当作 full $D_c$；
- full Galerkin 的 `R/P` 需包含两种 parity 的映射；
- coarse residual 与 fine MATPC residual 的重建步骤不同；
- legacy 资产若缺少 raw $Y$，不能反推出 strict 的 full nearest-neighbor support。

### 6.4 粗层求解与平滑

粗层不必使用与 fine 层相同的 solver。一个稳健组合是：

1. fine 层：MR 或 Chebyshev 做 $ν_{\mathrm{pre}}$ 次预平滑；
2. 限制 full residual 到 coarse；
3. 中间层递归 V-cycle；
4. 最粗层：CG（仅 SPD）或 BiCGStab/FGMRES/GCR；
5. 延拓后做 $ν_{\mathrm{post}}$ 次后平滑；
6. 用 full true residual 检查外层停止。

平滑次数过少会留下高频误差，过多则把粗网格优势消耗在 fine matvec 上。应记录每个层的 operator 应用次数和 coarse correction 的残差下降，而不只记录外层总迭代数。


### 6.5 粗算子资产和内存核对

对每个 coarse site，若粗自由度为 $E$，一个 dense onsite block 使用 $E^2$ 个复数元素。实际资产还可能包括：

| 资产 | 作用 | 释放时机 |
|---|---|---|
| `null_vectors` | 生成 $P/R$ | hierarchy 销毁或重新 setup |
| `X` | onsite block | coarse apply 和 $X^{-1}$ 构造期间 |
| `X_inv` | onsite inverse/factorization | coarse-PC 或 local solve 期间 |
| `Y_raw` | 未预条件最近邻 block | 诊断、重新生成 `Yhat` |
| `Yhat` | coarse-PC hopping | strict coarse kernel |
| `stencil_33` | legacy compact 资产 | legacy coarse apply |

内存检查必须区分“仍被 hierarchy 引用的 live bytes”和 CUDA allocator 的缓存。`empty_cache()` 只能释放缓存，不能证明 `set_ptrs` 中的指针已失效；正确做法是 close hierarchy 后检查引用、workspace 和 live allocation 是否回到基线。

### 6.6 Full、MATPC 与 compact 的映射表

| 操作 | 输入语义 | 输出语义 | 可与哪条路径互换 |
|---|---|---|---|
| full dslash | full spin-color field | full field | 仅 full operator |
| prepare | full rhs + target parity | compact Schur rhs | fine MATPC |
| compact MATPC | target parity field | target parity field | asymmetric Schur/compact hierarchy |
| reconstruct | compact solution + eliminated rhs | full solution | fine MATPC 边界 |
| strict restrict/prolong | full geometry + strict transfer | full coarse geometry | strict hierarchy |
| legacy restrict/prolong | hierarchy-specific parity/33-tensor | legacy coarse geometry | legacy hierarchy |

表中“可互换”只表示数据形状和数学语义同时兼容；不能因为两个张量都有 `E` 维就直接传递。迁移路径时至少要核对 parity、aggregate、spin order、方向次序、dagger 和 normalization。

---

## 7. 其他费米子作用量的对照

本节中的作用量用于理论和 QUDA 对照。除 Wilson/Clover 外，不应据此宣称 PyQCU 具有完整生产 API。

### 7.1 Twisted-mass

简并双重态可写为

$$
 D_{\mathrm{tm}}=D_W(m_0)+i\mu\gamma_5\tau_3,
$$

其中 $\tau_3$ 作用在 flavor 空间。非简并双重态常写为

$$
 D_{\mathrm{nd}}=D_W(m_0)+i\mu_\sigma\gamma_5\tau_1+\mu_\delta\tau_3.
$$

局部 flavor-spin block 会改变 onsite inverse；even-odd 分块时必须把 flavor 结构包含在 $A_e,A_o$ 中。QUDA 参考入口包括 `refer/git-rep/quda/lib/dirac_twisted_mass.cpp` 和 `dslash_ndeg_twisted_mass.cpp`；PyQCU 当前没有公开的完整 twisted-mass dslash/params ABI。

### 7.2 Twisted-clover

将 Clover 与 twist 合并得到

$$
 D_{\mathrm{tmC}}=D_W+C_{\mathrm{SW}}+i\mu\gamma_5\tau_3,
$$

或在非简并情形采用上式的 flavor block。Clover 与 twist 都属于 onsite 项，应在粗化时共同进入 $X_c$，不能把 twist 再当作额外 hopping 重复加入。

QUDA 对照文件为 `dirac_twisted_clover.cpp` 和 `dslash_twisted_clover_preconditioned.hpp`；PyQCU 仍标记为参考。

### 7.3 Staggered 与 improved-staggered

Staggered 费米子保留一个分量场 $\chi$，用相位

$$
 \eta_\mu(x)=(-1)^{\sum_{\nu<\mu}x_\nu}
$$

表示 spin-taste 结构。典型无质量 staggered 算子为

$$
 (D_{\mathrm{stag}}\chi)(x)=\frac12\sum_\mu\eta_\mu(x)
 \left[U_\mu(x)\chi(x+\hat\mu)-U_\mu^\dagger(x-\hat\mu)\chi(x-\hat\mu)\right].
$$

asqtad/HISQ 通过 Fat7、Lepage、Naik 等路径改进短程离散误差并抑制 taste breaking；它们产生更宽的 stencil，不能直接塞入只接受最近邻 `X/Y` 的 strict ABI。QUDA 参考文件包括 `dirac_staggered_kd.cpp` 和 `dirac_improved_staggered_kd.cpp`。

### 7.4 Domain-wall 与 Möbius

Domain-wall 把四维物理场扩展到第五维 $s=0,\ldots,L_s-1$。一个示意形式是

$$
 D_{5d}(s,s')=D_W^+(s)\delta_{s,s'}
 -P_-\delta_{s+1,s'}-P_+\delta_{s-1,s'}
 +m_f\left(P_-\delta_{s,0}\delta_{s',L_s-1}
 +P_+\delta_{s,L_s-1}\delta_{s',0}\right),
$$

其中 $P_\pm=(1\pm\gamma_5)/2$。有限 $L_s$ 的墙间混合产生 residual mass；Möbius 通过第五维系数重新参数化核，改善逼近 sign function 的效率。

第五维使每个站点的自由度和 halo/粗化结构改变。QUDA 的 `dirac_domain_wall_4d.cpp`、`dirac_mobius.cpp` 可作参考；QCU 当前 ABI 没有 5-d 动力学的完整入口。

### 7.5 Overlap

Overlap 算子基于 Wilson kernel 的 sign function，典型写法为

$$
 D_{\mathrm{ov}}(m)=\left(1-\frac{am}{2\rho}\right)D_{\mathrm{ov}}(0)+am,
$$

$$
 D_{\mathrm{ov}}(0)=\rho\left[1+\gamma_5\,\operatorname{sign}(H_W)\right],
 \qquad H_W=\gamma_5(D_W-\rho).
$$

sign function 通常用 rational approximation 或多项式近似，内部还需要多次 kernel solve；它不属于当前 QCU Wilson/Clover 直接 dslash 的同一成本模型。本文只给出理论参考，未发现 `DiracOverlap` 的 PyQCU 生产实现。

### 7.6 Action 对照表

| Action | 自由度/支撑 | 主要数值难点 | 本库状态 |
|---|---|---|---|
| Wilson | 四分量旋量、最近邻 | 轻质量条件数、加性质量重整化 | 实现 |
| Clover | Wilson + onsite spin-color block | 局部 inverse、非平凡 Schur | 实现 |
| Twisted-mass | flavor-spin onsite 项 | flavor block、dagger 约定 | QUDA 参考 |
| Twisted-clover | Clover + flavor-spin onsite | 更大的 $X$、非 Hermitian | QUDA 参考 |
| Staggered | 单分量、相位 stencil | taste、rooting 语义 | QUDA 参考 |
| asqtad/HISQ | 宽 stencil | 多路径通信、宽 support | QUDA 参考 |
| Domain-wall | 五维局部耦合 | $L_s$ 成本、残余质量 | QUDA 参考 |
| Möbius | 五维重参数化 | 第五维谱逼近 | QUDA 参考 |
| Overlap | sign function 非局部近似 | 多层嵌套求解 | 理论参考 |

---

## 8. PyQCU/QUDA 实现映射与 ABI 约束

### 8.1 Python 求解器和工具

| 功能 | 入口 | 备注 |
|---|---|---|
| BiCGStab | `pyqcu/solver/_bistabcg.py` | 支持 history、零 RHS/残差短路和 breakdown guard |
| FGMRES | `pyqcu/solver/_gmres.py` | 复数 Givens、restart、变化预条件 |
| MR | `pyqcu/solver/_mr.py` | 更适合作为 smoother，需核对 dagger 语义 |
| CA-CG | `pyqcu/solver/_cacg.py` | 块 least-residual，和 QUDA power-basis CA-CG 不同 |
| Lanczos | `pyqcu/solver/_lanczos.py` | thick restart、Ritz true residual |
| 多移位 CG | `pyqcu/solver/_multishift_cg.py` | 仅适用满足其假设的移位 SPD 系统 |
| native MG | `pyqcu/solver/_multigrid.py` | Python V-cycle，可调用 CUDA restrict |
| legacy MG | `pyqcu/solver/_quda_multigrid.py` | 旧/compact hierarchy 兼容层 |
| strict Galerkin | `pyqcu/tools/_strict_galerkin.py` | batched/colored full coarse 构造 |
| transfer/stencil | `pyqcu/tools/_multigrid.py` | null vector、R/P、33-tensor 等工具 |

纯 Python 模块遵循仓库约定：通过 `pyqcu.cann as _torch` 访问兼容层，不直接在 NPU 兼容路径中硬编码 `import torch`。这条约束与本文的数学内容无关，但决定文档中的代码入口是否可以在不同设备后端复用。

### 8.2 Cython/CUDA 接口

C API 在 `cpp/cuda/qcu/python/pyqcu.h` 声明，Cython 桥在 `pyqcu/cuda/qcu/qcu.pyx` 暴露。strict 相关入口包括：

- `applyMultigridStrictInitQcu` / `applyMultigridStrictEndQcu`；
- `applyMultigridStrictCoarseQcu`、`applyMultigridStrictMatPCQcu`；
- `applyMultigridStrictPrepareQcu`、`applyMultigridStrictReconstructQcu`；
- `applyMultigridStrictRestrictQcu`、`applyMultigridStrictProLongQcu`；
- `applyMultigridStrictVCycleQcu`、`applyMultigridStrictFgmresQcu`。

三个扁平桥接数组必须同步：

| 数组 | 当前约定 | 用途 |
|---|---|---|
| `params` | `int32[58]` | 计划、维度、索引、MG 开关和热启动标志 |
| `argv` | `float[7]` | 容差、质量或 solver 标量参数 |
| `set_ptrs` | `int64[100]` | C++ `LatticeSet`、scratch 和 hierarchy 指针 |

`pyqcu/cuda/define.py` 与 `cpp/cuda/qcu/include/define.h` 必须保持同一索引。dev84 起 `_MG_USE_DEFLATE_=55`、`_MG_MU_PRE_=56`；dev87 起 `_MG_USE_INIT_GUESS_=57`，热启动时由 `fermion_out` 提供初值。

### 8.3 生命周期和 `_SET_INDEX_`

普通 QCU 调用遵循

```text
applyInitQcu
  -> dslash / solver / restrict / coarse operation
  -> params[define._SET_INDEX_] += 1
  -> applyEndQcu
```

每次操作都必须递增 `_SET_INDEX_`。不递增会使 scratch 缓冲和 `LatticeSet` 索引复用，产生难以由单次 kernel 检查发现的结果错误。多线程多卡时，每个线程必须持有独立的 `params`、`argv`、`set_ptrs` 副本；Cython 在取指针时持有 GIL，进入 C++ 后在 `with nogil` 中运行。

### 8.4 MPI 与 strict gate

strict C++ 路径在当前实现中对单 rank 有明确门槛；代码会检查 MPI 状态，并在不满足条件时 fail-closed。不要把单 rank 的全局 dot、周期 true-residual 刷新或内存测量结果外推到多 rank。普通 Wilson/Clover QCU 路径支持 MPI halo，但 layout、rank topology 和边界条件仍需逐项记录。

### 8.5 HDF5、缓存和可复现性

所有持久化应通过 `pyqcu/tools/_io.py` 的 h5py 封装；每次调用使用独立 File 句柄，多线程场景不要共享可变句柄。null-vector 或 coarse hierarchy 的缓存应在一次句柄生命周期内写完全部 dataset，避免逐 dataset 覆盖式重建。保存文件时同时记录：格点尺寸、边界、dtype、$\kappa$ 或质量、Clover 系数、block、$n_v$、parity、gamma basis 和生成代码版本。


### 8.6 `_SET_PLAN_` 与参数索引速查

`_SET_PLAN_` 用整数选择后端操作。当前约定为：

| 值 | 计划 |
|---:|---|
| `-2` | Laplacian |
| `-1` | Gauss gauge |
| `0` | Wilson dslash |
| `1` | BiStabCG/CG |
| `2` | Clover dslash |

计划值只是选择主分支，仍需配合 `params` 中的几何、precision、parity 和 MG 标志。新增字段时要同时更新 Python `define.py`、C++ `define.h`、Cython pxd/pyx 和文档；只修改一端会造成 ABI 静默错位。

### 8.7 多 GPU 和线程隔离

`pyqcu/cuda/_multi_gpu.py` 的 `MultiGpuMultigrid` 使用一线程一卡模型。每个线程必须拥有独立的：

- CUDA device context；
- `params`、`argv`、`set_ptrs` 副本；
- hierarchy 和 scratch；
- `_SET_INDEX_` 计数器（各自从零开始）。

Cython 函数在 GIL 段只完成指针提取和参数检查，进入 C++ kernel 后使用 `with nogil`。pxd 的 cdef extern 声明必须标记 `nogil`，并用 `qcu_api.pxd` 别名避免与 pyx 的 Python `def` 同名。MPI 方面，MultiGpuMultigrid 要求单 MPI rank；C++ `LatticeSet` 会以 `COMM_WORLD` rank 覆盖后端的 `_NODE_RANK_`。

---

## 9. 误差、性能与精度预算

### 9.1 误差分解

一次求解的误差可按来源拆分为

$$
 e_{\mathrm{total}}
 \lesssim e_{\mathrm{disc}}
 +e_{\mathrm{setup}}
 +e_{\mathrm{alg}}
 +e_{\mathrm{round}}
 +e_{\mathrm{comm}}.
$$

- $e_{\mathrm{disc}}$：格点离散误差，如 Wilson 的 $O(a)$ 项或 Clover 改进后的剩余项；
- $e_{\mathrm{setup}}$：null vector 不充分、局部正交误差、Galerkin 截断误差；
- $e_{\mathrm{alg}}$：solver 尚未达到目标容差；
- $e_{\mathrm{round}}$：混合精度、归约和递推残差漂移；
- $e_{\mathrm{comm}}$：halo、MPI reduction 或布局转换中的实现误差。

solver 停止条件只直接控制 $e_{\mathrm{alg}}$ 的一部分；不能用更小的 `tol` 修复错误的 gamma 约定或错误的 parity 映射。

### 9.2 混合精度与可靠更新

在 fp32 kernel、fp64 累加或 c64/c128 混合路径中，递推残差可能与 true residual 脱钩。可靠更新的通用策略是：

1. 用低精度迭代推进；
2. 每隔固定迭代数或当递推残差下降到阈值时，重新计算 $r=b-Ax$；
3. 用 true residual 重置递推状态或重新启动外层 Krylov；
4. 记录刷新前后残差比，确认没有 NaN/Inf 或突然跃迁。

strict 单 rank 路径在主循环中有周期性 true-residual 刷新逻辑；多 rank 是否启用需以当前后端代码为准。任何性能报告都应同时给出精度和 residual 语义。

### 9.3 粗化的成本模型

令 fine 站点数为 $V_f$，coarse 站点数为 $V_c$，每个 coarse site 的自由度为 $E=n_vN_s^{\mathrm{coarse}}$。粗算子一次应用的主要成本近似为

$$
 C_{\mathrm{coarse}}
 \sim V_c\left(C_X E^2+8C_YE^2\right)
 +C_{\mathrm{reduce}}N_{\mathrm{global\ dot}}.
$$

setup 还包含 null vector 的存储和 Galerkin probes：

$$
 M_{\mathrm{setup}}
 \sim O\left(V_f n_vN_s\right)
 +O\left(V_cE^2\right).
$$

这只是量级模型。GPU 实际瓶颈可能由内存带宽、非合并访问、MPI latency、全局归约或临时张量峰值决定；没有同一设备上的 profiler 和 true residual 数据，不能从公式推出加速比。

### 9.4 可靠的性能对比

对两个 solver 做公平比较时至少固定：

- 同一 gauge、格点、边界和右端；
- 同一 gamma/color/layout 约定；
- 同一初值和容差定义；
- 同一精度、预热和 MPI rank/GPU 绑定；
- 以 full true residual 达标作为成功条件；
- 分开报告 setup、solve、通信和峰值显存时间。

只比较“外层迭代数”会掩盖每次迭代的 matvec 数、粗层成本和可靠刷新开销。

---

## 10. 可复现验证矩阵

### 10.1 算子级测试

| 编号 | 测试 | 通过条件 |
|---|---|---|
| O1 | $U_\mu=I$ 自由场 dslash | Python、QCU、参考实现逐元素一致 |
| O2 | Wilson $\gamma_5$-Hermiticity | $\|D^\dagger-\gamma_5D\gamma_5\|/\|D\|$ 在精度容差内 |
| O3 | Clover local block | $A_p$ 的布局、dagger 和批量 inverse 一致 |
| O4 | parity round-trip | full → parity → full 不改变字段 |
| O5 | Schur reconstruction | 重建后的 full residual 与直接 $D$ 应用一致 |
| O6 | transfer | $\|R P-I\|$ 和 aggregate 内正交误差可记录 |
| O7 | Galerkin | 逐列、batched、colored 构造在同一输入上等价 |
| O8 | strict support | 出现超出最近邻 support 时显式失败 |

### 10.2 Solver 级测试

| 编号 | 测试 | 必须记录 |
|---|---|---|
| S1 | 零右端 | 返回零解，不出现除零或 NaN |
| S2 | 已知解 | 设 $b=Ax_\star$，检查迭代解与 $x_\star$ 的误差 |
| S3 | breakdown | 分母为零/非有限时返回明确状态 |
| S4 | 递推 vs true residual | 定期重算并记录两者比值 |
| S5 | FGMRES 变化预条件 | 改变内层步数仍保持正确右预条件语义 |
| S6 | warm start | `x0=0` 与热启动分别报告迭代和最终残差 |
| S7 | mixed precision | 低精度推进、高精度刷新后仍达到同一容差 |
| S8 | MPI | rank 数变化时全局范数和解的一致性可解释 |

### 10.3 MG 级测试

```text
1. 生成固定 gauge、固定随机种子和固定 null vectors
2. 保存 layout、dtype、block、nvec、parity、边界和版本元数据
3. 分别构造 native、legacy/compact、strict hierarchy
4. 检查 R P、local orthogonality、X inverse 和 support
5. 对同一个 full rhs 运行一次 V-cycle
6. 用 full operator 计算 true residual 和误差传播因子
7. 再运行外层 FGMRES/BiCGStab，报告 setup/solve/通信/显存
```

### 10.4 源码证据与运行证据分离

源码检查可以证明公式、入口和边界判断存在；它不能证明特定设备上的性能或全部布局组合正确。报告、日志和论文中应把以下两类证据分开：

- **静态证据**：文件、函数、参数索引、kernel 和异常分支；
- **动态证据**：命令、设备、rank、耗时、迭代历史、true residual 和退出码。

当前文档只对仓库内源码锚点做静态整理；没有在本轮启动完整 CUDA/MPI 回归，因此性能和跨设备结论仍标记为未验证。


### 10.5 常见失败模式与定位顺序

| 现象 | 首先检查 | 物理/工程含义 |
|---|---|---|
| 自由场 dslash 方向符号错误 | forward/backward link、反周期边界 | 邻居索引或 gauge transport 反了 |
| $\gamma_5$-Hermiticity 失败 | gamma basis、Clover dagger、颜色矩阵次序 | 算子约定不一致，不能继续比较 solver |
| Schur 能收敛但 full residual 大 | prepare/reconstruct、被消去 parity 的局部 inverse | 压缩解没有正确恢复 |
| strict coarse 出现非最近邻 | transfer block、wide stencil action | strict ABI 不适用，应 fail-closed |
| 递推 residual 很小、true residual 不降 | mixed precision、reliable update、全局归约 | 数值漂移或 residual 语义混用 |
| 单线程正确，多线程错误 | `params/argv/set_ptrs` 是否共享 | scratch/索引发生跨线程复用 |
| MPI rank 增加后结果漂移大 | local/global dot、halo 边界和 layout | 归约或通信语义错误 |
| 显存持续增长 | hierarchy 引用、HDF5 句柄、workspace | allocator 缓存不等于真正泄漏 |

推荐顺序是“算子 → parity → transfer → coarse apply → 单次 V-cycle → 外层 solver”。不要在 full dslash 尚未通过时直接调大 MG 层数或改变 smoother，这会把根因隐藏在多个误差源中。

### 10.6 最小文档化运行记录

每次可复现实验至少保存如下元数据：

```text
commit/tag:        stab52 或工作区提交
lattice:           Lx x Ly x Lz x Lt
boundary:          periodic / anti-periodic(t)
action:            Wilson / Clover + 参数
layout:            PyQCU xyzt / HDF5 zyxt / QUDA parity-order
dtype:             complex64 / complex128
solver:            outer + inner + restart/block
mg:                levels, block, nvec, smoother, coarse solver
rhs/x0:            source type, seed, zero or warm start
stop:              atol, rtol, true-residual definition
runtime:           GPU, MPI ranks, threads, setup/solve time
result:            iterations, residual history, max memory, exit status
```

缺少这些字段时，报告可以描述算法，但不能声称两个结果具有严格的性能可比性。

---

## 11. 算法选择速查

| 问题 | 优先选择 | 关键理由 |
|---|---|---|
| SPD 的 $D^\dagger D$ 或 Hermitian coarse operator | CG、CA-CG、多移位 CG | 有共轭方向和最小化理论 |
| 原始 Wilson/Clover 或 asymmetric Schur | BiCGStab、FGMRES、GCR | 不要求算子 SPD |
| MG/SAP 内层精度或步数变化 | FGMRES、flexible-GCR | 允许 $M_k$ 变化 |
| 粗层高频误差 | MR、Chebyshev、Schwarz/SAP | 平滑局部高频 |
| 轻质量近零模 | MG、deflation、thick-restart Lanczos | 直接改善低模分辨率 |
| full Clover parity solve | asymmetric/symmetric Schur + prepare/reconstruct | 保证 full residual 语义 |
| legacy 与 strict 对照 | 分别保留 hierarchy 和 layout | 两者的 parity/资产定义不同 |
| 宽 stencil action | action-specific coarse operator | 最近邻 `X/Y` ABI 不足以表达全部 support |

一个最低风险的决策顺序是：先判定 operator 是否 SPD，再判定预条件器是否固定，最后判定粗算子是 full、Schur 还是宽 stencil。只要其中一项不满足，不能直接套用 CG 或 strict 最近邻 coarse kernel。

---


## 12. 资料与源码索引

### 12.1 PyQCU 与 QCU

1. `pyqcu/dslash/_wilson.py`、`pyqcu/dslash/_clover.py`、`pyqcu/dslash/_operator.py`；
2. `pyqcu/solver/_bistabcg.py`、`_gmres.py`、`_mr.py`、`_cacg.py`、`_lanczos.py`、`_multishift_cg.py`；
3. `pyqcu/solver/_multigrid.py`、`_quda_multigrid.py`；
4. `pyqcu/tools/_multigrid.py`、`_strict_galerkin.py`、`_io.py`；
5. `cpp/cuda/qcu/python/pyqcu.h`、`cpp/cuda/qcu/src/apply_multigrid_strict.cu`；
6. `pyqcu/cuda/qcu/qcu.pyx`、`pyqcu/cuda/define.py`、`cpp/cuda/qcu/include/define.h`。

### 12.2 QUDA 快照

1. `refer/git-rep/quda/lib/multigrid.cpp`、`dirac_coarse.cpp`、`coarse_op.cuh`；
2. `dirac_clover.cpp`、`dirac_twisted_mass.cpp`、`dirac_twisted_clover.cpp`；
3. `dirac_staggered_kd.cpp`、`dirac_improved_staggered_kd.cpp`；
4. `dirac_domain_wall_4d.cpp`、`dirac_mobius.cpp`；
5. `inv_gcr_quda.cpp`、`inv_ca_gcr.cpp`、`inv_bicgstabl_quda.cpp`。

### 12.3 理论参考

- K. G. Wilson, *Confinement of quarks*, Phys. Rev. D **10** (1974) 2445；
- B. Sheikholeslami and R. Wohlert, Nucl. Phys. B **259** (1985) 572；
- G. P. Lepage, Phys. Rev. D **59** (1999) 074502；Follana et al., Phys. Rev. D **75** (2007) 054502；
- D. B. Kaplan, Phys. Lett. B **288** (1992) 342；Y. Shamir, Nucl. Phys. B **406** (1993) 90；
- R. C. Brower, H. Neff and H. Orginos, Nucl. Phys. B Proc. Suppl. **153** (2006) 3；
- H. Neuberger, Phys. Lett. B **417** (1998) 141；
- Y. Saad and M. H. Schultz, SIAM J. Sci. Stat. Comput. **7** (1986) 856；
- H. A. van der Vorst, SIAM J. Sci. Stat. Comput. **13** (1992) 631。

### 12.4 结论边界

Wilson/Clover dslash、Python/native MG、legacy/compact hierarchy、strict full coarse、Galerkin transfer、MATPC、MR、CG、BiCGStab、FGMRES、局部 CGS/QR 和 QCU 生命周期在本库中有可追溯实现或接口。Twisted-mass、staggered/HISQ、domain-wall、Möbius、overlap 在本文中主要承担理论和 QUDA 对照角色，不应被当作 PyQCU 已公开的完整生产功能。任何加速比、跨 MPI 规模扩展性和物理观测量结论，都需要独立的运行记录和误差分析。

## 附录 A：从规范场到一次可审计求解的完整流程

下面的流程把物理输入、算子构造、MG setup、外层 Krylov 和验收连接起来。它也是排查“结果看似收敛但物理约定不一致”的推荐顺序。

### A.1 输入和不变量

```text
输入：U_mu(x), b, kappa 或 m0, lattice=(Lx,Ly,Lz,Lt)
可选：Clover coefficient cSW, null-vector count nv, block, levels
记录：gamma basis、边界条件、dtype、layout、MPI topology、随机种子
断言：所有时空轴在计算张量的最后四轴，U 的颜色矩阵维度为 3x3
```

在进入 kernel 前，至少验证

$$
 U_\mu(x)\in SU(3),
 \qquad
 \max_x\left|\det U_\mu(x)-1\right|\ll 1,
 \qquad
 \|U_\mu^\dagger U_\mu-I\|\ll 1.
$$

生成的 gauge 若不是严格 SU(3)，应记录投影/重unitarization 步骤；否则不同后端的差异可能来自输入场本身。

### A.2 算子和 parity 准备

```text
1. 按边界条件构造 forward/backward neighbor index
2. 应用 Wilson hopping；若有 Clover，批量生成 onsite block A_p=I+C_p
3. 按 p(x)=(x+y+z+t) mod 2 分离 e/o parity
4. 选择 full、asymmetric Schur 或 symmetric Schur 语义
5. 对 MATPC rhs 做 prepare，并保存被消去 parity 所需的中间量
6. 用随机向量检查 D、D^dagger、gamma5-D-gamma5 关系
```

对随机测试向量 $v$，算子等价性可用相对误差

$$
 \epsilon_{\mathrm{op}}(v)=
 \frac{\|A_1v-A_2v\|_2}
 {\max(\|A_1v\|_2,\|A_2v\|_2,\epsilon_{\mathrm{floor}})}.
$$

这里 $A_1,A_2$ 可以是 Python dslash 与 QCU dslash，也可以是 full operator 与 Schur 重建后的等价作用。必须在多个随机向量、多个边界和至少两个数据类型上测试。

### A.3 Null vector 和 coarse setup

```text
1. 选用随机、低模、CG/FGMRES 历史或 deflation 向量作为候选 B_l
2. 按 aggregate 切分 B_l，逐 aggregate 做 CGS/MGS/QR
3. 丢弃低于 rank threshold 的局部方向，记录实际 nvec
4. 形成 P_l；若采用 adjoint restriction，令 R_l=P_l^dagger
5. 验证 ||R_l P_l-I|| 与局部正交误差
6. 用 batched 或 colored probes 计算 D_{l+1}=R_l A_l P_l
7. 提取 X_l、Y_l^+、Y_l^-；发现宽 support 时停止 strict 构造
8. 批量构造 X_l^{-1}，再生成 Yhat_l
9. 持久化 hierarchy 元数据和所有必要资产
```

局部 rank threshold 必须与 dtype 和 aggregate 大小绑定。若阈值太小，近线性相关向量会放大 coarse inverse；若阈值太大，粗空间会丢失物理低模。建议同时报告保留奇异值、局部 condition estimate 和实际 coarse dof。

### A.4 一次外层 FGMRES + MG 右预条件

```text
x = x0
r = b - D(x)
for restart_cycle = 0, 1, ...:
    beta = norm(r)
    v[0] = r / beta
    for j = 0, ..., m-1:
        z[j] = Vcycle(v[j])       # 右预条件，Vcycle 可变化
        w = D(z[j])
        Arnoldi-MGS(w, v[0:j+1], H)
        apply Givens to H and small rhs g
        if estimated residual small: break
    y = solve_small_upper_hessenberg(H, g)
    x = x + sum_j y[j] z[j]
    r = b - D(x)                   # reliable full residual
    record cycle, inner steps, ||r||, setup state
    if ||r|| <= atol*||b||: return x
```

若目标是 fine MATPC，外层向量是 compact parity 还是 full vector 必须在流程头部固定；`Vcycle` 返回的 correction、`D` 的 matvec 和 true residual 不能跨两种语义混用。若 warm start 打开，第一轮应记录 $\|b-Dx_0\|$，不能默认它等于 $\|b\|$。

### A.5 关闭和资源回收

```text
1. 完成最后一次 true residual 与 solver status 记录
2. 停止新的 kernel 和 MPI collective
3. 释放 coarse assets、null vectors、scratch 和 Cython set_ptrs
4. 关闭每个线程的 CUDA stream/context
5. 独立关闭 HDF5 File 句柄
6. 检查 live allocation、文件句柄和 _SET_INDEX_ 状态
```

资源关闭顺序不能依赖 Python 的循环引用回收。特别是多线程场景，应在每个线程的 worker 内显式关闭 hierarchy，主线程只负责汇总结果。

## 附录 B：残差、误差与等价性的审计公式

### B.1 三种常用误差

给定近似解 $x$、右端 $b$ 和算子 $A$，分别记录

$$
 r=b-Ax,
 \qquad
 \rho_{\mathrm{abs}}=\|r\|_2,
 \qquad
 \rho_{\mathrm{rel}}=\frac{\|r\|_2}{\|b\|_2}.
$$

若 $\|A\|$ 可估计，还可以记录 backward error

$$
 \eta(x)=\frac{\|b-Ax\|_2}
 {\|A\|_2\|x\|_2+\|b\|_2}.
$$

`tol` 的具体含义必须在报告中绑定到上述某一个量；“残差达到 $10^{-8}$”如果没有说明 absolute/relative、递推/true 和 full/Schur，不能复现。

### B.2 Full 与 compact 的等价性

令 $E$ 为 prepare 映射、$J$ 为 reconstruct 映射，compact operator 为 $S$。理想情况下

$$
 S=E D J,
 \qquad
 D J y = \widetilde J S y
$$

其中 $\widetilde J$ 还包含被消去 parity 的恢复。数值测试可以使用

$$
 \epsilon_{\mathrm{Schur}}=
 \frac{\|E(DJ y)-S y\|_2}{\max(\|S y\|_2,\epsilon_{\mathrm{floor}})}
$$

以及 reconstruct 后的 full residual。只测 $S y$ 而不测 $D x$ 不能证明 compact 路径正确。

### B.3 Galerkin 一致性

对 fine 向量 $v_c$，以 full coarse operator 和显式 fine 应用分别计算

$$
 y_1=(R A_f P)v_c,
 \qquad
 y_2=A_c v_c.
$$

定义

$$
 \epsilon_{\mathrm{Galerkin}}=
 \frac{\|y_1-y_2\|_2}{\max(\|y_1\|_2,\|y_2\|_2,\epsilon_{\mathrm{floor}})}.
$$

若 $A_c$ 被截断为最近邻 `X/Y`，还要把截断误差单独记为

$$
 \epsilon_{\mathrm{support}}=
 \frac{\|(R A_f P-A_c)v_c\|_2}{\max(\|R A_f P v_c\|_2,\epsilon_{\mathrm{floor}})}.
$$

strict 构造只有在 support 检查通过时才允许把 $\epsilon_{\mathrm{support}}$ 解释为数值误差；宽 stencil 被静默丢弃时，它其实是模型误差。

### B.4 预条件器质量

对 residual $r$ 和 correction $z=M^{-1}r$，可记录

$$
 q_M=\frac{\|r-Az\|_2}{\|r\|_2}.
$$

$q_M$ 越小通常表示单次预条件质量越高，但过度精确的 coarse solve 可能使单次代价超过其收益。外层 FGMRES 的实际评价应同时记录 $q_M$、每次 V-cycle 成本和 full residual 下降。

## 附录 C：算法、物理对象与源码矩阵

| 对象 | 数学角色 | 关键源码 | 状态 |
|---|---|---|---|
| Wilson dslash | 最近邻非 Hermitian operator | `pyqcu/dslash/_wilson.py`, `cpp/cuda/qcu/include/wilson_dslash.h` | 实现 |
| Clover block | onsite spin-color matrix | `pyqcu/dslash/_clover.py` | 实现 |
| Schur prepare/reconstruct | parity elimination | `cpp/cuda/qcu/src/apply_multigrid_strict.cu` 相关入口 | 实现 |
| BiCGStab | 非 Hermitian Krylov | `pyqcu/solver/_bistabcg.py` | 实现 |
| FGMRES | 变化右预条件 | `pyqcu/solver/_gmres.py` | 实现 |
| MR | short smoother | `pyqcu/solver/_mr.py` | 实现 |
| CA-CG | block Krylov | `pyqcu/solver/_cacg.py` | 实现 |
| Lanczos | Hermitian eigenspace | `pyqcu/solver/_lanczos.py` | 实现 |
| Multi-shift CG | shifted SPD systems | `pyqcu/solver/_multishift_cg.py` | Python 实现 |
| Native MG | Python V-cycle | `pyqcu/solver/_multigrid.py` | 实现 |
| Legacy MG | compact/33-tensor | `pyqcu/solver/_quda_multigrid.py` | 实现/兼容 |
| Strict Galerkin | full $R A P$ | `pyqcu/tools/_strict_galerkin.py` | 原型/实现 |
| Strict CUDA solver | coarse, MATPC, FGMRES | `cpp/cuda/qcu/src/apply_multigrid_strict.cu` | 实现 |
| QUDA coarse op | reference coarse action | `refer/git-rep/quda/lib/coarse_op.cuh` | 参考 |
| Twisted mass | flavor-spin onsite term | `refer/git-rep/quda/lib/dirac_twisted_mass.cpp` | 参考 |
| Staggered/HISQ | one-component/wide stencil | `refer/git-rep/quda/lib/dirac_improved_staggered_kd.cpp` | 参考 |
| Domain-wall/Möbius | five-dimensional operator | `refer/git-rep/quda/lib/dirac_domain_wall_4d.cpp`, `dirac_mobius.cpp` | 参考 |
| Overlap | sign-function operator | 理论参考，当前未发现 PyQCU 生产入口 | 未验证 |

该表只说明“在哪里能找到定义或入口”，不替代实际运行验证。新增代码后应把函数名、参数语义、支持的 dtype/device 和 failure gate 一并更新。

## 附录 D：基准报告模板

```markdown
## Case: <short name>

- commit/tag:
- hardware / driver / CUDA:
- MPI ranks / threads / GPU binding:
- lattice and boundary:
- action and parameters:
- gamma basis / color convention:
- input layout and dtype:
- rhs source and seed:
- hierarchy (levels, block, nvec, transfer, coarse operator):
- outer / inner solver and restart:
- atol / rtol / true-residual definition:

### Static checks

- [ ] operator layout and shape
- [ ] gamma5-Hermiticity or declared non-Hermitian path
- [ ] parity round-trip
- [ ] R P and local orthogonality
- [ ] strict support / coarse asset schema
- [ ] params/argv/set_ptrs index agreement

### Dynamic results

| metric | value |
|---|---:|
| setup time | |
| solve time | |
| outer iterations | |
| operator applications | |
| final full true residual | |
| max live memory | |
| exit status | |

结论只根据上表给出的 full true residual 和退出状态判断；没有运行数据的项目保留“未验证”。
```

## 附录 E：谱分析、通信模型与验收阈值

### E.1 自由场谱的快速检查

在 $U_\mu=I$、周期边界且 $c_{\mathrm{SW}}=0$ 时，Wilson 算子在动量 $p$ 上的矩阵可以写成

$$
 D_W(p)=M(p)I+i\sum_\mu\gamma_\mu\sin p_\mu,
 \qquad
 M(p)=m_0+\sum_\mu(1-\cos p_\mu).
$$

若 $\gamma_\mu$ 满足 $\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}$，则

$$
 D_W^\dagger(p)D_W(p)=
 \left[M(p)^2+\sum_\mu\sin^2p_\mu\right]I.
$$

这个结果说明自由场 normal operator 的每个旋量分量有相同的特征值。数值实现可以选取所有 $p_\mu=0$、一个分量为 $\pi/2$、以及一个分量为 $\pi$ 的模式，分别检查质量项、动量项和 Wilson doubler 项。若这些模式的相对误差已超过数据类型允许的量级，应先修复 gamma、邻居或边界，不要继续调整 solver 容差。

反周期时间边界可以视为时间方向在跨边界 halo 时附加负号。对 $L_t$ 为偶数的格点，最低时间动量从 $2\pi n/L_t$ 改为 $2\pi(n+1/2)/L_t$；因此同一个 gauge 和质量下，周期/反周期测试的谱并不相同，比较时不能混淆。

### E.2 Krylov 收敛的多项式视角

对固定预条件器 $M^{-1}$，Krylov 方法的误差可以抽象为

$$
 e_k=p_k(M^{-1}A)e_0,
 \qquad p_k(0)=1.
$$

CG 在 SPD 情形选择对谱区间最有利的多项式，经典上界为

$$
 \frac{\|e_k\|_A}{\|e_0\|_A}
 \leq
 2\left(\frac{\sqrt{\kappa(A)}-1}
 {\sqrt{\kappa(A)}+1}\right)^k.
$$

这个上界不是任意格点算子的实测预测：非正规、非 Hermitian、有限精度和变化预条件都会破坏其直接适用性。它的用途是解释为什么改善低端谱、降低 coarse condition number 或使用 deflation 能减少迭代，而不是从迭代数反推出精确条件数。

BiCGStab 的多项式包含双线性 shadow 约束和局部最小残差因子 $1-\omega_k z$，因此可能在某些谱分布上振荡。FGMRES 则在扩展空间上最小化实际残差，代价是保存 $V_m$、$Z_m$ 和 Hessenberg 系统。报告中应同时说明“每步用了几次 operator application”和“每个 restart 保存多少向量”，否则不同 solver 的迭代数没有直接可比性。

### E.3 粗空间对低模的覆盖

设 fine operator 的低模为 $\{u_i\}$，粗空间投影为 $\Pi=P(RP)^{-1}R$。可以用近似覆盖误差

$$
 \epsilon_{\mathrm{low}}=
 \max_{i\in\mathcal I}
 \frac{\|(I-\Pi)u_i\|_2}{\|u_i\|_2}
$$

衡量 null vectors 是否覆盖目标低模。实际计算中可以用 Lanczos 或短 FGMRES 产生的 Ritz vectors 近似 $u_i$。如果 $\epsilon_{\mathrm{low}}$ 很大，增加 smoother 次数通常只能暂时缓解，真正的修复是增加、更新或重新正交化 null vectors。

局部 aggregate 的边界也会影响覆盖：过小 block 增加 coarse lattice 体积和通信，过大 block 则使局部基更难保持正交。选择 block 时应同时扫描

$$
 (n_v,\; \text{block volume},\; E=n_vN_s^{\mathrm{coarse}},\;
 \epsilon_{\mathrm{orth}},\; \epsilon_{\mathrm{low}}).
$$

不能只用外层迭代数选 block，因为 setup 成本和粗层显存可能在求解阶段之外主导总时间。

### E.4 通信和显存的量级模型

对四维局部 lattice $L_x\times L_y\times L_z\times L_t$，最近邻 dslash 每个方向有两个面。若每个面传递的复元素数为 $n_{\mathrm{face}}$，一次 halo 交换的元素量级为

$$
 N_{\mathrm{halo}}
 \approx 2n_{\mathrm{face}}
 \left(L_yL_zL_t+L_xL_zL_t+L_xL_yL_t+L_xL_yL_z\right).
$$

spin projection 会把每个面的旋量分量从四个减少到两个，但 Clover onsite 仍需完整 spin-color block。粗层如果把每个方向的 $Y$ 存成 dense $E\times E$ block，则通信字节数与 $E^2$ 成正比；这解释了 coarse level 过大时带宽和 MPI latency 会迅速成为瓶颈。

一个实用的峰值显存估算是

$$
 M_{\mathrm{peak}}
 \simeq M_{\mathrm{fine\ fields}}
 +M_{\mathrm{null}}
 +M_{\mathrm{coarse\ assets}}
 +M_{\mathrm{workspace}}
 +M_{\mathrm{allocator\ cache}}.
$$

其中最后一项不能当作 live tensor。报告显存时应至少给出 `allocated`、`reserved` 或后端等价指标，并说明测量发生在 setup、solve 还是 close 之后。

### E.5 action-specific support 检查

| action | fine support | 允许直接映射到 strict 最近邻 `X/Y` 吗 | 需要的额外处理 |
|---|---|---|---|
| Wilson | $pm\hat\mu$ 最近邻 | 可以 | 检查 projector 和 link dagger |
| Clover | Wilson + onsite | 可以 | 将 Clover 全部并入 $X$ |
| Twisted mass | Wilson + onsite flavor block | 形式上可以 | coarse $X$ 扩展 flavor，自身尚无 PyQCU 生产入口 |
| Staggered | $pm\hat\mu$，带 $\eta_\mu$ | 需单独 spin/taste 语义 | 不能复用 Wilson spin layout |
| asqtad/HISQ | 多路径、Naik 等宽 stencil | 通常不可以 | 采用宽 stencil coarse operator |
| Domain-wall | 四维最近邻 + 第五维邻居 | 仅在扩展几何后可行 | 把 $s$ 维纳入 layout 和 halo |
| Möbius | 五维重参数化 | 同上 | 记录第五维系数和边界耦合 |
| Overlap | sign function 的非局部近似 | 不可以直接套用 | 外层/内层嵌套求解或 rational coarse |

“最近邻”是矩阵 support 的陈述，不是“每次 kernel 只访问一个邻居”的实现细节。只要 action-specific 路径经多次 hopping 形成了远端耦合，粗算子就必须保留或明确截断这些项。

### E.6 推荐的验收阈值写法

阈值应绑定数据类型、格点规模和测量量。可以采用以下模板，而不是给出脱离上下文的单一数字：

| 检查量 | 建议记录 | 说明 |
|---|---|---|
| operator equivalence | `epsilon_op` + dtype | Python/QCU/参考路径逐向量比较 |
| gamma5-Hermiticity | `epsilon_g5` | 对 Wilson/Clover 的结构性检查 |
| parity round-trip | `epsilon_parity` | full 与 compact 互转 |
| local orthogonality | `epsilon_orth` | 每 aggregate 的最大值 |
| Galerkin | `epsilon_Galerkin` | batched/逐列或 full/coarse 一致性 |
| reliable update | refresh 前后 residual ratio | 判断递推漂移 |
| final solve | full `rho_abs` 和 `rho_rel` | 与停止条件同时保存 |
| memory | setup/solve/close 三个时刻 | 区分 live 与 allocator cache |

例如，在 complex64 运行中可以把“达到约 $10^{-6}$ 的相对算子误差”作为初步筛查，把更严格的阈值留给 complex128；具体数值仍需由格点体积、归约顺序和 kernel 实现校准。文档应写出“采用的阈值”和“阈值来源”，不能只写“误差足够小”。

### E.7 从源码锚点到运行证据

一份可审计的结论至少包含四段：

1. **定义**：给出矩阵、残差或 support 的数学式；
2. **入口**：指出对应 Python/Cython/C++/QUDA 文件和函数；
3. **断言**：写出输入、输出、dtype、parity、边界和失败条件；
4. **证据**：给出命令、退出码、数值和环境元数据。

例如，“strict coarse 支持最近邻”只能由 `_strict_galerkin.py` 的 support 检查和 C++ kernel 的方向索引共同证明；要声称“strict coarse 在四卡上加速”，还必须补充设备、MPI、setup/solve 时间和 full true residual。静态源码证据和动态性能证据不能互相替代。

### E.8 结果表中必须区分的时间和计数

把一次运行压缩成一个总时间会丢失最重要的诊断信息。建议至少拆分以下计数：

| 计数 | 含义 | 解释方式 |
|---|---|---|
| `setup_matvecs` | 生成 null vectors、Galerkin probes 和 coarse assets 所用的 fine operator 次数 | setup 重、solve 轻时应单独摊销 |
| `fine_matvecs` | 外层和 fine smoother 的算子应用 | 与通信/带宽强相关 |
| `coarse_matvecs` | 各层 coarse operator 应用 | 反映 hierarchy 是否真正被使用 |
| `global_dots` | MPI 全局归约或等价同步次数 | 常是强缩放的限制因素 |
| `reliable_refreshes` | full true residual 刷新次数 | 反映混合精度维护成本 |
| `restarts` | GMRES/FGMRES restart 次数 | 不能与外层总迭代直接等同 |
| `peak_live_bytes` | hierarchy 和 workspace 的 live 显存 | 应在 close 前后分别采样 |

对变化的预条件器，外层每一轮可能有不同 coarse solve 精度；因此建议保存每个 outer cycle 的 `q_M`、内层迭代数和 true residual，而不是只保存最终一行。对 MPI 运行，还应保存每次 global reduction 的平均或最大耗时，以区分 kernel 变慢和同步变慢。

### E.9 最小可接受结论

一条“实现正确”的最小结论应同时满足：

1. 算子级布局、边界和对称性检查通过；
2. full/compact 或 full/coarse 的映射误差在声明的 dtype 阈值内；
3. 求解器以 full true residual 达到明确的 absolute/relative criterion；
4. 所有 failure gate（breakdown、非有限数、strict support、MPI rank）都有明确状态；
5. 运行记录包含足够元数据，另一位开发者可以在同一仓库复现。

如果只满足第 3 条而不满足前两条，可能是错误算子上的“假收敛”；如果只满足静态源码检查而没有第 3 条，只能称为“代码路径存在”。这两个结论在报告中必须分开。
