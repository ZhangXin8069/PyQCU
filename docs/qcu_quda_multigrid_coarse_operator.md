# QCU 与 QUDA MultiGrid 粗层 Dslash 表示

本文回答一个容易混淆的问题：QCU 是否像 QUDA 一样，在 MultiGrid 粗层构造
“粗化 gauge 与 clover”，再由它们形成粗层 dslash。

结论需要按实现路径区分。QCU legacy MG 有完整的粗层 dslash，但直接保存
Schur/Galerkin 粗算子的 33-tensor block；QCU Strict-MG 才使用与 QUDA
`DiracCoarse` 对应的 `X/Y/Yhat` 数据组织。Strict/QUDA 中名为 `X/Y` 的量都是
有效粗算子系数，不是把原始细层 gauge link 与 clover 项做简单分块平均。

## 1. 结论

| 路径 | 是否有粗层 dslash | 粗层数据表示 | 是否显式使用 `X/Y/Yhat` |
|---|---|---|---|
| QCU legacy MG | 有 | compact odd coarse lattice 上的 33-tensor `A_c` | 否 |
| QCU Strict-MG | 有 | full-coarse lattice 上的 onsite 与正反向 link | 是 |
| QUDA `DiracCoarse` | 有 | full-coarse lattice 上的 coarse clover/link | 是 |

因此：

- “QCU 有没有粗层 dslash？”有。
- “QCU legacy 是否像 QUDA 一样拆成 `X/Y/Yhat`？”没有，legacy 直接保存
  33-tensor 粗算子。
- “QCU Strict 是否像 QUDA？”是，`X`、`Y`、`X^{-1}` 与 `Yhat` 的算法角色
  基本对应。
- “QUDA/Strict 是否直接粗化物理 gauge 与 clover？”不是。二者都使用由
  transfer operator 与 fine Dirac operator 得到的 Galerkin 有效量。

## 2. Legacy QCU：有效粗 Dslash，不是粗 Gauge/Clover 对

### 2.1 从 Fine Schur 补到粗算子

对 Wilson-Clover 算子按奇偶格点分块：

$$
D=\begin{pmatrix}
D_{ee} & -\kappa H_{eo}\\
-\kappa H_{oe} & D_{oo}
\end{pmatrix}.
$$

QCU legacy MG 先消去 even 分量，在 odd 子格上求解 Schur 补：

$$
S_o=D_{oo}-\kappa^2H_{oe}D_{ee}^{-1}H_{eo}.
$$

粗层通过 prolongation $P$ 与 restriction $R$ 做 Galerkin 投影：

$$
A_c=R S_o P.
$$

当前 C++ 注释使用 $A_c=P^T S_oP$ 的记号；本文在复内积语境中使用
$R=P^H$。引用源码时保留其原始记号。

### 2.2 为什么需要 33 个 Block

$S_o$ 中的 $H_{oe}D_{ee}^{-1}H_{eo}$ 是 odd 格点到 even 格点再返回 odd
格点的两跳过程。经过二倍粗化后，两跳位移映射为：

- $x_c\pm2\hat\mu$：粗格最近邻；
- $x_c\pm\hat\mu\pm\hat\nu$：粗格同平面内的对角连接；
- 两跳返回原 odd 格点：粗格 onsite 项。

因此 $A_c$ 的完整 Wilson-Clover 支撑由以下 block 组成：

| 数据类型 | 形状 | 位移数量 |
|---|---|---:|
| `sit_packed` | `[E_c, E_c, Xc, Yc, Zc, Tc]` | 1 |
| `hop_nn` | `[2, 4, E_c, E_c, Xc, Yc, Zc, Tc]` | 8 |
| `hop_diag` | `[2, 2, 6, E_c, E_c, Xc, Yc, Zc, Tc]` | 24 |

这里每个元素都是粗自由度之间的 $E_c\times E_c$ 矩阵，不是 $3\times3$
物理 gauge link，也不是细层 $12\times12$ clover 项的简单缩小版。

### 2.3 Legacy 粗层作用

legacy `coarse_dslash_op()` 直接读取 `sit_packed`、`hop_nn` 与
`hop_diag`。它可以概念性写成：

$$
\begin{aligned}
(A_c\psi)(x_c)={}&\sum_e S_{je}(x_c)\psi_e(x_c)\\
&+\sum_{\mu,\sigma=\pm}H_{je,\mu}^{\sigma}(x_c)
\psi_e(x_c+\sigma\hat\mu)\\
&+\sum_{\mu<\nu}\sum_{\sigma,\tau=\pm}
D_{je,\mu\nu}^{\sigma\tau}(x_c)
\psi_e(x_c+\sigma\hat\mu+\tau\hat\nu).
\end{aligned}
$$

源码中的粗位移定义、`coarse_dslash_op()` 入口和实际 kernel 参数见
[`lattice_clover_multigrid.h`](../cpp/cuda/qcu/include/lattice_clover_multigrid.h#L680-L685)
与
[`lattice_clover_multigrid.h`](../cpp/cuda/qcu/include/lattice_clover_multigrid.h#L885-L930)。
Python 侧 33-tensor 构建与参考作用见
[`_multigrid.py`](../pyqcu/torch/tools/_multigrid.py#L769-L883)。

## 3. QCU Strict-MG：QUDA 风格的 Full-Coarse 层级

Strict-MG 保留 full coarse lattice，并把当前粗层 Dirac 算子拆成 onsite 项
$X_l$ 与 hopping 项 $H_l$：

$$
D_l=X_l+H_l.
$$

构造下一层时先做左预条件，再做 Galerkin 投影：

$$
\widehat D_l=X_l^{-1}D_l,
\qquad
D_{l+1}=R_l\widehat D_lP_l.
$$

预条件 link 定义为：

$$
\widehat Y_l=X_l^{-1}Y_l.
$$

Strict 路径显式保存：

- `X`：粗层 onsite 或 coarse-clover 角色；
- `X_inv`：$X^{-1}$；
- `Y_forward/Y_backward`：粗层正反向 link；
- `Yhat`：预条件后的粗 link。

运行期通常只需要 `Yhat` 与 `(X, X^{-1})`。raw `Y` 可用于 setup 与诊断，
不是普通求解的必需常驻资产。相关定义和资产导出见
[`_multigrid_core.py`](../pyqcu/torch/solver/_multigrid_core.py#L1071-L1108)
与
[`_multigrid_core.py`](../pyqcu/torch/solver/_multigrid_core.py#L1496-L1512)。
C++ 运行期对粗层 link 使用 full-site 存储，并允许省略 raw `Y`，见
[`apply_multigrid_strict.cu`](../cpp/cuda/qcu/src/apply_multigrid_strict.cu#L2733-L2739)
与
[`apply_multigrid_strict.cu`](../cpp/cuda/qcu/src/apply_multigrid_strict.cu#L3467-L3474)。

因此 Strict-MG 在数据表示和算法角色上已经与 QUDA `DiracCoarse` 对齐，但它
仍不是“把细层物理 gauge 和 clover 直接分块”的粗离散化。

## 4. QUDA DiracCoarse 的 X/Y 也是有效量

QUDA `DiracCoarse` 保存：

- `Y_h/Y_d`：coarse link field；
- `X_h/X_d`：coarse clover term；
- `Xinv_h/Xinv_d`：coarse clover inverse；
- `Yhat_h/Yhat_d`：preconditioned coarse link field。

成员定义见历史 QUDA 安装树中的
`data/quda-sm70-install/include/dirac_quda.h`（约第 1877--1934 行）；该源码树未置于
当前工作树。

这些字段使用了 QUDA 的 `GaugeField` 容器和“coarse link/coarse clover”命名，
但其数学来源仍是 transfer operator 下的 Galerkin 粗化。$X$ 起 onsite 作用，
$Y$ 起 hopping 作用，不应理解成原始 $SU(3)$ gauge link 与原始 clover 矩阵的
逐块平均。

## 5. 两种粗层表示为何不同

legacy QCU 与 QUDA/Strict 的核心目标相同，都是避免每次粗层 matvec 回到细层，
区别在于先做何种代数重组：

| 维度 | QCU legacy | QCU Strict / QUDA |
|---|---|---|
| 粗格几何 | compact odd coarse lattice | full coarse lattice |
| 构造顺序 | fine Schur 消元后再 Galerkin | full fine operator 左预条件后 Galerkin |
| 主要对象 | 直接保存 $A_c$ | 保存 $D_c=X+H$ 的因子 |
| 邻接 | onsite + 8 nearest + 24 diagonal | onsite + 8 one-hop links |
| 经典应用 | 宽 stencil 粗 dslash | `DiracCoarse`/MATPC 与递归 MG |
| 粗层 parity | 粗层本身为 compact odd | 只在 fine R/P/MATPC 边界使用 parity |

33-tensor 表示直接编码 Schur 补产生的宽支撑，不需要再把 $A_c$ 强行分解成
一组 coarse gauge 和 coarse clover。Strict/QUDA 则利用 full-coarse 的一跳
结构，把粗算子组织成更适合统一 `DiracCoarse` 接口和递归 MG 的 `X/Y/Yhat`。

两种 cache、slot 顺序、几何和 C++ kernel 不兼容，不能仅因都叫“粗层算子”
就互换或在同名槽位间重排。

## 6. 一句话判据

- 问“有没有粗层有效 dslash”：legacy QCU、Strict QCU 与 QUDA 都有。
- 问“是否显式使用 coarse link 与 coarse clover 命名”：Strict QCU 与 QUDA 有，
  legacy QCU 没有。
- 问“是否直接粗化原始物理 gauge 与 clover”：三者都不是。

## 7. 调查边界

本文结论来自当前工作区源码静态核查，覆盖：

- legacy CUDA `applyCloverMultigridQcu`；
- CUDA Strict `applyMultigridStrictFgmresQcu` 的层次资产；
- 本地 QUDA 1.1.0-sm70 `DiracCoarse` 的接口定义。

本文不评价三条路径的求解性能、cache 命中选择或数值收敛优劣。相关算法组成、
子算法和 benchmark 分别见
[`qcu_quda_multigrid_algorithms.md`](qcu_quda_multigrid_algorithms.md)、
[`qcu_quda_multigrid_subalgorithms.md`](qcu_quda_multigrid_subalgorithms.md)
与
[`three_way_benchmark_20261022.md`](three_way_benchmark_20261022.md)。
